"""Persistence: schema, resume semantics, cost accounting, and never losing a segment."""

from __future__ import annotations

from pathlib import Path

import pytest

from folioai.errors import StoreError
from folioai.store import SCHEMA_VERSION, JobStore, SegmentRecord


def make_segments(n: int, *, chapter: str = "ch01") -> list[SegmentRecord]:
    return [
        SegmentRecord(
            segment_id=f"b{i:04d}",
            chapter_id=chapter,
            ordinal=i,
            kind="paragraph",
            source_text=f"Source paragraph {i}.",
            final_text=None,
            final_score=None,
            status="pending",
            needs_review=False,
            attempts_count=0,
        )
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    with JobStore(tmp_path / "job.db") as s:
        s.create_job(
            job_id="job1",
            source_path=tmp_path / "book.pdf",
            source_sha256="deadbeef",
            config={"target_lang": "de"},
            source_lang="en",
            target_lang="de",
        )
        yield s


def test_wal_mode_and_foreign_keys_are_on(store: JobStore) -> None:
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_schema_version_is_recorded(store: JobStore) -> None:
    row = store.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    assert row["v"] == SCHEMA_VERSION


def test_future_schema_version_refuses_to_open(tmp_path: Path) -> None:
    path = tmp_path / "job.db"
    with JobStore(path) as s:
        s.conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION + 5, "2030-01-01T00:00:00+00:00"),
        )
    with pytest.raises(StoreError) as excinfo:
        JobStore(path)
    assert "newer folioai" in str(excinfo.value)


def test_create_job_is_idempotent(store: JobStore, tmp_path: Path) -> None:
    again = store.create_job(
        job_id="job1",
        source_path=tmp_path / "book.pdf",
        source_sha256="deadbeef",
        config={"target_lang": "fr"},
    )
    assert again.target_lang == "de"  # existing row wins; resume must not clobber
    assert len(store.list_jobs()) == 1


def test_segments_roundtrip_and_total_is_maintained(store: JobStore) -> None:
    store.upsert_segments("job1", make_segments(5))
    job = store.get_job("job1")
    assert job is not None
    assert job.total_segments == 5
    assert [s.segment_id for s in store.list_segments("job1")] == [
        "b0000",
        "b0001",
        "b0002",
        "b0003",
        "b0004",
    ]


def test_reextraction_does_not_wipe_completed_work(store: JobStore) -> None:
    """Re-running extraction on a partially translated job must preserve translations."""
    store.upsert_segments("job1", make_segments(3))
    store.finalize_segment(
        "job1", "b0001", final_text="Übersetzt.", final_score=91.0, needs_review=False
    )

    store.upsert_segments("job1", make_segments(3))

    kept = store.get_segment("job1", "b0001")
    assert kept is not None
    assert kept.final_text == "Übersetzt."
    assert kept.status == "done"


def test_pending_includes_failed_and_in_flight_segments(store: JobStore) -> None:
    """A process killed mid-batch leaves rows in 'translating'; resume must reclaim them."""
    store.upsert_segments("job1", make_segments(6))
    store.set_segment_status("job1", ["b0000"], "translating")
    store.set_segment_status("job1", ["b0001"], "evaluating")
    store.set_segment_status("job1", ["b0002"], "failed")
    store.finalize_segment("job1", "b0003", final_text="ok", final_score=90.0, needs_review=False)

    pending = {s.segment_id for s in store.pending_segments("job1")}
    assert pending == {"b0000", "b0001", "b0002", "b0004", "b0005"}


def test_no_segment_is_ever_lost_across_a_simulated_kill(tmp_path: Path) -> None:
    """§21.3: kill mid-run, resume, and end with every segment accounted for exactly once."""
    path = tmp_path / "job.db"
    with JobStore(path) as first:
        first.create_job(
            job_id="job1",
            source_path=tmp_path / "book.pdf",
            source_sha256="abc",
            config={},
        )
        first.upsert_segments("job1", make_segments(10))
        for sid in ("b0000", "b0001", "b0002"):
            first.finalize_segment(
                "job1", sid, final_text=f"done {sid}", final_score=88.0, needs_review=False
            )
        first.set_segment_status("job1", ["b0003", "b0004"], "translating")
        # process dies here: no clean shutdown, no further writes

    with JobStore(path) as resumed:
        pending = resumed.pending_segments("job1")
        assert len(pending) == 7
        for seg in pending:
            resumed.finalize_segment(
                "job1", seg.segment_id, final_text="late", final_score=85.0, needs_review=False
            )
        done = resumed.list_segments("job1", status="done")
        assert len(done) == 10
        assert len({s.segment_id for s in done}) == 10  # no duplicates
        assert all(s.final_text for s in done)  # no gaps


def test_attempts_accumulate_and_never_overwrite(store: JobStore) -> None:
    store.upsert_segments("job1", make_segments(1))
    for attempt_no, model in enumerate(("translator", "translator", "escalation"), start=1):
        store.record_attempt(
            job_id="job1",
            segment_id="b0000",
            attempt_no=attempt_no,
            model=model,
            params={"temperature": 0.2},
            output_text=f"attempt {attempt_no}",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
        )
    rows = store.list_attempts("job1", "b0000")
    assert [r["attempt_no"] for r in rows] == [1, 2, 3]
    assert [r["model"] for r in rows] == ["translator", "translator", "escalation"]
    seg = store.get_segment("job1", "b0000")
    assert seg is not None and seg.attempts_count == 3


def test_evaluation_links_to_its_attempt(store: JobStore) -> None:
    store.upsert_segments("job1", make_segments(1))
    attempt_id = store.record_attempt(
        job_id="job1",
        segment_id="b0000",
        attempt_no=1,
        model="m",
        params={},
        output_text="x",
    )
    eval_id = store.record_evaluation(
        attempt_id=attempt_id,
        evaluator_model="judge",
        scores={"completeness": 95},
        issues=[],
        composite=93.5,
        passed=True,
    )
    row = store.conn.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
    assert row["attempt_id"] == attempt_id
    assert row["passed"] == 1


def test_usage_rolls_up_into_job_cost(store: JobStore) -> None:
    for _ in range(3):
        store.record_usage(
            job_id="job1",
            model="openai/gpt-4.1",
            endpoint="chat.completions",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_usd=0.02,
        )
    job = store.get_job("job1")
    assert job is not None
    assert job.cost_usd == pytest.approx(0.06)
    assert store.total_cost("job1") == pytest.approx(0.06)


def test_update_job_rejects_unknown_columns(store: JobStore) -> None:
    with pytest.raises(StoreError):
        store.update_job("job1", not_a_column="x")


def test_glossary_upsert_overwrites_by_source_term(store: JobStore) -> None:
    store.upsert_glossary(
        "job1", [{"source": "the Warden", "target": "der Wärter", "occurrences": 4}]
    )
    store.upsert_glossary(
        "job1",
        [{"source": "the Warden", "target": "der Aufseher", "locked": True, "occurrences": 214}],
    )
    terms = store.list_glossary("job1")
    assert len(terms) == 1
    assert terms[0]["target"] == "der Aufseher"
    assert terms[0]["locked"] is True


def test_transaction_rolls_back_on_error(store: JobStore) -> None:
    store.upsert_segments("job1", make_segments(2))
    with pytest.raises(RuntimeError), store.transaction() as conn:
        conn.execute("UPDATE segments SET final_text = 'x' WHERE job_id = 'job1'")
        raise RuntimeError("boom")
    assert all(s.final_text is None for s in store.list_segments("job1"))


def test_deleting_a_job_cascades(store: JobStore) -> None:
    store.upsert_segments("job1", make_segments(2))
    store.record_attempt(
        job_id="job1", segment_id="b0000", attempt_no=1, model="m", params={}, output_text="x"
    )
    store.delete_job("job1")
    assert store.conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"] == 0
    assert store.conn.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()["n"] == 0


def test_opening_a_missing_database_without_create_explains_itself(tmp_path: Path) -> None:
    with pytest.raises(StoreError) as excinfo:
        JobStore(tmp_path / "nope.db", create=False)
    assert "folioai jobs list" in (excinfo.value.remedy or "")
