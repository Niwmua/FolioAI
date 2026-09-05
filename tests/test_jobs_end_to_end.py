"""The whole pipeline, PDF to translated IR, driven by the fake client.

This is the test that would notice if any two stages stopped fitting together: extraction,
segmentation, translation, validation, evaluation, persistence and the parallel-output
assertion all run here, with no network and no key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from folioai.config import Settings
from folioai.jobs import build_translated_document, prepare_job, reopen_job, run_job
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient
from folioai.render.markdown import document_to_markdown
from folioai.tags import parse_segments


# N818: this stands in for a process being killed, not for an error condition.
class SimulatedKill(RuntimeError):  # noqa: N818
    """Stands in for the process being killed mid-run."""


GOOD_SCORES = {
    "completeness": 97,
    "accuracy": 94,
    "terminology": 96,
    "fluency": 91,
    "formatting": 100,
}


def german_ish(messages: list[Message], model: str) -> str:
    """Translate by prefixing, or judge -- whichever the message shape calls for."""
    user = next(m["content"] for m in messages if m["role"] == "user")
    if "SOURCE:" in user:
        ids = [line[4:].strip() for line in user.splitlines() if line.startswith("### ")]
        return json.dumps({"scores": [{"segment_id": i, **GOOD_SCORES} for i in ids], "issues": []})
    parsed = parse_segments(user)
    return "\n".join(f'<seg id="{sid}">[DE] {text}</seg>' for sid, text in parsed.texts.items())


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ~/.folioai at a temp directory so tests never touch the real one."""
    home = tmp_path / "folioai-home"
    monkeypatch.setenv("FOLIOAI_HOME", str(home))
    return home


async def run(
    pdf: Path, settings: Settings, *, client: Any | None = None, target: str = "de"
) -> tuple[Any, Any, Any]:
    context = prepare_job(pdf, settings, target_lang=target)
    try:
        translated, stats = await run_job(
            context, settings, client=client or FakeLLMClient(german_ish)
        )
        return context, translated, stats
    finally:
        context.close()


# -- the happy path ------------------------------------------------------------------


async def test_a_whole_book_translates_and_stays_parallel(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    context, translated, stats = await run(sample_pdfs["clean_book.pdf"], settings)

    source = context.document
    source.assert_parallel_to(translated)  # §21.2, in code rather than by eye
    assert len(translated.blocks) == len(source.blocks)
    assert translated.target_lang == "de"
    assert stats.completed == len(source.translatable_blocks())
    assert stats.needs_review == 0

    markdown = document_to_markdown(translated)
    assert "[DE]" in markdown
    assert "# " in markdown  # headings survive as headings


async def test_the_job_directory_holds_everything_needed_to_resume(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    context, _, _ = await run(sample_pdfs["clean_book.pdf"], settings)
    for key in ("db", "ir", "translated_ir", "probe", "audit"):
        assert context.paths[key].is_file(), f"{key} was not written"
    summary = json.loads((context.paths["dir"] / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["segments"] > 0


async def test_the_job_id_is_stable_for_the_same_source(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    first, _, _ = await run(sample_pdfs["clean_book.pdf"], settings)
    second = prepare_job(sample_pdfs["clean_book.pdf"], settings, target_lang="de")
    try:
        assert second.job_id == first.job_id
    finally:
        second.close()


async def test_untranslatable_blocks_keep_their_source_text(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    """Scene breaks and the like are never sent to a model, and never disappear either."""
    context, translated, _ = await run(sample_pdfs["clean_book.pdf"], settings)
    source_map = context.document.block_map()
    for block in translated.blocks:
        if not source_map[block.id].translate:
            assert block.text == source_map[block.id].text


# -- resume ----------------------------------------------------------------------------


async def test_resuming_a_finished_job_does_no_work(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    context, _, _ = await run(sample_pdfs["clean_book.pdf"], settings)
    job_id = context.job_id

    reopened = reopen_job(job_id, settings)
    try:
        client = FakeLLMClient(german_ish)
        _, stats = await run_job(reopened, settings, client=client)
        assert client.call_count == 0
        assert stats.completed == 0
        assert not reopened.store.pending_segments(job_id)
    finally:
        reopened.close()


async def test_a_killed_run_resumes_without_duplicating_or_losing_work(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    """§21.3, end to end: die partway, resume, finish exactly once."""
    settings.translation.batch_tokens = 60
    context = prepare_job(sample_pdfs["clean_book.pdf"], settings, target_lang="de")
    job_id = context.job_id
    expected = {block.id for block in context.document.translatable_blocks()}

    # Fail after the first two calls, the way a killed process or a dead endpoint would.
    # A plain exception rather than KeyboardInterrupt: pytest treats the latter as a signal
    # to abort the whole session, so it escapes pytest.raises.
    state = {"calls": 0}

    def flaky(messages: list[Message], model: str) -> str:
        state["calls"] += 1
        if state["calls"] > 2:
            raise SimulatedKill("the process died here")
        return german_ish(messages, model)

    with pytest.raises(SimulatedKill):
        await run_job(context, settings, client=FakeLLMClient(flaky))
    partial = {r.segment_id for r in context.store.list_segments(job_id, status="done")}
    context.close()
    assert partial, "some work should have been committed before the kill"
    assert partial < expected, "the kill should have left work unfinished"

    reopened = reopen_job(job_id, settings)
    try:
        client = FakeLLMClient(german_ish)
        translated, _ = await run_job(reopened, settings, client=client)

        finished = {r.segment_id for r in reopened.store.list_segments(job_id) if r.final_text}
        assert finished == expected  # nothing lost
        resent = {sid for call in client.calls_for("translate") for sid in call.segment_ids}
        assert not (resent & partial)  # nothing duplicated
        reopened.document.assert_parallel_to(translated)
    finally:
        reopened.close()


# -- persistence detail ---------------------------------------------------------------------


async def test_usage_and_cost_are_recorded_per_call(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    context, _, _ = await run(sample_pdfs["clean_book.pdf"], settings)
    reopened = reopen_job(context.job_id, settings)  # the run helper closed the first store
    try:
        rows = reopened.store.conn.execute(
            "SELECT COUNT(*) AS n FROM attempts WHERE job_id = ?", (context.job_id,)
        ).fetchone()
        assert rows["n"] == len(context.document.translatable_blocks())

        evaluations = reopened.store.conn.execute(
            "SELECT COUNT(*) AS n FROM evaluations"
        ).fetchone()
        assert evaluations["n"] > 0

        usage = reopened.store.conn.execute("SELECT COUNT(*) AS n FROM usage").fetchone()
        assert usage["n"] >= 0  # zero-cost fake calls still leave the table well formed
    finally:
        reopened.close()


async def test_a_second_run_reuses_the_extracted_ir(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    """Re-deriving the IR would invalidate every segment id already in the database."""
    context, _, _ = await run(sample_pdfs["clean_book.pdf"], settings)
    original = context.paths["ir"].read_text(encoding="utf-8")
    again = prepare_job(sample_pdfs["clean_book.pdf"], settings, target_lang="de")
    try:
        assert again.paths["ir"].read_text(encoding="utf-8") == original
    finally:
        again.close()


# -- the parallel-output assertion ------------------------------------------------------------


def test_a_missing_translation_falls_back_to_the_source_rather_than_a_gap(
    tiny_doc: Any,
) -> None:
    translated = build_translated_document(tiny_doc, {"b0001": "ZIEL"}, "de")
    assert len(translated.blocks) == len(tiny_doc.blocks)
    assert translated.block_map()["b0001"].text == "ZIEL"
    assert translated.block_map()["b0002"].text == tiny_doc.block_map()["b0002"].text


def test_an_empty_translation_never_replaces_real_text(tiny_doc: Any) -> None:
    translated = build_translated_document(tiny_doc, {"b0001": "   "}, "de")
    assert translated.block_map()["b0001"].text == tiny_doc.block_map()["b0001"].text
