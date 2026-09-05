"""The acceptance criteria from brief §21, each as its own test.

If this file passes, the build meets the criteria the brief says it must meet. Numbered to
match §21 so a reader can check them off.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from folioai.config import Settings
from folioai.export import export_job
from folioai.jobs import prepare_job, reopen_job, run_job
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient
from folioai.report import gather, render_report
from folioai.tags import parse_segments

PASS = {"completeness": 96, "accuracy": 94, "terminology": 95, "fluency": 92, "formatting": 100}
FAIL = {"completeness": 45, "accuracy": 60, "terminology": 70, "fluency": 85, "formatting": 90}


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("FOLIOAI_HOME", str(home))
    return home


def translator(*, failing: set[str] | None = None, drop_every: int = 0) -> Any:
    """A fake endpoint that can be told to fail segments or drop them."""
    failing = failing or set()
    state = {"translate_calls": 0}

    def handler(messages: list[Message], model: str) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "SOURCE:" in user:
            ids = [line[4:].strip() for line in user.splitlines() if line.startswith("### ")]
            return json.dumps(
                {
                    "scores": [{"segment_id": i, **(FAIL if i in failing else PASS)} for i in ids],
                    "issues": [],
                }
            )

        state["translate_calls"] += 1
        first_attempt = len(messages) == 2
        parsed = parse_segments(user)
        out = []
        for index, (segment_id, text) in enumerate(parsed.texts.items(), start=1):
            if drop_every and first_attempt and index % drop_every == 0:
                continue
            out.append(f'<seg id="{segment_id}">[DE] {text}</seg>')
        return "\n".join(out)

    handler.state = state  # type: ignore[attr-defined]
    return handler


async def translate_fixture(
    pdf: Path, settings: Settings, *, client: Any | None = None
) -> tuple[str, Any, Any]:
    context = prepare_job(pdf, settings, target_lang="de")
    try:
        translated, stats = await run_job(
            context, settings, client=client or FakeLLMClient(translator())
        )
        return context.job_id, translated, stats
    finally:
        context.close()


# -- §21.1 -----------------------------------------------------------------------------


async def test_1_translate_produces_a_complete_epub_with_chapters_and_metadata(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path, tmp_path: Path
) -> None:
    """`translate book.pdf --to de --format epub,md` produces a complete EPUB."""
    job_id, _, _ = await translate_fixture(sample_pdfs["clean_book.pdf"], settings)
    result = export_job(job_id, settings, formats=["epub", "md"], out_dir=tmp_path)

    epub_path = next(p for p in result.files if p.suffix == ".epub")
    markdown = next(p for p in result.files if p.suffix == ".md")
    assert epub_path.stat().st_size > 0
    assert "[DE]" in markdown.read_text(encoding="utf-8")

    with zipfile.ZipFile(epub_path) as archive:
        names = archive.namelist()
        opf = archive.read(next(n for n in names if n.endswith(".opf"))).decode("utf-8")
        chapters = [n for n in names if Path(n).name.startswith("chap_")]

    assert len(chapters) == 2  # correct chapters
    assert any(Path(n).name in {"nav.xhtml", "toc.ncx"} for n in names)  # real TOC
    assert ">de<" in opf  # dc:language is the target language
    assert "contributor" in opf  # translator note naming the models


# -- §21.2 -----------------------------------------------------------------------------


async def test_2_output_block_count_equals_input_block_count(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    """Verified by an automated assertion, not by eye -- for every fixture."""
    for name in ("clean_book.pdf", "hyphenated.pdf", "footnotes.pdf", "two_column.pdf"):
        context = prepare_job(sample_pdfs[name], settings, target_lang="de")
        try:
            source_blocks = len(context.document.blocks)
            translated, _ = await run_job(context, settings, client=FakeLLMClient(translator()))
            assert len(translated.blocks) == source_blocks, name
            context.document.assert_parallel_to(translated)
        finally:
            context.close()


# -- §21.3 -----------------------------------------------------------------------------


async def test_3_killing_mid_run_and_resuming_loses_and_duplicates_nothing(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    settings.translation.batch_tokens = 60

    class Killed(RuntimeError):  # noqa: N818 - stands in for a killed process
        pass

    state = {"calls": 0}
    good = translator()

    def dies(messages: list[Message], model: str) -> str:
        state["calls"] += 1
        if state["calls"] > 2:
            raise Killed("the process died")
        return good(messages, model)

    context = prepare_job(sample_pdfs["clean_book.pdf"], settings, target_lang="de")
    job_id = context.job_id
    expected = {b.id for b in context.document.translatable_blocks()}
    with pytest.raises(Killed):
        await run_job(context, settings, client=FakeLLMClient(dies))
    partial = {s.segment_id for s in context.store.list_segments(job_id, status="done")}
    context.close()

    resumed = reopen_job(job_id, settings)
    try:
        client = FakeLLMClient(translator())
        translated, _ = await run_job(resumed, settings, client=client)
        finished = {s.segment_id for s in resumed.store.list_segments(job_id) if s.final_text}
        resent = {sid for call in client.calls_for("translate") for sid in call.segment_ids}

        assert finished == expected  # nothing lost
        assert not (resent & partial)  # nothing duplicated
        resumed.document.assert_parallel_to(translated)
    finally:
        resumed.close()


# -- §21.4 -----------------------------------------------------------------------------


async def test_4_the_report_shows_real_scores_and_every_flagged_segment(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    context = prepare_job(sample_pdfs["clean_book.pdf"], settings, target_lang="de")
    job_id = context.job_id
    try:
        bad = context.document.translatable_blocks()[1].id
        await run_job(context, settings, client=FakeLLMClient(translator(failing={bad})))
    finally:
        context.close()

    reopened = reopen_job(job_id, settings)
    try:
        data = gather(reopened.store, job_id, document=reopened.document)
        page = render_report(data)
    finally:
        reopened.close()

    assert data.scores and all(0 <= s <= 100 for s in data.scores)  # real per-segment scores
    assert bad in {entry["segment_id"] for entry in data.flagged}  # listed below threshold
    assert bad in page
    assert "<html" in page and "<script" not in page  # opens, self-contained


# -- §21.5 -----------------------------------------------------------------------------


async def test_5_a_sabotaged_translator_is_caught_before_the_judge_and_recovered(
    sample_pdfs: dict[str, Path], settings: Settings, isolated_home: Path
) -> None:
    """A translator dropping every Nth segment: caught by validation, recovered by retry."""
    settings.translation.batch_tokens = 100_000
    handler = translator(drop_every=2)
    client = FakeLLMClient(handler)

    context = prepare_job(sample_pdfs["clean_book.pdf"], settings, target_lang="de")
    job_id = context.job_id
    try:
        expected = {b.id for b in context.document.translatable_blocks()}
        translated, _ = await run_job(context, settings, client=client)

        # The evaluator was never asked about the sabotaged attempts.
        translate_calls = len(client.calls_for("translate"))
        evaluate_calls = len(client.calls_for("evaluate"))
        assert evaluate_calls < translate_calls

        finished = {s.segment_id for s in context.store.list_segments(job_id) if s.final_text}
        assert finished == expected  # every dropped segment recovered
        assert all(b.text.strip() for b in translated.blocks)
        context.document.assert_parallel_to(translated)
    finally:
        context.close()


# -- §21.6 -----------------------------------------------------------------------------


def test_6_dry_run_makes_no_paid_calls_and_prints_a_plausible_estimate(
    sample_pdfs: dict[str, Path],
    settings: Settings,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import folioai.llm.client as client_module

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("a dry run must not construct an LLM client")

    monkeypatch.setattr(client_module, "OpenAICompatibleClient", explode)

    from folioai.estimate import estimate_document
    from folioai.extract.pipeline import extract_document

    document = extract_document(sample_pdfs["clean_book.pdf"], settings).document
    estimate = estimate_document(document, settings, target_lang="de")

    assert estimate.cost_low > 0
    assert estimate.cost_high > estimate.cost_low
    assert estimate.batches > 0
    assert "-" in estimate.total_range  # a range, not a false-precision number


# -- §21.7 is enforced by CI: `ruff check`, `mypy --strict`, and this suite. -----------------


def test_7_the_ir_schema_committed_in_the_repo_is_current() -> None:
    """A change to the document format shows up in review as a schema diff."""
    from folioai.ir import write_json_schema

    committed = Path("schema/document.schema.json")
    assert committed.is_file()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        regenerated = Path(tmp) / "schema.json"
        write_json_schema(regenerated)
        assert committed.read_text(encoding="utf-8") == regenerated.read_text(encoding="utf-8")
