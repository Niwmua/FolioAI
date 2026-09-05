"""Renderers, the export dispatcher, the quality report, and the review loop."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest
from sample_books import build_book as make_doc

from folioai.config import Settings
from folioai.errors import RenderError
from folioai.export import export_document, parse_formats
from folioai.ir import Block, Document
from folioai.render.base import (
    RenderContext,
    font_for,
    is_rtl,
    iter_chapters,
    script_for,
)
from folioai.render.html import inline_to_html, render_html
from folioai.render.markdown import document_to_markdown
from folioai.render.pdf import build_typst_source, render_txt
from folioai.report import chapter_chart_svg, gather, histogram_svg, render_report
from folioai.review import (
    ReviewItem,
    ReviewOutcome,
    collect_items,
    human_instructions,
    queue_for_retranslation,
    record_human_edit,
    run_review,
)
from folioai.store import JobStore, SegmentRecord


def source_doc() -> Document:
    doc = make_doc()
    for block in doc.blocks:
        block.text = f"SOURCE {block.id}"
    return doc


# -- shared render concerns ----------------------------------------------------------


@pytest.mark.parametrize(
    ("lang", "rtl"), [("de", False), ("ar", True), ("he", True), ("fa", True), ("ja", False)]
)
def test_rtl_detection(lang: str, rtl: bool) -> None:
    assert is_rtl(lang) is rtl


@pytest.mark.parametrize(
    ("lang", "font"),
    [
        ("de", "Noto Serif"),
        ("ja", "Noto Serif CJK"),
        ("zh-Hans", "Noto Serif CJK"),
        ("ar", "Noto Naskh Arabic"),
        ("he", "Noto Serif Hebrew"),
        ("ru", "Noto Serif"),
    ],
)
def test_font_is_script_appropriate(lang: str, font: str) -> None:
    """§14: a missing font must fail loudly, so the right one has to be named first."""
    assert font_for(lang) == font


def test_script_lookup_ignores_region_subtags() -> None:
    assert script_for("zh-Hans") == script_for("zh")


def test_every_block_reaches_a_chapter_even_when_orphaned() -> None:
    doc = make_doc()
    doc.blocks.append(Block(id="b9999", kind="paragraph", text="Orphan.", chapter_id=None))
    rendered = [b.id for chapter in iter_chapters(doc) for b in chapter.blocks]
    assert sorted(rendered) == sorted(b.id for b in doc.blocks)


# -- HTML ------------------------------------------------------------------------------


def test_inline_markup_becomes_html_and_prose_is_escaped() -> None:
    out = inline_to_html("A *word*, **strong**, `code`, and <not a tag> & an ampersand.")
    assert "<em>word</em>" in out
    assert "<strong>strong</strong>" in out
    assert "<code>code</code>" in out
    assert "&lt;not a tag&gt;" in out
    assert "&amp;" in out


def test_footnote_refs_become_links() -> None:
    assert 'href="#fn-1"' in inline_to_html("A sentence[^1].")


def test_html_is_self_contained_and_theme_aware() -> None:
    out = render_html(make_doc())
    assert out.startswith("<!doctype html>")
    assert "prefers-color-scheme" in out
    assert "<link" not in out and "src=" not in out  # nothing external
    assert 'lang="de"' in out and 'dir="ltr"' in out


def test_rtl_documents_get_rtl_direction() -> None:
    assert 'dir="rtl"' in render_html(make_doc("ar"))


def test_html_carries_the_translator_note() -> None:
    context = RenderContext(models={"translator": "some/model"})
    assert "some/model" in render_html(make_doc(), context)


def test_bilingual_html_shows_source_and_target() -> None:
    context = RenderContext(layout="bilingual-paragraph", source=source_doc())
    out = render_html(make_doc(), context)
    assert "SOURCE b0001" in out
    assert 'class="pair"' in out


def test_annotated_html_flags_low_scores() -> None:
    context = RenderContext(layout="annotated", scores={"b0001": 62.0, "b0005": 95.0}, min_score=80)
    out = render_html(make_doc(), context)
    assert 'data-score="62.0"' in out
    assert "flagged" in out


# -- Markdown, text, Typst --------------------------------------------------------------


def test_markdown_renders_every_block_kind() -> None:
    out = document_to_markdown(make_doc(), front_matter=False)
    assert "# Kapitel Eins" in out
    assert "> Ein Zitat." in out
    assert "* * *" in out
    assert "[^1]: Die Fußnote selbst." in out


def test_txt_is_plain_and_complete(tmp_path: Path) -> None:
    path = render_txt(make_doc(), tmp_path / "book.txt")
    text = path.read_text(encoding="utf-8")
    assert "KAPITEL EINS" in text
    assert "Ein Zitat." in text
    assert "<" not in text


def test_typst_source_escapes_prose_and_sets_typography() -> None:
    doc = make_doc()
    doc.blocks[1].text = "A #hash and a $dollar and an * asterisk."
    source = build_typst_source(doc, RenderContext())
    assert "\\#hash" in source
    assert "\\$dollar" in source
    assert 'font: "Noto Serif"' in source
    assert 'pagebreak(weak: true, to: "odd")' in source  # chapter openers on recto
    assert "header:" in source  # running heads


def test_typst_sets_rtl_for_arabic() -> None:
    assert "dir: rtl" in build_typst_source(make_doc("ar"), RenderContext())


# -- export dispatch -----------------------------------------------------------------------


def test_format_parsing_rejects_nonsense() -> None:
    assert parse_formats("md, epub ,pdf") == ["md", "epub", "pdf"]
    with pytest.raises(RenderError) as excinfo:
        parse_formats("md,pptx")
    assert "pptx" in excinfo.value.message


def test_export_writes_every_requested_format(tmp_path: Path, settings: Settings) -> None:
    result = export_document(
        make_doc(),
        tmp_path,
        formats=["md", "html", "txt", "epub", "docx"],
        context=RenderContext(),
        settings=settings,
    )
    written = {path.suffix for path in result.files}
    assert {".md", ".html", ".txt", ".epub", ".docx"} <= written
    assert all(path.stat().st_size > 0 for path in result.files)


def test_a_failing_format_does_not_stop_the_others(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing an EPUB because a PDF engine is missing would be absurd."""
    import folioai.render.pdf as pdf_module

    monkeypatch.setattr(pdf_module, "typst_available", lambda *_args: False)
    monkeypatch.setattr(pdf_module, "weasyprint_available", lambda *_args: False)

    result = export_document(
        make_doc(), tmp_path, formats=["md", "pdf"], context=RenderContext(), settings=settings
    )
    assert any(path.suffix == ".md" for path in result.files)
    assert any("pdf" in warning for warning in result.warnings)
    assert any("typst" in warning.lower() for warning in result.warnings)


def test_epub_is_a_valid_zip_with_the_right_language(tmp_path: Path, settings: Settings) -> None:
    result = export_document(
        make_doc("ar"), tmp_path, formats=["epub"], context=RenderContext(), settings=settings
    )
    epub_path = next(p for p in result.files if p.suffix == ".epub")
    with zipfile.ZipFile(epub_path) as archive:
        names = archive.namelist()
        assert "META-INF/container.xml" in names
        opf = next(n for n in names if n.endswith(".opf"))
        content = archive.read(opf).decode("utf-8")
    assert ">ar<" in content
    assert 'page-progression-direction="rtl"' in content


def test_epub_has_one_document_per_chapter(tmp_path: Path, settings: Settings) -> None:
    result = export_document(
        make_doc(), tmp_path, formats=["epub"], context=RenderContext(), settings=settings
    )
    with zipfile.ZipFile(result.files[0]) as archive:
        chapters = [n for n in archive.namelist() if Path(n).name.startswith("chap_")]
    assert len(chapters) == 2


def test_column_layout_warns_for_formats_that_cannot_do_it(
    tmp_path: Path, settings: Settings
) -> None:
    result = export_document(
        make_doc(),
        tmp_path,
        formats=["md", "html"],
        context=RenderContext(layout="bilingual-columns", source=source_doc()),
        settings=settings,
    )
    assert any("bilingual-columns" in w for w in result.warnings)
    assert result.files


def test_split_chapters_writes_one_markdown_file_each(tmp_path: Path, settings: Settings) -> None:
    result = export_document(
        make_doc(),
        tmp_path / "split",
        formats=["md"],
        context=RenderContext(),
        settings=settings,
        split_chapters=True,
    )
    assert len(result.files) == 2


# -- report ---------------------------------------------------------------------------------


def seed_job(store: JobStore, *, scores: dict[str, float], review: set[str]) -> None:
    doc = make_doc()
    store.upsert_segments(
        "job1",
        [
            SegmentRecord(
                segment_id=block.id,
                chapter_id=block.chapter_id,
                ordinal=index,
                kind=block.kind,
                source_text=f"SOURCE {block.id}",
                final_text=None,
                final_score=None,
                status="pending",
                needs_review=False,
                attempts_count=0,
            )
            for index, block in enumerate(doc.blocks)
        ],
    )
    for block in doc.blocks:
        attempt_id = store.record_attempt(
            job_id="job1",
            segment_id=block.id,
            attempt_no=1,
            model="test/translator",
            params={"temperature": 0.2},
            output_text=block.text,
            cost_usd=0.001,
        )
        score = scores.get(block.id)
        if score is not None:
            store.record_evaluation(
                attempt_id=attempt_id,
                evaluator_model="test/judge",
                scores={"segment_id": block.id, "completeness": int(score)},
                issues=(
                    [
                        {
                            "segment_id": block.id,
                            "dimension": "completeness",
                            "severity": "major",
                            "explanation": "a clause is missing",
                        }
                    ]
                    if score < 80
                    else []
                ),
                composite=score,
                passed=score >= 80,
            )
        store.finalize_segment(
            "job1",
            block.id,
            final_text=block.text,
            final_score=score,
            needs_review=block.id in review,
            status="review" if block.id in review else "done",
        )


def test_report_gathers_scores_and_flagged_segments(job_store: JobStore) -> None:
    seed_job(job_store, scores={"b0001": 62.0, "b0005": 95.0}, review={"b0001"})
    data = gather(job_store, "job1", document=make_doc())
    assert data.done_segments == 7
    assert data.needs_review == 1
    assert data.mean_score == pytest.approx(78.5)
    assert [entry["segment_id"] for entry in data.flagged] == ["b0001"]
    assert data.flagged[0]["attempts"][0]["issues"]


def test_report_renders_a_self_contained_page(job_store: JobStore) -> None:
    seed_job(job_store, scores={"b0001": 62.0, "b0005": 95.0}, review={"b0001"})
    data = gather(job_store, "job1", document=make_doc())
    page = render_report(data)

    assert page.startswith("<!doctype html>")
    assert "<link" not in page and "<script" not in page  # no assets, no network
    assert "b0001" in page
    assert "a clause is missing" in page
    assert "test/judge" in page
    assert "Extraction audit" in page


def test_report_survives_a_job_with_no_scores(job_store: JobStore) -> None:
    seed_job(job_store, scores={}, review=set())
    data = gather(job_store, "job1", document=make_doc())
    page = render_report(data)
    assert "No scores yet" in page
    assert data.mean_score == 0.0


def test_charts_handle_empty_and_tiny_inputs() -> None:
    assert "No scores yet" in histogram_svg([])
    assert "Not enough chapters" in chapter_chart_svg({"ch01": 90.0}, {})
    assert "<svg" in histogram_svg([85.0, 92.0, 61.0])
    assert "<svg" in chapter_chart_svg({"ch01": 90.0, "ch02": 75.0}, {"ch01": "One"})


def test_report_escapes_content_from_the_book(job_store: JobStore) -> None:
    """The text has been through a language model; it must not become markup."""
    seed_job(job_store, scores={"b0001": 50.0}, review={"b0001"})
    job_store.finalize_segment(
        "job1",
        "b0001",
        final_text="<script>alert(1)</script>",
        final_score=50.0,
        needs_review=True,
    )
    page = render_report(gather(job_store, "job1", document=make_doc()))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_report_data_handles_a_missing_job(job_store: JobStore) -> None:
    with pytest.raises(ValueError, match="no job row"):
        gather(job_store, "does-not-exist")


# -- review loop ---------------------------------------------------------------------------


def test_flagged_segments_come_back_worst_first(job_store: JobStore) -> None:
    seed_job(
        job_store, scores={"b0001": 62.0, "b0002": 41.0, "b0005": 95.0}, review={"b0001", "b0002"}
    )
    items = collect_items(job_store, "job1")
    assert [item.id for item in items] == ["b0002", "b0001"]
    assert items[0].issues


def test_max_score_widens_the_queue(job_store: JobStore) -> None:
    seed_job(job_store, scores={"b0001": 62.0, "b0005": 85.0}, review={"b0001"})
    assert len(collect_items(job_store, "job1")) == 1
    assert len(collect_items(job_store, "job1", max_score=90)) >= 2


def test_a_human_edit_is_a_new_attempt_not_an_overwrite(job_store: JobStore) -> None:
    """Nothing is lost: the model's version stays answerable for later (§15)."""
    seed_job(job_store, scores={"b0001": 62.0}, review={"b0001"})
    record_human_edit(job_store, "job1", "b0001", "Von Hand korrigiert.")

    rows = job_store.list_attempts("job1", "b0001")
    assert [row["model"] for row in rows] == ["test/translator", "human"]
    assert rows[0]["output_text"] != rows[1]["output_text"]

    segment = job_store.get_segment("job1", "b0001")
    assert segment is not None
    assert segment.final_text == "Von Hand korrigiert."
    assert not segment.needs_review
    assert segment.status == "done"


def test_a_human_edit_clears_the_machine_score(job_store: JobStore) -> None:
    """A human edit is the standard, not another candidate to be judged."""
    seed_job(job_store, scores={"b0001": 62.0}, review={"b0001"})
    record_human_edit(job_store, "job1", "b0001", "Korrigiert.")
    segment = job_store.get_segment("job1", "b0001")
    assert segment is not None and segment.final_score is None


def test_queueing_for_retranslation_makes_the_segment_pending_again(
    job_store: JobStore,
) -> None:
    seed_job(job_store, scores={"b0001": 62.0}, review={"b0001"})
    queue_for_retranslation(job_store, "job1", "b0001", "Keep the sarcasm.")

    assert "b0001" in {s.segment_id for s in job_store.pending_segments("job1")}
    assert human_instructions(job_store, "job1", "b0001") == ["Keep the sarcasm."]


def test_the_review_loop_dispatches_each_action(job_store: JobStore) -> None:
    seed_job(
        job_store,
        scores={"b0001": 62.0, "b0002": 41.0, "b0005": 70.0},
        review={"b0001", "b0002", "b0005"},
    )
    items = collect_items(job_store, "job1")
    actions = iter(["accept", "retranslate", "skip"])

    def present(item: ReviewItem, index: int, total: int) -> Any:
        return next(actions)

    outcome = run_review(
        job_store, "job1", items, present=present, ask_instruction=lambda item: "be literal"
    )
    assert outcome.accepted == 1
    assert outcome.retranslated == 1
    assert outcome.skipped == 1
    assert outcome.touched == 2


def test_quitting_stops_the_loop_without_touching_the_rest(job_store: JobStore) -> None:
    seed_job(job_store, scores={"b0001": 62.0, "b0002": 41.0}, review={"b0001", "b0002"})
    items = collect_items(job_store, "job1")

    outcome = run_review(
        job_store,
        "job1",
        items,
        present=lambda item, index, total: "quit",
        ask_instruction=lambda item: "",
    )
    assert outcome.quit_early
    assert outcome.touched == 0
    assert all(s.needs_review for s in job_store.list_segments("job1") if s.final_score)


def test_an_empty_retranslation_instruction_is_a_skip(job_store: JobStore) -> None:
    seed_job(job_store, scores={"b0001": 62.0}, review={"b0001"})
    items = collect_items(job_store, "job1")
    outcome = run_review(
        job_store,
        "job1",
        items,
        present=lambda item, index, total: "retranslate",
        ask_instruction=lambda item: "   ",
    )
    assert outcome.skipped == 1
    assert outcome.retranslated == 0


def test_review_outcome_counts_are_independent() -> None:
    outcome = ReviewOutcome(accepted=2, edited=1, retranslated=3, skipped=4)
    assert outcome.touched == 6
