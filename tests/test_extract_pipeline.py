"""End-to-end extraction over the synthetic PDFs, plus the Markdown renderer.

These are the golden tests for milestone 2: real PDFs in, IR and Markdown out, with the
specific pathologies each fixture encodes verified in the output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from folioai.config import Settings
from folioai.errors import ExtractionError
from folioai.extract.pipeline import build_extractor, extract_document, select_extractor
from folioai.extract.probe import probe_pdf
from folioai.render.markdown import document_to_markdown, write_markdown


def extract(path: Path, settings: Settings):  # type: ignore[no-untyped-def]
    return extract_document(path, settings)


# -- the happy path -------------------------------------------------------------------


def test_clean_book_extracts_with_chapters(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    result = extract(sample_pdfs["clean_book.pdf"], settings)
    doc = result.document

    assert doc.source_lang == "en"
    assert len(doc.chapters) == 2
    assert [c.title for c in doc.chapters] == ["Chapter One", "Chapter Two"]
    assert doc.extraction_report.structure_source == "outline"
    assert doc.extraction_report.extractor == "pymupdf"


def test_running_heads_and_folios_are_gone(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    markdown = document_to_markdown(extract(sample_pdfs["clean_book.pdf"], settings).document)
    assert "A Test Book" not in markdown
    assert "- 1 -" not in markdown
    assert "The lamp above the door" in markdown


def test_a_paragraph_split_over_a_page_break_is_rejoined(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    doc = extract(sample_pdfs["clean_book.pdf"], settings).document
    joined = [b for b in doc.blocks if "great many other things" in b.text]
    assert len(joined) == 1
    assert "wall she had built" in joined[0].text  # the continuation from the next page
    assert len(joined[0].source_pages) == 2


def test_every_block_belongs_to_a_chapter(sample_pdfs: dict[str, Path], settings: Settings) -> None:
    doc = extract(sample_pdfs["clean_book.pdf"], settings).document
    assert all(block.chapter_id for block in doc.blocks)
    listed = [bid for chapter in doc.chapters for bid in chapter.block_ids]
    assert sorted(listed) == sorted(block.id for block in doc.blocks)


def test_block_ids_are_sequential_and_stable(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    first = extract(sample_pdfs["clean_book.pdf"], settings).document
    second = extract(sample_pdfs["clean_book.pdf"], settings).document
    assert [b.id for b in first.blocks] == [b.id for b in second.blocks]
    assert [b.text for b in first.blocks] == [b.text for b in second.blocks]
    assert first.blocks[0].id == "b0000"


# -- the pathologies ------------------------------------------------------------------


def test_hyphenation_is_resolved_both_ways(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    result = extract(sample_pdfs["hyphenated.pdf"], settings)
    markdown = document_to_markdown(result.document)
    assert "extraordinary conclusion" in markdown
    assert "extraor- dinary" not in markdown
    assert "well-being of the town" in markdown
    assert "wellbeing" not in markdown
    assert result.audit.dehyphenations  # every decision recorded for audit


def test_running_furniture_is_stripped_but_body_text_survives(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    result = extract(sample_pdfs["furniture.pdf"], settings)
    markdown = document_to_markdown(result.document)
    assert "The Long Afternoon" not in markdown
    assert markdown.count("This is the body text of page") == 6
    assert result.audit.stripped_headers >= 6
    assert result.audit.stripped_page_numbers >= 6


def test_drop_cap_rejoins_its_word(sample_pdfs: dict[str, Path], settings: Settings) -> None:
    markdown = document_to_markdown(extract(sample_pdfs["dropcap.pdf"], settings).document)
    assert "When the bell rang" in markdown
    assert "W hen" not in markdown


def test_footnotes_become_their_own_blocks(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    doc = extract(sample_pdfs["footnotes.pdf"], settings).document
    notes = [b for b in doc.blocks if b.kind == "footnote"]
    assert len(notes) == 1
    assert "date is disputed" in notes[0].text

    body = [b for b in doc.blocks if b.kind != "footnote"]
    assert not any("date is disputed" in b.text for b in body)  # never left inline
    anchored = [b for b in body if b.footnote_refs]
    assert anchored and anchored[0].footnote_refs == ["1"]
    assert "generalindifference" not in document_to_markdown(doc)


def test_two_columns_read_in_order_not_interleaved(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    doc = extract(sample_pdfs["two_column.pdf"], settings).document
    markdown = document_to_markdown(doc)
    assert "The first column begins here and continues" in markdown
    assert "The second column is a separate thread" in markdown
    first = markdown.index("The first column")
    second = markdown.index("The second column")
    assert first < second


# -- invariants -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "clean_book.pdf",
        "hyphenated.pdf",
        "furniture.pdf",
        "dropcap.pdf",
        "footnotes.pdf",
        "two_column.pdf",
    ],
)
def test_no_fixture_loses_content(
    name: str, sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    """Block count equals paragraph count, and nothing is empty (§21.2)."""
    result = extract(sample_pdfs[name], settings)
    doc = result.document
    assert len(doc.blocks) == doc.extraction_report.block_count
    assert all(block.text.strip() for block in doc.blocks)
    assert len(doc.blocks) == len({block.id for block in doc.blocks})


def test_ir_round_trips_through_disk(
    sample_pdfs: dict[str, Path], settings: Settings, tmp_path: Path
) -> None:
    from folioai.ir import Document

    doc = extract(sample_pdfs["clean_book.pdf"], settings).document
    path = tmp_path / "ir.json"
    doc.save(path)
    assert Document.load(path) == doc


# -- extractor selection ------------------------------------------------------------------


def test_selection_honours_an_explicit_override(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    settings.extraction.extractor = "pymupdf"
    probe = probe_pdf(sample_pdfs["clean_book.pdf"], settings)
    extractor, reason = select_extractor(probe, settings)
    assert extractor.name == "pymupdf"
    assert "explicitly requested" in reason


def test_selection_follows_the_probe_by_default(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    probe = probe_pdf(sample_pdfs["clean_book.pdf"], settings)
    extractor, reason = select_extractor(probe, settings)
    assert extractor.name == "pymupdf"
    assert reason == probe.recommendation_reason


def test_marker_extractor_exists_and_reports_its_own_availability() -> None:
    """Never a hard dependency (§4.2): it says how to install itself instead of crashing."""
    extractor = build_extractor("marker")
    assert extractor.name == "marker"
    available, hint = extractor.available()
    if not available:
        assert "uv sync --extra ml" in hint


def test_requesting_an_unavailable_marker_explains_the_install(settings: Settings) -> None:
    from folioai.extract.marker import MarkerExtractor

    extractor = MarkerExtractor()
    if extractor.available()[0]:
        pytest.skip("marker is installed here, so there is no failure path to test")
    with pytest.raises(ExtractionError) as excinfo:
        extractor.extract(Path("whatever.pdf"), settings)
    assert "extra ml" in (excinfo.value.remedy or "")


def test_unknown_extractor_lists_the_real_ones() -> None:
    with pytest.raises(ExtractionError) as excinfo:
        build_extractor("magic")
    assert "pymupdf" in (excinfo.value.remedy or "")


def test_ocr_without_a_language_says_which_flag_to_pass(settings: Settings) -> None:
    from folioai.extract.ocr import OCRExtractor

    extractor = OCRExtractor()
    available, hint = extractor.available()
    if not available:
        assert "ocrmypdf" in hint
        return
    with pytest.raises(ExtractionError) as excinfo:
        extractor.extract(Path("nonexistent.pdf"), settings)
    assert "--ocr-lang" in (excinfo.value.remedy or "")


def test_poppler_reports_its_own_availability() -> None:
    from folioai.extract.poppler import PopplerExtractor

    available, hint = PopplerExtractor().available()
    assert available or "poppler" in hint


# -- markdown renderer ----------------------------------------------------------------------


def test_markdown_has_front_matter_and_headings(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    markdown = document_to_markdown(extract(sample_pdfs["clean_book.pdf"], settings).document)
    assert markdown.startswith("---\n")
    assert "# Chapter One" in markdown
    assert "source_lang: en" in markdown


def test_markdown_renders_one_chunk_per_block(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    doc = extract(sample_pdfs["clean_book.pdf"], settings).document
    body = document_to_markdown(doc, front_matter=False)
    chunks = [c for c in body.split("\n\n") if c.strip()]
    assert len(chunks) == len(doc.blocks)


def test_split_chapters_writes_one_file_each(
    sample_pdfs: dict[str, Path], settings: Settings, tmp_path: Path
) -> None:
    doc = extract(sample_pdfs["clean_book.pdf"], settings).document
    written = write_markdown(doc, tmp_path / "book", split_chapters=True)
    assert len(written) == len(doc.chapters)
    assert all(path.read_text(encoding="utf-8").strip() for path in written)
