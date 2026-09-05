"""Chapter subsetting, the vision fallback, and the marker extractor's Markdown round trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from folioai.chapters import apply_selection, parse_selection, selection_summary
from folioai.config import Settings
from folioai.errors import ConfigError, ExtractionError
from folioai.extract.base import RawDocument, RawLine, RawPage, RawSpan
from folioai.extract.marker import MarkerExtractor
from folioai.extract.vision import (
    VisionResult,
    apply_transcriptions,
    find_bad_pages,
    rasterize_page,
    transcribe_pages,
)
from folioai.ir import Block, Chapter, Document, ExtractionReport
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient


def book(chapter_count: int = 5, *, numbered: bool = True) -> Document:
    blocks: list[Block] = []
    chapters: list[Chapter] = []
    for index in range(1, chapter_count + 1):
        ids = [f"b{index:02d}{n}" for n in range(3)]
        blocks.append(
            Block(
                id=ids[0],
                kind="heading",
                level=1,
                text=f"Chapter {index}",
                chapter_id=f"ch{index:02d}",
            )
        )
        blocks.append(
            Block(
                id=ids[1],
                kind="paragraph",
                text=f"Body of chapter {index}. " * 10,
                chapter_id=f"ch{index:02d}",
            )
        )
        blocks.append(
            Block(
                id=ids[2],
                kind="scene_break",
                text="* * *",
                chapter_id=f"ch{index:02d}",
                translate=False,
            )
        )
        chapters.append(
            Chapter(
                id=f"ch{index:02d}",
                title=f"Chapter {index}",
                number=index if numbered else None,
                block_ids=ids,
            )
        )
    return Document(
        source_lang="en",
        blocks=blocks,
        chapters=chapters,
        extraction_report=ExtractionReport(extractor="test"),
    )


def raw_pages(lengths: list[int]) -> RawDocument:
    """A raw document whose pages hold the given number of characters."""
    raw = RawDocument(extractor="pymupdf")
    for number, length in enumerate(lengths, start=1):
        text = "word " * max(length // 5, 0)
        lines = (
            [
                RawLine(
                    spans=[RawSpan(text=text.strip(), size=10.0)],
                    bbox=(0.0, 0.0, 100.0, 12.0),
                    page=number,
                )
            ]
            if text.strip()
            else []
        )
        raw.pages.append(RawPage(number=number, width=612.0, height=792.0, lines=lines))
    return raw


# -- parsing ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3", {3}),
        ("3-7", {3, 4, 5, 6, 7}),
        ("3-5,9", {3, 4, 5, 9}),
        (" 1 , 2 ", {1, 2}),
        ("2-2", {2}),
    ],
)
def test_selections_parse(value: str, expected: set[int]) -> None:
    assert set(parse_selection(value).numbers) == expected


@pytest.mark.parametrize("value", ["", "abc", "3-", "-3", "3..5", "7-3"])
def test_malformed_selections_are_rejected(value: str) -> None:
    """Silently ignoring this would translate the wrong part of the book."""
    with pytest.raises(ConfigError) as excinfo:
        parse_selection(value)
    assert excinfo.value.remedy


def test_a_backwards_range_says_so() -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_selection("7-3")
    assert "backwards" in excinfo.value.message


# -- applying -------------------------------------------------------------------------


def test_only_the_selected_chapters_stay_translatable() -> None:
    doc = book(5)
    kept = apply_selection(doc, parse_selection("2-3"))
    translatable = {b.chapter_id for b in doc.translatable_blocks()}
    assert translatable == {"ch02", "ch03"}
    assert kept == 4  # two chapters, two translatable blocks each


def test_the_ir_keeps_every_block_so_output_stays_parallel() -> None:
    """§21.2 still has to hold: subsetting marks blocks, it never removes them."""
    doc = book(4)
    before = [b.id for b in doc.blocks]
    apply_selection(doc, parse_selection("2"))
    assert [b.id for b in doc.blocks] == before


def test_selection_never_switches_an_untranslatable_block_on() -> None:
    doc = book(3)
    apply_selection(doc, parse_selection("1-3"))
    scene_breaks = [b for b in doc.blocks if b.kind == "scene_break"]
    assert scene_breaks and all(not b.translate for b in scene_breaks)


def test_unnumbered_chapters_fall_back_to_position() -> None:
    doc = book(4, numbered=False)
    apply_selection(doc, parse_selection("2"))
    assert {b.chapter_id for b in doc.translatable_blocks()} == {"ch02"}


def test_a_selection_matching_nothing_is_an_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        apply_selection(book(3), parse_selection("40-42"))
    assert "3 chapter" in (excinfo.value.remedy or "")


def test_a_partly_matching_selection_still_works() -> None:
    doc = book(3)
    apply_selection(doc, parse_selection("2,99"))
    assert {b.chapter_id for b in doc.translatable_blocks()} == {"ch02"}


def test_summary_reports_what_will_be_translated() -> None:
    summary = selection_summary(book(5), parse_selection("2-3"))
    assert "2 of 5 chapters" in summary
    assert "words" in summary


# -- vision: choosing pages ---------------------------------------------------------------


def test_pages_with_almost_no_text_are_candidates(settings: Settings) -> None:
    raw = raw_pages([1000, 1000, 5, 1000, 1000])
    assert find_bad_pages(raw, settings, max_pages=5) == [3]


def test_pages_full_of_replacement_characters_are_candidates(settings: Settings) -> None:
    raw = raw_pages([1000, 1000, 1000])
    raw.pages[1].lines[0].spans[0].text = "�" * 200
    assert 2 in find_bad_pages(raw, settings, max_pages=5)


def test_a_uniformly_sparse_book_produces_no_candidates(settings: Settings) -> None:
    """Poetry is not a broken extraction; 'short' only means short *relative to the book*."""
    assert find_bad_pages(raw_pages([120, 130, 110, 125]), settings, max_pages=5) == []


def test_candidates_are_capped(settings: Settings) -> None:
    raw = raw_pages([1000] * 5 + [2] * 8)
    assert len(find_bad_pages(raw, settings, max_pages=3)) == 3


def test_an_empty_document_has_no_candidates(settings: Settings) -> None:
    assert find_bad_pages(RawDocument(), settings, max_pages=5) == []


# -- vision: transcription ------------------------------------------------------------------


def transcriber(text: str = "Transcribed line one.\n\nAnd a second paragraph.") -> object:
    def handler(messages: list[Message], model: str) -> str:
        return text

    return handler


async def test_transcription_replaces_the_page_and_records_it(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    client = FakeLLMClient(transcriber())
    result = await transcribe_pages(sample_pdfs["clean_book.pdf"], [2], client, settings)
    assert result.pages[2].startswith("Transcribed")

    raw = raw_pages([1000, 5, 1000])
    replaced = apply_transcriptions(raw, result, body_size=10.0)
    assert replaced == 1
    assert raw.pages[1].extractor == "vision"
    assert "Transcribed line one." in raw.pages[1].lines[0].text
    assert raw.fallback_pages["vision"] == [2]
    assert any("vision model" in w for w in raw.warnings)


async def test_the_page_cap_is_a_refusal_not_a_truncation(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    """Silently doing less than asked would hide the cost decision."""
    settings.extraction.vision_max_pages = 2
    with pytest.raises(ExtractionError) as excinfo:
        await transcribe_pages(
            sample_pdfs["clean_book.pdf"], [1, 2, 3, 4], FakeLLMClient(transcriber()), settings
        )
    assert "vision_max_pages" in (excinfo.value.remedy or "")


async def test_one_failed_page_does_not_lose_the_others(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    calls = {"n": 0}

    def flaky(messages: list[Message], model: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("vision model refused")
        return "Transcribed."

    result = await transcribe_pages(
        sample_pdfs["clean_book.pdf"], [1, 2], FakeLLMClient(flaky), settings
    )
    assert result.failed == [1]
    assert 2 in result.pages


async def test_an_empty_transcription_counts_as_a_failure(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    result = await transcribe_pages(
        sample_pdfs["clean_book.pdf"], [1], FakeLLMClient(transcriber("   ")), settings
    )
    assert result.failed == [1]
    assert not result.pages


def test_applying_nothing_changes_nothing() -> None:
    raw = raw_pages([100, 100])
    assert apply_transcriptions(raw, VisionResult(), body_size=10.0) == 0
    assert raw.warnings == []


def test_rasterizing_a_page_produces_a_png(sample_pdfs: dict[str, Path]) -> None:
    image = rasterize_page(sample_pdfs["clean_book.pdf"], 1, dpi=72)
    assert image[:8] == b"\x89PNG\r\n\x1a\n"


def test_rasterizing_a_page_that_does_not_exist_says_so(
    sample_pdfs: dict[str, Path],
) -> None:
    with pytest.raises(ExtractionError) as excinfo:
        rasterize_page(sample_pdfs["clean_book.pdf"], 99)
    assert "does not exist" in excinfo.value.message


async def test_the_pipeline_skips_vision_when_nothing_looks_broken(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    from folioai.extract.pipeline import extract_document_with_vision

    client = FakeLLMClient(transcriber())
    result = await extract_document_with_vision(sample_pdfs["clean_book.pdf"], settings, client)
    assert client.call_count == 0  # a clean book costs nothing
    assert result.document.blocks


# -- marker ---------------------------------------------------------------------------------


def test_marker_reports_its_own_availability() -> None:
    available, hint = MarkerExtractor().available()
    assert available or "extra ml" in hint


def test_marker_markdown_becomes_ordered_lines_with_heading_sizes() -> None:
    markdown = (
        "# Chapter One\n\nA first paragraph of prose.\n\n"
        "## A Section\n\n> A quotation.\n\n---\n\n# Chapter Two\n\nMore prose here.\n"
    )
    raw = MarkerExtractor()._markdown_to_document(markdown)

    assert raw.page_count == 2
    texts = [line.text for line in raw.lines()]
    assert texts[0] == "Chapter One"
    assert "A first paragraph of prose." in texts
    assert "A quotation." in texts

    heading = next(line for line in raw.lines() if line.text == "Chapter One")
    body = next(line for line in raw.lines() if line.text.startswith("A first paragraph"))
    assert heading.size > body.size  # so structure detection still has sizes to cluster


def test_marker_output_is_never_empty() -> None:
    raw = MarkerExtractor()._markdown_to_document("")
    assert raw.page_count == 1
