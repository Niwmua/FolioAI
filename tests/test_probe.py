"""Probe: diagnosis without external binaries, and a garbling score that knows its limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from folioai.config import Settings
from folioai.errors import ExtractionError
from folioai.extract.probe import (
    dominant_script,
    probe_pdf,
    render_probe_report,
    sample_page_numbers,
    score_garbling,
)

CLEAN_ENGLISH = (
    "The lamp above the door had been broken for a week, and nobody in the house had "
    "thought to mention it. She noticed it on the Tuesday, on her way back from the "
    "market with a basket she could barely carry, and said nothing about it at all."
)
GARBLED = "��\x01\x02\x03Ø¤¥§±¶‡�\x00\x00�¿¡«»\x04\x05��\x06"
JAPANESE = "その日の午後、彼女は市場からの帰り道でそれに気づいた。誰も何も言わなかった。"


def test_clean_english_prose_scores_clean(settings: Settings) -> None:
    scores = score_garbling(CLEAN_ENGLISH, "latin", settings)
    assert scores["verdict"] == "clean"


def test_newlines_do_not_make_prose_look_garbled(settings: Settings) -> None:
    """Line-broken samples are the normal case; newlines are Cc but not corruption."""
    sample = CLEAN_ENGLISH.replace(" ", "\n", 12)
    assert score_garbling(sample, "latin", settings)["verdict"] == "clean"


def test_mojibake_scores_garbled(settings: Settings) -> None:
    assert score_garbling(GARBLED, "latin", settings)["verdict"] == "garbled"


def test_cjk_reports_unknown_rather_than_guessing(settings: Settings) -> None:
    """Space and vowel statistics mean nothing here; guessing would route it to OCR."""
    scores = score_garbling(JAPANESE, "cjk", settings)
    assert scores["verdict"] == "unknown"
    assert scores["space_ratio"] is None
    assert scores["vowel_ratio"] is None


def test_cjk_with_real_corruption_is_still_caught(settings: Settings) -> None:
    """Script-independent evidence still applies to scripts with no prose heuristic."""
    assert score_garbling(JAPANESE + GARBLED * 3, "cjk", settings)["verdict"] == "garbled"


def test_empty_sample_is_unknown(settings: Settings) -> None:
    assert score_garbling("", "latin", settings)["verdict"] == "unknown"


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("The lamp above the door", "latin"),
        ("その日の午後、彼女は", "cjk"),
        ("Лампа над дверью", "cyrillic"),
        ("كان المصباح فوق الباب", "arabic"),
    ],
)
def test_script_detection(text: str, script: str) -> None:
    assert dominant_script(text) == script


def test_sampling_spreads_through_the_book_and_skips_front_matter(settings: Settings) -> None:
    pages = sample_page_numbers(300, settings)
    assert len(pages) == 3
    assert min(pages) > 30  # past the front matter
    assert max(pages) <= 300
    assert pages == sorted(pages)


def test_sampling_handles_a_one_page_document(settings: Settings) -> None:
    assert sample_page_numbers(1, settings) == [1]


def test_probe_reads_a_real_pdf(sample_pdfs: dict[str, Path], settings: Settings) -> None:
    result = probe_pdf(sample_pdfs["clean_book.pdf"], settings)
    assert result.page_count == 4
    assert result.has_text_layer
    assert result.source_lang == "en"
    assert result.garble_verdict == "clean"
    assert result.has_outline
    assert result.outline_entries == 2
    assert result.recommended_extractor == "pymupdf"
    assert result.columns == 1


def test_probe_detects_two_columns(sample_pdfs: dict[str, Path], settings: Settings) -> None:
    assert probe_pdf(sample_pdfs["two_column.pdf"], settings).columns == 2


def test_probe_reports_which_external_tools_exist(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    """The probe never requires them, but a missing binary should be visible, not mysterious."""
    result = probe_pdf(sample_pdfs["clean_book.pdf"], settings)
    assert set(result.external_tools) >= {"pdftotext", "ocrmypdf", "tesseract", "typst"}
    assert all(isinstance(v, bool) for v in result.external_tools.values())


def test_probe_lists_fonts_and_embedding(sample_pdfs: dict[str, Path], settings: Settings) -> None:
    result = probe_pdf(sample_pdfs["clean_book.pdf"], settings)
    assert result.fonts
    assert all(font.name for font in result.fonts)


def test_probe_report_renders_without_markup_errors(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    import io

    from rich.console import Console

    result = probe_pdf(sample_pdfs["clean_book.pdf"], settings)
    text = render_probe_report(result)
    buffer = io.StringIO()
    Console(file=buffer, width=100).print(text)  # raises if the markup is malformed
    assert "recommended extractor" in buffer.getvalue()


def test_opening_a_non_pdf_says_what_to_do(tmp_path: Path, settings: Settings) -> None:
    bogus = tmp_path / "not-a.pdf"
    bogus.write_text("this is not a pdf", encoding="utf-8")
    with pytest.raises(ExtractionError) as excinfo:
        probe_pdf(bogus, settings)
    assert excinfo.value.remedy
    assert "qpdf" in (excinfo.value.remedy or "")
