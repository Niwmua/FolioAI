"""Stage 1: diagnose a PDF before touching it (brief §4.1).

Pure PyMuPDF, no external binaries (PLAN §2.1 / D-10): page count and metadata replace
``pdfinfo``, ``get_page_fonts`` replaces ``pdffonts`` including its embedded-font column,
``get_text`` replaces the ``pdftotext`` sample, and ``get_page_images`` replaces
``pdfimages -list``. External tools are *reported* when present, never required.

The garbling heuristic is script-aware (PLAN §2.2 / D-11): space and vowel statistics only
mean something for alphabetic scripts, so for CJK and abjads the score reports ``unknown``
and the decision falls back to whether a text layer exists at all.
"""

from __future__ import annotations

import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..logging_setup import get_logger
from .base import RawDocument
from .pymupdf import PyMuPDFExtractor, open_pdf

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

ExtractorName = Literal["pymupdf", "poppler", "ocr", "marker"]
GarbleVerdict = Literal["clean", "suspect", "garbled", "unknown"]

#: Scripts where space frequency and vowel distribution are meaningful signals.
_ALPHABETIC_SCRIPTS = {"latin", "cyrillic", "greek"}
_VOWELS = set("aeiouáéíóúàèìòùâêîôûäëïöüåøæœyаэеиоуыюяєіїαεηιουω")


class FontInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    embedded: bool
    encoding: str = ""


class ProbeResult(BaseModel):
    """Everything the probe learned. Stored on the job and printed as a report."""

    model_config = ConfigDict(extra="forbid")

    path: str
    page_count: int
    pdf_version: str = ""
    encrypted: bool = False
    title: str | None = None
    author: str | None = None
    producer: str | None = None

    has_text_layer: bool = False
    sampled_pages: list[int] = Field(default_factory=list)
    sample_chars: int = 0
    fonts: list[FontInfo] = Field(default_factory=list)
    non_embedded_fonts: list[str] = Field(default_factory=list)
    identity_encoded_fonts: list[str] = Field(default_factory=list)

    garble_verdict: GarbleVerdict = "unknown"
    replacement_ratio: float = 0.0
    control_ratio: float = 0.0
    nonword_ratio: float = 0.0
    space_ratio: float | None = None
    vowel_ratio: float | None = None

    source_lang: str | None = None
    lang_confidence: float | None = None
    script: str = "unknown"

    columns: int = 1
    has_outline: bool = False
    outline_entries: int = 0
    image_count: int = 0

    recommended_extractor: ExtractorName = "pymupdf"
    recommendation_reason: str = ""
    external_tools: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def needs_ocr(self) -> bool:
        return self.recommended_extractor == "ocr"


def sample_page_numbers(page_count: int, settings: Settings) -> list[int]:
    """Pages to sample: spread through the body, skipping front matter (D-12)."""
    if page_count <= 0:
        return []
    wanted = min(settings.probe.sample_pages, page_count)
    start = int(page_count * settings.probe.sample_skip_front_fraction)
    span = max(page_count - start, 1)
    picks = sorted({start + int(span * (i + 1) / (wanted + 1)) for i in range(wanted)})
    return [min(max(p, 0), page_count - 1) + 1 for p in picks]  # 1-based


def dominant_script(text: str) -> str:
    """Best-guess script name from Unicode character names, lowercased."""
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        counts[name.split(" ")[0].lower()] += 1
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def score_garbling(text: str, script: str, settings: Settings) -> dict[str, object]:
    """Score how much a text sample looks like real prose rather than extraction sludge.

    Replacement and control characters are script-independent evidence of a broken
    extraction. Space and vowel frequency are only meaningful for alphabetic scripts; for
    everything else they are reported as ``None`` and excluded from the verdict.
    """
    total = len(text)
    if total == 0:
        return {
            "verdict": "unknown",
            "replacement_ratio": 0.0,
            "control_ratio": 0.0,
            "nonword_ratio": 0.0,
            "space_ratio": None,
            "vowel_ratio": None,
        }

    replacement = (text.count("�") + text.count("\x00")) / total
    # Whitespace is categorised Cc (newline, tab) but is not evidence of a broken
    # extraction -- counting it flagged every clean PDF as garbled.
    control = (
        sum(
            1
            for c in text
            if unicodedata.category(c) in {"Cc", "Cf", "Co", "Cn"} and not c.isspace()
        )
        / total
    )
    nonword = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:!?'\"-—–()[]"))
    nonword_ratio = nonword / total

    space_ratio: float | None = None
    vowel_ratio: float | None = None
    alphabetic = script in _ALPHABETIC_SCRIPTS

    if alphabetic:
        # Any whitespace counts: extracted samples are line-broken, and treating a
        # newline as "not a space" understates word separation badly.
        space_ratio = sum(1 for c in text if c.isspace()) / total
        letters = [c for c in text.lower() if c.isalpha()]
        vowel_ratio = (sum(1 for c in letters if c in _VOWELS) / len(letters)) if letters else 0.0

    cfg = settings.probe
    hard_evidence = (
        replacement > cfg.garbling_replacement_ratio
        or control > cfg.garbling_control_ratio
        or nonword_ratio > cfg.garbling_nonword_ratio
    )
    if hard_evidence:
        verdict: GarbleVerdict = "garbled"
    elif not alphabetic:
        # No usable prose heuristic for this script; say so rather than guessing (D-11).
        verdict = "unknown"
    elif space_ratio is not None and vowel_ratio is not None:
        plausible_spacing = 0.08 <= space_ratio <= 0.30
        plausible_vowels = 0.25 <= vowel_ratio <= 0.60
        if plausible_spacing and plausible_vowels:
            verdict = "clean"
        elif plausible_spacing or plausible_vowels:
            verdict = "suspect"
        else:
            verdict = "garbled"
    else:  # pragma: no cover - alphabetic always sets both
        verdict = "unknown"

    return {
        "verdict": verdict,
        "replacement_ratio": replacement,
        "control_ratio": control,
        "nonword_ratio": nonword_ratio,
        "space_ratio": space_ratio,
        "vowel_ratio": vowel_ratio,
    }


def detect_language(text: str) -> tuple[str | None, float | None]:
    """Detect the source language with lingua. Never asks an LLM (brief §4.1)."""
    stripped = text.strip()
    if len(stripped) < 20:
        return None, None
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:  # pragma: no cover - lingua is a hard dependency
        log.warning("lingua_missing")
        return None, None

    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    language = detector.detect_language_of(stripped)
    if language is None:
        return None, None
    confidence = detector.compute_language_confidence(stripped, language)
    return str(language.iso_code_639_1.name).lower(), float(confidence)


def _external_tools() -> dict[str, bool]:
    return {
        tool: shutil.which(tool) is not None
        for tool in ("pdftotext", "ocrmypdf", "tesseract", "typst", "epubcheck")
    }


def probe_pdf(path: Path, settings: Settings) -> ProbeResult:
    """Diagnose a PDF. Costs nothing and never modifies the file."""
    doc = open_pdf(path)
    try:
        page_count = doc.page_count
        metadata = dict(doc.metadata or {})
        result = ProbeResult(
            path=str(path),
            page_count=page_count,
            pdf_version=str(metadata.get("format", "")),
            encrypted=bool(doc.is_encrypted),
            title=(metadata.get("title") or None),
            author=(metadata.get("author") or None),
            producer=(metadata.get("producer") or None),
            external_tools=_external_tools(),
        )
        toc = doc.get_toc() or []
        result.has_outline = bool(toc)
        result.outline_entries = len(toc)

        sampled = sample_page_numbers(page_count, settings)
        result.sampled_pages = sampled

        fonts: dict[str, FontInfo] = {}
        sample_parts: list[str] = []
        image_count = 0
        for page_no in sampled:
            page = doc.load_page(page_no - 1)
            sample_parts.append(page.get_text())
            image_count += len(page.get_images(full=True))
            for entry in doc.get_page_fonts(page_no - 1):
                # (xref, ext, type, basefont, name, encoding)
                basefont = str(entry[3])
                fonts.setdefault(
                    basefont,
                    FontInfo(
                        name=basefont,
                        type=str(entry[2]),
                        embedded=str(entry[1]) not in {"n/a", ""},
                        encoding=str(entry[5]) if len(entry) > 5 else "",
                    ),
                )
        result.image_count = image_count
        result.fonts = sorted(fonts.values(), key=lambda f: f.name)
        result.non_embedded_fonts = [f.name for f in result.fonts if not f.embedded]
        result.identity_encoded_fonts = [
            f.name for f in result.fonts if "identity" in f.encoding.lower() and not f.embedded
        ]
    finally:
        doc.close()

    sample = "\n".join(sample_parts)
    result.sample_chars = len(sample.strip())
    result.has_text_layer = result.sample_chars >= settings.probe.min_chars_for_text_layer

    result.source_lang, result.lang_confidence = detect_language(sample)
    result.script = dominant_script(sample)

    scores = score_garbling(sample, result.script, settings)
    result.garble_verdict = scores["verdict"]  # type: ignore[assignment]
    result.replacement_ratio = float(scores["replacement_ratio"])  # type: ignore[arg-type]
    result.control_ratio = float(scores["control_ratio"])  # type: ignore[arg-type]
    result.nonword_ratio = float(scores["nonword_ratio"])  # type: ignore[arg-type]
    result.space_ratio = scores["space_ratio"]  # type: ignore[assignment]
    result.vowel_ratio = scores["vowel_ratio"]  # type: ignore[assignment]

    result.columns = _detect_columns(path, sampled, settings)
    _recommend(result)
    log.info(
        "probe_complete",
        path=str(path),
        pages=result.page_count,
        text_layer=result.has_text_layer,
        garble=result.garble_verdict,
        lang=result.source_lang,
        columns=result.columns,
        extractor=result.recommended_extractor,
    )
    return result


def _detect_columns(path: Path, sampled: list[int], settings: Settings) -> int:
    """Column count from the sampled pages: the majority verdict across them."""
    if not sampled:
        return 1
    extractor = PyMuPDFExtractor()
    raw: RawDocument = extractor.extract(path, settings, pages=sampled)
    counts = Counter(page.columns for page in raw.pages if page.lines)
    if not counts:
        return 1
    return int(counts.most_common(1)[0][0])


def _recommend(result: ProbeResult) -> None:
    """Pick an extractor from the evidence (brief §4.2 table)."""
    if result.encrypted:
        result.warnings.append("PDF is encrypted; extraction may fail or return nothing.")

    if not result.has_text_layer:
        result.recommended_extractor = "ocr"
        result.recommendation_reason = (
            f"No usable text layer ({result.sample_chars} characters across "
            f"{len(result.sampled_pages)} sampled pages): this looks scanned."
        )
        result.warnings.append("OCR requires ocrmypdf and a language pack; pass --ocr-lang.")
        return

    if result.garble_verdict == "garbled":
        result.recommended_extractor = "ocr"
        result.recommendation_reason = (
            "The text layer extracts as garbage (replacement characters, control characters, "
            "or implausible letter statistics), which usually means broken font encodings."
        )
        if result.identity_encoded_fonts:
            result.warnings.append(
                "Non-embedded Identity-H fonts found: "
                + ", ".join(result.identity_encoded_fonts[:5])
            )
        return

    if result.columns > 1:
        poppler_available = result.external_tools.get("pdftotext", False)
        result.recommended_extractor = "pymupdf"
        result.recommendation_reason = (
            f"{result.columns}-column layout detected; PyMuPDF with column-aware block "
            "sorting handles it"
            + (
                ", and pdftotext -layout is available as a cross-check."
                if poppler_available
                else ". Install poppler for the pdftotext -layout cross-check."
            )
        )
        return

    result.recommended_extractor = "pymupdf"
    result.recommendation_reason = "Clean single-column text layer."
    if result.garble_verdict == "suspect":
        result.warnings.append(
            "Text statistics are borderline; spot-check the extracted Markdown before "
            "spending money on translation."
        )
    if result.non_embedded_fonts:
        result.warnings.append(
            f"{len(result.non_embedded_fonts)} non-embedded font(s); watch for mojibake."
        )


def render_probe_report(result: ProbeResult) -> str:
    """Human-readable probe report for the terminal (Rich markup)."""
    lines: list[str] = []
    name = Path(result.path).name
    lines.append(f"[heading]{name}[/heading]")
    lines.append(f"  pages          {result.page_count}")
    if result.title:
        lines.append(f"  title          {result.title}")
    if result.author:
        lines.append(f"  author         {result.author}")
    if result.producer:
        lines.append(f"  producer       {result.producer}")
    lines.append(f"  pdf version    {result.pdf_version or 'unknown'}")

    layer = "[good]yes[/good]" if result.has_text_layer else "[bad]no (scanned?)[/bad]"
    lines.append(f"  text layer     {layer}  ({result.sample_chars} chars sampled)")
    verdict_style = {
        "clean": "good",
        "suspect": "warn",
        "garbled": "bad",
        "unknown": "muted",
    }[result.garble_verdict]
    detail = f"repl={result.replacement_ratio:.3f} ctrl={result.control_ratio:.3f}"
    if result.space_ratio is not None:
        detail += f" space={result.space_ratio:.3f} vowel={result.vowel_ratio:.3f}"
    else:
        detail += " (no prose heuristic for this script)"
    lines.append(
        f"  text quality   [{verdict_style}]{result.garble_verdict}[/{verdict_style}]  {detail}"
    )

    lang = result.source_lang or "unknown"
    conf = f" ({result.lang_confidence:.0%} confident)" if result.lang_confidence else ""
    lines.append(f"  language       {lang}{conf}  script={result.script}")
    lines.append(f"  columns        {result.columns}")
    outline = (
        f"[good]yes[/good] ({result.outline_entries} entries)"
        if result.has_outline
        else "[warn]no[/warn] (chapters will be inferred)"
    )
    lines.append(f"  outline        {outline}")
    lines.append(f"  images         {result.image_count} on sampled pages")
    lines.append(
        f"  fonts          {len(result.fonts)} ({len(result.non_embedded_fonts)} not embedded)"
    )

    tools = ", ".join(
        f"[good]{k}[/good]" if v else f"[muted]{k}[/muted]"
        for k, v in sorted(result.external_tools.items())
    )
    lines.append(f"  external tools {tools}")
    lines.append("")
    lines.append(f"  [info]recommended extractor:[/info] {result.recommended_extractor}")
    lines.append(f"  [muted]{result.recommendation_reason}[/muted]")
    for warning in result.warnings:
        lines.append(f"  [warn]warning:[/warn] {warning}")
    return "\n".join(lines)
