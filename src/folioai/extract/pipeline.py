"""Extraction pipeline: probe -> extract -> clean -> structure -> IR.

One entry point, ``extract_document``, so the CLI, the tests and later the translation
command all take exactly the same path to an IR. Costs nothing and makes no network calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.table import Table

from ..errors import ExtractionError
from ..ir import (
    IR_VERSION,
    Block,
    BlockKind,
    Chapter,
    Document,
    ExtractionReport,
    ImageRef,
    make_block_id,
)
from ..logging_setup import get_logger
from ..structure import StructurePlan, detect_structure
from .base import Extractor, RawDocument
from .clean import CleaningAudit, Paragraph, clean_document
from .ocr import OCRExtractor
from .poppler import PopplerExtractor
from .probe import ProbeResult, probe_pdf
from .pymupdf import PyMuPDFExtractor

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

#: Kinds that carry no prose worth paying to translate.
_UNTRANSLATABLE_KINDS: frozenset[str] = frozenset({"scene_break", "page_break"})


def build_extractor(name: str, *, workdir: Path | None = None) -> Extractor:
    """Instantiate an extractor by name, or explain why it cannot be used."""
    if name == "pymupdf":
        return PyMuPDFExtractor()
    if name == "poppler":
        return PopplerExtractor()
    if name == "ocr":
        return OCRExtractor(workdir=workdir)
    if name == "marker":
        raise ExtractionError(
            "The 'marker' extractor is not available in this build.",
            remedy=(
                "It arrives in milestone 8 behind the 'ml' extra. Use --extractor pymupdf, "
                "or --extractor poppler for stubborn multi-column layouts."
            ),
            context={"extractor": name},
        )
    raise ExtractionError(
        f"Unknown extractor: {name!r}.",
        remedy="Choose one of: auto, pymupdf, poppler, ocr, marker.",
        context={"extractor": name},
    )


def select_extractor(
    probe: ProbeResult, settings: Settings, *, workdir: Path | None = None
) -> tuple[Extractor, str]:
    """Pick an extractor from the probe, honouring an explicit ``--extractor`` override.

    Returns the extractor and the reason, which goes into the extraction report so the
    choice is auditable rather than magic.
    """
    requested = settings.extraction.extractor
    if requested != "auto":
        extractor = build_extractor(requested, workdir=workdir)
        ok, hint = extractor.available()
        if not ok:
            raise ExtractionError(
                f"The '{requested}' extractor was requested but is not available here.",
                remedy=hint,
                context={"extractor": requested},
            )
        return extractor, f"explicitly requested with --extractor {requested}"

    chosen = build_extractor(probe.recommended_extractor, workdir=workdir)
    ok, hint = chosen.available()
    if ok:
        return chosen, probe.recommendation_reason

    # The recommendation is unusable here. Fall back rather than dying, but say so loudly.
    fallback = PyMuPDFExtractor()
    log.warning(
        "extractor_unavailable",
        recommended=probe.recommended_extractor,
        falling_back_to=fallback.name,
    )
    if probe.recommended_extractor == "ocr":
        raise ExtractionError(
            f"{Path(probe.path).name} needs OCR ({probe.recommendation_reason}) but ocrmypdf "
            "is not installed, and extracting it without OCR would produce nothing usable.",
            remedy=hint,
            context={"path": probe.path},
        )
    return fallback, (
        f"{probe.recommended_extractor} was recommended but is unavailable ({hint}); "
        "fell back to pymupdf"
    )


@dataclass(slots=True)
class ExtractionResult:
    document: Document
    probe: ProbeResult
    audit: CleaningAudit
    plan: StructurePlan
    extractor_reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary_table(self) -> Table:
        """Compact, diffable summary of what extraction produced (PLAN §2.8)."""
        doc = self.document
        table = Table(title=f"{doc.title or Path(self.probe.path).name}", title_style="heading")
        table.add_column("chapter", no_wrap=True)
        table.add_column("title")
        table.add_column("page", justify="right", no_wrap=True)
        table.add_column("blocks", justify="right", no_wrap=True)
        table.add_column("words", justify="right", no_wrap=True)
        table.add_column("note", no_wrap=True)

        anomalies = self.plan.anomalies(
            [Paragraph(lines=[], text=b.text) for b in doc.blocks]  # word counts only
        )
        flagged = {name.split(" (")[0] for names in anomalies.values() for name in names}
        for chapter in doc.chapters:
            blocks = [b for b in doc.blocks if b.chapter_id == chapter.id]
            words = sum(b.word_count for b in blocks)
            note = ""
            if chapter.title in flagged or words < 200:
                note = "[warn]check[/warn]"
            table.add_row(
                chapter.id,
                chapter.title[:60],
                str(chapter.start_page or "—"),
                str(len(blocks)),
                f"{words:,}",
                note,
            )
        return table


def _chapter_for_page(page: int, chapters: list[Chapter]) -> str | None:
    """Chapter whose start page is the last one at or before this page."""
    best: str | None = None
    for chapter in chapters:
        if chapter.start_page is not None and chapter.start_page <= page:
            best = chapter.id
    return best or (chapters[0].id if chapters else None)


def build_document(
    paragraphs: list[Paragraph],
    footnotes: list[Paragraph],
    plan: StructurePlan,
    raw: RawDocument,
    probe: ProbeResult,
    report: ExtractionReport,
) -> Document:
    """Assemble the IR. Every paragraph becomes exactly one block -- no exceptions."""
    blocks: list[Block] = []
    chapters: list[Chapter] = []
    ordinal = 0
    index_to_chapter: dict[int, str] = {}

    for chapter_plan in plan.chapters:
        chapters.append(
            Chapter(
                id=chapter_plan.id,
                title=chapter_plan.title,
                number=chapter_plan.number,
                level=chapter_plan.level,
                start_page=chapter_plan.start_page,
            )
        )
        for para_index in chapter_plan.paragraph_indices:
            index_to_chapter[para_index] = chapter_plan.id

    chapter_map = {chapter.id: chapter for chapter in chapters}

    for index, para in enumerate(paragraphs):
        kind: BlockKind = plan.kinds[index]
        chapter_id = index_to_chapter.get(index)
        block = Block(
            id=make_block_id(ordinal),
            kind=kind,
            level=plan.levels[index],
            text=para.text,
            chapter_id=chapter_id,
            source_pages=para.pages,
            footnote_refs=para.footnote_refs,
            matter=plan.matter[index],
            translate=kind not in _UNTRANSLATABLE_KINDS,
            meta={"font_size": para.size, "x0": round(para.x0, 1), "bold": para.bold},
        )
        blocks.append(block)
        if chapter_id and chapter_id in chapter_map:
            chapter_map[chapter_id].block_ids.append(block.id)
        ordinal += 1

    # Footnotes go at the end of the chapter they belong to: separated from the sentence
    # they interrupt (§4.3.7), but never detached from their context or dropped.
    for note in footnotes:
        page = note.pages[0] if note.pages else 0
        chapter_id = _chapter_for_page(page, chapters)
        block = Block(
            id=make_block_id(ordinal),
            kind="footnote",
            text=note.text,
            chapter_id=chapter_id,
            source_pages=note.pages,
            matter="body",
            meta={"label": note.footnote_label} if note.footnote_label else {},
        )
        blocks.append(block)
        if chapter_id and chapter_id in chapter_map:
            chapter_map[chapter_id].block_ids.append(block.id)
        ordinal += 1

    images = [
        ImageRef(
            id=image.id,
            page=image.page,
            bbox=image.bbox,
            width=image.width,
            height=image.height,
        )
        for image in raw.images
    ]

    metadata = raw.metadata or {}
    return Document(
        ir_version=IR_VERSION,
        source_lang=probe.source_lang or "und",
        title=(metadata.get("title") or probe.title or None),
        author=(metadata.get("author") or probe.author or None),
        chapters=chapters,
        blocks=blocks,
        images=images,
        extraction_report=report,
    )


def extract_document(
    path: Path,
    settings: Settings,
    *,
    probe_result: ProbeResult | None = None,
    workdir: Path | None = None,
) -> ExtractionResult:
    """Run the full extraction pipeline over a PDF.

    Args:
        path: The source PDF.
        settings: Merged configuration.
        probe_result: A probe already run for this file, to avoid probing twice.
        workdir: Where derived files (an OCR'd PDF) may be written.

    Raises:
        ExtractionError: if the PDF cannot be opened, the chosen extractor is unavailable,
            or extraction produced no text at all.
    """
    started = time.perf_counter()
    probe = probe_result or probe_pdf(path, settings)
    extractor, reason = select_extractor(probe, settings, workdir=workdir)

    log.info("extraction_start", path=str(path), extractor=extractor.name, reason=reason)
    raw = extractor.extract(path, settings)

    if raw.line_count == 0:
        raise ExtractionError(
            f"No text could be extracted from {path.name} with the '{extractor.name}' extractor.",
            remedy=(
                "Run 'folioai probe' on the file. If it reports no text layer, the PDF is a "
                "scan and needs --extractor ocr --ocr-lang <code>."
            ),
            context={"path": str(path), "extractor": extractor.name},
        )

    cleaned = clean_document(raw, settings)
    plan = detect_structure(cleaned.paragraphs, raw, settings, cleaned.body_size)

    report = ExtractionReport(
        extractor=extractor.name,
        page_count=raw.page_count,
        fallback_pages=raw.fallback_pages,
        raw_line_count=raw.line_count,
        block_count=len(cleaned.paragraphs) + len(cleaned.footnotes),
        stripped_headers=cleaned.audit.stripped_headers,
        stripped_page_numbers=cleaned.audit.stripped_page_numbers,
        dehyphenations=cleaned.audit.joined_hyphens,
        drop_caps_repaired=len(cleaned.audit.drop_caps_repaired),
        footnotes_extracted=len(cleaned.footnotes),
        columns_detected=probe.columns,
        structure_source=plan.source,
        warnings=[*raw.warnings, *plan.warnings, *cleaned.audit.warnings],
    )

    document = build_document(cleaned.paragraphs, cleaned.footnotes, plan, raw, probe, report)
    report.duration_s = round(time.perf_counter() - started, 3)

    expected = len(cleaned.paragraphs) + len(cleaned.footnotes)
    if len(document.blocks) != expected:  # pragma: no cover - guarded by construction
        raise ExtractionError(
            f"Internal error: {expected} cleaned paragraphs produced {len(document.blocks)} "
            "IR blocks. Extraction must never lose or invent content.",
            remedy="Please report this with the PDF that triggered it.",
            context={"path": str(path)},
        )

    log.info(
        "extraction_complete",
        path=str(path),
        blocks=len(document.blocks),
        chapters=len(document.chapters),
        seconds=report.duration_s,
    )
    return ExtractionResult(
        document=document,
        probe=probe,
        audit=cleaned.audit,
        plan=plan,
        extractor_reason=reason,
        warnings=report.warnings,
    )
