"""PyMuPDF extraction: the default path for any PDF with a usable text layer.

``page.get_text("dict")`` keeps font size, weight and bbox, which heading detection,
footnote separation and drop-cap repair all need. Reading order is not something PDF
guarantees, so multi-column pages get an explicit column assignment and are sorted
column-then-y rather than trusting the file's block order.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pymupdf

from ..errors import ExtractionError
from ..logging_setup import get_logger
from .base import BBox, RawDocument, RawImage, RawLine, RawPage, RawSpan

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

# PyMuPDF span flag bits (see the "dict" output docs).
_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4


def open_pdf(path: Path) -> pymupdf.Document:
    """Open a PDF, turning every failure into an error that says what to do."""
    try:
        return pymupdf.open(path)
    except Exception as exc:  # pymupdf raises a broad family of errors
        raise ExtractionError(
            f"Could not open {path.name} as a PDF: {exc}",
            remedy=(
                "Check the file is a real, non-encrypted PDF. If it is password protected, "
                "decrypt it first (qpdf --decrypt in.pdf out.pdf)."
            ),
            context={"path": str(path)},
        ) from exc


def _span_to_raw(span: dict[str, Any]) -> RawSpan:
    flags = int(span.get("flags", 0))
    font = str(span.get("font", ""))
    return RawSpan(
        text=str(span.get("text", "")),
        size=float(span.get("size", 0.0)),
        font=font,
        bold=bool(flags & _FLAG_BOLD) or "bold" in font.lower() or "black" in font.lower(),
        italic=bool(flags & _FLAG_ITALIC) or "italic" in font.lower() or "oblique" in font.lower(),
        bbox=tuple(float(v) for v in span.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
    )


def assign_columns(
    lines: Sequence[RawLine],
    page_width: float,
    page_height: float,
    *,
    gap_fraction: float,
    min_share: float,
    margin_fraction: float,
) -> int:
    """Assign each line a column index in place; return the column count.

    Looks for a gutter: a vertical band the text avoids. A gutter counts only if it is wider
    than ``gap_fraction`` of the page, both sides hold at least ``min_share`` of the lines
    and at least three lines each, and both sides span a real part of the page -- otherwise
    an indented block quote, or a right-aligned running head, reads as a second column.

    Lines in the top and bottom margins are excluded from the decision and left in column 0.
    A running head spans the full measure and a centred folio sits directly under the
    gutter; including either one hides the gutter that is actually there.
    """
    for line in lines:
        line.column = 0
    if len(lines) < 6:
        return 1

    margin = page_height * margin_fraction
    body = [line for line in lines if line.top > margin and line.bottom < page_height - margin]
    if len(body) < 6:
        return 1

    starts = sorted(line.x0 for line in body)
    best_gap = 0.0
    best_split = 0.0
    for lower, upper in pairwise(starts):
        gap = upper - lower
        if gap > best_gap:
            best_gap = gap
            best_split = (lower + upper) / 2

    if best_gap < page_width * gap_fraction:
        return 1

    left_lines = [line for line in body if line.x0 < best_split]
    right_lines = [line for line in body if line.x0 >= best_split]
    minority = left_lines if len(left_lines) <= len(right_lines) else right_lines
    if len(minority) < 3 or len(minority) / len(body) < min_share:
        return 1

    body_extent = max(line.bottom for line in body) - min(line.top for line in body)
    minority_extent = max(line.bottom for line in minority) - min(line.top for line in minority)
    if body_extent <= 0 or minority_extent / body_extent < 0.25:
        return 1

    for line in body:
        line.column = 0 if line.x0 < best_split else 1
    return 2


def _sort_reading_order(lines: list[RawLine]) -> list[RawLine]:
    """Column, then vertical position, then horizontal -- never the file's own order."""
    return sorted(lines, key=lambda line: (line.column, round(line.top, 1), round(line.x0, 1)))


class PyMuPDFExtractor:
    """Default extractor. Fast, keeps geometry, no external binaries."""

    name = "pymupdf"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def extract(
        self, path: Path, settings: Settings, pages: Sequence[int] | None = None
    ) -> RawDocument:
        doc = open_pdf(path)
        wanted = set(pages) if pages else None
        raw = RawDocument(extractor=self.name)
        try:
            raw.metadata = dict(doc.metadata or {})
            raw.toc = [
                (int(level), str(title), int(page)) for level, title, page in doc.get_toc() or []
            ]
            for index in range(doc.page_count):
                page_no = index + 1
                if wanted is not None and page_no not in wanted:
                    continue
                page = doc.load_page(index)
                raw.pages.append(self._page_to_raw(page, page_no, settings))
                raw.images.extend(self._page_images(page, page_no))
        finally:
            doc.close()

        if raw.line_count == 0:
            raw.warnings.append("PyMuPDF found no text lines: this PDF probably has no text layer.")
        return raw

    def _page_to_raw(self, page: pymupdf.Page, page_no: int, settings: Settings) -> RawPage:
        data = page.get_text("dict")
        lines: list[RawLine] = []
        for block_index, block in enumerate(data.get("blocks", [])):
            if block.get("type", 0) != 0:  # 1 == image block, handled separately
                continue
            for line in block.get("lines", []):
                spans = [_span_to_raw(span) for span in line.get("spans", [])]
                if not spans or not any(span.text.strip() for span in spans):
                    continue
                bbox: BBox = tuple(float(v) for v in line.get("bbox", (0, 0, 0, 0)))  # type: ignore[assignment]
                lines.append(RawLine(spans=spans, bbox=bbox, page=page_no, block_index=block_index))

        width = float(data.get("width", page.rect.width))
        height = float(data.get("height", page.rect.height))
        columns = assign_columns(
            lines,
            width,
            height,
            gap_fraction=settings.probe.column_gap_fraction,
            min_share=settings.probe.column_min_share,
            margin_fraction=settings.cleaning.furniture_margin_fraction,
        )
        return RawPage(
            number=page_no,
            width=width,
            height=height,
            lines=_sort_reading_order(lines),
            columns=columns,
            extractor=self.name,
        )

    def _page_images(self, page: pymupdf.Page, page_no: int) -> list[RawImage]:
        images: list[RawImage] = []
        try:
            entries = page.get_images(full=True)
        except Exception as exc:  # a damaged xref should not kill extraction
            log.warning("image_enumeration_failed", page=page_no, error=str(exc))
            return images
        for entry in entries:
            xref = int(entry[0])
            try:
                rects = page.get_image_rects(xref)
                bbox = tuple(float(v) for v in rects[0]) if rects else None
            except Exception as exc:
                log.warning("image_rect_failed", page=page_no, xref=xref, error=str(exc))
                bbox = None
            images.append(
                RawImage(
                    id=f"img{page_no:04d}_{xref}",
                    page=page_no,
                    bbox=bbox,  # type: ignore[arg-type]
                    width=int(entry[2]) if len(entry) > 2 else None,
                    height=int(entry[3]) if len(entry) > 3 else None,
                    xref=xref,
                )
            )
        return images
