"""Optional ML extractor: marker (brief §4.2).

Marker is an ML PDF-to-Markdown converter with excellent structure recovery. It is slow,
downloads model weights on first use, and pulls in torch -- so it is never a hard dependency
and never chosen automatically. ``--extractor marker`` is the only way to get here.

The output is Markdown, not positioned spans, so everything downstream that depends on font
geometry degrades to pattern matching. That trade goes into the extraction report rather
than being discovered later in a chapter list that looks wrong for no visible reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ExtractionError
from ..logging_setup import get_logger
from .base import RawDocument, RawLine, RawPage, RawSpan

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

INSTALL_HINT = (
    "Install the ml extra: uv sync --extra ml. It pulls in torch and downloads model "
    "weights on first use, so expect a few gigabytes and a slow first run."
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_PAGE_BREAK_RE = re.compile(r"^\s*(?:\{(\d+)\}|-{3,})\s*$")

#: Nominal sizes so heading detection downstream still has something to cluster on.
BODY_SIZE = 10.0
HEADING_SIZES = {1: 20.0, 2: 17.0, 3: 15.0, 4: 13.0, 5: 12.0, 6: 11.0}


class MarkerExtractor:
    """PDF to Markdown via marker, then Markdown back into the raw line model."""

    name = "marker"

    def available(self) -> tuple[bool, str]:
        try:
            import marker  # noqa: F401
        except ImportError:
            return False, INSTALL_HINT
        return True, ""

    def extract(
        self,
        path: Path,
        settings: Settings,  # noqa: ARG002 - required by the Extractor protocol
        pages: Sequence[int] | None = None,
    ) -> RawDocument:
        ok, hint = self.available()
        if not ok:
            raise ExtractionError(
                "The 'marker' extractor is not installed.",
                remedy=hint,
                context={"extractor": self.name},
            )
        markdown = self._run_marker(path, pages)
        raw = self._markdown_to_document(markdown)
        raw.warnings.append(
            "marker returns Markdown, so font sizes and positions are reconstructed rather "
            "than measured: heading levels come from marker's own analysis, and the "
            "running-head and footnote rules have no geometry to work with."
        )
        return raw

    def _run_marker(self, path: Path, pages: Sequence[int] | None) -> str:
        """Call marker's Python API. Kept in one place so the import stays lazy."""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError as exc:
            raise ExtractionError(
                f"marker is installed but its API could not be imported: {exc}",
                remedy=INSTALL_HINT,
                context={"extractor": self.name},
            ) from exc

        config: dict[str, Any] = {"output_format": "markdown"}
        if pages:
            config["page_range"] = f"{min(pages) - 1}-{max(pages) - 1}"

        log.info("marker_start", path=str(path), pages=len(pages) if pages else None)
        try:
            converter = PdfConverter(artifact_dict=create_model_dict(), config=config)
            rendered = converter(str(path))
            text, _, _ = text_from_rendered(rendered)
        except Exception as exc:
            raise ExtractionError(
                f"marker failed on {path.name}: {exc}",
                remedy=(
                    "Try --extractor pymupdf. marker needs a working torch install and "
                    "enough memory to hold its models."
                ),
                context={"path": str(path), "extractor": self.name},
            ) from exc
        log.info("marker_complete", path=str(path), characters=len(text))
        return str(text)

    def _markdown_to_document(self, markdown: str) -> RawDocument:
        """Turn marker's Markdown back into positioned-ish lines.

        Positions are synthetic but *ordered*, which is what reading order needs; sizes are
        nominal but reflect marker's heading levels, which is what structure detection needs.
        """
        raw = RawDocument(extractor=self.name)
        page_no = 1
        lines: list[RawLine] = []
        y = 0.0

        def flush() -> None:
            nonlocal lines, y
            raw.pages.append(
                RawPage(
                    number=page_no,
                    width=612.0,
                    height=max(792.0, y + 40),
                    lines=lines,
                    extractor=self.name,
                )
            )
            lines = []
            y = 0.0

        for line_text in markdown.splitlines():
            stripped = line_text.strip()
            if not stripped:
                continue
            if _PAGE_BREAK_RE.match(stripped):
                if lines:
                    flush()
                    page_no += 1
                continue

            size = BODY_SIZE
            text = stripped
            indent = 0.0
            if heading := _HEADING_RE.match(stripped):
                size = HEADING_SIZES.get(len(heading.group(1)), BODY_SIZE * 1.3)
                text = heading.group(2).strip()
            elif stripped.startswith(">"):
                text = stripped.lstrip("> ").strip()
                indent = 30.0

            y += size * 1.4
            lines.append(
                RawLine(
                    spans=[RawSpan(text=text, size=size, bold=size > BODY_SIZE)],
                    bbox=(45.0 + indent, y, 45.0 + indent + len(text) * size * 0.5, y + size),
                    page=page_no,
                )
            )

        if lines:
            flush()
        if not raw.pages:
            raw.pages.append(RawPage(number=1, width=612.0, height=792.0, extractor=self.name))
        return raw
