"""``pdftotext -layout`` fallback, for multi-column pages PyMuPDF orders badly.

This is the one place an external binary earns its keep (PLAN §2.1): poppler's layout mode
has a column heuristic of its own that sometimes beats ours. The cost is that it returns
plain text -- no font sizes, no real bboxes -- so structure detection downstream drops to
pattern matching. That trade is recorded in the extraction report rather than hidden.

If the binary is absent, ``available()`` says so and names the install command. Nothing here
raises at import time.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ExtractionError
from ..logging_setup import get_logger
from .base import RawDocument, RawLine, RawPage, RawSpan
from .pymupdf import open_pdf

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

INSTALL_HINT = (
    "Install poppler-utils: Windows 'winget install oschwartz10612.Poppler' or "
    "'choco install poppler'; macOS 'brew install poppler'; Debian/Ubuntu "
    "'apt install poppler-utils'. Then make sure pdftotext is on PATH."
)

#: pdftotext -layout pads with spaces; treat one space as this many points when guessing x0.
_POINTS_PER_SPACE = 5.0
_LINE_HEIGHT = 12.0


def find_pdftotext() -> str | None:
    return shutil.which("pdftotext")


class PopplerExtractor:
    """Text extraction via ``pdftotext -layout``."""

    name = "poppler"

    def available(self) -> tuple[bool, str]:
        if find_pdftotext() is None:
            return False, INSTALL_HINT
        return True, ""

    def extract(
        self,
        path: Path,
        settings: Settings,  # noqa: ARG002 - required by the Extractor protocol
        pages: Sequence[int] | None = None,
    ) -> RawDocument:
        binary = find_pdftotext()
        if binary is None:
            raise ExtractionError(
                "The poppler extractor needs the 'pdftotext' binary, which is not on PATH.",
                remedy=INSTALL_HINT,
                context={"extractor": self.name},
            )

        args = [binary, "-layout", "-enc", "UTF-8"]
        if pages:
            args += ["-f", str(min(pages)), "-l", str(max(pages))]
        args += [str(path), "-"]

        try:
            completed = subprocess.run(  # argv list, never a shell string
                args, capture_output=True, timeout=300, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise ExtractionError(
                f"pdftotext timed out after 300s on {path.name}.",
                remedy="Try --extractor pymupdf, or extract a page range with --chapters.",
                context={"path": str(path)},
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ExtractionError(
                f"pdftotext failed on {path.name} (exit {completed.returncode}): {detail}",
                remedy="Try --extractor pymupdf, or check the PDF is not encrypted.",
                context={"path": str(path), "returncode": completed.returncode},
            )

        text = completed.stdout.decode("utf-8", errors="replace")
        raw = self._text_to_document(text, first_page=min(pages) if pages else 1)
        raw.warnings.append(
            "poppler returns plain text: font sizes are unavailable, so heading detection "
            "falls back to pattern matching."
        )
        self._attach_metadata(path, raw)
        return raw

    def _text_to_document(self, text: str, *, first_page: int) -> RawDocument:
        raw = RawDocument(extractor=self.name)
        # pdftotext separates pages with a form feed.
        for offset, page_text in enumerate(text.split("\f")):
            if not page_text.strip() and offset == len(text.split("\f")) - 1:
                continue
            page_no = first_page + offset
            lines: list[RawLine] = []
            for row, line_text in enumerate(page_text.splitlines()):
                if not line_text.strip():
                    continue
                indent = len(line_text) - len(line_text.lstrip(" "))
                x0 = indent * _POINTS_PER_SPACE
                top = row * _LINE_HEIGHT
                stripped = line_text.strip()
                lines.append(
                    RawLine(
                        spans=[RawSpan(text=stripped, size=0.0)],
                        bbox=(x0, top, x0 + len(stripped) * _POINTS_PER_SPACE, top + _LINE_HEIGHT),
                        page=page_no,
                    )
                )
            raw.pages.append(
                RawPage(number=page_no, width=612.0, height=792.0, lines=lines, extractor=self.name)
            )
        return raw

    def _attach_metadata(self, path: Path, raw: RawDocument) -> None:
        """Borrow metadata and the outline from PyMuPDF; poppler's text output has neither."""
        try:
            doc = open_pdf(path)
        except ExtractionError as exc:
            log.warning("metadata_unavailable", path=str(path), error=exc.message)
            return
        try:
            raw.metadata = dict(doc.metadata or {})
            raw.toc = [
                (int(level), str(title), int(page)) for level, title, page in doc.get_toc() or []
            ]
        finally:
            doc.close()
