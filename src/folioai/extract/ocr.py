"""OCR path: add a text layer with ocrmypdf, then extract normally.

``ocrmypdf`` is used rather than raw Tesseract because it handles deskew, rotation,
language packs and PDF/A output, and writes a real text layer back into a copy of the PDF --
which means every downstream stage keeps working unchanged, including the geometry that
heading detection depends on.

The OCR'd file is cached beside the job so a re-run does not pay for OCR twice.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ExtractionError
from ..logging_setup import get_logger
from .base import RawDocument
from .pymupdf import PyMuPDFExtractor

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

INSTALL_HINT = (
    "Install ocrmypdf and Tesseract with the language packs you need: "
    "Windows 'winget install ocrmypdf' plus 'winget install UB-Mannheim.TesseractOCR'; "
    "macOS 'brew install ocrmypdf tesseract-lang'; Debian/Ubuntu "
    "'apt install ocrmypdf tesseract-ocr-<lang>'."
)


class OCRExtractor:
    """Rasterised PDFs: OCR into a text layer, then run the normal PyMuPDF path."""

    name = "ocr"

    def __init__(self, workdir: Path | None = None) -> None:
        self.workdir = workdir

    def available(self) -> tuple[bool, str]:
        if shutil.which("ocrmypdf") is None:
            return False, INSTALL_HINT
        return True, ""

    def extract(
        self, path: Path, settings: Settings, pages: Sequence[int] | None = None
    ) -> RawDocument:
        ok, hint = self.available()
        if not ok:
            raise ExtractionError(
                f"{path.name} has no usable text layer, so it needs OCR, but ocrmypdf is "
                "not installed.",
                remedy=hint,
                context={"path": str(path)},
            )
        lang = settings.extraction.ocr_lang
        if not lang:
            raise ExtractionError(
                "OCR needs to know the source language of the scan.",
                remedy=(
                    "Pass --ocr-lang with a Tesseract language code, for example "
                    "--ocr-lang eng, --ocr-lang deu, or --ocr-lang eng+fra for mixed text."
                ),
                context={"path": str(path)},
            )

        target_dir = self.workdir or path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        ocred = target_dir / f"{path.stem}.ocr.pdf"

        if ocred.is_file() and ocred.stat().st_mtime >= path.stat().st_mtime:
            log.info("ocr_cache_hit", path=str(ocred))
        else:
            self._run_ocrmypdf(path, ocred, lang)

        raw = PyMuPDFExtractor().extract(ocred, settings, pages)
        raw.extractor = self.name
        raw.warnings.append(
            f"Text layer produced by OCR ({lang}); expect occasional character errors that "
            "no amount of translation quality control can recover."
        )
        for page in raw.pages:
            page.extractor = self.name
        return raw

    def _run_ocrmypdf(self, source: Path, target: Path, lang: str) -> None:
        args = [
            "ocrmypdf",
            "--language",
            lang,
            "--deskew",
            "--rotate-pages",
            "--optimize",
            "1",
            "--quiet",
            str(source),
            str(target),
        ]
        log.info("ocr_start", source=str(source), target=str(target), lang=lang)
        try:
            completed = subprocess.run(  # argv list, never a shell string
                args, capture_output=True, timeout=3600, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise ExtractionError(
                f"OCR timed out after an hour on {source.name}.",
                remedy="Try OCR-ing a page range first, or run ocrmypdf yourself and pass "
                "the result to folioai.",
                context={"path": str(source)},
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()[:400]
            raise ExtractionError(
                f"ocrmypdf failed on {source.name} (exit {completed.returncode}): {detail}",
                remedy=(
                    "A missing language pack is the usual cause. Check 'tesseract "
                    "--list-langs' includes the code you passed to --ocr-lang."
                ),
                context={"path": str(source), "returncode": completed.returncode},
            )
        log.info("ocr_complete", target=str(target))
