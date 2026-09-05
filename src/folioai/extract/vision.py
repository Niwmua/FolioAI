"""Page-level vision fallback for a handful of bad pages (brief §4.2).

When most of a book extracts cleanly but a few pages come out as garbage -- a scanned insert,
a page with a broken font -- rasterising just those pages and asking a vision model to
transcribe them is far cheaper than OCR-ing the whole book, and far better than shipping
the garbage.

Three rules, all from the brief:

* **Opt-in.** Off unless ``--vision-fallback``: it costs money, and silently spending it
  during what the user thinks is a free extraction would be indefensible.
* **Budget-capped.** Never more than ``extraction.vision_max_pages`` pages.
* **Recorded.** Which pages used it goes into the extraction report, because §4.2 says
  extractors must not be silently mixed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import ExtractionError
from ..llm.client import LLMClient
from ..logging_setup import get_logger
from .base import RawDocument, RawLine, RawSpan
from .pymupdf import open_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import Settings

log = get_logger(__name__)

#: 200 DPI, per §4.2. Enough for a vision model; small enough not to blow the token budget.
RENDER_DPI = 200

TRANSCRIBE_PROMPT = """\
Transcribe the text on this page image exactly as it appears.

Rules:
- Transcribe every word, including headers, footers, page numbers and captions.
- Preserve paragraph breaks as blank lines. Do not merge paragraphs.
- Do not translate. Do not summarise. Do not comment. Do not describe the page.
- Do not correct spelling, punctuation or grammar, even where it is clearly wrong.
- If part of the page is illegible, write [illegible] in its place rather than guessing.

Return only the transcribed text."""


@dataclass(slots=True)
class VisionResult:
    """What the fallback managed to transcribe."""

    pages: dict[int, str] = field(default_factory=dict)
    cost_usd: float = 0.0
    failed: list[int] = field(default_factory=list)

    @property
    def page_numbers(self) -> list[int]:
        return sorted(self.pages)


def rasterize_page(path: Path, page_no: int, *, dpi: int = RENDER_DPI) -> bytes:
    """Render one page to PNG bytes.

    PyMuPDF rather than ``pdftoppm``: the brief suggests the latter, but we already depend
    on PyMuPDF and it needs no external binary (D-10).
    """
    doc = open_pdf(path)
    try:
        if not 1 <= page_no <= doc.page_count:
            raise ExtractionError(
                f"Page {page_no} does not exist in {path.name} ({doc.page_count} page(s)).",
                remedy="Check the page numbers passed to the vision fallback.",
                context={"path": str(path), "page": page_no},
            )
        page = doc.load_page(page_no - 1)
        pixmap = page.get_pixmap(dpi=dpi)
        image: bytes = pixmap.tobytes("png")
        return image
    finally:
        doc.close()


def _data_url(image: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


async def transcribe_pages(
    path: Path,
    page_numbers: list[int],
    client: LLMClient,
    settings: Settings,
) -> VisionResult:
    """Transcribe the given pages with a vision model.

    Raises:
        ExtractionError: if more pages are requested than the configured cap allows. The
            cap is a budget guard, so exceeding it is a refusal rather than a truncation --
            silently doing less than asked would hide the cost decision.
    """
    cap = settings.extraction.vision_max_pages
    if len(page_numbers) > cap:
        raise ExtractionError(
            f"The vision fallback was asked for {len(page_numbers)} pages, but the cap is {cap}.",
            remedy=(
                "Raise extraction.vision_max_pages if you mean to spend that, or use "
                "--extractor ocr if most of the book needs re-reading."
            ),
            context={"requested": len(page_numbers), "cap": cap},
        )

    result = VisionResult()
    for page_no in page_numbers:
        image = rasterize_page(path, page_no)
        try:
            response = await client.complete(
                [
                    {
                        "role": "user",
                        "content": [  # type: ignore[dict-item]
                            {"type": "text", "text": TRANSCRIBE_PROMPT},
                            {"type": "image_url", "image_url": {"url": _data_url(image)}},
                        ],
                    }
                ],
                model=settings.models.role("vision"),
                temperature=0.0,
                purpose="vision-transcribe",
            )
        except Exception as exc:
            # One unreadable page must not cost the other nine their transcription.
            log.warning("vision_page_failed", page=page_no, error=str(exc)[:200])
            result.failed.append(page_no)
            continue

        text = response.text.strip()
        if text:
            result.pages[page_no] = text
            result.cost_usd += response.cost.usd
        else:
            result.failed.append(page_no)

    log.info(
        "vision_fallback_complete",
        transcribed=len(result.pages),
        failed=len(result.failed),
        cost_usd=round(result.cost_usd, 6),
    )
    return result


def apply_transcriptions(raw: RawDocument, result: VisionResult, body_size: float) -> int:
    """Replace the lines of transcribed pages, and record which pages were replaced.

    The replacement lines carry no real font geometry -- a transcription has none -- so they
    are given the document's body size. Heading detection on those pages falls back to
    pattern matching, which the report says out loud rather than leaving to be discovered.
    """
    if not result.pages:
        return 0

    size = body_size or 10.0
    replaced = 0
    for page in raw.pages:
        text = result.pages.get(page.number)
        if text is None:
            continue
        lines: list[RawLine] = []
        for index, line_text in enumerate(text.splitlines()):
            stripped = line_text.strip()
            if not stripped:
                continue
            top = index * (size * 1.2)
            lines.append(
                RawLine(
                    spans=[RawSpan(text=stripped, size=size)],
                    bbox=(0.0, top, size * 0.55 * len(stripped), top + size * 1.2),
                    page=page.number,
                )
            )
        page.lines = lines
        page.extractor = "vision"
        replaced += 1

    raw.fallback_pages.setdefault("vision", []).extend(sorted(result.pages))
    raw.warnings.append(
        f"{replaced} page(s) were transcribed by a vision model: "
        f"{', '.join(str(p) for p in sorted(result.pages))}. Font geometry is unavailable "
        "on those pages, so headings there are detected by pattern only."
    )
    if result.failed:
        raw.warnings.append(
            "The vision fallback could not transcribe page(s): "
            + ", ".join(str(p) for p in result.failed)
        )
    return replaced


def find_bad_pages(raw: RawDocument, settings: Settings, *, max_pages: int) -> list[int]:
    """Pages that look like extraction failed on them, worst first.

    A page is suspect when it produced almost no text while its neighbours produced plenty,
    or when what it produced is mostly replacement characters. Both are cheap to compute and
    neither needs a model.
    """
    if not raw.pages:
        return []

    lengths = {page.number: sum(len(line.text) for line in page.lines) for page in raw.pages}
    non_empty = [value for value in lengths.values() if value > 0]
    if not non_empty:
        return []

    typical = sorted(non_empty)[len(non_empty) // 2]
    threshold = max(typical * 0.15, 40)

    scored: list[tuple[float, int]] = []
    for page in raw.pages:
        text = "\n".join(line.text for line in page.lines)
        length = lengths[page.number]
        replacement_ratio = (text.count("�") / len(text)) if text else 0.0
        if replacement_ratio > settings.probe.garbling_replacement_ratio:
            scored.append((1.0 + replacement_ratio, page.number))
        elif length < threshold:
            scored.append((1.0 - (length / max(typical, 1)), page.number))

    scored.sort(reverse=True)
    chosen = [page_no for _, page_no in scored[:max_pages]]
    if chosen:
        log.info("vision_candidates", pages=chosen, typical_chars=typical)
    return chosen
