"""EPUB 3 via ebooklib (brief §14).

One XHTML file per chapter, a real nav TOC, `dc:language` set to the *target* language, a
translator note naming the models used, and `page-progression-direction="rtl"` for RTL
targets. Validated with epubcheck when it is installed, and the failures reported rather
than swallowed.
"""

from __future__ import annotations

import html
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import RenderError
from ..ir import Document
from ..logging_setup import get_logger
from .base import RenderContext, document_metadata, is_rtl, iter_chapters
from .epubcheck import ValidationResult
from .epubcheck import validate_epub as _validate
from .html import STYLE, block_to_html

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

INSTALL_HINT = "Install the render extra: uv sync --extra render"


def _require_ebooklib() -> Any:
    try:
        from ebooklib import epub
    except ImportError as exc:
        raise RenderError(
            "EPUB output needs the 'ebooklib' package, which is not installed.",
            remedy=INSTALL_HINT,
            context={"format": "epub"},
        ) from exc
    return epub


def render_epub(
    document: Document,
    path: Path,
    context: RenderContext | None = None,
    *,
    cover: Path | None = None,
) -> Path:
    """Write an EPUB 3. Returns the path written.

    Raises:
        RenderError: if ebooklib is missing, or the file cannot be written.
    """
    epub = _require_ebooklib()
    context = context or RenderContext()
    meta = document_metadata(document, context)
    lang = document.target_lang or document.source_lang

    book = epub.EpubBook()
    book.set_identifier(f"urn:uuid:{uuid.uuid4()}")
    book.set_title(meta["title"])
    book.set_language(lang)  # dc:language is the TARGET language, per §14
    if document.author:
        book.add_author(document.author)
    if "translator" in meta:
        # The reader who finds an odd sentence deserves to know what produced it.
        book.add_metadata("DC", "contributor", meta["translator"], {"role": "trl"})
    book.add_metadata("DC", "source", f"Translated from {document.source_lang}")

    if is_rtl(lang):
        book.set_direction("rtl")

    style = epub.EpubItem(
        uid="style",
        file_name="style/book.css",
        media_type="text/css",
        content=STYLE.replace("__FONT__", "serif").encode("utf-8"),
    )
    book.add_item(style)

    if cover is not None:
        if not cover.is_file():
            raise RenderError(
                f"Cover image not found: {cover}",
                remedy="Check the path passed to --cover, or omit it.",
                context={"cover": str(cover)},
            )
        book.set_cover("cover" + cover.suffix, cover.read_bytes())

    chapters: list[Any] = []
    for index, chapter in enumerate(iter_chapters(document), start=1):
        if not chapter.blocks:
            continue
        body = "\n".join(block_to_html(block, context) for block in chapter.blocks)
        title = chapter.title or f"Chapter {index}"
        item = epub.EpubHtml(
            title=title,
            file_name=f"chap_{index:03d}.xhtml",
            lang=lang,
            uid=chapter.id,
        )
        # No XML declaration here: ebooklib parses this string and writes its own prolog,
        # and lxml refuses a declaration on str input ("Document is empty").
        item.content = (
            f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{html.escape(lang)}" '
            f'dir="{"rtl" if is_rtl(lang) else "ltr"}">'
            f"<head><title>{html.escape(title)}</title>"
            f'<link rel="stylesheet" href="style/book.css" type="text/css"/></head>'
            f"<body>{body}</body></html>"
        )
        item.add_item(style)
        book.add_item(item)
        chapters.append(item)

    if not chapters:
        raise RenderError(
            "There is nothing to put in the EPUB: the document has no blocks.",
            remedy="Check the extraction step produced an IR with content.",
        )

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        epub.write_epub(str(path), book)
    except Exception as exc:
        raise RenderError(
            f"Could not write the EPUB to {path}: {exc}",
            remedy="Check the directory is writable and the path is not open elsewhere.",
            context={"path": str(path)},
        ) from exc

    log.info("epub_written", path=str(path), chapters=len(chapters), lang=lang)
    return path


def validate_epub(path: Path, settings: Settings | None = None) -> ValidationResult:
    """Validate an EPUB (§14).

    Re-exported from :mod:`folioai.render.epubcheck`, which owns both the built-in
    structural checks and the optional epubcheck integration.
    """
    return _validate(path, settings)
