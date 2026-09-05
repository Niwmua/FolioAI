"""PDF via Typst, falling back to WeasyPrint (brief §14, D-50).

Typst is preferred: fast, a single binary, no TeX install, and real book typography --
running heads, chapter openers on recto pages, hyphenation for the target language.
WeasyPrint is the fallback because it is pip-installable and needs no system toolchain.

Neither is installed on many machines, which makes the *failure* path the common path. It
therefore names both install commands and the font the script needs, rather than rendering
a book of empty boxes (D-51).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..errors import RenderError
from ..ir import Document
from ..logging_setup import get_logger
from .base import RenderContext, document_metadata, font_for, is_rtl, iter_chapters
from .html import inline_to_html, render_html

log = get_logger(__name__)

TYPST_HINT = (
    "Install Typst: Windows 'winget install Typst.Typst'; macOS 'brew install typst'; "
    "or download a single binary from https://github.com/typst/typst/releases"
)
WEASY_HINT = "Install the render extra for the WeasyPrint fallback: uv sync --extra render"


def typst_available() -> bool:
    return shutil.which("typst") is not None


def weasyprint_available() -> bool:
    """Whether WeasyPrint can actually run here, not merely whether it is installed.

    On a machine without its GTK/Pango libraries the import fails *and* prints a multi-line
    installation banner straight to stderr. We handle the failure perfectly well, so that
    banner is noise in the middle of an otherwise clean export: capture it.
    """
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            import weasyprint  # noqa: F401
    except Exception:  # missing system libraries, not just an absent package
        return False
    return True


def _escape_typst(text: str) -> str:
    """Escape Typst's markup characters. Prose is content, never syntax."""
    for char in ("\\", "#", "$", "*", "_", "`", "<", ">", "@", "=", "~"):
        text = text.replace(char, "\\" + char)
    return text


def build_typst_source(document: Document, context: RenderContext) -> str:
    """A Typst document with real book typography (§14)."""
    meta = document_metadata(document, context)
    lang = document.target_lang or document.source_lang
    font = font_for(lang)
    direction = "rtl" if is_rtl(lang) else "ltr"

    lines = [
        f'#set document(title: "{meta["title"]}"'
        + (f', author: "{document.author}"' if document.author else "")
        + ")",
        f'#set text(font: "{font}", size: 11pt, lang: "{lang.split("-")[0]}", dir: {direction})',
        "#set par(justify: true, leading: 0.72em, first-line-indent: 1.4em)",
        "#set page(",
        '  paper: "a5",',
        "  margin: (inside: 2.2cm, outside: 1.8cm, top: 2cm, bottom: 2.2cm),",
        '  numbering: "1",',
        "  header: context {",
        "    let page-no = counter(page).get().first()",
        "    if page-no > 1 {",
        "      let chapters = query(selector(heading.where(level: 1)).before(here()))",
        "      let title = if chapters.len() > 0 { chapters.last().body } else { [] }",
        '      set text(size: 8pt, style: "italic")',
        "      if calc.even(page-no) { align(left, title) } else { align(right, title) }",
        "    }",
        "  },",
        ")",
        "#set heading(numbering: none)",
        "#show heading.where(level: 1): it => [",
        '  #pagebreak(weak: true, to: "odd")',  # chapter openers on recto pages
        "  #v(3cm)",
        '  #set text(size: 17pt, weight: "semibold")',
        "  #block(it.body)",
        "  #v(1.2cm)",
        "]",
        "",
    ]
    if "translator" in meta:
        lines.append(f'#set document(keywords: ("{meta["translator"]}",))')

    for chapter in iter_chapters(document):
        for block in chapter.blocks:
            text = _escape_typst(block.text)
            if block.kind == "heading":
                level = min(max(block.level or 1, 1), 4)
                lines.append(f"{'=' * level} {text}\n")
            elif block.kind == "scene_break":
                lines.append("#align(center)[#v(0.6cm) \\* \\* \\* #v(0.6cm)]\n")
            elif block.kind == "page_break":
                lines.append("#pagebreak()\n")
            elif block.kind == "blockquote":
                lines.append(f"#block(inset: (left: 1.2em))[#emph[{text}]]\n")
            elif block.kind == "verse":
                body = text.replace("\n", " \\\n")
                lines.append(f"#block(inset: (left: 1.2em))[{body}]\n")
            elif block.kind == "footnote":
                lines.append(f"#text(size: 8pt)[{text}]\n")
            else:
                if context.bilingual and (source := context.source_text(block.id)):
                    lines.append(
                        f"#block(inset: (left: 0.8em), stroke: (left: 1pt + gray))"
                        f"[#text(size: 8.5pt, fill: gray)[{_escape_typst(source)}]]\n"
                    )
                lines.append(f"{text}\n")
    return "\n".join(lines)


def _render_with_typst(document: Document, path: Path, context: RenderContext) -> Path:
    source = build_typst_source(document, context)
    with tempfile.TemporaryDirectory(prefix="folioai-typst-") as tmp:
        typ = Path(tmp) / "book.typ"
        typ.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(  # argv list, never a shell string
                ["typst", "compile", str(typ), str(path)],
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RenderError(
                f"Typst failed to run: {exc}",
                remedy=TYPST_HINT,
                context={"path": str(path)},
            ) from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[:500]
        font = font_for(document.target_lang or document.source_lang)
        remedy = (
            f"If the error mentions a missing font, install {font} and try again; "
            "boxes instead of letters would be worse than this failure."
        )
        raise RenderError(
            f"Typst could not compile the book: {detail}",
            remedy=remedy,
            context={"path": str(path), "font": font},
        )
    log.info("pdf_written", path=str(path), engine="typst")
    return path


def _render_with_weasyprint(document: Document, path: Path, context: RenderContext) -> Path:
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from weasyprint import HTML
    except Exception as exc:
        raise RenderError(
            "WeasyPrint is not usable on this machine.",
            remedy=WEASY_HINT,
            context={"error": str(exc)[:200]},
        ) from exc

    markup = render_html(document, context)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        HTML(string=markup).write_pdf(str(path))
    except Exception as exc:
        raise RenderError(
            f"WeasyPrint could not render the PDF: {exc}",
            remedy=("Install Typst for a better result and no system dependencies. " + TYPST_HINT),
            context={"path": str(path)},
        ) from exc
    log.info("pdf_written", path=str(path), engine="weasyprint")
    return path


def render_pdf(
    document: Document,
    path: Path,
    context: RenderContext | None = None,
    *,
    engine: str = "auto",
) -> Path:
    """Render a PDF with Typst, WeasyPrint, or whichever of them exists.

    Raises:
        RenderError: naming both install commands when neither engine is available.
    """
    context = context or RenderContext()
    path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "typst":
        if not typst_available():
            raise RenderError(
                "--pdf-engine typst was requested but typst is not on PATH.",
                remedy=TYPST_HINT,
            )
        return _render_with_typst(document, path, context)
    if engine == "weasyprint":
        return _render_with_weasyprint(document, path, context)

    if typst_available():
        return _render_with_typst(document, path, context)
    if weasyprint_available():
        log.warning("typst_missing_using_weasyprint", remedy=TYPST_HINT)
        return _render_with_weasyprint(document, path, context)

    raise RenderError(
        "PDF output needs a rendering engine, and neither Typst nor WeasyPrint is available.",
        remedy=f"{TYPST_HINT}. Or use the fallback: {WEASY_HINT}",
        context={"format": "pdf"},
    )


def render_txt(document: Document, path: Path, context: RenderContext | None = None) -> Path:
    """Plain text, for diffing (§14)."""
    context = context or RenderContext()
    parts: list[str] = []
    for chapter in iter_chapters(document):
        for block in chapter.blocks:
            if block.kind == "heading":
                parts.append(block.text.upper())
            elif block.kind == "scene_break":
                parts.append("* * *")
            elif context.bilingual and (source := context.source_text(block.id)):
                parts.append(f"[{block.id}] {source}\n{block.text}")
            else:
                parts.append(block.text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
    log.info("txt_written", path=str(path))
    return path


# Re-exported so callers do not need to know which module holds the HTML renderer.
__all__ = [
    "build_typst_source",
    "inline_to_html",
    "render_pdf",
    "render_txt",
    "typst_available",
    "weasyprint_available",
]
