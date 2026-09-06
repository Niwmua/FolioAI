"""PDF via Typst, falling back to WeasyPrint (brief §14, D-50).

Typst is preferred: fast, a single binary, no TeX install, and real book typography --
running heads, chapter openers on recto pages, hyphenation for the target language.
WeasyPrint is the fallback because it is pip-installable and needs no system toolchain.

Neither is installed on many machines, which makes the *failure* path the common path. It
therefore names both install commands and the font the script needs, rather than rendering
a book of empty boxes (D-51).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import RenderError
from ..ir import Document
from ..logging_setup import get_logger
from .base import RenderContext, document_metadata, font_for, is_rtl, iter_chapters
from .html import inline_to_html, render_html

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

TYPST_HINT = (
    "Install Typst: Windows 'winget install Typst.Typst'; macOS 'brew install typst'; "
    "Linux 'cargo install typst-cli' or your package manager. It is one binary with no "
    "dependencies, so you can also just download it from "
    "https://github.com/typst/typst/releases and drop it in folioai's bin directory "
    "(see 'folioai paths'), or point export.typst_path at it."
)
WEASY_HINT = "Install the render extra for the WeasyPrint fallback: uv sync --extra render"


def find_typst(settings: Settings | None = None) -> Path | None:
    """Locate the Typst binary.

    Three places, in order: the configured path, PATH, and folioai's own bin directory.
    The last one matters because Typst ships as a single binary that people download rather
    than install, and "put it on PATH first" is a poor greeting for someone who just wants
    a PDF.
    """
    from ..paths import bin_dir

    if settings is not None and settings.export.typst_path:
        candidate = Path(settings.export.typst_path).expanduser()
        return candidate if candidate.is_file() else None

    found = shutil.which("typst")
    if found:
        return Path(found)

    for name in ("typst.exe", "typst"):
        candidate = bin_dir() / name
        if candidate.is_file():
            return candidate
    return None


def typst_available(settings: Settings | None = None) -> bool:
    return find_typst(settings) is not None


def font_directories(settings: Settings | None = None) -> list[Path]:
    """Font directories to hand Typst, on top of whatever the system provides.

    A PDF of empty boxes is worse than a failed render (D-51), and on most machines the
    system has no Persian, Arabic or CJK serif face at all. Dropping a font into the fonts
    directory is the whole fix, and needs no installer.
    """
    from ..paths import fonts_dir

    directories: list[Path] = []
    if settings is not None:
        directories.extend(Path(p).expanduser() for p in settings.export.font_paths)
    directories.append(fonts_dir())
    return [d for d in directories if d.is_dir()]


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


#: The IR's closed inline subset, in the order it must be matched.
_INLINE_RE = re.compile(
    r"(?P<strong>\*\*.+?\*\*)"
    r"|(?P<em>(?<!\*)\*(?!\*).+?(?<!\*)\*(?!\*))"
    r"|(?P<code>`[^`]+`)"
    r"|(?P<footnote>\[\^[A-Za-z0-9_-]+\])",
    re.DOTALL,
)


def _escape_typst(text: str) -> str:
    """Escape Typst's markup characters. Prose is content, never syntax."""
    for char in ("\\", "#", "$", "*", "_", "`", "<", ">", "@", "=", "~", "[", "]"):
        text = text.replace(char, "\\" + char)
    return text


def typst_inline(text: str) -> str:
    """Convert the IR's inline markdown subset into Typst markup.

    Escaping the whole string was wrong: it turned ``*emphasis*`` into the five literal
    characters, so every italic and bold in the book printed its own asterisks. Everything
    outside a recognised marker is still escaped, because prose that came out of a PDF and
    through a language model must never be able to become syntax.

    Emphasis is emitted as ``#emph[...]`` and ``#strong[...]``, not as Typst's shorthand
    ``_italic_`` and ``*bold*``. The shorthand only closes at a word boundary, and Persian
    hangs the ezafe straight onto the word it follows -- ``*Monte Cristo*ی`` becomes
    ``_Monte Cristo_ی``, whose closing delimiter Typst does not see, and the compile fails
    with "unclosed delimiter" a thousand pages later. English does the same with
    ``*Hamlet*'s``; Persian just makes it constant, 46 times in five chapters. The function
    form has no adjacency rule to trip over in any script.
    """
    parts: list[str] = []
    cursor = 0
    for match in _INLINE_RE.finditer(text):
        parts.append(_escape_typst(text[cursor : match.start()]))
        if inner := match.group("strong"):
            parts.append(f"#strong[{_escape_typst(inner[2:-2])}]")
        elif inner := match.group("em"):
            parts.append(f"#emph[{_escape_typst(inner[1:-1])}]")
        elif inner := match.group("code"):
            parts.append(f"`{inner[1:-1]}`")
        elif inner := match.group("footnote"):
            parts.append(f"#super[{_escape_typst(inner[2:-1])}]")
        cursor = match.end()
    parts.append(_escape_typst(text[cursor:]))
    return "".join(parts)


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
        "    let page-no = here().page()",
        "    if page-no > 1 {",
        # Filter by page rather than by document order. The header is laid out before the
        # body of its own page, so a heading that opens *this* page is not "before" the
        # header's location -- which left every chapter-opening page carrying the previous
        # chapter's name.
        "      let opened = query(heading.where(level: 1))",
        "        .filter(h => h.location().page() <= page-no)",
        "      let title = if opened.len() > 0 { opened.last().body } else { [] }",
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
            text = typst_inline(block.text)
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
                        f"[#text(size: 8.5pt, fill: gray)[{typst_inline(source)}]]\n"
                    )
                lines.append(f"{text}\n")
    return "\n".join(lines)


def _render_with_typst(
    document: Document,
    path: Path,
    context: RenderContext,
    settings: Settings | None = None,
) -> Path:
    binary = find_typst(settings)
    if binary is None:
        raise RenderError("Typst is not available.", remedy=TYPST_HINT)

    source = build_typst_source(document, context)
    command = [str(binary), "compile"]
    for directory in font_directories(settings):
        command += ["--font-path", str(directory)]

    with tempfile.TemporaryDirectory(prefix="folioai-typst-") as tmp:
        typ = Path(tmp) / "book.typ"
        typ.write_text(source, encoding="utf-8")
        command += [str(typ), str(path)]
        try:
            completed = subprocess.run(  # argv list, never a shell string
                command, capture_output=True, timeout=600, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RenderError(
                f"Typst failed to run: {exc}",
                remedy=TYPST_HINT,
                context={"path": str(path), "typst": str(binary)},
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            font = font_for(document.target_lang or document.source_lang)
            if "unknown font family" in detail.lower() or "font" in detail.lower():
                # The common failure, and the one worth answering precisely: name the font,
                # say where to put it, and never fall back to rendering boxes (D-51).
                from ..paths import fonts_dir

                remedy = (
                    f"The PDF needs the {font!r} font family for this script. Download it "
                    f"and drop the file into {fonts_dir()} -- no installer needed -- or set "
                    "export.font_paths to a directory that has it. Vazirmatn (Persian), "
                    "Noto Naskh Arabic, Noto Serif CJK and Noto Serif are all freely "
                    "available."
                )
            else:
                remedy = (
                    "Typst rejected the generated document. Please report this with the "
                    "book that triggered it."
                )
            raise RenderError(
                f"Typst could not compile the book: {detail[:600]}",
                remedy=remedy,
                context={"path": str(path), "font": font, "typst": str(binary)},
            )

    log.info(
        "pdf_written",
        path=str(path),
        engine="typst",
        size_bytes=path.stat().st_size if path.is_file() else 0,
    )
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
    settings: Settings | None = None,
) -> Path:
    """Render a PDF with Typst, WeasyPrint, or whichever of them exists.

    Raises:
        RenderError: naming both install commands when neither engine is available.
    """
    context = context or RenderContext()
    path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "typst":
        if not typst_available(settings):
            raise RenderError(
                "--pdf-engine typst was requested but Typst could not be found.",
                remedy=TYPST_HINT,
            )
        return _render_with_typst(document, path, context, settings)
    if engine == "weasyprint":
        return _render_with_weasyprint(document, path, context)

    if typst_available(settings):
        return _render_with_typst(document, path, context, settings)
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
    "find_typst",
    "font_directories",
    "inline_to_html",
    "render_pdf",
    "render_txt",
    "typst_available",
    "typst_inline",
    "weasyprint_available",
]
