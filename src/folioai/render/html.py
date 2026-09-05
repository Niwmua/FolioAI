"""HTML: one self-contained file, readable typography, dark mode (brief §14).

This renderer carries more weight than its format suggests. It is the basis of the
``annotated`` layout -- the artefact §14 calls "the one I will actually use to decide
whether to trust a run" -- and the same markup drives the EPUB and the WeasyPrint PDF path.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from ..ir import Block, Document
from .base import RenderContext, document_metadata, font_for, is_rtl, iter_chapters

_EM_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`", re.DOTALL)
_FOOTNOTE_RE = re.compile(r"\[\^([A-Za-z0-9_-]+)\]")


def inline_to_html(text: str) -> str:
    """Render the IR's closed inline subset. Everything else is escaped.

    Escaping first and converting after is deliberate: the IR is prose that has been through
    a language model, and a stray angle bracket in a novel must not become markup.
    """
    escaped = html.escape(text, quote=False)

    def replace(match: re.Match[str]) -> str:
        strong, em, code = match.groups()
        if strong is not None:
            return f"<strong>{strong}</strong>"
        if em is not None:
            return f"<em>{em}</em>"
        return f"<code>{code}</code>"

    converted = _EM_RE.sub(replace, escaped)
    return _FOOTNOTE_RE.sub(r'<sup class="fn"><a href="#fn-\1">\1</a></sup>', converted)


def block_to_html(block: Block, context: RenderContext) -> str:
    """One block, with its source alongside it when a bilingual layout asks for one."""
    body = inline_to_html(block.text)
    classes = [f"b-{block.kind}"]
    attrs = f' id="{block.id}"'

    if context.annotated:
        score = context.score(block.id)
        if context.is_low(block.id):
            classes.append("flagged")
        if score is not None:
            attrs += f' data-score="{score:.1f}"'

    if block.kind == "heading":
        level = min(max(block.level or 1, 1), 6)
        return f'<h{level}{attrs} class="{" ".join(classes)}">{body}</h{level}>'
    if block.kind == "scene_break":
        return f'<hr{attrs} class="scene-break">'
    if block.kind == "page_break":
        return f'<hr{attrs} class="page-break">'
    if block.kind == "footnote":
        label = block.meta.get("label") or block.id
        return (
            f'<p{attrs} class="{" ".join(classes)} footnote" id="fn-{label}">'
            f'<span class="fn-label">{html.escape(str(label))}</span> {body}</p>'
        )

    tag = {"blockquote": "blockquote", "list_item": "li", "verse": "p", "table": "p"}.get(
        block.kind, "p"
    )
    if block.kind == "verse":
        body = body.replace("\n", "<br>\n")

    rendered = f'<{tag}{attrs} class="{" ".join(classes)}">{body}</{tag}>'

    source = context.source_text(block.id) if context.bilingual else None
    if source and block.kind not in {"scene_break", "page_break"}:
        source_lang = context.source.source_lang if context.source else ""
        source_html = f'<p class="source" lang="{source_lang}">{inline_to_html(source)}</p>'
        wrapper = "pair" if context.layout == "bilingual-paragraph" else "pair columns"
        return f'<div class="{wrapper}">{source_html}{rendered}</div>'

    if context.annotated and context.score(block.id) is not None:
        score = context.score(block.id)
        badge = f'<span class="score">{score:.0f}</span>' if score is not None else ""
        return f'<div class="annotated-row">{rendered}{badge}</div>'

    return rendered


STYLE = """
:root {
  --bg: #fdfdfb; --fg: #1c1c1a; --muted: #6b6b66; --rule: #e0ded8;
  --flag: #b23a2e; --flag-bg: #fdf1ef; --accent: #2f5d8a;
  --measure: 34em;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16171a; --fg: #e6e4df; --muted: #9a978f; --rule: #2e3034;
    --flag: #f08a7c; --flag-bg: #2a1d1b; --accent: #7fb0e0;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg);
  font-family: __FONT__, Georgia, "Times New Roman", serif;
  font-size: 18px; line-height: 1.6;
  margin: 0 auto; padding: 3rem 1.5rem 6rem; max-width: var(--measure);
  text-rendering: optimizeLegibility;
}
h1, h2, h3, h4 { line-height: 1.25; margin: 2.5em 0 0.8em; font-weight: 600; }
h1 { font-size: 1.9em; } h2 { font-size: 1.45em; } h3 { font-size: 1.2em; }
p { margin: 0 0 1.1em; }
p + p { text-indent: 1.4em; margin-top: -0.6em; }
blockquote {
  margin: 1.4em 0; padding-inline-start: 1.2em;
  border-inline-start: 3px solid var(--rule); color: var(--muted);
}
.b-verse { white-space: normal; font-style: italic; padding-inline-start: 1.2em; }
hr.scene-break {
  border: 0; margin: 2.5em auto; width: 6em; text-align: center;
}
hr.scene-break::before { content: "* * *"; color: var(--muted); letter-spacing: 0.5em; }
.footnote { font-size: 0.85em; color: var(--muted); text-indent: 0; }
.fn-label { font-weight: 600; margin-inline-end: 0.4em; }
sup.fn a { text-decoration: none; color: var(--accent); }
header.book { border-bottom: 1px solid var(--rule); padding-bottom: 1.5rem; margin-bottom: 2rem; }
header.book .title { font-size: 2.1em; margin: 0 0 0.2em; }
header.book .meta { color: var(--muted); font-size: 0.85em; }

/* bilingual */
.pair { margin-bottom: 1.6em; }
.pair .source {
  color: var(--muted); font-size: 0.86em; margin-bottom: 0.35em; text-indent: 0;
  padding-inline-start: 0.8em; border-inline-start: 2px solid var(--rule);
}
.pair p + p { text-indent: 0; margin-top: 0; }
.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1.6em; align-items: start; }
@media (max-width: 40em) { .columns { grid-template-columns: 1fr; } }

/* annotated */
.annotated-row { display: grid; grid-template-columns: 1fr 3rem; gap: 0.8em; align-items: start; }
.annotated-row .score {
  color: var(--muted); font-size: 0.72em; text-align: end; padding-top: 0.5em;
}
.flagged {
  background: var(--flag-bg); border-inline-start: 3px solid var(--flag);
  padding: 0.5em 0.8em; margin-inline-start: -0.8em;
}
"""


def render_html(document: Document, context: RenderContext | None = None) -> str:
    """Render the whole document as one self-contained HTML file."""
    context = context or RenderContext()
    meta = document_metadata(document, context)
    lang = document.target_lang or document.source_lang
    direction = "rtl" if is_rtl(lang) else "ltr"

    parts: list[str] = [
        "<!doctype html>",
        f'<html lang="{html.escape(lang)}" dir="{direction}">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(meta['title'])}</title>",
    ]
    for key in ("author", "translator"):
        if key in meta:
            parts.append(f'<meta name="{key}" content="{html.escape(meta[key])}">')
    parts.append(f"<style>{STYLE.replace('__FONT__', font_for(lang))}</style>")
    parts.append("</head><body>")

    parts.append('<header class="book">')
    parts.append(f'<h1 class="title">{html.escape(meta["title"])}</h1>')
    byline = []
    if document.author:
        byline.append(html.escape(document.author))
    byline.append(f"{document.source_lang} → {lang}")
    if "translator" in meta:
        byline.append(html.escape(meta["translator"]))
    parts.append(f'<div class="meta">{" · ".join(byline)}</div>')
    parts.append("</header>")

    in_list = False
    for chapter in iter_chapters(document):
        for block in chapter.blocks:
            if block.kind == "list_item" and not in_list:
                parts.append("<ul>")
                in_list = True
            elif block.kind != "list_item" and in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(block_to_html(block, context))
    if in_list:
        parts.append("</ul>")

    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def write_html(document: Document, path: Path, context: RenderContext | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(document, context), encoding="utf-8")
    return path
