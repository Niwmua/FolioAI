"""Markdown renderer: the canonical output, closest to the IR (brief §14).

Markdown is where extraction quality is judged by eye, so this renderer is deliberately
literal -- it adds nothing and hides nothing. What you read here is what the translator
will be sent.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..ir import Block, Document

_FRONT_MATTER_KEYS = ("title", "author", "source_lang", "target_lang")


def _front_matter(doc: Document) -> str:
    data: dict[str, object] = {}
    for key in _FRONT_MATTER_KEYS:
        value = getattr(doc, key, None)
        if value:
            data[key] = value
    data["ir_version"] = doc.ir_version
    data["blocks"] = len(doc.blocks)
    data["chapters"] = len(doc.chapters)
    data["extractor"] = doc.extraction_report.extractor
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{dumped}\n---\n"


def block_to_markdown(block: Block) -> str:
    """Render one block. One block in, one chunk of Markdown out -- always."""
    text = block.text.strip()
    if block.kind == "heading":
        level = min(max(block.level or 1, 1), 6)
        return f"{'#' * level} {text}"
    if block.kind == "blockquote":
        return "\n".join(f"> {line}" for line in text.splitlines() or [""])
    if block.kind == "list_item":
        return f"- {text}"
    if block.kind == "verse":
        # Verse keeps its lineation, which means a hard break at each line (§7).
        return "\n".join(f"{line}  " for line in text.splitlines())
    if block.kind == "scene_break":
        return "* * *"
    if block.kind == "page_break":
        return "---"
    if block.kind == "footnote":
        label = block.meta.get("label") or block.id
        return f"[^{label}]: {text}"
    if block.kind == "figure_caption":
        return f"*{text}*"
    return text


def document_to_markdown(doc: Document, *, front_matter: bool = True) -> str:
    """Render the whole document to a single Markdown string."""
    parts: list[str] = []
    if front_matter:
        parts.append(_front_matter(doc))
    for block in doc.blocks:
        rendered = block_to_markdown(block)
        if rendered.strip():
            parts.append(rendered)
    return "\n\n".join(parts).strip() + "\n"


def write_markdown(doc: Document, path: Path, *, split_chapters: bool = False) -> list[Path]:
    """Write Markdown to disk, optionally one file per chapter. Returns the files written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not split_chapters:
        path.write_text(document_to_markdown(doc), encoding="utf-8")
        return [path]

    written: list[Path] = []
    directory = path if path.suffix == "" else path.parent
    directory.mkdir(parents=True, exist_ok=True)
    block_map = doc.block_map()
    for index, chapter in enumerate(doc.chapters, start=1):
        blocks = [block_map[bid] for bid in chapter.block_ids if bid in block_map]
        body = "\n\n".join(rendered for b in blocks if (rendered := block_to_markdown(b)).strip())
        target = directory / f"{index:02d}-{chapter.id}.md"
        target.write_text(body.strip() + "\n", encoding="utf-8")
        written.append(target)
    return written
