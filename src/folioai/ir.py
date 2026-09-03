"""The Document intermediate representation (brief §5).

Everything downstream depends on this, so it is versioned, JSON-schema-able, and persisted
next to the job. Any stage can be re-run in isolation from it.

Inline formatting is a deliberately tiny closed subset -- ``*em*``, ``**strong**``,
``` `code` ``` and ``[^fn1]`` -- because arbitrary HTML does not survive a round trip
through a language model.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

IR_VERSION = 1

BlockKind = Literal[
    "heading",
    "paragraph",
    "blockquote",
    "list_item",
    "verse",
    "dialogue",
    "footnote",
    "figure_caption",
    "table",
    "scene_break",
    "page_break",
    "front_matter",
]

MatterKind = Literal["cover", "copyright", "dedication", "toc", "index", "colophon", "body"]

#: Kinds that carry no translatable prose. Kept in the IR so block counts stay honest.
NON_TEXT_KINDS: frozenset[str] = frozenset({"scene_break", "page_break"})

_FOOTNOTE_REF_RE = re.compile(r"\[\^([A-Za-z0-9_-]+)\]")
_EMPHASIS_RE = re.compile(r"(\*\*|\*|`)")


class ImageRef(BaseModel):
    """A figure carried through from the source PDF."""

    model_config = ConfigDict(extra="forbid")

    id: str
    page: int
    bbox: tuple[float, float, float, float] | None = None
    width: int | None = None
    height: int | None = None
    path: str | None = Field(default=None, description="Relative path to the extracted image.")
    caption_block_id: str | None = None


class Block(BaseModel):
    """One translation unit. Blocks are never split across API calls (brief §6)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: BlockKind
    level: int | None = None
    text: str
    chapter_id: str | None = None
    source_pages: list[int] = Field(default_factory=list)
    footnote_refs: list[str] = Field(default_factory=list)
    lang: str | None = None
    translate: bool = True
    matter: MatterKind = "body"
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _no_seg_tags(cls, value: str) -> str:
        """A stray ``<seg>`` in the IR would corrupt the tag protocol downstream (§6)."""
        if "<seg" in value or "</seg>" in value:
            raise ValueError("block text must not contain <seg> tags")
        return value

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def emphasis_counts(self) -> Counter[str]:
        """Tally of inline markers, used by the formatting-fidelity validator."""
        return Counter(_EMPHASIS_RE.findall(self.text))

    def declared_footnote_refs(self) -> list[str]:
        """Footnote refs actually present in the text, as opposed to the declared list."""
        return _FOOTNOTE_REF_RE.findall(self.text)


class Chapter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    number: int | None = None
    level: int = 1
    block_ids: list[str] = Field(default_factory=list)
    start_page: int | None = None


class ExtractionReport(BaseModel):
    """What extraction did, so the quality report can audit it (brief §15)."""

    model_config = ConfigDict(extra="forbid")

    extractor: str
    page_count: int = 0
    fallback_pages: dict[str, list[int]] = Field(
        default_factory=dict, description="extractor name -> pages it handled as a fallback"
    )
    raw_line_count: int = 0
    block_count: int = 0
    stripped_headers: int = 0
    stripped_page_numbers: int = 0
    dehyphenations: int = 0
    drop_caps_repaired: int = 0
    footnotes_extracted: int = 0
    columns_detected: int = 1
    structure_source: Literal["outline", "font-clustering", "patterns", "none"] = "none"
    warnings: list[str] = Field(default_factory=list)
    duration_s: float = 0.0


class Document(BaseModel):
    """The whole book, after extraction or after translation."""

    model_config = ConfigDict(extra="forbid")

    ir_version: int = IR_VERSION
    source_lang: str
    target_lang: str | None = None
    title: str | None = None
    author: str | None = None
    chapters: list[Chapter] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    images: list[ImageRef] = Field(default_factory=list)
    extraction_report: ExtractionReport

    @field_validator("blocks")
    @classmethod
    def _unique_ids(cls, blocks: list[Block]) -> list[Block]:
        seen: set[str] = set()
        for block in blocks:
            if block.id in seen:
                raise ValueError(f"duplicate block id: {block.id}")
            seen.add(block.id)
        return blocks

    # -- lookups ------------------------------------------------------------------

    def block_map(self) -> dict[str, Block]:
        return {block.id: block for block in self.blocks}

    def chapter_map(self) -> dict[str, Chapter]:
        return {chapter.id: chapter for chapter in self.chapters}

    def blocks_for_chapter(self, chapter_id: str) -> list[Block]:
        return [block for block in self.blocks if block.chapter_id == chapter_id]

    def translatable_blocks(self) -> list[Block]:
        return [
            block
            for block in self.blocks
            if block.translate and block.kind not in NON_TEXT_KINDS and block.text.strip()
        ]

    def word_count(self) -> int:
        return sum(block.word_count for block in self.blocks)

    # -- persistence --------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Document:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    # -- the invariant -------------------------------------------------------------

    def assert_parallel_to(self, other: Document) -> None:
        """Assert a translated document matches this one block for block (§21.2).

        Raises:
            ValueError: naming the exact ids that were lost, invented, or reordered.
        """
        mine = [b.id for b in self.blocks]
        theirs = [b.id for b in other.blocks]
        if mine == theirs:
            return
        missing = [bid for bid in mine if bid not in set(theirs)]
        extra = [bid for bid in theirs if bid not in set(mine)]
        detail: list[str] = []
        if missing:
            detail.append(f"{len(missing)} block(s) lost: {', '.join(missing[:10])}")
        if extra:
            detail.append(f"{len(extra)} block(s) invented: {', '.join(extra[:10])}")
        if not detail:
            detail.append("same ids in a different order")
        raise ValueError("translated document is not parallel to the source: " + "; ".join(detail))


def write_json_schema(path: Path) -> None:
    """Write the IR's JSON Schema. Checked into the repo and asserted in tests (D-20)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = Document.model_json_schema()
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_block_id(ordinal: int) -> str:
    """Sequential, stable, sortable block id (D-14)."""
    return f"b{ordinal:04d}"


def make_chapter_id(ordinal: int) -> str:
    return f"ch{ordinal:02d}"
