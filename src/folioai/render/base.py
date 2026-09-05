"""Shared rendering concerns (brief §14).

Every renderer consumes the translated IR and must handle the same four things: the target
language tag, script-appropriate typography, images carried through from the source, and the
bilingual layouts. Putting those here keeps six renderers from inventing six answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..ir import Block, Document
from ..logging_setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

log = get_logger(__name__)

Layout = Literal["target-only", "bilingual-paragraph", "bilingual-columns", "annotated"]

#: Languages written right to left. Drives `dir=rtl` and EPUB page progression (§14).
RTL_LANGUAGES = frozenset({"ar", "fa", "he", "ur", "yi", "dv", "ps", "ku", "sd"})

#: Script -> the font a renderer must have. Missing fonts fail loudly rather than render
#: boxes: silent tofu in a 400-page book is worse than a failed render (D-51).
SCRIPT_FONTS: dict[str, str] = {
    "latin": "Noto Serif",
    "cyrillic": "Noto Serif",
    "greek": "Noto Serif",
    "cjk": "Noto Serif CJK",
    "hiragana": "Noto Serif CJK",
    "katakana": "Noto Serif CJK",
    "hangul": "Noto Serif CJK",
    "arabic": "Noto Naskh Arabic",
    "persian": "Vazirmatn",
    "hebrew": "Noto Serif Hebrew",
    "devanagari": "Noto Serif Devanagari",
    "thai": "Noto Serif Thai",
}

LANGUAGE_SCRIPTS: dict[str, str] = {
    "ja": "cjk", "zh": "cjk", "ko": "hangul",
    # Persian gets its own entry. The script is Arabic, but Persian set in an Arabic naskh
    # face reads wrong to a Persian reader: the letterforms, and above all the digits,
    # belong to a different typographic tradition.
    "fa": "persian", "ps": "persian",
    "ar": "arabic", "ur": "arabic",
    "he": "hebrew", "yi": "hebrew",
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
    "th": "thai",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic", "sr": "cyrillic",
    "el": "greek",
}  # fmt: skip


def base_language(lang: str | None) -> str:
    """``zh-Hans`` -> ``zh``. Region and script subtags do not change the font."""
    return (lang or "und").strip().lower().split("-")[0]


def is_rtl(lang: str | None) -> bool:
    return base_language(lang) in RTL_LANGUAGES


def script_for(lang: str | None) -> str:
    return LANGUAGE_SCRIPTS.get(base_language(lang), "latin")


def font_for(lang: str | None) -> str:
    """The font family a renderer needs for this language's script."""
    return SCRIPT_FONTS.get(script_for(lang), "Noto Serif")


@dataclass(slots=True)
class RenderContext:
    """Everything a renderer needs beyond the document itself."""

    layout: Layout = "target-only"
    source: Document | None = None
    scores: dict[str, float] = field(default_factory=dict)
    needs_review: set[str] = field(default_factory=set)
    min_score: int = 80
    models: dict[str, str] = field(default_factory=dict)
    include_images: bool = True

    @property
    def bilingual(self) -> bool:
        return self.layout in {"bilingual-paragraph", "bilingual-columns"}

    @property
    def annotated(self) -> bool:
        return self.layout == "annotated"

    def source_text(self, block_id: str) -> str | None:
        """The source text for a block, when a bilingual layout needs it."""
        if self.source is None:
            return None
        block = self.source.block_map().get(block_id)
        return block.text if block else None

    def score(self, block_id: str) -> float | None:
        return self.scores.get(block_id)

    def is_low(self, block_id: str) -> bool:
        score = self.scores.get(block_id)
        return block_id in self.needs_review or (score is not None and score < self.min_score)


@dataclass(slots=True)
class RenderedChapter:
    """One chapter's blocks, for renderers that emit a file or section per chapter."""

    id: str
    title: str
    number: int | None
    blocks: list[Block]


def iter_chapters(document: Document) -> Iterator[RenderedChapter]:
    """Chapters with their blocks resolved, in document order.

    Blocks belonging to no chapter are yielded in a synthetic trailing chapter rather than
    dropped -- the invariant is that every block reaches the output.
    """
    block_map = document.block_map()
    seen: set[str] = set()

    for chapter in document.chapters:
        blocks = [block_map[bid] for bid in chapter.block_ids if bid in block_map]
        seen.update(block.id for block in blocks)
        yield RenderedChapter(
            id=chapter.id, title=chapter.title, number=chapter.number, blocks=blocks
        )

    orphans = [block for block in document.blocks if block.id not in seen]
    if orphans:
        log.warning("orphan_blocks_in_render", count=len(orphans))
        yield RenderedChapter(id="orphans", title="", number=None, blocks=orphans)


def document_metadata(document: Document, context: RenderContext) -> dict[str, str]:
    """Metadata every format writes: title, author, language, and who translated it."""
    meta = {
        "title": document.title or "Untitled",
        "language": document.target_lang or document.source_lang,
        "source_language": document.source_lang,
    }
    if document.author:
        meta["author"] = document.author
    if context.models:
        # §14 asks the EPUB to carry a translator note naming the models used. Every format
        # gets it: a reader who finds an odd sentence deserves to know what produced it.
        parts = [f"{role}: {model}" for role, model in sorted(context.models.items())]
        meta["translator"] = "Machine translation (" + "; ".join(parts) + ")"
    return meta
