"""Structure: headings, chapters, and the front/back matter tagging (brief §4.4).

Signal priority, strongest first:

1. **The PDF outline.** If ``doc.get_toc()`` returns entries, trust them -- a human made
   them, and no heuristic beats that.
2. **Font clustering.** The body size is the modal size; sizes meaningfully above it,
   especially bold or alone on a page, are headings. Levels come from size rank, then the
   hierarchy is sanity-checked so an H3 cannot appear before an H1.
3. **Patterns.** ``Chapter N``, ``CHAPTER ONE``, a bare numeral on an otherwise empty page.
   Localisable through ``extraction.chapter_patterns``.

Getting chapter boundaries wrong wastes an entire run's budget, so the proposal is shown
for confirmation (unless ``--yes``) with anomalies highlighted rather than as a wall of
titles nobody reads (PLAN §2.8).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .extract.clean import Paragraph
from .ir import BlockKind, MatterKind, make_chapter_id
from .logging_setup import get_logger

if TYPE_CHECKING:
    from .config import Settings
    from .extract.base import RawDocument

log = get_logger(__name__)

StructureSource = Literal["outline", "font-clustering", "patterns", "none"]

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}  # fmt: skip

_MATTER_PATTERNS: list[tuple[MatterKind, re.Pattern[str]]] = [
    ("toc", re.compile(r"^\s*(table of )?contents\s*$|^\s*inhalt|^\s*sommaire|^\s*índice", re.I)),
    ("copyright", re.compile(r"copyright|all rights reserved|isbn|first published", re.I)),
    ("dedication", re.compile(r"^\s*(for|to)\s+\w+[\w\s,'.-]{0,40}$", re.I)),
    ("index", re.compile(r"^\s*index\s*$", re.I)),
    ("colophon", re.compile(r"colophon|printed (in|by)|typeset in", re.I)),
]

_QUOTE_OPENERS = ('"', "“", "«", "„", "‘", "'", "—", "–", "「", "『")


@dataclass(slots=True)
class ChapterPlan:
    """A proposed chapter, before it becomes an IR chapter."""

    id: str
    title: str
    number: int | None
    level: int
    start_index: int
    start_page: int | None
    paragraph_indices: list[int] = field(default_factory=list)

    def word_count(self, paragraphs: list[Paragraph]) -> int:
        return sum(len(paragraphs[i].text.split()) for i in self.paragraph_indices)


@dataclass(slots=True)
class StructurePlan:
    """The full proposal: per-paragraph kinds, levels, matter tags, and chapters."""

    kinds: list[BlockKind]
    levels: list[int | None]
    matter: list[MatterKind]
    chapters: list[ChapterPlan]
    source: StructureSource
    warnings: list[str] = field(default_factory=list)

    def anomalies(self, paragraphs: list[Paragraph]) -> dict[str, list[str]]:
        """Chapters worth a second look before spending money on them."""
        found: dict[str, list[str]] = {"short": [], "long": [], "empty": []}
        if not self.chapters:
            return found
        counts = [ch.word_count(paragraphs) for ch in self.chapters]
        median = statistics.median(counts) if counts else 0
        for chapter, count in zip(self.chapters, counts, strict=True):
            if count == 0:
                found["empty"].append(chapter.title)
            elif count < 200:
                found["short"].append(f"{chapter.title} ({count} words)")
            elif median and count > median * 3:
                found["long"].append(f"{chapter.title} ({count} words)")
        return found


def roman_to_int(text: str) -> int | None:
    lowered = text.strip().lower()
    if not lowered or any(c not in _ROMAN_VALUES for c in lowered):
        return None
    total = 0
    previous = 0
    for char in reversed(lowered):
        value = _ROMAN_VALUES[char]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total or None


def parse_chapter_number(title: str) -> int | None:
    """Pull a chapter number out of a heading, in digits, romans, or English words."""
    lowered = title.strip().lower()
    digits = re.search(r"\b(\d{1,3})\b", lowered)
    if digits:
        return int(digits.group(1))
    words = re.search(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lowered)
    if words:
        return _NUMBER_WORDS[words.group(1)]
    roman = re.search(r"\b([ivxlcdm]{1,7})\b", lowered)
    if roman:
        return roman_to_int(roman.group(1))
    return None


def matches_chapter_pattern(text: str, settings: Settings) -> bool:
    stripped = text.strip()
    if len(stripped) > 80:
        return False
    return any(
        re.search(pattern, stripped, re.IGNORECASE)
        for pattern in settings.extraction.chapter_patterns
    )


def matches_scene_break(text: str, settings: Settings) -> bool:
    stripped = text.strip()
    if len(stripped) > 20:
        return False
    return any(re.match(pattern, stripped) for pattern in settings.extraction.scene_break_patterns)


def _looks_like_verse(para: Paragraph, median_line_width: float) -> bool:
    """Three or more short, left-aligned lines in a column whose lines are normally longer.

    The comparison is against the typical line width *of this paragraph's column*. Compared
    against the paragraph itself, every narrow-measure book looks like verse; compared
    against the whole page, every left column of a two-column layout does.
    """
    if len(para.lines) < 3 or median_line_width <= 0:
        return False
    short = sum(1 for line in para.lines if line.x1 < median_line_width * 0.75)
    if short < len(para.lines) * 0.8:
        return False
    starts = [line.x0 for line in para.lines]
    return max(starts) - min(starts) < 20.0


def classify_paragraph(
    para: Paragraph,
    *,
    body_size: float,
    median_x0: float,
    median_line_width: float,
    settings: Settings,
) -> tuple[BlockKind, int | None]:
    """Assign a block kind (and heading level placeholder) to one paragraph."""
    text = para.text.strip()
    if para.is_footnote:
        return "footnote", None
    if matches_scene_break(text, settings):
        return "scene_break", None
    if re.match(r"^\s*(fig(?:ure)?\.?|table|plate)\s*\d+", text, re.IGNORECASE):
        return "figure_caption", None
    if re.match(r"^\s*([•‣◦⁃∙]|\(?[a-z0-9]{1,3}[.)])\s+", text):
        return "list_item", None

    is_big = body_size > 0 and para.size >= body_size * settings.extraction.min_heading_size_ratio
    short_and_unterminated = len(text) <= 90 and not text.rstrip().endswith((".", "!", "?", ";"))
    if is_big and short_and_unterminated:
        return "heading", 1
    if matches_chapter_pattern(text, settings) and short_and_unterminated:
        return "heading", 1
    if para.bold and short_and_unterminated and len(text) <= 70 and body_size > 0:
        return "heading", 2

    if _looks_like_verse(para, median_line_width):
        return "verse", None
    if median_x0 and para.x0 > median_x0 + 18 and para.size <= body_size:
        return "blockquote", None
    if text.startswith(_QUOTE_OPENERS):
        return "dialogue", None
    return "paragraph", None


def assign_heading_levels(
    paragraphs: list[Paragraph], kinds: list[BlockKind], levels: list[int | None]
) -> list[str]:
    """Rank heading font sizes into levels and repair impossible hierarchies.

    Returns any warnings raised while repairing, so the report can show them.
    """
    warnings: list[str] = []
    sizes = sorted(
        {round(paragraphs[i].size, 1) for i, kind in enumerate(kinds) if kind == "heading"},
        reverse=True,
    )
    if not sizes:
        return warnings

    rank = {size: index + 1 for index, size in enumerate(sizes[:6])}
    previous_level = 0
    for index, kind in enumerate(kinds):
        if kind != "heading":
            continue
        level = rank.get(round(paragraphs[index].size, 1), len(rank))
        # No H3 before an H1: a level may only descend one step at a time.
        if previous_level and level > previous_level + 1:
            warnings.append(
                f"heading level jump ({previous_level} -> {level}) repaired at "
                f"{paragraphs[index].text[:40]!r}"
            )
            level = previous_level + 1
        levels[index] = level
        previous_level = level
    return warnings


def _chapters_from_outline(
    paragraphs: list[Paragraph], raw: RawDocument, kinds: list[BlockKind]
) -> list[ChapterPlan] | None:
    """Map outline entries onto paragraphs by page, then by heading proximity."""
    if not raw.toc:
        return None
    chapters: list[ChapterPlan] = []
    used: set[int] = set()
    for level, title, page in raw.toc:
        if level > 2:  # sections below chapter level do not start a chapter
            continue
        candidates = [
            i
            for i, para in enumerate(paragraphs)
            if i not in used and para.pages and para.pages[0] >= page
        ]
        if not candidates:
            continue
        # Prefer a heading on the target page whose text matches the outline entry.
        target = None
        normalized = re.sub(r"\W+", " ", title).strip().lower()
        for index in candidates[:20]:
            para_text = re.sub(r"\W+", " ", paragraphs[index].text).strip().lower()
            if para_text and (para_text.startswith(normalized) or normalized.startswith(para_text)):
                target = index
                break
        if target is None:
            target = candidates[0]
        used.add(target)
        kinds[target] = "heading"
        chapters.append(
            ChapterPlan(
                id=make_chapter_id(len(chapters) + 1),
                title=title.strip(),
                number=parse_chapter_number(title),
                level=level,
                start_index=target,
                start_page=page,
            )
        )
    if not chapters:
        return None
    chapters.sort(key=lambda c: c.start_index)
    for ordinal, chapter in enumerate(chapters, start=1):
        chapter.id = make_chapter_id(ordinal)
    return chapters


def _chapters_from_headings(
    paragraphs: list[Paragraph], kinds: list[BlockKind], levels: list[int | None]
) -> list[ChapterPlan]:
    top = min((lvl for lvl in levels if lvl is not None), default=1)
    chapters: list[ChapterPlan] = []
    for index, kind in enumerate(kinds):
        if kind != "heading" or levels[index] != top:
            continue
        title = paragraphs[index].text.strip()
        chapters.append(
            ChapterPlan(
                id=make_chapter_id(len(chapters) + 1),
                title=title,
                number=parse_chapter_number(title),
                level=top,
                start_index=index,
                start_page=paragraphs[index].pages[0] if paragraphs[index].pages else None,
            )
        )
    return chapters


def classify_matter(
    paragraphs: list[Paragraph], chapters: list[ChapterPlan], kinds: list[BlockKind]
) -> list[MatterKind]:
    """Best-effort front/back matter tagging (§4.3.8). Tags only; never drops (D-18)."""
    matter: list[MatterKind] = ["body"] * len(paragraphs)
    first_chapter = chapters[0].start_index if chapters else 0
    in_toc = False

    for index, para in enumerate(paragraphs):
        text = para.text.strip()
        tagged: MatterKind | None = None
        for kind, pattern in _MATTER_PATTERNS:
            if pattern.search(text):
                tagged = kind
                break

        if tagged == "toc":
            in_toc = True
        elif in_toc and (kinds[index] == "heading" or index >= first_chapter):
            in_toc = False

        if tagged is not None:
            matter[index] = tagged
        elif in_toc:
            matter[index] = "toc"
        elif index < first_chapter:
            matter[index] = "cover" if index < 3 else "body"
    return matter


def detect_structure(
    paragraphs: list[Paragraph], raw: RawDocument, settings: Settings, body_size: float
) -> StructurePlan:
    """Produce the full structural proposal for a cleaned document."""
    widths_by_column: dict[int, list[float]] = {}
    for para in paragraphs:
        for line in para.lines:
            if line.x1 > 0:
                widths_by_column.setdefault(para.column, []).append(line.x1)
    median_width_by_column = {
        column: statistics.median(values) for column, values in widths_by_column.items()
    }
    # Indentation only means something against the other paragraphs in the same column:
    # in a two-column layout every right-column paragraph is "indented" relative to the
    # page, and would otherwise be classified as a block quote.
    per_column: dict[int, list[float]] = {}
    for para in paragraphs:
        if para.text:
            per_column.setdefault(para.column, []).append(para.x0)
    median_x0_by_column = {
        column: statistics.median(values) for column, values in per_column.items()
    }

    kinds: list[BlockKind] = []
    levels: list[int | None] = []
    for para in paragraphs:
        kind, level = classify_paragraph(
            para,
            body_size=body_size,
            median_x0=median_x0_by_column.get(para.column, 0.0),
            median_line_width=median_width_by_column.get(para.column, 0.0),
            settings=settings,
        )
        kinds.append(kind)
        levels.append(level)

    warnings = assign_heading_levels(paragraphs, kinds, levels)

    chapters = _chapters_from_outline(paragraphs, raw, kinds)
    source: StructureSource = "outline"
    if chapters is None:
        # Re-level, because the outline pass may have promoted paragraphs to headings.
        warnings += assign_heading_levels(paragraphs, kinds, levels)
        chapters = _chapters_from_headings(paragraphs, kinds, levels)
        source = "font-clustering" if chapters else "none"
        if not chapters:
            pattern_hits = [
                index
                for index, para in enumerate(paragraphs)
                if matches_chapter_pattern(para.text, settings)
            ]
            if pattern_hits:
                source = "patterns"
                for ordinal, index in enumerate(pattern_hits, start=1):
                    kinds[index] = "heading"
                    levels[index] = 1
                    title = paragraphs[index].text.strip()
                    chapters.append(
                        ChapterPlan(
                            id=make_chapter_id(ordinal),
                            title=title,
                            number=parse_chapter_number(title),
                            level=1,
                            start_index=index,
                            start_page=paragraphs[index].pages[0]
                            if paragraphs[index].pages
                            else None,
                        )
                    )
    else:
        for index, chapter in enumerate(chapters):
            levels[chapter.start_index] = 1 if chapter.level <= 1 else chapter.level
            del index

    # Everything before the first chapter becomes front matter, kept as its own chapter so
    # no block is ever orphaned.
    if chapters and chapters[0].start_index > 0:
        chapters.insert(
            0,
            ChapterPlan(
                id="ch00",
                title="Front matter",
                number=None,
                level=1,
                start_index=0,
                start_page=paragraphs[0].pages[0] if paragraphs[0].pages else None,
            ),
        )
    elif not chapters and paragraphs:
        chapters = [
            ChapterPlan(
                id=make_chapter_id(1),
                title="(untitled)",
                number=None,
                level=1,
                start_index=0,
                start_page=paragraphs[0].pages[0] if paragraphs[0].pages else None,
            )
        ]
        warnings.append(
            "No chapter structure found: the whole book is one chapter. Check the source "
            "PDF has an outline or recognisable chapter headings."
        )

    # Assign every paragraph to exactly one chapter -- the invariant the IR depends on.
    boundaries = [chapter.start_index for chapter in chapters]
    for position, chapter in enumerate(chapters):
        end = boundaries[position + 1] if position + 1 < len(boundaries) else len(paragraphs)
        chapter.paragraph_indices = list(range(chapter.start_index, end))

    matter = classify_matter(paragraphs, chapters, kinds)
    assigned = sum(len(chapter.paragraph_indices) for chapter in chapters)
    if assigned != len(paragraphs):  # pragma: no cover - guarded by construction
        warnings.append(f"chapter assignment covered {assigned} of {len(paragraphs)} paragraphs")

    log.info(
        "structure_detected",
        source=source,
        chapters=len(chapters),
        headings=sum(1 for k in kinds if k == "heading"),
        warnings=len(warnings),
    )
    return StructurePlan(
        kinds=kinds,
        levels=levels,
        matter=matter,
        chapters=chapters,
        source=source,
        warnings=warnings,
    )


def kind_histogram(kinds: list[BlockKind]) -> Counter[str]:
    return Counter(kinds)
