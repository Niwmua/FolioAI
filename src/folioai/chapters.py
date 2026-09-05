"""Chapter subsetting: ``--chapters 3-7,12`` (brief §16).

Essential for testing on a real book: translating three chapters of a 400-page novel costs
a few cents and tells you almost everything a full run would about whether the prompt, the
glossary and the extraction are right.

Subsetting *marks* blocks rather than removing them. The IR stays complete, so the output
is still parallel to the source (§21.2) and a later run can widen the selection without
re-extracting or renumbering a single block id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import ConfigError
from .ir import Document
from .logging_setup import get_logger

log = get_logger(__name__)

_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


@dataclass(frozen=True, slots=True)
class ChapterSelection:
    """A parsed ``--chapters`` argument."""

    numbers: frozenset[int]

    def __bool__(self) -> bool:
        return bool(self.numbers)

    def describe(self) -> str:
        return ", ".join(str(n) for n in sorted(self.numbers))


def parse_selection(value: str) -> ChapterSelection:
    """Parse ``3-7,12`` into a set of chapter numbers.

    Raises:
        ConfigError: on anything unparseable, or a reversed range. Silently ignoring a
            malformed selection would translate the wrong part of the book, and the user
            would not find out until the bill arrived.
    """
    numbers: set[int] = set()
    for part in value.split(","):
        if not part.strip():
            continue
        match = _RANGE_RE.match(part)
        if not match:
            raise ConfigError(
                f"Could not understand the chapter selection {part.strip()!r}.",
                remedy="Use numbers and ranges, for example: --chapters 3-7,12",
                context={"selection": value},
            )
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end < start:
            raise ConfigError(
                f"Chapter range {part.strip()!r} runs backwards.",
                remedy="Write ranges low to high, for example: --chapters 3-7",
                context={"selection": value},
            )
        numbers.update(range(start, end + 1))

    if not numbers:
        raise ConfigError(
            "The chapter selection is empty.",
            remedy="Pass something like --chapters 1-3, or omit the flag for the whole book.",
            context={"selection": value},
        )
    return ChapterSelection(numbers=frozenset(numbers))


def apply_selection(document: Document, selection: ChapterSelection) -> int:
    """Mark blocks outside the selection as not-to-translate, in place.

    Chapters are matched on their detected ``number`` where there is one, falling back to
    position -- a book whose chapter numbering could not be parsed should still be
    subsettable, and "the third chapter" is what the user means either way.

    Returns:
        The number of blocks left translatable.
    """
    if not selection:
        return len(document.translatable_blocks())

    wanted: set[str] = set()
    unmatched = set(selection.numbers)
    for position, chapter in enumerate(document.chapters, start=1):
        number = chapter.number if chapter.number is not None else position
        if number in selection.numbers:
            wanted.add(chapter.id)
            unmatched.discard(number)

    if not wanted:
        raise ConfigError(
            f"No chapters matched the selection {selection.describe()}.",
            remedy=(
                f"This book has {len(document.chapters)} chapter(s). Run 'folioai extract' "
                "to see them numbered."
            ),
            context={"selection": selection.describe(), "chapters": len(document.chapters)},
        )
    if unmatched:
        log.warning(
            "chapters_not_found",
            missing=sorted(unmatched),
            available=len(document.chapters),
        )

    kept = 0
    for block in document.blocks:
        inside = block.chapter_id in wanted
        # Never flip a block *on*: kinds like scene breaks are untranslatable for their own
        # reasons, and a selection must not override that.
        if not inside:
            block.translate = False
        elif block.translate:
            kept += 1

    log.info(
        "chapter_selection_applied",
        selected=selection.describe(),
        chapters=len(wanted),
        translatable_blocks=kept,
    )
    return kept


def selection_summary(document: Document, selection: ChapterSelection) -> str:
    """One line describing what a selection will and will not translate."""
    total = len(document.chapters)
    chosen = [
        chapter
        for position, chapter in enumerate(document.chapters, start=1)
        if (chapter.number if chapter.number is not None else position) in selection.numbers
    ]
    words = sum(
        block.word_count for block in document.blocks if block.chapter_id in {c.id for c in chosen}
    )
    return f"{len(chosen)} of {total} chapters ({selection.describe()}), about {words:,} words"
