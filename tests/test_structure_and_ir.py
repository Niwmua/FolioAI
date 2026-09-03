"""Structure detection and the IR's invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from folioai.config import Settings
from folioai.extract.base import RawDocument, RawLine, RawSpan
from folioai.extract.clean import Paragraph
from folioai.ir import Block, Chapter, Document, ExtractionReport, write_json_schema
from folioai.structure import (
    assign_heading_levels,
    detect_structure,
    parse_chapter_number,
    roman_to_int,
)

BODY = 11.0


def para(
    text: str,
    *,
    size: float = BODY,
    page: int = 1,
    x0: float = 45.0,
    bold: bool = False,
    x1: float = 375.0,
    lines: int = 1,
) -> Paragraph:
    raw_lines = [
        RawLine(
            spans=[RawSpan(text=text, size=size, bold=bold)],
            bbox=(x0, 100.0 + 15 * i, x1, 112.0 + 15 * i),
            page=page,
        )
        for i in range(lines)
    ]
    return Paragraph(lines=raw_lines, text=text).finalise()


def document(blocks: list[Block], chapters: list[Chapter] | None = None) -> Document:
    return Document(
        source_lang="en",
        chapters=chapters or [],
        blocks=blocks,
        extraction_report=ExtractionReport(extractor="test"),
    )


# -- numbering helpers -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("iv", 4), ("XII", 12), ("MCMXC", 1990), ("i", 1), ("", None), ("hello", None)],
)
def test_roman_numerals(text: str, expected: int | None) -> None:
    assert roman_to_int(text) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Chapter 4", 4),
        ("CHAPTER ONE", 1),
        ("Chapter XII", 12),
        ("Kapitel 7: Der Wärter", 7),
        ("Prologue", None),
    ],
)
def test_chapter_numbers_are_parsed_from_titles(title: str, expected: int | None) -> None:
    assert parse_chapter_number(title) == expected


# -- classification ------------------------------------------------------------------


def test_outline_is_trusted_over_font_size(settings: Settings) -> None:
    paragraphs = [
        para("Chapter One", size=18.0),
        para("Body text of the first chapter goes here.", lines=3),
        para("Chapter Two", size=18.0),
        para("Body text of the second chapter goes here.", page=2, lines=3),
    ]
    raw = RawDocument(toc=[(1, "Chapter One", 1), (1, "Chapter Two", 1)])
    plan = detect_structure(paragraphs, raw, settings, BODY)
    assert plan.source == "outline"
    assert [c.title for c in plan.chapters] == ["Chapter One", "Chapter Two"]


def test_font_clustering_finds_chapters_without_an_outline(settings: Settings) -> None:
    paragraphs = [
        para("Chapter One", size=18.0),
        para("Body text of the first chapter goes here.", lines=3),
        para("Chapter Two", size=18.0),
        para("Body text of the second chapter goes here.", page=2, lines=3),
    ]
    plan = detect_structure(paragraphs, RawDocument(), settings, BODY)
    assert plan.source == "font-clustering"
    assert len(plan.chapters) == 2
    assert plan.kinds[0] == "heading"
    assert plan.kinds[1] == "paragraph"


def test_heading_level_jumps_are_repaired(settings: Settings) -> None:
    paragraphs = [para("Big", size=24.0), para("Tiny heading", size=13.0)]
    kinds = ["heading", "heading"]
    levels: list[int | None] = [None, None]
    warnings = assign_heading_levels(paragraphs, kinds, levels)  # type: ignore[arg-type]
    assert levels == [1, 2]
    paragraphs = [para("Small first", size=13.0), para("Bigger later", size=24.0)]
    kinds = ["heading", "heading"]
    levels = [None, None]
    warnings = assign_heading_levels(paragraphs, kinds, levels)  # type: ignore[arg-type]
    assert levels[0] == 2  # ranked by size...
    assert levels[1] == 1  # ...and a jump downwards is fine
    assert isinstance(warnings, list)


def test_every_paragraph_lands_in_exactly_one_chapter(settings: Settings) -> None:
    paragraphs = [
        para("Front matter line"),
        para("Chapter One", size=18.0),
        para("Body one.", lines=3),
        para("Chapter Two", size=18.0),
        para("Body two.", lines=3),
    ]
    plan = detect_structure(paragraphs, RawDocument(), settings, BODY)
    assigned = [i for chapter in plan.chapters for i in chapter.paragraph_indices]
    assert sorted(assigned) == list(range(len(paragraphs)))
    assert len(assigned) == len(set(assigned))


def test_text_before_the_first_chapter_becomes_front_matter(settings: Settings) -> None:
    paragraphs = [para("Copyright 1998 by nobody"), para("Chapter One", size=18.0), para("Body.")]
    plan = detect_structure(paragraphs, RawDocument(), settings, BODY)
    assert plan.chapters[0].id == "ch00"
    assert plan.matter[0] == "copyright"


def test_a_document_with_no_headings_is_one_chapter_and_says_so(settings: Settings) -> None:
    plan = detect_structure([para("Just some prose.")], RawDocument(), settings, BODY)
    assert len(plan.chapters) == 1
    assert plan.warnings


def test_prose_in_a_narrow_measure_is_not_called_verse(settings: Settings) -> None:
    """Every line is short in a narrow column; that is not evidence of poetry."""
    paragraphs = [para("A line of ordinary prose.", x1=140.0, lines=4)]
    plan = detect_structure(paragraphs, RawDocument(), settings, BODY)
    assert plan.kinds[0] != "verse"


def test_short_lines_among_long_ones_are_verse(settings: Settings) -> None:
    paragraphs = [
        para("A long line of ordinary prose that runs the full measure of the page.", lines=4),
        para("A long line of ordinary prose that runs the full measure of the page.", lines=4),
        para("Shall I compare thee", x1=140.0, lines=4),
    ]
    plan = detect_structure(paragraphs, RawDocument(), settings, BODY)
    assert plan.kinds[2] == "verse"


def test_scene_breaks_are_recognised(settings: Settings) -> None:
    plan = detect_structure([para("* * *")], RawDocument(), settings, BODY)
    assert plan.kinds[0] == "scene_break"


# -- IR invariants ---------------------------------------------------------------------


def test_duplicate_block_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate block id"):
        document(
            [
                Block(id="b0001", kind="paragraph", text="a"),
                Block(id="b0001", kind="paragraph", text="b"),
            ]
        )


def test_seg_tags_cannot_enter_the_ir() -> None:
    """A stray tag in the IR would corrupt the tag protocol the whole design rests on."""
    with pytest.raises(ValueError, match="<seg>"):
        Block(id="b0001", kind="paragraph", text='<seg id="b0001">smuggled</seg>')


def test_parallel_assertion_names_lost_blocks() -> None:
    source = document([Block(id=f"b{i:04d}", kind="paragraph", text="x") for i in range(5)])
    truncated = document([Block(id=f"b{i:04d}", kind="paragraph", text="y") for i in range(4)])
    with pytest.raises(ValueError, match="b0004"):
        source.assert_parallel_to(truncated)


def test_parallel_assertion_names_invented_blocks() -> None:
    source = document([Block(id="b0000", kind="paragraph", text="x")])
    padded = document(
        [
            Block(id="b0000", kind="paragraph", text="y"),
            Block(id="b9999", kind="paragraph", text="invented"),
        ]
    )
    with pytest.raises(ValueError, match="b9999"):
        source.assert_parallel_to(padded)


def test_parallel_assertion_catches_reordering() -> None:
    source = document(
        [
            Block(id="b0000", kind="paragraph", text="x"),
            Block(id="b0001", kind="paragraph", text="y"),
        ]
    )
    swapped = document(
        [
            Block(id="b0001", kind="paragraph", text="y"),
            Block(id="b0000", kind="paragraph", text="x"),
        ]
    )
    with pytest.raises(ValueError, match="different order"):
        source.assert_parallel_to(swapped)


def test_parallel_assertion_passes_for_a_faithful_translation() -> None:
    source = document([Block(id=f"b{i:04d}", kind="paragraph", text="source") for i in range(3)])
    target = document([Block(id=f"b{i:04d}", kind="paragraph", text="ziel") for i in range(3)])
    source.assert_parallel_to(target)  # must not raise


def test_untranslatable_kinds_are_excluded_from_translation_units() -> None:
    doc = document(
        [
            Block(id="b0000", kind="paragraph", text="Real prose."),
            Block(id="b0001", kind="scene_break", text="* * *", translate=False),
            Block(id="b0002", kind="paragraph", text="   "),
        ]
    )
    assert [b.id for b in doc.translatable_blocks()] == ["b0000"]
    assert len(doc.blocks) == 3  # but nothing is dropped from the IR


def test_ir_survives_a_json_round_trip(tmp_path: Path) -> None:
    doc = document(
        [Block(id="b0000", kind="heading", level=1, text="Kapitel Eins", source_pages=[1])],
        [Chapter(id="ch01", title="Kapitel Eins", block_ids=["b0000"])],
    )
    path = tmp_path / "ir.json"
    doc.save(path)
    assert Document.load(path) == doc


def test_json_schema_can_be_written(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    write_json_schema(path)
    text = path.read_text(encoding="utf-8")
    assert '"ir_version"' in text
    assert '"blocks"' in text


def test_checked_in_schema_is_current(tmp_path: Path) -> None:
    """The committed schema is the review surface for IR changes; keep it in step."""
    committed = Path("schema/document.schema.json")
    assert committed.is_file(), "run: python -c 'from folioai.ir import write_json_schema; ...'"
    regenerated = tmp_path / "schema.json"
    write_json_schema(regenerated)
    assert committed.read_text(encoding="utf-8") == regenerated.read_text(encoding="utf-8"), (
        "The IR changed but schema/document.schema.json was not regenerated."
    )
