"""Cleaning rules, tested one at a time.

The false-positive cases matter more than the true positives here (brief §19): an
over-eager cleaner that eats good text is worse than one that leaves a running head in,
because the damage is invisible until someone reads the translation.
"""

from __future__ import annotations

import pytest

from folioai.config import Settings
from folioai.extract.base import RawDocument, RawLine, RawPage, RawSpan
from folioai.extract.clean import (
    CleaningAudit,
    clean_document,
    dehyphenate,
    find_running_furniture,
    furniture_key,
    group_footnotes,
    group_paragraphs,
    is_page_number,
    mark_footnote_anchors,
    normalize_text,
    repair_drop_caps,
    split_footnote_lines,
    strip_furniture,
)

PAGE_HEIGHT = 700.0
PAGE_WIDTH = 420.0
BODY = 11.0


def line(
    text: str,
    *,
    page: int = 1,
    top: float = 100.0,
    x0: float = 45.0,
    x1: float = 375.0,
    size: float = BODY,
    bold: bool = False,
    height: float = 12.0,
) -> RawLine:
    return RawLine(
        spans=[RawSpan(text=text, size=size, bold=bold)],
        bbox=(x0, top, x1, top + height),
        page=page,
    )


def page(number: int, lines: list[RawLine]) -> RawPage:
    return RawPage(number=number, width=PAGE_WIDTH, height=PAGE_HEIGHT, lines=lines)


# -- normalisation -----------------------------------------------------------------


def test_ligatures_become_ascii() -> None:
    assert normalize_text("ﬁnal ﬂight of the oﬀice") == "final flight of the office"


def test_soft_hyphens_and_zero_width_are_removed() -> None:
    assert normalize_text("co­operate​") == "cooperate"


def test_non_breaking_spaces_normalise() -> None:
    assert normalize_text("10 km") == "10 km"


def test_smart_quotes_are_preserved() -> None:
    """They carry dialogue structure; flattening them loses information (§4.3.5)."""
    text = "“Don’t,” she said."
    assert normalize_text(text) == text


def test_real_letters_are_not_treated_as_ligatures() -> None:
    assert normalize_text("Grüße, cœur, æther") == "Grüße, cœur, æther"


# -- running heads and page numbers ---------------------------------------------------


def test_furniture_key_masks_numbers_so_page_numbers_collapse() -> None:
    assert furniture_key("Page 47") == furniture_key("Page 48")
    assert furniture_key("Chapter IV") == furniture_key("Chapter XII")


@pytest.mark.parametrize(
    "text", ["47", "  12  ", "- 47 -", "[9]", "xiv", "Page 47", "3 of 20", "— 5 —"]
)
def test_page_number_shapes_are_recognised(text: str) -> None:
    assert is_page_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "1984 was a difficult year",
        "He counted to ten.",
        "Chapter 4: The Return",
        "47 Ronin",
        "",
    ],
)
def test_page_number_check_does_not_eat_prose(text: str) -> None:
    assert not is_page_number(text)


def test_running_heads_are_detected_across_pages() -> None:
    raw = RawDocument()
    for number in range(1, 9):
        raw.pages.append(
            page(
                number,
                [
                    line("The Long Afternoon", page=number, top=20.0, size=8.0),
                    line(f"Body text unique to page {number}.", page=number, top=120.0),
                    line("Something else entirely here.", page=number, top=140.0),
                    line(f"- {number} -", page=number, top=PAGE_HEIGHT - 20),
                ],
            )
        )
    furniture = find_running_furniture(raw, Settings().cleaning)
    assert all(0 in indices for indices in furniture.values())
    assert len(furniture) == 8


def test_repeated_body_text_in_the_text_block_survives() -> None:
    """A refrain repeated on every page is prose, not furniture -- position decides."""
    raw = RawDocument()
    for number in range(1, 9):
        raw.pages.append(
            page(
                number,
                [
                    line("The Long Afternoon", page=number, top=20.0, size=8.0),
                    line("And so it goes, he thought, and so it goes.", page=number, top=200.0),
                ],
            )
        )
    audit = CleaningAudit()
    strip_furniture(raw, Settings(), audit)
    remaining = [ln.text for pg in raw.pages for ln in pg.lines]
    assert remaining == ["And so it goes, he thought, and so it goes."] * 8
    assert audit.stripped_headers == 8


def test_short_documents_keep_their_headers() -> None:
    """Under four pages there is no statistic to stand on, so nothing is stripped."""
    raw = RawDocument(
        pages=[
            page(n, [line("A Running Head", page=n, top=20.0), line("Body.", page=n, top=200.0)])
            for n in (1, 2)
        ]
    )
    assert find_running_furniture(raw, Settings().cleaning) == {}


def test_page_numbers_are_stripped_from_the_margin_only() -> None:
    raw = RawDocument(
        pages=[
            page(
                1,
                [
                    line("42", page=1, top=PAGE_HEIGHT - 15),  # folio in the bottom margin
                    line("42", page=1, top=300.0),  # a bare number in the body text
                ],
            )
        ]
    )
    audit = CleaningAudit()
    strip_furniture(raw, Settings(), audit)
    assert [ln.top for ln in raw.pages[0].lines] == [300.0]
    assert audit.stripped_page_numbers == 1


# -- reflow ----------------------------------------------------------------------------


def test_lines_join_into_one_paragraph() -> None:
    lines = [
        line("The lamp above the door had been broken", top=100.0),
        line("for a week, and nobody in the house had", top=115.0),
        line("thought to mention it.", top=130.0, x1=200.0),
    ]
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    assert len(paragraphs) == 1
    assert paragraphs[0].text.startswith("The lamp above the door")
    assert paragraphs[0].text.endswith("thought to mention it.")


def test_a_large_vertical_gap_breaks_a_paragraph() -> None:
    lines = [
        line("First paragraph starts here and", top=100.0),
        line("carries on to a second line.", top=115.0),
        line("Second paragraph, well separated.", top=180.0),
    ]
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    assert len(paragraphs) == 2


def test_paragraphs_do_not_merge_across_a_page_break_by_default() -> None:
    """Defaulting to "continue" welded the next page's running head onto a paragraph."""
    lines = [
        line("A sentence that finishes properly.", page=1, top=600.0, x1=200.0),
        line("A Running Head", page=2, top=20.0),
    ]
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    assert len(paragraphs) == 2


def test_a_sentence_cut_off_mid_clause_continues_onto_the_next_page() -> None:
    lines = [
        line("She said nothing about the lamp, and nothing about a great many", page=1, top=600.0),
        line("other things that year.", page=2, top=100.0),
    ]
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    assert len(paragraphs) == 1
    assert "great many other things" in paragraphs[0].text


def test_reflow_can_be_switched_off() -> None:
    cfg = Settings().cleaning
    cfg.reflow_paragraphs = False
    lines = [line("One line.", top=100.0), line("Another line.", top=115.0)]
    assert len(group_paragraphs(lines, cfg)) == 2


# -- de-hyphenation ----------------------------------------------------------------------


def test_line_break_hyphen_is_joined() -> None:
    lines = [
        line("The committee reached an extraor-", top=100.0),
        line("dinary conclusion that evening.", top=115.0),
    ]
    audit = CleaningAudit()
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    dehyphenate(paragraphs, Settings().cleaning, audit)
    assert "extraordinary conclusion" in paragraphs[0].text
    assert "extraor- dinary" not in paragraphs[0].text
    assert audit.joined_hyphens == 1


def test_a_genuine_compound_keeps_its_hyphen() -> None:
    """`well-being` recurs hyphenated elsewhere, so the split occurrence keeps the hyphen."""
    lines = [
        line("They spoke about the well-", top=100.0),
        line("being of the town for an hour.", top=115.0),
        line("The well-being of the town was", top=180.0),
        line("all anyone discussed that week.", top=195.0),
        line("Nobody mentioned the well-being", top=250.0),
        line("of anybody else at all.", top=265.0),
    ]
    audit = CleaningAudit()
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    dehyphenate(paragraphs, Settings().cleaning, audit)
    assert "well-being of the town" in paragraphs[0].text
    assert "wellbeing" not in paragraphs[0].text
    assert audit.dehyphenations[0].joined is False


def test_dehyphenation_decisions_are_all_audited() -> None:
    lines = [
        line("an extraor-", top=100.0),
        line("dinary thing", top=115.0),
    ]
    audit = CleaningAudit()
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    dehyphenate(paragraphs, Settings().cleaning, audit)
    decision = audit.dehyphenations[0]
    assert decision.candidate == "extraor-dinary"
    assert decision.joined_form_count >= 1


def test_dehyphenation_can_be_switched_off_and_leaves_no_markers() -> None:
    cfg = Settings().cleaning
    cfg.dehyphenate = False
    lines = [line("an extraor-", top=100.0), line("dinary thing", top=115.0)]
    paragraphs = group_paragraphs(lines, cfg)
    dehyphenate(paragraphs, cfg, CleaningAudit())
    assert "\x00" not in paragraphs[0].text


# -- drop caps ------------------------------------------------------------------------------


def test_drop_cap_rejoins_its_word() -> None:
    first = RawLine(
        spans=[
            RawSpan(text="W", size=30.0),
            RawSpan(text="hen the bell rang for the second time", size=BODY),
        ],
        bbox=(45.0, 100.0, 375.0, 130.0),
        page=1,
    )
    paragraphs = group_paragraphs([first], Settings().cleaning)
    audit = CleaningAudit()
    repair_drop_caps(paragraphs, BODY, Settings().cleaning, audit)
    assert paragraphs[0].text.startswith("When the bell rang")
    assert len(audit.drop_caps_repaired) == 1


def test_an_ordinary_capital_is_not_a_drop_cap() -> None:
    lines = [line("When the bell rang for the second time", top=100.0)]
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    audit = CleaningAudit()
    repair_drop_caps(paragraphs, BODY, Settings().cleaning, audit)
    assert audit.drop_caps_repaired == []


# -- footnotes --------------------------------------------------------------------------------


def test_small_type_at_the_page_foot_becomes_footnotes() -> None:
    pg = page(
        1,
        [
            line("The treaty was signed that spring.", top=100.0),
            line("1. The date is disputed by some.", top=640.0, size=7.5),
            line("   The parish record is clear.", top=650.0, size=7.5),
        ],
    )
    body, footnotes = split_footnote_lines(pg, BODY, Settings().cleaning)
    assert len(body) == 1
    notes = group_footnotes(footnotes)
    assert len(notes) == 1
    assert notes[0].footnote_label == "1"
    assert notes[0].text.startswith("The date is disputed")


def test_small_type_in_the_middle_of_a_page_is_not_a_footnote() -> None:
    """An epigraph is set small too; only the bottom of the page counts (§4.3.7)."""
    pg = page(
        1,
        [
            line("A small epigraph, set in petit type.", top=90.0, size=8.0),
            line("The chapter proper begins here.", top=140.0),
            line("And continues for a while.", top=155.0),
        ],
    )
    body, footnotes = split_footnote_lines(pg, BODY, Settings().cleaning)
    assert footnotes == []
    assert len(body) == 3


def test_superscript_marker_becomes_a_footnote_ref_without_eating_spaces() -> None:
    """Rebuilding paragraph text from spans used to weld line ends together."""
    lines = [
        RawLine(
            spans=[
                RawSpan(text="signed in the spring of that year", size=BODY),
                RawSpan(text="1", size=6.5),
            ],
            bbox=(45.0, 100.0, 375.0, 112.0),
            page=1,
        ),
        line("and the terms were published later, to general", top=115.0),
        line("indifference and a single furious letter.", top=130.0, x1=200.0),
    ]
    assert mark_footnote_anchors(lines, BODY, Settings().cleaning) == 1
    paragraphs = group_paragraphs(lines, Settings().cleaning)
    text = paragraphs[0].text
    assert "year[^1] and the terms" in text
    assert "generalindifference" not in text
    assert paragraphs[0].footnote_refs == ["1"]


def test_a_small_word_is_not_mistaken_for_a_superscript_marker() -> None:
    lines = [
        RawLine(
            spans=[RawSpan(text="a note set in small type", size=6.5)],
            bbox=(45.0, 100.0, 375.0, 112.0),
            page=1,
        )
    ]
    assert mark_footnote_anchors(lines, BODY, Settings().cleaning) == 0


# -- the whole pipeline ----------------------------------------------------------------------


def test_clean_document_reports_everything_it_did() -> None:
    raw = RawDocument()
    for number in range(1, 7):
        raw.pages.append(
            page(
                number,
                [
                    line("A Running Head", page=number, top=20.0, size=8.0),
                    line(f"Body of page {number} begins here and", page=number, top=120.0),
                    line("continues onto a second line.", page=number, top=135.0, x1=200.0),
                    line(f"- {number} -", page=number, top=PAGE_HEIGHT - 15),
                ],
            )
        )
    result = clean_document(raw, Settings())
    assert len(result.paragraphs) == 6
    assert result.audit.stripped_headers == 6
    assert result.audit.stripped_page_numbers == 6
    assert all("Running Head" not in p.text for p in result.paragraphs)
