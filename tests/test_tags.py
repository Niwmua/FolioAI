"""The ``<seg>`` protocol: round trip, and every malformed shape a model can produce.

Parsing must never repair. A repaired response is indistinguishable from a correct one,
which would silently defeat the omission detection the whole design rests on.
"""

from __future__ import annotations

import pytest

from folioai.tags import extract_segment_ids, parse_segments, render_segments, strip_code_fences

REQUESTED = ["b0001", "b0002", "b0003"]


def wire(*pairs: tuple[str, str]) -> str:
    return render_segments(pairs)


# -- round trip ---------------------------------------------------------------------


def test_round_trip_preserves_ids_order_and_text() -> None:
    rendered = wire(("b0001", "First."), ("b0002", "Second."), ("b0003", "Third."))
    parsed = parse_segments(rendered)
    assert parsed.order == REQUESTED
    assert parsed.texts["b0002"] == "Second."
    assert parsed.missing(REQUESTED) == []
    assert parsed.unexpected(REQUESTED) == []
    assert not parsed.out_of_order(REQUESTED)
    assert parsed.stray_text == ""


def test_multiline_and_inline_markup_survive() -> None:
    text = "A line with *em*, **strong**, `code` and a ref[^4].\nA second line."
    parsed = parse_segments(wire(("b0001", text)))
    assert parsed.texts["b0001"] == text


def test_prose_containing_angle_brackets_survives() -> None:
    parsed = parse_segments(wire(("b0001", "She wrote a < b on the board, then 3 > 2.")))
    assert parsed.texts["b0001"] == "She wrote a < b on the board, then 3 > 2."


def test_extract_ids_reads_a_rendered_batch() -> None:
    assert extract_segment_ids(wire(("b0001", "x"), ("b0002", "y"))) == ["b0001", "b0002"]


# -- omission and invention ------------------------------------------------------------


def test_a_missing_segment_is_detected_for_free() -> None:
    parsed = parse_segments(wire(("b0001", "First."), ("b0003", "Third.")))
    assert parsed.missing(REQUESTED) == ["b0002"]


def test_an_invented_segment_is_detected() -> None:
    parsed = parse_segments(wire(*[(i, "x") for i in [*REQUESTED, "b9999"]]))
    assert parsed.unexpected(REQUESTED) == ["b9999"]


def test_reordering_is_detected_even_when_nothing_is_lost() -> None:
    parsed = parse_segments(wire(("b0002", "Second."), ("b0001", "First."), ("b0003", "Third.")))
    assert parsed.missing(REQUESTED) == []
    assert parsed.out_of_order(REQUESTED)


def test_duplicates_are_recorded_and_the_first_wins() -> None:
    parsed = parse_segments('<seg id="b0001">first copy</seg>\n<seg id="b0001">second copy</seg>')
    assert parsed.duplicates == ["b0001"]
    assert parsed.texts["b0001"] == "first copy"


# -- malformed shapes -------------------------------------------------------------------


def test_an_unclosed_tag_does_not_swallow_the_next_segment() -> None:
    """The dangerous case: a plausible-looking wrong answer rather than a visible failure."""
    response = '<seg id="b0001">First.\n<seg id="b0002">Second.</seg>'
    parsed = parse_segments(response)
    assert parsed.order == ["b0002"]
    assert parsed.texts["b0002"] == "Second."
    assert "First." in parsed.stray_text
    assert parsed.malformed_openings == 1
    assert parsed.missing(REQUESTED) == ["b0001", "b0003"]


def test_a_tag_with_no_id_is_malformed_not_guessed_at() -> None:
    parsed = parse_segments("<seg>Who is this for?</seg>")
    assert parsed.order == []
    assert parsed.malformed_openings == 1
    assert "Who is this for?" in parsed.stray_text


def test_a_preamble_lands_in_stray_text() -> None:
    parsed = parse_segments('Here is the translation:\n<seg id="b0001">Erster.</seg>')
    assert parsed.texts["b0001"] == "Erster."
    assert parsed.stray_text == "Here is the translation:"


def test_trailing_commentary_lands_in_stray_text() -> None:
    parsed = parse_segments(
        '<seg id="b0001">Erster.</seg>\nLet me know if you would like a different register!'
    )
    assert "different register" in parsed.stray_text


def test_a_refusal_parses_as_nothing_but_stray_text() -> None:
    parsed = parse_segments("I'm sorry, but I can't translate this passage.")
    assert parsed.order == []
    assert parsed.texts == {}
    assert parsed.stray_text.startswith("I'm sorry")


def test_an_empty_response_is_empty_not_an_exception() -> None:
    parsed = parse_segments("")
    assert parsed.order == []
    assert parsed.missing(REQUESTED) == REQUESTED


# -- tolerated variations ------------------------------------------------------------------


def test_a_markdown_fence_is_unwrapped() -> None:
    """A fence adds no content and hides none, so unwrapping cannot mask an omission."""
    fenced = '```xml\n<seg id="b0001">Erster.</seg>\n```'
    parsed = parse_segments(fenced)
    assert parsed.texts["b0001"] == "Erster."
    assert parsed.stray_text == ""


def test_stripping_a_fence_leaves_unfenced_text_alone() -> None:
    assert strip_code_fences("no fence here") == "no fence here"


@pytest.mark.parametrize(
    "response",
    [
        "<seg id='b0001'>Erster.</seg>",
        '<SEG ID="b0001">Erster.</SEG>',
        '<seg  id = "b0001" >Erster.</seg >',
    ],
)
def test_quoting_case_and_spacing_variations_parse(response: str) -> None:
    assert parse_segments(response).texts["b0001"] == "Erster."


# -- blocked segments ------------------------------------------------------------------------


def test_a_blocked_segment_is_present_not_missing() -> None:
    """§8: refusing must still return the tag, so the orchestrator can route it to review."""
    parsed = parse_segments(
        '<seg id="b0001">Erster.</seg>\n<seg id="b0002" status="blocked">reason given</seg>'
    )
    assert parsed.missing(["b0001", "b0002"]) == []
    assert parsed.blocked["b0002"] == "reason given"
    assert "b0002" not in parsed.texts


def test_a_self_closing_blocked_segment_parses() -> None:
    parsed = parse_segments('<seg id="b0002" status="blocked"/>')
    assert parsed.blocked["b0002"] == "blocked"
    assert parsed.missing(["b0002"]) == []
