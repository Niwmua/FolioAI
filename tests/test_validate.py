"""Deterministic checks, each with a true positive and a false positive.

The false-positive cases carry more weight here (brief §19). A check that flags good work
gets ignored, and once the report is ignored the real failures pass unread too.
"""

from __future__ import annotations

import pytest

from folioai.config import Settings
from folioai.glossary import Glossary, Term
from folioai.ir import Block
from folioai.llm.client import LLMResponse
from folioai.segment import Batch, Unit
from folioai.tags import parse_segments, render_segments
from folioai.translate import BatchTranslation
from folioai.validate import (
    LengthRatioTracker,
    aggregate_sentence_deltas,
    sentence_count_delta,
    validate_batch,
)

SOURCE = (
    "The lamp above the door had been broken for a week, and nobody in the house had "
    "thought to mention it."
)
GOOD = (
    "Die Lampe über der Tür war seit einer Woche kaputt, und niemand im Haus hatte daran "
    "gedacht, es zu erwähnen."
)

# Twice the expected length but not repetitive: long enough to trip the ratio check without
# tripping the degeneration check, which quite correctly treats a doubled sentence as a loop.
VERBOSE = (
    "Die Lampe über der Tür war nun schon seit einer ganzen Woche entzwei, und niemand in "
    "diesem Haushalt hatte bis dahin je daran gedacht, das Thema überhaupt zur Sprache zu "
    "bringen oder jemanden darauf hinzuweisen, obwohl es allen aufgefallen sein musste."
)


def make_batch(pairs: list[tuple[str, str]], *, chapter: str = "ch01") -> Batch:
    units = [
        Unit(block=Block(id=sid, kind="paragraph", text=text, chapter_id=chapter), tokens=40)
        for sid, text in pairs
    ]
    return Batch(index=0, chapter_id=chapter, units=units)


def make_result(
    batch: Batch, response_text: str, *, attempt_no: int = 1, finish_reason: str = "stop"
) -> BatchTranslation:
    return BatchTranslation(
        batch=batch,
        attempt_no=attempt_no,
        model="test/model",
        response=LLMResponse(text=response_text, model="test/model", finish_reason=finish_reason),
        parsed=parse_segments(response_text),
        messages=[],
    )


def validate(pairs: list[tuple[str, str]], response: str, settings: Settings, **kwargs):  # type: ignore[no-untyped-def]
    batch = make_batch(pairs)
    return validate_batch(make_result(batch, response), settings, **kwargs)


# -- segment integrity ------------------------------------------------------------


def test_a_complete_faithful_batch_produces_no_findings(settings: Settings) -> None:
    report = validate([("b0001", SOURCE)], render_segments([("b0001", GOOD)]), settings)
    assert report.findings == []


def test_a_dropped_segment_is_critical(settings: Settings) -> None:
    report = validate(
        [("b0001", SOURCE), ("b0002", SOURCE)],
        render_segments([("b0001", GOOD)]),
        settings,
    )
    assert report.has_critical
    assert any(f.segment_id == "b0002" and f.check == "segment_integrity" for f in report.critical)


def test_an_invented_segment_is_critical(settings: Settings) -> None:
    report = validate(
        [("b0001", SOURCE)],
        render_segments([("b0001", GOOD), ("b9999", "Etwas Erfundenes.")]),
        settings,
    )
    assert any(f.segment_id == "b9999" for f in report.critical)


def test_a_duplicated_segment_is_critical(settings: Settings) -> None:
    response = render_segments([("b0001", GOOD)]) + "\n" + render_segments([("b0001", GOOD)])
    assert validate([("b0001", SOURCE)], response, settings).has_critical


def test_reordered_segments_are_critical(settings: Settings) -> None:
    report = validate(
        [("b0001", SOURCE), ("b0002", SOURCE)],
        render_segments([("b0002", GOOD), ("b0001", GOOD)]),
        settings,
    )
    assert any("wrong order" in f.detail for f in report.critical)


# -- emptiness, refusal, degeneration ------------------------------------------------


def test_an_empty_translation_is_critical(settings: Settings) -> None:
    assert validate([("b0001", SOURCE)], '<seg id="b0001">   </seg>', settings).has_critical


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with translating this content.",
        "As an AI, I am unable to render this passage.",
        "I have omitted the offensive passage.",
        "[Content omitted]",
    ],
)
def test_refusals_and_meta_text_are_critical(text: str, settings: Settings) -> None:
    report = validate([("b0001", SOURCE)], render_segments([("b0001", text)]), settings)
    assert report.has_critical
    assert any(f.check == "refusal" for f in report.critical)


def test_a_bare_refusal_with_no_tags_is_caught(settings: Settings) -> None:
    report = validate([("b0001", SOURCE)], "I'm sorry, but I cannot translate this.", settings)
    assert any(f.check == "refusal" for f in report.critical)


def test_prose_about_a_character_apologising_is_not_a_refusal(settings: Settings) -> None:
    """A novel in which someone says sorry is not a model refusing to work."""
    line = (
        "„Es tut mir leid“, sagte sie, „aber ich kann Ihnen dabei nicht helfen.“ Sie wandte "
        "sich ab und ging zur Tür hinaus, ohne ein weiteres Wort zu sagen."
    )
    report = validate([("b0001", SOURCE)], render_segments([("b0001", line)]), settings)
    assert not report.has_critical


def test_a_blocked_segment_is_critical_but_not_missing(settings: Settings) -> None:
    response = '<seg id="b0001" status="blocked">contains slurs</seg>'
    report = validate([("b0001", SOURCE)], response, settings)
    assert any(f.check == "blocked" for f in report.critical)
    assert not any("was not returned" in f.detail for f in report.critical)


def test_degeneration_is_critical(settings: Settings) -> None:
    loop = " ".join(["die Wand des Hauses"] * 60)
    report = validate([("b0001", SOURCE)], render_segments([("b0001", loop)]), settings)
    assert any(f.check == "degeneration" for f in report.critical)


def test_legitimate_repetition_is_not_degeneration(settings: Settings) -> None:
    """Deliberate anaphora is a real device; it must not read as a sampling collapse."""
    source = "It was the best of times, it was the worst of times. " * 3
    text = (
        "Es war die beste aller Zeiten, es war die schlimmste aller Zeiten, es war das "
        "Zeitalter der Weisheit, es war das Zeitalter der Torheit, es war die Epoche des "
        "Glaubens, es war die Epoche der Ungläubigkeit."
    )
    report = validate([("b0001", source)], render_segments([("b0001", text)]), settings)
    assert not any(f.check == "degeneration" for f in report.critical)


def test_a_truncated_response_is_critical(settings: Settings) -> None:
    batch = make_batch([("b0001", SOURCE)])
    result = make_result(batch, render_segments([("b0001", GOOD)]), finish_reason="length")
    report = validate_batch(result, settings)
    assert any(f.check == "truncation" for f in report.critical)


# -- warnings ---------------------------------------------------------------------------


def test_length_ratio_only_fires_once_the_median_is_learned(settings: Settings) -> None:
    tracker = LengthRatioTracker(min_samples=5)
    for _ in range(6):
        tracker.record(SOURCE, GOOD)
    assert tracker.median is not None

    # Twice the expected length: well outside the ratio band, but under the 3x threshold
    # where degeneration takes over and short-circuits the warnings.
    report = validate(
        [("b0001", SOURCE)],
        render_segments([("b0001", VERBOSE)]),
        settings,
        length_tracker=tracker,
    )
    assert any(f.check == "length_ratio" for f in report.warnings)


def test_length_ratio_stays_silent_early_in_a_run(settings: Settings) -> None:
    """With three samples it has no opinion worth having, so it must not have one."""
    tracker = LengthRatioTracker()
    report = validate(
        [("b0001", SOURCE)],
        render_segments([("b0001", VERBOSE)]),
        settings,
        length_tracker=tracker,
    )
    assert not any(f.check == "length_ratio" for f in report.warnings)


def test_untranslated_passthrough_is_flagged(settings: Settings) -> None:
    report = validate([("b0001", SOURCE)], render_segments([("b0001", SOURCE)]), settings)
    assert any(f.check == "passthrough" for f in report.warnings)


def test_a_real_translation_is_not_flagged_as_passthrough(settings: Settings) -> None:
    report = validate([("b0001", SOURCE)], render_segments([("b0001", GOOD)]), settings)
    assert not any(f.check == "passthrough" for f in report.warnings)


def test_a_list_of_proper_nouns_is_not_flagged_as_passthrough(settings: Settings) -> None:
    """Names surviving translation intact is the correct answer, not a defect."""
    names = (
        "Ravenscroft Manor, Ashby Hall, Dunmore Castle, Ellerslie House, Whitcombe Grange, "
        "Pemberton Lodge, Harrowgate Abbey, Stanmore Priory"
    )
    report = validate([("b0001", names)], render_segments([("b0001", names)]), settings)
    assert not any(f.check == "passthrough" for f in report.warnings)


def test_glossary_violations_are_warnings(settings: Settings) -> None:
    glossary = Glossary(terms=[Term(source="the Warden", target="der Wärter", kind="title")])
    report = validate(
        [("b0001", "The Warden watched the gate all night.")],
        render_segments([("b0001", "Der Aufseher bewachte die ganze Nacht das Tor.")]),
        settings,
        glossary=glossary,
    )
    assert any(f.check == "glossary" for f in report.warnings)


def test_inflected_glossary_terms_are_accepted(settings: Settings) -> None:
    """`der Wärter` legitimately becomes `dem Wärter`; exact matching would drown in these."""
    glossary = Glossary(terms=[Term(source="the Warden", target="der Wärter", kind="title")])
    report = validate(
        [("b0001", "She spoke to the Warden about the gate.")],
        render_segments([("b0001", "Sie sprach mit dem Wärter über das Tor.")]),
        settings,
        glossary=glossary,
    )
    assert not any(f.check == "glossary" for f in report.warnings)


def test_missing_numbers_are_warned_about(settings: Settings) -> None:
    source = "He counted 47 sheep, 12 goats and 1899 head of cattle before dawn came."
    target = "Er zählte Schafe, Ziegen und Rinder, bevor die Morgendämmerung kam."
    report = validate([("b0001", source)], render_segments([("b0001", target)]), settings)
    assert any(f.check == "numbers" for f in report.warnings)


def test_preserved_numbers_are_not_warned_about(settings: Settings) -> None:
    source = "He counted 47 sheep and 12 goats before dawn."
    target = "Er zählte 47 Schafe und 12 Ziegen vor der Morgendämmerung."
    report = validate([("b0001", source)], render_segments([("b0001", target)]), settings)
    assert not any(f.check == "numbers" for f in report.warnings)


def test_a_single_number_does_not_trigger_the_check(settings: Settings) -> None:
    """One spelled-out numeral is normal prose, not evidence of anything."""
    source = "There were 3 of them at the table that evening, and nobody spoke."
    target = "Sie waren zu dritt am Tisch an jenem Abend, und niemand sprach ein Wort."
    report = validate([("b0001", source)], render_segments([("b0001", target)]), settings)
    assert not any(f.check == "numbers" for f in report.warnings)


def test_dropped_footnote_refs_are_warned_about(settings: Settings) -> None:
    source = "The treaty was signed that spring[^1] and published later[^2]."
    target = "Der Vertrag wurde in jenem Frühjahr unterzeichnet[^1] und später veröffentlicht."
    report = validate([("b0001", source)], render_segments([("b0001", target)]), settings)
    assert any("footnote refs dropped" in f.detail for f in report.warnings)


def test_unbalanced_emphasis_is_warned_about(settings: Settings) -> None:
    report = validate(
        [("b0001", "She was *quite* sure of it.")],
        render_segments([("b0001", "Sie war *ganz sicher.")]),
        settings,
    )
    assert any(f.check == "markup" for f in report.warnings)


def test_balanced_emphasis_is_accepted(settings: Settings) -> None:
    report = validate(
        [("b0001", "She was *quite* sure of it, and **nobody** disagreed with her.")],
        render_segments([("b0001", "Sie war *ganz* sicher, und **niemand** widersprach ihr.")]),
        settings,
    )
    assert not any(f.check == "markup" for f in report.warnings)


# -- ordering of work -------------------------------------------------------------------


def test_warnings_are_skipped_when_something_critical_fired(settings: Settings) -> None:
    """No point reporting nuance about text that is about to be thrown away and retried."""
    report = validate(
        [("b0001", SOURCE), ("b0002", SOURCE)],
        render_segments([("b0001", SOURCE)]),  # b0002 dropped, b0001 untranslated
        settings,
    )
    assert report.has_critical
    assert report.warnings == []


def test_report_helpers_separate_hints_from_problems(settings: Settings) -> None:
    report = validate(
        [("b0001", SOURCE), ("b0002", SOURCE)], render_segments([("b0001", GOOD)]), settings
    )
    assert report.mechanical_problems()
    assert report.failed_segments() == {"b0002"}


# -- sentence deltas (aggregate only) -------------------------------------------------------


def test_sentence_delta_is_computed_but_never_a_finding(settings: Settings) -> None:
    """Per segment it is noise; §9's info row is surfaced in aggregate instead (PLAN §2.4)."""
    source = "He left. She stayed. Nobody spoke."
    target = "Er ging, sie blieb, und niemand sprach."
    report = validate([("b0001", source)], render_segments([("b0001", target)]), settings)
    assert not any(f.check == "sentence_count" for f in report.findings)
    assert sentence_count_delta(source, target) == -2


def test_sentence_deltas_aggregate_by_chapter() -> None:
    pairs = [
        ("ch01", "He left. She stayed.", "Er ging, sie blieb."),
        ("ch01", "It rained. It stopped.", "Es regnete, dann hörte es auf."),
        ("ch02", "One sentence here.", "Ein Satz. Noch einer. Und ein dritter."),
    ]
    aggregated = aggregate_sentence_deltas(pairs)
    assert aggregated["ch01"] == -1.0
    assert aggregated["ch02"] == 2.0
