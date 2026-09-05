"""Glossary: injection, extraction, frequency filtering, review, and book-level adherence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from folioai.config import Settings
from folioai.errors import ConfigError, LLMError
from folioai.glossary import (
    Glossary,
    Term,
    count_occurrences,
    stem_prefix,
    target_form_present,
)
from folioai.glossary_build import (
    audit_adherence,
    build_glossary,
    coerce_kind,
    review_glossary,
    sample_passages,
)
from folioai.ir import Block, Document, ExtractionReport
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient

WARDEN = Term(source="the Warden", target="der Wärter", kind="title")
MANOR = Term(source="Ravenscroft Manor", target="Herrenhaus Ravenscroft", kind="place")
LOCKED = Term(source="Ashby", target="Ashby", kind="place", locked=True)


def doc_with(texts: list[str]) -> Document:
    return Document(
        source_lang="en",
        blocks=[
            Block(id=f"b{i:04d}", kind="paragraph", text=text, chapter_id="ch01")
            for i, text in enumerate(texts)
        ],
        extraction_report=ExtractionReport(extractor="test"),
    )


# -- injection -------------------------------------------------------------------


def test_only_terms_present_in_the_batch_are_injected() -> None:
    """A 200-entry glossary in every prompt costs money and buries what matters (§7)."""
    glossary = Glossary(terms=[WARDEN, MANOR])
    selected = glossary.for_text("The Warden closed the gate behind him.")
    assert [t.source for t in selected] == ["the Warden"]


def test_locked_terms_are_always_injected() -> None:
    glossary = Glossary(terms=[WARDEN, MANOR, LOCKED])
    selected = glossary.for_text("Nothing relevant appears in this sentence at all.")
    assert [t.source for t in selected] == ["Ashby"]


def test_a_locked_term_is_not_injected_twice() -> None:
    glossary = Glossary(terms=[LOCKED])
    selected = glossary.for_text("They rode past Ashby before noon.")
    assert len(selected) == 1


def test_term_matching_respects_word_boundaries() -> None:
    """'Ashby' must not match inside 'Ashbyville'."""
    assert LOCKED.occurs_in("They reached Ashby at dusk.")
    assert not LOCKED.occurs_in("They reached Ashbyville at dusk.")


def test_matching_is_case_insensitive() -> None:
    assert WARDEN.occurs_in("THE WARDEN SPOKE FIRST.")


def test_longest_first_ordering_is_available_for_replacement() -> None:
    glossary = Glossary(terms=[Term(source="Ravenscroft", target="X"), MANOR])
    assert glossary.sorted_by_length()[0].source == "Ravenscroft Manor"


# -- adherence matching ----------------------------------------------------------


def test_inflected_forms_count_as_adherence() -> None:
    assert target_form_present(WARDEN, "Sie sprach mit dem Wärter über das Tor.")
    assert target_form_present(WARDEN, "Des Wärters Schlüssel lag auf dem Tisch.")


def test_a_different_word_is_not_adherence() -> None:
    assert not target_form_present(WARDEN, "Sie sprach mit dem Aufseher über das Tor.")


def test_multi_word_terms_need_every_content_word() -> None:
    assert not target_form_present(MANOR, "Sie kamen in Ravenscroft an.")
    assert target_form_present(MANOR, "Sie kamen im Herrenhaus Ravenscroft an.")


def test_stem_prefix_keeps_short_words_whole() -> None:
    assert stem_prefix("der") == "der"
    assert stem_prefix("Wärter") in "wärter"


def test_occurrence_counting_spans_the_book() -> None:
    counts = count_occurrences(
        [WARDEN, MANOR],
        ["The Warden left.", "the warden returned", "Ravenscroft Manor stood empty."],
    )
    assert counts["the Warden"] == 2
    assert counts["Ravenscroft Manor"] == 1


# -- persistence --------------------------------------------------------------------


def test_glossary_round_trips_through_yaml(tmp_path: Path) -> None:
    glossary = Glossary(terms=[WARDEN, MANOR, LOCKED])
    path = tmp_path / "glossary.yaml"
    glossary.save(path)
    assert Glossary.load(path).terms == glossary.terms
    assert "der Wärter" in path.read_text(encoding="utf-8")  # readable, not escaped


def test_a_malformed_glossary_says_how_to_fix_it(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text("terms:\n  - [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        Glossary.load(path)
    assert excinfo.value.remedy


def test_a_glossary_missing_required_fields_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "glossary.yaml"
    path.write_text("terms:\n  - source: only a source\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        Glossary.load(path)
    assert "source" in (excinfo.value.remedy or "")


# -- extraction ------------------------------------------------------------------------


def extractor(terms_by_call: list[list[dict[str, str]]]) -> object:
    """A fake extractor returning a scripted term list per call."""
    state = {"call": 0}

    def handler(messages: list[Message], model: str) -> str:
        index = min(state["call"], len(terms_by_call) - 1)
        state["call"] += 1
        return json.dumps({"terms": terms_by_call[index]})

    return handler


def test_sampling_spans_the_whole_book(settings: Settings) -> None:
    """A character introduced on page 200 is the one most likely to drift (§7)."""
    document = doc_with([f"Passage {i}. " + ("word " * 40) for i in range(50)])
    passages = sample_passages(document, settings, count=5)
    assert len(passages) == 5
    assert "Passage 0." in passages[0]
    assert any("Passage 4" in p for p in passages[1:])  # not all from the opening
    assert "Passage 0." not in passages[-1]


def test_short_blocks_are_not_sampled(settings: Settings) -> None:
    assert sample_passages(doc_with(["too short", "also short"]), settings) == []


async def test_extraction_keeps_frequent_terms_and_drops_one_offs(
    settings: Settings,
) -> None:
    texts = [
        "The Warden watched the gate. The Warden said nothing at all that evening. " * 3,
        "The Warden returned at dawn, and the fog had not lifted from the river. " * 3,
        "A passing mention of Throgmorton, which occurs exactly once in this book. " * 1,
    ]
    client = FakeLLMClient(
        extractor(
            [
                [
                    {"source": "the Warden", "target": "der Wärter", "kind": "title"},
                    {"source": "Throgmorton", "target": "Throgmorton", "kind": "invented"},
                ]
            ]
        )
    )
    draft = await build_glossary(doc_with(texts), client, settings, target_lang="de", samples=3)

    sources = {t.source for t in draft.glossary.terms}
    assert "the Warden" in sources
    assert "Throgmorton" not in sources  # once is noise
    assert draft.rejected and draft.rejected[0][0] == "Throgmorton"
    assert draft.glossary.terms[0].occurrences >= 3


async def test_names_survive_the_frequency_filter_even_when_rare(
    settings: Settings,
) -> None:
    """A person named twice still needs a consistent rendering."""
    texts = ["Mrs Ainsworth crossed the yard in the rain, saying nothing at all. " * 4]
    client = FakeLLMClient(
        extractor([[{"source": "Mrs Ainsworth", "target": "Frau Ainsworth", "kind": "character"}]])
    )
    draft = await build_glossary(doc_with(texts), client, settings, target_lang="de", samples=1)
    assert [t.source for t in draft.glossary.terms] == ["Mrs Ainsworth"]


async def test_existing_terms_are_preserved_and_not_duplicated(settings: Settings) -> None:
    texts = ["The Warden watched the gate all night without speaking to anyone. " * 4]
    existing = Glossary(terms=[Term(source="the Warden", target="der Aufseher", locked=True)])
    client = FakeLLMClient(
        extractor([[{"source": "the Warden", "target": "der Wärter", "kind": "title"}]])
    )
    draft = await build_glossary(
        doc_with(texts), client, settings, target_lang="de", existing=existing, samples=1
    )
    matching = [t for t in draft.glossary.terms if t.source == "the Warden"]
    assert len(matching) == 1
    assert matching[0].target == "der Aufseher"  # the human decision wins


async def test_one_failed_sample_does_not_lose_the_glossary(settings: Settings) -> None:
    texts = [
        f"The Warden watched gate number {i} all night long, saying nothing. " * 4 for i in range(4)
    ]
    state = {"call": 0}

    def flaky(messages: list[Message], model: str) -> str:
        state["call"] += 1
        if state["call"] == 1:
            raise LLMError("simulated failure", context={"transient": False})
        return json.dumps(
            {"terms": [{"source": "the Warden", "target": "der Wärter", "kind": "title"}]}
        )

    draft = await build_glossary(
        doc_with(texts), FakeLLMClient(flaky), settings, target_lang="de", samples=4
    )
    assert [t.source for t in draft.glossary.terms] == ["the Warden"]


async def test_total_extraction_failure_is_an_error(settings: Settings) -> None:
    def always_fails(messages: list[Message], model: str) -> str:
        raise LLMError("endpoint down", context={"transient": False})

    with pytest.raises(LLMError) as excinfo:
        await build_glossary(
            doc_with(["The Warden watched the gate all night. " * 8]),
            FakeLLMClient(always_fails),
            settings,
            target_lang="de",
            samples=2,
        )
    assert "models.glossary" in (excinfo.value.remedy or "")


async def test_unparseable_extraction_output_is_skipped_not_fatal(settings: Settings) -> None:
    def chatty(messages: list[Message], model: str) -> str:
        return "Sure! Here are the terms I found: the Warden, Ravenscroft."

    draft = await build_glossary(
        doc_with(["The Warden watched the gate all night long, saying nothing. " * 6]),
        FakeLLMClient(chatty),
        settings,
        target_lang="de",
        samples=1,
    )
    assert draft.glossary.terms == []


async def test_json_wrapped_in_a_fence_is_accepted(settings: Settings) -> None:
    def fenced(messages: list[Message], model: str) -> str:
        payload = json.dumps(
            {"terms": [{"source": "the Warden", "target": "der Wärter", "kind": "title"}]}
        )
        return f"```json\n{payload}\n```"

    draft = await build_glossary(
        doc_with(["The Warden watched the gate all night long, saying nothing. " * 6]),
        FakeLLMClient(fenced),
        settings,
        target_lang="de",
        samples=1,
    )
    assert [t.source for t in draft.glossary.terms] == ["the Warden"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [("character", "character"), ("PLACE", "place"), ("nonsense", "other"), ("", "other")],
)
def test_unknown_kinds_default_rather_than_failing(given: str, expected: str) -> None:
    assert coerce_kind(given) == expected


# -- review -------------------------------------------------------------------------------


def test_review_writes_a_backup_before_editing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import folioai.glossary_build as module

    monkeypatch.setattr(module, "open_in_editor", lambda _path: False)
    path = tmp_path / "glossary.yaml"
    glossary = Glossary(terms=[WARDEN])
    review_glossary(glossary, path)
    assert path.with_suffix(".yaml.bak").is_file()


def test_review_reads_back_what_the_user_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import folioai.glossary_build as module

    def edit(path: Path) -> bool:
        Glossary(terms=[Term(source="the Warden", target="der Aufseher", locked=True)]).save(path)
        return True

    monkeypatch.setattr(module, "open_in_editor", edit)
    path = tmp_path / "glossary.yaml"
    result = review_glossary(Glossary(terms=[WARDEN]), path)
    assert result.terms[0].target == "der Aufseher"
    assert result.terms[0].locked


def test_an_editor_that_breaks_the_file_points_at_the_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import folioai.glossary_build as module

    def edit(path: Path) -> bool:
        path.write_text("terms: [broken\n", encoding="utf-8")
        return True

    monkeypatch.setattr(module, "open_in_editor", edit)
    path = tmp_path / "glossary.yaml"
    with pytest.raises(ConfigError) as excinfo:
        review_glossary(Glossary(terms=[WARDEN]), path)
    assert ".bak" in (excinfo.value.remedy or "")


def test_a_missing_editor_returns_the_glossary_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing to launch an editor is no reason to discard a glossary just paid for."""
    import folioai.glossary_build as module

    monkeypatch.setattr(module, "open_in_editor", lambda _path: False)
    glossary = Glossary(terms=[WARDEN])
    assert review_glossary(glossary, tmp_path / "g.yaml").terms == glossary.terms


# -- book-level adherence -------------------------------------------------------------------


def test_consistent_rendering_across_the_book_is_reported_as_consistent() -> None:
    glossary = Glossary(terms=[WARDEN])
    sources = {"b0001": "The Warden spoke.", "b0002": "The Warden left."}
    targets = {"b0001": "Der Wärter sprach.", "b0002": "Der Wärter ging."}
    usages = audit_adherence(glossary, sources, targets)
    assert len(usages) == 1
    assert usages[0].consistent
    assert usages[0].occurrences == 2


def test_a_term_rendered_two_ways_is_caught_even_if_each_segment_looked_fine() -> None:
    """The book-level view: chapter 3 and chapter 19 disagreeing with each other (§15)."""
    glossary = Glossary(terms=[WARDEN])
    sources = {"b0001": "The Warden spoke.", "b0400": "The Warden left."}
    targets = {"b0001": "Der Wärter sprach.", "b0400": "Der Aufseher ging."}
    usages = audit_adherence(glossary, sources, targets)
    assert not usages[0].consistent
    assert "(not found)" in usages[0].renderings
    assert usages[0].renderings["(not found)"] == ["b0400"]


def test_terms_that_never_appear_are_not_reported() -> None:
    glossary = Glossary(terms=[WARDEN, MANOR])
    usages = audit_adherence(
        glossary, {"b0001": "The Warden spoke."}, {"b0001": "Der Wärter sprach."}
    )
    assert [u.term.source for u in usages] == ["the Warden"]
