"""Segmentation rules and the cost projection.

The segmentation rules are absolutes, not preferences: never split a block, never cross a
chapter, never lose a unit. Each gets its own test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from folioai.config import Settings
from folioai.estimate import estimate_document, expansion_range, render_estimate
from folioai.ir import Block, Document, ExtractionReport
from folioai.segment import (
    batch_statistics,
    build_batches,
    build_units,
    iter_batch_context,
    segment_document,
)
from folioai.tags import parse_segments

WORD = "paragraph "


def doc_with(blocks: list[Block]) -> Document:
    return Document(
        source_lang="en", blocks=blocks, extraction_report=ExtractionReport(extractor="test")
    )


def block(index: int, *, words: int = 20, chapter: str = "ch01", kind: str = "paragraph") -> Block:
    return Block(
        id=f"b{index:04d}",
        kind=kind,  # type: ignore[arg-type]
        text=(WORD * words).strip(),
        chapter_id=chapter,
    )


# -- segmentation ------------------------------------------------------------------


def test_every_unit_appears_in_exactly_one_batch(settings: Settings) -> None:
    doc = doc_with([block(i) for i in range(50)])
    batches = segment_document(doc, settings)
    ids = [unit.id for batch in batches for unit in batch.units]
    assert ids == [b.id for b in doc.blocks]
    assert len(ids) == len(set(ids))


def test_a_batch_never_crosses_a_chapter_boundary(settings: Settings) -> None:
    doc = doc_with(
        [block(i, chapter="ch01") for i in range(5)]
        + [block(i, chapter="ch02") for i in range(5, 10)]
    )
    for batch in segment_document(doc, settings):
        chapters = {unit.block.chapter_id for unit in batch.units}
        assert len(chapters) == 1
        assert batch.chapter_id in chapters


def test_the_token_budget_is_respected(settings: Settings) -> None:
    settings.translation.batch_tokens = 200
    doc = doc_with([block(i, words=30) for i in range(40)])
    for batch in build_batches(build_units(doc), settings):
        if not batch.oversized:
            assert batch.source_tokens <= settings.translation.batch_tokens


def test_an_oversized_block_travels_alone_and_is_never_split(settings: Settings) -> None:
    """Rule 1 outranks the budget: splitting a paragraph loses the context that fixes it."""
    settings.translation.batch_tokens = 100
    doc = doc_with([block(0, words=10), block(1, words=900), block(2, words=10)])
    batches = build_batches(build_units(doc), settings)
    oversized = [b for b in batches if b.oversized]
    assert len(oversized) == 1
    assert oversized[0].ids == ["b0001"]
    assert oversized[0].source_tokens > settings.translation.batch_tokens
    assert sum(b.size for b in batches) == 3


def test_untranslatable_blocks_are_not_sent_but_are_not_lost(settings: Settings) -> None:
    doc = doc_with(
        [
            block(0),
            Block(id="b0001", kind="scene_break", text="* * *", translate=False),
            block(2),
        ]
    )
    ids = [unit.id for batch in segment_document(doc, settings) for unit in batch.units]
    assert ids == ["b0000", "b0002"]
    assert len(doc.blocks) == 3  # still in the IR, so the export can render it


def test_batches_render_as_valid_tagged_requests(settings: Settings) -> None:
    doc = doc_with([block(i) for i in range(6)])
    for batch in segment_document(doc, settings):
        parsed = parse_segments(batch.render())
        assert parsed.order == batch.ids
        assert parsed.missing(batch.ids) == []


def test_an_empty_document_segments_to_nothing(settings: Settings) -> None:
    assert segment_document(doc_with([]), settings) == []
    assert batch_statistics([])["batches"] == 0


def test_batch_indices_are_contiguous(settings: Settings) -> None:
    settings.translation.batch_tokens = 120
    doc = doc_with([block(i, words=25) for i in range(20)])
    batches = segment_document(doc, settings)
    assert [b.index for b in batches] == list(range(len(batches)))


# -- context -----------------------------------------------------------------------


def test_context_carries_previous_target_and_next_source(settings: Settings) -> None:
    settings.translation.batch_tokens = 60
    doc = doc_with([block(i, words=12) for i in range(10)])
    batches = segment_document(doc, settings)
    translations = {f"b{i:04d}": f"ZIEL {i}" for i in range(10)}

    pairs = list(iter_batch_context(batches, translations, settings))
    later = [ctx for batch, ctx in pairs if batch.index > 0]
    assert later, "expected more than one batch"
    first_later = later[0]
    assert all(text.startswith("ZIEL") for text in first_later.previous_target)
    assert len(first_later.previous_target) <= settings.context.previous_target_blocks
    assert len(first_later.next_source) <= settings.context.next_source_blocks


def test_the_first_batch_has_no_previous_target(settings: Settings) -> None:
    doc = doc_with([block(i) for i in range(4)])
    batches = segment_document(doc, settings)
    _, context = next(iter(iter_batch_context(batches, {}, settings)))
    assert context.previous_target == []


def test_context_uses_translations_not_source(settings: Settings) -> None:
    """Continuity of register comes from what the model already wrote, not the source."""
    settings.translation.batch_tokens = 40
    doc = doc_with([block(i, words=10) for i in range(6)])
    batches = segment_document(doc, settings)
    translations = {f"b{i:04d}": f"ZIEL {i}" for i in range(6)}
    pairs = list(iter_batch_context(batches, translations, settings))
    later_previous = [ctx.previous_target for _, ctx in pairs[1:]]
    assert later_previous, "expected more than one batch"
    assert any("ZIEL" in " ".join(prev) for prev in later_previous)
    assert not any("paragraph" in " ".join(prev) for prev in later_previous)


def test_untranslated_predecessors_are_simply_absent_from_context(settings: Settings) -> None:
    """On a resume, earlier blocks may not be translated yet: omit them, never guess."""
    settings.translation.batch_tokens = 40
    doc = doc_with([block(i, words=10) for i in range(6)])
    batches = segment_document(doc, settings)
    pairs = list(iter_batch_context(batches, {}, settings))
    assert all(ctx.previous_target == [] for _, ctx in pairs)


# -- estimation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lang", "expect_contraction"),
    [("de", False), ("zh-Hans", True), ("ja", False), ("klingon", False)],
)
def test_expansion_ranges_are_language_aware(lang: str, expect_contraction: bool) -> None:
    low, high = expansion_range(lang)
    assert 0 < low <= high
    if expect_contraction:
        assert low < 1.0


def test_estimate_projects_a_range_not_a_number(settings: Settings) -> None:
    doc = doc_with([block(i, words=40) for i in range(30)])
    estimate = estimate_document(doc, settings, target_lang="de")
    assert estimate.cost_low < estimate.cost_high
    assert estimate.batches > 0
    assert {p.name for p in estimate.phases} >= {"translation", "evaluation", "retries"}
    assert "-" in estimate.total_range


def test_estimate_scales_with_the_book(settings: Settings) -> None:
    small = estimate_document(doc_with([block(i) for i in range(5)]), settings, target_lang="de")
    large = estimate_document(doc_with([block(i) for i in range(200)]), settings, target_lang="de")
    assert large.cost_high > small.cost_high * 5


def test_sampling_reduces_the_evaluation_phase(settings: Settings) -> None:
    doc = doc_with([block(i, words=40) for i in range(40)])
    full = estimate_document(doc, settings, target_lang="de")
    settings.evaluation.sample = 0.25
    sampled = estimate_document(doc, settings, target_lang="de")
    full_eval = next(p for p in full.phases if p.name == "evaluation")
    sampled_eval = next(p for p in sampled.phases if p.name == "evaluation")
    assert sampled_eval.cost_high < full_eval.cost_high


def test_back_translation_mode_costs_more(settings: Settings) -> None:
    doc = doc_with([block(i, words=40) for i in range(20)])
    direct = estimate_document(doc, settings, target_lang="de")
    settings.evaluation.mode = "both"
    both = estimate_document(doc, settings, target_lang="de")
    assert both.cost_high > direct.cost_high
    assert any(p.name == "back-translation" for p in both.phases)


def test_an_unknown_model_makes_the_total_unknown_rather_than_wrong(
    settings: Settings,
) -> None:
    settings.models.translator = "nobody/knows-this"
    estimate = estimate_document(doc_with([block(0)]), settings, target_lang="de")
    assert not estimate.all_prices_known
    assert estimate.total_range == "unknown"
    assert any("pricing table" in w for w in estimate.warnings)


def test_same_model_for_both_roles_is_warned_about(settings: Settings) -> None:
    settings.models.evaluator = settings.models.translator
    estimate = estimate_document(doc_with([block(0)]), settings, target_lang="de")
    assert any("blind spots" in w for w in estimate.warnings)


def test_exceeding_max_cost_is_warned_about_before_the_run(settings: Settings) -> None:
    settings.budget.max_cost_usd = 0.0001
    estimate = estimate_document(
        doc_with([block(i, words=50) for i in range(50)]), settings, target_lang="de"
    )
    assert any("max-cost" in w for w in estimate.warnings)


def test_estimate_renders_in_an_eighty_column_terminal(settings: Settings) -> None:
    """A table whose every cell is an ellipsis communicates nothing."""
    import io

    from rich.console import Console

    doc = doc_with([block(i, words=40) for i in range(20)])
    estimate = estimate_document(doc, settings, target_lang="de", source_path="book.pdf")
    buffer = io.StringIO()
    Console(file=buffer, width=80).print(render_estimate(estimate, settings))
    output = buffer.getvalue()
    assert "translation" in output
    assert "$" in output
    assert "…" not in output.split("model")[0]  # the phase column is never truncated


def test_estimate_makes_no_network_calls(
    sample_pdfs: dict[str, Path], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§21.6: --dry-run and estimate must be free. Any client construction fails this test."""
    import folioai.llm.client as client_module

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("estimate must not construct an LLM client")

    monkeypatch.setattr(client_module, "OpenAICompatibleClient", explode)

    from folioai.extract.pipeline import extract_document

    result = extract_document(sample_pdfs["clean_book.pdf"], settings)
    estimate = estimate_document(result.document, settings, target_lang="de")
    assert estimate.blocks > 0
