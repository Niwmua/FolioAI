"""The one opt-in integration test that hits a real endpoint (brief §19).

Deselected by default. To run it:

    export FOLIOAI_API_KEY=sk-or-v1-...
    uv run pytest -m live -v

It costs a fraction of a cent: two short batches and one evaluation. What it proves is the
part no fake can — that the request shape, the tag protocol, the structured-output schema
and the cost accounting all survive contact with a real provider.

Every assertion here is about *our* contract, never about translation quality: a test that
fails because a model chose a different word is a test that gets deleted.
"""

from __future__ import annotations

import os

import pytest

from folioai.config import Settings, packaged_settings
from folioai.evaluate import Evaluator
from folioai.ir import Block, Document, ExtractionReport
from folioai.llm.client import OpenAICompatibleClient
from folioai.llm.pricing import format_usd
from folioai.segment import BatchContext, segment_document
from folioai.translate import Translator
from folioai.validate import validate_batch

pytestmark = pytest.mark.live

SOURCE = [
    "The lamp above the door had been broken for a week, and nobody in the house had "
    "thought to mention it.",
    "She said nothing about it, and the habit had begun to feel less like discretion and "
    "more like a wall she had built without meaning to.",
]


@pytest.fixture
def live_settings() -> Settings:
    settings = packaged_settings()
    if not (
        os.environ.get("FOLIOAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ):
        pytest.skip("no API key in the environment")

    from folioai.config import load_settings

    settings = load_settings()
    # Keep it cheap: small models, one batch, no retries beyond the first attempt.
    settings.models.translator = os.environ.get("FOLIOAI_LIVE_MODEL", "openai/gpt-4.1-mini")
    settings.models.evaluator = os.environ.get("FOLIOAI_LIVE_EVALUATOR", "openai/gpt-4.1-mini")
    settings.translation.batch_tokens = 2000
    settings.budget.max_cost_usd = 0.25  # a hard stop, in case a model runs away
    return settings


def live_document() -> Document:
    return Document(
        source_lang="en",
        target_lang="de",
        title="A Test Passage",
        blocks=[
            Block(id=f"b{index:04d}", kind="paragraph", text=text, chapter_id="ch01")
            for index, text in enumerate(SOURCE)
        ],
        extraction_report=ExtractionReport(extractor="test"),
    )


async def test_a_real_endpoint_honours_the_tag_protocol(live_settings: Settings) -> None:
    """The whole design rests on the model returning our ids. Verify it against a real one."""
    document = live_document()
    client = OpenAICompatibleClient(live_settings)
    try:
        batch = segment_document(document, live_settings)[0]
        translator = Translator(client, live_settings, document=document, target_lang="German")
        result = await translator.translate_batch(batch, BatchContext())
    finally:
        await client.aclose()

    assert result.complete, (
        f"missing={result.missing} unexpected={result.unexpected} "
        f"stray={result.parsed.stray_text[:200]!r}"
    )
    assert set(result.texts) == set(batch.ids)
    assert all(text.strip() for text in result.texts.values())

    # Real usage and real money, not the zeros a fake returns.
    assert result.response.prompt_tokens > 0
    assert result.response.completion_tokens > 0
    assert result.response.cost.known
    print(f"\ntranslation cost: {format_usd(result.response.cost.usd)}")


async def test_deterministic_validation_passes_a_real_translation(
    live_settings: Settings,
) -> None:
    """The checks must not fire on genuinely good output -- the false-positive case."""
    document = live_document()
    client = OpenAICompatibleClient(live_settings)
    try:
        batch = segment_document(document, live_settings)[0]
        translator = Translator(client, live_settings, document=document, target_lang="German")
        result = await translator.translate_batch(batch, BatchContext())
    finally:
        await client.aclose()

    report = validate_batch(result, live_settings)
    assert not report.has_critical, [f.describe() for f in report.critical]
    if report.warnings:
        print("\nwarnings (not failures):", [f.describe() for f in report.warnings])


async def test_a_real_evaluator_returns_the_structured_schema(
    live_settings: Settings,
) -> None:
    """Structured outputs are the part most likely to differ between providers (D-42)."""
    document = live_document()
    client = OpenAICompatibleClient(live_settings)
    try:
        batch = segment_document(document, live_settings)[0]
        translator = Translator(client, live_settings, document=document, target_lang="German")
        result = await translator.translate_batch(batch, BatchContext())

        evaluator = Evaluator(client, live_settings, source_lang="English", target_lang="German")
        evaluation = await evaluator.evaluate(result)
    finally:
        await client.aclose()

    assert set(evaluation.verdicts) == set(batch.ids)
    for segment_id, verdict in evaluation.verdicts.items():
        assert verdict.score is not None, f"{segment_id} came back unscored"
        assert 0 <= verdict.composite <= 100
    print(f"\nmean composite: {evaluation.mean_composite} via {evaluation.structured_output}")


async def test_the_cache_makes_the_second_identical_call_free(
    live_settings: Settings,
) -> None:
    """§8's content-hash cache, against a real endpoint rather than a stub."""
    document = live_document()
    client = OpenAICompatibleClient(live_settings)
    try:
        batch = segment_document(document, live_settings)[0]
        translator = Translator(client, live_settings, document=document, target_lang="German")
        first = await translator.translate_batch(batch, BatchContext())
        second = await translator.translate_batch(batch, BatchContext())
    finally:
        await client.aclose()

    assert not first.response.cached
    assert second.response.cached
    assert second.response.text == first.response.text
