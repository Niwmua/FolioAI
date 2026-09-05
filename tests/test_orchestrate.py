"""The retry ladder, the circuit breaker, and the promise that no segment is ever lost.

Everything here runs against ``FakeLLMClient``: no network, no key, no cost. The scripted
evaluator lets each test state exactly which segments the judge fails and on which attempt.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from folioai.config import Settings
from folioai.errors import BudgetExceeded
from folioai.evaluate import decide, parse_evaluation
from folioai.glossary import Glossary, Term
from folioai.ir import Document
from folioai.llm.client import Message
from folioai.llm.fake import (
    FakeLLMClient,
    dropping_translator,
    refusing_translator,
)
from folioai.orchestrate import CircuitBreakerTripped, Orchestrator, seed_segments
from folioai.store import JobStore
from folioai.tags import parse_segments

PASS_SCORES = {
    "completeness": 98,
    "accuracy": 95,
    "terminology": 95,
    "fluency": 90,
    "formatting": 100,
}
FAIL_SCORES = {
    "completeness": 40,
    "accuracy": 50,
    "terminology": 60,
    "fluency": 70,
    "formatting": 80,
}


def judge(
    *,
    failing: set[str] | None = None,
    fail_until_attempt: int = 0,
    issues_for: set[str] | None = None,
) -> Any:
    """A scripted evaluator.

    Args:
        failing: Segment ids the judge always fails.
        fail_until_attempt: Fail every segment until this many translate calls have been
            made, then pass everything -- for exercising the ladder.
        issues_for: Segment ids to attach a critical issue to.
    """
    failing = failing or set()
    issues_for = issues_for or set()
    state = {"translate_calls": 0}

    def handler(messages: list[Message], model: str) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")

        if "SOURCE:" not in user:  # a translate call, not an evaluate call
            state["translate_calls"] += 1
            parsed = parse_segments(user)
            return "\n".join(
                f'<seg id="{sid}">ZIEL {text}</seg>' for sid, text in parsed.texts.items()
            )

        ids = [line[4:].strip() for line in user.splitlines() if line.startswith("### ")]
        early = state["translate_calls"] <= fail_until_attempt
        scores = []
        issues = []
        for segment_id in ids:
            bad = segment_id in failing or early
            scores.append({"segment_id": segment_id, **(FAIL_SCORES if bad else PASS_SCORES)})
            if bad and segment_id in issues_for:
                issues.append(
                    {
                        "segment_id": segment_id,
                        "dimension": "completeness",
                        "severity": "critical",
                        "source_excerpt": "a clause",
                        "explanation": "a clause was dropped",
                    }
                )
        return json.dumps({"scores": scores, "issues": issues})

    return handler


def build(
    store: JobStore, document: Document, settings: Settings, client: FakeLLMClient, **kwargs: Any
) -> Orchestrator:
    seed_segments(store, "job1", document)
    return Orchestrator(
        client=client,
        settings=settings,
        document=document,
        target_lang="de",
        store=store,
        job_id="job1",
        **kwargs,
    )


# -- the happy path -----------------------------------------------------------------


async def test_a_clean_run_translates_everything_once(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    client = FakeLLMClient(judge())
    orchestrator = build(job_store, tiny_doc, settings, client)
    outcomes = await orchestrator.run()

    expected = {block.id for block in tiny_doc.translatable_blocks()}
    assert set(outcomes) == expected
    assert all(outcome.text.startswith("ZIEL") for outcome in outcomes.values())
    assert orchestrator.stats.needs_review == 0
    assert orchestrator.stats.retries == 0
    assert all(record.status == "done" for record in job_store.list_segments("job1"))


async def test_concurrency_is_bounded_by_the_setting(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    settings.translation.concurrency = 2
    settings.translation.batch_tokens = 40
    client = FakeLLMClient(judge(), latency_s=0.01)
    await build(job_store, tiny_doc, settings, client).run()
    assert client.max_concurrent <= 2


async def test_every_attempt_is_recorded_with_its_own_row(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    await build(job_store, tiny_doc, settings, FakeLLMClient(judge())).run()
    rows = job_store.list_attempts("job1", "b0001")
    assert len(rows) == 1
    assert rows[0]["model"] == settings.models.translator
    assert json.loads(rows[0]["params_json"])["temperature"] == pytest.approx(0.2)


# -- the ladder ------------------------------------------------------------------------


async def test_a_failing_segment_climbs_the_ladder_and_escalates(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """§11: attempt 3 uses the escalation model, and the ladder stops at max_attempts."""
    client = FakeLLMClient(judge(failing={"b0001"}))
    orchestrator = build(job_store, tiny_doc, settings, client)
    await orchestrator.run()

    rows = job_store.list_attempts("job1", "b0001")
    assert [row["attempt_no"] for row in rows] == [1, 2, 3]
    assert rows[2]["model"] == settings.models.escalation
    assert rows[0]["model"] == settings.models.translator
    assert len(rows) == settings.retry.max_attempts


async def test_temperatures_follow_the_ladder(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    client = FakeLLMClient(judge(failing={"b0001"}))
    await build(job_store, tiny_doc, settings, client).run()
    rows = job_store.list_attempts("job1", "b0001")
    temperatures = [json.loads(row["params_json"])["temperature"] for row in rows]
    assert temperatures == settings.retry.attempt_temperatures[:3]


async def test_a_retry_carries_the_previous_output_and_the_issues(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    client = FakeLLMClient(judge(failing={"b0001"}, issues_for={"b0001"}))
    await build(job_store, tiny_doc, settings, client).run()

    retry_calls = [call for call in client.calls_for("translate") if len(call.messages) > 2]
    assert retry_calls, "expected at least one retry call"
    correction = retry_calls[0].messages[-1]["content"]
    assert "did not meet the required" in correction
    assert "a clause was dropped" in correction
    assert "Do not restyle" in correction


async def test_the_ladder_stops_as_soon_as_a_segment_passes(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    client = FakeLLMClient(judge(fail_until_attempt=1))
    orchestrator = build(job_store, tiny_doc, settings, client)
    settings.translation.batch_tokens = 100_000  # one batch, so attempt counting is simple
    await orchestrator.run()
    rows = job_store.list_attempts("job1", "b0001")
    assert len(rows) == 2  # failed once, passed on the retry
    assert orchestrator.stats.needs_review == 0


async def test_the_best_attempt_is_kept_when_none_pass(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """§11: keep the highest-scoring attempt, flag it, never drop the content."""
    client = FakeLLMClient(judge(failing={"b0001"}))
    orchestrator = build(job_store, tiny_doc, settings, client)
    await orchestrator.run()

    outcome = orchestrator.outcomes["b0001"]
    assert outcome.needs_review
    assert outcome.text.strip()
    assert outcome.reason
    record = job_store.get_segment("job1", "b0001")
    assert record is not None
    assert record.needs_review
    assert record.status == "review"
    assert record.final_text


async def test_max_attempts_is_respected(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    settings.retry.max_attempts = 2
    client = FakeLLMClient(judge(failing={"b0001"}))
    await build(job_store, tiny_doc, settings, client).run()
    assert len(job_store.list_attempts("job1", "b0001")) == 2


# -- acceptance criterion §21.5 ------------------------------------------------------------


async def test_a_sabotaged_translator_is_caught_before_the_judge_and_recovered(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """§21.5: a translator that drops every Nth segment.

    The drop must be caught by deterministic validation *before* the evaluator is called,
    and every dropped segment must come back on retry.
    """
    calls = {"n": 0}
    good = judge()
    dropping = dropping_translator(every=2, prefix="ZIEL ")

    def handler(messages: list[Message], model: str) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "SOURCE:" in user:
            return good(messages, model)
        calls["n"] += 1
        if calls["n"] == 1:
            return dropping(messages, model)  # sabotage the first attempt only
        return good(messages, model)

    settings.translation.batch_tokens = 100_000  # one batch per chapter
    client = FakeLLMClient(handler)
    orchestrator = build(job_store, tiny_doc, settings, client)
    await orchestrator.run()

    # Deterministic validation caught the drop, so the judge was never asked about it:
    # exactly one translate attempt went unevaluated, and it was the sabotaged one.
    translate_calls = len(client.calls_for("translate"))
    assert len(client.calls_for("evaluate")) == translate_calls - 1
    assert calls["n"] == translate_calls

    # Nothing was lost: every block came back, and none is empty.
    expected = {block.id for block in tiny_doc.translatable_blocks()}
    assert set(orchestrator.outcomes) == expected
    assert all(outcome.text.strip() for outcome in orchestrator.outcomes.values())
    assert all(record.final_text for record in job_store.list_segments("job1"))


async def test_a_refusing_model_is_caught_and_retried(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    refusing = refusing_translator()
    good = judge()
    state = {"n": 0}

    def handler(messages: list[Message], model: str) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "SOURCE:" in user:
            return good(messages, model)
        state["n"] += 1
        return refusing(messages, model) if state["n"] == 1 else good(messages, model)

    settings.translation.batch_tokens = 100_000
    orchestrator = build(job_store, tiny_doc, settings, FakeLLMClient(handler))
    await orchestrator.run()
    assert all(o.text.startswith("ZIEL") for o in orchestrator.outcomes.values())


async def test_a_segment_that_never_translates_keeps_its_source_text(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """Never leave a gap in the output: an untranslated block beats a missing one."""
    settings.translation.batch_tokens = 100_000
    orchestrator = build(job_store, tiny_doc, settings, FakeLLMClient(refusing_translator()))
    await orchestrator.run()

    expected = {block.id for block in tiny_doc.translatable_blocks()}
    assert set(orchestrator.outcomes) == expected
    assert all(o.text.strip() for o in orchestrator.outcomes.values())
    assert all(o.needs_review for o in orchestrator.outcomes.values())
    source_texts = {b.id: b.text for b in tiny_doc.blocks}
    assert orchestrator.outcomes["b0001"].text == source_texts["b0001"]


# -- the circuit breaker ----------------------------------------------------------------


async def test_the_breaker_trips_when_a_chapter_fails_systematically(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    settings.retry.breaker_min_failures = 3
    settings.translation.batch_tokens = 60  # several batches, so the counter accumulates
    client = FakeLLMClient(judge(failing={f"b{i:04d}" for i in range(12)}))
    orchestrator = build(job_store, tiny_doc, settings, client)

    with pytest.raises(CircuitBreakerTripped) as excinfo:
        await orchestrator.run()
    assert "failing systematically" in excinfo.value.message
    assert excinfo.value.remedy


async def test_a_short_chapter_with_bad_luck_does_not_trip_the_breaker(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """D-34: 2 failures in 8 segments is noise, not systemic breakage."""
    settings.retry.breaker_min_failures = 8
    client = FakeLLMClient(judge(failing={"b0001", "b0002"}))
    orchestrator = build(job_store, tiny_doc, settings, client)
    await orchestrator.run()  # must not raise
    assert orchestrator.stats.needs_review == 2


async def test_work_completed_before_the_breaker_trips_is_committed(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """A trip stops the run; it does not discard what already finished.

    Chapter one segments into batches of 3, 2 and 1. A threshold of 4 lets the first batch
    finish its ladder and commit before the second batch pushes the count over the line.
    """
    settings.retry.breaker_min_failures = 4
    settings.translation.batch_tokens = 60
    settings.translation.concurrency = 1
    client = FakeLLMClient(judge(failing={f"b{i:04d}" for i in range(12)}))
    orchestrator = build(job_store, tiny_doc, settings, client)
    with pytest.raises(CircuitBreakerTripped):
        await orchestrator.run()

    finished = [r for r in job_store.list_segments("job1") if r.final_text]
    assert finished, "the batch that completed before the trip should be committed"
    assert all(r.needs_review for r in finished)
    # And whatever was mid-flight is still pending, so resume picks it up.
    assert job_store.pending_segments("job1")


async def test_a_trip_on_the_first_batch_leaves_everything_resumable(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """The breaker is allowed to fire before anything commits; nothing may be lost."""
    settings.retry.breaker_min_failures = 3
    settings.translation.batch_tokens = 60
    settings.translation.concurrency = 1
    client = FakeLLMClient(judge(failing={f"b{i:04d}" for i in range(12)}))
    with pytest.raises(CircuitBreakerTripped):
        await build(job_store, tiny_doc, settings, client).run()

    pending = {r.segment_id for r in job_store.pending_segments("job1")}
    assert pending == {b.id for b in tiny_doc.translatable_blocks()}


# -- resume ------------------------------------------------------------------------------


async def test_resume_skips_finished_work_and_loses_nothing(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    """§21.3: kill mid-run, resume, and every segment is done exactly once."""
    settings.translation.batch_tokens = 60
    first = FakeLLMClient(judge())
    orchestrator = build(job_store, tiny_doc, settings, first)

    # Simulate a kill after one batch: finalise only what the first batch produced.
    from folioai.segment import segment_document

    batches = segment_document(tiny_doc, settings)
    for unit in batches[0].units:
        job_store.finalize_segment(
            "job1", unit.id, final_text=f"ZIEL {unit.text}", final_score=93.0, needs_review=False
        )

    second = FakeLLMClient(judge())
    resumed = build(job_store, tiny_doc, settings, second)
    await resumed.run()

    translated = [r for r in job_store.list_segments("job1") if r.final_text]
    assert len(translated) == len(tiny_doc.translatable_blocks())
    assert len({r.segment_id for r in translated}) == len(translated)

    # The finished batch was never sent again.
    resent = {sid for call in second.calls_for("translate") for sid in call.segment_ids}
    assert not resent & {unit.id for unit in batches[0].units}
    del orchestrator, first


# -- budget --------------------------------------------------------------------------------


async def test_a_budget_stop_leaves_a_resumable_job(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    settings.translation.batch_tokens = 40
    settings.translation.concurrency = 1

    state = {"calls": 0}
    good = judge()

    def handler(messages: list[Message], model: str) -> str:
        state["calls"] += 1
        if state["calls"] > 3:
            raise BudgetExceeded("Budget of $1.00 reached.", remedy="Raise --max-cost and resume.")
        return good(messages, model)

    orchestrator = build(job_store, tiny_doc, settings, FakeLLMClient(handler))
    with pytest.raises(BudgetExceeded):
        await orchestrator.run()

    assert job_store.pending_segments("job1"), "unfinished segments must remain resumable"


# -- context and glossary plumbing -------------------------------------------------------------


async def test_glossary_terms_reach_the_prompt(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    glossary = Glossary(
        terms=[Term(source="Paragraph", target="Absatz", kind="other", locked=True)]
    )
    client = FakeLLMClient(judge())
    orchestrator = build(job_store, tiny_doc, settings, client, glossary=glossary)
    await orchestrator.run()
    assert any("Absatz" in call.system for call in client.calls_for("translate"))


async def test_previous_target_text_is_offered_as_context(
    tiny_doc: Document, settings: Settings, job_store: JobStore
) -> None:
    settings.translation.batch_tokens = 60
    settings.translation.concurrency = 1
    client = FakeLLMClient(judge())
    await build(job_store, tiny_doc, settings, client).run()

    later = client.calls_for("translate")[-1].system
    assert "DO NOT translate" in later


# -- evaluation decisions ---------------------------------------------------------------------


def test_composite_is_computed_not_taken_from_the_model(settings: Settings) -> None:
    """D-40: models are bad at arithmetic and worse at consistent weighting."""
    evaluation = parse_evaluation(
        json.dumps(
            {
                "scores": [
                    {
                        "segment_id": "b0001",
                        "completeness": 100,
                        "accuracy": 100,
                        "terminology": 100,
                        "fluency": 0,
                        "formatting": 0,
                        "composite": 99.9,
                    }
                ],
                "issues": [],
            }
        )
    )
    verdicts = decide(evaluation, ["b0001"], settings)
    assert verdicts["b0001"].composite == pytest.approx(80.0)


def test_a_dropped_sentence_fails_despite_a_passing_average(settings: Settings) -> None:
    """§10's hard-fail override: a weighted mean must not hide an omission."""
    evaluation = parse_evaluation(
        json.dumps(
            {
                "scores": [
                    {
                        "segment_id": "b0001",
                        "completeness": 60,
                        "accuracy": 95,
                        "terminology": 100,
                        "fluency": 100,
                        "formatting": 100,
                    }
                ],
                "issues": [],
            }
        )
    )
    verdict = decide(evaluation, ["b0001"], settings)["b0001"]
    assert verdict.composite >= settings.evaluation.min_score
    assert not verdict.passed
    assert "completeness" in verdict.reason


def test_a_critical_issue_fails_a_high_scoring_segment(settings: Settings) -> None:
    evaluation = parse_evaluation(
        json.dumps(
            {
                "scores": [{"segment_id": "b0001", **PASS_SCORES}],
                "issues": [
                    {
                        "segment_id": "b0001",
                        "dimension": "completeness",
                        "severity": "critical",
                        "explanation": "a whole clause is missing",
                    }
                ],
            }
        )
    )
    verdict = decide(evaluation, ["b0001"], settings)["b0001"]
    assert not verdict.passed
    assert "critical" in verdict.reason


def test_an_unjudged_segment_is_not_treated_as_a_pass(settings: Settings) -> None:
    evaluation = parse_evaluation(json.dumps({"scores": [], "issues": []}))
    verdict = decide(evaluation, ["b0001"], settings)["b0001"]
    assert not verdict.passed
    assert "no score" in verdict.reason
