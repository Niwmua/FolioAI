"""LLM plumbing: pricing, rate limiting, cache, retry classification, the fake client.

Nothing here touches the network. The real client is exercised through a stub transport so
its retry and error-mapping logic is tested without an endpoint or a key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from folioai.config import Settings
from folioai.errors import BudgetExceeded, LLMError, RateLimitError
from folioai.llm.cache import PromptCache, fingerprint
from folioai.llm.client import BudgetGuard, LLMResponse, _map_exception
from folioai.llm.fake import (
    FakeLLMClient,
    degenerate_translator,
    dropping_translator,
    echo_translator,
    inventing_translator,
    malformed_translator,
    refusing_translator,
    scripted_by_id,
)
from folioai.llm.pricing import Cost, format_range, format_usd, price_call, reset_warnings
from folioai.llm.ratelimit import RateLimiter, TokenBucket
from folioai.tags import parse_segments, render_segments
from folioai.tokens import HeuristicCounter, count_message_tokens, count_tokens

MESSAGES = [{"role": "system", "content": "You translate."}, {"role": "user", "content": "hi"}]


# -- tokens -------------------------------------------------------------------------


def test_token_counts_are_positive_and_scale_with_length() -> None:
    short = count_tokens("The lamp above the door.")
    long = count_tokens("The lamp above the door. " * 20)
    assert 0 < short < long


def test_empty_text_costs_nothing() -> None:
    assert count_tokens("") == 0


def test_heuristic_fallback_is_in_the_right_ballpark() -> None:
    """It only has to be close: it feeds budgeting, not billing."""
    text = "The lamp above the door had been broken for a week, and nobody mentioned it."
    heuristic = HeuristicCounter().count(text)
    assert 0.4 <= heuristic / max(count_tokens(text), 1) <= 2.5


def test_message_overhead_is_counted() -> None:
    assert count_message_tokens(MESSAGES) > count_tokens("You translate.") + count_tokens("hi")


# -- pricing -------------------------------------------------------------------------


def test_known_model_prices_by_the_million(settings: Settings) -> None:
    from folioai.config import ModelPrice

    settings.pricing["test/model"] = ModelPrice(prompt=3.0, completion=15.0)
    cost = price_call("test/model", 1_000_000, 1_000_000, settings)
    assert cost.usd == pytest.approx(18.0)
    assert cost.known


def test_unknown_model_costs_zero_and_says_so(settings: Settings) -> None:
    reset_warnings()
    cost = price_call("nobody/knows-this", 1000, 1000, settings)
    assert cost.usd == 0.0
    assert cost.known is False


def test_costs_add_and_carry_the_unknown_flag() -> None:
    known = Cost(10, 10, 1.0, known=True)
    unknown = Cost(5, 5, 0.0, known=False)
    total = known + unknown
    assert total.prompt_tokens == 15
    assert total.usd == pytest.approx(1.0)
    assert total.known is False


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(0.0, "$0.00"), (0.0001, "$0.0001"), (12.5, "$12.50"), (1234.5, "$1,234.50")],
)
def test_money_formatting(amount: float, expected: str) -> None:
    assert format_usd(amount) == expected


def test_cost_ranges_share_one_precision_and_one_symbol() -> None:
    assert format_range(0.0087, 0.0121) == "$0.009-0.012"
    assert format_range(0.0004, 0.0009) == "$0.0004-0.0009"
    assert format_range(1.5, 12.25) == "$1.50-12.25"
    assert format_range(0.0, 0.0, known=False) == "unknown"


# -- rate limiting --------------------------------------------------------------------


def test_bucket_allows_a_burst_up_to_capacity() -> None:
    bucket = TokenBucket(capacity=5, refill_per_second=1)
    assert all(bucket.try_acquire() for _ in range(5))
    assert not bucket.try_acquire()


def test_bucket_refills_over_time() -> None:
    bucket = TokenBucket(capacity=10, refill_per_second=1000)
    for _ in range(10):
        bucket.try_acquire()
    assert not bucket.try_acquire()
    import time

    time.sleep(0.02)
    assert bucket.try_acquire()


def test_bucket_rejects_a_nonsensical_configuration() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(capacity=0, refill_per_second=1)


async def test_an_oversized_request_is_clamped_not_deadlocked() -> None:
    """A single call larger than the whole bucket should be slow, never eternal."""
    bucket = TokenBucket(capacity=10, refill_per_second=10_000)
    await asyncio.wait_for(bucket.acquire(1_000_000), timeout=2.0)


async def test_limiter_enforces_both_budgets() -> None:
    limiter = RateLimiter(rpm=6000, tpm=6000)
    await limiter.acquire(100)
    assert limiter._tokens.tokens == pytest.approx(5900, abs=50)
    assert limiter._requests.tokens == pytest.approx(5999, abs=1)


async def test_refund_returns_unused_token_budget() -> None:
    limiter = RateLimiter(rpm=6000, tpm=6000)
    await limiter.acquire(1000)
    before = limiter._tokens.tokens
    limiter.refund(500)
    assert limiter._tokens.tokens > before


# -- cache -----------------------------------------------------------------------------


def test_fingerprint_is_stable_and_sensitive(tmp_path: Path) -> None:
    base: dict[str, Any] = {
        "model": "m",
        "messages": MESSAGES,
        "params": {"temperature": 0.2},
    }
    assert fingerprint(**base) == fingerprint(**base)
    assert fingerprint(**{**base, "model": "other"}) != fingerprint(**base)
    assert fingerprint(**{**base, "params": {"temperature": 0.3}}) != fingerprint(**base)
    changed = [MESSAGES[0], {"role": "user", "content": "different"}]
    assert fingerprint(**{**base, "messages": changed}) != fingerprint(**base)


def test_fingerprint_ignores_dict_ordering() -> None:
    a = fingerprint(model="m", messages=MESSAGES, params={"temperature": 0.2, "top_p": 1.0})
    b = fingerprint(model="m", messages=MESSAGES, params={"top_p": 1.0, "temperature": 0.2})
    assert a == b


def test_none_valued_params_do_not_change_the_key() -> None:
    a = fingerprint(model="m", messages=MESSAGES, params={"temperature": 0.2})
    b = fingerprint(model="m", messages=MESSAGES, params={"temperature": 0.2, "seed": None})
    assert a == b


def test_cache_round_trip(tmp_path: Path) -> None:
    with PromptCache(tmp_path / "cache.db") as cache:
        assert cache.get("k") is None
        cache.put("k", model="m", output_text="hello")
        assert cache.get("k") == "hello"
        assert cache.hits == 1
        assert cache.misses == 1


def test_a_disabled_cache_stores_nothing(tmp_path: Path) -> None:
    with PromptCache(tmp_path / "cache.db", enabled=False) as cache:
        cache.put("k", model="m", output_text="hello")
        assert cache.get("k") is None


def test_cache_survives_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "cache.db"
    with PromptCache(path) as first:
        first.put("k", model="m", output_text="persisted")
    with PromptCache(path) as second:
        assert second.get("k") == "persisted"


# -- budget ------------------------------------------------------------------------------


def test_budget_guard_stops_at_the_ceiling() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    guard.check()
    guard.add(Cost(1000, 1000, 0.6))
    guard.check()  # still under
    guard.add(Cost(1000, 1000, 0.6))
    with pytest.raises(BudgetExceeded) as excinfo:
        guard.check()
    assert "resume" in (excinfo.value.remedy or "")
    assert guard.remaining == 0.0


def test_no_ceiling_means_no_limit() -> None:
    guard = BudgetGuard(limit_usd=None)
    guard.add(Cost(1, 1, 10_000.0))
    guard.check()
    assert guard.remaining is None


# -- error classification ----------------------------------------------------------------


class _StubError(Exception):
    def __init__(self, status: int | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"stub error {status}")
        self.status_code = status
        if headers is not None:
            self.response = type("R", (), {"headers": headers})()


@pytest.mark.parametrize("status", [500, 502, 503, 529])
def test_server_errors_are_transient(status: int) -> None:
    mapped = _map_exception(_StubError(status), model="m")
    assert mapped.context.get("transient") is True


def test_rate_limit_is_transient_and_honours_retry_after() -> None:
    mapped = _map_exception(_StubError(429, {"retry-after": "12"}), model="m")
    assert isinstance(mapped, RateLimitError)
    assert mapped.retry_after == 12.0
    assert mapped.context.get("transient") is True


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_permanent(status: int) -> None:
    """Retrying a malformed prompt three times is three times as slow at being wrong."""
    mapped = _map_exception(_StubError(status), model="m")
    assert not mapped.context.get("transient")
    assert mapped.remedy


def test_a_400_names_the_likely_cause() -> None:
    assert "batch_tokens" in (_map_exception(_StubError(400), model="m").remedy or "")


def test_an_unknown_model_error_explains_vendor_prefixes() -> None:
    assert "vendor-prefixed" in (_map_exception(_StubError(404), model="m").remedy or "")


def test_a_timeout_is_transient() -> None:
    assert _map_exception(TimeoutError("slow"), model="m").context.get("transient") is True


# -- the fake client ------------------------------------------------------------------------


async def test_fake_client_echoes_every_requested_segment() -> None:
    client = FakeLLMClient(echo_translator())
    body = render_segments([("b0001", "First."), ("b0002", "Second.")])
    response = await client.complete([{"role": "user", "content": body}], model="fake/model")
    parsed = parse_segments(response.text)
    assert parsed.order == ["b0001", "b0002"]
    assert parsed.texts["b0001"] == "[t] First."
    assert client.call_count == 1
    assert client.calls[0].segment_ids == ["b0001", "b0002"]


async def test_fake_client_replays_a_script_and_refuses_to_run_dry() -> None:
    client = FakeLLMClient(responses=['<seg id="b0001">one</seg>'])
    await client.complete([{"role": "user", "content": "x"}], model="m")
    with pytest.raises(AssertionError, match="ran out of scripted responses"):
        await client.complete([{"role": "user", "content": "x"}], model="m")


async def test_fake_client_can_simulate_transient_failures() -> None:
    client = FakeLLMClient(echo_translator(), fail_times=2)
    for _ in range(2):
        with pytest.raises(LLMError):
            await client.complete([{"role": "user", "content": "x"}], model="m")
    response = await client.complete([{"role": "user", "content": "x"}], model="m")
    assert isinstance(response, LLMResponse)


async def test_fake_client_records_concurrency() -> None:
    client = FakeLLMClient(echo_translator(), latency_s=0.01)
    body = render_segments([("b0001", "x")])
    await asyncio.gather(
        *(client.complete([{"role": "user", "content": body}], model="m") for _ in range(5))
    )
    assert client.max_concurrent > 1


async def _run(handler: Any) -> str:
    client = FakeLLMClient(handler)
    body = render_segments([(f"b{i:04d}", f"Sentence {i}.") for i in range(1, 11)])
    response = await client.complete([{"role": "user", "content": body}], model="m")
    return response.text


async def test_misbehaviour_dropping() -> None:
    parsed = parse_segments(await _run(dropping_translator(every=10)))
    assert "b0010" not in parsed.ids
    assert len(parsed.order) == 9


async def test_misbehaviour_inventing() -> None:
    parsed = parse_segments(await _run(inventing_translator()))
    requested = [f"b{i:04d}" for i in range(1, 11)]
    assert parsed.unexpected(requested) == ["b9999"]


async def test_misbehaviour_malformed() -> None:
    parsed = parse_segments(await _run(malformed_translator()))
    assert parsed.malformed_openings >= 1
    assert parsed.stray_text


async def test_misbehaviour_refusal() -> None:
    parsed = parse_segments(await _run(refusing_translator()))
    assert parsed.order == []
    assert "can't help" in parsed.stray_text


async def test_misbehaviour_degeneration() -> None:
    parsed = parse_segments(await _run(degenerate_translator(repeats=30)))
    assert len(parsed.texts["b0001"]) > 300


async def test_scripted_by_id_translates_only_what_it_is_given() -> None:
    handler = scripted_by_id({"b0001": "Erster."})
    client = FakeLLMClient(handler)
    body = render_segments([("b0001", "First."), ("b0002", "Second.")])
    response = await client.complete([{"role": "user", "content": body}], model="m")
    parsed = parse_segments(response.text)
    assert parsed.texts["b0001"] == "Erster."
    assert parsed.texts["b0002"] == "Second."
