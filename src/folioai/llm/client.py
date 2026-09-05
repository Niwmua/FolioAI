"""Async OpenAI-compatible client (brief §8).

Design notes worth knowing before editing:

* **We own the retry loop.** The SDK is constructed with ``max_retries=0`` (D-30). Its own
  policy is opaque, and §12 requires a row per attempt with that attempt's tokens, latency
  and cost -- which is impossible if the SDK silently retries inside one call.
* **Transient and quality retries are different budgets** (D-31). A connection reset is not
  a failed translation, and must not consume one of the segment's three attempts.
* **Only transient failures retry.** A 400 from a malformed prompt fails immediately and
  loudly; retrying it three times just spends three times as long being wrong.
* **Every retry path has a hard cap** (§0). There is no ``while True`` in this file that is
  not bounded by ``max_transient_retries``.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..errors import BudgetExceeded, LLMError, RateLimitError
from ..logging_setup import get_logger
from ..paths import cache_db_path
from ..tokens import count_message_tokens
from .cache import PromptCache, fingerprint
from .pricing import Cost, format_usd, price_call
from .ratelimit import RateLimiter

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..config import Settings

log = get_logger(__name__)

Message = dict[str, str]

#: Models already warned about a thinking budget that ate their answer. One line per model
#: is a diagnosis; one per call is noise nobody reads.
_warned_reasoning: set[str] = set()


def _warn_once(seen: set[str], key: str, event: str, **fields: Any) -> None:
    if key in seen:
        return
    seen.add(key)
    log.warning(event, model=key, **fields)


@dataclass(slots=True)
class LLMResponse:
    """One completed call, with everything the store and the report need."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Of ``completion_tokens``, how many the model spent thinking. Invisible in ``text``
    #: but charged against ``max_tokens``, which is what makes a truncation diagnosable.
    reasoning_tokens: int = 0
    cost: Cost = field(default_factory=lambda: Cost(0, 0, 0.0))
    latency_ms: int = 0
    cached: bool = False
    transient_retries: int = 0
    finish_reason: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """The endpoint stopped for length, so the output is cut off mid-thought."""
        return self.finish_reason == "length"


class LLMClient(Protocol):
    """What the translation and evaluation engines depend on.

    Narrow on purpose: everything downstream can be developed and tested against
    ``FakeLLMClient`` without a network or an API key.
    """

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        purpose: str = "translate",
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


class BudgetGuard:
    """Enforces ``--max-cost`` (brief §13).

    Checked before a call and updated after it, so a run stops cleanly at the ceiling and
    leaves the job resumable rather than discovering the overspend at the end.
    """

    def __init__(self, limit_usd: float | None) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.calls = 0

    def check(self) -> None:
        if self.limit_usd is None:
            return
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"Budget of {format_usd(self.limit_usd)} reached "
                f"({format_usd(self.spent_usd)} spent over {self.calls} calls).",
                remedy=(
                    "The job is saved and resumable. Raise --max-cost and run "
                    "'folioai resume <job_id>' to carry on."
                ),
                context={"limit_usd": self.limit_usd, "spent_usd": self.spent_usd},
            )

    def add(self, cost: Cost) -> None:
        self.spent_usd += cost.usd
        self.calls += 1

    @property
    def remaining(self) -> float | None:
        if self.limit_usd is None:
            return None
        return max(self.limit_usd - self.spent_usd, 0.0)


class OpenAICompatibleClient:
    """Talks to any OpenAI-compatible endpoint via ``base_url``."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache: PromptCache | None = None,
        budget: BudgetGuard | None = None,
        on_usage: Callable[[str, str, int, int, float], None] | None = None,
    ) -> None:
        """
        Args:
            settings: Merged configuration.
            cache: Response cache; a shared on-disk cache is opened if omitted.
            budget: Spend ceiling. Omit for no limit.
            on_usage: Called as ``(model, purpose, prompt_tokens, completion_tokens, usd)``
                after every non-cached call, so the caller can write a ``usage`` row without
                this module knowing anything about SQLite.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - openai is a hard dependency
            raise LLMError(
                "The 'openai' package is required to talk to an LLM endpoint.",
                remedy="Run: uv sync",
            ) from exc

        if not settings.api_key:
            raise LLMError(
                "No API key found.",
                remedy=(
                    "Set FOLIOAI_API_KEY (or OPENROUTER_API_KEY / OPENAI_API_KEY) in your "
                    "environment or a .env file. Keys are never read from config files."
                ),
            )

        self.settings = settings
        self.budget = budget or BudgetGuard(settings.budget.max_cost_usd)
        self.cache = (
            cache
            if cache is not None
            else PromptCache(cache_db_path(), enabled=settings.llm.cache_enabled)
        )
        self.limiter = RateLimiter(rpm=settings.llm.rpm, tpm=settings.llm.tpm)
        self.on_usage = on_usage
        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.llm.base_url,
            timeout=settings.llm.timeout_s,
            max_retries=0,  # D-30: this module owns retries
        )

    # -- public API ---------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        purpose: str = "translate",
    ) -> LLMResponse:
        """Send one chat completion, with rate limiting, caching and bounded retries.

        Raises:
            LLMError: on a permanent failure, or after the transient retry cap.
            RateLimitError: only if rate limiting persists past the cap.
            BudgetExceeded: if the spend ceiling was already reached.
        """
        params: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed if seed is not None else self.settings.llm.seed,
            "response_format": response_format,
        }
        key = fingerprint(model=model, messages=messages, params=params)

        cached = self.cache.get(key)
        if cached is not None:
            log.debug("cache_hit", model=model, purpose=purpose)
            return LLMResponse(text=cached, model=model, cached=True, params=params)

        self.budget.check()
        estimated_prompt = count_message_tokens(messages)
        await self.limiter.acquire(estimated_prompt + (max_tokens or 0))

        response = await self._call_with_retries(
            messages, model=model, params=params, purpose=purpose
        )

        self.cache.put(
            key,
            model=model,
            output_text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return response

    async def aclose(self) -> None:
        await self._client.close()
        self.cache.close()

    # -- retry loop ------------------------------------------------------------------

    async def _call_with_retries(
        self,
        messages: list[Message],
        *,
        model: str,
        params: dict[str, Any],
        purpose: str,
    ) -> LLMResponse:
        cap = self.settings.llm.max_transient_retries
        last_error: Exception | None = None

        for attempt in range(cap + 1):
            started = time.perf_counter()
            try:
                return await self._call_once(
                    messages, model=model, params=params, purpose=purpose, retries=attempt
                )
            except RateLimitError as exc:
                last_error = exc
                delay = exc.retry_after if exc.retry_after is not None else self._backoff(attempt)
                log.warning(
                    "rate_limited_by_endpoint",
                    model=model,
                    attempt=attempt + 1,
                    cap=cap,
                    sleeping_s=round(delay, 2),
                )
            except LLMError as exc:
                if not _is_transient(exc):
                    raise
                last_error = exc
                delay = self._backoff(attempt)
                log.warning(
                    "transient_llm_failure",
                    model=model,
                    attempt=attempt + 1,
                    cap=cap,
                    error=exc.message,
                    sleeping_s=round(delay, 2),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )

            if attempt < cap:
                await asyncio.sleep(delay)

        raise LLMError(
            f"Giving up on {model} after {cap + 1} attempts: {last_error}",
            remedy=(
                "The endpoint is failing or rate-limiting persistently. Check its status, "
                "lower --concurrency, or reduce llm.rpm / llm.tpm in your config. The job "
                "is resumable."
            ),
            context={"model": model, "purpose": purpose, "attempts": cap + 1},
        )

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped by ``llm.backoff_max_s``."""
        cfg = self.settings.llm
        # 2.0** rather than 2**: int.__pow__ is typed as returning Any (negative
        # exponents give a float), which leaks Any into the return type under strict mode.
        base: float = min(cfg.backoff_initial_s * (2.0**attempt), cfg.backoff_max_s)
        jitter = base * cfg.backoff_jitter
        offset: float = random.uniform(-jitter, jitter)  # jitter, not cryptography
        return max(0.0, base + offset)

    async def _call_once(
        self,
        messages: list[Message],
        *,
        model: str,
        params: dict[str, Any],
        purpose: str,
        retries: int,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
        }
        for key in ("max_tokens", "seed", "response_format"):
            if params.get(key) is not None:
                request[key] = params[key]

        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**request)
        except Exception as exc:  # mapped below into our own hierarchy
            raise _map_exception(exc, model=model) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        if not completion.choices:
            raise LLMError(
                f"{model} returned no choices.",
                remedy="Retry, or try a different model with --translator-model.",
                context={"model": model, "purpose": purpose},
            )
        choice = completion.choices[0]
        text = choice.message.content or ""

        usage = completion.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if prompt_tokens == 0:
            # Some OpenAI-compatible endpoints omit usage; fall back to our own count so
            # cost tracking degrades rather than silently reporting zero spend.
            prompt_tokens = count_message_tokens(messages)
            completion_tokens = count_message_tokens([{"content": text}])
            log.debug("usage_missing_estimated", model=model)

        reasoning_tokens = _reasoning_tokens(usage)
        reported_usd = _reported_cost(usage)
        if reported_usd is None:
            cost = price_call(model, prompt_tokens, completion_tokens, self.settings)
        else:
            # The endpoint billed us and said what it charged. That beats a local price
            # table every time -- it is the actual number, it covers models the table has
            # never heard of, and it prices reasoning tokens the way the provider does.
            cost = Cost(prompt_tokens, completion_tokens, reported_usd, known=True)
        self.budget.add(cost)
        self.limiter.refund(max(0, (params.get("max_tokens") or 0) - completion_tokens))

        if self.on_usage is not None:
            self.on_usage(model, purpose, prompt_tokens, completion_tokens, cost.usd)

        log.info(
            "llm_call",
            model=model,
            purpose=purpose,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=round(cost.usd, 6),
            cost_source="endpoint" if reported_usd is not None else "table",
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
            max_tokens=params.get("max_tokens"),
            transient_retries=retries,
        )
        if choice.finish_reason == "length" and reasoning_tokens > completion_tokens // 2:
            # Say this out loud once per model. It is the difference between "the model is
            # bad at this" and "the model never got room to answer", and the two have
            # nothing in common as problems.
            _warn_once(
                _warned_reasoning,
                model,
                "reasoning_budget_exhausted",
                reasoning_tokens=reasoning_tokens,
                completion_tokens=completion_tokens,
                max_tokens=params.get("max_tokens"),
                remedy=(
                    f"{model} spent most of its completion budget thinking and was cut off "
                    "before finishing. Raise translation.reasoning_headroom_tokens "
                    "(FOLIOAI_TRANSLATION__REASONING_HEADROOM_TOKENS)."
                ),
            )
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost=cost,
            latency_ms=latency_ms,
            transient_retries=retries,
            finish_reason=choice.finish_reason,
            params=params,
        )


# -- usage extraction ---------------------------------------------------------------
#
# Both of these read fields that are not in the OpenAI schema but that most gateways send
# anyway. They are optional by nature, so every read is defensive and a missing field means
# "not reported" rather than an error: an endpoint that omits them must still work.


def _usage_extra(usage: Any, name: str) -> Any:
    """Read ``name`` off a usage object whether it is declared, extra, or a plain dict."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(name)
    value = getattr(usage, name, None)
    if value is None:
        extra = getattr(usage, "model_extra", None) or {}
        value = extra.get(name)
    return value


def _reasoning_tokens(usage: Any) -> int:
    """Thinking tokens, from ``completion_tokens_details.reasoning_tokens``.

    Charged against ``max_tokens`` but absent from the response text, so without this a
    truncated reasoning model looks like a model that simply stopped talking.
    """
    details = _usage_extra(usage, "completion_tokens_details")
    value = _usage_extra(details, "reasoning_tokens")
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _reported_cost(usage: Any) -> float | None:
    """What the endpoint says the call cost, in USD, or ``None`` if it did not say.

    OpenRouter-style gateways return this on every call. Preferring it over the local
    ``pricing:`` table is what stops a run reporting $0.00 for a model the table has never
    heard of -- the failure mode is not a wrong number, it is a confident zero.
    """
    value = _usage_extra(usage, "cost")
    if value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if cost >= 0 else None


def reset_warnings() -> None:
    """Forget which models have already warned. Test-support only."""
    _warned_reasoning.clear()


# -- error mapping ------------------------------------------------------------------


#: Marker used to tell a retryable failure from a permanent one.
_TRANSIENT = "transient"


def _is_transient(exc: LLMError) -> bool:
    return bool(exc.context.get(_TRANSIENT))


def _map_exception(exc: Exception, *, model: str) -> LLMError:
    """Translate an SDK exception into our hierarchy, deciding retryable vs fatal.

    The classification is the whole point: a 400 means the prompt is wrong and will be wrong
    again in five seconds, whereas a 503 means try later. Getting this backwards either
    burns money on hopeless retries or gives up on a blip.
    """
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    detail = str(exc)[:300]

    if name in {"APIConnectionError", "APITimeoutError"} or isinstance(exc, TimeoutError):
        return LLMError(
            f"Could not reach the endpoint for {model}: {detail}",
            remedy="Check your network and llm.base_url. This will be retried.",
            context={"model": model, _TRANSIENT: True},
        )

    if status == 429 or name == "RateLimitError":
        retry_after = _retry_after_seconds(exc)
        return RateLimitError(
            f"{model} is rate limited.",
            retry_after=retry_after,
            remedy="Lower --concurrency, or llm.rpm / llm.tpm.",
            context={"model": model, _TRANSIENT: True},
        )

    if status is not None and 500 <= int(status) < 600:
        return LLMError(
            f"{model} returned a server error ({status}): {detail}",
            remedy="A provider-side failure. This will be retried.",
            context={"model": model, "status": status, _TRANSIENT: True},
        )

    if status in {401, 403}:
        return LLMError(
            f"The endpoint rejected the API key ({status}).",
            remedy=(
                "Check FOLIOAI_API_KEY is set and valid for llm.base_url. Keys are read "
                "only from the environment or a .env file."
            ),
            context={"model": model, "status": status},
        )

    if status == 404:
        return LLMError(
            f"The endpoint does not know the model {model!r} ({status}).",
            remedy=(
                "Check the model name matches the provider's catalogue. On OpenRouter names "
                "are vendor-prefixed, e.g. 'openai/gpt-4.1'."
            ),
            context={"model": model, "status": status},
        )

    if status == 400:
        return LLMError(
            f"{model} rejected the request ({status}): {detail}",
            remedy=(
                "This is a malformed request, not a blip, so it is not retried. A batch "
                "over the model's context window is the usual cause: lower "
                "translation.batch_tokens."
            ),
            context={"model": model, "status": status},
        )

    return LLMError(
        f"Call to {model} failed ({name}): {detail}",
        remedy="If this repeats, try a different model or endpoint.",
        context={"model": model, "status": status},
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honour ``Retry-After`` when the provider sends one (§8)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for header in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        value = headers.get(header) if hasattr(headers, "get") else None
        if value:
            try:
                return max(0.0, float(str(value).rstrip("s")))
            except ValueError:
                continue
    return None
