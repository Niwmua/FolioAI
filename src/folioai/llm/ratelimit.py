"""Token-bucket rate limiting for requests and tokens (brief §8).

Two buckets, both required before a call goes out: one for requests per minute, one for
tokens per minute. Providers meter both, and hitting either produces a 429 that costs
latency and, on some endpoints, a wasted prompt.

The buckets are asyncio-aware and refill continuously rather than in per-minute steps, so a
burst at the start of a minute does not stall the next fifty-nine seconds.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class TokenBucket:
    """A continuously refilling bucket.

    Args:
        capacity: Bucket size, i.e. the largest burst allowed.
        refill_per_second: Steady-state rate.
    """

    capacity: float
    refill_per_second: float
    tokens: float = field(init=False)
    _updated: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.refill_per_second <= 0:
            raise ValueError("bucket capacity and refill rate must both be positive")
        self.tokens = self.capacity
        self._updated = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = max(now - self._updated, 0.0)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self._updated = now

    def _wait_time(self, amount: float, now: float) -> float:
        self._refill(now)
        if self.tokens >= amount:
            return 0.0
        return (amount - self.tokens) / self.refill_per_second

    async def acquire(self, amount: float = 1.0) -> float:
        """Wait until ``amount`` is available, then take it. Returns seconds waited.

        A request larger than the whole bucket is clamped to the capacity rather than
        deadlocking forever: one oversized call should be slow, not fatal.
        """
        wanted = min(amount, self.capacity)
        waited = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                delay = self._wait_time(wanted, now)
                if delay <= 0:
                    self.tokens -= wanted
                    return waited
            waited += delay
            await asyncio.sleep(delay)

    def try_acquire(self, amount: float = 1.0) -> bool:
        """Non-blocking take, for synchronous callers and tests."""
        wanted = min(amount, self.capacity)
        self._refill(time.monotonic())
        if self.tokens >= wanted:
            self.tokens -= wanted
            return True
        return False


class RateLimiter:
    """Requests-per-minute and tokens-per-minute, enforced together."""

    def __init__(self, *, rpm: int, tpm: int) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._requests = TokenBucket(capacity=float(rpm), refill_per_second=rpm / 60.0)
        self._tokens = TokenBucket(capacity=float(tpm), refill_per_second=tpm / 60.0)
        self.total_wait_s = 0.0

    async def acquire(self, estimated_tokens: int) -> float:
        """Block until both budgets allow the call. Returns seconds waited."""
        waited = await self._requests.acquire(1.0)
        waited += await self._tokens.acquire(float(max(estimated_tokens, 1)))
        self.total_wait_s += waited
        if waited > 1.0:
            log.debug("rate_limited", waited_s=round(waited, 2), tokens=estimated_tokens)
        return waited

    def refund(self, tokens: int) -> None:
        """Return unused token budget after a call came in smaller than estimated.

        Estimation is deliberately pessimistic, so without this the limiter would throttle
        to well under the configured rate over a long run.
        """
        if tokens <= 0:
            return
        self._tokens.tokens = min(self._tokens.capacity, self._tokens.tokens + tokens)
