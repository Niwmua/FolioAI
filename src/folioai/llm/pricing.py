"""Cost arithmetic (brief §13).

Prices live in ``config/default.yaml`` under ``pricing:`` because provider prices change
constantly and hardcoding them makes the tool wrong within a month. An unknown model warns
once and costs as zero -- never a crash, and never a silently wrong bill either, because
the zero is reported as "unknown" everywhere it surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..logging_setup import get_logger

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

_warned_models: set[str] = set()

#: Prices are quoted per million tokens.
PER = 1_000_000


@dataclass(frozen=True, slots=True)
class Cost:
    """What one call, or one projected phase, costs."""

    prompt_tokens: int
    completion_tokens: int
    usd: float
    known: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            usd=self.usd + other.usd,
            known=self.known and other.known,
        )


def price_call(model: str, prompt_tokens: int, completion_tokens: int, settings: Settings) -> Cost:
    """Cost of a single call. Unknown models cost zero and say so (D-33)."""
    price = settings.price_for(model)
    if price is None:
        if model not in _warned_models:
            log.warning(
                "unknown_model_price",
                model=model,
                remedy=f"add '{model}' under pricing: in your config to get real numbers",
            )
            _warned_models.add(model)
        return Cost(prompt_tokens, completion_tokens, 0.0, known=False)

    usd = (prompt_tokens * price.prompt + completion_tokens * price.completion) / PER
    return Cost(prompt_tokens, completion_tokens, usd, known=True)


def format_usd(amount: float, *, known: bool = True) -> str:
    """Money, rendered so small numbers stay readable and unknowns stay honest."""
    if not known:
        return "unknown"
    if amount == 0:
        return "$0.00"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:,.2f}"


def format_range(low: float, high: float, *, known: bool = True) -> str:
    """Render a cost range at one precision, chosen from its upper bound.

    Both ends share a precision so the pair reads as a range rather than two unrelated
    numbers, and the currency symbol appears once: "$0.009-0.012", not
    "$0.0087 - $0.0121", which no terminal column can hold.
    """
    if not known:
        return "unknown"
    places = 4 if high < 0.01 else (3 if high < 1 else 2)
    if high >= 1:
        return f"${low:,.{places}f}-{high:,.{places}f}"
    return f"${low:.{places}f}-{high:.{places}f}"


def reset_warnings() -> None:
    """Forget which models have already warned. Test-support only."""
    _warned_models.clear()
