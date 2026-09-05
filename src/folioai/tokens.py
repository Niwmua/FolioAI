"""Token counting.

Batch budgeting and cost estimation both need a real tokeniser; neither needs an exact one.
``tiktoken`` is used when installed (the ``tokens`` extra), and the fallback is a
characters-per-token ratio with a warning, so ``estimate`` still works on a bare install
rather than crashing (D-21).

The fallback is deliberately *pessimistic*: it is better for an estimate to come in high and
the run to cost less than quoted than the reverse.
"""

from __future__ import annotations

import functools
from typing import Protocol

from .logging_setup import get_logger

log = get_logger(__name__)

#: Fallback ratio. English prose runs ~4 chars/token; CJK is far denser, so this is a floor.
CHARS_PER_TOKEN = 3.6
DEFAULT_ENCODING = "o200k_base"

_warned = False


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenCounter:
    """Exact counts via tiktoken's BPE."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken:{encoding_name}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


class HeuristicCounter:
    """Character-ratio fallback, used when tiktoken is not installed."""

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, round(len(text) / CHARS_PER_TOKEN))


@functools.lru_cache(maxsize=4)
def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> Tokenizer:
    """Return the best available tokeniser, warning once if it is the fallback."""
    global _warned
    try:
        return TiktokenCounter(encoding_name)
    except Exception as exc:  # ImportError, or an unknown encoding name
        if not _warned:
            log.warning(
                "tokenizer_fallback",
                error=str(exc),
                remedy="install the 'tokens' extra for exact counts: uv sync --extra tokens",
            )
            _warned = True
        return HeuristicCounter()


def count_tokens(text: str, *, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in a string."""
    return get_tokenizer(encoding_name).count(text)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Approximate prompt tokens for a chat request.

    Adds a small per-message overhead for the role and separator tokens every chat format
    inserts. The exact number is provider-specific and not worth chasing: this feeds
    estimation and rate limiting, and the real numbers come back in the response's usage.
    """
    tokenizer = get_tokenizer()
    total = 0
    for message in messages:
        total += 4
        for value in message.values():
            total += tokenizer.count(value)
    return total + 3
