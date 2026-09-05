"""``FakeLLMClient``: the whole pipeline, testable without a network or a bill (brief §19).

Unit tests must never hit the network, so every stage above this one is developed against
this class. It ships with the misbehaviours the brief names -- dropped segment, extra
segment, malformed tags, refusal, degeneration, truncation -- because a validator that has
only ever seen well-formed input is a validator nobody has tested.

The behaviours are deliberately *mechanical* rather than random: a test that fails
intermittently teaches nothing.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import LLMError
from ..tags import extract_segment_ids, parse_segments
from .client import LLMResponse, Message
from .pricing import Cost

Handler = Callable[[list[Message], str], str]

#: Rough chars-per-token, only ever used to turn a simulated token budget into a character
#: cut. Exactness would not make the fake any more faithful.
_CHARS_PER_TOKEN = 4.0


@dataclass(slots=True)
class RecordedCall:
    """One call as the fake saw it, so tests can assert on what was actually sent."""

    messages: list[Message]
    model: str
    purpose: str
    params: dict[str, Any]

    @property
    def system(self) -> str:
        return next((m["content"] for m in self.messages if m.get("role") == "system"), "")

    @property
    def user(self) -> str:
        return next((m["content"] for m in self.messages if m.get("role") == "user"), "")

    @property
    def segment_ids(self) -> list[str]:
        return extract_segment_ids(self.user)


class FakeLLMClient:
    """A scripted stand-in for :class:`OpenAICompatibleClient`.

    Args:
        handler: Called as ``handler(messages, model)`` and returns the response text.
            Ignored if ``responses`` is given.
        responses: A fixed sequence of responses, consumed in order. Running out is an
            error rather than a silent repeat, because a test that accidentally reuses the
            last response proves nothing.
        latency_s: Simulated delay, for exercising concurrency.
        fail_times: Raise this many transient errors before succeeding, to test retry paths.
        reasoning_tokens: Tokens this model "thinks" before writing anything. When set, the
            response is clipped to whatever ``max_tokens`` is left over and ``finish_reason``
            becomes ``length`` -- which is how a reasoning model behaves against a budget
            sized only for the visible answer, and the exact shape of the bug that made a
            real run return chapter headings and nothing else.
    """

    def __init__(
        self,
        handler: Handler | None = None,
        *,
        responses: Sequence[str] | None = None,
        latency_s: float = 0.0,
        fail_times: int = 0,
        prompt_tokens: int = 100,
        completion_tokens: int = 120,
        reasoning_tokens: int = 0,
    ) -> None:
        if handler is None and responses is None:
            handler = echo_translator()
        self._handler = handler
        self._responses = list(responses) if responses is not None else None
        self.latency_s = latency_s
        self.remaining_failures = fail_times
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.calls: list[RecordedCall] = []
        self.max_concurrent = 0
        self._in_flight = 0

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def calls_for(self, purpose: str) -> list[RecordedCall]:
        return [call for call in self.calls if call.purpose == purpose]

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
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": response_format,
        }
        self.calls.append(RecordedCall(list(messages), model, purpose, params))

        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            if self.latency_s:
                await asyncio.sleep(self.latency_s)

            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                raise LLMError(
                    "simulated transient failure",
                    remedy="this is a test fixture",
                    context={"transient": True},
                )

            if self._responses is not None:
                if not self._responses:
                    raise AssertionError(
                        f"FakeLLMClient ran out of scripted responses on call "
                        f"{self.call_count} (model={model}, purpose={purpose})"
                    )
                text = self._responses.pop(0)
            else:
                assert self._handler is not None
                text = self._handler(messages, model)
        finally:
            self._in_flight -= 1

        text, finish_reason, completion_tokens = self._apply_budget(text, max_tokens)
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=min(self.reasoning_tokens, completion_tokens),
            cost=Cost(self.prompt_tokens, completion_tokens, 0.0),
            latency_ms=int(self.latency_s * 1000),
            finish_reason=finish_reason,
            params=params,
        )

    def _apply_budget(self, text: str, max_tokens: int | None) -> tuple[str, str, int]:
        """Spend ``reasoning_tokens`` first, then clip the answer to what is left.

        Mechanical, like every other behaviour here: the same budget always cuts at the
        same character, so a test that passes once passes every time.
        """
        if not self.reasoning_tokens or max_tokens is None:
            return text, "stop", self.completion_tokens

        visible_budget = max_tokens - self.reasoning_tokens
        if visible_budget <= 0:
            return "", "length", max_tokens
        limit = int(visible_budget * _CHARS_PER_TOKEN)
        if len(text) <= limit:
            spent = self.reasoning_tokens + int(len(text) / _CHARS_PER_TOKEN)
            return text, "stop", spent
        return text[:limit], "length", max_tokens

    async def aclose(self) -> None:
        return None


# -- handlers: the well-behaved case ------------------------------------------------


def _requested(messages: list[Message]) -> list[tuple[str, str]]:
    """The (id, text) pairs in the user message of a tagged request."""
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    parsed = parse_segments(user)
    return [(seg_id, parsed.texts.get(seg_id, "")) for seg_id in parsed.order]


def echo_translator(prefix: str = "[t] ") -> Handler:
    """Returns every segment, correctly tagged, prefixed so translation is visible."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        return "\n".join(f'<seg id="{sid}">{prefix}{text}</seg>' for sid, text in pairs)

    return handler


def reversing_translator() -> Handler:
    """Word-reverses each segment: obviously 'translated', still complete."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        return "\n".join(
            f'<seg id="{sid}">{" ".join(reversed(text.split()))}</seg>' for sid, text in pairs
        )

    return handler


# -- handlers: the misbehaviours the brief requires --------------------------------


def dropping_translator(every: int = 10, prefix: str = "[t] ") -> Handler:
    """Silently omits every Nth segment -- acceptance criterion §21.5."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        kept = [
            (sid, text) for index, (sid, text) in enumerate(pairs, start=1) if index % every != 0
        ]
        return "\n".join(f'<seg id="{sid}">{prefix}{text}</seg>' for sid, text in kept)

    return handler


def inventing_translator(prefix: str = "[t] ") -> Handler:
    """Returns every requested segment plus one that was never asked for."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        body = "\n".join(f'<seg id="{sid}">{prefix}{text}</seg>' for sid, text in pairs)
        return body + '\n<seg id="b9999">An entire paragraph the author never wrote.</seg>'

    return handler


def duplicating_translator(prefix: str = "[t] ") -> Handler:
    """Returns the first segment twice."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        lines = [f'<seg id="{sid}">{prefix}{text}</seg>' for sid, text in pairs]
        if lines:
            lines.insert(1, lines[0])
        return "\n".join(lines)

    return handler


def malformed_translator() -> Handler:
    """Emits tags that do not close, and one with no id at all."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        out = []
        for index, (sid, text) in enumerate(pairs):
            if index == 0:
                out.append(f'<seg id="{sid}">{text}')  # never closed
            elif index == 1:
                out.append(f"<seg>{text}</seg>")  # no id
            else:
                out.append(f'<seg id="{sid}">{text}</seg>')
        return "\n".join(out)

    return handler


def refusing_translator() -> Handler:
    """Refuses in prose instead of translating -- the failure the §9 regex must catch."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        return (
            "I'm sorry, but I can't help with translating this content. As an AI, I have "
            "to decline requests involving material of this nature. [Content omitted]"
        )

    return handler


def degenerate_translator(repeats: int = 60) -> Handler:
    """Loops one phrase forever -- the classic sampling collapse."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        loop = " ".join(["the wall of the house"] * repeats)
        return "\n".join(f'<seg id="{sid}">{loop}</seg>' for sid, _ in pairs)

    return handler


def truncating_translator(keep: float = 0.4, prefix: str = "[t] ") -> Handler:
    """Cuts off partway through, as a response that hits the token ceiling does."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        body = "\n".join(f'<seg id="{sid}">{prefix}{text}</seg>' for sid, text in pairs)
        return body[: max(1, int(len(body) * keep))]

    return handler


def passthrough_translator() -> Handler:
    """Returns the source unchanged: the 'forgot to translate' failure."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        return "\n".join(f'<seg id="{sid}">{text}</seg>' for sid, text in pairs)

    return handler


def chatty_translator(prefix: str = "[t] ") -> Handler:
    """Correct segments wrapped in a helpful preamble and a markdown fence."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        body = "\n".join(f'<seg id="{sid}">{prefix}{text}</seg>' for sid, text in pairs)
        return (
            "Here is the translation:\n\n"
            f"```xml\n{body}\n```\n\n"
            "Let me know if you need anything else!"
        )

    return handler


def scripted_by_id(translations: dict[str, str], prefix: str = "") -> Handler:
    """Returns specific text per segment id; ids with no entry are echoed."""

    def handler(messages: list[Message], model: str) -> str:  # noqa: ARG001
        pairs = _requested(messages)
        return "\n".join(
            f'<seg id="{sid}">{translations.get(sid, prefix + text)}</seg>' for sid, text in pairs
        )

    return handler


def failing_then(handler: Handler, *, failures: int) -> Handler:
    """Wrap a handler so the first ``failures`` calls raise a transient error."""
    state = {"left": failures}

    def wrapped(messages: list[Message], model: str) -> str:
        if state["left"] > 0:
            state["left"] -= 1
            raise LLMError("simulated transient failure", context={"transient": True})
        return handler(messages, model)

    return wrapped


# -- fixture files ---------------------------------------------------------------------


@dataclass(slots=True)
class FixtureClient:
    """Replays responses recorded in a JSON fixture file.

    Format::

        {"responses": [{"purpose": "translate", "text": "<seg ...>"}, ...]}

    Kept separate from :class:`FakeLLMClient` so a recorded conversation and a synthetic
    behaviour never get confused for one another in a test.
    """

    path: Path
    responses: list[dict[str, str]] = field(default_factory=list)
    index: int = 0

    def __post_init__(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.responses = list(data["responses"])

    def as_client(self, **kwargs: Any) -> FakeLLMClient:
        return FakeLLMClient(responses=[r["text"] for r in self.responses], **kwargs)


def normalise_whitespace(text: str) -> str:
    """Collapse whitespace, for comparing model output in assertions."""
    return re.sub(r"\s+", " ", text).strip()
