"""The ``<seg>`` tag protocol (brief §6).

Every batch goes out with each block wrapped in an ID tag, and the model must return the
same tags, in the same order, exactly once each::

    <seg id="b0142">Source paragraph text.</seg>
    <seg id="b0143">Next source paragraph.</seg>

This is what makes omission free to detect. If the model drops a paragraph, ``b0143`` is
simply missing from the response -- no second model, no judgement call, no cost.

Parsing is strict and *descriptive*: this module reports exactly what it found, including
text outside the tags and malformed openings, and leaves the verdict to ``validate.py``.
Parsing that quietly repairs a malformed response is the one thing that would defeat the
whole design, because the repair is indistinguishable from the model having got it right.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "TAG_TOKENS",
    "ParsedSegments",
    "extract_segment_ids",
    "parse_segments",
    "render_segments",
    "strip_code_fences",
    "tag_overhead",
]

#: Token cost of one ``<seg id="...."></seg>`` wrapper, measured at 10-11.2 across a real
#: book and rounded up. Small per segment and easy to forget, which is exactly the problem:
#: it is charged twice -- once in the prompt and again in the response, because the model
#: has to reproduce every wrapper -- and it scales with segment *count*, not with prose. A
#: chapter of 237 one-line contents entries carries 2,607 tokens of markup around 844 tokens
#: of text, so a budget that counts only the text is wrong by 300%.
TAG_TOKENS = 12


def tag_overhead(segments: int) -> int:
    """Token cost of wrapping ``segments`` blocks in the tag protocol."""
    return segments * TAG_TOKENS


# The body may not contain another "<seg" opening. Without that guard, an unclosed tag
# swallows everything up to the *next* segment's closing tag: the dropped segment is still
# detected as missing, but the surviving one silently acquires the other's text, which is a
# plausible-looking wrong answer -- the worst possible failure for this pipeline.
_SEG_RE = re.compile(
    r"<seg\s+id\s*=\s*[\"'](?P<id>[^\"']+)[\"'](?P<attrs>[^>]*)>"
    r"(?P<body>(?:(?!<seg\b).)*?)</seg\s*>",
    re.DOTALL | re.IGNORECASE,
)
_SELF_CLOSING_RE = re.compile(
    r"<seg\s+id\s*=\s*[\"'](?P<id>[^\"']+)[\"'](?P<attrs>[^>]*?)/>",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(r"status\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_ANY_OPEN_RE = re.compile(r"<seg\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)


@dataclass(slots=True)
class ParsedSegments:
    """What a model actually returned, described without judgement."""

    order: list[str] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
    stray_text: str = ""
    malformed_openings: int = 0

    @property
    def ids(self) -> set[str]:
        return set(self.order)

    def missing(self, requested: Sequence[str]) -> list[str]:
        """Requested ids that never came back -- the omission signal (§9)."""
        found = self.ids
        return [seg_id for seg_id in requested if seg_id not in found]

    def unexpected(self, requested: Sequence[str]) -> list[str]:
        """Ids the model invented, which are as serious as ids it dropped."""
        wanted = set(requested)
        return [seg_id for seg_id in self.order if seg_id not in wanted]

    def out_of_order(self, requested: Sequence[str]) -> bool:
        """True when the returned ids are the right set in the wrong sequence."""
        wanted = [seg_id for seg_id in requested if seg_id in self.ids]
        returned = [seg_id for seg_id in self.order if seg_id in set(requested)]
        return wanted != returned


def render_segments(segments: Iterable[tuple[str, str]]) -> str:
    """Render ``(id, text)`` pairs into the wire format sent as the user message."""
    return "\n".join(f'<seg id="{seg_id}">{text}</seg>' for seg_id, text in segments)


def extract_segment_ids(rendered: str) -> list[str]:
    """Ids present in a rendered batch, in order. Used by the fake client and tests."""
    return [match.group("id") for match in _SEG_RE.finditer(rendered)]


def strip_code_fences(text: str) -> str:
    """Unwrap a response the model wrapped in a markdown fence.

    This is the one normalisation allowed before parsing, and it is not a repair: a fence
    adds no content and hides none, so unwrapping it cannot mask an omission. Anything
    else -- missing tags, mangled ids, prose around the segments -- is reported, not fixed.
    """
    match = _FENCE_RE.match(text)
    return match.group("body") if match else text


def parse_segments(text: str) -> ParsedSegments:
    """Parse a model response into segments.

    Returns:
        A description of what was found: order, texts, blocked segments, duplicate ids,
        text outside any tag, and the count of ``<seg`` openings that failed to parse.
    """
    result = ParsedSegments()
    body = strip_code_fences(text)

    consumed: list[tuple[int, int]] = []

    for match in _SEG_RE.finditer(body):
        seg_id = match.group("id").strip()
        status = _STATUS_RE.search(match.group("attrs") or "")
        content = match.group("body")
        consumed.append((match.start(), match.end()))

        if seg_id in result.texts or seg_id in result.blocked:
            result.duplicates.append(seg_id)
            continue

        result.order.append(seg_id)
        if status and status.group(1).lower() == "blocked":
            result.blocked[seg_id] = content.strip()
        else:
            result.texts[seg_id] = content.strip()

    for match in _SELF_CLOSING_RE.finditer(body):
        seg_id = match.group("id").strip()
        if seg_id in result.texts or seg_id in result.blocked:
            result.duplicates.append(seg_id)
            continue
        consumed.append((match.start(), match.end()))
        result.order.append(seg_id)
        status = _STATUS_RE.search(match.group("attrs") or "")
        result.blocked[seg_id] = (status.group(1) if status else "blocked").strip()

    # Anything outside a well-formed tag: a preamble, an apology, a commentary paragraph,
    # or the text of a segment whose tag failed to close.
    consumed.sort()
    leftovers: list[str] = []
    cursor = 0
    for start, end in consumed:
        if start > cursor:
            leftovers.append(body[cursor:start])
        cursor = max(cursor, end)
    leftovers.append(body[cursor:])
    result.stray_text = "\n".join(part.strip() for part in leftovers if part.strip()).strip()

    total_openings = len(_ANY_OPEN_RE.findall(body))
    result.malformed_openings = max(0, total_openings - len(consumed))
    return result
