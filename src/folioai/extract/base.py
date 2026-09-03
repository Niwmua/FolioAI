"""The raw layer: what an extractor produces before any cleaning happens.

Deliberately *not* the IR. Extractors emit lines with font size, weight and position,
because heading detection, footnote separation, drop-cap repair and paragraph reflow all
need that geometry and it is gone by the time text is a string. Cleaning consumes this and
produces the IR.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..config import Settings

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class RawSpan:
    """A run of characters sharing one font and size."""

    text: str
    size: float
    font: str = ""
    bold: bool = False
    italic: bool = False
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class RawLine:
    """One visual line. The unit the cleaner reasons about."""

    spans: list[RawSpan]
    bbox: BBox
    page: int
    column: int = 0
    block_index: int = 0

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)

    @property
    def size(self) -> float:
        """Dominant font size: the size covering the most characters on the line."""
        if not self.spans:
            return 0.0
        weights: dict[float, int] = {}
        for span in self.spans:
            weights[round(span.size, 1)] = weights.get(round(span.size, 1), 0) + len(span.text)
        return max(weights.items(), key=lambda kv: kv[1])[0]

    @property
    def max_size(self) -> float:
        return max((span.size for span in self.spans), default=0.0)

    @property
    def bold(self) -> bool:
        """True when most of the line's characters are bold."""
        bold_chars = sum(len(s.text) for s in self.spans if s.bold)
        total = sum(len(s.text) for s in self.spans)
        return total > 0 and bold_chars / total > 0.6

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def bottom(self) -> float:
        return self.bbox[3]

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 0.0)


@dataclass(slots=True)
class RawPage:
    number: int  # 1-based, matching what a reader sees in a PDF viewer
    width: float
    height: float
    lines: list[RawLine] = field(default_factory=list)
    columns: int = 1
    extractor: str = ""

    def line_texts(self) -> list[str]:
        return [line.text for line in self.lines]


@dataclass(slots=True)
class RawImage:
    id: str
    page: int
    bbox: BBox | None = None
    width: int | None = None
    height: int | None = None
    xref: int | None = None


@dataclass(slots=True)
class RawDocument:
    """Everything an extractor found, in reading order."""

    pages: list[RawPage] = field(default_factory=list)
    images: list[RawImage] = field(default_factory=list)
    toc: list[tuple[int, str, int]] = field(default_factory=list)  # (level, title, page)
    metadata: dict[str, Any] = field(default_factory=dict)
    extractor: str = ""
    fallback_pages: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def lines(self) -> Iterator[RawLine]:
        for page in self.pages:
            yield from page.lines

    @property
    def line_count(self) -> int:
        return sum(len(page.lines) for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def body_size(self) -> float:
        """Modal font size across the document: the body text size.

        Weighted by characters, so a title page cannot outvote 300 pages of prose.
        """
        weights: dict[float, int] = {}
        for line in self.lines():
            for span in line.spans:
                key = round(span.size, 1)
                weights[key] = weights.get(key, 0) + len(span.text.strip())
        if not weights:
            return 0.0
        return max(weights.items(), key=lambda kv: kv[1])[0]

    def text(self) -> str:
        return "\n".join(line.text for line in self.lines())


@runtime_checkable
class Extractor(Protocol):
    """What every extraction strategy must provide (brief §4.2)."""

    name: str

    def available(self) -> tuple[bool, str]:
        """Whether this extractor can run here, and if not, what to install."""
        ...

    def extract(
        self, path: Path, settings: Settings, pages: Sequence[int] | None = None
    ) -> RawDocument:
        """Extract the given pages (1-based) or the whole document."""
        ...
