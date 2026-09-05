"""Glossary: the terms whose rendering must not drift (brief §7).

This module owns the data model, persistence and *injection*. Extraction (asking a model to
find the proper nouns) and the review loop arrive in milestone 6; the parts translation
depends on are here so the engine never has to grow a second way of doing this.

Injection is deliberately narrow: only the terms that actually occur in the batch, plus
every ``locked`` term, go into the prompt. A 200-entry glossary pasted into every call costs
real money per batch and buries the handful of terms that matter for the paragraph in hand.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigError
from .logging_setup import get_logger

log = get_logger(__name__)

TermKind = Literal["character", "place", "organization", "title", "invented", "honorific", "other"]


class Term(BaseModel):
    """One glossary entry."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: TermKind = "other"
    note: str | None = None
    locked: bool = False
    occurrences: int = 0

    def pattern(self) -> re.Pattern[str]:
        """Word-boundary matcher for the source term.

        Cached per call rather than per instance because a glossary is rebuilt rarely and
        matched often enough that correctness matters more than the microseconds.
        """
        return re.compile(rf"(?<!\w){re.escape(self.source)}(?!\w)", re.IGNORECASE)

    def occurs_in(self, text: str) -> bool:
        return bool(self.pattern().search(text))


class Glossary(BaseModel):
    """The whole term list for one job."""

    model_config = ConfigDict(extra="forbid")

    terms: list[Term] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[Term]:  # type: ignore[override]
        return iter(self.terms)

    @property
    def locked(self) -> list[Term]:
        return [term for term in self.terms if term.locked]

    def for_text(self, text: str) -> list[Term]:
        """Terms to inject for a given batch: those present, plus all locked terms (§7)."""
        present = [term for term in self.terms if term.occurs_in(text)]
        seen = {term.source for term in present}
        return present + [term for term in self.locked if term.source not in seen]

    def sorted_by_length(self) -> list[Term]:
        """Longest first, so 'Ravenscroft Manor' is matched before 'Ravenscroft'."""
        return sorted(self.terms, key=lambda term: len(term.source), reverse=True)

    # -- persistence ---------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "terms": [
                term.model_dump(exclude_none=True, exclude_defaults=False) for term in self.terms
            ]
        }
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> Glossary:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Could not parse the glossary at {path}.",
                remedy="Fix the YAML syntax error below, or delete the file to rebuild it.",
                context={"path": str(path), "error": str(exc)},
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigError(
                f"{path} must contain a mapping with a 'terms:' key.",
                remedy="See config/profiles for the expected shape, or rebuild the glossary.",
            )
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError(
                f"The glossary at {path} is not valid.",
                remedy="Each entry needs at least 'source' and 'target'.",
                context={"path": str(path), "error": str(exc)},
            ) from exc

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, object]]) -> Glossary:
        """Build from ``store.list_glossary()`` rows."""
        return cls(terms=[Term.model_validate(row) for row in rows])

    def to_rows(self) -> list[dict[str, object]]:
        return [term.model_dump() for term in self.terms]


def count_occurrences(terms: Sequence[Term], texts: Iterable[str]) -> dict[str, int]:
    """Count how often each term appears across the book.

    Used to filter one-off noise out of an extracted candidate list (§7) and to populate the
    ``occurrences`` column the report shows.
    """
    counts = dict.fromkeys((term.source for term in terms), 0)
    compiled = [(term.source, term.pattern()) for term in terms]
    for text in texts:
        for source, pattern in compiled:
            found = len(pattern.findall(text))
            if found:
                counts[source] += found
    return counts


def stem_prefix(word: str, *, keep: float = 0.7) -> str:
    """A crude stem: the leading fraction of a word.

    Inflected languages decline the target form, so an exact-string adherence check drowns
    in false positives -- ``der Wärter`` becomes ``dem Wärter``, ``des Wärters``. Comparing
    a prefix catches the real violations without flagging German grammar (§9).
    """
    if len(word) <= 4:
        return word.lower()
    return word.lower()[: max(4, int(len(word) * keep))]


#: Words this short are articles and particles -- der/dem/den, le/la, el, il, de, du. They
#: inflect independently of the term they attach to and carry no terminological weight.
FUNCTION_WORD_LENGTH = 3


def target_form_present(term: Term, translation: str) -> bool:
    """Whether a term's target rendering appears, allowing for inflection.

    Every *content* word of the target form must appear in stem-prefix form. Requiring all
    of them rather than any keeps multi-word terms honest -- 'Herrenhaus Ravenscroft' is not
    satisfied by 'Ravenscroft' alone -- while ignoring short function words, because a
    glossary entry of "der Wärter" is satisfied by "dem Wärter": German declines the article
    and that is not a terminology violation.
    """
    haystack = translation.lower()
    words = [w for w in re.findall(r"[^\W\d_]+", term.target) if w]
    if not words:
        return True
    content = [w for w in words if len(w) > FUNCTION_WORD_LENGTH]
    # A target made entirely of short words (common in CJK) has no function words to drop.
    required = content or words
    return all(stem_prefix(word) in haystack for word in required)
