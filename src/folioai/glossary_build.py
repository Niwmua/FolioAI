"""Glossary extraction and review (brief §7).

The data model, injection and adherence checking live in ``glossary.py``; this module is
the part that costs money and talks to a model.

Two properties the brief asks for explicitly, both of which shape the design:

* **Sample across the whole book, not just chapter 1.** A character who arrives on page 200
  is exactly the one whose name will drift, because by then the model has forgotten how it
  rendered the first mention.
* **Cross-check candidates against frequency counts from the raw text.** A model asked for
  proper nouns will happily return a dozen that appear once. Counting occurrences in the
  source is free and throws the noise away before a human has to read it.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ConfigError, LLMError
from .glossary import Glossary, Term, TermKind, count_occurrences
from .llm.client import LLMClient
from .logging_setup import get_logger
from .prompts import render
from .tokens import count_tokens

if TYPE_CHECKING:
    from .config import Settings
    from .ir import Document

log = get_logger(__name__)

GLOSSARY_SYSTEM = "glossary.system.j2"

_VALID_KINDS: frozenset[str] = frozenset(get_args(TermKind))


def coerce_kind(value: str) -> TermKind:
    """Map a model-supplied kind onto the closed set, defaulting rather than failing.

    A glossary is worth more with an unhelpfully labelled term in it than without the
    term at all, so an unrecognised label becomes 'other' instead of an error.
    """
    normalised = value.strip().lower()
    return cast("TermKind", normalised) if normalised in _VALID_KINDS else "other"


#: Terms occurring fewer times than this are noise unless the model marked them a person.
MIN_OCCURRENCES = 3
#: A name is worth keeping even when rare; a coined common noun is usually not.
ALWAYS_KEEP_KINDS = frozenset({"character", "place"})


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    target: str
    kind: str = "other"
    note: str | None = None


class _Candidates(BaseModel):
    model_config = ConfigDict(extra="ignore")

    terms: list[_Candidate] = Field(default_factory=list)


CANDIDATE_SCHEMA: dict[str, Any] = {
    "name": "glossary_candidates",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "character",
                                "place",
                                "organization",
                                "title",
                                "invented",
                                "honorific",
                                "other",
                            ],
                        },
                        "note": {"type": ["string", "null"]},
                    },
                    "required": ["source", "target"],
                },
            }
        },
        "required": ["terms"],
    },
}


@dataclass(slots=True)
class GlossaryDraft:
    """An extracted glossary, with the evidence behind each decision."""

    glossary: Glossary
    rejected: list[tuple[str, int]] = field(default_factory=list)
    samples: int = 0
    cost_usd: float = 0.0

    def summary(self) -> str:
        return (
            f"{len(self.glossary.terms)} term(s) from {self.samples} sample(s); "
            f"{len(self.rejected)} rejected as too rare"
        )


def sample_passages(document: Document, settings: Settings, *, count: int = 10) -> list[str]:
    """Evenly spaced passages spanning the whole book (§7).

    Sampling only the opening would miss every character introduced later -- which is the
    population most at risk of drifting, since nothing in the prompt reminds the model how
    it rendered them the first time.
    """
    blocks = [b for b in document.translatable_blocks() if len(b.text) > 120]
    if not blocks:
        return []

    budget = settings.translation.batch_tokens
    stride = max(1, len(blocks) // count)
    passages: list[str] = []

    for start in range(0, len(blocks), stride):
        chunk: list[str] = []
        tokens = 0
        for block in blocks[start:]:
            block_tokens = count_tokens(block.text)
            if tokens + block_tokens > budget and chunk:
                break
            chunk.append(block.text)
            tokens += block_tokens
        if chunk:
            passages.append("\n\n".join(chunk))
        if len(passages) >= count:
            break
    return passages


def _parse_candidates(text: str) -> list[_Candidate]:
    """Parse one extraction response, tolerating a fence or surrounding prose."""
    import json
    import re

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", candidate, flags=re.DOTALL)
    try:
        return _Candidates.model_validate_json(candidate).terms
    except (ValidationError, ValueError):
        pass
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        log.warning("glossary_response_unparseable", head=candidate[:120])
        return []
    try:
        return _Candidates.model_validate(json.loads(match.group(0))).terms
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        log.warning("glossary_response_invalid", error=str(exc)[:200])
        return []


async def build_glossary(
    document: Document,
    client: LLMClient,
    settings: Settings,
    *,
    target_lang: str,
    existing: Glossary | None = None,
    samples: int = 10,
    min_occurrences: int = MIN_OCCURRENCES,
) -> GlossaryDraft:
    """Extract a glossary from a document.

    Raises:
        LLMError: if every extraction call fails. A single failed sample is logged and the
            rest of the book still produces a glossary.
    """
    existing = existing or Glossary()
    passages = sample_passages(document, settings, count=samples)
    if not passages:
        log.warning("glossary_no_passages", blocks=len(document.blocks))
        return GlossaryDraft(glossary=existing)

    system = render(
        GLOSSARY_SYSTEM,
        source_lang=document.source_lang,
        target_lang=target_lang,
        existing=[term.model_dump() for term in existing.terms],
    )

    found: dict[str, _Candidate] = {}
    cost = 0.0
    failures = 0
    for index, passage in enumerate(passages):
        try:
            response = await client.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": passage},
                ],
                model=settings.models.role("glossary"),
                temperature=0.0,
                response_format={"type": "json_schema", "json_schema": CANDIDATE_SCHEMA},
                purpose="glossary",
            )
        except LLMError as exc:
            # One bad sample must not cost the whole glossary: the other nine still work.
            failures += 1
            log.warning("glossary_sample_failed", sample=index, error=exc.message)
            continue

        cost += response.cost.usd
        for candidate in _parse_candidates(response.text):
            key = candidate.source.strip()
            if key and key not in found:
                found[key] = candidate

    if failures == len(passages):
        raise LLMError(
            "Every glossary extraction call failed.",
            remedy="Check the endpoint and the model named in models.glossary, then retry.",
            context={"samples": len(passages)},
        )

    # Frequency cross-check against the real text, not against the model's confidence.
    texts = [block.text for block in document.translatable_blocks()]
    provisional = [
        Term(
            source=c.source.strip(),
            target=c.target.strip(),
            kind=coerce_kind(c.kind),
            note=c.note,
        )
        for c in found.values()
        if c.source.strip() and c.target.strip()
    ]
    counts = count_occurrences(provisional, texts)

    kept: list[Term] = []
    rejected: list[tuple[str, int]] = []
    for term in provisional:
        occurrences = counts.get(term.source, 0)
        term.occurrences = occurrences
        if occurrences >= min_occurrences or term.kind in ALWAYS_KEEP_KINDS:
            kept.append(term)
        else:
            rejected.append((term.source, occurrences))

    kept.sort(key=lambda t: (-t.occurrences, t.source.lower()))
    merged = Glossary(
        terms=[
            *existing.terms,
            *(t for t in kept if t.source not in {e.source for e in existing.terms}),
        ]
    )

    log.info(
        "glossary_built",
        samples=len(passages),
        candidates=len(provisional),
        kept=len(kept),
        rejected=len(rejected),
        failures=failures,
        cost_usd=round(cost, 6),
    )
    return GlossaryDraft(glossary=merged, rejected=rejected, samples=len(passages), cost_usd=cost)


# -- review ------------------------------------------------------------------------------


def editor_command() -> list[str]:
    """The user's editor, from ``$VISUAL`` / ``$EDITOR``, with a per-platform default."""
    configured = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if configured:
        return shlex.split(configured, posix=os.name != "nt")
    return ["notepad"] if os.name == "nt" else ["vi"]


def open_in_editor(path: Path) -> bool:
    """Open a file in the user's editor and wait. Returns False if that was not possible.

    Never raises: failing to launch an editor is a reason to tell the user the file's path,
    not a reason to abandon a glossary that was just paid for.
    """
    command = [*editor_command(), str(path)]
    try:
        completed = subprocess.run(command, check=False)  # argv list, never a shell string
    except (OSError, ValueError) as exc:
        log.warning("editor_failed", command=command[0], error=str(exc))
        return False
    if completed.returncode != 0:
        log.warning("editor_exited_nonzero", command=command[0], code=completed.returncode)
    return completed.returncode == 0


def review_glossary(glossary: Glossary, path: Path) -> Glossary:
    """Write a glossary, open it for editing, and read back whatever the user saved.

    Raises:
        ConfigError: if the edited file is no longer valid YAML. The original is preserved
            alongside so nothing is lost to a slip of the keyboard.
    """
    glossary.save(path)
    backup = path.with_suffix(".yaml.bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    if not open_in_editor(path):
        return glossary
    try:
        return Glossary.load(path)
    except ConfigError as exc:
        raise ConfigError(
            f"The edited glossary at {path} is not valid: {exc.message}",
            remedy=f"Fix the file, or restore the version saved at {backup}.",
            context={"path": str(path), "backup": str(backup)},
        ) from exc


def write_draft_for_review(draft: GlossaryDraft) -> Path:
    """Write a draft to a temp file, for a preview that is not attached to a job yet."""
    descriptor, name = tempfile.mkstemp(suffix=".glossary.yaml", text=True)
    os.close(descriptor)
    path = Path(name)
    draft.glossary.save(path)
    return path


# -- adherence across the whole book -----------------------------------------------------


@dataclass(slots=True)
class TermUsage:
    """How one term was actually rendered across a finished translation."""

    term: Term
    renderings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def consistent(self) -> bool:
        return len(self.renderings) <= 1

    @property
    def occurrences(self) -> int:
        return sum(len(ids) for ids in self.renderings.values())


def audit_adherence(
    glossary: Glossary,
    sources: dict[str, str],
    translations: dict[str, str],
) -> list[TermUsage]:
    """Find terms rendered inconsistently across the book (§15's glossary summary).

    Per-segment adherence is already checked in ``validate.py``; this is the book-level
    view, which catches the case where every individual segment looked fine but chapter 3
    and chapter 19 disagree with each other.
    """
    from .glossary import target_form_present

    usages: list[TermUsage] = []
    for term in glossary.terms:
        usage = TermUsage(term=term)
        for segment_id, source in sources.items():
            if not term.occurs_in(source):
                continue
            translation = translations.get(segment_id, "")
            key = term.target if target_form_present(term, translation) else "(not found)"
            usage.renderings.setdefault(key, []).append(segment_id)
        if usage.occurrences:
            usages.append(usage)

    inconsistent = [u for u in usages if not u.consistent]
    log.info(
        "glossary_adherence_audited",
        terms=len(usages),
        inconsistent=len(inconsistent),
    )
    return usages
