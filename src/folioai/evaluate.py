"""The second model: LLM-as-judge (brief §10).

Three things this module refuses to do, all deliberate:

* **It does not let the model report the composite.** The five dimension scores come from
  the model; the weighted total is arithmetic, and models are bad at arithmetic and worse
  at applying a weighting consistently (§10, D-40).
* **It does not let a good average hide an omission.** Any ``critical`` issue, or
  completeness below the floor, fails the segment whatever the composite says (D-41).
* **It does not silently accept unparseable output.** One repair retry; if that fails too,
  the batch is marked *unjudged* -- every segment fails, keeps its translation and is
  flagged for review (D-166). What it will not do is quietly treat an unreadable verdict as
  a passing one.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from .errors import EvaluationError
from .glossary import Glossary
from .llm.client import LLMClient, Message
from .logging_setup import get_logger
from .prompts import BACKTRANSLATE_SYSTEM, EVALUATE_SYSTEM, render
from .tags import parse_segments, render_segments

if TYPE_CHECKING:
    from .config import Settings
    from .segment import Batch
    from .translate import BatchTranslation
    from .validate import ValidationReport

log = get_logger(__name__)

Dimension = Literal["completeness", "accuracy", "terminology", "fluency", "formatting"]
IssueSeverity = Literal["critical", "major", "minor"]

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

#: How much of a quoted excerpt is kept. Long enough to identify the passage, short enough
#: that the report and the database stay manageable. Over-long quotes are trimmed, never
#: rejected -- see ``Issue._clip_excerpt``.
EXCERPT_LIMIT = 400


class Issue(BaseModel):
    """One defect the judge found.

    ``severity`` is load-bearing -- a ``critical`` issue fails the segment whatever the
    composite says -- so it stays required. ``dimension`` only labels the issue in the
    report, so a judge that does not supply one leaves it unset rather than having a
    plausible value invented for it.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    segment_id: str
    dimension: Dimension | None = None
    severity: IssueSeverity
    source_excerpt: str = Field(default="", max_length=EXCERPT_LIMIT)
    translation_excerpt: str | None = Field(default=None, max_length=EXCERPT_LIMIT)
    #: Models reach for "description" about as often as "explanation"; both name the same
    #: free-text field, and neither is a claim about the translation.
    explanation: str = Field(
        default="", validation_alias=AliasChoices("explanation", "description")
    )
    suggested_fix: str | None = None

    @field_validator("source_excerpt", "explanation", mode="before")
    @classmethod
    def _null_means_absent(cls, value: Any) -> Any:
        """``null`` and ``""`` say the same thing here: the judge quoted nothing.

        An issue reporting an *addition* has no source excerpt by definition, so a judge
        that sends ``null`` for it is being accurate, not sloppy. Rejecting the batch over
        it threw away the judge's verdict on twenty other segments.
        """
        return "" if value is None else value

    @field_validator("source_excerpt", "translation_excerpt", mode="before")
    @classmethod
    def _clip_excerpt(cls, value: Any) -> Any:
        """Trim an over-long quote rather than rejecting the batch it came in.

        The limit exists so the report stays readable and the database stays small -- a
        presentation concern. Enforcing it by *refusing* the response let a judge that
        quoted a whole paragraph throw away its own verdict on twenty other segments, which
        is a steep price for a long quotation. The excerpt only ever illustrates the issue;
        the severity and the scores, which are what anything acts on, are untouched.
        """
        if isinstance(value, str) and len(value) > EXCERPT_LIMIT:
            return value[: EXCERPT_LIMIT - 1] + "…"
        return value


class SegmentScore(BaseModel):
    """Five dimensions for one segment. ``composite`` is computed, never model-reported."""

    model_config = ConfigDict(extra="ignore")

    segment_id: str
    completeness: int = Field(ge=0, le=100)
    accuracy: int = Field(ge=0, le=100)
    terminology: int = Field(ge=0, le=100)
    fluency: int = Field(ge=0, le=100)
    formatting: int = Field(ge=0, le=100)
    confidence: Literal["high", "medium", "low"] = "medium"
    composite: float = 0.0

    def compute_composite(self, weights: Any) -> float:
        self.composite = round(
            self.completeness * weights.completeness
            + self.accuracy * weights.accuracy
            + self.terminology * weights.terminology
            + self.fluency * weights.fluency
            + self.formatting * weights.formatting,
            2,
        )
        return self.composite


class Evaluation(BaseModel):
    """The judge's whole verdict on one batch."""

    model_config = ConfigDict(extra="ignore")

    scores: list[SegmentScore] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    notes: str | None = None

    def score_for(self, segment_id: str) -> SegmentScore | None:
        return next((s for s in self.scores if s.segment_id == segment_id), None)

    def issues_for(self, segment_id: str) -> list[Issue]:
        return [i for i in self.issues if i.segment_id == segment_id]


#: JSON schema handed to endpoints that support structured outputs (D-42).
EVALUATION_SCHEMA: dict[str, Any] = {
    "name": "translation_evaluation",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "completeness": {"type": "integer", "minimum": 0, "maximum": 100},
                        "accuracy": {"type": "integer", "minimum": 0, "maximum": 100},
                        "terminology": {"type": "integer", "minimum": 0, "maximum": 100},
                        "fluency": {"type": "integer", "minimum": 0, "maximum": 100},
                        "formatting": {"type": "integer", "minimum": 0, "maximum": 100},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": [
                        "segment_id",
                        "completeness",
                        "accuracy",
                        "terminology",
                        "fluency",
                        "formatting",
                    ],
                },
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "dimension": {
                            "type": "string",
                            "enum": [
                                "completeness",
                                "accuracy",
                                "terminology",
                                "fluency",
                                "formatting",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                        "source_excerpt": {"type": ["string", "null"]},
                        "translation_excerpt": {"type": ["string", "null"]},
                        "explanation": {"type": "string"},
                        "suggested_fix": {"type": ["string", "null"]},
                    },
                    "required": ["segment_id", "dimension", "severity", "explanation"],
                },
            },
            "notes": {"type": ["string", "null"]},
        },
        "required": ["scores", "issues"],
    },
}


@dataclass(slots=True)
class SegmentVerdict:
    """The decision for one segment, after weighting and the hard-fail overrides."""

    segment_id: str
    score: SegmentScore | None
    issues: list[Issue] = field(default_factory=list)
    passed: bool = True
    reason: str = ""

    @property
    def composite(self) -> float:
        return self.score.composite if self.score else 0.0


@dataclass(slots=True)
class BatchEvaluation:
    """Verdicts for a whole batch, plus the raw evaluation for the record."""

    evaluation: Evaluation
    verdicts: dict[str, SegmentVerdict]
    evaluator_model: str
    structured_output: str = "json_schema"

    @property
    def failed(self) -> list[str]:
        return [sid for sid, verdict in self.verdicts.items() if not verdict.passed]

    @property
    def mean_composite(self) -> float:
        composites = [v.composite for v in self.verdicts.values() if v.score]
        return round(sum(composites) / len(composites), 2) if composites else 0.0


def _flatten_grouped_issues(payload: Any) -> Any:
    """Accept ``issues`` grouped per segment as well as flat.

    Judges mirror the shape of ``scores`` -- one object per segment -- and nest the issues
    inside it::

        {"segment_id": "b0313", "issues": [{"severity": "minor", "description": "..."}]}

    instead of one flat object per issue. Every field the caller acts on is present in
    both forms, in the same words, so rewriting one into the other moves information
    without adding or removing any:

        {"segment_id": "b0313", "severity": "minor", "description": "..."}

    This is normalisation of a *representation*, the same allowance ``strip_code_fences``
    gets in ``tags.py``, and not the repair of a malformed one. It is deliberately narrow:
    it fires only on an entry that pairs a ``segment_id`` with a list under ``issues``, and
    a nested entry that omits its severity still fails validation rather than acquiring a
    default. Widening it to guess at missing content would defeat the point -- a judge that
    cannot say how bad something is has not judged it.
    """
    if not isinstance(payload, dict):
        return payload
    issues = payload.get("issues")
    if not isinstance(issues, list):
        return payload

    flattened: list[Any] = []
    changed = False
    for entry in issues:
        nested = entry.get("issues") if isinstance(entry, dict) else None
        if isinstance(entry, dict) and isinstance(nested, list) and "segment_id" in entry:
            changed = True
            shared = {k: v for k, v in entry.items() if k != "issues"}
            for inner in nested:
                flattened.append({**shared, **inner} if isinstance(inner, dict) else inner)
        else:
            flattened.append(entry)

    if not changed:
        return payload
    log.info("evaluation_issues_regrouped", grouped=len(issues), flattened=len(flattened))
    return {**payload, "issues": flattened}


def parse_evaluation(text: str) -> Evaluation:
    """Parse the judge's JSON strictly, tolerating a fence or surrounding prose.

    Raises:
        EvaluationError: if nothing valid can be found. The caller does one repair retry.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", candidate, flags=re.DOTALL)

    match = _JSON_BLOCK_RE.search(candidate)
    if match is None:
        raise EvaluationError(
            "The evaluator returned no JSON at all.",
            remedy="Try a different --evaluator-model; this one is not following the schema.",
            context={"response_head": candidate[:200]},
        )

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            "The evaluator's response was not valid JSON.",
            remedy="Try a different --evaluator-model; this one is not following the schema.",
            context={"parse_error": str(exc)[:300], "response_head": candidate[:200]},
        ) from exc

    try:
        return Evaluation.model_validate(_flatten_grouped_issues(payload))
    except (PydanticValidationError, ValueError) as exc:
        raise EvaluationError(
            "The evaluator returned JSON that does not match the expected schema.",
            remedy=(
                "This usually means the evaluator model ignores structured outputs. Try "
                "--evaluator-model with a model that supports JSON mode."
            ),
            context={"schema_error": str(exc)[:300], "response_head": candidate[:200]},
        ) from exc


def decide(
    evaluation: Evaluation, segment_ids: list[str], settings: Settings
) -> dict[str, SegmentVerdict]:
    """Apply the weights and the hard-fail overrides (§10, D-40/D-41)."""
    weights = settings.evaluation.weights
    verdicts: dict[str, SegmentVerdict] = {}

    for segment_id in segment_ids:
        score = evaluation.score_for(segment_id)
        issues = evaluation.issues_for(segment_id)
        if score is not None:
            score.compute_composite(weights)

        verdict = SegmentVerdict(segment_id=segment_id, score=score, issues=issues)

        if score is None:
            # The judge skipped it. Not a pass: an unjudged segment is an unknown one.
            verdict.passed = False
            verdict.reason = "the evaluator returned no score for this segment"
        elif any(issue.severity == "critical" for issue in issues):
            verdict.passed = False
            verdict.reason = "a critical issue was reported"
        elif score.completeness < settings.evaluation.completeness_floor:
            # The override that matters: a segment can score 84 overall having dropped a
            # sentence, and the weighted mean must not be allowed to hide that (§10).
            verdict.passed = False
            verdict.reason = (
                f"completeness {score.completeness} is below the floor of "
                f"{settings.evaluation.completeness_floor}"
            )
        elif score.composite < settings.evaluation.min_score:
            verdict.passed = False
            verdict.reason = (
                f"composite {score.composite:.1f} is below --min-score "
                f"{settings.evaluation.min_score}"
            )

        verdicts[segment_id] = verdict
    return verdicts


def should_evaluate(
    batch: Batch,
    validation: ValidationReport,
    glossary: Glossary,
    settings: Settings,
    *,
    rng: random.Random | None = None,
) -> bool:
    """Whether to spend evaluator tokens on this batch (§10 cost control).

    Below a sample rate of 1.0, four categories are *always* evaluated regardless of the
    dice: chapter openings, anything a deterministic check flagged, anything containing a
    glossary term, and dialogue-heavy batches. Sampling must never skip a flagged segment.
    """
    sample = settings.evaluation.sample
    if sample >= 1.0:
        return True
    if sample <= 0.0:
        return False

    if validation.warnings:
        return True
    if batch.index == 0 or any(unit.block.kind == "heading" for unit in batch.units):
        return True

    source = "\n".join(unit.text for unit in batch.units)
    if glossary.terms and glossary.for_text(source):
        return True

    dialogue = sum(1 for unit in batch.units if unit.block.kind == "dialogue")
    if dialogue > len(batch.units) / 2:
        return True

    return (rng or random).random() < sample


class Evaluator:
    """Runs the judge over translated batches."""

    def __init__(
        self,
        client: LLMClient,
        settings: Settings,
        *,
        source_lang: str,
        target_lang: str,
        style_profile: dict[str, Any] | None = None,
        glossary: Glossary | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.style_profile = style_profile or {}
        self.glossary = glossary or Glossary()

        if settings.models.evaluator == settings.models.translator:
            # §10: warn loudly rather than refuse -- some endpoints serve only one model.
            log.warning(
                "evaluator_matches_translator",
                model=settings.models.evaluator,
                why=(
                    "two instances of one model share their blind spots; this is the main "
                    "failure mode of LLM-as-judge"
                ),
            )

    def build_messages(
        self,
        result: BatchTranslation,
        validation: ValidationReport | None,
    ) -> list[Message]:
        source = "\n".join(unit.text for unit in result.batch.units)
        terms = self.glossary.for_text(source)
        warnings = (
            validation.evaluator_hints()
            if validation is not None and self.settings.evaluation.show_validation_warnings
            else []
        )
        system = render(
            EVALUATE_SYSTEM,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            glossary=[term.model_dump() for term in terms],
            style_profile={
                k: v for k, v in self.style_profile.items() if not isinstance(v, list | dict)
            },
            validation_warnings=warnings,
        )
        body = "\n\n".join(
            f"### {unit.id}\nSOURCE:\n{unit.text}\n\nTRANSLATION:\n"
            f"{result.texts.get(unit.id, '(missing)')}"
            for unit in result.batch.units
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ]

    async def evaluate(
        self,
        result: BatchTranslation,
        validation: ValidationReport | None = None,
    ) -> BatchEvaluation:
        """Judge one translated batch.

        Never raises for a malformed response: after one repair retry the batch comes back
        marked ``structured_output="unparsed"`` with no scores, which ``decide`` turns into
        a failure for every segment (D-166). A judge having a bad minute must not end a
        translation that is 400 pages in.
        """
        messages = self.build_messages(result, validation)
        response = await self.client.complete(
            messages,
            model=self.settings.models.evaluator,
            temperature=self.settings.evaluation.temperature,
            response_format={"type": "json_schema", "json_schema": EVALUATION_SCHEMA},
            purpose="evaluate",
        )

        structured = "json_schema"
        try:
            evaluation = parse_evaluation(response.text)
        except EvaluationError as first_error:
            # One repair retry (D-42): ask again, plainly, for JSON and nothing else.
            log.warning("evaluation_parse_failed", error=first_error.message, retrying=True)
            repair = [
                *messages,
                {"role": "assistant", "content": response.text[:2000]},
                {
                    "role": "user",
                    "content": (
                        "That response could not be parsed. Return ONLY a JSON object with "
                        '"scores" and "issues" keys, no prose and no code fence.'
                    ),
                },
            ]
            retry_response = await self.client.complete(
                repair,
                model=self.settings.models.evaluator,
                temperature=0.0,
                response_format={"type": "json_object"},
                purpose="evaluate-repair",
            )
            try:
                evaluation = parse_evaluation(retry_response.text)
                structured = "repair-retry"
            except EvaluationError as second_error:
                # The judge has now failed twice. Do not take the book down with it (D-166):
                # this batch is *unjudged*, which the rest of the pipeline already has a
                # meaning for -- an unjudged segment is an unknown one, so it fails, keeps
                # its translation and is flagged for review. Nothing is accepted as good
                # without a verdict, and a systematically broken evaluator still trips the
                # circuit breaker, which counts exactly these outcomes.
                log.error(
                    "evaluation_unparseable",
                    batch=result.batch.index,
                    attempt=result.attempt_no,
                    segments=result.batch.size,
                    first_error=first_error.message,
                    second_error=second_error.message,
                    remedy=(
                        "these segments keep their translation and are flagged for review; "
                        "if this repeats, the evaluator model is not following the schema "
                        "-- try --evaluator-model"
                    ),
                )
                evaluation = Evaluation()
                structured = "unparsed"

        verdicts = decide(evaluation, result.batch.ids, self.settings)
        if structured == "unparsed":
            for verdict in verdicts.values():
                verdict.reason = "the evaluator's response could not be parsed"
        batch_evaluation = BatchEvaluation(
            evaluation=evaluation,
            verdicts=verdicts,
            evaluator_model=self.settings.models.evaluator,
            structured_output=structured,
        )
        log.info(
            "batch_evaluated",
            batch=result.batch.index,
            attempt=result.attempt_no,
            segments=len(verdicts),
            failed=len(batch_evaluation.failed),
            mean_composite=batch_evaluation.mean_composite,
            issues=len(evaluation.issues),
            structured_output=structured,
        )
        return batch_evaluation


class BackTranslator:
    """Blind back-translation for the `back-translation` and `both` eval modes (§10).

    Blind means exactly that: this call never sees the original source. Given the source it
    would copy it, and the check would confirm nothing (PLAN §2.6). The job records that
    this separation held, so the report can say so rather than asking to be trusted.
    """

    def __init__(
        self, client: LLMClient, settings: Settings, *, source_lang: str, target_lang: str
    ) -> None:
        self.client = client
        self.settings = settings
        self.source_lang = source_lang
        self.target_lang = target_lang

    async def back_translate(self, translations: dict[str, str]) -> dict[str, str]:
        """Render the *target* text back into the source language, source unseen."""
        if not translations:
            return {}
        system = render(
            BACKTRANSLATE_SYSTEM,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )
        body = render_segments(translations.items())
        response = await self.client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": body}],
            model=self.settings.models.role("back_translator"),
            temperature=0.0,
            purpose="back-translate",
        )
        return parse_segments(response.text).texts


def band_for_back_translation(verdicts: dict[str, SegmentVerdict], settings: Settings) -> list[str]:
    """Segments in the ambiguous band, where the extra signal changes the decision (§10)."""
    low, high = settings.evaluation.both_mode_band
    return [
        sid
        for sid, verdict in verdicts.items()
        if verdict.score is not None and low <= verdict.composite <= high
    ]
