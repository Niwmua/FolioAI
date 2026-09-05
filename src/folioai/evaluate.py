"""The second model: LLM-as-judge (brief §10).

Three things this module refuses to do, all deliberate:

* **It does not let the model report the composite.** The five dimension scores come from
  the model; the weighted total is arithmetic, and models are bad at arithmetic and worse
  at applying a weighting consistently (§10, D-40).
* **It does not let a good average hide an omission.** Any ``critical`` issue, or
  completeness below the floor, fails the segment whatever the composite says (D-41).
* **It does not silently accept unparseable output.** One repair retry, then an error.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
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


class Issue(BaseModel):
    """One defect the judge found."""

    model_config = ConfigDict(extra="ignore")

    segment_id: str
    dimension: Dimension
    severity: IssueSeverity
    source_excerpt: str = Field(default="", max_length=400)
    translation_excerpt: str | None = Field(default=None, max_length=400)
    explanation: str = ""
    suggested_fix: str | None = None


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
                        "source_excerpt": {"type": "string"},
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


def parse_evaluation(text: str) -> Evaluation:
    """Parse the judge's JSON strictly, tolerating a fence or surrounding prose.

    Raises:
        EvaluationError: if nothing valid can be found. The caller does one repair retry.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", candidate, flags=re.DOTALL)

    try:
        return Evaluation.model_validate_json(candidate)
    except (PydanticValidationError, ValueError):
        pass

    match = _JSON_BLOCK_RE.search(candidate)
    if match:
        try:
            return Evaluation.model_validate(json.loads(match.group(0)))
        except (PydanticValidationError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                "The evaluator returned JSON that does not match the expected schema.",
                remedy=(
                    "This usually means the evaluator model ignores structured outputs. Try "
                    "--evaluator-model with a model that supports JSON mode."
                ),
                context={"error": str(exc)[:300], "response_head": candidate[:200]},
            ) from exc

    raise EvaluationError(
        "The evaluator returned no JSON at all.",
        remedy="Try a different --evaluator-model; this one is not following the schema.",
        context={"response_head": candidate[:200]},
    )


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

        Raises:
            EvaluationError: if the response cannot be parsed after one repair retry.
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
            evaluation = parse_evaluation(retry_response.text)
            structured = "repair-retry"

        verdicts = decide(evaluation, result.batch.ids, self.settings)
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
            model=self.settings.models.back_translator,
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
