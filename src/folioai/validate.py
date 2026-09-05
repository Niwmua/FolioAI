"""Deterministic validation (brief §9).

Free, instant, and it catches the majority of real failures. Everything here runs *before*
the evaluator: paying a second model to tell you a response was malformed is the single
most avoidable cost in this pipeline.

Three severities:

* ``critical`` -- retry immediately, do not call the evaluator at all.
* ``warning``  -- pass to the evaluator as a hint to verify.
* ``info``     -- record for the report; never acts on its own.

**The false positives matter more than the true positives.** A check that flags good
translations trains you to ignore the report, at which point the real failures go past
unread too. Every heuristic here is therefore deliberately conservative, and the ones that
cannot be made reliable per segment (sentence counts) are reported in aggregate instead
(PLAN §2.4).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .glossary import Glossary, Term, target_form_present
from .logging_setup import get_logger

if TYPE_CHECKING:
    from .config import Settings
    from .translate import BatchTranslation

log = get_logger(__name__)

Severity = Literal["critical", "warning", "info"]

#: Meta-text that means the model talked *about* the task instead of doing it.
_REFUSAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "refusal",
        re.compile(r"\bI(?:'m| am) (?:sorry|afraid)\b.{0,40}\b(?:can(?:'t|not)|unable)\b", re.I),
    ),
    ("refusal", re.compile(r"\bI can(?:'t|not)\s+(?:help|assist|comply|translate|provide)", re.I)),
    ("refusal", re.compile(r"\bas an AI\b|\bas a language model\b", re.I)),
    ("refusal", re.compile(r"\bI (?:must|have to) decline\b", re.I)),
    ("meta", re.compile(r"^\s*(?:here (?:is|are)|below is) the translation\b", re.I)),
    ("meta", re.compile(r"\[(?:content (?:omitted|removed)|translator'?s? note|note)\]", re.I)),
    ("meta", re.compile(r"^\s*translator'?s note\s*:", re.I)),
    ("meta", re.compile(r"\bI (?:have )?(?:omitted|skipped|summari[sz]ed)\b", re.I)),
]

_FOOTNOTE_RE = re.compile(r"\[\^([A-Za-z0-9_-]+)\]")
_NUMBER_RE = re.compile(r"\d[\d.,/]*")

#: Persian and Arabic-Indic digits, folded to ASCII before numbers are compared. Without
#: it, a correct Persian translation writing 47 as its own numerals looks like a dropped
#: number, and the check fires on every page of the book.
_DIGIT_FOLD = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
#: Sentence enders across the scripts this tool sees. ؟ is the Arabic question mark
#: and ۔ the Urdu full stop; Persian pairs a Latin full stop with an Arabic question mark.
_SENTENCE_RE = re.compile(r"[.!?…。！？؟۔]+(?:\s|$)")

#: Scripts whose token overlap with the source is meaningful evidence of a copy-paste.
_LATIN_LIKE = {"latin", "cyrillic", "greek"}


@dataclass(slots=True)
class Finding:
    """One thing a check noticed."""

    check: str
    severity: Severity
    detail: str
    segment_id: str | None = None

    def describe(self) -> str:
        where = f"{self.segment_id}: " if self.segment_id else ""
        return f"[{self.severity}] {where}{self.detail}"


@dataclass(slots=True)
class ValidationReport:
    """Everything the deterministic checks found for one batch."""

    findings: list[Finding] = field(default_factory=list)

    def add(
        self, check: str, severity: Severity, detail: str, segment_id: str | None = None
    ) -> None:
        self.findings.append(Finding(check, severity, detail, segment_id))

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.findings)

    def for_segment(self, segment_id: str) -> list[Finding]:
        return [f for f in self.findings if f.segment_id == segment_id]

    def failed_segments(self) -> set[str]:
        return {f.segment_id for f in self.critical if f.segment_id}

    def evaluator_hints(self) -> list[str]:
        """Warning-level findings, phrased neutrally for the judge (PLAN §2.7)."""
        return [f.describe() for f in self.warnings]

    def mechanical_problems(self) -> list[str]:
        """Critical findings, phrased for the retry prompt."""
        return [f.describe() for f in self.critical]


class LengthRatioTracker:
    """Learns the target/source length ratio for this job as it goes (§9).

    Ratios differ enormously by language pair -- English to German expands, English to
    Chinese contracts by nearly half -- so a hardcoded band is wrong for most pairs. This
    learns the median from the run itself and only starts flagging once it has enough
    samples to have an opinion worth having.
    """

    def __init__(self, *, tolerance: float = 0.4, min_samples: int = 25) -> None:
        self.ratios: list[float] = []
        self.tolerance = tolerance
        self.min_samples = min_samples

    def record(self, source: str, target: str) -> float | None:
        if len(source.strip()) < 40:  # too short for the ratio to mean anything
            return None
        ratio = len(target) / max(len(source), 1)
        self.ratios.append(ratio)
        return ratio

    @property
    def median(self) -> float | None:
        if len(self.ratios) < self.min_samples:
            return None
        return statistics.median(self.ratios)

    def is_outlier(self, ratio: float) -> bool:
        median = self.median
        if median is None:
            return False
        return abs(ratio - median) > median * self.tolerance


# -- individual checks ----------------------------------------------------------------


def fold_digits(text: str) -> str:
    """Render Persian and Arabic-Indic digits as ASCII, for comparison only."""
    return text.translate(_DIGIT_FOLD)


def check_segment_integrity(result: BatchTranslation, report: ValidationReport) -> None:
    """Every requested id present exactly once, in order, with nothing invented (§9)."""
    for segment_id in result.missing:
        report.add(
            "segment_integrity",
            "critical",
            "segment was not returned at all (omitted by the model)",
            segment_id,
        )
    for segment_id in result.unexpected:
        report.add(
            "segment_integrity",
            "critical",
            "segment id was never requested (invented by the model)",
            segment_id,
        )
    for segment_id in result.parsed.duplicates:
        report.add("segment_integrity", "critical", "segment returned more than once", segment_id)
    if result.parsed.out_of_order(result.batch.ids):
        report.add("segment_integrity", "critical", "segments returned in the wrong order")
    if result.parsed.malformed_openings:
        report.add(
            "segment_integrity",
            "critical",
            f"{result.parsed.malformed_openings} malformed <seg> tag(s) in the response",
        )


def check_empty_output(result: BatchTranslation, report: ValidationReport) -> None:
    for segment_id, text in result.texts.items():
        if not text.strip():
            report.add("empty_output", "critical", "translation is empty", segment_id)


def check_refusal(result: BatchTranslation, report: ValidationReport) -> None:
    """Refusals and meta-commentary (§9).

    Checked in the segment bodies *and* in text outside the tags, because a model that
    refuses usually does so in prose instead of returning segments at all.
    """
    for label, pattern in _REFUSAL_PATTERNS:
        if result.parsed.stray_text and pattern.search(result.parsed.stray_text):
            report.add(
                "refusal",
                "critical",
                f"{label} text outside the segments: {result.parsed.stray_text[:120]!r}",
            )
            break

    for segment_id, text in result.texts.items():
        for label, pattern in _REFUSAL_PATTERNS:
            match = pattern.search(text)
            if match:
                report.add(
                    "refusal",
                    "critical",
                    f"{label} text in the translation: {match.group(0)!r}",
                    segment_id,
                )
                break

    for segment_id, reason in result.blocked.items():
        report.add(
            "blocked",
            "critical",
            f"model declined this segment: {reason[:120]!r}",
            segment_id,
        )


def _repeated_ngram_share(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are repeats. High values mean a sampling collapse."""
    words = text.split()
    if len(words) < n * 3:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def check_degeneration(
    result: BatchTranslation,
    sources: dict[str, str],
    settings: Settings,
    report: ValidationReport,
) -> None:
    """Repeated n-gram loops, or output far longer than the source could justify (§9)."""
    limit = settings.translation.max_completion_ratio
    for segment_id, text in result.texts.items():
        source = sources.get(segment_id, "")
        if source and len(text) > len(source) * limit and len(text) > 200:
            report.add(
                "degeneration",
                "critical",
                f"output is {len(text) / max(len(source), 1):.1f}x the source length",
                segment_id,
            )
            continue
        share = _repeated_ngram_share(text)
        if share > 0.3:
            report.add(
                "degeneration",
                "critical",
                f"{share:.0%} of 5-grams are repeats: the model is looping",
                segment_id,
            )

    if result.response.truncated:
        report.add("truncation", "critical", "the endpoint stopped for length: response is cut off")


def check_length_ratio(
    result: BatchTranslation,
    sources: dict[str, str],
    tracker: LengthRatioTracker,
    report: ValidationReport,
) -> None:
    """Compare each segment's length ratio against the running median (§9)."""
    for segment_id, text in result.texts.items():
        source = sources.get(segment_id, "")
        ratio = tracker.record(source, text)
        if ratio is None:
            continue
        if tracker.is_outlier(ratio):
            median = tracker.median or 0
            report.add(
                "length_ratio",
                "warning",
                f"length ratio {ratio:.2f} against a running median of {median:.2f} "
                f"for this language pair",
                segment_id,
            )


def _script_of(text: str) -> str:
    from .extract.probe import dominant_script

    return dominant_script(text)


def check_passthrough(
    result: BatchTranslation, sources: dict[str, str], report: ValidationReport
) -> None:
    """High token overlap with the source suggests the model copied instead of translating.

    Skipped unless both sides are in the same alphabetic script -- overlap means nothing
    across scripts -- and skipped for proper-noun-heavy segments, where a high overlap is
    the *correct* answer (§9).
    """
    for segment_id, text in result.texts.items():
        source = sources.get(segment_id, "")
        if len(source) < 60 or len(text) < 60:
            continue
        if _script_of(source) not in _LATIN_LIKE or _script_of(source) != _script_of(text):
            continue

        source_words = {w.lower() for w in _WORD_RE.findall(source)}
        target_words = {w.lower() for w in _WORD_RE.findall(text)}
        if not source_words or not target_words:
            continue

        capitalised = sum(1 for w in _WORD_RE.findall(source) if w[:1].isupper())
        if capitalised / max(len(_WORD_RE.findall(source)), 1) > 0.4:
            continue  # a list of names legitimately survives translation nearly intact

        jaccard = len(source_words & target_words) / len(source_words | target_words)
        if jaccard > 0.7:
            report.add(
                "passthrough",
                "warning",
                f"{jaccard:.0%} token overlap with the source: possibly untranslated",
                segment_id,
            )


def check_glossary(
    result: BatchTranslation,
    sources: dict[str, str],
    glossary: Glossary,
    report: ValidationReport,
) -> None:
    """Glossary adherence, allowing for inflection (§9).

    Stem-prefix matching rather than exact equality, or an inflected target language
    produces a wall of false positives: ``der Wärter`` legitimately becomes ``dem Wärter``.
    """
    if not glossary.terms:
        return
    for segment_id, text in result.texts.items():
        source = sources.get(segment_id, "")
        if not source:
            continue
        expected: list[Term] = [term for term in glossary.terms if term.occurs_in(source)]
        for term in expected:
            if not target_form_present(term, text):
                report.add(
                    "glossary",
                    "warning",
                    f"{term.source!r} should be rendered {term.target!r}, which does not "
                    "appear in this segment",
                    segment_id,
                )


def check_numbers(
    result: BatchTranslation, sources: dict[str, str], report: ValidationReport
) -> None:
    """Digits in the source should be represented in the target (§9).

    A hint, never a hard fail: numerals are legitimately spelled out in prose, and some
    languages prefer that. Only fires when several numbers go missing at once.
    """
    for segment_id, text in result.texts.items():
        source = fold_digits(sources.get(segment_id, ""))
        source_numbers = {n.rstrip(".,") for n in _NUMBER_RE.findall(source)}
        if len(source_numbers) < 2:
            continue
        # Folded on both sides: a Persian translation writing 47 in Persian numerals has
        # kept the number, and must not be reported as having dropped it.
        target_numbers = {n.rstrip(".,") for n in _NUMBER_RE.findall(fold_digits(text))}
        missing = source_numbers - target_numbers
        if len(missing) > len(source_numbers) / 2:
            report.add(
                "numbers",
                "warning",
                f"numbers absent from the translation: {', '.join(sorted(missing)[:5])}",
                segment_id,
            )


def check_markup(
    result: BatchTranslation, sources: dict[str, str], report: ValidationReport
) -> None:
    """Emphasis balanced, footnote refs preserved, no stray tags in the body (§9)."""
    for segment_id, text in result.texts.items():
        source = sources.get(segment_id, "")

        if "<seg" in text or "</seg>" in text:
            report.add(
                "markup", "warning", "a <seg> tag leaked into the translation body", segment_id
            )

        for marker in ("**", "`"):
            if text.count(marker) % 2:
                report.add("markup", "warning", f"unbalanced {marker!r} markers", segment_id)

        singles = len(re.findall(r"(?<!\*)\*(?!\*)", text))
        if singles % 2:
            report.add("markup", "warning", "unbalanced '*' emphasis markers", segment_id)

        source_refs = set(_FOOTNOTE_RE.findall(source))
        target_refs = set(_FOOTNOTE_RE.findall(text))
        if source_refs - target_refs:
            report.add(
                "markup",
                "warning",
                f"footnote refs dropped: {', '.join(sorted(source_refs - target_refs))}",
                segment_id,
            )
        if target_refs - source_refs:
            report.add(
                "markup",
                "warning",
                f"footnote refs invented: {', '.join(sorted(target_refs - source_refs))}",
                segment_id,
            )


def sentence_count_delta(source: str, target: str) -> int:
    """Difference in sentence count. Aggregated by the report, never flagged per segment.

    Per segment this fires constantly and legitimately: German compounds clauses that
    English splits, Japanese splits clauses that English compounds. In aggregate, across a
    chapter, a large systematic delta is real signal (PLAN §2.4).
    """
    return len(_SENTENCE_RE.findall(target)) - len(_SENTENCE_RE.findall(source))


# -- entry point ------------------------------------------------------------------------


def validate_batch(
    result: BatchTranslation,
    settings: Settings,
    *,
    glossary: Glossary | None = None,
    length_tracker: LengthRatioTracker | None = None,
) -> ValidationReport:
    """Run every deterministic check over one batch translation.

    Returns:
        A report. ``has_critical`` means retry now and skip the evaluator entirely.
    """
    report = ValidationReport()
    sources = result.batch.source_map()

    check_segment_integrity(result, report)
    check_empty_output(result, report)
    check_refusal(result, report)
    check_degeneration(result, sources, settings, report)

    # Warnings are only meaningful for a structurally sound response; running them over a
    # mangled one produces noise about text that is about to be thrown away and retried.
    if not report.has_critical:
        if length_tracker is not None:
            check_length_ratio(result, sources, length_tracker, report)
        check_passthrough(result, sources, report)
        if glossary is not None:
            check_glossary(result, sources, glossary, report)
        check_numbers(result, sources, report)
        check_markup(result, sources, report)

    if report.findings:
        log.info(
            "validation_findings",
            batch=result.batch.index,
            attempt=result.attempt_no,
            critical=len(report.critical),
            warnings=len(report.warnings),
            checks=sorted({f.check for f in report.findings}),
        )
    return report


def aggregate_sentence_deltas(
    pairs: Sequence[tuple[str, str, str]],
) -> dict[str, float]:
    """Per-chapter mean sentence-count delta, for the quality report (PLAN §2.4).

    Args:
        pairs: ``(chapter_id, source_text, target_text)`` triples.
    """
    by_chapter: dict[str, list[int]] = {}
    for chapter_id, source, target in pairs:
        by_chapter.setdefault(chapter_id, []).append(sentence_count_delta(source, target))
    return {
        chapter: round(statistics.mean(deltas), 2)
        for chapter, deltas in by_chapter.items()
        if deltas
    }
