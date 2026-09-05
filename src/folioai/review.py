"""The human review loop (brief §15).

Walk the flagged segments, and for each one: accept it, edit it in ``$EDITOR``, or send it
back for another translation with an instruction you type in.

An edit is stored as a **new attempt** with ``model: human`` rather than overwriting the
segment. Nothing is lost, the history stays readable ("the model said X, the judge said Y,
the human wrote Z"), and re-exporting picks up the edit like any other attempt.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .logging_setup import get_logger
from .store import JobStore, SegmentRecord

log = get_logger(__name__)

HUMAN_MODEL = "human"
Action = Literal["accept", "edit", "retranslate", "skip", "quit"]


@dataclass(slots=True)
class ReviewItem:
    """One segment presented for review, with everything needed to judge it."""

    segment: SegmentRecord
    issues: list[dict[str, object]] = field(default_factory=list)
    attempts: list[dict[str, object]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.segment.segment_id

    @property
    def score(self) -> float | None:
        return self.segment.final_score


@dataclass(slots=True)
class ReviewOutcome:
    """What the review session did, for the closing summary."""

    accepted: int = 0
    edited: int = 0
    retranslated: int = 0
    skipped: int = 0
    quit_early: bool = False

    @property
    def touched(self) -> int:
        return self.accepted + self.edited + self.retranslated


def collect_items(
    store: JobStore, job_id: str, *, max_score: float | None = None
) -> list[ReviewItem]:
    """Segments worth a human's attention, worst first.

    Worst first because review time runs out before the queue does, and the segment most
    likely to be wrong should not be the one nobody reached.
    """
    import json

    items: list[ReviewItem] = []
    for segment in store.list_segments(job_id):
        if not segment.final_text:
            continue
        below = max_score is not None and (
            segment.final_score is None or segment.final_score <= max_score
        )
        if not (segment.needs_review or below):
            continue

        rows = store.conn.execute(
            """
            SELECT a.attempt_no, a.model, a.output_text, e.composite, e.issues_json
            FROM attempts a
            LEFT JOIN evaluations e ON e.attempt_id = a.id
            WHERE a.job_id = ? AND a.segment_id = ?
            ORDER BY a.attempt_no
            """,
            (job_id, segment.segment_id),
        ).fetchall()

        attempts = [
            {
                "attempt_no": row["attempt_no"],
                "model": row["model"],
                "text": row["output_text"] or "",
                "composite": row["composite"],
            }
            for row in rows
        ]
        issues: list[dict[str, object]] = []
        for row in rows:
            if row["issues_json"]:
                issues.extend(json.loads(row["issues_json"]))

        items.append(ReviewItem(segment=segment, issues=issues, attempts=attempts))

    items.sort(key=lambda item: (item.score if item.score is not None else -1, item.id))
    return items


def edit_text(text: str, *, suffix: str = ".txt") -> str | None:
    """Open text in ``$EDITOR`` and return what was saved, or ``None`` if unchanged.

    Returns ``None`` rather than the original so the caller can tell "the user changed
    nothing" from "the user rewrote it identically" -- only the first should skip a write.
    """
    from .glossary_build import open_in_editor

    descriptor, name = tempfile.mkstemp(suffix=suffix, text=True)
    import os

    os.close(descriptor)
    path = Path(name)
    try:
        path.write_text(text, encoding="utf-8")
        if not open_in_editor(path):
            return None
        edited = path.read_text(encoding="utf-8")
        return edited if edited.strip() != text.strip() else None
    finally:
        path.unlink(missing_ok=True)


def record_human_edit(
    store: JobStore,
    job_id: str,
    segment_id: str,
    text: str,
    *,
    note: str = "",
) -> int:
    """Store a human edit as a new attempt and make it the segment's final text.

    The attempt row is what makes this non-destructive: the model's version stays in the
    database, so a later question about why a segment reads the way it does is answerable.
    """
    existing = store.list_attempts(job_id, segment_id)
    attempt_no = (max((row["attempt_no"] for row in existing), default=0)) + 1

    row_id = store.record_attempt(
        job_id=job_id,
        segment_id=segment_id,
        attempt_no=attempt_no,
        model=HUMAN_MODEL,
        params={"note": note} if note else {},
        output_text=text,
        cost_usd=0.0,
    )
    store.finalize_segment(
        job_id,
        segment_id,
        final_text=text,
        final_score=None,  # a human edit is not scored; it is the standard, not a candidate
        needs_review=False,
        status="done",
    )
    log.info("human_edit_recorded", job=job_id, segment=segment_id, attempt=attempt_no)
    return row_id


def mark_accepted(store: JobStore, job_id: str, segment_id: str) -> None:
    """Clear the review flag without changing the text."""
    segment = store.get_segment(job_id, segment_id)
    if segment is None or segment.final_text is None:
        return
    store.finalize_segment(
        job_id,
        segment_id,
        final_text=segment.final_text,
        final_score=segment.final_score,
        needs_review=False,
        status="done",
    )


def queue_for_retranslation(
    store: JobStore, job_id: str, segment_id: str, instruction: str
) -> None:
    """Put a segment back in the queue with an extra instruction for the next attempt.

    The instruction rides on a ``human`` attempt row rather than a separate table: the retry
    prompt already reads prior attempts, so this is the one place the orchestrator will
    naturally look.
    """
    existing = store.list_attempts(job_id, segment_id)
    attempt_no = (max((row["attempt_no"] for row in existing), default=0)) + 1
    store.record_attempt(
        job_id=job_id,
        segment_id=segment_id,
        attempt_no=attempt_no,
        model=HUMAN_MODEL,
        params={"instruction": instruction, "action": "retranslate"},
        output_text=None,
        cost_usd=0.0,
    )
    store.set_segment_status(job_id, [segment_id], "pending")
    log.info("segment_queued_for_retranslation", job=job_id, segment=segment_id)


def human_instructions(store: JobStore, job_id: str, segment_id: str) -> list[str]:
    """Instructions a reviewer left for this segment, for the retry prompt to honour."""
    import json

    rows = store.conn.execute(
        """
        SELECT params_json FROM attempts
        WHERE job_id = ? AND segment_id = ? AND model = ?
        ORDER BY attempt_no
        """,
        (job_id, segment_id, HUMAN_MODEL),
    ).fetchall()
    instructions = []
    for row in rows:
        params = json.loads(row["params_json"] or "{}")
        if instruction := params.get("instruction"):
            instructions.append(str(instruction))
    return instructions


def run_review(
    store: JobStore,
    job_id: str,
    items: Sequence[ReviewItem],
    *,
    present: Callable[[ReviewItem, int, int], Action],
    ask_instruction: Callable[[ReviewItem], str],
) -> ReviewOutcome:
    """Drive a review session.

    The terminal interaction is injected (``present``, ``ask_instruction``) so the loop
    itself is testable without a TTY, and so a future web UI can reuse it unchanged.
    """
    outcome = ReviewOutcome()
    for index, item in enumerate(items, start=1):
        action = present(item, index, len(items))

        if action == "quit":
            outcome.quit_early = True
            break
        if action == "skip":
            outcome.skipped += 1
            continue
        if action == "accept":
            mark_accepted(store, job_id, item.id)
            outcome.accepted += 1
            continue
        if action == "edit":
            edited = edit_text(item.segment.final_text or "")
            if edited is None:
                outcome.skipped += 1
                continue
            record_human_edit(store, job_id, item.id, edited.strip())
            outcome.edited += 1
            continue
        if action == "retranslate":
            instruction = ask_instruction(item)
            if not instruction.strip():
                outcome.skipped += 1
                continue
            queue_for_retranslation(store, job_id, item.id, instruction.strip())
            outcome.retranslated += 1

    log.info(
        "review_session_complete",
        job=job_id,
        accepted=outcome.accepted,
        edited=outcome.edited,
        retranslated=outcome.retranslated,
        skipped=outcome.skipped,
    )
    return outcome
