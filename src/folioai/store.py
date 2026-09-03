"""SQLite persistence. One database per job, WAL mode, explicit SQL, no ORM.

The schema is brief §12 verbatim plus a ``schema_version`` table (D-22), because migrating a
database whose version you have to guess is a bad afternoon. Transactions are per batch rather
than per row (D-24) so the resume boundary lines up with the pipeline's unit of work: a batch
either landed completely or not at all.

Nothing here knows about LLMs or PDFs. It stores rows and answers questions about them.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from .errors import StoreError
from .paths import job_db_path, job_dir

SCHEMA_VERSION = 1

SegmentStatus = Literal["pending", "translating", "evaluating", "done", "failed", "review"]
JobStatus = Literal["created", "extracted", "translating", "completed", "failed", "cancelled"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id                 TEXT PRIMARY KEY,
    source_path        TEXT NOT NULL,
    source_sha256      TEXT NOT NULL,
    source_lang        TEXT,
    target_lang        TEXT,
    config_json        TEXT NOT NULL,
    status             TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    total_segments     INTEGER NOT NULL DEFAULT 0,
    completed_segments INTEGER NOT NULL DEFAULT 0,
    cost_usd           REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS segments (
    job_id         TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    segment_id     TEXT    NOT NULL,
    chapter_id     TEXT,
    ordinal        INTEGER NOT NULL,
    kind           TEXT    NOT NULL,
    source_text    TEXT    NOT NULL,
    final_text     TEXT,
    final_score    REAL,
    status         TEXT    NOT NULL DEFAULT 'pending',
    needs_review   INTEGER NOT NULL DEFAULT 0,
    attempts_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job_id, segment_id)
);
CREATE INDEX IF NOT EXISTS idx_segments_status  ON segments(job_id, status);
CREATE INDEX IF NOT EXISTS idx_segments_ordinal ON segments(job_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_segments_chapter ON segments(job_id, chapter_id);

CREATE TABLE IF NOT EXISTS attempts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            TEXT    NOT NULL,
    segment_id        TEXT    NOT NULL,
    attempt_no        INTEGER NOT NULL,
    model             TEXT    NOT NULL,
    params_json       TEXT    NOT NULL,
    output_text       TEXT,
    latency_ms        INTEGER,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    created_at        TEXT    NOT NULL,
    FOREIGN KEY (job_id, segment_id) REFERENCES segments(job_id, segment_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_attempts_segment ON attempts(job_id, segment_id, attempt_no);

CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id      INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    evaluator_model TEXT    NOT NULL,
    scores_json     TEXT    NOT NULL,
    issues_json     TEXT    NOT NULL,
    composite       REAL    NOT NULL,
    passed          INTEGER NOT NULL,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_attempt ON evaluations(attempt_id);

CREATE TABLE IF NOT EXISTS glossary_terms (
    job_id      TEXT    NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    source      TEXT    NOT NULL,
    target      TEXT    NOT NULL,
    kind        TEXT,
    locked      INTEGER NOT NULL DEFAULT 0,
    note        TEXT,
    occurrences INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job_id, source)
);

CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts                TEXT NOT NULL,
    model             TEXT NOT NULL,
    endpoint          TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_usage_job ON usage(job_id, ts);

CREATE TABLE IF NOT EXISTS cache (
    prompt_fingerprint TEXT PRIMARY KEY,
    model              TEXT NOT NULL,
    output_text        TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
"""


def utcnow() -> str:
    """ISO-8601 UTC timestamp, the only time format stored anywhere."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class JobRecord:
    id: str
    source_path: str
    source_sha256: str
    source_lang: str | None
    target_lang: str | None
    status: str
    created_at: str
    updated_at: str
    total_segments: int
    completed_segments: int
    cost_usd: float
    config: dict[str, Any]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JobRecord:
        return cls(
            id=row["id"],
            source_path=row["source_path"],
            source_sha256=row["source_sha256"],
            source_lang=row["source_lang"],
            target_lang=row["target_lang"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            total_segments=row["total_segments"],
            completed_segments=row["completed_segments"],
            cost_usd=row["cost_usd"],
            config=json.loads(row["config_json"]),
        )

    @property
    def progress(self) -> float:
        if self.total_segments == 0:
            return 0.0
        return self.completed_segments / self.total_segments


@dataclass(slots=True)
class SegmentRecord:
    segment_id: str
    chapter_id: str | None
    ordinal: int
    kind: str
    source_text: str
    final_text: str | None
    final_score: float | None
    status: str
    needs_review: bool
    attempts_count: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SegmentRecord:
        return cls(
            segment_id=row["segment_id"],
            chapter_id=row["chapter_id"],
            ordinal=row["ordinal"],
            kind=row["kind"],
            source_text=row["source_text"],
            final_text=row["final_text"],
            final_score=row["final_score"],
            status=row["status"],
            needs_review=bool(row["needs_review"]),
            attempts_count=row["attempts_count"],
        )


class JobStore:
    """Repository over one job's SQLite database.

    Usable as a context manager. Opening a database creates it if absent; opening one from a
    future schema version is an error rather than a corruption risk.
    """

    def __init__(self, path: Path, *, create: bool = True) -> None:
        self.path = path
        if not create and not path.exists():
            raise StoreError(
                f"No job database at {path}.",
                remedy="Run 'folioai jobs list' to see the jobs that do exist.",
                context={"path": str(path)},
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
        except sqlite3.Error as exc:
            raise StoreError(
                f"Could not open the job database at {path}: {exc}",
                remedy="Check the file is readable and not held open by another process.",
                context={"path": str(path)},
            ) from exc
        self.conn.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    # -- lifecycle ---------------------------------------------------------------

    def _configure(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")

    def _migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] if row and row["v"] is not None else None
        if current is None:
            self.conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, utcnow()),
            )
        elif current > SCHEMA_VERSION:
            raise StoreError(
                f"Job database at {self.path} was written by a newer folioai "
                f"(schema v{current}; this build understands v{SCHEMA_VERSION}).",
                remedy="Upgrade folioai, or start a fresh job against the same PDF.",
                context={"path": str(self.path), "found": current, "expected": SCHEMA_VERSION},
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Batch-scoped transaction (D-24). Commits on success, rolls back on any exception."""
        self.conn.execute("BEGIN")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -- jobs --------------------------------------------------------------------

    def create_job(
        self,
        *,
        job_id: str,
        source_path: Path,
        source_sha256: str,
        config: dict[str, Any],
        source_lang: str | None = None,
        target_lang: str | None = None,
        status: str = "created",
    ) -> JobRecord:
        """Insert a job, or return the existing one for this id (resume-safe)."""
        existing = self.get_job(job_id)
        if existing is not None:
            return existing
        now = utcnow()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, source_path, source_sha256, source_lang, target_lang,
                                  config_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    str(source_path),
                    source_sha256,
                    source_lang,
                    target_lang,
                    json.dumps(config, ensure_ascii=False, default=str),
                    status,
                    now,
                    now,
                ),
            )
        job = self.get_job(job_id)
        assert job is not None  # just inserted
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return JobRecord.from_row(row) if row else None

    def list_jobs(self) -> list[JobRecord]:
        rows = self.conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [JobRecord.from_row(row) for row in rows]

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update whitelisted job columns. Unknown columns raise rather than silently no-op."""
        allowed = {
            "source_lang",
            "target_lang",
            "status",
            "total_segments",
            "completed_segments",
            "cost_usd",
            "config_json",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise StoreError(
                f"Cannot update unknown job column(s): {', '.join(sorted(unknown))}.",
                remedy="This is a bug in folioai; the column list is in store.update_job.",
            )
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values: list[Any] = [*fields.values(), utcnow(), job_id]
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments}, updated_at = ? WHERE id = ?",
                values,
            )

    def delete_job(self, job_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    # -- segments ----------------------------------------------------------------

    def upsert_segments(self, job_id: str, segments: Sequence[SegmentRecord]) -> int:
        """Insert segments, leaving any already-translated row untouched.

        Resume safety: re-running extraction on a partially translated job must not wipe
        completed work, so an existing ``segment_id`` keeps its ``final_text`` and status.
        Returns the number of newly inserted rows.
        """
        inserted = 0
        with self.transaction() as conn:
            for seg in segments:
                cur = conn.execute(
                    """
                    INSERT INTO segments (job_id, segment_id, chapter_id, ordinal, kind,
                                          source_text, final_text, final_score, status,
                                          needs_review, attempts_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id, segment_id) DO UPDATE SET
                        chapter_id  = excluded.chapter_id,
                        ordinal     = excluded.ordinal,
                        kind        = excluded.kind,
                        source_text = excluded.source_text
                    """,
                    (
                        job_id,
                        seg.segment_id,
                        seg.chapter_id,
                        seg.ordinal,
                        seg.kind,
                        seg.source_text,
                        seg.final_text,
                        seg.final_score,
                        seg.status,
                        int(seg.needs_review),
                        seg.attempts_count,
                    ),
                )
                inserted += 1 if cur.rowcount and cur.lastrowid is not None else 0
            conn.execute(
                """
                UPDATE jobs SET total_segments = (
                    SELECT COUNT(*) FROM segments WHERE job_id = ?
                ), updated_at = ? WHERE id = ?
                """,
                (job_id, utcnow(), job_id),
            )
        return inserted

    def get_segment(self, job_id: str, segment_id: str) -> SegmentRecord | None:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE job_id = ? AND segment_id = ?", (job_id, segment_id)
        ).fetchone()
        return SegmentRecord.from_row(row) if row else None

    def list_segments(
        self, job_id: str, *, status: str | None = None, chapter_id: str | None = None
    ) -> list[SegmentRecord]:
        sql = "SELECT * FROM segments WHERE job_id = ?"
        params: list[Any] = [job_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        if chapter_id is not None:
            sql += " AND chapter_id = ?"
            params.append(chapter_id)
        sql += " ORDER BY ordinal"
        return [SegmentRecord.from_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def pending_segments(self, job_id: str) -> list[SegmentRecord]:
        """Everything ``resume`` must pick up: never-started and previously failed work.

        ``translating``/``evaluating`` rows are included because a process killed mid-flight
        leaves them there; they are unfinished by definition.
        """
        rows = self.conn.execute(
            """
            SELECT * FROM segments
            WHERE job_id = ? AND status IN ('pending', 'failed', 'translating', 'evaluating')
            ORDER BY ordinal
            """,
            (job_id,),
        ).fetchall()
        return [SegmentRecord.from_row(r) for r in rows]

    def set_segment_status(
        self, job_id: str, segment_ids: Sequence[str], status: SegmentStatus
    ) -> None:
        if not segment_ids:
            return
        with self.transaction() as conn:
            conn.executemany(
                "UPDATE segments SET status = ? WHERE job_id = ? AND segment_id = ?",
                [(status, job_id, sid) for sid in segment_ids],
            )

    def finalize_segment(
        self,
        job_id: str,
        segment_id: str,
        *,
        final_text: str,
        final_score: float | None,
        needs_review: bool,
        status: SegmentStatus = "done",
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE segments
                SET final_text = ?, final_score = ?, needs_review = ?, status = ?
                WHERE job_id = ? AND segment_id = ?
                """,
                (final_text, final_score, int(needs_review), status, job_id, segment_id),
            )
            conn.execute(
                """
                UPDATE jobs SET completed_segments = (
                    SELECT COUNT(*) FROM segments WHERE job_id = ? AND status = 'done'
                ), updated_at = ? WHERE id = ?
                """,
                (job_id, utcnow(), job_id),
            )

    def segment_counts(self, job_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM segments WHERE job_id = ? GROUP BY status",
            (job_id,),
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    # -- attempts and evaluations -------------------------------------------------

    def record_attempt(
        self,
        *,
        job_id: str,
        segment_id: str,
        attempt_no: int,
        model: str,
        params: dict[str, Any],
        output_text: str | None,
        latency_ms: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> int:
        """Store one attempt and return its row id. Never overwrites a previous attempt."""
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO attempts (job_id, segment_id, attempt_no, model, params_json,
                                      output_text, latency_ms, prompt_tokens, completion_tokens,
                                      cost_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    segment_id,
                    attempt_no,
                    model,
                    json.dumps(params, ensure_ascii=False, default=str),
                    output_text,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    utcnow(),
                ),
            )
            conn.execute(
                """
                UPDATE segments SET attempts_count = (
                    SELECT COUNT(*) FROM attempts WHERE job_id = ? AND segment_id = ?
                ) WHERE job_id = ? AND segment_id = ?
                """,
                (job_id, segment_id, job_id, segment_id),
            )
            row_id = cur.lastrowid
        if row_id is None:  # pragma: no cover - sqlite always supplies one for AUTOINCREMENT
            raise StoreError("SQLite did not return a row id for the attempt insert.")
        return int(row_id)

    def list_attempts(self, job_id: str, segment_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM attempts WHERE job_id = ? AND segment_id = ? ORDER BY attempt_no",
                (job_id, segment_id),
            ).fetchall()
        )

    def record_evaluation(
        self,
        *,
        attempt_id: int,
        evaluator_model: str,
        scores: dict[str, Any] | list[Any],
        issues: list[Any],
        composite: float,
        passed: bool,
    ) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO evaluations (attempt_id, evaluator_model, scores_json, issues_json,
                                         composite, passed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    evaluator_model,
                    json.dumps(scores, ensure_ascii=False, default=str),
                    json.dumps(issues, ensure_ascii=False, default=str),
                    composite,
                    int(passed),
                    utcnow(),
                ),
            )
            row_id = cur.lastrowid
        if row_id is None:  # pragma: no cover
            raise StoreError("SQLite did not return a row id for the evaluation insert.")
        return int(row_id)

    # -- usage and cost -----------------------------------------------------------

    def record_usage(
        self,
        *,
        job_id: str,
        model: str,
        endpoint: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO usage (job_id, ts, model, endpoint, prompt_tokens,
                                   completion_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, utcnow(), model, endpoint, prompt_tokens, completion_tokens, cost_usd),
            )
            conn.execute(
                """
                UPDATE jobs SET cost_usd = (
                    SELECT COALESCE(SUM(cost_usd), 0.0) FROM usage WHERE job_id = ?
                ), updated_at = ? WHERE id = ?
                """,
                (job_id, utcnow(), job_id),
            )

    def total_cost(self, job_id: str) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM usage WHERE job_id = ?", (job_id,)
        ).fetchone()
        return float(row["total"])

    # -- glossary ------------------------------------------------------------------

    def upsert_glossary(self, job_id: str, terms: Sequence[dict[str, Any]]) -> None:
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO glossary_terms (job_id, source, target, kind, locked, note,
                                            occurrences)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, source) DO UPDATE SET
                    target = excluded.target, kind = excluded.kind,
                    locked = excluded.locked, note = excluded.note,
                    occurrences = excluded.occurrences
                """,
                [
                    (
                        job_id,
                        t["source"],
                        t["target"],
                        t.get("kind"),
                        int(bool(t.get("locked", False))),
                        t.get("note"),
                        int(t.get("occurrences", 0)),
                    )
                    for t in terms
                ],
            )

    def list_glossary(self, job_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM glossary_terms WHERE job_id = ? ORDER BY occurrences DESC, source",
            (job_id,),
        ).fetchall()
        return [
            {
                "source": r["source"],
                "target": r["target"],
                "kind": r["kind"],
                "locked": bool(r["locked"]),
                "note": r["note"],
                "occurrences": r["occurrences"],
            }
            for r in rows
        ]


def open_job_store(job_id: str, *, create: bool = True) -> JobStore:
    """Open (or create) the store for a job id under ``~/.folioai/jobs``."""
    return JobStore(job_db_path(job_id), create=create)


def discover_jobs() -> list[tuple[str, Path]]:
    """Every job id on disk with a database, newest directory first."""
    from .paths import jobs_dir

    root = jobs_dir()
    if not root.is_dir():
        return []
    found = [
        (path.name, path / "job.db")
        for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if path.is_dir() and (path / "job.db").is_file()
    ]
    return found


def job_paths(job_id: str) -> dict[str, Path]:
    """Canonical on-disk locations for one job's artefacts."""
    base = job_dir(job_id)
    return {
        "dir": base,
        "db": base / "job.db",
        "ir": base / "ir.json",
        "translated_ir": base / "ir.translated.json",
        "structure": base / "structure.json",
        "glossary": base / "glossary.yaml",
        "probe": base / "probe.json",
        "audit": base / "extraction-audit.json",
        "export": base / "export",
    }
