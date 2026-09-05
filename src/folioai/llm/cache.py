"""Content-hash cache for LLM responses (brief §8).

Identical source block plus identical prompt fingerprint reuses the prior translation, which
saves real money on the re-runs that follow a config tweak. The cache is shared across jobs
(``~/.folioai/cache.db``): re-translating the same book with a different output format, or a
second book that quotes the first, should not pay twice.

The fingerprint covers the *rendered* messages rather than their ingredients, so the glossary
subset, style profile, rolling context and prompt template version are all included by
construction -- anything left out of the key produces silently stale reuse (D-32).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from ..store import utcnow

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    prompt_fingerprint TEXT PRIMARY KEY,
    model              TEXT NOT NULL,
    output_text        TEXT NOT NULL,
    prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    completion_tokens  INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL
);
"""


def fingerprint(*, model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
    """Stable hash of everything that can change a response.

    Sorted keys and a canonical separator, so two structurally identical requests hash the
    same regardless of dict ordering.
    """
    payload = {
        "model": model,
        "messages": messages,
        "params": {k: v for k, v in sorted(params.items()) if v is not None},
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PromptCache:
    """SQLite-backed response cache. Safe to share across asyncio tasks and threads."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                path, isolation_level=None, check_same_thread=False, timeout=15.0
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def get(self, key: str) -> str | None:
        """Return a cached completion, or ``None``.

        A cache read must never be able to fail a run: a corrupt or locked cache logs and
        behaves as a miss.
        """
        if not self.enabled or self._conn is None:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT output_text FROM cache WHERE prompt_fingerprint = ?", (key,)
                ).fetchone()
        except sqlite3.Error as exc:
            log.warning("cache_read_failed", error=str(exc), path=str(self.path))
            return None
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return str(row["output_text"])

    def put(
        self,
        key: str,
        *,
        model: str,
        output_text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Store a completion. Failures are logged, never raised."""
        if not self.enabled or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO cache (prompt_fingerprint, model, output_text, prompt_tokens,
                                       completion_tokens, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(prompt_fingerprint) DO UPDATE SET
                        output_text = excluded.output_text,
                        model = excluded.model,
                        created_at = excluded.created_at
                    """,
                    (key, model, output_text, prompt_tokens, completion_tokens, utcnow()),
                )
        except sqlite3.Error as exc:
            log.warning("cache_write_failed", error=str(exc), path=str(self.path))

    def clear(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.execute("DELETE FROM cache")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PromptCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
