"""Filesystem locations. Everything the app writes outside the job's output goes here."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

APP_NAME = "folioai"
ENV_PREFIX = "FOLIOAI_"

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def home_dir() -> Path:
    """Root of the application's state directory (``~/.folioai``, or ``FOLIOAI_HOME``)."""
    override = os.environ.get(f"{ENV_PREFIX}HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / f".{APP_NAME}"


def jobs_dir() -> Path:
    return home_dir() / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


def job_db_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.db"


def logs_dir() -> Path:
    return home_dir() / "logs"


def job_log_path(job_id: str) -> Path:
    return logs_dir() / f"{job_id}.jsonl"


def user_config_path() -> Path:
    return home_dir() / "config.yaml"


def state_path() -> Path:
    """Small JSON blob for machine-level state, e.g. the first-run notice (D-53)."""
    return home_dir() / "state.json"


def cache_db_path() -> Path:
    """Prompt/response cache, shared across jobs so a re-run of a tweaked config is cheap."""
    return home_dir() / "cache.db"


def ensure_dirs() -> None:
    """Create the state directories. Safe to call repeatedly."""
    for path in (home_dir(), jobs_dir(), logs_dir()):
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str, *, max_length: int = 40) -> str:
    """Lowercase ASCII slug, for job ids and filenames."""
    slug = _SLUG_STRIP.sub("-", value.lower().strip()).strip("-")
    return slug[:max_length].strip("-") or "untitled"


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file's contents. Used to identify a source PDF across runs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_job_id(source_path: Path, source_sha256: str) -> str:
    """Human-typable, collision-resistant job id (D-23): ``<slug>-<8 hex>``."""
    return f"{slugify(source_path.stem)}-{source_sha256[:8]}"
