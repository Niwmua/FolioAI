"""Filesystem locations. Everything the app writes outside the job's output goes here.

Every path is overridable from ``config/.env`` (or the environment, or a project ``.env``).
The defaults derive from ``FOLIOAI_HOME``, so setting that one variable moves everything;
the individual variables exist for the cases where that is not enough -- a cache on a fast
disk, logs on a mounted volume, jobs somewhere with room for a few hundred books.

| Variable | Default | What it holds |
|---|---|---|
| ``FOLIOAI_HOME`` | ``~/.folioai`` | Everything below, unless overridden |
| ``FOLIOAI_JOBS_DIR`` | ``$HOME/jobs`` | One directory per job: IR, database, exports |
| ``FOLIOAI_LOGS_DIR`` | ``$HOME/logs`` | One JSONL file per job |
| ``FOLIOAI_CACHE_DB`` | ``$HOME/cache.db`` | Prompt/response cache, shared across jobs |
| ``FOLIOAI_STATE_FILE`` | ``$HOME/state.json`` | Machine-level state (the first-run notice) |
| ``FOLIOAI_USER_CONFIG`` | ``$HOME/config.yaml`` | The user's own settings |
| ``FOLIOAI_CONFIG_DIR`` | packaged ``config/`` | ``default.yaml``, ``profiles/`` and ``.env`` |
| ``FOLIOAI_BIN_DIR`` | ``$HOME/bin`` | Helper binaries: Typst, epubcheck |
| ``FOLIOAI_FONTS_DIR`` | ``$HOME/fonts`` | Extra fonts for PDF rendering |
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

APP_NAME = "folioai"
ENV_PREFIX = "FOLIOAI_"

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def package_dir() -> Path:
    """Directory of the installed ``folioai`` package."""
    return Path(__file__).resolve().parent


def packaged_config_dir() -> Path:
    """Where ``default.yaml``, the style profiles and the shipped ``.env`` live.

    Resolved without going through :mod:`folioai.env`, because that module reads its own
    ``.env`` from here -- one of the two has to be the fixed point, and it is this one.
    ``FOLIOAI_CONFIG_DIR`` still overrides it, straight from the process environment.

    The brief puts ``config/`` at the repository root, which is the right place to edit it,
    but that path does not exist inside an installed wheel; the build copies the directory
    to ``folioai/_config`` (see pyproject), so the packaged copy is preferred and the
    checkout is the fallback.
    """
    override = os.environ.get(f"{ENV_PREFIX}CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    packaged = package_dir() / "_config"
    if packaged.is_dir():
        return packaged
    return package_dir().parent.parent / "config"


def home_dir() -> Path:
    """Root of the application's state directory (``~/.folioai``, or ``FOLIOAI_HOME``)."""
    from .env import env_path

    return env_path(f"{ENV_PREFIX}HOME", Path.home() / f".{APP_NAME}")


def jobs_dir() -> Path:
    from .env import env_path

    return env_path(f"{ENV_PREFIX}JOBS_DIR", home_dir() / "jobs")


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


def job_db_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.db"


def logs_dir() -> Path:
    from .env import env_path

    return env_path(f"{ENV_PREFIX}LOGS_DIR", home_dir() / "logs")


def job_log_path(job_id: str) -> Path:
    return logs_dir() / f"{job_id}.jsonl"


def user_config_path() -> Path:
    from .env import env_path

    return env_path(f"{ENV_PREFIX}USER_CONFIG", home_dir() / "config.yaml")


def state_path() -> Path:
    """Small JSON blob for machine-level state, e.g. the first-run notice (D-53)."""
    from .env import env_path

    return env_path(f"{ENV_PREFIX}STATE_FILE", home_dir() / "state.json")


def cache_db_path() -> Path:
    """Prompt/response cache, shared across jobs so a re-run of a tweaked config is cheap."""
    from .env import env_path

    return env_path(f"{ENV_PREFIX}CACHE_DB", home_dir() / "cache.db")


def bin_dir() -> Path:
    """Where folioai looks for helper binaries it did not install itself.

    A single-binary tool like Typst is often just downloaded rather than installed, and
    landing it here means it works without touching PATH.
    """
    from .env import env_path

    return env_path(f"{ENV_PREFIX}BIN_DIR", home_dir() / "bin")


def fonts_dir() -> Path:
    """Fonts made available to the PDF renderer, in addition to the system's."""
    from .env import env_path

    return env_path(f"{ENV_PREFIX}FONTS_DIR", home_dir() / "fonts")


def profiles_dir() -> Path:
    """Shipped style profiles."""
    return packaged_config_dir() / "profiles"


def packaged_defaults_path() -> Path:
    """The shipped ``default.yaml``."""
    return packaged_config_dir() / "default.yaml"


def describe_paths() -> dict[str, Path]:
    """Every location the application uses, for ``folioai paths`` and the logs."""
    return {
        "home": home_dir(),
        "jobs": jobs_dir(),
        "logs": logs_dir(),
        "cache": cache_db_path(),
        "state": state_path(),
        "user config": user_config_path(),
        "packaged config": packaged_config_dir(),
        "binaries": bin_dir(),
        "fonts": fonts_dir(),
    }


def ensure_dirs() -> None:
    """Create the state directories. Safe to call repeatedly."""
    for path in (home_dir(), jobs_dir(), logs_dir()):
        path.mkdir(parents=True, exist_ok=True)
    cache_db_path().parent.mkdir(parents=True, exist_ok=True)


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
