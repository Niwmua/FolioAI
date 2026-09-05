"""``.env`` loading, and the single place that decides where files live.

Two files are read, lowest precedence first:

1. ``config/.env`` -- the shipped one, next to ``default.yaml``. Deployment-wide settings:
   where jobs, logs and the cache live.
2. ``./.env`` -- the project one, for whatever this checkout needs to differ on.

Neither ever overwrites a variable that is already set, so the real environment always wins
and ``FOLIOAI_HOME=/tmp/x folioai ...`` behaves the way anyone would expect.

**No secret in ``config/.env`` is ever committed.** ``.env`` is gitignored at every depth,
and ``config/.env.example`` is the committed template. §16 says keys come from the
environment or a ``.env`` and never from a config file in the repository; this module is
what makes the second half of that true rather than aspirational.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False

#: Every path the application uses, and the variable that overrides it.
PATH_VARIABLES = (
    "FOLIOAI_HOME",
    "FOLIOAI_JOBS_DIR",
    "FOLIOAI_LOGS_DIR",
    "FOLIOAI_CACHE_DB",
    "FOLIOAI_STATE_FILE",
    "FOLIOAI_USER_CONFIG",
    "FOLIOAI_CONFIG_DIR",
)


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` contents into a mapping.

    Deliberately small: ``KEY=value``, ``#`` comments, optional ``export`` prefix, and
    surrounding quotes stripped. A full dotenv implementation would bring interpolation and
    multi-line values, which is more syntax than a settings file for six paths deserves.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name:
            values[name] = value
    return values


def load_dotenv_file(path: Path) -> dict[str, str]:
    """Load one ``.env`` into the environment. Existing variables are never overwritten."""
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        # A .env that cannot be read is not a reason to refuse to start; the defaults are
        # perfectly usable and the user will see it in the paths that result.
        return {}

    applied: dict[str, str] = {}
    for name, value in parse_dotenv(text).items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def dotenv_paths(project_dir: Path | None = None) -> list[Path]:
    """The ``.env`` files to read, in the order they are read.

    Project first so that, with "never overwrite" semantics, the project file wins over the
    shipped one -- matching how ``folioai.yaml`` outranks the packaged defaults.
    """
    from .paths import packaged_config_dir

    project = project_dir or Path.cwd()
    return [project / ".env", packaged_config_dir() / ".env"]


def load_env(project_dir: Path | None = None, *, force: bool = False) -> dict[str, str]:
    """Load every ``.env``. Idempotent unless ``force``.

    Returns:
        The variables this call actually set, for logging and tests.
    """
    global _LOADED
    if _LOADED and not force:
        return {}

    applied: dict[str, str] = {}
    for path in dotenv_paths(project_dir):
        applied.update(load_dotenv_file(path))
    _LOADED = True
    return applied


def ensure_loaded() -> None:
    """Load ``.env`` files once, before anything reads a path from the environment."""
    if not _LOADED:
        load_env()


def reset_for_tests() -> None:
    """Forget that ``.env`` files were loaded. Test-support only."""
    global _LOADED
    _LOADED = False


def env_path(variable: str, default: Path) -> Path:
    """Read a path from the environment, falling back to ``default``.

    ``~`` is expanded and the result is made absolute, so a relative path in a ``.env``
    means "relative to where you ran the command", which is the only reading that does not
    surprise someone.
    """
    ensure_loaded()
    raw = os.environ.get(variable)
    if not raw or not raw.strip():
        return default
    return Path(raw.strip()).expanduser().resolve()
