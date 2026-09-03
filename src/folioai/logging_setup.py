"""Logging: JSON to file via structlog, human output to the terminal via Rich.

The two streams never mix (brief §2). ``get_logger()`` writes machine-readable events to
``~/.folioai/logs/<job_id>.jsonl``; ``console()`` writes to the terminal. Nothing writes to
both, and the log stream never goes to stdout, so piping ``folioai extract`` stays clean.

Secret redaction is a structlog *processor* rather than a call-site discipline (D-35): it is
applied to every event before rendering, so an API key cannot leak by someone forgetting.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.theme import Theme

_THEME = Theme(
    {
        "info": "cyan",
        "good": "green",
        "warn": "yellow",
        "bad": "bold red",
        "muted": "dim",
        "heading": "bold",
    }
)

_console: Console | None = None
_err_console: Console | None = None
_configured = False

# Anything that looks like a key, plus the values of fields whose *name* implies a secret.
_SECRET_VALUE_RE = re.compile(
    r"""(?xi)
    \b(
        sk-[A-Za-z0-9_\-]{16,}          # openai / openrouter style
      | sk-or-v1-[A-Za-z0-9_\-]{16,}
      | Bearer\s+[A-Za-z0-9._\-]{16,}
    )\b
    """
)
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|authorization|secret|token|password)")
REDACTED = "***redacted***"


def redact(value: str) -> str:
    """Mask anything that looks like a credential inside a free-text string."""
    return _SECRET_VALUE_RE.sub(REDACTED, value)


def _redact_processor(_logger: object, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(event_dict.items()):
        if _SECRET_KEY_RE.search(key):
            event_dict[key] = REDACTED
        elif isinstance(value, str):
            event_dict[key] = redact(value)
    return event_dict


def configure_logging(
    *,
    log_path: Path | None = None,
    level: str = "INFO",
    force: bool = False,
) -> None:
    """Point structlog at a JSONL file. Idempotent unless ``force``."""
    global _configured
    if _configured and not force:
        return

    handlers: list[logging.Handler] = []
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    else:
        # No job yet: send events to stderr so they never pollute piped stdout.
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    processors: list[Callable[..., Any]] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_processor,
        structlog.processors.JSONRenderer(),
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "folioai") -> structlog.stdlib.BoundLogger:
    """A bound JSON logger. Configures with defaults if nobody has yet."""
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def console() -> Console:
    """Rich console for human-facing stdout."""
    global _console
    if _console is None:
        _console = Console(theme=_THEME, highlight=False, soft_wrap=False)
    return _console


def err_console() -> Console:
    """Rich console for human-facing stderr (errors, warnings, progress on --no-tty)."""
    global _err_console
    if _err_console is None:
        _err_console = Console(theme=_THEME, highlight=False, stderr=True)
    return _err_console


def reset_for_tests() -> None:
    """Drop cached consoles and configuration. Test-support only."""
    global _console, _err_console, _configured
    _console = None
    _err_console = None
    _configured = False
    structlog.reset_defaults()
