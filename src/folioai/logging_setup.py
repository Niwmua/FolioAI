"""Logging: JSON to file via structlog, human output to the terminal via Rich.

The two streams never mix (brief §2). ``get_logger()`` writes machine-readable events to
``~/.folioai/logs/<job_id>.jsonl``; ``console()`` writes to the terminal. Nothing writes to
both, and the log stream never goes to stdout, so piping ``folioai extract`` stays clean.

Secret redaction is a structlog *processor* rather than a call-site discipline (D-35): it is
applied to every event before rendering, so an API key cannot leak by someone forgetting.
"""

from __future__ import annotations

import contextlib
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
#
# The JWT pattern is not hypothetical: gateways that front several providers commonly
# issue a JWT rather than an sk- string, and a redactor that only knows OpenAI's format
# would have let one straight into the logs.
_SECRET_VALUE_RE = re.compile(
    r"""(?xi)
    (
        \bsk-or-v1-[A-Za-z0-9_-]{16,}    # openrouter
      | \bsk-[A-Za-z0-9_-]{16,}          # openai, and the many endpoints that copy it
      | \bBearer\s+[A-Za-z0-9._-]{16,}
      | \beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}   # a JWT
      | \bgsk_[A-Za-z0-9]{16,}           # groq
      | \bAIza[A-Za-z0-9_-]{20,}         # google
      | \bhf_[A-Za-z0-9]{16,}            # hugging face
    )
    """
)
# Anchored on purpose. A loose match on "token" redacted `prompt_tokens`, `source_tokens`
# and `max_tokens` -- which is to say, every number the JSON logs exist to record (§18).
# A redactor that erases the diagnostics is not a safer redactor.
_SECRET_KEY_RE = re.compile(
    r"""(?ix)
    (^|_)(
        api[_-]?key | apikey | authorization | secret | password | passwd
      | credential(s)? | bearer
      | (access|refresh|id|session|auth)[_-]?token
      | token | key
    )($|_)
    """
)
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
    to_stderr: bool = False,
) -> None:
    """Point structlog at a JSONL file. Idempotent unless ``force``.

    Args:
        log_path: JSONL destination. When omitted there is no job yet.
        level: Minimum level to record.
        force: Reconfigure even if already configured.
        to_stderr: Also emit events to stderr. Off by default: §2 says the JSON stream and
            the human stream never mix, and a user who mistypes a job id should get one
            clean sentence, not that sentence plus its own log line. ``-v`` turns it on.
    """
    global _configured
    if _configured and not force:
        return

    handlers: list[logging.Handler] = []
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    if to_stderr:
        # Never stdout: piping `folioai extract` must stay clean.
        handlers.append(logging.StreamHandler(sys.stderr))
    if not handlers:
        handlers.append(logging.NullHandler())

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


def _prepare_stream(stream: object) -> None:
    """Coax a console stream into UTF-8 before Rich writes anything to it.

    A translation tool is going to print Arabic, Japanese and typographic punctuation. On a
    Windows console still running a legacy code page (cp1252, cp1256, cp932) the default
    encoder raises ``UnicodeEncodeError`` on the first arrow or em dash, which turns a cost
    estimate into a traceback. ``errors="replace"`` is the belt to that braces: if the
    terminal genuinely cannot render a glyph, it shows a placeholder instead of dying.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    # A redirected or already-detached stream simply keeps its own encoding.
    with contextlib.suppress(OSError, ValueError):
        reconfigure(encoding="utf-8", errors="replace")


def console() -> Console:
    """Rich console for human-facing stdout."""
    global _console
    if _console is None:
        _prepare_stream(sys.stdout)
        _console = Console(theme=_THEME, highlight=False, soft_wrap=False)
    return _console


def err_console() -> Console:
    """Rich console for human-facing stderr (errors, warnings, progress on --no-tty)."""
    global _err_console
    if _err_console is None:
        _prepare_stream(sys.stderr)
        _err_console = Console(theme=_THEME, highlight=False, stderr=True)
    return _err_console


def shutdown_logging() -> None:
    """Close the JSONL log file.

    The handler holds an open file for the life of the process, which is fine for the CLI
    but not for anything embedding folioai: on Windows an open handle stops the directory
    being removed, so a caller working in a temporary job directory cannot clean it up.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        with contextlib.suppress(OSError, ValueError):
            handler.close()


def reset_for_tests() -> None:
    """Drop cached consoles and configuration. Test-support only."""
    global _console, _err_console, _configured
    _console = None
    _err_console = None
    _configured = False
    structlog.reset_defaults()
