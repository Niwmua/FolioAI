"""Error contract and secret redaction.

Redaction is tested at the processor rather than the call site because that is the only
place it cannot be forgotten (D-35).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from folioai.errors import (
    BudgetExceeded,
    ExtractionError,
    FolioError,
    LLMError,
    RateLimitError,
    RenderError,
)
from folioai.logging_setup import configure_logging, get_logger, redact, reset_for_tests


def test_every_error_type_descends_from_the_root() -> None:
    for cls in (ExtractionError, LLMError, RateLimitError, BudgetExceeded, RenderError):
        assert issubclass(cls, FolioError)
    assert issubclass(RateLimitError, LLMError)


def test_exit_codes_are_distinct() -> None:
    codes = [
        cls.exit_code
        for cls in (ExtractionError, LLMError, RateLimitError, BudgetExceeded, RenderError)
    ]
    assert len(set(codes)) == len(codes)


def test_error_message_states_what_to_do_next() -> None:
    exc = RenderError(
        "Cannot render PDF: the font 'Noto Serif CJK' is not installed.",
        remedy="Install it with: winget install Google.NotoSerifCJK",
        context={"font": "Noto Serif CJK"},
    )
    rendered = exc.format_for_user()
    assert "Cannot render PDF" in rendered
    assert "What to do:" in rendered
    assert "winget install" in rendered


def test_error_without_remedy_renders_bare() -> None:
    assert FolioError("something broke").format_for_user() == "something broke"


def test_rate_limit_carries_retry_after() -> None:
    exc = RateLimitError("429 from the endpoint", retry_after=12.5)
    assert exc.retry_after == 12.5


@pytest.mark.parametrize(
    "text",
    [
        "key is sk-abcdefghijklmnop1234",
        "key is sk-or-v1-abcdefghijklmnop1234",
        "Authorization: Bearer abcdefghijklmnop1234",
    ],
)
def test_redact_masks_credential_shapes(text: str) -> None:
    assert "abcdefghijklmnop1234" not in redact(text)
    assert "redacted" in redact(text)


def test_log_events_redact_by_key_name_and_by_value(tmp_path: Path) -> None:
    log_path = tmp_path / "job.jsonl"
    reset_for_tests()
    configure_logging(log_path=log_path, force=True)
    get_logger("test").info(
        "llm_call",
        api_key="sk-or-v1-supersecretvalue123456",
        authorization="Bearer supersecretvalue123456",
        detail="failed with key sk-or-v1-anothersecret1234567",
        model="openai/gpt-4.1",
    )
    logging.shutdown()

    line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    event = json.loads(line)
    assert event["event"] == "llm_call"
    assert event["model"] == "openai/gpt-4.1"
    assert "supersecret" not in line
    assert "anothersecret" not in line
    assert event["api_key"] == "***redacted***"
    reset_for_tests()


def test_logs_are_json_one_line_per_event(tmp_path: Path) -> None:
    log_path = tmp_path / "job.jsonl"
    reset_for_tests()
    configure_logging(log_path=log_path, force=True)
    log = get_logger("test")
    for i in range(3):
        log.info("segment_done", segment_id=f"b{i:04d}", score=88.5)
    logging.shutdown()

    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        event = json.loads(line)
        assert event["event"] == "segment_done"
        assert "timestamp" in event and "level" in event
    reset_for_tests()
