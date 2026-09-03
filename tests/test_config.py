"""Config precedence, env parsing, and the rules that protect API keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from folioai.config import (
    Settings,
    config_from_env,
    load_dotenv,
    load_profile,
    load_settings,
)
from folioai.errors import ConfigError


def test_packaged_defaults_match_the_brief() -> None:
    settings = load_settings(project_dir=Path.cwd())
    assert settings.translation.batch_tokens == 1200
    assert settings.evaluation.min_score == 80
    assert settings.retry.max_attempts == 3
    assert settings.translation.concurrency == 4
    assert settings.translation.temperature == pytest.approx(0.2)
    assert settings.evaluation.temperature == pytest.approx(0.0)
    assert settings.evaluation.mode == "direct"
    assert settings.evaluation.sample == pytest.approx(1.0)


def test_default_evaluator_differs_from_translator() -> None:
    """Brief §10: correlated blind spots are the main LLM-as-judge failure mode."""
    settings = load_settings()
    assert settings.models.evaluator != settings.models.translator


def test_precedence_cli_beats_env_beats_project_file(tmp_path: Path) -> None:
    (tmp_path / "folioai.yaml").write_text(
        "translation:\n  batch_tokens: 500\n  concurrency: 9\n", encoding="utf-8"
    )
    env = {"FOLIOAI_TRANSLATION__BATCH_TOKENS": "700"}

    from_file = load_settings(project_dir=tmp_path)
    assert from_file.translation.batch_tokens == 500

    from_env = load_settings(project_dir=tmp_path, environ=env)
    assert from_env.translation.batch_tokens == 700
    assert from_env.translation.concurrency == 9  # untouched key survives the overlay

    from_cli = load_settings(
        project_dir=tmp_path, environ=env, cli_overrides={"translation": {"batch_tokens": 900}}
    )
    assert from_cli.translation.batch_tokens == 900


def test_env_flat_aliases_and_type_coercion() -> None:
    overlay = config_from_env(
        {
            "FOLIOAI_CONCURRENCY": "12",
            "FOLIOAI_BASE_URL": "http://localhost:11434/v1",
            "FOLIOAI_LLM__CACHE_ENABLED": "false",
            "FOLIOAI_EVALUATION__SAMPLE": "0.25",
            "UNRELATED": "ignored",
        }
    )
    assert overlay["translation"]["concurrency"] == 12
    assert overlay["llm"]["base_url"] == "http://localhost:11434/v1"
    assert overlay["llm"]["cache_enabled"] is False
    assert overlay["evaluation"]["sample"] == 0.25
    assert "unrelated" not in overlay


def test_api_key_never_comes_from_a_config_file(tmp_path: Path) -> None:
    (tmp_path / "folioai.yaml").write_text("api_key: sk-should-be-ignored\n", encoding="utf-8")
    settings = load_settings(project_dir=tmp_path, environ={})
    assert settings.api_key is None


def test_api_key_comes_from_the_environment() -> None:
    settings = load_settings(environ={"FOLIOAI_API_KEY": "sk-or-v1-secret"})
    assert settings.api_key == "sk-or-v1-secret"


def test_api_key_is_excluded_from_dumps() -> None:
    settings = load_settings(environ={"FOLIOAI_API_KEY": "sk-or-v1-secret"})
    assert "sk-or-v1-secret" not in settings.model_dump_json()
    assert "sk-or-v1-secret" not in repr(settings)


def test_unknown_key_is_an_error_not_a_shrug(tmp_path: Path) -> None:
    (tmp_path / "folioai.yaml").write_text("translation:\n  batch_tokes: 500\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(project_dir=tmp_path)
    assert excinfo.value.remedy


def test_malformed_yaml_names_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "folioai.yaml"
    bad.write_text("translation:\n  - [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(project_dir=tmp_path)
    assert excinfo.value.context["path"] == str(bad)


def test_rubric_weights_must_sum_to_one(tmp_path: Path) -> None:
    (tmp_path / "folioai.yaml").write_text(
        "evaluation:\n  weights:\n    completeness: 0.9\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_settings(project_dir=tmp_path)


def test_retry_temperatures_must_cover_max_attempts(tmp_path: Path) -> None:
    (tmp_path / "folioai.yaml").write_text("retry:\n  max_attempts: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(project_dir=tmp_path)


def test_dotenv_does_not_override_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("FOLIOAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("FOLIOAI_API_KEY", "from-shell")
    load_dotenv(tmp_path / ".env")
    import os

    assert os.environ["FOLIOAI_API_KEY"] == "from-shell"


def test_unknown_model_price_is_none_not_a_crash() -> None:
    settings = load_settings()
    assert settings.price_for("nonexistent/model-9000") is None
    assert settings.price_for(settings.models.translator) is not None


def test_shipped_profiles_cover_the_required_pairs() -> None:
    from folioai.config import available_profiles

    profiles = set(available_profiles())
    required = {"en-de", "en-es", "en-fr", "en-ja", "en-zh-hans", "en-ar", "generic"}
    assert required <= profiles


def test_load_profile_by_name_and_unknown_profile_lists_options() -> None:
    profile = load_profile("generic")
    assert "register" in profile
    with pytest.raises(ConfigError) as excinfo:
        load_profile("klingon")
    assert "generic" in (excinfo.value.remedy or "")


def test_settings_is_constructible_without_any_files() -> None:
    """A bare Settings() must be valid, so library use needs no YAML on disk."""
    settings = Settings()
    assert settings.translation.batch_tokens == 1200
