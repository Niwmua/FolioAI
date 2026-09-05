"""Path configuration via ``.env``, and Persian language support."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from folioai import env as env_module
from folioai import paths
from folioai.config import available_profiles, load_profile, packaged_settings
from folioai.env import ensure_loaded, load_dotenv_file, load_env, parse_dotenv, reset_for_tests
from folioai.estimate import expansion_range
from folioai.extract.clean import PERSIAN_FOLD, ZERO_WIDTH, normalize_text
from folioai.render.base import font_for, is_rtl, script_for
from folioai.validate import fold_digits, sentence_count_delta

ZWNJ = "‌"
ZWJ = "‍"


@pytest.fixture(autouse=True)
def clean_env() -> None:
    """Start and end each test with no folioai path variables set.

    Restored by hand rather than with monkeypatch: loading a ``.env`` writes straight to
    ``os.environ`` by design, and monkeypatch only rolls back what monkeypatch itself set.
    Without this, a variable loaded from a fixture ``.env`` leaks into every later test in
    the session -- which is exactly how these tests first went wrong.
    """
    # FOLIOAI_CONFIG_DIR is left alone: the session fixture points it at a copy of the
    # shipped config with no .env in it, and clearing it here would send these tests back
    # to reading the developer's real one.
    managed = [n for n in env_module.PATH_VARIABLES if n != "FOLIOAI_CONFIG_DIR"]
    saved = {name: os.environ.get(name) for name in managed}
    for name in managed:
        os.environ.pop(name, None)
    reset_for_tests()
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_for_tests()


# -- .env parsing --------------------------------------------------------------------


def test_dotenv_parsing_handles_the_shapes_people_write() -> None:
    parsed = parse_dotenv(
        "\n".join(
            [
                "# a comment",
                "",
                "FOLIOAI_HOME=/srv/folioai",
                "export FOLIOAI_LOGS_DIR=/var/log/folioai",
                'FOLIOAI_CACHE_DB="/fast/cache.db"',
                "FOLIOAI_JOBS_DIR='/mnt/jobs'",
                "   FOLIOAI_STATE_FILE = /tmp/state.json   ",
                "NOT_AN_ASSIGNMENT",
            ]
        )
    )
    assert parsed["FOLIOAI_HOME"] == "/srv/folioai"
    assert parsed["FOLIOAI_LOGS_DIR"] == "/var/log/folioai"
    assert parsed["FOLIOAI_CACHE_DB"] == "/fast/cache.db"
    assert parsed["FOLIOAI_JOBS_DIR"] == "/mnt/jobs"
    assert parsed["FOLIOAI_STATE_FILE"] == "/tmp/state.json"
    assert "NOT_AN_ASSIGNMENT" not in parsed


def test_a_dotenv_never_overwrites_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`FOLIOAI_HOME=/tmp/x folioai ...` has to behave the way anyone would expect."""
    monkeypatch.setenv("FOLIOAI_HOME", "/from/shell")
    path = tmp_path / ".env"
    path.write_text("FOLIOAI_HOME=/from/dotenv\n", encoding="utf-8")
    load_dotenv_file(path)
    assert os.environ["FOLIOAI_HOME"] == "/from/shell"


def test_a_missing_or_unreadable_dotenv_is_not_an_error(tmp_path: Path) -> None:
    assert load_dotenv_file(tmp_path / "nothing-here.env") == {}


def test_the_project_dotenv_outranks_the_shipped_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = tmp_path / "config"
    shipped.mkdir()
    (shipped / ".env").write_text("FOLIOAI_JOBS_DIR=/from/config\n", encoding="utf-8")
    (tmp_path / ".env").write_text("FOLIOAI_JOBS_DIR=/from/project\n", encoding="utf-8")
    monkeypatch.setenv("FOLIOAI_CONFIG_DIR", str(shipped))

    load_env(tmp_path, force=True)
    assert os.environ["FOLIOAI_JOBS_DIR"] == "/from/project"


def test_the_shipped_dotenv_is_used_when_the_project_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = tmp_path / "config"
    shipped.mkdir()
    (shipped / ".env").write_text("FOLIOAI_JOBS_DIR=/from/config\n", encoding="utf-8")
    monkeypatch.setenv("FOLIOAI_CONFIG_DIR", str(shipped))

    load_env(tmp_path / "empty-project", force=True)
    assert os.environ["FOLIOAI_JOBS_DIR"] == "/from/config"


def test_loading_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("FOLIOAI_JOBS_DIR=/first\n", encoding="utf-8")
    load_env(tmp_path, force=True)
    monkeypatch.delenv("FOLIOAI_JOBS_DIR", raising=False)
    assert load_env(tmp_path) == {}  # the second call does nothing


# -- paths from the environment ----------------------------------------------------------


def test_every_path_defaults_below_the_home_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLIOAI_HOME", "/srv/folioai")
    ensure_loaded()
    home = paths.home_dir()
    assert paths.jobs_dir() == home / "jobs"
    assert paths.logs_dir() == home / "logs"
    assert paths.cache_db_path() == home / "cache.db"
    assert paths.state_path() == home / "state.json"
    assert paths.user_config_path() == home / "config.yaml"


@pytest.mark.parametrize(
    ("variable", "accessor"),
    [
        ("FOLIOAI_JOBS_DIR", "jobs_dir"),
        ("FOLIOAI_LOGS_DIR", "logs_dir"),
        ("FOLIOAI_CACHE_DB", "cache_db_path"),
        ("FOLIOAI_STATE_FILE", "state_path"),
        ("FOLIOAI_USER_CONFIG", "user_config_path"),
    ],
)
def test_each_path_can_be_overridden_individually(
    variable: str, accessor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "somewhere-else"
    monkeypatch.setenv(variable, str(target))
    assert getattr(paths, accessor)() == target.resolve()


def test_a_relative_path_resolves_against_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOLIOAI_JOBS_DIR", "local-jobs")
    assert paths.jobs_dir() == (tmp_path / "local-jobs").resolve()


def test_a_tilde_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLIOAI_CACHE_DB", "~/folio-cache.db")
    assert paths.cache_db_path() == (Path.home() / "folio-cache.db").resolve()


def test_an_empty_value_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLIOAI_HOME", "/srv/folioai")
    monkeypatch.setenv("FOLIOAI_JOBS_DIR", "   ")
    assert paths.jobs_dir() == paths.home_dir() / "jobs"


def test_the_config_directory_can_be_moved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolved from the process environment directly: it is where .env itself lives."""
    monkeypatch.setenv("FOLIOAI_CONFIG_DIR", str(tmp_path))
    assert paths.packaged_config_dir() == tmp_path.resolve()
    assert paths.profiles_dir() == tmp_path.resolve() / "profiles"
    assert paths.packaged_defaults_path() == tmp_path.resolve() / "default.yaml"


def test_describe_paths_lists_every_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLIOAI_HOME", "/srv/folioai")
    described = paths.describe_paths()
    assert set(described) == {
        "home",
        "jobs",
        "logs",
        "cache",
        "state",
        "user config",
        "packaged config",
    }
    assert all(isinstance(value, Path) for value in described.values())


def test_the_shipped_dotenv_template_exists_and_documents_every_path() -> None:
    """The committed template is how anyone discovers these variables exist."""
    template = Path("config/.env.example")
    assert template.is_file()
    text = template.read_text(encoding="utf-8")
    for variable in env_module.PATH_VARIABLES:
        assert variable in text, variable


def test_the_real_dotenv_is_not_committed() -> None:
    """§16: a key in config/.env must never reach the repository."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "config/.env"], capture_output=True, check=False
    )
    assert result.returncode == 0, "config/.env must be gitignored"


# -- Persian: the ZWNJ bug ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "meaning"),
    [
        ("می" + ZWNJ + "رود", "mi-ravad, he goes"),
        ("کتاب" + ZWNJ + "ها", "ketab-ha, books"),
        ("بزرگ" + ZWNJ + "تر", "bozorg-tar, bigger"),
    ],
)
def test_the_zero_width_non_joiner_survives_normalisation(word: str, meaning: str) -> None:
    """It is spelling in Persian, not typography: stripping it misspells the word."""
    assert ZWNJ in normalize_text(word), meaning
    assert normalize_text(word) == word


def test_the_zero_width_joiner_survives_too() -> None:
    assert ZWJ in normalize_text(f"a{ZWJ}b")


def test_genuinely_invisible_characters_are_still_removed() -> None:
    assert normalize_text("co­operate") == "cooperate"  # soft hyphen
    assert normalize_text("a​b") == "ab"  # zero width space
    assert normalize_text("﻿start") == "start"  # byte order mark
    assert ZWNJ not in ZERO_WIDTH and ZWJ not in ZERO_WIDTH


# -- Persian: letter folding -------------------------------------------------------------------


def test_arabic_letterforms_are_folded_in_persian_text() -> None:
    """Persian typed on an Arabic keyboard uses ي and ك; they must fold to ی and ک."""
    assert normalize_text("كتاب", lang="fa") == "کتاب"
    assert normalize_text("يك", lang="fa") == "یک"


def test_arabic_text_keeps_its_own_letterforms() -> None:
    """In Arabic those forms are the correct ones; folding them would be the error."""
    arabic = "كتاب"
    assert normalize_text(arabic, lang="ar") == arabic
    assert normalize_text(arabic) == arabic  # and with no language known, leave it alone


def test_the_fold_table_only_contains_perso_arabic_letters() -> None:
    assert all(0x0600 <= ord(k) <= 0x06FF for k in PERSIAN_FOLD)


# -- Persian: numbers, sentences, typography ------------------------------------------------------


def test_persian_numerals_fold_for_comparison() -> None:
    assert fold_digits("۴۷") == "47"
    assert fold_digits("٤٧") == "47"  # Arabic-Indic
    assert fold_digits("47") == "47"


def test_a_persian_translation_using_persian_numerals_keeps_its_numbers(
    settings: object,
) -> None:
    """Without digit folding this fires on every page of a Persian book."""
    from folioai.ir import Block
    from folioai.llm.client import LLMResponse
    from folioai.segment import Batch, Unit
    from folioai.tags import parse_segments, render_segments
    from folioai.translate import BatchTranslation
    from folioai.validate import validate_batch

    source = "He counted 47 sheep and 12 goats before dawn came at last."
    target = "او ۴۷ گوسفند و ۱۲ بز را شمرد."
    batch = Batch(
        index=0,
        chapter_id="ch01",
        units=[Unit(block=Block(id="b0001", kind="paragraph", text=source), tokens=20)],
    )
    response = render_segments([("b0001", target)])
    result = BatchTranslation(
        batch=batch,
        attempt_no=1,
        model="m",
        response=LLMResponse(text=response, model="m", finish_reason="stop"),
        parsed=parse_segments(response),
        messages=[],
    )
    report = validate_batch(result, settings)  # type: ignore[arg-type]
    assert not any(f.check == "numbers" for f in report.warnings)


def test_the_arabic_question_mark_ends_a_sentence() -> None:
    assert sentence_count_delta("Who? Nobody.", "چه کسی؟") == -1


def test_persian_is_right_to_left_with_its_own_font() -> None:
    assert is_rtl("fa")
    assert script_for("fa") == "persian"
    assert font_for("fa") == "Vazirmatn"
    assert font_for("ar") == "Noto Naskh Arabic"  # and Arabic keeps its own


def test_persian_has_an_expansion_range() -> None:
    low, high = expansion_range("fa")
    assert 0.5 < low < high < 2.0


# -- Persian: the shipped profile ------------------------------------------------------------------


def test_the_persian_profile_ships() -> None:
    assert "en-fa" in available_profiles()


def test_the_persian_profile_covers_what_persian_needs() -> None:
    profile = load_profile("en-fa")
    assert profile["target_lang"] == "fa"
    assert profile["names"] == "transliterate"

    notes = " ".join(profile["notes"]).lower()
    for topic in ("rtl", "zero-width non-joiner", "ezāfe", "ta'arof", "ی"):
        assert topic in notes, topic


def test_the_persian_dialogue_convention_names_persian_punctuation() -> None:
    convention = load_profile("en-fa")["dialogue_convention"]
    assert "«" in convention and "؟" in convention


def test_a_persian_job_picks_the_persian_profile_by_default() -> None:
    from folioai.jobs import default_profile_name

    assert default_profile_name("en", "fa") == "en-fa"
    assert default_profile_name("en", "xx") == "generic"


def test_the_packaged_settings_still_load_with_the_new_profile() -> None:
    assert packaged_settings().translation.batch_tokens == 1200
