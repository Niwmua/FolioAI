"""Every model role, and every credential, comes from configuration -- never a hardcoded name.

The failure this guards against is specific and was real: a user configures a translator and
an evaluator for their own gateway, the run translates and evaluates correctly, and then
attempt 3 of the retry ladder calls a vendor's model the endpoint has never heard of --
halfway through a book, after the money is spent.
"""

from __future__ import annotations

import pytest

from folioai.config import ModelsConfig, Settings, config_from_env, load_settings
from folioai.logging_setup import redact

SECONDARY_ROLES = ("escalation", "summarizer", "glossary", "back_translator", "vision")
ALL_ROLES = ("translator", "evaluator", *SECONDARY_ROLES)


# -- inheritance -------------------------------------------------------------------


def test_unset_roles_inherit_the_translator() -> None:
    models = ModelsConfig(translator="my-gateway/model-a", evaluator="my-gateway/model-b")
    for role in SECONDARY_ROLES:
        assert models.role(role) == "my-gateway/model-a", role
    assert set(models.inherited) == set(SECONDARY_ROLES)


def test_a_configured_role_is_not_overridden_by_inheritance() -> None:
    models = ModelsConfig(translator="my-gateway/model-a", escalation="my-gateway/model-strong")
    assert models.role("escalation") == "my-gateway/model-strong"
    assert "escalation" not in models.inherited
    assert "summarizer" in models.inherited


def test_no_role_ever_resolves_to_none() -> None:
    models = ModelsConfig()
    assert all(isinstance(models.role(role), str) and models.role(role) for role in ALL_ROLES)


def test_configuring_two_models_leaves_no_third_vendor_in_play() -> None:
    """The whole point: name two models, and nothing else reaches for a different vendor."""
    settings = load_settings(
        environ={
            "FOLIOAI_TRANSLATOR_MODEL": "google/gemini-3.6-flash",
            "FOLIOAI_EVALUATOR_MODEL": "deepseek/deepseek-v4-flash",
        }
    )
    used = {settings.models.role(role) for role in ALL_ROLES}
    assert used == {"google/gemini-3.6-flash", "deepseek/deepseek-v4-flash"}


def test_the_shipped_defaults_still_cross_vendors_for_the_judge() -> None:
    """§10: an evaluator sharing the translator's blind spots is the main judge failure."""
    settings = load_settings(environ={})
    assert settings.models.evaluator != settings.models.translator


# -- environment aliases ---------------------------------------------------------------


@pytest.mark.parametrize("role", ALL_ROLES)
def test_every_role_has_a_short_environment_alias(role: str) -> None:
    """A role without an alias is a role that quietly keeps its default."""
    variable = f"FOLIOAI_{role.upper()}_MODEL"
    overlay = config_from_env({variable: "some/model"})
    assert overlay.get("models", {}).get(role) == "some/model", variable


@pytest.mark.parametrize(
    ("variable", "dotted", "value", "expected"),
    [
        ("FOLIOAI_BASE_URL", ("llm", "base_url"), "https://x/v1", "https://x/v1"),
        ("FOLIOAI_CONCURRENCY", ("translation", "concurrency"), "9", 9),
        ("FOLIOAI_BATCH_TOKENS", ("translation", "batch_tokens"), "900", 900),
        ("FOLIOAI_MIN_SCORE", ("evaluation", "min_score"), "75", 75),
        ("FOLIOAI_EVAL_MODE", ("evaluation", "mode"), "both", "both"),
        ("FOLIOAI_EVAL_SAMPLE", ("evaluation", "sample"), "0.25", 0.25),
        ("FOLIOAI_MAX_ATTEMPTS", ("retry", "max_attempts"), "2", 2),
        ("FOLIOAI_MAX_COST", ("budget", "max_cost_usd"), "5.5", 5.5),
        ("FOLIOAI_RPM", ("llm", "rpm"), "30", 30),
        ("FOLIOAI_TPM", ("llm", "tpm"), "60000", 60000),
        ("FOLIOAI_TIMEOUT", ("llm", "timeout_s"), "90", 90),
    ],
)
def test_the_settings_people_change_have_short_names(
    variable: str, dotted: tuple[str, str], value: str, expected: object
) -> None:
    overlay = config_from_env({variable: value})
    assert overlay[dotted[0]][dotted[1]] == expected


def test_the_env_beats_the_shipped_defaults_for_every_role() -> None:
    environ = {f"FOLIOAI_{role.upper()}_MODEL": f"mine/{role}" for role in ALL_ROLES}
    settings = load_settings(environ=environ)
    for role in ALL_ROLES:
        assert settings.models.role(role) == f"mine/{role}"
        assert ".env" in settings.origin(f"models.{role}")


# -- provenance ---------------------------------------------------------------------------


def test_provenance_says_where_each_setting_came_from() -> None:
    settings = load_settings(environ={"FOLIOAI_TRANSLATOR_MODEL": "mine/model"})
    assert ".env" in settings.origin("models.translator")
    assert "packaged defaults" in settings.origin("evaluation.min_score")
    assert settings.origin("nothing.set.here") == "built-in default"


def test_an_inherited_role_is_reported_as_inherited() -> None:
    settings = load_settings(environ={"FOLIOAI_TRANSLATOR_MODEL": "mine/model"})
    assert "summarizer" in settings.models.inherited
    # It has no source of its own, precisely because nothing set it.
    assert settings.origin("models.summarizer") == "built-in default"


def test_cli_overrides_outrank_the_env_in_provenance() -> None:
    settings = load_settings(
        environ={"FOLIOAI_TRANSLATOR_MODEL": "from/env"},
        cli_overrides={"models": {"translator": "from/flag"}},
    )
    assert settings.models.translator == "from/flag"
    assert settings.origin("models.translator") == "command line"


# -- the endpoint and the key ------------------------------------------------------------------


def test_the_base_url_is_configurable_and_not_pinned_to_one_vendor() -> None:
    settings = load_settings(environ={"FOLIOAI_BASE_URL": "https://ai.example.ir/api/x/v1"})
    assert settings.llm.base_url == "https://ai.example.ir/api/x/v1"


@pytest.mark.parametrize("variable", ["FOLIOAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"])
def test_the_key_is_read_from_any_of_the_usual_variables(variable: str) -> None:
    settings = load_settings(environ={variable: "a-secret-value"})
    assert settings.api_key == "a-secret-value"


def test_the_key_never_appears_in_a_dump_or_repr() -> None:
    settings = load_settings(environ={"FOLIOAI_API_KEY": "a-secret-value"})
    assert "a-secret-value" not in settings.model_dump_json()
    assert "a-secret-value" not in repr(settings)


# -- redaction of the key shapes real endpoints issue ---------------------------------------------


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p"),
        ("openai", "sk-abcdefghijklmnop1234567890"),
        ("openrouter", "sk-or-v1-abcdefghijklmnop1234"),
        ("bearer", "Bearer abcdefghijklmnopqrstuvwx"),
        ("google", "AIzaSyD-abcdefghijklmnopqrstuvwxyz123"),
        ("groq", "gsk_abcdefghijklmnopqrstuvwx"),
        ("huggingface", "hf_abcdefghijklmnopqrstuvwx"),
    ],
)
def test_every_common_key_shape_is_redacted(label: str, text: str) -> None:
    """A gateway fronting several providers often issues a JWT, not an sk- string."""
    assert "***redacted***" in redact(f"the failing call used {text} as its credential"), label


def test_redaction_leaves_ordinary_diagnostics_alone() -> None:
    line = "model google/gemini-3.6-flash used 1240 prompt tokens at 0.0031 USD"
    assert redact(line) == line


def test_a_base64ish_word_is_not_mistaken_for_a_jwt() -> None:
    """A JWT has three dot-separated parts; one long token is just a word."""
    assert "***redacted***" not in redact("eyJhbGciOiJIUzI1NiJ9")


# -- the shipped configuration ---------------------------------------------------------------------


def test_default_yaml_names_no_vendor_for_the_inheriting_roles() -> None:
    """Naming one here is what broke a user's run on their own gateway."""
    from folioai.paths import packaged_defaults_path

    text = packaged_defaults_path().read_text(encoding="utf-8")
    active = [
        line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    body = "\n".join(active)
    for role in SECONDARY_ROLES:
        assert f"{role}:" not in body, f"{role} should inherit, not name a vendor"


def test_the_env_template_documents_every_role_and_credential() -> None:
    from pathlib import Path

    text = Path("config/.env.example").read_text(encoding="utf-8")
    assert "FOLIOAI_API_KEY" in text
    assert "FOLIOAI_BASE_URL" in text
    for role in ALL_ROLES:
        assert f"FOLIOAI_{role.upper()}_MODEL" in text, role


def test_a_bare_settings_object_still_resolves_every_role() -> None:
    """Library use with no files and no environment must not produce a None model."""
    settings = Settings()
    assert all(settings.models.role(role) for role in ALL_ROLES)


def test_nothing_in_the_source_hardcodes_a_model_outside_configuration() -> None:
    """Model names belong in config/.env and default.yaml, nowhere else.

    Catches the pattern that caused this: a vendor's model name reached for in code, where
    no amount of configuration can dislodge it.
    """
    import re
    from pathlib import Path

    # A model name used as a *value*: assigned, or passed as an argument. Naming one inside
    # a sentence is fine and often helpful -- the error for an unknown model says
    # "names are vendor-prefixed, e.g. ..." and is better for the example.
    pattern = re.compile(
        r"""(?x)
        (?: = | \( | , | \[ ) \s*
        ["'](?:openai|anthropic|google|deepseek|meta-llama|mistralai)/
        """
    )
    offenders: list[str] = []
    for path in Path("src/folioai").rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line) and "config.py" not in str(path):
                offenders.append(f"{path}:{number}: {line.strip()}")
    assert not offenders, "model names outside configuration:\n" + "\n".join(offenders)
