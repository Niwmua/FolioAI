"""Configuration: packaged defaults, YAML files, environment, and CLI flags.

Precedence, highest first (brief §16)::

    CLI flags  >  FOLIOAI_* env vars  >  ./folioai.yaml  >  ~/.folioai/config.yaml  >  packaged

Everything is a Pydantic v2 model, so a typo in a YAML key is an error at load time rather
than a silently ignored setting. API keys are never read from a config file (§16) -- only
from the environment or a ``.env`` -- and never stored on the settings object's repr.

Environment variables map onto the nested structure with a double underscore::

    FOLIOAI_TRANSLATION__BATCH_TOKENS=900
    FOLIOAI_MODELS__TRANSLATOR=openai/gpt-4.1

with a handful of flat aliases for the things people set constantly (``FOLIOAI_API_KEY``,
``FOLIOAI_BASE_URL``, ``FOLIOAI_LOG_LEVEL``).
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .env import load_env
from .errors import ConfigError
from .paths import (
    ENV_PREFIX,
    packaged_defaults_path,
    profiles_dir,
    user_config_path,
)


# Resolved on every use rather than captured at import: FOLIOAI_CONFIG_DIR can point these
# somewhere else, and a module-level constant would freeze whatever was set when the first
# import happened -- which, in a test suite, is whatever ran first.
def _packaged_defaults() -> Path:
    return packaged_defaults_path()


def _profiles_dir() -> Path:
    return profiles_dir()


Severity = Literal["critical", "warning", "info"]


class _Model(BaseModel):
    """Base: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelsConfig(_Model):
    """Which model plays each role.

    Only ``translator`` and ``evaluator`` have shipped defaults. Every other role is
    ``None`` until set, and then **inherits the translator**.

    That inheritance is the important part. Naming a vendor's model as the default for
    five secondary roles means a user who configures two models against their own gateway
    gets three-quarters of a working system: translation and evaluation succeed, and then
    attempt 3 of the retry ladder and the rolling summary call a model the endpoint has
    never heard of, halfway through a book. Inheriting a model the user has actually named
    is always safer than guessing at one they have not.
    """

    translator: str = "anthropic/claude-sonnet-4.5"
    evaluator: str = "openai/gpt-4.1"
    escalation: str | None = None
    summarizer: str | None = None
    glossary: str | None = None
    back_translator: str | None = None
    vision: str | None = None

    #: Roles that fell back to the translator, for the provenance display.
    inherited: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _inherit_unset_roles(self) -> ModelsConfig:
        inherited: list[str] = []
        for role in ("escalation", "summarizer", "glossary", "back_translator", "vision"):
            if getattr(self, role) is None:
                object.__setattr__(self, role, self.translator)
                inherited.append(role)
        object.__setattr__(self, "inherited", tuple(inherited))
        return self

    def role(self, name: str) -> str:
        """The model for a role, guaranteed to be a string after validation."""
        value = getattr(self, name)
        if not value:  # pragma: no cover - the validator fills every role
            return self.translator
        return str(value)


class LLMConfig(_Model):
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = 180.0
    connect_timeout_s: float = 15.0
    max_transient_retries: int = Field(default=5, ge=0, le=20)
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 60.0
    backoff_jitter: float = Field(default=0.25, ge=0.0, le=1.0)
    rpm: int = Field(default=60, gt=0)
    tpm: int = Field(default=120_000, gt=0)
    cache_enabled: bool = True
    seed: int | None = 7


class TranslationConfig(_Model):
    batch_tokens: int = Field(default=1200, gt=0)
    concurrency: int = Field(default=4, gt=0, le=64)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_completion_ratio: float = Field(
        default=3.0, gt=1.0, description="Completion token budget as a multiple of source tokens."
    )


class ContextConfig(_Model):
    previous_target_blocks: int = Field(default=2, ge=0)
    next_source_blocks: int = Field(default=1, ge=0)
    summary_every: int = Field(default=8, gt=0)
    summary_max_words: int = Field(default=150, gt=0)


class RubricWeights(_Model):
    completeness: float = 0.35
    accuracy: float = 0.30
    terminology: float = 0.15
    fluency: float = 0.15
    formatting: float = 0.05

    @model_validator(mode="after")
    def _sum_to_one(self) -> RubricWeights:
        total = (
            self.completeness + self.accuracy + self.terminology + self.fluency + self.formatting
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rubric weights must sum to 1.0, got {total:.4f}")
        return self


class EvaluationConfig(_Model):
    mode: Literal["direct", "back-translation", "both"] = "direct"
    sample: float = Field(default=1.0, ge=0.0, le=1.0)
    min_score: int = Field(default=80, ge=0, le=100)
    completeness_floor: int = Field(default=70, ge=0, le=100)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    batch_tokens: int = Field(default=2400, gt=0)
    both_mode_band: tuple[int, int] = (80, 90)
    show_validation_warnings: bool = True
    weights: RubricWeights = Field(default_factory=RubricWeights)


class RetryConfig(_Model):
    max_attempts: int = Field(default=3, ge=1, le=10)
    attempt_temperatures: list[float] = Field(default_factory=lambda: [0.2, 0.3, 0.0])
    breaker_failure_rate: float = Field(default=0.25, gt=0.0, le=1.0)
    breaker_min_failures: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _temps_cover_attempts(self) -> RetryConfig:
        if len(self.attempt_temperatures) < self.max_attempts:
            raise ValueError(
                "retry.attempt_temperatures must have at least max_attempts entries "
                f"({self.max_attempts}), got {len(self.attempt_temperatures)}"
            )
        return self


class ProbeConfig(_Model):
    sample_pages: int = Field(default=3, ge=1)
    sample_skip_front_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    garbling_replacement_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    garbling_control_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    garbling_nonword_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    min_chars_for_text_layer: int = Field(default=200, ge=0)
    column_gap_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    column_min_share: float = Field(default=0.25, ge=0.0, le=1.0)


class CleaningConfig(_Model):
    """Every step is individually toggleable (brief §4.3, D-15)."""

    strip_running_heads: bool = True
    strip_page_numbers: bool = True
    dehyphenate: bool = True
    reflow_paragraphs: bool = True
    normalize_unicode: bool = True
    repair_drop_caps: bool = True
    extract_footnotes: bool = True
    classify_matter: bool = True

    header_page_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    header_similarity: float = Field(default=0.9, ge=0.0, le=1.0)
    furniture_margin_fraction: float = Field(
        default=0.12,
        gt=0.0,
        le=0.5,
        description="Fraction of page height at top/bottom where furniture can live.",
    )
    paragraph_gap_multiple: float = Field(
        default=1.6, gt=1.0, description="Vertical gap over this multiple of line height breaks."
    )
    drop_cap_size_ratio: float = Field(default=1.8, gt=1.0)
    footnote_size_ratio: float = Field(default=0.85, gt=0.0, lt=1.0)


class ExtractionConfig(_Model):
    extractor: Literal["auto", "pymupdf", "poppler", "ocr", "marker"] = "auto"
    ocr_lang: str | None = None
    vision_fallback: bool = False
    vision_max_pages: int = Field(default=10, ge=0)
    translate_front_matter: bool = True
    regenerate_toc: bool = True
    min_heading_size_ratio: float = Field(default=1.15, gt=1.0)
    chapter_patterns: list[str] = Field(
        default_factory=lambda: [
            r"^\s*chapter\s+([0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            r"^\s*part\s+([0-9]+|[ivxlcdm]+)\b",
            r"^\s*([0-9]{1,3})\s*$",
        ]
    )
    scene_break_patterns: list[str] = Field(
        default_factory=lambda: [r"^\s*[\*•·—\-]{1,5}(\s+[\*•·—\-]{1,5}){0,4}\s*$"]
    )


class ExportConfig(_Model):
    """Output settings, including where to find the tools that produce PDFs and check EPUBs.

    The tool paths are configurable rather than PATH-only because both are commonly just
    downloaded: Typst is a single binary, and epubcheck is a jar that needs a JVM. Requiring
    someone to modify PATH before they can export a PDF is a bad trade.
    """

    formats: list[str] = Field(default_factory=lambda: ["md"])
    typst_path: Path | None = Field(
        default=None, description="Typst binary. Searched on PATH and in the bin dir if unset."
    )
    epubcheck_path: Path | None = Field(
        default=None,
        description="epubcheck binary or .jar. Searched on PATH and in the bin dir if unset.",
    )
    font_paths: list[Path] = Field(
        default_factory=list,
        description="Extra font directories for the PDF renderer, beyond the fonts dir.",
    )
    layout: Literal["target-only", "bilingual-paragraph", "bilingual-columns", "annotated"] = (
        "target-only"
    )
    split_chapters: bool = False
    pdf_engine: Literal["auto", "typst", "weasyprint"] = "auto"
    cover: Path | None = None


class BudgetConfig(_Model):
    max_cost_usd: float | None = None
    expected_retry_rate: float = Field(default=0.15, ge=0.0, le=1.0)


class ModelPrice(_Model):
    prompt: float = Field(ge=0.0, description="USD per 1M prompt tokens.")
    completion: float = Field(ge=0.0, description="USD per 1M completion tokens.")


class LoggingConfig(_Model):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    tty: bool = True


class Settings(_Model):
    """The fully merged configuration for one invocation."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)

    # Populated from the environment only; never from a YAML file.
    api_key: str | None = Field(default=None, exclude=True, repr=False)

    @field_validator("pricing", mode="before")
    @classmethod
    def _coerce_pricing(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if v is not None}
        return value

    def price_for(self, model: str) -> ModelPrice | None:
        """Look up a model's price, or ``None`` if unknown (D-33: warn, never crash)."""
        return self.pricing.get(model)

    def source_description(self) -> str:
        return ", ".join(self._sources) if self._sources else "packaged defaults"

    def origin(self, dotted: str) -> str:
        """Which source last set a setting, e.g. ``models.translator``.

        The answer to "is it really using my .env?" should be checkable rather than
        promised, which is what this and ``folioai config`` are for.
        """
        return self._provenance.get(dotted, "built-in default")

    _sources: list[str] = []
    _provenance: dict[str, str] = {}


def flatten_keys(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict to ``section.key`` form, for provenance tracking."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_keys(value, f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``. Lists replace, they do not concatenate."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Could not parse the YAML in {path}.",
            remedy="Fix the syntax error reported below and re-run.",
            context={"path": str(path), "error": str(exc)},
        ) from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a YAML mapping at the top level, not {type(raw).__name__}.",
            remedy="Wrap the contents in top-level keys such as 'models:' or 'translation:'.",
            context={"path": str(path)},
        )
    return raw


#: Short names for the settings people actually set, so a .env does not have to be written
#: in FOLIOAI_SECTION__KEY form. Every model role has one: a role without an alias is a role
#: that quietly keeps its default when someone thinks they have configured everything.
_FLAT_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "API_KEY": ("api_key",),
    "BASE_URL": ("llm", "base_url"),
    "LOG_LEVEL": ("logging", "level"),
    "TRANSLATOR_MODEL": ("models", "translator"),
    "EVALUATOR_MODEL": ("models", "evaluator"),
    "ESCALATION_MODEL": ("models", "escalation"),
    "SUMMARIZER_MODEL": ("models", "summarizer"),
    "GLOSSARY_MODEL": ("models", "glossary"),
    "BACK_TRANSLATOR_MODEL": ("models", "back_translator"),
    "VISION_MODEL": ("models", "vision"),
    "TYPST_PATH": ("export", "typst_path"),
    "EPUBCHECK": ("export", "epubcheck_path"),
    "PDF_ENGINE": ("export", "pdf_engine"),
    "MAX_COST": ("budget", "max_cost_usd"),
    "CONCURRENCY": ("translation", "concurrency"),
    "TIMEOUT": ("llm", "timeout_s"),
    "RPM": ("llm", "rpm"),
    "TPM": ("llm", "tpm"),
    "MIN_SCORE": ("evaluation", "min_score"),
    "EVAL_MODE": ("evaluation", "mode"),
    "EVAL_SAMPLE": ("evaluation", "sample"),
    "BATCH_TOKENS": ("translation", "batch_tokens"),
    "MAX_ATTEMPTS": ("retry", "max_attempts"),
}

# Env keys that name a location rather than a setting.
_ENV_IGNORED = {"HOME"}


def _coerce_scalar(text: str) -> Any:
    """Interpret an env var the way YAML would, so ``true``/``12``/``0.3`` arrive typed."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    return text if value is None else value


def _assign(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = target
    for part in path[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[path[-1]] = value


def config_from_env(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a config overlay from ``FOLIOAI_*`` variables."""
    env = os.environ if environ is None else environ
    overlay: dict[str, Any] = {}
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        suffix = key[len(ENV_PREFIX) :]
        if suffix in _ENV_IGNORED or not raw:
            continue
        if suffix in _FLAT_ENV_ALIASES:
            _assign(overlay, _FLAT_ENV_ALIASES[suffix], _coerce_scalar(raw))
            continue
        if "__" in suffix:
            path = tuple(part.lower() for part in suffix.split("__"))
            _assign(overlay, path, _coerce_scalar(raw))
    return overlay


def _api_key_from_env(environ: dict[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    for name in (f"{ENV_PREFIX}API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = env.get(name)
        if value:
            return value
    return None


def load_dotenv(path: Path) -> dict[str, str]:
    """Load one ``.env`` file. Values already in the environment win.

    Kept as the historical name; the implementation lives in :mod:`folioai.env` alongside
    the rest of the file-location logic.
    """
    from .env import load_dotenv_file

    return load_dotenv_file(path)


def load_settings(
    *,
    cli_overrides: dict[str, Any] | None = None,
    project_dir: Path | None = None,
    extra_config: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Merge every configuration source and validate the result.

    Args:
        cli_overrides: Nested dict from parsed CLI flags. Highest precedence.
        project_dir: Directory to look in for ``folioai.yaml``. Defaults to cwd.
        extra_config: An explicit ``--config`` file, applied just below CLI flags.
        environ: Environment override, for tests.

    Raises:
        ConfigError: if any source is malformed or a value fails validation.
    """
    project_dir = project_dir or Path.cwd()
    sources: list[str] = []

    # .env first: it can move every path this function is about to read from.
    load_env(project_dir)

    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    def apply(overlay: dict[str, Any], label: str) -> None:
        """Merge one source and record which keys it set."""
        nonlocal merged
        if not overlay:
            return
        merged = _deep_merge(merged, overlay)
        for dotted in flatten_keys(overlay):
            provenance[dotted] = label
        sources.append(label)

    defaults = _packaged_defaults()
    if defaults.is_file():
        apply(_load_yaml(defaults), f"packaged defaults ({defaults.name})")

    for candidate, label in (
        (user_config_path(), "user config"),
        (project_dir / "folioai.yaml", "project config"),
    ):
        if candidate.is_file():
            apply(_load_yaml(candidate), f"{label} ({candidate})")

    if extra_config is not None:
        if not extra_config.is_file():
            raise ConfigError(
                f"Config file not found: {extra_config}",
                remedy="Check the path passed to --config.",
            )
        apply(_load_yaml(extra_config), f"--config ({extra_config})")

    # The .env files were already loaded into the environment by load_env() above, so this
    # single layer covers both them and anything exported in the shell. Which of the two it
    # was is reported separately by `folioai config`, from the files themselves.
    apply(config_from_env(environ), f"{ENV_PREFIX}* environment / .env")

    apply(cli_overrides or {}, "command line")

    # Keys never accepted from a file (§16).
    merged.pop("api_key", None)

    try:
        settings = Settings.model_validate(merged)
    except Exception as exc:  # pydantic ValidationError, deliberately broadened
        raise ConfigError(
            "Configuration is invalid.",
            remedy=(
                "Fix the offending key in your YAML, environment variable, or flag. "
                "The failing field path is shown below."
            ),
            context={"sources": sources, "error": str(exc)},
        ) from exc

    settings.api_key = _api_key_from_env(environ)
    settings._sources = sources
    settings._provenance = provenance
    return settings


def packaged_settings() -> Settings:
    """Settings from the packaged defaults alone.

    No user file, no project file, no environment. Used by tests and by library callers who
    want the shipped configuration without whatever happens to be on the machine -- notably
    the ``pricing:`` table, which lives in YAML by design (§13) and so is absent from a bare
    ``Settings()``.
    """
    defaults = _packaged_defaults()
    data = _load_yaml(defaults) if defaults.is_file() else {}
    data.pop("api_key", None)
    return Settings.model_validate(data)


def available_profiles() -> list[str]:
    """Names of the shipped style profiles."""
    directory = _profiles_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_profile(name_or_path: str) -> dict[str, Any]:
    """Load a style profile by shipped name or filesystem path."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return _load_yaml(candidate)
    packaged = _profiles_dir() / f"{name_or_path}.yaml"
    if packaged.is_file():
        return _load_yaml(packaged)
    raise ConfigError(
        f"Unknown style profile: {name_or_path!r}.",
        remedy=(
            "Pass a path to a YAML profile, or one of the shipped profiles: "
            + ", ".join(available_profiles())
        ),
    )
