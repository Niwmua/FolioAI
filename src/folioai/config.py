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

from .errors import ConfigError
from .paths import ENV_PREFIX, user_config_path

PACKAGED_DEFAULTS = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"
PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "profiles"

Severity = Literal["critical", "warning", "info"]


class _Model(BaseModel):
    """Base: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelsConfig(_Model):
    translator: str = "anthropic/claude-sonnet-4.5"
    evaluator: str = "openai/gpt-4.1"
    escalation: str = "anthropic/claude-opus-4.1"
    summarizer: str = "openai/gpt-4.1-mini"
    glossary: str = "openai/gpt-4.1"
    back_translator: str = "openai/gpt-4.1-mini"
    vision: str = "openai/gpt-4.1"


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
    formats: list[str] = Field(default_factory=lambda: ["md"])
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

    _sources: list[str] = []


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


_FLAT_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "API_KEY": ("api_key",),
    "BASE_URL": ("llm", "base_url"),
    "LOG_LEVEL": ("logging", "level"),
    "TRANSLATOR_MODEL": ("models", "translator"),
    "EVALUATOR_MODEL": ("models", "evaluator"),
    "ESCALATION_MODEL": ("models", "escalation"),
    "MAX_COST": ("budget", "max_cost_usd"),
    "CONCURRENCY": ("translation", "concurrency"),
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
    """Minimal ``.env`` reader. Values already in the environment win."""
    if not path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value
            loaded[name] = value
    return loaded


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

    merged: dict[str, Any] = {}
    if PACKAGED_DEFAULTS.is_file():
        merged = _load_yaml(PACKAGED_DEFAULTS)
        sources.append("packaged defaults")

    for candidate, label in (
        (user_config_path(), "user config"),
        (project_dir / "folioai.yaml", "project config"),
    ):
        if candidate.is_file():
            merged = _deep_merge(merged, _load_yaml(candidate))
            sources.append(f"{label} ({candidate})")

    if extra_config is not None:
        if not extra_config.is_file():
            raise ConfigError(
                f"Config file not found: {extra_config}",
                remedy="Check the path passed to --config.",
            )
        merged = _deep_merge(merged, _load_yaml(extra_config))
        sources.append(f"--config ({extra_config})")

    load_dotenv(project_dir / ".env")
    env_overlay = config_from_env(environ)
    if env_overlay:
        merged = _deep_merge(merged, env_overlay)
        sources.append(f"{ENV_PREFIX}* environment")

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
        sources.append("command line")

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
    return settings


def available_profiles() -> list[str]:
    """Names of the shipped style profiles."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name_or_path: str) -> dict[str, Any]:
    """Load a style profile by shipped name or filesystem path."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return _load_yaml(candidate)
    packaged = PROFILES_DIR / f"{name_or_path}.yaml"
    if packaged.is_file():
        return _load_yaml(packaged)
    raise ConfigError(
        f"Unknown style profile: {name_or_path!r}.",
        remedy=(
            "Pass a path to a YAML profile, or one of the shipped profiles: "
            + ", ".join(available_profiles())
        ),
    )
