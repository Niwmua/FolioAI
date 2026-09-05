"""Prompt rendering (brief §2: Jinja2 templates, one file per prompt).

Templates live in ``llm/prompts/`` as ``.j2`` files rather than as strings in code, so a
prompt change shows up in review as a diff of the prompt rather than of the machinery
around it. Prompts are the highest-leverage part of this system and the part most likely to
be edited by someone who is not editing the code.

Every render is versioned: ``TEMPLATE_VERSION`` participates in nothing directly, but the
rendered text is what the cache fingerprint hashes, so editing a template automatically
invalidates every cached response that used it (D-32).
"""

from __future__ import annotations

import functools
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from .errors import ConfigError
from .paths import package_dir

TEMPLATE_DIR = package_dir() / "llm" / "prompts"

TRANSLATE_SYSTEM = "translate.system.j2"
TRANSLATE_RETRY = "translate.retry.j2"
EVALUATE_SYSTEM = "evaluate.system.j2"
SUMMARIZE_SYSTEM = "summarize.system.j2"
BACKTRANSLATE_SYSTEM = "backtranslate.system.j2"


@functools.lru_cache(maxsize=1)
def environment() -> Environment:
    """Jinja environment for prompt templates.

    ``StrictUndefined`` on purpose: a typo'd variable must blow up at render time, not
    silently render an empty string into a system prompt where nobody will notice that the
    glossary or the style profile went missing.
    """
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=False,  # plain text, not HTML: escaping would corrupt the prose
    )


def render(template_name: str, /, **context: Any) -> str:
    """Render a prompt template.

    Raises:
        ConfigError: if the template is missing, which means a broken install or a typo in
            a template name constant.
    """
    try:
        template = environment().get_template(template_name)
    except TemplateNotFound as exc:
        raise ConfigError(
            f"Prompt template {template_name!r} is missing.",
            remedy=(
                f"Expected it at {TEMPLATE_DIR}. If folioai was installed rather than run "
                "from a checkout, reinstall it: uv sync"
            ),
            context={"template": template_name, "dir": str(TEMPLATE_DIR)},
        ) from exc
    return template.render(**context).strip() + "\n"


def available_templates() -> list[str]:
    return sorted(p.name for p in TEMPLATE_DIR.glob("*.j2"))
