# prompts.py
"""
Prompt template loader and renderer for the RCA generation pipeline.

Loads prompts.yaml once at import time (cached via module-level singleton),
then renders system and user prompts using Jinja2 with the structured
RCAInputModel data.

Security notes:
- Jinja2 is configured with undefined=StrictUndefined so missing template
  variables raise immediately rather than silently rendering as empty string.
- autoescape=False is intentional: output is Markdown, not HTML.
  We do NOT want HTML entity escaping in the generated document.
- The YAML file path is resolved relative to this file's location so the
  loader works regardless of the working directory at runtime.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError

from agent_root.models import RCAInputModel

logger = logging.getLogger("rca_generator.prompts")

# Constants
_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"
_TEMPLATE_KEY = "rca_generation"


# Custom exceptions
class PromptLoadError(RuntimeError):
    """Raised when prompts.yaml cannot be loaded or parsed."""


class PromptRenderError(ValueError):
    """Raised when a template variable is missing or rendering fails."""


# Internal helpers
def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Reads and parses prompts.yaml. Raises PromptLoadError on any failure.
    Called once; result is cached by get_prompt_templates().
    """
    if not path.exists():
        raise PromptLoadError(
            f"prompts.yaml not found at expected path: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise PromptLoadError(f"Failed to parse prompts.yaml: {exc}") from exc

    if not isinstance(raw, dict):
        raise PromptLoadError(
            "prompts.yaml must be a YAML mapping at the top level."
        )

    return cast(dict[str, Any], raw)


def _make_jinja_env() -> Environment:
    """
    Returns a Jinja2 Environment configured for safe Markdown rendering.
    StrictUndefined causes any missing variable to raise UndefinedError
    immediately rather than silently rendering as an empty string.
    """
    return Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


# Cached template access
@lru_cache(maxsize=1)
def get_prompt_templates() -> dict[str, str]:
    """
    Loads and validates prompts.yaml once per process.
    Returns the system and user template strings under _TEMPLATE_KEY.

    Raises PromptLoadError if the file is missing, malformed, or does not
    contain the expected keys.
    """
    logger.info("Loading prompt templates from %s", _PROMPTS_PATH)
    data = _load_yaml(_PROMPTS_PATH)

    if _TEMPLATE_KEY not in data:
        raise PromptLoadError(
            f"prompts.yaml is missing the top-level key '{_TEMPLATE_KEY}'."
        )

    section = data[_TEMPLATE_KEY]

    for required_key in ("system", "user"):
        if required_key not in section:
            raise PromptLoadError(
                f"prompts.yaml['{_TEMPLATE_KEY}'] is missing the '{required_key}' key."
            )
        if not isinstance(section[required_key], str) or not section[required_key].strip():
            raise PromptLoadError(
                f"prompts.yaml['{_TEMPLATE_KEY}']['{required_key}'] must be a non-empty string."
            )

    logger.info("Prompt templates loaded successfully.")
    return {
        "system": section["system"],
        "user": section["user"],
    }


def reload_prompt_templates() -> None:
    """
    Clears the lru_cache so prompts.yaml is reloaded on the next call.
    Use in tests or when hot-reloading config. Never call in production
    request handlers.
    """
    get_prompt_templates.cache_clear()
    logger.warning("Prompt template cache cleared.")


# Context builder
def _build_template_context(rca_input: RCAInputModel) -> dict[str, Any]:
    """
    Flattens the nested RCAInputModel into a single dict of template
    variables. All optional fields default to None so Jinja2 renders
    them as "Not available." via the {{ x | default("Not available.") }}
    filters defined in the template.
    """
    tm = rca_input.task_monitoring_data
    pr = rca_input.github_pr
    dev = rca_input.developer_issue_data
    qg = rca_input.quality_gate_data

    context: dict[str, Any] = {
        # TaskMonitoringData
        "issue_logs_id": tm.issue_logs_id,
        "title": tm.title,
        "product": tm.product,
        "client": tm.client,
        "issue_type": tm.issue_type.value,
        "issue_description": tm.issue_description,
        "implement_status": tm.implement_status.value,
        "pre_condition": tm.pre_condition,
        "test_steps": tm.test_steps,
        "expected_result": tm.expected_result,
        "recommended_solution": tm.recommended_solution,
        "error_message": tm.error_message,
        "urgency_level": tm.urgency_level.value,
        "impact_level": tm.impact_level.value,
        "priority_level": tm.priority_level.value,
        "module": tm.module,
        "core_function": tm.core_function,
        "is_recurring": tm.is_recurring,
        # GitHubPRData (all optional)
        "pr_number": pr.pr_number if pr else None,
        "pr_url": pr.pr_url if pr else None,
        "branch_name": pr.branch_name if pr else None,
        "affected_modules": pr.affected_modules if pr else [],
        "fixed_summary": pr.fixed_summary if pr else None,
        "prevention_steps": pr.prevention_steps if pr else None,
        "owner_review": pr.owner_review if pr else None,
        # DeveloperIssueData (all optional)
        "dev_status": dev.dev_status.value if (dev and dev.dev_status) else None,
        "pic_dev": dev.pic_dev if dev else None,
        "dev_resolved_on": (
            dev.dev_resolved_on.strftime("%Y-%m-%d %H:%M UTC")
            if (dev and dev.dev_resolved_on)
            else None
        ),
        "dev_end_date": (
            dev.dev_end_date.strftime("%Y-%m-%d %H:%M UTC")
            if (dev and dev.dev_end_date)
            else None
        ),
        "dev_notes": dev.dev_notes if dev else None,
        # QualityGateData (all optional)
        "validation_status": (
            qg.validation_status.value if (qg and qg.validation_status) else None
        ),
        "fc_failed_testing": qg.fc_failed_testing if qg else 0,
        "existing_report": qg.existing_report if qg else False,
        "quality_gate_first_pass": qg.quality_gate_first_pass if qg else None,
        "smoke_test_first_pass": qg.smoke_test_first_pass if qg else None,
        "reopen_count": qg.reopen_count if qg else 0,
        "qa_status": qg.qa_status.value if (qg and qg.qa_status) else None,
        "qa_validated_on": (
            qg.qa_validated_on.strftime("%Y-%m-%d %H:%M UTC")
            if (qg and qg.qa_validated_on)
            else None
        ),
        "pic_qa": qg.pic_qa if qg else None,
        "remarks": qg.remarks if qg else None,
    }

    return context


# Public rendering API
class RenderedPrompt:
    """Value object holding the rendered system and user prompts."""

    __slots__ = ("system", "user")

    def __init__(self, system: str, user: str) -> None:
        self.system = system
        self.user = user

    def __repr__(self) -> str:
        return (
            f"RenderedPrompt("
            f"system_chars={len(self.system)}, "
            f"user_chars={len(self.user)})"
        )


def render_rca_prompt(rca_input: RCAInputModel) -> RenderedPrompt:
    """
    Renders the system and user prompts for the given RCAInputModel.

    Steps:
      1. Load (or retrieve cached) templates from prompts.yaml.
      2. Flatten the model into a Jinja2 context dict.
      3. Render both templates with StrictUndefined — any missing variable
         raises PromptRenderError immediately.

    Returns a RenderedPrompt with .system and .user string attributes.
    Raises PromptLoadError if templates cannot be loaded.
    Raises PromptRenderError if template rendering fails.
    """
    templates = get_prompt_templates()
    context = _build_template_context(rca_input)
    env = _make_jinja_env()

    try:
        system_prompt = env.from_string(templates["system"]).render(**context)
        user_prompt = env.from_string(templates["user"]).render(**context)
    except UndefinedError as exc:
        raise PromptRenderError(
            f"Template variable missing during RCA prompt rendering: {exc}"
        ) from exc
    except Exception as exc:
        raise PromptRenderError(
            f"Unexpected error during RCA prompt rendering: {exc}"
        ) from exc

    logger.debug(
        "RCA prompt rendered | issue_id=%s | system_chars=%d | user_chars=%d",
        rca_input.task_monitoring_data.issue_logs_id,
        len(system_prompt),
        len(user_prompt),
    )

    return RenderedPrompt(system=system_prompt, user=user_prompt)