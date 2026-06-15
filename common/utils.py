# common/utils.py
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from agent_root.models import RCAInputModel

logger = logging.getLogger("rca_generator.utils")


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT PATH
# Resolves to <project_root>/agent_root/prompts.yaml regardless of the
# working directory at runtime.
# ─────────────────────────────────────────────────────────────────────────────

prompt_path: Path = (
    Path(__file__).resolve().parent.parent / "agent_root" / "prompts.yaml"
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class PromptLoadError(RuntimeError):
    """
    Raised when prompts.yaml cannot be found, read, or parsed.

    Covers:
    - File not found at prompt_path
    - Invalid YAML syntax
    - Top-level structure is not a dict
    - Missing required keys (e.g. 'rca_generation', 'system', 'user')
    - Empty prompt string values
    """


class PromptRenderError(ValueError):
    """
    Raised when Jinja2 rendering of a prompt template fails.

    Covers:
    - Missing template variable (StrictUndefined → UndefinedError)
    - Any unexpected rendering exception
    """


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

# Internal: YAML key that holds the RCA prompt pair
_TEMPLATE_KEY = "rca_generation"


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Reads and parses a YAML file at the given path.

    Args:
        path: Absolute path to the YAML file (typically prompt_path).

    Returns:
        Parsed YAML content as dict[str, Any].

    Raises:
        PromptLoadError: If file is missing, unreadable, or not valid YAML.
    """
    if not path.exists():
        raise PromptLoadError(
            f"prompts.yaml not found at expected path: {path}\n"
            f"Create the file at <project_root>/agent_root/prompts.yaml."
        )

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise PromptLoadError(
            f"Failed to parse prompts.yaml at '{path}': {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise PromptLoadError(
            "prompts.yaml must be a YAML mapping at the top level. "
            f"Got: {type(raw).__name__}"
        )

    return cast(dict[str, Any], raw)


def _make_jinja_env() -> Environment:
    """
    Returns a Jinja2 Environment configured for Markdown output.

    - StrictUndefined: missing variables raise UndefinedError immediately.
    - autoescape=False: no HTML escaping (output is Markdown).
    - keep_trailing_newline=True: preserves trailing newlines in templates.
    """
    return Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def load_sys_prompt(file_path: Path) -> dict[str, str]:
    """
    Loads and validates prompts.yaml, returning prompt templates as a
    dict[str, str] where keys are prompt names and values are prompt strings.

    Args:
        file_path: Path to prompts.yaml (pass prompt_path from this module).

    Returns:
        dict[str, str] — e.g.:
            {
                "system": "<full system prompt string>",
                "user":   "<full user prompt template string>",
            }

    Raises:
        PromptLoadError: If the file is missing, malformed, or keys are absent.

    Example:
        >>> from common.utils import load_sys_prompt, prompt_path
        >>> prompts = load_sys_prompt(prompt_path)
        >>> prompts["system"]   # → raw system prompt string
        >>> prompts["user"]     # → raw user prompt template string
    """
    logger.info("Loading prompt templates from: %s", file_path)
    data = _load_yaml(file_path)

    if _TEMPLATE_KEY not in data:
        raise PromptLoadError(
            f"prompts.yaml is missing the top-level key '{_TEMPLATE_KEY}'. "
            f"Available keys: {list(data.keys())}"
        )

    section = data[_TEMPLATE_KEY]

    for required_key in ("system", "user"):
        if required_key not in section:
            raise PromptLoadError(
                f"prompts.yaml['{_TEMPLATE_KEY}'] is missing the "
                f"'{required_key}' key."
            )
        if (
            not isinstance(section[required_key], str)
            or not section[required_key].strip()
        ):
            raise PromptLoadError(
                f"prompts.yaml['{_TEMPLATE_KEY}']['{required_key}'] "
                f"must be a non-empty string."
            )

    logger.info("Prompt templates loaded successfully.")

    return {
        "system": section["system"],
        "user":   section["user"],
    }


@lru_cache(maxsize=1)
def get_prompt_templates() -> dict[str, str]:
    """
    Cached wrapper around load_sys_prompt(prompt_path).
    Loads prompts.yaml once per process and caches the result.

    Returns:
        dict[str, str] with "system" and "user" keys.

    Raises:
        PromptLoadError: Propagated from load_sys_prompt.
    """
    return load_sys_prompt(prompt_path)


def reload_prompt_templates() -> None:
    """
    Clears the lru_cache so prompts.yaml is reloaded on the next call.
    Use in tests or hot-reload scenarios only — never in production handlers.
    """
    get_prompt_templates.cache_clear()
    logger.warning("Prompt template cache cleared — will reload on next call.")


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# Flattens RCAInputModel into a flat dict[str, Any] for Jinja2 rendering.
# ─────────────────────────────────────────────────────────────────────────────

def build_rca_template_context(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> dict[str, Any]:
    """
    Flattens the nested RCAInputModel into a single dict of Jinja2 template
    variables.

    All optional sub-model fields default to None so the Jinja2 template's
    {{ x | default("Not available.", true) }} filters handle missing values
    gracefully without raising UndefinedError.

    Args:
        rca_input:       Validated RCAInputModel instance.
        quality_summary: Pre-computed quality gate summary string produced by
                         the analyze_quality_gates graph node. Defaults to ""
                         so direct callers (e.g. unit tests) that bypass the
                         graph don't need to supply it.

    Returns:
        Flat dict[str, Any] ready to be unpacked into a Jinja2 template.
    """
    tm  = rca_input.task_monitoring_data
    pr  = rca_input.github_pr
    dev = rca_input.developer_issue_data
    qg  = rca_input.quality_gate_data

    context: dict[str, Any] = {

        # ── TaskMonitoringData ────────────────────────────────────────────
        "issue_logs_id":        tm.issue_logs_id,
        "title":                tm.title,
        "product":              tm.product,
        "client":               tm.client,
        "issue_type":           tm.issue_type.value,
        "issue_description":    tm.issue_description,
        "implement_status":     tm.implement_status.value,
        "pre_condition":        tm.pre_condition,
        "test_steps":           tm.test_steps,
        "expected_result":      tm.expected_result,
        "recommended_solution": tm.recommended_solution,
        "error_message":        tm.error_message,
        "urgency_level":        tm.urgency_level.value,
        "impact_level":         tm.impact_level.value,
        "priority_level":       tm.priority_level.value,
        "module":               tm.module,
        "core_function":        tm.core_function,
        "is_recurring":         tm.is_recurring,

        # ── GitHubPRData (all optional — None if pr is absent) ────────────
        "pr_number":        pr.pr_number           if pr else None,
        "pr_url":           str(pr.pr_url)         if (pr and pr.pr_url) else None,
        "branch_name":      pr.branch_name         if pr else None,
        "affected_modules": pr.affected_modules    if pr else [],
        "fixed_summary":    pr.fixed_summary       if pr else None,
        "prevention_steps": pr.prevention_steps    if pr else None,
        "owner_review":     pr.owner_review        if pr else None,

        # ── DeveloperIssueData (all optional) ────────────────────────────
        "dev_status": dev.dev_status.value if (dev and dev.dev_status) else None,
        "pic_dev":    dev.pic_dev          if dev else None,
        "dev_resolved_on": (
            dev.dev_resolved_on.strftime("%Y-%m-%d %H:%M UTC")
            if (dev and dev.dev_resolved_on) else None
        ),
        "dev_end_date": (
            dev.dev_end_date.strftime("%Y-%m-%d %H:%M UTC")
            if (dev and dev.dev_end_date) else None
        ),

        # Fact-locked developer evidence — REQUIRED by prompts.yaml for
        # the "CONFIRMED FACTS" block and Sections 2/4/6/8. Previously
        # missing here, causing the RCA generator to fall back to
        # "Dev Notes" only and produce generic wording.
        "affected_component":  dev.affected_component  if dev else None,
        "root_cause":          dev.root_cause           if dev else None,
        "fix_applied":         dev.fix_applied          if dev else None,
        "verification_result": dev.verification_result if dev else None,

        "dev_notes": dev.dev_notes if dev else None,

        # ── QualityGateData (all optional) ───────────────────────────────
        "validation_status": (
            qg.validation_status.value if (qg and qg.validation_status) else None
        ),
        "fc_failed_testing":       qg.fc_failed_testing       if qg else 0,
        "existing_report":         qg.existing_report         if qg else False,
        "quality_gate_first_pass": qg.quality_gate_first_pass if qg else None,
        "smoke_test_first_pass":   qg.smoke_test_first_pass   if qg else None,
        "reopen_count":            qg.reopen_count            if qg else 0,
        "qa_status": (
            qg.qa_status.value if (qg and qg.qa_status) else None
        ),
        "qa_validated_on": (
            qg.qa_validated_on.strftime("%Y-%m-%d %H:%M UTC")
            if (qg and qg.qa_validated_on) else None
        ),
        "pic_qa":  qg.pic_qa  if qg else None,
        "remarks": qg.remarks if qg else None,

        # ── Quality gate summary (pre-computed by graph node) ─────────────
        # Passed in as a parameter; defaults to "" for direct callers that
        # bypass the LangGraph pipeline (e.g. unit tests).
        "quality_summary": quality_summary,

        # ── Attachments ───────────────────────────────────────────────────
        # Always a list — empty list when no attachments are present.
        # The Jinja2 template uses {% if attachments %} to conditionally
        # render the attachments block, so an empty list is safe.
        "attachments": rca_input.attachments,
    }

    return context


# ─────────────────────────────────────────────────────────────────────────────
# RENDERED PROMPT VALUE OBJECT
# ─────────────────────────────────────────────────────────────────────────────

class RenderedPrompt:
    """
    Immutable value object holding the fully rendered system and user prompts.

    Attributes:
        system: Rendered system prompt string (passed to ChatPromptTemplate).
        user:   Rendered user prompt string   (passed to ChatPromptTemplate).
    """

    __slots__ = ("system", "user")

    def __init__(self, system: str, user: str) -> None:
        self.system = system
        self.user   = user

    def __repr__(self) -> str:
        return (
            f"RenderedPrompt("
            f"system_chars={len(self.system)}, "
            f"user_chars={len(self.user)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_rca_prompt(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> RenderedPrompt:
    """
    Renders system and user prompt templates for a given RCAInputModel.

    Steps:
        1. Load (or retrieve cached) raw templates via get_prompt_templates().
        2. Flatten rca_input into a Jinja2 context via build_rca_template_context().
        3. Render both templates — StrictUndefined raises PromptRenderError
           immediately on any missing variable.

    Args:
        rca_input:       Validated RCAInputModel instance.
        quality_summary: Pre-computed quality gate summary string produced by
                         the analyze_quality_gates graph node. Forwarded to
                         build_rca_template_context. Defaults to "" so direct
                         callers that bypass the graph don't need to supply it.

    Returns:
        RenderedPrompt with .system and .user string attributes.

    Raises:
        PromptLoadError:   If templates cannot be loaded from prompts.yaml.
        PromptRenderError: If a template variable is missing or rendering fails.
    """
    templates = get_prompt_templates()
    context   = build_rca_template_context(rca_input, quality_summary=quality_summary)
    env       = _make_jinja_env()

    try:
        system_prompt = env.from_string(templates["system"]).render(**context)
        user_prompt   = env.from_string(templates["user"]).render(**context)
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


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN CHATPROMPTTEMPLATE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_rca_chat_prompt(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> ChatPromptTemplate:
    """
    Builds a LangChain ChatPromptTemplate from the rendered RCA prompts.

    The system and user prompts are fully Jinja2-rendered before being wrapped
    into LangChain message templates — no further variable substitution occurs.
    Pass an empty dict ({}) when invoking the returned chain.

    Args:
        rca_input:       Validated RCAInputModel instance.
        quality_summary: Pre-computed quality gate summary string produced by
                         the analyze_quality_gates graph node. Forwarded to
                         render_rca_prompt. Defaults to "" so direct callers
                         that bypass the graph don't need to supply it.

    Returns:
        ChatPromptTemplate composed of:
            - SystemMessagePromptTemplate  (rendered system prompt)
            - HumanMessagePromptTemplate   (rendered user prompt)

    Raises:
        PromptLoadError:   Propagated from render_rca_prompt.
        PromptRenderError: Propagated from render_rca_prompt.

    Example:
        >>> from common.utils import build_rca_chat_prompt
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>>
        >>> llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        >>> prompt = build_rca_chat_prompt(rca_input)
        >>> chain  = prompt | llm
        >>> result = chain.invoke({})   # already fully rendered — pass empty dict
    """
    rendered = render_rca_prompt(rca_input, quality_summary=quality_summary)

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(  # type: ignore[assignment]
        [
            SystemMessagePromptTemplate.from_template(rendered.system),
            HumanMessagePromptTemplate.from_template(rendered.user),
        ]
    )
    return prompt