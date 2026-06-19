from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from agent_root.models import RCAInputModel

logger = logging.getLogger("rca_generator.utils")


# Prompt Path
prompt_path: Path = (
    Path(__file__).resolve().parent.parent / "agent_root" / "prompts.yaml"
)


# Audit Metadata Constants
PROMPT_VERSION = "1.1.0"
MODEL_NAME = "gemini-2.5-flash"


# Custom Exceptions
class PromptLoadError(RuntimeError):
    """Raised when prompts.yaml cannot be found, read, parsed, or validated."""


class PromptRenderError(ValueError):
    """Raised when Jinja2 rendering of a prompt template fails."""


# Prompt Utilities
_TEMPLATE_KEY = "rca_generation"
_ISSUE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_\-]")


def _sanitize_issue_id(issue_id: str) -> str:
    sanitized = _ISSUE_ID_PATTERN.sub("", issue_id)

    if sanitized != issue_id:
        logger.warning("issue_id sanitized: '%s' → '%s'", issue_id, sanitized)

    return sanitized


sanitize_issue_id = _sanitize_issue_id


def _load_yaml(path: Path) -> dict[str, Any]:
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
    return Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def load_sys_prompt(file_path: Path) -> dict[str, str]:
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

    return {
        "system": section["system"],
        "user": section["user"],
    }


def load_section_prompts(file_path: Path, section_key: str) -> dict[str, str]:
    logger.info(
        "Loading section prompt templates | section=%s | file=%s",
        section_key,
        file_path,
    )

    data = _load_yaml(file_path)

    if section_key not in data:
        raise PromptLoadError(
            f"prompts.yaml is missing the top-level section key '{section_key}'. "
            f"Available keys: {list(data.keys())}"
        )

    section = data[section_key]

    for required_key in ("system", "user"):
        if required_key not in section:
            raise PromptLoadError(
                f"prompts.yaml['{section_key}'] is missing the "
                f"'{required_key}' key."
            )

        if (
            not isinstance(section[required_key], str)
            or not section[required_key].strip()
        ):
            raise PromptLoadError(
                f"prompts.yaml['{section_key}']['{required_key}'] "
                f"must be a non-empty string."
            )

    return {
        "system": section["system"],
        "user": section["user"],
    }


@lru_cache(maxsize=1)
def get_prompt_templates() -> dict[str, str]:
    return load_sys_prompt(prompt_path)


@lru_cache(maxsize=32)
def get_section_prompt_templates(section_key: str) -> dict[str, str]:
    return load_section_prompts(prompt_path, section_key)


def reload_prompt_templates() -> None:
    get_prompt_templates.cache_clear()
    logger.warning("Prompt template cache cleared — will reload on next call.")


def reload_section_prompt_templates(section_key: str | None = None) -> None:
    get_section_prompt_templates.cache_clear()
    logger.warning(
        "Section prompt template cache cleared | section=%s",
        section_key if section_key else "ALL",
    )


# Input Completeness Scoring
def _compute_input_quality_score(
    root_cause: str | None,
    fix_applied: str | None,
    verification_result: str | None,
    technical_evidence: str | None,
) -> int:
    score = 0

    if root_cause:
        score += 25

    if fix_applied:
        score += 25

    if verification_result:
        score += 25

    if technical_evidence:
        score += 25

    return score


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR PLAN DEFAULT
#
# FIX: supervisor_plan is injected by graph.py only for section agents that
# run AFTER run_supervisor. The supervisor's own prompts (rca_supervisor,
# rca_supervisor_final_review) and any section rendered before supervisor_plan
# is written to state would trigger a StrictUndefined Jinja2 error because
# {{ supervisor_plan }} appears in every section's system prompt but the
# variable is only present in extra_context for downstream agents.
#
# Solution: always inject a safe default for supervisor_plan (and the two
# supplementary keys referenced in some templates) before rendering any
# section prompt. extra_context passed by graph.py will override the default
# for agents that have the real plan available.
# ─────────────────────────────────────────────────────────────────────────────

_SUPERVISOR_PLAN_DEFAULT: dict[str, Any] = {
    "route_decision": "CONTINUE",
    "execution_strategy": "",
    "missing_data": [],
    "placeholder_fields": [],
    "evidence_anchors": [],
    "supervisor_notes": "",
    "data_authenticity": "UNKNOWN",
    "overall_data_quality": "UNKNOWN",
    "required_sections": [],
    "critical_areas": [],
    "data_conflicts": [],
    "regeneration_targets": [],
    "section_quality_thresholds": {},
    "section_plan": {},
    "execution_sequence": [],
}


# Context Builder
def build_rca_template_context(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> dict[str, Any]:
    tm = rca_input.task_monitoring_data
    pr = rca_input.github_pr
    dev = rca_input.developer_issue_data
    qg = rca_input.quality_gate_data

    root_cause = dev.root_cause if dev else None
    fix_applied = dev.fix_applied if dev else None
    verification_result = dev.verification_result if dev else None
    technical_evidence = dev.technical_evidence if dev else None

    input_quality_score = _compute_input_quality_score(
        root_cause=root_cause,
        fix_applied=fix_applied,
        verification_result=verification_result,
        technical_evidence=technical_evidence,
    )

    logger.info(
        "Input quality score for issue '%s': %d/100",
        tm.issue_logs_id,
        input_quality_score,
    )

    audit_generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    context: dict[str, Any] = {
        "issue_logs_id": sanitize_issue_id(tm.issue_logs_id),
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
        "pr_number": pr.pr_number if pr else None,
        "pr_url": str(pr.pr_url) if (pr and pr.pr_url) else None,
        "branch_name": pr.branch_name if pr else None,
        "affected_modules": pr.affected_modules if pr else [],
        "fixed_summary": pr.fixed_summary if pr else None,
        "prevention_steps": pr.prevention_steps if pr else None,
        "owner_review": pr.owner_review if pr else None,
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
        "affected_component": dev.affected_component if dev else None,
        "root_cause": root_cause,
        "fix_applied": fix_applied,
        "verification_result": verification_result,
        "technical_evidence": technical_evidence,
        "dev_notes": dev.dev_notes if dev else None,
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
        "quality_summary": quality_summary,
        "input_quality_score": input_quality_score,
        "audit_model": MODEL_NAME,
        "audit_prompt_version": PROMPT_VERSION,
        "audit_generated_at": audit_generated_at,
        "audit_status": "generated",
        "attachments": rca_input.attachments,
        # ── FIX: always inject safe defaults so StrictUndefined never fires
        #    on supervisor_plan, supervisor_review, or supervisor_feedback.
        #    These are overridden by extra_context in graph.py for agents
        #    that have the real supervisor plan available.
        "supervisor_plan": _SUPERVISOR_PLAN_DEFAULT,
        "supervisor_review": "",
        "supervisor_feedback": "",
        # Section outputs — default to empty string; overridden via extra_context
        # when upstream sections have already been generated.
        "section_issue_summary": "",
        "section_root_cause": "",
        "section_impact_analysis": "",
        "section_affected_module": "",
        "section_quality_gate_findings": "",
        "section_corrective_action": "",
        "section_preventive_action": "",
        "section_owner_review": "",
        # Final review inputs
        "markdown_rca": "",
    }

    return context


# Rendered Prompt Value Object
class RenderedPrompt:
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


# Prompt Renderer
def render_rca_prompt(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> RenderedPrompt:
    templates = get_prompt_templates()
    context = build_rca_template_context(
        rca_input,
        quality_summary=quality_summary,
    )
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
        context["issue_logs_id"],
        len(system_prompt),
        len(user_prompt),
    )

    return RenderedPrompt(
        system=system_prompt,
        user=user_prompt,
    )


def render_section_prompt(
    section_key: str,
    rca_input: RCAInputModel,
    quality_summary: str = "",
    extra_context: dict[str, Any] | None = None,
) -> RenderedPrompt:
    templates = get_section_prompt_templates(section_key)
    context = build_rca_template_context(
        rca_input,
        quality_summary=quality_summary,
    )

    # FIX: merge extra_context AFTER base context so caller-supplied values
    # (including the real supervisor_plan dict from graph.py) override the
    # safe defaults injected by build_rca_template_context(). This is the
    # correct merge order — extra_context wins over base defaults.
    if extra_context:
        context = {**context, **extra_context}

    env = _make_jinja_env()

    try:
        system_prompt = env.from_string(templates["system"]).render(**context)
        user_prompt = env.from_string(templates["user"]).render(**context)
    except UndefinedError as exc:
        raise PromptRenderError(
            f"Template variable missing while rendering section "
            f"'{section_key}': {exc}"
        ) from exc
    except Exception as exc:
        raise PromptRenderError(
            f"Unexpected error while rendering section "
            f"'{section_key}': {exc}"
        ) from exc

    logger.debug(
        "Section prompt rendered | section=%s | issue_id=%s "
        "| system_chars=%d | user_chars=%d",
        section_key,
        context["issue_logs_id"],
        len(system_prompt),
        len(user_prompt),
    )

    return RenderedPrompt(
        system=system_prompt,
        user=user_prompt,
    )


# LangChain ChatPromptTemplate Builders
def build_rca_chat_prompt(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> ChatPromptTemplate:
    rendered = render_rca_prompt(
        rca_input,
        quality_summary=quality_summary,
    )

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=rendered.system),
            HumanMessage(content=rendered.user),
        ]
    )

    return prompt


def build_section_chat_prompt(
    section_key: str,
    rca_input: RCAInputModel,
    quality_summary: str = "",
    extra_context: dict[str, Any] | None = None,
) -> ChatPromptTemplate:
    rendered = render_section_prompt(
        section_key=section_key,
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=extra_context,
    )

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=rendered.system),
            HumanMessage(content=rendered.user),
        ]
    )

    logger.debug(
        "Section ChatPromptTemplate built | section=%s | issue_id=%s",
        section_key,
        rca_input.task_monitoring_data.issue_logs_id,
    )

    return prompt