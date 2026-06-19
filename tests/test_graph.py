from __future__ import annotations

from common.markdown_converter import convert_rca_section_fields

import json
import logging
from functools import lru_cache
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
from typing_extensions import TypedDict

from agent_root.models import (
    QualityGateData,
    RCAGenerationConfig,
    RCAInputModel,
    RCAOutputModel,
    TaskMonitoringData,
)
from common.utils import build_section_chat_prompt
from config.providers import (
    FallbackProvider,
    extract_text,
)
from config.settings import settings

logger = logging.getLogger("rca_generator.graph")


class GraphBuildError(RuntimeError):
    """Raised when the LangGraph workflow cannot be compiled."""


class RCAGraphInput(TypedDict):
    rca_input: RCAInputModel


class RCAGraphState(RCAGraphInput, total=False):
    issue_id: str
    task_data: TaskMonitoringData
    has_pr_data: bool
    has_dev_data: bool
    has_qa_data: bool
    has_attachments: bool

    quality_summary: str

    # ── Supervisor outputs ────────────────────────────────────────────────────
    # Phase 2: supervisor_plan is now a parsed dict (Supervisor Decision Object),
    # not a raw string. This enables supervisor_route() to make dynamic routing
    # decisions based on route_decision, missing_data, and section sufficiency.
    supervisor_plan: dict[str, Any]          # parsed JSON — was: str
    supervisor_notes: str
    supervisor_status: str                   # "approved" | "degraded"

    # ── Per-section outputs ───────────────────────────────────────────────────
    section_issue_summary: str
    section_root_cause: str
    section_impact_analysis: str
    section_affected_module: str
    section_quality_gate_findings: str
    section_corrective_action: str
    section_preventive_action: str
    section_owner_review: str

    # ── Assembled full RCA ────────────────────────────────────────────────────
    markdown_rca: str
    generation_error: Optional[str]

    review_passed: bool
    review_notes: str

    # ── Final supervisor approval ─────────────────────────────────────────────
    final_review_decision: str        # "APPROVED" | "REGENERATE"
    final_review_notes: str
    final_review_checks: dict[str, bool]

    # ── Phase 2: routing outcome tracking ────────────────────────────────────
    route_outcome: str               # "continue" | "stop_missing_data" | "review_failure"
    inconsistent_sections: list[str] # cross-section consistency failures

    # ── Fallback tracking ─────────────────────────────────────────────────────
    fallback_used_sections: list[str]  # sections where fallback was triggered

    # ── Assembled section texts (FIX: must be declared so LangGraph accepts
    #    the key written by assemble_rca) ─────────────────────────────────────
    assembled_sections: dict[str, str]

    rca_output: Optional[RCAOutputModel]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _supervisor_plan_to_str(supervisor_plan: dict[str, Any] | str | None) -> str:
    """
    Safely serialises supervisor_plan for injection into Jinja2 templates.

    Accepts dict (Phase 2 standard), str (legacy / degraded fallback), or None.
    Returns a compact JSON string that section agents can read as context.
    """
    if supervisor_plan is None:
        return ""
    if isinstance(supervisor_plan, dict):
        return json.dumps(supervisor_plan, indent=2)
    # Already a string (degraded mode or legacy caller)
    return str(supervisor_plan)


# ─────────────────────────────────────────────────────────────────────────────
# FIX #2 — ERROR SANITIZATION
# Raw provider errors (429 responses, stack traces, API keys, quota messages)
# must never appear verbatim in the generated PDF. This function converts any
# exception into a short, user-facing message and logs the full detail
# internally for debugging.
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER_ERROR_PATTERNS: list[str] = [
    "RESOURCE_EXHAUSTED",
    "429",
    "quota",
    "rate limit",
    "UNAVAILABLE",
    "ProviderStatus",
    "Both providers failed",
    "Primary:",
    "Fallback:",
    "ai.google.dev",
    "openai.com",
    "APIError",
    "RateLimitError",
]

_SANITIZED_GENERATION_UNAVAILABLE = (
    "Generation unavailable due to temporary AI provider limitations. "
    "Please retry the request or contact support if this persists."
)

_SANITIZED_SECTION_ERROR = (
    "This section could not be generated due to a temporary service interruption. "
    "Please retry the RCA generation request."
)


def _sanitize_error_message(exc: Exception, section_key: str, issue_id: str) -> str:
    """
    Converts a provider or generation exception into a safe, user-facing
    placeholder string. Logs the full error internally at WARNING level.

    Rules:
    - NEVER expose raw API error bodies, quota messages, HTTP status codes,
      provider-internal stack traces, or billing URLs in the output.
    - Always log the full detail so engineers can diagnose without seeing it
      in the PDF.
    - Return a consistent, short, non-technical message suitable for display
      in the RCA document as a section placeholder.

    Args:
        exc:         The caught exception.
        section_key: The RCA section that failed (e.g. "corrective_action").
        issue_id:    The issue being processed.

    Returns:
        A sanitized string safe to store in section state and render in the PDF.
    """
    raw_error = str(exc)

    # Log full detail internally — never surfaces in the PDF
    logger.warning(
        "Section generation failed — sanitizing error for PDF output | "
        "section=%s | issue_id=%s | raw_error=%s",
        section_key,
        issue_id,
        raw_error,
    )

    # Check whether this looks like a provider-level error (quota, rate limit,
    # connectivity) versus an unexpected application-level error.
    is_provider_error = any(
        pattern.lower() in raw_error.lower()
        for pattern in _PROVIDER_ERROR_PATTERNS
    )

    if is_provider_error:
        return _SANITIZED_GENERATION_UNAVAILABLE

    # Generic application error — still sanitized, slightly more informative
    return _SANITIZED_SECTION_ERROR


def _invoke_section(
    section_key: str,
    rca_input: RCAInputModel,
    quality_summary: str,
    extra_context: dict[str, Any] | None = None,
    issue_id: str = "unknown",
) -> str:
    """
    Renders the section prompt via build_section_chat_prompt, invokes the
    configured LLM for that section using FallbackProvider for automatic
    primary → fallback failover, and returns the stripped text output.

    Fallback Architecture per section:
        Primary  → settings.<section>_provider / settings.<section>_model
                   (or per-request override from rca_input.generation_config)
        Fallback → settings.fallback_provider / settings.fallback_model
                   (defaults: gemini / gemini-2.5-flash)

    Failover triggers:
        - Quota exceeded / rate limit (429)
        - Timeout
        - API error / 500 server error
        - Service outage
        - Invalid / empty response

    Phase 2: supervisor_plan in extra_context is serialised to a compact JSON
    string before template rendering so Jinja2 receives a plain string value.

    Args:
        section_key:     Top-level YAML key for the section (e.g. "root_cause").
        rca_input:       Validated RCAInputModel instance.
        quality_summary: Pre-computed quality gate summary string.
        extra_context:   Optional upstream section outputs + supervisor_plan dict.
        issue_id:        Issue ID for structured fallback logging.

    Returns:
        Stripped Markdown string produced by the section LLM.
    """
    # Resolve provider and model via three-level priority chain:
    #   1. Per-request override from rca_input.generation_config  (highest)
    #   2. Per-section setting from config/settings.py            (middle)
    #   3. Global defaults                                        (lowest)
    gen_cfg: Optional[RCAGenerationConfig] = rca_input.generation_config

    primary_provider: str = (
        (gen_cfg.get_provider(section_key) if gen_cfg else None)
        or getattr(settings, f"{section_key}_provider", None)
        or settings.primary_provider
        or "openai"
    )
    primary_model: str = (
        (gen_cfg.get_model(section_key) if gen_cfg else None)
        or getattr(settings, f"{section_key}_model", None)
        or settings.primary_model
        or settings.openai_default_model
    )

    # Fallback always comes from global fallback settings
    fallback_provider: str = settings.fallback_provider or "gemini"
    fallback_model: str    = settings.fallback_model or settings.gemini_model

    # Phase 2: serialise supervisor_plan dict → str for Jinja2 template injection
    serialised_context: dict[str, Any] | None = None
    if extra_context:
        serialised_context = dict(extra_context)
        if "supervisor_plan" in serialised_context:
            serialised_context["supervisor_plan"] = _supervisor_plan_to_str(
                serialised_context["supervisor_plan"]
            )

    prompt = build_section_chat_prompt(
        section_key=section_key,
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=serialised_context,
    )

    # Use FallbackProvider for automatic primary → fallback failover.
    response = FallbackProvider.invoke(
        prompt=prompt.format_messages(),
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        issue_id=issue_id,
    )

    return extract_text(response).strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SUPERVISOR QUALITY CONTROL VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

_QC_MIN_LENGTH: int = 100

_QC_EVIDENCE_MARKERS: list[str] = [
    "component", "module", "function", "query", "table", "field", "api",
    "error", "log", "trace", "fix", "branch", "pr", "commit", "test",
    "validation", "qa", "gate", "reopen", "failed", "passed",
]

_QC_HALLUCINATION_MARKERS: list[str] = [
    "probably",
    "maybe",
    "might be caused",
    "could be due to",
    "likely due to",
    "may be",
    "could be",
    "might be",
    "possibly",
    "it seems",
    "it appears",
]

_DOMAIN_GROUPS: list[list[str]] = [
    ["database", "query", "sql", "table", "record", "db", "schema", "migration"],
    ["api", "endpoint", "request", "response", "http", "rest", "graphql", "webhook"],
    ["cache", "redis", "memcache", "ttl", "invalidat", "evict"],
    ["auth", "token", "session", "permission", "role", "oauth", "jwt", "credential"],
    ["queue", "message", "broker", "kafka", "rabbitmq", "worker", "job", "async"],
    ["frontend", "ui", "component", "render", "react", "vue", "dom", "css"],
    ["file", "storage", "upload", "download", "s3", "blob", "disk", "path"],
    ["network", "timeout", "connection", "socket", "dns", "ssl", "tls", "latency"],
]


def _extract_domain_groups(text: str) -> set[int]:
    """
    Returns the set of domain group indices whose keywords appear in text.
    """
    lowered = text.lower()
    found: set[int] = set()
    for idx, keywords in enumerate(_DOMAIN_GROUPS):
        if any(kw in lowered for kw in keywords):
            found.add(idx)
    return found


def supervisor_validate(
    section_output: str,
    min_length: int = _QC_MIN_LENGTH,
    cross_section_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Validates a single section agent output against quality control criteria.
    """
    reasons: list[str] = []
    checks:  dict[str, bool] = {}

    lowered = section_output.lower().strip()

    # Check 0: Detect sanitized error placeholder
    is_sanitized_error = (
        "generation unavailable due to temporary" in lowered
        or "could not be generated due to a temporary" in lowered
    )
    if is_sanitized_error:
        checks["not_an_error_placeholder"] = False
        reasons.append(
            "Section output is a sanitized error placeholder. "
            "Generation failed — retry required."
        )
        return {"passed": False, "reasons": reasons, "checks": checks}

    checks["not_an_error_placeholder"] = True

    # Check 1: Minimum length
    checks["minimum_length"] = len(lowered) >= min_length
    if not checks["minimum_length"]:
        reasons.append(
            f"Output too short ({len(lowered)} chars). "
            f"Minimum required: {min_length} chars."
        )

    # Check 2: Technical evidence present
    has_evidence = any(marker in lowered for marker in _QC_EVIDENCE_MARKERS)
    checks["technical_evidence_present"] = has_evidence
    if not has_evidence:
        reasons.append(
            "No technical evidence markers detected. "
            "Output must reference specific components, errors, logs, or QA data."
        )

    # Check 3: No hallucination / speculative language
    found_hallucinations = [
        phrase for phrase in _QC_HALLUCINATION_MARKERS if phrase in lowered
    ]
    checks["no_hallucinations"] = len(found_hallucinations) == 0
    if found_hallucinations:
        reasons.append(
            "Speculative language detected: " + ", ".join(found_hallucinations)
        )

    # Check 4: Cross-section consistency
    checks["cross_section_consistency"] = True
    if cross_section_context and len(lowered) > 150:
        current_domains = _extract_domain_groups(section_output)

        for upstream_key, upstream_text in cross_section_context.items():
            if not upstream_text or len(upstream_text.strip()) < 150:
                continue

            upstream_domains = _extract_domain_groups(upstream_text)

            if (
                current_domains
                and upstream_domains
                and current_domains.isdisjoint(upstream_domains)
            ):
                checks["cross_section_consistency"] = False
                reasons.append(
                    f"Cross-section inconsistency detected: current section references "
                    f"domain group(s) {sorted(current_domains)} but '{upstream_key}' "
                    f"references domain group(s) {sorted(upstream_domains)}. "
                    f"Sections must address the same root technical area."
                )
                logger.warning(
                    "supervisor_validate: cross-section inconsistency | "
                    "current_domains=%s | upstream_key=%s | upstream_domains=%s",
                    sorted(current_domains),
                    upstream_key,
                    sorted(upstream_domains),
                )
                break

    return {
        "passed":  len(reasons) == 0,
        "reasons": reasons,
        "checks":  checks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD NODES
# ─────────────────────────────────────────────────────────────────────────────

def collect_issue_data(state: RCAGraphState) -> dict[str, Any]:
    rca_input: RCAInputModel = state["rca_input"]
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info("Graph node: collect_issue_data | issue_id=%s", issue_id)

    return {
        "issue_id":             issue_id,
        "task_data":            rca_input.task_monitoring_data,
        "has_pr_data":          rca_input.github_pr is not None,
        "has_dev_data":         rca_input.developer_issue_data is not None,
        "has_qa_data":          rca_input.quality_gate_data is not None,
        "has_attachments":      bool(rca_input.attachments),
        "fallback_used_sections": [],
    }


def analyze_quality_gates(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: analyze_quality_gates | issue_id=%s", issue_id)

    rca_input: RCAInputModel = state["rca_input"]
    qg: Optional[QualityGateData] = rca_input.quality_gate_data

    if qg is None:
        return {"quality_summary": "No quality gate data provided."}

    failed_checks: list[str] = []
    passed_checks: list[str] = []

    if qg.quality_gate_first_pass is True:
        passed_checks.append("Quality Gate First Pass: PASSED")
    elif qg.quality_gate_first_pass is False:
        failed_checks.append("Quality Gate First Pass: FAILED")

    if qg.smoke_test_first_pass is True:
        passed_checks.append("Smoke Test First Pass: PASSED")
    elif qg.smoke_test_first_pass is False:
        failed_checks.append("Smoke Test First Pass: FAILED")

    if qg.qa_status.value.lower() == "passed":
        passed_checks.append("QA validation passed.")
    else:
        failed_checks.append(
            f"QA validation did not pass (status: {qg.qa_status.value})."
        )

    if qg.validation_status.value.lower() == "validated":
        passed_checks.append("Validation status: Validated.")
    else:
        failed_checks.append(
            f"Validation status: {qg.validation_status.value}."
        )

    if qg.reopen_count > 0:
        failed_checks.append(f"Reopen Count: {qg.reopen_count}")

    if qg.fc_failed_testing > 0:
        failed_checks.append(f"FC Failed Testing: {qg.fc_failed_testing}")

    parts: list[str] = []
    if passed_checks:
        parts.append("Passed: " + " | ".join(passed_checks))
    if failed_checks:
        parts.append("Failed: " + " | ".join(failed_checks))
    if qg.remarks:
        parts.append(f"Remarks: {qg.remarks}")
    parts.append(f"QA Status: {qg.qa_status.value}")
    parts.append(f"Validation Status: {qg.validation_status.value}")

    summary = (
        " || ".join(parts)
        if parts
        else "Quality gate data present but all fields are default."
    )

    return {"quality_summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR NODE
# ─────────────────────────────────────────────────────────────────────────────

_SUPERVISOR_PLAN_REQUIRED_KEYS: list[str] = [
    "issue_id",
    "overall_data_quality",
    "route_decision",
    "required_sections",
    "critical_areas",
    "missing_data",
    "execution_strategy",
    "section_plan",
]


def run_supervisor(state: RCAGraphState) -> dict[str, Any]:
    """
    Invokes the rca_supervisor prompt to produce the Mother Agent Supervisor
    Decision Object.

    Uses FallbackProvider: primary (rca_reviewer_provider) → fallback (gemini).
    """
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: run_supervisor | issue_id=%s", issue_id)

    rca_input: RCAInputModel = state["rca_input"]
    quality_summary: str = state.get("quality_summary", "")

    prompt = build_section_chat_prompt(
        section_key="rca_supervisor",
        rca_input=rca_input,
        quality_summary=quality_summary,
    )

    try:
        response = FallbackProvider.invoke(
            prompt=prompt.format_messages(),
            primary_provider=settings.rca_reviewer_provider,
            primary_model=settings.rca_reviewer_model,
            fallback_provider=settings.fallback_provider or "gemini",
            fallback_model=settings.fallback_model or settings.gemini_model,
            issue_id=issue_id,
        )
        raw_text = extract_text(response).strip()

        clean_text = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            plan: dict[str, Any] = json.loads(clean_text)
        except (json.JSONDecodeError, ValueError) as parse_err:
            logger.warning(
                "Supervisor plan JSON parse failed | issue_id=%s | error=%s | raw=%s",
                issue_id,
                parse_err,
                raw_text[:500],
            )
            return {
                "supervisor_plan":   {},
                "supervisor_notes":  f"Supervisor plan parse failed: {parse_err}",
                "supervisor_status": "degraded",
                "route_outcome":     "continue",
            }

        missing_keys = [k for k in _SUPERVISOR_PLAN_REQUIRED_KEYS if k not in plan]
        if missing_keys:
            logger.warning(
                "Supervisor plan missing required keys | issue_id=%s | missing=%s",
                issue_id,
                missing_keys,
            )
            plan["_missing_keys"] = missing_keys

        route_decision: str = plan.get("route_decision", "CONTINUE").upper()
        route_outcome: str  = _normalise_route_decision(route_decision)

        logger.info(
            "Supervisor plan parsed | issue_id=%s | route_decision=%s | "
            "data_quality=%s | missing_data=%s",
            issue_id,
            route_decision,
            plan.get("overall_data_quality", "UNKNOWN"),
            plan.get("missing_data", []),
        )

        return {
            "supervisor_plan":   plan,
            "supervisor_notes":  plan.get("supervisor_notes", ""),
            "supervisor_status": "approved",
            "route_outcome":     route_outcome,
        }

    except Exception as exc:
        logger.exception(
            "Supervisor node failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )
        return {
            "supervisor_plan":   {},
            "supervisor_notes":  f"Supervisor failed: {exc}",
            "supervisor_status": "degraded",
            "route_outcome":     "continue",
        }


def _normalise_route_decision(decision: str) -> str:
    """
    Maps the supervisor's route_decision string to a canonical route_outcome
    value used by supervisor_route() for conditional edge routing.
    """
    mapping = {
        "CONTINUE":             "continue",
        "STOP_MISSING_DATA":    "stop_missing_data",
        "STOP_PLACEHOLDER_DATA": "stop_placeholder_data",
        "REVIEW_FAILURE":       "review_failure",
    }
    return mapping.get(decision.upper(), "continue")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: SUPERVISOR ROUTE
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_route(state: RCAGraphState) -> str:
    """
    Dynamic routing function for the conditional edge after run_supervisor.
    """
    issue_id      = state.get("issue_id", "unknown")
    route_outcome = state.get("route_outcome", "continue")
    supervisor_plan: dict[str, Any] = state.get("supervisor_plan", {})

    if route_outcome == "stop_missing_data":
        missing_data = supervisor_plan.get("missing_data", [])
        logger.warning(
            "supervisor_route: STOP — critical data missing | issue_id=%s | missing=%s",
            issue_id,
            missing_data,
        )
        return END  # type: ignore[return-value]

    if route_outcome == "stop_placeholder_data":
        supervisor_notes = supervisor_plan.get("supervisor_notes", "")
        logger.warning(
            "supervisor_route: STOP — placeholder/test data detected | "
            "issue_id=%s | notes=%s",
            issue_id,
            supervisor_notes,
        )
        return END  # type: ignore[return-value]

    if route_outcome == "review_failure":
        supervisor_notes = supervisor_plan.get("supervisor_notes", "")
        logger.warning(
            "supervisor_route: STOP — review failure | issue_id=%s | notes=%s",
            issue_id,
            supervisor_notes,
        )
        return END  # type: ignore[return-value]

    logger.info(
        "supervisor_route: CONTINUE | issue_id=%s | data_quality=%s",
        issue_id,
        supervisor_plan.get("overall_data_quality", "UNKNOWN"),
    )
    return "generate_incident_analysis"


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SECTION AGENT NODES (4-STEP FLOW)
#
# The original 8 individual section nodes are replaced by 4 combined nodes.
# Each combined node generates 2 tightly related sections in a single call,
# passing the first section's output as context to the second.
#
# Node mapping:
#   generate_incident_analysis  → issue_summary  + root_cause
#   generate_technical_impact   → impact_analysis + affected_module
#   generate_qa_resolution      → quality_gate_findings + corrective_action
#   generate_prevention_review  → preventive_action + owner_review
#
# assemble_rca() is kept unchanged — it still reads all 8 individual state
# keys and produces the full 8-section markdown document.
# ─────────────────────────────────────────────────────────────────────────────

def generate_incident_analysis(state: RCAGraphState) -> dict[str, Any]:
    """
    Combined node: generates issue_summary then root_cause.

    issue_summary is generated first (no upstream dependencies).
    root_cause uses issue_summary as context, matching the original
    extra_context_keys=["section_issue_summary"] dependency.
    """
    issue_id: str   = state.get("issue_id", "unknown")  # type: ignore[assignment]
    rca_input       = state["rca_input"]
    quality_summary = state.get("quality_summary", "")

    logger.info("Graph node: generate_incident_analysis | issue_id=%s", issue_id)

    supervisor_plan: dict[str, Any] | str = state.get("supervisor_plan", {})
    base_context: dict[str, Any] = {}
    if supervisor_plan:
        base_context["supervisor_plan"] = supervisor_plan

    # ── Step 1: issue_summary ─────────────────────────────────────────────────
    issue_summary_text = _safe_invoke_with_retry(
        section_key="issue_summary",
        state_output_key="section_issue_summary",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=base_context or None,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
    )

    # ── Step 2: root_cause (uses issue_summary as context) ───────────────────
    root_cause_context = dict(base_context)
    root_cause_context["section_issue_summary"] = issue_summary_text

    root_cause_text = _safe_invoke_with_retry(
        section_key="root_cause",
        state_output_key="section_root_cause",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=root_cause_context,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
    )

    return {
        "section_issue_summary": issue_summary_text,
        "section_root_cause":    root_cause_text,
    }


def generate_technical_impact(state: RCAGraphState) -> dict[str, Any]:
    """
    Combined node: generates impact_analysis then affected_module.

    Both sections use section_issue_summary and section_root_cause as context,
    matching the original extra_context_keys dependencies.
    """
    issue_id: str   = state.get("issue_id", "unknown")  # type: ignore[assignment]
    rca_input       = state["rca_input"]
    quality_summary = state.get("quality_summary", "")

    logger.info("Graph node: generate_technical_impact | issue_id=%s", issue_id)

    supervisor_plan: dict[str, Any] | str = state.get("supervisor_plan", {})
    base_context: dict[str, Any] = {}
    if supervisor_plan:
        base_context["supervisor_plan"] = supervisor_plan

    # Pull upstream outputs from state
    issue_summary = state.get("section_issue_summary", "")
    root_cause    = state.get("section_root_cause", "")

    if issue_summary:
        base_context["section_issue_summary"] = issue_summary
    if root_cause:
        base_context["section_root_cause"] = root_cause

    # ── Step 1: impact_analysis ───────────────────────────────────────────────
    impact_analysis_text = _safe_invoke_with_retry(
        section_key="impact_analysis",
        state_output_key="section_impact_analysis",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=base_context or None,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
        cross_section_upstream={"section_root_cause": root_cause} if root_cause else None,
    )

    # ── Step 2: affected_module (uses root_cause as context) ─────────────────
    affected_module_context = dict(base_context)
    affected_module_context["section_impact_analysis"] = impact_analysis_text

    affected_module_text = _safe_invoke_with_retry(
        section_key="affected_module",
        state_output_key="section_affected_module",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=affected_module_context,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
        cross_section_upstream={"section_root_cause": root_cause} if root_cause else None,
    )

    return {
        "section_impact_analysis": impact_analysis_text,
        "section_affected_module": affected_module_text,
    }


def generate_qa_resolution(state: RCAGraphState) -> dict[str, Any]:
    """
    Combined node: generates quality_gate_findings then corrective_action.

    quality_gate_findings has no upstream section dependencies (same as original).
    corrective_action uses section_root_cause and section_impact_analysis,
    matching the original extra_context_keys dependencies.
    """
    issue_id: str   = state.get("issue_id", "unknown")  # type: ignore[assignment]
    rca_input       = state["rca_input"]
    quality_summary = state.get("quality_summary", "")

    logger.info("Graph node: generate_qa_resolution | issue_id=%s", issue_id)

    supervisor_plan: dict[str, Any] | str = state.get("supervisor_plan", {})
    base_context: dict[str, Any] = {}
    if supervisor_plan:
        base_context["supervisor_plan"] = supervisor_plan

    # ── Step 1: quality_gate_findings (no upstream section deps) ─────────────
    qg_findings_text = _safe_invoke_with_retry(
        section_key="quality_gate_findings",
        state_output_key="section_quality_gate_findings",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=base_context or None,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
    )

    # ── Step 2: corrective_action (uses root_cause + impact_analysis) ────────
    corrective_context = dict(base_context)
    root_cause       = state.get("section_root_cause", "")
    impact_analysis  = state.get("section_impact_analysis", "")
    if root_cause:
        corrective_context["section_root_cause"] = root_cause
    if impact_analysis:
        corrective_context["section_impact_analysis"] = impact_analysis

    corrective_action_text = _safe_invoke_with_retry(
        section_key="corrective_action",
        state_output_key="section_corrective_action",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=corrective_context,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
        cross_section_upstream={"section_root_cause": root_cause} if root_cause else None,
    )

    return {
        "section_quality_gate_findings": qg_findings_text,
        "section_corrective_action":     corrective_action_text,
    }


def generate_prevention_review(state: RCAGraphState) -> dict[str, Any]:
    """
    Combined node: generates preventive_action then owner_review.

    preventive_action uses section_corrective_action as context.
    owner_review uses section_corrective_action + section_preventive_action,
    matching the original extra_context_keys dependencies.
    """
    issue_id: str   = state.get("issue_id", "unknown")  # type: ignore[assignment]
    rca_input       = state["rca_input"]
    quality_summary = state.get("quality_summary", "")

    logger.info("Graph node: generate_prevention_review | issue_id=%s", issue_id)

    supervisor_plan: dict[str, Any] | str = state.get("supervisor_plan", {})
    base_context: dict[str, Any] = {}
    if supervisor_plan:
        base_context["supervisor_plan"] = supervisor_plan

    corrective_action = state.get("section_corrective_action", "")
    root_cause        = state.get("section_root_cause", "")

    # ── Step 1: preventive_action (uses corrective_action as context) ────────
    preventive_context = dict(base_context)
    if corrective_action:
        preventive_context["section_corrective_action"] = corrective_action

    preventive_action_text = _safe_invoke_with_retry(
        section_key="preventive_action",
        state_output_key="section_preventive_action",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=preventive_context,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
        cross_section_upstream={
            "section_root_cause":        root_cause,
            "section_corrective_action": corrective_action,
        } if (root_cause or corrective_action) else None,
    )

    # ── Step 2: owner_review (uses corrective_action + preventive_action) ────
    owner_review_context = dict(preventive_context)
    owner_review_context["section_preventive_action"] = preventive_action_text

    owner_review_text = _safe_invoke_with_retry(
        section_key="owner_review",
        state_output_key="section_owner_review",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=owner_review_context,
        issue_id=issue_id,
        supervisor_plan=supervisor_plan,
        cross_section_upstream={
            "section_corrective_action": corrective_action,
            "section_preventive_action": preventive_action_text,
        } if corrective_action else None,
    )

    return {
        "section_preventive_action": preventive_action_text,
        "section_owner_review":      owner_review_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER: _invoke_section_with_retry
#
# Extracted from the (now removed) legacy single-section node to be reusable
# inside the combined node functions above. Handles QC validation and one
# retry on failure, mirroring the logic that previously applied to individual
# sections.
# ─────────────────────────────────────────────────────────────────────────────

def _invoke_section_with_retry(
    section_key: str,
    state_output_key: str,
    rca_input: RCAInputModel,
    quality_summary: str,
    extra_context: dict[str, Any] | None,
    issue_id: str,
    supervisor_plan: dict[str, Any] | str,
    cross_section_upstream: dict[str, str] | None = None,
) -> str:
    """
    Invokes a single section LLM call with QC validation and one auto-retry.

    This is the same logic previously embedded inside the legacy section node
    factory, factored out so combined nodes can call it for each of their two
    sub-sections.

    Args:
        section_key:            YAML prompt key (e.g. "root_cause").
        state_output_key:       State field name, used only for logging.
        rca_input:              Validated RCAInputModel.
        quality_summary:        Pre-computed quality gate summary.
        extra_context:          Context dict to pass to the prompt template.
        issue_id:                Issue ID for logging.
        supervisor_plan:        Supervisor plan dict or str (for threshold lookup).
        cross_section_upstream: Optional upstream sections for consistency check.

    Returns:
        Generated section text (sanitized on failure).
    """
    # Resolve per-section quality threshold from supervisor_plan
    quality_threshold = _QC_MIN_LENGTH
    plan_dict: dict[str, Any] = supervisor_plan if isinstance(supervisor_plan, dict) else {}  # type: ignore[assignment]
    thresholds: dict[str, Any] = plan_dict.get("section_quality_thresholds", {})
    if section_key in thresholds:
        quality_threshold = max(_QC_MIN_LENGTH, thresholds[section_key] * 3)

    # Make extra_context mutable for retry injection
    mutable_context: dict[str, Any] = dict(extra_context) if extra_context else {}

    try:
        text = _invoke_section(
            section_key=section_key,
            rca_input=rca_input,
            quality_summary=quality_summary,
            extra_context=mutable_context or None,
            issue_id=issue_id,
        )

        validation_result = supervisor_validate(
            text,
            min_length=quality_threshold,
            cross_section_context=cross_section_upstream,
        )

        if not validation_result["passed"]:
            logger.warning(
                "Section QC failed on first attempt | section=%s | issue_id=%s | reasons=%s",
                section_key,
                issue_id,
                validation_result["reasons"],
            )
            mutable_context["supervisor_validation_feedback"] = (
                "Previous output failed quality check. Reasons: "
                + "; ".join(validation_result["reasons"])
                + ". Please produce a more complete, evidence-grounded response."
            )
            text = _invoke_section(
                section_key=section_key,
                rca_input=rca_input,
                quality_summary=quality_summary,
                extra_context=mutable_context or None,
                issue_id=issue_id,
            )
            logger.info(
                "Section regenerated after QC failure | section=%s | issue_id=%s",
                section_key,
                issue_id,
            )

        return text

    except Exception as exc:
        logger.exception(
            "Section invocation failed | section=%s | issue_id=%s",
            section_key,
            issue_id,
        )
        return _sanitize_error_message(exc, section_key, issue_id)


def _safe_invoke_with_retry(
    section_key: str,
    state_output_key: str,
    rca_input: RCAInputModel,
    quality_summary: str,
    extra_context: dict[str, Any] | None,
    issue_id: str,
    supervisor_plan: dict[str, Any] | str,
    cross_section_upstream: dict[str, str] | None = None,
) -> str:
    """
    Defense-in-depth wrapper around _invoke_section_with_retry.

    _invoke_section_with_retry already sanitizes any exception raised by the
    underlying LLM call (_invoke_section) internally, so under normal
    operation it never raises. This wrapper exists so a combined node can
    never lose BOTH of its sections if _invoke_section_with_retry itself
    raises for any other reason — e.g. a bug in code that runs before its
    own try block, or it being replaced wholesale (as in tests that
    monkeypatch it directly with a side_effect). Each of the two sections
    inside a combined node is invoked through this wrapper independently,
    so a failure on one section never prevents the other from being
    attempted and reported.

    Args mirror _invoke_section_with_retry exactly; see that function's
    docstring for details.

    Returns:
        Generated section text, or a sanitized placeholder if
        _invoke_section_with_retry (or anything it calls) raises.
    """
    try:
        return _invoke_section_with_retry(
            section_key=section_key,
            state_output_key=state_output_key,
            rca_input=rca_input,
            quality_summary=quality_summary,
            extra_context=extra_context,
            issue_id=issue_id,
            supervisor_plan=supervisor_plan,
            cross_section_upstream=cross_section_upstream,
        )
    except Exception as exc:
        logger.exception(
            "_invoke_section_with_retry raised unexpectedly | section=%s | issue_id=%s",
            section_key,
            issue_id,
        )
        return _sanitize_error_message(exc, section_key, issue_id)


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLY NODE
#
# FIX #3 — affected_module population:
# assemble_rca reads all 8 individual state keys written by the 4 combined
# nodes. The assembled_sections dict exposes section_affected_module so the
# PDF assembler can access it via section["affected_module"].
# ─────────────────────────────────────────────────────────────────────────────

def assemble_rca(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: assemble_rca | issue_id=%s", issue_id)

    usage = FallbackProvider.get_usage_stats()
    logger.info(
        "FallbackProvider usage stats at assembly | issue_id=%s | "
        "primary_used=%d | fallback_used=%d | total=%d",
        issue_id,
        usage["primary_usage_count"],
        usage["fallback_usage_count"],
        usage["total"],
    )

    def _get(key: str) -> str:
        return state.get(key, "Not available.")  # type: ignore[call-overload]

    section_texts: dict[str, str] = {
        "issue_summary":         _get("section_issue_summary"),
        "root_cause":            _get("section_root_cause"),
        "impact_analysis":       _get("section_impact_analysis"),
        "affected_module":       _get("section_affected_module"),
        "quality_gate_findings": _get("section_quality_gate_findings"),
        "corrective_action":     _get("section_corrective_action"),
        "preventive_action":     _get("section_preventive_action"),
        "owner_review":          _get("section_owner_review"),
    }

    markdown_rca = "\n\n".join([
        f"## 1. Issue Summary\n\n{section_texts['issue_summary']}",
        f"## 2. Root Cause\n\n{section_texts['root_cause']}",
        f"## 3. Impact Analysis\n\n{section_texts['impact_analysis']}",
        f"## 4. Affected Module\n\n{section_texts['affected_module']}",
        f"## 5. Quality Gate Findings\n\n{section_texts['quality_gate_findings']}",
        f"## 6. Corrective Action\n\n{section_texts['corrective_action']}",
        f"## 7. Preventive Action\n\n{section_texts['preventive_action']}",
        f"## 8. Owner Review\n\n{section_texts['owner_review']}",
    ])

    logger.info(
        "RCA assembled | issue_id=%s | total_chars=%d | "
        "affected_module_chars=%d",
        issue_id,
        len(markdown_rca),
        len(section_texts["affected_module"]),
    )

    return {
        "markdown_rca":       markdown_rca,
        "assembled_sections": section_texts,
        "generation_error":   None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_SECTIONS: list[str] = [
    "## 1. Issue Summary",
    "## 2. Root Cause",
    "## 3. Impact Analysis",
    "## 4. Affected Module",
    "## 5. Quality Gate Findings",
    "## 6. Corrective Action",
    "## 7. Preventive Action",
    "## 8. Owner Review",
]

_FORBIDDEN_PHRASES: list[str] = [
    "probably",
    "maybe",
    "might be caused",
    "could be due to",
    "likely due to",
    "may be",
    "could be",
    "might be",
    "possibly",
    "it seems",
    "it appears",
]

_WEAK_PLACEHOLDERS: list[str] = [
    "not available",
    "not specified",
    "n/a",
]

_MIN_SECTION_LENGTH:          int = 50
_MAX_WEAK_PLACEHOLDER_COUNT:  int = 14


def _count_all_phrase_occurrences(text: str, phrases: list[str]) -> int:
    lowered = text.lower()
    total   = 0
    for phrase in phrases:
        start = 0
        while True:
            idx = lowered.find(phrase, start)
            if idx == -1:
                break
            total += 1
            start  = idx + len(phrase)
    return total


def _validate_rca_quality(markdown_rca: str) -> list[str]:
    warnings: list[str] = []

    placeholder_count = _count_all_phrase_occurrences(markdown_rca, _WEAK_PLACEHOLDERS)
    if placeholder_count >= _MAX_WEAK_PLACEHOLDER_COUNT:
        warnings.append(
            "RCA contains too many unavailable or unspecified fields. "
            "Input data may be incomplete."
        )

    root_cause_heading = "## 2. Root Cause"
    corrective_heading = "## 6. Corrective Action"

    if root_cause_heading in markdown_rca and corrective_heading in markdown_rca:
        root_cause_section = markdown_rca.split(root_cause_heading, 1)[1].split(
            "## 3. Impact Analysis", 1,
        )[0]
        if len(root_cause_section.strip()) < _MIN_SECTION_LENGTH:
            warnings.append(
                "Root Cause section is too short. "
                "It must explain what failed, where it failed, and why it failed."
            )

    if corrective_heading in markdown_rca:
        corrective_section = markdown_rca.split(corrective_heading, 1)[1].split(
            "## 7. Preventive Action", 1,
        )[0]
        if len(corrective_section.strip()) < _MIN_SECTION_LENGTH:
            warnings.append(
                "Corrective Action section is too short. "
                "It must explain the fix or required corrective action."
            )

    affected_module_heading = "## 4. Affected Module"
    if affected_module_heading in markdown_rca:
        affected_module_section = markdown_rca.split(affected_module_heading, 1)[1].split(
            "## 5. Quality Gate Findings", 1,
        )[0]
        if len(affected_module_section.strip()) < _MIN_SECTION_LENGTH:
            warnings.append(
                "Affected Module section is too short or empty. "
                "It must identify the specific component, service, or code path affected."
            )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# REVIEW NODE
# ─────────────────────────────────────────────────────────────────────────────

def review_rca(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: review_rca | issue_id=%s", issue_id)

    generation_error = state.get("generation_error")
    markdown_rca     = state.get("markdown_rca", "")

    if generation_error:
        return {
            "review_passed": False,
            "review_notes":  f"Generation failed: {generation_error}",
            "rca_output":    None,
        }

    if not markdown_rca.strip():
        return {
            "review_passed": False,
            "review_notes":  "RCA output is empty.",
            "rca_output":    None,
        }

    review_notes: list[str] = []

    missing_sections = [
        section for section in _REQUIRED_SECTIONS if section not in markdown_rca
    ]
    if missing_sections:
        review_notes.append("Missing required sections: " + ", ".join(missing_sections))
        logger.warning(
            "RCA review - missing sections | issue_id=%s | missing=%s",
            issue_id,
            missing_sections,
        )

    vague_phrases_found = [
        phrase for phrase in _FORBIDDEN_PHRASES if phrase.lower() in markdown_rca.lower()
    ]
    if vague_phrases_found:
        review_notes.append(
            "RCA contains vague or speculative language: "
            + ", ".join(vague_phrases_found)
        )
        logger.warning(
            "RCA review - speculative language detected | issue_id=%s | phrases=%s",
            issue_id,
            vague_phrases_found,
        )

    quality_warnings = _validate_rca_quality(markdown_rca)
    if quality_warnings:
        review_notes.extend(quality_warnings)
        logger.warning(
            "RCA review - quality safeguards triggered | issue_id=%s | notes=%s",
            issue_id,
            quality_warnings,
        )

    supervisor_status = state.get("supervisor_status", "")
    if supervisor_status == "degraded":
        review_notes.append(
            "Supervisor node ran in degraded mode — execution plan was not generated. "
            "Section consistency may be reduced."
        )
        logger.warning(
            "RCA review - supervisor degraded | issue_id=%s",
            issue_id,
        )

    usage = FallbackProvider.get_usage_stats()
    if usage["fallback_usage_count"] > 0:
        logger.warning(
            "RCA review - fallback was triggered %d time(s) during generation "
            "| issue_id=%s | primary_used=%d | fallback_used=%d",
            usage["fallback_usage_count"],
            issue_id,
            usage["primary_usage_count"],
            usage["fallback_usage_count"],
        )

    review_passed = not review_notes

    if review_passed:
        logger.info("RCA review passed | issue_id=%s", issue_id)
        return {
            "review_passed": True,
            "review_notes":  "All required sections present and quality checks passed.",
            "rca_output": RCAOutputModel(
                issue_id=issue_id,
                markdown_rca=markdown_rca.strip(),
                pdf_file_path=f"outputs/{issue_id}_rca.pdf",
            ),
        }

    logger.warning(
        "RCA review failed | issue_id=%s | notes=%s",
        issue_id,
        review_notes,
    )

    return {
        "review_passed": False,
        "review_notes":  " | ".join(review_notes),
        "rca_output":    None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR FINAL REVIEW NODE
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_final_review(state: RCAGraphState) -> dict[str, Any]:
    """
    Final holistic supervisor approval before PDF generation.
    Uses FallbackProvider: primary (rca_reviewer_provider) → fallback (gemini).
    """
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: supervisor_final_review | issue_id=%s", issue_id)

    rca_input: RCAInputModel = state["rca_input"]
    quality_summary: str     = state.get("quality_summary", "")
    markdown_rca: str        = state.get("markdown_rca", "")
    supervisor_plan_dict: dict[str, Any] = state.get("supervisor_plan", {})

    supervisor_plan_str = _supervisor_plan_to_str(supervisor_plan_dict)

    extra_context: dict[str, Any] = {
        "markdown_rca":    markdown_rca,
        "supervisor_plan": supervisor_plan_str,
    }

    prompt = build_section_chat_prompt(
        section_key="rca_supervisor_final_review",
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=extra_context,
    )

    try:
        response = FallbackProvider.invoke(
            prompt=prompt.format_messages(),
            primary_provider=settings.rca_reviewer_provider,
            primary_model=settings.rca_reviewer_model,
            fallback_provider=settings.fallback_provider or "gemini",
            fallback_model=settings.fallback_model or settings.gemini_model,
            issue_id=issue_id,
        )
        raw_text = extract_text(response).strip()

        try:
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            review_data: dict[str, Any] = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as parse_err:
            logger.warning(
                "Supervisor final review JSON parse failed | issue_id=%s | error=%s",
                issue_id,
                parse_err,
            )
            return {
                "final_review_decision":  "APPROVED",
                "final_review_notes":     (
                    f"Supervisor final review response could not be parsed: {parse_err}. "
                    "Proceeding with conditional approval."
                ),
                "final_review_checks":    {},
                "inconsistent_sections":  [],
            }

        decision: str              = review_data.get("decision", "APPROVED")
        failed_checks: list[str]   = review_data.get("failed_checks", [])
        final_notes: str           = review_data.get("supervisor_final_notes", "")
        checks: dict[str, bool]    = review_data.get("checks", {})
        inconsistent: list[str]    = review_data.get("inconsistent_sections", [])

        logger.info(
            "Supervisor final review complete | issue_id=%s | decision=%s | "
            "failed_checks=%s | inconsistent_sections=%s",
            issue_id,
            decision,
            failed_checks,
            inconsistent,
        )

        if decision == "REGENERATE":
            logger.warning(
                "Supervisor final review: REGENERATE | issue_id=%s | failed=%s",
                issue_id,
                failed_checks,
            )

        return {
            "final_review_decision":  decision,
            "final_review_notes":     final_notes,
            "final_review_checks":    checks,
            "inconsistent_sections":  inconsistent,
        }

    except Exception as exc:
        logger.exception(
            "Supervisor final review node failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )
        return {
            "final_review_decision":  "APPROVED",
            "final_review_notes":     (
                f"Supervisor final review failed with error: {exc}. "
                "Proceeding with conditional approval."
            ),
            "final_review_checks":    {},
            "inconsistent_sections":  [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH COMPILATION
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rca_graph() -> Any:
    logger.info("Compiling RCA LangGraph workflow (4-step combined nodes + Fallback Architecture)")

    try:
        builder: Any = StateGraph(RCAGraphState)

        # ── Register nodes ────────────────────────────────────────────────────
        builder.add_node("collect_issue_data",          collect_issue_data)
        builder.add_node("analyze_quality_gates",       analyze_quality_gates)
        builder.add_node("run_supervisor",              run_supervisor)
        builder.add_node("generate_incident_analysis",  generate_incident_analysis)
        builder.add_node("generate_technical_impact",   generate_technical_impact)
        builder.add_node("generate_qa_resolution",      generate_qa_resolution)
        builder.add_node("generate_prevention_review",  generate_prevention_review)
        builder.add_node("assemble_rca",                assemble_rca)
        builder.add_node("review_rca",                  review_rca)
        builder.add_node("supervisor_final_review",     supervisor_final_review)

        # ── Edges ─────────────────────────────────────────────────────────────
        builder.add_edge(START,                         "collect_issue_data")
        builder.add_edge("collect_issue_data",          "analyze_quality_gates")
        builder.add_edge("analyze_quality_gates",       "run_supervisor")

        # Conditional routing edge after supervisor
        builder.add_conditional_edges(
            "run_supervisor",
            supervisor_route,
            {
                "generate_incident_analysis": "generate_incident_analysis",
                END: END,
            },
        )

        # 4-step sequential section pipeline
        builder.add_edge("generate_incident_analysis",  "generate_technical_impact")
        builder.add_edge("generate_technical_impact",   "generate_qa_resolution")
        builder.add_edge("generate_qa_resolution",      "generate_prevention_review")
        builder.add_edge("generate_prevention_review",  "assemble_rca")

        builder.add_edge("assemble_rca",                "review_rca")
        builder.add_edge("review_rca",                  "supervisor_final_review")
        builder.add_edge("supervisor_final_review",     END)

        graph = builder.compile()

        logger.info(
            "RCA LangGraph workflow (4-step combined nodes + Fallback Architecture) compiled successfully"
        )
        return graph

    except Exception as exc:
        logger.exception("Failed to compile RCA LangGraph | error=%s", exc)
        raise GraphBuildError(
            f"RCA LangGraph compilation failed: {exc}"
        ) from exc


def reset_rca_graph() -> None:
    get_rca_graph.cache_clear()
    logger.warning("RCA graph cache cleared — test use only.")


# ─────────────────────────────────────────────────────────────────────────────
# PDF ASSEMBLER HELPER
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf_section_dict(
    final_state: RCAGraphState,
    rca_input: RCAInputModel,
) -> dict[str, Any]:
    """
    Builds the section dict consumed by the Jinja2 HTML template (rca.html).

    FIX #3: this is the authoritative place where section_affected_module is
    mapped to section["affected_module"] so the template's
    {{ section.affected_module | safe }} renders correctly.

    PHASE 1: the six markdown-bearing fields are converted to HTML via
    convert_rca_section_fields() immediately before this function returns.

    All section texts are sourced from assembled_sections (written by
    assemble_rca) with a fallback to individual state keys.

    Args:
        final_state: The completed RCAGraphState after graph.invoke().
        rca_input:   The original RCAInputModel for metadata fields.

    Returns:
        A dict with all keys expected by rca.html's {% for section in rca_sections %},
        with markdown-bearing fields already converted to HTML.
    """
    assembled: dict[str, str] = final_state.get("assembled_sections", {})

    def _section(key: str, fallback_state_key: str) -> str:
        text = assembled.get(key) or final_state.get(fallback_state_key, "")  # type: ignore[call-overload]
        return text.strip() if text else "Not available."

    task      = rca_input.task_monitoring_data
    qa        = rca_input.quality_gate_data
    approval  = rca_input.approval_data

    def _bool_to_pass_fail(value: Optional[bool]) -> str:
        if value is True:
            return "Passed"
        if value is False:
            return "Failed"
        return "Not recorded"

    section_dict: dict[str, Any] = {
        # Metadata / heading fields
        "issue_number":      1,
        "issue_title":       task.title,
        "issue_description": task.issue_description,
        "product":           task.product,
        "module":            task.module,
        "core_function":     task.core_function,

        # Section content (raw markdown; converted to HTML below)
        "cause":             _section("root_cause",            "section_root_cause"),
        "affected_module":   _section("affected_module",       "section_affected_module"),
        "impact_analysis":   _section("impact_analysis",       "section_impact_analysis"),
        "solution":          _section("corrective_action",     "section_corrective_action"),
        "preventive_action": _section("preventive_action",     "section_preventive_action"),
        "owner_review":      _section("owner_review",          "section_owner_review"),

        # Quality gate structured fields
        "quality_gate_pass": _bool_to_pass_fail(qa.quality_gate_first_pass if qa else None),
        "smoke_test_pass":   _bool_to_pass_fail(qa.smoke_test_first_pass   if qa else None),
        "reopen_count":      qa.reopen_count      if qa else 0,
        "fc_failed":         qa.fc_failed_testing if qa else 0,
        "qa_status":         qa.qa_status.value   if qa else "Not recorded",
        "pic_qa":            qa.pic_qa            if qa else "Not recorded",

        # Approval sign-off
        "approval": {
            "prepared_by":       approval.prepared_by       if approval else None,
            "reviewed_by_dev":   approval.reviewed_by_dev   if approval else None,
            "validated_by_qa":   approval.validated_by_qa   if approval else None,
            "approved_by_owner": approval.approved_by_owner if approval else None,
        } if approval else None,

        # Attachments
        "attachments": [a.file_path for a in rca_input.attachments],
    }

    # Convert markdown-bearing fields to HTML before returning
    return convert_rca_section_fields(section_dict)