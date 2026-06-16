from __future__ import annotations

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
from config.providers import extract_text, get_llm_for_agent
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
    rca_output: Optional[RCAOutputModel]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _invoke_section(
    section_key: str,
    rca_input: RCAInputModel,
    quality_summary: str,
    extra_context: dict[str, Any] | None = None,
) -> str:
    """
    Renders the section prompt via build_section_chat_prompt, invokes the
    configured LLM for that section, and returns the stripped text output.

    Args:
        section_key:     Top-level YAML key for the section (e.g. "root_cause").
        rca_input:       Validated RCAInputModel instance.
        quality_summary: Pre-computed quality gate summary string.
        extra_context:   Optional upstream section outputs to inject as
                         additional Jinja2 template variables.

    Returns:
        Stripped Markdown string produced by the section LLM.

    Raises:
        RuntimeError: Propagated from the LLM invocation on hard failures.
    """
    # Resolve provider and model using a three-level priority chain:
    #   1. Per-request override from rca_input.generation_config  (highest)
    #   2. Per-section setting from config/settings.py            (middle)
    #   3. Global Gemini singleton defaults                        (lowest)
    gen_cfg: Optional[RCAGenerationConfig] = rca_input.generation_config

    provider: str = (
        (gen_cfg.get_provider(section_key) if gen_cfg else None)
        or getattr(settings, f"{section_key}_provider", None)
        or "gemini"
    )
    model: str = (
        (gen_cfg.get_model(section_key) if gen_cfg else None)
        or getattr(settings, f"{section_key}_model", None)
        or settings.gemini_model
    )

    llm = get_llm_for_agent(provider, model)

    prompt = build_section_chat_prompt(
        section_key=section_key,
        rca_input=rca_input,
        quality_summary=quality_summary,
        extra_context=extra_context,
    )

    response = (prompt | llm).invoke({})
    return extract_text(response).strip()


def _section_node(
    section_key: str,
    state_output_key: str,
    extra_context_keys: list[str] | None = None,
) -> Any:
    """
    Factory that returns a LangGraph node function for a named RCA section.

    Args:
        section_key:        YAML key used to load prompts (e.g. "root_cause").
        state_output_key:   State field to write the result into
                            (e.g. "section_root_cause").
        extra_context_keys: List of state keys whose values should be passed
                            as extra_context to the section prompt renderer.
                            Useful for chaining upstream outputs into the
                            current section's Jinja2 template.

    Returns:
        A callable compatible with StateGraph.add_node().
    """
    def node(state: RCAGraphState) -> dict[str, Any]:
        issue_id        = state.get("issue_id", "unknown")
        rca_input       = state["rca_input"]
        quality_summary = state.get("quality_summary", "")

        logger.info(
            "Graph node: %s | issue_id=%s",
            state_output_key,
            issue_id,
        )

        extra_context: dict[str, Any] = {}
        if extra_context_keys:
            for key in extra_context_keys:
                value = state.get(key)  # type: ignore[call-overload]
                if value:
                    extra_context[key] = value

        try:
            text = _invoke_section(
                section_key=section_key,
                rca_input=rca_input,
                quality_summary=quality_summary,
                extra_context=extra_context or None,
            )
            return {state_output_key: text}

        except Exception as exc:
            logger.exception(
                "Section node failed | section=%s | issue_id=%s | error=%s",
                section_key,
                issue_id,
                exc,
            )
            # Return a fallback so downstream nodes can still run; the review
            # node will catch the missing / placeholder content.
            return {state_output_key: f"Not available. (generation error: {exc})"}

    node.__name__ = state_output_key
    return node


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD NODES
# ─────────────────────────────────────────────────────────────────────────────

def collect_issue_data(state: RCAGraphState) -> dict[str, Any]:
    rca_input: RCAInputModel = state["rca_input"]
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info("Graph node: collect_issue_data | issue_id=%s", issue_id)

    return {
        "issue_id":       issue_id,
        "task_data":      rca_input.task_monitoring_data,
        "has_pr_data":    rca_input.github_pr is not None,
        "has_dev_data":   rca_input.developer_issue_data is not None,
        "has_qa_data":    rca_input.quality_gate_data is not None,
        "has_attachments": bool(rca_input.attachments),
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
# SECTION AGENT NODES
#
# Each node maps to one ## section in the final RCA document.
# Nodes that depend on upstream section text pass those state keys via
# extra_context_keys so the Jinja2 template can reference them.
#
# Execution order (enforced by graph edges below):
#   1. issue_summary          — standalone; no upstream deps
#   2. root_cause             — receives issue_summary
#   3. impact_analysis        — receives issue_summary + root_cause
#   4. affected_module        — receives root_cause
#   5. quality_gate_findings  — standalone (uses quality_summary from state)
#   6. corrective_action      — receives root_cause + impact_analysis
#   7. preventive_action      — receives corrective_action
#   8. owner_review           — receives corrective_action + preventive_action
# ─────────────────────────────────────────────────────────────────────────────

generate_issue_summary = _section_node(
    section_key="issue_summary",
    state_output_key="section_issue_summary",
)

generate_root_cause = _section_node(
    section_key="root_cause",
    state_output_key="section_root_cause",
    extra_context_keys=["section_issue_summary"],
)

generate_impact_analysis = _section_node(
    section_key="impact_analysis",
    state_output_key="section_impact_analysis",
    extra_context_keys=["section_issue_summary", "section_root_cause"],
)

generate_affected_module = _section_node(
    section_key="affected_module",
    state_output_key="section_affected_module",
    extra_context_keys=["section_root_cause"],
)

generate_quality_gate_findings = _section_node(
    section_key="quality_gate_findings",
    state_output_key="section_quality_gate_findings",
)

generate_corrective_action = _section_node(
    section_key="corrective_action",
    state_output_key="section_corrective_action",
    extra_context_keys=["section_root_cause", "section_impact_analysis"],
)

generate_preventive_action = _section_node(
    section_key="preventive_action",
    state_output_key="section_preventive_action",
    extra_context_keys=["section_corrective_action"],
)

generate_owner_review = _section_node(
    section_key="owner_review",
    state_output_key="section_owner_review",
    extra_context_keys=["section_corrective_action", "section_preventive_action"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLY NODE
# Combines all section outputs into a single Markdown RCA document.
# ─────────────────────────────────────────────────────────────────────────────

def assemble_rca(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: assemble_rca | issue_id=%s", issue_id)

    def _get(key: str) -> str:
        return state.get(key, "Not available.")  # type: ignore[call-overload]

    markdown_rca = "\n\n".join([
        f"## 1. Issue Summary\n\n{_get('section_issue_summary')}",
        f"## 2. Root Cause\n\n{_get('section_root_cause')}",
        f"## 3. Impact Analysis\n\n{_get('section_impact_analysis')}",
        f"## 4. Affected Module\n\n{_get('section_affected_module')}",
        f"## 5. Quality Gate Findings\n\n{_get('section_quality_gate_findings')}",
        f"## 6. Corrective Action\n\n{_get('section_corrective_action')}",
        f"## 7. Preventive Action\n\n{_get('section_preventive_action')}",
        f"## 8. Owner Review\n\n{_get('section_owner_review')}",
    ])

    logger.info(
        "RCA assembled | issue_id=%s | total_chars=%d",
        issue_id,
        len(markdown_rca),
    )

    return {
        "markdown_rca":      markdown_rca,
        "generation_error":  None,
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

_MIN_SECTION_LENGTH: int = 50
_MAX_WEAK_PLACEHOLDER_COUNT: int = 14


def _count_all_phrase_occurrences(text: str, phrases: list[str]) -> int:
    """
    Count the total number of times any phrase from the list appears in text.
    """
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
    """Returns quality warning messages. Empty list means the RCA passed."""
    warnings: list[str] = []

    placeholder_count = _count_all_phrase_occurrences(markdown_rca, _WEAK_PLACEHOLDERS)

    if placeholder_count >= _MAX_WEAK_PLACEHOLDER_COUNT:
        warnings.append(
            "RCA contains too many unavailable or unspecified fields. "
            "Input data may be incomplete."
        )

    root_cause_heading  = "## 2. Root Cause"
    corrective_heading  = "## 6. Corrective Action"

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

    # 1. Required section check
    missing_sections = [
        section
        for section in _REQUIRED_SECTIONS
        if section not in markdown_rca
    ]

    if missing_sections:
        review_notes.append(
            "Missing required sections: " + ", ".join(missing_sections)
        )
        logger.warning(
            "RCA review - missing sections | issue_id=%s | missing=%s",
            issue_id,
            missing_sections,
        )

    # 2. Forbidden / speculative language check
    vague_phrases_found = [
        phrase
        for phrase in _FORBIDDEN_PHRASES
        if phrase.lower() in markdown_rca.lower()
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

    # 3. Quality depth checks
    quality_warnings = _validate_rca_quality(markdown_rca)

    if quality_warnings:
        review_notes.extend(quality_warnings)
        logger.warning(
            "RCA review - quality safeguards triggered | issue_id=%s | notes=%s",
            issue_id,
            quality_warnings,
        )

    review_passed = not review_notes

    if review_passed:
        logger.info("RCA review passed | issue_id=%s", issue_id)

        pdf_path = f"outputs/{issue_id}_rca.pdf"

        return {
            "review_passed": True,
            "review_notes":  "All required sections present and quality checks passed.",
            "rca_output": RCAOutputModel(
                issue_id=issue_id,
                markdown_rca=markdown_rca.strip(),
                pdf_file_path=pdf_path,
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
# GRAPH COMPILATION
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rca_graph() -> Any:
    logger.info("Compiling RCA LangGraph workflow")

    try:
        builder: Any = StateGraph(RCAGraphState)

        # ── Register nodes ────────────────────────────────────────────────────
        builder.add_node("collect_issue_data",          collect_issue_data)
        builder.add_node("analyze_quality_gates",       analyze_quality_gates)
        builder.add_node("generate_issue_summary",      generate_issue_summary)
        builder.add_node("generate_root_cause",         generate_root_cause)
        builder.add_node("generate_impact_analysis",    generate_impact_analysis)
        builder.add_node("generate_affected_module",    generate_affected_module)
        builder.add_node("generate_quality_gate_findings", generate_quality_gate_findings)
        builder.add_node("generate_corrective_action",  generate_corrective_action)
        builder.add_node("generate_preventive_action",  generate_preventive_action)
        builder.add_node("generate_owner_review",       generate_owner_review)
        builder.add_node("assemble_rca",                assemble_rca)
        builder.add_node("review_rca",                  review_rca)

        # ── Edges — sequential pipeline ───────────────────────────────────────
        builder.add_edge(START,                          "collect_issue_data")
        builder.add_edge("collect_issue_data",           "analyze_quality_gates")
        builder.add_edge("analyze_quality_gates",        "generate_issue_summary")
        builder.add_edge("generate_issue_summary",       "generate_root_cause")
        builder.add_edge("generate_root_cause",          "generate_impact_analysis")
        builder.add_edge("generate_impact_analysis",     "generate_affected_module")
        builder.add_edge("generate_affected_module",     "generate_quality_gate_findings")
        builder.add_edge("generate_quality_gate_findings", "generate_corrective_action")
        builder.add_edge("generate_corrective_action",   "generate_preventive_action")
        builder.add_edge("generate_preventive_action",   "generate_owner_review")
        builder.add_edge("generate_owner_review",        "assemble_rca")
        builder.add_edge("assemble_rca",                 "review_rca")
        builder.add_edge("review_rca",                   END)

        graph = builder.compile()

        logger.info("RCA LangGraph workflow compiled successfully")
        return graph

    except Exception as exc:
        logger.exception("Failed to compile RCA LangGraph | error=%s", exc)
        raise GraphBuildError(
            f"RCA LangGraph compilation failed: {exc}"
        ) from exc


def reset_rca_graph() -> None:
    get_rca_graph.cache_clear()
    logger.warning("RCA graph cache cleared — test use only.")