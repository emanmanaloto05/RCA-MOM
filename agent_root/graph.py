# agent_root/graph.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph  #   type: ignore[import-untyped]
from typing_extensions import TypedDict

from agent_root.chains import ChainBuildError, ChainInput, get_rca_chain
from agent_root.models import (
    QualityGateData,
    RCAInputModel,
    RCAOutputModel,
    TaskMonitoringData,
)

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

    markdown_rca: str
    generation_error: Optional[str]

    review_passed: bool
    review_notes: str
    rca_output: Optional[RCAOutputModel]


def collect_issue_data(state: RCAGraphState) -> dict[str, Any]:
    rca_input: RCAInputModel = state["rca_input"]
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info("Graph node: collect_issue_data | issue_id=%s", issue_id)

    has_attachments = bool(rca_input.attachments)

    return {
        "issue_id": issue_id,
        "task_data": rca_input.task_monitoring_data,
        "has_pr_data": rca_input.github_pr is not None,
        "has_dev_data": rca_input.developer_issue_data is not None,
        "has_qa_data": rca_input.quality_gate_data is not None,
        "has_attachments": has_attachments,
    }


def analyze_quality_gates(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: analyze_quality_gates | issue_id=%s", issue_id)

    rca_input: RCAInputModel = state["rca_input"]
    qg: Optional[QualityGateData] = rca_input.quality_gate_data

    if qg is None:
        return {"quality_summary": "No quality gate data provided."}

    signals: list[str] = []

    if qg.quality_gate_first_pass is not None:
        signals.append(
            f"Quality Gate First Pass: {'PASSED' if qg.quality_gate_first_pass else 'FAILED'}"
        )

    if qg.smoke_test_first_pass is not None:
        signals.append(
            f"Smoke Test First Pass: {'PASSED' if qg.smoke_test_first_pass else 'FAILED'}"
        )

    if qg.reopen_count > 0:
        signals.append(f"Reopen Count: {qg.reopen_count}")

    if qg.fc_failed_testing > 0:
        signals.append(f"FC Failed Testing: {qg.fc_failed_testing}")

    if qg.qa_status:
        signals.append(f"QA Status: {qg.qa_status.value}")

    if qg.validation_status:
        signals.append(f"Validation Status: {qg.validation_status.value}")

    summary = (
        " | ".join(signals)
        if signals
        else "Quality gate data present but all fields are default."
    )

    return {"quality_summary": summary}


def generate_rca_node(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: generate_rca | issue_id=%s", issue_id)

    rca_input: RCAInputModel = state["rca_input"]
    quality_summary: str = state.get("quality_summary", "")

    try:
        chain = get_rca_chain()

        chain_input: ChainInput = {
            "rca_input": rca_input,
            "quality_summary": quality_summary,
        }

        markdown_rca: str = chain.invoke(chain_input)

        if not markdown_rca or not markdown_rca.strip():
            return {
                "markdown_rca": "",
                "generation_error": "Chain returned an empty response.",
            }

        return {
            "markdown_rca": markdown_rca.strip(),
            "generation_error": None,
        }

    except ChainBuildError as exc:
        logger.exception(
            "Chain build failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )
        return {
            "markdown_rca": "",
            "generation_error": f"Chain build error: {exc}",
        }

    except Exception as exc:
        logger.exception(
            "Unexpected error in generate_rca node | issue_id=%s | error=%s",
            issue_id,
            exc,
        )
        return {
            "markdown_rca": "",
            "generation_error": f"Unexpected generation error: {exc}",
        }


_REQUIRED_HEADINGS: list[str] = [
    "## 1. Issue Summary",
    "## 2. Root Cause",
    "## 3. Impact Analysis",
    "## 4. Affected Module",
    "## 5. Quality Gate Findings",
    "## 6. Corrective Action",
    "## 7. Preventive Action",
    "## 8. Owner Review",
]


_GENERIC_PHRASES: list[str] = [
    "may be",
    "could be",
    "might be",
    "likely",
    "possibly",
    "probably",
    "it seems",
    "it appears",
    "further investigation",
    "cannot be precisely determined",
    "not enough information",
]


_WEAK_PLACEHOLDERS: list[str] = [
    "not available",
    "not specified",
    "n/a",
]


GENERIC_PHRASES: list[str] = [
    "may be",
    "could be",
    "might be",
    "possibly",
    "probably",
    "further investigation",
]


def _count_phrase_matches(text: str, phrases: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for phrase in phrases if phrase in lowered)


def _validate_rca_quality(markdown_rca: str) -> list[str]:
    """
    Returns quality warning messages.
    Empty list means the RCA passed quality checks.
    """
    warnings: list[str] = []

    generic_count = _count_phrase_matches(markdown_rca, _GENERIC_PHRASES)
    placeholder_count = _count_phrase_matches(markdown_rca, _WEAK_PLACEHOLDERS)

    if generic_count >= 3:
        warnings.append(
            "RCA contains excessive speculative language. "
            "The root cause may be too generic and requires human review."
        )

    if placeholder_count >= 6:
        warnings.append(
            "RCA contains too many unavailable or unspecified fields. "
            "Input data may be incomplete."
        )

    root_cause_heading = "## 2. Root Cause"
    corrective_heading = "## 6. Corrective Action"

    if root_cause_heading in markdown_rca and corrective_heading in markdown_rca:
        root_cause_section = markdown_rca.split(root_cause_heading, 1)[1].split(
            "## 3. Impact Analysis",
            1,
        )[0]

        if len(root_cause_section.strip()) < 80:
            warnings.append(
                "Root Cause section is too short. "
                "It must explain what failed, where it failed, and why it failed."
            )

    if corrective_heading in markdown_rca:
        corrective_section = markdown_rca.split(corrective_heading, 1)[1].split(
            "## 7. Preventive Action",
            1,
        )[0]

        if len(corrective_section.strip()) < 80:
            warnings.append(
                "Corrective Action section is too short. "
                "It must explain the fix or required corrective action."
            )

    return warnings


def review_rca(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: review_rca | issue_id=%s", issue_id)

    generation_error = state.get("generation_error")
    markdown_rca = state.get("markdown_rca", "")

    if generation_error:
        return {
            "review_passed": False,
            "review_notes": f"Generation failed: {generation_error}",
            "rca_output": None,
        }

    if not markdown_rca.strip():
        return {
            "review_passed": False,
            "review_notes": "RCA output is empty.",
            "rca_output": None,
        }

    missing = [heading for heading in _REQUIRED_HEADINGS if heading not in markdown_rca]

    if missing:
        notes = f"Missing sections: {', '.join(missing)}"

        logger.warning(
            "RCA review failed — missing sections | issue_id=%s | missing=%s",
            issue_id,
            missing,
        )

        return {
            "review_passed": False,
            "review_notes": notes,
            "rca_output": None,
        }

    phrase_count = sum(
        1
        for phrase in GENERIC_PHRASES
        if phrase in markdown_rca.lower()
    )

    if phrase_count >= 2:
        logger.warning(
            "RCA review failed — speculative language threshold reached | "
            "issue_id=%s | phrase_count=%s",
            issue_id,
            phrase_count,
        )

        return {
            "review_passed": False,
            "review_notes": (
                "RCA contains speculative language "
                "and requires manual review."
            ),
            "rca_output": None,
        }

    quality_warnings = _validate_rca_quality(markdown_rca)

    if quality_warnings:
        notes = " | ".join(quality_warnings)

        logger.warning(
            "RCA review failed — quality safeguards triggered | issue_id=%s | notes=%s",
            issue_id,
            notes,
        )

        return {
            "review_passed": False,
            "review_notes": notes,
            "rca_output": None,
        }

    logger.info("RCA review passed | issue_id=%s", issue_id)

    return {
        "review_passed": True,
        "review_notes": "All required sections present and quality checks passed.",
        "rca_output": RCAOutputModel(
            issue_id=issue_id,
            markdown_rca=markdown_rca.strip(),
            pdf_file_path=None,
        ),
    }


@lru_cache(maxsize=1)
def get_rca_graph() -> Any:
    logger.info("Compiling RCA LangGraph workflow")

    try:
        builder: Any = StateGraph(RCAGraphState)

        builder.add_node("collect_issue_data", collect_issue_data)
        builder.add_node("analyze_quality_gates", analyze_quality_gates)
        builder.add_node("generate_rca", generate_rca_node)
        builder.add_node("review_rca", review_rca)

        builder.add_edge(START, "collect_issue_data")
        builder.add_edge("collect_issue_data", "analyze_quality_gates")
        builder.add_edge("analyze_quality_gates", "generate_rca")
        builder.add_edge("generate_rca", "review_rca")
        builder.add_edge("review_rca", END)

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