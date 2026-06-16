from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
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

    failed_checks: list[str] = []
    passed_checks: list[str] = []

    # Quality Gate First Pass
    if qg.quality_gate_first_pass is True:
        passed_checks.append("Quality Gate First Pass: PASSED")
    elif qg.quality_gate_first_pass is False:
        failed_checks.append("Quality Gate First Pass: FAILED")

    # Smoke Test First Pass
    if qg.smoke_test_first_pass is True:
        passed_checks.append("Smoke Test First Pass: PASSED")
    elif qg.smoke_test_first_pass is False:
        failed_checks.append("Smoke Test First Pass: FAILED")

    # QA Status
    if qg.qa_status is not None:
        if qg.qa_status.value.lower() == "passed":
            passed_checks.append("QA validation passed.")
        else:
            failed_checks.append(
                f"QA validation did not pass (status: {qg.qa_status.value})."
            )

    # Validation Status
    if qg.validation_status is not None:
        if qg.validation_status.value.lower() == "validated":
            passed_checks.append("Validation status: Validated.")
        else:
            failed_checks.append(
                f"Validation status: {qg.validation_status.value}."
            )

    # Reopen Count
    if qg.reopen_count > 0:
        failed_checks.append(f"Reopen Count: {qg.reopen_count}")

    # FC Failed Testing
    if qg.fc_failed_testing > 0:
        failed_checks.append(f"FC Failed Testing: {qg.fc_failed_testing}")

    # Build structured summary string for prompt rendering
    parts: list[str] = []

    if passed_checks:
        parts.append("Passed: " + " | ".join(passed_checks))

    if failed_checks:
        parts.append("Failed: " + " | ".join(failed_checks))

    if qg.qa_status:
        parts.append(f"QA Status: {qg.qa_status.value}")

    if qg.validation_status:
        parts.append(f"Validation Status: {qg.validation_status.value}")

    if qg.remarks:
        parts.append(f"Remarks: {qg.remarks}")

    summary = (
        " || ".join(parts)
        if parts
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


# -- Section & phrase validators ----------------------------------------------

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
    "further investigation",
    "it seems",
    "it appears",
    "cannot be precisely determined",
    "not enough information",
]

_WEAK_PLACEHOLDERS: list[str] = [
    "not available",
    "not specified",
    "n/a",
]

_MIN_SECTION_LENGTH: int = 50

# The prompt template renders "Not available." as a default for every optional
# field. A complete, valid RCA generated from partial input will naturally
# contain many such occurrences (error_message, recommended_solution, pr_url,
# branch_name, dev_resolved_on, dev_end_date, etc.).  The original threshold
# of 6 was too low and caused false-positive review failures on otherwise
# correct RCA documents.  We raise the threshold to 14 to only flag cases
# where even the mandatory core fields are missing or unpopulated.
_MAX_WEAK_PLACEHOLDER_COUNT: int = 14


def _count_all_phrase_occurrences(text: str, phrases: list[str]) -> int:
    """
    Count the total number of times any phrase from the list appears in text.
    Unlike _count_phrase_matches (which counts distinct phrases), this counts
    every individual occurrence so repeated "Not available." entries are each
    tallied separately.
    """
    lowered = text.lower()
    total = 0
    for phrase in phrases:
        start = 0
        while True:
            idx = lowered.find(phrase, start)
            if idx == -1:
                break
            total += 1
            start = idx + len(phrase)
    return total


def _validate_rca_quality(markdown_rca: str) -> list[str]:
    """
    Returns quality warning messages.
    Empty list means the RCA passed quality checks.
    """
    warnings: list[str] = []

    # Count every individual occurrence of weak placeholder strings.
    # The prompt template alone can produce up to ~12 "Not available." entries
    # for optional fields (error_message, recommended_solution, pr_number,
    # pr_url, branch_name, dev_resolved_on, dev_end_date, qa_validated_on,
    # etc.).  We only flag when the count significantly exceeds that baseline,
    # indicating that even mandatory content fields are missing.
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
            "## 3. Impact Analysis",
            1,
        )[0]

        if len(root_cause_section.strip()) < _MIN_SECTION_LENGTH:
            warnings.append(
                "Root Cause section is too short. "
                "It must explain what failed, where it failed, and why it failed."
            )

    if corrective_heading in markdown_rca:
        corrective_section = markdown_rca.split(corrective_heading, 1)[1].split(
            "## 7. Preventive Action",
            1,
        )[0]

        if len(corrective_section.strip()) < _MIN_SECTION_LENGTH:
            warnings.append(
                "Corrective Action section is too short. "
                "It must explain the fix or required corrective action."
            )

    return warnings


# -- Review node --------------------------------------------------------------

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

        issue_id_str = state.get("issue_id", "unknown")
        pdf_path = f"outputs/{issue_id_str}_rca.pdf"

        return {
            "review_passed": True,
            "review_notes": "All required sections present and quality checks passed.",
            "rca_output": RCAOutputModel(
                issue_id=issue_id_str,
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
        "review_notes": " | ".join(review_notes),
        "rca_output": None,
    }


# -- Graph compilation --------------------------------------------------------

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