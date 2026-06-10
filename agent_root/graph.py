# graph.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
from typing_extensions import TypedDict

from agent_root.chains import ChainBuildError, get_rca_chain
from agent_root.models import (
    QualityGateData,
    RCAInputModel,
    RCAOutputModel,
    TaskMonitoringData,
)

logger = logging.getLogger("rca_generator.graph")


# Custom exception
class GraphBuildError(RuntimeError):
    """Raised when the LangGraph workflow cannot be compiled."""


# State definition
#
# Split into two TypedDicts:
# - RCAGraphInput:   required keys (always present in initial state)
# - RCAGraphState:   full state with node-populated optional keys
#
# This prevents Pylance flagging state["rca_input"] as potentially missing.


class RCAGraphInput(TypedDict):
    """Required keys — must be present in the initial state."""
    rca_input: RCAInputModel


class RCAGraphState(RCAGraphInput, total=False):
    """Full graph state — required input plus all node-populated keys."""

    # Populated by collect_issue_data
    issue_id: str
    task_data: TaskMonitoringData
    has_pr_data: bool
    has_dev_data: bool
    has_qa_data: bool

    # Populated by analyze_quality_gates
    quality_summary: str

    # Populated by generate_rca_node
    markdown_rca: str
    generation_error: Optional[str]

    # Populated by review_rca
    review_passed: bool
    review_notes: str

    # Final output
    rca_output: Optional[RCAOutputModel]



# Node: collect_issue_data
def collect_issue_data(state: RCAGraphState) -> dict[str, Any]:
    rca_input: RCAInputModel = state["rca_input"]
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info("Graph node: collect_issue_data | issue_id=%s", issue_id)

    return {
        "issue_id": issue_id,
        "task_data": rca_input.task_monitoring_data,
        "has_pr_data": rca_input.github_pr is not None,
        "has_dev_data": rca_input.developer_issue_data is not None,
        "has_qa_data": rca_input.quality_gate_data is not None,
    }



# Node: analyze_quality_gates
def analyze_quality_gates(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: analyze_quality_gates | issue_id=%s", issue_id)

    rca_input: RCAInputModel = state["rca_input"]
    qg: Optional[QualityGateData] = rca_input.quality_gate_data

    if qg is None:
        logger.debug("Quality gate data absent | issue_id=%s", issue_id)
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

    logger.debug(
        "Quality gate analysis complete | issue_id=%s | summary=%s",
        issue_id,
        summary,
    )
    return {"quality_summary": summary}



# Node: generate_rca_node
def generate_rca_node(state: RCAGraphState) -> dict[str, Any]:
    issue_id = state.get("issue_id", "unknown")
    logger.info("Graph node: generate_rca | issue_id=%s", issue_id)

    rca_input: RCAInputModel = state["rca_input"]

    try:
        chain = get_rca_chain()
        markdown_rca: str = chain.invoke(rca_input)

        if not markdown_rca or not markdown_rca.strip():
            return {"markdown_rca": "", "generation_error": "Chain returned an empty response."}

        logger.info(
            "RCA Markdown generated | issue_id=%s | chars=%d",
            issue_id,
            len(markdown_rca),
        )
        return {"markdown_rca": markdown_rca.strip(), "generation_error": None}

    except ChainBuildError as exc:
        logger.exception("Chain build failed | issue_id=%s | error=%s", issue_id, exc)
        return {"markdown_rca": "", "generation_error": f"Chain build error: {exc}"}

    except Exception as exc:
        logger.exception(
            "Unexpected error in generate_rca node | issue_id=%s | error=%s", issue_id, exc
        )
        return {"markdown_rca": "", "generation_error": f"Unexpected generation error: {exc}"}



# Node: review_rca
_REQUIRED_HEADINGS = [
    "## 1. Issue Summary",
    "## 2. Root Cause",
    "## 3. Impact Analysis",
    "## 4. Affected Module",
    "## 5. Quality Gate Findings",
    "## 6. Corrective Action",
    "## 7. Preventive Action",
    "## 8. Owner Review",
]


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

    missing = [h for h in _REQUIRED_HEADINGS if h not in markdown_rca]

    if missing:
        notes = f"Missing sections: {', '.join(missing)}"
        logger.warning(
            "RCA review failed — missing sections | issue_id=%s | missing=%s",
            issue_id,
            missing,
        )
        return {"review_passed": False, "review_notes": notes, "rca_output": None}

    logger.info("RCA review passed | issue_id=%s", issue_id)

    return {
        "review_passed": True,
        "review_notes": "All required sections present.",
        "rca_output": RCAOutputModel(
            issue_id=issue_id,
            markdown_rca=markdown_rca,
            pdf_file_path=None,
        ),
    }



# Graph factory
@lru_cache(maxsize=1)
def get_rca_graph() -> Any:
    """
    Compiles and caches the RCA LangGraph workflow.

    Returns Any because LangGraph does not ship Pylance-compatible type
    stubs — annotating the return as StateGraph or CompiledStateGraph
    causes reportMissingTypeStubs / reportUnknownVariableType cascades.
    The runtime type is CompiledStateGraph and .invoke() works correctly.

    Raises GraphBuildError if compilation fails.
    """
    logger.info("Compiling RCA LangGraph workflow")

    try:
        # builder typed as Any — LangGraph has no Pylance-compatible stubs,
        # so StateGraph members (add_node, add_edge, compile) resolve as
        # Unknown. Annotating as Any silences the cascade without changing
        # runtime behaviour.
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
        raise GraphBuildError(f"RCA LangGraph compilation failed: {exc}") from exc


def reset_rca_graph() -> None:
    """
    Clears the lru_cache so a fresh graph is compiled on the next call.
    Use in tests only — never call in production request handlers.
    """
    get_rca_graph.cache_clear()
    logger.warning("RCA graph cache cleared — test use only.")