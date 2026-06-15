# tests/test_graph.py
"""
Tests for agent_root/graph.py

Covers: collect_issue_data, analyze_quality_gates, generate_rca_node,
        review_rca, get_rca_graph (compilation & caching), reset_rca_graph.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_root.graph import (
    GraphBuildError,
    RCAGraphState,
    analyze_quality_gates,
    collect_issue_data,
    generate_rca_node,
    get_rca_graph,
    reset_rca_graph,
    review_rca,
)
from agent_root.models import (
    QualityGateData,
    RCAInputModel,
    RCAOutputModel,
    Status,
)
from tests.conftest import VALID_MARKDOWN


# =============================================================================
# Helpers
# =============================================================================

def _state(rca_input: RCAInputModel, **extra: Any) -> RCAGraphState:
    """Build a minimal RCAGraphState for node testing."""
    base: RCAGraphState = {"rca_input": rca_input}  # type: ignore[typeddict-item]
    base.update(extra)  # type: ignore[typeddict-unknown-key]
    return base


# =============================================================================
# Node 1 — collect_issue_data
# =============================================================================

class TestCollectIssueData:
    def test_extracts_issue_id(self, rca_input: RCAInputModel) -> None:
        result = collect_issue_data(_state(rca_input))
        assert result["issue_id"] == "EIL_TEST001"

    def test_extracts_task_data(self, rca_input: RCAInputModel) -> None:
        result = collect_issue_data(_state(rca_input))
        assert result["task_data"] is rca_input.task_monitoring_data

    def test_has_pr_data_true(self, rca_input: RCAInputModel) -> None:
        result = collect_issue_data(_state(rca_input))
        assert result["has_pr_data"] is True

    def test_has_pr_data_false_when_absent(self, rca_input_minimal: RCAInputModel) -> None:
        result = collect_issue_data(_state(rca_input_minimal))
        assert result["has_pr_data"] is False

    def test_has_dev_data_true(self, rca_input: RCAInputModel) -> None:
        result = collect_issue_data(_state(rca_input))
        assert result["has_dev_data"] is True

    def test_has_qa_data_true(self, rca_input: RCAInputModel) -> None:
        result = collect_issue_data(_state(rca_input))
        assert result["has_qa_data"] is True

    def test_has_all_optional_false_for_minimal(self, rca_input_minimal: RCAInputModel) -> None:
        result = collect_issue_data(_state(rca_input_minimal))
        assert result["has_pr_data"] is False
        assert result["has_dev_data"] is False
        assert result["has_qa_data"] is False


# =============================================================================
# Node 2 — analyze_quality_gates
# =============================================================================

class TestAnalyzeQualityGates:
    def test_returns_no_data_string_when_qa_absent(
        self, rca_input_minimal: RCAInputModel
    ) -> None:
        result = analyze_quality_gates(_state(rca_input_minimal, issue_id="EIL_TEST001"))
        assert result["quality_summary"] == "No quality gate data provided."

    def test_passes_first_pass_signal(self, rca_input: RCAInputModel) -> None:
        result = analyze_quality_gates(_state(rca_input, issue_id="EIL_TEST001"))
        assert "Quality Gate First Pass: PASSED" in result["quality_summary"]
        assert "Smoke Test First Pass: PASSED" in result["quality_summary"]

    def test_failed_signals(self, rca_input_minimal: RCAInputModel, task_data: Any) -> None:
        qg = QualityGateData(
            quality_gate_first_pass=False,
            smoke_test_first_pass=False,
            reopen_count=3,
            fc_failed_testing=2,
            qa_status=Status.OPEN,
        )
        inp = RCAInputModel(
            task_monitoring_data=task_data,
            quality_gate_data=qg,
        )
        result = analyze_quality_gates(_state(inp, issue_id="EIL_X"))
        summary = result["quality_summary"]
        assert "FAILED" in summary
        assert "Reopen Count: 3" in summary
        assert "FC Failed Testing: 2" in summary

    def test_default_qa_data_returns_default_message(self, task_data: Any) -> None:
        inp = RCAInputModel(
            task_monitoring_data=task_data,
            quality_gate_data=QualityGateData(),
        )
        result = analyze_quality_gates(_state(inp, issue_id="EIL_X"))
        assert "all fields are default" in result["quality_summary"]

    def test_uses_unknown_issue_id_when_missing_from_state(
        self, rca_input_minimal: RCAInputModel
    ) -> None:
        # Should not raise even without issue_id in state
        result = analyze_quality_gates(_state(rca_input_minimal))
        assert "quality_summary" in result


# =============================================================================
# Node 3 — generate_rca_node
# =============================================================================

class TestGenerateRcaNode:
    def test_successful_generation(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = VALID_MARKDOWN

        with patch("agent_root.graph.get_rca_chain", return_value=mock_chain):
            result = generate_rca_node(_state(rca_input, issue_id="EIL_TEST001"))

        assert result["markdown_rca"] == VALID_MARKDOWN.strip()
        assert result["generation_error"] is None

    def test_chain_returns_empty_string(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "   "

        with patch("agent_root.graph.get_rca_chain", return_value=mock_chain):
            result = generate_rca_node(_state(rca_input, issue_id="EIL_TEST001"))

        assert result["markdown_rca"] == ""
        assert result["generation_error"] == "Chain returned an empty response."

    def test_chain_build_error_captured(self, rca_input: RCAInputModel) -> None:
        from agent_root.chains import ChainBuildError

        with patch(
            "agent_root.graph.get_rca_chain",
            side_effect=ChainBuildError("Build failed"),
        ):
            result = generate_rca_node(_state(rca_input, issue_id="EIL_TEST001"))

        assert result["markdown_rca"] == ""
        assert "Chain build error" in result["generation_error"]

    def test_unexpected_exception_captured(self, rca_input: RCAInputModel) -> None:
        with patch(
            "agent_root.graph.get_rca_chain",
            side_effect=RuntimeError("Unexpected"),
        ):
            result = generate_rca_node(_state(rca_input, issue_id="EIL_TEST001"))

        assert "Unexpected generation error" in result["generation_error"]


# =============================================================================
# Node 4 — review_rca
# =============================================================================

class TestReviewRca:
    def test_passes_with_all_sections(self, rca_input: RCAInputModel) -> None:
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            markdown_rca=VALID_MARKDOWN,
            generation_error=None,
        )
        result = review_rca(state)
        assert result["review_passed"] is True
        assert isinstance(result["rca_output"], RCAOutputModel)
        assert result["rca_output"].issue_id == "EIL_TEST001"

    def test_fails_when_generation_error_set(self, rca_input: RCAInputModel) -> None:
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            markdown_rca="",
            generation_error="Chain crashed",
        )
        result = review_rca(state)
        assert result["review_passed"] is False
        assert "Chain crashed" in result["review_notes"]
        assert result["rca_output"] is None

    def test_fails_on_empty_markdown(self, rca_input: RCAInputModel) -> None:
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            markdown_rca="   ",
            generation_error=None,
        )
        result = review_rca(state)
        assert result["review_passed"] is False
        assert "empty" in result["review_notes"].lower()

    def test_fails_on_missing_section(self, rca_input: RCAInputModel) -> None:
        # Remove one required section heading
        broken = VALID_MARKDOWN.replace("## 3. Impact Analysis\n", "")
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            markdown_rca=broken,
            generation_error=None,
        )
        result = review_rca(state)
        assert result["review_passed"] is False
        assert "## 3. Impact Analysis" in result["review_notes"]

    def test_rca_output_contains_correct_markdown(self, rca_input: RCAInputModel) -> None:
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            markdown_rca=VALID_MARKDOWN,
            generation_error=None,
        )
        result = review_rca(state)
        assert result["rca_output"].markdown_rca.strip() == VALID_MARKDOWN.strip()

    def test_review_notes_all_sections_present(self, rca_input: RCAInputModel) -> None:
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            markdown_rca=VALID_MARKDOWN,
            generation_error=None,
        )
        result = review_rca(state)
        assert "All required sections present" in result["review_notes"]


# =============================================================================
# get_rca_graph
# =============================================================================

class TestGetRcaGraph:
    def test_graph_compiles_successfully(self) -> None:
        reset_rca_graph()
        graph = get_rca_graph()
        assert graph is not None

    def test_graph_is_cached(self) -> None:
        reset_rca_graph()
        g1 = get_rca_graph()
        g2 = get_rca_graph()
        assert g1 is g2

    def test_reset_clears_cache(self) -> None:
        reset_rca_graph()
        g1 = get_rca_graph()
        reset_rca_graph()
        g2 = get_rca_graph()
        assert g1 is not g2

    def test_graph_build_error_on_failure(self) -> None:
        reset_rca_graph()
        with patch("agent_root.graph.StateGraph", side_effect=RuntimeError("boom")):
            with pytest.raises(GraphBuildError, match="compilation failed"):
                get_rca_graph()
        reset_rca_graph()  # restore for subsequent tests

    def test_compiled_graph_has_invoke(self) -> None:
        reset_rca_graph()
        graph = get_rca_graph()
        assert callable(getattr(graph, "invoke", None))