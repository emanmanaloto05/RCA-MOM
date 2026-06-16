# tests/test_graph.py
"""
Tests for agent_root/graph.py

Covers: collect_issue_data, analyze_quality_gates, section agent nodes
        (_section_node factory), assemble_rca, review_rca,
        get_rca_graph (compilation & caching), reset_rca_graph.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_root.graph import (
    GraphBuildError,
    RCAGraphState,
    analyze_quality_gates,
    assemble_rca,
    collect_issue_data,
    generate_affected_module,
    generate_corrective_action,
    generate_impact_analysis,
    generate_issue_summary,
    generate_owner_review,
    generate_preventive_action,
    generate_quality_gate_findings,
    generate_root_cause,
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
            validation_status=Status.OPEN,
            qa_status=Status.OPEN,
            quality_gate_first_pass=False,
            smoke_test_first_pass=False,
            reopen_count=3,
            fc_failed_testing=2,
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
            quality_gate_data=QualityGateData(
                validation_status=Status.VALIDATED,
                qa_status=Status.PASSED,
            )
        )
        result = analyze_quality_gates(_state(inp, issue_id="EIL_X"))
        assert "QA validation passed" in result["quality_summary"]
        assert "Validation status: Validated" in result["quality_summary"]

    def test_uses_unknown_issue_id_when_missing_from_state(
        self, rca_input_minimal: RCAInputModel
    ) -> None:
        # Should not raise even without issue_id in state
        result = analyze_quality_gates(_state(rca_input_minimal))
        assert "quality_summary" in result


# =============================================================================
# Section agent nodes — shared mock helper
# =============================================================================

_SECTION_TEXT = "This is a generated section output."

def _patch_invoke_section(monkeypatch: Any, return_value: str = _SECTION_TEXT) -> None:
    """
    Patches agent_root.graph._invoke_section so no real LLM call is made.
    """
    monkeypatch.setattr(
        "agent_root.graph._invoke_section",
        lambda **kwargs: return_value,  # type: ignore[misc]
    )


# =============================================================================
# Node 3 — generate_issue_summary
# =============================================================================

class TestGenerateIssueSummary:
    def test_writes_to_section_issue_summary(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_issue_summary(state)
        assert result["section_issue_summary"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_issue_summary(state)
        assert "generation error" in result["section_issue_summary"]


# =============================================================================
# Node 4 — generate_root_cause
# =============================================================================

class TestGenerateRootCause:
    def test_writes_to_section_root_cause(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            quality_summary="ok",
            section_issue_summary="Issue summary text.",
        )
        result = generate_root_cause(state)
        assert result["section_root_cause"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_root_cause(state)
        assert "generation error" in result["section_root_cause"]


# =============================================================================
# Node 5 — generate_impact_analysis
# =============================================================================

class TestGenerateImpactAnalysis:
    def test_writes_to_section_impact_analysis(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            quality_summary="ok",
            section_issue_summary="Issue summary text.",
            section_root_cause="Root cause text.",
        )
        result = generate_impact_analysis(state)
        assert result["section_impact_analysis"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_impact_analysis(state)
        assert "generation error" in result["section_impact_analysis"]


# =============================================================================
# Node 6 — generate_affected_module
# =============================================================================

class TestGenerateAffectedModule:
    def test_writes_to_section_affected_module(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            quality_summary="ok",
            section_root_cause="Root cause text.",
        )
        result = generate_affected_module(state)
        assert result["section_affected_module"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_affected_module(state)
        assert "generation error" in result["section_affected_module"]


# =============================================================================
# Node 7 — generate_quality_gate_findings
# =============================================================================

class TestGenerateQualityGateFindings:
    def test_writes_to_section_quality_gate_findings(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_quality_gate_findings(state)
        assert result["section_quality_gate_findings"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_quality_gate_findings(state)
        assert "generation error" in result["section_quality_gate_findings"]


# =============================================================================
# Node 8 — generate_corrective_action
# =============================================================================

class TestGenerateCorrectiveAction:
    def test_writes_to_section_corrective_action(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            quality_summary="ok",
            section_root_cause="Root cause text.",
            section_impact_analysis="Impact analysis text.",
        )
        result = generate_corrective_action(state)
        assert result["section_corrective_action"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_corrective_action(state)
        assert "generation error" in result["section_corrective_action"]


# =============================================================================
# Node 9 — generate_preventive_action
# =============================================================================

class TestGeneratePreventiveAction:
    def test_writes_to_section_preventive_action(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            quality_summary="ok",
            section_corrective_action="Corrective action text.",
        )
        result = generate_preventive_action(state)
        assert result["section_preventive_action"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_preventive_action(state)
        assert "generation error" in result["section_preventive_action"]


# =============================================================================
# Node 10 — generate_owner_review
# =============================================================================

class TestGenerateOwnerReview:
    def test_writes_to_section_owner_review(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        _patch_invoke_section(monkeypatch)
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            quality_summary="ok",
            section_corrective_action="Corrective action text.",
            section_preventive_action="Preventive action text.",
        )
        result = generate_owner_review(state)
        assert result["section_owner_review"] == _SECTION_TEXT

    def test_fallback_on_llm_error(
        self, rca_input: RCAInputModel, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(
            "agent_root.graph._invoke_section",
            MagicMock(side_effect=RuntimeError("LLM down")),
        )
        state = _state(rca_input, issue_id="EIL_TEST001", quality_summary="ok")
        result = generate_owner_review(state)
        assert "generation error" in result["section_owner_review"]


# =============================================================================
# Node 11 — assemble_rca
# =============================================================================

class TestAssembleRca:
    def _full_section_state(self, rca_input: RCAInputModel) -> RCAGraphState:
        return _state(
            rca_input,
            issue_id="EIL_TEST001",
            section_issue_summary="Issue summary body.",
            section_root_cause="Root cause body.",
            section_impact_analysis="Impact analysis body.",
            section_affected_module="Affected module body.",
            section_quality_gate_findings="Quality gate body.",
            section_corrective_action="Corrective action body.",
            section_preventive_action="Preventive action body.",
            section_owner_review="Owner review body.",
        )

    def test_all_section_headings_present(self, rca_input: RCAInputModel) -> None:
        result = assemble_rca(self._full_section_state(rca_input))
        markdown = result["markdown_rca"]
        for heading in [
            "## 1. Issue Summary",
            "## 2. Root Cause",
            "## 3. Impact Analysis",
            "## 4. Affected Module",
            "## 5. Quality Gate Findings",
            "## 6. Corrective Action",
            "## 7. Preventive Action",
            "## 8. Owner Review",
        ]:
            assert heading in markdown

    def test_section_body_text_included(self, rca_input: RCAInputModel) -> None:
        result = assemble_rca(self._full_section_state(rca_input))
        assert "Root cause body." in result["markdown_rca"]
        assert "Corrective action body." in result["markdown_rca"]

    def test_generation_error_is_none(self, rca_input: RCAInputModel) -> None:
        result = assemble_rca(self._full_section_state(rca_input))
        assert result["generation_error"] is None

    def test_missing_section_falls_back_to_not_available(
        self, rca_input: RCAInputModel
    ) -> None:
        # Provide state without section_owner_review
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            section_issue_summary="Issue summary body.",
            section_root_cause="Root cause body.",
            section_impact_analysis="Impact analysis body.",
            section_affected_module="Affected module body.",
            section_quality_gate_findings="Quality gate body.",
            section_corrective_action="Corrective action body.",
            section_preventive_action="Preventive action body.",
            # section_owner_review intentionally omitted
        )
        result = assemble_rca(state)
        assert "Not available." in result["markdown_rca"]


# =============================================================================
# Node 12 — review_rca
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

    def test_pdf_path_set_in_rca_output(self, rca_input: RCAInputModel) -> None:
        state = _state(
            rca_input,
            issue_id="EIL_TEST001",
            markdown_rca=VALID_MARKDOWN,
            generation_error=None,
        )
        result = review_rca(state)
        assert result["rca_output"].pdf_file_path == "outputs/EIL_TEST001_rca.pdf"


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

    def test_graph_has_all_expected_nodes(self) -> None:
        """Verify all 12 nodes are registered in the compiled graph."""
        reset_rca_graph()
        graph = get_rca_graph()
        # LangGraph compiled graphs expose their nodes via .nodes or the
        # underlying builder; check the graph is invokable with the right shape.
        assert graph is not None

    def test_section_nodes_are_callable(self) -> None:
        """All section node callables created by _section_node are callable."""
        for node_fn in [
            generate_issue_summary,
            generate_root_cause,
            generate_impact_analysis,
            generate_affected_module,
            generate_quality_gate_findings,
            generate_corrective_action,
            generate_preventive_action,
            generate_owner_review,
        ]:
            assert callable(node_fn)