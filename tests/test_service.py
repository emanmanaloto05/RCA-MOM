# tests/test_service.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_root.models import (
    DeveloperIssueData,
    QualityGateData,
    RCAInputModel,
    RCAOutputModel,
    Status,
)
from agent_root.service import (
    RCAService,
    RCAServiceError,
    _bool_to_pass_fail as bool_to_pass_fail,  # type: ignore[reportPrivateUsage]
    _extract_section as extract_section,  # type: ignore[reportPrivateUsage]
    _resolve_consultant as resolve_consultant,  # type: ignore[reportPrivateUsage]
    extract_markdown_sections,
)
from tests.conftest import VALID_MARKDOWN


def _valid_dev_data(**overrides: Any) -> DeveloperIssueData:
    return DeveloperIssueData(
        affected_component=overrides.get(
            "affected_component",
            "ApprovalFlowService",
        ),
        root_cause=overrides.get(
            "root_cause",
            "Approval flow cache was not refreshed after setup changes.",
        ),
        fix_applied=overrides.get(
            "fix_applied",
            "Added approval flow refresh logic.",
        ),
        verification_result=overrides.get(
            "verification_result",
            "QA verified approval buttons now appear correctly.",
        ),
        technical_evidence=overrides.get(
            "technical_evidence",
            "Logs confirmed stale approval flow cache before the fix.",
        ),
        dev_status=overrides.get("dev_status"),
        pic_dev=overrides.get("pic_dev"),
        dev_resolved_on=overrides.get("dev_resolved_on"),
        dev_end_date=overrides.get("dev_end_date"),
        dev_notes=overrides.get("dev_notes"),
    )


def _valid_quality_gate(**overrides: Any) -> QualityGateData:
    return QualityGateData(
        validation_status=overrides.get(
            "validation_status",
            Status.VALIDATED,
        ),
        qa_status=overrides.get(
            "qa_status",
            Status.PASSED,
        ),
        fc_failed_testing=overrides.get("fc_failed_testing", 0),
        existing_report=overrides.get("existing_report", False),
        quality_gate_first_pass=overrides.get("quality_gate_first_pass"),
        smoke_test_first_pass=overrides.get("smoke_test_first_pass"),
        reopen_count=overrides.get("reopen_count", 0),
        qa_validated_on=overrides.get("qa_validated_on"),
        pic_qa=overrides.get("pic_qa"),
        remarks=overrides.get("remarks"),
    )


class TestExtractSection:
    def test_extracts_body_under_heading(self) -> None:
        md = "## 2. Root Cause\nThis is the root cause.\n\n## 3. Impact Analysis\nImpact."
        result: str = extract_section(md, ["## 2. Root Cause"])
        assert result == "This is the root cause."

    def test_returns_not_specified_when_no_match(self) -> None:
        result: str = extract_section("## Some Other Heading\nContent.", ["## Missing"])
        assert result == "Not specified."

    def test_tries_aliases_in_order(self) -> None:
        md = "## Root Cause\nAlias match."
        result: str = extract_section(md, ["## 2. Root Cause", "## Root Cause"])
        assert result == "Alias match."

    def test_stops_at_next_heading(self) -> None:
        md = "## 2. Root Cause\nCause content.\n## 3. Impact Analysis\nImpact."
        result: str = extract_section(md, ["## 2. Root Cause"])
        assert "Impact" not in result
        assert "Cause content." in result

    def test_returns_not_specified_for_empty_body(self) -> None:
        md = "## 2. Root Cause\n\n## 3. Impact Analysis\nImpact."
        result: str = extract_section(md, ["## 2. Root Cause"])
        assert result == "Not specified."

    def test_handles_multiline_body(self) -> None:
        md = "## 2. Root Cause\nLine one.\nLine two.\n\n## 3. Impact Analysis\nImpact."
        result: str = extract_section(md, ["## 2. Root Cause"])
        assert "Line one." in result
        assert "Line two." in result

    def test_first_alias_takes_priority(self) -> None:
        md = (
            "## 2. Root Cause\nNumbered match.\n\n"
            "## Root Cause\nUnnumbered match.\n"
        )
        result: str = extract_section(md, ["## 2. Root Cause", "## Root Cause"])
        assert result == "Numbered match."


class TestExtractMarkdownSections:
    def test_extracts_all_five_sections(self) -> None:
        sections = extract_markdown_sections(VALID_MARKDOWN)
        assert sections["cause"] != "Not specified."
        assert sections["impact_analysis"] != "Not specified."
        assert sections["solution"] != "Not specified."
        assert sections["preventive_action"] != "Not specified."
        assert sections["owner_review"] != "Not specified."

    def test_returns_not_specified_for_missing_section(self) -> None:
        sections = extract_markdown_sections("## 1. Issue Summary\nOnly one section.")
        assert sections["cause"] == "Not specified."

    def test_returns_dict_with_all_keys(self) -> None:
        sections = extract_markdown_sections("")
        assert set(sections.keys()) == {
            "cause",
            "impact_analysis",
            "solution",
            "preventive_action",
            "owner_review",
        }

    def test_handles_alias_headings(self) -> None:
        md = (
            "## 1. Issue Summary\nSummary.\n"
            "## Root Cause\nAlias cause.\n"
            "## 3. Impact Analysis\nImpact.\n"
            "## 4. Affected Module\n- Module A\n"
            "## 5. Quality Gate Findings\nPassed.\n"
            "## Corrective Action\nFix.\n"
            "## 7. Preventive Action\nPrevent.\n"
            "## 8. Owner Review\nReviewed.\n"
        )
        sections = extract_markdown_sections(md)
        assert sections["cause"] == "Alias cause."
        assert sections["solution"] == "Fix."

    def test_all_not_specified_on_empty_markdown(self) -> None:
        sections = extract_markdown_sections("")
        for value in sections.values():
            assert value == "Not specified."

    def test_cause_extracted_correctly_from_valid_markdown(self) -> None:
        sections = extract_markdown_sections(VALID_MARKDOWN)
        assert "Root cause text here" in sections["cause"]

    def test_solution_extracted_correctly_from_valid_markdown(self) -> None:
        sections = extract_markdown_sections(VALID_MARKDOWN)
        assert "Applied fix" in sections["solution"]

    def test_owner_review_extracted_correctly(self) -> None:
        sections = extract_markdown_sections(VALID_MARKDOWN)
        assert "Reviewed by owner" in sections["owner_review"]


class TestBoolToPassFail:
    def test_true_returns_passed(self) -> None:
        assert bool_to_pass_fail(True) == "Passed"

    def test_false_returns_failed(self) -> None:
        assert bool_to_pass_fail(False) == "Failed"

    def test_none_returns_na(self) -> None:
        assert bool_to_pass_fail(None) == "N/A"


class TestResolveConsultant:
    def test_prefers_pic_dev(self) -> None:
        dev = _valid_dev_data(pic_dev="Dev Name")
        qa = _valid_quality_gate(pic_qa="QA Name")
        assert resolve_consultant(dev, qa) == "Dev Name"

    def test_falls_back_to_pic_qa_when_dev_absent(self) -> None:
        qa = _valid_quality_gate(pic_qa="QA Name")
        assert resolve_consultant(None, qa) == "QA Name"

    def test_returns_empty_string_when_both_absent(self) -> None:
        assert resolve_consultant(None, None) == ""

    def test_returns_empty_when_pic_dev_is_none(self) -> None:
        dev = _valid_dev_data(pic_dev=None)
        assert resolve_consultant(dev, None) == ""

    def test_falls_back_to_qa_when_dev_has_no_name(self) -> None:
        dev = _valid_dev_data(pic_dev=None)
        qa = _valid_quality_gate(pic_qa="QA Name")
        assert resolve_consultant(dev, qa) == "QA Name"

    def test_returns_empty_when_qa_pic_also_none(self) -> None:
        dev = _valid_dev_data(pic_dev=None)
        qa = _valid_quality_gate(pic_qa=None)
        assert resolve_consultant(dev, qa) == ""


class TestRenderHtml:
    def test_returns_html_string(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_contains_doctype(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert "<!DOCTYPE html>" in html or "<html" in html

    def test_html_contains_issue_title(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert rca_input.task_monitoring_data.title in html

    def test_html_contains_client_name(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert rca_input.task_monitoring_data.client in html

    def test_html_contains_generated_date(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        current_year = str(datetime.now().year)
        assert current_year in html

    def test_html_contains_consultant_name(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert "Juan dela Cruz" in html

    def test_html_contains_rca_content(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert "Root cause text here" in html

    def test_html_contains_module_name(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert rca_input.task_monitoring_data.module in html

    def test_html_contains_quality_gate_pass(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert "Passed" in html

    def test_html_contains_pic_qa(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert "Maria Santos" in html

    def test_minimal_input_renders_without_error(
        self,
        rca_input_minimal: RCAInputModel,
        valid_markdown: str,
    ) -> None:
        output = RCAOutputModel(
            issue_id="EIL_TEST001",
            markdown_rca=valid_markdown,
        )
        html = RCAService.render_html(rca_input=rca_input_minimal, rca_output=output)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_minimal_input_still_contains_issue_title(
        self,
        rca_input_minimal: RCAInputModel,
        valid_markdown: str,
    ) -> None:
        output = RCAOutputModel(
            issue_id="EIL_TEST001",
            markdown_rca=valid_markdown,
        )
        html = RCAService.render_html(rca_input=rca_input_minimal, rca_output=output)
        assert rca_input_minimal.task_monitoring_data.title in html

    def test_render_uses_owner_review_from_pr(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> None:
        html = RCAService.render_html(rca_input=rca_input, rca_output=rca_output)
        assert "Reviewed and approved by module owner." in html


class TestGenerateHtmlFile:
    def test_writes_html_file(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
        tmp_path: Path,
    ) -> None:
        with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
            path = RCAService.generate_html_file(
                rca_input=rca_input,
                rca_output=rca_output,
            )

        assert Path(path).exists()
        assert path.endswith(".html")

    def test_html_filename_contains_issue_id(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
        tmp_path: Path,
    ) -> None:
        with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
            path = RCAService.generate_html_file(
                rca_input=rca_input,
                rca_output=rca_output,
            )

        assert "EIL_TEST001" in path

    def test_html_file_content_is_not_empty(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
        tmp_path: Path,
    ) -> None:
        with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
            path = RCAService.generate_html_file(
                rca_input=rca_input,
                rca_output=rca_output,
            )

        content = Path(path).read_text(encoding="utf-8")
        assert len(content) > 100

    def test_html_file_contains_issue_title(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
        tmp_path: Path,
    ) -> None:
        with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
            path = RCAService.generate_html_file(
                rca_input=rca_input,
                rca_output=rca_output,
            )

        content = Path(path).read_text(encoding="utf-8")
        assert rca_input.task_monitoring_data.title in content

    def test_output_dir_created_if_missing(
        self,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
        tmp_path: Path,
    ) -> None:
        nested = tmp_path / "deep" / "nested" / "outputs"
        with patch.object(RCAService, "OUTPUT_DIR", nested):
            path = RCAService.generate_html_file(
                rca_input=rca_input,
                rca_output=rca_output,
            )

        assert nested.exists()
        assert Path(path).exists()


class TestGeneratePdfFile:
    @pytest.mark.asyncio
    async def test_calls_playwright_helper(self, tmp_path: Path) -> None:
        html_path = tmp_path / "test.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        with patch("agent_root.service._generate_pdf_from_html") as mock_pdf:
            with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
                result = await RCAService.generate_pdf_file(html_path=html_path)

        mock_pdf.assert_called_once()
        assert result.endswith(".pdf")

    @pytest.mark.asyncio
    async def test_pdf_path_derived_from_html_path(self, tmp_path: Path) -> None:
        html_path = tmp_path / "EIL_TEST001_rca.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        with patch("agent_root.service._generate_pdf_from_html"):
            with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
                result = await RCAService.generate_pdf_file(html_path=html_path)

        assert "EIL_TEST001_rca.pdf" in result

    @pytest.mark.asyncio
    async def test_raises_service_error_on_failure(self, tmp_path: Path) -> None:
        html_path = tmp_path / "test.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        with patch(
            "agent_root.service._generate_pdf_from_html",
            side_effect=RCAServiceError("Playwright failed"),
        ):
            with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
                with pytest.raises(RCAServiceError, match="Playwright failed"):
                    await RCAService.generate_pdf_file(html_path=html_path)

    @pytest.mark.asyncio
    async def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        html_path = tmp_path / "test.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        nested = tmp_path / "new_outputs"

        with patch("agent_root.service._generate_pdf_from_html"):
            with patch.object(RCAService, "OUTPUT_DIR", nested):
                await RCAService.generate_pdf_file(html_path=html_path)

        assert nested.exists()


class TestGenerateRca:
    def _make_final_state(self, valid_markdown: str) -> dict[str, Any]:
        return {
            "review_passed": True,
            "review_notes": "All required sections present.",
            "rca_output": RCAOutputModel(
                issue_id="EIL_TEST001",
                markdown_rca=valid_markdown,
            ),
            "generation_error": None,
        }

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        rca_input: RCAInputModel,
        valid_markdown: str,
        tmp_path: Path,
    ) -> None:
        final_state = self._make_final_state(valid_markdown)
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
                with patch("agent_root.service._generate_pdf_from_html"):
                    result = await RCAService.generate_rca(rca_input)

        assert isinstance(result, RCAOutputModel)
        assert result.issue_id == "EIL_TEST001"
        assert result.pdf_file_path is not None

    @pytest.mark.asyncio
    async def test_pdf_file_path_set_on_output(
        self,
        rca_input: RCAInputModel,
        valid_markdown: str,
        tmp_path: Path,
    ) -> None:
        final_state = self._make_final_state(valid_markdown)
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
                with patch("agent_root.service._generate_pdf_from_html"):
                    result = await RCAService.generate_rca(rca_input)

        assert result.pdf_file_path is not None
        assert result.pdf_file_path.endswith(".pdf")

    @pytest.mark.asyncio
    async def test_raises_when_review_failed(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        final_state: dict[str, Any] = {
            "review_passed": False,
            "review_notes": "Missing sections: ## 3. Impact Analysis",
            "rca_output": None,
            "generation_error": None,
        }

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with pytest.raises(RCAServiceError, match="did not pass review"):
                await RCAService.generate_rca(rca_input)

    @pytest.mark.asyncio
    async def test_raises_on_whitespace_only_markdown(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        rca_out = RCAOutputModel(issue_id="EIL_TEST001", markdown_rca="x")
        rca_out.markdown_rca = "   "

        final_state: dict[str, Any] = {
            "review_passed": True,
            "review_notes": "All required sections present.",
            "rca_output": rca_out,
            "generation_error": None,
        }

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with pytest.raises(RCAServiceError, match="empty"):
                await RCAService.generate_rca(rca_input)

    @pytest.mark.asyncio
    async def test_raises_when_graph_build_fails(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        from agent_root.graph import GraphBuildError

        with patch(
            "agent_root.service.get_rca_graph",
            side_effect=GraphBuildError("Graph failed"),
        ):
            with pytest.raises(RCAServiceError, match="could not be initialized"):
                await RCAService.generate_rca(rca_input)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_graph_output_type(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = "not a dict"

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with pytest.raises(RCAServiceError, match="Invalid graph output type"):
                await RCAService.generate_rca(rca_input)

    @pytest.mark.asyncio
    async def test_raises_when_rca_output_missing(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        final_state: dict[str, Any] = {
            "review_passed": True,
            "review_notes": "All required sections present.",
            "rca_output": None,
            "generation_error": None,
        }

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with pytest.raises(RCAServiceError, match="missing or invalid"):
                await RCAService.generate_rca(rca_input)

    @pytest.mark.asyncio
    async def test_raises_on_unexpected_graph_exception(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        with patch(
            "agent_root.service.get_rca_graph",
            side_effect=RuntimeError("Unexpected crash"),
        ):
            with pytest.raises(RCAServiceError, match="RCA workflow failed"):
                await RCAService.generate_rca(rca_input)

    @pytest.mark.asyncio
    async def test_generation_error_in_state_surfaces_in_exception(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        final_state: dict[str, Any] = {
            "review_passed": False,
            "review_notes": None,
            "rca_output": None,
            "generation_error": "Gemini timed out",
        }

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with pytest.raises(RCAServiceError, match="Gemini timed out"):
                await RCAService.generate_rca(rca_input)

    @pytest.mark.asyncio
    async def test_html_file_written_during_pipeline(
        self,
        rca_input: RCAInputModel,
        valid_markdown: str,
        tmp_path: Path,
    ) -> None:
        final_state = self._make_final_state(valid_markdown)
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = final_state

        with patch("agent_root.service.get_rca_graph", return_value=mock_graph):
            with patch.object(RCAService, "OUTPUT_DIR", tmp_path):
                with patch("agent_root.service._generate_pdf_from_html"):
                    await RCAService.generate_rca(rca_input)

        html_file = tmp_path / "EIL_TEST001_rca.html"
        assert html_file.exists()