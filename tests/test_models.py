# tests/test_models.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from agent_root.models import (
    DeveloperIssueData,
    GitHubPRData,
    ImpactLevel,
    IssueType,
    PriorityLevel,
    QualityGateData,
    RCAInputModel,
    RCAOutputModel,
    Status,
    TaskMonitoringData,
    UrgencyLevel,
)


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
            "Added approval flow refresh logic after setup changes.",
        ),
        verification_result=overrides.get(
            "verification_result",
            "QA verified approval buttons now appear correctly.",
        ),
        technical_evidence=overrides.get(
            "technical_evidence",
            "Application logs confirmed stale approval flow cache before the fix.",
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
        fc_failed_testing=overrides.get(
            "fc_failed_testing",
            0,
        ),
        existing_report=overrides.get(
            "existing_report",
            False,
        ),
        quality_gate_first_pass=overrides.get(
            "quality_gate_first_pass",
        ),
        smoke_test_first_pass=overrides.get(
            "smoke_test_first_pass",
        ),
        reopen_count=overrides.get(
            "reopen_count",
            0,
        ),
        qa_validated_on=overrides.get(
            "qa_validated_on",
        ),
        pic_qa=overrides.get(
            "pic_qa",
        ),
        remarks=overrides.get(
            "remarks",
        ),
    )


class TestTaskMonitoringData:
    def test_valid_construction(self, task_data: TaskMonitoringData) -> None:
        assert task_data.issue_logs_id == "EIL_TEST001"
        assert task_data.is_recurring is False

    def test_whitespace_stripped(self) -> None:
        td = TaskMonitoringData(
            issue_logs_id="  EIL_001  ",
            title="  Title  ",
            product="  Lotus  ",
            client="  ClientA  ",
            issue_type=IssueType.ISSUE_ERROR,
            issue_description="desc",
            implement_status=Status.OPEN,
            pre_condition="pre",
            test_steps="steps",
            expected_result="result",
            urgency_level=UrgencyLevel.U1_CRITICAL,
            impact_level=ImpactLevel.I1_CRITICAL,
            priority_level=PriorityLevel.P1_CRITICAL,
            module="Module",
            core_function="CoreFn",
        )
        assert td.issue_logs_id == "EIL_001"
        assert td.title == "Title"

    def test_empty_issue_logs_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            TaskMonitoringData(
                issue_logs_id="",
                title="T",
                product="P",
                client="C",
                issue_type=IssueType.ISSUE_ERROR,
                issue_description="d",
                implement_status=Status.OPEN,
                pre_condition="p",
                test_steps="s",
                expected_result="e",
                urgency_level=UrgencyLevel.U4_LOW,
                impact_level=ImpactLevel.I4_LOW,
                priority_level=PriorityLevel.P4_LOW,
                module="M",
                core_function="CF",
            )

    def test_invalid_enum_raises(self) -> None:
        with pytest.raises(ValidationError):
            TaskMonitoringData(
                issue_logs_id="EIL_001",
                title="T",
                product="P",
                client="C",
                issue_type="INVALID_TYPE",  # type: ignore[arg-type]
                issue_description="d",
                implement_status=Status.OPEN,
                pre_condition="p",
                test_steps="s",
                expected_result="e",
                urgency_level=UrgencyLevel.U4_LOW,
                impact_level=ImpactLevel.I4_LOW,
                priority_level=PriorityLevel.P4_LOW,
                module="M",
                core_function="CF",
            )

    def test_optional_fields_default_to_none(self) -> None:
        td = TaskMonitoringData(
            issue_logs_id="EIL_001",
            title="T",
            product="P",
            client="C",
            issue_type=IssueType.FEATURE,
            issue_description="d",
            implement_status=Status.OPEN,
            pre_condition="p",
            test_steps="s",
            expected_result="e",
            urgency_level=UrgencyLevel.U3_MEDIUM,
            impact_level=ImpactLevel.I3_MEDIUM,
            priority_level=PriorityLevel.P3_MEDIUM,
            module="M",
            core_function="CF",
        )
        assert td.recommended_solution is None
        assert td.error_message is None


class TestGitHubPRData:
    def test_valid_github_pr(self, github_pr: GitHubPRData) -> None:
        assert github_pr.pr_number == 42
        assert github_pr.pr_url is not None
        assert github_pr.pr_url.startswith("https://github.com/")

    def test_invalid_pr_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            GitHubPRData(pr_url="https://gitlab.com/org/repo/pull/1")

    def test_pr_url_none_is_allowed(self) -> None:
        pr = GitHubPRData(pr_url=None)
        assert pr.pr_url is None

    def test_pr_number_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            GitHubPRData(pr_number=0)

    def test_affected_modules_defaults_to_empty_list(self) -> None:
        pr = GitHubPRData()
        assert pr.affected_modules == []

    def test_all_optional_fields_default_none(self) -> None:
        pr = GitHubPRData()
        assert pr.pr_number is None
        assert pr.branch_name is None
        assert pr.fixed_summary is None
        assert pr.prevention_steps is None
        assert pr.owner_review is None

    def test_pr_url_whitespace_stripped(self) -> None:
        pr = GitHubPRData(pr_url="  https://github.com/org/repo/pull/1  ")
        assert pr.pr_url == "https://github.com/org/repo/pull/1"


class TestDeveloperIssueData:
    def test_required_fields_must_be_supplied(self) -> None:
        with pytest.raises(ValidationError):
            DeveloperIssueData(**{})  # type: ignore[arg-type]
            
    def test_valid_dev_data(self, developer_data: DeveloperIssueData) -> None:
        assert developer_data.dev_status == Status.FOR_TESTING
        assert developer_data.pic_dev == "Juan dela Cruz"
        assert developer_data.affected_component
        assert developer_data.root_cause
        assert developer_data.fix_applied
        assert developer_data.verification_result
        assert developer_data.technical_evidence

    def test_datetime_fields(self) -> None:
        now = datetime.now(timezone.utc)
        dev = _valid_dev_data(
            dev_resolved_on=now,
            dev_end_date=now,
        )
        assert dev.dev_resolved_on == now
        assert dev.dev_end_date == now

    def test_short_technical_evidence_raises(self) -> None:
        with pytest.raises(ValidationError):
            _valid_dev_data(technical_evidence="short")


class TestQualityGateData:
    def test_required_status_fields_must_be_supplied(self) -> None:
        with pytest.raises(ValidationError):
            QualityGateData(**{})  # type: ignore[arg-type]
            
    def test_defaults_with_required_statuses(self) -> None:
        qg = _valid_quality_gate()
        assert qg.validation_status == Status.VALIDATED
        assert qg.qa_status == Status.PASSED
        assert qg.fc_failed_testing == 0
        assert qg.existing_report is False
        assert qg.reopen_count == 0
        assert qg.quality_gate_first_pass is None

    def test_negative_fc_failed_testing_raises(self) -> None:
        with pytest.raises(ValidationError):
            _valid_quality_gate(fc_failed_testing=-1)

    def test_negative_reopen_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            _valid_quality_gate(reopen_count=-1)

    def test_valid_quality_gate_data(
        self,
        quality_gate_data: QualityGateData,
    ) -> None:
        assert quality_gate_data.quality_gate_first_pass is True
        assert quality_gate_data.smoke_test_first_pass is True
        assert quality_gate_data.pic_qa == "Maria Santos"


class TestRCAInputModel:
    def test_full_model(self, rca_input: RCAInputModel) -> None:
        assert rca_input.task_monitoring_data.issue_logs_id == "EIL_TEST001"
        assert rca_input.github_pr is not None
        assert rca_input.developer_issue_data is not None
        assert rca_input.quality_gate_data is not None

    def test_minimal_model_optional_none(
        self,
        rca_input_minimal: RCAInputModel,
    ) -> None:
        assert rca_input_minimal.github_pr is None
        assert rca_input_minimal.developer_issue_data is None
        assert rca_input_minimal.quality_gate_data is None

    def test_missing_task_data_raises(self) -> None:
        with pytest.raises(ValidationError):
            RCAInputModel()  # type: ignore[call-arg]


class TestRCAOutputModel:
    def test_valid_output(self, rca_output: RCAOutputModel) -> None:
        assert rca_output.issue_id == "EIL_TEST001"
        assert "## 1. Issue Summary" in rca_output.markdown_rca

    def test_generated_at_defaults_to_utc_now(
        self,
        rca_output: RCAOutputModel,
    ) -> None:
        assert rca_output.generated_at.tzinfo == timezone.utc

    def test_pdf_file_path_optional(self) -> None:
        out = RCAOutputModel(
            issue_id="EIL_001",
            markdown_rca="# RCA",
        )
        assert out.pdf_file_path is None

    def test_empty_issue_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            RCAOutputModel(
                issue_id="",
                markdown_rca="# RCA",
            )

    def test_empty_markdown_raises(self) -> None:
        with pytest.raises(ValidationError):
            RCAOutputModel(
                issue_id="EIL_001",
                markdown_rca="",
            )


class TestEnums:
    def test_issue_type_values(self) -> None:
        assert IssueType.ISSUE_ERROR.value == "Issue/Error"
        assert IssueType.FEATURE.value == "Feature"

    def test_urgency_levels_ordered(self) -> None:
        levels = [u.value for u in UrgencyLevel]
        assert levels[0].startswith("U1")
        assert levels[-1].startswith("U4")

    def test_priority_levels(self) -> None:
        assert PriorityLevel.P1_CRITICAL.value == "P1 - Critical"
        assert PriorityLevel.P4_LOW.value == "P4 - Low"

    def test_status_closed_exists(self) -> None:
        assert Status.CLOSED.value == "Closed"