# tests/conftest.py
"""
Shared pytest fixtures used across all test modules.
"""
from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# Minimal TaskMonitoringData
# ---------------------------------------------------------------------------

@pytest.fixture()
def task_data() -> TaskMonitoringData:
    return TaskMonitoringData(
        issue_logs_id="EIL_TEST001",
        title="Test Issue Title",
        product="Lotus",
        client="TestClient",
        issue_type=IssueType.ISSUE_ERROR,
        issue_description="Something broke in the approval flow.",
        implement_status=Status.OPEN,
        pre_condition="System is running.",
        test_steps="Navigate to module > trigger action.",
        expected_result="Action completes without error.",
        urgency_level=UrgencyLevel.U2_HIGH,
        impact_level=ImpactLevel.I2_HIGH,
        priority_level=PriorityLevel.P2_HIGH,
        module="Approval Flow",
        core_function="NaveeWorkforce",
        is_recurring=False,
    )


@pytest.fixture()
def github_pr() -> GitHubPRData:
    return GitHubPRData(
        pr_number=42,
        pr_url="https://github.com/TechIgnite-Business-Solutions-Inc/repo/pull/42",
        branch_name="fix/approval-flow",
        affected_modules=["Approval Flow", "Notifications"],
        fixed_summary="Refreshed approval flow state after DA record creation.",
        prevention_steps="Add regression test for this path.",
        owner_review="Reviewed and approved by module owner.",
    )


@pytest.fixture()
def developer_data() -> DeveloperIssueData:
    return DeveloperIssueData(
        dev_status=Status.FOR_TESTING,
        pic_dev="Juan dela Cruz",
        dev_notes="Root cause traced to stale approval flow cache.",
    )


@pytest.fixture()
def quality_gate_data() -> QualityGateData:
    return QualityGateData(
        validation_status=Status.VALIDATED,
        fc_failed_testing=0,
        existing_report=False,
        quality_gate_first_pass=True,
        smoke_test_first_pass=True,
        reopen_count=0,
        qa_status=Status.PASSED,
        pic_qa="Maria Santos",
        remarks="All tests passed.",
    )


@pytest.fixture()
def rca_input(
    task_data: TaskMonitoringData,
    github_pr: GitHubPRData,
    developer_data: DeveloperIssueData,
    quality_gate_data: QualityGateData,
) -> RCAInputModel:
    return RCAInputModel(
        task_monitoring_data=task_data,
        github_pr=github_pr,
        developer_issue_data=developer_data,
        quality_gate_data=quality_gate_data,
    )


@pytest.fixture()
def rca_input_minimal(task_data: TaskMonitoringData) -> RCAInputModel:
    """RCAInputModel with only the required task data — all optionals absent."""
    return RCAInputModel(task_monitoring_data=task_data)


# ---------------------------------------------------------------------------
# Full markdown that passes review_rca validation
# ---------------------------------------------------------------------------

VALID_MARKDOWN = """\
## 1. Issue Summary
Summary text here.

## 2. Root Cause
Root cause text here.

## 3. Impact Analysis
Impact text here.

## 4. Affected Module
- Approval Flow

## 5. Quality Gate Findings
All passed.

## 6. Corrective Action
Applied fix.

## 7. Preventive Action
Added regression test.

## 8. Owner Review
Reviewed by owner.
"""


@pytest.fixture()
def valid_markdown() -> str:
    return VALID_MARKDOWN


@pytest.fixture()
def rca_output(rca_input: RCAInputModel, valid_markdown: str) -> RCAOutputModel:
    return RCAOutputModel(
        issue_id=rca_input.task_monitoring_data.issue_logs_id,
        markdown_rca=valid_markdown,
    )