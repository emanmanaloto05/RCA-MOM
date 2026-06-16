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
# Full markdown that passes review_rca validation.
#
# Rules it must satisfy (enforced by review_rca / _validate_rca_quality):
#   1. All 8 required section headings present.
#   2. No forbidden/speculative phrases (probably, maybe, could be, etc.).
#   3. Weak-placeholder occurrences ("not available", "not specified", "n/a")
#      must stay below _MAX_WEAK_PLACEHOLDER_COUNT (14).
#   4. "## 2. Root Cause" section body >= _MIN_SECTION_LENGTH (50 chars).
#   5. "## 6. Corrective Action" section body >= _MIN_SECTION_LENGTH (50 chars).
#
# Additionally, test_service.py asserts these exact substrings are present:
#   - "Root cause text here"   (TestExtractMarkdownSections)
#   - "Applied fix"            (TestExtractMarkdownSections)
#   - "Reviewed by owner"      (TestExtractMarkdownSections / TestRenderHtml)
# ---------------------------------------------------------------------------

VALID_MARKDOWN = """\
## 1. Issue Summary
Issue EIL_TEST001 affects the Approval Flow module in the Lotus product for TestClient.
The current implementation status is Open with a High priority level.

## 2. Root Cause
Root cause text here. The approval flow configuration was initialized only at record
creation time and was not refreshed when the approval flow setup was subsequently
updated, causing the Approve and Reject buttons to disappear for existing records.

## 3. Impact Analysis
The issue has a High urgency and High impact level. Approvers could not action existing
records after an approval flow update, blocking downstream processing. Not recurring.

## 4. Affected Module
- Product: Lotus
- Module: Approval Flow
- Core Function: NaveeWorkforce
- Affected Component: ApprovalFlowService

## 5. Quality Gate Findings
Quality Gate First Pass: Passed. Smoke Test First Pass: Passed. QA Status: Passed.
Validation Status: Validated. Reopen Count: 0. FC Failed Testing: 0.
Remarks: All tests passed.

## 6. Corrective Action
Applied fix: approval flow refresh logic was added to ApprovalFlowService so that the
system reloads the latest approval configuration when approval flow setup changes are
detected for existing records, ensuring the correct approver is resolved at runtime.

## 7. Preventive Action
A regression test case covering approval flow updates on existing records was added
to the test suite to prevent recurrence of this class of defect in the future.

## 8. Owner Review
Reviewed by owner. The module owner confirmed that the fix resolves the issue and
that the QA team validated the corrective action successfully.
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