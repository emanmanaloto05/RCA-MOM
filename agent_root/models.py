# agent_root/models.py
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IssueType(str, Enum):
    ISSUE_ERROR = "Issue/Error"
    FEATURE = "Feature"
    ENHANCEMENT = "Enhancement"
    ROADMAP = "Roadmap"


class Status(str, Enum):
    OPEN = "Open"
    ONGOING = "Ongoing"
    CANCELLED = "Cancelled"
    VALIDATED = "Validated"
    FOR_TESTING = "For Testing"
    FOR_DEPLOYMENT = "For Deployment"
    CLOSED = "Closed"
    PASSED = "Passed"
    TAGGED_TO_DEV = "Tagged to Dev"


class UrgencyLevel(str, Enum):
    U1_CRITICAL = "U1 - Critical"
    U2_HIGH = "U2 - High"
    U3_MEDIUM = "U3 - Medium"
    U4_LOW = "U4 - Low"


class ImpactLevel(str, Enum):
    I1_CRITICAL = "I1 - Critical"
    I2_HIGH = "I2 - High"
    I3_MEDIUM = "I3 - Medium"
    I4_LOW = "I4 - Low"


class PriorityLevel(str, Enum):
    P1_CRITICAL = "P1 - Critical"
    P2_HIGH = "P2 - High"
    P3_MEDIUM = "P3 - Medium"
    P4_LOW = "P4 - Low"


class ApprovalStatus(str, Enum):
    DRAFT = "Draft"
    FOR_REVIEW = "For Review"
    APPROVED = "Approved"
    REJECTED = "Rejected"


class TaskMonitoringData(BaseModel):
    issue_logs_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    product: str = Field(..., min_length=1)
    client: str = Field(..., min_length=1)
    issue_type: IssueType
    issue_description: str = Field(..., min_length=1)
    implement_status: Status
    pre_condition: str = Field(..., min_length=1)
    test_steps: str = Field(..., min_length=1)
    expected_result: str = Field(..., min_length=1)
    recommended_solution: Optional[str] = None
    error_message: Optional[str] = None
    urgency_level: UrgencyLevel
    impact_level: ImpactLevel
    priority_level: PriorityLevel
    module: str = Field(..., min_length=1)
    core_function: str = Field(..., min_length=1)
    is_recurring: bool = False

    model_config = ConfigDict(str_strip_whitespace=True)


class GitHubPRData(BaseModel):
    pr_number: Optional[int] = Field(default=None, gt=0)
    pr_url: Optional[str] = None
    branch_name: Optional[str] = None
    affected_modules: list[str] = Field(default_factory=list)
    fixed_summary: Optional[str] = None
    prevention_steps: Optional[str] = None
    owner_review: Optional[str] = None

    @field_validator("pr_url")
    @classmethod
    def validate_pr_url(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value.startswith("https://github.com/"):
                raise ValueError(
                    "pr_url must be a valid GitHub URL starting with https://github.com/"
                )
        return value

    model_config = ConfigDict(str_strip_whitespace=True)


class DeveloperIssueData(BaseModel):
    dev_status: Optional[Status] = None
    pic_dev: Optional[str] = None
    dev_resolved_on: Optional[datetime] = None
    dev_end_date: Optional[datetime] = None
    affected_component: Optional[str] = None
    root_cause: Optional[str] = None
    fix_applied: Optional[str] = None
    verification_result: Optional[str] = None
    dev_notes: Optional[str] = None
    technical_evidence: Optional[str] = Field(
        default=None,
        min_length=10,
        description=(
            "Concrete technical evidence such as logs, error messages, stack traces, "
            "affected function, API responses, or database query results that confirm "
            "the root cause of the issue."
        ),
    )

    model_config = ConfigDict(str_strip_whitespace=True)


class QualityGateData(BaseModel):
    validation_status: Optional[Status] = None
    fc_failed_testing: int = Field(default=0, ge=0)
    existing_report: bool = False
    quality_gate_first_pass: Optional[bool] = None
    smoke_test_first_pass: Optional[bool] = None
    reopen_count: int = Field(default=0, ge=0)
    qa_status: Optional[Status] = None
    qa_validated_on: Optional[datetime] = None
    pic_qa: Optional[str] = None
    remarks: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True)


class Attachment(BaseModel):
    filename: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    content_type: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True)


class RCAApprovalData(BaseModel):
    prepared_by: Optional[str] = None
    reviewed_by_dev: Optional[str] = None
    validated_by_qa: Optional[str] = None
    approved_by_owner: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True)


class RCAInputModel(BaseModel):
    task_monitoring_data: TaskMonitoringData
    github_pr: Optional[GitHubPRData] = None
    developer_issue_data: Optional[DeveloperIssueData] = None
    quality_gate_data: Optional[QualityGateData] = None
    approval_data: Optional[RCAApprovalData] = None
    attachments: list[Attachment] = Field(
        default_factory=lambda: []
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2025000185",
                        "title": "Approval Flow does not Take Effect if Updated",
                        "product": "Lotus",
                        "client": "AMC",
                        "issue_type": "Issue/Error",
                        "issue_description": "Approval flow does not take effect when the approval flow setup is updated after a Disciplinary Action record has already been created.",
                        "implement_status": "For Testing",
                        "pre_condition": "A Disciplinary Action record exists and Approval Flow Setup is configured for the module.",
                        "test_steps": "1. File a Disciplinary Action record.\n2. Update the Approval Flow Setup after the DA record is created.\n3. Log in as the assigned approver.\n4. Open the existing DA record.\n5. Verify whether the Approve and Reject buttons are displayed.",
                        "expected_result": "The approver should see the Approve and Reject buttons based on the latest approval flow configuration.",
                        "recommended_solution": "Ensure that existing Disciplinary Action records reload or refresh the latest approval flow configuration after approval-flow setup changes.",
                        "error_message": "Not available.",
                        "urgency_level": "U4 - Low",
                        "impact_level": "I4 - Low",
                        "priority_level": "P4 - Low",
                        "module": "Disciplinary Action",
                        "core_function": "NaveeWorkforce",
                        "is_recurring": False,
                    },
                    "github_pr": {
                        "pr_number": 1,
                        "pr_url": "https://github.com/TechIgnite-Business-Solutions-Inc/rnd-rca-gen/pull/1",
                        "branch_name": "fix/approval-flow-refresh",
                        "affected_modules": [
                            "Disciplinary Action",
                            "Approval Flow",
                            "ApprovalFlowService",
                        ],
                        "fixed_summary": "Added approval flow refresh logic so the system reloads the latest approval configuration when approval-flow setup changes are detected for existing Disciplinary Action records.",
                        "prevention_steps": "Add regression test cases for Disciplinary Action approval-flow updates.",
                        "owner_review": "Reviewed by the module owner.",
                    },
                    "developer_issue_data": {
                        "dev_status": "For Testing",
                        "pic_dev": "Jomar Talambayan",
                        "affected_component": "ApprovalFlowService",
                        "root_cause": "The approval workflow configuration is initialized only during record creation.",
                        "fix_applied": "Added approval flow refresh logic.",
                        "verification_result": "QA validated that Approve and Reject buttons are now displayed correctly.",
                        "dev_notes": "Approval flow configuration now refreshes for existing DA records.",
                        "technical_evidence": "Approval API returned HTTP 500 when approver_id was missing from the request payload after approval flow setup was updated. Server logs showed NullReferenceException in ApprovalFlowService.GetCurrentApprover() at line 142.",
                    },
                    "quality_gate_data": {
                        "validation_status": "Validated",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": True,
                        "smoke_test_first_pass": True,
                        "reopen_count": 0,
                        "qa_status": "Passed",
                        "pic_qa": "Joan Marie Piñeda",
                        "remarks": "QA validated the fix.",
                    },
                    "approval_data": {
                        "prepared_by": "Jomar Talambayan",
                        "reviewed_by_dev": "Jomar Talambayan",
                        "validated_by_qa": "Joan Marie Piñeda",
                        "approved_by_owner": "Module Owner",
                    },
                    "attachments": [],
                },
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2026003374",
                        "title": "Inconsistent Job Level validation in All Applications module",
                        "product": "Lotus",
                        "client": "DBTI",
                        "issue_type": "Issue/Error",
                        "issue_description": "Job Level validation behaves inconsistently between manual application creation and imported application records.",
                        "implement_status": "For Testing",
                        "pre_condition": "All Applications module is accessible and Job Level setup contains active validation rules.",
                        "test_steps": "1. Create an application manually with Job Level data.\n2. Import an application record with Job Level data.\n3. Compare validation behavior.",
                        "expected_result": "Manual and imported application records should follow the same Job Level validation rules.",
                        "recommended_solution": "Standardize Job Level validation logic.",
                        "error_message": "Not available.",
                        "urgency_level": "U2 - High",
                        "impact_level": "I2 - High",
                        "priority_level": "P2 - High",
                        "module": "All Applications",
                        "core_function": "NaveeHire",
                        "is_recurring": False,
                    },
                    "github_pr": {
                        "pr_number": 2,
                        "pr_url": "https://github.com/TechIgnite-Business-Solutions-Inc/rnd-rca-gen/pull/2",
                        "branch_name": "fix/job-level-validation",
                        "affected_modules": [
                            "All Applications",
                            "Job Level Validation",
                            "Application Import",
                        ],
                        "fixed_summary": "Aligned Job Level validation rules.",
                        "prevention_steps": "Add validation test cases.",
                        "owner_review": "Reviewed by recruitment module owner.",
                    },
                    "developer_issue_data": {
                        "dev_status": "For Testing",
                        "pic_dev": "Christian Longos",
                        "affected_component": "JobLevelValidationService",
                        "root_cause": "Manual application creation and import processing used separate validation paths.",
                        "fix_applied": "Updated import validation path to reuse the same Job Level validation rules.",
                        "verification_result": "QA validated consistent validation behavior.",
                        "dev_notes": "Both creation paths now share one validation service.",
                        "technical_evidence": "Import endpoint bypassed JobLevelValidationService.Validate() and called a legacy validateJobLevel() function directly, confirmed via stack trace in application logs showing divergent call paths for POST /api/applications vs POST /api/applications/import.",
                    },
                    "quality_gate_data": {
                        "validation_status": "Validated",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": True,
                        "smoke_test_first_pass": True,
                        "reopen_count": 0,
                        "qa_status": "Passed",
                        "pic_qa": "Daniela Mhaey Buen",
                        "remarks": "QA validated consistency.",
                    },
                    "approval_data": {
                        "prepared_by": "Christian Longos",
                        "reviewed_by_dev": "Christian Longos",
                        "validated_by_qa": "Daniela Mhaey Buen",
                        "approved_by_owner": "Recruitment Module Owner",
                    },
                    "attachments": [],
                },
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2026003361",
                        "title": "Incorrect Adjustment Log details on Adjustment Processing",
                        "product": "Lotus",
                        "client": "TopBond",
                        "issue_type": "Issue/Error",
                        "issue_description": "Adjustment Log displays unnecessary entries during Adjustment Processing.",
                        "implement_status": "For Testing",
                        "pre_condition": "Late approved application exists and Adjustment Processing is available.",
                        "test_steps": "1. File a late approved application.\n2. Run Adjustment Processing.\n3. Open Adjustment Log.\n4. Validate records.",
                        "expected_result": "Adjustment Log should only display records directly related to the processed adjustment.",
                        "recommended_solution": "Filter Adjustment Log output.",
                        "error_message": "Not available.",
                        "urgency_level": "U1 - Critical",
                        "impact_level": "I1 - Critical",
                        "priority_level": "P1 - Critical",
                        "module": "Adjustment Log",
                        "core_function": "NaveePay",
                        "is_recurring": False,
                    },
                    "github_pr": {
                        "pr_number": 3,
                        "pr_url": "https://github.com/TechIgnite-Business-Solutions-Inc/rnd-rca-gen/pull/3",
                        "branch_name": "fix/adjustment-log-filtering",
                        "affected_modules": [
                            "Adjustment Log",
                            "Adjustment Processing",
                            "Payroll Adjustment",
                        ],
                        "fixed_summary": "Restricted Adjustment Log output.",
                        "prevention_steps": "Add regression tests.",
                        "owner_review": "Reviewed by payroll module owner.",
                    },
                    "developer_issue_data": {
                        "dev_status": "For Testing",
                        "pic_dev": "Reymond Biol",
                        "affected_component": "AdjustmentLogService",
                        "root_cause": "Adjustment Log retrieval was not scoped to the current processed adjustment transaction.",
                        "fix_applied": "Updated Adjustment Log filtering condition.",
                        "verification_result": "QA confirmed unnecessary entries no longer appear.",
                        "dev_notes": "Filtering now uses adjustment transaction ID.",
                        "technical_evidence": "Database query in AdjustmentLogService.GetLogs() returned 47 unrelated adjustment entries for transaction ID ADJ-2026-00391 because the WHERE clause lacked a transaction_id filter. Raw SQL log confirmed: SELECT * FROM adjustment_log WHERE employee_id = :emp_id (missing AND transaction_id = :txn_id).",
                    },
                    "quality_gate_data": {
                        "validation_status": "Validated",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": True,
                        "smoke_test_first_pass": True,
                        "reopen_count": 0,
                        "qa_status": "Passed",
                        "pic_qa": "Maria Santos",
                        "remarks": "QA confirmed only relevant records appear.",
                    },
                    "approval_data": {
                        "prepared_by": "Reymond Biol",
                        "reviewed_by_dev": "Reymond Biol",
                        "validated_by_qa": "Maria Santos",
                        "approved_by_owner": "Payroll Module Owner",
                    },
                    "attachments": [
                        {
                            "filename": "adjustment_log_sample.pdf",
                            "file_path": "uploads/EIL_2026003361/adjustment_log_sample.pdf",
                            "content_type": "application/pdf",
                            "description": "Sample Adjustment Log showing unnecessary entries before filtering correction.",
                        }
                    ],
                },
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2026003509",
                        "title": "Attendance Summary computation of Work and Absent Hours for Straight Time",
                        "product": "Lotus",
                        "client": "Mamasitas",
                        "issue_type": "Issue/Error",
                        "issue_description": "Attendance Summary displays incorrect work hours and absent hours for straight-time schedules.",
                        "implement_status": "For Testing",
                        "pre_condition": "Employee is assigned to a straight-time work schedule.",
                        "test_steps": "1. Assign employee to straight-time schedule.\n2. Generate attendance logs.\n3. Open Attendance Summary.\n4. Compare computed hours.",
                        "expected_result": "Work hours and absent hours should be computed correctly.",
                        "recommended_solution": "Correct Attendance Summary computation logic.",
                        "error_message": "Not available.",
                        "urgency_level": "U2 - High",
                        "impact_level": "I2 - High",
                        "priority_level": "P2 - High",
                        "module": "Attendance Summary",
                        "core_function": "NaveeTime",
                        "is_recurring": True,
                    },
                    "github_pr": {
                        "pr_number": 4,
                        "pr_url": "https://github.com/TechIgnite-Business-Solutions-Inc/rnd-rca-gen/pull/4",
                        "branch_name": "fix/straight-time-attendance-computation",
                        "affected_modules": [
                            "Attendance Summary",
                            "Timekeeping Computation",
                            "Straight-Time Schedule",
                        ],
                        "fixed_summary": "Corrected straight-time attendance computation.",
                        "prevention_steps": "Add smoke and regression tests.",
                        "owner_review": "Reviewed by timekeeping module owner.",
                    },
                    "developer_issue_data": {
                        "dev_status": "For Testing",
                        "pic_dev": "Lovely Bactol",
                        "affected_component": "AttendanceSummaryComputationService",
                        "root_cause": "Straight-time schedule computation did not consistently apply the expected scheduled work-hour basis.",
                        "fix_applied": "Updated attendance computation logic.",
                        "verification_result": "QA confirmed work hours and absent hours are correct.",
                        "dev_notes": "Schedule basis is now resolved per employee schedule assignment.",
                        "technical_evidence": "AttendanceSummaryComputationService.ComputeHours() applied a default 8-hour basis instead of the employee's assigned straight-time schedule hours (7.5 hrs), confirmed via debug log showing schedule_basis=DEFAULT for employee ID EMP-20045 on 2026-05-12. Computed work_hours=8.0 vs expected 7.5, absent_hours=0.5 vs expected 0.0.",
                    },
                    "quality_gate_data": {
                        "validation_status": "Validated",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": True,
                        "smoke_test_first_pass": True,
                        "reopen_count": 0,
                        "qa_status": "Passed",
                        "pic_qa": "Lovely Bactol",
                        "remarks": "QA confirmed correct Attendance Summary computation.",
                    },
                    "approval_data": {
                        "prepared_by": "Lovely Bactol",
                        "reviewed_by_dev": "Lovely Bactol",
                        "validated_by_qa": "Lovely Bactol",
                        "approved_by_owner": "Timekeeping Module Owner",
                    },
                    "attachments": [],
                },
            ]
        }
    )


class RCAOutputModel(BaseModel):
    issue_id: str = Field(..., min_length=1)
    markdown_rca: str = Field(..., min_length=1)
    pdf_file_path: Optional[str] = None
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "issue_id": "EIL_2025000185",
                "markdown_rca": "# Root Cause Analysis\n\n## 1. Issue Summary\n...\n\n## 2. Root Cause\n...",
                "pdf_file_path": "outputs/EIL_2025000185_rca.pdf",
                "approval_status": "Draft",
                "generated_at": "2026-06-10T12:00:00Z",
            }
        },
    )