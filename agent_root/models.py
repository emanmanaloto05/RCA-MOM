# models.py

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
    """
    Represents GitHub Pull Request data.

    Fields sourced directly from the GitHub REST API (v2026-03-10):
        - pr_number  → maps to `number` (integer) in GitHub PR response
        - pr_url     → maps to `html_url` in GitHub PR response
        - branch_name → maps to `head.ref` in GitHub PR response

    Fields that are manually populated by developers / QA (NOT from GitHub API):
        - affected_modules
        - fixed_summary
        - prevention_steps
        - owner_review
    """

    # --- GitHub API-sourced fields ---
    pr_number: Optional[int] = Field(
        default=None,
        description="GitHub PR number (integer). Maps to `number` in the GitHub REST API response.",
        gt=0,
    )
    pr_url: Optional[str] = Field(
        default=None,
        description="Full URL to the PR on GitHub. Maps to `html_url` in the GitHub REST API response.",
    )
    branch_name: Optional[str] = Field(
        default=None,
        description="Source branch name. Maps to `head.ref` in the GitHub REST API response.",
    )

    # --- Manually populated enrichment fields ---
    affected_modules: list[str] = Field(
        default_factory=list,
        description="List of modules affected by this PR. Manually populated by developer.",
    )
    fixed_summary: Optional[str] = Field(
        default=None,
        description="Summary of what was fixed. Manually populated by developer.",
    )
    prevention_steps: Optional[str] = Field(
        default=None,
        description="Steps to prevent recurrence. Manually populated by developer or QA.",
    )
    owner_review: Optional[str] = Field(
        default=None,
        description="Notes from the module owner's review. Manually populated.",
    )

    @field_validator("pr_url")
    @classmethod
    def validate_pr_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate that pr_url is a well-formed GitHub URL if provided."""
        if v is not None:
            v = v.strip()
            if not v.startswith("https://github.com/"):
                raise ValueError("pr_url must be a valid GitHub URL starting with https://github.com/")
        return v

    model_config = ConfigDict(str_strip_whitespace=True)


class DeveloperIssueData(BaseModel):
    dev_status: Optional[Status] = None
    pic_dev: Optional[str] = None
    dev_resolved_on: Optional[datetime] = None
    dev_end_date: Optional[datetime] = None
    dev_notes: Optional[str] = None

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


class RCAInputModel(BaseModel):
    task_monitoring_data: TaskMonitoringData
    github_pr: Optional[GitHubPRData] = None
    developer_issue_data: Optional[DeveloperIssueData] = None
    quality_gate_data: Optional[QualityGateData] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                # -------------------------------------------------------
                # Example 1 — EIL_2025000185: Disciplinary Action Approval Flow
                # -------------------------------------------------------
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2025000185",
                        "title": "Approval Flow of Disciplinary Action Module does not Take Effect if Approval Flow is updated",
                        "product": "Lotus",
                        "client": "Roadmap",
                        "issue_type": "Issue/Error",
                        "issue_description": "If an approval flow is created or updated after a DA application is created, the approval flow is not taking effect.",
                        "implement_status": "Cancelled",
                        "pre_condition": "DA Application, Approval Flow Setup",
                        "test_steps": "File a DA > Update Approval Flow > Login as an Approver",
                        "expected_result": "The approver should be able to view the Approve and Reject Button",
                        "recommended_solution": "Review approval flow refresh logic after DA creation.",
                        "error_message": "N/A",
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
                        "branch_name": "release/17FP2512_PL00",
                        "affected_modules": [
                            "Disciplinary Action",
                            "Approval Flow",
                        ],
                        "fixed_summary": "Updated approval flow checking logic.",
                        "prevention_steps": "Add regression test for approval flow updates after DA creation.",
                        "owner_review": "Reviewed by developer and QA.",
                    },
                    "developer_issue_data": {
                        "dev_status": "For Testing",
                        "pic_dev": "Jomar Talambayan",
                        "dev_resolved_on": "2025-12-17T13:00:00",
                        "dev_end_date": "2025-12-17T13:00:00",
                        "dev_notes": "Developer encountered approval flow state not refreshing after DA record creation.",
                    },
                    "quality_gate_data": {
                        "validation_status": "Validated",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": True,
                        "smoke_test_first_pass": True,
                        "reopen_count": 0,
                        "qa_status": "Ongoing",
                        "qa_validated_on": None,
                        "pic_qa": "Joan Marie Piñeda",
                        "remarks": "Ready for RCA testing.",
                    },
                },
                # -------------------------------------------------------
                # Example 2 — EIL_2026003374: Job Level Validation Inconsistency
                # -------------------------------------------------------
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2026003374",
                        "title": "DIREC Test Site (223): Inconsistent Job Level validation in All Applications module",
                        "product": "Lotus",
                        "client": "DBTI",
                        "issue_type": "Issue/Error",
                        "issue_description": "There is an inconsistency in how Job Level is validated across different modules.",
                        "implement_status": "Open",
                        "pre_condition": "All Applications module is accessible.",
                        "test_steps": "Navigate to the selected module and validate Job Level rules.",
                        "expected_result": "Validation rules for manual creation and import should be consistent.",
                        "recommended_solution": "Standardize Job Level validation across related modules.",
                        "error_message": "N/A",
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
                        "branch_name": "job-level-validation-fix",
                        "affected_modules": [
                            "All Applications",
                            "Job Level",
                        ],
                        "fixed_summary": "Aligned Job Level validation rules across modules.",
                        "prevention_steps": "Add validation test cases for manual creation and import flow.",
                        "owner_review": "Pending developer and QA review.",
                    },
                    "developer_issue_data": {
                        "dev_status": "Open",
                        "pic_dev": "Christian Longos",
                        "dev_notes": "Developer needs to check validation behavior difference between manual and imported records.",
                    },
                    "quality_gate_data": {
                        "validation_status": "Validated",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": False,
                        "smoke_test_first_pass": False,
                        "reopen_count": 0,
                        "qa_status": "Open",
                        "pic_qa": "Daniela Mhaey Buen",
                    },
                },
                # -------------------------------------------------------
                # Example 3 — EIL_2026003361: Adjustment Log Incorrect Data
                # -------------------------------------------------------
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2026003361",
                        "title": "TOPBOND: Incorrect data - Adjustment Log details on Adjustment Processing",
                        "product": "Lotus",
                        "client": "TopBond",
                        "issue_type": "Issue/Error",
                        "issue_description": "Upon validating the Adjustment Log, unnecessary details are shown and may confuse the payroll processors.",
                        "implement_status": "Open",
                        "pre_condition": "Late approved application and Adjustment Processing.",
                        "test_steps": "File late approved application and process adjustment processing.",
                        "expected_result": "Adjustment Log should only show adjusted data based on the processed adjustment.",
                        "recommended_solution": "Filter Adjustment Log entries to display only relevant adjusted records.",
                        "error_message": "N/A",
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
                        "branch_name": "adjustment-log-data-filtering",
                        "affected_modules": [
                            "Adjustment Log",
                            "Adjustment Processing",
                        ],
                        "fixed_summary": "Restricted Adjustment Log output to relevant processed adjustment records.",
                        "prevention_steps": "Add regression test for late approved application adjustment processing.",
                        "owner_review": "Pending payroll module owner review.",
                    },
                    "developer_issue_data": {
                        "dev_status": "Open",
                        "pic_dev": "Reymond Biol",
                        "dev_notes": "Developer encountered incorrect Adjustment Log display during payroll adjustment processing.",
                    },
                    "quality_gate_data": {
                        "validation_status": "Open",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": False,
                        "smoke_test_first_pass": False,
                        "reopen_count": 0,
                        "qa_status": "Open",
                    },
                },
                # -------------------------------------------------------
                # Example 4 — EIL_2026003509: Attendance Summary Computation
                # -------------------------------------------------------
                {
                    "task_monitoring_data": {
                        "issue_logs_id": "EIL_2026003509",
                        "title": "Attendance Summary computation of Work and Absent Hours for Straight Time",
                        "product": "Lotus",
                        "client": "Mamasitas",
                        "issue_type": "Issue/Error",
                        "issue_description": "The computation of work hours and absent hours is displayed incorrectly in Attendance Summary.",
                        "implement_status": "Open",
                        "pre_condition": "Work shift assigned should be straight time.",
                        "test_steps": "Navigate to Attendance Summary and validate the logs of the employee.",
                        "expected_result": "Work hours and absent hours should be computed correctly.",
                        "recommended_solution": "Review Attendance Summary computation for straight-time schedules.",
                        "error_message": "N/A",
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
                        "branch_name": "attendance-summary-straight-time-computation",
                        "affected_modules": [
                            "Attendance Summary",
                            "Timekeeping Computation",
                        ],
                        "fixed_summary": "Adjusted straight-time work and absent hour computation logic.",
                        "prevention_steps": "Add smoke test for straight-time schedule computation.",
                        "owner_review": "Pending timekeeping module owner review.",
                    },
                    "developer_issue_data": {
                        "dev_status": "Open",
                        "pic_dev": "Lovely Bactol",
                        "dev_notes": "Developer needs to review timekeeping computation for straight-time work schedules.",
                    },
                    "quality_gate_data": {
                        "validation_status": "Open",
                        "fc_failed_testing": 0,
                        "existing_report": False,
                        "quality_gate_first_pass": False,
                        "smoke_test_first_pass": False,
                        "reopen_count": 0,
                        "qa_status": "Open",
                    },
                },
            ]
        }
    )


class RCAOutputModel(BaseModel):
    issue_id: str = Field(..., min_length=1)
    markdown_rca: str = Field(..., min_length=1)
    pdf_file_path: Optional[str] = None
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "issue_id": "EIL_2025000185",
                "markdown_rca": "# Root Cause Analysis\n\n## Issue\n...\n\n## Root Cause\n...\n\n## Solution\n...",
                "pdf_file_path": "outputs/EIL_2025000185_rca.pdf",
                "generated_at": "2026-06-10T12:00:00Z",
            }
        },
    )