from typing import Optional
from pydantic import BaseModel, ConfigDict


class TaskMonitoringData(BaseModel):
    issue_logs_id: str
    title: str
    product: Optional[str] = None
    client: Optional[str] = None
    issue_type: Optional[str] = None
    issue_description: str
    implement_status: Optional[str] = None
    pre_condition: Optional[str] = None
    test_steps: Optional[str] = None
    expected_result: Optional[str] = None
    error_message: Optional[str] = None
    urgency_level: Optional[str] = None
    impact_level: Optional[str] = None
    priority_level: Optional[str] = None
    module: Optional[str] = None
    core_function: Optional[str] = None
    is_recurring: Optional[bool] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "error_message": "N/A",
                "urgency_level": "U4 - Low",
                "impact_level": "I4 - Low",
                "priority_level": "P4 - Low",
                "module": "Disciplinary Action",
                "core_function": "NaveeWorkforce",
                "is_recurring": False
            }
        }
    )


class QualityGateData(BaseModel):
    fc_failed_testing: Optional[int] = 0
    validation_status: Optional[str] = None
    existing_report: Optional[bool] = False
    quality_gate_first_pass: Optional[bool] = None
    smoke_test_first_pass: Optional[bool] = None
    reopen_count: Optional[int] = 0


class GitHubPRData(BaseModel):
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    branch_name: Optional[str] = None
    affected_modules: list[str] = []
    fixed_summary: Optional[str] = None
    prevention_steps: Optional[str] = None
    owner_review: Optional[str] = None


class RCAInputModel(BaseModel):
    task_monitoring_data: TaskMonitoringData
    quality_gate_data: Optional[QualityGateData] = None
    github_pr: Optional[GitHubPRData] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                    "error_message": "N/A",
                    "urgency_level": "U4 - Low",
                    "impact_level": "I4 - Low",
                    "priority_level": "P4 - Low",
                    "module": "Disciplinary Action",
                    "core_function": "NaveeWorkforce",
                    "is_recurring": False
                },
                "quality_gate_data": {
                    "fc_failed_testing": 0,
                    "validation_status": "Validated",
                    "existing_report": False,
                    "quality_gate_first_pass": True,
                    "smoke_test_first_pass": True,
                    "reopen_count": 0
                },
                "github_pr": {
                    "pr_number": 1,
                    "pr_url": "https://github.com/sample/pr/1",
                    "branch_name": "release/17FP2512_PL00",
                    "affected_modules": ["Disciplinary Action", "Approval Flow"],
                    "fixed_summary": "Updated approval flow checking logic.",
                    "prevention_steps": "Add regression test for approval flow updates after DA creation.",
                    "owner_review": "Reviewed by assigned developer and QA."
                }
            }
        }
    )


class RCAOutputModel(BaseModel):
    issue_id: str
    markdown_rca: str
    pdf_file_path: Optional[str] = None