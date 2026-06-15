from __future__ import annotations

import asyncio
from pathlib import Path

from agent_root.models import (
    DeveloperIssueData,
    GitHubPRData,
    ImpactLevel,
    IssueType,
    PriorityLevel,
    QualityGateData,
    RCAApprovalData,
    RCAInputModel,
    Status,
    TaskMonitoringData,
    UrgencyLevel,
)
from agent_root.service import RCAService


async def main() -> None:
    payload = RCAInputModel(
        task_monitoring_data=TaskMonitoringData(
            issue_logs_id="EIL_REAL_TEST_001",
            title="Attendance Summary computation issue",
            product="Lotus",
            client="Mamasitas",
            issue_type=IssueType.ISSUE_ERROR,
            issue_description=(
                "Attendance Summary displays incorrect "
                "work hours and absent hours."
            ),
            implement_status=Status.FOR_TESTING,
            pre_condition=(
                "Employee is assigned to a straight-time schedule."
            ),
            test_steps=(
                "1. Assign schedule\n"
                "2. Generate logs\n"
                "3. Open attendance summary"
            ),
            expected_result=(
                "Work hours and absent hours "
                "must be computed correctly."
            ),
            recommended_solution=(
                "Correct attendance computation logic."
            ),
            urgency_level=UrgencyLevel.U2_HIGH,
            impact_level=ImpactLevel.I2_HIGH,
            priority_level=PriorityLevel.P2_HIGH,
            module="Attendance Summary",
            core_function="NaveeTime",
            is_recurring=True,
        ),
        github_pr=GitHubPRData(
            pr_number=999,
            pr_url=(
                "https://github.com/"
                "TechIgnite-Business-Solutions-Inc/"
                "rnd-rca-gen/pull/999"
            ),
            branch_name="test/real-integration",
            affected_modules=[
                "Attendance Summary",
                "Timekeeping",
            ],
            fixed_summary="Fixed attendance computation logic.",
            prevention_steps="Added regression tests.",
            owner_review="Reviewed by module owner.",
        ),
        developer_issue_data=DeveloperIssueData(
            dev_status=Status.FOR_TESTING,
            pic_dev="Test Developer",
            affected_component="AttendanceSummaryService",
            root_cause=(
                "Incorrect schedule basis was used "
                "during computation."
            ),
            fix_applied=(
                "Updated attendance computation."
            ),
            verification_result=(
                "Verified through QA."
            ),
            dev_notes="Real integration test.",
        ),
        quality_gate_data=QualityGateData(
            validation_status=Status.VALIDATED,
            fc_failed_testing=0,
            existing_report=False,
            quality_gate_first_pass=True,
            smoke_test_first_pass=True,
            reopen_count=0,
            qa_status=Status.PASSED,
            pic_qa="QA Tester",
            remarks="Passed all tests.",
        ),
        approval_data=RCAApprovalData(
            prepared_by="Test Developer",
            reviewed_by_dev="Test Developer",
            validated_by_qa="QA Tester",
            approved_by_owner="Module Owner",
        ),
    )

    print("=" * 80)
    print("STARTING REAL RCA GENERATION TEST")
    print("=" * 80)

    result = await RCAService.generate_rca(payload)

    print("\nSUCCESS")
    print("=" * 80)

    print(f"Issue ID          : {result.issue_id}")
    print(f"Approval Status   : {result.approval_status}")
    print(f"PDF Download URL  : {result.pdf_file_path}")
    print(f"Generated At      : {result.generated_at}")

    print("\nMARKDOWN PREVIEW")
    print("=" * 80)
    print(result.markdown_rca[:1000])

    html_file = Path(
        f"outputs/{payload.task_monitoring_data.issue_logs_id}_rca.html"
    )

    pdf_file = Path(
        f"outputs/{payload.task_monitoring_data.issue_logs_id}_rca.pdf"
    )

    audit_file = Path(
        f"audit_logs/{payload.task_monitoring_data.issue_logs_id}.json"
    )

    print("\nARTIFACT CHECK")
    print("=" * 80)

    print(
        f"HTML Generated  : "
        f"{'YES' if html_file.exists() else 'NO'}"
    )

    print(
        f"PDF Generated   : "
        f"{'YES' if pdf_file.exists() else 'NO'}"
    )

    print(
        f"Audit Log       : "
        f"{'YES' if audit_file.exists() else 'NO'}"
    )

    print("=" * 80)

    if (
        html_file.exists()
        and pdf_file.exists()
        and audit_file.exists()
    ):
        print("INTEGRATION TEST PASSED")
    else:
        print("INTEGRATION TEST FAILED")


if __name__ == "__main__":
    asyncio.run(main())