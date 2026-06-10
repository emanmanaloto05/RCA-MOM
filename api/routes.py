from fastapi import APIRouter

from agent_root.models import RCAInputModel, RCAOutputModel

router = APIRouter(
    prefix="/api",
    tags=["RCA"]
)


@router.post("/generate-rca", response_model=RCAOutputModel)
def generate_rca(payload: RCAInputModel):
    markdown_rca = f"""
# Root Cause Analysis

## Issue ID
{payload.task_monitoring_data.issue_logs_id}

## Issue
{payload.task_monitoring_data.title}

## Description
{payload.task_monitoring_data.issue_description}

## Precondition
{payload.task_monitoring_data.pre_condition}

## Expected Result
{payload.task_monitoring_data.expected_result}

## Initial RCA Status
RCA input model is working. AI generation will be connected in Day 2.
"""

    return RCAOutputModel(
        issue_id=payload.task_monitoring_data.issue_logs_id,
        markdown_rca=markdown_rca,
        pdf_file_path=None
    )