# api/routes.py
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from agent_root.models import RCAInputModel, RCAOutputModel
from agent_root.service import RCAService, RCAServiceError
from api.dependencies import verify_api_key
from common.rate_limit import limiter
from config.settings import settings

logger = logging.getLogger("rca_generator.routes")

router = APIRouter(
    prefix="/api",
    tags=["RCA"],
)


@router.post(
    "/generate-rca",
    response_model=RCAOutputModel,
    dependencies=[Depends(verify_api_key)],
    status_code=status.HTTP_200_OK,
)
@limiter.limit(settings.rate_limit_rca_generation)  # type: ignore[misc]
async def generate_rca(
    request: Request,
    response: Response,
    payload: RCAInputModel,
) -> RCAOutputModel:
    issue_id = payload.task_monitoring_data.issue_logs_id

    logger.info(
        "RCA generation request received | issue_id=%s | client=%s",
        issue_id,
        request.client.host if request.client else "unknown",
    )

    try:
        return await RCAService.generate_rca(payload)

    except RCAServiceError as exc:
        logger.exception(
            "RCA generation failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate RCA. Please check server logs.",
        ) from exc


@router.get(
    "/rca/{issue_id}/download",
    dependencies=[Depends(verify_api_key)],
    status_code=status.HTTP_200_OK,
)
async def download_rca_pdf(issue_id: str) -> FileResponse:
    pdf_path = Path("outputs") / f"{issue_id}_rca.pdf"

    if not pdf_path.exists() or not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF not found for issue {issue_id}.",
        )

    logger.info(
        "RCA PDF download requested | issue_id=%s | path=%s",
        issue_id,
        pdf_path,
    )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{issue_id}_rca.pdf",
    )