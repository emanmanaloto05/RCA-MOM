# routes.py
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agent_root.models import RCAInputModel, RCAOutputModel
from agent_root.service import RCAService, RCAServiceError
from api.dependencies import verify_api_key
from common.rate_limit import limiter

logger = logging.getLogger("rca_generator.routes")

router = APIRouter(
    prefix="/api",
    tags=["RCA"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/generate-rca", response_model=RCAOutputModel)
@limiter.limit("10/minute")  # type: ignore[misc]
async def generate_rca(
    request: Request,
    payload: RCAInputModel,
) -> RCAOutputModel:
    issue_id = payload.task_monitoring_data.issue_logs_id

    logger.info("RCA generation request received | issue_id=%s", issue_id)

    try:
        return await RCAService.generate_rca(payload)

    except RCAServiceError as exc:
        logger.exception("RCA generation failed | issue_id=%s", issue_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate RCA. Please check server logs.",
        ) from exc