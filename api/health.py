from typing import Any

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from config.providers import get_gemini_provider
from config.settings import settings

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("/")
async def public_health_check() -> dict[str, str]:
    return {
        "status": "running",
        "message": "RCA Generator API is working",
    }


@router.get("/secure", dependencies=[Depends(verify_api_key)])
async def secure_health_check() -> dict[str, Any]:
    return {
        "status": "running",
        "message": "RCA Generator secure health check passed",
        "app": "RCA Generator",
        "langsmith_enabled": settings.langsmith_enabled,
    }


@router.get("/gemini", dependencies=[Depends(verify_api_key)])
async def gemini_health_check() -> dict[str, Any]:
    provider = get_gemini_provider()
    result = provider.health_check()
    return result.model_dump()


@router.get("/langsmith", dependencies=[Depends(verify_api_key)])
async def langsmith_health_check() -> dict[str, Any]:
    return {
        "tracing": settings.langsmith_tracing,
        "enabled": settings.langsmith_enabled,
        "project": settings.langsmith_project,
    }