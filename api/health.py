from typing import Any

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from config.providers import get_gemini_provider, get_openai_provider, FallbackProvider
from config.settings import settings

# ---------------------------------------------------------------------------
# Router
#
# ALL endpoints are protected — an API key is required for every route,
# including GET /health/. This is enforced via the router-level dependency
# so there is no risk of accidentally adding a public route in future.
#
# Infrastructure tooling (load balancers, Kubernetes, Docker, uptime monitors)
# must supply the API key in the X-API-Key header (or however verify_api_key
# is implemented) when calling the liveness probe.
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/health",
    tags=["Health"],
    dependencies=[Depends(verify_api_key)],   # ← applies to every route below
)


@router.get("/")
async def health_check() -> dict[str, str]:
    """
    Liveness probe. Returns immediately with no external calls.
    Requires a valid API key — configure your infra tooling to supply it.
    """
    return {
        "status": "running",
        "message": "RCA Generator API is working",
    }


@router.get("/secure")
async def secure_health_check() -> dict[str, Any]:
    """
    Authenticated liveness + basic config check.
    Safe to call frequently; makes no LLM requests.
    """
    return {
        "status": "running",
        "message": "RCA Generator secure health check passed",
        "app": "RCA Generator",
        "langsmith_enabled": settings.langsmith_enabled,
    }


@router.get("/gemini")
async def gemini_health_check() -> dict[str, Any]:
    """
    Pings the Gemini API with a short probe prompt and returns latency.
    Status values: healthy | degraded | unavailable
    """
    provider = get_gemini_provider()
    result = provider.health_check()
    return result.model_dump()


@router.get("/openai")
async def openai_health_check() -> dict[str, Any]:
    """
    Pings the OpenAI API with a short probe prompt and returns latency.
    Status values: healthy | degraded | unavailable
    """
    provider = get_openai_provider()
    result = provider.health_check()
    return result.model_dump()


@router.get("/langsmith")
async def langsmith_health_check() -> dict[str, Any]:
    """
    Returns current LangSmith tracing configuration.
    Does not make a live network call to LangSmith.
    """
    return {
        "tracing": settings.langsmith_tracing,
        "enabled": settings.langsmith_enabled,
        "project": settings.langsmith_project,
    }


@router.get("/providers")
async def providers_health_check() -> dict[str, Any]:
    """
    Combined health status for all LLM providers plus fallback usage stats.

    Returns:
        {
            "gemini":  { "status": "healthy",     "latency_ms": 312.4, ... },
            "openai":  { "status": "unavailable", "error": "...",      ... },
            "fallback_usage": {
                "primary_usage_count":  95,
                "fallback_usage_count":  5,
                "total":               100,
                "per_agent": { ... },
            },
        }
    """
    gemini_result = get_gemini_provider().health_check()
    openai_result = get_openai_provider().health_check()

    return {
        "gemini":         gemini_result.model_dump(),
        "openai":         openai_result.model_dump(),
        "fallback_usage": FallbackProvider.get_usage_stats(),
    }