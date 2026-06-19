from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langsmith import Client
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.dependencies import verify_api_key
from api.health import router as health_router
from api.routes import router as rca_router
from common.rate_limit import limiter
from config.providers import get_gemini_model, get_openai_provider
from config.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("rca_generator")

try:
    langsmith_client = Client()
    logger.info("LangSmith client initialized successfully.")
except Exception as exc:
    langsmith_client = None
    logger.warning("LangSmith client not initialized: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# FIX — section key list kept in sync with agent_root/graph.py
#
# These MUST match the `section_key` strings passed to _invoke_section() in
# graph.py exactly (e.g. "quality_gate_findings", not "quality_gate"), since
# that's what the per-section settings attribute names
# (f"{section_key}_provider") are derived from.
# ─────────────────────────────────────────────────────────────────────────────
_SECTION_KEYS: list[str] = [
    "issue_summary",
    "root_cause",
    "impact_analysis",
    "affected_module",
    "quality_gate_findings",
    "corrective_action",
    "preventive_action",
    "owner_review",
]


def _openai_is_used() -> bool:
    """
    Determines whether any section's resolved provider is OpenAI, so startup
    validation knows whether to initialize the OpenAI client.

    FIX: previously read settings.<section>_provider directly, which raised
    AttributeError whenever a per-section override field wasn't declared on
    the Settings model (Pydantic only allows attribute access for declared
    fields). This now uses getattr(..., None) — the same safe-access pattern
    _invoke_section() in graph.py uses — and falls back to
    settings.primary_provider (then "openai") to mirror graph.py's actual
    provider-resolution chain, so this check stays accurate even when no
    per-section overrides are configured at all.
    """
    per_section_providers = [
        getattr(settings, f"{section_key}_provider", None)
        for section_key in _SECTION_KEYS
    ]

    if any(
        provider is not None and provider.lower() == "openai"
        for provider in per_section_providers
    ):
        return True

    # No per-section override resolved to OpenAI — fall back to the global
    # default, exactly as _invoke_section() does: primary_provider, else "openai".
    fallback_provider = settings.primary_provider or "openai"
    return fallback_provider.lower() == "openai"


def _validate_providers() -> None:
    errors: list[str] = []

    if not settings.google_api_key:
        errors.append("GOOGLE_API_KEY is missing or empty.")
    else:
        try:
            get_gemini_model()
            logger.info(
                "Startup validation: Gemini provider OK | model=%s",
                settings.gemini_model,
            )
        except Exception as exc:
            errors.append(f"Gemini provider init failed: {exc}")

    if _openai_is_used():
        try:
            get_openai_provider()
            logger.info(
                "Startup validation: OpenAI provider OK | model=%s",
                settings.openai_default_model,
            )
        except Exception as exc:
            errors.append(f"OpenAI provider init failed: {exc}")
    else:
        logger.info(
            "Startup validation: OpenAI skipped because no section uses OpenAI."
        )

    if errors:
        raise RuntimeError(
            "Provider startup validation failed:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("RCA Generator API starting up — running provider validation...")

    try:
        _validate_providers()
        logger.info("All provider validations passed. API is ready.")
    except RuntimeError as exc:
        logger.critical(
            "Startup validation failed — server will not start.\n%s",
            exc,
        )
        raise

    yield

    logger.info("RCA Generator API shutting down.")


app = FastAPI(
    title="RCA Generator API",
    description="AI-powered Root Cause Analysis Generator",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded. Please try again later."
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=[
        settings.api_key_header,
        "Content-Type",
        "Authorization",
    ],
)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = str(uuid4())
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request error | request_id=%s | method=%s | path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    duration = round(time.perf_counter() - start_time, 4)
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed | request_id=%s | method=%s | path=%s | status=%s | duration=%ss",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


app.include_router(health_router)
app.include_router(rca_router)


@app.get("/", dependencies=[Depends(verify_api_key)])
async def root() -> dict[str, str]:
    return {
        "message": "RCA Generator API is running"
    }