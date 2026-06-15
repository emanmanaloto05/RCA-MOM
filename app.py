from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
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

app = FastAPI(
    title="RCA Generator API",
    description="AI-powered Root Cause Analysis Generator",
    version="1.0.0",
)

# Rate limiting
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


# CORS
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