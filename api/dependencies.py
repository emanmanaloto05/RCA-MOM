from __future__ import annotations

import hmac
import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from config.settings import settings

logger = logging.getLogger("rca_generator.security")


_api_key_scheme = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False,
    description=(
        f"Supply a valid API key in the {settings.api_key_header} header."
    ),
)

_api_auth_map = settings.api_auth_map


async def verify_api_key(
    request: Request,
    _swagger_key: str | None = Depends(_api_key_scheme),
) -> str:
    if not _api_auth_map:
        logger.critical("API authentication is not configured.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication is not configured.",
        )

    attempted = False

    for header_name, allowed_keys in _api_auth_map.items():
        provided_key = request.headers.get(header_name)

        if provided_key is None:
            continue

        attempted = True

        for allowed_key in allowed_keys:
            if hmac.compare_digest(provided_key, allowed_key):
                return provided_key

    expected_headers = ", ".join(sorted(_api_auth_map.keys()))

    if not attempted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing API key header. "
                f"Expected one of: {expected_headers}"
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key.",
    )