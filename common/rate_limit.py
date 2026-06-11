# common/rate_limit.py

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings


def rate_limit_key_func(request: Request) -> str:
    """
    Uses client IP address as the rate-limit key.
    """
    return get_remote_address(request)


limiter = Limiter(
    key_func=rate_limit_key_func,  # type: ignore[arg-type]
    default_limits=[settings.rate_limit_default],
    headers_enabled=True,
)