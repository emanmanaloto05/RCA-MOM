# dependencies.py
from fastapi import Header, HTTPException, status

from config.settings import settings


def verify_api_key(x_api_key: str = Header(...)):
    if not settings.api_key or settings.api_key == "CHANGE_ME":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return True