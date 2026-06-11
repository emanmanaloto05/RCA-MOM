from fastapi import HTTPException
from fastapi import Security
from fastapi import status
from fastapi.security import APIKeyHeader

from config.settings import settings


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str = Security(
        api_key_header
    )
):
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

    return api_key