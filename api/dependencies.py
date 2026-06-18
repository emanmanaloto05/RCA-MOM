import hmac

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from config.settings import settings


api_key_scheme = APIKeyHeader(
    name=settings.API_KEY_HEADER,
    auto_error=False,
    description="Enter your API key"
)


async def verify_api_key(
    request: Request,
    swagger_key: str | None = Depends(api_key_scheme)
):
    if not settings.API_AUTH_MAP:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication is not configured."
        )

    for header_name, allowed_keys in settings.API_AUTH_MAP.items():
        provided_key = request.headers.get(header_name)

        if not provided_key:
            provided_key = swagger_key

        if provided_key:
            for allowed_key in allowed_keys:
                print("Header Name:", header_name)
                print("Provided Key:", provided_key)
                print("Allowed Keys:", allowed_keys)
                if allowed_key and hmac.compare_digest(
                    provided_key,
                    allowed_key
                ):
                    return provided_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key."
    )