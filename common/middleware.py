import time

from fastapi import Request

from common.logging import logger


async def process_time_middleware(
    request: Request,
    call_next
):
    start_time = time.time()

    response = await call_next(request)

    process_time = (
        time.time() - start_time
    )

    logger.info(
        f"{request.method} "
        f"{request.url.path} "
        f"took {process_time:.2f}s"
    )

    return response