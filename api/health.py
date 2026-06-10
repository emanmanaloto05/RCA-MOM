from fastapi import APIRouter

from config.settings import settings

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("/")
def health_check():
    return {
        "status": "running",
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION
    }