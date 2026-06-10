#health.py
from fastapi import APIRouter, Depends
from api.dependencies import verify_api_key

router = APIRouter(
    prefix="/health",
    tags=["Health"],
    dependencies=[Depends(verify_api_key)],
)

@router.get("/")
async def health_check():
    return {
        "status": "running",
        "message": "RCA Generator API is working",
    }