from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import UploadFile

from api.dependencies import verify_api_key

router = APIRouter(
    prefix="/mom",
    tags=["MOM Generator"]
)


@router.post("/upload")
async def upload_meeting_recording(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    return {
        "message": "Meeting recording uploaded successfully",
        "filename": file.filename,
        "content_type": file.content_type
    }