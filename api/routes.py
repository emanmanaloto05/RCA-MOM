from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import UploadFile

from api.dependencies import api_key_dependency

router = APIRouter(
    prefix="/mom",
    tags=["MOM Generator"]
)


@router.post("/upload")
async def upload_meeting_recording(
    file: UploadFile = File(...),
    _: None = Depends(api_key_dependency)
):
    return {
        "message": "Meeting recording uploaded successfully",
        "filename": file.filename,
        "content_type": file.content_type
    }