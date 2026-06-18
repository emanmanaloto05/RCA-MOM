from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from agent_1_sum.graph import mom_graph
from api.dependencies import verify_api_key
from agent_1_sum.models import GenerateMOMResponse, DownloadDocument


router = APIRouter()

TRANSCRIPT_DIR = Path("agent_1_sum/output/transcripts")
PDF_DIR = Path("agent_1_sum/output/pdf")
DOCX_DIR = Path("agent_1_sum/output/docx")


@router.get("/download/txt/{filename}")
async def download_txt(
    filename: str,
    api_key: str = Depends(verify_api_key)
):
    file_path = TRANSCRIPT_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"TXT file not found: {file_path}"
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/plain"
    )


@router.get("/download/pdf/{filename}")
async def download_pdf(
    filename: str,
    api_key: str = Depends(verify_api_key)
):
    file_path = PDF_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {file_path}"
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )


@router.get("/download/docx/{filename}")
async def download_docx(
    filename: str,
    api_key: str = Depends(verify_api_key)
):
    file_path = DOCX_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"DOCX file not found: {file_path}"
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


@router.post(
    "/generate-mom",
    response_model=GenerateMOMResponse,
    dependencies=[Depends(verify_api_key)]
)
async def generate_mom(
    audio_file: UploadFile = File(...),
    generate_docx: bool = Form(False)
):
    upload_dir = Path("uploads")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    audio_path = upload_dir / audio_file.filename

    contents = await audio_file.read()

    audio_path.write_bytes(
        contents
    )

    state = {
        "audio_path": str(audio_path),

        "transcript_path": "",
        "summary_path": "",

        "html_path": "",
        "pdf_path": "",
        "docx_path": "",

        "generate_docx": generate_docx,

        "transcript": "",
        "structured_summary": "",

        "subject": "",
        "attendees": "",
        "agenda": "",

        "summary": "",
        "decisions_made": "",
        "tasks": "",
        "owners": "",
        "blockers": "",
        "followups": "",
        "meeting_outcome": "",

        "supervisor_review": "",

        "mom": ""
    }

    result = mom_graph.invoke(
        state
    )

    downloads = [
        Path(result["transcript_path"]).name,
        Path(result["pdf_path"]).name,
    ]

    if result.get("docx_path"):
        downloads.append(
            Path(result["docx_path"]).name
        )

    documents = [
        DownloadDocument(
            filename=Path(result["transcript_path"]).name,
            document_type="txt",
            path=result["transcript_path"]
        ),

        DownloadDocument(
            filename=Path(result["pdf_path"]).name,
            document_type="pdf",
            path=result["pdf_path"]
        )
    ]

    if result.get("docx_path"):
        documents.append(
            DownloadDocument(
                filename=Path(result["docx_path"]).name,
                document_type="docx",
                path=result["docx_path"]
            )
        )

    return GenerateMOMResponse(
        message="MOM generated successfully",

        transcript_path=result["transcript_path"],

        pdf_path=result["pdf_path"],

        docx_path=result.get("docx_path"),

        downloads=downloads,

        documents=documents,

        supervisor_review=result.get("supervisor_review")
    )