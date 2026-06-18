from typing import TypedDict
from pydantic import BaseModel, Field
from fastapi import UploadFile, File, Form

class MOMState(TypedDict):
    audio_path: str

    transcript_path: str
    summary_path: str

    html_path: str
    pdf_path: str
    docx_path: str

    generate_docx: bool

    transcript: str
    structured_summary: str

    subject: str
    attendees: str
    agenda: str

    summary: str

    decisions_made: str
    tasks: str
    owners: str

    blockers: str
    followups: str

    meeting_outcome: str

    mom: str
    supervisor_review: str
    
class MOMInput(BaseModel):
    audio_file: UploadFile = File(...)
    generate_docx: bool = Form(False)


class SummaryOutput(BaseModel):
    subject_title: str
    attendees: list[str]
    owners: list[str]
    detailed_summary: str


class MOMOutput(BaseModel):
    subject: str
    attendees: str
    agenda: str

    executive_summary: str
    discussion_points: str

    decisions_made: str

    action_items: str

    owners: str

    blockers: str

    followups: str

    meeting_outcome: str


class GeneratedDocuments(BaseModel):
    transcript_path: str

    pdf_path: str

    docx_path: str | None = None


class DownloadDocument(BaseModel):
    filename: str
    document_type: str
    path: str


class GenerateMOMResponse(BaseModel):
    message: str

    transcript_path: str

    pdf_path: str

    docx_path: str | None = None

    downloads: list[str]

    documents: list[DownloadDocument] = []

    supervisor_review: str | None = None