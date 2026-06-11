from typing import TypedDict


class MOMState(TypedDict):
    audio_path: str
    transcript_path: str
    html_path: str
    transcript: str
    summary: str
    owners: str
    tasks: str
    blockers: str
    followups: str
    mom: str