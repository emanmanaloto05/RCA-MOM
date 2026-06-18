import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    API_KEY = os.getenv("API_KEY")
    API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")

    API_AUTH_MAP = {
        API_KEY_HEADER: [
            API_KEY
        ]
    }

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini")
    OPENAI_TRANSCRIBE_MODEL = os.getenv(
        "OPENAI_TRANSCRIBE_MODEL",
        "gpt-4o-mini-transcribe"
    )

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    GEMINI_TRANSCRIBE_MODEL = os.getenv(
        "GEMINI_TRANSCRIBE_MODEL",
        "gemini-3.5-flash"
    )

    GEMINI_SUMMARY_MODEL = os.getenv(
        "GEMINI_SUMMARY_MODEL",
        "gemini-2.5-flash-lite"
    )

    GEMINI_MOM_MODEL = os.getenv(
        "GEMINI_MOM_MODEL",
        "gemini-2.5-flash"
    )

    PROJECT_NAME = "RND MOM Generator"
    PROJECT_VERSION = "1.0.0"

    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "rnd-mom-gen"
    )
    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING",
        "true"
    )

settings = Settings()