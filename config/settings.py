from dotenv import load_dotenv
import os

load_dotenv()


class Settings:

    PROJECT_NAME = "RND MOM Generator"

    PROJECT_VERSION = "1.0.0"

    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY"
    )

    LANGSMITH_API_KEY = os.getenv(
        "LANGSMITH_API_KEY"
    )

    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "rnd-mom-gen"
    )

    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING",
        "true"
    )


settings = Settings()