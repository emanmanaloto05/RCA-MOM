from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from .env and environment variables."""

    # API security settings
    api_key: str = Field(
        default="CHANGE_ME",
        description="API key required in X-API-Key header.",
    )

    # Google Gemini settings
    google_api_key: str = Field(
        default="CHANGE_ME",
        description="Google AI Studio API key.",
    )

    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name.",
    )

    gemini_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Gemini temperature.",
    )

    gemini_max_output_tokens: int = Field(
        default=8192,
        ge=256,
        le=32768,
        description="Maximum output tokens.",
    )

    gemini_top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p sampling value.",
    )

    gemini_top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Top-k sampling value.",
    )

    gemini_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum Gemini retry count.",
    )

    gemini_request_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Gemini timeout in seconds.",
    )

    # LangSmith tracing settings
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing.",
    )

    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith endpoint.",
    )

    langsmith_api_key: str | None = Field(
        default=None,
        description="LangSmith API key.",
    )

    langsmith_project: str = Field(
        default="HRS AI Implementation",
        description="LangSmith project name.",
    )

    # CORS settings
    allowed_origins: str = Field(
        default="http://127.0.0.1:8000,http://localhost:8000",
        description="Comma-separated allowed origins.",
    )

    # Rate limiting settings
    rate_limit_default: str = Field(
        default="60/minute",
        description="Default rate limit.",
    )

    rate_limit_rca_generation: str = Field(
        default="10/minute",
        description="RCA endpoint rate limit.",
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Application log level.",
    )

    # API key validation
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("API_KEY cannot be empty.")

        if value in {"CHANGE_ME", "password123"}:
            raise ValueError(
                "API_KEY cannot use placeholder values."
            )

        return value

    # Google API key validation
    @field_validator("google_api_key")
    @classmethod
    def validate_google_api_key(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("GOOGLE_API_KEY cannot be empty.")

        if value in {"CHANGE_ME", "password123"}:
            raise ValueError(
                "GOOGLE_API_KEY cannot use placeholder values."
            )

        return value

    # Gemini model validation
    @field_validator("gemini_model")
    @classmethod
    def validate_gemini_model(cls, value: str) -> str:
        allowed_models = {
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-flash",
        }

        if value not in allowed_models:
            raise ValueError(
                f"Invalid GEMINI_MODEL '{value}'. "
                f"Allowed values: {sorted(allowed_models)}"
            )

        return value

    # LangSmith key validation
    @field_validator("langsmith_api_key")
    @classmethod
    def validate_langsmith_api_key(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value if value else None

    # Allowed origins validation
    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "ALLOWED_ORIGINS cannot be empty."
            )

        return value

    # Log level validation
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        value = value.upper()

        if value not in allowed_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of: {sorted(allowed_levels)}"
            )

        return value

    # Parsed CORS origins helper
    @property
    def allowed_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    # LangSmith enabled helper
    @property
    def langsmith_enabled(self) -> bool:
        return (
            self.langsmith_tracing
            and self.langsmith_api_key is not None
        )

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()