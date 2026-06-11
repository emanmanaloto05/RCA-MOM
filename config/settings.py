# config/settings.py
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, cast

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from .env and environment variables.

    Note:
        This file should only contain configuration and validation.
        FastAPI dependencies such as verify_api_key must stay in api/dependencies.py.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # API SECURITY
    # =========================================================================

    api_key: str = Field(
        default="CHANGE_ME",
        description="Primary API key. Must be at least 32 characters.",
    )

    api_key_header: str = Field(
        default="X-API-Key",
        description="HTTP header name used to supply the API key.",
    )

    api_key_headers_json: str = Field(
        default="",
        description=(
            "Optional JSON mapping of additional API key headers. "
            'Example: {"X-CLIENT-A-KEY": ["key1"], "X-CLIENT-B-KEY": "key2"}'
        ),
    )

    # =========================================================================
    # GOOGLE GEMINI
    # =========================================================================

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
        description="Gemini sampling temperature.",
    )

    gemini_max_output_tokens: int = Field(
        default=8192,
        ge=256,
        le=32768,
        description="Maximum Gemini output tokens.",
    )

    gemini_top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Gemini top-p value.",
    )

    gemini_top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Gemini top-k value.",
    )

    gemini_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum Gemini retry attempts.",
    )

    gemini_request_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Gemini request timeout in seconds.",
    )

    # =========================================================================
    # LANGSMITH
    # =========================================================================

    langsmith_tracing: bool = Field(
        default=False,
        description="Enable or disable LangSmith tracing.",
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
        default="RCA Generator",
        description="LangSmith project name.",
    )

    # =========================================================================
    # CORS
    # =========================================================================

    allowed_origins: str = Field(
        default="http://127.0.0.1:8000,http://localhost:8000",
        description="Comma-separated allowed CORS origins.",
    )

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    rate_limit_default: str = Field(
        default="60/minute",
        description="Default API rate limit.",
    )

    rate_limit_rca_generation: str = Field(
        default="10/minute",
        description="RCA generation endpoint rate limit.",
    )

    # =========================================================================
    # LOGGING
    # =========================================================================

    log_level: str = Field(
        default="INFO",
        description="Application log level.",
    )

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("API_KEY cannot be empty.")

        forbidden_values = {
            "CHANGE_ME",
            "password123",
            "admin",
            "test",
            "apikey",
            "secret",
        }

        if value.lower() in {item.lower() for item in forbidden_values}:
            raise ValueError("API_KEY cannot use a placeholder or weak value.")

        if len(value) < 32:
            raise ValueError("API_KEY must be at least 32 characters long.")

        return value

    @field_validator("api_key_header")
    @classmethod
    def validate_api_key_header(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("API_KEY_HEADER cannot be empty.")

        return value

    @field_validator("google_api_key")
    @classmethod
    def validate_google_api_key(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("GOOGLE_API_KEY cannot be empty.")

        forbidden_values = {
            "CHANGE_ME",
            "password123",
            "your_api_key_here",
            "your_actual_gemini_api_key",
        }

        if value in forbidden_values:
            raise ValueError("GOOGLE_API_KEY cannot use a placeholder value.")

        if len(value) < 20:
            raise ValueError("GOOGLE_API_KEY appears too short to be valid.")

        if " " in value:
            raise ValueError("GOOGLE_API_KEY must not contain spaces.")

        return value

    @field_validator("gemini_model")
    @classmethod
    def validate_gemini_model(cls, value: str) -> str:
        value = value.strip()

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

    @field_validator("langsmith_api_key")
    @classmethod
    def validate_langsmith_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()
        return value if value else None

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("ALLOWED_ORIGINS cannot be empty.")

        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.strip().upper()

        allowed_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if value not in allowed_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of: {sorted(allowed_levels)}"
            )

        return value

    # =========================================================================
    # HELPER PROPERTIES
    # =========================================================================

    @property
    def allowed_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing and self.langsmith_api_key is not None

    @property
    def is_production_ready(self) -> bool:
        return (
            self.api_key != "CHANGE_ME"
            and self.google_api_key != "CHANGE_ME"
        )

    @property
    def api_auth_map(self) -> dict[str, set[str]]:
        """
        Builds header-to-allowed-keys mapping.

        Example output:
            {
                "X-API-Key": {"main-key"},
                "X-CLIENT-A-KEY": {"client-a-key-1", "client-a-key-2"},
            }
        """

        mapping: dict[str, set[str]] = {
            self.api_key_header.strip(): {self.api_key}
        }

        raw = self.api_key_headers_json.strip()

        if not raw:
            return mapping

        try:
            loaded: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"API_KEY_HEADERS_JSON contains invalid JSON: {exc}"
            ) from exc

        if not isinstance(loaded, dict):
            raise ValueError("API_KEY_HEADERS_JSON must be a JSON object.")

        parsed: dict[str, Any] = cast(dict[str, Any], loaded)

        for header, values in parsed.items():
            header_name = str(header).strip()

            if not header_name:
                continue

            key_set: set[str] = set()

            if isinstance(values, str):
                cleaned_value = values.strip()

                if cleaned_value:
                    key_set.add(cleaned_value)

            elif isinstance(values, list):
                items: list[Any] = cast(list[Any], values)

                key_set = {
                    str(item).strip()
                    for item in items
                    if str(item).strip()
                }

            else:
                raise ValueError(
                    "Each API_KEY_HEADERS_JSON value must be a string "
                    "or list of strings."
                )

            existing_keys = mapping.get(header_name, set())
            mapping[header_name] = existing_keys | key_set

        return mapping


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()