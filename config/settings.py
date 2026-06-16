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
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Security
    api_key: str = Field(default="CHANGE_ME")
    api_key_header: str = Field(default="X-API-Key")
    api_key_headers_json: str = Field(default="")

    # Google Gemini
    google_api_key: str = Field(default="CHANGE_ME")
    gemini_default_model: str = Field(default="gemini-2.5-flash")
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_top_k: int = Field(default=40, ge=1, le=100)

    # OpenAI
    openai_api_key: str = Field(default="CHANGE_ME")
    openai_default_model: str = Field(default="gpt-5.5")

    # RCA Agent - Issue Summary
    issue_summary_provider: str = Field(default="gemini")
    issue_summary_model: str = Field(default="gemini-2.5-flash")

    # RCA Agent - Root Cause
    root_cause_provider: str = Field(default="openai")
    root_cause_model: str = Field(default="gpt-5.5-pro")

    # RCA Agent - Impact Analysis
    impact_analysis_provider: str = Field(default="openai")
    impact_analysis_model: str = Field(default="gpt-5.5")

    # RCA Agent - Affected Module
    affected_module_provider: str = Field(default="gemini")
    affected_module_model: str = Field(default="gemini-2.5-flash")

    # RCA Agent - Quality Gate Findings
    quality_gate_provider: str = Field(default="gemini")
    quality_gate_model: str = Field(default="gemini-2.5-flash")

    # RCA Agent - Corrective Action
    corrective_action_provider: str = Field(default="openai")
    corrective_action_model: str = Field(default="gpt-5.5")

    # RCA Agent - Preventive Action
    preventive_action_provider: str = Field(default="openai")
    preventive_action_model: str = Field(default="gpt-5.5")

    # RCA Agent - Owner Review
    owner_review_provider: str = Field(default="gemini")
    owner_review_model: str = Field(default="gemini-2.5-flash")

    # RCA Agent - Final Reviewer
    rca_reviewer_provider: str = Field(default="openai")
    rca_reviewer_model: str = Field(default="gpt-5.5-pro")

    # RCA Agent - Final Assembler
    rca_assembler_provider: str = Field(default="gemini")
    rca_assembler_model: str = Field(default="gemini-2.5-flash")

    # Shared AI Parameters
    ai_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    ai_max_output_tokens: int = Field(default=8192, ge=256, le=32768)
    ai_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    ai_max_retries: int = Field(default=3, ge=1, le=10)
    ai_request_timeout: int = Field(default=60, ge=10, le=300)

    # Backward-compatible Gemini Parameters
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    gemini_max_output_tokens: int = Field(default=8192, ge=256, le=32768)
    gemini_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    gemini_max_retries: int = Field(default=3, ge=1, le=10)
    gemini_request_timeout: int = Field(default=60, ge=10, le=300)

    # OpenAI Parameters
    openai_reasoning_effort: str = Field(default="medium")

    # LangSmith
    langsmith_tracing: bool = Field(default=False)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str = Field(default="HRS AI Implementation")

    # CORS
    allowed_origins: str = Field(
        default="http://127.0.0.1:8000,http://localhost:8000"
    )

    # Rate Limiting
    rate_limit_default: str = Field(default="60/minute")
    rate_limit_rca_generation: str = Field(default="10/minute")

    # Logging
    log_level: str = Field(default="INFO")

    # Validators
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
            "replace_with_your_google_ai_studio_key",
        }

        if value in forbidden_values:
            raise ValueError("GOOGLE_API_KEY cannot use a placeholder value.")

        if len(value) < 20:
            raise ValueError("GOOGLE_API_KEY appears too short to be valid.")

        if " " in value:
            raise ValueError("GOOGLE_API_KEY must not contain spaces.")

        return value

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("OPENAI_API_KEY cannot be empty.")

        forbidden_values = {
            "CHANGE_ME",
            "password123",
            "your_api_key_here",
            "replace_with_openai_api_key",
        }

        if value in forbidden_values:
            raise ValueError("OPENAI_API_KEY cannot use a placeholder value.")

        if " " in value:
            raise ValueError("OPENAI_API_KEY must not contain spaces.")

        return value

    @field_validator(
        "gemini_model",
        "gemini_default_model",
        "issue_summary_model",
        "root_cause_model",
        "impact_analysis_model",
        "affected_module_model",
        "quality_gate_model",
        "corrective_action_model",
        "preventive_action_model",
        "owner_review_model",
        "rca_reviewer_model",
        "rca_assembler_model",
    )
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Model name cannot be empty.")

        return value

    @field_validator(
        "issue_summary_provider",
        "root_cause_provider",
        "impact_analysis_provider",
        "affected_module_provider",
        "quality_gate_provider",
        "corrective_action_provider",
        "preventive_action_provider",
        "owner_review_provider",
        "rca_reviewer_provider",
        "rca_assembler_provider",
    )
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip().lower()

        allowed_providers = {"gemini", "openai"}

        if value not in allowed_providers:
            raise ValueError(
                f"Provider must be one of: {sorted(allowed_providers)}"
            )

        return value

    @field_validator("openai_reasoning_effort")
    @classmethod
    def validate_openai_reasoning_effort(cls, value: str) -> str:
        value = value.strip().lower()

        allowed_values = {
            "minimal",
            "low",
            "medium",
            "high",
        }

        if value not in allowed_values:
            raise ValueError(
                f"OPENAI_REASONING_EFFORT must be one of: {sorted(allowed_values)}"
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

    # Helper Properties
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
            and self.openai_api_key != "CHANGE_ME"
        )

    @property
    def api_auth_map(self) -> dict[str, set[str]]:
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