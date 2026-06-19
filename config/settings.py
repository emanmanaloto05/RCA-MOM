from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, cast

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    allow_degraded_rca: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Security
    api_key: str = Field(default="dev-api-key")
    api_key_header: str = Field(default="X-API-Key")
    api_key_headers_json: str = Field(default="")

    # Google Gemini
    google_api_key: str = Field(default="")
    gemini_default_model: str = Field(default="gemini-2.5-flash")
    gemini_model: str = Field(default="gemini-2.5-flash")

    # OpenAI
    openai_api_key: str = Field(default="")
    openai_default_model: str = Field(default="gpt-5.5")
    openai_enabled: bool = Field(default=False)

    # Global Provider Routing
    primary_provider: str = Field(default="gemini")
    primary_model: str = Field(default="gemini-2.5-flash")
    fallback_provider: str = Field(default="gemini")
    fallback_model: str = Field(default="gemini-2.0-flash")

    # ── RCA Combined Section Agents (4-step flow) ─────────────────────────────
    # Each key corresponds to one combined node in graph.py and one prompt key
    # in prompts.yaml. The section_key passed to _invoke_section_with_retry()
    # matches these names exactly:
    #   incident_analysis  → graph.generate_incident_analysis()
    #   technical_impact   → graph.generate_technical_impact()
    #   qa_resolution      → graph.generate_qa_resolution()
    #   prevention_review  → graph.generate_prevention_review()

    incident_analysis_provider: str = Field(default="gemini")
    incident_analysis_model: str = Field(default="gemini-2.5-flash")

    technical_impact_provider: str = Field(default="gemini")
    technical_impact_model: str = Field(default="gemini-2.5-flash")

    qa_resolution_provider: str = Field(default="gemini")
    qa_resolution_model: str = Field(default="gemini-2.5-flash")

    prevention_review_provider: str = Field(default="gemini")
    prevention_review_model: str = Field(default="gemini-2.5-flash")

    # ── Supervisor / Assembler Agents (unchanged) ─────────────────────────────
    rca_reviewer_provider: str = Field(default="gemini")
    rca_reviewer_model: str = Field(default="gemini-2.5-flash")

    rca_assembler_provider: str = Field(default="gemini")
    rca_assembler_model: str = Field(default="gemini-2.5-flash")

    # Shared AI Parameters
    ai_temperature: float = Field(default=0.2)
    ai_max_output_tokens: int = Field(default=8192)
    ai_top_p: float = Field(default=0.95)
    ai_max_retries: int = Field(default=3)
    ai_request_timeout: int = Field(default=60)

    # Gemini Parameters
    gemini_temperature: float = Field(default=0.2)
    gemini_max_output_tokens: int = Field(default=8192)
    gemini_top_p: float = Field(default=0.95)
    gemini_top_k: int = Field(default=40)
    gemini_max_retries: int = Field(default=3)
    gemini_request_timeout: int = Field(default=60)

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

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        value = value.strip()
        return value or "dev-api-key"

    @field_validator("api_key_header")
    @classmethod
    def validate_api_key_header(cls, value: str) -> str:
        value = value.strip()
        return value or "X-API-Key"

    @field_validator("google_api_key")
    @classmethod
    def validate_google_api_key(cls, value: str) -> str:
        return value.strip()

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_api_key(cls, value: str) -> str:
        return value.strip()

    @field_validator(
        "primary_provider",
        "fallback_provider",
        "incident_analysis_provider",
        "technical_impact_provider",
        "qa_resolution_provider",
        "prevention_review_provider",
        "rca_reviewer_provider",
        "rca_assembler_provider",
    )
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"gemini", "openai"}:
            return "gemini"
        return value

    @field_validator(
        "gemini_default_model",
        "gemini_model",
        "openai_default_model",
        "primary_model",
        "fallback_model",
        "incident_analysis_model",
        "technical_impact_model",
        "qa_resolution_model",
        "prevention_review_model",
        "rca_reviewer_model",
        "rca_assembler_model",
    )
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()
        return value or "gemini-2.5-flash"

    @field_validator("openai_reasoning_effort")
    @classmethod
    def validate_openai_reasoning_effort(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"minimal", "low", "medium", "high"}:
            return "medium"
        return value

    @field_validator("langsmith_api_key")
    @classmethod
    def validate_langsmith_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, value: str) -> str:
        value = value.strip()
        return value or "http://127.0.0.1:8000,http://localhost:8000"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            return "INFO"
        return value

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
        return bool(self.api_key and self.google_api_key)

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
        except json.JSONDecodeError:
            return mapping

        if not isinstance(loaded, dict):
            return mapping

        parsed: dict[str, Any] = cast(dict[str, Any], loaded)

        for header, values in parsed.items():
            header_name = str(header).strip()

            if not header_name:
                continue

            if isinstance(values, str):
                cleaned = values.strip()
                if cleaned:
                    mapping.setdefault(header_name, set()).add(cleaned)

            elif isinstance(values, list):
                items: list[Any] = cast(list[Any], values)
                for item in items:
                    cleaned = str(item).strip()
                    if cleaned:
                        mapping.setdefault(header_name, set()).add(cleaned)

        return mapping

    @property
    def effective_provider_and_model(self) -> tuple[str, str]:
        if self.openai_enabled and self.primary_provider == "openai":
            return self.primary_provider, self.primary_model
        return self.fallback_provider, self.fallback_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()