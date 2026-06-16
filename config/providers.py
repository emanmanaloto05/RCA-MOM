# config/providers.py
import logging
import re
import time
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Optional, Union, cast

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, field_validator

from config.settings import settings

# Production env vars take priority over .env file
load_dotenv(override=False)

logger = logging.getLogger("rca_generator.providers")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GeminiModel(str, Enum):
    FLASH_15    = "gemini-1.5-flash"
    FLASH_15_8B = "gemini-1.5-flash-8b"
    PRO_15      = "gemini-1.5-pro"
    FLASH_20    = "gemini-2.0-flash"
    FLASH_25    = "gemini-2.5-flash"


class ProviderStatus(str, Enum):
    HEALTHY     = "healthy"
    DEGRADED    = "degraded"
    UNAVAILABLE = "unavailable"


class ProviderName(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ProviderInitError(RuntimeError):
    """Raised when the provider fails to initialise."""


class ProviderNotConfiguredError(RuntimeError):
    """Raised when the provider is accessed before initialisation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_error_message(exc: Exception) -> str:
    """
    Strips any API key fragments from exception messages before
    they reach logs or HTTP responses.
    Truncated to 500 chars to prevent oversized log lines.
    """
    message = str(exc)
    message = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "[REDACTED_API_KEY]", message)
    message = re.sub(r"sk-[A-Za-z0-9\-_]{10,}", "[REDACTED_API_KEY]", message)
    message = re.sub(r"key=[^&\s]+", "key=[REDACTED]", message)
    return message[:500]


def extract_text(response: Any) -> str:
    """
    Safely extracts plain text from a LangChain response.
    Handles str, list, and arbitrary objects — robust against
    LangChain response format changes across versions.
    """
    content: Any = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        items = cast(list[Any], content)
        return "\n".join(str(part) for part in items).strip()

    return str(content).strip()


# ---------------------------------------------------------------------------
# Provider configs — validated at startup via Pydantic
# ---------------------------------------------------------------------------

class GeminiProviderConfig(BaseModel):
    api_key: SecretStr
    model: GeminiModel = GeminiModel.FLASH_25
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=8192, ge=256, le=32768)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    max_retries: int = Field(default=3, ge=1, le=5)
    request_timeout: int = Field(default=60, ge=10, le=300)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()

        if not raw:
            raise ValueError("GOOGLE_API_KEY must not be empty.")

        if raw in {"CHANGE_ME", "your_api_key_here", "your_actual_gemini_api_key"}:
            raise ValueError(
                "GOOGLE_API_KEY is still set to a placeholder value. "
                "Set a real key in your .env file."
            )

        if len(raw) < 20:
            raise ValueError(
                "GOOGLE_API_KEY appears too short to be valid. "
                "Check your GOOGLE_API_KEY value."
            )

        if " " in raw:
            raise ValueError("GOOGLE_API_KEY must not contain spaces.")

        return value

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }


class OpenAIProviderConfig(BaseModel):
    api_key: SecretStr
    model: str = Field(default="gpt-5.5")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_completion_tokens: int = Field(default=8192, ge=256, le=32768)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=1, le=5)
    timeout: int = Field(default=60, ge=10, le=300)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()

        if not raw:
            raise ValueError("OPENAI_API_KEY must not be empty.")

        if raw in {"CHANGE_ME", "your_api_key_here", "replace_with_openai_api_key"}:
            raise ValueError(
                "OPENAI_API_KEY is still set to a placeholder value. "
                "Set a real key in your .env file."
            )

        if " " in raw:
            raise ValueError("OPENAI_API_KEY must not contain spaces.")

        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("OpenAI model name cannot be empty.")
        return value

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }


# ---------------------------------------------------------------------------
# Health check result
# ---------------------------------------------------------------------------

class ProviderHealthResult(BaseModel):
    status: ProviderStatus
    model: str
    provider: str
    latency_ms: Optional[float] = None
    checked_at: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM type alias
# ---------------------------------------------------------------------------

LLMClient = Union[ChatGoogleGenerativeAI, ChatOpenAI]


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiProvider:
    """
    Wraps ChatGoogleGenerativeAI with:
    - Pydantic config validation (SecretStr — key never logged or printed)
    - Lazy singleton LLM instance via get_llm()
    - Safe invoke() with empty prompt guard
    - Health check with latency measurement and degraded state detection
    - __repr__ / __str__ never expose the API key
    """

    def __init__(self, config: GeminiProviderConfig) -> None:
        self._config = config
        self._llm: Optional[ChatGoogleGenerativeAI] = None

        logger.info(
            "GeminiProvider initialized | model=%s | temperature=%s | max_output_tokens=%s",
            self._config.model.value,
            self._config.temperature,
            self._config.max_output_tokens,
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def get_config(self) -> GeminiProviderConfig:
        """Returns the provider config. Use this instead of accessing _config directly."""
        return self._config

    def _build_llm(self) -> ChatGoogleGenerativeAI:
        try:
            logger.info("Building Gemini LLM client | model=%s", self._config.model.value)
            return ChatGoogleGenerativeAI(
                model=self._config.model.value,
                google_api_key=self._config.api_key.get_secret_value(),
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
                top_p=self._config.top_p,
                top_k=self._config.top_k,
                max_retries=self._config.max_retries,
                request_timeout=self._config.request_timeout,
                convert_system_message_to_human=True,
            )
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.exception(
                "Failed to build Gemini LLM client | model=%s | error=%s",
                self._config.model.value,
                safe_error,
            )
            raise ProviderInitError(
                f"Could not initialize Gemini provider: {safe_error}"
            ) from exc

    def invoke(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        try:
            response = self.get_llm().invoke(prompt)
            return extract_text(response)
        except ValueError:
            raise
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.exception("Gemini invoke failed | error=%s", safe_error)
            raise RuntimeError(f"Gemini generation failed: {safe_error}") from exc

    def health_check(self) -> ProviderHealthResult:
        checked_at = datetime.now(timezone.utc).isoformat()

        try:
            start = time.perf_counter()
            response = self.get_llm().invoke("Reply with only the word: OK")
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            text = extract_text(response)

            if "OK" not in text.upper():
                logger.warning(
                    "Gemini health check degraded | model=%s | response=%s",
                    self._config.model.value,
                    text[:100],
                )
                return ProviderHealthResult(
                    status=ProviderStatus.DEGRADED,
                    model=self._config.model.value,
                    provider="gemini",
                    latency_ms=latency_ms,
                    checked_at=checked_at,
                    error="Unexpected health check response.",
                )

            logger.info(
                "Gemini health check passed | model=%s | latency_ms=%s",
                self._config.model.value,
                latency_ms,
            )
            return ProviderHealthResult(
                status=ProviderStatus.HEALTHY,
                model=self._config.model.value,
                provider="gemini",
                latency_ms=latency_ms,
                checked_at=checked_at,
            )

        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.warning(
                "Gemini health check failed | model=%s | error=%s",
                self._config.model.value,
                safe_error,
            )
            return ProviderHealthResult(
                status=ProviderStatus.UNAVAILABLE,
                model=self._config.model.value,
                provider="gemini",
                checked_at=checked_at,
                error=safe_error,
            )

    def get_info(self) -> dict[str, Any]:
        return {
            "provider": "google_gemini",
            "model": self._config.model.value,
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "max_retries": self._config.max_retries,
            "request_timeout": self._config.request_timeout,
            "api_key_configured": True,
        }

    def __repr__(self) -> str:
        return (
            "GeminiProvider("
            f"model={self._config.model.value!r}, "
            f"temperature={self._config.temperature}, "
            "api_key='[REDACTED]'"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider:
    """
    Wraps ChatOpenAI with:
    - Pydantic config validation (SecretStr — key never logged or printed)
    - Lazy singleton LLM instance via get_llm()
    - Safe invoke() with empty prompt guard
    - Health check with latency measurement and degraded state detection
    - __repr__ / __str__ never expose the API key
    """

    def __init__(self, config: OpenAIProviderConfig) -> None:
        self._config = config
        self._llm: Optional[ChatOpenAI] = None

        logger.info(
            "OpenAIProvider initialized | model=%s | temperature=%s | max_completion_tokens=%s",
            self._config.model,
            self._config.temperature,
            self._config.max_completion_tokens,
        )

    def get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def get_config(self) -> OpenAIProviderConfig:
        """Returns the provider config. Use this instead of accessing _config directly."""
        return self._config

    def _build_llm(self) -> ChatOpenAI:
        try:
            logger.info("Building OpenAI LLM client | model=%s", self._config.model)
            return ChatOpenAI(
                model=self._config.model,
                api_key=SecretStr(self._config.api_key.get_secret_value()),
                temperature=self._config.temperature,
                max_completion_tokens=self._config.max_completion_tokens,
                top_p=self._config.top_p,
                max_retries=self._config.max_retries,
                timeout=self._config.timeout,
            )
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.exception(
                "Failed to build OpenAI LLM client | model=%s | error=%s",
                self._config.model,
                safe_error,
            )
            raise ProviderInitError(
                f"Could not initialize OpenAI provider: {safe_error}"
            ) from exc

    def invoke(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        try:
            response = self.get_llm().invoke(prompt)
            return extract_text(response)
        except ValueError:
            raise
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.exception("OpenAI invoke failed | error=%s", safe_error)
            raise RuntimeError(f"OpenAI generation failed: {safe_error}") from exc

    def health_check(self) -> ProviderHealthResult:
        checked_at = datetime.now(timezone.utc).isoformat()

        try:
            start = time.perf_counter()
            response = self.get_llm().invoke("Reply with only the word: OK")
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            text = extract_text(response)

            if "OK" not in text.upper():
                logger.warning(
                    "OpenAI health check degraded | model=%s | response=%s",
                    self._config.model,
                    text[:100],
                )
                return ProviderHealthResult(
                    status=ProviderStatus.DEGRADED,
                    model=self._config.model,
                    provider="openai",
                    latency_ms=latency_ms,
                    checked_at=checked_at,
                    error="Unexpected health check response.",
                )

            logger.info(
                "OpenAI health check passed | model=%s | latency_ms=%s",
                self._config.model,
                latency_ms,
            )
            return ProviderHealthResult(
                status=ProviderStatus.HEALTHY,
                model=self._config.model,
                provider="openai",
                latency_ms=latency_ms,
                checked_at=checked_at,
            )

        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.warning(
                "OpenAI health check failed | model=%s | error=%s",
                self._config.model,
                safe_error,
            )
            return ProviderHealthResult(
                status=ProviderStatus.UNAVAILABLE,
                model=self._config.model,
                provider="openai",
                checked_at=checked_at,
                error=safe_error,
            )

    def get_info(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_completion_tokens": self._config.max_completion_tokens,
            "top_p": self._config.top_p,
            "max_retries": self._config.max_retries,
            "timeout": self._config.timeout,
            "api_key_configured": True,
        }

    def __repr__(self) -> str:
        return (
            "OpenAIProvider("
            f"model={self._config.model!r}, "
            f"temperature={self._config.temperature}, "
            "api_key='[REDACTED]'"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# Singleton factories
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_gemini_provider() -> GeminiProvider:
    """
    Returns the process-level GeminiProvider singleton.
    lru_cache(maxsize=1) guarantees a single instance per process.
    Call reset_gemini_provider() in tests to get a fresh instance.
    """
    logger.info("Creating GeminiProvider singleton")

    try:
        config = GeminiProviderConfig(
            api_key=SecretStr(settings.google_api_key),
            model=GeminiModel(settings.gemini_model),
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            top_p=settings.gemini_top_p,
            top_k=settings.gemini_top_k,
            max_retries=settings.gemini_max_retries,
            request_timeout=settings.gemini_request_timeout,
        )
        return GeminiProvider(config)

    except Exception as exc:
        safe_error = _safe_error_message(exc)
        logger.exception("GeminiProvider configuration failed | error=%s", safe_error)
        raise ProviderInitError(
            f"Gemini provider configuration failed: {safe_error}"
        ) from exc


@lru_cache(maxsize=1)
def get_openai_provider() -> OpenAIProvider:
    """
    Returns the process-level OpenAIProvider singleton.
    lru_cache(maxsize=1) guarantees a single instance per process.
    Call reset_openai_provider() in tests to get a fresh instance.
    """
    logger.info("Creating OpenAIProvider singleton")

    try:
        config = OpenAIProviderConfig(
            api_key=SecretStr(settings.openai_api_key),
            model=settings.openai_default_model,
            temperature=settings.ai_temperature,
            max_completion_tokens=settings.ai_max_output_tokens,
            top_p=settings.ai_top_p,
            max_retries=settings.ai_max_retries,
            timeout=settings.ai_request_timeout,
        )
        return OpenAIProvider(config)

    except Exception as exc:
        safe_error = _safe_error_message(exc)
        logger.exception("OpenAIProvider configuration failed | error=%s", safe_error)
        raise ProviderInitError(
            f"OpenAI provider configuration failed: {safe_error}"
        ) from exc


# ---------------------------------------------------------------------------
# Provider router
# Resolves a (provider_name, model_name) pair to the correct LLM client.
# Used by every agent node to honour per-agent provider settings from .env.
# ---------------------------------------------------------------------------

def get_llm_for_agent(provider: str, model: str) -> LLMClient:
    """
    Returns the appropriate LangChain LLM client for the given provider
    and model name, as configured per-agent in .env / settings.py.

    Routing:
        "gemini" → ChatGoogleGenerativeAI (via GeminiProvider singleton)
        "openai" → ChatOpenAI             (via OpenAIProvider singleton)

    The model string overrides the singleton's default model by constructing
    a lightweight client directly — this avoids mutating the cached singleton
    while still honouring per-agent model overrides.

    Args:
        provider: One of "gemini" or "openai" (case-insensitive).
        model:    Model name string, e.g. "gemini-2.5-flash" or "gpt-5.5-pro".

    Returns:
        A ready-to-use LangChain chat model instance.

    Raises:
        ProviderInitError: If provider is unknown or client cannot be built.

    Example:
        >>> llm = get_llm_for_agent(
        ...     settings.root_cause_provider,
        ...     settings.root_cause_model,
        ... )
        >>> chain = prompt | llm
    """
    provider_lower = provider.strip().lower()

    if provider_lower == ProviderName.GEMINI:
        base = get_gemini_provider()
        cfg = base.get_config()
        # If the model matches the singleton's model, reuse it directly.
        if model == cfg.model.value:
            return base.get_llm()
        # Otherwise build a per-agent client with the requested model.
        try:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=SecretStr(cfg.api_key.get_secret_value()),
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_retries=cfg.max_retries,
                request_timeout=cfg.request_timeout,
                convert_system_message_to_human=True,
            )
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            raise ProviderInitError(
                f"Failed to build Gemini client for model '{model}': {safe_error}"
            ) from exc

    if provider_lower == ProviderName.OPENAI:
        base_oai = get_openai_provider()
        cfg_oai = base_oai.get_config()
        # If the model matches the singleton's model, reuse it directly.
        if model == cfg_oai.model:
            return base_oai.get_llm()
        # Otherwise build a per-agent client with the requested model.
        try:
            return ChatOpenAI(
                model=model,
                api_key=SecretStr(cfg_oai.api_key.get_secret_value()),
                temperature=cfg_oai.temperature,
                max_completion_tokens=cfg_oai.max_completion_tokens,
                top_p=cfg_oai.top_p,
                max_retries=cfg_oai.max_retries,
                timeout=cfg_oai.timeout,
            )
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            raise ProviderInitError(
                f"Failed to build OpenAI client for model '{model}': {safe_error}"
            ) from exc

    raise ProviderInitError(
        f"Unknown provider '{provider}'. Supported values: 'gemini', 'openai'."
    )


# ---------------------------------------------------------------------------
# Module-level helpers for service layer
# ---------------------------------------------------------------------------

def get_gemini_model() -> ChatGoogleGenerativeAI:
    """
    Returns the raw LangChain LLM instance for Gemini.
    Use this when passing the LLM directly into a LangChain chain
    or LangGraph node that doesn't need per-agent model overrides.
    """
    return get_gemini_provider().get_llm()


def invoke_gemini(prompt: str) -> str:
    """
    One-line helper for simple prompt → text generation via Gemini.
    Recommended entry point for RCAService when Gemini is the sole provider.
    """
    return get_gemini_provider().invoke(prompt)


def invoke_openai(prompt: str) -> str:
    """
    One-line helper for simple prompt → text generation via OpenAI.
    """
    return get_openai_provider().invoke(prompt)


def reset_gemini_provider() -> None:
    """
    Clears the lru_cache so tests can inject a fresh Gemini provider.
    Never call this in production code.
    """
    get_gemini_provider.cache_clear()
    logger.warning("GeminiProvider singleton cache cleared — test use only.")


def reset_openai_provider() -> None:
    """
    Clears the lru_cache so tests can inject a fresh OpenAI provider.
    Never call this in production code.
    """
    get_openai_provider.cache_clear()
    logger.warning("OpenAIProvider singleton cache cleared — test use only.")


def reset_all_providers() -> None:
    """
    Clears all provider singleton caches.
    Convenience helper for test teardown.
    """
    reset_gemini_provider()
    reset_openai_provider()