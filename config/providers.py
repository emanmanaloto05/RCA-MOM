# providers.py
import logging
import re
import time
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Optional, cast

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, SecretStr, field_validator

from config.settings import settings

# Production env vars take priority over .env file
load_dotenv(override=False)

logger = logging.getLogger("rca_generator.providers")

# Enums
class GeminiModel(str, Enum):
    FLASH_15 = "gemini-1.5-flash"
    FLASH_15_8B = "gemini-1.5-flash-8b"
    PRO_15 = "gemini-1.5-pro"
    FLASH_20 = "gemini-2.0-flash"
    FLASH_25 = "gemini-2.5-flash"


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


# Custom exceptions
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
    message = re.sub(r"key=[^&\s]+", "key=[REDACTED]", message)
    return message[:500]

def extract_text(response: Any) -> str:
    """
    Safely extracts plain text from a LangChain response.
    Defined at module level so Pylance can fully resolve its type signature.
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
# Provider config — validated at startup via Pydantic
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


# ---------------------------------------------------------------------------
# Health check result
# ---------------------------------------------------------------------------


class ProviderHealthResult(BaseModel):
    status: ProviderStatus
    model: str
    latency_ms: Optional[float] = None
    checked_at: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core provider class
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

    # ------------------------------------------------------------------
    # LLM access
    # ------------------------------------------------------------------

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Returns the cached LLM instance, building it on first call.
        The API key is only unwrapped here and passed directly to the
        LangChain client — never stored as a plain string.
        """
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def _build_llm(self) -> ChatGoogleGenerativeAI:
        try:
            logger.info(
                "Building Gemini LLM client | model=%s",
                self._config.model.value,
            )
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

    # ------------------------------------------------------------------
    # Invoke
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> str:
        """
        Sends a prompt to Gemini and returns the text response.
        Guards against empty prompts to prevent wasted API quota.
        Sanitizes error messages before raising.
        """
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

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> ProviderHealthResult:
        """
        Sends a minimal probe to Gemini and returns a ProviderHealthResult.
        Never raises — always returns a typed result so the health route
        can call this safely without a try/except wrapper.

        Returns DEGRADED if the model responds but with unexpected content.
        Returns UNAVAILABLE if the call fails entirely.
        """
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
                checked_at=checked_at,
                error=safe_error,
            )

    # ------------------------------------------------------------------
    # Provider metadata — safe, no secrets exposed
    # ------------------------------------------------------------------

    def get_info(self) -> dict[str, Any]:
        """
        Returns provider metadata safe to expose in health endpoints.
        api_key_configured is always True — config validation guarantees
        the key is set before this instance is created.
        """
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
# Singleton factory — one provider per process
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_gemini_provider() -> GeminiProvider:
    """
    Returns the process-level GeminiProvider singleton.

    Config values are read from settings.py (typed Pydantic fields) —
    no bare float()/int() casting. Pydantic handles coercion and raises
    a clear error if any value is invalid.

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
        logger.exception(
            "GeminiProvider configuration failed | error=%s", safe_error
        )
        raise ProviderInitError(
            f"Gemini provider configuration failed: {safe_error}"
        ) from exc


# ---------------------------------------------------------------------------
# Module-level helpers for service layer
# ---------------------------------------------------------------------------


def get_gemini_model() -> ChatGoogleGenerativeAI:
    """
    Returns the raw LangChain LLM instance.
    Use this when passing the LLM directly into a LangChain chain
    or LangGraph node.
    """
    return get_gemini_provider().get_llm()


def invoke_gemini(prompt: str) -> str:
    """
    One-line helper for simple prompt -> text generation.
    Recommended entry point for RCAService.
    """
    return get_gemini_provider().invoke(prompt)


def reset_gemini_provider() -> None:
    """
    Clears the lru_cache so tests can inject a fresh provider.
    Never call this in production code.
    """
    get_gemini_provider.cache_clear()
    logger.warning("GeminiProvider singleton cache cleared — test use only.")