# config/providers.py
import logging
import random
import re
import time
from collections import defaultdict
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


class FallbackActivatedError(RuntimeError):
    """Raised when the primary provider failed and fallback was used."""


class SectionGenerationFailedError(RuntimeError):
    """
    Raised when both primary and fallback providers fail for a section,
    even after exponential backoff retries on the primary.

    Contains the section name so the caller can mark it for manual review
    instead of crashing the whole RCA pipeline (Missing #4: Dual Failure Recovery).
    """

    def __init__(self, section: str, primary_error: str, fallback_error: str) -> None:
        self.section        = section
        self.primary_error  = primary_error
        self.fallback_error = fallback_error
        super().__init__(
            f"Both providers failed for section '{section}'. "
            f"Primary: {primary_error}. Fallback: {fallback_error}."
        )


# ---------------------------------------------------------------------------
# Exponential backoff configuration
#
# Applied ONLY inside FallbackProvider.invoke() — the single correct place
# for retry/backoff/failover logic.  graph.py, service.py, routes.py, and
# chains.py must NOT implement their own retry loops.
#
# Strategy (with jitter to prevent thundering-herd):
#   Attempt 1 → fail → wait  2 s  + jitter(0–1 s) → retry
#   Attempt 2 → fail → wait  4 s  + jitter(0–1 s) → retry
#   Attempt 3 → fail → wait  8 s  + jitter(0–1 s) → retry
#   All attempts exhausted → switch to fallback provider
#
# Formula:  delay = (2 ** attempt) + random.uniform(0, 1)
#   attempt=1 → ~2.0–3.0 s
#   attempt=2 → ~4.0–5.0 s
#   attempt=3 → ~8.0–9.0 s
#
# Why backoff matters:
#   Gemini 429 RESOURCE_EXHAUSTED and 503 UNAVAILABLE errors are transient —
#   the service typically recovers within a few seconds.  Without backoff,
#   rapid retries hammer the API and immediately hit the block again.
#   With backoff, we give the provider time to recover before the next attempt.
# ---------------------------------------------------------------------------

# Base delays in seconds for attempts 1, 2, 3
_BACKOFF_BASE_DELAYS: list[int] = [2, 4, 8]

# Jitter range added to each base delay: random.uniform(0, _BACKOFF_JITTER_MAX)
_BACKOFF_JITTER_MAX: float = 1.0


def _backoff_delay(attempt: int) -> float:
    """
    Returns the wait time for the given 1-based retry attempt.

    Formula: (2 ** attempt) + random.uniform(0, 1)

    Args:
        attempt: 1-based attempt number (1, 2, 3, ...).

    Returns:
        Seconds to sleep before the next retry.
    """
    base  = 2 ** attempt                              # 2, 4, 8, 16, …
    jitter = random.uniform(0, _BACKOFF_JITTER_MAX)  # 0.0 – 1.0
    return base + jitter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_error_message(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "[REDACTED_API_KEY]", message)
    message = re.sub(r"sk-[A-Za-z0-9\-_]{10,}", "[REDACTED_API_KEY]", message)
    message = re.sub(r"key=[^&\s]+", "key=[REDACTED]", message)
    return message[:500]


def extract_text(response: Any) -> str:
    content: Any = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        items = cast(list[Any], content)
        return "\n".join(str(part) for part in items).strip()
    return str(content).strip()


# ---------------------------------------------------------------------------
# Provider configs
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
            raise ValueError("GOOGLE_API_KEY is still set to a placeholder value.")
        if len(raw) < 20:
            raise ValueError("GOOGLE_API_KEY appears too short to be valid.")
        if " " in raw:
            raise ValueError("GOOGLE_API_KEY must not contain spaces.")
        return value

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


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
            raise ValueError("OPENAI_API_KEY is still set to a placeholder value.")
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

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


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


LLMClient = Union[ChatGoogleGenerativeAI, ChatOpenAI]


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

class GeminiProvider:
    def __init__(self, config: GeminiProviderConfig) -> None:
        self._config = config
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        logger.info(
            "GeminiProvider initialized | model=%s | temperature=%s | max_output_tokens=%s",
            self._config.model.value, self._config.temperature, self._config.max_output_tokens,
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def get_config(self) -> GeminiProviderConfig:
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
                self._config.model.value, safe_error,
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
                    self._config.model.value, text[:100],
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
                self._config.model.value, latency_ms,
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
                self._config.model.value, safe_error,
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
            "provider":           "google_gemini",
            "model":              self._config.model.value,
            "temperature":        self._config.temperature,
            "max_output_tokens":  self._config.max_output_tokens,
            "top_p":              self._config.top_p,
            "top_k":              self._config.top_k,
            "max_retries":        self._config.max_retries,
            "request_timeout":    self._config.request_timeout,
            "api_key_configured": True,
        }

    def __repr__(self) -> str:
        return (
            f"GeminiProvider(model={self._config.model.value!r}, "
            f"temperature={self._config.temperature}, api_key='[REDACTED]')"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class OpenAIProvider:
    def __init__(self, config: OpenAIProviderConfig) -> None:
        self._config = config
        self._llm: Optional[ChatOpenAI] = None
        logger.info(
            "OpenAIProvider initialized | model=%s | temperature=%s | max_completion_tokens=%s",
            self._config.model, self._config.temperature, self._config.max_completion_tokens,
        )

    def get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def get_config(self) -> OpenAIProviderConfig:
        return self._config

    def _build_llm(self) -> ChatOpenAI:
        try:
            logger.info("Building OpenAI LLM client | model=%s", self._config.model)
            return ChatOpenAI(
                model=self._config.model,                              # was: model_name
                api_key=SecretStr(self._config.api_key.get_secret_value()),  # was: openai_api_key
                temperature=self._config.temperature,
                max_completion_tokens=self._config.max_completion_tokens,  # was: max_tokens
                model_kwargs={"top_p": self._config.top_p},
                max_retries=self._config.max_retries,
                timeout=self._config.timeout,                          # was: request_timeout
            )
        except Exception as exc:
            safe_error = _safe_error_message(exc)
            logger.exception(
                "Failed to build OpenAI LLM client | model=%s | error=%s",
                self._config.model, safe_error,
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
                    self._config.model, text[:100],
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
                self._config.model, latency_ms,
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
                self._config.model, safe_error,
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
            "provider":               "openai",
            "model":                  self._config.model,
            "temperature":            self._config.temperature,
            "max_completion_tokens":  self._config.max_completion_tokens,
            "top_p":                  self._config.top_p,
            "max_retries":            self._config.max_retries,
            "timeout":                self._config.timeout,
            "api_key_configured":     True,
        }

    def __repr__(self) -> str:
        return (
            f"OpenAIProvider(model={self._config.model!r}, "
            f"temperature={self._config.temperature}, api_key='[REDACTED]')"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# Singleton factories
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_gemini_provider() -> GeminiProvider:
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
# ---------------------------------------------------------------------------

def get_llm_for_agent(provider: str, model: str) -> LLMClient:
    provider_lower = provider.strip().lower()

    if provider_lower == ProviderName.GEMINI:
        base = get_gemini_provider()
        cfg = base.get_config()
        if model == cfg.model.value:
            return base.get_llm()
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
        if model == cfg_oai.model:
            return base_oai.get_llm()
        try:
            return ChatOpenAI(
                model=model,                                           # was: model_name
                api_key=SecretStr(cfg_oai.api_key.get_secret_value()),  # was: openai_api_key
                temperature=cfg_oai.temperature,
                max_completion_tokens=cfg_oai.max_completion_tokens,  # was: max_tokens
                model_kwargs={"top_p": cfg_oai.top_p},
                max_retries=cfg_oai.max_retries,
                timeout=cfg_oai.timeout,                              # was: request_timeout
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
# Missing #1: Health-Based Routing
#
# Checks the primary provider's health BEFORE invoking so an already-known
# unhealthy provider is skipped immediately instead of wasting time waiting
# for a timeout to confirm what the health check already told us.
#
# Flow:
#   Check Primary Health
#       ↓ HEALTHY              → proceed with primary invoke (with backoff)
#       ↓ DEGRADED/UNAVAILABLE → route directly to fallback
# ---------------------------------------------------------------------------

def _check_provider_health(provider: str, model: str) -> ProviderStatus:
    """
    Lightweight health-status query used by FallbackProvider for pre-routing.
    Returns UNAVAILABLE on any unexpected error so the caller safely falls back.
    """
    provider_lower = provider.strip().lower()
    try:
        if provider_lower == ProviderName.GEMINI:
            return get_gemini_provider().health_check().status
        if provider_lower == ProviderName.OPENAI:
            return get_openai_provider().health_check().status
    except Exception as exc:
        logger.warning(
            "Health check call itself failed | provider=%s | error=%s",
            provider, _safe_error_message(exc),
        )
    return ProviderStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# Missing #3: Fallback Tracking Per Agent
#
# Replaces the two global counters with per-agent breakdowns so management
# reports can show per-section reliability, e.g.:
#   Root Cause Agent:   95% OpenAI  /  5% Gemini Fallback
#   Owner Review Agent: 100% OpenAI /  0% Gemini Fallback
# ---------------------------------------------------------------------------

class _AgentUsageStats:
    """
    In-process per-agent usage counter.
    Thread-safety note: add a threading.Lock if moving to multi-threaded workers.
    """

    def __init__(self) -> None:
        # { agent_name: {"primary": int, "fallback": int} }
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"primary": 0, "fallback": 0}
        )

    def record(self, agent: str, slot: str) -> None:
        self._counts[agent][slot] += 1

    def get(self, agent: str) -> dict[str, Any]:
        c = self._counts.get(agent, {"primary": 0, "fallback": 0})
        total = c["primary"] + c["fallback"]
        return {
            "primary":      c["primary"],
            "fallback":     c["fallback"],
            "total":        total,
            "primary_pct":  round(c["primary"]  / total * 100, 1) if total else 0.0,
            "fallback_pct": round(c["fallback"] / total * 100, 1) if total else 0.0,
        }

    def get_all(self) -> dict[str, dict[str, Any]]:
        return {agent: self.get(agent) for agent in self._counts}

    def reset(self, agent: Optional[str] = None) -> None:
        if agent:
            self._counts.pop(agent, None)
        else:
            self._counts.clear()


_agent_usage = _AgentUsageStats()  # module-level singleton


# ---------------------------------------------------------------------------
# FallbackProvider
#
# Enhanced with all 4 missing features:
#   #1 Health-Based Routing    — pre-flight health check before primary invoke
#   #2 Exponential Backoff     — wait 2s/4s/8s (+jitter) between primary retries
#                                before switching to fallback
#   #3 Per-Agent Tracking      — per-section usage counters
#   #4 Dual Failure Recovery   — SectionGenerationFailedError instead of crash
#
# ── ARCHITECTURE NOTE ──────────────────────────────────────────────────────
# All retry, backoff, failover, and recovery logic lives HERE and nowhere
# else.  graph.py, service.py, routes.py, and chains.py must stay clean:
# they simply call FallbackProvider.invoke() (or LLMWithFallback.invoke())
# and handle SectionGenerationFailedError if dual failure occurs.
#
# invoke() flow:
#
#   ① Pre-flight health check (#1)
#       DEGRADED/UNAVAILABLE → skip to fallback immediately
#       HEALTHY              → continue to ②
#
#   ② Primary invoke with exponential backoff retries (#2)
#       Attempt 1 fails → wait 2s + jitter → retry
#       Attempt 2 fails → wait 4s + jitter → retry
#       Attempt 3 fails → wait 8s + jitter → switch to fallback
#       Attempt 1 succeeds → record primary usage (#3) → return
#
#   ③ Fallback invoke
#       Succeeds → record fallback usage (#3) → raise FallbackActivatedError
#       Fails    → raise SectionGenerationFailedError (#4)
#                  (caller writes GENERATION_FAILED_MARKER, RCA continues)
# ---------------------------------------------------------------------------

class FallbackProvider:
    """
    LLM Fallback Architecture wrapper with exponential backoff + jitter.

    All retry, backoff, and failover logic is centralised here so that
    graph.py and other callers remain clean orchestration code.
    """

    primary_usage_count:  int = 0
    fallback_usage_count: int = 0

    @staticmethod
    def invoke(
        prompt: Any,
        primary_provider: str,
        primary_model: str,
        fallback_provider: str,
        fallback_model: str,
        issue_id: str = "unknown",
        agent_name: str = "unknown",
        primary_retries: int = 3,
        skip_health_check: bool = False,
    ) -> Any:
        """
        Invoke with health-based routing, exponential backoff retries,
        per-agent tracking, and graceful dual-failure recovery.

        Args:
            prompt:            LangChain messages list or string prompt.
            primary_provider:  e.g. "openai"
            primary_model:     e.g. "gpt-5.5"
            fallback_provider: e.g. "gemini"
            fallback_model:    e.g. "gemini-2.5-flash"
            issue_id:          For structured logging.
            agent_name:        For per-agent usage stats (#3).
            primary_retries:   Number of retry attempts on primary before
                               switching to fallback. Default 3 matches the
                               three backoff delays [2s, 4s, 8s].
            skip_health_check: Bypass upfront health check if True (#1).

        Returns:
            LangChain response from whichever provider succeeded.

        Raises:
            SectionGenerationFailedError: Both providers failed (#4).
                Callers should catch this, write GENERATION_FAILED_MARKER
                into the section, and continue the RCA pipeline.
        """
        safe_primary_error: str = ""

        # ── ① Pre-flight health check (Missing #1) ──────────────────────────
        if not skip_health_check:
            primary_health = _check_provider_health(primary_provider, primary_model)
            if primary_health != ProviderStatus.HEALTHY:
                logger.warning(
                    "FallbackProvider: primary unhealthy, routing to fallback immediately | "
                    "issue_id=%s | agent=%s | primary=%s | status=%s",
                    issue_id, agent_name, primary_provider, primary_health,
                )
                safe_primary_error = f"Pre-flight health check: {primary_health}"
                return FallbackProvider._try_fallback(
                    prompt=prompt,
                    fallback_provider=fallback_provider,
                    fallback_model=fallback_model,
                    issue_id=issue_id,
                    agent_name=agent_name,
                    safe_primary_error=safe_primary_error,
                )

        # ── ② Primary invoke with exponential backoff + jitter (Missing #2) ─
        #
        # Attempt 0 = initial try (no delay before it).
        # Attempts 1..primary_retries = retries, each preceded by a backoff
        # sleep of (2 ** attempt) + random.uniform(0, 1) seconds.
        #
        # Example with primary_retries=3:
        #   attempt=0 → invoke immediately
        #   attempt=1 → wait ~2.0–3.0 s → invoke
        #   attempt=2 → wait ~4.0–5.0 s → invoke
        #   attempt=3 → wait ~8.0–9.0 s → invoke
        #   all failed → switch to fallback
        primary_llm = get_llm_for_agent(primary_provider, primary_model)

        for attempt in range(primary_retries + 1):  # 0, 1, 2, 3

            # Sleep before every retry (not before the very first attempt)
            if attempt > 0:
                delay = _backoff_delay(attempt)
                logger.warning(
                    "FallbackProvider: primary attempt %d/%d failed, "
                    "backing off %.2fs before retry | "
                    "issue_id=%s | agent=%s | error=%s",
                    attempt, primary_retries,
                    delay,
                    issue_id, agent_name, safe_primary_error,
                )
                time.sleep(delay)

            try:
                response = primary_llm.invoke(prompt)
                FallbackProvider.primary_usage_count += 1
                _agent_usage.record(agent_name, "primary")   # Missing #3
                logger.info(
                    "FallbackProvider: primary succeeded | "
                    "issue_id=%s | agent=%s | provider=%s | model=%s | attempt=%d",
                    issue_id, agent_name, primary_provider, primary_model, attempt,
                )
                return response

            except Exception as exc:
                safe_primary_error = _safe_error_message(exc)
                if attempt == primary_retries:
                    # All attempts exhausted — log final failure before fallback
                    logger.warning(
                        "FallbackProvider: primary exhausted all %d attempts, "
                        "activating fallback | "
                        "issue_id=%s | agent=%s | primary=%s | fallback=%s | error=%s",
                        primary_retries + 1,
                        issue_id, agent_name,
                        primary_provider, fallback_provider,
                        safe_primary_error,
                    )

        # ── ③ Fallback invoke ───────────────────────────────────────────────
        return FallbackProvider._try_fallback(
            prompt=prompt,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            issue_id=issue_id,
            agent_name=agent_name,
            safe_primary_error=safe_primary_error,
        )

    @staticmethod
    def _try_fallback(
        prompt: Any,
        fallback_provider: str,
        fallback_model: str,
        issue_id: str,
        agent_name: str,
        safe_primary_error: str,
    ) -> Any:
        """
        Attempts the fallback provider (no additional backoff — fallback is a
        different service, not a retry of the same failing one).

        On success: updates stats and returns the response.
        On failure: raises SectionGenerationFailedError (#4).
        """
        try:
            fallback_llm = get_llm_for_agent(fallback_provider, fallback_model)
            response = fallback_llm.invoke(prompt)
            FallbackProvider.fallback_usage_count += 1
            _agent_usage.record(agent_name, "fallback")       # Missing #3
            logger.warning(
                "FallbackProvider: fallback succeeded | "
                "issue_id=%s | agent=%s | fallback_provider=%s | fallback_model=%s",
                issue_id, agent_name, fallback_provider, fallback_model,
            )
            return response

        except Exception as fallback_exc:
            safe_fallback_error = _safe_error_message(fallback_exc)
            logger.exception(
                "FallbackProvider: both providers failed | "
                "issue_id=%s | agent=%s | "
                "primary_error=%s | fallback_error=%s",
                issue_id, agent_name,
                safe_primary_error, safe_fallback_error,
            )
            # Missing #4: structured error — caller marks section, RCA continues
            raise SectionGenerationFailedError(
                section=agent_name,
                primary_error=safe_primary_error,
                fallback_error=safe_fallback_error,
            ) from fallback_exc

    @classmethod
    def get_usage_stats(cls) -> dict[str, Any]:
        """
        Returns global usage counts plus per-agent breakdowns (#3).

        Example:
            {
                "primary_usage_count": 95,
                "fallback_usage_count": 5,
                "total": 100,
                "per_agent": {
                    "root_cause":      {"primary": 95, "fallback": 5,
                                        "total": 100, "primary_pct": 95.0,
                                        "fallback_pct": 5.0},
                    "impact_analysis": {"primary": 98, "fallback": 2,
                                        "total": 100, ...},
                    "owner_review":    {"primary": 100, "fallback": 0,
                                        "total": 100, ...},
                },
            }
        """
        total = cls.primary_usage_count + cls.fallback_usage_count
        return {
            "primary_usage_count":  cls.primary_usage_count,
            "fallback_usage_count": cls.fallback_usage_count,
            "total":                total,
            "per_agent":            _agent_usage.get_all(),
        }

    @classmethod
    def reset_usage_stats(cls) -> None:
        """Resets all usage counters. Use in tests or periodic metric flushes."""
        cls.primary_usage_count  = 0
        cls.fallback_usage_count = 0
        _agent_usage.reset()
        logger.info("FallbackProvider usage stats reset.")


# ---------------------------------------------------------------------------
# Convenience wrapper: get_llm_with_fallback / LLMWithFallback
# ---------------------------------------------------------------------------

def get_llm_with_fallback(
    primary_provider: str,
    primary_model: str,
    fallback_provider: str,
    fallback_model: str,
    issue_id: str = "unknown",
    agent_name: str = "unknown",
    primary_retries: int = 3,
    skip_health_check: bool = False,
) -> "LLMWithFallback":
    """
    Returns an LLMWithFallback bound to the given primary/fallback config.

    primary_retries defaults to 3 to align with the three backoff tiers
    [2s, 4s, 8s].  Set lower (e.g. 1) for fast-fail tests.

    Usage:
        llm = get_llm_with_fallback(
            primary_provider="openai",
            primary_model="gpt-5.5",
            fallback_provider="gemini",
            fallback_model="gemini-2.5-flash",
            issue_id=issue_id,
            agent_name="root_cause",
        )
        try:
            response = llm.invoke(prompt.format_messages())
            state["root_cause"] = extract_text(response)
        except SectionGenerationFailedError as exc:         # Missing #4
            state["root_cause"] = GENERATION_FAILED_MARKER
            state.setdefault("failed_sections", []).append(exc.section)
    """
    return LLMWithFallback(
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        issue_id=issue_id,
        agent_name=agent_name,
        primary_retries=primary_retries,
        skip_health_check=skip_health_check,
    )


class LLMWithFallback:
    """
    Thin wrapper exposing .invoke(prompt) that delegates to
    FallbackProvider.invoke() with the configured primary/fallback pair.
    Drop-in compatible with LangChain's (prompt | llm).invoke({}) pattern.
    """

    def __init__(
        self,
        primary_provider: str,
        primary_model: str,
        fallback_provider: str,
        fallback_model: str,
        issue_id: str = "unknown",
        agent_name: str = "unknown",
        primary_retries: int = 3,
        skip_health_check: bool = False,
    ) -> None:
        self._primary_provider  = primary_provider
        self._primary_model     = primary_model
        self._fallback_provider = fallback_provider
        self._fallback_model    = fallback_model
        self._issue_id          = issue_id
        self._agent_name        = agent_name
        self._primary_retries   = primary_retries
        self._skip_health_check = skip_health_check

    def invoke(self, prompt: Any) -> Any:
        return FallbackProvider.invoke(
            prompt=prompt,
            primary_provider=self._primary_provider,
            primary_model=self._primary_model,
            fallback_provider=self._fallback_provider,
            fallback_model=self._fallback_model,
            issue_id=self._issue_id,
            agent_name=self._agent_name,
            primary_retries=self._primary_retries,
            skip_health_check=self._skip_health_check,
        )

    def __repr__(self) -> str:
        return (
            f"LLMWithFallback("
            f"primary={self._primary_provider}/{self._primary_model}, "
            f"fallback={self._fallback_provider}/{self._fallback_model}, "
            f"agent={self._agent_name!r}, retries={self._primary_retries})"
        )


# ---------------------------------------------------------------------------
# Missing #4: Dual Failure Recovery — constants and helpers
#
# Graph nodes in graph.py should handle SectionGenerationFailedError like this:
#
#   try:
#       response = llm.invoke(prompt.format_messages())
#       state["root_cause"] = extract_text(response)
#   except SectionGenerationFailedError as exc:
#       state["root_cause"]      = GENERATION_FAILED_MARKER
#       state["failed_sections"] = state.get("failed_sections", []) + [exc.section]
#       logger.warning("Section marked for manual review | section=%s", exc.section)
#
# The RCA pipeline continues; the failed section is clearly labelled in the
# output document instead of the whole job crashing and losing all other sections.
# ---------------------------------------------------------------------------

GENERATION_FAILED_MARKER = (
    "[Generation Failed - Manual Review Required]\n\n"
    "This section could not be generated automatically because both AI providers "
    "were unavailable at the time of generation. Please complete this section manually."
)


def is_generation_failed(text: str) -> bool:
    """Returns True if the section contains the dual-failure marker."""
    return "[Generation Failed - Manual Review Required]" in text


# ---------------------------------------------------------------------------
# Module-level helpers for service layer
# ---------------------------------------------------------------------------

def get_gemini_model() -> ChatGoogleGenerativeAI:
    return get_gemini_provider().get_llm()


def invoke_gemini(prompt: str) -> str:
    return get_gemini_provider().invoke(prompt)


def invoke_openai(prompt: str) -> str:
    return get_openai_provider().invoke(prompt)


def reset_gemini_provider() -> None:
    get_gemini_provider.cache_clear()
    logger.warning("GeminiProvider singleton cache cleared — test use only.")


def reset_openai_provider() -> None:
    get_openai_provider.cache_clear()
    logger.warning("OpenAIProvider singleton cache cleared — test use only.")


def reset_all_providers() -> None:
    reset_gemini_provider()
    reset_openai_provider()