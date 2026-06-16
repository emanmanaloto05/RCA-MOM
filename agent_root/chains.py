# agent_root/chains.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, TypedDict

from google.genai import errors as genai_errors
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from agent_root.models import RCAInputModel
from common.utils import (
    PromptLoadError,
    PromptRenderError,
    build_rca_chat_prompt,
)
from config.providers import extract_text, get_gemini_model

logger = logging.getLogger("rca_generator.chains")


class ChainBuildError(RuntimeError):
    """Raised when the RCA chain cannot be constructed."""


class ChainExecutionError(RuntimeError):
    """Raised when the RCA chain fails during execution."""


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN INPUT TYPE
#
# The chain previously accepted a bare RCAInputModel, which meant
# quality_summary (computed by the analyze_quality_gates graph node) had
# no way to reach the Jinja2 prompt renderer.
#
# ChainInput is a TypedDict that carries both the model and the pre-computed
# summary so _build_prompt_messages can forward it to build_rca_chat_prompt.
# ─────────────────────────────────────────────────────────────────────────────

class ChainInput(TypedDict):
    """
    Input envelope for the RCA LangChain pipeline.

    Fields
    ──────
    rca_input       : RCAInputModel
        The fully validated issue payload.
    quality_summary : str
        Pre-computed quality gate summary produced by the
        analyze_quality_gates graph node.  Empty string when the
        caller bypasses the graph (e.g. unit tests).
    """
    rca_input:       RCAInputModel
    quality_summary: str


# ─────────────────────────────────────────────────────────────────────────────
# RETRY CONFIGURATION
#
# Gemini occasionally returns transient errors:
#   - 503 UNAVAILABLE (model overloaded / high demand)
#   - 429 RESOURCE_EXHAUSTED (rate limiting)
#   - other 5xx server-side errors
#
# These are NOT caused by our code and typically resolve within seconds.
# RETRYABLE_GENAI_ERRORS is passed to Runnable.with_retry() so the LLM
# step retries with exponential backoff + jitter before the whole graph
# run is marked as failed.
#
# NOTE: This tuple is also used directly in `except` clauses below.
# Python requires exception types in `except` to be literal class references
# or a literal tuple — a variable holding a tuple works correctly only when
# used as-is (not via `except variable`). We define it here once and reference
# it by name in both with_retry() and the except clauses, which is valid
# because Python evaluates the except expression at runtime.
# ─────────────────────────────────────────────────────────────────────────────

RETRYABLE_GENAI_ERRORS: tuple[type[Exception], ...] = (
    genai_errors.ServerError,   # covers 500 / 503 UNAVAILABLE / high demand
    genai_errors.ClientError,   # covers 429 RESOURCE_EXHAUSTED rate limits
)

LLM_RETRY_STOP_AFTER_ATTEMPT: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt_messages(chain_input: ChainInput) -> list[BaseMessage]:
    """
    Builds the RCA ChatPromptTemplate and converts it into
    LangChain chat messages for Gemini.

    Reads
    ─────
    chain_input["rca_input"]       : RCAInputModel
    chain_input["quality_summary"] : str

    Both are forwarded to build_rca_chat_prompt so the Jinja2 renderer has
    access to every variable referenced in prompts.yaml — including
    {{ attachments }} and {{ quality_summary }}.
    """
    rca_input:       RCAInputModel = chain_input["rca_input"]
    quality_summary: str           = chain_input.get("quality_summary", "")  # type: ignore[typeddict-item]

    try:
        prompt   = build_rca_chat_prompt(rca_input, quality_summary=quality_summary)
        messages = prompt.format_messages()

        if not messages:
            raise ChainExecutionError(
                "build_rca_chat_prompt returned empty messages."
            )

        return messages

    except (PromptLoadError, PromptRenderError) as exc:
        issue_id = rca_input.task_monitoring_data.issue_logs_id

        logger.exception(
            "Prompt build failed inside chain | issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise ChainExecutionError(
            f"Failed to build RCA prompt for issue {issue_id}: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LLM OUTPUT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_output(response: Any) -> str:
    """
    Extracts clean Markdown text from the Gemini response.
    """
    markdown = extract_text(response)

    if not markdown or not markdown.strip():
        raise ChainExecutionError("Gemini returned an empty RCA response.")

    return markdown.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN FACTORY
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rca_chain() -> Runnable[ChainInput, str]:
    """
    Builds and caches the RCA LangChain pipeline.

    Flow:
        ChainInput  (rca_input + quality_summary)
        -> list[BaseMessage]      (_build_prompt_messages)
        -> Gemini LLM response    (retried on transient errors)
        -> Markdown RCA string    (_parse_llm_output)

    The LLM step is wrapped with `.with_retry()` so transient Gemini
    errors (503 UNAVAILABLE / high demand, 429 rate limits, and other
    5xx server errors) are retried with exponential backoff + jitter
    before the whole graph run is marked as failed. Non-retryable
    errors (e.g. malformed prompts, ChainExecutionError) propagate
    immediately without retry.

    Returns
    ───────
    Runnable[ChainInput, str]

    Raises
    ──────
    ChainBuildError : If the LLM cannot be instantiated.
    """
    logger.info("Building RCA LangChain chain")

    try:
        llm = get_gemini_model()

        resilient_llm = llm.with_retry(
            retry_if_exception_type=RETRYABLE_GENAI_ERRORS,
            wait_exponential_jitter=True,
            stop_after_attempt=LLM_RETRY_STOP_AFTER_ATTEMPT,
        )

        chain: Runnable[ChainInput, str] = (
            RunnableLambda(_build_prompt_messages)
            | resilient_llm
            | RunnableLambda(_parse_llm_output)
        )

        logger.info(
            "RCA LangChain chain built successfully | "
            "llm_retry_attempts=%d | retryable_errors=%s",
            LLM_RETRY_STOP_AFTER_ATTEMPT,
            [exc.__name__ for exc in RETRYABLE_GENAI_ERRORS],
        )
        return chain

    except Exception as exc:
        logger.exception(
            "Failed to build RCA LangChain chain | error=%s",
            exc,
        )

        raise ChainBuildError(
            f"RCA chain construction failed: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INVOKE HELPERS
# These are convenience wrappers used outside the LangGraph pipeline
# (e.g. scripts, tests, or future API endpoints that call the chain directly).
# Both accept the same ChainInput dict so callers always supply quality_summary.
# ─────────────────────────────────────────────────────────────────────────────

def invoke_rca_chain(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> str:
    """
    Synchronously runs the cached RCA chain.

    Args:
        rca_input:       Validated RCAInputModel instance.
        quality_summary: Pre-computed quality gate summary string.
                         Defaults to "" for callers that bypass the graph.

    Returns:
        Stripped Markdown RCA string.

    Raises:
        ChainExecutionError: On empty output or any chain-level failure
                              (including exhausted retries on transient
                              Gemini errors).
    """
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info(
        "Invoking RCA LangChain chain | issue_id=%s",
        issue_id,
    )

    try:
        chain_input: ChainInput = {
            "rca_input":       rca_input,
            "quality_summary": quality_summary,
        }

        result = get_rca_chain().invoke(chain_input)

        if not result or not result.strip():
            raise ChainExecutionError(
                f"RCA chain returned empty output for issue {issue_id}."
            )

        logger.info(
            "RCA LangChain chain completed | issue_id=%s | output_chars=%d",
            issue_id,
            len(result),
        )

        return result.strip()

    except ChainExecutionError:
        raise

    except (genai_errors.ServerError, genai_errors.ClientError) as exc:
        logger.exception(
            "RCA LangChain chain failed after exhausting retries "
            "| issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise ChainExecutionError(
            f"Gemini was unavailable after {LLM_RETRY_STOP_AFTER_ATTEMPT} "
            f"attempts for issue {issue_id}: {exc}"
        ) from exc

    except Exception as exc:
        logger.exception(
            "RCA LangChain chain execution failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise ChainExecutionError(
            f"RCA chain execution failed for issue {issue_id}: {exc}"
        ) from exc


async def ainvoke_rca_chain(
    rca_input: RCAInputModel,
    quality_summary: str = "",
) -> str:
    """
    Asynchronously runs the cached RCA chain.

    Args:
        rca_input:       Validated RCAInputModel instance.
        quality_summary: Pre-computed quality gate summary string.
                         Defaults to "" for callers that bypass the graph.

    Returns:
        Stripped Markdown RCA string.

    Raises:
        ChainExecutionError: On empty output or any chain-level failure
                              (including exhausted retries on transient
                              Gemini errors).
    """
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info(
        "Async invoking RCA LangChain chain | issue_id=%s",
        issue_id,
    )

    try:
        chain_input: ChainInput = {
            "rca_input":       rca_input,
            "quality_summary": quality_summary,
        }

        result = await get_rca_chain().ainvoke(chain_input)

        if not result or not result.strip():
            raise ChainExecutionError(
                f"RCA chain returned empty output for issue {issue_id}."
            )

        logger.info(
            "Async RCA LangChain chain completed | issue_id=%s | output_chars=%d",
            issue_id,
            len(result),
        )

        return result.strip()

    except ChainExecutionError:
        raise

    except (genai_errors.ServerError, genai_errors.ClientError) as exc:
        logger.exception(
            "Async RCA LangChain chain failed after exhausting retries "
            "| issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise ChainExecutionError(
            f"Gemini was unavailable after {LLM_RETRY_STOP_AFTER_ATTEMPT} "
            f"attempts for issue {issue_id}: {exc}"
        ) from exc

    except Exception as exc:
        logger.exception(
            "Async RCA chain execution failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise ChainExecutionError(
            f"Async RCA chain execution failed for issue {issue_id}: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# CACHE RESET — test / hot-reload use only
# ─────────────────────────────────────────────────────────────────────────────

def reset_rca_chain() -> None:
    """
    Clears the cached RCA chain.
    Use only for tests or hot reload.
    """
    get_rca_chain.cache_clear()
    logger.warning("RCA chain cache cleared — test use only.")