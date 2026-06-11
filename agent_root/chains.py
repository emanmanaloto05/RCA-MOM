# agent_root/chains.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

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


def _build_prompt_messages(rca_input: RCAInputModel) -> list[BaseMessage]:
    """
    Builds the RCA ChatPromptTemplate and converts it into
    LangChain chat messages for Gemini.
    """
    try:
        prompt = build_rca_chat_prompt(rca_input)
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


def _parse_llm_output(response: Any) -> str:
    """
    Extracts clean Markdown text from the Gemini response.
    """
    markdown = extract_text(response)

    if not markdown or not markdown.strip():
        raise ChainExecutionError("Gemini returned an empty RCA response.")

    return markdown.strip()


@lru_cache(maxsize=1)
def get_rca_chain() -> Runnable[RCAInputModel, str]:
    """
    Builds and caches the RCA LangChain pipeline.

    Flow:
        RCAInputModel
        -> ChatPromptTemplate messages
        -> Gemini LLM
        -> Markdown RCA string
    """
    logger.info("Building RCA LangChain chain")

    try:
        llm = get_gemini_model()

        chain: Runnable[RCAInputModel, str] = (
            RunnableLambda(_build_prompt_messages)
            | llm
            | RunnableLambda(_parse_llm_output)
        )

        logger.info("RCA LangChain chain built successfully")
        return chain

    except Exception as exc:
        logger.exception(
            "Failed to build RCA LangChain chain | error=%s",
            exc,
        )

        raise ChainBuildError(
            f"RCA chain construction failed: {exc}"
        ) from exc


def invoke_rca_chain(rca_input: RCAInputModel) -> str:
    """
    Synchronously runs the cached RCA chain.
    """
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info(
        "Invoking RCA LangChain chain | issue_id=%s",
        issue_id,
    )

    try:
        result = get_rca_chain().invoke(rca_input)

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

    except Exception as exc:
        logger.exception(
            "RCA LangChain chain execution failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise ChainExecutionError(
            f"RCA chain execution failed for issue {issue_id}: {exc}"
        ) from exc


async def ainvoke_rca_chain(rca_input: RCAInputModel) -> str:
    """
    Asynchronously runs the cached RCA chain.
    """
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info(
        "Async invoking RCA LangChain chain | issue_id=%s",
        issue_id,
    )

    try:
        result = await get_rca_chain().ainvoke(rca_input)

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

    except Exception as exc:
        logger.exception(
            "Async RCA chain execution failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )

        raise ChainExecutionError(
            f"Async RCA chain execution failed for issue {issue_id}: {exc}"
        ) from exc


def reset_rca_chain() -> None:
    """
    Clears the cached RCA chain.
    Use only for tests or hot reload.
    """
    get_rca_chain.cache_clear()
    logger.warning("RCA chain cache cleared — test use only.")