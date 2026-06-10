#chains.py
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from agent_root.models import RCAInputModel
from agent_root.prompts import PromptLoadError, PromptRenderError, render_rca_prompt
from config.providers import extract_text, get_gemini_model

logger = logging.getLogger("rca_generator.chains")


class ChainBuildError(RuntimeError):
    """Raised when the RCA chain cannot be constructed."""


class ChainExecutionError(RuntimeError):
    """Raised when the RCA chain fails during execution."""


def _render_to_prompt_string(rca_input: RCAInputModel) -> str:
    try:
        rendered = render_rca_prompt(rca_input)

        prompt = (
            f"{rendered.system.strip()}\n\n"
            "---\n\n"
            f"{rendered.user.strip()}"
        )

        if not prompt.strip():
            raise ChainExecutionError("Rendered RCA prompt is empty.")

        return prompt

    except (PromptLoadError, PromptRenderError) as exc:
        issue_id = rca_input.task_monitoring_data.issue_logs_id
        logger.exception(
            "Prompt rendering failed inside chain | issue_id=%s | error=%s",
            issue_id,
            exc,
        )
        raise ChainExecutionError(
            f"Failed to render RCA prompt for issue {issue_id}: {exc}"
        ) from exc


def _parse_llm_output(response: Any) -> str:
    markdown = extract_text(response)

    if not markdown or not markdown.strip():
        raise ChainExecutionError("Gemini returned an empty RCA response.")

    return markdown.strip()


@lru_cache(maxsize=1)
def get_rca_chain() -> Runnable[RCAInputModel, str]:
    logger.info("Building RCA LangChain chain")

    try:
        llm = get_gemini_model()

        chain: Runnable[RCAInputModel, str] = (
            RunnableLambda(_render_to_prompt_string)
            | llm
            | RunnableLambda(_parse_llm_output)
        )

        logger.info("RCA LangChain chain built successfully")
        return chain

    except Exception as exc:
        logger.exception("Failed to build RCA LangChain chain | error=%s", exc)
        raise ChainBuildError(
            f"RCA chain construction failed: {exc}"
        ) from exc


def invoke_rca_chain(rca_input: RCAInputModel) -> str:
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info("Invoking RCA LangChain chain | issue_id=%s", issue_id)

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
    issue_id = rca_input.task_monitoring_data.issue_logs_id

    logger.info("Async invoking RCA LangChain chain | issue_id=%s", issue_id)

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
            "Async RCA LangChain chain execution failed | issue_id=%s | error=%s",
            issue_id,
            exc,
        )
        raise ChainExecutionError(
            f"Async RCA chain execution failed for issue {issue_id}: {exc}"
        ) from exc


def reset_rca_chain() -> None:
    get_rca_chain.cache_clear()
    logger.warning("RCA chain cache cleared — test use only.")