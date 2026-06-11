# agent_root/service.py

"""
RCA Service — orchestrates the full RCA generation pipeline via LangGraph.

Flow:
    RCAInputModel → RCAGraphState → LangGraph workflow → RCAOutputModel
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from agent_root.graph import GraphBuildError, RCAGraphState, get_rca_graph
from agent_root.models import RCAInputModel, RCAOutputModel

logger = logging.getLogger("rca_generator.service")


class RCAServiceError(RuntimeError):
    """Raised when the RCA generation pipeline fails."""


class RCAService:
    """Stateless service class for RCA generation."""

    @classmethod
    async def generate_rca(cls, rca_input: RCAInputModel) -> RCAOutputModel:
        issue_id = rca_input.task_monitoring_data.issue_logs_id

        if not issue_id.strip():
            raise RCAServiceError("Issue ID is required for RCA generation.")

        logger.info("RCA generation started | issue_id=%s", issue_id)

        initial_state: RCAGraphState = {
            "rca_input": rca_input
        }

        try:
            graph: Any = get_rca_graph()

            final_state_raw: Any = await asyncio.to_thread(
                graph.invoke,
                initial_state,
            )

            if not isinstance(final_state_raw, dict):
                raise RCAServiceError(
                    f"Invalid graph output type for issue {issue_id}: "
                    f"{type(final_state_raw).__name__}"
                )

            final_state = cast(dict[str, Any], final_state_raw)

        except GraphBuildError as exc:
            logger.exception(
                "Graph build failed | issue_id=%s | error=%s",
                issue_id,
                exc,
            )
            raise RCAServiceError(
                f"RCA workflow could not be initialized for issue {issue_id}."
            ) from exc

        except RCAServiceError:
            raise

        except Exception as exc:
            logger.exception(
                "Graph invocation failed | issue_id=%s | error=%s",
                issue_id,
                exc,
            )
            raise RCAServiceError(
                f"RCA workflow failed for issue {issue_id}."
            ) from exc

        review_passed = final_state.get("review_passed", False)

        if not review_passed:
            generation_error = final_state.get("generation_error")
            review_notes = final_state.get("review_notes")

            error_detail = (
                generation_error
                or review_notes
                or "Unknown RCA review failure."
            )

            logger.error(
                "RCA review failed | issue_id=%s | detail=%s",
                issue_id,
                error_detail,
            )

            raise RCAServiceError(
                f"RCA generation did not pass review for issue {issue_id}: "
                f"{error_detail}"
            )

        rca_output = final_state.get("rca_output")

        if not isinstance(rca_output, RCAOutputModel):
            logger.error(
                "Invalid or missing RCA output | issue_id=%s | output_type=%s",
                issue_id,
                type(rca_output).__name__,
            )
            raise RCAServiceError(
                f"RCA output is missing or invalid for issue {issue_id}."
            )

        if not rca_output.markdown_rca.strip():
            raise RCAServiceError(
                f"Generated RCA markdown is empty for issue {issue_id}."
            )

        logger.info(
            "RCA generation completed | issue_id=%s | output_chars=%d",
            issue_id,
            len(rca_output.markdown_rca),
        )

        return rca_output