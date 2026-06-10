# service.py
"""
RCA Service — orchestrates the full generation pipeline via LangGraph.

Flow:
    RCAInputModel → RCAGraphState → LangGraph workflow → RCAOutputModel
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_root.graph import GraphBuildError, RCAGraphState, get_rca_graph
from agent_root.models import RCAInputModel, RCAOutputModel

logger = logging.getLogger("rca_generator.service")


class RCAServiceError(RuntimeError):
    """
    Raised when the RCA generation pipeline fails for any reason.
    Wraps lower-level exceptions so the route layer catches one type.
    The original cause is always chained via `from exc`.
    """


class RCAService:
    """
    Stateless service class for RCA generation.
    All methods are classmethods — no instance is needed.
    """

    @classmethod
    async def generate_rca(cls, rca_input: RCAInputModel) -> RCAOutputModel:
        """
        Runs the LangGraph RCA workflow asynchronously.

        Steps:
          1. Build the initial graph state from the input model.
          2. Invoke the compiled graph (offloaded to a thread).
          3. Inspect the final state for errors or a missing output.
          4. Return the typed RCAOutputModel.

        Raises RCAServiceError on any failure.
        """
        issue_id = rca_input.task_monitoring_data.issue_logs_id

        logger.info("RCA generation started | issue_id=%s", issue_id)

        # Step 1 — Build initial state
        initial_state: RCAGraphState = {"rca_input": rca_input}

        # Step 2 — Run the graph (sync → thread)
        # get_rca_graph() returns Any because LangGraph has no Pylance stubs.
        # We call .invoke() via a lambda to satisfy asyncio.to_thread's
        # type checker, which requires a concrete callable signature.
        try:
            graph: Any = get_rca_graph()
            final_state: dict[str, Any] = await asyncio.to_thread(
                lambda: graph.invoke(initial_state)
            )
        except GraphBuildError as exc:
            logger.exception(
                "Graph build failed | issue_id=%s | error=%s", issue_id, exc
            )
            raise RCAServiceError(
                f"RCA workflow could not be initialised for issue {issue_id}: {exc}"
            ) from exc
        except Exception as exc:
            logger.exception(
                "Graph invocation failed | issue_id=%s | error=%s", issue_id, exc
            )
            raise RCAServiceError(
                f"RCA workflow failed for issue {issue_id}: {exc}"
            ) from exc

        # Step 3 — Inspect final state
        if not final_state.get("review_passed"):
            generation_error = final_state.get("generation_error")
            review_notes = final_state.get("review_notes", "Unknown review failure.")
            error_detail = generation_error or review_notes
            logger.error(
                "RCA review failed | issue_id=%s | detail=%s", issue_id, error_detail
            )
            raise RCAServiceError(
                f"RCA generation did not pass review for issue {issue_id}: {error_detail}"
            )

        rca_output: RCAOutputModel | None = final_state.get("rca_output")

        if rca_output is None:
            raise RCAServiceError(
                f"RCA output is missing from final graph state for issue {issue_id}."
            )

        logger.info(
            "RCA generation completed | issue_id=%s | output_chars=%d",
            issue_id,
            len(rca_output.markdown_rca),
        )

        return rca_output