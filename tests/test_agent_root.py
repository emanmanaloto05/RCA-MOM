# tests/test_agent_root.py
"""
Tests for agent_root/chains.py

Covers: _build_prompt_messages, _parse_llm_output,
        get_rca_chain (cache), invoke_rca_chain, ainvoke_rca_chain,
        reset_rca_chain.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_root.chains import (
    ChainBuildError,
    ChainExecutionError,
    ainvoke_rca_chain,
    get_rca_chain,
    invoke_rca_chain,
    reset_rca_chain,
)
from agent_root.models import RCAInputModel
from tests.conftest import VALID_MARKDOWN


# =============================================================================
# get_rca_chain — build & cache
# =============================================================================

class TestGetRcaChain:
    def setup_method(self) -> None:
        reset_rca_chain()

    def test_returns_runnable(self) -> None:
        with patch("agent_root.chains.get_gemini_model", return_value=MagicMock()):
            chain = get_rca_chain()
        assert chain is not None

    def test_chain_is_cached(self) -> None:
        with patch("agent_root.chains.get_gemini_model", return_value=MagicMock()):
            c1 = get_rca_chain()
            c2 = get_rca_chain()
        assert c1 is c2

    def test_reset_clears_cache(self) -> None:
        with patch("agent_root.chains.get_gemini_model", return_value=MagicMock()):
            c1 = get_rca_chain()
            reset_rca_chain()
            c2 = get_rca_chain()
        assert c1 is not c2

    def test_raises_chain_build_error_on_failure(self) -> None:
        reset_rca_chain()
        with patch(
            "agent_root.chains.get_gemini_model",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            with pytest.raises(ChainBuildError, match="RCA chain construction failed"):
                get_rca_chain()


# =============================================================================
# invoke_rca_chain
# =============================================================================

class TestInvokeRcaChain:
    def setup_method(self) -> None:
        reset_rca_chain()

    def test_returns_markdown_on_success(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = VALID_MARKDOWN

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            result = invoke_rca_chain(rca_input)

        assert result == VALID_MARKDOWN.strip()

    def test_strips_output_whitespace(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = f"  {VALID_MARKDOWN}  "

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            result = invoke_rca_chain(rca_input)

        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_raises_execution_error_on_empty_result(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "   "

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            with pytest.raises(ChainExecutionError, match="empty output"):
                invoke_rca_chain(rca_input)

    def test_re_raises_chain_execution_error(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ChainExecutionError("Already an execution error")

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            with pytest.raises(ChainExecutionError, match="Already an execution error"):
                invoke_rca_chain(rca_input)

    def test_wraps_unexpected_exception(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("Connection reset")

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            with pytest.raises(ChainExecutionError, match="RCA chain execution failed"):
                invoke_rca_chain(rca_input)


# =============================================================================
# ainvoke_rca_chain
# =============================================================================

class TestAinvokeRcaChain:
    def setup_method(self) -> None:
        reset_rca_chain()

    @pytest.mark.asyncio
    async def test_returns_markdown_on_success(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=VALID_MARKDOWN)

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            result = await ainvoke_rca_chain(rca_input)

        assert result == VALID_MARKDOWN.strip()

    @pytest.mark.asyncio
    async def test_raises_on_empty_async_result(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="")

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            with pytest.raises(ChainExecutionError, match="empty output"):
                await ainvoke_rca_chain(rca_input)

    @pytest.mark.asyncio
    async def test_re_raises_execution_error(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(
            side_effect=ChainExecutionError("Async exec error")
        )

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            with pytest.raises(ChainExecutionError, match="Async exec error"):
                await ainvoke_rca_chain(rca_input)

    @pytest.mark.asyncio
    async def test_wraps_unexpected_async_exception(self, rca_input: RCAInputModel) -> None:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=ValueError("Bad value"))

        with patch("agent_root.chains.get_rca_chain", return_value=mock_chain):
            with pytest.raises(ChainExecutionError, match="Async RCA chain execution failed"):
                await ainvoke_rca_chain(rca_input)