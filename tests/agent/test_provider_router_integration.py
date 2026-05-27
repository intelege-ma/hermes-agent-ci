"""Phase 1b: LOG-ONLY integration of UsageTracker into the conversation loop.

These tests verify that the agent loop calls UsageTracker.record() after
successful API calls and UsageTracker.mark_rate_limited() / record_error_signal()
after failures — WITHOUT altering any routing or fallback behavior.

RED phase: these should FAIL until the integration is implemented.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", usage=None, model="test/model"):
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model=model)
    if usage:
        resp.usage = SimpleNamespace(**usage)
    else:
        resp.usage = None
    return resp


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def _setup_agent(agent):
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.provider = "openrouter"
    agent.model = "anthropic/claude-sonnet-4"


# ===================================================================
# 1. Success path: tracker.record() called on successful API call
# ===================================================================

class TestUsageTrackerSuccessPath:
    """After a successful API response, the loop should call tracker.record()."""

    def test_record_called_on_success(self, agent):
        _setup_agent(agent)
        resp = _mock_response(
            content="Final answer",
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        agent.client.chat.completions.create.return_value = resp

        mock_tracker = MagicMock()

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop._get_usage_tracker", return_value=mock_tracker),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        mock_tracker.record.assert_called_once()
        call_kwargs = mock_tracker.record.call_args
        # Verify positional args: provider, model
        assert call_kwargs[0][0] == "openrouter"
        assert call_kwargs[0][1] == "anthropic/claude-sonnet-4"
        # Verify keyword args include tokens
        assert call_kwargs[1]["tokens_in"] == 100
        assert call_kwargs[1]["tokens_out"] == 50

    def test_record_skipped_when_no_usage(self, agent):
        """If response has no usage data, the token accounting block is skipped
        entirely (upstream guard), so tracker.record() is not called.
        This is correct — without usage there are no meaningful tokens to log."""
        _setup_agent(agent)
        resp = _mock_response(content="No usage info", finish_reason="stop", usage=None)
        agent.client.chat.completions.create.return_value = resp

        mock_tracker = MagicMock()

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop._get_usage_tracker", return_value=mock_tracker),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        # tracker.record is NOT called because the usage block is guarded
        mock_tracker.record.assert_not_called()


# ===================================================================
# 2. Error path: tracker.mark_rate_limited() on 429
# ===================================================================

class TestUsageTrackerRateLimitPath:
    """On a 429 error, the tracker should be notified via mark_rate_limited()."""

    def test_mark_rate_limited_on_429(self, agent):
        _setup_agent(agent)

        # First call: 429 error
        error_429 = Exception("Rate limited")
        error_429.status_code = 429
        error_429.headers = {"retry-after": "30"}

        # Second call: success
        resp = _mock_response(content="Retry worked", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [error_429, resp]

        mock_tracker = MagicMock()

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop._get_usage_tracker", return_value=mock_tracker),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        mock_tracker.mark_rate_limited.assert_called_once()
        call_kwargs = mock_tracker.mark_rate_limited.call_args
        assert call_kwargs[0][0] == "openrouter"  # provider


# ===================================================================
# 3. Error path: tracker.record_error_signal() on non-429 error
# ===================================================================

class TestUsageTrackerErrorPath:
    """On a non-429 API error, the tracker should get record_error_signal()."""

    def test_record_error_signal_on_500(self, agent):
        _setup_agent(agent)

        error_500 = Exception("Server error")
        error_500.status_code = 500
        error_500.headers = {}

        resp = _mock_response(content="Retry worked", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [error_500, resp]

        mock_tracker = MagicMock()

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop._get_usage_tracker", return_value=mock_tracker),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        mock_tracker.record_error_signal.assert_called()
        call_kwargs = mock_tracker.record_error_signal.call_args
        assert call_kwargs[0][0] == "openrouter"  # provider
        assert call_kwargs[1]["status_code"] == 500


# ===================================================================
# 4. Safety: tracker failure never breaks the agent loop
# ===================================================================

class TestUsageTrackerSafety:
    """A broken tracker must never crash or alter the conversation loop."""

    def test_tracker_record_exception_does_not_crash(self, agent):
        _setup_agent(agent)
        resp = _mock_response(
            content="Still works",
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        )
        agent.client.chat.completions.create.return_value = resp

        mock_tracker = MagicMock()
        mock_tracker.record.side_effect = RuntimeError("DB is broken")

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop._get_usage_tracker", return_value=mock_tracker),
        ):
            result = agent.run_conversation("hello")

        # Conversation must complete successfully despite tracker failure
        assert result["completed"] is True
        assert result["final_response"] == "Still works"

    def test_tracker_mark_rate_limited_exception_does_not_crash(self, agent):
        _setup_agent(agent)

        error_429 = Exception("Rate limited")
        error_429.status_code = 429
        error_429.headers = {"retry-after": "30"}

        resp = _mock_response(content="Retry worked", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [error_429, resp]

        mock_tracker = MagicMock()
        mock_tracker.mark_rate_limited.side_effect = RuntimeError("DB is broken")

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop._get_usage_tracker", return_value=mock_tracker),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Retry worked"

    def test_tracker_none_means_no_calls(self, agent):
        """If _get_usage_tracker returns None, no tracker calls are made."""
        _setup_agent(agent)
        resp = _mock_response(
            content="No tracker",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        agent.client.chat.completions.create.return_value = resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.conversation_loop._get_usage_tracker", return_value=None),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "No tracker"
