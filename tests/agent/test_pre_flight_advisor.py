"""Phase 2a: Advisory pre-flight gate tests.

The advisory gate reads tracker cooldown state and returns a recommendation
WITHOUT enforcing it. It logs what *would* happen for measurement during the
soak period.

Design decisions (user-approved 2026-05-27):
- Conservative: only recommend skip if cooldown until is still in the future
- No routing enforcement in Phase 2a — advisory log-only
- Default backoff of 60s when no retry-after header (already applied by tracker)
- If fallback also fails, retry original (no permanent exile) — handled by
  Phase 2b, not tested here
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.provider_router.usage_tracker import UsageTracker


class FrozenClock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs) -> None:
        self.value = self.value + timedelta(**kwargs)


# ── RED: These tests exercise the pre_flight_advisor module that does NOT yet exist.
# ── They should FAIL on import until the module is implemented.


def test_advisory_returns_skip_when_provider_in_cooldown(tmp_path):
    """If a provider is in active cooldown, the advisor should recommend skipping."""
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        provider_pools={"opencode-go": [("opencode-go", "deepseek-v4-pro")]},
        now_fn=clock,
    )
    tracker.mark_rate_limited("opencode-go", "deepseek-v4-pro", retry_after=120)

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)
    result = advisor.check("opencode-go", "deepseek-v4-pro")

    assert result.should_skip is True
    assert "cooldown" in result.reason.lower()
    assert result.cooldown_remaining_s > 0


def test_advisory_returns_no_skip_when_provider_healthy(tmp_path):
    """If a provider has no cooldown, the advisor should recommend NOT skipping."""
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        now_fn=clock,
    )

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)
    result = advisor.check("opencode-go", "deepseek-v4-pro")

    assert result.should_skip is False
    assert result.reason == ""
    assert result.cooldown_remaining_s == 0


def test_advisory_returns_no_skip_after_cooldown_expires(tmp_path):
    """After cooldown expires, the advisor should no longer recommend skipping."""
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        now_fn=clock,
    )
    tracker.mark_rate_limited("zai", "glm-5.1", duration_s=60)

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)

    # Should be in cooldown now
    result_before = advisor.check("zai", "glm-5.1")
    assert result_before.should_skip is True

    # Advance past cooldown
    clock.advance(seconds=61)

    result_after = advisor.check("zai", "glm-5.1")
    assert result_after.should_skip is False


def test_advisory_includes_cooldown_remaining_seconds(tmp_path):
    """The advisory result should include how many seconds remain in cooldown."""
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        now_fn=clock,
    )
    tracker.mark_rate_limited("zai", "glm-5.1", retry_after=300)

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)

    result = advisor.check("zai", "glm-5.1")
    assert result.should_skip is True
    # Should be close to 300s (may be off by 1 due to rounding)
    assert 295 <= result.cooldown_remaining_s <= 300


def test_advisory_propagates_to_pool_members(tmp_path):
    """If one pool member is rate-limited, all pool members show cooldown."""
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        provider_pools={"opencode-go": [
            ("opencode-go", "deepseek-v4-pro"),
            ("opencode-go", "deepseek-v4-flash"),
        ]},
        now_fn=clock,
    )
    tracker.mark_rate_limited("opencode-go", "deepseek-v4-pro", retry_after=120)

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)

    # Both pool members should be in cooldown
    result_pro = advisor.check("opencode-go", "deepseek-v4-pro")
    result_flash = advisor.check("opencode-go", "deepseek-v4-flash")

    assert result_pro.should_skip is True
    assert result_flash.should_skip is True


def test_advisory_check_without_model_checks_provider_wildcard(tmp_path):
    """If model is None, check the provider-level wildcard cooldown."""
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        now_fn=clock,
    )
    tracker.mark_rate_limited("zai", model=None, duration_s=120)

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)

    result = advisor.check("zai", model=None)
    assert result.should_skip is True


def test_advisory_result_has_advisory_flag_true(tmp_path):
    """Phase 2a results must always have advisory=True (no enforcement)."""
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        now_fn=clock,
    )
    tracker.mark_rate_limited("zai", "glm-5.1", duration_s=60)

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)

    result = advisor.check("zai", "glm-5.1")
    assert result.advisory is True
    assert result.enforced is False


def test_advisory_logs_decision_without_changing_provider(tmp_path):
    """The advisory gate should produce a log line but never alter provider state.

    This test verifies the advisor itself is side-effect-free — it reads
    cooldown state and returns a recommendation. No provider mutation.
    """
    clock = FrozenClock(datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "test.db",
        now_fn=clock,
    )

    from agent.provider_router.pre_flight_advisor import PreFlightAdvisor

    advisor = PreFlightAdvisor(tracker=tracker)

    # No cooldown — should be a clean no-op
    result = advisor.check("deepseek", "deepseek-v4-pro")
    assert result.should_skip is False

    # Mark cooldown, check again — still no side effects on tracker
    tracker.mark_rate_limited("deepseek", "deepseek-v4-pro", retry_after=60)
    result = advisor.check("deepseek", "deepseek-v4-pro")
    assert result.should_skip is True

    # The tracker DB should only have the cooldown row, nothing else added
    import sqlite3
    with sqlite3.connect(tmp_path / "test.db") as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}

    # Advisor should not create any new tables
    assert "cooldowns" in table_names  # from tracker
    assert "usage" in table_names  # from tracker
    # No extra tables from advisor (sqlite_sequence is auto-created by AUTOINCREMENT)
    expected = {"usage", "cooldowns", "error_signals", "config_snapshot", "sqlite_sequence"}
    assert table_names == expected
