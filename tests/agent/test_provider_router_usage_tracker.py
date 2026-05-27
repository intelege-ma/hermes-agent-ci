from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta

import pytest

from agent.provider_router.usage_tracker import UsageTracker


class FrozenClock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs) -> None:
        self.value = self.value + timedelta(**kwargs)


def test_record_creates_sqlite_wal_and_windowed_counters(tmp_path):
    clock = FrozenClock(datetime(2026, 5, 26, 10, 31, 45, tzinfo=timezone.utc))
    tracker = UsageTracker(db_path=tmp_path / "provider_usage.db", now_fn=clock)

    tracker.record("zai", "glm-5.1", tokens_in=100, tokens_out=25)
    tracker.record("zai", "glm-5.1", tokens_in=50, tokens_out=10)

    hourly = tracker.check("zai", "glm-5.1", window_type="hourly")
    daily = tracker.check("zai", "glm-5.1", window_type="daily")
    weekly = tracker.check("zai", "glm-5.1", window_type="weekly")

    assert hourly.tokens_in == 150
    assert hourly.tokens_out == 35
    assert hourly.calls == 2
    assert hourly.window_start == datetime(2026, 5, 26, 10, 0, tzinfo=timezone.utc)
    assert daily.tokens_in == 150
    assert weekly.tokens_out == 35

    with sqlite3.connect(tmp_path / "provider_usage.db") as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode.lower() == "wal"


def test_check_pool_sums_current_pool_members_without_stored_pool_rows(tmp_path):
    tracker = UsageTracker(
        db_path=tmp_path / "provider_usage.db",
        provider_pools={"zai-weekly": [("zai", "glm-4.7"), ("zai", "glm-5.1")]},
        now_fn=lambda: datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
    )

    tracker.record("zai", "glm-4.7", tokens_in=20, tokens_out=5)
    tracker.record("zai", "glm-5.1", tokens_in=30, tokens_out=7)
    tracker.record("deepseek", "deepseek-v4-pro", tokens_in=999, tokens_out=999)

    pool = tracker.check_pool("zai-weekly", window_type="daily")

    assert pool.provider == "zai-weekly"
    assert pool.model == "*"
    assert pool.tokens_in == 50
    assert pool.tokens_out == 12
    assert pool.calls == 2

    with sqlite3.connect(tmp_path / "provider_usage.db") as conn:
        stored_pool_rows = conn.execute(
            "SELECT COUNT(*) FROM usage WHERE provider = ?", ("zai-weekly",)
        ).fetchone()[0]
    assert stored_pool_rows == 0


def test_mark_rate_limited_propagates_to_pool_members_and_expires(tmp_path):
    clock = FrozenClock(datetime(2026, 5, 26, 9, 0, tzinfo=timezone.utc))
    tracker = UsageTracker(
        db_path=tmp_path / "provider_usage.db",
        provider_pools={"zai-weekly": [("zai", "glm-4.7"), ("zai", "glm-5.1")]},
        now_fn=clock,
    )

    tracker.mark_rate_limited("zai", "glm-4.7", duration_s=120, reason="rate_limit")

    assert tracker.is_in_cooldown("zai", "glm-4.7") is True
    assert tracker.is_in_cooldown("zai", "glm-5.1") is True
    assert tracker.is_in_cooldown("deepseek", "deepseek-v4-pro") is False

    clock.advance(seconds=121)

    assert tracker.is_in_cooldown("zai", "glm-4.7") is False
    assert tracker.is_in_cooldown("zai", "glm-5.1") is False


def test_record_error_signal_stores_bounded_relevant_diagnostics(tmp_path):
    tracker = UsageTracker(
        db_path=tmp_path / "provider_usage.db",
        now_fn=lambda: datetime(2026, 5, 26, 13, 0, tzinfo=timezone.utc),
    )
    body = "x" * 900
    headers = {
        "retry-after": "60",
        "x-ratelimit-reset-requests": "60",
        "authorization": "Bearer secret-must-not-be-stored",
    }

    row_id = tracker.record_error_signal(
        "opencode-go",
        "deepseek-v4-pro",
        status_code=429,
        body=body,
        headers=headers,
        detected_as="rate_limit",
    )

    with sqlite3.connect(tmp_path / "provider_usage.db") as conn:
        row = conn.execute(
            "SELECT body_snippet, headers_json, detected_as FROM error_signals WHERE id = ?",
            (row_id,),
        ).fetchone()

    assert len(row[0]) == 500
    assert "retry-after" in row[1]
    assert "x-ratelimit-reset-requests" in row[1]
    assert "authorization" not in row[1]
    assert "secret" not in row[1]
    assert row[2] == "rate_limit"
