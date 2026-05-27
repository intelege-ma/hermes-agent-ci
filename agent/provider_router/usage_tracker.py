from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from hermes_constants import get_hermes_home

WindowType = str
ProviderModel = tuple[str, str]

_RELEVANT_ERROR_HEADERS = {
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-requests-1h",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-limit-tokens-1h",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-requests-1h",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-remaining-tokens-1h",
    "x-ratelimit-reset",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-requests-1h",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset-tokens-1h",
}


@dataclass(frozen=True)
class UsageSnapshot:
    provider: str
    model: str
    window_type: WindowType
    window_start: datetime
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class UsageTracker:
    """SQLite-backed provider usage tracker for LOG-ONLY routing data.

    This module deliberately does not make routing decisions or alter live
    provider selection. It records enough data for later pre-flight routing:
    windowed usage counters, cooldowns, and bounded error diagnostics.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        provider_pools: Optional[Mapping[str, Iterable[ProviderModel]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser() if db_path is not None else self._default_db_path()
        self.provider_pools: dict[str, tuple[ProviderModel, ...]] = {
            str(name): tuple((str(provider), str(model)) for provider, model in members)
            for name, members in (provider_pools or {}).items()
        }
        self._pool_by_member: dict[ProviderModel, str] = {}
        for pool_name, members in self.provider_pools.items():
            for member in members:
                self._pool_by_member[member] = pool_name
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._init_db()

    @staticmethod
    def _default_db_path() -> Path:
        return Path(get_hermes_home()) / "state" / "provider_usage.db"

    def record(
        self,
        provider: str,
        model: str,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        calls: int = 1,
        at: datetime | None = None,
    ) -> None:
        provider = self._clean(provider)
        model = self._clean(model)
        if not provider or not model:
            raise ValueError("provider and model are required")
        when = self._coerce_dt(at) if at is not None else self._now()
        tokens_in = max(0, int(tokens_in or 0))
        tokens_out = max(0, int(tokens_out or 0))
        calls = max(0, int(calls or 0))

        with self._connect() as conn:
            for window_type in ("hourly", "daily", "weekly"):
                window_start = self._window_start(when, window_type)
                conn.execute(
                    """
                    INSERT INTO usage (
                        provider, model, window_type, window_start,
                        tokens_in, tokens_out, calls
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, model, window_type, window_start)
                    DO UPDATE SET
                        tokens_in = tokens_in + excluded.tokens_in,
                        tokens_out = tokens_out + excluded.tokens_out,
                        calls = calls + excluded.calls
                    """,
                    (
                        provider,
                        model,
                        window_type,
                        self._to_iso(window_start),
                        tokens_in,
                        tokens_out,
                        calls,
                    ),
                )

    def check(self, provider: str, model: str, *, window_type: WindowType = "daily") -> UsageSnapshot:
        provider = self._clean(provider)
        model = self._clean(model)
        window_start = self._window_start(self._now(), window_type)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT tokens_in, tokens_out, calls
                FROM usage
                WHERE provider = ? AND model = ? AND window_type = ? AND window_start = ?
                """,
                (provider, model, window_type, self._to_iso(window_start)),
            ).fetchone()
        if row is None:
            return UsageSnapshot(provider, model, window_type, window_start)
        return UsageSnapshot(provider, model, window_type, window_start, int(row[0]), int(row[1]), int(row[2]))

    def check_pool(self, pool_name: str, *, window_type: WindowType = "daily") -> UsageSnapshot:
        pool_name = self._clean(pool_name)
        members = self.provider_pools.get(pool_name, ())
        window_start = self._window_start(self._now(), window_type)
        if not members:
            return UsageSnapshot(pool_name, "*", window_type, window_start)

        placeholders = ",".join("(?, ?)" for _ in members)
        params: list[str] = []
        for provider, model in members:
            params.extend([provider, model])
        params.extend([window_type, self._to_iso(window_start)])

        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM(tokens_in), 0),
                       COALESCE(SUM(tokens_out), 0),
                       COALESCE(SUM(calls), 0)
                FROM usage
                WHERE (provider, model) IN ({placeholders})
                  AND window_type = ?
                  AND window_start = ?
                """,
                tuple(params),
            ).fetchone()
        return UsageSnapshot(pool_name, "*", window_type, window_start, int(row[0]), int(row[1]), int(row[2]))

    def mark_rate_limited(
        self,
        provider: str,
        model: str | None = None,
        *,
        duration_s: int | float | None = None,
        retry_after: int | float | None = None,
        reason: str = "rate_limit",
    ) -> None:
        provider = self._clean(provider)
        model = self._clean(model) if model is not None else "*"
        seconds = retry_after if retry_after is not None else duration_s
        seconds = float(seconds if seconds is not None else 60)
        until = self._now().timestamp() + max(0.0, seconds)
        retry_after_int = int(float(retry_after)) if retry_after is not None else None

        targets = self._cooldown_targets(provider, model)
        with self._connect() as conn:
            for target_provider, target_model in targets:
                conn.execute(
                    """
                    INSERT INTO cooldowns (provider, model, until, reason, retry_after)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(provider, model)
                    DO UPDATE SET
                        until = excluded.until,
                        reason = excluded.reason,
                        retry_after = excluded.retry_after
                    """,
                    (target_provider, target_model, self._to_iso(datetime.fromtimestamp(until, tz=timezone.utc)), reason, retry_after_int),
                )

    def is_in_cooldown(self, provider: str, model: str | None = None) -> bool:
        provider = self._clean(provider)
        model = self._clean(model) if model is not None else "*"
        now_iso = self._to_iso(self._now())
        candidates = [(provider, model), (provider, "*")]
        pool_name = self._pool_by_member.get((provider, model))
        if pool_name:
            candidates.append((pool_name, "*"))

        with self._connect() as conn:
            for candidate_provider, candidate_model in candidates:
                row = conn.execute(
                    "SELECT until FROM cooldowns WHERE provider = ? AND model = ?",
                    (candidate_provider, candidate_model),
                ).fetchone()
                if row is None:
                    continue
                if str(row[0]) > now_iso:
                    return True
                conn.execute(
                    "DELETE FROM cooldowns WHERE provider = ? AND model = ?",
                    (candidate_provider, candidate_model),
                )
        return False

    def record_error_signal(
        self,
        provider: str,
        model: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
        headers: Mapping[str, object] | None = None,
        detected_as: str = "unknown",
        confirmed: bool = False,
    ) -> int:
        body_snippet = str(body or "")[:500]
        headers_json = json.dumps(self._safe_error_headers(headers), sort_keys=True)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO error_signals (
                    provider, model, status_code, body_snippet, headers_json,
                    detected_as, confirmed, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._clean(provider),
                    self._clean(model),
                    status_code,
                    body_snippet,
                    headers_json,
                    self._clean(detected_as) or "unknown",
                    1 if confirmed else 0,
                    self._to_iso(self._now()),
                ),
            )
            lastrowid = cursor.lastrowid
            if lastrowid is None:
                raise RuntimeError("failed to record provider router error signal")
            return int(lastrowid)

    def _cooldown_targets(self, provider: str, model: str) -> tuple[ProviderModel, ...]:
        if model == "*":
            return ((provider, "*"),)
        pool_name = self._pool_by_member.get((provider, model))
        if pool_name:
            return self.provider_pools[pool_name]
        return ((provider, model),)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    window_type TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    calls INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (provider, model, window_type, window_start)
                );

                CREATE TABLE IF NOT EXISTS cooldowns (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    until TEXT NOT NULL,
                    reason TEXT,
                    retry_after INTEGER,
                    PRIMARY KEY (provider, model)
                );

                CREATE TABLE IF NOT EXISTS error_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status_code INTEGER,
                    body_snippet TEXT,
                    headers_json TEXT,
                    detected_as TEXT,
                    confirmed INTEGER NOT NULL DEFAULT 0,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS config_snapshot (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _now(self) -> datetime:
        return self._coerce_dt(self._now_fn())

    @staticmethod
    def _coerce_dt(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _window_start(cls, value: datetime, window_type: WindowType) -> datetime:
        value = cls._coerce_dt(value)
        if window_type == "hourly":
            return value.replace(minute=0, second=0, microsecond=0)
        if window_type == "daily":
            return value.replace(hour=0, minute=0, second=0, microsecond=0)
        if window_type == "weekly":
            day_start = value.replace(hour=0, minute=0, second=0, microsecond=0)
            return day_start - timedelta(days=day_start.weekday())
        raise ValueError(f"unsupported window_type: {window_type}")

    @staticmethod
    def _to_iso(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _clean(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _safe_error_headers(headers: Mapping[str, object] | None) -> dict[str, str]:
        if not headers:
            return {}
        safe: dict[str, str] = {}
        for key, value in headers.items():
            normalized = str(key).strip().lower()
            if normalized in _RELEVANT_ERROR_HEADERS:
                safe[normalized] = str(value)[:200]
        return safe
