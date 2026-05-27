"""Phase 2a: Advisory pre-flight gate.

Reads tracker cooldown state and returns a routing recommendation WITHOUT
enforcing it. This is the LOG-ONLY measurement phase — we log what *would*
happen during the soak period to assess false positive rates before Phase 2b
enforcement.

Design decisions (user-approved 2026-05-27):
- Conservative: only recommend skip if cooldown `until` is in the future
- Advisory-only: never alters provider selection
- No permanent exile: if fallback also fails, original gets retried (Phase 2b)
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from agent.provider_router.usage_tracker import UsageTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvisoryResult:
    """Result of an advisory pre-flight check.

    Attributes:
        should_skip: Whether the provider should be skipped (advisory).
        reason: Human-readable reason for the recommendation.
        cooldown_remaining_s: Seconds until cooldown expires. 0 if not in cooldown.
        advisory: Always True in Phase 2a (no enforcement).
        enforced: Always False in Phase 2a.
    """
    should_skip: bool
    reason: str = ""
    cooldown_remaining_s: float = 0.0
    advisory: bool = True
    enforced: bool = False


class PreFlightAdvisor:
    """Advisory pre-flight gate — checks cooldown state without enforcement.

    Usage:
        advisor = PreFlightAdvisor(tracker=my_tracker)
        result = advisor.check("opencode-go", "deepseek-v4-pro")
        if result.should_skip:
            # LOG-ONLY: record what would happen, do NOT skip
            logger.info("Advisory skip: %s", result.reason)
    """

    def __init__(self, tracker: UsageTracker) -> None:
        self._tracker = tracker

    def check(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> AdvisoryResult:
        """Check whether a provider/model should be skipped (advisory).

        Args:
            provider: Provider name (e.g. "opencode-go").
            model: Model name (e.g. "deepseek-v4-pro"). None checks wildcard.

        Returns:
            AdvisoryResult with recommendation and cooldown details.
        """
        in_cooldown = self._tracker.is_in_cooldown(provider, model)

        if not in_cooldown:
            logger.debug(
                "Advisory check: %s/%s — healthy (no cooldown)",
                provider, model or "*",
            )
            return AdvisoryResult(should_skip=False)

        remaining = self._get_cooldown_remaining(provider, model)
        model_str = model or "*"
        reason = (
            f"Provider {provider}/{model_str} in cooldown "
            f"({remaining:.0f}s remaining)"
        )

        logger.info(
            "Advisory skip recommended: %s — advisory=True, enforced=False",
            reason,
        )

        return AdvisoryResult(
            should_skip=True,
            reason=reason,
            cooldown_remaining_s=remaining,
        )

    def _get_cooldown_remaining(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> float:
        """Get seconds remaining in cooldown for a provider/model.

        Returns 0.0 if not in cooldown or if the until timestamp cannot be parsed.
        """
        model = model if model is not None else "*"
        try:
            with self._tracker._connect() as conn:
                # Check both specific model and wildcard
                candidates = [(provider, model)]
                if model != "*":
                    candidates.append((provider, "*"))
                pool_name = self._tracker._pool_by_member.get((provider, model))
                if pool_name:
                    candidates.append((pool_name, "*"))

                for candidate_provider, candidate_model in candidates:
                    row = conn.execute(
                        "SELECT until FROM cooldowns WHERE provider = ? AND model = ?",
                        (candidate_provider, candidate_model),
                    ).fetchone()
                    if row is not None:
                        until_str = str(row[0])
                        try:
                            until_dt = datetime.fromisoformat(
                                until_str.replace("Z", "+00:00")
                            )
                            if until_dt.tzinfo is None:
                                until_dt = until_dt.replace(tzinfo=timezone.utc)
                        except (ValueError, TypeError):
                            continue

                        now = self._tracker._now()
                        remaining = (until_dt - now).total_seconds()
                        if remaining > 0:
                            return remaining
        except Exception:
            pass

        return 0.0
