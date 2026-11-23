"""Purged Walk-Forward rolling splitter enforcing >= 5 trading-day embargo and label purging.

Guardrail #3: Training and test windows MUST be separated by at least 5 trading days (embargo)
AND any training sample whose forward label horizon reaches into the embargo/test zone is purged.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.core.constants import MIN_EMBARGO_DAYS
from app.core.exceptions import LookAheadBiasError


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    embargo_dates: pd.DatetimeIndex
    purged_dates: pd.DatetimeIndex

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_dates[0]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "train_days": len(self.train_dates),
            "purged_days": len(self.purged_dates),
            "embargo_days": len(self.embargo_dates),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "test_days": len(self.test_dates),
        }


class PurgedWalkForwardSplitter:
    """Rolling Purged Walk-Forward time-series cross-validator."""

    def __init__(
        self,
        train_window: int = 252,
        embargo_days: int = 5,
        test_window: int = 63,
        label_horizon: int = 5,
        step_size: int | None = None,
        adaptive: bool = True,
    ) -> None:
        if embargo_days < MIN_EMBARGO_DAYS:
            raise LookAheadBiasError(
                f"embargo_days={embargo_days} violates guardrail #3 (minimum {MIN_EMBARGO_DAYS} trading days)"
            )
        if label_horizon < 1:
            raise ValueError("label_horizon must be >= 1")
        if train_window < 30 or test_window < 5:
            raise ValueError("train_window >= 30 and test_window >= 5 required")

        self.train_window = train_window
        self.embargo_days = embargo_days
        self.test_window = test_window
        self.label_horizon = label_horizon
        self.step_size = step_size or test_window
        self.adaptive = adaptive

    def split(self, dates: pd.DatetimeIndex) -> list[WalkForwardFold]:
        ordered = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
        n = len(ordered)
        gap = self.embargo_days + self.label_horizon

        train_w = self.train_window
        test_w = self.test_window
        step = self.step_size

        # Gracefully scale train/test window on shorter sample histories while NEVER shrinking embargo
        if self.adaptive and n < train_w + gap + test_w:
            avail = n - gap
            if avail < 60:
                raise ValueError(f"Insufficient trading days ({n}) for purged walk-forward with gap={gap}")
            train_w = max(40, int(avail * 0.55))
            test_w = max(15, min(test_w, int((avail - train_w) / 2)))
            step = test_w

        folds: list[WalkForwardFold] = []
        fold_id = 0
        test_start_idx = train_w + gap
        while test_start_idx < n:
            test_end_idx = min(n, test_start_idx + test_w)
            embargo_start = test_start_idx - self.embargo_days
            purge_start = embargo_start - self.label_horizon
            train_end_excl = purge_start
            train_start_idx = max(0, train_end_excl - train_w)

            train_dates = ordered[train_start_idx:train_end_excl]
            purged_dates = ordered[purge_start:embargo_start]
            embargo_dates = ordered[embargo_start:test_start_idx]
            test_dates = ordered[test_start_idx:test_end_idx]

            # Guardrail invariants
            assert len(embargo_dates) >= MIN_EMBARGO_DAYS
            assert len(train_dates) > 0 and len(test_dates) > 0
            assert train_dates[-1] < purged_dates[0] if len(purged_dates) else train_dates[-1] < embargo_dates[0]
            assert embargo_dates[-1] < test_dates[0]

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_dates=train_dates,
                    test_dates=test_dates,
                    embargo_dates=embargo_dates,
                    purged_dates=purged_dates,
                )
            )
            fold_id += 1
            test_start_idx += step

        return folds
