"""Custom domain exceptions for data, strategy, backtest and risk control."""

from __future__ import annotations


class AlphaStudioError(Exception):
    """Base exception for A-Share Alpha Studio."""


class DataProviderError(AlphaStudioError):
    """Raised when all configured real data providers fail."""


class LookAheadBiasError(AlphaStudioError):
    """Raised when a split or data alignment violates anti-lookahead guardrails."""


class RiskRejectedError(AlphaStudioError):
    """Raised when RiskMiddleware rejects an order."""

    def __init__(self, rule_name: str, message: str, detail: dict | None = None):
        self.rule_name = rule_name
        self.message = message
        self.detail = detail or {}
        super().__init__(f"[{rule_name}] {message}")


class BrokerExecutionError(AlphaStudioError):
    """Raised when a broker adapter fails to execute an order."""
