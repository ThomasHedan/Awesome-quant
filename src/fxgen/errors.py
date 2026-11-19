"""Module-level error types with user-facing messages."""

from __future__ import annotations


class FxgenError(Exception):
    """Base class for all fxgen errors."""


class DataError(FxgenError):
    """Invalid, missing or inconsistent market data."""


class ConfigError(FxgenError):
    """Invalid configuration."""


class TreeError(FxgenError):
    """Malformed decision tree."""


class HoldoutLockedError(FxgenError):
    """An operation tried to optimise on, or re-read, the locked holdout."""


class StorageError(FxgenError):
    """SQLite storage failure."""
