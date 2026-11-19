"""Logging configuration shared by the CLI, the pipeline and the MCP server."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging to stderr (stdout is reserved for MCP stdio)."""
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )
