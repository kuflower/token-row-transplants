"""Shared pytest configuration for portable temporary paths."""

from __future__ import annotations

import tempfile
from pathlib import Path


def pytest_configure() -> None:
    """Use the physical temporary root on platforms with an aliased path."""

    temporary_root = Path(tempfile.gettempdir())
    resolved_root = temporary_root.resolve(strict=True)
    if resolved_root != temporary_root:
        tempfile.tempdir = str(resolved_root)
