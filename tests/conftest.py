"""Shared test setup. pytest loads this automatically."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def anyio_backend():
    # Lets tests marked @pytest.mark.anyio run as async functions (anyio ships with mcp).
    return "asyncio"
