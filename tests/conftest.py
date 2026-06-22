"""Shared pytest setup.

Puts ``src/`` (for ``import hindi_llm``) and ``scripts/`` (so tests can import
the pipeline scripts directly) on the path, and exposes a couple of paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"

for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_corpus() -> Path:
    return REPO_ROOT / "data" / "sample_hindi.txt"
