"""Shared fixtures. Synthetic PDFs are built once per session into a temp directory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from make_pdfs import build_all

from folioai.config import Settings


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every synthetic fixture PDF, built reproducibly with ReportLab."""
    directory = tmp_path_factory.mktemp("pdfs")
    return build_all(directory)


@pytest.fixture
def settings() -> Settings:
    """Packaged defaults, with no user or project config layered on."""
    return Settings()
