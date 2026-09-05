"""Shared fixtures. Synthetic PDFs are built once per session into a temp directory."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from make_pdfs import build_all

from folioai.config import Settings, packaged_settings


@pytest.fixture(scope="session", autouse=True)
def hermetic_config(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Run every test against a copy of the shipped config, never the developer's own.

    Once ``.env`` became authoritative for models, endpoints and paths, the suite started
    reading whatever the person running it happened to have configured -- so a test asserting
    the shipped default model would pass on one machine and fail on the next. Tests get a
    copy of ``config/`` with no ``.env`` in it, and no ``FOLIOAI_*`` variables set.

    Tests that need their own values set them with monkeypatch as usual; this only removes
    the ambient ones.
    """
    import os
    import shutil

    from folioai import env as env_module

    source = Path(__file__).resolve().parent.parent / "config"
    target = tmp_path_factory.mktemp("packaged-config")
    shutil.copy(source / "default.yaml", target / "default.yaml")
    if (source / "profiles").is_dir():
        shutil.copytree(source / "profiles", target / "profiles")

    saved = {k: v for k, v in os.environ.items() if k.startswith("FOLIOAI_")}
    for name in saved:
        os.environ.pop(name, None)
    os.environ["FOLIOAI_CONFIG_DIR"] = str(target)
    env_module.reset_for_tests()

    try:
        yield target
    finally:
        for name in [k for k in os.environ if k.startswith("FOLIOAI_")]:
            os.environ.pop(name, None)
        os.environ.update(saved)
        env_module.reset_for_tests()


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every synthetic fixture PDF, built reproducibly with ReportLab."""
    directory = tmp_path_factory.mktemp("pdfs")
    return build_all(directory)


@pytest.fixture
def settings() -> Settings:
    """The shipped defaults, with no user or project config layered on.

    Reads config/default.yaml rather than constructing a bare Settings(), so tests see the
    same pricing table and thresholds a user gets -- and a change to the shipped defaults
    shows up as a test failure rather than a surprise in production.
    """
    return packaged_settings()


@pytest.fixture
def tiny_doc():
    """A small two-chapter document for engine tests."""
    from folioai.ir import Block, Chapter, Document, ExtractionReport

    blocks = []
    for index in range(12):
        chapter = "ch01" if index < 6 else "ch02"
        blocks.append(
            Block(
                id=f"b{index:04d}",
                kind="heading" if index in (0, 6) else "paragraph",
                level=1 if index in (0, 6) else None,
                text=(
                    f"Chapter {index // 6 + 1}"
                    if index in (0, 6)
                    else f"Paragraph {index} of the book. It says something ordinary, at "
                    f"enough length to look like real prose rather than a stub."
                ),
                chapter_id=chapter,
                source_pages=[index // 3 + 1],
            )
        )
    return Document(
        source_lang="en",
        target_lang="de",
        title="A Test Book",
        author="Nobody",
        chapters=[
            Chapter(id="ch01", title="Chapter 1", number=1, block_ids=[b.id for b in blocks[:6]]),
            Chapter(id="ch02", title="Chapter 2", number=2, block_ids=[b.id for b in blocks[6:]]),
        ],
        blocks=blocks,
        extraction_report=ExtractionReport(extractor="test"),
    )


@pytest.fixture
def job_store(tmp_path: Path):
    """An open JobStore with one job row, for orchestration tests."""
    from folioai.store import JobStore

    with JobStore(tmp_path / "job.db") as store:
        store.create_job(
            job_id="job1",
            source_path=tmp_path / "book.pdf",
            source_sha256="abc123",
            config={},
            source_lang="en",
            target_lang="de",
        )
        yield store
