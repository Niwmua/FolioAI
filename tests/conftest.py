"""Shared fixtures. Synthetic PDFs are built once per session into a temp directory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from make_pdfs import build_all

from folioai.config import Settings, packaged_settings


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
