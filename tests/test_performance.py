"""Performance characteristics the brief sets targets for (§20).

These are shape tests, not benchmarks: they assert that cost grows *linearly* with the book
and that nothing accumulates per batch. Wall-clock numbers on a CI machine prove nothing;
an O(n²) that only shows up at 400 pages proves quite a lot.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from folioai.config import Settings
from folioai.ir import Block, Chapter, Document, ExtractionReport
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient
from folioai.orchestrate import Orchestrator, seed_segments
from folioai.segment import segment_document
from folioai.store import JobStore
from folioai.tags import parse_segments

PASS = {"completeness": 96, "accuracy": 94, "terminology": 95, "fluency": 92, "formatting": 100}


def big_book(paragraphs: int, *, chapters: int = 10) -> Document:
    """A synthetic book of roughly novel proportions."""
    blocks: list[Block] = []
    chapter_defs: list[Chapter] = []
    per_chapter = max(paragraphs // chapters, 1)

    for index in range(paragraphs):
        chapter_no = min(index // per_chapter + 1, chapters)
        blocks.append(
            Block(
                id=f"b{index:05d}",
                kind="paragraph",
                text=(
                    f"Paragraph {index}. The lamp above the door had been broken for a week, "
                    "and nobody in the house had thought to mention it at all that season."
                ),
                chapter_id=f"ch{chapter_no:02d}",
            )
        )
    for chapter_no in range(1, chapters + 1):
        chapter_defs.append(
            Chapter(
                id=f"ch{chapter_no:02d}",
                title=f"Chapter {chapter_no}",
                number=chapter_no,
                block_ids=[b.id for b in blocks if b.chapter_id == f"ch{chapter_no:02d}"],
            )
        )
    return Document(
        source_lang="en",
        target_lang="de",
        blocks=blocks,
        chapters=chapter_defs,
        extraction_report=ExtractionReport(extractor="test"),
    )


def judge(messages: list[Message], model: str) -> str:
    user = next(m["content"] for m in messages if m["role"] == "user")
    if "SOURCE:" in user:
        ids = [line[4:].strip() for line in user.splitlines() if line.startswith("### ")]
        return json.dumps({"scores": [{"segment_id": i, **PASS} for i in ids], "issues": []})
    parsed = parse_segments(user)
    return "\n".join(f'<seg id="{s}">ZIEL {t}</seg>' for s, t in parsed.texts.items())


def test_segmentation_is_linear_in_book_length(settings: Settings) -> None:
    small = big_book(200)
    large = big_book(2000)

    start = time.perf_counter()
    segment_document(small, settings)
    small_time = time.perf_counter() - start

    start = time.perf_counter()
    batches = segment_document(large, settings)
    large_time = time.perf_counter() - start

    assert len(batches) > 10
    # 10x the book should not cost 40x the time. Generous, because this has to hold on a
    # loaded CI machine; an accidental quadratic blows through it by orders of magnitude.
    assert large_time < max(small_time * 40, 0.5)


async def test_context_assembly_does_not_rescan_the_book_per_batch(
    settings: Settings, tmp_path: Path
) -> None:
    """The regression this guards: context lookup used to re-segment the whole document
    once per batch, which is invisible on a fixture and minutes of CPU on a novel."""
    settings.translation.batch_tokens = 200
    settings.translation.concurrency = 8
    document = big_book(600, chapters=6)

    with JobStore(tmp_path / "job.db") as store:
        store.create_job(
            job_id="perf",
            source_path=tmp_path / "book.pdf",
            source_sha256="perf",
            config={},
            source_lang="en",
            target_lang="de",
        )
        seed_segments(store, "perf", document)
        orchestrator = Orchestrator(
            client=FakeLLMClient(judge),
            settings=settings,
            document=document,
            target_lang="de",
            store=store,
            job_id="perf",
        )
        start = time.perf_counter()
        outcomes = await orchestrator.run()
        elapsed = time.perf_counter() - start

    assert len(outcomes) == 600
    assert elapsed < 30.0  # a quadratic here takes minutes, not seconds


async def test_memory_does_not_grow_with_attempt_history(
    settings: Settings, tmp_path: Path
) -> None:
    """§20: attempts are persisted per batch, never held for the whole book."""
    settings.translation.batch_tokens = 150
    document = big_book(300, chapters=3)

    with JobStore(tmp_path / "job.db") as store:
        store.create_job(
            job_id="perf",
            source_path=tmp_path / "book.pdf",
            source_sha256="perf",
            config={},
            source_lang="en",
            target_lang="de",
        )
        seed_segments(store, "perf", document)
        orchestrator = Orchestrator(
            client=FakeLLMClient(judge),
            settings=settings,
            document=document,
            target_lang="de",
            store=store,
            job_id="perf",
        )
        await orchestrator.run()

        # Every attempt is in the database, and the orchestrator keeps one outcome per
        # segment rather than a growing pile of response bodies.
        rows = store.conn.execute(
            "SELECT COUNT(*) AS n FROM attempts WHERE job_id = 'perf'"
        ).fetchone()
        assert rows["n"] == 300
        assert len(orchestrator.outcomes) == 300
        assert all(
            len(o.attempts) <= settings.retry.max_attempts for o in orchestrator.outcomes.values()
        )


@pytest.mark.parametrize("count", [50, 500])
def test_export_scales_with_the_book(count: int, tmp_path: Path, settings: Settings) -> None:
    from folioai.export import export_document
    from folioai.render.base import RenderContext

    start = time.perf_counter()
    result = export_document(
        big_book(count),
        tmp_path / str(count),
        formats=["md", "html", "txt"],
        context=RenderContext(),
        settings=settings,
    )
    elapsed = time.perf_counter() - start
    assert len(result.files) == 3
    assert elapsed < 10.0  # §20 allows 30s per format for a whole novel


def test_extraction_of_the_fixture_is_fast(
    sample_pdfs: dict[str, Path], settings: Settings
) -> None:
    """§20 allows 60s for a 300-page book; a 4-page fixture should be near-instant."""
    from folioai.extract.pipeline import extract_document

    start = time.perf_counter()
    result = extract_document(sample_pdfs["clean_book.pdf"], settings)
    elapsed = time.perf_counter() - start
    assert result.document.blocks
    assert elapsed < 5.0
