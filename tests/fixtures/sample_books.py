"""A small book in IR form, shared by the renderer and validator tests.

Lives here rather than in a test module because ``conftest.py`` puts this directory on the
path, which is the one way to share a helper between test files that works under every
pytest import mode.
"""

from __future__ import annotations

from folioai.ir import Block, Chapter, Document, ExtractionReport


def build_book(target: str = "de", *, title: str = "A Test Book") -> Document:
    """Two chapters covering every block kind a renderer has to handle."""
    blocks = [
        Block(id="b0000", kind="heading", level=1, text="Kapitel Eins", chapter_id="ch01"),
        Block(
            id="b0001",
            kind="paragraph",
            text="Ein *kursiver* Satz mit **Betonung**.",
            chapter_id="ch01",
        ),
        Block(id="b0002", kind="blockquote", text="Ein Zitat.", chapter_id="ch01"),
        Block(
            id="b0003",
            kind="scene_break",
            text="* * *",
            chapter_id="ch01",
            translate=False,
        ),
        Block(id="b0004", kind="heading", level=1, text="Kapitel Zwei", chapter_id="ch02"),
        Block(
            id="b0005",
            kind="paragraph",
            text="Noch ein Absatz mit einer Fußnote[^1].",
            chapter_id="ch02",
            footnote_refs=["1"],
        ),
        Block(
            id="b0006",
            kind="footnote",
            text="Die Fußnote selbst.",
            chapter_id="ch02",
            meta={"label": "1"},
        ),
    ]
    return Document(
        source_lang="en",
        target_lang=target,
        title=title,
        author="Nobody",
        chapters=[
            Chapter(
                id="ch01",
                title="Kapitel Eins",
                number=1,
                block_ids=["b0000", "b0001", "b0002", "b0003"],
            ),
            Chapter(
                id="ch02",
                title="Kapitel Zwei",
                number=2,
                block_ids=["b0004", "b0005", "b0006"],
            ),
        ],
        blocks=blocks,
        extraction_report=ExtractionReport(extractor="test", page_count=4, block_count=7),
    )
