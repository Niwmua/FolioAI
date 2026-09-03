"""Generate the synthetic test PDFs with ReportLab.

Committed as a *script*, not as binaries (brief §19): every fixture is reproducible, and a
reviewer can see exactly what pathology each one encodes rather than trusting an opaque blob.

Run directly (``uv run python tests/fixtures/make_pdfs.py``) or let the ``sample_pdfs``
fixture build them into a temp directory.

Each PDF isolates a known way PDFs ruin text:

* ``clean_book.pdf``      -- the happy path, with an outline, chapters and running heads.
* ``hyphenated.pdf``      -- line-break hyphens, both joinable and genuinely hyphenated.
* ``furniture.pdf``       -- running heads and footers with changing page numbers.
* ``dropcap.pdf``         -- an oversized initial letter starting a chapter.
* ``footnotes.pdf``       -- superscript markers with small-type notes at the page foot.
* ``two_column.pdf``      -- a two-column layout that breaks naive top-to-bottom reading.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A5
from reportlab.pdfgen import canvas

BODY_SIZE = 11.0
BODY_FONT = "Times-Roman"
HEADING_FONT = "Times-Bold"
MARGIN = 45.0
LEADING = 15.0


def _text_block(
    pdf: canvas.Canvas,
    lines: list[str],
    *,
    x: float,
    y: float,
    size: float = BODY_SIZE,
    font: str = BODY_FONT,
    leading: float = LEADING,
) -> float:
    pdf.setFont(font, size)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _running_head(
    pdf: canvas.Canvas, page_no: int, title: str, width: float, height: float
) -> None:
    pdf.setFont(BODY_FONT, 8)
    pdf.drawString(MARGIN, height - 28, title)
    pdf.drawRightString(width - MARGIN, height - 28, "A Test Book")
    # ASCII decoration on purpose: the base-14 Times font cannot encode an em dash, and a
    # fixture full of replacement characters tests the extractor against a bug of its own
    # making rather than against a real book.
    pdf.drawCentredString(width / 2, 24, f"- {page_no} -")


def clean_book(path: Path) -> Path:
    """Two chapters, an outline, running heads, ordinary paragraphs."""
    width, height = A5
    pdf = canvas.Canvas(str(path), pagesize=A5)

    chapters = [
        (
            "Chapter One",
            [
                "The lamp above the door had been broken for a week, and nobody in the",
                "house had thought to mention it. Mrs Ainsworth noticed it on the Tuesday,",
                "on her way back from the market with a basket she could barely carry.",
            ],
            [
                "She said nothing about the lamp, and nothing about a great many other",
                "things that year, until the habit had begun to feel less like discretion",
                "and more like a wall she had built without ever meaning to.",
            ],
        ),
        (
            "Chapter Two",
            [
                "Rain came in from the east and stayed for three days. The gutters",
                "overflowed, the yard turned to a shallow brown lake, and the dog refused",
                "to go outside at all, which everyone agreed was sensible of him.",
            ],
            [
                "On the fourth morning the sun returned as though nothing at all had",
                "happened, which is a talent the weather has and one that people would",
                "do well to learn: to arrive without apology and be forgiven at once.",
            ],
        ),
    ]

    # Each chapter runs over two pages, so the fixture exercises the things that only go
    # wrong at a page boundary: running-head statistics (which need four or more pages to
    # mean anything) and a paragraph that continues across the break.
    page_no = 0
    for index, (title, para_one, para_two) in enumerate(chapters, start=1):
        page_no += 1
        _running_head(pdf, page_no, title, width, height)
        y = height - 90
        pdf.setFont(HEADING_FONT, 18)
        pdf.drawString(MARGIN, y, title)
        pdf.bookmarkPage(f"ch{index}")
        pdf.addOutlineEntry(title, f"ch{index}", level=0)
        y -= 34
        y = _text_block(pdf, para_one, x=MARGIN, y=y)
        y -= LEADING * 0.6
        # This paragraph is cut off mid-sentence and resumes on the next page in lower case.
        _text_block(pdf, para_two[:1], x=MARGIN + 12, y=y)
        pdf.showPage()

        page_no += 1
        _running_head(pdf, page_no, title, width, height)
        y = height - 90
        _text_block(pdf, para_two[1:], x=MARGIN, y=y)
        pdf.showPage()

    pdf.showOutline()
    pdf.save()
    return path


def hyphenated(path: Path) -> Path:
    """Line-break hyphens: one that must be joined, one that must be kept."""
    width, height = A5
    pdf = canvas.Canvas(str(path), pagesize=A5)
    _running_head(pdf, 1, "Chapter One", width, height)
    y = height - 90
    pdf.setFont(HEADING_FONT, 18)
    pdf.drawString(MARGIN, y, "Chapter One")
    y -= 34
    # "extraordinary" is split across lines and must be rejoined.
    # "well-being" is genuinely hyphenated and recurs hyphenated, so it must be kept.
    y = _text_block(
        pdf,
        [
            "The committee had reached an extraor-",
            "dinary conclusion, and every member of it",
            "understood that the well-",
            "being of the town now rested on a vote",
            "nobody wanted to cast.",
        ],
        x=MARGIN,
        y=y,
    )
    y -= LEADING
    _text_block(
        pdf,
        [
            "Later they spoke of the well-being of the",
            "town again, and of the extraordinary",
            "weather, and of nothing else at all.",
        ],
        x=MARGIN,
        y=y,
    )
    pdf.showPage()
    pdf.save()
    return path


def furniture(path: Path) -> Path:
    """Six pages of identical running heads and changing page numbers."""
    width, height = A5
    pdf = canvas.Canvas(str(path), pagesize=A5)
    for page_no in range(1, 7):
        _running_head(pdf, page_no, "The Long Afternoon", width, height)
        y = height - 90
        _text_block(
            pdf,
            [
                f"This is the body text of page {page_no}. It says something ordinary",
                "and then continues onto a second line so that the paragraph has",
                "enough substance to be recognised as prose rather than furniture.",
            ],
            x=MARGIN,
            y=y,
        )
        pdf.showPage()
    pdf.save()
    return path


def dropcap(path: Path) -> Path:
    """An oversized initial letter that must rejoin the word it starts."""
    width, height = A5
    pdf = canvas.Canvas(str(path), pagesize=A5)
    _running_head(pdf, 1, "Chapter One", width, height)
    y = height - 90
    pdf.setFont(HEADING_FONT, 18)
    pdf.drawString(MARGIN, y, "Chapter One")
    y -= 40

    pdf.setFont(BODY_FONT, 30)
    pdf.drawString(MARGIN, y, "W")
    pdf.setFont(BODY_FONT, BODY_SIZE)
    pdf.drawString(MARGIN + 22, y, "hen the bell rang for the second time that")
    y -= LEADING
    _text_block(
        pdf,
        [
            "evening, nobody moved. The sound went out across the",
            "fields and came back thinner, as though the distance had",
            "taken something from it on the way.",
        ],
        x=MARGIN,
        y=y,
    )
    pdf.showPage()
    pdf.save()
    return path


def footnotes(path: Path) -> Path:
    """Superscript markers in the body, small-type notes at the foot of the page."""
    width, height = A5
    pdf = canvas.Canvas(str(path), pagesize=A5)
    _running_head(pdf, 1, "Chapter One", width, height)
    y = height - 90
    pdf.setFont(HEADING_FONT, 18)
    pdf.drawString(MARGIN, y, "Chapter One")
    y -= 34

    sentence = "The treaty was signed in the spring of that year"
    pdf.setFont(BODY_FONT, BODY_SIZE)
    pdf.drawString(MARGIN, y, sentence)
    text_width = pdf.stringWidth(sentence, BODY_FONT, BODY_SIZE)
    pdf.setFont(BODY_FONT, 6.5)
    pdf.drawString(MARGIN + text_width, y + 4, "1")
    y -= LEADING
    y = _text_block(
        pdf,
        [
            "and the terms were published a fortnight later, to general",
            "indifference and a single furious letter in the county paper.",
        ],
        x=MARGIN,
        y=y,
    )

    # The note itself: small type at the bottom of the page.
    _text_block(
        pdf,
        [
            "1. The date is disputed; some sources give the following autumn,",
            "   though the parish record is unambiguous.",
        ],
        x=MARGIN,
        y=70,
        size=7.5,
        leading=10,
    )
    pdf.showPage()
    pdf.save()
    return path


def two_column(path: Path) -> Path:
    """Two columns whose naive top-to-bottom reading interleaves the sentences."""
    width, height = A5
    pdf = canvas.Canvas(str(path), pagesize=A5)
    _running_head(pdf, 1, "Two Columns", width, height)
    gutter = width / 2 + 12
    left_x = MARGIN
    y_start = height - 90

    # Lines are kept short so the gutter between the columns is a realistic width; a book
    # laid out with a 3pt gutter does not exist, and testing against one proves nothing.
    left = [
        "The first column begins",
        "here and continues for",
        "several lines, each of them",
        "part of one argument, none",
        "of which makes any sense",
        "beside its neighbour.",
    ]
    right = [
        "The second column is a",
        "separate thread. It starts",
        "level with the first and must",
        "not be interleaved with it",
        "when the text is put back",
        "into reading order.",
    ]
    _text_block(pdf, left, x=left_x, y=y_start, size=9)
    _text_block(pdf, right, x=gutter, y=y_start, size=9)
    pdf.showPage()
    pdf.save()
    return path


BUILDERS = {
    "clean_book.pdf": clean_book,
    "hyphenated.pdf": hyphenated,
    "furniture.pdf": furniture,
    "dropcap.pdf": dropcap,
    "footnotes.pdf": footnotes,
    "two_column.pdf": two_column,
}


def build_all(directory: Path) -> dict[str, Path]:
    """Build every fixture into ``directory`` and return name -> path."""
    directory.mkdir(parents=True, exist_ok=True)
    return {name: builder(directory / name) for name, builder in BUILDERS.items()}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent / "pdfs"
    built = build_all(here)
    for name, path in built.items():
        print(f"{name:20s} {path}")
