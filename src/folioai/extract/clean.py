"""Cleaning: turning PDF line soup into paragraphs worth translating (brief §4.3).

Books extract badly by default. Every rule here is independently toggleable, independently
tested, and writes to an audit log, because a cleaner that silently eats a line is worse
than no cleaner at all -- the whole point of this pipeline is that loss is visible.

Order matters and is deliberate:

1. Unicode and ligature normalisation, so every later comparison sees the same characters.
2. Running header/footer and page-number removal, which needs whole-book statistics.
3. Footnote separation, *before* reflow -- a footnote left inline corrupts a sentence and
   the model will faithfully translate the corruption.
4. Paragraph reflow, using PyMuPDF's y-gaps where available rather than guessing from text.
5. Drop-cap repair.
6. De-hyphenation, which needs a whole-book frequency map and so runs last (PLAN §2.3).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import pairwise
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ..logging_setup import get_logger
from .base import RawDocument, RawLine, RawPage

if TYPE_CHECKING:
    from ..config import CleaningConfig, Settings

log = get_logger(__name__)

# -- normalisation ----------------------------------------------------------------

#: Ligatures that are typography, not letters. Kept out: æ, œ, ß -- those are real letters.
LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
    "Ĳ": "IJ",
    "ĳ": "ij",
}
#: Spaces that should behave like a plain space for our purposes.
SPACE_CHARS = "    ⁠"
#: Characters that carry no information and only break comparisons.
ZERO_WIDTH = "­​‌‍﻿"

_ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")
_PAGE_NUMBER_RE = re.compile(
    r"""^\s*(?:
        [-–—|\[\(]*\s*\d{1,4}\s*[-–—|\]\)]*        # 47, - 47 -, [47], | 47 |
      | [-–—|\[\(]*\s*[ivxlcdmIVXLCDM]{1,7}\s*[-–—|\]\)]*   # roman numerals
      | (?:page|p\.|pg\.?|seite|página|pagina)\s*\d{1,4}
      | \d{1,4}\s*(?:of|/)\s*\d{1,4}
    )\s*$""",
    re.VERBOSE | re.IGNORECASE,
)
_TERMINAL_PUNCT = tuple(".!?…\"'”’»)』」")
_BULLET_RE = re.compile(r"^\s*([•‣◦⁃∙*+\-–—]|\(?[a-z0-9]{1,3}[.)])\s+")
_FIGURE_RE = re.compile(r"^\s*(fig(?:ure)?\.?|table|plate|abb\.?|tabelle)\s*\d+", re.IGNORECASE)
_FOOTNOTE_MARK_RE = re.compile(r"^\s*(\d{1,3}|[*†‡§¶]{1,3})[.):\s]\s*")


def normalize_text(text: str) -> str:
    """Ligatures to ASCII, odd spaces to spaces, zero-width gone, NFC.

    Smart quotes are deliberately preserved: they carry dialogue structure, and flattening
    them loses information the target language's punctuation conventions need.
    """
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)
    for char in ZERO_WIDTH:
        text = text.replace(char, "")
    for char in SPACE_CHARS:
        text = text.replace(char, " ")
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def normalize_document(raw: RawDocument) -> None:
    """Apply :func:`normalize_text` to every span, in place."""
    for line in raw.lines():
        for span in line.spans:
            span.text = normalize_text(span.text)


# -- audit ------------------------------------------------------------------------


class StrippedLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int
    text: str
    reason: str


class DehyphenationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: str
    joined: bool
    joined_form_count: int
    hyphenated_form_count: int
    page: int


class CleaningAudit(BaseModel):
    """What the cleaner did, so §15's extraction audit can show it."""

    model_config = ConfigDict(extra="forbid")

    stripped: list[StrippedLine] = Field(default_factory=list)
    dehyphenations: list[DehyphenationDecision] = Field(default_factory=list)
    drop_caps_repaired: list[str] = Field(default_factory=list)
    footnotes_extracted: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def stripped_headers(self) -> int:
        return sum(1 for s in self.stripped if s.reason.startswith("running"))

    @property
    def stripped_page_numbers(self) -> int:
        return sum(1 for s in self.stripped if s.reason == "page-number")

    @property
    def joined_hyphens(self) -> int:
        return sum(1 for d in self.dehyphenations if d.joined)


# -- running headers and footers ---------------------------------------------------


def furniture_key(text: str) -> str:
    """Normalised form for comparing page furniture across pages.

    Digits are masked so that ``Page 47`` and ``Page 48`` compare equal (D-16) -- without
    that, a running head with a page number in it looks unique on every page and survives.
    """
    lowered = normalize_text(text).strip().lower()
    masked = _DIGITS_RE.sub("#", lowered)
    masked = re.sub(r"\b[ivxlcdm]+\b", "#", masked)
    return re.sub(r"[^\w#]+", " ", masked).strip()


def _similar(a: str, b: str, threshold: float) -> bool:
    if not a or not b:
        return False
    if abs(len(a) - len(b)) / max(len(a), len(b)) > 0.4:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _furniture_candidate_indices(
    page: RawPage, cfg: CleaningConfig, body_size: float = 0.0
) -> list[int]:
    """Line indices eligible to be page furniture: small type sitting in the page margins.

    Two independent conditions, both required.

    *Position.* Judging by ordinal position instead -- "the first two and last two lines" --
    fails twice over: it deletes a repeated body sentence that happens to open a page, and
    on a multi-column page, where lines are ordered column by column, it inspects the wrong
    lines entirely and misses the footer.

    *Size.* A running head is never set larger than the body text. Without this, a chapter
    opener whose heading repeats the running head ("Chapter One" in both places, which is
    the normal design of a novel) has its real heading deleted as a duplicate. It also
    keeps a chapter number set large on an otherwise empty page, which §4.4 relies on as a
    chapter marker and which the page-number rule would otherwise happily eat.
    """
    if page.height <= 0:
        return list(range(len(page.lines)))
    margin = page.height * cfg.furniture_margin_fraction
    bottom_limit = page.height - margin
    return [
        index
        for index, line in enumerate(page.lines)
        if (line.top <= margin or line.bottom >= bottom_limit)
        and (body_size <= 0 or line.size <= body_size + 0.5)
    ]


def find_running_furniture(
    raw: RawDocument, cfg: CleaningConfig, body_size: float = 0.0
) -> dict[int, set[int]]:
    """Identify running heads and feet.

    A candidate is any of the first or last ``header_lines_per_edge`` lines on a page. A
    candidate becomes furniture when its digit-masked form appears on more than
    ``header_page_fraction`` of pages, either exactly or above the similarity threshold.

    Returns:
        Page number -> set of line indices to remove.
    """
    pages = [p for p in raw.pages if p.lines]
    if len(pages) < 4:  # too few pages for the statistic to mean anything
        return {}

    candidates: dict[str, list[tuple[int, int]]] = {}
    for page in pages:
        for index in _furniture_candidate_indices(page, cfg, body_size):
            key = furniture_key(page.lines[index].text)
            if not key or len(key) > 120:
                continue
            candidates.setdefault(key, []).append((page.number, index))

    # Fold near-identical keys together, so OCR jitter or a changing chapter title still
    # collapses into one group.
    merged: dict[str, list[tuple[int, int]]] = {}
    for key, hits in sorted(candidates.items(), key=lambda kv: -len(kv[1])):
        for existing in merged:
            if _similar(key, existing, cfg.header_similarity):
                merged[existing].extend(hits)
                break
        else:
            merged[key] = list(hits)

    threshold = max(2, int(len(pages) * cfg.header_page_fraction))
    furniture: dict[int, set[int]] = {}
    for key, hits in merged.items():
        distinct_pages = {page_no for page_no, _ in hits}
        if len(distinct_pages) < threshold:
            continue
        for page_no, index in hits:
            furniture.setdefault(page_no, set()).add(index)
        log.debug("running_furniture", key=key, pages=len(distinct_pages))
    return furniture


def is_page_number(text: str) -> bool:
    """Standalone page numbers, roman numerals, and ``— 47 —`` decorations (§4.3.2)."""
    stripped = normalize_text(text).strip()
    if not stripped or len(stripped) > 20:
        return False
    return bool(_PAGE_NUMBER_RE.match(stripped))


def strip_furniture(raw: RawDocument, settings: Settings, audit: CleaningAudit) -> None:
    """Remove running heads, feet and page numbers from every page, in place."""
    cfg = settings.cleaning
    # Computed before stripping: the modal size is dominated by body text either way, and
    # the furniture rules need it to tell a running head from a chapter heading.
    body_size = raw.body_size()
    furniture = find_running_furniture(raw, cfg, body_size) if cfg.strip_running_heads else {}

    for page in raw.pages:
        keep: list[RawLine] = []
        drop_indices = furniture.get(page.number, set())
        margin_indices = set(_furniture_candidate_indices(page, cfg, body_size))
        for index, line in enumerate(page.lines):
            text = line.text.strip()
            if not text:
                continue
            if index in drop_indices:
                # Label by what the line is, not by which rule caught it: a folio repeats
                # on every page and so trips the furniture statistic first, and calling it
                # a running head in the audit would make the report lie.
                reason = "page-number" if is_page_number(text) else "running-head-or-foot"
                audit.stripped.append(StrippedLine(page=page.number, text=text, reason=reason))
                continue
            in_margin = index in margin_indices
            if cfg.strip_page_numbers and in_margin and is_page_number(text):
                audit.stripped.append(
                    StrippedLine(page=page.number, text=text, reason="page-number")
                )
                continue
            keep.append(line)
        page.lines = keep


# -- footnotes ----------------------------------------------------------------------


@dataclass(slots=True)
class Paragraph:
    """A reflowed paragraph: the thing that becomes an IR block."""

    lines: list[RawLine]
    text: str = ""
    pages: list[int] = field(default_factory=list)
    size: float = 0.0
    bold: bool = False
    x0: float = 0.0
    column: int = 0
    is_footnote: bool = False
    footnote_label: str | None = None
    footnote_refs: list[str] = field(default_factory=list)

    def finalise(self) -> Paragraph:
        self.pages = sorted({line.page for line in self.lines})
        sizes: Counter[float] = Counter()
        for line in self.lines:
            for span in line.spans:
                sizes[round(span.size, 1)] += max(len(span.text.strip()), 1)
        self.size = sizes.most_common(1)[0][0] if sizes else 0.0
        self.bold = sum(1 for line in self.lines if line.bold) > len(self.lines) / 2
        self.x0 = min((line.x0 for line in self.lines), default=0.0)
        columns = Counter(line.column for line in self.lines)
        self.column = columns.most_common(1)[0][0] if columns else 0
        return self


def split_footnote_lines(
    page: RawPage, body_size: float, cfg: CleaningConfig
) -> tuple[list[RawLine], list[RawLine]]:
    """Separate a page's footnote lines from its body lines (§4.3.7).

    A footnote block is a contiguous run at the bottom of the page whose font is smaller
    than the body's. Detection is anchored to the *bottom* deliberately: small type in the
    middle of a page is usually an epigraph or a caption, not a note.
    """
    if not cfg.extract_footnotes or not page.lines or body_size <= 0:
        return page.lines, []

    threshold = body_size * cfg.footnote_size_ratio
    small_from: int | None = None
    for index in range(len(page.lines) - 1, -1, -1):
        if page.lines[index].size <= threshold and page.lines[index].size > 0:
            small_from = index
        else:
            break

    if small_from is None:
        return page.lines, []

    footnotes = page.lines[small_from:]
    # One short small line at the bottom is more likely a folio or a caption than a note.
    if len(footnotes) == 1 and len(footnotes[0].text.strip()) < 15:
        return page.lines, []
    return page.lines[:small_from], footnotes


def group_footnotes(lines: list[RawLine]) -> list[Paragraph]:
    """Group footnote lines into one paragraph per note, keyed by its leading marker."""
    notes: list[Paragraph] = []
    current: list[RawLine] = []
    label: str | None = None
    for line in lines:
        match = _FOOTNOTE_MARK_RE.match(line.text)
        if match and current:
            notes.append(_finish_footnote(current, label))
            current = [line]
            label = match.group(1)
        elif match:
            current = [line]
            label = match.group(1)
        else:
            current.append(line)
    if current:
        notes.append(_finish_footnote(current, label))
    return notes


def _finish_footnote(lines: list[RawLine], label: str | None) -> Paragraph:
    para = Paragraph(lines=list(lines), is_footnote=True, footnote_label=label)
    text = " ".join(line.text.strip() for line in lines).strip()
    if label:
        text = _FOOTNOTE_MARK_RE.sub("", text, count=1).strip()
    para.text = re.sub(r"\s{2,}", " ", text)
    return para.finalise()


def mark_footnote_anchors(lines: list[RawLine], body_size: float, cfg: CleaningConfig) -> int:
    """Rewrite superscript markers as ``[^n]`` refs, in place, before reflow.

    A superscript is a span noticeably smaller than the body whose text is a bare numeral or
    a note symbol. Anything else is left alone: a wrong rewrite here invents a footnote
    reference with no note behind it.

    This edits span text rather than rebuilding the paragraph string afterwards. Rebuilding
    bypassed the line-joining rules and welded the last word of one line onto the first word
    of the next -- the kind of corruption that surfaces as a translation error three stages
    downstream, where nobody thinks to blame the extractor.

    Returns:
        The number of anchors rewritten.
    """
    if not cfg.extract_footnotes or body_size <= 0:
        return 0
    threshold = body_size * cfg.footnote_size_ratio
    count = 0
    for line in lines:
        for span in line.spans:
            stripped = span.text.strip()
            if 0 < span.size <= threshold and re.fullmatch(r"\d{1,3}|[*†‡§¶]{1,3}", stripped):
                span.text = f"[^{stripped}]"
                count += 1
    return count


# -- paragraph reflow ----------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _line_gap(previous: RawLine, current: RawLine) -> float:
    return current.top - previous.bottom


def should_break(
    previous: RawLine,
    current: RawLine,
    *,
    median_gap: float,
    median_height: float,
    column_right: float,
    cfg: CleaningConfig,
) -> bool:
    """Decide whether ``current`` starts a new paragraph (§4.3.4).

    Uses geometry where PyMuPDF gives it to us -- vertical gap and indentation -- and falls
    back to text evidence only when it must, because "line ends with a period" alone breaks
    every paragraph containing an abbreviation.
    """
    if current.column != previous.column:
        return True
    if current.page != previous.page:
        # A paragraph continues across a page break only when the evidence says so: the
        # previous page ended mid-sentence and the new page picks up in lower case.
        # Defaulting to "continue" merged the next page's running head into the previous
        # paragraph, which is exactly the silent corruption this pipeline exists to avoid.
        previous_text = previous.text.strip()
        current_text = current.text.lstrip()
        continues = (
            bool(previous_text)
            and not previous_text.endswith(_TERMINAL_PUNCT)
            and bool(current_text)
            and current_text[0].islower()
        )
        return not continues

    gap = _line_gap(previous, current)
    if median_gap >= 0 and gap > max(median_gap, median_height * 0.2) * cfg.paragraph_gap_multiple:
        return True

    # A first-line indent: the new line starts measurably right of the block's left edge.
    indent = current.x0 - previous.x0
    if indent > median_height * 0.8 and previous.text.strip().endswith(_TERMINAL_PUNCT):
        return True

    # A short previous line that ends a sentence, with the next line back at the margin,
    # is the classic end-of-paragraph shape.
    short_line = previous.x1 < column_right - median_height * 1.5
    if short_line and previous.text.strip().endswith(_TERMINAL_PUNCT):
        return True

    # A list item beginning where the previous line was not one starts a new block.
    return bool(_BULLET_RE.match(current.text)) and not _BULLET_RE.match(previous.text)


def group_paragraphs(lines: list[RawLine], cfg: CleaningConfig) -> list[Paragraph]:
    """Join lines into paragraphs. Never merges across a column boundary."""
    if not lines:
        return []
    if not cfg.reflow_paragraphs:
        return [Paragraph(lines=[line], text=line.text.strip()).finalise() for line in lines]

    heights = [line.height for line in lines if line.height > 0]
    median_height = _median(heights) or 12.0
    gaps: list[float] = []
    for prev, curr in pairwise(lines):
        if prev.page == curr.page and prev.column == curr.column:
            gap = _line_gap(prev, curr)
            if -median_height < gap < median_height * 4:
                gaps.append(gap)
    median_gap = _median(gaps)
    column_right = max((line.x1 for line in lines), default=0.0)

    paragraphs: list[Paragraph] = []
    current: list[RawLine] = [lines[0]]
    for prev, curr in pairwise(lines):
        if should_break(
            prev,
            curr,
            median_gap=median_gap,
            median_height=median_height,
            column_right=column_right,
            cfg=cfg,
        ):
            paragraphs.append(_join(current))
            current = [curr]
        else:
            current.append(curr)
    paragraphs.append(_join(current))
    return [p for p in paragraphs if p.text.strip()]


def _join(lines: list[RawLine]) -> Paragraph:
    """Join lines into one paragraph, leaving hyphens for the de-hyphenation pass."""
    para = Paragraph(lines=list(lines))
    parts: list[str] = []
    for index, line in enumerate(lines):
        text = line.text.strip()
        if index == 0:
            parts.append(text)
            continue
        previous = parts[-1] if parts else ""
        if previous.endswith("-"):
            parts.append("\x00" + text)  # marker: a line-break hyphen, resolved later
        else:
            parts.append(" " + text)
    para.text = re.sub(r"\s{2,}", " ", "".join(parts)).strip()
    para.footnote_refs = re.findall(r"\[\^([A-Za-z0-9_-]+)\]", para.text)
    return para.finalise()


# -- drop caps -------------------------------------------------------------------------


def repair_drop_caps(
    paragraphs: list[Paragraph], body_size: float, cfg: CleaningConfig, audit: CleaningAudit
) -> None:
    """Rejoin a drop cap with the word it belongs to (§4.3.6)."""
    if not cfg.repair_drop_caps or body_size <= 0:
        return
    threshold = body_size * cfg.drop_cap_size_ratio
    for para in paragraphs:
        if not para.lines or not para.lines[0].spans:
            continue
        first = para.lines[0].spans[0]
        letter = first.text.strip()
        if len(letter) != 1 or not letter.isalpha() or first.size < threshold:
            continue
        # Text after the drop cap, with any separating space removed.
        remainder = para.text[len(first.text) :].lstrip()
        if not remainder:
            continue
        para.text = letter + remainder
        audit.drop_caps_repaired.append(para.text[:60])


# -- de-hyphenation ----------------------------------------------------------------------


def _word_forms(text: str) -> Counter[str]:
    return Counter(word.lower() for word in re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", text))


def dehyphenate(paragraphs: list[Paragraph], cfg: CleaningConfig, audit: CleaningAudit) -> None:
    """Resolve line-break hyphens using whole-book frequency evidence (§4.3.3, D-17).

    Every candidate carries a ``\\x00`` marker left by the reflow pass. We count how often
    the joined and hyphenated forms occur elsewhere in the book and join when the joined
    form is at least as common. A tie joins, because a false join is a visible typo whereas
    a false split silently corrupts a token mid-sentence.
    """
    if not cfg.dehyphenate:
        for para in paragraphs:
            para.text = para.text.replace("\x00", "")
        return

    corpus: Counter[str] = Counter()
    for para in paragraphs:
        corpus.update(_word_forms(para.text.replace("-\x00", "").replace("\x00", "")))
        corpus.update(_word_forms(para.text.replace("\x00", "")))

    for para in paragraphs:
        if "\x00" not in para.text:
            continue
        page = para.pages[0] if para.pages else 0
        out: list[str] = []
        remaining = para.text
        while "\x00" in remaining:
            head, _, tail = remaining.partition("\x00")
            prefix_match = re.search(r"([^\W\d_]+)-$", head)
            suffix_match = re.match(r"([^\W\d_]+)", tail)
            if not prefix_match or not suffix_match:
                out.append(head)
                remaining = tail
                continue
            left, right = prefix_match.group(1), suffix_match.group(1)
            joined_form = f"{left}{right}".lower()
            hyphen_form = f"{left}-{right}".lower()
            joined_count = corpus.get(joined_form, 0)
            hyphen_count = corpus.get(hyphen_form, 0)
            # The candidate itself contributes to both counts; compare the rest of the book.
            join = joined_count >= hyphen_count
            audit.dehyphenations.append(
                DehyphenationDecision(
                    candidate=hyphen_form,
                    joined=join,
                    joined_form_count=joined_count,
                    hyphenated_form_count=hyphen_count,
                    page=page,
                )
            )
            out.append(head[:-1] if join else head)
            remaining = tail
        out.append(remaining)
        para.text = "".join(out)


# -- entry point --------------------------------------------------------------------------


@dataclass(slots=True)
class CleanResult:
    paragraphs: list[Paragraph]
    footnotes: list[Paragraph]
    audit: CleaningAudit
    body_size: float


def clean_document(raw: RawDocument, settings: Settings) -> CleanResult:
    """Run the whole cleaning pipeline over an extractor's output."""
    cfg = settings.cleaning
    audit = CleaningAudit()

    if cfg.normalize_unicode:
        normalize_document(raw)

    strip_furniture(raw, settings, audit)
    body_size = raw.body_size()

    body_lines: list[RawLine] = []
    footnote_paragraphs: list[Paragraph] = []
    for page in raw.pages:
        kept, footnote_lines = split_footnote_lines(page, body_size, cfg)
        body_lines.extend(kept)
        if footnote_lines:
            notes = group_footnotes(footnote_lines)
            footnote_paragraphs.extend(notes)
            audit.footnotes_extracted.extend(note.text[:60] for note in notes)

    anchors = mark_footnote_anchors(body_lines, body_size, cfg)
    paragraphs = group_paragraphs(body_lines, cfg)
    repair_drop_caps(paragraphs, body_size, cfg, audit)
    dehyphenate(paragraphs, cfg, audit)
    dehyphenate(footnote_paragraphs, cfg, audit)

    log.info(
        "cleaning_complete",
        paragraphs=len(paragraphs),
        footnotes=len(footnote_paragraphs),
        stripped=len(audit.stripped),
        anchors=anchors,
        dehyphenated=audit.joined_hyphens,
        body_size=body_size,
    )
    return CleanResult(
        paragraphs=paragraphs,
        footnotes=footnote_paragraphs,
        audit=audit,
        body_size=body_size,
    )
