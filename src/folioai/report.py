"""The quality report (brief §15): one self-contained HTML file, no assets, no network.

What it must answer, in this order, because that is the order a reader actually asks:

1. Can I trust this run? — models, cost, mean score, how much needs review.
2. Where is it weakest? — score distribution, per-chapter means.
3. Show me the bad ones. — every segment below threshold, with source, all attempts, and
   the evaluator's issues.
4. Did it stay consistent? — glossary terms rendered more than one way.
5. What did extraction do to the text before any of this? — the cleaning audit.

Charts are inline SVG computed here rather than a charting library: a report that needs a
CDN is not self-contained, and one that needs a build step will not be written.
"""

from __future__ import annotations

import html
import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ir import Document
from .logging_setup import get_logger
from .store import JobStore

if TYPE_CHECKING:
    from .glossary_build import TermUsage

log = get_logger(__name__)

BUCKETS = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]


@dataclass(slots=True)
class ReportData:
    """Everything the report shows, gathered before any HTML is written."""

    job_id: str
    title: str
    source_lang: str
    target_lang: str
    status: str
    models: list[str] = field(default_factory=list)
    evaluator_models: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    total_segments: int = 0
    done_segments: int = 0
    needs_review: int = 0
    scores: list[float] = field(default_factory=list)
    chapter_means: dict[str, float] = field(default_factory=dict)
    chapter_titles: dict[str, str] = field(default_factory=dict)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    glossary_usages: list[TermUsage] = field(default_factory=list)
    extraction: dict[str, Any] = field(default_factory=dict)
    sentence_deltas: dict[str, float] = field(default_factory=dict)
    duration_s: float | None = None

    @property
    def mean_score(self) -> float:
        return round(statistics.mean(self.scores), 1) if self.scores else 0.0

    @property
    def median_score(self) -> float:
        return round(statistics.median(self.scores), 1) if self.scores else 0.0

    @property
    def review_share(self) -> float:
        return (self.needs_review / self.done_segments) if self.done_segments else 0.0


def _attempts_for(conn: sqlite3.Connection, job_id: str, segment_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT a.attempt_no, a.model, a.output_text, a.cost_usd,
               e.composite, e.passed, e.issues_json, e.scores_json
        FROM attempts a
        LEFT JOIN evaluations e ON e.attempt_id = a.id
        WHERE a.job_id = ? AND a.segment_id = ?
        ORDER BY a.attempt_no
        """,
        (job_id, segment_id),
    ).fetchall()
    attempts = []
    for row in rows:
        issues = json.loads(row["issues_json"]) if row["issues_json"] else []
        attempts.append(
            {
                "attempt_no": row["attempt_no"],
                "model": row["model"],
                "text": row["output_text"] or "",
                "composite": row["composite"],
                "passed": bool(row["passed"]) if row["passed"] is not None else None,
                "issues": issues,
                "cost_usd": row["cost_usd"],
            }
        )
    return attempts


def gather(
    store: JobStore,
    job_id: str,
    *,
    document: Document | None = None,
    glossary_usages: list[TermUsage] | None = None,
    max_flagged: int = 200,
) -> ReportData:
    """Collect everything the report needs from the job database and the IR."""
    job = store.get_job(job_id)
    if job is None:
        raise ValueError(f"no job row for {job_id}")

    segments = store.list_segments(job_id)
    done = [s for s in segments if s.final_text]
    scores = [float(s.final_score) for s in done if s.final_score is not None]

    models = [
        row["model"]
        for row in store.conn.execute(
            "SELECT DISTINCT model FROM attempts WHERE job_id = ? ORDER BY model", (job_id,)
        ).fetchall()
    ]
    evaluators = [
        row["evaluator_model"]
        for row in store.conn.execute(
            "SELECT DISTINCT evaluator_model FROM evaluations ORDER BY evaluator_model"
        ).fetchall()
    ]

    by_chapter: dict[str, list[float]] = {}
    for segment in done:
        if segment.final_score is not None and segment.chapter_id:
            by_chapter.setdefault(segment.chapter_id, []).append(float(segment.final_score))

    data = ReportData(
        job_id=job_id,
        title=(document.title if document else None) or Path(job.source_path).stem,
        source_lang=job.source_lang or "?",
        target_lang=job.target_lang or "?",
        status=job.status,
        models=models,
        evaluator_models=evaluators,
        cost_usd=job.cost_usd,
        total_segments=job.total_segments,
        done_segments=len(done),
        needs_review=sum(1 for s in segments if s.needs_review),
        scores=scores,
        chapter_means={k: round(statistics.mean(v), 1) for k, v in by_chapter.items()},
        glossary_usages=glossary_usages or [],
    )

    if document is not None:
        data.chapter_titles = {c.id: c.title for c in document.chapters}
        data.extraction = document.extraction_report.model_dump()

        from .validate import aggregate_sentence_deltas

        pairs = [
            (s.chapter_id or "", s.source_text, s.final_text or "") for s in done if s.chapter_id
        ]
        data.sentence_deltas = aggregate_sentence_deltas(pairs)

    threshold_failures = [
        s for s in done if s.needs_review or (s.final_score is not None and s.final_score < 80)
    ]
    for segment in threshold_failures[:max_flagged]:
        data.flagged.append(
            {
                "segment_id": segment.segment_id,
                "chapter_id": segment.chapter_id,
                "score": segment.final_score,
                "source": segment.source_text,
                "final": segment.final_text or "",
                "attempts": _attempts_for(store.conn, job_id, segment.segment_id),
            }
        )

    log.info(
        "report_gathered",
        job=job_id,
        segments=data.done_segments,
        flagged=len(data.flagged),
        mean=data.mean_score,
    )
    return data


# -- charts (inline SVG, no dependencies) --------------------------------------------------


def histogram_svg(scores: list[float], *, width: int = 620, height: int = 170) -> str:
    """Score distribution. Buckets are uneven on purpose: 90-100 is where most runs land."""
    if not scores:
        return '<p class="muted">No scores yet.</p>'

    counts = [sum(1 for s in scores if low <= s < high) for low, high in BUCKETS]
    peak = max(counts) or 1
    bar_width = (width - 60) / len(BUCKETS)
    bars = []
    for index, (count, (low, high)) in enumerate(zip(counts, BUCKETS, strict=True)):
        bar_height = (count / peak) * (height - 46)
        x = 40 + index * bar_width
        y = height - 26 - bar_height
        colour = "var(--flag)" if high <= 80 else "var(--accent)"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 6:.1f}" '
            f'height="{bar_height:.1f}" fill="{colour}" rx="2"/>'
            f'<text x="{x + (bar_width - 6) / 2:.1f}" y="{y - 4:.1f}" '
            f'class="bar-label">{count or ""}</text>'
            f'<text x="{x + (bar_width - 6) / 2:.1f}" y="{height - 8}" '
            f'class="axis">{low}-{high - 1 if high < 101 else 100}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="score distribution">{"".join(bars)}</svg>'
    )


def chapter_chart_svg(
    means: dict[str, float], titles: dict[str, str], *, width: int = 620, height: int = 180
) -> str:
    """Per-chapter mean score, in reading order, so a bad stretch is visible as a dip."""
    if len(means) < 2:
        return '<p class="muted">Not enough chapters to plot.</p>'

    ordered = sorted(means.items())
    values = [value for _, value in ordered]
    low = min(min(values) - 2, 70)
    high = 100
    span = max(high - low, 1)
    step = (width - 60) / max(len(ordered) - 1, 1)

    points = []
    for index, value in enumerate(values):
        x = 40 + index * step
        y = 20 + (1 - (value - low) / span) * (height - 50)
        points.append(f"{x:.1f},{y:.1f}")

    dots = "".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="2.5" '
        f'fill="{"var(--flag)" if value < 80 else "var(--accent)"}">'
        f"<title>{html.escape(titles.get(cid, cid))}: {value}</title></circle>"
        for p, (cid, value) in zip(points, ordered, strict=True)
    )
    threshold_y = 20 + (1 - (80 - low) / span) * (height - 50)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="mean score by chapter">'
        f'<line x1="40" y1="{threshold_y:.1f}" x2="{width - 20}" y2="{threshold_y:.1f}" '
        f'class="threshold"/>'
        f'<text x="{width - 18}" y="{threshold_y - 4:.1f}" class="axis">80</text>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="var(--accent)" '
        f'stroke-width="1.5"/>{dots}</svg>'
    )


# -- the page ---------------------------------------------------------------------------------

STYLE = """
:root {
  --bg:#fdfdfb; --fg:#1c1c1a; --muted:#6b6b66; --rule:#e4e2dc; --card:#fff;
  --flag:#b23a2e; --flag-bg:#fdf2f0; --accent:#2f5d8a; --good:#2d6a4f;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#16171a; --fg:#e6e4df; --muted:#9a978f; --rule:#2e3034; --card:#1d1f23;
          --flag:#f08a7c; --flag-bg:#2a1d1b; --accent:#7fb0e0; --good:#7fc8a0; }
}
* { box-sizing:border-box; }
body { margin:0; padding:2.5rem 1.5rem 6rem; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system, "Segoe UI", Roboto, system-ui, sans-serif;
  max-width:64rem; margin-inline:auto; }
h1 { font-size:1.6rem; margin:0 0 .2em; }
h2 { font-size:1.05rem; margin:2.5rem 0 .8rem; padding-bottom:.4rem;
  border-bottom:1px solid var(--rule); font-weight:600; }
.muted { color:var(--muted); }
.sub { color:var(--muted); margin:0 0 2rem; font-size:.9rem; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(8.5rem,1fr)); gap:.8rem; }
.card { background:var(--card); border:1px solid var(--rule); border-radius:6px;
  padding:.9rem 1rem; }
.card .k { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
.card .v { font-size:1.5rem; font-weight:600; margin-top:.15rem; }
.card .v.bad { color:var(--flag); } .card .v.good { color:var(--good); }
svg { width:100%; height:auto; }
.bar-label { font-size:9px; fill:var(--muted); text-anchor:middle; }
.axis { font-size:9px; fill:var(--muted); text-anchor:middle; }
.threshold { stroke:var(--flag); stroke-width:1; stroke-dasharray:3 3; opacity:.6; }
table { width:100%; border-collapse:collapse; font-size:.86rem; }
th, td { text-align:start; padding:.4rem .6rem; border-bottom:1px solid var(--rule);
  vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:.75rem; text-transform:uppercase;
  letter-spacing:.05em; }
td.num { text-align:end; font-variant-numeric:tabular-nums; }
details.seg { border:1px solid var(--rule); border-radius:6px; margin-bottom:.7rem;
  background:var(--card); }
details.seg summary { padding:.6rem .9rem; cursor:pointer; display:flex; gap:.8rem;
  align-items:baseline; }
details.seg[open] summary { border-bottom:1px solid var(--rule); }
.sid { font-family:ui-monospace, Menlo, Consolas, monospace; font-size:.8rem; color:var(--muted); }
.score-pill { font-weight:600; font-size:.8rem; padding:.05rem .45rem; border-radius:10px;
  background:var(--flag-bg); color:var(--flag); }
.excerpt { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.body { padding:.9rem; display:grid; gap:.9rem; }
.pane { border-inline-start:3px solid var(--rule); padding-inline-start:.8rem; }
.pane h4 { margin:0 0 .3rem; font-size:.72rem; text-transform:uppercase; color:var(--muted);
  letter-spacing:.05em; }
.issue { background:var(--flag-bg); border-radius:4px; padding:.5rem .7rem; margin:.3rem 0;
  font-size:.85rem; }
.issue .sev { font-weight:600; color:var(--flag); text-transform:uppercase; font-size:.7rem; }
.attempt { font-size:.85rem; border-top:1px dashed var(--rule); padding-top:.5rem; }
"""


def _excerpt(text: str, limit: int = 90) -> str:
    flat = " ".join(text.split())
    return html.escape(flat[:limit] + ("…" if len(flat) > limit else ""))


def _flagged_html(entry: dict[str, Any]) -> str:
    score = entry["score"]
    pill = (
        f'<span class="score-pill">{score:.0f}</span>'
        if score is not None
        else ('<span class="score-pill">no score</span>')
    )
    parts = [
        "<details class='seg'>",
        f'<summary><span class="sid">{html.escape(entry["segment_id"])}</span>{pill}'
        f'<span class="excerpt">{_excerpt(entry["source"])}</span></summary>',
        '<div class="body">',
        f'<div class="pane"><h4>source</h4>{html.escape(entry["source"])}</div>',
        f'<div class="pane"><h4>kept translation</h4>{html.escape(entry["final"])}</div>',
    ]
    for attempt in entry["attempts"]:
        composite = attempt["composite"]
        header = f"attempt {attempt['attempt_no']} — {html.escape(attempt['model'])}"
        if composite is not None:
            header += f" — {composite:.1f}"
        parts.append(f'<div class="attempt"><strong>{header}</strong>')
        if attempt["text"] and attempt["text"] != entry["final"]:
            parts.append(f'<div class="muted">{html.escape(attempt["text"][:600])}</div>')
        for issue in attempt["issues"]:
            parts.append(
                # `or ""` rather than a .get default: these keys exist and can be null --
                # a judge may decline to name a dimension -- and html.escape(None) raises.
                f'<div class="issue"><span class="sev">{html.escape(issue.get("severity") or "")}'
                f"</span> {html.escape(issue.get('dimension') or '')} — "
                f"{html.escape(issue.get('explanation') or '')}</div>"
            )
        parts.append("</div>")
    parts.append("</div></details>")
    return "".join(parts)


def render_report(data: ReportData) -> str:
    """Render the whole report as one self-contained HTML file."""
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    review_class = "bad" if data.review_share > 0.1 else "good"
    mean_class = "bad" if data.mean_score < 80 else "good"

    cards = [
        ("segments", f"{data.done_segments}/{data.total_segments}", ""),
        ("mean score", f"{data.mean_score}", mean_class),
        ("median", f"{data.median_score}", ""),
        ("needs review", f"{data.needs_review}", review_class),
        ("cost", f"${data.cost_usd:,.2f}", ""),
        ("status", html.escape(data.status), ""),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'
        for k, v, cls in cards
    )

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(data.title)} — quality report</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<h1>{html.escape(data.title)}</h1>",
        f'<p class="sub">{html.escape(data.source_lang)} → {html.escape(data.target_lang)} · '
        f'job <span class="sid">{html.escape(data.job_id)}</span> · generated {generated}<br>'
        f"translated by {html.escape(', '.join(data.models) or 'unknown')}"
        + (
            f" · judged by {html.escape(', '.join(data.evaluator_models))}"
            if data.evaluator_models
            else ""
        )
        + "</p>",
        f'<div class="cards">{card_html}</div>',
        "<h2>Score distribution</h2>",
        histogram_svg(data.scores),
    ]

    if data.chapter_means:
        parts.append("<h2>Mean score by chapter</h2>")
        parts.append(chapter_chart_svg(data.chapter_means, data.chapter_titles))

    parts.append(f"<h2>Segments needing review ({len(data.flagged)})</h2>")
    if not data.flagged:
        parts.append('<p class="muted">Nothing fell below the threshold.</p>')
    else:
        parts.extend(_flagged_html(entry) for entry in data.flagged)

    parts.append("<h2>Glossary adherence</h2>")
    inconsistent = [u for u in data.glossary_usages if not u.consistent]
    if not data.glossary_usages:
        parts.append('<p class="muted">No glossary was used for this job.</p>')
    elif not inconsistent:
        parts.append(
            f'<p class="muted">All {len(data.glossary_usages)} term(s) were rendered '
            "consistently throughout.</p>"
        )
    else:
        rows = "".join(
            f"<tr><td>{html.escape(u.term.source)}</td>"
            f"<td>{html.escape(u.term.target)}</td>"
            f'<td class="num">{u.occurrences}</td>'
            + "<td>"
            + html.escape(", ".join(f"{k} ({len(v)})" for k, v in u.renderings.items()))
            + "</td>"
            "</tr>"
            for u in inconsistent
        )
        parts.append(
            "<table><thead><tr><th>term</th><th>expected</th><th>uses</th>"
            f"<th>renderings found</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if data.extraction:
        parts.append("<h2>Extraction audit</h2>")
        interesting = [
            ("extractor", data.extraction.get("extractor")),
            ("pages", data.extraction.get("page_count")),
            ("blocks", data.extraction.get("block_count")),
            ("structure from", data.extraction.get("structure_source")),
            ("running heads stripped", data.extraction.get("stripped_headers")),
            ("page numbers stripped", data.extraction.get("stripped_page_numbers")),
            ("hyphenations joined", data.extraction.get("dehyphenations")),
            ("drop caps repaired", data.extraction.get("drop_caps_repaired")),
            ("footnotes extracted", data.extraction.get("footnotes_extracted")),
            ("columns detected", data.extraction.get("columns_detected")),
        ]
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td class='num'>{html.escape(str(v))}</td></tr>"
            for k, v in interesting
            if v is not None
        )
        parts.append(f"<table><tbody>{rows}</tbody></table>")
        for warning in data.extraction.get("warnings", []) or []:
            parts.append(f'<p class="issue">{html.escape(str(warning))}</p>')

    if data.sentence_deltas:
        # PLAN §2.4: per segment this is noise; per chapter a systematic delta is signal.
        outliers = {k: v for k, v in data.sentence_deltas.items() if abs(v) >= 1.0}
        if outliers:
            parts.append("<h2>Sentence structure drift</h2>")
            parts.append(
                '<p class="muted">Mean difference in sentence count per segment. Some '
                "variation is normal for any language pair; a large systematic value means "
                "the translation is restructuring the prose.</p>"
            )
            rows = "".join(
                f"<tr><td>{html.escape(data.chapter_titles.get(k, k))}</td>"
                f'<td class="num">{v:+.1f}</td></tr>'
                for k, v in sorted(outliers.items(), key=lambda kv: -abs(kv[1]))
            )
            parts.append(f"<table><tbody>{rows}</tbody></table>")

    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def write_report(data: ReportData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(data), encoding="utf-8")
    log.info("report_written", path=str(path), flagged=len(data.flagged))
    return path
