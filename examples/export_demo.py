"""Milestone 7 demo: translate a fixture, then export every format and the report.

    uv run python examples/export_demo.py

Runs the real pipeline against a fake model, then produces Markdown, HTML, EPUB, DOCX, plain
text and (if Typst or WeasyPrint is installed) PDF, in both ``target-only`` and ``annotated``
layouts -- plus the HTML quality report. Nothing here needs an API key.

The output directory is printed at the end; open the annotated HTML and the report side by
side to see what a real run gives you to judge it with.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from folioai.config import packaged_settings
from folioai.export import export_job
from folioai.glossary import Glossary, Term
from folioai.glossary_build import audit_adherence
from folioai.jobs import prepare_job, run_job
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient
from folioai.logging_setup import console, shutdown_logging
from folioai.report import gather, write_report
from folioai.tags import parse_segments

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "pdfs" / "clean_book.pdf"
OUT = REPO / "demo-output"

PASS = {"completeness": 96, "accuracy": 94, "terminology": 95, "fluency": 92, "formatting": 100}
WEAK = {"completeness": 68, "accuracy": 74, "terminology": 80, "fluency": 88, "formatting": 95}
#: One segment scores badly so the annotated layout and the report have something to show.
WEAK_SEGMENT = "b0002"


def fake_endpoint() -> object:
    def handler(messages: list[Message], model: str) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "SOURCE:" in user:
            ids = [line[4:].strip() for line in user.splitlines() if line.startswith("### ")]
            scores, issues = [], []
            for segment_id in ids:
                weak = segment_id == WEAK_SEGMENT
                scores.append({"segment_id": segment_id, **(WEAK if weak else PASS)})
                if weak:
                    issues.append(
                        {
                            "segment_id": segment_id,
                            "dimension": "completeness",
                            "severity": "major",
                            "source_excerpt": "and more like a wall she had built",
                            "translation_excerpt": "(nothing corresponding)",
                            "explanation": "the final clause is not represented in the target",
                            "suggested_fix": "restore the omitted clause",
                        }
                    )
            return json.dumps({"scores": scores, "issues": issues})

        parsed = parse_segments(user)
        return "\n".join(f'<seg id="{s}">[DE] {t}</seg>' for s, t in parsed.texts.items())

    return handler


async def main() -> int:
    if not FIXTURE.is_file():
        console().print(
            "[bad]Fixtures missing.[/bad] Build them: "
            "[muted]uv run python tests/fixtures/make_pdfs.py[/muted]"
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    # Isolate the job *state* only. Relocating FOLIOAI_HOME would also move the bin and
    # fonts directories, and the demo would then report that no PDF engine is installed
    # while one sits in the real one.
    os.environ["FOLIOAI_JOBS_DIR"] = str(OUT / "jobs")
    os.environ["FOLIOAI_LOGS_DIR"] = str(OUT / "logs")
    os.environ["FOLIOAI_STATE_FILE"] = str(OUT / "state.json")

    settings = packaged_settings()
    settings.translation.batch_tokens = 400

    context = prepare_job(FIXTURE, settings, target_lang="de")
    try:
        # A glossary, so the report has adherence to talk about.
        context.glossary = Glossary(
            terms=[Term(source="lamp", target="Lampe", kind="other", occurrences=1)]
        )
        context.store.upsert_glossary(context.job_id, context.glossary.to_rows())

        console().print(f"[heading]translating[/heading] {FIXTURE.name} -> de\n")
        _, stats = await run_job(context, settings, client=FakeLLMClient(fake_endpoint()))
        console().print(
            f"[good]done[/good] {stats.completed} segments, mean {stats.mean_score}, "
            f"{stats.needs_review} flagged\n"
        )

        segments = context.store.list_segments(context.job_id)
        translations = {s.segment_id: s.final_text or "" for s in segments if s.final_text}
        sources = {s.segment_id: s.source_text for s in segments}
        usages = audit_adherence(context.glossary, sources, translations)

        data = gather(
            context.store, context.job_id, document=context.document, glossary_usages=usages
        )
        report_path = write_report(data, OUT / "report.html")
        console().print(f"[good]report[/good] {report_path}")
    finally:
        job_id = context.job_id
        context.close()
        shutdown_logging()

    for layout in ("target-only", "annotated", "bilingual-paragraph"):
        result = export_job(
            job_id,
            settings,
            formats=["md", "html", "txt", "epub", "docx", "pdf"],
            layout=layout,  # type: ignore[arg-type]
            out_dir=OUT / layout,
        )
        console().print(f"\n[heading]{layout}[/heading]")
        for path in result.files:
            console().print(f"  [good]wrote[/good] {path.relative_to(REPO)}")
        for warning in result.warnings:
            console().print(f"  [warn]note:[/warn] {warning.splitlines()[0]}")

    console().print(f"\n[muted]everything is under {OUT.relative_to(REPO)}[/muted]")
    shutdown_logging()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
