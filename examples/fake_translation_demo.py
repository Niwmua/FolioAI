"""Run the whole pipeline on a fixture PDF with a fake model. No key, no network, no cost.

    uv run python examples/fake_translation_demo.py

This is the milestone 4-5 demo. It exercises every stage the real command does -- extract,
segment, translate, validate, evaluate, retry, persist, assert the output is parallel to the
source -- with ``FakeLLMClient`` standing in for the endpoint. The "translator" prefixes each
segment with ``[DE]``; the "judge" scores everything, and deliberately fails one segment so
the retry ladder and the escalation model appear in the output.

Point it at your own PDF with an argument:

    uv run python examples/fake_translation_demo.py path/to/book.pdf
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from folioai.config import packaged_settings
from folioai.jobs import prepare_job, run_job
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient
from folioai.logging_setup import console, shutdown_logging
from folioai.render.markdown import document_to_markdown
from folioai.tags import parse_segments

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "pdfs" / "clean_book.pdf"

PASS = {"completeness": 96, "accuracy": 93, "terminology": 95, "fluency": 90, "formatting": 100}
FAIL = {"completeness": 55, "accuracy": 60, "terminology": 70, "fluency": 80, "formatting": 90}

#: Failed on the first attempt only, so the ladder runs and then recovers.
TROUBLESOME = "b0002"


def fake_endpoint() -> object:
    """A model that translates by prefixing, and a judge that fails one segment once."""
    seen: set[str] = set()

    def handler(messages: list[Message], model: str) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")

        if "SOURCE:" in user:  # an evaluation call
            ids = [line[4:].strip() for line in user.splitlines() if line.startswith("### ")]
            scores, issues = [], []
            for segment_id in ids:
                bad = segment_id == TROUBLESOME and segment_id not in seen
                if bad:
                    seen.add(segment_id)
                    issues.append(
                        {
                            "segment_id": segment_id,
                            "dimension": "completeness",
                            "severity": "major",
                            "source_excerpt": "the second clause of the sentence",
                            "explanation": "the subordinate clause is missing",
                        }
                    )
                scores.append({"segment_id": segment_id, **(FAIL if bad else PASS)})
            return json.dumps({"scores": scores, "issues": issues})

        parsed = parse_segments(user)
        return "\n".join(f'<seg id="{s}">[DE] {t}</seg>' for s, t in parsed.texts.items())

    return handler


async def main() -> int:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    if not pdf.is_file():
        console().print(
            f"[bad]No such file:[/bad] {pdf}\n"
            "[muted]Build the fixtures first: uv run python tests/fixtures/make_pdfs.py[/muted]"
        )
        return 1

    settings = packaged_settings()
    settings.translation.batch_tokens = 400  # several batches out of a small fixture

    with tempfile.TemporaryDirectory(prefix="folioai-demo-", ignore_cleanup_errors=True) as tmp:
        import os

        os.environ["FOLIOAI_HOME"] = tmp  # keep the demo out of the real ~/.folioai

        context = prepare_job(pdf, settings, target_lang="de")
        try:
            console().print(
                f"[heading]{pdf.name}[/heading] -> de   "
                f"{len(context.document.blocks)} blocks, "
                f"{len(context.document.chapters)} chapters, "
                f"job [info]{context.job_id}[/info]\n"
            )
            translated, stats = await run_job(
                context, settings, client=FakeLLMClient(fake_endpoint())
            )

            context.document.assert_parallel_to(translated)
            console().print(
                f"[good]done[/good]  {stats.completed} segments, mean score "
                f"[heading]{stats.mean_score}[/heading], {stats.retries} retries, "
                f"{stats.needs_review} flagged for review"
            )

            rows = context.store.list_attempts(context.job_id, TROUBLESOME)
            if len(rows) > 1:
                ladder = " -> ".join(f"attempt {r['attempt_no']} ({r['model']})" for r in rows)
                console().print(f"[muted]{TROUBLESOME} climbed the ladder: {ladder}[/muted]")

            console().print(
                f"\n[muted]block count in {len(context.document.blocks)}, "
                f"out {len(translated.blocks)} -- asserted, not eyeballed[/muted]\n"
            )
            console().print(document_to_markdown(translated, front_matter=False)[:900])
        finally:
            context.close()
            shutdown_logging()  # release the job's log file so the temp dir can be removed
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
