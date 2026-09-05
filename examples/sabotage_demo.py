"""Acceptance criterion §21.5, as a runnable demonstration.

    uv run python examples/sabotage_demo.py

A deliberately sabotaged translator drops every 3rd segment. The demonstration is that:

1. deterministic validation catches every drop **before** the evaluator is ever called, so
   no money is spent asking a second model whether a malformed response was any good; and
2. every dropped segment is recovered by the retry, so the output has the same number of
   blocks as the input.

The counters printed at the end are read back out of the job database, not accumulated in
memory, so they are the same numbers ``folioai status`` would show.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from folioai.config import packaged_settings
from folioai.jobs import prepare_job, run_job
from folioai.llm.client import Message
from folioai.llm.fake import FakeLLMClient
from folioai.logging_setup import console, shutdown_logging
from folioai.tags import parse_segments

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "pdfs" / "clean_book.pdf"

PASS = {"completeness": 96, "accuracy": 94, "terminology": 95, "fluency": 92, "formatting": 100}
DROP_EVERY = 3


def sabotaged() -> object:
    """Drops every 3rd segment on first attempts; behaves on retries."""
    state = {"translate_calls": 0, "evaluate_calls": 0, "dropped": 0}

    def handler(messages: list[Message], model: str) -> str:
        user = next(m["content"] for m in messages if m["role"] == "user")

        if "SOURCE:" in user:
            state["evaluate_calls"] += 1
            ids = [line[4:].strip() for line in user.splitlines() if line.startswith("### ")]
            return json.dumps({"scores": [{"segment_id": i, **PASS} for i in ids], "issues": []})

        state["translate_calls"] += 1
        first_attempt = len(messages) == 2  # a retry carries a third, corrective message
        parsed = parse_segments(user)
        out = []
        for index, (segment_id, text) in enumerate(parsed.texts.items(), start=1):
            if first_attempt and index % DROP_EVERY == 0:
                state["dropped"] += 1
                continue  # the sabotage: silently omit the segment
            out.append(f'<seg id="{segment_id}">[DE] {text}</seg>')
        return "\n".join(out)

    handler.state = state  # type: ignore[attr-defined]
    return handler


async def main() -> int:
    if not FIXTURE.is_file():
        console().print(
            "[bad]Fixtures missing.[/bad] Build them first: "
            "[muted]uv run python tests/fixtures/make_pdfs.py[/muted]"
        )
        return 1

    settings = packaged_settings()
    settings.translation.batch_tokens = 400

    with tempfile.TemporaryDirectory(prefix="folioai-sabotage-", ignore_cleanup_errors=True) as tmp:
        os.environ["FOLIOAI_HOME"] = tmp
        handler = sabotaged()
        context = prepare_job(FIXTURE, settings, target_lang="de")
        try:
            expected = len(context.document.translatable_blocks())
            console().print(
                f"[heading]Sabotage test[/heading]  a translator that drops every "
                f"{DROP_EVERY}rd segment\n"
                f"[muted]{expected} translatable blocks in {FIXTURE.name}[/muted]\n"
            )

            translated, stats = await run_job(context, settings, client=FakeLLMClient(handler))
            state = handler.state  # type: ignore[attr-defined]

            rows = context.store.conn.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE job_id = ?", (context.job_id,)
            ).fetchone()
            translated_rows = [
                r for r in context.store.list_segments(context.job_id) if r.final_text
            ]

            console().print(f"  segments dropped by the saboteur   {state['dropped']}")
            console().print(f"  translate calls                    {state['translate_calls']}")
            console().print(
                f"  evaluate calls                     {state['evaluate_calls']}  "
                f"[muted](the mangled attempts were never sent to the judge)[/muted]"
            )
            console().print(f"  attempts recorded in the database  {rows['n']}")
            console().print(f"  segments with a final translation  {len(translated_rows)}")
            console().print(f"  flagged for review                 {stats.needs_review}")

            context.document.assert_parallel_to(translated)
            ok = (
                state["dropped"] > 0
                and len(translated_rows) == expected
                and state["evaluate_calls"] < state["translate_calls"]
            )
            console().print(
                f"\n[{'good' if ok else 'bad'}]"
                f"{'PASS' if ok else 'FAIL'}[/{'good' if ok else 'bad'}]"
                "  every dropped segment was detected without a judge, recovered on retry, "
                "and the output is block-for-block parallel to the source."
            )
            return 0 if ok else 1
        finally:
            context.close()
            shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
