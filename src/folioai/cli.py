"""Command line surface (brief §16).

Every command is thin: parse flags, merge config, call into a module, render the result.
Nothing in here does real work, so the pipeline stays usable as a library and testable
without a terminal.

Commands not yet implemented raise ``NotImplementedError`` with the milestone that will
deliver them. They do not return plausible-looking fake data (brief §0).
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from rich.table import Table

from . import __version__
from .config import Settings, load_settings
from .errors import ConfigError, FolioError
from .llm.pricing import format_usd
from .logging_setup import configure_logging, console, err_console, get_logger
from .paths import ensure_dirs, home_dir, state_path
from .store import JobStore, discover_jobs, job_paths

F = TypeVar("F", bound=Callable[..., Any])

app = typer.Typer(
    name="folioai",
    help="Faithful book translation: PDF in, verified translation out.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
jobs_app = typer.Typer(help="Inspect and manage jobs.", no_args_is_help=True)
glossary_app = typer.Typer(help="Build and edit the per-job glossary.", no_args_is_help=True)
app.add_typer(jobs_app, name="jobs")
app.add_typer(glossary_app, name="glossary")

_STATE: dict[str, Any] = {"verbosity": 0, "config_path": None, "tty": True}

RIGHTS_NOTICE = (
    "Note: you are responsible for holding the rights to translate the material you feed in."
)


def handle_errors(func: F) -> F:
    """Turn a ``FolioError`` into a clean message and exit code instead of a traceback."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FolioError as exc:
            get_logger().error(
                "command_failed", error=type(exc).__name__, message=exc.message, **exc.context
            )
            err_console().print(f"[bad]error:[/bad] {exc.format_for_user()}")
            if _STATE["verbosity"] >= 2 and exc.context:
                err_console().print(f"[muted]{json.dumps(exc.context, default=str)}[/muted]")
            raise typer.Exit(code=exc.exit_code) from exc
        except NotImplementedError as exc:
            err_console().print(f"[warn]not implemented yet:[/warn] {exc}")
            raise typer.Exit(code=69) from exc
        except KeyboardInterrupt:
            err_console().print("\n[warn]interrupted.[/warn]")
            raise typer.Exit(code=130) from None

    return wrapper  # type: ignore[return-value]


def _settings(**overrides: Any) -> Settings:
    """Load settings for this invocation, applying CLI overrides at the top of the stack."""
    nested: dict[str, Any] = {}
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = nested
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    settings = load_settings(
        cli_overrides=nested,
        extra_config=_STATE["config_path"],
    )
    verbosity = int(_STATE["verbosity"])
    level = settings.logging.level if verbosity == 0 else "DEBUG"
    configure_logging(level=level, force=True, to_stderr=verbosity > 0)
    return settings


def _maybe_show_rights_notice() -> None:
    """Print the §1 reminder once per machine, gating nothing (D-53)."""
    ensure_dirs()
    path = state_path()
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        get_logger().warning("state_read_failed", path=str(path), error=str(exc))
        state = {}
    if state.get("rights_notice_shown"):
        return
    console().print(f"[muted]{RIGHTS_NOTICE}[/muted]")
    state["rights_notice_shown"] = True
    try:
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        get_logger().warning("state_write_failed", path=str(path), error=str(exc))


def _version_callback(value: bool) -> None:
    if value:
        console().print(f"folioai {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="Increase log verbosity.")
    ] = 0,
    config: Annotated[
        Path | None, typer.Option("--config", help="Extra YAML config, applied below CLI flags.")
    ] = None,
    no_tty: Annotated[
        bool, typer.Option("--no-tty", help="Plain line-per-event output for cron and CI.")
    ] = False,
    _version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """Global options, applied before any command runs."""
    _STATE["verbosity"] = verbose
    _STATE["config_path"] = config
    _STATE["tty"] = not no_tty
    if verbose:
        # Read by jobs.py when it reconfigures logging for a specific job.
        os.environ["FOLIOAI_LOG_STDERR"] = "1"
    _maybe_show_rights_notice()


# --------------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------------


@jobs_app.command("list")
@handle_errors
def jobs_list() -> None:
    """List every job on this machine."""
    found = discover_jobs()
    if not found:
        console().print(
            f"[muted]No jobs yet. Job state lives in {home_dir()}.\n"
            "Start one with:[/muted] folioai translate BOOK.pdf --to de"
        )
        return

    from rich import box

    # No "source" column: the job id already begins with the source file's name, and seven
    # columns do not fit in an 80-column terminal -- they just truncate into nonsense.
    table = Table(
        box=box.SIMPLE_HEAD,
        pad_edge=False,
        padding=(0, 1),
        title=f"folioai jobs ({home_dir()})",
        title_style="heading",
    )
    table.add_column("job id", style="info", no_wrap=True, overflow="ellipsis", max_width=24)
    table.add_column("lang", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("progress", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)
    table.add_column("updated", no_wrap=True)

    for job_id, db_path in found:
        with JobStore(db_path, create=False) as store:
            for job in store.list_jobs():
                langs = f"{job.source_lang or '?'}>{job.target_lang or '?'}"
                status_style = {
                    "completed": "good",
                    "failed": "bad",
                    "cancelled": "warn",
                }.get(job.status, "info")
                progress = (
                    f"{job.completed_segments}/{job.total_segments}" if job.total_segments else "-"
                )
                table.add_row(
                    job_id,
                    langs,
                    f"[{status_style}]{job.status}[/{status_style}]",
                    progress,
                    f"${job.cost_usd:,.2f}",
                    job.updated_at.replace("T", " ").removesuffix("+00:00")[:16],
                )
    console().print(table)


@jobs_app.command("rm")
@handle_errors
def jobs_rm(
    job_id: Annotated[str, typer.Argument(help="Job id to delete.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a job and everything stored with it."""
    paths = job_paths(job_id)
    if not paths["dir"].is_dir():
        err_console().print(f"[warn]No such job:[/warn] {job_id}")
        raise typer.Exit(code=1)
    if not yes:
        typer.confirm(f"Delete job {job_id} and all its data?", abort=True)
    shutil.rmtree(paths["dir"])
    console().print(f"[good]removed[/good] {job_id}")


@jobs_app.command("prune")
@handle_errors
def jobs_prune(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete jobs whose source PDF no longer exists on disk."""
    stale: list[str] = []
    for job_id, db_path in discover_jobs():
        with JobStore(db_path, create=False) as store:
            for job in store.list_jobs():
                if not Path(job.source_path).exists():
                    stale.append(job_id)
    if not stale:
        console().print("[muted]Nothing to prune.[/muted]")
        return
    console().print("These jobs' source files are gone:")
    for job_id in stale:
        console().print(f"  {job_id}")
    if not yes:
        typer.confirm(f"Delete {len(stale)} job(s)?", abort=True)
    for job_id in stale:
        shutil.rmtree(job_paths(job_id)["dir"], ignore_errors=True)
    console().print(f"[good]pruned[/good] {len(stale)} job(s)")


@app.command()
@handle_errors
def status(
    job_id: Annotated[str | None, typer.Argument(help="Job id; omit for the most recent.")] = None,
) -> None:
    """Show progress, cost and per-status segment counts for a job."""
    found = discover_jobs()
    if not found:
        console().print("[muted]No jobs yet.[/muted]")
        return
    target = job_id or found[0][0]
    paths = job_paths(target)
    if not paths["db"].is_file():
        err_console().print(f"[warn]No such job:[/warn] {target}")
        raise typer.Exit(code=1)

    with JobStore(paths["db"], create=False) as store:
        job = store.get_job(target)
        if job is None:
            err_console().print(f"[warn]Job database has no row for:[/warn] {target}")
            raise typer.Exit(code=1)
        counts = store.segment_counts(target)

    console().print(f"[heading]{job.id}[/heading]")
    console().print(f"  source     {job.source_path}")
    console().print(f"  languages  {job.source_lang or '?'} → {job.target_lang or '?'}")
    console().print(f"  status     {job.status}")
    console().print(f"  segments   {job.completed_segments}/{job.total_segments}")
    console().print(f"  cost       ${job.cost_usd:,.4f}")
    console().print(f"  updated    {job.updated_at}")
    if counts:
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        console().print(f"  breakdown  {breakdown}")
    pending = counts.get("pending", 0) + counts.get("failed", 0)
    if pending:
        console().print(f"\n[info]resume with:[/info] folioai resume {job.id}")


# --------------------------------------------------------------------------------
# pipeline commands
# --------------------------------------------------------------------------------


# Named explicitly: the function cannot be called `paths` without shadowing the module of
# that name, and Typer would otherwise expose it as "paths-command".
@app.command("paths")
@handle_errors
def paths_command() -> None:
    """Show where folioai reads and writes, and which .env files it loaded."""
    from rich import box
    from rich.table import Table

    from .env import PATH_VARIABLES, dotenv_paths
    from .paths import describe_paths

    _settings()  # loads the .env files, which is what decides these answers

    table = Table(
        box=box.SIMPLE_HEAD,
        pad_edge=False,
        padding=(0, 1),
        title="folioai paths",
        title_style="heading",
    )
    table.add_column("what", no_wrap=True)
    table.add_column("where", overflow="fold")
    for name, path in describe_paths().items():
        exists = "" if path.exists() else "  [muted](not created yet)[/muted]"
        table.add_row(name, f"{path}{exists}")
    console().print(table)

    console().print(
        "\n[heading].env files[/heading]  [muted](later ones do not override earlier ones)[/muted]"
    )
    for path in dotenv_paths():
        mark = "[good]read[/good]" if path.is_file() else "[muted]absent[/muted]"
        console().print(f"  {mark}  {path}")

    set_here = [name for name in PATH_VARIABLES if os.environ.get(name)]
    if set_here:
        console().print("\n[heading]overridden[/heading]  " + ", ".join(set_here))
    else:
        console().print(
            "\n[muted]No path variables set; everything is at its default. "
            "See config/.env.example to change that.[/muted]"
        )


@app.command()
@handle_errors
def probe(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Source PDF.")],
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the probe result as JSON here.")
    ] = None,
) -> None:
    """Diagnose a PDF: text layer, fonts, columns, images, language. Costs nothing."""
    from .extract.probe import probe_pdf, render_probe_report

    settings = _settings()
    result = probe_pdf(pdf, settings)
    console().print(render_probe_report(result))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console().print(f"[good]wrote[/good] {json_out}")


@app.command()
@handle_errors
def extract(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Source PDF.")],
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Write the IR JSON here.")] = None,
    markdown: Annotated[
        Path | None, typer.Option("--markdown", "-m", help="Also write Markdown here.")
    ] = None,
    extractor: Annotated[
        str | None, typer.Option("--extractor", help="auto|pymupdf|poppler|ocr|marker")
    ] = None,
    ocr_lang: Annotated[str | None, typer.Option("--ocr-lang")] = None,
    audit: Annotated[
        Path | None, typer.Option("--audit", help="Write the cleaning audit log here.")
    ] = None,
) -> None:
    """Extract, clean and structure a PDF into the document IR. Costs nothing."""
    from .extract.pipeline import extract_document
    from .render.markdown import document_to_markdown

    settings = _settings(**{"extraction.extractor": extractor, "extraction.ocr_lang": ocr_lang})
    result = extract_document(pdf, settings)
    doc = result.document

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        console().print(f"[good]wrote IR[/good] {out}  ({len(doc.blocks)} blocks)")
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(document_to_markdown(doc), encoding="utf-8")
        console().print(f"[good]wrote Markdown[/good] {markdown}")
    if audit is not None:
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(result.audit.model_dump_json(indent=2), encoding="utf-8")
        console().print(f"[good]wrote audit[/good] {audit}")
    if out is None and markdown is None:
        console().print(document_to_markdown(doc))
    else:
        console().print(result.summary_table())


@app.command()
@handle_errors
def estimate(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    to: Annotated[str, typer.Option("--to", help="Target language (BCP-47 or plain name).")],
    translator_model: Annotated[str | None, typer.Option("--translator-model")] = None,
    evaluator_model: Annotated[str | None, typer.Option("--evaluator-model")] = None,
    eval_sample: Annotated[float | None, typer.Option("--eval-sample", min=0.0, max=1.0)] = None,
    eval_mode: Annotated[
        str | None, typer.Option("--eval-mode", help="direct|back-translation|both")
    ] = None,
    batch_tokens: Annotated[int | None, typer.Option("--batch-tokens", min=1)] = None,
    max_cost: Annotated[float | None, typer.Option("--max-cost", min=0.0)] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the estimate as JSON here.")
    ] = None,
) -> None:
    """Project the cost of translating a book. Makes no paid calls at all."""
    from dataclasses import asdict

    from .estimate import estimate_document, render_estimate
    from .extract.pipeline import extract_document

    settings = _settings(
        **{
            "models.translator": translator_model,
            "models.evaluator": evaluator_model,
            "evaluation.sample": eval_sample,
            "evaluation.mode": eval_mode,
            "translation.batch_tokens": batch_tokens,
            "budget.max_cost_usd": max_cost,
        }
    )
    result = extract_document(pdf, settings)
    projection = estimate_document(result.document, settings, target_lang=to, source_path=pdf)

    console().print(render_estimate(projection, settings))
    for warning in projection.warnings:
        console().print(f"[warn]note:[/warn] {warning}")
    console().print(
        "\n[muted]No paid calls were made. Extraction and segmentation are free; the "
        "figures above are a projection, not a quote.[/muted]"
    )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(asdict(projection), indent=2), encoding="utf-8")
        console().print(f"[good]wrote[/good] {json_out}")


def _extract_with_vision(pdf: Path, settings: Settings) -> Any:
    """Extract with the vision fallback enabled (§4.2).

    Kept out of ``prepare_job`` because it needs an event loop and an LLM client, and
    extraction is otherwise entirely free and offline -- a property worth not giving up
    just to save a branch here.
    """
    import asyncio

    from .extract.pipeline import extract_document_with_vision
    from .llm.client import OpenAICompatibleClient

    client = OpenAICompatibleClient(settings)

    async def run() -> Any:
        try:
            return await extract_document_with_vision(pdf, settings, client)
        finally:
            await client.aclose()

    console().print(
        "[muted]vision fallback enabled: badly extracted pages will be re-read by "
        f"{settings.models.vision} (max {settings.extraction.vision_max_pages} pages)[/muted]"
    )
    result = asyncio.run(run())
    for warning in result.warnings:
        console().print(f"[warn]note:[/warn] {warning}")
    return result.document


@app.command()
@handle_errors
def translate(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Source PDF.")],
    to: Annotated[str, typer.Option("--to", help="Target language (BCP-47 or plain name).")],
    from_lang: Annotated[
        str | None, typer.Option("--from", help="Override the detected source language.")
    ] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Style profile name or path.")
    ] = None,
    translator_model: Annotated[str | None, typer.Option("--translator-model")] = None,
    evaluator_model: Annotated[str | None, typer.Option("--evaluator-model")] = None,
    escalation_model: Annotated[str | None, typer.Option("--escalation-model")] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Any OpenAI-compatible endpoint.")
    ] = None,
    min_score: Annotated[int | None, typer.Option("--min-score", min=0, max=100)] = None,
    eval_mode: Annotated[
        str | None, typer.Option("--eval-mode", help="direct|back-translation|both")
    ] = None,
    eval_sample: Annotated[float | None, typer.Option("--eval-sample", min=0.0, max=1.0)] = None,
    max_attempts: Annotated[int | None, typer.Option("--max-attempts", min=1, max=10)] = None,
    concurrency: Annotated[int | None, typer.Option("--concurrency", min=1, max=64)] = None,
    batch_tokens: Annotated[int | None, typer.Option("--batch-tokens", min=1)] = None,
    extractor: Annotated[
        str | None, typer.Option("--extractor", help="auto|pymupdf|poppler|ocr|marker")
    ] = None,
    ocr_lang: Annotated[str | None, typer.Option("--ocr-lang")] = None,
    chapters: Annotated[
        str | None,
        typer.Option("--chapters", help="Translate a subset, e.g. 3-7,12."),
    ] = None,
    vision_fallback: Annotated[
        bool,
        typer.Option("--vision-fallback", help="Re-read badly extracted pages with a model."),
    ] = False,
    max_cost: Annotated[float | None, typer.Option("--max-cost", min=0.0)] = None,
    formats: Annotated[
        str | None, typer.Option("--format", help="Export on completion: md,epub,pdf,...")
    ] = None,
    layout: Annotated[
        str, typer.Option("--layout", help="target-only|bilingual-paragraph|annotated")
    ] = "target-only",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Estimate and show the plan; make no paid calls.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmations.")] = False,
) -> None:
    """Translate a book end to end."""
    settings = _settings(
        **{
            "models.translator": translator_model,
            "models.evaluator": evaluator_model,
            "models.escalation": escalation_model,
            "llm.base_url": base_url,
            "evaluation.min_score": min_score,
            "evaluation.mode": eval_mode,
            "evaluation.sample": eval_sample,
            "retry.max_attempts": max_attempts,
            "translation.concurrency": concurrency,
            "translation.batch_tokens": batch_tokens,
            "extraction.extractor": extractor,
            "extraction.ocr_lang": ocr_lang,
            "budget.max_cost_usd": max_cost,
        }
    )
    from .jobs import prepare_job

    document = None
    if vision_fallback:
        document = _extract_with_vision(pdf, settings)

    context = prepare_job(
        pdf,
        settings,
        target_lang=to,
        profile=profile,
        document=document,
        chapters=chapters,
    )
    try:
        if chapters:
            from .chapters import parse_selection, selection_summary

            console().print(
                f"[info]chapter subset:[/info] "
                f"{selection_summary(context.document, parse_selection(chapters))}"
            )
        if from_lang:
            context.document.source_lang = from_lang
            context.document.save(context.paths["ir"])

        _show_structure(context, confirm=not yes)

        if dry_run:
            from .estimate import estimate_document, render_estimate

            projection = estimate_document(
                context.document, settings, target_lang=to, source_path=pdf
            )
            console().print(render_estimate(projection, settings))
            for warning in projection.warnings:
                console().print(f"[warn]note:[/warn] {warning}")
            console().print(
                "\n[muted]--dry-run: no paid calls were made. Remove it to run the "
                "translation.[/muted]"
            )
            return

        _run_and_report(context, settings)
        if formats:
            _export_after_run(context, settings, formats, layout)
    finally:
        context.close()


@app.command()
@handle_errors
def resume(
    job_id: Annotated[str, typer.Argument(help="Job id from 'folioai jobs list'.")],
    max_cost: Annotated[float | None, typer.Option("--max-cost", min=0.0)] = None,
    concurrency: Annotated[int | None, typer.Option("--concurrency", min=1, max=64)] = None,
) -> None:
    """Resume an interrupted job, picking up every unfinished segment."""
    from .jobs import reopen_job

    settings = _settings(
        **{"budget.max_cost_usd": max_cost, "translation.concurrency": concurrency}
    )
    context = reopen_job(job_id, settings)
    try:
        counts = context.store.segment_counts(job_id)
        pending = len(context.store.pending_segments(job_id))
        if not pending:
            console().print(
                f"[good]Nothing to do:[/good] all {counts.get('done', 0)} segments are finished."
            )
            return
        console().print(f"Resuming [info]{job_id}[/info]: {pending} segment(s) left.")
        _run_and_report(context, settings)
    finally:
        context.close()


def _show_structure(context: Any, *, confirm: bool) -> None:
    """Show the detected chapter structure before spending money on it (§4.4)."""
    from rich import box
    from rich.table import Table

    doc = context.document
    table = Table(
        box=box.SIMPLE_HEAD,
        pad_edge=False,
        padding=(0, 1),
        title=f"{doc.title or context.source_path.name} -> {context.target_lang}",
        title_style="heading",
    )
    table.add_column("chapter", no_wrap=True)
    table.add_column("title", overflow="ellipsis", max_width=34)
    table.add_column("page", justify="right", no_wrap=True)
    table.add_column("blocks", justify="right", no_wrap=True)
    table.add_column("words", justify="right", no_wrap=True)
    table.add_column("", no_wrap=True)

    counts = [
        sum(b.word_count for b in doc.blocks if b.chapter_id == chapter.id)
        for chapter in doc.chapters
    ]
    median = sorted(counts)[len(counts) // 2] if counts else 0
    for chapter, words in zip(doc.chapters, counts, strict=True):
        blocks = sum(1 for b in doc.blocks if b.chapter_id == chapter.id)
        note = ""
        if words == 0:
            note = "[bad]empty[/bad]"
        elif words < 200:
            note = "[warn]short[/warn]"
        elif median and words > median * 3:
            note = "[warn]long[/warn]"
        table.add_row(
            chapter.id,
            chapter.title,
            str(chapter.start_page or "-"),
            str(blocks),
            f"{words:,}",
            note,
        )
    console().print(table)
    console().print(
        f"[muted]{len(doc.blocks):,} blocks, {doc.word_count():,} words, "
        f"source language {doc.source_lang}[/muted]"
    )
    if confirm:
        typer.confirm("Translate with this structure?", default=True, abort=True)


def _run_and_report(context: Any, settings: Settings) -> None:
    """Run a job with a live progress display and print the summary."""
    import asyncio

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    from .jobs import run_job
    from .llm.client import OpenAICompatibleClient

    total = len(context.store.pending_segments(context.job_id))
    use_tty = bool(_STATE["tty"])

    # Build the client first. A missing key is the most common failure there is, and
    # discovering it after a spinner has been turning for a second reads like a crash.
    client = OpenAICompatibleClient(
        settings,
        on_usage=lambda model, purpose, prompt, completion, usd: context.store.record_usage(
            job_id=context.job_id,
            model=model,
            endpoint=purpose,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=usd,
        ),
    )

    if not use_tty:
        # --no-tty: one line per event, for cron and CI.
        def on_progress(stats: Any) -> None:
            console().print(
                f"batch {stats.batches_done}/{stats.batches_total} "
                f"segments {stats.completed}/{stats.total_segments} "
                f"mean {stats.mean_score} review {stats.needs_review} "
                f"cost ${stats.cost_usd:.4f}"
            )

        _, stats = asyncio.run(run_job(context, settings, client=client, on_progress=on_progress))
    else:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[extra]}"),
            TimeRemainingColumn(),
            console=console(),
        )
        with progress:
            task_id = progress.add_task("translating", total=total or None, extra="")

            def on_progress(stats: Any) -> None:
                progress.update(
                    task_id,
                    completed=stats.completed,
                    extra=(
                        f"mean {stats.mean_score}  review {stats.needs_review}  "
                        f"${stats.cost_usd:.4f}"
                    ),
                )

            _, stats = asyncio.run(
                run_job(context, settings, client=client, on_progress=on_progress)
            )

    console().print(
        f"\n[good]done[/good] {stats.completed} segments, mean score "
        f"[heading]{stats.mean_score}[/heading], {stats.needs_review} need review, "
        f"{stats.retries} retries, cost ${stats.cost_usd:,.4f}"
    )
    console().print(f"[muted]translated IR:[/muted] {context.paths['translated_ir']}")
    remaining = len(context.store.pending_segments(context.job_id))
    if remaining:
        console().print(
            f"[warn]{remaining} segment(s) unfinished.[/warn] Resume with: "
            f"folioai resume {context.job_id}"
        )
    if stats.needs_review:
        console().print(f"[info]review them with:[/info] folioai review {context.job_id}")


def _export_after_run(context: Any, settings: Settings, formats: str, layout: str) -> None:
    """Export immediately on completion, per --format on `translate` (§16)."""
    from .export import export_job, parse_formats

    result = export_job(
        context.job_id,
        settings,
        formats=parse_formats(formats),
        layout=layout,  # type: ignore[arg-type]
    )
    for path in result.files:
        console().print(f"[good]wrote[/good] {path}")
    for warning in result.warnings:
        console().print(f"[warn]note:[/warn] {warning}")


@app.command()
@handle_errors
def review(
    job_id: Annotated[str, typer.Argument(help="Job id from 'folioai jobs list'.")],
    max_score: Annotated[
        float | None,
        typer.Option("--max-score", help="Also review anything scoring at or below this."),
    ] = None,
) -> None:
    """Walk flagged segments: accept, edit in $EDITOR, or send back for another attempt."""
    from .jobs import reopen_job
    from .review import ReviewItem, collect_items, run_review

    settings = _settings()
    context = reopen_job(job_id, settings)
    try:
        items = collect_items(context.store, job_id, max_score=max_score)
        if not items:
            console().print("[good]Nothing flagged for review.[/good]")
            return
        console().print(
            f"[heading]{len(items)} segment(s) to review[/heading] "
            "[muted](worst first; a=accept, e=edit, r=re-translate, s=skip, q=quit)[/muted]\n"
        )

        def present(item: ReviewItem, index: int, total: int) -> str:
            score = f"{item.score:.0f}" if item.score is not None else "no score"
            console().rule(f"[muted]{index}/{total}[/muted]  {item.id}  score {score}")
            console().print("[muted]source[/muted]")
            console().print(item.segment.source_text)
            console().print("\n[muted]translation[/muted]")
            console().print(item.segment.final_text or "")
            for issue in item.issues:
                console().print(
                    f"  [warn]{issue.get('severity', '')}[/warn] "
                    f"{issue.get('dimension', '')}: {issue.get('explanation', '')}"
                )
            choice = typer.prompt(
                "\n[a]ccept / [e]dit / [r]e-translate / [s]kip / [q]uit",
                default="s",
                show_default=False,
            )
            return {
                "a": "accept",
                "e": "edit",
                "r": "retranslate",
                "s": "skip",
                "q": "quit",
            }.get(choice.strip().lower()[:1], "skip")

        def ask_instruction(item: ReviewItem) -> str:  # noqa: ARG001 - callback signature
            instruction: str = typer.prompt(
                "What should the next attempt do differently?", default=""
            )
            return instruction

        outcome = run_review(
            context.store,
            job_id,
            items,
            present=present,  # type: ignore[arg-type]
            ask_instruction=ask_instruction,
        )
        console().print(
            f"\n[good]reviewed[/good] {outcome.accepted} accepted, {outcome.edited} edited, "
            f"{outcome.retranslated} queued for re-translation, {outcome.skipped} skipped"
        )
        if outcome.retranslated:
            console().print(f"[info]run:[/info] folioai resume {job_id}")
    finally:
        context.close()


@app.command()
@handle_errors
def export(
    job_id: Annotated[str, typer.Argument(help="Job id from 'folioai jobs list'.")],
    formats: Annotated[
        str, typer.Option("--format", help="md,epub,pdf,docx,html,txt (comma separated)")
    ] = "md",
    layout: Annotated[
        str,
        typer.Option(
            "--layout",
            help="target-only|bilingual-paragraph|bilingual-columns|annotated",
        ),
    ] = "target-only",
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Output directory.")] = None,
    split_chapters: Annotated[
        bool, typer.Option("--split-chapters", help="One Markdown file per chapter.")
    ] = False,
    cover: Annotated[Path | None, typer.Option("--cover", help="Cover image for the EPUB.")] = None,
) -> None:
    """Render a completed job into one or more output formats."""
    from .export import export_job, parse_formats

    settings = _settings()
    if layout not in {"target-only", "bilingual-paragraph", "bilingual-columns", "annotated"}:
        raise ConfigError(
            f"Unknown layout: {layout!r}.",
            remedy=(
                "Choose one of: target-only, bilingual-paragraph, bilingual-columns, annotated."
            ),
        )

    result = export_job(
        job_id,
        settings,
        formats=parse_formats(formats),
        layout=layout,  # type: ignore[arg-type]
        out_dir=out,
        split_chapters=split_chapters,
        cover=cover,
    )
    for path in result.files:
        console().print(f"[good]wrote[/good] {path}")
    for warning in result.warnings:
        console().print(f"[warn]note:[/warn] {warning}")
    if not result.files:
        console().print("[bad]Nothing was written.[/bad]")
        raise typer.Exit(code=9)


@app.command()
@handle_errors
def report(
    job_id: Annotated[str, typer.Argument(help="Job id from 'folioai jobs list'.")],
    out: Annotated[
        Path | None, typer.Option("-o", "--out", help="Where to write the HTML report.")
    ] = None,
    open_it: Annotated[
        bool, typer.Option("--open", help="Open the report in your browser when done.")
    ] = False,
) -> None:
    """Write the self-contained HTML quality report."""
    from .glossary_build import audit_adherence
    from .jobs import reopen_job
    from .report import gather, write_report

    settings = _settings()
    context = reopen_job(job_id, settings)
    try:
        segments = context.store.list_segments(job_id)
        translations = {s.segment_id: s.final_text or "" for s in segments if s.final_text}
        sources = {s.segment_id: s.source_text for s in segments}
        usages = (
            audit_adherence(context.glossary, sources, translations)
            if context.glossary.terms and translations
            else []
        )

        data = gather(context.store, job_id, document=context.document, glossary_usages=usages)
        target = out or (context.paths["dir"] / f"{job_id}-report.html")
        write_report(data, target)
        console().print(
            f"[good]wrote[/good] {target}  [muted]({data.done_segments} segments, mean "
            f"{data.mean_score}, {data.needs_review} flagged)[/muted]"
        )
        if open_it:
            import webbrowser

            webbrowser.open(target.resolve().as_uri())
    finally:
        context.close()


def _open_target(target: str, settings: Settings, *, target_lang: str | None = None) -> Any:
    """Open a job by id, or prepare one from a PDF path -- whichever the user named."""
    from .jobs import prepare_job, reopen_job

    path = Path(target)
    if path.is_file():
        if not target_lang:
            raise ConfigError(
                "Building a glossary from a PDF needs to know the target language.",
                remedy="Pass --to, for example: folioai glossary build book.pdf --to de",
            )
        return prepare_job(path, settings, target_lang=target_lang)
    return reopen_job(target, settings)


def _render_glossary(glossary: Any, *, title: str) -> None:
    from rich import box
    from rich.table import Table

    if not glossary.terms:
        console().print("[muted]The glossary is empty.[/muted]")
        return
    table = Table(
        box=box.SIMPLE_HEAD, pad_edge=False, padding=(0, 1), title=title, title_style="heading"
    )
    table.add_column("source", overflow="ellipsis", max_width=26)
    table.add_column("target", overflow="ellipsis", max_width=26)
    table.add_column("kind", no_wrap=True)
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("", no_wrap=True)
    for term in sorted(glossary.terms, key=lambda t: (-t.occurrences, t.source.lower())):
        table.add_row(
            term.source,
            term.target,
            term.kind,
            str(term.occurrences or "-"),
            "[info]locked[/info]" if term.locked else "",
        )
    console().print(table)


@glossary_app.command("build")
@handle_errors
def glossary_build(
    target: Annotated[str, typer.Argument(help="Job id, or a PDF path with --to.")],
    to: Annotated[str | None, typer.Option("--to", help="Target language, for a PDF.")] = None,
    samples: Annotated[
        int, typer.Option("--samples", min=1, max=60, help="Passages to sample.")
    ] = 10,
    min_occurrences: Annotated[int, typer.Option("--min-occurrences", min=1)] = 3,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the editor review.")] = False,
) -> None:
    """Extract proper nouns and recurring terms into a reviewable glossary."""
    import asyncio

    from .glossary_build import build_glossary, review_glossary
    from .llm.client import OpenAICompatibleClient

    settings = _settings()
    context = _open_target(target, settings, target_lang=to)
    try:
        client = OpenAICompatibleClient(settings)

        async def run() -> Any:
            try:
                return await build_glossary(
                    context.document,
                    client,
                    settings,
                    target_lang=context.target_lang,
                    existing=context.glossary,
                    samples=samples,
                    min_occurrences=min_occurrences,
                )
            finally:
                await client.aclose()

        draft = asyncio.run(run())
        console().print(f"[good]built[/good] {draft.summary()}, {format_usd(draft.cost_usd)}")
        if draft.rejected:
            preview = ", ".join(f"{s} ({n})" for s, n in draft.rejected[:6])
            console().print(f"[muted]rejected as too rare: {preview}[/muted]")

        glossary = draft.glossary
        if not yes:
            console().print(
                f"[muted]opening {context.paths['glossary']} for review; "
                "lock a term with 'locked: true' to force it into every prompt[/muted]"
            )
            glossary = review_glossary(glossary, context.paths["glossary"])
        else:
            glossary.save(context.paths["glossary"])

        context.store.upsert_glossary(context.job_id, glossary.to_rows())
        _render_glossary(glossary, title=f"{context.job_id} glossary")
        console().print(f"[muted]saved to[/muted] {context.paths['glossary']}")
    finally:
        context.close()


@glossary_app.command("show")
@handle_errors
def glossary_show(
    job_id: Annotated[str, typer.Argument()],
    audit: Annotated[
        bool, typer.Option("--audit", help="Check adherence across the finished translation.")
    ] = False,
) -> None:
    """Print a job's glossary, optionally auditing how it was actually used."""
    from .glossary_build import audit_adherence

    settings = _settings()
    context = _open_target(job_id, settings)
    try:
        _render_glossary(context.glossary, title=f"{job_id} glossary")
        if not audit:
            return

        segments = context.store.list_segments(job_id)
        translations = {s.segment_id: s.final_text or "" for s in segments if s.final_text}
        if not translations:
            console().print("[muted]Nothing translated yet, so there is nothing to audit.[/muted]")
            return

        sources = {s.segment_id: s.source_text for s in segments}
        usages = audit_adherence(context.glossary, sources, translations)
        inconsistent = [u for u in usages if not u.consistent]
        if not inconsistent:
            console().print(
                f"[good]consistent[/good] all {len(usages)} term(s) rendered the same way "
                "throughout"
            )
            return
        console().print(f"[warn]{len(inconsistent)} term(s) rendered inconsistently:[/warn]")
        for usage in inconsistent:
            console().print(f"  [heading]{usage.term.source}[/heading] -> {usage.term.target}")
            for rendering, ids in sorted(usage.renderings.items(), key=lambda kv: -len(kv[1])):
                sample = ", ".join(ids[:4]) + ("…" if len(ids) > 4 else "")
                console().print(f"    {len(ids):4d}x  {rendering}  [muted]{sample}[/muted]")
    finally:
        context.close()


@glossary_app.command("edit")
@handle_errors
def glossary_edit(job_id: Annotated[str, typer.Argument()]) -> None:
    """Open a job's glossary in $EDITOR and save what you write back."""
    from .glossary_build import review_glossary

    settings = _settings()
    context = _open_target(job_id, settings)
    try:
        glossary = review_glossary(context.glossary, context.paths["glossary"])
        context.store.upsert_glossary(job_id, glossary.to_rows())
        _render_glossary(glossary, title=f"{job_id} glossary")
    finally:
        context.close()


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except FolioError as exc:  # safety net for anything raised outside a command body
        err_console().print(f"[bad]error:[/bad] {exc.format_for_user()}")
        sys.exit(exc.exit_code)


if __name__ == "__main__":
    main()
