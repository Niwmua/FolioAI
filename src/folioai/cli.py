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
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from rich.table import Table

from . import __version__
from .config import Settings, load_settings
from .errors import FolioError
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
    level = {0: "INFO", 1: "DEBUG"}.get(int(_STATE["verbosity"]), "DEBUG")
    if _STATE["verbosity"] == 0:
        level = settings.logging.level
    configure_logging(level=level, force=True)
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

    table = Table(title=f"folioai jobs ({home_dir()})", title_style="heading", expand=False)
    table.add_column("job id", style="info", no_wrap=True)
    table.add_column("source")
    table.add_column("lang", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("progress", justify="right", no_wrap=True)
    table.add_column("cost", justify="right", no_wrap=True)
    table.add_column("updated", no_wrap=True)

    for job_id, db_path in found:
        with JobStore(db_path, create=False) as store:
            for job in store.list_jobs():
                langs = f"{job.source_lang or '?'}→{job.target_lang or '?'}"
                pct = f"{job.progress * 100:5.1f}%" if job.total_segments else "  —  "
                status_style = {
                    "completed": "good",
                    "failed": "bad",
                    "cancelled": "warn",
                }.get(job.status, "info")
                table.add_row(
                    job_id,
                    Path(job.source_path).name,
                    langs,
                    f"[{status_style}]{job.status}[/{status_style}]",
                    f"{pct} ({job.completed_segments}/{job.total_segments})",
                    f"${job.cost_usd:,.2f}",
                    job.updated_at.replace("T", " ").removesuffix("+00:00"),
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
    to: Annotated[str, typer.Option("--to", help="Target language.")],
) -> None:
    """Project the cost of translating a book before spending anything."""
    raise NotImplementedError("estimate arrives in milestone 3 (LLM plumbing).")


@app.command()
@handle_errors
def translate(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    to: Annotated[str, typer.Option("--to", help="Target language.")],
) -> None:
    """Translate a book end to end."""
    raise NotImplementedError("translate arrives in milestone 4 (translation engine).")


@app.command()
@handle_errors
def resume(job_id: Annotated[str, typer.Argument()]) -> None:
    """Resume an interrupted job."""
    raise NotImplementedError("resume arrives in milestone 4 (translation engine).")


@app.command()
@handle_errors
def review(job_id: Annotated[str, typer.Argument()]) -> None:
    """Walk flagged segments and accept, edit, or re-translate them."""
    raise NotImplementedError("review arrives in milestone 7 (export, reports, review).")


@app.command()
@handle_errors
def export(
    job_id: Annotated[str, typer.Argument()],
    formats: Annotated[str, typer.Option("--format", help="md,epub,pdf,docx,html,txt")] = "md",
) -> None:
    """Render a completed job into one or more output formats."""
    raise NotImplementedError("export arrives in milestone 7 (export, reports, review).")


@app.command()
@handle_errors
def report(job_id: Annotated[str, typer.Argument()]) -> None:
    """Write the self-contained HTML quality report."""
    raise NotImplementedError("report arrives in milestone 7 (export, reports, review).")


@glossary_app.command("build")
@handle_errors
def glossary_build(target: Annotated[str, typer.Argument(help="Job id or PDF path.")]) -> None:
    """Extract proper nouns and recurring terms into a reviewable glossary."""
    raise NotImplementedError("glossary arrives in milestone 6.")


@glossary_app.command("show")
@handle_errors
def glossary_show(job_id: Annotated[str, typer.Argument()]) -> None:
    """Print a job's glossary."""
    raise NotImplementedError("glossary arrives in milestone 6.")


@glossary_app.command("edit")
@handle_errors
def glossary_edit(job_id: Annotated[str, typer.Argument()]) -> None:
    """Open a job's glossary in $EDITOR."""
    raise NotImplementedError("glossary arrives in milestone 6.")


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except FolioError as exc:  # safety net for anything raised outside a command body
        err_console().print(f"[bad]error:[/bad] {exc.format_for_user()}")
        sys.exit(exc.exit_code)


if __name__ == "__main__":
    main()
