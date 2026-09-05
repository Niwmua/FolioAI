"""Job lifecycle: create, run, resume (brief §12).

Both ``translate`` and ``resume`` come through here, which is what makes them the same code
path -- a resume that took a different route through the program would be a resume nobody
should trust.

Everything a run needs to restart is on disk before any paid call happens: the IR, the
config it was launched with, and one ``segments`` row per block. Killing the process at any
point after that leaves a job ``resume`` can finish exactly once.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import load_profile
from .errors import BudgetExceeded, FolioError, StoreError
from .glossary import Glossary
from .ir import Document
from .llm.client import LLMClient, OpenAICompatibleClient
from .logging_setup import configure_logging, get_logger
from .orchestrate import CircuitBreakerTripped, Orchestrator, RunStats, seed_segments
from .paths import ensure_dirs, job_log_path, make_job_id, sha256_file
from .store import JobStore, job_paths

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import Settings

log = get_logger(__name__)


def _log_to_stderr() -> bool:
    """Mirror events to stderr only when the user asked for verbosity."""
    return bool(os.environ.get("FOLIOAI_LOG_STDERR"))


@dataclass(slots=True)
class JobContext:
    """An open job: its id, store, document and paths."""

    job_id: str
    store: JobStore
    document: Document
    target_lang: str
    paths: dict[str, Path]
    style_profile: dict[str, Any]
    glossary: Glossary
    source_path: Path

    def close(self) -> None:
        self.store.close()


def default_profile_name(source_lang: str, target_lang: str) -> str:
    """Shipped profile for a language pair, or the generic fallback."""
    from .config import available_profiles

    candidate = f"{source_lang.lower()}-{target_lang.lower()}"
    return candidate if candidate in available_profiles() else "generic"


def prepare_job(
    pdf: Path,
    settings: Settings,
    *,
    target_lang: str,
    profile: str | None = None,
    force_extract: bool = False,
    document: Document | None = None,
    chapters: str | None = None,
) -> JobContext:
    """Create or reopen a job for a PDF, extracting it if needed.

    Extraction output is cached in the job directory: reopening a job -- which is what
    ``resume`` does -- must not re-derive the IR, both because it is slow and because a
    different IR would invalidate every segment id already in the database.
    """
    from .extract.pipeline import extract_document

    ensure_dirs()
    digest = sha256_file(pdf)
    job_id = make_job_id(pdf, digest)
    paths = job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    configure_logging(
        log_path=job_log_path(job_id),
        level=settings.logging.level,
        force=True,
        to_stderr=_log_to_stderr(),
    )

    if document is not None:
        # Handed in by a caller that already extracted -- the vision fallback path, which
        # has to run inside an event loop and so cannot be reached from here.
        document.target_lang = target_lang
        document.save(paths["ir"])
        log.info("ir_provided", job=job_id, blocks=len(document.blocks))
    elif paths["ir"].is_file() and not force_extract:
        document = Document.load(paths["ir"])
        log.info("ir_reused", job=job_id, blocks=len(document.blocks))
    else:
        result = extract_document(pdf, settings, workdir=paths["dir"])
        document = result.document
        document.target_lang = target_lang
        document.save(paths["ir"])
        paths["probe"].write_text(result.probe.model_dump_json(indent=2), encoding="utf-8")
        paths["audit"].write_text(result.audit.model_dump_json(indent=2), encoding="utf-8")
        log.info("ir_extracted", job=job_id, blocks=len(document.blocks))

    if chapters:
        # Applied to the in-memory document only. The IR on disk stays complete, so widening
        # the selection later needs no re-extraction and renumbers nothing (§16).
        from .chapters import apply_selection, parse_selection

        apply_selection(document, parse_selection(chapters))

    store = JobStore(paths["db"])
    store.create_job(
        job_id=job_id,
        source_path=pdf.resolve(),
        source_sha256=digest,
        config=settings.model_dump(mode="json"),
        source_lang=document.source_lang,
        target_lang=target_lang,
    )
    seed_segments(store, job_id, document)

    profile_name = profile or default_profile_name(document.source_lang, target_lang)
    style_profile = load_profile(profile_name)

    glossary = (
        Glossary.load(paths["glossary"])
        if paths["glossary"].is_file()
        else Glossary.from_rows(store.list_glossary(job_id))
    )

    return JobContext(
        job_id=job_id,
        store=store,
        document=document,
        target_lang=target_lang,
        paths=paths,
        style_profile=style_profile,
        glossary=glossary,
        source_path=pdf,
    )


def reopen_job(job_id: str, settings: Settings) -> JobContext:
    """Reopen an existing job for ``resume``.

    Raises:
        StoreError: if the job or its IR is missing.
    """
    paths = job_paths(job_id)
    if not paths["db"].is_file():
        raise StoreError(
            f"No job named {job_id!r}.",
            remedy="Run 'folioai jobs list' to see what exists.",
            context={"job_id": job_id},
        )
    if not paths["ir"].is_file():
        raise StoreError(
            f"Job {job_id} has no extracted document ({paths['ir']} is missing).",
            remedy="Start it again with 'folioai translate <pdf> --to <lang>'.",
            context={"job_id": job_id},
        )

    configure_logging(
        log_path=job_log_path(job_id),
        level=settings.logging.level,
        force=True,
        to_stderr=_log_to_stderr(),
    )
    store = JobStore(paths["db"], create=False)
    record = store.get_job(job_id)
    if record is None:
        raise StoreError(
            f"Job database for {job_id} has no job row.",
            remedy="The job is corrupt; delete it with 'folioai jobs rm' and start again.",
        )

    document = Document.load(paths["ir"])
    target_lang = record.target_lang or document.target_lang or "und"
    profile_name = default_profile_name(document.source_lang, target_lang)
    glossary = (
        Glossary.load(paths["glossary"])
        if paths["glossary"].is_file()
        else Glossary.from_rows(store.list_glossary(job_id))
    )
    return JobContext(
        job_id=job_id,
        store=store,
        document=document,
        target_lang=target_lang,
        paths=paths,
        style_profile=load_profile(profile_name),
        glossary=glossary,
        source_path=Path(record.source_path),
    )


def build_translated_document(
    document: Document, translations: dict[str, str], target_lang: str
) -> Document:
    """Apply translations to the IR, keeping it parallel to the source (§21.2).

    Blocks with no translation keep their source text rather than disappearing. The
    assertion at the end is the acceptance criterion, enforced in code.
    """
    translated = document.model_copy(deep=True)
    translated.target_lang = target_lang
    for block in translated.blocks:
        replacement = translations.get(block.id)
        if replacement is not None and replacement.strip():
            block.text = replacement
    document.assert_parallel_to(translated)
    return translated


async def run_job(
    context: JobContext,
    settings: Settings,
    *,
    client: LLMClient | None = None,
    on_progress: Callable[[RunStats], None] | None = None,
) -> tuple[Document, RunStats]:
    """Run (or finish) a job and write the translated IR.

    SIGINT is trapped: in-flight tasks are cancelled, everything already finished stays
    committed, and the caller is told the resume command (§12).
    """
    owns_client = client is None
    llm: LLMClient = client or OpenAICompatibleClient(
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

    orchestrator = Orchestrator(
        client=llm,
        settings=settings,
        document=context.document,
        target_lang=context.target_lang,
        store=context.store,
        job_id=context.job_id,
        style_profile=context.style_profile,
        glossary=context.glossary,
        on_progress=on_progress,
    )

    context.store.update_job(context.job_id, status="translating")
    interrupted = False
    task = asyncio.ensure_future(orchestrator.run())
    _install_sigint(task)

    try:
        await task
    except asyncio.CancelledError:
        interrupted = True
        log.warning("run_interrupted", job=context.job_id)
    except (BudgetExceeded, CircuitBreakerTripped) as exc:
        context.store.update_job(context.job_id, status="failed")
        _write_partial(context, orchestrator)
        log.warning("run_stopped", job=context.job_id, reason=exc.message)
        raise
    except FolioError:
        context.store.update_job(context.job_id, status="failed")
        _write_partial(context, orchestrator)
        raise
    finally:
        if owns_client:
            await llm.aclose()

    translated = build_translated_document(
        context.document, orchestrator.translations, context.target_lang
    )
    translated.save(context.paths["translated_ir"])

    counts = context.store.segment_counts(context.job_id)
    remaining = counts.get("pending", 0) + counts.get("failed", 0)
    status = "cancelled" if interrupted else ("translating" if remaining else "completed")
    context.store.update_job(context.job_id, status=status)

    _write_run_summary(context, orchestrator.stats)
    return translated, orchestrator.stats


def _write_partial(context: JobContext, orchestrator: Orchestrator) -> None:
    """Save whatever was translated before the stop, so nothing done is lost."""
    with suppress(Exception):
        partial = build_translated_document(
            context.document, orchestrator.translations, context.target_lang
        )
        partial.save(context.paths["translated_ir"])


def _write_run_summary(context: JobContext, stats: RunStats) -> None:
    summary = {
        "job_id": context.job_id,
        "segments": stats.completed,
        "needs_review": stats.needs_review,
        "retries": stats.retries,
        "mean_score": stats.mean_score,
        "cost_usd": round(stats.cost_usd, 6),
    }
    with suppress(OSError):
        (context.paths["dir"] / "run-summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def _install_sigint(task: asyncio.Future[Any]) -> None:
    """Trap SIGINT so a kill mid-book flushes cleanly instead of shredding the run.

    Falls back to leaving Python's default handler in place where the loop cannot install
    one (Windows ProactorEventLoop refuses ``add_signal_handler``); the store commits per
    batch either way, so a hard kill still resumes -- it just prints an uglier traceback.
    """
    loop = asyncio.get_running_loop()
    with suppress(NotImplementedError, RuntimeError, ValueError):
        loop.add_signal_handler(signal.SIGINT, task.cancel)
