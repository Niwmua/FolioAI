"""Export dispatch (brief §14).

One entry point that turns a finished job into files. Its job is to decide *what* to render
-- which document, which layout, which per-segment scores the annotated layout needs -- and
hand that to the format renderers, which know nothing about jobs or databases.

Before anything is written it asserts the translated document is parallel to the source
(§21.2). Exporting a book with a missing paragraph would be the exact failure the whole
pipeline exists to prevent, and the last moment it can still be caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import RenderError
from .ir import Document
from .logging_setup import get_logger
from .render.base import Layout, RenderContext
from .render.epubcheck import ValidationResult
from .render.markdown import write_markdown
from .store import JobStore

if TYPE_CHECKING:
    from .config import Settings

log = get_logger(__name__)

FORMATS = ("md", "epub", "pdf", "docx", "html", "txt")

#: Formats that can lay text out in two columns. The others fall back to paragraph pairs.
COLUMN_CAPABLE = frozenset({"pdf", "html"})


@dataclass(slots=True)
class ExportResult:
    files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Informational, not problems: an absent epubcheck belongs here, not in warnings.
    notes: list[str] = field(default_factory=list)
    epub_validation: ValidationResult | None = None

    def describe(self) -> str:
        return ", ".join(path.name for path in self.files)


def parse_formats(value: str) -> list[str]:
    """Parse ``--format md,epub,pdf`` into a validated list."""
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in FORMATS]
    if unknown:
        raise RenderError(
            f"Unknown output format(s): {', '.join(unknown)}.",
            remedy=f"Choose from: {', '.join(FORMATS)}.",
            context={"requested": requested},
        )
    return requested or ["md"]


def build_context(
    store: JobStore,
    job_id: str,
    *,
    layout: Layout,
    source: Document,
    settings: Settings,
) -> RenderContext:
    """Assemble what the renderers need: scores, review flags, and the models used."""
    segments = store.list_segments(job_id)
    scores = {s.segment_id: s.final_score for s in segments if s.final_score is not None}
    review = {s.segment_id for s in segments if s.needs_review}

    models: dict[str, str] = {}
    rows = store.conn.execute(
        "SELECT DISTINCT model FROM attempts WHERE job_id = ? AND model != 'human'", (job_id,)
    ).fetchall()
    used = sorted(row["model"] for row in rows)
    if used:
        models["translator"] = ", ".join(used)
    models["evaluator"] = settings.models.evaluator

    return RenderContext(
        layout=layout,
        source=source,
        scores={k: float(v) for k, v in scores.items()},
        needs_review=review,
        min_score=settings.evaluation.min_score,
        models=models,
    )


def export_document(
    document: Document,
    out_dir: Path,
    *,
    formats: list[str],
    context: RenderContext,
    stem: str = "book",
    settings: Settings | None = None,
    split_chapters: bool = False,
    cover: Path | None = None,
) -> ExportResult:
    """Render a document into every requested format.

    A format that fails is reported and the rest still run: losing an EPUB because a PDF
    engine is missing would be absurd.
    """
    from .render.docx import render_docx
    from .render.epub import render_epub, validate_epub
    from .render.html import write_html
    from .render.pdf import render_pdf, render_txt

    out_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult()

    if context.layout == "bilingual-columns":
        unsupported = [f for f in formats if f not in COLUMN_CAPABLE]
        if unsupported:
            result.warnings.append(
                f"bilingual-columns is only available for {', '.join(sorted(COLUMN_CAPABLE))}; "
                f"{', '.join(unsupported)} fell back to paragraph pairs"
            )

    for fmt in formats:
        target = out_dir / f"{stem}.{fmt}"
        try:
            if fmt == "md":
                written = write_markdown(document, target, split_chapters=split_chapters)
                result.files.extend(written)
            elif fmt == "html":
                result.files.append(write_html(document, target, context))
            elif fmt == "txt":
                result.files.append(render_txt(document, target, context))
            elif fmt == "docx":
                result.files.append(render_docx(document, target, context))
            elif fmt == "epub":
                result.files.append(render_epub(document, target, context, cover=cover))
                validation = validate_epub(target, settings)
                result.epub_validation = validation
                for problem in validation.errors[:5]:
                    result.warnings.append(f"epub: {problem.describe()}")
                if validation.ok:
                    log.info("epub_valid", path=str(target), summary=validation.summary())
                for note in validation.notes:
                    result.notes.append(note)
            elif fmt == "pdf":
                engine = settings.export.pdf_engine if settings else "auto"
                result.files.append(
                    render_pdf(document, target, context, engine=engine, settings=settings)
                )
        except RenderError as exc:
            log.warning("format_failed", format=fmt, error=exc.message)
            result.warnings.append(f"{fmt}: {exc.format_for_user()}")

    log.info(
        "export_complete",
        files=len(result.files),
        formats=formats,
        layout=context.layout,
        warnings=len(result.warnings),
    )
    return result


def export_job(
    job_id: str,
    settings: Settings,
    *,
    formats: list[str],
    layout: Layout = "target-only",
    out_dir: Path | None = None,
    split_chapters: bool = False,
    cover: Path | None = None,
) -> ExportResult:
    """Export a job from its stored translated IR.

    Raises:
        RenderError: if the job has not been translated, or its output is not parallel to
            the source -- the §21.2 assertion, enforced before anything is written.
    """
    from .jobs import build_translated_document, reopen_job

    context_job = reopen_job(job_id, settings)
    try:
        source = context_job.document
        translated_path = context_job.paths["translated_ir"]

        if translated_path.is_file():
            translated = Document.load(translated_path)
        else:
            segments = context_job.store.list_segments(job_id)
            translations = {s.segment_id: s.final_text or "" for s in segments if s.final_text}
            if not translations:
                raise RenderError(
                    f"Job {job_id} has nothing translated yet.",
                    remedy=f"Run it first: folioai resume {job_id}",
                    context={"job_id": job_id},
                )
            translated = build_translated_document(source, translations, context_job.target_lang)

        try:
            source.assert_parallel_to(translated)
        except ValueError as exc:
            raise RenderError(
                f"Refusing to export {job_id}: {exc}",
                remedy=(
                    "This means content was lost between extraction and translation, which "
                    "should be impossible. Please report it with the job directory."
                ),
                context={"job_id": job_id},
            ) from exc

        render_context = build_context(
            context_job.store, job_id, layout=layout, source=source, settings=settings
        )
        return export_document(
            translated,
            out_dir or context_job.paths["export"],
            formats=formats,
            context=render_context,
            stem=job_id,
            settings=settings,
            split_chapters=split_chapters,
            cover=cover,
        )
    finally:
        context_job.close()
