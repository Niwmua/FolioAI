"""EPUB validation (brief §14).

Two layers, because the tool the brief names is not always installable:

1. **Built-in structural checks**, always run, no dependencies. They verify the things that
   actually break readers: the ``mimetype`` entry's position and compression, the container
   pointing at a real OPF, every manifest href existing in the archive, every spine
   reference resolving, a nav document, and every XHTML file parsing as XML.

2. **epubcheck**, when it can be found. It is the authority, and it catches conformance
   details a hand-written checker never will -- but it is a Java jar, and requiring a JVM
   before anyone can export a book would be a poor trade. When it is absent that fact is
   reported, never silently skipped.

An unvalidated EPUB is not a broken one, so a missing epubcheck is a note rather than a
failure. A *failing* structural check is a real failure and is reported as one.
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..logging_setup import get_logger

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

Severity = Literal["error", "warning", "info"]

MIMETYPE = "application/epub+zip"
CONTAINER = "META-INF/container.xml"

_NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}

EPUBCHECK_HINT = (
    "epubcheck is optional. To use it, install a JRE and get epubcheck from "
    "https://github.com/w3c/epubcheck/releases, then either put it on PATH or set "
    "export.epubcheck_path to the jar (folioai runs it with 'java -jar')."
)


@dataclass(slots=True)
class Problem:
    """One thing wrong with an EPUB."""

    severity: Severity
    check: str
    detail: str

    def describe(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"


@dataclass(slots=True)
class ValidationResult:
    """What validation found, and who did the finding."""

    problems: list[Problem] = field(default_factory=list)
    used_epubcheck: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(p.severity == "error" for p in self.problems)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    def add(self, severity: Severity, check: str, detail: str) -> None:
        self.problems.append(Problem(severity, check, detail))

    def summary(self) -> str:
        checker = "epubcheck + structural checks" if self.used_epubcheck else "structural checks"
        if self.ok and not self.warnings:
            return f"valid ({checker})"
        return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s) ({checker})"


# -- the built-in structural checks --------------------------------------------------


def _check_mimetype(archive: zipfile.ZipFile, result: ValidationResult) -> None:
    """``mimetype`` must be first and stored uncompressed.

    This is the one rule a reader checks before anything else, and the one a naive zip
    writer breaks. ebooklib gets it right; a hand-assembled archive often does not.
    """
    names = archive.namelist()
    if not names:
        result.add("error", "mimetype", "the archive is empty")
        return
    if names[0] != "mimetype":
        result.add(
            "error", "mimetype", f"must be the first entry in the archive, found {names[0]!r}"
        )
        return

    info = archive.getinfo("mimetype")
    if info.compress_type != zipfile.ZIP_STORED:
        result.add("error", "mimetype", "must be stored uncompressed")
    content = archive.read("mimetype").decode("ascii", errors="replace").strip()
    if content != MIMETYPE:
        result.add("error", "mimetype", f"must contain {MIMETYPE!r}, found {content!r}")


def _opf_path(archive: zipfile.ZipFile, result: ValidationResult) -> str | None:
    if CONTAINER not in archive.namelist():
        result.add("error", "container", f"{CONTAINER} is missing")
        return None
    try:
        root = ElementTree.fromstring(archive.read(CONTAINER))
    except ElementTree.ParseError as exc:
        result.add("error", "container", f"{CONTAINER} is not valid XML: {exc}")
        return None

    rootfile = root.find(".//container:rootfile", _NS)
    if rootfile is None:
        rootfile = root.find(".//{*}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        result.add("error", "container", "no rootfile with a full-path attribute")
        return None

    path = str(rootfile.get("full-path"))
    if path not in archive.namelist():
        result.add(
            "error", "container", f"rootfile points at {path!r}, which is not in the archive"
        )
        return None
    return path


def _check_package(archive: zipfile.ZipFile, opf_path: str, result: ValidationResult) -> None:
    try:
        package = ElementTree.fromstring(archive.read(opf_path))
    except ElementTree.ParseError as exc:
        result.add("error", "package", f"{opf_path} is not valid XML: {exc}")
        return

    base = str(Path(opf_path).parent).replace("\\", "/")
    base = "" if base == "." else base + "/"
    names = set(archive.namelist())

    # -- metadata the readers actually use
    metadata = package.find("{*}metadata")
    if metadata is None:
        result.add("error", "metadata", "the package has no <metadata> element")
    else:
        for field_name in ("title", "language", "identifier"):
            if metadata.find(f"{{{_NS['dc']}}}{field_name}") is None:
                result.add("error", "metadata", f"dc:{field_name} is missing")

        unique_id = package.get("unique-identifier")
        if unique_id:
            identifiers = {
                element.get("id") for element in metadata.findall(f"{{{_NS['dc']}}}identifier")
            }
            if unique_id not in identifiers:
                result.add(
                    "error",
                    "metadata",
                    f"unique-identifier {unique_id!r} matches no dc:identifier id",
                )
        else:
            result.add("warning", "metadata", "the package has no unique-identifier attribute")

    # -- manifest: every declared file must exist
    manifest = package.find("{*}manifest")
    items: dict[str, str] = {}
    nav_found = False
    if manifest is None:
        result.add("error", "manifest", "the package has no <manifest> element")
    else:
        for item in manifest.findall("{*}item"):
            item_id = item.get("id") or ""
            href = item.get("href") or ""
            items[item_id] = href
            if "nav" in (item.get("properties") or "").split():
                nav_found = True
            if href.startswith(("http://", "https://", "/")) or ".." in href:
                result.add("warning", "manifest", f"href {href!r} is not a plain relative path")
                continue
            target = f"{base}{href}".replace("//", "/")
            if target not in names:
                result.add("error", "manifest", f"declares {href!r}, which is not in the archive")
        if not nav_found:
            result.add(
                "warning",
                "manifest",
                'no item with properties="nav": EPUB 3 readers expect a nav document',
            )

    # -- spine: the reading order must resolve
    spine = package.find("{*}spine")
    if spine is None:
        result.add("error", "spine", "the package has no <spine> element")
    else:
        refs = spine.findall("{*}itemref")
        if not refs:
            result.add("error", "spine", "the spine is empty, so the book has no reading order")
        for ref in refs:
            idref = ref.get("idref") or ""
            if idref not in items:
                result.add("error", "spine", f"itemref {idref!r} is not in the manifest")

    # -- every content document must parse
    for item_id, href in items.items():
        if not href.endswith((".xhtml", ".html", ".xml", ".ncx")):
            continue
        target = f"{base}{href}".replace("//", "/")
        if target not in names:
            continue
        try:
            ElementTree.fromstring(archive.read(target))
        except ElementTree.ParseError as exc:
            result.add("error", "content", f"{href} ({item_id}) is not well-formed XML: {exc}")


def validate_structure(path: Path) -> ValidationResult:
    """Validate an EPUB with the built-in checks. No dependencies, always available."""
    result = ValidationResult()
    if not path.is_file():
        result.add("error", "archive", f"{path} does not exist")
        return result

    try:
        with zipfile.ZipFile(path) as archive:
            broken = archive.testzip()
            if broken is not None:
                result.add("error", "archive", f"corrupt entry: {broken}")
                return result
            _check_mimetype(archive, result)
            opf_path = _opf_path(archive, result)
            if opf_path is not None:
                _check_package(archive, opf_path, result)
    except zipfile.BadZipFile as exc:
        result.add("error", "archive", f"not a valid zip archive: {exc}")

    return result


# -- epubcheck, when it is available -----------------------------------------------------


def find_epubcheck(settings: Settings | None = None) -> list[str] | None:
    """The command that runs epubcheck, or None.

    Handles both shapes it comes in: a native launcher on PATH, and the jar, which needs a
    JVM. Returning the argv prefix rather than a path keeps that difference in one place.
    """
    from ..paths import bin_dir

    def as_command(candidate: Path) -> list[str] | None:
        if candidate.suffix == ".jar":
            java = shutil.which("java")
            if java is None:
                return None
            return [java, "-jar", str(candidate)]
        return [str(candidate)]

    if settings is not None and settings.export.epubcheck_path:
        configured = Path(settings.export.epubcheck_path).expanduser()
        return as_command(configured) if configured.is_file() else None

    found = shutil.which("epubcheck")
    if found:
        return [found]

    for name in ("epubcheck.jar", "epubcheck.exe", "epubcheck"):
        candidate = bin_dir() / name
        if candidate.is_file():
            command = as_command(candidate)
            if command is not None:
                return command
    return None


def run_epubcheck(path: Path, settings: Settings | None = None) -> ValidationResult | None:
    """Run epubcheck. Returns None when it is not available."""
    command = find_epubcheck(settings)
    if command is None:
        return None

    result = ValidationResult(used_epubcheck=True)
    try:
        completed = subprocess.run(  # argv list, never a shell string
            [*command, str(path)], capture_output=True, timeout=300, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.add("warning", "epubcheck", f"could not be run: {exc}")
        return result

    output = (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ERROR", "FATAL")):
            result.add("error", "epubcheck", stripped[:300])
        elif stripped.startswith("WARNING"):
            result.add("warning", "epubcheck", stripped[:300])
    if completed.returncode != 0 and not result.problems:
        result.add("error", "epubcheck", f"exited {completed.returncode}")
    return result


def validate_epub(path: Path, settings: Settings | None = None) -> ValidationResult:
    """Validate an EPUB: structural checks always, epubcheck when it can be found."""
    result = validate_structure(path)

    external = run_epubcheck(path, settings)
    if external is None:
        result.notes.append(
            "epubcheck is not installed, so only the built-in structural checks ran. "
            + EPUBCHECK_HINT
        )
    else:
        result.used_epubcheck = True
        result.problems.extend(external.problems)

    log.info(
        "epub_validated",
        path=str(path),
        ok=result.ok,
        errors=len(result.errors),
        warnings=len(result.warnings),
        epubcheck=result.used_epubcheck,
    )
    return result
