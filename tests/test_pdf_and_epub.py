"""PDF rendering and EPUB validation, against real files.

The PDF tests are skipped when no engine is installed; the EPUB ones always run, because
the structural checks have no dependencies. Both halves matter: rendering a PDF proves the
Typst template compiles, and corrupting an EPUB proves the validator says so -- a validator
that only ever reports "valid" is worse than none, because it is believed.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pytest
from sample_books import build_book as make_doc

from folioai.config import Settings
from folioai.errors import RenderError
from folioai.render.base import RenderContext
from folioai.render.epub import render_epub
from folioai.render.epubcheck import (
    find_epubcheck,
    run_epubcheck,
    validate_epub,
    validate_structure,
)
from folioai.render.pdf import (
    build_typst_source,
    find_typst,
    font_directories,
    render_pdf,
    typst_available,
    typst_inline,
)

needs_typst = pytest.mark.skipif(not typst_available(), reason="no Typst on this machine")


# -- finding the tools ---------------------------------------------------------------


def test_typst_is_found_in_the_bin_directory(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single binary people download rather than install must not need PATH surgery."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    name = "typst.exe" if Path("x").drive or __import__("os").name == "nt" else "typst"
    (fake_bin / name).write_bytes(b"not really typst")
    monkeypatch.setenv("FOLIOAI_BIN_DIR", str(fake_bin))
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    found = find_typst(settings)
    assert found is not None and found.parent == fake_bin


def test_a_configured_typst_path_wins(tmp_path: Path, settings: Settings) -> None:
    binary = tmp_path / "my-typst"
    binary.write_bytes(b"x")
    settings.export.typst_path = binary
    assert find_typst(settings) == binary


def test_a_configured_path_that_does_not_exist_is_not_silently_ignored(
    tmp_path: Path, settings: Settings
) -> None:
    settings.export.typst_path = tmp_path / "nowhere"
    assert find_typst(settings) is None


def test_the_fonts_directory_is_offered_to_the_renderer(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    monkeypatch.setenv("FOLIOAI_FONTS_DIR", str(fonts))
    assert fonts.resolve() in [d.resolve() for d in font_directories(settings)]


def test_extra_font_directories_from_config_are_included(
    tmp_path: Path, settings: Settings
) -> None:
    extra = tmp_path / "extra-fonts"
    extra.mkdir()
    settings.export.font_paths = [extra]
    assert extra in font_directories(settings)


# -- the Typst source ---------------------------------------------------------------------


def test_inline_markup_becomes_typst_markup_not_literal_asterisks() -> None:
    """Escaping the whole string printed the asterisks in the finished book."""
    out = typst_inline("A *word*, **strong**, `code`, and a ref[^1].")
    assert "_word_" in out  # Typst spells italic with underscores
    assert "*strong*" in out  # ...and bold with asterisks
    assert "`code`" in out
    assert "#super[1]" in out


def test_prose_that_looks_like_syntax_is_escaped() -> None:
    out = typst_inline("A # hash, a $ dollar, an @ at, and a [bracket].")
    for char in ("#", "$", "@", "["):
        assert f"\\{char}" in out


def test_escaping_survives_a_language_model_returning_markup() -> None:
    """The text came out of a PDF and through a model; it must never become syntax."""
    out = typst_inline("#set page(width: 1pt)")
    assert out.startswith("\\#")


@needs_typst
def test_the_generated_source_compiles(tmp_path: Path, settings: Settings) -> None:
    render_pdf(make_doc("de"), tmp_path / "book.pdf", RenderContext(), settings=settings)
    assert (tmp_path / "book.pdf").stat().st_size > 1000


# -- the rendered PDF ------------------------------------------------------------------------


@needs_typst
def test_the_pdf_contains_the_text_and_real_typography(tmp_path: Path, settings: Settings) -> None:
    import pymupdf

    path = render_pdf(make_doc("de"), tmp_path / "book.pdf", RenderContext(), settings=settings)
    document = pymupdf.open(path)
    try:
        pages = [document.load_page(i).get_text() for i in range(document.page_count)]
    finally:
        document.close()

    everything = "\n".join(pages)
    assert "Kapitel Eins" in everything
    assert "Kapitel Zwei" in everything
    # The markup itself must not appear: it should have become italic and bold.
    assert "*kursiver*" not in everything
    assert "**Betonung**" not in everything
    # Running heads and folios.
    assert pages[0].strip().splitlines()[-1].strip() == "1"


@needs_typst
def test_running_heads_name_the_chapter_the_page_belongs_to(
    tmp_path: Path, settings: Settings
) -> None:
    """A chapter-opening page used to carry the *previous* chapter's name."""
    import pymupdf

    path = render_pdf(make_doc("de"), tmp_path / "book.pdf", RenderContext(), settings=settings)
    document = pymupdf.open(path)
    try:
        heads = [
            document.load_page(i).get_text().strip().splitlines()[0]
            for i in range(document.page_count)
        ]
    finally:
        document.close()
    assert heads[0] == "Kapitel Eins"
    assert heads[-1] == "Kapitel Zwei"


@needs_typst
def test_a_persian_book_embeds_a_persian_font(tmp_path: Path, settings: Settings) -> None:
    """Boxes instead of letters would be worse than a failed render (D-51)."""
    import pymupdf

    path = render_pdf(make_doc("fa"), tmp_path / "fa.pdf", RenderContext(), settings=settings)
    document = pymupdf.open(path)
    try:
        fonts = {f[3] for i in range(document.page_count) for f in document.get_page_fonts(i)}
    finally:
        document.close()
    assert fonts, "the PDF embedded no fonts at all"
    assert any("Vazir" in name or "Noto" in name for name in fonts), fonts


@needs_typst
def test_a_missing_font_fails_loudly_and_names_the_font(tmp_path: Path, settings: Settings) -> None:
    """The failure has to be actionable: which font, and where to put it."""
    settings.export.font_paths = []
    document = make_doc("ja")  # no CJK face is installed here
    try:
        render_pdf(document, tmp_path / "ja.pdf", RenderContext(), settings=settings)
    except RenderError as exc:
        assert "Noto Serif CJK" in (exc.remedy or "") + exc.message
        assert "font" in (exc.remedy or "").lower()
    else:
        pytest.skip("a CJK font is installed on this machine, so there is no failure to see")


def test_asking_for_typst_when_it_is_absent_says_how_to_get_it(
    tmp_path: Path, settings: Settings
) -> None:
    settings.export.typst_path = tmp_path / "nowhere"
    with pytest.raises(RenderError) as excinfo:
        render_pdf(
            make_doc(), tmp_path / "x.pdf", RenderContext(), engine="typst", settings=settings
        )
    assert "typst" in (excinfo.value.remedy or "").lower()


def test_rtl_and_language_reach_the_typst_source() -> None:
    source = build_typst_source(make_doc("fa"), RenderContext())
    assert "dir: rtl" in source
    assert 'lang: "fa"' in source


# -- EPUB validation: the happy path -----------------------------------------------------------


@pytest.fixture
def valid_epub(tmp_path: Path) -> Path:
    return render_epub(make_doc("de"), tmp_path / "book.epub", RenderContext())


def test_a_freshly_written_epub_validates(valid_epub: Path) -> None:
    result = validate_structure(valid_epub)
    assert result.ok, [p.describe() for p in result.problems]
    assert not result.errors


def test_validation_reports_which_checker_ran(valid_epub: Path) -> None:
    result = validate_epub(valid_epub)
    assert "structural checks" in result.summary()
    if not result.used_epubcheck:
        assert any("epubcheck" in note for note in result.notes)


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    result = validate_structure(tmp_path / "nothing.epub")
    assert not result.ok
    assert any(p.check == "archive" for p in result.errors)


def test_something_that_is_not_a_zip_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "fake.epub"
    path.write_text("this is not a zip archive", encoding="utf-8")
    result = validate_structure(path)
    assert not result.ok
    assert any("zip" in p.detail for p in result.errors)


# -- EPUB validation: it must actually catch things --------------------------


def rebuild(
    source: Path,
    target: Path,
    *,
    skip: set[str] | None = None,
    replace: dict[str, bytes] | None = None,
    compress_mimetype: bool = False,
    mimetype_last: bool = False,
) -> Path:
    """Rewrite an EPUB with a specific defect introduced."""
    skip = skip or set()
    replace = replace or {}
    with zipfile.ZipFile(source) as original:
        entries = [(i.filename, original.read(i.filename)) for i in original.infolist()]

    if mimetype_last:
        entries = [e for e in entries if e[0] != "mimetype"] + [
            (name, data) for name, data in entries if name == "mimetype"
        ]

    with zipfile.ZipFile(target, "w") as archive:
        for name, data in entries:
            if name in skip:
                continue
            payload = replace.get(name, data)
            if name == "mimetype" and not compress_mimetype:
                archive.writestr(name, payload, compress_type=zipfile.ZIP_STORED)
            else:
                archive.writestr(name, payload, compress_type=zipfile.ZIP_DEFLATED)
    return target


def opf_name(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return next(n for n in archive.namelist() if n.endswith(".opf"))


def test_a_mimetype_that_is_not_first_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    broken = rebuild(valid_epub, tmp_path / "late.epub", mimetype_last=True)
    result = validate_structure(broken)
    assert not result.ok
    assert any(p.check == "mimetype" and "first" in p.detail for p in result.errors)


def test_a_compressed_mimetype_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    broken = rebuild(valid_epub, tmp_path / "zipped.epub", compress_mimetype=True)
    result = validate_structure(broken)
    assert any("uncompressed" in p.detail for p in result.errors)


def test_a_wrong_mimetype_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    broken = rebuild(valid_epub, tmp_path / "wrong.epub", replace={"mimetype": b"text/plain"})
    result = validate_structure(broken)
    assert any(p.check == "mimetype" for p in result.errors)


def test_a_missing_container_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    broken = rebuild(valid_epub, tmp_path / "nocontainer.epub", skip={"META-INF/container.xml"})
    result = validate_structure(broken)
    assert any(p.check == "container" for p in result.errors)


def test_a_manifest_entry_with_no_file_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    """The defect that produces a book with a blank chapter in a reader."""
    with zipfile.ZipFile(valid_epub) as archive:
        chapter = next(n for n in archive.namelist() if Path(n).name.startswith("chap_"))
    broken = rebuild(valid_epub, tmp_path / "missing.epub", skip={chapter})
    result = validate_structure(broken)
    assert not result.ok
    assert any(p.check == "manifest" for p in result.errors)


def test_a_spine_pointing_at_nothing_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    name = opf_name(valid_epub)
    with zipfile.ZipFile(valid_epub) as archive:
        opf = archive.read(name).decode("utf-8")
    tampered = opf.replace("<spine", '<spine toc="ncx"><itemref idref="does-not-exist"/>', 1)
    broken = rebuild(valid_epub, tmp_path / "spine.epub", replace={name: tampered.encode("utf-8")})
    result = validate_structure(broken)
    assert any(p.check == "spine" for p in result.errors)


def test_malformed_xml_in_a_chapter_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(valid_epub) as archive:
        chapter = next(n for n in archive.namelist() if Path(n).name.startswith("chap_"))
    broken = rebuild(
        valid_epub,
        tmp_path / "badxml.epub",
        replace={chapter: b"<html><body><p>unclosed</body></html>"},
    )
    result = validate_structure(broken)
    assert any(p.check == "content" for p in result.errors)


def test_missing_metadata_is_caught(valid_epub: Path, tmp_path: Path) -> None:
    name = opf_name(valid_epub)
    with zipfile.ZipFile(valid_epub) as archive:
        opf = archive.read(name).decode("utf-8")
    import re

    tampered = re.sub(r"<dc:language>.*?</dc:language>", "", opf, flags=re.DOTALL)
    broken = rebuild(valid_epub, tmp_path / "nolang.epub", replace={name: tampered.encode("utf-8")})
    result = validate_structure(broken)
    assert any("dc:language" in p.detail for p in result.errors)


# -- epubcheck integration ------------------------------------------------------------------------


def test_epubcheck_is_optional_and_its_absence_is_reported(
    valid_epub: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    settings.export.epubcheck_path = None
    assert run_epubcheck(valid_epub, settings) is None

    result = validate_epub(valid_epub, settings)
    assert result.ok  # an unvalidated EPUB is not a broken one
    assert any("epubcheck" in note for note in result.notes)


def test_a_configured_jar_is_run_through_java(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """epubcheck ships as a jar; folioai should not require a native launcher."""
    jar = tmp_path / "epubcheck.jar"
    jar.write_bytes(b"not really a jar")
    settings.export.epubcheck_path = jar
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/java" if name == "java" else None)

    command = find_epubcheck(settings)
    assert command is not None
    assert command[0].endswith("java")
    assert command[1] == "-jar"
    assert command[2] == str(jar)


def test_a_jar_without_a_jvm_is_not_offered(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    jar = tmp_path / "epubcheck.jar"
    jar.write_bytes(b"x")
    settings.export.epubcheck_path = jar
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert find_epubcheck(settings) is None


def test_epubcheck_output_is_classified_by_severity(
    valid_epub: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    class Completed:
        returncode = 1
        stdout = b"ERROR(RSC-001): missing thing\nWARNING(OPF-003): odd thing\nInfo: fine\n"
        stderr = b""

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/epubcheck")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Completed())
    settings.export.epubcheck_path = None

    result = run_epubcheck(valid_epub, settings)
    assert result is not None
    assert len(result.errors) == 1
    assert len(result.warnings) == 1
    assert result.used_epubcheck


# -- the export path ------------------------------------------------------------


def test_export_reports_epub_validation(tmp_path: Path, settings: Settings) -> None:
    from folioai.export import export_document

    result = export_document(
        make_doc("de"), tmp_path, formats=["epub"], context=RenderContext(), settings=settings
    )
    assert result.epub_validation is not None
    assert result.epub_validation.ok
    assert not any("epub:" in w for w in result.warnings)


@needs_typst
def test_export_produces_a_pdf(tmp_path: Path, settings: Settings) -> None:
    from folioai.export import export_document

    result = export_document(
        make_doc("de"), tmp_path, formats=["pdf"], context=RenderContext(), settings=settings
    )
    pdfs = [p for p in result.files if p.suffix == ".pdf"]
    assert pdfs and pdfs[0].stat().st_size > 1000
    assert not result.warnings
