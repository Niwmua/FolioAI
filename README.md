# FolioAI

Book-length PDF in, faithfully translated book out.

`folioai` extracts a PDF with its structure intact, translates it with an LLM, has a *second*
LLM judge every segment against the source, re-translates anything that falls short, and
exports to Markdown, EPUB, PDF, DOCX, HTML or plain text.

The property it is built around is **fidelity**. The translation must not drop a sentence,
add a sentence, summarise, "improve" the prose, soften anything, or quietly skip material it
finds awkward. Fluency matters, but never at the cost of completeness — so the architecture
is arranged to make omission and drift *detectable* rather than invisible:

- Blocks travel with ID tags, so a dropped paragraph is a missing tag, caught for free.
- Deterministic checks run before the paid judge, so nothing pays a model to be told a
  response was malformed.
- The rubric weights completeness at 35% and hard-fails below 70 regardless of the total,
  because a weighted average must not be able to hide a lost sentence.
- Block count in equals block count out, asserted in code, not checked by eye.

> You are responsible for holding the rights to translate the material you feed in. The tool
> prints this once on first run and gates nothing on it.

## Status

Under construction, milestone by milestone (see [PLAN.md](PLAN.md) §3).

| Milestone | State |
|---|---|
| 1. Skeleton — config, CLI, logging, SQLite store, errors | **done** |
| 2. Extraction — probe, extractors, cleaning, IR, structure | **done** |
| 3. LLM plumbing — async client, rate limits, pricing, cache, estimate | next |
| 4–8. Translation, evaluation, glossary, export, polish | not started |

Commands that are not built yet say so and name the milestone that will deliver them. Nothing
returns plausible-looking fake data.

## Install

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra render --extra tokens
uv run folioai --help
```

No external binaries are required. `folioai` uses PyMuPDF for everything on the extraction
path, and *reports* which optional tools it found (`pdftotext`, `ocrmypdf`, `tesseract`,
`typst`, `epubcheck`) rather than depending on them. Missing ones degrade a single feature
and say what to install.

## Try it

The test fixtures are synthetic PDFs that encode the ways real PDFs ruin text — running heads,
hyphens broken across lines, drop caps, footnotes, two columns. Build them:

```bash
uv run python tests/fixtures/make_pdfs.py
```

**Diagnose a PDF** — free, reads nothing but the file:

```bash
uv run folioai probe tests/fixtures/pdfs/clean_book.pdf
```

```
clean_book.pdf
  pages          4
  text layer     yes  (899 chars sampled)
  text quality   clean  repl=0.000 ctrl=0.000 space=0.193 vowel=0.405
  language       en (100% confident)  script=latin
  columns        1
  outline        yes (2 entries)
  external tools epubcheck, ocrmypdf, pdftotext, tesseract, typst

  recommended extractor: pymupdf
  Clean single-column text layer.
```

**Extract to Markdown** — also free, no network:

```bash
uv run folioai extract tests/fixtures/pdfs/clean_book.pdf
uv run folioai extract book.pdf -o ir.json -m book.md --audit audit.json
```

The audit file records every line the cleaner removed and every de-hyphenation decision it
made, with the frequency evidence behind each one. A cleaner that silently eats a line is
worse than no cleaner at all.

**See what extraction is up against.** These fixtures each isolate one pathology:

```bash
uv run folioai extract tests/fixtures/pdfs/hyphenated.pdf   # extraordinary joined, well-being kept
uv run folioai extract tests/fixtures/pdfs/furniture.pdf    # running heads and folios gone, body intact
uv run folioai extract tests/fixtures/pdfs/footnotes.pdf    # notes lifted out, [^1] anchors left behind
uv run folioai extract tests/fixtures/pdfs/two_column.pdf   # columns read in order, not interleaved
```

**Job state:**

```bash
uv run folioai jobs list
uv run folioai status
```

## Configuration

Precedence, highest first:

```
CLI flags  >  FOLIOAI_* env vars  >  ./folioai.yaml  >  ~/.folioai/config.yaml  >  packaged defaults
```

Packaged defaults are in [`config/default.yaml`](config/default.yaml), fully commented. Every
key is validated on load, so a typo is an error rather than a setting that silently does
nothing.

API keys come from the environment or a `.env` only — never from a config file, and never
into a log line:

```bash
export FOLIOAI_API_KEY=sk-or-v1-...      # or OPENROUTER_API_KEY / OPENAI_API_KEY
export FOLIOAI_BASE_URL=https://openrouter.ai/api/v1
```

Defaults target OpenRouter, which makes it easy to put the translator and the evaluator on
models from *different* vendors — two instances of one model share their blind spots, which
is the main failure mode of LLM-as-judge. Any OpenAI-compatible endpoint works; set
`llm.base_url` and the `models:` block to match.

Style profiles for `en→de`, `es`, `fr`, `ja`, `zh-Hans`, `ar` and a generic fallback live in
[`config/profiles/`](config/profiles/). They encode the decisions a human translator settles
before page one: register, address form, dialogue punctuation, how idioms and measurements
are handled, whether names are transliterated.

## Development

```bash
uv run pytest              # unit tests; never touch the network
uv run pytest -m live      # opt-in integration tests against a real endpoint
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

The IR's JSON Schema is committed at [`schema/document.schema.json`](schema/document.schema.json)
and a test fails if it drifts out of step with the models, so a change to the document format
shows up in review as a schema diff.

## Design documents

- [PLAN.md](PLAN.md) — interpretation of the brief, and every point where this
  implementation disagrees with it.
- [DECISIONS.md](DECISIONS.md) — each choice the brief left open, with the reasoning.
