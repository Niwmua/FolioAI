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

All eight milestones from the brief are built (see [PLAN.md](PLAN.md) §3): extraction,
LLM plumbing, translation, validation and evaluation, the retry ladder, the glossary,
every export format, the quality report, the review loop, OCR, the vision fallback and
chapter subsetting.

Not built, deliberately: the FastAPI review UI, which the brief marks as phase 2.

The whole pipeline has been exercised end to end against a fake model — the two demos below
run it with no API key — but **not yet against a real endpoint or a real book**. That is the
one thing standing between this and a first real translation.

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

Optional extras:

| Extra | Gives you | Cost |
|---|---|---|
| `render` | EPUB, DOCX, and the WeasyPrint PDF fallback | small |
| `tokens` | exact token counts via tiktoken (otherwise estimated) | small |
| `ml` | the `marker` extractor: ML structure recovery for hard PDFs | pulls in torch, downloads weights |

`ml` conflicts with `render` over pillow, so install it on its own:
`uv sync --extra ml`.

### PDF output

PDF uses [Typst](https://typst.app) — one binary, no system dependencies, and real book
typography: running heads that name the current chapter, chapter openers on recto pages,
proper margins and hyphenation for the target language.

```bash
winget install Typst.Typst        # Windows
brew install typst                # macOS
```

Or download the binary and drop it in folioai's bin directory — no installer, no PATH
surgery:

```bash
uv run folioai paths              # shows the bin directory
```

WeasyPrint is the fallback when Typst is absent, and `export.typst_path` points at a binary
anywhere.

**Fonts.** Most machines have no Persian, Arabic or CJK serif face, and a book of empty
boxes is worse than a failed render — so a missing font is an error that names the font and
tells you where to put it. Drop any `.ttf` or `.otf` into the fonts directory (also shown by
`folioai paths`) and it is offered to the renderer automatically:

| Target | Font | Where |
|---|---|---|
| Persian | Vazirmatn | github.com/rastikerdar/vazirmatn |
| Arabic | Noto Naskh Arabic | fonts.google.com/noto |
| CJK | Noto Serif CJK | github.com/notofonts |
| Latin, Cyrillic, Greek | Noto Serif, or your system serif | usually already present |

### EPUB validation

Every EPUB is validated as it is written. Two layers:

**Structural checks, always.** No dependencies. They verify the things that actually break
readers: the `mimetype` entry's position and compression, the container pointing at a real
OPF, `dc:title`/`language`/`identifier`, every manifest href existing in the archive, every
spine reference resolving, a nav document, and every XHTML file parsing as XML.

**epubcheck, when it can be found.** It is the authority for conformance detail, but it is a
Java jar, and needing a JVM before you can export a book is a poor trade — so its absence is
reported rather than silently skipped. Point at it with `export.epubcheck_path`; a `.jar` is
run through `java -jar` automatically.

```
$ folioai export <job_id> --format epub
wrote ...book.epub
epub: valid (structural checks)
epubcheck is not installed, so only the built-in structural checks ran.
```

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

**Project the cost before spending anything** — extraction and segmentation are free, and
this makes no API calls at all:

```bash
uv run folioai estimate book.pdf --to de
```

Run against the tiny `clean_book.pdf` fixture, that prints:

```
                             clean_book.pdf -> de

 phase         model                  calls     tokens in/out             cost
 ─────────────────────────────────────────────────────────────────────────────
 translation   anthropic/claude-so…       2   1,975 / 187-259   $0.0087-0.0098
 evaluation    openai/gpt-4.1             2   1,838 / 230-461   $0.0055-0.0074
 retries       anthropic/claude-so…       1       948 / 11-77   $0.0030-0.0040

 total                                                            $0.017-0.021

     161 words - 6 blocks - 2 calls - 2 chapters - eval 100%, retries 15%
```

A range, not a number: expansion varies by language pair (en→zh contracts by nearly half)
and the retry rate is not knowable until the run is under way, so a single figure would be
false precision.

**Translate:**

```bash
export FOLIOAI_API_KEY=sk-or-v1-...
uv run folioai translate book.pdf --to de --profile en-de
uv run folioai translate book.pdf --to de --dry-run     # plan and cost, no paid calls
uv run folioai resume <job_id>                          # finish an interrupted run
```

`translate` shows the detected chapter structure for confirmation first (anomalously short or
long chapters flagged), because getting chapter boundaries wrong wastes a whole run's budget.
`--yes` skips the prompt.

**See the machinery work without an API key.** Both demos run the real pipeline against a
fake model:

```bash
uv run python examples/fake_translation_demo.py   # full run: retry ladder, escalation, IR out
uv run python examples/sabotage_demo.py           # §21.5: a translator that drops segments
uv run python examples/export_demo.py             # every format, every layout, plus the report
```

The sabotage demo is the one worth reading. A deliberately broken translator drops every
third segment; the run shows those drops caught by deterministic validation *before* the
evaluator is ever called, and every dropped segment recovered on retry:

```
  segments dropped by the saboteur   2
  translate calls                    4
  evaluate calls                     2  (the mangled attempts were never sent to the judge)
  segments with a final translation  6

PASS  every dropped segment was detected without a judge, recovered on retry,
and the output is block-for-block parallel to the source.
```

**Build a glossary** so names stay consistent across 400 pages:

```bash
uv run folioai glossary build <job_id>          # samples the whole book, opens $EDITOR
uv run folioai glossary show <job_id> --audit   # how each term was actually rendered
```

Extraction samples passages from across the *whole* book — a character who arrives on page
200 is exactly the one whose name drifts — then cross-checks candidates against real
frequency counts, so one-off noise never reaches your editor. Mark a term `locked: true` and
it goes into every prompt.

**Export and review:**

```bash
uv run folioai export <job_id> --format epub,md,pdf
uv run folioai export <job_id> --format html --layout annotated
uv run folioai export <job_id> --format pdf  --layout bilingual-paragraph
uv run folioai report <job_id> --open
uv run folioai review <job_id> --max-score 85
```

`--layout annotated` is the artefact worth reading: the translation with per-segment scores
in the margin and every below-threshold segment highlighted. `review` walks the flagged
segments — accept, edit in `$EDITOR`, or send back with an instruction you type. An edit is
stored as a new attempt with `model: human`, so nothing is overwritten and re-export picks it
up.

**Working on a real book:**

```bash
uv run folioai translate book.pdf --to de --chapters 3-7      # test on a subset first
uv run folioai translate book.pdf --to de --vision-fallback   # re-read badly extracted pages
uv run folioai translate scan.pdf --to de --extractor ocr --ocr-lang eng
uv run folioai translate hard.pdf --to de --extractor marker  # needs --extra ml
```

`--chapters` is the one to reach for first: three chapters of a 400-page novel cost a few
cents and tell you almost everything about whether the prompt, the glossary and the
extraction are right.

**Job state:**

```bash
uv run folioai jobs list
uv run folioai status
uv run folioai jobs rm <job_id>
uv run folioai jobs prune          # drop jobs whose source PDF is gone
```

## How a translation is judged

Every batch goes out with its blocks wrapped in ID tags and comes back the same way, so a
dropped paragraph is a missing tag — detected for free, without asking a second model.

What follows is deliberately ordered cheapest-first:

1. **Deterministic checks** (`validate.py`) — free and instant. Segment integrity, empty
   output, refusals and meta-text, degeneration, truncation are `critical` and retry
   immediately *without* calling the evaluator. Length ratio, untranslated passthrough,
   glossary adherence, number retention and markup fidelity are `warning` and become hints
   for the judge.
2. **The judge** (`evaluate.py`) — a second model, on a different vendor by default, scoring
   five dimensions. The composite is computed in Python, never taken from the model. Any
   `critical` issue, or completeness below 70, fails the segment regardless of the total: a
   weighted average must not be able to hide a lost sentence.
3. **The ladder** (`orchestrate.py`) — attempt 1 at 0.2, attempt 2 at 0.3 with the previous
   output and the reviewer's issues attached, attempt 3 on the escalation model at 0.0. After
   that the highest-scoring attempt is kept and flagged `needs_review`. Content is never
   dropped, and the output never has a gap.
4. **The circuit breaker** — if a quarter of a chapter's segments *and* at least eight of
   them fail their first attempt, the run stops. That pattern means the prompt, the model or
   the extraction is broken, and grinding through 300 more pages of it just burns money.

Every attempt, score and issue is stored, so a re-run answers "why is this segment bad" from
the database rather than from a re-run.

## Configuration

Precedence, highest first:

```
CLI flags  >  FOLIOAI_* env vars  >  ./folioai.yaml  >  ~/.folioai/config.yaml  >  packaged defaults
```

Packaged defaults are in [`config/default.yaml`](config/default.yaml), fully commented. Every
key is validated on load, so a typo is an error rather than a setting that silently does
nothing.

### `.env` and where files live

Copy [`config/.env.example`](config/.env.example) to `config/.env` and edit it. That file is
gitignored, so it is also the right place for an API key on a machine you control — keys are
never read from a YAML config (§16), and never reach a log line.

```bash
cp config/.env.example config/.env
```

```ini
FOLIOAI_API_KEY=sk-or-v1-...
FOLIOAI_HOME=D:/folioai            # everything below defaults under this
FOLIOAI_CACHE_DB=D:/fast/cache.db  # ...but any of them can move on its own
FOLIOAI_JOBS_DIR=E:/books/jobs
```

Two `.env` files are read — `./.env` then `config/.env` — and neither ever overwrites a
variable already set in your shell, so `FOLIOAI_HOME=/tmp/x folioai jobs list` behaves the
way you would expect. Every location is overridable:

| Variable | Default | Holds |
|---|---|---|
| `FOLIOAI_HOME` | `~/.folioai` | Everything below, unless overridden |
| `FOLIOAI_JOBS_DIR` | `$HOME/jobs` | One directory per job: IR, database, exports |
| `FOLIOAI_LOGS_DIR` | `$HOME/logs` | One JSONL file per job |
| `FOLIOAI_CACHE_DB` | `$HOME/cache.db` | Prompt cache, shared across jobs |
| `FOLIOAI_STATE_FILE` | `$HOME/state.json` | Machine-level state |
| `FOLIOAI_USER_CONFIG` | `$HOME/config.yaml` | Your own settings |
| `FOLIOAI_CONFIG_DIR` | packaged `config/` | `default.yaml`, `profiles/`, `.env` |

### Models

There are seven model roles. Only `translator` and `evaluator` have shipped defaults; the
other five **inherit the translator** unless you name them, so configuring two models gives
you a system where every call goes to a model your endpoint actually has.

```ini
FOLIOAI_TRANSLATOR_MODEL=google/gemini-3.6-flash
FOLIOAI_EVALUATOR_MODEL=deepseek/deepseek-v4-flash
# FOLIOAI_ESCALATION_MODEL=       # attempt 3 of the retry ladder
# FOLIOAI_SUMMARIZER_MODEL=       # the rolling chapter summary
# FOLIOAI_GLOSSARY_MODEL=         # term extraction
# FOLIOAI_BACK_TRANSLATOR_MODEL=  # --eval-mode back-translation / both
# FOLIOAI_VISION_MODEL=           # --vision-fallback page transcription
```

That inheritance is not a convenience. Naming a vendor's model as the default for a
secondary role means someone who configured two models for their own gateway gets a system
that translates and evaluates correctly and then, on attempt 3 of the retry ladder, calls a
model the endpoint has never heard of — halfway through a book, after the money is spent.

### Checking what is actually in effect

```bash
uv run folioai config           # every model and setting, and which source set it
uv run folioai config --check   # ...and whether your endpoint really has those models
uv run folioai paths            # file locations, and which .env files were read
```

```
 role              model                        from           exists
 ────────────────────────────────────────────────────────────────────
 translator        google/gemini-3.6-flash      .env           yes
 evaluator         deepseek/deepseek-v4-flash   .env           yes
 escalation        google/gemini-3.6-flash      = translator   yes
```

`--check` calls `GET /models` on your endpoint, which is free, and is the cheapest possible
answer to "will this run actually work".

To see where any path resolves to:

```bash
uv run folioai paths
```

Defaults target OpenRouter, which makes it easy to put the translator and the evaluator on
models from *different* vendors — two instances of one model share their blind spots, which
is the main failure mode of LLM-as-judge. Any OpenAI-compatible endpoint works; set
`llm.base_url` and the `models:` block to match.

### Style profiles

Profiles for `en→de`, `es`, `fr`, `ja`, `zh-Hans`, `ar`, **`fa`** and a generic fallback live
in [`config/profiles/`](config/profiles/). They encode the decisions a human translator
settles before page one: register, address form, dialogue punctuation, how idioms and
measurements are handled, whether names are transliterated. The right one is picked from the
language pair automatically; `--profile` overrides it.

### Persian

Persian (`--to fa`) is supported end to end, and needed more than a profile:

- **The zero-width non-joiner is spelling, not spacing.** `می‌رود` without its ZWNJ is
  `میرود`, a misspelling; so is `کتابها` for `کتاب‌ها`. Unicode normalisation preserves
  U+200C and U+200D, having originally stripped them along with the invisible junk.
- **Arabic letterforms are folded to Persian ones** — `ي`→`ی`, `ك`→`ک` — when the source is
  Persian, because Persian typed on Arabic keyboards is full of them and they break every
  kind of matching. Arabic text keeps its own forms, which are correct there.
- **Persian numerals count as numbers.** `۴۷` and `47` compare equal, so a faithful
  translation is not reported as having dropped every figure in the book.
- **`؟` and `۔` end sentences**, so the sentence-drift statistic means something.
- **RTL throughout**: `dir="rtl"` in HTML, `page-progression-direction="rtl"` in EPUB,
  `dir: rtl` in Typst, and its own font (Vazirmatn) rather than an Arabic naskh face.

```bash
uv run folioai translate book.pdf --to fa --profile en-fa
```

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

## What is not done

- **No real book has been through this.** Every test runs against synthetic PDFs built by
  `tests/fixtures/make_pdfs.py` and a fake model. Real PDFs are stranger than synthetic ones.
- **No real endpoint has been called.** The client, rate limiter, cost accounting and retry
  classification are unit-tested against a stub transport, not against OpenRouter.
- **The FastAPI review UI (§17) is not built.** The brief marks it phase 2, after everything
  else works.
- **epubcheck itself has not run here** — no JVM on this machine — so the epubcheck
  integration is tested against a stubbed process, while the structural checks are tested
  against genuinely corrupted EPUBs.

## Design documents

- [PLAN.md](PLAN.md) — interpretation of the brief, and every point where this
  implementation disagrees with it.
- [DECISIONS.md](DECISIONS.md) — each choice the brief left open, with the reasoning.
