# doc2md

> Convert document batches into Markdown trustworthy enough to ground an LLM — verified, not just converted.

**doc2md** turns folders of documents — PDFs, Word and Confluence exports,
configuration XML, and more — into Markdown built to serve as **grounding
sources for AI/LLM applications and RAG pipelines**, where a silently garbled
conversion is worse than none at all. Unlike general-purpose document-to-Markdown
converters that optimize for breadth and one-shot convenience, doc2md is
**fidelity-first**: every file is independently validated — its text
cross-checked against a separate extraction engine, scored, and flagged in a
pass/fail report — and anything Markdown can't faithfully represent, such as
diagrams and complex tables, is extracted to images and referenced inline rather
than dropped. Those references are kept adjacent to their surrounding context so
retrieval chunks stay intact, and the pipeline pairs self-healing,
confidence-scored PDF extraction with content-aware routing per source type —
deliberately trading breadth for the verifiable accuracy that technical and
architecture work demands.

`doc2md.py` walks an input directory, converts every supported document to
Markdown, extracts visual content that has no faithful Markdown representation
to PNG (referenced inline), validates each conversion, and writes both a
machine-readable `conversion_report.json` and a human-readable stdout summary.
Conversion, visual-extraction, and validation helpers are independent functions
so they can be imported and unit-tested on their own.

---

## How it works

| Stage | What happens |
| --- | --- |
| **Detect** | Each file is classified by **content, not extension** — a Confluence "Word" export is really MHTML named `.doc`, and `.xml` may be config or documentation. |
| **Convert** | Routed by detected type (table below). Unsupported formats are logged and skipped. Files are processed in parallel (`--workers`, default 4). |
| **Extract visuals** | PDF: a PyMuPDF pass renders content with no faithful Markdown form to PNG. Word/MHTML: embedded raster images are recovered (UI icons filtered). |
| **Validate** | PDF: `pdftotext` vs. generated Markdown via `difflib.SequenceMatcher` + pdfmux-confidence gate. Word: source text vs. Markdown similarity. All: structural-emptiness check. |
| **Report** | `conversion_report.json` (per-file score, confidence, structural + figure counts, errors) and a ranked stdout summary. Non-zero exit if any file fails — usable as a CI gate. |

### Formats and routing

| Input (detected) | Converter | Notes |
| --- | --- | --- |
| **PDF** | [pdfmux](https://pypi.org/project/pdfmux/) + PyMuPDF figures | self-healing extraction + confidence; the image path |
| **`.docx`** (Word, Google Docs) | [pandoc](https://pandoc.org/) (`--extract-media`) | semantic structure + embedded images |
| **Confluence "Word" export** (MHTML, usually `.doc`) | extract HTML + base64 images (stdlib) → pandoc | images recovered & inlined; UI icons filtered |
| **Single-file HTML** (incl. a Jira/Confluence "Save as Word" page named `.doc`/`.htm`) | de-chrome + inline `data:` figures → pandoc | sniffed by content, not extension; UI icons filtered |
| **Config XML** (syntax-critical) | **verbatim** — fenced ```xml``` + generated index | lossless; exact tags/attributes/values preserved |
| **Config XML → YAML** (opt-in `--yaml`) | structure-preserving YAML (`.yaml`) | lower-token grounding for LLMs; round-trip-verified against the source XML; add `--yaml-index` for RAG-locatable breadcrumbs |
| **Config XML → RAG-optimized YAML** (opt-in `--rag`) | flattened path-qualified variant (`.rag.txt`) | grounding LLMs can trust under naive vector chunking: summary of origins+variables plus one self-contained `path = value` line per leaf, so retrieval stays accurate even when chunks split the file; value-level fidelity |
| **Documentation XML** | **transform** — structured Markdown (headings/lists) | for XML that is really a document |
| EPUB / RTF / ODT | pandoc | |
| legacy binary `.doc` (OLE) | — | not supported; re-save as `.docx` or export PDF |

### Maximum local effort by default; LLM on demand

The default is the strongest **local** pipeline: pdfmux `quality=standard` (the
full agentic audit → re-extract loop) with whatever local backends you install
(OCR, Docling tables). Very large PDFs (over `--large-doc-pages`, default 1000)
automatically drop to `fast` — Docling's per-page table model would otherwise
turn a multi-thousand-page doc into a near-hour run for little fidelity gain.
OCR is skipped on born-digital PDFs (`--ocr auto`), since the text layer is
already authoritative. Output Markdown is also cleaned for LLM consumption by
default, per format — PDF (page-number/date furniture, empty duplicate tables,
line-wrap token splits), MHTML (single-page-app chrome), docx/web (whitespace);
add `--strip-line` for corpus-specific chrome. Every cleanup pass is verified
lossless and falls back to the raw text if not, so cleaning never costs
fidelity; use `--no-clean` for the raw extraction. No document is sent to a
cloud service by default.

For the occasional document local extraction can't handle, opt in to an LLM:

```bash
python3 doc2md.py ./docs ./out --llm claude          # or gemini | openai | ollama
python3 doc2md.py ./docs ./out --llm gemini --llm-budget 0.50
```

`ollama` keeps it fully local. The report flags which documents have
low-confidence pages, so you know which ones to re-run with `--llm`.

### Picking the best source format

Every conversion hop can lose structure or images, so prefer the least-lossy
export of a given document:

- **Confluence / Jira** — export to **Word** for text-centric pages: it keeps
  semantic headings/tables and (in the MHTML `.doc` variant) embeds images,
  which doc2md recovers and inlines. Export to **PDF** when a page's diagrams are
  drawn as native vectors that the Word export would rasterize poorly — the PDF
  path captures those.
- **Google Docs** — download as **`.docx`** (structure + images preserved). Avoid
  the Markdown download: it drops images. (Zipped HTML is a fine alternative.)
- **Configuration XML** (where syntax details must be preserved) — feed the
  **XML directly**; don't pre-convert it to PDF. doc2md keeps syntax-critical
  config verbatim, so exact tags/attributes/values survive (which a PDF
  round-trip would mangle). The Markdown output is suitable for downstream LLM
  tools without the lossy PDF step.
  - For LLM grounding specifically, pass **`--yaml`** to emit the XML as
    structure-preserving YAML (`.yaml`) instead. YAML carries the same data as
    fenced-XML Markdown but **without the closing-tag and angle-bracket
    overhead, so it consumes markedly fewer tokens** — the reason this mode
    exists. Every file is verified to round-trip back to the source XML, so the
    saving costs no fidelity. **XML comments are preserved and positioned** — a
    comment is kept as a `_comment` entry inside the block it sits above (config
    comments often carry the *why*, e.g. a Jira reference, for the logic beneath
    them), so the annotation stays with its block; only processing instructions
    and the DOCTYPE are dropped. Best for config-style XML;
    documentation-style XML (prose with mid-sentence inline tags) reads better as
    Markdown — leave `--yaml` off for those.
  - For **RAG** ingestion of large, deeply-nested config (e.g. NotebookLM), add
    **`--yaml-index`** (implies `--yaml`). RAG systems retrieve by similarity over
    *chunks*, not by reading the whole file, so a deeply-nested block is retrieved
    stripped of its ancestor keys and can't be located. `--yaml-index` prefixes
    every block with a `# path: a > b(@value=…) > c` structural breadcrumb, so
    each chunk carries its own location and the path tokens help match
    natural-language queries. The breadcrumbs are YAML comments (dropped on parse),
    so fidelity is unchanged — the indexed output minus the `# path:` lines is the
    plain `--yaml` output. Long plain scalars (e.g. space-separated IP/CIDR lists)
    are also kept on a single line, since the multi-line plain scalars that
    default line-wrapping produces trip up strict/lightweight YAML parsers.
  - For maximum retrieval accuracy in a RAG pipeline, pass **`--rag`** to emit a
    **RAG-optimized variant of the YAML** (`.rag.txt`) tuned for how LLMs actually
    retrieve config. Breadcrumb comments only anchor a block's top, so when a
    chunk window splits the block the later leaves are re-orphaned and embedders
    down-weight the comment tokens. `--rag` reshapes the same data so every line
    answers for itself: a deterministic **DOCUMENT SUMMARY** (origin hostnames +
    every variable's value — the facts that take whole-document context to answer)
    followed by one fully path-qualified `a > b(value=…) > c = value` line **per
    leaf**. The hierarchy becomes *content* rather than a comment, so each line
    stands alone no matter where a chunk boundary lands and a conditional override
    (e.g. a failover origin) carries its match path inline — which is what makes
    it grounding LLMs can trust under naive vector chunking. It's a retrieval
    *projection*, not the structured source of truth: it reshapes the YAML for
    ingestion rather than round-tripping back to it (use `--yaml` when you need
    the round-trippable form), so fidelity is verified at the value level instead
    — every source leaf must appear in the output. XML only; the `.rag.txt` is
    ingested directly by tools like NotebookLM.

### Provenance metadata for RAG

By default (disable with `--no-rag-metadata`) every Markdown output carries
invisible provenance so a retrieved chunk can be traced back to its source —
without polluting the text or the fidelity score (both layers below are stripped
before the similarity check, so they never affect validation).

- **YAML frontmatter** — a block at the top of every `.md` file, the citation
  anchor for retrieved chunks:

  ```yaml
  ---
  title: "Network Design Spec"
  source_file: "network-design-spec.pdf"
  source_path: "specs/network-design-spec.pdf"
  format: "pdf"
  engine: "pdfmux"
  quality: "standard"
  page_count: 142
  confidence: 0.9123
  converted: "2024-03-02"
  doc2md_version: "0.3.0"
  ---
  ```

  Routing/filtering facts only. `quality`/`page_count`/`confidence` are PDF-only,
  and `confidence` is omitted when absent so typed loaders (e.g. Pydantic) never
  meet an unexpected null. `converted` is the **source file's mtime** (not
  wall-clock now), quoted as a string — so re-converting an unchanged file
  produces no diff (idempotent, avoids needless re-embeds) and strict schema
  validators don't choke on a native date. `source_path` is **relative to the
  input root** by default to avoid leaking machine/folder names into a shared
  corpus; pass `--source-abspath` for an absolute path. String values are
  JSON-encoded, which both escapes them and freezes the YAML type (so a title
  like `NO` or a version like `1.10` can't be coerced to a bool/float).

  **CSV** output gets the same block on **every split part** (`<stem>-partNNN.md`),
  each additionally stamped `part`/`parts` so an atomic file knows its slice.

- **Page-boundary markers** — `<!-- doc2md:page=N -->` HTML comments at each PDF
  page boundary, so a chunk's source page survives even after page numbers are
  stripped from the prose. Emitted wherever real per-page text is available,
  which doc2md obtains from pdfmux's per-page streaming extractor. They are
  suppressed (rather than emitting a misleading lone `page=1`) only on the
  single-blob paths: documents that route to Docling's table extractor at
  `--quality standard`, and the `--quality high` LLM path — both kept on
  pdfmux's higher-fidelity multi-pass output. `--quality fast` forces per-page
  markers everywhere. When markers are suppressed for a multi-page document it
  is reported as a `[auto] page-markers = off` decision (and in
  `conversion_report.json`).

### Figure & table extraction

The goal is to capture, for grounding, anything that **lacks a faithful Markdown
representation** — while avoiding image bloat:

| Content | Handling |
| --- | --- |
| Information-bearing **raster images** (diagrams/screenshots embedded as images) | **PNG**, referenced inline *(default on)* |
| **Complex tables** (merged/spanning cells, ragged rows, or unextractable) | **PNG + best-effort Markdown**, co-located *(default on)* |
| **Simple tables** (incl. multi-line cells) | Markdown only — multi-line cells become `<br>`, no image |
| Text, headings, lists, code | Markdown only |
| **Vector diagrams** (drawn as native shapes) | **opt-in** via `--vector-diagrams` *(see caveat)* |

Figures are placed next to the text of the page they came from **when per-page
text is available** (the streaming extractor path). On the single-blob paths
(Docling tables at `--quality standard`, or `--quality high`) there is no
per-page anchor to interleave into, so the figures are instead appended
**grouped by page** under a `## Figures & tables (by page)` heading at the end of
the file. Provenance is unaffected either way — every block carries its
`[Figure — p.N]` label and page-stamped alt text — and placement returns to
inline automatically wherever per-page text is available.

Complex-table blocks put the image reference **immediately before** the
best-effort Markdown table, with no heading between them, so a structural or
token-bounded RAG chunker keeps the image and table in the same chunk:

```markdown
> **[Table — p.7]** Rendered as image (authoritative); best-effort Markdown follows.
> ![Network Design Spec — page 7, complex table: Field definitions](network-design-spec/figures/network-design-spec-p007-table01.png)

| field | type | notes |
| --- | --- | --- |
| ... | ... | ... |
```

#### Naming (designed for LLM consumption)

All generated names are **slugified** — lowercase, alphanumeric, hyphen-separated
— because Markdown image links can't contain unescaped spaces. Output for a
document `<Doc Name>.pdf` looks like:

```
<output>/<doc-slug>.md
<output>/<doc-slug>/figures/<doc-slug>-p<PPP>-<kind><NN>.png
```

- `<doc-slug>` e.g. `network-design-spec`; the slug prefixes every PNG so files
  stay unique and traceable even if relocated by a RAG pipeline.
- `p<PPP>` = 1-based page number (zero-padded); `<kind>` = `figure` / `table` /
  `diagram`; `<NN>` = per-page index.
- **Alt text** carries the semantics LLMs actually read: original document title,
  page number, kind, and any nearby caption — so a decontextualized chunk still
  identifies the source.

#### Caveat on `--vector-diagrams`

Auto-detecting vector diagrams is **off by default and best-effort**. In PDFs
exported from Google Docs / Confluence, tables, tables-of-contents, colored
letter-badges and highlight bars are all drawn as vector rectangles and are not
reliably distinguishable from real diagrams by geometry — an always-on detector
ends up imaging text and TOC pages. Diagrams embedded as **raster images** are
captured by the default image pass; diagram **text labels** are always captured
by pdfmux. Enable `--vector-diagrams` only for genuinely diagram-centric PDFs,
and review the output.

---

## Requirements

> **Just want to get running?** See **[INSTALL.md](INSTALL.md)** for copy-paste
> setup recipes (macOS with/without Homebrew, Ubuntu, and `uv`). The tables below
> are the reference for *what* each dependency is and *why*.

### Python packages (`requirements.txt`)

| Dependency | Used for |
| --- | --- |
| **pdfmux[ocr,tables]** (==1.7.0) | Primary PDF → Markdown converter (self-healing extraction). The `[ocr]`/`[tables]` extras add local backends (scanned pages, high-fidelity tables) that the router uses for the default "max local effort". |
| **pymupdf** (>=1.24.0) | Figure/table rendering to PNG and `--preview` page slicing. |

```bash
pip install -r requirements.txt
```

### System tools (not pip-installable)

| Tool | Used for | Install |
| --- | --- | --- |
| **pandoc** | Word (`.docx` + Confluence MHTML), single-file HTML, EPUB / RTF / ODT → Markdown | macOS: `brew install pandoc` · Debian: `sudo apt-get install pandoc` |
| **pdftotext** (poppler-utils) | Raw text for the fidelity comparison. Without it PDFs still convert but get no similarity score. | macOS: `brew install poppler` · Debian: `sudo apt-get install poppler-utils` |

`doc2md.py` checks for all of these on startup and prints install instructions
for anything missing rather than failing silently; missing tools degrade
gracefully (affected files are recorded as errors, the batch continues).

---

## Usage

```bash
python3 doc2md.py INPUT_PATH [OUTPUT_DIR] [options]
```

`INPUT_PATH` is a **single document or a directory**. `OUTPUT_DIR` is optional —
when omitted it defaults to a **`markdown/` folder at the documents' path**
(inside the input directory, or next to a single input file). That nested
`markdown/` folder is automatically excluded from scanning, so re-runs don't
reprocess generated output.

| Option | Default | Description |
| --- | --- | --- |
| `-w`, `--workers N` | `4` | Parallel worker count. |
| `-t`, `--threshold R` | `0.90` | Minimum ordered text-similarity ratio to pass validation. Web exports (MHTML and single-file HTML) are capped at a relaxed `0.80` bar, since SPA chrome and per-token code markup make their character-diff inherently noisier than PDF/DOCX. |
| `--min-confidence R` | `0.70` | Minimum pdfmux confidence to pass. A low ordered-similarity score is reported but **not fatal** when either pdfmux confidence clears this bar *or* order-insensitive **content recall** (≥ 0.90 of the source's word tokens present) is high. The ordered char-diff tanks on faithful conversions that reflow content — tabular/multi-column PDFs (where `pdftotext` itself splits words at line-wraps) and HTML→Markdown (Confluence/Apple MHTML) — so recall lets a content-complete conversion pass while genuine loss (low recall too) still fails. |
| `-q`, `--quality {auto,fast,standard,high}` | `auto` | Local extraction quality. `auto` uses `standard` (max local effort) but drops to `fast` for very large PDFs (see `--large-doc-pages`), where Docling's per-page table model isn't worth the time. `fast` = PyMuPDF only. |
| `--large-doc-pages N` | `1000` | With `--quality=auto`, PDFs larger than N pages use `fast` instead of `standard`. |
| `--ocr {auto,on,off}` | `auto` | OCR control for PDF extraction. `auto` skips OCR on born-digital PDFs (much faster, no fidelity loss — the text layer is authoritative) and keeps it for scanned/image PDFs. `on` forces it; `off` disables it. |
| `--timeout SECONDS` | auto | Per-document extraction timeout (sets `PDFMUX_TIMEOUT`). Omitted: auto-scaled by page count (≥300s, ~3s/page) so large docs don't hit pdfmux's 300s default. `0` = no limit. |
| `--llm {gemini,claude,openai,ollama}` | off | Enable LLM fallback for hard documents. |
| `--llm-budget USD` | — | Per-document spend cap when `--llm` is set. |
| `--no-figures` | off | Text-only Markdown (disable all image extraction). |
| `--no-clean` | off (cleaning **on**) | Disable the default Markdown cleanup. Cleanup is per-format and verified lossless (falls back to raw if it can't be proven): **PDF** — safe page-number/date furniture, empty duplicate tables, line-wrap token-split repair (`<a:b.ma x-c>` → `<a:b.max-c>`); **MHTML / single-file HTML** — single-page-app chrome (`data:` icon images, empty `div`/`span` scaffolding); **docx/web (pandoc)** — whitespace normalization + `--strip-line`. Use `--no-clean` for the raw, true-to-source extraction. |
| `--strip-line REGEX` | — | During cleanup (any format), also remove lines that *fully* match REGEX — for document/corpus-specific header/footer chrome (company names, confidentiality notices, running titles) that can't be detected automatically. Repeatable. Still verified lossless. |
| `--vector-diagrams` | off | Best-effort vector-diagram extraction (see caveat). |
| `--figure-dpi N` | `150` | Render DPI for extracted PNGs. |
| `--xml-mode {auto,verbatim,transform,yaml,rag}` | `auto` | XML handling: `verbatim` (lossless fenced, for config), `transform` (structured Markdown, for doc-XML), `yaml` (structure-preserving `.yaml`), `rag` (flattened `.rag.txt` index, see `--rag`), or auto-detect. |
| `--yaml` | off | Shorthand for `--xml-mode=yaml`: emit XML as YAML (`.yaml`) — fewer tokens than Markdown, round-trip-verified to the source. XML inputs only; takes precedence over `--xml-mode`. |
| `--yaml-index` | off | Implies `--yaml`, and prefixes each nested block with a `# path: …` structural breadcrumb so RAG ingesters (e.g. NotebookLM) can locate deeply-nested blocks. Comments don't change the parsed data, so round-trip fidelity is unaffected. |
| `--rag` | off | Convert config XML to a RAG-optimized variant of the YAML (`.rag.txt`) tuned for LLM retrieval: a deterministic summary (origins + variables) plus one fully path-qualified `a > b > c = value` line per leaf, so every line stays accurate even when a RAG chunk splits the file. A retrieval projection rather than the round-trippable source — use `--yaml` for that; fidelity is checked at the value level. XML inputs only. |
| `--no-rag-metadata` | off (metadata **on**) | Disable RAG provenance metadata. By default every Markdown output gets a YAML **frontmatter** block (source file, format, engine, page count, confidence, source mtime, version) as a citation anchor for retrieved chunks — including **every CSV split part**, which also carries `part`/`parts` so each atomic file knows its slice. PDF output additionally gets `<!-- doc2md:page=N -->` **page-boundary markers** wherever the extractor exposes per-page text (omitted when it only returns one combined blob). All are invisible to humans and stripped before the fidelity check, so they never affect the similarity score. Turn off if a downstream tool can't tolerate frontmatter or HTML comments. |
| `--source-abspath` | off | Emit the frontmatter `source_path` as an **absolute** path instead of relative to the input root. Off by default — relative paths avoid leaking machine/folder names into a shared Markdown corpus. |
| `--preview` | off | Process only the first few pages of each PDF (quick sanity check). |
| `--preview-pages N` | `3` | Pages to use with `--preview`. |

### Examples

```bash
# Folder in, output defaults to ./docs/markdown/
python3 doc2md.py ./docs

# Single file in, output defaults to its folder's markdown/
python3 doc2md.py "./docs/Network Design Spec.pdf"

# Explicit output directory
python3 doc2md.py ./docs ./out

# Quick sanity check on the first 3 pages
python3 doc2md.py ./docs --preview

# Config XML → low-token YAML for LLM grounding (verified lossless)
python3 doc2md.py ./configs --yaml

# Same, with structural-path breadcrumbs for RAG retrieval (e.g. NotebookLM)
python3 doc2md.py ./configs --yaml-index

# Diagram-centric corpus, higher-res figures
python3 doc2md.py ./docs --vector-diagrams --figure-dpi 200

# Re-run a problem document through an LLM with a budget cap
python3 doc2md.py "./docs/hard.pdf" --llm claude --llm-budget 0.50
```

The input tree's structure is mirrored under the output directory; each file is
rewritten to a slugified `.md`, with figures in a sibling `<doc-slug>/figures/`
folder.

---

## Output

### `conversion_report.json`

```jsonc
{
  "summary": {
    "total_files": 23, "converted": 21, "skipped": 2, "errors": 0,
    "passed": 20, "failed_validation": 1, "figures_extracted": 47
  },
  "files": [
    {
      "filename": "Network Design Spec.pdf",
      "converter": "pdfmux", "status": "converted",
      "output_path": "docs/markdown/network-design-spec.md",
      "similarity": 0.94, "pdfmux_confidence": 0.97, "min_page_confidence": 0.88,
      "structural": { "headings": 31, "code_blocks": 4, "tables": 12, "images": 10 },
      "figures": { "diagrams": 0, "images": 10, "complex_tables": 0 },
      "similarity_ok": true, "confidence_ok": true, "structural_ok": true,
      "passed": true, "preview": false, "used_llm": false, "error": null
    }
  ]
}
```

### stdout summary

Totals (converted / passed / failed / skipped / errored), figures extracted, a
ranked list of the lowest-scoring files with confidence, and per-file failure
reasons.

---

## Project layout

| File | Purpose |
| --- | --- |
| `doc2md.py` | CLI tool + conversion / visual-extraction / validation helpers. |
| `requirements.txt` | Python dependencies. |
| `SECURITY.md` | How to report vulnerabilities. |
| `README.md` | This file. |
