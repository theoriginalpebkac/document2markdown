# doc2md

Batch-convert a folder of documents to Markdown with **per-file quality
validation** and a **summary report**.

`doc2md.py` walks an input directory, converts every supported document to
Markdown, checks each conversion for fidelity and structure, and writes both a
machine-readable `conversion_report.json` and a human-readable summary to
stdout. It is built so the conversion and validation helpers can be imported and
unit-tested independently.

---

## How it works

| Stage | What happens |
| --- | --- |
| **Convert** | PDFs go through [pdfmux](https://pypi.org/project/pdfmux/), which routes each page to the best extractor, audits its own output, and re-extracts low-confidence pages. Everything else (DOCX, HTML, EPUB, RTF, ODT) is converted with [pandoc](https://pandoc.org/) to GitHub-Flavored Markdown. Unsupported formats are logged and skipped, never fatal. |
| **Validate** | For PDFs, raw plain text is pulled from the original with `pdftotext` and compared character-for-character against the Markdown (after Markdown syntax is stripped) using `difflib.SequenceMatcher`. A structural check counts headings, fenced code blocks, and pipe tables and flags documents that are long yet have none. |
| **Report** | `conversion_report.json` records every file's similarity score, pass/fail verdict, structural counts, pdfmux confidence, and any errors. A summary with totals and a ranked list of the lowest-scoring files is printed to stdout. |

Files are processed in parallel (configurable worker count, default 4). The exit
code is non-zero if any file errors or fails validation, which makes the tool
usable as a CI gate.

---

## Requirements

### Python package dependencies (`requirements.txt`)

| Dependency | Used for |
| --- | --- |
| **pdfmux[ocr]** (>=1.7.0) | Primary PDF → Markdown converter. Per-page self-healing extraction with confidence scoring; the `[ocr]` extra adds RapidOCR so scanned pages don't come back empty. **Requires Python 3.11+.** |
| **pymupdf** (>=1.24.0) | Used directly by `--preview` to slice the first *N* pages of a PDF into a temporary file so the full pdfmux pipeline runs on a smaller document. (Already a pdfmux dependency; listed because `doc2md.py` imports it.) |

Install with:

```bash
pip install -r requirements.txt
```

### System tools (not pip-installable)

| Tool | Used for | Install |
| --- | --- | --- |
| **pandoc** | Converts DOCX / HTML / EPUB / RTF / ODT to Markdown. | macOS: `brew install pandoc` · Debian/Ubuntu: `sudo apt-get install pandoc` · [other](https://pandoc.org/installing.html) |
| **pdftotext** (from poppler-utils) | Extracts raw plain text from the original PDF for the fidelity comparison. Without it, PDFs still convert but get no similarity score. | macOS: `brew install poppler` · Debian/Ubuntu: `sudo apt-get install poppler-utils` |

`doc2md.py` checks for all three on startup and prints actionable install
instructions for anything missing rather than failing silently. Missing tools
degrade gracefully: the relevant files are recorded as errors and the batch
continues.

> **Python version note:** pdfmux requires Python **3.11+**. `doc2md.py` itself
> is written to parse and run on 3.9+, but the PDF path needs a 3.11+
> interpreter with pdfmux installed.

---

## Usage

```bash
python3 doc2md.py INPUT_DIR OUTPUT_DIR [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `-w`, `--workers N` | `4` | Parallel worker count. |
| `-t`, `--threshold R` | `0.85` | Minimum similarity ratio to pass validation. |
| `-q`, `--quality {fast,standard,high}` | `standard` | pdfmux extraction quality. |
| `--preview` | off | Quick sanity check: process only the first few pages of each PDF. |
| `--preview-pages N` | `3` | Number of pages to use with `--preview`. |

### Examples

```bash
# Convert everything under ./docs into ./out
python3 doc2md.py ./docs ./out

# Fast sanity check on the first 3 pages of each PDF, 8 workers
python3 doc2md.py ./docs ./out --preview --workers 8

# Stricter fidelity bar, high-quality extraction
python3 doc2md.py ./docs ./out --threshold 0.92 --quality high
```

The input tree's structure is mirrored under the output directory, with each
file rewritten to `.md` (e.g. `docs/sub/report.pdf` → `out/sub/report.md`).

---

## A note on `--preview`

pdfmux's public API (`extract_text`, `pipeline.process`, and the `pdfmux
convert` CLI) does **not** expose a page-range option — only an internal
extractor does, and using it would bypass the router/audit pipeline that is the
whole reason for choosing pdfmux. So `--preview` slices the first *N* pages into
a temporary PDF with PyMuPDF and runs the full self-healing pipeline on that
slice. `pdftotext` is given a matching `-l N` so the fidelity comparison stays
like-for-like.

---

## Output

### `conversion_report.json`

```jsonc
{
  "summary": {
    "total_files": 12,
    "converted": 10,
    "skipped": 1,
    "errors": 1,
    "passed": 9,
    "failed_validation": 1
  },
  "files": [
    {
      "filename": "report.pdf",
      "source_path": "docs/report.pdf",
      "converter": "pdfmux",
      "status": "converted",
      "output_path": "out/report.md",
      "similarity": 0.94,
      "pdfmux_confidence": 0.97,
      "structural": { "headings": 8, "code_blocks": 0, "tables": 3 },
      "structural_ok": true,
      "structural_reason": null,
      "passed": true,
      "preview": false,
      "error": null
    }
  ]
}
```

### stdout summary

Totals (converted / passed / failed / skipped / errored), a ranked list of the
lowest-scoring files, any structurally suspicious documents, and per-file
errors.

---

## Project layout

| File | Purpose |
| --- | --- |
| `doc2md.py` | The CLI tool and all conversion/validation helpers. |
| `requirements.txt` | Python dependencies (see above). |
| `README.md` | This file. |
