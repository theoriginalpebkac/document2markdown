#!/usr/bin/env python3
"""doc2md.py — batch-convert a folder of documents to Markdown with per-file
quality validation and a summary report.

Conversion
    * PDF              -> pdfmux  (per-page self-healing extraction + confidence)
    * DOCX/HTML/EPUB/RTF -> pandoc

Validation (PDF only, for the fidelity part)
    Raw plain text is extracted from the original PDF with ``pdftotext`` and
    compared character-for-character against the Markdown output (after the
    Markdown syntax is stripped) using ``difflib.SequenceMatcher``. A structural
    check counts headings / fenced code blocks / pipe tables and flags documents
    where those are suspiciously absent given the document length.

Report
    ``conversion_report.json`` is written to the output directory and a
    human-readable summary is printed to stdout.

The conversion and validation helpers are deliberately small, side-effect-light,
and independent so they can be imported and unit-tested on their own.

NOTE ON --preview: pdfmux's public API (``extract_text`` / ``pipeline.process``
and the ``pdfmux convert`` CLI) does *not* expose a page-range option — only an
internal extractor does, and using it would bypass the router/audit pipeline
that is the whole reason for choosing pdfmux. So ``--preview`` slices the first
N pages into a temporary PDF with PyMuPDF (already a pdfmux dependency) and runs
the full self-healing pipeline on that slice. Same intent, public API only.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Configuration / defaults
# --------------------------------------------------------------------------- #

DEFAULT_WORKERS = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_PREVIEW_PAGES = 3

# A document whose stripped plain-text is at least this many characters but has
# zero headings, code blocks, or tables is considered structurally suspicious.
STRUCT_MIN_CHARS = 3000

PANDOC_EXTENSIONS = {".docx", ".html", ".htm", ".epub", ".rtf", ".odt"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | PANDOC_EXTENSIONS


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class FileRecord:
    """Everything we know about one input file after processing."""

    filename: str
    source_path: str
    converter: Optional[str] = None  # "pdfmux" | "pandoc" | None
    status: str = "pending"  # "converted" | "skipped" | "error"
    output_path: Optional[str] = None
    similarity: Optional[float] = None
    pdfmux_confidence: Optional[float] = None
    structural: Dict[str, int] = field(default_factory=dict)
    structural_ok: Optional[bool] = None
    structural_reason: Optional[str] = None
    passed: Optional[bool] = None
    preview: bool = False
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Dependency checks
# --------------------------------------------------------------------------- #


def check_dependencies() -> Dict[str, bool]:
    """Probe for the external tools / package we rely on.

    Returns a mapping of dependency name -> availability. Prints actionable
    install instructions for anything missing rather than failing silently.
    """
    import importlib.util

    available = {
        "pdfmux": importlib.util.find_spec("pdfmux") is not None,
        "pandoc": shutil.which("pandoc") is not None,
        "pdftotext": shutil.which("pdftotext") is not None,
    }

    instructions = {
        "pdfmux": (
            "pdfmux (PDF -> Markdown converter) is not importable.\n"
            "    Install it with:  pip install -r requirements.txt\n"
            "    or directly:      pip install 'pdfmux[ocr]'\n"
            "    (pdfmux requires Python 3.11+.)"
        ),
        "pandoc": (
            "pandoc (DOCX/HTML/EPUB/RTF -> Markdown) was not found on PATH.\n"
            "    macOS:           brew install pandoc\n"
            "    Debian/Ubuntu:   sudo apt-get install pandoc\n"
            "    Other:           https://pandoc.org/installing.html"
        ),
        "pdftotext": (
            "pdftotext (from poppler-utils, used for fidelity validation) was "
            "not found on PATH.\n"
            "    macOS:           brew install poppler\n"
            "    Debian/Ubuntu:   sudo apt-get install poppler-utils\n"
            "    Without it, PDF similarity scores cannot be computed."
        ),
    }

    for name, ok in available.items():
        if not ok:
            print("[warning] " + instructions[name], file=sys.stderr)

    return available


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def _slice_pdf_pages(src: Path, n_pages: int, dest_dir: Path) -> Path:
    """Write the first ``n_pages`` of ``src`` to a temp PDF and return its path.

    Used by --preview so the full pdfmux pipeline still runs (just on a smaller
    document). Relies on PyMuPDF, which pdfmux already depends on.
    """
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 exposes the `pymupdf` name
    except ImportError:  # pragma: no cover - older PyMuPDF
        import fitz  # type: ignore

    doc = fitz.open(str(src))
    try:
        last = min(n_pages, doc.page_count) - 1
        out = fitz.open()
        try:
            out.insert_pdf(doc, from_page=0, to_page=last)
            dest = dest_dir / src.name
            out.save(str(dest))
        finally:
            out.close()
    finally:
        doc.close()
    return dest


def convert_pdf(
    src: Path,
    dest: Path,
    quality: str = "standard",
    preview_pages: Optional[int] = None,
) -> Tuple[str, Optional[float]]:
    """Convert a PDF to Markdown with pdfmux and write it to ``dest``.

    Returns ``(markdown_text, pdfmux_confidence)``. ``confidence`` is ``None`` if
    the running pdfmux version doesn't surface it. Raises on extraction failure;
    callers are expected to catch and record the error.
    """
    work_src = src
    tmpdir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if preview_pages is not None:
            tmpdir = tempfile.TemporaryDirectory(prefix="doc2md_preview_")
            work_src = _slice_pdf_pages(src, preview_pages, Path(tmpdir.name))

        markdown, confidence = _extract_pdf_markdown(work_src, quality=quality)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    return markdown, confidence


def _extract_pdf_markdown(
    src: Path, quality: str = "standard"
) -> Tuple[str, Optional[float]]:
    """Run pdfmux and return ``(markdown, confidence)``.

    Prefers ``pdfmux.pipeline.process`` (which also yields a confidence score);
    falls back to the stable public ``pdfmux.extract_text`` if the internal
    module layout ever changes. Extraction errors propagate to the caller.
    """
    try:
        from pdfmux.pipeline import process  # documented surface under batch_extract
    except ImportError:
        process = None  # type: ignore

    if process is not None:
        result = process(file_path=str(src), output_format="markdown", quality=quality)
        return result.text, getattr(result, "confidence", None)

    import pdfmux

    return pdfmux.extract_text(str(src), quality=quality), None


def convert_with_pandoc(src: Path, dest: Path) -> str:
    """Convert a non-PDF document to GFM Markdown with pandoc.

    Returns the Markdown text. Raises ``RuntimeError`` with pandoc's stderr on
    failure so the caller can record a useful message.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["pandoc", "--to=gfm", "--output", str(dest), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "pandoc failed (exit %d): %s" % (proc.returncode, proc.stderr.strip())
        )
    return dest.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def extract_pdf_plaintext(src: Path, last_page: Optional[int] = None) -> str:
    """Extract raw plain text from a PDF with ``pdftotext``.

    ``last_page`` limits extraction to the first N pages (used for --preview so
    the comparison is like-for-like). Raises ``RuntimeError`` on failure.
    """
    cmd = ["pdftotext", "-q"]
    if last_page is not None:
        cmd += ["-l", str(last_page)]
    cmd += [str(src), "-"]  # "-" => write to stdout
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "pdftotext failed (exit %d): %s" % (proc.returncode, proc.stderr.strip())
        )
    return proc.stdout


# Pattern for a GFM table delimiter row, e.g. "| --- | :--: |". One per table.
_TABLE_DELIM_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$", re.MULTILINE
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*(```|~~~)", re.MULTILINE)


def strip_markdown(md: str) -> str:
    """Reduce Markdown to its plain-text content for fidelity comparison.

    Removes fences/headings/emphasis/list markers/table pipes/links/images/HTML
    while keeping the underlying words. Intentionally lightweight — exactness is
    not the goal; comparability with ``pdftotext`` output is.
    """
    text = md
    text = re.sub(r"^\s*(```|~~~).*$", "", text, flags=re.MULTILINE)  # fence lines
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> link text
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # heading marks
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)  # blockquotes
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # ordered lists
    text = _TABLE_DELIM_RE.sub("", text)  # table delimiter rows
    text = text.replace("|", " ")  # remaining table pipes
    text = re.sub(r"(\*\*|__|\*|_|~~)", "", text)  # emphasis markers
    text = re.sub(r"<[^>]+>", "", text)  # stray HTML tags
    return text


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def similarity_ratio(original_text: str, markdown_text: str) -> float:
    """Character-level similarity between source plain text and stripped Markdown.

    Both sides are whitespace-normalized and lowercased so that layout/casing
    noise doesn't dominate, then compared with ``difflib.SequenceMatcher``.
    Returns a ratio in [0.0, 1.0]; 1.0 for two empty strings.
    """
    a = _normalize_whitespace(original_text)
    b = _normalize_whitespace(strip_markdown(markdown_text))
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def structural_counts(md: str) -> Dict[str, int]:
    """Count headings, fenced code blocks, and pipe tables in Markdown."""
    return {
        "headings": len(_HEADING_RE.findall(md)),
        "code_blocks": len(_FENCE_RE.findall(md)) // 2,
        "tables": len(_TABLE_DELIM_RE.findall(md)),
    }


def structural_check(
    counts: Dict[str, int], plain_text: str, min_chars: int = STRUCT_MIN_CHARS
) -> Tuple[bool, Optional[str]]:
    """Flag documents that are long yet have no structural elements at all.

    Returns ``(ok, reason)``. ``ok`` is False when the document is at least
    ``min_chars`` long but contains zero headings, code blocks, and tables —
    a strong signal that structure was lost during conversion.
    """
    total = counts.get("headings", 0) + counts.get("code_blocks", 0) + counts.get("tables", 0)
    if len(plain_text) >= min_chars and total == 0:
        return (
            False,
            "document has %d chars but no headings, code blocks, or tables"
            % len(plain_text),
        )
    return True, None


def validate(
    markdown: str,
    original_plaintext: Optional[str],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, object]:
    """Run the fidelity + structural checks and decide pass/fail.

    ``original_plaintext`` is ``None`` for non-PDF inputs (no pdftotext source),
    in which case the similarity check is skipped and the verdict rests on the
    structural check alone.
    """
    counts = structural_counts(markdown)
    plain = strip_markdown(markdown)
    struct_ok, struct_reason = structural_check(counts, plain)

    similarity: Optional[float] = None
    sim_ok = True
    if original_plaintext is not None:
        similarity = similarity_ratio(original_plaintext, markdown)
        sim_ok = similarity >= similarity_threshold

    return {
        "similarity": similarity,
        "structural": counts,
        "structural_ok": struct_ok,
        "structural_reason": struct_reason,
        "passed": bool(sim_ok and struct_ok),
    }


# --------------------------------------------------------------------------- #
# Per-file orchestration
# --------------------------------------------------------------------------- #


def _output_path_for(src: Path, input_dir: Path, output_dir: Path) -> Path:
    """Mirror the input file's relative location under the output dir as .md."""
    rel = src.relative_to(input_dir)
    return (output_dir / rel).with_suffix(".md")


def process_file(
    src: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    deps: Dict[str, bool],
    quality: str = "standard",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    preview_pages: Optional[int] = None,
) -> FileRecord:
    """Convert + validate a single file. Never raises — errors go in the record."""
    record = FileRecord(
        filename=src.name,
        source_path=str(src),
        preview=preview_pages is not None,
    )
    ext = src.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        record.status = "skipped"
        record.error = "unsupported format: %s" % (ext or "<none>")
        return record

    dest = _output_path_for(src, input_dir, output_dir)

    try:
        if ext in PDF_EXTENSIONS:
            if not deps.get("pdfmux"):
                raise RuntimeError("pdfmux is not installed; cannot convert PDFs")
            record.converter = "pdfmux"
            markdown, confidence = convert_pdf(
                src, dest, quality=quality, preview_pages=preview_pages
            )
            record.pdfmux_confidence = confidence

            original_plaintext: Optional[str] = None
            if deps.get("pdftotext"):
                original_plaintext = extract_pdf_plaintext(
                    src, last_page=preview_pages
                )
        else:
            if not deps.get("pandoc"):
                raise RuntimeError("pandoc is not installed; cannot convert %s" % ext)
            record.converter = "pandoc"
            markdown = convert_with_pandoc(src, dest)
            original_plaintext = None  # pdftotext is PDF-only; structural check only

        record.output_path = str(dest)
        result = validate(markdown, original_plaintext, similarity_threshold)
        record.similarity = result["similarity"]  # type: ignore[assignment]
        record.structural = result["structural"]  # type: ignore[assignment]
        record.structural_ok = result["structural_ok"]  # type: ignore[assignment]
        record.structural_reason = result["structural_reason"]  # type: ignore[assignment]
        record.passed = result["passed"]  # type: ignore[assignment]
        record.status = "converted"
    except Exception as exc:  # noqa: BLE001 - record any failure, keep batch going
        record.status = "error"
        record.error = str(exc)

    return record


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def write_report(records: List[FileRecord], output_dir: Path) -> Path:
    """Write conversion_report.json to the output directory; return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "conversion_report.json"

    converted = [r for r in records if r.status == "converted"]
    passed = [r for r in converted if r.passed]
    failed = [r for r in converted if r.passed is False]

    payload = {
        "summary": {
            "total_files": len(records),
            "converted": len(converted),
            "skipped": sum(1 for r in records if r.status == "skipped"),
            "errors": sum(1 for r in records if r.status == "error"),
            "passed": len(passed),
            "failed_validation": len(failed),
        },
        "files": [asdict(r) for r in records],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def print_summary(records: List[FileRecord], top_n: int = 10) -> None:
    """Print a human-readable summary with a ranked list of lowest scorers."""
    converted = [r for r in records if r.status == "converted"]
    passed = [r for r in converted if r.passed]
    failed = [r for r in converted if r.passed is False]
    skipped = [r for r in records if r.status == "skipped"]
    errored = [r for r in records if r.status == "error"]

    print("\n" + "=" * 60)
    print("doc2md conversion summary")
    print("=" * 60)
    print("Total files seen     : %d" % len(records))
    print("Converted            : %d" % len(converted))
    print("  Passed validation  : %d" % len(passed))
    print("  Failed validation  : %d" % len(failed))
    print("Skipped (unsupported): %d" % len(skipped))
    print("Errored              : %d" % len(errored))

    scored = [r for r in converted if r.similarity is not None]
    if scored:
        scored.sort(key=lambda r: r.similarity)  # type: ignore[arg-type, return-value]
        print("\nLowest-scoring files (similarity):")
        for r in scored[:top_n]:
            flag = "FAIL" if r.passed is False else "ok"
            print("  %-6s %.3f  %s" % (flag, r.similarity, r.filename))

    flagged = [r for r in failed if r.structural_ok is False]
    if flagged:
        print("\nStructurally suspicious (passed/failed shown above):")
        for r in flagged:
            print("  %s — %s" % (r.filename, r.structural_reason))

    if errored:
        print("\nErrors:")
        for r in errored:
            print("  %s — %s" % (r.filename, r.error))

    if skipped:
        print("\nSkipped:")
        for r in skipped:
            print("  %s — %s" % (r.filename, r.error))
    print("")


# --------------------------------------------------------------------------- #
# Batch driver + CLI
# --------------------------------------------------------------------------- #


def discover_files(input_dir: Path) -> List[Path]:
    """Return all regular files under ``input_dir`` (recursive), sorted."""
    return sorted(p for p in input_dir.rglob("*") if p.is_file())


def run_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    quality: str = "standard",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    preview_pages: Optional[int] = None,
    deps: Optional[Dict[str, bool]] = None,
) -> List[FileRecord]:
    """Convert + validate every file under ``input_dir`` in parallel."""
    if deps is None:
        deps = check_dependencies()

    files = discover_files(input_dir)
    # Skip our own report if it lands inside the scanned tree on a re-run.
    files = [f for f in files if f.name != "conversion_report.json"]

    records: List[FileRecord] = []
    if not files:
        return records

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                process_file,
                f,
                input_dir,
                output_dir,
                deps=deps,
                quality=quality,
                similarity_threshold=similarity_threshold,
                preview_pages=preview_pages,
            ): f
            for f in files
        }
        for fut in as_completed(futures):
            records.append(fut.result())

    # Stable, predictable ordering for the report.
    records.sort(key=lambda r: r.source_path)
    return records


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="doc2md.py",
        description="Batch-convert a folder of documents to Markdown with "
        "per-file quality validation and a summary report.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory of source documents")
    parser.add_argument("output_dir", type=Path, help="Where to write .md + report")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel worker count (default: %d)" % DEFAULT_WORKERS,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Similarity threshold for passing validation "
        "(default: %.2f)" % DEFAULT_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=("fast", "standard", "high"),
        default="standard",
        help="pdfmux extraction quality (default: standard)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Quick sanity check: process only the first %d pages of each PDF "
        "(via a temporary page slice)." % DEFAULT_PREVIEW_PAGES,
    )
    parser.add_argument(
        "--preview-pages",
        type=int,
        default=DEFAULT_PREVIEW_PAGES,
        help="Number of pages to use with --preview (default: %d)"
        % DEFAULT_PREVIEW_PAGES,
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.input_dir.is_dir():
        print("error: input directory not found: %s" % args.input_dir, file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    deps = check_dependencies()
    if not any(deps.values()):
        print(
            "error: none of pdfmux / pandoc / pdftotext are available; "
            "nothing can be done. See the messages above.",
            file=sys.stderr,
        )
        return 1

    preview_pages = args.preview_pages if args.preview else None

    records = run_batch(
        args.input_dir,
        args.output_dir,
        workers=args.workers,
        quality=args.quality,
        similarity_threshold=args.threshold,
        preview_pages=preview_pages,
        deps=deps,
    )

    report_path = write_report(records, args.output_dir)
    print_summary(records)
    print("Report written to: %s" % report_path)

    # Non-zero exit if anything failed validation or errored, so this is
    # CI-friendly.
    bad = sum(1 for r in records if r.status == "error" or r.passed is False)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
