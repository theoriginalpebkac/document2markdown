#!/usr/bin/env python3
"""doc2md.py — batch-convert a folder of documents to Markdown with per-file
quality validation, visual (figure/table) extraction, and a summary report.

Conversion (handler chosen by content sniffing, not extension)
    * PDF                       -> pdfmux (per-page self-healing + confidence)
    * .docx (Word, Google Docs) -> pandoc (+ --extract-media)
    * Confluence "Word" export  -> MHTML: extract HTML + base64 images -> pandoc
    * single-file HTML          -> de-chrome + inline data: figures -> pandoc
    *   (incl. a Jira/Confluence "Save as Word" page misnamed .doc/.htm)
    * config XML                -> verbatim (fenced, lossless, + index)
    * documentation XML         -> transform (structured Markdown)
    *   (or --yaml / --xml-mode=yaml: XML -> structure-preserving .yaml)
    * EPUB/RTF/ODT              -> pandoc
    * CSV                       -> one Markdown card per row (heading + key/value
                                   fields; comma-delimited columns auto-detected
                                   and expanded as bullet lists); supports
                                   ``--split-file`` for RAG-optimal chunking

    Maximum *local* effort is the default (pdfmux ``quality="standard"`` — the
    full agentic audit/re-extract loop). Cloud LLM extraction is opt-in via
    ``--llm`` and is meant for the occasional document that local backends can't
    handle, not the common case.

Visual extraction (PDF only)
    pdfmux emits no image/figure references, so visual content would silently
    vanish from the Markdown. doc2md adds a PyMuPDF pass that renders content
    *without a faithful Markdown representation* to PNG and references it inline:

      * Information-bearing raster images       -> PNG  (default on)
      * Complex tables (merged/ragged/nested)   -> PNG **plus** best-effort
                                                   Markdown, co-located so a RAG
                                                   chunk keeps image + table
                                                   together (reference first)
      * Simple tables (incl. multi-line cells)  -> Markdown only (newline -> <br>)
      * Text, headings, lists                   -> Markdown only

    Vector-diagram detection is opt-in (--vector-diagrams). In PDFs exported
    from Google Docs / Confluence, tables, TOCs, colored badges and highlight
    bars are all drawn as vector rectangles and can't be reliably told apart
    from real diagrams by geometry, so it is OFF by default to avoid imaging
    text/TOC pages. Decorative vectors, rules/underlines and small logos are
    filtered out. Diagram *labels* are captured as text by pdfmux regardless.

Validation (PDF)
    Raw plain text is extracted from the original PDF with ``pdftotext`` and
    compared against the generated Markdown (Markdown syntax stripped) using
    ``difflib.SequenceMatcher``. A document also fails if pdfmux's own per-page
    confidence drops below a threshold, or if it is long yet structurally empty.

Report
    ``conversion_report.json`` is written to the output directory and a
    human-readable summary is printed to stdout.

The conversion, visual-extraction, and validation helpers are deliberately
small and independent so they can be imported and unit-tested on their own.

NOTE ON --preview: pdfmux exposes no public page-range option, so ``--preview``
slices the first N pages into a temporary PDF with PyMuPDF and runs the full
pipeline on that slice. Visuals are extracted from the same slice.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import difflib
import functools
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import pymupdf  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover - exercised only where PyMuPDF is absent
    try:
        import fitz as pymupdf  # type: ignore  # older PyMuPDF name
    except ImportError:
        pymupdf = None  # type: ignore

# --------------------------------------------------------------------------- #
# Configuration / defaults
# --------------------------------------------------------------------------- #

__version__ = "0.9.0"

# Resolved once and cached. ``None`` when git or the repo is unavailable.
_GIT_COMMIT_UNSET = object()
_git_commit_cache: object = _GIT_COMMIT_UNSET


def git_commit() -> Optional[str]:
    """Short commit hash of the doc2md.py checkout (``-dirty`` when doc2md.py
    itself has uncommitted changes), or ``None`` if git/the repo is unavailable.

    Stamped into every output's provenance alongside ``__version__``. The SHA is
    the *precise* regeneration key — it pins the exact code with no manual
    discipline, where SemVer depends on remembering to bump it. Detected once and
    cached; any failure (no git on PATH, not a checkout, the single file copied
    out of the repo) degrades silently to ``None`` so provenance is best-effort
    and never blocks a conversion.
    """
    global _git_commit_cache
    if _git_commit_cache is not _GIT_COMMIT_UNSET:
        return _git_commit_cache  # type: ignore[return-value]
    _git_commit_cache = None
    self_path = Path(__file__).resolve()
    repo = str(self_path.parent)
    try:
        rev = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if rev.returncode != 0:
            return None
        sha = rev.stdout.strip()
        if not sha:
            return None
        # Scope the dirty check to doc2md.py itself, not the whole tree: it is the
        # only file the SHA is meant to pin, so a dirty README/sibling shouldn't
        # mark the *code* as -dirty — and a path-scoped status skips the full-tree
        # walk that can blow the timeout when doc2md.py is vendored in a monorepo.
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--", str(self_path)],
            capture_output=True, text=True, timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            sha += "-dirty"
        _git_commit_cache = sha
    except (OSError, subprocess.SubprocessError):
        _git_commit_cache = None
    return _git_commit_cache  # type: ignore[return-value]


DEFAULT_WORKERS = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.90
# Web exports (MHTML) carry single-page-app chrome and per-token code markup, so
# the character-diff fidelity check is inherently noisier than for PDF/DOCX even
# when the prose is faithful. Hold them to a relaxed bar instead of 0.90.
DEFAULT_WEB_SIMILARITY_THRESHOLD = 0.80
# Order-insensitive content-recall bar. The ordered char-diff (above) tanks on
# faithful conversions that legitimately reflow content (tabular PDFs, HTML→MD),
# so a doc also passes fidelity if it preserves at least this share of the
# source's word tokens — caught only when there's no confident extractor signal.
DEFAULT_CONTAINMENT_THRESHOLD = 0.90
DEFAULT_MIN_CONFIDENCE = 0.70
DEFAULT_PREVIEW_PAGES = 3
DEFAULT_FIGURE_DPI = 150
# Above this page count, --quality=auto picks "fast" (pymupdf4llm only) instead
# of "standard": on huge born-digital docs Docling's per-page table model turns
# minutes into nearly an hour for little fidelity gain (~1.3s/page on standard,
# ~0.2s/page on fast — measured on a 2,552-page doc: 55 min vs ~10 min).
DEFAULT_LARGE_DOC_PAGES = 1000
# Per-page wall-clock budget used to auto-scale PDFMUX_TIMEOUT (seconds/page).
# Generous vs the ~1.3s/page worst case so large docs never hit pdfmux's 300s
# default mid-extraction. pdfmux can't truly interrupt a running extraction, so
# this is a ceiling, not a target.
TIMEOUT_PER_PAGE_BUDGET = 3
TIMEOUT_FLOOR = 300
TIMEOUT_UNLIMITED = 86400  # what `--timeout 0` maps to (effectively no limit)

# A document whose stripped plain-text is at least this many characters but has
# zero headings, code blocks, tables, or extracted figures is structurally
# suspicious.
STRUCT_MIN_CHARS = 3000

# Extensions worth *attempting*. The actual handler is chosen by content
# sniffing (detect_format), not the extension — e.g. a Confluence ".doc" is
# really MHTML, and ".xml" may be config or documentation.
PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx", ".doc", ".mht", ".mhtml"}
XML_EXTENSIONS = {".xml"}
PANDOC_EXTENSIONS = {".html", ".htm", ".epub", ".rtf", ".odt"}
CSV_EXTENSIONS = {".csv"}
SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS | WORD_EXTENSIONS | XML_EXTENSIONS | PANDOC_EXTENSIONS | CSV_EXTENSIONS
)

# Embedded raster images smaller than this (max dimension in px, or byte size
# when dimensions can't be read) are treated as UI icons/badges and skipped.
WORD_IMAGE_MIN_PX = 64
WORD_IMAGE_MIN_BYTES = 12000

# An XML document is treated as "documentation" (transform mode) rather than
# "config" (verbatim mode) when at least this fraction of its elements are
# prose/markup tags.
XML_PROSE_TAG_RATIO = 0.15


@dataclass(frozen=True)
class VisualConfig:
    """Thresholds for the figure/table extraction pass (points unless noted).

    Defaults are calibrated against born-digital technical design PDFs whose
    diagrams are vector drawings (not raster images). The goal is to capture
    genuine diagrams/complex-tables while filtering out the thousands of
    border/rule/underline primitives that dense documents contain.
    """

    enabled: bool = True
    dpi: int = DEFAULT_FIGURE_DPI

    # Which kinds to extract. Raster images and genuinely-complex tables are
    # reliable. Vector-diagram detection is OFF by default: in PDFs exported
    # from Google Docs / Confluence, tables, tables-of-contents, colored
    # letter-badges and highlight bars are all drawn as vector rectangles and
    # are not reliably distinguishable from real diagrams by geometry, so an
    # always-on detector images text/TOC pages (bloat). Opt in with
    # --vector-diagrams when a corpus is genuinely diagram-centric. Diagram
    # *labels* are still captured as text by pdfmux regardless.
    extract_images: bool = True
    extract_tables: bool = True
    extract_diagrams: bool = False

    # Drawing pre-filter: drop thin rules/underlines and tiny specks.
    rule_thickness: float = 4.0  # a line thinner than this is a rule/underline
    rule_min_len: float = 36.0  # ...and longer than this in the other axis
    speck_size: float = 6.0  # drop primitives smaller than this in both axes

    # Clustering: merge drawing primitives whose (expanded) bboxes touch.
    cluster_gap: float = 12.0

    # A cluster is a diagram only if it is big and busy enough, and is not
    # really a table. The key anti-false-positive signal is the count of *large*
    # primitives: pages of text with inline colored glyph-badges produce many
    # tiny drawings but few large ones, whereas real diagrams have several big
    # boxes/arrows/lines.
    diagram_min_prims: int = 6
    diagram_min_w: float = 120.0
    diagram_min_h: float = 90.0
    diagram_large_prim: float = 24.0  # a primitive this big (max dim) is "large"
    diagram_min_large: int = 4  # a diagram cluster needs at least this many
    table_overlap_max: float = 0.55  # >this fraction over a table => skip (it's a table)

    # Raster images: skip anything smaller than this (logos, bullets, icons).
    image_min_w: float = 80.0
    image_min_h: float = 80.0


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class Visual:
    """One extracted visual element rendered to PNG and referenced in Markdown."""

    kind: str  # "diagram" | "image" | "table"
    page_num: int  # 0-indexed source page
    index: int  # per-page sequence number for this kind
    png_relpath: str  # path relative to the generated .md file
    alt_text: str
    caption: Optional[str] = None
    best_effort_md: Optional[str] = None  # tables only


@dataclass
class PdfConversion:
    """Return value of :func:`convert_pdf`."""

    markdown: str
    confidence: Optional[float]  # document-level pdfmux confidence
    min_page_confidence: Optional[float]
    visuals: List[Visual] = field(default_factory=list)
    cleaning: Dict[str, object] = field(default_factory=dict)
    page_source: str = "process-fallback"  # see :func:`_extract_pdf_pages`
    page_markers_applied: bool = False  # were page-boundary markers woven in
    # PyMuPDF table objects find_tables() returned but whose ``bbox`` couldn't be
    # computed (degenerate, empty cell list). Skipped, not imaged — surfaced so a
    # silent drop is auditable rather than invisible.
    degenerate_tables: int = 0


@dataclass
class FileRecord:
    """Everything we know about one input file after processing."""

    filename: str
    source_path: str
    converter: Optional[str] = None  # "pdfmux" | "pandoc" | None
    status: str = "pending"  # "converted" | "skipped" | "error"
    output_path: Optional[str] = None
    similarity: Optional[float] = None
    similarity_method: Optional[str] = None  # "sequence" (difflib) | "containment"
    content_recall: Optional[float] = None  # order-insensitive token recall
    pdfmux_confidence: Optional[float] = None
    min_page_confidence: Optional[float] = None
    structural: Dict[str, int] = field(default_factory=dict)
    figures: Dict[str, int] = field(default_factory=dict)
    structural_ok: Optional[bool] = None
    structural_reason: Optional[str] = None
    similarity_ok: Optional[bool] = None
    confidence_ok: Optional[bool] = None
    fidelity_ok: Optional[bool] = None  # --yaml: YAML round-trips to source XML
    fidelity_reason: Optional[str] = None
    passed: Optional[bool] = None
    preview: bool = False
    used_llm: bool = False
    # Choices doc2md made automatically (e.g. "auto" OCR / xml-mode), each with
    # the value chosen, why, and the flag to override it. Surfaced on stdout and
    # in conversion_report.json so an auto decision is never silent.
    auto_decisions: List[Dict[str, str]] = field(default_factory=list)
    # Markdown-cleanup stats (lines/tables removed, splits repaired); empty when
    # cleaning is disabled (--no-clean) or the format isn't PDF.
    cleaning: Dict[str, object] = field(default_factory=dict)
    # Non-fatal extraction anomalies worth auditing, keyed by kind -> count (e.g.
    # ``degenerate_tables``: PyMuPDF tables skipped because their bbox couldn't be
    # computed). Empty when nothing odd happened. Kept out of ``figures`` so it
    # never skews the figures-extracted total.
    extraction_warnings: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Dependency checks
# --------------------------------------------------------------------------- #


def check_dependencies(*, need_yaml: bool = False) -> Dict[str, bool]:
    """Probe for the external tools / packages we rely on.

    Returns a mapping of dependency name -> availability. Prints actionable
    install instructions for anything missing rather than failing silently.

    ``xmltodict``/``PyYAML`` are optional (only ``--yaml`` needs them), so they
    are probed and warned about only when ``need_yaml`` is set — keeping the
    output clean for the common Markdown-only run.
    """
    import importlib.util

    available = {
        "csv": True,  # stdlib, always available
        "pdfmux": importlib.util.find_spec("pdfmux") is not None,
        "pymupdf": pymupdf is not None,
        "pandoc": shutil.which("pandoc") is not None,
        "pdftotext": shutil.which("pdftotext") is not None,
    }

    instructions: Dict[str, str] = {
        "csv": "",  # always available (stdlib); no warning needed
        "pdfmux": (
            "pdfmux (PDF -> Markdown converter) is not importable.\n"
            "    Install it with:  pip install -r requirements.txt\n"
            "    (pdfmux requires Python 3.11+.)"
        ),
        "pymupdf": (
            "PyMuPDF is not importable; figure/table extraction and --preview "
            "will be unavailable.\n"
            "    Install it with:  pip install -r requirements.txt"
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

    if need_yaml:
        available["xmltodict"] = importlib.util.find_spec("xmltodict") is not None
        available["yaml"] = importlib.util.find_spec("yaml") is not None
        instructions["xmltodict"] = instructions["yaml"] = (
            "--yaml/--rag need the 'xmltodict' and 'PyYAML' packages, which are "
            "not importable.\n"
            "    Install them with:  pip install -r requirements.txt"
        )

    seen: set = set()
    for name, ok in available.items():
        msg = instructions.get(name, "")
        if not ok and msg and msg not in seen:  # yaml deps share one message -> warn once
            print("[warning] " + msg, file=sys.stderr)
            seen.add(msg)

    return available


# --------------------------------------------------------------------------- #
# RAG provenance metadata: YAML frontmatter + page-boundary markers
# --------------------------------------------------------------------------- #
#
# Two layers of "invisible" provenance for RAG pipelines, both governed by one
# switch (``--no-rag-metadata`` to disable; on by default):
#
#  * **Frontmatter** — a YAML block at the top of every ``.md`` output carrying
#    routing/filtering facts (source file, engine, confidence, …). It is the
#    citation anchor for retrieved chunks. String values are emitted via
#    :func:`safe_yaml_string` (``json.dumps`` — JSON being a subset of YAML)
#    so colons/quotes/Unicode can't break the block and YAML 1.1's type coercion
#    (the "Norway problem": ``NO`` -> ``False``; ``1.10`` -> ``1.1``) can't
#    silently corrupt a string field.
#  * **Page markers** — ``<!-- doc2md:page=N -->`` HTML comments at every PDF
#    page boundary, so a chunk's source page survives even after page numbers are
#    stripped from the prose. Namespaced so the fidelity check (and downstream
#    parsers) can target *our* markers without touching genuine source comments.
#
# Both are stripped before the fidelity comparison (see :func:`strip_markdown`),
# so they never depress the pdftotext-similarity score.

# Marker emitted at every PDF page boundary. The ``=`` is load-bearing twice
# over: it makes the pattern trivially greppable downstream, and it makes
# clean_markdown's tag-split repair skip the marker (it bails on any ``<...>``
# containing ``=``), so the marker survives cleaning untouched.
PAGE_MARKER_TEMPLATE = "<!-- doc2md:page=%d -->"
# Bare marker only (no surrounding whitespace) so stripping an *inline* marker
# leaves a separator behind — "long <!-- ... --> sentence" must not collapse to
# "longsentence". The fidelity comparison normalizes whitespace afterwards.
_PAGE_MARKER_RE = re.compile(r"<!--\s*doc2md:page=\d+\s*-->")
# A leading YAML frontmatter block (only ever at the very top of a file).
_FRONTMATTER_RE = re.compile(r"\A﻿?---\n.*?\n---\n", re.S)

# Sentence-final punctuation, optionally trailed by closing quotes/brackets, at
# end of a line (":" included — it terminates a clause before a list/figure).
_SENTENCE_END_RE = re.compile(r"[.!?:][\"')\]»”]*\s*$")
# Same, mid-text, used to find the first sentence boundary to snap a marker to.
# Excludes ":" so "Note: ..." doesn't snap early; the boundary is *after* the
# punctuation and any closing quotes, before whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?][\"')\]»”]*(?=\s)")


def safe_yaml_string(val: object) -> str:
    """Quote a value as a YAML string that survives both YAML 1.1 and 1.2.

    JSON is a subset of YAML 1.2, and ``json.dumps`` always double-quotes a
    string with standard escapes that YAML 1.1 also accepts — so this both
    escapes colons/quotes/control chars *and* freezes the type as a string,
    neutralizing YAML 1.1's coercion of bare scalars (``NO`` -> bool,
    ``1.10`` -> float). ``ensure_ascii=False`` keeps Unicode readable.
    """
    return json.dumps(str(val), ensure_ascii=False)


def _is_structural_line(stripped: str) -> bool:
    """True if a (already-stripped) line is a Markdown block element, not prose.

    Used to keep sentence-snap from placing a page marker inside a table, code
    fence, list item, heading, blockquote, image, or existing comment.
    """
    if not stripped:
        return True
    if stripped.startswith(("#", ">", "|", "```", "~~~", "<!--", "![")):
        return True
    if re.match(r"[-*+]\s", stripped):
        return True
    if re.match(r"\d+[.)]\s", stripped):
        return True
    if _TABLE_DELIM_RE.match(stripped):
        return True
    return False


def _last_nonempty_line(text: str) -> str:
    for ln in reversed(text.splitlines()):
        if ln.strip():
            return ln.strip()
    return ""


def _first_nonempty_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _ends_midsentence(text: str) -> bool:
    """True if ``text``'s last prose line looks like a sentence cut by a page break.

    A structural last line (table row, heading, list item, …) is never a
    mid-sentence break. A line ending in sentence-final punctuation is complete;
    a line ending in a hyphen, or in anything else, is treated as continuing.
    """
    line = _last_nonempty_line(text)
    if not line or _is_structural_line(line):
        return False
    if _SENTENCE_END_RE.search(line):
        return False
    return True


def _starts_continuation(text: str) -> bool:
    """True if ``text``'s first line reads as prose continuing a prior sentence."""
    line = _first_nonempty_line(text)
    if not line or _is_structural_line(line):
        return False
    return line[0].isalnum() or line[0] in "(\"'«“"


def _split_first_sentence(text: str) -> Tuple[Optional[str], str]:
    """Split ``text`` after the first sentence boundary in its leading prose.

    Returns ``(head, rest)`` where ``head`` ends the bridging sentence. The
    search is confined to the leading prose paragraph (it stops at the first
    blank or structural line), so a marker can never be snapped into a table or
    code block. A paragraph break counts as a boundary even without terminal
    punctuation. Returns ``(None, text)`` only when the whole page is one
    unbroken prose paragraph with no sentence end — the drift-cap case.
    """
    lines = text.splitlines(keepends=True)
    region_len = 0
    for ln in lines:
        if _is_structural_line(ln.strip()):  # also true for blank lines
            break
        region_len += len(ln)
    if region_len == 0:
        return None, text
    region = text[:region_len]
    m = _SENTENCE_SPLIT_RE.search(region)
    if m:
        cut = m.end()
        return text[:cut].rstrip(), text[cut:].lstrip()
    if region_len < len(text):  # paragraph ends at a blank/structural boundary
        return region.rstrip(), text[region_len:].lstrip()
    return None, text


def _join_bridge(prev_tail: str, head: str) -> str:
    """Join a sentence fragment split across a page break into contiguous prose.

    Dehyphenates a word broken by the page break (``config-`` + ``uration`` ->
    ``configuration``); otherwise joins with a single space so the sentence
    reflows naturally instead of being torn into two paragraphs by a hard break.
    """
    p = prev_tail.rstrip()
    h = head.lstrip()
    if not h:
        return p
    if p.endswith("-") and len(p) >= 2 and p[-2].isalpha() and h[0].isalpha():
        return p[:-1] + h
    return p + " " + h


def _assemble_with_page_markers(
    units: List[Tuple[int, str, str]], mark_pages: bool
) -> str:
    """Assemble per-page (index, text, visual_md) units into the document body.

    With ``mark_pages`` off this reproduces the legacy join exactly (page text
    and visual blocks separated by blank lines). With it on, a
    ``<!-- doc2md:page=N -->`` marker is emitted at every page boundary
    (including blank pages, for continuity), a sentence bridging two pages is
    rejoined into contiguous prose, and the marker is snapped forward to just
    after that sentence completes — so the marker never falls mid-sentence and
    the bridged sentence is never split across chunks. When a page is one
    unbroken paragraph with no sentence end (drift cap), the marker is placed
    inline at the exact boundary rather than drifting past the next page.
    """
    if not mark_pages:
        parts: List[str] = []
        for _, text, visual_md in units:
            if text.strip():
                parts.append(text)
            if visual_md:
                parts.append(visual_md)
        return "\n\n".join(p for p in parts if p)

    segs: List[List[str]] = []  # [text, separator_before]

    def emit(text: str, sep: str) -> None:
        segs.append([text, sep])

    prose_idx: Optional[int] = None  # seg index of the last bridge-eligible prose
    prose_text = ""
    for page_index, text, visual_md in units:
        marker = PAGE_MARKER_TEMPLATE % (page_index + 1)
        if text.strip():
            bridging = (
                prose_idx is not None
                and _ends_midsentence(prose_text)
                and _starts_continuation(text)
            )
            if bridging:
                head, rest = _split_first_sentence(text)
                if head is not None:
                    segs[prose_idx][0] = _join_bridge(segs[prose_idx][0], head)
                    emit(marker, "\n\n")  # snap: marker after the completed sentence
                    if rest.strip():
                        emit(rest, "\n\n")
                        prose_idx = len(segs) - 1
                        prose_text = rest
                    else:  # whole page was just the bridging tail
                        prose_text = segs[prose_idx][0]
                else:  # drift cap: inline marker, keep prose contiguous
                    emit(marker, " ")
                    emit(text, " ")
                    prose_idx = len(segs) - 1
                    prose_text = text
            else:
                emit(marker, "\n\n")
                emit(text, "\n\n")
                prose_idx = len(segs) - 1
                prose_text = text
        else:  # blank page: still mark it, but it breaks prose adjacency
            emit(marker, "\n\n")
            prose_idx, prose_text = None, ""
        if visual_md:  # a figure block separates this page's prose from the next
            emit(visual_md, "\n\n")
            prose_idx, prose_text = None, ""

    out = ""
    for text, sep in segs:
        if not text:
            continue
        out = text if out == "" else out + sep + text
    return out


def _derive_title(markdown: str, src: Path, fmt: str) -> str:
    """Best-effort document title: PDF metadata -> first H1 -> de-slugified name."""
    if fmt == "pdf" and pymupdf is not None:
        try:
            doc = pymupdf.open(str(src))
            try:
                meta_title = (doc.metadata or {}).get("title") or ""
            finally:
                doc.close()
            if meta_title.strip():
                return meta_title.strip()
        except Exception:  # pragma: no cover - metadata read is best-effort
            pass
    m = re.search(r"^\s{0,3}#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return re.sub(r"[-_]+", " ", src.stem).strip()


def _frontmatter_source_path(src: Path, input_root: Path, absolute: bool) -> str:
    """The ``source_path`` value: relative to the input root by default (so a
    shared corpus doesn't leak machine/folder names), absolute when requested."""
    if absolute:
        return str(src.resolve())
    try:
        return str(src.resolve().relative_to(input_root.resolve()))
    except ValueError:  # src outside input_root (e.g. single-file run)
        return src.name


def build_frontmatter(
    *,
    title: str,
    source_file: str,
    source_path: str,
    fmt: str,
    engine: Optional[str],
    mtime: float,
    quality: Optional[str] = None,
    page_count: Optional[int] = None,
    confidence: Optional[float] = None,
    part: Optional[int] = None,
    parts: Optional[int] = None,
) -> str:
    """Build the YAML frontmatter block (provenance + engine facts) for a ``.md``.

    Strings go through :func:`safe_yaml_string`; only genuine numerics
    (``page_count``, ``confidence``, ``part``/``parts``) are bare. ``confidence``
    is omitted when ``None`` so typed downstream loaders never meet an unexpected
    null; ``quality``/``page_count``/``confidence`` are PDF-only. ``part``/
    ``parts`` carry split-output provenance (e.g. a large CSV split into
    ``<stem>-partNNN.md``) so every atomic file knows which slice it is.
    ``converted`` is the source file's mtime (deterministic, idempotent — not
    wall-clock "now") as a quoted ISO date string.
    """
    converted = datetime.date.fromtimestamp(mtime).isoformat()
    lines = ["---"]
    lines.append("title: " + safe_yaml_string(title))
    lines.append("source_file: " + safe_yaml_string(source_file))
    lines.append("source_path: " + safe_yaml_string(source_path))
    lines.append("format: " + safe_yaml_string(fmt))
    if engine:
        lines.append("engine: " + safe_yaml_string(engine))
    if quality:
        lines.append("quality: " + safe_yaml_string(quality))
    if page_count is not None:
        lines.append("page_count: %d" % page_count)
    if part is not None:
        lines.append("part: %d" % part)
    if parts is not None:
        lines.append("parts: %d" % parts)
    if confidence is not None:
        lines.append("confidence: %s" % round(confidence, 4))
    lines.append("converted: " + safe_yaml_string(converted))
    lines.append("doc2md_version: " + safe_yaml_string(__version__))
    commit = git_commit()
    if commit:
        lines.append("doc2md_commit: " + safe_yaml_string(commit))
    lines.append("---")
    return "\n".join(lines)


def yaml_provenance_header() -> str:
    """Comment-line provenance stamp prepended to ``.yaml`` outputs.

    YAML outputs (XML→config conversions) can't carry a ``---`` frontmatter block
    — its top-level ``---`` would open a second YAML document — so the same
    version/commit provenance rides as leading ``#`` comments instead. Comments
    are ignored on parse, so this is invisible to downstream loaders and to
    :func:`yaml_fidelity_check`. Trailing newline included so the body follows
    cleanly. Mirrors the ``doc2md_version``/``doc2md_commit`` keys in
    :func:`build_frontmatter`.
    """
    lines = ["# doc2md_version: %s" % __version__]
    commit = git_commit()
    if commit:
        lines.append("# doc2md_commit: %s" % commit)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Visual extraction (PyMuPDF) — figures, diagrams, complex tables
# --------------------------------------------------------------------------- #


def _rect(obj):  # small helper: build a pymupdf.Rect
    return pymupdf.Rect(obj)


def _area(r) -> float:  # version-proof rect area (clamped to >= 0)
    return max(0.0, r.width) * max(0.0, r.height)


def _coverage_ratio(box, others: List["pymupdf.Rect"]) -> float:
    """Approx fraction of ``box`` area covered by the union of ``others``.

    Uses summed intersection area (slight double-counting on overlap) capped at
    1.0 — good enough to tell "mostly text/table" from "mostly drawing".
    """
    area = _area(box)
    if area <= 0:
        return 0.0
    covered = 0.0
    for o in others:
        inter = box & o
        if not inter.is_empty:
            covered += _area(inter)
    return min(covered / area, 1.0)


def _cluster_rects(
    rects: List["pymupdf.Rect"], gap: float
) -> List[Tuple["pymupdf.Rect", List["pymupdf.Rect"]]]:
    """Union-find clustering of rects whose bboxes touch when expanded by ``gap``.

    Returns a list of (merged_bbox, member_rects). O(n^2); fine for the per-page
    primitive counts (hundreds) seen in real documents.
    """
    n = len(rects)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    grown = [r + (-gap, -gap, gap, gap) for r in rects]
    for i in range(n):
        for j in range(i + 1, n):
            if grown[i].intersects(grown[j]):
                union(i, j)

    groups: Dict[int, List["pymupdf.Rect"]] = {}
    for i, r in enumerate(rects):
        groups.setdefault(find(i), []).append(r)

    clusters = []
    for members in groups.values():
        bbox = members[0]
        for m in members[1:]:
            bbox = bbox | m
        clusters.append((bbox, members))
    return clusters


def detect_diagrams(page, table_rects, cfg: VisualConfig) -> List["pymupdf.Rect"]:
    """Detect vector-diagram regions on a page, filtering out rules and prose.

    This is the anti-bloat core: dense docs contain thousands of thin
    border/rule/underline primitives, which are dropped before clustering;
    surviving clusters must be large, busy, and not dominated by text or tables.
    """
    page_rect = page.rect
    rects: List["pymupdf.Rect"] = []
    for d in page.get_drawings():
        r = _rect(d["rect"]) & page_rect
        if r.is_empty:
            continue
        w, h = r.width, r.height
        # thin long rule / underline / column separator
        if (h < cfg.rule_thickness and w > cfg.rule_min_len) or (
            w < cfg.rule_thickness and h > cfg.rule_min_len
        ):
            continue
        # tiny speck (bullet glyph, dot)
        if w < cfg.speck_size and h < cfg.speck_size:
            continue
        rects.append(r)

    if not rects:
        return []

    def _is_large(r) -> bool:
        return max(r.width, r.height) >= cfg.diagram_large_prim

    out: List["pymupdf.Rect"] = []
    for bbox, members in _cluster_rects(rects, cfg.cluster_gap):
        if len(members) < cfg.diagram_min_prims:
            continue
        if sum(1 for m in members if _is_large(m)) < cfg.diagram_min_large:
            continue  # mostly tiny glyphs (inline badges/icons) => not a diagram
        if bbox.width < cfg.diagram_min_w or bbox.height < cfg.diagram_min_h:
            continue
        if _coverage_ratio(bbox, table_rects) > cfg.table_overlap_max:
            continue
        out.append(bbox)
    return out


def detect_images(page, cfg: VisualConfig) -> List["pymupdf.Rect"]:
    """Information-bearing raster images on a page (logos/icons filtered by size)."""
    page_rect = page.rect
    out: List["pymupdf.Rect"] = []
    seen: List["pymupdf.Rect"] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            r = _rect(r) & page_rect
            if r.width < cfg.image_min_w or r.height < cfg.image_min_h:
                continue
            if any(_area(r & s) / max(_area(r), 1.0) > 0.8 for s in seen):
                continue  # de-dupe near-identical placements
            seen.append(r)
            out.append(r)
    return out


def table_is_complex(table, cfg: VisualConfig) -> bool:
    """Decide whether a table's *grid structure* lacks a faithful GFM representation.

    Only genuinely un-linearizable tables are imaged: ragged rows, merged/
    spanning cells, or extraction failure. Multi-line cell *text* is NOT complex
    — it is representable as Markdown (newline -> ``<br>`` in
    :func:`_safe_to_markdown`). This avoids imaging the very common wrapped-text
    tables in dense reference docs, which would otherwise cause severe bloat.
    """
    try:
        rows = table.extract()
    except Exception:
        return True  # cannot extract cleanly -> image it

    if not rows:
        return False

    if len({len(r) for r in rows}) > 1:
        return True  # ragged rows => spanning structure that pipes can't show

    # Merged/spanning cells leave None placeholders in PyMuPDF's geometric grid.
    try:
        if any(c is None for c in table.cells):
            return True
    except Exception:
        pass

    return False


def _safe_to_markdown(table) -> str:
    """Best-effort Markdown for a table, falling back to a manual pipe build."""
    try:
        md = table.to_markdown()
        if md and md.strip():
            return md.strip()
    except Exception:
        pass
    try:
        rows = table.extract()
    except Exception:
        return "_(table could not be linearized — see image)_"
    rows = [
        ["" if c is None else str(c).strip().replace("\n", "<br>") for c in r]
        for r in rows
    ]
    if not rows:
        return "_(empty table — see image)_"
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _nearest_caption(page, box, max_gap: float = 60.0) -> Optional[str]:
    """A short caption near a region — prefer a 'Figure/Table N' line below it."""
    candidates = []
    for b in page.get_text("blocks"):
        if len(b) < 7 or b[6] != 0:
            continue
        text = str(b[4]).strip()
        if not text:
            continue
        rb = _rect(b[:4])
        below_gap = rb.y0 - box.y1
        above_gap = box.y0 - rb.y1
        horiz_overlap = min(box.x1, rb.x1) - max(box.x0, rb.x0) > 0
        if horiz_overlap and 0 <= below_gap <= max_gap:
            candidates.append((0, below_gap, text))  # below preferred
        elif horiz_overlap and 0 <= above_gap <= max_gap:
            candidates.append((1, above_gap, text))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    caption = candidates[0][2]
    caption = re.sub(r"\s+", " ", caption)
    return caption[:160] + ("…" if len(caption) > 160 else "")


def _render_region(page, box, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, clip=box)
    pix.save(str(out_path))  # PNG inferred from extension


# Maps internal kind -> filename token and human-readable phrase for alt text.
_KIND_WORD = {"image": "figure", "table": "table", "diagram": "diagram"}
_KIND_PHRASE = {"image": "figure", "table": "complex table", "diagram": "diagram"}


def _alt_text(doc_title: str, kind: str, page_index: int, caption: Optional[str]) -> str:
    base = "%s — page %d, %s" % (doc_title, page_index + 1, _KIND_PHRASE[kind])
    return base + (": " + caption if caption else "")


@contextlib.contextmanager
def _classic_table_detection():
    """Force PyMuPDF's deterministic line-based table finder for our visual pass.

    PyMuPDF 1.28+ ships an ONNX "layout" analyzer. When pdfmux/pymupdf4llm is in
    the process it installs that analyzer globally as ``pymupdf._get_layout``
    (just importing ``pymupdf4llm`` is enough), and ``Page.find_tables()`` then
    routes through it. On some PDFs that native path **segfaults** inside MuPDF
    (``fz_table_hunt_within_bounds``) — an uncatchable crash that no ``try`` can
    stop. doc2md's figure pass only needs geometric table *regions* (to extract
    genuinely-complex tables and mask them out of diagram detection), which the
    classic line-based detector provides and was calibrated against. So we
    disable the layout analyzer just around our own ``find_tables()`` call and
    restore it after — pdfmux's extraction keeps using layout mode for its own
    Markdown. No-op on PyMuPDF builds without the layout hook.
    """
    if pymupdf is None or not hasattr(pymupdf, "_get_layout"):
        yield
        return
    saved = pymupdf._get_layout
    pymupdf._get_layout = None
    try:
        yield
    finally:
        pymupdf._get_layout = saved


def extract_page_visuals(
    page,
    page_index: int,
    fig_dir: Path,
    rel_base: str,
    slug: str,
    doc_title: str,
    cfg: VisualConfig,
    stats: Optional[Dict[str, int]] = None,
) -> List[Visual]:
    """Render diagrams, info images, and complex tables on one page to PNG.

    Pure with respect to pdfmux — takes a PyMuPDF page, so it can be tested
    against any PDF independently of the conversion pipeline. ``slug`` prefixes
    every filename so PNGs stay unique/traceable even if relocated, and
    ``doc_title`` is woven into alt text for decontextualized RAG chunks.

    When ``stats`` is given, non-fatal extraction anomalies are accumulated into
    it (currently ``degenerate_tables``) so callers can surface them.
    """
    visuals: List[Visual] = []
    page_rect = page.rect

    def _emit(kind: str, idx: int, rect, best_md: Optional[str] = None) -> Visual:
        fname = "%s-p%03d-%s%02d.png" % (slug, page_index + 1, _KIND_WORD[kind], idx)
        _render_region(page, rect, fig_dir / fname, cfg.dpi)
        cap = _nearest_caption(page, rect)
        return Visual(
            kind=kind,
            page_num=page_index,
            index=idx,
            png_relpath="%s/%s" % (rel_base, fname),
            alt_text=_alt_text(doc_title, kind, page_index, cap),
            caption=cap,
            best_effort_md=best_md,
        )

    # Tables first, so diagram detection can avoid double-imaging table borders.
    # find_tables() is needed for table extraction *and* to mask table regions
    # out of diagram detection, so run it whenever either is enabled.
    table_objs = []
    if cfg.extract_tables or cfg.extract_diagrams:
        try:
            with _classic_table_detection():
                table_objs = list(page.find_tables().tables)
        except Exception:
            table_objs = []
    # PyMuPDF's ``Table.bbox`` raises ValueError("min() iterable argument is
    # empty") on degenerate tables whose cell list is empty, so compute each
    # rect defensively and drop any table that can't yield one — keeping
    # table_objs and table_rects aligned for the zip() below.
    paired = []
    for t in table_objs:
        try:
            paired.append((t, _rect(t.bbox) & page_rect))
        except Exception:
            if stats is not None:
                stats["degenerate_tables"] = stats.get("degenerate_tables", 0) + 1
            continue
    table_objs = [t for t, _ in paired]
    table_rects = [r for _, r in paired]

    t_idx = 0
    for tab, trect in zip(table_objs, table_rects) if cfg.extract_tables else []:
        if trect.is_empty or trect.width < 40 or trect.height < 20:
            continue
        if not table_is_complex(tab, cfg):
            continue  # simple table -> pdfmux renders it as Markdown; skip
        t_idx += 1
        visuals.append(_emit("table", t_idx, trect, best_md=_safe_to_markdown(tab)))

    d_idx = 0
    diagram_rects = detect_diagrams(page, table_rects, cfg) if cfg.extract_diagrams else []
    for drect in diagram_rects:
        d_idx += 1
        visuals.append(_emit("diagram", d_idx, drect))

    i_idx = 0
    image_rects = detect_images(page, cfg) if cfg.extract_images else []
    for irect in image_rects:
        if _coverage_ratio(irect, table_rects) > 0.8:
            continue  # already captured as a table image
        i_idx += 1
        visuals.append(_emit("image", i_idx, irect))

    return visuals


def render_visual_markdown(visuals: List[Visual]) -> str:
    """Render a co-located Markdown block for a page's visuals (reference-first).

    For complex tables the image reference is emitted immediately *before* the
    best-effort Markdown table, with no heading between them, so a structural or
    token-bounded RAG chunker keeps the image and the table in the same chunk.
    """
    blocks: List[str] = []
    for v in visuals:
        if v.kind == "table":
            blocks.append(
                "> **[Table — p.%d]** Rendered as image (authoritative); "
                "best-effort Markdown follows.\n"
                "> ![%s](%s)\n\n%s"
                % (v.page_num + 1, v.alt_text, v.png_relpath, v.best_effort_md or "")
            )
        else:
            label = "Diagram" if v.kind == "diagram" else "Figure"
            note = v.caption or "Visual element with no Markdown equivalent — review source."
            blocks.append(
                "> **[%s — p.%d]** %s\n> ![%s](%s)"
                % (label, v.page_num + 1, note, v.alt_text, v.png_relpath)
            )
    return "\n\n".join(blocks)


def build_markdown_with_visuals(
    pages_text: List[Tuple[int, str]],
    pdf_path: Path,
    dest: Path,
    cfg: VisualConfig,
    doc_title: str,
    mark_pages: bool = False,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[str, List[Visual]]:
    """Combine pdfmux text with extracted visual blocks, keeping figures near
    their page's context.

    Visual extraction runs over the PDF's **real page range** (PyMuPDF owns page
    geometry), *independent* of how pdfmux segmented the text — so figures on
    every page are captured, not just those on the one page that happens to align
    with a text unit. PNGs go in ``<slug>/figures/`` next to the .md, named
    ``<slug>-pNNN-<kind>NN.png``. ``doc_title`` is used in figure alt text.

    Placement depends on whether pdfmux gave us **per-page text**:

    * Per-page (``len(pages_text) > 1``): each page's visuals are interleaved
      right after that page's text, and — with ``mark_pages`` — a
      ``<!-- doc2md:page=N -->`` marker is woven in at every page boundary.
    * Single combined blob (the Docling-table / LLM ``process()`` paths — see
      :func:`_extract_pdf_pages`): there is no per-page anchor to interleave
      into, so visuals are appended **grouped by page** under a heading at the
      end of the document. Provenance survives via each block's ``[Figure —
      p.N]`` label and page-stamped alt text; only inline placement is degraded.
      Page markers are suppressed here for the same reason (a lone ``page=1``
      would mislead).
    """
    visuals_enabled = cfg.enabled and pymupdf is not None
    have_per_page_text = len(pages_text) > 1
    mark_pages = mark_pages and have_per_page_text

    slug = dest.stem  # already slugified by _output_path_for
    fig_dir = dest.parent / slug / "figures"
    rel_base = "%s/figures" % slug

    visuals_all: List[Visual] = []
    visual_md_by_page: Dict[int, str] = {}
    doc = pymupdf.open(str(pdf_path)) if visuals_enabled else None
    try:
        if doc is not None:
            for page_index in range(doc.page_count):
                vis = extract_page_visuals(
                    doc[page_index], page_index, fig_dir, rel_base, slug, doc_title, cfg,
                    stats=stats,
                )
                if vis:
                    visual_md_by_page[page_index] = render_visual_markdown(vis)
                    visuals_all.extend(vis)
    finally:
        if doc is not None:
            doc.close()

    if have_per_page_text:
        units = [
            (idx, text or "", visual_md_by_page.get(idx, "")) for idx, text in pages_text
        ]
        return _assemble_with_page_markers(units, mark_pages), visuals_all

    # Single combined blob: emit the text, then append page-ordered visual blocks.
    blob = "\n\n".join(t for _, t in pages_text if t and t.strip())
    grouped = [visual_md_by_page[i] for i in sorted(visual_md_by_page)]
    if not grouped:
        return blob, visuals_all
    parts = ([blob] if blob else []) + ["## Figures & tables (by page)"] + grouped
    return "\n\n".join(parts), visuals_all


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def _slice_pdf_pages(src: Path, n_pages: int, dest_dir: Path) -> Path:
    """Write the first ``n_pages`` of ``src`` to a temp PDF and return its path."""
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is required for --preview but is not installed")
    doc = pymupdf.open(str(src))
    try:
        last = min(n_pages, doc.page_count) - 1
        out = pymupdf.open()
        try:
            out.insert_pdf(doc, from_page=0, to_page=last)
            dest = dest_dir / src.name
            out.save(str(dest))
        finally:
            out.close()
    finally:
        doc.close()
    return dest


# --------------------------------------------------------------------------- #
# Docling OCR policy
# --------------------------------------------------------------------------- #
#
# pdfmux routes table-bearing PDFs to Docling, which builds a bare
# ``DocumentConverter()`` — and Docling defaults to running full-page OCR
# (Tesseract) on *every* page on top of its TableFormer structure model. On a
# born-digital PDF the text layer is already authoritative, so that OCR pass is
# pure wasted time (and the source of the "Image too small to scale" / "Line
# cannot be recognized" noise) without improving fidelity: TableFormer reads
# cell text from the text layer, not from OCR. On a hundreds-of-pages,
# table-dense document this is the difference between minutes and a run that
# never finishes.
#
# So we disable Docling's OCR for PDFs that are clearly text-based, while
# leaving it ON for scanned / image PDFs that genuinely need it. The decision
# is made per file from the path passed to each Docling ``convert()`` call, so
# it stays correct under both pdfmux's internal worker thread and doc2md's own
# ProcessPoolExecutor — there is no shared mutable selection state to race on.

# A PDF is treated as text-based when at least this share of its non-empty
# sampled pages carry a real text layer (mirrors pdfmux's own 0.80 digital
# ratio). Conservative by design: when unsure, leave OCR on.
DOCLING_TEXT_BASED_RATIO = 0.80
_DOCLING_TEXT_SAMPLE_PAGES = 30

# How to override the auto OCR decision, shown to the user with every report.
_OCR_OVERRIDE_HINT = "--ocr on (force OCR) | --ocr off (force skip)"


def _ocr_decision(src: Path, mode: str) -> Dict[str, str]:
    """Resolve the Docling OCR decision for one PDF into a reportable record.

    Single source of truth shared by the report/stdout path and the converter
    wrapper, so what we tell the user always matches what actually runs. For
    ``mode="auto"`` it classifies the PDF (see :func:`_pdf_is_text_based`);
    ``"on"``/``"off"`` are forced. Returns a dict with ``setting``, ``choice``
    ("on"/"off"), a human ``reason``, and the ``override`` hint.
    """
    if mode == "on":
        choice, reason = "on", "forced on (--ocr on)"
    elif mode == "off":
        choice, reason = "off", "forced off (--ocr off)"
    elif _pdf_is_text_based(src):
        choice = "off"
        reason = "auto: born-digital PDF, text layer authoritative — OCR skipped"
    else:
        choice = "on"
        reason = "auto: scanned/image PDF — OCR needed"
    return {
        "setting": "ocr",
        "choice": choice,
        "reason": reason,
        "override": _OCR_OVERRIDE_HINT,
    }


def _pdf_is_text_based(path: Path) -> bool:
    """True if *path* is a born-digital PDF whose text layer makes OCR redundant.

    Samples up to ``_DOCLING_TEXT_SAMPLE_PAGES`` pages spread across the
    document and counts those carrying a real text layer (>50 chars — the same
    bar pdfmux uses). Returns True only when the text-bearing share of non-empty
    pages clears :data:`DOCLING_TEXT_BASED_RATIO`, so scanned/image PDFs keep
    OCR. On any error (or without PyMuPDF) it returns ``False`` — i.e. it leaves
    OCR enabled rather than risk dropping it on a doc that needs it.
    """
    if pymupdf is None:
        return False
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        return False
    try:
        total = doc.page_count
        if total == 0:
            return False
        if total <= _DOCLING_TEXT_SAMPLE_PAGES:
            sample = range(total)
        else:
            step = total / _DOCLING_TEXT_SAMPLE_PAGES
            sample = sorted({int(i * step) for i in range(_DOCLING_TEXT_SAMPLE_PAGES)})
        text_pages = 0
        non_empty = 0
        for pn in sample:
            page = doc[pn]
            text_len = len(page.get_text("text").strip())
            has_images = bool(page.get_images(full=True))
            if text_len < 20 and not has_images:
                continue  # empty page — excluded from the ratio (matches pdfmux)
            non_empty += 1
            if text_len > 50:
                text_pages += 1
        if non_empty == 0:
            return False
        return (text_pages / non_empty) >= DOCLING_TEXT_BASED_RATIO
    finally:
        doc.close()


def _doc_to_path(doc) -> Optional[Path]:
    """Best-effort source path for a ``pymupdf4llm.to_markdown`` argument.

    ``doc`` may be a path (str/Path) or an open ``pymupdf.Document`` whose
    ``.name`` is its file path. Returns ``None`` if neither yields a path.
    """
    try:
        if isinstance(doc, (str, Path)):
            return Path(str(doc))
        name = getattr(doc, "name", None)
        if name:
            return Path(name)
    except Exception:
        pass
    return None


def _patch_pymupdf4llm_ocr(mode: str) -> bool:
    """Make pdfmux's pymupdf4llm extraction honor the OCR policy.

    pdfmux's FastExtractor / multi-pass path calls ``pymupdf4llm.to_markdown``
    *without* ``use_ocr``; with PyMuPDF's layout engine active that defaults to
    OCR on, so Tesseract runs on every page the parser deems "needs OCR" — the
    real cost (and the "Using Tesseract" / "Image too small to scale" noise) on
    a born-digital, table-dense doc. We wrap ``to_markdown`` to inject
    ``use_ocr=False`` when the per-file decision is "off", unless the caller set
    ``use_ocr``/``force_ocr`` explicitly. No-op when the layout engine is
    inactive (the legacy path does no OCR and would just warn about the kwarg).
    Returns True if the patch is in place.
    """
    try:
        import pymupdf4llm
    except Exception:
        return False
    if not getattr(pymupdf4llm, "_use_layout", False):
        return False  # legacy (non-layout) path does no OCR
    if getattr(pymupdf4llm.to_markdown, "_doc2md_ocr_wrapped", False):
        return True  # idempotent

    _orig = pymupdf4llm.to_markdown

    def _to_markdown(doc, *args, **kwargs):
        if "use_ocr" not in kwargs and "force_ocr" not in kwargs:
            src = _doc_to_path(doc)
            if src is not None:
                try:
                    if _ocr_decision(src, mode)["choice"] == "off":
                        kwargs["use_ocr"] = False
                except Exception:
                    pass  # err toward keeping OCR
        return _orig(doc, *args, **kwargs)

    _to_markdown._doc2md_ocr_wrapped = True  # type: ignore[attr-defined]
    pymupdf4llm.to_markdown = _to_markdown
    return True


def _patch_docling_ocr(mode: str) -> bool:
    """Make pdfmux's Docling table extraction honor the OCR policy.

    Replaces pdfmux's ``_get_converter`` with a stand-in whose ``convert()``
    picks an OCR-on or OCR-off Docling converter per source PDF (both built once
    and cached, so the only per-call work is the cheap sampled text-layer
    check). Returns True if the patch is in place; False (silently) when Docling
    or pdfmux aren't importable — Docling is an optional extractor.
    """
    try:
        from pdfmux.extractors import tables as _tables
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception:  # pragma: no cover - depends on optional deps
        return False

    cache: Dict[bool, object] = {}
    lock = threading.Lock()

    def _converter_for(do_ocr: bool):
        with lock:
            conv = cache.get(do_ocr)
            if conv is None:
                opts = PdfPipelineOptions()
                opts.do_ocr = do_ocr
                opts.do_table_structure = True
                conv = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
                )
                cache[do_ocr] = conv
            return conv

    class _OcrPolicyConverter:
        """Drop-in for Docling's DocumentConverter that selects an OCR-on or
        OCR-off real converter based on the source PDF handed to ``convert``."""

        def convert(self, source, *args, **kwargs):
            # Decide per source PDF (the report path logs the decision; here we
            # only act on it). Errs toward keeping OCR if classification fails.
            try:
                do_ocr = _ocr_decision(Path(str(source)), mode)["choice"] == "on"
            except Exception:
                do_ocr = True
            return _converter_for(do_ocr).convert(source, *args, **kwargs)

    _singleton = _OcrPolicyConverter()
    _tables._get_converter = lambda: _singleton
    # Some pdfmux versions read the cached singleton directly; keep it coherent.
    _tables._converter_instance = _singleton
    return True


def _pdf_page_count(path: Path) -> int:
    """Page count for a PDF, or 0 if it can't be opened (no PyMuPDF / bad file)."""
    if pymupdf is None:
        return 0
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        return 0
    try:
        return doc.page_count
    finally:
        doc.close()


def resolve_timeout(arg_timeout: Optional[int], max_pages: int) -> int:
    """Resolve the PDFMUX_TIMEOUT (seconds) for this run.

    ``arg_timeout`` is the ``--timeout`` value: ``None`` auto-scales by the
    largest input's page count (``max(TIMEOUT_FLOOR, pages * budget)``), ``0``
    means effectively unlimited, and a positive value is used verbatim. pdfmux's
    300s default is far too low for large docs, so the auto default scales up.
    """
    if arg_timeout is None:
        return max(TIMEOUT_FLOOR, max_pages * TIMEOUT_PER_PAGE_BUDGET)
    if arg_timeout <= 0:
        return TIMEOUT_UNLIMITED
    return arg_timeout


def resolve_quality(quality: str, page_count: int, large_doc_pages: int) -> Tuple[str, Optional[Dict[str, str]]]:
    """Resolve ``--quality`` for one PDF, returning (effective_quality, decision).

    Only ``"auto"`` is resolved here: it picks ``"fast"`` for PDFs larger than
    ``large_doc_pages`` (Docling's per-page cost is not worth it on huge
    born-digital docs) and ``"standard"`` otherwise. The returned ``decision`` is
    an :func:`_note_auto_decision`-shaped dict when the size-based downgrade
    fires (so it's announced + recorded), else ``None``. Explicit qualities
    (fast/standard/high) pass through unchanged with no decision.
    """
    if quality != "auto":
        return quality, None
    if page_count > large_doc_pages:
        est_min = max(1, round(page_count * 0.2 / 60))  # ~0.2s/page on fast
        return "fast", {
            "setting": "quality",
            "choice": "fast",
            "reason": "auto: large PDF (%d pages > %d) — fast avoids ~%dx slower "
            "Docling for little fidelity gain (~%d min est)"
            % (page_count, large_doc_pages, 6, est_min),
            "override": "--quality standard (force Docling tables)",
        }
    return "standard", None


class _MinLevelFilter(logging.Filter):
    """Drop log records below ``min_level``. Attached to a logger, it survives a
    library re-setting its own level on import (which a plain ``setLevel`` would
    not), because logger filters are consulted on every emit regardless of level.
    """

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


def quiet_third_party_logs() -> None:
    """Silence noisy third-party chatter so only doc2md's own output shows.

    Some extraction backends attach their own handlers / progress bars: RapidOCR
    prints colored INFO lines for every model it loads, and Docling/transformers
    emit a ``Loading weights`` tqdm bar. doc2md uses plain ``print`` for its
    messages, so raising third-party log levels to WARNING hides none of ours —
    and WARNING/ERROR still surface real problems. Idempotent; safe to call once
    up front (it works even before the libraries are imported).
    """
    # Model-download / load progress bars ("Loading weights: 100%|...").
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warn_only = _MinLevelFilter(logging.WARNING)
    for name in ("RapidOCR", "rapidocr", "docling", "transformers", "easyocr", "onnxruntime"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.addFilter(warn_only)


def install_ocr_policy(mode: str) -> None:
    """Control OCR across pdfmux's extraction engines per ``mode``.

    ``"on"`` keeps every engine's default (OCR on); ``"off"`` disables OCR for
    every PDF; ``"auto"`` disables OCR for born-digital PDFs (see
    :func:`_pdf_is_text_based`) and keeps it for scanned/image PDFs.

    The OCR work lives in two engines — **pymupdf4llm** (FastExtractor /
    multi-pass) and **Docling** (table extraction) — so both are patched. Never
    raises: an engine that isn't importable in a shape we recognize is simply
    left at its default.
    """
    if mode == "on":
        return
    patched = []
    if _patch_pymupdf4llm_ocr(mode):
        patched.append("pymupdf4llm")
    if _patch_docling_ocr(mode):
        patched.append("docling")
    if not patched:
        print(
            "[info] OCR policy not applied; extractors keep their defaults",
            file=sys.stderr,
        )


def _pdf_routes_to_docling(src: Path) -> bool:
    """True if pdfmux's router would push this PDF through the Docling table path.

    ``process()`` runs Docling's multi-pass / table overlay there; the streaming
    per-page extractor is pymupdf4llm-only and can't reproduce that, so we keep
    such docs on ``process()`` (and forgo per-page markers for them). Uses
    pdfmux's own ``classify().has_tables`` signal. Errs toward ``False`` (allow
    streaming) when classification isn't available — the common digital case.
    """
    try:
        from pdfmux.detect import classify
    except Exception:
        return False
    try:
        return bool(getattr(classify(str(src)), "has_tables", False))
    except Exception:
        return False


def _extract_pages_via_streaming(
    src: Path, quality: str
) -> Optional[Tuple[List[Tuple[int, str]], Optional[float], Optional[float]]]:
    """Per-page text via ``pdfmux.streaming.process_streaming`` (1.7.0+).

    Returns ``(pages, doc_confidence, min_page_confidence)`` with one
    ``(page_index, text)`` per page in document order — gaps filled with empty
    text so page markers stay continuous and figure interleave stays aligned to
    PyMuPDF's 0-based page indices. Returns ``None`` if the streaming API isn't
    importable or yields no pages, so the caller can fall back to ``process()``.

    The fast pass reads each page through the same ``pymupdf4llm.to_markdown``
    that ``process()``'s digital path uses (so doc2md's OCR policy still applies
    via :func:`_patch_pymupdf4llm_ocr`); bad/empty pages are re-extracted with
    OCR exactly as the multi-pass pipeline does, except in ``fast`` quality.
    """
    try:
        from pdfmux.streaming import process_streaming
    except Exception:
        return None
    page_map: Dict[int, Tuple[str, Optional[float]]] = {}
    doc_conf: Optional[float] = None
    page_count: Optional[int] = None
    try:
        for ev in process_streaming(str(src), quality=quality):
            if ev.type == "classified":
                page_count = ev.data.get("page_count")
            elif ev.type == "page":
                d = ev.data
                page_map[int(d["page_num"])] = (d.get("text") or "", d.get("confidence"))
            elif ev.type == "complete":
                doc_conf = ev.data.get("total_confidence")
    except Exception:
        return None
    if not page_map:
        return None
    n = page_count if page_count else max(page_map) + 1
    pages: List[Tuple[int, str]] = []
    confs: List[float] = []
    for i in range(n):
        text, conf = page_map.get(i, ("", None))
        pages.append((i, text))
        if conf is not None:
            confs.append(conf)
    min_conf = min(confs) if confs else doc_conf
    return pages, doc_conf, min_conf


def _extract_pages_via_process(
    src: Path, quality: str
) -> Tuple[List[Tuple[int, str]], Optional[float], Optional[float]]:
    """Single combined-blob extraction via ``pdfmux.pipeline.process`` (legacy).

    The page-marker-free path: ``process()`` returns a ``ConversionResult`` whose
    ``.text`` is one blob (no per-page split as of pdfmux 1.7.0), so markers stay
    dormant. Used for the Docling-table / LLM routes and whenever streaming is
    unavailable. The ``getattr(result, "pages", ...)`` branch is kept dormant so
    real per-page text auto-activates if ``process()`` ever populates ``.pages``.
    Falls back to public ``extract_text`` only if the internal layout changed.
    """
    try:
        from pdfmux.pipeline import process
    except ImportError:
        process = None  # type: ignore

    if process is not None:
        result = process(file_path=str(src), output_format="markdown", quality=quality)
        pages = getattr(result, "pages", None)
        if pages:
            page_text = [(int(getattr(p, "page_num", i)), p.text) for i, p in enumerate(pages)]
            confs = [getattr(p, "confidence", None) for p in pages]
            confs = [c for c in confs if c is not None]
            min_conf = min(confs) if confs else getattr(result, "confidence", None)
            return page_text, getattr(result, "confidence", None), min_conf
        return [(0, result.text)], getattr(result, "confidence", None), getattr(result, "confidence", None)

    import pdfmux

    return [(0, pdfmux.extract_text(str(src), quality=quality))], None, None


def _extract_pdf_pages(
    src: Path, quality: str
) -> Tuple[List[Tuple[int, str]], Optional[float], Optional[float], str]:
    """Run pdfmux → ``(pages, doc_confidence, min_page_confidence, source)``.

    ``pages`` is a list of ``(page_index, text)``. ``source`` records which path
    produced it (consumed by :func:`_page_marker_decision` for provenance):

    * ``"streaming"`` — real per-page text from ``process_streaming``; page
      markers + inline figure interleave activate.
    * ``"process-tables"`` / ``"process-llm"`` / ``"process-fallback"`` — a
      single combined blob from ``process()`` (markers dormant), used when the
      doc routes to Docling tables (``--quality standard`` + detected tables),
      the LLM path (``--quality high``), or streaming is unavailable.

    The Docling/LLM routes stay on ``process()`` because its multi-pass /
    table-overlay output is higher fidelity than streaming's pymupdf4llm-only
    per-page text there. ``--quality fast`` forces the per-page path everywhere
    (``process()`` itself skips Docling re-extraction in fast mode, so there's no
    fidelity to lose). ``quality`` is already resolved (never ``"auto"``) here.
    """
    if quality == "high":
        return (*_extract_pages_via_process(src, quality), "process-llm")
    if quality == "standard" and _pdf_routes_to_docling(src):
        return (*_extract_pages_via_process(src, quality), "process-tables")
    streamed = _extract_pages_via_streaming(src, quality)
    if streamed is not None:
        return (*streamed, "streaming")
    return (*_extract_pages_via_process(src, quality), "process-fallback")


_PAGE_MARKER_OVERRIDE = "--no-rag-metadata (drops all RAG provenance)"


def _page_marker_decision(source: str) -> Optional[Dict[str, str]]:
    """Reportable record (à la :func:`_ocr_decision`) for whether RAG page
    markers were woven, keyed by the :func:`_extract_pdf_pages` ``source``.

    ``None`` when there's nothing notable to record. Callers announce the
    ``"off"`` (suppressed) cases and keep the ``"on"`` default silent.
    """
    if source in ("streaming", "process-pages"):
        return {
            "setting": "page-markers",
            "choice": "on",
            "reason": "per-page text available — page-boundary markers woven in",
            "override": _PAGE_MARKER_OVERRIDE,
        }
    suppressed = {
        "process-tables": (
            "table doc at --quality standard — kept on process() for Docling "
            "table fidelity (single blob, no per-page markers)",
            "--quality fast (forces per-page markers via pymupdf4llm)",
        ),
        "process-llm": (
            "--quality high LLM path — extracted via process() "
            "(single blob, no per-page markers)",
            _PAGE_MARKER_OVERRIDE,
        ),
        "process-fallback": (
            "pdfmux streaming unavailable — fell back to process() "
            "(single blob, no per-page markers)",
            _PAGE_MARKER_OVERRIDE,
        ),
    }
    if source in suppressed:
        reason, override = suppressed[source]
        return {
            "setting": "page-markers",
            "choice": "off",
            "reason": reason,
            "override": override,
        }
    return None


def convert_pdf(
    src: Path,
    dest: Path,
    *,
    quality: str = "standard",
    preview_pages: Optional[int] = None,
    visual_cfg: Optional[VisualConfig] = None,
    clean: bool = True,
    strip_patterns: Optional[List[str]] = None,
    page_markers: bool = False,
) -> PdfConversion:
    """Convert a PDF to Markdown (pdfmux) + extract visuals (PyMuPDF), write ``dest``.

    When ``clean`` (default), the assembled Markdown is de-noised for LLM
    consumption: furniture lines matching safe built-in patterns (bare page
    number / date) or a user ``strip_patterns`` regex are removed, empty
    duplicate tables are dropped, and line-wrap token splits are repaired. The
    pass is character-conservation-checked and falls back to the raw text if
    losslessness can't be proven.
    """
    cfg = visual_cfg or VisualConfig()
    work_src = src
    tmpdir: Optional[tempfile.TemporaryDirectory] = None
    cleaning: Dict[str, object] = {}
    try:
        if preview_pages is not None:
            tmpdir = tempfile.TemporaryDirectory(prefix="doc2md_preview_")
            work_src = _slice_pdf_pages(src, preview_pages, Path(tmpdir.name))

        pages_text, confidence, min_conf, page_source = _extract_pdf_pages(work_src, quality)
        visual_stats: Dict[str, int] = {}
        markdown, visuals = build_markdown_with_visuals(
            pages_text, work_src, dest, cfg, doc_title=src.stem, mark_pages=page_markers,
            stats=visual_stats,
        )
        markers_applied = page_markers and len(pages_text) > 1
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    if clean:
        markdown, cleaning = clean_markdown(markdown, strip_patterns=strip_patterns)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    return PdfConversion(
        markdown=markdown,
        confidence=confidence,
        min_page_confidence=min_conf,
        visuals=visuals,
        cleaning=cleaning,
        page_source=page_source,
        page_markers_applied=markers_applied,
        degenerate_tables=visual_stats.get("degenerate_tables", 0),
    )


def convert_with_pandoc(
    src: Path, dest: Path, *, clean: bool = True, strip_patterns: Optional[List[str]] = None
) -> Tuple[str, Dict[str, object]]:
    """Convert a non-PDF document to GFM Markdown with pandoc.

    Returns ``(markdown, cleaning_stats)``. With ``clean`` (default) the safe
    subset (:func:`clean_safe_subset` — whitespace + ``--strip-line``) is applied
    and the cleaned text re-written to ``dest``.
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
    markdown = dest.read_text(encoding="utf-8")
    cleaning: Dict[str, object] = {}
    if clean:
        markdown, cleaning = clean_safe_subset(markdown, strip_patterns=strip_patterns)
        dest.write_text(markdown, encoding="utf-8")
    return markdown, cleaning


# --------------------------------------------------------------------------- #
# Format detection (content sniffing, not extension)
# --------------------------------------------------------------------------- #

_MIME_HEADER_RE = re.compile(
    r"(?is)^\s*(from|date|message-id|subject|mime-version|content-type)\s*:"
)


def detect_format(path: Path) -> str:
    """Classify a file by *content*, returning one of:

    ``pdf``, ``docx``, ``mhtml``, ``html``, ``doc-binary``, ``xml``, ``pandoc``,
    ``unsupported``. Extensions lie (Confluence "Word" is MHTML named ``.doc``),
    so we sniff magic bytes / structure first and fall back to the extension.
    """
    try:
        head = path.read_bytes()[:4096]
    except OSError:
        return "unsupported"

    if head[:4] == b"%PDF":
        return "pdf"
    if head[:4] == b"PK\x03\x04":  # zip container
        try:
            import zipfile

            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
            if "word/document.xml" in names:
                return "docx"
            if "mimetype" in names or any(n.startswith("OEBPS/") for n in names):
                return "pandoc"  # epub / odt -> pandoc
        except Exception:
            return "unsupported"
        return "unsupported"
    if head[:4] == b"\xd0\xcf\x11\xe0":  # OLE2 -> legacy binary .doc/.xls
        return "doc-binary"

    text = head.decode("utf-8", "replace").lstrip("﻿").lstrip()
    low = text.lower()
    if text.startswith("<?xml") or path.suffix.lower() in XML_EXTENSIONS:
        # XHTML is rare here; treat .xml / xml-declared content as XML.
        if not low.startswith("<html") and "<html" not in low[:200]:
            return "xml"
    if _MIME_HEADER_RE.match(text) and ("multipart/related" in low or "mime-version" in low):
        return "mhtml"
    if path.suffix.lower() in (".mht", ".mhtml"):
        return "mhtml"
    # Bare single-file HTML, sniffed by content so the extension can't hide it:
    # Jira/Confluence "Save as Word" frequently emit a plain HTML page named
    # ``.doc`` (or ``.htm``). MHTML is matched first above because it, too,
    # contains ``<html>``. Match ``<html``/``<!doctype html`` only as the opening
    # token (after an optional XML prolog or comments) so prose that merely
    # mentions ``<html>`` isn't misread. Routes to the de-chroming Word/HTML path.
    probe = re.sub(r"^\s*(?:<\?xml[^>]*\?>\s*|<!--.*?-->\s*)+", "", low, flags=re.S)
    if probe.startswith("<!doctype html") or probe.startswith("<html"):
        return "html"
    if path.suffix.lower() in PANDOC_EXTENSIONS:
        return "pandoc"
    if path.suffix.lower() == ".docx":
        return "docx"
    if path.suffix.lower() == ".csv":
        return "csv"
    return "unsupported"


# --------------------------------------------------------------------------- #
# Word / MHTML conversion (Confluence "Word" export is MHTML with base64 images)
# --------------------------------------------------------------------------- #


@dataclass
class WordConversion:
    markdown: str
    images: int
    ref_plaintext: Optional[str]  # source text for the fidelity check
    cleaning: Dict[str, object] = field(default_factory=dict)


@dataclass
class CsvConversion:
    """Return value of :func:`convert_csv`."""

    cards: int
    rows: int
    output_paths: List[Path]
    warnings: List[str] = field(default_factory=list)


def _sniff_image(data: bytes) -> Optional[str]:
    """Return a file extension for raster/vector image bytes, or None."""
    if data[:8].startswith(b"\x89PNG"):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:3] == b"GIF":
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] == b"\x01\x00\x00\x00" or data[40:44] == b" EMF":
        return "emf"
    if data[:4] == b"\xd7\xcd\xc6\x9a" or data[:4] == b"\x01\x00\x09\x00":
        return "wmf"
    return None


def _image_dims(data: bytes, fmt: str) -> Optional[Tuple[int, int]]:
    """Best-effort (width, height) from raster image bytes; None if unknown."""
    try:
        if fmt == "png":
            return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if fmt == "gif":
            return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
        if fmt == "bmp":
            return int.from_bytes(data[18:22], "little"), int.from_bytes(data[22:26], "little")
        if fmt == "jpg":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    return (
                        int.from_bytes(data[i + 7:i + 9], "big"),
                        int.from_bytes(data[i + 5:i + 7], "big"),
                    )
                i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    except Exception:
        pass
    return None


def _image_is_content(data: bytes, fmt: str) -> bool:
    """Keep real figures, drop tiny UI icons/badges.

    Confluence chrome (emoticons, status badges, expand arrows) is small in both
    byte size and dimensions, while real screenshots/diagrams are large. Byte
    size is the most reliable discriminator (a simple PNG can be ≥64px yet only
    a few KB), backed by a pixel-dimension floor.
    """
    if len(data) < WORD_IMAGE_MIN_BYTES:
        return False
    dims = _image_dims(data, fmt)
    return dims is None or min(dims) >= WORD_IMAGE_MIN_PX


def extract_mhtml(path: Path) -> Tuple[str, List[Tuple[str, bytes]]]:
    """Parse an MHTML file into ``(html, [(part_key, bytes), ...])``.

    ``part_key`` is the Content-Location basename (or Content-ID) — note it can
    be a *prefix* of the ``<img src>`` hash, so callers resolve by prefix.
    """
    import email

    msg = email.message_from_bytes(path.read_bytes())
    html = ""
    images: List[Tuple[str, bytes]] = []
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype == "text/html" and not html:
            html = part.get_payload(decode=True).decode(
                part.get_content_charset() or "utf-8", "replace"
            )
        elif part.get_content_maintype() in ("application", "image"):
            data = part.get_payload(decode=True)
            if not data:
                continue
            ref = part.get("Content-Location", "") or part.get("Content-ID", "")
            key = ref.split("/")[-1].strip("<>")
            images.append((key, data))
    return html, images


def _resolve_img_src(src: str, key_to_rel: Dict[str, str]) -> Optional[str]:
    """Match an <img> src against part keys by equality or prefix."""
    for key, rel in key_to_rel.items():
        if src == key or src.startswith(key) or key.startswith(src):
            return rel
    return None


def _rewrite_img_srcs(html: str, key_to_meta: Dict[str, Tuple[str, str]]) -> str:
    """Replace matched <img> tags with a *clean* ``<img src alt>`` pointing at the
    extracted figure; drop unresolved ones.

    Confluence adds ``class``/``draggable``/``height`` attributes, and pandoc
    keeps an attribute-rich ``<img>`` as raw HTML instead of emitting Markdown
    ``![]()``. Reducing it to ``src`` + ``alt`` makes pandoc produce a proper
    Markdown image with the document title woven into the alt text.
    """

    def repl(m: "re.Match") -> str:
        sm = re.search(r'src\s*=\s*["\']?([^"\'> ]+)', m.group(0), re.I)
        if not sm:
            return ""
        meta = _resolve_img_src(sm.group(1), key_to_meta)
        if meta is None:
            return ""  # filtered icon / unresolved -> remove tag
        rel, alt = meta
        return '<img src="%s" alt="%s" />' % (rel, alt.replace('"', "'"))

    return re.sub(r"<img\b[^>]*>", repl, html, flags=re.I)


_DATA_URI_IMG_RE = re.compile(
    r"""<img\b[^>]*?\bsrc\s*=\s*["'](data:image/[^;"']+;base64,([^"']+))["'][^>]*>""",
    re.I,
)


def _extract_data_uri_images(
    html: str, fig_dir: Path, slug: str, rel_base: str, title: str
) -> Tuple[str, int]:
    """Pull base64 ``data:`` ``<img>`` blobs out of single-file HTML into figure
    files and rewrite each tag to a slim ``<img src alt>`` reference.

    Single-file HTML exports (Jira/Confluence "Save as Word") embed content
    images inline as ``data:image/...;base64`` URIs. Decoding the content-sized
    ones to PNG/JPG files (UI icons/badges are dropped by the same
    :func:`_image_is_content` floor used for MHTML) keeps real figures as proper
    Markdown images instead of letting the cleaner discard them as decorative
    ``data:`` blobs. Returns ``(html, n_extracted)``.
    """
    import base64

    state = {"i": 0}

    def repl(m: "re.Match") -> str:
        try:
            data = base64.b64decode(m.group(2), validate=False)
        except Exception:
            return m.group(0)  # malformed -> leave for the chrome cleaner
        ifmt = _sniff_image(data)
        if ifmt is None or ifmt in ("emf", "wmf"):
            return m.group(0)
        if not _image_is_content(data, ifmt):
            return ""  # decorative icon/badge -> drop the tag
        state["i"] += 1
        fig_dir.mkdir(parents=True, exist_ok=True)
        fname = "%s-figure%02d.%s" % (slug, state["i"], ifmt)
        (fig_dir / fname).write_bytes(data)
        alt = "%s — figure %d" % (title, state["i"])
        return '<img src="%s/%s" alt="%s" />' % (rel_base, fname, alt.replace('"', "'"))

    return _DATA_URI_IMG_RE.sub(repl, html), state["i"]


# Site chrome that single-page-app exports (Apple/Confluence MHTML) render into
# the DOM but that carries no document content: global nav, breadcrumbs, footer,
# sidebars, and script/style. Pandoc drops these from the Markdown, so the
# fidelity reference must drop them too or it penalizes a faithful conversion.
_NONCONTENT_REGION_RE = re.compile(
    r"(?is)<(script|style|nav|header|footer|aside|template|noscript)\b[^>]*>.*?</\1>"
)


def _strip_noncontent_regions(html: str) -> str:
    """Remove navigational/boilerplate regions before plaintext extraction."""
    return _NONCONTENT_REGION_RE.sub(" ", html)


def _strip_html(html: str) -> str:
    """Crude HTML -> plain text for the fidelity comparison (no deps)."""
    import html as _h

    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _h.unescape(text)


def html_to_markdown(html: str, have_pandoc: bool) -> str:
    """Convert an HTML string to GFM via pandoc (stdin)."""
    if not have_pandoc:
        raise RuntimeError(
            "pandoc is required to convert Word/HTML content to Markdown; "
            "install pandoc (see requirements.txt)."
        )
    proc = subprocess.run(
        ["pandoc", "-f", "html", "-t", "gfm", "--wrap=none"],
        input=html,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("pandoc html->md failed: %s" % proc.stderr.strip())
    return proc.stdout


def _strip_pandoc_html_chrome(markdown: str, *, count: bool = False):
    """Remove non-content HTML that pandoc emits or passes through when
    converting single-page-app exports (Apple/Confluence MHTML, raw HTML).

    With ``count=True`` returns ``(markdown, n_data_uri_imgs, n_tags_unwrapped)``
    for cleanup stats; otherwise returns just the cleaned ``markdown``.

    Two kinds of noise bloat the output and tank the fidelity score (they are
    absent from the plaintext reference, so ``difflib`` penalizes them):

    - ``<img src="data:...">`` blobs — pandoc re-encodes inline ``<svg>`` icons as
      base64 data URIs. They are decorative and carry no text. Extracted figures
      reference real file paths, so only ``data:`` sources are dropped.
    - Empty ``<div>``/``<span>`` scaffolding (Vue ``data-v-*`` wrappers, breadcrumb
      and related-topic chrome) left as raw passthrough HTML. Tags are removed but
      any inner text is kept, so content survives the unwrap.
    """
    # Drop inline-svg / data-URI images; keep real extracted-figure references.
    markdown, n_imgs = re.subn(
        r"""<img\b[^>]*\bsrc\s*=\s*["']data:[^"']*["'][^>]*>""", "", markdown, flags=re.I
    )
    # Unwrap scaffolding tags (keep inner text); div/span carry no Markdown meaning.
    markdown, n_tags = re.subn(r"</?(?:div|span)\b[^>]*>", "", markdown, flags=re.I)
    # Collapse the blank-line runs and trailing spaces the removals leave behind.
    markdown = re.sub(r"[ \t]+$", "", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return (markdown, n_imgs, n_tags) if count else markdown


def convert_word(
    src: Path,
    dest: Path,
    fmt: str,
    *,
    have_pandoc: bool,
    visual_cfg: Optional[VisualConfig] = None,
    clean: bool = True,
    strip_patterns: Optional[List[str]] = None,
) -> WordConversion:
    """Convert a Word-family document: OOXML ``.docx``, Confluence MHTML, or a
    bare single-file ``html`` page (a Jira/Confluence "Save as Word" export,
    often misnamed ``.doc``).

    When ``clean`` (default), MHTML/HTML output is de-chromed via
    :func:`clean_html_markdown` (visible-text-lossless) and ``.docx`` output gets
    the :func:`clean_safe_subset` (whitespace + ``--strip-line``). ``--no-clean``
    yields the raw pandoc conversion.
    """
    cfg = visual_cfg or VisualConfig()
    slug = dest.stem
    fig_dir = dest.parent / slug / "figures"
    rel_base = "%s/figures" % slug
    dest.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "mhtml":
        html, parts = extract_mhtml(src)
        ref_plaintext = _strip_html(_strip_noncontent_regions(html))
        key_to_meta: Dict[str, Tuple[str, str]] = {}
        n = 0
        if cfg.enabled and cfg.extract_images:
            idx = 0
            for key, data in parts:
                ifmt = _sniff_image(data)
                if ifmt is None or ifmt in ("emf", "wmf"):
                    continue  # unknown or unrenderable vector -> skip
                if not _image_is_content(data, ifmt):
                    continue  # tiny icon/badge
                idx += 1
                fig_dir.mkdir(parents=True, exist_ok=True)
                fname = "%s-figure%02d.%s" % (slug, idx, ifmt)
                (fig_dir / fname).write_bytes(data)
                alt = "%s — figure %d" % (src.stem, idx)
                key_to_meta[key] = ("%s/%s" % (rel_base, fname), alt)
                n += 1
        html = _rewrite_img_srcs(html, key_to_meta)
        markdown = html_to_markdown(html, have_pandoc)
        cleaning: Dict[str, object] = {}
        if clean:
            markdown, cleaning = clean_html_markdown(markdown, strip_patterns=strip_patterns)
        dest.write_text(markdown, encoding="utf-8")
        return WordConversion(
            markdown=markdown, images=n, ref_plaintext=ref_plaintext, cleaning=cleaning
        )

    if fmt == "html":
        # Bare single-file HTML (e.g. a Jira/Confluence "Save as Word" page named
        # ``.doc``). Same pipeline as the MHTML branch minus the MIME unwrapping:
        # extract inline ``data:`` figures, convert via pandoc, then de-chrome.
        html = src.read_text(encoding="utf-8", errors="replace")
        ref_plaintext = _strip_html(_strip_noncontent_regions(html))
        n = 0
        if cfg.enabled and cfg.extract_images:
            html, n = _extract_data_uri_images(html, fig_dir, slug, rel_base, src.stem)
        markdown = html_to_markdown(html, have_pandoc)
        cleaning = {}
        if clean:
            markdown, cleaning = clean_html_markdown(markdown, strip_patterns=strip_patterns)
        dest.write_text(markdown, encoding="utf-8")
        return WordConversion(
            markdown=markdown, images=n, ref_plaintext=ref_plaintext, cleaning=cleaning
        )

    if fmt == "docx":
        if not have_pandoc:
            raise RuntimeError("pandoc is required to convert .docx; see requirements.txt.")
        args = ["pandoc", "-f", "docx", "-t", "gfm", "--wrap=none"]
        media_root = dest.parent / slug
        if cfg.enabled and cfg.extract_images:
            args += ["--extract-media", str(media_root)]
        args.append(str(src))
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("pandoc docx->md failed: %s" % proc.stderr.strip())
        markdown = proc.stdout
        media_dir = media_root / "media"
        n = len(list(media_dir.glob("*"))) if media_dir.is_dir() else 0
        ref = subprocess.run(
            ["pandoc", "-f", "docx", "-t", "plain", "--wrap=none", str(src)],
            capture_output=True,
            text=True,
        )
        ref_plaintext = ref.stdout if ref.returncode == 0 else None
        cleaning = {}
        if clean:
            markdown, cleaning = clean_safe_subset(markdown, strip_patterns=strip_patterns)
        dest.write_text(markdown, encoding="utf-8")
        return WordConversion(
            markdown=markdown, images=n, ref_plaintext=ref_plaintext, cleaning=cleaning
        )

    raise RuntimeError("unhandled Word format: %s" % fmt)


# --------------------------------------------------------------------------- #
# XML conversion — verbatim (config, syntax-critical) or transform (doc)
# --------------------------------------------------------------------------- #

_XML_INLINE_TAGS = {"b", "i", "em", "strong", "br", "link", "a", "code", "tt", "u", "sub", "sup"}


def _xml_local(elem) -> str:
    return elem.tag.split("}")[-1] if isinstance(elem.tag, str) else "_"


def _xml_attrs(elem) -> str:
    return " ".join('%s="%s"' % (k.split("}")[-1], v) for k, v in elem.attrib.items())


def xml_choose_mode(path: Path, raw: str) -> str:
    """Auto-pick 'transform' (documentation) vs 'verbatim' (config)."""
    import xml.etree.ElementTree as ET
    from collections import Counter

    try:
        root = ET.fromstring(raw)
    except Exception:
        return "verbatim"
    tags: "Counter" = Counter(_xml_local(e) for e in root.iter())
    total = sum(tags.values())
    prose = sum(tags[t] for t in ("doc", "p", "li", "dt", "dd", "b", "br", "para", "section"))
    return "transform" if total and prose / total >= XML_PROSE_TAG_RATIO else "verbatim"


def xml_to_markdown_verbatim(path: Path, raw: str) -> str:
    """Lossless: the XML is preserved exactly inside a fenced block, with a
    generated index of top-level elements for navigation. For syntax-critical
    configuration where exact tags/attributes/values matter."""
    import xml.etree.ElementTree as ET

    index: List[str] = []
    try:
        root = ET.fromstring(raw)
        for child in list(root)[:300]:
            tag = _xml_local(child)
            label = child.get("name") or child.findtext("name") or child.get("id") or ""
            index.append("- `%s`%s" % (tag, " — %s" % label if label else ""))
    except Exception:
        pass

    out = ["# %s" % path.stem, "", "_Configuration XML — preserved verbatim below._", ""]
    if index:
        out += ["## Index", ""] + index + [""]
    out += ["## Source", "", "```xml", raw.rstrip(), "```", ""]
    return "\n".join(out)


def _xml_inline_text(elem) -> str:
    """Serialize an element's mixed content with light Markdown formatting."""
    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        name = _xml_local(child)
        inner = _xml_inline_text(child)
        if name in ("b", "strong"):
            parts.append("**%s**" % inner if inner else "")
        elif name in ("i", "em"):
            parts.append("*%s*" % inner if inner else "")
        elif name == "br":
            parts.append("  \n")
        elif name in ("link", "a"):
            href = child.get("href") or child.get("id") or ""
            parts.append("[%s](%s)" % (inner or href, href) if href else inner)
        elif name in ("code", "tt"):
            parts.append("`%s`" % inner if inner else "")
        else:
            parts.append(inner)
        if child.tail:
            parts.append(child.tail)
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def _xml_is_container(elem) -> bool:
    return any(_xml_local(c) not in _XML_INLINE_TAGS for c in elem)


def _render_xml_node(elem, lines: List[str], depth: int) -> None:
    name = _xml_local(elem)
    attrs = _xml_attrs(elem)
    if not _xml_is_container(elem):
        text = _xml_inline_text(elem)
        label = "**%s**%s" % (name, " (%s)" % attrs if attrs else "")
        if text:
            lines.append("- %s: %s" % (label, text))
        elif attrs:
            lines.append("- %s" % label)
        return
    heading = "#" * min(depth + 1, 6)
    lines += ["", "%s %s%s" % (heading, name, " (%s)" % attrs if attrs else ""), ""]
    if elem.text and elem.text.strip():
        lines.append(elem.text.strip())
    for child in elem:
        if _xml_local(child) in _XML_INLINE_TAGS:
            inline = _xml_inline_text(child)
            if inline:
                lines.append(inline)
        else:
            _render_xml_node(child, lines, depth + 1)
        if child.tail and child.tail.strip():
            lines.append(child.tail.strip())


def xml_to_markdown_transform(path: Path, raw: str) -> str:
    """Render documentation-style XML as navigable Markdown (headings/lists)."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(raw)
    lines: List[str] = ["# %s" % path.stem, ""]
    _render_xml_node(root, lines, depth=1)
    return "\n".join(lines)


# Tokenizes XML into comments / CDATA / DOCTYPE / PI / end / start-or-selfclose.
# Start/self-close alternative is quote-aware so a ``>`` inside an attribute value
# (legal in XML) doesn't end the tag early.
_XML_TOKEN_RE = re.compile(
    r"<!--.*?-->"
    r"|<!\[CDATA\[.*?\]\]>"
    r"|<!DOCTYPE(?:[^<>\"']|\"[^\"]*\"|'[^']*')*>"
    r"|<\?.*?\?>"
    r"|</[^>]*>"
    r"|<(?:[^<>\"']|\"[^\"]*\"|'[^']*')*>",
    re.S,
)


def _xml_comments_to_elements(raw: str, tag: str = "_comment") -> str:
    """Rewrite XML comments as ``<_comment>`` child elements of the block they
    annotate, so a comment survives ``xmltodict``'s by-key grouping *and* keeps
    its position.

    A comment is moved inside the element that immediately follows it (the block
    it describes); a self-closing target is expanded so the comment can live
    inside it; a comment sitting just before a closing tag stays in place as the
    last child of the enclosing element (e.g. an ``END`` marker). Consecutive
    comments attach as a list to the same block. Comments inside CDATA are left
    untouched. Text is XML-escaped, and newlines are preserved (element content,
    unlike an attribute, is not whitespace-normalized).

    This is a surgical string transform — every non-comment byte is preserved, so
    namespace prefixes and formatting are untouched.
    """
    def esc(t: str) -> str:
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def cel(text: str) -> str:
        return "<%s>%s</%s>" % (tag, esc(text.strip()), tag)

    toks: List[List[str]] = []
    pos = 0
    for m in _XML_TOKEN_RE.finditer(raw):
        t = m.group(0)
        gap = raw[pos:m.start()]
        pos = m.end()
        if t.startswith("<!--"):
            kind = "comment"
        elif t.startswith("</"):
            kind = "end"
        elif t.endswith("/>"):
            kind = "selfclose"
        elif t.startswith("<!") or t.startswith("<?"):
            kind = "other"
        else:
            kind = "start"
        toks.append([kind, t, gap])
    trailing = raw[pos:]

    out: List[str] = []
    i, n = 0, len(toks)
    while i < n:
        kind, tok, gap = toks[i]
        if kind != "comment":
            out.append(gap)
            out.append(tok)
            i += 1
            continue
        out.append(gap)
        # collect a run of consecutive comments (only whitespace between them)
        comments: List[str] = []
        j = i
        while j < n and toks[j][0] == "comment" and (j == i or toks[j][2].strip() == ""):
            comments.append(re.match(r"<!--(.*)-->$", toks[j][1], re.S).group(1))
            j += 1
        nxt = toks[j] if j < n else None
        if nxt and nxt[0] == "start" and nxt[2].strip() == "":
            out.append(nxt[2])  # indentation before the start tag
            out.append(nxt[1])
            out.extend(cel(c) for c in comments)
            i = j + 1
        elif nxt and nxt[0] == "selfclose" and nxt[2].strip() == "":
            name = re.match(r"<\s*([A-Za-z_][\w.\-:]*)", nxt[1]).group(1)
            out.append(nxt[2])
            out.append(nxt[1][:-2].rstrip() + ">")  # <x .../> -> <x ...>
            out.extend(cel(c) for c in comments)
            out.append("</%s>" % name)
            i = j + 1
        else:  # trailing comment (before a close tag / EOF): keep in place
            for off, c in enumerate(comments):
                if off > 0:
                    out.append(toks[i + off][2])
                out.append(cel(c))
            i = j
    out.append(trailing)
    return "".join(out)


_YAML_DISC_KEYS = ("'@value'", "'@name'", "'@result'", "'@field'", "name")


def _yaml_breadcrumbs(text: str, min_depth: int = 2, sep: str = " > ") -> str:
    """Insert ``# path: a > b > c`` breadcrumb comments into dumped YAML.

    RAG ingesters (NotebookLM and friends) retrieve by embedding similarity over
    *chunks*, not by reading the whole document — so a deeply-nested block like
    ``forward:cache-parent`` is retrieved stripped of its ancestor keys and the
    model can't tell what it is or where it lives (it "looks in the wrong place").
    Prepending each block with its structural path restores that context inside
    whatever chunk the block lands in, and the path text adds natural-language
    tokens that bridge the prose-query ↔ config-key vocabulary gap.

    Each path segment carries the node's discriminating attribute
    (``@value``/``@name``/``@result``/``@field``/``name``) when present, because
    config XML repeats keys constantly (``match:request.type`` etc.) and a bare
    key-path would collide across the document.

    Pure structural derivation: no free-text is read from the data, so it never
    picks up junk (unlike comment-derived labels). Breadcrumbs are YAML comments,
    dropped on parse — so structural/fidelity round-tripping is unaffected.

    Operates on the PyYAML text dumped by :func:`xml_to_yaml` (2-space indent).
    PyYAML dumps list items at the SAME indent as their owning key with content
    at +2; the list owner frame holds a discriminator that resets on each ``-``
    so successive items don't leak each other's identity.
    """
    lines = text.split("\n")
    # indent of the next non-blank line, used to tell a block from a leaf scalar
    nb_next: List[Optional[int]] = [None] * len(lines)
    nxt: Optional[int] = None
    for i in range(len(lines) - 1, -1, -1):
        nb_next[i] = nxt
        if lines[i].strip():
            nxt = i

    def _indent(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    stack: List[Dict[str, Any]] = []  # frames: col, label, disc, kind(map|list)

    def path_str() -> str:
        segs = []
        for f in stack:
            seg = f["label"] + ("(%s)" % f["disc"] if f["disc"] else "")
            segs.append(seg)
        return sep.join(segs)

    out: List[str] = []
    # Index up to which lines are the verbatim body of a literal/folded block
    # scalar (``|``/``>``) already emitted by the opener below. Their interior is
    # arbitrary text — a line like ``Possible values are:`` must NOT be mistaken
    # for a mapping key, or a breadcrumb would be injected into the string value
    # and corrupt the round-trip. So we pass block-scalar bodies through untouched.
    skip_to = -1
    for i, line in enumerate(lines):
        if i < skip_to:
            out.append(line)
            continue
        if not line.strip():
            out.append(line)
            continue

        col = _indent(line)
        body = line[col:]
        is_dash = body.startswith("- ")
        content = body[2:] if is_dash else body
        kcol = col + 2 if is_dash else col  # column the key text starts at

        if is_dash:
            # Pop frames deeper than this list level and any prior item's
            # leftover map frames; stop at the owning list frame (col == col).
            while stack and (stack[-1]["col"] > col or
                             (stack[-1]["col"] == col and stack[-1]["kind"] != "list")):
                stack.pop()
            if stack and stack[-1]["col"] == col and stack[-1]["kind"] == "list":
                stack[-1]["disc"] = None  # new item -> reset owner discriminator
        else:
            while stack and stack[-1]["col"] >= kcol:
                stack.pop()

        m = re.match(r"^(.*?):(?:\s+(.*))?$", content)
        key = m.group(1) if m else None
        val = m.group(2) if m else None

        # A block-scalar opener introduces literal text whose body is indented
        # deeper than this line. It appears either as ``key: |`` (mapping value) or
        # as a bare ``- |-`` list item; in both forms the style indicator is the
        # only token. Emit the opener, then pass its whole body through verbatim.
        if key is None:
            scalar = content
        elif val is not None:
            scalar = val
        else:
            scalar = None
        if scalar is not None and re.match(r"^[|>][0-9+-]*\s*$", scalar):
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > col):
                j += 1
            skip_to = j
            out.append(line)
            continue

        # Block detection: empty inline value AND the next line is deeper, or a
        # dash at the same column (PyYAML lists sit at the owning key's indent).
        opens_block = is_list = False
        if key is not None and (val is None or val == ""):
            j = nb_next[i]
            if j is not None:
                jc = _indent(lines[j])
                jdash = lines[j][jc:].startswith("- ")
                if jc > kcol:
                    opens_block = True
                elif jdash and jc == kcol:
                    opens_block = is_list = True

        # Discriminator capture: an inline @value/@name/etc. scalar describes the
        # frame it sits in (current top — a map block, or a list owner mid-item).
        if key in _YAML_DISC_KEYS and val not in (None, "") and stack \
                and not stack[-1]["disc"]:
            stack[-1]["disc"] = "%s=%s" % (key.strip("'"), val.strip("'"))

        if opens_block and key is not None:
            if len(stack) + 1 >= min_depth:
                bc = path_str()
                full = bc + (sep if bc else "") + key
                out.append("%s# path: %s" % (" " * col, full))
            stack.append({"col": kcol, "label": key, "disc": None,
                          "kind": "list" if is_list else "map"})

        out.append(line)

    return "\n".join(out)


def xml_to_yaml(raw: str, index: bool = False) -> str:
    """Convert XML to YAML (structure-preserving, low-token, no Markdown wrapper).

    Uses ``xmltodict`` to build a dict — attributes become ``@name`` keys, element
    text ``#text``, and repeated siblings collapse to lists — then dumps YAML with
    document order preserved. Intended for configuration XML headed for LLM/RAG
    ingestion (e.g. NotebookLM, which won't parse XML fenced inside Markdown).

    XML comments are preserved and *positioned* — important in config XML, where a
    comment often carries the *why* (e.g. a Jira reference) for the block beneath
    it. :func:`_xml_comments_to_elements` rewrites each comment into a ``_comment``
    child of the block it annotates before parsing, so it stays attached through
    ``xmltodict``'s by-key grouping. Processing instructions and the DOCTYPE are
    still dropped. Mixed prose+inline-tag content is awkward, so this is *not* used
    for documentation-style XML.

    When ``index`` is set (``--yaml-index``), each nested block is prefixed with a
    ``# path: ...`` structural-path breadcrumb (see :func:`_yaml_breadcrumbs`) to
    make deeply-nested blocks locatable by RAG retrieval.

    Raises ``RuntimeError`` if the optional ``xmltodict``/``PyYAML`` packages are
    not installed; propagates ``expat.ExpatError`` on malformed XML so callers can
    fall back to a lossless Markdown rendering.
    """
    try:
        import xmltodict
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "--yaml needs the 'xmltodict' and 'PyYAML' packages; "
            "install them with:  pip install -r requirements.txt"
        ) from exc

    # Render multiline strings (large comment blocks, multiline text) as literal
    # block scalars (``|``) instead of escaped double-quoted blobs — far more
    # readable and lower-token. PyYAML falls back to a quoted style on its own for
    # strings a literal block can't represent (e.g. trailing whitespace), so this
    # never breaks round-tripping. Single-line scalars are untouched.
    class _BlockDumper(yaml.Dumper):
        pass

    def _str_rep(dumper, value):
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    _BlockDumper.add_representer(str, _str_rep)

    parsed = xmltodict.parse(_xml_comments_to_elements(raw))
    # width=inf disables PyYAML's default 80-column line wrapping. Long *plain*
    # scalars (e.g. space-separated IP/CIDR lists in config XML) would otherwise
    # wrap into multi-line plain scalars — valid YAML 1.2, but strict/lightweight
    # parsers (notably NotebookLM's RAG ingester) choke on the continuation lines
    # and silently truncate the rest of the document. Keeping every scalar on one
    # line avoids that failure mode at the cost of some long lines.
    dumped = yaml.dump(
        parsed, Dumper=_BlockDumper, default_flow_style=False, sort_keys=False,
        allow_unicode=True, width=float("inf"),
    )
    # --yaml-index: prepend each nested block with a structural-path breadcrumb so
    # RAG retrieval can locate it. Comments don't affect parsing, so output is the
    # plain YAML plus comment lines (clean diff against a non-indexed run).
    return _yaml_breadcrumbs(dumped) if index else dumped


# --------------------------------------------------------------------------- #
# XML -> RAG-optimized flat index (--rag)
# --------------------------------------------------------------------------- #
#
# --yaml/--yaml-index emit *valid, round-trippable* YAML; great for fidelity, but
# a chunk-based RAG ingester (NotebookLM) splits a large config into fixed-size
# windows and a deeply-nested leaf gets stripped of its ancestor keys — it "looks
# in the wrong place." --yaml-index restores context with `# path:` *comments*,
# but a comment only anchors a block's top: if the window splits the block, every
# later leaf is re-orphaned, and embedders down-weight comment tokens anyway.
#
# --rag instead emits one fully path-qualified `a > b > c = value` line per leaf:
# the complete ancestor path is *content*, not a comment, so every line stands
# alone regardless of where a chunk boundary falls, and the path tokens
# embed/keyword-match. A deterministic DOCUMENT SUMMARY (origins + variables) is
# prepended for facts that need whole-document aggregation to answer. This is a
# *projection*, NOT valid YAML and NOT round-trippable — use --yaml for that.
# Fidelity is instead checked at the value level (every source leaf present); see
# :func:`rag_fidelity_check`.

# Discriminator keys as they appear in the xmltodict-parsed dict (raw/unquoted) —
# the same set :func:`_yaml_breadcrumbs` matches in dumped YAML (_YAML_DISC_KEYS).
_RAG_DISC_KEYS = ("@value", "@name", "@result", "@field", "name")
_RAG_SEP = " > "


def _rag_discriminator(node: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(key, "key=value")`` discriminator for a mapping node, else ``(None, None)``.

    Mirrors the discriminator capture in :func:`_yaml_breadcrumbs`: a node is
    identified by its first present ``@value``/``@name``/``@result``/``@field``/
    ``name`` scalar so repeated keys (``match:request.type`` …) don't collide
    along a path. Booleans are skipped (they aren't config identities).
    """
    if not isinstance(node, dict):
        return None, None
    for k in _RAG_DISC_KEYS:
        v = node.get(k)
        if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v) != "":
            return k, "%s=%s" % (k.lstrip("@"), v)
    return None, None


def _rag_leaf_value(node: Any) -> str:
    """One-line rendering of a leaf scalar: internal whitespace/newlines collapsed
    so every emitted line stays a single self-contained record (a rare multi-line
    comment value would otherwise span lines and break per-leaf grounding)."""
    if node is None:
        return ""
    return " ".join(str(node).split())


def _rag_text(node: Any) -> Any:
    """The element's own scalar text, unwrapping ``xmltodict``'s mixed-content
    node. An element that has text *and* children (here always a positioned
    ``_comment``, e.g. ``<value><_comment>2^19</_comment>524288</value>``) parses
    to ``{'#text': '524288', '_comment': '2^19'}``; the bare value is its
    ``#text``. Returns ``node`` unchanged when it isn't such a mapping, so callers
    can blindly pass any leaf/value through. Prevents the raw Python ``dict``
    repr from leaking into a value position (the summary's variable values and any
    discriminator value)."""
    if isinstance(node, dict) and "#text" in node:
        return node["#text"]
    return node


def _rag_flatten(node: Any, path: List[str], out: List[str]) -> None:
    """Append one ``a > b > c = value`` line per leaf scalar under ``node``.

    Lists share their owning key's path; each item re-derives its own
    discriminator so siblings don't blur together. The key chosen as a node's
    discriminator is folded into the path segment and not re-emitted as its own
    leaf (that would just duplicate the segment).
    """
    if isinstance(node, dict):
        dkey, dsuf = _rag_discriminator(node)
        if path:
            seg = path[-1] + ("(%s)" % dsuf if dsuf else "")
            here = path[:-1] + [seg]
        else:
            here = path
        before = len(out)
        if "#text" in node and here:
            # Mixed content: the element has its own text *plus* children (always a
            # positioned ``_comment`` here). The text is this element's value — emit
            # it on the element's own segment, not as a spurious ``> #text`` child.
            out.append("%s = %s" % (_RAG_SEP.join(here), _rag_leaf_value(node["#text"])))
        for k, v in node.items():
            if k == dkey or k == "#text":
                continue  # discriminator folds into the segment; #text handled above
            _rag_flatten(v, here + [k], out)
        if dkey is not None and len(out) == before:
            # The discriminator was this node's ONLY content (e.g. a bare
            # ``comment:note`` ``@value``); with no sibling leaf to carry it, emit
            # it directly so the value isn't lost (rag_fidelity_check enforces this).
            out.append("%s = %s" % (_RAG_SEP.join(here), _rag_leaf_value(node[dkey])))
    elif isinstance(node, list):
        for item in node:
            _rag_flatten(item, path, out)
    else:
        out.append("%s = %s" % (_RAG_SEP.join(path), _rag_leaf_value(node)))


def _rag_manifest(parsed: Any) -> str:
    """Deterministic DOCUMENT SUMMARY: facts that need whole-document aggregation.

    Scoped to exactly that — origin hostnames (gathered from every forward block)
    and ``assign:variable`` name→value definitions (scattered across the config).
    Local, single-location facts (ports, individual match rules) are already
    answered by the flattened leaves, so duplicating them here would only bloat
    and drift. Conditional *logic* (failover triggers) is intentionally deferred
    to a future opt-in ``--llm`` prose tier.
    """
    origins: List[str] = []
    seen: set = set()
    variables: List[Tuple[str, Any]] = []

    def emit_origin(h: Any) -> None:
        if isinstance(h, str) and h and h not in seen:
            seen.add(h)
            origins.append(h)

    def rec(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k.startswith("forward:origin-server") or k == "forward:modify-host-header":
                    for item in (v if isinstance(v, list) else [v]):
                        if isinstance(item, dict):
                            emit_origin(_rag_text(item.get("host")))
                            emit_origin(_rag_text(item.get("value")))
                            dns = item.get("dns-name")
                            if isinstance(dns, dict):
                                emit_origin(_rag_text(dns.get("value")))
                if k == "assign:variable":
                    for item in (v if isinstance(v, list) else [v]):
                        if isinstance(item, dict) and isinstance(item.get("name"), str):
                            variables.append((item["name"], _rag_text(item.get("value"))))
                rec(v)
        elif isinstance(node, list):
            for item in node:
                rec(item)

    rec(parsed)
    lines = ["# DOCUMENT SUMMARY (deterministic; derived from this config)", ""]
    lines.append("## Origin hostnames referenced in this property")
    for h in origins:
        lines.append("- This property forwards to origin host: %s" % h)
    if not origins:
        lines.append("- (none found)")
    lines.append("")
    lines.append("## Variables defined in this property (name = value)")
    for name, val in variables:
        lines.append("- Variable %s is set to: %s" % (name, _rag_leaf_value(val)))
    if not variables:
        lines.append("- (none found)")
    return "\n".join(lines)


def xml_to_rag(raw: str) -> str:
    """Convert config XML to a RAG-optimized flat index (NOT valid YAML).

    Emits a deterministic :func:`_rag_manifest` summary followed by one fully
    path-qualified ``a > b > c = value`` line per leaf scalar (:func:`_rag_flatten`)
    — a retrieval *projection* for chunk-based ingesters where each line must
    stand alone. Unlike :func:`xml_to_yaml` it does **not** round-trip to the
    source structure; value-level fidelity is verifiable instead (every source
    leaf appears — :func:`rag_fidelity_check`).

    Shares ``xml_to_yaml``'s parse (``xmltodict`` + positioned ``_comment`` nodes)
    so comments and namespaces are carried. Raises ``RuntimeError`` if the
    optional ``xmltodict`` package is missing; propagates ``expat.ExpatError`` on
    malformed XML so the caller can fall back to lossless Markdown.
    """
    try:
        import xmltodict
    except ImportError as exc:
        raise RuntimeError(
            "--rag needs the 'xmltodict' and 'PyYAML' packages; "
            "install them with:  pip install -r requirements.txt"
        ) from exc

    parsed = xmltodict.parse(_xml_comments_to_elements(raw))
    leaves: List[str] = []
    _rag_flatten(parsed, [], leaves)
    return "".join([
        _rag_manifest(parsed),
        "\n\n# FLATTENED CONFIGURATION (path-qualified leaves)\n\n",
        "\n".join(leaves),
        "\n",
    ])


def convert_xml(
    src: Path, dest: Path, mode: str = "auto", yaml_index: bool = False
) -> Tuple[str, str, Path]:
    """Convert XML to Markdown, YAML, or a RAG index. ``mode`` is
    auto | verbatim | transform | yaml | rag.

    Returns ``(content, mode_used, dest_written)``. Verbatim is lossless and the
    safe default for unknown/config XML; transform is for documentation-style XML;
    yaml emits structure-preserving YAML to a sibling ``.yaml`` file; rag emits a
    flattened path-qualified index to a sibling ``.rag.txt`` (the output path
    differs from the ``.md`` ``dest`` passed in, hence the returned path).
    ``yaml_index`` adds structural-path breadcrumbs to the YAML for RAG retrieval.

    A ``yaml`` request on malformed XML falls back to lossless ``verbatim``
    Markdown; a missing optional dependency is a hard error.
    """
    import xml.parsers.expat

    raw = src.read_text(encoding="utf-8", errors="replace")
    chosen = xml_choose_mode(src, raw) if mode == "auto" else mode
    if chosen == "yaml":
        try:
            content = xml_to_yaml(raw, index=yaml_index)
        except xml.parsers.expat.ExpatError:
            chosen = "verbatim"  # malformed XML -> lossless Markdown fallback
        else:
            yaml_dest = dest.with_suffix(".yaml")
            yaml_dest.parent.mkdir(parents=True, exist_ok=True)
            yaml_dest.write_text(content, encoding="utf-8")
            return content, "yaml", yaml_dest
    if chosen == "rag":
        try:
            content = xml_to_rag(raw)
        except xml.parsers.expat.ExpatError:
            chosen = "verbatim"  # malformed XML -> lossless Markdown fallback
        else:
            # Not YAML: a sibling ``.rag.txt`` (NotebookLM ingests .txt directly).
            rag_dest = dest.parent / (dest.stem + ".rag.txt")
            rag_dest.parent.mkdir(parents=True, exist_ok=True)
            rag_dest.write_text(content, encoding="utf-8")
            return content, "rag", rag_dest
    if chosen == "transform":
        try:
            markdown = xml_to_markdown_transform(src, raw)
        except Exception:
            markdown, chosen = xml_to_markdown_verbatim(src, raw), "verbatim"
    else:
        markdown = xml_to_markdown_verbatim(src, raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    return markdown, chosen, dest


# --------------------------------------------------------------------------- #
# CSV conversion — one Markdown card per row
# --------------------------------------------------------------------------- #

DEFAULT_CSV_LIST_MIN_SEGMENTS = 3
DEFAULT_CSV_LIST_MAX_SEGMENT_LEN = 40


def _parse_split_size(spec: str) -> int:
    """Parse '3MB', '512K', '1.5M', or bare bytes -> int bytes."""
    m = re.fullmatch(r"(\d+\.?\d*)\s*([KMGT]?)B?", spec.strip().upper())
    if not m:
        raise ValueError(
            "unrecognized size %r — use e.g. 3MB, 512K, 1500000" % spec
        )
    n = float(m.group(1))
    unit = m.group(2)
    return int(n * {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}.get(unit, 1))


def _detect_csv_list_columns(
    rows: List[Dict[str, str]],
    min_segments: int,
    max_segment_len: int,
    force_list: Optional[List[str]] = None,
) -> set:
    """Return column names whose values look like comma-separated lists.

    For each column, sample all rows: if the median comma-segment count meets
    ``min_segments`` AND the mean segment length is ≤ ``max_segment_len``
    characters, the column is treated as a list. ``force_list`` columns are
    always included regardless of heuristic. Both conditions must hold to
    avoid flagging prose fields that happen to contain commas.
    """
    if not rows:
        return set(force_list or [])
    columns = list(rows[0].keys())
    list_cols: set = set(force_list or [])
    for col in columns:
        if col in list_cols:
            continue
        values = [r.get(col, "").strip() for r in rows if r.get(col, "").strip()]
        if not values:
            continue
        seg_counts = []
        seg_lengths = []
        for v in values:
            segs = [s.strip() for s in v.split(",") if s.strip()]
            seg_counts.append(len(segs))
            if len(segs) > 1:
                seg_lengths.extend(len(s) for s in segs)
        median_segs = sorted(seg_counts)[len(seg_counts) // 2]
        mean_len = sum(seg_lengths) / len(seg_lengths) if seg_lengths else 999
        if median_segs >= min_segments and mean_len <= max_segment_len:
            list_cols.add(col)
    return list_cols


def _detect_csv_title_column(
    columns: List[str],
    rows: List[Dict[str, str]],
    override: Optional[str] = None,
) -> Optional[str]:
    """Column to use as the card heading — override, then name heuristic, then first non-ID."""
    if override:
        return override if override in columns else None
    for col in columns:
        cl = col.lower()
        if cl in ("name", "title") or cl.endswith("name") or cl.endswith("title"):
            return col
    sample = rows[:20]
    for col in columns:
        cl = col.lower()
        is_id = cl == "id" or cl.endswith("id")
        vals = [r.get(col, "").strip() for r in sample if r.get(col, "").strip()]
        all_numeric = bool(vals) and all(v.isdigit() for v in vals)
        if not is_id and not all_numeric:
            return col
    return columns[0] if columns else None


def _detect_csv_id_column(
    columns: List[str], rows: List[Dict[str, str]]
) -> Optional[str]:
    """Column that looks like a record ID (name ends in 'id', or all-numeric values)."""
    for col in columns:
        cl = col.lower()
        if cl == "id" or cl.endswith("id"):
            return col
    sample = rows[:20]
    for col in columns:
        vals = [r.get(col, "").strip() for r in sample if r.get(col, "").strip()]
        if vals and all(v.isdigit() for v in vals):
            return col
    return None


def _render_csv_card(
    row: Dict[str, str],
    title_col: Optional[str],
    id_col: Optional[str],
    list_cols: set,
    skip_cols: set,
) -> str:
    """Render one CSV row as a self-contained Markdown card."""
    title = row.get(title_col, "").strip() if title_col else ""
    id_val = row.get(id_col, "").strip() if id_col else ""
    if title and id_val:
        heading = "## %s (ID: %s)" % (title, id_val)
    elif title:
        heading = "## %s" % title
    elif id_val:
        heading = "## ID: %s" % id_val
    else:
        heading = "## (unnamed)"

    # Heading columns are already represented above — don't repeat them as fields.
    heading_cols = {c for c in (title_col, id_col) if c}

    lines = [heading, ""]
    for col, val in row.items():
        if col in skip_cols or col in heading_cols:
            continue
        val = val.strip()
        if not val:
            continue
        if col in list_cols:
            segs = [s.strip() for s in val.split(",") if s.strip()]
            if len(segs) > 1:
                lines.append("- **%s (%d):**" % (col, len(segs)))
                for seg in segs:
                    lines.append("  - %s" % seg)
                continue
        lines.append("- **%s:** %s" % (col, val))
    return "\n".join(lines)


def convert_csv(
    src: Path,
    dest: Path,
    *,
    list_min_segments: int = DEFAULT_CSV_LIST_MIN_SEGMENTS,
    list_max_segment_len: int = DEFAULT_CSV_LIST_MAX_SEGMENT_LEN,
    list_columns: Optional[List[str]] = None,
    skip_columns: Optional[List[str]] = None,
    title_column: Optional[str] = None,
    split_bytes: Optional[int] = None,
    frontmatter: Optional[Callable[[Optional[int], Optional[int]], str]] = None,
) -> CsvConversion:
    """Convert a CSV to Markdown cards (one card per row) and write to ``dest``.

    Each row becomes a ``##`` heading (auto-detected or overridden title column
    + ID column) followed by key/value bullet fields. Columns whose values look
    like comma-delimited lists (heuristic: median ≥ ``list_min_segments``
    segments, mean length ≤ ``list_max_segment_len`` chars) are expanded as
    indented bullet sub-lists instead of flat strings.

    When ``split_bytes`` is set, output is split into ``<stem>-part001.md``,
    ``<stem>-part002.md``, … at card boundaries so no file exceeds the threshold.
    Cards are never split mid-card. The ``--clean`` pass is intentionally
    skipped for CSV — the output is generated clean by construction.

    ``frontmatter`` is an optional ``(part, parts) -> yaml_block`` callable. It is
    prepended to **every** output file so each atomic split part retains its
    provenance (called with ``(None, None)`` for unsplit output, and ``(i, M)``
    per part otherwise). The block counts toward the per-part byte budget so the
    split guarantee still holds.
    """
    import csv as _csv

    with src.open(newline="", encoding="utf-8-sig") as fh:
        reader = _csv.DictReader(fh)
        rows = list(reader)

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        dest.write_text("", encoding="utf-8")
        return CsvConversion(cards=0, rows=0, output_paths=[dest])

    columns = list(rows[0].keys())
    skip_set = set(skip_columns or [])
    list_cols = _detect_csv_list_columns(
        rows, list_min_segments, list_max_segment_len, list_columns
    )
    title_col = _detect_csv_title_column(columns, rows, title_column)
    id_col = _detect_csv_id_column(columns, rows)
    if id_col == title_col:
        id_col = None  # same column — just show it as the heading, no "(ID: …)"

    warnings: List[str] = []
    cards: List[str] = []
    for row in rows:
        title_val = row.get(title_col, "").strip() if title_col else ""
        if not title_val:
            warnings.append(
                "empty heading value in row %d (title column: %s)"
                % (len(cards) + 1, title_col or "<none>")
            )
        cards.append(_render_csv_card(row, title_col, id_col, list_cols, skip_set))

    def _write_part(buf: List[str], path: Path, fm: str = "") -> None:
        body = "\n\n---\n\n".join(buf) + "\n"
        path.write_text((fm + "\n\n" + body) if fm else body, encoding="utf-8")

    if split_bytes is None:
        _write_part(cards, dest, frontmatter(None, None) if frontmatter else "")
        return CsvConversion(cards=len(cards), rows=len(rows), output_paths=[dest], warnings=warnings)

    # Partition cards into byte-bounded groups (never splitting a card), then
    # write each — frontmatter is prepended to every part with the now-known
    # total. Cards are already all in memory, so grouping first costs nothing
    # extra and lets each part stamp "part i of M".
    sep_size = len("\n\n---\n\n".encode())
    # Reserve the frontmatter's worst-case size from the budget so a part with
    # its YAML block still respects split_bytes (digits of part/parts vary by a
    # few bytes — overestimate with a high sample).
    fm_overhead = (
        len((frontmatter(999, 999) + "\n\n").encode()) if frontmatter else 0
    )
    budget = max(1, split_bytes - fm_overhead)

    groups: List[List[str]] = []
    buf, buf_size = [], 0
    for card in cards:
        card_bytes = len(card.encode())
        overhead = sep_size if buf else 0
        if buf and buf_size + overhead + card_bytes > budget:
            groups.append(buf)
            buf, buf_size = [], 0
        buf.append(card)
        buf_size += (sep_size if len(buf) > 1 else 0) + card_bytes
    if buf:
        groups.append(buf)

    output_paths: List[Path] = []
    total = len(groups)
    for i, group in enumerate(groups, 1):
        pth = dest.parent / ("%s-part%03d%s" % (dest.stem, i, dest.suffix))
        _write_part(group, pth, frontmatter(i, total) if frontmatter else "")
        output_paths.append(pth)

    return CsvConversion(
        cards=len(cards), rows=len(rows), output_paths=output_paths, warnings=warnings
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def extract_pdf_plaintext(src: Path, last_page: Optional[int] = None) -> str:
    """Extract raw plain text from a PDF with ``pdftotext`` (optionally 1..N)."""
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


_TABLE_DELIM_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$", re.MULTILINE
)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*(```|~~~)", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def strip_markdown(md: str) -> str:
    """Reduce Markdown to plain-text content for fidelity comparison."""
    text = md
    # Drop our RAG provenance layers first so injected metadata never registers
    # as "added" text against the pdftotext baseline and depresses similarity.
    text = _FRONTMATTER_RE.sub("", text)  # leading YAML frontmatter block
    text = _PAGE_MARKER_RE.sub("", text)  # doc2md:page=N boundary markers
    text = re.sub(r"^\s*(```|~~~).*$", "", text, flags=re.MULTILINE)  # fence lines
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    text = _IMAGE_RE.sub("", text)  # images (refs carry no source text)
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
    """Whitespace-normalized character similarity (difflib) in [0.0, 1.0]."""
    a = _normalize_whitespace(original_text)
    b = _normalize_whitespace(strip_markdown(markdown_text))
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# Above this normalized length, difflib's O(n*m) ratio is too slow to be worth
# it (minutes on a multi-MB doc), so fall back to an O(n) token-recall metric.
_SIMILARITY_SEQUENCE_MAX_CHARS = 200_000


def _containment_ratio(original_text: str, markdown_text: str) -> float:
    """O(n) token recall: fraction of the source's word tokens present in the md.

    Order-insensitive, so it isn't fooled by layout reflow (multi-column/tabular
    PDFs) or by HTML→Markdown reordering, and it's linear, not quadratic.

    Tokenizes both sides with a plain ``\\w+`` pattern and *no* markup stripping:
    that pulls the words out of inline-tag / code content alike (``<assign:var>``
    -> ``assign``, ``var``), so docs whose body is XML/code examples — which
    ``strip_markdown`` would wrongly delete as tags — are scored on real content.
    Extra tokens the Markdown adds (link URLs, etc.) don't affect recall.
    """
    from collections import Counter

    src = Counter(re.findall(r"\w+", original_text.lower()))
    out = Counter(re.findall(r"\w+", markdown_text.lower()))
    total = sum(src.values())
    if total == 0:
        return 1.0
    covered = sum(min(n, out[w]) for w, n in src.items())
    return covered / total


def compute_similarity(original_text: str, markdown_text: str) -> Tuple[float, str]:
    """Source-vs-Markdown fidelity score and the method used.

    Uses difflib's character ratio for normal-size docs ("sequence"); for very
    large docs it switches to the O(n) token-recall metric ("containment") so
    validation doesn't spend minutes on an O(n*m) diff — which matters because
    the score is only advisory when pdfmux confidence is high anyway.
    """
    a = _normalize_whitespace(original_text)
    b = _normalize_whitespace(strip_markdown(markdown_text))
    if not a and not b:
        return 1.0, "sequence"
    if max(len(a), len(b)) > _SIMILARITY_SEQUENCE_MAX_CHARS:
        return _containment_ratio(original_text, markdown_text), "containment"
    return difflib.SequenceMatcher(None, a, b).ratio(), "sequence"


# --------------------------------------------------------------------------- #
# Markdown cleanup (deterministic, lossless, default-on; --no-clean to skip)
# --------------------------------------------------------------------------- #
#
# PDF extraction leaves non-content noise that hurts LLM consumption without
# adding information: page headers/footers repeated on every page, empty
# duplicate tables at page breaks, and word-fragments split across line-wraps
# inside inline tags (e.g. ``<forward:...ma x-reconnects>``). We remove/repair
# these with deterministic rules (no LLM) and *prove* losslessness with a
# character-conservation check: every non-whitespace character of the input must
# end up either in the cleaned output or in the audited set of removed lines.
# The removal steps only delete provably-repeated furniture / empty tables; the
# repair steps only delete whitespace (rejoining split tokens). If the check
# fails for any reason, the original markdown is returned untouched — default-on
# can therefore never silently lose detail.

_TAGNAME_RE = re.compile(r"/?[A-Za-z][\w.:-]*")

# Universally-safe furniture: standalone lines that are *only* a page number or
# a date. Such lines are page chrome, effectively never body content. Textual
# chrome (company names, "Confidential", running titles) can't be told from
# content automatically on this scrambled single-blob output, so it's left to
# the user's explicit ``--strip-line`` patterns rather than guessed at.
_SAFE_FURNITURE_PATTERNS = [
    re.compile(r"\d{1,5}"),  # bare page number
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),  # date m/d/yy or m/d/yyyy
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),  # date yyyy-mm-dd
    re.compile(r"[Pp]age \d{1,5}(\s+of\s+\d{1,5})?"),  # "Page N" / "Page N of M"
]


def _nonspace_counter(s: str):
    """Multiset of non-whitespace characters — the unit of the fidelity proof."""
    from collections import Counter

    return Counter(ch for ch in s if not ch.isspace())


def _visible_text_counter(s: str):
    """Multiset of visible-text word tokens (HTML tags stripped).

    The fidelity unit for HTML-derived Markdown, whose cleanup deliberately
    removes *markup* characters (tags, ``data:`` blobs). Comparing visible text
    proves the rendered content is preserved even though markup chars are not.
    """
    from collections import Counter

    return Counter(re.findall(r"\w+", _strip_html(s).lower()))


def _compile_strip_patterns(strip_patterns: Optional[List[str]], *, include_safe: bool):
    """Compile line-removal patterns: optional safe built-ins + user regexes.

    ``include_safe`` adds the page-number/date :data:`_SAFE_FURNITURE_PATTERNS`
    (PDF only — those are print chrome; on web/Word output a bare number is
    likely content, so it's off there). Each pattern is matched (``fullmatch``)
    against the *stripped* line, so it removes only lines it fully describes,
    never a substring of content. Invalid regexes are skipped, not fatal.
    """
    patterns = list(_SAFE_FURNITURE_PATTERNS) if include_safe else []
    for p in strip_patterns or []:
        try:
            patterns.append(re.compile(p))
        except re.error:
            print("[clean] ignoring invalid --strip-line regex: %r" % p, file=sys.stderr)
    return patterns


def _strip_lines_by_pattern(lines: List[str], patterns) -> Tuple[List[str], List[str]]:
    """Split ``lines`` into (kept, removed) by full-line match against patterns."""
    kept: List[str] = []
    removed: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s and any(p.fullmatch(s) for p in patterns):
            removed.append(ln)
        else:
            kept.append(ln)
    return kept, removed


def _drop_empty_tables(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Drop table blocks whose data rows are all empty (page-break dup fragments).

    Conservative: only blocks that have data rows *and* every data cell is blank
    are removed; header-only tables are kept. Returns ``(kept, removed)``.
    """
    out: List[str] = []
    removed: List[str] = []
    i, n = 0, len(lines)

    def is_row(ln: str) -> bool:
        return ln.lstrip().startswith("|")

    def is_delim(ln: str) -> bool:
        return bool(_TABLE_DELIM_RE.match(ln))

    def cells(ln: str) -> List[str]:
        return [c.strip() for c in ln.strip().strip("|").split("|")]

    while i < n:
        if is_row(lines[i]) and i + 1 < n and is_delim(lines[i + 1]):
            j = i + 2
            data = []
            while j < n and is_row(lines[j]) and not is_delim(lines[j]):
                data.append(j)
                j += 1
            if data and all(all(c == "" for c in cells(lines[k])) for k in data):
                removed.extend(lines[i:j])
                i = j
                continue
        out.append(lines[i])
        i += 1
    return out, removed


def _repair_xml_tag_splits(text: str) -> Tuple[str, int]:
    """Rejoin tag paths split by a line-wrap space, e.g. ``<a:b.ma x-c>``.

    Only touches ``<...>`` whose contents (minus spaces) form a valid element-tag
    name and which carry no attributes (no ``=``/quotes), so prose like ``x < 5
    and y > 3`` is never altered. Deletes whitespace only.
    """
    n = 0

    def repl(m: "re.Match") -> str:
        nonlocal n
        inner = m.group(1)
        if " " not in inner or "=" in inner or '"' in inner or "'" in inner:
            return m.group(0)
        joined = inner.replace(" ", "")
        if _TAGNAME_RE.fullmatch(joined):
            n += 1
            return "<" + joined + ">"
        return m.group(0)

    return re.sub(r"<([^<>]*)>", repl, text), n


def _dehyphenate_table_cells(text: str) -> Tuple[str, int]:
    """Rejoin enum values wrap-split inside table cells, e.g. ``client -request``.

    Scoped to table rows and to ``word -word`` (no space after the hyphen), so
    real dashes (``a - b``) and prose are untouched. Deletes whitespace only.
    """
    n = 0
    pat = re.compile(r"(\w) -(\w)")
    lines = text.split("\n")
    for idx, ln in enumerate(lines):
        if ln.lstrip().startswith("|"):
            new, k = pat.subn(r"\1-\2", ln)
            if k:
                n += k
                lines[idx] = new
    return "\n".join(lines), n


def _collapse_blank_runs(text: str) -> str:
    """Strip per-line trailing whitespace and collapse 3+ blank lines to one."""
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_markdown(
    md: str, *, strip_patterns: Optional[List[str]] = None
) -> Tuple[str, Dict[str, object]]:
    """Clean assembled PDF-derived Markdown for LLM consumption.

    Removes furniture lines that fully match a safe built-in pattern (bare page
    number / date) or a user ``--strip-line`` regex; drops empty duplicate
    tables (page-break fragments); and repairs line-wrap token splits inside
    tags / table cells. Returns ``(cleaned, stats)``.

    Guarantees no data-fidelity loss via a non-whitespace character-conservation
    check: ``chars(input) == chars(cleaned) + chars(removed)`` — furniture
    removal only drops whole audited lines, the repair steps delete only
    whitespace. If that can't be shown (or anything errors), the original is
    returned with ``stats["fell_back"]`` set, so callers always ship faithful
    Markdown.
    """
    stats: Dict[str, object] = {
        "kind": "pdf",
        "applied": False,
        "fell_back": False,
        "furniture_lines_removed": 0,
        "empty_table_rows_removed": 0,
        "xml_tag_joins": 0,
        "cell_dehyphenations": 0,
        "furniture_samples": [],
    }
    try:
        patterns = _compile_strip_patterns(strip_patterns, include_safe=True)
        kept, removed = _strip_lines_by_pattern(md.split("\n"), patterns)
        stats["furniture_lines_removed"] = len(removed)
        stats["furniture_samples"] = sorted({r.strip() for r in removed})[:15]

        kept, removed_tbl = _drop_empty_tables(kept)
        removed.extend(removed_tbl)
        stats["empty_table_rows_removed"] = len(removed_tbl)

        text = "\n".join(kept)
        text, n_xml = _repair_xml_tag_splits(text)
        text, n_cell = _dehyphenate_table_cells(text)
        cleaned = _collapse_blank_runs(text)

        # Fidelity gate: every non-whitespace character is preserved in the
        # output or accounted for in the removed (audited) lines.
        if _nonspace_counter(md) != _nonspace_counter(cleaned) + _nonspace_counter("\n".join(removed)):
            stats["fell_back"] = True
            return md, stats

        stats["xml_tag_joins"] = n_xml
        stats["cell_dehyphenations"] = n_cell
        stats["applied"] = True
        return cleaned, stats
    except Exception as exc:  # pragma: no cover - cleanup must never break a run
        stats["fell_back"] = True
        stats["error"] = str(exc)
        return md, stats


def clean_html_markdown(
    md: str, *, strip_patterns: Optional[List[str]] = None
) -> Tuple[str, Dict[str, object]]:
    """Clean Markdown converted from HTML/MHTML (SPA exports) for LLM use.

    Removes single-page-app chrome that pandoc passes through (:func:
    `_strip_pandoc_html_chrome`: ``data:`` icon images, empty ``div``/``span``
    scaffolding) and any user ``--strip-line`` lines. Unlike the PDF cleaner it
    does *not* strip bare-number/date lines (on web content those are usually
    real) and has no token-split repairs.

    Fidelity is proven at the *visible-text* level (:func:`_visible_text_counter`)
    rather than character level, because the transforms intentionally drop markup
    characters: the rendered text must be conserved (output + audited removals).
    Falls back to the raw Markdown if it can't be shown.
    """
    stats: Dict[str, object] = {
        "kind": "html",
        "applied": False,
        "fell_back": False,
        "furniture_lines_removed": 0,
        "chrome_imgs_removed": 0,
        "tags_unwrapped": 0,
        "furniture_samples": [],
    }
    try:
        patterns = _compile_strip_patterns(strip_patterns, include_safe=False)
        kept, removed = _strip_lines_by_pattern(md.split("\n"), patterns)
        stats["furniture_lines_removed"] = len(removed)
        stats["furniture_samples"] = sorted({r.strip() for r in removed})[:15]

        text, n_imgs, n_tags = _strip_pandoc_html_chrome("\n".join(kept), count=True)
        cleaned = _collapse_blank_runs(text)

        # Fidelity gate: visible *body* text (between tags) is preserved —
        # chrome-strip removes only markup and decorative ``data:`` icon images.
        # Note: an icon's alt attribute lives inside the tag and so isn't covered
        # by this text-level check; that matches the established behavior (these
        # are decorative inline icons — real file-path figures are never removed),
        # modulo the audited --strip-line removals.
        if _visible_text_counter(md) != _visible_text_counter(cleaned) + _visible_text_counter("\n".join(removed)):
            stats["fell_back"] = True
            return md, stats

        stats["chrome_imgs_removed"] = n_imgs
        stats["tags_unwrapped"] = n_tags
        stats["applied"] = True
        return cleaned, stats
    except Exception as exc:  # pragma: no cover - cleanup must never break a run
        stats["fell_back"] = True
        stats["error"] = str(exc)
        return md, stats


def clean_safe_subset(
    md: str, *, strip_patterns: Optional[List[str]] = None
) -> Tuple[str, Dict[str, object]]:
    """Format-agnostic safe cleanup for docx / pandoc Markdown.

    Only the universally-safe operations: drop user ``--strip-line`` lines and
    collapse trailing/blank whitespace. No PDF furniture/repairs, no HTML
    chrome-stripping. Character-conservation-gated; falls back to raw on doubt.
    """
    stats: Dict[str, object] = {
        "kind": "safe",
        "applied": False,
        "fell_back": False,
        "furniture_lines_removed": 0,
        "furniture_samples": [],
    }
    try:
        patterns = _compile_strip_patterns(strip_patterns, include_safe=False)
        kept, removed = _strip_lines_by_pattern(md.split("\n"), patterns)
        cleaned = _collapse_blank_runs("\n".join(kept))
        if _nonspace_counter(md) != _nonspace_counter(cleaned) + _nonspace_counter("\n".join(removed)):
            stats["fell_back"] = True
            return md, stats
        stats["furniture_lines_removed"] = len(removed)
        stats["furniture_samples"] = sorted({r.strip() for r in removed})[:15]
        stats["applied"] = True
        return cleaned, stats
    except Exception as exc:  # pragma: no cover - cleanup must never break a run
        stats["fell_back"] = True
        stats["error"] = str(exc)
        return md, stats


def structural_counts(md: str) -> Dict[str, int]:
    """Count headings, fenced code blocks, pipe tables, and image references."""
    return {
        "headings": len(_HEADING_RE.findall(md)),
        "code_blocks": len(_FENCE_RE.findall(md)) // 2,
        "tables": len(_TABLE_DELIM_RE.findall(md)),
        "images": len(_IMAGE_RE.findall(md)),
    }


def structural_check(
    counts: Dict[str, int], plain_text: str, min_chars: int = STRUCT_MIN_CHARS
) -> Tuple[bool, Optional[str]]:
    """Flag long documents that have no structural elements at all."""
    total = (
        counts.get("headings", 0)
        + counts.get("code_blocks", 0)
        + counts.get("tables", 0)
        + counts.get("images", 0)
    )
    if len(plain_text) >= min_chars and total == 0:
        return (
            False,
            "document has %d chars but no headings, code, tables, or figures"
            % len(plain_text),
        )
    return True, None


def _count_yaml_keys(node) -> int:
    """Recursively count mapping keys in a parsed YAML structure."""
    if isinstance(node, dict):
        return len(node) + sum(_count_yaml_keys(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_yaml_keys(v) for v in node)
    return 0


def yaml_structural_check(content: str) -> Tuple[Dict[str, int], bool, Optional[str]]:
    """Structural validity for ``--yaml`` output: it must parse and be non-empty.

    Replaces the Markdown-oriented :func:`structural_check` for YAML output, which
    legitimately contains no headings/tables/code fences. Returns
    ``(counts, ok, reason)`` mirroring the Markdown path's shape.
    """
    import yaml

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        return {"keys": 0}, False, "output is not valid YAML: %s" % exc
    keys = _count_yaml_keys(data)
    if data is None or keys == 0:
        return {"keys": keys}, False, "YAML output is empty (no mapping keys)"
    return {"keys": keys}, True, None


def rag_structural_check(content: str) -> Tuple[Dict[str, int], bool, Optional[str]]:
    """Structural validity for ``--rag`` output: a summary block + ≥1 leaf line.

    The flattened index isn't YAML, so :func:`yaml_structural_check` doesn't apply.
    A valid index has the deterministic DOCUMENT SUMMARY header and at least one
    path-qualified ``… = value`` leaf. Returns ``(counts, ok, reason)`` mirroring
    the other structural checks' shape.
    """
    has_summary = "# DOCUMENT SUMMARY" in content
    leaves = sum(1 for ln in content.splitlines() if " = " in ln)
    ok = has_summary and leaves > 0
    reason = None if ok else "RAG output missing summary or path-qualified leaves"
    return {"leaves": leaves}, ok, reason


def _collect_xml_scalars(node: Any, out: List[str]) -> None:
    """Gather every scalar leaf value from an xmltodict parse (recursively)."""
    if isinstance(node, dict):
        for v in node.values():
            _collect_xml_scalars(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_xml_scalars(v, out)
    elif node is not None:
        out.append(str(node))


def rag_fidelity_check(
    content: str, source_xml: Optional[str]
) -> Tuple[Optional[bool], Optional[str]]:
    """Value-level fidelity for ``--rag`` output: every source leaf value present.

    The flattened index intentionally does NOT round-trip to the source structure
    (unlike :func:`yaml_fidelity_check`), so fidelity is verified at the value
    level instead: collect every scalar leaf from the source XML and require each
    to appear in the emitted text (whitespace-normalized to match
    :func:`_rag_leaf_value`'s one-line rendering). A miss means the flattener
    dropped a leaf. Returns ``(None, None)`` when no source is available.
    """
    if source_xml is None:
        return None, None
    try:
        import xmltodict

        expected = xmltodict.parse(_xml_comments_to_elements(source_xml))
    except Exception as exc:  # noqa: BLE001 - any failure means we can't vouch for it
        return False, "could not verify RAG fidelity: %s" % exc
    scalars: List[str] = []
    _collect_xml_scalars(expected, scalars)
    norm_content = " ".join(content.split())
    missing = []
    for s in scalars:
        norm = " ".join(s.split())
        if norm and norm not in norm_content:
            missing.append(s)
    if missing:
        sample = ", ".join(repr(s) for s in missing[:3])
        return False, "%d source value(s) missing from RAG output (e.g. %s)" % (
            len(missing), sample,
        )
    return True, None


def yaml_fidelity_check(
    yaml_content: str, source_xml: Optional[str]
) -> Tuple[Optional[bool], Optional[str]]:
    """Verify ``--yaml`` output reproduces the source XML's structure exactly.

    Re-parses the emitted YAML and compares it to ``xmltodict``'s parse of the
    source XML. Because the conversion is
    ``yaml.dump(xmltodict.parse(_xml_comments_to_elements(xml)))``, a faithful run
    must satisfy ``safe_load(yaml) == xmltodict.parse(_xml_comments_to_elements(xml))``;
    any mismatch means a bad parse/serialization path (e.g. a scalar that YAML
    re-reads as a bool/null, or a dropped/misplaced comment), which this fails on
    rather than shipping silently. The comment preprocessing mirrors
    :func:`xml_to_yaml` so the comparison verifies the positioned ``_comment``
    nodes too.

    Returns ``(None, None)`` when no source is available (fidelity unassessable).
    """
    if source_xml is None:
        return None, None
    try:
        import xmltodict
        import yaml

        expected = xmltodict.parse(_xml_comments_to_elements(source_xml))
        actual = yaml.safe_load(yaml_content)
    except Exception as exc:  # noqa: BLE001 - any failure means we can't vouch for it
        return False, "could not verify YAML fidelity: %s" % exc
    if actual == expected:
        return True, None
    return False, "YAML output does not round-trip to the source XML structure"


def validate(
    markdown: str,
    original_plaintext: Optional[str],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    confidence: Optional[float] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    output_format: str = "markdown",
    source_xml: Optional[str] = None,
) -> Dict[str, object]:
    """Run fidelity + confidence + structural checks and decide pass/fail.

    ``original_plaintext`` is ``None`` for non-PDF inputs (no pdftotext source),
    in which case the similarity check is skipped. ``confidence`` is pdfmux's
    document confidence (PDF only); below ``min_confidence`` fails the document.

    A low ordered similarity is only a *hard* failure when neither fallback
    signal vouches for the conversion: the char-diff is a flat cross-check, not
    ground truth, and it tanks whenever content is faithfully *reflowed* —
    tabular/multi-column PDFs (pdftotext also splits words at line-wraps) and
    HTML→Markdown (Confluence/Apple MHTML). So the document still passes if
    either pdfmux **confidence** is high *or* order-insensitive **content
    recall** (``content_recall`` ≥ ``containment_threshold``) is high; the low
    score stays visible (``similarity_ok``) for the report. This is what lets a
    faithful, content-complete MHTML/web conversion pass despite a low char-diff,
    while genuine content loss (low recall too) still fails.

    Structural emptiness (a long document with no headings/tables/code/figures)
    is only a *hard* failure when we have no fidelity signal — if text
    similarity is high the conversion is demonstrably faithful, so a
    prose-heavy doc that simply lacks structure is flagged (``structural_ok``)
    but not failed.

    ``output_format="yaml"`` switches the structural check from Markdown elements
    to YAML validity (the output must parse and be a non-empty mapping), since
    ``--yaml`` output legitimately has no headings/tables/code fences. When
    ``source_xml`` is also supplied, the YAML is additionally verified to
    round-trip to the source XML's structure (``fidelity_ok``); a mismatch fails
    the document. ``output_format="rag"`` (the ``--rag`` flat index) instead
    checks for a summary block plus path-qualified leaves, and verifies fidelity
    at the value level — every source leaf must appear (it deliberately does not
    round-trip).
    """
    fidelity_ok: Optional[bool] = None
    fidelity_reason: Optional[str] = None
    if output_format == "yaml":
        counts, struct_ok, struct_reason = yaml_structural_check(markdown)
        fidelity_ok, fidelity_reason = yaml_fidelity_check(markdown, source_xml)
    elif output_format == "rag":
        counts, struct_ok, struct_reason = rag_structural_check(markdown)
        fidelity_ok, fidelity_reason = rag_fidelity_check(markdown, source_xml)
    else:
        counts = structural_counts(markdown)
        plain = strip_markdown(markdown)
        struct_ok, struct_reason = structural_check(counts, plain)

    similarity: Optional[float] = None
    sim_ok: Optional[bool] = None
    sim_method: Optional[str] = None
    content_recall: Optional[float] = None
    content_ok: Optional[bool] = None
    if original_plaintext is not None:
        similarity, sim_method = compute_similarity(original_plaintext, markdown)
        sim_ok = similarity >= similarity_threshold
        # Order-insensitive content recall as a second fidelity signal (reuse the
        # primary score when it already is containment, for huge docs).
        content_recall = (
            similarity
            if sim_method == "containment"
            else _containment_ratio(original_plaintext, markdown)
        )
        content_ok = content_recall >= containment_threshold

    conf_ok: Optional[bool] = None
    if confidence is not None:
        conf_ok = confidence >= min_confidence

    # Structural emptiness is fatal only when fidelity is otherwise unverifiable.
    struct_fatal = (not struct_ok) and (sim_ok is None)
    # A low ordered similarity is fatal only when *both* fallback signals also
    # fail. The ordered char-diff is a flat cross-check, not ground truth: it
    # tanks on faithful conversions that reflow content — tabular/multi-column
    # PDFs (where pdftotext also shatters words at line-wraps, "configuration" ->
    # "config" + "uration") and HTML->Markdown (Confluence/Apple MHTML), whose
    # XML/code examples and chrome removal reorder text. So we don't fail when
    # either a confident extractor signal (pdfmux) OR high order-insensitive
    # content recall vouches for it; the low score stays visible in the report.
    sim_fatal = (sim_ok is False) and (conf_ok is not True) and (content_ok is not True)
    passed = (
        (not sim_fatal)
        and (conf_ok is not False)
        and (fidelity_ok is not False)
        and not struct_fatal
    )

    return {
        "similarity": similarity,
        "similarity_method": sim_method,
        "similarity_ok": sim_ok,
        "content_recall": content_recall,
        "confidence_ok": conf_ok,
        "structural": counts,
        "structural_ok": struct_ok,
        "structural_reason": struct_reason,
        "fidelity_ok": fidelity_ok,
        "fidelity_reason": fidelity_reason,
        "passed": bool(passed),
    }


# --------------------------------------------------------------------------- #
# Per-file orchestration
# --------------------------------------------------------------------------- #


def slugify(text: str) -> str:
    """Filesystem/URL-safe slug: lowercase, alphanumeric, hyphen-separated.

    Markdown image links can't contain unescaped spaces, so every generated
    name (``.md`` and figure PNGs) is slugified for safe LLM/tooling consumption.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "document"


def _output_path_for(src: Path, input_root: Path, output_dir: Path) -> Path:
    """Mirror the input file's relative location under the output dir as a
    slugified ``.md`` (parent subdirectories preserved)."""
    rel = src.relative_to(input_root)
    return output_dir / rel.parent / (slugify(rel.stem) + ".md")


def _note_auto_decision(
    record: FileRecord, decision: Dict[str, str], *, announce: bool = True
) -> None:
    """Record an auto-made choice on ``record`` and (optionally) echo it.

    Keeps ``conversion_report.json`` authoritative: every entry is added to
    ``record.auto_decisions``. When ``announce`` is True it's also printed with
    its override hint, so notable decisions are visible interactively. Pass
    ``announce=False`` for unremarkable defaults (e.g. quality resolving to the
    usual ``standard``) to keep batch output clean while still recording them.
    """
    record.auto_decisions.append(decision)
    if announce:
        print(
            "[auto] %s: %s = %s — %s (override: %s)"
            % (
                record.filename,
                decision["setting"],
                decision["choice"],
                decision["reason"],
                decision["override"],
            ),
            file=sys.stderr,
        )


def _report_cleaning(record: FileRecord) -> None:
    """Echo a one-line cleanup summary (and any lossless-fallback), per kind."""
    c = record.cleaning
    if not c:
        return
    if c.get("fell_back"):
        print(
            "[clean] %s: skipped — losslessness not verifiable, kept raw text"
            % record.filename,
            file=sys.stderr,
        )
    if not c.get("applied"):
        return
    kind = c.get("kind")
    n_furn = int(c.get("furniture_lines_removed", 0))
    if kind == "pdf":
        detail = (
            "-%d furniture lines, -%d empty-table rows, %d tag-splits + %d cell-splits repaired"
            % (
                n_furn,
                int(c.get("empty_table_rows_removed", 0)),
                int(c.get("xml_tag_joins", 0)),
                int(c.get("cell_dehyphenations", 0)),
            )
        )
    elif kind == "html":
        detail = "-%d data-URI icons, -%d scaffolding tags, -%d --strip-line" % (
            int(c.get("chrome_imgs_removed", 0)),
            int(c.get("tags_unwrapped", 0)),
            n_furn,
        )
    else:  # safe subset (docx / pandoc)
        detail = "-%d --strip-line lines, whitespace normalized" % n_furn
    print("[clean] %s: %s (lossless verified)" % (record.filename, detail), file=sys.stderr)


def process_file(
    src: Path,
    input_root: Path,
    output_dir: Path,
    *,
    deps: Dict[str, bool],
    quality: str = "standard",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    preview_pages: Optional[int] = None,
    visual_cfg: Optional[VisualConfig] = None,
    xml_mode: str = "auto",
    yaml_index: bool = False,
    ocr_mode: str = "auto",
    large_doc_pages: int = DEFAULT_LARGE_DOC_PAGES,
    clean: bool = True,
    strip_patterns: Optional[List[str]] = None,
    used_llm: bool = False,
    rag_metadata: bool = True,
    source_abspath: bool = False,
    csv_list_min_segments: int = DEFAULT_CSV_LIST_MIN_SEGMENTS,
    csv_list_max_segment_len: int = DEFAULT_CSV_LIST_MAX_SEGMENT_LEN,
    csv_list_columns: Optional[List[str]] = None,
    csv_skip_columns: Optional[List[str]] = None,
    csv_title_column: Optional[str] = None,
    split_bytes: Optional[int] = None,
) -> FileRecord:
    """Convert + validate a single file. Never raises — errors go in the record.

    The handler is chosen by content sniffing (:func:`detect_format`), not the
    extension — so a Confluence ``.doc`` (MHTML) and a config ``.xml`` route to
    the right converter.
    """
    record = FileRecord(
        filename=src.name,
        source_path=str(src),
        preview=preview_pages is not None,
        used_llm=used_llm,
    )

    if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
        record.status = "skipped"
        record.error = "unsupported format: %s" % (src.suffix.lower() or "<none>")
        return record

    fmt = detect_format(src)
    if fmt in ("unsupported",):
        record.status = "skipped"
        record.error = "unrecognized content (%s)" % src.suffix.lower()
        return record

    dest = _output_path_for(src, input_root, output_dir)

    try:
        original_plaintext: Optional[str] = None
        confidence: Optional[float] = None
        output_format = "markdown"
        source_xml: Optional[str] = None
        fm_quality: Optional[str] = None  # frontmatter: resolved quality tier (PDF)
        fm_page_count: Optional[int] = None  # frontmatter: page count (PDF)

        if fmt == "pdf":
            if not deps.get("pdfmux"):
                raise RuntimeError("pdfmux is not installed; cannot convert PDFs")
            record.converter = "pdfmux"
            fm_page_count = _pdf_page_count(src)
            eff_quality, q_decision = resolve_quality(
                quality, fm_page_count, large_doc_pages
            )
            fm_quality = eff_quality
            if q_decision is not None:  # announce the large-doc fast downgrade
                _note_auto_decision(record, q_decision)
            # Only the local Docling/pymupdf4llm paths consult the OCR policy; the
            # LLM path (quality="high") doesn't, so don't claim a decision there.
            if eff_quality != "high":
                _note_auto_decision(record, _ocr_decision(src, ocr_mode))
            conv = convert_pdf(
                src, dest, quality=eff_quality, preview_pages=preview_pages,
                visual_cfg=visual_cfg, clean=clean, strip_patterns=strip_patterns,
                page_markers=rag_metadata,
            )
            markdown = conv.markdown
            confidence = conv.confidence
            record.pdfmux_confidence = conv.confidence
            record.min_page_confidence = conv.min_page_confidence
            record.figures = _figure_counts(conv.visuals)
            if conv.degenerate_tables:
                record.extraction_warnings["degenerate_tables"] = conv.degenerate_tables
                print(
                    "[warn] %s: skipped %d degenerate table(s) PyMuPDF could not "
                    "bound (empty cells) — not imaged"
                    % (record.filename, conv.degenerate_tables),
                    file=sys.stderr,
                )
            record.cleaning = conv.cleaning
            _report_cleaning(record)
            # Record whether RAG page markers were woven (only meaningful for a
            # multi-page doc with provenance on). Announce only the suppressed
            # cases; the per-page default is unremarkable and stays silent.
            if rag_metadata and fm_page_count > 1:
                pm_decision = _page_marker_decision(conv.page_source)
                if pm_decision is not None:
                    _note_auto_decision(
                        record, pm_decision, announce=pm_decision["choice"] == "off"
                    )
            if deps.get("pdftotext"):
                original_plaintext = extract_pdf_plaintext(src, last_page=preview_pages)
        elif fmt in ("mhtml", "html", "docx"):
            record.converter = {"mhtml": "mhtml", "html": "html"}.get(fmt, "pandoc(docx)")
            wc = convert_word(
                src, dest, fmt, have_pandoc=bool(deps.get("pandoc")), visual_cfg=visual_cfg,
                clean=clean, strip_patterns=strip_patterns,
            )
            markdown = wc.markdown
            original_plaintext = wc.ref_plaintext
            record.figures = {"diagrams": 0, "images": wc.images, "complex_tables": 0}
            record.cleaning = wc.cleaning
            _report_cleaning(record)
        elif fmt == "xml":
            markdown, mode_used, dest = convert_xml(
                src, dest, mode=xml_mode, yaml_index=yaml_index
            )
            record.converter = "xml-%s" % mode_used
            if xml_mode == "auto":
                _note_auto_decision(record, {
                    "setting": "xml-mode",
                    "choice": mode_used,
                    "reason": "auto: chosen by content sniffing",
                    "override": "--xml-mode verbatim|transform|yaml|rag (or --yaml/--rag)",
                })
            output_format = {"yaml": "yaml", "rag": "rag"}.get(mode_used, "markdown")
            if output_format in ("yaml", "rag"):
                source_xml = src.read_text(encoding="utf-8", errors="replace")
        elif fmt == "csv":
            record.converter = "csv"
            # Frontmatter is injected per output file (so every split part keeps
            # its provenance), not via the central prepend below — CSV returns
            # early. The closure stamps part/parts once convert_csv knows them.
            csv_fm: Optional[Callable[[Optional[int], Optional[int]], str]] = None
            if rag_metadata:
                csv_src_path = _frontmatter_source_path(src, input_root, source_abspath)
                csv_mtime = src.stat().st_mtime

                def csv_fm(part, parts, _path=csv_src_path, _mtime=csv_mtime):
                    return build_frontmatter(
                        title=_derive_title("", src, "csv"),
                        source_file=src.name,
                        source_path=_path,
                        fmt="csv",
                        engine="csv",
                        mtime=_mtime,
                        part=part,
                        parts=parts,
                    )

            cc = convert_csv(
                src, dest,
                list_min_segments=csv_list_min_segments,
                list_max_segment_len=csv_list_max_segment_len,
                list_columns=csv_list_columns,
                skip_columns=csv_skip_columns,
                title_column=csv_title_column,
                split_bytes=split_bytes,
                frontmatter=csv_fm,
            )
            for w in cc.warnings:
                print("[csv] %s: warning — %s" % (src.name, w), file=sys.stderr)
            parts = len(cc.output_paths)
            struct: Dict[str, int] = {"cards": cc.cards, "rows": cc.rows}
            if parts > 1:
                struct["parts"] = parts
            record.structural = struct
            record.output_path = str(cc.output_paths[0]) if cc.output_paths else str(dest)
            record.passed = cc.cards == cc.rows
            record.status = "converted"
            return record
        elif fmt == "doc-binary":
            raise RuntimeError(
                "legacy binary .doc (OLE) needs LibreOffice/antiword; not supported. "
                "Re-save as .docx, or export from Confluence as PDF/Word(MHTML)."
            )
        else:  # pandoc: html/htm/epub/rtf/odt
            if not deps.get("pandoc"):
                raise RuntimeError("pandoc is not installed; cannot convert %s" % src.suffix)
            record.converter = "pandoc"
            markdown, cleaning = convert_with_pandoc(
                src, dest, clean=clean, strip_patterns=strip_patterns
            )
            record.cleaning = cleaning
            _report_cleaning(record)

        # Prepend provenance metadata centrally so every format gets it
        # identically. Markdown gets a "---" YAML frontmatter block; .yaml gets
        # the same version/commit as leading "#" comments instead (its own
        # top-level "---" would start a second YAML document). The stamp is part
        # of the validated string (== on disk) and, for Markdown, is stripped
        # before the fidelity comparison; YAML comments are ignored on parse. CSV
        # returns earlier and injects its own frontmatter per split part.
        if rag_metadata and output_format == "markdown":
            try:
                frontmatter = build_frontmatter(
                    title=_derive_title(markdown, src, fmt),
                    source_file=src.name,
                    source_path=_frontmatter_source_path(src, input_root, source_abspath),
                    fmt=fmt,
                    engine=record.converter,
                    mtime=src.stat().st_mtime,
                    quality=fm_quality,
                    page_count=fm_page_count,
                    confidence=confidence,
                )
                markdown = frontmatter + "\n\n" + markdown
                Path(dest).write_text(markdown, encoding="utf-8")
            except Exception as exc:  # frontmatter must never break a conversion
                print(
                    "[meta] %s: skipped frontmatter — %s" % (src.name, exc),
                    file=sys.stderr,
                )
        elif rag_metadata and output_format in ("yaml", "rag"):
            try:
                markdown = yaml_provenance_header() + markdown
                Path(dest).write_text(markdown, encoding="utf-8")
            except Exception as exc:  # provenance must never break a conversion
                print(
                    "[meta] %s: skipped %s provenance — %s"
                    % (src.name, output_format, exc),
                    file=sys.stderr,
                )

        record.output_path = str(dest)
        # Web (MHTML / single-file HTML) fidelity is noisier than PDF/DOCX —
        # SPA chrome and per-token code markup tank the char-diff — so cap its bar
        # at the web threshold even when a stricter global --threshold is set.
        eff_threshold = similarity_threshold
        if fmt in ("mhtml", "html"):
            eff_threshold = min(similarity_threshold, DEFAULT_WEB_SIMILARITY_THRESHOLD)
        result = validate(
            markdown,
            original_plaintext,
            similarity_threshold=eff_threshold,
            confidence=confidence,
            min_confidence=min_confidence,
            output_format=output_format,
            source_xml=source_xml,
        )
        record.similarity = result["similarity"]  # type: ignore[assignment]
        record.similarity_method = result["similarity_method"]  # type: ignore[assignment]
        record.content_recall = result["content_recall"]  # type: ignore[assignment]
        record.similarity_ok = result["similarity_ok"]  # type: ignore[assignment]
        record.confidence_ok = result["confidence_ok"]  # type: ignore[assignment]
        record.structural = result["structural"]  # type: ignore[assignment]
        record.structural_ok = result["structural_ok"]  # type: ignore[assignment]
        record.structural_reason = result["structural_reason"]  # type: ignore[assignment]
        record.fidelity_ok = result["fidelity_ok"]  # type: ignore[assignment]
        record.fidelity_reason = result["fidelity_reason"]  # type: ignore[assignment]
        record.passed = result["passed"]  # type: ignore[assignment]
        record.status = "converted"
    except Exception as exc:  # noqa: BLE001 - record any failure, keep batch going
        record.status = "error"
        record.error = str(exc)

    return record


def _figure_counts(visuals: List[Visual]) -> Dict[str, int]:
    counts = {"diagrams": 0, "images": 0, "complex_tables": 0}
    for v in visuals:
        if v.kind == "diagram":
            counts["diagrams"] += 1
        elif v.kind == "image":
            counts["images"] += 1
        elif v.kind == "table":
            counts["complex_tables"] += 1
    return counts


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
            "figures_extracted": sum(sum(r.figures.values()) for r in converted),
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
    figs = sum(sum(r.figures.values()) for r in converted)

    print("\n" + "=" * 60)
    print("doc2md conversion summary")
    print("=" * 60)
    print("Total files seen     : %d" % len(records))
    print("Converted            : %d" % len(converted))
    print("  Passed validation  : %d" % len(passed))
    print("  Failed validation  : %d" % len(failed))
    print("Skipped (unsupported): %d" % len(skipped))
    print("Errored              : %d" % len(errored))
    print("Figures extracted    : %d" % figs)

    scored = [r for r in converted if r.similarity is not None]
    if scored:
        scored.sort(key=lambda r: r.similarity)  # type: ignore[arg-type,return-value]
        print("\nLowest-scoring files (similarity):")
        for r in scored[:top_n]:
            flag = "FAIL" if r.passed is False else "ok"
            conf = "" if r.pdfmux_confidence is None else " conf=%.2f" % r.pdfmux_confidence
            method = " (%s)" % r.similarity_method if r.similarity_method == "containment" else ""
            recall = "" if r.content_recall is None else " recall=%.2f" % r.content_recall
            print("  %-6s %.3f%s%s%s  %s" % (flag, r.similarity, conf, method, recall, r.filename))

    for r in failed:
        reasons = []
        if r.similarity_ok is False:
            reasons.append("low similarity %.3f" % (r.similarity or 0.0))
        if r.confidence_ok is False:
            reasons.append("low confidence %.2f" % (r.pdfmux_confidence or 0.0))
        if r.structural_ok is False:
            reasons.append(r.structural_reason or "structural")
        if r.fidelity_ok is False:
            reasons.append(r.fidelity_reason or "YAML fidelity")
        if reasons:
            print("  FAIL %s — %s" % (r.filename, "; ".join(reasons)))

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


def resolve_input(input_path: Path) -> Tuple[Path, List[Path]]:
    """Return ``(input_root, files)`` for a file or directory input.

    For a single file, the root is its parent (so output mirrors just the
    filename). For a directory, all files beneath it are discovered.
    """
    if input_path.is_file():
        return input_path.parent, [input_path]
    return input_path, discover_files(input_path)


def _worker_init(ocr_mode: str, pdfmux_available: bool) -> None:
    """Re-establish the in-process global setup that ``spawn`` workers don't inherit.

    Each worker is a fresh interpreter. Environment variables set in :func:`main`
    (``PDFMUX_*``, ``TQDM_DISABLE``, …) carry over because the OS inherits the
    parent's environment at spawn time — but in-memory state does not. The OCR
    monkey-patches (:func:`install_ocr_policy`) and the third-party log filters
    (:func:`quiet_third_party_logs`) live only in memory, so without replaying
    them here a worker would run full-page OCR on born-digital PDFs (the slow
    path the policy exists to avoid) and leak backend INFO chatter. Both helpers
    are idempotent and never raise, so a worker init can't break the pool.
    """
    quiet_third_party_logs()
    if pdfmux_available and ocr_mode != "on":
        install_ocr_policy(ocr_mode)


def _crash_record(src: Path, detail: str) -> FileRecord:
    """A failure record for a file whose worker process died (native crash)."""
    rec = FileRecord(filename=src.name, source_path=str(src))
    rec.status = "error"
    rec.passed = False
    rec.error = "worker process crashed (likely a native PyMuPDF/MuPDF segfault): %s" % detail
    return rec


def _run_one_isolated(
    task: Callable[[Path], FileRecord], src: Path, ctx, init_args: Tuple[str, bool]
) -> FileRecord:
    """Run one file in its own short-lived process — perfect crash isolation.

    Used to finish survivors after a parallel worker dies (and for ``--workers
    1``). One process at a time means no two native MuPDF regions ever run
    concurrently, so the thread/concurrency race can't recur; a file that *still*
    crashes alone is genuinely poison and gets recorded as an error rather than
    aborting the batch.
    """
    with ProcessPoolExecutor(
        max_workers=1, mp_context=ctx,
        initializer=_worker_init, initargs=init_args,
    ) as pool:
        fut = pool.submit(task, src)
        try:
            return fut.result()
        except BrokenProcessPool:
            return _crash_record(src, "crashed when run in isolation")
        except Exception as exc:  # process_file is meant never to raise; be safe
            return _crash_record(src, repr(exc))


def _run_batch_isolated(
    files: List[Path],
    task: Callable[[Path], FileRecord],
    workers: int,
    init_args: Tuple[str, bool],
) -> Dict[Path, FileRecord]:
    """Convert every file across worker processes, surviving native crashes.

    PyMuPDF/MuPDF and pdfmux's native extractors are not thread-safe and only
    loosely process-safe: concurrent native work (or a malformed page) can hard
    crash a worker with SIGSEGV. Each file runs in its own ``spawn`` process so
    that native state is never shared; when a worker dies the pool raises
    :class:`BrokenProcessPool`. We salvage the results that already landed, then
    finish the survivors **serially** (:func:`_run_one_isolated`) so the
    concurrency race can't recur and one poison document can never abort the run.
    """
    ctx = multiprocessing.get_context("spawn")
    results: Dict[Path, FileRecord] = {}
    parallel = max(1, workers)

    if parallel > 1:
        with ProcessPoolExecutor(
            max_workers=parallel, mp_context=ctx,
            initializer=_worker_init, initargs=init_args,
        ) as pool:
            fut_to_file = {pool.submit(task, f): f for f in files}
            try:
                for fut in as_completed(fut_to_file):
                    f = fut_to_file[fut]
                    try:
                        results[f] = fut.result()
                    except BrokenProcessPool:
                        break  # pool is dead; salvage + recover below
                    except Exception as exc:
                        results[f] = _crash_record(f, repr(exc))
            except BrokenProcessPool:
                pass
        # Salvage any futures that finished before / alongside the crash.
        for fut, f in fut_to_file.items():
            if f in results or not fut.done() or fut.cancelled():
                continue
            try:
                results[f] = fut.result()
            except Exception:
                pass  # broken/crashed → handled as a survivor below

    survivors = [f for f in files if f not in results]
    if survivors and parallel > 1:
        logging.getLogger("doc2md").warning(
            "a worker process crashed (likely a native PyMuPDF/MuPDF segfault); "
            "finishing %d remaining file(s) one at a time",
            len(survivors),
        )
    for f in survivors:
        results[f] = _run_one_isolated(task, f, ctx, init_args)
    return results


def run_batch(
    input_path: Path,
    output_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    quality: str = "standard",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    preview_pages: Optional[int] = None,
    visual_cfg: Optional[VisualConfig] = None,
    xml_mode: str = "auto",
    yaml_index: bool = False,
    ocr_mode: str = "auto",
    large_doc_pages: int = DEFAULT_LARGE_DOC_PAGES,
    clean: bool = True,
    strip_patterns: Optional[List[str]] = None,
    used_llm: bool = False,
    rag_metadata: bool = True,
    source_abspath: bool = False,
    deps: Optional[Dict[str, bool]] = None,
    csv_list_min_segments: int = DEFAULT_CSV_LIST_MIN_SEGMENTS,
    csv_list_max_segment_len: int = DEFAULT_CSV_LIST_MAX_SEGMENT_LEN,
    csv_list_columns: Optional[List[str]] = None,
    csv_skip_columns: Optional[List[str]] = None,
    csv_title_column: Optional[str] = None,
    split_bytes: Optional[int] = None,
) -> List[FileRecord]:
    """Convert + validate a file or every file under a directory, in parallel.

    The output directory is excluded from discovery, so a default output nested
    inside the input (e.g. ``<input>/markdown/``) is never reprocessed on a
    re-run.
    """
    if deps is None:
        deps = check_dependencies()

    input_root, discovered = resolve_input(input_path)
    out_resolved = output_dir.resolve()
    files = [
        f
        for f in discovered
        if f.name != "conversion_report.json"
        and out_resolved not in f.resolve().parents
    ]

    records: List[FileRecord] = []
    if not files:
        return records

    # Each file runs in its own process (spawn). PyMuPDF/MuPDF and pdfmux's
    # native extractors are not thread-safe and only loosely process-safe — a
    # threaded pool corrupts MuPDF's shared C state and hard-crashes the
    # interpreter with a SIGSEGV. Process isolation keeps that state per-file and
    # lets us survive a worker dying (see :func:`_run_batch_isolated`).
    task = functools.partial(
        process_file,
        input_root=input_root,
        output_dir=output_dir,
        deps=deps,
        quality=quality,
        similarity_threshold=similarity_threshold,
        min_confidence=min_confidence,
        preview_pages=preview_pages,
        visual_cfg=visual_cfg,
        xml_mode=xml_mode,
        yaml_index=yaml_index,
        ocr_mode=ocr_mode,
        large_doc_pages=large_doc_pages,
        clean=clean,
        strip_patterns=strip_patterns,
        used_llm=used_llm,
        rag_metadata=rag_metadata,
        source_abspath=source_abspath,
        csv_list_min_segments=csv_list_min_segments,
        csv_list_max_segment_len=csv_list_max_segment_len,
        csv_list_columns=csv_list_columns,
        csv_skip_columns=csv_skip_columns,
        csv_title_column=csv_title_column,
        split_bytes=split_bytes,
    )
    init_args = (ocr_mode, bool(deps.get("pdfmux")))
    results = _run_batch_isolated(files, task, workers, init_args)

    records = list(results.values())
    records.sort(key=lambda r: r.source_path)
    return records


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="doc2md.py",
        description="Batch-convert documents to Markdown with figure/table "
        "extraction, quality validation, and a summary report.",
    )
    parser.add_argument(
        "input_path", type=Path,
        help="A document file or a directory of documents.",
    )
    parser.add_argument(
        "output_dir", type=Path, nargs="?", default=None,
        help="Where to write .md + figures + report. Defaults to a 'markdown' "
        "folder at the documents' path (inside the input dir, or next to a "
        "single input file).",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=DEFAULT_WORKERS,
        help="Parallel worker count (default: %d)" % DEFAULT_WORKERS,
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Similarity threshold for passing validation (default: %.2f)"
        % DEFAULT_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum pdfmux confidence to pass (default: %.2f)" % DEFAULT_MIN_CONFIDENCE,
    )
    parser.add_argument(
        "-q", "--quality", choices=("auto", "fast", "standard", "high"), default="auto",
        help="pdfmux local extraction quality. 'auto' (default): 'standard' "
        "(max local effort), but 'fast' for very large PDFs (> --large-doc-pages) "
        "where Docling's per-page cost isn't worth it. 'fast' = PyMuPDF only; "
        "'high' implies cloud LLM — prefer --llm for that.",
    )
    parser.add_argument(
        "--large-doc-pages", type=int, default=DEFAULT_LARGE_DOC_PAGES, metavar="N",
        help="With --quality=auto, PDFs larger than N pages use 'fast' instead of "
        "'standard' (default: %d)." % DEFAULT_LARGE_DOC_PAGES,
    )
    parser.add_argument(
        "--timeout", type=int, default=None, metavar="SECONDS",
        help="Per-document extraction timeout (sets PDFMUX_TIMEOUT). Omitted: "
        "auto-scale by page count (max %ds, %ds/page) so large docs don't hit "
        "pdfmux's 300s default. 0 = no limit." % (TIMEOUT_FLOOR, TIMEOUT_PER_PAGE_BUDGET),
    )
    parser.add_argument(
        "--ocr", choices=("auto", "on", "off"), default="auto",
        help="Docling OCR control for PDF table extraction. 'auto' (default) "
        "disables OCR on born-digital PDFs — much faster, no fidelity loss since "
        "the text layer is authoritative — and keeps it on for scanned/image "
        "PDFs. 'on' forces OCR on every page (pdfmux's stock behavior); 'off' "
        "disables it for all PDFs.",
    )
    parser.add_argument(
        "--llm", metavar="PROVIDER", choices=("gemini", "claude", "openai", "ollama"),
        default=None,
        help="Enable cloud/local LLM fallback for hard pages (off by default). "
        "Use only for documents local extraction can't handle.",
    )
    parser.add_argument(
        "--llm-budget", type=float, default=None, metavar="USD",
        help="Per-document spend cap when --llm is set.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Disable all figure/table image extraction (text-only Markdown).",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Disable the default Markdown cleanup. By default doc2md cleans per "
        "format — PDF: safe page-number/date furniture, empty duplicate tables, "
        "line-wrap token-split repair; MHTML: single-page-app chrome (data-URI "
        "icons, div/span scaffolding); docx/web: whitespace + --strip-line — each "
        "verified lossless (falls back to raw if not). Use this for raw output.",
    )
    parser.add_argument(
        "--strip-line", action="append", metavar="REGEX", default=None,
        help="Remove lines that fully match REGEX during cleanup (all formats) — "
        "for document/corpus-specific header/footer chrome that can't be detected "
        "automatically (e.g. 'Acme Corp, Inc\\.', 'Confidential.*'). Repeatable. "
        "Matched against the whole stripped line; still verified lossless.",
    )
    parser.add_argument(
        "--vector-diagrams", action="store_true",
        help="Also extract vector diagrams to PNG (best-effort, OFF by default). "
        "On PDFs exported from Google Docs/Confluence this can mis-fire on "
        "tables/TOC/badge pages; review the output.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=DEFAULT_FIGURE_DPI,
        help="Render DPI for extracted PNGs (default: %d)" % DEFAULT_FIGURE_DPI,
    )
    parser.add_argument(
        "--xml-mode", choices=("auto", "verbatim", "transform", "yaml", "rag"),
        default="auto",
        help="XML handling: 'verbatim' (fenced, lossless — for syntax-critical "
        "config), 'transform' (structured Markdown — for documentation XML), "
        "'yaml' (structure-preserving YAML to a .yaml file — low-token, parser- "
        "friendly for LLM/RAG ingestion), 'rag' (flattened '.rag.txt' index for "
        "chunk-based retrieval — see --rag), or 'auto' (default: detect by "
        "content).",
    )
    parser.add_argument(
        "--yaml", action="store_true",
        help="Shorthand for --xml-mode=yaml: emit XML inputs as YAML (.yaml) "
        "instead of Markdown. Applies to XML only; other formats are unaffected. "
        "Takes precedence over --xml-mode.",
    )
    parser.add_argument(
        "--yaml-index", action="store_true",
        help="Implies --yaml, and prefixes each nested block with a '# path: ...' "
        "structural breadcrumb so RAG ingesters (e.g. NotebookLM) can locate "
        "deeply-nested blocks. Comments don't change the parsed data.",
    )
    parser.add_argument(
        "--rag", action="store_true",
        help="Convert config XML to a RAG-optimized flat index ('.rag.txt'): a "
        "deterministic summary (origins + variables) plus one fully "
        "path-qualified 'a > b > c = value' line per leaf, so every line "
        "survives chunk-based retrieval (e.g. NotebookLM). NOT valid YAML and "
        "not round-trippable — use --yaml for the structured form. XML only.",
    )
    parser.add_argument(
        "--no-rag-metadata", action="store_true",
        help="Disable RAG provenance metadata (on by default): the YAML "
        "frontmatter block on every .md output, and '<!-- doc2md:page=N -->' "
        "page-boundary markers in PDF output where the extractor exposes per-page "
        "text (omitted when it only returns a single combined blob). Both are "
        "invisible to humans and stripped before the fidelity check; turn off if "
        "a downstream tool can't tolerate frontmatter or HTML comments.",
    )
    parser.add_argument(
        "--source-abspath", action="store_true",
        help="Emit the frontmatter 'source_path' as an absolute path instead of "
        "relative to the input root. Off by default — relative paths avoid "
        "leaking machine/folder names into a shared Markdown corpus.",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Quick sanity check: process only the first %d pages of each PDF."
        % DEFAULT_PREVIEW_PAGES,
    )
    parser.add_argument(
        "--preview-pages", type=int, default=DEFAULT_PREVIEW_PAGES,
        help="Pages to use with --preview (default: %d)" % DEFAULT_PREVIEW_PAGES,
    )

    csv_grp = parser.add_argument_group("CSV options")
    csv_grp.add_argument(
        "--csv-list-min-segments", type=int, default=DEFAULT_CSV_LIST_MIN_SEGMENTS,
        metavar="N",
        help="Min comma-segments for a CSV column to be auto-detected as a list "
        "(default: %d). Raise to be more conservative." % DEFAULT_CSV_LIST_MIN_SEGMENTS,
    )
    csv_grp.add_argument(
        "--csv-list-max-segment-len", type=int, default=DEFAULT_CSV_LIST_MAX_SEGMENT_LEN,
        metavar="L",
        help="Max mean segment length (chars) to auto-detect a list column "
        "(default: %d). Lower = stricter." % DEFAULT_CSV_LIST_MAX_SEGMENT_LEN,
    )
    csv_grp.add_argument(
        "--csv-list-columns", default=None, metavar="COL,...",
        help="Force these CSV columns to be expanded as bullet lists regardless of "
        "the heuristic (comma-separated column names).",
    )
    csv_grp.add_argument(
        "--csv-skip-columns", default=None, metavar="COL,...",
        help="Omit these CSV columns from Markdown output entirely (comma-separated). "
        "Use for redundant columns, e.g. a count column when the list itself is kept.",
    )
    csv_grp.add_argument(
        "--csv-title-column", default=None, metavar="COL",
        help="Column to use as the card heading (auto-detected if not set: prefers "
        "columns named 'name'/'title', then first non-ID text column).",
    )
    csv_grp.add_argument(
        "--split-file", default=None, metavar="SIZE",
        help="Split CSV Markdown output into multiple files at card boundaries so "
        "each file stays under SIZE (e.g. 3MB, 512K). Files are named "
        "<stem>-part001.md, <stem>-part002.md, …",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_path = args.input_path
    if not input_path.exists():
        print("error: input path not found: %s" % input_path, file=sys.stderr)
        return 2

    # Default output: a "markdown" folder at the documents' path (inside the
    # input directory, or alongside a single input file).
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        base = input_path if input_path.is_dir() else input_path.parent
        output_dir = base / "markdown"

    output_dir.mkdir(parents=True, exist_ok=True)
    quiet_third_party_logs()  # before any backend imports, so env/filters take
    if args.rag:
        xml_mode = "rag"
    elif args.yaml or args.yaml_index:
        xml_mode = "yaml"
    else:
        xml_mode = args.xml_mode
    deps = check_dependencies(need_yaml=xml_mode in ("yaml", "rag"))
    if not any(deps.values()):
        print(
            "error: none of pdfmux / pandoc / pdftotext are available; nothing "
            "can be done. See the messages above.",
            file=sys.stderr,
        )
        return 1

    quality = args.quality
    used_llm = False
    if args.llm:
        # pdfmux reads these env vars; "high" routes hard pages to the LLM.
        os.environ["PDFMUX_LLM_PROVIDER"] = args.llm
        os.environ["PDFMUX_MODE"] = "premium"
        if args.llm_budget is not None:
            os.environ["PDFMUX_BUDGET"] = str(args.llm_budget)
        quality = "high"
        used_llm = True
        print("[info] LLM fallback enabled via provider '%s' (mode=premium)" % args.llm)

    # Size the extraction timeout before pdfmux is imported (its pipeline reads
    # PDFMUX_TIMEOUT at import time). Auto-scale by the largest input PDF so big
    # docs don't die at pdfmux's 300s default mid-extraction.
    if deps.get("pdfmux"):
        max_pages = 0
        try:
            _, _discovered = resolve_input(input_path)
            max_pages = max(
                (_pdf_page_count(f) for f in _discovered if f.suffix.lower() == ".pdf"),
                default=0,
            )
        except Exception:
            max_pages = 0
        timeout_s = resolve_timeout(args.timeout, max_pages)
        os.environ["PDFMUX_TIMEOUT"] = str(timeout_s)
        print(
            "[info] extraction timeout: %ds%s"
            % (timeout_s, (" (largest input ~%d pages)" % max_pages) if max_pages else ""),
            file=sys.stderr,
        )

    # Decide Docling/pymupdf4llm OCR behavior before any PDF is processed. Skips
    # wasted full-page OCR on born-digital PDFs (the common slow/erroring case)
    # while leaving it on for scanned docs. No-op when pdfmux is absent or
    # --ocr=on.
    if deps.get("pdfmux") and args.ocr != "on":
        install_ocr_policy(args.ocr)

    visual_cfg = VisualConfig(
        enabled=not args.no_figures,
        dpi=args.figure_dpi,
        extract_diagrams=args.vector_diagrams,
    )
    preview_pages = args.preview_pages if args.preview else None

    csv_list_cols = (
        [c.strip() for c in args.csv_list_columns.split(",")]
        if args.csv_list_columns else None
    )
    csv_skip_cols = (
        [c.strip() for c in args.csv_skip_columns.split(",")]
        if args.csv_skip_columns else None
    )
    split_bytes: Optional[int] = None
    if args.split_file:
        try:
            split_bytes = _parse_split_size(args.split_file)
        except ValueError as exc:
            print("error: --split-file: %s" % exc, file=sys.stderr)
            return 2

    records = run_batch(
        input_path,
        output_dir,
        workers=args.workers,
        quality=quality,
        similarity_threshold=args.threshold,
        min_confidence=args.min_confidence,
        preview_pages=preview_pages,
        visual_cfg=visual_cfg,
        xml_mode=xml_mode,
        yaml_index=args.yaml_index,
        ocr_mode=args.ocr,
        large_doc_pages=args.large_doc_pages,
        clean=not args.no_clean,
        strip_patterns=args.strip_line,
        used_llm=used_llm,
        rag_metadata=not args.no_rag_metadata,
        source_abspath=args.source_abspath,
        deps=deps,
        csv_list_min_segments=args.csv_list_min_segments,
        csv_list_max_segment_len=args.csv_list_max_segment_len,
        csv_list_columns=csv_list_cols,
        csv_skip_columns=csv_skip_cols,
        csv_title_column=args.csv_title_column,
        split_bytes=split_bytes,
    )

    report_path = write_report(records, output_dir)
    print_summary(records)
    print("Report written to: %s" % report_path)

    bad = sum(1 for r in records if r.status == "error" or r.passed is False)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
