#!/usr/bin/env python3
"""doc2md.py — batch-convert a folder of documents to Markdown with per-file
quality validation, visual (figure/table) extraction, and a summary report.

Conversion
    * PDF              -> pdfmux  (per-page self-healing extraction + confidence)
    * DOCX/HTML/EPUB/RTF -> pandoc

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
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

DEFAULT_WORKERS = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_MIN_CONFIDENCE = 0.70
DEFAULT_PREVIEW_PAGES = 3
DEFAULT_FIGURE_DPI = 150

# A document whose stripped plain-text is at least this many characters but has
# zero headings, code blocks, tables, or extracted figures is structurally
# suspicious.
STRUCT_MIN_CHARS = 3000

PANDOC_EXTENSIONS = {".docx", ".html", ".htm", ".epub", ".rtf", ".odt"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | PANDOC_EXTENSIONS


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
    min_page_confidence: Optional[float] = None
    structural: Dict[str, int] = field(default_factory=dict)
    figures: Dict[str, int] = field(default_factory=dict)
    structural_ok: Optional[bool] = None
    structural_reason: Optional[str] = None
    similarity_ok: Optional[bool] = None
    confidence_ok: Optional[bool] = None
    passed: Optional[bool] = None
    preview: bool = False
    used_llm: bool = False
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Dependency checks
# --------------------------------------------------------------------------- #


def check_dependencies() -> Dict[str, bool]:
    """Probe for the external tools / packages we rely on.

    Returns a mapping of dependency name -> availability. Prints actionable
    install instructions for anything missing rather than failing silently.
    """
    import importlib.util

    available = {
        "pdfmux": importlib.util.find_spec("pdfmux") is not None,
        "pymupdf": pymupdf is not None,
        "pandoc": shutil.which("pandoc") is not None,
        "pdftotext": shutil.which("pdftotext") is not None,
    }

    instructions = {
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

    for name, ok in available.items():
        if not ok:
            print("[warning] " + instructions[name], file=sys.stderr)

    return available


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


def extract_page_visuals(
    page,
    page_index: int,
    fig_dir: Path,
    rel_base: str,
    slug: str,
    doc_title: str,
    cfg: VisualConfig,
) -> List[Visual]:
    """Render diagrams, info images, and complex tables on one page to PNG.

    Pure with respect to pdfmux — takes a PyMuPDF page, so it can be tested
    against any PDF independently of the conversion pipeline. ``slug`` prefixes
    every filename so PNGs stay unique/traceable even if relocated, and
    ``doc_title`` is woven into alt text for decontextualized RAG chunks.
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
            table_objs = list(page.find_tables().tables)
        except Exception:
            table_objs = []
    table_rects = [(_rect(t.bbox) & page_rect) for t in table_objs]

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
) -> Tuple[str, List[Visual]]:
    """Interleave pdfmux per-page text with inline visual blocks for that page.

    ``pages_text`` is ``[(page_index, markdown_text), ...]`` from pdfmux. Figures
    land directly after the text of the page they came from, keeping them near
    their context for RAG. PNGs go in ``<slug>/figures/`` next to the .md, named
    ``<slug>-pNNN-<kind>NN.png`` (slug = the .md stem). ``doc_title`` is the
    original document name, used in figure alt text.
    """
    if not cfg.enabled or pymupdf is None:
        return "\n\n".join(t for _, t in pages_text), []

    slug = dest.stem  # already slugified by _output_path_for
    fig_dir = dest.parent / slug / "figures"
    rel_base = "%s/figures" % slug

    visuals_all: List[Visual] = []
    parts: List[str] = []
    doc = pymupdf.open(str(pdf_path))
    try:
        for page_index, text in pages_text:
            if text and text.strip():
                parts.append(text)
            if 0 <= page_index < doc.page_count:
                vis = extract_page_visuals(
                    doc[page_index], page_index, fig_dir, rel_base, slug, doc_title, cfg
                )
                if vis:
                    parts.append(render_visual_markdown(vis))
                    visuals_all.extend(vis)
    finally:
        doc.close()

    return "\n\n".join(p for p in parts if p), visuals_all


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


def _extract_pdf_pages(src: Path, quality: str) -> Tuple[List[Tuple[int, str]], Optional[float], Optional[float]]:
    """Run pdfmux and return per-page (index, text), doc confidence, min-page confidence.

    Prefers ``pdfmux.pipeline.process`` (gives per-page text + confidence). Falls
    back to public ``pdfmux.extract_text`` (single blob, no confidence) only if
    the internal module layout changed.
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


def convert_pdf(
    src: Path,
    dest: Path,
    *,
    quality: str = "standard",
    preview_pages: Optional[int] = None,
    visual_cfg: Optional[VisualConfig] = None,
) -> PdfConversion:
    """Convert a PDF to Markdown (pdfmux) + extract visuals (PyMuPDF), write ``dest``."""
    cfg = visual_cfg or VisualConfig()
    work_src = src
    tmpdir: Optional[tempfile.TemporaryDirectory] = None
    try:
        if preview_pages is not None:
            tmpdir = tempfile.TemporaryDirectory(prefix="doc2md_preview_")
            work_src = _slice_pdf_pages(src, preview_pages, Path(tmpdir.name))

        pages_text, confidence, min_conf = _extract_pdf_pages(work_src, quality)
        markdown, visuals = build_markdown_with_visuals(
            pages_text, work_src, dest, cfg, doc_title=src.stem
        )
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(markdown, encoding="utf-8")
    return PdfConversion(markdown=markdown, confidence=confidence, min_page_confidence=min_conf, visuals=visuals)


def convert_with_pandoc(src: Path, dest: Path) -> str:
    """Convert a non-PDF document to GFM Markdown with pandoc; return the text."""
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


def validate(
    markdown: str,
    original_plaintext: Optional[str],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    confidence: Optional[float] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> Dict[str, object]:
    """Run fidelity + confidence + structural checks and decide pass/fail.

    ``original_plaintext`` is ``None`` for non-PDF inputs (no pdftotext source),
    in which case the similarity check is skipped. ``confidence`` is pdfmux's
    document confidence (PDF only); below ``min_confidence`` fails the document.
    """
    counts = structural_counts(markdown)
    plain = strip_markdown(markdown)
    struct_ok, struct_reason = structural_check(counts, plain)

    similarity: Optional[float] = None
    sim_ok: Optional[bool] = None
    if original_plaintext is not None:
        similarity = similarity_ratio(original_plaintext, markdown)
        sim_ok = similarity >= similarity_threshold

    conf_ok: Optional[bool] = None
    if confidence is not None:
        conf_ok = confidence >= min_confidence

    passed = struct_ok and (sim_ok is not False) and (conf_ok is not False)

    return {
        "similarity": similarity,
        "similarity_ok": sim_ok,
        "confidence_ok": conf_ok,
        "structural": counts,
        "structural_ok": struct_ok,
        "structural_reason": struct_reason,
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
    used_llm: bool = False,
) -> FileRecord:
    """Convert + validate a single file. Never raises — errors go in the record."""
    record = FileRecord(
        filename=src.name,
        source_path=str(src),
        preview=preview_pages is not None,
        used_llm=used_llm,
    )
    ext = src.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        record.status = "skipped"
        record.error = "unsupported format: %s" % (ext or "<none>")
        return record

    dest = _output_path_for(src, input_root, output_dir)

    try:
        original_plaintext: Optional[str] = None
        confidence: Optional[float] = None

        if ext in PDF_EXTENSIONS:
            if not deps.get("pdfmux"):
                raise RuntimeError("pdfmux is not installed; cannot convert PDFs")
            record.converter = "pdfmux"
            conv = convert_pdf(
                src, dest, quality=quality, preview_pages=preview_pages, visual_cfg=visual_cfg
            )
            markdown = conv.markdown
            confidence = conv.confidence
            record.pdfmux_confidence = conv.confidence
            record.min_page_confidence = conv.min_page_confidence
            record.figures = _figure_counts(conv.visuals)
            if deps.get("pdftotext"):
                original_plaintext = extract_pdf_plaintext(src, last_page=preview_pages)
        else:
            if not deps.get("pandoc"):
                raise RuntimeError("pandoc is not installed; cannot convert %s" % ext)
            record.converter = "pandoc"
            markdown = convert_with_pandoc(src, dest)

        record.output_path = str(dest)
        result = validate(
            markdown,
            original_plaintext,
            similarity_threshold=similarity_threshold,
            confidence=confidence,
            min_confidence=min_confidence,
        )
        record.similarity = result["similarity"]  # type: ignore[assignment]
        record.similarity_ok = result["similarity_ok"]  # type: ignore[assignment]
        record.confidence_ok = result["confidence_ok"]  # type: ignore[assignment]
        record.structural = result["structural"]  # type: ignore[assignment]
        record.structural_ok = result["structural_ok"]  # type: ignore[assignment]
        record.structural_reason = result["structural_reason"]  # type: ignore[assignment]
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
            print("  %-6s %.3f%s  %s" % (flag, r.similarity, conf, r.filename))

    for r in failed:
        reasons = []
        if r.similarity_ok is False:
            reasons.append("low similarity %.3f" % (r.similarity or 0.0))
        if r.confidence_ok is False:
            reasons.append("low confidence %.2f" % (r.pdfmux_confidence or 0.0))
        if r.structural_ok is False:
            reasons.append(r.structural_reason or "structural")
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
    used_llm: bool = False,
    deps: Optional[Dict[str, bool]] = None,
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

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                process_file,
                f,
                input_root,
                output_dir,
                deps=deps,
                quality=quality,
                similarity_threshold=similarity_threshold,
                min_confidence=min_confidence,
                preview_pages=preview_pages,
                visual_cfg=visual_cfg,
                used_llm=used_llm,
            ): f
            for f in files
        }
        for fut in as_completed(futures):
            records.append(fut.result())

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
        "-q", "--quality", choices=("fast", "standard", "high"), default="standard",
        help="pdfmux local extraction quality; 'standard' is the max local effort "
        "(default). 'high' implies cloud LLM — prefer --llm for that.",
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
        "--preview", action="store_true",
        help="Quick sanity check: process only the first %d pages of each PDF."
        % DEFAULT_PREVIEW_PAGES,
    )
    parser.add_argument(
        "--preview-pages", type=int, default=DEFAULT_PREVIEW_PAGES,
        help="Pages to use with --preview (default: %d)" % DEFAULT_PREVIEW_PAGES,
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
    deps = check_dependencies()
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

    visual_cfg = VisualConfig(
        enabled=not args.no_figures,
        dpi=args.figure_dpi,
        extract_diagrams=args.vector_diagrams,
    )
    preview_pages = args.preview_pages if args.preview else None

    records = run_batch(
        input_path,
        output_dir,
        workers=args.workers,
        quality=quality,
        similarity_threshold=args.threshold,
        min_confidence=args.min_confidence,
        preview_pages=preview_pages,
        visual_cfg=visual_cfg,
        used_llm=used_llm,
        deps=deps,
    )

    report_path = write_report(records, output_dir)
    print_summary(records)
    print("Report written to: %s" % report_path)

    bad = sum(1 for r in records if r.status == "error" or r.passed is False)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
