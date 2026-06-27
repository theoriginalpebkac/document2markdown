#!/usr/bin/env python3
"""doc2md.py — batch-convert a folder of documents to Markdown with per-file
quality validation, visual (figure/table) extraction, and a summary report.

Conversion (handler chosen by content sniffing, not extension)
    * PDF                       -> pdfmux (per-page self-healing + confidence)
    * .docx (Word, Google Docs) -> pandoc (+ --extract-media)
    * Confluence "Word" export  -> MHTML: extract HTML + base64 images -> pandoc
    * config XML                -> verbatim (fenced, lossless, + index)
    * documentation XML         -> transform (structured Markdown)
    *   (or --yaml / --xml-mode=yaml: XML -> structure-preserving .yaml)
    * HTML/EPUB/RTF/ODT         -> pandoc

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

# Extensions worth *attempting*. The actual handler is chosen by content
# sniffing (detect_format), not the extension — e.g. a Confluence ".doc" is
# really MHTML, and ".xml" may be config or documentation.
PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx", ".doc", ".mht", ".mhtml"}
XML_EXTENSIONS = {".xml"}
PANDOC_EXTENSIONS = {".html", ".htm", ".epub", ".rtf", ".odt"}
SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS | WORD_EXTENSIONS | XML_EXTENSIONS | PANDOC_EXTENSIONS
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
    fidelity_ok: Optional[bool] = None  # --yaml: YAML round-trips to source XML
    fidelity_reason: Optional[str] = None
    passed: Optional[bool] = None
    preview: bool = False
    used_llm: bool = False
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

    if need_yaml:
        available["xmltodict"] = importlib.util.find_spec("xmltodict") is not None
        available["yaml"] = importlib.util.find_spec("yaml") is not None
        instructions["xmltodict"] = instructions["yaml"] = (
            "--yaml needs the 'xmltodict' and 'PyYAML' packages, which are not "
            "importable.\n"
            "    Install them with:  pip install -r requirements.txt"
        )

    seen: set = set()
    for name, ok in available.items():
        msg = instructions[name]
        if not ok and msg not in seen:  # yaml deps share one message -> warn once
            print("[warning] " + msg, file=sys.stderr)
            seen.add(msg)

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
# Format detection (content sniffing, not extension)
# --------------------------------------------------------------------------- #

_MIME_HEADER_RE = re.compile(
    r"(?is)^\s*(from|date|message-id|subject|mime-version|content-type)\s*:"
)


def detect_format(path: Path) -> str:
    """Classify a file by *content*, returning one of:

    ``pdf``, ``docx``, ``mhtml``, ``doc-binary``, ``xml``, ``pandoc``,
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
    if path.suffix.lower() in PANDOC_EXTENSIONS:
        return "pandoc"
    if path.suffix.lower() == ".docx":
        return "docx"
    return "unsupported"


# --------------------------------------------------------------------------- #
# Word / MHTML conversion (Confluence "Word" export is MHTML with base64 images)
# --------------------------------------------------------------------------- #


@dataclass
class WordConversion:
    markdown: str
    images: int
    ref_plaintext: Optional[str]  # source text for the fidelity check


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


def convert_word(
    src: Path,
    dest: Path,
    fmt: str,
    *,
    have_pandoc: bool,
    visual_cfg: Optional[VisualConfig] = None,
) -> WordConversion:
    """Convert a Word-family document (OOXML ``.docx`` or Confluence MHTML)."""
    cfg = visual_cfg or VisualConfig()
    slug = dest.stem
    fig_dir = dest.parent / slug / "figures"
    rel_base = "%s/figures" % slug
    dest.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "mhtml":
        html, parts = extract_mhtml(src)
        ref_plaintext = _strip_html(html)
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
        dest.write_text(markdown, encoding="utf-8")
        return WordConversion(markdown=markdown, images=n, ref_plaintext=ref_plaintext)

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
        dest.write_text(markdown, encoding="utf-8")
        return WordConversion(markdown=markdown, images=n, ref_plaintext=ref_plaintext)

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


def xml_to_yaml(raw: str) -> str:
    """Convert XML to YAML (structure-preserving, low-token, no Markdown wrapper).

    Uses ``xmltodict`` to build a dict — attributes become ``@name`` keys, element
    text ``#text``, and repeated siblings collapse to lists — then dumps YAML with
    document order preserved. Intended for configuration XML headed for LLM/RAG
    ingestion (e.g. NotebookLM, which won't parse XML fenced inside Markdown).

    Lossy for XML comments and processing instructions, and awkward for mixed
    prose+inline-tag content, so it is *not* used for documentation-style XML.

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
    parsed = xmltodict.parse(raw)
    return yaml.dump(parsed, default_flow_style=False, sort_keys=False, allow_unicode=True)


def convert_xml(src: Path, dest: Path, mode: str = "auto") -> Tuple[str, str, Path]:
    """Convert XML to Markdown or YAML. ``mode`` is auto | verbatim | transform | yaml.

    Returns ``(content, mode_used, dest_written)``. Verbatim is lossless and the
    safe default for unknown/config XML; transform is for documentation-style XML;
    yaml emits structure-preserving YAML to a sibling ``.yaml`` file (the output
    path differs from the ``.md`` ``dest`` passed in, hence the returned path).

    A ``yaml`` request on malformed XML falls back to lossless ``verbatim``
    Markdown; a missing optional dependency is a hard error.
    """
    import xml.parsers.expat

    raw = src.read_text(encoding="utf-8", errors="replace")
    chosen = xml_choose_mode(src, raw) if mode == "auto" else mode
    if chosen == "yaml":
        try:
            content = xml_to_yaml(raw)
        except xml.parsers.expat.ExpatError:
            chosen = "verbatim"  # malformed XML -> lossless Markdown fallback
        else:
            yaml_dest = dest.with_suffix(".yaml")
            yaml_dest.parent.mkdir(parents=True, exist_ok=True)
            yaml_dest.write_text(content, encoding="utf-8")
            return content, "yaml", yaml_dest
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


def yaml_fidelity_check(
    yaml_content: str, source_xml: Optional[str]
) -> Tuple[Optional[bool], Optional[str]]:
    """Verify ``--yaml`` output reproduces the source XML's structure exactly.

    Re-parses the emitted YAML and compares it to ``xmltodict``'s parse of the
    source XML. Because the conversion is ``yaml.dump(xmltodict.parse(xml))``, a
    faithful run must satisfy ``safe_load(yaml) == xmltodict.parse(xml)``; any
    mismatch means a bad parse/serialization path (e.g. a scalar that YAML
    re-reads as a bool/null), which this fails on rather than shipping silently.

    Returns ``(None, None)`` when no source is available (fidelity unassessable).
    """
    if source_xml is None:
        return None, None
    try:
        import xmltodict
        import yaml

        expected = xmltodict.parse(source_xml)
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
    confidence: Optional[float] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    output_format: str = "markdown",
    source_xml: Optional[str] = None,
) -> Dict[str, object]:
    """Run fidelity + confidence + structural checks and decide pass/fail.

    ``original_plaintext`` is ``None`` for non-PDF inputs (no pdftotext source),
    in which case the similarity check is skipped. ``confidence`` is pdfmux's
    document confidence (PDF only); below ``min_confidence`` fails the document.

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
    the document.
    """
    fidelity_ok: Optional[bool] = None
    fidelity_reason: Optional[str] = None
    if output_format == "yaml":
        counts, struct_ok, struct_reason = yaml_structural_check(markdown)
        fidelity_ok, fidelity_reason = yaml_fidelity_check(markdown, source_xml)
    else:
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

    # Structural emptiness is fatal only when fidelity is otherwise unverifiable.
    struct_fatal = (not struct_ok) and (sim_ok is None)
    passed = (
        (sim_ok is not False)
        and (conf_ok is not False)
        and (fidelity_ok is not False)
        and not struct_fatal
    )

    return {
        "similarity": similarity,
        "similarity_ok": sim_ok,
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
    used_llm: bool = False,
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

        if fmt == "pdf":
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
        elif fmt in ("mhtml", "docx"):
            record.converter = "mhtml" if fmt == "mhtml" else "pandoc(docx)"
            wc = convert_word(
                src, dest, fmt, have_pandoc=bool(deps.get("pandoc")), visual_cfg=visual_cfg
            )
            markdown = wc.markdown
            original_plaintext = wc.ref_plaintext
            record.figures = {"diagrams": 0, "images": wc.images, "complex_tables": 0}
        elif fmt == "xml":
            markdown, mode_used, dest = convert_xml(src, dest, mode=xml_mode)
            record.converter = "xml-%s" % mode_used
            output_format = "yaml" if mode_used == "yaml" else "markdown"
            if output_format == "yaml":
                source_xml = src.read_text(encoding="utf-8", errors="replace")
        elif fmt == "doc-binary":
            raise RuntimeError(
                "legacy binary .doc (OLE) needs LibreOffice/antiword; not supported. "
                "Re-save as .docx, or export from Confluence as PDF/Word(MHTML)."
            )
        else:  # pandoc: html/htm/epub/rtf/odt
            if not deps.get("pandoc"):
                raise RuntimeError("pandoc is not installed; cannot convert %s" % src.suffix)
            record.converter = "pandoc"
            markdown = convert_with_pandoc(src, dest)

        record.output_path = str(dest)
        result = validate(
            markdown,
            original_plaintext,
            similarity_threshold=similarity_threshold,
            confidence=confidence,
            min_confidence=min_confidence,
            output_format=output_format,
            source_xml=source_xml,
        )
        record.similarity = result["similarity"]  # type: ignore[assignment]
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
            print("  %-6s %.3f%s  %s" % (flag, r.similarity, conf, r.filename))

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
                xml_mode=xml_mode,
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
        "--xml-mode", choices=("auto", "verbatim", "transform", "yaml"), default="auto",
        help="XML handling: 'verbatim' (fenced, lossless — for syntax-critical "
        "config), 'transform' (structured Markdown — for documentation XML), "
        "'yaml' (structure-preserving YAML to a .yaml file — low-token, parser- "
        "friendly for LLM/RAG ingestion), or 'auto' (default: detect by content).",
    )
    parser.add_argument(
        "--yaml", action="store_true",
        help="Shorthand for --xml-mode=yaml: emit XML inputs as YAML (.yaml) "
        "instead of Markdown. Applies to XML only; other formats are unaffected. "
        "Takes precedence over --xml-mode.",
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
    xml_mode = "yaml" if args.yaml else args.xml_mode
    deps = check_dependencies(need_yaml=xml_mode == "yaml")
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
        xml_mode=xml_mode,
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
