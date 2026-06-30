"""Regression tests for per-page figure extraction.

The visual-extraction pass is a *PyMuPDF* concern (page geometry + rendering),
not a pdfmux one. It must iterate the PDF's real page range — not the page units
pdfmux's text extractor happened to return. pdfmux 1.6.4 collapses a document to
a single combined text blob, so a loop driven by that text once extracted figures
from page 0 only, silently dropping every figure on later pages
(:func:`doc2md.build_markdown_with_visuals`).

When there is no per-page text to interleave into, the figures are appended
grouped by page under a heading; when per-page text *is* available they are
interleaved after their page. Both are exercised here.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

pymupdf = pytest.importorskip("pymupdf")


def _pdf_with_images_on_pages(path, image_pages, n_pages):
    """Write an ``n_pages`` PDF with a raster image on each 0-based page in
    ``image_pages`` and prose on page 0."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 300))
    pix.set_rect(pymupdf.IRect(0, 0, 400, 300), (40, 120, 200))
    png = pix.tobytes("png")
    doc = pymupdf.open()
    for i in range(n_pages):
        pg = doc.new_page()
        if i == 0:
            pg.insert_textbox(pymupdf.Rect(72, 72, 520, 720), "Page one prose only.", fontsize=12)
        if i in image_pages:
            pg.insert_image(pymupdf.Rect(80, 80, 480, 380), stream=png)
    doc.save(str(path))
    doc.close()


def test_figures_extracted_from_non_first_pages(tmp_path):
    pdf = tmp_path / "figs.pdf"
    _pdf_with_images_on_pages(pdf, image_pages={1, 2}, n_pages=3)  # images on pages 2 & 3
    dest = tmp_path / "figs.md"

    # Single combined text blob, as pdfmux 1.6.4 returns.
    body, visuals = doc2md.build_markdown_with_visuals(
        [(0, "Page one prose only.")], pdf, dest, doc2md.VisualConfig(), "figs"
    )

    assert len(visuals) == 2  # was 0 before the fix
    assert {v.page_num for v in visuals} == {1, 2}
    # Grouped under a heading at the end, page-stamped for provenance.
    assert "## Figures & tables (by page)" in body
    assert "p.2" in body and "p.3" in body
    assert (tmp_path / "figs" / "figures" / "figs-p002-figure01.png").exists()
    assert (tmp_path / "figs" / "figures" / "figs-p003-figure01.png").exists()


def test_no_grouped_heading_when_no_figures(tmp_path):
    pdf = tmp_path / "plain.pdf"
    _pdf_with_images_on_pages(pdf, image_pages=set(), n_pages=2)
    dest = tmp_path / "plain.md"
    body, visuals = doc2md.build_markdown_with_visuals(
        [(0, "Just prose.")], pdf, dest, doc2md.VisualConfig(), "plain"
    )
    assert visuals == []
    assert "Figures & tables" not in body
    assert body == "Just prose."


def test_per_page_text_interleaves_visuals_inline(tmp_path):
    # Future path: when per-page text IS available, visuals land after their
    # page's text (no trailing grouped section).
    pdf = tmp_path / "inter.pdf"
    _pdf_with_images_on_pages(pdf, image_pages={1}, n_pages=2)  # image on page 2
    dest = tmp_path / "inter.md"
    pages_text = [(0, "First page text."), (1, "Second page text.")]
    body, visuals = doc2md.build_markdown_with_visuals(
        pages_text, pdf, dest, doc2md.VisualConfig(), "inter", mark_pages=True
    )
    assert len(visuals) == 1 and visuals[0].page_num == 1
    assert "## Figures & tables (by page)" not in body  # interleaved, not grouped
    # The figure block follows the second page's text, and page markers are woven in.
    assert body.index("Second page text.") < body.index("p.2")
    assert "<!-- doc2md:page=2 -->" in body


class _BboxRaisingTable:
    """Stand-in for a PyMuPDF ``Table`` whose ``bbox`` blows up.

    PyMuPDF's real ``Table.bbox`` raises ``ValueError("min() iterable argument
    is empty")`` on a degenerate detection with an empty cell list. We must skip
    it, not let it abort the whole document (the original crash).
    """

    @property
    def bbox(self):
        raise ValueError("min() iterable argument is empty")


class _FakeFindTables:
    def __init__(self, tables):
        self.tables = tables


def test_degenerate_table_is_skipped_and_counted(tmp_path, monkeypatch):
    pdf = tmp_path / "degen.pdf"
    _pdf_with_images_on_pages(pdf, image_pages=set(), n_pages=1)  # prose only
    doc = pymupdf.open(str(pdf))
    try:
        page = doc[0]
        # One table PyMuPDF "detected" but can't bound — exactly the crash case.
        monkeypatch.setattr(
            page, "find_tables", lambda *a, **k: _FakeFindTables([_BboxRaisingTable()])
        )
        stats: dict = {}
        cfg = doc2md.VisualConfig()  # tables on by default
        # Must not raise (was: ValueError aborting the conversion).
        visuals = doc2md.extract_page_visuals(
            page, 0, tmp_path / "figs", "figs/figures", "degen", "Degen", cfg, stats=stats
        )
    finally:
        doc.close()

    assert visuals == []  # degenerate table yields no image
    assert stats["degenerate_tables"] == 1  # but it is counted, not silently lost
