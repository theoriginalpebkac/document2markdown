"""Regression test for the layout-analyzer neutralization in the visual pass.

PyMuPDF 1.28+ ships an ONNX "layout" analyzer that pdfmux/pymupdf4llm installs
globally as ``pymupdf._get_layout`` (importing ``pymupdf4llm`` is enough). Once
installed, ``Page.find_tables()`` routes through that native path, which
segfaults inside MuPDF on some PDFs — an uncatchable crash. doc2md's figure pass
must therefore disable the analyzer around its own ``find_tables()`` call and
restore it afterward, so the deterministic line-based detector is used instead.

These tests pin that save/disable/restore contract without needing the heavy
ONNX model or a crashing fixture.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

pymupdf = pytest.importorskip("pymupdf")


def test_neutralizes_and_restores_layout_hook():
    """Inside the context the hook is None; afterward the original is restored."""
    if not hasattr(pymupdf, "_get_layout"):
        pytest.skip("this PyMuPDF build has no layout hook")
    sentinel = object()  # stand in for the installed ONNX layout callable
    saved = pymupdf._get_layout
    try:
        pymupdf._get_layout = sentinel
        with doc2md._classic_table_detection():
            assert pymupdf._get_layout is None, "layout analyzer must be disabled inside"
        assert pymupdf._get_layout is sentinel, "original hook must be restored"
    finally:
        pymupdf._get_layout = saved


def test_restores_layout_hook_on_exception():
    """The hook is restored even if the wrapped find_tables() raises."""
    if not hasattr(pymupdf, "_get_layout"):
        pytest.skip("this PyMuPDF build has no layout hook")
    sentinel = object()
    saved = pymupdf._get_layout
    try:
        pymupdf._get_layout = sentinel
        with pytest.raises(ValueError):
            with doc2md._classic_table_detection():
                raise ValueError("boom")
        assert pymupdf._get_layout is sentinel
    finally:
        pymupdf._get_layout = saved
