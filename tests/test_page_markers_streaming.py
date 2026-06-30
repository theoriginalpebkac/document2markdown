"""Tests for reviving RAG page markers off pdfmux 1.7.0's per-page streaming API.

``_extract_pdf_pages`` chooses between two extraction sources:

* ``process_streaming`` — real per-page text, so page markers + inline figure
  interleave activate (``source == "streaming"``).
* ``process()`` single blob — for the Docling-table route (``--quality
  standard`` + detected tables), the LLM route (``--quality high``), or when
  streaming is unavailable. Markers stay dormant there.

These tests pin the routing/guard dispatch, the per-page assembly from raw
stream events (gap-filling + confidence math), and the reportable decision —
all without touching pdfmux or any real PDF (the extractor seams are stubbed).
"""

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

DUMMY = pathlib.Path("/nonexistent/doc.pdf")


# --------------------------------------------------------------------------- #
# _extract_pdf_pages: routing + guard
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_extractors(monkeypatch):
    """Stub the two extraction seams so dispatch is observable without pdfmux.

    ``via_process`` returns a recognizable 3-tuple; ``via_streaming`` returns a
    different one (or ``None`` to simulate "streaming unavailable"). Returns a
    setter for the streaming return value.
    """
    proc_result = ([(0, "BLOB")], 0.8, 0.8)
    stream_result = {"value": ([(0, "PAGE0"), (1, "PAGE1")], 0.9, 0.7)}

    monkeypatch.setattr(doc2md, "_extract_pages_via_process", lambda s, q: proc_result)
    monkeypatch.setattr(
        doc2md, "_extract_pages_via_streaming", lambda s, q: stream_result["value"]
    )
    return stream_result


def test_high_quality_routes_to_process_llm(stub_extractors, monkeypatch):
    # The guard must not even be consulted on the LLM path.
    monkeypatch.setattr(
        doc2md, "_pdf_routes_to_docling", lambda s: pytest.fail("guard consulted")
    )
    pages, conf, min_conf, source = doc2md._extract_pdf_pages(DUMMY, "high")
    assert source == "process-llm"
    assert pages == [(0, "BLOB")]


def test_standard_with_tables_routes_to_process(stub_extractors, monkeypatch):
    monkeypatch.setattr(doc2md, "_pdf_routes_to_docling", lambda s: True)
    pages, conf, min_conf, source = doc2md._extract_pdf_pages(DUMMY, "standard")
    assert source == "process-tables"
    assert pages == [(0, "BLOB")]


def test_standard_without_tables_streams(stub_extractors, monkeypatch):
    monkeypatch.setattr(doc2md, "_pdf_routes_to_docling", lambda s: False)
    pages, conf, min_conf, source = doc2md._extract_pdf_pages(DUMMY, "standard")
    assert source == "streaming"
    assert pages == [(0, "PAGE0"), (1, "PAGE1")]
    assert (conf, min_conf) == (0.9, 0.7)


def test_fast_streams_without_consulting_guard(stub_extractors, monkeypatch):
    # Fast mode never engages Docling in process() either, so there's no fidelity
    # to protect — stream unconditionally (guard must not be called).
    monkeypatch.setattr(
        doc2md, "_pdf_routes_to_docling", lambda s: pytest.fail("guard consulted")
    )
    _, _, _, source = doc2md._extract_pdf_pages(DUMMY, "fast")
    assert source == "streaming"


def test_streaming_unavailable_falls_back_to_process(stub_extractors, monkeypatch):
    monkeypatch.setattr(doc2md, "_pdf_routes_to_docling", lambda s: False)
    stub_extractors["value"] = None  # simulate streaming import/runtime failure
    pages, _, _, source = doc2md._extract_pdf_pages(DUMMY, "standard")
    assert source == "process-fallback"
    assert pages == [(0, "BLOB")]


# --------------------------------------------------------------------------- #
# _extract_pages_via_streaming: event assembly
# --------------------------------------------------------------------------- #


def _install_fake_streaming(monkeypatch, events):
    """Inject a fake ``pdfmux.streaming.process_streaming`` yielding ``events``."""
    pkg = types.ModuleType("pdfmux")
    mod = types.ModuleType("pdfmux.streaming")

    def process_streaming(path, quality="standard", **kw):
        yield from events

    mod.process_streaming = process_streaming
    pkg.streaming = mod
    monkeypatch.setitem(sys.modules, "pdfmux", pkg)
    monkeypatch.setitem(sys.modules, "pdfmux.streaming", mod)


def _ev(kind, **data):
    return types.SimpleNamespace(type=kind, data=data)


def test_streaming_assembles_pages_in_order_with_confidence(monkeypatch):
    events = [
        _ev("classified", page_count=3, page_types=["digital"] * 3),
        # Out-of-order emission (good pages first, re-extracted later) must sort.
        _ev("page", page_num=2, text="Third", confidence=0.95),
        _ev("page", page_num=0, text="First", confidence=0.90),
        _ev("page", page_num=1, text="Second", confidence=0.70),
        _ev("complete", total_confidence=0.85, ocr_pages=[1], page_count=3),
    ]
    _install_fake_streaming(monkeypatch, events)
    pages, doc_conf, min_conf = doc2md._extract_pages_via_streaming(DUMMY, "standard")
    assert pages == [(0, "First"), (1, "Second"), (2, "Third")]
    assert doc_conf == 0.85  # from the complete event
    assert min_conf == 0.70  # min across per-page confidences


def test_streaming_fills_missing_pages_with_empty_text(monkeypatch):
    # page_count says 4 but only pages 0 and 3 emitted — gaps must be filled so
    # markers stay continuous and figure interleave aligns to PyMuPDF indices.
    events = [
        _ev("classified", page_count=4, page_types=["digital"] * 4),
        _ev("page", page_num=0, text="A", confidence=0.9),
        _ev("page", page_num=3, text="D", confidence=0.9),
        _ev("complete", total_confidence=0.9, page_count=4),
    ]
    _install_fake_streaming(monkeypatch, events)
    pages, _, _ = doc2md._extract_pages_via_streaming(DUMMY, "standard")
    assert [p[0] for p in pages] == [0, 1, 2, 3]
    assert pages[1] == (1, "") and pages[2] == (2, "")


def test_streaming_returns_none_when_no_pages(monkeypatch):
    events = [_ev("classified", page_count=0, page_types=[])]
    _install_fake_streaming(monkeypatch, events)
    assert doc2md._extract_pages_via_streaming(DUMMY, "standard") is None


# --------------------------------------------------------------------------- #
# _page_marker_decision: provenance reporting
# --------------------------------------------------------------------------- #


def test_decision_on_for_streaming():
    d = doc2md._page_marker_decision("streaming")
    assert d["setting"] == "page-markers" and d["choice"] == "on"


@pytest.mark.parametrize(
    "source", ["process-tables", "process-llm", "process-fallback"]
)
def test_decision_off_for_blob_sources(source):
    d = doc2md._page_marker_decision(source)
    assert d["choice"] == "off"
    assert d["reason"] and d["override"]


def test_decision_none_for_unknown_source():
    assert doc2md._page_marker_decision("something-else") is None
