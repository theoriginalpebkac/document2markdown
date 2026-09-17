"""Tests for the fidelity-gate rule that closes the Docling silent-drop hole.

``validate`` lets a *confident* extractor vouch for a low ordered char-diff
similarity, because faithful conversions reflow content (tabular/multi-column
PDFs, HTML→Markdown) and tank that flat cross-check. But pdfmux's single-blob
``process()`` paths — Docling tables (``"process-tables"``), the LLM path
(``"process-llm"``) and the streaming-unavailable fallback (``"process-
fallback"``) — return only the text the extractor emitted, and its confidence
scores just that text: it is blind to a block the layout model silently dropped
(observed live, where a Docling extraction discarded a 22-item bullet list yet
reported confidence 1.0).

So on those paths a low order-insensitive **content recall** (real missing
tokens, immune to reflow) must fail the document *regardless* of confidence,
while the faithful per-page ``"streaming"`` path and non-PDF inputs keep the
lenient rule (a low recall there is usually a pdftotext line-wrap artifact).
These tests pin that split without touching pdfmux or any real PDF.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

# 100 distinct source tokens; the "lossy" Markdown keeps only the first 70, so
# order-insensitive content recall is ~0.70 (well under the 0.90 containment
# gate) and the ordered similarity is also below the 0.90 threshold.
_SOURCE = " ".join("word%02d" % i for i in range(100))
_LOSSY_MD = " ".join("word%02d" % i for i in range(70))
_COMPLETE_MD = _SOURCE


def _validate(md, source):
    return doc2md.validate(
        md,
        _SOURCE,
        confidence=1.0,  # maximally confident extractor
        min_confidence=doc2md.DEFAULT_MIN_CONFIDENCE,
        extraction_source=source,
    )


@pytest.mark.parametrize("source", ["process-tables", "process-llm", "process-fallback"])
def test_low_recall_fails_on_process_blob_paths_despite_confidence(source):
    """A dropped block (low recall) is fatal on every single-blob process() path,
    even at confidence 1.0 — the hole that let the Docling drop ship."""
    result = _validate(_LOSSY_MD, source)
    assert result["content_recall"] < doc2md.DEFAULT_CONTAINMENT_THRESHOLD
    assert result["confidence_ok"] is True  # confidence would have vouched before
    assert result["passed"] is False


def test_low_recall_still_lenient_on_streaming_path():
    """The faithful per-page streaming path keeps the old rule: high confidence
    vouches for a low ordered score (its low recall is usually a reflow artifact),
    so the document still passes."""
    result = _validate(_LOSSY_MD, "streaming")
    assert result["similarity_ok"] is False
    assert result["confidence_ok"] is True
    assert result["passed"] is True


def test_non_pdf_input_unaffected():
    """Non-PDF inputs (no extraction_source) keep the lenient rule."""
    result = _validate(_LOSSY_MD, None)
    assert result["confidence_ok"] is True
    assert result["passed"] is True


def test_complete_docling_extraction_still_passes():
    """The tightening must not create false failures: a Docling extraction with
    full content recall passes even if the ordered char-diff is low (reflow)."""
    result = _validate(_COMPLETE_MD, "process-tables")
    assert result["content_recall"] >= doc2md.DEFAULT_CONTAINMENT_THRESHOLD
    assert result["passed"] is True


# --------------------------------------------------------------------------- #
# convert_pdf_with_fallback: auto-recover from a silent content drop
# --------------------------------------------------------------------------- #

DUMMY_SRC = pathlib.Path("/nonexistent/doc.pdf")


def _conv(markdown, page_source):
    return doc2md.PdfConversion(
        markdown=markdown,
        confidence=1.0,
        min_page_confidence=1.0,
        page_source=page_source,
    )


def _stub_convert_pdf(monkeypatch, dest, by_quality):
    """Patch ``convert_pdf`` to return a scripted PdfConversion per quality and
    write its Markdown to ``dest`` (like the real one)."""
    def fake(src, d, *, quality, **kw):
        conv = by_quality[quality]
        d.write_text(conv.markdown, encoding="utf-8")
        return conv
    monkeypatch.setattr(doc2md, "convert_pdf", fake)


def test_fallback_recovers_from_docling_drop(monkeypatch, tmp_path):
    """A Docling blob path below the recall gate re-extracts on the faithful
    path and returns the higher-recall output + a fallback decision."""
    dest = tmp_path / "out.md"
    _stub_convert_pdf(monkeypatch, dest, {
        "standard": _conv(_LOSSY_MD, "process-tables"),   # dropped content
        "fast": _conv(_COMPLETE_MD, "streaming"),          # faithful, complete
    })
    conv, decision = doc2md.convert_pdf_with_fallback(
        DUMMY_SRC, dest, quality="standard", reference_plaintext=_SOURCE,
    )
    assert conv.page_source == "streaming"
    assert conv.markdown == _COMPLETE_MD
    assert decision is not None and decision["setting"] == "extraction-fallback"
    assert dest.read_text() == _COMPLETE_MD  # chosen output is on disk


def test_no_fallback_when_recall_is_healthy(monkeypatch, tmp_path):
    """A complete Docling extraction is kept as-is (no needless re-extraction)."""
    dest = tmp_path / "out.md"
    calls = []
    def fake(src, d, *, quality, **kw):
        calls.append(quality)
        d.write_text(_COMPLETE_MD, encoding="utf-8")
        return _conv(_COMPLETE_MD, "process-tables")
    monkeypatch.setattr(doc2md, "convert_pdf", fake)
    conv, decision = doc2md.convert_pdf_with_fallback(
        DUMMY_SRC, dest, quality="standard", reference_plaintext=_SOURCE,
    )
    assert decision is None
    assert calls == ["standard"]  # never retried


def test_fallback_keeps_original_when_faithful_path_no_better(monkeypatch, tmp_path):
    """If the faithful path isn't more complete (e.g. streaming unavailable, or
    image-only loss), keep the original output and let the gate report on it."""
    dest = tmp_path / "out.md"
    _stub_convert_pdf(monkeypatch, dest, {
        "standard": _conv(_LOSSY_MD, "process-tables"),
        "fast": _conv(_LOSSY_MD, "process-fallback"),  # no improvement
    })
    conv, decision = doc2md.convert_pdf_with_fallback(
        DUMMY_SRC, dest, quality="standard", reference_plaintext=_SOURCE,
    )
    assert decision is None
    assert conv.page_source == "process-tables"
    assert dest.read_text() == _LOSSY_MD  # original restored to disk


def test_no_fallback_without_reference_text(monkeypatch, tmp_path):
    """Without a pdftotext baseline the drop can't be measured — no retry."""
    dest = tmp_path / "out.md"
    calls = []
    def fake(src, d, *, quality, **kw):
        calls.append(quality)
        d.write_text(_LOSSY_MD, encoding="utf-8")
        return _conv(_LOSSY_MD, "process-tables")
    monkeypatch.setattr(doc2md, "convert_pdf", fake)
    conv, decision = doc2md.convert_pdf_with_fallback(
        DUMMY_SRC, dest, quality="standard", reference_plaintext=None,
    )
    assert decision is None
    assert calls == ["standard"]
