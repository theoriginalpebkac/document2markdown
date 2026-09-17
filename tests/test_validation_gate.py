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
