"""Tests for the RAG provenance metadata layer: YAML frontmatter + page markers.

Two concerns are exercised:

* **Frontmatter** — :func:`doc2md.safe_yaml_string` must escape arbitrary text
  *and* freeze its YAML type, so YAML 1.1's coercion of bare scalars (the
  "Norway problem": ``NO`` -> ``False``; version truncation ``1.10`` -> ``1.1``)
  can never silently corrupt a string field. :func:`doc2md.build_frontmatter`
  must omit null fields, keep numerics bare, and stamp the source ``mtime``
  (deterministic) rather than wall-clock now.
* **Page markers** — :func:`doc2md._assemble_with_page_markers` must mark every
  page, rejoin a sentence split across a page break into contiguous prose, snap
  the marker to a sentence boundary (never mid-sentence, never inside a block),
  and fall back to an inline marker only in the unbounded drift-cap case. Both
  layers must be invisible to the fidelity comparison.

All tests are synthetic; no external corpus is touched.
"""

import datetime
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

yaml = pytest.importorskip("yaml")  # for the round-trip-parse assertions only


def _parse_frontmatter(block: str):
    """Parse a ``---\\n...\\n---`` frontmatter block into a dict (drop the fences)."""
    body = "\n".join(block.splitlines()[1:-1])
    return yaml.safe_load(body)


# --------------------------------------------------------------------------- #
# safe_yaml_string: escaping + YAML 1.1 type-coercion shield
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "value",
    [
        "NO",            # Norway problem: bare -> bool False
        "Yes",
        "off",
        "1.10",          # bare -> float 1.1 (version truncation)
        "0xFF",
        "Q3: Results",   # colon would break a bare scalar
        'she said "hi"',  # embedded quotes
        "line1\nline2",  # control char
        "café — naïve",  # non-ASCII must survive readable
        "",
    ],
)
def test_safe_yaml_string_roundtrips_as_string(value):
    encoded = doc2md.safe_yaml_string(value)
    # Parses back to the *exact* string, with the type frozen (never bool/float).
    assert yaml.safe_load(encoded) == value
    assert isinstance(yaml.safe_load(encoded), str)


def test_safe_yaml_string_keeps_unicode_readable():
    # ensure_ascii=False: no \uXXXX escape soup in the emitted frontmatter.
    assert "\\u" not in doc2md.safe_yaml_string("café")
    assert "café" in doc2md.safe_yaml_string("café")


# --------------------------------------------------------------------------- #
# build_frontmatter: field selection, numeric typing, deterministic date
# --------------------------------------------------------------------------- #

def test_frontmatter_full_pdf_fields():
    mtime = datetime.datetime(2021, 1, 2, 12, 0).timestamp()
    block = doc2md.build_frontmatter(
        title='Q3: Results "special" café',
        source_file="report.pdf",
        source_path="sub/report.pdf",
        fmt="pdf",
        engine="pdfmux",
        mtime=mtime,
        quality="standard",
        page_count=42,
        confidence=0.912345,
    )
    data = _parse_frontmatter(block)
    assert data["title"] == 'Q3: Results "special" café'
    assert data["source_file"] == "report.pdf"
    assert data["source_path"] == "sub/report.pdf"
    assert data["format"] == "pdf"
    assert data["engine"] == "pdfmux"
    assert data["quality"] == "standard"
    assert data["page_count"] == 42 and isinstance(data["page_count"], int)
    assert data["confidence"] == pytest.approx(0.9123)
    # converted is the *source mtime*, quoted -> a string, not a date object.
    assert data["converted"] == datetime.date.fromtimestamp(mtime).isoformat()
    assert isinstance(data["converted"], str)
    assert isinstance(data["doc2md_version"], str)


def test_frontmatter_omits_null_fields():
    block = doc2md.build_frontmatter(
        title="Plain Doc",
        source_file="a.docx",
        source_path="a.docx",
        fmt="docx",
        engine="pandoc(docx)",
        mtime=datetime.datetime(2022, 5, 6).timestamp(),
        quality=None,
        page_count=None,
        confidence=None,
    )
    data = _parse_frontmatter(block)
    assert "confidence" not in data  # null confidence never emitted
    assert "page_count" not in data
    assert "quality" not in data
    assert data["engine"] == "pandoc(docx)"


def test_frontmatter_norway_title_stays_string():
    block = doc2md.build_frontmatter(
        title="NO", source_file="no.pdf", source_path="no.pdf", fmt="pdf",
        engine="pdfmux", mtime=0.0,
    )
    assert _parse_frontmatter(block)["title"] == "NO"


# --------------------------------------------------------------------------- #
# Page-marker assembly
# --------------------------------------------------------------------------- #

def _units(*pages):
    """(text, ...) or (text, visual_md) tuples -> [(index, text, visual_md)]."""
    out = []
    for i, p in enumerate(pages):
        if isinstance(p, tuple):
            text, visual = p
        else:
            text, visual = p, ""
        out.append((i, text, visual))
    return out


def test_legacy_join_when_markers_off():
    body = doc2md._assemble_with_page_markers(
        _units("A", ("B", "FIG")), mark_pages=False
    )
    assert body == "A\n\nB\n\nFIG"
    assert "doc2md:page" not in body


def test_every_page_marked_including_blank():
    body = doc2md._assemble_with_page_markers(
        _units("Page one.", "   ", "Page three."), mark_pages=True
    )
    for n in (1, 2, 3):
        assert "<!-- doc2md:page=%d -->" % n in body


def test_sentence_bridge_is_contiguous_and_marker_snaps_after():
    body = doc2md._assemble_with_page_markers(
        _units("The quick brown fox jumps over the lazy",
               "dog and runs away. Next sentence here."),
        mark_pages=True,
    )
    # The bridged sentence reflows with a single space, no hard break / marker.
    assert "lazy dog and runs away." in body
    # page=2 marker is snapped to *after* the completed bridging sentence.
    assert body.index("away.") < body.index("doc2md:page=2") < body.index("Next sentence")


def test_bridge_dehyphenates_word_split_across_page_break():
    body = doc2md._assemble_with_page_markers(
        _units("The system reads its config-",
               "uration file at startup. Done."),
        mark_pages=True,
    )
    assert "configuration file at startup." in body
    assert "config- uration" not in body


def test_drift_cap_places_inline_marker_without_breaking_prose():
    body = doc2md._assemble_with_page_markers(
        _units("This starts a very long",
               "sentence that simply keeps going without any end"),
        mark_pages=True,
    )
    # No sentence boundary on page 2 -> marker inline at the exact boundary,
    # prose stays on one line (no blank-line hard break around the marker).
    assert "very long <!-- doc2md:page=2 --> sentence that simply" in body


def test_marker_not_snapped_into_a_table():
    body = doc2md._assemble_with_page_markers(
        _units("Some prose with no ending",
               "| a | b |\n| --- | --- |\n| 1 | 2 |"),
        mark_pages=True,
    )
    # Page 2 opens with a table (structural) -> no bridge; marker sits before it
    # on its own line, never between table rows.
    assert "<!-- doc2md:page=2 -->\n\n| a | b |" in body
    assert "| 1 |" in body.split("doc2md:page=2")[1]


def test_first_page_marker_has_no_leading_blank():
    body = doc2md._assemble_with_page_markers(_units("Hello."), mark_pages=True)
    assert body.startswith("<!-- doc2md:page=1 -->\n\nHello.")


# --------------------------------------------------------------------------- #
# Fidelity-path stripping
# --------------------------------------------------------------------------- #

def test_page_marker_regex_is_namespaced():
    assert doc2md._PAGE_MARKER_RE.sub("", "<!-- doc2md:page=5 -->") == ""
    # A genuine source comment is left untouched by the namespaced pattern.
    assert doc2md._PAGE_MARKER_RE.sub("", "<!-- real comment -->") == "<!-- real comment -->"
    # Stripping an inline marker leaves a separator (no word merge).
    assert "ab" not in doc2md._PAGE_MARKER_RE.sub("", "a <!-- doc2md:page=2 --> b")


def test_strip_markdown_removes_frontmatter_and_markers():
    fm = doc2md.build_frontmatter(
        title="Doc", source_file="d.pdf", source_path="d.pdf", fmt="pdf",
        engine="pdfmux", mtime=0.0, page_count=2, confidence=0.9,
    )
    md = fm + "\n\n<!-- doc2md:page=1 -->\n\nHello world.\n\n<!-- doc2md:page=2 -->\n\nMore text."
    stripped = doc2md.strip_markdown(md)
    assert "doc2md:page" not in stripped
    assert "doc2md_version" not in stripped  # frontmatter gone
    assert "Hello world." in stripped and "More text." in stripped


def test_frontmatter_does_not_change_under_a_later_horizontal_rule():
    # _FRONTMATTER_RE is anchored to the start; a real "---" rule mid-body stays.
    md = "Intro paragraph.\n\n---\n\nAfter the rule."
    assert doc2md.strip_markdown(md).count("After the rule") == 1
    assert "Intro paragraph." in doc2md.strip_markdown(md)


def test_metadata_does_not_affect_similarity_score():
    original = "Hello world. This is a sample document with several words in it."
    plain = "Hello world. This is a sample document with several words in it."
    fm = doc2md.build_frontmatter(
        title="Sample", source_file="s.pdf", source_path="s.pdf", fmt="pdf",
        engine="pdfmux", mtime=0.0, page_count=1, confidence=0.95,
    )
    with_meta = fm + "\n\n<!-- doc2md:page=1 -->\n\n" + plain
    assert doc2md.compute_similarity(original, plain) == doc2md.compute_similarity(
        original, with_meta
    )


# --------------------------------------------------------------------------- #
# Title derivation
# --------------------------------------------------------------------------- #

def test_derive_title_prefers_first_h1():
    title = doc2md._derive_title("# Real Title\n\nbody", pathlib.Path("the-file.docx"), "docx")
    assert title == "Real Title"


def test_derive_title_falls_back_to_deslugified_filename():
    title = doc2md._derive_title("no heading here", pathlib.Path("network_design-spec.docx"), "docx")
    assert title == "network design spec"
