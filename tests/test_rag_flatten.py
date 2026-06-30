"""Regression tests for the ``--rag`` flat-index conversion.

``--rag`` projects config XML into a deterministic DOCUMENT SUMMARY plus one
fully path-qualified ``a > b > c = value`` line per leaf (:func:`doc2md.xml_to_rag`).
Unlike ``--yaml`` it does *not* round-trip to the source structure, so fidelity
is enforced at the value level instead: every scalar leaf in the source XML must
appear in the output (:func:`doc2md.rag_fidelity_check`). The format-agnostic
properties covered here:

* every emitted body line is self-contained (carries its full path), so it
  survives a RAG chunk boundary;
* the node discriminator (``@value``/``@name``/``@result``/``name`` …) is folded
  into the path segment, *except* when it's a node's only content, where it must
  still be emitted as a leaf or the value is lost (the bug the fidelity gate caught);
* the deterministic summary block is present (its origin/variable *extraction* is
  vocabulary-specific and exercised by the local-only suite — see below);
* the value-recall fidelity gate passes on faithful output and fails on a drop.

The sample here uses a **generic, vendor-neutral** config XML on purpose: the
``--rag`` conversion is format-agnostic, and keeping the committed corpus neutral
keeps any particular vendor's config shape out of the public repo. Realistic
vendor-shaped (but sanitized) configs live under ``tests/local/`` (git-ignored)
and the optional ``DOC2MD_PEARLS_DIR`` corpus check below; both are skipped here
when absent so nothing vendor-specific lives in git.
"""

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

pytest.importorskip("xmltodict")


# Fully synthetic, vendor-neutral config. Exercises: nested mappings, repeated
# siblings collapsing to a list (``param``), a ``name``-element discriminator, an
# ``@value`` attribute discriminator (``route``), a node carrying both ``@value``
# and ``@result`` (only the first folds; the other stays a leaf), a sole-attribute
# node (``note``) whose value must still surface, and a positioned comment.
SAMPLE_XML = """<?xml version="1.0"?>
<settings xmlns:cfg="uri:example.com/config/1.0">
  <!-- service definitions -->
  <service name="api">
    <param>
      <name>MAX_RETRIES</name>
      <value>3</value>
      <hidden>off</hidden>
    </param>
    <param>
      <name>TIMEOUT_MS</name>
      <value>500</value>
    </param>
    <route value="/v1/*">
      <target>
        <host>upstream-a.example.com</host>
        <weight>10</weight>
      </target>
      <match:header value="canary" result="true">
        <target>
          <host>upstream-b.example.com</host>
        </target>
      </match:header>
    </route>
    <note value="schema revision 3"/>
  </service>
</settings>
"""


@pytest.fixture(scope="module")
def rag_output() -> str:
    return doc2md.xml_to_rag(SAMPLE_XML)


@pytest.fixture(scope="module")
def leaf_lines(rag_output: str):
    body = rag_output.split(
        "# FLATTENED CONFIGURATION (path-qualified leaves)\n\n", 1
    )[1]
    return [ln for ln in body.splitlines() if ln.strip()]


def test_every_body_line_is_self_contained(leaf_lines):
    """No orphaned lines: each flattened leaf carries its path and a value."""
    for ln in leaf_lines:
        assert " = " in ln, "leaf is not a path-qualified record: %r" % ln
        path = ln.split(" = ", 1)[0]
        assert path.startswith("settings"), "leaf lost its root path: %r" % ln


def test_discriminator_folded_into_segment(leaf_lines):
    """A node's identity rides in the path segment, not as a bare key."""
    joined = "\n".join(leaf_lines)
    # the conditional override's match context travels with the leaf
    assert any(
        "route(value=/v1/*)" in ln
        and "match:header(value=canary)" in ln
        and "upstream-b.example.com" in ln
        for ln in leaf_lines
    ), "conditional path not denormalized onto the overridden target leaf"
    # discriminator 'name' is in the segment, not re-emitted as '> name = X'
    assert "param(name=MAX_RETRIES)" in joined
    assert "> name = MAX_RETRIES" not in joined
    # a non-chosen discriminator key (@result, after @value folds) stays a leaf
    assert any(ln.endswith("@result = true") for ln in leaf_lines)


def test_sole_discriminator_node_still_emits_value(leaf_lines):
    """A bare ``<note value="..."/>`` (discriminator is its only content) must
    still surface its value as a leaf, not vanish into a segment."""
    assert any("schema revision 3" in ln.split(" = ", 1)[1] for ln in leaf_lines)


def test_mixed_content_text_folds_no_dict_or_hash_text():
    """A comment positioned next to an element's text makes ``xmltodict`` parse it
    as a mixed-content ``{'#text': value, '_comment': note}`` node. The value must
    surface on the element's own segment (not a spurious ``> #text`` child), and
    the raw Python ``dict`` repr must never leak — neither in the flattened leaves
    nor in the summary's variable values (the bugs a chunk-based reviewer hit)."""
    xml = (
        '<?xml version="1.0"?>\n'
        "<configs>\n"
        "  <assign:variable>\n"
        "    <name>FLAG_CW</name>\n"
        "    <!-- 2^19 -->\n"
        "    <value>524288</value>\n"
        "  </assign:variable>\n"
        "</configs>\n"
    )
    out = doc2md.xml_to_rag(xml)
    assert "#text" not in out, "mixed-content text leaked as a '> #text' segment"
    assert "{'" not in out and "': '" not in out, "raw dict repr leaked into output"
    # the value rides on the element's own segment...
    assert "> value = 524288" in out
    # ...the variable summary shows the bare value, not a dict...
    assert "Variable FLAG_CW is set to: 524288" in out
    # ...and the annotating comment is still present (value-recall fidelity).
    assert "> value > _comment = 2^19" in out
    ok, reason = doc2md.rag_fidelity_check(out, xml)
    assert ok is True, reason


def test_summary_block_present(rag_output):
    """The deterministic summary block is emitted. Its origin/variable extraction
    is vocabulary-specific, so with neutral input both sections report none —
    extraction is verified in the local-only suite."""
    summary = rag_output.split("# FLATTENED CONFIGURATION", 1)[0]
    assert "# DOCUMENT SUMMARY" in summary
    assert summary.count("(none found)") == 2


def test_fidelity_passes_on_faithful_output(rag_output):
    ok, reason = doc2md.rag_fidelity_check(rag_output, SAMPLE_XML)
    assert ok is True, reason


def test_fidelity_fails_when_a_leaf_is_dropped(rag_output):
    """Value-recall gate must reject output missing a source leaf value."""
    mangled = rag_output.replace("upstream-b.example.com", "REDACTED")
    ok, reason = doc2md.rag_fidelity_check(mangled, SAMPLE_XML)
    assert ok is False
    assert "missing" in reason


def test_structural_check(rag_output):
    counts, ok, reason = doc2md.rag_structural_check(rag_output)
    assert ok is True, reason
    assert counts["leaves"] > 0
    _bad_counts, bad_ok, _ = doc2md.rag_structural_check("no summary, no leaves here")
    assert bad_ok is False


def test_no_source_is_unassessable():
    assert doc2md.rag_fidelity_check("anything", None) == (None, None)


@pytest.mark.skipif(
    not os.environ.get("DOC2MD_PEARLS_DIR"),
    reason="set DOC2MD_PEARLS_DIR to a dir of real .xml configs to corpus-test",
)
def test_corpus_value_recall():
    corpus = pathlib.Path(os.environ["DOC2MD_PEARLS_DIR"])
    xmls = sorted(corpus.glob("*.xml"))
    assert xmls, "DOC2MD_PEARLS_DIR has no .xml files"
    for xml_path in xmls:
        raw = xml_path.read_text(encoding="utf-8", errors="replace")
        out = doc2md.xml_to_rag(raw)
        ok, reason = doc2md.rag_fidelity_check(out, raw)
        assert ok is True, "%s: %s" % (xml_path.name, reason)
