"""Regression tests for the ``--rag`` flat-index conversion.

``--rag`` projects config XML into a deterministic DOCUMENT SUMMARY plus one
fully path-qualified ``a > b > c = value`` line per leaf (:func:`doc2md.xml_to_rag`).
Unlike ``--yaml`` it does *not* round-trip to the source structure, so fidelity
is enforced at the value level instead: every scalar leaf in the source XML must
appear in the output (:func:`doc2md.rag_fidelity_check`). The properties that
matter and are covered here:

* every emitted body line is self-contained (carries its full path), so it
  survives a RAG chunk boundary;
* the node discriminator (``@value``/``@name``/``name`` …) is folded into the
  path segment, *except* when it's a node's only content, where it must still be
  emitted as a leaf or the value is lost (the bug the fidelity gate caught);
* the manifest aggregates origins + variables;
* the value-recall fidelity gate passes on faithful output and fails when a leaf
  is dropped.

The committed tests are fully synthetic (no external corpus). An optional corpus
check runs the same value-recall over real XML when ``DOC2MD_PEARLS_DIR`` points
at a directory of ``.xml`` files; it is skipped otherwise so no proprietary
content lives in git.
"""

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

pytest.importorskip("xmltodict")


# Fully synthetic config (generic placeholder hostnames/values — no real corpus).
SAMPLE_XML = """<?xml version="1.0"?>
<configs xmlns:match="uri:akamai.com/metadata/match/5.0">
  <!-- top of property -->
  <akamai:edge-config version="5.0">
    <assign:variable>
      <name>PMUSER_ORIGIN_BACKUP</name>
      <value>backup.origin.example.net</value>
      <hidden>off</hidden>
    </assign:variable>
    <assign:variable>
      <name>PMUSER_ORIGIN_FAILOVER</name>
      <value>1</value>
    </assign:variable>
    <match:uri.component value="AK_PM_VPATH*">
      <forward:origin-server>
        <host>primary.origin.example.com</host>
        <dns-name>
          <status>on</status>
          <value>primary.origin.example.com</value>
        </dns-name>
      </forward:origin-server>
      <match:request.header value="true">
        <forward:origin-server>
          <host>%(PMUSER_ORIGIN_BACKUP)</host>
        </forward:origin-server>
      </match:request.header>
    </match:uri.component>
    <comment:note value="Catalog version: 1.2.3"/>
  </akamai:edge-config>
</configs>
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
        assert path.startswith("configs"), "leaf lost its root path: %r" % ln


def test_discriminator_folded_into_segment(leaf_lines):
    """A node's identity rides in the path segment, not as a bare key."""
    joined = "\n".join(leaf_lines)
    # the failover override's conditional context travels with the leaf
    assert any(
        "match:request.header(value=true)" in ln
        and "forward:origin-server" in ln
        and "%(PMUSER_ORIGIN_BACKUP)" in ln
        for ln in leaf_lines
    ), "conditional path not denormalized onto the failover origin leaf"
    # discriminator 'name' is in the segment, not re-emitted as '> name = X'
    assert "assign:variable(name=PMUSER_ORIGIN_BACKUP)" in joined
    assert "> name = PMUSER_ORIGIN_BACKUP" not in joined


def test_sole_discriminator_node_still_emits_value(leaf_lines):
    """A bare ``<comment:note value="..."/>`` (discriminator is its only content)
    must still surface its value as a leaf, not vanish into a segment."""
    assert any("Catalog version: 1.2.3" in ln.split(" = ", 1)[1] for ln in leaf_lines)


def test_manifest_aggregates_origins_and_variables(rag_output):
    summary = rag_output.split("# FLATTENED CONFIGURATION", 1)[0]
    assert "# DOCUMENT SUMMARY" in summary
    # origins gathered from forward blocks (primary + variable-backed backup)
    assert "primary.origin.example.com" in summary
    assert "%(PMUSER_ORIGIN_BACKUP)" in summary
    # variable definitions resolved as facts
    assert (
        "Variable PMUSER_ORIGIN_BACKUP is set to: "
        "backup.origin.example.net" in summary
    )


def test_fidelity_passes_on_faithful_output(rag_output):
    ok, reason = doc2md.rag_fidelity_check(rag_output, SAMPLE_XML)
    assert ok is True, reason


def test_fidelity_fails_when_a_leaf_is_dropped(rag_output):
    """Value-recall gate must reject output missing a source leaf value."""
    mangled = rag_output.replace(
        "backup.origin.example.net", "REDACTED"
    )
    ok, reason = doc2md.rag_fidelity_check(mangled, SAMPLE_XML)
    assert ok is False
    assert "missing" in reason


def test_structural_check(rag_output):
    counts, ok, reason = doc2md.rag_structural_check(rag_output)
    assert ok is True, reason
    assert counts["leaves"] > 0
    bad_counts, bad_ok, _ = doc2md.rag_structural_check("no summary, no leaves here")
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
