"""Regression tests for ``--yaml-index`` breadcrumb injection.

The breadcrumb pass (:func:`doc2md._yaml_breadcrumbs`) is a line-based
post-processor over the dumped YAML. It must never inject a ``# path: ...``
comment into the body of a multiline block scalar (``|``/``>``): those bodies are
literal text, and an interior line that happens to end in ``:`` (e.g.
``Possible values are:``) must not be mistaken for a mapping key. A regression
there corrupts the string value so the YAML no longer round-trips to the source
XML, which the fidelity gate (:func:`doc2md.yaml_fidelity_check`) rejects.

The committed tests are fully synthetic (no external corpus). An optional
corpus check runs the same round-trip over real XML when ``DOC2MD_PEARLS_DIR``
points at a directory of ``.xml`` files; it is skipped otherwise so the path
never lives in git.
"""

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

yaml = pytest.importorskip("yaml")
xmltodict = pytest.importorskip("xmltodict")


def _round_trips(xml: str, index: bool) -> bool:
    """True when the emitted YAML parses back to ``xmltodict``'s view of ``xml``."""
    expected = xmltodict.parse(doc2md._xml_comments_to_elements(xml))
    actual = yaml.safe_load(doc2md.xml_to_yaml(xml, index=index))
    return actual == expected


# A single comment renders as a mapping-value block scalar (``_comment: |-``);
# two siblings collapse to a list of bare block-scalar items (``- |-``). Both
# bodies contain colon-terminated interior lines followed by a deeper line —
# the exact shape that fooled the breadcrumb walker into thinking a new block
# had opened.
_XML = """<configs>
  <template name="DEMO">
    <body>
      <step name="single">
        <!-- Sets the retrieval method.
             Possible values are:
                 NS - from NetStorage
                 FMA - from FMA-T -->
        <value>1</value>
      </step>
      <step name="multi">
        <!-- First note:
                 detail line A -->
        <!-- Second note:
                 detail line B -->
        <value>2</value>
      </step>
    </body>
  </template>
</configs>
"""


def test_block_scalar_bodies_do_not_break_round_trip():
    # Baseline: without breadcrumbs the conversion is already faithful.
    assert _round_trips(_XML, index=False)
    # Regression: breadcrumb injection must not corrupt block-scalar bodies.
    assert _round_trips(_XML, index=True)


def test_breadcrumbs_are_still_emitted_for_real_blocks():
    # The fix skips block-scalar bodies but must not suppress genuine breadcrumbs.
    out = doc2md.xml_to_yaml(_XML, index=True)
    crumbs = [l for l in out.splitlines() if l.lstrip().startswith("# path:")]
    assert crumbs, "expected structural breadcrumbs for nested blocks"
    # No breadcrumb may land inside a comment body (those lines are indented far
    # deeper than any real key and would carry interior comment text).
    assert not any("Possible values are" in c or "detail line" in c for c in crumbs)


def test_colon_line_stays_inside_comment_value():
    # The interior 'Possible values are:' line must survive verbatim in the value,
    # not be split out as its own node by an injected breadcrumb.
    data = yaml.safe_load(doc2md.xml_to_yaml(_XML, index=True))
    comment = data["configs"]["template"]["body"]["step"][0]["value"]["_comment"]
    assert "Possible values are:" in comment
    assert "# path:" not in comment


@pytest.mark.skipif(
    not os.environ.get("DOC2MD_PEARLS_DIR"),
    reason="set DOC2MD_PEARLS_DIR to a dir of .xml files to run the corpus check",
)
def test_corpus_round_trips_with_index():
    """Round-trip every XML in ``$DOC2MD_PEARLS_DIR`` with breadcrumbs enabled.

    Gated on an env var so real (potentially confidential) corpora are never
    referenced from git. Skipped by default.
    """
    corpus = pathlib.Path(os.environ["DOC2MD_PEARLS_DIR"])
    xmls = sorted(corpus.glob("*.xml"))
    assert xmls, "no .xml files found in %s" % corpus
    failures = []
    for xml_path in xmls:
        raw = xml_path.read_text(encoding="utf-8", errors="replace")
        try:
            if not _round_trips(raw, index=True):
                failures.append(xml_path.name)
        except Exception as exc:  # noqa: BLE001 - report which file blew up
            failures.append("%s (%s)" % (xml_path.name, exc))
    assert not failures, "did not round-trip: %s" % ", ".join(failures)
