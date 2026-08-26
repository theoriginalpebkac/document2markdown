"""Regression tests for interleaved-sibling order preservation in ``--yaml``/``--rag``.

``xmltodict`` groups repeated sibling tags into one list keyed by tag name. That
is lossless only when the repeats are *contiguous*; once a different tag
intervenes and the tag reappears, grouping silently pulls the reappearance back
next to its earlier siblings. In this config XML sibling order encodes
execution/override order, so that reordering is a real fidelity loss.

:func:`doc2md._xml_to_ir` parses into an ordered tree and the projections decide
*per node* whether the children are interleaved: a non-interleaved node keeps the
compact grouped form (byte-identical to the old output); an interleaved node
becomes an ordered sequence of single-key mappings (``--yaml``) / keeps document
order with positional ``[N]`` disambiguation (``--rag``).

The core guarantee is checked structurally: for **every** parent element, the
child tag sequence implied by the generated YAML must equal ElementTree's true
document-order sequence (:func:`child_sequence_mismatches`). The committed sample
is synthetic and vendor-neutral; an optional corpus check runs the same harness
over real ``.xml`` when ``DOC2MD_PEARLS_DIR`` points at a directory, skipped
otherwise so no corpus path lives in git.
"""

import io
import os
import pathlib
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

yaml = pytest.importorskip("yaml")
pytest.importorskip("xmltodict")


# A vendor-neutral config that reproduces the real failure shape:
#   * ``origin`` appears at positions 0, then again at 4 and 5 *after* two
#     ``match:*`` tags intervene -> interleaved (must become an ordered sequence),
#     and it carries no discriminator -> --rag must disambiguate with ``[N]``;
#   * ``match:protocol`` repeats *contiguously* with an ``@value`` discriminator
#     -> stays grouped, no ``[N]``;
#   * ``settings`` is a separate, non-interleaved block whose repeated ``var``
#     children (with a ``name`` discriminator) must stay in the compact grouped
#     form, proving the decision is per-node, not global.
SAMPLE_XML = """<?xml version="1.0"?>
<config xmlns:match="uri:example.com/match/1.0">
  <origin>
    <host>a.example.com</host>
    <dns-name>
      <status>on</status>
    </dns-name>
  </origin>
  <match:type value="cname">
    <rewrite>on</rewrite>
  </match:type>
  <match:protocol value="HTTP">
    <port>80</port>
  </match:protocol>
  <match:protocol value="HTTPS">
    <port>443</port>
  </match:protocol>
  <origin>
    <ip-version>4</ip-version>
  </origin>
  <origin>
    <host>b.example.com</host>
  </origin>
  <settings>
    <var>
      <name>MAX_RETRIES</name>
      <value>3</value>
    </var>
    <var>
      <name>TIMEOUT_MS</name>
      <value>500</value>
    </var>
  </settings>
</config>
"""


# --------------------------------------------------------------------------- #
# ElementTree cross-check: YAML-implied child order == true document order
# --------------------------------------------------------------------------- #

def _uri_to_prefix(pre_xml: str) -> dict:
    """Map namespace URI -> prefix from the document's ``xmlns:*`` declarations, so
    ElementTree's Clark-notation ``{uri}local`` tags can be rendered back as the
    literal ``prefix:local`` names the converter (and ``xmltodict``) use."""
    out = {}
    for ev, data in ET.iterparse(io.StringIO(pre_xml), events=("start-ns",)):
        prefix, uri = data
        out.setdefault(uri, prefix)
    return out


def _pfx_tag(el: "ET.Element", uri2pfx: dict) -> str:
    t = el.tag
    if t.startswith("{"):
        uri, local = t[1:].split("}", 1)
        prefix = uri2pfx.get(uri, "")
        return "%s:%s" % (prefix, local) if prefix else local
    return t


def child_sequence_mismatches(raw_xml: str):
    """Return a list of ``(path, actual_tags, implied_tags)`` where the child tag
    sequence implied by the generated YAML disagrees with ElementTree's true
    document-order sequence. Empty list == the projection preserved order at every
    parent element. Both sides consume the same comment-preprocessed XML so the
    positioned ``_comment`` elements line up.
    """
    pre = doc2md._xml_comments_to_elements(raw_xml)
    uri2pfx = _uri_to_prefix(pre)
    root = ET.fromstring(pre)
    obj = doc2md._xml_to_yaml_obj(pre)

    mismatches = []

    def walk(elem, yval, path):
        actual = [_pfx_tag(c, uri2pfx) for c in list(elem)]
        if isinstance(yval, list):
            # interleaved node: value is a sequence of single-key mappings
            items = []
            for it in yval:
                (k, v), = it.items()
                if not k.startswith("@") and k != "#text":
                    items.append((k, v))
            implied = [k for k, _ in items]
            if actual != implied:
                mismatches.append((path, actual, implied))
                return
            for child, (k, sub) in zip(list(elem), items):
                walk(child, sub, path + "/" + k)
        elif isinstance(yval, dict):
            # grouped node: must not itself be interleaved
            if doc2md._is_interleaved(actual):
                mismatches.append((path, actual, "<grouped, but ET is interleaved>"))
                return
            counts = Counter(actual)
            idx = defaultdict(int)
            for child in list(elem):
                t = _pfx_tag(child, uri2pfx)
                val = yval.get(t)
                # index into the grouped list only when there really are >1 such
                # children; a single interleaved child's value is itself a list.
                if counts[t] > 1:
                    if not isinstance(val, list) or len(val) != counts[t]:
                        mismatches.append((path, t, "<grouped count/list mismatch>"))
                        return
                    sub = val[idx[t]]
                else:
                    sub = val
                idx[t] += 1
                walk(child, sub, path + "/" + t)
        else:
            if actual:
                mismatches.append((path, actual, "<scalar/None in YAML>"))

    walk(root, obj[_pfx_tag(root, uri2pfx)], _pfx_tag(root, uri2pfx))
    return mismatches


# --------------------------------------------------------------------------- #
# _is_interleaved unit coverage
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tags, expected", [
    (["a", "a", "b"], False),          # contiguous repeat, then a new tag
    (["a", "b", "b"], False),          # a new tag, then a contiguous repeat
    (["a", "b", "a"], True),           # a reappears after b
    (["a", "b", "b", "a"], True),      # a reappears after a run of b
    (["a", "b", "c", "b"], True),      # b reappears after c
    (["a"], False),
    ([], False),
])
def test_is_interleaved(tags, expected):
    assert doc2md._is_interleaved(tags) is expected


# --------------------------------------------------------------------------- #
# --yaml
# --------------------------------------------------------------------------- #

def test_yaml_child_order_matches_elementtree():
    """The heart of the fix: every parent's YAML child order == document order."""
    assert child_sequence_mismatches(SAMPLE_XML) == []


def test_interleaved_node_becomes_ordered_sequence():
    data = yaml.safe_load(doc2md.xml_to_yaml(SAMPLE_XML))
    config = data["config"]
    # interleaved -> the node's value is a list of single-key mappings, one per
    # child in document order (attributes lead as {'@k': v} items).
    assert isinstance(config, list)
    keys = [next(iter(item)) for item in config]
    # the node's attribute leads the sequence (attributes precede children in XML)
    assert keys[0] == "@xmlns:match"
    child_keys = [k for k in keys if not k.startswith("@")]
    assert child_keys == [
        "origin", "match:type", "match:protocol", "match:protocol",
        "origin", "origin", "settings",
    ]
    # the reappearing origins keep their true positions and payloads
    origins = [item["origin"] for item in config if "origin" in item]
    assert origins[0]["host"] == "a.example.com"
    assert origins[1] == {"ip-version": "4"}
    assert origins[2]["host"] == "b.example.com"


def test_noninterleaved_node_stays_grouped():
    """Per-node decision: ``settings`` is not interleaved, so it keeps the compact
    grouped form (``var`` collapses to a list) rather than a sequence."""
    data = yaml.safe_load(doc2md.xml_to_yaml(SAMPLE_XML))
    settings = [i["settings"] for i in data["config"] if "settings" in i][0]
    assert isinstance(settings, dict)
    assert isinstance(settings["var"], list)
    assert [v["name"] for v in settings["var"]] == ["MAX_RETRIES", "TIMEOUT_MS"]


def test_yaml_round_trip_fidelity():
    ok, reason = doc2md.yaml_fidelity_check(doc2md.xml_to_yaml(SAMPLE_XML), SAMPLE_XML)
    assert ok is True, reason
    ok_i, reason_i = doc2md.yaml_fidelity_check(
        doc2md.xml_to_yaml(SAMPLE_XML, index=True), SAMPLE_XML)
    assert ok_i is True, reason_i


# --------------------------------------------------------------------------- #
# --rag
# --------------------------------------------------------------------------- #

def _rag_leaves(rag_output: str):
    body = rag_output.split(doc2md._RAG_FLATTEN_HEADER, 1)[1]
    return [ln for ln in body.splitlines() if " = " in ln]


def test_rag_disambiguates_repeated_no_discriminator_blocks():
    leaves = _rag_leaves(doc2md.xml_to_rag(SAMPLE_XML))
    joined = "\n".join(leaves)
    # the three no-discriminator origins get positional indices...
    assert "config > origin[0] > host = a.example.com" in joined
    assert "config > origin[1] > ip-version = 4" in joined
    assert "config > origin[2] > host = b.example.com" in joined
    # ...while the discriminator-bearing repeat is disambiguated by its value,
    # not an index (no ``[N]`` on match:protocol).
    assert "match:protocol(value=HTTP) > port = 80" in joined
    assert "match:protocol(value=HTTPS) > port = 443" in joined
    assert "match:protocol[" not in joined


def test_rag_preserves_document_order():
    """The intervening ``match:*`` leaves fall *between* origin[0] and origin[1],
    exactly as in the source — not pulled apart by tag grouping."""
    leaves = _rag_leaves(doc2md.xml_to_rag(SAMPLE_XML))

    def first_index(needle):
        return next(i for i, ln in enumerate(leaves) if needle in ln)

    assert (first_index("origin[0]")
            < first_index("match:protocol(value=HTTP)")
            < first_index("origin[1]")
            < first_index("origin[2]"))


def test_rag_leaf_paths_are_unique():
    leaves = _rag_leaves(doc2md.xml_to_rag(SAMPLE_XML))
    paths = [ln.split(" = ", 1)[0] for ln in leaves]
    dupes = [p for p, c in Counter(paths).items() if c > 1]
    assert not dupes, "duplicate leaf paths (merged siblings): %s" % dupes


def test_rag_fidelity_passes():
    ok, reason = doc2md.rag_fidelity_check(doc2md.xml_to_rag(SAMPLE_XML), SAMPLE_XML)
    assert ok is True, reason


def test_rag_fidelity_detects_merged_path():
    """The strengthened gate rejects output where two leaves collapse onto one
    path (the pre-fix merge signature), even though every *value* still appears."""
    out = doc2md.xml_to_rag(SAMPLE_XML)
    merged = out.replace("config > origin[0] > host", "config > origin > host")
    merged = merged.replace("config > origin[2] > host", "config > origin > host")
    ok, reason = doc2md.rag_fidelity_check(merged, SAMPLE_XML)
    assert ok is False
    assert "not unique" in reason


# --------------------------------------------------------------------------- #
# Optional real-corpus check (git-clean by default)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not os.environ.get("DOC2MD_PEARLS_DIR"),
    reason="set DOC2MD_PEARLS_DIR to a dir of real .xml configs to corpus-test",
)
def test_corpus_child_order_matches_elementtree():
    corpus = pathlib.Path(os.environ["DOC2MD_PEARLS_DIR"])
    xmls = sorted(corpus.glob("*.xml"))
    assert xmls, "DOC2MD_PEARLS_DIR has no .xml files"
    failures = []
    for xml_path in xmls:
        raw = xml_path.read_text(encoding="utf-8", errors="replace")
        mism = child_sequence_mismatches(raw)
        if mism:
            failures.append("%s: %s" % (xml_path.name, mism[:3]))
    assert not failures, "child-order mismatches:\n" + "\n".join(failures)
