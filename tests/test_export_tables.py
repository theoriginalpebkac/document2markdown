"""Regression tests for :func:`doc2md.simplify_export_html`.

Jira/Confluence "Save as Word" exports lay the whole document out with HTML
tables (issue-metadata grids, single-cell description/comment wrappers) and bury
genuine data tables *inside* comment bodies. Pandoc passes all of that through
as raw HTML, so doc2md reshapes the recognizable layout roles into Markdown
before pandoc. These tests assert on the transformed *HTML* (bs4 only, no pandoc
needed): each layout role is flattened, while a real nested data table and the
code-language class survive.

The fixture is synthetic but mirrors the exact shapes a Jira issue export emits:
``class="grid"`` key/value rows, a one-row section divider, a ``descriptionArea``
wrapper, ``comment-header``/``comment-body`` rows, Confluence ``code panel``
div wrappers, and decorative ``/images/icons/`` gifs.
"""

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

pytest.importorskip("bs4")


JIRA_EXPORT = """<!DOCTYPE html>
<html><body>
<table class="tableBorder" border="0" width="100%">
  <tr><td colspan="2" bgcolor="#f0f0f0"><h3 class="formtitle">[ABC-1]
      <a href="https://x/browse/ABC-1">The title</a></h3></td></tr>
  <tr><td width="20%"><b>Status:</b></td><td width="80%">Open</td></tr>
  <tr><td><b>Reporter:</b></td>
      <td><a class="user-hover" rel="jdoe" id="rep" href="https://x/u/jdoe">Jane Doe</a></td></tr>
</table>
<br />
<table cellpadding="2" border="0"><tr><td><b>Description</b></td><td></td></tr></table>
<table><tr><td id="descriptionArea">
  <p>First paragraph.</p>
  <div class="code panel" style="border-width: 1px;"><div class="codeContent panelContent">
  <pre class="code-java"><code>if (a &lt; b) doThing();</code></pre>
  </div></div>
</td></tr></table>
<br />
<table cellpadding="2" border="0"><tr><td><b>Comments</b></td><td></td></tr></table>
<table class="grid">
  <tr id="comment-header-1"><td bgcolor="#f0f0f0">Comment by
      <a href="https://x/u/jdoe" rel="jdoe">Jane Doe</a> [ 01-Jan-26 ]</td></tr>
  <tr id="comment-body-1"><td bgcolor="#ffffff">
    <p>Look at this table:</p>
    <table class="confluenceTable"><tbody>
      <tr><th>Code</th><th>Meaning</th></tr>
      <tr><td>e</td><td>socket error</td></tr>
    </tbody></table>
    <p><img src="https://x/jira/images/icons/attach/image.gif" alt="icon"/>
       <img src="https://x/jira/secure/attachment/9/shot.png" alt="real shot"/></p>
  </td></tr>
  <tr id="comment-header-2"><td bgcolor="#f0f0f0">Comment by Bob [ 02-Jan-26 ]</td></tr>
  <tr id="comment-body-2"><td bgcolor="#ffffff"><p>Second comment.</p></td></tr>
</table>
</body></html>
"""


@pytest.fixture(scope="module")
def out():
    return doc2md.simplify_export_html(JIRA_EXPORT)


def test_metadata_grid_flattened_to_key_value(out):
    assert "<strong>Status:</strong> Open" in out
    # The title row keeps its heading rather than becoming a key/value pair.
    assert "<h3" in out and "The title" in out


def test_section_dividers_become_headings(out):
    assert "<h2>Description</h2>" in out
    assert "<h2>Comments</h2>" in out


def test_description_wrapper_unwrapped(out):
    # The single-cell wrapper table is gone; its prose/code are now top level.
    assert "First paragraph." in out
    # ...and the code language survives (drives a fenced ```code-java block).
    assert 'class="code-java"' in out


def test_comments_rendered_with_separators(out):
    assert "<strong>Comment by Jane Doe [ 01-Jan-26 ]</strong>" in out
    assert "<strong>Comment by Bob [ 02-Jan-26 ]</strong>" in out
    # One <hr/> between the two comments (not before the first).
    assert out.count("<hr/>") == 1


def test_nested_data_table_preserved(out):
    # The Confluence data table inside a comment body stays a real table.
    assert "<table" in out
    assert "socket error" in out
    assert "<th>Code</th>" in out or "<th>Code" in out


def test_layout_chrome_removed(out):
    # No layout tables/divs/spans, and no presentation attributes survive.
    assert 'class="grid"' not in out
    assert 'class="tableBorder"' not in out
    assert "<div" not in out and "<span" not in out
    assert "bgcolor" not in out
    assert "valign" not in out
    assert 'width="20%"' not in out


def test_link_attrs_stripped_but_href_kept(out):
    # rel/id/class dropped so pandoc emits a clean [text](href); href preserved.
    assert 'href="https://x/u/jdoe"' in out
    assert "rel=" not in out


def test_decorative_icon_dropped_content_image_kept(out):
    assert "/images/icons/" not in out
    assert "shot.png" in out


def test_missing_bs4_returns_input_unchanged(monkeypatch):
    # Best-effort: if bs4 can't be imported, the raw HTML passes through so the
    # run falls back to pandoc's own conversion instead of crashing.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "bs4" or name.startswith("bs4."):
            raise ImportError("no bs4")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert doc2md.simplify_export_html("<table><tr><td>x</td></tr></table>") == (
        "<table><tr><td>x</td></tr></table>"
    )
