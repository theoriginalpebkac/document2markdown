"""Regression tests for bare single-file HTML handling.

A Jira/Confluence "Save as Word" export is frequently a plain HTML page named
``.doc`` (or ``.htm``). Because doc2md classifies by *content* and not by
extension, :func:`doc2md.detect_format` must sniff such a file as ``html`` and
route it to the de-chroming Word/HTML path — rather than falling through to
``unsupported`` (the bug that left ``WEBGESC-314.doc`` "unrecognized content").

All fixtures here are synthetic. The data-URI image fixtures are valid PNG
*headers* padded to a target size: the unit under test only reads the magic
bytes, the IHDR width/height (offsets 16:24), and the byte length, so a
renderable pixel payload is unnecessary.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402


def _png(width: int, height: int, size: int) -> bytes:
    """A PNG with a valid signature + IHDR (so dims read), padded to ``size``."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = (
        b"\x00\x00\x00\x0d" + b"IHDR"
        + width.to_bytes(4, "big") + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00" + b"\x00\x00\x00\x00"  # bit depth/colour + crc
    )
    head = sig + ihdr
    return head + b"\x00" * max(0, size - len(head))


# A real figure: above both the byte floor (12000) and the 64px floor.
CONTENT_PNG = _png(100, 100, 13000)
# A UI icon: small in bytes (and dims), so it must be dropped.
ICON_PNG = _png(16, 16, 500)


# --------------------------------------------------------------------------- #
# detect_format — content wins over the extension
# --------------------------------------------------------------------------- #


def test_bare_html_named_doc_is_html(tmp_path):
    f = tmp_path / "WEBGESC-314.doc"
    f.write_text("<!DOCTYPE html>\n<html><head><title>x</title></head>"
                 "<body><h1>Hi</h1></body></html>")
    assert doc2md.detect_format(f) == "html"


def test_html_tag_without_doctype(tmp_path):
    f = tmp_path / "page.doc"
    f.write_text("<html><body><p>no doctype</p></body></html>")
    assert doc2md.detect_format(f) == "html"


def test_html_after_leading_comment(tmp_path):
    f = tmp_path / "page.doc"
    f.write_text("<!-- saved from url -->\n<html><body>x</body></html>")
    assert doc2md.detect_format(f) == "html"


def test_plain_html_extension_routes_to_html_path(tmp_path):
    # .html no longer goes to the generic pandoc path — it gets de-chroming too.
    f = tmp_path / "page.html"
    f.write_text("<!doctype html><html><body><h1>x</h1></body></html>")
    assert doc2md.detect_format(f) == "html"


def test_mhtml_multipart_beats_html(tmp_path):
    # MHTML also contains <html>; the MIME sniff must win so images are unwrapped.
    f = tmp_path / "export.doc"
    f.write_text(
        "MIME-Version: 1.0\r\n"
        "Content-Type: multipart/related; boundary=\"b\"\r\n\r\n"
        "--b\r\nContent-Type: text/html\r\n\r\n<html><body>x</body></html>\r\n--b--\r\n"
    )
    assert doc2md.detect_format(f) == "mhtml"


def test_config_xml_still_xml(tmp_path):
    f = tmp_path / "config.xml"
    f.write_text("<?xml version=\"1.0\"?>\n<config><a>1</a></config>")
    assert doc2md.detect_format(f) == "xml"


def test_plain_text_doc_is_unsupported(tmp_path):
    # A genuinely non-HTML .doc body must not be mis-sniffed as html.
    f = tmp_path / "notes.doc"
    f.write_text("Just some plain notes mentioning <html> in prose. " * 40)
    assert doc2md.detect_format(f) == "unsupported"


# --------------------------------------------------------------------------- #
# _extract_data_uri_images — inline figures survive, icons are dropped
# --------------------------------------------------------------------------- #


def _data_uri(png: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def test_content_data_uri_extracted_to_file(tmp_path):
    fig_dir = tmp_path / "slug" / "figures"
    html = '<p>before</p><img src="%s" class="x"/><p>after</p>' % _data_uri(CONTENT_PNG)
    out, n = doc2md._extract_data_uri_images(html, fig_dir, "slug", "slug/figures", "Doc")
    assert n == 1
    written = list(fig_dir.glob("*.png"))
    assert len(written) == 1
    assert written[0].read_bytes() == CONTENT_PNG
    assert "data:image" not in out  # the blob is gone from the HTML
    assert 'src="slug/figures/slug-figure01.png"' in out
    assert 'alt="Doc — figure 1"' in out


def test_icon_data_uri_dropped(tmp_path):
    fig_dir = tmp_path / "slug" / "figures"
    html = '<p>x</p><img src="%s"/>' % _data_uri(ICON_PNG)
    out, n = doc2md._extract_data_uri_images(html, fig_dir, "slug", "slug/figures", "Doc")
    assert n == 0
    assert not fig_dir.exists() or not list(fig_dir.glob("*"))
    assert "data:image" not in out  # decorative icon tag removed entirely


def test_malformed_data_uri_left_for_cleaner(tmp_path):
    fig_dir = tmp_path / "slug" / "figures"
    html = '<img src="data:image/png;base64,@@@not-base64@@@"/>'
    out, n = doc2md._extract_data_uri_images(html, fig_dir, "slug", "slug/figures", "Doc")
    assert n == 0
    assert out == html  # untouched; clean_html_markdown drops it later


# --------------------------------------------------------------------------- #
# End-to-end through pandoc (skipped when pandoc is unavailable)
# --------------------------------------------------------------------------- #


pandoc = pytest.mark.skipif(
    doc2md.shutil.which("pandoc") is None, reason="pandoc not on PATH"
)


@pandoc
def test_convert_word_html_end_to_end(tmp_path):
    src = tmp_path / "issue.doc"
    src.write_text(
        "<!DOCTYPE html>\n<html><head><title>Issue</title></head><body>"
        "<h1>Cache-Tag bug</h1>"
        "<div><span>chrome</span></div>"
        "<p>The body paragraph with the real content.</p>"
        '<img src="%s"/>'
        "</body></html>" % _data_uri(CONTENT_PNG)
    )
    dest = tmp_path / "out" / "issue.md"
    wc = doc2md.convert_word(src, dest, "html", have_pandoc=True)
    assert dest.exists()
    assert "Cache-Tag bug" in wc.markdown
    assert "real content" in wc.markdown
    assert wc.images == 1
    # The inline figure is rewritten to a real file reference, not a data: blob.
    assert "data:image" not in wc.markdown
    assert (tmp_path / "out" / "issue" / "figures" / "issue-figure01.png").exists()
