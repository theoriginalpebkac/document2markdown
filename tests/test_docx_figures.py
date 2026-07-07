"""Regression test: docx-embedded images must land in the same
``<slug>/figures/<slug>-figureNN.png`` layout as the MHTML/HTML Word paths,
not pandoc's raw ``<slug>/media/imageN.png`` + ``<img>`` HTML tag output.

The docx fixture here is a hand-built minimal OOXML zip (not python-docx,
which isn't a project dependency) mirroring the structure Word/python-docx
actually produce, so pandoc parses it exactly as it would a real .docx.
"""

import pathlib
import sys
import zipfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import doc2md  # noqa: E402

pandoc = pytest.mark.skipif(
    doc2md.shutil.which("pandoc") is None, reason="pandoc not on PATH"
)

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_PKG_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
)

_DRAWING_PARA = """<w:p><w:r><w:drawing><wp:inline><wp:extent cx="254000" cy="254000"/>
<wp:docPr id="%(n)d" name="Picture %(n)d"/>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:blipFill><a:blip r:embed="rId%(rid)d"/></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="254000" cy="254000"/></a:xfrm>
<a:prstGeom prst="rect"/></pic:spPr></pic:pic>
</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"""


def _png(width: int, height: int) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = (
        b"\x00\x00\x00\x0d" + b"IHDR"
        + width.to_bytes(4, "big") + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00" + b"\x00\x00\x00\x00"
    )
    idat = (
        b"\x00\x00\x00\x0a" + b"IDATx\x9c\x63\x00\x00\x00\x02\x00\x01"
        + b"\x00\x00\x00\x00"
    )
    iend = b"\x00\x00\x00\x00" + b"IEND" + b"\xae\x42\x60\x82"
    return sig + ihdr + idat + iend


def _make_docx(path: pathlib.Path, n_images: int) -> None:
    body = ["<w:p><w:r><w:t>Intro paragraph.</w:t></w:r></w:p>"]
    doc_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    ]
    for i in range(1, n_images + 1):
        body.append(_DRAWING_PARA % {"n": i, "rid": i})
        doc_rels.append(
            '<Relationship Id="rId%d" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            'Target="media/image%d.png"/>' % (i, i)
        )
    doc_rels.append("</Relationships>")

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        "<w:document %s><w:body>%s</w:body></w:document>"
        % (_DOC_NS, "".join(body))
    )

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _PKG_RELS)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", "".join(doc_rels))
        for i in range(1, n_images + 1):
            z.writestr("word/media/image%d.png" % i, _png(50 + i, 50 + i))


@pandoc
def test_convert_docx_figures_match_word_family_convention(tmp_path):
    src = tmp_path / "issue.docx"
    _make_docx(src, n_images=2)
    dest = tmp_path / "out" / "issue.md"

    wc = doc2md.convert_word(src, dest, "docx", have_pandoc=True)

    assert wc.images == 2
    # Clean Markdown image syntax, not a raw pandoc <img> HTML tag.
    assert "<img" not in wc.markdown
    assert "![issue — figure 1](issue/figures/issue-figure01.png)" in wc.markdown
    assert "![issue — figure 2](issue/figures/issue-figure02.png)" in wc.markdown

    fig_dir = tmp_path / "out" / "issue" / "figures"
    assert (fig_dir / "issue-figure01.png").exists()
    assert (fig_dir / "issue-figure02.png").exists()

    # No leftover pandoc scratch dir (<slug>/media/...) alongside figures/.
    assert not (tmp_path / "out" / "issue" / "media").exists()
