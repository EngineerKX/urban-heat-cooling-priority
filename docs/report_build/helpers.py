"""Shared helpers for the evaluation reports: a small .docx writer on the
project template (docs/report_build/template.docx, the land-cover report's
styles plus Heading 2/3), with chapter-aware headings so the same chapter code
builds both a standalone report and a chapter of the main report.

Standalone: chapter title = Title style, sections = Heading 1 ("1. Setup"),
subsections = Heading 2 ("1.1 ...").
Main report: chapter title = Heading 1 ("3. Land cover"), sections =
Heading 2 ("3.1 Setup"), subsections = Heading 3 ("3.1.1 ...").
"""

import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

TEMPLATE = Path(__file__).with_name("template.docx")
PAGE_W = 9900  # A4 width minus 1000-dxa margins, as in the template


class Doc:
    def __init__(self, chapter_number=None, date=None):
        """chapter_number=None builds a standalone report; an int builds a
        chapter of the main report with that number. `date` is appended to a
        standalone report's subtitle."""
        self.body = []
        self.date = date
        self.chapter = chapter_number
        self._section = 0
        self._subsection = 0

    # --- headings ---------------------------------------------------------
    def _style(self, level):
        """level 0 = chapter title, 1 = section, 2 = subsection."""
        level += 0 if self.chapter is None else 1
        return "Title" if level == 0 else f"Heading{level}"

    def title(self, text, subtitle=None):
        label = text if self.chapter is None else f"{self.chapter}. {text}"
        self.para(label, self._style(0))
        if self.chapter is None and self.date:
            subtitle = f"{subtitle} | {self.date}" if subtitle else self.date
        if subtitle:
            self.para(subtitle, "Subtitle")

    def section(self, text, numbered=True):
        if numbered:
            self._section += 1
            self._subsection = 0
            prefix = f"{self._section}." if self.chapter is None else f"{self.chapter}.{self._section}"
            text = f"{prefix} {text}"
        self.para(text, self._style(1))

    def subsection(self, text):
        self._subsection += 1
        prefix = (f"{self._section}.{self._subsection}" if self.chapter is None
                  else f"{self.chapter}.{self._section}.{self._subsection}")
        self.para(f"{prefix} {text}", self._style(2))

    # --- content ----------------------------------------------------------
    def para(self, text, style=None):
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        self.body.append(f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')

    def rich(self, parts):
        """One paragraph from (text, bold) runs."""
        runs = "".join(
            f'<w:r>{"<w:rPr><w:b/></w:rPr>" if b else ""}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'
            for t, b in parts
        )
        self.body.append(f"<w:p>{runs}</w:p>")

    def lead(self, lead, text):
        """Paragraph starting with a bold lead-in."""
        self.rich([(lead, True), (text, False)])

    def table(self, header, rows, widths):
        assert len(header) == len(widths) and abs(sum(widths) - PAGE_W) < 5, (header, sum(widths))
        for r in rows:
            assert len(r) == len(header), r
        border = "".join(
            f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
            for s in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        xml = [
            f'<w:tbl><w:tblPr><w:tblW w:w="{PAGE_W}" w:type="dxa"/><w:tblBorders>{border}</w:tblBorders>'
            '<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
            '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
            'w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>'
            + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
        ]
        for i, row in enumerate([header] + rows):
            trpr = "<w:trPr><w:tblHeader/></w:trPr>" if i == 0 else "<w:trPr><w:cantSplit/></w:trPr>"
            cells = []
            for w, text in zip(widths, row):
                shd = '<w:shd w:val="clear" w:color="auto" w:fill="E2E8F0"/>' if i == 0 else ""
                paras = "".join(
                    f'<w:p><w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>'
                    for line in str(text).split("\n")
                )
                cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}</w:tcPr>{paras}</w:tc>')
            xml.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
        xml.append("</w:tbl>")
        self.body.append("".join(xml))
        self.body.append("<w:p/>")

    def page_break(self):
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    # --- output -----------------------------------------------------------
    def save(self, out, title):
        src = zipfile.ZipFile(TEMPLATE)
        doc = src.read("word/document.xml").decode("utf8")
        head = doc[: doc.index("<w:body>") + len("<w:body>")]
        sect = re.search(r"<w:sectPr.*?</w:sectPr>", doc, re.S).group(0)
        new_doc = head + "".join(self.body) + sect + "</w:body></w:document>"
        core = src.read("docProps/core.xml").decode("utf8")
        core = re.sub(r"<dc:title>.*?</dc:title>", "", core)
        core = core.replace("<cp:lastModifiedBy>", f"<dc:title>{escape(title)}</dc:title><cp:lastModifiedBy>", 1)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for info in src.infolist():
                if info.filename == "word/document.xml":
                    data = new_doc.encode("utf8")
                elif info.filename == "docProps/core.xml":
                    data = core.encode("utf8")
                else:
                    data = src.read(info.filename)
                z.writestr(info, data)
        print("wrote", out)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def signed(x, d=2):
    """Signed number without a minus sign on a value that rounds to zero."""
    return f"{0:.{d}f}" if abs(x) < 0.5 * 10 ** -d else f"{x:+.{d}f}"
