from __future__ import annotations

import argparse
import os
import xml.sax.saxutils as xml
import zipfile
from dataclasses import dataclass
from pathlib import Path


def _workspace_dir() -> Path:
    path = Path(os.environ.get("WORKSPACE_DIR", ".")).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _filename(value: str) -> str:
    name = value.strip() or "report.docx"
    if not name.endswith(".docx"):
        name += ".docx"
    return Path(name).name


def _paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return (
        "<w:p>"
        f"{style_xml}"
        "<w:r>"
        f"<w:t>{xml.escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


@dataclass
class Block:
    kind: str
    heading: str
    text: str


class AppendBlock(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        blocks = getattr(namespace, "blocks", None)
        if blocks is None:
            blocks = []
            setattr(namespace, "blocks", blocks)
        value = values.strip()
        if not value:
            return
        if option_string == "--paragraph":
            blocks.append(Block("paragraph", "", value))
            return
        heading, text = _split_pair(
            value,
            "Highlights" if option_string == "--bullet" else "Details",
        )
        if text:
            blocks.append(Block("bullet" if option_string == "--bullet" else "section", heading, text))


def _split_pair(value: str, default_heading: str = "") -> tuple[str, str]:
    if "::" not in value:
        return default_heading, value.strip()
    heading, text = value.split("::", 1)
    return heading.strip() or default_heading, text.strip()


def _document_xml(title: str, blocks: list[Block]) -> str:
    paragraphs = [_paragraph(title, "Title")]
    current_heading = ""
    for block in blocks:
        if block.kind == "paragraph":
            paragraphs.append(_paragraph(block.text))
            continue
        if block.heading and block.heading != current_heading:
            paragraphs.append(_paragraph(block.heading, "Heading1"))
            current_heading = block.heading
        prefix = "- " if block.kind == "bullet" else ""
        paragraphs.append(_paragraph(f"{prefix}{block.text}"))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(paragraphs)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        "</w:sectPr></w:body></w:document>"
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
        '<w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
        "</w:styles>"
    )


def _write_docx(path: Path, title: str, blocks: list[Block]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            "</Types>",
        )
        docx.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>',
        )
        docx.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/></Relationships>',
        )
        docx.writestr(
            "word/document.xml",
            _document_xml(title, blocks),
        )
        docx.writestr("word/styles.xml", _styles_xml())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a general Word DOCX document.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--paragraph", action=AppendBlock, default=[])
    parser.add_argument("--section", action=AppendBlock)
    parser.add_argument("--bullet", action=AppendBlock)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    blocks: list[Block] = getattr(args, "blocks", [])
    if not blocks:
        blocks.append(Block("paragraph", "", f"{args.title}"))

    output_name = _filename(args.output or args.title.lower().replace(" ", "-"))
    output_path = _workspace_dir() / output_name

    _write_docx(output_path, args.title, blocks)
    print(str(output_path))


if __name__ == "__main__":
    main()
