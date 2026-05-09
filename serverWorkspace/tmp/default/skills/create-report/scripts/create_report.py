from __future__ import annotations

import argparse
import os
import xml.sax.saxutils as xml
import zipfile
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


def _document_xml(
    title: str,
    audience: str,
    findings: list[str],
    recommendations: list[str],
) -> str:
    paragraphs = [
        _paragraph(title, "Title"),
        _paragraph(f"Audience: {audience}"),
        _paragraph("Executive Summary", "Heading1"),
        _paragraph(f"This report summarizes {title} for {audience}."),
        _paragraph("Key Findings", "Heading1"),
    ]
    paragraphs.extend(_paragraph(f"- {finding}") for finding in findings)
    paragraphs.append(_paragraph("Recommendations", "Heading1"))
    paragraphs.extend(_paragraph(f"- {item}") for item in recommendations)
    paragraphs.extend(
        [
            _paragraph("Next Steps", "Heading1"),
            _paragraph("- Review this report with the owner."),
            _paragraph("- Update open questions before sharing externally."),
        ]
    )
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


def _write_docx(
    path: Path,
    title: str,
    audience: str,
    findings: list[str],
    recommendations: list[str],
) -> None:
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
            _document_xml(title, audience, findings, recommendations),
        )
        docx.writestr("word/styles.xml", _styles_xml())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Word DOCX report.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--audience", default="General audience")
    parser.add_argument("--finding", action="append", default=[])
    parser.add_argument("--recommendation", action="append", default=[])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    findings = args.finding or ["No findings were provided."]
    recommendations = args.recommendation or ["No recommendations were provided."]
    output_name = _filename(args.output or args.title.lower().replace(" ", "-"))
    output_path = _workspace_dir() / output_name

    _write_docx(output_path, args.title, args.audience, findings, recommendations)
    print(str(output_path))


if __name__ == "__main__":
    main()
