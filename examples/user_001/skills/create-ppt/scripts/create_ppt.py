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
    name = value.strip() or "deck.pptx"
    if not name.endswith(".pptx"):
        name += ".pptx"
    return Path(name).name


def _slide_xml(title: str, body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        "<p:cSld><p:spTree>"
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr><a:spLocks noGrp="1"/>'
        '</p:cNvSpPr><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="685800" y="457200"/>'
        '<a:ext cx="7772400" cy="914400"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/>'
        '<a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="3600" b="1"/>'
        f"<a:t>{xml.escape(title)}</a:t>"
        "</a:r></a:p></p:txBody></p:sp>"
        '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Content"/><p:cNvSpPr><a:spLocks noGrp="1"/>'
        '</p:cNvSpPr><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="914400" y="1600200"/>'
        '<a:ext cx="7315200" cy="3657600"/></a:xfrm></p:spPr><p:txBody><a:bodyPr wrap="square"/>'
        '<a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="2400"/>'
        f"<a:t>{xml.escape(body)}</a:t>"
        "</a:r></a:p></p:txBody></p:sp>"
        "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    )


@dataclass
class Slide:
    title: str
    body: str


def _split_slide(value: str) -> Slide:
    if "::" not in value:
        return Slide(value.strip(), "")
    title, body = value.split("::", 1)
    return Slide(title.strip(), body.strip())


def _presentation_xml(slide_count: int) -> str:
    slide_ids = []
    for index in range(1, slide_count + 1):
        slide_ids.append(f'<p:sldId id="{255 + index}" r:id="rId{index}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        "<p:sldIdLst>"
        + "".join(slide_ids)
        + "</p:sldIdLst><p:sldSz cx=\"9144000\" cy=\"5143500\" type=\"screen16x9\"/>"
        '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    )


def _presentation_rels(slide_count: int) -> str:
    rels = []
    for index in range(1, slide_count + 1):
        rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="slides/slide{index}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def _content_types(slide_count: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
    ]
    for index in range(1, slide_count + 1):
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )


def _write_pptx(path: Path, title: str, subtitle: str, slides: list[Slide]) -> None:
    slide_count = 1 + len(slides)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as pptx:
        pptx.writestr("[Content_Types].xml", _content_types(slide_count))
        pptx.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="ppt/presentation.xml"/></Relationships>',
        )
        pptx.writestr("ppt/presentation.xml", _presentation_xml(slide_count))
        pptx.writestr("ppt/_rels/presentation.xml.rels", _presentation_rels(slide_count))
        pptx.writestr(
            "ppt/slides/slide1.xml",
            _slide_xml(title, subtitle),
        )
        for index, slide in enumerate(slides, start=2):
            pptx.writestr(
                f"ppt/slides/slide{index}.xml",
                _slide_xml(slide.title, slide.body),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a general PowerPoint PPTX deck.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--slide", action="append", default=[])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    slides = [_split_slide(item) for item in args.slide if item.strip()]
    if not slides:
        slides = [
            Slide("Overview", "Summarize the topic and why it matters."),
            Slide("Key Points", "Add the most important details for this audience."),
            Slide("Next Steps", "Close with the action, question, or takeaway."),
        ]
    output_name = _filename(args.output or args.title.lower().replace(" ", "-"))
    output_path = _workspace_dir() / output_name

    _write_pptx(output_path, args.title, args.subtitle, slides)
    print(str(output_path))


if __name__ == "__main__":
    main()
