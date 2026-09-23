"""Build the local documentation search index without third-party dependencies."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

WEBSITE = Path(__file__).resolve().parent


class DocumentationParser(HTMLParser):
    def __init__(self, filename: str) -> None:
        super().__init__(convert_charrefs=True)
        self.url = "/" if filename == "index.html" else "/" + filename
        self.page_title = ""
        self.in_title = False
        self.section: dict | None = None
        self.in_heading = False
        self.in_paragraph = False
        self.heading: list[str] = []
        self.paragraph: list[str] = []
        self.text: list[str] = []
        self.entries: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "section" and attributes.get("id"):
            self.section = {
                "url": self.url + "#" + str(attributes["id"]),
                "page": self.page_title.split(" · ")[0].strip(),
                "title": "",
                "summary": "",
                "featured": "hero" in (attributes.get("class") or "").split(),
            }
            self.text = []
        if self.section is None:
            return
        if tag in {"h1", "h2"} and not self.section["title"]:
            self.in_heading = True
            self.heading = []
        if tag == "p" and not self.section["summary"]:
            self.in_paragraph = True
            self.paragraph = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if self.section is None:
            return
        if tag in {"h1", "h2"} and self.in_heading:
            self.section["title"] = " ".join(" ".join(self.heading).split()).removesuffix(" #")
            self.in_heading = False
        if tag == "p" and self.in_paragraph:
            self.section["summary"] = " ".join(" ".join(self.paragraph).split())[:210]
            self.in_paragraph = False
        if tag == "section":
            self.section["text"] = " ".join(" ".join(self.text).split())
            self.entries.append(self.section)
            self.section = None

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.page_title += data
        if self.section is not None:
            self.text.append(data)
            if self.in_heading:
                self.heading.append(data)
            if self.in_paragraph:
                self.paragraph.append(data)


def main() -> None:
    entries = []
    for filename in (
        "index.html", "agent.html", "prompt-layers.html",
        "tools.html", "capabilities.html", "integrations.html",
    ):
        parser = DocumentationParser(filename)
        parser.feed((WEBSITE / filename).read_text())
        entries.extend(parser.entries)
    (WEBSITE / "search-index.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Indexed {len(entries)} sections across six documentation pages.")


if __name__ == "__main__":
    main()

