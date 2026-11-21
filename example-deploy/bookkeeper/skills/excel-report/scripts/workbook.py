"""Write a summary as an Excel workbook: workbook.py SUMMARY.json REPORT.xlsx

Only the standard library: an .xlsx file is a zip of XML files.
"""

import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

PLAIN, BOLD, MONEY, BOLD_MONEY = 0, 1, 2, 3  # the styles of STYLES, by position
STYLES = (
    '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<numFmts count="1"><numFmt numFmtId="164" formatCode="#,##0.00"/></numFmts>'
    '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
    '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
    '<fills count="2"><fill><patternFill patternType="none"/></fill>'
    '<fill><patternFill patternType="gray125"/></fill></fills>'
    "<borders count=\"1\"><border/></borders>"
    '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
    '<cellXfs count="4"><xf/><xf fontId="1" applyFont="1"/>'
    '<xf numFmtId="164" applyNumberFormat="1"/>'
    '<xf numFmtId="164" fontId="1" applyFont="1" applyNumberFormat="1"/></cellXfs>'
    '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
    "</styleSheet>"
)
MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE = "http://schemas.openxmlformats.org/package/2006"
KIND = "application/vnd.openxmlformats-officedocument.spreadsheetml"


def cell(place: str, value: object, style: int) -> str:
    """One cell: a number, a formula (it starts with ``=``) or a text."""
    if isinstance(value, (int, float)):
        return f'<c r="{place}" s="{style}"><v>{value}</v></c>'
    if isinstance(value, str) and value.startswith("="):
        return f'<c r="{place}" s="{style}"><f>{escape(value[1:])}</f></c>'
    return f'<c r="{place}" s="{style}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def sheet(rows: list[list[tuple[object, int]]]) -> str:
    """A sheet from its rows; each cell is its value and its style."""
    lines = "".join(
        f'<row r="{number}">'
        + "".join(cell(f"{'AB'[column]}{number}", *given) for column, given in enumerate(row))
        + "</row>"
        for number, row in enumerate(rows, start=1)
    )
    widths = '<cols><col min="1" max="1" width="44" customWidth="1"/>'
    widths += '<col min="2" max="2" width="16" customWidth="1"/></cols>'
    return f'<worksheet xmlns="{MAIN}">{widths}<sheetData>{lines}</sheetData></worksheet>'


def table(header: str, entries: list[dict]) -> list[list[tuple[object, int]]]:
    """One line per entry, the largest first, and a total written as a formula."""
    rows: list[list[tuple[object, int]]] = [[(header, BOLD), ("Amount", BOLD)]]
    for entry in sorted(entries, key=lambda entry: -entry["amount"]):
        rows.append([(entry["name"], PLAIN), (entry["amount"], MONEY)])
    rows.append([("Total", BOLD), (f"=SUM(B2:B{len(rows)})", BOLD_MONEY)])
    return rows


def main(source: str, target: str) -> None:
    summary = json.loads(Path(source).read_text())

    # 1. Sales and expenses, each with its total.
    sales = table("Product", summary["sales"])
    expenses = table("Category", summary["expenses"])

    # 2. The summary reads those totals, so it follows any change.
    first: list[list[tuple[object, int]]] = [
        [(f'{summary["business"]}: {summary["title"]}', BOLD), (f'in {summary["currency"]}', PLAIN)],
        [("Sales", PLAIN), (f"=Sales!B{len(sales)}", MONEY)],
        [("Expenses", PLAIN), (f"=Expenses!B{len(expenses)}", MONEY)],
        [("Profit", BOLD), ("=B2-B3", BOLD_MONEY)],
        [],
    ]
    first += [[(note, PLAIN)] for note in summary.get("notes", [])]

    # 3. The workbook: its sheets and the files that tie them together.
    sheets = {"Summary": first, "Sales": sales, "Expenses": expenses}
    numbered = list(enumerate(sheets, start=1))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as book:
        book.writestr("[Content_Types].xml", (
            f'<Types xmlns="{PACKAGE}/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/xl/workbook.xml" ContentType="{KIND}.sheet.main+xml"/>'
            f'<Override PartName="/xl/styles.xml" ContentType="{KIND}.styles+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{n}.xml" ContentType="{KIND}.worksheet+xml"/>'
                for n, _ in numbered
            ) + "</Types>"
        ))
        book.writestr("_rels/.rels", (
            f'<Relationships xmlns="{PACKAGE}/relationships"><Relationship Id="rId1" '
            f'Type="{RELS}/officeDocument" Target="xl/workbook.xml"/></Relationships>'
        ))
        book.writestr("xl/workbook.xml", (
            f'<workbook xmlns="{MAIN}" xmlns:r="{RELS}"><sheets>'
            + "".join(f'<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>' for n, name in numbered)
            + '</sheets><calcPr fullCalcOnLoad="1"/></workbook>'
        ))
        book.writestr("xl/_rels/workbook.xml.rels", (
            f'<Relationships xmlns="{PACKAGE}/relationships">'
            + "".join(
                f'<Relationship Id="rId{n}" Type="{RELS}/worksheet" Target="worksheets/sheet{n}.xml"/>'
                for n, _ in numbered
            )
            + f'<Relationship Id="rId{len(numbered) + 1}" Type="{RELS}/styles" Target="styles.xml"/>'
            "</Relationships>"
        ))
        book.writestr("xl/styles.xml", STYLES)
        for n, name in numbered:
            book.writestr(f"xl/worksheets/sheet{n}.xml", sheet(sheets[name]))
    print(f"Workbook written to {target}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Use: workbook.py SUMMARY.json REPORT.xlsx")
    main(sys.argv[1], sys.argv[2])
