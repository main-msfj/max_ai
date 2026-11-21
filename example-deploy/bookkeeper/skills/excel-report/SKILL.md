---
name: excel-report
description: Turns the summary of a month (a JSON file) into an Excel workbook with its totals as formulas. Use it when the user wants the monthly close as a spreadsheet.
---
# excel-report

The script needs nothing installed: it uses only Python's standard library.

1. Use the same `summary.json` the html-report skill asks for.
2. Run `python3 scripts/workbook.py SUMMARY.json REPORT.xlsx`, giving the script
   by its full path inside this skill folder and both files by their full path
   in your working directory.
3. Tell the user where the workbook is.

The workbook has three sheets: Summary, Sales and Expenses. The totals are
formulas, so the user can change a number and see the result.
