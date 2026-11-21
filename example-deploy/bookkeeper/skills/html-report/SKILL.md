---
name: html-report
description: Turns the summary of a month (a JSON file) into a report as one web page. Use it to hand in a monthly close the user can open in the browser.
---
# html-report

1. Write `summary.json` in your working directory, with this shape and every
   amount as a number in one currency:

   ```json
   {
     "title": "October 2026",
     "business": "Name of the business",
     "currency": "GTQ",
     "cards": [{"label": "Sales", "value": 0}, {"label": "Expenses", "value": 0}, {"label": "Profit", "value": 0}],
     "sales": [{"name": "Product", "amount": 0}],
     "expenses": [{"name": "Category", "amount": 0}],
     "notes": ["USD converted at 7.70 (source: ...)"]
   }
   ```

2. Run `python3 scripts/render.py SUMMARY.json REPORT.html`, giving the script
   by its full path inside this skill folder and both files by their full path
   in your working directory.
3. Tell the user where the page is. Do not write the HTML by hand: the
   template keeps every report alike.
4. A monthly close is also handed in as a workbook: read the excel-report
   skill and make it from the same `summary.json`.

Rules: at most four cards; the rows ordered from the largest; every
assumption and the source of an exchange rate go in `notes`. Compute the
numbers with a script that reads the CSV files, never by hand. An entry
with no category is never guessed: ask the user what it is.
