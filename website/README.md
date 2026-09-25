# Max AI website

Static English documentation for Max AI's pluggable agent harness. The Flask app in `app.py` serves the same files locally; Azure Static Web Apps deploys the `website/` directory directly.

## Run locally

From the repository root, run `python -m http.server 8000 --directory website`. Open `http://127.0.0.1:8000`. If Flask is installed, `python -m website.app` also serves the site and adds a `/health` route; `WEBSITE_HOST`, `WEBSITE_PORT`, and `WEBSITE_DEBUG` configure that server.

## Edit the site

Read [website-design.md](docs/website-design.md) for the page tree and content boundaries. Edit the HTML files directly. Shared styles and interactions live in `assets/`. Every main section needs a stable `id` for links and search.

Each example has a separate HTML page generated from its Python source and the English descriptions in `example_catalog.json`. Update that catalog to change the explanations. The shared example layout is in `templates/example.html.template`.

Store pages by section in `overview/`, `agents/`, `basecomponents/`, `core/`, `capabilities/`, `types/`, `events/`, `examples/`, and `cli/`. Shared images live in `images/`, browser assets in `assets/`, source catalogs and generated search data in `data/`, and page-building code in `scripts/`.

After editing examples or capability pages, run:

```sh
python website/scripts/generate_capability_pages.py
python website/scripts/generate_example_pages.py
python website/scripts/generate_reference_pages.py
python website/scripts/sync_navigation.py
python website/scripts/build_search_index.py
```

`scripts/sync_navigation.py` owns the sidebar tree for all HTML pages. `scripts/build_search_index.py` builds `data/search-index.json` from those pages with the standard library. Keep `docs/architecture.md` accurate.

The site uses local HTML, CSS, JavaScript, and SVG without a frontend build or external font. Search, theme choice, mobile navigation, and code-copy controls run in the browser.
