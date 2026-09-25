"""Build an examples index and one static page per example; never execute examples."""
import json
from html import escape
from pathlib import Path

from site_paths import page_path, page_url
from sync_navigation import sidebar

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent


def main():
    entries = json.loads((HERE / 'data' / 'example_catalog.json').read_text())
    template = (HERE / 'templates/example.html.template').read_text()

    def write(filename, title, description, content, sections):
        links = ''.join(f'<a class="toc-link" href="#{anchor}">{escape(label)}</a>' for anchor, label in sections)
        main = ('<main class="main-content" id="main-content" tabindex="-1">'
                '<nav class="breadcrumb" aria-label="Breadcrumb"><a href="/overview/index.html">Docs</a><span>›</span>'
                '<a href="/examples/index.html">Examples</a><span>›</span>' + escape(title) + '</nav>'
                '<details class="mobile-toc"><summary>On this page</summary><nav aria-label="Page sections">' + links + '</nav></details>'
                + content + '<footer class="footer">Max AI · Python agent framework</footer></main>')
        toc = '<aside class="page-toc" aria-label="On this page"><p>On this page</p><nav>' + links + '</nav></aside>'
        target_path = page_path(filename)
        replacements = {'MAIN': main, 'TOC': toc, 'SIDEBAR': sidebar(target_path), 'TITLE': escape(title),
                        'DESCRIPTION': escape(description), 'PAGE': target_path}
        page = template
        for key, value in replacements.items():
            page = page.replace('{{' + key + '}}', value)
        target = HERE / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page)

    catalog = '<section class="hero content-section page-hero" id="examples-top"><h1>Examples</h1><p class="hero-copy">Explore minimal Python examples. Each example has its own page with an explanation and the complete code.</p></section>'
    for level in ('Basic', 'Advanced'):
        catalog += f'<section class="section-block content-section" id="{level.lower()}-examples"><h2>{level} examples</h2><div class="example-index">'
        for entry in entries:
            if entry['level'] != level:
                continue
            slug = entry['slug']
            catalog += f'<article id="{slug}"><h3><a href="{page_url(f"example-{slug}.html")}">{escape(entry["title"])}</a></h3><p>{escape(entry["what"])}</p></article>'
        catalog += '</div></section>'
    write('examples.html', 'Examples', 'Minimal Python examples, with explanations and complete source code.', catalog,
          [('examples-top', 'Overview'), ('basic-examples', 'Basic examples'), ('advanced-examples', 'Advanced examples')])

    for entry in entries:
        slug = entry['slug']
        code = (ROOT / entry['source']).read_text()
        content = f'<section class="hero content-section page-hero" id="{slug}"><p class="eyebrow">{entry["level"]} example</p><h1>{escape(entry["title"])}</h1><p class="hero-copy">{escape(entry["what"])}</p></section>'
        content += f'<section class="section-block content-section" id="how-it-works"><h2>How it works</h2><p>{escape(entry["how"])}</p></section>'
        content += '<section class="section-block content-section" id="code"><h2>Complete example</h2><p>Read or copy the code below. This documentation page does not execute it.</p><div class="code-card"><div class="code-head"><span>Python</span><button class="copy-button" type="button" aria-label="Copy code">Copy</button></div><pre tabindex="0"><code class="language-python">' + escape(code) + '</code></pre></div></section>'
        if slug == 'serialize-agent':
            serialized = '''{
  "provider": "max_ai.agents.agent.Agent",
  "config": {
    "name": "StoredAssistant",
    "description": "An agent ready to be stored.",
    "instructions": "Answer in English.",
    "client": {
      "provider": "max_ai.capabilities.clients.openai.client.OpenAIChatCompletionClient",
      "component_type": "model",
      "version": 1,
      "component_version": 1,
      "label": "OpenAIChatCompletionClient",
      "config": {
        "model": "gpt-5.6-luna",
        "api_key_env": "OPENAI_API_KEY"
      }
    },
    "reasoning": {"provider": "...", "config": {"...": "..."}},
    "workspace": {"provider": "...", "config": {"...": "..."}},
    "executor": {"provider": "...", "config": {"...": "..."}},
    "toolset": [],
    "mcp": [],
    "gates": [],
    "middlewares": [],
    "prompt_layers": []
  }
}'''
            content += '<section class="section-block content-section" id="serialized-output"><h2>Serialized agent</h2><p>The saved JSON stores the Agent and each serializable component by provider and configuration. The API key itself is never included; only its environment variable name is saved. This abbreviated view shows the shape of the file.</p><div class="code-card"><div class="code-head"><span>agent.json · abbreviated</span></div><pre tabindex="0"><code class="language-json">' + escape(serialized) + '</code></pre></div></section>'
        content += '<nav class="page-pagination" aria-label="Examples navigation"><a href="/examples/index.html"><small>Examples</small><strong>Browse all examples</strong></a></nav>'
        sections = [(slug, 'What it does'), ('how-it-works', 'How it works'), ('code', 'Complete example')]
        if slug == 'serialize-agent':
            sections.append(('serialized-output', 'Serialized agent'))
        write(f'example-{slug}.html', entry['title'], entry['what'], content, sections)
    print(f'Built {len(entries)} individual example pages and the examples index.')


if __name__ == '__main__':
    main()
