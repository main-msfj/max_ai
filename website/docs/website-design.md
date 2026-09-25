# Website design map

The site is static English documentation. Keep pages grouped by subject, explanations concise, and code examples focused.

```text
website/
├── overview/              Overview and installation
├── agents/                Public Agent API and lifecycle
├── basecomponents/        Contracts defined by max_ai.base
├── core/                  Runtime, events flow concepts, integrations, prompt layers, roadmap
├── capabilities/          Implementations grouped by category and backend
├── types/                 Runtime data types, including RunContext
├── events/                Runtime event reference
├── examples/
│   ├── basic/             One page per basic example
│   └── advanced/          One page per advanced example
├── cli/                   Command line guide
├── images/                Architecture and CLI image assets
├── assets/                CSS, JavaScript, and site icon
├── data/                  Example catalog and generated search index
├── templates/             Shared HTML templates
├── scripts/               Generators, route mapping, navigation, and search indexing
├── docs/                  Architecture note and this site map
└── app.py                 Optional local Flask server
```

The root `index.html` is the static hosting entry point and forwards visitors to `overview/index.html`. Link to canonical pages in the folders above.

`core/message-flow.html` is a standalone, full-width visual walkthrough using `images/max_ai.png`. Overview and Loop & conversations link to it and to `max_ai_animation.html`, the standalone interactive animation. Both visual pages provide a return link to the documentation.

The documentation mirrors the package boundary: `max_ai.base` defines contracts, `max_ai.core` holds runtime types and orchestration, `max_ai.capabilities` supplies implementations, and `max_ai.agents.Agent` composes them. Agent serialization stores reusable configuration; `RunContext` is per conversation and must be stored separately. Serialized examples must never include secret values.

`scripts/site_paths.py` maps logical source names to canonical files. `scripts/sync_navigation.py` owns the sidebar and normalizes page outline links. After editing generated pages, run the generators, navigation sync, and search index build in that order.
