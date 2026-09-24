# Basic Examples

Each file demonstrates one feature. From the repository root, replace
`api_key="YOUR_API_KEY"` in the example you want to run, then use:

```sh
uv run python -m examples.basic.01_basic_agent
```

OpenAI examples use the `gpt-5.6-luna` model. The full agent and local
knowledge search require `maxai[embeddings]`. The Ollama example requires
Ollama and a downloaded model. The MCP example requires an HTTP MCP server at
`http://localhost:8000/mcp`. `15_full_agent` combines tools, memory, knowledge,
skills, compaction, middleware, and sessions in one configuration.
