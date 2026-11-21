# Advanced examples

Pieces of your own, each by extending one base class.

| File | Base class | What you write |
|---|---|---|
| `01_native_tool.py` | `CoreTool` | the schema and `execute(tool_request, tool_ctx)` |
| `02_custom_middleware.py` | `CoreMiddleware` | any of `turn`, `model`, `stream`, `tool` |
| `03_in_memory_session_store.py` | `CoreSessionStore` | `_load`, `_save`, `_list`, `_delete` |
| `04_custom_prompt_layer.py` | `CoreLayer` | the text of a part of the system prompt |
| `05_custom_embedding.py` | `CoreEmbedding` | `model_id` and `_embed` (no key needed) |
| `06_save_and_load_agent.py` | | an agent as YAML, with no secrets, and back |

Each one was run against a real model (DeepSeek through OpenRouter).
