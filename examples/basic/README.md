# Basic examples

| File | What it shows | Needs besides `OPENAI_API_KEY` |
|---|---|---|
| `01_basic_agent.py` | the smallest agent | |
| `02_openrouter_agent.py` | OpenRouter with fallback models | `OPENROUTER_API_KEY` instead |
| `03_function_tools.py` | a Python function as a tool | |
| `04_tool_approval.py` | a tool that waits for the user, and going on after | |
| `05_sessions.py` | saving a conversation and continuing it (run it twice) | |
| `06_structured_output.py` | an answer that follows a Pydantic model | |
| `07_streaming.py` | the answer as it is written | |
| `08_middleware_and_tracing.py` | logging and OpenTelemetry traces | |
| `09_concurrent_runs.py` | one agent, two users at once | |
| `10_local_sandbox.py` | a sandbox, its files and `BashTool` | |
| `11_memory.py` | facts about the user kept between sessions | |
| `12_skills.py` | a skill installed from GitHub | `GITHUB_TOKEN` if GitHub answers 403 |
| `13_mcp_tools.py` | tools from an MCP server, with its secret | `EXA_API_KEY` |
| `14_full_agent.py` | most pieces together | |

Checked by running each one with a model that is faked, so the framework's
part really runs; `12_skills.py` could not be checked that way (GitHub's rate
limit). None was run against a real model.
