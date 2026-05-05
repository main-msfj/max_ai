# Tool Repositories

Plain Python functions intended to be mounted into a Docker worker or
wrapped with `FunctionAsTool` locally.

Rules for this folder:

- functions must be top-level and importable
- functions should be read-only unless explicitly reviewed
- no delete/remove/destructive operations
- inputs and outputs should be JSON-friendly

Example import path:

```text
tool_repositories.files:list_directory
tool_repositories.text:extract_markdown_outline
tool_repositories.data:profile_csv
```

