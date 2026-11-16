# Cosas por borrar

Lista viva de lo que sobra en el repo. Se anota aquí cuando se decide, y se tacha o se quita la línea cuando se borra.

Última actualización: 2026-11-16.

## Decidido: se borra

| Qué | Dónde | Por qué | Qué lo bloquea |
|---|---|---|---|
| `ContextLayer` | `capabilities/stacks/context_layer.py`, `prompts/ContextLayer.j2` | nadie la pone en el stack | la nombra `core/stacks/container.py` |
| `SessionStateLayer` | `capabilities/stacks/session_state_layer.py`, `prompts/SessionStateLayer.j2` | la reemplaza `SessionLayer`; el plan sale del system prompt | la usan `agent.py` y `reasoning.py` |
| `validate_name`, `validate_description` | `base/tools.py` | el tool debe ser independiente; valida quien arma el agente | las tools que las anulan (MCP) |
| `policy_subject`, `permission_subjects`, `matches` | `base/tools.py` | los permisos son del hook del harness | `core/policy.py` |
| `runs_commands` | `base/tools.py` | venía del executor viejo; ahora lo dice `sandbox` | `core/policy.py`, `agent.py` |
| `CoreRuntimeTool` | `base/tools.py` | lo reemplaza `sandbox = True` | `tools/bash/_tool.py` hereda de él; se exporta en `base/__init__.py` y `tools/__init__.py` |
| `_check_type`, `_JSON_TYPE_MAP` | `base/tools.py` | nadie los usa; la validación la hace `jsonschema` | nada |

## Candidatos: falta decidir

| Qué | Dónde | Por qué |
|---|---|---|
| Copias de respaldo | `base/tools_draft.py`, `core/tool/dispatcher_draft.py`, `core/tool/state_draft.py` | se guardaron por si había que volver a mirar algo |
| `file_system copy/` | `capabilities/tools/file_system copy/` | carpeta duplicada |
| `config.py` de MCP | `capabilities/mcp/config.py` | repite lo que exporta `__init__.py` |
| `integration.py` de MCP | `capabilities/mcp/integration.py` | una función que solo usa un test |
| `CompletionCheck` | `base/completion_gate.py` | nadie lo usa, solo se exporta |
| `on_call_tool`, `on_tool_response`, `on_model_response` | `base/completion_gate.py` | repiten lo que hacen `ToolHook` y el middleware |
| `PromptBuilder` viejo del agente | `_prompt_variables` y `_build_prompt_stack` en `agents/agent.py` | los reemplaza `PromptBuilder` |
| `waits_for_user` | `base/tools.py` | depende de cómo quede `ask_user` |
| `read_only` | `base/tools.py` | hoy decide qué llamadas corren en paralelo y cuáles se repiten en MCP |

## En espera

| Qué | Dónde | Por qué espera |
|---|---|---|
| `MemoryLayer` | `capabilities/stacks/memory_layer.py` | hasta decidir el diseño de memoria |
| Executor viejo | `base/executor.py`, `capabilities/executor/` | lo reemplaza el sandbox; lo usa `agent.py` |
