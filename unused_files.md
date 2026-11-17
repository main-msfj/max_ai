# Archivos de `max_ai` que el agente nuevo no usa

Lista sacada el 2026-11-17 sobre el commit `8cc0b2f` más los cambios sin commit de ese día.

## Cómo se sacó

Un script siguió los `import` desde el agente nuevo (`max_ai/agents/agent.py`) y desde las piezas nuevas que un developer importa directo: clientes de OpenAI y OpenRouter, sandbox local y docker, skills, loop, compaction, memoria, MCP, capas de prompt, middlewares, harness y las tools nuevas. Todo archivo `.py` de `max_ai/` al que no se llega por ese camino sale aquí.

De 165 archivos que no se alcanzan (sobre 321 en total), abajo van agrupados por qué hacer con cada grupo.

Límites del método:

- Solo ve `import` escritos en el código. No ve un archivo que se lee como texto ni uno que se carga por nombre (`KNOWN_PROVIDERS` en `base/component.py`).
- Los `__init__.py` que solo re-exportan (`max_ai`, `max_ai.base`, `max_ai.core`, `max_ai.capabilities`, `max_ai.types`, `max_ai.errors`, `max_ai.agents`, `max_ai.capabilities.tools`) no cuentan como uso. Hoy varios de ellos importan archivos viejos; al borrar un archivo hay que quitar su línea ahí y su entrada en `KNOWN_PROVIDERS`.
- No se revisaron `tests/`, `examples/` ni `example-for-deployment/`: los que usan el agente viejo se van con él.
- Nada se borró. Antes de borrar un grupo conviene buscar su nombre en el repo y correr la suite.

## 1. No borrar: el script los marca, pero se usan

Se mandan al sandbox o al contenedor como texto, no con `import`.

- `max_ai/base/layout/script.py`
- `max_ai/capabilities/memory/index.py`
- `max_ai/capabilities/sandbox/docker/proxy.py`
- `max_ai/capabilities/tools/filesystem/edit_file/script.py`
- `max_ai/capabilities/tools/filesystem/read_file/script.py`
- `max_ai/capabilities/tools/filesystem/write_file/script.py`

## 2. Revisar uno por uno antes de borrar

Knowledge está pendiente ("lo haremos después") y se va a usar. Los demás son `__init__.py` de paquetes vivos o textos (`harness/messages.py`) que hoy solo toca el agente viejo; puede que sobren, pero no lo verifiqué.

- `max_ai/capabilities/clients/__init__.py`
- `max_ai/capabilities/knowledge/__init__.py`
- `max_ai/capabilities/knowledge/local/__init__.py`
- `max_ai/capabilities/knowledge/local/_model.py`
- `max_ai/capabilities/knowledge/local/_registry.py`
- `max_ai/capabilities/knowledge/mongodb/__init__.py`
- `max_ai/capabilities/knowledge/mongodb/_model.py`
- `max_ai/capabilities/knowledge/mongodb/_registry.py`
- `max_ai/capabilities/knowledge/sqlite/__init__.py`
- `max_ai/capabilities/knowledge/sqlite/_model.py`
- `max_ai/capabilities/knowledge/sqlite/_registry.py`
- `max_ai/capabilities/mongodb_vector.py`
- `max_ai/capabilities/sandbox/__init__.py`
- `max_ai/core/harness/__init__.py`
- `max_ai/core/harness/messages.py`
- `max_ai/core/middleware/__init__.py`
- `max_ai/core/model/__init__.py`

## 3. CLI: en arreglo

Hoy usa el agente viejo; se está pasando al nuevo. No se borra.

- `max_ai/cli/__init__.py`
- `max_ai/cli/app.py`
- `max_ai/cli/blocks.py`
- `max_ai/cli/events.py`
- `max_ai/cli/theme.py`
- `max_ai/cli/widgets.py`

## 4. Borrar ya: borradores y copias

Nadie los importa, ni el agente viejo.

- `max_ai/base/clients copy.py`
- `max_ai/base/tools_draft.py`
- `max_ai/core/tool/dispatcher_draft.py`
- `max_ai/core/tool/state_draft.py`

- `max_ai/base/skills_previos.py` **no** va aquí: el agente viejo toma de ahí `CoreSkillRegistry`; se borra junto con él (grupo 6).

## 5. Borrar cuando `max_ai.agents.Agent` pase al nuevo: cuarentena

Código viejo movido a `to_delete/` para que el agente viejo siga vivo (44 archivos).

- `max_ai/capabilities/clients/to_delete/__init__.py`
- `max_ai/capabilities/clients/to_delete/_schema.py`
- `max_ai/capabilities/clients/to_delete/openai/__init__.py`
- `max_ai/capabilities/clients/to_delete/openai/_model.py`
- `max_ai/capabilities/clients/to_delete/openai/client.py`
- `max_ai/capabilities/clients/to_delete/openrouter/__init__.py`
- `max_ai/capabilities/clients/to_delete/openrouter/_model.py`
- `max_ai/capabilities/clients/to_delete/openrouter/client.py`
- `max_ai/capabilities/compaction/to_delete/__init__.py`
- `max_ai/capabilities/compaction/to_delete/summary/__init__.py`
- `max_ai/capabilities/compaction/to_delete/summary/_model.py`
- `max_ai/capabilities/compaction/to_delete/summary/_strategy.py`
- `max_ai/capabilities/compaction/to_delete/window/__init__.py`
- `max_ai/capabilities/compaction/to_delete/window/_model.py`
- `max_ai/capabilities/compaction/to_delete/window/_strategy.py`
- `max_ai/capabilities/memory/to_delete/__init__.py`
- `max_ai/capabilities/memory/to_delete/local/__init__.py`
- `max_ai/capabilities/memory/to_delete/local/_model.py`
- `max_ai/capabilities/memory/to_delete/local/_registry.py`
- `max_ai/capabilities/memory/to_delete/mongodb/__init__.py`
- `max_ai/capabilities/memory/to_delete/mongodb/_model.py`
- `max_ai/capabilities/memory/to_delete/mongodb/_registry.py`
- `max_ai/capabilities/memory/to_delete/sqlite/__init__.py`
- `max_ai/capabilities/memory/to_delete/sqlite/_model.py`
- `max_ai/capabilities/memory/to_delete/sqlite/_registry.py`
- `max_ai/capabilities/tools/to_delete/__init__.py`
- `max_ai/capabilities/tools/to_delete/bash/__init__.py`
- `max_ai/capabilities/tools/to_delete/bash/_permissions.py`
- `max_ai/capabilities/tools/to_delete/bash/_tool.py`
- `max_ai/capabilities/tools/to_delete/bash/constant.py`
- `max_ai/capabilities/tools/to_delete/file_system/__init__.py`
- `max_ai/capabilities/tools/to_delete/file_system/_capability.py`
- `max_ai/capabilities/tools/to_delete/file_system/_toolset.py`
- `max_ai/capabilities/tools/to_delete/file_system/constant.py`
- `max_ai/capabilities/tools/to_delete/filesystem/__init__.py`
- `max_ai/capabilities/tools/to_delete/filesystem/_base.py`
- `max_ai/capabilities/tools/to_delete/filesystem/edit/__init__.py`
- `max_ai/capabilities/tools/to_delete/filesystem/edit/_base.py`
- `max_ai/capabilities/tools/to_delete/filesystem/read/__init__.py`
- `max_ai/capabilities/tools/to_delete/filesystem/read/_base.py`
- `max_ai/capabilities/tools/to_delete/filesystem/write/__init__.py`
- `max_ai/capabilities/tools/to_delete/filesystem/write/_base.py`
- `max_ai/capabilities/tools/to_delete/filesystem/read/script.py`
- `max_ai/capabilities/tools/to_delete/filesystem/script.py`

## 6. Borrar cuando `max_ai.agents.Agent` pase al nuevo: piezas del agente viejo

Agente viejo, su loop (`reasoning`), workspace, executor, gates, bus de eventos, capas viejas, y los errores y tipos que solo ellos usan (70 archivos). El sandbox, el loop nuevo y `LoopHook` / `ToolHook` los reemplazan.

- `max_ai/agents/agent_old.py`
- `max_ai/base/clients_old.py`
- `max_ai/base/compaction_old.py`
- `max_ai/base/context.py`
- `max_ai/base/executor.py`
- `max_ai/base/memory.py`
- `max_ai/base/reasoning.py`
- `max_ai/base/workspace.py`
- `max_ai/capabilities/completion_gate/__init__.py`
- `max_ai/capabilities/completion_gate/_model.py`
- `max_ai/capabilities/completion_gate/gate.py`
- `max_ai/capabilities/context/__init__.py`
- `max_ai/capabilities/context/local/__init__.py`
- `max_ai/capabilities/context/local/_model.py`
- `max_ai/capabilities/context/local/_registry.py`
- `max_ai/capabilities/context/sqlite/__init__.py`
- `max_ai/capabilities/context/sqlite/_model.py`
- `max_ai/capabilities/context/sqlite/_registry.py`
- `max_ai/capabilities/executor/__init__.py`
- `max_ai/capabilities/executor/docker/__init__.py`
- `max_ai/capabilities/executor/docker/_executor.py`
- `max_ai/capabilities/executor/docker/_model.py`
- `max_ai/capabilities/executor/local/__init__.py`
- `max_ai/capabilities/executor/local/_executor.py`
- `max_ai/capabilities/executor/local/_model.py`
- `max_ai/capabilities/executor/modal/__init__.py`
- `max_ai/capabilities/executor/modal/_executor.py`
- `max_ai/capabilities/executor/modal/_model.py`
- `max_ai/capabilities/reasoning/__init__.py`
- `max_ai/capabilities/reasoning/guards.py`
- `max_ai/capabilities/reasoning/react/__init__.py`
- `max_ai/capabilities/reasoning/react/loop.py`
- `max_ai/capabilities/stacks/context_layer.py`
- `max_ai/capabilities/tools/workspace/__init__.py`
- `max_ai/capabilities/tools/workspace/_toolset.py`
- `max_ai/capabilities/workspace/__init__.py`
- `max_ai/capabilities/workspace/_remote.py`
- `max_ai/capabilities/workspace/azure_blob/__init__.py`
- `max_ai/capabilities/workspace/azure_blob/_model.py`
- `max_ai/capabilities/workspace/azure_blob/_workspace.py`
- `max_ai/capabilities/workspace/local/__init__.py`
- `max_ai/capabilities/workspace/local/_filesystem.py`
- `max_ai/capabilities/workspace/local/_model.py`
- `max_ai/capabilities/workspace/local/_workspace.py`
- `max_ai/capabilities/workspace/minio/__init__.py`
- `max_ai/capabilities/workspace/minio/_model.py`
- `max_ai/capabilities/workspace/minio/_workspace.py`
- `max_ai/core/environment/__init__.py`
- `max_ai/core/environment/manager.py`
- `max_ai/core/events_bus/__init__.py`
- `max_ai/core/events_bus/bus.py`
- `max_ai/core/events_bus/hooks.py`
- `max_ai/core/executor/__init__.py`
- `max_ai/core/executor/process.py`
- `max_ai/core/executor/remote.py`
- `max_ai/core/executor/sync.py`
- `max_ai/core/runtime/__init__.py`
- `max_ai/core/runtime/status.py`
- `max_ai/core/stacks/__init__.py`
- `max_ai/core/stacks/container.py`
- `max_ai/errors/agent.py`
- `max_ai/errors/capabilities.py`
- `max_ai/errors/compaction.py`
- `max_ai/errors/executor.py`
- `max_ai/errors/manager.py`
- `max_ai/errors/memory.py`
- `max_ai/errors/workspace.py`
- `max_ai/types/runtime.py`
- `max_ai/types/stacks.py`
- `max_ai/types/workspace.py`

## 7. Por decidir: guardar sesiones y cuotas

El agente nuevo no guarda la conversación por sí mismo (recibe `save=`), así que no usa estos almacenes. `quota_store` lo usa `BudgetMiddleware` solo si se configura una cuota. Sirven si se quiere ofrecer un guardado listo; si no, se borran.

- `max_ai/capabilities/quota_store/__init__.py`
- `max_ai/capabilities/quota_store/local/__init__.py`
- `max_ai/capabilities/quota_store/local/_model.py`
- `max_ai/capabilities/quota_store/local/_store.py`
- `max_ai/capabilities/quota_store/mongodb/__init__.py`
- `max_ai/capabilities/quota_store/mongodb/_model.py`
- `max_ai/capabilities/quota_store/mongodb/_store.py`
- `max_ai/capabilities/session_store/__init__.py`
- `max_ai/capabilities/session_store/local/__init__.py`
- `max_ai/capabilities/session_store/local/_model.py`
- `max_ai/capabilities/session_store/local/_store.py`
- `max_ai/capabilities/session_store/mongodb/__init__.py`
- `max_ai/capabilities/session_store/mongodb/_model.py`
- `max_ai/capabilities/session_store/mongodb/_store.py`

## 8. Sin clasificar

No los alcanza el agente nuevo y no revisé quién más los usa.

- `max_ai/capabilities/mcp/config.py`
- `max_ai/core/model/observation.py`
- `max_ai/core/observation/__init__.py`

## También dentro de archivos que sí se usan

- `max_ai/types/run_context.py`: campos del agente viejo (`compaction`, `message_history`, `plan`, `completion_state`, `runtime_state`...).
- `max_ai/core/model/agent.py` (`AgentSpec`): `workspace`, `executor`, `compaction`, `policy`.
- `max_ai/types/agent_response.py`: el agente nuevo devuelve `LoopState`; `AgentResponse` solo lo usan el agente viejo y los middlewares (`on_run_end`).
- `max_ai/base/middleware.py` y los tres middlewares: cambian de firma (opción B) junto con el paso final.
- `max_ai/base/component.py`: entradas de `KNOWN_PROVIDERS` que apuntan a `to_delete/` y a clases viejas.
