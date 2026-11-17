# Lo que queda por hacer

Rama `feat/workspace-new-approach`, último commit `209deed` (2026-11-17).

## Dónde estamos

- `max_ai/agents/agent.py` es el agente nuevo: guarda sus piezas, valida que cada una sea de su clase Core (`_validate`), aplica las reglas del harness (`_run_harness_validation`) y se guarda y reconstruye (`_to_config`, `_from_config`). 
- `max_ai/agents/agent_old.py` es el agente viejo; `from max_ai.agents import Agent` sigue dando el viejo.
- `max_ai/core/harness/rules.py` tiene la clase `Harness` con las reglas; los números viven en `max_ai/config.py` (`harness_*`, `summary_model`).
- Loop (`base/loop.py`), compaction (`base/compaction.py`) y conteo de tokens (`ToolRegistry.tokens`, `PromptBuilder.tokens`, `CoreLoop.measure`) están hechos y probados con tests.

## Hecho el 2026-11-17 (subido, último commit `209deed`)

- `hooks.py`: `HarnessLoopHook` (índice de memoria tras compactar, guardar `ctx` tras cada tanda de tools, tareas de MCP como notas) y `HarnessToolHook` (memoria de solo lectura, middlewares alrededor de las tools, permisos).
- `agent.py` ya corre: `run` (devuelve `LoopState`), `run_stream_events`, `close` (espera a las corridas vivas), candado por sesión, MCP conectado una vez, sandbox por usuario con skills.
- Prompt: capas nativas siempre (`AgentPolicyLayer` con name, description e instructions; `TaskAnalysisLayer`, `RenderingLayer`, `ToolsLayer`, `SessionLayer`), más `WorkspaceLayer`, `SkillsLayer`, `MemoryFilesLayer` (nueva) y `KnowledgeLayer` según las piezas; en `STANDARD_ORDER`.
- Reglas nuevas: tope de 20 000 tokens para todas las tools (`harness_tools_max_tokens`), prompt revisado en cada corrida.
- Permisos (`core/harness/permissions.py`): reglas `allow` / `ask` / `deny` del usuario en `.agent/setting.json`; lista fija del harness que siempre pregunta (`rm`, `sudo`, `git push --force`, `| sh`, `$(...)`, tocar `setting.json`); `sandbox.add_permission` guarda una regla.

## Siguiente, en orden

1. **Middlewares (decidido: opción B)**: cambiar la firma de `CoreMiddleware` para que reciba `ctx` y `LoopState` en vez de `AgentResponse` y `ModelRequest`, y ajustar `TracingMiddleware`, `LoggingMiddleware` y `BudgetMiddleware`. Llamarlos en: `Agent._turn` (inicio, fin y error de corrida), `HarnessLoopHook.before_model` / `after_model` (modelo) y `run_stream_events` (trozos). Hoy solo ven las tools. **Sin empezar.**
2. Probar en vivo el camino completo de una aprobación: `agent.approve(ctx, call_id, always=True)` y luego `agent.run(None, ctx)`. `approve` y `run_stream` solo tienen tests con modelo falso.
3. De último: cambiar `max_ai.agents.Agent` al nuevo y borrar lo viejo: `agent_old.py`, `compaction_old.py`, `clients_old.py`, `reasoning.py`, carpetas `to_delete/`, y los campos `workspace`, `executor`, `compaction`, `policy` de `AgentSpec`.

Hecho después del resumen anterior: `Agent.run_stream` (solo texto) y `Agent.approve(ctx, call_id, always=...)`, que guarda la regla por la primera palabra del comando (`bash(pytest*)`) o la tool entera; lo peligroso nunca se guarda. Bash puede escribir en la memoria de solo lectura: es intencional.

## Sin conectar todavía

- `knowledge` y `gates` se aceptan y se guardan; nada los usa.
- Los permisos detectan por texto: un `rm` dentro de un script no se ve. Sin sandbox solo aplica la lista fija.

## Puntos a revisar

- `Harness.model_rules` cambia el `config` del cliente que recibe (ventana 128 000, salida 12 800); afecta al cliente también fuera del agente.
- `Memory` no es una clase Core ni un componente; se guarda como `{"read_only": ...}`.
- Una capa de prompt propia no se reconstruye sin `allow_providers`.
- La compaction nueva y el reparto de la ventana (`ctx.context`) no se han corrido en vivo desde el último cambio.
- Loop: faltan los escenarios de segunda pasada (stream cortado, alternancia A-B-A-B, cadenas de tools que fallan, texto cortado por longitud, fallo del hook).
- Una conversación compactada puede empezar con mensaje del asistente; Anthropic lo va a rechazar.
- `max_ai/base/clients copy.py` sigue en el repo; decidir si se borra.

## Para después

Dream dentro de la compaction, knowledge, cliente de OpenAI Responses, revisión de eventos, tareas de MCP.

## Cómo comprobar

- Tests: `pytest tests/agents/test_new_agent.py tests/agents/test_harness_rules.py tests/loop tests/compaction/test_new_compaction.py`
- La suite completa tiene un conjunto fijo de fallos viejos (142 fallos y 9 errores); comparar la lista antes y después, no el número.
- Antes de dar algo por hecho: pyright estricto y ruff limpios.
