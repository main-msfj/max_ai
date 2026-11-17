# Lo que queda por hacer

Rama `feat/workspace-new-approach`, último commit `f4b410d` (2026-11-17).

## Dónde estamos

- `max_ai/agents/agent.py` es el agente nuevo: guarda sus piezas, valida que cada una sea de su clase Core (`_validate`), aplica las reglas del harness (`_run_harness_validation`) y se guarda y reconstruye (`_to_config`, `_from_config`). 
- `max_ai/agents/agent_old.py` es el agente viejo; `from max_ai.agents import Agent` sigue dando el viejo.
- `max_ai/core/harness/rules.py` tiene la clase `Harness` con las reglas; los números viven en `max_ai/config.py` (`harness_*`, `summary_model`).
- Loop (`base/loop.py`), compaction (`base/compaction.py`) y conteo de tokens (`ToolRegistry.tokens`, `PromptBuilder.tokens`, `CoreLoop.measure`) están hechos y probados con tests.

## Hecho el 2026-11-17 (sin commit)

- `hooks.py`: `HarnessLoopHook` (índice de memoria tras compactar, guardar `ctx` tras cada tanda de tools, tareas de MCP como notas) y `HarnessToolHook` (memoria de solo lectura no se escribe; los middlewares ven cada tool).
- `agent.py` ya corre: `run`, `run_stream_events`, `close`, candado por sesión, MCP conectado una vez, sandbox por usuario con skills, prompt armado y revisado por `Harness.prompt_rules`, `loop.measure()` antes del modelo, compaction con resumen por defecto.
- `run` devuelve el `LoopState` (respuesta en `state.message`, motivo en `state.finish_reason`).

## Siguiente, en orden

1. **Prueba en vivo** del agente nuevo con OpenRouter (pedir permiso antes); solo se probó con un modelo falso.
2. **Prompt**: hoy es `instructions` + capas del developer + textos del harness en `Agent._prompt`. Falta decidir si entran las capas estándar (`STANDARD_ORDER`).
3. **`run_stream`** (solo texto) y un tipo de respuesta nuevo si `LoopState` no alcanza (`AgentResponse` es del viejo).
4. Middlewares: solo se llaman alrededor de las tools; faltan inicio y fin de corrida y alrededor del modelo, así que el trazado queda incompleto.
5. Cambiar `max_ai.agents.Agent` al nuevo y borrar lo viejo: `agent_old.py`, `compaction_old.py`, `clients_old.py`, `reasoning.py`, carpetas `to_delete/`, y los campos `workspace`, `executor`, `compaction`, `policy` de `AgentSpec`.

## Sin conectar todavía

- `knowledge` y `gates` se aceptan y se guardan; nada los usa.
- Bash puede escribir en `memory/` aunque sea de solo lectura; el hook solo frena `write_file` y `edit_file`.
- `close()` no espera a las corridas activas.

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
