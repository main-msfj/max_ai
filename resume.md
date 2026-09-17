# Resumen de continuidad — filesystem, Bash, ask_user, plan y eventos

## Actualización: API de Agent simplificada (16:06 UTC)

Por solicitud del usuario se retiró registry del constructor público de Agent.
Agent crea `_registry` internamente y recibe tools por `toolset`. Se conservaron
los registros host de filesystem, ask_user y AgentUpdatePlanTool, además de
deps['run_context']. No se cambió reasoning ni plan. La API de Agent no expone
por ahora un override host para tools personalizadas; no asumir que clientes MCP
vivos se pueden enviar a un proveedor remoto. Referencias a agent.registry en
secciones históricas deben leerse como anteriores a este cambio.

## Aviso para quien retome esto

Este archivo ha sido reescrito varias veces por sesiones distintas (el usuario
alterna de LLM por cuota). Las secciones de abajo están **verificadas contra
el código real al momento de escribir esta versión** (grep/lectura directa),
no repetidas de una versión anterior de este mismo archivo. Antes de actuar
sobre cualquier "pendiente" que menciones aquí, vuelve a comprobar contra el
código — el árbol se mueve rápido y este archivo puede volver a quedar
desactualizado en la próxima sesión.

## Instrucciones del usuario (vigentes)

- Trabajar en `/max_ai`, conversar en español y mantener la solución sencilla.
- NO agregar ni ejecutar tests por ahora. Solo py_compile, imports reales
  (incluido el peor orden posible para detectar ciclos) y `git diff`.
- El árbol tiene muchos cambios previos del usuario, incluso archivos sin
  seguimiento. Preservarlos; no hacer reset ni limpieza.
- Usar `.venv/bin/python3` para cualquier verificación — el `python3` del
  sistema no tiene `pydantic` instalado y falla de forma engañosa.
- Patrón de carpeta-por-tool ya aplicado a `bash/`, `file_system/`,
  `ask_user/` y `plan/`: `__init__.py` público (reexporta, a veces con
  `__getattr__` lazy) + módulos privados `_tool.py`/`_model.py`/
  `_permissions.py`/`_toolset.py`/`_capability.py`. Seguir el mismo patrón
  para cualquier tool nueva o reorganizada.

## Estado verificado del dispatcher/manager/Bash — YA RESUELTO, no repetir

- `base/agent.py:88` construye `ToolDispatcher(self.registry, source=name,
  manager=self._manager)`; `base/tool_dispatcher.py` ya acepta `manager` en
  su constructor. `Agent` se construye sin error.
- `ToolDispatcher._dispatch` ya llama `tool.permission_for(command)` para
  Bash: `deny` bloquea, `allow` fuerza `AUTO_APPROVED`, cualquier otra cosa
  deja `ASK_APPROVED`. Ya no es aprobación global por tool para Bash.
- `runtime/remote.py::_emit_events` ya reconstruye los 4 eventos de Bash
  además de los de filesystem.
- `tools/_recovery.py` **no existe** (se verificó con `ls`, no está). El
  usuario rechazó esa capa compartida de mensajes de recuperación; cada tool
  (ask_user, Bash, filesystem) devuelve sus propias indicaciones desde
  validación/ejecución en vez de pasar por un helper común.
- `tools/filesystem.py` y `tools/workspace.py` (con su tool `WorkspaceTool`)
  fueron eliminados, no solo dejados como shim. Los tests que los usaban
  fueron actualizados o retirados.

## ask_user — reestructurado en carpeta, con un bug real sin resolver

- Vive en `tools/ask_user/` (`_tool.py` con `AskUserTool`, `__init__.py`
  reexporta — mismo import path de siempre:
  `from max_ai.tools.ask_user import AskUserTool`).
- Schema actual de `options`: `list[{label: str, description: str}]`
  (cambiado desde el formato viejo `"Label — description"` en string). Se
  agregó `multiSelect: bool` al schema, documentado como no conectado
  todavía (toda respuesta sigue siendo un solo string).
- Estado de la tool en sí, verificado leyendo el archivo completo ahora
  mismo: **sin** `validate_parameters` propio, **sin** import de
  `_recovery`, **sin** `additionalProperties`/`pattern`/`const` extra — una
  versión anterior de esta sesión (visible en un diff intermedio) sí tenía
  esos campos y esa lógica; fueron revertidos después. No asumir que existen
  sin volver a leer el archivo.

- **BUG CONFIRMADO, verificado leyendo el código, no documentado hasta
  ahora**: hay dos caminos que interceptan `ask_user`, y solo uno adapta el
  formato nuevo de `options`:
  - `base/tool_dispatcher.py` (usado por `Agent`, sin cobertura de test):
    líneas ~108-112, sí convierte cada `{label, description}` a
    `f"{label} — {description}"` antes de guardar en el record.
  - `base/tool_executor.py` (usado por `ReActLoopSelfDirected`, **el camino
    con tests, el que realmente se ejercita hoy**): líneas 617-629, filtra
    `options` con `isinstance(o, str)` — como el modelo ahora manda dicts,
    **todos** fallan ese filtro, `options` queda en `None`, y la pregunta
    degrada silenciosamente a texto libre sin botones. No es un crash (el
    comentario del propio código dice "an empty/invalid list degrades to a
    free-text question", diseñado para otro caso), pero anula el propósito
    del cambio de schema en el camino que de verdad se usa.
  - Una vez ese filtro se arregla (aceptar dicts y convertir igual que
    `tool_dispatcher.py`), **no hace falta tocar nada más**: se verificó que
    `ToolCallRecord.input_options` (`list[str]`),
    `UserInputRequestEvent.options` (`list[str] | None`),
    `cli/renderer.py::show_question` y `ui/server.py` ya esperan strings y
    los reciben strings en cuanto el sanitizador los convierte — no son
    parte del problema, no cambiar su tipado.
  - `multiSelect` real (que `user_answer` acepte `str | list[str]`) sigue
    fuera de alcance a propósito — decisión explícita del usuario, es un
    cambio de estado más grande, paso aparte.

## plan — reestructurado en carpeta, con hallazgo de ciclo de imports

- Movido de dos archivos sueltos (`tools/plan.py`, `tools/update_plan.py`) a
  `tools/plan/`: `_model.py` (`AgentPlan`, `PlanStep`, Pydantic puro),
  `_tool.py` (`UpdatePlanTool`, depende de `base.reasoning`), `__init__.py`
  con patrón **lazy**: `AgentPlan`/`PlanStep` se exportan de inmediato;
  `UpdatePlanTool` se resuelve solo al pedirse (`__getattr__`), igual que
  `tools/__init__.py` ya hace con `BashTool`/`FileSystem`.
- **Por qué el lazy es obligatorio, no cosmético**: `core/event_type.py` y
  `types/run_context.py` importan `AgentPlan` directamente. `base/reasoning.py`
  importa `core.event_type`. Si el `__init__.py` del paquete cargara
  `UpdatePlanTool` de forma ansiosa, se cerraría el ciclo
  `core.event_type → tools.plan → base.reasoning → core.event_type`. Se
  verificó forzando el peor orden de import posible
  (`import max_ai.core.event_type` primero) y no rompe.
- Se actualizaron 3 sitios que importaban `tools.update_plan` (módulo que ya
  no existe): `reasoning/react_self_directed.py`,
  `tests/reasoning/test_self_directed.py`,
  `tests/reasoning/test_human_in_loop.py`. Sin shim de compatibilidad (pocos
  usos, todos internos).

## Decisión de arquitectura: dónde viven los eventos

Confirmado sin cambio de código: las **clases** de evento se quedan
centralizadas en `core/event_type.py`, no repartidas por carpeta de tool —
mismo riesgo de ciclo que en `plan`. Cada tool sigue siendo responsable de
**emitir** instancias vía `ToolContext.emit_event` (convención ya existente:
Bash 4 eventos, FileSystem 7, `ask_user` 1, `plan` 1) — eso no cambia.

## Lo que sigue pendiente, confirmado

- `max_ai/executor/docker/worker.py` — sin tocar, `git diff` vacío contra
  HEAD. Su fallback propio de Bash (`_WorkerBashTool`) no se revisó contra
  la migración de Bash a `Environment`.
- `max_ai/executor/routing.py` tiene un diff pequeño (2 líneas) sin revisar
  en detalle todavía.
- No dar la migración por terminada — sigue sin tests ejecutados end-to-end.
- Mantener separado: estado de ejecución, `action`/`description` declarados,
  evidencia real de archivos, decisión del `CompletionGate`, publicación
  explícita del workspace. Nada de esta conversación introdujo publicación
  automática ni infirió completitud desde texto de Bash.

## Siguiente paso autorizado (donde retomar)

Arreglar el bug confirmado de arriba, un solo archivo:

`base/tool_executor.py`, líneas ~617-629 — cambiar el sanitizador de
`options` para aceptar `{label, description}` (además de, o en vez de,
strings sueltos) y convertirlos a `f"{label} — {description}"` antes de
`record.await_user_input(...)`, igual que ya hace `tool_dispatcher.py`.
Después de ese cambio, no tocar `ToolCallRecord`, `UserInputRequestEvent`,
`cli/renderer.py` ni `ui/server.py` — ya están listos para recibir strings.

No tocar `multiSelect` real todavía — paso aparte, pendiente de decisión
explícita del usuario sobre cuándo abordarlo.
