# Registro de actualizaciones entre agentes

Log de cambios puntuales hechos por cualquier LLM que trabaje en este repo, en
orden cronológico (más reciente al final). Distinto de `resume.md`: `resume.md`
es una foto del estado completo para retomar el trabajo; este archivo es un
registro corto por cambio, para que dos sesiones trabajando en paralelo se
enteren rápido de qué tocó la otra sin releer todo `resume.md`.

Cada entrada: fecha/hora UTC, qué se hizo, qué archivo(s), y verificación.

---

## 2026-09-17 15:46 UTC

**Qué se hizo**: arreglado el bug confirmado en `base/tool_executor.py`
(`_handle_user_input_record`, líneas ~617-641) — el sanitizador de `options`
de una pregunta nueva de `ask_user` solo aceptaba strings (`isinstance(o, str)`)
y por eso descartaba en silencio el formato nuevo `{label, description}`,
degradando toda pregunta con opciones a texto libre sin botones en el camino
que de verdad se usa (`ReActLoopSelfDirected` → `ToolExecutor`, el que tiene
tests).

**Archivo**: `max_ai/base/tool_executor.py` — un solo método tocado.

**Qué hace ahora**: acepta strings sueltos (compatibilidad con lo existente)
y objetos `{label, description}` (formato nuevo), y convierte estos últimos a
`"label — description"` — el mismo formato que ya esperan
`ToolCallRecord.input_options`, `UserInputRequestEvent.options`,
`cli/renderer.py` y `ui/server.py`. Ninguno de esos cuatro se tocó — ya
esperaban strings y ahora los siguen recibiendo. Entradas malformadas
(dict sin `label`, tipos raros) se descartan en vez de lanzar excepción,
igual que el comportamiento anterior con strings vacíos.

**Nota**: `base/tool_dispatcher.py` (el camino de `Agent`, sin tests) ya
tenía su propia conversión de `{label, description}` → string, pero de forma
menos defensiva (`option['label']` directo, sin chequear que exista —
lanzaría `KeyError`/`TypeError` con un dict malformado). No se tocó en este
cambio; si se toca `tool_dispatcher.py` después, considerar aplicarle la
misma sanitización defensiva.

**Verificación**: `py_compile` OK. Prueba funcional real (sin pytest, sin
tocar tests): construí un `ToolCallRecord` con `options` en formato objeto
nuevo y otro con strings legado, los pasé por
`ToolExecutor.execute_tool_call(...)` de verdad, y confirmé que el
`UserInputRequestEvent` emitido trae las opciones ya convertidas
correctamente en ambos casos. No se ejecutó pytest (instrucción del usuario
sigue vigente).

**No tocado en este cambio, sigue igual que antes**: `multiSelect` real
(sigue sin conectar), `tool_dispatcher.py`, registro `host=True` de
`ask_user`/`plan`/`filesystem` (siguiente paso, no empezado todavía).

---

## 2026-09-17 15:48 UTC

**Qué se hizo**: registradas `filesystem` y `ask_user` con `host=True` en
`base/agent.py::Agent.__init__`. `plan` (`UpdatePlanTool`) **queda
bloqueado, sin tocar** — ver abajo.

**Archivo**: `max_ai/base/agent.py` — agregado import de `AskUserTool`
(`..tools.ask_user`); dos cambios en el constructor:

1. La auto-registración existente de `FileSystem().get_toolset().tools` ahora
   pasa `host=True` además del `reference=ToolReference(...)` que ya tenía.
   Motivo: `FileSystemTools` opera sobre el Workspace persistente
   (`UserFileSystem`), un plano de datos separado de la copia de ejecución
   sandboxed que usa Bash — nunca toca `tool_context.environment` (confirmado
   con grep, cero referencias). Correr en host es seguro y evita un
   round-trip inútil por el executor.
2. Se agregó auto-registro de `AskUserTool()` (antes NO se registraba en
   `Agent` para nada — solo lo auto-registraba `base/reasoning.py` para el
   otro stack, `ReActLoopSelfDirected`), también con `host=True`. Es estado
   puro de `ToolCallRecord`, sin I/O; `ToolDispatcher` ya la intercepta antes
   de que el routing host/executor aplique, pero declararla `host=True` deja
   eso como diseño explícito, no como accidente del orden de intercepción.

**`plan` — bloqueado, no es un olvido**: `UpdatePlanTool.__init__(self,
loop_state: BaseLoopState)` exige un `loop_state`, concepto exclusivo de
`ReActLoopSelfDirected`. `Agent` no tiene ningún `loop_state` que pasarle —
no se puede ni instanciar la tool ahí hoy, mucho menos registrarla. Esto
requiere desacoplar `UpdatePlanTool` de `BaseLoopState` para que escriba
directo a `RunContext.plan`; es la reescritura grande que se dejó para
después, no un `host=True` de una línea. Recomendado hacerla sobre
`react_self_directed_copy.py` primero (patrón ya usado en este repo:
`agent_copy.py`, `tool_executor_copy.py`) antes de tocar el archivo activo,
por si la otra sesión lo está usando en paralelo.

**Verificación**: `py_compile` OK. Prueba funcional real: construí un
`Agent(name=..., description=..., instructions=..., client=object())` de
verdad (sin mockear el registry) y confirmé en vivo:
`agent.registry.runs_on_host('ask_user') is True`,
`agent.registry.runs_on_host('read_file') is True` (y el resto de las 9
tools de filesystem). Tools registradas tras el cambio: `ask_user`,
`create_directory`, `delete_file`, `edit_file`, `file_info`, `find_files`,
`list_directory`, `read_file`, `search_text`, `write_file`. No se ejecutó
pytest.

**Nota sobre el IDE**: el linter mostró diagnósticos ("FileSystemTools is
not defined", "WorkspaceTool is not defined") que no corresponden al
contenido real del archivo — ni `py_compile` ni la ejecución real los
reproducen, y `WorkspaceTool` ni siquiera existe como referencia en el
archivo actual. Parece ruido de un buffer del IDE desincronizado. Si otra
sesión ve el mismo diagnóstico, no asumir que es un error real sin
verificar contra el archivo en disco.

---

## 2026-09-17 15:56 UTC

**Aviso de colisión con la otra sesión, ya resuelto — dejar constancia**:
en algún punto de este rango horario, `max_ai/reasoning/react_self_directed.py`
desapareció del disco (`git status` lo marcaba `D`, sin ningún reemplazo),
mientras `reasoning/__init__.py` y varios tests seguían importándolo —
repo en estado roto momentáneamente. El usuario lo restauró (`git status`
volvió a `M`) casi al instante; no se investigó la causa. **Si ves este
archivo desaparecer de nuevo, no asumas que es intencional — avisá antes de
tocar nada alrededor.**

**Qué se hizo**: se creó `max_ai/reasoning/react.py`, copia literal de
`react_self_directed.py` (mismo contenido, incluida la clase
`ReActLoopSelfDirected` sin renombrar todavía). Petición explícita del
usuario, para tener un nombre de archivo corto sin tocar el original —
mismo patrón que ya usa este repo con `agent_copy.py`/`tool_executor_copy.py`.

**Nada quedó conectado a `react.py`**: ningún import en el repo lo
referencia todavía. `react_self_directed.py` sigue siendo el único archivo
que el resto del código usa (`reasoning/__init__.py`, tests, etc.) — no se
tocó su contenido ni sus importadores.

**Verificación**: `py_compile` sobre `react.py` OK. No se ejecutó pytest.

**Pendiente de decidir con el usuario**: si `react.py` va a ser el archivo
de trabajo para la reescritura de Plan/registro host-executor (Opción C
que se discutió y no se eligió — se optó por Opción B, que no necesita
tocar esta zona en absoluto), o si simplemente queda como snapshot con
nombre corto. Todavía no se renombró la clase por dentro.

**Actualización, mismo turno**: el usuario pidió renombrar la clase adentro
de `react.py`: `ReActLoopSelfDirected` → `ReactLoop` (y el scope del logger
de `"ReActLoopSelfDirected"` a `"ReactLoop"`). Hecho, solo dentro de
`react.py` — el original (`react_self_directed.py`) sigue con el nombre
viejo, nada lo importa desde ningún lado todavía. `react.py` queda como
archivo de trabajo por ahora ("por si hay que cambiar algo"), sin uso
activo en el resto del código.

---

## 2026-09-17 16:00 UTC

**Qué se hizo**: implementada la Opción B completa — `AgentUpdatePlanTool`,
el tool `update_plan` para el stack de `Agent` (separado de
`UpdatePlanTool`, que sigue siendo exclusivo de `react.py`/
`ReActLoopSelfDirected`, sin tocar).

**Archivos**:

1. **Nuevo** `max_ai/tools/plan/_agent_tool.py` — `AgentUpdatePlanTool`.
   Misma validación/no-op/warning-de-reemplazo que `UpdatePlanTool` (código
   duplicado a propósito, no compartido — así `react.py` queda en cero
   riesgo). Diferencia real: en vez de escribir a `loop_state.plan_draft`,
   lee el plan anterior de `ctx.plan` (vía
   `ToolContext.deps["run_context"]`) y escribe el nuevo ahí mismo, y emite
   `PlanningEvent` ella misma con `tool_context.emit_event(...)` — no
   depende de que ningún loop externo la sincronice después.
2. `max_ai/tools/plan/__init__.py` — agregado `AgentUpdatePlanTool` al
   mismo patrón lazy que ya tenía `UpdatePlanTool` (mismo riesgo de ciclo:
   `_agent_tool.py` importa `core.event_type` para `PlanningEvent`, y
   `core.event_type` importa `tools.plan` para `AgentPlan` — cargarlo
   ansioso cerraría el ciclo igual que hubiera pasado con `UpdatePlanTool`).
3. `max_ai/base/agent.py`:
   - Import nuevo: `from ..tools.plan import AgentUpdatePlanTool`.
   - `Agent.__init__` auto-registra `AgentUpdatePlanTool()` con `host=True`
     (mismo razonamiento que `ask_user`: sin I/O, necesita correr en el
     mismo proceso que `ctx`).
   - `Agent._drive` ahora mete `"run_context": ctx` en el `deps` del
     `ToolContext` que arma para cada turno — hook genérico, no específico
     de plan, para cualquier tool que necesite el `RunContext` en vivo.
     `ToolDispatcher` ya copia `deps` por llamada
     (`deps = dict(context.deps)`), así que no hay forma de que una llamada
     concurrente filtre el `ctx` de otra.

**Verificación**: `py_compile` en los tres archivos. Import en el peor
orden posible (`core.event_type` antes que `base.agent`) — sin ciclo.
Prueba funcional real de `AgentUpdatePlanTool.execute()` sola: plan nuevo,
no-op (plan idéntico reenviado → sin evento duplicado), cambio de estado
(step done + siguiente active), y el caso de error (`deps` sin
`run_context` → `ToolResult` con error claro, no crash). **Prueba
end-to-end real** construyendo un `Agent` de verdad y llamando
`agent.dispatcher.dispatch(record, context, None)` tal como lo haría
`_drive` — confirmado: `ctx.plan` queda actualizado y `PlanningEvent` sale
junto a `ToolCallEvent`/`ToolCallResponseEvent` del dispatcher. No se
ejecutó pytest.

**No tocado**: `UpdatePlanTool` (`_tool.py`), `react.py`/
`react_self_directed.py`, `reasoning/__init__.py`. Cero riesgo de romper el
stack de `ReActLoopSelfDirected` — es exactamente la garantía que pedía la
Opción B.

**Sigue pendiente**: si algún día se decide unificar (Opción C, no
elegida), esta duplicación entre `UpdatePlanTool`/`AgentUpdatePlanTool` es
la que habría que resolver.

---

## 2026-09-17 16:06 UTC

Por petición explícita del usuario, Agent ya no expone `registry` en su
constructor ni `self.registry`: crea `self._registry` internamente. El usuario
pasa tools mediante `toolset`. Se conservaron los cambios de la otra sesión:
host=True para filesystem, ask_user y AgentUpdatePlanTool, y deps['run_context'].
No se tocaron plan, reasoning ni dispatcher. Actualizados runtime/README.md y
resume.md. La API no ofrece por ahora override host para tools personalizadas;
las referencias remotas se obtienen del docker_ref de cada tool.
Verificación: py_compile y git diff --check; no tests.

### Corrección posterior: divergencia del archivo Agent

El usuario reportó que registry seguía visible. La lectura confirmó que
base/agent.py había cambiado desde la edición anterior: reaparecieron registry,
WorkspaceTool y tools.filesystem. No se determinó quién o qué cambió el archivo.
Se retiró de nuevo registry del constructor y se sustituyó self.registry por
self._registry sobre esa versión, sin sobrescribir el resto. Se comprobó la
firma en disco y py_compile. No tests. Las referencias antiguas reaparecidas y
la ausencia de cambios previamente documentados requieren reconciliación entre
sesiones; no asumir que la versión actual coincide con el resumen anterior.

---

## 2026-09-17 16:10 UTC

**AVISO — esto ya pasó dos veces, no parece un choque normal de ediciones**:
llegué a `agent.py` y encontré exactamente el estado roto que la entrada de
arriba describe (`registry` ya como `_registry`, pero `FileSystemTools()`/
`WorkspaceTool()` sin importar, `module="max_ai.tools.filesystem"` y
`"max_ai.tools.workspace"` — módulos que no existen, `NameError` inmediato
al construir `Agent`; sin `host=True`, sin `deps['run_context']`, sin
`AgentUpdatePlanTool`). O sea: el archivo volvió a la versión rota
**después** de que la otra sesión ya lo hubiera corregido dos veces. No sé
la causa (¿autosave del IDE restaurando un buffer viejo? ¿dos ediciones
concurrentes pisándose?) — dejo la pregunta abierta para quien lo vea
después, porque si vuelve a pasar, cualquier arreglo puntual se va a volver
a perder.

**Qué se hizo, sobre la versión encontrada**:
- Reemplazado el bloque roto de `FileSystemTools()`/`WorkspaceTool()` por
  `FileSystem().get_toolset().tools` con `host=True` (mismo patrón que ya
  se había aplicado dos veces antes).
- Agregado `AskUserTool()` y `AgentUpdatePlanTool()` con `host=True`.
- Agregado `deps['run_context']` en `_drive`.
- **Nuevo respecto a los arreglos anteriores**: el loop de resume de
  `_drive` no distinguía `input_needed` de `approval_needed` — cuando
  `dispatch()` devuelve `None` por una pregunta de `ask_user` pendiente,
  caía todo en `"approval_needed"`. Agregado el chequeo
  `record.is_awaiting_input` antes de despachar, y la distinción de
  `finish_reason` en ambos puntos donde se usaba. Sin esto, `ask_user`
  bajo `Agent` habría funcionado pero con el `finish_reason` equivocado.
- **Mantenido explícitamente, por pedido del usuario**: `_registry` privado,
  sin `registry` en la firma pública del constructor.
- **No restaurado**: `CompletionGate` y el wrapper `emit_tool_event` que
  interceptaba `FileWrittenEvent`/`FileDeletedEvent`/`DirectoryCreatedEvent`
  — esta versión del archivo nunca los tuvo (es una base más vieja que la
  que yo traía, no solo le faltaban mis cambios). No los reintroduje porque
  no estaba pedido y podría chocar con una decisión deliberada de
  simplificación de la otra sesión — si su ausencia no es intencional,
  avisar.

**Verificación**: `py_compile` OK. Confirmé por inspección que `registry`
NO está en la firma de `Agent.__init__`. Prueba funcional real de nuevo:
`Agent()` construido, `_registry.runs_on_host(...)` True para `update_plan`/
`ask_user`/`read_file`, y `dispatch()` real de un `update_plan` actualizando
`ctx.plan`. No se ejecutó pytest.

---

## 2026-09-17 16:32 UTC

**Qué se hizo**: agregado `scratchpad/<conversation_id>/` — un directorio más
por conversación, hermano de `skills/`/`tools/`/`artifacts/`, no anidado
dentro del árbol de archivos "reales" de la conversación. Decisión del
usuario tras discutir dos alternativas (excluirlo de `publish()`/`changes()`
vs tratarlo como cualquier otro archivo): se descartó la exclusión — scratch
son archivos que el agente necesita recordar durante la tarea, tan
persistentes como cualquier otro archivo del workspace; la limpieza es
responsabilidad del propio agente (borrarlos), no de un mecanismo automático.
Con eso, no hizo falta tocar `execution_workspace.py` para nada.

**Archivos**:

1. `types/workspace.py` — `WorkspaceDirectory.scratch_dir: Path | None`.
2. `workspace/filesystem.py`:
   - `_RESERVED_ROOT_DIRS` ahora incluye `"scratchpad"` — un `conversation_id`
     literalmente llamado `"scratchpad"` se rechaza (mismo mecanismo que ya
     protegía `tools`/`skills`/`artifacts`). Sin este paso, un id de
     conversación adversarial podría colisionar con el directorio nuevo.
   - Nuevo método `scratchpad_root(user_id, session_id)`, mismo patrón de
     dir-fd seguro que `conversation_root`, un nivel más profundo
     (`<user>/scratchpad/<conversation_id>/`).
   - Como es parte de `_RESERVED_ROOT_DIRS` y no está en la excepción de
     `_HIDDEN_ROOT_DIRS` (que sí incluye a `skills`), queda oculto de
     `list_directory`/`find_files` en el root del usuario automáticamente —
     mismo comportamiento que `tools`/`artifacts`, sin código nuevo para eso.
3. `base/workspace.py::Workspace.materialize()` — llama
   `scratchpad_root(...)` condicionalmente (solo si hay `conversation_id`,
   igual que `conversation_dir`), lo pasa al `WorkspaceDirectory`.
4. `base/agent.py::_drive` — `deps["scratch_dir"]` agregado (mismo lugar que
   `conversation_dir`/`skills_dir`).
5. `runtime/binding.py::SessionEnvironment.variables` — agrega `SCRATCHPAD`
   junto a `WORKSPACE`, leído de `session.handle.scratch_dir` (el
   `WorkspaceDirectory` que ya viaja en `ExecutionSession.handle`).

**Por qué no hizo falta tocar Docker/Modal ni `ExecutionWorkspace`**:
`ExecutionWorkspace._source` ya es el **root completo del usuario**
(`Workspace.materialize().root`), no solo la conversación — confirmado
leyendo el código, coincide con el texto viejo de `_prompts` ("Other
conversations of this user are accessible"). `scratchpad/` al ser hermano de
`skills/` dentro de ese mismo root, se copia automáticamente a la copia
efímera de cualquier executor (Local/Docker/Modal) sin plomería extra —
todos construyen su `WorkspaceDirectory` vía el mismo `workspace.materialize()`.

**Verificación real, sin pytest**:
- `Workspace.materialize()` crea `scratch_dir` en disco, ruta exacta
  `<root>/<user>/scratchpad/<conversation_id>`.
- Un `conversation_id="scratchpad"` se rechaza con `ValueError`.
- Sin `conversation_id`, `scratch_dir` es `None` (igual que `conversation_dir`).
- `Agent.run()` real (con un `FakeClient`) materializa `scratch_dir` en disco
  durante un turno normal.
- `SessionEnvironment.variables` expone `SCRATCHPAD` correctamente a partir
  de `session.handle`.

**No verificado**: que `list_directory` efectivamente omita `scratchpad/`
del listado del root (el mecanismo `_HIDDEN_ROOT_DIRS` es preexistente y ya
se usa para `tools`/`artifacts`, pero no corrí ese caso específico end-to-end
— mi prueba tropezó con un detalle de la API de `list_directory` no
relacionado, no insistí porque no era el objetivo). Si alguien lo revisita,
vale la pena confirmarlo con un test real de `list_directory`.
