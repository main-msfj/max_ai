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

---

## 2026-09-18 01:06 UTC

**Qué se hizo**: conectado `scratchpad/` a `FileSystemTools` — hasta este
punto los archivos existían en disco pero ninguna tool del modelo (aparte de
Bash vía `$SCRATCHPAD`) podía llegar ahí.

**El problema encontrado**: `scratchpad` se había marcado ayer como
"oculto" (`_HIDDEN_ROOT_DIRS`), pero ese mismo set también se usaba para
**bloquear la resolución de rutas** (`_visible_parts` lanzaba
`ValueError` para cualquier intento de `scratchpad/...`, igual que ya hacía
con `tools`/`artifacts`). Hacía falta separar "oculto de listar" de
"bloqueado de resolver" — antes eran el mismo mecanismo.

**Archivos**:

1. `workspace/filesystem.py`:
   - Nuevo `_BLOCKED_PATH_ROOTS = _RESERVED_ROOT_DIRS - {"skills", "scratchpad"}`
     (antes `_visible_parts` usaba `_HIDDEN_ROOT_DIRS` para bloquear; ahora
     usa este set más chico — `tools`/`artifacts` siguen bloqueados,
     `scratchpad` ya no).
   - `create_text_file`/`create_directory`: si `parts[0] == "scratchpad"`,
     no vuelven a prefijar con `session_id` (el path ya viene completo).
2. `tools/file_system/_toolset.py`:
   - `_conversation_path` — rama nueva para `scratchpad`, mismo patrón que
     ya existía para `skills` (cubre read_file/edit_file/delete_file/
     file_info/list_directory gratis, porque esos ya pasan el path resuelto
     completo al backend).
   - Nuevo `_write_target(context, path)` — solo para `write_file`/
     `create_directory`, que usan una convención de firma distinta
     (`session_id` separado + path relativo, con prefijo automático en el
     backend) y por eso no bastaba con arreglar `_conversation_path`.
3. `base/agent.py::_prompts` — le explica al modelo la convención
   `scratchpad/notas.txt`, que no se limpia solo, y que Bash ve los mismos
   archivos vía `$SCRATCHPAD`.

**Verificación real, sin pytest** (todo con `FileSystemTools` real, no
mocks): `write_file`/`read_file`/`create_directory` contra
`scratchpad/notes.txt`/`scratchpad/subdir` funcionan; escritura normal
(`real.txt`) sin cambios; `list_directory` en la raíz del usuario **no**
muestra `scratchpad` pero navegarlo explícito sí lista su contenido;
confirmado por lectura directa de disco que el archivo que escribió
`write_file` está exactamente en el path que usaría `$SCRATCHPAD` para
Bash — mismo `Workspace`, mismos archivos, dos puntos de entrada. Repetí
el chequeo de `tools/`/`artifacts` bloqueados para confirmar que no se
aflojó esa protección al separar los dos sets.

**Con esto, el diseño de scratchpad que se discutió ayer y hoy queda
completamente implementado** — ubicación, persistencia, sincronización
entre executors, y ahora también accesible desde ambos tipos de tool
(host y sandbox). Ver `resume.md` para el resumen de la decisión completa.

---

## 2026-09-18 (sesión larga, reorganización Día 2 completa)

**Contexto**: continuación de la reorganización `core/`/`base/`/`capabilities/`/
`legacy/` iniciada antes del checkpoint `fa4aa0b`. Resumen agrupado por tema,
no cronológico estricto — la sesión fue larga y con mucha discusión de diseño
intercalada.

### 1. Bug de sed corregido (import corruption)

Un `sed` de la sesión anterior (`s/^(\s*from \.+)tools\b/\1capabilities.tools/`)
sobre-escribió imports de un solo punto (`from .tools import CoreTool` →
`base/tools.py`, nada que ver con el paquete movido) convirtiéndolos en
`from .capabilities.tools import CoreTool` (ruta inexistente). Corregido en
`base/{routines,tool_dispatcher,agent,tool_registry,knowledge,memory,context}.py`
y `base/memory copy.py`. De paso, `capabilities/tools/__init__.py` tenía su
`_LAZY_EXPORTS` apuntando a las rutas viejas `max_ai.tools.*` (strings, el sed
no las tocó) — corregido a `max_ai.capabilities.tools.*`.

### 2. `Executor` → `ExecutorBase`, contrato completo restaurado

`base/executor.py` es ahora el único ABC de ejecución (`connect`/`sync`/
`disconnect`/`clean`/`rebuild`/`run_tool`/`execute`/`execute_argv`), renombrado
`Executor` → `ExecutorBase` (regla: "Base" para todo lo abstracto/pluggable).
`LocalExecutor` (`capabilities/executor/local/_executor.py`) reexpandido para
implementar el contrato completo (session tracking `_sessions`/`_closed`/
`_check`). Referencias corregidas en 8 archivos (`agent.py`,
`capabilities/executor/{__init__,remote,local/_executor}.py`,
`core/environment/manager.py`, `legacy/environment/{binding,session_manager}.py`).

### 3. `EnvironmentManager` movido a `core/environment/manager.py`, sin copia de trabajo

La versión completa que ya existía en `max_ai/environment/session_manager.py`
se migró a `core/environment/manager.py` (ubicación canónica). Decisión de
diseño importante: **se eliminó `ExecutionWorkspace`** (copia de trabajo
intermedia) del flujo de `acquire()` — Local y Docker ya comparten filesystem
(bind mount), copiar ahí no protege nada; solo un backend remoto de verdad
(Modal) necesitaría staging, y eso le toca a su propio `sync()`, no al manager
genérico. `execution_workspace.py` → `legacy/execution_workspace.py`.
`base/environment.py` (el `Environment` ABC viejo, con `SessionEnvironment`
en `capabilities/executor/binding.py`) también se confirmó sin uso real
(nada leía `context.environment`) y se movió completo a `legacy/environment/`
junto con `manager.py`/`docker.py`/`session_manager.py` (generación vieja,
factory-based). `ToolContext` (`base/tools.py`) perdió el campo `environment`.

### 4. `core/tool/` nuevo — `ToolRegistry` + `ToolDispatcher` salen de `base/`

Por no tener `abstractmethod` (fijos, no pluggable): `base/tool_registry.py` →
`core/tool/registry.py`, `base/tool_dispatcher.py` → `core/tool/dispatcher.py`.
Referencias corregidas en `base/agent.py`, `tests/v2/test_tools_and_bash.py`.

### 5. Colisión de `base/` resuelta

Una sesión concurrente dejó `max_ai/base/` casi vacío (solo `__init__.py`) y
copió el contenido real a `max_ai/base copy/` (backup deliberado del usuario,
no limpieza). Los 26 archivos reales se restauraron en `base/` (más los 4
`_copy`: `agent_copy.py`, `memory copy.py`, `skills copy.py`,
`tool_executor_copy.py`, recuperados de git `fa4aa0b` sin pérdida). Luego se
recreó `base copy/` como snapshot del `base/` restaurado, para no perder el
backup del usuario. **Aprendizaje**: no asumir que una carpeta `X copy/` es
limpieza a borrar — puede ser backup intencional; preguntar antes de tocar.

### 6. `$SCRATCHPAD` cableado hasta Bash

`RuntimeDirs` (`types/tools.py`) ganó el campo `scratch`. `BashTool`
(`capabilities/tools/bash/_tool.py`) ahora expone **tanto** `$WORKSPACE` como
`$SCRATCHPAD` al subprocess — `$WORKSPACE` tampoco se seteaba antes pese a que
el prompt ya lo prometía. `AgentPolicyLayer.j2` actualizado para que el modelo
sepa que existe `$SCRATCHPAD`.

### 7. `Loop` separado de `Agent`, generación vieja limpiada

Por decisión del usuario, el loop de razonamiento no vive dentro de
`Agent._drive()` — es un componente pluggable separado (mismo patrón que
`ExecutorBase`/`WorkspaceBase`), para que otra estrategia de loop se pueda
escribir sin tocar `Agent`. Como paso hacia eso:

- `base/reasoning.py` (`BaseLoopState`/`BaseReasoning`): quitados
  `plan_draft`/`plan_updated` (movidos a `ReActLoopState`, específicos de esa
  estrategia) y el import de `AgentPlan`. `guard_state` se queda por ahora
  (explícitamente diferido, no analizado todavía).
- `reasoning/react.py` → `reasoning/react/loop.py` (carpeta nueva). Los tres
  métodos que accedían a `plan_draft`/`plan_updated` se retiparon
  `ReActLoopState` en vez de `BaseLoopState`.
- **`plan_draft`/`plan_updated` eliminados por completo** (no solo movidos):
  ese staging existía porque la `UpdatePlanTool` vieja no tenía acceso al
  `RunContext`, solo a `ToolContext`. La `AgentUpdatePlanTool` nueva (la que ya
  usa `Agent`) escribe `ctx.plan` directo vía `deps["run_context"]` — sin
  staging. `_sync_plan()` se eliminó; "¿se tocó el plan este round?" ahora se
  deriva de `assistant_msg.tool_calls`/`pending` en vez de un flag de estado.
  `_register_runtime_tools` registra `AgentUpdatePlanTool` en vez de
  `UpdatePlanTool(loop_state=...)`.
- Identificado (no resuelto): el nudge de plan (`_PLAN_NUDGE`, hardcodeado en
  el cuerpo del loop) debería ser un `LoopGuard` más, igual que
  `RepetitionGuard`/`BudgetGuard` — no lógica ad-hoc en `execute_reasoning_loop`.
  Pendiente para la próxima sesión de diseño del Loop nuevo.
- Middleware/hooks: `MiddlewareChain` (`middleware/chain.py`) es genérica
  (parametrizada por `action: str`), así que un solo `MiddlewareChain`
  compartido puede cubrir tanto `action="model_call"` (ya lo hace el Loop) como
  `action="tool_call"` (falta agregarlo a `ToolDispatcher`, que hoy no toca
  middleware para nada). Decidido conceptualmente, no implementado todavía.

### 8. `agents/agent.py` — copia paralela para pruebas

`max_ai/agents/agent.py` creado como copia de `base/agent.py` (imports
ajustados a la nueva profundidad de carpeta), para empezar a probar sin tocar
el archivo activo. `base/agent.py` queda intacto.

**Pendiente confirmado y NO tocado en esta sesión** (decisión explícita del
usuario, limpieza manual): `base/tool_executor.py` (`ToolExecutor`, generación
vieja) tiene el import roto `from .executor_legacy import CoreExecutor`
(`executor_legacy.py` fue borrado manualmente) y también construye
`ToolContext(..., environment=self.environment)`, kwarg que ya no existe tras
el punto 3. Este archivo, `reasoning/react_self_directed.py`,
`reasoning/guards.py` y el paquete `max_ai/executor/` (viejo, distinto de
`capabilities/executor/`) forman el stack completo de la generación anterior —
su destino final (legacy/, reescritura, o borrado) sigue sin decidir.

**Verificación**: cada paso de esta sesión se verificó con `py_compile` +
pruebas funcionales reales (sin pytest, sin mocks) construyendo los objetos
reales — `LocalExecutor`+`EnvironmentManager` conectando/ejecutando/
sincronizando de punta a punta, `BashTool` real confirmando `$WORKSPACE`/
`$SCRATCHPAD` en el subprocess, `Agent`/`max_ai.agents.agent.Agent`
importando completos. Ningún test nuevo agregado ni ejecutado.

---

## 2026-09-18 — Agent conectado al ReactLoop y componentes locales

Por petición del usuario, se trabajó en `max_ai/agents/agent.py`, sin editar
`base/agent.py` ni `base copy/agent_copy.py`.

- `Agent` acepta `reasoning` y `environment`, además de `skills`, `workspace`
  y `executor`. Defaults: ReactLoop, LocalWorkspace, LocalExecutor,
  LocalSkillRegistry vacío y EnvironmentManager creado por Agent.
- `environment` significa el EnvironmentManager actual, NO el Environment
  legacy. Si se proporciona, se usan su workspace/executor; configuraciones
  explícitas contradictorias se rechazan. Agent solo cierra el manager propio.
  LocalExecutor no ofrece aislamiento de sandbox.
- `_drive` prepara workspace/skills/ToolContext y delega a
  `reasoning.bind(...).execute_reasoning_loop(...)`. Eliminado el ciclo LLM
  duplicado y su acumulador de streaming de este Agent.
- Bash registrado por defecto (permite override por nombre en toolset).
  Filesystem, ask_user y plan siguen host; registry permanece interno.
  Metadata y rutas SKILL.md se incluyen en el prompt tras prepare().
- BaseReasoning admite dispatcher + ToolContext en bind; mantiene el argumento
  ToolExecutor anterior para no romper sus otros consumidores. El catálogo y
  `_execute_tools` seleccionan el backend conectado, sin importar legacy en runtime.
- ToolDispatcher.dispatch_many transmite eventos en vivo y ToolMessages en orden.
  ReactLoop usa ese recorrido; ya no importa USER_INPUT_TOOL_NAMES desde el
  ToolExecutor roto, ni duplica registros/eventos de plan en el stack nuevo.
- ReactLoop drena pendientes antes de comprobar el límite de iteraciones,
  incluyendo preguntas y aprobaciones aún sin resolver. Agent conserva métricas,
  duración y guard_state al pausar. Se agregó guard_state a ReActLoopState porque
  los guards actuales lo requieren. APIs resume, resume_stream y
  resume_stream_events disponibles.
- CoreSkillRegistry ahora permite una lista vacía para el default local sin
  paquetes seleccionados. No hay descubrimiento automático de skills.
- `from max_ai.agents import Agent` exporta este Agent nuevo. No se cambió la
  exportación principal del framework ni se migraron otros consumidores.

Verificación: py_compile de archivos modificados, importación de Agent,
ReactLoop y EnvironmentManager usando `.venv/bin/python`, git diff --check OK.
No tests ni ejecución end-to-end. El Python global no tiene pydantic; usar .venv.
Pendiente: middleware tool_call, compaction/store y demás capacidades de la
copia antigua no fueron incorporadas en este cambio. No compartir una instancia
de reasoning entre agentes concurrentes: bind almacena dependencias mutables.

## 2026-09-18 — Ejemplo OpenAI local, ejecución bloqueada por credenciales

Nuevo `examples/agent_local_openai.py`, ejecutable con
`.venv/bin/python -m examples.agent_local_openai`. Usa `max_ai.agents.Agent`,
defaults locales y `gpt-5.6-luna`, reasoning_effort="low" (menor esfuerzo
con razonamiento activo según documentación oficial; "min" no figura).
Lee OPENAI_KEY / OPENAI_API_KEY / OPENA_KEY sin imprimir secretos.
Tarea inicial: ejecutar únicamente pwd con Bash y responder en español.

Se ejecutó realmente, sin pytest. Tras autorizar escritura al workspace .agents
(protegido en el entorno de Codex), alcanzó OpenAI pero devolvió HTTP 401,
`not_authorized_invalid_project`: el proyecto solicitado está archivado.
No hubo respuesta del modelo ni ejecución de Bash. Hace falta configurar una
clave/proyecto activo antes de continuar la comprobación end-to-end.

---

## 2026-09-18 — Verificación end-to-end + bugs de workspace encontrados probando con modelo real

Retomando el wiring `Agent`/`ReactLoop`/`ToolDispatcher` que dejó la sesión
anterior (entradas de arriba): se verificó funcionando de punta a punta con
`FakeClient` (tool call → approval_needed → aprobar → respuesta final) y
luego el usuario lo probó de verdad con `gpt-5.6-luna` vía
`examples/agent_local_openai.py`. De ahí salieron varios bugs reales.

**`guard_state` silenciosamente roto**: los 4 guards en `reasoning/guards.py`
(`RepetitionGuard`, `BudgetGuard`, `NoProgressGuard`, nudge de plan) leen
`state.guard_state`, que había quedado comentado en `BaseLoopState`
(`base/reasoning.py`) por decisión explícita del usuario en una sesión
anterior ("no lo necesitamos ahorita"). `_guard_steering` atrapa el
`AttributeError` y sigue de largo — de ahí el "Loop guard failed; skipping"
en cada iteración. Confirmado, no arreglado — decisión pendiente del usuario
(reactivarlo o no).

**Directorios sueltos que no debían existir** (`.agents/user_001/tools`,
`.agents/user_001/artifacts`): `RuntimeDirs`/`BashTool._runtime_dirs()`
(`capabilities/tools/bash/_tool.py`) creaba `tools/` y `artifacts/` como
fallback cuando `deps` no traía esas claves — nadie las traía nunca. `tools`
resultó ser un concepto exclusivo del paquete viejo `max_ai/executor/docker/`
(no compartido con `bash/_tool.py`); `artifacts` era redundante con la
carpeta de conversación (`WorkspaceDirectory.artifacts_dir` ya colapsa a
`conversation_dir` cuando hay conversación). Se quitaron ambos campos de
`RuntimeDirs` (`types/tools.py`) y de `_runtime_dirs()`/`execute()` por
completo — ya no se crean.

**Conversaciones sueltas en la raíz del usuario, no anidadas**: el usuario
pidió `conversation/<id>/` en vez de `<user_root>/<id>/` directo. Cambio
coordinado (si se cambia solo un lado, `write_file` y `materialize()`
divergen y el traversal-check se rompe):
- `workspace_copy/filesystem.py`: `_RESERVED_ROOT_DIRS` ganó `"conversation"`
  (reservado pero **visible** — a diferencia de `tools`/`artifacts` — para
  que `find_files`/`search_text` sigan atravesando todas las conversaciones;
  ocultarlo habría roto "buscar en todas mis conversaciones" en silencio).
  `conversation_root()` anida bajo `conversation/<id>/` (mismo patrón dir-fd
  que `scratchpad_root()`). `create_text_file`/`create_directory` prependen
  `("conversation", session, ...)`. `_conversation_file_parts` exige y valida
  ese prefijo de 3 segmentos.
- `capabilities/tools/file_system/_toolset.py::_conversation_path()`:
  construye el mismo prefijo `"conversation/<id>/..."`.
- Verificado con las tools reales: write/read/edit/delete/list/find sobre la
  estructura nueva, traversal (`..`) y bloqueo de `tools/` intactos.

**Bash no coincidía con dónde escriben los file tools** — el bug más
importante, encontrado porque el modelo bajó un `.html` y aseguró (mal) que
había quedado "en el workspace de la conversación": `BashTool` corría con
`cwd` = raíz del usuario (`session.workspace_path`, fijado en
`LocalExecutor.connect()`), NO la carpeta de conversación. Fix quirúrgico en
`capabilities/tools/bash/_tool.py` (sin tocar `ExecutionSession.workspace_path`,
que también usan Docker/Modal): `RuntimeDirs` ganó un campo `cwd` separado de
`root` — `root` sigue siendo la raíz de seguridad para el chequeo
`relative_to` (skills/scratchpad viven fuera de `conversation/<id>/`); `cwd`
= `deps["conversation_dir"]` (fallback a `root`), y ahí arranca el
subprocess; `$WORKSPACE` también apunta a `cwd` ahora. Verificado en vivo:
un archivo escrito por `bash` aterriza en el mismo lugar que `write_file`.

**Prompt actualizado** (`agents/agent.py::_prompts`):
- El texto viejo ("file tools usan rutas user-relative, excepto write_file")
  ya no era cierto tras el fix de nesting — corregido para explicar que los
  file tools son conversation-relative por defecto y que `find_files`/
  `search_text` buscan en todas las conversaciones.
- Regla scratchpad-vs-conversación explícita con ejemplo: artefactos
  intermedios (descargas, datos scrapeados, scripts descartables) →
  `$SCRATCHPAD`; si no se necesita después, ni guardarlo (procesar al
  vuelo); solo lo que el usuario pidió o va a revisar → la conversación.
- Instrucción de autonomía: seguir al siguiente paso solo tras un tool call
  en vez de parar a reportar y esperar; solo detenerse si está realmente
  bloqueado (aprobación, falta info, o tarea completa). Se confirmó con el
  usuario que esto NO era un bug del loop — `ReactLoop` seguía llamando al
  modelo mientras hubiera tool calls; era el modelo el que decidía devolver
  control. El fix es enteramente de prompt, no de mecanismo.

**Tool calls con descripción, no parámetros crudos** — a pedido del usuario,
replicando cómo funciona el propio Claude Code:
- `BashTool.parameters` (`capabilities/tools/bash/_tool.py`) ganó un campo
  `description` (string, obligatorio junto a `command`).
- `core/tool/dispatcher.py::_dispatch()`: `reason_for_approval` ahora usa
  `record.parameters.get("description")` genéricamente (cualquier tool con
  ese campo se beneficia, no solo bash), con fallback al mensaje genérico
  anterior para tools sin ese campo.
- `examples/agent_local_openai.py`: tanto la línea `Tool: ...` como el
  prompt de aprobación muestran la descripción en vez del dict de parámetros
  cuando está disponible.

**`examples/agent_local_openai.py` reescrito** para conversación continua
(reutiliza el mismo `RunContext` entre turnos de `input()`), con
`BashTool()`, `get_weather` (`AUTO_APPROVED`) y `send_email`
(`ASK_APPROVED`) en el toolset — demuestra ambos modos de aprobación. El
helper `_drive()` consume el stream de eventos, y cuando la respuesta queda
en `needs_approval`/`needs_input`, pregunta y/n (o la pregunta de
`ask_user`) en la terminal, aplica la decisión, y llama
`agent.resume_stream_events(...)` recursivamente — soporta varias pausas
seguidas en un mismo turno.

**Verificación**: cada fix de esta entrada se probó en vivo (sin pytest):
`py_compile` en todos los archivos tocados, `LocalExecutor`+`EnvironmentManager`
conectando/ejecutando, `FileSystemTools` reales sobre `conversation/<id>/`,
`BashTool` real confirmando `cwd`/`$WORKSPACE` = conversación, y el `Agent`
completo con `FakeClient` reproduciendo el flujo de aprobación con
descripción de principio a fin. El usuario también lo corrió con
`gpt-5.6-luna` real — así se encontraron los bugs de directorios y el de
autonomía documentados arriba.

**Pendiente, no tocado esta vez**: `guard_state` sigue desactivado (decisión
del usuario). Stack viejo (`tool_executor.py`, `max_ai/executor/`,
`reasoning/react_self_directed.py`, `reasoning/guards.py`) sigue con el
import roto (`executor_legacy.py` borrado manualmente) — destino sin
decidir. `CompletionGate`/middleware de `tool_call` siguen huérfanos, sin
conectar a `ToolDispatcher`.

## 2026-09-20 — Loop estabilizado, workspace único por usuario, EventBus con
ctx, CompletionGate rediseñado como ABC serializable

Sesión larga, varios cambios encadenados. Todo verificado en vivo con
objetos reales (`py_compile` + scripts descartables construyendo
`RunContext`/`FakeClient`/`Agent` real), sin tocar el stack de tests viejo.

**`reasoning/react/loop.py` terminado y revisado** (el usuario lo reescribió
a mano, yo lo revisé por rondas): quedaron resueltos `paused_in_call` (nunca
se ponía `True`, así que una pausa a mitad de la llamada al modelo caía en
`result = loop_state.last_result` con un valor viejo/`None` en vez de
pausar), el `break` faltante en `max_iterations`, la rama de tool calls
nuevos (registrar + `_execute_tools` + fold de resultados), y el stub final
(`raise NotImplementedError`) reemplazado por el `yield
self._reasoning_complete(...)` real. Se extrajo `_run_tools()` como helper
compartido entre el preámbulo de resume y el bloque de tool calls nuevos
(mismo drenado de `ToolMessage`/`ToolApprovalEvent`/`UserInputRequestEvent`
en los dos lugares). Se corrigió que `_run_tools()` yieldeaba el
`ReasoningCompleteEvent` terminal él mismo (acoplaba una señal interna al
tipo de evento público) — ahora solo setea `loop_state.finish_reason` y el
caller (`execute_reasoning_loop`) es el único lugar que construye/yieldea
ese evento. `loop_state` con tipo incorrecto ahora tira `TypeError`
explícito en vez de copiar en silencio y perder la referencia del llamador
(rompía el contrato "todo vive y muere en el objeto que sostiene el
caller"). En `base/reasoning.py`: `_call_llm`/`_call_llm_stream` resetean
`loop_state.last_result = None` al entrar, antes de intentar la llamada —
sin esto, un stream que corta sin chunk final dejaba el resultado de la
iteración anterior, y el loop podía reintentar registrar los mismos tool
calls y explotar con colisión de id en `ctx.tool_state.add()`.

**Tests nuevos en `tests/v2/reasoning/`** (`conftest.py`, `test_react_loop.py`,
`test_base_reasoning.py`): respuesta simple, tool→respuesta, aprobación→
resume, stream incompleto sin reusar `last_result` obsoleto, y el
`TypeError` de `loop_state`. Bloqueado por ahora: `tests/v2/conftest.py`
(el compartido, no el nuevo) importa `Workspace` desde
`max_ai.base.workspace`, que ya no existe ahí tras la reorganización —
rompe la colección de TODO `tests/v2/`, no solo lo nuevo. `test_agent_loop.py`
en el mismo directorio también quedó viejo (usa el `Agent`/`Workspace`
pre-reorg). Decisión del usuario: dejarlo así por ahora, no perseguirlo.

**Workspace único por usuario, `conversation/<id>/` eliminado**: el usuario
decidió que para una coding ability no tiene sentido particionar archivos
por conversación — quiere que todas las conversaciones de un usuario vean
el mismo proyecto, como reabrir el mismo repo. Cambio coordinado en:
`workspace_copy/filesystem.py` (`conversation_root()` → `workspace_root()`,
sin `session_id`; `create_text_file`/`create_directory` ya no prependen
nada — reciben el path ya resuelto por el caller, igual que `write_bytes`;
`_conversation_file_parts` → `_workspace_file_parts`, exige `workspace/...`
en vez de `conversation/<id>/...`), `types/workspace.py`
(`WorkspaceDirectory.conversation_dir` → `workspace_dir`, deja de ser
opcional), `base/workspace.py` (`materialize()` ya no depende de
`conversation_id` para resolver el workspace), `capabilities/tools/
file_system/_toolset.py` (sin `conversation_id` en ninguna tool,
`_conversation_path` → `_workspace_path`), `agents/agent.py` (el dep pasa
de `conversation_dir` a `workspace_dir`, prompt actualizado),
`capabilities/tools/bash/_tool.py` (lee `workspace_dir` para el `cwd` del
subprocess). Bug encontrado en el barrido final, no estaba en el plan
original: `capabilities/executor/local/_executor.py` (`LocalExecutor`, el
vivo) también leía `directory.conversation_dir` — arreglado. `base/agent.py`
(el `Agent` viejo, usado por `cli/chat.py`/`cli/repl.py`) quedó sin tocar a
propósito, por la nota de memoria de cero inversión en `AgentAsTool`.
Bug real encontrado con un script de verificación propio (no hipotético):
`write_file`/`create_directory` con un path tipo `scratchpad/x` quedaban
doble-prefijados (`workspace/scratchpad/session/x` en vez de
`scratchpad/session/x`) porque `_write_target` ya resolvía el prefijo y
`create_text_file` volvía a prepender el suyo — arreglado moviendo TODA la
resolución de prefijo a `_write_target`, `create_text_file`/
`create_directory` dejaron de prependir nada.

**IDs cortos**: nuevo `max_ai/ids.py` con `short_id()` (8 hex = 32 bits,
documentado el porqué). Aplicado en `RunContext.run_id`, `ToolCallRecord.id`,
`ToolCall.id`, `CoreEvent.event_id`, `ObservationRecord.id`, y los session
ids de `LocalExecutor`/`ModalExecutor`. Verificado que los cuatro dan 8
caracteres. Se dejó sin tocar el nombre de archivo temporal de
`capabilities/executor/sync.py` (`.sync-{uuid4().hex}`) — no es un id que
nadie referencia, la entropía completa no molesta ahí.

**Detección de `/tmp` en `BashTool`**: se encontró en vivo (transcript real
con `gpt-5.6-luna`) que el modelo usaba `/tmp` en vez de `$SCRATCHPAD` para
archivos intermedios — confirmado que no era un bug de wiring (`$SCRATCHPAD`
sí llega bien al subprocess), y que `LocalExecutor` no tiene ningún
sandboxing de sistema de archivos (docstring propio: "no sandbox isolation,
no containers"), así que nada le impide al shell tocar `/tmp`. Se descartó
reestructurar directorios o bloquear el comando; en cambio,
`capabilities/tools/bash/_tool.py::execute()` detecta `/tmp` en el comando
(regex best-effort, no un parser de shell real) y agrega un campo `"note"`
al resultado que el modelo ya lee normalmente, sugiriendo `$SCRATCHPAD` — no
bloqueante, mismo canal que ya usa el modelo para stdout/stderr.

**`EventBus` — nombres de hooks + `ctx` en las firmas**: `on_call_tool_start`
→ `on_call_tool` (dispara antes de correr la tool, sobre `ToolCallEvent`),
`on_call_tool` (el de la respuesta) → `on_tool_response` (sobre
`ToolCallResponseEvent`) — nombres más claros sobre cuándo dispara cada uno.
Las tres firmas de observación del `CompletionHandler` Protocol
(`on_call_llm`, `on_call_tool`, `on_tool_response`) ahora reciben `ctx:
RunContext` además del evento — necesario para que un handler pueda guardar
evidencia en `ctx` (el único lugar que sobrevive un resume) en vez de en
`self` (que es una instancia compartida por TODOS los turnos de un
`Agent`, no se reconstruye por turno como `loop_state`). `EventBus.emit()`
y la clausura `emit` de `agents/agent.py::_drive()` pasan `ctx` ahora.
`check_final_response()` ya no explota si un handler no implementa
`on_final_response` (mismo `getattr(..., None)` protector que ya usaba
`emit()`).

**`CompletionGate` rediseñado — de diccionario de callbacks a ABC
serializable** (`base/completion_gate.py`): el diseño anterior
(`CompletionGate(checks={...})`) no dejaba reaccionar en vivo a eventos ni
acumular evidencia de dominio de forma segura entre resumes. Ahora:
`CompletionBase(ComponentBase[CompletionConfig], ABC)` — mismo patrón de
serialización que `WorkspaceBase` (`component_type`/`component_schema`/
`_to_config`/`_from_config`), con `on_call_llm`/`on_call_tool`/
`on_tool_response` no-op por default (documentados uno por uno) y
`on_final_response(ctx) -> CompletionDecision` abstracto — el único punto
donde un gate puede forzar que el loop siga iterando. `runtime_status(ctx,
*, cancelled, exhausted_limits)` se separó como función reusable con los 3
chequeos que aplican a cualquier agente (cancelado/límite agotado/tool call
pendiente), para que una subclase de dominio no tenga que reimplementarlos.
`SimpleCompletionGate(CompletionBase)` mantiene el estilo de diccionario de
`checks` para casos simples que no necesitan hooks en vivo. `agents/
agent.py::Agent.__init__` ganó `completion_handlers: Sequence[
CompletionHandler] | None`, cableado a `EventBus(handlers=...)`.
Verificado de punta a punta: un `FlightCompletionGate(CompletionBase)` de
ejemplo (rechaza la respuesta final si no se llamó `search_flights` con
éxito) forzó al modelo fake a reintentar y buscar antes de aceptar la
respuesta — 3 llamadas al modelo en vez de 1, `finish_reason == "stop"` en
la tercera. Serialización (`dump_component`/`load_component`) verificada
sobre `SimpleCompletionGate`.

**Pendiente, no tocado esta vez**: `AgentResponse.completion` (el campo ya
existe en `types/agent_response.py`) todavía no se llena en `Agent.run()` —
la decisión del gate controla el loop pero no vuelve al caller. Falta crear
el gate default que usará `Agent` cuando nadie pasa `completion_handlers`
(discutido: `max_ai/completion/default/gate.py`, clase `RuntimeCompletionGate`
o reusar `SimpleCompletionGate()` sin checks — decisión de nombre/ubicación
tomada, implementación pendiente). `tests/v2/conftest.py` compartido sigue
roto (ver arriba). Prompt de `agents/agent.py::_prompts` tiene una línea
vieja (`"New files belong in conversation {ctx.session_id}/."`) que quedó
contradictoria tras el cambio de workspace único — identificada, no
arreglada todavía (el usuario prefirió esperar a tener un mecanismo de
composición de prompts antes de seguir editando el string a mano).

## 2026-09-20 (continuación) — Ciclo de vida completo de CompletionGate:
estados, evidencia por turno, TaskCompleteEvent, soporte async

Retomando la entrada anterior del mismo día. Dos rondas de feedback de otro
LLM (revisando el diseño en discusión, sin acceso al código) señalaron 4 + 5
problemas reales, todos verificados contra el código antes de aceptarlos —
no se aplicó ninguno a ciegas. Todo probado en vivo con `Agent`/`FakeClient`
reales, sin pytest (mismo criterio de siempre).

**Bug real: el bus perdía el significado del estado** (`core/events_bus/
bus.py::check_final_response`). Agregaba por "¿hay texto en `reasons`?" en
vez de por el `status` en sí — un handler diciendo `cancelled`/`waiting`
con reasons vacíos terminaba agregado como `"completed"`. Fix: severidad
explícita (`cancelled > waiting > incomplete > completed`, dict
`_STATUS_SEVERITY`), el peor status gana sin importar si trae texto.

**El loop trataba `incomplete`/`waiting`/`cancelled` igual** — mete mensaje
y reintenta, sin importar cuál. Separado en `reasoning/react/loop.py`
(constantes nuevas `WAITING`/`CANCELLED`): `completed` → cierra;
`cancelled` → `raise asyncio.CancelledError()` (mismo camino que la
cancelación externa); `waiting` → pausa el turno como `approval_needed`
(`yield self._reasoning_complete(...)`, `return`, sin reintentar — no tiene
sentido re-preguntarle al modelo algo que no cambió); `incomplete` → único
caso que sigue reintentando. Verificado con gates de prueba para cada
estado (`WaitingGate` pausa en 1 sola llamada, `CancelledGate` lanza la
excepción, el flujo de reintento de siempre sigue igual para `incomplete`).

**Dos `Literal` desincronizados, encontrados por el propio test**:
`AgentResponse.finish_reason` (`types/agent_response.py`) no tenía
`"waiting"` — agregado. Y `loop.py` usaba `NO_LLM_RESULT = "no_llm_result"`
mientras `AgentResponse` solo aceptaba `"no_result"` — estandarizado a
`"no_result"` en `loop.py` (y en el test que lo verificaba).

**`SimpleCompletionGate` perdía sus `checks` al serializar, en silencio**
(bug real, no hipotético): `_to_config`/`_from_config` los heredaba de
`CompletionBase` sin sobreescribirlos, así que `dump_component()` guardaba
config vacío y `load_component()` reconstruía un gate sin ningún check —
"cargaba bien" pero dejaba de proteger cualquier cosa. Causa raíz: el padre
(`CompletionBase`) tenía un `_to_config`/`_from_config` concretos (aunque
vacíos) en vez de dejarlos sin implementar como hace `ComponentBase` con
todo lo demás (`CoreTool` tampoco los implementa — serialización es opt-in
por clase concreta, no un default del padre). Se los quité a `CompletionBase`
— ahora `dump_component()` falla fuerte (`NotImplementedError`) en vez de
mentir. `SimpleCompletionGate` en sí NO se tocó, a pedido explícito del
usuario — el fix fue puramente en el padre.

**`ctx.tool_state.records` se acumulaba para siempre entre tareas del mismo
`ctx`** — confirmado con un test real (no hipotético): tarea 1 busca un
vuelo con éxito, tarea 2 (pregunta ajena, mismo `ctx` reusado) heredaba ese
`search_flights` viejo en `tool_state.records`, así que un gate de dominio
podía dar por completada la tarea 2 sin evidencia propia. Existía
`ToolState.reset()` (`core/tool_state.py`) pero nadie lo llamaba. Fix en
`agents/agent.py::run_stream_events`: al aceptar una tarea nueva (después
de confirmar que no hay tool calls pendientes), `ctx.tool_state.reset()`.

**Hooks renombrados** (`core/events_bus/hooks.py`, `bus.py`,
`completion_gate.py`): `on_call_tool_start` → `on_call_tool` (dispara antes
de correr la tool), el viejo `on_call_tool` (el de la respuesta) →
`on_tool_response`, y `on_call_llm` → `on_model_response` (dispara con la
respuesta ya en mano, no al iniciar la llamada — el nombre viejo era
impreciso). Quedan 4 hooks nada más: `on_model_response`, `on_call_tool`,
`on_tool_response`, `on_final_response` — decidido explícitamente NO
agregar `on_task_start`/clasificador de tareas (ver evidencia por turno más
abajo, se resuelve distinto).

**`runtime_status()` pasó de "cada gate debería llamarla" a "el framework la
garantiza"**: antes dependía de que cada gate se acordara de invocarla
dentro de su propio `on_final_response`. Ahora `loop.py` la llama ÉL MISMO
antes de preguntarle a ningún gate (`blocked = runtime_status(ctx); gate_result
= blocked if blocked is not None else await self.completion_bus.
check_final_response(ctx)`) — un gate de dominio ya no necesita ese
boilerplate, viene garantizado. Verificado: `FlightCompletionGate` sin
llamar `runtime_status()` ella misma sigue funcionando igual.

**Ciclo de vida de evidencia por turno, sin clasificador de tareas** (esto
fue lo que pidió el usuario explícitamente, coincidiendo con la segunda
ronda de feedback del otro LLM — "sin introducir un clasificador de tareas
ni una entidad de tarea adicional"):
- `RunContext.completion_state: dict[str, dict[str, JsonValue]]` — evidencia
  por gate, separada por `gate_id` (nueva `@property` en `CompletionBase`,
  default `type(self).__name__`, overrideable). `JsonValue` en vez de `Any`
  para señalar (a nivel de tipo, no 100% forzado en runtime sobre mutación
  in-place del dict) que debe ser serializable.
- `EventBus.__init__` rechaza `gate_id` duplicados entre handlers
  registrados (`ValueError`) — dos gates con el mismo id se pisarían la
  evidencia en silencio.
- Reset de `completion_state` en el MISMO punto donde ya se resetea
  `tool_state` (`run_stream_events`, tarea nueva) — conservado en pausa/
  resume porque vive en `ctx`, que no se reconstruye ahí.
- Cada gate decide su propia relevancia con reglas condicionales (ej.: "si
  se llamó `reserve_flight`, verificar que la respuesta coincide con la
  reserva; si nunca se llamó, esta regla no bloquea — otro gate puede
  exigir explícitamente que se haya llamado") — no hace falta que el
  framework sepa de qué trata cada tarea.

**`TaskCompleteEvent` conectado, `AgentResponse.completion` poblado**:
existía una clase `TaskCompleteEvent(TasksEvent)` en `core/event_type.py`
reservada pero sin usar (`result: str` genérico) — se reutilizó en vez de
crear una nueva, cambiando el campo a `decision: CompletionDecision`.
`loop.py` la emite (`yield TaskCompleteEvent(...)`) una sola vez, solo en
la rama `COMPLETED`, nunca en pausa/error/límite agotado. `BaseLoopState`
(`base/reasoning.py`) ganó `last_completion_decision: CompletionDecision |
None`, seteado en TODAS las ramas de decisión (no solo completed) justo
antes del branch — así `AgentResponse.completion` (poblado en
`agents/agent.py::_drive`) refleja la última decisión sin importar el
desenlace. Verificado: `TaskCompleteEvent` aparece exactamente 1 vez en el
stream cuando el gate acepta, cero veces cuando pausa por `waiting`, y en
ambos casos `response.completion` queda poblado.

**Soporte para gates async (juez-LLM)**: `on_final_response` puede ser
`def` o `async def` — `EventBus.check_final_response` (ahora `async def`)
llama al handler normal y usa `inspect.isawaitable(decision)` sobre el
RESULTADO (no `inspect.iscoroutinefunction` sobre el callable — más
robusto, cubre un sync que devuelve un awaitable por su cuenta) para
decidir si hace `await`. El docstring de `CompletionGate` ya avisaba que
"expensive verification" debía correr "elsewhere" — ahora puede correr
directamente adentro del gate. Verificado con gates sync y async
registrados juntos, y con un gate async que efectivamente rechaza (fuerza
un segundo `llm_calls`) y no solo pasa siempre.

**Pendiente, no tocado todavía**: el gate default (`RuntimeCompletionGate`
en `max_ai/completion/default/gate.py`, o reusar `SimpleCompletionGate()`)
sigue sin escribirse — es el próximo paso. `tests/v2/conftest.py` compartido
sigue roto. La línea vieja del prompt (`conversation {ctx.session_id}/`)
sigue sin arreglar, mismo motivo que antes.

## 2026-09-20 — Corrección de pausa/resume de CompletionGate en waiting

Por pedido del usuario, corregidos dos casos reproducibles sin cambiar el
almacenamiento de evidencia ni SimpleCompletionGate:

- `agents/agent.py`: un turno nuevo elimina también el snapshot
  `runtime_state.shared_state["reasoning_loop"]`, evitando heredar el presupuesto
  de un turno anterior en waiting. Las pausas guardan/restauran la última decisión;
  waiting guarda además el resultado del modelo pendiente, en formato JSON.
- `reasoning/react/loop.py`: resume de waiting reevalúa el gate antes de consumir
  otra iteración del modelo, incluso al alcanzar el límite. Se comparte la
  evaluación con el recorrido normal mediante `_check_completion`. Si sigue
  esperando, pausa; si aprueba, emite TaskCompleteEvent una vez; si rechaza, continúa
  respetando el presupuesto y conserva la decisión incomplete al agotarlo.

Verificación funcional con Agent real, cliente simulado y workspace temporal,
sin pytest: turno nuevo tras waiting con max_iterations=1; resume aprobado sin
otra llamada al modelo; espera repetida sin evento de finalización; rechazo al
límite; serialización/restauración JSON antes del resume. Todos pasan.
`git diff --check` global detecta dos espacios finales preexistentes en
`base/reasoning.py` (líneas 538 y 658), archivo no modificado en este arreglo.

## 2026-09-20 — RuntimeCompletionGate obligatorio para cierre del turno

`capabilities/completion_gate/gate.py` ya no infiere archivos desde comandos
Bash ni requiere workspace. Comprueba únicamente que un plan existente no tenga
pasos pending/active; done y failed permiten cerrar. Cierre no implica éxito del
objetivo de dominio. Cancelación y tools pendientes siguen comprobándose una
sola vez en el loop, mediante runtime_status.

`agents/agent.py` registra siempre RuntimeCompletionGate además de los handlers
personalizados. Un turno nuevo limpia también ctx.plan para no bloquearse por
un plan del turno anterior; resume conserva el plan.

Verificación con scripts funcionales, sin pytest: los cuatro estados del plan,
plan abierto que exige otra iteración, cierre con paso failed, veto de gate
personalizado, reset de plan entre turnos y regresiones de waiting/resume.
Agent real con cliente simulado y workspace temporal. Diff check de los archivos
modificados OK.

## 2026-09-20 — Bash expected_outputs y verificación de entregables

`BashTool.parameters` incorpora `expected_outputs` opcional: rutas literales de
archivos entregables relativas al workspace, independientes de cd en el comando.
Se rechazan rutas absolutas, traversal, variables y globs. Bash obtiene evidencia
antes/después desde el filesystem: existencia, tamaño, SHA-256 y clasificación
created/modified/unchanged/unavailable; no infiere archivos de stdout ni del shell.
No se aceptan symlinks como evidencia. La descripción de la tool pide declarar
entregables y excluir temporales.

`RuntimeCompletionGate(workspace)` comprueba al cierre que los entregables
explícitos de comandos Bash con exit_code=0 sigan siendo archivos. Usa los
parámetros originales y metadata de Bash (tool_kind), no texto del modelo.
Permite reparación posterior, detecta borrado posterior y no convierte por sí
solo un comando fallido en bloqueo permanente. Los pasos failed del plan siguen
permitiendo cerrar. Agent inyecta su workspace al gate obligatorio.
Existencia no certifica corrección del contenido; unchanged no se presenta como
archivo nuevo. Omitir expected_outputs no demuestra ausencia de entregables.

Verificación funcional sin pytest con Bash real y workspace temporal: creación,
modificación, contenido idéntico, cd y espacios, ausencia, reparación, borrado,
exit_code no cero, directorios, rutas inválidas, symlinks y roundtrip JSON. Pasó
fuera del sandbox tras autorización (subprocesos bloqueaban intermitentemente
dentro). Regresiones Agent waiting/resume también pasan. Diff check acotado OK.

**Aclaración del estado final para retomar**: las entradas anteriores que dejan
el gate default pendiente o dicen que RuntimeCompletionGate no requiere workspace
quedaron superadas por estos cambios. La implementación vigente está en
`max_ai/capabilities/completion_gate/gate.py`, recibe el workspace y se registra
automáticamente en `max_ai/agents/agent.py` junto con los gates personalizados.
El Agent antiguo de `max_ai/base/agent.py` no fue modificado.

Ejemplo del contrato nuevo de Bash:

```json
{
  "command": "python generar_reporte.py",
  "description": "Generar el reporte solicitado",
  "expected_outputs": ["reports/ventas.csv"]
}
```

Los cambios de esta sesión quedaron concentrados en `agents/agent.py`,
`reasoning/react/loop.py`, `capabilities/completion_gate/gate.py` y
`capabilities/tools/bash/_tool.py`. No se implementó un juez LLM ni validación
del contenido de los entregables; el soporte async permite añadir esos gates.

---

## 2026-09-21 — Bug reportado por el usuario: tool no autorizada no cancela la tarea

**No arreglado todavía, dejar constancia para la próxima sesión.**

**Reporte**: tarea con dos pasos dependientes — buscar información en internet
y luego traducirla. El modelo pidió permiso para `curl` (necesario para el
paso de búsqueda), el usuario lo denegó, y en vez de detener la tarea ahí el
modelo siguió adelante: pidió permiso para el paso de traducción (que no
tiene sentido sin la información que nunca llegó) y terminó fallando en el
gate de todos modos.

**Problema de diseño a investigar**: cuando una tool queda no autorizada
(denegada por el usuario) y un paso posterior depende de su resultado, el
turno debería cancelarse o pausar en ese punto — no seguir pidiendo permisos
para pasos río abajo que ya no pueden completarse con datos válidos. Ver si
esto se resuelve en `RuntimeCompletionGate`/`runtime_status()`
(`capabilities/completion_gate/gate.py`) o en el loop mismo
(`reasoning/react/loop.py`), y si hace falta que el gate sepa distinguir
"tool denegada, dependencia rota" de "tool simplemente no se llamó todavía".

**Arreglado, mismo día**: sin rastrear dependencias entre tools (imposible
en general) — se adoptó la misma regla que usa Claude Code con el propio
usuario: un rechazo siempre detiene el turno de una, sin importar qué más
haga el modelo con eso después.

1. `core/primitives.py` — nuevo `FailureReason.APPROVAL_DENIED`, distinto
   del `EXECUTION_ERROR` genérico.
2. `types/tool_call.py` — nueva factory `ToolResult.approval_denied(...)`.
3. `core/tool/dispatcher.py` — los dos puntos que antes devolvían
   `execution_error` para un bloqueo (`record.is_rejected` explícito del
   usuario, y el `permission == "deny"` de `BashTool`) ahora usan
   `approval_denied`.
4. `reasoning/react/loop.py`:
   - Nuevo `TOOL_DENIED = "tool_denied"`.
   - `_run_tools()` revisa, después de correr `_execute_tools`, si algún
     `record` de ese batch quedó con `result.failure_reason ==
     APPROVAL_DENIED`. Si sí, pausa el turno ahí (`finish_reason =
     TOOL_DENIED`) — tiene prioridad sobre cualquier otro
     `ToolApprovalEvent`/`UserInputRequestEvent` pendiente del mismo batch.
   - Los dos puntos que chequeaban `finish_reason in (APPROVAL_NEEDED,
     ASK_USER)` para terminar el turno ahora incluyen `TOOL_DENIED`.
   - Como la pausa ocurre antes de llegar a `_check_completion()`, el gate
     de dominio ni se consulta — no hay forma de que el modelo siga
     intentando pasos posteriores con la dependencia rota.
5. `types/agent_response.py` — `"tool_denied"` agregado al `Literal` de
   `FinishReason`.
6. `agents/agent.py::_drive` — `"tool_denied"` agregado a la tupla que
   conserva el snapshot de métricas/duración para resumir (mismo trato que
   `approval_needed`/`input_needed`/`waiting`; sin esto se perdían las
   métricas acumuladas del turno al resumir después de una denegación).

**Verificación real, sin pytest** (`ReactLoop`/`ToolDispatcher`/
`ToolRegistry`/`RunContext` reales, cliente y tool fake tomados de
`tests/v2/reasoning/test_react_loop.py`): tool `ASK_APPROVED` rechazada con
`.reject("nope")` y turno resumido → `finish_reason == "tool_denied"`,
`client.calls` se queda en 1 (el modelo **no** se vuelve a llamar durante el
resume), `tool.calls == 0` (la tool nunca corre), y el record queda
`CONSUMED` con `result.failure_reason == "approval_denied"`. Regresión del
camino de aprobación normal (`.approve("ok")`) confirmada aparte: sigue
llegando a `finish_reason == "stop"` con la tool ejecutada una vez.
`py_compile` OK en los 6 archivos tocados.

**Nota aparte, no relacionada con este cambio**: al armar la verificación
tropecé con que `tests/v2/reasoning/conftest.py::prompts` (`PromptCtx(stack=
LayerContainer([]))`) ya no funciona — `LayerContainer` ahora exige al menos
una layer y tira `LayerContainerError`. Da igual porque la colección de
pytest de `tests/v2/` ya estaba rota antes por otro motivo (`tests/v2/
conftest.py` compartido importando `Workspace` desde `base/workspace`, que
ya no existe ahí — issue documentado arriba, 09-18). No se tocó ningún
archivo de test.

---

## 2026-09-21 — Loop guards reactivados (stack nuevo), NoProgress/PlanCompletion
absorbidos por RuntimeCompletionGate

Decisión del usuario tras revisar `reasoning/guards.py` guard por guard: de
los 5 guards existentes, 3 se cablean al stack nuevo (`reasoning/react/
loop.py`) tal cual, 2 se dan de baja como guards porque su trabajo ya vive
(o pasa a vivir) en `RuntimeCompletionGate`. **El stack viejo
(`react_self_directed.py`) no se tocó** — sigue usando `default_guards()`
con los 5 guards intactos, cero inversión ahí (nota de memoria vigente).

**1. `guard_state` reactivado** (`base/reasoning.py`): estaba comentado
desde antes (bug documentado el 09-18, "guard_state silenciosamente roto").
Descomentado tal cual — de yapa esto también arregla el stack viejo, que
ya intentaba leerlo y lo tenía roto en silencio; no fue trabajo extra,
solo destapar el field.

**2. `BudgetGuard` — mensaje actualizado** (`reasoning/guards.py`, clase
compartida por ambos stacks). Antes solo decía "consolidá tu respuesta";
ahora, a pedido del usuario, apunta a una decisión concreta: si `ask_user`
está en el toolset, sugiere preguntarle al usuario si seguir; si no,
sugiere darle un resumen de lo que hay. Incluye `~{pct}%` del budget usado
además de iteraciones restantes.

**3. `SchemaRetryGuard`/`RepetitionGuard`/`BudgetGuard` cableados en
`reasoning/react/loop.py`** — sin tocar `default_guards()` (que sigue
sirviendo al stack viejo con los 5): `ReactLoop.__init__` ahora acepta
`guards: Sequence[LoopGuard] | None`, default explícito
`[SchemaRetryGuard(), RepetitionGuard(), BudgetGuard()]` (no
`NoProgressGuard`/`PlanCompletionGuard` — ver punto 5). Nuevo
`_run_guards()`: corre `after_tool_round` de cada guard después de un
round de tool calls que no pausó el turno, junta el steering en un
`SystemMessage(source=HARNESS, ...)`, y lo guarda en una variable local
(`pending_steering`) que viaja como `transient_messages` a la SIGUIENTE
llamada al modelo únicamente — nunca se escribe en `ctx.messages`
(`base/reasoning.py::_model_context` ya soportaba `transient_messages`
desde antes, sin uso real en este loop hasta ahora). `GuardContext.tools`
se arma desde `self.tool_catalog` (ya existía como property).

**4. No se implementó el hook `on_final_answer`** en el loop nuevo — con
`NoProgressGuard`/`PlanCompletionGuard` fuera del set del stack nuevo,
ningún guard cableado lo usa, así que no hace falta el mecanismo de veto
sobre la respuesta final. Simplifica el alcance del cambio.

**5. `NoProgressGuard`/`PlanCompletionGuard` — bajados como guards del
stack nuevo, sin borrar las clases** (siguen existiendo en `guards.py`
para el stack viejo):
- `PlanCompletionGuard`: decisión del usuario — su chequeo (plan con pasos
  sin terminar) ya está cubierto por `RuntimeCompletionGate.
  on_final_response` (`ctx.plan.has_unfinished_steps()`). No se tocó el
  mensaje de esa rama del gate en este cambio.
- `NoProgressGuard`: su lógica se re-implementó DENTRO de
  `RuntimeCompletionGate` (`capabilities/completion_gate/gate.py`), pero
  con una diferencia de comportamiento a pedido del usuario — no
  "1 veto y después se acepta en silencio" (lo que hacía el guard viejo),
  sino **2 strikes con corte real**: contador `empty_streak` en
  `ctx.completion_state["RuntimeCompletionGate"]` (turn-scoped, mismo
  mecanismo que ya usan los gates de dominio). 1ª respuesta vacía (sin
  texto, sin tool_calls) → `status="incomplete"`, reintenta una vez. 2ª
  vacía seguida → `status="waiting"` — el loop ya sabe pausar ahí sin
  volver a llamar al modelo (mismo camino que cualquier otro
  `"waiting"`), en vez de aceptarla como `"completed"` en silencio. No
  hizo falta tocar `loop.py` para esto — reusa el status `"waiting"` que
  ya existía.

**Verificación real, sin pytest** (`ReactLoop`/`ToolDispatcher`/
`RuntimeCompletionGate`/`EventBus` reales, cliente scripted):
- `RepetitionGuard` activo por default: 3 llamadas idénticas a una tool
  corren las 3 (el guard no bloquea ejecución, solo steerea), y el texto
  de steering aparece en el `ctx` que ve el cliente en alguna llamada
  posterior — confirmado que NUNCA aparece en `ctx.messages` real (queda
  transitorio).
- `BudgetGuard` dispara cerca del techo de iteraciones (`max_iter=4`,
  3 rounds de tool calls) — el texto "iteration(s) left" aparece en el
  contexto que ve el modelo.
- Gate con dos respuestas vacías seguidas: exactamente 2 llamadas al
  modelo (no 3), `finish_reason == "waiting"`,
  `ctx.completion_state["RuntimeCompletionGate"]["empty_streak"] == 2`,
  y aparece un `CompletionRejectedEvent` de la 1ª vez.
- Regresión de `tool_denied` (sesión anterior) re-corrida, intacta.
- `py_compile` OK en `base/reasoning.py`, `reasoning/guards.py`,
  `reasoning/react/loop.py`, `capabilities/completion_gate/gate.py`, y de
  paso `reasoning/react_self_directed.py`/`agents/agent.py` (para
  confirmar que reactivar `guard_state` no rompe el stack viejo).

**No tocado**: `reasoning/__init__.py` (sigue exportando los 5 guards para
el stack viejo), `default_guards()` (sigue devolviendo los 5), `guards.py`
en general fuera del mensaje de `BudgetGuard`, `react_self_directed.py`.

---

## 2026-09-21 — `reasoning/eval.py` borrado, era 100% código muerto

Decisión del usuario tras revisar para qué servía: `EvalCheck`/`EvalResult`/
`EvalConfig` (auto-evaluación post-loop con score fraccionario vs.
threshold, reintentando el loop entero hasta `max_retries`) nunca se
conectó a ningún loop — confirmado por grep, cero lugares construían un
`EvalResult` o emitían el `EvalEvent` correspondiente en todo el repo.
Solo existían 3 piezas de andamiaje sin productor: el módulo en sí, el
evento en `core/event_type.py`, y un handler de renderizado en
`cli/renderer.py` esperando un evento que nunca llegaba. Su trabajo
(juez de calidad) ya está cubierto y es estrictamente más barato con un
`CompletionBase` async (no reinicia el loop entero, solo pide una vuelta
más con la razón puntual) — la única capacidad que `EvalConfig` tenía y
`CompletionDecision` no tiene hoy es un score fraccionario multi-criterio
(no binario); si algún día hace falta, se implementa como gate propio, no
resucitando esto.

**Borrado**: `reasoning/eval.py` (archivo completo).
**Editado**: `core/event_type.py` (import de `EvalResult`, clase
`EvalEvent`, entrada en el union `AgentEvents`), `cli/renderer.py` (import
de `EvalEvent`, la rama `isinstance(event, EvalEvent)` en el dispatch, y
el método `_on_eval`).

**Verificación**: grep confirma cero referencias a `EvalResult`/
`EvalCheck`/`EvalConfig`/`EvalEvent`/`reasoning.eval` en todo el repo tras
el borrado. `py_compile` OK en `core/event_type.py`/`cli/renderer.py`,
import real de ambos módulos sin error. Re-corridas las verificaciones
funcionales de la entrada anterior (guards + gate + tool_denied) — todas
siguen pasando, `core/event_type.py` es ampliamente importado y no se
rompió nada. 

---

## 2026-09-21 15:47 UTC — Contrato de memoria por sesión y categoría

**Decisión del usuario**: guardar memoria es una instrucción flexible para el
agente: puede hacerlo por petición del usuario o por iniciativa propia cuando
sea útil. Actualizar la memoria de sesión durante la compactación será un paso
obligatorio, pero pertenece a otro sistema y NO se implementó en este cambio.

**Editado**: `max_ai/base/memory.py`. Cada `CoreMemoryRegistry` queda ligado a
`user_id` y `session_id`. `MemoryRecord` ahora contiene `category`, `memory` y
`updated` (fecha UTC asignada por el sistema). Una categoría identifica una
entrada dentro de la sesión; actualizarla reemplaza todo su contenido. Se
eliminaron `RecallQuery`, `MergePolicy`, `EmbedOne`, `EmbedMany`, los parámetros
de embeddings y la fusión/deduplicación automática del contrato base.

**Herramientas**:
- `get_context()`: memorias de la sesión actual, más recientes primero,
  filtradas por `context_days` (default 30; `None` desactiva el filtro). La tool
  devuelve `session_id` y `memories`, con fechas serializadas para JSON.
- `list_category()`: todas las categorías de la sesión, incluso las excluidas
  del contexto por antigüedad.
- `create_or_update(category, memory)`: crea o reemplaza y comunica cuál ocurrió;
  puede ejecutarse sin compactación ni indexación.
- `delete_memory(category)`: elimina en la sesión actual y distingue entre
  eliminado y no encontrado; conserva aprobación `ASK_APPROVED`.
- `search_memory(text)`: delega recuperación de otras sesiones del mismo usuario
  al backend; devuelve categoría/contenido/fecha y sesión de origen mediante
  `MemorySearchResult`, sin copiar esos recuerdos a la sesión actual.

Se conserva `MemoryToolMode`: `NONE` sin tools, `READ_ONLY` con contexto/listado/
búsqueda, `FULL` agrega escritura y borrado. Categorías sensibles a mayúsculas,
con espacios extremos eliminados; entradas vacías y `context_days` inválidos
se rechazan.

**Nuevo contrato del backend**: `connect`, `disconnect`, `_read_session`,
`_write_memory` (upsert atómico, devuelve si creó), `_delete_memory` (devuelve si
existía) y `_search_memory`. El backend debe aislar usuarios/sesiones y eliminar
o invalidar chunks obsoletos al reemplazar/borrar. Embeddings, ranking, límites
de resultados, persistencia e indexación quedan fuera de la base. Se acordó
persistencia JSON/YAML por sesión, pero el backend de archivos todavía no se
adaptó; este cambio solo define el contrato y las herramientas.

**Pendiente / incompatibilidad deliberada del cambio de contrato**: adaptar
`max_ai/capabilities/memory/local.py`, `sqlite.py`, la plantilla `MemoryLayer.j2`
y sus consumidores. SQLite todavía importa símbolos eliminados; los backends
y tests antiguos de memoria NO son compatibles con esta base. No se migraron
bases de datos, no se conectó memoria al runtime y no se implementaron búsqueda
semántica real, indexación ni integración con compactación.

**Verificación**: nuevo `tests/unit/test_memory_contract.py`, usando un backend
en memoria para comprobar reemplazo/borrado por sesión, ventana temporal sin
borrado, categorías antiguas, búsqueda con procedencia sin copiar resultados,
modos de herramientas, ejecución real de `FunctionAsTool`, serialización y
validación de entradas. `.venv/bin/python -m pytest -q
 tests/unit/test_memory_contract.py --tb=short`: **8 passed**. Estos tests no
validan los backends antiguos ni embeddings. `git diff --check` limitado a los
archivos de implementación/pruebas modificados pasó.

---

## 2026-09-22 — `capabilities/memory/local/` reescrito contra el contrato
actual; `capabilities/skills/local/` movido al mismo patrón de carpeta

Contexto: el usuario ya había construido `base/memory.py` (contrato
`CoreMemoryRegistry`, session+category, sólido) y `base/skills.py`
(`CoreSkillRegistry`, ya funcionando). Al revisar antes de conectarlos a
`agents/agent.py`, encontré que **los backends concretos de memoria
estaban rotos contra el contrato actual**:

- `capabilities/memory/sqlite.py` fallaba al importar —
  `ImportError: cannot import name 'RecallQuery' from 'max_ai.base.memory'`
  (tipo que ya no existe). Como `capabilities/memory/__init__.py` lo
  importaba eager, **todo el paquete `capabilities.memory` era
  inimportable**.
- `capabilities/memory/local.py` (el archivo viejo) implementaba métodos
  con otros nombres (`_read_all`/`_write_many`/`_delete_many`, usando
  `record.key`, que `MemoryRecord` ni tiene) en vez de los abstractos
  reales del contrato (`_read_session`/`_write_memory`/`_delete_memory`/
  `_search_memory`), y su `__init__` no pasaba `session_id` (obligatorio
  en `CoreMemoryRegistry.__init__`). No instanciaba.

Diagnóstico: el contrato se rediseñó (session-scoped, category-based) en
algún momento y los backends se quedaron con una versión vieja.
`capabilities/skills/local.py`, en cambio, **sí coincidía** con
`base/skills.py` — solo necesitaba la reestructuración de carpeta, no un
rewrite.

**`capabilities/memory/local/`** (nuevo paquete, reemplaza
`local.py`, a pedido explícito del usuario — "creemos una carpeta local
... ignoraremos el otro [archivo]"):
- `__init__.py` + `_registry.py`, mismo patrón que
  `capabilities/executor/local/`.
- `LocalMemoryRegistry` reescrito contra los 4 abstractos reales.
  Layout: un JSON por `(user_id, session_id)` en
  `{base_path}/memory/{user_id}/{session_id}.json` — session-scoped
  como pide el contrato (`_read_session` solo lee el propio archivo,
  `_search_memory` recorre los `*.json` de otras sesiones del mismo
  `user_id`). `user_id`/`session_id` validados contra
  `^[A-Za-z0-9_-]+$` (son dos segmentos de path ahora, directorio +
  archivo — más superficie que el `local.py` viejo, que solo usaba
  `user_id`).
- `capabilities/memory/__init__.py`: el import de `SQLiteMemoryRegistry`
  pasó a lazy (PEP 562, mismo patrón que `reasoning/__init__.py`) — así
  el paquete es importable aunque `sqlite.py` siga roto.
  **`sqlite.py` NO se tocó, sigue roto** — fuera de alcance de hoy.

**`capabilities/skills/local/`** (nuevo paquete, reemplaza `local.py`):
mismo patrón de carpeta, contenido movido sin cambios de lógica (solo un
punto más de profundidad en los imports relativos). `agents/agent.py`
sigue importando por la misma ruta (`capabilities.skills.local`), sin
tocar ese archivo.

**Verificación real, sin pytest**: `LocalMemoryRegistry` — crear/
reemplazar categoría, `list_category`, búsqueda cross-session (encuentra
memoria de otra sesión del mismo usuario, NO se auto-encuentra en la
propia — el filtro ya lo hace `base/memory.py`), borrar/borrar-de-nuevo,
las 5 tools se generan bien, bloqueo de `user_id="../evil"` confirmado.
`LocalSkillRegistry` — instancia igual que antes por ambas rutas
(`capabilities.skills.local` y el re-export de `capabilities.skills`),
`agents/agent.py` importa sin error. `py_compile` OK en todos los
archivos nuevos/tocados. Grep confirma cero referencias sueltas a las
rutas de archivo viejas.

**Pendiente, próximo paso**: conectar `memory`/`skills` de verdad a
`agents/agent.py` — hoy `Agent.__init__` no tiene parámetro `memory` en
absoluto, y skills se inyecta al prompt con un string armado a mano en
`_prompts()` en vez de vía `SkillsLayer` (que existe pero nunca se
agregó a `self._stack`, que hoy solo tiene `AgentPolicyLayer`).
`sqlite.py` de memoria sigue roto, sin decidir si se reescribe o se
retira.

---

## 2026-09-22 (continuación) — Auditoría de `knowledge`, `capabilities/
knowledge/local/` reestructurado, `KnowledgeBlock.score` eliminado

**Auditoría, a pedido del usuario**: a diferencia de memoria, `base/
knowledge.py` (`CoreKnowledgeRegistry`, contrato: `search(query, limit)`)
y `capabilities/knowledge/local.py` (el archivo viejo) **coincidían
perfectamente** — sin mismatch de contrato. Confirmado corriendo
`tests/test_local_registries.py`: antes de tocar nada, ya pasaban
`test_local_knowledge_registry_search`/`_missing_file`. De paso, correr
toda esa suite reveló 3 fallas preexistentes, ninguna de knowledge, no
causadas por ningún cambio de esta sesión — quedan anotadas, no
arregladas: dos tests de memoria (`test_local_memory_registry_round_trip`,
`test_local_memory_rejects_invalid_keys`) siguen escritos contra la API
vieja de memoria (`update_fact`/`delete_fact`, `.key`/`.category`/
`.content`) de antes del rediseño de contrato del 21; un test de skills
(`test_local_skill_registry_loads_skill`) falla por un campo de
`WorkspaceDirectory` (`workspace_dir`) relacionado al rename
`conversation_dir → workspace_dir` del 20, no a nada de hoy. También
`tests/test_capability_registry.py`/`test_core_agent_smoke.py` ni
colectan — imports a rutas pre-reorg del 18 (`max_ai.tools`,
`max_ai.executor.local`).

**Reestructuración**: `capabilities/knowledge/local/` (nuevo paquete,
`__init__.py` + `_registry.py`, mismo patrón que memory/skills),
`local.py` viejo borrado. Contenido movido sin cambios de lógica.

**`KnowledgeBlock.score` eliminado** (`core/blocks.py`), a pedido
explícito del usuario: nunca se usaba fuera de la propia ranking interna
de `LocalKnowledgeRegistry.search()`, y el valor en sí (cosine similarity
crudo) no le aporta nada al modelo — confirmado por grep, `KnowledgeLayer`
(`stacks/knowledge_layer.py`) documenta explícito que no renderiza
resultados de búsqueda en el prompt en absoluto, y ningún otro consumidor
en el repo lo leía. La similitud se sigue calculando y usando para
ordenar/filtrar dentro de `search()` (variable local `score`, ya no
pegada al `KnowledgeBlock` devuelto). Actualizados docstrings (el
"file shape" de ejemplo ya no muestra `"score": null`) y
`tests/test_local_registries.py::test_local_knowledge_registry_search`
(las 2 aserciones de score cambiadas por orden de resultados — el
ranking se sigue verificando, solo que por posición en la lista en vez
de por valor de score expuesto).

**Verificación**: `py_compile` OK. `tests/test_local_registries.py`:
mismas 6 pasan + las 3 fallas preexistentes intactas — 0 regresiones,
`test_local_knowledge_registry_search` verde de nuevo tras el ajuste.
Nota: pydantic ignora campos extra por default, así que archivos de
knowledge viejos en disco con `"score": null` siguen leyéndose sin
error — no hace falta migrar datos existentes.

---

## 2026-09-22 (continuación) — `capabilities/executor/` reorganizado en
carpetas por backend (docker/, modal/), a pedido del usuario

Mismo patrón de carpeta que memory/skills/knowledge, aplicado acá con más
cuidado porque hay scripts invocados por subprocess con paths punteados
literales (`python -m ...`) dentro del sandbox — un typo ahí falla recién
en runtime, no al importar.

**Split backend-specific vs. compartido**, decidido por quién importa qué
(`docker.py` solo usaba `.process`/`.remote`; `modal.py` solo usaba
`.remote`/`.sync`; nadie más usaba `.sync`/`modal_command.py` — Local/
Docker ya comparten filesystem por bind mount, así que nunca necesitaron
snapshot/apply explícito, solo Modal):

- **`capabilities/executor/docker/`** (nuevo, reemplaza `docker.py`):
  `__init__.py` + `_executor.py` — mismo patrón que `local/`.
- **`capabilities/executor/modal/`** (nuevo, reemplaza `modal.py` +
  `modal_command.py` + `sync.py`, los 3 juntos porque son exclusivos de
  Modal): `_executor.py`, `modal_command.py`, `sync.py` — estos dos
  últimos SIN el prefijo `_` porque se invocan como
  `python -m max_ai.capabilities.executor.modal.{modal_command,sync}`
  dentro del sandbox; les agregué un comentario en cada uno avisando que
  mover/renombrar el archivo exige actualizar ese string en
  `_executor.py`.
- **Quedaron en el nivel de `capabilities/executor/`** (compartidos por
  ≥1 backend, o referenciados por su propio path estable):
  `process.py` (usa `local/`, `docker/`, `modal/modal_command.py`),
  `reference.py`/`remote.py` (clase base común de Docker+Modal),
  `worker.py` (entrypoint compartido, invocado como
  `python -m max_ai.capabilities.executor.worker` desde `remote.py` —
  ese string no cambió, `worker.py` no se movió).

**Bug real encontrado probando, no hipotético**: `modal/__init__.py`
importaba `ModalExecutor` eager (`from ._executor import ModalExecutor`),
y `_executor.py` importa `.sync` — así que al invocar
`python -m max_ai.capabilities.executor.modal.sync` dentro del sandbox,
Python tiene que inicializar el paquete `modal` primero (requisito de
`-m` para paths con puntos), lo que ya importaba `sync.py` como módulo
normal ANTES de que `runpy` lo ejecutara como `__main__` —
`RuntimeWarning: 'max_ai.capabilities.executor.modal.sync' found in
sys.modules ... prior to execution`. Funcionaba igual (resultado
correcto) pero es un olor real que no existía con el archivo suelto de
antes. Arreglado poniendo `modal/__init__.py` en lazy-load (PEP 562,
mismo patrón que ya usa `capabilities/executor/__init__.py`) — así
importar el paquete `modal` para resolver `modal.sync` ya no arrastra
`_executor.py`. `docker/__init__.py` se dejó eager (no tiene un caso
análogo, ningún script propio invocado por `-m` adentro de `docker/`).

**Verificación real, sin pytest**: import de `DockerExecutor`/
`ModalExecutor` por las 3 rutas (paquete propio, top-level
`capabilities.executor`, instanciación real de cada uno). Invocación real
`python -m max_ai.capabilities.executor.modal.sync snapshot <dir>` contra
un directorio temporal real — devuelve el snapshot correcto, sin warning
tras el fix. Invocación real `python -m
max_ai.capabilities.executor.modal.modal_command` con un payload JSON por
stdin — ejecuta el comando y devuelve el `ExecutionResult` correcto.
Grep confirmó cero referencias sueltas a los paths viejos en todo el
repo, incluidos `Dockerfile`/`README.md` de esa carpeta. `py_compile` OK
en los 6 archivos nuevos/tocados. Re-corridas las verificaciones de
sesiones anteriores (`test_local_registries.py`, guards/gate) — mismas 6
pasan + 3 fallas preexistentes sin relación, sin regresión.

**No verificado end-to-end** (requiere Docker real corriendo / cuenta de
Modal real): el flujo completo de `DockerExecutor.connect()`/
`ModalExecutor.connect()` contra un daemon/SDK real — fuera de alcance
sin esas credenciales/entorno.

---

## 2026-09-22 (continuación) — `core/executor/` nuevo (lo compartido de
verdad), `Dockerfile` movido a `docker/`, `README.md` de executor borrado

Ajuste a la reorganización anterior, a pedido del usuario: `process.py`,
`reference.py`, `remote.py`, `worker.py` (los 4 archivos compartidos que
se habían quedado en `capabilities/executor/`) se movieron a
`max_ai/core/executor/` — mismo criterio que ya se usó en la reorg del
18/09 para `ToolRegistry`/`ToolDispatcher` (`base/` → `core/tool/`): son
fijos, no pluggable, a diferencia de los backends (`local/`/`docker/`/
`modal/`, que sí son intercambiables y se quedan en `capabilities/`).
`core/` ya tenía precedente de depender de `capabilities/` cuando hace
falta (`core/tool/dispatcher.py` ya importa `capabilities.tools.ask_user`/
`capabilities.tools.bash` directo) — mismo patrón acá con
`core/executor/worker.py` importando `capabilities.executor.local`.

**Detalle importante, ya resuelto**: como `core/executor/` y
`capabilities/executor/` están a la MISMA profundidad respecto a
`max_ai/` (2 niveles), casi ningún import relativo interno de los 4
archivos movidos cambió — los `...base.x`/`...types.x` siguen con la
misma cantidad de puntos. Solo cambiaron los que cruzan de un árbol al
otro: `worker.py::from .local import LocalExecutor` (sibling antes) →
`from ...capabilities.executor.local import LocalExecutor` (cross-package
ahora), y el string de subprocess en `remote.py`
(`"max_ai.capabilities.executor.worker"` → `"max_ai.core.executor.worker"`).
Los 3 backends (`local/_executor.py`, `docker/_executor.py`,
`modal/_executor.py`, `modal/modal_command.py`) actualizaron sus imports
de `..process`/`..remote` a `....core.executor.process`/
`....core.executor.remote`. Dos consumidores externos por fuera de
`executor/` también apuntaban a la ruta vieja de `reference.py`:
`agents/agent.py` (el activo) y `base/agent.py` (el viejo) — ambos
actualizados, porque dejarlos así los rompía de verdad (no es "invertir"
en el stack viejo, es no dejarlo roto por un movimiento que hice yo).
`base copy/agent.py` (snapshot/backup deliberado) se dejó sin tocar,
a propósito.

**`Dockerfile` → `capabilities/executor/docker/Dockerfile`**: no es
"el backend Docker", es la receta de la imagen de runtime que necesitan
Docker Y Modal (mismos requisitos: usuario no-root, `/workspaces`
escribible, `max_ai` instalado) — pero el usuario decidió que igual
viva junto a Docker, que es quien la referencia por nombre en su comando
de build. Actualizado el comentario interno con el path nuevo
(`docker build -f max_ai/capabilities/executor/docker/Dockerfile ...`).
El `COPY pyproject.toml README.md /opt/maxai/` de adentro del Dockerfile
apunta al `README.md` de la RAÍZ del repo (contexto de build), no al que
se borró — sin relación.

**`capabilities/executor/README.md` borrado**: estaba muy desactualizado
(referenciaba `from max_ai.base.agent import Agent` como si fuera el
stack activo, `from max_ai.runtime import ...` — un path que ya no
existe, y describía el diseño viejo de `Agent` con `registry` público) —
de antes de la reorg del 18/09, ninguna referencia externa a él en el
repo.

**Bug preexistente encontrado, NO causado por este cambio ni arreglado**:
`core/executor/worker.py` (movido tal cual, contenido idéntico) tiene
`from ...base.workspace import Workspace` — `Workspace` ya no existe en
`base/workspace.py` desde la reorg del 18/09 (quedó `WorkspaceBase`).
Confirmado que el archivo viejo, antes de moverlo, ya tenía exactamente
este mismo import roto — mismo patrón que `tests/v2/conftest.py` y
`tool_executor.py`, ya documentados. No se tocó — es un fix de código
real (entender la API actual de Workspace), no de reestructuración.

**Verificación real, sin pytest**: import + instanciación real de
`LocalExecutor`/`DockerExecutor`/`ModalExecutor` y de
`core.executor.{RemoteExecutor,ToolReference,reference_for,run_process}`.
Confirmado el string de subprocess actualizado en `remote.py`. Import de
`agents/agent.py` y `base/agent.py` sin error. `py_compile` OK en los 11
archivos nuevos/tocados. Grep confirma cero referencias sueltas a los
paths viejos (`capabilities.executor.{process,reference,remote,worker}`,
`capabilities/executor/Dockerfile`, `capabilities/executor/README.md`)
en todo el repo. Re-corridas `test_local_registries.py` y las
verificaciones de guards/gate/tool_denied de sesiones anteriores —
mismos resultados, 0 regresión.

## 2026-09-22 — Textual CLI inicial

Se creó el CLI activo en `max_ai/cli/` con `MaxAIApp`, basado en Textual.
`cli copy/` queda intacto como respaldo del REPL Rich anterior. La interfaz
incluye transcript central con streaming, sidebar con agente/capacidades,
editor inferior, plan/tool events, errores, aprobaciones y preguntas del
agente mediante una future asíncrona. `run_repl(agent)` mantiene el punto de
entrada público usado por el ejemplo existente.

Se agregó `textual>=4.0.0` y se actualizó `uv.lock` (resuelto Textual 8.2.8).
Smoke test real con `App.run_test()`: montaje, widgets y binding de limpiar
transcript OK. `py_compile` y `git diff --check` OK.

## 2026-09-22 — Capacidades opcionales conectadas al prompt de Agent

`agents/agent.py` acepta `memory`, `skills` y una secuencia `knowledge`.
Solo las capacidades suministradas aportan capas/herramientas; ya no crea
LocalSkillRegistry por defecto. Funciones separadas para registrar herramientas,
construir el stack, preparar skills, validar el alcance de memoria, recoger
variables y renderizar PromptCtx. Las herramientas ligadas a los registros se
ejecutan en host; nombres duplicados fallan explícitamente.

Stack base: AgentPolicyLayer, TaskAnalysisLayer y RenderingLayer. Se agregan
SkillsLayer, KnowledgeLayer y MemoryLayer según configuración. ContextLayer
queda fuera por decisión del usuario. Se eliminó la concatenación manual de
workspace/skills y su instrucción obsoleta de carpeta por conversación.
Skills usa get_skills() y conserva materialize() para los archivos reales.
Memory se lee en cada run/resume; exige un RunContext cuyo user_id/session_id
coincida con el registro. Los registros suministrados siguen siendo propiedad
del caller. Knowledge conecta al ejecutar su herramienta de búsqueda.

La plantilla MemoryLayer se adaptó a category/memory/updated y al reemplazo
completo por categoría, respetando modos de lectura/escritura. El snapshot del
prompt no se refresca entre iteraciones de un mismo loop: los resultados de
tools llegan por mensajes y el siguiente run/resume obtiene un snapshot nuevo.

Verificación: tests/unit/test_agent_capability_prompts.py, 7 passed, con Agent
y dispatcher reales y cliente de modelo simulado: ausencia de capacidades,
guardado de memoria por tool y refresco posterior, rechazo de otra sesión,
modos NONE/READ_ONLY/FULL, catálogo/materialización de skills, ejecución de
search_docs y colisiones de nombres. git diff --check acotado OK. No se tocó
la copia de respaldo del agente ni el builder del stack antiguo.

---

## 2026-09-22 (continuación) — Type hints agregados a `core/environment/manager.py`

A pedido del usuario, tipado completo de las firmas que no lo tenían:
`_validate`, `_expire`, `_disconnect` (sin ningún tipo antes), y return
types faltantes en `acquire` (`AsyncIterator[ExecutionSession]`, es un
`@asynccontextmanager`), `rebuild`, `close`, `__aenter__`
(`EnvironmentManager`), `__aexit__`. De paso, `_Entry.idle:
asyncio.Task | None` → `asyncio.Task[None] | None` (parametrizado, ya
tenía tipo pero genérico sin especializar). Agregado `from __future__
import annotations` al archivo (no lo tenía) para permitir el
auto-retorno `-> EnvironmentManager` sin comillas.

**Verificación real, sin pytest**: `py_compile` OK. Ciclo completo real
con `LocalExecutor`+`LocalWorkspace` — `acquire()` conecta, ejecuta un
comando real (`echo hi`), libera; un segundo `acquire()` reutiliza la
misma sesión (confirmado por `session.id` igual); `close()` limpia sin
errores. Solo anotaciones — cero cambio de comportamiento, confirmado.
`agents/agent.py` (que construye `EnvironmentManager` internamente)
sigue importando sin error.

**Aviso de colisión con otra sesión, ya resuelto**: justo después de
aplicar y verificar lo de arriba, el archivo cambió en disco por otra
sesión guardando desde una copia vieja (sin mis hints, pero con
`__init__`/`_validate` reformateados en multilínea — alguien más lo tocó
en paralelo) — los hints se perdieron. Los reapliqué sobre el contenido
nuevo (mismo formato multilínea, respetado) y re-verifiqué todo de
nuevo (compile + el mismo ciclo funcional real) — quedaron. Si ves este
archivo sin los return types otra vez, no asumas que fue un olvido mío;
puede ser el mismo patrón de sesiones concurrentes pisándose que ya se
documentó varias veces en este log (17/09, 18/09).

---

## 2026-09-22 (continuación) — `tool_denied` ya no es un pause mudo:
followup restringido a `ask_user`/`update_plan`

El usuario volvió a pegar una transcripción real (`examples/
agent_local_openai.py`, corrida con `gpt-5.6-luna`) mostrando el mismo
síntoma que la entrada del 21 (bug reportado) — tras rechazar
`send_email`, apareció un saludo genérico de auto-presentación
("Hellooo! I'm LocalDemo...") en vez de una respuesta relacionada.

**Causa raíz del síntoma exacto (nueva, no reportada antes)**:
confirmado leyendo `AgentResponse.final_message`
(`types/agent_response.py`) — busca hacia atrás en TODO el historial
(`message_history` + `messages` de la sesión completa, no solo el turno
actual) el último `AssistantMessage` con texto no vacío. Como
`tool_denied` no generaba ningún mensaje nuevo con texto, el fallback
encontraba el saludo de auto-presentación de un turno viejo (la primera
vez que el modelo se presentó en esa conversación) y lo re-mostraba cada
vez que un turno terminaba sin texto propio — por eso aparecía dos veces
idéntico.

**El fix, tal como se diseñó en la sesión anterior**: en vez de terminar
el turno en silencio al detectar `tool_denied`, se hace UNA llamada más
al modelo con el toolset restringido a `{ask_user, update_plan}` — el
modelo no puede reabrir la cadena de aprobaciones, solo reconocer el
rechazo (vía `ask_user`) o cerrar el plan. Steering corto en inglés (a
pedido del usuario, para ahorrar tokens):

> "The user denied: {tool} ({reason}). Assess how much this blocks the
> task. Explain to the user what happened and why, then call ask_user
> with exactly one of: (1) cancel the remaining task, (2) retry the
> tool, or (3) other instructions on how to proceed."

**Archivos**:
1. `base/reasoning.py` — `_call_llm`/`_call_llm_stream` ganaron
   `tools_override: Sequence[CoreTool] | None`. Reemplaza `self._tools`
   (el catálogo completo) solo para esa llamada puntual; sin él, se
   comportan exactamente igual que antes.
2. `reasoning/react/loop.py`:
   - `_denied_records()` — extraído de `_run_tools` (antes inline), ahora
     reusable.
   - `_tool_denied_followup(denied)` — arma el steering + la lista de
     tools restringida (`ask_user` + `update_plan`, los que estén
     registrados). Devuelve `None` si `ask_user` NO está registrado —
     en ese caso no tiene sentido pedirle que pregunte, así que cae al
     comportamiento viejo (pausa dura, `finish_reason="tool_denied"`,
     sin llamada extra) — mismo resultado que la entrada del 21 para
     agentes sin `ask_user`.
   - Los dos puntos que manejaban `TOOL_DENIED` (preámbulo de resume y
     la rama de tool_calls del loop principal) ya no lo tratan igual
     que `APPROVAL_NEEDED`/`ASK_USER` (pausa dura) — arman el followup,
     resetean `finish_reason` a `"unknown"`, y siguen el loop (`continue`
     en vez de `return`) en vez de terminar el turno ahí. La llamada al
     modelo siguiente sale con `transient_messages`+`tools_override`
     puestos; ambos se resetean a `None` apenas se consumen (una sola
     llamada).
   - Si el modelo llama `ask_user` en ese followup, no hizo falta NADA
     nuevo — cae directo en el mecanismo de pausa `input_needed` que ya
     existía, incluido en `examples/agent_local_openai.py`'s loop de
     pausa (`response.needs_input`) sin tocar ese archivo — confirmado
     con el flujo completo, no solo en teoría.

**Verificación real, sin pytest**: script con `ReactLoop`/
`ToolDispatcher`/`AskUserTool`/`ToolRegistry` reales —
(1) tool `ASK_APPROVED` rechazada → resume → followup con toolset
`{ask_user, update_plan}` confirmado (inspeccionando qué tools recibió
el cliente en esa llamada exacta) → el modelo llama `ask_user` →
`finish_reason == "input_needed"`, la tool denegada nunca ejecuta
(`send_email.calls == 0`), exactamente 2 llamadas al modelo. (2) el
usuario responde la pregunta → resume → toolset completo restaurado
(`send_email` visible de nuevo) → respuesta final normal, 3 llamadas
totales. (3) regresión: sin `ask_user` registrado, cae al pause mudo de
antes — 1 sola llamada, `finish_reason == "tool_denied"`, igual que la
verificación del 21. Re-corridas todas las verificaciones previas
(guards/gate, `test_local_registries.py`) — mismos resultados, 0
regresión. `py_compile` OK.

**No tocado**: `examples/agent_local_openai.py` — su loop de pausa ya
maneja `needs_input` de antes, así que el flujo nuevo lo atraviesa sin
ningún cambio de código ahí. Si en algún momento se usa un agente sin
`ask_user` registrado, ese script SÍ volvería a mostrar el síntoma
original (nada nuevo que mostrar) — pero `Agent.__init__` registra
`ask_user` por default siempre que `enable_human_input=True` (el
default), así que no aplica al caso real de este ejemplo.

## 2026-09-22 — CLI acercado al flujo visual de Toad

`max_ai/cli/app.py` evolucionó de chat simple a layout de trabajo: árbol de
archivos del workspace, transcript central, Markdown streaming en un panel
dedicado y editor multilínea Markdown. `Ctrl+Enter` envía, Enter conserva
nueva línea, `!comando` ejecuta shell en el workspace y `@ruta` adjunta el
contenido de un archivo legible dentro del prompt. Se conservaron eventos de
plan/tools, aprobaciones y preguntas del agente.

Verificación: montaje real con Textual `App.run_test()`, presencia de árbol,
editor y panel Markdown, expansión real de `@README.md`, `py_compile` y
`git diff --check` OK.

## 2026-09-22 — `agent_local_openai.py` conectado al Textual CLI

El ejemplo ahora solo construye el cliente OpenAI, las tools y un `Agent`.
Dentro del contexto async del agente llama a `await run_repl(agent)`, dejando
en `max_ai/cli` el streaming, el contexto de conversación, aprobaciones,
preguntas, planes y renderizado. Se eliminaron el REPL manual y el renderer
duplicado del ejemplo; el cierre del cliente OpenAI se conserva en `finally`.

Verificación: import del módulo, import de `MaxAIApp`/`run_repl`, `py_compile`
y `git diff --check` OK. No se hizo una llamada real a OpenAI.

---

## 2026-09-22 (continuación) — Memory/knowledge/skills conectados en
`examples/agent_local_openai.py`, fixtures reestructurados

A pedido del usuario, se cableó el ejemplo para probar el `Agent` con las
tres capacidades juntas, usando los fixtures de `examples/local/` creados
antes en la sesión.

**Fixtures reestructurados** (antes flat, `memory.json`/`knowledge.json`
sueltos — quedaron anotados como pendiente de resolver "cuando
compongamos el ejemplo"): movidos a la estructura anidada que exigen los
backends reales:
- `examples/local/memory/user_001/demo.json` (antes `memory.json`) —
  `LocalMemoryRegistry` espera `{base_path}/memory/{user_id}/{session_id}.json`.
- `examples/local/knowledge/framework.json` (antes `knowledge.json`) —
  `LocalKnowledgeRegistry` espera `{base_path}/knowledge/{name}.json`,
  nombre de fuente `"framework"`.

**`cli/app.py`** (el nuevo TUI Textual, construido hoy por otra sesión —
no tocado en su diseño, solo una adición chica): `MaxAIApp.__init__`/
`run_repl` ganaron `initial_context: RunContext | None = None`
(default `None`, compatible con todo lo existente). Hacía falta: sin
esto, `self.context` arranca en `None` y el primer turno genera un
`session_id` aleatorio — nunca coincidiría con la ruta fija del fixture
de memoria (`.../user_001/demo.json`). `examples/agent_local_openai.py`
ahora pasa `RunContext(user_id="user_001", session_id="demo")` — mismos
`user_id`/`session_id` que usa `LocalMemoryRegistry`.

**`examples/agent_local_openai.py`**: agregado `LocalMemoryRegistry`,
`LocalKnowledgeRegistry` (fuente `"framework"`, descripción explicando
que es "hechos sobre cómo funciona el framework max_ai mismo"), y
`LocalSkillRegistry` apuntando a `examples/LocalSkills/` (`create-report`,
`create-ppt` — reusado tal cual, sin crear nada nuevo, a pedido explícito
del usuario en la entrada anterior). Los tres pasados al `Agent(...)`
(`memory=`, `knowledge=`, `skills=`). Imports reordenados con
`ruff check --select I --fix` (config de ruff agregada hoy).

**Verificación real, sin pytest**: `Agent` real construido con las 3
capacidades — `agent.memory is not None`, `agent.knowledge` con 1 fuente,
`agent.skills is not None`, los tres `True`. `memory.get_context()`
devuelve las 2 categorías del fixture. `knowledge[0].search("how does
the completion gate work")` devuelve `completion_gate` como resultado
top (embedding real, no mock). `skills.prepare()` + `get_skills()` carga
`create-report`/`create-ppt`. `MaxAIApp(agent, initial_context=...)`
confirma `self.context` queda seteado; sin el kwarg, sigue en `None`
(compatibilidad hacia atrás confirmada). `py_compile` OK en los 3
archivos tocados.

---

## 2026-09-22 (continuación) — TUI: eventos compactos, tokens debajo del
chat, composer más chico

A pedido del usuario, tres ajustes de UX sobre `cli/app.py`/`cli/events.py`
(el TUI Textual, construido hoy por otra sesión — se editó, no se
rediseñó).

**1. Eventos en una línea, no paneles con JSON**: `cli/events.py` tenía
`event_panel(event) -> Panel` — un `rich.Panel` con borde por evento,
cuerpo con el `model_dump()` completo pretty-printed. Reemplazado por
`event_line(event) -> Text` — una sola línea `"{ícono} {label corto}"`,
sin bordes ni JSON. Casos especiales cortos: `tool_call` → "▶ {tool}
running"; `tool_call_response` → "✓ done (Nms)" / "✗ failed (Nms):
{error corto}" (sin nombre de tool — no está en ese evento, el
`tool_call` de la línea de arriba ya lo dijo, repetirlo era ruido);
`task_complete`/`completion_rejected` → "✓ gate completed" / "↻ gate
retry — {razón corta}"; `reasoning_iteration`/`reasoning_complete` →
"· iteration N/M" / "■ turn ended · {finish_reason}"; fallback genérico
para el resto. Duración calculada de `started_at`/`completed_at` del
propio `ToolResult` (no es un campo serializado, es una `@property` —
se recalcula a mano). `app.py` actualizado (`event_panel` → `event_line`,
import y call site).

**2. Tokens movidos de la sidebar a debajo del chat**: `Static(id="usage")`
vivía en `#sidebar`, entre el status del agente y el árbol de archivos.
Movido a `#main`, como último hijo, justo debajo de `#composer`. Mensaje
condensado de 5 líneas ("Session tokens\nInput: ...\nOutput: ...\n...")
a una sola: `"tokens · in N · out N · cached N · total N"`.

**3. Composer más chico**: `#composer` de `height: 10` a `5`, `#prompt`
de `height: 7` a `3` (confirmado con `run_test()`: el área de contenido
real del prompt quedó en 3 filas). Placeholder acortado ("Describe a
task…  @file.md  !pytest -q" → "Task…  @file.md  !cmd") y el hint de
teclas igual, más corto.

**Verificación real, sin pytest**: `event_line()` llamado sobre instancias
reales de `ToolCallEvent`/`ToolCallResponseEvent` (éxito y fallo, con
`started_at`/`completed_at` reales)/`TaskCompleteEvent`/
`CompletionRejectedEvent`/`ReasoningIterationEvent`/`ReasoningCompleteEvent`/
`PlanningEvent`/`UserInputRequestEvent` — confirmado el texto exacto de
cada línea. App Textual montada de verdad con `MaxAIApp(...).run_test()`
(headless, sin mocks): `#usage` confirmado como hijo de `#main` y NO de
`#sidebar`; tamaños reales de `#composer`/`#prompt` confirmados en 3 de
alto; `_write_event()` llamado con eventos reales escribe al transcript
sin error. `py_compile` OK en ambos archivos.

---

## 2026-09-22 (continuación) — Bug reportado por el usuario: un skill que
falla no se reporta, el gate lo deja cerrar igual

**No arreglado todavía — pendiente para mañana.**

**Reporte**: el usuario corrió un skill; el `bash` interno que ejecutaba
falló, pero el turno cerró igual (`CompletionGate` lo dejó pasar) y el
LLM nunca mencionó que hubo un error en su respuesta final. Recién al
preguntarle directamente, el modelo admitió que había fallado — pero no
lo había dicho antes ("lo pasé por alto").

**Causa raíz, confirmada leyendo el código actual**
(`capabilities/completion_gate/gate.py::RuntimeCompletionGate.
on_final_response`): es un comportamiento **a propósito**, decisión de
una sesión anterior documentada en el propio docstring de la clase —
"Both done and failed steps are terminal" / "Failed commands alone do
not prevent closing a turn". Dos consecuencias concretas de eso:
1. Los `expected_outputs` declarados en una llamada a `bash` **solo se
   verifican si `result.success and exit_code == 0`** (líneas ~62-69) —
   si el comando falló, esa lista de deliverables nunca se agrega al set
   a chequear. Osea: hoy, un `bash` fallido con `expected_outputs`
   declarados no se verifica en absoluto, ni para bien ni para mal.
2. Nada obliga al modelo a mencionar el fallo en su respuesta final —
   puede cerrar el turno con texto que no diga nada sobre el error, y el
   gate no tiene forma de saber si el texto "cubre" o no el fallo
   ocurrido.

**Dirección propuesta por el usuario para mañana**: si un `bash` (dentro
de un skill o no) declaró `expected_outputs`, el runtime tiene que
verificar que existan/coincidan — sin importar el `exit_code`. Puntos a
decidir al implementarlo:
- ¿Verificar `expected_outputs` siempre (no solo en éxito), y que un
  mismatch bloquee el cierre igual que hoy lo hace un archivo faltante
  en el caso de éxito?
- ¿Hace falta además forzar que el modelo *mencione* el fallo en el
  texto final (no solo que el archivo exista), o alcanza con que el
  gate rechace el cierre cuando el deliverable no está?
- Relacionado con el patrón de `tool_denied`→`ask_user` que se armó hoy
  mismo: un fallo silencioso de skill podría beneficiarse del mismo
  enfoque (un followup que obligue a reconocer el problema) en vez de
  solo bloquear con un mensaje genérico de gate.

---

## 2026-09-23 — Arreglado: los bash fallidos ya no pasan en silencio

Resuelve el pendiente de la entrada anterior. Diseño acordado con el
usuario antes de implementar.

**`capabilities/completion_gate/gate.py` (`RuntimeCompletionGate`)**:
- Un bash **falló** si corrió y salió con `exit_code != 0`, o si falló con
  `EXECUTION_ERROR`/`TIMEOUT`. No cuentan los denegados (esos ya pasan por
  `ask_user`), los cancelados ni los de parámetros inválidos (nunca corrieron).
- Un fallo está **resuelto** si existen los `expected_outputs` que declaró
  (el modelo lo arregló de otra forma), o si el mismo comando exacto después
  terminó con exit 0 (lo reintentó y salió).
- Un fallo **sin resolver** bloquea el primer intento de cierre una sola vez:
  `incomplete` con comando, exit code y stderr cortado, más "fix it, or tell
  the user it failed". El id queda en `ctx.completion_state[...]
  ["failures_nudged"]` y no vuelve a bloquear (evita el loop hasta
  `max_iterations` cuando el fallo no tiene arreglo).
- Si el turno cierra con fallos sin resolver, la decisión es `completed` con
  `reasons` = "Closed with unresolved failure: …". Esas razones llegan a
  `TaskCompleteEvent` y a `AgentResponse.completion`, así que el usuario se
  entera aunque el texto del modelo no lo mencione.
- Sin cambios: un bash con exit 0 cuyo `expected_output` no existe sigue
  bloqueando siempre. Chequeo de archivo extraído a `_missing_output()`.

**`core/events_bus/bus.py` (`check_final_response`)**: antes descartaba las
`reasons` de toda decisión `completed`. Ahora las guarda como notas y las
devuelve si el resultado final es `completed`. Si algún handler bloquea,
devuelve solo las razones que bloquean, igual que antes.

**`cli/events.py`**: `task_complete` con notas se muestra "⚠ gate completed ·
{notas}" en amarillo. Sin notas sigue siendo "✓ gate completed".

**Verificación real, sin pytest** (`Agent` real, `BashTool` real,
`LocalWorkspace` en directorio temporal, cliente scripted):
1. `exit 3` y el modelo dice "Done!" sin mencionar el fallo: el gate lo manda
   de vuelta una vez con "failed (exit 3)". En el segundo "Done!" cierra con
   la nota, que llega a `TaskCompleteEvent` y a `response.completion`.
   3 llamadas al modelo.
2. `false` con `expected_outputs=["out.txt"]` y después otro comando que crea
   `out.txt`: queda resuelto, sin aviso y sin nota.
3. El mismo comando falla y en el reintento sale con exit 0: queda resuelto,
   sin aviso y sin nota.

Regresiones re-corridas (guards/gate, `tool_denied`, followup de `ask_user`,
`test_memory_contract.py`, `test_local_registries.py`): mismos resultados,
solo las 3 fallas viejas ya documentadas. `ruff` y `py_compile` OK.

---

## 2026-09-23 — Nuevo cliente LLM: OpenRouter (para probar con modelos free)

A pedido del usuario: un cliente para OpenRouter, así el agente se puede
probar continuamente con modelos `:free` sin gastar en OpenAI.

**Diseño**: OpenRouter habla el protocolo OpenAI Chat Completions, así que
`clients/openrouter/client.py::OpenRouterChatCompletionClient` hereda de
`OpenAIChatCompletionClient` y solo sobreescribe lo que difiere (verificado
con llamadas reales antes de escribir código):
- **Razonamiento**: llega en `message.reasoning` / `delta.reasoning` → se
  expone como `thinking` (en el mensaje y como chunks en streaming).
- **Routing** (`fallback_models` → `models`, `reasoning`, `provider`): no son
  params del SDK de OpenAI, viajan en `extra_body`. `fallback_models` es lo
  que hace usables los free: si el principal está rate-limited upstream,
  OpenRouter prueba el siguiente en el mismo request.
- **Errores en el body**: OpenRouter puede responder 200 (o a mitad de un
  stream) con un objeto `error` ("Upstream error from Nvidia: Service
  temporarily overloaded"). Se convierten en `ClientError` (`rate_limit` si
  code 429, si no `api_error`) — ambos ya son transitorios para el loop, que
  reintenta con backoff; el cliente no reintenta por su cuenta.
- **API key**: `api_key` o `$OPENROUTER_API_KEY`; si falta, `ValueError`.
  Nunca cae a `OPENAI_API_KEY` (el SDK lo haría y mandaría la key de OpenAI
  a OpenRouter).
- Headers de atribución opcionales `app_name`/`app_url` (`X-Title`/`HTTP-Referer`).

**Cambios en el cliente OpenAI** (mínimos, para que la subclase no duplique
80 líneas de streaming): hook `_extract_thinking()` (devuelve `None` en
OpenAI), usado en `_parse_response` e `_iter_chunks`; `PROVIDER_NAME` en vez
del literal `"OpenAI"` en `_map_openai_error` (ahora classmethod).

**Otros**: `core/models.py::OpenRouterChatCompletionClientConfig`
(serialización), `base/component.py::KNOWN_PROVIDERS` con las dos claves de
OpenRouter, export en `max_ai/clients/__init__.py`.

**Ejemplo**: `examples/agent_local_openrouter.py` (TUI, mismas tools/memory/
knowledge/skills). Para no duplicar, `examples/agent_local_openai.py` ahora
tiene `run_demo(client, provider)` compartido; su `main()` hace lo mismo que
antes.

**Verificación (llamadas reales a OpenRouter, sin mocks)**:
- Cliente: tool call con thinking; streaming con thinking + contenido + usage
  final (un "overloaded" a mitad del stream fue detectado y reintentado);
  fallback qwen (rate-limited) → servido por nemotron; modelo inválido →
  `invalid_request`; round-trip `dump_component`/`load_component`; key
  faltante rechazada.
- `Agent` + `ReactLoop` reales: llamó `get_weather`, gate `completed`,
  respuesta correcta.
- Config completa del ejemplo (TUI reemplazada por un turno headless): usó
  `search_framework` del knowledge y respondió con su contenido.
- `pytest tests/test_config.py tests/test_ollama_*.py`: 27 passed.
  `tests/test_capability_registry.py` no colecta (importa `max_ai.tools`,
  que no existe desde la reorganización) — preexistente, no relacionado.

**Nota**: los free cambian seguido y se saturan; el ejemplo usa
`nvidia/nemotron-3-super-120b-a12b:free` con fallback a
`qwen/qwen3.8-27b:free` y `google/gemma-4-31b-it:free` (todos con tools).
No se implementó el reenvío de `reasoning_details` entre turnos (lo exigen
algunos modelos pagos como Anthropic/Gemini vía OpenRouter; los free
probados no lo necesitan).

**Actualización (mismo día) — ejemplos numerados**: a pedido del usuario,
los ejemplos quedaron como `examples/01_agent_with_openai.py` y
`examples/02_agent_with_openrouter.py` (se corren con
`.venv/bin/python -m examples.01_agent_with_openai` / `...02_agent_with_openrouter`).
Lo compartido (tools, memory, knowledge, skills, apertura de la TUI) vive en
`examples/cli_agent.py::run_agent_in_cli(client, provider)` (antes
`run_demo`) — en un módulo propio porque un nombre que empieza con dígito no
se puede importar. `examples/agent_local_openai.py` y
`examples/agent_local_openrouter.py` borrados. Verificado: el 02 corrió un
turno real headless (un "Transient LLM error" del free, reintentado por el
loop, luego `search_framework` y cierre `completed`); el 01 importa bien.

---

## 2026-09-23 — CLI estilo Claude Code (thinking plegable y opcional)

A pedido del usuario ("mejorá el CLI tipo Claude Code, el thinking opcional
y en un cuadrito que se pueda compactar, sorprendeme").

**Cambio de fondo**: el transcript dejó de ser un `RichLog` (texto plano,
inmutable) y pasó a ser un `VerticalScroll` de widgets (`cli/blocks.py`),
para que cada pieza del turno se pueda plegar y actualizar en su lugar:
- `ThinkingBlock`: caja que streamea el razonamiento y al terminar se
  pliega a una línea (`✻ Thought for 3s · 120 words ▸ expand`), click para
  expandir. `show_thinking=False` / `ctrl+t` / `/thinking` la ocultan (los
  pensamientos se siguen guardando, reaparecen al reactivar).
- `ToolBlock`: `● name(argumento principal)` + `⎿ resumen`, spinner mientras
  corre, verde/rojo al terminar (rojo también con exit code ≠ 0), click
  para ver la salida. Un call pausado por aprobación se re-emite al
  reanudar: se reusa el mismo bloque por `tool_call_id` (bug encontrado en
  la prueba real) y muestra "Waiting for approval…" mientras espera.
- `AssistantBlock` (Markdown vía `MarkdownStream`, no re-parsea todo en cada
  chunk), `UserBlock`, `NoteLine`, `WelcomeBox`.

**Resto**: línea de estado con spinner y verbos ("✻ Brewing… (12s · ↓ 340
tokens · esc to interrupt)"); `esc` interrumpe vía `CancellationToken` y
restaura un snapshot (`model_copy(deep=True)`) del `RunContext` previo al
turno — sin eso el contexto quedaba con tool calls colgados; comandos `/help
/clear /thinking /verbose /files /exit` con sugerencias al tipear `/`
(`/clear` = conversación nueva con los mismos `user_id`/`session_id`);
`ctrl+o` modo verbose (eventos runtime vía `event_line`, que por defecto se
ocultan salvo gate retry/errores/compaction); sidebar de archivos oculto por
defecto (`ctrl+b`); el transcript no toma foco (click en un bloque ya no le
roba el foco al prompt — bug encontrado en la prueba headless), scroll con
rueda o `pgup/pgdn`; cuadro de aprobación/pregunta con opciones numeradas.

**Verificación**: headless (`App.run_test`) con `Agent` real + cliente
scripted streaming: turno completo, plegado/expandido por click, `ctrl+t`,
sugerencias `/`, `/help`, `esc` con rollback exacto del contexto, turno
posterior OK, `/clear`. Headless con modelo real de OpenRouter (nemotron
free): thinking real plegado, cuadro de aprobación, un solo bloque por tool,
respuesta final. Capturas SVG revisadas como texto.
`tests/cli/test_textual_interaction.py` actualizado a la API nueva (estaba
roto: importaba `event_panel`): 5 passed. `tests/cli/test_repl.py` sigue sin
colectar — prueba el REPL viejo (`max_ai.cli.repl`, ya no existe),
preexistente.

**Actualización (mismo día) — plan/ask_user plegables, menú `/`, línea única de cierre**:
- `cli/blocks.py`: base `FoldBlock` (cuadro plegable con click);
  `ThinkingBlock`, `PlanBlock` (`update_plan` → checklist, plegado a
  `☰ Plan 2/5 · paso actual`; updates consecutivos refrescan el mismo cuadro,
  los anteriores se pliegan) y `AskUserBlock` (`? pregunta → respuesta`,
  plegado mientras espera para no duplicar las opciones del cuadro de
  respuesta; `ask_user` guarda la respuesta como `"label — description"`,
  se compara contemplando eso). `update_plan`/`ask_user` ya no salen como
  `ToolBlock` genérico. El panel fijo `#plan` arriba se quitó.
- Menú `/` (`widgets.py::CommandMenu`, nunca toma foco): ↑↓ mueve, tab
  completa, enter ejecuta, esc cierra. Incluye `/skills`, `/tools` y cada
  skill cargada como `/<nombre> [args]` (manda `Use the "<nombre>" skill.
  <args>` como turno). Nota: un binding de la subclase sobre la misma tecla
  reemplaza al de `TextArea`, así que `PromptEditor` hace él mismo el
  cursor up/down/tab cuando el menú está cerrado.
- `Agent.tools` (property pública, solo lectura) para listar tools sin
  tocar `_registry`.
- Quitado el indicador "✻ thinking" de la barra inferior (no hacía nada).
- Fin de turno en UNA línea construida desde `AgentResponse`
  (`finish_reason` + `completion`) más los `completion_rejected` del turno:
  `✓ Done in 16s · stop` / `⚠ Done … · gate retried 1×: … · gate: …` /
  `■ Stopped after … · max_iterations`. `task_complete`,
  `reasoning_complete` y `completion_rejected` ya no generan líneas propias.
- Verificado headless (4 escenarios de la línea de cierre, plan+ask_user
  plegables, menú con flechas/tab/enter/esc, invocación de skill, `/tools`)
  + turno real contra OpenRouter + `tests/cli/test_textual_interaction.py`
  (5 passed).

**Fix (mismo día) — crash `NoMatches: '.title' on ToolBlock()`** reportado por
el usuario al pedir "crear un file para scrapear una página": el
`ToolBlock` se registraba en `_tools` antes de terminar de montarse y el
spinner (`_tick`, cada 0.12s) le pegaba en ese instante. Arreglo en el
bloque: el estado vive en el propio widget y `_redraw()` pinta desde ahí
solo si los hijos existen; `on_mount` pinta el estado acumulado (sirve
también si el resultado llega antes del mount). Mismo patrón en
`FoldBlock.refresh_block`/`ThinkingBlock.append`. Además, el bloque se
registra en `_tools` después de `await mount`. De paso aparecieron dos
choques de nombres con `Widget` de Textual (`_render` — lo usa el
compositor, pisarlo rompe el dibujado — y `expand`, un atributo): renombrados
a `_redraw` y `set_expanded`. Verificado: prueba que actualiza bloques antes
y durante el mount; el pedido real del usuario con el ejemplo 02 contra
OpenRouter (write_file + bash aprobado, `✓ Done in 29s · stop`); el resto de
checks de la CLI y `tests/cli/test_textual_interaction.py` (5 passed).

**Actualización (mismo día) — identidad visual propia de MaxAI**: a pedido del
usuario ("parece una copia de Claude Code"), la CLI pasó a acento lila
(`#b794f6`, constantes `ACCENT/BORDER/BORDER_HOVER/USER_BG` en
`cli/blocks.py`), bordes/fondos con tinte violeta, y glifos propios: marca
`◆ M A X · A I`, spinner de diamantes `◇◈◆◈`, thinking `◈`, respuesta `✦`,
tool `◆ name(arg)` + `╰─ resumen`, plan `◇`/`▶`, prompt `❯`, mensaje del
usuario con barra lila a la izquierda. La línea de estado dice siempre
"Thinking…" (se quitaron los verbos al azar). README de la CLI actualizado.
Checks headless + `tests/cli/test_textual_interaction.py` (5 passed) OK.

**Ajuste (mismo día)**: el usuario aclaró que no quería todo en lila, solo
que no fuera el naranja de Claude. Texto normal, fondos y bordes en reposo
volvieron a grises neutros; el lila (`ACCENT`) queda solo en acentos:
glifos (`◆ ◈ ✦ ❯ ▶ ◇`), spinner, borde del prompt con foco, borde al
pasar el mouse sobre un cuadro, hover de opciones, nombres de comandos,
barra del mensaje del usuario y la marca `M A X · A I`. Checks + tests OK.

**Actualización (mismo día) — `ask_user` se responde dentro de su cuadro + 2 fixes**:
- El usuario mostró una captura: 3 `ask_user` seguidos quedaban como
  "waiting for your answer…" separados por huecos, más un cuadro aparte
  "Question 1 of 3" que repetía las opciones. Ahora `AskUserBlock` tiene
  tres estados: en cola (una línea, "queued"), activo (borde lila, opciones
  como `ChoiceRow`: ↑↓ + enter, click, número o texto libre en el prompt) y
  respondido (plegado `? pregunta → respuesta`). `_ask_inline` reemplaza al
  cuadro para las preguntas que tienen bloque; el cuadro queda como respaldo
  (p.ej. al retomar un contexto guardado sin el evento de la tool).
- Los huecos: el modelo streamea `"\n\n"` antes de los tool calls y eso
  montaba un `AssistantBlock` vacío. Ya no se crea un bloque de respuesta
  hasta que llega texto no vacío.
- Crash al cerrar la app a mitad de turno (`NoMatches '#status'` desde
  `_tick`): `_refresh_status`/`_refresh_usage` buscan los widgets con
  `_static()`, que devuelve `None` si la pantalla ya no existe.
- Los `ErrorEvent`/`FatalErrorEvent` se mostraban como "• error" sin texto:
  `event_line` ahora muestra `✗ tipo: mensaje`.
- Verificado headless: el caso exacto del usuario (3 preguntas + `"\n\n"`,
  respondidas con ↓+enter, texto libre y click), salir a mitad de turno, y
  el resto de checks + `tests/cli/test_textual_interaction.py` (5 passed).
  **No se pudo verificar contra OpenRouter real**: la key es free tier y
  devuelve "Rate limit exceeded" en todos los modelos free tras 4 intentos
  (cuota diaria de requests free agotada por las pruebas del día).

**Actualización (mismo día) — preguntas múltiples en un solo formulario**:
a pedido del usuario ("como Claude Code, que te deja verlas todas en un solo
viaje"). `AskUserBlock`/`ChoiceRow` reemplazados por
`cli/blocks.py::QuestionForm`: al pausar por `input_needed`, TODAS las
preguntas pendientes van en un cuadro con pestañas (`☐/☒` + título corto sin
palabras de relleno), `←→` cambia de pregunta (solo con el prompt vacío),
`↑↓` + enter elige y avanza a la siguiente sin responder, click en una opción
también, y lo que se escriba en el prompt es respuesta libre (o número de
opción). Con varias preguntas hay una pestaña **Submit** que muestra el
resumen y envía todo junto; con una sola, enter responde y envía. Al
terminar se pliega a `? Answered N questions ▸ expand`. Los params de
`ask_user` se guardan por `tool_call_id` al ver el `ToolCallEvent` (labels y
descripciones); lo que se devuelve es siempre el string de opción del record.
El cuadro de respuesta aparte ya no se usa para preguntas (sí para
aprobaciones). `PromptEditor` ganó `tabs_open` + `←→` (con el mismo patrón
de "si no aplica, mover el cursor").
Verificado headless: el caso de la captura (3 preguntas + `"\n\n"`):
↓+enter, texto libre, ←→, click en opción → pestaña Submit → enter envía;
las 3 respuestas llegan como ToolMessage correctas; una sola pregunta sin
Submit; flechas normales después. Resto de checks + tests (5 passed) OK.

**Actualización (mismo día) — `ask_user` con varias preguntas en UNA llamada + opción "Other"**:
- Problema reportado: el modelo (free) a veces preguntaba de a una y le
  explicaba al usuario "ask_user solo permite una pregunta por llamada". La
  descripción ya pedía llamadas paralelas, pero el contrato era 1 pregunta
  por llamada. Cambio de contrato (como `AskUserQuestion` de Claude Code):
  `ask_user` recibe `questions: [{question, header, options?}]` (1-4);
  `header` = etiqueta corta de pestaña; `options` 2-4 `{label, description}`,
  con la indicación de NO agregar "Other" (la UI siempre lo ofrece).
- `capabilities/tools/ask_user/_tool.py::pending_questions(params)` normaliza
  la llamada a `[{question, header, options(str "label — description")}]`,
  aceptando también el formato viejo (`question`/`options`).
  `AskUserTool.validate_parameters` reescribe records viejos al formato
  nuevo, así contextos pausados antes del cambio siguen validando al
  reanudar.
- `ToolCallRecord`: nuevos `input_questions` (todas las preguntas) y
  `user_answers` ({pregunta: respuesta}); `input_question/input_options`
  reflejan la primera (compatibilidad con consumidores de una sola, p.ej.
  `ui/server.py`, que sigue mostrando solo la primera — pendiente si se
  quiere la UI web). `apply_user_answer` (record y `ToolState`) acepta `str`
  o `dict`. El dispatcher devuelve `{"answers": {...}}` para multi y el
  formato de siempre para una. `base/tool_executor.py` (legacy) solo ganó un
  fallback de 2 líneas para no mostrar una pregunta vacía.
- CLI: `QuestionForm` arma una pestaña por pregunta (de una o varias
  llamadas; título = `header` si viene), fila explícita "✎ Other — type your
  own answer" al final de cada pregunta (al elegirla, el prompt pide la
  respuesta; en preguntas sin opciones es la única fila), y `result()` agrupa
  por record (dict si el record tenía varias preguntas).
- Verificado headless: una llamada con 3 preguntas (headers, Other + texto,
  click, pregunta sin opciones) → el modelo recibe exactamente
  `{"answers": {...}}`; varias llamadas con el formato viejo siguen en un
  solo formulario; todos los checks previos (CLI, tool_denied, guards/gate,
  bash fallido) + `tests/cli/test_textual_interaction.py` (5 passed).
  `tests/reasoning/test_human_in_loop.py` y `tests/test_agent_run.py` ya no
  colectaban/fallaban antes de este cambio (`CoreExecutor` inexistente; 7/8
  fallando) — no se usaron como señal. No verificado contra un modelo real
  (OpenRouter free sigue en rate limit).

**Fix (mismo día) — esc durante una pausa + "sombra" en el prompt**:
- Esc no hacía nada mientras el turno estaba pausado en un formulario de
  preguntas o una aprobación (`action_interrupt` lo ignoraba con
  `_waiting`). Ahora también cancela ahí: `CancellationToken.cancel()` +
  `cancel()` del future que espera la respuesta; `_run_turn` lo trata como
  cualquier interrupt (restaura el snapshot previo al turno → sin records
  pendientes, el siguiente mensaje funciona). El formulario queda como
  `? N questions · cancelled`; la tool que esperaba aprobación como
  `◇ tool(...) ╰─ cancelled` (gris, ya no "Error: interrupted" en rojo). La
  línea de estado dice "esc to cancel the turn" y el formulario "esc cancel".
- La franja clara en el prompt era el resaltado de línea del cursor de
  `TextArea`: `PromptEditor(..., highlight_cursor_line=False)`.
- Verificado headless: esc en formulario y en aprobación → rollback exacto,
  sin records pendientes, turno siguiente OK; resto de checks + tests OK.

## 2026-09-23 — Compactación, paso 0: un solo contador de tokens

Antes de diseñar el contrato de compactación, se unificó cómo se cuentan
los tokens (pedido de marvin).

- `core/compaction.py` → paquete `core/compaction/` (`models.py` +
  `token_counter.py`; `__init__` reexporta todo). El paquete vacío recién
  creado tapaba al archivo y `import max_ai.agents` fallaba.
- `TokenCounter` se mudó de `base/compaction.py` a
  `core/compaction/token_counter.py`. Imports actualizados en
  `base/reasoning.py`, `compaction/sliding_window.py`, `ui/server.py` y los
  tests de compactación. Encoders cacheados (`lru_cache`) y
  `default_counter()` compartido.
- `setting.default_tokenizer` (`DEFAULT_TOKENIZER` en `.env`, por defecto
  `o200k_base`). Lo usan `TokenCounter`, su fallback y
  `ModelConfig.tokenizer_base` (vía `default_factory`, no al importar).
- `CoreMessage.with_token_count(counter=None)` ahora delega en
  `TokenCounter.count_message` (antes `len/4` y solo texto: los tool calls
  contaban 0 y `count_message` reusaba ese número bajo). Import perezoso
  para evitar el ciclo messages ↔ compaction.

Verificado: user msg 12 = 12 y assistant con tool call 114 = 114 en ambos
caminos; `DEFAULT_TOKENIZER=cl100k_base` se respeta; nombre inválido cae a
`o200k_base`. `tests/compaction` + CLI: 18 passed; scripts CLI y
`verify_ctx_roundtrip` OK. El resto del suite tiene 79 fallos + 21 errores de
colección previos (tests de APIs viejas: `CoreExecutor`, `conversation_dir`,
`Agent(store=…)`…), ninguno relacionado con tokens o compactación.

## 2026-09-23 — Limpieza: fuera routines; embeddings, observation y modelos a core (opción B)

- **Routines eliminado por completo** (ya no se usaba): `base/routines.py`,
  `types/routines.py`, `legacy/routines/`, `RoutineBlocks` (`core/blocks.py`,
  `core/__init__`), el slot `routines`/`has_routines` de
  `manager/capabilities.py`, `"routines"` en `ComponentType`,
  `RoutineRegistryError` (sin uso), exports de `base/__init__`, los 2 tests de
  routines en `test_local_registries.py`, las partes de routines en
  `test_capability_registry.py` y `examples/file.py`. Copia de respaldo en el
  scratchpad (`backup_routines/`).
- **Layout elegido (B)**: los modelos de cada módulo del core viven en
  `core/model/<módulo>.py`; cada módulo tiene su paquete que los reexporta.
  - `core/model/compaction.py` (antes `core/compaction/models.py`)
  - `core/model/observation.py` (antes `base/observation.py`) +
    `core/observation/__init__.py`
  - `core/embeddings/lightweight.py` (antes `base/embeddings.py`) +
    `core/embeddings/__init__.py`
  - `core/model/__init__.py` no reexporta nada a propósito (evita ciclos).
- Imports actualizados en capabilities (knowledge/context/memory),
  `base/context.py`, `base/__init__.py` y `tests/test_embeddings.py`.

Verificado: suite idéntico a la foto previa (277 passed / 79 failed / 36
errores, todos preexistentes), salvo los 2 tests de routines eliminados.
Scripts CLI + `verify_ctx_roundtrip` OK.
Pendiente: `tests/test_core_agent_smoke.py` aún menciona routines, pero ya no
cargaba antes (importa `stacks.priority_tools_layer`, que no existe).
`core/models.py` (configs de agente/stack/clientes) sigue fuera de
`core/model/`.

## 2026-09-23 — Core ordenado: configs a `core/model/`, plugins a `capabilities/`

**`core/models.py` eliminado**, repartido según quién es dueño de cada modelo:
- `core/model/llm.py` → `ModelConfig`
- `core/model/stacks.py` → `StackConfig`
- `core/model/agent.py` → `AgentConfig`, `AgentComponentConfig`
- configs de cada cliente junto a su cliente, como ya exige
  `test_provider_models` para capabilities: `clients/<x>/_model.py`
  (`OpenAIChatCompletionClientConfig` + `OpenAIReasoningEffort`,
  `OllamaChatCompletionClientConfig` + `OllamaThink`,
  `OpenRouterChatCompletionClientConfig`); cada paquete exporta su Config.

**Lo intercambiable pasa a `capabilities/`** (pedido de marvin):
`clients/`, `stacks/` (con `prompts/*.j2`), `reasoning/` y los middlewares
concretos (`console_trace`, `logging`). `MiddlewareChain` no es un plugin
(es lo que los ejecuta, lo usa `base/reasoning.py`) → `core/middleware/`.
- Script `scratchpad/move_plugins.py`: movió los archivos y recalculó cada
  import relativo/absoluto (58 archivos), incluidas las rutas de
  `KNOWN_PROVIDERS`. `base/layer.py` apunta a
  `capabilities/stacks/prompts`. Los alias `maxai.llm.*` / `maxai.stacks.*`
  no cambian, así que los configs serializados siguen cargando.
- Respaldo: `scratchpad/backup_plugins/`, `backup_moves/core_models.py`.

Verificado: suite idéntico test por test (284 passed / 73 failed / 36
errores, todos preexistentes); round-trip de un cliente OpenRouter por
`dump_component`/`load_component`; 13 scripts (CLI, ctx round-trip, guards,
gate, tool denied) OK; ejemplos `01`, `02`, `cli_agent` importan.
Preexistente, sin tocar: 12 módulos viejos no importan (`CoreExecutor`,
`Workspace`, `EmbedMany`) y hay un ciclo `base/context.py` ↔
`function_as_tool` que solo falla al importar `function_as_tool` primero.

## 2026-09-23 — `termination/` → `core/termination/`

La cancelación (`CancellationToken`) es runtime, no un plugin: la usan el
loop, el dispatcher, las tools y la CLI. Movida con el mismo script
(`scratchpad/move_termination.py`, 30 archivos reescritos; respaldo en
`backup_plugins/termination`). Ahora se importa como
`from max_ai.core.termination import CancellationToken`.
Verificado: suite idéntico (284/73/36); `verify_esc_waiting` (esc con
rollback), `verify_exit_midturn`, CLI, ctx round-trip, guards y tool denied
OK; ejemplos `01`, `02`, `cli_agent` importan.

## 2026-09-23 — MongoDB: cada backend autocontenido

Otro modelo había creado `capabilities/_mongodb/` (un `MongoDBConfig` base y un
`MongoDBRegistryMixin` que redefinía `connect`/`disconnect` por MRO) y un
`capabilities/MONGODB.md` suelto. Rompía la regla de que cada backend es una
carpeta que se entiende sola (`memory/local`, `knowledge/local`…).

- `memory/mongodb/` y `knowledge/mongodb/` ahora tienen su config completa
  (campos de conexión incluidos, sin herencia) y su propia conexión
  (`connect`/`disconnect`/`_create_indexes`, upsert con reintento por
  `DuplicateKeyError`) dentro del registry. Se duplican ~50 líneas a cambio
  de independencia, igual que entre `local` y `sqlite`.
- Mismo comportamiento y mismos atributos privados (`_mongo_client`,
  `_collection`, `_mongo_config`) que usan los tests.
- `MONGODB.md` repartido en `memory/mongodb/README.md` y
  `knowledge/mongodb/README.md`. Eliminados `_mongodb/` y `MONGODB.md`
  (respaldo en `scratchpad/backup_mongodb/`).

Verificado: `test_mongodb_registries` + `test_provider_models` 29 passed;
round-trip `dump_component`/`load_component` de ambos, sin la URI en la
config; suite idéntico (284/73/36); `examples/mongodb_backends.py` importa.
No probado contra un MongoDB real (no hay servidor en el sandbox).

## 2026-09-23 — `core/tool_state.py` → `core/tool/state.py`

Movido con `scratchpad/move_tool_state.py` (5 archivos). Abría un ciclo
(`types.run_context` → `core.tool.state` → `core/tool/__init__` →
dispatcher → `RunContext`), así que `core/tool/__init__.py` ahora carga
`ToolDispatcher`/`ToolRegistry` perezosamente (PEP 562, como
`capabilities/reasoning`) y exporta `ToolState` directo.
Verificado: suite idéntico (284/73/36); ctx round-trip, esc, preguntas, CLI,
tool denied y guards OK.

Revisión (sin cambios todavía): `base/tool_executor.py` no se puede importar
(pide `CoreExecutor` de `base/executor_legacy`, que ya no existe). Solo lo
usan `ReActLoopSelfDirected` (por eso tampoco importa), la rama legacy de
`BaseReasoning.bind(tool_executor=…)` y 4 tests viejos. `base/capability.py`
sí es clave: es la base de memory/knowledge/skills/context/workspace.
`manager/capabilities.py` (`AgentCapabilities`) no lo usa el Agent nuevo.

## 2026-09-23 — Fuera el stack legacy de ejecución de tools

Confirmado por marvin ("borrar todo lo legacy"). Respaldo completo en
`scratchpad/backup_legacy/`.

- Borrados: `base/tool_executor.py` (ya no importaba: pedía `CoreExecutor` de
  `executor_legacy`), `capabilities/reasoning/react_self_directed.py` (el loop
  que lo usaba; tampoco importaba), `capabilities/tools/plan/_tool.py`
  (`UpdatePlanTool`, solo para ese loop; el Agent usa `AgentUpdatePlanTool`),
  `manager/capabilities.py` (`AgentCapabilities`, el Agent nuevo no lo usa;
  `manager` exporta solo `LayerContainer`), los ejemplos
  `agent_self_directed_loop.py` y `agent_cli.py` (usaban el loop borrado y
  `max_ai.tools`, que no existe) y 10 archivos de tests que ya no cargaban.
- `BaseReasoning`: sin rama `tool_executor`. `bind()` exige `dispatcher` +
  `tool_context`; nueva propiedad `dispatcher` (error claro si no está
  enlazado); `tool_catalog` y `_execute_tools` van solo por el dispatcher; se
  quitó `_register_runtime_tools` (solo lo usaba el loop borrado).
- `capabilities/reasoning` exporta `ReactLoop`/`ReActLoopState` (lazy) en
  lugar del loop borrado. Docstrings de plan y skills actualizados.
- `base/capability.py` se queda: es la base (`CoreAgentCapabilities`) de
  memory, knowledge, skills, context y workspace.

Verificado: los mismos 284 tests pasan; errores de colección 36 → 26 (se
fueron los 10 muertos); 11 scripts OK; ejemplos `01`, `02`, `cli_agent` y
`mongodb_backends` importan.

## 2026-09-23 — Bug: plan paso a paso con el usuario → loop hasta max_iterations

Reporte de marvin (captura): pidió un plan y avanzar "entre cada step". El
modelo pausaba ("escribe siguiente") y repetía ese mensaje ~15 veces.

Dos causas:
1. `RuntimeCompletionGate` rechazaba **toda** respuesta final con el plan sin
   terminar, sin límite. Esperar al usuario era imposible: cada cierre se
   rebotaba hasta `max_iterations`.
2. `Agent` hacía `ctx.plan = None` en cada tarea nueva: al escribir
   "siguiente", el plan ya no existía.

Arreglo:
- Gate: avisa **una vez por estado del plan** (huella `id:status` en
  `completion_state["plan_nudged"]`) con una instrucción clara (seguir, o
  avisar al usuario y cerrar: el plan queda abierto). Si el modelo cierra otra
  vez sin cambiar el plan, se acepta como `completed` con la nota "Turn closed
  with the plan still open (N/M steps closed)". Si el plan cambió (avanzó y
  dice "listo" antes de tiempo), se vuelve a avisar.
- Agent: en tarea nueva solo descarta el plan si ya está terminado; uno
  abierto sigue en la conversación.

Verificado con `scratchpad/verify_plan_pause.py` (reproducía 20 llamadas /
19 mensajes repetidos / max_iterations): ahora pausa en 3 llamadas; el plan
sobrevive a un turno sin `update_plan`; "siguiente" lo avanza; el plan
terminado se descarta; un cierre prematuro tras avanzar se vuelve a frenar.
Suite idéntico; scripts de gate/guards/CLI OK. No probado con modelo real.
Nota para compactación: el plan no está en el system prompt, el modelo lo ve
por su llamada a `update_plan` en `ctx.messages`; si se resume ese mensaje,
hay que renderizar `ctx.plan` en el prompt.

## 2026-09-23 — `executor/` fuera, `manager` → `core/stacks`, `UserFileSystem` a su lugar

- `base/capability.py` se queda: es el contrato `CoreAgentCapabilities`.
- `max_ai/executor/` (1353 líneas) borrado: ningún módulo importaba (pedían
  `CoreExecutor`); los ejecutores vivos están en `capabilities/executor/`.
  También sus 4 tests (ya no cargaban). `examples/file.py` apunta a
  `capabilities.executor` (sigue roto por `max_ai.tools`, preexistente).
- `manager/stacks.py` → `core/stacks/container.py` (`LayerContainer`, arma
  el prompt en cada turno); `manager/` eliminado.
- **`workspace_copy/` no estaba muerto**: `filesystem.py` (`UserFileSystem`)
  es lo que usan las tools de archivos y `WorkspaceBase.get_filesystem()`.
  Movido a `capabilities/workspace/local/_filesystem.py` (exportado por
  `capabilities.workspace.local`). El resto de `workspace_copy/` (artifacts,
  azure, managed, sync, system) solo lo usan `ui/server.py`
  (`ArtifactConflict`) y tests que ya fallaban; queda pendiente de decisión.
- Script `scratchpad/move_core_stacks.py`; respaldo en `backup_legacy2/`.

Verificado: mismos 284 passed; errores de colección 26 → 22 (los 4 de
executor); `verify_fs_tools.py` (el agente escribe y lee `notes.txt` con sus
tools, archivo en `u/workspace/`); CLI, ctx round-trip, plan, gate OK.

## 2026-09-23 — Fuera la UI web y `workspace_copy/`

marvin: no habrá UI por ahora. Como `ui/server.py` importaba de
`workspace_copy/`, se fueron juntos (respaldo en `scratchpad/backup_ui/`):
- `max_ai/workspace_copy/` (artifacts, azure_artifacts, managed, sync,
  system). `UserFileSystem` ya vivía en `capabilities/workspace/local/`.
- `max_ai/ui/` (server + static) y `examples/file.py` (solo levantaba la UI;
  ya estaba roto por `max_ai.tools`).
- Tests que solo probaban eso: `test_ui_server`, `ui/test_display_log`,
  `test_workspace_artifacts_ui`, `test_artifact_integration`,
  `workspace/test_artifact_store`, `workspace/test_azure_artifacts`,
  `workspace/test_workspace_sync` (carpetas `tests/ui`, `tests/workspace`).

Verificado: las 39 líneas que salieron del suite son todas de esos archivos
(272 passed / 47 failed / 21 errores restantes, sin regresiones); módulos que
no importan 12 → 4 (preexistentes: `EmbedMany` en memory sqlite,
`core/executor/worker` pide `Workspace`, ciclo `function_as_tool` ↔
`base/context`); scripts de CLI, plan, ctx, esc, filesystem OK; ejemplos
`01`, `02`, `cli_agent`, `mongodb_backends` importan.
Pendiente: `README.md` usa rutas viejas (y documenta la UI); en
`pyproject.toml` quedan `fastapi`, `uvicorn` y el extra `artifacts-azure`.
`base/agent.py` (Agent viejo) sigue: lo usa `AgentAsTool`.

## 2026-09-23 — Compactación: esqueleto del contrato `CoreCompaction`

Escrito por Claude a pedido de marvin (la lógica la diseñamos juntos después).
- `core/model/compaction.py`: nuevos `CompactionConfig` (threshold,
  keep_ratio, min_keep_groups, truncate_tool_outputs, tool_output_max_tokens,
  drop_harness_messages, client serializado) y `CompactionState` (compactions,
  archived_messages, `state` propio de la estrategia). `CompactionResult`
  rediseñado: `messages` (ventana nueva), `old_messages`, `state`,
  `tokens_before/after`, `pruned_only`. Exportados por `core.compaction`.
- `RunContext.compaction: CompactionState` (sesiones guardadas sin el campo
  siguen cargando).
- `base/compaction.py`: `CoreCompaction(ComponentBase[CompactionConfig])`.
  Implementado: constructor (valida vía la config), `_to_config`/`_from_config`
  (incluye el cliente propio serializado), `render()` por defecto `None`.
  Solo firmas + TODO: `should_compact`, `compact` (pipeline concreto),
  `_compact` (abstracto), `_prune`, `_update_memory`, `_validate`.
- `BaseReasoning`: eliminado el cableado viejo sin uso (`_should_compact`,
  `_run_mid_loop_compaction`, `_compaction_token_counter`); el paso 8 lo
  reemplaza en `ReactLoop`.

Verificado: imports OK; una estrategia mínima se instancia, serializa y
restaura con su cliente OpenRouter; la base abstracta no se instancia;
threshold inválido se rechaza. Suite: solo caen los 5 tests de
`test_sliding_window` (esperado, diseño viejo a rehacer); ctx round-trip, esc,
CLI, plan, preguntas y filesystem OK.
Pendiente: los helpers `live_message_*` siguen leyendo `setting` global en vez
de la config de la estrategia.

## 2026-09-23 — `base/compaction.py` queda solo con el contrato

Los helpers que no cambian por estrategia pasaron a `core/compaction/`, como
`token_counter.py`:
- `budget.py`: `client_max_output_tokens`, `live_message_capacity_tokens`,
  `live_message_threshold_tokens`, `live_message_budget_tokens` (ventana =
  prompt + mensajes vivos + salida reservada + margen).
- `groups.py`: `group_atomic_messages`, `split_recent_messages` (bloques
  tool call + resultados que nunca se separan).
Todo se reexporta desde `max_ai.core.compaction`. `base/compaction.py` (182
líneas) solo tiene `CoreCompaction`. Imports actualizados en
`compaction/sliding_window.py` y `tests/compaction/test_helpers.py`.
Verificado: suite idéntico al paso anterior (solo siguen cayendo los 5 de
sliding window); ctx round-trip OK.

## 2026-09-23 — Compactación paso 1: `should_compact`

- `CoreCompaction.should_compact`: ventana desconocida (≤0) → nunca; cuenta
  `message_history` + `messages` (el historial también se envía al modelo);
  umbral = capacidad × `self.config.threshold`, con la capacidad calculada
  con el tamaño real del system prompt (`prompts.prompt_tokens`).
  "No compactar con tool calls pendientes" NO va acá (es reescribible): esa
  garantía la pone el loop en el paso 8.
- `core/compaction/budget.py`: `live_message_threshold_tokens(..., ratio=)` y
  `live_message_budget_tokens(capacity, ratio=)`; `None` cae al `setting`
  global (compatibilidad), las estrategias pasan su config.
- Tests nuevos `tests/compaction/test_should_compact.py` (6): borde exacto
  6000/6001, ventana 0, historial cuenta, umbral por estrategia (0.5 vs 0.9
  con el mismo ctx), prompt más grande deja menos lugar, no muta el ctx.
Suite: +6 passed, sin otros cambios.

## 2026-09-23 — Compactación paso 2: `_validate` + `CompactionError`

- `errors/compaction.py`: `CompactionError.orphan_tool_result(index, id)` y
  `.unanswered_tool_calls(index, ids)` (mismo estilo que los demás errores).
- `CoreCompaction._validate(messages)`: recorre la ventana por bloques; cada
  `ToolMessage` debe responder a una llamada del `AssistantMessage` que abre
  su bloque (sin duplicados), y cada tool call debe tener su resultado antes
  del siguiente mensaje o del final. Sin excepción para pendientes: el loop no
  compacta con tools esperando aprobación.
- Tests `tests/compaction/test_validate.py` (11): 6 ventanas válidas
  (orden libre dentro del bloque, ventana que empieza en un bloque) y 5
  inválidas (huérfano, de otro bloque, duplicado, llamada sin respuesta en el
  medio y al final).
- `scratchpad/verify_validate_real.py`: una transcripción real del Agent (2
  tools en paralelo, update_plan, ask_user con pausa y respuesta) valida OK.
Suite: +11 passed, sin otros cambios.

## 2026-09-23 — Compactación paso 3: `_prune` (limpieza barata, sin LLM)

- `core/messages.py`: `HARNESS_SOURCE = "harness"`; `loop.py` lo usa
  (`HARNESS = HARNESS_SOURCE`) para que loop y compactación compartan el valor.
- `core/compaction/groups.py`: `current_turn_start(blocks)` → índice del
  bloque con el último mensaje real del usuario (los del harness no abren turno).
- `TokenCounter.encode/decode` públicos.
- `CoreCompaction._prune(blocks, current_turn_start=)`: nunca toca los
  `min_keep_groups` bloques más nuevos. Con `drop_harness_messages` borra
  mensajes del harness de turnos anteriores (los del turno actual se quedan:
  el modelo tiene que actuar sobre ellos). Con `truncate_tool_outputs`
  (`_shrink`/`_cut`) deja los primeros `tool_output_max_tokens` de cada
  resultado grande + "[… N tokens, kept the first 500]" + "[bash result
  ok|failed: …; truncated by compaction. Call the tool again…]", conservando
  tool_call_id/tool_name/success/error; recorta igual los argumentos de texto
  gigantes. Copias con `model_copy` (mensajes inmutables), `token_count=0`
  para recontar, bloque con total recalculado, bloques sin cambios se
  reusan tal cual.
- Tests `tests/compaction/test_prune.py` (9), todos validan la ventana
  resultante con `_validate`. Ejemplo real: un log de 4000 tokens → 533.
Suite: +9 passed, sin otros cambios; plan, gate, esc y validate real OK.

## 2026-09-23 — Compactación paso 4: `compact()` (el pipeline)

`CoreCompaction.compact(ctx, prompts, max_context_tokens, client, memory)`,
nunca muta `ctx`:
0. Con tool calls pendientes (aprobación/ask_user) devuelve la ventana tal
   cual: la garantía vive acá (no se reescribe) y no en el loop.
1. Presupuesto con la config de la estrategia: umbral (`threshold`) y lo que
   se conserva (`keep_ratio`) menos lo que ocupa `message_history`.
2. `group_atomic_messages` → `_prune(current_turn_start)`.
3. Si con la poda ya queda bajo el umbral → `pruned_only=True`, sin llamar a
   la estrategia (sin LLM).
4. Si no → `_compact(blocks, state=copia, budget_tokens, client)`; el cliente
   de la estrategia gana sobre el del agente.
5. `_validate` del resultado (una estrategia que parte un bloque → error).
6. Memoria conectada + `old_messages` → `_update_memory`; si falla, se
   registra y la compactación sigue.
7. `tokens_before` / `tokens_after` (historial incluido).
Tests `tests/compaction/test_compact.py` (8): poda suficiente sin estrategia,
estrategia con presupuesto 3000 y estado copiado (el de ctx no cambia),
historial resta presupuesto (2000), cliente propio vs del agente, bloque
partido rechazado, memoria recibe lo que salió y su fallo no rompe, sin
memoria se salta, record pendiente bloquea.
Suite: +8 passed, sin otros cambios.

## 2026-09-23 — Primera estrategia: `SlidingWindowCompaction` (últimos N turnos)

Definición de marvin: sliding window = se mueve siempre, conserva los
últimos N turnos, sin resumen. Summary (siguiente) = se dispara por tokens y
mantiene [resumen que vive siempre] + [mensajes recientes] + [respuesta] +
[margen]. El `max_ai/compaction/sliding_window.py` viejo es en realidad la
de resumen (nombre engañoso); se reemplaza con `SummaryCompaction`.

- Base: nuevo `_over_limit(blocks, live_tokens, threshold)` reescribible, que
  usan `should_compact` y el atajo post-poda de `compact` (antes, "si ya
  entra en tokens no llamo a la estrategia" impedía deslizar por turnos).
  Default: tokens > umbral (umbral 0 si la ventana es desconocida).
- `core/compaction/groups.py`: `turn_starts(blocks)`; `current_turn_start`
  se apoya en él. Los mensajes del harness no abren turno.
- `capabilities/compaction/window/` (`_model.py`, `_strategy.py`):
  `SlidingWindowCompaction(max_turns=10)`: se dispara con más de N turnos
  (aunque sobren tokens) o por umbral de tokens; corta en el inicio del
  N-ésimo turno desde el final y, si lo que queda supera el presupuesto,
  recorta bloques viejos dentro de esos turnos. Con ventana desconocida solo
  cuenta turnos. Sin estado ni `render`. Registrado como
  `maxai.compaction.SlidingWindowCompaction`. Borrado el placeholder vacío
  `window/slide_windows.py`.
- Tests `tests/compaction/test_window_strategy.py` (6).
Suite: +6 passed, sin otros cambios; ctx, plan y CLI OK.

## 2026-09-23 — `SummaryCompaction` + adiós al `max_ai/compaction/` viejo

`capabilities/compaction/summary/` (`_model.py`, `_strategy.py`):
- Esquema de marvin: [system + resumen que vive siempre] + [mensajes
  recientes] + [respuesta] + [margen]. Se dispara por umbral de tokens.
- El resumen vive en `ctx.compaction.state["summary"]` (viaja con la sesión
  al serializar el ctx); la estrategia no guarda nada, así que una instancia
  sirve a todos los usuarios. Resumen incremental: cada compactación fusiona
  el resumen anterior con lo que sale.
- `summary_max_tokens` (tope de salida de cada llamada, y se reserva del
  presupuesto porque el resumen vuelve al prompt) y `message_cap_tokens`
  (cada mensaje viejo se corta antes de resumir) en la config, ya no en
  `setting`.
- `_split_index`: el corte avanza hasta el siguiente inicio de turno para que
  la ventana no arranque con una respuesta sin su pregunta (Anthropic exige
  empezar con user); en un turno actual larguísimo corta entre bloques y la
  pregunta queda en el resumen.
- Transcript grande → varias llamadas encadenadas (resumen del bloque N es el
  "anterior" del N+1), presupuesto = mitad de la ventana del resumidor.
  Modelos sin structured output → el texto va a `summary`.
- `render(state)`: bloque `<conversation_summary>` con secciones
  (Objectives, Pending, Done, Decisions, Important context), vacías omitidas.
- Registrado `maxai.compaction.SummaryCompaction`.

Bug encontrado por los tests: la base tenía un helper `_cut(text, cap)` y la
estrategia definió `_cut(blocks, budget)`, pisándolo (`_prune` llamaba al de
la estrategia). Renombrados a `_truncate_text` (base) y `_split_index`
(summary).

Borrado `max_ai/compaction/` (el "SlidingWindow" viejo que en realidad
resumía) y sus 5 tests; respaldo en `scratchpad/backup_old_compaction/`.
Tests `tests/compaction/test_summary_strategy.py` (8). Suite 315 passed, los
47 fallos restantes son los preexistentes; ctx, plan, CLI y esc OK.

## 2026-09-23 — Compactación conectada al loop (Agent + ReactLoop)

- `Agent(compaction=CoreCompaction | None)` (None = no compacta). `bind`
  recibe `compaction`, `max_context_tokens` (de `client.config`, 0 si un
  cliente propio no la declara) y `memory`.
- Nueva capa `SessionStateLayer` (siempre en el stack, vacía si no hay nada):
  `compaction_summary` (= `strategy.render(ctx.compaction.state)`) y
  `current_plan` (`AgentPlan.as_text()`). Así el plan sobrevive aunque la
  llamada a `update_plan` salga de la ventana.
- `BaseReasoning._refresh_session_state(ctx, prompts)`: antes de cada llamada
  al modelo; re-renderiza el stack solo si cambió el resumen o el plan, y
  re-mide el prompt (`PromptCtx.measure`).
- `BaseReasoning._compact_if_needed(ctx, prompts)`: `should_compact` →
  `CompactionEvent(start)` → `compact` → aplica `result.messages`,
  `result.state`, `compactions`, `archived_messages` → refresca el prompt →
  `CompactionEvent(end)`. Si falla, `ErrorEvent(compaction_failed,
  recoverable)` y el turno sigue sin compactar. `ReactLoop` lo llama donde
  estaba el placeholder.
- `CompactionEvent` rediseñado: phase, strategy, changed, pruned_only,
  tokens_before/after, kept_message_count, old_messages (para archivar),
  summary (lo que ve el modelo).
- `Agent._prompts` mide el system prompt real (antes `prompt_tokens=0` y el
  presupuesto usaba un fijo de 6000).
- Bug de fondo encontrado: el entorno Jinja de las capas escapaba HTML en
  TODAS las variables (`select_autoescape` con `default_for_string=True`): una
  memoria "O'Brien & Co <vip>" llegaba como "O&#39;Brien &amp; Co &lt;vip&gt;".
  Ahora `autoescape=False` (los prompts son texto para el modelo).
- Tests `tests/compaction/test_agent_compaction.py` (3, end to end con
  Agent real): con prompt medido de 3633 tokens y ventana de 8633, 12 turnos →
  poda (3354→3174), resumen (3641→934, 14 msgs), resumen (3610→934, 12 msgs);
  el system prompt siguiente trae `<conversation_summary>` y `<current_plan>`
  aunque la llamada a update_plan ya no está; ventana siempre válida; el ctx
  serializa el estado. Sliding window por turnos con ventana desconocida; sin
  estrategia no cambia nada.
- `test_agent_capability_prompts::test_no_capabilities` actualizado (la capa
  nueva está siempre, vacía).
Suite: 318 passed (+3), mismos 47 fallos preexistentes; 13 scripts OK.
Nota: `compactions` también cuenta las pasadas que solo podaron.

## 2026-09-23 — CLI: barra de ventana y bloque de compactación

- Línea de uso: `demo · in 40.3k · out 1.4k · ctx ▓▓▓▓▓▓░░░░ 36% left ·
  summary compaction at ~83%`. `context_bar()` (blocks.py): verde, ámbar desde
  60% usado, rojo desde 85%. El % sale de los `tokens_input` reales del
  proveedor; tras compactar se descuenta lo liberado hasta la próxima llamada.
  "at ~N%" = system prompt + umbral de la estrategia sobre la capacidad,
  usando `ModelCallEvent.prompt_tokens` (lo agregó la sesión paralela
  max-ai-1b en `core/event_type.py` y `base/reasoning.py`). Sin `total`, y
  `cached` solo si >0 (para que entre en ~90 columnas).
- `CompactionBlock` (FoldBlock): "◇ Compacting context…" en vivo; al terminar
  se pliega a "Compacted · N messages → summary · 3.1k → 726 tokens",
  "Trimmed old tool output · …" (solo poda) o "Window slid · N messages out";
  click muestra lo que ve el modelo (el resumen). Si terminó sin cambios, el
  bloque se quita (sin ruido). `ErrorEvent(compaction_failed)` → "Compaction
  failed · … · continuing uncompacted".
- `OpenRouterChatCompletionClient.fetch_context_window(models)`: la ventana
  más chica entre modelo y fallbacks desde `/models` (262144 para los 3
  free); 0 si falla o falta un modelo.
- Ejemplos: `02` usa esa ventana; `01` la toma de `MAX_CONTEXT_WINDOW` (OpenAI
  no la expone); en ambos `MAX_CONTEXT_WINDOW=8000` fuerza una ventana chica
  para ver compactar. `cli_agent.py` usa `SummaryCompaction()`.
- Verificado headless (`scratchpad/verify_cli_compaction.py`): 7 turnos, la
  barra baja 46%→0% y tras compactar vuelve a 36%, un bloque plegado que se
  expande con el resumen. Tests CLI actualizados al nuevo formato de uso
  (`test_textual_interaction`: "in 20 · out 6 · cached 4"). Suite 318 passed,
  mismos 47 fallos preexistentes; 11 scripts OK.

## 2026-09-23 — `max_context_window` siempre entre 128K y 1M

Reporte de marvin (ejemplo 01 con OpenAI): la CLI mostraba "summary
compaction" sin barra de contexto, porque `max_context_window` quedaba en 0 y
la barra solo se dibuja con ventana conocida.
Regla pedida: mínimo 128K, máximo 1M.
- `core/model/llm.py`: `MIN_CONTEXT_WINDOW = 128_000`, `MAX_CONTEXT_WINDOW =
  1_000_000`; default 128K y un validador que ajusta al rango (0/desconocida
  o menor → 128K, mayor → 1M). Se eligió ajustar en vez de rechazar porque
  el ejemplo 01 lee 1.05M del catálogo de OpenRouter para gpt-5.6-luna y
  habría fallado al arrancar.
- Ejemplos: `MAX_CONTEXT_WINDOW` sigue como override (dentro del rango); para
  ver compactar en pocos turnos, `COMPACTION_THRESHOLD=0.05` en
  `cli_agent.py` (umbral de `SummaryCompaction`, default 0.8).
- Tests con ventanas chicas (`test_agent_compaction`,
  `verify_cli_compaction`) usan una config propia del cliente de prueba (como
  un cliente personalizado), no `ModelConfig`.
Verificado: headless con el cliente real de OpenAI del ejemplo 01 → "ctx
░░░░░░░░░░ 99% left" con 1M (catálogo) y "▓░░░░░░░░░ 95% left" con 128K
(sin ventana). Suite sin cambios; scripts OK.

## 2026-09-23 — Ventana por defecto 128K en los ejemplos; barra con tokens reales

Reporte de marvin: la barra nunca se llenaba ("99% left" con in 42.2k) y
"compaction at ~76%" quedaba pegado al final. Causa: el ejemplo 01 leía la
ventana de gpt-5.6-luna del catálogo de OpenRouter (1.05M → 1M); contra 1M, ~10k
de contexto es <1% y compactar al 76% son ~760k tokens.
Decisión de marvin: 128K es el default y solo cambia si el usuario lo pide.
- Ejemplos 01 y 02 ya no consultan el catálogo: `ModelConfig` por defecto
  (128K), `MAX_CONTEXT_WINDOW` como único override. `fetch_context_window()`
  queda disponible en el cliente de OpenRouter para quien lo quiera.
- `context_bar` muestra tokens reales: "▓░░░░░░░░░ 9.6k / 128k" (antes solo
  "% left"); `_k` formatea 128k / 1M.
- Con 128K y prompt ~3.6k, "summary compaction at ~76%" ≈ 97k tokens.
Verificado headless con el cliente real de OpenAI: default → "9.6k / 128k ·
summary compaction at ~76%"; tests CLI + compaction OK; scripts OK.

## 2026-09-23 — Compactación cerrada: prueba real + `_update_memory`

Prueba real con OpenAI gpt-5.6-luna (`scratchpad/verify_compaction_real.py`):
turno 1 "mi perro se llama Toby y vivo en Lima", 8 turnos de relleno, pregunta
final. Resultado: compactó en el turno 7 (1,923 → 334 tokens, 12 mensajes
fuera), "Toby" ya no está en los mensajes crudos y el modelo responde bien
desde el resumen.

Bugs encontrados por la prueba real:
- `keep_ratio >= threshold` hacía la compactación un no-op eterno (conservaba
  todo lo que la disparaba; pasaba con `COMPACTION_THRESHOLD=0.05`).
  `CompactionConfig` ahora lo rechaza con un error claro; `cli_agent.py` usa
  `keep_ratio = threshold / 2`.
- El resumen con tope de tokens bajo cortaba el JSON → "Failed to parse
  structured output" y el JSON roto quedaba como texto del resumen. Ahora el
  pedido incluye el presupuesto ("Keep the whole summary under N tokens") y un
  JSON cortado se rescata con `from_json(allow_partial=True)` (se conservan
  los campos completos). Repetida la prueba con 300 tokens: resumen limpio.

`_update_memory` (base):
- Le pasa al LLM las memorias actuales (`category: content`) y los mensajes
  que salen; devuelve `MemoryMaintenanceOutput` con el contenido COMPLETO de
  cada categoría que cambia (porque `create_or_update` reemplaza) y se guarda
  con `memory.create_or_update`. Salida en prosa → no guarda nada.
- `MemoryFactUpdate` sin `key` (la identidad es la categoría).
- Config: `update_memory` (default True), `memory_max_tokens` (1000).
- Helpers compartidos en la base: `_transcript(messages, cap)` y
  `_ask(client, task, output_format, max_tokens)` (structured, JSON parcial o
  texto); `SummaryCompaction` los usa en vez de sus copias.
- Real: memoria guardada "Datos personales: El usuario vive en Lima y tiene un
  perro llamado Toby." (sin la charla de relleno).
Tests nuevos: `test_update_memory.py` (4, con `LocalMemoryRegistry` real) y
JSON cortado en `test_summary_strategy.py`. Suite sin regresiones.

## 2026-09-23 — Sesiones: `CoreSessionStore` + Agent sin estado (memoria desde el ctx)

Decisiones de marvin: el framework no guarda nada por su cuenta; el host
(servidor, CLI) hace load → run → save. El Agent es uno para todos: el backend
de memoria se configura una vez y el RunContext dice user_id/session_id.
Modelo de despliegue previsto: agente serializado en la base, deserializar →
correr → morir (la config del agente no lleva usuario ni sesión).

Session store:
- `core/model/session.py`: `SessionInfo` (user_id, session_id, title,
  updated_at, message_count, compactions).
- `base/session_store.py`: `CoreSessionStore` (componente con ciclo de vida):
  `load(user_id, session_id)`, `save(ctx)`, `list_sessions(user_id, limit)`,
  `delete(...)`; valida ids ([A-Za-z0-9_-], 1-128) y rechaza cargar una
  sesión guardada de otro usuario. Backends implementan `_load/_save/_list/
  _delete`.
- `capabilities/session_store/local/`: `LocalSessionStore(base_path)` →
  `<base>/<user>/<session>.json` (RunContext) + `.meta.json` (SessionInfo),
  escritura atómica. Registrado `maxai.session_store.LocalSessionStore`.
- Reemplaza a `max_ai/persistence/` (RunContextStore por run_id, sin usuario);
  borrados con sus tests y `tests/integration/test_ollama_approval_flow.py`
  (Agent viejo). Respaldo en `scratchpad/backup_persistence/`.
- Tests `tests/session_store/` (13), incluido pausa → save → otro proceso
  load → aprobar → resume.

Memoria sin estado:
- `CoreMemoryRegistry`: user_id/session_id opcionales; `bind(user_id,
  session_id)` devuelve una copia atada que comparte la conexión del backend;
  operaciones sin atar → `MemoryError.unbound()`; hook `_validate_scope`.
- Tools de memoria reciben `ToolContext | None` y operan sobre
  `self._for_run(context)` (el usuario/sesión de la corrida; sin contexto, el
  alcance propio). `FunctionAsTool` ahora acepta `ToolContext | None` como
  primer parámetro (pasa None fuera de una corrida).
- Local: `connect` ya no crea la carpeta del usuario (se crea al escribir);
  Mongo: `_to_config` toma el alcance de la instancia. Configs con
  user_id/session_id opcionales.
- Agent: `_memory_for(ctx)` conecta el backend una vez y lo ata al ctx; lo
  usan el prompt (MemoryLayer) y la compactación. Ya no existe
  `_validate_memory_scope`.
- `cli_agent.py`: `LocalMemoryRegistry(base_path=...)` sin usuario.
- Tests `tests/agents/test_stateless_memory.py` (6): un Agent, Ana y Beto con
  archivos y prompts separados; search cruza sesiones del mismo usuario y
  nunca de otro; sin atar no toca storage; copias Mongo comparten cliente;
  la config serializada no lleva usuario. `test_agent_capability_prompts`
  actualizado (otra sesión → memoria propia vacía, no error).
Suite: 314 passed (+19 nuevos), sin regresiones; scripts OK.
Siguiente: parte B (CLI con store: guardar por turno, /resume, --session).

## 2026-09-23 — CLI como host de sesiones: guardar por turno, --session, /resume, /new

Solo la CLI (el framework no cambia): hace lo mismo que un servidor
(load → run → save).
- `run_repl(agent, store=..., user_id=..., session_id=...)` y `MaxAIApp`
  con los mismos parámetros. Sin store, nada se guarda (como antes) y
  `/resume` lo avisa. `initial_context` sigue para casos avanzados.
- Arranque: con `session_id` carga del store y redibuja la conversación; si no
  existe, empieza una nueva con ese id. Sin `session_id`, id nuevo
  (`short_id`) y una línea "Session X · saved after every turn · resume it
  with /resume or --session X".
- Cada turno se guarda al terminar (también en pausa resuelta o interrumpido
  y revertido); un fallo al guardar se muestra y no tumba la UI.
- `/resume [id]`: selector con el mismo `QuestionForm` de ask_user (↑↓ enter,
  "Other" para pegar un id, esc cancela); lista las otras sesiones del
  usuario, más recientes primero, con "title — 5 min ago · 24 msgs · 1
  compaction". `/new` y `/clear` abren una sesión nueva (antes /clear
  reutilizaba el id y habría pisado la guardada). `/session` muestra el id.
- Redibujado (`_replay`): mensajes del usuario, respuestas, una línea por tool
  call, el plan; si hubo compactación, `PastSummaryBlock` plegado ("Earlier
  conversation summarized · N messages", click muestra el resumen).
- Bug encontrado con la captura: cerrar la app con el selector abierto
  escribía en una pantalla ya destruida; ahora solo esc (nuestro token) se
  trata como cancelación, el cierre de la app se propaga.
- Ejemplo: `cli_agent.py` usa `LocalSessionStore(examples/local/sessions)`
  (ignorado por git vía `user*/`) y `--session <id>`.
Tests `tests/cli/test_sessions.py` (7): guardar por turno + reabrir con
--session y continuar, id desconocido, selector cambia de sesión (más reciente
primero), esc cancela, /new conserva la anterior, resumen al retomar sesión
compactada, sin store no guarda. Suite 321 passed, sin regresiones; 10
scripts CLI OK (`verify_cli_blocks` actualizado: /clear = sesión nueva).

## 2026-09-23 — Agent serializable, paso 1: ningún componente guarda secretos

Regla (decisión de marvin): un componente serializable guarda el NOMBRE de la
variable de entorno, nunca el secreto (la config del agente irá a una base).
- Bug encontrado: los 3 clientes LLM guardaban `api_key` como SecretStr → en
  JSON "**********" y al deserializar quedaban con esa clave literal (401
  confuso en la primera llamada).
- Clientes: `CoreChatCompletionClient.API_KEY_ENV` + parámetro `api_key_env`;
  la clave es `api_key` (código) o `os.environ[api_key_env]`. Configs:
  `api_key_env` en vez de `api_key` (OpenAI "OPENAI_API_KEY", OpenRouter
  "OPENROUTER_API_KEY", Ollama "OLLAMA_API_KEY", opcional). OpenAI/OpenRouter
  sin clave → "OpenAI needs an API key: pass api_key or set $X".
- MCP: `token_env`, `headers_env` (header → var) y `env_from` (var del hijo →
  var del host), resueltos al conectar (`request_headers`, `process_env`);
  var faltante → "MCP server 'x' needs env var Y". `token` literal: solo en
  código (`exclude=True`, fuera de repr) y `serialize_mcp_servers` lo rechaza
  (`ensure_serializable`). Aprobación por tool ya existía
  (`tool_approval_modes`).
- Ejemplos: 01 pasa `api_key_env` con la variable que exista (OPENAI_KEY…);
  02 deja que el cliente lea OPENROUTER_API_KEY.
- Tests: `tests/test_no_secrets_in_configs.py` (7, guardia: ningún secreto ni
  "*****" en el JSON y el componente restaurado sigue usable) y 2 nuevos en
  `test_mcp_integration.py`; el round-trip de MCP ya no acepta token literal.
Verificado con OpenAI real: cliente → JSON (sin clave) → load_component →
respuesta "ok". Suite 330 passed, sin regresiones; scripts OK.

## 2026-09-23 — Agent serializable, puntos 1-3: reasoning, gate y lista blanca

Nombres (pedido de marvin): `dump_component`/`load_component` → `serialize()`
/ `deserialize()` en TODOS los componentes; `deserialize` acepta el
ComponentModel, su dict o el JSON (str/bytes), p. ej. directo de una fila.
`FunctionAsTool.serialize()` falla con "exponela vía MCP".

1. Reasoning y guards son componentes (marvin: quien use otro loop pasa el
   suyo, con su config):
   - `BaseReasoning(ComponentBase[ReasoningConfig])`, component_type
     "reasoning". `ReactLoop` → `ReactLoopConfig(max_loop_iterations,
     max_connection_retries, guards)`; `guards=None` = "los defaults" (un
     agente guardado recibe defaults mejorados), `[]` = sin guards.
   - `LoopGuard(ComponentBase[GuardConfig])` con serialización genérica desde
     su `component_schema`; configs para RepetitionGuard, BudgetGuard y
     PlanCompletionGuard. Proveedores `maxai.reasoning.ReactLoop` y
     `maxai.guards.*` en KNOWN_PROVIDERS.
2. `RuntimeCompletionGate(workspace, config=RuntimeGateConfig(...))`: opciones
   `enabled`, `plan_must_close`, `check_bash_outputs`, `nudge_bash_failures`
   (todas True por defecto). El workspace sigue inyectándolo el Agent (es
   runtime, no config). Gates propios: componentes por proveedor.
3. Lista blanca de proveedores en `ComponentBase.deserialize` (antes importaba
   cualquier clase que dijera el JSON): siempre `max_ai.`; más
   `allow_providers("mi_empresa.", ...)`, `MAXAI_ALLOWED_PROVIDERS`
   (comma-separated) o `"*"` para desarrollo. Se chequea antes del import.
Tests `tests/test_serializable_runtime.py` (7): loop por defecto/guards
propios/sin guards, loop y guard de terceros rechazados hasta permitirlos,
env y wildcard, proveedor falsificado `os.system` rechazado, opciones del
gate. Suite sin regresiones.
Pendiente: Agent.serialize()/deserialize() (juntar todo, gate options y
gates propios en el Agent, error con tools de función), output_format como
JSON Schema (punto 4), test con MCP real.

## Agent.serialize() / Agent.deserialize()
- `Agent` es ahora un `ComponentBase[AgentSpec]` (provider `maxai.agents.Agent`). `AgentSpec` (core/model/agent.py) reemplaza al viejo `AgentComponentConfig`: client, reasoning, workspace, executor, memory, skills, knowledge, compaction, toolset, mcp_servers, completion (RuntimeGateConfig), completion_handlers (gates propios, por provider) y output_format.
- Nuevo parámetro `Agent(completion=RuntimeGateConfig(...))`.
- Tools: las built-in no se guardan (el Agent las recrea). Las del desarrollador se guardan solo si son componentes; una función Python da `TypeError` indicando usar MCP. Un gate propio sin `_to_config` da un error igual de claro.
- output_format se guarda como JSON Schema y se reconstruye con `core/model/json_schema.model_from_schema` (misma forma y validación de tipos/required/enums/anidados; se pierden validators y métodos propios).
- Providers públicos (`component_provider_override`) para LocalWorkspace, LocalExecutor, LocalMemoryRegistry, LocalSkillRegistry, LocalKnowledgeRegistry y los context registries; `RuntimeCompletionGate` registrado en KNOWN_PROVIDERS; `CoreTool.component_type = "tool"`.
- Tests: tests/agents/test_agent_serialization.py (6).
- Pendiente: decidir cuándo aplica output_format (hoy va en cada llamada del loop), test con MCP real.

## output_format solo en la respuesta final
- Antes: `output_format` iba como response_format/format en CADA llamada del loop.
- Ahora: el loop trabaja sin formato; cuando los gates aceptan la respuesta, `ReactLoop._format_final_answer` hace UNA llamada extra sin tools con el formato (instrucción `FORMAT_FINAL_ANSWER`, transitoria) y pone el objeto en `structured_output` del mensaje final. El transcript conserva la prosa. Sin output_format no hay llamada extra.
- Seguridad: `load_type_ref` (rehidratación de structured_output al cargar una sesión) ahora pasa por la lista blanca de providers antes de importar.
- Tests: tests/agents/test_output_format_final_only.py (2). Prueba real pendiente: .env sin OPENAI_API_KEY, OpenRouter sin créditos y los modelos :free con rate limit.

## Respuestas cortadas por max_tokens, tope de 32K, CLI y ejemplos
- Bug (capturas del Snake): con `max_tokens=1500` el `write_file` del HTML se cortaba (`finish_reason: length`). En streaming, `_build_tool_calls_from_chunks` descartaba en silencio la llamada rota → mensaje vacío → el gate decía "no answer was generated" → el modelo repetía → `waiting`. Reproducido con la API real.
- Fix: `ChatCompletionChunk.finish_reason` (OpenAI y Ollama lo reportan en streaming). En `ReactLoop`, si `finish_reason == "length"`: se descartan las tool calls rotas y se le dice al modelo el límite concreto (`OUTPUT_CUT_OFF`, "~1,500 tokens"). Al segundo corte seguido el turno termina con `finish_reason="output_limit"` y la CLI se lo explica al usuario. Prueba real: con 1500 el modelo repetía el intento completo hasta max_iterations aunque se le avisara → por eso el corte a los 2.
- `MAX_OUTPUT_TOKENS = 32_000` (core/model/llm.py, `output_token_limit()`): lo aplican los clientes OpenAI/Ollama (OpenRouter hereda), `ModelConfig.max_output_tokens` y `client_max_output_tokens` (budget de compactación, también para clientes propios). Con la ventana mínima (128K) quedan ~81K para mensajes y la compactación salta en ~65K.
- Ejemplos: `max_tokens=32_000`; prueba real con gpt-5.6-luna: `write_file` de 19,362 caracteres en una sola llamada.
- CLI: "gate retried N×" (steering para el modelo) solo en modo verbose (ctrl+o); se quitó "gate waiting"; la barra de métricas tiene padding abajo.
- Ejemplos 01/02: cada uno arma su `Agent` completo a la vista y llama `run_repl(agent, store=..., user_id=..., session_id=...)`. `examples/cli_agent.py` eliminado; lo compartido (rutas, get_weather, send_email, `session_arg`) quedó en `examples/shared.py`.
- Tests: test_output_cut_off.py (2), test_output_token_limit.py (2).
- Propuesta pendiente: `write_file(append=True)` para escribir archivos grandes por partes.

## core/harness, run_cli y revisión de event_type
- Nuevo `max_ai/core/harness/messages.py`: todos los textos que el harness le manda al modelo (loop: max iterations, output cortado, formato final, tool denegada; gate del framework; guards). Constantes para textos fijos y funciones para los que dependen del run. Loop, gate y guards importan `from ...core.harness import messages as harness` y solo deciden cuándo hablar.
- Arreglado: al llegar a max_iterations el loop metía el texto literal "max_iterations_reached" como mensaje de usuario; ahora es una instrucción real (contar dónde quedó antes de empezar algo nuevo).
- Los prompts de compactación (SUMMARY_TASK, MEMORY_TASK) quedan en su estrategia: son de la estrategia, no del harness.
- `run_repl` → `run_cli` (cli/__init__, app, README, ejemplos). README de la CLI al día (sesiones, /resume /new /session, run_cli con store/user_id/session_id).
- Revisado el cambio de otro agente en core/event_type.py: correcto. La unión se llamaba igual que la clase base `OrchestrationEvent` y la pisaba; ahora es `OrchestrationEvents` (como `AgentEvents`).

## Middleware nuevo (componente serializable) + BudgetMiddleware
- `base/middleware.py` reescrito: `CoreMiddleware(ComponentBase[MiddlewareConfig])`, hooks opcionales (async simples, no generadores): `on_run_start`, `on_model_request`, `on_model_chunk`, `on_model_response`, `on_model_error` (puede recuperar con otro resultado), `on_tool_request` (devolver un ToolResult bloquea la tool), `on_tool_response`, `on_final_response` (mapear la respuesta aceptada), `on_run_end`. Serialización genérica por `component_schema` (como los guards).
- Estado por run en `RunContext` vía `mw.state(self)` (un Agent sirve a todos). `ModelRequest`/`ToolRequest` llevan `metadata` para emparejar request/response (tiempos, trazas). `ModelRequest.model_config` trae capacidades y precios.
- `StopRun(message, finish_reason)`: desde run_start o hooks de modelo termina el turno limpio; `AgentResponse.stop_message`; FinishReason suma `budget_exceeded` y `stopped`. Las tools no usan StopRun (dejaría tool calls sin resultado): devuelven un ToolResult.
- Dónde se enganchan: run en `Agent._drive_connected`, modelo en `_call_llm`/`_call_llm_stream`, tools en `ToolDispatcher` (después de la aprobación, justo antes de ejecutar), respuesta final en el loop tras el gate (y tras output_format).
- `Agent(middlewares=[...])`, serializable en `AgentSpec.middlewares`.
- `BudgetMiddleware(max_tokens, max_cost_usd, max_seconds, max_model_calls, max_tool_calls)` por tarea (una tarea sigue contando al reanudar; el tiempo solo cuenta mientras el agente corre). `spent(ctx)` para el host. Costo con `ModelConfig.input_cost_per_mtok/output_cost_per_mtok` (sin precios, max_cost_usd da error claro).
- `LoggingMiddleware` portado (ahora también tools y run). Eliminados `ConsoleTraceMiddleware`, `types/middleware.py` (MiddlewareCtx) y `loggers/middleware.py`: solo servían al API viejo, nadie los usaba.
- CLI muestra `stop_message`. Tests: tests/middleware/test_middleware.py (11). Prueba real (gpt-5.6-luna, streaming): logging completo y presupuesto de 1 tool → `budget_exceeded`.
- Pendiente: cuota por usuario entre tareas (`CoreQuotaStore`), trazas OpenTelemetry.

## Cuota por usuario (BudgetMiddleware + CoreQuotaStore)
- `core/model/quota.py`: `QuotaLimits(period="day"|"month", max_tokens, max_cost_usd, max_tasks)` con `period_key()` (UTC, p. ej. `day:2026-09-23`), `resets_at()` y `exceeded()` (el límite de tareas solo aplica al iniciar una tarea). `QuotaUsage(tokens, cost_usd, tasks)`.
- `base/quota_store.py`: `CoreQuotaStore` con `usage(user, period_key)` y `add(user, period_key, delta)` (atómico, devuelve el total). `LocalQuotaStore` (`capabilities/quota_store/local`): un JSON por usuario, lock por usuario + escritura atómica, guarda los últimos 60 periodos. Varios procesos compartiendo carpeta necesitan un store con base de datos.
- `BudgetMiddleware(..., quota=QuotaLimits(...), quota_store=...)`: al iniciar lee el uso (y rechaza si se agotó, con la hora de reinicio); cuenta la tarea; después de cada llamada al modelo suma tokens/costo en el store; antes de cada llamada revisa. Serializa el store anidado. `quota_used(user)` para el host.
- Tests: tests/middleware/test_quota.py (5): cuota compartida entre tareas e instancias nuevas del Agent (modelo serverless), usuarios independientes, límite de tareas, reinicio del periodo, costo mensual con precios, 50 cargos concurrentes sin pérdidas, serialización.

## MongoDBQuotaStore y firma del Agent más limpia
- `MongoDBQuotaStore` (capabilities/quota_store/mongodb): un documento por (user_id, period), `add` = un `$inc` upsert atómico (find_one_and_update), índice único; URI por `uri_env`. Probado contra MongoDB 8 real (contenedor temporal, ya borrado): 200 cargos concurrentes desde dos stores sin pérdidas y cuota compartida entre Agents nuevos. tests/middleware/test_quota_mongodb.py se salta sin MONGODB_URI.
- `setting.max_loop_iterations` (20, env MAX_LOOP_ITERATIONS) y `setting.environment_idle_timeout` (300, env ENVIRONMENT_IDLE_TIMEOUT).
- `ReactLoop(max_loop_iterations=None)`: None = sigue el setting, también serializado. El límite de iteraciones vive solo en el loop.
- `EnvironmentManager(idle_timeout=None)` usa el setting.
- Agent: fuera `max_iterations`, `idle_timeout` y el alias `mcp_servers` (queda `mcp`); `toolset` pasa a keyword-only. `AgentSpec`: sin max_iterations/idle_timeout, `mcp_servers` → `mcp`.
- Loop vs BudgetMiddleware: no se comparan; cada límite se revisa por su lado y el primero que se alcanza (el más estricto) detiene el turno.
- Tests: tests/test_runtime_settings.py (2).

Pendiente para la próxima sesión: `TracingMiddleware` con spans de OpenTelemetry (convenciones GenAI), probado con Langfuse como backend (OTLP). Después: CoreEmbedding en core/embeddings, archivo/búsqueda de conversaciones en el session store.

## Trazas: TracingMiddleware (OpenTelemetry) + Langfuse
- `capabilities/middleware/tracing.py`: `TracingMiddleware(capture_content=True, max_content_chars=20000, tracer_provider=None)`. Una traza por run: `invoke_agent <agent>` (user.id, session.id, input/output, finish_reason, tokens) con hijos `chat <model>` (tokens, finish_reasons, costo si hay precios, input/output) y `execute_tool <name>` (argumentos, resultado, error). Atributos `gen_ai.*` (convenciones GenAI de OTel) + `langfuse.*` (tipo agent/generation/tool, trace input/output, cost_details, level). Sin provider configurado los spans son no-op.
- Los spans vivos no van al RunContext: quedan en memoria por run_id y terminan con el run. Una pausa cierra la traza; la reanudación abre otra con el mismo session.id.
- Nuevo hook `on_run_error(mw, error)`: el Agent lo llama si el run se cae o se cancela (antes el span quedaba abierto y el error no aparecía). La cadena lo entrega a todos aunque alguno falle.
- `configure_langfuse()`: TracerProvider + BatchSpanProcessor + OTLP/HTTP a `{LANGFUSE_HOST}/api/public/otel/v1/traces` con Basic auth desde LANGFUSE_PUBLIC_KEY/SECRET_KEY. Devuelve el provider: `force_flush()` antes de que termine una Lambda, `shutdown()` al salir.
- Ejemplos 01/02: si hay LANGFUSE_PUBLIC_KEY, activan el tracing solos.
- Tests: tests/middleware/test_tracing.py (5). Prueba real con gpt-5.6-luna (exporter en memoria): árbol correcto. Pendiente: probar contra Langfuse real (faltan las claves en .env).

## Langfuse en Docker (probado de punta a punta)
- `docker-infra/observability/compose.yaml` (incluido en compose.yaml, perfil `observability`): Langfuse v4 (web + worker) con Postgres 17, ClickHouse 25.12, Redis 7 y MinIO, basado en el compose oficial. El primer arranque crea org `MaxAI`, proyecto `max_ai`, claves `pk-lf-maxai-local`/`sk-lf-maxai-local` y el usuario de la UI (admin@maxai.local / maxai-local-ui). Solo UI (3000) y MinIO (9090) publicados en 127.0.0.1. Documentado en docker-infra/README.md; variables en .env.example.
- Prueba real (gpt-5.6-luna → configure_langfuse → Langfuse 4.43 local): Langfuse reconoce AGENT/GENERATION/TOOL, usuario, sesión, input/output, tokens y costo (desde los precios del ModelConfig). Lectura por `GET /api/public/v2/observations` (en v4 `/api/public/traces` ya no existe).
- TracingMiddleware: `user.id`/`session.id` ahora también en los spans hijos (Langfuse v4 filtra por observación).

## Concurrencia: un Agent atiende muchos runs a la vez
- Antes: un lock global (`_turn_lock`) serializaba todos los runs del Agent, y `reasoning.bind()` escribía cliente, tool_context, memoria y loop_state en la instancia compartida (sin el lock, un run usaba el contexto/memoria de otro usuario). MCP se conectaba/desconectaba en cada run (un run al terminar cortaba la conexión de otro).
- `BaseReasoning.bind()` devuelve una copia superficial ligada al run; la instancia compartida nunca cambia.
- Lock por `(user_id, session_id)` (WeakValueDictionary): sesiones distintas corren en paralelo, mensajes de la misma sesión en orden. `close()` espera los runs activos (contador + Event) y luego cierra MCP y las sesiones de ejecución.
- MCP: `_ensure_mcp()` conecta una vez (con lock) y registra las tools una vez; conexión compartida hasta `close()`. `MCPClientManager.connect` reconecta si el worker murió.
- Tests: tests/agents/test_concurrency.py (4): 10 usuarios a la vez ~0.9s vs 6s en serie, aislamiento (tool context, workspace, eventos del stream, memoria en el prompt), misma sesión en orden, loop compartido intacto, MCP con 8 runs simultáneos = 1 conexión. tests/mcp/test_agent_mcp.py actualizado al nuevo comportamiento. Prueba real OpenAI: 5 usuarios 4.2s vs 11.7s, cada uno con su dato.
- Pendiente: versión/concurrencia optimista en CoreSessionStore para varios procesos sobre la misma sesión.
