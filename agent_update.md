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
