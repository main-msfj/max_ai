# 01 — Feedback: qué está bien, qué está mal, qué no hace sentido

Las rutas son relativas a la raíz del repo. Cada punto dice si lo **comprobé** (lo ejecuté) o si es **lectura** del código.

---

## 1. Lo que está bien (no lo toques)

Estas piezas están mejor resueltas que en varios frameworks conocidos. Vale la pena conservarlas tal cual cuando muevas cosas al harness.

- **`Agent` sin estado por corrida.** `reasoning.bind()` devuelve una copia por run ([base/reasoning.py:253](../max_ai/base/reasoning.py#L253)), el estado vive en `RunContext` y `loop_state`. Un solo `Agent` sirve a muchos usuarios. Esto es lo que hace posible multi-agent y always-on sin rediseñar.
- **`RunContext` serializable + pausa/resume.** Aprobaciones, preguntas al usuario y el `inbox` a mitad de turno ya funcionan sobre datos, no sobre corrutinas vivas. Es la base correcta para un agente que duerme y despierta.
- **El transcript nunca queda roto.** `answer_unanswered`, `close_interrupted`, `_with_unique_call_ids`, `_without_broken_calls`: toda tool call termina con su respuesta aunque haya cancelación, corte por tokens o ids repetidos. Esto es de lo más difícil de un loop y está bien hecho ([core/tool/dispatcher.py:93-130](../max_ai/core/tool/dispatcher.py#L93-L130), [react/loop.py:75-101](../max_ai/capabilities/reasoning/react/loop.py#L75-L101)).
- **Policy en un solo lugar**, con el estilo `Tool(pattern)`, `deny > ask > allow`, y el análisis de bash que desarma `env sudo x`, `bash -c '...'`, `eval` ([tools/bash/_permissions.py](../max_ai/capabilities/tools/bash/_permissions.py)). El concepto `certain` (si no pude leer toda la línea, no la permito sola) es la decisión correcta.
- **Lotes de lectura en paralelo, escritura en orden** ([dispatcher.py:64-76](../max_ai/core/tool/dispatcher.py#L64-L76)).
- **Los textos del harness en un archivo** ([core/harness/messages.py](../max_ai/core/harness/messages.py)) y el steering como mensaje `[harness]` de usuario en vez de system tardío.
- **`wrapper.sh` de bash**: cwd que persiste, salida cortada por el medio, lista de archivos cambiados, descriptores cerrados para que un `server &` no cuelgue la llamada. Bien pensado.
- **`call_tool` se desenvuelve antes de registrar** ([loop.py:220-224](../max_ai/capabilities/reasoning/react/loop.py#L220-L224)): Policy, aprobación y tracing ven la tool real.
- **`@tool`** ([tools/decorator.py](../max_ai/capabilities/tools/decorator.py), [function_as_tool.py](../max_ai/capabilities/tools/function_as_tool.py)): esquema desde los type hints, `ToolContext` inyectado, `ToolRetry`. Ya tienes la forma limpia de escribir tools; las tools nuevas de archivos no la usan (ver archivo 02).
- **Tests**: 647 pasando en 35 s es una red real para refactorizar.

---

## 2. La rama `feat/workspace-new-approach`

### 2.1 Está mal (comprobado)

**A. El test nuevo no importa.**
[tests/tools/test_sandbox_file_tools.py:7](../tests/tools/test_sandbox_file_tools.py#L7) importa `filesystem_tools`, que no existe en [filesystem/__init__.py](../max_ai/capabilities/tools/filesystem/__init__.py) (exporta `FileSystem`, `ReadFile`, `WriteFile`, `EditFile`).

```
ImportError: cannot import name 'filesystem_tools' from 'max_ai.capabilities.tools.filesystem'
```

Además el test describe otro contrato: rutas sin prefijo (`notes/a.txt`), `offset` desde 0, contenido sin numerar, tools construidas con el sandbox. El código actual pide `$SESSION/`, `offset` desde 1, líneas numeradas y saca el sandbox de `deps`. El test es de un diseño anterior.

**B. Hay dos scripts con contratos distintos.**

| | [filesystem/script.py](../max_ai/capabilities/tools/filesystem/script.py) | [filesystem/read/script.py](../max_ai/capabilities/tools/filesystem/read/script.py) |
|---|---|---|
| Operaciones | read, write, edit (campo `op`) | solo read |
| `offset` | desde 0 | desde 1 |
| Salida | texto crudo | líneas numeradas + `stamp` sha256 |
| Lo usa alguien | no | sí, `ReadFile` |

`resolve()` está copiada en los dos. Uno de los dos sobra.

**C. `session_id` permite salir del workspace.**
[`FileSystem.locate`](../max_ai/capabilities/tools/filesystem/_base.py#L21) usa `session_id` como carpeta raíz sin validarlo, y el script comprueba que la ruta no salga de *esa raíz*, no del workspace. Probado:

```
session_id=".."  +  path="$SESSION/setting.json"
→ (True, {'content': '     1\t{"secret": 1}', ...})
```

`$WORKSPACE/../setting.json` sí se rechaza; el hueco es solo por `session_id`. El session store valida ids con `^[A-Za-z0-9_-]{1,128}$` ([base/session_store.py:22](../max_ai/base/session_store.py#L22)), pero el `Agent` acepta cualquier `RunContext` sin pasar por el store. Arreglo: validar `session_id` con esa misma función en `locate`, y que el script reciba **dos** raíces: la del límite (`.agent/workspace`) y la de trabajo.

**D. Las reglas de la Policy dejan de aplicar con los prefijos.**
[`permission_subjects`](../max_ai/base/tools.py#L164) quita `./` y `workspace/`, no `$SESSION/`. Probado:

```
Policy(deny=["read_file(secrets/**)"])
  path="secrets/key"            → deny
  path="$SESSION/secrets/key"   → allow
```

Una regla `deny` que deja de negar en silencio es el peor tipo de fallo. `FileSystem` debe sobreescribir `permission_subjects` y devolver la ruta sin prefijo (y decidir si `$WORKSPACE/otra_sesion/secrets/x` también cuenta como `secrets/**`).

**E. `LocalSandbox.execute_command` revienta con salida binaria.**
[sandbox/local/layout.py:68](../max_ai/capabilities/sandbox/local/layout.py#L68) hace `stdout.decode()` estricto. `printf '\xff\xfe'` lanza `UnicodeDecodeError`. El contrato dice "solo problemas de infraestructura lanzan". Usa `decode(errors="replace")`.

**F. La configuración serializada no es portable.**
`LocalSandbox(root=".")` serializa la ruta absoluta de la máquina (`_to_config` devuelve `str(self.root.parent)`, [layout.py:31](../max_ai/capabilities/sandbox/local/layout.py#L31)). Un YAML guardado en tu máquina apunta a tu carpeta en la de otro. Guarda el `root` tal como lo recibiste.

### 2.2 Está incompleto o no conectado (lectura)

- `WriteFile.execute` y `EditFile.execute` lanzan `NotImplementedError`, y **están exportadas**. `FileSystem.all()` las devuelve; quien las registre tendrá tools que fallan al llamarlas.
- Las descripciones no coinciden: `read_file` dice que la ruta empieza con `$SESSION/`, `write_file` y `edit_file` dicen "relative to the workspace".
- `Agent(sandbox=...)` se guarda y se serializa pero **no se usa**: nadie pone `deps["sandbox"]`, así que `read_file` siempre responde "needs a sandbox". El `Agent` sigue registrando el `FileSystem` viejo ([agent.py:51](../max_ai/agents/agent.py#L51), [agent.py:236](../max_ai/agents/agent.py#L236)).
- `LocalSandbox`: `synchronize_environment`, `upload_file` y `get_or_update_skill` lanzan `NotImplementedError`. Como son abstractos, toda implementación está obligada a declararlos aunque no existan todavía.
- `Settings.create_global()` dice "The Agent calls it when it starts" ([config.py:77](../max_ai/config.py#L77)); nadie lo llama fuera de los tests.

### 2.3 Sobra en el árbol

- `max_ai/capabilities/tools/file_system copy/` — copia idéntica de `file_system/`, con espacio en el nombre.
- `max_ai/capabilities/tools/workspace/` — dos archivos vacíos.
- `root-test/` — un `.agent/setting.json` de prueba.
- `sandbox/docker/__init__.py` — solo un docstring.
- `bash.md`, `agent_update.md` (219 KB), `var/`, `.agents/` en la raíz.

Nada de esto rompe, pero `file_system` / `file_system copy` / `filesystem` / `workspace`, cuatro carpetas para lo mismo, es justo la sensación de desorden que querías quitar.

### 2.4 No hace sentido (diseño)

**G. Dos clases que se llaman `FileSystem`.**
`tools.file_system.FileSystem` (la vieja, con `get_toolset()`) y `tools.filesystem.FileSystem` (la nueva, base `CoreTool`). El `Agent` importa la vieja. Un import equivocado compila y hace otra cosa.

**H. `FileSystem.all()` con `__subclasses__()`.**
[_base.py:16](../max_ai/capabilities/tools/filesystem/_base.py#L16). Devuelve solo las subclases directas que ya se importaron; el orden depende del orden de import (salió `EditFile, ReadFile, WriteFile`), y cualquier subclase que alguien defina en un test o en su proyecto entra sola. Una lista explícita de tres elementos es más corta y no sorprende.

**I. Tres carpetas, seis archivos, para tres tools.**
`read/__init__.py` + `read/_base.py` + `read/script.py`, igual para write y edit. Cada `_base.py` repite el mismo `__init__`, el mismo esquema a mano y el mismo `execute` de 30 líneas (sacar sandbox, localizar, lanzar script, traducir errores). Solo cambia el nombre de la operación. Ver archivo 02.

**J. `SandboxCore` *es* un `Layout`.**
[base/sandbox.py:28](../max_ai/base/sandbox.py#L28) hereda de `Layout`. Con `LocalSandbox` funciona porque las rutas del host y las del sandbox son las mismas. Con Docker ya no: `sandbox.workspace` ¿es la ruta en el host o dentro del contenedor? Un sandbox *tiene* un layout (dentro) y el proyecto *tiene* otro (fuera). Composición: `sandbox.layout`.

**K. El constructor crea carpetas.**
`LocalSandbox.__init__` llama `create_project()`. Construir el objeto, deserializar un YAML o validar una config ya escribe `.agent/` en disco. Eso va en un `prepare()` / `connect()` explícito, que es donde el harness lo llamaría.

**L. `.agent/setting.json` está al alcance del modelo.**
`execute_command` corre con `cwd=.agent/workspace`; `cat ../setting.json` o `echo ... > ../setting.json` funcionan. Tu propia nota del 9 de noviembre ya lo dice: si de ese archivo salen permisos o skills, una inyección de prompt se los concede. En local no hay aislamiento que lo impida, así que lo que sea *de confianza* tiene que vivir fuera de `.agent/` (por ejemplo en `~/.max_ai/`) y `.agent/` debe tratarse como datos que el modelo puede cambiar.

**M. `execute_command` sin cancelación ni entorno limpio.**
No recibe `CancellationToken` (el `ExecutorBase.execute` viejo sí), y hereda todas las variables de entorno del proceso, incluidas las API keys del `.env`. Cualquier comando del modelo puede hacer `env`. Pasa un entorno mínimo y explícito.

**N. Lanzar el script entero en cada llamada.**
`python3 -c <script>` con el script completo como argumento funciona, pero: requiere `python3` dentro de todo sandbox, arranca un intérprete por cada lectura, y la línea de comando con el script queda en `ps` y en logs. Para local no hay ninguna razón para salir del proceso. Ver la propuesta de `FileOps` en el archivo 02.

---

## 3. Cosas del núcleo que no hacen sentido o están mal

### 3.1 El plan vive en el system prompt (costo real)

`_refresh_session_state` vuelve a renderizar las capas cuando cambia el plan ([base/reasoning.py:296-313](../max_ai/base/reasoning.py#L296-L313)) y el cliente arma un solo mensaje system con todas ([clients/openai/client.py:184-198](../max_ai/capabilities/clients/openai/client.py#L184-L198)). Cada `update_plan` cambia el primer mensaje → el proveedor no puede reutilizar el caché de **nada** de lo que sigue. En una tarea de 20 pasos con el plan actualizándose, pagas la conversación entera a precio completo casi en cada llamada.

Lo que hacen los harness de referencia: el plan no está en el system prompt. Vive en el resultado de la tool `update_plan` (ya queda en el transcript) y, si hace falta recordarlo, en un mensaje `[harness]` al final. El system prompt queda byte a byte igual durante toda la sesión. El resumen de compactación sí puede ir arriba: cuando compactas el caché ya se perdió.

Lo mismo aplica a `persistent_memories` y `current_date`: se renderizan por corrida, así que cambian entre turnos de la misma sesión.

### 3.2 Cuatro mecanismos para lo mismo

| Mecanismo | Dónde | Puede bloquear | Puede cambiar | Estado |
|---|---|---|---|---|
| `Policy` | dispatcher | sí (tool) | no | reglas |
| `CoreMiddleware` | loop + dispatcher | sí (tool, run) | sí | `mw.state()` |
| `CompletionBase` (gates) en `EventBus` | agent + loop | sí (fin de turno) | no | `ctx.completion_state` |
| `LoopGuard` | loop | no, solo steering | no | `loop_state.guard_state` |

Cuatro contratos, cuatro lugares donde guardar estado, cuatro formas de serializar, y el orden entre ellos está repartido entre `agent.py`, `loop.py` y `dispatcher.py`. Síntomas concretos:

- `NoProgressGuard` y `PlanCompletionGuard` existen como guards **y** su trabajo está en `RuntimeCompletionGate`; el docstring de `ReactLoop` tiene que explicar cuál se usa ([loop.py:162-167](../max_ai/capabilities/reasoning/react/loop.py#L162-L167)).
- `EventBus` se llama bus pero solo acepta gates, solo enruta 3 eventos y no deja suscribirse a nadie más ([events_bus/hooks.py](../max_ai/core/events_bus/hooks.py)). No es un bus: es el despachador de gates.
- `on_tool_request` del middleware solo ve las llamadas **ya aprobadas**. No hay forma de escribir "antes de pedir aprobación, revisa X" ni "cambia los argumentos".
- `ModelRequest.messages` existe pero cambiarlo no hace nada: el cliente recibe `ctx`, no `request.messages` ([reasoning.py:700-708](../max_ai/base/reasoning.py#L700-L708)). El docstring lo admite a medias ("may change tools and options"). Un campo que se puede escribir y se ignora es una trampa.

Esto es lo que resuelve el harness con hooks (archivo 03).

### 3.3 `ToolContext.deps` es un diccionario de strings

`deps` lleva `runtime_root`, `workspace_dir`, `skills_dir`, `filesystem_root`, `workspace_filesystem`, `run_context`, `tool_call_id`, `executor`, `execution_session`, y ahora `sandbox`. Cada tool hace `deps.get("...")` y comprueba `None`. Un typo en la clave es un `None` en ejecución. Tres de esas claves (`runtime_root`, `workspace_dir`, `skills_dir`) no las lee nadie.

Como `ToolContext` ya tiene `__slots__`, lo natural es que sean atributos: `ctx.sandbox`, `ctx.run`, `ctx.call_id`. Las tools dejan de defenderse de `None`.

### 3.4 Las capas se importan al revés

`base/` y `core/` deberían no saber de `capabilities/`. Hoy:

- [types/run_context.py:11](../max_ai/types/run_context.py#L11) y [core/event_type.py:17](../max_ai/core/event_type.py#L17) importan `capabilities.tools.plan.AgentPlan`.
- [core/stacks/container.py:10-16](../max_ai/core/stacks/container.py#L10-L16) importa siete capas concretas.
- [core/model/agent.py:7](../max_ai/core/model/agent.py#L7) importa configs de MCP.
- `base/context.py`, `base/knowledge.py`, `base/memory.py` importan `FunctionAsTool`.
- `base/workspace.py` importa `UserFileSystem` de la implementación local.

Funciona por imports perezosos y orden de carga. `AgentPlan` es un tipo de datos: va en `types/`. `FunctionAsTool` y `@tool` son el contrato público de tools: van en `base/`.

### 3.5 Nombres de tools escritos a mano

Ya lo tenías anotado para `ask_user`. Siguen:

- `"update_plan"` en [loop.py:663](../max_ai/capabilities/reasoning/react/loop.py#L663) y [guards.py:293](../max_ai/capabilities/reasoning/guards.py#L293).
- `"bash"` y `"write_file"` en el gate ([completion_gate/gate.py:37,69,89,93](../max_ai/capabilities/completion_gate/gate.py#L37)) y en [agent.py:253](../max_ai/agents/agent.py#L253).
- `edit_file`, `update_plan`, `call_tool` dentro de los textos de [harness/messages.py](../max_ai/core/harness/messages.py).

`BashTool(name="shell")` es un parámetro público y el gate dejaría de ver sus fallos sin avisar. Lo que el gate quiere saber no es el nombre sino el **tipo**: "¿esta tool corre comandos?" (`tool.runs_commands`), "¿escribe archivos?". Pregunta por la capacidad, no por el string.

### 3.6 El loop importa una tool concreta

[loop.py:55](../max_ai/capabilities/reasoning/react/loop.py#L55) importa `AskUserTool` para decidir qué tools ofrecer tras una negación. El loop ya tiene el dato genérico: `tool.waits_for_user`. La regla "después de una negación solo puedes preguntar o cerrar el plan" es política del harness, no del loop.

### 3.7 Código muerto o engañoso

- `build_tool_definition` ([base/tools.py:230](../max_ai/base/tools.py#L230)) arma `read_file_v1.0.0` — con puntos, que tu propio `tool_name_pattern` rechaza. Nadie lo llama.
- `_check_type` y `_JSON_TYPE_MAP` en `CoreTool`: nadie los usa; la validación es `Draft202012Validator`.
- `CoreRuntimeTool`: una clase marcadora vacía; el comportamiento real lo da `runs_commands`. Dos formas de decir lo mismo.
- `paused_in_call` en el loop ([loop.py:426-449](../max_ai/capabilities/reasoning/react/loop.py#L426-L449)): `_call_llm` nunca emite `ToolApprovalEvent` (solo el dispatcher lo crea), así que esa rama no se ejecuta.
- `hasattr(loop_state, "guard_state")` ([agent.py:559](../max_ai/agents/agent.py#L559)): `BaseLoopState` siempre lo tiene.
- `RunContext.file_reads` con `(mtime_ns, size)` pertenece al diseño viejo; el nuevo usa `stamp` sha256. Uno de los dos se va.
- `version="1.0.0"` en cada tool: no afecta nada visible.

### 3.8 Estado en el objeto de la tool

`BashTool._cwd` ([bash/_tool.py:116](../max_ai/capabilities/tools/bash/_tool.py#L116)) es un diccionario en memoria por `(user, conversation)` que nunca se limpia y se pierde al reiniciar. Rompe la regla que el resto del framework cumple ("el Agent no guarda estado por corrida") y hace que un agente always-on o con dos procesos pierda el directorio. Va en `RunContext` (ya lo tenías como idea: `cwd` y `workspace_root` como punteros).

### 3.9 `prepare()` por corrida que escribe en el Agent

`_drive_connected` llama `self.prepare()` en **cada** run, y `prepare()` hace `self._skill_blocks = await self.skills.get_skills()` ([agent.py:425-430](../max_ai/agents/agent.py#L425-L430)): estado compartido reescrito por corridas concurrentes, más `executor.prepare()`, `workspace.download`, `materialize` y `skills.materialize` cada vez. Para una sesión es "al abrirla", no "en cada mensaje".

### 3.10 Razones de fin de turno como strings sueltos

`"approval_needed"`, `"input_needed"`, `"waiting"`, `"tool_denied"` están como constantes en `loop.py`, como literales en [agent.py:620-625](../max_ai/agents/agent.py#L620-L625) y como `Literal` en `types/agent_response.py`. Tres copias que hay que mantener iguales. Un solo `Enum` en `types/`.

### 3.11 Un timeout para todo

`tool_timeout_seconds = 360` (descrito como "max seconds one bash command may run") es también el límite de cualquier `@tool`, de MCP y, +30, la red del dispatcher. Una búsqueda en memoria que tarda 6 minutos no debería ser normal; un subagente necesitará mucho más. El timeout es propiedad de la tool, con ese valor como tope.

### 3.12 Un gate que lanza tumba la corrida

`EventBus.emit` llama a los handlers sin `try` ([bus.py:78-85](../max_ai/core/events_bus/bus.py#L78-L85)). Un observador con un bug detiene el turno. Los observadores no deberían poder romper nada; solo los que deciden.

### 3.13 Documentación que estorba

`base/reasoning.py` tiene docstrings generados del tipo "Return the tool dispatcher bound to this loop. Returns ------- ToolDispatcher: The tool dispatcher bound to this loop." en cada property. Son ~150 líneas que no dicen nada y esconden los comentarios buenos (que los hay, y muy buenos). Lo mismo en `base/tools.py` y `base/middleware.py`.

---

## 4. Resumen por gravedad

| Gravedad | Punto |
|---|---|
| Seguridad | 2.1-C (`session_id`), 2.1-D (Policy con prefijos), 2.4-L (`setting.json`), 2.4-M (entorno heredado) |
| Rompe | 2.1-A (test), 2.1-E (decode), 2.2 (`write`/`edit` sin implementar pero exportadas) |
| Costo | 3.1 (plan en el system prompt) |
| Diseño | 3.2 (cuatro mecanismos), 3.3 (`deps`), 3.4 (capas), 2.4-J (sandbox es Layout) |
| Limpieza | 2.3, 3.5, 3.7, 3.10, 3.13 |
