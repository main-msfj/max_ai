# 03 — De `Agent` a `Harness`

## Qué es un harness (y qué no)

Un agente tipo Claude Code, Codex u opencode tiene tres piezas con dueños distintos:

| Pieza | Qué es | Cambia cuando… |
|---|---|---|
| **Definición del agente** | nombre, instrucciones, modelo, qué tools, qué permisos | quieres otro agente |
| **Loop** | llamar al modelo → correr tools → repetir | casi nunca |
| **Harness** | todo lo demás: sesiones, entorno, registro de tools, permisos, hooks, contexto, compactación, steering, eventos, persistencia | quieres otro *producto* |

La definición es **datos**. El loop es **un algoritmo corto**. El harness es **el runtime** y es dueño de todo lo que tiene ciclo de vida.

Hoy en max_ai las tres piezas están dentro de `Agent`:

```
Agent.__init__          registra tools, crea MCP manager, EnvironmentManager,
                        dispatcher, gates, bus, locks, prompt stack
Agent._drive            conecta MCP, baja el workspace
Agent._drive_connected  prepare, materialize, estado de pausa, ToolContext,
                        bind del loop, middleware, usage, AgentResponse
Agent.run_stream_events lock de sesión, validación de pendientes, limpiar estado, cola
Agent.close             espera, desconecta todo
```

`Agent` **es** el harness. Por eso se siente que "falta harnessing": no falta, está escondido en una clase que también es la definición, y repartido con el loop (steering, compactación, guards) y el dispatcher (policy, middleware).

## La forma objetivo

```
AgentSpec ──────────────┐  (ya existe: core/model/agent.py)
  datos, serializable   │
                        ▼
                   ┌─────────┐   open(user, session)   ┌──────────┐
                   │ Harness │ ──────────────────────► │ Session  │
                   └─────────┘                         └──────────┘
   dueño de:                                  dueño de:
   · cliente(s) del modelo                    · RunContext (cargar/guardar)
   · ToolRegistry + MCP                       · sandbox de esta sesión
   · Policy (settings + agente)               · cwd, file_stamps, plan
   · HookRegistry                             · lock de la sesión
   · SessionStore                             · send(msg) / resume() / interrupt()
   · SandboxProvider
   · prompt stack
                                 Session.send()
                                       │
                                       ▼
                               loop.run(turn)      ← solo: modelo → tools → repetir
                                       │
                         cada paso pasa por harness.hooks
```

Uso:

```python
harness = Harness(spec, hooks=[...], store=LocalSessionStore(), sandbox=LocalSandbox())
async with harness.session(user_id="u1", session_id="s1") as session:
    async for event in session.send("arregla el test"):
        ...
```

Y `Agent` queda como la fachada corta de siempre, para no romper los ejemplos:

```python
class Agent:
    def __init__(self, name, instructions, client, **options):
        self.harness = Harness(AgentSpec(...), ...)
    async def run(self, task, run_context=None):
        async with self.harness.session_for(run_context) as s:
            return await s.run(task)
```

### Por qué `Session` es la pieza que hoy falta

Hoy la "sesión" no existe como objeto: es una tupla `(user_id, session_id)` que se pasa a todos lados, y cada componente guarda su parte por su cuenta:

| Estado de sesión | Dónde vive hoy |
|---|---|
| mensajes, plan, tool_state | `RunContext` (bien) |
| lock | `Agent._session_locks` |
| contenedor / lease | `EnvironmentManager._entries` |
| cwd de bash | `BashTool._cwd` |
| archivos leídos | `RunContext.file_reads` / stamps nuevos |
| estado del loop en pausa | `ctx.runtime_state.shared_state["reasoning_loop"]` |
| guardar/cargar | "el host lo hace" ([session_store.py:3](../max_ai/base/session_store.py#L3)) |

Con `Session`, abrir es: cargar del store → tomar lock → preparar sandbox → sincronizar skills. Cerrar es lo inverso. `prepare()`, `download`, `materialize` dejan de correr en cada mensaje. Y multi-agent / always-on reciben gratis un objeto al que mandarle mensajes.

## Hooks: un solo mecanismo en vez de cuatro

Los hooks de un harness son **puntos con nombre** donde código externo puede observar, cambiar o bloquear. La diferencia con tu middleware actual no es técnica, es de cobertura y de uniformidad: todos los puntos, una sola forma de respuesta.

### Los puntos

| Hook | Cuándo | Puede |
|---|---|---|
| `session_start` | al abrir o reanudar sesión | añadir contexto |
| `user_message` | llega un mensaje del usuario | añadir contexto, bloquear |
| `pre_model` | antes de cada llamada al modelo | cambiar tools/opciones, añadir contexto |
| `post_model` | respuesta del modelo completa | cambiarla |
| `pre_tool` | antes de **decidir** permiso | permitir / pedir / negar, cambiar argumentos |
| `post_tool` | la tool terminó | cambiar resultado, añadir contexto |
| `stop` | el modelo no pidió tools (quiere terminar) | bloquear con razón → el loop sigue |
| `pre_compact` | antes de compactar | añadir instrucciones al resumen |
| `subagent_stop` | un subagente terminó | bloquear con razón |
| `session_end` | al cerrar | solo observar |
| `notification` | el harness necesita al usuario (aprobación, pregunta) | solo observar |

### Una sola forma de respuesta

```python
@dataclass
class HookResult:
    decision: Literal["continue", "allow", "ask", "deny", "block"] = "continue"
    reason: str | None = None           # se le dice al modelo (deny/block)
    context: str | None = None          # se añade como mensaje [harness]
    update: dict | None = None          # argumentos / resultado cambiados

class Hooks:                            # heredas y sobreescribes lo que necesites
    matcher: str = "*"                  # "bash", "write_file|edit_file", "mcp_*"
    async def pre_tool(self, e: ToolEvent) -> HookResult | None: ...
    async def post_tool(self, e: ToolEvent) -> HookResult | None: ...
    async def stop(self, e: StopEvent) -> HookResult | None: ...
    ...
```

Reglas de combinación, las mismas que ya usa tu Policy: `deny` gana a `ask`, `ask` a `allow`; los `context` se concatenan; un hook que lanza se registra y no tumba la corrida (salvo que esté marcado como obligatorio).

### Lo que ya tienes se convierte en hooks

Esta es la parte importante: no pierdes nada, lo reubicas.

| Hoy | Como hook | Qué ganas |
|---|---|---|
| `Policy.decide()` | hook `pre_tool` incorporado, corre primero | las reglas del usuario y las de código se combinan igual |
| `RuntimeCompletionGate` y `CompletionBase` | hooks `stop` (+ `post_tool` para juntar evidencia) | desaparecen `EventBus`, `EVENT_HOOKS`, `completion_bus` |
| `SchemaRetryGuard`, `RepetitionGuard` | `post_tool` que devuelve `context` | desaparece `LoopGuard` y `guard_state` |
| `BudgetGuard` | `pre_model` que devuelve `context` | |
| `NoProgressGuard`, `PlanCompletionGuard` | `stop` | se acaba la duplicidad gate/guard |
| regla "tras negar, solo ask_user/update_plan" | `pre_model` que restringe `tools` | el loop deja de importar `AskUserTool` |
| middleware `on_tool_request/response` | `pre_tool` / `post_tool` | ahora ve la llamada **antes** de la aprobación |
| middleware `on_model_*`, `on_run_*` | `pre_model` / `post_model` / `session_*` | |
| tracing, logging, budget | hooks que solo observan | |

El estado por corrida de cualquier hook va en un solo lugar: `ctx.hook_state[hook.key]` (hoy son tres: `mw.state()`, `completion_state`, `guard_state`).

`ReactLoop` baja de ~700 líneas a lo que de verdad es suyo: llamar al modelo, registrar llamadas, despachar, decidir si sigue. Todo `if` sobre negaciones, guards, gates y steering se vuelve "pregúntale al harness".

### Hooks desde configuración

Una vez que existen en código, exponerlos en `setting.json` es poco trabajo y es lo que los hace útiles para quien no programa:

```json
{ "hooks": { "post_tool": [ { "matcher": "write_file|edit_file", "command": "ruff check --fix $FILE" } ] } }
```

El comando recibe el evento en JSON por stdin; código de salida 0 = sigue, 2 = bloquea y stderr va al modelo. Recuerda el punto 2.4-L del archivo 01: este archivo ejecuta comandos, así que **no** puede ser el `.agent/setting.json` que el modelo alcanza con bash. Solo el global, o uno de proyecto fuera de `.agent/`.

## Contexto: quién escribe qué en el prompt

Otra responsabilidad que hoy está repartida (capas en `Agent`, `_refresh_session_state` en el loop, steering transitorio en `_model_context`). En el harness, una sola regla:

| Zona | Contenido | Cambia |
|---|---|---|
| **System prompt** | identidad, reglas, lista de tools, skills, entorno | nunca dentro de una sesión |
| **Mensajes** | conversación, resultados de tools | solo se añade al final |
| **Recordatorios `[harness]`** | plan actual, memoria nueva, fecha, steering de hooks | se añaden al final, como mensaje |

Consecuencias:

- el caché del proveedor se conserva toda la sesión (arregla el punto 3.1 del archivo 01);
- el steering **se persiste** en el transcript. Hoy es transitorio: el modelo reacciona a una instrucción que en la siguiente llamada ya no está, y al depurar en Langfuse ves la reacción sin la causa. Guardado como mensaje `[harness]` con una marca para que la UI lo oculte, el historial es reproducible. La compactación puede descartarlos primero.

## Plan de migración, sin big-bang

Cada paso deja los 647 tests verdes y se puede commitear solo.

1. **`ToolContext` con atributos.** `ctx.sandbox`, `ctx.run`, `ctx.call_id` en vez de `deps[...]`. Mecánico.
2. **Extraer `Harness`.** Mover de `Agent` a `Harness` el registro de tools, MCP, dispatcher, prompt stack y `_drive_connected`. `Agent` delega. Cero cambio de comportamiento.
3. **`Session`.** Mover el lock, el estado de pausa, `cwd`, stamps y el lease del entorno. `run_stream_events` se convierte en `session.send`. Aquí el harness empieza a hablar con el `SessionStore`.
4. **`HookRegistry` con `pre_tool` / `post_tool`.** Policy pasa a ser el primer hook. El middleware de tools se adapta con un envoltorio.
5. **`stop`.** Gates y guards de fin de turno pasan a hooks. Se borra `EventBus`.
6. **`pre_model` / `post_model`.** Guards de steering y middleware de modelo. Se borra `LoopGuard`.
7. **Plan y recordatorios fuera del system prompt.**
8. **Sandbox nuevo dentro de `Session`**, reemplazando `EnvironmentManager` + `WorkspaceBase` + `ExecutorBase`.

Los pasos 1–3 son mover código. Los pasos 4–6 son donde se borra más de lo que se escribe.

## Qué no meter en el harness

- **Lógica de producto de la CLI** (render, atajos). El harness emite eventos; la CLI los pinta.
- **El loop en sí.** Debe seguir siendo reemplazable (`reasoning=`), y cuanto menos sepa, más fácil es escribir otro.
- **Conocimiento de tools concretas.** El harness pregunta por capacidades (`runs_commands`, `waits_for_user`, `read_only`), nunca por nombres.
