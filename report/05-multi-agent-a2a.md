# 05 — Guía: multi-agent y A2A

Son dos cosas distintas que conviene no mezclar:

- **Multi-agent**: varios agentes *dentro de tu harness*. Comparten proceso, registro de tools, política, store.
- **A2A**: un protocolo para que tu agente hable con agentes *de otros*, por red, sin compartir nada.

Construye primero multi-agent. A2A termina siendo un adaptador de 200 líneas encima.

---

# Parte 1 — Multi-agent

## El patrón que sí funciona: subagente como tool

De todos los esquemas (grafo, handoff, swarm, debate), el que usan los harness de producción es el más simple:

> El agente principal llama a una tool `agent(prompt, type)`. El harness abre **una sesión nueva con contexto vacío**, corre un agente hasta que termina, y devuelve **solo su mensaje final** como resultado de la tool.

Por qué funciona:

- **Aísla contexto.** El subagente lee 40 archivos y gasta 100k tokens; al principal le llegan 300 tokens con la conclusión. Es la razón número uno para usar subagentes: no es paralelismo, es proteger la ventana del principal.
- **No hay protocolo nuevo.** Para el principal es una tool más: pasa por Policy, hooks, tracing.
- **El transcript del principal sigue siendo lineal.**

El costo: el subagente arranca sin saber nada. El prompt que le escribe el principal tiene que traer todo el contexto, y eso hay que decírselo al modelo en la descripción de la tool.

## Sobre tu framework

Con `Harness` y `Session` (archivo 03) es casi directo, porque el `Agent` ya no guarda estado por corrida:

```python
@tool
async def agent(ctx: ToolContext, description: str, prompt: str, agent_type: str = "general") -> str:
    """Launch another agent on a self-contained task. It starts with no memory of
    this conversation: put everything it needs in ``prompt``. Returns its final answer."""
    spec = ctx.harness.agents[agent_type]                 # AgentSpec registrado
    child = ctx.harness.child_session(
        spec,
        parent=ctx.session,
        session_id=f"{ctx.session_id}-{short_id()}",
    )
    async with child:
        response = await child.run(prompt, cancellation=ctx.cancellation)
    return response.final_text()
```

### Las decisiones que hay que tomar

| Decisión | Recomendación | Por qué |
|---|---|---|
| **Tools del hijo** | Las que diga su `AgentSpec`, y **nunca** la tool `agent` salvo permiso explícito | Sin tope de profundidad, un modelo delega para siempre. Empieza con profundidad 1. |
| **Permisos** | La Policy del hijo = la del padre ∩ la suya. Lo que el usuario permitió en la sesión (`allowed_rules`) se hereda. | Un hijo no puede tener más permisos que quien lo lanzó. |
| **Aprobaciones** | Suben al padre: el evento `ToolApprovalEvent` del hijo se reemite con `agent_path=["main","reviewer"]` | Hoy una aprobación pausa el turno. En un hijo tiene que pausar al hijo y al padre a la vez, y al reanudar, bajar. Es lo más delicado de todo. |
| **Workspace** | Mismo `$WORKSPACE`, `$SESSION` propio (o el del padre si el hijo debe ver sus archivos); `$TMP` propio | Con lectores no importa. Con dos escritores en paralelo, ver "worktrees" abajo. |
| **`ask_user`** | El hijo no la tiene. Si le falta información, lo dice en su respuesta y el padre decide. | Una persona no debería recibir preguntas de un agente que no sabe que existe. |
| **Eventos** | Todos los eventos del hijo salen por el stream del padre con `agent_path` | La CLI puede anidar; el tracing queda como árbol. |
| **Persistencia** | La sesión del hijo se guarda con `parent_session_id` | Depurar, y poder **continuar** al mismo hijo (ver abajo). |
| **Timeout** | Propio de la tool, no los 360 s globales | Un subagente tarda minutos. |
| **Presupuesto** | Los tokens del hijo cuentan en el `Usage` del padre | Si no, el costo real queda escondido. |
| **Cancelación** | El token del padre cancela al hijo | Ya tienes `CancellationToken.link_future`. |

### Paralelo

Tu dispatcher ya corre en paralelo las llamadas `read_only` contiguas. Un subagente de exploración es `read_only=True`: tres llamadas a `agent` en el mismo mensaje corren a la vez **sin escribir código nuevo**. Para subagentes que escriben, el dispatcher los pone en serie, que es lo seguro.

### Escritores en paralelo: worktrees

Si dos subagentes editan el mismo repo a la vez, se pisan. La solución que ya anotaste en tu diseño del workspace es la correcta: cada hijo recibe un `git worktree` propio como `$SESSION`. Por eso "crear la carpeta de la sesión" debe ser un método reemplazable del sandbox. No lo hagas ahora; solo no cierres la puerta.

### Continuar a un hijo

Como la sesión del hijo se guardó, una segunda tool permite seguir hablándole con su contexto intacto:

```python
@tool
async def send_message(ctx, agent_id: str, message: str) -> str:
    """Continue a previously launched agent; it keeps its context."""
```

Esto convierte "subagente desechable" en "compañero", y es la base de equipos de agentes. Segundo paso, no primero.

### En segundo plano

`agent(..., run_in_background=True)` devuelve un id de inmediato y, al terminar, mete un evento `agent:<id>` en el mailbox de la sesión padre. Es **exactamente** la pieza 4 del archivo 04. Multi-agent en paralelo real y always-on son la misma infraestructura.

### Hooks

`subagent_start` y `subagent_stop`. El segundo puede bloquear: "el revisor terminó sin correr los tests → sigue". Es tu `RuntimeCompletionGate` aplicado al hijo.

## Patrones encima de la tool

Con solo `agent` + `send_message` salen los patrones conocidos, sin clases nuevas:

- **Orquestador / trabajadores**: el principal planifica, lanza N exploradores en paralelo, sintetiza.
- **Especialistas**: `AgentSpec` distintos (`explore` solo lectura y modelo barato, `plan` sin escritura, `review`). Ya los puedes escribir en YAML con `Agent.from_yaml`.
- **Verificador**: tras terminar, un hijo con contexto limpio revisa el resultado. Funciona mejor que pedirle al mismo agente que se revise.
- **Handoff**: el hijo responde directo al usuario. Es el mismo mecanismo más un indicador "tu respuesta es la final". Úsalo poco: rompe la linealidad del transcript.

## Qué evitar

- **Grafos declarativos** de agentes como primer diseño. Mucha superficie, y el modelo decide mejor a quién delegar que un grafo fijo.
- **Compartir el transcript** entre agentes. Se pierde la ventaja principal.
- **Agentes que conversan entre sí sin tope.** Siempre un máximo de vueltas.
- **Delegar por defecto.** Cada subagente arranca en frío: cuesta. La descripción de la tool debe decir cuándo *no* usarla.

## Orden

1. `Harness.child_session` + tool `agent`, síncrona, profundidad 1, sin `ask_user`.
2. Eventos con `agent_path` y `Usage` sumado.
3. Aprobaciones que suben y bajan.
4. Registro de `AgentSpec` por nombre (carpeta `.agent/agents/*.yaml`).
5. `send_message`.
6. Segundo plano (necesita mailbox).
7. Worktrees.

---

# Parte 2 — A2A (Agent2Agent)

## Qué es

Un protocolo abierto (lo empezó Google, hoy está en la Linux Foundation) para que agentes de distintos frameworks y empresas colaboren **como agentes**, no como tools. La diferencia con MCP:

| | MCP | A2A |
|---|---|---|
| Expone | tools, recursos, prompts | un agente completo |
| El otro lado es | una función: entrada → salida | opaco: tiene su propio modelo, estado y criterio |
| Duración | una llamada | una **tarea** que puede durar, pedir más datos, emitir avances |
| Tú ya lo tienes | sí, como cliente | no |

Se complementan: usas MCP para darle manos a tu agente y A2A para que hable con colegas.

> El protocolo cambia entre versiones (nombres de métodos, ruta de la tarjeta, transporte). Lo de abajo es el modelo conceptual, que es estable. Antes de escribir el adaptador, lee la especificación vigente en `a2a-protocol.org` y usa el SDK oficial de Python (`a2a-sdk`) en vez de implementar el cable a mano.

## Los conceptos

- **Agent Card**: un JSON público (en una ruta `/.well-known/…`) que dice quién es el agente, su URL, qué *skills* ofrece, qué formatos acepta, si hace streaming, cómo autenticarse.
- **Task**: la unidad de trabajo. Tiene id y un estado: enviada → trabajando → *necesita entrada* → completada / fallida / cancelada / rechazada.
- **Context**: un id que agrupa varias tareas de la misma conversación.
- **Message**: un turno, con rol (`user` / `agent`) y **partes** (texto, archivo, datos estructurados).
- **Artifact**: el resultado que produce la tarea (un documento, un archivo), también en partes.
- **Transporte**: JSON-RPC sobre HTTP; streaming por SSE; notificaciones push por webhook para tareas largas.
- **Operaciones**: enviar mensaje, enviar con streaming, consultar tarea, cancelar tarea, suscribirse de nuevo.

## Por qué encaja con lo que ya tienes

Tu modelo de corrida es casi el de A2A con otros nombres:

| A2A | max_ai |
|---|---|
| `contextId` | `session_id` |
| `taskId` | `run_id` |
| estado *working* | turno en curso |
| estado *input-required* | `finish_reason` = `input_needed` / `approval_needed` |
| estado *completed* | `stop` |
| estado *failed* / *canceled* | error / `CancellationToken` |
| Message con partes | `UserMessage` / `AssistantMessage` con bloques |
| Artifact | archivos en `$SESSION` + la respuesta final |
| streaming de estado | `run_stream_events` |
| Agent Card | `AgentSpec` (nombre, descripción) + skills |
| cancelar tarea | `CancellationToken.cancel()` |

Que ya tengas pausa/resume sobre datos es lo que hace esto barato: *input-required* es literalmente tu `resume()`.

## Lado servidor: exponer tu agente

```
cliente A2A ──HTTP──► adaptador A2A ──► Harness.session(user, contextId) ──► session.send()
                          │                                                      │
                          └──────◄── traduce CoreEvent → eventos de tarea ◄──────┘
```

El adaptador hace cinco cosas:

1. **Tarjeta**: se genera desde `AgentSpec` y la lista de skills. No la escribas a mano.
2. **Enviar mensaje**: `contextId` → `session_id` (uno nuevo si no viene); crea `run_id` = `taskId`; llama `session.send`.
3. **Traducir el final del turno**:
   - `stop` → tarea *completed*, con la respuesta como artifact;
   - `input_needed` → *input-required*, con la pregunta de `ask_user` como mensaje;
   - `approval_needed` → **decisión tuya** (abajo).
4. **Segundo mensaje con el mismo `taskId`** → es la respuesta: aplicarla al `ToolCallRecord` pendiente y `resume()`.
5. **Streaming**: `ModelStreamChunkEvent` y `ToolCallEvent` → actualizaciones de estado; archivos escritos → actualizaciones de artifact.

Esto es un paquete aparte (`max_ai.server.a2a`) que depende del harness. El núcleo no sabe que A2A existe.

### Lo que hay que decidir

- **Identidad.** `user_id` sale de la autenticación del cliente, nunca del cuerpo del mensaje. Cada cliente remoto es un usuario con su propio sandbox.
- **Aprobaciones.** El que llama es otro agente, no una persona. Opciones: (a) Policy estricta para sesiones A2A y lo que pediría aprobación se **niega**; (b) se traduce a *input-required* y el otro lado decide. Empieza con (a): un agente remoto aprobando tus `bash` es un riesgo.
- **Qué expones.** No todo el agente: declara skills concretos en la tarjeta y usa una Policy específica para lo que entra por A2A.
- **El texto que llega no es de confianza.** Es contenido de otro sistema. Sandbox aislado obligatorio (no `LocalSandbox`), y nada de memoria compartida con tus usuarios reales.
- **Tareas largas.** Si el cliente se desconecta, la tarea sigue y se consulta después. Necesitas guardar el `RunContext` durante el turno y el supervisor del archivo 04.

## Lado cliente: tu agente llama a otro

Aquí está la parte elegante: un agente remoto es **un subagente cuya sesión vive en otra máquina**. Usa la misma tool de la Parte 1.

```python
class RemoteAgent:
    """Un agente A2A detrás de la misma interfaz que child_session."""
    def __init__(self, card_url: str): ...
    async def run(self, prompt: str, *, context_id: str | None, cancellation) -> AgentResult:
        # enviar mensaje → esperar/stream hasta estado final → devolver texto + artifacts
```

```python
harness.agents["legal_review"] = RemoteAgent("https://legal.acme.com/.well-known/...")
# el modelo llama: agent(prompt="...", agent_type="legal_review")
```

Para el modelo no hay diferencia entre un colega local y uno remoto. Para ti:

- la tarjeta remota da la descripción que ve el modelo (trátala como texto no confiable: no la pegues en el system prompt sin límite de tamaño);
- *input-required* del remoto → el resultado de la tool es la pregunta, y tu agente decide si la contesta él o se la pasa al usuario con `ask_user`; luego `send_message` con el mismo `taskId`;
- los artifacts que devuelva se guardan en `$TMP` y la tool responde con las rutas;
- la Policy aplica igual: `agent(legal_review)` puede ser `ask`, porque **enviar datos a un tercero es publicarlos**.

## Orden

1. Multi-agent local (Parte 1, pasos 1–4). Sin esto no hay dónde enchufar el cliente.
2. `RemoteAgent` cliente sobre el SDK oficial. Es lo más corto y lo primero útil.
3. Servidor: tarjeta + enviar mensaje sin streaming + consultar tarea.
4. Streaming.
5. *input-required* de ida y vuelta.
6. Notificaciones push (necesita el supervisor de always-on).

## Cómo se relaciona todo

```
                    Harness + Session + Hooks          (archivo 03)
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              subagentes   always-on    A2A
              (sesión      (mailbox +   (sesión detrás
               hija)        supervisor)  de HTTP)
                    └──────────┼──────────┘
                               ▼
             las tres son "alguien que no es el usuario
              le manda un mensaje a una Session"
```

Por eso el harness va primero: las tres cosas que quieres construir después son la misma operación con distinto remitente.
