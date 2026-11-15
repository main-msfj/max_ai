# 04 — Agentes always-on: cómo funcionan y qué le falta al tuyo

Entendí tu pregunta como "agentes **always on**": los que no esperan a que alguien escriba, sino que siguen vivos, reaccionan a eventos y trabajan solos durante horas o días. Si te referías a otra cosa, dime y lo rehago.

## La idea central

Un agente always-on **no es un proceso que piensa sin parar**. Es:

> una sesión guardada en disco + una cola de cosas que la despiertan + un proceso que, al despertar, corre **un turno normal** y vuelve a dormir.

El modelo no "está encendido". Lo que está encendido es un **supervisor** barato que espera eventos. Cada despertar es exactamente el mismo `session.send(mensaje)` que una conversación de chat; lo único distinto es quién escribió el mensaje.

```
   disparadores                     supervisor                      turno
 ┌──────────────┐              ┌──────────────────┐         ┌────────────────┐
 │ usuario      │              │                  │         │ cargar sesión  │
 │ cron / timer │──► cola ────►│ toma 1 evento    │────────►│ loop normal    │
 │ webhook      │   (durable)  │ por sesión       │         │ guardar sesión │
 │ otro agente  │              │                  │◄────────│ ¿programó algo?│
 │ tarea termina│              └──────────────────┘         └────────────────┘
 └──────────────┘                      ▲                            │
                                       └──── schedule_wakeup ───────┘
```

## Las siete piezas

### 1. Sesión durable
Todo el estado en datos, guardado después de cada paso que importa. Si el proceso muere, otro lo retoma.

### 2. Cola de entrada por sesión (mailbox)
Cada evento es un mensaje con origen: `user`, `timer`, `webhook`, `agent:<nombre>`, `task:<id>`. Entran aunque la sesión esté trabajando. Un evento a la vez por sesión.

### 3. Disparadores
- **Tiempo**: "despiértame en 20 minutos", "todos los días a las 9".
- **Externos**: webhook, mensaje en una cola, archivo que cambió.
- **Internos**: una tarea en segundo plano terminó, un subagente respondió.

### 4. Trabajo en segundo plano
Una tool puede devolver "empezó, id=42" y el turno termina. Cuando el trabajo acaba, entra un evento `task:42` a la cola y la sesión despierta. Sin esto, un `npm test` de 15 minutos mantiene un turno abierto y bloqueado.

### 5. Tools para que el modelo maneje su propio tiempo
`schedule_wakeup(delay, reason)`, `cron_create(...)`, `task_status(id)`, `task_stop(id)`. El modelo decide cuándo volver a mirar.

### 6. Contexto que no crece sin límite
Compactación, más memoria en archivos: notas que el agente escribe para su yo futuro, porque tras compactar o reiniciar no recordará. Sin esto el agente se degrada en horas.

### 7. Límites y supervisión
Presupuesto por día, tope de despertares por hora, detección de bucles ("tres despertares sin hacer nada → dormir más"), un registro que una persona pueda leer, y una forma de pararlo.

## Qué tienes y qué falta

| Pieza | Estado en max_ai | Qué falta |
|---|---|---|
| Sesión durable | **Lo tienes.** `RunContext` serializable, `CoreSessionStore` local y Mongo, pausa/resume | Guardar *durante* el turno (hoy "el host" guarda al final; si muere a mitad, se pierde el turno). Versionado optimista del store (ya está en tu roadmap). |
| Mailbox | **A medias.** `RunContext.inbox` existe y el loop lo lee | Vive dentro del `RunContext` en memoria: solo sirve si el mismo proceso tiene la sesión abierta. Tiene que ser una cola durable aparte, con origen del mensaje. |
| Un evento a la vez por sesión | **Lo tienes** en un proceso (`_session_locks`) | Con dos procesos, nada lo impide. Un lease en el store. |
| Disparadores | **No existe** | Scheduler + punto de entrada para eventos externos. |
| Trabajo en segundo plano | **No existe.** Todo corre dentro de la tool, con tope de 360 s | Registro de tareas en `Session`; `bash(run_in_background=True)`; evento al terminar. |
| Tools de tiempo | **No existe** | `schedule_wakeup`, cron. |
| Contexto | **Lo tienes.** Compactación (ventana y resumen), memoria | Notas en archivos que el harness recarga al despertar. |
| Límites | **A medias.** Middleware de budget, `max_loop_iterations`, guards | Presupuesto *por sesión a lo largo del tiempo*, no por turno. |
| Supervisor | **No existe** | El proceso que espera eventos y abre sesiones. |
| Aprobaciones sin persona delante | **A medias.** El turno pausa con `approval_needed` | Notificar a alguien (hook `notification`), y decidir qué pasa si nadie contesta: esperar, negar, o seguir con otra cosa. |

La buena noticia: lo difícil (estado en datos, transcript siempre válido, pausa/resume) ya está. Lo que falta es infraestructura alrededor, y casi todo cae solo cuando exista `Session` (archivo 03).

## Diseño mínimo sobre el harness

```python
class Supervisor:
    def __init__(self, harness: Harness, mailbox: Mailbox, scheduler: Scheduler): ...

    async def run_forever(self):
        async for event in self.mailbox.listen():            # bloquea barato
            asyncio.create_task(self._handle(event))

    async def _handle(self, event):
        async with self.harness.session(event.user_id, event.session_id) as session:  # lease
            async for _ in session.send(event.as_message()):  # un turno normal
                pass                                           # los hooks observan
```

```python
@dataclass
class Event:
    user_id: str
    session_id: str
    source: str        # "user" | "timer" | "webhook:github" | "agent:reviewer" | "task:42"
    content: str
```

El evento entra al transcript como mensaje de usuario con su origen visible, igual que ya haces con `[harness]`:

```
[timer] You asked to be woken: "check if the deploy finished".
[task:42] Background command finished with exit code 1. Output: ...
```

El modelo tiene que saber **quién** le habla: un mensaje de timer no es una instrucción del usuario, y un webhook es texto no confiable.

### Tarea en segundo plano

```python
@tool
async def bash(ctx, command: str, description: str, run_in_background: bool = False) -> str:
    if run_in_background:
        task = ctx.session.tasks.start(command)       # el sandbox lo deja corriendo
        return f"Started in background as task {task.id}. You will be told when it ends."
    ...
```

Al terminar: `mailbox.put(Event(source=f"task:{id}", ...))`. El turno que lo lanzó ya terminó hace rato.

### Despertarse solo

```python
@tool
async def schedule_wakeup(ctx, delay_seconds: int, reason: str) -> str:
    """Come back to this conversation later."""
    ctx.harness.scheduler.at(now() + delay_seconds, Event(source="timer", content=reason, ...))
    return f"You will be woken in {delay_seconds}s."
```

Con tope mínimo y máximo, y un hook `stop` que revise que el agente no se programe en bucle.

## Orden

1. `Session` en el harness (archivo 03, paso 3). Sin esto todo lo demás es más caro.
2. Guardar el `RunContext` después de cada ronda de tools, no solo al final.
3. `Mailbox` durable: empieza con una tabla SQLite o una carpeta; la interfaz es `put` / `listen`.
4. `Supervisor` + `schedule_wakeup`. Con esto ya tienes un agente que vuelve solo.
5. Tareas en segundo plano.
6. Cron y webhooks: son solo más productores de `Event`.
7. Lease entre procesos y presupuesto por sesión.

## Trampas conocidas

- **Costo silencioso.** Un agente que despierta cada minuto con 80k tokens de contexto cuesta aunque no haga nada. El caché de prompt (archivo 01, punto 3.1) pasa de "optimización" a "requisito", y los despertares deben ser largos por defecto.
- **Bucles amables.** "Sigo esperando" → dormir → "sigo esperando". Cuenta despertares sin progreso y alarga el intervalo.
- **Aprobaciones huérfanas.** Sin persona, un `approval_needed` es una sesión parada para siempre. Define un vencimiento.
- **Eventos como instrucciones.** El contenido de un webhook o de la salida de un comando es texto no confiable. Márcalo con su origen y no le des la autoridad del usuario.
- **Dos procesos, una sesión.** Sin lease, dos supervisores corren el mismo turno y uno pisa al otro al guardar.
