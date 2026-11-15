# 06 — Subagentes al estilo Claude Code / Codex, y el orquestador

El archivo 05 explica el patrón general ("subagente como tool"). Este baja al detalle: cómo lo hacen dos harness reales, cómo se ve un orquestador, y qué piezas concretas necesita max_ai para cada uno.

> Lo de Claude Code lo describo con precisión: es el harness en el que estoy corriendo y veo sus tools. Lo de Codex es de memoria y cambia rápido (su multi-agent ha sido experimental); tómalo como la forma general y confirma los nombres en su repo antes de copiar nada.

---

## 1. Hay tres cosas distintas que se llaman "multi-agent"

| | Quién decide | Cómo se comunican | Cuándo usarlo |
|---|---|---|---|
| **A. Subagente (spawn)** | el modelo, en el momento | una tool: prompt de ida, respuesta final de vuelta | aislar contexto, explorar, verificar |
| **B. Equipo** | el modelo, con agentes que siguen vivos | mensajes entre agentes con nombre | trabajos largos con roles |
| **C. Orquestador** | **código** (un script o un flujo) | el código reparte y junta | muchos elementos iguales, pasos fijos, resultados que se validan |

Casi todo el valor está en A. B es A más "el hijo no se destruye". C no es un agente más listo: es un programa que usa agentes como funciones.

---

## 2. Cómo lo hace Claude Code

### La tool `Agent`

Una sola tool. Sus parámetros:

| Parámetro | Para qué |
|---|---|
| `description` | 3–5 palabras, para la interfaz |
| `prompt` | la tarea completa; el hijo no sabe nada más |
| `subagent_type` | qué definición de agente usar |
| `model` | opcional: otro modelo para este hijo (uno barato para buscar) |
| `isolation: "worktree"` | el hijo trabaja en un git worktree propio |
| `run_in_background` | devuelve al instante; avisa al terminar |

Lo que importa de su diseño:

**1. Los tipos de agente son archivos, no clases.** Un `.md` con cabecera:

```markdown
---
name: explore
description: Read-only search agent for broad searches. Use when…
tools: Read, Grep, Glob, Bash        # lista blanca
model: haiku
---
You are a search specialist. You locate code; you do not review it…
```

El cuerpo es el system prompt del hijo. La `description` es lo que ve el **padre** para decidir cuándo usarlo. Crear un agente nuevo es crear un archivo.

**2. La lista de tipos llega al padre como contexto**, con su descripción y sus tools. El modelo elige leyendo eso, igual que elige una tool.

**3. El hijo arranca en frío**, y la descripción de la tool se lo repite al padre varias veces: pon todo en el prompt, di qué formato de respuesta quieres, di si debe escribir código o solo investigar.

**4. `fork`**: un tipo especial donde el hijo **hereda toda la conversación** del padre. Sirve para "haz esto en paralelo sin que yo tenga que reexplicar"; cuesta más tokens, pero comparte el caché del padre.

**5. Solo vuelve el mensaje final.** El transcript del hijo se guarda aparte. El usuario no ve la respuesta del hijo: el padre tiene que contarla.

**6. Seguir hablando con un hijo.** `SendMessage(to=<nombre o id>, message)` continúa al mismo hijo con su contexto. Hay un `ListAgents` para ver quién existe.

**7. En segundo plano.** El padre sigue trabajando; cuando el hijo termina, el harness le inyecta una notificación como mensaje. La regla escrita en el prompt: nunca inventes el resultado de un agente que no ha terminado.

**8. Aislamiento con worktree.** Cada hijo que escribe recibe su copia del repo en una rama; si no cambió nada, se borra sola.

**9. Hooks `SubagentStart` / `SubagentStop`**, y permisos heredados del padre.

**10. Anidamiento limitado.** Un hijo normalmente no puede lanzar más hijos.

**11. Freno explícito.** El prompt del padre dice cuándo **no** delegar: cada hijo arranca en frío y es la vía cara. Sin ese texto, los modelos delegan de más.

### Qué te falta para tener esto

| Pieza de Claude Code | En max_ai |
|---|---|
| Definición de agente en archivo | **Ya casi**: `AgentSpec` + `Agent.from_yaml`. Falta una carpeta (`.agent/agents/`) y un campo `tools:` como lista blanca |
| Lista de tipos en el prompt | Una capa de prompt nueva, igual a la de deferred tools |
| Tool `agent` | No existe. ~40 líneas con `Harness.child_session` (archivo 05) |
| `model` por hijo | El harness necesita un registro de clientes por nombre, no un solo `client` |
| Hijo en frío / `fork` | En frío es `RunContext()` nuevo. `fork` es `ctx.model_copy()` con otro `session_id` |
| Solo vuelve el final | `AgentResponse` ya lo tiene |
| `SendMessage` | Guardar la sesión del hijo con `parent_session_id` y reabrirla |
| Segundo plano + notificación | Mailbox (archivo 04). Tu `inbox` es el embrión |
| Worktree | Método reemplazable "crear carpeta de sesión" en el sandbox |
| Hooks de subagente | Con el `HookRegistry` (archivo 03) |
| Eventos anidados | `agent_path` en `CoreEvent` |

---

## 3. Cómo lo hace Codex

La misma idea con otra forma: en vez de **una** tool que bloquea hasta la respuesta, da **varias tools de control** y el hijo siempre corre aparte:

| Tool | Hace |
|---|---|
| `spawn_agent(message, agent_type?)` | crea un hijo y devuelve su **id** de inmediato |
| `send_input(id, message)` | le manda otro mensaje (puede interrumpirlo) |
| `wait(ids, timeout)` | espera a que alguno termine y devuelve su estado |
| `close_agent(id)` | lo cierra |

Los roles (`default`, `explorer`, `worker`…) se definen en configuración, con su modelo y sus instrucciones, y hay un tope de hilos y de profundidad.

### Diferencia real entre los dos

| | Claude Code | Codex |
|---|---|---|
| Por defecto | síncrono: una llamada, una respuesta | asíncrono: spawn + wait |
| El modelo maneja | nada; el harness espera | el ciclo de vida completo |
| Paralelo | varias llamadas `Agent` en un mismo mensaje | varios `spawn_agent`, luego un `wait` |
| Riesgo | menos control fino | el modelo olvida esperar o cerrar |

**Para max_ai empieza por la forma de Claude Code.** Razones: es una tool normal (tu dispatcher, Policy y aprobación ya la cubren), el paralelo sale gratis de tu lote de `read_only`, y no depende de que el modelo recuerde cerrar nada — importante si pruebas con modelos baratos. La forma de Codex necesita el mailbox y el registro de tareas; llega sola cuando hagas segundo plano, y entonces son las mismas cuatro tools.

---

## 4. El orquestador

### Qué es

Un orquestador es **código determinista** que decide cuántos agentes lanzar, con qué prompt, en qué orden, y qué hacer con cada resultado. El modelo no decide el reparto.

Cuándo gana a "el agente principal delega":

- **Muchos elementos iguales**: revisar 200 archivos, migrar 50 módulos. Un modelo repartiendo 200 tareas se pierde; un `for` no.
- **Pasos fijos**: buscar → verificar → sintetizar.
- **Resultados que hay que validar**: cada hijo devuelve JSON con esquema; el código filtra, deduplica, reintenta.
- **Costo predecible**: sabes cuántas llamadas habrá antes de empezar.

### Las tres primitivas

Con `Harness` (archivo 03) hacen falta solo estas:

```python
async def agent(harness, prompt, *, type="general", schema=None, isolation=None) -> T | None:
    """Corre un agente hasta el final. Con ``schema``, devuelve el objeto validado.
    None si falló o se canceló."""

async def parallel(thunks, *, limit=8) -> list:
    """Corre todos, como mucho ``limit`` a la vez. Un fallo no tumba a los demás."""

async def pipeline(items, *stages) -> list:
    """Cada item pasa por todas las etapas sin esperar a los otros items."""
```

Ya tienes las piezas: `output_format` da el `schema`, `max_parallel_tools` es el `limit`, `CancellationToken` cancela todo.

### Ejemplo: revisar y verificar

```python
class Finding(BaseModel):
    file: str
    line: int
    problem: str

class Verdict(BaseModel):
    real: bool
    why: str

async def review(harness, files):
    async def one(path):
        found = await agent(harness, f"Review {path} for bugs.", type="reviewer", schema=list[Finding])
        checked = await parallel([
            lambda f=f: agent(harness, f"Try to refute: {f.problem} at {f.file}:{f.line}",
                              type="skeptic", schema=Verdict)
            for f in found or []
        ])
        return [f for f, v in zip(found or [], checked) if v and v.real]

    per_file = await parallel([lambda p=p: one(p) for p in files])
    return [f for group in per_file for f in group or []]
```

Nota el patrón: cada hallazgo pasa por **otro agente con contexto limpio que intenta refutarlo**. Es lo que más sube la calidad en multi-agent, mucho más que añadir agentes que opinan lo mismo.

### Los patrones que valen la pena

| Patrón | Forma | Para |
|---|---|---|
| **Fan-out / fan-in** | N en paralelo → juntar | buscar en muchos lugares |
| **Pipeline por elemento** | cada item: A → B → C, sin esperar al resto | revisar → verificar → corregir |
| **Verificación adversaria** | uno afirma, otro intenta tumbarlo | hallazgos, hechos |
| **Jurado** | 3 intentos independientes → uno elige | diseño, decisiones con varias salidas |
| **Hasta secar** | repetir "busca más" hasta 2 rondas sin nada nuevo | auditorías de tamaño desconocido |
| **Completitud** | al final: "¿qué faltó?" → otra ronda | investigación |

### Dos formas de exponerlo

1. **Como API de Python** (`max_ai.orchestrate`): el desarrollador escribe el flujo. Es lo primero, y es poco código.
2. **Como tool** `run_workflow(script)`: el **modelo** escribe el flujo y el harness lo ejecuta en segundo plano. Es potente, pero es ejecutar código del modelo: exige sandbox aislado, tope de agentes y de tokens, y que cada agente del flujo pase por la Policy. Déjalo para después.

### Lo que el orquestador le pide al harness

- **Tope global de agentes simultáneos**, no solo por lote (dos flujos a la vez no deben sumar 16).
- **Presupuesto del flujo**: tokens totales, con corte limpio.
- **Fallos parciales**: un hijo que falla devuelve `None`; el flujo sigue.
- **Reanudar**: guardar el resultado de cada llamada por `(prompt, tipo)`; al relanzar, lo hecho no se repite. Sin esto, un flujo de 100 agentes que muere en el 97 se paga dos veces.
- **Progreso**: eventos por etapa, para que la CLI muestre "revisar 12/50".
- **Rate limits**: 50 agentes en paralelo contra el mismo proveedor; tu `quota_store` y el backoff ya existen, hay que compartirlos entre hijos.

---

## 5. Orden recomendado

1. `Harness` + `Session` (archivo 03). Requisito de todo lo demás.
2. Tool `agent` síncrona, profundidad 1, tipos desde `.agent/agents/*.yaml`, lista blanca de tools.
3. Eventos con `agent_path`; tokens del hijo sumados al padre.
4. Primitivas `agent()` / `parallel()` / `pipeline()` como API de Python. Con 2 + 4 ya tienes subagentes **y** orquestador.
5. `model` por tipo de agente (explorador barato).
6. Aprobaciones que suben al padre.
7. `send_message` (el hijo sigue vivo).
8. Segundo plano: `spawn` / `wait` al estilo Codex, sobre el mailbox.
9. Worktrees.
10. `run_workflow` como tool.

Los pasos 2–4 son pocos cientos de líneas si el harness existe. Sin el harness, cada uno obliga a tocar `Agent`, el loop y el dispatcher a la vez.
