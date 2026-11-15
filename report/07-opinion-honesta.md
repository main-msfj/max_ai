# 07 — Opinión honesta del framework

Sin suavizar. Es mi lectura después de revisar el código; puedes no estar de acuerdo y hay cosas del contexto (para qué lo quieres, cuánto tiempo tienes) que no conozco.

## En una frase

**El núcleo es mejor de lo que crees y el framework es más ancho de lo que te conviene.** Lo difícil está bien hecho; lo que ve el modelo (las tools) es lo menos terminado; y hay demasiadas piezas alrededor para un proyecto de una persona sin usuarios.

---

## Lo que de verdad es bueno

No lo digo por cortesía: son cosas que frameworks conocidos hacen peor.

1. **El estado es datos.** `Agent` sin estado por corrida, `RunContext` serializable, pausa y resume sobre registros y no sobre corrutinas. La mayoría de los frameworks de agentes no pueden pausar un turno, guardarlo en una base de datos y retomarlo en otro proceso. El tuyo sí. Es la decisión más valiosa del proyecto.
2. **El transcript nunca queda inválido.** Cancelación, corte por tokens, ids repetidos, llamadas a medias: todo termina con cada tool call respondida. Eso solo se aprende rompiéndolo contra proveedores reales, y se nota que lo hiciste.
3. **La Policy.** Una sola decisión, reglas legibles, y el análisis de bash con la noción de "si no pude leerlo entero, no lo permito solo". Está al nivel de un producto.
4. **Disciplina.** 647 tests en 35 segundos, commits que dicen qué cambió, comentarios que explican el porqué. Se puede refactorizar sin miedo, y eso es raro.
5. **Criterio para simplificar.** Borraste `AgentAsTool`, moviste el steering a un mensaje de usuario cuando viste que Gemma se descarrilaba, hiciste de `ask_user` una tool normal. Sabes quitar.

---

## Lo que no está bien

### 1. Demasiado ancho para lo profundo que es

28.500 líneas. Tres clientes de modelo. Tres executors. Tres workspaces. Memoria en local, SQLite y Mongo. Knowledge en local, SQLite y Mongo. Session store en dos. Quota store en dos. Contexto en dos. Skills desde disco y GitHub. Dos compactaciones. MCP. Una CLI de 2.400 líneas. Serialización YAML de todo.

Y mientras tanto, `read_file` / `write_file` / `edit_file` — las tres tools que un agente usa en cada turno — están a medio reescribir por segunda vez, con dos scripts contradictorios y un test que no importa.

Eso es construir el edificio por las terrazas. Cada backend extra es código que hay que mover cuando cambia el contrato de arriba, y los contratos de arriba todavía se están moviendo: por eso el rediseño del workspace te obliga a tocar skills, sandbox, tools y entornos a la vez.

### 2. Dos generaciones de la misma idea conviviendo

`ExecutorBase` + `WorkspaceBase` + `EnvironmentManager` (viejo) junto a `SandboxCore` + `Layout` (nuevo). `file_system/` junto a `filesystem/`. `file_reads` junto a stamps. Gates junto a guards que hacen lo mismo. "No borrar nada mientras aprendo" es razonable por una semana; la rama lleva señales de que el costo ya se nota: dos clases `FileSystem`, un `Agent` que acepta `sandbox=` y lo ignora.

Tu propia nota dice que no hay usuarios y no hay que conservar compatibilidad. Úsalo: cuando lo nuevo lea un archivo, borra lo viejo ese mismo día.

### 3. Todo es serializable, y eso se paga en cada pieza

`ComponentBase` con `component_schema`, `_to_config`, `_from_config` en tools, middlewares, gates, guards, layers, executors, sandboxes. Es lo que permite guardar un agente en YAML, y es una idea buena para **la definición** del agente. Pero aplicada a todo, cada pieza nueva nace con 30 líneas de ceremonia antes de hacer algo, y es parte de por qué una tool de lectura ocupa 86 líneas. Serializa el `AgentSpec` y las referencias por nombre; deja que el resto sea código.

### 4. Mucha maquinaria para compensar modelos débiles

`SchemaRetryGuard`, `RepetitionGuard`, `BudgetGuard`, `NoProgressGuard`, `PlanCompletionGuard`, el gate de archivos intermedios, el seguimiento tras una negación, `TaskAnalysisLayer`. Buena parte existe porque pruebas con modelos gratuitos que repiten llamadas, devuelven vacío o se descarrilan con un system tardío.

Dos consecuencias. Una: el framework se está afinando contra el peor caso, y parte de esa complejidad sobra — o estorba — con un modelo fuerte. Dos: no sabes cuál, porque **no hay evaluaciones**. La calidad se comprueba mirando Langfuse a mano. Sin un conjunto de 20–30 tareas con resultado verificable que puedas correr tras cada cambio, cada guard es una creencia. Es lo que más falta en el proyecto, más que hooks.

### 5. Le falta lo que hace barato a un harness

No hay cliente de Anthropic ni manejo de caché de prompt, y el plan se reescribe en el system prompt. Un agente tipo harness hace decenas de llamadas por turno con el mismo prefijo: sin caché, el costo y la latencia son varias veces mayores. Para chat corto no importa; para lo que quieres construir (subagentes, always-on, orquestador) es condición.

### 6. No queda claro para quién es

Hoy max_ai intenta ser a la vez: un SDK para servidores multiusuario (session stores, quotas, Mongo, `user_id` en todo), un agente de código local (CLI, bash, edición de archivos, `.agent/` en el proyecto) y una plataforma (YAML, skills, MCP). Son tres productos con prioridades opuestas: el servidor necesita aislamiento y cero estado local; el agente de código necesita tools excelentes y velocidad; la plataforma necesita estabilidad de contratos.

Y contra lo que ya existe — el SDK de agentes de Claude, el de OpenAI, pydantic-ai, LangGraph — "otro framework de agentes con muchas integraciones" no es una razón para elegirlo. Lo que sí tienes y ellos no juntan es: **agente sin estado, pausable y reanudable desde una base de datos, con política de permisos real, independiente del proveedor**. Eso es un harness para ejecutar agentes en servidores. Es un nicho concreto y defendible. Las integraciones no lo son.

### 7. El riesgo de fondo: rediseñar sin terminar

El historial reciente es: steering, ask_user, tools dinámicas, workspace, sandbox, tools de archivos, y ahora harness, hooks, multi-agent, A2A, always-on. Cada una es buena idea. Pero la lista de lo que quieres crece más rápido que la de lo que cierras, y varias quedan "diseñado, no aplicado". Un harness central es la dirección correcta **y** es otra reescritura grande encima de una rama que no está cerrada. Si empiezas el harness sin cerrar las tools, tendrás tres cosas a medias en vez de dos.

---

## Qué haría yo

1. **Cerrar la rama esta semana**: un solo diseño de tools de archivos (archivo 02), los dos fallos de seguridad, borrar lo viejo.
2. **Congelar la periferia.** Un cliente, un sandbox local, un store. Lo demás no se toca — o se saca a una carpeta `contrib/` — hasta que el núcleo deje de moverse. No añadir backends.
3. **Un conjunto de evaluación pequeño** antes del harness: 20 tareas reales con verificación automática, corridas con un modelo fuerte y uno barato. Te dirá qué guards sirven y te protegerá durante la migración.
4. **Caché de prompt**: plan fuera del system prompt y un cliente que lo aproveche.
5. **Harness + Session + hooks** (archivo 03), moviendo código, no reescribiendo.
6. **Subagente como tool** y las primitivas de orquestador (archivo 06).
7. **Usarlo tú para algo real todos los días.** Un agente que te resuelva algo propio encuentra en una semana más problemas que un mes de diseño. Hoy el único usuario son los tests.
8. Always-on y A2A al final. Son capas finas si lo anterior existe, y pozos sin fondo si no.

## Veredicto

Como ingeniería: **sólido en lo difícil, desordenado en lo reciente, sobredimensionado en lo accesorio.** Como proyecto: tiene un núcleo que vale la pena y le falta decir que no. No necesitas más piezas; necesitas menos piezas, terminadas, y una forma de medir si el agente mejora.

Si sigues como vas, en tres meses tendrás un framework con más conceptos y la misma sensación de desorden. Si estrechas y cierras, en tres meses tendrás un harness pequeño que hace bien algo que pocos hacen.
