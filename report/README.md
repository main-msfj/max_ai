# Revisión de max_ai — 2026-11-14

Rama revisada: `feat/workspace-new-approach` (commit `5bd68a1` más lo que está sin commitear).
Leí el agente, el loop, el dispatcher, la policy, el middleware, el bus de eventos, los guards, el gate, bash, las tools de archivos viejas y nuevas, el sandbox nuevo y el Layout. Corrí la suite: **647 pasan, 10 se saltan**; el test nuevo de la rama no llega a correr (detalle en el archivo 01).

No toqué nada del código. Todo lo que hay aquí es lectura, pruebas en una carpeta temporal y propuestas.

## Archivos

| Archivo | Qué responde |
|---|---|
| [01-feedback.md](01-feedback.md) | Qué está bien, qué está mal, qué no hace sentido. Con archivo y línea. |
| [02-tools-limpias.md](02-tools-limpias.md) | Cómo hacer read / write / edit / bash con poco código. |
| [03-harness.md](03-harness.md) | Cómo pasar de "Agent que lo hace todo" a un harness central, con hooks. |
| [04-always-on.md](04-always-on.md) | Cómo funcionan los agentes always-on y qué le falta al tuyo. |
| [05-multi-agent-a2a.md](05-multi-agent-a2a.md) | Guía de multi-agent y de A2A sobre tu framework. |
| [06-subagentes-y-orquestador.md](06-subagentes-y-orquestador.md) | Subagentes como Claude Code y Codex (`spawn_agent`), y el orquestador multi-agent. |
| [07-opinion-honesta.md](07-opinion-honesta.md) | Mi opinión del framework, sin suavizar. |
| [08-dream-system.md](08-dream-system.md) | Memoria como archivos en el sandbox y cómo hacer un "dreaming" como el de Anthropic. |

## Lo más importante, en orden

1. **La rama está a medias y se contradice a sí misma.** Hay dos scripts de archivos con contratos distintos, un test que importa una función que no existe, y `write_file` / `edit_file` sin implementar. Antes de seguir conviene decidir un solo diseño (propuesta en el archivo 02).
2. **Dos fallos de seguridad reales en lo nuevo**, los dos comprobados:
   - un `session_id` igual a `..` hace que `read_file` lea `.agent/setting.json` (sale del workspace);
   - las reglas `deny` de la Policy dejan de aplicar con los prefijos nuevos: `read_file(secrets/**)` niega `secrets/key` pero **permite** `$SESSION/secrets/key`.
3. **`.agent/setting.json` vive donde el bash del modelo puede escribir.** Si de ahí van a salir permisos o skills, el modelo se los puede dar solo.
4. **El plan se escribe en el system prompt.** Cada `update_plan` cambia el principio del prompt y tira el caché de toda la conversación. Es el cambio con mejor relación costo/beneficio de toda la lista.
5. **Hay cuatro mecanismos que deciden o corrigen el rumbo** (Policy, Middleware, gates en el EventBus, LoopGuards) y el `Agent` cablea los cuatro a mano. Eso es exactamente lo que un harness con hooks unifica (archivo 03).
6. **El `Agent` es el harness disfrazado**: 811 líneas, 16 parámetros, registra tools, conecta MCP, sincroniza workspace, arma el `ToolContext`, guarda estado de pausa. Moverlo a un `Harness` no es reescribir: es cambiar de dueño a código que ya existe.

## Lo que haría primero

1. Cerrar la rama con un solo diseño de tools de archivos y arreglar los dos fallos de seguridad.
2. Sacar el plan del system prompt.
3. Crear `Harness` moviendo `_drive_connected` y el cableado, sin cambiar comportamiento.
4. Hooks sobre ese harness (reemplazan gates + guards + middleware de tools).
5. Subagente como tool. Después, A2A y always-on, que ya son capas finas encima.
