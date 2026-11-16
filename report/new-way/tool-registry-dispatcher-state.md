# Feedback: `core/tool` (registry, state, dispatcher)

Revisado el 2026-11-15 sobre la rama `feat/workspace-new-approach`. Objetivo de la revisión: que estas piezas se coordinen con el sandbox nuevo y que cada una se pueda usar sola o armada por el `Agent`.

**Resumen:** `registry.py` y `state.py` ya son independientes y necesitan poco. Todo el trabajo está en `dispatcher.py`: depende del `EnvironmentManager` viejo, de una bolsa `deps` sin tipo y de `RunContext`.

| Archivo | Líneas | Estado |
|---|---|---|
| `registry.py` | 40 | bien |
| `state.py` | 229 | bien, con algo de sobra |
| `dispatcher.py` | 422 | hay que rehacer cómo recibe el entorno |

## `registry.py`

Un catálogo por nombre, sin dependencias. Se usa solo. Está bien.

- **Corregido el 2026-11-16.** `read_only` sí está declarado en `CoreTool`; lo que había era un `getattr(..., "read_only", False)` en el dispatcher para cubrir una tool desconocida. Ahora lo lee como atributo normal.

## `state.py`

Es solo datos, se serializa y no depende de nada del runtime. Eso está bien.

- **Hay dos caminos para cambiar un record.** `ToolState` ofrece `consume`, `apply_approval` y `fail_stale`, pero el dispatcher casi nunca pasa por `ToolState`: llama directo a `record.mark_consumed()`, `record.auto_approve()` y `record.force_consume()`. Con dos caminos, tarde o temprano uno hace algo que el otro no (por ejemplo, el log solo sale por uno). Recomiendo que el record sea el dueño de sus cambios y `ToolState` quede como colección y consultas.
- **Cuatro métodos repiten lo mismo**: buscar el record y lanzar `KeyError` si no está. Es un método privado `_record(id)`.
- **`load_from(state, approval=...)` hace dos cosas**: cargar y aplicar una aprobación. Son dos llamadas.
- **Diez propiedades hacen el mismo filtro** sobre `records` con una condición distinta. No es un problema, pero cada estado nuevo pide una propiedad nueva.

## `dispatcher.py`

### Lo que está bien y hay que conservar

- Las llamadas de solo lectura que van seguidas corren a la vez, y los resultados entran al transcript en el orden en que se pidieron.
- Una llamada que espera aprobación devuelve `None` y se puede retomar después.
- Un resultado muy largo se corta por el medio y el modelo sabe cuánto falta.
- El middleware solo ve llamadas ya aprobadas y puede responder en lugar de la tool.
- Hay una red de seguridad de tiempo para una tool que nunca vuelve.

### Lo que impide coordinarlo con el sandbox

**1. Depende del `EnvironmentManager`.** Lo usa para tres cosas distintas:

- exigir que exista si la tool corre comandos (`runs_commands`);
- abrir una sesión del executor (`manager.acquire(...)`), compartida por un lote de lecturas;
- preguntar `manager.executor.isolated` para la Policy.

El sandbox nuevo no tiene sesiones que abrir ni cerrar: se le pasa un comando y responde. Con `sandbox=None` en el constructor se van `shared_session`, `run_in`, el caso especial de `run_batch` y el parámetro `session` de `dispatch`: unas 50 líneas. `isolated` pasa a ser una propiedad del sandbox (`LocalSandbox` no, `DockerSandbox` sí).

**2. `deps` es una bolsa sin tipo.** Hoy lleva `run_context`, `tool_call_id`, `executor` y `execution_session`; las tools nuevas esperan además `sandbox`, `file_stamps` y `bash_cwd`. El propio dispatcher saca de ahí `run_context` para leer `allowed_rules` y para armar el middleware. Nadie puede saber qué poner en `deps` sin leer el código, y un nombre mal escrito falla en silencio. Es el mayor freno para usar las piezas por separado.

`ToolContext` debería tener campos reales:

```python
ToolContext(
    run_id, session_id, user_id,
    tool_call_id=...,
    sandbox=...,            # ya con bind(identifier)
    allowed_rules=(...),    # lo que el usuario permitió en la sesión
    state={...},            # lo que las tools guardan entre llamadas: stamps, cwd
    emit_event=...,
)
```

**3. `_dispatch` tiene 190 líneas con ocho etapas seguidas**: validar, Policy, esperar al usuario, aprobar, middleware, ejecutar, tiempo límite y cerrar. Se lee con dificultad y no hay un lugar claro donde entren los hooks. Partido en tres queda así:

```
decidir   validar parámetros → Policy → ¿allow / ask / deny?
aprobar   esperar al usuario o a la aprobación (devuelve None si falta)
ejecutar  hooks de antes → tool.execute → hooks de después → cerrar el record
```

Los hooks de después son el lugar para ocultar los valores de las variables en la salida.

**4. La Policy solo ve los parámetros.** No puede saber si una ruta cae en otra sesión o fuera del workspace, y una regla como `write_file(secrets/**)` no coincide con la ruta absoluta del mismo archivo. La etapa "decidir" tiene que recibir el `layout` del sandbox y el `session_id`. Ahí se resuelven las dos cosas pendientes: "otra sesión → preguntar al usuario" y las reglas con rutas absolutas.

**5. Las funciones del transcript no son de despachar.** `close_unanswered`, `answer_unanswered`, `close_interrupted` y `tool_message` escriben en `ctx.messages`. Atan el dispatcher a `RunContext` y son la razón del import perezoso en `__init__.py`. En un archivo aparte (`core/tool/transcript.py`), el dispatcher deja de conocer `RunContext`.

**6. Lee `setting` global en cuatro lugares**: tiempo límite, llamadas en paralelo, límite de caracteres. Como parámetros del constructor con esos valores por defecto, se puede probar sin tocar la configuración global y una tool puede tener su propio tiempo límite.

### Fallas pequeñas

- **`ToolCallEvent` se emite dos veces** para una llamada que pide aprobación: al pedirla y al retomarla. Se emite al inicio de cada `_dispatch`.
- **`tool_message` pasa todo por `json.dumps`**, así que una tool no puede devolver una imagen. Hace falta para `read_file` con imágenes y para MCP.
- **Una llamada que quedó en `EXECUTING` lanza `ValueError`** en `dispatch` en vez de devolver un resultado. Dentro de `dispatch_many` eso corta todo el lote.

## Cómo quedaría por piezas

```
ToolRegistry                               solo: catálogo de tools
ToolDispatcher(registry)                   tools de función, sin sandbox
ToolDispatcher(registry, sandbox=...)      + read_file, write_file, edit_file, bash
               policy= / middleware=       opcionales, como hoy
Agent                                      arma todo y además hace skills.sync()
```

Cada línea funciona sin las de abajo.

## Orden recomendado

| # | Cambio | Por qué en ese orden |
|---|---|---|
| 1 | `ToolContext` con campos reales | todo lo demás se apoya en eso |
| 2 | `manager` → `sandbox` en el dispatcher | quita la dependencia del executor viejo |
| 3 | Sacar las funciones del transcript | el dispatcher deja de conocer `RunContext` |
| 4 | Partir `_dispatch` y pasar el layout a la Policy | abre el lugar de los hooks y del permiso por sesión |
| 5 | `setting` al constructor, limpiar `state.py`, `read_only` en `CoreTool` | limpieza, sin riesgo |

Después de 1 y 2 ya se pueden conectar las tools de `new_tool/`.
