# 08 — Memoria en el sandbox y un sistema de "dreaming"

Escrito el 2026-11-16. Responde tres preguntas:

1. ¿Cómo funciona la memoria y el *dreaming* de Anthropic?
2. ¿Deberíamos pasar la memoria al sandbox? ¿Y los resúmenes?
3. ¿Cómo se construiría un sistema parecido en `max_ai`?

Fuentes: la documentación de Managed Agents de Anthropic, páginas [memory](https://platform.claude.com/docs/en/managed-agents/memory) y [dreams](https://platform.claude.com/docs/en/managed-agents/dreams), leídas hoy. La memoria está en beta y los dreams en *research preview*, así que pueden cambiar.

## Respuesta corta

- **Memoria al sandbox: sí.** Es lo que hace Anthropic, y encaja con lo que ya construimos: la memoria pasa a ser una carpeta de archivos de texto que el modelo lee y escribe con `read_file`, `write_file` y `edit_file`. Se van las cinco tools de memoria y los tres backends.
- **Resúmenes al sandbox: no.** El resumen de compactación es parte de la conversación, no un archivo del usuario. El harness lo necesita antes de cada llamada al modelo, incluso sin sandbox. Se queda con la sesión.
- **Dreaming: es una corrida más de tu propio agente**, con un prompt especial, que lee la memoria y conversaciones pasadas y escribe una memoria nueva en otra carpeta. No necesita infraestructura nueva.

Queda en dos lugares en vez de cinco:

```
 HARNESS (tu proceso)                    SANDBOX (.agent/ del usuario)
 ┌──────────────────────────┐            ┌──────────────────────────────┐
 │ session store            │            │ workspace/<sesión>/          │
 │   mensajes               │            │ skills/                      │
 │   resumen de compactación│            │ memory/        ← nuevo       │
 │   estado de las tools    │            │ tmp/<sesión>/tool_results/   │
 └──────────────────────────┘            └──────────────────────────────┘
   lo que es la conversación               lo que el modelo toca como archivos
```

## 1. Cómo lo hace Anthropic

### Memoria

| Qué | Cómo |
|---|---|
| Qué es | un *memory store*: una colección de documentos de texto |
| Dónde la ve el agente | montada como carpeta dentro del sandbox, en `/mnt/memory/<nombre>/` |
| Cómo la usa | con las mismas tools de archivos que usa para todo lo demás; no hay tools de memoria |
| Cómo sabe que existe | se agrega sola una nota al system prompt: nombre, ruta, modo de acceso, descripción e instrucciones |
| Permisos | `read_write` (por defecto) o `read_only`, aplicado en el sistema de archivos |
| Límites | 100 kB por memoria (unos 25.000 tokens), 10.000 memorias por store |
| Historial | cada escritura crea una versión; se guardan 30 días |
| Recomendación de ellos | muchos archivos pequeños y enfocados, no pocos grandes; un store por usuario y otro para conocimiento compartido |

Dos detalles que importan para nosotros:

- **El sandbox no es el dueño de la memoria.** La memoria vive en un servicio aparte y se monta en el sandbox. Con sandbox propio (*self-hosted*) no hay montaje: un proceso baja una copia antes de que corran las tools, la sincroniza como mucho cada 15 segundos y otra vez al terminar la sesión. Es el mismo patrón que nuestras skills.
- **Advierten de un riesgo:** con `read_write`, si el agente procesa texto no confiable (una web, un archivo ajeno), una inyección de prompt puede escribir contenido malicioso en la memoria, y las sesiones siguientes lo leen como si fuera de confianza. Para material de referencia recomiendan `read_only`.

### Dreams

Con el uso, la memoria acumula duplicados, contradicciones y datos viejos, porque el agente escribe poco a poco y sin ver el conjunto. Un *dream* lo limpia.

```
 memoria actual ─────┐
                     ├──►  dream (trabajo en segundo plano)  ──►  memoria NUEVA, en otro store
 1 a 100 sesiones ───┘        con un modelo que tú eliges
 pasadas (transcripts)
```

| Qué | Cómo |
|---|---|
| Entradas | un memory store y de 1 a 100 sesiones pasadas |
| Salida | otro memory store; **la entrada nunca se modifica** |
| Qué hace | une duplicados, reemplaza lo viejo o contradicho por el valor más reciente, y agrega lo nuevo que encuentra en las sesiones |
| Cómo se guía | un campo `instructions` ("enfócate en preferencias de estilo; ignora notas de depuración de una sola vez") |
| Estados | `pending` → `running` → `completed`, `failed` o `canceled` |
| Cuánto tarda | de minutos a unas horas, según cuántas sesiones |
| Costo | tokens normales del modelo elegido; crece más o menos en línea con el número y largo de las sesiones |
| Después | revisas la salida y la adoptas (la usas en las sesiones siguientes) o la descartas |

Lo más útil para nosotros: **por dentro, un dream es una sesión normal de agente**. El dream tiene un `session_id` que apunta a la sesión que ejecuta el trabajo, y se pueden ver sus eventos. Además Anthropic publica una receta para construir tu propio dream con tus prompts y tus modelos. No es una tecnología especial: es un agente con un encargo.

## 2. ¿Pasamos la memoria al sandbox?

Sí. Hoy la memoria de `max_ai` es un mundo aparte:

| Hoy | Con la memoria en el sandbox |
|---|---|
| `CoreMemoryRegistry` con tres backends (local, SQLite, MongoDB) | una carpeta `.agent/memory/` |
| cinco tools: `get_context`, `list_category`, `search_memory`, `create_or_update`, `delete_memory` | ninguna tool nueva: `read_file`, `write_file`, `edit_file` y `bash` (para buscar con `grep`) |
| registros `categoría → texto` | archivos Markdown, uno por tema |
| búsqueda por significado propia | el modelo lee un índice y abre lo que necesita |
| `bind(user_id, session_id)` propio | el sandbox ya es del usuario |

### Lo que se gana

- **Una sola forma de trabajar.** El modelo ya sabe leer y editar archivos; no aprende cinco tools más. Menos tools también es menos prompt.
- **Mucho menos código**: un padre, tres backends y cinco tools se reemplazan por una carpeta y una nota en el prompt.
- **Coherente con lo demás**: skills, workspace, resultados grandes y memoria quedan bajo `.agent/`, con las mismas reglas de rutas y permisos.
- **El usuario puede abrir su memoria** y corregirla a mano: son archivos de texto.

### Lo que se pierde o hay que cuidar

- **La búsqueda por significado.** Con archivos el modelo busca por nombre, por índice y con `grep`. Para memorias de un usuario (decenas o cientos de archivos pequeños) alcanza. Para miles de documentos no: eso es *knowledge*, que es otra pieza y puede seguir con su búsqueda.
- **El sandbox no es una base de datos.** Si la memoria solo vive en el volumen de Docker y el volumen se borra, se pierde. Anthropic lo resuelve teniendo la memoria en un servicio y montándola. Recomendación: empezar con la memoria solo en el sandbox (en local y Docker el volumen persiste) y, cuando haga falta, agregar una pieza que la traiga y la devuelva, igual que `SkillsCore.sync`.
- **Inyección de prompt.** La advertencia de Anthropic aplica igual. Hace falta el modo `read_only`, y eso lo decide el harness en `ToolHook.before`: una escritura dentro de `memory/` con la memoria en solo lectura se niega.
- **Tamaño.** Copiar la regla: archivos pequeños, con un tope por archivo. El recorte de 40.000 caracteres que ya hace `ToolHook` protege al leer; al escribir falta un límite.

### ¿Y los resúmenes?

Depende de qué resumen:

| Resumen | Dónde debe vivir | Por qué |
|---|---|---|
| Resumen de compactación (lo que reemplaza a los mensajes viejos) | con la sesión, en el harness | es parte de la conversación; se necesita antes de cada llamada al modelo y también cuando no hay sandbox; leerlo del sandbox en cada turno costaría un comando por turno |
| Lo que el agente decide recordar de una sesión ("el usuario prefiere pytest") | `memory/` en el sandbox | es conocimiento duradero del usuario, lo usa en otras sesiones |
| Notas de trabajo de la sesión (plan, pendientes) | `workspace/<sesión>/` o el `RunContext` | son de esa sesión |

La regla para decidir: **si el modelo lo abre como un archivo cuando lo necesita, va en el sandbox; si el harness lo necesita para armar la petición, va con la sesión.**

Tu intuición ("qué sentido tiene conectar todo a distintos lugares") es correcta para lo que el modelo toca. Lo único que no conviene mover es el estado de la conversación.

## 3. Cómo construir el dreaming en `max_ai`

### Las piezas

```
.agent/
    memory/                  la memoria en uso
        MEMORY.md            índice: una línea por archivo
        preferences.md
        project-acme.md
    memory.dream-<id>/       salida de un dream, sin adoptar
    memory.previous/         la anterior, por si hay que volver
    tmp/dream-<id>/sessions/ las conversaciones que el dream lee, como texto
```

| Pieza | Qué es | Existe hoy |
|---|---|---|
| Carpeta `memory/` en `Layout` | una propiedad más, creada en `prepare()` | no; es una línea en `layout/root.py` |
| Nota en el prompt | "tu memoria está en tal ruta, lee `MEMORY.md` primero" | no; igual que `skills.blocks()` |
| Modo `read_only` | el hook niega escrituras en `memory/` | no; va en `ToolHook.before` del harness |
| Exportar una sesión a texto | pasar los mensajes del session store a un `.md` | no |
| El dream | una corrida del `Agent` con un prompt de consolidación | el `Agent` sí; el prompt no |
| Adoptar | cambiar `memory.dream-<id>` por `memory`, en un solo paso | el script del layout ya hace este cambio atómico para las skills |
| Estado del dream | `pending`, `running`, `completed`, `failed`, `canceled` y el uso de tokens | no |

### El recorrido de un dream

```
1. preparar    copia memory/ → memory.dream-<id>/
               exporta las sesiones elegidas a tmp/dream-<id>/sessions/*.md

2. extraer     por cada sesión: ¿qué se aprendió que valga para el futuro?
               (preferencias, decisiones, hechos del proyecto, errores que no repetir)

3. consolidar  contra la memoria copiada:
               - une lo duplicado
               - si dos datos se contradicen, queda el más reciente
               - borra lo que ya no aplica
               - reparte en archivos pequeños por tema y rehace MEMORY.md

4. informe     escribe DREAM.md: qué agregó, qué cambió, qué borró y por qué

5. fin         estado completed; la memoria en uso sigue intacta
```

Los pasos 2 y 3 son el agente trabajando con sus tools de archivos dentro del sandbox, con escritura permitida solo en `memory.dream-<id>/`. El paso 2 puede usar un modelo barato y el 3 uno mejor; el costo crece con el número de sesiones, así que conviene empezar con pocas.

### Cómo se usaría

```python
dream = await agent.dream(
    user_id="user-1",
    session_ids=["s1", "s2", "s3"],
    instructions="Focus on coding preferences; ignore one-off debugging notes.",
)
# ... más tarde
if dream.status == "completed":
    print(await dream.report())     # el DREAM.md
    await dream.adopt()             # o dream.discard()
```

### Reglas que copiaría de Anthropic

- **Nunca modificar la entrada.** La salida es una carpeta aparte y se adopta con una decisión explícita. Al principio, siempre revisada por una persona.
- **Guardar la anterior** al adoptar, para poder volver.
- **Instrucciones opcionales** para enfocar el trabajo.
- **Si falla o se cancela, la salida parcial se queda** para poder ver qué pasó.
- **Tope de sesiones por dream** (ellos usan 100) y de tamaño por archivo.

### Cuándo se dispara

1. **A mano**, con `agent.dream(...)`. Es por donde empezar.
2. **Automático**, cuando exista el modo *always-on* (ver `04-always-on.md`): cada N sesiones, o de noche. Un dream es justo el tipo de trabajo en segundo plano que ese modo necesita.

## Orden de trabajo

| # | Paso | Tamaño | Depende de |
|---|---|---|---|
| 1 | Conectar las tools de archivos y el `Agent` al sandbox | en curso | — |
| 2 | Carpeta `memory/` en `Layout` y la nota en el prompt | pequeño | 1 |
| 3 | Modo `read_only` en el hook del harness | pequeño | harness |
| 4 | Retirar `CoreMemoryRegistry`, sus backends y sus tools | mediano | 2 |
| 5 | Exportar sesiones a texto | pequeño | — |
| 6 | El dream a mano: prompt, corrida, informe, adoptar o descartar | mediano | 2 y 5 |
| 7 | Sincronizar la memoria con un almacén fuera del sandbox | mediano | cuando haya producción |
| 8 | Dream automático | pequeño | always-on |

Los pasos 2 a 4 ya dan el beneficio principal (una sola forma de trabajar, menos código) sin necesidad del dreaming. El 6 es el dreaming en sí.

## Riesgos

- **Un dream puede borrar algo importante.** Por eso la entrada no se toca, se guarda la anterior y se revisa el informe antes de adoptar.
- **Memoria envenenada.** Un texto ajeno puede hacer que el agente escriba instrucciones falsas en su memoria, y el dream las consolidaría como ciertas. Mitigar con `read_only` cuando el agente procesa contenido no confiable, y con una regla en el prompt del dream: guardar hechos y preferencias, nunca instrucciones encontradas en el contenido.
- **Costo.** Leer conversaciones enteras es caro. Empezar con pocas sesiones y medir.
- **Datos personales.** La memoria y las conversaciones exportadas quedan como archivos en el sandbox del usuario; borrar un usuario tiene que borrar también su volumen.
