# MCP: estado y pendientes

Actualizado el 2026-11-16 sobre la rama `feat/workspace-new-approach`. La primera versión de este archivo (2026-11-15) era una revisión por lectura; esta recoge lo que ya se corrigió, lo que se añadió y lo que falta.

**Resumen:** las correcciones de seguridad y las fallas pequeñas están hechas. Además MCP ahora maneja archivos (imágenes y binarios), progreso de llamadas largas y tareas en segundo plano. Lo que falta está fuera de MCP: el loop tiene que recoger las tareas terminadas, y `agent.py` tiene que pasar la configuración.

| Archivo | Qué hace |
|---|---|
| `_model.py` | configuración de servidores; los secretos nunca se guardan |
| `client_manager.py` | conexiones, reintentos, límites de tiempo, tareas |
| `tool.py` | `MCPTool` y `MCPResourceTool`: adaptan un tool o recurso del servidor |
| `task.py` | tareas en segundo plano (nuevo) |
| `transport.py` | crea el cliente del SDK según el tipo de servidor |
| `integration.py`, `config.py` | sobran; ver Limpieza |

## Lo que ya estaba bien

- **Los secretos nunca se guardan.** Los campos `*_env` nombran la variable y se resuelven al conectar; `serialize_mcp_servers` rechaza un token literal.
- **Una tarea de asyncio es dueña de cada conexión** y cada llamada corre en su propia subtarea, así que las llamadas en paralelo no se esperan entre sí.
- **Si la conexión se cae a media llamada, solo se repiten las de solo lectura.** Una que cambia algo le dice al modelo que quizá sí corrió.
- **Un servidor caído no se vuelve a intentar durante `retry_after_seconds`**, y `required` decide si la corrida falla o sigue sin él.
- **Las tools de MCP son diferidas por defecto**, así que no llenan el prompt.

## Lo que se corrigió

| Problema | Cómo quedó |
|---|---|
| El servidor decidía qué se aprueba sin preguntar | Sus marcas (`read_only_hint`) solo cuentan si la config dice `trusted=True`. Por defecto manda `approval_mode`. |
| Un servidor stdio es un programa que corre en tu máquina | `MCPClientManager(stdio_commands=[...])` limita qué comandos se aceptan. `None` permite cualquiera; una lista vacía, ninguno. |
| `{"result": x, "warnings": [...]}` perdía los avisos | Solo se desenvuelve cuando `result` o `results` es la única clave. |
| Los fallos de red por HTTP no se reintentaban | `_unreachable` reconoce también `httpx2.TransportError`. |
| `disconnect_all` se detenía si una conexión había muerto con error | Cierra todas aunque una falle. |
| El límite de tiempo se leía de `setting` | Son parámetros del manager; nada de MCP lee `setting`. |

## Lo que se añadió

### Imágenes y binarios

Un tool de MCP puede devolver imágenes, audio o archivos. Antes llegaban al modelo como texto en base64.

- `MCPTool` separa el texto de los archivos: el texto va en el resultado y los archivos en `ToolResult.files`.
- `ToolHook.keep_files` los guarda en `tmp/<sesión>/tool_results/` del sandbox y añade la ruta al texto. Tope de 20 MB por archivo.
- Sin sandbox, el modelo recibe un aviso de qué archivo era y que no se guardó.
- El modelo recibe hoy solo texto con la ruta. Pasarle la imagen directamente está decidido pero no construido (ver Pendientes).

### Progreso y límites de tiempo

```python
MCPClientManager(idle_timeout=360, max_timeout=600, task_wait=60)
```

- **`idle_timeout`**: segundos que una llamada puede pasar sin dar señales. Cada aviso de progreso del servidor reinicia la cuenta.
- **`max_timeout`**: tope total de una llamada, avise o no.
- Al cortarse, el modelo recibe un error normal del tool (`MCP tool 'x' of 'server' stopped answering.`) y la conexión sigue sirviendo.
- `MCPTool` convierte cada aviso en un `MCPProgressEvent`, que sale por el mismo flujo de eventos que los demás.
- El dispatcher respeta el `timeout` que declara cada tool; `MCPTool` declara el `max_timeout` del manager.

Probado contra un servidor real por stdio: una llamada de 5 segundos con avisos termina bien aunque el límite sin señales sea 3; una callada se corta a los 3; una que avisa sin parar se corta en el tope total.

### Tareas en segundo plano

Algunos servidores pueden correr un tool como tarea: responden enseguida con un id y el trabajo sigue en el servidor. Es la extensión `io.modelcontextprotocol/tasks` del protocolo `2026-07-28`.

```
1. el modelo llama un tool
2. el servidor devuelve una tarea  ──►  el manager espera hasta task_wait (60 s)
       terminó en ese rato   ──►  resultado normal; el modelo no nota la diferencia
       no terminó            ──►  3
3. MCPTool anota la tarea en RunContext.mcp_tasks y responde:
       "The tool is still working on the server. Its result will arrive when it is ready."
4. más tarde:  manager.collect_tasks(ctx)
       sigue trabajando  ──►  se queda anotada
       terminó o falló   ──►  se borra de la lista y se devuelve su resultado
       el servidor ya no la tiene, o caducó  ──►  se borra y se devuelve None
```

- **La lista vive en `RunContext.mcp_tasks`**, no en el texto de la conversación. Sobrevive a guardar y cargar la sesión y a una compactación, y no depende de que el modelo se acuerde de consultar.
- **`collect_tasks(ctx)`** funciona desde un manager nuevo y una conexión nueva: es el caso "termino, y más tarde el usuario dice hola".
- **`cancel_task(ctx, task_id)`** pide al servidor que la detenga y la borra.
- **El resultado caduca.** El servidor dice cuánto lo guarda (15 minutos por defecto en FastMCP); pasado ese tiempo ya no se le pregunta.
- **Una tarea que pide datos al usuario a mitad de camino** se cierra con error; este cliente no los contesta.
- Los servidores sin tareas no notan nada.

Probado contra un servidor real de tareas (FastMCP 4.2, instalado aparte y no como dependencia del proyecto): tarea corta, tarea larga recogida después desde otra conexión, tarea que falla, id desconocido y cancelación.

## Decisiones tomadas

- **Las descripciones de los tools no se recortan.** Quien conecta un servidor MCP asume ese riesgo. Lo que se controla es cuánto entra al prompt: los tools son diferidos y `search_tools` enseña solo nombre y descripción.
- **La consulta de tareas es automática**, no un tool que el modelo deba llamar. Un modelo débil no consultaría.
- **Los resultados grandes** los corta o guarda el `ToolHook` general (más de 40,000 caracteres), igual que para cualquier tool.

## Pendientes

### Fuera de MCP

| Dónde | Qué falta |
|---|---|
| Loop | Llamar a `collect_tasks(ctx)` antes de cada llamada al modelo y meter cada resultado como mensaje `[harness]`. Sin esto la tarea queda anotada y nadie la recoge. |
| `agent.py` | Pasar `stdio_commands`, `idle_timeout`, `max_timeout` y `task_wait`. |
| Client del modelo | Pasar la imagen al modelo cuando lee imágenes. Decidido: se queda en el historial y, tras una compactación, se cambia por "image omitted" con la ruta. |
| Producción | Una copia del manager por usuario. Hoy todos los usuarios comparten una conexión y un token. |

### Dentro de MCP

- **Dos tools pueden terminar con el mismo nombre.** `_sanitize_name` convierte `a-b` y `a_b` en lo mismo y no recorta a los 64 caracteres que suelen aceptar los proveedores.
- **Un tool que el servidor quitó sigue registrado.** Tampoco se atiende el aviso del servidor de que su lista cambió.
- **Un servidor stdio corre fuera del sandbox.** La lista de comandos lo limita, pero no lo aísla.
- **Nadie muestra el progreso todavía.** El evento existe; falta que el CLI lo pinte.

### Limpieza

- `config.py` repite lo que ya exporta `__init__.py`.
- `integration.py` tiene una función que solo usa un test.
- `tool.py` repite unas 40 líneas entre `MCPTool` y `MCPResourceTool`.
- MCP se serializa con funciones propias en vez de `ComponentBase`.

## Pruebas

| Archivo | Qué cubre |
|---|---|
| `tests/mcp/test_mcp_safety.py` | `trusted`, lista de comandos stdio, errores de red, desenvolver `result`, imágenes |
| `tests/mcp/test_mcp_progress.py` | progreso, corte por silencio, tope total, evento de progreso |
| `tests/mcp/test_mcp_tasks.py` | tarea anotada en `RunContext`, `collect_tasks` |
| `tests/mcp/test_mcp_integration.py`, `test_mcp_unavailable.py` | conexión, reintentos, servidor caído |

`tests/mcp/test_agent_mcp.py` falla hasta que `agent.py` use el dispatcher nuevo.
