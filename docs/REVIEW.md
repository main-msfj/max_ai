> Copia de la revisión del framework al 2 nov 2026. La versión viva es la Claude Doc "Revisión de max_ai: dónde mejorar".

# Revisión de max\_ai: dónde mejorar

Oct 29, 2026 · @Marvin G

Siete problemas críticos (seguridad, cuelgues y concurrencia) deben corregirse antes que nada; el resto son mejoras de rendimiento, diseño y limpieza.

Alcance: repo [main-msfj/max\_ai](https://github.com/main-msfj/max_ai), commit `9859f6f` (\~30k líneas en `max_ai/`). Revisión estática: los tests no se ejecutaron porque PyPI no era alcanzable desde el entorno de revisión.

## Estado de avance

12 de 20 puntos resueltos y 1 en parte, todos en la rama [claude/keen-ride-k2y3mm](https://github.com/main-msfj/max_ai/tree/claude/keen-ride-k2y3mm) (mergeados en main con los PR #6 a #9). #1, #2, #10 y #9 quedan para después.

Las tablas van ordenadas por estado: pendientes arriba, luego en parte y los resueltos al final.

| # | Problema | Estado | Commit | Nota |
| --- | --- | --- | --- | --- |
| 1 | Bash filtra secretos del entorno | Pendiente |  | Siguiente: usar run\_process con entorno limpio |
| 2 | Timeout de bash puede colgar el run | Pendiente |  | Siguiente: matar el grupo de procesos |
| 3 | Stream duplicado al reintentar | Pendiente |  |  |
| 6 | Escrituras concurrentes en el workspace | Pendiente |  |  |
| 9 | Race en prepare() | Pendiente |  | Menor: ambos runs asignan la misma lista; solo es trabajo repetido |
| 10 | Bash carga toda la salida a memoria | Pendiente |  | Se arregla junto con #1 y #2 |
| 12 | Memoria cargada 2 veces por run | Pendiente |  | a) \_memory\_for 2 veces: trivial. b) sync remota archivo por archivo: solo importa con Azure/MinIO, junto con #6. c) \_locks por usuario sin límite: trivial |
| 13 | isinstance en el dispatcher | Pendiente |  |  |
| 16 | Empaquetado y API pública | En parte | 23b9945 | Falta: ollama/openai como extras, nombre maxai vs max\_ai |
| 4 | Reintentos en doble capa | Hecho | dbfe702 | Solo el loop reintenta (max\_retries=0 en el SDK); ClientError lleva status y retry\_after; no reintenta 403/404/422; jitter y tope de 60 s |
| 5 | MCP colgado congela todo | Hecho | 9511867 | Timeout al conectar (30 s), servidores aislados en paralelo, reintento cada 60 s, required=True opcional |
| 7 | Tool calls rotas en streaming | Hecho | d4d5f20 | Streaming marca parsing\_error como el modo normal |
| 8 | Conteo de tokens O(n²) | Hecho | 3d4dcfb | TokenCounter cuenta cada mensaje una vez (cache por mensaje); el thinking ya no se cuenta (PR #9) |
| 14 | Mensajes permanentes del gate | Hecho | 417d445 | El rechazo del gate solo va en la siguiente llamada; el aviso de límite de iteraciones se queda guardado a propósito (PR #9) |
| 15 | Skills nuevas no llegan al workspace | Hecho |  | Ya resuelto en el código: \_sync\_skill actualiza por hash con manifiesto |
| 17 | eval\_criteria y gate mutado | Hecho | 23b9945 |  |
| 18 | Docstrings autogenerados | Hecho | 9a56334 | 390 borrados, 7 recortados |
| 19 | Documentación desactualizada | Hecho | 4f1f894 |  |
| 20 | Lista deny de bash cosmética | Hecho | 364e8b0 | Deny ve rutas, wrappers y bash -c; warning en LocalExecutor |
| 11 | Resultados de tool con tokens de más | Hecho | 5bcad82 | Bash devuelve solo exit\_code y stdout; stderr y datos de corte solo cuando hay algo (\~50 tokens menos por comando, PR #9) |

### Otros hallazgos

Puntos fuera de la lista original, encontrados al revisar el código.

| Hallazgo | Estado | Commit | Nota |
| --- | --- | --- | --- |
| El agente necesita internet la primera vez: tiktoken descarga o200k\_base al contar tokens | Pendiente |  | Importa para despliegues sin internet; toca el core, queda para después |
| El README decía que skills con LocalExecutor lanza un error de seguridad; esa validación no existe | Hecho | e24e207 | README corregido (PR #8). Bloquearlo de verdad rompería examples/basic/10\_skills.py |
| 51 docstrings de relleno de una línea que quedaron tras #18 | Hecho | 5a05785 | Solo funciones; los de clase se usan como descripción del componente (PR #8) |
| Test del CLI test\_sessions aparentemente roto | Hecho |  | No es un bug: falla por la descarga de tiktoken, como los otros 105 |

### Feedback externo (2 nov 2026)

Una segunda revisión confirma 3 bugs con tests (#21, #22 y el #3 ya listado: el test obtiene "Hola Hola mundo") y añade riesgos en workspace y skills. Orden sugerido por esa revisión: #21, #22, #23, #24 y después #3; el #21 va antes del trabajo de steering, porque "detener" lo dispara constantemente.

| # | Problema | Dónde | Estado | Fix y nota |
| --- | --- | --- | --- | --- |
| 21 | Detener un run en medio de una tool rompe el historial: los resultados del batch se escriben solo al terminar todo el batch. Si se cancela o una llamada lanza error, se pierden, los records quedan consumidos y el siguiente mensaje sale con un tool call sin respuesta (OpenAI responde 400) | `core/tool/dispatcher.py:89–116` y `:319–321` | Pendiente | Emitir siempre un resultado por record ("cancelado" o "falló") antes de re-lanzar; `gather(..., return_exceptions=True)`; al iniciar el run, reparar los tool calls sin respuesta desde su record. Confirmado con test |
| 22 | Si el proveedor en streaming no manda id de tool call, el fallback `openai_tool_call_{index}` se repite en la segunda iteración y `ToolState.add` lanza "already tracked". Afecta OpenRouter y servidores tipo vLLM | `clients/openai/client.py:456`, `core/tool/state.py:107` | Pendiente | Usar `short_id()` en el fallback, como ya hace el modo sin streaming; reasignar el id en el loop si ya existe. Confirmado con test |
| 23 | Un workspace por usuario compartido por todas las conversaciones y agentes; el lock es por usuario y sesión, y solo dentro del proceso. Dos pestañas o subagentes escriben la misma carpeta; entre pods gana la última subida | `base/workspace.py`, `workspace/_remote.py`, `agents/agent.py:647–651` | Pendiente | Carpeta de trabajo por sesión o subagente, con un paso explícito que promueve resultados a los archivos compartidos del usuario. Amplía el #6 y responde su pregunta abierta |
| 24 | Las skills son editables y sus cambios persisten; las de distintos agentes se mezclan en la misma carpeta, que no se sincroniza entre pods. Un run con prompt injection puede modificar una skill que otra sesión ejecuta después | `base/skills.py:105–138` | Pendiente | Skills de solo lectura y versionadas por agente; nunca guardar cambios del sandbox en ellas. Choca con el diseño actual de `_sync_skill`, que respeta a propósito las ediciones locales; encaja con meter las skills en la imagen |
| 25 | Cada run hashea todo el workspace del usuario dos veces (`_scan` al bajar y al subir) | `workspace/_remote.py` | Pendiente | Bien por ahora; caro con muchos usuarios y archivos. Va con el #12b y el plan de backup incremental en la VM |
| 26 | El steering se manda como system message a mitad de la conversación; algunas plantillas de modelos abiertos solo aceptan uno al inicio y la API de Anthropic no acepta el rol system dentro de messages | `reasoning/react/loop.py` (`transient_messages`) | Pendiente | Probar con los modelos reales; donde no se soporte, mapearlo a un mensaje de usuario etiquetado. No verificado con test |
| 27 | No hay tope genérico de tamaño para resultados de tools: bash trunca, pero MCP y las tools de funciones no, y una respuesta grande llena el contexto | `core/tool/dispatcher.py:98–104` | Pendiente | Tope en el dispatcher para cualquier tool. No verificado con test |
| 28 | Al llegar al tope de iteraciones el turno termina sin respuesta; el modelo solo explica en el turno siguiente | `reasoning/react/loop.py:370–375` | Pendiente | Una llamada final sin tools que resuma el avance. No verificado con test |

## Críticos (bugs o seguridad)

Los dos de bash (#1 y #2) son los más urgentes: uno filtra secretos y el otro puede colgar un run para siempre.

| # | Problema | Dónde | Fix |
| --- | --- | --- | --- |
| 1 | Bash pasa todo el entorno del host (`OPENAI_API_KEY`, URIs de Mongo…) a una shell que controla el LLM; un prompt injection con `env \| curl` las exfiltra | `tools/bash/_tool.py:260` (`os.environ.copy()`) | Allowlist de variables: `PATH`, `HOME`, `LANG`, `WORKSPACE`, `SCRATCHPAD`, `SKILLS` |
| 2 | El timeout solo mata a `sh`; si un hijo mantiene el pipe (`server &`), `await proc.communicate()` espera para siempre. El `CancelledError` se traga sin re-lanzar | `_tool.py:327` y `:333` | `start_new_session=True`, `os.killpg(pgid, SIGKILL)`, timeout en el `communicate()` final, re-lanzar `CancelledError` |
| 3 | Si el stream falla tras emitir chunks, el reintento los emite otra vez: el usuario ve la respuesta duplicada | `base/reasoning.py`, `_call_llm_stream` | No reintentar si ya salió un chunk, o emitir un evento de reset |
| 4 | `AsyncOpenAI` reintenta 2 veces por defecto más tus 3: hasta 12 llamadas. `retry_after` se parsea pero se ignora; backoff sin jitter; `api_error` reintenta 403/404/422 | `clients/openai/client.py:134` y `:759` | `AsyncOpenAI(max_retries=0)`, respetar `retry_after`, jitter, reintentar solo 408/409/429/5xx |
| 5 | `await ready` sin timeout bajo el `_mcp_lock` global: un servidor MCP colgado congela todos los runs. Si uno falla, `connect_all` desconecta todos, incluidos los que usan otros runs | `mcp/client_manager.py` (`connect`, `connect_all`) | `asyncio.wait_for` en la conexión, fallos aislados por servidor, seguir sin sus tools con un warning |
| 6 | Workspace por usuario pero lock por (usuario, sesión): dos sesiones escriben a la vez. `upload` es last-writer-wins sin ETag, así que con varias réplicas se pierden cambios | `base/workspace.py`, `workspace/_remote.py` | Workspace por sesión o lock por usuario en escrituras; escrituras condicionales (`If-Match`) |
| 7 | Tool calls con JSON roto: sin streaming se marcan con `parsing_error`; en streaming se descartan en silencio y el gate puede tomar el mensaje vacío como respuesta final | `base/reasoning.py:~1032` | Mismo camino (`parsing_error`) en ambos modos |

## Rendimiento

El mayor costo es la re-tokenización del historial completo en cada iteración, que además bloquea el event loop.

| # | Problema | Dónde | Fix |
| --- | --- | --- | --- |
| 8 | `_input_messages_with_token_counts` cuenta tokens sobre copias y nunca guarda el conteo: O(n²). tiktoken es síncrono y bloquea el loop para runs concurrentes. `_model_input_messages` se calcula 2 veces por llamada | `base/reasoning.py:586` | Cachear `token_count` en los mensajes originales; calibrar la compactación con `usage.prompt_tokens` real |
| 9 | `prepare()` corre en cada run y muta `self._skill_blocks`, compartido entre runs concurrentes (race condition) | `agents/agent.py:489` | Una sola vez por agente, con lock |
| 10 | Bash lee toda la salida a memoria antes de truncar; un archivo de 2 GB puede tumbar el proceso | `tools/bash/_tool.py` | Leer el stream con límite |
| 11 | Cada resultado de tool lleva `cwd`, el comando repetido y `*_original_chars`: tokens de más | `core/tool/dispatcher.py:102` | Formato compacto para el modelo |
| 12 | `_memory_for` 2 veces por run; sync remota archivo por archivo con `read_bytes` síncrono; `_locks` por usuario crece sin límite | `agent.py`, `workspace/_remote.py` | Una llamada; `gather` con semáforo y `to_thread`; locks con expiración |

## Diseño y API

El dispatcher y el empaquetado son los puntos que más limitan a quien extienda o instale el framework.

| # | Problema | Dónde | Fix |
| --- | --- | --- | --- |
| 13 | Casos especiales `isinstance(AskUserTool)` e `isinstance(BashTool)` con imports perezosos; una tool custom no puede tener aprobación dinámica | `core/tool/dispatcher.py:190-236` | Hook en la tool, tipo `tool.resolve_approval(record)` |
| 14 | El rechazo del gate y `MAX_ITERATIONS` escriben mensajes permanentes en `ctx.messages`, contra el diseño de steering transitorio; `SystemMessage` a mitad del transcript rompe con Anthropic y Gemini | `reasoning/react/loop.py:386` y `:605` | Hacerlos transitorios |
| 15 | Las skills solo se copian si no existen: una versión nueva nunca llega al workspace | `base/skills.py:131` | Comparar por hash o versión |
| 16 | Sin `max_ai/__init__.py` ni `__version__`; distribución `maxai` vs import `max_ai`; imports muy profundos; `flask` es dependencia core y no se usa; `ollama` y `openai` son obligatorias | `pyproject.toml`, paquete raíz | API pública curada (`from max_ai import Agent, OpenAIClient, tool`); quitar `flask`; `ollama`/`openai` como extras. Hecho en parte en 23b9945: \_\_init\_\_.py con \_\_version\_\_ y exports (Agent, OpenAIChatCompletionClient, tool…), flask quitado. Pendiente: ollama/openai como extras y el nombre maxai vs max\_ai. |
| 17 | `eval_criteria` en la firma abstracta sin uso; `RuntimeCompletionGate.bind_workspace` muta la instancia del usuario y rompe si se comparte entre agentes | `base/reasoning.py`, `completion_gate/gate.py` | Quitar el parámetro; copiar el gate al enlazarlo. Hecho en 23b9945. |

## Calidad de código

Los tres puntos están resueltos en la rama claude/keen-ride-k2y3mm (sin PR todavía).

- [x] **#18 Docstrings sin valor.** \~700 "Value supplied for `x`" y 246 "Perform the … operation", con indentación rota, en 61 archivos. Hecho en 9a56334: 390 docstrings borrados y 7 recortados (se conservó el texto real); sin cambios en docstrings de clase ni en descripciones de tools.
- [x] **#19 Documentación desactualizada.** El docstring de `agent.py:3` habla de una "Parallel copy of base/agent.py" que ya no existe; hay un comentario en español mezclado en el docstring de `ReactLoop._register_tool_calls`. Hecho en 4f1f894: nota obsoleta quitada y frase traducida.
- [x] **#20 La lista `deny` de bash es cosmética.** `/usr/bin/sudo`, `env sudo` o `bash -c "sudo …"` caen en "ask", no en "deny". Con `LocalExecutor` y un patrón permitido como `python:*`, el modelo ejecuta código arbitrario en el host, lo que anula el blindaje del filesystem. Usar Docker por defecto fuera de desarrollo, o mostrar un warning ruidoso. Hecho en 364e8b0: deny y ask ven a través de rutas, VAR=x, wrappers (env, nohup, timeout, xargs…) y bash -c / eval; allow sigue literal. El Agent avisa una vez cuando usa LocalExecutor; el default no se cambió a Docker para no romper a quien no lo tiene instalado.

## Fortalezas y por dónde empezar

La base es sólida; empezar por #1, #2, #5, #3/#4 y #8, en ese orden: son cambios chicos con mucho impacto.

Lo que ya está bien:

- Filesystem blindado contra path traversal y symlinks (`dir_fd` + `O_NOFOLLOW`).
- Allowlist de providers al deserializar (`_check_allowed`).
- Diseño serializable con pausa y resume bien pensado, y steering transitorio.
- Batching de tools read-only.
- \~70 archivos de tests con CI en Python 3.11 y 3.12.

Orden sugerido:

- [ ] \#1 Allowlist de entorno en bash
- [ ] \#2 Matar el grupo de procesos en timeout y cancelación
- [ ] \#5 Timeout y aislamiento de fallos en MCP
- [ ] \#3 y #4 Reintentos: sin duplicar stream, sin doble capa, respetando `Retry-After`
- [ ] \#8 Cachear conteo de tokens
