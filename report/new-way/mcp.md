# Feedback: `capabilities/mcp`

Revisado el 2026-11-15 sobre la rama `feat/workspace-new-approach`. Es una revisión por lectura: salvo donde se indica, no ejecuté los casos.

**Resumen:** la parte difícil (mantener la conexión viva, reconectar, no repetir llamadas que cambian algo) está bien hecha. Los problemas son de seguridad y de varios usuarios, no de código: un servidor stdio corre en tu máquina fuera del sandbox, el servidor decide solo qué se aprueba sin preguntar, y todos los usuarios comparten una conexión y un token.

| Archivo | Líneas | Estado |
|---|---|---|
| `_model.py` | 138 | bien |
| `client_manager.py` | 381 | bien, con dos fallas |
| `tool.py` | 307 | funciona; código repetido y una transformación que pierde datos |
| `transport.py` | 35 | bien; mezcla `httpx2` con `httpx` |
| `integration.py` | 29 | solo lo usa un test |
| `config.py` | 15 | sobra |

## Lo que está bien

- **Los secretos nunca se guardan.** Los campos `*_env` nombran la variable y se resuelven al conectar; `token` está excluido y `serialize_mcp_servers` rechaza un token literal.
- **Una tarea es dueña de cada conexión** y cada llamada corre en su propia subtarea. Es la forma correcta con el SDK, y las llamadas en paralelo no se esperan entre sí.
- **Si la conexión se cae a media llamada, solo se repiten las de solo lectura.** Una que cambia algo le dice al modelo que quizá sí corrió.
- **Un servidor caído no se vuelve a intentar durante `retry_after_seconds`**, y `required` decide si la corrida falla o sigue sin él.
- **Las tools de MCP son diferidas por defecto**, así que no llenan el prompt.

## Problemas importantes

### 1. Un servidor stdio corre en tu máquina, fuera del sandbox

`StdioMCPServerConfig(command=..., args=...)` lanza ese comando en el proceso del harness, con acceso a tu disco y a tu red. El sandbox no interviene.

Esto pesa más por cómo usas el framework: los agentes te llegan como JSON o YAML. Quien escribe ese archivo elige el `command`, así que cargar un agente ajeno equivale a ejecutar un programa suyo en tu servidor.

Opciones, de menos a más trabajo:

- En producción aceptar solo servidores HTTP.
- Una lista de comandos permitidos, definida en el código y no en el archivo.
- Lanzar el servidor stdio dentro del sandbox. Hoy no se puede: `execute_command` espera a que el comando termine, y un servidor necesita una conexión que quede abierta.

### 2. El servidor decide qué se aprueba sin preguntar

`_approval_mode_for_tool` pone `AUTO_APPROVED` cuando el servidor declara `read_only_hint`. Esas marcas las escribe el servidor y nadie las comprueba: un servidor que miente consigue que sus tools corran sin aprobación. También se usan para decidir qué llamadas se repiten tras un corte.

Recomiendo que las marcas solo cuenten si la config dice que confías en ese servidor (por ejemplo `trusted: bool = False`). Sin eso, manda `approval_mode` y `tool_approval_modes`.

### 3. Todos los usuarios comparten una conexión y un token

El `Agent` tiene un solo `MCPClientManager` y los headers se resuelven una vez, al conectar, desde las variables del proceso. Un agente que sirve a varios usuarios habla con el servidor MCP como una sola identidad. Es el mismo problema que señalaste con las variables de las skills.

No hay que resolverlo ahora, pero el diseño debería dejarle lugar: lo natural es lo mismo que en el sandbox, una copia por usuario (`manager.bind(identifier, tokens)`).

### 4. Las descripciones del servidor entran al prompt sin límite

Nombres y descripciones de tools, y la lista completa de recursos en `MCPResourceTool._build_description`, vienen del servidor y llegan al modelo tal cual. Un servidor con 500 recursos produce una descripción enorme, y es texto ajeno dentro del prompt. Falta un límite de tamaño por descripción y por lista.

## Fallas

- **`{"result": x, ...}` pierde datos.** En `MCPTool.execute`, si la respuesta es un diccionario con la clave `results` o `result`, se devuelve solo esa clave y el resto se descarta. Una tool que responde `{"result": 5, "warnings": [...]}` pierde los avisos.
- **Los fallos de red por HTTP no se reintentan.** `transport.py` crea el cliente con `httpx2`, pero `_unreachable` en `client_manager.py` busca `httpx.TransportError`. Comprobé que `httpx2.TransportError` no es una subclase de `httpx.TransportError`, así que un error de conexión de `httpx2` no cuenta como "vale la pena reintentar". No lo probé contra un servidor caído.
- **`disconnect` puede fallar con un error viejo.** Si la conexión murió con una excepción, `await worker` la vuelve a lanzar, y `disconnect_all` se detiene ahí sin cerrar los demás servidores. Visto por lectura, no lo ejecuté.
- **Dos tools pueden terminar con el mismo nombre.** `_sanitize_name` convierte `a-b` y `a_b` en lo mismo, y el registro rechaza el duplicado al arrancar. Tampoco recorta el nombre: los proveedores suelen aceptar hasta 64 caracteres.
- **Las imágenes llegan como texto.** `_extract_tool_result` las devuelve en base64 dentro del JSON, así que el modelo recibe una cadena larga que además se corta por el medio. Es la misma limitación de `tool_message` en el dispatcher.
- **Una tool que el servidor quitó sigue registrada.** Al reconectar, el `Agent` reemplaza las que cambiaron pero no retira las que ya no existen. Tampoco se atiende el aviso del servidor de que su lista cambió.
- **El tiempo límite se aplica de dos formas**: `call_tool` se lo pasa al SDK y `read_resource` usa `asyncio.wait_for`.

## Limpieza

- **`config.py` sobra.** Dice "compatibility imports" y repite lo que ya exporta `__init__.py`. Sin usuarios todavía, no hay nada que mantener compatible.
- **`integration.py`** tiene una función de 29 líneas que solo usa un test; el `Agent` usa el manager directo. Borrarla o hacer que el `Agent` la use.
- **`tool.py` repite unas 40 líneas** entre `MCPTool` y `MCPResourceTool`: validar, registrar y la cadena de `except`. Las dos vuelven a validar parámetros que el dispatcher ya validó.
- **`env` y `transport_protocol` están en la clase base** pero `env` solo sirve a stdio.
- **Dos reglas distintas para un token literal.** MCP se niega a serializar un `token` pasado en código; `GitHubSkills` lo acepta y al serializar escribe el nombre de la variable. Conviene una sola regla para todo el framework.
- **MCP no es un componente.** Todo lo demás se serializa con `ComponentBase`; MCP tiene sus propias `serialize_mcp_servers` y `deserialize_mcp_servers`.

## Cómo encaja con lo nuevo

```
Agent
 ├─ sandbox   ← copia por usuario (bind)
 ├─ skills    ← llegan al sandbox
 ├─ tools     ← dispatcher + Policy
 └─ mcp       ← hoy: una conexión para todos, stdio fuera del sandbox
```

MCP es la única pieza que todavía no tiene idea de "por usuario" ni de "dentro del sandbox". No bloquea lo que estamos haciendo, pero los puntos 1 y 2 conviene cerrarlos antes de aceptar agentes de terceros.

## Orden recomendado

| # | Cambio | Tamaño |
|---|---|---|
| 1 | Marcas del servidor solo si `trusted` | pequeño |
| 2 | stdio: solo HTTP en producción, o lista de comandos permitidos | pequeño |
| 3 | Fallas: `result`/`results`, `httpx2`, `disconnect`, nombres | pequeño |
| 4 | Límite de tamaño a descripciones y lista de recursos | pequeño |
| 5 | Limpieza: borrar `config.py`, unir lo repetido en `tool.py` | pequeño |
| 6 | Una copia del manager por usuario | grande, junto con producción |
