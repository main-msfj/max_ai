# Infraestructura local

Requiere Docker Engine y Docker Compose 2.20.3 o superior, por el uso de `include`.

```text
docker-infra/
├── compose.yaml           # Reúne los tres grupos
├── .env.example           # Imágenes, puertos y credenciales locales
├── llm/compose.yaml       # Ollama
├── backend/compose.yaml   # MongoDB y Mongo Express
└── mcp/compose.yaml       # DuckDuckGo MCP por stdio

llm-models/ollama/          # Datos de Ollama, fuera de Git
```

## Descargar y levantar

Desde la raíz del repositorio:

```bash
cd docker-infra
cp .env.example .env
docker compose --profile mcp pull
docker compose up -d --wait
```

`pull` descarga las cuatro imágenes. `up` inicia MongoDB, Mongo Express y Ollama; DuckDuckGo se
inicia desde el cliente MCP con el comando de la sección siguiente. Cada grupo
tiene su propia red. Los puertos publicados están enlazados a `127.0.0.1`.

Los valores de `.env.example` son para desarrollo local. Puedes cambiar imágenes,
puertos y credenciales en `docker-infra/.env`. Las etiquetas `latest` permiten
actualizar con `pull`; para fijar una versión, configura una etiqueta o digest.

## LLM: Ollama

La API queda en `http://localhost:11434`. La configuración inicial usa CPU.
Los datos de `/root/.ollama` se conservan en `llm-models/ollama/`, en la raíz del
repositorio, incluso después de eliminar el contenedor.

La imagen del servidor no incluye modelos. Para descargar uno, por ejemplo el
usado en el quickstart del proyecto:

```bash
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama list
```

## Backend: MongoDB

Con las credenciales predeterminadas, la URI desde el host es:

```text
mongodb://maxai:maxai-local-dev@localhost:27017/max_ai?authSource=admin
```

MongoDB guarda los datos en el volumen `mongodb-data` y la configuración en
`mongodb-config`. La base `max_ai` se crea cuando la aplicación escribe datos.
El usuario se inicializa en `admin` durante el primer arranque con un volumen
vacío; cambiar `.env` después no cambia las credenciales del usuario existente.

### Ver los datos con Mongo Express

Abre <http://localhost:8081> e inicia sesión con los valores predeterminados:

- Usuario web: `maxai`
- Contraseña web: `maxai-local-ui`

Puedes explorar las bases, abrir una colección y ver sus documentos. La base
`max_ai` aparecerá cuando la aplicación haya guardado datos. La interfaz también
permite editar documentos y colecciones.

Mongo Express espera a que MongoDB esté saludable y se conecta por la red
`backend`, usando el nombre `mongodb` y las credenciales de la base. El acceso
web tiene credenciales independientes, configurables con
`MONGO_EXPRESS_USERNAME` y `MONGO_EXPRESS_PASSWORD`; el puerto se cambia con
`MONGO_EXPRESS_PORT`.

Para agregarlo a una instalación existente, desde `docker-infra`:

```bash
docker compose up -d --wait mongo-express
```

Este comando también inicia MongoDB si hace falta. Si ya tienes `.env`, puedes
añadir las variables `MONGO_EXPRESS_*` de `.env.example`; los valores anteriores
funcionan por defecto sin reemplazar tu configuración.

Se fija la imagen `mongo-express:1.0.2-20-alpine`. Docker Hub marca esta imagen
como discontinuada; esta configuración la publica únicamente en localhost
para desarrollo local. Consulta la [ficha de la imagen](https://hub.docker.com/_/mongo-express).

## MCP: DuckDuckGo

La imagen `mcp/duckduckgo` del catálogo de Docker ofrece `search` y
`fetch_content`. Su transporte es `stdio`: el cliente inicia el contenedor y
conversa por stdin/stdout. Por eso usa el perfil `mcp`, sin publicar un puerto.

Comando para un cliente MCP, desde `docker-infra`:

```bash
docker compose run --rm -T --no-deps duckduckgo
```

`-T` evita que un terminal altere el protocolo. No añadas `-d`: el cliente necesita
mantener stdin/stdout conectados. Cada conexión inicia su propio contenedor y
`--rm` lo elimina al terminar. El perfil se activa al indicar el servicio.

Para conectarlo a un agente Max AI que corre en el host, desde la raíz del repo:

```python
from pathlib import Path

from max_ai.capabilities.mcp import StdioMCPServerConfig

compose_file = Path("docker-infra/compose.yaml").resolve()
duckduckgo = StdioMCPServerConfig(
    server_id="duckduckgo",
    command="docker",
    args=[
        "compose", "-f", str(compose_file),
        "run", "--rm", "-T", "--no-deps", "duckduckgo",
    ],
)

# Pasa esta configuración al construir tu agente:
# Agent(..., mcp=[duckduckgo])
```

El proceso del agente necesita acceso a Docker. La ruta absoluta permite iniciar
el comando MCP desde otro directorio después de construir la configuración.

## Operación

Desde `docker-infra`, puedes administrar cada servicio por separado:

```bash
docker compose up -d --wait ollama
docker compose up -d --wait mongodb
docker compose up -d --wait mongo-express
docker compose ps
docker compose logs -f ollama mongodb mongo-express
docker compose down
```

`down` conserva los modelos y los volúmenes de MongoDB. Usa siempre el Compose
principal para mantener el mismo proyecto y los mismos volúmenes.

## Referencias

- [Compose include](https://docs.docker.com/compose/how-tos/multiple-compose-files/include/)
- [Ollama en Docker](https://docs.ollama.com/docker)
- [Imagen oficial de MongoDB](https://hub.docker.com/_/mongo)
- [Mongo Express](https://github.com/mongo-express/mongo-express-docker)
- [DuckDuckGo en el catálogo MCP](https://hub.docker.com/mcp/server/duckduckgo/overview)
- [Dockerfile de DuckDuckGo: transporte stdio](https://github.com/nickclyde/duckduckgo-mcp-server/blob/8992977d65a086995c82826ceead42e890aa17c1/Dockerfile)

## Observabilidad: Langfuse

Langfuse recibe las trazas de `TracingMiddleware` (OpenTelemetry) y las muestra
en su UI. Son seis contenedores (web, worker, Postgres, ClickHouse, Redis y
MinIO), por eso van en el perfil `observability` y no arrancan con el `up`
normal:

```bash
docker compose --profile observability up -d --wait
```

- UI: http://localhost:3000 — usuario `admin@maxai.local`, contraseña `maxai-local-ui`.
- El primer arranque crea la organización `MaxAI`, el proyecto `max_ai` y sus
  claves (`pk-lf-maxai-local` / `sk-lf-maxai-local`). Cambiarlas en `.env`
  después no cambia las ya creadas.
- Solo la UI (3000) y MinIO (9090, para archivos adjuntos) se publican, en
  `127.0.0.1`. Postgres, ClickHouse y Redis quedan dentro de la red `observability`.

Para enviar trazas desde el agente, en el `.env` de la raíz del repositorio:

```text
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-maxai-local
LANGFUSE_SECRET_KEY=sk-lf-maxai-local
```

Con esas variables, `examples/01_agent_with_openai.py` y `02_...` activan el
tracing solos. En código: `provider = configure_langfuse()` y
`Agent(..., middlewares=[TracingMiddleware()])`.

Para detenerlo: `docker compose --profile observability down` (agrega `-v` para
borrar también las trazas guardadas).
