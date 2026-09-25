# Infraestructura local

Requiere Docker Engine y Docker Compose 2.20.3 o superior, por el uso de `include`.

```text
docker-infra/
├── compose.yaml           # Reúne los grupos
├── .env.example           # Imágenes, puertos y credenciales locales
├── llm/compose.yaml       # Ollama
├── capabilities/compose.yaml  # Lo que usan los agentes: MongoDB, Mongo Express,
│                              # MinIO y Azurite (workspaces)
├── observability/compose.yaml # Langfuse y sus dependencias (con su propio MinIO)
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

`pull` descarga las imágenes del perfil MCP. `up` inicia MongoDB, Mongo Express, Azurite, MinIO y Ollama; DuckDuckGo se
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

## Capabilities: MongoDB

Con las credenciales predeterminadas, la URI desde el host es:

```text
mongodb://maxai:maxai-local-dev@localhost:27017/max_ai?authSource=admin&directConnection=true
```

La imagen es `mongodb/mongodb-atlas-local`: MongoDB con el motor de búsqueda
(`mongot`), así que la búsqueda vectorial (`$vectorSearch`) de memoria y
knowledge funciona en local. Es un replica set de un nodo: la URI necesita
`directConnection=true`, o el driver intenta el nombre interno del contenedor.

MongoDB guarda los datos en el volumen `mongodb-atlas-data` y la configuración en
`mongodb-atlas-config`. Los volúmenes `mongodb-data`/`mongodb-config` de la
imagen `mongo:8.0` anterior quedan sin uso; bórralos con `docker volume rm` si
no necesitas esos datos. La base `max_ai` se crea cuando la aplicación escribe datos.
El usuario se inicializa en `admin` durante el primer arranque con un volumen
vacío; cambiar `.env` después no cambia las credenciales del usuario existente.

### Ver los datos con Mongo Express

Abre <http://localhost:8081> e inicia sesión con los valores predeterminados:

- Usuario web: `maxai`
- Contraseña web: `maxai-local-ui`

El primer acceso muestra el diálogo de autenticación HTTP del navegador.
Una respuesta `401 Unauthorized` antes de iniciar sesión es normal; las
credenciales anteriores corresponden a la interfaz web, no a MongoDB.

Puedes explorar las bases, abrir una colección y ver sus documentos. La base
`max_ai` aparecerá cuando la aplicación haya guardado datos. La interfaz también
permite editar documentos y colecciones.

Mongo Express espera a que MongoDB esté saludable y se conecta por la red
`capabilities`, usando el nombre `mongodb` y las credenciales de la base. El acceso
web tiene credenciales independientes, configurables con
`MONGO_EXPRESS_USERNAME` y `MONGO_EXPRESS_PASSWORD`; el puerto se cambia con
`MONGO_EXPRESS_PORT`.

### Desde un devcontainer

`localhost` dentro del devcontainer es el propio devcontainer, no el host de
Docker. El script `.devcontainer/post-start.sh` conecta el devcontainer a las
redes de infraestructura que ya existan y abre puentes locales para las
interfaces en los puertos 8081 y 3000. Si levantaste Compose después de abrir
el devcontainer, conecta la red de capabilities desde su terminal:

```bash
docker compose -f docker-infra/compose.yaml up -d --wait mongodb mongo-express
docker network connect max-ai-infra_capabilities "$(hostname)"
export MONGODB_URI='mongodb://maxai:maxai-local-dev@mongodb:27017/max_ai?authSource=admin&directConnection=true'
```

La URI mostrada usa las credenciales locales predeterminadas; ajústala si
cambiaste `docker-infra/.env`. El nombre `mongodb` resuelve solo desde
contenedores unidos a esa red. Dentro del devcontainer, Mongo Express responde
en `http://localhost:8081` mediante el puente local. La publicación de Compose
escucha únicamente en `127.0.0.1` del host de Docker.

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
en su UI. Son cinco contenedores (web, worker, Postgres, ClickHouse y Redis) y
guarda sus eventos en su propio MinIO (`langfuse-minio`, separado del de los
agentes); van en el perfil
`observability` y no arrancan con el `up` normal. Postgres, ClickHouse y Redis
son de Langfuse, no del framework (Langfuse no soporta MongoDB):

```bash
docker compose --profile observability up -d --wait
```

- UI: http://localhost:3000 — usuario `admin@maxai.local`, contraseña `maxai-local-ui`.
- Ollama API: http://localhost:11434/api/tags; Mongo Express: http://localhost:8081.
- El primer arranque crea la organización `MaxAI`, el proyecto `max_ai` y sus
  claves (`pk-lf-maxai-local` / `sk-lf-maxai-local`). Cambiarlas en `.env`
  después no cambia las ya creadas.
- La UI (3000) se publica en `127.0.0.1`. Postgres, ClickHouse y Redis quedan
  dentro de la red `observability`.

Estas direcciones `localhost` corresponden al host de Docker. Dentro del
devcontainer, Mongo Express y Langfuse también responden en
`http://localhost:8081` y `http://localhost:3000`, respectivamente, gracias
a los puentes locales. Las direcciones de servicio siguen disponibles en
`http://mongo-express:8081`, `http://langfuse-web:3000`,
`http://minio:9001` y `http://ollama:11434/api/tags` si el contenedor
está conectado a las redes de Compose. VS Code reenvía los puertos 8081 y 3000
del devcontainer al equipo local al abrir o recargar la ventana.

Para enviar trazas desde el agente, en el `.env` de la raíz del repositorio:

```text
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-maxai-local
LANGFUSE_SECRET_KEY=sk-lf-maxai-local
```

Si el agente corre dentro del devcontainer, configura
`LANGFUSE_HOST=http://langfuse-web:3000` y conecta el devcontainer a la red
`max-ai-infra_observability`. Las credenciales de ejemplo son para desarrollo
local; utiliza las de tu `.env` si las cambiaste.

### Abrir las interfaces desde VS Code

Si VS Code está conectado a un devcontainer o a una máquina remota,
`localhost` en el navegador de tu equipo no necesariamente es el host de
Docker. Ejecuta **Developer: Reload Window** en VS Code para aplicar
`forwardPorts` de `.devcontainer/devcontainer.json`. Después abre la pestaña
**Ports** del panel inferior y usa **Open in Browser** sobre **Mongo Express**,
**Langfuse** o **MinIO Console**. Usa la dirección local que muestre esa
pestaña: puede tener un puerto distinto o un dominio reenviado.

Si un puerto no aparece, en **Ports** elige **Forward a Port** y escribe
`8081` para Mongo Express o `3000` para Langfuse. Desde el propio
devcontainer puedes comprobar los puentes con `curl -I http://localhost:8081`
(responde 401 hasta iniciar sesión) y `curl -I http://localhost:3000`
(responde 200). Si no responden, ejecuta
`bash /max_ai/.devcontainer/post-start.sh` en el devcontainer para reconectar
las redes y reiniciar los puentes.

Con esas variables, los agentes de `example-for-deployment/` activan el
tracing solos. En código: `provider = configure_langfuse()` y
`Agent(..., middlewares=[TracingMiddleware()])`.

Para detenerlo: `docker compose --profile observability down` (agrega `-v` para
borrar también las trazas guardadas).

## Capabilities: workspaces remotos (MinIO y Azurite)

`AzureBlobWorkspace` y `MinIOWorkspace` guardan los archivos del usuario fuera
del proceso: el agente los descarga al empezar cada run y sube lo que cambió al
terminar, así otra instancia (otro servidor, otra invocación serverless) los ve.

| Servicio | URL | Credenciales |
|---|---|---|
| Azurite (Azure Blob) | `http://localhost:10000/devstoreaccount1/<contenedor>` | cuenta `devstoreaccount1`, clave de desarrollo de Azurite |
| MinIO (S3) | API `http://localhost:9000`, consola http://localhost:9001 | `maxai` / `maxai-local-dev` |

```python
from max_ai.capabilities.workspace import AzureBlobWorkspace, MinIOWorkspace

AzureBlobWorkspace("http://localhost:10000/devstoreaccount1/workspaces")  # lee $AZURE_STORAGE_KEY
MinIOWorkspace("http://localhost:9000", bucket="workspaces")  # lee $MINIO_ACCESS_KEY / $MINIO_SECRET_KEY
```

La clave de Azurite es pública (la misma para todos):
`Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==`.
En Azure real, `blob_url` es `https://<cuenta>.blob.core.windows.net/<contenedor>`
y `AZURE_STORAGE_KEY` la clave de la cuenta o un token SAS.

## Probar todo junto

Con todo arriba, `example-for-deployment/travel_agent.py` abre la CLI con
búsqueda web (Exa), el skill de spreadsheet, el sandbox de Modal, los archivos
del usuario en MinIO, las sesiones en MongoDB y trazas en Langfuse:

```bash
docker compose --profile observability up -d --wait
cd .. && .venv/bin/python example-for-deployment/travel_agent.py
```

Si ejecutas este ejemplo desde un devcontainer, usa primero la conexión a la
red y `MONGODB_URI` de la sección anterior; el valor por defecto del ejemplo
apunta a `localhost` y no llega al MongoDB de Compose desde otro contenedor.

Los datos se ven en Mongo Express (http://localhost:8081, usuario `maxai`,
contraseña `maxai-local-ui`) y las trazas en Langfuse (http://localhost:3000).
