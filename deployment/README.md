# Pipelines de despliegue

Hay dos pipelines de Azure DevOps, cada uno con su propio YAML:

- `website/azure-pipelines.yml` publica los archivos estáticos del sitio en Azure Static Web Apps al actualizar `main`.
- `library/azure-pipelines.yml` construye y publica la librería Python en Azure Artifacts cuando se crea un tag `v*` (por ejemplo, `v0.1.1`).

## Sitio web

Al crear el pipeline en Azure DevOps, selecciona `deployment/website/azure-pipelines.yml`. En la configuración de variables del pipeline, crea `AZURE_STATIC_WEB_APPS_API_TOKEN` y márcala como secreta. El valor se obtiene de los tokens de despliegue de la Static Web App. Ajusta `appLocation` si los archivos HTML se mueven; para este sitio el valor es `website` y no hace falta compilarlo.

Los parámetros permiten configurar las ubicaciones del sitio y de una API opcional, el entorno de despliegue y el nombre de la variable secreta del token. Un `deploymentEnvironment` vacío despliega a producción.

## Librería

Al crear el pipeline, selecciona `deployment/library/azure-pipelines.yml`. Cambia `artifactFeed` al feed de Azure Artifacts, con formato `PROYECTO/FEED` para un feed de proyecto o `FEED` para un feed de organización. `twineRepository` debe ser el nombre del feed. También puedes ajustar `pythonVersion` y `packagePath` al ejecutar el pipeline.

Concede el rol **Feed Publisher (Contributor)** al Build Service del proyecto y al Project Collection Build Service en los permisos del feed. Antes de crear un tag, actualiza la versión de `[project].version` en `pyproject.toml`; Azure Artifacts no acepta volver a publicar la misma versión.
