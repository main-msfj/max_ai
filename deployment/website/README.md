# Despliegue del website en Azure

El website es HTML/CSS/JavaScript estático y se despliega a **Azure Static Web Apps** con el pipeline de Azure DevOps [`azure-pipelines.yml`](azure-pipelines.yml). El pipeline se activa en `main` cuando cambian `website/**` o `deployment/website/**`; la librería se publica por separado con `deployment/library/azure-pipelines.yml`.

En **Pipelines → New pipeline → Azure Repos Git → Existing Azure Pipelines YAML file**, selecciona `deployment/website/azure-pipelines.yml`. Crea la variable secreta `AZURE_STATIC_WEB_APPS_API_TOKEN` con el token de despliegue de la Static Web App. El nombre de la variable es configurable con el parámetro `apiTokenVariableName`.

Parámetros disponibles:

| Parámetro | Valor inicial | Uso |
| --- | --- | --- |
| `appLocation` | `website` | Carpeta de archivos estáticos. |
| `apiLocation` | vacío | Carpeta de Azure Functions si se agrega una API. |
| `outputLocation` | vacío | Carpeta de salida; vacío porque no hay build frontend. |
| `deploymentEnvironment` | vacío | Entorno de Static Web Apps; vacío despliega a producción. |
| `apiTokenVariableName` | `AZURE_STATIC_WEB_APPS_API_TOKEN` | Nombre de la variable secreta del token. |

Las páginas públicas están agrupadas en `website/overview/`, `agents/`, `basecomponents/`, `core/`, `capabilities/`, `types/`, `events/`, `examples/` y `cli/`. El pipeline publica `website/` directamente, sin compilar una aplicación frontend. Antes de publicar cambios generados, ejecuta desde la raíz del repositorio:

```sh
python website/scripts/generate_capability_pages.py
python website/scripts/generate_example_pages.py
python website/scripts/generate_reference_pages.py
python website/scripts/sync_navigation.py
python website/scripts/build_search_index.py
```

El token de despliegue debe guardarse en Azure DevOps como variable secreta `AZURE_STATIC_WEB_APPS_API_TOKEN`.

El despliegue solo se ejecuta desde `main`, incluso en ejecuciones manuales. Las validaciones de pull requests no despliegan. Crea primero el recurso Azure Static Web Apps (origen Other) y copia su token desde Manage deployment token. La URL pública aparece en Overview del recurso.
