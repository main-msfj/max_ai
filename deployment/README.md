# Despliegues desde Azure Pipelines

El repositorio completo puede vivir en Azure Repos. Crea dos pipelines mediante
**Pipelines → New pipeline → Azure Repos Git → Existing Azure Pipelines YAML file**:

| Pipeline | Archivo YAML | Activación | Variable secreta |
| --- | --- | --- | --- |
| Librería → PyPI | `/deployment/library/azure-pipelines.yml` | Tags `v*` | `PYPI_API_TOKEN` |
| Web → Azure Static Web Apps | `/deployment/website/azure-pipelines.yml` | Cambios en `website/**` o `deployment/website/**` en `main` | `AZURE_STATIC_WEB_APPS_API_TOKEN` |

Guarda cada variable en su pipeline y marca **Keep this value secret**.
La web requiere un recurso Azure Static Web Apps y su token de despliegue.
La librería requiere permiso para publicar el paquete `maxai` en PyPI.

Una ejecución manual de la librería sobre una rama solo valida y construye;
para publicar, selecciona un tag de versión. La web solo despliega desde `main`.
Los pipelines requieren un agente Linux; usan `ubuntu-latest`. La organización
Azure DevOps debe tener capacidad disponible para ejecutar jobs hospedados.

Consulta [librería](library/README.md) y [web](website/README.md) para los pasos
completos. No hace falta separar carpetas con `.gitignore` ni crear repositorios
Git anidados.
