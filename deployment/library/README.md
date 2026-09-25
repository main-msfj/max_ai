# Publicación de la librería desde Azure Pipelines

El pipeline [`azure-pipelines.yml`](azure-pipelines.yml) prueba y construye `maxai`
y publica wheel y sdist en **PyPI**. Los usuarios instalan el paquete con
`pip install maxai` o `uv add maxai`.

## Configuración inicial

1. Sube el repositorio completo a Azure Repos, incluidos `deployment/`,
   `pyproject.toml`, `LICENSE`, `NOTICE`, el código y los tests.
2. En Azure DevOps, abre **Pipelines → New pipeline → Azure Repos Git**,
   selecciona el repositorio y **Existing Azure Pipelines YAML file**.
3. Selecciona `/deployment/library/azure-pipelines.yml` en `main` y guarda el
   pipeline con un nombre como `maxai-library`.
4. En PyPI, crea un API token para un proyecto que controles. Para la primera
   publicación de un proyecto nuevo, usa un token de cuenta; después sustitúyelo
   por uno limitado al proyecto `maxai`. El nombre debe estar disponible o debes
   tener permisos sobre el proyecto existente.
5. En las variables del pipeline, crea `PYPI_API_TOKEN`, pega el token y marca
   **Keep this value secret**. No guardes el token en Git.

## Publicar una versión

Actualiza `[project].version` en `pyproject.toml`, confirma los cambios y sube
el commit. Después crea y sube el tag correspondiente al remoto de Azure Repos:

```sh
# Ejemplo para version = "0.1.0"; azure es el nombre de tu remoto de Azure Repos.
git tag v0.1.0
git push azure v0.1.0
```

Solo los tags `v*` disparan la publicación automática. El tag debe coincidir
exactamente con `v` + la versión del paquete. PyPI no permite reutilizar una
versión publicada. El YAML debe existir en el commit etiquetado.

El primer stage instala dependencias con uv, ejecuta Ruff y pytest, construye
wheel y sdist, verifica sus metadatos y ejecuta `twine check --strict`. Guarda
las distribuciones como artefacto; el segundo stage descarga esos mismos
archivos y los publica con Twine. El token solo se pasa al paso de publicación.

Una ejecución manual sobre una rama valida y construye, pero no publica.
Para publicar manualmente, selecciona un tag `v*` en **Run pipeline**.
El parámetro `pythonVersion` permite cambiar Python (predeterminado: `3.11`).

El workflow de GitHub `.github/workflows/publish-library.yml` queda disponible
solo para ejecuciones manuales y conserva su autenticación OIDC; Azure es quien
publica automáticamente los tags. No ejecutes ambos para la misma versión.

Referencia: [Publicar paquetes Python con Azure Pipelines](https://learn.microsoft.com/en-us/azure/devops/pipelines/artifacts/pypi?view=azure-devops).
