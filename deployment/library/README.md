# Publicación de la librería en PyPI desde GitHub

El workflow activo está en [`.github/workflows/publish-library.yml`](../../.github/workflows/publish-library.yml). GitHub Actions construye y publica el paquete en **PyPI**; Azure Artifacts ya no interviene. PyPI es el índice que usan `pip` y `uv` por defecto, de modo que, tras la primera publicación, los usuarios podrán ejecutar:

```sh
pip install maxai
uv add maxai
```

## Configuración inicial

1. En GitHub, crea el environment `pypi` para `main-msfj/max_ai`. Puedes limitarlo a tags de release protegidos.
2. En PyPI, registra un [Trusted Publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) para el proyecto `maxai`. Si aún no existe, crea un *pending publisher*. Usa estos valores exactos: owner `main-msfj`, repository `max_ai`, workflow `publish-library.yml`, environment `pypi`.
3. Asegúrate de que `pyproject.toml`, `LICENSE` y `NOTICE` estén incluidos en el commit que se etiquetará.

La publicación usa OIDC de GitHub y [la acción oficial de PyPA](https://docs.pypi.org/trusted-publishers/using-a-publisher/). No requiere guardar un token de PyPI en GitHub.

## Cuándo publica

Un push de un tag `v*` dispara el workflow. También se puede ejecutar manualmente desde **Actions → Publish Python library to PyPI**, indicando un tag existente en `tag`. El tag debe coincidir exactamente con `v` + `[project].version` de `pyproject.toml`; por ejemplo, la versión `0.1.0` usa `v0.1.0`. PyPI no permite volver a subir la misma versión: aumenta `[project].version` para cada release.

El job de build instala dependencias, ejecuta lint y tests, genera wheel y sdist, y verifica nombres y metadatos. Solo después, un job separado con permiso OIDC (`id-token: write`) publica los dos archivos en PyPI.

## Parámetros

- `workflow_dispatch.inputs.tag`: tag existente para una ejecución manual.
- Variable de repositorio `RELEASE_PYTHON_VERSION`: versión de Python del build; predeterminada `3.11`.
- `[project].version` en `pyproject.toml`: versión publicada, que debe coincidir con el tag.

La primera publicación requiere que PyPI acepte el nombre `maxai` y que el Trusted Publisher coincida exactamente con el workflow y environment. Hasta entonces, `pip install maxai` y `uv add maxai` no instalarán este proyecto desde PyPI.
