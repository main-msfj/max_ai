# Publicación de la librería con GitHub Actions

El workflow [`.github/workflows/publish-library.yml`](../../.github/workflows/publish-library.yml)
prueba, construye y publica `maxai` en PyPI con autenticación OIDC. Se ejecuta
manualmente desde GitHub Actions con un tag de versión existente.

1. Configura el entorno `pypi` en GitHub y registra este repositorio como
   publicador de confianza para el proyecto `maxai` en PyPI.
2. Actualiza `[project].version` en `pyproject.toml`, confirma el cambio y sube
   el commit y su tag correspondiente a GitHub, por ejemplo `v0.1.0`.
3. En **Actions → Publish Python library to PyPI → Run workflow**, indica ese tag.

El workflow obtiene el commit del tag, instala dependencias, ejecuta Ruff y
pytest, construye wheel y sdist, comprueba el tag y los metadatos con
[`verify_release.py`](verify_release.py), y publica las distribuciones en PyPI.
Una versión ya publicada no se puede reutilizar.
