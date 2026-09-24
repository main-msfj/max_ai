# Despliegues

Hay dos rutas independientes:

- [Website](website/README.md): Azure DevOps despliega el HTML estático de `website/` a Azure Static Web Apps mediante `deployment/website/azure-pipelines.yml`.
- [Librería](library/README.md): GitHub Actions prueba y empaqueta `maxai`, y publica wheel y sdist en PyPI desde un tag `v*`. El workflow está en `.github/workflows/publish-library.yml`.

Después de la primera publicación en PyPI, los usuarios podrán instalar la librería con `pip install maxai` o agregarla a un proyecto con `uv add maxai`. Los parámetros y pasos de preparación de cada despliegue están en su carpeta.
