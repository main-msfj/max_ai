# Despliegues

- [Librería](library/README.md): GitHub Actions prueba y empaqueta `maxai`, y publica wheel y sdist en PyPI desde un tag `v*`. El workflow está en `.github/workflows/publish-library.yml`.

Después de la primera publicación en PyPI, los usuarios podrán instalar la librería con `pip install maxai` o agregarla a un proyecto con `uv add maxai`. Los parámetros y pasos de preparación están en [library/README.md](library/README.md).
