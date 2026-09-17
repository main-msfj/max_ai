# Filesystem de usuario

Las herramientas de archivos usan un espacio por usuario y conversación:

```text
<root>/
└── <user_id>/
    ├── tools/                 # herramientas runtime existentes
    ├── skills/                # skills runtime existentes
    ├── artifacts/             # artifacts legacy
    └── <session_id>/
        ├── informe.txt        # archivos directamente en la conversación
        └── ...
```

Por defecto, `<root>` es `setting.root_dir / "Workspace"`. El agente pasa
`filesystem_root`, `runtime_root` y `artifacts_dir` a las dependencias internas
de las herramientas; `runtime_root` identifica al usuario y `artifacts_dir`
identifica la conversación. Los valores confiables de `user_id` y `session_id`
proceden de `RunContext`, no de parámetros del modelo. Si falta `session_id`,
`Agent.run_stream_events` usa `run_id`, que conserva el mismo espacio al reanudar
la ejecución. Los identificadores se validan como segmentos individuales.

`list_files`, `find_files`, `search_text` y `read_file` exploran o leen el árbol
del usuario usando rutas relativas al usuario. `find_files` busca por basename
o glob y devuelve rutas relativas al usuario, de modo que archivos con el mismo
nombre en conversaciones distintas se pueden distinguir. `write_file` crea un
archivo nuevo con una ruta relativa a la conversación actual. `edit_file`
reemplaza una coincidencia exacta después de verificar el SHA-256; su ruta es
relativa al usuario y puede apuntar a otra conversación. Ambas escrituras
requieren aprobación y solo admiten texto UTF-8 de hasta 1 MiB. No interpretan
ni editan estructuras internas de DOCX, PPT u otros formatos de documento.

`read_file` calcula el SHA-256 leyendo el archivo completo, con un máximo de
8 MiB, y devuelve por defecto un preview de 64 KiB. Se puede pedir otro tamaño
de preview hasta 1 MiB. Los binarios devuelven metadata en vez de contenido
editable. `workspace` lista y lee, con límites, solo la conversación actual;
sus rutas son relativas a esa conversación.

El agente registra estas herramientas de forma predeterminada:

```python
from max_ai.base.agent import Agent
from max_ai.workspace.system import LocalWorkSpace

# Raíz predeterminada: setting.root_dir / "Workspace".
agent = Agent(
    name="assistant",
    description="Asistente",
    instructions="...",
    client=client,
)

# Raíz propia para el runtime y las herramientas de archivos.
custom_agent = Agent(
    name="assistant",
    description="Asistente",
    instructions="...",
    client=client,
    workspace=LocalWorkSpace(root="/srv/max-ai/Workspace"),
)
```

Para construir las herramientas fuera de un agente:

```python
from max_ai.tools import FileSystemTools

tools = FileSystemTools().tools
custom_tools = FileSystemTools("/srv/max-ai/Workspace").tools
```

Una raíz explícita en `FileSystemTools` tiene prioridad sobre la dependencia
`filesystem_root`. Sin raíz explícita, se usa la dependencia del agente y,
si falta, `setting.root_dir / "Workspace"`.

## Límites

- El acceso seguro requiere POSIX, `dir_fd` y `O_NOFOLLOW`; rechaza symlinks,
  hardlinks y archivos que no sean regulares.
- Lecturas y listados tienen límites. Las escrituras son atómicas y usan locks
  locales al proceso; esos locks no coordinan procesos distintos.
- Las herramientas Python personalizadas se ejecutan dentro del proceso local.
  Elegir una raíz propia no añade aislamiento de procesos. El executor de sandbox
  se requiere para herramientas `CoreRuntimeTool`.
- Los artifacts antiguos no se migran automáticamente al nuevo layout. El camino
  legacy de `WorkspaceTool` se mantiene para integraciones que no entreguen
  `filesystem_root`.
- La UI usa la identidad de usuario que ya proporciona el servidor; esta
  integración no añade autenticación nueva.

## Validación

Verificación: **96 pruebas aprobadas** con `uv run pytest`: 86 del runtime,
herramientas, aprobaciones y Docker simulado, más 10 de los endpoints de la UI.
No se ejecutó Docker real. Los cambios de JavaScript se revisaron manualmente;
el entorno no dispone de Node.js para ejecutar `node --check`.
