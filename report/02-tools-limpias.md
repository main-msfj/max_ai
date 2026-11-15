# 02 — read / write / edit / bash con poco código

## El problema de hoy

Para leer un archivo, la rama tiene esto:

```
filesystem/_base.py        FileSystem(CoreTool): all(), locate()
filesystem/script.py       read/write/edit genérico      (no lo usa nadie)
filesystem/read/__init__.py
filesystem/read/_base.py   ReadFile: __init__, parameters, execute  (86 líneas)
filesystem/read/script.py  read + resolve                 (48 líneas)
filesystem/write/…         igual
filesystem/edit/…          igual
```

Cada tool repite cuatro cosas que no son suyas:

1. El `__init__` que llama a `super().__init__(name=..., description=..., approval_mode=...)`.
2. El esquema JSON escrito a mano.
3. Sacar el sandbox de `deps` y comprobar `None`.
4. Convertir `CommandResult` → `ToolResult` (timeout, exit code, `ok: false`).

Lo único que es de cada tool son 5–15 líneas: qué hace con el archivo.

## La idea: tres capas, cada una hace una sola cosa

```
tool (lo que ve el modelo)      @tool async def read_file(ctx, path, offset, limit)
        │   nombres, descripciones, errores que enseñan
        ▼
FileOps (lo que hace)           ops.read(path) / ops.write(path, text) / ops.edit(...)
        │   stamps, búsqueda del texto, guardado atómico
        ▼
Sandbox (dónde corre)           sandbox.read_bytes(path) / write_bytes(path, data) / run(cmd)
            local: Path.read_bytes      docker/vm: un comando
```

La decisión clave es **dónde va la lógica**. Hoy va dentro de un script que viaja al sandbox. Si en cambio el sandbox solo sabe tres cosas (leer bytes, escribir bytes de forma atómica, correr un comando), toda la lógica queda en Python normal, en el proceso del harness, escrita una vez, testeable sin lanzar procesos, y cada proveedor nuevo implementa tres métodos cortos.

Esto no contradice tu decisión de que "read/write/edit corren en el entorno": los bytes siguen saliendo del entorno. Lo que se queda en el harness es *decidir qué bytes escribir*.

### Lo que hay que resolver: la carrera entre comprobar y escribir

Tu razón para el script era buena: hacer el chequeo del stamp y el `os.replace` en el mismo lugar, sin hueco entre uno y otro. Con primitivas se resuelve dándole el stamp esperado a la escritura:

```python
class Sandbox(Protocol):
    async def read_bytes(self, path: str) -> bytes: ...
    async def write_bytes(self, path: str, data: bytes, *, if_stamp: str | None = None) -> None:
        """Escribe con archivo temporal + replace. Con ``if_stamp``, falla
        con StaleFile si el archivo ya no tiene ese sha256."""
    async def run(self, command: str, *, cwd: str, timeout: float, stdin: bytes | None = None) -> CommandResult: ...
```

- **Local**: `write_bytes` son 8 líneas de Python (leer, comparar hash, `mkstemp`, `os.replace`). Cero procesos.
- **Docker / VM / Modal**: `write_bytes` es **un** comando con el contenido por stdin, y ahí sí va un script mínimo de 10 líneas que compara y reemplaza. Ese script es del proveedor, no de la tool, y es el único.

El hueco entre comparar y reemplazar dentro de `write_bytes` sigue existiendo en teoría (otro proceso escribe en ese microsegundo); es el mismo que tiene tu script hoy y el mismo que tienen Claude Code y opencode. Es aceptable.

## Capa 1 — rutas: una función, un solo lugar

```python
# filesystem/paths.py
PREFIXES = {"$SESSION": "workspace/{session}", "$WORKSPACE": "workspace", "$TMP": "tmp/{session}"}

def resolve(path: str, session_id: str) -> str:
    """``$SESSION/a.txt`` → ``workspace/<session>/a.txt`` (relativa a .agent/).
    Rechaza lo que no empiece con un prefijo o salga de su carpeta."""
    prefix, _, rest = path.partition("/")
    if prefix not in PREFIXES:
        raise ToolRetry(f"{path!r} must start with $SESSION/, $WORKSPACE/ or $TMP/")
    base = PurePosixPath(PREFIXES[prefix].format(session=validate_id("session_id", session_id)))
    full = PurePosixPath(posixpath.normpath(base / rest))
    if not full.is_relative_to(base):
        raise ToolRetry(f"{path!r} leaves {prefix}")
    return str(full)
```

Tres cosas que arregla respecto a `locate()`:

- valida `session_id` (cierra el hueco 2.1-C del archivo 01);
- el error es un `ToolRetry` que le enseña al modelo el prefijo;
- devuelve una ruta relativa a `.agent/`, igual dentro y fuera del sandbox. El sandbox además comprueba con `realpath` que no se salga (por los symlinks); son dos chequeos distintos y hacen falta los dos.

La misma función sirve para que la Policy vea la ruta sin prefijo (arregla 2.1-D).

## Capa 2 — `FileOps`: la lógica, sin saber de tools ni de procesos

```python
# filesystem/ops.py
class FileOps:
    def __init__(self, sandbox: Sandbox, stamps: dict[str, str]):
        self.sandbox, self.stamps = sandbox, stamps   # stamps vive en RunContext

    async def read(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        data = await self.sandbox.read_bytes(path)
        self.stamps[path] = sha256(data).hexdigest()
        lines = data.decode("utf-8").splitlines()
        shown = lines[offset - 1 : offset - 1 + limit]
        body = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(shown, start=offset))
        more = len(lines) - (offset - 1 + len(shown))
        return body + (f"\n[{more} more lines; read again with offset={offset + len(shown)}]" if more else "")

    async def write(self, path: str, content: str) -> str:
        await self.sandbox.write_bytes(path, content.encode(), if_stamp=self._known(path))
        self.stamps[path] = sha256(content.encode()).hexdigest()
        return f"Wrote {len(content.splitlines())} lines to {path}"

    async def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> str:
        stamp = self._known(path, required=True)
        text = (await self.sandbox.read_bytes(path)).decode("utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolRetry("old_string was not found. Read the file again and copy the text exactly.")
        if count > 1 and not replace_all:
            raise ToolRetry(f"old_string appears {count} times. Add surrounding lines or set replace_all.")
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        await self.sandbox.write_bytes(path, updated.encode(), if_stamp=stamp)
        self.stamps[path] = sha256(updated.encode()).hexdigest()
        return f"Replaced {count if replace_all else 1} occurrence(s) in {path}"

    def _known(self, path: str, required: bool = False) -> str | None:
        """El stamp de la última lectura. Editar exige haber leído."""
        if path not in self.stamps and required:
            raise ToolRetry(f"Read {path} before editing it.")
        return self.stamps.get(path)
```

Eso es todo read + write + edit: unas 40 líneas. La cascada estilo opencode (exacto → tipográfico → espacios finales) entra como una función `find(text, old)` que reemplaza a `text.count(old)`, sin tocar nada más.

Nota sobre `write`: con `if_stamp=None` un archivo que el modelo no ha leído se sobreescribe. Si quieres la regla "para sobreescribir hay que haber leído", `write_bytes` necesita distinguir "no existe" de "existe y no lo leíste"; es un parámetro `must_be_new=True` cuando no hay stamp.

## Capa 3 — las tools: lo que ve el modelo

Con el `@tool` que ya tienes, y `ToolContext` con atributos en vez de `deps`:

```python
# filesystem/tools.py
PATH = Annotated[str, Field(description="Starts with $SESSION/ (this session), $WORKSPACE/ (all sessions) or $TMP/ (scratch).")]

def _ops(ctx: ToolContext) -> FileOps:
    return FileOps(ctx.sandbox, ctx.run.file_stamps)

@tool(read_only=True, policy_subject="path")
async def read_file(ctx: ToolContext, path: PATH, offset: int = 1, limit: int = 2000) -> str:
    """Read a text file, optionally a range of its lines."""
    return await _ops(ctx).read(resolve(path, ctx.session_id), offset, limit)

@tool(policy_subject="path")
async def write_file(ctx: ToolContext, path: PATH, content: str) -> str:
    """Create a file, or replace all of its content."""
    return await _ops(ctx).write(resolve(path, ctx.session_id), content)

@tool(policy_subject="path")
async def edit_file(ctx: ToolContext, path: PATH, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact piece of text in a file with new text."""
    return await _ops(ctx).edit(resolve(path, ctx.session_id), old_string, new_string, replace_all)

FILE_TOOLS = [read_file, write_file, edit_file]
```

Tres tools en 20 líneas. Sin `__init__`, sin esquema a mano, sin `NotImplementedError`, sin `__subclasses__()`. El esquema sale de los tipos, el nombre de la función, la descripción del docstring.

### Tres ajustes pequeños al `@tool` para que alcance

1. **Devolver texto plano.** Hoy `tool_message` hace `json.dumps(result.result)` ([dispatcher.py:141-145](../max_ai/core/tool/dispatcher.py#L141-L145)); un `str` se convierte en `"línea1\\nlínea2"` con comillas y escapes, que el modelo lee peor y cuesta más tokens. Si el resultado es `str`, va tal cual.
2. **Tools internas serializables.** `FunctionAsTool.serialize()` lanza `TypeError` porque una función del usuario es código. Las tools del framework no necesitan serializarse: el `Agent` las vuelve a crear. Con no meterlas en `toolset=` basta (ya es así con las built-in).
3. **Errores que no sean `ToolRetry` también deben llegar legibles.** `FileNotFoundError` hoy llega como `Error executing FunctionTool 'read_file': FileNotFoundError(2, 'No such file...')`. Un `except OSError as e: raise ToolRetry(e.strerror)` en `FileOps` lo deja en "No such file or directory".

## bash con la misma forma

`BashTool` hace hoy cuatro cosas: arma el script, lo corre, separa las marcas de la salida y **guarda el cwd en `self._cwd`**. Con el sandbox nuevo:

```python
@tool(policy_subject="command")      # la Policy sigue usando split_command
async def bash(ctx: ToolContext, command: str, description: str) -> str:
    """Run a shell command in the workspace."""
    result = await ctx.sandbox.run(
        wrap(command),                       # wrapper.sh, igual que hoy
        cwd=ctx.run.cwd or resolve("$SESSION/", ctx.session_id),
        timeout=setting.tool_timeout_seconds,
    )
    output, files, cwd = parse(result.stdout)   # las marcas, igual que hoy
    ctx.run.cwd = cwd if inside_workspace(cwd) else None
    return render(result.exit_code, output, files, result.timed_out)
```

Lo que cambia:

- `cwd` vive en `RunContext`: sobrevive a reinicios, a varios procesos, y un subagente puede tener el suyo.
- `WORKSPACE`, `SESSION`, `TMP` como variables de entorno del comando salen de la misma tabla `PREFIXES`. El modelo escribe `cat $SESSION/a.txt` en bash y `$SESSION/a.txt` en `read_file`: el mismo nombre en los dos lados.
- `permission_subjects` y `matches` (lo que hoy sobreescribe `BashTool`) pasan a ser dos parámetros opcionales del decorador, o `bash` sigue siendo la única tool-clase. Las dos opciones están bien; una clase para la tool más especial no es desorden.

`wrapper.sh` y `_permissions.py` no cambian: son la parte buena.

## Qué queda en disco

```
tools/filesystem/
    __init__.py     FILE_TOOLS
    paths.py        resolve(), PREFIXES            ~20 líneas
    ops.py          FileOps                        ~50 líneas
    tools.py        read_file, write_file, edit_file  ~25 líneas
tools/bash/
    _tool.py  _permissions.py  wrapper.sh  constant.py   (como hoy, sin _cwd)
```

Contra hoy: `file_system/` (419 + 99 líneas), `file_system copy/`, `filesystem/` (9 archivos), `workspace/`, más `capabilities/workspace/local/_filesystem.py` (762 líneas) que el diseño viejo necesita.

## Cómo se prueba

- `FileOps` con un `FakeSandbox` que es un diccionario `{ruta: bytes}`: cada caso de edit es un test de tres líneas, sin disco ni procesos.
- `resolve()` con una tabla de entradas maliciosas (`..`, `$SESSION/../x`, `session_id="../.."`, ruta absoluta, sin prefijo).
- Un test de contrato que corre igual contra `LocalSandbox` y, más adelante, `DockerSandbox`: lee, escribe, `if_stamp` viejo falla, symlink hacia fuera falla.

## Orden sugerido, un paso a la vez

1. `paths.resolve()` + sus tests. Cierra los dos fallos de seguridad.
2. `Sandbox.read_bytes` / `write_bytes` en `LocalSandbox`.
3. `FileOps.read` + la tool `read_file`.
4. `write`, luego `edit`.
5. Conectar en el `Agent` (o ya en el `Harness`): `ctx.sandbox`, `ctx.run.file_stamps`.
6. Borrar `file_system/`, la copia, `workspace/` y `RunContext.file_reads`.
