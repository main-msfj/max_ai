# Tools nuevas: cómo se hacen, worktrees y dependencias

Guía para escribir una tool en `max_ai`, con `EnterWorktree` y `ExitWorktree` como ejemplo, y cómo pasarle a una tool objetos propios (dependencias), comparado con Pydantic AI.

Las tools de worktree ya existen en `max_ai/capabilities/tools/worktree/` y entran solas cuando el agente tiene bash (`Harness.tool_combination`); el código de la sección 2 es una versión resumida de ellas. Lo de dependencias en el agente sigue siendo una propuesta.

## 1. Cómo se añade una tool

Hay dos formas.

### Una función con `@tool`

Para algo simple que corre en el host. Si el primer parámetro está anotado como `ToolContext`, la función recibe el contexto.

```python
from max_ai.base.tools import ToolContext
from max_ai.capabilities.tools.decorator import tool


@tool
def weather(tool_ctx: ToolContext, city: str) -> str:
    """The weather in a city."""
    return tool_ctx.deps["weather_api"].today(city)
```

Una función no se puede guardar con `agent.serialize()`: es código. Para guardar el agente, la tool va como clase o por un servidor MCP.

### Una clase que hereda de `CoreTool`

Para todo lo demás. Una tool tiene cuatro partes:

| Parte | Qué es |
|---|---|
| `__init__` | nombre, descripción y `approval_mode` (`AUTO_APPROVED` o `ASK_APPROVED`) |
| `parameters` | el JSON Schema de sus argumentos |
| `validate_parameters` (opcional) | lo que el schema no alcanza a revisar |
| `execute(tool_request, tool_ctx)` | hace el trabajo y devuelve un `ToolResult` |

Y tres datos de clase que importan:

- `sandbox = True`: la tool corre dentro del sandbox; `False`: en el host.
- `instructions`: cómo usarla; va al prompt.
- `timeout`: segundos que puede correr una llamada.

Lo que la tool tiene a mano en `tool_ctx`:

- `tool_ctx.run_context`: la conversación (`cwd`, `todos`, `files_seen`, `user_id`, `session_id`).
- `tool_ctx.sandbox`: el sandbox del usuario, ya ligado a él.
- `tool_ctx.deps`: los objetos del developer (sección 3).
- `tool_ctx.emit_event`: para avisar hacia afuera.
- `tool_ctx.cancelled`: el usuario canceló.

### Dónde va

- En el repo: una carpeta en `max_ai/capabilities/tools/<nombre>/` con `__init__.py` y `_toolset.py`. Si manda un script al sandbox, el script va en su propio archivo.
- En el agente: `Agent(..., toolset=[MiTool()])`.

### Reglas del harness que la tool debe cumplir

- Nombre en `snake_case`, máximo 64 caracteres.
- Descripción de máximo 100 tokens (`harness_tool_description_max_tokens`); el detalle va en `instructions` o en los parámetros.
- Todas las tools cargadas juntas, máximo 20 000 tokens (`harness_tools_max_tokens`). Una tool con `deferred=True` no cuenta: se encuentra con `search_tools`.
- Si la tool corre comandos, necesita sandbox.
- Los permisos (`setting.json` y la lista fija del harness) deciden por el argumento `command` o `path`. Una tool nueva que borre o ejecute cosas debe usar `ASK_APPROVED`.

## 2. `EnterWorktree` y `ExitWorktree`

### Qué hacen

Un worktree de git es otra carpeta con otra rama del mismo repositorio. Sirve para que el agente trabaje en una rama aparte sin tocar los archivos que el usuario tiene abiertos.

- `enter_worktree(name)`: crea el worktree con una rama nueva y mueve ahí al agente.
- `exit_worktree(action)`: devuelve al agente a la carpeta de trabajo; `keep` deja el worktree, `remove` lo borra.

### Cómo encaja en lo que ya existe

- **Dónde está el agente** ya vive en `ctx.cwd`: bash arranca ahí y las tools de archivos resuelven las rutas relativas desde ahí. Entrar a un worktree es crear la carpeta y cambiar `ctx.cwd`; salir es poner `ctx.cwd = None` (vuelve a la carpeta de trabajo).
- **El worktree va dentro de la carpeta de trabajo de la sesión** (`<workspace>/.worktrees/<name>`). Así sigue dentro de las carpetas permitidas y bash no lo rechaza.
- **Corren en el sandbox** (`sandbox = True`), con `tool_ctx.sandbox.execute_command(...)`.
- **Después de compactar** el modelo sabe dónde está porque `ctx.cwd` entra en la posición que el loop pone en el prompt.

### Código (resumido)

```python
import shlex
import typing as t

from max_ai.base.tools import CoreTool, ToolContext
from max_ai.types.tool_call import ToolCallRecord, ToolResult
from max_ai.types.tools import ToolApprovalMode


class EnterWorktreeTool(CoreTool):
    """Creates a git worktree for the session and moves the agent into it."""

    TOOL_NAME: t.ClassVar[str] = "enter_worktree"
    DESCRIPTION: t.ClassVar[str] = (
        "Create a separate copy of the repository on a new branch and work there, "
        "so the files of the main folder are not touched."
    )
    FOLDER: t.ClassVar[str] = ".worktrees"
    sandbox: t.ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME, description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string", "pattern": "^[a-z0-9][a-z0-9-]{0,40}$",
                    "description": "Name of the worktree and of its branch.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        box, ctx = tool_ctx.sandbox, tool_ctx.run_context
        if box is None:
            return ToolResult.tool_failure(tool_request.id, "This tool needs a sandbox.")
        # 1. The worktree lives inside the session's workspace.
        workspace = box.session_paths(ctx.session_id or ctx.run_id)["workspace"]
        name = tool_request.parameters["name"]
        target = f"{workspace}/{self.FOLDER}/{name}"
        # 2. Create it on a new branch; git says why when it cannot.
        command = (
            f"cd {shlex.quote(workspace)} && "
            f"git worktree add -b {shlex.quote(name)} {shlex.quote(target)}"
        )
        ran = await box.execute_command(command)
        if ran.exit_code != 0:
            return ToolResult.tool_failure(tool_request.id, ran.stderr or ran.stdout)
        # 3. The agent is there now: bash and the file tools start from it.
        ctx.cwd = target
        return ToolResult.success_result(
            tool_request.id, f"You are now in {target}, on branch {name}."
        )


class ExitWorktreeTool(CoreTool):
    """Takes the agent back to the workspace; keeps or removes the worktree."""

    TOOL_NAME: t.ClassVar[str] = "exit_worktree"
    DESCRIPTION: t.ClassVar[str] = (
        "Go back to the main folder. 'keep' leaves the worktree and its branch; "
        "'remove' deletes the worktree, and fails if it has changes not committed."
    )
    sandbox: t.ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.TOOL_NAME, description=self.DESCRIPTION,
            approval_mode=ToolApprovalMode.ASK_APPROVED,
        )

    @property
    def parameters(self) -> dict[str, t.Any]:
        return {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["keep", "remove"]}},
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
        box, ctx = tool_ctx.sandbox, tool_ctx.run_context
        target = ctx.cwd
        if box is None or not target or f"/{EnterWorktreeTool.FOLDER}/" not in target:
            return ToolResult.tool_failure(tool_request.id, "You are not in a worktree.")
        workspace = box.session_paths(ctx.session_id or ctx.run_id)["workspace"]
        # 1. Remove only when asked; without --force git refuses to lose changes.
        if tool_request.parameters["action"] == "remove":
            command = (
                f"cd {shlex.quote(workspace)} && git worktree remove {shlex.quote(target)}"
            )
            ran = await box.execute_command(command)
            if ran.exit_code != 0:
                return ToolResult.tool_failure(tool_request.id, ran.stderr or ran.stdout)
        # 2. Back to the workspace.
        ctx.cwd = None
        return ToolResult.success_result(tool_request.id, f"You are back in {workspace}.")
```

### Puntos abiertos

- **La carpeta de trabajo tiene que ser un repositorio git.** Si no lo es, `git worktree add` falla y la tool devuelve ese error al modelo.
- **`remove` sin `--force`**: si hay cambios sin commit, git se niega. Es intencional: el modelo debe hacer commit o usar `keep`.
- **Un worktree por sesión o varios**: el boceto permite varios y solo recuerda el actual (`ctx.cwd`). Si hace falta listarlos, `git worktree list`.
- **Permisos**: las dos piden aprobación. La lista fija del harness mira `command` y `path`; estas tools no usan esos argumentos, así que dependen de su `approval_mode`.
- **`.worktrees/` en `.gitignore`** del repositorio, para que no aparezca como archivos sin seguimiento.

## 3. Dependencias

### Cómo lo hace Pydantic AI

- Las dependencias van en un contenedor, casi siempre un `dataclass`.
- El **tipo** se declara en el agente (`deps_type=MyDeps`); sirve solo para el chequeo de tipos.
- La **instancia** se pasa en cada corrida (`agent.run(..., deps=deps)`).
- Las tools, las instrucciones y los validadores la leen de `ctx.deps`, con `RunContext[MyDeps]` como primer parámetro.
- En tests se reemplaza con `agent.override(deps=...)`.

```python
@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent('openai:gpt-6.1-sol', deps_type=MyDeps)

result = await agent.run('Tell me a joke.', deps=MyDeps('foobar', client))
```

### Cómo es hoy en `max_ai`

La pieza ya existe en el dispatcher:

```python
dispatcher = ToolDispatcher(registry, sandbox, deps={"weather_api": api, "db": pool})
```

Cada tool recibe una copia de ese diccionario en `tool_ctx.deps`:

```python
async def execute(self, tool_request: ToolCallRecord, tool_ctx: ToolContext) -> ToolResult:
    rows = await tool_ctx.deps["db"].fetch(tool_request.parameters["query"])
    ...
```

Lo que tiene a favor: las dependencias viven en el host y nunca entran al sandbox ni a sus variables de entorno. Una llave o una conexión que se pasa así no la puede leer un comando de bash.

### Diferencias con Pydantic AI

| | Pydantic AI | `max_ai` hoy |
|---|---|---|
| Contenedor | `dataclass` con tipo | `dict[str, Any]`, sin tipo |
| Dónde se pasa | en cada `run(deps=...)` | al crear el `ToolDispatcher` |
| Quién las lee | tools, instrucciones, validadores | solo las tools |
| Por usuario | sí, cambian en cada corrida | no: son las mismas para todas las corridas |
| En tests | `agent.override(deps=...)` | crear otro dispatcher |

### Lo que falta

El agente nuevo arma su dispatcher por dentro y **no recibe dependencias**: hoy solo llegan a una tool si alguien usa el loop y el dispatcher a mano.

Propuesta, para decidir:

1. `agent.run(task, ctx, deps=...)`, igual en `run_stream` y `run_stream_events`. El agente las pasa al `ToolDispatcher` de esa corrida. Así cambian por usuario, que es lo que necesita una API que atiende a muchos.
2. `Agent(..., deps=...)` para las que no cambian (un cliente HTTP compartido). Las de la corrida ganan sobre las del agente.
3. Aceptar un `dataclass` además del diccionario, para tener tipos: `tool_ctx.deps` devolvería el objeto tal cual.
4. Las dependencias nunca se guardan con `serialize()`: son objetos vivos.

Regla que se mantiene: una dependencia con secretos no se manda al sandbox. Si una tool la necesita dentro de un comando, la tool hace la llamada en el host y manda al sandbox solo el resultado.
