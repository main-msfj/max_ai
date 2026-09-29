1. En la API de Anthropic (versión genérica)
Es una client tool (el modelo no ejecuta nada):

harness responde con un bloque tool_use:JSON{
  "type": "tool_use",
  "name": "bash",
  "input": {
    "command": "ls -la *.py",
    "restart": false   // opcional
  }
}
Tu aplicación ejecuta el comando en una sesión bash persistente.
Devuelves el resultado (stdout + stderr juntos) en un tool_result.
La sesión se mantiene viva entre llamadas → el directorio de trabajo y archivos creados persisten (pero según la implementación, las variables de entorno a veces no).

Documentación oficial: Bash tool.
2. Dentro de harness Code (la implementación real)
Es mucho más compleja. Según análisis del código fuente (v2.1.x, ~500k líneas de TypeScript):
Parámetros que harness puede enviar:

command (obligatorio)
description (recomendado, descripción legible)
timeout (ms, por defecto ~2 min, máximo ~10 min)
run_in_background (boolean)
dangerouslyDisableSandbox (raro)

Pipeline de ejecución (aprox. 7 capas de seguridad + ejecución):

Validación de input (patrones peligrosos, sleep loops, etc.).
Sistema de permisos (bashPermissions.ts + reglas Bash(patrón)).
Modos: ask / allow / deny.
Reglas por proyecto, usuario, políticas, etc.

Análisis de seguridad AST (bashSecurity.ts – miles de líneas).
Detecta inyección, rm -rf, curl | bash, sudo, ofuscación, etc.

Construcción del comando real (bashProvider.ts):
Se carga un snapshot del entorno del usuario (~/.harness/shell-snapshots/...).
Se hace source del snapshot + script de entorno de sesión.
El comando del modelo se envuelve en eval 'comando_entre_comillas'.
Se captura el pwd al final.

Sandbox (opcional pero recomendado):
macOS → sandbox-exec
Linux → bubblewrap + seccomp

Ejecución (runShellCommand como AsyncGenerator):
Proceso separado (cada comando suele ser un shell fresco).
Streaming de output.
Soporte de background tasks.

Post-procesamiento:
Límites de output (~30k caracteres inline en éxito, menos en error; el resto se guarda en archivo).
Interpretación semántica de códigos de salida.
El directorio de trabajo sí persiste entre comandos (si se queda dentro del proyecto).
Las variables de entorno (export) no persisten entre llamadas (cada comando es un shell relativamente fresco, aunque se cargan aliases/funciones del snapshot).


Resumen de persistencia en harness Code:

Working directory → sí (dentro de los directorios permitidos).
Variables de entorno → no (salvo las del snapshot inicial o harness_ENV_FILE).
Archivos creados → sí.

Por qué Bash es tan importante
Es el “escape hatch”. Las herramientas Read/Edit/Grep/Glob son más seguras y estructuradas, pero Bash permite:

Correr tests, builds, linters.
Git (status, commit, push…).
Instalar paquetes.
Cualquier CLI del sistema.
Pipelines complejos.

Por eso tiene la mayor cantidad de código de seguridad de todas las tools (miles de líneas solo de validación).
Si quieres profundizar en alguna parte concreta (permisos, snapshot, sandbox, cómo se ve el schema exacto, o comparación con el code-execution tool de la API), dime y te detallo más.

Osea que el runshellcolmand es esta dentro de un exec h le pasa el string ahí , como se vería una llamada de esa tool

Flujo interno en harness Code

harness emite un tool_use con el nombre Bash y un command (string).
El harness llama a runShellCommand(...) (es un AsyncGenerator).
Dentro de eso se llama a Shell.exec(command, ...).
Shell.exec llama a buildExecCommand(command) → arma el string final.
Ese string final se pasa a spawn / exec del proceso (bash/zsh).

El string que realmente se ejecuta no es el que harness mandó. Se ve más o menos así:
Bashsource /tmp/snapshot-xxxx.sh 2>/dev/null || true
&& source /path/to/session-env.sh 2>/dev/null || true
&& shopt -u extglob 2>/dev/null || true   # o setopt NO_EXTENDED_GLOB
&& eval 'npm test -- --coverage'          # ← el command de harness va aquí, entre comillas simples
&& pwd -P >| /tmp/harness-XXXX-cwd
Por eso usan eval '...': el snapshot se carga antes y los aliases/funciones del usuario ya existen cuando se parsea el comando de harness.

Cómo se ve una llamada de la tool (lo que harness envía)
Ejemplo simple (API / tool_use):
JSON{
  "type": "tool_use",
  "id": "toolu_01A09q90qw90lq917835lq9",
  "name": "Bash",
  "input": {
    "command": "npm test -- --coverage",
    "description": "Run test suite with coverage",
    "timeout": 120000
  }
}
Ejemplo con background:
JSON{
  "type": "tool_use",
  "name": "Bash",
  "input": {
    "command": "npm run dev",
    "description": "Start the development server",
    "run_in_background": true
  }
}
Ejemplo de cadena:
JSON{
  "type": "tool_use",
  "name": "Bash",
  "input": {
    "command": "git add src/utils.ts && git commit -m 'fix: handle null input'",
    "description": "Stage and commit utility fix"
  }
}

Campos que harness puede mandar


1. Schema de entrada (BashTool.tsx)
TypeScript// src/tools/BashTool/BashTool.tsx
const fullInputSchema = lazySchema(() => z.strictObject({
  command: z.string(),
  timeout: semanticNumber(z.number().optional()),
  description: z.string().optional(),
  run_in_background: semanticBoolean(z.boolean().optional()),
  dangerouslyDisableSandbox: semanticBoolean(z.boolean().optional()),

  // NUNCA se expone al modelo — solo lo setea el diálogo de permiso de sed
  _simulatedSedEdit: z.object({
    filePath: z.string(),
    newContent: z.string()
  }).optional()
}))

// Schema público: se oculta _simulatedSedEdit (y a veces run_in_background)
const inputSchema = lazySchema(() =>
  isBackgroundTasksDisabled
    ? fullInputSchema().omit({ run_in_background: true, _simulatedSedEdit: true })
    : fullInputSchema().omit({ _simulatedSedEdit: true })
)

2. runShellCommand — el generador principal
TypeScript// src/tools/BashTool/BashTool.tsx (~líneas 200-280)
async function* runShellCommand({
  input, abortController, setAppState, ...
}): AsyncGenerator<ProgressUpdate, ExecResult, void> {

  const { command, timeout, run_in_background } = input
  const timeoutMs = timeout || getDefaultTimeoutMs()

  // 1. ¿Se permite auto-background?
  const shouldAutoBackground = !isBackgroundTasksDisabled
    && isAutobackgroundingAllowed(command)

  // 2. Ejecutar vía Shell.exec()
  const shellCommand = await exec(command, abortController.signal, 'bash', {
    timeout: timeoutMs,
    onProgress(lastLines, allLines, totalLines, totalBytes) {
      // Despierta el generador para hacer yield de progreso
      resolveProgress?.()
    },
    shouldUseSandbox: shouldUseSandbox(input),
    shouldAutoBackground
  })

  // 3. Esperar umbral inicial (2s) antes de mostrar progreso
  const initialResult = await Promise.race([
    resultPromise,
    new Promise(r => setTimeout(r, PROGRESS_THRESHOLD_MS))
  ])
  if (initialResult !== null) return initialResult  // comando rápido

  // 4. Loop de progreso
  TaskOutput.startPolling(shellCommand.taskOutput.taskId)
  while (true) {
    const result = await Promise.race([resultPromise, progressSignal])
    if (result !== null) return result
    if (backgroundShellId) return backgroundResult
    yield { type: 'progress', output, elapsedTimeSeconds, ... }
  }
}

3. Shell.exec — arma el comando y hace spawn
TypeScript// src/utils/Shell.ts
export async function exec(
  command: string,
  abortSignal: AbortSignal,
  shellType: ShellType,
  options?: ExecOptions,
): Promise<ShellCommand> {
  const {
    timeout,
    onProgress,
    preventCwdChanges,
    shouldUseSandbox,
    shouldAutoBackground,
  } = options ?? {}

  const provider = await resolveProvider[shellType]()
  const id = Math.floor(Math.random() * 0x10000).toString(16)

  // Aquí se construye el string final
  const { commandString: builtCommand, cwdFilePath } =
    await provider.buildExecCommand(command, {
      id,
      sandboxTmpDir: shouldUseSandbox ? sandboxTmpDir : undefined,
      useSandbox: shouldUseSandbox ?? false,
    })

  let commandString = builtCommand
  let cwd = pwd()

  // Si hay sandbox, se envuelve
  if (shouldUseSandbox) {
    commandString = await SandboxManager.wrap(commandString, ...)
  }

  // Spawn real
  const childProcess = spawn(spawnBinary, shellArgs, {
    env: {
      ...subprocessEnv(),
      SHELL: shellType === 'bash' ? binShell : undefined,
      GIT_EDITOR: 'true',
      harnessCODE: '1',
      ...
    },
    stdio: ['pipe', outputHandle.fd, outputHandle.fd], // stdout+stderr al mismo fd
    detached: true,
    windowsHide: true,
  })

  // ...
}

4. buildExecCommand — el string que realmente se ejecuta
TypeScript// utils/shell/bashProvider.ts (conceptual, según el pipeline documentado)

async function buildExecCommand(command: string, opts) {
  // 1. Reescribir redirecciones de Windows
  command = rewriteWindowsNullRedirect(command)  // 2>nul → 2>/dev/null

  // 2. ¿Añadir < /dev/null?
  if (shouldAddStdinRedirect(command)) {
    command = command + ' < /dev/null'
  }

  // 3. Poner entre comillas simples para eval
  const quoted = quoteShellCommand(command)  // singleQuoteForEval

  // 4. Si hay pipe + stdin redirect → reordenar
  //    eval 'rg foo | wc -l' < /dev/null  →  eval 'rg foo' < /dev/null | wc -l

  // 5. Ensamblar el string final
  const parts = [
    snapshotFilePath
      ? `source ${snapshotFilePath} 2>/dev/null || true`
      : null,
    sessionEnvScript
      ? `source ${sessionEnvScript} 2>/dev/null || true`
      : null,
    `shopt -u extglob 2>/dev/null || true`,   // o setopt NO_EXTENDED_GLOB
    `eval ${quoted}`,                         // ← el comando de harness va aquí
    `pwd -P >| ${cwdFilePath}`,
  ].filter(Boolean)

  return {
    commandString: parts.join(' && '),
    cwdFilePath,
  }
}
Resultado concreto cuando harness manda npm test:
Bashsource /tmp/snapshot-zsh-1730...sh 2>/dev/null || true
&& source /tmp/harness-session-env.sh 2>/dev/null || true
&& shopt -u extglob 2>/dev/null || true
&& eval 'npm test'
&& pwd -P >| /tmp/harness-a3f2-cwd

5. Schema de salida
TypeScriptz.object({
  stdout: z.string(),
  stderr: z.string(),           // en la práctica suele ir vacío (stderr se mergea a stdout)
  interrupted: z.boolean(),
  rawOutputPath: z.string().optional(),
  isImage: z.boolean().optional(),
  backgroundTaskId: z.string().optional(),
  backgroundedByUser: z.boolean().optional(),
  assistantAutoBackgrounded: z.boolean().optional(),
  dangerouslyDisableSandbox: z.boolean().optional(),
  returnCodeInterpretation: z.string().optional(),
  noOutputExpected: z.boolean().optional(),
  persistedOutputPath: z.string().optional(),  // si > ~30KB se guarda a disco
  persistedOutputSize: z.number().optional(),
})

Flujo completo en una línea
textharness tool_use {command}
  → runShellCommand()          // AsyncGenerator
    → Shell.exec()
      → buildExecCommand()     // source snapshot + eval 'cmd' + pwd
        → spawn(bash, ["-c", commandString])
Si quieres, te puedo mostrar también el código de los validadores de seguridad (bashSecurity.ts) o el de permisos (bashPermissions.ts).

Así se vería una versión simplificada en Python del flujo de harness Code (schema → runShellCommand → buildExecCommand → exec/spawn):
Pythonimport os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

# ─────────────────────────────────────────────
# 1. Lo que harness envía (tool_use input)
# ─────────────────────────────────────────────

@dataclass
class BashInput:
    command: str
    description: Optional[str] = None
    timeout: int = 120_000          # ms
    run_in_background: bool = False
    dangerously_disable_sandbox: bool = False

# Ejemplo de llamada del modelo:
# {
#   "command": "npm test -- --coverage",
#   "description": "Run test suite with coverage",
#   "timeout": 120000
# }

# ─────────────────────────────────────────────
# 2. buildExecCommand — arma el string real
# ─────────────────────────────────────────────

def single_quote_for_eval(cmd: str) -> str:
    """Envuelve en comillas simples escapando ' internas como '\"'\"'"""
    return "'" + cmd.replace("'", "'\"'\"'") + "'"

def build_exec_command(
    command: str,
    snapshot_path: Optional[str] = None,
    session_env_script: Optional[str] = None,
    cwd_file: Optional[str] = None,
) -> tuple[str, str]:
    """
    Construye el string que realmente se pasa a bash -c.
    Equivalente a bashProvider.buildExecCommand().
    """
    if cwd_file is None:
        cwd_file = os.path.join(tempfile.gettempdir(), f"harness-{uuid.uuid4().hex[:8]}-cwd")

    parts = []

    # 1. Cargar snapshot del entorno del usuario
    if snapshot_path and os.path.exists(snapshot_path):
        parts.append(f"source {single_quote_for_eval(snapshot_path)} 2>/dev/null || true")

    # 2. Cargar env de sesión (harness_ENV_FILE, etc.)
    if session_env_script and os.path.exists(session_env_script):
        parts.append(f"source {single_quote_for_eval(session_env_script)} 2>/dev/null || true")

    # 3. Desactivar extglob (seguridad post-validación)
    parts.append("shopt -u extglob 2>/dev/null || true")

    # 4. eval del comando de harness (segunda pasada de parse para aliases)
    quoted = single_quote_for_eval(command)
    parts.append(f"eval {quoted}")

    # 5. Capturar CWD al final
    parts.append(f"pwd -P >| {single_quote_for_eval(cwd_file)}")

    command_string = " && ".join(parts)
    return command_string, cwd_file

# ─────────────────────────────────────────────
# 3. exec / spawn
# ─────────────────────────────────────────────

@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int
    interrupted: bool = False
    cwd: Optional[str] = None

def shell_exec(
    command: str,
    timeout_ms: int = 120_000,
    snapshot_path: Optional[str] = None,
    cwd: Optional[str] = None,
) -> ExecResult:
    """
    Equivalente a Shell.exec().
    Cada comando = proceso nuevo (shell fresco + snapshot).
    """
    command_string, cwd_file = build_exec_command(
        command,
        snapshot_path=snapshot_path,
    )

    # Lo que realmente se ejecuta:
    # bash -c "source snapshot ... && eval 'npm test' && pwd -P >| /tmp/..."
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command_string],
            capture_output=True,
            text=True,
            timeout=timeout_ms / 1000,
            cwd=cwd or os.getcwd(),
            env={
                **os.environ,
                "harnessCODE": "1",
                "GIT_EDITOR": "true",
            },
        )

        new_cwd = None
        if os.path.exists(cwd_file):
            with open(cwd_file) as f:
                new_cwd = f.read().strip()
            os.unlink(cwd_file)

        # En harness Code stderr se mergea a stdout (mismo fd).
        # Aquí los devolvemos separados por claridad.
        return ExecResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
            cwd=new_cwd,
        )

    except subprocess.TimeoutExpired:
        return ExecResult(
            stdout="",
            stderr="Command timed out",
            return_code=-1,
            interrupted=True,
        )

# ─────────────────────────────────────────────
# 4. runShellCommand — versión sync simple
# ─────────────────────────────────────────────

def run_shell_command(input: BashInput, snapshot_path: Optional[str] = None) -> ExecResult:
    """
    Equivalente simplificado de runShellCommand (sin AsyncGenerator ni background).
    """
    if input.run_in_background:
        # En la versión real: spawn detached + devolver backgroundTaskId
        raise NotImplementedError("Background tasks omitted in this demo")

    return shell_exec(
        command=input.command,
        timeout_ms=input.timeout,
        snapshot_path=snapshot_path,
    )

# ─────────────────────────────────────────────
# 5. Uso
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # harness envió esto:
    tool_input = BashInput(
        command="ls -la *.py",
        description="List Python files",
        timeout=30_000,
    )

    # buildExecCommand produciría algo como:
    cmd_str, _ = build_exec_command(tool_input.command)
    print("=== Comando real que se ejecuta ===")
    print(cmd_str)
    print()

    # Ejecutar
    result = run_shell_command(tool_input)
    print("=== Resultado ===")
    print(f"exit={result.return_code}")
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)
Ejemplo de salida de build_exec_command("npm test")
Bashshopt -u extglob 2>/dev/null || true
&& eval 'npm test'
&& pwd -P >| '/tmp/harness-a3f2b1c0-cwd'
Con snapshot:
Bashsource '/home/user/.harness/shell-snapshots/snapshot-bash-....sh' 2>/dev/null || true
&& shopt -u extglob 2>/dev/null || true
&& eval 'npm test'
&& pwd -P >| '/tmp/harness-a3f2b1c0-cwd'
Versión async (más fiel al AsyncGenerator)
Pythonimport asyncio

async def run_shell_command_async(
    input: BashInput,
    snapshot_path: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Versión async estilo runShellCommand de harness Code.
    Yield de progreso, return final con el resultado.
    """
    command_string, cwd_file = build_exec_command(
        input.command,
        snapshot_path=snapshot_path,
    )

    proc = await asyncio.create_subprocess_exec(
        "/bin/bash", "-c", command_string,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # merge como harness Code
    )

    output_chunks = []
    try:
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
            except asyncio.TimeoutError:
                # yield progreso periódico
                yield {
                    "type": "progress",
                    "output": "".join(output_chunks)[-2000:],
                    "elapsed": ...,
                }
                continue

            if not line:
                break
            text = line.decode()
            output_chunks.append(text)
            yield {"type": "progress", "output": text}

        await proc.wait()
        yield {
            "type": "result",
            "stdout": "".join(output_chunks),
            "return_code": proc.returncode,
        }
    finally:
        if os.path.exists(cwd_file):
            os.unlink(cwd_file)
Resumen del mapeo

harness Code (TS)Pythontool_use.inputBashInputrunShellCommand()run_shell_command() / async generatorShell.exec()shell_exec() → subprocess.run / create_subprocess_execbuildExecCommand()build_exec_command()eval 'cmd'eval {single_quote_for_eval(cmd)}spawn(bash, ['-c', ...])subprocess.run(["/bin/bash", "-c", command_string])