# Cómo construimos la tool `bash` en max_ai

Este documento explica el diseño de la tool `bash` del harness max_ai: qué hace el
modelo, qué hace el harness, y cómo se conectan el workspace, las skills y el
entorno de ejecución. Está pensado para reimplementarlo en otro proyecto.

---

## 1. La idea central

> **El modelo solo manda un comando. Todo lo demás lo decide el harness.**

El modelo manda esto:

```json
{"command": "python make.py", "description": "Generar el reporte"}
```

No manda rutas, carpetas, timeouts, variables de entorno ni en qué máquina corre.
El harness decide todo eso:

| Lo decide el harness                         | Cómo                                                        |
|----------------------------------------------|-------------------------------------------------------------|
| Dónde están los archivos del usuario         | El workspace (sección 3)                                     |
| En qué máquina corre el comando              | El executor: local, Docker o Modal (sección 4)              |
| En qué carpeta arranca                       | La carpeta donde terminó el comando anterior (`cwd` persiste) |
| Cuánto puede tardar                          | Un solo `TOOL_TIMEOUT_SECONDS` global (360 s), fijo          |
| Cuánto output ve el modelo                   | 30 KB; lo largo se corta al medio y se guarda completo       |
| Si el comando se permite, se pregunta o se niega | Políticas allow / ask / deny (sección 8)                 |
| Qué archivos cambió el comando               | El harness lo detecta solo (sección 6)                       |

La razón: todo lo que se le pide al modelo lo hace mal a veces. Si el modelo tiene
que acordarse de poner `$WORKSPACE/`, de declarar los archivos que va a crear o de
usar un scratchpad, a veces se olvida. Si lo hace el harness, pasa siempre.

---

## 2. El recorrido de un comando

```
modelo:  bash({"command": "python make.py", "description": "..."})
   │
   ▼
1. Validar        schema, comando no vacío, deny_patterns  → si choca: error, nunca llega al humano
2. Permiso        allow / ask / deny; "ask" pausa el turno y el usuario aprueba (HITL)
3. Sesión         EnvironmentManager.acquire(): conecta el sandbox si hace falta
4. Sync ↓         host → sandbox: solo los archivos que cambiaron
5. Script         build_script(): valores + wrapper.sh
6. Ejecutar       executor.execute(session, script, timeout)  →  bash -c  con timeout
7. Wrapper        imprime output (cortado), archivos cambiados y cwd final
8. Sync ↑         sandbox → host: lo que el comando creó, cambió o borró
9. Resultado      {"exit_code": 0, "output": "...", "files": {...}}
10. Gate          al cerrar el turno: ¿comandos fallidos sin resolver? ¿archivos temporales?
```

Solo `bash` entra al sandbox. Todas las demás tools (las de archivos, `ask_user`,
el plan, MCP, las tools Python del desarrollador) corren en el proceso del agente.

---

## 3. El workspace: dónde viven los archivos

El workspace es la copia **oficial** de los archivos del usuario. Siempre está en el
host (la máquina donde corre el agente).

```
.agents/<usuario>/                    ← LocalWorkspace (por defecto)
.agents/minio-<id>/<usuario>/         ← MinIOWorkspace / AzureBlobWorkspace (copia local)
    workspace/     los archivos del usuario: compartidos entre todas sus conversaciones
    skills/        las skills del agente, copiadas en cada corrida
```

- **LocalWorkspace**: el disco local es la fuente de verdad.
- **MinIO / Azure**: la fuente de verdad es el bucket. El agente trabaja sobre una
  copia local en `.agents/`: la baja al empezar cada turno (`download`) y sube los
  cambios al terminar (`upload`). Solo viaja lo que cambió (manifest con SHA-256).
- Las file tools (`ReadFile`, `WriteFile`, `EditFile`, ...) trabajan siempre sobre
  esta copia del host. El modelo pasa rutas cortas (`reporte.xlsx`), nunca rutas
  absolutas; el harness las resuelve dentro del workspace del usuario.

---

## 4. El entorno de ejecución (executor)

El executor decide **dónde corre bash**. Nada más.

| Executor | Dónde corre | Aislado | Bash |
|---|---|---|---|
| `LocalExecutor()` (por defecto) | en la máquina del agente | no | **apagado** |
| `LocalExecutor(allow_commands=True)` | en la máquina del agente | no | encendido; **cada comando pide aprobación** |
| `DockerExecutor()` | contenedor Docker, uid 1000, sin root, read-only, sin red por defecto | sí | encendido |
| `ModalExecutor()` | sandbox de Modal, uid 1000, red solo a pip/npm por defecto | sí | encendido |

El contrato de un executor es mínimo:

```python
async def execute(session, script: str, *, timeout: float, cancellation_token=None) -> ExecutionResult
# más el ciclo de vida: connect / sync / disconnect / clean / rebuild
```

`execute` recibe **un string** y lo corre con `bash -c`. El executor no sabe nada de
workspaces, cwd, output ni permisos: eso lo hace el harness antes y después.

**Dentro del sandbox no hay nada de max_ai.** Solo bash y herramientas GNU estándar.
Cualquier imagen Debian/Ubuntu sirve. La imagen por defecto (587 MB) trae Python 3.12,
Node 22, uv, git y ripgrep; el agente instala lo demás con `pip`, `npm` o `uv`.

---

## 5. Cómo se conectan workspace, skills y entorno

```
          HOST (proceso del agente)                     SANDBOX (Docker / Modal)
 ┌────────────────────────────────────┐          ┌──────────────────────────────────┐
 │ .agents/<usuario>/                 │          │ /workspaces/<usuario>/           │
 │   workspace/  ← file tools escriben│ ◄──────► │   workspace/  ← bash escribe      │
 │   skills/     ← el registry copia  │   sync   │   skills/                        │
 └────────────────────────────────────┘ (bash+tar)└──────────────────────────────────┘
          ▲                                        variables que ve el comando:
          │ download / upload                        WORKSPACE=/workspaces/<u>/workspace
   MinIO / Azure (si aplica)                         SKILLS=/workspaces/<u>/skills
```

**En cada turno:**

1. El workspace baja los archivos del usuario (si es remoto) y crea las carpetas.
2. El skill registry copia las skills seleccionadas a `skills/`.
3. El prompt lista las skills disponibles (nombre y descripción, no el contenido).

**Cuando el modelo usa una skill:**

1. Lee `skills/<nombre>/SKILL.md` con `ReadFile` (en el host).
2. Si la skill trae scripts, los corre con bash: `python "$SKILLS/<nombre>/scripts/run.py"`.
3. Los comandos arrancan en `$WORKSPACE`; los entregables quedan ahí.

**Alrededor de cada bash**, el `EnvironmentManager` sincroniza las dos copias:

- **Antes** (host → sandbox): lo que las file tools cambiaron desde el último sync,
  y `skills/` queda idéntica a la del host.
- **Después** (sandbox → host): lo que el comando creó, cambió o borró **en
  `workspace/`**. Nada más vuelve.

**Qué se sincroniza y qué no:**

| Carpeta | Dirección | Por qué |
|---|---|---|
| `workspace/` | en las dos direcciones | son los archivos del usuario |
| `skills/` | solo host → sandbox | si el modelo edita una skill (o una prompt injection lo hace), el cambio se pisa antes del siguiente comando y nunca llega al host |
| archivos `.maxai-*` (manifests del harness) | nunca | son estado interno: el manifest de MinIO decide qué se borra al subir |
| el resto de la carpeta del usuario | nunca | un archivo que el modelo deja fuera de `workspace/` se queda en el sandbox |

### El sync, con bash + tar

1. **¿Qué cambió?** Cada lado lista sus archivos con un hash:
   `find . -type f ! -name '.maxai-*' -print0 | xargs -0 sha256sum -z`. Son 64 caracteres por archivo,
   pese lo que pese.
2. **Comparar** contra el `baseline` (los hashes del último sync):
   - lo cambió solo el lado que manda → se copia (o se borra)
   - lo cambiaron los dos lados → **conflicto**: el sync se detiene, no pisa nada
3. **Mover solo lo que cambió** en un `tar`, en base64, por stdin/stdout:
   - host → sandbox: `base64 -d | tar -x`, y `rm` de lo borrado
   - sandbox → host: `tar -c | base64`; el host lo extrae con el filtro seguro de
     Python (`filter="data"`): solo archivos normales, sin `..`, sin symlinks
4. El baseline avanza solo para los archivos que quedaron iguales en los dos lados.

Tope: 100 MB por sync. El sync anterior mandaba todo el contenido en un JSON en cada
comando y tenía un tope de 8 MB.

---

## 6. La regla: workspace sí, bash solo si hay dónde correrlo

> **Puede haber workspace sin bash. No puede haber bash sin workspace.**

- El workspace **siempre existe**: si no se pasa uno, el agente usa
  `LocalWorkspace(".agents")`. Las file tools lo necesitan para funcionar.
- Bash **solo existe si el executor corre comandos** (`executor.runs_commands`):
  - `LocalExecutor()` por defecto no corre comandos → el agente no tiene bash.
  - Pasar un `BashTool` con comandos apagados es un error al construir el agente.
  - Si hay skills y no hay comandos, se avisa por log y en el CLI.
- Bash **no puede existir sin workspace** porque todo lo que hace depende de él:
  arranca en `$WORKSPACE`, el cwd solo puede quedar en el workspace, las skills o
  `/tmp`, el sync mueve el workspace, y la detección de archivos lo recorre.

**Sin bash, el prompt no menciona bash.** No se le dice al modelo "no tienes bash"
(eso lo hace pedirlo); simplemente las secciones de bash, scripts y `/tmp` no
aparecen. Las skills se usan con las file tools.

| Configuración | Workspace | Bash | Skills |
|---|---|---|---|
| `Agent()` | `.agents/` local | no | solo lectura con file tools |
| `Agent(executor=LocalExecutor(allow_commands=True))` | local | sí, cada comando se aprueba | completas |
| `Agent(executor=DockerExecutor())` | local o remoto | sí, aislado | completas |
| `Agent(executor=ModalExecutor(), workspace=MinIOWorkspace(...))` | MinIO | sí, aislado | completas |

---

## 7. El script que envuelve cada comando

El harness nunca manda el comando solo. Manda un script: unas líneas con los valores
(cada uno con `shlex.quote`) y después `wrapper.sh`, siempre igual:

```bash
WORKSPACE='/workspaces/ana/workspace'
SKILLS='/workspaces/ana/skills'
__maxai_cwd='/workspaces/ana/workspace/informes'
__maxai_log='/tmp/maxai-bash/c1/8f2a.log'
__maxai_max_bytes='30000'
__maxai_command='python make.py'
# ---- wrapper.sh ----
readonly __maxai_cwd __maxai_log __maxai_max_bytes __maxai_command   # el comando no las puede pisar
export WORKSPACE SKILLS
cd "$__maxai_cwd" || cd "$WORKSPACE"
trap __maxai_finish EXIT          # corre siempre: con exit, con error, con timeout
trap 'exit 143' TERM
__maxai_snapshot > before         # foto del workspace: ruta, tamaño, fecha
{ eval "$__maxai_command"
} < /dev/null > "$__maxai_log" 2>&1   # stdin cerrado; stdout y stderr juntos
```

`__maxai_finish` (el trap) imprime, en este orden:

1. **El output**: si pasa de 30 KB, los primeros 15 KB, un aviso
   `[... N bytes cut; full output in <log> ...]` y los últimos 15 KB.
2. **Los archivos cambiados**: otra foto y `comm -3 before after`, tras la marca
   `__MAXAI_FILES__=`.
3. **El cwd final**: `pwd -P`, tras la marca `__MAXAI_CWD__=`.
4. Sale con el exit code del comando.

La tool quita las dos marcas del output, así que el modelo no las ve.

**Por qué así:**

- `eval` en el mismo shell: un `cd` dentro del comando llega hasta el trap, y así el
  cwd persiste entre llamadas. Las variables de entorno no persisten (cada llamada es
  un shell nuevo).
- Los valores van en variables y no pegados dentro del código: no hay problemas de
  comillas, y el `wrapper.sh` se puede leer y probar como un script normal.
- El trap en `EXIT` garantiza que el modelo recibe el output parcial y el exit code
  aunque el comando haga `exit 3` o lo corte el timeout.
- `< /dev/null`: un comando que espera input (`cat`, `read`) termina en vez de
  colgarse hasta el timeout.

---

## 8. Lo que devuelve la tool

```json
{
  "exit_code": 0,
  "output": "Report written",
  "files": {"created": ["reporte.xlsx"], "modified": ["data/raw.csv"]},
  "note": "Stopped after 360 seconds, the time limit for a command."
}
```

- `exit_code`: el del comando. Un comando que falla es un resultado normal, no un
  error de la tool. Un timeout da 124.
- `output`: stdout y stderr juntos, en orden.
- `files`: aparece solo si hubo cambios. Rutas desde el workspace, `.git` ignorado,
  máximo 50 por grupo. El modelo ve lo que creó sin tener que declararlo.
- `note`: solo cuando hace falta (timeout, o el comando salió de los archivos del
  usuario y el próximo arranca en `$WORKSPACE`).

---

## 9. Timeouts y cancelación

- **Un solo límite**: `TOOL_TIMEOUT_SECONDS` (360 s), global. El modelo no lo elige.
- **Dentro del sandbox**, el comando corre bajo GNU `timeout -k 5`, que mata **todo
  el grupo de procesos**: un `python` que lanzó hijos no deja huérfanos.
- **Cancelación** (el usuario corta el turno): el harness manda `TERM` al grupo por un
  segundo `exec`, usando el PID que el wrapper guardó en `/tmp`.
- **Red de seguridad**: si ni `timeout` logra parar el comando, el host espera
  `timeout + 30` segundos y destruye el contenedor.

---

## 10. Permisos y seguridad

**Orden de decisión:** la política se evalúa primero.

1. `deny` → el comando nunca llega al humano; es un error de validación.
2. `ask` → el turno se pausa y el usuario aprueba o rechaza (human in the loop).
3. `allow` → corre.

- `BashPermissions` clasifica por patrones de palabras (`git status`, `git push:*`),
  revisando cada parte de `&&`, `||`, `;` y pipes. Comillas, expansiones, redirects y
  subshells siempre piden aprobación.
- **Si el executor no está aislado** (local), `allow` se convierte en `ask`: en tu
  propia máquina, apruebas cada comando.
- **Local sin secretos**: el comando corre con `env -i` y solo recibe `PATH`, `HOME`,
  `LANG` y `WORKSPACE`. Nunca ve las API keys del proceso del agente.
- **Sandbox**: usuario sin root (uid 1000), `--read-only`, `--cap-drop=ALL`,
  `no-new-privileges`, límite de procesos, memoria y CPU; red apagada o limitada a
  los registros de paquetes.
- **El sync no confía en el sandbox**: lo que baja se extrae con el filtro seguro de
  `tarfile` y cada ruta se verifica dentro del workspace. Solo vuelve `workspace/`:
  ni las skills, ni los `.maxai-*`, ni nada que el modelo deje con `cd ..`.
- **El modelo puede moverse dentro del sandbox** (`cd ..`, `/tmp`): el muro es el
  sandbox, no la carpeta. Pero si un comando termina fuera del workspace, las skills
  o `/tmp`, el siguiente vuelve a arrancar en `$WORKSPACE` (como Claude Code) y el
  resultado lo avisa en `note`.

---

## 11. Lo que el harness revisa al cerrar el turno

El completion gate lee los resultados de las tools, nunca el disco:

- **Comando fallido sin resolver** (exit ≠ 0 y el mismo comando no volvió a
  funcionar): un aviso al modelo — "arréglalo o dile al usuario que falló". Si cierra
  igual, se acepta con una nota.
- **Archivos temporales en el workspace**: si el turno dejó archivos dentro de
  `tmp/`, `temp/`, `scratch/` o `__pycache__/` (según los `files` de cada bash, menos
  lo que se borró después), un aviso — "bórralos o muévelos junto a los entregables".
  El harness nunca los borra por su cuenta.

---

## 12. Decisiones que cambiamos (y por qué)

| Antes | Problema | Ahora |
|---|---|---|
| El modelo tenía un `$SCRATCHPAD` para temporales | Los modelos (y las skills de terceros) lo ignoraban y creaban `tmp/` y `output/` en el workspace | Temporales en `/tmp`, que es lo que los modelos ya hacen; el gate avisa si queda algo en `tmp/` |
| El modelo declaraba `expected_outputs` | Se olvidaba o declaraba mal | El harness detecta los archivos cambiados con dos fotos del workspace |
| max_ai instalado dentro del sandbox (worker, supervisor, sync en Python) | Host y sandbox corrían versiones distintas; había que hacer merge a `main` para que el sandbox viera los cambios; imagen de 1.8 GB | El sandbox solo ejecuta un string; sync con bash + tar; imagen de 587 MB |
| Timeout por tool, y el modelo podía pasar uno | Inconsistente | Un solo `TOOL_TIMEOUT_SECONDS` global |
| Sync con el contenido completo en JSON en cada comando | Lento, tope de 8 MB | Hashes para comparar, tar solo con lo cambiado |
| Tools Python del desarrollador corriendo dentro del sandbox | Necesitaba max_ai adentro | Corren en el host; en producción se usa MCP |

---

## 13. Checklist para reimplementarlo

1. **Contrato del executor**: `execute(session, script: str, timeout) -> (stdout, stderr, exit_code, timed_out)`. Nada más entra al sandbox.
2. **La tool** recibe solo `command` y `description`; arma el script con los valores entre comillas seguras + un wrapper fijo.
3. **Wrapper**: `cd` al cwd guardado, `trap` en EXIT que imprime output cortado + archivos cambiados + cwd con marcas, `eval` del comando con stdin cerrado y stdout/stderr a un log.
4. **cwd por conversación** guardado en el harness; solo se acepta si queda en el workspace, las skills o `/tmp` (nunca en la carpeta del usuario, que guarda archivos del harness).
5. **Detección de archivos**: foto `find -printf '%P\t%s\t%T@'` antes y después, `comm -3`.
6. **Timeout global** con `timeout -k` en el sandbox + red de seguridad en el host; cancelación matando el grupo de procesos.
7. **Workspace siempre**; bash solo si el executor corre comandos; sin bash, el prompt no lo menciona.
8. **Sync** por hashes + tar, con baseline y detección de conflictos; extracción segura en el host.
9. **Política** deny → ask → allow evaluada antes del human in the loop; en local, allow se vuelve ask.
10. **Gate** que lee resultados: fallos sin resolver y temporales dejados, un aviso cada uno.
