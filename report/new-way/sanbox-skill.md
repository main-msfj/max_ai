# Sandbox y skills: cómo se usan y cómo se relacionan

Estado al 2026-11-15. Las dos piezas están escritas y probadas juntas, pero el `Agent` todavía no las llama: hoy se usan a mano, como en los ejemplos de abajo.

## La idea en una frase

**Las skills dicen de dónde vienen. El sandbox es donde quedan instaladas.** Ninguna conoce los detalles de la otra.

```
   TU PROCESO (harness)                              ENVIRONMENT (local, Docker…)
 ┌────────────────────────────┐                    ┌──────────────────────────────┐
 │  Skills                    │                    │  .agent/                     │
 │   LocalSkills   (carpeta)  │   key + archivos   │    setting.json   ← keys     │
 │   GitHubSkills  (repo)     │ ─────────────────► │    skills/<nombre>/          │
 │                            │                    │    workspace/<sesión>/       │
 │  Sandbox                   │   comandos         │    tmp/<sesión>/             │
 │   LocalSandbox             │ ─────────────────► │                              │
 │   DockerSandbox            │                    │                              │
 └────────────────────────────┘                    └──────────────────────────────┘
        el token de GitHub y los secretos                 aquí solo llegan los archivos
        se quedan de este lado                            y las variables que la skill pide
```

## Qué hace cada una

| | Skills | Sandbox |
|---|---|---|
| Sabe | dónde están las skills (carpeta o repo) y qué variables piden | dónde está `.agent/` y cómo correr un comando ahí |
| No sabe | nada del disco del environment | nada de GitHub ni de carpetas de tu máquina |
| Padre | `SkillsCore` en `max_ai/base/skills.py` | `SandboxCore` en `max_ai/base/sandbox.py` |
| Hijos | `LocalSkills`, `GitHubSkills` | `LocalSandbox`, `DockerSandbox` |
| Lo que escribe cada hijo | `get_or_update_key(name)` y `pack(name)` | `execute_command(...)` |

Todo lo demás está escrito una sola vez en los padres, así que cualquier skill funciona con cualquier sandbox.

## Cómo se usa

```python
from max_ai.capabilities.sandbox.docker import DockerSandbox
from max_ai.capabilities.skills import GitHubSkills

skills = GitHubSkills(
    "https://github.com/org/repo/tree/main/skills",
    skills=["pdf", "docx"],
    token="ghp_...",                          # o se toma de $GITHUB_TOKEN
    skill_env={"ANTHROPIC_API": "ANT_KEY"},   # nombre en la skill → nombre en tu entorno
)
# Solo la configuración: no crea ningún contenedor.
config = DockerSandbox(networks=["pypi.org"])

# Al empezar una corrida de un usuario:
sandbox = config.bind("user-1", skills.variables())   # 1. su contenedor y sus variables
await sandbox.prepare()                               # 2. crea .agent/ si falta
await skills.sync(sandbox)                            # 3. instala lo que cambió
prompt = await skills.blocks(sandbox)                 # 4. texto para el prompt
```

Las otras combinaciones solo cambian el constructor:

```python
LocalSkills("./skills", ["pdf", "docx"], skill_env={...})
LocalSandbox(root=".")
```

## Los cuatro pasos

### 1. `skills.variables()` → `sandbox.bind(identifier, variables)`

Una skill puede pedir una variable, por ejemplo `os.getenv("ANTHROPIC_API")`. En tu entorno ese valor puede llamarse de otra forma. `skill_env` hace la traducción:

```
skill_env = {"ANTHROPIC_API": "ANT_KEY"}

tu entorno:   ANT_KEY=sk-123
                  │  skills.variables()
                  ▼
              {"ANTHROPIC_API": "sk-123"}
                  │  sandbox.bind("user-1", …)
                  ▼
dentro:       ANTHROPIC_API=sk-123
```

- Si falta una variable, falla ahí mismo y dice cuál falta.
- `bind` devuelve una copia para ese usuario. En Docker cada usuario tiene su propio contenedor y volumen (`docker-sandbox-<identifier>`); lo único común es la configuración.
- Las variables nunca se serializan.

### 2. `sandbox.prepare()`

Crea `.agent/` con `setting.json`, `skills/`, `workspace/` y `tmp/` si no existen. Si ya existen no toca nada.

### 3. `skills.sync(sandbox)`

```
skills.keys()                      ──► {"pdf": "ab12…", "docx": "cd34…"}
sandbox.outdated_skills(keys)      ──► compara con setting.json ──► ["docx"]
skills.pack("docx")                ──► los archivos de esa skill, comprimidos
sandbox.install_skill("docx", …)   ──► reemplaza la carpeta y guarda la key
```

Devuelve los nombres de las skills que instaló. Si nada cambió, devuelve `[]` y no envía nada.

**La key** es lo que decide si una skill cambió. Cambia sola, sin que el developer tenga que acordarse de subir una versión:

| Skills | Key | Costo de comparar |
|---|---|---|
| `LocalSkills` | hash del nombre, permiso y contenido de cada archivo | leer la carpeta |
| `GitHubSkills` | identificador que git da a la carpeta de la skill | una petición para todas |

`GitHubSkills` baja las skills una por una: nunca descarga el repo entero, y baja exactamente la versión cuya key comparó.

### 4. `skills.blocks(sandbox)`

Devuelve una línea por skill, con la ruta real dentro del sandbox:

```
- pdf: Work with pdf files (/home/agent/.agent/skills/pdf/SKILL.md)
- docx: Work with docx files (/home/agent/.agent/skills/docx/SKILL.md)
```

La descripción sale de la línea `description:` del `SKILL.md`. Se guarda en `setting.json` al instalar, así que no hace falta descargar la skill para escribir el prompt. Una skill sin `description:` no se instala.

## `setting.json`

Es lo único que une a las dos piezas: las skills escriben ahí a través del sandbox, y de ahí se lee qué hay instalado.

```json
{
  "skills": {
    "pdf":  {"key": "ab12…", "description": "Work with pdf files"},
    "docx": {"key": "cd34…", "description": "Work with docx files"}
  }
}
```

## Qué pasa si algo falla a la mitad

- **Se corta la conexión mientras se envía una skill**: no se toca nada; queda la versión anterior.
- **Dos procesos instalan en el mismo sandbox a la vez**: uno espera al otro; si el segundo trae la misma key, no hace nada.
- **Nunca hay una key sin su carpeta completa**: en el peor caso la key se borra y la siguiente corrida reinstala.
- **Una skill que intenta escribir fuera de su carpeta** se rechaza entera. Los links dentro de una skill se saltan.

## Cómo se ve serializado

```json
{
  "provider": "max_ai.capabilities.skills.github.GitHubSkills",
  "component_type": "skills",
  "config": {
    "url": "https://github.com/org/repo/tree/main/skills",
    "skills": ["pdf", "docx"],
    "token_env": "GITHUB_TOKEN",
    "skill_env": {"ANTHROPIC_API": "ANT_KEY"}
  }
}
```

Solo nombres, nunca valores: el token se vuelve a leer de `$GITHUB_TOKEN` al cargar.

## Dónde está cada cosa

| Archivo | Qué tiene |
|---|---|
| `max_ai/base/layout/` | las rutas de `.agent/` y el script que corre dentro del environment |
| `max_ai/base/sandbox.py` | `SandboxCore`: `bind`, `prepare`, `upload_file`, `outdated_skills`, `installed_skills`, `install_skill` |
| `max_ai/base/skills.py` | `SkillsCore`: `keys`, `variables`, `sync`, `blocks` |
| `max_ai/capabilities/sandbox/local/`, `docker/` | `LocalSandbox`, `DockerSandbox` |
| `max_ai/capabilities/skills/local/`, `github/` | `LocalSkills`, `GitHubSkills` |
| `tests/sandbox/` | los tests de todo lo anterior; `test_skills_sync.py` prueba las cuatro combinaciones |

## Lo que falta

1. **Conectarlo al `Agent`**: que haga los cuatro pasos al empezar una corrida y ponga `blocks()` en el prompt.
2. **`base/__init__.py` y `agent.py`** todavía importan `CoreSkillRegistry`, que ya no existe: el paquete no importa solo hasta que se cambien.
3. **Las tools de `new_tool/`** usan `sandbox.workspace`; ahora es `sandbox.layout.workspace`.
4. **Producción**: de dónde salen las variables de cada usuario (Key Vault) sin ponerlas todas en el entorno del proceso.
5. **Ocultar los valores** de las variables en la salida de las tools, con los hooks.

## Límites

- GitHub sin token permite 60 peticiones por hora por IP.
- GitHub lista hasta 1.000 entradas por carpeta.
- Las skills necesitan `python3` dentro del environment para instalarse.
