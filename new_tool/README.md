# new_tool — read_file, write_file, edit_file y bash sobre el sandbox

Las cuatro tools probadas contra `LocalSandbox`. No están conectadas al `Agent`: es código para revisar y mover cuando quieras.

```
new_tool/
    _base.py      FileTool: lo común de las tres tools de archivos (corre en el harness)
    _common.py    lo común de los tres scripts (corre en el environment)
    read/         _base.py (la tool)   script.py (lo que corre dentro)
    write/        _base.py             script.py
    edit/         _base.py             script.py
    bash/         _base.py             (usa el wrapper.sh que ya existe)
```

## La idea en un dibujo

```
            HARNESS (tu proceso)                         ENVIRONMENT (local, Docker, Modal…)
 ┌──────────────────────────────────────┐             ┌──────────────────────────────────┐
 │ modelo: edit_file("src/a.py", …)     │             │                                  │
 │            │                         │             │                                  │
 │            ▼                         │             │                                  │
 │ FileTool.execute                     │             │                                  │
 │  1. locate(): ¿a dónde apunta?       │             │                                  │
 │     sesión / otra sesión / fuera     │             │                                  │
 │  2. busca el stamp que guardó        │             │                                  │
 │     cuando el modelo leyó            │   stdin     │  python3 -c  (_common + script)  │
 │  3. manda script + JSON  ────────────┼────────────►│   a. realpath: ¿sigue dentro?    │
 │                                      │             │   b. ¿el archivo es el que leyó? │
 │                                      │             │   c. busca old_string            │
 │                                      │   stdout    │   d. guarda en un solo paso      │
 │  4. guarda el stamp nuevo  ◄─────────┼─────────────│   e. devuelve JSON + stamp nuevo │
 │  5. responde al modelo (sin stamp)   │             │                                  │
 └──────────────────────────────────────┘             └──────────────────────────────────┘
```

El harness **decide** (dónde apunta la ruta, si puede seguir, qué stamp corresponde). El script **hace** (leer, comparar, guardar), dentro del environment, en una sola llamada.

## Cómo sabe edit que el archivo cambió

Esta es la parte difícil, y se resuelve con una sola cosa: el **stamp**.

**El stamp es el sha256 del contenido completo del archivo.** Mismos bytes, mismo stamp. No depende de la fecha ni del tamaño.

```
  read_file(a.py)     el script calcula el stamp ──► el harness lo guarda:  stamps[(sesión, ruta)] = "ab12…"
                                                     (el modelo nunca lo ve)

  edit_file(a.py)     el harness manda "ab12…" ───► el script vuelve a calcular el stamp del archivo
                                                        │
                              ┌─────────────────────────┴─────────────────────────┐
                        igual a "ab12…"                                    distinto
                              │                                                   │
                     edita y guarda                         "it changed since you read it;
                     devuelve el stamp NUEVO                 read it again before changing it"
                              │
              el harness reemplaza el stamp guardado
              → el siguiente edit funciona sin releer
```

Los cuatro casos:

| Situación | Stamp guardado | Qué pasa |
|---|---|---|
| El modelo nunca leyó el archivo | no hay | `read it before changing it` |
| Lo leyó y nadie lo tocó | coincide | se edita |
| Lo leyó y **algo lo cambió** (bash, otro proceso, el usuario) | no coincide | `it changed since you read it` |
| Lo acaba de escribir o editar él mismo | es el que devolvió esa escritura | se edita, sin releer |

Por qué funciona sin vigilar nada: no hace falta saber **quién** cambió el archivo ni **cuándo**. Si bash le agrega una línea, el contenido es otro, el stamp es otro, y el edit se detiene solo. Por eso bash no tiene que avisarle nada a las tools de archivos.

Detalles que importan:

- **El stamp cubre el archivo entero aunque el modelo lea solo unas líneas.** Leer las líneas 1–50 alcanza para editar la 900. `read/script.py` recorre todo el archivo para el hash pero solo devuelve las líneas pedidas.
- **La comparación y el guardado van en el mismo script.** Si el harness comparara primero y mandara a guardar después, serían dos llamadas con un hueco en medio.
- **`write_file` usa la misma regla** para un archivo que ya existe. Uno nuevo no necesita nada.
- **Los stamps viven en un diccionario del harness**, `deps["file_stamps"]`, con clave `(session_id, ruta absoluta)`. Si el proceso se reinicia se pierden, y lo único que pasa es que el modelo recibe "read it before changing it" y relee.

## Cómo busca `old_string`

De estricto a tolerante; se queda con la primera forma que encuentre algo:

1. **Exacto.**
2. **Tipográfico**: comillas curvas, guiones largos y espacios duros se leen como los normales. Los modelos los cambian sin querer.
3. **Espacios al final de línea**: compara líneas completas ignorando lo que sobra al final.

Si la coincidencia no fue exacta, la respuesta lo dice (`matched ignoring … differences`) para que el modelo sepa que su copia no era idéntica.

Después:

- **ninguna coincidencia** → error. Si `new_string` ya está en el archivo, el error dice que el edit quizá ya estaba aplicado (pasa cuando el modelo repite una llamada).
- **varias y sin `replace_all`** → error con el número de veces.
- **una, o `replace_all`** → reemplaza.

Dos errores no necesitan el archivo y los corta el harness antes de mandar nada (`EditFile.validate_parameters`): `old_string` vacío, y `old_string` igual a `new_string`.

También: un archivo con finales de línea de Windows (`\r\n`) los conserva, y un archivo existente conserva sus permisos (un `.sh` ejecutable sigue siéndolo).

## Las rutas: sin prefijos

El modelo escribe rutas reales, como acordamos:

| El modelo escribe | Apunta a | Resultado |
|---|---|---|
| `src/a.py` | `workspace/<sesión>/src/a.py` | permitido |
| ruta absoluta dentro de su sesión o de `tmp/<sesión>` | ahí mismo | permitido |
| ruta dentro de `workspace/` pero de otra sesión | ahí mismo | **leer sí, cambiar no** |
| cualquier otra | — | error que le dice cuál es su carpeta |

Hay dos comprobaciones, y hacen falta las dos:

- **`locate()` en el harness** mira el texto de la ruta ya normalizada (`..` resuelto) y decide la zona.
- **`resolve()` en el script** sigue los symlinks con `realpath` dentro del environment. Un symlink dentro del workspace que apunta a `/etc/passwd` solo se ve ahí.

El `session_id` se valida con la misma función del session store: `..` como id ya no sirve para salir.

## bash

`new_tool/bash` hereda del `BashTool` actual: parámetros, reglas de Policy y formato de salida son los mismos, y el script sigue siendo `wrapper.sh`. Cambia dónde corre:

- usa `sandbox.execute_command` en vez del executor y su sesión;
- arranca en la carpeta de la sesión;
- el directorio actual se guarda en `deps["bash_cwd"]` (un diccionario del harness), no dentro del objeto de la tool;
- exporta `TMPDIR` apuntando a `tmp/<sesión>`;
- el shell puede moverse por todo `workspace/` (otras sesiones incluidas) y por su tmp; si termina en otro lugar, el siguiente comando vuelve a la sesión.

## Qué se probó

Corrí cada tool contra `LocalSandbox` en una carpeta temporal:

- crear, editar justo después de escribir, editar dos veces seguidas sin releer;
- `old_string` repetido, `replace_all`, no encontrado, ya aplicado, vacío, igual a `new_string`;
- bash cambia el archivo → edit y write se detienen; tras releer, edit funciona;
- editar y sobreescribir un archivo nunca leído → se detienen;
- coincidencia por espacios finales y tipográfica; `\r\n` conservado; permisos `755` conservados;
- leer otra sesión (permitido), editarla (rechazado), `/etc/passwd`, `../../setting.json`, tmp de otra sesión, `session_id=".."`, symlink hacia fuera (todos rechazados);
- archivo inexistente, carpeta, archivo binario: error corto, sin la ruta real de la máquina;
- bash: cwd que persiste, `TMPDIR`, salir del workspace, lista de archivos cambiados.

No hay tests guardados en el repo: eran pruebas a mano.

## Lo que falta para usarlas

1. **Conectarlas.** El `Agent` tiene que poner en `ToolContext.deps` tres cosas: `sandbox`, `file_stamps` (un `dict`) y `bash_cwd` (un `dict`). Los dos diccionarios deben ser el mismo objeto en todas las llamadas; lo natural es que acaben en `RunContext` para que se guarden con la sesión.
2. **Pedir permiso para otra sesión.** Hoy cambiar un archivo de otra sesión se rechaza, porque la tool sola no puede preguntarle al usuario: eso lo hace el dispatcher antes de ejecutar. Cuando el dispatcher pueda llamar a `locate()`, ese rechazo pasa a ser "preguntar". Es una línea en `FileTool.execute`.
3. **La Policy con rutas absolutas.** Una regla como `write_file(secrets/**)` se compara con lo que escribió el modelo. Con `secrets/key` funciona; con la ruta absoluta del mismo archivo, no. Para arreglarlo, `permission_subjects` necesita conocer la sesión, y hoy solo recibe los parámetros.
4. **bash y el dispatcher viejo.** `runs_commands = True` hace que el dispatcher exija el `EnvironmentManager` y abra una sesión del executor viejo antes de llamar a la tool. Funciona, pero arrastra la maquinaria que quieres quitar.
5. **`python3` dentro del environment.** Las tres tools de archivos lo necesitan.
6. **Moverlas.** Los imports son `from max_ai…` absolutos porque la carpeta está en la raíz del repo; dentro de `max_ai/capabilities/tools/` pasan a relativos.

## Pendiente: leer más que texto

Anotado el 2026-11-14, sin empezar. Hoy `read_file` rechaza todo lo que no sea UTF-8.

Hay dos familias:

| Tipo | Cómo lo recibe el modelo | Qué hace falta |
|---|---|---|
| Notebook (`.ipynb`) | texto | nada nuevo: es un JSON, se lee con la librería estándar |
| PDF con texto | texto | `pypdf` o `pdftotext` en el environment, y un parámetro `pages` |
| Word, PowerPoint, Excel | texto | por skills, no dentro de `read_file` |
| Imagen (`.png`, `.jpg`) | imagen | cambios en el harness (abajo) |
| PDF escaneado | imagen por página | las dos cosas |

**Los que se convierten a texto** caben en el diseño actual: `read/script.py` mira la extensión, convierte dentro del environment y devuelve texto. Si falta una librería, el error dice cuál.

**Las imágenes** piden tocar el harness, porque hoy el resultado de una tool siempre termina como texto:

1. El resultado de una tool tiene que poder llevar una imagen. Hoy `tool_message` en el dispatcher hace `json.dumps` de todo.
2. Cada cliente decide dónde va: en la API de chat de OpenAI un mensaje de tool solo acepta texto, así que la imagen va en un mensaje de usuario justo después.
3. Comprobar `supports_vision` del modelo; si no ve imágenes, responder que no puede mostrarla.
4. Límite de tamaño: reducir la imagen antes de enviarla (`Pillow` en el environment).
5. La compactación debe quitar primero las imágenes viejas.

Ya existen `ImagePart` y `FilePart` en `core/messages.py`, `supports_vision` en la configuración del modelo, y el cliente de OpenAI ya convierte un `ImagePart`. El script solo tendría que devolver los bytes en base64 y el tipo.

Orden acordado: notebook → imágenes → PDF → Word y PowerPoint como skills.

## Límites que conviene conocer

- Entre comparar el stamp y reemplazar el archivo pasan microsegundos dentro del mismo script. Si otro proceso escribe justo ahí, gana el edit. Es el mismo hueco que tienen los demás harness.
- `write_file` manda el contenido por stdin, así que el tamaño no choca con el límite de la línea de comando.
- El script completo viaja en cada llamada (3,5 KB el de read, 6 KB el de edit). Para local es un proceso de Python por llamada.
- `read_file` no corta líneas muy largas: un archivo minificado de una sola línea vuelve entero y lo recorta el dispatcher.
