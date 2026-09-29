1. GlobTool — implementación
No usa fs.glob ni fast-glob. Delega a ripgrep:
TypeScript// utils/glob.ts
export async function glob(
  filePattern: string,
  cwd: string,
  { limit, offset }: { limit: number; offset: number },
  abortSignal: AbortSignal,
  toolPermissionContext,
): Promise<{ files: string[]; truncated: boolean }> {

  const args = [
    '--files',              // listar archivos, no buscar contenido
    '--glob', searchPattern,
    '--sort=modified',      // mtime descendente
    ...(noIgnore ? ['--no-ignore'] : []),  // por defecto NO respeta .gitignore
    ...(hidden ? ['--hidden'] : []),
  ]
  // + patrones de ignore de permisos
  // + exclusiones de cache de plugins

  const allPaths = await ripGrep(args, searchDir, abortSignal)
  const truncated = allPaths.length > offset + limit
  const files = allPaths.slice(offset, offset + limit)
  return { files, truncated }
}

call() de GlobTool:

Resuelve path (default cwd)
Llama glob(pattern, path, { limit: 100, offset: 0 }, …)
Convierte a paths relativos (ahorra tokens)
Devuelve { filenames, numFiles, truncated, durationMs }


2. GrepTool — implementación
También usa ripGrep de utils/ripgrep.ts. El call() traduce parámetros del modelo a flags de rg:
TypeScript// conceptual — GrepTool.call()
async call(input, { abortController }) {
  const args = [
    '--hidden',
    '--glob', '!.git',
    '--max-columns', '500',   // evita líneas enormes (minificados, base64)
  ]

  if (input['-i']) args.push('-i')
  if (input.glob)  args.push('--glob', input.glob)
  if (input.type)  args.push('--type', input.type)

  switch (input.output_mode ?? 'files_with_matches') {
    case 'files_with_matches':
      args.push('-l')           // solo paths
      break
    case 'count':
      args.push('-c')
      break
    case 'content':
      args.push('-n')           // con número de línea
      if (input['-C']) args.push('-C', String(input['-C']))
      if (input['-A']) args.push('-A', String(input['-A']))
      if (input['-B']) args.push('-B', String(input['-B']))
      break
  }

  if (input.multiline) args.push('-U', '--multiline-dotall')
  if (input.head_limit) args.push('--max-count', String(input.head_limit))

  args.push(input.pattern, input.path ?? '.')
  return await ripGrep(args, cwd, abortController.signal)
}
utils/ripgrep.ts elige cómo invocar rg:

Sistema (rg en PATH), si USE_BUILTIN_RIPGREP=false
Embedded en el binario Bun (process.execPath con argv0: 'rg')
Vendor bundled

Maneja reintentos EAGAIN, timeout → SIGKILL, streaming de output.


3. FileReadTool (Read)
TypeScript// flujo de call()
async call({ file_path, offset, limit, pages }, context) {
  // 1. Expandir path (~, relativo → absoluto)
  const fullPath = expandPath(file_path)

  // 2. Bloquear paths peligrosos (/proc/self/environ, /dev/stdin, …)

  // 3. Detectar tipo por extensión
  if (isImage(fullPath))   return readImage(...)      // base64 + resize por token budget
  if (isPDF(fullPath))     return readPDF(pages)      // máx 20 páginas/request
  if (isNotebook(fullPath)) return readNotebook(...)

  // 4. Texto: leer, paginar, numerar
  const content = await fs.readFile(fullPath, encoding)
  const lines = content.split('\n')
  const start = (offset ?? 1) - 1
  const slice = lines.slice(start, start + (limit ?? 2000))

  // 5. Guardar en readFileState (mtime + contenido) para Edit/Write
  context.readFileState.set(fullPath, {
    timestamp: mtime,
    content,
    isPartialView: limit != null || offset != null,
  })

  // 6. Formato cat -n
  return slice.map((line, i) =>
    `${String(start + i + 1).padStart(6)}→${line}`
  ).join('\n')
}

Detalles importantes:

Dedup: si el archivo no cambió desde el último Read, puede devolver un stub liviano
Límite de tokens (~25k) y tamaño (~256KB) → error pidiendo offset/limit
isPartialView: true no sirve para autorizar Edit (hay que leer “entero” o suficiente)


4. FileEditTool (Edit) — el más interesante
Validación antes de escribir (códigos de error del harness):
TypeScriptasync function validateEdit(input, context) {
  const fullPath = expandPath(input.file_path)
  const state = context.readFileState.get(fullPath)

  // Error 6: nunca se leyó
  if (!state || state.isPartialView) {
    return { ok: false, errorCode: 6, message: 'File has not been read yet.' }
  }

  // Error 7: cambió en disco desde el Read
  const currentMtime = (await fs.stat(fullPath)).mtimeMs
  if (currentMtime !== state.timestamp) {
    return { ok: false, errorCode: 7, message: 'File has been modified since read.' }
  }

  const fileContent = await fs.readFile(fullPath, 'utf-8')
  // normalización de quotes tipográficas, etc.

  // Error 8: no existe old_string
  if (!fileContent.includes(input.old_string)) {
    return { ok: false, errorCode: 8, message: 'String not found' }
  }

  // Error 9: no único
  const matches = fileContent.split(input.old_string).length - 1
  if (matches > 1 && !input.replace_all) {
    return { ok: false, errorCode: 9, message: `Found ${matches} matches.` }
  }

  return { ok: true, fileContent }
}
Aplicación del edit:
TypeScriptasync call(input, context) {
  const { fileContent } = await validateEdit(input, context)

  const newContent = input.replace_all
    ? fileContent.split(input.old_string).join(input.new_string)
    : fileContent.replace(input.old_string, input.new_string)

  // Escritura (a menudo atómica: temp + rename)
  await fs.writeFile(fullPath, newContent, 'utf-8')

  // Actualizar readFileState + file history + notificar LSP
  context.readFileState.set(fullPath, { timestamp: newMtime, content: newContent })
  notifyLSP(fullPath)

  return { structuredPatch, file_path, replaceAll: input.replace_all }
}
Match = substring literal, sin regex.

5. FileWriteTool (Write)
TypeScriptasync call({ file_path, content }, context) {
  const fullPath = expandPath(file_path)
  const exists = await fileExists(fullPath)

  if (exists) {
    // Mismo gate: debe haber Read previo + mtime coherente
    const state = context.readFileState.get(fullPath)
    if (!state) throw errorCode(2) // no read
    // check mtime stale…
  }

  await fs.mkdir(dirname(fullPath), { recursive: true })

  // Line endings: fuerza LF
  const normalized = content.replace(/\r\n/g, '\n')

  // Historial (before/after) para undo / audit
  const before = exists ? await fs.readFile(fullPath, 'utf-8') : null
  await fs.writeFile(fullPath, normalized, 'utf-8')
  fileHistory.record({ path: fullPath, before, after: normalized, … })

  notifyLSP(fullPath)
  context.readFileState.set(fullPath, { timestamp, content: normalized })
}

Por qué no usan Bash para esto

Permisos granulares por path (Read(~/secrets/**), Edit(/src/**))
Audit trail claro (tool_use con path + diff)
Control de tokens (límites, paths relativos, dedup de Read)
Safety (read-before-write, unicidad, stale mtime)
Performance (ripgrep embebido, no shell parsing)