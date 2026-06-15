# ReAct Loop — Antes vs. Ahora

Documento visual del loop de razonamiento en `max_ai/reasoning/react_planning.py`
(`ReActLoopPlanning`). Compara el estado **antes** de la sesión P0 con el estado
**actual**, y deja anotado lo que viene en P1.A (self-eval).

---

## 1. ANTES (commit base `a11e4fb`)

```
                          ┌─────────────────────────────────────┐
                          │  ReActLoop  (nombre viejo)           │
                          │  max_loop_iterations = 3   ⚠️        │
                          └─────────────────────────────────────┘

   run()
     │
     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  EVAL WRAPPER   (solo si enable_self_eval=True)                │
   │  while eval_attempt <= max_eval_retries:                       │
   │     iteration = 0   ← reinicia                                 │
   │                                                                │
   │   ┌────────────────── ReAct loop ────────────────────────┐    │
   │   │  while iteration < 3:        ⚠️ tope MUY bajo         │    │
   │   │                                                       │    │
   │   │    drenar approvals pendientes (resume)               │    │
   │   │            │                                          │    │
   │   │            ▼                                          │    │
   │   │    ¿compactar contexto? ──sí──▶ compacta             │    │
   │   │            │                                          │    │
   │   │            ▼                                          │    │
   │   │    iteration += 1                                     │    │
   │   │            ▼                                          │    │
   │   │    LLM call ──▶ ¿pausa por approval? ──sí──▶ return   │    │
   │   │            │                                          │    │
   │   │            ▼                                          │    │
   │   │    ¿hay tool_calls?                                   │    │
   │   │       no ──▶ finish ("stop")  ─── break               │    │
   │   │       sí ──▶ ejecuta tools                            │    │
   │   │                  │                                    │    │
   │   │                  ▼                                    │    │
   │   │            append ToolMessages ──▶ loop back          │    │
   │   │                                                       │    │
   │   │  else: finish_reason = "max_iterations_exceeded" ⚠️  │    │
   │   └───────────────────────────────────────────────────────┘    │
   │                                                                │
   │   ¿enable_self_eval?  no ──▶ break                             │
   │   ¿finish_reason == "stop"? no ──▶ break                       │
   │            │ sí                                                 │
   │            ▼                                                    │
   │   EVAL STEP                                                     │
   │     - busca el ÚLTIMO AssistantMessage                          │
   │     - lo evalúa contra criterios GENÉRICOS hardcodeados ⚠️     │
   │     - score = checks_passed / total                            │
   │            │                                                    │
   │     passed? ──sí──▶ break                                       │
   │            │ no                                                 │
   │            ▼                                                    │
   │     eval_attempt += 1                                           │
   │     inyecta "mejora esto..." como UserMessage                   │
   │     └───▶ VUELVE ARRIBA y RE-CORRE EL LOOP ENTERO   💸💸💸     │
   │            (sin medir cuánto cuesta)                            │
   └──────────────────────────────────────────────────────────────┘

   Problemas marcados con ⚠️ / 💸:
   ⚠️  tope de 3 iteraciones → el agente se rinde a media tarea
   ⚠️  eval solo mira la respuesta FINAL, nunca pasos intermedios
   ⚠️  criterios genéricos, iguales para toda tarea
   💸  cada retry re-corre TODO el loop, sin presupuesto → costo invisible
```

---

## 2. AHORA (rama `p0-embeddings-cleanup`, commit `4c57611`)

```
                          ┌─────────────────────────────────────┐
                          │  ReActLoopPlanning  (renombrado)     │
                          │  max_loop_iterations = 10   ✅       │
                          │  planning / self-eval = OFF default  │
                          └─────────────────────────────────────┘

   run()
     │
     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  EVAL WRAPPER   (solo si enable_self_eval=True — opt-in)       │
   │  while eval_attempt <= max_eval_retries:                       │
   │     iteration = 0                                              │
   │                                                                │
   │   ┌────────────────── ReAct loop ────────────────────────┐    │
   │   │  while iteration < 10:       ✅ margen real           │    │
   │   │                                                       │    │
   │   │    (mismo cuerpo que antes: approvals, compaction,    │    │
   │   │     LLM call, tools, loop back)                       │    │
   │   │                                                       │    │
   │   │  else: finish_reason = "max_iterations_exceeded"      │    │
   │   └───────────────────────────────────────────────────────┘    │
   │                                                                │
   │   EVAL STEP   (igual que antes — TODAVÍA sin tocar)            │
   │     - sigue mirando solo la respuesta final     ⚠️ (pendiente)│
   │     - sigue con criterios genéricos             ⚠️ (pendiente)│
   │     - sigue re-corriendo todo el loop al fallar 💸 (pendiente)│
   └──────────────────────────────────────────────────────────────┘

   Lo que cambió en P0:
   ✅  max_loop_iterations 3 → 10  (ya no se rinde a media tarea)
   ✅  rename a ReActLoopPlanning   (distingue del ReActLoop simple)
   ✅  planning/self-eval opt-in    (el agente default = ReAct puro)

   Lo que NO cambió todavía (es P1.A):
   ⚠️💸  el EVAL STEP sigue igual — eso es lo próximo
```

---

## 3. LO QUE VIENE — P1.A (self-eval), aún no implementado

```
   EVAL WRAPPER  (versión objetivo)
   ┌──────────────────────────────────────────────────────────────┐
   │  baseline_tokens = total_tokens   ← foto antes de los retries  │
   │  while eval_attempt <= max_eval_retries:                       │
   │     ... corre loop ...                                         │
   │     passed? ──sí──▶ break                                      │
   │                                                                │
   │     ┌─── PASO 1: CAP DE COSTO ────────────────────────────┐    │
   │     │ spent = total_tokens - baseline_tokens              │    │
   │     │ if spent >= eval_max_extra_tokens:                  │    │
   │     │     yield EvalEvent(phase="skipped")  ← VISIBLE      │    │
   │     │     break    ← deja de gastar                        │    │
   │     └─────────────────────────────────────────────────────┘    │
   │                                                                │
   │     eval_attempt += 1 ; inyecta feedback ; reintenta          │
   └──────────────────────────────────────────────────────────────┘

   PASO 2: criterios por-run  (pasar eval_criteria por run(), no solo __init__)
   PASO 3: eval intermedio    (revisar tool results a mitad, no solo el final)
```
```
```
