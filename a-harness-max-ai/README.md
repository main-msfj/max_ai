# Evaluación del harness MaxAI

## Plan de migración

**Empezar por [06 — Plan de migración documental](06-plan-migracion.md).** Contiene fases, dependencias, contratos, pruebas de aceptación, compatibilidad y rollback; después incorpora multiagent y deja coding para una etapa posterior.

[05 — Investigación de Luna sobre Pydantic AI y Pydantic AI Harness](05-pydantic-ai-luna.md) aporta la nueva referencia arquitectónica. El plan combina esa investigación con Grok, DeepSeek, OpenCode y el diagnóstico local.

## Diagnóstico previo

Primera evaluación: 2026-09-11. Alcance: código local y contraste con `ai-harness-research/08-max-ai-harness.html`. No se ha modificado el runtime.

Leer [evaluación y arquitectura propuesta](01-evaluacion.md).

Continuar con [componentes de Grok, DeepSeek y OpenCode — investigación de Luna](02-componentes-open-source.md) y [diagnóstico de plan, workspace y ejecución en MaxAI](03-diagnostico-capabilities.md).

El diagnóstico concreto actualiza la prioridad inicial: resolver la composición de capacidades de coding y el workspace antes de añadir el CompletionGate.

[Memoria, gates de finalización y presupuestos](04-memoria-gates-presupuestos.md): responsabilidades del runtime, ejemplos de poema/email/código y control acumulado de tokens, tiempo y coste.

Este directorio reúne el diagnóstico y servirá para diseñar cambios incrementales. Los diagramas Mermaid se pueden visualizar con una extensión compatible en la vista previa Markdown de VS Code.
