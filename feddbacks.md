Hallazgos

Hay un bug real en el ciclo de errores de herramientas: cuando una tool falla antes de ejecutarse de verdad, el framework emite ToolMessage/ToolCallResponseEvent, pero no deja el ToolCallRecord en un estado terminal. Eso pasa en “tool no encontrada”, rechazo, validación inválida y fallo del runner paralelo en tool_executor.py (line 182), tool_executor.py (line 206), tool_executor.py (line 213), tool_executor.py (line 425) y queda explícito en [_yield_failure](/max_ai/max_ai/base/tool_executor.py:449). Como ToolCallRecord.is_actionablesigue siendoTrueparaAUTO_APPROVED/APPROVEDhasta pasar pormark_consumed()en [tool_call.py](/max_ai/max_ai/types/tool_call.py:213), puedes terminar con tools “falladas” que siguen pareciendo reintentables o incluso conneeds_approval` incoherente.

El manejo de session_id está desalineado entre modelos. RunContext expone session_id en types/run_context.py (line 21), pero el runtime usa ctx.runtime_state.session_id en tool_executor.py (line 370), react.py (line 145) y base/reasoning.py (line 260). Resultado: si el usuario llena RunContext.session_id, muchas partes importantes del framework no lo ven. Eso te puede romper trazabilidad, tools context-aware y rehidratación entre corridas.

Hay una segunda implementación paralela de contexto que parece vieja y rota: context/run_context.py (line 18) y context/messages.py (line 14). En esa versión se llama CoreMessage.parse_message(...) en context/messages.py (line 41), pero el API real es parse_msg(...) en core/messages.py (line 192). Eso hoy es una trampa de importación: alguien puede usar el módulo “equivocado” y encontrarse un fallo difícil de diagnosticar.

La propia base de razonamiento reconoce que una instancia compartida de reasoning no es segura para agent.run() concurrentes, porque bind() reescribe estado mutable por ejecución en base/reasoning.py (line 190). No te explota en uso simple, pero sí es un riesgo de arquitectura si luego quieres multi-tenant real o paralelismo por sesión.

La documentación principal está bastante desfasada con respecto al código actual. El README todavía marca como “faltante” Agent.run, tool execution, middleware y streaming en README.md (line 157), pero esas piezas sí existen. Eso no rompe runtime, pero sí baja mucho la confianza del framework para cualquiera que entra nuevo.

Lo que está bueno
Tu framework está bastante bien pensado. Lo mejor que vi es la separación de capas: Agent, CapabilityRegistry, PromptStack, ToolExecutor, Reasoning y Client tienen responsabilidades bastante limpias. Eso no es tan común; muchos frameworks de agentes se vuelven una sola clase gigante muy rápido.

También me gustó que diseñaste con lifecycle claro: construir, prepare(), correr. La validación temprana en CapabilityRegistry y PromptStack es una muy buena decisión porque empuja errores al inicio y no al medio de una conversación. El loop ReAct y el sistema de middleware también muestran una intención madura: observabilidad, aprobación, streaming y recuperación no están “pegados” a mano sino modelados como primitivas.

Los tests además cubren bastante más que un prototipo típico. Hay unitarios, integración y smoke tests, y eso le da solidez.

Qué mejoraría
Priorizaría esto en orden:

Unificar una sola fuente de verdad para contexto/runtime.
Arreglar los caminos de error de tools para que todo ToolCallRecord termine siempre en un estado consistente.
Borrar o aislar los módulos/copias viejas que hoy meten ruido, por ejemplo archivos como tool_call copy.py (line 1) y test_react_loop copy.py (line 1).
Actualizar README y naming para que lo conceptual coincida con lo ejecutable.
Si quieres que el framework crezca, hacer reasoning stateless por run o clonar la instancia al ejecutar.
Mi opinión general: la idea es buena y la base técnica también. No lo siento como “toy framework”; sí lo siento como un framework prometedor que ya necesita una pasada fuerte de consolidación para evitar que las duplicaciones y pequeños desajustes de estado se conviertan en deuda seria.

No hice cambios ni corrí la suite; esta opinión sale de lectura de arquitectura e implementación. Si quieres, en el siguiente paso te puedo hacer una segunda pasada más accionable: “plan de refactor por etapas” o incluso corregirte los 2 bugs más importantes directamente.