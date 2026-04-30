═══════════════════════════════════════════════════════════════════════════════
                    MAXAI FRAMEWORK — CURRENT ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER CODE                                       │
│                                                                              │
│   agent = Agent(                                                         │
│     name="...", description="...", instructions="...",                       │
│     client=AnthropicClient(...),                                             │
│     memory=RedisMemoryRegistry(user_id="u1"),                                │
│     skills=LocalSkillRegistry(path="...", skills=["pr_review"]),             │
│     context=PgContextRegistry(user_id="u1", session_id="s1"),                │
│     knowledge=[QdrantKnowledge(name="docs"), ...],                           │
│     routines=[RedisRoutineRegistry(...)],                                    │
│     tools=[my_func_a, my_func_b],                                            │
│     priority_tools=["search_docs"],                                          │
│   )                                                                          │
│   await agent.prepare()                                                      │
│   await agent.run(...)                                                       │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
                       ╔══════════════════════════╗
                       ║      Agent           ║
                       ║      __init__ (sync)     ║
                       ╚══════════════════════════╝
                                     │
                ┌────────────────────┼─────────────────────┐
                ▼                    ▼                     ▼
    ┌───────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
    │ CapabilityRegistry│  │  LayerContainer     │  │ PromptVariablesBldr  │
    │  (sync validate)  │  │  (sync validate) │  │   (just stores ref)  │
    └─────────┬─────────┘  └────────┬─────────┘  └──────────────────────┘
              │                     │
              ▼                     ▼
   ┌──────────────────────┐  ┌─────────────────────────────────────────┐
   │ VALIDATES SYNC:      │  │ VALIDATES SYNC for each layer:          │
   │ • Tool name unique?  │  │ • Template parses (Jinja2 syntax)       │
   │   (across tools +    │  │ • Required vars referenced in template  │
   │    memory + knowl    │  │ • No undeclared vars in template        │
   │    + routines +      │  │ • No duplicate layer types in stack     │
   │    context tools)    │  │                                         │
   │ • priority_tools     │  │ DEFAULT LAYERS (in order):              │
   │   reference real     │  │   AgentPolicyLayer                      │
   │   tools?             │  │   TaskAnalysisLayer                     │
   │ • Tool inputs        │  │   RenderingLayer                        │
   │   coercible?         │  │   PriorityToolsLayer                    │
   │                      │  │   SkillsLayer                           │
   │ ❌ FAILS LOUD if not │  │   RoutineLayer                          │
   │                      │  │   KnowledgeLayer                        │
   │ NOT YET LOADED:      │  │   ContextLayer                          │
   │ • Skills (async)     │  │   MemoryLayer                           │
   │ • Skill tools        │  │                                         │
   │ • read_skill_resource│  │ ❌ FAILS LOUD if any layer broken      │
   └──────────────────────┘  └─────────────────────────────────────────┘

                                     │
                                     │  Construction done.
                                     │  Agent exists, NOT YET RUNNABLE.
                                     │
═══════════════════════════════════════════════════════════════════════════════
                                     │
                                     ▼
                       ╔══════════════════════════╗
                       ║   await agent.prepare()  ║
                       ║       (async)            ║
                       ╚══════════════════════════╝
                                     │
            ┌────────────────────────┼─────────────────────────┐
            ▼                        ▼                         ▼
  ┌──────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
  │ STEP 1               │  │ STEP 2              │  │ STEP 3           │
  │ capabilities.prepare │  │ For each layer:     │  │ Mark prepared    │
  │                      │  │   collect vars      │  │                  │
  │ • Load skills async  │  │   render layer      │  │ _prepared = True │
  │ • Build read_skill_  │  │   store rendered    │  │                  │
  │   resource tool      │  │                     │  │ rendered_layers  │
  │ • Re-validate names  │  │ Store result in     │  │ now accessible   │
  │   with skill tools   │  │ _rendered_layers    │  │                  │
  │   merged in          │  │ keyed by type       │  │                  │
  │                      │  │                     │  │                  │
  │ ❌ FAILS LOUD if     │  │ ❌ FAILS LOUD if    │  │                  │
  │   • skill broken    │  │    layer.render()   │  │                  │
  │   • duplicate name  │  │    raises (wrapped  │  │                  │
  │   • registry connect│  │    with layer name) │  │                  │
  └──────────────────────┘  └─────────────────────┘  └──────────────────┘

  STEP 2 in detail — what variables go to each layer:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ PromptVariablesBuilder.collect(LayerType)                                │
  │                                                                          │
  │ AgentPolicyLayer  → {name, description, instructions}                    │
  │                     [from agent attrs]                                   │
  │                                                                          │
  │ MemoryLayer       → {persistent_memories, [memory_tools]}                │
  │                     [await memory.get_context()]                         │
  │                     [tool names from capabilities.memory_tools]          │
  │                                                                          │
  │ KnowledgeLayer    → {[retrieval_tools]}                                  │
  │                     [tool names from capabilities.knowledge_tools]       │
  │                                                                          │
  │ RoutineLayer      → {[routine_search_tool], [routine_fetch_tool]}        │
  │                     [match by fixed names: "search_routines",            │
  │                      "get_routine"]                                      │
  │                                                                          │
  │ SkillsLayer       → {loaded_skills}                                      │
  │                     [from capabilities.loaded_skills, post skill load]   │
  │                                                                          │
  │ PriorityToolsLayer→ {priority_tools}                                     │
  │                     [from capabilities.priority_tools]                   │
  │                                                                          │
  │ ContextLayer      → {current_session_summary}                            │
  │                     [await context.get_current_session_summary()]        │
  │                                                                          │
  │ RenderingLayer    → {} (static template, no vars)                        │
  │ TaskAnalysisLayer → {} (static template, no vars)                        │
  │                                                                          │
  │ <CustomLayer>     → {} (renders with extra_variables only)               │
  └──────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

  STATE AFTER prepare() COMPLETES:

  ┌────────────────────────────────────────────────────────────────────────┐
  │  agent._prepared = True                                                 │
  │                                                                         │
  │  agent.capabilities.all_tools  ← list[CoreTool], includes:              │
  │    • explicit tools (my_func_a, my_func_b)                              │
  │    • memory_tools (list/update/delete)                                  │
  │    • knowledge_tools (search_docs, search_tickets)                      │
  │    • routine_tools (search_routines, get_routine)                       │
  │    • context_tools (search_context)                                     │
  │    • skill_tools (skill funcs + read_skill_resource)                    │
  │                                                                         │
  │  agent.rendered_layers ← dict[type[CoreLayer], str]                     │
  │    {                                                                    │
  │      AgentPolicyLayer:    "<AGENT_POLICY>...</...>",                    │
  │      TaskAnalysisLayer:   "<TASK_AND_ANALYSIS>...</...>",               │
  │      RenderingLayer:      "<RENDERING>...</...>",                       │
  │      PriorityToolsLayer:  "<PRIORITY_TOOLS>...</...>" (or empty),       │
  │      SkillsLayer:         "# Skills\n## pr_review...",                  │
  │      RoutineLayer:        "<ROUTINES>..." (or empty),                   │
  │      KnowledgeLayer:      "<RETRIEVAL_AUGMENTED_INSTRUCTIONS>..."       │
  │                              (or empty),                                │
  │      ContextLayer:        "<SESSION_CONTEXT>..." (or empty),            │
  │      MemoryLayer:         "<PERSISTENT_MEMORY>...",                     │
  │    }                                                                    │
  │                                                                         │
  │  agent is now READY for run().                                          │
  └────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

  WHAT'S MISSING (next steps):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  ❌ Agent.run(...)                                              │
  │     The actual react-loop. Takes user input + chat_history.         │
  │     Assembles system_prompt from rendered_layers in client-native   │
  │     order. Calls client.complete(). Handles tool calls. Loops.      │
  │                                                                     │
  │  ❌ CoreChatCompletionClient implementations                        │
  │     • AnthropicClient.format_messages(rendered_layers, history)     │
  │     • OpenAIClient.format_messages(rendered_layers, history)        │
  │     Each provider orders/groups layers differently.                 │
  │                                                                     │
  │  ❌ Tool execution + approval handling                              │
  │     AUTO_APPROVED runs immediately.                                 │
  │     ASK_APPROVED pauses, surfaces to user, awaits decision.         │
  │                                                                     │
  │  ❌ Middleware pipeline                                             │
  │     @routine and #skill triggers — intercept before LLM.            │
  │     Observability events.                                           │
  │                                                                     │
  │  ❌ Streaming support                                               │
  │     Token-by-token streaming through middlewares.                   │
  └─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════