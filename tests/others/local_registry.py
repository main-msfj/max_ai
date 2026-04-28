import asyncio
from max_ai.capabilities.skills import LocalSkillRegistry
from max_ai.capabilities.memory import LocalMemoryRegistry
from max_ai.capabilities.context import LocalContextRegistry
from max_ai.capabilities.routines import LocalRoutineRegistry
from max_ai.capabilities.knowledge import LocalKnowledgeRegistry


async def main():

    print("\n================================================================================= MEMORY REGISTRY ================")
    mem = LocalMemoryRegistry(
        user_id="u1",
        base_path="/max_ai/tmp/",
    )
    
    async with mem:  # connect

        all_facts = await mem.get_context()
        print("Initial memories:", len(all_facts), "facts", "\n")
        for f in all_facts:
            print(f"[{f.category}] {f.content}")
    
    print("\n================ CONTEXT REGISTRY ================")
    ctx = LocalContextRegistry(
        user_id="u1",
        session_id="session_002",  # current session
        base_path="/max_ai/tmp/",
    )
    
    async with ctx:
        # Get current session's summary
        summary = await ctx.get_current_session_summary()
        print("Current session summary:", summary)
        
        # Search for past conversations
        results = await ctx.search("spaceship rocket")
        for block in results:
            print(f"[{block.session_id}] score={block.score:.3f}")
            print(block.content)

    print("\n================ ROUTINE REGISTRY ================")
    reg = LocalRoutineRegistry(
        source_path="/max_ai/tmp/",
        routines=["client_followup", "email_reply"],   # ← solo dos autorizadas
    )
    
    async with reg:
        # Catalog: solo las dos autorizadas, no las 3 del disco
        catalog = await reg.get_catalog()
        print("Catalog:")
        for s in catalog:
            print(f"  - {s.name}")
        
        # Search: solo busca dentro de las autorizadas
        results = await reg.search("email")
        print(f"\nSearch 'email': {[r.name for r in results]}")
        
        # Fetch autorizada: OK
        block = await reg.fetch("client_followup")
        print(f"\nFetched: {block.name}")
        
        # Fetch no autorizada: fail con lista
        try:
            await reg.fetch("secret_internal")
        except ValueError as e:
            print(f"\nExpected error: {e}")
        
        # Inspect tmp
        print(f"\nTmp dir: {reg._tmp_root}")

    print("\n================ KNOWLEDGE REGISTRY ================")
    kb = LocalKnowledgeRegistry(
        name="docs",
        description="Internal Python framework documentation.",
        base_path="/max_ai/tmp/",
    )
    
    async with kb:
        results_kb = await kb.search("Python framework", limit=3)
        for block in results_kb:
            print(f"score={block.score:.3f} | {block.content}")
            print(f"  metadata: {block.metadata}\n")

    print("\n================ SKILL REGISTRY ================")
    reg = LocalSkillRegistry(
        name="dev_skills",
        source_path="/max_ai/tmp/skills/",  # directory containing skill subdirs
        skills=["pr_review", "bug_triage"],   # ← LOS DOS
    )
    
    async with reg:
        skills = await reg.load(reg.skills)
        
        for skill in skills:
            print(f"\n--- {skill.block.name} ---")
            print(f"Description: {skill.block.description}")
            print(f"Tools: {[t.name for t in skill.tools]}")
            print(f"Resources: {list(skill.resources.keys())}")
        
        # Verify tmp staging
        print(f"\nTmp root: {reg._tmp_root}")
        # ls /tmp/maxai_skills_*/pr_review/
        # ls /tmp/maxai_skills_*/bug_triage/
asyncio.run(main())