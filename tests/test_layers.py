"""
Smoke tests for every prompt layer.

One test per layer. Each test:
  1. Constructs the layer with its default template.
  2. Renders it with realistic variables.
  3. Asserts the output is non-empty and contains a sentinel string
     that proves the variables made it through to the rendered output.

If a layer's .j2 file has a typo, breaks the contract, or fails to
parse — the test fails immediately. That's all this suite is for.
"""



from max_ai.capabilities.stacks.agent_policy_layer import AgentPolicyLayer
from max_ai.capabilities.stacks.context_layer import ContextLayer
from max_ai.capabilities.stacks.knowledge_layer import KnowledgeLayer
from max_ai.capabilities.stacks.memory_layer import MemoryLayer
from max_ai.capabilities.stacks.rendering_layer import RenderingLayer
from max_ai.capabilities.stacks.skills_layer import SkillsLayer
from max_ai.capabilities.stacks.task_analysis_layer import TaskAnalysisLayer
from max_ai.base.memory import MemoryRecord
from max_ai.core.blocks import SkillBlock


def test_agent_policy_layer():
    layer = AgentPolicyLayer()
    out = layer.render({
        "name": "TestAgent",
        "description": "An agent for testing.",
        "instructions": "Always answer in lowercase.",
    })
    assert out
    assert "TestAgent" in out
    assert "Always answer in lowercase." in out


def test_memory_layer_with_facts_and_tools():
    layer = MemoryLayer()
    out = layer.render({
        "persistent_memories": [
            MemoryRecord(category="location", memory="Lives in Buenos Aires"),
            MemoryRecord(category="role", memory="Software engineer"),
        ],
        "memory_tools": ["create_or_update", "delete_memory"],
    })
    assert out
    assert "Lives in Buenos Aires" in out
    assert "[location]" in out
    assert "create_or_update(category, memory)" in out
    assert "<memory_management>" in out


def test_memory_layer_no_tools():
    """When memory has no tools, the management block must disappear."""
    layer = MemoryLayer()
    out = layer.render({
        "persistent_memories": [
            MemoryRecord(category="role", memory="Software engineer"),
        ],
        # No memory_tools — read-only memory case.
    })
    assert out
    assert "Software engineer" in out
    assert "<memory_management>" not in out


def test_memory_layer_empty():
    """No memories yet, no tools — still renders the wrapper but says so."""
    layer = MemoryLayer()
    out = layer.render({"persistent_memories": []})
    assert out
    assert "No memories are included" in out


def test_knowledge_layer_with_tools():
    layer = KnowledgeLayer()
    out = layer.render({
        "retrieval_tools": ["search_docs", "search_tickets"],
    })
    assert out
    assert "search_docs" in out
    assert "search_tickets" in out


def test_knowledge_layer_no_tools():
    """No retrieval tools → entire layer renders empty."""
    layer = KnowledgeLayer()
    out = layer.render({"retrieval_tools": []})
    assert out.strip() == ""


def test_skills_layer_with_skills():
    layer = SkillsLayer()
    skill = SkillBlock(
        name="pr_review",
        description="Review pull requests.",
    )
    out = layer.render({"loaded_skills": [skill]})
    assert out
    assert "pr_review" in out
    assert "Review pull requests." in out
    assert "`pr_review`" in out
    assert "skills/pr_review/SKILL.md" in out  # read with read_file
    assert "$SKILLS/<skill-name>/" in out  # where bash finds its scripts
    assert "$WORKSPACE/skills" not in out  # skills are not inside the workspace
    assert "Never treat a skill name as a callable tool." in out


def test_skills_layer_empty():
    layer = SkillsLayer()
    out = layer.render({"loaded_skills": []})
    assert out.strip() == ""


def test_context_layer_with_summary():
    layer = ContextLayer()
    out = layer.render({
        "current_session_summary": (
            "We were discussing spaceship designs and rocket fuel options."
        ),
    })
    assert out
    assert "spaceship designs" in out
    assert "<SESSION_CONTEXT>" in out


def test_context_layer_no_summary():
    layer = ContextLayer()
    out = layer.render({})
    assert out.strip() == ""


def test_task_analysis_layer():
    """Static layer — no variables. Just confirm it builds and renders."""
    layer = TaskAnalysisLayer()
    out = layer.render({})
    assert out
    assert "<TASK_AND_ANALYSIS>" in out


def test_rendering_layer():
    """Static layer — no variables. Just confirm it builds and renders."""
    layer = RenderingLayer()
    out = layer.render({})
    assert out  # whatever the rendering layer says, it should say something
