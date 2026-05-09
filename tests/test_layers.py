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


from max_ai.core import MemoryBlock
from max_ai.core.blocks import SkillBlock

from max_ai.stacks.memory_layer import MemoryLayer
from max_ai.stacks.skills_layer import SkillsLayer
from max_ai.stacks.routine_layer import RoutineLayer
from max_ai.stacks.context_layer import ContextLayer
from max_ai.stacks.knowledge_layer import KnowledgeLayer
from max_ai.stacks.rendering_layer import RenderingLayer
from max_ai.stacks.agent_policy_layer import AgentPolicyLayer
from max_ai.stacks.task_analysis_layer import TaskAnalysisLayer
from max_ai.stacks.priority_tools_layer import PriorityToolsLayer

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
            MemoryBlock(category="location", content="Lives in Buenos Aires"),
            MemoryBlock(category="role", content="Software engineer"),
        ],
        "memory_tools": ["list_memories", "update_memory", "delete_memory"],
    })
    assert out
    assert "Lives in Buenos Aires" in out
    assert "[location]" in out
    assert "update_memory" in out
    assert "<memory_management>" in out


def test_memory_layer_no_tools():
    """When memory has no tools, the management block must disappear."""
    layer = MemoryLayer()
    out = layer.render({
        "persistent_memories": [
            MemoryBlock(category="role", content="Software engineer"),
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
    assert "No persistent memories" in out


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


def test_routine_layer_full():
    layer = RoutineLayer()
    out = layer.render({
        "routine_search_tool": "search_routines",
        "routine_fetch_tool": "get_routine",
    })
    assert out
    assert "search_routines" in out
    assert "get_routine" in out


def test_routine_layer_search_only():
    """Search without fetch — single-tool design."""
    layer = RoutineLayer()
    out = layer.render({
        "routine_search_tool": "search_routines",
    })
    assert out
    assert "search_routines" in out
    assert "get_routine" not in out


def test_routine_layer_no_tools():
    layer = RoutineLayer()
    out = layer.render({})
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
    assert "skill_bash" in out
    assert "search_skills" in out


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


def test_priority_tools_layer_with_tools():
    layer = PriorityToolsLayer()
    out = layer.render({
        "priority_tools": ["search_inventory", "check_order_status"],
    })
    assert out
    assert "search_inventory" in out
    assert "check_order_status" in out


def test_priority_tools_layer_empty():
    layer = PriorityToolsLayer()
    out = layer.render({"priority_tools": []})
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
