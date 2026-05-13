"""Complete example for serving a Docker-backed Agent in the Max AI Web UI."""

from pathlib import Path

from max_ai.base.agent import Agent
from max_ai.capabilities.skills.local import LocalSkillRegistry
from max_ai.clients.ollama import OllamaChatCompletionClient
from max_ai.core.models import ModelConfig
from max_ai.executor import DockerExecutor
from max_ai.ui import server

try:
    from .docker_tools import (
        extract_action_items,
        make_decision_matrix,
        project_health_score,
        readability_snapshot,
    )
except ImportError:
    from docker_tools import (
        extract_action_items,
        make_decision_matrix,
        project_health_score,
        readability_snapshot,
    )

EXAMPLE_DIR = Path(__file__).parent
TOOL_SOURCE = EXAMPLE_DIR / "docker_tools.py"
SKILLS_SOURCE = EXAMPLE_DIR / "LocalSkills"
SKILL_NAMES = ["create-report", "create-ppt"]

TOOLSET = [
    extract_action_items,
    readability_snapshot,
    project_health_score,
    make_decision_matrix,
]


def build_client() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model="gemma4:e2b-it-q4_K_M",
        host="http://ollama:11434",
        config=ModelConfig(
            max_context_window=15000,
            supports_function_calling=True,
            supports_thinking=True,
            supports_vision=True,
        ),
        think=True,
        num_predict=10000,
    )


def build_executor() -> DockerExecutor:
    return DockerExecutor(
        tool_source=TOOL_SOURCE,
        image="maxai-sandbox:skills-demo-v3",
    )


def build_skills() -> LocalSkillRegistry:
    return LocalSkillRegistry(
        source=SKILLS_SOURCE,
        skills=SKILL_NAMES,
    )


def build_agent() -> Agent:
    return Agent(
        name="Sara",
        description="Example Max AI agent.",
        instructions=(
            "You are a helpful assistant. Be concise and useful. "
            "Use the local skills when the user asks for a report or presentation. "
            "For report or presentation requests, call search_skills first, then "
            "use skill_bash to inspect SKILL.md and run the skill script. "
            "Do not say a file was created until a tool result confirms it. "
            "Created files must go in the user workspace so they can be read or "
            "edited later with the workspace tool."
        ),
        client=build_client(),
        skills=build_skills(),
        toolset=TOOLSET,
        executor=build_executor(),
    )


agent = build_agent()

if __name__ == "__main__":
    server(agent, host="0.0.0.0", port=8000)
