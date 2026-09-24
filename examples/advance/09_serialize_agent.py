"""Save an agent configuration as JSON."""

from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient


def main() -> None:
    agent = Agent(
        name="StoredAssistant",
        description="An agent ready to be stored.",
        instructions="Answer in English.",
        client=OpenAIChatCompletionClient(model="gpt-5.6-luna", api_key="YOUR_API_KEY"),
    )
    path = Path("./local/agent.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(agent.serialize().model_dump_json(indent=2), encoding="utf-8")
    print(f"Configuration saved to {path}")


if __name__ == "__main__":
    main()
