"""An agent as a file: its pieces and settings, never a key. Needs OPENAI_API_KEY
to build the client again."""

from pathlib import Path

from max_ai.agents import Agent
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.capabilities.middleware import LoggingMiddleware


def main() -> None:
    agent = Agent(
        name="StoredAssistant",
        description="An agent ready to be stored.",
        instructions="Answer briefly.",
        client=OpenAIChatCompletionClient(model="gpt-5.4-nano"),
        middlewares=[LoggingMiddleware()],
    )
    # 1. Save: only what differs from the defaults, so it reads like one written by hand.
    path = Path("./local/agent.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(agent.to_yaml(), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))

    # 2. Load: the key is read from the environment again.
    again = Agent.from_yaml(path)
    print(again.name, again.client.model, [type(m).__name__ for m in again.middlewares])


if __name__ == "__main__":
    main()
