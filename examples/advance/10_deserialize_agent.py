"""Rebuild an agent from the saved JSON."""

from pathlib import Path

from max_ai.agents import Agent


def main() -> None:
    path = Path("./local/agent.json")
    agent = Agent.deserialize(path.read_text(encoding="utf-8"))
    print(agent.name)
    print(agent.client.model)


if __name__ == "__main__":
    main()
