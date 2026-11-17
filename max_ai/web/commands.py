"""The ``/`` commands of the web UI, the same as the CLI's. Those that show
something answer with a panel; the page does the rest on its own (``where``)."""

from __future__ import annotations

import time
import typing as t

from ..base.tools import CoreTool
from ..capabilities.mcp.tool import MCPTool

if t.TYPE_CHECKING:
    from .conversation import Conversation
    from .server import WebHost

# name → (what it does, where it runs: "panel" on the server, "page" in the browser)
COMMANDS: dict[str, tuple[str, str]] = {
    "/help": ("show commands and shortcuts", "panel"),
    "/tools": ("list the tools the agent can call", "panel"),
    "/mcp": ("list the MCP servers with their tools", "panel"),
    "/skills": ("list the agent's skills", "panel"),
    "/session": ("this conversation's id and folder", "panel"),
    "/permissions": ("the sandbox and your allow, ask and deny rules", "panel"),
    "/resume": ("pick a saved conversation", "panel"),
    "/files": ("open the workspace files", "page"),
    "/trace": ("open the Trajectory: model calls and tools", "page"),
    "/thinking": ("show or hide thinking", "page"),
    "/new": ("start a new conversation", "page"),
    "/clear": ("start a new conversation (same as /new)", "page"),
}

SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("enter", "send · shift+enter for a new line"),
    ("esc", "stop the running turn, or close a menu or panel"),
    ("/", "open the command menu · ↑↓ move · tab or enter pick"),
    ("@path", "attach a workspace file · @ opens the file menu"),
    ("!cmd", "run a shell command in the workspace"),
    ("ctrl+b", "workspace files"),
    ("ctrl+.", "Trajectory"),
)


class Commands:
    """What each panel command shows, and the skills it lists."""

    RULES_FOR = 60  # Seconds the rules of setting.json are kept before reading them again

    def __init__(self, host: WebHost) -> None:
        self.host = host
        self.skills: list[tuple[str, str]] | None = None  # name, description; loaded once
        self.rules: tuple[float, dict[str, list[str]]] | None = None  # When read, and the rules

    async def menu(self) -> list[dict[str, str]]:
        """The entries of the ``/`` menu: commands, then skills."""
        entries = [
            {"name": name, "description": description, "kind": where}
            for name, (description, where) in COMMANDS.items()
        ]
        skills = await self.load_skills()
        return entries + [{"name": f"/{n}", "description": d, "kind": "skill"} for n, d in skills]

    async def panel(self, conversation: Conversation, name: str) -> tuple[str, list[dict[str, t.Any]]]:
        """The title and the rows of a command's panel. A row has ``name``,
        ``text`` and maybe ``tag`` (a warning), ``href`` or ``group``."""
        if name == "/help":
            rows = [{"name": n, "text": d} for n, (d, _) in COMMANDS.items()]
            rows += [{"group": "Shortcuts"}] + [{"name": k, "text": d} for k, d in SHORTCUTS]
            return "Help", rows
        if name == "/tools":
            return "Tools", [self._tool_row(tool, tool.name) for tool in await self.all_tools()]
        if name == "/mcp":
            return "MCP servers", await self._mcp_rows()
        if name == "/skills":
            skills = await self.load_skills()
            if not skills:
                return "Skills", [{"text": "This agent has no skills."}]
            return "Skills", [{"name": f"/{n}", "text": d} for n, d in skills]
        if name == "/session":
            store = self.host.store
            return "Session", [
                {"name": "id", "text": conversation.sid},
                {"name": "folder", "text": self.host.workspace.folder(conversation.sid)},
                {"name": "saved", "text": "yes" if store is not None else "no store: nothing is saved"},
                {"name": "messages", "text": str(len(conversation.context.messages))},
            ]
        if name == "/permissions":
            return "Permissions", await self._permission_rows()
        if name == "/resume":
            return "Resume", await self._session_rows()
        return "Unknown command", [{"text": f"{name} is not a command. Try /help."}]

    async def all_tools(self) -> list[CoreTool]:
        """Every tool of a run, MCP ones included; the agent's own if MCP fails."""
        agent = self.host.agent
        try:
            registry = await agent._registry()  # pyright: ignore[reportPrivateUsage]
        except Exception:  # noqa: BLE001 — a server down must not break the panel
            return list(agent.tools)
        return registry.all_tools()

    async def load_skills(self) -> list[tuple[str, str]]:
        """The agent's skills with their descriptions, installed in the user's sandbox."""
        if self.skills is not None:
            return self.skills
        agent = self.host.agent
        skills, sandbox = agent.skills, agent.sandbox
        if skills is None or sandbox is None:
            self.skills = []
            return self.skills
        try:
            box = sandbox.bind(self.host.user_id, skills.variables())
            await box.prepare()
            await skills.sync(box)
            installed = await box.installed_skills()
        except Exception:  # noqa: BLE001 — a bad skill must not break the menu
            return []
        self.skills = [(n, installed.get(n, {}).get("description", "")) for n in skills.skills]
        return self.skills

    async def _mcp_rows(self) -> list[dict[str, t.Any]]:
        servers = [server.server_id for server in self.host.agent.mcp]
        if not servers:
            return [{"text": "This agent has no MCP servers."}]
        tools = [tool for tool in await self.all_tools() if isinstance(tool, MCPTool)]
        rows: list[dict[str, t.Any]] = []
        for sid in servers:
            mine = [tool for tool in tools if tool.server_id == sid]
            rows.append({"group": f"{sid} · {len(mine)} tools" if mine else f"{sid} · not connected"})
            rows += [self._tool_row(tool, tool.name.removeprefix(sid + "_")) for tool in mine]
        return rows

    async def _permission_rows(self) -> list[dict[str, t.Any]]:
        """Where commands run, and the rules of the user's setting.json."""
        sandbox = self.host.agent.sandbox
        rows: list[dict[str, t.Any]] = [{"name": "sandbox", "text": self.host.sandbox_label()}]
        if sandbox is None:
            return rows + [{"text": "Without a sandbox there is no setting.json: each tool asks as it is built to."}]
        box = sandbox.bind(self.host.user_id)
        rows.append({"name": "settings", "text": str(box.layout.settings)})
        try:
            rules = await self.permissions()
        except Exception as error:  # noqa: BLE001 — the sandbox may be down
            return rows + [{"text": f"Could not read the rules: {error}"}]
        for kind, meaning in (("allow", "run without asking"), ("ask", "always asked"), ("deny", "never run")):
            found = rules.get(kind, [])
            rows.append({"group": f"{kind} · {meaning}"})
            rows += [{"name": rule, "text": ""} for rule in found] or [{"text": "none"}]
        rows.append({"group": "Always asked, whatever the rules say"})
        rows.append({"text": "rm, sudo, git push --force, git reset --hard, kill, eval, commands built with $( ) or backticks, and setting.json itself."})
        return rows

    async def permissions(self) -> dict[str, list[str]]:
        """The rules of setting.json, read again after ``RULES_FOR`` seconds or
        when "always allow" keeps one (``self.rules = None``)."""
        if self.rules is not None and time.monotonic() - self.rules[0] < self.RULES_FOR:
            return self.rules[1]
        sandbox = self.host.agent.sandbox
        if sandbox is None:
            return {}
        box = sandbox.bind(self.host.user_id)
        try:
            rules = await box.permissions()
        except Exception:  # noqa: BLE001 — no .agent/ yet: made, then read again
            await box.prepare()
            rules = await box.permissions()
        self.rules = (time.monotonic(), rules)
        return rules

    async def _session_rows(self) -> list[dict[str, t.Any]]:
        store = self.host.store
        if store is None:
            return [{"text": "No session store: nothing is saved."}]
        found = await store.list_sessions(self.host.user_id, limit=30)
        return [
            {"name": self.host.ago(info.updated_at), "text": info.title or "(untitled)",
             "href": f"/s/{info.session_id}"}
            for info in found
        ] or [{"text": "No saved conversations yet."}]

    @staticmethod
    def _tool_row(tool: CoreTool, name: str) -> dict[str, t.Any]:
        mode = str(getattr(tool.approval_mode, "value", tool.approval_mode)).lower()
        lines = (tool.description or "").strip().splitlines()
        return {"name": name, "text": lines[0][:140] if lines else "", "tag": "asks" if "ask" in mode else ""}
