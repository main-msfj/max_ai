"""
Contracts for skill registries.

Skills are self-contained packages (instructions + tools + resources)
that live in an external repository. A registry knows how to resolve
skill names into fully-loaded Skill objects ready to be plugged into
an agent.

Unlike Memory / Knowledge / Routines, skills do not generate their
own tools via ``as_tools()``. Instead, the registry returns Skill
value objects that carry their tools directly; the agent merges them
into its tool registry at init time.

The registry also exposes a single global ``read_skill_resource``
tool (built via ``make_read_resource_tool``) that lets the LLM pull
auxiliary reference files on demand. The *how* of reading (filesystem,
HTTP, DB) is backend-specific via ``_read_resource_impl``; the
validation, error messages, and tool wrapping are shared.
"""

import re
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

from .capability import CoreAgentCapabilities
from ..types.tools import ToolApprovalMode
from ..tools.function_as_tool import FunctionAsTool

if t.TYPE_CHECKING:
    from .tools import CoreTool
    from ..types.skills import Skill, ResourceMeta


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class CoreSkillRegistry(CoreAgentCapabilities[BaseModel], ABC):
    """Abstract base class for skill registries.

    A registry resolves skill names into loaded ``Skill`` objects. The
    matching (which skills apply to a given user request) happens
    outside the registry — typically in the UI layer — and the result
    is passed as a list of names to ``load()``.
    """

    def __init__(self, name: str, skills: list[str]) -> None:
        super().__init__()
        self.name: str = self._validate_name(name)
        self.skills: list[str] = self._validate_skill_names(skills)

    @staticmethod
    def _validate_skill_names(skills: list[str]) -> list[str]:
        if not isinstance(skills, list):
            raise TypeError(f"skills must be a list, got {type(skills).__name__}")
        if not skills:
            raise ValueError("skills list cannot be empty")
        if not all(isinstance(s, str) and s for s in skills):
            raise ValueError("skills must be a list of non-empty strings")
        dupes = {s for s in skills if skills.count(s) > 1}
        if dupes:
            raise ValueError(f"Duplicate skill names: {sorted(dupes)}")
        return list(skills)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name:
            raise TypeError("name must be a non-empty string")
        if not _VALID_NAME_RE.match(name):
            raise ValueError(
                f"Invalid registry name {name!r}. Allowed characters: "
                "letters, digits, underscores, hyphens."
            )
        return name

    # -------- READ OPERATIONS -----------------------------------------------------------
    @abstractmethod
    async def load(self, names: list[str]) -> list["Skill"]:
        """Resolve skill names into fully-loaded Skill objects.

        Called by the framework during ``agent.prepare()``, typically
        with ``self.skills`` as argument.

        Validates each skill end-to-end: frontmatter, script discovery,
        type-hint completeness, resource catalog consistency. Fails
        loud on any inconsistency — a broken skill never loads
        partially.

        Args:
            names: List of skill names to resolve.

        Returns:
            A list of Skill objects, one per requested name, in the
            same order.
        """
        ...

    # -------- RESOURCE ACCESS -----------------------------------------------------------
    @abstractmethod
    async def _read_resource_impl(
        self, skill_name: str, resource: "ResourceMeta"
    ) -> str:
        """Backend-specific: read the actual content of a resource.

        Called by the ``read_skill_resource`` tool *after* the skill
        and filename have been validated against the loaded catalog.
        Implementations only need to fetch the bytes — no validation
        required.

        Args:
            skill_name: Name of the skill the resource belongs to.
                Provided for backends that need it for routing
                (e.g. an HTTP backend keying by skill).
            resource: The ``ResourceMeta`` from the loaded skill,
                with its path/locator already resolved.

        Returns:
            The textual content of the resource.
        """
        ...

    def make_read_resource_tool(self, loaded_skills: list["Skill"]) -> "CoreTool":
        """Build the global ``read_skill_resource`` tool.

        Called by ``AgentCapabilities.prepare()`` once all skills are
        loaded. The returned tool closes over the catalog of every
        loaded skill, so name/filename validation is local and the
        error messages can list what's actually available.

        Args:
            loaded_skills: The Skill objects produced by ``load()``.

        Returns:
            A ``FunctionAsTool`` wrapping a closure that validates
            inputs, then delegates the actual read to
            ``_read_resource_impl``.
        """
        # Build the catalog: {skill_name: {filename: ResourceMeta}}.
        catalog: dict[str, dict[str, "ResourceMeta"]] = {
            skill.block.name: dict(skill.resources) for skill in loaded_skills
        }

        registry = self  # capture for the closure

        async def read_skill_resource(skill_name: str, resource_filename: str) -> str:
            """Read an auxiliary reference file from a loaded skill.

            Use this when a skill's instructions point you to one of
            its reference files (listed in the skill block under
            "resources"). The file is loaded on demand so it doesn't
            occupy context until you actually need it.

            Args:
                skill_name: Name of the skill the resource belongs to.
                resource_filename: Filename of the resource as
                    declared in the skill's resource list.

            Returns:
                The textual content of the resource file.
            """
            if skill_name not in catalog:
                available = sorted(catalog.keys())
                raise ValueError(
                    f"Unknown skill {skill_name!r}. Available skills: {available}"
                )
            skill_resources = catalog[skill_name]
            if resource_filename not in skill_resources:
                available = sorted(skill_resources.keys())
                raise ValueError(
                    f"Unknown resource {resource_filename!r} in skill "
                    f"{skill_name!r}. Available resources: {available}"
                )
            return await registry._read_resource_impl(
                skill_name, skill_resources[resource_filename]
            )

        return FunctionAsTool(
            func=read_skill_resource,
            approval_mode=ToolApprovalMode.AUTO_APPROVED,
        )
