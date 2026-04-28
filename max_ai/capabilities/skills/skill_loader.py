"""
Backend-agnostic skill loader.

Takes a directory laid out in the standard skill format and produces a
fully-validated ``Skill`` object. Knows nothing about where the
directory came from — could be a local copy, an extracted tarball, a
git checkout, anything that ends up as a real filesystem path.

Each registry implementation (``LocalSkillRegistry``,
``S3SkillRegistry``, ``GitSkillRegistry``, ...) is responsible for
materializing skill packages into a tmp directory and then handing the
path to ``load_skill_from_dir``. The loader itself never reaches across
the network or knows about the original source.

Public entry point: ``load_skill_from_dir(skill_dir)``.

Expected layout::

    skill_dir/
        SKILL.md               # frontmatter + body
        scripts/               # optional, *.py modules with public functions
        references/            # optional, files declared in frontmatter
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import typing as t
from pathlib import Path

import frontmatter  # pyright: ignore[reportMissingTypeStubs]

from ...core.blocks import SkillBlock
from ...types.skills import Skill, ResourceMeta
from ...types.tools import ToolApprovalMode
from ...tools.function_as_tool import FunctionAsTool


# -------- FILENAMES & CONVENTIONS -----------------------------------------------------------
SKILL_FILE = "SKILL.md"
SCRIPTS_DIR = "scripts"
REFERENCES_DIR = "references"


# -------- PUBLIC ENTRY POINT -----------------------------------------------------------
def load_skill_from_dir(skill_dir: Path) -> Skill:
    """Load and validate a skill from a directory.

    Fails loud on any inconsistency: missing SKILL.md, bad frontmatter,
    missing type hints in scripts, mismatch between declared and
    actual resources, etc.
    """
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    metadata, instructions = _parse_skill_md(skill_dir)

    name = metadata["name"]
    description = metadata["description"]
    approval_overrides: dict[str, ToolApprovalMode] = _parse_approval_overrides(
        metadata.get("approval_overrides", {})
    )
    declared_resources: dict[str, str] = metadata.get("resources", {}) or {}

    tools = _discover_and_wrap_tools(
        skill_name=name,
        skill_dir=skill_dir,
        approval_overrides=approval_overrides,
    )

    resources = _catalog_resources(
        skill_dir=skill_dir,
        declared=declared_resources,
    )

    return Skill(
        block=SkillBlock(
            name=name,
            description=description,
            instructions=instructions,
        ),
        tools=tools,
        resources=resources,
    )


# -------- SKILL.MD PARSING -----------------------------------------------------------
def _parse_skill_md(skill_dir: Path) -> tuple[dict[str, t.Any], str]:
    """Read SKILL.md and return (frontmatter_dict, body_markdown)."""
    skill_md = skill_dir / SKILL_FILE
    if not skill_md.is_file():
        raise FileNotFoundError(
            f"Skill {skill_dir.name!r} is missing required file {SKILL_FILE!r}"
        )

    try:
        post = frontmatter.load(skill_md)
    except Exception as e:
        raise ValueError(
            f"Skill {skill_dir.name!r}: failed to parse {SKILL_FILE}: {e}"
        ) from e

    metadata = dict(post.metadata)

    for required in ("name", "description"):
        if not metadata.get(required):
            raise ValueError(
                f"Skill {skill_dir.name!r}: missing required frontmatter "
                f"field {required!r} in {SKILL_FILE}"
            )

    return metadata, post.content


def _parse_approval_overrides(
    raw: dict[str, t.Any],
) -> dict[str, ToolApprovalMode]:
    """Convert frontmatter approval_overrides into enum values."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"approval_overrides must be a mapping, got {type(raw).__name__}"
        )
    parsed: dict[str, ToolApprovalMode] = {}
    for fn_name, mode in raw.items():
        try:
            parsed[fn_name] = ToolApprovalMode(mode)
        except ValueError as e:
            raise ValueError(
                f"Invalid approval mode {mode!r} for function {fn_name!r}: {e}"
            ) from e
    return parsed


# -------- SCRIPT DISCOVERY & TOOL WRAPPING -----------------------------------------------------------
def _discover_and_wrap_tools(
    skill_name: str,
    skill_dir: Path,
    approval_overrides: dict[str, ToolApprovalMode],
) -> list[FunctionAsTool]:
    """Import every .py under scripts/ and wrap public functions as tools."""
    scripts_dir = skill_dir / SCRIPTS_DIR
    if not scripts_dir.is_dir():
        # Skills without scripts are valid (pure instructions).
        if approval_overrides:
            raise ValueError(
                f"Skill {skill_name!r}: approval_overrides declared but "
                f"no {SCRIPTS_DIR}/ directory exists"
            )
        return []

    tools: list[FunctionAsTool] = []
    seen_names: set[str] = set()
    referenced_overrides: set[str] = set()

    for py_file in sorted(scripts_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module = _import_script(skill_name, py_file)
        public_fns = _extract_public_functions(module)

        for fn_name, fn in public_fns:
            _validate_type_hints(skill_name, fn)

            if fn_name in seen_names:
                raise ValueError(
                    f"Skill {skill_name!r}: duplicate function name "
                    f"{fn_name!r} across scripts"
                )
            seen_names.add(fn_name)

            approval = approval_overrides.get(fn_name, ToolApprovalMode.ASK_APPROVED)
            if fn_name in approval_overrides:
                referenced_overrides.add(fn_name)

            tools.append(FunctionAsTool(func=fn, approval=approval))

    unreferenced = set(approval_overrides) - referenced_overrides
    if unreferenced:
        raise ValueError(
            f"Skill {skill_name!r}: approval_overrides references unknown "
            f"functions: {sorted(unreferenced)}"
        )

    return tools


def _import_script(skill_name: str, py_file: Path) -> t.Any:
    """Import a .py file under a unique sys.modules name to isolate skills.

    Two skills can have a module named ``utils.py`` without colliding
    because each ends up as ``_skills.{skill}.{module}`` in sys.modules.
    """
    module_name = f"_skills.{skill_name}.{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Skill {skill_name!r}: could not build import spec for {py_file}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Clean up the half-loaded module so retries don't see a stub.
        sys.modules.pop(module_name, None)
        raise ImportError(
            f"Skill {skill_name!r}: failed to import {py_file.name}: {e}"
        ) from e
    return module


def _extract_public_functions(module: t.Any) -> list[tuple[str, t.Callable]]:
    """Return (name, fn) pairs for public top-level functions defined in module.

    Honors ``__all__`` if present. Otherwise uses visibility convention
    (no leading underscore) and filters out anything imported from
    elsewhere so helpers brought in via ``from foo import bar`` don't
    accidentally get exposed as tools.
    """
    declared = getattr(module, "__all__", None)
    candidates: list[tuple[str, t.Any]] = []

    if declared is not None:
        for n in declared:
            if not hasattr(module, n):
                raise ValueError(
                    f"Module {module.__name__!r} declares {n!r} in __all__ "
                    "but it is not defined"
                )
            candidates.append((n, getattr(module, n)))
    else:
        for n in dir(module):
            if n.startswith("_"):
                continue
            candidates.append((n, getattr(module, n)))

    result: list[tuple[str, t.Callable]] = []
    for n, obj in candidates:
        if not (inspect.isfunction(obj) or inspect.iscoroutinefunction(obj)):
            continue
        if obj.__module__ != module.__name__:
            # Imported from elsewhere; skip unless __all__ explicitly lists it.
            if declared is None:
                continue
        result.append((n, obj))
    return result


def _validate_type_hints(skill_name: str, fn: t.Callable) -> None:
    """Ensure every parameter has a type hint. Return type is optional."""
    try:
        hints = t.get_type_hints(fn)
    except Exception as e:
        raise ValueError(
            f"Skill {skill_name!r}: could not resolve type hints for "
            f"{fn.__name__!r}: {e}"
        ) from e

    sig = inspect.signature(fn)
    missing = [p for p in sig.parameters if p not in hints]
    if missing:
        raise ValueError(
            f"Skill {skill_name!r}: function {fn.__name__!r} is missing "
            f"type hints for parameter(s): {missing}"
        )


# -------- RESOURCE CATALOG -----------------------------------------------------------
def _catalog_resources(
    skill_dir: Path,
    declared: dict[str, str],
) -> dict[str, ResourceMeta]:
    """Cross-check declared resources against references/ on disk.

    Fail loud on any mismatch in either direction.
    """
    refs_dir = skill_dir / REFERENCES_DIR
    on_disk: set[str] = set()
    if refs_dir.is_dir():
        on_disk = {p.name for p in refs_dir.iterdir() if p.is_file()}

    declared_set = set(declared)

    missing_on_disk = declared_set - on_disk
    if missing_on_disk:
        raise ValueError(
            f"Skill {skill_dir.name!r}: declared resources not found in "
            f"{REFERENCES_DIR}/: {sorted(missing_on_disk)}"
        )

    undeclared_on_disk = on_disk - declared_set
    if undeclared_on_disk:
        raise ValueError(
            f"Skill {skill_dir.name!r}: files in {REFERENCES_DIR}/ are not "
            f"declared in frontmatter resources: {sorted(undeclared_on_disk)}"
        )

    return {
        filename: ResourceMeta(
            filename=filename,
            path=(refs_dir / filename).resolve(),
            description=description,
        )
        for filename, description in declared.items()
    }
