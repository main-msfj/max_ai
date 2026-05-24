"""Stack Core (Jinja2 version)"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateSyntaxError,
    meta,
    select_autoescape,
)
from pydantic import BaseModel

from .component import ComponentBase
from ..errors.stacks import StackError


class CoreLayer(ComponentBase[BaseModel], ABC):
    """
    Immutable base for prompt stacks (Jinja2-backed).

    Uses Jinja2 syntax:
      - {{ name }}                  variable
      - {% if x %}...{% endif %}    conditional
      - {% for item in items %}     loop
      - {% include "block.j2" %}    composition
      - {# comment #}               comment

    Lifecycle:
      - Template is parsed, AST-analyzed, and validated ONCE at construction.
      - `render(variables)` reuses the compiled template — fast, stateless.

    Contract semantics:
      - Required variables MUST be referenced in the template and MUST be
        passed to render(). A required variable not referenced is an error
        (schema drift: you declared it but forgot to use it).
      - Optional variables MAY be referenced; if referenced and dereferenced
        at runtime without being passed, StrictUndefined raises.
      - Extra variables provided via ``extra_variables`` at construction
        are treated as optional and merged into render() inputs (with
        render-time variables winning on collision).
    """

    _DEFAULT_TEMPLATE_PATH = Path(__file__).parent.parent / "stacks" / "prompts"

    def __init__(
        self,
        name: str,
        template: str | None = None,
        load_from: str | Path | None = None,
        extra_variables: dict[str, t.Any] | None = None,
    ) -> None:
        if template is not None and load_from is not None:
            raise StackError.template_or_load_from()

        self.name = name
        self._inline_template = template
        self._load_from = load_from
        self._extra_variables: dict[str, t.Any] = dict(extra_variables or {})

        # Build environment first — needed for parsing AND rendering.
        # Includes the load_from directory in search paths so `{% include %}`
        # can reference sibling files next to the user's template.
        self._env = self._build_environment()

        self._template_source: str = self._resolve_template()

        # Parse once, analyze AST, compile, validate.
        self._template_variables: frozenset[str] = self._parse_placeholders(
            self._template_source
        )
        self.validate_placeholders()
        self._compiled: Template = self._env.from_string(self._template_source)

    # -------- ENVIRONMENT -----------------------------------------------------------
    def _build_environment(self) -> Environment:
        """
        Build a Jinja2 Environment.

        - StrictUndefined: missing-variable access raises instead of becoming "".
        - trim_blocks / lstrip_blocks: clean output around {% %} tags.
        - Loader search path includes both the default prompts dir and the
          directory of `load_from` (if set), so includes resolve naturally.
        """
        search_paths: list[str] = [str(self._DEFAULT_TEMPLATE_PATH)]
        if self._load_from is not None:
            load_dir = Path(self._load_from).parent
            # User's directory gets priority over default.
            search_paths.insert(0, str(load_dir))

        return Environment(
            loader=FileSystemLoader(search_paths),
            undefined=StrictUndefined,
            autoescape=select_autoescape(enabled_extensions=(), default=False),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

    # -------- TEMPLATES -----------------------------------------------------------
    @abstractmethod
    def _default_template(self) -> str:
        """Subclasses provide their own default template."""
        raise NotImplementedError

    def _resolve_template(self) -> str:
        # Priority: inline user template > external file > subclass default
        if self._inline_template is not None:
            return self._inline_template
        if self._load_from is not None:
            return self._load_file(self._load_from)
        return self._default_template()

    @staticmethod
    def _load_file(file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise StackError(f"Template file not found: {path}")
        if not path.is_file():
            raise StackError(f"Template path is not a file: {path}")
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise StackError(f"Failed to read template {path}: {e}")

    @property
    def template(self) -> str:
        """Resolved template source string (read-only)."""
        return self._template_source

    def _parse_placeholders(self, template: str) -> frozenset[str]:
        """
        Extract all undeclared variables referenced in the template via
        Jinja's AST. Loop locals (`item` in `{% for item in items %}`) and
        `{% set %}` locals are excluded — only caller-provided vars remain.
        """
        try:
            ast = self._env.parse(template)
        except TemplateSyntaxError as e:
            raise StackError(f"Template syntax error in {self.name!r}: {e}")
        return frozenset(meta.find_undeclared_variables(ast))

    # -------- CONTRACT -----------------------------------------------------------
    def _required_variables(self) -> set[str]:
        """Variables that MUST be supplied to render()."""
        return set()

    def _optional_variables(self) -> set[str]:
        """Variables that MAY be supplied to render()."""
        return set()

    def validate_placeholders(self) -> None:
        """
        Verify template variables match the declared contract. Runs once at
        construction — catches typos and schema drift before any render().

        ``extra_variables`` keys are treated as optional: a custom template
        may reference them without the subclass needing to override
        ``_optional_variables()``.
        """
        required = self._required_variables()
        optional = self._optional_variables() | set(self._extra_variables.keys())
        declared = required | optional

        # Required vars must actually be referenced in the template.
        missing_in_template = required - self._template_variables
        if missing_in_template:
            raise StackError.missing_placeholders(
                name=self.name, missing=missing_in_template
            )

        # Template must not reference undeclared variables.
        unexpected = self._template_variables - declared
        if unexpected:
            raise StackError.unexpected_placeholders(
                name=self.name, unexpected=unexpected
            )

    # -------- RENDER -----------------------------------------------------------
    def render(self, variables: t.Mapping[str, t.Any]) -> str:
        """
        Render the compiled template with the given variables.

        Variables provided here are merged on top of ``extra_variables``
        passed at construction time — i.e. render-time inputs win on
        collision. Required variables must be present in the merged
        result. Optional variables may be omitted, but the template
        should guard their access with ``{% if %}`` — otherwise
        StrictUndefined raises at render time.
        """
        merged: dict[str, t.Any] = {**self._extra_variables, **variables}
    
        required = self._required_variables()
        missing = required - merged.keys()
        if missing:
            raise StackError.missing_variable(name=self.name, var=missing)
        
        # Fill in optional variables that weren't provided, so {% if %} works
        # without needing `is defined` checks in every template.
        for opt in self._optional_variables():
            merged.setdefault(opt, None)
        
        return self._compiled.render(**merged)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
