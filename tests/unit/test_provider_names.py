"""The name a component is stored under is where it is imported from."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import max_ai

NAME = re.compile(r'component_provider_override = \(?\s*"([\w.]+)"')


def test_every_stored_name_imports_the_class_that_declares_it():
    # 1. The names the source declares, read as text: nothing is imported yet.
    package = Path(max_ai.__file__).parent
    declared = {
        name for file in package.rglob("*.py") for name in NAME.findall(file.read_text())
    }
    assert len(declared) > 30

    # 2. Each is a real path inside the package, to the class that declares it.
    for name in declared:
        assert name.startswith("max_ai."), name
        module, _, class_name = name.rpartition(".")
        cls = getattr(importlib.import_module(module), class_name)
        assert cls.component_provider_override == name
