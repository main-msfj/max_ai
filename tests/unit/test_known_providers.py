"""Every short provider name a class declares must be in the alias table."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import max_ai
from max_ai.base.component import KNOWN_PROVIDERS

SHORT_NAME = re.compile(r'component_provider_override = "(maxai\.[\w.]+)"')


def test_every_short_provider_name_is_in_the_table():
    # 1. The short names the source declares, read as text: nothing is imported.
    package = Path(max_ai.__file__).parent
    declared = {
        name for file in package.rglob("*.py") for name in SHORT_NAME.findall(file.read_text())
    }
    assert declared

    # 2. Each one is in the table.
    assert declared - set(KNOWN_PROVIDERS) == set()

    # 3. And the table points at the class that declares it.
    for name in declared:
        module, _, class_name = KNOWN_PROVIDERS[name].rpartition(".")
        cls = getattr(importlib.import_module(module), class_name)
        assert cls.component_provider_override == name
