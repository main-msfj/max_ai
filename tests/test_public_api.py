"""The package root re-exports the common entry points."""

import max_ai


def test_root_exports_resolve():
    assert isinstance(max_ai.__version__, str) and max_ai.__version__
    for name in max_ai.__all__:
        assert getattr(max_ai, name) is not None, name
    from max_ai.agents import Agent
    assert max_ai.Agent is Agent
