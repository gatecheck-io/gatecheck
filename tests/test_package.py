"""Package-level contract: clean top-level API, provenance docstring, no upstream imports."""

from __future__ import annotations

import sys


def test_top_level_api_imports():
    import gatecheck

    assert gatecheck.__version__ == "0.1.0"
    for name in gatecheck.__all__:
        assert hasattr(gatecheck, name), f"__all__ exports missing attribute: {name}"


def test_provenance_docstring():
    import gatecheck

    assert "market_state" in gatecheck.__doc__
    assert "2026-07-02" in gatecheck.__doc__


def test_no_upstream_market_os_dependency():
    """The package must be fully self-contained: importing it never pulls market_os."""
    import gatecheck  # noqa: F401 — ensure fully imported

    assert not any(m == "market_os" or m.startswith("market_os.") for m in sys.modules)
