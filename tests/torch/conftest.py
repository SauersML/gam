"""Skip the gamfit.torch test suite when the torch extra is not installed.

This conftest is loaded by pytest before each test module in ``tests/torch/``
is imported. If the optional :mod:`torch` dependency is not present, we
register every test module in this directory for collection skip rather
than calling :func:`pytest.importorskip` at module top level. Top-level
``importorskip`` raises :class:`pytest.skip.Exception`, which under
``--import-mode=importlib`` is not caught by pytest's conftest loader and
aborts the entire test session instead of skipping just this directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

collect_ignore_glob: list[str] = []

if importlib.util.find_spec("torch") is None:
    _here = Path(__file__).parent
    collect_ignore_glob.extend(p.name for p in _here.glob("test_*.py"))
