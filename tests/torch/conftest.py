"""Skip the gamfit.torch test suite when the torch extra is not installed."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Protocol, cast


class _PytestModule(Protocol):
    def importorskip(self, modname: str) -> ModuleType: ...


pytest = cast(_PytestModule, import_module("pytest"))

pytest.importorskip("torch")
