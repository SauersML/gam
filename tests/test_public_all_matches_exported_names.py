from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def test___all___matches_real_exports() -> None:
    exported = set(gamfit.__all__)
    missing = sorted(name for name in exported if not hasattr(gamfit, name))
    assert not missing, f"__all__ should not contain names that are not actually exported: {missing}"
