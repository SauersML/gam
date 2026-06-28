from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def test_validate_formula_accepts_documented_operators_and_rejects_ill_formed() -> None:
    rows = [
        {"y": 1.0, "x1": 0.0, "x2": 1.0, "x3": 0.5, "g": "a"},
        {"y": 2.0, "x1": 1.0, "x2": 0.0, "x3": 0.25, "g": "b"},
        {"y": 3.0, "x1": 2.0, "x2": 1.0, "x3": 0.75, "g": "a"},
    ]
    # Non-redundant use of the documented `*` (crossing) and `/` (nesting)
    # operators plus group(). The literal `x1*x2 + x1/x2` overlap double-lists
    # x1/x1:x2 and is now correctly rejected as rank-deficient, so this exercises
    # the same operator family without the duplicate-term collision.
    valid_formula = "y ~ x3*x2 + x1/x2 + group(g)"
    info = gamfit.validate_formula(rows, valid_formula)
    assert info.supported_by_python is True, "validate_formula should accept formulas using documented operator combinations"

    with pytest.raises(Exception):
        gamfit.validate_formula(rows, "y ~ x1 + + x2")
