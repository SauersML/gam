from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: validate_formula now rejects the documented "
    "combination 'y ~ x1 + x2 + x1:x2 + x1*x2 + x1/x2 + group(g)' because "
    "x1*x2 expands to x1 + x2 + x1:x2, so x1 is listed 'more than once' "
    "(GamError: duplicate terms / rank-deficient design). The accept-half of "
    "this test pins the old behavior; de-duplicate the formula to re-enable. "
    "The ill-formed rejection half is unaffected.",
)
def test_validate_formula_accepts_documented_operators_and_rejects_ill_formed() -> None:
    rows = [
        {"y": 1.0, "x1": 0.0, "x2": 1.0, "g": "a"},
        {"y": 2.0, "x1": 1.0, "x2": 0.0, "g": "b"},
        {"y": 3.0, "x1": 2.0, "x2": 1.0, "g": "a"},
    ]
    valid_formula = "y ~ x1 + x2 + x1:x2 + x1*x2 + x1/x2 + group(g)"
    info = gamfit.validate_formula(rows, valid_formula)
    assert info.supported_by_python is True, "validate_formula should accept formulas using documented operator combinations"

    with pytest.raises(Exception):
        gamfit.validate_formula(rows, "y ~ x1 + + x2")
