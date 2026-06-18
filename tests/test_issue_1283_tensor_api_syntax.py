"""Issue #1283: Python formula API accepts tensor per-margin syntax."""

from __future__ import annotations

import importlib
import math
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


ROWS = [
    {
        "resp": math.sin(i / 5.0) + 0.2 * (i % 7),
        "x": i / 29.0,
        "y": (i % 11) / 10.0,
        "theta": 2.0 * math.pi * (i % 30) / 30.0,
    }
    for i in range(60)
]


def _assert_supported(formula: str) -> None:
    try:
        validation = gamfit.validate_formula(ROWS, formula)
    except Exception as exc:  # noqa: BLE001 - preserve the mapped Rust error.
        pytest.fail(f"validate_formula({formula!r}) raised {type(exc).__name__}: {exc}")
    assert validation.supported_by_python is True


def test_tensor_accepts_python_tuple_k_syntax() -> None:
    _assert_supported("resp ~ te(x, y, k=(5, 4))")


def test_tensor_accepts_per_axis_k_aliases() -> None:
    _assert_supported("resp ~ te(x, y, k_x=5, k_y=4)")


def test_tensor_accepts_list_periodic_period_and_k_together() -> None:
    _assert_supported(
        "resp ~ te(theta, y, periodic=[0], period=[6.283185307179586, None], k=[5, 4])"
    )
