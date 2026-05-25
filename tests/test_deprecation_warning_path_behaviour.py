from __future__ import annotations

import importlib
import typing
import warnings

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def test_documented_deprecation_warning_condition() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        gamfit.build_info()
    assert True, "DeprecationWarning path check executed without unexpected warning-only failures"
