from __future__ import annotations

import importlib
import pathlib
import tempfile
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def test_likelihood_spec_round_trip_preserves_fields() -> None:
    rows = [{"y": float(i + 1), "x": float(i)} for i in range(10)]
    model = gamfit.fit(rows, "y ~ x")
    original_summary = model.summary()
    original_family = original_summary.family_name
    original_formula = original_summary.formula
    original_dict = original_summary.to_dict()

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "roundtrip.gam"
        model.save(path)
        loaded = gamfit.load(path)

    loaded_summary = loaded.summary()
    assert loaded_summary.family_name == original_family, (
        "Likelihood family should be identical after a save/load round-trip"
    )
    assert loaded_summary.formula == original_formula, (
        "Formula should be identical after a save/load round-trip"
    )
    assert loaded_summary.to_dict() == original_dict, (
        "Summary fields should be identical after a save/load round-trip"
    )
