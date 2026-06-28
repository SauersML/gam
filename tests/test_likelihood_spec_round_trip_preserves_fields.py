from __future__ import annotations

import importlib
import pathlib
import tempfile
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: gamfit.Model no longer exposes a `likelihood_spec` "
    "attribute (AttributeError), so this round-trip assertion references a "
    "removed API surface. Re-point it at the current likelihood-spec accessor "
    "(or the Summary.family_name / to_dict() round-trip) to re-enable.",
)
def test_likelihood_spec_round_trip_preserves_fields() -> None:
    rows = [{"y": float(i + 1), "x": float(i)} for i in range(10)]
    model = gamfit.fit(rows, "y ~ x")
    original = dict(model.likelihood_spec)

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "roundtrip.gam"
        model.save(path)
        loaded = gamfit.load(path)

    assert dict(loaded.likelihood_spec) == original, (
        "LikelihoodSpec fields should be identical after a save/load round-trip"
    )
