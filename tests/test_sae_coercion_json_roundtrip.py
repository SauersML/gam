"""#2091 — pin the Rust `py_any_to_json_value` report-block coercion.

The `from_fit_payload` builder (a later increment) carries the raw fit payload's
opaque report blocks (`diagnostics`, `atom_inference`, `hybrid_split`, …) through
to `serde_json::Value` with `py_any_to_json_value`, mirroring the Python
`_jsonable` coercion `to_dict` applies. This test drives that helper from Python
via the `sae_coercion_json_roundtrip` round-trip (py -> Value -> py) and asserts
it reproduces `_jsonable`: numpy arrays / scalars flatten via `.tolist()`, dict
keys stringify, and integral values stay JSON integers (not floats)."""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import rust_module  # noqa: E402


def _roundtrip(obj):
    fn = getattr(rust_module(), "sae_coercion_json_roundtrip", None)
    if fn is None:
        pytest.skip("wheel predates sae_coercion_json_roundtrip (#2091 coercion slice)")
    return fn(obj)


def _jsonable(value):
    """The exact `to_dict._jsonable` reference."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@pytest.mark.parametrize(
    "obj",
    [
        None,
        True,
        False,
        3,
        3.0,
        -2.5,
        "hello",
        [1, 2, 3],
        [1.0, 2.0, 3.0],
        (4, 5, 6),
        {"a": 1, "b": [1.0, 2.0], "c": {"nested": "x"}},
        {"arr": np.array([1.0, 2.0, 3.0]), "flag": True, "n": 7},
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        {"mixed": [np.array([1.0, 2.0]), {"k": np.float64(0.5)}]},
    ],
)
def test_json_roundtrip_matches_jsonable(obj):
    got = _roundtrip(obj)
    expected = _jsonable(obj)
    assert got == expected, f"{got!r} != {expected!r}"


def test_json_roundtrip_preserves_int_vs_float():
    # An integer must stay an int (not become 3.0) — the builder relies on this
    # to reproduce json.dumps's int/float distinction.
    got = _roundtrip({"i": 3, "f": 3.0})
    assert isinstance(got["i"], int) and not isinstance(got["i"], bool)
    assert isinstance(got["f"], float)


def test_json_roundtrip_numpy_int_scalar_is_int():
    got = _roundtrip(np.int64(5))
    assert got == 5 and isinstance(got, int)
