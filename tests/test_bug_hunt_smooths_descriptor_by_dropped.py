"""Regression for issue #1160.

`Smooth(by=...)` (and the topology constructors that thread `by` onto a
`Smooth`) used to be silently dropped on the formula ``smooths={}`` descriptor
path: ``to_rust_descriptor`` never serialized ``by`` and the Rust
descriptor-merge in ``smooth_overrides.rs`` had no consumer for it. A user
writing ``fit(df, "y ~ s(x)", smooths={"x": Duchon(by="g")})`` got a fit as if
``by=None`` with no error or warning.

The fix wires ``by`` end-to-end on the descriptor path: on this path ``by`` is
the *name* of a data-frame column (the gating variable), serialized by
``to_rust_descriptor`` and resolved to a ``by_col`` in ``smooth_overrides.rs``,
which then wraps the inner basis in the ``SmoothBasisSpec::ByVariable``
envelope — bit-identical to what the formula ``s(x, by=g)`` syntax produces.

A raw per-row ``by`` *array* has no data frame to name on this path (that is the
contract of the primitive numpy API), so it is rejected loudly at
``_normalize_smooths`` time with a pointer.

This test exercises the pure-Python normalization + serialization layer, so it
does not require the compiled extension. The Rust ``ByVariable``-envelope wiring
is covered by ``by_column_name_wraps_in_by_variable_envelope`` in
``src/terms/smooth_overrides.rs``.
"""

import numpy as np
import pytest

from gamfit._api import _normalize_smooths
from gamfit.smooth import BSpline, Duchon, Matern


@pytest.mark.parametrize(
    "descriptor",
    [
        Duchon(by="g"),
        Matern(by="g"),
        BSpline(by="g"),
    ],
)
def test_by_column_name_survives_to_rust_descriptor(descriptor):
    # A column-name `by` is serialized through to the Rust merge payload.
    out = _normalize_smooths({"x": descriptor})
    assert out["x"]["by"] == "g"
    assert out["x"]["vars"] == ["x"]


@pytest.mark.parametrize(
    "descriptor",
    [
        Duchon(by=np.ones(8)),
        Matern(by=np.ones(8)),
        BSpline(by=np.arange(8.0)),
    ],
)
def test_by_array_on_smooths_descriptor_is_rejected_not_dropped(descriptor):
    with pytest.raises(ValueError) as excinfo:
        _normalize_smooths({"x": descriptor})
    msg = str(excinfo.value)
    assert "by=" in msg
    # Must point users at the primitive numpy API for raw per-row arrays.
    assert "primitive numpy API" in msg


def test_normalize_smooths_without_by_is_unaffected():
    # The control: a descriptor with no `by` still normalizes cleanly and the
    # `by` key never leaks into the serialized payload.
    out = _normalize_smooths({"x": Duchon()})
    assert "x" in out
    assert "by" not in out["x"]
    assert out["x"]["vars"] == ["x"]
