"""Regression for issue #1160.

`Smooth(by=...)` (and the topology constructors that thread `by` onto a
`Smooth`) used to be silently dropped on the formula ``smooths={}`` descriptor
path: ``to_rust_descriptor`` never serialized ``by`` and the Rust
descriptor-merge in ``smooth_overrides.rs`` has no consumer for it (it is
governed by ``deny_unknown_fields``). A user writing
``fit(df, "y ~ s(x)", smooths={"x": Duchon(by=w)})`` got a fit as if
``by=None`` with no error or warning.

The fix rejects ``by=`` loudly at ``_normalize_smooths`` time (mirroring how the
same path already rejects the unsupported ``double_penalty`` key), pointing the
user at the formula by-smooth syntax that does wire ``by`` (``s(x, by=g)``).

This test exercises the pure-Python normalization layer, so it does not require
the compiled extension.
"""

import numpy as np
import pytest

from gamfit._api import _normalize_smooths
from gamfit.smooth import BSpline, Duchon, Matern
from gamfit.topology import Circle, Cylinder, EuclideanPatch, Sphere, Torus


@pytest.mark.parametrize(
    "descriptor",
    [
        Duchon(by=np.ones(8)),
        Matern(by=np.ones(8)),
        BSpline(by=np.arange(8.0)),
        Circle(by=np.ones(8)),
        Cylinder(by=np.ones(8)),
        Torus(by=np.ones(8)),
        Sphere(by=np.ones(8)),
        EuclideanPatch(by=np.ones(8)),
    ],
)
def test_by_on_smooths_descriptor_is_rejected_not_dropped(descriptor):
    with pytest.raises(ValueError) as excinfo:
        _normalize_smooths({"x": descriptor})
    msg = str(excinfo.value)
    assert "by=" in msg
    # Must point users at the formula syntax that actually wires `by`.
    assert "s(x, by=g)" in msg


def test_normalize_smooths_without_by_is_unaffected():
    # The control: a descriptor with no `by` still normalizes cleanly and the
    # `by` key never leaks into the serialized payload.
    out = _normalize_smooths({"x": Duchon()})
    assert "x" in out
    assert "by" not in out["x"]
    assert out["x"]["vars"] == ["x"]
