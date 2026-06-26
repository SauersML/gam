"""Regression for issue #1565 — the ``smooths={}`` descriptor bridge.

Two distinct bugs in ``gamfit/smooth.py`` are locked down here at the pure
descriptor level (no compiled Rust extension required to exercise the Python
serialization logic):

Bug 1 — ``Smooth.to_rust_descriptor()`` crashed for *every* subclass.
    The base ``Smooth`` and each concrete subclass are ``@dataclass(slots=True)``.
    ``slots=True`` makes the dataclass decorator recreate the class object, so a
    method's implicit ``__class__`` cell no longer matches the live class and a
    zero-arg ``super()`` raises ``TypeError: super(type, obj): obj must be an
    instance or subtype of type``. The fix uses the explicit two-arg
    ``super(ClassName, self)`` form, which resolves ``ClassName`` from the module
    global (the recreated live class) at call time.

Bug 2 — ``double_penalty=False`` was silently dropped.
    The base only emitted the key when truthy, so a user passing
    ``double_penalty=False`` produced a descriptor byte-identical to the default
    and could never request a single-penalty smooth. The fix emits the key for
    both ``True`` and ``False`` (guarded by ``is not None`` so the tri-state
    ``MeasureJet`` Optional override keeps "None means defer to engine").
"""

from __future__ import annotations

import numpy as np

from gamfit.smooth import (
    BSpline,
    Categorical,
    Duchon,
    Matern,
    MeasureJet,
    Pca,
    PeriodicSplineCurve,
    Sphere,
    TensorBSpline,
)


def _affected_instances():
    """One minimal valid instance per subclass fixed for bug 1."""
    centers = np.zeros((4, 2))
    return {
        "bspline": BSpline(),
        "duchon": Duchon(),
        "tensor_bspline": TensorBSpline(marginals=[BSpline()]),
        "matern": Matern(centers=centers),
        "measurejet": MeasureJet(),
        "pca": Pca(),
        "sphere": Sphere(),
        "periodic_spline_curve": PeriodicSplineCurve(),
        "categorical": Categorical(),
    }


def test_every_subclass_descriptor_does_not_raise():
    # Bug 1: zero-arg super() under @dataclass(slots=True) used to raise
    # TypeError for every subclass. Each must now serialize cleanly with its
    # canonical `kind` discriminator.
    for expected_kind, obj in _affected_instances().items():
        out = obj.to_rust_descriptor()
        assert isinstance(out, dict)
        assert out["kind"] == expected_kind


def test_bspline_descriptor_kind():
    out = BSpline().to_rust_descriptor()
    assert out["kind"] == "bspline"


def test_double_penalty_true_is_transmitted():
    out = BSpline(double_penalty=True).to_rust_descriptor()
    assert out["double_penalty"] is True


def test_double_penalty_false_is_transmitted():
    # Bug 2: `double_penalty=False` used to emit NO key, so the flag could not
    # be toggled. It must now travel as an explicit `false`.
    out = BSpline(double_penalty=False).to_rust_descriptor()
    assert out["double_penalty"] is False


def test_double_penalty_true_and_false_descriptors_differ():
    # The core of bug 2: the two descriptors must no longer be byte-identical.
    true_desc = BSpline(double_penalty=True).to_rust_descriptor()
    false_desc = BSpline(double_penalty=False).to_rust_descriptor()
    assert true_desc != false_desc


def test_measurejet_default_omits_double_penalty_key():
    # MeasureJet redefines `double_penalty: bool | None = None`; None means
    # "defer to the engine default", so NO key may be emitted. The bug-2 fix
    # must not regress this tri-state contract.
    out = MeasureJet().to_rust_descriptor()
    assert "double_penalty" not in out


def test_measurejet_explicit_double_penalty_is_transmitted():
    assert MeasureJet(double_penalty=False).to_rust_descriptor()["double_penalty"] is False
    assert MeasureJet(double_penalty=True).to_rust_descriptor()["double_penalty"] is True
