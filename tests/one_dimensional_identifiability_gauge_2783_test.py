"""#2783 regression, approached from the gauge rather than from the option.

``tests/bug_hunt_one_dimensional_identifiability_option_is_never_parsed_test.py``
pins the *symptom*: ``identifiability=`` used to be whitelisted by
``validate_known_options`` and then never read on the 1-D B-spline path, so
every value (including ``'totally_bogus'``) was accepted and the fit was
bit-identical.  Counting coefficients is enough to catch that, but it would not
catch a regression that reads the token and then applies the wrong chart --
e.g. wiring ``'none'`` to a policy that merely reorders columns, or restoring
the centering somewhere downstream of the builder.

So this file asserts what each policy MEANS about the fitted function:

* ``sum_tozero`` -- the smooth's fitted values have zero (weighted) sample mean,
  which is what makes it non-competitive with the global intercept;
* ``none``       -- they do not, and the level moved into the smooth;
* ``linear``     -- neither the constant nor the linear direction survives, so
  the smooth is orthogonal to ``[1, x]`` and a parametric ``x`` term keeps the
  whole slope.

and that the three structurally unsatisfiable combinations are refused with a
message that names the conflict, rather than being silently resolved one way.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _data(n: int = 300) -> dict[str, Any]:
    rng = np.random.default_rng(2783)
    x = rng.uniform(0.0, 1.0, n)
    # A deliberately off-center, sloped signal: every gauge below has something
    # real to move, so "the constraint did nothing" cannot pass by accident.
    y = 3.0 + 1.5 * x + np.sin(2.0 * np.pi * x) + 0.2 * rng.standard_normal(n)
    return {"x": x, "y": y}


def _smooth_values(model: Any, data: dict[str, Any], term_prefix: str) -> Any:
    """The named smooth term's own contribution at ``data``'s rows.

    Read straight off the fitted affine design (``offset + matrix @ beta``
    reproduces the linear predictor) so the assertion is about the shipped
    design block, not about a convenience accessor that might re-center.
    """
    design = model.design_matrix(data)
    matrix = np.asarray(design.matrix, dtype=float)
    beta = np.asarray(design.coefficients, dtype=float)
    block = next(
        b for b in model.term_blocks if b.name.startswith(term_prefix)
    )
    return matrix[:, block.start : block.end] @ beta[block.start : block.end]


def _fit(formula: str, data: dict[str, Any]) -> Any:
    return gamfit.fit(data, formula, family="gaussian")


def test_default_and_sum_tozero_center_the_smooth() -> None:
    data = _data()
    for formula in ("y ~ s(x, k=10)", "y ~ s(x, k=10, identifiability='sum_tozero')"):
        model = _fit(formula, data)
        values = _smooth_values(model, data, "s(x")
        assert abs(float(values.mean())) < 1e-8, (
            f"{formula}: a centered smooth must have zero sample mean, "
            f"got {values.mean()!r}"
        )


def test_identifiability_none_releases_the_centering_constraint() -> None:
    data = _data()
    centered = _smooth_values(_fit("y ~ s(x, k=10)", data), data, "s(x")
    uncentered = _smooth_values(
        _fit("y ~ s(x, k=10, identifiability='none')", data), data, "s(x"
    )
    assert abs(float(centered.mean())) < 1e-8
    assert abs(float(uncentered.mean())) > 1e-3, (
        "identifiability='none' must keep the constant direction in the smooth, "
        f"but its fitted values still have mean {uncentered.mean()!r}"
    )
    # The two charts describe the same function up to a constant: the model is
    # re-gauged, not re-fit into something else.
    shifted = uncentered - uncentered.mean()
    assert float(np.max(np.abs(shifted - centered))) < 5e-2


def _smooth_block(model: Any, data: dict[str, Any], term_prefix: str) -> Any:
    """The named smooth term's design columns at ``data``'s rows."""
    matrix = np.asarray(model.design_matrix(data).matrix, dtype=float)
    block = next(b for b in model.term_blocks if b.name.startswith(term_prefix))
    return matrix[:, block.start : block.end]


def _residual_fraction(columns: Any, target: Any) -> float:
    """Fraction of ``target``'s energy that ``columns`` cannot represent."""
    fitted = columns @ np.linalg.lstsq(columns, target, rcond=None)[0]
    return float(np.sum((target - fitted) ** 2) / np.sum(target**2))


def test_identifiability_linear_removes_the_affine_directions_from_the_span() -> None:
    """`linear` must leave no nonzero affine function in the smooth's span.

    Asserting this on the SPAN rather than on the fitted values is the honest
    statement of what `RemoveLinearTrend` does. It deletes the constant and
    linear directions of the Greville coefficient chart, which removes every
    affine function from the model space; it does not make the surviving basis
    functions L2-orthogonal to ``x`` at the sample points, and a test that
    demanded that would be pinning a property the policy never had.

    The default gauge is the control: it removes only the constant, so the
    centered spline space still reproduces ``x - mean(x)`` exactly (a cubic
    B-spline span contains every cubic, hence every line).
    """
    data = _data()
    x = data["x"]
    centered_x = x - x.mean()

    default_block = _smooth_block(_fit("y ~ s(x, k=10)", data), data, "s(x")
    linear_block = _smooth_block(
        _fit("y ~ s(x, k=10, identifiability='linear')", data), data, "s(x"
    )

    assert _residual_fraction(default_block, centered_x) < 1e-12, (
        "the centered spline span must still contain the linear function"
    )
    assert _residual_fraction(linear_block, centered_x) > 0.5, (
        "identifiability='linear' must remove the linear direction from the "
        "span, but the block still reproduces x - mean(x) to within "
        f"{_residual_fraction(linear_block, centered_x):.3e} of its energy"
    )
    # The constant is gone from both.
    ones = np.ones_like(x)
    for block, label in ((default_block, "default"), (linear_block, "linear")):
        assert _residual_fraction(block, ones) > 0.5, (
            f"the {label} gauge must not span the constant"
        )

    # And the extra removed direction really costs a coefficient.
    assert (
        len(_fit("y ~ s(x, k=10, identifiability='linear')", data).summary().coefficients)
        + 1
        == len(_fit("y ~ s(x, k=10)", data).summary().coefficients)
    )


def test_cyclic_smooth_gauge_is_read_and_still_wraps_under_none() -> None:
    """The cyclic arm honours the option without giving up its seam."""
    rng = np.random.default_rng(11)
    t = rng.uniform(0.0, 1.0, 400)
    data = {"t": t, "y": 2.0 + np.sin(2.0 * np.pi * t) + 0.2 * rng.standard_normal(t.size)}
    seam = {"t": np.array([0.0, 1.0])}

    centered = _fit("y ~ cyclic(t, k=10, period=1)", data)
    uncentered = _fit("y ~ cyclic(t, k=10, period=1, identifiability='none')", data)

    for model in (centered, uncentered):
        p = np.asarray(
            model.predict(seam, return_type="dict")["posterior_mean"], dtype=float
        )
        assert abs(float(p[0] - p[1])) < 1e-9, "a cyclic smooth must close its seam"

    assert abs(float(_smooth_values(centered, data, "cyclic(t").mean())) < 1e-8
    assert abs(float(_smooth_values(uncentered, data, "cyclic(t").mean())) > 1e-3


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        (
            "y ~ s(x, k=10, bc_left=anchored, anchor_left=0, identifiability='sum_tozero')",
            "anchored endpoint",
        ),
        (
            "y ~ s(x, k=10, periodic=true, period=1, identifiability='linear')",
            "periodic",
        ),
        ("y ~ s(x, k=10, bs='cr', identifiability='linear')", "cr"),
        ("y ~ s(x, k=10, identifiability='totally_bogus')", "totally_bogus"),
        ("y ~ s(x, k=10, identifiability='frozen')", "internal-only"),
    ],
)
def test_unsatisfiable_or_unknown_gauges_are_refused_with_a_reason(
    formula: str, expected: str
) -> None:
    data = _data(120)
    with pytest.raises(gamfit.GamfitError) as excinfo:
        _fit(formula, data)
    assert expected in str(excinfo.value), (
        f"{formula} should be refused with a message naming {expected!r}, "
        f"got: {excinfo.value}"
    )
