"""#2782 regression, approached through the margin's realized function space.

``tests/bug_hunt_tensor_degree_and_penalty_order_are_inert_on_default_margins_test.py``
pins the symptom -- ``te(x, z, k=5, degree=1)`` came back bit-identical to
``te(x, z, k=5)``.  "The fit changed" is necessary but not sufficient: an
implementation could satisfy it by perturbing the margin in any way at all.

What must actually hold is that the margin is realized as the basis the caller
asked for.  So this file asserts

* ``degree=1`` makes the surface PIECEWISE LINEAR along that margin (its second
  difference on a fine grid collapses to a few knot spikes), while the default
  cubic margin curves everywhere;
* naming the DEFAULTS (``degree=3``, ``penalty_order=2``) is bit-identical to
  not naming them -- the invariant that makes "route off the cr margin when the
  request differs" safe, and the thing a naive "any explicit request leaves cr"
  rule would break;
* a per-margin list moves only the margin it names;
* an explicit ``knot_placement`` is no longer collapsed onto "unset";

and that the malformed per-margin forms are refused rather than falling back to
the default the way ``option_usize`` used to.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(2782)
    x = rng.uniform(0.0, 1.0, 500)
    z = rng.uniform(0.0, 1.0, x.size)
    return {
        "x": x,
        "z": z,
        "y": np.sin(2.0 * np.pi * x) + 0.7 * z * z + 0.3 * rng.standard_normal(x.size),
    }


def _fit(formula: str) -> Any:
    return gamfit.fit(_data(), formula, family="gaussian")


def _signature(formula: str) -> tuple[float, float, tuple[float, ...]]:
    model = _fit(formula)
    summary = model.summary()
    grid = {
        "x": np.linspace(0.05, 0.95, 11),
        "z": np.full(11, 0.5),
    }
    predictions = np.asarray(
        model.predict(grid, return_type="dict")["posterior_mean"], dtype=float
    )
    return float(summary.deviance), float(summary.edf_total), tuple(predictions.tolist())


def _curvature_along_x(formula: str, n: int = 401) -> Any:
    """Second difference of the fitted surface along x at fixed z."""
    model = _fit(formula)
    x = np.linspace(0.02, 0.98, n)
    frame = {"x": x, "z": np.full(n, 0.5)}
    f = np.asarray(
        model.predict(frame, return_type="dict")["posterior_mean"], dtype=float
    )
    return np.abs(np.diff(f, n=2))


def test_degree_one_margin_is_piecewise_linear_along_that_axis() -> None:
    """A linear margin bends only at its knots; a cubic one bends everywhere."""
    linear = _curvature_along_x("y ~ te(x, z, k=5, degree=1)")
    cubic = _curvature_along_x("y ~ te(x, z, k=5)")

    # Normalize by each surface's own bending scale so the comparison is about
    # WHERE the curvature lives, not how wiggly the two fits happen to be.
    linear_share = float(np.mean(linear > 0.02 * linear.max()))
    cubic_share = float(np.mean(cubic > 0.02 * cubic.max()))
    assert linear_share < 0.1, (
        "a degree=1 margin must be piecewise linear along x, but "
        f"{linear_share:.0%} of the grid carries curvature"
    )
    assert cubic_share > 0.8, (
        "the default cubic margin must curve almost everywhere, but only "
        f"{cubic_share:.0%} of the grid does"
    )


@pytest.mark.parametrize("kind", ["te", "ti"])
@pytest.mark.parametrize(
    "option", ["degree=3", "penalty_order=2", "degree=3, penalty_order=2"]
)
def test_naming_the_default_is_a_no_op(kind: str, option: str) -> None:
    """Requesting what the default margin already IS must not change the fit.

    This is the load-bearing half of the routing rule. `degree` and
    `penalty_order` move a margin off the natural cubic regression basis only
    when the requested value differs from what a cr margin is (cubic,
    second-order); a rule keyed on "was the option present" instead would make
    the option change the fit merely by being mentioned.
    """
    assert _signature(f"y ~ {kind}(x, z, k=5, {option})") == _signature(
        f"y ~ {kind}(x, z, k=5)"
    )


@pytest.mark.parametrize("kind", ["te", "ti"])
@pytest.mark.parametrize(
    "option",
    ["degree=1", "degree=2", "degree=4", "penalty_order=1", "penalty_order=3"],
)
def test_a_non_default_request_changes_the_fit(kind: str, option: str) -> None:
    assert _signature(f"y ~ {kind}(x, z, k=5, {option})") != _signature(
        f"y ~ {kind}(x, z, k=5)"
    )


def test_per_margin_list_moves_only_the_margin_it_names() -> None:
    base = _signature("y ~ te(x, z, k=5)")
    both_linear = _signature("y ~ te(x, z, k=5, degree=[1,1])")
    x_linear = _signature("y ~ te(x, z, k=5, degree=[1,3])")
    z_linear = _signature("y ~ te(x, z, k=5, degree=[3,1])")

    assert _signature("y ~ te(x, z, k=5, degree=[3,3])") == base
    assert both_linear == _signature("y ~ te(x, z, k=5, degree=1)")
    for label, variant in (("x", x_linear), ("z", z_linear)):
        assert variant != base, f"degree=1 on the {label} margin must change the fit"
        assert variant != both_linear, (
            f"only the {label} margin was asked for degree=1, but the fit matches "
            "the both-linear one"
        )
    assert x_linear != z_linear


def test_explicit_knot_placement_is_not_collapsed_onto_unset() -> None:
    base = _signature("y ~ te(x, z, k=5)")
    assert _signature("y ~ te(x, z, k=5, knot_placement='uniform')") != base
    assert _signature("y ~ te(x, z, k=5, knot_placement='quantile')") != base


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("degree=[1,2,3]", "3 entries"),
        ("penalty_order=[1,2,3]", "3 entries"),
        ("degree=[1,banana]", "banana"),
        ("bc=['periodic','natural','free']", "3 entries"),
    ],
)
def test_malformed_per_margin_requests_are_refused(option: str, expected: str) -> None:
    """They used to fall back to the default instead of being refused."""
    with pytest.raises(gamfit.errors.GamfitError) as excinfo:
        _fit(f"y ~ te(x, z, k=5, {option})")
    assert expected in str(excinfo.value), f"got: {excinfo.value}"


def test_scalar_boundary_token_reaches_every_margin() -> None:
    """`bc='periodic'` used to be dropped by a length guard, silently building
    an APERIODIC tensor. It now broadcasts, so it needs a period like any other
    periodic margin -- and is refused, loudly, when none is given."""
    with pytest.raises(gamfit.errors.GamfitError) as excinfo:
        _fit("y ~ te(x, z, k=5, bc='periodic')")
    assert "periodic" in str(excinfo.value)

    # ...and a scalar non-periodic token stays a legitimate no-op.
    assert _signature("y ~ te(x, z, k=5, bc='natural')") == _signature(
        "y ~ te(x, z, k=5)"
    )
