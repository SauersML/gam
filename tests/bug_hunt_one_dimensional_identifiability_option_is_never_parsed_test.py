"""Bug hunt: ``identifiability=`` on the 1-D ``s()`` path is never parsed. Every
value -- including obvious nonsense -- is accepted and has no effect, while every
sibling smooth kind honours it and rejects an invalid token loudly.

``docs/formulas.md:155-156``:

    The 1-D B-spline path accepts these options plus `periodic`, `period`,
    `periods`, `period_start`, `period_end`, `origin`, `identifiability`.

Measured (n=250, ``y = sin(2*pi*x) + 0.5*z + N(0, 0.25)``), ``deviance`` and
coefficient count for each smooth kind under ``identifiability=``:

    s(x, k=8)                  default 19.056203092 / 8    none 19.056203092 / 8   <- unchanged
                               sum_tozero 19.056203092 / 8  centered 19.056203092 / 8
                               totally_bogus 19.056203092 / 8  <- accepted!
    te(x, z, k=5)              default 14.8901634681 / 25   none 14.8902269334 / 26
                               totally_bogus -> InvalidConfigurationError
    matern(x, z, centers=12)   default 17.920984918 / 12    none 16.4488998173 / 13
                               totally_bogus -> InvalidConfigurationError
    thinplate(x, z, ...)       default 23.2530695654 / 12   none 23.2499534845 / 13
                               totally_bogus -> InvalidConfigurationError
    duchon(x, z, ...)          totally_bogus -> InvalidConfigurationError

``identifiability='none'`` means "keep unconstrained basis columns"
(``crates/gam-terms/src/basis/types.rs:338-360``), so it must drop the
sum-to-zero centering and return the coefficient it removes -- exactly the
25 -> 26 and 12 -> 13 jumps the tensor and radial smooths show. The 1-D smooth
stays at 8.

Mechanism (``crates/gam-terms/src/term_builder.rs``): the
``"bspline" | "ps" | "p-spline" | "cr" | "cs"`` arm whitelists
``"identifiability"`` in ``validate_known_options`` (~line 2417) -- which is why
the bogus value does not trip the unknown-option refusal -- and then decides the
policy without ever reading the option (~lines 2477-2481):

    let identifiability = if boundary_conditions.has_anchor() {
        BSplineIdentifiability::None
    } else {
        BSplineIdentifiability::default()      // WeightedSumToZero
    };

The cyclic arm hardcodes ``BSplineIdentifiability::default()`` the same way
(~line 2346). There is no ``parse_bspline_identifiability`` at all: the three
parsers that exist are ``parse_tensor_identifiability`` (~line 1633),
``parse_matern_identifiability`` (~line 5047) and
``parse_spatial_identifiability`` (~line 5067), each of which validates its
token and errors on anything else -- which is precisely the behaviour the 1-D
path is missing.

Observed: a documented 1-D option that does nothing and validates nothing.

Expected: ``s(x, identifiability='none')`` drops the centering constraint (the
coefficient count rises by one, as it does for ``te``/``matern``/``thinplate``),
and an invalid token is rejected the way the siblings reject it.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, 250)
    z = rng.uniform(0.0, 1.0, 250)
    return {
        "x": x,
        "z": z,
        "y": np.sin(2.0 * np.pi * x) + 0.5 * z + 0.25 * rng.standard_normal(x.size),
    }


def _fit(formula: str) -> tuple[float, int]:
    summary = gamfit.fit(_data(), formula, family="gaussian").summary()
    return float(summary.deviance), len(summary.coefficients)


@pytest.mark.parametrize(
    "term", ["te(x, z, k=5)", "matern(x, z, centers=12)", "thinplate(x, z, centers=12)"]
)
def test_control_siblings_honour_identifiability_none(term: str) -> None:
    """Green today: `none` drops the centering constraint and returns a column."""
    base_dev, base_p = _fit(f"y ~ {term}")
    none_dev, none_p = _fit(f"y ~ {term[:-1]}, identifiability='none')")
    assert none_p == base_p + 1, f"{term}: 'none' did not restore a coefficient"
    assert none_dev != base_dev


@pytest.mark.parametrize(
    "term",
    [
        "te(x, z, k=5)",
        "matern(x, z, centers=12)",
        "thinplate(x, z, centers=12)",
        "duchon(x, z, centers=12)",
    ],
)
def test_control_siblings_reject_an_invalid_identifiability_token(term: str) -> None:
    """Green today: an unknown token raises for every multi-d smooth."""
    with pytest.raises(gamfit.GamfitError):
        _fit(f"y ~ {term[:-1]}, identifiability='totally_bogus')")


def test_one_dimensional_smooth_honours_identifiability_none() -> None:
    base_dev, base_p = _fit("y ~ s(x, k=8)")
    none_dev, none_p = _fit("y ~ s(x, k=8, identifiability='none')")
    assert (none_dev, none_p) != (base_dev, base_p), (
        "s(x, k=8, identifiability='none') is bit-identical to the centered "
        f"default: deviance {none_dev!r}, {none_p} coefficients"
    )
    assert none_p == base_p + 1, (
        "dropping the sum-to-zero constraint must restore the coefficient it "
        f"removes: got {none_p}, expected {base_p + 1}"
    )


def test_one_dimensional_smooth_rejects_an_invalid_identifiability_token() -> None:
    with pytest.raises(gamfit.GamfitError):
        _fit("y ~ s(x, k=8, identifiability='totally_bogus')")


def test_cyclic_smooth_honours_identifiability_none() -> None:
    """The cyclic arm hardcodes the default policy the same way."""
    base_dev, base_p = _fit("y ~ cyclic(x, k=8, period=1)")
    none_dev, none_p = _fit("y ~ cyclic(x, k=8, period=1, identifiability='none')")
    assert (none_dev, none_p) != (base_dev, base_p), (
        "cyclic(x, identifiability='none') is bit-identical to the centered default"
    )
