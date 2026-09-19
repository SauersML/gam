"""Bug hunt: declaring a period on a smooth is a silent no-op unless the
separate ``periodic=`` / ``bc='periodic'`` switch is ALSO present. The fit comes
back bit-identical to one with no period declaration at all -- no error, no
warning, no inference note.

    y ~ s(t)                                    deviance 105.929061   |f(0)-f(24)| = 1.98e-2
    y ~ s(t, period=24)                         deviance 105.929061   |f(0)-f(24)| = 1.98e-2
    y ~ s(t, periods=24)                        deviance 105.929061   |f(0)-f(24)| = 1.98e-2
    y ~ s(t, period_start=0, period_end=24)     deviance 105.929061   |f(0)-f(24)| = 1.98e-2
    y ~ s(t, origin=0, period=24)               deviance 105.929061   |f(0)-f(24)| = 1.98e-2
    y ~ s(t, bc='periodic', period=24)          deviance 105.586532   |f(0)-f(24)| = 0
    y ~ s(t, periodic=true, period=24)          deviance 105.586532   |f(0)-f(24)| = 0

(hour-of-day data, n=1200, y = 2 sin(2*pi*t/24) + cos(4*pi*t/24) + N(0, 0.3)).
The inert rows are equal to the plain fit to every printed digit, so the
declaration has no effect whatsoever. The same holds on the tensor path:
``te(theta, h, periods=[2*pi, None])`` and ``te(theta, h, period=[2*pi, None])``
are both bit-identical to ``te(theta, h)``, while
``te(theta, h, periodic=[true,false], period=[2*pi,None])`` and
``te(theta, h, bc=['periodic','natural'], period=[2*pi, None])`` wrap exactly.

The docs list all of these as options of these paths --

    docs/formulas.md:155-156
        The 1-D B-spline path accepts these options plus `periodic`, `period`,
        `periods`, `period_start`, `period_end`, `origin`, `identifiability`.

    docs/formulas.md:366 (tensor)
        | `periodic`, `period`, `periods`, `origin`, `origins` | Per-margin
          periodicity (see below). |

-- and promise the opposite of silence:

    docs/formulas.md:167-168
        An unparseable endpoint or an unknown option is rejected rather than
        silently dropped.

An unknown option IS rejected (``s(t, nonsense_option=3)`` -> "bspline() does not
accept option `nonsense_option`"). A KNOWN-but-inert one is not.

Mechanism (``crates/gam-terms/src/term_builder.rs``): in the
``"bspline" | "ps" | "p-spline" | "cr" | "cs"`` arm, ``validate_known_options``
whitelists ``periodic``/``period``/``periods``/``period_start``/``period_end``/
``origin`` (~lines 2407-2413), but the switch is
``let periodic_axes = parse_periodic_axes(options, 1)`` (~line 2434) and

    fn parse_periodic_axes(options, dim) -> Result<Vec<bool>, String> {      // ~line 1188
        let mut axes = vec![false; dim];
        if let Some(raw) = options.get("periodic").or_else(|| options.get("cyclic")) { ... }

reads ONLY ``periodic`` / ``cyclic``. Every period value is then parsed *inside*
``if periodic_axes[0] { ... }`` (~line 2485), so with the flag absent the branch
is skipped and the declarations are dropped on the floor. The
``"cc" | "cp" | "cyclic"`` arm (~line 2300) hardcodes ``periodic_axes = [true]``,
which is exactly why ``cyclic(t, period=24)`` works and ``s(t, period=24)`` does
not.

Observed: a period declaration without ``periodic=``/``bc='periodic'`` is
accepted and discarded; the user gets an aperiodic smooth with a discontinuity
at the seam and no indication anything was ignored.

Expected: the declaration is honoured (a period is not a meaningful property of
an aperiodic basis, and the tensor docs call the whole family "per-margin
periodicity") -- or, failing that, rejected. This file asserts the disjunction,
so either resolution turns it green; what it forbids is the silent no-op.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_PERIOD = 24.0
_SEAM = {"t": np.array([0.0, _PERIOD])}


def _hourly() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    t = rng.uniform(0.0, _PERIOD, 1200)
    y = (
        2.0 * np.sin(2.0 * np.pi * t / _PERIOD)
        + 1.0 * np.cos(4.0 * np.pi * t / _PERIOD)
        + 0.3 * rng.standard_normal(t.size)
    )
    return {"t": t, "y": y}


def _seam_gap(model: Any) -> float:
    p = np.asarray(
        model.predict(_SEAM, return_type="dict")["posterior_mean"], dtype=float
    )
    return abs(float(p[0] - p[1]))


def _plain() -> tuple[float, float]:
    m = gamfit.fit(_hourly(), "y ~ s(t)", family="gaussian")
    return float(m.summary().deviance), _seam_gap(m)


def test_reference_periodic_and_plain_fits_differ() -> None:
    """Premise check: the switch-carrying spelling really does wrap."""
    plain_dev, plain_gap = _plain()
    cyclic = gamfit.fit(
        _hourly(), "y ~ s(t, periodic=true, period=24)", family="gaussian"
    )
    assert _seam_gap(cyclic) < 1e-9
    assert plain_gap > 1e-3
    assert abs(float(cyclic.summary().deviance) - plain_dev) > 1e-6


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ s(t, period=24)",
        "y ~ s(t, periods=24)",
        "y ~ s(t, period_start=0, period_end=24)",
        "y ~ s(t, origin=0, period=24)",
    ],
)
def test_one_dimensional_period_declaration_is_not_a_silent_no_op(formula: str) -> None:
    plain_dev, _ = _plain()
    try:
        model = gamfit.fit(_hourly(), formula, family="gaussian")
    except gamfit.GamfitError:
        return  # rejected rather than silently dropped: acceptable resolution

    deviance = float(model.summary().deviance)
    assert not (
        abs(deviance - plain_dev) < 1e-9 and _seam_gap(model) > 1e-3
    ), (
        f"{formula} was accepted but produced the plain aperiodic fit "
        f"(deviance {deviance!r} vs plain {plain_dev!r}, "
        f"seam gap {_seam_gap(model):.3e})"
    )
    assert _seam_gap(model) < 1e-9, (
        f"{formula} declared a period of 24 but f(0) != f(24) "
        f"(gap {_seam_gap(model):.3e})"
    )


def _cylinder() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    theta = rng.uniform(0.0, 2.0 * np.pi, 800)
    h = rng.uniform(0.0, 1.0, theta.size)
    return {
        "theta": theta,
        "h": h,
        "y": np.sin(theta) + 0.5 * h + 0.2 * rng.standard_normal(theta.size),
    }


def _tensor_seam_gap(model: Any) -> float:
    frame = {"theta": np.array([0.0, 2.0 * np.pi]), "h": np.array([0.5, 0.5])}
    p = np.asarray(
        model.predict(frame, return_type="dict")["posterior_mean"], dtype=float
    )
    return abs(float(p[0] - p[1]))


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ te(theta, h, periods=[2*pi, None])",
        "y ~ te(theta, h, period=[2*pi, None])",
        "y ~ te(theta, h, origins=[0, None], periods=[2*pi, None])",
    ],
)
def test_tensor_period_declaration_is_not_a_silent_no_op(formula: str) -> None:
    data = _cylinder()
    plain = gamfit.fit(data, "y ~ te(theta, h)", family="gaussian")
    plain_dev = float(plain.summary().deviance)

    try:
        model = gamfit.fit(data, formula, family="gaussian")
    except gamfit.GamfitError:
        return  # rejected rather than silently dropped: acceptable resolution

    deviance = float(model.summary().deviance)
    assert not (
        abs(deviance - plain_dev) < 1e-9 and _tensor_seam_gap(model) > 1e-4
    ), (
        f"{formula} was accepted but produced the plain aperiodic tensor fit "
        f"(deviance {deviance!r}, seam gap {_tensor_seam_gap(model):.3e})"
    )
    assert _tensor_seam_gap(model) < 1e-9, (
        f"{formula} declared a 2*pi period on the theta margin but "
        f"f(0, 0.5) != f(2*pi, 0.5) (gap {_tensor_seam_gap(model):.3e})"
    )
