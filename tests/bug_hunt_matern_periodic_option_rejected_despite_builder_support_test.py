"""Bug hunt: ``matern(..., periodic=...)`` is rejected as an unknown option even
though the Matern builder fully supports periodicity — the squash-merge that
advertised the feature wired the spec but forgot the option whitelist.

Commit c8c3192fa ("feat(#580): periodic period derivation for radial builders —
boolean periodic= (scalar + per-axis list) on duchon/tps/matern") added periodic
support to the radial builders. In ``crates/gam-terms/src/term_builder.rs`` the
``matern`` arm DOES thread it into the basis spec:

    Ok(SmoothBasisSpec::Matern {
        feature_cols: cols.to_vec(),
        spec: MaternBasisSpec {
            center_strategy,
            periodic: parse_periodic_axes_option(options, cols.len())?,   // <- wired
            ...

and the Matern kernel builder consumes it (``expand_periodic_centers`` in
``crates/gam-terms/src/basis/matern_kernel.rs``). But the matern arm's
``validate_known_options("matern", options, &[...])` whitelist (term_builder.rs
~2718-2742) lists only ``nu``/``length_scale``/``centers``/``k``/``knots``/... —
it omits ``periodic``/``cyclic``/``period``/``period_start``/``period_end``. The
sibling ``duchon`` arm's whitelist (~2826-2858) DOES include all of them, which
is why ``duchon(x, periodic=true)`` fits and ``matern(x, periodic=true)`` does
not.

So the option is rejected before the (working) builder ever runs:

    InvalidConfigurationError: matern() does not accept option `periodic`.
    Valid options: [__by_col, __secondary_center_cap, basis-dim, basis_dim, ...]

Observed: every spelling — ``matern(x, periodic=true)``,
``matern(x, z, periodic=c(1,1))``, with or without an explicit ``period=`` — is
rejected as an unknown option, in 1-D and 2-D alike. Expected: the same periodic
Matern smooth the engine already builds when called through
``gam::fit_from_formula`` with the option accepted (verified directly in Rust:
adding the five keys to the whitelist makes ``matern(x, periodic=true,
period=2*pi)`` fit a finite 150-coefficient periodic Matern smooth).

This test fits a clean periodic signal on ``[0, 2*pi)`` with a periodic Matern
smooth and asserts (a) the call is accepted (the bug raises here), (b) the
predictions are finite, and (c) the fit is non-degenerate — it tracks the
periodic signal rather than collapsing to a flat line. It fails today at the
``fit`` call. Adding the missing keys to the matern whitelist makes it pass with
no further edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

TWO_PI = 2.0 * np.pi


def test_matern_periodic_smooth_is_accepted_and_fits() -> None:
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0.0, TWO_PI, n)
    f = np.sin(x) + 0.5 * np.cos(2.0 * x)
    y = f + rng.normal(0.0, 0.15, n)
    df = pd.DataFrame({"x": x, "y": y})

    # The bug: this raises InvalidConfigurationError("matern() does not accept
    # option `periodic`") today, even though the Matern builder threads and uses
    # `periodic` and the sibling duchon() accepts the very same options.
    model = gamfit.fit(
        df,
        "y ~ matern(x, periodic=true, period=6.283185307179586)",
        family="gaussian",
    )

    grid = np.linspace(0.1, TWO_PI - 0.1, 24)
    preds = np.asarray(model.predict(pd.DataFrame({"x": grid}))).ravel()
    truth = np.sin(grid) + 0.5 * np.cos(2.0 * grid)

    assert np.all(np.isfinite(preds)), f"periodic Matern predictions non-finite: {preds}"
    # Non-degenerate: a real periodic fit tracks the signal (truth std ~0.8),
    # not a collapsed flat line.
    assert preds.std() > 0.3, f"periodic Matern fit looks flat (std={preds.std():.3f})"
    corr = float(np.corrcoef(preds, truth)[0, 1])
    assert corr > 0.6, f"periodic Matern fit does not track the periodic signal (corr={corr:.3f})"
