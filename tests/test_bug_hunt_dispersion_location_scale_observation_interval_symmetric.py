"""Bug hunt: the **dispersion location-scale** (GAMLSS) observation/prediction
interval is a symmetric ``mu ± z·sqrt(SE^2 + sigma^2)`` band — the equal-tailed
skew fix (#817 Gamma / #1193 NegativeBinomial / #1194 Beta) was never propagated
to the two-block dispersion path.

A standard (single-block) Gamma fit routes its observation interval through the
skew-aware ``family_observation_band`` in ``src/inference/predict/mod.rs``: it
builds the edges from equal-tailed quantiles of a moment-matched Gamma
predictive, so on a right-skewed Gamma the upper edge sits well above the mean
and the lower edge well below it (``(hi-mu)/(mu-lo) >> 1``) and each tail covers
near nominal.

The *dispersion location-scale* fit (Gamma/NB/Beta/Tweedie + ``noise_formula``)
instead routes through ``DispersionLocationScalePredictor::observation_noise``
(``src/inference/predict/dispersion_location_scale.rs:226``), which returns
``Some(noise_sd)``. The generic predict drivers then build the band in
``src/inference/predict/interval_policy.rs`` (full-uncertainty arm, lines
~303-317) and ``src/inference/predict/mod.rs`` (posterior-mean arm, lines
~728-743) as a **symmetric** ``mu ± z·sqrt(SE^2 + sigma^2)`` — never calling
``family_observation_band``. The band's own comment even acknowledges it is
applied to "the NB/Gamma/Beta/Tweedie dispersion-LS models".

Symptom (measured, train 6000 / test 30000, right-skewed Gamma, both mean and
shape varying in x):

    standard   upper_tail=0.027  (hi-mu)/(mu-lo)=2.23   <- equal-tailed, correct
    loc-scale  upper_tail=0.052  (hi-mu)/(mu-lo)=1.00   <- symmetric, BUG

i.e. the location-scale upper tail under-covers at ~2x nominal while the band is
*exactly* symmetric about the mean (ratio 1.000), even though the identical data
through the standard Gamma path produces a strongly right-skewed (ratio ~2.2)
band. The lower edge is clamped at the support floor 0 and over-covers, hiding
the defect in the two-sided number — the location-scale sibling of #817.

When the equal-tailed fix is propagated to the dispersion location-scale path,
this test starts passing without edits: the band becomes right-skewed
(``ratio >> 1``) and the upper tail covers near nominal.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit


def _skewed_gamma_gamlss(rng, n):
    """Right-skewed Gamma whose mean AND shape (precision) both vary with x, so a
    dispersion location-scale model has a genuine non-constant scale channel to
    fit. ``Var(Y|mu) = mu^2 / nu`` => coefficient of variation ``1/sqrt(nu)``,
    skewness ``2/sqrt(nu)`` (clearly right-skewed for these nu)."""
    x = rng.uniform(0.0, 1.0, n)
    mu = np.exp(0.6 + 1.0 * np.sin(2.0 * np.pi * x))          # mean ~0.7..5
    nu = np.exp(0.5 + 0.4 * np.cos(2.0 * np.pi * x))          # shape ~1.1..2.5
    y = rng.gamma(nu, mu / nu)                                # scale = mu/nu
    return pd.DataFrame({"x": x, "y": y})


def _band_stats(pred, y):
    lo = pred["observation_lower"].to_numpy()
    hi = pred["observation_upper"].to_numpy()
    mn = pred["mean"].to_numpy()
    two_sided = float(np.mean((y >= lo) & (y <= hi)))
    upper_tail = float(np.mean(y > hi))
    lower_tail = float(np.mean(y < lo))
    # Right-skew of the predictive band: how much further the upper edge sits
    # above the mean than the lower edge sits below it. A genuinely equal-tailed
    # Gamma band has this >> 1; a symmetric mu +/- z*sigma band has it == 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        skew_ratio = float(np.median((hi - mn) / (mn - lo)))
    return two_sided, upper_tail, lower_tail, skew_ratio


def test_dispersion_location_scale_observation_band_is_skewed_not_symmetric():
    rng = np.random.default_rng(11)
    train = _skewed_gamma_gamlss(rng, 6000)
    test = _skewed_gamma_gamlss(rng, 30000)
    y = test["y"].to_numpy()

    model = gamfit.fit(train, "y ~ s(x)", family="gamma", noise_formula="s(x)")
    pred = model.predict(test, interval=0.95, observation_interval=True)
    two_sided, upper_tail, lower_tail, skew_ratio = _band_stats(pred, y)

    # Sanity: it is a *shape* defect, not a width defect — total coverage near
    # nominal (passes both pre- and post-fix).
    assert two_sided > 0.90, f"two-sided coverage collapsed: {two_sided}"

    # Core assertion: a right-skewed Gamma's predictive interval cannot be
    # symmetric about the mean. The buggy symmetric band gives skew_ratio == 1.0
    # exactly; an equal-tailed Gamma band gives ratio well above 1.
    assert skew_ratio > 1.3, (
        "dispersion location-scale Gamma observation band is symmetric "
        f"(median (hi-mu)/(mu-lo)={skew_ratio:.3f}); a right-skewed Gamma "
        "predictive interval must be right-skewed (equal-tailed quantiles)."
    )

    # The symmetric band's upper edge sits below the true upper quantile, so the
    # upper tail under-covers (~0.052 measured) at a nominal 2.5%-per-tail band.
    assert upper_tail <= 0.04, (
        f"upper tail under-covers: P(Y > observation_upper)={upper_tail:.3f} "
        "(nominal 0.025); the symmetric band undershoots the Gamma upper quantile."
    )


def test_location_scale_band_matches_standard_path_skew_on_identical_data():
    """Cross-check from a different angle: the standard (single-block) Gamma fit
    and the dispersion location-scale fit, given the *same* data, must produce
    comparably right-skewed observation bands. The standard path already builds
    equal-tailed Gamma quantiles (#817); the location-scale path must too.

    Pre-fix the standard band is strongly skewed (ratio ~2.2) while the
    location-scale band is exactly symmetric (ratio 1.0), so their ratio of
    ratios is ~2.2 — far from parity. Post-fix both are equal-tailed and the two
    skews are within a small factor of each other.
    """
    rng = np.random.default_rng(11)
    train = _skewed_gamma_gamlss(rng, 6000)
    test = _skewed_gamma_gamlss(rng, 30000)
    y = test["y"].to_numpy()

    std = gamfit.fit(train, "y ~ s(x)", family="gamma")
    ls = gamfit.fit(train, "y ~ s(x)", family="gamma", noise_formula="s(x)")

    _, _, _, std_skew = _band_stats(
        std.predict(test, interval=0.95, observation_interval=True), y
    )
    _, _, _, ls_skew = _band_stats(
        ls.predict(test, interval=0.95, observation_interval=True), y
    )

    # The standard path is genuinely right-skewed (guards against the data not
    # actually being skewed).
    assert std_skew > 1.5, f"standard Gamma band not skewed enough: {std_skew:.3f}"

    # The location-scale band must share that skew character, not collapse to a
    # symmetric (ratio 1.0) band.
    assert ls_skew > 0.6 * std_skew, (
        f"location-scale band skew {ls_skew:.3f} is far below the standard "
        f"path's {std_skew:.3f}: the equal-tailed fix did not reach the "
        "dispersion location-scale predictor."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
