"""Regression for the *over-correction* companion of the smooth_significance()
LR reference-d.f. bug (#1766).

The first fix for #1766 stopped the false-positive collapse (a flat smooth no
longer reported ``p ~ 1e-12``) but did so by flooring ``ref_df`` at
``design_term.nullspace_dims.iter().sum()`` — the **sum** of the per-penalty
null-space dimensions. That sum *unions* the null spaces (the #1360 defect that
``joint_unpenalized_dim`` exists to avoid): a double-penalty smooth carries a
bending penalty (null space = its polynomial part) plus a complementary
null-space ridge (which penalizes exactly that polynomial part), so the two
null spaces are disjoint and the sum equals nearly the full basis dimension.
The floor therefore pinned ``ref_df`` to a constant ~``k`` (e.g. 19 for a
``k=20`` smooth) for **every** fit, regardless of the fitted complexity.

That silences the collapse yet makes the whole-term LR test badly conservative
for genuine moderate signals: a term with effective d.f. ~5 was judged against
``chi^2_{19}`` instead of ``~chi^2_{6}``. The original RED test and the
lower-bound invariant (``ref_df >= edf``) both still passed under that
over-correction — so this file guards the upper side.

The reference d.f. is an ``edf1``-style *effective* d.f.: it must TRACK the
smooth's fitted complexity, not saturate to the basis dimension. We assert:

1. ``ref_df`` stays within the ``edf1`` band ``[term_edf, 2*term_edf + slack]``
   (Wood's ``edf <= edf1 <= 2*edf``) across a flat -> wiggly sweep, and
2. ``ref_df`` genuinely VARIES with complexity — the buggy constant floor gave
   a spread of exactly 0.

Both fail loudly the instant ``ref_df`` is pinned to the basis dimension.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit


K = 20  # basis dimension: the value the buggy floor pinned ref_df to (~K-1)
N = 300


def _sweep() -> list[dict[str, float]]:
    # Increasing true wiggliness at a fixed larger basis so the fitted smooth
    # spans effective d.f. from ~0 (flat) up to ~15, all well below K. A
    # ref_df pinned to the basis dimension (~19) is impossible to miss here.
    freqs = [0.0, 0.0, 1.0, 2.0, 3.5, 6.0]
    out: list[dict[str, float]] = []
    for i, f in enumerate(freqs):
        rng = np.random.default_rng(500 + i)
        x = np.linspace(0.0, 1.0, N)
        sig = np.zeros_like(x) if f == 0.0 else np.sin(f * np.pi * x)
        y = sig + 0.3 * rng.standard_normal(N)
        model = gamfit.fit({"x": list(x), "y": list(y)}, f"y ~ s(x, k={K})")
        summary = model.summary()
        # For a single s(x) the unpenalized parametric part is the intercept
        # (exactly 1 d.f. under the identity link), so the smooth term's own
        # effective d.f. is edf_total - 1.
        term_edf = max(float(summary.edf_total) - 1.0, 0.0)
        rec = model.smooth_significance({"x": list(x), "y": list(y)})[0]
        out.append(
            {
                "freq": f,
                "term_edf": term_edf,
                "ref_df": float(rec["ref_df"]),
                "p": float(rec["p_value_corrected"]),
                # The fit's own convergence certificate, not an iteration-count
                # proxy. Every fit in this sweep is well posed (a flat or a
                # smooth mean, Gaussian noise, N = 300 >> K = 20), so each one
                # must certify: a pure-noise fit that stalls is a solver defect
                # to fix, not a fit to leave out of the band check (SPEC.md:
                # "In general, do not paper over solver issues.").
                "certified": bool(model.convergence["certified"]),
            }
        )
    uncertified = [r["freq"] for r in out if not r["certified"]]
    assert not uncertified, (
        "well-posed y ~ s(x) fits did not certify convergence (freq = "
        f"{uncertified}); the ref_df band cannot be judged on a fit whose "
        "smoothing parameters are not at a stationary point"
    )
    return out


def test_ref_df_stays_in_edf1_band_not_basis_dimension() -> None:
    # Wood's edf1 satisfies edf <= edf1 <= 2*edf (eigenvalues of the smoother
    # block lie in [0, 1]). For a CONVERGED fit the reference d.f. must respect
    # the upper side too; the buggy per-penalty-sum floor pinned it to ~K-1 and
    # blew past this band for every moderately-shrunk fit.
    sweep = _sweep()
    offenders = []
    for r in sweep:
        upper = 2.0 * r["term_edf"] + 3.0  # generous slack over the 2*edf bound
        if r["ref_df"] > upper:
            offenders.append(r)
    assert not offenders, (
        "ref_df exceeded the edf1 band [term_edf, 2*term_edf+3] on a CONVERGED "
        "fit — it is saturating toward the basis dimension instead of tracking "
        "the fitted complexity (the #1766 per-penalty-sum over-correction). "
        "Offenders: "
        + "; ".join(
            f"(freq={o['freq']}, term_edf={o['term_edf']:.2f}, ref_df={o['ref_df']:.2f})"
            for o in offenders
        )
    )


def test_ref_df_varies_with_fitted_complexity() -> None:
    # The single most direct signature of the basis-dimension pin: ref_df is a
    # constant across fits of wildly different complexity. A correct edf1-style
    # reference spans a wide range as the smooth goes flat -> wiggly. `_sweep`
    # requires every fit to certify, so no non-converged fallback (~K) can
    # supply the spread.
    refs = [r["ref_df"] for r in _sweep()]
    spread = max(refs) - min(refs)
    assert spread > 4.0, (
        f"ref_df barely varied across a flat->wiggly sweep (spread={spread:.3f}); "
        f"it appears pinned to a constant near the basis dimension K={K}. "
        f"ref_df values: {[round(v, 3) for v in refs]}"
    )


def test_moderate_signal_is_not_judged_over_conservatively() -> None:
    # A genuine moderate signal (effective d.f. ~5, comfortably below K) must be
    # detectable. Referencing its LR statistic against chi^2_{K-1} instead of
    # ~chi^2_{edf1} inflates the p-value and can hide a real effect.
    rng = np.random.default_rng(4242)
    x = np.linspace(0.0, 1.0, N)
    y = np.sin(1.0 * np.pi * x) + 0.3 * rng.standard_normal(N)
    model = gamfit.fit({"x": list(x), "y": list(y)}, f"y ~ s(x, k={K})")
    term_edf = max(float(model.summary().edf_total) - 1.0, 0.0)
    rec = model.smooth_significance({"x": list(x), "y": list(y)})[0]
    assert term_edf > 2.0, f"setup: expected a moderate fit, got term_edf={term_edf:.2f}"
    assert rec["ref_df"] <= 2.0 * term_edf + 3.0, (
        f"moderate fit judged against an inflated ref_df={rec['ref_df']:.2f} "
        f"(term_edf={term_edf:.2f})"
    )
    assert rec["p_value_corrected"] < 1e-2, (
        "a genuine moderate signal was not detected "
        f"(term_edf={term_edf:.2f}, ref_df={rec['ref_df']:.2f}, "
        f"p={rec['p_value_corrected']:.3g}) — reference d.f. is over-conservative"
    )


def test_pure_noise_fits_converge_and_null_p_values_are_uniform() -> None:
    # Pure noise: the smooth has no effect, so the term's p-value must be
    # U(0, 1) -- neither piled near 0 (the #1766 collapse, where a stalled
    # flat-valley fit's large W was referenced against its ~0 edf) nor pushed
    # toward 1 (a conservative reference such as chi^2 on the full basis
    # dimension). Both are the same defect seen from the two sides.
    #
    # This test used to REQUIRE that some of these fits stall (`stalls > 0`,
    # a stall being `outer_iterations >= 200`) and to go red when none did, as
    # the recorded python-contracts run 30600186060 shows. A test that demands
    # the REML optimizer fail on a well-posed y ~ s(x) fit papers over the
    # stall it names (SPEC.md: "In general, do not paper over solver
    # issues."). Every fit must instead certify, which is what the
    # smooth_significance calibration study measures on its cells
    # (bench/pvalue_calibration/pv-lr-refit: 500 of 500 fits converged, null
    # p-values uniform, no mass at p = 1).
    #
    # Two-sided gates, derived from the null law with R = 60 fits:
    #  * the mean p-value of U(0, 1) is 1/2 with variance 1/12, so the mean of
    #    R null p-values has sd sqrt(1/(12 R)) = 0.037; it must lie within
    #    4 sd of 1/2;
    #  * the rejection rate at alpha = 0.05 keeps its existing upper bar of
    #    0.15 (Binomial(R, alpha) sd on the rate is 0.028, so the bar is 3.6 sd
    #    above alpha). Its lower side is below zero at this R, which is why the
    #    mean gate is the one that sees a conservative test.
    n_seeds = 60
    alpha = 0.05
    ps: list[float] = []
    uncertified = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(3000 + seed)
        x = np.linspace(0.0, 1.0, N)
        y = rng.standard_normal(N)  # pure noise: no smooth effect
        model = gamfit.fit({"x": list(x), "y": list(y)}, "y ~ s(x)")
        if not bool(model.convergence["certified"]):
            uncertified.append((seed, int(model.outer_iterations)))
        rec = model.smooth_significance({"x": list(x), "y": list(y)})[0]
        ps.append(float(rec["p_value_corrected"]))
    assert not uncertified, (
        f"pure-noise y ~ s(x) fits did not certify convergence "
        f"(seed, outer_iterations) = {uncertified}; the flat-valley REML stall "
        "(#1762) is a solver defect, not an expected outcome"
    )
    mean_p = float(np.mean(ps))
    mean_sd = (1.0 / (12.0 * n_seeds)) ** 0.5
    fpr = sum(p < alpha for p in ps) / n_seeds
    assert abs(mean_p - 0.5) <= 4.0 * mean_sd, (
        f"null p-values are not U(0, 1): mean p = {mean_p:.3f} over {n_seeds} "
        f"pure-noise fits, outside 1/2 +- 4 sd = +-{4.0 * mean_sd:.3f} "
        f"({'conservative' if mean_p > 0.5 else 'anti-conservative'}); "
        f"sorted p = {[round(p, 3) for p in sorted(ps)]}"
    )
    assert fpr <= 0.15, (
        f"null false-positive rate {fpr:.3f} at alpha = {alpha} over {n_seeds} "
        "pure-noise fits exceeds 0.15; the #1766 over-rejection is back"
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    test_ref_df_stays_in_edf1_band_not_basis_dimension()
    test_ref_df_varies_with_fitted_complexity()
    test_moderate_signal_is_not_judged_over_conservatively()
    test_pure_noise_fits_converge_and_null_p_values_are_uniform()
    print("ok")
