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
                # A fit that stalls without converging (the flat-valley REML
                # stall on an unidentified term, #1762) has an untrustworthy edf,
                # so smooth_significance deliberately references it against the
                # conservative full basis dimension rather than the tight edf1.
                # The tight-band assertion below therefore only applies to
                # CONVERGED fits; the pure-noise flat fits routinely stall.
                "converged": int(summary.iterations) < 200,
            }
        )
    return out


def test_ref_df_stays_in_edf1_band_not_basis_dimension() -> None:
    # Wood's edf1 satisfies edf <= edf1 <= 2*edf (eigenvalues of the smoother
    # block lie in [0, 1]). For a CONVERGED fit the reference d.f. must respect
    # the upper side too; the buggy per-penalty-sum floor pinned it to ~K-1 and
    # blew past this band for every moderately-shrunk fit.
    checked = 0
    offenders = []
    for r in _sweep():
        if not r["converged"]:
            continue  # non-converged fits are deliberately referenced at ~K
        checked += 1
        upper = 2.0 * r["term_edf"] + 3.0  # generous slack over the 2*edf bound
        if r["ref_df"] > upper:
            offenders.append(r)
    assert checked >= 3, f"expected several converged fits in the sweep, got {checked}"
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
    # reference spans a wide range as the smooth goes flat -> wiggly. Restrict to
    # converged fits so the non-converged conservative fallback (~K) does not
    # itself supply the spread.
    refs = [r["ref_df"] for r in _sweep() if r["converged"]]
    assert len(refs) >= 3, f"expected several converged fits, got {len(refs)}"
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


def test_nonconverged_flat_fit_is_not_flagged_significant() -> None:
    # The flat-valley REML stall (#1762) on an unidentified smooth over pure
    # noise leaves a NON-CONVERGED fit whose smoothing parameters rail out and
    # whose influence-based edf reads ~0 even though the term still carries a
    # large, wiggly beta (so the refit LR statistic W is large). Referencing that
    # large W against the ~0 edf manufactured overwhelming significance — the
    # dominant driver of the null-FPR blow-up in #1766. smooth_significance now
    # references a non-converged term against its full basis dimension, so those
    # stalls are no longer spuriously flagged. Verify the null false-positive
    # rate stays near alpha ACROSS the seeds where the stall actually happens.
    n_seeds = 60
    rej = 0
    stalls = 0
    worst = None
    for seed in range(n_seeds):
        rng = np.random.default_rng(3000 + seed)
        x = np.linspace(0.0, 1.0, N)
        y = rng.standard_normal(N)  # pure noise: no smooth effect
        model = gamfit.fit({"x": list(x), "y": list(y)}, "y ~ s(x)")
        summary = model.summary()
        rec = model.smooth_significance({"x": list(x), "y": list(y)})[0]
        p = float(rec["p_value_corrected"])
        if int(summary.iterations) >= 200:  # the stall signature
            stalls += 1
            if worst is None or p < worst[1]:
                worst = (seed, p, float(rec["statistic_lr"]), float(rec["ref_df"]))
        if p < 0.05:
            rej += 1
    fpr = rej / n_seeds
    assert stalls > 0, (
        "setup: expected some fits to hit the flat-valley stall; none did — "
        "the guard is untested by this sample"
    )
    assert fpr <= 0.15, (
        f"null false-positive rate {fpr:.3f} ({rej}/{n_seeds}) on pure noise "
        f"exceeds the tolerance; the non-converged over-rejection is back. "
        f"({stalls} fits stalled; worst stalled p={worst})"
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    test_ref_df_stays_in_edf1_band_not_basis_dimension()
    test_ref_df_varies_with_fitted_complexity()
    test_moderate_signal_is_not_judged_over_conservatively()
    test_nonconverged_flat_fit_is_not_flagged_significant()
    print("ok")
