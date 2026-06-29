"""Bug hunt (#1617): a Gaussian smooth fit with integer case weights must be
identical to the fit on the row-expanded data (each row physically repeated
``w`` times), exactly as every fixed-dispersion family already is.

``weights`` are documented frequency / case weights: a row observed with weight
``w`` contributes ``w`` copies to the log-likelihood
(``crates/gam-terms/src/inference/lawley.rs``, ``RowKappas::weighted``). Under
that contract the weighted fit and the row-expanded fit encode the *identical*
likelihood, so ``beta_hat``, the EDF, the predicted curve and every standard
error must agree to numerical tolerance.

Root cause (fixed): the Gaussian identity-link *profiled* REML/scale denominator
counted the number of positive-weight ROWS (``n_+``) instead of the total weight
``sum(w_i)``. The deviance numerator already carries total mass ``~ sum(w)``, so
dividing by ``n_+`` inflated ``phi_hat`` by ``sum(w)/n_+``; that inflated scale
drove the REML lambda-selection toward more smoothing, so the weighted fit
systematically *over-smoothed* (EDF ~1.0 below the row-expanded fit, predicted
curve drifting ~0.5-1.2% of the signal range) and every confidence band was
inflated by ``~sqrt(sum(w)/n_+)``. Fixed-dispersion families (Poisson/binomial/
Gamma) have no profiled scale and so were always exact; they serve here as the
machinery/knot-placement control.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable

import numpy as np
import pandas as pd

gamfit = import_module("gamfit")


def _weighted_vs_expanded(
    family: str,
    ygen: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    seed: int,
    weight_hi: int = 4,
) -> dict[str, object]:
    """Fit ``family`` with integer case weights and on the row-expanded data.

    ``bs="ps"`` uses uniform knots over ``[min(x), max(x)]``, which row
    duplication cannot move, so the two encodings share an identical basis —
    any disagreement is in the estimator, not the knots.
    """
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    w = rng.integers(1, weight_hi, n).astype(float)  # frequency weights
    y = ygen(x, rng)
    df = pd.DataFrame({"x": x, "y": y, "w": w})
    idx = np.repeat(np.arange(n), w.astype(int))
    dfe = df.iloc[idx].reset_index(drop=True)  # row expansion

    formula = 'y ~ s(x, bs="ps", k=15)'
    mw = gamfit.fit(df, formula, family=family, weights="w")
    me = gamfit.fit(dfe, formula, family=family)

    grid = pd.DataFrame({"x": np.linspace(0.03, 0.97, 80)})
    pw = np.asarray(mw.predict(grid), dtype=float)
    pe = np.asarray(me.predict(grid), dtype=float)
    rel = float(np.max(np.abs(pw - pe)) / np.ptp(pe))

    return {
        "rel": rel,
        "edf_w": float(mw.summary().edf_total),
        "edf_e": float(me.summary().edf_total),
        "model_w": mw,
        "model_e": me,
        "grid": grid,
    }


def _confidence_se(model, grid: pd.DataFrame) -> np.ndarray:
    # `interval=<level>` (a float) turns on the SE-only confidence band; the
    # returned table gains a `std_error` column scaling as sqrt(phi_hat).
    pred = model.predict(grid, interval=0.95)
    return np.asarray(pred.std_error, dtype=float)


def _gaussian(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) + r.normal(0.0, 0.4, len(x))


def _poisson(x: np.ndarray, r: np.random.Generator) -> np.ndarray:
    return r.poisson(np.exp(0.5 + np.sin(2.0 * np.pi * x))).astype(float)


def test_poisson_case_weights_match_row_expansion_control() -> None:
    """Control: a fixed-dispersion family already matches row expansion, which
    proves the ``bs="ps"`` knots are duplication-invariant (uniform over the
    unchanged data range). If this fails, the harness itself is broken and the
    Gaussian assertion below would be meaningless."""
    res = _weighted_vs_expanded("poisson", _poisson, 11)
    assert res["rel"] < 1.5e-3, f"Poisson pred drift {res['rel']:.3e}"
    assert abs(res["edf_w"] - res["edf_e"]) < 0.05, (
        f"Poisson EDF {res['edf_w']:.4f} vs {res['edf_e']:.4f}"
    )


def test_gaussian_case_weights_match_row_expansion() -> None:
    """The headline #1617 regression: the weighted Gaussian smooth must
    reproduce the row-expanded fit's predictions and EDF. Before the fix the
    weighted EDF was consistently ~1.0 below the expanded EDF (over-smoothed)
    and the curve drifted ~0.5-1.2% of the signal range across seeds."""
    worst_rel = 0.0
    for seed in range(6):
        res = _weighted_vs_expanded("gaussian", _gaussian, seed)
        worst_rel = max(worst_rel, res["rel"])
        assert abs(res["edf_w"] - res["edf_e"]) < 0.15, (
            f"seed {seed}: weighted EDF {res['edf_w']:.4f} vs expanded "
            f"{res['edf_e']:.4f} (diff {res['edf_w'] - res['edf_e']:+.4f}) — "
            "the over-smoothing signature of #1617"
        )
    assert worst_rel < 1.5e-3, (
        f"worst Gaussian weighted-vs-expanded prediction drift {worst_rel:.3e} "
        "exceeds tolerance — the case-weight fit is not replication-equivalent"
    )


def test_gaussian_case_weights_confidence_band_matches_row_expansion() -> None:
    """Different angle on the same root cause: the smooth-fit *confidence
    standard error* — which scales as ``sqrt(phi_hat)`` — must coincide between
    the weighted and row-expanded Gaussian fits. Before the fix it was inflated
    by ``~sqrt(sum(w)/n_+)`` (the same defect #1618 reports parametrically). Uses
    a wider weight range {1..4} so ``sum(w)/n`` is far from 1, making the
    pre-fix inflation unmistakable."""
    for seed in range(4):
        res = _weighted_vs_expanded("gaussian", _gaussian, seed, weight_hi=5)
        se_w = _confidence_se(res["model_w"], res["grid"])
        se_e = _confidence_se(res["model_e"], res["grid"])
        # Median ratio is robust to the few grid points where the band is tiny.
        ratio = float(np.median(se_w / se_e))
        assert abs(ratio - 1.0) < 1e-2, (
            f"seed {seed}: weighted confidence SE / expanded SE median ratio "
            f"{ratio:.4f} (should be 1.0) — profiled scale not counting sum(w)"
        )
