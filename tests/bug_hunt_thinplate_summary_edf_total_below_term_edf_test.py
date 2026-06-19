"""Bug hunt: `Summary.edf_total` reported *below* a single smooth term's own
per-term EDF for certain 2-D thin-plate fits (issue #1356).

`edf_total` is the trace of the whole-model influence (hat) operator `F`, and a
term's per-term EDF is the trace of `F` restricted to that term's coefficient
block. A sum of non-negative per-term contributions can never be smaller than
any one of them, so the invariant `edf_total >= max(per-term EDF)` must always
hold — for a one-smooth model `edf_total ~= 1 (intercept) + smooth_term_edf`.

Observed on seeds 2 and 10 of `y ~ thinplate(x1, x2)`: the fit is fine
(R^2 ~ 0.995) and the single smooth term legitimately reports ~70.9 EDF, but
`edf_total` collapses onto its null/intercept floor of exactly 1.0 — the total
dropped the entire smooth contribution.

Root cause: the `edf_total` trace channel (`p - sum_k lambda_k*tr(H^-1 S_k)`)
factorized the TRANSFORMED stabilized Hessian with a bespoke 10x-escalation
ridge loop. On the degenerate-Hessian thin-plate corner that loop takes an
enormous ridge, inflating sum tr_kk toward `p` and clamping `edf_total` onto its
floor, while the influence matrix `F` (built from the rank-revealing inverse of
the original-basis Hessian) — which the prediction and the per-term EDF both
consume — keeps the honest ~71 EDF. The fix reconciles the two channels to the
same rank-revealing inverse so `edf_total = tr(F) = sum(per-term EDF)`.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def _fit_summary(seed, n=300):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    truth = np.sin(2 * x1) + np.cos(1.5 * x2) + 0.5 * x1 * x2
    y = truth + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    m = gamfit.fit(df, "y ~ thinplate(x1,x2)")
    s = m.summary()
    pred = np.asarray(m.predict(df[["x1", "x2"]])).reshape(-1)
    denom = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - np.sum((pred - truth) ** 2) / denom
    return s, r2


def _check_seed(seed):
    s, r2 = _fit_summary(seed)
    edf_total = float(s.edf_total)
    term_edf = max(float(t["edf"]) for t in s.smooth_terms)

    # Sanity: this is a good fit, not a collapsed one — the smooth genuinely
    # spent its degrees of freedom.
    assert r2 > 0.9, f"seed {seed}: unexpectedly poor fit R^2={r2:.4f}"
    assert term_edf > 5.0, f"seed {seed}: smooth term EDF too small ({term_edf:.3f})"

    # #1356 invariant: the model total cannot fall below a single term's EDF.
    assert edf_total + 1e-6 >= term_edf, (
        f"seed {seed}: edf_total ({edf_total:.4f}) is below the single smooth "
        f"term's own EDF ({term_edf:.4f}); the total-EDF trace channel is out of "
        f"sync with the influence-matrix EDF (#1356)"
    )


def test_thinplate_edf_total_not_below_term_edf_seed2():
    _check_seed(2)


def test_thinplate_edf_total_not_below_term_edf_seed10():
    _check_seed(10)


def test_thinplate_edf_total_consistent_baseline_seed0():
    # Seed 0 was already consistent (edf_total = smooth + 1); guard against a
    # regression in the well-behaved case.
    _check_seed(0)
