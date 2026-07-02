"""#2033: ``Model.sample_replicates`` must not drop the prior-weights column.

Regression for the #2025 → #2033 chain. #2025 ("Scale Gaussian replicate
observation noise by prior weights") made the generative-replicate path resolve
the fitted model's weight column so that replicate observation noise is
heteroskedastic, ``Var(y_i) = sigma_hat^2 / w_i`` (i.e. ``sigma_i =
sigma_hat / sqrt(w_i)``). #2033 was the regression that shipped with it: the
replicate frame is first narrowed to the model's *consumable* columns, and that
set did not include the weight column, so ``resolve_weight_column`` looked it up
in a frame from which it had already been projected away and raised
``weights column 'w' not found`` for **every** weighted model — even when the
caller passed the weight column in explicitly.

The fix keeps the weight column in
``prediction_consumable_columns`` (so it survives projection) and, as a
belt-and-suspenders default, falls back to unit weights in the replicate path
when the caller's frame simply does not carry the column, rather than erroring.

Objective assertions (the generative law is the ground truth — no reference tool
needed):
  (a) a weighted model's ``sample_replicates`` succeeds and returns
      ``(n_draws, n_rows)`` — the raw #2033 repro;
  (b) the drawn replicates are heteroskedastic in the prior weights: per-row
      replicate spread tracks ``sigma_hat / sqrt(w_i)`` (the #2025 contract);
  (c) a weighted model whose replicate frame omits the weight column degrades
      gracefully to the pooled scalar noise instead of raising (the fallback
      branch, a different code path from (a)/(b));
  (d) an unweighted model is unaffected (control).
"""
from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def _weighted_rows(
    n: int, seed: int, *, w_lo: float, w_hi: float
) -> tuple[list[dict[str, float]], np.ndarray]:
    """A Gaussian problem with a sharp two-level weight split so the
    heteroskedastic replicate spread is unambiguous. The first half of the rows
    carry the *low* weight ``w_lo`` (large observation variance) and the second
    half the *high* weight ``w_hi`` (small observation variance)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n)
    y = np.sin(1.3 * x) + 0.3 + rng.normal(scale=0.4, size=n)
    w = np.empty(n)
    w[: n // 2] = w_lo
    w[n // 2 :] = w_hi
    rows = [{"y": float(y[i]), "x": float(x[i]), "w": float(w[i])} for i in range(n)]
    return rows, w


def test_weighted_model_sample_replicates_does_not_drop_weight_column() -> None:
    """(a) The raw #2033 repro: a weighted Gaussian fit's replicate draw must
    succeed with the right shape when the frame carries the weight column."""
    rows, _ = _weighted_rows(200, seed=0, w_lo=0.5, w_hi=3.0)
    model = gamfit.fit(rows, "y ~ s(x)", family="gaussian", weights="w")
    reps = np.asarray(model.sample_replicates(rows, 20, seed=7), dtype=float)
    assert reps.shape == (20, len(rows))
    assert np.all(np.isfinite(reps))


def test_weighted_replicates_are_heteroskedastic_in_prior_weights() -> None:
    """(b) The #2025 contract: replicate observation noise is
    ``sigma_i = sigma_hat / sqrt(w_i)``. High-weight rows must show a materially
    smaller replicate spread than low-weight rows, and the empirical per-row
    spread must track ``1/sqrt(w_i)`` up to the shared scalar ``sigma_hat``."""
    n = 240
    w_lo, w_hi = 0.5, 8.0
    rows, w = _weighted_rows(n, seed=1, w_lo=w_lo, w_hi=w_hi)
    model = gamfit.fit(rows, "y ~ s(x)", family="gaussian", weights="w")

    reps = np.asarray(model.sample_replicates(rows, 6000, seed=3), dtype=float)
    per_row_std = reps.std(axis=0)

    lo = per_row_std[: n // 2]
    hi = per_row_std[n // 2 :]
    # Heteroskedastic: the high-weight half is drawn with a smaller scale.
    assert hi.mean() < lo.mean(), (
        f"expected high-weight rows to have smaller replicate spread; "
        f"got hi.mean()={hi.mean():.4f} >= lo.mean()={lo.mean():.4f}"
    )
    # Quantitatively, the ratio of spreads tracks sqrt(w_hi / w_lo). The #2025
    # law predicts lo.mean()/hi.mean() == sqrt(w_hi/w_lo); allow 20% Monte-Carlo
    # slack. Crucially this ratio is FAR from 1.0, which is what the pooled
    # scalar (the pre-#2025 / #2033-broken behaviour) would give.
    expected_ratio = float(np.sqrt(w_hi / w_lo))
    observed_ratio = float(lo.mean() / hi.mean())
    assert abs(observed_ratio - expected_ratio) / expected_ratio < 0.2, (
        f"replicate-spread ratio {observed_ratio:.3f} strayed from the "
        f"analytic-weight prediction sqrt(w_hi/w_lo)={expected_ratio:.3f}"
    )

    # And the full per-row spread is proportional to 1/sqrt(w_i): the implied
    # per-row sigma_hat = per_row_std * sqrt(w_i) must be near-constant across
    # rows (a single pooled scale), unlike the broken pooled draw whose
    # per_row_std is already constant so sigma_hat would spuriously track
    # sqrt(w_i).
    implied_sigma = per_row_std * np.sqrt(w)
    cv = implied_sigma.std() / implied_sigma.mean()
    assert cv < 0.1, (
        f"per-row implied sigma_hat = std * sqrt(w) is not constant "
        f"(coefficient of variation {cv:.3f}); replicate noise is not "
        f"sigma_hat/sqrt(w_i)"
    )


def test_weighted_model_replicate_frame_without_weight_column_degrades_gracefully() -> None:
    """(c) Fallback branch: if the caller's replicate frame simply omits the
    weight column, the replicate path must fall back to unit weights (pooled
    scalar noise) rather than raising. This exercises the projection-independent
    ``col_map``-membership guard, a different path from (a)/(b)."""
    rows, _ = _weighted_rows(160, seed=2, w_lo=0.5, w_hi=3.0)
    model = gamfit.fit(rows, "y ~ s(x)", family="gaussian", weights="w")
    frame_no_w = [{"y": r["y"], "x": r["x"]} for r in rows]
    reps = np.asarray(model.sample_replicates(frame_no_w, 20, seed=9), dtype=float)
    assert reps.shape == (20, len(rows))
    assert np.all(np.isfinite(reps))
    # With unit weights every row shares the pooled scalar scale, so the per-row
    # spreads are homoskedastic (no 1/sqrt(w) structure).
    per_row_std = reps.std(axis=0)
    cv = per_row_std.std() / per_row_std.mean()
    assert cv < 0.3, (
        f"fallback replicate spread should be ~homoskedastic; got cv={cv:.3f}"
    )


def test_unweighted_model_replicates_unaffected() -> None:
    """(d) Control: an unweighted fit never touches the weight-resolution path
    and continues to draw homoskedastic pooled-scalar noise."""
    rows, _ = _weighted_rows(160, seed=4, w_lo=0.5, w_hi=3.0)
    frame = [{"y": r["y"], "x": r["x"]} for r in rows]
    model = gamfit.fit(frame, "y ~ s(x)", family="gaussian")
    reps = np.asarray(model.sample_replicates(frame, 20, seed=11), dtype=float)
    assert reps.shape == (20, len(frame))
    assert np.all(np.isfinite(reps))
