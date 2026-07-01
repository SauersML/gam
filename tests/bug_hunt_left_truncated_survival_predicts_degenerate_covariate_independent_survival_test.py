"""Bug hunt: left-truncated (delayed-entry) survival fits collapse to a
degenerate, covariate-independent survival curve.

The three-argument ``Surv(entry, exit, event)`` form denotes *left truncation*
(delayed entry): a subject is only under observation from ``entry`` onward, and
its likelihood contribution conditions on survival to ``entry``. This is a
standard, documented survival feature; the two-argument ``Surv(exit, event)``
form (equivalently ``entry == 0``) is the ordinary right-censored case.

Observed: any nonzero ``entry`` destroys the fit. On data with a clear
covariate effect (hazard ``0.4·exp(0.9·x)``):

* ``entry == 0`` (control): predicted survival is well-posed —
  ``S(0.5) ≈ 0.90`` for the low-hazard covariate, ``S`` decreasing in ``t``,
  and the two covariate values give clearly *different* curves. The predicted
  cumulative hazard matches the truth (``H(1) ≈ 0.2``).
* ``entry == 0.05`` (or any ``entry > 0``): the predicted cumulative hazard is
  inflated by ~10³× (``H ≈ 186``, and nearly flat in ``t``), so ``S(t)``
  collapses to ``0`` at every queried time AND becomes *identical* across
  covariate values — the covariate no longer affects the prediction at all.
  The fit itself emits a railed smoothing parameter / gradient-objective
  desync under left truncation.

This reproduces deterministically for every seed and every ``entry > 0``
(``1e-6 … 0.5``); ``entry == 0`` always fits correctly.

Root-cause read: the delayed-entry design (the ``x_entry_time`` log-entry basis
built in ``crates/gam-models/src/survival/construction.rs`` ~lines 1250-1359)
destabilizes the transformation-survival smoothing selection, railing a penalty
direction and producing a degenerate baseline whose cumulative hazard swamps
the covariate smooth. The Python ``gamfit/_survival.py`` interpolation shell
faithfully returns the degenerate Rust surface.

Contract asserted here (well-posed): a left-truncated survival model with a
genuine covariate effect must produce a NON-DEGENERATE, COVARIATE-DEPENDENT
survival curve — the same qualitative behavior the ``entry == 0`` fit produces
on identical ``exit``/``event``/``x``. This test fits with ``entry = 0.05`` and
asserts (a) early-time survival for the low-hazard covariate is not collapsed
to ~0 and (b) the two covariate values give materially different survival
curves. It currently fails (``S ≡ 0``, curves identical); once left truncation
is handled correctly it passes without edits. The ``entry == 0`` control is
asserted well-posed too, to pin the defect to the delayed-entry path.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_GRID = np.array([0.5, 1.0, 2.0])
_X_LOW, _X_HIGH = -0.8, 0.8  # low-hazard vs high-hazard covariate values


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return fn(*args, **kwargs)


def _fit_and_survival(entry: float, seed: int) -> "np.ndarray":
    """Return the (2, len(grid)) survival surface for x in {low, high}."""
    rng = np.random.default_rng(seed)
    n = 1500
    x = rng.uniform(-1.0, 1.0, n)
    lam = 0.4 * np.exp(0.9 * x)
    t_event = rng.exponential(1.0 / lam)
    t_cens = rng.exponential(5.0, n)
    exit_t = np.maximum(np.minimum(t_event, t_cens), entry + 0.11)
    event = (t_event <= t_cens).astype(int)
    data = {
        "entry": np.full(n, entry).tolist(),
        "exit": exit_t.tolist(),
        "event": event.tolist(),
        "x": x.tolist(),
    }
    model = _quiet(gamfit.fit, data, "Surv(entry, exit, event) ~ s(x)", family="survival")
    pred = _quiet(
        model.predict,
        {"entry": [0, 0], "exit": [1, 1], "event": [1, 1], "x": [_X_LOW, _X_HIGH]},
    )
    return np.asarray(pred.survival_at(_GRID), dtype=float)


def test_zero_entry_control_is_wellposed() -> None:
    # Ordinary right-censored fit (entry == 0): a sanity anchor that the data,
    # formula, and prediction path are sound, so the failure below is isolated
    # to the delayed-entry handling rather than a broken harness.
    surv = _fit_and_survival(entry=0.0, seed=11)
    s_low = surv[0]
    assert s_low[0] > 0.5, f"control S_low(0.5)={s_low[0]:.3f} should be ~0.9"
    assert not np.allclose(surv[0], surv[1]), "control survival should depend on covariate"


def test_left_truncated_survival_is_nondegenerate_and_covariate_dependent() -> None:
    entry = 0.05
    failures: list[str] = []
    for seed in (1, 11, 22):
        surv = _fit_and_survival(entry=entry, seed=seed)
        s_low, s_high = surv[0], surv[1]

        # (a) Not collapsed: early-time survival for the LOW-hazard covariate
        # should be substantial (truth S(0.5) ~ 0.9; a correct fit clears 0.3).
        if not (s_low[0] > 0.3):
            failures.append(
                f"seed={seed}: S_low(0.5)={s_low[0]:.3f} collapsed (expected ~0.9)"
            )
        # (b) Covariate-dependent: the two hazard-distinct covariates must give
        # materially different survival curves.
        if np.max(np.abs(s_low - s_high)) <= 0.05:
            failures.append(
                f"seed={seed}: survival identical across covariates "
                f"(max|Δ|={np.max(np.abs(s_low - s_high)):.4f}); S_low={np.round(s_low,3)}"
            )

    assert not failures, (
        "left-truncated (entry>0) survival fit is degenerate / covariate-independent:\n  "
        + "\n  ".join(failures)
    )
