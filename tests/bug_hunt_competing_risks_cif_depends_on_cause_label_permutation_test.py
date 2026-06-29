"""Gauge-invariance audit (#1593): a competing-risks fit must not depend on the
arbitrary *labeling* of the cause codes.

A competing-risks model fits one cause-specific Royston-Parmar hazard block per
positive event code ``k`` (``event == k`` is the event of interest for block
``k``; every competing cause is treated as censoring), plus a pooled
any-event baseline. Per-cause baselines are addressed by the integer event code:
cause ``k`` -> coefficient block ``k-1`` -> endpoint name ``cause_k`` (see
``crates/gam-models/src/survival/predict.rs`` and ``base.rs``
``cause_specific_event_indicator``). The integer code is the *physical cause
identity*; which integer a given physical cause is assigned is an arbitrary gauge
choice.

Permuting the cause labels in the input data (e.g. swapping event codes 1<->2)
must therefore leave the predicted cumulative incidence functions (CIFs)
invariant *up to the relabeling*: the CIF for endpoint ``cause_k`` under the
permuted labeling must equal the CIF for endpoint ``cause_{pi(k)}`` under the
original labeling, where ``pi`` is the applied permutation. Concretely:

  * each cause-specific block fits ``event == k`` independently, so a permuted
    block fits byte-identical data to the original block it maps to;
  * the pooled any-event baseline and the overall survival ``S = exp(-sum_j H_j)``
    are symmetric in the causes, hence permutation-invariant;
  * the Aalen-Johansen CIF assembly splits each interval failure by the symmetric
    ratio ``dH_k / sum_j dH_j``, so ``F_k`` carries the cause identity ``k`` and
    nothing else.

So the prediction *should* be invariant up to relabeling. This banks that as a
regression guard: it would surface a real #1593-class frame bug were a per-cause
baseline ever keyed to a sorted/discovery position rather than to the cause
identity, or were any non-symmetric cause ordering to leak into the fit or the
CIF assembly.

It fits the model on the original labeling and on a relabeling that cyclically
permutes the cause codes, predicts on a shared covariate grid, realigns the
permuted endpoints back to the physical cause identity via the inverse
permutation, and asserts the cross-labeling drift is tiny.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

# Three competing causes; codes {0=censored, 1, 2, 3}. Distinct cause-specific
# log-hazard covariate slopes so the per-cause CIFs are genuinely different (an
# all-equal fit would make the invariance assertion vacuous).
_N_CAUSES = 3
_SLOPES = np.array([0.25, -0.20, 0.05])
_INTERCEPTS = np.array([-3.0, -3.2, -3.4])


def _make_competing_risks(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    x = (age - 55.0) / 10.0
    # One latent cause-specific exponential time per cause, plus a censoring time;
    # the realized event is the argmin (classic latent-failure competing risks).
    cause_times = np.empty((_N_CAUSES, n))
    for c in range(_N_CAUSES):
        rate = np.exp(_INTERCEPTS[c] + _SLOPES[c] * x)
        cause_times[c] = rng.exponential(scale=1.0 / rate, size=n)
    censor = rng.exponential(scale=22.0, size=n)
    first_cause = np.argmin(cause_times, axis=0)  # 0-based winning cause
    first_time = cause_times[first_cause, np.arange(n)]
    observed = np.minimum(first_time, censor) + 0.1
    # event code: 0 if censored, else winning cause code (1-based).
    event = np.where(first_time < censor, first_cause + 1, 0).astype(float)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": observed,
            "event": event,
            "age": age,
        }
    )


# Symmetric per-cause precision hyperpriors keyed by the cause-specific penalty
# block labels the engine emits (#1512:
# `cause_specific_survival_cause_{k}_penalty_0`). Symmetric across causes so the
# prior itself does not break the permutation invariance under test.
_HYPERPRIORS = {
    f"cause_specific_survival_cause_{k}_penalty_0": [2.0, 1.0]
    for k in range(1, _N_CAUSES + 1)
}


def _fit(df: pd.DataFrame):
    return gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="weibull",
        precision_hyperpriors=_HYPERPRIORS,
    )


def _predict_cif(model, pred_rows: pd.DataFrame):
    """Return (times, cif_by_cause) with cif_by_cause shape (n_causes, n_rows, n_times)."""
    pred = model.predict(pred_rows)
    assert isinstance(pred, gamfit.CompetingRisksPrediction)
    times = np.asarray(pred.times, dtype=float)
    cif = np.asarray(pred.cif, dtype=float)
    overall = np.asarray(pred.overall_survival, dtype=float)
    n_rows = overall.shape[0]
    cause_count = len(pred.endpoint_names)
    assert cause_count == _N_CAUSES
    assert cif.shape[0] == cause_count * n_rows
    # `cif` is endpoint-major (cause, row, time) — mirror the origin-guard test.
    cif_by_cause = cif.reshape(cause_count, n_rows, times.size)
    return times, cif_by_cause, overall


def _permute_events(event: np.ndarray, perm: list[int]) -> np.ndarray:
    """Relabel positive event codes by `perm` (1-based perm[k-1] = new code for
    old physical cause k). Censoring (0) is left untouched."""
    out = event.copy()
    for old_code in range(1, _N_CAUSES + 1):
        out[event == old_code] = perm[old_code - 1]
    return out


def test_competing_risks_cif_invariant_to_cause_label_permutation() -> None:
    train = _make_competing_risks()
    pred_rows = pd.DataFrame(
        {
            "entry": [0.0, 0.0, 0.0],
            "exit": [5.0, 10.0, 20.0],
            "event": [1, 1, 1],
            "age": [45.0, 55.0, 65.0],
        }
    )

    base_model = _fit(train)
    base_times, base_cif, base_overall = _predict_cif(base_model, pred_rows)

    # The CIFs must be physically distinct across causes, otherwise the
    # permutation-invariance check below is vacuous.
    spread = max(
        float(np.max(np.abs(base_cif[i] - base_cif[j])))
        for i in range(_N_CAUSES)
        for j in range(i + 1, _N_CAUSES)
    )
    assert spread > 1e-3, (
        f"degenerate fit: per-cause CIFs are nearly identical (max pairwise "
        f"spread {spread:.3e}); the invariance assertion would be vacuous"
    )

    # Same-labeling refit must be deterministic (control for solver noise so the
    # cross-labeling drift below is attributable to a real label dependence).
    refit_model = _fit(train)
    _, refit_cif, _ = _predict_cif(refit_model, pred_rows)
    refit_noise = float(np.max(np.abs(base_cif - refit_cif)))
    assert refit_noise < 1e-6, (
        f"same-labeling refit is non-deterministic (drift {refit_noise:.3e})"
    )

    # Cyclic relabeling: old physical cause 1->2, 2->3, 3->1. The inverse maps a
    # NEW endpoint code back to the OLD physical cause it represents.
    perm = [2, 3, 1]  # perm[old-1] = new code
    # inverse[new-1] = old physical cause code for that new endpoint.
    inverse = [0] * _N_CAUSES
    for old in range(1, _N_CAUSES + 1):
        inverse[perm[old - 1] - 1] = old

    permuted = train.copy()
    permuted["event"] = _permute_events(train["event"].to_numpy(), perm)
    perm_model = _fit(permuted)
    perm_times, perm_cif, perm_overall = _predict_cif(perm_model, pred_rows)

    # The default prediction grid is built from the (unchanged) entry/exit times,
    # so the two fits must share an identical time grid for a meaningful compare.
    assert perm_times.shape == base_times.shape
    np.testing.assert_allclose(
        perm_times, base_times, rtol=0.0, atol=1e-9,
        err_msg="permuting cause labels changed the prediction time grid",
    )

    # Overall (all-cause) survival is symmetric in the causes -> invariant.
    overall_drift = float(np.max(np.abs(perm_overall - base_overall)))
    assert overall_drift < 1e-3, (
        f"overall survival depends on cause labeling (drift {overall_drift:.3e}); "
        "S = exp(-sum_k H_k) is symmetric in the causes and must be invariant"
    )

    # Cause-specific CIFs must match under the inverse permutation: the permuted
    # endpoint `cause_{new}` represents physical cause `inverse[new-1]`.
    drift = 0.0
    for new_code in range(1, _N_CAUSES + 1):
        old_code = inverse[new_code - 1]
        drift = max(
            drift,
            float(np.max(np.abs(perm_cif[new_code - 1] - base_cif[old_code - 1]))),
        )
    assert drift < 1e-3, (
        f"competing-risks CIF depends on the arbitrary cause-label permutation "
        f"(max drift {drift:.3e} after realigning by the inverse permutation; "
        f"per-cause CIF spread {spread:.3e}, refit noise {refit_noise:.3e}). "
        "Per-cause baselines must be keyed to the cause identity, not a sorted "
        "index position (#1593 class)."
    )
