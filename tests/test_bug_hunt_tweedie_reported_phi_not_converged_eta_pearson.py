"""Regression for #2105: the reported Tweedie dispersion ``phi`` must equal the
converged-mean Pearson estimate AND recover the true dispersion — it was ~13%
high, inflating every Tweedie SE / interval by ~6%.

ROOT CAUSE (verified at the time). The reported ``phi`` DID equal
``estimate_tweedie_phi_from_eta(final_eta)``. The bias was one level deeper: a bare
``family="tweedie"`` then ESTIMATED the variance power ``p`` by a saddlepoint
profile likelihood (#2026) that was biased LOW (``p_hat ~ 1.33`` on ``p = 1.5``
data). Because the reported dispersion is the Pearson estimate
``phi_hat = sum w (y - mu)^2 / mu^p / sum w``, an under-estimated ``p`` inflated
``phi_hat`` (0.676 vs the true 0.600 here).

HISTORY: a893d85bc retired the power profile (a derivative-free hyperparameter
search, forbidden by SPEC.md), so a bare ``tweedie`` is refused and the caller
names the power. The dispersion contract this file guards is unchanged, and it
is now checked at the requested power:

  1. reported ``phi`` == Pearson dispersion at the model's OWN converged mean and
     OWN power (the self-consistency invariant the code documents),
  2. the fitted power is exactly the requested one (no silent re-estimation), and
  3. reported ``phi`` recovers the true 0.6 within a tight band (the bug: 0.676).
"""

import json

import numpy as np
import pandas as pd
import pytest

import gamfit

P_TRUE = 1.5
PHI_TRUE = 0.6
N = 8000


def _tweedie_sample(rng, mu, phi, power):
    """Compound Poisson-Gamma (Jorgensen) Tweedie variate per row."""
    lam = mu ** (2.0 - power) / (phi * (2.0 - power))
    alpha = (2.0 - power) / (power - 1.0)
    scale = phi * (power - 1.0) * mu ** (power - 1.0)
    n_jumps = rng.poisson(lam)
    out = np.zeros_like(mu)
    for i in range(len(mu)):
        if n_jumps[i] > 0:
            out[i] = rng.gamma(alpha * n_jumps[i], scale[i])
    return out


def _find_key(obj, key):
    """Robustly pull every value stored under ``key`` from the JSON payload."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found += _find_key(v, key)
    elif isinstance(obj, list):
        for it in obj:
            found += _find_key(it, key)
    return found


def _fit_true_power_tweedie(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, N)
    mu = np.exp(0.7 + 0.5 * x)
    y = _tweedie_sample(rng, mu, PHI_TRUE, P_TRUE)
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ x", family=f"tweedie({P_TRUE})")
    return m, df, y


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_reported_phi_is_converged_eta_pearson_and_recovers_truth(seed):
    m, df, y = _fit_true_power_tweedie(seed)
    payload = json.loads(m.dumps())

    reported = _find_key(payload, "EstimatedTweediePhi")
    assert reported, "fit must report an estimated Tweedie phi"
    phi_reported = reported[0]["phi"]

    powers = [d["p"] for d in _find_key(payload, "Tweedie") if isinstance(d, dict) and "p" in d]
    assert powers, "fit must carry a Tweedie variance power"

    # (2) Every Tweedie power the payload records is the requested one: nothing
    #     re-estimated or replaced it.
    assert all(abs(p - P_TRUE) < 1e-12 for p in powers), (
        f"the payload records Tweedie powers {powers}, not the requested p={P_TRUE}"
    )
    p_fit = powers[-1]

    eta = np.asarray(
        m.predict(df, interval=0.9, return_type="pandas")[
            "linear_predictor_plugin"
        ]
    )
    mu_hat = np.exp(eta)

    # (1) Self-consistency invariant: reported phi == Pearson at the model's OWN
    #     converged mean AND OWN power.
    pearson_at_own_power = float(np.mean((y - mu_hat) ** 2 / mu_hat ** p_fit))
    assert phi_reported == pytest.approx(pearson_at_own_power, rel=0.03), (
        f"reported phi {phi_reported} must equal the converged-mean Pearson "
        f"{pearson_at_own_power} at the model's own power p={p_fit}"
    )

    # (3) The user-visible symptom: reported dispersion recovers the true 0.6.
    #     Under the biased power estimate this was ~0.676 (ratio 1.127 to the truth).
    assert abs(phi_reported - PHI_TRUE) < 0.04, (
        f"reported Tweedie phi={phi_reported:.4f} does not recover the true "
        f"phi={PHI_TRUE}; a ~13%-high phi inflates every SE/interval by ~6%"
    )
    # And it must NOT sit at the pre-fix inflated value.
    assert phi_reported < 0.65, (
        f"reported phi={phi_reported:.4f} is still near the pre-fix inflated 0.676"
    )
