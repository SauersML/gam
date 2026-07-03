"""Regression for #2105: the reported Tweedie dispersion ``phi`` must equal the
converged-mean Pearson estimate AND recover the true dispersion — it was ~13%
high, inflating every Tweedie SE / interval by ~6%.

ROOT CAUSE (verified). The reported ``phi`` DID equal ``estimate_tweedie_phi_from_
eta(final_eta)`` — the ``refine_dispersion_at_converged_eta`` block lands correctly.
The bias was one level deeper: a bare ``family="tweedie"`` ESTIMATES the variance
power ``p`` by profile likelihood (#2026, mgcv ``tw()`` semantics), and that profile
was scored with the **saddlepoint** density (``tweedie_saddlepoint_loglik``). The
saddlepoint is asymptotically exact only in the many-jumps (large Poisson-rate)
limit; at the moderate rate of a typical Tweedie fit its missing ``O(1/lambda)``
normalizer biases the profile maximizer of ``p`` LOW (``p_hat ~ 1.33`` on ``p = 1.5``
data). Because the reported dispersion is the Pearson estimate
``phi_hat = sum w (y - mu)^2 / mu^p / sum w``, an under-estimated ``p`` inflates
``phi_hat`` (0.676 vs the true 0.600 here).

FIX (#2105). Profile ``p`` on the EXACT Jorgensen compound-Poisson-gamma series
(``tweedie_exact_loglik_total``), as mgcv's ``ldTweedie`` does. This recovers
``p_hat ~ 1.5`` and hence ``phi_hat ~ 0.6``.

This test asserts the user-visible symptom directly, plus the power that drives it:

  1. reported ``phi`` == Pearson dispersion at the model's OWN converged mean and
     OWN power (the self-consistency invariant the code documents), and
  2. reported ``phi`` recovers the true 0.6 within a tight band (the bug: 0.676), and
  3. the estimated power recovers the true 1.5 (the bug: ~1.33).
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


def _fit_bare_tweedie(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, N)
    mu = np.exp(0.7 + 0.5 * x)
    y = _tweedie_sample(rng, mu, PHI_TRUE, P_TRUE)
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ x", family="tweedie")
    return m, df, y


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_reported_phi_is_converged_eta_pearson_and_recovers_truth(seed):
    m, df, y = _fit_bare_tweedie(seed)
    payload = json.loads(m.dumps())

    reported = _find_key(payload, "EstimatedTweediePhi")
    assert reported, "fit must report an estimated Tweedie phi"
    phi_reported = reported[0]["phi"]

    # The power the model actually FIT (the last-written Tweedie response power,
    # not the pre-estimation placeholder). Take the value furthest from the
    # placeholder 1.5 fallback if present, else the sole value.
    powers = [d["p"] for d in _find_key(payload, "Tweedie") if isinstance(d, dict) and "p" in d]
    assert powers, "fit must carry a Tweedie variance power"
    p_hat = powers[-1]

    eta = np.asarray(
        m.predict(df, interval=0.9, return_type="pandas")["linear_predictor"]
    )
    mu_hat = np.exp(eta)

    # (1) Self-consistency invariant: reported phi == Pearson at the model's OWN
    #     converged mean AND OWN power. This held before the fix too; keep it as a
    #     guard so a future refresh regression cannot pass by chance.
    pearson_at_own_power = float(np.mean((y - mu_hat) ** 2 / mu_hat ** p_hat))
    assert phi_reported == pytest.approx(pearson_at_own_power, rel=0.03), (
        f"reported phi {phi_reported} must equal the converged-mean Pearson "
        f"{pearson_at_own_power} at the model's own power p_hat={p_hat}"
    )

    # (2) The estimated power must recover the truth (the driver of the bug):
    #     the saddlepoint profile returned ~1.33; the exact-series profile ~1.5.
    assert abs(p_hat - P_TRUE) < 0.08, (
        f"estimated Tweedie power p_hat={p_hat:.4f} is biased away from the true "
        f"p={P_TRUE} (the saddlepoint profile gave ~1.33)"
    )

    # (3) The user-visible symptom: reported dispersion recovers the true 0.6.
    #     Before the fix this was ~0.676 (ratio 1.127 to the truth).
    assert abs(phi_reported - PHI_TRUE) < 0.04, (
        f"reported Tweedie phi={phi_reported:.4f} does not recover the true "
        f"phi={PHI_TRUE}; a ~13%-high phi inflates every SE/interval by ~6%"
    )
    # And it must NOT sit at the pre-fix inflated value.
    assert phi_reported < 0.65, (
        f"reported phi={phi_reported:.4f} is still near the pre-fix inflated 0.676"
    )
