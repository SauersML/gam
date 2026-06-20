"""compare_models / the REML evidence headline always prefers a model with a
pure-noise smooth added — it cannot tell a useful predictor from noise.

``compare_models`` ranks fits on the REML/LAML marginal-likelihood evidence
headline (the per-fit ``reml_score``, which numerically equals the model's own
``Model.evidence`` property; the score table the API returns carries only
``reml_score`` / ``delta_reml`` / ``bayes_factor_best_over_model`` /
``effective_dof``).  The module behind it
(``src/inference/model_comparison.rs``) advertises "honest, calibrated model
comparison" aimed squarely at the "random-effect-vs-null, is-a-wiggle-real"
question.

The defect
----------
Take a response that depends only on ``x`` and a covariate ``z`` drawn
*independently* of the response (pure noise).  Compare

    small:  y ~ s(x)
    big:    y ~ s(x) + s(z)

``compare_models([small, big])`` selects ``big`` on **100%** of seeds, with
Bayes factors of ~7-60 "in favour of" adding the noise smooth.  A correct
marginal-likelihood comparison must Occam-penalise the spurious term, so for a
genuinely-null ``z`` the *smaller* model should win the majority of the time.

The model's own evidence numbers show the failure directly: ``small`` and
``big`` have ``reml_score`` like 151.28 vs 148.34 — the noise-augmented model
gets the *better* (lower) score every time, by a near-constant margin of ~3
nats — so it is the evidence/REML score itself that fails to penalise the extra
term, not a ranking-direction or normalisation disagreement between entry
points (``Model.evidence`` and ``Model.bayes_factor_vs`` agree with
``compare_models`` here, all preferring ``big``).

The direction is confirmed by a relevant-``z`` control: when ``z`` genuinely
drives the response, ``big`` *should* and *does* win — so the tool's
optimisation direction is right; it simply selects the bigger model in *both*
cases.  In other words, ``compare_models`` cannot distinguish a useful
predictor from pure noise: it always says "add it".

This is the model-evidence / comparison sibling of the per-smooth-term Wald
false positive in the summary table.
Related: #1360.

What this test asserts
----------------------
A correct model-comparison tool must *discriminate*:

* with ``z`` pure noise, it must NOT select the noise-augmented model on
  (nearly) every seed — the big-model selection rate must stay well below 0.5
  (a calibrated Occam comparison sits much lower);
* with ``z`` genuinely relevant, it MUST select the bigger model (power), so a
  trivial "always pick the smaller model" change cannot satisfy the test.

Currently the noise-case big-selection rate is 1.0, so the calibration
assertion fails.  When the evidence comparison is fixed it should drop well
below 0.5 while the relevant-case power assertion keeps passing.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


N_NOISE = 20
N_RELEVANT = 10
N = 700


def _big_selection_rate(z_relevant: bool, n_seeds: int) -> tuple[float, list[float]]:
    """Fraction of seeds on which compare_models selects ``y ~ s(x) + s(z)``
    over ``y ~ s(x)``. With z_relevant=False, z is drawn independently of y."""
    big_wins = 0
    bayes_factors: list[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(3_000 + seed)
        x = rng.uniform(0.0, 1.0, N)
        z = rng.uniform(0.0, 1.0, N)
        signal_z = 1.5 * np.cos(5.0 * z) if z_relevant else 0.0
        y = np.sin(3.0 * x) + signal_z + 0.3 * rng.standard_normal(N)
        data = {"x": x, "z": z, "y": y}
        small = gamfit.fit(data, "y ~ s(x)")
        big = gamfit.fit(data, "y ~ s(x) + s(z)")
        result = gamfit.compare_models([small, big], names=["small", "big"])
        if result["winner"] == "big":
            big_wins += 1
            # Bayes factor of the winner (big) over small, for diagnostics.
            for row in result["score_table"]:
                if row["name"] == "small":
                    bayes_factors.append(float(row["bayes_factor_best_over_model"]))
    return big_wins / n_seeds, bayes_factors


def test_compare_models_does_not_always_prefer_a_pure_noise_smooth() -> None:
    # Power control: when z genuinely drives the response, the bigger model is
    # correct and must be selected. This keeps the calibration check below
    # well-posed (a fix cannot just always choose the smaller model).
    rate_relevant, _ = _big_selection_rate(z_relevant=True, n_seeds=N_RELEVANT)
    assert rate_relevant >= 0.8, (
        "compare_models failed to select the bigger model when z genuinely "
        f"drives the response (selected big on only {rate_relevant:.0%} of "
        f"{N_RELEVANT} seeds); the comparison has lost power"
    )

    # Calibration: z is drawn independently of the response, so the smooth s(z)
    # is null and a correct marginal-likelihood comparison must Occam-penalise
    # it. The bigger (noise-augmented) model should therefore win only a
    # minority of the time.
    rate_noise, bfs = _big_selection_rate(z_relevant=False, n_seeds=N_NOISE)
    median_bf = float(np.median(bfs)) if bfs else float("nan")
    assert rate_noise <= 0.5, (
        "compare_models selected the model with a PURE-NOISE smooth s(z) added "
        f"(z drawn independently of y) on {rate_noise:.0%} of {N_NOISE} seeds "
        f"(median Bayes factor 'big over small' = {median_bf:.1f}). A calibrated "
        "evidence comparison Occam-penalises a null term and should prefer the "
        "smaller model the majority of the time; instead the REML evidence "
        "headline improves whenever a spurious smooth is added (see "
        "src/inference/model_comparison.rs and the REML/LAML score it ranks on)."
    )
