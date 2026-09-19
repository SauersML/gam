"""The LR null model is the full model with one block at zero, at the same λ̂.

`Model.smooth_significance` compares the fitted model against the model with
the tested smooth's coefficients constrained to zero, and scores the statistic
against the null law of THAT constraint at the full fit's smoothing
parameters. The reduced model used to be refitted from scratch, re-running
REML for every surviving `λ`. When the tested term sits at its penalty null
space the two REML optima then differ only by the outer search's tolerance,
and that tolerance was the statistic: the audit's binomial replicate 14
published `p < 1.1e-237` for `s(x2)`, a term with no effect in the DGP, and
replicate 204 `p < 9e-54`.

Pinned here: on those replicates the null term's p-value is a resolved value
in `(1e-3, 1]`, and the strong term is still detected — the fix moves the
nested fit, not the power.

Replicates are seeded `default_rng(1000 + rep)` over the pyGAM audit's binomial
inference cell (bench/pygam_audit; bench/pvalue_calibration/pv-lr-refit).
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _binomial_cell(rep: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1000 + rep)
    X = rng.uniform(0, 1, (400, 3))
    eta = 1.5 * np.sin(2 * np.pi * X[:, 0]) + 0.6 * np.cos(2 * np.pi * X[:, 2])
    y = (rng.uniform(size=400) < 1 / (1 + np.exp(-eta))).astype(float)
    return dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)


@pytest.mark.parametrize("rep", [14, 204])
def test_a_null_term_is_not_scored_on_the_outer_search_tolerance(rep: int) -> None:
    data = _binomial_cell(rep)
    model = gamfit.fit(data, "y ~ s(x1) + s(x2) + s(x3)", family="binomial")
    rows = {r["name"]: r for r in model.smooth_significance(data)}

    null = rows["s(x2)"]
    assert null["unavailable_reason"] is None, null["unavailable_message"]
    assert null["p_value_upper_bound"] is None, (
        f"s(x2) (no effect in the DGP) published p < {null['p_value_upper_bound']:.3g} "
        f"from W = {null['statistic_lr']:.3g}"
    )
    # A calibrated null p-value is uniform; 1e-3 separates the defect (tens to
    # hundreds of orders below it) from the bottom 0.1% of a correct law.
    assert 1e-3 < null["p_value"] <= 1.0, (null["p_value"], null["statistic_lr"])

    strong = rows["s(x1)"]
    rejected = strong["p_value"] if strong["p_value"] is not None else strong["p_value_upper_bound"]
    assert rejected is not None and rejected < 1e-6, strong
