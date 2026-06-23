"""Bug hunt: ``bounded(x, min, max)`` posterior draws escape ``[min, max]`` on a
Gaussian model — the dedicated in-bounds latent sampler is dead code for the
default (and most common) family.

``bounded(x, min=a, max=b)`` constrains its coefficient to the open interval
``(a, b)`` by fitting through an interval transform ``β = a + (b−a)·sigmoid(θ)``
of an unconstrained latent ``θ``. The posterior is Gaussian on the *latent*
scale, so correct draws are ``θ ~ N(θ_mode, H_latent⁻¹)`` pushed forward through
the interval map and therefore lie *strictly inside* ``[a, b]``. The engine has
exactly this routine — ``sample_standard_bounded`` →
``sample_bounded_latent_posterior_internal`` — whose own docstring promises
"user-scale draws that always lie strictly inside the interval"
(``src/inference/sample.rs:437-451``).

For a Gaussian-identity model that path is never reached. ``sample_standard``
(``src/inference/sample.rs:407``) returns the unconstrained
``laplace_gaussian_fallback`` at the very top (``:415-417``) — *before* the
``has_bounded`` dispatch at ``:452-458``. The fallback draws a plain Gaussian on
the user scale, centred on the in-bounds mode, which spills mass outside
``[a, b]``. Because ``bounded()`` is only accepted for Gaussian and binomial
families (Poisson/Gamma reject it at fit time) and Gaussian is the default, the
latent sampler effectively only ever runs for binomial — so the documented
in-bounds guarantee is silently broken for the dominant use case.

This test contrasts the two families on the identical ``bounded(x, min=0,
max=1)`` term:

* **Gaussian** (default): asserts every posterior draw of the bounded
  coefficient lies in ``[0, 1]``. This currently FAILS — a sizeable fraction of
  draws fall outside (observed range overshoots ~1.7 and dips below 0).
* **Binomial** (control): the same assertion already holds, isolating the defect
  to the Gaussian early-return ordering rather than to the bounded machinery
  itself.

When the ``has_bounded`` check runs before the Gaussian-identity shortcut, the
Gaussian assertion holds without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _bounded_slope_draws(model: Any, frame: pd.DataFrame) -> np.ndarray:
    samples = model.sample(frame, samples=4000, chains=2, seed=1)
    names = list(samples.coefficient_names)
    slope_idx = names.index("beta_1") if "beta_1" in names else len(names) - 1
    return np.asarray(samples.to_numpy())[:, slope_idx]


def test_binomial_bounded_posterior_stays_in_interval_control() -> None:
    """Control: the binomial path already routes through the latent sampler."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.5 * x)))
    y = (rng.uniform(0.0, 1.0, n) < p).astype(float)
    frame = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(frame, "y ~ bounded(x, min=0, max=1)", family="binomial")
    draws = _bounded_slope_draws(model, frame)
    frac_outside = float(((draws < 0.0) | (draws > 1.0)).mean())
    assert frac_outside == 0.0, (
        "binomial bounded() posterior should already lie strictly in [0, 1]; "
        f"got {frac_outside:.1%} outside"
    )


def test_gaussian_bounded_posterior_stays_in_interval() -> None:
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    # In-bounds true slope (0.8) so the optimum is interior and the posterior is a
    # well-defined latent-Gaussian pushed through the interval map — every draw
    # must land in (0, 1).
    y = 2.0 + 0.8 * x + rng.standard_normal(n) * 0.1
    frame = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(frame, "y ~ bounded(x, min=0, max=1)")
    fitted_slope = model.summary().coefficients[1]["estimate"]
    assert 0.0 < fitted_slope < 1.0, f"fitted slope should be interior; got {fitted_slope}"

    draws = _bounded_slope_draws(model, frame)
    frac_outside = float(((draws < 0.0) | (draws > 1.0)).mean())
    assert frac_outside < 0.005, (
        "Gaussian bounded(x, min=0, max=1) posterior draws must lie inside [0, 1]; "
        f"got {frac_outside:.1%} outside (range [{draws.min():.4f}, {draws.max():.4f}])"
    )
