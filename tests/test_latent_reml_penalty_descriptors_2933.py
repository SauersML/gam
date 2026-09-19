"""#2933 F02: gamfit's latent REML fits hand their analytic-penalty descriptors to
Rust as JSON text, and the backward companion carries their gradient.

`gaussian_reml_fit_latent` and `gaussian_reml_fit_latent_backward` take
``penalties=`` wrappers, and the Rust entry points take the descriptors as
``analytic_penalties: Option<String>``. The forward adds each descriptor's energy
to ``reml_score``; the backward adds its target gradient to ``grad_t``. ARD at
descriptor weight ``w`` (rho = 0) has the closed forms

    P(t) = sum_j [ 1/2 * w * ||t[:, j]||^2 - (n / 2) * log(w) ],    dP/dt = w * t,

computed here independently of both routes, so both comparisons are free to
disagree.
"""

from __future__ import annotations

import numpy as np

import gamfit


def _problem() -> tuple[np.ndarray, tuple, dict]:
    rng = np.random.default_rng(2933)
    n, dim, k = 16, 2, 6
    t = rng.uniform(0.2, 0.9, (n, dim))
    centers = rng.uniform(0.1, 0.95, (k, dim))
    y = np.sin(3.0 * t[:, :1]) + 0.3 * t[:, 1:] + 0.05 * rng.normal(size=(n, 1))
    return t, (y, n, dim, centers, np.eye(k)), dict(m=2)


def test_latent_reml_fit_prices_ard_descriptor_energy_2933() -> None:
    t, args, kwargs = _problem()
    w = 0.7
    n, dim = t.shape
    bare = gamfit.reml.gaussian_reml_fit_latent(t.ravel(), *args, **kwargs)
    priced = gamfit.reml.gaussian_reml_fit_latent(
        t.ravel(), *args, penalties=[gamfit.penalties.ARDPenalty(weight=w)], **kwargs
    )
    energy = sum(
        0.5 * w * float(np.sum(t[:, j] ** 2)) - 0.5 * n * float(np.log(w))
        for j in range(dim)
    )
    added = float(priced["reml_score"]) - float(bare["reml_score"])
    # The fit never sees the penalty, so the two scores differ only by the priced
    # energy, up to the roundoff of one addition at the score's magnitude.
    tolerance = 1e-10 * max(1.0, abs(float(bare["reml_score"])))
    print(
        f"[#2933 F02 gamfit JSON] ARD energy = {energy:.12e}, reml_score added = "
        f"{added:.12e}, tol = {tolerance:.3e}"
    )
    assert abs(energy) > 1e6 * tolerance, "premise: the ARD energy must be resolvable"
    assert abs(added - energy) <= tolerance, (
        f"reml_score must add the ARD energy: {added:.12e} vs {energy:.12e}"
    )


def test_latent_reml_backward_carries_ard_descriptor_gradient_2933() -> None:
    t, args, kwargs = _problem()
    w = 0.7
    bare = np.asarray(
        gamfit.reml.gaussian_reml_fit_latent_backward(
            t.ravel(), *args, grad_reml_score=1.0, **kwargs
        )["grad_t"]
    )
    priced = np.asarray(
        gamfit.reml.gaussian_reml_fit_latent_backward(
            t.ravel(),
            *args,
            grad_reml_score=1.0,
            penalties=[gamfit.penalties.ARDPenalty(weight=w)],
            **kwargs,
        )["grad_t"]
    )
    leg = priced - bare
    expected = w * t
    tolerance = 1e-10 * max(1.0, float(np.abs(bare).max()))
    print(
        f"[#2933 F02 gamfit JSON] max|w t| = {np.abs(expected).max():.6e}, "
        f"max|leg - w t| = {np.abs(leg - expected).max():.3e}, tol = {tolerance:.3e}"
    )
    assert np.abs(expected).max() > 1e6 * tolerance, (
        "premise: the ARD gradient must be resolvable"
    )
    np.testing.assert_allclose(leg, expected, rtol=0.0, atol=tolerance)
