"""Regression tests for issue #627: the latent coordinate must be *estimated*.

`gaussian_reml_fit_latent` is a single ``β | t`` solve — it never moves ``t``,
so its fit quality is whatever the *input* ``t`` already encodes. Issue #627
asks for a latent-optimizing entry point that actually recovers ``t`` from a
poor init. `gamfit.gaussian_reml_optimize_latent` provides it: a spectral
(Laplacian-eigenmaps) warm start plus Riemannian refinement.

These tests pin the contract on the canonical oracle from the issue (a shuffled
parabola — a 1-D curve in 2-D whose intrinsic coordinate is the arc position):

* the recovered fit reaches R² ≈ 1 from a *random* init (the failing case);
* recovery is init-independent (random / true / reversed converge alike — the
  property the issue says a real latent fit must have);
* the recovered coordinate matches the true order up to monotone gauge
  (|Spearman| ≈ 1) — a different-angle guard so a future regression cannot pass
  by overfitting reconstruction with a garbage coordinate;
* the recovered coordinate is actually returned (``t`` / ``latent`` / ``t_flat``);
* ``init="caller"`` is a pure local solve (no seed), so a bad ``t`` stays bad and
  a good ``t`` stays good — proving the seed is what does the work.

``gaussian_reml_optimize_latent`` only ever returns a *converged* fit (SPEC
rule 20): non-stationarity raises ``gamfit.RemlConvergenceError`` carrying a
``checkpoint_t`` resume checkpoint. The helper below implements the sanctioned
checkpoint/resume loop — continue from the checkpoint (``init="caller"``) with
a doubled iteration budget until stationarity is certified — so these quality
assertions always run on an honestly converged fit.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

M = 14
N = 180


def _optimize_to_convergence(*args, max_iter, max_rounds=6, **kwargs):
    """Resume-loop wrapper for the SPEC-20 convergence contract.

    Returns ``(payload, first_init)`` where ``first_init`` is the init mode the
    FIRST attempt reported (resumes are always ``init="caller"`` from the
    checkpoint, so the payload's own ``init`` reflects the last round).
    """
    budget = int(max_iter)
    first_init = None
    for _ in range(max_rounds):
        try:
            out = gamfit.gaussian_reml_optimize_latent(*args, max_iter=budget, **kwargs)
            return out, (out["init"] if first_init is None else first_init)
        except gamfit.RemlConvergenceError as exc:
            if first_init is None:
                first_init = str(exc.init)
            kwargs["t"] = np.asarray(exc.checkpoint_t, dtype=float).reshape(-1)
            kwargs["init"] = "caller"
            budget *= 2
    # Last round propagates the typed error as an honest failure.
    out = gamfit.gaussian_reml_optimize_latent(*args, max_iter=budget, **kwargs)
    return out, first_init


def _abs_spearman(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return abs(np.corrcoef(ra, rb)[0, 1])


def _shuffled_parabola(seed, n=N):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, n, endpoint=False)
    u = 2.0 * base - 1.0
    y = np.c_[u, u**2 - 0.33]
    perm = rng.permutation(n)
    return y[perm], base[perm]


def _centers_and_penalty():
    # #1512: gamfit.duchon_function_norm_penalty now requires 2D centers with
    # shape (K, d) and raises "centers must be 2D ... got 1D" on the bare
    # np.linspace 1D vector this orphaned test used to pass. Build the (K, 1)
    # column up front and feed the same 2D array to both the penalty and the
    # latent fit below.
    centers = np.linspace(0.0, 1.0, M).reshape(-1, 1)
    penalty = np.asarray(gamfit.duchon_function_norm_penalty(centers, m=2))
    return centers, penalty


def _r2(y, fitted):
    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    return 1.0 - ((y - fitted) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def test_optimizer_recovers_coordinate_from_random_init():
    """Random init (the #627 failure) must converge to R² ≈ 1 and recover t."""
    y, true_t = _shuffled_parabola(seed=0)
    centers, penalty = _centers_and_penalty()
    rng = np.random.default_rng(123)

    out, first_init = _optimize_to_convergence(
        y, N, 1, centers, penalty,
        t=rng.random(N), m=2, max_iter=120,
    )

    assert _r2(y, out["fitted"]) >= 0.99
    assert _abs_spearman(out["t"], true_t) >= 0.95
    assert first_init == "spectral"


def test_recovery_is_initialization_independent():
    """A real latent fit reaches the same optimum from any init (issue text).

    The single ``β | t`` solve gives R² ≈ 1 for the true order but R² ≈ 0 for a
    random order; the optimizer must erase that dependence.
    """
    y, true_t = _shuffled_parabola(seed=1)
    centers, penalty = _centers_and_penalty()

    # The bug, pinned: the forward primitive does not move t.
    fixed_true = gamfit.gaussian_reml_fit_latent(true_t, y, N, 1, centers, penalty, m=2)
    rng = np.random.default_rng(7)
    fixed_rand = gamfit.gaussian_reml_fit_latent(rng.random(N), y, N, 1, centers, penalty, m=2)
    assert _r2(y, fixed_true["fitted"]) >= 0.99
    assert _r2(y, fixed_rand["fitted"]) <= 0.5  # random order is not fit

    # The fix: every init converges to the same high-quality optimum.
    inits = {
        "true": true_t,
        "random": rng.random(N),
        "reversed": true_t[::-1].copy(),
    }
    for name, t0 in inits.items():
        out = gamfit.gaussian_reml_optimize_latent(
            y, N, 1, centers, penalty, t=t0, m=2, max_iter=120,
        )
        assert _r2(y, out["fitted"]) >= 0.99, f"init={name} failed to recover"
        assert _abs_spearman(out["t"], true_t) >= 0.95, f"init={name} bad coordinate"


def test_recovered_latent_is_returned_with_shape():
    y, _ = _shuffled_parabola(seed=2)
    centers, penalty = _centers_and_penalty()

    out = gamfit.gaussian_reml_optimize_latent(y, N, 1, centers, penalty, m=2, max_iter=60)

    for key in ("t", "latent", "t_flat"):
        assert key in out, f"missing recovered-coordinate key {key!r}"
    assert np.asarray(out["t"]).shape == (N, 1)
    assert np.asarray(out["latent"]).shape == (N, 1)
    assert np.asarray(out["t_flat"]).shape == (N,)
    # t is optional: omitting it must not change that a coordinate is recovered.
    assert _r2(y, out["fitted"]) >= 0.99


def test_caller_init_is_a_pure_local_solve():
    """init="caller" disables the seed: bad t stays bad, good t stays good."""
    y, true_t = _shuffled_parabola(seed=3)
    centers, penalty = _centers_and_penalty()
    rng = np.random.default_rng(99)

    bad = gamfit.gaussian_reml_optimize_latent(
        y, N, 1, centers, penalty, t=rng.random(N), m=2, init="caller", max_iter=120,
    )
    good = gamfit.gaussian_reml_optimize_latent(
        y, N, 1, centers, penalty, t=true_t, m=2, init="caller", max_iter=60,
    )

    assert bad["init"] == "caller"
    # A near-correct warm start is preserved (and is high quality).
    assert _r2(y, good["fitted"]) >= 0.99
    # The local solve from a random order cannot match the seeded recovery; this
    # is exactly why the spectral seed is the default.
    assert _r2(y, bad["fitted"]) < _r2(y, good["fitted"])
