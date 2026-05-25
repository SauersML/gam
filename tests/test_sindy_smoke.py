"""Smoke test for :class:`gamfit.SINDyAtoms` on the Lorenz-63 system.

Integrates the Lorenz attractor with parameters (σ, ρ, β) = (10, 28, 8/3),
constructs the analytic derivatives ``ẋ = σ(y−x)``, ``ẏ = x(ρ−z) − y``,
``ż = xy − βz``, and runs STLSQ with the canonical SINDy library
``['const', 'id', 'product']``. Asserts the recovered coefficient matrix
matches the known Lorenz coefficients within 0.5 in absolute value
(well within Brunton 2016's recovery tolerance for clean derivatives).
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _lorenz_rhs(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array(
        [sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=np.float64
    )


def _integrate_lorenz(
    n: int, dt: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """RK4 integration of Lorenz-63 producing both trajectory and analytic
    derivatives at each sampled row."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rng = np.random.default_rng(seed)
    state = np.array([-8.0, 7.0, 27.0], dtype=np.float64) + 0.1 * rng.normal(size=3)
    traj = np.empty((n, 3), dtype=np.float64)
    dz = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        traj[i] = state
        dz[i] = _lorenz_rhs(state, sigma, rho, beta)
        k1 = _lorenz_rhs(state, sigma, rho, beta)
        k2 = _lorenz_rhs(state + 0.5 * dt * k1, sigma, rho, beta)
        k3 = _lorenz_rhs(state + 0.5 * dt * k2, sigma, rho, beta)
        k4 = _lorenz_rhs(state + dt * k3, sigma, rho, beta)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return traj, dz


def test_sindy_recovers_lorenz_63() -> None:
    pytest.importorskip("gamfit._rust", reason="Rust extension required for SINDy")
    n = 2000
    dt = 0.01
    traj, dz = _integrate_lorenz(n, dt, seed=0)

    sindy = gamfit.SINDyAtoms(
        library=["const", "id", "product"],
        sparsity={"kind": "stlsq", "lam": 1.0e-3},
        threshold={"kind": "stlsq", "tol": 0.05, "max_rounds": 10},
        state_dim=3,
    )
    sindy.fit(traj, dz_dt=dz)

    assert sindy.theta is not None
    assert sindy.theta.shape == (3, len(sindy.term_names))
    # Library order: ['1', 'x', 'y', 'z', 'xy', 'xz', 'yz']
    assert sindy.term_names == ["1", "x", "y", "z", "xy", "xz", "yz"]

    coef = sindy.theta
    # dx/dt = -10x + 10y
    assert abs(coef[0, sindy.term_names.index("x")] - (-10.0)) < 0.5
    assert abs(coef[0, sindy.term_names.index("y")] - 10.0) < 0.5
    # dy/dt = 28x - y - xz
    assert abs(coef[1, sindy.term_names.index("x")] - 28.0) < 0.5
    assert abs(coef[1, sindy.term_names.index("y")] - (-1.0)) < 0.5
    assert abs(coef[1, sindy.term_names.index("xz")] - (-1.0)) < 0.5
    # dz/dt = xy - (8/3) z
    assert abs(coef[2, sindy.term_names.index("xy")] - 1.0) < 0.5
    assert abs(coef[2, sindy.term_names.index("z")] - (-8.0 / 3.0)) < 0.5

    eqs = sindy.equations_human_readable(["x", "y", "z"])
    assert len(eqs) == 3
    assert eqs[0].startswith("dx/dt = ")
    assert eqs[1].startswith("dy/dt = ")
    assert eqs[2].startswith("dz/dt = ")
