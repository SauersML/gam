"""Facade defaults must pass the native front door's own validation (#2323).

``sae_manifold_fit``'s keyword defaults are the documented public contract
(``docs/manifold-sae.md`` lists ``sparsity_weight`` default ``1.0``), and the
engine's own seed default is ``sparsity_strength: 1.0``. Before the fix, the
facade materialized ``0.0`` and every default-argument call — including 10
shipped examples — died at input validation with ``sparsity_strength must be
finite and positive; got 0`` without ever reaching the solver.

These tests do NOT assert the fit converges (solver behavior is owned
elsewhere); they assert the failure mode, if any, is not the facade's own
default failing validation.
"""

from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _tiny_circle(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.c_[np.cos(t), np.sin(t)] + 0.05 * rng.standard_normal((n, 2))


def test_default_arguments_pass_input_validation() -> None:
    x = _tiny_circle()
    try:
        gamfit.sae_manifold_fit(x, K=1, d_atom=1, atom_topology="circle", n_iter=2)
    except Exception as error:  # noqa: BLE001 — asserting on the message either way
        message = str(error)
        assert "sparsity_strength must be finite and positive" not in message, (
            "the facade's own default failed the native front door's validation: "
            + message
        )


def test_facade_default_matches_documented_and_engine_default() -> None:
    import inspect

    signature = inspect.signature(gamfit.sae_manifold_fit)
    default = signature.parameters["sparsity_weight"].default
    # docs/manifold-sae.md documents 1.0; SaeManifoldFitSeed's Default is
    # sparsity_strength: 1.0. The facade must agree with both.
    assert default == 1.0, default
