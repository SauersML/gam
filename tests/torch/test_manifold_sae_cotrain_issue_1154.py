"""Torch-facing regression for issue #1154 co-training wiring."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

import gamfit  # noqa: E402  (after importorskip)


def _circle(n: int, *, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64) + float(phase)
    return np.stack(
        [np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)],
        axis=1,
    )


def _stack_t_init(latents: dict[str, Any]) -> np.ndarray:
    coords = [np.asarray(block, dtype=np.float64) for block in latents["coords"]]
    return np.ascontiguousarray(np.stack(coords, axis=0))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def test_cotrained_torch_fit_surfaces_report_and_matches_sequential_circle() -> None:
    torch.manual_seed(1154)
    train = _circle(48)
    heldout = _circle(17, phase=0.013)
    cfg = gt.ManifoldSAEConfig(
        input_dim=2,
        n_atoms=1,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        n_basis_per_atom=5,
        sparsity={"kind": "ibp_gumbel", "init_alpha": 1.0, "tau_start": 0.5, "tau_min": 0.5},
        reml={"enabled": True, "select": ["lambda"], "max_iter": 8},
        dtype=torch.float64,
    )
    module = gt.ManifoldSAE(cfg).double()
    train_t = torch.as_tensor(train, dtype=torch.float64)

    initializers = module._closed_form_initializers(train_t)
    assert initializers["t_init"].shape == (1, train.shape[0], 1)
    assert initializers["a_init"].shape == (train.shape[0], 1)
    cotrained = module.fit(train_t, max_iter=8, random_state=1154)
    sequential = gamfit.sae_manifold_fit(
        X=train,
        K=1,
        d_atom=1,
        atom_topology="circle",
        atom_basis=cfg.closed_form_basis_kind(),
        assignment=cfg.closed_form_assignment(),
        schedule=cfg.sparsity.gumbel_schedule(),
        n_iter=8,
        random_state=1154,
    )

    assert cotrained.cotrain is not None
    assert module.cotrain == cotrained.cotrain
    assert cotrained.cotrain["n_encodes"] > 0
    assert cotrained.cotrain["recon_consistency"] <= 1.0e-2
    assert cotrained.cotrain["uncertified_fraction"] < 1.0

    cotrain_train_mse = _mse(cotrained.fitted, train)
    sequential_train_mse = _mse(sequential.fitted, train)
    assert cotrain_train_mse <= sequential_train_mse + 1.0e-2

    cotrain_heldout = cotrained.reconstruct(heldout)
    sequential_heldout = sequential.reconstruct(heldout)
    assert _mse(cotrain_heldout, heldout) <= _mse(sequential_heldout, heldout) + 1.0e-2

    exact = cotrained.converged_latents(heldout)
    warm = cotrained.converged_latents(
        heldout,
        t_init=_stack_t_init(exact),
        a_init=np.asarray(exact["logits"], dtype=np.float64),
    )
    np.testing.assert_allclose(warm["fitted"], exact["fitted"], rtol=1.0e-7, atol=1.0e-8)
    np.testing.assert_allclose(
        warm["assignments"],
        exact["assignments"],
        rtol=1.0e-7,
        atol=1.0e-8,
    )

    clone = gt.ManifoldSAE(cfg).double()
    clone.load_state_dict(module.state_dict())
    assert clone.cotrain == module.cotrain
    np.testing.assert_allclose(
        clone(torch.as_tensor(heldout, dtype=torch.float64)).x_hat.detach().numpy(),
        cotrain_heldout,
        rtol=1.0e-7,
        atol=1.0e-8,
    )
