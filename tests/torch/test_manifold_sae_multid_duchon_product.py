"""Multi-dimensional (intrinsic_rank > 1) Duchon product patch in torch ManifoldSAE.

The forward ``_basis_rust`` path used to refuse *every* ``intrinsic_rank > 1``
configuration on the claim that the ``bspline``/``duchon`` ``basis_with_jet``
kernels are 1-D. That was stale for Duchon: the Rust ``duchon_basis_with_jet``
kernel is fully ``d``-dimensional — it takes ``(N, d)`` points and ``(K, d)``
centers and returns a ``(N, M, d)`` per-axis Jacobian. So a flat
``atom_manifold='product'`` patch with ``intrinsic_rank=2`` and
``atom_basis='duchon'`` is genuinely fittable by backprop, with the second
intrinsic coordinate carrying real gradient (not silently dead).

This test plants a sum of genuinely 2-D parametric surface patches in ``R^D``,
trains the SAE by backprop through the multi-d Duchon forward, and asserts the
reconstruction explains the data AND that the second intrinsic axis is *live*
(its encoder gradient is non-zero). It also pins the two honest refusals that
remain: ``cylinder`` (no topology-faithful periodic torch kernel) and
``bspline`` with ``intrinsic_rank > 1`` (the bspline kernel is genuinely 1-D).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def _planted_surface_dataset(
    *, n: int, d_ambient: int, n_features: int, seed: int
) -> np.ndarray:
    """Sparse sum of genuinely 2-D surface patches in mutually-orthogonal subspaces.

    Each feature owns a distinct 3-D ambient subspace and contributes a smooth
    surface ``s_f(u, v)`` (a 2-parameter patch, so BOTH intrinsic coordinates
    matter). Returns ``X`` of shape ``(n, D)``.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d_ambient, d_ambient)))
    subspaces = [q[:, 3 * f : 3 * f + 3] for f in range(n_features)]

    def _surface(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        # A 2-D coordinate patch embedded in 3 ambient dims; every output axis
        # depends on both u and v so neither intrinsic coordinate is redundant.
        return np.stack([u, v, u * v + 0.5 * (u * u - v * v)], axis=-1)

    active = (rng.random((n, n_features)) < 0.4).astype(np.float64)
    for f in range(n_features):
        if active[:, f].sum() < n // 6:
            idx = rng.choice(n, size=n // 6, replace=False)
            active[idx, f] = 1.0
    u = rng.random((n, n_features))
    v = rng.random((n, n_features))
    x = np.zeros((n, d_ambient), dtype=np.float64)
    for f in range(n_features):
        coord = _surface(u[:, f], v[:, f])  # (n, 3)
        contrib = coord @ subspaces[f].T  # (n, D)
        x += active[:, f : f + 1] * contrib
    x += 0.02 * rng.standard_normal((n, d_ambient))
    return x


def _build_product(d_ambient: int, n_atoms: int) -> Any:
    cfg = gt.ManifoldSAEConfig(
        input_dim=d_ambient,
        n_atoms=n_atoms,
        intrinsic_rank=2,
        atom_manifold="product",
        atom_basis="duchon",
        basis_order=2,
        n_basis_per_atom=12,
        sparsity={
            "kind": "softmax_topk",
            "target_k": 2,
            "tau_start": 1.0,
            "tau_min": 0.1,
            "tau_steps": 300,
        },
        reml={"enabled": True},
    )
    return gt.ManifoldSAE(cfg).double()


def test_multid_product_forward_fits_and_reconstructs() -> None:
    """A 2-D Duchon product patch fits + reconstructs via the multi-d forward."""
    torch.manual_seed(0)
    D, F = 24, 4
    x_np = _planted_surface_dataset(n=2000, d_ambient=D, n_features=F, seed=0)
    x = torch.as_tensor(x_np, dtype=torch.float64)

    sae = _build_product(D, n_atoms=F + 2)

    # The forward must produce a finite reconstruction with the right shape and
    # the expected per-atom intrinsic dimension (2 columns of positions).
    out0 = sae(x[:8])
    assert out0.x_hat.shape == (8, D)
    assert out0.positions.shape == (8, F + 2, 2)
    assert torch.isfinite(out0.x_hat).all()

    g = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    nrows = x.shape[0]
    for _ in range(1200):
        idx = torch.randint(0, nrows, (128,), generator=g)
        xb = x[idx]
        out = sae(xb)
        recon = ((out.x_hat - xb) ** 2).mean()
        sparsity = sae.sparsity_penalty(out.gate)
        loss = recon + 1e-3 * sparsity
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.sparsity.advance_temperature()

    with torch.no_grad():
        out = sae(x)
    ev = 1.0 - float(((out.x_hat - x) ** 2).mean() / x.var())
    assert ev > 0.5, f"multi-d Duchon product reconstruction collapsed: EV={ev:.3f}"


def test_multid_product_second_axis_is_live() -> None:
    """The second intrinsic coordinate carries real gradient (not silently dead).

    The stale refusal's failure mode was a model where ``∂x̂/∂t_j ≡ 0`` for the
    second intrinsic axis. With the genuine multi-d Duchon jet the reconstruction
    depends on BOTH axes, so a gradient pushed back to the encoder's second-axis
    output must be non-negligible.
    """
    torch.manual_seed(1)
    D, F = 24, 3
    x_np = _planted_surface_dataset(n=600, d_ambient=D, n_features=F, seed=1)
    x = torch.as_tensor(x_np, dtype=torch.float64)
    sae = _build_product(D, n_atoms=F + 1)

    out = sae(x)
    loss = (out.x_hat ** 2).mean()
    loss.backward()

    # The encoder linear maps to F*(d+1) outputs laid out as [positions(F*d),
    # amplitudes(F)]; the second intrinsic axis occupies columns {f*d + 1}.
    enc = sae.encoder
    assert isinstance(enc, torch.nn.Linear)
    grad = enc.weight.grad
    assert grad is not None
    d = 2
    second_axis_rows = [f * d + 1 for f in range(F)]
    second_axis_grad_norm = float(grad[second_axis_rows, :].abs().sum())
    assert second_axis_grad_norm > 1e-8, (
        "second intrinsic axis is dead (zero encoder gradient) — the multi-d "
        f"Duchon jet did not reach axis 1: grad_norm={second_axis_grad_norm:.3e}"
    )


def test_cylinder_forward_refused_with_accurate_message() -> None:
    """Cylinder forward stays refused (no topology-faithful periodic torch kernel)."""
    cfg = gt.ManifoldSAEConfig(
        input_dim=16,
        n_atoms=3,
        intrinsic_rank=2,
        atom_manifold="cylinder",
        atom_basis="duchon",
        n_basis_per_atom=10,
    )
    sae = gt.ManifoldSAE(cfg).double()
    x = torch.zeros((5, 16), dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="cylinder.*topology-faithful"):
        sae(x)


def test_bspline_multid_refused() -> None:
    """B-spline with intrinsic_rank > 1 stays refused (the bspline kernel is 1-D)."""
    cfg = gt.ManifoldSAEConfig(
        input_dim=16,
        n_atoms=3,
        intrinsic_rank=2,
        atom_manifold="product",
        atom_basis="bspline",
        n_basis_per_atom=10,
    )
    sae = gt.ManifoldSAE(cfg).double()
    x = torch.zeros((5, 16), dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="bspline.*1-D"):
        sae(x)
