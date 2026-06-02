"""Issue #612: a solved :class:`ManifoldSAE` must be genuinely usable.

After ``.fit(X)`` the torch module used to be a hybrid that could only forward
the exact training batch and could not survive a ``state_dict`` round-trip. This
test pins the fixed contract:

* ``module(X_new)`` for unseen rows returns finite reconstructions of the right
  shape (no ``NotImplementedError``) — out-of-sample forward re-encodes the new
  rows against the fitted decoder via the frozen-decoder inner solve.
* in-sample ``module(X)`` still reproduces the closed-form reconstruction.
* ``load_state_dict(module.state_dict())`` into a fresh module reproduces both
  the in-sample and out-of-sample reconstructions within tolerance.

The actual-fit calls are guarded so the test skips cleanly when the compiled
``gam-pyffi`` extension is unavailable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

from gamfit._binding import RustExtensionUnavailableError  # noqa: E402


def _make_synth(n: int, d: int, k: int, seed: int) -> np.ndarray:
    """Mixture-of-cosines synthetic with ``k`` latent atoms in ``R^d``."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
    bases = rng.standard_normal(size=(k, d))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True) + 1e-9
    amps = rng.uniform(0.5, 1.5, size=(n, k))
    x = (amps * np.cos(angles))[:, :, None] * bases[None, :, :]
    return x.sum(axis=1) + 0.05 * rng.standard_normal((n, d))


def _build_cfg(d: int) -> "gt.ManifoldSAEConfig":
    return gt.ManifoldSAEConfig(
        input_dim=d,
        n_atoms=3,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        basis_order=2,
        n_basis_per_atom=4,
        sparsity={"kind": "ibp_gumbel", "init_alpha": 1.0, "tau_schedule": "linear:4.0->1.0"},
        decoder={"ortho_weight": 0.0, "monotonicity_weight": 0.0},
        reml={"enabled": True, "select": ["lambda"]},
    )


def _fit_module(seed: int = 0) -> tuple["gt.ManifoldSAE", np.ndarray, np.ndarray]:
    """Build + fit a module, returning ``(module, X_train, X_new)``.

    Skips when the compiled extension is unavailable.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = 6
    X_train = _make_synth(n=48, d=d, k=3, seed=seed)
    X_new = _make_synth(n=17, d=d, k=3, seed=seed + 100)
    cfg = _build_cfg(d)
    module = gt.ManifoldSAE(cfg).double()
    x_train_t = torch.as_tensor(X_train, dtype=torch.float64)
    try:
        module.fit(x_train_t, max_iter=10, random_state=7)
    except RustExtensionUnavailableError:
        pytest.skip("gam-pyffi extension unavailable")
    return module, X_train, X_new


def test_oos_forward_returns_finite_reconstruction():
    module, _X_train, X_new = _fit_module()
    x_new_t = torch.as_tensor(X_new, dtype=torch.float64)

    out = module(x_new_t)  # must NOT raise NotImplementedError

    assert out.x_hat.shape == x_new_t.shape
    assert torch.isfinite(out.x_hat).all()
    # Per-token latents must align with the OOS batch size, not the training one.
    assert out.assignments.shape[0] == X_new.shape[0]
    assert out.positions.shape[0] == X_new.shape[0]
    assert torch.isfinite(out.assignments).all()
    assert torch.isfinite(out.positions).all()


def test_in_sample_forward_matches_closed_form():
    module, X_train, _X_new = _fit_module()
    x_train_t = torch.as_tensor(X_train, dtype=torch.float64)
    fit = module._last_fit
    assert fit is not None

    out = module(x_train_t)

    expected = np.asarray(fit.fitted, dtype=np.float64)
    np.testing.assert_allclose(out.x_hat.detach().cpu().numpy(), expected, rtol=0, atol=1e-9)


def test_state_dict_round_trip_reproduces_oos_and_in_sample():
    module, X_train, X_new = _fit_module()
    x_train_t = torch.as_tensor(X_train, dtype=torch.float64)
    x_new_t = torch.as_tensor(X_new, dtype=torch.float64)

    ref_in = module(x_train_t).x_hat.detach().cpu().numpy()
    ref_oos = module(x_new_t).x_hat.detach().cpu().numpy()

    # Fresh, unfitted module of the same config; its forward would otherwise use
    # the random eager encoder path. After loading the state_dict it must route
    # through the rebuilt closed-form fit.
    reloaded = gt.ManifoldSAE(_build_cfg(X_train.shape[1])).double()
    assert reloaded._last_fit is None
    reloaded.load_state_dict(module.state_dict())
    assert reloaded._last_fit is not None

    got_in = reloaded(x_train_t).x_hat.detach().cpu().numpy()
    got_oos = reloaded(x_new_t).x_hat.detach().cpu().numpy()

    np.testing.assert_allclose(got_in, ref_in, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got_oos, ref_oos, rtol=1e-7, atol=1e-7)


def test_unfitted_state_dict_round_trip_keeps_eager_path():
    """An unfitted module's blob is empty; a round trip must not fabricate a fit."""
    cfg = _build_cfg(6)
    src = gt.ManifoldSAE(cfg).double()
    assert src._last_fit is None
    assert src._fit_blob.numel() == 0

    dst = gt.ManifoldSAE(cfg).double()
    dst.load_state_dict(src.state_dict())
    assert dst._last_fit is None
    assert dst._fit_blob.numel() == 0
