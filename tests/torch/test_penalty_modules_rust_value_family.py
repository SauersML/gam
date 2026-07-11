"""RED tests for issue #231: every torch penalty module must expose `rust_value`.

The closed-form helper used inside `forward` (`_call_rust_value_grad`) is not
exposed as a module method, but tests / external callers compare against
`module.rust_value(latent_np, **kwargs_np)`. These tests pin the contract for
the entire family of penalty modules in `gamfit/torch/penalties.py`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

from gamfit.torch.penalties import _call_rust_value_grad  # type: ignore[attr-defined]


def _latent() -> torch.Tensor:
    return torch.tensor(
        [[0.2, -0.1, 0.7], [0.5, 0.3, -0.4], [-0.2, 0.8, 0.1]],
        dtype=torch.float64,
    )


def _basis() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64)


def _logits() -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.as_tensor(rng.standard_normal((4, 3)), dtype=torch.float64)


def _aux() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(3, 2)


# ---------------------------------------------------------------------------
# Per-module rust_value tests. Each must:
#   1. expose `rust_value` as a callable method,
#   2. accept the same numpy-array contract as the existing closed-form test,
#   3. return a float equal (atol=1e-10) to the corresponding forward value.
# ---------------------------------------------------------------------------


def _check_rust_value_matches_forward(
    name: str,
    module: torch.nn.Module,
    primary: torch.Tensor,
    extra_tensor_kwargs: dict[str, torch.Tensor] | None = None,
) -> None:
    extra_tensor_kwargs = extra_tensor_kwargs or {}
    value = module(primary, **extra_tensor_kwargs)
    assert bool(torch.isfinite(value)), f"{name}.forward produced non-finite value"

    assert hasattr(module, "rust_value"), (
        f"{name} must expose a `rust_value` method (issue #231)"
    )

    primary_np = primary.detach().cpu().numpy()
    extra_np = {k: v.detach().cpu().numpy() for k, v in extra_tensor_kwargs.items()}
    rust_value = float(module.rust_value(primary_np, **extra_np))

    np.testing.assert_allclose(
        float(value.detach().cpu().item()),
        rust_value,
        rtol=0.0,
        atol=1e-10,
        err_msg=f"{name}.rust_value disagrees with {name}.forward",
    )


def test_ard_penalty_rust_value() -> None:
    _check_rust_value_matches_forward("ARDPenalty", gt.ARDPenalty(weight=1.3), _latent())


def test_isometry_penalty_rust_value() -> None:
    _check_rust_value_matches_forward(
        "IsometryPenalty",
        gt.IsometryPenalty(weight=0.8),
        _latent(),
        {"basis": _basis()},
    )


def test_smooth_threshold_penalty_rust_value() -> None:
    thresholds = torch.full((3,), 0.05, dtype=torch.float64)
    _check_rust_value_matches_forward(
        "SmoothThresholdPenalty",
        gt.SmoothThresholdPenalty(thresholds, weight=1.1),
        _latent(),
    )


def test_block_orthogonality_penalty_rust_value() -> None:
    _check_rust_value_matches_forward(
        "BlockOrthogonalityPenalty",
        gt.BlockOrthogonalityPenalty(groups=[[0, 1], [2]], weight=0.7, n_eff=3),
        _latent(),
    )


def test_monotonicity_penalty_rust_value() -> None:
    _check_rust_value_matches_forward(
        "MonotonicityPenalty",
        gt.MonotonicityPenalty(weight=0.9, n_eff=3, direction=1.0),
        _latent(),
    )


def test_mechanism_sparsity_penalty_rust_value() -> None:
    weights = torch.tensor(
        [[0.4, -0.2, 0.1], [0.0, 0.5, -0.3], [0.6, 0.1, 0.0], [-0.2, 0.0, 0.4]],
        dtype=torch.float64,
    )
    _check_rust_value_matches_forward(
        "MechanismSparsityPenalty",
        gt.MechanismSparsityPenalty(
            feature_groups=[[0, 1], [2, 3]], weight=0.5, n_eff=3.0
        ),
        weights,
    )


def test_ordered_beta_bernoulli_assignment_penalty_rust_value() -> None:
    _check_rust_value_matches_forward(
        "OrderedBetaBernoulliPenalty",
        gt.OrderedBetaBernoulliPenalty(k_max=3, alpha=1.2, tau=0.7),
        _logits(),
    )


def test_ivae_ridge_mean_gauge_rust_value() -> None:
    _check_rust_value_matches_forward(
        "IvaeRidgeMeanGauge",
        gt.IvaeRidgeMeanGauge(aux=_aux(), weight=0.4, n_eff=3),
        _latent(),
    )


# ---------------------------------------------------------------------------
# Cross-check: rust_value must agree with the internal `_call_rust_value_grad`
# helper that `forward` already uses. This makes sure the public method
# doesn't drift from the value path the autograd Function calls.
# ---------------------------------------------------------------------------


def test_rust_value_matches_internal_helper_for_ard() -> None:
    module = gt.ARDPenalty(weight=1.3)
    latent = _latent()
    _ = module(latent)  # materialize log_precision parameter

    assert hasattr(module, "rust_value"), "ARDPenalty must expose rust_value (issue #231)"

    direct = float(module.rust_value(latent.detach().cpu().numpy()))

    rho = module.log_precision.detach().to(dtype=latent.dtype)
    from gamfit._penalty_bridge import fixed_weight_schedule
    from gamfit.torch.penalties import (  # type: ignore[attr-defined]
        _latent_json,
        _penalty_json,
    )

    descriptor: dict[str, Any] = {"kind": "ard", "target": "t"}
    schedule = fixed_weight_schedule(module.weight)
    if schedule is not None:
        descriptor["weight_schedule"] = schedule

    value_t, _, _, _ = _call_rust_value_grad(
        latent,
        rho,
        _latent_json(latent.shape[0], latent.shape[1], name="t"),
        _penalty_json(descriptor),
    )
    np.testing.assert_allclose(direct, float(value_t.item()), rtol=0.0, atol=1e-12)


def test_rust_value_matches_internal_helper_for_isometry() -> None:
    module = gt.IsometryPenalty(weight=0.8)
    latent = _latent()
    basis = _basis()
    _ = module(latent, basis)

    assert hasattr(module, "rust_value"), (
        "IsometryPenalty must expose rust_value (issue #231)"
    )

    direct = float(
        module.rust_value(latent.detach().cpu().numpy(), basis=basis.detach().cpu().numpy())
    )

    from gamfit.torch.penalties import _latent_json, _penalty_json  # type: ignore[attr-defined]

    p_out = int(basis.shape[0])
    descriptor = {
        "kind": "isometry",
        "target": "t",
        "weight": 1.0,
        "p_out": p_out,
    }
    rho = torch.full((1,), float(np.log(module.weight)), dtype=latent.dtype)
    jacobian = basis.unsqueeze(0).expand(latent.shape[0], -1, -1).reshape(latent.shape[0], -1)

    value_t, _, _, _ = _call_rust_value_grad(
        latent,
        rho,
        _latent_json(latent.shape[0], latent.shape[1], name="t"),
        _penalty_json(descriptor),
        isometry_jacobian=jacobian,
    )
    np.testing.assert_allclose(direct, float(value_t.item()), rtol=0.0, atol=1e-12)
