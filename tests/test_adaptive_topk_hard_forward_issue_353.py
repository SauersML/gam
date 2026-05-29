"""Regression test for #353: ``AdaptiveTopK`` must sparsify its forward output.

The bug: ``_AdaptiveTopKSTE.forward`` built the straight-through value as
``z_active = z + (z * hard_mask - z_active_soft).detach()``, which numerically
equals ``z * (1 + hard - soft) ~= z`` for every atom (``hard ~= soft`` on both
top-K and non-top-K entries). The output was therefore fully *dense* — all ``F``
atoms nonzero — despite a correct per-row ``k_pred`` / ``hard_mask``, so any SAE
built on the primitive had zero sparsity (silent; no error).

The contract (``_AdaptiveTopKSTE``'s own docstring): "hard per-row
top-``round(k_pred)`` mask forward, soft-mask gradient backward". The forward
value must equal ``z * hard_mask`` exactly — exactly ``round(k_pred_i)`` nonzeros
per row — while the gradient still flows through the soft carrier.

The fix bases the STE on ``z_active_soft``:

    z_active = z_active_soft + (z * hard_mask - z_active_soft).detach()

so the forward value collapses to ``z * hard_mask`` while ``.detach()`` keeps the
soft mask in the autograd graph for the analytic backward.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gamfit_torch = pytest.importorskip("gamfit.torch")

from gamfit.torch import AdaptiveTopK  # noqa: E402


def test_forward_is_hard_top_k_not_dense() -> None:
    # Exact repro from the issue: F=64, learned per-row K in [4, 16].
    torch.manual_seed(0)
    g = AdaptiveTopK(F=64, k_min=4, k_max=16, head="linear", temperature=0.1)
    z = torch.randn(8, 64)
    z_active, k_pred = g(z)

    # k_pred stays within the configured bounds (this part was already correct).
    assert bool((k_pred >= 4.0 - 1e-6).all()) and bool((k_pred <= 16.0 + 1e-6).all()), (
        f"k_pred out of [k_min, k_max]: {k_pred.tolist()}"
    )

    # The output must be sparse: nonzeros per row == round(k_pred_i), NOT 64.
    nonzeros = (z_active != 0).sum(dim=-1)
    assert bool((nonzeros < 64).all()), (
        "AdaptiveTopK forward returned fully dense codes (64 nonzeros/row); "
        f"expected ~round(k_pred) per row, got {nonzeros.tolist()}"
    )
    expected_k = k_pred.round().clamp(min=1, max=64).to(torch.long)
    assert torch.equal(nonzeros, expected_k), (
        "nonzeros per row must equal round(k_pred) (hard top-K forward); "
        f"got nonzeros {nonzeros.tolist()} vs round(k_pred) {expected_k.tolist()}"
    )


def test_forward_value_equals_hard_masked_input() -> None:
    # The surviving (nonzero) entries must be the untouched top-K magnitudes of z,
    # i.e. the forward value equals z * hard_mask exactly — no soft attenuation.
    torch.manual_seed(1)
    g = AdaptiveTopK(F=32, k_min=2, k_max=8, head="linear", temperature=0.1)
    z = torch.randn(5, 32)
    z_active, k_pred = g(z)

    for row in range(z.shape[0]):
        kk = int(k_pred[row].round().clamp(min=1, max=32).item())
        _, idx = torch.topk(z[row].abs(), k=kk)
        hard = torch.zeros_like(z[row])
        hard[idx] = 1.0
        expected = z[row] * hard
        assert torch.equal(z_active[row], expected), (
            f"row {row}: forward must equal z * hard_mask exactly; "
            f"got {z_active[row].tolist()} vs {expected.tolist()}"
        )


def test_backward_still_flows_through_soft_carrier() -> None:
    # The .detach() must keep the soft mask in the graph: gradients flow to z
    # and to the k-head parameters, and are finite. (The fix must not break the
    # analytic backward by zeroing the carrier.)
    torch.manual_seed(2)
    g = AdaptiveTopK(F=16, k_min=2, k_max=6, head="linear", temperature=0.1)
    z = torch.randn(4, 16, requires_grad=True)
    z_active, k_pred = g(z)
    loss = (z_active * z_active).sum() + k_pred.sum()
    loss.backward()

    assert z.grad is not None and torch.isfinite(z.grad).all(), "z grad must be finite"
    head_grads = [p.grad for p in g.k_head.parameters() if p.requires_grad]
    assert head_grads, "k-head must expose trainable parameters"
    assert all(gr is not None and torch.isfinite(gr).all() for gr in head_grads), (
        "k-head gradients must flow and be finite (soft carrier must stay in the graph)"
    )
