"""Small synthetic SAE-style end-to-end test.

Wires an encoder Linear → positions in [0,1] → F=4 Duchon atoms via
``gamfit.torch.fit`` → loss → ``backward()``. Confirms gradient flows
back to the encoder's weights through the analytic REML VJP.

Skipped (with reason) if the multi-block REML path is missing.
"""

from __future__ import annotations

import pytest

gt = pytest.importorskip("gamfit.torch")
torch = pytest.importorskip("torch")

if not hasattr(gt, "gaussian_reml_fit_blocks"):
    pytest.skip(
        "gamfit.torch.gaussian_reml_fit_blocks missing; "
        "the multi-block REML path is required for the SAE end-to-end test.",
        allow_module_level=True,
    )


def test_encoder_receives_gradient_through_reml():
    torch.manual_seed(0)
    N = 32
    D = 32
    F = 4
    K = 6

    # Random input batch.
    x = torch.randn(N, D, dtype=torch.float64)

    # Encoder 32 → F positions in [0, 1] via sigmoid.
    encoder = torch.nn.Linear(D, F, dtype=torch.float64)
    pos_raw = encoder(x)
    positions = torch.sigmoid(pos_raw)  # (N, F)

    # Decoder isn't used directly because Duchon's `by` modulation handles
    # per-atom contribution; we leave atom_active=1 everywhere as instructed.
    centers = torch.linspace(0.0, 1.0, K, dtype=torch.float64).unsqueeze(1)
    amp = torch.ones(N, dtype=torch.float64)

    # One smooth per atom; per-atom 1D position as that smooth's points.
    points_list = [positions[:, k].unsqueeze(1) for k in range(F)]
    smooths_list = [gt.Duchon(centers=centers, m=2, by=amp) for _ in range(F)]

    # Use mean-centered input as the response: shape (N,) scalar response
    # (multi-block REML requires D=1 on the multi-block path).
    response = x.mean(dim=1)

    result = gt.fit(
        points=points_list,
        response=response,
        smooths=smooths_list,
    )
    loss = ((result.fitted.squeeze(-1) - response) ** 2).mean()
    loss.backward()

    assert encoder.weight.grad is not None, (
        "encoder.weight.grad is None — gradient did not flow through REML"
    )
    grad_norm = encoder.weight.grad.abs().sum().item()
    assert grad_norm > 0.0, (
        f"encoder.weight.grad has zero magnitude (sum |g|={grad_norm}); "
        "REML VJP didn't reach the encoder."
    )
    assert torch.isfinite(encoder.weight.grad).all()
