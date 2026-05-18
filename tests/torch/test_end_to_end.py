"""End-to-end smoke test composing a fitted gamfit model inside a torch graph."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit  # noqa: E402
import gamfit.torch as gt  # noqa: E402


def _have_ffi(*names: str) -> bool:
    from gamfit._binding import rust_module

    m = rust_module()
    return all(hasattr(m, n) for n in names)


@pytest.mark.skipif(
    not _have_ffi("fit_array", "predict_array"),
    reason="engine missing fit_array/predict_array",
)
def test_from_fitted_inside_torch_module() -> None:
    rng = np.random.default_rng(42)
    n, f = 80, 3
    X = rng.standard_normal((n, f))
    y_true = np.sin(X[:, 0]) + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    Y = y_true + 0.05 * rng.standard_normal(n)

    model = gamfit.fit_array(X, Y, "y ~ s(x0) + x1 + x2")
    wrapped = gt.from_fitted(model)

    # Composing into a tiny torch network: a learned linear head on top of the
    # frozen GAM's mean prediction column. The GAM's `mean` channel sits at the
    # second emitted column (index 1) for Gaussian families.
    head = torch.nn.Linear(1, 1, dtype=torch.float64)

    X_t = torch.as_tensor(X, dtype=torch.float64)
    preds = wrapped(X_t)
    assert preds.shape[0] == n
    assert preds.shape[1] >= 2  # eta + mean (at least)

    mean_col = preds[:, 1:2]
    out = head(mean_col).squeeze(1)
    target = torch.as_tensor(Y, dtype=torch.float64)
    initial_loss = torch.nn.functional.mse_loss(out, target).item()

    optim = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(60):
        optim.zero_grad()
        preds = wrapped(X_t).detach()  # frozen
        mean_col = preds[:, 1:2]
        out = head(mean_col).squeeze(1)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optim.step()
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, (initial_loss, final_loss)


@pytest.mark.skipif(
    not _have_ffi("gaussian_reml_fit_positions"),
    reason="engine missing position REML FFI",
)
def test_reml_positions_gradients_flow_through_torch_loop() -> None:
    rng = np.random.default_rng(7)
    n = 40
    t_np = np.sort(rng.uniform(0.05, 0.95, size=n))
    y_np = np.sin(2 * np.pi * t_np).reshape(-1, 1) + 0.05 * rng.standard_normal((n, 1))
    knots = np.linspace(0.0, 1.0, 8)
    M = knots.size + 3 - 1
    penalty = np.eye(M)

    t_param = torch.as_tensor(t_np, dtype=torch.float64, requires_grad=True)
    y_t = torch.as_tensor(y_np, dtype=torch.float64)
    k_t = torch.as_tensor(knots, dtype=torch.float64)
    p_t = torch.as_tensor(penalty, dtype=torch.float64)

    out = gt.gaussian_reml_fit_positions(t_param, y_t, "bspline", k_t, p_t)
    loss = (out.fitted - y_t).pow(2).mean() + 0.01 * out.reml_score
    loss.backward()
    assert t_param.grad is not None
    assert torch.isfinite(t_param.grad).all()
