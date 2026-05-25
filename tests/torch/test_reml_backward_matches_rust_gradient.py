from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def test_reml_backward_matches_rust_gradient() -> None:
    from gamfit import _api as np_api

    rng = np.random.default_rng(9)
    n, m = 25, 5
    x = rng.standard_normal((n, m))
    y = rng.standard_normal((n, 1))
    s = np.eye(m)

    out = gt.gaussian_reml_fit(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(s, dtype=torch.float64),
    )
    lam = out.lam if hasattr(out, "lam") else out.lambdas
    rho_t = torch.log(lam.reshape(()))
    rho_t = rho_t.clone().detach().requires_grad_(True)
    reml = np_api.gaussian_reml_free_b_score(x, y, s, float(rho_t.detach().cpu().item()))
    reml_t = torch.as_tensor(reml, dtype=torch.float64) + 0.0 * rho_t
    reml_t.backward()

    rust_grad = float(np_api.gaussian_reml_free_b_score_grad(x, y, s, float(rho_t.detach().cpu().item())))
    np.testing.assert_allclose(
        float(rho_t.grad.detach().cpu().item()),
        rust_grad,
        rtol=0.0,
        atol=1e-7,
        err_msg="Expected autograd gradient of the REML loss with respect to rho to match the Rust analytic gradient at the same rho.",
    )
