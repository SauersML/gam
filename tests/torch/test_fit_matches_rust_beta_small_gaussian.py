from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def test_torch_fit_matches_rust_beta_small_gaussian() -> None:
    from gamfit import _api as np_api

    rng = np.random.default_rng(123)
    n, m = 40, 6
    x = rng.standard_normal((n, m))
    beta_true = rng.standard_normal((m, 1))
    y = x @ beta_true + 0.02 * rng.standard_normal((n, 1))
    penalty = np.eye(m)

    rust = np_api.gaussian_reml_fit(x, y, penalty)
    torch_out = gt.gaussian_reml_fit(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(penalty, dtype=torch.float64),
    )

    np.testing.assert_allclose(
        torch_out.coefficients.detach().cpu().numpy(),
        rust["coefficients"],
        rtol=0.0,
        atol=1e-5,
        err_msg="Expected torch Gaussian REML coefficients to match Rust coefficients within 1e-5 on a small Gaussian problem.",
    )
