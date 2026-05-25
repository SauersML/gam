from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def test_basis_evaluators_match_rust() -> None:
    from gamfit import _api as np_api

    rng = np.random.default_rng(44)

    t = np.sort(rng.uniform(0.0, 1.0, size=16))
    knots = np.linspace(0.0, 1.0, 8)
    got_bs = gt.bspline_basis(torch.as_tensor(t, dtype=torch.float64), torch.as_tensor(knots, dtype=torch.float64), degree=3, periodic=False)
    exp_bs = np_api.bspline_basis(t, knots, degree=3, periodic=False)
    np.testing.assert_allclose(got_bs.detach().cpu().numpy(), exp_bs, rtol=0.0, atol=1e-7, err_msg="Expected torch B-spline basis evaluator to match the Rust evaluator at identical inputs.")

    centers = np.linspace(0.0, 1.0, 6)
    got_du = gt.duchon_basis(torch.as_tensor(t, dtype=torch.float64), torch.as_tensor(centers, dtype=torch.float64), m=2)
    exp_du = np_api.duchon_basis(t, centers, m=2)
    np.testing.assert_allclose(got_du.detach().cpu().numpy(), exp_du, rtol=0.0, atol=1e-7, err_msg="Expected torch Duchon basis evaluator to match the Rust evaluator at identical inputs.")
