from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def test_penalty_modules_match_rust_closed_form() -> None:
    latent = torch.tensor([[0.2, -0.1, 0.7], [0.5, 0.3, -0.4]], dtype=torch.float64)
    basis = torch.eye(3, dtype=torch.float64)

    checks: list[tuple[str, torch.nn.Module, dict[str, Any]]] = [
        ("ARDPenalty", gt.ARDPenalty(weight=1.3), {}),
        ("IsometryPenalty", gt.IsometryPenalty(weight=0.8), {"basis": basis}),
        (
            "SmoothThresholdPenalty",
            gt.SmoothThresholdPenalty(
                torch.full((3,), 0.05, dtype=torch.float64), weight=1.1
            ),
            {},
        ),
    ]

    for name, module, kwargs in checks:
        value = module(latent, **kwargs) if kwargs else module(latent)
        assert bool(torch.isfinite(value)), f"Expected {name} forward value to be finite when compared to the Rust closed-form penalty evaluator."

        rust_value = float(module.rust_value(latent.detach().cpu().numpy(), **{k: v.detach().cpu().numpy() for k, v in kwargs.items()}))
        np.testing.assert_allclose(
            float(value.detach().cpu().item()),
            rust_value,
            rtol=0.0,
            atol=1e-10,
            err_msg=f"Expected {name}.forward to match the Rust closed-form penalty value on the same input.",
        )
