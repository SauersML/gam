"""The Torch manifold-SAE surface is a frozen view of one native fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gamfit = pytest.importorskip("gamfit")
gt = pytest.importorskip("gamfit.torch")


def _native_fit():
    path = Path(__file__).parents[1] / "fixtures" / "manifold_sae" / "golden_full.json"
    payload = json.loads(path.read_text())
    return gamfit.ManifoldSAE.from_dict(payload)


def test_adapter_has_no_parameters_and_rejects_input_gradients() -> None:
    module = gt.ManifoldSAE(_native_fit())
    assert list(module.parameters()) == []
    x = torch.zeros(2, module.input_dim, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="input gradients are unavailable"):
        module(x)


def test_adapter_output_is_one_native_converged_latent_state() -> None:
    native = _native_fit()
    module = gt.ManifoldSAE(native)
    x_np = np.asarray(native.fitted, dtype=np.float64)[:4]
    expected = native.converged_latents(np.ascontiguousarray(x_np))

    out = module(torch.from_numpy(x_np.copy()))
    np.testing.assert_allclose(out.reconstruction.numpy(), expected["fitted"])
    np.testing.assert_allclose(out.codes.numpy(), expected["assignments"])
    assert len(out.coordinates) == len(expected["coords"])
    for actual, reference in zip(out.coordinates, expected["coords"]):
        np.testing.assert_allclose(actual.numpy(), reference)
    assert out.penalized_laml_criterion.item() == pytest.approx(
        native.penalized_laml_criterion
    )


def test_adapter_state_dict_restores_the_serialized_native_fit() -> None:
    first = gt.ManifoldSAE(_native_fit())
    second = gt.ManifoldSAE(_native_fit())
    second.load_state_dict(first.state_dict())
    assert second.fitted.to_json() == first.fitted.to_json()
