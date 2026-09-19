"""The Torch manifold-SAE surface is a frozen view of one native fit."""

from __future__ import annotations

import dataclasses
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
    return gamfit.sae.ManifoldSAE.from_dict(payload)


def _residual_energy(x: np.ndarray, reconstruction: torch.Tensor) -> float:
    residual = x - reconstruction.numpy()
    return float(0.5 * np.sum(residual * residual))


def _assert_fit_metadata_is_the_native_fit(fit_metadata, native) -> None:
    assert fit_metadata.penalized_loss_score is not None
    assert fit_metadata.penalized_loss_score.item() == native.penalized_loss_score
    assert (
        fit_metadata.penalized_quasi_laplace_criterion.item()
        == native.penalized_quasi_laplace_criterion
    )
    expected_log_lambdas = native.selected_log_lambda_smooth
    if expected_log_lambdas is None:
        assert fit_metadata.selected_smooth_lambdas is None
    else:
        assert fit_metadata.selected_smooth_lambdas is not None
        np.testing.assert_allclose(
            fit_metadata.selected_smooth_lambdas.numpy(),
            np.exp(np.asarray(expected_log_lambdas, dtype=np.float64)),
        )


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
    assert out.batch_penalized_loss_score.item() == pytest.approx(
        expected["oos_penalized_loss"]
    )
    _assert_fit_metadata_is_the_native_fit(out.fit, native)


def test_batch_scores_follow_the_batch_while_fit_metadata_is_invariant() -> None:
    """#2933 F42: a batch output must not report training scalars as batch values.

    Every golden decoder row lies in span{(1, 1, 1, 1), (0, 1, 2, 3)}, so no
    reconstruction can absorb the orthogonal direction (1, -1, -1, 1). Shifting
    four rows by 10 along it adds at least 0.5 * 4 * 400 = 800 to the residual
    energy. A top-level scalar of the output that is identical for both batches
    is a training-fit value presented as a batch value.
    """
    native = _native_fit()
    module = gt.ManifoldSAE(native)
    near = np.ascontiguousarray(np.asarray(native.fitted, dtype=np.float64)[:4])
    far = np.ascontiguousarray(near + 10.0 * np.array([1.0, -1.0, -1.0, 1.0]))
    out_near = module(torch.from_numpy(near.copy()))
    out_far = module(torch.from_numpy(far.copy()))

    energy_near = _residual_energy(near, out_near.reconstruction)
    energy_far = _residual_energy(far, out_far.reconstruction)
    assert energy_far >= 800.0 * (1.0 - 1e-12), (energy_near, energy_far)

    scalar_fields = [
        field.name
        for field in dataclasses.fields(gt.ManifoldSAEOutput)
        if isinstance(getattr(out_near, field.name), torch.Tensor)
        and getattr(out_near, field.name).dim() == 0
    ]
    assert scalar_fields, "the batch output carries no batch-scoped loss"
    for name in scalar_fields:
        near_value = getattr(out_near, name).item()
        far_value = getattr(out_far, name).item()
        assert near_value != far_value, (
            f"ManifoldSAEOutput.{name} is {near_value!r} for both batches although "
            f"their residual energies are {energy_near:.6g} and {energy_far:.6g}"
        )

    _assert_fit_metadata_is_the_native_fit(out_near.fit, native)
    _assert_fit_metadata_is_the_native_fit(out_far.fit, native)

    # The batch score is the native loss of this batch's own solve, and its data
    # fit is the residual energy of the reconstruction the adapter returned.
    for x, out, energy in ((near, out_near, energy_near), (far, out_far, energy_far)):
        latents = native.converged_latents(x)
        np.testing.assert_allclose(out.reconstruction.numpy(), latents["fitted"])
        assert out.batch_penalized_loss_score.item() == pytest.approx(
            latents["oos_penalized_loss"], rel=1e-12
        )
        breakdown = latents["penalized_loss_breakdown"]
        assert breakdown["data_fit"] == pytest.approx(energy, rel=1e-10)
        assert out.batch_penalized_loss_score.item() == pytest.approx(
            -breakdown["total_penalized_loss"], rel=1e-12
        )


def test_adapter_state_dict_restores_the_serialized_native_fit() -> None:
    first = gt.ManifoldSAE(_native_fit())
    second = gt.ManifoldSAE(_native_fit())
    second.load_state_dict(first.state_dict())
    assert second.fitted.to_json() == first.fitted.to_json()
