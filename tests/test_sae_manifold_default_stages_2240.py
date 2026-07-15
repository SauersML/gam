import inspect

import numpy as np
import pytest

from gamfit import _sae_manifold


def test_python_fit_defaults_match_unbundled_native_pipeline(monkeypatch):
    captured = {}
    result = object()

    class NativeStub:
        def sae_manifold_fit_model(self, *args, **kwargs):
            captured.update(kwargs)
            return result

    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: NativeStub())

    actual = _sae_manifold.sae_manifold_fit(
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        K=1,
        d_atom=1,
        n_iter=1,
    )

    assert actual is result
    assert captured["promote_from_residual"] is False
    assert captured["run_structure_search"] is False
    assert captured["structured_residual_passes"] == 0
    assert type(captured["structured_residual_passes"]) is int
    assert captured["sparsity_strength"] is None
    assert captured["gpu_policy"] == "auto"


def test_python_gpu_policy_is_per_fit_and_forwarded_without_global_mutation(
    monkeypatch,
):
    captured = {}

    class NativeStub:
        def sae_manifold_fit_model(self, *args, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: NativeStub())
    _sae_manifold.sae_manifold_fit(
        np.asarray([[0.0], [1.0]], dtype=np.float64),
        K=1,
        d_atom=1,
        n_iter=1,
        gpu="off",
    )

    assert captured["gpu_policy"] == "off"


def test_python_sparsity_default_is_owned_by_native_front_door():
    parameter = inspect.signature(_sae_manifold.sae_manifold_fit).parameters[
        "sparsity_weight"
    ]
    assert parameter.default is None


def test_external_certificate_defaults_to_evaluation_only():
    parameter = inspect.signature(
        _sae_manifold.sae_manifold_certify_external
    ).parameters["run_structure_search"]
    assert parameter.default is False


def test_structured_pass_count_does_not_truncate_non_integer(monkeypatch):
    class NativeStub:
        def sae_manifold_fit_model(self, *args, **kwargs):
            raise AssertionError(
                "invalid request must be rejected before native dispatch"
            )

    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: NativeStub())
    with pytest.raises(TypeError, match="must be an integer"):
        _sae_manifold.sae_manifold_fit(
            np.asarray([[0.0], [1.0]]),
            K=1,
            d_atom=1,
            structured_residual_passes=1.5,
        )
