import inspect

import numpy as np

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


def test_external_certificate_defaults_to_evaluation_only():
    parameter = inspect.signature(
        _sae_manifold.sae_manifold_certify_external
    ).parameters["run_structure_search"]
    assert parameter.default is False
