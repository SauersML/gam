from __future__ import annotations

from typing import Any

import numpy as np

from gamfit import _sae_manifold
from gamfit import _sparse_dictionary
from gamfit._sparse_dictionary import SparseDictionaryFit


class _AdmissionRust:
    def sae_fit_admission(self, n_obs: int, output_dim: int, n_atoms: int) -> dict[str, Any]:
        dense_assignment_cells = int(n_obs) * int(n_atoms)
        response_cells = int(n_obs) * int(output_dim)
        return {
            "lane": "sparse_codes",
            "n_obs": int(n_obs),
            "output_dim": int(output_dim),
            "n_atoms": int(n_atoms),
            "dense_assignment_cells": dense_assignment_cells,
            "response_cells": response_cells,
        }


def test_public_sae_fit_uses_sparse_artifact_above_front_door_crossover(monkeypatch: Any) -> None:
    """Scaled acceptance for the p=4096, K=32000, N=1000000 front-door case.

    The real shape is too large for CI. This test preserves the invariant
    instead: for K > P the public facade must return a sparse-code artifact whose
    retained training payload is only fixed-width ``N x active`` indices/codes,
    never ``N x K`` assignments and never a stored second ``N x P`` fitted copy.
    """

    calls: dict[str, Any] = {}

    def fake_sparse_fit(X: Any, K: int, **kwargs: Any) -> SparseDictionaryFit:
        x = np.asarray(X)
        active = int(kwargs["active"])
        calls["shape"] = tuple(int(v) for v in x.shape)
        calls["K"] = int(K)
        calls["active"] = active
        return SparseDictionaryFit(
            decoder=np.zeros((int(K), x.shape[1]), dtype=np.float32),
            indices=np.zeros((x.shape[0], active), dtype=np.uint32),
            codes=np.zeros((x.shape[0], active), dtype=np.float32),
            explained_variance=0.0,
            epochs=1,
            converged=True,
            active=active,
            score_route_stats={},
        )

    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: _AdmissionRust())
    monkeypatch.setattr(_sae_manifold, "sparse_dictionary_fit", fake_sparse_fit)

    n_obs = 16
    output_dim = 4
    n_atoms = 9
    x = np.zeros((n_obs, output_dim), dtype=np.float32)
    fit = _sae_manifold.sae_manifold_fit(x, K=n_atoms, n_iter=1)

    assert isinstance(fit, SparseDictionaryFit)
    assert calls == {"shape": (n_obs, output_dim), "K": n_atoms, "active": 2}
    assert fit.retained_training_payload_cells == n_obs * fit.active * 2
    assert fit.retained_training_payload_cells < n_obs * n_atoms
    assert fit.retained_training_payload_cells <= n_obs * output_dim
    assert "fitted" not in SparseDictionaryFit.__dataclass_fields__
    assert not hasattr(fit, "__dict__")


def test_sparse_dictionary_facade_accepts_no_eager_fitted_payload(monkeypatch: Any) -> None:
    class _SparseRust:
        def sparse_dictionary_fit(self, X: Any, K: int, **kwargs: Any) -> dict[str, Any]:
            x = np.asarray(X)
            active = int(kwargs["active"])
            return {
                "front_door_lane": "sparse_codes",
                "decoder": np.zeros((int(K), x.shape[1]), dtype=np.float32),
                "indices": np.zeros((x.shape[0], active), dtype=np.uint32),
                "codes": np.zeros((x.shape[0], active), dtype=np.float32),
                "explained_variance": 0.0,
                "epochs": 1,
                "converged": True,
                "active": active,
                "score_route_stats": {},
            }

        def sparse_dictionary_reconstruct_ffi(
            self,
            decoder: Any,
            indices: Any,
            codes: Any,
        ) -> np.ndarray:
            return np.zeros((np.asarray(indices).shape[0], np.asarray(decoder).shape[1]), dtype=np.float32)

    monkeypatch.setattr(_sparse_dictionary, "rust_module", lambda: _SparseRust())

    fit = _sparse_dictionary.sparse_dictionary_fit(
        np.zeros((8, 3), dtype=np.float32),
        7,
        active=1,
    )

    assert fit.retained_training_payload_cells == 16
    assert "fitted" not in SparseDictionaryFit.__dataclass_fields__
    materialized = fit.fitted
    assert materialized.shape == (8, 3)
