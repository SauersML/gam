from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gamfit import _sae_manifold
from gamfit import _sparse_dictionary
from gamfit._sparse_dictionary import (
    SparseDictionaryConvergence,
    SparseDictionaryFit,
)


class _ReachedCurvedEngine(RuntimeError):
    """Sentinel: the facade handed the fit to the curved manifold FFI."""


class _AdmissionRust:
    """Fake of the Rust front door mirroring its lane rule: penalty-gated
    K > P demotes to sparse codes; a hard top-k support request at K > P is
    admitted to the curved framed/streaming lane. Also mirrors the thin
    Rust-owned vocabulary/default helpers the facade consults on the way to
    the admission (assignment canonicalization, topology→basis resolution,
    the large-K top-k default) so the fake tracks the real FFI surface."""

    def __init__(self) -> None:
        self.admission_calls: list[dict[str, Any]] = []

    def sae_canonical_assignment_kind(self, kind: str) -> str:
        mapping = {
            "softmax": "softmax",
            "topk": "topk",
            "top_k": "topk",
            "ibp": "ibp_map",
            "ibp-map": "ibp_map",
            "ibp_map": "ibp_map",
            "threshold_gate": "threshold_gate",
            "gated": "threshold_gate",
            "jump_relu": "threshold_gate",
            "jumprelu": "threshold_gate",
        }
        key = str(kind).strip().lower()
        if key not in mapping:
            raise ValueError(f"unknown assignment kind {kind!r}")
        return mapping[key]

    def sae_basis_kind_for_topology(self, name: str) -> str:
        return {
            "circle": "periodic",
            "periodic": "periodic",
            "linear": "linear",
            "euclidean": "euclidean",
            "auto": "auto",
        }.get(str(name), str(name))

    def sae_atom_topologies(self, bases: list[str]) -> tuple[str | None, list[str]]:
        to_topology = {
            "periodic": "circle",
            "linear": "linear",
            "euclidean": "euclidean",
            "auto": "auto",
        }
        per_atom = [to_topology.get(str(b), str(b)) for b in bases]
        scalar = per_atom[0] if per_atom else None
        return scalar, per_atom

    def sae_canonical_topology(self, name: str) -> str:
        return str(name)

    def sae_default_top_k_for_large_dictionary(
        self, n_obs: int, k_atoms: int
    ) -> int | None:
        # Mirror of assignment::default_top_k_for_large_dictionary: None when
        # the dense softmax path is admitted (N/K >= K, or K <= 1), else
        # clamp(ceil(N/K), 1, K-1).
        if k_atoms <= 1 or n_obs >= k_atoms * k_atoms:
            return None
        cap = -(-int(n_obs) // int(k_atoms))
        return max(1, min(cap, int(k_atoms) - 1))

    def sae_fit_admission(
        self,
        n_obs: int,
        output_dim: int,
        n_atoms: int,
        d_max: int = 1,
        topk_support: int | None = None,
    ) -> dict[str, Any]:
        self.admission_calls.append(
            {
                "n_obs": int(n_obs),
                "output_dim": int(output_dim),
                "n_atoms": int(n_atoms),
                "d_max": int(d_max),
                "topk_support": None if topk_support is None else int(topk_support),
            }
        )
        dense_assignment_cells = int(n_obs) * int(n_atoms)
        response_cells = int(n_obs) * int(output_dim)
        if dense_assignment_cells <= response_cells:
            lane = "dense_certification"
        elif topk_support is not None:
            lane = "curved_streaming"
        else:
            lane = "sparse_codes"
        return {
            "lane": lane,
            "n_obs": int(n_obs),
            "output_dim": int(output_dim),
            "n_atoms": int(n_atoms),
            "dense_assignment_cells": dense_assignment_cells,
            "response_cells": response_cells,
        }

    def sae_manifold_fit_minimal(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise _ReachedCurvedEngine(
            f"curved engine invoked with top_k={kwargs.get('top_k')!r}"
        )


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
            convergence=SparseDictionaryConvergence(
                inner_ev_residual=0.0,
                inner_tolerance=1.0e-6,
                decoder_residual=0.0,
                decoder_tolerance=1.0e-6,
                routing_residual=0.0,
                routing_tolerance=1.0e-6,
                outer_rho_residual=0.0,
                outer_tolerance=1.0e-6,
                selected_rho=1.0e-6,
                outer_iterations=1,
            ),
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


def test_topk_assignment_above_crossover_reaches_curved_engine(monkeypatch: Any) -> None:
    """K > P with assignment='topk' is admitted to the CURVED lane.

    The facade must pass the hard support size to the front-door admission
    (topk_support), receive the 'curved_streaming' lane, and hand the fit to
    the curved manifold FFI — never to the linear sparse-code trainer.
    """

    rust = _AdmissionRust()
    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: rust)

    def forbidden_sparse_fit(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "a topk manifold request must never reroute to sparse_dictionary_fit"
        )

    monkeypatch.setattr(_sae_manifold, "sparse_dictionary_fit", forbidden_sparse_fit)

    n_obs = 16
    output_dim = 4
    n_atoms = 9
    x = np.zeros((n_obs, output_dim), dtype=np.float32)
    with pytest.raises(_ReachedCurvedEngine):
        _sae_manifold.sae_manifold_fit(
            x, K=n_atoms, assignment="topk", top_k=2, n_iter=1
        )

    assert rust.admission_calls, "the front-door admission must be consulted"
    call = rust.admission_calls[0]
    assert call["n_obs"] == n_obs
    assert call["output_dim"] == output_dim
    assert call["n_atoms"] == n_atoms
    assert call["topk_support"] == 2
    assert call["d_max"] == 2  # default d_atom


def test_topk_assignment_requires_explicit_support_size(monkeypatch: Any) -> None:
    """assignment='topk' without top_k fails eagerly with an actionable error,
    before any admission or fit call — a support-less topk request must never
    fall through to the penalty-gated sparse reroute."""

    rust = _AdmissionRust()
    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: rust)

    x = np.zeros((16, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="requires top_k"):
        _sae_manifold.sae_manifold_fit(x, K=9, assignment="topk", n_iter=1)
    assert rust.admission_calls == []


class _LinearAdmissionRust(_AdmissionRust):
    """Front-door fake extended with the #2232 Inc-5b modeling-choice rule:
    an explicit linear-dictionary request is sparse_codes at ANY K."""

    def __init__(self) -> None:
        super().__init__()
        self.linear_admission_calls: list[dict[str, Any]] = []

    def sae_linear_dictionary_admission(
        self, n_obs: int, output_dim: int, n_atoms: int, block_size: int = 1
    ) -> dict[str, Any]:
        self.linear_admission_calls.append(
            {
                "n_obs": int(n_obs),
                "output_dim": int(output_dim),
                "n_atoms": int(n_atoms),
                "block_size": int(block_size),
            }
        )
        return {
            "lane": "sparse_codes",
            "n_obs": int(n_obs),
            "output_dim": int(output_dim),
            "n_atoms": int(n_atoms),
            "dense_assignment_cells": int(n_obs) * int(n_atoms),
            "response_cells": int(n_obs) * int(output_dim),
        }


def test_explicit_linear_dictionary_routes_sparse_lane_below_crossover(
    monkeypatch: Any,
) -> None:
    """#2232 Inc 5b (Gap B): atom_topology='linear' + assignment='topk' names
    the linear sparse-dictionary model, admitted to the unified linear schedule
    at ANY K — including K <= P, where the shape-derived default would force
    the dense engine. d_atom=1 routes the atom schedule."""

    rust = _LinearAdmissionRust()
    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: rust)

    calls: dict[str, Any] = {}

    def fake_sparse_fit(X: Any, K: int, **kwargs: Any) -> str:
        calls["K"] = int(K)
        calls["active"] = int(kwargs["active"])
        return "atom-schedule-fit"

    monkeypatch.setattr(_sae_manifold, "sparse_dictionary_fit", fake_sparse_fit)

    # K=6 <= P=24: below the shape crossover, so only the EXPLICIT request
    # can select the sparse lane.
    x = np.zeros((64, 24), dtype=np.float32)
    fit = _sae_manifold.sae_manifold_fit(
        x, K=6, d_atom=1, atom_topology="linear", assignment="topk", top_k=2, n_iter=1
    )
    assert fit == "atom-schedule-fit"
    assert calls == {"K": 6, "active": 2}
    assert rust.linear_admission_calls == [
        {"n_obs": 64, "output_dim": 24, "n_atoms": 6, "block_size": 1}
    ]
    # The shape-derived admission is never consulted: the request owns the lane.
    assert rust.admission_calls == []


def test_explicit_linear_block_dictionary_routes_block_lane(monkeypatch: Any) -> None:
    """#2232 Inc 5b: uniform d_atom=b>=2 linear atoms are the Grassmann block
    lane — framed Euclidean d=b atoms with block-TopK — through the single
    public entry."""

    rust = _LinearAdmissionRust()
    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: rust)

    calls: dict[str, Any] = {}

    def fake_block_fit(X: Any, n_blocks: int, **kwargs: Any) -> str:
        calls["n_blocks"] = int(n_blocks)
        calls["block_size"] = int(kwargs["block_size"])
        calls["block_topk"] = int(kwargs["block_topk"])
        return "block-schedule-fit"

    def forbidden_sparse_fit(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a d=b>=2 linear request must take the BLOCK lane")

    monkeypatch.setattr(_sae_manifold, "block_sparse_dictionary_fit", fake_block_fit)
    monkeypatch.setattr(_sae_manifold, "sparse_dictionary_fit", forbidden_sparse_fit)

    x = np.zeros((64, 24), dtype=np.float32)
    fit = _sae_manifold.sae_manifold_fit(
        x, K=4, d_atom=3, atom_topology="linear", assignment="topk", top_k=2, n_iter=1
    )
    assert fit == "block-schedule-fit"
    assert calls == {"n_blocks": 4, "block_size": 3, "block_topk": 2}
    assert rust.linear_admission_calls == [
        {"n_obs": 64, "output_dim": 24, "n_atoms": 4, "block_size": 3}
    ]


def test_linear_topology_without_topk_keeps_dense_routing(monkeypatch: Any) -> None:
    """A penalty-gated (softmax) linear request keeps the historical
    shape-derived routing: only the HARD-SUPPORT linear request names the
    sparse-dictionary model (its gates are live Newton state otherwise)."""

    rust = _LinearAdmissionRust()
    monkeypatch.setattr(_sae_manifold, "rust_module", lambda: rust)

    x = np.zeros((64, 24), dtype=np.float32)
    with pytest.raises(_ReachedCurvedEngine):
        _sae_manifold.sae_manifold_fit(
            x, K=6, d_atom=1, atom_topology="linear", n_iter=1
        )
    assert rust.linear_admission_calls == []
    assert rust.admission_calls, "softmax linear must take the shape-derived door"


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
                "convergence": {
                    "inner_ev_residual": 0.0,
                    "inner_tolerance": 1.0e-6,
                    "decoder_residual": 0.0,
                    "decoder_tolerance": 1.0e-6,
                    "routing_residual": 0.0,
                    "routing_tolerance": 1.0e-6,
                    "outer_rho_residual": 0.0,
                    "outer_tolerance": 1.0e-6,
                    "selected_rho": 1.0e-6,
                    "outer_iterations": 1,
                },
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
    assert "converged" not in SparseDictionaryFit.__dataclass_fields__
    assert fit.convergence.outer_iterations == 1
    assert not hasattr(fit.convergence, "__dict__")
    with pytest.raises(AttributeError):
        setattr(fit.convergence, "outer_iterations", 2)
    materialized = fit.fitted
    assert materialized.shape == (8, 3)
