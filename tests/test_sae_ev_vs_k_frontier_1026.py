from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

_SRC = (Path(__file__).resolve().parents[1] / "gamfit" / "_sae_manifold.py").read_text()


def _as_2d_float(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D or 2D numeric array")
    return np.ascontiguousarray(arr)


def _load_frontier_namespace():
    ns = {
        "np": np,
        "Any": object,
        "Mapping": dict,
        "_as_2d_float": _as_2d_float,
    }
    for name in (
        "_frontier_reconstruction_ev",
        "_frontier_k_values",
        "_frontier_basis_for_k",
        "_frontier_d_atom_for_k",
        "sae_ev_vs_k_frontier",
    ):
        match = re.search(r"\ndef " + name + r"\(.*?\n(?=\ndef |\Z)", _SRC, re.S)
        assert match is not None, f"{name} not found"
        exec(match.group(0), ns)
    return ns


def _ev(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1.0e-12)


class _FakeManifold:
    def __init__(self, factor: float, *, k: int, basis: list[str]) -> None:
        self.factor = float(factor)
        self.chosen_k = int(k)
        self.atom_topologies = [
            "circle" if b == "periodic" else "linear" for b in basis
        ]
        self.hybrid_split = {
            "curved_atom_count": sum(b == "periodic" for b in basis),
            "linear_atom_count": sum(b == "linear" for b in basis),
            "atoms": [],
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * self.factor


class _FakeLinear:
    def __init__(self, factor: float, *, top_k: int) -> None:
        self.factor = float(factor)
        self.top_k = int(top_k)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def reconstruct(self, assignments: np.ndarray) -> np.ndarray:
        return np.asarray(assignments, dtype=float) * self.factor


def test_sae_ev_vs_k_frontier_scores_heldout_hybrid_and_linear(monkeypatch):
    ns = _load_frontier_namespace()
    calls: list[tuple[str, int, object, object]] = []

    def fake_sae_fit(x, *, K, d_atom, atom_basis, **kwargs):
        calls.append(("hybrid", int(K), d_atom, list(atom_basis)))
        assert kwargs["assignment"] == "ibp_map"
        return _FakeManifold({1: 0.55, 2: 0.82}[int(K)], k=K, basis=list(atom_basis))

    def fake_linear_fit(x, K, **kwargs):
        calls.append(("linear", int(K), kwargs.get("top_k"), None))
        return _FakeLinear({1: 0.45, 2: 0.60}[int(K)], top_k=kwargs["top_k"])

    ns["sae_manifold_fit"] = fake_sae_fit
    ns["linear_dictionary_fit"] = fake_linear_fit
    ns["ev_knee_k"] = lambda hybrid, return_details=False: {"k": 2, "flag": "knee"}
    ns["wager_verdict"] = lambda hybrid, linear: {"confirmed": True}

    train = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    test = np.array([[-1.5, 0.5], [0.5, -0.5], [1.0, 0.0]])
    basis_by_k = {1: ["periodic"], 2: ["periodic", "linear"]}

    frontier = ns["sae_ev_vs_k_frontier"](
        train,
        test,
        [1, 2],
        hybrid_atom_basis=basis_by_k,
        d_atom={1: [1], 2: [1, 1]},
        sae_fit_kwargs={"assignment": "ibp_map"},
        linear_fit_kwargs={"top_k": 1, "max_iter": 3},
    )

    assert calls == [
        ("hybrid", 1, [1], ["periodic"]),
        ("linear", 1, 1, None),
        ("hybrid", 2, [1, 1], ["periodic", "linear"]),
        ("linear", 2, 1, None),
    ]
    assert frontier["hybrid"][1] == pytest.approx(_ev(test, test * 0.55))
    assert frontier["hybrid"][2] == pytest.approx(_ev(test, test * 0.82))
    assert frontier["linear"][1] == pytest.approx(_ev(test, test * 0.45))
    assert frontier["linear"][2] == pytest.approx(_ev(test, test * 0.60))
    assert frontier["rows"][1]["hybrid_basis"] == ["periodic", "linear"]
    assert frontier["rows"][1]["hybrid_atom_topologies"] == ["circle", "linear"]
    assert frontier["rows"][1]["linear_top_k"] == 1
    assert frontier["verdict"]["confirmed"] is True


def test_sae_ev_vs_k_frontier_requires_explicit_basis_plan_for_each_k():
    ns = _load_frontier_namespace()
    train = np.ones((4, 2))
    test = np.ones((3, 2))
    with pytest.raises(ValueError, match="missing an explicit basis plan for K=2"):
        ns["sae_ev_vs_k_frontier"](
            train,
            test,
            [1, 2],
            hybrid_atom_basis={1: ["periodic"]},
        )


def test_sae_ev_vs_k_frontier_rejects_scalar_hybrid_basis():
    ns = _load_frontier_namespace()
    train = np.ones((4, 2))
    test = np.ones((3, 2))
    with pytest.raises(ValueError, match="per-atom basis list"):
        ns["sae_ev_vs_k_frontier"](
            train,
            test,
            [1],
            hybrid_atom_basis="periodic",
        )
