"""#2091 — full-fit bitwise equivalence: the Rust `from_fit_payload` builder vs
the Python `ManifoldSAE.from_payload` -> `to_dict` path.

Design (A) moves the raw-payload -> flat-`to_dict`-schema coercion into Rust so
`sae_manifold_fit` can return a Rust-owned `ManifoldSaeCore` with no Python
dataclass. This test proves the builder reproduces the dataclass serialization
EXACTLY: it captures the real raw `sae_manifold_fit_minimal` payload from a live
fit, runs both paths, and asserts `core.to_dict() == fit.to_dict()`. Any field
the builder mis-assembles (kind/topology derivation, n_harmonics, per-atom
gate-column slicing, channel-cov factor, periodic shape-band reorder, report
passthroughs) surfaces as a dict mismatch. Three atom kinds are exercised
(circle=periodic shape-band path, euclidean + duchon = duchon_centers path).

MAINLINE only: no Fisher shard, no linear_block relabel (follow-up arms). The
mainline fixture is asserted NON-FINITE-FREE so a future data change cannot
silently exercise the NaN branch, and the builder's NaN handling is pinned as
CONSISTENT with the legacy `from_json` reader (both reject via serde parse)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import rust_module  # noqa: E402


def _builder():
    fn = getattr(rust_module(), "sae_manifold_core_from_fit_payload", None)
    if fn is None:
        pytest.skip("wheel predates sae_manifold_core_from_fit_payload (#2091 builder)")
    return fn


def _jsonable(value):
    """The exact `to_dict._jsonable`: numpy -> list, dict keys stringified."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _no_nonfinite(v) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, dict):
        return all(_no_nonfinite(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return all(_no_nonfinite(x) for x in v)
    return True


def _data_for(topology: str, n: int, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if topology == "circle":
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        harm = np.column_stack([np.cos(theta), np.sin(theta)])
        x = harm @ rng.standard_normal((2, p))
    else:
        t = rng.standard_normal((n, 2))
        x = t @ rng.standard_normal((2, p)) + 0.3 * (t[:, :1] ** 2) @ rng.standard_normal((1, p))
    x = x + 0.02 * rng.standard_normal((n, p))
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float64)


@pytest.mark.parametrize("topology,d_atom", [("circle", 1), ("euclidean", 2), ("duchon", 2)])
def test_builder_full_fit_equiv(topology, d_atom, monkeypatch):
    builder = _builder()
    rm = rust_module()
    captured: dict = {}
    orig = rm.sae_manifold_fit_minimal

    def capture(*args, **kwargs):
        payload = orig(*args, **kwargs)
        captured["raw"] = dict(payload)
        return payload

    monkeypatch.setattr(rm, "sae_manifold_fit_minimal", capture)

    x = _data_for(topology, n=60, p=5, seed=0)
    fit = gamfit.sae_manifold_fit(
        X=x, K=2, d_atom=d_atom, atom_topology=topology, assignment="softmax",
        n_iter=8, random_state=0,
    )
    assert "raw" in captured, "raw sae_manifold_fit_minimal payload was not captured"

    raw_json = json.dumps(_jsonable(captured["raw"]))
    penalties = list(fit.primitive_names[1:])
    core = builder(
        raw_json,
        x,
        str(fit.atom_topology),
        fit.assignment,
        fit.assignment_label,
        penalties,
        float(fit.alpha),
        bool(fit.learnable_alpha),
        float(fit.tau),
        float(fit.sparsity_strength),
        float(fit.smoothness),
        float(fit.learning_rate),
        int(fit.max_iter),
        int(fit.random_state),
        fit.top_k,
        float(fit.jumprelu_threshold),
    )

    old = fit.to_dict()
    new = core.to_dict()
    # Precondition: the mainline fixture must be non-finite-free so this test can
    # never silently exercise the NaN branch (whose handling is pinned separately).
    assert _no_nonfinite(old), "mainline equiv fixture unexpectedly carries a non-finite value"
    assert new == old


def test_builder_rejects_nonfinite_consistent_with_from_json():
    """The JSON-marshalled builder parses with serde, which rejects the bare
    `NaN` literal exactly as `ManifoldSaeCore.__new__` (`from_json`) does — so the
    two paths are NaN-CONSISTENT (both reject), no silent divergence. The distinct
    `py_any_to_json_value` Null policy (for the future perf direct-read builder) is
    pinned in test_sae_coercion_json_roundtrip."""
    builder = _builder()
    bad_json = '{"atom_plans": [], "atoms": [], "chosen_k": 0, "dispersion": NaN}'
    x = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        builder(
            bad_json, x, "euclidean", "softmax", "softmax", [],
            1.0, False, 0.5, 1.0, 1.0, 0.04, 50, 0, None, 0.0,
        )
