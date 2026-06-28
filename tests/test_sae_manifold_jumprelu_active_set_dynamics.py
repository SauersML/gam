"""End-to-end coverage for JumpReLU (``assignment='jumprelu'``) active-set
dynamics during a real SAE-manifold fit.

JumpReLU uses variable-stride compact row layouts where the per-row active
atoms (logit > threshold) change during optimization. The Rust kernel
recomputes the active set on every ``assemble_arrow_schur`` call, updates
logits in ``apply_newton_step``, and expands the compact delta against the
layout from the most recent assembly. The existing ``jumprelu_ste.rs``
tests only check penalty value/gradient at fixed thresholds, and
``test_sae_manifold_top_k_issue.py`` exercises softmax top-k, not the
JumpReLU path. Nothing exercises the variable-stride active-set evolution
end to end.

These tests fit on data with distinct per-atom activation patterns and
confirm (1) active sets genuinely differ across rows (the variable-stride
path is exercised) and (2) reconstruction quality holds as active sets
evolve. Assertions are framed across the row population (not per-row) so a
single row that keeps a stable active set cannot cause a false negative.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# #1512: this fit exceeds the standard Python-API CI runner budget (>60s in
# triage), so it is tagged slow and excluded from the directory-level
# `-m "not slow"` CI step while still being collected (run by a bare pytest).
pytestmark = pytest.mark.slow

gamfit = pytest.importorskip("gamfit")


def _multimodal_atom_data(
    n: int, p: int, k: int, noise: float, seed: int
) -> np.ndarray:
    """``k`` circular harmonics, each switched ON for a disjoint subset of
    rows. Different rows therefore want different active atoms, which is
    exactly what should drive heterogeneous JumpReLU gates."""
    rng = np.random.default_rng(seed)
    z = np.zeros((n, p), dtype=float)
    # Assign each row a random subset of "on" atoms (1..k of them) so the
    # ideal active set is genuinely row-dependent and multi-modal.
    for i in range(n):
        n_on = int(rng.integers(1, k + 1))
        on = rng.choice(k, size=n_on, replace=False)
        for atom in on:
            theta = rng.uniform(0.0, 2.0 * math.pi)
            harm = np.array([math.cos(theta), math.sin(theta)])
            mixing = rng.normal(size=(2, p))
            mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
            z[i] += harm @ mixing
    z += noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _active_sets(assignments: np.ndarray, threshold: float) -> list[frozenset[int]]:
    """Per-row set of active atom indices under a gate threshold."""
    A = np.asarray(assignments)
    return [frozenset(np.flatnonzero(row > threshold).tolist()) for row in A]


def test_jumprelu_active_sets_change_across_iterations():
    """Two JumpReLU fits that differ only in iteration budget must end with
    different active-set configurations across the row population — the
    variable-stride layout is rebuilt every Newton step as logits cross
    the threshold, so the active-set distribution should not be frozen at
    its initial value. We also confirm the gate is not collapsed: across
    rows, every atom is active for at least one row (no atom is
    permanently dead)."""
    z = _multimodal_atom_data(n=300, p=48, k=3, noise=0.05, seed=1)
    threshold = 0.05

    snapshots: dict[int, list[frozenset[int]]] = {}
    for iters in (1, 10, 40):
        fit = gamfit.sae_manifold_fit(
            X=z,
            K=3,
            atom_basis="periodic",
            d_atom=2,
            assignment="jumprelu",
            n_iter=iters,
            learning_rate=0.05,
            random_state=1,
        )
        A = np.asarray(fit.assignments)
        assert A.shape == (z.shape[0], 3), (
            f"JumpReLU assignments shape {A.shape} should be (n=300, K=3)"
        )
        assert np.all(np.isfinite(A)), (
            f"JumpReLU assignments at {iters} iters contain non-finite entries"
        )
        snapshots[iters] = _active_sets(A, threshold)

    # The active-set configuration over the row population must evolve as
    # iterations increase. Compare the *multiset* of per-row active sets;
    # if the variable-stride path were inert this would be identical.
    early = sorted(map(tuple, map(sorted, snapshots[1])))
    late = sorted(map(tuple, map(sorted, snapshots[40])))
    assert early != late, (
        "JumpReLU active-set distribution did not change between 1 and 40 "
        "iterations; the variable-stride active-set recomputation appears "
        "inert. Early distribution == late distribution.\n"
        f"early sizes={sorted(len(s) for s in snapshots[1])}\n"
        f"late sizes={sorted(len(s) for s in snapshots[40])}"
    )

    # Across the row population there must be more than one distinct active
    # set (rows are heterogeneous) — otherwise the gate degenerated to a
    # single global support.
    distinct_late = set(map(frozenset, snapshots[40]))
    assert len(distinct_late) > 1, (
        "all rows ended with an identical active set under JumpReLU; the "
        "per-row gate is not heterogeneous as the multimodal data requires."
    )

    # No atom may be permanently inactive: the union of active atoms over
    # all rows must include every atom index.
    union = frozenset().union(*snapshots[40]) if snapshots[40] else frozenset()
    assert union == frozenset(range(3)), (
        f"some atoms are never active under JumpReLU; active union={set(union)}, "
        f"expected all of {{0, 1, 2}}."
    )


def test_jumprelu_reconstruction_stable_with_variable_active_sets():
    """As JumpReLU active sets evolve over a multi-iteration fit, held-out
    reconstruction must remain accurate (R^2 >= 0.65). This guards the
    compact-delta expansion: each Arrow-Schur assembly sizes the delta for
    the *current* layout, and ``apply_newton_step`` expands it against that
    same layout. An off-by-one or stale-layout mismatch would corrupt the
    Newton step and collapse OOS accuracy."""
    z_full = _multimodal_atom_data(n=400, p=48, k=4, noise=0.05, seed=2)
    z_train = z_full[:250]
    z_test = z_full[250:]

    fit = gamfit.sae_manifold_fit(
        X=z_train,
        K=4,
        atom_basis="periodic",
        d_atom=2,
        assignment="jumprelu",
        n_iter=40,
        learning_rate=0.05,
        random_state=2,
    )

    assert hasattr(fit, "reconstruct"), (
        "JumpReLU fit must expose reconstruct() for OOS scoring"
    )
    oos_fitted = fit.reconstruct(z_test)
    assert np.all(np.isfinite(oos_fitted)), (
        "OOS reconstruction under JumpReLU produced non-finite entries — "
        "likely a compact-delta expansion mismatch as active sets changed."
    )

    oos = _r2(z_test, oos_fitted)
    assert oos >= 0.65, (
        f"JumpReLU OOS reconstruction R^2 = {oos:.4f}; expected >= 0.65. "
        f"Accuracy collapsing here points at the variable-stride active-set "
        f"path corrupting the Newton step as gates flip."
    )
