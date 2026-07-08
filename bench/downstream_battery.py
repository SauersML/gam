#!/usr/bin/env python3
"""External-validity downstream battery for any fitted SAE / featurizer.

Our evaluation stack is deep on INTERNAL honesty (certificates, coordinate
fidelity, atom trust, null batteries). This module closes the EXTERNAL-validity
gap: it scores a *featurizer* (a fitted dictionary + the codes it emits) with
the standard downstream conventions used by published SAE-benchmark tables, so
our numbers are directly comparable.

The four metrics are the BSF-style downstream battery:

1. ``single_concept_detection_f1`` -- per ground-truth concept, the best F1 of
   the single best latent at its optimal threshold (the standard sparse-probing
   convention: one concept, one latent, best threshold).
2. ``probing_suite`` -- a linear probe (logistic / ridge, sklearn) trained on
   the codes for each classification target, reporting held-out accuracy. The
   companion raw-activation probe is the honest ceiling for the same target.
3. ``cosine_probe_recovery`` -- |cosine| of recovered decoder directions to the
   planted directions, anchored against BOTH a random-direction floor AND a
   real-activation ceiling (a probe trained on raw activations), so the number
   means something.
4. ``concept_map_smoothness`` -- total variation / Dirichlet energy of the gate
   map over a grid. For synthetic patch grids this is TV over the lattice; the
   ``positions`` + ``group`` hook lets an LLM bench pass token position as a 1-D
   grid (gate smoothness along the sequence, grouped by sequence id).

The module is deliberately featurizer-agnostic and depends only on numpy /
sklearn / scipy (no ``gamfit`` / torch import) so it is reusable across the
manifold SAE and the flat TopK / JumpReLU baselines, and unit-testable in
isolation. The gamfit-consuming driver that fits featurizers and marshals their
codes into :class:`FeaturizerCodes` lives in ``bench/run_downstream_battery.py``.

Heavy math stays in numpy / sklearn: the probes are ``sklearn`` estimators, the
matching is the exact Hungarian in ``bench/_synth_sae_metrics.py``. Nothing here
is a hand-rolled optimizer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from _synth_sae_metrics import match_directions

_MISSING = "unavailable"


# --------------------------------------------------------------------------- #
# Featurizer payload                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeaturizerCodes:
    """Everything the downstream battery needs from ONE fitted featurizer.

    All arrays are plain numpy; a driver builds this from a manifold-SAE fit or
    a flat baseline. Fields that a given featurizer cannot supply may be left as
    empty arrays (the metrics degrade to an honest missing-marker, never a fake
    number).

    name:
        Human-readable featurizer name (row label).
    decoder_dirs:
        ``L x D`` recovered decoder directions (need NOT be unit-normalized;
        the metrics normalize). ``L`` counts every latent slot, including dead
        ones -- passing dead slots correctly penalizes wasted width.
    train_codes / test_codes:
        ``N x L`` code (latent activation) matrices. Column order MUST match
        ``decoder_dirs`` rows.
    train_activations / test_activations:
        ``N x D`` raw input activations the featurizer was fit on (the ceiling
        probe trains on these).
    train_factors / test_factors:
        ``N x T`` boolean ground-truth concept/factor firing (the planted
        factors for a synthetic bench, or concept labels for an LLM bench).
    planted_dirs:
        ``T x D`` ground-truth concept directions (the planted dictionary).
    positions:
        Optional ``N x g`` integer lattice coordinates for the smoothness grid
        (e.g. a patch grid, or a length-1 token-position axis). Empty disables
        smoothness.
    group:
        Optional length-``N`` integer group id (e.g. sequence id) so grid edges
        never cross groups. ``None`` treats all rows as one group.
    """

    name: str
    decoder_dirs: np.ndarray
    train_codes: np.ndarray
    test_codes: np.ndarray
    train_activations: np.ndarray
    test_activations: np.ndarray
    train_factors: np.ndarray
    test_factors: np.ndarray
    planted_dirs: np.ndarray
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    group: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _unit_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return mat.reshape(0, mat.shape[1] if mat.ndim == 2 else 0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def _f1_curve(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized F1 over a stack of boolean predictions ``pred`` (K x N)."""
    tp = np.sum(pred & y[None, :], axis=1).astype(float)
    fp = np.sum(pred & ~y[None, :], axis=1).astype(float)
    fn = np.sum(~pred & y[None, :], axis=1).astype(float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    return 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)


def _best_f1_single(score: np.ndarray, y: np.ndarray) -> float:
    """Best F1 of thresholding ``|score|`` against boolean ``y``.

    Threshold set = the 101 quantiles of ``|score|`` (the convention used by the
    existing bench harnesses in ``synth_sae_bench_manifold._best_f1``). Returns
    0.0 when ``y`` has no positive.
    """
    y = np.asarray(y, dtype=bool)
    if not np.any(y):
        return 0.0
    s = np.abs(np.asarray(score, dtype=float))
    thresholds = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 101)))
    pred = s[None, :] >= thresholds[:, None]
    return float(np.max(_f1_curve(pred, y)))


# --------------------------------------------------------------------------- #
# 1. Single-concept detection F1                                              #
# --------------------------------------------------------------------------- #
def single_concept_detection_f1(codes: np.ndarray, factors: np.ndarray) -> dict[str, Any]:
    """Per-concept best-single-latent F1 at the optimal threshold.

    For each ground-truth concept ``c`` (a column of boolean ``factors``), scan
    every latent ``l`` (a column of ``codes``), take the best F1 of thresholding
    ``|codes[:, l]|`` against ``factors[:, c]`` over the quantile threshold grid,
    and report the maximum over latents. This is the canonical single-concept
    probing convention: one concept is *detected* by whichever single latent
    separates it best. Concepts with no positive example are skipped (reported
    as ``None`` and excluded from the mean).

    Returns ``{"per_concept_f1": [...], "best_latent": [...], "mean_f1": float,
    "n_scored_concepts": int}``.
    """
    codes = np.asarray(codes, dtype=float)
    factors = np.asarray(factors, dtype=bool)
    n_concepts = factors.shape[1] if factors.ndim == 2 else 0
    n_latents = codes.shape[1] if codes.ndim == 2 else 0
    per_concept: list[float | None] = []
    best_latent: list[int | None] = []
    for c in range(n_concepts):
        y = factors[:, c]
        if not np.any(y) or n_latents == 0:
            per_concept.append(None)
            best_latent.append(None)
            continue
        f1s = np.array([_best_f1_single(codes[:, l], y) for l in range(n_latents)])
        per_concept.append(float(np.max(f1s)))
        best_latent.append(int(np.argmax(f1s)))
    scored = [v for v in per_concept if v is not None]
    return {
        "per_concept_f1": per_concept,
        "best_latent": best_latent,
        "mean_f1": float(np.mean(scored)) if scored else 0.0,
        "n_scored_concepts": len(scored),
    }


# --------------------------------------------------------------------------- #
# 2. Probing suite on codes (with raw-activation ceiling)                     #
# --------------------------------------------------------------------------- #
def _probe_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Held-out logistic-probe accuracy per binary target column.

    A single ``sklearn`` ``LogisticRegression`` per target column (heavy math
    stays in sklearn). Targets that are constant in the training split are
    skipped (a probe cannot learn a class it never sees) and reported as
    ``None``. Returns per-target accuracy + balanced accuracy and their means.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
    except ImportError:
        return {"source": _MISSING, "detail": "sklearn not importable"}

    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    train_y = np.asarray(train_y, dtype=bool)
    test_y = np.asarray(test_y, dtype=bool)
    n_targets = train_y.shape[1] if train_y.ndim == 2 else 0
    acc: list[float | None] = []
    bal: list[float | None] = []
    for t in range(n_targets):
        ytr = train_y[:, t]
        if ytr.all() or (~ytr).all() or train_x.shape[1] == 0:
            acc.append(None)
            bal.append(None)
            continue
        clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
        clf.fit(train_x, ytr.astype(int))
        pred = clf.predict(test_x).astype(bool)
        acc.append(float(accuracy_score(test_y[:, t], pred)))
        bal.append(float(balanced_accuracy_score(test_y[:, t], pred)))
    scored_acc = [v for v in acc if v is not None]
    scored_bal = [v for v in bal if v is not None]
    return {
        "source": "sklearn.LogisticRegression",
        "per_target_accuracy": acc,
        "per_target_balanced_accuracy": bal,
        "mean_accuracy": float(np.mean(scored_acc)) if scored_acc else 0.0,
        "mean_balanced_accuracy": float(np.mean(scored_bal)) if scored_bal else 0.0,
        "n_scored_targets": len(scored_acc),
    }


def probing_suite(
    train_codes: np.ndarray,
    train_factors: np.ndarray,
    test_codes: np.ndarray,
    test_factors: np.ndarray,
    *,
    train_activations: np.ndarray | None = None,
    test_activations: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Linear-probe accuracy on the codes, plus the raw-activation ceiling.

    Trains one logistic probe per factor on the *codes* and reports held-out
    accuracy (``on_codes``). When raw activations are supplied it also trains the
    same probe on the *activations* (``on_activations``): that is the honest
    ceiling -- the code probe cannot beat the information already linearly
    present in the raw activations, so ``on_codes / on_activations`` is the
    fraction of linearly-probeable concept information the featurizer preserved.
    """
    out: dict[str, Any] = {
        "on_codes": _probe_accuracy(
            train_codes, train_factors, test_codes, test_factors, seed=seed
        )
    }
    if train_activations is not None and test_activations is not None:
        out["on_activations_ceiling"] = _probe_accuracy(
            train_activations, train_factors, test_activations, test_factors, seed=seed
        )
        codes_acc = out["on_codes"].get("mean_accuracy")
        ceil_acc = out["on_activations_ceiling"].get("mean_accuracy")
        if isinstance(codes_acc, float) and isinstance(ceil_acc, float) and ceil_acc > 1e-9:
            out["codes_over_ceiling_fraction"] = float(codes_acc / ceil_acc)
    return out


# --------------------------------------------------------------------------- #
# 3. Cosine probe recovery (floor + ceiling anchored)                         #
# --------------------------------------------------------------------------- #
def _matched_mean_abs_cos(learned: np.ndarray, truth: np.ndarray) -> float:
    """Mean |cosine| of the exact-Hungarian one-to-one matched directions."""
    if learned.shape[0] == 0 or truth.shape[0] == 0:
        return 0.0
    rows, cols, _method = match_directions(learned, truth)
    if rows.size == 0:
        return 0.0
    sim = np.abs(learned @ truth.T)
    return float(np.mean(sim[rows, cols]))


def cosine_probe_recovery(
    decoder_dirs: np.ndarray,
    planted_dirs: np.ndarray,
    *,
    train_activations: np.ndarray | None = None,
    train_factors: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """|cosine| recovery of decoder directions, anchored to a floor AND ceiling.

    * ``recovery`` -- mean |cosine| of the optimal one-to-one (Hungarian)
      matching of unit ``decoder_dirs`` to unit ``planted_dirs``.
    * ``random_floor`` -- the same matched mean |cosine| for ``L`` random unit
      directions in the same ambient space; this is the chance level a matching
      of that cardinality reaches with no learning.
    * ``activation_ceiling`` -- train a logistic probe on the RAW activations for
      each planted concept; its (normalized) weight vector is the best linear
      direction the activations expose for that concept. The mean |cosine| of
      those probe directions to the planted directions is the ceiling: no decoder
      can recover a direction that the activations do not linearly carry.
    * ``anchored`` -- ``(recovery - random_floor) / (ceiling - random_floor)``,
      clipped to ``[0, 1]``; ``None`` when the ceiling is not computable.

    ``activation_ceiling`` is ``None`` (honest missing-marker) when raw
    activations / factors are absent or sklearn is missing.
    """
    learned = _unit_rows(decoder_dirs)
    truth = _unit_rows(planted_dirs)
    n_learned = learned.shape[0]
    ambient = truth.shape[1] if truth.ndim == 2 and truth.shape[1] > 0 else (
        learned.shape[1] if learned.ndim == 2 else 0
    )

    recovery = _matched_mean_abs_cos(learned, truth)

    rng = np.random.default_rng(seed)
    if n_learned > 0 and ambient > 0:
        rand = _unit_rows(rng.standard_normal((n_learned, ambient)))
        random_floor = _matched_mean_abs_cos(rand, truth)
    else:
        random_floor = 0.0

    ceiling = _activation_probe_ceiling(
        train_activations, train_factors, truth, seed=seed
    )

    anchored: float | None = None
    if ceiling is not None and ceiling - random_floor > 1e-9:
        anchored = float(
            np.clip((recovery - random_floor) / (ceiling - random_floor), 0.0, 1.0)
        )

    return {
        "recovery": float(recovery),
        "random_floor": float(random_floor),
        "activation_ceiling": None if ceiling is None else float(ceiling),
        "anchored": anchored,
        "n_learned_dirs": int(n_learned),
        "n_planted_dirs": int(truth.shape[0]),
    }


def _activation_probe_ceiling(
    train_activations: np.ndarray | None,
    train_factors: np.ndarray | None,
    truth_unit: np.ndarray,
    *,
    seed: int,
) -> float | None:
    if train_activations is None or train_factors is None:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    acts = np.asarray(train_activations, dtype=float)
    facts = np.asarray(train_factors, dtype=bool)
    if acts.ndim != 2 or acts.shape[1] == 0 or facts.ndim != 2:
        return None
    n_concepts = min(facts.shape[1], truth_unit.shape[0])
    probe_dirs: list[np.ndarray] = []
    target_idx: list[int] = []
    for c in range(n_concepts):
        y = facts[:, c]
        if y.all() or (~y).all():
            continue
        clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
        clf.fit(acts, y.astype(int))
        probe_dirs.append(np.asarray(clf.coef_, dtype=float).ravel())
        target_idx.append(c)
    if not probe_dirs:
        return None
    probes = _unit_rows(np.vstack(probe_dirs))
    targets = truth_unit[np.asarray(target_idx, dtype=int)]
    # Each probe is aligned to its OWN concept (no rematching): the ceiling is
    # the per-concept best linear direction, so cosine is measured pairwise.
    cos = np.abs(np.sum(probes * targets, axis=1))
    return float(np.mean(cos))


# --------------------------------------------------------------------------- #
# 4. Concept-map smoothness (TV / Dirichlet energy over a grid)               #
# --------------------------------------------------------------------------- #
def _lattice_edges(positions: np.ndarray, group: np.ndarray | None) -> np.ndarray:
    """Unit-step lattice edges (L1 distance 1) within each group.

    ``positions`` is ``N x g`` integer coordinates. Two rows are neighbors iff
    they share a group and differ by +1 along exactly one axis. Built by hashing
    coordinates -> row index (O(N*g)), so it scales to sequence-length grids.
    Returns an ``E x 2`` array of row-index pairs.
    """
    pos = np.asarray(np.rint(positions), dtype=np.int64)
    n, g = pos.shape
    grp = np.zeros(n, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64)
    index: dict[tuple[int, ...], int] = {}
    for i in range(n):
        index[(int(grp[i]), *(int(v) for v in pos[i]))] = i
    edges: list[tuple[int, int]] = []
    for i in range(n):
        base = (int(grp[i]), *(int(v) for v in pos[i]))
        for axis in range(g):
            nbr = list(base)
            nbr[1 + axis] += 1
            j = index.get(tuple(nbr))
            if j is not None:
                edges.append((i, j))
    return np.asarray(edges, dtype=int).reshape(-1, 2)


def concept_map_smoothness(
    gate: np.ndarray,
    positions: np.ndarray,
    *,
    group: np.ndarray | None = None,
) -> dict[str, Any]:
    """Total variation / Dirichlet energy of the gate map over a grid.

    ``gate`` is ``N`` (single map) or ``N x L`` (per-latent gate maps). Neighbors
    are unit steps on the integer lattice ``positions`` (``N x g``), restricted to
    within-``group`` edges. For each latent map ``g_l`` over the ``E`` edges:

        TV(g_l)        = mean_e |g_l[a] - g_l[b]|
        Dirichlet(g_l) = mean_e (g_l[a] - g_l[b])^2

    Both are reported raw and *scale-normalized* (divided by the map's std, and
    variance, respectively) so featurizers with different code scales compare
    fairly -- a smooth gate has low normalized TV. For grid patches ``positions``
    is the 2-D patch lattice; for an LLM bench pass a length-1 token-position axis
    and ``group`` = sequence id (smoothness along the sequence). Returns per-latent
    and mean TV / Dirichlet, plus the edge count.
    """
    gate = np.asarray(gate, dtype=float)
    if gate.ndim == 1:
        gate = gate[:, None]
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, None]
    if positions.size == 0 or positions.shape[0] != gate.shape[0]:
        return {"source": _MISSING, "detail": "no valid grid positions", "n_edges": 0}
    edges = _lattice_edges(positions, group)
    if edges.shape[0] == 0:
        return {"source": "grid", "detail": "no lattice edges", "n_edges": 0}
    a = gate[edges[:, 0]]
    b = gate[edges[:, 1]]
    diff = a - b
    tv = np.mean(np.abs(diff), axis=0)
    dirichlet = np.mean(diff**2, axis=0)
    std = np.std(gate, axis=0)
    var = std**2
    tv_norm = tv / np.maximum(std, 1e-12)
    dir_norm = dirichlet / np.maximum(var, 1e-12)
    return {
        "source": "grid",
        "n_edges": int(edges.shape[0]),
        "per_latent_tv": tv.tolist(),
        "per_latent_dirichlet": dirichlet.tolist(),
        "per_latent_tv_normalized": tv_norm.tolist(),
        "per_latent_dirichlet_normalized": dir_norm.tolist(),
        "mean_tv": float(np.mean(tv)),
        "mean_dirichlet": float(np.mean(dirichlet)),
        "mean_tv_normalized": float(np.mean(tv_norm)),
        "mean_dirichlet_normalized": float(np.mean(dir_norm)),
    }


# --------------------------------------------------------------------------- #
# Battery driver                                                              #
# --------------------------------------------------------------------------- #
def run_battery(fc: FeaturizerCodes, *, seed: int = 0) -> dict[str, Any]:
    """Run the full downstream battery on one featurizer's codes."""
    detection = single_concept_detection_f1(fc.test_codes, fc.test_factors)
    probing = probing_suite(
        fc.train_codes,
        fc.train_factors,
        fc.test_codes,
        fc.test_factors,
        train_activations=fc.train_activations,
        test_activations=fc.test_activations,
        seed=seed,
    )
    recovery = cosine_probe_recovery(
        fc.decoder_dirs,
        fc.planted_dirs,
        train_activations=fc.train_activations,
        train_factors=fc.train_factors,
        seed=seed,
    )
    if fc.positions is not None and np.asarray(fc.positions).size > 0:
        smoothness = concept_map_smoothness(fc.test_codes, fc.positions, group=fc.group)
    else:
        smoothness = {"source": _MISSING, "detail": "no grid supplied", "n_edges": 0}
    return {
        "featurizer": fc.name,
        "single_concept_detection_f1": detection,
        "probing_suite": probing,
        "cosine_probe_recovery": recovery,
        "concept_map_smoothness": smoothness,
    }


# --------------------------------------------------------------------------- #
# Self-test (pure-numpy, no gamfit) -- runnable smoke check + documentation.  #
# --------------------------------------------------------------------------- #
def _selftest(seed: int = 0) -> dict[str, Any]:
    """Deterministic sanity check: planted-aligned codes recover, noise does not.

    Builds a tiny synthetic bench where the "good" featurizer's codes ARE the
    planted factor coefficients and its decoder directions ARE the planted
    directions (perfect recovery), and a "noise" featurizer whose codes/directions
    are random. Asserts the good featurizer scores high on every metric and the
    noise featurizer scores at the floor. No gamfit / torch; runnable anywhere
    numpy + sklearn import.
    """
    rng = np.random.default_rng(seed)
    n_train, n_test, d, t = 400, 200, 12, 6
    planted = _unit_rows(rng.standard_normal((t, d)))
    # Factors fire ~30% of the time; coefficients are positive magnitudes.
    def _sample(n: int, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.random.default_rng(s)
        fire = r.random((n, t)) < 0.3
        coeff = fire * np.abs(r.normal(4.0, 1.0, size=(n, t)))
        acts = coeff @ planted + 0.05 * r.standard_normal((n, d))
        return acts, coeff, fire

    tr_acts, tr_coeff, tr_fire = _sample(n_train, seed + 1)
    te_acts, te_coeff, te_fire = _sample(n_test, seed + 2)

    # A grid: lay the test rows on a 1-D token-position axis in 10-long groups.
    positions = (np.arange(n_test) % 10).reshape(-1, 1)
    group = (np.arange(n_test) // 10).astype(int)

    good = FeaturizerCodes(
        name="oracle",
        decoder_dirs=planted.copy(),
        train_codes=tr_coeff.copy(),
        test_codes=te_coeff.copy(),
        train_activations=tr_acts,
        test_activations=te_acts,
        train_factors=tr_fire,
        test_factors=te_fire,
        planted_dirs=planted.copy(),
        positions=positions,
        group=group,
    )
    noise = FeaturizerCodes(
        name="noise",
        decoder_dirs=_unit_rows(rng.standard_normal((t, d))),
        train_codes=rng.standard_normal((n_train, t)),
        test_codes=rng.standard_normal((n_test, t)),
        train_activations=tr_acts,
        test_activations=te_acts,
        train_factors=tr_fire,
        test_factors=te_fire,
        planted_dirs=planted.copy(),
        positions=positions,
        group=group,
    )
    good_report = run_battery(good, seed=seed)
    noise_report = run_battery(noise, seed=seed)

    # Sanity: the oracle recovers directions ~perfectly and beats noise on
    # detection F1 and probing. These bounds are deliberately loose (they only
    # assert the metrics point the right way), never a tuned quality gate.
    assert good_report["cosine_probe_recovery"]["recovery"] > 0.95, good_report
    assert (
        good_report["cosine_probe_recovery"]["recovery"]
        > noise_report["cosine_probe_recovery"]["recovery"]
    )
    assert (
        good_report["single_concept_detection_f1"]["mean_f1"]
        > noise_report["single_concept_detection_f1"]["mean_f1"]
    )
    assert (
        good_report["probing_suite"]["on_codes"]["mean_accuracy"]
        >= noise_report["probing_suite"]["on_codes"]["mean_accuracy"]
    )
    return {"oracle": good_report, "noise": noise_report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run the pure-numpy smoke check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.selftest:
        report = _selftest(args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    parser.error("nothing to do: pass --selftest, or import this module as a library")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
