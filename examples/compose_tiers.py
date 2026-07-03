"""Tiered SAE composition driver: linear base (T1) + curved residual (T2).

This is the orchestration for a two-tier sparse dictionary:

    T1  a large collapsed-linear sparse dictionary (``sparse_dictionary_fit``)
        captures the bulk additive structure over the full corpus.
    T2  a SMALL curved SAE manifold (``sae_manifold_fit``) is fit on the T1
        *residual* (``X - T1_recon``) over a stratified subsample, then routed
        over the full residual to sculpt the structure the linear tier misses.

The single artifact is the additive composition ``T1_recon + T2_recon``. There
is no ``merge_tiers`` API in gamfit today (grep confirms), so we compose
additively in Python and return a :class:`TieredComposition` holding both tiers,
the combined reconstruction, and the explained-variance ledger.

Design notes (SPEC.md):
  * CLI flags, not env vars (argparse below).
  * Python is a thin wrapper: T1/T2 fitting and all heavy math live in the Rust
    fitters; this file only routes arrays, subsamples rows, and adds two
    reconstructions.
  * The whole corpus is never materialized for the curved fit -- T2 sees only a
    stratified subsample (curved fitting needs statistical sufficiency, not the
    whole corpus).

Integration seams (concurrent work):
  * Curved-tier kwargs ``structured_residual_passes`` / ``promote_from_residual``
    are plumbed through ``sae_manifold_fit`` (agent W1).
  * Sharded-memmap activation I/O (``examples/residual_shard_io.py`` -- agent W4,
    ``read_manifest`` / ``iter_shard_rows``) is imported lazily; when absent, the
    ``--synthetic`` path exercises the full pipeline.
  * The stratified draw replicates the ``gam_sae::corpus::rho_cascade`` contract
    (deterministic ``splitmix64`` row-hash Bernoulli inclusion + ``1/fraction``
    importance weight). That schedule is Rust-only today; ``_row_in_fraction``
    below mirrors its ``row_in_fraction`` so a subsample drawn here matches the
    one the streaming lane would pick. Swap to the Rust binding once exposed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import gamfit
from gamfit._sae_manifold import _default_ibp_concentration_for_k_atoms

from compose_artifact_schema import (
    explained_variance_train_mean,
    require_gamfit_version,
)

# Mask constant shared with `gam_sae::corpus::rho_cascade` (full u64 hash space).
_U64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(state: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer (matches ``gam_linalg::utils``)."""
    with np.errstate(over="ignore"):
        z = (state + np.uint64(0x9E3779B97F4A7C15)) & _U64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _U64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _U64
        return (z ^ (z >> np.uint64(31))) & _U64


def _row_in_fraction(row_ids: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Deterministic Bernoulli(``fraction``) row inclusion (rho_cascade contract).

    A row is included iff ``splitmix64(row_id ^ seed) < fraction * 2^64``. Mirrors
    ``gam_sae::corpus::rho_cascade::row_in_fraction`` so the subsample drawn here
    is the one the streaming ρ-cascade would pick for the same rows.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    keyed = (row_ids.astype(np.uint64) ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)) & _U64
    threshold = np.uint64(min(fraction, 1.0) * 2.0**64) if fraction < 1.0 else _U64
    return _splitmix64(keyed) <= threshold


def stratified_subsample(
    n_rows: int, target: int, seed: int
) -> tuple[np.ndarray, float]:
    """Draw <= ``target`` rows via the rho_cascade hashed-inclusion schedule.

    Returns ``(row_index, importance_weight)`` where ``importance_weight`` is
    ``1/fraction`` (the unbiasing weight an included row carries). If the corpus
    already fits in ``target`` rows the full set is returned with weight 1.
    """
    if target >= n_rows:
        return np.arange(n_rows, dtype=np.int64), 1.0
    fraction = target / float(n_rows)
    row_ids = np.arange(n_rows, dtype=np.uint64)
    mask = _row_in_fraction(row_ids, fraction, seed)
    return np.nonzero(mask)[0].astype(np.int64), 1.0 / fraction


def explained_variance(
    x: np.ndarray, recon: np.ndarray, *, train_mean: np.ndarray | None = None
) -> float:
    """EV ``1 - RSS/TSS`` against an explicit train-mean baseline.

    For in-sample synthetic smoke runs ``train_mean`` defaults to ``x.mean(0)``.
    Held-out driver code should pass the train-split Tier-0 mean explicitly.
    """
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    baseline = x.mean(axis=0) if train_mean is None else np.asarray(train_mean, dtype=np.float64)
    return explained_variance_train_mean(x, recon, baseline)


@dataclass
class TieredComposition:
    """The single composed artifact: additive T1 + T2 reconstruction."""

    t1: gamfit.SparseDictionaryFit
    t2: Any  # gamfit ManifoldSAE (joint) or StagewiseSAE (grown)
    t1_recon: np.ndarray
    t2_recon: np.ndarray
    combined_recon: np.ndarray
    t1_ev: float
    combined_ev: float
    subsample_rows: int
    importance_weight: float
    alternated: bool
    # Which curved-tier engine produced ``t2`` / the artifact: ``"joint"`` (the
    # simultaneous ``sae_manifold_fit(K=k2)``, routed OOS over the full residual)
    # or ``"stagewise"`` (the SAC grow-from-residual dictionary, evaluated on the
    # subsample it was composed on — the stagewise payload carries no OOS router).
    t2_engine: str = "joint"
    # The support ``combined_ev`` is measured on: ``"full_corpus"`` (joint OOS) or
    # ``"subsample"`` (stagewise in-sample composed dictionary).
    eval_support: str = "full_corpus"
    # The grown-vs-joint discriminator (SAC WS-A): both arms fit on the SAME
    # subsample, so the EV gap is the architecture datum, not a data-split
    # artifact. ``None`` when the discriminator was not requested. Keys:
    # ``joint_ev`` / ``stagewise_ev`` (combined EV on the subsample),
    # ``joint_chosen_k`` / ``stagewise_k``, ``stagewise_births_accepted``,
    # ``stagewise_collapse_events`` (0 by construction — the live-decoder collapse
    # answer on the real target), ``stagewise_ev_trace`` (monotone by construction).
    discriminator: dict | None = None

    @property
    def ev_gain(self) -> float:
        return self.combined_ev - self.t1_ev


def compose_tiers(
    X: np.ndarray,
    *,
    k1: int,
    k2: int,
    t1_active: int = 4,
    t1_max_epochs: int = 30,
    d_atom: int = 2,
    atom_topology: str = "circle",
    assignment: str = "threshold_gate",
    residual_passes: int = 3,
    promote_from_residual: bool = True,
    t2_n_iter: int = 50,
    subsample_tokens: int = 1_000_000,
    alternation: bool = True,
    t2_engine: str = "joint",
    discriminator: bool = False,
    stagewise_max_births: int = 24,
    stagewise_min_effect_ev: float = 0.0,
    random_state: int = 0,
) -> TieredComposition:
    """Fit the two-tier composition and return the combined artifact.

    All heavy math is inside the two Rust fitters. This function only: fits T1,
    forms the residual, draws the stratified subsample, fits the (small, curved)
    T2 on it, optionally runs one deflation alternation, and adds the two
    reconstructions.

    ``t2_engine`` selects the curved tier:

    * ``"joint"`` — the simultaneous ``sae_manifold_fit(K=k2)`` (the exact call
      that co-collapses on real activations), routed OOS over the FULL residual.
    * ``"stagewise"`` — the SAC ``sae_manifold_fit_stagewise`` grow-from-residual
      dictionary (forward births + backfitting, guards disarmed by construction).
      The compact stagewise payload carries no OOS router, so its artifact is
      evaluated ON THE SUBSAMPLE it was composed on (``eval_support="subsample"``).

    ``discriminator=True`` fits BOTH arms on the SAME subsample and records the
    grown-vs-joint comparison (EV gap, collapse-event count, births) in
    ``result.discriminator`` — the SAC WS-A log line, a matched in-sample compare.
    """
    require_gamfit_version()
    x = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n = x.shape[0]
    if k2 > 64:
        raise ValueError(f"curved tier K must stay small (K<=64); got {k2}")
    engine = str(t2_engine).strip().lower()
    if engine not in ("joint", "stagewise"):
        raise ValueError(f"t2_engine must be 'joint' or 'stagewise'; got {t2_engine!r}")

    # --- Tier 1: collapsed-linear sparse dictionary over the FULL corpus -----
    t1 = gamfit.sparse_dictionary_fit(
        x, K=k1, active=t1_active, max_epochs=t1_max_epochs
    )
    t1_recon = t1.fitted

    def _fit_joint(sub: np.ndarray) -> Any:
        return gamfit.sae_manifold_fit(
            sub,
            K=k2,
            d_atom=d_atom,
            atom_topology=atom_topology,
            assignment=assignment,
            # K-aware concentration so the ordered prior spans the whole (small)
            # dictionary instead of masking every atom past the first few.
            alpha=_default_ibp_concentration_for_k_atoms(k2),
            structured_residual_passes=residual_passes,
            promote_from_residual=promote_from_residual,
            n_iter=t2_n_iter,
            random_state=random_state,
        )

    def _fit_stagewise(sub: np.ndarray) -> Any:
        # SAC grow-from-residual: one K=1 seed, then evidence-gated births. K is
        # DISCOVERED (capped at ``k2`` births), not fixed — the grown analogue of
        # the joint K=k2 arm.
        return gamfit.sae_manifold_fit_stagewise(
            sub,
            d_atom=d_atom,
            atom_topology=atom_topology,
            assignment=assignment,
            max_births=int(min(stagewise_max_births, k2)),
            max_backfit_sweeps=4,
            min_effect_ev=float(stagewise_min_effect_ev),
            n_iter=t2_n_iter,
            random_state=random_state,
        )

    def _fit_curved(residual: np.ndarray) -> Any:
        # Stratified subsample: curved fitting needs statistical sufficiency,
        # never the whole corpus. Both arms fit on the SAME draw so the grown-vs-
        # joint comparison is matched.
        idx, weight = stratified_subsample(n, subsample_tokens, random_state)
        sub = np.ascontiguousarray(residual[idx])
        joint = _fit_joint(sub) if (engine == "joint" or discriminator) else None
        grown = _fit_stagewise(sub) if (engine == "stagewise" or discriminator) else None
        return joint, grown, sub, idx, weight

    def _stagewise_ev(grown: Any, sub: np.ndarray) -> float:
        recon = np.asarray(grown.reconstruct(), dtype=np.float64)
        return explained_variance(sub, recon)

    residual = x - t1_recon
    joint, grown, sub, idx, weight = _fit_curved(residual)

    # --- Optional one alternation (joint engine only: it routes the full corpus
    # for the T1 deflation target). ------------------------------------------
    alternated = False
    if alternation and engine == "joint" and joint is not None:
        t2_recon_full = np.asarray(joint.reconstruct(residual), dtype=np.float32)
        deflated = np.ascontiguousarray(x - t2_recon_full)
        t1 = gamfit.sparse_dictionary_fit(
            deflated, K=k1, active=t1_active, max_epochs=max(1, t1_max_epochs // 3)
        )
        t1_recon = t1.fitted
        residual = x - t1_recon
        joint, grown, sub, idx, weight = _fit_curved(residual)
        alternated = True

    # --- Assemble the discriminator (both arms on the subsample) -------------
    disc: dict | None = None
    if discriminator and joint is not None and grown is not None:
        joint_sub_recon = np.asarray(joint.reconstruct(sub), dtype=np.float64)
        joint_sub_ev = explained_variance(sub, joint_sub_recon)
        grown_sub_ev = _stagewise_ev(grown, sub)
        disc = {
            "subsample_rows": int(idx.size),
            "joint_ev": joint_sub_ev,
            "stagewise_ev": grown_sub_ev,
            "ev_gap_grown_minus_joint": grown_sub_ev - joint_sub_ev,
            "joint_chosen_k": int(getattr(joint, "chosen_k", len(joint.atoms))),
            "stagewise_k": int(grown.k),
            "stagewise_births_accepted": int(grown.births_accepted),
            "stagewise_births_rejected": int(grown.births_rejected),
            "stagewise_collapse_events": int(len(grown.collapse_events)),
            "stagewise_stopped_reason": grown.stopped_reason,
            "stagewise_ev_trace": [float(e) for e in grown.ev_trace],
        }

    # --- Build the artifact from the selected engine -------------------------
    if engine == "stagewise":
        # No OOS router: the composed dictionary reconstructs the subsample it was
        # grown on, so the artifact is evaluated on the subsample support.
        t2 = grown
        x_sub = np.ascontiguousarray(x[idx])
        t1_sub = np.ascontiguousarray(t1_recon[idx])
        t2_recon = np.asarray(grown.reconstruct(), dtype=np.float32)
        combined_recon = t1_sub + t2_recon
        t1_ev = explained_variance(x_sub, t1_sub)
        combined_ev = explained_variance(x_sub, combined_recon)
        eval_support = "subsample"
    else:
        t2 = joint
        t2_recon = np.asarray(joint.reconstruct(residual), dtype=np.float32)
        combined_recon = t1_recon + t2_recon
        t1_ev = explained_variance(x, t1_recon)
        combined_ev = explained_variance(x, combined_recon)
        eval_support = "full_corpus"

    return TieredComposition(
        t1=t1,
        t2=t2,
        t1_recon=t1_recon,
        t2_recon=t2_recon,
        combined_recon=combined_recon,
        t1_ev=t1_ev,
        combined_ev=combined_ev,
        subsample_rows=int(idx.size),
        importance_weight=weight,
        alternated=alternated,
        t2_engine=engine,
        eval_support=eval_support,
        discriminator=disc,
    )


@dataclass
class BlockNurseryComposition:
    """A block-sparse Tier-1 factored into per-block K=1 curved Tier-2 charts.

    The block-sparse T1 discovers ``G`` low-dimensional block subspaces; each block
    becomes a well-seeded nursery for ONE curved chart fit in the block's own b-dim
    coordinates (never the full ambient joint fit that co-collapses). The composed
    artifact is the additive sum of every block chart lifted back to ambient. This
    is the executable direction ⊂ block ⊂ chart ladder.
    """

    t1: Any  # gamfit.BlockSparseDictionaryFit
    manifest: dict
    per_block: list[dict]
    t1_block_ev: float
    composed_ev: float
    combined_recon: np.ndarray
    mdl: dict | None

    @property
    def ev_gain(self) -> float:
        return self.composed_ev - self.t1_block_ev


def emit_block_seed_manifest(
    t1: Any,
    X: np.ndarray,
    out_prefix: str | None,
    *,
    n_basis_chart: int = 4,
    residual_target: bool = True,
) -> dict:
    """Emit the Tier-1 -> Tier-2 seeds manifest (JSON + npz) a K=1-per-block curved
    stage consumes, mirroring the block_nursery hand-off (per-block basis ``Q``,
    in-block coordinates, and per-block stats + MDL featurizer rows).

    The compact JSON (``{prefix}.seeds.json``) carries the ``(p, b)`` bases + scalar
    stats + a flat ``mdl_featurizers`` list (ready for ``mdl.score_json``); the
    companion npz (``{prefix}.seeds.npz``) carries the heavy per-block arrays:
    ``block{g}_basis`` (p x b, column-orthonormal ``Q = D_gᵀ``) and ``block{g}_coords``
    (n x b, the residual-projected in-block coordinates the chart is fit on).
    Returns the manifest dict (also written to JSON when ``out_prefix`` is given).
    """
    x = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    manifest = t1.seed_manifest(
        x, n_basis_chart=n_basis_chart, residual_target=residual_target
    )
    if out_prefix is not None:
        prefix = Path(out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".seeds.json").write_text(json.dumps(manifest, indent=2))
        arrays: dict[str, np.ndarray] = {}
        for g in range(t1.n_blocks):
            arrays[f"block{g}_basis"] = t1.block_frame(g).T  # (p, b) = Q
            coords = (
                t1.project_residual(x, g) if residual_target else t1.block_coords(x, g)
            )
            arrays[f"block{g}_coords"] = coords
        np.savez(str(prefix.with_suffix(".seeds.npz")), **arrays)
        print(
            f"[compose_tiers] wrote seeds manifest "
            f"{prefix.with_suffix('.seeds.json')} (+ .seeds.npz, {t1.n_blocks} blocks)"
        )
    return manifest


def _score_block_mdl(manifest: dict) -> dict | None:
    """Score the manifest's block-vs-chart crossover ``f*`` per block via M-mdl's
    scorer, if it is importable. Pure-numpy JSON path (no model load)."""
    import importlib
    import sys as _sys

    candidates = [
        Path("/Users/user/Manifold-SAE/experiments/mdl_ladder"),
        Path(__file__).resolve().parent.parent.parent / "Manifold-SAE"
        / "experiments" / "mdl_ladder",
    ]
    for cand in candidates:
        if (cand / "mdl.py").exists():
            _sys.path.insert(0, str(cand))
            try:
                mdl = importlib.import_module("mdl")
            except Exception:
                return None
            feats = manifest["mdl_featurizers"]
            payload = {
                "delta2": None,
                "l_param_bits": None,
                "featurizers": feats,
            }
            # Crossover f* resolves from PAYLOAD-level block_name/chart_name (per the
            # mdl_ladder schema). Point them at the first block's own two rungs so the
            # response carries a block->chart f* for that representative block.
            if len(feats) >= 2:
                payload["block_name"] = feats[0]["name"]
                payload["chart_name"] = feats[1]["name"]
            try:
                return mdl.score_json(payload)
            except Exception as exc:  # pragma: no cover - scorer-side issue
                return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return None


def compose_block_charts(
    X: np.ndarray,
    *,
    n_blocks: int,
    block_size: int = 2,
    block_topk: int = 1,
    t1_max_epochs: int = 30,
    aux_k: int = 0,
    chart_n_iter: int = 50,
    chart_topology: str = "circle",
    chart_d_atom: int = 2,
    n_basis_chart: int = 4,
    residual_target: bool = True,
    seed_out: str | None = None,
    score_mdl: bool = True,
    random_state: int = 0,
) -> BlockNurseryComposition:
    """Block-sparse Tier-1 -> per-block K=1 curved Tier-2 nursery -> additive compose.

    (1) Fit a block-sparse T1 (``block_sparse_dictionary_fit``). (2) Emit the seeds
    manifest. (3) For EACH block fit ONE curved chart (``sae_manifold_fit`` K=1,
    ``d_atom = block_size``) on that block's in-block coordinates — a tiny, well-
    seeded ``d≈b`` fit, never the full-ambient joint fit that co-collapses — then
    lift the chart back with the block frame and sum. Composition EV is measured
    honestly against ``X``.

    Only the seeds manifest + per-block coords cross the T1/T2 seam, so the two
    stages stay decoupled (a different T2 fitter can consume the same manifest).
    """
    x = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    t1 = gamfit.block_sparse_dictionary_fit(
        x,
        n_blocks,
        block_size=block_size,
        block_topk=block_topk,
        max_epochs=t1_max_epochs,
        aux_k=aux_k,
    )
    manifest = emit_block_seed_manifest(
        t1, x, seed_out, n_basis_chart=n_basis_chart, residual_target=residual_target
    )

    composed = np.zeros_like(x, dtype=np.float64)
    firings = t1.block_firings(x)
    per_block: list[dict] = []
    for g in range(t1.n_blocks):
        coords = (
            t1.project_residual(x, g) if residual_target else t1.block_coords(x, g)
        )
        rec: dict[str, Any] = {
            "block": g,
            "block_dim": int(block_size),
            "n_firings": int(firings[g]),
            "utilization": float(t1.block_utilization[g]),
            "stable_rank": float(t1.block_stable_rank[g]),
        }
        # One tiny curved chart in the block's own coordinates.
        try:
            chart = gamfit.sae_manifold_fit(
                np.ascontiguousarray(coords, dtype=np.float32),
                K=1,
                # d_atom is the atom manifold's embedding dim (a circle is 2), not
                # the block width; clamp to the b-dim block coordinate space.
                d_atom=min(chart_d_atom, block_size),
                atom_topology=chart_topology,
                n_iter=chart_n_iter,
                random_state=random_state,
            )
            coords_hat = np.asarray(chart.reconstruct(coords), dtype=np.float64)
            rec["chart_status"] = "CONVERGED"
            rec["chart_ev_block_coords"] = explained_variance(coords, coords_hat)
            composed += t1.lift_block(coords_hat.astype(np.float32), g).astype(np.float64)
        except Exception as exc:  # curved fit unavailable / hangs in this env
            rec["chart_status"] = f"{type(exc).__name__}"
            rec["chart_error"] = str(exc)[:160]
            # Fall back to the block's LINEAR reconstruction so the composition is
            # still defined (the direction⊂block rung without the chart rung).
            composed += t1.lift_block(coords, g).astype(np.float64)
            rec["chart_ev_block_coords"] = None
        per_block.append(rec)

    combined = composed.astype(np.float32)
    mdl = _score_block_mdl(manifest) if score_mdl else None
    return BlockNurseryComposition(
        t1=t1,
        manifest=manifest,
        per_block=per_block,
        t1_block_ev=float(t1.explained_variance),
        composed_ev=explained_variance(x, combined),
        combined_recon=combined,
        mdl=mdl,
    )


def load_seed_manifest(prefix: str) -> tuple[dict, dict, dict]:
    """Load a seeds manifest written by :func:`emit_block_seed_manifest`.

    Returns ``(manifest, bases, coords)`` where ``bases[g]`` is the ``(p, b)``
    column-orthonormal basis ``Q`` and ``coords[g]`` the ``(n, b)`` in-block
    coordinates for block ``g`` — everything a Tier-2 stage needs, read from files
    ALONE (no Tier-1 fit object). This is the decoupled T1/T2 seam made literal.
    """
    p = Path(prefix)
    manifest = json.loads(p.with_suffix(".seeds.json").read_text())
    npz = np.load(str(p.with_suffix(".seeds.npz")))
    bases, coords = {}, {}
    for g in range(int(manifest["n_blocks"])):
        bases[g] = np.ascontiguousarray(npz[f"block{g}_basis"], dtype=np.float32)
        coords[g] = np.ascontiguousarray(npz[f"block{g}_coords"], dtype=np.float32)
    return manifest, bases, coords


def compose_charts_from_manifest(
    prefix: str,
    *,
    X: np.ndarray | None = None,
    chart_n_iter: int = 50,
    chart_topology: str = "circle",
    chart_d_atom: int = 2,
    random_state: int = 0,
) -> tuple[dict, np.ndarray]:
    """Tier-2 stage that consumes a seeds manifest FROM FILES (no Tier-1 object):
    fit one K=1 curved chart per block on the stored in-block coordinates, lift
    with the stored basis ``Q``, and additively compose.

    Returns ``(report, combined_recon)``; when the original ``X`` is supplied the
    report includes the composed ambient EV (else only per-block chart EV in the
    block's own coordinates, which needs no ambient reference). Proves the block ->
    chart hand-off is a genuine decoupled file boundary a separate T2 driver
    (e.g. block_nursery) can consume.
    """
    manifest, bases, coords = load_seed_manifest(prefix)
    n_blocks = int(manifest["n_blocks"])
    p = int(manifest["ambient_p"])
    n_rows = next(iter(coords.values())).shape[0]
    composed = np.zeros((n_rows, p), dtype=np.float64)
    per_block: list[dict] = []
    for g in range(n_blocks):
        q = bases[g]  # (p, b)
        z = coords[g]  # (n, b)
        b = int(q.shape[1])
        rec: dict[str, Any] = {"block": g, "block_dim": b}
        try:
            chart = gamfit.sae_manifold_fit(
                np.ascontiguousarray(z, dtype=np.float32),
                K=1,
                d_atom=min(chart_d_atom, b),
                atom_topology=chart_topology,
                n_iter=chart_n_iter,
                random_state=random_state,
            )
            z_hat = np.asarray(chart.reconstruct(z), dtype=np.float64)
            rec["chart_status"] = "CONVERGED"
            rec["chart_ev_block_coords"] = explained_variance(z, z_hat)
            composed += z_hat @ q.T  # lift with the stored (p, b) basis
        except Exception as exc:  # curved fit unavailable / hangs in this env
            rec["chart_status"] = f"{type(exc).__name__}"
            rec["chart_error"] = str(exc)[:160]
            composed += z.astype(np.float64) @ q.T  # linear-lift fallback
            rec["chart_ev_block_coords"] = None
        per_block.append(rec)
    combined = composed.astype(np.float32)
    report: dict[str, Any] = {
        "manifest_prefix": str(prefix),
        "n_blocks": n_blocks,
        "per_block": per_block,
        "combined_recon_shape": list(combined.shape),
    }
    if X is not None:
        x = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if x.shape == combined.shape:
            report["composed_ambient_ev"] = explained_variance(x, combined)
    return report, combined


def _load_activations(args: argparse.Namespace) -> np.ndarray:
    """Load real sharded activations (W4) or synthesize planted structure."""
    if args.shard_manifest is not None:
        # Lazy import: examples/residual_shard_io.py (agent W4). Interface:
        # read_manifest(path) -> manifest; iter_shard_rows(manifest) -> row blocks.
        try:
            from residual_shard_io import iter_shard_rows, read_manifest
        except ImportError as exc:  # pragma: no cover - depends on W4 landing
            raise SystemExit(
                "residual_shard_io not found; --shard-manifest needs agent W4's "
                f"examples/residual_shard_io.py on sys.path ({exc})"
            )
        manifest = read_manifest(args.shard_manifest)
        blocks = [np.asarray(b, dtype=np.float32) for b in iter_shard_rows(manifest)]
        return np.concatenate(blocks, axis=0)
    return _planted_activations(args)


def _planted_activations(args: argparse.Namespace) -> np.ndarray:
    """Synthetic corpus: a linear sparse-dictionary tier + a curved tier.

    The linear part is what T1 should absorb; the curved (sin/cos + harmonic)
    part is genuinely nonlinear, so a linear dictionary cannot fully explain it
    and the curved T2 tier must add reconstruction (the EV assertion).
    """
    rng = np.random.default_rng(args.random_state)
    n, p, k1, k2 = args.n_tokens, args.p, args.k1, args.k2

    # Linear tier: sparse combinations of k1 unit atoms.
    lin_dict = rng.standard_normal((k1, p)).astype(np.float32)
    lin_dict /= np.linalg.norm(lin_dict, axis=1, keepdims=True)
    codes = np.zeros((n, k1), dtype=np.float32)
    for i in range(n):
        active = rng.choice(k1, size=args.t1_active, replace=False)
        codes[i, active] = rng.standard_normal(args.t1_active).astype(np.float32)
    linear_part = codes @ lin_dict

    # Curved tier: each row assigned to one of k2 circle atoms; the signal is a
    # nonlinear (first + second harmonic) function of a per-row angle.
    u = rng.standard_normal((k2, p)).astype(np.float32)
    v = rng.standard_normal((k2, p)).astype(np.float32)
    w = rng.standard_normal((k2, p)).astype(np.float32)
    assign = rng.integers(0, k2, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    curved_part = (
        np.sin(theta)[:, None] * u[assign]
        + np.cos(theta)[:, None] * v[assign]
        + 0.4 * np.sin(2.0 * theta)[:, None] * w[assign]
    ).astype(np.float32)

    noise = (args.noise * rng.standard_normal((n, p))).astype(np.float32)
    return np.ascontiguousarray(linear_part + args.curved_scale * curved_part + noise)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_argument_group("data source")
    src.add_argument("--shard-manifest", default=None,
                     help="path to a W4 residual-shard manifest (real activations)")
    src.add_argument("--synthetic", action="store_true",
                     help="use planted synthetic activations (default when no manifest)")
    src.add_argument("--n-tokens", type=int, default=5000)
    src.add_argument("--p", type=int, default=64)
    src.add_argument("--noise", type=float, default=0.05)
    src.add_argument("--curved-scale", type=float, default=1.0)

    tiers = ap.add_argument_group("tiers")
    tiers.add_argument("--t1-mode", choices=("sparse", "block"), default="sparse",
                       help="Tier-1 engine: 'sparse' (collapsed-linear atoms) or "
                            "'block' (block-sparse -> per-block curved nursery)")
    tiers.add_argument("--k1", type=int, default=16, help="linear dictionary width")
    tiers.add_argument("--k2", type=int, default=8, help="curved SAE width (K<=64)")
    tiers.add_argument("--t1-active", type=int, default=4)
    tiers.add_argument("--t1-max-epochs", type=int, default=30)
    tiers.add_argument("--d-atom", type=int, default=2)
    tiers.add_argument("--atom-topology", default="circle")
    tiers.add_argument("--assignment", default="threshold_gate")
    tiers.add_argument("--t2-n-iter", type=int, default=50)
    tiers.add_argument("--t2-engine", choices=("joint", "stagewise"), default="joint",
                       help="curved-tier engine: 'joint' sae_manifold_fit(K=k2) or "
                            "'stagewise' SAC grow-from-residual (sae_manifold_fit_stagewise)")
    tiers.add_argument("--discriminator", action="store_true", default=False,
                       help="fit BOTH curved arms on the same subsample and log the "
                            "grown-vs-joint EV gap + collapse-event count (SAC WS-A)")
    tiers.add_argument("--stagewise-max-births", type=int, default=24,
                       help="SAC forward-birth safety cap (capped further by k2)")
    tiers.add_argument("--stagewise-min-effect-ev", type=float, default=0.0,
                       help="SAC minimum-effect salience floor a birth's dEV must clear "
                            "(0.0 = null-recovering, evidence-only)")

    block = ap.add_argument_group("block tier (--t1-mode block)")
    block.add_argument("--n-blocks", type=int, default=8,
                       help="number of block subspaces G (K = G*block_size)")
    block.add_argument("--block-size", type=int, default=2, help="atoms per block b (2-4)")
    block.add_argument("--block-topk", type=int, default=1, help="blocks active per row")
    block.add_argument("--aux-k", type=int, default=2, help="AuxK dead-block revival budget")
    block.add_argument("--chart-n-iter", type=int, default=50,
                       help="curved chart n_iter per block")
    block.add_argument("--n-basis-chart", type=int, default=4,
                       help="Fourier basis count per circle chart (for MDL n_params)")
    block.add_argument("--direct-coords", dest="residual_target", action="store_false",
                       default=True,
                       help="seed charts on direct block coords X D_gᵀ instead of the "
                            "leave-one-block-out residual (default: residual)")
    block.add_argument("--emit-seeds", default=None,
                       help="path prefix to write the seeds manifest "
                            "(<prefix>.seeds.json + <prefix>.seeds.npz)")
    block.add_argument("--from-seeds", default=None,
                       help="Tier-2 ONLY: consume an existing seeds manifest prefix "
                            "(fit K=1 chart per block from files, no Tier-1 refit)")
    block.add_argument("--no-mdl", dest="score_mdl", action="store_false", default=True,
                       help="skip the block-vs-chart f* MDL scoring")

    resid = ap.add_argument_group("residual / composition")
    resid.add_argument("--residual-passes", type=int, default=3)
    resid.add_argument("--promote-from-residual", dest="promote_from_residual",
                       action="store_true", default=True)
    resid.add_argument("--no-promote-from-residual", dest="promote_from_residual",
                       action="store_false")
    resid.add_argument("--alternation", dest="alternation",
                       action="store_true", default=True,
                       help="one deflation alternation (default on)")
    resid.add_argument("--no-alternation", dest="alternation", action="store_false")
    resid.add_argument("--subsample-tokens", type=int, default=1_000_000,
                       help="stratified curved-fit subsample target (rho_cascade draw)")
    ap.add_argument("--random-state", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)

    if args.from_seeds is not None:
        # Tier-2 ONLY: run per-block charts straight off a seeds manifest, proving
        # the T1/T2 seam is a decoupled file boundary. X is loaded only to score
        # the composed ambient EV (skipped if shapes disagree).
        X = _load_activations(args)
        report, _ = compose_charts_from_manifest(
            args.from_seeds,
            X=X,
            chart_n_iter=args.chart_n_iter,
            chart_topology=args.atom_topology,
            chart_d_atom=args.d_atom,
            random_state=args.random_state,
        )
        n_charts = sum(1 for r in report["per_block"] if r.get("chart_status") == "CONVERGED")
        print(f"[compose_tiers] Tier-2 from seeds {args.from_seeds}: "
              f"{n_charts}/{report['n_blocks']} charts fit")
        if "composed_ambient_ev" in report:
            print(f"[compose_tiers] composed ambient EV = {report['composed_ambient_ev']:.4f}")
        return report

    X = _load_activations(args)

    if args.t1_mode == "block":
        print(f"[compose_tiers] corpus X: {X.shape} "
              f"(T1 block-sparse G={args.n_blocks} b={args.block_size} "
              f"top-k={args.block_topk} -> per-block K=1 {args.atom_topology} charts)")
        bn = compose_block_charts(
            X,
            n_blocks=args.n_blocks,
            block_size=args.block_size,
            block_topk=args.block_topk,
            t1_max_epochs=args.t1_max_epochs,
            aux_k=args.aux_k,
            chart_n_iter=args.chart_n_iter,
            chart_topology=args.atom_topology,
            chart_d_atom=args.d_atom,
            n_basis_chart=args.n_basis_chart,
            residual_target=args.residual_target,
            seed_out=args.emit_seeds,
            score_mdl=args.score_mdl,
            random_state=args.random_state,
        )
        n_charts = sum(1 for r in bn.per_block if r.get("chart_status") == "CONVERGED")
        print(f"[compose_tiers] T1 block-sparse EV = {bn.t1_block_ev:.4f}")
        print(f"[compose_tiers] composed (block->chart) EV = {bn.composed_ev:.4f} "
              f"(+{bn.ev_gain:.4f} from curved charts; {n_charts}/{len(bn.per_block)} "
              f"charts fit)")
        if bn.mdl and "crossover" in bn.mdl:
            xo = bn.mdl["crossover"]
            print(f"[compose_tiers] MDL block->chart f* = {xo.get('f_star')} "
                  f"(chart wins at actual f: {xo.get('chart_wins_at_actual_f')})")
        return bn

    print(f"[compose_tiers] corpus X: {X.shape} "
          f"(T1 K={args.k1}, T2 K={args.k2}, topology={args.atom_topology})")

    result = compose_tiers(
        X,
        k1=args.k1,
        k2=args.k2,
        t1_active=args.t1_active,
        t1_max_epochs=args.t1_max_epochs,
        d_atom=args.d_atom,
        atom_topology=args.atom_topology,
        assignment=args.assignment,
        residual_passes=args.residual_passes,
        promote_from_residual=args.promote_from_residual,
        t2_n_iter=args.t2_n_iter,
        subsample_tokens=args.subsample_tokens,
        alternation=args.alternation,
        t2_engine=args.t2_engine,
        discriminator=args.discriminator,
        stagewise_max_births=args.stagewise_max_births,
        stagewise_min_effect_ev=args.stagewise_min_effect_ev,
        random_state=args.random_state,
    )

    print(f"[compose_tiers] curved fit ({result.t2_engine}) on {result.subsample_rows} rows "
          f"(importance weight {result.importance_weight:.3f}), "
          f"alternation={'on' if result.alternated else 'off'}")
    print(f"[compose_tiers] T1-only EV   = {result.t1_ev:.4f}  (support={result.eval_support})")
    print(f"[compose_tiers] combined  EV = {result.combined_ev:.4f} "
          f"(+{result.ev_gain:.4f} from the curved tier)")
    if result.discriminator is not None:
        d = result.discriminator
        # The SAC WS-A grown-vs-joint discriminator, matched on one subsample.
        print(f"[compose_tiers] DISCRIMINATOR (grown vs joint, subsample={d['subsample_rows']}): "
              f"stagewise EV={d['stagewise_ev']:.4f} (K={d['stagewise_k']}, "
              f"births={d['stagewise_births_accepted']}, collapse_events="
              f"{d['stagewise_collapse_events']}) vs joint EV={d['joint_ev']:.4f} "
              f"(K={d['joint_chosen_k']}); gap={d['ev_gap_grown_minus_joint']:+.4f}")
    hs = getattr(result.t2, "hybrid_split", None)
    if hs is not None:
        print(f"[compose_tiers] curved-tier hybrid_split keys: {sorted(hs)}")
    return result


if __name__ == "__main__":
    main()
