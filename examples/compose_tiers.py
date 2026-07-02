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
from dataclasses import dataclass
from typing import Any

import numpy as np

import gamfit
from gamfit._sae_manifold import _default_ibp_concentration_for_k_atoms

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


def explained_variance(x: np.ndarray, recon: np.ndarray) -> float:
    """Held-in EV ``1 - RSS/TSS`` against the column-mean baseline."""
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


@dataclass
class TieredComposition:
    """The single composed artifact: additive T1 + T2 reconstruction."""

    t1: gamfit.SparseDictionaryFit
    t2: Any  # gamfit ManifoldSAE
    t1_recon: np.ndarray
    t2_recon: np.ndarray
    combined_recon: np.ndarray
    t1_ev: float
    combined_ev: float
    subsample_rows: int
    importance_weight: float
    alternated: bool

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
    random_state: int = 0,
) -> TieredComposition:
    """Fit the two-tier composition and return the combined artifact.

    All heavy math is inside the two Rust fitters. This function only: fits T1,
    forms the residual, draws the stratified subsample, fits the (small, curved)
    T2 on it and routes it over the full residual, optionally runs one deflation
    alternation, and adds the two reconstructions.
    """
    x = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n = x.shape[0]
    if k2 > 64:
        raise ValueError(f"curved tier K must stay small (K<=64); got {k2}")

    # --- Tier 1: collapsed-linear sparse dictionary over the FULL corpus -----
    t1 = gamfit.sparse_dictionary_fit(
        x, K=k1, active=t1_active, max_epochs=t1_max_epochs
    )
    t1_recon = t1.fitted

    def _fit_curved(residual: np.ndarray) -> Any:
        # Stratified subsample: curved fitting needs statistical sufficiency,
        # never the whole corpus. Fit on the draw, then route the full residual.
        idx, weight = stratified_subsample(n, subsample_tokens, random_state)
        sub = np.ascontiguousarray(residual[idx])
        fit = gamfit.sae_manifold_fit(
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
        return fit, idx, weight

    residual = x - t1_recon
    t2, idx, weight = _fit_curved(residual)
    t2_recon = np.asarray(t2.reconstruct(residual), dtype=np.float32)

    # --- Optional one alternation: deflate T2, warm-refit T1, refit T2 -------
    # There is no decoder-seeding API on sparse_dictionary_fit, so the "warm"
    # T1 epoch is a short refit on the T2-deflated target (X - T2_recon).
    alternated = False
    if alternation:
        deflated = np.ascontiguousarray(x - t2_recon)
        t1 = gamfit.sparse_dictionary_fit(
            deflated, K=k1, active=t1_active, max_epochs=max(1, t1_max_epochs // 3)
        )
        t1_recon = t1.fitted
        residual = x - t1_recon
        t2, idx, weight = _fit_curved(residual)
        t2_recon = np.asarray(t2.reconstruct(residual), dtype=np.float32)
        alternated = True

    combined_recon = t1_recon + t2_recon
    return TieredComposition(
        t1=t1,
        t2=t2,
        t1_recon=t1_recon,
        t2_recon=t2_recon,
        combined_recon=combined_recon,
        t1_ev=explained_variance(x, t1_recon),
        combined_ev=explained_variance(x, combined_recon),
        subsample_rows=int(idx.size),
        importance_weight=weight,
        alternated=alternated,
    )


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
    tiers.add_argument("--k1", type=int, default=16, help="linear dictionary width")
    tiers.add_argument("--k2", type=int, default=8, help="curved SAE width (K<=64)")
    tiers.add_argument("--t1-active", type=int, default=4)
    tiers.add_argument("--t1-max-epochs", type=int, default=30)
    tiers.add_argument("--d-atom", type=int, default=2)
    tiers.add_argument("--atom-topology", default="circle")
    tiers.add_argument("--assignment", default="threshold_gate")
    tiers.add_argument("--t2-n-iter", type=int, default=50)

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


def main(argv: list[str] | None = None) -> TieredComposition:
    args = build_parser().parse_args(argv)
    X = _load_activations(args)
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
        random_state=args.random_state,
    )

    print(f"[compose_tiers] curved fit on {result.subsample_rows} rows "
          f"(importance weight {result.importance_weight:.3f}), "
          f"alternation={'on' if result.alternated else 'off'}")
    print(f"[compose_tiers] T1-only EV   = {result.t1_ev:.4f}")
    print(f"[compose_tiers] combined  EV = {result.combined_ev:.4f} "
          f"(+{result.ev_gain:.4f} from the curved tier)")
    hs = getattr(result.t2, "hybrid_split", None)
    if hs is not None:
        print(f"[compose_tiers] curved-tier hybrid_split keys: {sorted(hs)}")
    return result


if __name__ == "__main__":
    main()
