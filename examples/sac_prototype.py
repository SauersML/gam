"""Sequential Atom Composition (SAC) — Python prototype over the gamfit FFI.

This is the WS-A day-one prototype from SAC_PLAN Part 2. It replaces the single
joint ``sae_manifold_fit(K=k2)`` call inside ``compose_tiers`` (the exact call
that co-collapses on real activations) with a stagewise construction that only
ever runs the *proven* K=1 manifold fit:

    forward births  ->  backfitting sweeps  ->  (terminal joint assembly, pending)

The thesis (SAC_PLAN Part 1): K=1 curved fits succeed on real data; the
simultaneous cold-start joint fit of K atoms is the wrong algorithm. So build K
from K=1. Nothing here tunes the joint-fit guard stack — the guard stack is
bypassed by construction because every fit is K=1 (it never trips collapse
detection / reseed-all / separation barriers).

Honesty notes:
  * This is explicitly TEMPORARY Python scaffolding (SPEC.md: math lives in
    Rust). It orchestrates FFI calls; it contains no model math beyond residual
    subtraction and an explained-variance ledger.
  * "Seed from the residual, not global PCs" is achieved for free: the K=1 fit
    computes its PCA seed on whatever array it is handed, so passing the running
    residual R IS residual seeding.
  * The whitened / structured likelihood is applied from atom one via
    ``structured_residual_passes`` on every K=1 fit (SAC_PLAN: "the whitened
    likelihood now applies from atom one, not pass one").
  * No wall-clock budgets or deadlines anywhere (SPEC.md).

Phase 3 (terminal joint assembly via a single evaluate-don't-optimize arrow-Schur
pass) needs a ``merge_tiers`` / ``frozen_evaluate`` FFI verb that does not exist
yet; it stays unimplemented until that verb lands. Everything the joint fit uniquely provided
that Phase 3 would recover (joint Laplace evidence, cross-atom covariance) is
orthogonal to the reconstruction/structure claims these experiments test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import gamfit


# --------------------------------------------------------------------------- #
# Explained-variance ledger (against the column-mean baseline of the target X). #
# --------------------------------------------------------------------------- #
def _ev(x: np.ndarray, recon: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


def _rss(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


# --------------------------------------------------------------------------- #
# One K=1 manifold fit on a residual, optionally warm-started.                 #
# --------------------------------------------------------------------------- #
def _fit_k1(
    residual: np.ndarray,
    *,
    d_atom: int,
    atom_topology: str,
    assignment: str,
    structured_residual_passes: int,
    n_iter: int,
    random_state: int,
    isometry_weight: float = 1.0,
    t_init: np.ndarray | None = None,
    a_init: np.ndarray | None = None,
) -> Any:
    """Run the proven single-atom fit and return the ``ManifoldSAE``.

    ``residual`` is the current running residual R; the fit seeds its chart from
    the top PC pair of R (residual seeding). ``t_init``/``a_init`` warm-start a
    backfitting refit from the atom's current state. ``isometry_weight`` selects
    the whitened / isometry-gauged fit (1.0 = the W6 whitened default).
    """
    r = np.ascontiguousarray(np.asarray(residual, dtype=np.float32))
    return gamfit.sae_manifold_fit(
        r,
        K=1,
        d_atom=d_atom,
        atom_topology=atom_topology,
        assignment=assignment,
        isometry_weight=isometry_weight,
        structured_residual_passes=structured_residual_passes,
        promote_from_residual=False,  # K=1: no nursery promotion inside a single atom
        n_iter=n_iter,
        random_state=random_state,
        t_init=t_init,
        a_init=a_init,
    )


@dataclass
class SacAtom:
    """One accepted atom and the state needed to refit / route it."""

    fit: Any                    # ManifoldSAE (K=1)
    recon: np.ndarray           # (N, p) gated reconstruction over the FULL target
    delta_ev: float             # marginal EV this atom added at birth
    topology: str
    hybrid_verdict: str         # "curved" / "linear" from the hybrid_split report
    coords: np.ndarray          # (N, d) recovered on-atom coordinates
    assignments: np.ndarray     # (N,) gate for this atom


@dataclass
class SacResult:
    atoms: list[SacAtom]
    t1_recon: np.ndarray
    t2_recon: np.ndarray        # sum of accepted atom recons
    combined_recon: np.ndarray
    t1_ev: float
    combined_ev: float
    ev_trace: list[float]       # combined EV after each accepted birth
    birth_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.atoms)

    @property
    def ev_gain(self) -> float:
        return self.combined_ev - self.t1_ev


def _hybrid_verdict(fit: Any) -> str:
    """Read the per-atom curved-vs-linear verdict from the FFI hybrid_split."""
    hs = getattr(fit, "hybrid_split", None)
    if not hs:
        return "unknown"
    # hybrid_split reports a per-atom frontier; a linear collapse flips the atom
    # to a straight image. Surface whatever the FFI decided, verbatim.
    for key in ("verdict", "chosen", "type", "kind"):
        if key in hs:
            return str(hs[key])
    return "reported"


def sac_fit(
    X: np.ndarray,
    *,
    t1_recon: np.ndarray | None = None,
    max_atoms: int = 16,
    d_atom: int = 2,
    atom_topology: str = "circle",
    assignment: str = "ordered_beta_bernoulli",
    ev_floor: float = 5e-3,
    structured_residual_passes: int = 2,
    n_iter: int = 50,
    backfit_sweeps: int = 2,
    isometry_weight: float = 1.0,
    stop_on_rejections: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> SacResult:
    """Compose K atoms one at a time on the residual of X (minus an optional T1).

    Phase 1 (forward births): repeatedly fit a K=1 atom on the running residual,
    accept it iff its marginal EV clears ``ev_floor`` (the explicit salience dial
    — evidence alone keeps true-but-trivial wiggles forever at frontier scale),
    subtract it, and stop after two consecutive rejections.

    Phase 2 (backfitting): for each atom, form the leave-one-out residual and
    refit the K=1 atom warm-started from its current coords/gate. At fixed
    smoothing this is exact block-coordinate descent on the joint penalized
    objective, hence monotone in combined EV.
    """
    x = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, p = x.shape
    if t1_recon is None:
        t1 = np.zeros_like(x)
    else:
        t1 = np.ascontiguousarray(np.asarray(t1_recon, dtype=np.float32))
    t1_ev = _ev(x, t1)

    target = x - t1                      # what T2 must explain
    tss = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    residual = target.copy()             # running residual R
    atoms: list[SacAtom] = []
    ev_trace: list[float] = []
    birth_log: list[dict[str, Any]] = []

    # ----- Phase 1: forward births ---------------------------------------- #
    consecutive_rejects = 0
    birth = 0
    while len(atoms) < max_atoms and (
        not stop_on_rejections or consecutive_rejects < 2
    ):
        seed = random_state + 1000 * birth
        fit = _fit_k1(
            residual,
            d_atom=d_atom,
            atom_topology=atom_topology,
            assignment=assignment,
            structured_residual_passes=structured_residual_passes,
            n_iter=n_iter,
            random_state=seed,
            isometry_weight=isometry_weight,
        )
        recon = np.asarray(fit.reconstruct(residual), dtype=np.float64)
        # Marginal EV this atom adds to the T2 target, relative to the T2 TSS.
        rss_before = float(np.sum(residual.astype(np.float64) ** 2))
        rss_after = _rss(residual, recon)
        delta_ev = (rss_before - rss_after) / tss if tss > 0.0 else 0.0

        cleared_floor = delta_ev >= ev_floor
        # In forced-birth mode (stop_on_rejections=False) every K=1 fit is kept so
        # the kill-criterion can read all 8 atoms' health and the EV climb; the
        # floor is still recorded for reporting.
        accepted = cleared_floor or not stop_on_rejections
        log = {
            "birth": birth,
            "delta_ev": delta_ev,
            "accepted": bool(accepted),
            "cleared_floor": bool(cleared_floor),
            "hybrid": _hybrid_verdict(fit),
            "r2": float(getattr(fit, "reconstruction_r2", float("nan"))),
            "recon_finite": bool(np.all(np.isfinite(recon))),
        }
        birth_log.append(log)
        if verbose:
            print(
                f"[SAC] birth {birth}: dEV={delta_ev:+.4f} "
                f"{'ACCEPT' if accepted else 'reject'} "
                f"(hybrid={log['hybrid']}, r2={log['r2']:.3f})"
            )

        if accepted:
            atoms.append(
                SacAtom(
                    fit=fit,
                    recon=recon,
                    delta_ev=delta_ev,
                    topology=atom_topology,
                    hybrid_verdict=log["hybrid"],
                    coords=fit.coords[0].copy(),
                    assignments=np.asarray(fit.atoms[0].assignments, dtype=np.float64),
                )
            )
            residual = (residual.astype(np.float64) - recon).astype(np.float32)
            t2_now = sum((a.recon for a in atoms), np.zeros((n, p)))
            ev_trace.append(_ev(x, t1 + t2_now))
            consecutive_rejects = 0
        else:
            consecutive_rejects += 1
        birth += 1

    # ----- Phase 2: backfitting sweeps (LOO per-atom refits) --------------- #
    for sweep in range(backfit_sweeps):
        if not atoms:
            break
        moved = False
        for k, atom in enumerate(atoms):
            others = sum(
                (a.recon for j, a in enumerate(atoms) if j != k),
                np.zeros((n, p)),
            )
            loo_residual = (target.astype(np.float64) - others).astype(np.float32)
            # Warm-start from the atom's current chart + gate (#357 hooks).
            t_init = atom.coords[None, ...].astype(np.float64)          # (1, N, d)
            a_init = atom.assignments[:, None].astype(np.float64)        # (N, 1)
            refit = _fit_k1(
                loo_residual,
                d_atom=d_atom,
                atom_topology=atom.topology,
                assignment=assignment,
                structured_residual_passes=structured_residual_passes,
                n_iter=n_iter,
                random_state=random_state + 7 * (k + 1),
                isometry_weight=isometry_weight,
                t_init=t_init,
                a_init=a_init,
            )
            new_recon = np.asarray(refit.reconstruct(loo_residual), dtype=np.float64)
            # Accept the refit only if it does not worsen this block's fit
            # (block-coordinate descent must be monotone).
            if _rss(loo_residual, new_recon) <= _rss(loo_residual, atom.recon) + 1e-9:
                atoms[k] = SacAtom(
                    fit=refit,
                    recon=new_recon,
                    delta_ev=atom.delta_ev,
                    topology=atom.topology,
                    hybrid_verdict=_hybrid_verdict(refit),
                    coords=refit.coords[0].copy(),
                    assignments=np.asarray(refit.atoms[0].assignments, dtype=np.float64),
                )
                moved = True
        t2_now = sum((a.recon for a in atoms), np.zeros((n, p)))
        cur_ev = _ev(x, t1 + t2_now)
        ev_trace.append(cur_ev)
        if verbose:
            print(f"[SAC] backfit sweep {sweep}: combined EV={cur_ev:.4f}")
        if not moved:
            break

    t2_recon = sum((a.recon for a in atoms), np.zeros((n, p)))
    combined = t1.astype(np.float64) + t2_recon
    return SacResult(
        atoms=atoms,
        t1_recon=t1,
        t2_recon=t2_recon.astype(np.float32),
        combined_recon=combined.astype(np.float32),
        t1_ev=t1_ev,
        combined_ev=_ev(x, combined),
        ev_trace=ev_trace,
        birth_log=birth_log,
    )
