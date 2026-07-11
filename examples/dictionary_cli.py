"""Three verbs over the tiered dictionary artifact: fit / steer / diff.

WS-J. A thin CLI/API surface (SPEC.md: Python is a thin wrapper — every verb
dispatches into a Rust fitter or a frozen-artifact reader; the only Python logic
is argument routing and the canonical-hash / subspace-diff serialization glue,
which lives in ``tiered_artifact.py``).

    fit     drive the tiered pipeline (T1 → SAC T2 → assemble) → write artifact dir
    steer   dose-calibrated chart move on one atom; predicted_nats before the edit
    diff    compare two artifacts: atom matching, subspace angles, per-tier hash deltas

Import as a library (``fit_artifact`` / ``steer_atom`` /
``diff_artifacts``) or run as ``python examples/dictionary_cli.py <verb> ...``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiered_artifact import (  # noqa: E402
    TieredArtifact,
    load_sac_result,
    load_t0_from_manifest,
    load_tier1_artifact,
    t2_dictionary_hash,
)


# ======================================================================= #
# fit — drive the tiered pipeline and emit a content-hashed artifact.      #
# ======================================================================= #
def fit_artifact(
    X: np.ndarray,
    *,
    out_dir: str,
    k1: int = 16,
    k2: int = 8,
    use_sac: bool = True,
    d_atom: int = 2,
    atom_topology: str = "circle",
    t0: dict[str, Any] | None = None,
    random_state: int = 0,
    **kwargs: Any,
) -> TieredArtifact:
    """Fit T1 (+ SAC T2) on ``X`` and serialize the union artifact.

    ``use_sac=True`` runs the stagewise K=1 composition (``sac_prototype.sac_fit``,
    the SAC_PLAN Part-2 path that avoids the joint co-collapse); ``False`` selects
    the joint ``compose_tiers`` path. Heavy math is entirely in the Rust
    fitters underneath both.
    """
    import gamfit

    x = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    t1 = gamfit.sparse_dictionary_fit(x, K=k1, active=kwargs.get("t1_active", 4),
                                      max_epochs=kwargs.get("t1_max_epochs", 30))
    t1_recon = t1.fitted

    art = TieredArtifact(t0=t0)
    art.t1_decoder = np.ascontiguousarray(t1.decoder, dtype=np.float32)
    if getattr(t1, "indices", None) is not None:
        art.t1_indices = np.asarray(t1.indices)
        art.t1_codes = np.asarray(t1.codes, dtype=np.float32)

    if use_sac:
        from sac_prototype import sac_fit
        sac = sac_fit(x, t1_recon=t1_recon, max_atoms=k2, d_atom=d_atom,
                      atom_topology=atom_topology, random_state=random_state,
                      verbose=kwargs.get("verbose", False))
        _, specs, meta = load_sac_result(sac)
        art.t2_atoms = specs
        art.t2_meta = meta
    else:
        from compose_tiers import compose_tiers
        comp = compose_tiers(x, k1=k1, k2=k2, d_atom=d_atom,
                             atom_topology=atom_topology, random_state=random_state)
        if hasattr(comp.t2, "to_dict"):
            art.t2_manifold_payload = comp.t2.to_dict()
        art.t2_meta = {"t1_ev": comp.t1_ev, "combined_ev": comp.combined_ev,
                       "ev_gain": comp.ev_gain}

    art.provenance = {"fitter": "sac" if use_sac else "compose_tiers",
                      "n": int(x.shape[0]), "p": int(x.shape[1]),
                      "k1": k1, "k2": k2, "random_state": random_state}
    content_hash = art.save(out_dir)
    art.provenance["content_hash"] = content_hash
    return art


# ======================================================================= #
# steer — dose-calibrated chart move (W8 machinery).                       #
# ======================================================================= #
def steer_atom(
    fit: Any,
    atom_k: int,
    t_from: Any,
    t_to: Any,
    *,
    metric_row: int = 0,
    amplitude: float = 1.0,
) -> dict[str, Any]:
    """Predicted output effect of moving atom ``atom_k`` from ``t_from`` to ``t_to``.

    Thin pass-through to ``ManifoldSAE.steer`` (the SAC_PLAN W8 dose machinery): the
    fitted chart carries an output-Fisher metric and path-integrates it to report
    ``predicted_nats`` — how far the model's output distribution moves — *before*
    the edit. The dose-calibration experiment validated this predictor (slope
    ≈ 0.85, unbiased median ratio ≈ 1.1); see REPORT.md control row.
    """
    plan = fit.steer(
        int(atom_k),
        int(metric_row),
        float(amplitude),
        np.atleast_1d(np.asarray(t_from, dtype=float)),
        np.atleast_1d(np.asarray(t_to, dtype=float)),
    )
    return dict(plan)


# ======================================================================= #
# diff — compare two artifacts (W9 harness: match, angles, hash deltas).   #
# ======================================================================= #
def _span(D: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    U, s, _ = np.linalg.svd(np.asarray(D, dtype=float).T, full_matrices=False)
    return U[:, s > tol * (s[0] if s.size else 1.0)]


def _subspace_cos(Q1: np.ndarray, Q2: np.ndarray) -> tuple[float, float]:
    """Mean/min cos of principal angles between two column-spans (W9)."""
    try:
        from scipy.linalg import subspace_angles
        if Q1.size == 0 or Q2.size == 0:
            return 0.0, 0.0
        cs = np.cos(subspace_angles(Q1, Q2))
        return float(np.mean(cs)), float(np.min(cs))
    except Exception:
        return float("nan"), float("nan")


def _latent_match_cos(D1: np.ndarray, D2: np.ndarray) -> float:
    """Hungarian-matched mean cosine between individual atom directions (W9)."""
    try:
        from scipy.optimize import linear_sum_assignment
        A = _unit_rows(D1)
        B = _unit_rows(D2)
        C = np.abs(A @ B.T)
        r, c = linear_sum_assignment(-C)
        return float(np.mean(C[r, c]))
    except Exception:
        return float("nan")


def _unit_rows(D: np.ndarray) -> np.ndarray:
    D = np.atleast_2d(np.asarray(D, dtype=float))
    n = np.linalg.norm(D, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return D / n


def diff_artifacts(left_dir: str, right_dir: str) -> dict[str, Any]:
    """Compare two tiered artifacts. Reports per-tier hash equality plus, for the
    dictionaries, latent-level (Hungarian) and subspace-level (principal-angle)
    agreement — the W9 pattern where byte-inequality with subspace-cos ≈ 1 is the
    'seed-unstable latents, seed-stable subspace' signal (SAC_PLAN stability)."""
    L = TieredArtifact.load(left_dir)
    R = TieredArtifact.load(right_dir)

    out: dict[str, Any] = {
        "left_hash": L.content_hash(),
        "right_hash": R.content_hash(),
        "hash_equal": L.content_hash() == R.content_hash(),
        "tiers": {},
    }

    # T1 decoder agreement.
    if L.t1_decoder is not None and R.t1_decoder is not None:
        d1, d2 = np.asarray(L.t1_decoder), np.asarray(R.t1_decoder)
        out["tiers"]["t1"] = {
            "latent_cos": _latent_match_cos(d1, d2),
            "union_subspace_cos": _subspace_cos(_span(d1), _span(d2))[0],
            "shapes_equal": d1.shape == d2.shape,
        }

    # T2 typed-atom agreement + Rust-mirrored dictionary hash delta.
    ls, rs = L._t2_atom_specs(), R._t2_atom_specs()
    if ls and rs:
        lh = t2_dictionary_hash(ls, (L.t2_meta or {}).get("gauge_certificate", "unspecified"))
        rh = t2_dictionary_hash(rs, (R.t2_meta or {}).get("gauge_certificate", "unspecified"))
        lD = np.vstack([np.atleast_2d(np.asarray(a["frame"], dtype=float)).reshape(1, -1)
                        if np.atleast_2d(np.asarray(a["frame"])).shape[0] == 1
                        else np.atleast_2d(np.asarray(a["frame"], dtype=float)).ravel()[None, :]
                        for a in ls])
        rD = np.vstack([np.atleast_2d(np.asarray(a["frame"], dtype=float)).ravel()[None, :]
                        for a in rs])
        matched = min(lD.shape[1], rD.shape[1])
        latent = _latent_match_cos(lD[:, :matched], rD[:, :matched])
        subspace = _subspace_cos(_span(lD[:, :matched]), _span(rD[:, :matched]))[0]
        out["tiers"]["t2"] = {
            "dictionary_hash_left": lh,
            "dictionary_hash_right": rh,
            "dictionary_hash_equal": lh == rh,
            "n_atoms_left": len(ls),
            "n_atoms_right": len(rs),
            "latent_cos": latent,
            "union_subspace_cos": subspace,
            "subspace_beats_latent": (subspace - latent) if np.isfinite(subspace) and np.isfinite(latent) else None,
        }
    return out


# ======================================================================= #
# CLI plumbing.                                                            #
# ======================================================================= #
def _load_X(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return z[z.files[0]]
    # A shard directory (WS-D) — read the whole harvest (small local runs only).
    from residual_shard_io import load_shards
    return load_shards(path).read_all()


def _cmd_fit(a: argparse.Namespace) -> None:
    t0 = load_t0_from_manifest(a.t0_manifest) if a.t0_manifest else None
    if a.t1_source:
        decoder, baked_t0 = load_tier1_artifact(a.t1_source)
        t0 = t0 or baked_t0
    X = _load_X(a.data)
    art = fit_artifact(X, out_dir=a.out, k1=a.k1, k2=a.k2, use_sac=not a.joint,
                       d_atom=a.d_atom, atom_topology=a.atom_topology, t0=t0,
                       random_state=a.random_state, verbose=a.verbose)
    print(json.dumps({"content_hash": art.content_hash(),
                      "out": a.out, "provenance": art.provenance}, indent=2))


def _cmd_diff(a: argparse.Namespace) -> None:
    print(json.dumps(diff_artifacts(a.left, a.right), indent=2))


def _cmd_inspect(a: argparse.Namespace) -> None:
    art = TieredArtifact.load(a.artifact)
    with open(os.path.join(a.artifact, "artifact.json")) as f:
        manifest = json.load(f)
    print(json.dumps({"content_hash": art.content_hash(),
                      "tiers_present": manifest.get("tiers_present"),
                      "t2_meta": art.t2_meta, "provenance": art.provenance}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="verb", required=True)

    f = sub.add_parser("fit", help="drive the tiered pipeline → artifact dir")
    f.add_argument("--data", required=True, help=".npy/.npz array or a shard dir")
    f.add_argument("--out", required=True, help="artifact output directory")
    f.add_argument("--t0-manifest", default=None, help="WS-D ShardWriter manifest for T0")
    f.add_argument("--t1-source", default=None, help="WS-C dictionary_artifact dir/json")
    f.add_argument("--k1", type=int, default=16)
    f.add_argument("--k2", type=int, default=8)
    f.add_argument("--d-atom", type=int, default=2)
    f.add_argument("--atom-topology", default="circle")
    f.add_argument("--joint", action="store_true", help="use joint compose_tiers instead of SAC")
    f.add_argument("--random-state", type=int, default=0)
    f.add_argument("--verbose", action="store_true")
    f.set_defaults(func=_cmd_fit)

    d = sub.add_parser("diff", help="compare two artifacts")
    d.add_argument("left")
    d.add_argument("right")
    d.set_defaults(func=_cmd_diff)

    i = sub.add_parser("inspect", help="print an artifact's hash + tiers")
    i.add_argument("artifact")
    i.set_defaults(func=_cmd_inspect)

    # Steering needs a live ManifoldSAE in-process, so it remains a library API;
    # the CLI exposes fit/diff/inspect.
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
