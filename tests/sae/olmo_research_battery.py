#!/usr/bin/env python3
"""#977 / #1026 — comprehensive REAL-LLM SAE research battery on OLMo activations.

ONE runner that turns a held A100 + the gamfit wheel + the staged OLMo
activations into a full research-matrix sweep, writing every measured result to
`olmo_battery_results.json` and printing a summary table. It is HONEST: a
cluster/euclidean shape verdict is a real finding, not a failure; every cell is
wrapped in try/except so one bad fit never sinks the run.

The matrix (see the module-level run with --print-run-spec for the exact MSI
command):

  1. REAL MODELS / multi-layer — manifold-SAE on OLMo L25 (self/qualia) AND
     L44 (color, under extra/) residual-stream activations.
  2. SAE-VARIANT sweep — K in {1,2,3,4} x atom topology {euclidean, circle,
     torus, sphere} x basis {periodic, linear}; record fit success, R^2 / EV,
     recovered atom_topology, runtime.
  3. RECON-PARITY (#1026) — manifold-SAE vs a linear top-K SAE baseline:
     EV-vs-K frontier, reconstruction MSE, sparsity; plus the exact-vs-amortized
     encoder gap (predict() reconstruction vs the fitted reconstruction).
  4. ADJUDICATION (#907/#977) — per recovered atom, the Rust cross-class race
     (gamfit.adjudicate_atom_shape: circle vs euclidean vs k-cluster mixture on
     held-out predictive loglik) on the calibrated latent coords, plus a
     superposition/binding read by prompt kind/role (#975).
  5. HILLCLIMB — a small coordinate search over (tau, n_harmonics,
     intrinsic_rank) maximizing held-out EV; report the best config + the climb
     trace.

USAGE:
  python tests/sae/olmo_research_battery.py --data olmo_data/instruct --out olmo_battery_results.json
  python tests/sae/olmo_research_battery.py --print-run-spec      # MSI srun recipe, no data
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _mse(x: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.mean((x - fitted) ** 2))


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def load_layer(data: Path, layer: int) -> np.ndarray:
    """Return the centered (n, 5120) activation slice for `layer`.
    Accepts either activations.npy [635,64,5120] or a pre-sliced [635,5120]."""
    act_path = data / "activations.npy"
    acts = np.load(act_path)
    if acts.ndim == 3:
        z = acts[:, layer, :].astype(np.float64)
    elif acts.ndim == 2:
        z = acts.astype(np.float64)
    else:
        raise ValueError(f"unexpected activations shape {acts.shape}")
    return z - z.mean(axis=0, keepdims=True)


def load_color_layer(data: Path) -> np.ndarray | None:
    """L44 color harvest lives under extra/ in the bank (alltok_L44 or a
    color activations file). Best-effort; returns None if not present."""
    for cand in (data / "extra" / "activations.npy", data / "extra" / "color_activations.npy"):
        if cand.exists():
            acts = np.load(cand)
            z = acts[:, 44, :] if acts.ndim == 3 else acts
            z = z.astype(np.float64)
            return z - z.mean(axis=0, keepdims=True)
    return None


def load_prompts(data: Path, n: int) -> tuple[list[str], list[str]]:
    """Return (kinds, roles) aligned to the n activation rows, or ([],[])."""
    p = data / "prompts.jsonl"
    if not p.exists():
        return [], []
    kinds, roles = [], []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {}
            kinds.append(str(obj.get("kind", "?")))
            roles.append(str(obj.get("role", "?")))
    if len(kinds) != n:
        return [], []
    return kinds, roles


# ---------------------------------------------------------------------------
# Linear top-K SAE baseline (recon-parity #1026). Self-contained: a thin
# dictionary-learning baseline (PCA dictionary + per-row top-K coefficients).
# No sklearn dependency required; it is the honest linear-shattering competitor.
# ---------------------------------------------------------------------------

def linear_topk_sae(z: np.ndarray, n_atoms: int, k_active: int) -> dict:
    """Fit a linear dictionary by SVD and reconstruct each row from its top-`k_active`
    dictionary coefficients. Reports EV, MSE, and the realized sparsity."""
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    d = vt[:n_atoms]                       # (n_atoms, p) dictionary atoms (unit rows)
    coeff = z @ d.T                        # (n, n_atoms)
    # Keep the top-k_active coefficients per row (hard sparsity).
    if k_active < n_atoms:
        thresh_idx = np.argsort(-np.abs(coeff), axis=1)[:, k_active:]
        for i in range(coeff.shape[0]):
            coeff[i, thresh_idx[i]] = 0.0
    recon = coeff @ d
    return {
        "n_atoms": n_atoms,
        "k_active": k_active,
        "ev": _r2(z, recon),
        "mse": _mse(z, recon),
        "sparsity": float(np.mean(np.abs(coeff) > 1e-9)),
    }


# ---------------------------------------------------------------------------
# Manifold-SAE fit wrapper (robust).
# ---------------------------------------------------------------------------

def manifold_fit(gamfit, z, *, K, atom_topology, atom_basis, d_atom, assignment,
                 n_iter, lr, tau, n_harmonics, intrinsic_rank, seed):
    kwargs = dict(
        X=z, K=K, d_atom=d_atom, assignment=assignment,
        n_iter=n_iter, learning_rate=lr, random_state=seed,
    )
    # atom_basis and atom_topology are alternative spellings the engine accepts;
    # prefer atom_topology (the dim-matrix path) and pass basis when it differs.
    kwargs["atom_topology"] = atom_topology
    if atom_basis is not None:
        kwargs["atom_basis"] = atom_basis
    if tau is not None:
        kwargs["tau"] = tau
    if n_harmonics is not None:
        kwargs["_n_harmonics"] = [n_harmonics] * K
    if intrinsic_rank is not None:
        kwargs["intrinsic_rank"] = intrinsic_rank
    t0 = time.time()
    fit = gamfit.sae_manifold_fit(**kwargs)
    dt = time.time() - t0
    return fit, dt


def adjudicate(gamfit, coords, seed):
    """Cross-class shape race through the Rust FFI (single evidence impl)."""
    if hasattr(gamfit, "adjudicate_atom_shape") and coords.shape[1] == 2:
        return gamfit.adjudicate_atom_shape(np.ascontiguousarray(coords), folds=5, seed=seed)
    return {"winner": "n/a", "note": "adjudicate_atom_shape unavailable or d_atom!=2"}


def binding_by_attribute(coords, labels):
    """#975 read: mean per-group angular concentration on the atlas. R->1 means
    the attribute binds the coordinate; R->0 means unbound / superposed."""
    if coords.shape[1] != 2 or not labels:
        return None
    ang = np.arctan2(coords[:, 1], coords[:, 0])
    rs = []
    for u in sorted(set(labels)):
        m = np.array([x == u for x in labels])
        if m.sum() >= 3:
            rs.append(math.hypot(np.cos(ang[m]).mean(), np.sin(ang[m]).mean()))
    return float(np.mean(rs)) if rs else None


# ---------------------------------------------------------------------------
# The battery.
# ---------------------------------------------------------------------------

def run_battery(data: Path, out_path: Path, seed: int, n_iter: int) -> dict:
    import gamfit

    results: dict = {
        "meta": {
            "data": str(data),
            "seed": seed,
            "n_iter": n_iter,
            "gamfit_has_adjudicator": hasattr(gamfit, "adjudicate_atom_shape"),
        },
        "layers": {},
        "variant_sweep": [],
        "recon_parity": [],
        "encoder_gap": [],
        "adjudication": [],
        "hillclimb": {},
    }

    # --- experiment 1: real models, multi-layer load ------------------------
    layers = {}
    z25 = load_layer(data, 25)
    layers["L25_selfqualia"] = z25
    results["layers"]["L25_selfqualia"] = {"n": z25.shape[0], "p": z25.shape[1]}
    z44 = load_color_layer(data)
    if z44 is None:
        try:
            z44 = load_layer(data, 44)
        except Exception:  # noqa: BLE001
            z44 = None
    if z44 is not None:
        layers["L44_color"] = z44
        results["layers"]["L44_color"] = {"n": z44.shape[0], "p": z44.shape[1]}
    kinds, roles = load_prompts(data, z25.shape[0])
    results["meta"]["has_prompt_labels"] = bool(kinds)

    # --- experiment 2: SAE-variant sweep ------------------------------------
    Ks = [1, 2, 3, 4]
    topologies = ["euclidean", "circle", "torus", "sphere"]
    bases = ["periodic", "linear"]
    for layer_name, z in layers.items():
        for K in Ks:
            for topo in topologies:
                for basis in bases:
                    d_atom = 1 if topo in ("circle", "euclidean") and basis == "linear" else 2
                    cell = {"layer": layer_name, "K": K, "topology": topo, "basis": basis, "d_atom": d_atom}
                    try:
                        fit, dt = manifold_fit(
                            gamfit, z, K=K, atom_topology=topo, atom_basis=basis,
                            d_atom=d_atom, assignment="ibp_map", n_iter=n_iter, lr=0.04,
                            tau=None, n_harmonics=None, intrinsic_rank=None, seed=seed,
                        )
                        r2 = float(getattr(fit, "reconstruction_r2", _r2(z, np.asarray(fit.fitted))))
                        cell.update({
                            "ok": True, "r2": r2, "runtime_s": dt,
                            "recovered_topology": str(getattr(fit, "atom_topology", "?")),
                        })
                    except Exception as exc:  # noqa: BLE001
                        cell.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    results["variant_sweep"].append(cell)
                    print(f"  sweep {layer_name} K={K} {topo}/{basis}: "
                          + ("R2=%.4f t=%.1fs" % (cell.get('r2', float('nan')), cell.get('runtime_s', 0.0))
                             if cell.get('ok') else "FAIL " + cell.get('error', '')))

    # --- experiment 3: recon parity (#1026) manifold vs linear -------------
    z = layers["L25_selfqualia"]
    for K in Ks:
        # manifold EV at this K (best single periodic circle atom stack)
        man = {"K": K}
        try:
            fit, dt = manifold_fit(
                gamfit, z, K=K, atom_topology="circle", atom_basis="periodic",
                d_atom=2, assignment="ibp_map", n_iter=n_iter, lr=0.04,
                tau=None, n_harmonics=None, intrinsic_rank=None, seed=seed,
            )
            man["manifold_ev"] = float(getattr(fit, "reconstruction_r2", _r2(z, np.asarray(fit.fitted))))
            man["manifold_mse"] = _mse(z, np.asarray(fit.fitted))
            man["manifold_runtime_s"] = dt
        except Exception as exc:  # noqa: BLE001
            man["manifold_error"] = f"{type(exc).__name__}: {exc}"
        # linear top-K baseline at matched atom budget
        try:
            lin = linear_topk_sae(z, n_atoms=max(2 * K, 4), k_active=K)
            man["linear_ev"] = lin["ev"]
            man["linear_mse"] = lin["mse"]
            man["linear_sparsity"] = lin["sparsity"]
        except Exception as exc:  # noqa: BLE001
            man["linear_error"] = f"{type(exc).__name__}: {exc}"
        results["recon_parity"].append(man)
        print(f"  recon-parity K={K}: manifold_ev={man.get('manifold_ev', float('nan')):.4f} "
              f"linear_ev={man.get('linear_ev', float('nan')):.4f}")

    # --- exact vs amortized encoder gap ------------------------------------
    try:
        ntr = z.shape[0] // 2
        fit, _ = manifold_fit(
            gamfit, z[:ntr], K=1, atom_topology="circle", atom_basis="periodic",
            d_atom=2, assignment="ibp_map", n_iter=n_iter, lr=0.04,
            tau=None, n_harmonics=None, intrinsic_rank=None, seed=seed,
        )
        in_ev = float(getattr(fit, "reconstruction_r2", _r2(z[:ntr], np.asarray(fit.fitted))))
        oos_ev = float("nan")
        if hasattr(fit, "predict"):
            oos = np.asarray(fit.predict(z[ntr:]))
            oos_ev = _r2(z[ntr:], oos)
        results["encoder_gap"].append({"in_sample_ev": in_ev, "oos_ev": oos_ev, "gap": in_ev - oos_ev})
        print(f"  encoder gap: in={in_ev:.4f} oos={oos_ev:.4f}")
    except Exception as exc:  # noqa: BLE001
        results["encoder_gap"].append({"error": f"{type(exc).__name__}: {exc}"})

    # --- experiment 4: adjudication (#907/#977) per recovered atom ----------
    for layer_name, z in layers.items():
        entry = {"layer": layer_name}
        try:
            fit, _ = manifold_fit(
                gamfit, z, K=1, atom_topology="circle", atom_basis="periodic",
                d_atom=2, assignment="ibp_map", n_iter=n_iter, lr=0.04,
                tau=None, n_harmonics=None, intrinsic_rank=None, seed=seed,
            )
            coords = np.asarray(fit.coords[0], dtype=float)
            entry["reconstruction_r2"] = float(getattr(fit, "reconstruction_r2", _r2(z, np.asarray(fit.fitted))))
            entry["shape_verdict"] = adjudicate(gamfit, coords, seed + 11)
            if kinds:
                entry["kind_binding_R"] = binding_by_attribute(coords, kinds)
            if roles:
                entry["role_binding_R"] = binding_by_attribute(coords, roles)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results["adjudication"].append(entry)
        v = entry.get("shape_verdict", {})
        print(f"  adjudication {layer_name}: shape={v.get('winner', '?')} "
              f"margin={v.get('circle_margin', float('nan'))}")

    # --- experiment 5: hillclimb over (tau, n_harmonics, intrinsic_rank) ----
    z = layers["L25_selfqualia"]
    ntr = z.shape[0] // 2

    def heldout_ev(tau, nh, rank):
        fit, _ = manifold_fit(
            gamfit, z[:ntr], K=1, atom_topology="circle", atom_basis="periodic",
            d_atom=2, assignment="softmax", n_iter=n_iter, lr=0.04,
            tau=tau, n_harmonics=nh, intrinsic_rank=rank, seed=seed,
        )
        if hasattr(fit, "predict"):
            return _r2(z[ntr:], np.asarray(fit.predict(z[ntr:])))
        return float(getattr(fit, "reconstruction_r2", _r2(z[:ntr], np.asarray(fit.fitted))))

    trace = []
    best = {"config": None, "ev": -math.inf}
    cfg = {"tau": 0.5, "n_harmonics": 2, "intrinsic_rank": 1}
    for axis, grid in (("tau", [0.25, 0.5, 0.7]),
                       ("n_harmonics", [1, 2, 3]),
                       ("intrinsic_rank", [1, 2])):
        for val in grid:
            trial = dict(cfg)
            trial[axis] = val
            try:
                ev = heldout_ev(trial["tau"], trial["n_harmonics"], trial["intrinsic_rank"])
                trace.append({"config": dict(trial), "heldout_ev": ev})
                if ev > best["ev"]:
                    best = {"config": dict(trial), "ev": ev}
            except Exception as exc:  # noqa: BLE001
                trace.append({"config": dict(trial), "error": f"{type(exc).__name__}: {exc}"})
        if best["config"] is not None:
            cfg = dict(best["config"])  # coordinate ascent: keep the best on this axis
    results["hillclimb"] = {"best": best, "trace": trace}
    print(f"  hillclimb best: {best['config']} heldout_ev={best['ev']:.4f}")

    # --- experiment 6: GPU cross-fit multiplex throughput (#1017/#1026) ------
    # Measured cross_fit_speedup on the REAL color-arm variant matrix
    # (K{1..4}×topology{4}×basis{periodic,linear}, n=180, p=5120), bit-parity
    # asserted vs sequential. Quoted over the resident kernel's deterministic
    # frames (real shapes) until the per-cell real-slab fits are wired through.
    if hasattr(gamfit, "sweep_color_arm_throughput"):
        try:
            tp = gamfit.sweep_color_arm_throughput()
            results["gpu_multiplex_throughput"] = tp
            print(f"  gpu multiplex: cells={tp['cells']} speedup={tp['cross_fit_speedup']:.2f}x "
                  f"(mux={tp['multiplexed_fits_per_second']:.1f} fits/s, "
                  f"seq={tp['sequential_fits_per_second']:.1f} fits/s, used_device={tp['used_device']})")
        except Exception as exc:  # noqa: BLE001
            results["gpu_multiplex_throughput"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        results["gpu_multiplex_throughput"] = {"note": "sweep_color_arm_throughput not in this wheel"}

    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out_path}")
    return results


def print_summary(results: dict) -> None:
    print("\n=== SUMMARY ===")
    print("layers:", {k: v for k, v in results["layers"].items()})
    ok = [c for c in results["variant_sweep"] if c.get("ok")]
    print(f"variant sweep: {len(ok)}/{len(results['variant_sweep'])} cells fit ok")
    if ok:
        best = max(ok, key=lambda c: c.get("r2", -1))
        print(f"  best cell: {best['layer']} K={best['K']} {best['topology']}/{best['basis']} R2={best['r2']:.4f}")
    for rp in results["recon_parity"]:
        print(f"  recon K={rp.get('K')}: manifold_ev={rp.get('manifold_ev')} linear_ev={rp.get('linear_ev')}")
    for a in results["adjudication"]:
        v = a.get("shape_verdict", {})
        print(f"  adjudication {a.get('layer')}: {v.get('winner', a.get('error', '?'))} "
              f"(kind_R={a.get('kind_binding_R')}, role_R={a.get('role_binding_R')})")
    print(f"hillclimb best: {results['hillclimb'].get('best')}")


RUN_SPEC = """\
=== #977 OLMo research battery — MSI run spec (overlap on the held A100) ===
# data already staged at olmo_data/instruct; wheel built by the lead's GPU job.
srun --overlap --jobid <ALLOC_JOBID> --gres=gpu:a100:1 -t 60 bash -lc '
  source /projects/standard/hsiehph/sauer354/gam_env.sh
  cd /projects/standard/hsiehph/sauer354/gam
  python tests/sae/olmo_research_battery.py \\
    --data /projects/standard/hsiehph/sauer354/olmo_data/instruct/<rev> \\
    --out olmo_battery_results.json'
# Then send olmo_battery_results.json back; I post the tables on #977 / #1026.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out", type=str, default="olmo_battery_results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--print-run-spec", action="store_true")
    args = ap.parse_args()

    if args.print_run_spec:
        print(RUN_SPEC)
        return 0
    if not args.data:
        print("ERROR: --data <dir> required (or --print-run-spec).", file=sys.stderr)
        return 2
    try:
        import gamfit  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKER: gamfit wheel not importable ({exc}).", file=sys.stderr)
        return 3
    data = Path(args.data)
    if not (data / "activations.npy").exists():
        print(f"BLOCKER: {data}/activations.npy missing.", file=sys.stderr)
        return 4
    try:
        results = run_battery(data, Path(args.out), args.seed, args.n_iter)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"BLOCKER: battery aborted before completion: {exc}", file=sys.stderr)
        return 5
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
