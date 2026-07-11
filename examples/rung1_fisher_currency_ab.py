"""Rung-1 (B4) Fisher-currency A/B: iid reconstruction vs Fisher-GLS in nats.

The thesis. Plain MSE prices a reconstruction error ``e = x - x̂`` by its
Euclidean size; the model reads the activation only through the rest of the
network, so the behavioural cost of ``e`` is the KL between the clean and
corrupted next-token distributions, ``KL ≈ ½ eᵀ G(x) e`` with ``G = JᵀFJ`` the
network-Jacobian pullback of the output Fisher (units: nats). This script runs
ONE stagewise (SAC) fit twice on the SAME activations:

  * arm ``iid``  — Euclidean loss ``½‖e‖²`` (``fisher_factors=None``);
  * arm ``gls``  — behavioural-Fisher loss ``½ eᵀ G_n e`` (nats), by installing
    the harvest-emitted ``behavioral_fisher`` shard as the reconstruction
    likelihood weight on the seed AND every born atom.

Only the reconstructor differs; the acts, the tier-0 normalisation, and the
harvested ``G`` used for SCORING are shared. The A/B is judged on the
BEHAVIOURAL currency, never on EV:

  * ``loss_recovered`` (nats)  — fraction of behavioural Fisher mass explained,
    ``1 - Σ½eᵀG_n e / Σ½xᵀG_n x``. This is the GLS analogue of EV, in nats. It
    is computed here directly from the harvest factors ``U`` (``G_n = U_nU_nᵀ``,
    so ``½eᵀG_n e = ½‖U_nᵀe‖²``) as a self-contained surrogate;
  * ``KL_patched``  — the gold-standard realised KL from patching ``x̂`` back
    into the model and running the rest. Owned by the TORCH fidelity harness;
    this script hands it the two ``(N, p)`` reconstructions.

The expected, load-bearing result is an INVERSION: the GLS arm may show LOWER
Euclidean EV yet HIGHER ``loss_recovered`` / lower ``KL_patched`` — because it
spends capacity where ``G`` is large (behaviourally read directions) and lets
Euclidean error in ``G``'s null space go free. That inversion IS the argument
that EV is the wrong currency.

Run (self-contained scoring, no model needed):
    python examples/rung1_fisher_currency_ab.py --acts X.npy --shard probe.npz

``--shard`` is a ``save_harvest_shard`` npz (or a directory) carrying the
``behavioral_fisher`` ``U (n, p, s)``. If omitted and ``--model`` is given, the
probes are harvested here (piggybacking a loaded model).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from typing import Any

import numpy as np


def _behavioral_mass(U: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Per-row ``½ eᵀ G_n e = ½‖U_nᵀ e‖²`` given probe factors ``U (n, p, s)`` and
    residual (or signal) rows ``e (n, p)``. Vectorised over rows."""
    # w[n, k] = Σ_i U[n, i, k] e[n, i]  →  (n, s)
    w = np.einsum("npk,np->nk", U, e)
    return 0.5 * np.sum(w * w, axis=1)


def loss_recovered_nats(U: np.ndarray, x: np.ndarray, xhat: np.ndarray) -> float:
    """Behavioural fraction-of-mass explained (nats currency), the GLS analogue of
    EV: ``1 - Σ ½eᵀG_n e / Σ ½xᵀG_n x`` where the SIGNAL mass is measured against
    the tier-0 baseline (mean-ablated), the same baseline the Euclidean EV uses.
    Scored with the SHARED harvested ``G`` regardless of which metric each arm was
    trained under — an honest, arm-independent judge."""
    x0 = x - x.mean(axis=0, keepdims=True)
    resid_mass = float(np.sum(_behavioral_mass(U, x - xhat)))
    signal_mass = float(np.sum(_behavioral_mass(U, x0)))
    if signal_mass <= 0.0:
        return float("nan")
    return 1.0 - resid_mass / signal_mass


def euclidean_ev(x: np.ndarray, xhat: np.ndarray) -> float:
    """The OLD currency, reported only to expose the inversion vs loss_recovered."""
    x0 = x - x.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum(x0 * x0))
    ss_res = float(np.sum((x - xhat) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit_worker(
    x: np.ndarray,
    fisher_factors: Any,
    topology: str,
    assignment: str,
    max_births: int,
    n_iter: int,
    random_state: int,
    out_q: "mp.Queue[Any]",
) -> None:
    """Child-process body: one stagewise fit, result pushed to ``out_q``. Runs in
    its own process so a native Rust HANG (STATE.md BLOCKER-1: K≥2 stagewise can
    spin) is hard-killable by the parent — a Python SIGALRM cannot interrupt a
    long FFI call that never yields to the interpreter."""
    try:
        import gamfit

        model = gamfit.sae_manifold_fit_stagewise(
            np.ascontiguousarray(x, dtype=np.float64),
            d_atom=1,
            atom_topology=topology,
            assignment=assignment,
            fisher_factors=fisher_factors,  # None ⇒ iid; shard ⇒ GLS (nats)
            max_births=max_births,
            max_backfit_sweeps=2,
            n_iter=n_iter,
            random_state=random_state,
        )
        out_q.put({
            "ok": True,
            "xhat": np.ascontiguousarray(model.fitted, dtype=np.float64),
            "k_final": int(model.k),
            "terminal_joint_penalized_laml": float(model.terminal_joint_penalized_laml),
        })
    except Exception as exc:  # a genuine fit failure is recorded, not hidden
        out_q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _fit_arm(
    x: np.ndarray,
    *,
    fisher_factors: Any,
    topology: str,
    assignment: str,
    max_births: int,
    n_iter: int,
    random_state: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Fit one stagewise arm in a HARD-KILLABLE child process. On timeout the
    child is terminated and the record carries the failure honestly (no silenced
    stall). 'spawn' start method avoids fork-after-BLAS-threads hazards."""
    ctx = mp.get_context("spawn")
    out_q: "mp.Queue[Any]" = ctx.Queue()
    proc = ctx.Process(
        target=_fit_worker,
        args=(x, fisher_factors, topology, assignment, max_births, n_iter, random_state, out_q),
    )
    t0 = time.time()
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(10.0)
        if proc.is_alive():
            proc.kill()
        return {"ok": False, "error": f"TIMEOUT: fit exceeded {timeout_s:.0f}s (hard-killed)",
                "seconds": time.time() - t0}
    try:
        rec = out_q.get_nowait()
    except Exception:
        return {"ok": False, "error": f"child exited (code {proc.exitcode}) with no result",
                "seconds": time.time() - t0}
    rec["seconds"] = time.time() - t0
    return rec


def _resolve_fit_space(
    fit_space: str, p: int, tier0_mean: Any, tier0_scale: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, scale_eff)`` (each length-p) for the requested fit space.

    * ``"raw"``         — mean=0, scale=1: fit the activations as-is.
    * ``"mean_center"`` — subtract the tier0 mean, scale=1. The residual ``e`` is
      mean-invariant, so the raw output-Fisher factors ``U`` stay valid unchanged.
    * ``"zscore"``      — full tier0 z-score. The residual now lives in normalized
      coords, so the GLS metric must be rescaled ``U'[i,k] = scale[i]·U[i,k]``
      (since ``∂logits/∂x_norm = ∂logits/∂x_raw · scale``); the caller does that.
    """
    mean = np.zeros(p) if tier0_mean is None else np.asarray(tier0_mean, float).reshape(-1)
    scale = np.ones(p) if tier0_scale is None else np.asarray(tier0_scale, float).reshape(-1)
    if fit_space == "raw":
        return np.zeros(p), np.ones(p)
    if fit_space == "mean_center":
        return mean, np.ones(p)
    if fit_space == "zscore":
        if not np.all(scale > 0):
            raise ValueError("zscore fit_space needs a strictly-positive tier0_scale")
        return mean, scale
    raise ValueError(f"fit_space must be raw/mean_center/zscore; got {fit_space!r}")


def run_ab(
    x: np.ndarray,
    shard: Any,
    *,
    topology: str = "circle",
    assignment: str = "ordered_beta_bernoulli",
    max_births: int = 8,
    n_iter: int = 40,
    random_state: int = 0,
    timeout_s: float = 900.0,
    fit_space: str = "raw",
    tier0_mean: Any = None,
    tier0_scale: Any = None,
) -> dict[str, Any]:
    """Run both arms and score them in BOTH currencies. ``shard`` is a HarvestShard
    / dict / npz-loaded object carrying ``behavioral_fisher`` ``U``.

    ``fit_space`` selects the coordinate the arms are FIT in (``raw`` /
    ``mean_center`` / ``zscore``); the metric ``U`` is rescaled to that space so
    the GLS loss stays the true output-Fisher, and reconstructions are inverted
    back to RAW space for the TORCH patch-in. Scoring (euclidean_ev,
    loss_recovered_nats) is always done in RAW space with the RAW ``U`` so the
    two arms are compared in the model's own currency regardless of fit space.
    """
    from gamfit.torch.harvest import load_harvest_shard

    if isinstance(shard, str):
        shard = load_harvest_shard(shard)
    U = np.ascontiguousarray(np.asarray(getattr(shard, "U", shard["U"]), dtype=np.float64))
    n, p, s = U.shape
    if x.shape != (n, p):
        raise ValueError(f"acts {x.shape} disagree with shard U {(n, p, s)}")

    mean, scale_eff = _resolve_fit_space(fit_space, p, tier0_mean, tier0_scale)
    x_fit = np.ascontiguousarray((x - mean[None, :]) / scale_eff[None, :])
    # Rescale U into the fit space so the GLS metric is the true output-Fisher there.
    u_fit = np.ascontiguousarray(U * scale_eff[None, :, None])
    gls_shard = {"U": u_fit, "provenance": "behavioral_fisher"}

    def _to_raw(xhat_fit: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(xhat_fit * scale_eff[None, :] + mean[None, :])

    arms: dict[str, Any] = {}
    arms["iid"] = _fit_arm(
        x_fit, fisher_factors=None, topology=topology, assignment=assignment,
        max_births=max_births, n_iter=n_iter, random_state=random_state, timeout_s=timeout_s,
    )
    arms["gls"] = _fit_arm(
        x_fit, fisher_factors=gls_shard, topology=topology, assignment=assignment,
        max_births=max_births, n_iter=n_iter, random_state=random_state, timeout_s=timeout_s,
    )

    table: dict[str, Any] = {"n": n, "p": p, "probes_s": s, "fit_space": fit_space, "arms": {}}
    reconstructions: dict[str, np.ndarray] = {}
    for name, rec in arms.items():
        if not rec["ok"]:
            table["arms"][name] = {"status": rec["error"], "seconds": rec["seconds"]}
            continue
        xhat = _to_raw(rec["xhat"])  # back to RAW space for scoring + patch-in
        reconstructions[name] = xhat
        table["arms"][name] = {
            "status": "ok",
            "seconds": round(rec["seconds"], 1),
            "k_final": rec["k_final"],
            "euclidean_ev": euclidean_ev(x, xhat),
            "loss_recovered_nats": loss_recovered_nats(U, x, xhat),
            "terminal_joint_penalized_laml": rec["terminal_joint_penalized_laml"],
        }
    # RAW-space reconstructions + the raw ablate mean for the TORCH fidelity harness
    # (CallableReconstructor(lambda a: xhat_raw, ablate_mean=tier0_mean)).
    table["_reconstructions"] = reconstructions
    table["_ablate_mean_raw"] = (
        np.asarray(tier0_mean, float).reshape(-1) if tier0_mean is not None
        else x.mean(axis=0)
    )
    return table


def _format_table(table: dict[str, Any]) -> str:
    lines = [
        f"Rung-1 Fisher-currency A/B  (n={table['n']}, p={table['p']}, s={table['probes_s']} probes)",
        "",
        f"{'arm':>5} | {'status':>8} | {'euclid_EV':>10} | {'loss_recov_nats':>16} | {'k':>3} | {'reml':>10}",
        "-" * 74,
    ]
    for name, a in table["arms"].items():
        if a["status"] != "ok":
            lines.append(f"{name:>5} | {a['status']:>8}")
            continue
        lines.append(
            f"{name:>5} | {'ok':>8} | {a['euclidean_ev']:>10.4f} | "
            f"{a['loss_recovered_nats']:>16.4f} | {a['k_final']:>3} | {a['terminal_joint_penalized_laml']:>10.2f}"
        )
    arms = table["arms"]
    if arms.get("iid", {}).get("status") == "ok" and arms.get("gls", {}).get("status") == "ok":
        d_ev = arms["gls"]["euclidean_ev"] - arms["iid"]["euclidean_ev"]
        d_nats = arms["gls"]["loss_recovered_nats"] - arms["iid"]["loss_recovered_nats"]
        lines += [
            "",
            f"Δ(gls - iid):  euclidean_EV = {d_ev:+.4f}   loss_recovered_nats = {d_nats:+.4f}",
            "THESIS CONFIRMED" if (d_nats > 0.0 and d_ev <= d_nats) else "inconclusive on this slice",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True, help="(N, p) tier-0-normalised clean acts .npy")
    ap.add_argument("--shard", required=True, help="behavioral_fisher shard (npz / dir)")
    ap.add_argument("--topology", default="circle")
    ap.add_argument("--assignment", default="ordered_beta_bernoulli")
    ap.add_argument("--max-births", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--fit-space", default="raw", choices=("raw", "mean_center", "zscore"),
                    help="coordinate the arms are fit in; U is rescaled to match")
    ap.add_argument("--tier0", default=None,
                    help="tier-0 npz with 'mean' (+ 'scale' for zscore) for the fit space + ablate")
    ap.add_argument("--out", default=None, help="write reconstructions + table json/npz here (prefix)")
    args = ap.parse_args()

    tier0_mean = tier0_scale = None
    if args.tier0:
        t0 = np.load(args.tier0)
        tier0_mean = t0["mean"] if "mean" in t0 else None
        tier0_scale = t0["scale"] if "scale" in t0 else None

    x = np.ascontiguousarray(np.load(args.acts), dtype=np.float64)
    table = run_ab(
        x, args.shard, topology=args.topology, assignment=args.assignment,
        max_births=args.max_births, n_iter=args.n_iter, random_state=args.random_state,
        timeout_s=args.timeout_s, fit_space=args.fit_space,
        tier0_mean=tier0_mean, tier0_scale=tier0_scale,
    )
    recs = table.pop("_reconstructions")
    ablate = table.pop("_ablate_mean_raw")
    print(_format_table(table))
    if args.out:
        with open(f"{args.out}.table.json", "w") as fh:
            json.dump(table, fh, indent=2, default=float)
        np.savez(f"{args.out}.recon.npz", ablate_mean_raw=ablate, **recs)
        print(f"\nwrote {args.out}.table.json and {args.out}.recon.npz "
              f"(RAW-space reconstructions + ablate_mean_raw for the TORCH KL_patched harness)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
