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
import signal
import sys
import time
from contextlib import contextmanager
from typing import Any

import numpy as np


class _FitTimeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float):
    """External wall-clock guard around a fit cell (STATE.md: every fit cell needs
    a timeout so a co-collapse HANG is RECORDED, not a silent stall). SIGALRM is
    POSIX-only; the fleet runs on Linux/macOS so this is sufficient."""
    def _handler(signum: int, frame: Any) -> None:
        raise _FitTimeout(f"fit exceeded {seconds:.0f}s wall-clock budget")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


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
    """Fit one stagewise arm under an external timeout. Returns the reconstruction
    ``(N, p)`` plus a compact record; on timeout/hang the record carries the
    failure honestly (no silenced stall)."""
    import gamfit

    t0 = time.time()
    try:
        with _time_limit(timeout_s):
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
    except _FitTimeout as exc:
        return {"ok": False, "error": f"TIMEOUT: {exc}", "seconds": time.time() - t0}
    except Exception as exc:  # a genuine fit failure is recorded, not hidden
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - t0}
    return {
        "ok": True,
        "seconds": time.time() - t0,
        "xhat": np.ascontiguousarray(model.fitted, dtype=np.float64),
        "k_final": int(model.k),
        "terminal_joint_reml": float(model.terminal_joint_reml),
    }


def run_ab(
    x: np.ndarray,
    shard: Any,
    *,
    topology: str = "circle",
    assignment: str = "ibp_map",
    max_births: int = 8,
    n_iter: int = 40,
    random_state: int = 0,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Run both arms and score them in BOTH currencies. ``shard`` is a HarvestShard
    / dict / npz-loaded object carrying ``behavioral_fisher`` ``U``."""
    from gamfit.torch.harvest import load_harvest_shard

    if isinstance(shard, str):
        shard = load_harvest_shard(shard)
    U = np.ascontiguousarray(np.asarray(getattr(shard, "U", shard["U"]), dtype=np.float64))
    n, p, s = U.shape
    if x.shape != (n, p):
        raise ValueError(f"acts {x.shape} disagree with shard U {(n, p, s)}")

    arms: dict[str, Any] = {}
    arms["iid"] = _fit_arm(
        x, fisher_factors=None, topology=topology, assignment=assignment,
        max_births=max_births, n_iter=n_iter, random_state=random_state, timeout_s=timeout_s,
    )
    arms["gls"] = _fit_arm(
        x, fisher_factors=shard, topology=topology, assignment=assignment,
        max_births=max_births, n_iter=n_iter, random_state=random_state, timeout_s=timeout_s,
    )

    table: dict[str, Any] = {"n": n, "p": p, "probes_s": s, "arms": {}}
    for name, rec in arms.items():
        if not rec["ok"]:
            table["arms"][name] = {"status": rec["error"], "seconds": rec["seconds"]}
            continue
        xhat = rec["xhat"]
        table["arms"][name] = {
            "status": "ok",
            "seconds": round(rec["seconds"], 1),
            "k_final": rec["k_final"],
            "euclidean_ev": euclidean_ev(x, xhat),
            "loss_recovered_nats": loss_recovered_nats(U, x, xhat),
            "terminal_joint_reml": rec["terminal_joint_reml"],
        }
    # Persist the reconstructions for the TORCH fidelity harness (KL_patched).
    table["_reconstructions"] = {
        name: arms[name]["xhat"] for name in arms if arms[name].get("ok")
    }
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
            f"{a['loss_recovered_nats']:>16.4f} | {a['k_final']:>3} | {a['terminal_joint_reml']:>10.2f}"
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
    ap.add_argument("--assignment", default="ibp_map")
    ap.add_argument("--max-births", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--out", default=None, help="write reconstructions + table json/npz here (prefix)")
    args = ap.parse_args()

    x = np.ascontiguousarray(np.load(args.acts), dtype=np.float64)
    table = run_ab(
        x, args.shard, topology=args.topology, assignment=args.assignment,
        max_births=args.max_births, n_iter=args.n_iter, random_state=args.random_state,
        timeout_s=args.timeout_s,
    )
    recs = table.pop("_reconstructions")
    print(_format_table(table))
    if args.out:
        with open(f"{args.out}.table.json", "w") as fh:
            json.dump(table, fh, indent=2, default=float)
        np.savez(f"{args.out}.recon.npz", **recs)
        print(f"\nwrote {args.out}.table.json and {args.out}.recon.npz "
              f"(reconstructions for the TORCH KL_patched harness)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
