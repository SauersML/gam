"""VPD's standard eval table for the paper decomposition (s-55ea3f9b) on held-out Pile val rows.

  table   N_BATCHES   per 128x512 batch: CI L0 (per site / layer / total), CE/KL/top-1 under
                      ci / unmasked / stoch / random / rounded / zero masks, hidden-acts MSE and
                      attention-pattern KL under ci and stoch masks; plus alive counts over all rows
  ppgd    ROWS STEPS  PPGD-scope adversary (per-position sources, VPD's persistent-PGD Adam)
  pgd     ROWS OFFSET STEPS STEP_SIZE DELTA(0|1)   shared-across-batch PGD (standard: 20 @ 0.1;
                      harsh: 500 @ 0.05)
  pareto  ROWS OFFSET TAUS   S2: gates thresholded at tau (g -> g * [g > tau]); per tau the L0
                      (count g > tau), CI-masked / rounded KL and PGD-20 @ 0.1 shared KL (delta
                      excluded, matching the program our method is scored on)

usage: vpd_table.py MODE ARGS... OUT.json
"""

import json
import resource
import sys
import time

import torch

from vpd_eval import ci_usage, full_battery, gates_and_l0, pgd_recon, ppgd_recon
from vpd_model import load_target, load_vpd, val_tokens

mode, args, out_path = sys.argv[1], sys.argv[2:-1], sys.argv[-1]
t0 = time.time()
target = load_target("mps")
vpd = load_vpd(target, "mps")


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def log(msg: str):
    print(f"[{time.time() - t0:7.0f}s rss {rss_gb():.1f}G] {msg}", flush=True)


results: dict = {"mode": mode, "args": args}
if mode == "table":
    n_batches = int(args[0])
    results["batches"] = []
    usage = None
    for bi in range(n_batches):
        ids = val_tokens(128, offset=bi * 128).to("mps")
        sg, l0 = gates_and_l0(vpd, ids)
        r = {"batch": bi, "l0_total": sum(l0.values()), "l0": l0,
             **{f"l0_layer_{k}": sum(v for n, v in l0.items() if n.startswith(f"h.{k}.")) for k in range(4)}}
        r.update(full_battery(vpd, ids, sg, seed=bi))
        u = ci_usage(vpd, ids)
        usage = u if usage is None else {n: (usage[n][0] + u[n][0], usage[n][1] + u[n][1]) for n in u}
        log(f"batch {bi}: L0 {r['l0_total']:.1f} " + " ".join(
            f"{k}={r[k]:.4f}" for k in r if k.startswith(("kl_", "top1_")) or k.endswith("ReconLoss")))
        results["batches"].append(r)
        del sg
        torch.mps.empty_cache()
        n_tok = (bi + 1) * 128 * 512
        results["n_tokens_usage"] = n_tok
        results["alive_meanci_gt_1e-6"] = {n: int((s / n_tok > 1e-6).sum()) for n, (s, c) in usage.items()}
        results["alive_fired_once"] = {n: int((c > 0).sum()) for n, (s, c) in usage.items()}
        results["alive_total_meanci_gt_1e-6"] = sum(results["alive_meanci_gt_1e-6"].values())
        results["alive_total_fired_once"] = sum(results["alive_fired_once"].values())
        torch.save(usage, out_path.replace(".json", "_usage.pt"))
        json.dump(results, open(out_path, "w"), indent=1)
    log(f"alive (mean CI > 1e-6): {results['alive_total_meanci_gt_1e-6']}  fired once: {results['alive_total_fired_once']}")
elif mode == "ppgd":
    rows, steps = int(args[0]), [int(x) for x in args[1].split(",")]
    ids = val_tokens(rows, offset=0).to("mps")
    results["ppgd"] = ppgd_recon(vpd, ids, steps)
    log(f"ppgd {results['ppgd']}")
elif mode == "pgd":
    rows, offset, steps, step_size, delta = int(args[0]), int(args[1]), [int(x) for x in args[2].split(",")], float(args[3]), bool(int(args[4]))
    ids = val_tokens(rows, offset=offset).to("mps")
    sg, l0 = gates_and_l0(vpd, ids)
    results["l0_total"] = sum(l0.values())
    # the adversary only needs the gates, now stored: drop the CI network (0.54B reals on MPS)
    vpd.ci_fn = None
    import gc
    gc.collect()
    torch.mps.empty_cache()
    log(f"gates stored, CI network freed; L0 {results['l0_total']:.1f}")
    def rung(o):
        results["pgd"] = o
        json.dump(results, open(out_path, "w"), indent=1)
        log(f"pgd so far {o}")
    results["pgd"] = pgd_recon(vpd, ids, sg, steps, step_size=step_size, with_delta=delta, seed=offset, on_rung=rung,
                               state_path=out_path.replace(".json", "_state.pt"))
    json.dump(results, open(out_path, "w"), indent=1)
    log(f"pgd {results['pgd']}")
elif mode == "set":
    # the full table on an explicit token set (e.g. the engine export's 256 x 128)
    rows, seq, offset = int(args[0]), int(args[1]), int(args[2])
    ids = val_tokens(rows, seq=seq, offset=offset).to("mps")
    sg, l0 = gates_and_l0(vpd, ids)
    results.update({"rows": rows, "seq": seq, "offset": offset, "l0_total": sum(l0.values()), "l0": l0})
    results.update(full_battery(vpd, ids, sg, seed=offset))
    log(" ".join(f"{k}={results[k]:.4f}" for k in results if k.startswith(("kl_", "top1_"))))
    for d in (False, True):
        key = "pgd_shared_0.1_with_delta" if d else "pgd_shared_0.1_no_delta"
        results[key] = pgd_recon(vpd, ids, sg, [20, 40], with_delta=d, seed=offset)
        log(f"{key}: {results[key]}")
elif mode == "pareto":
    from vpd_eval import ce_kl_battery

    class Thresholded:
        def __init__(self, base, tau):
            self.base, self.tau, self.names, self.C = base, tau, base.names, base.C

        def target_forward(self, x):
            return self.base.target_forward(x)

        def target_and_ci(self, x):
            lg, g = self.base.target_and_ci(x)
            return lg, {n: v * (v > self.tau) for n, v in g.items()}

        def masked(self, x, m, d):
            return self.base.masked(x, m, d)

    rows, offset, taus = int(args[0]), int(args[1]), [float(x) for x in args[2].split(",")]
    ids = val_tokens(rows, offset=offset).to("mps")
    results["points"] = []
    for tau in taus:
        ex = Thresholded(vpd, tau)
        sg, l0 = gates_and_l0(ex, ids)
        r = {"tau": tau, "l0_total": sum(l0.values()), "l0": l0}
        r.update(ce_kl_battery(ex, ids, sg, seed=0, with_delta=False, strategies=("ci_masked", "rounded_masked")))
        r["pgd20_shared_no_delta"] = pgd_recon(ex, ids, sg, [20], with_delta=False, seed=offset)[20]
        results["points"].append(r)
        log(f"tau {tau}: L0 {r['l0_total']:.1f} kl_ci {r['kl_ci_masked']:.4f} kl_rounded {r['kl_rounded_masked']:.4f} pgd20 {r['pgd20_shared_no_delta']:.4f}")
        json.dump(results, open(out_path, "w"), indent=1)
        del sg
        torch.mps.empty_cache()
json.dump(results, open(out_path, "w"), indent=1)
log("done")
