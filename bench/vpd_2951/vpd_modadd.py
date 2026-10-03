"""VPD's own decomposition of the mod-31 addition transformer (#2951), exported as an engine library.

VPD runs through the paper's torch code (~/mpd-data/spd-vpd-paper, `run_param_decomp.optimize`) on
its 1-layer toy recipe (experiments/tms/tms_40-10_config.yaml, resid_mlp1_config.yaml): per-site
layerwise `mlp` causal-importance functions (hidden [16]), leaky-hard sigmoid, the Δ-component,
StochasticRecon + StochasticReconLayerwise + ImportanceMinimality (p = 2), 200 faithfulness warmup
steps, cosine lr 1e-3. The reconstruction loss is VPD's KL on the logits at `=`
(`recon_loss_kl`), the data the full family: all 961 (a, b, =) rows each step. The six decomposed
maps are nn.Linear modules named after the engine's sites (blocks.0.W_Q ... blocks.0.W_out), whose
weights are the export's (d_out × d_in) matrices, so VPD's U (C × d_out) and Vᵀ (C × d_in) are the
engine's u and v as they stand. A variant may add VPD's FaithfulnessLoss (the paper LM run's 1e7).

The engine reads K and V at every position (gam_mpd::masked::sites: W_K and W_V are 96 × 96,
block-diagonal over the three positions; W_Q, W_O, W_in, W_out read only the `=` position once
the program is pruned to its readout). A VPD subcomponent c of W_K therefore becomes three engine
subcomponents (position j, c), numbered j·C + c, each on where VPD's causal importance of c at
position j clears the threshold; the other sites use position 2's importances.

Subcommands:
  train OUT --imp COEFF [--faith COEFF] [--steps N] [--device D]   one VPD run -> OUT/model.pth
  sweep OUT --imps a,b,... --faiths x,y,... [...]                  every (imp, faith) run in parallel,
                                                                    then `export` of each
  export RUN OUT                                                    engine-format library, CSR sets,
                                                                    summary.json, checks

usage: python vpd_modadd.py SUBCOMMAND ...   (cluster: venv-vpd312; outputs under mpd-data/cluster)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

PAPER = Path.home() / "mpd-data/spd-vpd-paper"
MODEL = Path.home() / "mpd-data/engine/p31_s0_generic"
SITES = ["blocks.0.W_K", "blocks.0.W_O", "blocks.0.W_Q", "blocks.0.W_V", "blocks.0.W_in", "blocks.0.W_out"]
# Sites the engine reads at every position (block-diagonal over positions) vs only at `=`.
ALL_POSITIONS = {"blocks.0.W_K", "blocks.0.W_V"}
READOUT = 2


def f64(name, shape):
    return np.fromfile(MODEL / f"{name}.f64", dtype="<f8").reshape(shape)


def weights():
    rec = json.load(open(MODEL / "export.json"))
    return {k: f64(k, v["shape"]) for k, v in rec["files"].items()}, rec


def inputs():
    rec = json.load(open(MODEL / "export.json"))
    return f64("inputs", rec["samples"]["shape"]).astype(np.int64)


def paper_imports():
    sys.path.insert(0, str(PAPER))
    os.environ.setdefault("PARAM_DECOMP_OUT_DIR", str(Path.home() / "mpd-data/cluster/vpd_modadd/pd_out"))
    for name in ("wandb_workspaces", "wandb_workspaces.reports", "wandb_workspaces.reports.v2", "wandb_workspaces.workspaces"):
        sys.modules.setdefault(name, types.ModuleType(name))


def target_model(dtype):
    import torch
    import torch.nn as nn

    W, rec = weights()
    cfg = rec["config"]
    d, dh, dm = cfg["d_model"], cfg["d_head"], cfg["d_mlp"]
    assert cfg["n_layers"] == 1 and cfg["n_heads"] == 1 and cfg["causal"] and cfg["normalization"] is None

    def linear(name, bias=None):
        w = torch.tensor(W[f"blocks.0.{name}"], dtype=dtype)
        m = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, dtype=dtype)
        m.weight.data.copy_(w)
        if bias is not None:
            m.bias.data.copy_(torch.tensor(W[f"blocks.0.{bias}"].reshape(-1), dtype=dtype))
        return m

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.W_Q, self.W_K, self.W_V, self.W_O = linear("W_Q"), linear("W_K"), linear("W_V"), linear("W_O")
            self.W_in, self.W_out = linear("W_in", "b_in"), linear("W_out", "b_out")
            self.register_buffer("causal", torch.tril(torch.ones(3, 3, dtype=torch.bool)))

        def forward(self, x):
            q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            pattern = torch.softmax(scores.masked_fill(~self.causal, float("-inf")), dim=-1)
            x = x + self.W_O(pattern @ v)
            return x + self.W_out(torch.relu(self.W_in(x)))

    class ModAdd(nn.Module):
        """Logits at `=` of the trained one-layer transformer (bench/mpd_modadd_2951.py's forward)."""

        def __init__(self):
            super().__init__()
            self.register_buffer("W_E", torch.tensor(W["W_E"], dtype=dtype))
            self.register_buffer("W_pos", torch.tensor(W["W_pos"], dtype=dtype))
            self.register_buffer("W_U", torch.tensor(W["W_U"], dtype=dtype))
            self.blocks = nn.ModuleList([Block()])

        def forward(self, tokens):
            x = self.W_E[tokens] + self.W_pos
            for b in self.blocks:
                x = b(x)
            return x[:, READOUT] @ self.W_U.T

    assert d == W["W_E"].shape[1] and dm == W["blocks.0.W_in"].shape[0]
    model = ModAdd()
    model.requires_grad_(False)
    return model.eval()


def c_per_site():
    # Twice the matrix's larger side, the toys' overcomplete margin (TMS 40-10: C = 200 for 40
    # features; ResidMLP: C = 2 × d_mlp).
    W, _ = weights()
    return {s: 2 * max(W[s].shape) for s in SITES}


def run_config(args):
    from param_decomp.configs import Config

    losses = [
        {"classname": "ImportanceMinimalityLoss", "coeff": args.imp, "pnorm": 2.0, "beta": 0.0},
        {"classname": "StochasticReconLayerwiseLoss", "coeff": 1.0},
        {"classname": "StochasticReconLoss", "coeff": 1.0},
    ]
    if args.faith > 0:
        losses.append({"classname": "FaithfulnessLoss", "coeff": args.faith})
    lr = {"start_val": args.lr, "fn_type": "cosine", "warmup_pct": 0.0, "final_val_frac": 0.0}
    raw = {
        "wandb_project": None,
        "seed": args.seed,
        "autocast_bf16": False,
        "n_mask_samples": 1,
        "ci_config": {"mode": "layerwise", "fn_type": "mlp", "hidden_dims": [16]},
        "sigmoid_type": "leaky_hard",
        "module_info": [{"module_pattern": s, "C": c} for s, c in c_per_site().items()],
        "identity_module_info": None,
        "use_delta_component": True,
        "loss_metric_configs": losses,
        "batch_size": 961,
        "eval_batch_size": 961,
        "steps": args.steps,
        "components_optimizer": {"lr_schedule": lr},
        "ci_fn_optimizer": {"lr_schedule": lr},
        "faithfulness_warmup_steps": 200,
        "faithfulness_warmup_lr": 0.01,
        "faithfulness_warmup_weight_decay": 0.1,
        "train_log_freq": min(500, args.steps),
        "eval_freq": min(2000, args.steps),
        "n_eval_steps": 1,
        "slow_eval_freq": min(2000, args.steps) * -(-args.steps // min(2000, args.steps)),
        "slow_eval_on_first_step": False,
        "save_freq": None,
        "ci_alive_threshold": 0.1,
        "eval_metric_configs": [{"classname": "CI_L0", "groups": None}],
        "pretrained_model_class": "bench.vpd_2951.vpd_modadd.ModAdd",
        "task_config": {"task_name": "ih"},
    }
    return Config(**raw)


def cmd_train(args):
    paper_imports()
    import torch
    from torch.utils.data import DataLoader

    import param_decomp.run_param_decomp as loop
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_passthrough

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    os.makedirs(out, exist_ok=True)
    config = run_config(args)
    torch.manual_seed(config.seed)
    rows = torch.tensor(inputs())
    gen = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(rows, batch_size=len(rows), shuffle=True, generator=gen)
    target = target_model(torch.float32).to(args.device)
    started = time.time()
    loop.optimize(target_model=target, config=config, device=args.device, train_loader=loader, eval_loader=loader,
                  run_batch=run_batch_passthrough, reconstruction_loss=recon_loss_kl, out_dir=out)
    seconds = time.time() - started
    os.replace(out / f"model_{args.steps}.pth", out / "model.pth")
    json.dump({"imp": args.imp, "faith": args.faith, "steps": args.steps, "lr": args.lr, "seed": args.seed, "seconds": seconds,
               "device": args.device, "config": json.loads(config.model_dump_json())}, open(out / "run.json", "w"), indent=1)
    print(f"{out}: trained {args.steps} steps in {seconds:.0f}s", flush=True)


def numpy_forward(W, tokens, mats):
    """Logits at `=` with the six maps replaced by `mats` (engine orientation, 32-wide K/V)."""
    x = W["W_E"][tokens] + W["W_pos"]
    q, k, v = x[:, READOUT] @ mats["blocks.0.W_Q"].T, x @ mats["blocks.0.W_K"].T, x @ mats["blocks.0.W_V"].T
    s = np.einsum("nd,npd->np", q, k) / math.sqrt(q.shape[1])
    a = np.exp(s - s.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    h = x[:, READOUT] + np.einsum("np,npd->nd", a, v) @ mats["blocks.0.W_O"].T
    h = h + np.maximum(h @ mats["blocks.0.W_in"].T + W["blocks.0.b_in"].reshape(-1), 0) @ mats["blocks.0.W_out"].T + W["blocks.0.b_out"].reshape(-1)
    return h @ W["W_U"].T


def kl_rows(p_logits, q_logits):
    lp = p_logits - np.logaddexp.reduce(p_logits, axis=1, keepdims=True)
    lq = q_logits - np.logaddexp.reduce(q_logits, axis=1, keepdims=True)
    return (np.exp(lp) * (lp - lq)).sum(1)


def cmd_export(args):
    paper_imports()
    import torch

    from param_decomp.configs import Config
    from param_decomp.models.batch_and_loss_fns import run_batch_passthrough
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.models.components import make_mask_infos
    from param_decomp.utils.module_utils import expand_module_patterns

    run, out = Path(args.run), Path(args.out)
    os.makedirs(out, exist_ok=True)
    record = json.load(open(run / "run.json"))
    config = Config(**record["config"])
    target = target_model(torch.float64)
    model = ComponentModel(target_model=target, run_batch=run_batch_passthrough,
                           module_path_info=expand_module_patterns(target, config.all_module_info),
                           ci_config=config.ci_config, sigmoid_type=config.sigmoid_type)
    state = torch.load(str(run / "model.pth"), map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in state.items()})
    model.double().eval()
    W, _ = weights()
    tokens = inputs()
    ids = torch.tensor(tokens)
    with torch.no_grad():
        o = model(ids, cache_type="input")
        ci = {s: c.numpy() for s, c in model.calc_causal_importances(pre_weight_acts=o.cache, sampling=config.sampling).lower_leaky.items()}
        target_logits = o.output.numpy()
        assert np.allclose(target_logits, numpy_forward(W, tokens, {s: W[s] for s in SITES}), atol=1e-9), "torch target != numpy target"

        def masked(masks):
            return model(ids, mask_infos=make_mask_infos({s: torch.tensor(m) for s, m in masks.items()})).numpy()

        kl = {"ci_masked": float(kl_rows(target_logits, masked(ci)).mean())}
        for t in (0.0, config.ci_alive_threshold):
            kl[f"rounded_gt_{t:g}"] = float(kl_rows(target_logits, masked({s: (c > t).astype(np.float64) for s, c in ci.items()})).mean())
        kl["all_on"] = float(kl_rows(target_logits, masked({s: np.ones_like(c) for s, c in ci.items()})).mean())
    # The engine's library: v (C × d_in), u (C × d_out); K and V block-diagonal over positions.
    sites, mats, offsets, total = [], {}, {}, 0
    for s in SITES:
        comp = model.components[s]
        V = comp.V.detach().numpy()  # d_in × C
        U = comp.U.detach().numpy()  # C × d_out
        C, (di, do) = U.shape[0], (V.shape[0], U.shape[1])
        assert W[s].shape == (do, di)
        mats[s] = U.T @ V.T
        if s in ALL_POSITIONS:
            v = np.zeros((3 * C, 3 * di))
            u = np.zeros((3 * C, 3 * do))
            for j in range(3):
                v[j * C:(j + 1) * C, j * di:(j + 1) * di] = V.T
                u[j * C:(j + 1) * C, j * do:(j + 1) * do] = U
            w = np.kron(np.eye(3), W[s])
        else:
            v, u, w = V.T, U, W[s]
        np.ascontiguousarray(v, dtype="<f8").tofile(out / f"{s}.v.f64")
        np.ascontiguousarray(u, dtype="<f8").tofile(out / f"{s}.u.f64")
        rel = float(np.linalg.norm(w - u.T @ v) / np.linalg.norm(w))
        sites.append({"site": s, "C": int(u.shape[0]), "C_vpd": int(C), "d_in": int(v.shape[1]), "d_out": int(u.shape[1]),
                      "relative_frobenius_error": rel})
        offsets[s] = total
        total += u.shape[0]
    with open(out / "sites.txt", "w") as fh:
        for e in sites:
            fh.write(f"{e['site']} {e['C']}\n")
    allon = numpy_forward(W, tokens, mats)
    allon_kl = kl_rows(target_logits, allon)

    def on_sets(t):
        """Per input, the engine subcomponents on (ci > t), ascending."""
        per_row = [[] for _ in range(len(tokens))]
        l0 = {}
        for s in SITES:
            c = ci[s]  # rows × 3 × C
            C = c.shape[2]
            if s in ALL_POSITIONS:
                on = (c > t).reshape(len(tokens), 3 * C)  # (row, j·C + c)
            else:
                on = c[:, READOUT] > t
            l0[s] = float(on.sum(1).mean())
            r, k = np.nonzero(on)
            for a, b in zip(r, k):
                per_row[a].append(offsets[s] + b)
        indptr = np.concatenate([[0], np.cumsum([len(x) for x in per_row])]).astype("<i8")
        indices = np.concatenate([np.array(sorted(x), dtype="<i8") for x in per_row]).astype("<i8")
        return indptr, indices, l0

    thr = config.ci_alive_threshold
    indptr, indices, l0 = on_sets(thr)
    indptr.tofile(out / "indptr.i64")
    indices.tofile(out / "indices.i64")
    indptr0, indices0, l0_0 = on_sets(0.0)
    indptr0.tofile(out / "indptr_ci0.i64")
    indices0.tofile(out / "indices_ci0.i64")
    # The engine-style masked forward on the exported sets (no Δ), as a check of the CSR's numbering.
    def engine_kl(indptr, indices):
        mask = np.zeros((len(tokens), total))
        for r in range(len(tokens)):
            mask[r, indices[indptr[r]:indptr[r + 1]]] = 1
        logits = np.zeros_like(target_logits)
        for r in range(len(tokens)):
            ms = {}
            for e in sites:
                s, o, C = e["site"], offsets[e["site"]], e["C"]
                v = np.fromfile(out / f"{s}.v.f64").reshape(C, e["d_in"])
                u = np.fromfile(out / f"{s}.u.f64").reshape(C, e["d_out"])
                m = (u * mask[r, o:o + C, None]).T @ v
                d = e["d_in"] // 3
                ms[s] = [m[j * d:(j + 1) * d, j * d:(j + 1) * d] for j in range(3)] if s in ALL_POSITIONS else m
            x = W["W_E"][tokens[r]] + W["W_pos"]
            k = np.stack([ms["blocks.0.W_K"][j] @ x[j] for j in range(3)])
            vv = np.stack([ms["blocks.0.W_V"][j] @ x[j] for j in range(3)])
            q = ms["blocks.0.W_Q"] @ x[READOUT]
            sc = k @ q / math.sqrt(q.shape[0])
            a = np.exp(sc - sc.max())
            a /= a.sum()
            h = x[READOUT] + ms["blocks.0.W_O"] @ (a @ vv)
            h = h + ms["blocks.0.W_out"] @ np.maximum(ms["blocks.0.W_in"] @ h + W["blocks.0.b_in"].reshape(-1), 0) + W["blocks.0.b_out"].reshape(-1)
            logits[r] = W["W_U"] @ h
        return float(kl_rows(target_logits, logits).mean()), float((logits.argmax(1) == target_logits.argmax(1)).mean())

    kl_thr, agree_thr = engine_kl(indptr, indices)
    kl_0, agree_0 = engine_kl(indptr0, indices0)
    log = [json.loads(line) for line in open(run / "metrics.jsonl")] if (run / "metrics.jsonl").exists() else []
    summary = {
        "method": "VPD (paper torch code, spd-vpd-paper run_param_decomp.optimize), 1-layer toy recipe",
        "run": str(run), "imp_coeff": record["imp"], "faithfulness_coeff": record["faith"], "steps": record["steps"],
        "training_seconds": record["seconds"], "device": record["device"],
        "inputs": "mpd-data/engine/p31_s0_generic/inputs.f64 row order (961 rows)",
        "sites": sites,
        "threshold": {"value": thr, "rule": "on iff VPD causal importance (lower_leaky) > ci_alive_threshold, the toy configs' L0 threshold; "
                      "indptr_ci0/indices_ci0 hold the CI > 0 sets (VPD's rounded masks, rounding_threshold 0)",
                      "positions": "W_K, W_V: subcomponent j*C_vpd + c is VPD's c at position j; other sites: position 2 (`=`)"},
        "l0_per_input": {"threshold": l0, "threshold_total": float(sum(l0.values())), "ci_gt_0": l0_0, "ci_gt_0_total": float(sum(l0_0.values()))},
        "kl_nats_mean_over_961": {
            "vpd_ci_masked_with_delta_off": kl["ci_masked"], "vpd_rounded_gt_0": kl["rounded_gt_0"], f"vpd_rounded_gt_{thr:g}": kl[f"rounded_gt_{thr:g}"],
            "vpd_all_on_no_delta": kl["all_on"], "engine_format_all_on": float(allon_kl.mean()),
            "engine_format_sets_threshold": kl_thr, "engine_format_sets_ci_gt_0": kl_0,
        },
        "argmax_agreement": {"sets_threshold": agree_thr, "sets_ci_gt_0": agree_0},
        "max_abs_logit_diff_all_on": float(np.abs(allon - target_logits).max()),
        "final_train_log": next((r for r in reversed(log) if any(k.startswith("train/loss") for k in r)), None),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("imp_coeff", "faithfulness_coeff", "l0_per_input", "kl_nats_mean_over_961", "argmax_agreement")}, indent=1))
    print("relative errors:", {e["site"]: round(e["relative_frobenius_error"], 6) for e in sites}, flush=True)


def cmd_sweep(args):
    out = Path(args.out)
    jobs = []
    for imp in args.imps.split(","):
        for faith in args.faiths.split(","):
            name = f"imp{imp}_faith{faith}"
            d = out / name
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, __file__, "train", str(d), "--imp", imp, "--faith", faith, "--steps", str(args.steps),
                   "--device", args.device, "--threads", str(args.threads), "--lr", str(args.lr)]
            jobs.append((name, d, subprocess.Popen(cmd, stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)))
    for name, d, p in jobs:
        rc = p.wait()
        print(f"{name}: train rc={rc}", flush=True)
        if rc == 0:
            r = subprocess.run([sys.executable, __file__, "export", str(d), str(d / "export")], capture_output=True, text=True)
            print(r.stdout[-4000:], r.stderr[-4000:], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("out")
    t.add_argument("--imp", type=float, required=True)
    t.add_argument("--faith", type=float, default=0.0)
    for p in (t, s := sub.add_parser("sweep")):
        p.add_argument("--steps", type=int, default=20000)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=1)
    t.add_argument("--seed", type=int, default=0)
    s.add_argument("out")
    s.add_argument("--imps", required=True)
    s.add_argument("--faiths", default="0")
    e = sub.add_parser("export")
    e.add_argument("run")
    e.add_argument("out")
    args = ap.parse_args()
    {"train": cmd_train, "sweep": cmd_sweep, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
