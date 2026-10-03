"""Matched-compute race on VPD-4L (#2951): VPD's own trainer against the masked-subcomponents
engine (crates/gam-mpd/examples/mpd_pieces_masked_2951.rs), both from scratch on the same Pile
val rows, both scored on the 32 frontier passages by the engine's scorer.

VPD runs through the paper's own torch code (~/mpd-data/spd-vpd-paper, the code behind
goodfire/spd/runs/s-55ea3f9b): its `optimize` loop on its migrated final config, with `steps`,
`batch_size`, the causal-importance network and the subcomponent counts scaled to the budget.
Its schedules run on normalized progress, so a shorter run anneals fully on a compressed schedule
(docs/skill.md). Training FLOPs are counted by torch's FlopCounterMode around the whole loop
(faithfulness warmup, target forward, causal importances, persistent-PGD warmups, stochastic and
adversarial reconstructions, backward), so nothing is estimated.

Subcommands:
  export N OUT_DIR                  engine export: the stream's first N rows, then the 32 frontier rows
  train OUT_DIR --steps S ...       VPD training (FLOPs counted), checkpoint OUT_DIR/model.pth
  sets RUN_DIR                      the run's library (U, V per site) and CI>0 sets on val rows
                                    1024..1151 (frontier first), in vpd_sets_export.py's formats
  score RUN_DIR                     the engine's scorer on the run's library and sets (after
                                    `sets`): RUN_DIR/score.json, whose `start` is VPD's own sets
  engine-flops LOG [--train N]      training FLOPs of an engine run, from its log's events

Training data for both methods: one stream of Pile val-00000 rows, val 1056..1151 (the rows the
trainer's own from-scratch race trains on) and then val 2048 onward, disjoint from the 32 frontier
rows (val 1024..1055); each method consumes the prefix its budget reaches.

usage: MPD_MEM_GIB=8 ~/mpd-data/venv/bin/python vpd_matched_compute.py SUBCOMMAND ...
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PAPER = Path.home() / "mpd-data/spd-vpd-paper"
VPD = Path.home() / "mpd-data/vpd"
FRONTIER32 = Path.home() / "mpd-data/engine/vpd4l_frontier32"
FRONTIER_ROW0, SCORED_ROWS = 1024, 128
# The training stream: the coder's held-out rows (val 1056..1151, the trainer's race rows), then
# val 2048 onward.
STREAM = [(1056, 96), (2048, 46590)]


def train_tokens(n: int, seq: int = 512):
    """The first n rows of the training stream."""
    import torch

    parts, left = [], n
    for offset, rows in STREAM:
        take = min(left, rows)
        if take:
            parts.append(val_tokens(take, seq=seq, offset=offset))
        left -= take
    assert left == 0, f"the training stream has fewer than {n} rows"
    return torch.cat(parts)


def stream_rows(n: int) -> str:
    out, left = [], n
    for offset, rows in STREAM:
        take = min(left, rows)
        if take:
            out.append(f"val {offset}..{offset + take}")
        left -= take
    return ", ".join(out)
KINDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "c_fc", "down_proj": "down_proj"}

sys.path.insert(0, str(Path(__file__).parent))
from vpd_model import site_names, val_tokens  # noqa: E402


def engine_name(vpd_name: str) -> str:
    layer, _, kind = vpd_name.split(".")[1:]
    return f"blocks.{layer}.{KINDS[kind]}"


def cmd_export(n: int, out: Path) -> None:
    """The frontier export's tensors with N training rows ahead of its 32 eval rows."""
    os.makedirs(out, exist_ok=True)
    record = json.load(open(FRONTIER32 / "export.json"))
    record["files"] = {k: v for k, v in record["files"].items() if k != "row_ids" and not k.startswith("logits_")}
    for name in record["files"]:
        if name != "tokens" and not (out / f"{name}.f64").exists():
            os.symlink((FRONTIER32 / f"{name}.f64").resolve(), out / f"{name}.f64")
    ids = np.concatenate([train_tokens(n, seq=513).numpy(), val_tokens(32, seq=513, offset=FRONTIER_ROW0).numpy()])
    ids.astype("<f8").tofile(out / "tokens.f64")
    record["files"]["tokens"] = {"shape": list(ids.shape)}
    record["source"]["token_rows"] = f"{stream_rows(n)} (training), then the frontier's 32 eval rows from val {FRONTIER_ROW0}"
    with open(out / "export.json", "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"export {out}: {n} training + 32 frontier rows")


def paper_imports():
    sys.path.insert(0, str(PAPER))
    os.environ.setdefault("PARAM_DECOMP_OUT_DIR", str(VPD / "matched"))
    # wandb report building is imported at module level but never used offline.
    import types

    for name in ("wandb_workspaces", "wandb_workspaces.reports", "wandb_workspaces.reports.v2", "wandb_workspaces.workspaces"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import torch
    import yaml
    from safetensors.torch import load_file

    from param_decomp.configs import Config
    from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig

    def target_model():
        cfg = yaml.safe_load(open(VPD / "t-9d2b8f02/model_config.yaml"))
        cfg["flash_attention"] = True  # F.scaled_dot_product_attention; same function, fused kernel
        model = LlamaSimpleMLP(LlamaSimpleMLPConfig(**cfg))
        missing, unexpected = model.load_state_dict(load_file(str(VPD / "t-9d2b8f02/model_step_99999.safetensors")), strict=False)
        assert set(missing) <= {"lm_head.weight", "wte.weight"} and not unexpected, (missing, unexpected)
        model.requires_grad_(False)
        return model.eval()

    return torch, yaml, Config, target_model


def run_config(args, Config, yaml):
    """The paper run's final config, scaled: steps, batch, CI network, subcomponent counts."""
    raw = yaml.safe_load(open(VPD / "s-55ea3f9b/final_config.yaml"))
    raw["pretrained_model_class"] = "param_decomp.pretrain.models.llama_simple_mlp.LlamaSimpleMLP"
    raw["wandb_project"] = None
    raw["steps"] = args.steps
    raw["batch_size"] = args.batch
    # Training only: one tiny eval forward at the first and last step (the loop always runs one),
    # no metrics, logging at the end.
    raw["eval_metric_configs"] = []
    raw["eval_batch_size"] = 1
    raw["eval_freq"] = args.steps
    raw["slow_eval_freq"] = args.steps
    raw["slow_eval_on_first_step"] = False
    raw["train_log_freq"] = max(1, args.steps // 20)
    raw["faithfulness_warmup_steps"] = args.warmup
    raw["lr_schedule"]["start_val"] = args.lr
    d_model, n_blocks, hidden, heads = args.ci
    t = raw["ci_config"]["simple_transformer_ci_cfg"]
    t["d_model"], t["n_blocks"], t["mlp_hidden_dim"] = d_model, n_blocks, [hidden]
    t["attn_config"]["n_heads"] = heads
    for m in raw["module_info"]:
        m["C"] = max(1, round(m["C"] * args.c_scale))
    return Config(**raw)


def cmd_train(args) -> None:
    torch, yaml, Config, target_model = paper_imports()
    from torch.utils.data import DataLoader
    from torch.utils.flop_counter import FlopCounterMode

    import param_decomp.run_param_decomp as loop
    from param_decomp.models.batch_and_loss_fns import make_run_batch, recon_loss_kl

    out = Path(args.out)
    os.makedirs(out, exist_ok=True)
    config = run_config(args, Config, yaml)
    torch.manual_seed(config.seed)
    rows = args.batch * (args.steps + 2)
    ids = train_tokens(rows)
    train_loader = DataLoader(list(ids), batch_size=args.batch, shuffle=False, collate_fn=torch.stack)
    eval_loader = DataLoader(list(val_tokens(4, offset=FRONTIER_ROW0 + SCORED_ROWS)), batch_size=1, collate_fn=torch.stack)
    target = target_model()
    started = time.time()
    counter = FlopCounterMode(display=False)
    # FLOPs by phase: the faithfulness warmup (the components' initialization) and the loop's
    # evaluation forwards (training only needs neither's output here) are kept apart.
    phases = {"warmup": 0, "eval": 0}

    def tallied(phase, fn):
        def run(*a, **k):
            before = counter.get_total_flops()
            out = fn(*a, **k)
            phases[phase] += counter.get_total_flops() - before
            return out

        return run

    loop.run_faithfulness_warmup = tallied("warmup", loop.run_faithfulness_warmup)
    loop.evaluate = tallied("eval", loop.evaluate)
    with counter:
        loop.optimize(
            target_model=target,
            config=config,
            device=args.device,
            train_loader=train_loader,
            eval_loader=eval_loader,
            run_batch=make_run_batch(config.output_extract),
            reconstruction_loss=recon_loss_kl,
            out_dir=out,
        )
    if args.device == "mps":
        torch.mps.synchronize()
    seconds = time.time() - started
    flops = counter.get_total_flops()
    # The loop runs steps + 1 training passes (the last one only logs) after one probe forward.
    per_step = (flops - phases["warmup"] - phases["eval"]) / (args.steps + 1)
    final = out / f"model_{args.steps}.pth"
    os.replace(final, out / "model.pth")
    record = {
        "steps": args.steps, "batch": args.batch, "ci": args.ci, "c_scale": args.c_scale, "warmup": args.warmup, "lr": args.lr,
        "train_rows": stream_rows(rows), "tokens": rows * 512,
        "flops": flops, "warmup_flops": phases["warmup"], "eval_flops": phases["eval"], "flops_per_step": per_step,
        "training_flops": phases["warmup"] + per_step * args.steps,
        "seconds": seconds, "device": args.device,
        "config": json.loads(config.model_dump_json()),
    }
    json.dump(record, open(out / "run.json", "w"), indent=1)
    print(f"trained {args.steps} steps × {args.batch}: {record['training_flops']:.4e} training FLOPs "
          f"(warmup {phases['warmup']:.3e}, {per_step:.4e} per step), {seconds:.0f}s")


def cmd_sets(run: Path, device: str, check: int) -> None:
    """The run's library and its rounded (CI > 0) sets, in the formats of the reference run's
    vpd4l_library and masks_vpd4l.npz. On the first `check` frontier rows it also reports the
    sets' own L0, KL and greedy agreement (VPD's rounded-masked forward without Δ-components),
    a quick reading ahead of the engine's scorer."""
    torch, yaml, Config, target_model = paper_imports()
    import torch.nn.functional as F

    from param_decomp.models.batch_and_loss_fns import make_run_batch
    from param_decomp.models.component_model import ComponentModel
    from param_decomp.models.components import make_mask_infos
    from param_decomp.utils.module_utils import expand_module_patterns

    record = json.load(open(run / "run.json"))
    config = Config(**record["config"])
    target = target_model()
    model = ComponentModel(
        target_model=target,
        run_batch=make_run_batch(config.output_extract),
        module_path_info=expand_module_patterns(target, config.all_module_info),
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
    )
    model.load_state_dict(torch.load(str(run / "model.pth"), map_location="cpu", weights_only=True))
    model.to(device).eval()
    names = site_names()
    library = run / "library"
    os.makedirs(library, exist_ok=True)
    manifest = {}
    for n in names:
        comp = model.components[n]
        V = comp.V.detach().double().cpu().numpy()  # [d_in, C]
        U = comp.U.detach().double().cpu().numpy()  # [C, d_out]
        W = model.target_weight(n).detach().double().cpu().numpy()  # [d_out, d_in]
        e = engine_name(n)
        np.ascontiguousarray(V.T).astype("<f8").tofile(library / f"{e}.v.f64")
        np.ascontiguousarray(U).astype("<f8").tofile(library / f"{e}.u.f64")
        manifest[e] = {"vpd": n, "pieces": int(V.shape[1]), "d_in": int(V.shape[0]), "d_out": int(U.shape[1]),
                       "delta_relative": float(np.linalg.norm(W - (V @ U).T) / np.linalg.norm(W))}
    json.dump(manifest, open(library / "manifest.json", "w"), indent=1)
    cs = [manifest[engine_name(n)]["pieces"] for n in names]
    offsets = np.concatenate([[0], np.cumsum(cs)])
    ids = val_tokens(SCORED_ROWS, offset=FRONTIER_ROW0)
    indptr, indices, total = [np.zeros(1, dtype=np.int64)], [], 0
    l0, kl, agree = [], [], []
    with torch.no_grad():
        for r in range(SCORED_ROWS):
            out = model(ids[r:r + 1].to(device), cache_type="input")
            ci = model.calc_causal_importances(pre_weight_acts=out.cache, sampling=config.sampling).lower_leaky
            if r < check:
                logits = model(ids[r:r + 1].to(device), mask_infos=make_mask_infos({n: (ci[n] > 0).float() for n in names}))
                p = F.log_softmax(out.output[0].double(), -1)
                q = F.log_softmax(logits[0].double(), -1)
                kl.append(float((p.exp() * (p - q)).sum(-1).mean()))
                agree.append(float((p.argmax(-1) == q.argmax(-1)).double().mean()))
                l0.append(float(sum((ci[n][0] > 0).sum() for n in names)) / ids.shape[1])
            on = torch.cat([(ci[n][0] > 0) for n in names], dim=-1).cpu().numpy()  # [S, total C]
            pos, comp = np.nonzero(on)
            indices.append(comp.astype(np.int64))
            indptr.append(total + np.cumsum(np.bincount(pos, minlength=on.shape[0])))
            total += len(comp)
    np.savez(run / "masks.npz", ids=ids.numpy(), site_names=np.array(names), vpd_offsets=offsets,
             vpd_indptr=np.concatenate(indptr), vpd_indices=np.concatenate(indices))
    quick = {"rows": check, "l0": float(np.mean(l0)), "kl": float(np.mean(kl)), "agree": float(np.mean(agree))} if check else {}
    json.dump({"delta_relative": {k: v["delta_relative"] for k, v in manifest.items()}, "check": quick}, open(run / "sets.json", "w"), indent=1)
    print(f"{run}: library {offsets[-1]} subcomponents, {total / (SCORED_ROWS * ids.shape[1]):.1f} on per position; check {quick}")


SCORER = Path.home() / "mpd-data/engine/bin/mpd_pieces_masked_race"


def cmd_score(run: Path, lease: int) -> None:
    """The scorer the reference VPD point went through (stepA): the frontier export, OBSERVATIONS
    1024, the run's library, nothing trained, its sets on the 32 frontier rows given, its sets on
    the 96 held-out rows counted by the context coder."""
    import subprocess

    subprocess.run([sys.executable, str(Path(__file__).parent / "vpd_sets_export.py"), str(run / "sets"),
                    "--masks", str(run / "masks.npz"), "--library", str(run / "library")], check=True)
    with open(run / "score.log", "w") as log:
        subprocess.run([str(Path.home() / ".local/bin/mem-lease"), str(lease), str(SCORER), str(FRONTIER32), str(run / "score.json"),
                        "1024", f"library:{run / 'library'}", "0", "32", "512", "auto", str(run / "sets")],
                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)
    point = json.load(open(run / "score.json"))["points"][-1]
    print(json.dumps({"vpd_sets": point["start"], "reselected": {k: point[k] for k in ("l0", "kl", "context_bits", "code", "agree")}}, indent=1))


# ---------------------------------------------------------------- engine FLOPs from its log

T, D, VOCAB, LAYERS, HEADS = 512, 768, 50277, 4, 6
SITE_DIMS = {"q": (D, D), "k": (D, D), "v": (D, D), "o": (D, D), "c_fc": (D, 3072), "down_proj": (3072, D)}  # (d_in, d_out)


def engine_costs(pieces_per_site: float) -> dict[str, float]:
    """Matmul FLOPs of the engine's primitives on one 512-token sequence (f64), with every site
    holding `pieces_per_site` times its Fisher-SVD rank of subcomponents."""
    dims = [SITE_DIMS[k] for k in SITE_DIMS] * LAYERS
    rank = [min(i, o) * pieces_per_site for i, o in dims]
    site = sum(2 * T * c * (i + o) for c, (i, o) in zip(rank, dims))
    attn = LAYERS * 4 * T * T * D
    unemb = 2 * T * D * VOCAB
    target_sites = sum(2 * T * i * o for i, o in dims)
    fwd = site + attn + unemb
    vjp = site + 2 * attn + unemb
    written = sum(2 * T * o * o for _, o in dims)
    read = sum(2 * T * i * i for i, _ in dims)
    eigh = sum(11 * (i ** 3 + o ** 3) + 2 * c * (i * i + o * o) for c, (i, o) in zip(rank, dims))
    # The box claim's excess (`gam_mpd::masked::box_excess`): a reverse pass, the off part's write
    # S = (a ∘ off) U, S F, each rank-1 block's U_c F U_cᵀ; with gradients also g_S Uᵀ, aᵀ g_S and the
    # read gradient.
    box = vjp + sum(2 * T * c * o + 2 * T * o * o + 2 * c * o * o for c, (i, o) in zip(rank, dims))
    box_gradients = box + sum(4 * T * c * o + 2 * T * c * i + 2 * c * o * o for c, (i, o) in zip(rank, dims))
    return {
        "fwd": fwd,
        "vjp": vjp,
        "target_fwd": target_sites + attn + unemb,
        "target_vjp": target_sites + 2 * attn + unemb,
        "fisher_diag": vjp,  # per sample
        "fisher_written": vjp + written,  # per sample
        "gradients": vjp + site,
        "jvp": 2 * site + 2 * attn + unemb,
        "read_cov": read,
        "written": written,
        "eigh": eigh,
        "box": box,
        "box_gradients": box_gradients,
    }


def cmd_engine_flops(log: Path, train: int, box: bool, samples: int = 2) -> dict:
    """Training FLOPs of an engine run (module note): its start (site statistics on every training
    sequence, Fisher-SVD), and per training sequence its target forward, start forward, every
    selection round, its pieces step and its growth tests. Eval selections are left out."""
    lines = open(log).read().splitlines()
    pieces = 1.0
    total = {"start": 0.0, "selection": 0.0, "step": 0.0, "growth": 0.0, "eval": 0.0}
    c = engine_costs(pieces)
    sites = len(SITE_DIMS) * LAYERS
    # Site statistics: per training sequence a target forward, the read moments, `samples`
    # sampled reverse passes with the written Fisher; then one Fisher-SVD per site.
    total["start"] += train * (c["target_fwd"] + c["read_cov"] + samples * (c["target_vjp"] + c["written"]))
    total["start"] += sum(30 * (i ** 3 + o ** 3) for i, o in [SITE_DIMS[k] for k in SITE_DIMS] * LAYERS)
    rounds = []  # (tried, kept) of the selection being read
    sequences = 0
    in_eval = False

    def selection_cost(rs):
        # Round 1: forward, mask gradients, Fisher diagonal. Each round: the proposal's forward.
        # The next round reuses that forward when every proposer kept (gradients only), reuses
        # everything when none kept, else recomputes both.
        cost = c["fwd"] + c["vjp"] + samples * c["fisher_diag"]
        for i, (tried, kept) in enumerate(rs):
            cost += c["fwd"]
            if i + 1 < len(rs):
                cost += c["vjp"] if kept == tried else (0.0 if kept == 0 else c["fwd"] + c["vjp"])
        return cost

    pending = []
    for line in lines:
        m = re.search(r"selection round \d+ .*?: (\d+) inputs flipped \d+ entries, (\d+) kept", line)
        if m:
            rounds.append((int(m.group(1)), int(m.group(2))))
            continue
        m = re.search(r'"pieces":(\d+)', line)
        if m and line.startswith("eval "):
            pieces = int(m.group(1)) / sum(min(i, o) for i, o in [SITE_DIMS[k] for k in SITE_DIMS] * LAYERS)
            c = engine_costs(pieces)
        if re.search(r"eval sequence \d+:", line):
            total["eval"] += 2 * c["target_fwd"] / 2 + 2 * c["fwd"] + selection_cost(rounds)
            rounds = []
            continue
        if re.search(r"pass \d+ sequence \d+:", line):
            sequences += 1
            # Target forward, start forward, selection, then the step: forward, gradients, the
            # Fisher with the written blocks, read covariances, two shrunk inverses per site,
            # the tangent, and one line-search forward.
            total["selection"] += c["target_fwd"] + c["fwd"] + selection_cost(rounds)
            total["step"] += c["fwd"] + c["gradients"] + samples * c["fisher_written"] + c["read_cov"] + c["eigh"] + c["jvp"] + c["fwd"]
            if box:
                # The box error and its gradients at the start, and the box error of the trial.
                total["step"] += c["box_gradients"] + c["box"]
            rounds = []
            continue
        if "split test" in line or "dropped-atoms test" in line:
            # The candidate's target trace, its selection, and its comparison forward.
            total["growth"] += c["target_fwd"] + selection_cost(rounds) + c["fwd"]
            rounds = []
    total["train"] = total["start"] + total["selection"] + total["step"] + total["growth"]
    total["sequences"] = sequences
    total["final_pieces_per_rank"] = pieces
    total["per_primitive_sequence"] = c
    print(json.dumps({k: v for k, v in total.items() if k != "per_primitive_sequence"}, indent=1))
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("n", type=int)
    e.add_argument("out", type=Path)
    t = sub.add_parser("train")
    t.add_argument("out")
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--ci", type=int, nargs=4, default=[2048, 8, 8192, 16], metavar=("D_MODEL", "BLOCKS", "HIDDEN", "HEADS"))
    t.add_argument("--c-scale", type=float, default=1.0)
    t.add_argument("--warmup", type=int, default=400)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--device", default="mps")
    s = sub.add_parser("sets")
    s.add_argument("run", type=Path)
    s.add_argument("--device", default="mps")
    s.add_argument("--check", type=int, default=4)
    c = sub.add_parser("score")
    c.add_argument("run", type=Path)
    c.add_argument("--lease", type=int, default=8)
    f = sub.add_parser("engine-flops")
    f.add_argument("log", type=Path)
    f.add_argument("--train", type=int, required=True)
    f.add_argument("--box", action="store_true", help="the run trained under the box claim")
    a = p.parse_args()
    match a.cmd:
        case "export":
            cmd_export(a.n, a.out)
        case "train":
            cmd_train(a)
        case "sets":
            cmd_sets(a.run, a.device, a.check)
        case "score":
            cmd_score(a.run, a.lease)
        case "engine-flops":
            cmd_engine_flops(a.log, a.train, a.box)


if __name__ == "__main__":
    main()
