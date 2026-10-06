"""Label table for the oracle on a library of a Hugging Face Qwen3 model (#2951): the measured behaviour of
MLP functions (parts) on the real model M, under the edit families the fit and the edit-faithfulness
driver use (interchange's Patch::Part, PartFrom and Parts), in vpd_labels.py's schema so vpd_oracle.py
reads it.

A part i of layer l's MLP reads a_i = relu(v_i . x + b_i) from M's own MLP input x (the normed residual
entering the MLP) and writes a_i u_i: Patch::Part's definition, the same on M and on P. Its edit with
factor alpha adds (alpha - 1) a_i u_i to M's MLP output at the edited rows (alpha = 0
removes the part, alpha = 2 amplifies it), every later computation run again (no linear
approximation). Families (`--family`, recorded):
  row    Patch::Part: the part at the context's peak p only;
  from   Patch::PartFrom from the first token after the attention sink: the part at every row 1..T-1,
         its effect at p including the earlier rows' through attention;
  parts  Patch::Parts: the part together with every other part of its layer active at p, removed at p
         (their ids in `parts`, -1 padded).
The effect on M is measured at p both ways: kl (KL(M || M_e), nats) and effect (KL(M_e || M), bits, the
edits driver's binning quantity).

Contexts. Windows of `--tokens` (windows_T128.u32 of the Qwen3 data release, rows of 128 tokens): a pool
of `--pool` rows from `--offset`. For each part: its `--top` contexts of largest peak activity in the
pool and `--random` more drawn uniformly from the rest (seeded).

Per part and context (p = the position of largest activity, after the first token):
  activity [T]          a_i at every position (float16)
  position              p
  for alpha in (0, 2): kl, effect, next (the change of log p of the actual next token), the 10 tokens
                        whose probability rises most and the 10 that fall most (ids, change of
                        probability, change of log p).

Functions: `vpd_lens.py functions` (a transcoder library's start) or mpd_library_mdl_2951's export
(EditSettings.functions, the fit's posterior mean), per layer l h.{l}.mlp.function.U [C, d], .V [d, C],
.bias [C] (.index [C]: the part's row in the layer's operators).

  qwen_labels.py labels --model SNAPSHOT --functions F --tokens WINDOWS.u32 --out DIR [--offset 0] [--pool 2048]
                 [--top 16] [--random 16] [--layers 0,1,...] [--limit N] [--rows 64] [--seed 0]
                 [--family row|from|parts]
  qwen_labels.py parity --model SNAPSHOT --functions F --tokens WINDOWS.u32 --held-out FIRST --layers K
                 --records OUT/EDITS_{name}.experiments.jsonl
      the same edits as the Rust driver's records (Patch::Part, PartFrom, Parts), on M cut to its first
      K blocks in float64: KL(M_e || M) at the edited token against the driver's, to 1e-6 bits
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

TOP = 10
ALPHAS = (("ablate", 0.0), ("amplify", 2.0))
T = 128


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Edit:
    """A forward hook on one layer's MLP. Per batch row r its parts (k of them: reads V [R, k, d] and
    biases b [R, k], writes U [R, k, d]); it records each part's activity at every position, and when
    `scale` is set adds scale[r] * sum_k a_rk(t) u_rk to the MLP's output at the rows `rows` [R, T]
    (a 0/1 mask) marks."""

    def __init__(self):
        self.V = self.b = self.U = self.rows = self.scale = None
        self.activity = None

    def __call__(self, module, args, output):
        if self.V is None:
            return output
        x = args[0].to(self.V.dtype)
        a = torch.relu(torch.einsum("rtd,rkd->rkt", x, self.V) + self.b[:, :, None])
        self.activity = a
        if self.scale is None:
            return output
        delta = torch.einsum("rkt,rkd->rtd", a, self.U) * (self.scale[:, None] * self.rows)[..., None]
        return (output.to(delta.dtype) + delta).to(output.dtype)


def load_model(snapshot: str, dev: torch.device, dtype, layers: int | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(snapshot)
    if layers is not None:
        config.num_hidden_layers = layers  # the first `layers` blocks, then the final norm and head
    return AutoModelForCausalLM.from_pretrained(snapshot, config=config, dtype=dtype).to(dev).eval()


def parts_of(fn: dict, layer: int, dev, dtype):
    n = f"h.{layer}.mlp.function"
    return fn[f"{n}.U"].to(dev, dtype), fn[f"{n}.V"].T.to(dev, dtype), fn[f"{n}.bias"].to(dev, dtype)  # U [C, d], V [C, d], b [C]


@torch.no_grad()
def labels(args):
    dev = device()
    wide = torch.float64 if dev.type != "mps" else torch.float32  # log-probabilities normalized in float64 where the device has it
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(args.model, dev, torch.float32)
    layers_all = model.model.layers
    hooks = [Edit() for _ in layers_all]
    for layer, h in zip(layers_all, hooks):
        layer.mlp.register_forward_hook(h)
    fn = load_file(args.functions)
    layers = [int(x) for x in args.layers.split(",") if x] or sorted({int(k.split(".")[1]) for k in fn if k.endswith(".mlp.function.U")})
    windows = np.fromfile(args.tokens, dtype="<u4").reshape(-1, T)
    ids = torch.from_numpy(windows[args.offset : args.offset + args.pool].astype(np.int64)).to(dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"rows": torch.arange(args.offset, args.offset + args.pool, dtype=torch.int64), "tokens": ids.to(torch.int32).cpu()}, str(out / "contexts.safetensors"))
    # Every part of each layer (the parts family removes all those active at p); the first `--limit` are asked about.
    parts = {l: parts_of(fn, l, dev, torch.float32) for l in layers}

    # The clean pass: every part's activity on every pool row; its peak after the first token (the attention sink).
    peaks = {l: torch.empty(args.pool, parts[l][0].shape[0], device=dev) for l in layers}
    for s in range(0, args.pool, args.rows):
        rows = ids[s : s + args.rows]
        for l in layers:
            U, V, b = parts[l]
            hooks[l].V, hooks[l].b, hooks[l].U, hooks[l].scale = V[None].expand(len(rows), -1, -1), b[None].expand(len(rows), -1), U[None].expand(len(rows), -1, -1), None
        model.model(input_ids=rows)
        for l in layers:
            peaks[l][s : s + args.rows] = hooks[l].activity[:, :, 1:].amax(-1)
            hooks[l].V = None
    rng = np.random.default_rng(args.seed)
    K = args.top + args.random
    for l in layers:
        tag = f"{l}_function"
        if (out / f"site_{tag}.json").exists():
            continue
        started = time.time()
        U, V, b = parts[l]
        C = U.shape[0] if not args.limit else min(args.limit, U.shape[0])
        order = peaks[l][:, :C].T.argsort(dim=-1, descending=True).cpu().numpy()
        ctx = np.empty((C, K), dtype=np.int64)
        for c in range(C):
            ctx[c] = np.concatenate([order[c, : args.top], rng.choice(order[c, args.top :], size=args.random, replace=False)])
        ctx_t = torch.from_numpy(ctx).to(dev)
        activity = torch.empty(C, K, T, device=dev)
        position = torch.empty(C, K, dtype=torch.int64, device=dev)
        together = []
        rec = {}
        fields = {name: {} for name, _ in ALPHAS}
        hook = hooks[l]
        pairs = torch.cartesian_prod(torch.arange(C, device=dev), torch.arange(K, device=dev))
        for s in range(0, len(pairs), args.rows):
            cs, ks = pairs[s : s + args.rows, 0], pairs[s : s + args.rows, 1]
            rows = ctx_t[cs, ks]
            R_ = len(rows)
            at = torch.arange(R_, device=dev)
            # The clean pass over these rows: every part's activity (to find the others active at p).
            hook.V, hook.b, hook.U, hook.scale = V[None].expand(R_, -1, -1), b[None].expand(R_, -1), U[None].expand(R_, -1, -1), None
            hidden = model.model(input_ids=ids[rows]).last_hidden_state
            every = hook.activity  # [R, C, T]
            act = every[at, cs]
            pos = act[:, 1:].argmax(-1) + 1
            activity[cs, ks], position[cs, ks] = act, pos
            lp_c = torch.log_softmax(model.lm_head(hidden[at, pos]).to(wide), -1)
            if args.family == "parts":
                active = every[at, :, pos] > 0  # [R, C]
                active[at, cs] = True
                width = int(active.sum(1).max())
                chosen = torch.full((R_, width), -1, dtype=torch.int64, device=dev)
                for r in range(R_):
                    w = torch.nonzero(active[r]).reshape(-1)
                    chosen[r, : len(w)] = w
                together.append(chosen.cpu())
                safe = chosen.clamp(min=0)
                keep = (chosen >= 0).to(U.dtype)
                hook.V, hook.b, hook.U = V[safe], b[safe], U[safe] * keep[..., None]
            else:
                hook.V, hook.b, hook.U = V[cs][:, None], b[cs][:, None], U[cs][:, None]
            t_ = torch.arange(T, device=dev)[None]
            hook.rows = (t_ == pos[:, None]) if args.family != "from" else (t_ >= 1)
            hook.rows = hook.rows.to(U.dtype)
            for name, alpha in ALPHAS:
                hook.scale = torch.full((R_,), alpha - 1.0, device=dev)
                lp_e = torch.log_softmax(model.lm_head(model.model(input_ids=ids[rows]).last_hidden_state[at, pos]).to(wide), -1)
                delta, dp = lp_e - lp_c, lp_e.exp() - lp_c.exp()
                upi, dni = dp.topk(TOP, dim=-1).indices, (-dp).topk(TOP, dim=-1).indices
                nt = ids[rows, (pos + 1).clamp(max=T - 1)]
                nd = delta.gather(-1, nt[:, None])[:, 0]
                vals = {"kl": (lp_c.exp() * -delta).sum(-1), "effect": (lp_e.exp() * delta).sum(-1) / math.log(2),
                        "next": torch.where(pos + 1 < T, nd, torch.full_like(nd, float("nan"))),
                        "up_ids": upi, "up_dp": dp.gather(-1, upi), "up_dlogp": delta.gather(-1, upi),
                        "down_ids": dni, "down_dp": dp.gather(-1, dni), "down_dlogp": delta.gather(-1, dni)}
                for key, v in vals.items():
                    fields[name].setdefault(key, []).append(v.cpu())
            hook.scale = None
        hook.V = None
        for name, _ in ALPHAS:
            for key, parts_ in fields[name].items():
                v = torch.cat(parts_).reshape(C, K, *parts_[0].shape[1:])
                rec[f"{key}_{name}"] = v.to(torch.int32) if key.endswith("_ids") else v.to(torch.float32)
        rec.update({"contexts": torch.from_numpy(ctx).to(torch.int32), "activity": activity.to(torch.float16).cpu(), "position": position.to(torch.int16).cpu()})
        if together:
            width = max(x.shape[1] for x in together)
            rec["parts"] = torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1]), value=-1) for x in together]).reshape(C, K, width).to(torch.int32)
        save_file({k: v.contiguous() for k, v in rec.items()}, str(out / f"site_{tag}.safetensors"))
        meta = {"site": f"h.{l}.mlp.function", "layer": l, "subcomponents": C, "top": args.top, "random": args.random, "pool_offset": args.offset, "pool": args.pool,
                "alphas": dict(ALPHAS), "amplify": dict(ALPHAS)["amplify"], "family": args.family, "edit": "Patch::Part's", "edit_from": 1 if args.family == "from" else None,
                "seconds": time.time() - started, "seed": args.seed, "functions": str(args.functions), "model": str(args.model), "model_layers": len(layers_all)}
        (out / f"site_{tag}.json").write_text(json.dumps(meta))
        print(json.dumps({"site": meta["site"], "parts": C, "seconds": round(meta["seconds"], 1), "median_effect_bits_ablate": float(rec["effect_ablate"].median())}), flush=True)


@torch.no_grad()
def parity(args):
    """The Rust driver's edits (records of Patch::Part, PartFrom and Parts) applied here: KL(M_e || M)
    at the edited token, in bits, against the driver's."""
    dev, dtype = torch.device("cpu"), torch.float64
    model = load_model(args.model, dev, dtype, args.layers)
    hooks = [Edit() for _ in model.model.layers]
    for layer, h in zip(model.model.layers, hooks):
        layer.mlp.register_forward_hook(h)
    fn = load_file(args.functions)
    windows = np.fromfile(args.tokens, dtype="<u4").reshape(-1, args.context)
    worst, checked = 0.0, 0
    for line in open(args.records):
        r = json.loads(line)
        if r["family"] not in ("remove_part", "amplify_part", "remove_parts", "remove_part_from"):
            continue
        layer = r["parts"][0][0]
        n = f"h.{layer}.mlp.function"
        index = fn[f"{n}.index"].round().long().tolist()
        cols = [index.index(p[1]) for p in r["parts"]]
        U, V, b = parts_of(fn, layer, dev, dtype)
        ids = torch.from_numpy(windows[args.held_out + r["sequence"]].astype(np.int64))[None]
        p = r["position"]
        hook = hooks[layer]
        hook.V, hook.b, hook.U, hook.scale = None, None, None, None
        lp_c = torch.log_softmax(model(input_ids=ids).logits[0, p].to(dtype), -1)
        hook.V, hook.b, hook.U = V[cols][None], b[cols][None], U[cols][None]
        t_ = torch.arange(ids.shape[1])[None]
        hook.rows = ((t_ >= p) if r["family"] == "remove_part_from" else (t_ == p)).to(dtype)
        hook.scale = torch.tensor([r["factor"] - 1.0], dtype=dtype)
        lp_e = torch.log_softmax(model(input_ids=ids).logits[0, p].to(dtype), -1)
        hook.V = None
        ours = float((lp_e.exp() * (lp_e - lp_c)).sum() / math.log(2))
        theirs = r["effect_bits_at_edited_token"]
        diff = abs(ours - theirs)
        worst, checked = max(worst, diff), checked + 1
        print(json.dumps({"family": r["family"], "parts": r["parts"], "position": p, "factor": r["factor"], "rust_bits": theirs, "python_bits": ours, "diff": diff}))
    print(json.dumps({"checked": checked, "max_abs_diff_bits": worst}))
    assert checked > 0 and worst <= args.tolerance, f"parity: worst difference {worst} bits over {checked} edits"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("labels")
    a.add_argument("--model", required=True)
    a.add_argument("--functions", required=True)
    a.add_argument("--tokens", required=True, help="windows of 128 tokens, raw uint32 (windows_T128.u32)")
    a.add_argument("--out", required=True)
    a.add_argument("--offset", type=int, default=0)
    a.add_argument("--pool", type=int, default=2048)
    a.add_argument("--top", type=int, default=16)
    a.add_argument("--random", type=int, default=16)
    a.add_argument("--layers", default="")
    a.add_argument("--limit", type=int, default=0, help="each layer's first N parts only")
    a.add_argument("--rows", type=int, default=64)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--family", default="row", choices=("row", "from", "parts"))
    p = sub.add_parser("parity")
    p.add_argument("--model", required=True)
    p.add_argument("--functions", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--held-out", type=int, required=True, help="the driver's settings held_out[0]: the windows row of its held-out sequence 0")
    p.add_argument("--layers", type=int, default=None, help="the driver's settings layers (M cut to its first K blocks)")
    p.add_argument("--records", required=True)
    p.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()
    {"labels": labels, "parity": parity}[args.command](args)


if __name__ == "__main__":
    main()
