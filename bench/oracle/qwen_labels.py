"""Label table for the oracle on a library of a Hugging Face Qwen3 model (#2951): the measured behaviour of
MLP functions (parts) on the real model M, under the edit family the fit and the edit-faithfulness
driver use (interchange's Patch::Part, 84cfd567ec), in vpd_labels.py's schema so vpd_oracle.py reads it.

A part i of layer l's MLP reads a_i = relu(v_i . x + b_i) from M's own MLP input x (the normed residual
entering the MLP) and writes a_i u_i. Its edit with factor alpha at position p adds (alpha - 1) a_i(p) u_i
to M's MLP output at row p only (alpha = 0 removes the part, alpha = 2 amplifies it); every later
computation is run again (no linear approximation). The edit is a claim about M: that M's MLP output
holds u_i with the coefficient the part computes from M's own input.

Contexts. Windows of `--tokens` (windows_T128.u32 of the Qwen3 data release, rows of 128 tokens): a pool
of `--pool` rows from `--offset`. For each part: its `--top` contexts of largest peak activity in the
pool and `--random` more drawn uniformly from the rest (seeded).

Per part and context (p = the position of largest activity, after the first token):
  activity [T]          a_i at every position (float16)
  position              p
  for alpha in (0, 2): kl (KL(clean || edited) of the next-token distribution at p, nats), next (the
                        change of log p of the actual next token), the 10 tokens whose probability rises
                        most and the 10 that fall most (ids, change of probability, change of log p).

Functions: `vpd_lens.py functions` (a transcoder library's start) or a fit's export, per layer l
h.{l}.mlp.function.U [C, d], .V [d, C], .bias [C].

  qwen_labels.py --model SNAPSHOT --functions F --tokens WINDOWS.u32 --out DIR [--offset 0] [--pool 2048]
                 [--top 16] [--random 16] [--layers 0,1,...] [--limit N] [--rows 64] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
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
    """A forward hook on one layer's MLP: records each row's activity of its part and, when `scale` is
    set, adds scale[r] a_r(p_r) u_r to the MLP's output at row r, position p_r."""

    def __init__(self):
        self.V = self.b = self.U = self.pos = self.scale = None
        self.activity = None

    def __call__(self, module, args, output):
        if self.V is None:
            return output
        x = args[0].float()
        a = torch.relu(torch.einsum("rtd,rd->rt", x, self.V) + self.b[:, None])
        self.activity = a
        if self.scale is None:
            return output
        at = torch.arange(len(a), device=a.device)
        delta = torch.zeros_like(output, dtype=torch.float32)
        delta[at, self.pos] = (self.scale * a[at, self.pos])[:, None] * self.U
        return (output.float() + delta).to(output.dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--functions", required=True)
    ap.add_argument("--tokens", required=True, help="windows of 128 tokens, raw uint32 (windows_T128.u32)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pool", type=int, default=2048)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--random", type=int, default=16)
    ap.add_argument("--layers", default="")
    ap.add_argument("--limit", type=int, default=0, help="each layer's first N parts only")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM

    dev = device()
    wide = torch.float64 if dev.type != "mps" else torch.float32  # log-probabilities normalized in float64 where the device has it
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev).eval()
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
    parts = {}
    for l in layers:
        n = f"h.{l}.mlp.function"
        C = fn[f"{n}.U"].shape[0] if not args.limit else min(args.limit, fn[f"{n}.U"].shape[0])
        parts[l] = (fn[f"{n}.U"][:C].float().to(dev), fn[f"{n}.V"][:, :C].float().to(dev), fn[f"{n}.bias"][:C].float().to(dev))

    # The clean pass: each part's peak activity per pool row (after the first token, the attention sink).
    peaks = {l: torch.empty(args.pool, parts[l][0].shape[0], device=dev) for l in layers}
    captured = {}

    def capture(l):
        def hook(module, a, o):
            captured[l] = a[0]
        return hook

    handles = [layers_all[l].mlp.register_forward_hook(capture(l)) for l in layers]
    t0 = time.time()
    for s in range(0, args.pool, args.rows):
        model.model(input_ids=ids[s : s + args.rows])
        for l in layers:
            U, V, b = parts[l]
            peaks[l][s : s + args.rows] = torch.relu(captured[l][:, 1:].float() @ V + b).amax(1)
    for h in handles:
        h.remove()
    print(json.dumps({"pool": args.pool, "clean_seconds": round(time.time() - t0, 1)}), flush=True)
    rng = np.random.default_rng(args.seed)
    K = args.top + args.random
    for l in layers:
        tag = f"{l}_function"
        if (out / f"site_{tag}.json").exists():
            continue
        started = time.time()
        U, V, b = parts[l]
        C = U.shape[0]
        order = peaks[l].T.argsort(dim=-1, descending=True).cpu().numpy()
        ctx = np.empty((C, K), dtype=np.int64)
        for c in range(C):
            ctx[c] = np.concatenate([order[c, : args.top], rng.choice(order[c, args.top :], size=args.random, replace=False)])
        ctx_t = torch.from_numpy(ctx).to(dev)
        activity = torch.empty(C, K, T, device=dev)
        position = torch.empty(C, K, dtype=torch.int64, device=dev)
        rec = {}
        fields = {name: {} for name, _ in ALPHAS}
        hook = hooks[l]
        pairs = torch.cartesian_prod(torch.arange(C, device=dev), torch.arange(K, device=dev))
        for s in range(0, len(pairs), args.rows):
            cs, ks = pairs[s : s + args.rows, 0], pairs[s : s + args.rows, 1]
            rows = ctx_t[cs, ks]
            hook.V, hook.b, hook.U, hook.scale = V[:, cs].T, b[cs], U[cs], None
            at = torch.arange(len(rows), device=dev)
            hidden = model.model(input_ids=ids[rows]).last_hidden_state
            act = hook.activity
            pos = act[:, 1:].argmax(-1) + 1
            activity[cs, ks], position[cs, ks] = act, pos
            lp_c = torch.log_softmax(model.lm_head(hidden[at, pos]).to(wide), -1)
            for name, alpha in ALPHAS:
                hook.pos, hook.scale = pos, torch.full((len(rows),), alpha - 1.0, device=dev)
                lp_e = torch.log_softmax(model.lm_head(model.model(input_ids=ids[rows]).last_hidden_state[at, pos]).to(wide), -1)
                delta, dp = lp_e - lp_c, lp_e.exp() - lp_c.exp()
                upi, dni = dp.topk(TOP, dim=-1).indices, (-dp).topk(TOP, dim=-1).indices
                nt = ids[rows, (pos + 1).clamp(max=T - 1)]
                nd = delta.gather(-1, nt[:, None])[:, 0]
                vals = {"kl": (lp_c.exp() * -delta).sum(-1), "next": torch.where(pos + 1 < T, nd, torch.full_like(nd, float("nan"))),
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
        save_file({k: v.contiguous() for k, v in rec.items()}, str(out / f"site_{tag}.safetensors"))
        meta = {"site": f"h.{l}.mlp.function", "layer": l, "subcomponents": C, "top": args.top, "random": args.random, "pool_offset": args.offset, "pool": args.pool,
                "alphas": dict(ALPHAS), "amplify": dict(ALPHAS)["amplify"], "edit": "Patch::Part at the peak position only", "seconds": time.time() - started,
                "seed": args.seed, "functions": str(args.functions), "model": str(args.model), "model_layers": len(layers_all)}
        (out / f"site_{tag}.json").write_text(json.dumps(meta))
        print(json.dumps({"site": meta["site"], "parts": C, "seconds": round(meta["seconds"], 1), "median_kl_ablate": float(rec["kl_ablate"].median())}), flush=True)


if __name__ == "__main__":
    main()
