"""Label table for the introspective oracle (#2951): exact measured behaviour of Qwen3-0.6B's MLP neurons on
FineWeb held-out text, the labels a reader of the model's own components is trained and scored on.

Neurons. A neuron (layer l, index i) is one coordinate of the MLP's hidden layer: its activation at a
position is a_i = SiLU(g_i . x) (u_i . x) for the normed stream x, the input of the down map, and it writes
a_i d_i (d_i the down map's column i). `--per-layer` neurons are drawn uniformly without replacement in
every layer (seeded); contexts are `--contexts` rows of the held-out 128-token windows (seeded).

Edits. W_down[:, i] <- alpha W_down[:, i] for alpha in EDITS (0: removed; 1.5: amplified). The down map is
linear, so the edited layer's output equals the down map applied to the activations with coordinate i
multiplied by alpha; every downstream computation is then run again at every position (no linear
approximation, no reuse of any state the edit changes). What precedes the edited down map is the same as
the clean run's and is reused: per context the clean pass keeps the stream after each layer's attention
and the down map's input, and an edited row starts there. Clean values come from the same code path; the
largest difference between an alpha = 1 row and the clean pass (batch shapes only) is recorded per shard
as the floor below which changes are rounding. Float32 throughout, TF32 off.

Records, per layer (shard_LL.safetensors and shard_LL.json):
  activation   [N, C, T] float16   the neuron's activation at every position
  positions    [N, C, K] int16     the K positions of largest activation in each context
  per edit e:  kl_e [N, C, K] f32       KL(edited || clean) of the next-token distribution there, nats
               next_e [N, C, K] f32     change of log p(actual next token) (NaN at the last position)
               up_ids_e                 [N, C, K, 10] the 10 tokens whose probability rises most
               up_dp_e, up_dlogp_e      [N, C, K, 10] their change of probability, and of log-probability
               down_ids_e, down_dp_e, down_dlogp_e   the same for the 10 that fall most
                                        (ranked by probability, not log-probability, so the rare tokens
                                        whose log-probability swings without mass do not fill the lists)
  where        [N, M, 2] int16     (context, position) of the neuron's M largest activations, distinct contexts
  clean, amplified [N, M, S] int32 greedy continuations of S tokens after that position, without the edit
                                   and with alpha = AMPLIFY
The json holds the neurons, contexts (window row ids), model, edits, K, M, S and timings; tokens.safetensors
holds the contexts' tokens [C, T].

  labels.py --model DIR --windows WINDOWS_T128.u32 --out DIR [--per-layer 179] [--contexts 32]
            [--layers 0-27] [--rows 64] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

EDITS = (("ablate", 0.0), ("scale", 1.5))
AMPLIFY = 1.5
K, M, S, TOP = 3, 3, 16, 10


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Runner:
    """Qwen3's decoder run layer by layer with the clean pass's per-layer state kept for edited rows."""

    def __init__(self, path: str, dev: torch.device):
        from transformers import AutoModelForCausalLM

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.dev = dev
        # SDPA with no mask is causal attention over the full window (no padding in a window).
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
        self.inner = self.model.model
        self.layers = list(self.inner.layers)

    def rotary(self, h):
        position_ids = torch.arange(h.shape[1], device=self.dev)[None]
        return self.inner.rotary_emb(h, position_ids)

    def attention_half(self, layer, h, rot):
        """The stream after the layer's attention, and the down map's input (the neurons' activations)."""
        residual = h
        x = layer.input_layernorm(h)
        x = layer.self_attn(hidden_states=x, attention_mask=None, position_embeddings=rot)[0]
        mid = residual + x
        mlp = layer.mlp
        z = layer.post_attention_layernorm(mid)
        act = mlp.act_fn(mlp.gate_proj(z)) * mlp.up_proj(z)
        return mid, act

    @torch.no_grad()
    def clean(self, tokens: torch.Tensor):
        """Per layer the stream after attention and the activations; and the final normed stream."""
        h = self.inner.embed_tokens(tokens)
        rot = self.rotary(h)
        mids, acts = [], []
        for layer in self.layers:
            mid, act = self.attention_half(layer, h, rot)
            mids.append(mid)
            acts.append(act)
            h = mid + layer.mlp.down_proj(act)
        return mids, acts, self.inner.norm(h)

    @torch.no_grad()
    def edited(self, l: int, mid: torch.Tensor, act: torch.Tensor, index: torch.Tensor, alpha: torch.Tensor):
        """The final normed stream of rows whose layer-l activation `index[r]` is multiplied by
        `alpha[r]` (mid, act: the rows' clean state at layer l)."""
        act = act.clone()
        rows = torch.arange(act.shape[0], device=self.dev)
        act[rows, :, index] = act[rows, :, index] * alpha[:, None]
        h = mid + self.layers[l].mlp.down_proj(act)
        rot = self.rotary(h)
        for layer in self.layers[l + 1 :]:
            mid_, act_ = self.attention_half(layer, h, rot)
            h = mid_ + layer.mlp.down_proj(act_)
        return self.inner.norm(h)

    def log_probs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.model.lm_head(h).float(), dim=-1)


def parse_layers(text: str, count: int) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")] if text else list(range(count))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-layer", type=int, default=179)
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--layers", default="")
    ap.add_argument("--rows", type=int, default=64, help="edited rows per forward pass (memory)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.model, dev)
    cfg = runner.model.config
    layers = parse_layers(args.layers, cfg.num_hidden_layers)
    rng = np.random.default_rng(args.seed)
    windows = np.fromfile(args.windows, dtype="<u4").reshape(-1, 128)
    context_rows = np.sort(rng.choice(len(windows), size=args.contexts, replace=False))
    # Neurons per layer drawn for every layer, so a run over some layers draws the same neurons.
    neurons_of = {l: np.sort(rng.choice(cfg.intermediate_size, size=args.per_layer, replace=False)) for l in range(cfg.num_hidden_layers)}
    tokens = torch.from_numpy(windows[context_rows].astype(np.int64)).to(dev)
    C, T = tokens.shape
    save_file({"tokens": tokens.to(torch.int32).cpu()}, str(out / "tokens.safetensors"))
    started = time.time()
    mids, acts, final = runner.clean(tokens)
    for l in layers:
        if (out / f"shard_{l:02d}.json").exists():
            continue
        t0 = time.time()
        idx = torch.from_numpy(neurons_of[l]).to(dev)
        N = len(idx)
        activation = acts[l][:, :, idx].permute(2, 0, 1).contiguous()  # [N, C, T]
        positions = activation.topk(K, dim=-1).indices  # [N, C, K]
        rec = {"activation": activation.to(torch.float16).cpu(), "positions": positions.to(torch.int16).cpu()}
        # The shared path's floor: alpha = 1 rows against the clean pass (they differ only by batch shape).
        check = runner.edited(l, mids[l][:2], acts[l][:2], idx[:2], torch.ones(2, device=dev))
        floor = float((runner.log_probs(check[:, -1]) - runner.log_probs(final[:2, -1])).abs().max())
        for name, alpha in EDITS:
            kl = torch.empty(N, C, K)
            nxt = torch.empty(N, C, K)
            up_ids = torch.empty(N, C, K, TOP, dtype=torch.int32)
            up_dp, up_dl = torch.empty(N, C, K, TOP), torch.empty(N, C, K, TOP)
            down_ids = torch.empty(N, C, K, TOP, dtype=torch.int32)
            down_dp, down_dl = torch.empty(N, C, K, TOP), torch.empty(N, C, K, TOP)
            pairs = [(n, c) for n in range(N) for c in range(C)]
            for s in range(0, len(pairs), args.rows):
                chunk = pairs[s : s + args.rows]
                ns = torch.tensor([n for n, _ in chunk], device=dev)
                cs = torch.tensor([c for _, c in chunk], device=dev)
                h = runner.edited(l, mids[l][cs], acts[l][cs], idx[ns], torch.full((len(chunk),), alpha, device=dev))
                pos = positions[ns, cs]  # [R, K]
                at = torch.arange(len(chunk), device=dev)[:, None]
                lp_e = runner.log_probs(h[at, pos])  # [R, K, V]
                lp_c = runner.log_probs(final[cs][at, pos])
                delta = lp_e - lp_c
                dp = lp_e.exp() - lp_c.exp()
                k = (lp_e.exp() * delta).sum(-1)
                upi = dp.topk(TOP, dim=-1).indices
                dni = (-dp).topk(TOP, dim=-1).indices
                nt = torch.where(pos + 1 < T, tokens[cs][torch.arange(len(chunk), device=dev)[:, None], (pos + 1).clamp(max=T - 1)], torch.zeros_like(pos))
                nd = delta.gather(-1, nt[..., None])[..., 0]
                nd = torch.where(pos + 1 < T, nd, torch.full_like(nd, float("nan")))
                for r, (n, c) in enumerate(chunk):
                    kl[n, c], nxt[n, c] = k[r].cpu(), nd[r].cpu()
                    up_ids[n, c], down_ids[n, c] = upi[r].to(torch.int32).cpu(), dni[r].to(torch.int32).cpu()
                    up_dp[n, c], up_dl[n, c] = dp[r].gather(-1, upi[r]).cpu(), delta[r].gather(-1, upi[r]).cpu()
                    down_dp[n, c], down_dl[n, c] = dp[r].gather(-1, dni[r]).cpu(), delta[r].gather(-1, dni[r]).cpu()
            rec.update({f"kl_{name}": kl, f"next_{name}": nxt, f"up_ids_{name}": up_ids, f"up_dp_{name}": up_dp, f"up_dlogp_{name}": up_dl,
                        f"down_ids_{name}": down_ids, f"down_dp_{name}": down_dp, f"down_dlogp_{name}": down_dl})
        # Continuations at each neuron's M largest activations in distinct contexts.
        best_c = activation.amax(-1)  # [N, C]
        where = torch.empty(N, M, 2, dtype=torch.int64)
        for n in range(N):
            cs = best_c[n].topk(M).indices
            where[n, :, 0] = cs.cpu()
            where[n, :, 1] = activation[n, cs].argmax(-1).cpu()
        clean_cont, amp_cont = continuations(runner, tokens, l, idx, where)
        rec.update({"where": where.to(torch.int16), "clean": clean_cont, "amplified": amp_cont})
        save_file({k: v.contiguous() for k, v in rec.items()}, str(out / f"shard_{l:02d}.safetensors"))
        meta = {"layer": l, "neurons": [[l, int(i)] for i in neurons_of[l]], "contexts": [int(r) for r in context_rows], "model": args.model,
                "edits": [{"name": n, "alpha": a} for n, a in EDITS], "amplify": AMPLIFY, "K": K, "M": M, "S": S, "top": TOP,
                "alpha_one_max_log_prob_difference": floor, "seconds": time.time() - t0, "seed": args.seed}
        (out / f"shard_{l:02d}.json").write_text(json.dumps(meta))
        print(json.dumps({"layer": l, "neurons": N, "seconds": round(time.time() - t0, 1), "alpha_one_floor": floor,
                          "median_kl_ablate": float(rec["kl_ablate"].median()), "median_kl_scale": float(rec["kl_scale"].median())}), flush=True)
    print(json.dumps({"total_seconds": round(time.time() - started, 1)}), flush=True)


@torch.no_grad()
def continuations(runner: Runner, tokens, l: int, idx, where):
    """Greedy S-token continuations after each (context, position) of `where`, without and with the
    neuron's down column multiplied by AMPLIFY, every row its own neuron, the edit applied in that row's
    down map at every position and step (a weight edit acts everywhere). Rows are batched by prefix
    length, so no row is padded."""
    N = where.shape[0]
    by_length: dict[int, list[tuple[int, int]]] = {}
    for n in range(N):
        for m in range(M):
            by_length.setdefault(int(where[n, m, 1]) + 1, []).append((n, m))
    outs = {}
    for alpha_name, alpha in (("clean", 1.0), ("amplified", AMPLIFY)):
        result = torch.full((N, M, S), -1, dtype=torch.int32)
        for length, jobs in by_length.items():
            for s in range(0, len(jobs), 64):
                chunk = jobs[s : s + 64]
                ids = torch.stack([tokens[int(where[n, m, 0]), :length] for n, m in chunk])
                neuron = idx[torch.tensor([n for n, _ in chunk], device=runner.dev)]

                def hook(module, inputs):
                    (x,) = inputs
                    x = x.clone()
                    rows = torch.arange(x.shape[0], device=x.device)
                    x[rows, :, neuron] = x[rows, :, neuron] * alpha
                    return (x,)

                handle = runner.layers[l].mlp.down_proj.register_forward_pre_hook(hook) if alpha != 1.0 else None
                try:
                    gen = runner.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=S, do_sample=False, pad_token_id=0)
                finally:
                    if handle is not None:
                        handle.remove()
                new = gen[:, length:].to(torch.int32).cpu()
                for r, (n, m) in enumerate(chunk):
                    result[n, m, : new.shape[1]] = new[r]
        outs[alpha_name] = result
    return outs["clean"], outs["amplified"]


if __name__ == "__main__":
    main()
