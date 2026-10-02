"""The CEGAR verifier's adversary on the 4L Pile target (#2951): input-space ascent of the per-row
mean per-position KL(target || program), the same adversary for VPD's program and for ours.

Programs
  vpd       VPD's explanation at its causal importances, mask = g (VPD's ci_masked program; the
            gates are recomputed on every candidate input, so the program is the input -> output
            map VPD claims)
  dense:B   our program: the 24 target matrices on the per-tensor dyadic lattice at resolution B
            (vpd_bits.precision_for), B chosen by `select`

Adversary (cegar.rs, token slots): at the current row, the gradient of the objective with respect
to each position's one-hot is g_e @ wte.T (gates held at their values for VPD: the gradient only
proposes). The Frank-Wolfe oracle is the best token of every position; its linearised gains rank
the single-position swaps. Candidates are evaluated exactly in the gain order in batches of MB
rows (the memory's batch), and a step takes the best candidate of the first batch that holds a
strict increase; the ascent stops when a batch-sweep of every positive-gain position finds none.
The ladder is the row-mean objective after k accepted steps (an ascent that stopped keeps its
endpoint), at the rungs VPD reports.

usage:
  vpd_input_ladder.py select ROWS B[,B..]                        bits + N KL / ln 2 per B
  vpd_input_ladder.py ladder PROGRAM ROWS OFFSET SEQ STEPS OUT   PROGRAM = vpd | dense:B
"""

import json
import math
import sys
import time

import torch
import torch.nn.functional as F

from vpd_bits import lattice_bits, precision_for, quantize
from vpd_model import gelu_tanh, load_target, load_vpd, rms, site_names, val_tokens

DEV = "mps"
MB = 8  # candidates per exact-evaluation batch of the ascent
EVAL = int(__import__("os").environ.get("VPD_EVAL", "1"))  # rows per forward pass (memory only; the ascent is the same for any EVAL)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
target = load_target(DEV)
W0 = {n: target.site(n).W.clone() for n in site_names()}


def forward_embedded(x: torch.Tensor) -> torch.Tensor:
    """Target.forward from the input embeddings (the relaxation's entry point)."""
    B, T, _ = x.shape
    tg = target
    for i in range(tg.n_layer):
        s = lambda k: tg.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
        h = rms(x, tg.norms[2 * i], tg.eps)
        q = s("q_proj")(h).view(B, T, tg.n_head, tg.hd).transpose(1, 2)
        k = s("k_proj")(h).view(B, T, tg.n_head, tg.hd).transpose(1, 2)
        v = s("v_proj")(h).view(B, T, tg.n_head, tg.hd).transpose(1, 2)
        q, k = tg._rope(q, T), tg._rope(k, T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s("o_proj")(y.transpose(1, 2).reshape(B, T, -1))
        h = rms(x, tg.norms[2 * i + 1], tg.eps)
        x = x + s("down_proj")(gelu_tanh(s("c_fc")(h)))
    x = rms(x, tg.ln_f, tg.eps)
    return x @ tg.wte.T


def row_kl(ref: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(ref, -1)
    return (lp.exp() * (lp - F.log_softmax(cand, -1))).sum(-1).mean(-1)


class Dense:
    def __init__(self, b: int):
        self.Wq, self.bits = {}, 0
        for n in site_names():
            p = precision_for(W0[n], b)
            self.Wq[n] = quantize(W0[n], p)
            self.bits += lattice_bits(W0[n].cpu(), p)

    def run(self, ids, x=None):
        for n in site_names():
            target.site(n).W = self.Wq[n]
        try:
            return forward_embedded(target.wte[ids] if x is None else x)
        finally:
            for n in site_names():
                target.site(n).W = W0[n]


class Vpd:
    def __init__(self):
        self.vpd = load_vpd(target, DEV)

    def run(self, ids, x=None):
        _, g = self.vpd.target_and_ci(ids)
        for n in self.vpd.names:
            target.site(n).mask = g[n]
            target.site(n).delta_mask = None
        try:
            return forward_embedded(target.wte[ids] if x is None else x)
        finally:
            self.vpd.clear()


def reference(ids):
    with torch.no_grad():
        return forward_embedded(target.wte[ids])


def objective(prog, ids):
    """Row-mean KL of every row of ids, in batches of MB."""
    out = []
    with torch.no_grad():
        for i in range(0, ids.shape[0], EVAL):
            b = ids[i:i + EVAL]
            out.append(row_kl(reference(b), prog.run(b)))
    return torch.cat(out)


def token_gains(prog, row):
    """Linearised gain of every (position, token) swap at `row` (one sequence)."""
    ids = row[None]
    x_ref = target.wte[ids].clone().requires_grad_(True)
    x_prog = target.wte[ids].clone().requires_grad_(True)
    with torch.enable_grad():
        kl = row_kl(forward_embedded(x_ref), prog.run(ids, x_prog)).sum()
        g_ref, g_prog = torch.autograd.grad(kl, [x_ref, x_prog])
    g = (g_ref + g_prog)[0]  # [T, d]
    scores = g @ target.wte.T  # [T, V]
    return scores - scores.gather(1, row[:, None])


def ascend(prog, row, steps):
    """One ascent from `row`; returns the objective after each accepted step."""
    cur = row.clone()
    val = objective(prog, cur[None]).item()
    path = [val]
    while len(path) <= steps:
        gains = token_gains(prog, cur)
        best_gain, best_tok = gains.max(1)
        order = [int(p) for p in torch.argsort(best_gain, descending=True) if best_gain[p] > 0]
        accepted = False
        for i in range(0, len(order), MB):
            pos = order[i:i + MB]
            cands = cur[None].repeat(len(pos), 1)
            for k, p in enumerate(pos):
                cands[k, p] = best_tok[p]
            vals = objective(prog, cands)
            j = int(vals.argmax())
            if vals[j].item() > val:
                cur, val, accepted = cands[j], vals[j].item(), True
                break
        if not accepted:
            break
        path.append(val)
        del gains
        torch.mps.empty_cache()
        log(f"  step {len(path) - 1}: KL {val:.4f} ({i // MB + 1} batches)")
    return path, cur


mode = sys.argv[1]
if mode == "select":
    rows, bs = int(sys.argv[2]), [int(b) for b in sys.argv[3].split(",")]
    ids = val_tokens(rows, offset=0).to(DEV)
    n_positions = 4096 * 512  # the declared behaviour: the val shard's 4096 rows of 512 tokens
    table = []
    for b in bs:
        d = Dense(b)
        kl = objective(d, ids).mean().item()
        total = d.bits + n_positions * kl / math.log(2)
        table.append({"b": b, "program_bits": d.bits, "mean_kl": kl, "data_bits": n_positions * kl / math.log(2), "total": total})
        log(json.dumps(table[-1]))
    best = min(table, key=lambda r: r["total"])
    log(f"selected b = {best['b']}")
    print(json.dumps({"table": table, "selected": best}, indent=1))
elif mode == "ladder":
    name, rows, offset, seq, steps, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    prog = Vpd() if name == "vpd" else Dense(int(name.split(":")[1]))
    ids = val_tokens(rows, seq=seq, offset=offset).to(DEV)
    paths = []
    for r in range(rows):
        path, _ = ascend(prog, ids[r], steps)
        paths.append(path)
        log(f"{name} row {r}: steps {len(path) - 1}, KL {path[0]:.4f} -> {path[-1]:.4f}")
        rungs = [k for k in (0, 1, 2, 5, 10, 20, 40, 100) if k <= steps]
        ladder = {k: sum(p[min(k, len(p) - 1)] for p in paths) / len(paths) for k in rungs}
        worst = {k: max(p[min(k, len(p) - 1)] for p in paths) for k in rungs}
        json.dump({"program": name, "rows_done": len(paths), "seq": seq, "offset": offset,
                   "ladder_mean": ladder, "ladder_worst": worst, "paths": paths}, open(out, "w"), indent=1)
    log(f"ladder mean {ladder} worst {worst}")
