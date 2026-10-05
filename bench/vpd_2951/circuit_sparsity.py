"""Circuit sparsity against node basis on the 4-layer Pile model M (#2951), following Arora, Wu et
al. (arXiv 2601.22594) and Marks et al. (2025): faithfulness and completeness of circuits of the k
nodes of highest RelP attribution, on the subject-verb agreement pairs (simple, rc, within_rc,
nounpp; feature-circuits repo data).

Node bases (every one a decomposition of M's own forward pass, so M is evaluated in each):
    neurons  M's MLP neurons: the GELU outputs that the down projection reads (4 x 3072)
    vpd      VPD's subcomponents: at each of the 24 sites, c's activation a_c = x . V_c; the site
             computes sum_c a_c U_c + x Delta^T (Delta = W - V U, kept, not a node) (38912)

Circuit C of node set S: every node outside S is set to its mean at that position over the task's
training prompts (clean and counterfactual), everything else computed. m = logit(y) - logit(y') at
the last position on the clean prompt (y its verb form, y' the counterfactual's).
    faithfulness(S) = (E m(C_S) - E m(none)) / (E m(M) - E m(none))
    completeness(S) = (E m(C_{all minus S}) - E m(none)) / (E m(M) - E m(none))
over the held-out pairs; "none" ablates every node.

RelP (Jafari et al. 2025; Arora, Wu et al. eq. 11): attribution of node v on a pair (x, x') is
(v(x) - v(x')) . dm/dv at x, the gradient taken through M with its nonlinearities linearized (the
RMS norm's scale, GELU's gate 0.5(1 + tanh(.)) and the attention pattern frozen), summed over
positions and averaged over the training pairs. With the pattern frozen, query and key reads have
no gradient path, so VPD's q_proj and k_proj subcomponents get zero attribution. Circuits take the k nodes of largest |attribution|.

usage: circuit_sparsity.py OUT.json [TRAIN TEST]   (pairs per task, default 300 and 40)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor

from vpd_model import HERE, load_target, load_vpd

DEVICE = "mps"
DATA = Path.home() / "mpd-data/circuits/sva"
TASKS = ("simple", "rc", "within_rc", "nounpp")


def pairs(tokenizer: Tokenizer, task: str, split: str, n: int, seed: int) -> list[tuple[list[int], list[int], int, int]]:
    """n pairs (clean ids, counterfactual ids, y, y') whose prefixes have equal token counts and
    whose answers are one token each, drawn uniformly without replacement."""
    rows = [json.loads(line) for line in open(DATA / f"{task}_{split}.json")]
    out = []
    for i in np.random.default_rng(seed).permutation(len(rows)):
        r = rows[i]
        a, b = tokenizer.encode(r["clean_prefix"]).ids, tokenizer.encode(r["patch_prefix"]).ids
        y, y2 = tokenizer.encode(r["clean_answer"]).ids, tokenizer.encode(r["patch_answer"]).ids
        if len(a) == len(b) and len(y) == 1 and len(y2) == 1:
            out.append((a, b, y[0], y2[0]))
            if len(out) == n:
                break
    return out


class Model:
    """M's forward pass with each node group's activations open to interventions and to RelP's
    linearized backward (the forward value is unchanged)."""

    def __init__(self, vpd, basis: str):
        self.t, self.vpd, self.basis = vpd.target, vpd, basis
        self.groups = [f"h.{i}.mlp" for i in range(self.t.n_layer)] if basis == "neurons" else list(vpd.names)
        if basis == "vpd":
            self.delta = {n: self.t.site(n).W - (self.t.site(n).V @ self.t.site(n).U).T for n in vpd.names}

    def sizes(self) -> dict[str, int]:
        return {g: (self.t.site(g + ".c_fc").W.shape[0] if self.basis == "neurons" else self.vpd.C[g]) for g in self.groups}

    def node(self, g: str, a: Tensor, keep: dict | None, means: dict | None, record: dict | None) -> Tensor:
        if keep is not None:
            a = torch.where(keep[g], a, means[g][:a.shape[1]])
        if record is not None:
            record[g] = a.detach()
            if torch.is_grad_enabled():
                # a zero probe whose gradient is dm/da along every path
                probe = torch.zeros_like(a, requires_grad=True)
                record[g + "/probe"] = probe
                a = a + probe
        return a

    def site(self, name: str, x: Tensor, ctx) -> Tensor:
        st = self.t.site(name)
        if self.basis == "neurons":
            return x @ st.W.T
        return self.node(name, x @ st.V, *ctx) @ st.U + x @ self.delta[name].T

    def forward(self, ids: Tensor, keep=None, means=None, record=None, linear: bool = False) -> Tensor:
        t = self.t
        B, T = ids.shape
        freeze = (lambda z: z.detach()) if linear else (lambda z: z)
        ctx = (keep, means, record)

        def rms(x, w):
            return w * x * freeze(torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + t.eps))

        x = t.wte[ids]
        causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
        for i in range(t.n_layer):
            name = lambda k: f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"
            h = rms(x, t.norms[2 * i])
            heads = lambda z: z.view(B, T, t.n_head, t.hd).transpose(1, 2)
            q, k, v = (heads(self.site(name(p), h, ctx)) for p in ("q_proj", "k_proj", "v_proj"))
            q, k = t._rope(q, T), t._rope(k, T)
            z = (q @ k.transpose(-1, -2)) / math.sqrt(t.hd)
            a = z.masked_fill(~causal, float("-inf")).softmax(-1)
            y = (freeze(a) @ v).transpose(1, 2).reshape(B, T, -1)
            x = x + self.site(name("o_proj"), y, ctx)
            h = rms(x, t.norms[2 * i + 1])
            pre = self.site(name("c_fc"), h, ctx)
            gate = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (pre + 0.044715 * pre.pow(3))))
            act = pre * freeze(gate)
            if self.basis == "neurons":
                act = self.node(f"h.{i}.mlp", act, *ctx)
            x = x + self.site(name("down_proj"), act, ctx)
        return rms(x, t.ln_f) @ t.wte.T


def by_length(ps):
    groups = {}
    for p in ps:
        groups.setdefault(len(p[0]), []).append(p)
    return groups.values()


def metric(model: Model, ps, keep=None, means=None) -> float:
    """Mean over pairs of logit(y) - logit(y') at the clean prompt's last position."""
    total = 0.0
    with torch.no_grad():
        for g in by_length(ps):
            ids = torch.tensor([p[0] for p in g], device=DEVICE)
            lg = model.forward(ids, keep, means)[:, -1]
            y, y2 = torch.tensor([p[2] for p in g], device=DEVICE), torch.tensor([p[3] for p in g], device=DEVICE)
            total += (lg.gather(1, y[:, None]) - lg.gather(1, y2[:, None])).sum().item()
    return total / len(ps)


def means_and_relp(model: Model, ps) -> tuple[dict, dict]:
    longest = max(len(p[0]) for p in ps)
    sizes = model.sizes()
    sums = {g: torch.zeros(longest, sizes[g], device=DEVICE) for g in model.groups}
    count = torch.zeros(longest, device=DEVICE)
    attr = {g: torch.zeros(sizes[g], device=DEVICE) for g in model.groups}
    for g in by_length(ps):
        clean = torch.tensor([p[0] for p in g], device=DEVICE)
        patch = torch.tensor([p[1] for p in g], device=DEVICE)
        y, y2 = torch.tensor([p[2] for p in g], device=DEVICE), torch.tensor([p[3] for p in g], device=DEVICE)
        rec_c, rec_p = {}, {}
        with torch.no_grad():
            model.forward(patch, record=rec_p)
        with torch.enable_grad():
            lg = model.forward(clean, record=rec_c, linear=True)[:, -1]
            m = (lg.gather(1, y[:, None]) - lg.gather(1, y2[:, None])).sum()
            # the frozen attention pattern leaves the query and key reads without a path: zero
            grads = torch.autograd.grad(m, [rec_c[k + "/probe"] for k in model.groups], allow_unused=True)
        T = clean.shape[1]
        count[:T] += 2 * clean.shape[0]
        for k, gr in zip(model.groups, grads, strict=True):
            a, b = rec_c[k], rec_p[k]
            if gr is not None:
                attr[k] = attr[k] + ((a - b) * gr).sum((0, 1))
            sums[k][:T] += a.sum(0) + b.sum(0)
        del rec_c, rec_p, grads
    # per position: Marks et al.'s mean ablation takes the batch mean at each position
    return {k: v / count[:, None].clamp(min=1) for k, v in sums.items()}, {k: v / len(ps) for k, v in attr.items()}


def curve(model: Model, train, test) -> dict:
    means, attr = means_and_relp(model, train)
    sizes = model.sizes()
    flat = torch.cat([attr[g].abs().float().cpu() for g in model.groups])
    order = torch.argsort(flat, descending=True)
    offsets = np.cumsum([0] + [sizes[g] for g in model.groups])
    total = int(offsets[-1])

    def keep_of(selected: Tensor) -> dict:
        mask = torch.zeros(total, dtype=torch.bool)
        mask[selected] = True
        return {g: mask[offsets[j]:offsets[j + 1]].to(DEVICE) for j, g in enumerate(model.groups)}

    full = metric(model, test)
    empty = metric(model, test, keep_of(order[:0]), means)
    ks = sorted({2 ** j for j in range(int(math.log2(total)) + 1)} | {total})
    faith, comp = [], []
    for k in ks:
        faith.append((metric(model, test, keep_of(order[:k]), means) - empty) / (full - empty))
        comp.append((metric(model, test, keep_of(order[k:]), means) - empty) / (full - empty))
    return {"nodes": total, "m_model": full, "m_empty": empty, "k": ks, "faithfulness": faith, "completeness": comp}


def main():
    out_path = sys.argv[1]
    n_train, n_test = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (300, 40)
    t0 = time.time()
    tok = Tokenizer.from_file(str(HERE / "t-9d2b8f02" / "tokenizer.json"))
    vpd = load_vpd(load_target(DEVICE), DEVICE)
    result = {"train_pairs": n_train, "test_pairs": n_test, "tasks": {}}
    for task in TASKS:
        train, test = pairs(tok, task, "train", n_train, 0), pairs(tok, task, "test", n_test, 1)
        result["tasks"][task] = {"train": len(train), "test": len(test)}
        for basis in ("neurons", "vpd"):
            result["tasks"][task][basis] = curve(Model(vpd, basis), train, test)
            json.dump(result, open(out_path, "w"), indent=1)
            r = result["tasks"][task][basis]
            print(f"[{time.time() - t0:.0f}s] {task} {basis}: m(M) {r['m_model']:.3f} m(none) {r['m_empty']:.3f} "
                  + " ".join(f"k={k}:{f:.2f}/{c:.2f}" for k, f, c in zip(r["k"], r["faithfulness"], r["completeness"])), flush=True)
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
