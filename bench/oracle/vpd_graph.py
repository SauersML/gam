"""The graph of VPD's vpd4l decomposition for the oracle (#2951): edges between subcomponents through the
residual stream, their exact strengths, every subcomponent's neighbourhood, and measured edge labels by
path patching.

Writers and readers. A subcomponent of an o_proj or down_proj site writes a_A(t) u_A into the residual
stream (a_A = v_A . x_A its activity); one of a q, k, v_proj or c_fc site reads the stream through its
layer's norm, a_B = v_B . (g ⊙ x / r(x)), g the norm's gain and r(x) = sqrt(mean(x^2) + eps). An o_proj of
layer l writes before its layer's MLP reads; a down_proj of layer l before layer l + 1 reads.

Edges. Writer A's write enters reader B's activity as c_AB(t) = a_A(t) w_AB / r_B(t), w_AB = v_B . (g_B ⊙
u_A), exactly (as readout's Library::contributions: a read is linear in the stream before its norm, at the
stream's actual RMS). The edge's strength is the mean over tokens of |c_AB(t)| = |w_AB| mean_t(|a_A(t)| /
r_B(t)), over the first `--stats` contexts of the label pool. Each reader keeps its `--k` strongest
upstream writers, each writer its `--k` strongest downstream readers (attention's value path and the MLP's
hidden layer are not residual edges and are not listed).

Path patches (measured edges, as readout's Library::path_patch). For reader B, at each of its first
`--contexts` top contexts (vpd_labels.py's) and for each of its `--patch` strongest upstream writers A:
the stream B reads becomes x - a_A u_A (A's clean write taken out of B's read only), B's activity is
recomputed through its norm on that stream (a'_B), and B's site output gains (a'_B(t) - a_B(t)) u_B at
every position, everything after it run again. Recorded at B's peak position p of that context: the
change of B's activity there, and of the next-token distribution: KL(clean || cut), the next token's
log-probability change, and the 10 tokens rising and the 10 falling most in probability.

Output under OUT (beside a label run): graph.safetensors and graph.json. Global ids number the
subcomponents site by site in vpd_model.site_names() order.

  vpd_graph.py --labels LABEL_DIR --uv UV --out DIR [--k 8] [--patch 4] [--contexts 8] [--stats 256]
               [--target DIR] [--data NPY] [--rows 256]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vpd_2951"))
sys.path.insert(0, str(HERE))
import vpd_model as VM  # noqa: E402
from vpd_labels import TOP, Model, device, load_uv  # noqa: E402

WRITE_KINDS = ("o_proj", "down_proj")
READ_KINDS = ("q_proj", "k_proj", "v_proj", "c_fc")


def read_site(layer: int, kind: str) -> int:
    """The stream position a reader reads: 2l before layer l's attention, 2l + 1 before its MLP."""
    return 2 * layer + (1 if kind == "c_fc" else 0)


def written_before(layer: int, kind: str) -> int:
    """The first stream position that holds a writer's write."""
    return 2 * layer + (1 if kind == "o_proj" else 2)


class Ids:
    """Global subcomponent ids: sites in vpd_model.site_names() order, then index within the site."""

    def __init__(self, uv):
        self.names = VM.site_names()
        self.offset, at = {}, 0
        for n in self.names:
            self.offset[n] = at
            at += uv[n][0].shape[0]
        self.total = at

    def of(self, name: str, c) -> torch.Tensor | int:
        return self.offset[name] + c


@torch.no_grad()
def streams(model: Model, ids: torch.Tensor):
    """Per read position (2l, 2l + 1) the stream before its norm; per writer site its input; and the
    final normed stream."""
    t = model.t
    for n in VM.site_names():
        if n.split(".")[-1] in WRITE_KINDS:
            t.site(n).cache_input = True
    x = t.wte[ids]
    positions = []
    B, L, _ = x.shape
    for i in range(t.n_layer):
        positions.append(x)
        x, _ = model.layer(i, x)
    final = VM.rms(x, t.ln_f, t.eps)
    # The stream before each MLP read: after the layer's attention (recomputed from the entering stream).
    mids = []
    for i in range(t.n_layer):
        h = VM.rms(positions[i], t.norms[2 * i], t.eps)
        s = lambda k: t.site(f"h.{i}.attn.{k}")  # noqa: E731
        q = s("q_proj")(h).view(B, L, t.n_head, t.hd).transpose(1, 2)
        k = s("k_proj")(h).view(B, L, t.n_head, t.hd).transpose(1, 2)
        v = s("v_proj")(h).view(B, L, t.n_head, t.hd).transpose(1, 2)
        q, k = t._rope(q, L), t._rope(k, L)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        mids.append(positions[i] + s("o_proj")(y.transpose(1, 2).reshape(B, L, -1)))
    reads = [p for i in range(t.n_layer) for p in (positions[i], mids[i])]
    inputs = {}
    for n in VM.site_names():
        st = t.site(n)
        if st.cache_input:
            inputs[n] = st.last_input
            st.last_input, st.cache_input = None, False
    return reads, inputs, final


def rms_inverse(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(-1) + eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--uv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--contexts", type=int, default=8)
    ap.add_argument("--stats", type=int, default=256)
    ap.add_argument("--target")
    ap.add_argument("--data")
    ap.add_argument("--rows", type=int, default=256)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.target:
        VM.TARGET_DIR = Path(args.target)
    if args.data:
        VM.DATA = Path(args.data)
    dev = device()
    model = Model(dev)
    t = model.t
    uv = load_uv(dev, Path(args.uv))
    ids_of = Ids(uv)
    labels = Path(args.labels)
    tokens = load_file(str(labels / "contexts.safetensors"))["tokens"].long().to(dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    writers = [n for n in VM.site_names() if n.split(".")[-1] in WRITE_KINDS]
    readers = [n for n in VM.site_names() if n.split(".")[-1] in READ_KINDS]
    gains = [t.norms[i] for i in range(2 * t.n_layer)]

    # Activity statistics: m[A][s] = mean_t |a_A(t)| / r_s(t), for every read position s after A writes.
    m = {n: torch.zeros(uv[n][0].shape[0], 2 * t.n_layer, device=dev) for n in writers}
    count = 0
    for s0 in range(0, args.stats, 32):
        batch = tokens[s0 : min(args.stats, s0 + 32)]
        reads, inputs, _ = streams(model, batch)
        inv = [rms_inverse(r, t.eps).reshape(-1) for r in reads]  # per read position, [tokens]
        for n in writers:
            a = (inputs[n] @ uv[n][1]).abs().reshape(-1, uv[n][1].shape[1])  # [tokens, C]
            for s in range(2 * t.n_layer):
                m[n][:, s] += inv[s] @ a
        count += batch.shape[0] * batch.shape[1]
    for n in writers:
        m[n] /= count

    # Couplings and neighbourhoods.
    up_ids, up_strength, up_coupling = {}, {}, {}
    down_best: dict[str, list] = {n: [] for n in writers}
    for rn in readers:
        layer, kind = int(rn.split(".")[1]), rn.split(".")[-1]
        s = read_site(layer, kind)
        VB = uv[rn][1] * gains[s][:, None]  # [768, C_B], the read through the gain
        cands, strengths, couplings = [], [], []
        for wn in writers:
            wl, wk = int(wn.split(".")[1]), wn.split(".")[-1]
            if written_before(wl, wk) > s:
                continue
            w = uv[wn][0] @ VB  # [C_A, C_B]
            e = w.abs() * m[wn][:, s][:, None]
            cands.append(torch.full((uv[wn][0].shape[0],), ids_of.offset[wn], device=dev) + torch.arange(uv[wn][0].shape[0], device=dev))
            strengths.append(e)
            couplings.append(w)
            # Downstream lists of these writers: keep each writer's best readers of this site.
            top = e.topk(min(args.k, e.shape[1]), dim=1)
            down_best[wn].append((top.values, top.indices + ids_of.offset[rn]))
        if not cands:
            continue
        allc, alls, allw = torch.cat(cands), torch.cat(strengths), torch.cat(couplings)  # [sum C_A], [sum C_A, C_B]
        top = alls.topk(args.k, dim=0)
        up_ids[rn] = allc[top.indices].T  # [C_B, k]
        up_strength[rn] = top.values.T
        up_coupling[rn] = allw.gather(0, top.indices).T
    down_ids, down_strength = {}, {}
    for wn, parts in down_best.items():
        if not parts:
            continue
        vals = torch.cat([p[0] for p in parts], 1)
        idx = torch.cat([p[1] for p in parts], 1)
        top = vals.topk(min(args.k, vals.shape[1]), dim=1)
        down_ids[wn], down_strength[wn] = idx.gather(1, top.indices), top.values
    graph_seconds = time.time() - started
    print(json.dumps({"neighbourhoods_seconds": round(graph_seconds, 1)}), flush=True)

    # Path patches of each reader's strongest upstream edges on its top contexts.
    site_of_id = []
    for n in VM.site_names():
        site_of_id += [n] * uv[n][0].shape[0]
    rec = {}
    for rn in readers:
        if rn not in up_ids:
            continue
        layer, kind = int(rn.split(".")[1]), rn.split(".")[-1]
        s = read_site(layer, kind)
        tag = rn.replace("h.", "").replace(".attn.", "_").replace(".mlp.", "_")
        if not (labels / f"site_{tag}.safetensors").exists():
            continue
        lab = load_file(str(labels / f"site_{tag}.safetensors"))
        C = lab["contexts"].shape[0]
        J, P = args.contexts, args.patch
        d_act = torch.empty(C, J, P, device=dev)
        kl = torch.empty(C, J, P, dtype=model.wide, device=dev)
        nxt = torch.empty(C, J, P, dtype=model.wide, device=dev)
        up_t = torch.empty(C, J, P, TOP, dtype=torch.int64, device=dev)
        dn_t = torch.empty(C, J, P, TOP, dtype=torch.int64, device=dev)
        up_p, dn_p = torch.empty(C, J, P, TOP, dtype=model.wide, device=dev), torch.empty(C, J, P, TOP, dtype=model.wide, device=dev)
        contexts = lab["contexts"][:, :J].long().to(dev)
        peaks = lab["position"][:, :J].long().to(dev)
        triples = torch.cartesian_prod(torch.arange(C, device=dev), torch.arange(J, device=dev), torch.arange(P, device=dev))
        for s0 in range(0, len(triples), args.rows):
            cs, js, ps = triples[s0 : s0 + args.rows].unbind(1)
            R = len(cs)
            ctx = contexts[cs, js]
            reads, inputs, final = streams(model, tokens[ctx])
            x = reads[s]  # [R, T, 768]
            a_ids = up_ids[rn][cs, ps]
            # The writers' clean writes on each row's context.
            write = torch.empty_like(x)
            for wn in {site_of_id[int(i)] for i in a_ids.tolist()}:
                sel = torch.tensor([site_of_id[int(i)] == wn for i in a_ids.tolist()], device=dev)
                local = a_ids[sel] - ids_of.offset[wn]
                a_A = torch.einsum("rtd,dr->rt", inputs[wn][sel], uv[wn][1][:, local])
                write[sel] = a_A[..., None] * uv[wn][0][local][:, None, :]
            vB = uv[rn][1][:, cs].T  # [R, 768]
            g = gains[s]
            a_clean = torch.einsum("rtd,rd->rt", VM.rms(x, g, t.eps), vB)
            a_cut = torch.einsum("rtd,rd->rt", VM.rms(x - write, g, t.eps), vB)
            delta = a_cut - a_clean
            pos = peaks[cs, js]
            at = torch.arange(R, device=dev)
            d_act[cs, js, ps] = delta[at, pos]
            # B's site output gains delta(t) u_B: an edit with a per-position coefficient.
            entering = reads[2 * layer]
            uB = uv[rn][0][cs]
            h = cut_run(model, layer, rn, entering, delta, uB)
            lp_e = model.log_probs(h[at, pos])
            lp_c = model.log_probs(final[at, pos])
            dl = lp_e - lp_c
            dp = lp_e.exp() - lp_c.exp()
            kl[cs, js, ps] = (lp_c.exp() * -dl).sum(-1)
            ui, di = dp.topk(TOP, dim=-1).indices, (-dp).topk(TOP, dim=-1).indices
            up_t[cs, js, ps], dn_t[cs, js, ps] = ui, di
            up_p[cs, js, ps], dn_p[cs, js, ps] = dp.gather(-1, ui), dp.gather(-1, di)
            T = tokens.shape[1]
            nt = tokens[ctx, (pos + 1).clamp(max=T - 1)]
            nd = dl.gather(-1, nt[:, None])[:, 0]
            nxt[cs, js, ps] = torch.where(pos + 1 < T, nd, torch.full_like(nd, float("nan")))
        f32 = lambda z: z.to(torch.float32).cpu()  # noqa: E731
        rec.update({f"{rn}.patch_activity": f32(d_act), f"{rn}.patch_kl": f32(kl), f"{rn}.patch_next": f32(nxt),
                    f"{rn}.patch_up_ids": up_t.to(torch.int32).cpu(), f"{rn}.patch_up_dp": f32(up_p),
                    f"{rn}.patch_down_ids": dn_t.to(torch.int32).cpu(), f"{rn}.patch_down_dp": f32(dn_p)})
        print(json.dumps({"reader_site": rn, "subcomponents": C, "seconds": round(time.time() - started, 1),
                          "median_kl_cut": float(kl.median()), "median_abs_activity_change": float(d_act.abs().median())}), flush=True)
    for rn in up_ids:
        rec[f"{rn}.up_ids"], rec[f"{rn}.up_strength"], rec[f"{rn}.up_coupling"] = up_ids[rn].to(torch.int32).cpu(), up_strength[rn].cpu(), up_coupling[rn].cpu()
    for wn in down_ids:
        rec[f"{wn}.down_ids"], rec[f"{wn}.down_strength"] = down_ids[wn].to(torch.int32).cpu(), down_strength[wn].cpu()
    save_file({k: v.contiguous() for k, v in rec.items()}, str(out / "graph.safetensors"))
    (out / "graph.json").write_text(json.dumps({"k": args.k, "patch": args.patch, "contexts": args.contexts, "stats_contexts": args.stats,
                                                "site_offsets": ids_of.offset, "total": ids_of.total, "labels": str(labels), "seconds": time.time() - started}))
    print(json.dumps({"total_seconds": round(time.time() - started, 1)}), flush=True)


@torch.no_grad()
def cut_run(model: Model, layer: int, name: str, entering: torch.Tensor, delta: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The final normed stream when site `name`'s output gains delta(t) u at every position (each row
    its own delta and u), layer `layer` and every later one run from the stream entering it."""
    t = model.t
    st = t.site(name)
    original = st.forward

    def forward(x):
        return original(x) + delta[..., None] * u[:, None, :]

    st.forward = forward
    try:
        x = entering
        for i in range(layer, t.n_layer):
            x, _ = model.layer(i, x)
    finally:
        st.forward = original
    return VM.rms(x, t.ln_f, t.eps)


if __name__ == "__main__":
    main()
