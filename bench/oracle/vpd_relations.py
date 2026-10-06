"""Second label pass for the vpd4l oracle (#2951): amplified continuations, measured neighbourhoods, edge
labels and per-token attributions, all from exact native edits (a removal is W - u v^T for one rank-one
subcomponent; every later computation is run again).

Parts (--parts, default all):
  continuations  greedy continuations of S tokens after each subcomponent's strongest peak with W(alpha)
                 = W + (alpha - 1) u v^T for alpha in ALPHAS, beside the clean one.
  downstream     at each of a subcomponent A's `--contexts` strongest contexts, at its peak p: the exact
                 change of every later subcomponent's activity at p when A is removed; kept: the `--k`
                 largest relative to each one's own peak |activity| (the components whose activity most
                 depends on A).
  upstream       at each of B's strongest contexts, at its peak p: proposals among the subcomponents
                 that can reach B's read directly, ranked by a first-order estimate of their effect on
                 a_B(p) (residual writers: a_A(p) v_B . (g ⊙ u_A) / r(p); for a down_proj reader the same
                 layer's c_fc subcomponents through the GELU's slope; for an o_proj reader the same
                 layer's v_proj subcomponents, |u_A . v_B| max_s |a_A(s)|), estimates used only to choose
                 what to measure; then each proposal's exact removal effect on a_B(p); kept: the `--k`
                 largest measured.
  edges          path patches (A's write taken out of B's read only, everything after run again) for each
                 B's two strongest measured residual or c_fc upstream neighbours and one proposal whose
                 measured removal effect is the smallest (a near-zero edge): the change of a_B(p) and of
                 the next-token distribution at p (KL(clean || patched), the 10 tokens rising and the 10
                 falling most in probability), at each reader's `--edge-contexts` strongest contexts.
  attribution    at `--positions` positions of each of `--attribution-contexts` contexts: X = the model's
                 most probable next token there; proposals among subcomponents writing to the residual
                 stream ranked by their direct effect on X's logit, a_A(t) u_A . (g_f ⊙ e_X) / r_f(t), and
                 the exact change of log p(X) at t when each is removed.

Inputs: a vpd_labels.py run (its contexts and peaks) and the subcomponents. Output under OUT:
relations_<part>.safetensors and relations.json. Global ids number the subcomponents site by site in
vpd_model.site_names() order.

  vpd_relations.py --labels DIR --uv UV --out DIR [--parts continuations,downstream,upstream,edges,attribution]
                   [--target DIR] [--data NPY] [--contexts 4] [--k 8] [--proposals 24] [--rows 256]
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
from vpd_labels import TOP, Model, device, greedy, load_uv  # noqa: E402

ALPHAS = (2.0, 4.0, 8.0)
S = 12
WRITERS = ("o_proj", "down_proj")
RESIDUAL_READERS = ("q_proj", "k_proj", "v_proj", "c_fc")


class Graph:
    """Sites in forward order with their global id offsets, and the clean pass's per-site inputs."""

    def __init__(self, model: Model, uv):
        self.model, self.uv = model, uv
        self.names = VM.site_names()
        self.offset, at = {}, 0
        for n in self.names:
            self.offset[n] = at
            at += uv[n][0].shape[0]
        self.total = at
        self.site_of = np.concatenate([np.full(uv[n][0].shape[0], i) for i, n in enumerate(self.names)])
        self.local = np.concatenate([np.arange(uv[n][0].shape[0]) for n in self.names])
        # Forward order: within a layer q, k, v read together, then o, c_fc, down.
        rank = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 1, "c_fc": 2, "down_proj": 3}
        self.order = {n: 4 * int(n.split(".")[1]) + rank[n.split(".")[-1]] for n in self.names}

    @torch.no_grad()
    def inputs(self, ids: torch.Tensor, edit=None) -> dict[str, torch.Tensor]:
        """Every site's input on `ids` (one forward), under an optional per-row edit."""
        t = self.model.t
        for n in self.names:
            t.site(n).cache_input = True
        x = t.wte[ids]
        layer = int(edit[0].split(".")[1]) if edit is not None else -1
        for i in range(t.n_layer):
            x, _ = self.model.layer(i, x, edit if i == layer else None)
        out = {}
        for n in self.names:
            st = t.site(n)
            out[n] = st.last_input
            st.last_input, st.cache_input = None, False
        out["final"] = VM.rms(x, t.ln_f, t.eps)
        return out

    def activities_at(self, inputs: dict, pos: torch.Tensor) -> torch.Tensor:
        """Every subcomponent's activity at each row's position: [R, total]."""
        at = torch.arange(len(pos), device=pos.device)
        return torch.cat([inputs[n][at, pos] @ self.uv[n][1] for n in self.names], 1)


def removal(g: Graph, gid: torch.Tensor) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """A per-row removal edit of global ids that all sit at one site."""
    sites = set(g.site_of[gid.cpu().numpy()].tolist())
    assert len(sites) == 1
    name = g.names[sites.pop()]
    local = torch.from_numpy(g.local[gid.cpu().numpy()]).to(gid.device)
    U, V = g.uv[name]
    return (name, V[:, local].T, U[local], torch.full((len(gid),), -1.0, device=gid.device))


def site_rows(labels: Path, name: str):
    tag = name.replace("h.", "").replace(".attn.", "_").replace(".mlp.", "_")
    return load_file(str(labels / f"site_{tag}.safetensors"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--uv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="continuations,downstream,upstream,edges,attribution")
    ap.add_argument("--target")
    ap.add_argument("--data")
    ap.add_argument("--contexts", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--proposals", type=int, default=24)
    ap.add_argument("--edge-contexts", type=int, default=1, help="edges: the strongest contexts of each reader patched")
    ap.add_argument("--attribution-contexts", type=int, default=1024)
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only the first N subcomponents of each site (a smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.target:
        VM.TARGET_DIR = Path(args.target)
    if args.data:
        VM.DATA = Path(args.data)
    dev = device()
    model = Model(dev)
    uv = load_uv(dev, Path(args.uv))
    g = Graph(model, uv)
    labels = Path(args.labels)
    tokens = load_file(str(labels / "contexts.safetensors"))["tokens"].long().to(dev)
    T = tokens.shape[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = args.parts.split(",")
    started = time.time()
    J = args.contexts
    # Per subcomponent: its contexts, peaks and peak |activity| (the relative scale of its changes).
    ctx_of, pos_of, peak_of = [], [], []
    for n in g.names:
        d = site_rows(labels, n)
        C = uv[n][0].shape[0]
        ctx_of.append(d["contexts"][:C, :J].long())
        pos_of.append(d["position"][:C, :J].long())
        peak_of.append(d["activity"][:C].float().abs().amax((1, 2)))
    ctx_of, pos_of = torch.cat(ctx_of).to(dev), torch.cat(pos_of).to(dev)
    peak_of = torch.cat(peak_of).to(dev).clamp(min=torch.finfo(torch.float32).tiny)
    chosen = torch.cat([g.offset[n] + torch.arange(min(args.limit, uv[n][0].shape[0]) if args.limit else uv[n][0].shape[0]) for n in g.names]).to(dev)
    meta = {"alphas": ALPHAS, "steps": S, "contexts": J, "k": args.k, "proposals": args.proposals, "site_offsets": g.offset, "total": g.total, "labels": str(labels)}

    if "continuations" in parts:
        res = {f"alpha_{a:g}": torch.full((g.total, S), -1, dtype=torch.int32) for a in ALPHAS}
        res["clean"] = torch.full((g.total, S), -1, dtype=torch.int32)
        for n in g.names:
            ids_ = chosen[torch.from_numpy(g.site_of[chosen.cpu().numpy()] == g.names.index(n)).to(dev)]
            for s0 in range(0, len(ids_), 64):
                gid = ids_[s0 : s0 + 64]
                prefixes = [tokens[int(ctx_of[i, 0]), : int(pos_of[i, 0]) + 1] for i in gid.tolist()]
                res["clean"][gid.cpu()] = greedy(model, prefixes, None, S).to(torch.int32).cpu()
                name, v, u, _ = removal(g, gid)
                for a in ALPHAS:
                    res[f"alpha_{a:g}"][gid.cpu()] = greedy(model, prefixes, (name, v, u, torch.full((len(gid),), a - 1.0, device=dev)), S).to(torch.int32).cpu()
        save_file(res, str(out / "relations_continuations.safetensors"))
        changed = {f"alpha_{a:g}": float((res[f"alpha_{a:g}"][chosen.cpu()] != res["clean"][chosen.cpu()]).any(-1).float().mean()) for a in ALPHAS}
        meta["continuations_changed"] = changed
        print(json.dumps({"part": "continuations", "changed": changed, "seconds": round(time.time() - started, 1)}), flush=True)

    clean_cache: dict[int, dict] = {}

    def clean_inputs(ctx: torch.Tensor) -> dict:
        return g.inputs(tokens[ctx])

    if "downstream" in parts:
        k = args.k
        ids_out = torch.full((g.total, J, k), -1, dtype=torch.int32)
        delta_out = torch.zeros(g.total, J, k)
        rel_out = torch.zeros(g.total, J, k)
        for n in g.names:
            site_ids = chosen[torch.from_numpy(g.site_of[chosen.cpu().numpy()] == g.names.index(n)).to(dev)]
            pairs = torch.cartesian_prod(site_ids, torch.arange(J, device=dev))
            later = torch.tensor([g.order[g.names[s]] > g.order[n] for s in g.site_of], device=dev)
            for s0 in range(0, len(pairs), args.rows):
                gid, j = pairs[s0 : s0 + args.rows, 0], pairs[s0 : s0 + args.rows, 1]
                ctx, pos = ctx_of[gid, j], pos_of[gid, j]
                clean = g.activities_at(clean_inputs(ctx), pos)
                edited = g.activities_at(g.inputs(tokens[ctx], removal(g, gid)), pos)
                delta = (edited - clean) * later
                rel = delta / peak_of
                top = rel.abs().topk(k, dim=1).indices
                ids_out[gid.cpu(), j.cpu()] = top.to(torch.int32).cpu()
                delta_out[gid.cpu(), j.cpu()] = delta.gather(1, top).cpu()
                rel_out[gid.cpu(), j.cpu()] = rel.gather(1, top).cpu()
        save_file({"ids": ids_out, "delta": delta_out, "relative": rel_out}, str(out / "relations_downstream.safetensors"))
        print(json.dumps({"part": "downstream", "median_top_relative_change": float(rel_out[chosen.cpu(), :, 0].abs().median()), "seconds": round(time.time() - started, 1)}), flush=True)

    if "upstream" in parts or "edges" in parts:
        k, P = args.k, args.proposals
        up_ids = torch.full((g.total, J, P), -1, dtype=torch.int32)
        up_delta = torch.zeros(g.total, J, P)
        gains = [model.t.norms[i] for i in range(2 * model.t.n_layer)]
        for n in g.names:
            layer, kind = int(n.split(".")[1]), n.split(".")[-1]
            site_ids = chosen[torch.from_numpy(g.site_of[chosen.cpu().numpy()] == g.names.index(n)).to(dev)]
            pairs = torch.cartesian_prod(site_ids, torch.arange(J, device=dev))
            for s0 in range(0, len(pairs), max(1, args.rows // P)):
                gid, j = pairs[s0 : s0 + max(1, args.rows // P), 0], pairs[s0 : s0 + max(1, args.rows // P), 1]
                ctx, pos = ctx_of[gid, j], pos_of[gid, j]
                inp = clean_inputs(ctx)
                acts = g.activities_at(inp, pos)  # [R, total]
                at = torch.arange(len(gid), device=dev)
                local_b = torch.from_numpy(g.local[gid.cpu().numpy()]).to(dev)
                vB = uv[n][1][:, local_b].T  # [R, d_in]
                score = torch.zeros_like(acts)
                if kind in RESIDUAL_READERS:
                    read = 2 * layer + (1 if kind == "c_fc" else 0)
                    x = residual_before(model, tokens[ctx], read)[at, pos]
                    inv = torch.rsqrt(x.pow(2).mean(-1) + model.t.eps)
                    for w in g.names:
                        wl, wk = int(w.split(".")[1]), w.split(".")[-1]
                        if wk not in WRITERS or 2 * wl + (1 if wk == "o_proj" else 2) > read:
                            continue
                        lo = g.offset[w]
                        coupling = (vB * gains[read]) @ uv[w][0].T  # [R, C_w]
                        score[:, lo : lo + coupling.shape[1]] = (acts[:, lo : lo + coupling.shape[1]] * coupling * inv[:, None]).abs()
                elif kind == "down_proj":
                    w = f"h.{layer}.mlp.c_fc"
                    lo = g.offset[w]
                    h = VM.rms(residual_before(model, tokens[ctx], 2 * layer + 1), model.t.norms[2 * layer + 1], model.t.eps)[at, pos]
                    zpre = model.t.site(w)(h)  # [R, 3072]
                    slope = gelu_slope(zpre)
                    coupling = (vB * slope) @ uv[w][0].T  # [R, C_w]
                    score[:, lo : lo + coupling.shape[1]] = (acts[:, lo : lo + coupling.shape[1]] * coupling).abs()
                elif kind == "o_proj":
                    w = f"h.{layer}.attn.v_proj"
                    lo = g.offset[w]
                    coupling = vB @ uv[w][0].T
                    peak_w = inp[w] @ uv[w][1]  # [R, T, C_w]
                    score[:, lo : lo + coupling.shape[1]] = coupling.abs() * peak_w.abs().amax(1)
                score[at, gid] = 0
                prop = score.topk(P, dim=1).indices  # [R, P]
                # Exact removal effect of each proposal on B's activity at its peak.
                flat_ctx = ctx.repeat_interleave(P)
                flat_pos = pos.repeat_interleave(P)
                flat_prop = prop.reshape(-1)
                delta = torch.empty(len(flat_prop), device=dev)
                for s1 in range(0, len(flat_prop), args.rows):
                    sl = slice(s1, s1 + args.rows)
                    sites = g.site_of[flat_prop[sl].cpu().numpy()]
                    for site in np.unique(sites):
                        m = torch.from_numpy(sites == site).to(dev)
                        idx = torch.arange(s1, min(s1 + args.rows, len(flat_prop)), device=dev)[m]
                        inputs_e = g.inputs(tokens[flat_ctx[idx]], removal(g, flat_prop[idx]))
                        x_b = inputs_e[n][torch.arange(len(idx), device=dev), flat_pos[idx]]
                        a_new = (x_b * vB.repeat_interleave(P, 0)[idx]).sum(-1)
                        delta[idx] = a_new - acts[torch.arange(len(gid), device=dev), gid].repeat_interleave(P)[idx]
                order = delta.reshape(len(gid), P).abs().argsort(dim=1, descending=True)
                up_ids[gid.cpu(), j.cpu()] = prop.gather(1, order).to(torch.int32).cpu()
                up_delta[gid.cpu(), j.cpu()] = delta.reshape(len(gid), P).gather(1, order).cpu()
        save_file({"ids": up_ids, "delta": up_delta}, str(out / "relations_upstream.safetensors"))
        print(json.dumps({"part": "upstream", "median_strongest_relative_change": float((up_delta[chosen.cpu(), :, 0].abs() / peak_of[chosen].cpu()[:, None]).median()),
                          "seconds": round(time.time() - started, 1)}), flush=True)

    if "edges" in parts:
        edges(model, g, uv, tokens, ctx_of, pos_of, chosen, up_ids, up_delta, out, args)
        print(json.dumps({"part": "edges", "seconds": round(time.time() - started, 1)}), flush=True)

    if "attribution" in parts:
        attribution(model, g, uv, tokens, out, args)
        print(json.dumps({"part": "attribution", "seconds": round(time.time() - started, 1)}), flush=True)

    (out / "relations.json").write_text(json.dumps(meta))
    print(json.dumps({"total_seconds": round(time.time() - started, 1)}), flush=True)


@torch.no_grad()
def cut_run(model: Model, name: str, entering: torch.Tensor, delta: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The final normed stream when site `name`'s output gains delta(t) u at every position (per row),
    its layer and every later one run from the stream entering it."""
    t = model.t
    st = t.site(name)
    original = st.forward

    def forward(x):
        return original(x) + delta[..., None] * u[:, None, :]

    st.forward = forward
    try:
        x = entering
        for i in range(int(name.split(".")[1]), t.n_layer):
            x, _ = model.layer(i, x)
    finally:
        st.forward = original
    return VM.rms(x, t.ln_f, t.eps)


@torch.no_grad()
def edges(model: Model, g: Graph, uv, tokens, ctx_of, pos_of, chosen, up_ids, up_delta, out: Path, args):
    """Path patches of each reader's two strongest measured upstream neighbours that write into what it
    reads (the residual stream, or for a down_proj reader its layer's c_fc subcomponents) and of its
    weakest measured proposal of that kind (module note)."""
    dev = tokens.device
    t = model.t
    J = min(args.edge_contexts, ctx_of.shape[1])
    E = 3
    keys = ("activity", "kl", "next")
    res = {k: torch.full((g.total, J, E), float("nan")) for k in keys}
    res["ids"] = torch.full((g.total, J, E), -1, dtype=torch.int32)
    res["strong"] = torch.zeros(g.total, J, E, dtype=torch.int8)
    res["up_ids"] = torch.full((g.total, J, E, TOP), -1, dtype=torch.int32)
    res["down_ids"] = torch.full((g.total, J, E, TOP), -1, dtype=torch.int32)
    res["up_dp"] = torch.zeros(g.total, J, E, TOP)
    res["down_dp"] = torch.zeros(g.total, J, E, TOP)
    for n in g.names:
        layer, kind = int(n.split(".")[1]), n.split(".")[-1]
        if kind not in RESIDUAL_READERS and kind != "down_proj":
            continue
        site_ids = chosen[torch.from_numpy(g.site_of[chosen.cpu().numpy()] == g.names.index(n)).to(dev)]
        for gid_t in site_ids.tolist():
            for j in range(J):
                cands = up_ids[gid_t, j].tolist()
                deltas = up_delta[gid_t, j].tolist()
                eligible = [(a, d) for a, d in zip(cands, deltas) if a >= 0 and writes_into(g, a, n)]
                if len(eligible) < 2:
                    continue
                picks = [(eligible[0][0], 1), (eligible[1][0], 1), (min(eligible, key=lambda x: abs(x[1]))[0], 0)]
                ctx, p = int(ctx_of[gid_t, j]), int(pos_of[gid_t, j])
                ids = tokens[ctx : ctx + 1].repeat(E, 1)
                inp = g.inputs(ids[:1])
                b = int(g.local[gid_t])
                vB, uB = uv[n][1][:, b], uv[n][0][b]
                deltas_t = []
                for a_gid, _ in picks:
                    an = g.names[int(g.site_of[a_gid])]
                    a = int(g.local[a_gid])
                    a_act = inp[an][0] @ uv[an][1][:, a]  # [T]
                    write = a_act[:, None] * uv[an][0][a][None, :]
                    if kind == "down_proj":
                        h = VM.rms(residual_before(model, ids[:1], 2 * layer + 1), t.norms[2 * layer + 1], t.eps)[0]
                        z = t.site(f"h.{layer}.mlp.c_fc")(h)
                        clean_b, cut_b = VM.gelu_tanh(z) @ vB, VM.gelu_tanh(z - write) @ vB
                    else:
                        read = 2 * layer + (1 if kind == "c_fc" else 0)
                        x = residual_before(model, ids[:1], read)[0]
                        clean_b = VM.rms(x, t.norms[read], t.eps) @ vB
                        cut_b = VM.rms(x - write, t.norms[read], t.eps) @ vB
                    deltas_t.append(cut_b - clean_b)
                delta = torch.stack(deltas_t)  # [E, T]
                entering = residual_before(model, ids, 2 * layer)
                h = cut_run(model, n, entering, delta, uB[None, :].repeat(E, 1))
                clean_final = inp["final"][0, p]
                lc, le = model.log_probs(clean_final), model.log_probs(h[:, p])
                dl = le - lc[None]
                dp = le.exp() - lc.exp()[None]
                res["ids"][gid_t, j] = torch.tensor([a for a, _ in picks], dtype=torch.int32)
                res["strong"][gid_t, j] = torch.tensor([s_ for _, s_ in picks], dtype=torch.int8)
                res["activity"][gid_t, j] = delta[:, p].cpu()
                res["kl"][gid_t, j] = (lc.exp()[None] * -dl).sum(-1).float().cpu()
                nt = int(tokens[ctx, min(p + 1, tokens.shape[1] - 1)])
                res["next"][gid_t, j] = dl[:, nt].float().cpu() if p + 1 < tokens.shape[1] else float("nan")
                ui, di = dp.topk(TOP, -1).indices, (-dp).topk(TOP, -1).indices
                res["up_ids"][gid_t, j], res["down_ids"][gid_t, j] = ui.to(torch.int32).cpu(), di.to(torch.int32).cpu()
                res["up_dp"][gid_t, j], res["down_dp"][gid_t, j] = dp.gather(-1, ui).float().cpu(), dp.gather(-1, di).float().cpu()
    save_file(res, str(out / "relations_edges.safetensors"))


def writes_into(g: Graph, a_gid: int, reader: str) -> bool:
    """Whether subcomponent a writes into what `reader` reads: a residual writer before a residual
    reader, or a c_fc subcomponent of a down_proj reader's layer."""
    an = g.names[int(g.site_of[a_gid])]
    al, ak = int(an.split(".")[1]), an.split(".")[-1]
    rl, rk = int(reader.split(".")[1]), reader.split(".")[-1]
    if rk == "down_proj":
        return ak == "c_fc" and al == rl
    read = 2 * rl + (1 if rk == "c_fc" else 0)
    return ak in WRITERS and 2 * al + (1 if ak == "o_proj" else 2) <= read


@torch.no_grad()
def attribution(model: Model, g: Graph, uv, tokens, out: Path, args):
    """At sampled positions: the model's most probable next token X, the residual writers ranked by
    their direct effect on X's logit, and each one's exact removal effect on log p(X) (module note)."""
    dev = tokens.device
    t = model.t
    rng = np.random.default_rng(args.seed)
    N, Q, P = args.attribution_contexts, args.positions, args.proposals * 2
    writers = [n for n in g.names if n.split(".")[-1] in WRITERS]
    rows = {"context": [], "position": [], "token": [], "ids": [], "direct": [], "delta": []}
    for c in range(min(N, tokens.shape[0])):
        ids = tokens[c : c + 1]
        inp = g.inputs(ids)
        final = inp["final"][0]
        positions = rng.choice(np.arange(16, tokens.shape[1] - 1), size=Q, replace=False)
        for p in positions.tolist():
            lc = model.log_probs(final[p])
            X = int(lc.argmax())
            e = t.wte[X]
            # Direct effect on X's logit of each writer's write at p through the final norm (its gain
            # and the stream's RMS at p): a_A(p) u_A . (g_f ⊙ e_X) / r_f(p).
            scores, gids = [], []
            stream = final_stream(model, ids)[0, p]
            inv = torch.rsqrt(stream.pow(2).mean() + t.eps)
            for w in writers:
                a = inp[w][0, p] @ uv[w][1]  # [C]
                direct = a * (uv[w][0] @ (t.ln_f * e)) * inv
                scores.append(direct)
                gids.append(g.offset[w] + torch.arange(len(a), device=dev))
            scores, gids = torch.cat(scores), torch.cat(gids)
            top = scores.abs().topk(P).indices
            cand, direct = gids[top], scores[top]
            delta = torch.empty(P, device=dev)
            for site in np.unique(g.site_of[cand.cpu().numpy()]):
                m = torch.from_numpy(g.site_of[cand.cpu().numpy()] == site).to(dev)
                idx = torch.arange(P, device=dev)[m]
                edited = g.inputs(ids.repeat(len(idx), 1), removal(g, cand[idx]))["final"][:, p]
                delta[idx] = (model.log_probs(edited)[:, X] - lc[X]).to(delta.dtype)  # log_probs normalizes in float64 on CUDA
            rows["context"].append(c)
            rows["position"].append(p)
            rows["token"].append(X)
            rows["ids"].append(cand.cpu())
            rows["direct"].append(direct.float().cpu())
            rows["delta"].append(delta.float().cpu())
    save_file({"context": torch.tensor(rows["context"]), "position": torch.tensor(rows["position"]), "token": torch.tensor(rows["token"]),
               "ids": torch.stack(rows["ids"]).to(torch.int32), "direct": torch.stack(rows["direct"]), "delta": torch.stack(rows["delta"])},
              str(out / "relations_attribution.safetensors"))


@torch.no_grad()
def final_stream(model: Model, ids: torch.Tensor) -> torch.Tensor:
    """The residual stream after the last layer, before the final norm."""
    t = model.t
    x = t.wte[ids]
    for i in range(t.n_layer):
        x, _ = model.layer(i, x)
    return x


def gelu_slope(z: torch.Tensor) -> torch.Tensor:
    """d/dz of the tanh GELU."""
    c = (2.0 / np.pi) ** 0.5
    inner = c * (z + 0.044715 * z.pow(3))
    t = torch.tanh(inner)
    return 0.5 * (1 + t) + 0.5 * z * (1 - t.pow(2)) * c * (1 + 3 * 0.044715 * z.pow(2))


@torch.no_grad()
def residual_before(model: Model, ids: torch.Tensor, read: int) -> torch.Tensor:
    """The residual stream at read position `read` (2l before layer l's attention, 2l + 1 before its MLP)."""
    t = model.t
    x = t.wte[ids]
    for i in range(read // 2):
        x, _ = model.layer(i, x)
    if read % 2 == 0:
        return x
    l = read // 2
    B, L, _ = x.shape
    h = VM.rms(x, t.norms[2 * l], t.eps)
    s = lambda k: t.site(f"h.{l}.attn.{k}")  # noqa: E731
    q = s("q_proj")(h).view(B, L, t.n_head, t.hd).transpose(1, 2)
    k = s("k_proj")(h).view(B, L, t.n_head, t.hd).transpose(1, 2)
    v = s("v_proj")(h).view(B, L, t.n_head, t.hd).transpose(1, 2)
    q, k = t._rope(q, L), t._rope(k, L)
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return x + s("o_proj")(y.transpose(1, 2).reshape(B, L, -1))


if __name__ == "__main__":
    main()
