"""Computational graphs of vpd4l's predictions (#2951 graph oracle), run natively in torch.

The model runs with every weight matrix replaced by VPD's subcomponents and remainder,
    y = ((x V) * m) U + m_delta x (W - V U)^T,
masks m in [0, 1] per position. An answer is a graph: nodes are subcomponents at positions; edges are connections the
model has (an attention or MLP output into a later query, key, value or MLP input at its position, or into the
prediction; a value into the same layer's attention output at that or a later position; an MLP input into the same
MLP's output). Running a graph (Native.run): its nodes at mask 1; every other subcomponent at an ablation mask u; a node
receives its declared parents' outputs in full and every other output scaled by that output's u; the token embedding,
which VPD does not decompose, in full. With u = 1 everywhere this is the model.

A graph is correct within eps when the KL in bits of the model's next-token distribution at the targets from the
graph's stays at most eps under both ablations of what it leaves out (Native.score, kl_bits the larger):
  deletion     u = 0;
  random       u drawn uniformly per position and subcomponent, the mean over RANDOM draws (VPD's stochastic test).
The KL is of the whole distribution, so a graph that makes the prediction more confident than the model fails too.
An adversary choosing u for one text breaks VPD's own answer (91.7 bits on text9000, 101.1 for an empty graph), and
one shared across 32 texts still breaks it by 14 to 17 bits at their predictions: as a constraint it would make
every graph nearly the whole model (VPD's appendix A.3.4 finds per-input adversaries too strict for the same reason).
Among correct graphs, fewer nodes plus edges is better (score.order).

teach() finds a graph for one prediction and precision eps: node strengths g in (0, 1) over every subcomponent at every
position that can reach the target, minimizing their mean subject to the KL tests at eps / 2 with VPD's masks
m = g + (1 - g) r (r uniform or 0), the multiplier set by the constraint (dual ascent); the nodes with g > 1/2; then
edge strengths over every connection the model has between them, the same way at eps; the edges with strength > 1/2;
then the exact tests, adding back the strongest dropped edges until the graph is correct.

  native.py teach --split train --n N [--offset K --stride S] [--eps 0.25 1]   -> texts/graphs[_heldout]/<id>.<eps>.py, .json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "vpd_2951"))

import mech  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts"
RANDOM = 8  # random draws per score
CANDIDATES = 32  # replacement tokens the teacher tries per source position for an interchange claim
TEACH_STEPS, EDGE_STEPS, TEACH_LR = 2000, 600, 0.05  # the teacher's optimization (Adam on the strengths' logits)
LN2 = math.log(2)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def site_name(layer: int, kind: str) -> str:
    return f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"


def _forward_cached(self, x):
    """Site._forward with the remainder W - V U computed once."""
    if self.mask is None:
        return x @ self.W.T
    out = ((x @ self.V) * self.mask) @ self.U
    if self.delta_mask is not None:
        out = out + self.delta_mask.unsqueeze(-1) * (x @ self.delta_T)
    return out


UV = Path.home() / "mpd-data/oracle/vpd/uv.safetensors"  # VPD's subcomponents alone (published for pods)


@lru_cache(None)
def model(dev: str):
    """vpd4l with VPD's decomposition installed, each matrix's remainder cached. Without VPD's checkpoint (a pod), the
    subcomponents come from UV, identical to the checkpoint's, and there is no causal-importance network."""
    import vpd_model

    target = vpd_model.load_target(dev)
    if vpd_model.VPD_PTH.exists():
        vpd = vpd_model.load_vpd(target, dev)
    else:
        from safetensors.torch import load_file

        uv = load_file(str(UV))
        vpd = vpd_model.VPD(target, None, {n: (uv[f"{n}.U"].float().to(dev), uv[f"{n}.V"].float().to(dev)) for n in vpd_model.site_names()})
    for n in vpd.names:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(_forward_cached, st)
    return target, vpd


class Graph:
    """One answer's computational graph on one text. nodes: (weight matrix, position, subcomponent) triples; parents:
    reader node -> the writer nodes whose outputs it reads; out: the writer nodes the prediction at the targets reads.
    complete=True instead means every connection the model has between the graph's nodes (and from each of its
    residual writers at a target to the prediction): the graph claims nodes but no structure; masks holds a complete
    graph's nodes as matrix -> bool [T, C] in place of `nodes`."""

    def __init__(self, nodes=(), parents=None, out=None, complete: bool = False, masks: dict | None = None):
        self.masks = masks
        self.nodes = set(nodes)
        self.parents = {r: list(w) for r, w in (parents or {}).items()}
        self.out = list(out or [])
        self.complete = complete or masks is not None

    def count(self) -> int:
        return int(sum(int(m.sum()) for m in self.masks.values())) if self.masks is not None else len(self.nodes)

    def node_set(self) -> set:
        if self.masks is None:
            return self.nodes
        return {(n, t, c) for n, m in self.masks.items() for t, c in m.nonzero().tolist()}

    def edges(self) -> int:
        return sum(len(w) for w in self.parents.values()) + len(self.out)

    def size(self) -> int:
        return self.count() + self.edges()


def _layer_kind(nd) -> tuple[int, str]:
    return int(nd[0].split(".")[1]), nd[0].split(".")[-1]


def connects(writer: tuple, reader: tuple | None, targets: list[int]) -> bool:
    """Whether the model connects a writer node to a reader node (None: the prediction at the targets); nodes are
    (matrix name, position, index). mech.connects holds the rule."""
    wl, wk = _layer_kind(writer)
    if reader is None:
        return mech.connects(wl, wk, writer[1], None, None, None, targets)
    rl, rk = _layer_kind(reader)
    return mech.connects(wl, wk, writer[1], rl, rk, reader[1], targets)


def all_edges(nodes, targets: list[int]) -> tuple[dict, list]:
    """Every connection the model has between `nodes`: (parents, out)."""
    by = {}
    for nd in nodes:
        by.setdefault(_layer_kind(nd)[1], []).append(nd)
    parents = {}
    for r in nodes:
        rk = _layer_kind(r)[1]
        cand = (by.get("o_proj", []) + by.get("down_proj", []) if rk in mech.RESID_READERS
                else by.get("v_proj", []) if rk == "o_proj" else by.get("c_fc", []) if rk == "down_proj" else [])
        ws = [w for w in cand if connects(w, r, targets)]
        if ws:
            parents[r] = ws
    out = [w for w in by.get("o_proj", []) + by.get("down_proj", []) if connects(w, None, targets)]
    return parents, out


class Native:
    """Runs, scores and finds graphs on one device."""

    def __init__(self, dev: str | None = None):
        self.dev = dev or device()
        self.target, self.vpd = model(self.dev)
        self.names = self.vpd.names
        self.C = self.vpd.C

    # ---- graphs from answers and from VPD

    def from_ir(self, ir: dict) -> Graph:
        """The Graph of a traced answer (mech's IR "graph": nodes [layer, kind, position, index], parents [[reader,
        writer], ...] and out [writer, ...] as node indices)."""
        g = ir["graph"]
        nodes = [(site_name(layer, kind), t, c) for layer, kind, t, c in g["nodes"]]
        parents = {}
        for r, w in g["parents"]:
            parents.setdefault(nodes[r], []).append(nodes[w])
        return Graph(nodes, parents, [nodes[w] for w in g["out"]])

    @torch.no_grad()
    def importance(self, ids: list[int]) -> dict:
        """VPD's causal importance [T, C] per weight matrix on one sequence."""
        _, ci = self.vpd.target_and_ci(torch.tensor([ids], device=self.dev))
        return {n: ci[n][0] for n in self.names}

    @torch.no_grad()
    def activations(self, ids: list[int]) -> dict:
        """Every subcomponent's activation a_c(t) = (x_t . v_c) |u_c| on the model's own run, [T, C] per matrix."""
        self.vpd.clear()
        for n in self.names:
            self.target.site(n).cache_input = True
        self.target(torch.tensor([ids], device=self.dev))
        out = {n: (self.target.site(n).last_input[0] @ self.target.site(n).V) * self.target.site(n).U.norm(dim=1) for n in self.names}
        self.vpd.clear()
        return out

    def vpd_answer(self, ids: list[int]) -> Graph:
        """VPD's own answer, for comparison: at every position the subcomponents whose causal importance there is above
        zero, complete."""
        return Graph(masks={n: m > 0 for n, m in self.importance(ids).items()})

    def everything(self, T: int) -> Graph:
        """Every subcomponent at every position, complete: the model itself."""
        return Graph(masks={n: torch.ones(T, self.C[n], dtype=torch.bool, device=self.dev) for n in self.names})

    def reachable(self, T: int, targets: list[int]) -> dict:
        """[T, C] per matrix: the subcomponents at positions that can affect the targets (none after the last target,
        and before the first none of the last layer's other than key and value, whose outputs reach no later
        position's input)."""
        last, first, final = self.target.n_layer - 1, min(targets), max(targets)
        out = {}
        for n in self.names:
            m = torch.ones(T, self.C[n], dtype=torch.bool, device=self.dev)
            m[final + 1:] = False
            layer, kind = int(n.split(".")[1]), n.split(".")[-1]
            if layer == last and kind not in ("k_proj", "v_proj"):
                m[:first] = False
            out[n] = m
        return out

    # ---- running graphs

    @torch.no_grad()
    def reference(self, ids: list[list[int]], targets: list[int]) -> torch.Tensor:
        """log p of the model at the targets of each sequence [B, len(targets), V]."""
        logits = self.vpd.target_forward(torch.tensor(ids, device=self.dev))[:, targets]
        return torch.log_softmax(logits.float(), -1)

    def _plan(self, graphs: list[Graph], T: int) -> dict:
        """Index tensors for a batch of graphs: per matrix the [B, T, C] node mask, and for the graphs that are not
        complete, the reader nodes with parents, the writer nodes with children, and the edges between them (in the
        order of each graph's parents lists, then its out list)."""
        dev = self.dev
        G = {n: torch.zeros(len(graphs), T, self.C[n], dtype=torch.bool, device=dev) for n in self.names}
        for b, g in enumerate(graphs):
            if g.masks is not None:
                for n in self.names:
                    G[n][b] = g.masks[n]
                continue
            per = {}
            for n, t, c in g.nodes:
                per.setdefault(n, ([], []))
                per[n][0].append(t)
                per[n][1].append(c)
            for n, (ts, cs) in per.items():
                G[n][b, torch.tensor(ts, device=dev), torch.tensor(cs, device=dev)] = True
        readers, writers, edges, out, order = {}, {}, {}, {}, []

        def row(table, node, b):
            rows = table.setdefault(node[0], {})
            return rows.setdefault((b, node[1], node[2]), len(rows))

        for b, g in enumerate(graphs):
            if g.complete:
                continue
            for r, ws in g.parents.items():
                ri = row(readers, r, b)
                for w in ws:
                    lst = edges.setdefault((r[0], w[0]), [])
                    order.append(("edge", (r[0], w[0]), len(lst)))
                    lst.append((ri, row(writers, w, b)))
            for w in g.out:
                lst = out.setdefault(w[0], [])
                order.append(("out", w[0], len(lst)))
                lst.append((b, w[1], row(writers, w, b)))

        def cols(rows):
            keys = sorted(rows, key=rows.get)
            return {k: torch.tensor([x[k] for x in keys], device=dev) for k in (0, 1, 2)}

        return {"G": G, "complete": torch.tensor([g.complete for g in graphs], device=dev)[:, None, None],
                "readers": {n: cols(r) for n, r in readers.items()}, "writers": {n: cols(w) for n, w in writers.items()},
                "edges": {k: (torch.tensor([e[0] for e in v], device=dev), torch.tensor([e[1] for e in v], device=dev)) for k, v in edges.items()},
                "out": {n: tuple(torch.tensor([e[i] for e in v], device=dev) for i in range(3)) for n, v in out.items()},
                "order": order}

    def edge_weights(self, plan: dict, w: torch.Tensor) -> tuple[dict, dict]:
        """Per-edge strengths in plan order (a vector over every edge of every graph) -> run()'s ew and ow."""
        src_e = {k: [] for k in plan["edges"]}
        src_o = {k: [] for k in plan["out"]}
        for j, (kind, key, _) in enumerate(plan["order"]):
            (src_e if kind == "edge" else src_o)[key].append(j)
        ew = {k: w[torch.tensor(src_e[k], device=self.dev)] for k in src_e}
        ow = {k: w[torch.tensor(src_o[k], device=self.dev)] for k in src_o}
        return ew, ow

    def run(self, ids: list[int], targets: list[int], plan: dict, u: dict, ur: dict, ew: dict | None = None, ow: dict | None = None,
            qk: dict | None = None, patch: dict | None = None, record: dict | None = None) -> torch.Tensor:
        """log q at the targets [B, len(targets), V] of each graph under one ablation: u[matrix] [Bu, S, C] and
        ur[matrix] [Bu, S] in [0, 1], Bu = B or 1, S = T or 1. A subcomponent outside a graph runs at mask u; a graph
        node at 1. An output reaches a reader through a declared edge of strength e at e + (1 - e) u (e = 1 unless ew
        / ow give it) and through every other connection at its writer's u. A complete graph's nodes read each other
        in full. qk[matrix] [Bu, T, C] gives a query or key node a strength e: it enters the attention pattern at
        e + (1 - e) u (1 without qk). The token embedding always enters in full; remainders W - V U run at ur.
        patch[matrix] = (mask [B, T, C], values [B, T, C]) replaces those subcomponents' activations (an interchange);
        record, a dict, receives every matrix's activations [B, T, C]."""
        import vpd_model

        tg, dev = self.target, self.dev
        G, comp, R, Wr, E, O = plan["G"], plan["complete"], plan["readers"], plan["writers"], plan["edges"], plan["out"]
        B = next(iter(G.values())).shape[0]
        T = len(ids)
        H, hd, eps = tg.n_head, tg.hd, tg.eps
        rms, gelu = vpd_model.rms, vpd_model.gelu_tanh

        def ux(n):  # [B, T, C]
            return u[n].expand(B, T, -1)

        def scale(n):  # 1 in the graph (a query or key node's strength under qk), u outside
            if qk is not None and n in qk:
                w = qk[n].expand(B, T, -1)
                return torch.where(G[n], w + (1 - w) * ux(n), ux(n))
            return torch.where(G[n], 1.0, ux(n))

        def uat(n, b, t, c):
            return u[n][b if u[n].shape[0] > 1 else torch.zeros_like(b), t if u[n].shape[1] > 1 else torch.zeros_like(t), c]

        def rest(n, z):
            return ur[n].expand(B, T)[..., None] * (z @ tg.site(n).delta_T)

        def strength(key, w, table):
            return 1.0 if table is None or key not in table else table[key]

        def fixed(n, A):  # an activation tensor after patching, recorded
            if patch is not None and n in patch:
                A = torch.where(patch[n][0], patch[n][1], A)
            if record is not None:
                record[n] = A.detach()
            return A

        emb = tg.wte[torch.tensor(ids, device=dev)]
        x = emb[None].expand(B, T, -1).clone()  # every output at its scale
        xg = x.clone()  # complete graphs: their nodes' outputs in full
        acts = {}

        def writer_out(wn, wi, space_U):  # (b, t, the writer's activation, its u) of writer rows wi
            b, t, c = Wr[wn][0][wi], Wr[wn][1][wi], Wr[wn][2][wi]
            return b, t, c, acts[wn][b, t, c], uat(wn, b, t, c)

        def topup(n, width, contrib):  # each reader node's declared parents' extra share
            out = torch.zeros(len(R[n][0]), width, device=dev)
            for (rn, wn), (ri, wi) in E.items():
                if rn == n:
                    out = out.index_add(0, ri, contrib(wn, wi, strength((rn, wn), wi, ew)))
            return out

        def resid_contrib(wn, wi, e):
            _, _, c, a, uw = writer_out(wn, wi, None)
            return (e * (1 - uw) * a)[:, None] * tg.site(wn).U[c]

        def read_resid(n, norm, z, zg, xs):
            V = tg.site(n).V
            A = torch.where(G[n] & comp, zg @ V, z @ V)
            if n in R:
                b, t, c = R[n][0], R[n][1], R[n][2]
                inp = rms(xs[b, t] + topup(n, xs.shape[-1], resid_contrib), norm, eps)
                A = A.index_put((b, t, c), (inp * V.T[c]).sum(-1))
            return A

        for i in range(tg.n_layer):
            nm = {k: site_name(i, k) for k in vpd_model.KINDS}
            n1, n2 = tg.norms[2 * i], tg.norms[2 * i + 1]
            z, zg = rms(x, n1, eps), rms(xg, n1, eps)
            for k in ("q_proj", "k_proj", "v_proj"):
                acts[nm[k]] = fixed(nm[k], read_resid(nm[k], n1, z, zg, x))
            qv = (acts[nm["q_proj"]] * scale(nm["q_proj"])) @ tg.site(nm["q_proj"]).U + rest(nm["q_proj"], z)
            kv = (acts[nm["k_proj"]] * scale(nm["k_proj"])) @ tg.site(nm["k_proj"]).U + rest(nm["k_proj"], z)
            Uv = tg.site(nm["v_proj"]).U
            rv = rest(nm["v_proj"], z)
            vs = (acts[nm["v_proj"]] * ux(nm["v_proj"])) @ Uv + rv
            vg = (acts[nm["v_proj"]] * scale(nm["v_proj"])) @ Uv + rv
            heads = lambda y: y.view(B, T, H, hd).transpose(1, 2)  # noqa: E731
            q, kk = tg._rope(heads(qv), T), tg._rope(heads(kv), T)
            causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
            P = ((q @ kk.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)  # [B, H, T, T]
            att_s = (P @ heads(vs)).transpose(1, 2).reshape(B, T, -1)
            att_g = (P @ heads(vg)).transpose(1, 2).reshape(B, T, -1)
            no = nm["o_proj"]
            Vo = tg.site(no).V
            Ao = torch.where(G[no] & comp, att_g @ Vo, att_s @ Vo)
            if no in R:
                b, t, c = R[no][0], R[no][1], R[no][2]
                extra = torch.zeros(len(b), Vo.shape[0], device=dev)
                for (rn, wn), (ri, wi) in E.items():
                    if rn != no:
                        continue
                    wb, wt, wc, a, uw = writer_out(wn, wi, None)
                    share = strength((rn, wn), wi, ew) * (1 - uw) * a
                    pat = P[wb, :, t[ri], wt]  # [E, H]: how much the reader's position attends to the writer's
                    extra = extra.index_add(0, ri, share[:, None] * (pat.repeat_interleave(hd, dim=1) * Uv[wc]))
                Ao = Ao.index_put((b, t, c), ((att_s[b, t] + extra) * Vo.T[c]).sum(-1))
            acts[no] = Ao = fixed(no, Ao)
            Uo = tg.site(no).U
            ro = rest(no, att_s)
            x = x + (Ao * ux(no)) @ Uo + ro
            xg = xg + (Ao * scale(no)) @ Uo + ro
            z2, zg2 = rms(x, n2, eps), rms(xg, n2, eps)
            nf, nd = nm["c_fc"], nm["down_proj"]
            acts[nf] = fixed(nf, read_resid(nf, n2, z2, zg2, x))
            Uf = tg.site(nf).U
            rf = rest(nf, z2)
            pre_s = (acts[nf] * ux(nf)) @ Uf + rf
            pre_g = (acts[nf] * scale(nf)) @ Uf + rf
            Vd = tg.site(nd).V
            Ad = torch.where(G[nd] & comp, gelu(pre_g) @ Vd, gelu(pre_s) @ Vd)
            if nd in R:
                b, t, c = R[nd][0], R[nd][1], R[nd][2]

                def mlp_contrib(wn, wi, e):
                    _, _, wc, a, uw = writer_out(wn, wi, None)
                    return (e * (1 - uw) * a)[:, None] * Uf[wc]
                Ad = Ad.index_put((b, t, c), (gelu(pre_s[b, t] + topup(nd, Uf.shape[1], mlp_contrib)) * Vd.T[c]).sum(-1))
            acts[nd] = Ad = fixed(nd, Ad)
            Ud = tg.site(nd).U
            rd = rest(nd, gelu(pre_s))
            x = x + (Ad * ux(nd)) @ Ud + rd
            xg = xg + (Ad * scale(nd)) @ Ud + rd
        zt = torch.where(comp[:, :, :1].expand(B, 1, 1), xg[:, targets], x[:, targets])
        for wn, (b, t, wi) in O.items():
            pos = torch.tensor([targets.index(int(a)) for a in t.tolist()], device=dev)
            zt = zt.index_put((b, pos), resid_contrib(wn, wi, strength(wn, wi, ow)), accumulate=True)
        return torch.log_softmax((rms(zt, tg.ln_f, eps) @ tg.wte.T).float(), -1)

    # ---- the tests

    def _uniform(self, T: int, seed: int) -> tuple[dict, dict]:
        g = torch.Generator(device="cpu").manual_seed(seed)
        return ({n: torch.rand(1, T, self.C[n], generator=g).to(self.dev) for n in self.names},
                {n: torch.rand(1, T, generator=g).to(self.dev) for n in self.names})

    def _kl(self, logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
        return (logp.exp() * (logp - logq)).sum(-1).sum(-1) / LN2

    def score(self, ids: list[int], targets: list[int], graphs: list[Graph], seed: int = 0, logp: torch.Tensor | None = None) -> list[dict]:
        """{"kl_bits", "kl_deleted_bits", "kl_random_bits", "nodes", "edges", "size"} per graph."""
        B, T = len(graphs), len(ids)
        if B == 0:
            return []
        logp = self.reference([ids], targets) if logp is None else logp
        plan = self._plan(graphs, T)
        with torch.no_grad():
            zero = ({n: torch.zeros(1, 1, self.C[n], device=self.dev) for n in self.names}, {n: torch.zeros(1, 1, device=self.dev) for n in self.names})
            deleted = self._kl(logp, self.run(ids, targets, plan, *zero))
            random = torch.stack([self._kl(logp, self.run(ids, targets, plan, *self._uniform(T, seed * 1000 + j))) for j in range(RANDOM)]).mean(0)
        total = torch.maximum(deleted, random)
        return [{"kl_bits": float(total[b]), "kl_deleted_bits": float(deleted[b]), "kl_random_bits": float(random[b]), "nodes": graphs[b].count(),
                 "edges": None if graphs[b].complete else graphs[b].edges(), "size": None if graphs[b].complete else graphs[b].size()} for b in range(B)]

    # ---- interchanges

    @torch.no_grad()
    def interchange(self, ids: list[int], edited: list[list[int]], targets: list[int], nodes: set) -> list[dict]:
        """For each edited sequence (same length): the model's prediction at the targets on the original text, on the
        edited text, and on the original text with only `nodes`' activations taken from the edited run (everything
        downstream recomputed): their top tokens, and the KLs in bits of the edited prediction from the original's
        and from the patched one's."""
        T = len(ids)
        plan = self._plan([self.everything(T)], T)
        one = ({n: torch.ones(1, 1, self.C[n], device=self.dev) for n in self.names}, {n: torch.ones(1, 1, device=self.dev) for n in self.names})
        lp = self.run(ids, targets, plan, *one)
        masks = {}
        for n, t, c in nodes:
            masks.setdefault(n, torch.zeros(1, T, self.C[n], dtype=torch.bool, device=self.dev))[0, t, c] = True
        out = []
        for e in edited:
            rec = {}
            le = self.run(e, targets, plan, *one, record=rec)
            lq = self.run(ids, targets, plan, *one, patch={n: (m, rec[n]) for n, m in masks.items()})
            out.append({"top": lp.argmax(-1)[0].tolist(), "edited_top": le.argmax(-1)[0].tolist(), "patched_top": lq.argmax(-1)[0].tolist(),
                        "edit_kl_bits": float(self._kl(le, lp)[0]), "residual_kl_bits": float(self._kl(le, lq)[0])})
        return out

    @torch.no_grad()
    def claims(self, ids: list[int], targets: list[int], g: Graph, candidates: int = CANDIDATES) -> list[tuple[int, int, int]]:
        """The teacher's interchange claims about graph g, one per source position p of its pathways (pathways()):
        among the `candidates` tokens whose embeddings are nearest the token at p, the replacement that changes the
        model's top prediction the most (edit KL) while the pathway from p alone reproduces the new top prediction when
        its activations come from the edited run. (p, replacement token id, the new top token id) per claim."""
        wte = self.target.wte
        unit = wte / wte.norm(dim=1, keepdim=True)
        out = []
        for p, nodes in pathways(g, targets).items():
            near = (unit @ unit[ids[p]]).topk(candidates + 1).indices.tolist()
            subs = [c for c in near if c != ids[p]][:candidates]
            edited = [ids[:p] + [c] + ids[p + 1:] for c in subs]
            res = self.interchange(ids, edited, targets, nodes)
            ok = [(r["edit_kl_bits"], c, r["edited_top"][0]) for c, r in zip(subs, res)
                  if r["edited_top"] != r["top"] and r["patched_top"] == r["edited_top"]]
            if ok:
                _, c, top = max(ok)
                out.append((p, c, top))
        return out

    # ---- the teacher

    def _strength_loop(self, forward, logits: torch.Tensor, target: float, log=None, what: str = "", steps: int = TEACH_STEPS) -> torch.Tensor:
        """Minimize mean(sigmoid(logits)) subject to max(KL_deleted, KL_random) <= target by Adam on the logits and dual
        ascent on the multiplier; forward(strengths, "zero" or "random") -> KL. Returns the final strengths."""
        logits = logits.clone().requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=TEACH_LR)
        mu = 0.0
        for step in range(steps):
            with torch.enable_grad():
                g = torch.sigmoid(logits)
                k = torch.maximum(forward(g, "zero"), forward(g, "random"))
                loss = g.mean() + mu * (k - target)
                opt.zero_grad()
                loss.backward()
            opt.step()
            mu = max(0.0, mu + float(k) - target)
            if log and (step % 50 == 0 or step == steps - 1):
                log(f"{what} step {step}: kl {float(k):.4f} (target {target}), mean strength {float(g.mean()):.4f}, above 1/2: {int((g > 0.5).sum())}, mu {mu:.3f}")
        return torch.sigmoid(logits.detach())

    def teach(self, ids: list[int], targets: list[int], eps: float, seed: int = 0, log=None) -> tuple[Graph, dict]:
        """(one prediction's graph at precision eps, its score); see the module docstring."""
        T = len(ids)
        logp = self.reference([ids], targets)
        ids_b = torch.tensor([ids], device=self.dev)
        reach = self.reachable(T, targets)
        names = self.names
        flat = torch.cat([reach[n].flatten() for n in names])
        sizes = [T * self.C[n] for n in names]
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def split(v):  # a vector over the reachable (matrix, position, index) slots -> [T, C] per matrix
            full = torch.zeros(len(flat), device=self.dev)
            full[flat] = v
            return {n: p.view(T, self.C[n]) for n, p in zip(names, torch.split(full, sizes))}

        def node_forward(g, r):
            m = split(g)
            if r == "zero":
                masks, rest = {n: m[n][None] for n in names}, {n: torch.zeros(1, T, device=self.dev) for n in names}
            else:
                rr = {n: torch.rand(T, self.C[n], generator=gen).to(self.dev) for n in names}
                masks = {n: (m[n] + (1 - m[n]) * rr[n])[None] for n in names}
                rest = {n: torch.rand(1, T, generator=gen).to(self.dev) for n in names}
            lq = torch.log_softmax(self.vpd.masked(ids_b, masks, rest)[:, targets].float(), -1)
            return self._kl(logp, lq)[0]

        g = self._strength_loop(node_forward, torch.full((int(flat.sum()),), 2.0, device=self.dev), eps / 2, log, "nodes")
        sel = split(g)
        nodes = {(n, t, c) for n in names for t, c in (sel[n] > 0.5).nonzero().tolist()}
        parents, out = all_edges(nodes, targets)
        if log:
            log(f"nodes kept: {len(nodes)}; connections between them: {sum(len(w) for w in parents.values()) + len(out)}")
        full = Graph(nodes, parents, out)
        plan = self._plan([full], T)
        n_edges = len(plan["order"])

        qk_nodes = sorted(nd for nd in nodes if _layer_kind(nd)[1] in ("q_proj", "k_proj"))  # they act through attention patterns
        qk_sites = sorted({nd[0] for nd in qk_nodes})
        qk_idx = {n: [j for j, nd in enumerate(qk_nodes) if nd[0] == n] for n in qk_sites}

        def qk_of(w):
            out_ = {}
            for n in qk_sites:
                js = qk_idx[n]
                t = torch.tensor([qk_nodes[j][1] for j in js], device=self.dev)
                c = torch.tensor([qk_nodes[j][2] for j in js], device=self.dev)
                out_[n] = torch.zeros(1, T, self.C[n], device=self.dev).index_put((torch.zeros_like(t), t, c), w[torch.tensor(js, device=self.dev)])
            return out_

        def edge_forward(e, r):
            ew, ow = self.edge_weights(plan, e[:n_edges])
            qk = qk_of(e[n_edges:])
            if r == "zero":
                u = ({n: torch.zeros(1, 1, self.C[n], device=self.dev) for n in names}, {n: torch.zeros(1, 1, device=self.dev) for n in names})
            else:
                u = self._uniform(T, int(torch.randint(1 << 30, (1,), generator=gen)))
            return self._kl(logp, self.run(ids, targets, plan, *u, ew, ow, qk))[0]

        n_items = n_edges + len(qk_nodes)
        e = self._strength_loop(edge_forward, torch.full((n_items,), 2.0, device=self.dev), eps, log, "edges", EDGE_STEPS)
        edge_list = [(r, w) for r, ws in parents.items() for w in ws] + [(None, w) for w in out]  # plan order
        assert len(edge_list) == n_edges
        ranked = sorted(range(n_items), key=lambda j: -float(e[j]))
        keep = int((e > 0.5).sum())

        def build(js):
            par, o, used = {}, [], set()
            for j in js:
                if j >= n_edges:
                    used.add(qk_nodes[j - n_edges])
                    continue
                r, w = edge_list[j]
                used.add(w)
                if r is None:
                    o.append(w)
                else:
                    par.setdefault(r, []).append(w)
                    used.add(r)
            return Graph(used, par, o)

        while True:  # the strongest edges and query and key nodes, more of them until the exact tests pass
            g_out = build(ranked[:keep])
            s = self.score(ids, targets, [g_out], seed, logp)[0]
            if s["kl_bits"] <= eps or keep >= n_items:
                return self.clean(ids, targets, g_out, s, eps, seed, logp)
            keep = min(n_items, max(keep + 1, int(keep * 1.25)))

    def clean(self, ids: list[int], targets: list[int], g: Graph, s: dict, eps: float, seed: int = 0, logp=None) -> tuple[Graph, dict]:
        """g without what lies on no path to the prediction (pathways()), kept when it still passes the tests at eps."""
        on_path = set().union(*pathways(g, targets).values()) if g.out else set()
        if on_path == g.nodes:
            return g, s
        h = Graph(on_path, {r: [w for w in ws if w in on_path] for r, ws in g.parents.items() if r in on_path}, [w for w in g.out if w in on_path])
        h.parents = {r: ws for r, ws in h.parents.items() if ws}
        sh = self.score(ids, targets, [h], seed, logp)[0]
        return (h, sh) if sh["kl_bits"] <= eps else (g, s)


def pathways(g: Graph, targets: list[int]) -> dict[int, set]:
    """Per source position p, the graph's nodes on a path from a node at p (which reads token p) to the prediction: a
    reader's parents feed it, a query feeds its layer's attention outputs at its position, a key those at its position
    and later."""
    children: dict = {}
    for r, ws in g.parents.items():
        for w in ws:
            children.setdefault(w, set()).add(r)
    outs = [nd for nd in g.nodes if _layer_kind(nd)[1] == "o_proj"]
    for nd in g.nodes:
        layer, kind = _layer_kind(nd)
        if kind in ("q_proj", "k_proj"):
            for o in outs:
                if _layer_kind(o)[0] == layer and (o[1] == nd[1] if kind == "q_proj" else o[1] >= nd[1]):
                    children.setdefault(nd, set()).add(o)
    parents: dict = {}
    for w, rs in children.items():
        for r in rs:
            parents.setdefault(r, set()).add(w)
    ancestors, stack = set(g.out), list(g.out)
    while stack:
        for w in parents.get(stack.pop(), ()):
            if w not in ancestors:
                ancestors.add(w)
                stack.append(w)
    out = {}
    for p in sorted({nd[1] for nd in ancestors}):
        seen, stack = set(), [nd for nd in ancestors if nd[1] == p]
        while stack:
            nd = stack.pop()
            if nd in seen or nd not in ancestors:
                continue
            seen.add(nd)
            stack += list(children.get(nd, ()))
        out[p] = seen
    return out


def program(g: Graph, claims: list | None = None) -> str:
    """A graph as an answer: graph(tokens, targets) returning {(position, reader): its parents, "out": the
    prediction's parents, "claims": [(position, replacement, top), ...]}; parents at the reader's own position are one
    string of subcomponents, an attention output's parents (values) a {position: string}; a node with no parents is
    listed with "" when nothing reads it; claims hold token strings."""
    sites = list(mech.SITES.values())
    codes = {v: k for k, v in mech.SITES.items()}

    def tok(nd):
        layer, kind = _layer_kind(nd)
        return f"<p:{layer}.{codes[kind]}.{nd[2]}>"

    def key(nd):
        layer, kind = _layer_kind(nd)
        return (nd[1], layer, sites.index(kind), nd[2])

    written = {w for ws in g.parents.values() for w in ws} | set(g.out)
    readers = set(g.parents) | {nd for nd in g.nodes if nd not in written}
    lines = []
    for r in sorted(readers, key=key):
        ws = sorted(g.parents.get(r, []), key=key)
        if r[0].endswith("o_proj"):
            per = {}
            for w in ws:
                per.setdefault(w[1], []).append(tok(w))
            val = "{" + ", ".join(f'{t}: "' + "".join(v) + '"' for t, v in sorted(per.items())) + "}"
        else:
            val = '"' + "".join(tok(w) for w in ws) + '"'
        lines.append(f'        ({r[1]}, "{tok(r)}"): {val},')
    lines.append('        "out": "' + "".join(tok(w) for w in sorted(g.out, key=key)) + '",')
    if claims:
        lines.append('        "claims": [' + ", ".join(f"({p}, {a!r}, {b!r})" for p, a, b in claims) + "],")
    return "def graph(tokens, targets):\n    return {\n" + "\n".join(lines) + "\n    }\n"


def token_claims(claims: list[tuple[int, int, int]]) -> list[tuple[int, str, str]]:
    """Claims with their tokens as strings, keeping those whose strings read back as the same single tokens."""
    tk = mech.tokenizer("vpd4l")
    out = []
    for p, a, b in claims:
        sa, sb = tk.decode([a]), tk.decode([b])
        if tk.encode(sa, add_special_tokens=False).ids == [a] and tk.encode(sb, add_special_tokens=False).ids == [b]:
            out.append((p, sa, sb))
    return out


def tasks(split: str) -> list[Path]:
    """The split's task files in text order."""
    return [p for p in sorted((TEXTS / "vpd4l").glob("*.json"), key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())["split"] == split]


def text(p: Path) -> tuple[list[int], list[int]]:
    task = json.loads(p.read_text())["prompts"][0]
    return task["token_ids"], task["target_positions"]


def teach(split: str, n: int, offset: int = 0, stride: int = 1, epsilons=(0.25, 1.0), out: Path | None = None) -> None:
    """Native.teach on the split's texts offset, offset + stride, ... of its first n, at each eps ->
    OUT/<id>.<eps>.py and .json (its score and the seconds); OUT defaults to texts/graphs[_heldout]."""
    nat = Native()
    out = Path(out) if out else TEXTS / ("graphs" if split == "train" else "graphs_heldout")
    out.mkdir(parents=True, exist_ok=True)
    for p in tasks(split)[:n][offset::stride]:
        ids, targets = text(p)
        for eps in epsilons:
            stem = f"{p.stem}.{eps:g}"
            if (out / f"{stem}.json").exists():
                continue
            t0 = time.time()
            g, s = nat.teach(ids, targets, eps)
            claims = token_claims(nat.claims(ids, targets, g))
            (out / f"{stem}.py").write_text(program(g, claims))
            (out / f"{stem}.json").write_text(json.dumps({"score": s, "eps": eps, "claims": claims, "seconds": round(time.time() - t0, 1)}))
            print(f"{stem}: {s['nodes']} nodes, {s['edges']} edges, kl {s['kl_bits']:.3f} (eps {eps}), {len(claims)} claims, {time.time() - t0:.0f} s", flush=True)


def clean_dir(directory: Path, split: str) -> None:
    """Native.clean every teacher graph <task>.<eps>.py in a directory (a pod's output), rewriting it and its .json."""
    nat = Native()
    by = {p.stem: p for p in tasks(split)}
    for py in sorted(Path(directory).glob("*.py")):
        task_id = py.stem.split(".")[0]  # <task>.<eps>
        eps = float(py.stem[len(task_id) + 1:])
        rec = json.loads(py.with_suffix(".json").read_text())
        ids, targets = text(by[task_id])
        task = json.loads(by[task_id].read_text())
        ir = mech.trace_inline(py.read_text(), "vpd4l", task)
        g = nat.from_ir(ir)
        h, s = nat.clean(ids, targets, g, rec["score"], eps)
        if h is not g:
            claims = rec.get("claims", [])
            py.write_text(program(h, claims))
            rec.update(score=s, cleaned=True)
            py.with_suffix(".json").write_text(json.dumps(rec))
            print(f"{py.stem}: {rec['score']['nodes']} nodes, {rec['score']['edges']} edges after cleaning", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clean")
    c.add_argument("directory", type=Path)
    c.add_argument("--split", choices=("train", "heldout"), required=True)
    t = sub.add_parser("teach")
    t.add_argument("--split", choices=("train", "heldout"), required=True)
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--offset", type=int, default=0)
    t.add_argument("--stride", type=int, default=1)
    t.add_argument("--eps", type=float, nargs="+", default=[0.25, 1.0])
    t.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.cmd == "clean":
        clean_dir(args.directory, args.split)
    else:
        teach(args.split, args.n, args.offset, args.stride, tuple(args.eps), args.out)


if __name__ == "__main__":
    main()
