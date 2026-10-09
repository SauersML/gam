"""Computational graphs of vpd4l's predictions (#2951 graph oracle), run natively in torch: the verifier.

The model runs with every weight matrix replaced by VPD's subcomponents and remainder,
    y = ((x V) * m) U + m_delta x (W - V U)^T,
masks m in [0, 1] per position. An answer is a graph: nodes are subcomponents at positions; edges are connections the
model has (an attention or MLP output into a later query, key, value or MLP input at its position, or into the
prediction; a value into the same layer's attention output at that or a later position; an MLP input into the same
MLP's output). Native.run runs graphs: their nodes at mask 1; every other subcomponent at an ablation mask u; a node
receives its declared parents' outputs in full and every other output scaled by that output's u; the token embedding,
which VPD does not decompose, in full. With u = 1 everywhere this is the model; with u = 0 (the verifier's setting) the
graph runs alone: nothing it leaves out, remainders included, contributes.

Faithfulness (Native.faithfulness), in bits: the KL of the model's next-token distribution at the target from the
graph's run alone, averaged over changed prompts of the text (Native.changes): the token at one position, drawn
uniformly from the positions whose token the model predicts (1 to the target), replaced by a draw from the model's own
prediction there; a draw can return the original token, so the text itself is among them. The verifier draws them,
never the answer. A graph that leaves out a path by which a token affects the prediction fails on the prompts that
change that token, and one more confident than the model fails because the KL is of the whole distribution. No test
proves a graph is the model's mechanism; this measures how far its outputs are from the model's on these experiments.

Description length (Graph.bits), in bits: the nodes listed, each one choice among the text's positions times the
model's 38,912 subcomponents, then each edge as its reader's and its writer's index in that list, and each edge into the
prediction as its writer's index.

Native.ordered is a search that bootstraps the oracle, not the method: integrated-gradients rankings of subcomponents
and then of the connections between the top ones, averaged over the changed prompts; the answer adds the ranked
connections in order in steps that double their number, each step keeping what lies on a path to the prediction, and
ends at the first step that lowers the KL no further.

  native.py search --split train --n N [--offset K --stride S] [--out DIR]   -> DIR/<task>.py, .json
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
CHANGES = 16  # changed prompts per text: draws of the average that defines faithfulness (they set its precision, not what it is)
MAX_CONNECTIONS = 150000  # the search's memory: the connections it ranks at once
IG_STEPS = 16  # the search's integration points along each ranking's path
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

    def bits(self, positions: int, subcomponents: int) -> float:
        """Description length: each node one choice among positions x subcomponents, each edge its reader's and
        writer's indices among the nodes, each edge into the prediction its writer's index (a complete graph states
        no edges)."""
        n = self.count()
        if n == 0:
            return 0.0
        nodes = n * math.log2(positions * subcomponents)
        if self.complete:
            return nodes
        index = math.log2(n)
        return nodes + index * (2 * sum(len(w) for w in self.parents.values()) + len(self.out))


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


def _candidates(nodes) -> dict:
    """Per reader node, the nodes among `nodes` the model connects into it (mech.connects): residual writers at its
    position written before it, values of its layer at its position or earlier, its MLP's input at its position."""
    resid, values, inputs = {}, {}, {}
    for nd in nodes:
        layer, kind = _layer_kind(nd)
        if kind in mech.RESID_WRITERS:
            resid.setdefault(nd[1], []).append((mech.stage(layer, kind), nd))
        elif kind == "v_proj":
            values.setdefault(layer, []).append(nd)
        elif kind == "c_fc":
            inputs.setdefault((layer, nd[1]), []).append(nd)
    out = {}
    for r in nodes:
        layer, kind = _layer_kind(r)
        if kind in mech.RESID_READERS:
            s = mech.stage(layer, kind)
            out[r] = [w for st, w in resid.get(r[1], ()) if st <= s]
        elif kind == "o_proj":
            out[r] = [w for w in values.get(layer, ()) if w[1] <= r[1]]
        elif kind == "down_proj":
            out[r] = list(inputs.get((layer, r[1]), ()))
    return out


def all_edges(nodes, targets: list[int]) -> tuple[dict, list]:
    """Every connection the model has between `nodes`: (parents, out)."""
    parents = {r: ws for r, ws in _candidates(nodes).items() if ws}
    out = [w for w in nodes if _layer_kind(w)[1] in mech.RESID_WRITERS and w[1] in targets]
    return parents, out


def count_edges(nodes, targets: list[int]) -> int:
    """How many connections all_edges would give."""
    return sum(len(ws) for ws in _candidates(nodes).values()) + sum(1 for w in nodes if _layer_kind(w)[1] in mech.RESID_WRITERS and w[1] in targets)


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

    # ---- the verifier

    def _kl(self, logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
        return (logp.exp() * (logp - logq)).sum(-1).sum(-1) / LN2

    def _zero(self) -> tuple[dict, dict]:
        return ({n: torch.zeros(1, 1, self.C[n], device=self.dev) for n in self.names}, {n: torch.zeros(1, 1, device=self.dev) for n in self.names})

    @torch.no_grad()
    def changes(self, ids: list[int], targets: list[int], n: int = CHANGES, seed: int = 0) -> list[list[int]]:
        """n changed prompts of a text: each replaces the token at a position drawn uniformly from 1 to the last target
        (the positions whose token the model predicts) by a draw from the model's prediction there."""
        last = max(targets)
        if last < 1:
            return [list(ids)] * n
        probs = self.vpd.target_forward(torch.tensor([ids], device=self.dev))[0, :last].float().softmax(-1).cpu()
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = []
        for _ in range(n):
            p = int(torch.randint(1, last + 1, (1,), generator=g))
            x = list(ids)
            x[p] = int(torch.multinomial(probs[p - 1], 1, generator=g))
            out.append(x)
        return out

    def faithfulness(self, ids: list[int], targets: list[int], graphs: list[Graph], prompts: list[list[int]], chunk: int = 16) -> list[float]:
        """Per graph, the mean over `prompts` (changed prompts of ids) of the KL in bits of the model's next-token
        distribution at the targets from the graph's run alone."""
        if not graphs:
            return []
        T = len(ids)
        refs = [self.reference([x], targets) for x in prompts]
        total = torch.zeros(len(graphs), device=self.dev)
        zero = self._zero()
        for s in range(0, len(graphs), chunk):
            part = graphs[s:s + chunk]
            plan = self._plan(part, T)
            with torch.no_grad():
                for x, logp in zip(prompts, refs):
                    total[s:s + len(part)] += self._kl(logp, self.run(x, targets, plan, *zero))
        return (total / len(prompts)).tolist()

    def positions(self, targets: list[int]) -> int:
        """The positions a graph's nodes can sit at: 0 to the last target."""
        return max(targets) + 1

    def score(self, ids: list[int], targets: list[int], graphs: list[Graph], prompts: list[list[int]]) -> list[dict]:
        """{"kl_bits", "bits", "nodes", "edges"} per graph: its faithfulness and description length."""
        total = sum(self.C.values())
        kl = self.faithfulness(ids, targets, graphs, prompts)
        return [{"kl_bits": k, "bits": g.bits(self.positions(targets), total), "nodes": g.count(), "edges": None if g.complete else g.edges()}
                for g, k in zip(graphs, kl)]

    # ---- the bootstrap search

    def _node_ranking(self, ids: list[int], targets: list[int], prompts: list[list[int]], reach: dict) -> tuple:
        """Every reachable (matrix, position, subcomponent), most important first: integrated gradients of the run-alone
        KL along the path that scales every subcomponent (and remainder) from 1 to 0 together, summed over the changed
        prompts: minus the integral of d KL / d m, each one's share of the KL of removing everything. Only an order."""
        T = len(ids)
        total = {n: torch.zeros(T, self.C[n], device=self.dev) for n in self.names}
        for x in prompts:
            logp = self.reference([x], targets)
            xb = torch.tensor([x], device=self.dev)
            for k in range(IG_STEPS):
                a = (k + 0.5) / IG_STEPS
                masks = {n: torch.full((1, T, self.C[n]), a, device=self.dev, requires_grad=True) for n in self.names}
                rest = {n: torch.full((1, T), a, device=self.dev) for n in self.names}
                with torch.enable_grad():
                    lq = torch.log_softmax(self.vpd.masked(xb, masks, rest)[:, targets].float(), -1)
                    grads = torch.autograd.grad(self._kl(logp, lq)[0], [masks[n] for n in self.names])
                for n, gr in zip(self.names, grads):
                    total[n] -= gr[0]
        mi, ti, ci, val = [], [], [], []
        for j, n in enumerate(self.names):
            t, c = reach[n].nonzero(as_tuple=True)
            mi.append(torch.full_like(t, j))
            ti.append(t)
            ci.append(c)
            val.append(total[n][t, c])
        mi, ti, ci, val = torch.cat(mi), torch.cat(ti), torch.cat(ci), torch.cat(val)
        o = (-val).argsort()
        return [(self.names[int(m)], int(t), int(c)) for m, t, c in zip(mi[o].tolist(), ti[o].tolist(), ci[o].tolist())]

    def _edge_ranking(self, ids: list[int], targets: list[int], prompts: list[list[int]], nodes: set) -> list:
        """Every connection the model has between `nodes` ((reader, writer), reader None for the prediction) and every
        query or key node among them ((node, None)), most important first: integrated gradients of the run-alone KL
        as all their strengths go from 1 to 0 together, summed over the changed prompts."""
        T = len(ids)
        parents, out = all_edges(nodes, targets)
        g = Graph(nodes, parents, out)
        plan = self._plan([g], T)
        items = [(r, w) for r, ws in parents.items() for w in ws] + [(None, w) for w in out]  # plan order
        n_edges = len(items)
        qk = sorted(nd for nd in nodes if _layer_kind(nd)[1] in ("q_proj", "k_proj"))
        sites = sorted({nd[0] for nd in qk})
        idx = {n: [j for j, nd in enumerate(qk) if nd[0] == n] for n in sites}

        def qk_of(w):
            res = {}
            for n in sites:
                js = idx[n]
                t = torch.tensor([qk[j][1] for j in js], device=self.dev)
                c = torch.tensor([qk[j][2] for j in js], device=self.dev)
                res[n] = torch.zeros(1, T, self.C[n], device=self.dev).index_put((torch.zeros_like(t), t, c), w[torch.tensor(js, device=self.dev)])
            return res

        zero = self._zero()
        ig = torch.zeros(n_edges + len(qk), device=self.dev)
        for x in prompts:
            logp = self.reference([x], targets)
            for k in range(IG_STEPS):
                w = torch.full((n_edges + len(qk),), (k + 0.5) / IG_STEPS, device=self.dev, requires_grad=True)
                with torch.enable_grad():
                    ew, ow = self.edge_weights(plan, w[:n_edges])
                    kl = self._kl(logp, self.run(x, targets, plan, *zero, ew, ow, qk_of(w[n_edges:])))[0]
                    (gr,) = torch.autograd.grad(kl, w)
                ig -= gr
        allitems = items + [(nd, None) for nd in qk]
        return [allitems[j] for j in (-ig).argsort().tolist()]

    def ordered(self, ids: list[int], targets: list[int], prompts: list[list[int]], log=None) -> tuple[list[Graph], list[dict]]:
        """A bootstrap answer: (the graph after each step, their scores); see the module docstring. The subcomponents
        whose connections are ranked: the most top-ranked whose connections fit MAX_CONNECTIONS."""
        T = len(ids)
        ranked = self._node_ranking(ids, targets, prompts, self.reachable(T, targets))
        lo = 1
        while lo < len(ranked) and count_edges(ranked[:2 * lo], targets) <= MAX_CONNECTIONS:  # doubling, then bisection:
            lo *= 2  # the largest prefix within MAX_CONNECTIONS
        hi = min(2 * lo, len(ranked))
        lo = min(lo, len(ranked))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_edges(ranked[:mid], targets) <= MAX_CONNECTIONS:
                lo = mid
            else:
                hi = mid - 1
        nodes = set(ranked[:lo])
        items = self._edge_ranking(ids, targets, prompts, nodes)
        if log:
            log(f"{lo} subcomponents ranked first; {len(items)} connections and query/key nodes between them")
        graphs, k = [], 1
        while True:
            g = on_path(build(items[:k]), targets)
            if g.count() and (not graphs or g.size() > graphs[-1].size()):
                graphs.append(g)
            if k >= len(items):
                break
            k = min(2 * k, len(items))
        scores = self.score(ids, targets, graphs, prompts)
        keep = 1
        while keep < len(graphs) and scores[keep]["kl_bits"] < scores[keep - 1]["kl_bits"]:
            keep += 1
        return graphs[:keep], scores[:keep]


def build(items: list) -> Graph:
    """The graph of ranked items: (reader, writer) connections (reader None: the prediction) and (node, None) query or
    key nodes."""
    par, out, used = {}, [], set()
    for r, w in items:
        if w is None:
            used.add(r)
        elif r is None:
            out.append(w)
            used.add(w)
        else:
            par.setdefault(r, []).append(w)
            used |= {r, w}
    return Graph(used, par, out)


def on_path(g: Graph, targets: list[int]) -> Graph:
    """g without what lies on no path to the prediction: a reader's parents feed it, a query feeds its layer's
    attention outputs at its position, a key those at its position and later."""
    feeds: dict = {}
    for r, ws in g.parents.items():
        for w in ws:
            feeds.setdefault(r, set()).add(w)
    outs = [nd for nd in g.nodes if _layer_kind(nd)[1] == "o_proj"]
    for nd in g.nodes:
        layer, kind = _layer_kind(nd)
        if kind in ("q_proj", "k_proj"):
            for o in outs:
                if _layer_kind(o)[0] == layer and (o[1] == nd[1] if kind == "q_proj" else o[1] >= nd[1]):
                    feeds.setdefault(o, set()).add(nd)
    keep, stack = set(g.out), list(g.out)
    while stack:
        for w in feeds.get(stack.pop(), ()):
            if w not in keep:
                keep.add(w)
                stack.append(w)
    par = {r: [w for w in ws if w in keep] for r, ws in g.parents.items() if r in keep}
    return Graph(keep, {r: ws for r, ws in par.items() if ws}, [w for w in g.out if w in keep])


def program(steps: list[Graph], uses: list[list[str]] | None = None, notes: list[str] | None = None, explanation: str | None = None) -> str:
    """Graphs as an ordered answer: graph(tokens, targets) returning a list of steps, the k-th adding what steps[k]
    has beyond steps[k - 1] (a step's graph contains the previous one's); a step is a dict {(position, reader): its
    new parents, "out": the prediction's new parents, "uses": library entries}; parents at the reader's own position
    are one string of subcomponents, an attention output's parents (values) a {position: string}; a node with no
    parents is listed with "" when nothing reads it. notes: a comment line above each step; explanation: the
    function's docstring."""
    sites = list(mech.SITES.values())
    codes = {v: k for k, v in mech.SITES.items()}

    def tok(nd):
        layer, kind = _layer_kind(nd)
        return f"<p:{layer}.{codes[kind]}.{nd[2]}>"

    def key(nd):
        layer, kind = _layer_kind(nd)
        return (nd[1], layer, sites.index(kind), nd[2])

    lines = ["def graph(tokens, targets):"]
    if explanation:
        lines.append('    """' + explanation.replace('"""', "'''").strip() + '"""')
    lines.append("    return [")
    prev = Graph()
    for k, g in enumerate(steps):
        seen = {(r, w) for r, ws in prev.parents.items() for w in ws}
        new_par = {r: [w for w in ws if (r, w) not in seen] for r, ws in g.parents.items()}
        new_par = {r: ws for r, ws in new_par.items() if ws}
        new_out = [w for w in g.out if w not in prev.out]
        written = {w for ws in g.parents.values() for w in ws} | set(g.out)
        lone = {nd for nd in g.nodes - prev.nodes if nd not in written and nd not in g.parents}
        body = []
        for r in sorted(set(new_par) | lone, key=key):
            ws = sorted(new_par.get(r, []), key=key)
            if r[0].endswith("o_proj"):
                per = {}
                for w in ws:
                    per.setdefault(w[1], []).append(tok(w))
                val = "{" + ", ".join(f'{t}: "' + "".join(v) + '"' for t, v in sorted(per.items())) + "}"
            else:
                val = '"' + "".join(tok(w) for w in ws) + '"'
            body.append(f'            ({r[1]}, "{tok(r)}"): {val},')
        if new_out:
            body.append('            "out": "' + "".join(tok(w) for w in sorted(new_out, key=key)) + '",')
        if uses and k < len(uses) and uses[k]:
            body.append('            "uses": [' + ", ".join(repr(u) for u in uses[k]) + "],")
        if notes and k < len(notes) and notes[k]:
            lines.append("        # " + notes[k].strip().replace("\n", " "))
        lines += ["        {"] + body + ["        },"]
        prev = g
    lines.append("    ]")
    return "\n".join(lines) + "\n"


def tasks(split: str) -> list[Path]:
    """The split's task files in text order."""
    return [p for p in sorted((TEXTS / "vpd4l").glob("*.json"), key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())["split"] == split]


def text(p: Path) -> tuple[list[int], list[int]]:
    task = json.loads(p.read_text())["prompts"][0]
    return task["token_ids"], task["target_positions"]


def task_seed(task_id: str) -> int:
    """A text's own seed for its changed prompts (the same in every process)."""
    return int.from_bytes(task_id.encode()[-8:].rjust(8, b"\0"), "big") % (1 << 31)


def search(split: str, n: int, offset: int = 0, stride: int = 1, out: Path | None = None) -> None:
    """Native.ordered on the split's texts offset, offset + stride, ... of its first n -> OUT/<id>.py (the answer) and
    .json (each step's score and the seconds); OUT defaults to texts/search[_heldout]."""
    nat = Native()
    out = Path(out) if out else TEXTS / ("search" if split == "train" else "search_heldout")
    out.mkdir(parents=True, exist_ok=True)
    for p in tasks(split)[:n][offset::stride]:
        if (out / f"{p.stem}.json").exists():
            continue
        t0 = time.time()
        ids, targets = text(p)
        prompts = nat.changes(ids, targets, seed=task_seed(p.stem))
        graphs, scores = nat.ordered(ids, targets, prompts)
        empty = nat.score(ids, targets, [Graph()], prompts)[0]
        (out / f"{p.stem}.py").write_text(program(graphs))
        (out / f"{p.stem}.json").write_text(json.dumps({"empty": empty, "steps": scores, "seconds": round(time.time() - t0, 1)}))
        print(f"{p.stem}: {len(graphs)} steps, KL {empty['kl_bits']:.1f} -> " + " -> ".join(f"{s['kl_bits']:.2f} ({s['bits']:.0f} bits)" for s in scores)
              + f", {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--split", choices=("train", "heldout"), required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--out", type=Path)
    args = ap.parse_args()
    search(args.split, args.n, args.offset, args.stride, args.out)


if __name__ == "__main__":
    main()
