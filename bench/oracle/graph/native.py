"""Native scoring of circuits on vpd4l (#2951 graph oracle), in torch.

The model runs with every site's weight replaced by VPD's subcomponents and remainder,
    y = ((x V) * m) U + m_delta x (W - V U)^T,
masks m in [0, 1] per position. A circuit names, per position, the subcomponents (and remainders "<p:L.S.rest>") that act
there; they are held at 1. Its claim is that everything it leaves out can be ablated in any combination not tuned to this
input. It is tested against ablation patterns fixed before the input is seen, and kl_bits is the largest KL in bits of the
model's next-token distribution at the targets from the circuit's over them:
  deletion   every left-out mask 0 (kl_deleted_bits);
  random     masks drawn uniformly in [0, 1] per position (VPD's stochastic test), the mean over RANDOM draws of the
             step's seed (kl_random_bits);
  universal  adversarial patterns, a mask per position and subcomponent shared by every text, each found by PGD against
             VPD's own answers at every position of a panel of 128 training texts (VPD's evaluation: a source shared
             across a batch of 128 sequences, the KL averaged over all their positions, a uniform start, 20
             sign-gradient steps of 0.1), the largest over the pool (kl_universal_bits).
A pattern tuned to one input can exploit interference noise in circuitry the computation does not use, and then breaks
VPD's own answer too (91.7 bits on text9000, against 101.1 for naming nothing): VPD's appendix A.3.4 shares its
adversary across a batch of inputs for this reason, and so do these patterns.

The bar is VPD's answer's mean kl_bits on training texts outside the panels. prune() finds a circuit for one prediction:
from VPD's answer (causal importance > 0 at every position) it removes what cannot reach the target, then removes
(subcomponent, position) pairs while kl_bits stays within the bar, testing removals in batches.

  native.py universal          the pool of universal patterns -> texts/universal.pt
  native.py bar --n 200        VPD's answer's mean kl_bits on N training texts after the panels -> texts/bar.json
  native.py teach --split train --n N [--offset K --stride S]   prune() -> texts/circuits[_heldout]/<id>.py, .json
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
PGD_STEPS, PGD_STEP = 20, 0.1  # VPD's evaluation PGD (run s-55ea3f9b's PGDReconLoss)
RANDOM = 8  # random draws per score
POOL, PANEL = 4, 128  # universal patterns, and the training texts each is found on (VPD's evaluation batch)
CHUNK = 32  # panel texts per forward pass
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


@lru_cache(None)
def model(dev: str):
    """vpd4l with VPD's decomposition installed, each site's remainder cached."""
    import vpd_model

    target = vpd_model.load_target(dev)
    vpd = vpd_model.load_vpd(target, dev)
    for n in vpd.names:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(_forward_cached, st)
    return target, vpd


class Graph:
    """One answer's computational graph on one text. nodes: (weight matrix, position, subcomponent) triples; parents:
    reader node -> the writer nodes whose outputs it reads; out: the writer nodes the prediction at the targets reads.
    complete=True instead means every connection the model has between the graph's nodes (and from each of its
    residual writers at a target to the prediction): the graph claims nodes but no structure."""

    def __init__(self, nodes=(), parents=None, out=None, complete: bool = False, masks: dict | None = None):
        self.masks = masks  # a complete graph's nodes as weight matrix -> bool [T, C], in place of `nodes`
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


RESID_WRITERS = ("o_proj", "down_proj")
RESID_READERS = ("q_proj", "k_proj", "v_proj", "c_fc")


def stage(layer: int, kind: str) -> float:
    """A residual writer's place in its position's residual stream: attention output of layer l at l + 0.5, MLP
    output at l + 1; a reader reads what was written before its own place (q, k, v at l, MLP input at l + 0.5)."""
    return layer + (0.5 if kind in ("o_proj", "c_fc") else (1.0 if kind == "down_proj" else 0.0))


def connects(writer: tuple, reader: tuple | None, targets: list[int]) -> bool:
    """Whether the model connects a writer node to a reader node (None: the prediction at the targets): the residual
    stream at one position (an attention or MLP output into a later query, key, value or MLP input, or into the
    prediction), attention (a value at a position into the same layer's attention output at that or a later
    position), or one MLP (an MLP input into the same MLP's output). Nodes are (weight matrix name, position, index)."""
    wt, wl, wk = writer[1], int(writer[0].split(".")[1]), writer[0].split(".")[-1]
    if reader is None:
        return wk in RESID_WRITERS and wt in targets
    rt, rl, rk = reader[1], int(reader[0].split(".")[1]), reader[0].split(".")[-1]
    if wk in RESID_WRITERS and rk in RESID_READERS:
        return wt == rt and stage(wl, wk) <= stage(rl, rk)
    if wk == "v_proj" and rk == "o_proj":
        return wl == rl and wt <= rt
    if wk == "c_fc" and rk == "down_proj":
        return wl == rl and wt == rt
    return False


def all_edges(nodes, targets: list[int]) -> tuple[dict, list]:
    """Every connection the model has between `nodes`: (parents, out)."""
    by = {}
    for nd in nodes:
        by.setdefault(nd[0].split(".")[-1], []).append(nd)
    parents = {}
    for r in nodes:
        rk = r[0].split(".")[-1]
        cand = by.get("o_proj", []) + by.get("down_proj", []) if rk in RESID_READERS else (by.get("v_proj", []) if rk == "o_proj" else (by.get("c_fc", []) if rk == "down_proj" else []))
        ws = [w for w in cand if connects(w, r, targets)]
        if ws:
            parents[r] = ws
    out = [w for w in by.get("o_proj", []) + by.get("down_proj", []) if connects(w, None, targets)]
    return parents, out


class Native:
    """Runs and scores graphs on one device."""

    def __init__(self, dev: str | None = None, universal: Path | None = TEXTS / "universal.pt"):
        self.dev = dev or device()
        self.target, self.vpd = model(self.dev)
        self.names = self.vpd.names
        self.C = self.vpd.C
        self.pool = None
        if universal is not None and Path(universal).exists():
            raw = torch.load(universal, map_location=self.dev)
            self.pool = {"parts": raw["parts"], "rest": raw["rest"]}  # matrix -> [POOL, T, C], [POOL, T]

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
    def vpd_answer(self, ids: list[int]) -> Graph:
        """VPD's answer: at every position, the subcomponents whose causal importance there is above zero; complete."""
        _, ci = self.vpd.target_and_ci(torch.tensor([ids], device=self.dev))
        return Graph(masks={n: ci[n][0] > 0 for n in self.names})

    # ---- running a batch of graphs under one ablation pattern

    @torch.no_grad()
    def reference(self, ids: list[list[int]], targets: list[int]) -> torch.Tensor:
        """log p of the model at the targets of each sequence [B, len(targets), V]."""
        logits = self.vpd.target_forward(torch.tensor(ids, device=self.dev))[:, targets]
        return torch.log_softmax(logits.float(), -1)

    def _plan(self, graphs: list[Graph], T: int) -> dict:
        """Index tensors for a batch of graphs: per weight matrix the [B, T, C] node mask, and for the graphs that are
        not complete, the reader nodes with parents, the writer nodes with children, and the edges between them."""
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
        readers, writers, edges, out = {}, {}, {}, {}

        def row(table, node, b):
            rows = table.setdefault(node[0], {})
            return rows.setdefault((b, node[1], node[2]), len(rows))

        for b, g in enumerate(graphs):
            if g.complete:
                continue
            for r, ws in g.parents.items():
                ri = row(readers, r, b)
                for w in ws:
                    edges.setdefault((r[0], w[0]), []).append((ri, row(writers, w, b)))
            for w in g.out:
                out.setdefault(w[0], []).append((b, w[1], row(writers, w, b)))

        def cols(rows):
            keys = sorted(rows, key=rows.get)
            return {k: torch.tensor([x[k] for x in keys], device=dev) for k in (0, 1, 2)}

        return {"G": G, "complete": torch.tensor([g.complete for g in graphs], device=dev)[:, None, None],
                "readers": {n: cols(r) for n, r in readers.items()}, "writers": {n: cols(w) for n, w in writers.items()},
                "edges": {k: (torch.tensor([e[0] for e in v], device=dev), torch.tensor([e[1] for e in v], device=dev)) for k, v in edges.items()},
                "out": {n: tuple(torch.tensor([e[i] for e in v], device=dev) for i in range(3)) for n, v in out.items()}}

    def run(self, ids: list[int], targets: list[int], plan: dict, u: dict, ur: dict) -> torch.Tensor:
        """log q at the targets [B, len(targets), V] of each graph under one ablation pattern: u[matrix] [S, C] and
        ur[matrix] [S] in [0, 1] with S = T or 1 (the same at every position). A subcomponent outside a graph runs at
        mask u; a graph node at 1. An output reaches a reader through a declared edge in full and through every other
        connection scaled by its writer's u (so a graph's node outside every edge still writes at u to readers it does
        not declare). A complete graph's nodes read each other in full. The token embedding, which VPD does not
        decompose, always enters in full; remainders W - V U run at ur. With u = 1 everywhere this is the model."""
        import vpd_model

        tg, dev = self.target, self.dev
        G, comp, R, Wr, E, O = plan["G"], plan["complete"], plan["readers"], plan["writers"], plan["edges"], plan["out"]
        B = next(iter(G.values())).shape[0]
        T = len(ids)
        H, hd, eps = tg.n_head, tg.hd, tg.eps
        rms, gelu = vpd_model.rms, vpd_model.gelu_tanh

        def scale(n):  # [B or 1, T, C] mask of every subcomponent: 1 in the graph, u outside
            return torch.where(G[n], 1.0, u[n].expand(T, -1)[None])

        def uat(n, t, c):  # u at given positions and indices
            return u[n][t if u[n].shape[0] > 1 else torch.zeros_like(t), c]

        def rest(n, z):
            st = tg.site(n)
            return ur[n].expand(T)[None, :, None] * (z @ st.delta_T)

        emb = tg.wte[torch.tensor(ids, device=dev)]
        x = emb[None].expand(B, T, -1).clone()  # every output at its scale
        xg = x.clone()  # complete graphs: their nodes' outputs in full
        acts = {}

        def topup(n, width, contrib):  # sum over n's reader nodes of their declared parents' extra (1 - u) share
            out = torch.zeros(len(R[n][0]), width, device=dev)
            for (rn, wn), (ri, wi) in E.items():
                if rn == n:
                    out.index_add_(0, ri, contrib(wn, wi))
            return out

        def resid_contrib(wn, wi):
            b, t, c = Wr[wn][0][wi], Wr[wn][1][wi], Wr[wn][2][wi]
            return ((1 - uat(wn, t, c)) * acts[wn][b, t, c])[:, None] * tg.site(wn).U[c]

        def read_resid(n, norm, z, zg, xs):
            """Activations [B, T, C] of a matrix reading the residual stream: from z (every output at its scale) outside
            graphs, zg inside complete graphs, and for a graph's reader node the stream plus its parents' full outputs."""
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
                acts[nm[k]] = read_resid(nm[k], n1, z, zg, x)
            qv = (acts[nm["q_proj"]] * scale(nm["q_proj"])) @ tg.site(nm["q_proj"]).U + rest(nm["q_proj"], z)
            kv = (acts[nm["k_proj"]] * scale(nm["k_proj"])) @ tg.site(nm["k_proj"]).U + rest(nm["k_proj"], z)
            Uv = tg.site(nm["v_proj"]).U
            vs = (acts[nm["v_proj"]] * u[nm["v_proj"]].expand(T, -1)[None]) @ Uv + rest(nm["v_proj"], z)
            vg = (acts[nm["v_proj"]] * scale(nm["v_proj"])) @ Uv + rest(nm["v_proj"], z)
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
                    wb, wt, wc = Wr[wn][0][wi], Wr[wn][1][wi], Wr[wn][2][wi]
                    share = ((1 - uat(wn, wt, wc)) * acts[wn][wb, wt, wc])  # [E]
                    pat = P[wb, :, t[ri], wt]  # [E, H]: how much the reader's position attends to the writer's
                    extra.index_add_(0, ri, share[:, None] * (pat.repeat_interleave(hd, dim=1) * Uv[wc]))
                Ao = Ao.index_put((b, t, c), ((att_s[b, t] + extra) * Vo.T[c]).sum(-1))
            acts[no] = Ao
            Uo = tg.site(no).U
            ro = rest(no, att_s)
            x = x + (Ao * u[no].expand(T, -1)[None]) @ Uo + ro
            xg = xg + (Ao * scale(no)) @ Uo + ro
            z2, zg2 = rms(x, n2, eps), rms(xg, n2, eps)
            nf, nd = nm["c_fc"], nm["down_proj"]
            acts[nf] = read_resid(nf, n2, z2, zg2, x)
            Uf = tg.site(nf).U
            rf = rest(nf, z2)
            pre_s = (acts[nf] * u[nf].expand(T, -1)[None]) @ Uf + rf
            pre_g = (acts[nf] * scale(nf)) @ Uf + rf
            Vd = tg.site(nd).V
            Ad = torch.where(G[nd] & comp, gelu(pre_g) @ Vd, gelu(pre_s) @ Vd)
            if nd in R:
                b, t, c = R[nd][0], R[nd][1], R[nd][2]

                def mlp_contrib(wn, wi):
                    wb, wt, wc = Wr[wn][0][wi], Wr[wn][1][wi], Wr[wn][2][wi]
                    return ((1 - uat(wn, wt, wc)) * acts[wn][wb, wt, wc])[:, None] * Uf[wc]
                Ad = Ad.index_put((b, t, c), (gelu(pre_s[b, t] + topup(nd, Uf.shape[1], mlp_contrib)) * Vd.T[c]).sum(-1))
            acts[nd] = Ad
            Ud = tg.site(nd).U
            rd = rest(nd, gelu(pre_s))
            x = x + (Ad * u[nd].expand(T, -1)[None]) @ Ud + rd
            xg = xg + (Ad * scale(nd)) @ Ud + rd
        zt = torch.where(comp[:, :, :1].expand(B, 1, 1), xg[:, targets], x[:, targets])
        for wn, (b, t, wi) in O.items():
            pos = torch.tensor([targets.index(int(a)) for a in t.tolist()], device=dev)
            zt = zt.index_put((b, pos), resid_contrib(wn, wi), accumulate=True)
        return torch.log_softmax((rms(zt, tg.ln_f, eps) @ tg.wte.T).float(), -1)

    # ---- the faithfulness tests

    def patterns(self, T: int, seed: int) -> list[tuple[dict, dict]]:
        """The ablation patterns of a score: deletion, RANDOM uniform draws of `seed`, and the universal pool."""
        zero = ({n: torch.zeros(1, self.C[n], device=self.dev) for n in self.names}, {n: torch.zeros(1, device=self.dev) for n in self.names})
        g = torch.Generator(device="cpu").manual_seed(seed)
        rand = [({n: torch.rand(T, self.C[n], generator=g).to(self.dev) for n in self.names},
                 {n: torch.rand(T, generator=g).to(self.dev) for n in self.names}) for _ in range(RANDOM)]
        uni = []
        if self.pool is not None:
            P = next(iter(self.pool["rest"].values())).shape[0]
            uni = [({n: self.pool["parts"][n][j][:T] for n in self.names}, {n: self.pool["rest"][n][j][:T] for n in self.names}) for j in range(P)]
        return [zero] + rand + uni

    @torch.no_grad()
    def score(self, ids: list[int], targets: list[int], graphs: list[Graph], seed: int = 0, logp: torch.Tensor | None = None) -> list[dict]:
        """{"kl_bits", "kl_deleted_bits", "kl_random_bits", "kl_universal_bits", "nodes", "edges", "size"} per graph."""
        B, T = len(graphs), len(ids)
        if B == 0:
            return []
        logp = self.reference([ids], targets) if logp is None else logp
        plan = self._plan(graphs, T)
        kls = []
        for u, ur in self.patterns(T, seed):
            logq = self.run(ids, targets, plan, u, ur)
            kls.append((logp.exp() * (logp - logq)).sum(-1).sum(-1) / LN2)
        kls = torch.stack(kls)  # [patterns, B]
        deleted, random = kls[0], kls[1:1 + RANDOM].mean(0)
        universal = kls[1 + RANDOM:].max(0).values if len(kls) > 1 + RANDOM else torch.zeros_like(deleted)
        total = torch.stack([deleted, random, universal]).max(0).values
        return [{"kl_bits": float(total[b]), "kl_deleted_bits": float(deleted[b]), "kl_random_bits": float(random[b]),
                 "kl_universal_bits": float(universal[b]), "nodes": graphs[b].count(),
                 "edges": None if graphs[b].complete else graphs[b].edges(), "size": None if graphs[b].complete else graphs[b].size()} for b in range(B)]

    def find_universal(self, texts: list[tuple[list[int], list[int]]], seed: int) -> tuple[dict, dict, float]:
        """One universal pattern: a mask per position and subcomponent (and per position and remainder), shared by every
        text, maximizing the KL of VPD's answers averaged over every position of `texts` (sequences of one length) by
        PGD from a uniform start; the step with the largest mean KL is kept."""
        T = len(texts[0][0])
        everywhere = list(range(T))
        chunks = []
        for k in range(0, len(texts), CHUNK):
            part = texts[k:k + CHUNK]
            chunks.append([(i, self.reference([i], everywhere), self._plan([self.vpd_answer(i)], T)) for i, _ in part])
        g = torch.Generator(device="cpu").manual_seed(seed)
        u = {n: torch.rand(T, self.C[n], generator=g).to(self.dev) for n in self.names}
        ur = {n: torch.rand(T, generator=g).to(self.dev) for n in self.names}
        best, best_u, best_ur = -1.0, None, None
        for k in range(PGD_STEPS + 1):
            total, grads = 0.0, None
            for n in self.names:
                u[n].requires_grad_(True)
                ur[n].requires_grad_(True)
            for chunk in chunks:
                with torch.enable_grad():
                    kl = sum(((lp.exp() * (lp - self.run(i, everywhere, plan, u, ur))).sum() / LN2) for i, lp, plan in chunk) / (len(texts) * T)
                total += float(kl)
                if k < PGD_STEPS:
                    gs = torch.autograd.grad(kl, [u[n] for n in self.names] + [ur[n] for n in self.names])
                    grads = list(gs) if grads is None else [a + b for a, b in zip(grads, gs)]
            if total > best:
                best, best_u, best_ur = total, {n: u[n].detach().clone() for n in self.names}, {n: ur[n].detach().clone() for n in self.names}
            if k == PGD_STEPS:
                break
            with torch.no_grad():
                for n, gr in zip(self.names, grads[: len(self.names)]):
                    u[n] = (u[n] + PGD_STEP * gr.sign()).clamp(0, 1)
                for n, gr in zip(self.names, grads[len(self.names):]):
                    ur[n] = (ur[n] + PGD_STEP * gr.sign()).clamp(0, 1)
        return best_u, best_ur, best

    # ---- one prediction's graph

    def everything(self, T: int) -> Graph:
        """Every subcomponent at every position, complete: the model itself."""
        return Graph(masks={n: torch.ones(T, self.C[n], dtype=torch.bool, device=self.dev) for n in self.names})

    def reachable(self, g: Graph, targets: list[int]) -> Graph:
        """g's nodes (a mask graph) that can affect the targets: none after the last target, and before the first
        target none of the last layer's other than key and value (their outputs reach no later position's input)."""
        last, first, final = self.target.n_layer - 1, min(targets), max(targets)
        masks = {}
        for n in self.names:
            m = g.masks[n].clone()
            m[final + 1:] = False
            layer, kind = int(n.split(".")[1]), n.split(".")[-1]
            if layer == last and kind not in ("k_proj", "v_proj"):
                m[:first] = False
            masks[n] = m
        return Graph(masks=masks)

    def _ladder(self, items: list, ok, batch: int, log=None, what: str = "") -> list:
        """Greedy removal from `items` (each a removable unit, in the order to try): each round tests, in one batch,
        dropping the first k untested units for k = n/2, n/4, ..., 2 and each of the next units alone; it keeps the
        largest prefix drop that passes ok, else the first single that does, and marks failed singles as kept. ok:
        a list of candidate remainders -> a list of (passes, score). Returns the remaining units and their score."""
        kept, rounds, score = set(), 0, None
        current = list(items)
        while True:
            rounds += 1
            order = [x for x in current if x not in kept]
            if not order:
                break
            ladder, k = [], len(order) // 2
            while k >= 2:
                ladder.append(k)
                k //= 2
            singles = order[: max(1, batch - len(ladder))]
            drops = [set(order[:k]) for k in ladder] + [{x} for x in singles]
            results = ok([[x for x in current if x not in d] for d in drops])
            passing = [j for j, (p, _) in enumerate(results) if p]
            prefix = [j for j in passing if j < len(ladder)]
            if prefix:
                j = prefix[0]
            else:
                kept |= {singles[j - len(ladder)] for j in range(len(ladder), len(drops)) if j not in passing}
                single = [j for j in passing if j >= len(ladder)]
                if not single:
                    continue
                j = single[0]
            current = [x for x in current if x not in drops[j]]
            score = results[j][1]
            if log:
                log(f"{what} round {rounds}: dropped {len(drops[j])}, {len(current)} left, {len(kept)} kept")
        return current, score

    def prune(self, ids: list[int], targets: list[int], bar: float, batch: int = 32, seed: int = 0, log=None) -> tuple[Graph, dict]:
        """(one prediction's graph, its score). Nodes first: every subcomponent at every position that can reach the
        targets (the model itself), then greedy removal of nodes (as a complete graph) within the bar, tried in order of
        VPD's causal importance there (lowest first; only an order, every removal is measured); then edges: every
        connection the model has between the kept nodes, removed greedily within the bar (nodes left without edges are
        dropped). A start over the bar comes back as it is."""
        logp = self.reference([ids], targets)
        start = self.reachable(self.everything(len(ids)), targets)
        s = self.score(ids, targets, [start], seed, logp)[0]
        if s["kl_bits"] > bar:
            return start, s
        nodes = self._prune_nodes(ids, targets, start, bar, batch, seed, logp, log)
        parents, out = all_edges(nodes, targets)
        edge_list = [(r, w) for r, ws in parents.items() for w in ws] + [(None, w) for w in out]

        def build(es):
            par, o = {}, []
            for r, w in es:
                if r is None:
                    o.append(w)
                else:
                    par.setdefault(r, []).append(w)
            used = {w for _, w in es} | {r for r, _ in es if r is not None}
            return Graph(used, par, o)

        def ok_edges(cands):
            res = self.score(ids, targets, [build(c) for c in cands], seed, logp)
            return [(r["kl_bits"] <= bar, r) for r in res]

        full = build(edge_list)
        s_full = self.score(ids, targets, [full], seed, logp)[0]
        if s_full["kl_bits"] > bar:  # the kept nodes need connections beyond the model's own between them
            g = Graph(nodes, complete=True)
            return g, self.score(ids, targets, [g], seed, logp)[0]
        es, score = self._ladder(edge_list, ok_edges, batch, log, "edges")
        return build(es), score or s_full

    def _prune_nodes(self, ids, targets, start: Graph, bar: float, batch: int, seed: int, logp, log=None) -> set:
        """The node stage of prune() on masks: the ladder of _ladder over the nodes of `start` in _node_order."""
        names = self.names
        order = self._node_order(ids, targets, start)  # (matrix index, position, index) tensors, lowest first
        mi, ti, ci = order
        N = len(mi)
        alive = torch.ones(N, dtype=torch.bool, device=self.dev)
        kept = torch.zeros(N, dtype=torch.bool, device=self.dev)
        T = len(ids)

        def masks_of(a):
            out = {n: torch.zeros(T, self.C[n], dtype=torch.bool, device=self.dev) for n in names}
            for j, n in enumerate(names):
                sel = a & (mi == j)
                out[n][ti[sel], ci[sel]] = True
            return out

        rounds = 0
        while True:
            rounds += 1
            untested = (alive & ~kept).nonzero().flatten()
            if len(untested) == 0:
                break
            ladder, k = [], len(untested) // 2
            while k >= 2:
                ladder.append(k)
                k //= 2
            singles = untested[: max(1, batch - len(ladder))]
            cands = []
            for k in ladder:
                a = alive.clone()
                a[untested[:k]] = False
                cands.append(a)
            for j in singles:
                a = alive.clone()
                a[j] = False
                cands.append(a)
            res = self.score(ids, targets, [Graph(masks=masks_of(a)) for a in cands], seed, logp)
            passing = [j for j, r in enumerate(res) if r["kl_bits"] <= bar]
            prefix = [j for j in passing if j < len(ladder)]
            if prefix:
                j = prefix[0]
            else:
                failed = [int(singles[j - len(ladder)]) for j in range(len(ladder), len(cands)) if j not in passing]
                kept[failed] = True
                single = [j for j in passing if j >= len(ladder)]
                if not single:
                    continue
                j = single[0]
            alive = cands[j]
            if log:
                log(f"nodes round {rounds}: {int(alive.sum())} left, {int(kept.sum())} kept, kl {res[j]['kl_bits']:.4f}")
        idx = alive.nonzero().flatten()
        return {(names[int(mi[i])], int(ti[i]), int(ci[i])) for i in idx}

    @torch.no_grad()
    def _node_order(self, ids: list[int], targets: list[int], g: Graph) -> tuple:
        """g's nodes (a mask graph) as (matrix index, position, index) tensors ordered by VPD's causal importance at
        their position, lowest first, ties by the size of what they write (|activation| |u|) on the model's run: the
        order removals are tried in (only an order; every removal is measured)."""
        ids_b = torch.tensor([ids], device=self.dev)
        _, ci = self.vpd.target_and_ci(ids_b)
        for n in self.names:
            self.target.site(n).cache_input = True
        self.target(ids_b)
        mis, tis, cis, keys = [], [], [], []
        for j, n in enumerate(self.names):
            st = self.target.site(n)
            size = ((st.last_input[0] @ st.V).abs() * st.U.norm(dim=1)).float()
            t, c = g.masks[n].nonzero(as_tuple=True)
            mis.append(torch.full_like(t, j))
            tis.append(t)
            cis.append(c)
            keys.append(ci[n][0][t, c].float() * 1e6 + size[t, c] / (1 + size.max()))
        self.vpd.clear()
        mi, ti, cc, key = torch.cat(mis), torch.cat(tis), torch.cat(cis), torch.cat(keys)
        o = key.argsort()
        return mi[o], ti[o], cc[o]


def program(g: Graph) -> str:
    """A graph as an answer: graph(tokens, targets) returning {(position, reader): its parents, "out": the
    prediction's parents}, parents at the reader's own position as one string of subcomponents, an attention output's
    parents (values) as {position: string}; nodes without parents but with children are not listed as readers."""
    sites = list(mech.SITES.values())
    codes = {v: k for k, v in mech.SITES.items()}

    def tok(nd):
        layer, kind = int(nd[0].split(".")[1]), nd[0].split(".")[-1]
        return f"<p:{layer}.{codes[kind]}.{nd[2]}>"

    def key(nd):
        layer, kind = int(nd[0].split(".")[1]), nd[0].split(".")[-1]
        return (nd[1], layer, sites.index(kind), nd[2])

    lines = []
    readers = set(g.parents) | {nd for nd in g.nodes if not any(nd in ws for ws in g.parents.values()) and nd not in g.out}
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
    return "def graph(tokens, targets):\n    return {\n" + "\n".join(lines) + "\n    }\n"


def tasks(split: str) -> list[Path]:
    """The split's task files in text order."""
    return [p for p in sorted((TEXTS / "vpd4l").glob("*.json"), key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())["split"] == split]


def text(p: Path) -> tuple[list[int], list[int]]:
    task = json.loads(p.read_text())["prompts"][0]
    return task["token_ids"], task["target_positions"]


def universal() -> dict:
    """POOL universal patterns, pattern j found on training texts j * PANEL ... (j + 1) * PANEL - 1."""
    nat = Native(universal=None)
    train = tasks("train")
    parts = {n: [] for n in nat.names}
    rest = {n: [] for n in nat.names}
    kls = []
    for j in range(POOL):
        u, ur, kl = nat.find_universal([text(p) for p in train[j * PANEL:(j + 1) * PANEL]], seed=j)
        for n in nat.names:
            parts[n].append(u[n])
            rest[n].append(ur[n])
        kls.append(kl)
        print(f"pattern {j}: VPD's answers' KL {kl:.3f} bits per position on its panel", flush=True)
    torch.save({"parts": {n: torch.stack(v) for n, v in parts.items()}, "rest": {n: torch.stack(v) for n, v in rest.items()},
                "panel_kl_bits": kls, "panel": PANEL}, TEXTS / "universal.pt")
    return {"panel_kl_bits": kls}


def bar(n: int) -> dict:
    """VPD's answer's mean kl_bits on n training texts after the panels."""
    nat = Native()
    rows = [nat.score(*text(p), [nat.vpd_answer(text(p)[0])])[0] for p in tasks("train")[POOL * PANEL:POOL * PANEL + n]]
    out = {k: sum(r[k] for r in rows) / len(rows) for k in ("kl_bits", "kl_deleted_bits", "kl_random_bits", "kl_universal_bits", "nodes")}
    out.update(texts=len(rows), random=RANDOM, pool=POOL, panel=PANEL)
    (TEXTS / "bar.json").write_text(json.dumps(out))
    return out


def teach(split: str, n: int, offset: int = 0, stride: int = 1) -> None:
    """prune() on the split's texts offset, offset + stride, ... of its first n -> texts/circuits[_heldout]/<id>.py and
    <id>.json (its score and VPD's answer's)."""
    nat = Native()
    b = json.loads((TEXTS / "bar.json").read_text())["kl_bits"]
    out = TEXTS / ("circuits" if split == "train" else "circuits_heldout")
    out.mkdir(exist_ok=True)
    for p in tasks(split)[:n][offset::stride]:
        if (out / f"{p.stem}.py").exists():
            continue
        ids, targets = text(p)
        t0 = time.time()
        full = nat.score(ids, targets, [nat.vpd_answer(ids)])[0]
        g, s = nat.prune(ids, targets, b)
        (out / f"{p.stem}.py").write_text(program(g) if not g.complete else "")
        (out / f"{p.stem}.json").write_text(json.dumps({"score": s, "vpd": full, "bar": b, "complete": g.complete, "seconds": round(time.time() - t0, 1)}))
        print(f"{p.stem}: VPD {full['nodes']} nodes kl {full['kl_bits']:.3f} -> {s['nodes']} nodes, {s['edges']} edges, kl {s['kl_bits']:.3f} (bar {b:.3f}), {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("universal")
    a = sub.add_parser("bar")
    a.add_argument("--n", type=int, default=200)
    t = sub.add_parser("teach")
    t.add_argument("--split", choices=("train", "heldout"), required=True)
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--offset", type=int, default=0)
    t.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "universal":
        print(json.dumps(universal()))
    elif args.cmd == "bar":
        print(json.dumps(bar(args.n)))
    else:
        teach(args.split, args.n, args.offset, args.stride)


if __name__ == "__main__":
    main()
