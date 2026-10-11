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

Faithfulness (Native.faithfulness), in bits: the mean KL of the model's next-token distribution at the target from the
graph's over experiments, one per changed prompt of the text (Native.changes: the token at one position, drawn
uniformly from the positions whose token the model predicts, 1 to the target, replaced by a draw from the model's own
prediction there; a draw can return the original token). The answer program runs on each changed prompt, so a general
rule can bind differently there. In each experiment every subcomponent of the graph is held at its value on the original
text with probability 1/2, and what lies outside the graph is removed or, with probability 1/2, set to independent
uniform levels (VPD's stochastic test); the graph runs on the changed prompt so, the whole model on the changed prompt
with the same subcomponents held at its own values on the text. So every experiment tests that nothing outside the
graph matters at any level, and holds test that the graph computes from its parts what the model computes from the
same parts (interchange interventions), whatever steps the answer groups them into. The verifier draws the experiments,
never the answer. A graph more confident than the model fails too, the KL being of the whole distribution. No test proves a
graph is the model's mechanism; this measures how far it is from the model on these experiments.

Description length (Graph.bits), in bits: the nodes listed, each one choice among the text's positions times the
model's 38,912 subcomponents, then each edge as its reader's and its writer's index in that list, and each edge into the
prediction as its writer's index.

Native.ordered is a search that bootstraps the oracle, not the method: integrated-gradients rankings of subcomponents
and then of the connections between the top ones, averaged over the changed prompts; the answer adds the ranked
connections in order in steps that double their number, each step keeping what computes something and lies on a path
to the prediction (live), and ends at the step with the lowest KL.

  native.py search --split train --n N [--offset K --stride S] [--out DIR]   -> DIR/<task>.py, .json
  native.py tidy DIR [--max-chars N]      every answer in DIR made live (live()), in place
  native.py ranked --split S [S ...] --n N [--top K]   -> texts/vpd_ranked/<task>.json, VPD's first K at the target
  native.py responses --split S [S ...] --n N [--positions K]   -> texts/vpd_responses/<task>.json (responses())
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "vpd_2951"))

import mech  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts"
CHANGES = 16  # changed prompts per text: draws of the average that defines faithfulness (they set its precision, not what it is)
MAX_CONNECTIONS = 150000  # the search's memory: the connections it ranks at once
IG_STEPS = 16  # the search's integration points along each ranking's path
ADV_STEPS, ADV_STEP = 20, 0.1  # Native.adversarial (evaluation only): VPD's headline setting is 20 steps shared across its batch
LN2 = math.log(2)


def free(dev: str) -> None:
    """Return cached device memory (the Mac's shared memory fills with MPS's cache during the search's rankings)."""
    if str(dev).startswith("mps"):
        torch.mps.empty_cache()


def device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _first_draw(seeds: np.ndarray) -> np.ndarray:
    """The 24 bits behind torch.rand(1, generator=torch.Generator(device="cpu").manual_seed(s)) for each seed s (uint64):
    the first output of mt19937 seeded with s's low 32 bits (torch's CPU generator), rand = bits / 2^24."""
    M = np.uint64(0xFFFFFFFF)
    s0 = np.asarray(seeds, dtype=np.uint64) & M
    st, s1 = s0, None
    for j in range(1, 398):
        st = (np.uint64(1812433253) * (st ^ (st >> np.uint64(30))) + np.uint64(j)) & M
        if j == 1:
            s1 = st
    y = (s0 & np.uint64(0x80000000)) | (s1 & np.uint64(0x7FFFFFFF))
    v = st ^ (y >> np.uint64(1)) ^ np.where((y & np.uint64(1)) == 1, np.uint64(0x9908B0DF), np.uint64(0))
    v ^= v >> np.uint64(11)
    v ^= (v << np.uint64(7)) & np.uint64(0x9D2C5680)
    v ^= (v << np.uint64(15)) & np.uint64(0xEFC60000)
    v ^= v >> np.uint64(18)
    return v & np.uint64(0xFFFFFF)


_DRAW_CHECKED: list = []


def below_half(seeds: np.ndarray) -> np.ndarray:
    """torch.rand(1, generator=torch.Generator(device="cpu").manual_seed(s)) < 0.5 for each seed s, without a generator
    per seed (_first_draw; checked against torch once per process)."""
    if not _DRAW_CHECKED:
        probe = [0, 1, 12345, (1 << 32) + 7, (1 << 62) - 1] + [(k * 2654435761) % (1 << 62) for k in range(1, 60)]
        mine = _first_draw(np.array(probe, dtype=np.uint64))
        for s, d in zip(probe, mine.tolist()):
            if float(torch.rand(1, generator=torch.Generator(device="cpu").manual_seed(s))) != d / float(1 << 24):
                raise RuntimeError("torch's CPU generator no longer matches _first_draw")
        _DRAW_CHECKED.append(True)
    return _first_draw(seeds) < np.uint64(1 << 23)


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


class Changed(list):
    """A changed prompt: its token ids, and .weight, its importance weight (Native.changes)."""

    def __init__(self, ids, weight: float):
        super().__init__(ids)
        self.weight = weight


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

    def bits(self, positions: int, subcomponents: int, targets: list[int]) -> float:
        """Description length, what a reader of the graph takes in: each node one choice among positions x
        subcomponents, each edge its reader's and writer's indices among the nodes, each edge into the prediction its
        writer's index. A complete graph keeps every connection the model has among its nodes (implied_edges) and pays
        for each like a stated one: claiming all of them is no shorter to read than listing them."""
        n = self.count()
        if n == 0:
            return 0.0
        nodes = n * math.log2(positions * subcomponents)
        inner, out = implied_edges(self.node_set(), targets) if self.complete else (sum(len(w) for w in self.parents.values()), len(self.out))
        return nodes + math.log2(n) * (2 * inner + out)


@lru_cache(None)
def _name_layer_kind(name: str) -> tuple[int, str]:
    return int(name.split(".")[1]), name.split(".")[-1]


def _layer_kind(nd) -> tuple[int, str]:
    return _name_layer_kind(nd[0])


def connects(writer: tuple, reader: tuple | None, targets: list[int]) -> bool:
    """Whether the model connects a writer node to a reader node (None: the prediction at the targets); nodes are
    (matrix name, position, index). mech.connects holds the rule."""
    wl, wk = _layer_kind(writer)
    if reader is None:
        return mech.connects(wl, wk, writer[1], None, None, None, targets)
    rl, rk = _layer_kind(reader)
    return mech.connects(wl, wk, writer[1], rl, rk, reader[1], targets)


def implied_edges(nodes, targets: list[int]) -> tuple[int, int]:
    """(connections among the nodes, connections into the prediction) the model has (mech.connects' rule, as
    _candidates lists them), counted without listing them: residual readers from residual writers before them at their
    position, attention outputs from values of their layer at their position or earlier, MLP outputs from their MLP's
    input at their position; the prediction from residual writers at the targets."""
    import bisect

    resid, values, inputs = {}, {}, {}
    for nd in nodes:
        layer, kind = _layer_kind(nd)
        if kind in mech.RESID_WRITERS:
            resid.setdefault(nd[1], []).append(mech.stage(layer, kind))
        elif kind == "v_proj":
            values.setdefault(layer, []).append(nd[1])
        elif kind == "c_fc":
            inputs[(layer, nd[1])] = inputs.get((layer, nd[1]), 0) + 1
    for v in (*resid.values(), *values.values()):
        v.sort()
    inner = 0
    for nd in nodes:
        layer, kind = _layer_kind(nd)
        if kind in mech.RESID_READERS:
            inner += bisect.bisect_right(resid.get(nd[1], []), mech.stage(layer, kind))
        elif kind == "o_proj":
            inner += bisect.bisect_right(values.get(layer, []), nd[1])
        elif kind == "down_proj":
            inner += inputs.get((layer, nd[1]), 0)
    return inner, sum(len(resid.get(t, [])) for t in set(targets))


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
    """How many connections all_edges would give (_candidates' lists counted, not built)."""
    from bisect import bisect_right

    resid, values, inputs, readers, out = {}, {}, {}, {}, 0
    for nd in nodes:
        layer, kind = _layer_kind(nd)
        if kind in mech.RESID_WRITERS:
            resid.setdefault(nd[1], []).append(mech.stage(layer, kind))
            out += nd[1] in targets
        elif kind == "v_proj":
            values.setdefault(layer, []).append(nd[1])
        elif kind == "c_fc":
            inputs[(layer, nd[1])] = inputs.get((layer, nd[1]), 0) + 1
        if kind in mech.RESID_READERS or kind in ("o_proj", "down_proj"):
            readers[nd] = (layer, kind)
    for v in list(resid.values()) + list(values.values()):
        v.sort()
    n = out
    for r, (layer, kind) in readers.items():
        if kind in mech.RESID_READERS:
            n += bisect_right(resid.get(r[1], ()), mech.stage(layer, kind))
        elif kind == "o_proj":
            n += bisect_right(values.get(layer, ()), r[1])
        else:
            n += inputs.get((layer, r[1]), 0)
    return n


class Native:
    """Runs, scores and finds graphs on one device."""

    def __init__(self, dev: str | None = None):
        self.dev = dev or device()
        self.target, self.vpd = model(self.dev)
        self.names = self.vpd.names
        self.C = self.vpd.C
        self._consts: dict = {}  # zero and one ablations, full plans, causal masks: built once, never written
        self._rec: tuple = (None, None, None)  # the last text's whole-model run: (ids, (activations, states), states per batch size)

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

    @property
    def has_importance(self) -> bool:
        """Whether VPD's causal-importance network is loaded (a pod has only the subcomponents, UV)."""
        return self.vpd.ci_fn is not None

    def vpd_answer(self, ids: list[int]) -> Graph:
        """VPD's own answer, for comparison: at every position the subcomponents whose causal importance there is above
        zero, complete."""
        return Graph(masks={n: m > 0 for n, m in self.importance(ids).items()})

    def _vpd_rank(self, ids: list[int], targets: list[int]) -> dict:
        """Each subcomponent with causal importance above zero -> its place in vpd_steps' order (0 first)."""
        return {nd: place for place, nd in enumerate(self._vpd_order(ids, targets))}

    def _vpd_order(self, ids: list[int], targets: list[int]) -> list:
        """The subcomponents with causal importance above zero, (matrix, position, index), in vpd_steps' order."""
        ci = self.importance(ids)
        writes = self.contributions(ids, targets)
        flat = torch.cat([m.flatten() for m in ci.values()])
        at_target = torch.cat([torch.isin(torch.arange(m.shape[0], device=m.device), torch.tensor(targets, device=m.device))[:, None].expand_as(m).flatten() for m in ci.values()])
        rank = torch.cat([(ci[n] * writes[n]).flatten() for n in ci])
        rank = rank + at_target * (rank.max() + 1)
        order = rank.argsort(descending=True)[: int((flat > 0).sum())].cpu().numpy().astype(np.int64)
        names, sizes = list(ci), [ci[n].numel() for n in ci]
        starts = np.cumsum([0] + sizes[:-1]).astype(np.int64)
        j = np.searchsorted(starts, order, side="right") - 1  # the matrix of each flat index
        off = order - starts[j]
        C = np.array([ci[n].shape[1] for n in names], dtype=np.int64)[j]
        return [(names[a], t, c) for a, t, c in zip(j.tolist(), (off // C).tolist(), (off % C).tolist())]

    def vpd_steps(self, ids: list[int], targets: list[int]) -> list[Graph]:
        """VPD's answer as steps, most important first: its subcomponents with causal importance above zero, ranked by
        importance x how much each writes there on the text (native.contributions; many are tied at importance 1),
        those at the predicted positions first (VPD's mask covers every position's prediction; this one is computed
        mostly at its own position), over all matrices, the k-th step the top 2^(k-1) of them (the last step all), each complete. Its
        first steps are VPD's own short answers, comparable with any answer's prefixes."""
        ci = self.importance(ids)
        writes = self.contributions(ids, targets)
        flat = torch.cat([m.flatten() for m in ci.values()])
        at_target = torch.cat([torch.isin(torch.arange(m.shape[0], device=m.device), torch.tensor(targets, device=m.device))[:, None].expand_as(m).flatten() for m in ci.values()])
        rank = torch.cat([(ci[n] * writes[n]).flatten() for n in ci])
        rank = rank + at_target * (rank.max() + 1)  # the predicted position's subcomponents first, each group by its own order
        order = rank.argsort(descending=True)[: int((flat > 0).sum())]
        sizes, n = [], 1
        while n < len(order):
            sizes.append(n)
            n *= 2
        sizes.append(len(order))
        names, offsets, o = list(ci), [], 0
        for name in names:
            offsets.append(o)
            o += ci[name].numel()
        steps = []
        for k in sizes:
            keep = torch.zeros_like(flat, dtype=torch.bool)
            keep[order[:k]] = True
            steps.append(Graph(masks={name: keep[off:off + ci[name].numel()].view_as(ci[name]) for name, off in zip(names, offsets)}))
        return steps

    def everything(self, T: int) -> Graph:
        """Every subcomponent at every position, complete: the model itself."""
        g = Graph(masks={n: torch.ones(T, self.C[n], dtype=torch.bool, device=self.dev) for n in self.names})
        g.full = True  # _plan: every node, no mask to copy
        return g

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

    def _ints(self, lists: list) -> list[torch.Tensor]:
        """Lists (or arrays) of ints -> int64 tensors on the device, sent in one transfer (slices of one buffer)."""
        sizes = [len(x) for x in lists]
        flat = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=sum(sizes))
        buf = torch.from_numpy(flat).to(self.dev)
        out, o = [], 0
        for s in sizes:
            out.append(buf[o:o + s])
            o += s
        return out

    def _const(self, key, make):
        if key not in self._consts:
            self._consts[key] = make()
        return self._consts[key]

    def _full_plan(self, T: int, B: int) -> dict:
        """_plan of B copies of everything(T), kept (nothing writes a plan's tensors)."""
        return self._const(("full", T, B), lambda: self._plan([self.everything(T)] * B, T))

    def _plan(self, graphs: list[Graph], T: int) -> dict:
        """Index tensors for a batch of graphs: per matrix the [B, T, C] node mask, and for the graphs that are not
        complete, the reader nodes with parents, the writer nodes with children, and the edges between them (in the
        order of each graph's parents lists, then its out list). A batch of complete graphs and others is planned as
        the two batches apart ("split"; run() runs them so: only complete graphs need the stream of their nodes'
        outputs in full, and rows are computed independently)."""
        dev, B = self.dev, len(graphs)
        complete = [bool(g.complete) for g in graphs]
        if 0 < sum(complete) < B:
            groups = [[b for b in range(B) if not complete[b]], [b for b in range(B) if complete[b]]]
            return {"B": B, "split": [(torch.tensor(idx, device=dev), self._plan([graphs[b] for b in idx], T)) for idx in groups]}
        full = all(getattr(g, "full", False) for g in graphs)
        stacked = not full and all(g.masks is not None for g in graphs)
        if full:
            G = {n: self._const(("ones", T, n), lambda n=n: torch.ones(1, T, self.C[n], dtype=torch.bool, device=dev)).expand(B, -1, -1) for n in self.names}
        elif stacked:
            G = {n: torch.stack([g.masks[n] for g in graphs]) for n in self.names}
        else:
            G = {n: torch.zeros(B, T, self.C[n], dtype=torch.bool, device=dev) for n in self.names}
        at = {}  # matrix -> node (graph, position, index) lists
        for b, g in enumerate(graphs):
            if g.masks is not None:
                if not (full or stacked):
                    for n in self.names:
                        G[n][b] = g.masks[n]
                continue
            for n, t, c in g.nodes:
                lst = at.setdefault(n, ([], [], []))
                lst[0].append(b)
                lst[1].append(t)
                lst[2].append(c)
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
        lists = [[int(c) for c in complete]]  # every index list, sent at once
        for n in at:
            lists += list(at[n])
        for table in (readers, writers):
            for rows in table.values():
                keys = list(rows)  # in order of their row numbers
                lists += [[k[j] for k in keys] for j in (0, 1, 2)]
        for v in edges.values():
            lists += [[e[0] for e in v], [e[1] for e in v]]
        for v in out.values():
            lists += [[e[j] for e in v] for j in (0, 1, 2)]
        it = iter(self._ints(lists))
        comp = next(it).bool()[:, None, None]
        for n in at:
            bi, ti, ci = next(it), next(it), next(it)
            G[n][bi, ti, ci] = True
        R = {n: (next(it), next(it), next(it)) for n in readers}
        Wr = {n: (next(it), next(it), next(it)) for n in writers}
        E = {k: (next(it), next(it)) for k in edges}
        O = {n: (next(it), next(it), next(it)) for n in out}
        return {"B": B, "G": G, "complete": comp, "any_complete": any(complete), "all_complete": all(complete),
                "readers": R, "writers": Wr, "edges": E, "out": O, "out_t": {n: [e[1] for e in v] for n, v in out.items()}, "order": order,
                "readers_t": {n: [k[1] for k in rows] for n, rows in readers.items()}}

    def edge_weights(self, plan: dict, w: torch.Tensor) -> tuple[dict, dict]:
        """Per-edge strengths in plan order (a vector over every edge of every graph) -> run()'s ew and ow."""
        if "_order_idx" not in plan:
            src_e = {k: [] for k in plan["edges"]}
            src_o = {k: [] for k in plan["out"]}
            for j, (kind, key, _) in enumerate(plan["order"]):
                (src_e if kind == "edge" else src_o)[key].append(j)
            idx = self._ints(list(src_e.values()) + list(src_o.values()))
            plan["_order_idx"] = (dict(zip(src_e, idx[:len(src_e)])), dict(zip(src_o, idx[len(src_e):])))
        ie, io = plan["_order_idx"]
        return {k: w[v] for k, v in ie.items()}, {k: w[v] for k, v in io.items()}

    def run(self, ids: list[int], targets: list[int], plan: dict, u: dict, ur: dict, ew: dict | None = None, ow: dict | None = None,
            qk: dict | None = None, patch: dict | None = None, record: dict | None = None, keep=None, fill: dict | None = None,
            reuse: tuple | None = None) -> torch.Tensor:
        """log q at the targets [B, len(targets), V] of each graph under one ablation: u[matrix] [Bu, S, C] and
        ur[matrix] [Bu, S] in [0, 1], Bu = B or 1, S = T or 1. A subcomponent outside a graph runs at mask u; a graph
        node at 1. An output reaches a reader through a declared edge of strength e at e + (1 - e) u (e = 1 unless ew
        / ow give it) and through every other connection at its writer's u. A complete graph's nodes read each other
        in full. qk[matrix] [Bu, T, C] gives a query or key node a strength e: it enters the attention pattern at
        e + (1 - e) u (1 without qk). The token embedding always enters in full; remainders W - V U run at ur.
        patch[matrix] = (mask [B, T, C], values [B, T, C]) replaces those subcomponents' activations (an interchange);
        record, a dict, receives every matrix's activations [B, T, C] (keep: only those of these matrices).

        fill, a dict, receives the run's states at every position; reuse = (fill of a run of the same plan and
        ablation on other tokens base, base): the model is causal, so before the first position where ids differ
        from base every state is that run's (a patch there must give that run's values), and only the positions from
        there on are computed (_stream1)."""
        import vpd_model

        tg = self.target
        return torch.log_softmax((vpd_model.rms(self._stream(ids, targets, plan, u, ur, ew, ow, qk, patch, record, keep, fill, reuse), tg.ln_f, tg.eps)
                                  @ tg.wte.T).float(), -1)

    def _stream(self, ids, targets, plan, u, ur, ew, ow, qk, patch, record, keep, fill=None, reuse=None) -> torch.Tensor:
        """run()'s residual stream at the targets [B, len(targets), d], before the final norm; a split plan's two batches
        run apart and are put back in order."""
        if "split" not in plan:
            return self._stream1(ids, targets, plan, u, ur, ew, ow, qk, patch, record, keep, fill, reuse)
        if ew is not None or ow is not None or qk is not None:
            raise ValueError("edge and query/key strengths need a plan of complete graphs or of others, not both")
        B = plan["B"]

        def rows(d, idx):  # per-graph tensors cut to a batch (the same dict when every tensor is shared by the graphs)
            if all(v.shape[0] == 1 for v in d.values()):
                return d
            return {n: v[idx] if v.shape[0] > 1 else v for n, v in d.items()}

        zt, recs = None, []
        for k, (idx, sub) in enumerate(plan["split"]):
            pt = None if patch is None else {n: (m[idx], v[idx] if v.shape[0] > 1 else v) for n, (m, v) in patch.items()}
            rec = None if record is None else {}
            z = self._stream1(ids, targets, sub, rows(u, idx), rows(ur, idx), None, None, None, pt, rec, keep,
                              None if fill is None else fill.setdefault(k, {}), None if reuse is None else (reuse[0][k], reuse[1]))
            if zt is None:
                zt = torch.empty(B, *z.shape[1:], dtype=z.dtype, device=z.device)
            zt.index_copy_(0, idx, z)
            recs.append((idx, rec))
        if record is not None:
            for n in recs[0][1]:
                full = None
                for idx, rec in recs:
                    if full is None:
                        full = torch.empty(B, *rec[n].shape[1:], dtype=rec[n].dtype, device=rec[n].device)
                    full.index_copy_(0, idx, rec[n])
                record[n] = full
        return zt

    def _stream1(self, ids, targets, plan, u, ur, ew, ow, qk, patch, record, keep, fill=None, reuse=None) -> torch.Tensor:
        """_stream for a batch of complete graphs or of others. Two residual streams: x, every output at its scale, and
        xg, a complete graph's nodes' outputs in full. Only complete graphs read xg; when every graph is complete and
        each node's scale is its u (u = 1 everywhere, or no node held: plan "none") the two are the same numbers and one
        is computed. With u = 0 and the remainders off (the _zero() ablation) and no complete graph, nothing outside a
        graph writes to x: it stays the embedding, the same for every graph, and only the graphs' nodes and the
        queries and keys are computed; the numbers left out are exact zeros.

        With reuse, the products and the MLP run on the positions from s on only (s: the first that differs, moved
        earlier to keep at least 64 rows per product, whose rows then come out as in the whole batch). Attention reads
        the queries, keys and values before s, and the graph's writers their activations there, from the reused run's
        states; the norms and graph readers run on tensors of the whole sequence's shape, zeros before s, whose rows
        come out as in the whole run. So every number is the one the whole run gives."""
        import vpd_model

        tg, dev = self.target, self.dev
        G, comp, R, Wr, E, O = plan["G"], plan["complete"], plan["readers"], plan["writers"], plan["edges"], plan["out"]
        B = plan["B"]
        T = len(ids)
        H, hd, eps = tg.n_head, tg.hd, tg.eps
        rms, gelu = vpd_model.rms, vpd_model.gelu_tanh
        zu, zur = self._zero()
        no_rest = ur is zur  # remainders at 0: they add exact zeros
        empty = u is zu and no_rest and not plan["any_complete"]
        same = plan["all_complete"] and qk is None and (u is self._one()[0] or plan.get("none", False))
        both = plan["any_complete"] and not same
        s = 0
        if reuse is not None and not empty and record is None:
            base = reuse[1]
            s = next((p for p in range(min(len(base), T)) if ids[p] != base[p]), min(len(base), T))
            s = min(s, T - -(-64 // B), min(targets))
            s = max(s, 0)
        cache = reuse[0] if s > 0 else None
        if empty:
            fill = None

        def full(key, w):  # [B, T, ...]: the reused run's first s positions, then w (the positions from s)
            if cache is None:
                if fill is not None:
                    fill[key] = w
                return w
            c = cache[key]
            return torch.cat([c[:, :s].expand(w.shape[0], *([-1] * (c.dim() - 1))), w], 1)

        def padded(w):  # [B, T, ...]: zeros at the first s positions, then w. For a per-position step (a norm, a
            # reader's inputs) whose results before s are not used: each position's result is the one the whole
            # sequence gives, the shape being the same
            return w if s == 0 else torch.cat([w.new_zeros(w.shape[0], s, *w.shape[2:]), w], 1)

        def win(t):  # positions from s of a [*, T, ...] tensor (or one shared across positions)
            return t if s == 0 or t.shape[1] == 1 else t[:, s:]

        L = T - s

        def ux(n):  # [B, L, C]
            return win(u[n]).expand(B, L, -1)

        def scale(n):  # 1 in the graph (a query or key node's strength under qk), u outside
            if qk is not None and n in qk:
                w = win(qk[n]).expand(B, L, -1)
                return torch.where(win(G[n]), w + (1 - w) * ux(n), ux(n))
            return torch.where(win(G[n]), 1.0, ux(n))

        def uat(n, b, t, c):
            return u[n][b if u[n].shape[0] > 1 else torch.zeros_like(b), t if u[n].shape[1] > 1 else torch.zeros_like(t), c]

        def rest(n, z):
            return None if no_rest else win(ur[n]).expand(B, L)[..., None] * (z @ tg.site(n).delta_T)

        def plus(a, r):
            return a if r is None else a + r

        def strength(key, w, table):
            return 1.0 if table is None or key not in table else table[key]

        def fixed(n, A):  # an activation tensor after patching, recorded
            if patch is not None and n in patch:
                A = torch.where(win(patch[n][0]), win(patch[n][1]), A)
            if record is not None and (keep is None or n in keep):
                record[n] = A.detach()
            return A

        tok = torch.tensor(ids, device=dev)
        emb = tg.wte[tok if s == 0 else tok[s:]]
        x = emb[None] if empty else emb[None].expand(B, L, -1).clone()  # every output at its scale
        xg = x.clone() if both else x  # complete graphs: their nodes' outputs in full
        causal = self._const(("causal", T), lambda: torch.ones(T, T, dtype=torch.bool, device=dev).tril())
        acts = {}

        def writer_out(wn, wi, space_U):  # (b, t, c, the writer's activation, its u) of writer rows wi
            b, t, c = Wr[wn][0][wi], Wr[wn][1][wi], Wr[wn][2][wi]
            if cache is None:
                if fill is not None:
                    fill[("a", wn)] = acts[wn]
                a = acts[wn][b, t, c]
            else:  # before s from the reused run (kept at every position), from s on from this one
                kept = cache[("a", wn)]
                a = torch.where(t < s, kept[b if kept.shape[0] > 1 else torch.zeros_like(b), t, c], acts[wn][b, (t - s).clamp(min=0), c])
            return b, t, c, a, uat(wn, b, t, c)

        def topup(n, width, contrib):  # each reader node's declared parents' extra share
            out = torch.zeros(len(R[n][0]), width, device=dev)
            for (rn, wn), (ri, wi) in E.items():
                if rn == n:
                    out = out.index_add(0, ri, contrib(wn, wi, strength((rn, wn), wi, ew)))
            return out

        def resid_contrib(wn, wi, e):
            _, _, c, a, uw = writer_out(wn, wi, None)
            return (e * (1 - uw) * a)[:, None] * tg.site(wn).U[c]

        def put(n, A, vals):  # reader nodes' activations into A (those at positions from s)
            b, t, c = R[n]
            if s == 0:
                return A.index_put((b, t, c), vals)
            key = ("_window", n, s)
            if key not in plan:
                ts = plan["readers_t"][n]
                sel = [j for j, p in enumerate(ts) if p >= s]
                plan[key] = (self._ints([sel, [ts[j] - s for j in sel]]) if sel else None)
            if plan[key] is None:
                return A
            sel, tw = plan[key]
            return A.index_put((b[sel], tw, c[sel]), vals[sel])

        def read_resid(n, norm, z, zg, xs):  # xs: the stream at every position
            V = tg.site(n).V
            A = torch.where(win(G[n]) & comp, zg @ V, z @ V) if both else (z @ V).expand(B, L, -1)
            if n in R:
                b, t, c = R[n][0], R[n][1], R[n][2]
                inp = rms(xs.expand(B, T, -1)[b, t] + topup(n, xs.shape[-1], resid_contrib), norm, eps)
                A = put(n, A, (inp * V.T[c]).sum(-1))
            return A

        def normed(x, norm):  # (the stream at every position (zeros before s), its norm at the positions computed)
            xf = padded(x)
            if empty:  # one row for every graph; the norm over all of them, as each graph's
                return xf, rms(xf.expand(B, T, -1).contiguous(), norm, eps)[:1]
            z = rms(xf, norm, eps)
            return xf, (z if s == 0 else z[:, s:].contiguous())

        for i in range(tg.n_layer):
            nm = {k: site_name(i, k) for k in vpd_model.KINDS}
            n1, n2 = tg.norms[2 * i], tg.norms[2 * i + 1]
            xf, z = normed(x, n1)
            zg = normed(xg, n1)[1] if both else None
            for k in ("q_proj", "k_proj", "v_proj"):
                acts[nm[k]] = fixed(nm[k], read_resid(nm[k], n1, z, zg, xf))
            qv = full(("qv", i), plus((acts[nm["q_proj"]] * scale(nm["q_proj"])) @ tg.site(nm["q_proj"]).U, rest(nm["q_proj"], z)))
            kv = full(("kv", i), plus((acts[nm["k_proj"]] * scale(nm["k_proj"])) @ tg.site(nm["k_proj"]).U, rest(nm["k_proj"], z)))
            Uv = tg.site(nm["v_proj"]).U
            rv = rest(nm["v_proj"], z)
            vs = None if empty else full(("vs", i), plus((acts[nm["v_proj"]] * ux(nm["v_proj"])) @ Uv, rv))  # empty: u = 0, the values write nothing
            vg = full(("vg", i), plus((acts[nm["v_proj"]] * scale(nm["v_proj"])) @ Uv, rv)) if both else vs
            heads = lambda y: y.view(B, T, H, hd).transpose(1, 2)  # noqa: E731
            no = nm["o_proj"]
            if not empty or no in R:
                q, kk = tg._rope(heads(qv), T), tg._rope(heads(kv), T)
                P = ((q @ kk.transpose(-1, -2)) / math.sqrt(hd)).masked_fill(~causal, float("-inf")).softmax(-1)  # [B, H, T, T]
            att_s = None if empty else (P @ heads(vs)).transpose(1, 2).reshape(B, T, -1)
            att_g = (P @ heads(vg)).transpose(1, 2).reshape(B, T, -1) if both else att_s
            att_sw = None if empty else (att_s if s == 0 else att_s[:, s:].contiguous())
            Vo = tg.site(no).V
            if both:
                Ao = torch.where(win(G[no]) & comp, (att_g if s == 0 else att_g[:, s:].contiguous()) @ Vo, att_sw @ Vo)
            else:
                Ao = torch.zeros(B, T, Vo.shape[1], device=dev) if empty else att_sw @ Vo
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
                Ao = put(no, Ao, (((torch.zeros(len(b), Vo.shape[0], device=dev) if empty else att_s[b, t]) + extra) * Vo.T[c]).sum(-1))
            acts[no] = Ao = fixed(no, Ao)
            Uo = tg.site(no).U
            ro = rest(no, att_sw)
            if not empty:
                x = plus(x + (Ao * ux(no)) @ Uo, ro)
            xg = plus(xg + (Ao * scale(no)) @ Uo, ro) if both else x
            xf2, z2 = normed(x, n2)
            zg2 = normed(xg, n2)[1] if both else None
            nf, nd = nm["c_fc"], nm["down_proj"]
            acts[nf] = fixed(nf, read_resid(nf, n2, z2, zg2, xf2))
            Uf = tg.site(nf).U
            rf = rest(nf, z2)
            pre_s = None if empty else plus((acts[nf] * ux(nf)) @ Uf, rf)
            pre_g = plus((acts[nf] * scale(nf)) @ Uf, rf) if both else pre_s
            gs = None if empty else gelu(pre_s)
            Vd = tg.site(nd).V
            if both:
                Ad = torch.where(win(G[nd]) & comp, gelu(pre_g) @ Vd, gs @ Vd)
            else:
                Ad = torch.zeros(B, T, Vd.shape[1], device=dev) if empty else gs @ Vd
            if nd in R:
                b, t, c = R[nd][0], R[nd][1], R[nd][2]

                def mlp_contrib(wn, wi, e):
                    _, _, wc, a, uw = writer_out(wn, wi, None)
                    return (e * (1 - uw) * a)[:, None] * Uf[wc]
                pre = torch.zeros(len(b), Uf.shape[1], device=dev) if empty else padded(pre_s)[b, t]
                Ad = put(nd, Ad, (gelu(pre + topup(nd, Uf.shape[1], mlp_contrib)) * Vd.T[c]).sum(-1))
            acts[nd] = Ad = fixed(nd, Ad)
            Ud = tg.site(nd).U
            rd = rest(nd, gs)
            if not empty:
                x = plus(x + (Ad * ux(nd)) @ Ud, rd)
            xg = plus(xg + (Ad * scale(nd)) @ Ud, rd) if both else x
            for n in nm.values():  # only the graph's writers' activations are read after their layer
                if n not in Wr:
                    acts.pop(n, None)
        tw = [p - s for p in targets]
        zt = torch.where(comp[:, :, :1].expand(B, 1, 1), xg[:, tw], x[:, tw]) if both else x[:, tw].expand(B, -1, -1)
        for wn, (b, t, wi) in O.items():
            key = ("_pos", wn, tuple(targets))
            if key not in plan:
                plan[key] = self._ints([[targets.index(a) for a in plan["out_t"][wn]]])[0]
            zt = zt.index_put((b, plan[key]), resid_contrib(wn, wi, strength(wn, wi, ow)), accumulate=True)
        return zt

    # ---- the verifier

    def _kl(self, logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
        return (logp.exp() * (logp - logq)).sum(-1).sum(-1) / LN2

    def _zero(self) -> tuple[dict, dict]:
        """The ablation u = 0, remainders at 0 (the same objects every call: run() takes its shortcut for them)."""
        return self._const("zero", lambda: ({n: torch.zeros(1, 1, self.C[n], device=self.dev) for n in self.names},
                                            {n: torch.zeros(1, 1, device=self.dev) for n in self.names}))

    def _one(self) -> tuple[dict, dict]:
        """The ablation u = 1, remainders at 1: the whole model (the same objects every call)."""
        return self._const("one", lambda: ({n: torch.ones(1, 1, self.C[n], device=self.dev) for n in self.names},
                                           {n: torch.ones(1, 1, device=self.dev) for n in self.names}))

    @torch.no_grad()
    def _model_record(self, ids: list[int]) -> tuple[dict, dict]:
        """(every matrix's activations [1, T, C], the run's states for run(reuse=)) of the whole model on ids (the last
        text's kept)."""
        if self._rec[0] != tuple(ids):
            rec, states = {}, {}
            self.run(ids, [len(ids) - 1], self._full_plan(len(ids), 1), *self._one(), record=rec, fill=states)
            self._rec = (tuple(ids), (rec, states), {1: states})
        return self._rec[1]

    def _drop_states(self) -> None:
        """Release the batch-sized copies of the whole model's states (_model_states), keeping the text's own."""
        if self._rec[2]:
            for B in [B for B in self._rec[2] if B != 1]:
                del self._rec[2][B]

    @torch.no_grad()
    def _model_states(self, ids: list[int], B: int) -> dict:
        """The states (run(fill=)) of B copies of the whole model on ids, for run(reuse=) by a batch of B (a batch of
        another size can round its norms differently)."""
        self._model_record(ids)
        by = self._rec[2]
        if B not in by:
            st = {}
            self.run(ids, [len(ids) - 1], self._full_plan(len(ids), B), *self._one(), fill=st)
            by[B] = {k: v[:1] for k, v in st.items()}  # the B rows are the same numbers: one kept, expanded where read
        return by[B]

    def _node_arrays(self, g: Graph, kept: dict) -> dict:
        """A graph's nodes as matrix -> (positions, indices) int64 arrays, read once per `kept` (keyed by the graph
        object, which it keeps alive; the graph must not change meanwhile)."""
        if id(g) not in kept:
            if g.masks is not None:
                nz = {n: m.nonzero().cpu().numpy() for n, m in g.masks.items()}
                kept[id(g)] = (g, {n: (a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)) for n, a in nz.items() if len(a)})
            else:
                per = {}
                for n, t, c in g.nodes:
                    lst = per.setdefault(n, ([], []))
                    lst[0].append(t)
                    lst[1].append(c)
                kept[id(g)] = (g, {n: (np.array(ts, dtype=np.int64), np.array(cs, dtype=np.int64)) for n, (ts, cs) in per.items()})
        return kept[id(g)][1]

    def _held(self, graphs: list[Graph], T: int, seed: int, i: int, kept: dict) -> dict:
        """faithfulness' holds on prompt i: per matrix with any, the [B, T, C] mask of the graphs' subcomponents held at
        their values on the text; (matrix j, position t, index c) is held when torch.rand(1) under the CPU generator
        seeded (seed * 7919 + i * 104729 + j * 1299709 + t * 15485863 + c) mod 2^62 is below 1/2 (below_half)."""
        base = (seed * 7919 + i * 104729) % (1 << 62)
        per = {}
        for b, g in enumerate(graphs):
            for n, (t, c) in self._node_arrays(g, kept).items():
                per.setdefault(n, []).append((b, t, c))
        out = {}
        for n, parts in per.items():
            t = np.concatenate([p[1] for p in parts])
            c = np.concatenate([p[2] for p in parts])
            b = np.concatenate([np.full(len(p[1]), p[0], dtype=np.int64) for p in parts])
            j = self.names.index(n)
            seeds, inv = np.unique(np.uint64(base) + np.uint64(j * 1299709) + t.astype(np.uint64) * np.uint64(15485863) + c.astype(np.uint64),
                                   return_inverse=True)  # a subcomponent in several graphs: drawn once
            on = below_half(seeds)[inv.reshape(-1)]
            if on.any():
                m = torch.zeros(len(graphs), T, self.C[n], dtype=torch.bool, device=self.dev)
                bi, ti, ci = self._ints([b[on], t[on], c[on]])
                m[bi, ti, ci] = True
                out[n] = m
        return out

    @torch.no_grad()
    def changes(self, ids: list[int], targets: list[int], n: int = CHANGES, seed: int = 0, chunk: int = 64) -> list[Changed]:
        """n changed prompts of a text, each the token at one position replaced by a draw from the model's prediction
        there among the other tokens (a changed prompt changes its token: a draw of the token already there tests
        nothing, and where the model is sure of the text's token, as at every token it can copy from earlier, it would
        be drawn nearly always, so experiments would never test what the prediction reads from such tokens), the
        positions being 1 to the last target (those whose token the model predicts). The position is drawn half
        uniformly and half in proportion to how far one such change there moves the model's prediction at the last
        target (a probe per position), and every experiment counts the same: the measure is how the graph follows the
        model on the changes the prediction responds to, with every position still tried. (Weighting each position
        equally instead made the one token a prediction copies from count 1/T: an answer that left out the copy
        scored as faithful.)"""
        last = max(targets)
        if last < 1:
            return [Changed(ids, 1.0) for _ in range(n)]
        g = torch.Generator(device="cpu").manual_seed(seed)
        moved, other = self.sensitivity(ids, targets, g)  # 16 probes at a time: their full logits are ~1.6 GB
        uniform = torch.full((last,), 1.0 / last)
        q = 0.5 * uniform + 0.5 * (moved / moved.sum() if moved.sum() > 0 else uniform)
        out = []
        for _ in range(n):
            j = int(torch.multinomial(q, 1, generator=g))
            x = list(ids)
            x[j + 1] = int(torch.multinomial(other[j], 1, generator=g))
            out.append(Changed(x, 1.0))
        return out

    def sensitivity(self, ids: list[int], targets: list[int], g: torch.Generator, chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        """(moved [last], other [T - 1, V]): moved[p - 1], the KL (nats) of the model's prediction at the last target
        after the token at position p is replaced by one draw from other[p - 1], the model's prediction for position p
        without the text's token there (changes' probe: where the prediction responds to the text)."""
        last = max(targets)
        probs = self.vpd.target_forward(torch.tensor([ids], device=self.dev))[0].float().softmax(-1).cpu()
        base = torch.log(probs[last].clamp_min(1e-30))
        other = probs.clone()
        other[torch.arange(len(ids) - 1), torch.tensor(ids[1:])] = 0.0
        probe = []
        for p in range(1, last + 1):
            x = list(ids)
            x[p] = int(torch.multinomial(other[p - 1], 1, generator=g))
            probe.append(x)
        lqs = []
        with torch.no_grad():
            for s0 in range(0, len(probe), chunk):
                batch = torch.tensor(probe[s0:s0 + chunk], device=self.dev)
                if len(batch) > 1:  # the logits at the last target alone: the numbers they have in the whole sequence's
                    self.vpd.clear()
                    logits = (self.target.hidden(batch)[:, last:last + 1] @ self.target.wte.T)[:, 0]
                else:
                    logits = self.vpd.target_forward(batch)[:, last]
                lqs.append(torch.log_softmax(logits.float(), -1))
            lqs = torch.cat(lqs).cpu()  # one transfer
        moved = []
        for s0 in range(0, len(probe), chunk):
            lq = lqs[s0:s0 + chunk]
            moved.append((base.exp() * (base - lq)).sum(-1).clamp_min(0))
        return torch.cat(moved), other

    def faithfulness(self, ids: list[int], targets: list[int], graphs: list[Graph] | list[list[Graph]], prompts: list[list[int]], seed: int = 0,
                     chunk: int = 16) -> list[float]:
        """Per graph, the mean KL in bits of the model side's next-token distribution at the targets from the graph
        side's over experiments, one per changed prompt x' of the text, weighted by its importance weight (Changed).
        graphs: one list for every prompt, or graphs[i] the instances on prompt i (an answer program run on x', which
        may bind differently there); index b is the same answer throughout.

        An experiment draws, by the experiment's seed and shared by every graph (the same subcomponent is held in every
        graph that has it), (1) which of the graph's subcomponents are held at their values on the original text: each
        with probability 1/2, so how an answer groups them into steps changes nothing; (2) whether what lies outside the
        graph is removed or set to an independent uniform level per subcomponent and position (VPD's stochastic test),
        with probability 1/2 each. Graph side: the graph runs on x' with the rest as drawn, the held subcomponents at
        their values when it runs so on the text. Model side: the whole model on x', the same subcomponents at their
        values in the whole model on the text. Every experiment tests that nothing outside the graph matters, at any
        level (the model side has it whole, the graph side not or partly), and holds test that the graph computes from
        its parts what the model computes from the same parts (interchange interventions)."""
        if not graphs:
            return []
        per_prompt = isinstance(graphs[0], list)
        at = graphs if per_prompt else [graphs] * len(prompts)
        T, nb = len(ids), len(at[0])
        refs = [self.reference([x], targets) for x in prompts]
        zero, one = self._zero(), self._one()
        rec_m = self._model_record(ids)[0]
        total = torch.zeros(nb, device=self.dev)
        kept, plans = {}, {}  # per graph object its nodes; per chunk the last plan, reused while its graph objects are (prompts sharing graphs)
        for i, (x, logp) in enumerate(zip(prompts, refs)):
            g = torch.Generator(device="cpu").manual_seed((seed * 1_000_003 + i) % (1 << 62))
            partial = bool(torch.rand(1, generator=g) < 0.5)
            if partial:  # drawn on the CPU in the same order, sent at once
                drawn = [torch.rand(1, T, self.C[n], generator=g) for n in self.names] + [torch.rand(1, T, generator=g) for n in self.names]
                sent = torch.cat([d.flatten() for d in drawn]).to(self.dev).split([d.numel() for d in drawn])
                rest = ({n: v.view(1, T, self.C[n]) for n, v in zip(self.names, sent)}, {n: v.view(1, T) for n, v in zip(self.names, sent[len(self.names):])})
            else:
                rest = zero
            w = getattr(x, "weight", 1.0)
            for s0 in range(0, nb, chunk):
                part = at[i][s0:s0 + chunk]
                B = len(part)
                key = tuple(id(gr) for gr in part)
                if plans.get(s0, (None,))[0] != key:
                    plans[s0] = (key, part, self._plan(part, T))
                plan = plans[s0][2]
                mask = self._held(part, T, seed, i, kept)
                with torch.no_grad():
                    if not mask:
                        total[s0:s0 + B] += w * self._kl(logp, self.run(x, targets, plan, *rest))
                        continue
                    rec_g, states = {}, {}
                    self.run(ids, targets, plan, *rest, record=rec_g, keep=set(mask), fill=states)
                    lq = self.run(x, targets, plan, *rest, patch={n: (m, rec_g[n]) for n, m in mask.items()}, reuse=(states, ids))
                    lp = self.run(x, targets, self._full_plan(T, B), *one, patch={n: (m, rec_m[n].expand(B, -1, -1)) for n, m in mask.items()},
                                  reuse=(self._model_states(ids, B), ids))
                    del states
                    total[s0:s0 + B] += w * (lp.exp() * (lp - lq)).sum(-1).sum(-1) / LN2
                free(self.dev)
        self._drop_states()
        return (total / len(prompts)).tolist()

    @torch.no_grad()
    def events(self, ids: list[int], targets: list[int], prompts: list[list[int]], holds: list[list[set]] | None = None, seed: int = 0) -> list:
        """The English reader's questions (reader.py) with their answers in the model. Per changed prompt that changes
        a token: {"position", "old", "new" (token ids), "down": whether the model's probability of its most likely next
        token on the text (at the last target) is lower on the changed prompt, "weight": its importance weight}.
        holds[j]: answer j's steps (node sets); then out[j] holds those events and, per changed prompt, the same question
        with one of its steps (drawn uniformly, the draw shared across answers) held in the model at its values on the
        text ("hold": the step's number from 1). Without holds: the input events alone."""
        last = max(targets)
        T = len(ids)
        base = self.reference([ids], [last])[0, 0]
        top = int(base.argmax())
        plain = []
        for x in prompts:
            diff = [p for p in range(len(ids)) if x[p] != ids[p]]
            if diff:
                lp = self.reference([x], [last])[0, 0]
                plain.append((x, {"position": diff[0], "old": ids[diff[0]], "new": x[diff[0]], "down": bool(lp[top] < base[top]), "weight": getattr(x, "weight", 1.0)}))
        if holds is None:
            return [e for _, e in plain]
        one = self._one()
        rec = self._model_record(ids)[0]
        g = torch.Generator(device="cpu").manual_seed(seed % (1 << 62))
        draws = torch.rand(len(plain), generator=g)
        out = [[e for _, e in plain] for _ in holds]
        live = [j for j, st in enumerate(holds) if st]
        for i, (x, e) in enumerate(plain):
            if not live:
                break
            at, ks = {}, {}
            for b, j in enumerate(live):
                k = min(int(draws[i] * len(holds[j])), len(holds[j]) - 1)
                ks[j] = k
                for n, t, c in holds[j][k]:
                    lst = at.setdefault(n, ([], [], []))
                    lst[0].append(b)
                    lst[1].append(t)
                    lst[2].append(c)
            mask = {}
            for n, lst in at.items():
                mask[n] = torch.zeros(len(live), T, self.C[n], dtype=torch.bool, device=self.dev)
                bi, ti, ci = self._ints(list(lst))
                mask[n][bi, ti, ci] = True
            lp = self.run(x, [last], self._full_plan(T, len(live)), *one, patch={n: (m, rec[n].expand(len(live), -1, -1)) for n, m in mask.items()},
                          reuse=(self._model_states(ids, len(live)), ids))
            down = (lp[:, 0, top] < base[top]).tolist()  # log-probabilities both
            for b, j in enumerate(live):
                out[j].append({**e, "hold": ks[j] + 1, "down": down[b]})
            free(self.dev)
        self._drop_states()
        return out


    @torch.no_grad()
    def necessity(self, ids: list[int], targets: list[int], g: Graph, prompts: list[list[int]]) -> float:
        """The mean over the text's changed prompts of the KL in bits of the model's next-token distribution from the
        model's with the graph's subcomponents removed (everything else kept): how much the prediction depends on the
        graph's parts, the test an edit or unlearning relies on. A part with a backup elsewhere lowers it; it is not a
        verdict on the graph."""
        nodes = g.node_set()
        total = 0.0
        for x in prompts:
            total += getattr(x, "weight", 1.0) * float(self._kl(self.reference([x], targets), self.without(x, targets, nodes)))
        return total / len(prompts)

    def without(self, ids: list[int], targets: list[int], nodes) -> torch.Tensor:
        """log q at the targets [1, len(targets), V] of the whole model with `nodes` ((matrix, position, subcomponent))
        removed and everything else kept."""
        T = len(ids)
        u = {n: torch.ones(1, T, self.C[n], device=self.dev) for n in self.names}
        for n, t, c in nodes:
            u[n][0, t, c] = 0.0
        ur = {n: torch.ones(1, T, device=self.dev) for n in self.names}
        none = {n: torch.zeros(1, T, self.C[n], dtype=torch.bool, device=self.dev) for n in self.names}
        plan = {**self._full_plan(T, 1), "G": none, "none": True}  # no node held at 1: every subcomponent runs at its u
        return self.run(ids, targets, plan, u, ur)

    def contributions(self, ids: list[int], targets: list[int]) -> dict:
        """Per matrix, how much each subcomponent writes at each position on the text: |activation| x |write vector|,
        [T, C]."""
        rec = self._model_record(ids)[0]
        return {n: rec[n][0].abs() * self.target.site(n).U.norm(dim=-1) for n in self.names}

    def adversarial(self, cases: list[tuple], steps: int = ADV_STEPS, step_size: float = ADV_STEP, seed: int = 0) -> list[float]:
        """VPD's adversarial test of graphs, an evaluation measure: every subcomponent outside a graph (and every
        remainder) at a mask in [0, 1] chosen to maximize the summed KL in bits of the model's next-token distribution at
        the targets from the graph's, one mask per subcomponent shared by every position and every case (VPD's tying
        across a batch: the adversary must exploit systematic weaknesses), by `steps` signed-gradient steps of size
        step_size projected onto [0, 1] from a uniform draw (VPD's PGD). cases: (ids, targets, graph). The KL of each case
        at the final masks. Compare graphs only against others under the same adversary (VPD's own answer): no
        decomposition survives it outright."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        u = {n: torch.rand(self.C[n], generator=g).to(self.dev) for n in self.names}
        ur = {n: torch.rand(1, generator=g).to(self.dev) for n in self.names}
        prepared = [(ids, targets, self._plan([gr], len(ids)), self.reference([ids], targets)) for ids, targets, gr in cases]

        def kls(grad: bool):
            out = []
            for ids, targets, plan, logp in prepared:
                uu = {n: u[n].view(1, 1, -1) for n in self.names}
                rr = {n: ur[n].view(1, 1) for n in self.names}
                with torch.set_grad_enabled(grad):
                    k = self._kl(logp, self.run(ids, targets, plan, uu, rr))[0]
                    if grad:
                        k.backward()
                out.append(float(k.detach()))
            return out

        for _ in range(steps):
            for n in self.names:
                u[n].requires_grad_(True)
                ur[n].requires_grad_(True)
            kls(True)
            with torch.no_grad():
                for n in self.names:
                    u[n] = (u[n] + step_size * u[n].grad.sign()).clamp(0, 1).detach()
                    ur[n] = (ur[n] + step_size * ur[n].grad.sign()).clamp(0, 1).detach()
        return kls(False)

    def positions(self, targets: list[int]) -> int:
        """The positions a graph's nodes can sit at: 0 to the last target."""
        return max(targets) + 1

    def everything_bits(self, targets: list[int]) -> float:
        """Graph.bits of the whole model, every subcomponent at every position a node can sit at and every connection
        among them (implied_edges' rule, counted per site without listing): the top of the description-length range."""
        P, total = self.positions(targets), sum(self.C.values())
        C = {_layer_kind((n, 0, 0)): c for n, c in self.C.items()}
        writers = [(mech.stage(l, k), c) for (l, k), c in C.items() if k in mech.RESID_WRITERS]
        per_position = sum(c * sum(cw for st, cw in writers if st <= mech.stage(l, k)) for (l, k), c in C.items() if k in mech.RESID_READERS)
        layers = {l for l, _ in C}
        inner = P * per_position + sum(C.get((l, "o_proj"), 0) * C.get((l, "v_proj"), 0) * P * (P + 1) // 2 + C.get((l, "down_proj"), 0) * C.get((l, "c_fc"), 0) * P for l in layers)
        out = len(set(targets)) * sum(c for _, c in writers)
        n = P * total
        return n * math.log2(P * total) + math.log2(n) * (2 * inner + out)

    def score(self, ids: list[int], targets: list[int], graphs: list[Graph], prompts: list[list[int]], seed: int = 0) -> list[dict]:
        """{"kl_bits", "bits", "nodes", "edges"} per graph: its faithfulness and its graph's description length (Graph.bits)."""
        total = sum(self.C.values())
        kl = self.faithfulness(ids, targets, graphs, prompts, seed)
        return [{"kl_bits": k, "bits": g.bits(self.positions(targets), total, targets), "nodes": g.count(), "edges": None if g.complete else g.edges()}
                for g, k in zip(graphs, kl)]

    @torch.no_grad()
    def lens(self, nodes, k: int = 3) -> dict:
        """Per residual writer (attention or MLP output) among `nodes`: the k tokens its write vector raises most at the
        prediction, (U_c * final norm weight) @ embedding^T (the logit lens of what it writes)."""
        tk = mech.tokenizer("vpd4l")
        out = {}
        for nd in nodes:
            if _layer_kind(nd)[1] in mech.RESID_WRITERS:
                logits = (self.target.site(nd[0]).U[nd[2]] * self.target.ln_f) @ self.target.wte.T
                out[nd] = [tk.decode([int(i)]) for i in logits.topk(k).indices.tolist()]
        return out

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
                    grads = torch.autograd.grad(getattr(x, "weight", 1.0) * self._kl(logp, lq)[0], [masks[n] for n in self.names])
                for n, gr in zip(self.names, grads):
                    total[n] -= gr[0]
            free(self.dev)
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

        at = {n: self._ints([[qk[j][1] for j in idx[n]], [qk[j][2] for j in idx[n]], idx[n]]) for n in sites}  # (positions, indices, rows of w)

        def qk_of(w):
            res = {}
            for n in sites:
                t, c, js = at[n]
                res[n] = torch.zeros(1, T, self.C[n], device=self.dev).index_put((torch.zeros_like(t), t, c), w[js])
            return res

        zero = self._zero()
        ig = torch.zeros(n_edges + len(qk), device=self.dev)
        for x in prompts:
            logp = self.reference([x], targets)
            for k in range(IG_STEPS):
                w = torch.full((n_edges + len(qk),), (k + 0.5) / IG_STEPS, device=self.dev, requires_grad=True)
                with torch.enable_grad():
                    ew, ow = self.edge_weights(plan, w[:n_edges])
                    kl = getattr(x, "weight", 1.0) * self._kl(logp, self.run(x, targets, plan, *zero, ew, ow, qk_of(w[n_edges:])))[0]
                    (gr,) = torch.autograd.grad(kl, w)
                ig -= gr
            free(self.dev)
        allitems = items + [(nd, None) for nd in qk]
        return [allitems[j] for j in (-ig).argsort().tolist()]

    def ordered(self, ids: list[int], targets: list[int], prompts: list[list[int]], log=None, nodes_from: str = "ig") -> tuple[list[Graph], list[dict]]:
        """A bootstrap answer: (the graph after each step, their scores); see the module docstring. The subcomponents
        whose connections are ranked: the most top-ranked whose connections fit MAX_CONNECTIONS, ranked by integrated
        gradients (nodes_from "ig") or in VPD's order (nodes_from "vpd": vpd_steps' ranking, causal importance x how
        much each writes, the predicted position first)."""
        T = len(ids)
        if nodes_from == "vpd":
            ranked = self._vpd_order(ids, targets)  # vpd_steps' last step, its nodes in its order
        else:
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
            g = live(build(items[:k]), targets)
            if g.count() and (not graphs or g.size() > graphs[-1].size()):
                graphs.append(g)
            if k >= len(items):
                break
            k = min(2 * k, len(items))
        scores = self.score(ids, targets, graphs, prompts, seed=task_seed(str(ids[:8])))
        keep = min(range(len(scores)), key=lambda j: scores[j]["kl_bits"]) + 1  # a connection does nothing until its path to the
        return graphs[:keep], scores[:keep]  # input is in, so a step may lower nothing and a later one a lot: end at the lowest KL


def prefix(ir: dict, k: int) -> tuple[Graph, list[str]]:
    """The graph of an answer's first k steps as written (library entries not expanded) and the entries those steps
    use; ir is mech's trace of the answer."""
    g = ir["graph"]
    nd = [(site_name(layer, kind), t, c) for layer, kind, t, c in g["nodes"]]
    nodes = {nd[i] for i, s in enumerate(g["node_step"]) if s < k}
    if g.get("complete_step") is not None and g["complete_step"] < k:  # "parts": every connection among the nodes
        return Graph(nodes, complete=True), [u for u, s in zip(g["uses"], g["uses_step"]) if s < k]
    parents = {}
    for (r, w), s in zip(g["parents"], g["parent_step"]):
        if s < k:
            parents.setdefault(nd[r], []).append(nd[w])
    out = [nd[w] for w, s in zip(g["out"], g["out_step"]) if s < k]
    return Graph(nodes, parents, out), [u for u, s in zip(g["uses"], g["uses_step"]) if s < k]


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
    queries, keys = {}, {}  # (layer, position) -> its queries; layer -> its keys by position
    for nd in g.nodes:
        layer, kind = _layer_kind(nd)
        if kind == "q_proj":
            queries.setdefault((layer, nd[1]), []).append(nd)
        elif kind == "k_proj":
            keys.setdefault(layer, []).append(nd)
    for ks in keys.values():
        ks.sort(key=lambda nd: nd[1])
    reached = {}  # layer -> how many of its keys (by position) feed an attention output kept so far
    keep, stack = set(g.out), list(g.out)
    while stack:
        nd = stack.pop()
        new = list(feeds.get(nd, ()))
        layer, kind = _layer_kind(nd)
        if kind == "o_proj" and nd in g.nodes:
            new += queries.get((layer, nd[1]), [])
            ks, i = keys.get(layer, []), reached.get(layer, 0)
            while i < len(ks) and ks[i][1] <= nd[1]:
                new.append(ks[i])
                i += 1
            reached[layer] = i
        for w in new:
            if w not in keep:
                keep.add(w)
                stack.append(w)
    par = {r: [w for w in ws if w in keep] for r, ws in g.parents.items() if r in keep}
    return Graph(keep, {r: ws for r, ws in par.items() if ws}, [w for w in g.out if w in keep])


def live(g: Graph, targets: list[int]) -> Graph:
    """g without what computes nothing when it runs alone: an attention output with no value among its parents and an
    MLP output with no MLP input among them are zero (a query, key, value or MLP input reads the embedding at least),
    then whatever lies on no path to the prediction (on_path), until nothing changes. Its run alone is the same."""
    while True:
        h = on_path(g, targets)
        dead = {nd for nd in h.nodes if _layer_kind(nd)[1] in ("o_proj", "down_proj") and not h.parents.get(nd)}
        if not dead:
            return h
        par = {r: [w for w in ws if w not in dead] for r, ws in h.parents.items() if r not in dead}
        g = Graph(h.nodes - dead, {r: ws for r, ws in par.items() if ws}, [w for w in h.out if w not in dead])


def tidy(directory: Path, max_chars: int | None = None) -> None:
    """Every answer DIR/<task>.py rewritten with each step's graph made live (live()), steps that add nothing dropped,
    comments and docstring kept: the same runs alone, shorter descriptions. max_chars: first cut each answer to its
    most first steps within that many characters (the part an output budget can hold)."""
    for py in sorted(Path(directory).glob("*.py")):
        task = json.loads((TEXTS / "vpd4l" / f"{py.stem.split('.')[0]}.json").read_text())
        targets = task["prompts"][0]["target_positions"]
        source = py.read_text()
        parts = steps_of(source) if max_chars and len(source) > max_chars else None
        if parts:
            k, size = 0, len(parts[0]) + len(parts[2])
            while k < len(parts[1]) and size + len(parts[1][k]) <= max_chars:
                size += len(parts[1][k])
                k += 1
            source = first_steps(source, k)
        ir = mech.trace_inline(source, "vpd4l", task)
        if not ir["valid"]:
            continue
        g = ir["graph"]
        steps, notes, prev = [], [], None
        for k in range(1, g["steps"] + 1):
            h = live(prefix(ir, k)[0], targets)
            if h.count() and (prev is None or h.size() > prev.size()):
                steps.append(h)
                notes.append(g["notes"][k - 1] if k - 1 < len(g["notes"]) else "")
                prev = h
        py.write_text(program(steps, notes=notes if any(notes) else None, explanation=g.get("explanation") or None))


def part_token(nd) -> str:
    """A node (matrix name, position, index) as its subcomponent token "<p:L.S.I>"."""
    layer, kind = _layer_kind(nd)
    return f"<p:{layer}.{({v: k for k, v in mech.SITES.items()})[kind]}.{nd[2]}>"


def parts_program(steps: list[Graph], explanation: str | None = None) -> str:
    """Complete graphs (each containing the one before) as an answer of "parts" steps: the k-th lists the
    subcomponents steps[k] has beyond steps[k - 1], by position (mech: every connection among them is in)."""
    lines = ["def graph(tokens, targets):"]
    if explanation:
        lines.append(f"    {explanation!r}")
    lines.append("    return [")
    prev = set()
    for g in steps:
        new = sorted(g.node_set() - prev, key=lambda nd: (nd[1], nd[0], nd[2]))
        by = {}
        for nd in new:
            by.setdefault(nd[1], []).append(part_token(nd))
        lines.append("        {\"parts\": {" + ", ".join(f"{t}: \"{''.join(ts)}\"" for t, ts in by.items()) + "}},")
        prev |= g.node_set()
    lines.append("    ]")
    return "\n".join(lines) + "\n"


def program(steps: list[Graph], uses: list[list[str]] | None = None, notes: list[str] | None = None, explanation: str | None = None) -> str:
    """Graphs as an ordered answer: graph(tokens, targets) returning a list of steps, the k-th adding what steps[k]
    has beyond steps[k - 1] (a step's graph contains the previous one's); a step is a dict {(position, reader): its
    new parents, "out": the prediction's new parents, "uses": library entries}; parents at the reader's own position
    are one string of subcomponents, an attention output's parents (values) a {position: string}; a node with no
    parents is listed with "" when nothing reads it. notes: a comment line above each step; explanation: the
    function's docstring."""
    sites = list(mech.SITES.values())

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
                    per.setdefault(w[1], []).append(part_token(w))
                val = "{" + ", ".join(f'{t}: "' + "".join(v) + '"' for t, v in sorted(per.items())) + "}"
            else:
                val = '"' + "".join(part_token(w) for w in ws) + '"'
            body.append(f'            ({r[1]}, "{part_token(r)}"): {val},')
        if new_out:
            body.append('            "out": "' + "".join(part_token(w) for w in sorted(new_out, key=key)) + '",')
        if uses and k < len(uses) and uses[k]:
            body.append('            "uses": [' + ", ".join(repr(u) for u in uses[k]) + "],")
        if notes and k < len(notes) and notes[k]:
            lines.append("        # " + notes[k].strip().replace("\n", " "))
        lines += ["        {"] + body + ["        },"]
        prev = g
    lines.append("    ]")
    return "\n".join(lines) + "\n"


def steps_of(source: str) -> tuple[str, list[str], str] | None:
    """An answer in program()'s layout split into (its head through "return [", each step's text with its comment
    line, its tail); None for any other layout."""
    lines = source.splitlines(keepends=True)
    try:
        start = next(i for i, x in enumerate(lines) if x.strip() == "return [")
        end = max(i for i, x in enumerate(lines) if x.strip() == "]")
    except (StopIteration, ValueError):
        return None
    steps, cur = [], []
    for x in lines[start + 1:end]:
        cur.append(x)
        if x.strip() == "},":
            steps.append("".join(cur))
            cur = []
    if cur:
        return None
    return "".join(lines[:start + 1]), steps, "".join(lines[end:])


def first_steps(source: str, k: int) -> str:
    """An answer in program()'s layout cut to its first k steps (the whole source in any other layout)."""
    parts = steps_of(source)
    if parts is None:
        return source
    head, steps, tail = parts
    return head + "".join(steps[:k]) + tail


def tasks(split: str) -> list[Path]:
    """The split's task files in text order."""
    files = [p for p in (TEXTS / "vpd4l").glob("*.json") if not p.name.startswith(".")]  # not macOS's ._ metadata files
    return [p for p in sorted(files, key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())["split"] == split]


def text(p: Path) -> tuple[list[int], list[int]]:
    task = json.loads(p.read_text())["prompts"][0]
    return task["token_ids"], task["target_positions"]


def task_seed(task_id: str) -> int:
    """A text's own seed for its changed prompts (the same in every process)."""
    return int.from_bytes(task_id.encode()[-8:].rjust(8, b"\0"), "big") % (1 << 31)


def claim(path: Path) -> bool:
    """Whether this process takes the text whose claim file is PATH, created here exclusively: processes sharing an
    output directory, on one machine or several, split its texts between them (a claim left by a process that died
    keeps its text skipped until the file is removed)."""
    try:
        path.open("x").close()
        return True
    except FileExistsError:
        return False


def search(split: str, n: int, offset: int = 0, stride: int = 1, out: Path | None = None, reverse: bool = False, nodes_from: str = "ig") -> None:
    """Native.ordered on the split's texts offset, offset + stride, ... of its first n, each text not yet answered or
    claimed (claim) -> OUT/<id>.py (the answer) and .json (each step's score and the seconds); OUT defaults to
    texts/search (train) or texts/search_<split>."""
    nat = Native()
    out = Path(out) if out else TEXTS / ("search" if split == "train" else f"search_{split}")
    out.mkdir(parents=True, exist_ok=True)
    for p in (lambda ps: ps[::-1] if reverse else ps)(tasks(split)[:n][offset::stride]):  # reverse: last text first (to meet another machine working forward)
        if (out / f"{p.stem}.json").exists() or not claim(out / f".{p.stem}.claim"):
            continue
        t0 = time.time()
        ids, targets = text(p)
        prompts = nat.changes(ids, targets, seed=task_seed(p.stem))
        graphs, scores = nat.ordered(ids, targets, prompts, nodes_from=nodes_from)
        empty = nat.score(ids, targets, [Graph()], prompts)[0]
        (out / f"{p.stem}.py").write_text(program(graphs))
        (out / f"{p.stem}.json").write_text(json.dumps({"empty": empty, "steps": scores, "seconds": round(time.time() - t0, 1)}))
        print(f"{p.stem}: {len(graphs)} steps, KL {empty['kl_bits']:.1f} -> " + " -> ".join(f"{s['kl_bits']:.2f} ({s['bits']:.0f} bits)" for s in scores)
              + f", {time.time() - t0:.0f} s", flush=True)


def ranked(split: str, n: int, top: int, out: Path | None = None) -> None:
    """VPD's ranking (vpd_steps' order) of the split's first n texts -> OUT/<id>.json, its first `top` subcomponents at
    the predicted positions, [[position, "<p:L.S.I>"], ...] most important first (elsewhere its ranking is led by the
    same few subcomponents at every position); OUT defaults to texts/vpd_ranked (prompt.with_vpd lists them in the
    oracle's question)."""
    nat = Native()
    out = Path(out) if out else TEXTS / "vpd_ranked"
    out.mkdir(parents=True, exist_ok=True)
    for p in tasks(split)[:n]:
        if (out / f"{p.stem}.json").exists():
            continue
        ids, targets = text(p)
        with torch.no_grad():
            rank = nat._vpd_rank(ids, targets)
        best = [nd for nd in sorted(rank, key=rank.get) if nd[1] in targets][:top]
        (out / f"{p.stem}.json").write_text(json.dumps([[nd[1], part_token(nd)] for nd in best]))


def responses(nat: Native, ids: list[int], targets: list[int], task_id: str, positions: int) -> list:
    """Where the prediction responds to the text, in subcomponent tokens: the predicted positions, then the `positions`
    others whose token, changed to another the model finds likely there, moves the prediction most (sensitivity, the
    verifier's probe), in that order; at each, per weight matrix, the subcomponent that writes most there beyond what it
    writes on average over the text (contributions minus their mean over positions, so a subcomponent active
    everywhere does not lead every position's list). [[position, ["<p:L.S.I>", ...]], ...]"""
    moved = nat.sensitivity(ids, targets, torch.Generator(device="cpu").manual_seed(task_seed(task_id)))[0] if max(targets) >= 1 else torch.zeros(0)  # nothing before position 0
    excess = {n: w - w.mean(0, keepdim=True) for n, w in nat.contributions(ids, targets).items()}
    best = {n: e.argmax(-1).tolist() for n, e in excess.items()}
    order = sorted(targets) + [p for p in (moved.argsort(descending=True) + 1).tolist() if p not in targets][:positions]
    return [[p, [part_token((n, p, best[n][p])) for n in nat.names]] for p in order]


def respond(split: str, n: int, positions: int, out: Path | None = None) -> None:
    """responses() of the split's first n texts -> OUT/<id>.json (default texts/vpd_responses; prompt.with_vpd lists
    them in the oracle's question)."""
    nat = Native()
    out = Path(out) if out else TEXTS / "vpd_responses"
    out.mkdir(parents=True, exist_ok=True)
    for p in tasks(split)[:n]:
        if (out / f"{p.stem}.json").exists() or not claim(out / f".{p.stem}.claim"):  # processes sharing OUT split the texts
            continue
        ids, targets = text(p)
        with torch.no_grad():
            got = responses(nat, ids, targets, p.stem, positions)
        (out / f"{p.stem}.json").write_text(json.dumps(got))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--split", choices=("train", "heldout", "hard"), required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--out", type=Path)
    s.add_argument("--reverse", action="store_true", help="last text first")
    s.add_argument("--nodes-from", choices=("ig", "vpd"), default="ig", help="the candidate subcomponents: integrated gradients' ranking or VPD's (Native.ordered)")
    r = sub.add_parser("ranked")
    r.add_argument("--split", choices=("train", "heldout", "hard"), nargs="+", required=True)
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--top", type=int, default=256)
    r.add_argument("--out", type=Path)
    e = sub.add_parser("responses")
    e.add_argument("--split", choices=("train", "heldout", "hard"), nargs="+", required=True)
    e.add_argument("--n", type=int, required=True)
    e.add_argument("--positions", type=int, default=32)
    e.add_argument("--out", type=Path)
    t = sub.add_parser("tidy")
    t.add_argument("directory", type=Path)
    t.add_argument("--max-chars", type=int, help="cut each answer to its most first steps within this many characters first")
    args = ap.parse_args()
    if args.cmd == "tidy":
        tidy(args.directory, args.max_chars)
    elif args.cmd == "ranked":
        for split in args.split:
            ranked(split, args.n, args.top, args.out)
    elif args.cmd == "responses":
        for split in args.split:
            respond(split, args.n, args.positions, args.out)
    else:
        search(args.split, args.n, args.offset, args.stride, args.out, args.reverse, args.nodes_from)


if __name__ == "__main__":
    main()
