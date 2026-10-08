"""The structural code on the toy gate's real-valued MLP toys (#2951 design, 10-07): a standalone reference
prototype, fast enough to iterate in seconds.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/structural.py TOY_DIR [--bits B] [--out OUT.json]

Representation, exact by construction on the stream's reachable span:
- Atoms of the stream: the input atoms and every neuron's write (its down_proj column). The input atoms are the
  coordinates the layer-0 stream varies in, when they are as many as its rank (axes are named by index); else its
  independent components (FastICA, as many as the centred stream's rank, each signed so its largest
  entry is positive), each made exact where the data pins it: one within 11 degrees of a coordinate axis becomes the axis
  (an axis is named by its index, a direction costs its numbers), and one along which some stream rows lie alone (two or more
  rows along one exact direction near it) becomes that direction, and components that span coordinate axes become those axes; and, where the stream's mean leaves their span, that remainder (the constant atom,
  whose edges are biases); every input atom has unit norm. The stream at layer l is A_l s_l exactly: s_l holds each input's coefficients (zero where
  absent) and each earlier neuron's activation.
- Edges: c_fc^l is the sum over neurons n and atoms k of e_n E_l[n, k] (dual_k)^T with E_l = W_fc^l A_l (the
  overcomplete frame's writes W F^+, F the atoms' dual frame), the head likewise (H = head A_L), and each neuron's
  down_proj slice writes its own atom. Directions the stream never takes carry nothing.
- A block is any set of edges and neuron down slices (cells), in any maps and layers, with one gate: the norm of its
  own write at its gate stage (the last stage of its earliest layer: the pre-activations its c_fc edges write, or
  after the activation the writes of its cells, or the head's outputs), on above zero (to rounding).

Code, in bits (numbers at B bits, an index at log2 of its dictionary, a set of k of n at log2 C(n, k) + log2(n + 1)):
- A block's parts: raw parts, per map its receivers, senders and numbers (dense, zeros charged, or the edges named
  in the rectangle), and bindings of rules. A cell's units: the neuron with its in-edges, its down slice and its
  atom's out-edges in the block. Senders or receivers shared by every unit of the block with one weight are the
  block's roles (named per block); the rest bind units into bindings (connected components). A binding's pattern
  is its units' weights with its own senders and receivers as slots; a pattern used by two or more bindings is a
  rule: its body (the pattern's numbers) is defined once, and each binding is a row of its table.
- Per token: each block on costs log2(#blocks), its roles' names, its raw parts, and log2(#bindings of the rule)
  per binding; each rule used by a block on costs its body once.
- Definitions (paid once through F): the input atoms (learned), each block's threshold, roles and raw parts, each
  binding's table row, each rule's body. M's own neuron writes and head are the dictionary every explanation shares.

Explanations scored: the true blocks (as rules where the code finds repeats, and dense), the pieces (the start: each
atom's fan-out into each map, each neuron's cell with its bias), and the one the search finds: from the pieces,
merges of blocks (a block with the cells its edges feed, the fan-outs of one atom, blocks that fire together),
each kept where it lowers J = E_t[bits per token] + definitions / N and keeps every gated run exact."""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.special import gammaln
from sklearn.decomposition import FastICA

HELD = 4096
LN2 = math.log(2)


def log2C(n, k):
    return float((gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / LN2)


def name_set(n, k):
    return log2C(n, k) + math.log2(n + 1)


class Toy:
    def __init__(self, d: Path):
        rec = json.loads((d / "export.json").read_text())
        self.rec, c = rec, rec["config"]
        W = lambda n: np.fromfile(d / f"{n}.f64", dtype="<f8").reshape(rec["files"][n]["shape"])  # noqa: E731
        self.name, self.kind = d.name, rec.get("real_valued")
        self.L = c["n_layers"]
        self.wte = W("wte")
        self.fc = [W(f"blocks.{l}.mlp.c_fc") for l in range(self.L)]
        self.dn = [W(f"blocks.{l}.mlp.down_proj") for l in range(self.L)]
        for l in range(self.L):
            if any(np.any(W(f"blocks.{l}.attn.{k}_proj")) for k in "qkvo"):
                raise SystemExit("a toy with attention")
        self.head = W("gaussian_head")
        self.bias = W("gaussian_head.bias")[0] if "gaussian_head.bias" in rec["files"] else np.zeros(self.head.shape[0])
        self.relu_head = c["head"]["relu"]
        self.act = (lambda x: np.maximum(x, 0.0)) if c["mlp_act"] == "relu" else (lambda x: x)
        self.truth = json.loads((d / "truth.json").read_text())
        self.dir = d

    def out(self, r):
        y = r @ self.head.T + self.bias
        return np.maximum(y, 0.0) if self.relu_head else y

    def run(self, R):
        r = R
        for l in range(self.L):
            r = r + self.act(r @ self.fc[l].T) @ self.dn[l].T
        return self.out(r)


class Rep:
    """The atoms, the edges and the items (edges and cells) every explanation partitions into blocks."""

    def __init__(self, toy: Toy, fit_rows: np.ndarray):
        self.toy = toy
        self.A0, self.const = input_atoms(fit_rows)
        self.n_in = self.A0.shape[1]
        self.dual0 = sla.pinv(self.A0)
        L, m = toy.L, [w.shape[0] for w in toy.fc]
        # global atom ids: inputs 0..n_in-1, then each layer's neurons
        self.off = [self.n_in + sum(m[:l]) for l in range(L + 1)]
        self.n_atoms = self.off[L]
        self.A = [np.column_stack([self.A0] + toy.dn[:l]) for l in range(L + 1)]
        self.E = [toy.fc[l] @ self.A[l] for l in range(L)]
        self.H = toy.head @ self.A[L]
        self.G = [toy.dn[l].T @ toy.dn[l] for l in range(L)]
        # items: (kind, map, receiver, sender) with kind 'e' (an edge of c_fc^map or of the head, map = L) or 'c'
        # (a cell: neuron receiver of layer map)
        items, w = [], []
        for l in range(L + 1):
            M_ = self.E[l] if l < L else self.H
            tol = 1e-9 * np.abs(M_).max()
            for i, k in zip(*np.nonzero(np.abs(M_) > tol)):
                items.append(("e", l, int(i), int(k)))
                w.append(float(M_[i, k]))
        for l in range(L):
            for n in range(m[l]):
                items.append(("c", l, n, self.off[l] + n))
                w.append(1.0)
        self.items, self.w = items, np.array(w)
        self.index = {it: j for j, it in enumerate(items)}
        self.m = m
        self.n_recv = m + [toy.head.shape[0]]
        self.scale = max(np.abs(self.E[l]).max() for l in range(L)) if L else 1.0
        self.scale = max(self.scale, np.abs(self.H).max())

    def coeffs(self, R):
        return R @ self.dual0.T

    def check_split(self, R):
        """M through the edges (every item on) against M itself: the split's error on these inputs."""
        s = self.coeffs(R)
        recon = s @ self.A0.T - R
        for l in range(self.toy.L):
            a = self.toy.act(s @ self.E[l].T)
            s = np.column_stack([s, a])
        y = s @ self.H.T + self.toy.bias
        y = np.maximum(y, 0.0) if self.toy.relu_head else y
        return float(np.abs(recon).max()), float(np.abs(y - self.toy.run(R)).max())


def input_atoms(R):
    """The input atoms (module note) as columns, and the constant atom's index (None without one)."""
    mu = R.mean(0)
    X = R - mu
    sv = sla.svdvals(X)
    rank = int(np.sum(sv > sv[0] * max(X.shape) * np.finfo(float).eps))
    support = np.nonzero(np.abs(X).max(0) > 0)[0]
    if len(support) == rank:
        # the stream varies in exactly `rank` coordinates: they are the atoms
        A = np.eye(R.shape[1])[:, support]
        rest = mu - A @ (A.T @ mu)
        if np.linalg.norm(rest) <= 1e-12 * max(np.linalg.norm(mu), 1e-300):
            return A, None
        return np.column_stack([snap(rest), A]), 0
    A = FastICA(rank, whiten="unit-variance", random_state=0, max_iter=2000, tol=1e-6).fit(X).mixing_
    A = A * np.sign(A[np.abs(A).argmax(0), np.arange(rank)])
    norms = np.linalg.norm(A, axis=0)
    U = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-300)
    for k in range(rank):
        top = int(np.abs(A[:, k]).argmax())
        if abs(A[top, k]) / norms[k] > 0.98:
            A[:, k] = 0.0
            A[top, k] = norms[k]
            continue
        # rows near this component; among them the largest set along one exact direction
        near = U[np.abs(U @ (A[:, k] / norms[k])) > 0.99]
        if len(near) >= 2:
            near = near * np.sign(near @ A[:, k])[:, None]
            same = (near @ near.T) > 1 - 1e-10
            top = int(same.sum(1).argmax())
            if same[top].sum() >= 2:
                A[:, k] = near[same[top]].mean(0)
                A[:, k] *= norms[k] / np.linalg.norm(A[:, k])
    A = A / np.linalg.norm(A, axis=0)
    rest = mu - A @ sla.lstsq(A, mu)[0]
    if np.linalg.norm(rest) <= 1e-9 * max(np.linalg.norm(mu), 1e-300):
        return A, None
    return np.column_stack([snap(rest), A]), 0


def snap(v):
    """A unit direction: the coordinate axis it lies within 11 degrees of, else itself."""
    top = int(np.abs(v).argmax())
    if abs(v[top]) / np.linalg.norm(v) > 0.98:
        return np.eye(len(v))[top] * np.sign(v[top])
    return v / np.linalg.norm(v)


class Run:
    """M's run in atom coefficients on fixed inputs, with each item's contribution, for gates and checks."""

    def __init__(self, rep: Rep, R):
        self.rep, toy = rep, rep.toy
        s = rep.coeffs(R)
        self.s, self.pre, self.a = [], [], []
        for l in range(toy.L):
            self.s.append(s)
            pre = s @ rep.E[l].T
            a = toy.act(pre)
            self.pre.append(pre)
            self.a.append(a)
            s = np.column_stack([s, a])
        self.s.append(s)
        self.T = R.shape[0]
        self.y = toy.out(R + sum(self.a[l] @ toy.dn[l].T for l in range(toy.L)))

    def contrib(self, j):
        """Item j's contribution per token: an edge's E[i, k] s_k, a cell's activation."""
        kind, l, i, k = self.rep.items[j]
        if kind == "c":
            return self.a[l][:, i]
        return self.rep.w[j] * self.s[l][:, k]


def gate_stage(rep, members):
    """(layer, stage) of a block's gate: its earliest layer; there the cells' writes (stage 1) if it has cells in that
    layer, else the c_fc pre-activations (stage 0); the head is layer L."""
    layers = [rep.items[j][1] for j in members]
    l0 = min(layers)
    has_cell = any(rep.items[j][0] == "c" and rep.items[j][1] == l0 for j in members)
    return l0, 1 if has_cell else 0


def own_write(rep, run, members):
    """The block's own write norm per token at its gate stage, from the run's (all-on) quantities."""
    l0, st = gate_stage(rep, members)
    if l0 < rep.toy.L and st == 1:
        cells = [rep.items[j][2] for j in members if rep.items[j][0] == "c" and rep.items[j][1] == l0]
        a = run.a[l0][:, cells]
        return np.sqrt(np.maximum(np.einsum("ti,ij,tj->t", a, rep.G[l0][np.ix_(cells, cells)], a), 0.0))
    js = [j for j in members if rep.items[j][1] == l0 and rep.items[j][0] == "e"]
    recv = np.array([rep.items[j][2] for j in js])
    vals = np.stack([run.contrib(j) for j in js], 1)
    u, inv = np.unique(recv, return_inverse=True)
    acc = np.zeros((run.T, len(u)))
    np.add.at(acc.T, inv, vals.T)
    return np.sqrt((acc ** 2).sum(1))


class Blocks:
    """A partition of the items into blocks, with each block's gate (own write > tol) on the run."""

    def __init__(self, rep, run, assign):
        self.rep, self.run = rep, run
        self.members = {}
        for j, b in enumerate(assign):
            self.members.setdefault(int(b), []).append(j)
        self.on = {b: self.gate(ms) for b, ms in self.members.items()}

    def gate(self, ms):
        w = own_write(self.rep, self.run, ms)
        return w > 1e-9 * max(1.0, float(w.max()))


# --------------------------------------------------------------------------------------------- the code


_DESC, _RAW = {}, {}


def describe(rep, ms):
    key = tuple(sorted(ms))
    if key not in _DESC:
        _DESC[key] = _describe(rep, ms)
    return _DESC[key]


def _describe(rep, ms):
    """A block's content: its units (cells with their in-edges and their atom's out-edges in the block), its roles
    (senders and receivers every unit shares with one weight), its bindings (components of units sharing other
    senders or receivers) with their patterns, and its raw edges (those outside the units)."""
    toy = rep.toy
    cells = {(rep.items[j][1], rep.items[j][2]) for j in ms if rep.items[j][0] == "c"}
    atom_of = {rep.off[l] + n: (l, n) for l, n in cells}
    units = {u: {"in": [], "out": []} for u in cells}
    raw = []
    rnd = lambda x: round(x / rep.scale, 6)  # noqa: E731
    for j in ms:
        kind, l, i, k = rep.items[j]
        if kind == "c":
            continue
        if (l, i) in units:
            units[(l, i)]["in"].append((("a", k), rnd(rep.w[j])))
        elif k in atom_of:
            units[atom_of[k]]["out"].append((("r", l, i), rnd(rep.w[j])))
        else:
            raw.append(j)
    roles = set()
    if len(units) >= 2:
        for side in ("in", "out"):
            seen = {}
            for u, d in units.items():
                for key, w in d[side]:
                    seen.setdefault(key, []).append(w)
            roles |= {(side, key) for key, ws in seen.items() if len(ws) == len(units) and len(set(ws)) == 1}
    # bindings: units joined by the senders or receivers they share outside the roles
    parent = {u: u for u in units}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    by_key = {}
    for u, d in units.items():
        for side in ("in", "out"):
            for key, _ in d[side]:
                if (side, key) not in roles:
                    by_key.setdefault((side, key), []).append(u)
    for us in by_key.values():
        for u in us[1:]:
            parent[find(u)] = find(us[0])
    comps = {}
    for u in units:
        comps.setdefault(find(u), []).append(u)
    bindings = []
    for us in comps.values():
        bindings.append((pattern(units, us, roles), us))
    return {"units": units, "roles": roles, "bindings": bindings, "raw": raw}


def pattern(units, us, roles):
    """A binding's canonical pattern: its units' weights, the roles by weight, its own senders and receivers as slots
    numbered in order of appearance, minimized over the orders of its units (up to 4; beyond, sorted)."""
    def enc(order):
        slots, out = {}, []
        for u in order:
            d = units[u]
            rec = [u[0]]
            for side in ("in", "out"):
                rw = sorted(w for key, w in d[side] if (side, key) in roles)
                own = []
                for key, w in sorted(((key, w) for key, w in d[side] if (side, key) not in roles), key=lambda kw: kw[1]):
                    slot = slots.setdefault((side, key), len(slots))
                    own.append((slot, key[0] if side == "in" else ("r", key[1]), w))
                rec.append((tuple(rw), tuple(own)))
            out.append(tuple(rec))
        return tuple(out)
    if len(us) <= 4:
        return min(enc(p) for p in itertools.permutations(us))
    return enc(sorted(us, key=lambda u: enc([u])))


def raw_bits(rep, js, B):
    """Raw edges, per map: receivers and senders named, numbers dense (zeros charged) or the edges named in the
    rectangle and their numbers."""
    bits = 0.0
    by_map = {}
    for j in js:
        _, l, i, k = rep.items[j]
        by_map.setdefault(l, []).append((i, k))
    for l, ed in by_map.items():
        R_, S_ = {i for i, _ in ed}, {k for _, k in ed}
        n_s = rep.off[l] if l < rep.toy.L else rep.n_atoms
        bits += name_set(rep.n_recv[l], len(R_)) + name_set(n_s, len(S_))
        rect = len(R_) * len(S_)
        bits += min(rect * B, log2C(rect, len(ed)) + len(ed) * B)
    return bits


def block_cost(rep, ms, count, B, rules_on=True):
    """A block's per-token content bits (all but its index) and its definition bits, given every pattern's count of
    bindings in the explanation (a pattern used twice or more is a rule)."""
    d = describe(rep, ms)
    is_rule = tuple(bool(rules_on and count.get(p, 0) >= 2) for p, _ in d["bindings"])
    key = (tuple(sorted(ms)), is_rule)
    if key not in _RAW:
        rawj = list(d["raw"])
        cells_raw = 0
        for (p, us), r in zip(d["bindings"], is_rule):
            if not r:
                cells_raw += len(us)
                unit_atoms = {rep.off[l] + n for l, n in us}
                for j in ms:
                    kind, l2, i, k = rep.items[j]
                    if kind == "e" and ((l2, i) in us or k in unit_atoms):
                        rawj.append(j)
        _RAW[key] = raw_bits(rep, rawj, B) + (name_set(sum(rep.m), cells_raw) if cells_raw else 0.0)
    role_names = sum(math.log2(rep.n_atoms) if side == "in" else math.log2(max(rep.n_recv[k_[1]], 2)) for side, k_ in d["roles"])
    bits = role_names + _RAW[key]
    dbits = B + role_names + _RAW[key]
    for (p, us), r in zip(d["bindings"], is_rule):
        if r:
            bits += math.log2(count[p])
            dbits += table_bits(rep, p, us)
    return bits, dbits


def patterns_of(rep, ms):
    return [p for p, _ in describe(rep, ms)["bindings"]]


def code(rep, blocks: Blocks, B, N, rules_on=True):
    """Per-token bits (array over the run's tokens) and the definitions' bits of an explanation (rules_on False: every
    binding raw, the dense code of the same blocks)."""
    count = {}
    for ms in blocks.members.values():
        for p in patterns_of(rep, ms):
            count[p] = count.get(p, 0) + 1
    rules = {p: c for p, c in count.items() if c >= 2 and rules_on}
    nB = len(blocks.members)
    per_tok = np.zeros(blocks.run.T)
    defs = rep.n_in * rep.A0.shape[0] * B   # the learned input atoms
    rule_on = {p: np.zeros(blocks.run.T, bool) for p in rules}
    per_block = {}
    for b, ms in blocks.members.items():
        bits, dbits = block_cost(rep, ms, count, B, rules_on)
        per_block[b] = bits + math.log2(max(nB, 2))
        per_tok += blocks.on[b] * per_block[b]
        defs += dbits
        for p in patterns_of(rep, ms):
            if p in rules:
                rule_on[p] |= blocks.on[b]
    for p in rules:
        per_tok += rule_on[p] * body_bits(p, B)
        defs += body_bits(p, B)
    return per_tok, defs, {"blocks": nB, "rules": len(rules), "bindings_in_rules": sum(rules.values()), "per_block": per_block, "rule_count": rules}


class State:
    """An explanation under search, with what a move's change of J needs: each block's costs, the patterns' counts and
    the blocks using each, the blocks on per token."""

    def __init__(self, rep, run, members, on, B, N):
        self.rep, self.run, self.B, self.N = rep, run, B, N
        self.members, self.on = dict(members), dict(on)
        self.count, self.users = {}, {}
        for b, ms in self.members.items():
            for p in patterns_of(rep, ms):
                self.count[p] = self.count.get(p, 0) + 1
                self.users.setdefault(p, set()).add(b)
        self.cost = {b: block_cost(rep, ms, self.count, B) for b, ms in self.members.items()}
        self.n_on = sum(on.astype(float) for on in self.on.values())
        self.fresh = max(self.members) + 1
        self.J = self.total()

    def total(self):
        pt, defs, _ = code(self.rep, self, self.B, self.N)
        return float(pt.mean()) + defs / self.N

    def rule_on(self, p, drop=(), add=()):
        on = np.zeros(self.run.T, bool)
        for b in self.users.get(p, ()):
            if b not in drop:
                on |= self.on[b]
        for ms, o in add:
            if p in patterns_of(self.rep, ms):
                on |= o
        return on

    def delta(self, c, parts, ons):
        """J after replacing blocks c by `parts` (gated by `ons`), minus J now."""
        rep, B = self.rep, self.B
        new_count = dict(self.count)
        for b in c:
            for p in patterns_of(rep, self.members[b]):
                new_count[p] -= 1
        part_pats = [patterns_of(rep, ms) for ms in parts]
        for ps in part_pats:
            for p in ps:
                new_count[p] = new_count.get(p, 0) + 1
        touched = {p for b in c for p in patterns_of(rep, self.members[b])} | {p for ps in part_pats for p in ps}
        changed = {p for p in touched if new_count.get(p, 0) != self.count.get(p, 0)}
        affected = set().union(*[self.users.get(p, set()) for p in changed]) - set(c) if changed else set()
        nB, nB2 = len(self.members), len(self.members) - len(c) + len(parts)
        n_on2 = self.n_on - sum(self.on[b] for b in c) + sum(ons)
        d_tok = math.log2(max(nB2, 2)) * n_on2 - math.log2(max(nB, 2)) * self.n_on
        d_def = 0.0
        for b in c:
            d_tok = d_tok - self.on[b] * self.cost[b][0]
            d_def -= self.cost[b][1]
        for ms, on in zip(parts, ons):
            bits, dbits = block_cost(rep, ms, new_count, B)
            d_tok = d_tok + on * bits
            d_def += dbits
        for b in affected:
            bits, dbits = block_cost(rep, self.members[b], new_count, B)
            d_tok = d_tok + self.on[b] * (bits - self.cost[b][0])
            d_def += dbits - self.cost[b][1]
        adds = list(zip(parts, ons))
        for p in touched:
            old_r, new_r = self.count.get(p, 0) >= 2, new_count.get(p, 0) >= 2
            if old_r or new_r:
                bb = body_bits(p, B)
                if old_r:
                    d_tok = d_tok - bb * self.rule_on(p)
                    d_def -= bb
                if new_r:
                    d_tok = d_tok + bb * self.rule_on(p, drop=c, add=adds)
                    d_def += bb
        return float(np.mean(d_tok)) + d_def / self.N

    def apply(self, c, parts, ons):
        for b in c:
            for p in patterns_of(self.rep, self.members[b]):
                self.count[p] -= 1
                self.users[p].discard(b)
            self.n_on = self.n_on - self.on[b]
            del self.members[b], self.on[b], self.cost[b]
        for ms, on in zip(parts, ons):
            b = self.fresh
            self.fresh += 1
            self.members[b], self.on[b] = ms, on
            for p in patterns_of(self.rep, ms):
                self.count[p] = self.count.get(p, 0) + 1
                self.users.setdefault(p, set()).add(b)
            self.n_on = self.n_on + on
        self.cost = {b: block_cost(self.rep, ms, self.count, self.B) for b, ms in self.members.items()}
        self.J = self.total()

    def gate(self, ms):
        return Blocks.gate(self, ms)


def body_bits(p, B):
    """A rule's body: its numbers (each unit's role and slot weights) and its slots' kinds."""
    nums = sum(len(rw) + len(own) for unit in p for rw, own in unit[1:])
    return nums * B + sum(len(own) for unit in p for _, own in unit[1:]) * 2 + len(p)


def table_bits(rep, p, us):
    """A binding's row in its rule's table: its units' neurons and its slots' atoms or receivers."""
    bits = len(us) * math.log2(sum(rep.m))
    slots = {s for unit in p for _, own in unit[1:] for s, _, _ in own}
    return bits + len(slots) * math.log2(rep.n_atoms)


# ------------------------------------------------------------------------------------------ explanations


def pieces(rep):
    """The start: each atom's fan-out into each map, each neuron's cell with its bias edge (from the mean atom)."""
    keys, assign = {}, []
    for it in rep.items:
        kind, l, i, k = it
        if kind == "c":
            key = ("cell", l, i)
        elif k == rep.const and l < rep.toy.L:
            key = ("cell", l, i)            # a bias edge goes with its neuron
        else:
            key = ("fan", l, k)
        assign.append(keys.setdefault(key, len(keys)))
    return np.array(assign)


def neurons(rep):
    """Rank-one neuron pieces (descent's rot arm's answer): each neuron one block, its in-edges, its cell and its atom's
    out-edges; edges from input atoms to the head each atom's own block."""
    assign = []
    for kind, l, i, k in rep.items:
        if kind == "c" or (kind == "e" and l < rep.toy.L):
            key = (l, i)
        elif k >= rep.n_in:
            lay = max(x for x in range(rep.toy.L) if rep.off[x] <= k)
            key = (lay, k - rep.off[lay])
        else:
            key = ("in", k)
        assign.append(key)
    ids = {key: x for x, key in enumerate(dict.fromkeys(assign))}
    return np.array([ids[key] for key in assign])


def truth(rep, dense_neurons=False):
    """The true blocks. gated_copy: per mechanism its neurons' cells, every edge into them and their atoms' out-edges.
    resid_mlp: per function the fan-out of the input atom matched to its embedding direction (into both layers' c_fc
    and the head), per neuron its cell with its bias and its atom's out-edges (dense_neurons: all neurons one block,
    as SPD's W_out). Items no mechanism names stay pieces."""
    toy = rep.toy
    start = pieces(rep)
    assign = start + 10_000
    mechs = toy.truth["mechanisms"]
    if toy.kind == "gated_copy":
        for mi, m in enumerate(mechs):
            dn = np.fromfile(toy.dir / m["operators"]["blocks.0.mlp.down_proj"]["file"], dtype="<f8").reshape(m["operators"]["blocks.0.mlp.down_proj"]["shape"])
            neurons = set(np.nonzero(np.abs(dn).sum(0) > 0)[0].tolist())
            atoms = {rep.off[0] + n for n in neurons}
            for j, (kind, l, i, k) in enumerate(rep.items):
                if (kind == "c" and i in neurons) or (kind == "e" and l == 0 and i in neurons) or (kind == "e" and k in atoms):
                    assign[j] = mi
    elif toy.kind == "resid_mlp":
        E = toy.head * toy.rec["config"]["head"]["task_residual"]       # the embedding rows (W_U = E^T)
        first = 0 if rep.const is None else 1
        Ain = rep.A0[:, first:]
        cos = np.abs((E / np.linalg.norm(E, axis=1, keepdims=True)) @ (Ain / np.linalg.norm(Ain, axis=0, keepdims=True)))
        match = first + cos.argmax(1)
        rep.atom_cos = cos.max(1)
        fn = {int(k): f for f, k in enumerate(match)}
        for j, (kind, l, i, k) in enumerate(rep.items):
            if kind == "e" and k in fn:
                assign[j] = fn[k]
            elif kind == "c" or (kind == "e" and k == rep.const and l < toy.L) or (kind == "e" and k >= rep.n_in):
                n_id = (l, i) if kind == "c" or k == rep.const else None
                if kind == "e" and k >= rep.n_in:
                    lay = max(x for x in range(toy.L) if rep.off[x] <= k)
                    n_id = (lay, k - rep.off[lay])
                assign[j] = 20_000 if dense_neurons else 30_000 + n_id[0] * 1000 + n_id[1]
    return assign


# ------------------------------------------------------------------------------------------------- search


def J_of(rep, blocks, B, N, rules_on=True):
    pt, defs, info = code(rep, blocks, B, N, rules_on)
    return float(pt.mean()) + defs / N, pt, defs, info


def exact_if_merged(rep, run, ms, on):
    """Whether gating the merged block by `on` keeps P = M: on its off tokens, every edge whose receiver's cell is
    outside the block writes nothing, and every cell in it is inactive."""
    off = ~on
    if not off.any():
        return True
    cells = {(rep.items[j][1], rep.items[j][2]) for j in ms if rep.items[j][0] == "c"}
    for j in ms:
        kind, l, i, k = rep.items[j]
        if kind == "c" or (l < rep.toy.L and (l, i) not in cells) or l == rep.toy.L:
            if np.abs(run.contrib(j)[off]).max() > 1e-9 * rep.scale:
                return False
    return True


def search(rep, run, assign, B, N, log=print, max_rounds=400, verbose=False):
    """Local search on J from `assign`: each round scores every merge (candidates) and every split (splits) by its
    change of J, then applies the improving ones on disjoint blocks, best first, each scored again just before (so
    every applied move lowers J), with every new block's gating exact; ends when no move lowers J."""
    blocks = Blocks(rep, run, assign)
    st = State(rep, run, blocks.members, blocks.on, B, N)
    log(f"  start: {len(st.members)} blocks, J {st.J:.1f}")
    bad = set()
    for rnd in range(max_rounds):
        moves = [(c, [[j for b in c for j in st.members[b]]]) for c in candidates(rep, st)]
        moves += [(frozenset([b]), parts) for b, parts in splits(rep, st)]
        moves += transfers(rep, st)
        scored = []
        for c, parts in moves:
            key = (tuple(sorted(c)), tuple(sorted(len(p) for p in parts)))
            if key in bad:
                continue
            ons = [st.gate(ms) for ms in parts]
            if not all(exact_if_merged(rep, run, ms, on) for ms, on in zip(parts, ons)):
                bad.add(key)
                continue
            dJ = st.delta(c, parts, ons)
            if dJ < -1e-9:
                scored.append((dJ, c, parts, ons))
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        used, applied = set(), 0
        for dJ, c, parts, ons in scored:
            if used & set(c):
                continue
            if applied and st.delta(c, parts, ons) >= -1e-9:
                continue
            J0 = st.J
            st.apply(c, parts, ons)
            used |= set(c)
            applied += 1
            if verbose:
                log(f"    {'merge' if len(parts) == 1 else 'split'} {len(c)} -> {[len(p) for p in parts]} items, on {[round(float(o.mean()), 3) for o in ons]}, J {J0:.2f} -> {st.J:.2f}")
        log(f"  round {rnd}: {applied} moves, now {len(st.members)} blocks, J {st.J:.1f}")
    log(f"  end: {len(st.members)} blocks, J {st.J:.1f}")
    assign = np.zeros(len(rep.items), dtype=np.int64)
    for b, ms in st.members.items():
        assign[ms] = b
    return assign, st


def splits(rep, blocks):
    """Splits to try: a block's bindings (with the edges into and out of their units) grouped by their own gates'
    tokens, the rest of its items one more block; and each such group peeled off alone."""
    out = []
    for b, ms in blocks.members.items():
        d = describe(rep, ms)
        if len(d["bindings"]) < 2:
            continue
        unit_of = {}
        for g, (_, us) in enumerate(d["bindings"]):
            for u in us:
                unit_of[u] = g
        groups = {}
        rest = []
        for j in ms:
            kind, l, i, k = rep.items[j]
            if kind == "c" or (l < rep.toy.L and (l, i) in unit_of):
                groups.setdefault(unit_of[(l, i)], []).append(j)
                continue
            src = next(((l2, n) for (l2, n) in unit_of if rep.off[l2] + n == k), None)
            if src is not None:
                groups.setdefault(unit_of[src], []).append(j)
            else:
                rest.append(j)
        by_on = {}
        for g, js in groups.items():
            by_on.setdefault(blocks.gate(js).tobytes(), []).extend(js)
        parts = list(by_on.values())
        if len(parts) < 2:
            continue
        out.append((b, parts + ([rest] if rest else [])))
        for p_ in parts:
            others = [j for j in ms if j not in set(p_)]
            out.append((b, [p_, others]))
    return out


def transfers(rep, blocks):
    """Moves of items between two blocks: the edges of a block that feed the cells another block holds, or that leave
    the atoms of its cells, into that block."""
    cell_block = {}
    for b, ms in blocks.members.items():
        for j in ms:
            if rep.items[j][0] == "c":
                cell_block[(rep.items[j][1], rep.items[j][2])] = b
    atom_block = {rep.off[l] + n: b for (l, n), b in cell_block.items()}
    out = []
    for b, ms in blocks.members.items():
        to = {}
        for j in ms:
            kind, l, i, k = rep.items[j]
            if kind != "e":
                continue
            x = cell_block.get((l, i)) if l < rep.toy.L else None
            if x is None or x == b:
                x = atom_block.get(k)
            if x is not None and x != b:
                to.setdefault(x, []).append(j)
        for x, js in to.items():
            moved = set(js)
            rest = [j for j in ms if j not in moved]
            parts = ([rest] if rest else []) + [blocks.members[x] + js]
            out.append((frozenset([b, x]), parts))
    return out


def candidates(rep, blocks):
    """Merges to try: a block with the blocks holding the cells its c_fc edges feed, or the edges out of its cells'
    atoms, or the edges into its cells; the blocks holding one atom's fan-outs and its cell; blocks that fire on the
    same tokens."""
    where = {}
    for b, ms in blocks.members.items():
        for j in ms:
            where[j] = b
    cell_block = {(rep.items[j][1], rep.items[j][2]): b for j, b in where.items() if rep.items[j][0] == "c"}
    atom_blocks = {}
    for j, b in where.items():
        kind, l, i, k = rep.items[j]
        a = rep.off[l] + i if kind == "c" else k
        if kind == "c" or k != rep.const:
            atom_blocks.setdefault(a, set()).add(b)
    edges_from, edges_into = {}, {}
    for j, b in where.items():
        kind, l, i, k = rep.items[j]
        if kind == "e":
            edges_from.setdefault(k, set()).add(b)
            if l < rep.toy.L:
                edges_into.setdefault((l, i), set()).add(b)
    out = set()
    for b, ms in blocks.members.items():
        fed = {cell_block[(rep.items[j][1], rep.items[j][2])] for j in ms
               if rep.items[j][0] == "e" and rep.items[j][1] < rep.toy.L and (rep.items[j][1], rep.items[j][2]) in cell_block}
        if fed - {b}:
            out.add(frozenset(fed | {b}))
        # the blocks holding the edges out of this block's cells' atoms, and those holding the edges into its cells
        cells = [(rep.items[j][1], rep.items[j][2]) for j in ms if rep.items[j][0] == "c"]
        fan = set().union(*[edges_from.get(rep.off[l] + n, set()) for l, n in cells]) if cells else set()
        feed = set().union(*[edges_into.get(c, set()) for c in cells]) if cells else set()
        for grp in (fan, feed, fan | feed):
            if grp - {b}:
                out.add(frozenset(grp | {b}))
    for a, bs in atom_blocks.items():
        if len(bs) > 1:
            out.add(frozenset(bs))
            for x, y in itertools.combinations(sorted(bs), 2):
                out.add(frozenset((x, y)))
    by_on = {}
    for b, on in blocks.on.items():
        by_on.setdefault(on.tobytes(), []).append(b)
    for bs in by_on.values():
        if len(bs) > 1:
            out.add(frozenset(bs))
            for x, y in itertools.combinations(sorted(bs), 2):
                out.add(frozenset((x, y)))
    return [c for c in out if len(c) > 1]


# ----------------------------------------------------------------------------------------------- scoring


def p_run_error(rep, R, members):
    """P's hard-gated run on R (gates from P's own run), its error against M in bits per token (Gaussian head in units
    of the task residual), and with every block on."""
    toy = rep.toy
    out = {}
    for mode in ("hard", "all"):
        s = rep.coeffs(R)
        gates = {}
        for l in range(toy.L + 1):
            if l < toy.L:
                pre_all = s @ rep.E[l].T
                a_all = toy.act(pre_all)
            # gates of the blocks whose stage is in this layer
            sub = Run.__new__(Run)
            sub.rep, sub.T = rep, R.shape[0]
            sub.s = [None] * (toy.L + 1)
            sub.a = [None] * toy.L
            sub.s[l] = s
            if l < toy.L:
                sub.a[l] = a_all
            for b, ms in members.items():
                if gate_stage(rep, ms)[0] == l:
                    w = own_write(rep, sub, ms) if mode == "hard" else np.ones(R.shape[0])
                    gates[b] = w > 1e-9 * max(1.0, float(w.max())) if mode == "hard" else np.ones(R.shape[0], bool)
            if l == toy.L:
                break
            g_item = {}
            for b, ms in members.items():
                for j in ms:
                    g_item[j] = b
            pre = np.zeros((R.shape[0], rep.m[l]))
            cellg = np.ones((R.shape[0], rep.m[l]))
            for j, (kind, l2, i, k) in enumerate(rep.items):
                if l2 != l:
                    continue
                g = gates.get(g_item[j])
                if g is None:   # a block gated later (a later layer's stage) is on until its gate
                    g = np.ones(R.shape[0], bool)
                if kind == "e":
                    pre[:, i] += g * rep.w[j] * s[:, k]
                else:
                    cellg[:, i] = g
            a = toy.act(pre) * cellg
            s = np.column_stack([s, a])
        y = np.zeros((R.shape[0], rep.H.shape[0])) + toy.bias
        for j, (kind, l2, i, k) in enumerate(rep.items):
            if kind == "e" and l2 == toy.L:
                y[:, i] += gates[g_item[j]] * rep.w[j] * s[:, k]
        y = np.maximum(y, 0.0) if toy.relu_head else y
        out[mode] = float(((y - toy.run(R)) ** 2).sum(1).mean() / (2 * LN2))
    return out


def summary(rep, run, name, assign, B, N, true_assign=None, rules_on=True):
    blocks = Blocks(rep, run, assign)
    J, pt, defs, info = J_of(rep, blocks, B, N, rules_on)
    sizes = sorted((len(ms) for ms in blocks.members.values()), reverse=True)
    rec = {"explanation": name, "blocks": info["blocks"], "rules": info["rules"], "bindings_in_rules": info["bindings_in_rules"],
           "bits_per_token": float(pt.mean()), "definitions_bits": float(defs), "J": J,
           "blocks_on_per_token": float(np.mean([on.mean() for on in blocks.on.values()]) * len(blocks.on)),
           "largest_blocks_items": sizes[:5]}
    if true_assign is not None:
        rec["recovery"] = recovery(rep, run, blocks, Blocks(rep, run, true_assign))
    return rec, blocks


def recovery(rep, run, found, true):
    """Per true block (of more than one item): the found block holding most of its items, the share of its items it
    holds (and of the found block's items that are the true block's), and their gates' agreement (Jaccard)."""
    where = {j: b for b, ms in found.members.items() for j in ms}
    rows = []
    for tb, ms in true.members.items():
        if len(ms) < 2 or tb >= 10_000 and tb < 20_000:
            continue
        hits = {}
        for j in ms:
            hits[where[j]] = hits.get(where[j], 0) + 1
        fb = max(hits, key=hits.get)
        on_t, on_f = true.on[tb], found.on[fb]
        rows.append({"true": int(tb), "items": len(ms), "found": int(fb), "found_items": len(found.members[fb]),
                     "recall": hits[fb] / len(ms), "precision": hits[fb] / len(found.members[fb]),
                     "gate_jaccard": float((on_t & on_f).sum() / max((on_t | on_f).sum(), 1)),
                     "layers": sorted({rep.items[j][1] for j in found.members[fb]})})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("toy")
    ap.add_argument("--bits", type=float, default=16.0)
    ap.add_argument("--out")
    ap.add_argument("--tokens", type=int, default=4096)
    a = ap.parse_args()
    toy = Toy(Path(a.toy))
    train = toy.wte[HELD:]
    rep = Rep(toy, train)
    print(f"{toy.name}: {rep.n_in} input atoms ({'one constant' if rep.const is not None else 'no constant'}), {sum(rep.m)} neurons, {len(rep.items)} items; "
          f"split error (stream, outputs) {rep.check_split(toy.wte[:HELD])}", flush=True)
    run = Run(rep, train[: a.tokens])
    N = train.shape[0]
    t_assign = truth(rep)
    res = []
    named = [("truth", t_assign, True), ("truth, dense code", t_assign, False), ("neuron pieces", neurons(rep), True),
             ("neuron pieces, dense code", neurons(rep), False), ("pieces (start)", pieces(rep), True)]
    if toy.kind == "resid_mlp":
        named.append(("truth, neurons one block", truth(rep, True), True))
    for name, asg, ro in named:
        rec, _ = summary(rep, run, name, asg, a.bits, N, rules_on=ro)
        res.append(rec)
    print("search:", flush=True)
    found, _ = search(rep, run, pieces(rep), a.bits, N)
    rec, fb = summary(rep, run, "found", found, a.bits, N, t_assign)
    res.append(rec)
    held = toy.wte[:HELD][:1024]
    for r_, asg in zip(res, [x[1] for x in named] + [found]):
        r_["kl_bits_per_token"] = p_run_error(rep, held, Blocks(rep, Run(rep, held[:8]), asg).members)
    for r_ in res:
        rr = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r_.items() if k != "recovery"}
        print(json.dumps(rr), flush=True)
    rows = res[-1]["recovery"]
    exact = sum(r["recall"] == 1.0 and r["precision"] == 1.0 for r in rows)
    print(f"recovery: {exact} of {len(rows)} true blocks found exactly; mean recall {np.mean([r['recall'] for r in rows]):.3f}, "
          f"precision {np.mean([r['precision'] for r in rows]):.3f}, gate Jaccard {np.mean([r['gate_jaccard'] for r in rows]):.3f}")
    for r in rows:
        if r["recall"] < 1.0 or r["precision"] < 1.0:
            print("  not exact:", r)
    if getattr(rep, "atom_cos", None) is not None:
        print(f"input atoms against the embedding directions: |cos| mean {rep.atom_cos.mean():.4f}, min {rep.atom_cos.min():.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
