"""The structural code on the toy gate's real-valued MLP toys (#2951 design, 10-07): a standalone reference
prototype, built to scale linearly in the model: no dense all-pairs comparison, exactness through local solves,
the dictionary from the weights' own vectors, execution batched over items, guards that are small reads.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/structural.py TOY_DIR [--bits B] [--gates STEPS] [--out OUT.json]

Dictionary of the stream (one, shared across layers), from the weights' own vectors: the embedding's rows (each
input's write), every neuron's write (its down_proj column) and the head's rows (its reads). Directions two or more
of these vectors share (unit vectors hashed at B-bit precision: no pairwise comparison) are the candidate atoms.
Each write vector is coded greedily on them, every step a least-squares solve on its own small support (a local
frame, never a pseudo-inverse of a map), and what remains on coordinate axes, so every code is exact. A vector whose
code would cost more than the vector (more than d B / (log2 d + B) axes) is its own atom (its code: that atom, weight 1). An input's stream
coefficients are its embedding row's code and a neuron writes its own code, so the stream at every layer is A s
exactly.

Items: read edges (a c_fc row's, or a head row's, inner product with an atom it can meet: w_i . a_k, nonzero only)
and write edges (a neuron's code entries). With every item on, P is M (checked at every evaluation). A block is any
set of items, in any maps and layers, with one guard: the norm of its own write at its gate stage (the last stage
of its earliest layer: the pre-activations its read edges write, after the activation its neurons' writes, or the
head's outputs), on above zero to rounding: a read of the block's own output, no gate network.

Code, in bits (numbers at B bits, an index at log2 of its dictionary, k of n named at log2 C(n, k) + log2(n + 1)).
A block's units are its neurons that write (with their read edges in, their writes, and the reads of the atoms they
write); senders every unit shares with one weight are the block's roles; the rest join units into bindings. A
binding's pattern is its weights (quantized at B bits) with its own atoms and receivers as slots; a pattern two or
more bindings share is a rule (found by hashing patterns: the candidate list is the pattern's bucket).
- A number costs B bits, a weight of exactly +-1 its sign.
- Per token, each block on: log2(#blocks), its roles' names, its raw items (names and numbers), log2(#bindings of
  its rule) per binding; each rule used by a block on: its body once.
- Definitions, paid once through F: the learned atoms, each block's threshold, roles and raw items, each binding's
  table row, each rule's body. M's own maps are the dictionary every explanation shares.

Search: from the pieces (each atom's reads per map, each neuron with its bias and its writes), moves from short
candidate lists: a block with the blocks its edges feed, or that feed its neurons, or that read its neurons'
atoms; all blocks reading one atom; all blocks firing on the same tokens, and each with the next such block; a
block's bindings split by their tokens; a block's edges moved to the block holding the neurons they feed; and every
move of one shape made at once (moves whose blocks share a signature, found by hashing: a rule's bindings change
together and keep their rule). Each is scored by its change of J = E_t[bits per token] + definitions / N from aggregates (no full recount), applied best
first on disjoint blocks, kept only if every new block's gating stays exact. A merge whose guard reads the same stage
as some merged blocks' guards is first scored with their union as its tokens, and dropped if that loses."""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

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
        self.name, self.kind, self.dir = d.name, rec.get("real_valued"), d
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

    def run(self, R):
        r = R
        for l in range(self.L):
            r = r + self.act(r @ self.fc[l].T) @ self.dn[l].T
        y = r @ self.head.T + self.bias
        return np.maximum(y, 0.0) if self.relu_head else y


# ----------------------------------------------------------------------------------------------- dictionary


def unit_rows(V):
    """Rows as unit vectors signed so their largest entry (the first, on ties) is positive, and their scales."""
    n = np.linalg.norm(V, axis=1)
    U = V / np.maximum(n, 1e-300)[:, None]
    top = np.abs(U).argmax(1)
    sgn = np.sign(U[np.arange(len(U)), top])
    return U * sgn[:, None], n * sgn


def recurring(V, B):
    """Directions two or more rows of V share, by hashing their unit vectors at B-bit precision."""
    U, n = unit_rows(V[np.linalg.norm(V, axis=1) > 0])
    keys = np.round(U * 2.0 ** B).astype(np.int64)
    groups = {}
    for x, key in enumerate(map(bytes, keys)):
        groups.setdefault(key, []).append(x)
    return np.array([U[xs].mean(0) for xs in groups.values() if len(xs) >= 2]).reshape(-1, V.shape[1])


class Dictionary:
    """The stream's atoms: recurring directions, coordinate axes and own atoms (module note), as dense unit columns
    (axes included: the toys are small; at scale an axis is its index)."""

    def __init__(self, toy: Toy, B):
        d = toy.wte.shape[1]
        writes = [toy.wte] + [w.T for w in toy.dn]
        R = recurring(np.vstack(writes + [toy.head]), B)
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        self.d, self.B = d, B
        self.R = R                                  # recurring atoms (unit rows)
        self.atoms = {}                             # key -> column index; key ('r', i), ('x', j) or ('o', name)
        self.cols = []
        self.limit = d * B / (math.log2(d) + B)     # an axes code longer than this: the vector is its own atom
        self.own_count = 0
        self.codes = {}
        self.input_codes = self.code(toy.wte, "input")
        self.neuron_codes = [self.code(w.T, f"neuron{l}") for l, w in enumerate(toy.dn)]
        self.A = np.column_stack(self.cols) if self.cols else np.zeros((d, 0))

    def atom(self, key, vec=None):
        """The column of atom `key` (an axis ('x', j) is built from its index; others from vec)."""
        if key not in self.atoms:
            self.atoms[key] = len(self.cols)
            if key[0] == "x":
                vec = np.zeros(self.d)
                vec[key[1]] = 1.0
            self.cols.append(vec)
        return self.atoms[key]

    def code(self, V, tag):
        """Exact sparse codes of V's rows: (rows, atom index, coefficient) triples."""
        rows, idx, val = [], [], []
        corr = V @ self.R.T if len(self.R) else np.zeros((len(V), 0))
        for r in range(len(V)):
            v = V[r]
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            support, res = [], v.copy()
            c = corr[r].copy()
            coef = np.zeros(0)
            while len(support) < 64 and len(self.R):
                if support:
                    c = self.R @ res
                k = int(np.abs(c).argmax())
                if abs(c[k]) <= 1e-9 * nv or k in support:
                    break
                support.append(k)
                S = self.R[support].T
                coef = np.linalg.lstsq(S, v, rcond=None)[0]
                res = v - S @ coef
                if np.linalg.norm(res) <= 1e-10 * nv:
                    break
            res[np.abs(res) <= 1e-12 * nv] = 0.0
            axes = np.nonzero(res)[0]
            if len(support) + len(axes) > self.limit:
                k = self.atom(("o", f"{tag}:{r}"), v.copy())   # the vector itself: its code is the atom, weight 1
                self.own_count += 1
                rows.append(r), idx.append(k), val.append(1.0)
                continue
            for k, cf in zip(support, coef):
                if abs(cf) > 1e-12 * nv:
                    nz = np.nonzero(self.R[k])[0]
                    if len(nz) == 1:   # a recurring direction that is an axis is named as the axis
                        rows.append(r), idx.append(self.atom(("x", int(nz[0])))), val.append(cf * self.R[k, nz[0]])
                    else:
                        rows.append(r), idx.append(self.atom(("r", int(k)), self.R[k])), val.append(cf)
            for j in axes:
                rows.append(r), idx.append(self.atom(("x", int(j)))), val.append(res[j])
        return np.array(rows, dtype=np.int64), np.array(idx, dtype=np.int64), np.array(val)

    def bits(self):
        """The learned atoms' definition: a recurring or own atom its d numbers, an axis its index."""
        learned = sum(1 for key in self.atoms if key[0] in ("r", "o"))
        return learned * self.d * self.B + len(self.atoms) * math.log2(max(self.d, 2))


# ---------------------------------------------------------------------------------------------- the split


class Rep:
    """The items every explanation partitions into blocks: read edges ('r', map, receiver, atom) and write edges
    ('w', layer, neuron, atom), with their weights."""

    def __init__(self, toy: Toy, B):
        t0 = time.time()
        self.toy, self.B = toy, B
        D = self.D = Dictionary(toy, B)
        L, self.m = toy.L, [w.shape[0] for w in toy.fc]
        self.C = D.A.shape[1]
        rows, idx, val = D.input_codes
        self.S0 = sp.csr_matrix((val, (rows, idx)), shape=(toy.wte.shape[0], self.C))
        readable = set(idx.tolist())
        items, w = [], []
        self.readable = []
        for l in range(L + 1):
            M_ = toy.fc[l] if l < L else toy.head
            ks = np.array(sorted(readable), dtype=np.int64)
            self.readable.append(ks)
            E = M_ @ D.A[:, ks]
            tol = 1e-12 * max(np.abs(E).max(), 1e-300)
            for i, x in zip(*np.nonzero(np.abs(E) > tol)):
                items.append(("r", l, int(i), int(ks[x])))
                w.append(float(E[i, x]))
            if l < L:
                nr, nk, nv = D.neuron_codes[l]
                for n, k, c in zip(nr, nk, nv):
                    items.append(("w", l, int(n), int(k)))
                    w.append(float(c))
                readable |= set(nk.tolist())
        self.items, self.w = items, np.array(w)
        self.n_recv = self.m + [toy.head.shape[0]]
        self.scale = float(np.abs(self.w).max())
        self.unit = int(round(1.0 / self.scale * 2.0 ** B))   # a weight of exactly 1, quantized
        self.const = None
        # the constant atom: one every input carries with one coefficient (its read edges are biases)
        col = self.S0.tocsc()
        for k in range(self.C):
            v = col[:, k].toarray().ravel()
            if np.all(v != 0) and np.ptp(v) <= 1e-12 * abs(v[0]):
                self.const = k
                break
        self.own_atom = {}
        for key, k in D.atoms.items():
            if key[0] == "o" and key[1].startswith("neuron"):
                l_, n_ = key[1][6:].split(":")
                self.own_atom[(int(l_), int(n_))] = k
        self.kind = np.array([it[0] == "w" for it in items])
        self.layer = np.array([it[1] for it in items])
        self.recv = np.array([it[2] for it in items])
        self.atom = np.array([it[3] for it in items])
        self.seconds = time.time() - t0


EPS = 1e-9   # a guard is on where its block's own write exceeds this (to rounding: exact zeros are off)


def m_run(rep, tokens):
    """M's run in atom coefficients through the items (every one on): per layer the stream coefficients and the
    activations, and the outputs."""
    prog = Program(rep, {0: list(range(len(rep.items)))})
    ss, aa = [], []
    y = prog.run(rep.S0[tokens].toarray(), "all", ss, aa)
    return ss, aa, y


def scatter(vals, to, n):
    """Sum the columns of vals into n columns by `to` (one sparse product)."""
    S = sp.csr_matrix((np.ones(len(to)), (np.arange(len(to)), to)), shape=(len(to), n))
    return np.asarray(vals @ S) if vals.shape[1] else np.zeros((vals.shape[0], n))


class Run:
    def __init__(self, rep, tokens):
        self.rep, self.T = rep, len(tokens)
        self.s, self.a, self.y = m_run(rep, tokens)
        self.tokens = tokens

    def live(self, l, k):
        """Tokens on which atom k's coefficient is nonzero at layer l's input (cached)."""
        key = (l, k)
        if not hasattr(self, "_live"):
            self._live = {}
        if key not in self._live:
            self._live[key] = self.s[min(l, len(self.s) - 1)][:, k] != 0
        return self._live[key]

    def contrib(self, js):
        """Items' contributions per token (tokens x items): a read edge w s_k, a write edge c a_n."""
        rep = self.rep
        js = np.asarray(js)
        out = np.empty((self.T, len(js)))
        for l in set(rep.layer[js].tolist()):
            sel = rep.layer[js] == l
            jj = js[sel]
            wr = rep.kind[jj]
            vals = np.empty((self.T, len(jj)))
            if (~wr).any():
                vals[:, ~wr] = self.s[l][:, rep.atom[jj[~wr]]] * rep.w[jj[~wr]]
            if wr.any():
                vals[:, wr] = self.a[l][:, rep.recv[jj[wr]]] * rep.w[jj[wr]]
            out[:, sel] = vals
        return out


def gate_stage(rep, ms):
    ms = np.asarray(ms)
    l0 = int(rep.layer[ms].min())
    st = int(bool((rep.kind[ms] & (rep.layer[ms] == l0)).any()))
    return l0, st


def own_write(rep, run, ms):
    """The block's own write norm per token at its gate stage: its items there summed per receiver (a neuron's
    pre-activation, an atom's coefficient, an output)."""
    ms = np.asarray(ms)
    l0, st = gate_stage(rep, ms)
    js = ms[(rep.layer[ms] == l0) & (rep.kind[ms] == bool(st))]
    to = rep.atom[js] if st else rep.recv[js]
    u, inv = np.unique(to, return_inverse=True)
    acc = scatter(run.contrib(js), inv, len(u))
    return np.sqrt((acc ** 2).sum(1))


def gate_of(rep, run, ms):
    return own_write(rep, run, ms) > EPS


class Program:
    """P compiled for batched execution: per map, its items grouped by (block, sender): each group's gated sender
    column, one sparse product to the receivers; each stage's guards from the same groups, summed per (block,
    receiver) by one sparse product and per block by another. Per token the work is the items' count, a constant
    factor of M's."""

    def __init__(self, rep, members):
        self.rep = rep
        self.ids = sorted(members)
        blk = np.zeros(len(rep.items), dtype=np.int64)
        for x, b in enumerate(self.ids):
            blk[members[b]] = x
        stage = np.array([gate_stage(rep, members[b]) for b in self.ids]).reshape(-1, 2)
        L = rep.toy.L
        self.read, self.write, self.gread, self.gwrite = [], [], [], []
        for l in range(L + 1):
            r_ = np.nonzero((~rep.kind) & (rep.layer == l))[0]
            self.read.append(self.groups(blk[r_], rep.atom[r_], rep.recv[r_], rep.w[r_], rep.n_recv[l]))
            here = np.nonzero((stage[:, 0] == l) & (stage[:, 1] == 0))[0]
            sel = r_[np.isin(blk[r_], here)]
            self.gread.append((here, self.guard_groups(blk[sel], rep.atom[sel], rep.recv[sel], rep.w[sel])))
            if l < L:
                w_ = np.nonzero(rep.kind & (rep.layer == l))[0]
                self.write.append(self.groups(blk[w_], rep.recv[w_], rep.atom[w_], rep.w[w_], rep.C))
                here = np.nonzero((stage[:, 0] == l) & (stage[:, 1] == 1))[0]
                sel = w_[np.isin(blk[w_], here)]
                self.gwrite.append((here, self.guard_groups(blk[sel], rep.recv[sel], rep.atom[sel], rep.w[sel])))

    @staticmethod
    def groups(b, src, dst, w, n_dst):
        key = b * (int(src.max(initial=0)) + 1) + src
        u, inv = np.unique(key, return_inverse=True)
        gb, gs = u // (int(src.max(initial=0)) + 1), u % (int(src.max(initial=0)) + 1)
        M = sp.csr_matrix((w, (inv, dst)), shape=(len(u), n_dst))
        return gb, gs, M

    def guard_groups(self, b, src, dst, w):
        if not len(b):
            return None
        key_g = b * (int(src.max()) + 1) + src
        ug, inv_g = np.unique(key_g, return_inverse=True)
        gs = ug % (int(src.max()) + 1)
        key_r = b * (int(dst.max()) + 1) + dst
        ur, inv_r = np.unique(key_r, return_inverse=True)
        M = sp.csr_matrix((w, (inv_g, inv_r)), shape=(len(ug), len(ur)))
        to_block = sp.csr_matrix((np.ones(len(ur)), (np.arange(len(ur)), ur // (int(dst.max()) + 1))), shape=(len(ur), len(self.ids)))
        return gs, M, to_block

    @staticmethod
    def apply(g, x, grp):
        gb, gs, M = grp
        if not len(gb):
            return np.zeros((x.shape[0], M.shape[1]))
        return np.asarray((g[:, gb] * x[:, gs]) @ M)

    def guards(self, g, x, gg):
        here, G = gg
        if G is None:
            return
        gs, M, to_block = G
        own = np.sqrt(np.asarray((np.asarray(x[:, gs] @ M) ** 2) @ to_block))
        g[:, here] = own[:, here] > EPS

    def run(self, s, mode="hard", ss=None, aa=None):
        rep, toy = self.rep, self.rep.toy
        g = np.ones((s.shape[0], len(self.ids)))
        for l in range(toy.L + 1):
            if ss is not None:
                ss.append(s)
            if mode == "hard":
                self.guards(g, s, self.gread[l])
            out = self.apply(g, s, self.read[l])
            if l == toy.L:
                y = out + toy.bias
                return np.maximum(y, 0.0) if toy.relu_head else y
            if mode == "hard":
                self.guards(g, toy.act(self.apply(np.ones_like(g), s, self.read[l])), self.gwrite[l])
            a = toy.act(out)
            if aa is not None:
                aa.append(a)
            s = s + self.apply(g, a, self.write[l])


# --------------------------------------------------------------------------------------------- the code


_DESC, _RAW = {}, {}


def describe(rep, ms):
    key = tuple(sorted(ms))
    if key not in _DESC:
        _DESC[key] = _describe(rep, list(key))
    return _DESC[key]


def _describe(rep, ms):
    """Units (neurons writing in the block, with their read edges in and their writes), the block's reads of atoms its
    units write (each with the binding of the writers), roles, bindings with their patterns, raw items."""
    q = lambda x: int(round(x / rep.scale * 2.0 ** rep.B))  # noqa: E731
    units, writers = {}, {}
    for j in ms:
        if rep.kind[j]:
            u = (int(rep.layer[j]), int(rep.recv[j]))
            units.setdefault(u, {"in": [], "w": [], "items": []})
            writers.setdefault(int(rep.atom[j]), set()).add(u)
    raw, reads = [], {}
    for j in ms:
        l, i, k = int(rep.layer[j]), int(rep.recv[j]), int(rep.atom[j])
        if rep.kind[j]:
            units[(l, i)]["w"].append((("a", k), q(rep.w[j])))
            units[(l, i)]["items"].append(j)
        elif (l, i) in units:
            units[(l, i)]["in"].append((("a", k), q(rep.w[j])))
            units[(l, i)]["items"].append(j)
        elif k in writers and all(u[0] < l for u in writers[k]):
            reads.setdefault(k, []).append(j)
        else:
            raw.append(j)
    roles = set()
    if len(units) >= 2:
        seen = {}
        for d in units.values():
            for key, w in d["in"]:
                seen.setdefault(key, []).append(w)
        roles = {key for key, ws in seen.items() if len(ws) == len(units) and len(set(ws)) == 1}
    parent = {u: u for u in units}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    by_key = {}
    for u, d in units.items():
        for key, _ in d["in"] + d["w"]:
            if key not in roles:
                by_key.setdefault(key, []).append(u)
    for us in by_key.values():
        for u in us[1:]:
            parent[find(u)] = find(us[0])
    comps = {}
    for u in units:
        comps.setdefault(find(u), []).append(u)
    comp_of = {u: r for r, us in comps.items() for u in us}
    comp_reads = {}
    for k, js in reads.items():
        comp_reads.setdefault(comp_of[min(writers[k])], []).extend(js)
    bindings = []
    for r, us in comps.items():
        rd = [(("a", int(rep.atom[j])), ("r", int(rep.layer[j]), int(rep.recv[j])), q(rep.w[j])) for j in comp_reads.get(r, [])]
        items = [j for u in us for j in units[u]["items"]] + comp_reads.get(r, [])
        bindings.append((pattern(units, us, roles, rd), us, items))
    return {"units": units, "roles": roles, "bindings": bindings, "raw": raw}


def pattern(units, us, roles, reads):
    """A binding's canonical pattern: per unit its layer, its roles' weights, its own read edges and its writes, then
    the binding's reads of the atoms it writes, with its atoms and receivers as slots numbered in order of
    appearance; minimized over the orders of its units (up to 4; beyond, sorted by their own encodings). Ends with
    the slots' count. A binding of more than GROUP units has no pattern another can share (it stays raw)."""
    def enc(order):
        slots, out = {}, []
        for u in order:
            d = units[u]
            rw = tuple(sorted(w for key, w in d["in"] if key in roles))
            own_in = tuple((slots.setdefault(key, len(slots)), w) for key, w in sorted(((k_, w) for k_, w in d["in"] if k_ not in roles), key=lambda kw: kw[1]))
            wr = tuple((slots.setdefault(key, len(slots)), w) for key, w in sorted(d["w"], key=lambda kw: kw[1]))
            out.append((u[0], rw, own_in, wr))
        rd = tuple(sorted((slots.setdefault(ak, len(slots)), slots.setdefault(rk, len(slots)), w) for ak, rk, w in sorted(reads, key=lambda x: (x[2], x[1]))))
        return (tuple(out), rd, len(slots))
    if len(us) > GROUP:
        return ("raw", tuple(sorted(us)))   # beyond a group of GROUP units a binding is no rule's
    if len(us) <= 4:
        return min(enc(p) for p in itertools.permutations(us))
    return enc(sorted(us, key=lambda u: enc([u])))


GROUP = 32   # the most units a rule's binding holds (local: canonical forms cost the group, never the model)


COST = "bits"   # the per-token count: "bits" (description bits) or "concepts" (the 10-08 redesign)


def body_bits(p, B, unit=None):
    """A rule's body: its numbers (a weight of exactly +-1 costs its sign; `unit` its quantized value) and its units'
    and slots' structure; under COST "concepts", its free numbers."""
    units, rd, n_slots = p
    ws = [w for u in units for w in u[1]] + [w for u in units for part in (u[2], u[3]) for _, w in part] + [w for _, _, w in rd]
    if COST == "concepts":
        return float(len(ws))
    return sum(1 if unit is not None and abs(w) == unit else B for w in ws) + len(units) + n_slots


def index_bits(n):
    """A block's index per token: log2 of the blocks under bits, none under concepts (a reader holds no names)."""
    return 0.0 if COST == "concepts" else math.log2(max(n, 2))


def library_bits(rep):
    return 0.0 if COST == "concepts" else rep.D.bits()


def number_bits(vals, B, tol=1e-12):
    """Numbers at B bits each, an entry of exactly +-1 its sign."""
    vals = np.asarray(vals, dtype=float)
    return float(np.where(np.abs(np.abs(vals) - 1.0) <= tol, 1.0, float(B)).sum())


def code_cost(D, v):
    """The bits of v's exact code on dictionary D (as Dictionary.code codes a write, without adding atoms): its atoms'
    names and numbers, or, where that costs more than v itself, v's d numbers."""
    nv = np.linalg.norm(v)
    if nv == 0:
        return 0.0
    support, res, coef = [], v.copy(), np.zeros(0)
    c = D.R @ v if len(D.R) else np.zeros(0)
    while len(support) < 64 and len(D.R):
        if support:
            c = D.R @ res
        k = int(np.abs(c).argmax())
        if abs(c[k]) <= 1e-9 * nv or k in support:
            break
        support.append(k)
        coef = np.linalg.lstsq(D.R[support].T, v, rcond=None)[0]
        res = v - D.R[support].T @ coef
        if np.linalg.norm(res) <= 1e-10 * nv:
            break
    res[np.abs(res) <= 1e-12 * nv] = 0.0
    axes = np.nonzero(res)[0]
    n = len(support) + len(axes)
    if n > D.limit:
        return float(len(v) * D.B)
    C = max(len(D.atoms), 2)
    return name_set(C, n) + number_bits(np.concatenate([coef, res[axes]]), D.B)


def rank_one_bits(rep, l, kind, u, v):
    """A rank-one piece u v^T of layer l's c_fc (kind 'fc': u over its neurons, v over the stream) or down_proj
    (kind 'dn': u over the stream, v over its neurons) under the same code (the outer-product core form): c_fc, its
    receivers (u != 0) and its senders (the atoms the read v^T A_l touches) named, nnz(u) + nnz(v^T A_l) numbers;
    down_proj, its senders (v != 0) named with nnz(v) numbers and its write u coded exactly on the dictionary
    (code_cost). Excludes the per-token index (pieces_bits_per_token adds it)."""
    B = rep.B
    if kind == "fc":
        r = v @ rep.D.A[:, rep.readable[l]]
        r = r[np.abs(r) > 1e-12 * max(np.abs(r).max(), 1e-300)]
        nu = u[u != 0]
        return name_set(rep.m[l], len(nu)) + name_set(rep.C, len(r)) + number_bits(nu, B) + number_bits(r, B)
    nv = v[v != 0]
    return name_set(rep.m[l], len(nv)) + number_bits(nv, B) + code_cost(rep.D, np.asarray(u, dtype=float))


def pieces_bits_per_token(rep, pieces, ons):
    """Per-token bits of an explanation by rank-one pieces [(layer, kind, u, v)] gated by ons (tokens x pieces,
    True where a piece is on, e.g. VPD's causal importance > 0): each piece on costs log2(#pieces) and its
    rank_one_bits. Returns the mean over tokens and each piece's bits."""
    per = np.array([rank_one_bits(rep, l, k, u, v) for l, k, u, v in pieces])
    ons = np.asarray(ons, dtype=float)
    return float((ons @ (per + math.log2(max(len(pieces), 2)))).mean()), per


def raw_bits(rep, js, B, ms=()):
    """Raw items, per map and kind: receivers and senders named, numbers dense (zeros charged) or the items named
    in their rectangle and their numbers. Names are relative to what the block (ms) already holds: a neuron's own
    atom is implied by the neuron, and an atom a neuron of the block writes is named among the block's atoms."""
    bits = 0.0
    groups = {}
    for j in js:
        groups.setdefault((bool(rep.kind[j]), int(rep.layer[j])), []).append(j)
    block_atoms = {int(rep.atom[j]) for j in ms if rep.kind[j]}
    for (wr, l), g in groups.items():
        R_ = set(rep.recv[g].tolist())
        if wr:
            S_ = {int(rep.atom[j]) for j in g if rep.own_atom.get((l, int(rep.recv[j]))) != int(rep.atom[j])}
            inner = set()
        else:
            S_all = set(rep.atom[g].tolist())
            inner = S_all & block_atoms
            S_ = S_all - inner
        bits += name_set(rep.n_recv[l], len(R_)) + (name_set(rep.C, len(S_)) if S_ else 0.0)
        bits += name_set(len(block_atoms), len(inner)) if inner else 0.0
        rect = len(R_) * len(set(rep.atom[g].tolist()))
        nums = sum(1 if abs(abs(rep.w[j]) - 1.0) <= 1e-12 else B for j in g)
        bits += min(rect * B, log2C(rect, len(g)) + nums)
    return bits


def block_concepts(rep, ms, rule_status):
    """A block's concepts (the redesign's count): 1 for its gate, the directions it reads (nodes its items read and
    none of them writes: atoms, or neurons whose input is outside it), the directions it writes (nodes its items
    write and none of them reads: atoms, neurons, outputs), and the free numbers of its core: its items' weights
    outside rule bindings (a rule's body is counted once per token by its rule), a neuron's write of its own atom
    at weight 1 being no number (it defines the atom)."""
    d = describe(rep, ms)
    srcs, tgts = set(), set()
    for j in ms:
        l, i, k = int(rep.layer[j]), int(rep.recv[j]), int(rep.atom[j])
        if rep.kind[j]:
            srcs.add(("n", l, i))
            tgts.add(("a", k))
        else:
            srcs.add(("a", k))
            tgts.add(("n", l, i) if l < rep.toy.L else ("o", i))
    raw = list(d["raw"]) + [j for (p, us, js), r in zip(d["bindings"], rule_status) if not r for j in js]
    nums = sum(0 if rep.kind[j] and rep.own_atom.get((int(rep.layer[j]), int(rep.recv[j]))) == int(rep.atom[j]) and abs(rep.w[j] - 1.0) <= 1e-12 else 1
               for j in raw)
    return float(1 + len(srcs - tgts) + len(tgts - srcs) + nums), 0.0


def interface(rep, ms):
    """The directions a block reads and writes (block_concepts's nodes)."""
    srcs, tgts = set(), set()
    for j in ms:
        l, i, k = int(rep.layer[j]), int(rep.recv[j]), int(rep.atom[j])
        if rep.kind[j]:
            srcs.add(("n", l, i))
            tgts.add(("a", k))
        else:
            srcs.add(("a", k))
            tgts.add(("n", l, i) if l < rep.toy.L else ("o", i))
    return srcs - tgts, tgts - srcs


def total_concepts(rep, st):
    """The whole decomposition's concepts at the library level: every distinct direction read or written by any part
    counted once, each part's gate and its core's free numbers outside rule bindings, each rule's body once."""
    dirs, total = set(), 0.0
    for b, ms in st.members.items():
        r_, w_ = interface(rep, ms)
        dirs |= r_ | w_
        status = st.status(ms)
        d = describe(rep, ms)
        raw = list(d["raw"]) + [j for (p, us, js), r in zip(d["bindings"], status) if not r for j in js]
        total += 1 + sum(0 if rep.kind[j] and rep.own_atom.get((int(rep.layer[j]), int(rep.recv[j]))) == int(rep.atom[j]) and abs(rep.w[j] - 1.0) <= 1e-12 else 1
                         for j in raw)
    saved = COST
    globals()["COST"] = "concepts"
    bodies = sum(body_bits(p, rep.B) for p, c in st.count.items() if c >= 2)
    globals()["COST"] = saved
    return {"total_concepts": float(len(dirs) + total + bodies), "distinct_directions": len(dirs), "parts": len(st.members),
            "gates_and_core_numbers": float(total), "rule_bodies": float(bodies)}


def block_cost(rep, ms, rule_status):
    """(per-token content bits without the block's index and its rules' binding indices, definition bits)
    given which of its bindings are rules; under COST "concepts", block_concepts."""
    if COST == "concepts":
        key = ("concepts", tuple(sorted(ms)), tuple(rule_status))
        if key not in _RAW:
            _RAW[key] = block_concepts(rep, ms, rule_status)
        return _RAW[key]
    d = describe(rep, ms)
    key = (tuple(sorted(ms)), tuple(rule_status))
    if key not in _RAW:
        rawj = list(d["raw"])
        for (p, us, js), r in zip(d["bindings"], rule_status):
            if not r:
                rawj += js
        rb = raw_bits(rep, rawj, rep.B, ms)
        role_names = len(d["roles"]) * math.log2(rep.C)
        table = sum(len(us) * math.log2(sum(rep.m)) + p[2] * math.log2(rep.C) for (p, us, js), r in zip(d["bindings"], rule_status) if r)
        _RAW[key] = (role_names + rb, rep.B + role_names + rb + table)
    return _RAW[key]


class State:
    """An explanation (blocks and their guards' tokens) with the aggregates a move's change of J reads: per block
    its costs; per pattern its bindings' count, its users, per token its bindings on and its users on."""

    def __init__(self, rep, run, members, N):
        self.rep, self.run, self.N, self.B = rep, run, N, rep.B
        self.members = {b: list(ms) for b, ms in members.items()}
        self.on = {b: gate_of(rep, run, ms) for b, ms in self.members.items()}
        self.fresh = max(self.members) + 1
        self.count, self.users = {}, {}
        self.bind_on, self.users_on = {}, {}
        for b, ms in self.members.items():
            self._add_patterns(b, +1)
        self.cost = {b: block_cost(rep, ms, self.status(ms)) for b, ms in self.members.items()}
        self.n_on = sum(o.astype(float) for o in self.on.values())
        self.J = self.total()

    def pats(self, ms):
        out = {}
        for p, us, js in describe(self.rep, ms)["bindings"]:
            out[p] = out.get(p, 0) + 1
        return out

    def status(self, ms, count=None):
        count = self.count if count is None else count
        return [count.get(p, 0) >= 2 for p, _, _ in describe(self.rep, ms)["bindings"]]

    def _add_patterns(self, b, sign):
        on = self.on[b].astype(float)
        for p, nb in self.pats(self.members[b]).items():
            self.count[p] = self.count.get(p, 0) + sign * nb
            us = self.users.setdefault(p, set())
            (us.add if sign > 0 else us.discard)(b)
            self.bind_on[p] = self.bind_on.get(p, 0.0) + sign * nb * on
            self.users_on[p] = self.users_on.get(p, 0.0) + sign * on

    def pattern_terms(self, p, count, bind_on, users_on):
        if count < 2:
            return 0.0
        return (0.0 if COST == "concepts" else math.log2(count)) * bind_on + body_bits(p, self.B, self.rep.unit) * (users_on > 0)

    def total(self):
        """J counted from scratch (the check on the aggregates)."""
        nB = len(self.members)
        tok = sum(self.on[b] * (self.cost[b][0] + index_bits(nB)) for b in self.members)
        defs = library_bits(self.rep) + sum(self.cost[b][1] for b in self.members)
        for p, c in self.count.items():
            if c >= 2:
                tok = tok + self.pattern_terms(p, c, self.bind_on[p], self.users_on[p])
                defs += 0.0 if COST == "concepts" else body_bits(p, self.B, self.rep.unit)
        return float(np.mean(tok)) + defs / self.N

    def delta(self, c, parts, ons):
        """J after replacing blocks c by parts (with guards' tokens ons), minus J now."""
        count = dict(self.count)
        touched = {}
        for b in c:
            for p, nb in self.pats(self.members[b]).items():
                count[p] -= nb
                touched.setdefault(p, [0.0, 0.0])
                touched[p][0] = touched[p][0] - nb * self.on[b]
                touched[p][1] = touched[p][1] - self.on[b]
        for ms, on in zip(parts, ons):
            for p, nb in self.pats(ms).items():
                count[p] = count.get(p, 0) + nb
                touched.setdefault(p, [0.0, 0.0])
                touched[p][0] = touched[p][0] + nb * on
                touched[p][1] = touched[p][1] + on
        flips = {p for p in touched if (count.get(p, 0) >= 2) != (self.count.get(p, 0) >= 2)}
        affected = set().union(*[self.users.get(p, set()) for p in flips]) - set(c) if flips else set()
        nB, nB2 = len(self.members), len(self.members) - len(c) + len(parts)
        lg, lg2 = index_bits(nB), index_bits(nB2)
        n_on2 = self.n_on - sum(self.on[b] for b in c) + sum(ons)
        tok = lg2 * n_on2 - lg * self.n_on
        dfs = 0.0
        for b in c:
            tok = tok - self.on[b] * self.cost[b][0]
            dfs -= self.cost[b][1]
        for ms, on in zip(parts, ons):
            cb = block_cost(self.rep, ms, self.status(ms, count))
            tok = tok + on * cb[0]
            dfs += cb[1]
        for b in affected:
            cb = block_cost(self.rep, self.members[b], self.status(self.members[b], count))
            tok = tok + self.on[b] * (cb[0] - self.cost[b][0])
            dfs += cb[1] - self.cost[b][1]
        for p, (dbo, duo) in touched.items():
            old = self.pattern_terms(p, self.count.get(p, 0), self.bind_on.get(p, 0.0), self.users_on.get(p, 0.0))
            new = self.pattern_terms(p, count.get(p, 0), self.bind_on.get(p, 0.0) + dbo, self.users_on.get(p, 0.0) + duo)
            tok = tok + (new - old)
            flip = (count.get(p, 0) >= 2) - (self.count.get(p, 0) >= 2)
            if flip:
                dfs += (0.0 if COST == "concepts" else body_bits(p, self.B, self.rep.unit)) * flip
        return float(np.mean(tok)) + dfs / self.N

    def apply(self, c, parts, ons):
        dJ = self.delta(c, parts, ons)
        flips_before = dict(self.count)
        for b in c:
            self._add_patterns(b, -1)
            self.n_on = self.n_on - self.on[b]
            del self.members[b], self.on[b], self.cost[b]
        new = []
        for ms, on in zip(parts, ons):
            b = self.fresh
            self.fresh += 1
            self.members[b], self.on[b] = list(ms), on
            self._add_patterns(b, +1)
            self.n_on = self.n_on + on
            new.append(b)
        changed = {p for p in set(flips_before) | set(self.count) if (flips_before.get(p, 0) >= 2) != (self.count.get(p, 0) >= 2)}
        redo = set(new) | (set().union(*[self.users.get(p, set()) for p in changed]) if changed else set())
        for b in redo:
            self.cost[b] = block_cost(self.rep, self.members[b], self.status(self.members[b]))
        self.J += dJ


# ------------------------------------------------------------------------------------------------- moves


def exact_if_gated(rep, run, ms, on):
    """Gating the block by `on` keeps P = M: on its off tokens every item whose effect leaves the block (a read edge
    into a neuron the block does not write for, a write, a head edge) contributes nothing."""
    off = ~on
    if not off.any():
        return True
    ms = np.asarray(ms)
    writers = set(zip(rep.layer[ms][rep.kind[ms]].tolist(), rep.recv[ms][rep.kind[ms]].tolist()))
    keep = np.array([j for j in ms if rep.kind[j] or rep.layer[j] == rep.toy.L or (int(rep.layer[j]), int(rep.recv[j])) not in writers], dtype=np.int64)
    if not len(keep):
        return True
    # per item the largest |contribution| over the off tokens: |weight| times its sender's largest |value| there
    worst = 0.0
    for l in set(rep.layer[keep].tolist()):
        for wr in (False, True):
            jj = keep[(rep.layer[keep] == l) & (rep.kind[keep] == wr)]
            if not len(jj):
                continue
            src = rep.recv[jj] if wr else rep.atom[jj]
            u, inv = np.unique(src, return_inverse=True)
            vals = (run.a[l] if wr else run.s[l])[off][:, u]
            worst = max(worst, float((np.abs(vals).max(0)[inv] * np.abs(rep.w[jj])).max()))
    return worst <= 1e-9 * rep.scale


def candidates(rep, st):
    """Merges, from short lists (no pairs over a whole bucket): a block with the blocks its read edges feed, or
    that feed its neurons, or that read the atoms its neurons write (at most GROUP of them); all blocks reading one
    atom; all blocks firing on the same tokens, and each with the next one."""
    where = {j: b for b, ms in st.members.items() for j in ms}
    writer_block, reader_blocks, feed_blocks = {}, {}, {}
    for j, b in where.items():
        l, i, k = int(rep.layer[j]), int(rep.recv[j]), int(rep.atom[j])
        if rep.kind[j]:
            writer_block[(l, i)] = b
        else:
            if k != rep.const:
                reader_blocks.setdefault(k, set()).add(b)
            if l < rep.toy.L:
                feed_blocks.setdefault((l, i), set()).add(b)
    out = set()
    for b, ms in st.members.items():
        ms = np.asarray(ms)
        reads = ms[~rep.kind[ms] & (rep.layer[ms] < rep.toy.L)]
        fed = {writer_block[(int(rep.layer[j]), int(rep.recv[j]))] for j in reads if (int(rep.layer[j]), int(rep.recv[j])) in writer_block}
        neurons = set(zip(rep.layer[ms][rep.kind[ms]].tolist(), rep.recv[ms][rep.kind[ms]].tolist()))
        feed = set().union(*[feed_blocks.get(u, set()) for u in neurons]) if neurons else set()
        fan = set().union(*[reader_blocks.get(int(k), set()) for k in rep.atom[ms][rep.kind[ms]]]) if neurons else set()
        for grp in (fed, feed, fan, feed | fan):
            if grp - {b} and len(grp | {b}) <= GROUP + 1:   # a closure is local: at most a group of blocks
                out.add(frozenset(grp | {b}))
    for bs in reader_blocks.values():
        if len(bs) > 1:
            out.add(frozenset(bs))
            s_ = sorted(bs)
            out.update(frozenset(x) for x in zip(s_, s_[1:]))
    by_on = {}
    for b, on in st.on.items():
        by_on.setdefault(on.tobytes(), []).append(b)
    for bs in by_on.values():
        if len(bs) > 1:
            out.add(frozenset(bs))
            s_ = sorted(bs)
            out.update(frozenset(x) for x in zip(s_, s_[1:]))
    return [(c, [[j for b in c for j in st.members[b]]]) for c in out if len(c) > 1]


def splits(rep, st):
    """A block's bindings grouped by their own guards' tokens (the rest one more block), and each group alone."""
    out = []
    for b, ms in st.members.items():
        d = describe(rep, ms)
        if len(d["bindings"]) < 2:
            continue
        by_on = {}
        for p, us, js in d["bindings"]:
            by_on.setdefault(gate_of(rep, st.run, js).tobytes(), []).extend(js)
        parts = list(by_on.values())
        if len(parts) < 2:
            continue
        rest = d["raw"]
        out.append((frozenset([b]), parts + ([rest] if rest else [])))
        for p_ in parts:
            ps = set(p_)
            out.append((frozenset([b]), [p_, [j for j in ms if j not in ps]]))
    return out


def transfers(rep, st):
    """A block's read edges into neurons another block writes for, or reading atoms another block's neurons write,
    moved into that block, where the target's guard is on wherever the moved edges write (the move can stay exact;
    a bit test, before any scoring); per block only the TRANSFERS targets nearest by their guards' tokens."""
    writer_block, atom_block = {}, {}
    for b, ms in st.members.items():
        for j in ms:
            if rep.kind[j]:
                writer_block[(int(rep.layer[j]), int(rep.recv[j]))] = b
                atom_block[int(rep.atom[j])] = b
    out = []
    for b, ms in st.members.items():
        to = {}
        for j in ms:
            if rep.kind[j]:
                continue
            x = writer_block.get((int(rep.layer[j]), int(rep.recv[j]))) if rep.layer[j] < rep.toy.L else None
            if x is None or x == b:
                x = atom_block.get(int(rep.atom[j]))
            if x is not None and x != b:
                to.setdefault(x, []).append(j)
        near = []
        for x, js in to.items():
            live = np.zeros(st.run.T, bool)
            for k in set(rep.atom[js].tolist()):
                live |= st.run.live(int(rep.layer[js[0]]), k)
            if (live & ~st.on[x]).any():
                continue
            near.append((float((live & st.on[x]).sum() / max((live | st.on[x]).sum(), 1)), x, js))
        # the short list: the TRANSFERS targets whose guards' tokens are nearest the moved edges' (Jaccard)
        for _, x, js in sorted(near, key=lambda t: -t[0])[:TRANSFERS]:
            moved = set(js)
            rest = [j for j in ms if j not in moved]
            out.append((frozenset([b, x]), ([rest] if rest else []) + [st.members[x] + js]))
    return out


TRANSFERS = 2


def block_signature(rep, ms):
    """A block's shape without its atoms' and receivers' identities: its bindings' patterns and its raw items'
    kinds, layers and quantized weights (hashable)."""
    d = describe(rep, ms)
    q = lambda x: int(round(x / rep.scale * 2.0 ** rep.B))  # noqa: E731
    raw = tuple(sorted((bool(rep.kind[j]), int(rep.layer[j]), q(rep.w[j])) for j in d["raw"]))
    return (tuple(sorted(hash(p) for p, _, _ in d["bindings"])), raw)


def lifted(rep, st, moves):
    """Moves of one shape made at once: moves whose blocks have the same signatures (and parts the same sizes) are
    joined, on disjoint blocks, into one move (a rule's bindings change together, keeping their rule). Grouping is
    by hashing: no pairs."""
    sig_cache = {}

    def sig(b):
        if b not in sig_cache:
            sig_cache[b] = block_signature(rep, st.members[b])
        return sig_cache[b]
    groups = {}
    for c, parts in moves:
        key = (tuple(sorted(hash(sig(b)) for b in c)), tuple(sorted(len(p) for p in parts)))
        groups.setdefault(key, []).append((c, parts))
    out = []
    for ms_ in groups.values():
        if len(ms_) < 2:
            continue
        used, cs, ps = set(), set(), []
        for c, parts in ms_:
            if used & set(c):
                continue
            used |= set(c)
            cs |= set(c)
            ps += parts
        if len(ps) > len(ms_[0][1]):
            out.append((frozenset(cs), ps))
    return out


def search(rep, run, assign, N, log=print, max_rounds=200):
    """Local search on J (module note). Returns the state and per-round wall times."""
    members = {}
    for j, b in enumerate(assign):
        members.setdefault(int(b), []).append(j)
    st = State(rep, run, members, N)
    log(f"  start: {len(st.members)} blocks, J {st.J:.1f}")
    bad, times = set(), []
    for rnd in range(max_rounds):
        t0 = time.time()
        moves = candidates(rep, st) + splits(rep, st) + transfers(rep, st)
        moves += lifted(rep, st, moves)
        scored = []
        for c, parts in moves:
            key = (tuple(sorted(c)), tuple(sorted(len(p) for p in parts)))
            if key in bad:
                continue
            if len(parts) == 1:
                # a merge's guard reads its gate stage's items: where some merged blocks' guards read the same stage,
                # their union is the merged guard's tokens (to cancellation); a merge that loses even so is not scored
                stg = gate_stage(rep, parts[0])
                same = [b for b in c if gate_stage(rep, st.members[b]) == stg]
                if same and st.delta(c, parts, [np.logical_or.reduce([st.on[b] for b in same])]) >= -1e-9:
                    continue
            ons = [gate_of(rep, run, ms) for ms in parts]
            if not all(exact_if_gated(rep, run, ms, on) for ms, on in zip(parts, ons)):
                bad.add(key)
                continue
            dJ = st.delta(c, parts, ons)
            if dJ < -1e-9:
                scored.append((dJ, c, parts, ons))
        applied = 0
        used = set()
        for dJ, c, parts, ons in sorted(scored, key=lambda x: x[0]):
            if used & set(c) or (applied and st.delta(c, parts, ons) >= -1e-9):
                continue
            st.apply(c, parts, ons)
            used |= set(c)
            applied += 1
        times.append({"round": rnd, "moves": len(moves), "applied": applied, "seconds": time.time() - t0})
        if not applied:
            break
        log(f"  round {rnd}: {len(moves)} moves scored, {applied} applied, now {len(st.members)} blocks, J {st.J:.1f}, {times[-1]['seconds']:.2f} s")
    J_full = st.total()
    if abs(J_full - st.J) > 1e-6 * max(1.0, abs(J_full)):
        log(f"  aggregates drifted: J {st.J:.4f} against a full count {J_full:.4f}")
    st.J = J_full
    log(f"  end: {len(st.members)} blocks, J {st.J:.1f}")
    return st, times


# ------------------------------------------------------------------------------------------ explanations


def pieces(rep):
    """The start: each atom's reads per map; each neuron with its bias (the constant atom's read into it) and its
    writes."""
    keys = []
    for j in range(len(rep.items)):
        l, i, k = int(rep.layer[j]), int(rep.recv[j]), int(rep.atom[j])
        if rep.kind[j] or (k == rep.const and l < rep.toy.L):
            keys.append(("n", l, i))
        else:
            keys.append(("a", l, k))
    ids = {key: x for x, key in enumerate(dict.fromkeys(keys))}
    return np.array([ids[key] for key in keys])


def neurons(rep):
    """Rank-one neuron pieces: each neuron's read edges in, its writes and the reads of the atoms only it writes."""
    writer = {}
    for j in np.nonzero(rep.kind)[0]:
        writer.setdefault(int(rep.atom[j]), set()).add((int(rep.layer[j]), int(rep.recv[j])))
    keys = []
    for j in range(len(rep.items)):
        l, i, k = int(rep.layer[j]), int(rep.recv[j]), int(rep.atom[j])
        if rep.kind[j] or l < rep.toy.L:
            keys.append(("n", l, i))
        elif k in writer and len(writer[k]) == 1:
            keys.append(("n",) + min(writer[k]))
        else:
            keys.append(("a", l, k))
    ids = {key: x for x, key in enumerate(dict.fromkeys(keys))}
    return np.array([ids[key] for key in keys])


def truth(rep):
    """The true blocks. gated_copy: per mechanism its neurons' reads in, their writes and the head's reads of the
    atoms they write. resid_mlp: per function every read of its input atom (both layers' c_fc and the head); per
    neuron its writes and the reads of its atom. Items no mechanism names stay pieces."""
    toy = rep.toy
    assign = pieces(rep) + 10_000
    if toy.kind == "gated_copy":
        for mi, m in enumerate(toy.truth["mechanisms"]):
            e = m["operators"]["blocks.0.mlp.down_proj"]
            dn = np.fromfile(toy.dir / e["file"], dtype="<f8").reshape(e["shape"])
            mine = set(np.nonzero(np.abs(dn).sum(0) > 0)[0].tolist())
            atoms = {int(rep.atom[j]) for j in np.nonzero(rep.kind)[0] if rep.recv[j] in mine}
            for j in range(len(rep.items)):
                if (rep.layer[j] == 0 and rep.recv[j] in mine) or (not rep.kind[j] and rep.layer[j] == toy.L and rep.atom[j] in atoms):
                    assign[j] = mi
    elif toy.kind == "resid_mlp":
        E = toy.head * toy.rec["config"]["head"]["task_residual"]
        A = rep.D.A / np.linalg.norm(rep.D.A, axis=0)
        cos = np.abs((E / np.linalg.norm(E, axis=1, keepdims=True)) @ A)
        match = cos.argmax(1)
        rep.atom_cos = cos.max(1)
        fn = {int(k): f for f, k in enumerate(match)}
        writer = {int(rep.atom[j]): (int(rep.layer[j]), int(rep.recv[j])) for j in np.nonzero(rep.kind)[0]}
        for j in range(len(rep.items)):
            k = int(rep.atom[j])
            if not rep.kind[j] and k in fn:
                assign[j] = fn[k]
            elif rep.kind[j]:
                assign[j] = 30_000 + 1000 * int(rep.layer[j]) + int(rep.recv[j])
            elif k in writer:
                assign[j] = 30_000 + 1000 * writer[k][0] + writer[k][1]
    return assign


# ------------------------------------------------------------------------------------------------ execution


def execute(rep, tokens, members, mode="hard", chunk=512):
    """P on `tokens` (Program), every block gated by its guard on P's own run (mode 'hard') or every block on."""
    prog = Program(rep, members)
    return np.vstack([prog.run(rep.S0[tokens[i:i + chunk]].toarray(), mode) for i in range(0, len(tokens), chunk)])


def errors(rep, tokens, members):
    """P's error against M in bits per token (Gaussian head, outputs in units of the task residual), gated and with
    every block on, and the seconds of P's and M's runs."""
    y_m = rep.toy.run(rep.toy.wte[tokens])
    t0 = time.time()
    y_m = rep.toy.run(rep.toy.wte[tokens])
    tm = time.time() - t0
    t0 = time.time()
    y_h = execute(rep, tokens, members, "hard")
    tp = time.time() - t0
    y_a = execute(rep, tokens, members, "all")
    kl = lambda y: float(((y - y_m) ** 2).sum(1).mean() / (2 * LN2))  # noqa: E731
    return {"hard": kl(y_h), "all_on": kl(y_a), "P_seconds": tp, "M_seconds": tm}


def summary(rep, run, name, assign, N, true_assign=None):
    members = {}
    for j, b in enumerate(assign):
        members.setdefault(int(b), []).append(j)
    st = State(rep, run, members, N)
    rules = {p: c for p, c in st.count.items() if c >= 2}
    tok = float(st.J - (library_bits(rep) + sum(st.cost[b][1] for b in st.members) + (0.0 if COST == "concepts" else sum(body_bits(p, rep.B, rep.unit) for p in rules))) / N)
    rec = {"explanation": name, "cost": COST, "blocks": len(st.members), "rules": len(rules), "bindings_in_rules": int(sum(rules.values())),
           "bits_per_token": tok, "definitions_bits": float((st.J - tok) * N), "J": st.J,
           "blocks_on_per_token": float(st.n_on.mean()), "largest_blocks_items": sorted((len(ms) for ms in st.members.values()), reverse=True)[:5]}
    rec.update(total_concepts(rep, st))
    if true_assign is not None:
        rec["recovery"] = recovery(rep, run, st, true_assign)
    return rec, st


def recovery(rep, run, st, true_assign):
    where = {j: b for b, ms in st.members.items() for j in ms}
    true = {}
    for j, b in enumerate(true_assign):
        true.setdefault(int(b), []).append(j)
    rows = []
    for tb, ms in true.items():
        if len(ms) < 2 or 10_000 <= tb < 20_000:
            continue
        hits = {}
        for j in ms:
            hits[where[j]] = hits.get(where[j], 0) + 1
        fb = max(hits, key=hits.get)
        on_t, on_f = gate_of(rep, run, ms), st.on[fb]
        rows.append({"true": int(tb), "items": len(ms), "found": int(fb), "found_items": len(st.members[fb]),
                     "recall": hits[fb] / len(ms), "precision": hits[fb] / len(st.members[fb]),
                     "gate_jaccard": float((on_t & on_f).sum() / max((on_t | on_f).sum(), 1)),
                     "layers": sorted(set(rep.layer[st.members[fb]].tolist()))})
    return rows


# --------------------------------------------------------------------------------------------- gate training


SCALE = 1.0   # the guards' noise in training, in e-folds of the own write


def train_gates(rep, run, members, N, steps=600, seed=0, log=print):
    """The guards' thresholds trained as the design's gates are (torch, CPU): each block on with probability Phi(z),
    z = (ln own write - ln tau) / SCALE (a block writing nothing is off exactly; the own write from M's run), drawn each
    pass with Phi's gradient straight through (a closed gate is still drawn on, so it keeps a gradient); loss = the
    hard program's data term (bits per token) + lambda E[bits per token], lambda fixed at the balance rate (the median
    over thresholds of |d data / d tau| / |d E / d tau| at the start), and after every step the least common shift of all
    thresholds puts the hard program's bits at most K (K the exact program's bits per token). Started all on (tau 3 SCALE e-folds below
    each block's least write), then shifted. Reports the budget's gradient on the
    thresholds at the start, the trained hard gates against the exact ones, and the seconds per step."""
    import torch
    torch.manual_seed(seed)
    toy = rep.toy
    st = State(rep, run, members, N)
    ids = sorted(st.members)
    col = {b: x for x, b in enumerate(ids)}
    lg = math.log2(max(len(ids), 2))
    K = float(st.J - (rep.D.bits() + sum(st.cost[b][1] for b in ids) + sum(body_bits(p, rep.B, rep.unit) for p, c in st.count.items() if c >= 2)) / N)
    own = np.stack([own_write(rep, run, st.members[b]) for b in ids], 1)
    exact = np.stack([st.on[b] for b in ids], 1)
    # the guard reads its own write's magnitude on a log scale: z = (ln own - theta) / S, theta = ln tau, so a block
    # that writes nothing is off exactly (z = -inf) and the noise is relative (S e-folds)
    lown = np.log(np.maximum(own, 1e-300))
    lown[own <= 0] = -np.inf
    rules = [(p, c) for p, c in st.count.items() if c >= 2]
    bits = np.array([st.cost[b][0] + lg for b in ids])
    per_bind = np.zeros((len(ids), len(rules)))
    for r, (p, c) in enumerate(rules):
        for b in st.users[p]:
            per_bind[col[b], r] = st.pats(st.members[b])[p]
    T_ = torch.tensor
    bits_t = T_(bits, dtype=torch.float32) + T_(per_bind @ np.array([math.log2(c) for p, c in rules]) if rules else np.zeros(len(ids)), dtype=torch.float32)
    users = [T_(np.nonzero(per_bind[:, r])[0]) for r in range(len(rules))]
    bodies = [body_bits(p, rep.B, rep.unit) for p, c in rules]
    prog = Program(rep, st.members)

    def tgroup(grp):
        gb, gs, M = grp
        Mt = M.T.tocoo()
        return T_(gb), T_(gs), torch.sparse_coo_tensor(np.vstack([Mt.row, Mt.col]), T_(Mt.data, dtype=torch.float32), Mt.shape)
    maps = [(tgroup(prog.read[l]), tgroup(prog.write[l]) if l < toy.L else None) for l in range(toy.L + 1)]
    s0 = T_(run.s[0], dtype=torch.float32)
    y_m = T_(run.y, dtype=torch.float32)
    lown_t, exact_t = T_(lown, dtype=torch.float32), T_(exact)
    floor = np.array([lown[np.isfinite(lown[:, x]), x].min() if np.isfinite(lown[:, x]).any() else 0.0 for x in range(len(ids))])
    theta = T_(floor - 3 * SCALE, dtype=torch.float32).requires_grad_()
    opt = torch.optim.Adam([theta], lr=0.05)
    act = torch.relu if toy.rec["config"]["mlp_act"] == "relu" else (lambda x: x)
    bias = T_(toy.bias, dtype=torch.float32)

    def apply(g, x, grp):
        gb, gs, Mt = grp
        return torch.sparse.mm(Mt, (g[:, gb] * x[:, gs]).T).T

    def forward(g):
        s_ = s0
        for l in range(toy.L + 1):
            out = apply(g, s_, maps[l][0])
            if l == toy.L:
                y = bias + out
                return torch.relu(y) if toy.relu_head else y
            s_ = s_ + apply(g, act(out), maps[l][1])

    def expected_bits(phi):
        e = (phi * bits_t).sum(1)
        for u, bb in zip(users, bodies):
            e = e + bb * (1 - torch.prod(1 - phi[:, u], 1))
        return e.mean()

    def phi_of(th):
        return 0.5 * (1 + torch.erf((lown_t - th) / (SCALE * math.sqrt(2))))

    def project():
        # one common shift of every threshold: the least for which the hard program's bits per token are at most K
        # (bisection; the hard bits fall as the shift grows). The soft expectation is not pinned: tokens whose own
        # write is exactly zero keep Phi(-tau) of it, which would push every threshold past the true writes.
        with torch.no_grad():
            if expected_bits((lown_t - theta + 30.0 > 0).float()).item() <= K:
                return                              # the budget binds at no shift: none
            lo, hi = -30.0, 30.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if expected_bits((lown_t - theta - mid > 0).float()).item() > K else (lo, mid)
            theta.add_(hi)

    project()
    # lambda at the balance rate: the median over thresholds of |d data/d theta| / |d E/d theta| at the projected start
    phi = phi_of(theta)
    g = torch.bernoulli(phi.detach()) + phi - phi.detach()
    data, e = ((forward(g) - y_m) ** 2).sum(1).mean() / (2 * LN2), expected_bits(phi)
    gd = torch.autograd.grad(data, theta, retain_graph=True)[0].abs()
    ge = torch.autograd.grad(e, theta)[0].abs()
    grad0 = ge.mean().item()
    ok = (gd > 0) & (ge > 0)
    lam = float((gd[ok] / ge[ok]).median()) if ok.any() else 1e-3
    hist, t0 = [], time.time()
    for step in range(steps + 1):
        phi = phi_of(theta)
        g = torch.bernoulli(phi.detach()) + phi - phi.detach()
        data = ((forward(g) - y_m) ** 2).sum(1).mean() / (2 * LN2)
        e = expected_bits(phi)
        opt.zero_grad()
        (data + lam * e).backward()
        opt.step()
        project()
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                hard = lown_t - theta > 0
                kl = ((forward(hard.float()) - y_m) ** 2).sum(1).mean().item() / (2 * LN2)
                jac = float(((hard & exact_t).sum() / (hard | exact_t).sum().clamp_min(1)).item())
                eh = expected_bits(hard.float()).item()
                es = expected_bits(phi_of(theta)).item()
            hist.append({"step": step, "E_bits_soft": round(es, 1), "E_bits_hard": round(eh, 1), "K": round(K, 1),
                         "hard_kl_bits": kl, "jaccard_with_exact_gates": round(jac, 4), "lambda": lam})
            log(f"  gates step {step}: E[bits] {es:.1f} (hard {eh:.1f}, K {K:.1f}), hard KL {kl:.3g}, gates vs exact Jaccard {jac:.4f}, lambda {lam:.3g}")
    return {"budget_grad_on_thresholds_at_start": grad0, "seconds_per_step": (time.time() - t0) / (steps + 1), "trace": hist}


# ------------------------------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("toy")
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--out")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--gates", type=int, default=0, help="steps of gate training on the found blocks (0: none)")
    ap.add_argument("--cost", choices=("bits", "concepts"), default="bits", help="the per-token count the search minimizes")
    a = ap.parse_args()
    global COST
    COST = a.cost
    toy = Toy(Path(a.toy))
    rep = Rep(toy, a.bits)
    D = rep.D
    print(f"{toy.name}: {rep.C} atoms ({sum(1 for k in D.atoms if k[0] == 'r')} recurring, {sum(1 for k in D.atoms if k[0] == 'x')} axes, "
          f"{D.own_count} own), {sum(rep.m)} neurons, {len(rep.items)} items, constant atom {rep.const}; dictionary and split {rep.seconds:.1f} s", flush=True)
    train = np.arange(HELD, toy.wte.shape[0])
    run = Run(rep, train[: a.tokens])
    held = np.arange(min(HELD, 1024))
    N = len(train)
    t_assign = truth(rep)
    named = [("truth", t_assign), ("neuron pieces", neurons(rep)), ("pieces (start)", pieces(rep))]
    res = []
    for name, asg in named:
        rec, st = summary(rep, run, name, asg, N)
        rec["error_bits_per_token"] = errors(rep, held, st.members)
        res.append(rec)
    print("search:", flush=True)
    t0 = time.time()
    st, times = search(rep, run, pieces(rep), N)
    search_s = time.time() - t0
    found = np.zeros(len(rep.items), dtype=np.int64)
    for b, ms in st.members.items():
        found[ms] = b
    rec, st = summary(rep, run, "found", found, N, t_assign)
    rec["error_bits_per_token"] = errors(rep, held, st.members)
    rec["search_seconds"] = search_s
    rec["round_seconds"] = [round(t["seconds"], 3) for t in times]
    res.append(rec)
    for r_ in res:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r_.items() if k not in ("recovery", "round_seconds")}), flush=True)
    rows = res[-1]["recovery"]
    exact = sum(r["recall"] == 1.0 and r["precision"] == 1.0 for r in rows)
    print(f"recovery: {exact} of {len(rows)} true blocks found exactly; mean recall {np.mean([r['recall'] for r in rows]):.3f}, "
          f"precision {np.mean([r['precision'] for r in rows]):.3f}, gate Jaccard {np.mean([r['gate_jaccard'] for r in rows]):.3f}")
    for r in rows:
        if r["recall"] < 1.0 or r["precision"] < 1.0:
            print("  not exact:", r)
            break
    if getattr(rep, "atom_cos", None) is not None:
        print(f"input atoms against the embedding directions: |cos| mean {rep.atom_cos.mean():.6f}, min {rep.atom_cos.min():.6f}")
    scale = {"toy": toy.name, "neurons": sum(rep.m), "atoms": rep.C, "items": len(rep.items), "dictionary_split_seconds": rep.seconds,
             "library_bits": res[-1]["definitions_bits"], "bits_per_token": res[-1]["bits_per_token"], "search_seconds": search_s,
             "search_rounds": len(times), "mean_round_seconds": float(np.mean([t["seconds"] for t in times])),
             "P_over_M_seconds": res[-1]["error_bits_per_token"]["P_seconds"] / max(res[-1]["error_bits_per_token"]["M_seconds"], 1e-9)}
    if a.gates:
        print("gate training on the found blocks:", flush=True)
        gt = train_gates(rep, Run(rep, train[:1024]), st.members, N, a.gates)
        print(f"  the budget's gradient on the thresholds at the start: mean |dE/dtheta| {gt['budget_grad_on_thresholds_at_start']:.4g}; {gt['seconds_per_step']:.3f} s per step")
        scale["gate_step_seconds"] = gt["seconds_per_step"]
        res.append({"gate_training": gt})
    print("scale:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in scale.items()}))
    res.append({"scale": scale})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
