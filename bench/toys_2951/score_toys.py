"""Score a decomposition of a ground-truth toy against its known mechanisms (#2951 toy gate). A
regression gate for the fitting method, never a result.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/score_toys.py TOY_DIR PARTS_DIR [PARTS_DIR...]
       ~/mpd-data/venv/bin/python bench/toys_2951/score_toys.py TOY_DIR --references OUT_ROOT

TOY_DIR is a toy that train_toys.py wrote. A PARTS_DIR holds parts.json and its float64 blobs:

  {"parts": [{"name": str,
              "slices": {OPERATOR: {"U": FILE, "V": FILE, "rank": r}},   W_b on OPERATOR = U V^T
              "gate": {"kind": "own" | "direction" | "always",
                       "read": OPERATOR,      the operator whose input the gate reads
                       "tau": float,          own: z = ||V_read^T x|| - tau; direction: z = g^T x - tau
                       "g": FILE}}],          direction arm only
   "active": FILE,                optional: rows x parts hard gates on the held-out rows (the
                                  fitter's own; required for the transformer toys)
   "description_bits": float,     optional: the fitter's description length of P
   "kept": [OPERATOR],            optional: operators P runs as M's own (not compared)
   "fitter": str}

Every part is on at a row iff its gate z > 0 (H(0) = 0), read in P's own run. Operators no part
spans are absent from P (the toys' biases and fixed embeddings are not decomposed and stay).

Reports per PARTS_DIR, as JSON on stdout and in PARTS_DIR/SCORE_<toy>.json:
 (a) recovery: per known mechanism the fitted part of largest cosine (the parts' weights on every
     operator, concatenated), its cosine, and whether its rank on each operator equals the
     mechanism's; distinct parts matched; parts matching no mechanism.
 (b) active parts per row against the mechanisms' (truth_active), and per matched mechanism the
     agreement of its part's gate with its activity (precision, recall).
 (c) held-out verbatim edits (real-input toys): the same operation at the same site applied to M
     and to P on each model's own values, divergence D(M_e || P_e) in bits per row (Gaussian
     predictive distributions of M's own output variance s^2: ||y_M - y_P||^2 / (2 s^2 ln 2)),
     per family (clean, zero, scale, push, swap), with the edit's effect D(M_e || M) beside it.
 (d) description: P's parameters and bits (the fitter's description_bits when given, else 32 bits
     per parameter) against M's parameters at 32 bits.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)


def load(dir_: Path, entry) -> np.ndarray:
    shape = entry["shape"]
    return np.fromfile(dir_ / entry["file"], dtype="<f8").reshape(shape)


def read_blob(dir_: Path, name: str) -> np.ndarray:
    return np.fromfile(dir_ / name, dtype="<f8")


class Toy:
    def __init__(self, dir_: Path):
        self.dir = dir_
        self.record = json.loads((dir_ / "export.json").read_text())
        self.truth = json.loads((dir_ / "truth.json").read_text())
        self.kind = self.record["kind"]
        self.weights = {name: np.fromfile(dir_ / f"{name}.f64", dtype="<f8").reshape(e["shape"]) for name, e in self.record["files"].items()}
        self.mechanisms = []
        for m in self.truth["mechanisms"]:
            ops = {op: load(dir_, e) for op, e in m["operators"].items()}
            self.mechanisms.append({**m, "deltas": ops})
        self.active = load(dir_, self.truth["active"]).astype(bool)
        if self.kind in ("tms", "resid_mlp"):
            self.operators = self.record["operators"]
            self.inputs = self.weights["inputs"]
        else:
            self.operators = [n for n in self.record["files"] if n != "tokens" and not n.endswith("gain")]
        self.m_params = sum(self.weights[op].size for op in self.operators)

    # -------------------------------------------------------------------- real-input forwards

    def sites(self):
        if self.kind == "tms":
            return ["hidden"] + (["hidden2"] if self.record["config"]["identity"] else []) + ["pre"]
        L = self.record["config"]["n_layers"]
        return [s for l in range(L) for s in (f"resid.{l}", f"neurons.{l}", f"mlp_out.{l}")] + [f"resid.{L}"]

    def run(self, weight, x, edit=None):
        """The toy's forward with operator weights `weight(op, read_input)` (rows x out), applying
        `edit(site, value)` at each site; returns the outputs and every site's value."""
        values = {}

        def site(name, v):
            if edit is not None:
                v = edit(name, v)
            values[name] = v
            return v

        if self.kind == "tms":
            h = site("hidden", weight("linear1", x))
            if self.record["config"]["identity"]:
                h = site("hidden2", weight("hidden_layers.0", h))
            pre = site("pre", weight("linear2", h) + self.weights["linear2.bias"])
            return np.maximum(pre, 0.0), values
        E = self.weights["W_E"]
        r = x @ E
        L = self.record["config"]["n_layers"]
        for l in range(L):
            r = site(f"resid.{l}", r)
            n = site(f"neurons.{l}", np.maximum(weight(f"blocks.{l}.mlp.W_in", r), 0.0))
            r = r + site(f"mlp_out.{l}", weight(f"blocks.{l}.mlp.W_out", n))
        r = site(f"resid.{L}", r)
        return r @ E.T, values


class Parts:
    def __init__(self, dir_: Path, toy: Toy):
        self.dir = dir_
        self.record = json.loads((dir_ / "parts.json").read_text())
        self.parts = []
        for p in self.record["parts"]:
            slices = {}
            for op, s in p["slices"].items():
                rows, cols = toy.weights[op].shape
                U = read_blob(dir_, s["U"]).reshape(rows, -1)
                V = read_blob(dir_, s["V"]).reshape(cols, -1)
                slices[op] = (U, V)
            gate = dict(p["gate"])
            if gate["kind"] == "direction":
                gate["g_vec"] = read_blob(dir_, gate["g"])
            self.parts.append({"name": p["name"], "slices": slices, "gate": gate})
        self.active = None
        if "active" in self.record:
            self.active = read_blob(dir_, self.record["active"]).reshape(-1, len(self.parts)).astype(bool)
        self.params = sum(U.size + V.size for p in self.parts for (U, V) in p["slices"].values()) + sum(
            1 + (p["gate"]["g_vec"].size if p["gate"]["kind"] == "direction" else 0) for p in self.parts if p["gate"]["kind"] != "always")

    def weight(self, part, op):
        U, V = part["slices"][op]
        return U @ V.T

    def gate(self, part, x_read):
        g = part["gate"]
        if g["kind"] == "always":
            return np.ones(x_read.shape[0], dtype=bool)
        if g["kind"] == "own":
            _, V = part["slices"][g["read"]]
            z = np.linalg.norm(x_read @ V, axis=1) - g["tau"]
        else:
            z = x_read @ g["g_vec"] - g["tau"]
        return z > 0

    def forward(self, toy: Toy, x, edit=None):
        """P's run: each part's gate read at its read operator's input in P's own run."""
        on = {}

        def weight(op, inp):
            out = 0.0
            for b, part in enumerate(self.parts):
                if b not in on and part["gate"]["read"] == op:
                    on[b] = self.gate(part, inp)
                if op in part["slices"]:
                    gate = on.get(b)
                    if gate is None:
                        raise ValueError(f"part {part['name']} runs on {op} before its gate reads {part['gate']['read']}")
                    out = out + (inp @ self.weight(part, op).T) * gate[:, None]
            if np.isscalar(out):
                out = np.zeros((inp.shape[0], toy.weights[op].shape[0]))
            return out

        y, values = toy.run(weight, x, edit)
        gates = np.stack([on.get(b, np.zeros(x.shape[0], dtype=bool)) for b in range(len(self.parts))], 1)
        return y, values, gates


def cosine(a: dict, b: dict) -> float:
    ops = set(a) | set(b)
    dot = sum(float(np.sum(a[o] * b[o])) for o in ops if o in a and o in b)
    na = math.sqrt(sum(float(np.sum(v * v)) for v in a.values()))
    nb = math.sqrt(sum(float(np.sum(v * v)) for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def numerical_rank(m: np.ndarray) -> int:
    s = np.linalg.svd(m, compute_uv=False)
    return int(np.sum(s > s[0] * max(m.shape) * np.finfo(float).eps)) if s.size and s[0] > 0 else 0


def score(toy: Toy, parts: Parts, seed: int = 0) -> dict:
    fitted = [{op: parts.weight(p, op) for op in p["slices"]} for p in parts.parts]
    # operators the explanation runs as M's own (`kept`) are not compared
    kept = set(parts.record.get("kept", []))
    deltas = [{op: d for op, d in m["deltas"].items() if op not in kept} for m in toy.mechanisms]
    # (a) recovery
    recovery, matched = [], set()
    for m, delta in zip(toy.mechanisms, deltas):
        cos = [cosine(delta, f) for f in fitted]
        b = int(np.argmax(cos)) if cos else -1
        truth_rank = {op: numerical_rank(d) for op, d in delta.items()}
        part_rank = {op: parts.parts[b]["slices"][op][0].shape[1] for op in parts.parts[b]["slices"]} if b >= 0 else {}
        recovery.append({"mechanism": m["name"], "part": parts.parts[b]["name"] if b >= 0 else None, "cosine": cos[b] if b >= 0 else 0.0,
                         "rank_right": all(part_rank.get(op) == r for op, r in truth_rank.items()) and set(part_rank) == set(truth_rank)})
        matched.add(b)
    best_for_part = [max((cosine(d, f) for d in deltas), default=0.0) for f in fitted]
    cosines = np.array([r["cosine"] for r in recovery])
    out = {
        "parts": len(parts.parts),
        "mechanisms": len(toy.mechanisms),
        "recovery": {
            "mean_cosine": float(cosines.mean()),
            "min_cosine": float(cosines.min()),
            "recovered_cos_0.9": int(np.sum(cosines >= 0.9)),
            "rank_right_of_recovered": int(sum(r["rank_right"] for r in recovery if r["cosine"] >= 0.9)),
            "distinct_parts_matched": len(matched),
            "parts_matching_no_mechanism_cos_0.5": int(sum(c < 0.5 for c in best_for_part)),
            "per_mechanism": recovery,
        },
        "description": {"P_params": int(parts.params), "M_params": int(toy.m_params),
                        "P_bits": float(parts.record.get("description_bits", 32 * parts.params)),
                        "P_bits_source": "fitter" if "description_bits" in parts.record else "32 bits per parameter",
                        "M_bits": 32 * int(toy.m_params)},
    }
    truth_active = toy.active
    if toy.kind in ("tms", "resid_mlp"):
        out |= edits(toy, parts, seed)
        gates = out.pop("_gates")
    else:
        gates = parts.active
    # (b) activity
    if gates is not None:
        act = {"P_active_per_row": float(gates.sum(1).mean()), "truth_active_per_row": float(truth_active.sum(1).mean())}
        agree = []
        for i, r in enumerate(recovery):
            if r["cosine"] < 0.9:
                continue
            b = [p["name"] for p in parts.parts].index(r["part"])
            g, t = gates[:, b], truth_active[:, i]
            agree.append((float((g & t).sum() / max(g.sum(), 1)), float((g & t).sum() / max(t.sum(), 1))))
        if agree:
            act["matched_gate_precision"] = float(np.mean([a[0] for a in agree]))
            act["matched_gate_recall"] = float(np.mean([a[1] for a in agree]))
        out["activity"] = act
    return out


def edits(toy: Toy, parts: Parts, seed: int) -> dict:
    """Held-out verbatim edits on the shared sites: per family one operation at one site per row,
    applied identically to M and P, each on its own values."""
    x = toy.inputs
    rows = x.shape[0]
    s2 = toy.record["output_variance"]

    def native(op, inp):
        return inp @ toy.weights[op].T

    y_m, sites_m = toy.run(native, x)
    y_p, sites_p, gates = parts.forward(toy, x)
    rng = np.random.default_rng(seed)
    names = toy.sites()
    site_of_row = rng.integers(0, len(names), rows)
    typical = {s: float(np.sqrt((sites_m[s] ** 2).sum(1).mean())) for s in names}
    pushes = {s: rng.standard_normal(sites_m[s].shape[1]) for s in names}
    pushes = {s: v / np.linalg.norm(v) * typical[s] for s, v in pushes.items()}
    factor = rng.uniform(0.0, 2.0, rows)
    donor = (np.arange(rows) + 1) % rows

    def div(a, b):
        return ((a - b) ** 2).sum(1) / (2 * s2 * LN2)

    result = {"clean": {"gap_bits": float(div(y_m, y_p).mean())}}
    for family in ["zero", "scale", "push", "swap"]:
        def make(own_sites):
            def edit(name, v):
                hit = (site_of_row == names.index(name))[:, None]
                if family == "zero":
                    new = np.zeros_like(v)
                elif family == "scale":
                    new = v * factor[:, None]
                elif family == "push":
                    new = v + pushes[name][None, :]
                else:
                    new = own_sites[name][donor]
                return np.where(hit, new, v)
            return edit
        y_me, _ = toy.run(native, x, make(sites_m))
        y_pe, _, _ = parts.forward(toy, x, make(sites_p))
        gap = div(y_me, y_pe)
        result[family] = {"gap_bits": float(gap.mean()), "gap_p99_bits": float(np.quantile(gap, 0.99)), "effect_bits": float(div(y_me, y_m).mean()),
                          "ignoring_bits": float(div(y_me, y_p).mean())}
    return {"edits_bits_per_row": result, "_gates": gates}


# ------------------------------------------------------------------------------ references


def write_parts(dir_: Path, parts: list, extra: dict | None = None):
    dir_.mkdir(parents=True, exist_ok=True)
    record = []
    for i, p in enumerate(parts):
        slices = {}
        for op, (U, V) in p["slices"].items():
            np.ascontiguousarray(U, dtype="<f8").tofile(dir_ / f"p{i}.{op}.U.f64")
            np.ascontiguousarray(V, dtype="<f8").tofile(dir_ / f"p{i}.{op}.V.f64")
            slices[op] = {"U": f"p{i}.{op}.U.f64", "V": f"p{i}.{op}.V.f64", "rank": int(U.shape[1])}
        gate = dict(p["gate"])
        if "g_vec" in gate:
            np.ascontiguousarray(gate.pop("g_vec"), dtype="<f8").tofile(dir_ / f"p{i}.g.f64")
            gate["g"] = f"p{i}.g.f64"
        record.append({"name": p["name"], "slices": slices, "gate": gate})
    (dir_ / "parts.json").write_text(json.dumps({"parts": record} | (extra or {}), indent=1))


def low_rank(m: np.ndarray):
    """m as U V^T at its numerical rank (an exact factorization)."""
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    r = numerical_rank(m)
    return u[:, :r] * s[:r], vt[:r].T


def references(toy: Toy, root: Path):
    """Two reference decompositions that validate the harness: the native start (each operator of
    M one always-on part, exact) and the truth (the known mechanisms as parts, gated by their own
    read; the real-input toys only, whose gates are defined on the input)."""
    native = [{"name": op, "slices": {op: low_rank(toy.weights[op])}, "gate": {"kind": "always", "read": op}} for op in toy.operators]
    extra = {"fitter": "native start: each operator one always-on part"}
    if toy.kind not in ("tms", "resid_mlp"):
        extra["active"] = "active.f64"
    write_parts(root / "native", native, extra)
    if toy.kind not in ("tms", "resid_mlp"):
        rows = toy.active.shape[0]
        np.ones((rows, len(native))).astype("<f8").tofile(root / "native" / "active.f64")
        truth = []
        for i, m in enumerate(toy.mechanisms):
            truth.append({"name": m["name"], "slices": {op: low_rank(d) for op, d in m["deltas"].items()}, "gate": {"kind": "always", "read": next(iter(m["deltas"]))}})
        write_parts(root / "truth", truth, {"fitter": "truth (weights only; gates from truth_active)", "active": "active.f64"})
        toy.active.astype("<f8").tofile(root / "truth" / "active.f64")
        return
    truth = []
    for i, m in enumerate(toy.mechanisms):
        slices = {op: low_rank(d) for op, d in m["deltas"].items()}
        first = [op for op in toy.operators if op in slices][0]
        if m["gate"] == "always":
            gate = {"kind": "always", "read": first}
        else:
            # the gate reads the mechanism's own feature: V's single column on the first operator
            # (e_i for TMS, the embedding's dual frame vector for compressed computation); tau is
            # above that column's factorization rounding (~1e-16 on the other features)
            gate = {"kind": "own", "read": first, "tau": 1e-9}
        truth.append({"name": m["name"], "slices": slices, "gate": gate})
    write_parts(root / "truth", truth, {"fitter": "truth: the known mechanisms as parts, gated by their feature"})


if __name__ == "__main__":
    toy = Toy(Path(sys.argv[1]))
    if sys.argv[2] == "--references":
        root = Path(sys.argv[3])
        references(toy, root)
        dirs = [root / "native", root / "truth"]
    else:
        dirs = [Path(a) for a in sys.argv[2:]]
    table = {}
    for d in dirs:
        s = score(toy, Parts(d, toy))
        (d / f"SCORE_{toy.dir.name}.json").write_text(json.dumps(s, indent=1))
        s["recovery"].pop("per_mechanism")
        table[str(d)] = s
    print(json.dumps(table, indent=1))
