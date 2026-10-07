"""Score a decomposition of a ground-truth toy against its known mechanisms (#2951 toy gate). A
regression gate for the fitting method, never a result.

usage: ~/mpd-data/venv/bin/python bench/toys_2951/score_toys.py TOY_DIR PARTS_DIR [PARTS_DIR...]
       ~/mpd-data/venv/bin/python bench/toys_2951/score_toys.py TOY_DIR --references OUT_ROOT

TOY_DIR is a toy that train_toys.py wrote (every toy is a language model of the engine). A
PARTS_DIR holds parts.json and its float64 blobs, as the fitter's parts dump writes them
(mpd_library_mdl_2951 ... parts DIR, gam_mpd::library_vpd::dump_parts):

  {"parts": [{"name": str,
              "slices": {OPERATOR: {"U": FILE, "V": FILE, "rank": r}},   W_b on OPERATOR = U V^T
              "gate": {"kind": "own" | "direction" | "always", "read": OPERATOR, "tau": float,
                       "g": FILE}}],     direction arm only
   "active": FILE,                rows x parts: each part's hard gate on every held-out token in
                                  P's own run
   "description_bits": float,     optional: the fitter's description length of P
   "kept": [OPERATOR],            optional: operators P runs as M's own (not compared)
   "fitter": str}

OPERATOR is an export name (blocks.{l}.attn.q_proj, blocks.{l}.mlp.c_fc, ...). Reports per
PARTS_DIR, as JSON on stdout and in PARTS_DIR/SCORE_<toy>.json:
 (a) recovery: per known mechanism the fitted part of largest cosine (the parts' weights on every
     operator, concatenated), its cosine, its norm on the mechanism's operators over the
     mechanism's (SPD's ML2R for a whole part), and whether its rank on each operator equals the
     mechanism's; distinct parts matched; parts matching no mechanism.
 (b) active parts per token against the mechanisms' (truth_active), and per recovered mechanism the
     agreement of its part's gate with its activity (precision, recall) and, where the toy has its
     mechanisms' measured effect per token (truth_effect), the share of that effect on the tokens
     its gate is on and the share of tokens it is on.
 (c) description: P's parameters and bits (the fitter's description_bits when given, else 32 bits
     per parameter) against M's parameters of the decomposed maps at 32 bits.
Held-out verbatim edits are the engine's (engine_edits.py), on the same explanation.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


def load(dir_: Path, entry) -> np.ndarray:
    return np.fromfile(dir_ / entry["file"], dtype="<f8").reshape(entry["shape"])


def read_blob(dir_: Path, name: str) -> np.ndarray:
    return np.fromfile(dir_ / name, dtype="<f8")


class Toy:
    def __init__(self, dir_: Path):
        self.dir = dir_
        self.record = json.loads((dir_ / "export.json").read_text())
        self.truth = json.loads((dir_ / "truth.json").read_text())
        # the decomposed maps: every nonzero block operator (the embedding, norms and head stay
        # M's; a real-valued toy's zero attention and the induction toy's zero MLP compute nothing)
        self.shapes = {name: e["shape"] for name, e in self.record["files"].items()}
        blocks = [n for n in self.shapes if n.startswith("blocks.") and not n.endswith("gain") and not n.endswith("bias")]
        self.operators = [n for n in blocks if np.any(self.weight(n))]
        self.mechanisms = [{**m, "deltas": {op: load(dir_, e) for op, e in m["operators"].items()}} for m in self.truth["mechanisms"]]
        self.active = load(dir_, self.truth["active"]).astype(bool)
        # each mechanism's measured effect per token (train_toys.py effects:TOY), when written
        self.effect = load(dir_, self.truth["effect"]) if "effect" in self.truth else None
        self.m_params = sum(int(np.prod(self.shapes[op])) for op in self.operators)

    def weight(self, op: str) -> np.ndarray:
        return np.fromfile(self.dir / f"{op}.f64", dtype="<f8").reshape(self.shapes[op])


class Parts:
    def __init__(self, dir_: Path, toy: Toy):
        self.dir = dir_
        self.record = json.loads((dir_ / "parts.json").read_text())
        self.parts = []
        for p in self.record["parts"]:
            slices = {}
            for op, s in p["slices"].items():
                rows, cols = toy.shapes[op]
                slices[op] = (read_blob(dir_, s["U"]).reshape(rows, -1), read_blob(dir_, s["V"]).reshape(cols, -1))
            self.parts.append({"name": p["name"], "slices": slices, "gate": p["gate"]})
        self.active = read_blob(dir_, self.record["active"]).reshape(-1, len(self.parts)).astype(bool)
        self.params = sum(U.size + V.size for p in self.parts for (U, V) in p["slices"].values()) + sum(
            1 + (toy.shapes[p["gate"]["read"]][1] if p["gate"]["kind"] == "direction" else 0) for p in self.parts if p["gate"]["kind"] != "always")

    @staticmethod
    def weight(part, op):
        U, V = part["slices"][op]
        return U @ V.T


def cosine(a: dict, b: dict) -> float:
    dot = sum(float(np.sum(a[o] * b[o])) for o in set(a) & set(b))
    na = math.sqrt(sum(float(np.sum(v * v)) for v in a.values()))
    nb = math.sqrt(sum(float(np.sum(v * v)) for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def numerical_rank(m: np.ndarray) -> int:
    s = np.linalg.svd(m, compute_uv=False)
    return int(np.sum(s > s[0] * max(m.shape) * np.finfo(float).eps)) if s.size and s[0] > 0 else 0


def score(toy: Toy, parts: Parts) -> dict:
    fitted = [{op: Parts.weight(p, op) for op in p["slices"]} for p in parts.parts]
    # operators the explanation runs as M's own (`kept`) are not compared
    kept = set(parts.record.get("kept", []))
    deltas = [{op: d for op, d in m["deltas"].items() if op not in kept} for m in toy.mechanisms]
    norm = lambda w: math.sqrt(sum(float(np.sum(v * v)) for v in w.values()))  # noqa: E731
    # (a) recovery
    recovery, matched = [], set()
    for m, delta in zip(toy.mechanisms, deltas):
        cos = [cosine(delta, f) for f in fitted]
        b = int(np.argmax(cos)) if cos else -1
        truth_rank = {op: numerical_rank(d) for op, d in delta.items()}
        part_rank = {op: parts.parts[b]["slices"][op][0].shape[1] for op in parts.parts[b]["slices"]} if b >= 0 else {}
        recovery.append({"mechanism": m["name"], "part": b, "cosine": cos[b] if b >= 0 else 0.0,
                         "norm_ratio": norm({op: fitted[b][op] for op in delta if op in fitted[b]}) / norm(delta) if b >= 0 else 0.0,
                         "rank_right": all(part_rank.get(op) == r for op, r in truth_rank.items())})
        matched.add(b)
    best_for_part = [max((cosine(d, f) for d in deltas), default=0.0) for f in fitted]
    # Pieces: each part goes to the mechanism its weights are most aligned with (none where it is
    # orthogonal to all); per mechanism its pieces, their slices, the cosine of their sum with the
    # mechanism (the span the pieces recover together) and whether any piece's gate is on against
    # the mechanism's activity (precision, recall).
    owner = [int(np.argmax([cosine(d, f) for d in deltas])) if best_for_part[b] > 0 else -1 for b, f in enumerate(fitted)]
    rows = min(len(parts.active), len(toy.active))
    for i, (r, delta) in enumerate(zip(recovery, deltas)):
        mine = [b for b, o in enumerate(owner) if o == i]
        total = {op: sum((fitted[b][op] for b in mine if op in fitted[b]), np.zeros_like(d)) for op, d in delta.items()}
        on, t = parts.active[:rows][:, mine].any(1), toy.active[:rows, i]
        r["pieces"] = {"parts": len(mine), "slices": int(sum(U.shape[1] for b in mine for U, _ in parts.parts[b]["slices"].values())),
                       "sum_cosine": cosine(delta, total) if mine else 0.0,
                       "gate_precision": float((on & t).sum() / max(on.sum(), 1)), "gate_recall": float((on & t).sum() / max(t.sum(), 1))}
    cosines = np.array([r["cosine"] for r in recovery])
    recovered = [r for r in recovery if r["cosine"] >= 0.9]
    out = {
        "parts": len(parts.parts),
        "mechanisms": len(toy.mechanisms),
        "recovery": {
            "mean_cosine": float(cosines.mean()),
            "min_cosine": float(cosines.min()),
            "recovered_cos_0.9": len(recovered),
            "mean_norm_ratio_of_recovered": float(np.mean([r["norm_ratio"] for r in recovered])) if recovered else float("nan"),
            "rank_right_of_recovered": int(sum(r["rank_right"] for r in recovered)),
            "distinct_parts_matched": len(matched),
            "parts_matching_no_mechanism_cos_0.5": int(sum(c < 0.5 for c in best_for_part)),
            "per_mechanism": recovery,
        },
        "description": {"P_params": int(parts.params), "M_params": int(toy.m_params),
                        "P_bits": float(parts.record.get("description_bits", 32 * parts.params)),
                        "P_bits_source": "fitter" if "description_bits" in parts.record else "32 bits per parameter",
                        "M_bits": 32 * int(toy.m_params)},
    }
    # (b) activity, on the held-out tokens the truth covers
    gates, truth = parts.active, toy.active
    rows = min(len(gates), len(truth))
    gates, truth = gates[:rows], truth[:rows]
    act = {"P_active_per_token": float(gates.sum(1).mean()), "truth_active_per_token": float(truth.sum(1).mean())}
    agree = []
    for i, r in enumerate(recovery):
        if r["cosine"] >= 0.9:
            g, t = gates[:, r["part"]], truth[:, i]
            agree.append((float((g & t).sum() / max(g.sum(), 1)), float((g & t).sum() / max(t.sum(), 1))))
    if agree:
        act["matched_gate_precision"] = float(np.mean([a[0] for a in agree]))
        act["matched_gate_recall"] = float(np.mean([a[1] for a in agree]))
    # Against the measured effect: per recovered mechanism, the share of its effect on the tokens
    # its part's gate is on, and the share of tokens it is on (the truth's on/off is not used).
    if toy.effect is not None:
        effect = toy.effect[:rows]
        covered = [(float(effect[gates[:, r["part"]], i].sum() / max(effect[:, i].sum(), 1e-300)), float(gates[:, r["part"]].mean()))
                   for i, r in enumerate(recovery) if r["cosine"] >= 0.9]
        if covered:
            act["matched_effect_covered"] = float(np.mean([c[0] for c in covered]))
            act["matched_on_rate"] = float(np.mean([c[1] for c in covered]))
    out["activity"] = act
    return out


# ------------------------------------------------------------------------------ references


def write_parts(dir_: Path, parts: list, active: np.ndarray, extra: dict | None = None):
    dir_.mkdir(parents=True, exist_ok=True)
    record = []
    for i, p in enumerate(parts):
        slices = {}
        for op, (U, V) in p["slices"].items():
            np.ascontiguousarray(U, dtype="<f8").tofile(dir_ / f"p{i}.{op}.U.f64")
            np.ascontiguousarray(V, dtype="<f8").tofile(dir_ / f"p{i}.{op}.V.f64")
            slices[op] = {"U": f"p{i}.{op}.U.f64", "V": f"p{i}.{op}.V.f64", "rank": int(U.shape[1])}
        record.append({"name": p["name"], "slices": slices, "gate": p["gate"]})
    np.ascontiguousarray(active, dtype="<f8").tofile(dir_ / "active.f64")
    (dir_ / "parts.json").write_text(json.dumps({"parts": record, "active": "active.f64"} | (extra or {}), indent=1))


def low_rank(m: np.ndarray):
    """m as U V^T at its numerical rank (an exact factorization)."""
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    r = numerical_rank(m)
    return u[:, :r] * s[:r], vt[:r].T


def references(toy: Toy, root: Path):
    """Two reference decompositions that validate the harness: the native start (each decomposed
    operator of M one always-on part, exact) and the truth (the known mechanisms as parts, active
    where the truth says)."""
    rows = toy.active.shape[0]
    native = [{"name": op, "slices": {op: low_rank(toy.weight(op))}, "gate": {"kind": "always", "read": op}} for op in toy.operators]
    # the embedding and readout are M's in every explanation the fitter makes (its dump's `kept`)
    kept = ["wte", "lm_head"]
    write_parts(root / "native", native, np.ones((rows, len(native))), {"fitter": "native start: each operator one always-on part", "kept": kept})
    truth = [{"name": m["name"], "slices": {op: low_rank(d) for op, d in m["deltas"].items()}, "gate": {"kind": "always", "read": next(iter(m["deltas"]))}}
             for m in toy.mechanisms]
    write_parts(root / "truth", truth, toy.active.astype(float), {"fitter": "truth: the known mechanisms, active where the truth says", "kept": kept})


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
