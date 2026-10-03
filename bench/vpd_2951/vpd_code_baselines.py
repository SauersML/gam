"""Does the masked driver's two-part code have a trivial minimum on the 4L Pile target (#2951)?

The code of a corpus of T words under a library of R reals and per-word sets is
    32 R  +  sum over words of [context bits + n KL / ln 2],
the context bits being `gam_mpd::masked::Coder::bits` (each piece on at the word before sends
whether it stays, at its Krichevsky-Trofimov stay rate; new pieces are listed at their KT rates among
new activations, less log2 of their count's factorial), from the counts of some sets.

`Coder` below is that code, line for line, on CSR sets; it is checked against the driver's own
`context_bits` for the given sets it scored (VPD's rounded sets on the eval rows, coded by the
held-out rows' counts). It then prices the trivial explanations exactly:
  (a) the 24 dense maps as 24 always-on subcomponents (KL 0: the model itself);
  (b) all 38,912 VPD subcomponents always on (KL from `vpd_stepA_honesty.py box`, fixed_kl.all_on);
each under two coders: the counts of its own sets on the held-out rows (always on there too, as an
adaptive code learns them), and the Step A coder (VPD's held-out sets' counts). Driver runs on other
libraries ((c) native neurons, (d) per-map SVD, (e) a random basis; `vpd_baseline_libraries.py`)
are read from their OUT.json (last point) with their library sizes.

usage: vpd_code_baselines.py OUT.json STEP_A_PROGRESS.json [--honesty HONESTY.json]
       [--run NAME=OUT.json=LIBRARY_DIR ...] [--observations 1024]
Writes OUT["code"].
"""

import argparse
import fcntl
import json
import math
from pathlib import Path

import numpy as np

H = Path.home() / "mpd-data"
parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("step_a", type=Path)
parser.add_argument("--honesty", type=Path)
parser.add_argument("--run", action="append", default=[])
parser.add_argument("--observations", type=float, default=1024.0)
parser.add_argument("--sets", type=Path, default=H / "pieces/vpd4l_sets")
parser.add_argument("--library", type=Path, default=H / "pieces/vpd4l_library")
parser.add_argument("--eval", type=int, default=32)
args = parser.parse_args()
S = 512
LN2 = math.log(2.0)
BITS_PER_REAL = 32.0


def log2_factorial(k: int) -> float:
    return math.lgamma(k + 1) / LN2


class Counts:
    """`gam_mpd::masked::Context` over pieces numbered globally."""

    def __init__(self, pieces: int):
        self.stayed, self.was_on, self.new = np.zeros(pieces), np.zeros(pieces), np.zeros(pieces)

    def absorb(self, rows: list[np.ndarray]):
        before = np.zeros(0, dtype=np.int64)
        for now in rows:
            self.was_on[before] += 1
            self.stayed[np.intersect1d(before, now, assume_unique=True)] += 1
            self.new[np.setdiff1d(now, before, assume_unique=True)] += 1
            before = now

    def coder(self):
        total = (self.new + 0.5).sum()
        return np.log2(total / (self.new + 0.5)), (self.stayed + 0.5) / (self.was_on + 1.0)


def bits(rows: list[np.ndarray], costs: np.ndarray, stay: np.ndarray) -> float:
    """`Coder::bits` summed over one sequence's words."""
    out = 0.0
    before = np.zeros(0, dtype=np.int64)
    for now in rows:
        on = np.isin(before, now, assume_unique=True)
        q = stay[before]
        out -= np.log2(q[on]).sum() + np.log2(1.0 - q[~on]).sum()
        fresh = np.setdiff1d(now, before, assume_unique=True)
        out += costs[fresh].sum() - log2_factorial(len(fresh))
        before = now
    return out


indptr = np.fromfile(args.sets / "indptr.i64", "<i8")
indices = np.fromfile(args.sets / "indices.i64", "<i8")
manifest = json.load(open(args.library / "manifest.json"))
pieces = sum(m["pieces"] for m in manifest.values())
sequences = (len(indptr) - 1) // S


def sequence(s: int) -> list[np.ndarray]:
    return [indices[indptr[s * S + r]:indptr[s * S + r + 1]] for r in range(S)]


held = Counts(pieces)
for s in range(args.eval, sequences):
    held.absorb(sequence(s))
costs, stay = held.coder()
step_a = json.load(open(args.step_a))
evaluated = step_a["eval_sequences"]
replica = sum(bits(sequence(s), costs, stay) for s in range(evaluated)) / (evaluated * S)
reported = step_a["start"]["context_bits"]
print(f"Coder replica on VPD's sets, {evaluated} eval sequences: {replica:.6f} bits per word; driver {reported:.6f}")
assert abs(replica - reported) < 1e-6 * reported, "the replica is not the driver's code"


def always_on(k: int, held_sequences: int, coder=None) -> float:
    """Context bits per word of k pieces on at every word, by the counts of the same sets on
    `held_sequences` sequences (or by `coder` = (costs, stay) for pieces 0..k-1)."""
    if coder is None:
        c = Counts(k)
        c.new += held_sequences
        c.was_on += held_sequences * (S - 1)
        c.stayed += held_sequences * (S - 1)
        coder = c.coder()
    cst, sty = coder
    first = cst[:k].sum() - log2_factorial(k)
    rest = -(S - 1) * np.log2(sty[:k]).sum()
    return (first + rest) / S


def vpd_reals() -> int:
    return sum(m["pieces"] * (m["d_in"] + m["d_out"]) for m in manifest.values())


dense_reals = sum(m["d_in"] * m["d_out"] for m in manifest.values())
held_sequences = sequences - args.eval
n_scale = args.observations / LN2
kl_all_on = None
if args.honesty is not None and args.honesty.exists():
    kl_all_on = json.load(open(args.honesty))["box"]["fixed_kl"]["all_on"]["mean"]
# The Step A coder prices pieces never on in VPD's held-out sets at their (high) KT listing cost,
# so (b) under it lists each piece by its own held-out rate.
order = np.arange(pieces)
rows = {
    "a_dense_24_always_on": {
        "library_reals": dense_reals, "kl": 0.0, "l0": 24,
        "context_bits_own_counts": always_on(24, held_sequences),
    },
    "b_vpd_all_always_on": {
        "library_reals": vpd_reals(), "kl": kl_all_on, "l0": pieces,
        "context_bits_own_counts": always_on(pieces, held_sequences),
        "context_bits_step_a_coder": always_on(pieces, held_sequences, (costs[order], stay[order])),
    },
    "step_a_vpd_sets": {
        "library_reals": vpd_reals(), "kl": step_a["start"]["kl"], "l0": step_a["start"]["l0"],
        "context_bits_step_a_coder": step_a["start"]["context_bits"],
    },
    "step_a_selected": {
        "library_reals": vpd_reals(), "kl": step_a["kl"], "l0": step_a["l0"],
        "context_bits_step_a_coder": step_a["context_bits"],
    },
}
for spec in args.run:
    name, out_json, lib = spec.split("=")
    points = json.load(open(out_json))["points"]
    p = points[-1]
    lib_manifest = json.load(open(Path(lib) / "manifest.json"))
    rows[name] = {
        "library_reals": sum(m["pieces"] * (m["d_in"] + m["d_out"]) for m in lib_manifest.values()),
        "kl": p["kl"], "l0": p["l0"], "pieces": p["pieces"], "passes": len(points),
        "context_bits_own_counts": p["context_bits"], "eval_sequences": p["eval_sequences"],
    }
for r in rows.values():
    if r["kl"] is None:
        continue
    for coder in ("own_counts", "step_a_coder"):
        if f"context_bits_{coder}" in r:
            r[f"per_word_{coder}"] = r[f"context_bits_{coder}"] + n_scale * r["kl"]
    r["library_bits"] = BITS_PER_REAL * r["library_reals"]
    best = min(v for k, v in r.items() if k.startswith("per_word_"))
    for words in (32 * S, sequences * S):
        r[f"total_bits_at_{words}_words"] = r["library_bits"] + words * best
reference = rows["step_a_selected"]
for name, r in rows.items():
    if r["kl"] is None or name == "step_a_selected":
        continue
    best = min(v for k, v in r.items() if k.startswith("per_word_"))
    r["per_word_minus_step_a"] = best - reference["per_word_step_a_coder"]
    gap = reference["per_word_step_a_coder"] - best
    lib_gap = r["library_bits"] - reference["library_bits"]
    r["shorter_than_step_a_at_every_corpus_size"] = bool(gap > 0 and lib_gap <= 0)
    r["break_even_words"] = (lib_gap / gap) if gap > 0 and lib_gap > 0 else None
    print(f"{name}: per word {best:.2f} vs Step A {reference['per_word_step_a_coder']:.2f}; library "
          f"{r['library_bits'] / 1e9:.2f} vs {reference['library_bits'] / 1e9:.2f} Gbit")
# The honesty runs share the file (vpd_stepA_honesty.py): read-modify-write under their lock.
lock = open(args.out.with_suffix(".lock"), "w")
fcntl.flock(lock, fcntl.LOCK_EX)
data = json.load(open(args.out)) if args.out.exists() else {}
data["code"] = {"observations": args.observations, "bits_per_real": BITS_PER_REAL, "words_per_sequence": S,
                "held_out_sequences": held_sequences, "replica_check": {"replica": replica, "driver": reported},
                "rows": rows}
tmp = args.out.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=1))
tmp.replace(args.out)
