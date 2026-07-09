#!/usr/bin/env python3
"""Print bits(flat) - bits(hybrid) per distortion target from a #1026 run.

The driver runs ONE arm per process (each a short schedulable job), so the
cross-arm Eq-4 comparison the #1026 close reports — the MDL margin gam#2233
predicts curved atoms win by — is assembled here from the results jsonl after
the arms land. Lower bits is better; a POSITIVE bits(flat) - bits(hybrid) means
the hybrid dictionary describes the same held-out cloud in fewer bits.

    python3 compare_bits.py results_1026.jsonl [--seed 0] [--tag ...]

For each (flat arm, hybrid arm) pair at the matched seed/tag it prints the
per-target delta and the support/code/residual breakdown that localizes WHERE
the win comes from (the theorem: circles win support+residual, not code).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

FLAT_ARMS = ("gam_flat", "external_topk")
HYBRID_ARMS = ("hybrid_rust", "hybrid")
TARGETS = ("0.99", "0.95", "0.9", "0.8")


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def latest_by_arm(recs, seed, tag):
    """Last record per arm matching seed (and tag if given) — the freshest run."""
    by_arm = {}
    for r in recs:
        if seed is not None and r.get("seed") != seed:
            continue
        if tag is not None and r.get("tag") != tag:
            continue
        if f"bits_bits_at_r2_{TARGETS[0]}" not in r and "bits_bits_at_r2_0.99" not in r:
            # accept any record carrying bits fields
            if not any(k.startswith("bits_bits_at_r2_") for k in r):
                continue
        by_arm[r["arm"]] = r
    return by_arm


def bits_at(rec, target):
    return rec.get(f"bits_bits_at_r2_{target}")


def breakdown(rec, target):
    return (
        rec.get("bits_support_bits"),
        rec.get(f"bits_code_bits_at_r2_{target}"),
        rec.get(f"bits_resid_bits_at_r2_{target}"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="results_1026.jsonl written by the driver")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    recs = load(args.results)
    by_arm = latest_by_arm(recs, args.seed, args.tag)
    flats = [a for a in FLAT_ARMS if a in by_arm]
    hybrids = [a for a in HYBRID_ARMS if a in by_arm]
    if not flats or not hybrids:
        print(f"[compare_bits] need >=1 flat {FLAT_ARMS} and >=1 hybrid "
              f"{HYBRID_ARMS} arm with bits fields; found flats={flats} "
              f"hybrids={hybrids}")
        return 1

    for flat in flats:
        for hyb in hybrids:
            fr, hr = by_arm[flat], by_arm[hyb]
            print(f"\n=== bits({flat}) - bits({hyb})  "
                  f"[seed={fr.get('seed')} K={fr.get('K')} top_k={fr.get('top_k')}] ===")
            print(f"  EV: {flat}={fr.get('ev'):.5f}  {hyb}={hr.get('ev'):.5f}")
            print(f"  {'target R2':>10} {'bits(flat)':>12} {'bits(hyb)':>12} "
                  f"{'Δ=flat-hyb':>12}  (Δsupport/Δcode/Δresid)")
            for t in TARGETS:
                bf, bh = bits_at(fr, t), bits_at(hr, t)
                if bf is None or bh is None:
                    continue
                fs, fc, frd = breakdown(fr, t)
                hs, hc, hrd = breakdown(hr, t)
                dsup = (fs - hs) if (fs is not None and hs is not None) else float("nan")
                dcode = (fc - hc) if (fc is not None and hc is not None) else float("nan")
                dres = (frd - hrd) if (frd is not None and hrd is not None) else float("nan")
                print(f"  {t:>10} {bf:>12.1f} {bh:>12.1f} {bf - bh:>12.1f}"
                      f"   ({dsup:+.1f}/{dcode:+.1f}/{dres:+.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
