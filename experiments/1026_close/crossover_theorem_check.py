#!/usr/bin/env python3
"""Closed-form instantiation of the gam#2233 MDL crossover theorem.

This reproduces the EXACT bit terms the Rust Eq-4 scorer
(`crates/gam-sae/src/eq4_description_length.rs`) charges, in pure numpy, so the
theorem's parameter-count inequality and the support/dict crossover can be
checked at the REAL creditscope scale (K=32768, P=2048, L0=32, N=8192) without
the wheel or a GPU. It does NOT replace the scorer-in-the-loop run (which also
prices the water-filled code + residual terms on fitted spectra); it certifies
the two terms the theorem turns on and the config that makes the contest fair.

Scorer terms reproduced verbatim (see eq4_description_length.rs):
  * support_bits   = log2 C(G, round(L0))                      [selection_bits]
  * dictionary_bits = 0.5 * dictionary_params / N * log2(max(N,2))
The code + residual terms are joint reverse-water-filling on fitted spectra;
for a MATCHED-recon circle (s = d+1) the theorem gives dcode = 0, dresid = 0,
so at fixed R^2 the flat-vs-hybrid gap is exactly dsupport - ddict, which this
script evaluates in closed form. Higher-span kinds (s > d+1) add a strictly
POSITIVE dcode on top, so the circle case is the theorem's floor.
"""
from __future__ import annotations

import argparse
import math


def selection_bits(g_dict: int, k_active: int) -> float:
    """log2 C(G, k) via the scorer's overflow-safe product form."""
    if g_dict <= 0 or k_active <= 0:
        return 0.0
    k = min(k_active, g_dict)
    return sum(math.log2((g_dict - k + i) / i) for i in range(1, k + 1))


def dict_bits(dictionary_params: int, n: int) -> float:
    """0.5 * params / N * log2(max(N,2)) — the scorer's BIC dictionary term."""
    return 0.5 * dictionary_params / n * math.log2(max(n, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=32768, help="external flat atom count")
    ap.add_argument("--P", type=int, default=2048, help="ambient (decoder-row) dim")
    ap.add_argument("--N", type=int, default=8192, help="bits-scoring sample size")
    ap.add_argument("--top-k", type=int, default=32, help="mean active atoms/token L0")
    ap.add_argument("--curved-atoms", type=int, default=256)
    ap.add_argument("--harmonics", type=int, default=3, help="H; circle basis b = 2H+1")
    ap.add_argument("--k-circ-active", type=int, default=8,
                    help="how many of the L0 active flat slots are circle planes "
                         "(each = 2 flat atoms the curved tier collapses to 1)")
    args = ap.parse_args()

    K, P, N, L0 = args.K, args.P, args.N, args.top_k
    b = 2 * args.harmonics + 1           # circle Fourier basis width (decoder cols/atom)
    Kc = args.curved_atoms
    s = 2                                 # circle spans s = d+1 = 2 ambient dims
    d = 1

    print(f"# gam#2233 crossover theorem @ real scale  K={K} P={P} N={N} L0={L0}")
    print(f"# circle kind: intrinsic d={d}, span s={s}, basis b=2H+1={b}, "
          f"curved atoms Kc={Kc}")
    print()

    # ------------------------------------------------------------------ #
    # 1. The dictionary-parameter inequality (the ~95% term at K=32768).
    # ------------------------------------------------------------------ #
    dp_ext = K * P
    # (a) STACKED hybrid (the current driver default): all K flat atoms kept,
    #     curved atoms added on top.
    dp_stacked = K * P + Kc * b * P
    # (b) FAITHFUL hybrid: reduce the flat atom count so params match EXACTLY.
    #     k_flat = K - Kc*b  =>  dp = (K - Kc*b)*P + Kc*b*P = K*P.
    k_flat_faithful = K - Kc * b
    dp_faithful = k_flat_faithful * P + Kc * b * P

    print("## 1. dictionary parameter inequality  dict_params(hybrid) <= K*P ?")
    print(f"  external (K flat)          params = {dp_ext:>14,}  "
          f"dict_bits = {dict_bits(dp_ext, N):10.1f}")
    print(f"  STACKED hybrid  (k_flat=K) params = {dp_stacked:>14,}  "
          f"dict_bits = {dict_bits(dp_stacked, N):10.1f}   "
          f"SURCHARGE +{dict_bits(dp_stacked, N) - dict_bits(dp_ext, N):.1f} bits  "
          f"[faithful={dp_stacked <= dp_ext}]")
    print(f"  FAITHFUL hybrid k_flat={k_flat_faithful:<6d}  params = {dp_faithful:>14,}  "
          f"dict_bits = {dict_bits(dp_faithful, N):10.1f}   "
          f"EQUAL (delta {dict_bits(dp_faithful, N) - dict_bits(dp_ext, N):+.1f})  "
          f"[faithful={dp_faithful <= dp_ext}]")
    print(f"  => the theorem-faithful sbatch sets --k-flat {k_flat_faithful} "
          f"(= K - curved_atoms*(2H+1))")
    print()

    # ------------------------------------------------------------------ #
    # 2. The support win at the faithful config (dict term neutralized).
    #    Flat lights s=2 atoms per circle; curved lights 1 => L0 drops by
    #    k_circ_active on the hybrid. Support = log2 C(G, L0).
    # ------------------------------------------------------------------ #
    kca = args.k_circ_active
    l0_flat = L0
    l0_hyb = L0 - kca                      # each circle: 2 flat slots -> 1 curved
    G_hyb = k_flat_faithful + Kc           # atoms the hybrid actually indexes
    sup_flat = selection_bits(K, l0_flat)
    sup_hyb = selection_bits(G_hyb, l0_hyb)
    slot = math.log2(K / max(l0_hyb, 1))   # analytic per-slot cost log2(G/L0)

    print("## 2. support win at the faithful config (dcode=dresid=0 for a circle)")
    print(f"  flat   L0={l0_flat}  support = log2 C({K},{l0_flat}) = {sup_flat:10.1f} bits")
    print(f"  hybrid L0={l0_hyb}  support = log2 C({G_hyb},{l0_hyb}) = {sup_hyb:10.1f} bits")
    print(f"  d_support = {sup_flat - sup_hyb:+.1f} bits  "
          f"(~ {kca} freed slots * log2(G/L0)={slot:.1f})")
    print()

    # ------------------------------------------------------------------ #
    # 3. Net bits at fixed R^2 (circle floor: total gap = dsupport - ddict).
    # ------------------------------------------------------------------ #
    net_faithful = (sup_flat + dict_bits(dp_ext, N)) - (sup_hyb + dict_bits(dp_faithful, N))
    net_stacked = (sup_flat + dict_bits(dp_ext, N)) - (sup_hyb + dict_bits(dp_stacked, N))
    print("## 3. net total-bits gap flat - hybrid at fixed R^2 (circle: +ve = hybrid wins)")
    print(f"  STACKED  : d_support {sup_flat - sup_hyb:+.1f}  - d_dict "
          f"{dict_bits(dp_stacked, N) - dict_bits(dp_ext, N):+.1f}  "
          f"= NET {net_stacked:+.1f} bits  (hybrid {'WINS' if net_stacked > 0 else 'LOSES'})")
    print(f"  FAITHFUL : d_support {sup_flat - sup_hyb:+.1f}  - d_dict "
          f"{dict_bits(dp_faithful, N) - dict_bits(dp_ext, N):+.1f}  "
          f"= NET {net_faithful:+.1f} bits  (hybrid {'WINS' if net_faithful > 0 else 'LOSES'})")
    print()

    # ------------------------------------------------------------------ #
    # 4. The per-factor DICTIONARY crossover: when does curvature also cut
    #    dict params? A curved atom of basis b replaces s flat atoms; the
    #    dict term saves iff b <= s. A circle (s=2, b>=3) NEVER saves dict on
    #    1:1 replacement -> it wins on support only. A high-span kind does.
    # ------------------------------------------------------------------ #
    print("## 4. per-factor dict crossover  (curved atom width b vs flat span s)")
    print(f"  circle: b={b} > s={s}  => 1 curved atom COSTS more dict than the {s} "
          f"flat atoms it replaces (win is SUPPORT, not dict).")
    for span in (2, 4, 8, 16):
        verdict = "SAVES dict" if b <= span else "surcharges dict"
        print(f"  span s={span:>2d}: replace {span} flat with 1 curved(b={b}) "
              f"-> dict delta {(b - span) * P:+d} params/factor  [{verdict}]")
    print(f"  => a dict-NEGATIVE hybrid needs high-span kinds (s >= b={b}); the "
          f"circle-dominated creditscope hybrid is dict-neutral-by-construction "
          f"(config in 1) and wins on support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
