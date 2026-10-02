"""Print the census table (bits saved by each shared-structure family) for both models."""
import json
from pathlib import Path

F = Path.home() / "mpd-data/frontier"
for m in ("vpd4l", "pythia70m"):
    p = F / f"census_{m}.json"
    if not p.exists():
        continue
    r = json.load(open(p))
    D = r["dense_bits"]
    print(f"== {m}: dense b=4 {D:.4g} bits")
    print(f"  per-matrix SVD+residual saved vs dense: {r['per_matrix']['saved_vs_dense']} ({r['per_matrix']['saved_vs_dense'] / D:.2e})")
    s = r.get("subspace", {})
    best = max(s.items(), key=lambda kv: kv[1]["saved"]) if s else None
    print(f"  shared subspace: sum over groups {sum(v['saved'] for v in s.values())}; best {best[0]} {best[1]['saved']} (rank {best[1]['rank']})" if best else "  subspace: -")
    c = r.get("clusters", {})
    print("  clusters best per scope:", {k: max(x["saved"] for x in v) for k, v in c.items()})
    g = r.get("gauge")
    if g:
        print(f"  permutations: {g['permutation_bits']:.0f} bits; continuous orbit bound {g['orbit_bound_bits']:.4g} bits "
              f"({g['orbit_reals_ov']} OV + {g['orbit_reals_qk']} QK reals); heads-up-to-gauge realized {g['heads_up_to_gauge_saved_bits']}")
        print(f"  nearest-head gauge-invariant rel dist: QK {g['head_nearest_rel_dist_qk_min']:.3f} OV {g['head_nearest_rel_dist_ov_min']:.3f}")
        print("  neuron duplicates:", {k: v for k, v in g["neuron_duplicates"].items()})
    L = F / f"census_lossy_{m}.json"
    if L.exists():
        lo = json.load(open(L))
        for k, v in lo.items():
            rows = [(x["frac"], round(x["kl_budget"], 3), x["per_matrix_reals"], x["shared_reals"]) for x in v["rows"]]
            print(f"  lossy {k}: {rows}")
