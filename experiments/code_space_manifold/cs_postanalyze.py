"""Post-analysis of code_space_manifold outputs: within-block edge diagnostic (is a
discovered 'secant edge' just the dictionary's own block pair, or a genuine cross-block
manifold adjacency?), RD Pareto verdict, and calendar readout. Prints a markdown block.
"""
import json, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "code_space_out_0"
G = json.load(open(f"{d}/discovered_groups.json"))
RD = json.load(open(f"{d}/rate_distortion.json"))
S = G["summary"]
groups = G["groups"]

# within-block: block dict has block_size=2 -> atoms 2g,2g+1 form block g.
def same_block(a, b):
    return a // 2 == b // 2
n_edges = 0; n_within = 0
for g in groups:
    for a, b, c in g["edges"]:
        n_edges += 1
        if same_block(a, b):
            n_within += 1
frac_within = n_within / max(1, n_edges)

# RD Pareto: for each flat point, is there a manifold point with <= bits AND >= EV?
flat = RD["flat"]; mani = RD["manifold"]
def dominated_by_manifold(fp):
    return any(mp["bits_per_token"] <= fp["bits_per_token"] + 1e-9 and mp["ev"] >= fp["ev"] - 1e-9
              for mp in mani)
def manifold_below_left():
    # is any manifold point strictly below-left of ALL flat points at its EV?
    wins = []
    for mp in mani:
        # nearest flat point at >= mp ev
        cand = [fp for fp in flat if fp["ev"] >= mp["ev"] - 1e-9]
        if cand:
            best = min(cand, key=lambda fp: fp["bits_per_token"])
            wins.append((mp["ev"], mp["bits_per_token"], best["bits_per_token"],
                         mp["bits_per_token"] < best["bits_per_token"]))
    return wins

print("## Post-analysis\n")
print(f"- N_tokens: {S['N_tokens']}  active: {S['active']}  K: {S['K']}  groups: {S['n_groups']}")
print(f"- flat active-32 recon EV: {S['flat_active_recon_ev']:.4f}")
print(f"- manifold max-fidelity (2-hot) EV: {S['manifold_maxfidelity_ev']:.4f}  "
      f"(re-code distortion drop {S['manifold_recode_distortion_ev_drop']:.4f})")
print(f"- fraction of firings captured by 1-param groups: {S['frac_firings_grouped']:.4f} "
      f"({S['n_grouped_secant_firings']}/{S['N_tokens']*S['active']})")
print(f"- topology histogram: {S['topology_histogram']}")
print(f"- size histogram: {S['size_histogram']}")
print(f"- barycentric groups (2-hot>0.7 & decoder-adjacent): {S['n_barycentric_groups']}")
print(f"- WITHIN-BLOCK edge fraction: {frac_within:.3f} ({n_within}/{n_edges}) "
      f"[high => 'secant edges' are the dict's own block pairs, not cross-block manifolds]")
print(f"- calendar candidates: {S['calendar_candidates']}")
print("\n### RD Pareto (manifold vs flat)\n")
print("manifold below-left check (ev, mani_bits, best_flat_bits_at>=ev, mani_wins):")
for w in manifold_below_left():
    print(f"  ev={w[0]:.4f}  mani={w[1]:.0f}b  flat={w[2]:.0f}b  win={w[3]}")
any_win = any(w[3] for w in manifold_below_left())
print(f"\nVERDICT: manifold coding {'BEATS' if any_win else 'does NOT beat'} flat "
      f"(fewer bits at matched EV) on the same dictionary.")
