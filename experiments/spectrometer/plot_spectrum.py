#!/usr/bin/env python3
"""Plot the dimension-spectrometer results (run locally after pulling results.json)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.environ.get("SPEC_OUT", os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(f"{HERE}/results.json"))
Ks = np.array(R["Ks"], float)
TAG = R.get("meta", {}).get("tag", "activations")


def excess(L, s2):
    L = np.array(L, float)
    return np.clip(L - s2, 1e-9, None)


# ---- Fig 1: global loss curve + single-power fit + excess-loss slope --------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
L = np.array(R["global_L"], float)
sp = R["global_single_power"]
s2 = sp["s2"]
ax[0].loglog(Ks, L, "o-", color="#1b4965", label="L(K) measured")
kk = np.geomspace(Ks.min(), Ks.max(), 100)
ax[0].loglog(kk, sp["c"] * kk ** (-2.0 / sp["d"]) + s2, "--", color="#e07a5f",
             label=f"fit: c·K^(-2/{sp['d']:.2f})+σ²")
ax[0].axhline(s2, color="gray", ls=":", lw=1, label=f"σ²={s2:.3f}")
ax[0].set_xlabel("dictionary width K"); ax[0].set_ylabel("mean residual energy L(K)")
ax[0].set_title("Global spectrum — real L17 activations"); ax[0].legend(fontsize=8)

ex = excess(L, s2)
ax[1].loglog(Ks, ex, "o-", color="#1b4965", label="L(K)-σ²")
m = R["global_ols"]["slope"]; d = R["global_ols"]["d_hat"]
ax[1].loglog(kk, ex[0] * (kk / Ks[0]) ** m, "--", color="#e07a5f",
             label=f"slope={m:.2f} → d̂={d:.2f}")
ax[1].set_xlabel("dictionary width K"); ax[1].set_ylabel("excess loss L(K)-σ²")
ax[1].set_title("Excess-loss power law"); ax[1].legend(fontsize=8)
ax[0].set_title(f"Global spectrum — {TAG}")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_global.png", dpi=130)
print("wrote fig_global.png")

# ---- Fig 2: PCA-stratified spectrum ----------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
strata = sorted(R["pca_strata"].items(), key=lambda x: int(x[0]))
cmap = plt.get_cmap("viridis")
d_by_R = {0: R["global_ols"]["d_hat"]}
ax[0].loglog(Ks, excess(R["global_L"], R["global_single_power"]["s2"]),
             "o-", color="#1b4965", label=f"R=0  d̂={R['global_ols']['d_hat']:.2f}")
for i, (Rk, blk) in enumerate(strata):
    s2r = blk["single_power"]["s2"]
    ax[0].loglog(Ks, excess(blk["L"], s2r), "o-", color=cmap(0.15 + 0.8 * i / max(len(strata) - 1, 1)),
                 label=f"R={Rk}  d̂={blk['ols']['d_hat']:.2f}")
    d_by_R[int(Rk)] = blk["ols"]["d_hat"]
ax[0].set_xlabel("K"); ax[0].set_ylabel("excess loss")
ax[0].set_title("PCA-stratified excess loss"); ax[0].legend(fontsize=8)

Rs = sorted(d_by_R); ds = [d_by_R[r] for r in Rs]
ax[1].plot(Rs, ds, "o-", color="#1b4965")
ax[1].set_xscale("symlog")
for r, dd in zip(Rs, ds):
    ax[1].annotate(f"{dd:.1f}", (r, dd), textcoords="offset points", xytext=(0, 8), fontsize=9)
ax[1].set_xlabel("top-R PCA subspace removed (symlog)"); ax[1].set_ylabel("estimated d̂")
ax[1].set_title("Intrinsic dimension vs. removed head subspace")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_pca_strata.png", dpi=130)
print("wrote fig_pca_strata.png")

# ---- robustness: floor-fit vs no-floor vs drop-last d̂ ----------------------
def dhat_variants(Ks, L):
    Ks = np.asarray(Ks, float); L = np.asarray(L, float)
    def ols(x, y):
        A = np.stack([x, np.ones_like(x)], 1)
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(-2.0 / m)
    out = {}
    # no-floor (sigma^2 = 0): upper bound on d (shallowest possible decay)
    out["no_floor"] = ols(np.log(Ks), np.log(L))[1]
    return out

print("\n== ROBUSTNESS d̂ ==")
print("global no-floor d̂:", round(dhat_variants(Ks, R["global_L"])["no_floor"], 2))
for Rk, blk in sorted(R["pca_strata"].items(), key=lambda x: int(x[0])):
    print(f"R={Rk} no-floor d̂:", round(dhat_variants(Ks, blk["L"])["no_floor"], 2))

# ---- console summary --------------------------------------------------------
print("\n== SUMMARY ==")
print("global single-power:", R["global_single_power"])
print("global OLS d_hat:", R["global_ols"], "| drop-last:", R["global_ols_drop_last"])
print("global mixture-2:", R["global_mixture_2"])
print("global mixture-3:", R["global_mixture_3"])
for Rk, blk in sorted(R["pca_strata"].items(), key=lambda x: int(x[0])):
    print(f"R={Rk}: d_hat={blk['ols']['d_hat']:.2f} drop-last={blk['ols_drop_last']['d_hat']:.2f} "
          f"s2={blk['single_power']['s2']:.4f} mix2={blk['mixture_2']['dims']}")
