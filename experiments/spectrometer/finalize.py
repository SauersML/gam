#!/usr/bin/env python3
"""Assemble the depth-resolved dimension spectrum from the per-condition
results_*.json files pulled from MSI. Emits a markdown block for results.md
and a combined depth-trend PNG. Pure numpy/json (no scipy)."""
import json, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# condition -> (label, depth_key for ordering)
CONDS = [
    ("results_q8b_L6.json",  "Qwen3-8B L6 (early)",   6),
    ("results_q8b_L18.json", "Qwen3-8B L18 (mid)",    18),
    ("results_q8b_L30.json", "Qwen3-8B L30 (late)",   30),
    ("results_q36b_L17.json","Qwen3.6-35B-A3B L17",   17),
]


def ols(logK, logL):
    A = np.stack([logK, np.ones_like(logK)], 1)
    m, b = np.linalg.lstsq(A, logL, rcond=None)[0]
    return float(m), float(-2.0 / m)


def dhats(Ks, L, s2):
    Ks = np.asarray(Ks, float); L = np.asarray(L, float)
    ex = np.clip(L - s2, 1e-12, None)
    ok = ex > 1e-11
    m_full, d_full = ols(np.log(Ks[ok]), np.log(ex[ok]))
    # drop last rung
    if ok.sum() > 3:
        m_dl, d_dl = ols(np.log(Ks[ok][:-1]), np.log(ex[ok][:-1]))
    else:
        m_dl, d_dl = m_full, d_full
    # no-floor
    m_nf, d_nf = ols(np.log(Ks), np.log(np.clip(L, 1e-12, None)))
    return dict(d_full=d_full, d_droplast=d_dl, d_nofloor=d_nf)


def main():
    rows = []
    for fn, label, depth in CONDS:
        p = f"{HERE}/{fn}"
        if not os.path.exists(p):
            print(f"MISSING {fn}"); continue
        R = json.load(open(p))
        Ks = R["Ks"]
        g = R["global_single_power"]; gd = dhats(Ks, R["global_L"], g["s2"])
        # PCA strata d_hat (peeling the massive directions)
        strat = {}
        for Rk, blk in R.get("pca_strata", {}).items():
            strat[int(Rk)] = dhats(Ks, blk["L"], blk["single_power"]["s2"])["d_full"]
        rows.append(dict(label=label, depth=depth, d_model=R["p"], N=R["N"],
                         top_var=R["meta"].get("top_var_frac"),
                         g_single_d=g["d"], g_single_s2=g["s2"],
                         d_full=gd["d_full"], d_droplast=gd["d_droplast"],
                         d_nofloor=gd["d_nofloor"],
                         mix2=R["global_mixture_2"]["dims"],
                         mix2_w=R["global_mixture_2"]["weights"],
                         strata=strat))

    # markdown
    print("\n### Depth-resolved d̂ (global, full ladder / drop-last / no-floor)\n")
    print("| condition | d_model | N | R=0 full | drop-last | no-floor | single-power fit | mixture-2 dims (weights) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        mix = ", ".join(f"{d:.1f}" for d in r["mix2"])
        w = ", ".join(f"{x:.2g}" for x in r["mix2_w"])
        print(f"| {r['label']} | {r['d_model']} | {r['N']} | {r['d_full']:.1f} | "
              f"{r['d_droplast']:.1f} | {r['d_nofloor']:.1f} | d={r['g_single_d']:.1f}, σ²={r['g_single_s2']:.3g} | {mix} ({w}) |")

    print("\n### PCA-stratified d̂ (peeling the massive directions)\n")
    print("| condition | R=0 | R=1 | R=2 | R=16 | R=64 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        s = r["strata"]
        cells = " | ".join(f"{s.get(R, float('nan')):.1f}" if R in s else "—"
                           for R in (0, 1, 2, 16, 64))
        print(f"| {r['label']} | {cells} |")

    # combined depth-trend PNG (Qwen3-8B only)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        q8 = [r for r in rows if "8B" in r["label"]]
        q8.sort(key=lambda r: r["depth"])
        if len(q8) >= 2:
            depths = [r["depth"] for r in q8]
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            for key, lab, c in [("d_full", "R=0 full-ladder", "#1b4965"),
                                 ("d_droplast", "R=0 drop-last", "#e07a5f")]:
                ax.plot(depths, [r[key] for r in q8], "o-", color=c, label=lab)
            # R=16 stratum (post-massive-direction removal)
            r16 = [r["strata"].get(16, np.nan) for r in q8]
            ax.plot(depths, r16, "s--", color="#81b29a", label="R=16 (post-peel)")
            ax.set_xlabel("Qwen3-8B layer (depth)"); ax.set_ylabel("estimated intrinsic d̂")
            ax.set_title("Depth-resolved intrinsic dimension — Qwen3-8B"); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(f"{HERE}/fig_depth_trend.png", dpi=130)
            print("\nwrote fig_depth_trend.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
