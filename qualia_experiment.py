#!/usr/bin/env python3
"""
Within-layer manifold-SAE interpretability experiment on the OLMo-3-32B qualia probe.

Data:
  activations.npy : (635 prompts, 64 layers, 5120 hidden)  residual stream
  prompts.jsonl   : 635 rows in 254 contrastive pairs (side in {exp, noexp})

Science:
  1. qualia axis = difference-of-means(exp - noexp) at layer L; logistic probe AUC.
  2. PCA scatter colored by exp/noexp -> clean direction vs curved?
  3. manifold-SAE WITHIN layer L (circle vs euclidean): convergence + held-out EV + atom-axis alignment.
  4. across-layer AUC trajectory (qualia emerges at layer X).
"""
import sys, json, os
import numpy as np

ROOT = "/projects/standard/hsiehph/sauer354"
sys.path.insert(0, ROOT + "/gam")
import gamfit

DATA = ROOT + "/olmo_data"
PLOTS = ROOT + "/plots"
os.makedirs(PLOTS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------- load ----------
acts = np.load(DATA + "/base/activations.npy")  # (635, 64, 5120)
prompts = [json.loads(l) for l in open(DATA + "/base/prompts.jsonl")]
assert acts.shape[0] == len(prompts), (acts.shape, len(prompts))
N, NL, H = acts.shape
print(f"activations {acts.shape}; {len(prompts)} prompts")

# label: 1 = exp (experiential), 0 = noexp. row order == id order (verified)
ids = np.array([p["id"] for p in prompts])
assert (ids == np.arange(N)).all(), "prompt id order must match activation rows"
y = np.array([1 if p["side"] == "exp" else 0 for p in prompts])
pair_id = np.array([p["pair_id"] for p in prompts])
print(f"exp={int(y.sum())} noexp={int((1-y).sum())} pairs={len(set(pair_id))}")


def auc(scores, labels):
    # rank-based AUC (Mann-Whitney)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def dom_axis(X, lab):
    """difference-of-means unit axis exp - noexp"""
    ax = X[lab == 1].mean(0) - X[lab == 0].mean(0)
    return ax / (np.linalg.norm(ax) + 1e-12)


def logistic_probe_auc(X, lab, folds_pairs, l2=1.0):
    """Pair-held-out logistic-regression AUC (gradient descent, no sklearn dep)."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    scores = np.full(len(lab), np.nan)
    upairs = np.unique(folds_pairs)
    # 5-fold over pairs
    fold_of = {p: i % 5 for i, p in enumerate(rng.permutation(upairs))}
    foldid = np.array([fold_of[p] for p in folds_pairs])
    for f in range(5):
        tr = foldid != f
        te = foldid == f
        Xtr, ytr = Xs[tr], lab[tr].astype(float)
        w = np.zeros(Xs.shape[1]); b = 0.0
        lr = 0.5
        for _ in range(300):
            z = Xtr @ w + b
            p = 1 / (1 + np.exp(-z))
            g = p - ytr
            gw = Xtr.T @ g / len(ytr) + l2 * w / len(ytr)
            gb = g.mean()
            w -= lr * gw; b -= lr * gb
        scores[te] = Xs[te] @ w + b
    return auc(scores, lab)


def pca(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :k] * S[:k]
    return Z, Vt[:k], S


# =========================================================
# PART 4 first (cheap): across-layer AUC trajectory
# =========================================================
print("\n=== across-layer qualia separability ===")
layer_auc_dom = []
layer_auc_probe = []
for L in range(NL):
    X = acts[:, L, :].astype(np.float64)
    ax = dom_axis(X, y)
    layer_auc_dom.append(auc(X @ ax, y))
    Zp, _, _ = pca(X, 50)
    layer_auc_probe.append(logistic_probe_auc(Zp, y, pair_id))
layer_auc_dom = np.array(layer_auc_dom)
layer_auc_probe = np.array(layer_auc_probe)
best_layer = int(np.argmax(layer_auc_probe))
print(f"best probe layer = {best_layer} (auc={layer_auc_probe[best_layer]:.4f})")
print(f"layer 18 dom-auc={layer_auc_dom[18]:.4f} probe-auc={layer_auc_probe[18]:.4f}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(NL), layer_auc_dom, "-o", ms=3, label="diff-of-means axis AUC", color="C0")
ax1.plot(range(NL), layer_auc_probe, "-s", ms=3, label="logistic probe AUC (pair-held-out, 50 PC)", color="C3")
ax1.axvline(18, color="gray", ls="--", alpha=0.6, label="bank layer 18")
ax1.axhline(0.5, color="k", ls=":", alpha=0.4)
ax1.set_xlabel("layer"); ax1.set_ylabel("AUC (exp vs noexp)")
ax1.set_title("OLMo-3-32B: qualia-axis separability across layers")
ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(PLOTS + "/qualia_layer_trajectory.png", dpi=140)
plt.close(fig)
print("saved qualia_layer_trajectory.png")

# =========================================================
# PART 2: PCA scatter + DOM axis + probe reproduction per layer
# =========================================================
def analyze_layer(L):
    print(f"\n=== LAYER {L} ===")
    X = acts[:, L, :].astype(np.float64)
    ax = dom_axis(X, y)
    auc_dom = auc(X @ ax, y)
    Zp, _, _ = pca(X, 50)
    auc_probe = logistic_probe_auc(Zp, y, pair_id)
    print(f"  diff-of-means AUC = {auc_dom:.4f}; logistic probe AUC = {auc_probe:.4f}")

    # PCA scatter; also project dom axis onto PC plane to see if direction is clean
    Z, Vt, S = pca(X, 4)
    qcoord = X @ ax  # qualia coordinate
    # curvature diagnostic: is class boundary linear in PC space? compare linear-probe
    # AUC on full PCA vs nonlinear (quadratic) probe
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for cls, col, lbl in [(1, "#d62728", "exp (experiential)"), (0, "#1f77b4", "noexp")]:
        m = y == cls
        axes[0].scatter(Z[m, 0], Z[m, 1], c=col, s=18, alpha=0.7, label=lbl, edgecolors="none")
    # project dom axis into PC1-PC2 plane
    a2 = Vt[:2] @ ax
    a2 = a2 / (np.linalg.norm(a2) + 1e-12)
    sc = (Z[:, 0].std() + Z[:, 1].std())
    axes[0].annotate("", xy=(a2[0]*sc*1.5, a2[1]*sc*1.5), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="k", lw=2))
    axes[0].text(a2[0]*sc*1.6, a2[1]*sc*1.6, "qualia axis", fontsize=9)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title(f"layer {L} PCA (PC1-PC2)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # qualia-coordinate vs PC1 colored: shows whether separation is 1D-clean
    axes[1].scatter(qcoord, Z[:, 1], c=[("#d62728" if t else "#1f77b4") for t in y], s=18, alpha=0.7)
    axes[1].set_xlabel("qualia coordinate (X . dom-axis)"); axes[1].set_ylabel("PC2")
    axes[1].set_title(f"layer {L}: qualia coord vs orthogonal PC2  (AUC={auc_dom:.3f})")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS + f"/qualia_pca_layer{L}.png", dpi=140)
    plt.close(fig)
    print(f"  saved qualia_pca_layer{L}.png")
    return dict(layer=L, auc_dom=auc_dom, auc_probe=auc_probe, ax=ax, Vt=Vt, S=S)


layer_results = {L: analyze_layer(L) for L in (12, 18, 25)}


# =========================================================
# PART 3: manifold-SAE WITHIN layer (circle vs euclidean)
# =========================================================
def held_out_ev(model, Xte):
    """explained variance of model reconstruction on held-out rows via encode/predict."""
    try:
        rec = model.predict(Xte)
    except Exception:
        try:
            rec = model.reconstruct(Xte)
        except Exception:
            return None
    rec = np.asarray(rec)
    ss_res = ((Xte - rec) ** 2).sum()
    ss_tot = ((Xte - Xte.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot


def sae_within_layer(L, K=2, pca_dim=24, seed=0):
    print(f"\n=== manifold-SAE WITHIN layer {L} (K={K}, pca_dim={pca_dim}) ===")
    X = acts[:, L, :].astype(np.float64)
    Z, Vt, S = pca(X, pca_dim)          # PCA-reduce as pipeline does
    # standardize PCA scores
    Zs = Z / (Z.std(0) + 1e-9)
    # train/held-out split by PAIR (keep pairs together)
    upairs = np.unique(pair_id)
    te_pairs = set(rng.permutation(upairs)[: len(upairs) // 5])
    te = np.array([p in te_pairs for p in pair_id])
    tr = ~te
    ax_lat = dom_axis(Zs, y)  # qualia axis in latent (PCA) space
    out = {}
    for topo in ("euclidean", "circle"):
        try:
            m = gamfit.sae_manifold_fit(
                Zs[tr], K=K, d_atom=1, atom_topology=topo,
                n_iter=300, random_state=seed,
            )
            evtr = getattr(m, "reconstruction_r2", None)
            evte = held_out_ev(m, Zs[te])
            reml = getattr(m, "reml_score", None)
            # atom principal-direction alignment with qualia axis
            cos_best = None
            atoms = getattr(m, "atoms", [])
            for a in atoms:
                B = np.asarray(a.decoder_coefficients)  # (M_k, p=pca_dim)
                # principal output direction of this atom's decoder
                u, s_, vt_ = np.linalg.svd(B, full_matrices=False)
                d = vt_[0]
                d = d / (np.linalg.norm(d) + 1e-12)
                c = abs(float(d @ ax_lat))
                cos_best = c if cos_best is None else max(cos_best, c)
            out[topo] = dict(conv=True, ev_tr=evtr, ev_te=evte, reml=reml, cos_qualia=cos_best,
                             n_atoms=len(atoms))
            print(f"  [{topo}] converged: ev_tr={evtr}, ev_te={evte}, reml={reml}, "
                  f"cos(atom,qualia)={cos_best}")
        except Exception as e:
            out[topo] = dict(conv=False, err=repr(e)[:300])
            print(f"  [{topo}] FAILED: {repr(e)[:300]}")
    out["ax_lat_norm"] = float(np.linalg.norm(ax_lat))
    out["n_tr"] = int(tr.sum()); out["n_te"] = int(te.sum())
    return out


sae_results = {}
for L in (18, 12, 25):
    sae_results[L] = {}
    for K in (1, 2, 3):
        sae_results[L][K] = sae_within_layer(L, K=K)

# =========================================================
# summary JSON
# =========================================================
summary = dict(
    n_prompts=N, n_layers=NL, hidden=H,
    layer_auc_dom=layer_auc_dom.tolist(),
    layer_auc_probe=layer_auc_probe.tolist(),
    best_probe_layer=best_layer,
    layer18_dom_auc=float(layer_auc_dom[18]),
    layer18_probe_auc=float(layer_auc_probe[18]),
    per_layer={str(L): {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                        for k, v in r.items() if k in ("auc_dom", "auc_probe")}
               for L, r in layer_results.items()},
    sae={str(L): {str(K): {t: {kk: (vv if isinstance(vv, (bool, str)) or vv is None
                                     else float(vv))
                               for kk, vv in d.items()}
                           if isinstance(d, dict) else d
                           for t, d in kd.items()}
                  for K, kd in ld.items()}
         for L, ld in sae_results.items()},
)
with open(PLOTS + "/qualia_experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nWROTE", PLOTS + "/qualia_experiment_summary.json")
print("DONE")
