"""
Curved-vs-linear on REAL Qwen3-8B L18 (sink-peeled), unsupervised, matched capacity.
The manifold-SAE premise as a head-to-head: a CHART ATLAS (K local curved charts) vs a
GLOBAL LINEAR code. If the atlas reconstructs better at matched parameter budget, curvature
is exploitable on real data; if not, it's a real negative. Also tries the actual gamfit fitter.
Saves curved_vs_linear.png + curved_vs_linear.json.
"""
import json, os, numpy as np

def required_env(name, message):
    try:
        return os.environ[name]
    except KeyError as exc:
        raise SystemExit(message) from exc

R=required_env("GAM_MSI_DATA", "set GAM_MSI_DATA to the activation-harvest root")
OUT=f"{R}/gam_ceiling_fable/experiments/curved_vs_linear"; os.makedirs(OUT, exist_ok=True)
os.environ["MPLCONFIGDIR"]=f"{R}/scratch/mplcache"; os.makedirs(f"{R}/scratch/mplcache",exist_ok=True)

# ---- load real L18 activations ----
path=f"{R}/harvest_out/qwen3_8b_wikitext/resid_L18.npy"
X=np.load(path, mmap_mode="r")
N=min(30000, X.shape[0]); rng=np.random.default_rng(0)
idx=np.sort(rng.choice(X.shape[0], N, replace=False))
X=np.asarray(X[idx], dtype=np.float64)
print(f"loaded L18 {X.shape}", flush=True)

# ---- peel the attention sink (top-k global PCs that carry the massive channel) ----
mu=X.mean(0); Xc=X-mu
U,S,Vt=np.linalg.svd(Xc, full_matrices=False)
var=(S**2)/ (S**2).sum()
n_peel=int((np.cumsum(var)<0.90).sum())+1  # peel until <90% var removed (the sink)
n_peel=max(1,min(n_peel,16))
Xp = Xc - (Xc@Vt[:n_peel].T)@Vt[:n_peel]     # residual after removing top-n_peel PCs (sink-peeled)
print(f"peeled top {n_peel} PCs (sink); top var frac was {var[0]:.3f}", flush=True)
tot = (Xp**2).sum()

def ev(recon): return 1.0 - ((Xp-recon)**2).sum()/tot

# ---- LINEAR baseline: global top-M PCA reconstruction, EV vs capacity ----
Up,Sp,Vtp=np.linalg.svd(Xp, full_matrices=False)
lin={}
for M in [1,2,4,8,16,32,64]:
    recon=(Xp@Vtp[:M].T)@Vtp[:M]
    # capacity = M directions (M*D dict params) + N*M codes
    lin[M]=dict(ev=float(ev(recon)), params=M)
print("LINEAR (global PCA):", {M:round(v["ev"],3) for M,v in lin.items()}, flush=True)

# ---- CURVED atlas: K local charts (k-means clusters), each a local d-dim PCA (a curved chart) ----
# This is the manifold-SAE model: sparse (1 chart active per row) + low-dim coordinate on a curved chart.
def kmeans_np(Xin, K, iters=25, seed=0):
    rg=np.random.default_rng(seed); C=Xin[rg.choice(len(Xin),K,replace=False)].copy()
    lab=np.zeros(len(Xin),dtype=int)
    xn=(Xin**2).sum(1)
    for _ in range(iters):
        d2=xn[:,None]-2*Xin@C.T+(C**2).sum(1)[None,:]
        newlab=d2.argmin(1)
        if (newlab==lab).all() and _>0: break
        lab=newlab
        for k in range(K):
            m=lab==k
            if m.any(): C[k]=Xin[m].mean(0)
    return lab
atlas={}
for (K,d) in [(8,1),(16,1),(32,1),(8,2),(16,2),(32,2),(64,2)]:
    lab=kmeans_np(Xp, K, seed=0); recon=np.zeros_like(Xp)
    for k in range(K):
        m=lab==k
        if m.sum()<d+1: recon[m]=Xp[m].mean(0); continue
        Ck=Xp[m]; ck=Ck.mean(0); Cc=Ck-ck
        uu,ss,vv=np.linalg.svd(Cc, full_matrices=False)
        recon[m]=ck+(Cc@vv[:d].T)@vv[:d]     # local chart reconstruction (piecewise-curved)
    # capacity ~ K charts * (d directions + centroid) ; sparse: 1 chart + d coords per row
    atlas[(K,d)]=dict(ev=float(ev(recon)), K=K, d=d, eff_dim=d, params=K*(d+1))
    print(f"  atlas K={K} d={d}: EV={atlas[(K,d)]['ev']:.3f}", flush=True)

# ---- head-to-head at matched EFFECTIVE capacity: atlas(K,d) sparse code uses d coords/row (like linear M=d)
# fair sparse comparison: 1 active chart, d coordinates  vs  linear M=d global dims (also d coords/row)
compare=[]
for (K,d),a in atlas.items():
    lin_same=lin.get(d)
    if lin_same:
        compare.append(dict(K=K, d=d, atlas_ev=a["ev"], linear_ev=lin_same["ev"],
                            curved_gain=a["ev"]-lin_same["ev"]))
compare.sort(key=lambda r:-r["curved_gain"])
print("\nHEAD-TO-HEAD (same coords/row d): atlas EV vs linear EV", flush=True)
for r in compare[:6]:
    print(f"  K={r['K']} d={r['d']}: atlas={r['atlas_ev']:.3f} linear={r['linear_ev']:.3f} gain={r['curved_gain']:+.3f}", flush=True)
best=compare[0]

# ---- try the ACTUAL gamfit manifold SAE if available ----
gamfit_result="not available"
try:
    import gamfit
    fns=[x for x in dir(gamfit) if any(k in x.lower() for k in ("sae","manifold","spectral"))]
    gamfit_result=f"gamfit importable; sae fns: {fns}"
except Exception as e:
    gamfit_result=f"gamfit import failed: {type(e).__name__}: {str(e)[:80]}"
print("\nGAMFIT:", gamfit_result, flush=True)

# ---- plot EV vs coords/row: linear vs best atlas at each d ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ds=[1,2];
lin_pts=[lin[d]["ev"] for d in ds]
atl_pts=[max(a["ev"] for (K,dd),a in atlas.items() if dd==d) for d in ds]
fig,ax=plt.subplots(figsize=(6.5,4.5))
ax.plot(ds, lin_pts, "o-", label="linear (global PCA)", color="#c0392b", lw=2, ms=9)
ax.plot(ds, atl_pts, "s-", label="curved atlas (best K local charts)", color="#2471a3", lw=2, ms=9)
for d,l,a in zip(ds,lin_pts,atl_pts):
    ax.annotate(f"+{a-l:.3f}", (d,(a+l)/2), fontsize=9, color="#2471a3")
ax.set_xlabel("coordinates per row (sparse: 1 atom active)"); ax.set_ylabel("explained variance (sink-peeled L18)")
ax.set_title(f"Qwen3-8B L18: curved atlas vs linear, matched sparse capacity\nbest curved gain = {best['curved_gain']:+.3f} EV at d={best['d']}, K={best['K']}")
ax.set_xticks(ds); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/curved_vs_linear.png", dpi=140)
print(f"saved {OUT}/curved_vs_linear.png", flush=True)

summary=dict(layer="L18", N=N, n_peel=n_peel, linear=lin,
             atlas={f"K{K}_d{d}":a for (K,d),a in atlas.items()},
             head_to_head=compare, best=best, gamfit=gamfit_result)
json.dump(summary, open(f"{OUT}/curved_vs_linear.json","w"), indent=1)
print("\n==== RESULT ====", flush=True)
print(f"best curved gain over linear at matched sparse capacity: {best['curved_gain']:+.3f} EV "
      f"(atlas K={best['K']} d={best['d']}: {best['atlas_ev']:.3f} vs linear {best['linear_ev']:.3f})", flush=True)
