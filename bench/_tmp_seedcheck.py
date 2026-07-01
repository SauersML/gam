import numpy as np, time, gamfit, sys
from sklearn.datasets import load_digits
X=load_digits().data.astype(np.float64)
rng=np.random.default_rng(0); X=X[rng.permutation(len(X))][:800]
mu=X.mean(0,keepdims=True); X=X-mu; N,p=X.shape
K=128; use_seed = sys.argv[1]=="seed" if len(sys.argv)>1 else True
kw={}
if use_seed:
    R=rng.standard_normal((K,p)); R/=np.linalg.norm(R,axis=1,keepdims=True)
    proj=X@R.T; lo=proj.min(0,keepdims=True); hi=proj.max(0,keepdims=True)
    kw["t_init"]=((proj-lo)/np.maximum(hi-lo,1e-9)).T[:,:,None].copy()
print(f"start K={K} p={p} N={N} diverse_seed={use_seed}", flush=True)
t0=time.time()
try:
    m=gamfit.sae_manifold_fit(X=X, n_atoms=K, atom_topology="circle", d_atom=1,
        assignment="jumprelu", top_k=16, isometry_weight=0.0, ard_per_atom=False,
        sparsity_weight=0.01, smoothness_weight=0.01, n_iter=5, random_state=0, **kw)
    R2=np.asarray(m.reconstruct(X))
    ev=float(1-((R2-X)**2).sum()/((X)**2).sum())
    print(f"RESULT diverse_seed={use_seed}: train_EV={ev:.4f} fit={time.time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"RESULT diverse_seed={use_seed} FAIL {time.time()-t0:.0f}s: {repr(e)[:200]}", flush=True)
