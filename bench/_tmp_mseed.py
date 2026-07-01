import numpy as np, time, sys, gamfit
X=np.load("/tmp/real_acts48.npy").astype(np.float64); ntr=6000
Xtr,Xte=X[:ntr],X[ntr:]
K=int(sys.argv[1]); ni=int(sys.argv[2]); nsub=int(sys.argv[3])
asg=sys.argv[4] if len(sys.argv)>4 else "jumprelu"
Xs=Xtr[:nsub]; N,p=Xs.shape
ev=lambda Xt,R: float(1-((R-Xt)**2).sum()/((Xt)**2).sum())
# DIVERSE GENERIC SEED: K distinct random projection directions -> K distinct
# 1-D coordinate fibers, normalized to the periodic domain [0,1). Generic, not a
# search; gives K distinct atoms for ANY K (breaks the K>>p PCA duplicate wall).
rng=np.random.default_rng(0)
R=rng.standard_normal((K,p)); R/=np.linalg.norm(R,axis=1,keepdims=True)
proj=Xs@R.T                       # (N,K)
lo=proj.min(0,keepdims=True); hi=proj.max(0,keepdims=True)
coords=((proj-lo)/np.maximum(hi-lo,1e-9)).T   # (K,N) in [0,1)
t_init=coords[:,:,None].copy()               # (K,N,1)
t0=time.time()
try:
    m=gamfit.sae_manifold_fit(X=Xs, n_atoms=K, atom_topology="circle", d_atom=1,
        assignment=asg, top_k=32, isometry_weight=0.0, ard_per_atom=False,
        sparsity_weight=0.01, smoothness_weight=0.01, n_iter=ni, random_state=0,
        t_init=t_init)
    dt=time.time()-t0
    Rte=np.asarray(m.predict(Xte))
    print(f"MSEED K={K} n={nsub} it={ni} {asg}: test_EV={ev(Xte,Rte):.4f} fit={dt:.0f}s")
except Exception as e:
    print(f"MSEED K={K} {asg} FAIL {time.time()-t0:.0f}s: {repr(e)[:280]}")
