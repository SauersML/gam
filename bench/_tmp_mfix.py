import numpy as np, time, sys, gamfit
X=np.load("/tmp/real_acts48.npy").astype(np.float64); ntr=6000
Xtr,Xte=X[:ntr],X[ntr:]
K=int(sys.argv[1]); ni=int(sys.argv[2]); nsub=int(sys.argv[3])
asg=sys.argv[4] if len(sys.argv)>4 else "jumprelu"
Xs=Xtr[:nsub]
ev=lambda Xt,R: float(1-((R-Xt)**2).sum()/((Xt)**2).sum())
t0=time.time()
try:
    m=gamfit.sae_manifold_fit(X=Xs, n_atoms=K, atom_topology="circle", d_atom=1,
        assignment=asg, top_k=32, isometry_weight=0.0, ard_per_atom=False,
        sparsity_weight=0.01, smoothness_weight=0.01, n_iter=ni, random_state=0)
    dt=time.time()-t0
    Rte=np.asarray(m.predict(Xte))
    print(f"MFIX K={K} n={nsub} it={ni} {asg}: test_EV={ev(Xte,Rte):.4f} fit={dt:.0f}s")
except Exception as e:
    print(f"MFIX K={K} {asg} FAIL {time.time()-t0:.0f}s: {repr(e)[:280]}")
