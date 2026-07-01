import numpy as np, time, torch, gamfit
from sklearn.datasets import load_digits
from dictionary_learning.trainers.top_k import AutoEncoderTopK
torch.manual_seed(0); torch.set_num_threads(3)
D=load_digits(); X=D.data.astype(np.float64)          # (1797,64) REAL handwritten digits
rng=np.random.default_rng(0); perm=rng.permutation(len(X)); X=X[perm]
ntr=1400; Xtr,Xte=X[:ntr],X[ntr:]
mu=Xtr.mean(0,keepdims=True); Xtr-=mu; Xte-=mu
p=X.shape[1]; K=128; kact=16                          # 2x overcomplete (K>p), RAM-safe
ev=lambda Xt,R: float(1-((R-Xt)**2).sum()/((Xt)**2).sum())
# external TopK SAE
Xtr_t=torch.tensor(Xtr,dtype=torch.float32); Xte_t=torch.tensor(Xte,dtype=torch.float32)
ae=AutoEncoderTopK(p,K,kact); opt=torch.optim.Adam(ae.parameters(),lr=1e-3); best=-9
for s in range(2000):
    idx=torch.randint(0,ntr,(256,)); x=Xtr_t[idx]
    loss=((ae(x)-x)**2).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if s%400==0:
        with torch.no_grad(): best=max(best,(1-((ae(Xte_t)-Xte_t)**2).sum()/((Xte_t)**2).sum()).item())
with torch.no_grad(): best=max(best,(1-((ae(Xte_t)-Xte_t)**2).sum()/((Xte_t)**2).sum()).item())
print(f"EXTERNAL TopK K={K} k={kact} p={p}: test_EV={best:.4f}")
# manifold with DIVERSE seed (K distinct random projection fibers) + known-good weights
N=ntr; R=rng.standard_normal((K,p)); R/=np.linalg.norm(R,axis=1,keepdims=True)
proj=Xtr@R.T; lo=proj.min(0,keepdims=True); hi=proj.max(0,keepdims=True)
t_init=((proj-lo)/np.maximum(hi-lo,1e-9)).T[:,:,None].copy()   # (K,N,1)
t0=time.time()
try:
    m=gamfit.sae_manifold_fit(X=Xtr, n_atoms=K, atom_topology="circle", d_atom=1,
        assignment="jumprelu", top_k=kact, isometry_weight=0.0, ard_per_atom=False,
        sparsity_weight=0.01, smoothness_weight=0.01, n_iter=8, random_state=0, t_init=t_init)
    Rte=np.asarray(m.predict(Xte))
    print(f"MANIFOLD(diverse-seed) K={K}: test_EV={ev(Xte,Rte):.4f} fit={time.time()-t0:.0f}s")
except Exception as e:
    print(f"MANIFOLD FAIL {time.time()-t0:.0f}s: {repr(e)[:200]}")
