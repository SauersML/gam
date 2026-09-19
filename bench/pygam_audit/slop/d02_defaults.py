# pyGAM default-behaviour slop: fixed lam, scale-dependent ridge, soft monotone, non-convergence, identifiability
import warnings; warnings.filterwarnings("ignore")
import numpy as np, io, contextlib
from pygam import LinearGAM, LogisticGAM, s, l, f
rng=np.random.default_rng(2)
n=500; x=rng.uniform(0,1,n)

print("== A: default .fit() never selects lam (lam=0.6 fixed) ==")
for name,mu in [("pure noise",0*x),("wiggly sin(8pi x)",np.sin(8*np.pi*x))]:
    y=mu+rng.normal(0,.3,n)
    g=LinearGAM(s(0)).fit(x[:,None],y)
    print(f" {name:18s} lam={g.lam} edof={g.statistics_['edof']:.2f}")
y=np.sin(8*np.pi*x)+rng.normal(0,.3,n)
g=LinearGAM(s(0)).fit(x[:,None],y)
print(" wiggly: RMSE vs truth default =",np.sqrt(np.mean((g.predict(x[:,None])-np.sin(8*np.pi*x))**2)))

print("== B: l(0) ridge at lam=0.6 on raw scale -> not scale/unit invariant ==")
xl=rng.normal(0,1,n); yl=0.5*xl+rng.normal(0,1,n)
for c in [1,0.01,0.001]:
    g=LinearGAM(l(0)).fit((c*xl)[:,None],yl)
    slope_per_orig_unit=g.coef_[0]*c
    print(f" x scaled by {c:6}: slope (orig units) = {slope_per_orig_unit:.4f}  (OLS = {np.polyfit(xl,yl,1)[0]:.4f})")

print("== C: monotonic_inc is a soft penalty (1e9 * violating diffs), not a hard constraint ==")
xm=np.sort(rng.uniform(0,1,300)); ym=np.where(xm<.5,1.,0.)+rng.normal(0,.05,300)  # truth decreasing, fit inc
g=LinearGAM(s(0,constraints='monotonic_inc',n_splines=20)).fit(xm[:,None],ym)
grid=np.linspace(0,1,2001); p=g.predict(grid[:,None]); d=np.diff(p)
print(" min diff of fitted 'monotonic_inc' curve:",d.min(), " n decreasing steps:",(d<-1e-10).sum())
print(" coef diffs min:",np.diff(g.coef_[:20]).min())
ci=g.confidence_intervals(grid[:,None],width=.95); w=ci[:,1]-ci[:,0]
print(" CI width in constrained region: min=%.2e median=%.2e  (collapses because 1e9 penalty is treated as prior)"%(w.min(),np.median(w)))
gu=LinearGAM(s(0,n_splines=20)).fit(xm[:,None],ym); wu=np.diff(gu.confidence_intervals(grid[:,None],width=.95),axis=1)
print(" unconstrained CI width median=%.2e"%np.median(wu))

print("== H: non-convergence only prints; model still returned and usable ==")
yb=(x>.5).astype(float)  # perfect separation
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    gb=LogisticGAM(s(0),max_iter=3).fit(x[:,None],yb)
print(" stdout captured:",repr(buf.getvalue().strip()[:80]))
print(" returned model _is_fitted=",gb._is_fitted," statistics_ has AIC:",gb.statistics_['AIC'])

print("== I: no identifiability constraint: basis confounded with intercept ==")
g=LinearGAM(s(0)+s(1)).fit(np.c_[x,rng.uniform(0,1,n)], np.sin(2*np.pi*x)+rng.normal(0,.3,n))
X=g._modelmat(np.c_[x,rng.uniform(0,1,n)]).toarray()
print(" model matrix cols=",X.shape[1]," rank=",np.linalg.matrix_rank(X))
pd=g.partial_dependence(term=0,X=g.generate_X_grid(term=0))
print(" mean partial dependence of s(0) over grid =",pd.mean(),"(arbitrary level; not centred)")
print(" intercept coef =",g.coef_[-1])
