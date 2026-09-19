# pyGAM sample(): (1) lam 'random search' is lognormal(-3,6), not [1e-3,1e3]; (2) incumbent's ORIGINAL-data score competes
# with bootstrap-data scores, so the 'bootstrap' lam is frequently just the original lam.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, io, contextlib
from pygam import LinearGAM, s
np.random.seed(0)
g=np.exp(np.random.randn(100000)*6-3)
print(" lam draws: frac<1e-3=%.3f frac>1e3=%.3f  2.5%%=%.2e 97.5%%=%.2e"%((g<1e-3).mean(),(g>1e3).mean(),*np.quantile(g,[.025,.975])))
rng=np.random.default_rng(0); n=300
x=rng.uniform(0,1,n); y=np.sin(2*np.pi*x)+rng.normal(0,.3,n)
gam=LinearGAM(s(0)).gridsearch(x[:,None],y,progress=False)
lam0=np.array(gam.lam).ravel().copy()
with contextlib.redirect_stdout(io.StringIO()):
    coefs,covs=gam._bootstrap_samples_of_smoothing(x[:,None],y,n_bootstraps=41)
same=[np.allclose(c,coefs[0]) for c in coefs[1:]]
print(" original lam:",lam0," bootstrap replicates whose coef == original fit (lam unchanged): %d/%d"%(sum(same),len(same)))
