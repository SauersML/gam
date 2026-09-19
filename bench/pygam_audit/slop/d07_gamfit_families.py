# gamfit counterparts for pyGAM family bugs: Gamma dispersion, Poisson exposure via offset, binomial/weights, McFadden-type stats.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gamfit, io, contextlib
from pygam import PoissonGAM, s
rng=np.random.default_rng(1)
q=lambda f: (lambda *a,**k: (lambda b: (b, f(*a,**k)))(None))
def fit(*a,**k):
    with contextlib.redirect_stdout(io.StringIO()):
        return gamfit.fit(*a,**k)
# Gamma phi=0.5
xg=rng.uniform(0,1,3000); mu=np.exp(1+np.sin(2*np.pi*xg)); yg=rng.gamma(shape=2.0, scale=mu/2.0)
m=fit({"x":xg,"y":yg},"y ~ s(x)",family="gamma")
sm=m.summary(); print(" gamfit Gamma summary:"); print(str(sm)[:600])
for a in ["scale","dispersion","phi"]:
    if hasattr(m,a): print(" m.%s ="%a, getattr(m,a))
# Poisson with exposure: truth rate exp(1+sin), exposure E in [0.5, 5]
n=2000; x=rng.uniform(0,1,n); E=rng.uniform(0.5,5,n); lam=np.exp(0.2+np.sin(2*np.pi*x)); y=rng.poisson(E*lam)
mp=fit({"x":x,"y":y,"logE":np.log(E)},"y ~ s(x)",family="poisson",offset="logE")
grid=np.linspace(0.02,.98,200); tr=np.exp(0.2+np.sin(2*np.pi*grid))
pg=np.asarray(mp.predict({"x":grid,"logE":np.zeros_like(grid)})).ravel()
print(" gamfit Poisson+offset rate RMSE:",np.sqrt(np.mean((pg-tr)**2)))
gp=PoissonGAM(s(0)).fit(x[:,None],y,exposure=E)
pp=gp.predict(grid[:,None])  # rate
print(" pyGAM Poisson exposure (y/E, w=E) rate RMSE:",np.sqrt(np.mean((pp-tr)**2)))
print(" pyGAM loglik with exposure:",gp.statistics_['loglikelihood'])
import scipy.stats as st
print(" true-model loglik at pyGAM mu*E:",st.poisson.logpmf(y,gp.predict_mu(x[:,None])*E).sum())
