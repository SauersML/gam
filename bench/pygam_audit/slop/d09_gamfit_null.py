# Null recovery: pure noise / linear truth / wiggly truth; pyGAM default & gridsearch vs gamfit default. Reports per-term edf.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gamfit, io, contextlib
from pygam import LinearGAM, s
rng=np.random.default_rng(2); n=500
for name,f in [("noise",lambda x:0*x),("linear",lambda x:2*x),("sin(8pi x)",lambda x:np.sin(8*np.pi*x))]:
    x=rng.uniform(0,1,n); y=f(x)+rng.normal(0,.3,n)
    g0=LinearGAM(s(0)).fit(x[:,None],y)
    with contextlib.redirect_stdout(io.StringIO()):
        g1=LinearGAM(s(0)).gridsearch(x[:,None],y,progress=False)
        m=gamfit.fit({"x":x,"y":y},"y ~ s(x)")
        st=m.summary().smooth_terms
    mu=np.asarray(m.predict({"x":x})).ravel()
    r=lambda p: np.sqrt(np.mean((p-f(x))**2))
    print(f" {name:11s} pyGAM default edof={g0.statistics_['edof']-1:5.2f} rmse={r(g0.predict(x[:,None])):.4f} | "
          f"gridsearch lam={np.ravel(g1.lam)[0]:8.3g} edof={g1.statistics_['edof']-1:5.2f} rmse={r(g1.predict(x[:,None])):.4f} | "
          f"gamfit smooth edf={st[0]['edf'] if isinstance(st[0],dict) else st[0]} rmse={r(mu):.4f}")
