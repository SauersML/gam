import gamfit, numpy as np, warnings, pygam
warnings.simplefilter("ignore")
from cases import CASES
for seed in [6,7,8]:
    r=np.random.default_rng(seed); x=r.uniform(0,1,400); y=np.sin(6*x)+0.3*r.standard_cauchy(400)
    px=np.linspace(0.02,0.98,200); t=np.sin(6*px)
    m=gamfit.fit({"x":x,"y":y},"y ~ s(x)"); pg=m.predict({"x":px}); pg=pg["posterior_mean"] if isinstance(pg,dict) else np.asarray(pg).ravel()
    g=pygam.LinearGAM(pygam.s(0)).fit(x[:,None],y); pp=g.predict(px[:,None])
    print(seed,"gamfit rmse %.3f edf %.2f"%(np.sqrt(np.mean((pg-t)**2)),m.summary().edf_total),"pygam rmse %.3f"%np.sqrt(np.mean((pp-t)**2)))
