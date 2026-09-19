import os; os.environ["RAYON_NUM_THREADS"]="1"
import numpy as np, gamfit, warnings; warnings.filterwarnings("ignore")
from pygam import LinearGAM, s
def doppler(x): return np.sqrt(x*(1-x))*np.sin(2.1*np.pi/(x+0.05))*3
for name,fn in [("sin6",lambda x: np.sin(2*np.pi*6*x)),("doppler",doppler),("sin3",lambda x: np.sin(2*np.pi*3*x))]:
  for n in (500,5000):
    rng=np.random.default_rng(1); x=rng.uniform(0,1,n); mu=fn(x); y=mu+rng.normal(0,.3*np.std(mu)+.1,n)
    xg=np.linspace(0.001,.999,2000); tg=fn(xg)
    out=[]
    for k in (None,20,40,80):
        fml="y ~ s(x)" if k is None else f"y ~ s(x, k={k})"
        m=gamfit.fit({"x":x,"y":y},fml,family="gaussian")
        out.append((f"gamfit k={k or 'def'}", np.mean((m.predict({"x":xg})-tg)**2), m.summary().edf_total))
    g=LinearGAM(s(0)).gridsearch(x[:,None],y,progress=False); out.append(("pygam grid",np.mean((g.predict(xg[:,None])-tg)**2),g.statistics_['edof']))
    print(name,n," | ".join(f"{a}: mse={b:.4g} edf={c:.1f}" for a,b,c in out))
