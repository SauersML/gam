# Soft-penalty monotone constraint: violation magnitude scales with data scale / n (penalty 1e9 is not a hard bound)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from pygam import LinearGAM, s
rng=np.random.default_rng(3)
for n,scale in [(300,1),(300,1e4),(20000,1e4),(20000,1e6)]:
    xm=rng.uniform(0,1,n); ym=scale*(np.cos(np.pi*xm)+rng.normal(0,.05,n))  # decreasing truth
    g=LinearGAM(s(0,constraints='monotonic_inc')).fit(xm[:,None],ym)
    grid=np.linspace(0,1,2001); p=g.predict(grid[:,None])
    drop=(np.maximum.accumulate(p)-p).max()
    print(f" n={n:6d} y-scale={scale:8.0e}: max drop of 'increasing' fit = {drop:.3e}  ({drop/scale:.2e} in y-units/scale)")
