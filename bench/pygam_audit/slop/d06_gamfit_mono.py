# gamfit hard monotone constraint: same decreasing-truth data as d03 (fit with monotone_increasing).
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gamfit, io, contextlib
rng=np.random.default_rng(3)
for n,scale in [(300,1),(20000,1e4)]:
    x=rng.uniform(0,1,n); y=scale*(np.cos(np.pi*x)+rng.normal(0,.05,n))
    d={"x":x,"y":y}
    with contextlib.redirect_stdout(io.StringIO()):
        m=gamfit.fit(d,"y ~ s(x)",constraints={"s(x)":"monotone_increasing"})
    grid=np.linspace(x.min(),x.max(),20001)
    p=np.asarray(m.predict({"x":grid})).ravel()
    drop=(np.maximum.accumulate(p)-p).max()
    print(f" gamfit n={n} scale={scale:g}: max drop of increasing fit = {drop:.3e} ({drop/scale:.2e} rel)")
