from common import *
import time
rng=np.random.default_rng(102); x=rng.uniform(0,1,200); y=1/(1+np.exp(-30*(x-.5)))+rng.normal(0,.3,200)
for form in ['y ~ s(x)', 'y ~ s(x, shape=monotone_increasing)']:
    m=gamfit.fit(dict(x=x,y=y),form)
    for npts in [1, 20]:
        t=time.time(); out=m.predict(dict(x=np.linspace(0,1,npts)), interval=0.95); print(form, npts, "pts: %.1fs"%(time.time()-t), flush=True)
