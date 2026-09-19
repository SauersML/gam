from common import *
import time
rng=np.random.default_rng(102); x=rng.uniform(0,1,200); y=1/(1+np.exp(-30*(x-.5)))+rng.normal(0,.3,200)
t=time.time(); m=gamfit.fit(dict(x=x,y=y),'y ~ s(x, shape=monotone_increasing)'); print("fit %.1fs"%(time.time()-t), flush=True)
t=time.time(); p=m.predict(dict(x=np.linspace(0,1,5))); print("predict(no interval, 5 pts) %.1fs"%(time.time()-t), flush=True)
t=time.time(); out=m.predict(dict(x=np.linspace(0,1,5)), interval=0.95); print("predict(interval, 5 pts) %.1fs"%(time.time()-t), {k:np.round(np.asarray(v),3) for k,v in out.items() if k!='covariance_source'}, flush=True)
