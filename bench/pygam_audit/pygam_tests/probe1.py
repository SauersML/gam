import numpy as np, time, gamfit
import pygam.datasets.load_datasets as L; L.PATH='data'
from pygam.datasets import mcycle
X,y=mcycle(return_X_y=True)
d={"x":X[:,0].astype(float),"y":np.asarray(y,float)}
t=time.time(); m=gamfit.fit(d,"y ~ s(x)"); print("fit", time.time()-t)
p=m.predict(d); print(type(p), np.shape(p))
s=m.summary(); print(type(s)); print(s)
print([ (b.name, b) for b in m.term_blocks])
pd_=m.partial_dependence("s(x)"); print({k:(np.shape(v) if hasattr(v,'shape') else v) for k,v in pd_.items()})
r=m.predict(d, interval=0.95); print(type(r), list(r.keys()) if hasattr(r,'keys') else None)
print(dir(s))
