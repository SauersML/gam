import numpy as np, time, gamfit, warnings
warnings.simplefilter("ignore")
import pygam.datasets.load_datasets as L; L.PATH='data'
from pygam.datasets import mcycle
X,y=mcycle(return_X_y=True)
d={"x":X[:,0].astype(float),"y":np.asarray(y,float)}
m=gamfit.fit(d,"y ~ s(x)")
pd_=m.partial_dependence("s(x)", d); print({k:(np.shape(v) if hasattr(v,'shape') else v) for k,v in pd_.items()})
pd2=m.partial_dependence("s(x)", d, grid=d["x"])
pred=m.predict(d)
st=m._coefficient_state(); print(st.keys())
b=[b for b in m.term_blocks if b.name=="intercept"][0]
print("intercept coef", st.get("beta", st.get("coefficients"))[:2] if isinstance(st,dict) else None)
dm=m.design_matrix(d); icpt=dm.coefficients[0]
print("pred - (icpt+pdep) max abs:", np.max(np.abs(pred-(icpt+pd2["predicted"]))))
r=m.predict(d, interval=0.95); print(type(r), list(r.keys()))
s=m.summary(); print([a for a in dir(s) if not a.startswith('_')])
