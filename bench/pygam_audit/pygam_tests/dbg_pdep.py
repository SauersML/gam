import sys; sys.argv=['x']
from conftest import *
import conftest as c, numpy as np, gamfit
X,y=c._ds.mcycle(return_X_y=True); d={"x":X[:,0].astype(float),"y":np.asarray(y,float)}
m=gamfit.fit(d,"y ~ s(x)")
p=c.pdep(m,"s(x)",d,grid=d["x"])
r=m.predict(d, interval=0.95); r0=m.predict(d); print(type(r0), np.max(np.abs(r0-np.asarray(r["posterior_mean"]))), np.max(np.abs(r0-np.asarray(r["mean_plugin"]))))
print(list(r.keys()))
for k in ["linear_predictor_plugin","mean_plugin","posterior_mean"]:
    print(k, np.max(np.abs(np.asarray(r[k])-(c.intercept(m,d)+p["predicted"]))))
print(m.term_blocks)
dm=m.design_matrix(d); print(np.max(np.abs(np.asarray(dm.matrix)@np.asarray(dm.coefficients)+np.asarray(dm.offset)-np.asarray(r["linear_predictor_plugin"]))))
st=m._coefficient_state(); print(np.max(np.abs(np.asarray(st["beta"])-np.asarray(dm.coefficients))))
