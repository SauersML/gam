import numpy as np, pandas as pd, gamfit, traceback
r=np.random.default_rng(0); centers=r.normal(0,5,size=(4,3)); rows=[]
for c in range(4):
    P=centers[c]+r.normal(0,0.5,size=(200,3)); z=r.normal(size=200)
    eta=0.4*z+0.3*P[:,0]-0.2*P[:,1]; y=(r.uniform(size=200)<1/(1+np.exp(-eta))).astype(int)
    rows.append(pd.DataFrame({"y":y,"PGS_z":z,"PC1":P[:,0],"PC2":P[:,1],"PC3":P[:,2]}))
d=pd.concat(rows,ignore_index=True); s="matern(PC1, PC2, PC3, centers=12)"
try:
    gamfit.fit(d,"y ~ "+s,family="bernoulli-marginal-slope",link="probit",z_column="PGS_z",logslope_formula=s)
    print("CONVERGED")
except Exception as e:
    print("FULL_ERROR:", str(e))
