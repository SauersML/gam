import numpy as np, pandas as pd, gamfit, time
r=np.random.default_rng(0); n=600; z=r.normal(size=n); P=r.normal(size=(n,3))
eta=0.4*z+0.3*P[:,0]-0.2*P[:,1]
T=r.exponential(1/(0.05*np.exp(np.clip(eta,-10,10)))); C=r.uniform(0,20,n)
d=pd.DataFrame({'z':z,'PC1':P[:,0],'PC2':P[:,1],'PC3':P[:,2],'sex':r.integers(0,2,n).astype(float),'t0':0.0,'t1':np.maximum(np.minimum(T,C),1e-3),'event':(T<=C).astype(int)})
s='matern(PC1, PC2, PC3, centers=12)'
try:
    m=gamfit.fit(d,'Surv(t0, t1, event) ~ '+s+' + sex',z_column='z',logslope_formula=s,survival_likelihood='marginal-slope'); print('788_RESULT: OK')
except Exception as e: print(f'788_RESULT: FAIL {type(e).__name__}: {str(e)[:80]}')
