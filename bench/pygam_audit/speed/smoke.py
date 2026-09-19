import time
t0=time.perf_counter()
import numpy as np, pandas as pd
t1=time.perf_counter()
import gamfit
t2=time.perf_counter()
rng=np.random.default_rng(0)
n=1000
x=rng.uniform(0,1,n); y=np.sin(2*np.pi*x)+rng.normal(0,0.3,n)
df={'x':x,'y':y}
t3=time.perf_counter()
m=gamfit.fit(df,"y ~ s(x)",family="gaussian")
t4=time.perf_counter()
m2=gamfit.fit(df,"y ~ s(x)",family="gaussian")
t5=time.perf_counter()
p=m.predict(df)
t6=time.perf_counter()
p=m.predict(df)
t7=time.perf_counter()
print(f"import np/pd {t1-t0:.3f} gamfit {t2-t1:.3f} fit1 {t4-t3:.3f} fit2 {t5-t4:.3f} pred1 {t6-t5:.3f} pred2 {t7-t6:.3f}")
print(type(p), getattr(p,'shape',None))
print(m.summary() if hasattr(m,'summary') else m)
