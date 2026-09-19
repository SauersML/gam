import os; os.environ["RAYON_NUM_THREADS"]="1"
import numpy as np, gamfit, warnings
rng=np.random.default_rng(1); n=500; x=rng.uniform(0,1,n); mu=np.sin(2*np.pi*6*x); y=mu+rng.normal(0,.3*np.std(mu)+.1,n)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m=gamfit.fit({"x":x,"y":y},"y ~ s(x)",family="gaussian")
    print("WARNINGS:",[str(i.message)[:300] for i in w])
print(m.basis_check({"x":x,"y":y}))
print(m.summary().basis_checks)
