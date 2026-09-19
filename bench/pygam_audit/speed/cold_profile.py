import time, sys
import numpy as np
n=int(sys.argv[1]) if len(sys.argv)>1 else 1000
reps=int(sys.argv[2]) if len(sys.argv)>2 else 1
rng=np.random.default_rng(0)
x=rng.uniform(0,1,n); y=np.sin(2*np.pi*x)+rng.normal(0,0.3,n)
import gamfit
for r in range(reps):
    t=time.perf_counter()
    m=gamfit.fit({'x':x,'y':y},"y ~ s(x)",family="gaussian")
    print(f"fit{r} {time.perf_counter()-t:.4f}", file=sys.stderr)
