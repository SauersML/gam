import os; os.environ["RAYON_NUM_THREADS"]="1"
import numpy as np, gamfit, warnings; warnings.filterwarnings("ignore")
d=dict(np.load("wage_fail_fold2.npz"))
try: gamfit.fit({"year":d["year"],"y":d["y"]},"y ~ s(year)",family="gaussian")
except Exception as e: print("wage fold2 s(year):",type(e).__name__, e)
# synthetic: 7-level discrete covariate, nearly flat effect
fails=0
for seed in range(20):
    rng=np.random.default_rng(seed); n=2400
    x=rng.integers(2003,2010,n).astype(float); y=100+0.8*(x-2003)+rng.normal(0,40,n)
    try: gamfit.fit({"x":x,"y":y},"y ~ s(x)",family="gaussian")
    except Exception as e: fails+=1; last=str(e)
print("synthetic 7-level linear-trend s(x) failures:",fails,"/20")
for seed in range(20):
    rng=np.random.default_rng(seed); n=2400
    x=rng.integers(0,7,n).astype(float); y=0.8*x+rng.normal(0,40,n)
    try: gamfit.fit({"x":x,"y":y},"y ~ s(x)",family="gaussian")
    except Exception as e: fails+=1
print("cumulative incl. shifted x:",fails)
