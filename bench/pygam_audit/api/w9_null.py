import numpy as np, warnings, time
warnings.simplefilter("ignore")
import gamfit
from pygam import LinearGAM, s
fails=0; pf=0
for seed in range(12):
    rng = np.random.default_rng(seed); X = rng.normal(size=(200,2)); y = rng.normal(size=200)
    t=time.time()
    try:
        m = gamfit.fit({"x0":X[:,0],"x1":X[:,1],"y":y}, "y ~ s(x0)+s(x1)"); r="OK edf=%.2f"%m.summary().edf_total
    except Exception as e: r="FAIL "+type(e).__name__+": "+str(e)[:90]; fails+=1
    try: g=LinearGAM(s(0)+s(1)).fit(X,y); pr="OK edf=%.2f"%g.statistics_['edof']
    except Exception as e: pr="FAIL "+str(e)[:60]; pf+=1
    print(seed, "%.1fs"%(time.time()-t), "gamfit:", r, "| pygam:", pr)
print("gamfit fails", fails, "/12; pygam fails", pf)
