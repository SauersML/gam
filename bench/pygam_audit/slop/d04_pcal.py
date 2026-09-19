# Null calibration of smooth-term p-values: y depends on x only; test s(z). Fraction p<0.05 / p<0.01 should be ~0.05 / 0.01.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gamfit, time, sys, io, contextlib
from pygam import LinearGAM, s
R=int(sys.argv[1]) if len(sys.argv)>1 else 200
n=200
py_def=[]; py_gs=[]; gf=[]; gfu=[]; t0=time.time()
for r in range(R):
    rng=np.random.default_rng(1000+r)
    x=rng.uniform(0,1,n); z=rng.uniform(0,1,n); y=np.sin(2*np.pi*x)+rng.normal(0,.5,n)
    X=np.c_[x,z]
    g=LinearGAM(s(0)+s(1)).fit(X,y); py_def.append(g.statistics_['p_values'][1])
    with contextlib.redirect_stdout(io.StringIO()):
        g2=LinearGAM(s(0)+s(1)).gridsearch(X,y,progress=False)
    py_gs.append(g2.statistics_['p_values'][1])
    d={"x":x,"z":z,"y":y}
    with contextlib.redirect_stdout(io.StringIO()):
        m=gamfit.fit(d,"y ~ s(x) + s(z)")
        ss=m.smooth_significance(d)
    gf.append(ss[1]['p_value_corrected']); gfu.append(ss[1]['p_value_conditional'])
    if (r+1)%5==0:
        print("== after",r+1,"reps, secs",round(time.time()-t0),flush=True)
        for name,p in [("pyGAM default lam=0.6",py_def),("pyGAM gridsearch(GCV)",py_gs),("gamfit p_value_corrected",gf),("gamfit p_value_conditional",gfu)]:
            p=np.array(p); print(f" {name:28s} P(p<.05)={np.mean(p<.05):.3f}  P(p<.01)={np.mean(p<.01):.3f}  median p={np.median(p):.3f}",flush=True)
for name,p in [("pyGAM default lam=0.6",py_def),("pyGAM gridsearch(GCV)",py_gs),("gamfit p_value_corrected",gf),("gamfit p_value_conditional",gfu)]:
    p=np.array(p); print(f" {name:28s} P(p<.05)={np.mean(p<.05):.3f}  P(p<.01)={np.mean(p<.01):.3f}  median p={np.median(p):.3f}")
print(" reps",R," secs",time.time()-t0)
