# (e) default gridsearch uses ONE lam for all terms; (f) boundary hits of the 11-point grid; compare gamfit per-term REML.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gamfit, io, contextlib, sys
from pygam import LinearGAM, s
R=int(sys.argv[1]) if len(sys.argv)>1 else 20
n=400; res=[]
for r in range(R):
    rng=np.random.default_rng(500+r)
    x=rng.uniform(0,1,n); z=rng.uniform(0,1,n); w=rng.uniform(0,1,n)
    fx=np.sin(6*np.pi*x); fz=0.5*z-0.25; fw=0*w
    y=fx+fz+rng.normal(0,.4,n); X=np.c_[x,z,w]
    g=LinearGAM(s(0)+s(1)+s(2)).gridsearch(X,y,progress=False)
    lam=np.array(g.lam).ravel()
    grid=np.linspace(0,1,200); Xg=np.c_[grid,grid,grid]
    pz=g.partial_dependence(term=1,X=Xg); pz-=pz.mean()
    px=g.partial_dependence(term=0,X=Xg); px-=px.mean()
    tx=np.sin(6*np.pi*grid); tx-=tx.mean(); tz=0.5*grid; tz-=tz.mean()
    py_mse=np.mean((g.predict(X)-(fx+fz))**2)
    d={"x":x,"z":z,"w":w,"y":y}
    with contextlib.redirect_stdout(io.StringIO()):
        m=gamfit.fit(d,"y ~ s(x) + s(z) + s(w)")
        mu=np.asarray(m.predict(d)).ravel()
    gf_mse=np.mean((mu-(fx+fz))**2)
    edf=g.statistics_['edof_per_coef']
    res.append((lam[0],lam[1],lam[2],py_mse,gf_mse,g.statistics_['edof']))
res=np.array(res)
print(" pyGAM gridsearch lams per term identical in all reps:", np.all(res[:,0]==res[:,1]) and np.all(res[:,1]==res[:,2]))
print(" distinct chosen lam values:",np.unique(res[:,0]))
print(" mean in-sample MSE vs truth: pyGAM gridsearch = %.4f   gamfit REML = %.4f  (ratio %.2f)"%(res[:,3].mean(),res[:,4].mean(),res[:,3].mean()/res[:,4].mean()))
print(" gamfit better in %d/%d reps"%((res[:,4]<res[:,3]).sum(),R))
