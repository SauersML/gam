"""Independent LAML surface for coal fold 0 (diagnostic only, not production).
Cubic B-spline, k=12 (8 interior knots), exact int f''^2 penalty (numerical quadrature),
null-space ridge (double penalty), sum-to-zero, intercept. Poisson log link."""
import os; os.environ["RAYON_NUM_THREADS"]="1"
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.interpolate import BSpline
import pygam.datasets.load_datasets as L; L.PATH=os.environ.get("PYGAM_DATA_DIR", os.path.expanduser("~/.cache/gamfit-bench/pygam_data"))
from sklearn.model_selection import KFold
import gamfit
X,y=L.coal(); x=X[:,0]; y=np.asarray(y,float)
tr,te=next(iter(KFold(5,shuffle=True,random_state=0).split(x)))
xt,yt=x[tr],y[tr]
a,b=xt.min(),xt.max(); nk=8; deg=3
inner=np.quantile(xt,np.linspace(0,1,nk+2)[1:-1])
t=np.r_[[a]*(deg+1),inner,[b]*(deg+1)]; K=len(t)-deg-1
def B(xx,d=0):
    out=np.zeros((len(xx),K))
    for j in range(K):
        c=np.zeros(K); c[j]=1; s=BSpline(t,c,deg,extrapolate=True)
        out[:,j]=s.derivative(d)(xx) if d else s(xx)
    return out
xq=np.linspace(a,b,4001); B2=B(xq,2); w=np.full(len(xq),(b-a)/4000); w[[0,-1]]/=2
S1=B2.T@(B2*w[:,None])
Bt=B(xt); cm=Bt.mean(0)
Q,_=np.linalg.qr(cm[:,None],mode='complete'); Z=Q[:,1:]   # sum-to-zero
Xd=np.c_[np.ones(len(xt)),Bt@Z]; S1z=Z.T@S1@Z
ev,U=np.linalg.eigh(S1z); null=U[:,ev<ev.max()*1e-9]; S2z=null@null.T
p=Xd.shape[1]
def fitb(l1,l2,beta=None):
    S=np.zeros((p,p)); S[1:,1:]=l1*S1z+l2*S2z
    beta=np.zeros(p) if beta is None else beta.copy(); beta[0]=np.log(yt.mean()) if beta is None or not beta.any() else beta[0]
    for it in range(200):
        eta=Xd@beta; mu=np.exp(eta)
        g=Xd.T@(yt-mu)-S@beta; H=Xd.T@(Xd*mu[:,None])+S
        step=np.linalg.solve(H,g); beta+=step
        if np.max(np.abs(step))<1e-10: break
    eta=Xd@beta; mu=np.exp(eta); H=Xd.T@(Xd*mu[:,None])+S
    ll=np.sum(yt*eta-mu)
    Sp=l1*S1z+l2*S2z; lds=np.linalg.slogdet(Sp)[1]
    V=-ll+0.5*beta@S@beta+0.5*np.linalg.slogdet(H)[1]-0.5*lds
    edf=np.trace(np.linalg.solve(H,Xd.T@(Xd*mu[:,None])))
    return V,edf,beta
best=None
for r1 in np.linspace(-6,14,41):
    for r2 in np.linspace(-6,14,21):
        V,edf,_=fitb(np.exp(r1),np.exp(r2))
        if best is None or V<best[0]: best=(V,r1,r2,edf)
print("grid LAML min: V=%.4f rho1=%.2f rho2=%.2f edf=%.2f"%best)
# profile along rho1 with rho2 minimised
for r1 in np.linspace(-6,14,21):
    vals=[fitb(np.exp(r1),np.exp(r2))[:2] for r2 in np.linspace(-6,14,21)]
    V,edf=min(vals); print(" rho1=%5.1f  min_rho2 V=%.4f edf=%.2f"%(r1,V,edf))
m=gamfit.fit({"x":xt,"y":yt},"y ~ s(x)",family="poisson")
print("gamfit: edf",m.summary().edf_total,"lambdas",m.smoothing_parameters(),"reml",m.summary().reml_score)
