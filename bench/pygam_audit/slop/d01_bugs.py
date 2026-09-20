# Small, deterministic pyGAM bug demos (reported statistics / distributions)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.stats as st
from pygam import LinearGAM, GammaGAM, PoissonGAM, LogisticGAM, s, l
from pygam.distributions import InvGaussDist, NormalDist, GammaDist
rng = np.random.default_rng(1)

print("== D-UBRE: (~add_scale) is -2, not False ==")
n=400; x=rng.uniform(0,1,n); y=rng.poisson(np.exp(1+np.sin(2*np.pi*x)))
g=PoissonGAM(s(0)).fit(x[:,None],y)
dev=g.distribution.deviance(y=y,mu=g.predict_mu(x[:,None]),scaled=False).sum()
edof=g.statistics_['edof']
print(" ~True =", ~True)
print(" reported UBRE      =", g.statistics_['UBRE'])
print(" dev/n + 2*1.4*edof/n (add_scale=True intent: + scale) =", dev/n + 2*1.4*edof/n + 1.0)
print(" dev/n + 2*scale + 2*1.4*edof/n (what code computes) =", dev/n + 2.0 + 2*1.4*edof/n)

print("== D-McFadden: reports ll_full/ll_null, i.e. 1 - R2_McFadden ==")
yb=rng.binomial(1, 1/(1+np.exp(-4*(x-.5))))
gb=LogisticGAM(s(0)).fit(x[:,None],yb)
ll=gb.statistics_['loglikelihood']; ll0=st.binom.logpmf(yb,1,yb.mean()).sum()
print(" reported McFadden =", gb.statistics_['pseudo_r2']['McFadden'], " correct 1-ll/ll0 =", 1-ll/ll0)

print("== D-GammaScale: stores sqrt(phi) as Gamma dispersion ==")
xg=rng.uniform(0,1,3000); mu=np.exp(1+np.sin(2*np.pi*xg)); k=2.0  # phi = 1/k = 0.5
yg=rng.gamma(shape=k, scale=mu/k)
gg=GammaGAM(s(0)).fit(xg[:,None],yg)
print(" true phi = 0.5 ; pyGAM statistics_['scale'] =", gg.statistics_['scale'])
draws=gg.distribution.sample(np.full(200000,2.0))
print(" sample() at mu=2: var =",draws.var()," (true var mu^2*phi = 2.0; pyGAM implies mu^2*sqrt(phi)=",4*gg.statistics_['scale'],")")
ll_py = gg.statistics_['loglikelihood']
mfit=gg.predict_mu(xg[:,None])
ll_true = st.gamma.logpdf(yg, a=k, scale=mfit/k).sum()
print(" pyGAM loglik =",ll_py, " loglik at same mu with correct phi=0.5:", ll_true)

print("== D-InvGauss: log_pdf has mean mu*phi, not mu ==")
d=InvGaussDist(scale=0.25); m0=3.0; gam_=1/d.scale
print(" scipy invgauss(mu, scale=1/gamma).mean() =", st.invgauss.mean(m0, scale=1/gam_), " (should be", m0,")")

print("== D-NormalWeights: log_pdf uses sd/w (var/w^2) instead of sd/sqrt(w) ==")
dn=NormalDist(scale=1.0)
print(" pyGAM logpdf(y=1,mu=0,w=4) =", dn.log_pdf(np.array([1.]),np.array([0.]),weights=np.array([4.]))[0],
      " correct N(0, 1/4) logpdf =", st.norm.logpdf(1,0,0.5))

print("== D-BinomialWeights: log_pdf ignores weights ==")
from pygam.distributions import BinomialDist
db=BinomialDist()
print(" w=1:",db.log_pdf(np.array([1.]),np.array([.3]),np.array([1.]))[0]," w=5:",db.log_pdf(np.array([1.]),np.array([.3]),np.array([5.]))[0])

print("== D-GCVgamma: statistics_['GCV'] uses a hard-coded gamma=1.4 ==")
yl=np.sin(2*np.pi*x)+rng.normal(0,.3,n)
gl=LinearGAM(s(0)).fit(x[:,None], yl)
D=((yl-gl.predict(x[:,None]))**2).sum(); ed=gl.statistics_['edof']
print(" stored GCV=",gl.statistics_['GCV']," n*D/(n-1.4*edof)^2=",n*D/(n-1.4*ed)**2," n*D/(n-edof)^2=",n*D/(n-ed)**2)
