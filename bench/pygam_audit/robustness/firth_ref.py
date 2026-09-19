import numpy as np
from cases import CASES
c=CASES["logit_linear_sep_param"](); x=c["data"]["x"]; y=c["data"]["y"]
X=np.c_[np.ones_like(x),x]; b=np.zeros(2)
for it in range(500):
    eta=X@b; p=1/(1+np.exp(-eta)); W=p*(1-p)
    I=X.T@(W[:,None]*X); Ii=np.linalg.inv(I)
    h=W*np.einsum('ij,jk,ik->i',X,Ii,X)
    U=X.T@(y-p+h*(0.5-p))
    step=Ii@U
    # step halving on penalized loglik
    def pll(b):
        eta=X@b; p=1/(1+np.exp(-eta)); W=p*(1-p)
        return np.sum(y*eta-np.logaddexp(0,eta))+0.5*np.linalg.slogdet(X.T@(W[:,None]*X))[1]
    f0=pll(b); s=1.0
    while pll(b+s*step)<f0-1e-12 and s>1e-8: s/=2
    b=b+s*step
    if np.max(np.abs(s*step))<1e-10: break
print("firth ref coef",b,"iters",it)
print("firth pred at -0.5,0.5:",1/(1+np.exp(-(b[0]+b[1]*np.array([-.5,.5])))))
se=np.sqrt(np.diag(Ii)); print("se",se)
