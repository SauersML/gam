import gamfit, numpy as np, warnings
warnings.simplefilter("ignore")
from cases import CASES
from scipy.special import expit
for name in ["logit_linear_sep_param","extrapolate_logit","bool_y_logit"]:
    c=CASES[name](); m=gamfit.fit(c["data"],c["formula"],family="binomial")
    a=m.design_matrix(c["pred"]); G=np.asarray(a.eta_gradient)
    eta=np.asarray(a.offset)+np.asarray(a.matrix)@np.asarray(a.coefficients)
    r=m.predict(c["pred"],interval=0.95)
    for lab in ["covariance_conditional","covariance_smoothing_corrected"]:
        V=np.asarray(getattr(a,lab))
        se=np.sqrt(np.einsum('ij,jk,ik->i',G,V,G))
        # Gauss-Hermite exact-ish
        x,w=np.polynomial.hermite_e.hermegauss(200); w=w/w.sum()
        P=expit(eta[:,None]+se[:,None]*x[None,:])
        Em=(P*w).sum(1); Sd=np.sqrt(np.maximum((P**2*w).sum(1)-Em**2,0))
        print(name,lab,"se_eta",np.round(se,3)); print("   GH E[p]",Em,"\n   GH SD[p]",Sd)
    print("   reported mean",r["posterior_mean"],"\n   reported SE",r["posterior_mean_standard_error"],"\n   plugin",r["mean_plugin"])
