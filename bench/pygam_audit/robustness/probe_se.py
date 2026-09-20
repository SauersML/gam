import gamfit, numpy as np, warnings
warnings.simplefilter("ignore")
from cases import CASES
from scipy.special import expit
for name in ["logit_linear_sep_param","extrapolate_logit","bool_y_logit"]:
    c=CASES[name](); m=gamfit.fit(c["data"],c["formula"],family="binomial")
    a=m.design_matrix(c["pred"]); G=np.asarray(a.eta_gradient); V=np.asarray(a.covariance_smoothing_corrected)
    eta=np.asarray(a.offset)+np.asarray(a.matrix)@np.asarray(a.coefficients)
    se_eta=np.sqrt(np.einsum('ij,jk,ik->i',G,V,G))
    r=m.predict(c["pred"],interval=0.95)
    rng=np.random.default_rng(0); d=rng.standard_normal((400000,len(eta)))*se_eta+eta
    p=expit(d)
    print(name,"eta",np.round(eta,2),"se_eta",np.round(se_eta,2))
    print("  MC E[p]",p.mean(0)," MC SD[p]",p.std(0))
    print("  reported posterior_mean",r["posterior_mean"]," reported SE",r["posterior_mean_standard_error"])
