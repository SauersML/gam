import gamfit, numpy as np, warnings, time
warnings.simplefilter("ignore")
from cases import CASES
for name, form, kw in [("logit_perfect_sep","y ~ s(x)",dict(firth=True)),
                       ("logit_linear_sep_param","y ~ x",dict(firth=True)),
                       ("logit_linear_sep_param","y ~ x",dict()),
                       ("logit_linear_sep_param","y ~ s(x)",dict())]:
    c=CASES[name](); t=time.time()
    try:
        m=gamfit.fit(c["data"],form,family="binomial",**kw)
        s=m.summary()
        r=m.predict(c["pred"],interval=0.95)
        print(name,form,kw,"OK t=%.1f"%(time.time()-t),"coef",[round(v,2) for v in np.asarray(s.coefficients_frame()["estimate"])[:3]],"conv",s.convergence.get("certified"),s.convergence.get("inner_status"),"pm",np.round(r["posterior_mean"],4),"plug",np.round(r["mean_plugin"],4))
    except Exception as e:
        print(name,form,kw,"ERR t=%.1f"%(time.time()-t),type(e).__name__,str(e)[:300])
