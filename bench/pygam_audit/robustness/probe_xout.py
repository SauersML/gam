import gamfit, numpy as np, warnings, sys
warnings.simplefilter("ignore")
from cases import CASES
c=CASES["x_with_outlier_x"]()
for f in ["y ~ s(x)", "y ~ s(x, knot_placement=quantile)", "y ~ s(x, bs=tp)"]:
    try:
        m=gamfit.fit(c["data"],f); p=m.predict(c["pred"])
        p=p["posterior_mean"] if isinstance(p,dict) else np.asarray(p).ravel()
        print(f,"OK edf",round(m.summary().edf_total,2),"pred",np.round(p[:6],3),"truth",np.round(np.asarray(c["truth"])[:6],3))
    except Exception as e: print(f,"ERR",type(e).__name__,str(e)[:200])
