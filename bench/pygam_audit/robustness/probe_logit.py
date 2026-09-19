import gamfit, numpy as np, warnings
warnings.simplefilter("ignore")
from cases import CASES
for name in ["extrapolate_logit","logit_linear_sep_param"]:
    c=CASES[name]()
    m=gamfit.fit(c["data"],c["formula"],family="binomial")
    r=m.predict(c["pred"],interval=0.95)
    for k,v in r.items(): print(name,k,v)
    s=m.summary(); print(s.coefficients_frame() if callable(getattr(s,'coefficients_frame',None)) else s.coefficients)
