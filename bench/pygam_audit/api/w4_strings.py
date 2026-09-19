import numpy as np, warnings, pandas as pd
warnings.simplefilter("always")
rng=np.random.default_rng(0)
n=400
X=np.column_stack([rng.uniform(0,1,n), rng.uniform(-2,2,n), rng.integers(0,4,n)])
y=np.sin(2*np.pi*X[:,0]) + 0.5*X[:,1]**2 + rng.normal(0,0.3,n)
from gamfit.sklearn import GAMRegressor
import gamfit
Xs=X.astype(object); Xs[0,1]="abc"
r=GAMRegressor(formula="s(x0)+s(x1)+factor(x2)").fit(Xs,y)
print(r.summary())
print(r.model_.summary().smooth_terms_frame())
print("coefs", len(r.model_.summary().coefficients))
try: print("predict numeric x1:", r.predict(X[:3]))
except Exception as e: print("predict numeric:", type(e).__name__, e)
# DataFrame version
df=pd.DataFrame(X,columns=["a","b","c"]); df["b"]=df["b"].astype(object); df.loc[0,"b"]="abc"
try:
    m=gamfit.fit(df.assign(y=y),"y ~ s(a)+s(b)+factor(c)"); print("DF str in numeric: OK", m.summary().smooth_terms_frame())
except Exception as e: print("DF:",type(e).__name__, e)
# a "numeric string" column e.g. read_csv mistakes: '1.5'
df2=pd.DataFrame(X,columns=["a","b","c"]); df2["b"]=df2["b"].map(lambda v:f"{v:.3f}")
try:
    m=gamfit.fit(df2.assign(y=y),"y ~ s(a)+s(b)+factor(c)"); print("DF numeric-as-string: OK", m.summary().smooth_terms_frame())
except Exception as e: print("DF numeric-as-string:",type(e).__name__, e)
# s() over a pure string column
df3=pd.DataFrame({"a":X[:,0],"g":rng.choice(list("pqrstuvwxyz"),n),"y":y})
try:
    m=gamfit.fit(df3,"y ~ s(a)+s(g)"); print("s() on string column: OK (silently)", m.summary().smooth_terms_frame().to_string())
except Exception as e: print("s() on string col:",type(e).__name__, e)
