import numpy as np, pandas as pd, gamfit, traceback, warnings
warnings.simplefilter("ignore")
rng=np.random.default_rng(0); n=300
df=pd.DataFrame({"a":rng.uniform(size=n),"g":rng.choice(["u","v","w"],n)}); df["y"]=np.sin(6*df.a)+(df.g=="v")+rng.normal(0,.3,n)
try: gamfit.fit(df,"y ~ s(a)+g")
except Exception: traceback.print_exc()
df["g"]=df.g.astype("category")
try: m=gamfit.fit(df,"y ~ s(a)+g"); print("category dtype OK", m.summary().coefficients_frame().head(6).to_string())
except Exception as e: print("category:", type(e).__name__, e)
