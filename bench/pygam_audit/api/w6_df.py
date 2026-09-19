import numpy as np, pandas as pd, gamfit, warnings, time
warnings.simplefilter("ignore")
def T(label, f):
    t=time.time()
    try:
        r=f(); print(f"[OK ] {label} ({time.time()-t:.2f}s):", repr(r)[:300]); return r
    except Exception as e:
        print(f"[ERR] {label}: {type(e).__name__}: {str(e)[:350]}")
rng=np.random.default_rng(0); n=400
df=pd.DataFrame({"age":rng.uniform(20,70,n),"bmi":rng.uniform(18,35,n),"region":rng.choice(["north","south","east"],n)})
df["y"]=np.sin(df.age/8)+0.05*df.bmi+(df.region=="south")*0.7+rng.normal(0,.3,n)
from gamfit.sklearn import GAMRegressor
m=T("fit DF", lambda: gamfit.fit(df,"y ~ s(age)+s(bmi)+region"))
T("predict DF returns", lambda: type(m.predict(df.head())))
T("predict DF", lambda: m.predict(df.head()))
T("predict DF interval", lambda: m.predict(df.head(2), interval=0.95))
T("predict reordered cols", lambda: m.predict(df[["region","bmi","age"]].head(2)))
T("predict missing col", lambda: m.predict(df[["age","bmi"]].head(2)))
T("predict unseen level", lambda: m.predict(pd.DataFrame({"age":[30.],"bmi":[25.],"region":["west"]})))
T("predict with NaN in region", lambda: m.predict(pd.DataFrame({"age":[30.],"bmi":[25.],"region":[None]})))
d2=df.copy(); d2["bmi"]=d2.bmi.astype(object); d2.loc[3,"bmi"]="n/a"
m2=T("fit DF with 'n/a' string inside numeric bmi, s(bmi)", lambda: gamfit.fit(d2,"y ~ s(age)+s(bmi)+region"))
if m2 is not None: T("  -> smooth table", lambda: m2.summary().smooth_terms_frame().to_dict("records"))
d3=df.copy(); d3["bmi"]=d3.bmi.map(lambda v:f"{v:.2f}")
m3=T("fit DF with numeric-as-string bmi, s(bmi)", lambda: gamfit.fit(d3,"y ~ s(age)+s(bmi)+region"))
if m3 is not None: T("  -> smooth table", lambda: m3.summary().smooth_terms_frame().to_dict("records"))
m4=T("fit s(region) (smooth of a string column)", lambda: gamfit.fit(df,"y ~ s(age)+s(region)"))
if m4 is not None: T("  -> smooth table", lambda: m4.summary().smooth_terms_frame().to_dict("records"))
T("fit typo column", lambda: gamfit.fit(df,"y ~ s(agee)+region"))
T("fit typo function", lambda: gamfit.fit(df,"y ~ sm(age)+region"))
T("fit missing tilde and no y", lambda: gamfit.fit(df,"s(age)+region"))
T("fit wrong family", lambda: gamfit.fit(df,"y ~ s(age)", family="gausian"))
T("fit binomial on continuous", lambda: gamfit.fit(df,"y ~ s(age)", family="binomial"))
T("fit n=5", lambda: gamfit.fit(df.head(5),"y ~ s(age)+s(bmi)"))
T("fit constant column", lambda: gamfit.fit(df.assign(c=1.0),"y ~ s(age)+s(c)"))
T("fit DF integer-named columns", lambda: gamfit.fit(pd.DataFrame(np.column_stack([df.age,df.bmi,df.y])),"2 ~ s(0)+s(1)"))
T("fit DF column with space", lambda: gamfit.fit(df.rename(columns={"age":"patient age"}),"y ~ s(`patient age`)"))
# sklearn wrapper with DF
r=T("GAMRegressor DF X, Series y", lambda: GAMRegressor(formula="s(age)+s(bmi)+region").fit(df[["age","bmi","region"]], df.y))
T("  feature_names_in_", lambda: r.feature_names_in_)
T("  predict DF", lambda: r.predict(df[["age","bmi","region"]].head(3)))
T("  predict numpy after DF fit", lambda: r.predict(df[["age","bmi","region"]].head(3).to_numpy()))
T("  score", lambda: r.score(df[["age","bmi","region"]], df.y))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
ct=ColumnTransformer([("num",StandardScaler(),["age","bmi"]),("cat",OneHotEncoder(sparse_output=False),["region"])])
T("Pipeline(ColumnTransformer->GAMRegressor) formula over x0..", lambda: make_pipeline(ct, GAMRegressor(formula="s(x0)+s(x1)+x2+x3+x4")).fit(df, df.y).score(df, df.y))
ct2=ColumnTransformer([("num",StandardScaler(),["age","bmi"]),("cat","passthrough",["region"])], verbose_feature_names_out=False).set_output(transform="pandas")
T("Pipeline(ColumnTransformer pandas output->GAMRegressor) named formula", lambda: make_pipeline(ct2, GAMRegressor(formula="s(age)+s(bmi)+region")).fit(df, df.y).score(df, df.y))
