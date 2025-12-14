## Running a polygenic score for obesity on the _All of Us_ cohort

Let's view the polygenic score catalog for obesity:
[https://www.pgscatalog.org/trait/MONDO_0001627/
](https://www.pgscatalog.org/trait/EFO_0001073/)

And BMI:
[https://www.pgscatalog.org/trait/MONDO_0004975/
](https://www.pgscatalog.org/trait/EFO_0004340/)

Strong performers include:
- PGS004378
- PGS003897
- PGS005199
- PGS005198
- PGS003400
- PGS005235
- PGS005203
- PGS004150

We can use the microarray `gs://fc-aou-datasets-controlled/v8/microarray/plink/` for scoring.

Create a standard cloud analysis environment with 8 CPUs and 30 GB RAM (though using lower or higher values should also work similarly).

Install gnomon if you haven't already:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && git clone https://github.com/SauersML/gnomon.git && cd gnomon && rustup override set nightly && cargo build --release && cd ~
```

Let's download the microarray data:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Now we can run the scores. It should be faster to run them all at once instead of one at a time.
```
./gnomon/target/release/gnomon score "PGS004378, PGS003897, PGS005199, PGS005198, PGS003400, PGS005235, PGS005203, PGS004150" arrays
```

This should take 23 minutes to run, and output a file called:
```
arrays.sscore
```
Let's open a Jupyter analysis notebook.
```
!head ../../arrays.sscore
```

This shows us the columns of the score output:
```
#IID	PGS003400_AVG	PGS003400_MISSING_PCT	PGS003897_AVG	PGS003897_MISSING_PCT	PGS004150_AVG	PGS004150_MISSING_PCT	PGS004378_AVG	PGS004378_MISSING_PCT	PGS005198_AVG	PGS005198_MISSING_PCT	PGS005199_AVG	PGS005199_MISSING_PCT	PGS005203_AVG	PGS005203_MISSING_PCT	PGS005235_AVG	PGS005235_MISSING_PCT
```

Let's plot missingness for each score:
```
import pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv('../../arrays.sscore', sep='\t')
for c in d.columns[d.columns.str.endswith('_MISSING_PCT')]: d[c].hist(bins=50); plt.title(c); plt.xlabel(c); plt.ylabel('count'); plt.show()
```

Most samples' missingness should be under 2%.

Now let's check the correlation matrix of the scores. We expect moderate correlations.

```
import pandas as pd, matplotlib.pyplot as plt
c=pd.read_csv('../../arrays.sscore',sep='\t').filter(regex='_AVG$').corr()
plt.figure(figsize=(8,8)); im=plt.imshow(c,vmin=-1,vmax=1); plt.colorbar(im); plt.xticks(range(len(c)),c.columns,rotation=90); plt.yticks(range(len(c)),c.columns); plt.tight_layout(); plt.show()
```

<img width="780" height="758" alt="image" src="https://github.com/user-attachments/assets/8598d356-92bd-4ff1-8f7e-f77672f72e37" />


This lets us know that we should drop PGS005199 or PGS005203 since they are highly correlated. Looking at the PGS catalog, PGS005199 is multi-ancestry and better performing.


```
# Read the file
df = pd.read_csv('../../arrays.sscore', sep='\t')

# Drop the PGS005203 columns
df = df.drop(columns=['PGS005203_AVG', 'PGS005203_MISSING_PCT'])

# Save back to file
df.to_csv('../../arrays.sscore', sep='\t', index=False)
```

Let's make the case definition.

ICD-10:

* E66.811
* E66.812
* E66.813
* E66.01
* E66.09
* E66.1
* E66.2
* E66.89
* E66.9
* E88.82

ICD-9:

* 278.00
* 278.01
* 278.03

```
import os
from google.cloud import bigquery as bq

cdr_id = os.environ["WORKSPACE_CDR"]

icd10_obesity = [
    "E66.811", "E66.812", "E66.813",
    "E66.01", "E66.09",
    "E66.1", "E66.2",
    "E66.89", "E66.9",
    "E88.82",
]
icd9_obesity = [
    "278.00", "278.01", "278.03",
]

codes = icd10_obesity + icd9_obesity
codes = [c.upper() for c in (codes + [c.replace(".", "") for c in codes])]

sql = f"""
SELECT DISTINCT CAST(person_id AS STRING) AS person_id
FROM `{cdr_id}.condition_occurrence`
WHERE condition_source_value IS NOT NULL
  AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
"""

cases = (
    bq.Client()
    .query(sql, job_config=bq.QueryJobConfig(
        query_parameters=[bq.ArrayQueryParameter("codes", "STRING", codes)]
    ))
    .to_dataframe()["person_id"]
    .astype(str)
)
```

For simplicity, we aren't using a time-to-event model. Some of our "controls" will go on to develop colorectal cancer later in life.

Calculate metrics:
```
import pandas as pd, numpy as np, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, norm, chi2
d=pd.read_csv('../../arrays.sscore',sep='\t'); idcol=d.columns[0]; y=d[idcol].astype(str).isin(set(cases)).astype(int)
res=[]
for s in d.filter(regex='_AVG$').columns:
    x=pd.to_numeric(d[s],errors='coerce'); t=pd.DataFrame({'y':y,'x':x}).dropna()
    if t['y'].nunique()<2 or t['x'].std(ddof=0)==0: continue
    z=(t['x']-t['x'].mean())/t['x'].std(ddof=0); X=sm.add_constant(pd.DataFrame({'z':z}))
    m=sm.Logit(t['y'],X).fit(disp=False, maxiter=200); beta=float(m.params['z']); se=float(m.bse['z'])
    OR=np.exp(beta); p_or=1-norm.cdf(beta/se)
    auc=roc_auc_score(t['y'],t['x']); p_auc=mannwhitneyu(t.loc[t['y']==1,'x'],t.loc[t['y']==0,'x'],alternative='greater').pvalue
    n=len(t); r2=(1-np.exp((2/n)*(m.llnull-m.llf)))/(1-np.exp(2*m.llnull/n)); p_r2=1-chi2.cdf(2*(m.llf-m.llnull),1)
    res.append([s,n,int(t['y'].sum()),int(n-t['y'].sum()),OR,p_or,auc,p_auc,r2,p_r2])
pd.DataFrame(res,columns=['score','n','cases','controls','OR_perSD','p_one_sided_OR>1','AUROC','p_one_sided_AUC>0.5','Nagelkerke_R2','p_one_sided_R2>0']).sort_values('Nagelkerke_R2',ascending=False)
```

The best score is PGS005198 with an OR/SD of 1.453851 and AUROC of 0.602658. PGS004378 is in second place, with an AUROC of 0.589763. PGS005235 is in third, with an AUROC of 0.583957.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

d = pd.read_csv('../../arrays.sscore', sep='\t')
d['#IID'] = d['#IID'].astype(str)
d = d.set_index('#IID')
score_cols = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
storage_opts = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs_raw = pd.read_csv(PCS_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
pc_df = (pd.DataFrame(pcs_raw["pca_features"].apply(parse).to_list(), columns=pc_cols)
         .assign(person_id=pcs_raw["research_id"].astype(str)).set_index("person_id"))

sex_df = (pd.read_csv(SEX_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","dragen_sex_ploidy"])
          .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"]
          .map({"XX":0,"XY":1}).to_frame("sex"))

covars = pc_df.join(sex_df, how="inner").dropna()
case_idx = pd.Index(pd.Series(cases, dtype=str).unique(), name="person_id")

def eval_block(df):
    if df.empty: 
        return {"n": 0, "cases": 0, "controls": 0, "AUROC": np.nan, "ACC": np.nan}
    y = df.index.to_series().isin(case_idx).astype('i1').to_numpy()
    X = df.astype(float).to_numpy()
    mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    mdl.fit(X, y)
    p = mdl.predict_proba(X)[:,1]
    return {"n": int(len(df)), "cases": int(y.sum()), "controls": int(len(y)-y.sum()),
            "AUROC": float(roc_auc_score(y, p)), "ACC": float(accuracy_score(y, (p>=0.5).astype(int)))}

rows = []
for sc in score_cols:
    s = d[[sc]].dropna()
    s_sex = d[[sc]].join(sex_df, how="inner").dropna()[[sc,"sex"]]
    s_sex_pcs = d[[sc]].join(covars, how="inner").dropna()[["sex"]+pc_cols+[sc]]

    r1 = eval_block(s[[sc]])
    r2 = eval_block(s_sex)
    r3 = eval_block(s_sex_pcs)

    rows.append({
        "score": sc.replace("_AVG",""),
        "n_s": r1["n"], "cases_s": r1["cases"], "AUROC_s": r1["AUROC"], "ACC_s": r1["ACC"],
        "n_sx": r2["n"], "cases_sx": r2["cases"], "AUROC_sx": r2["AUROC"], "ACC_sx": r2["ACC"],
        "n_sxpc": r3["n"], "cases_sxpc": r3["cases"], "AUROC_sxpc": r3["AUROC"], "ACC_sxpc": r3["ACC"],
    })

out = pd.DataFrame(rows).sort_values("AUROC_sxpc", ascending=False).reset_index(drop=True)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
```

| Score ID  | AUROC — Score only | AUROC — Score + Sex | AUROC — Score + Sex + PCs |
| :-------- | -----------------: | ------------------: | ------------------------: |
| PGS005199 |              0.588 |               0.587 |                     0.640 |
| PGS005198 |              0.603 |               0.615 |                     0.631 |
| PGS004150 |              0.570 |               0.585 |                     0.615 |
| PGS004378 |              0.590 |               0.604 |                     0.614 |
| PGS003897 |              0.576 |               0.593 |                     0.608 |
| PGS005235 |              0.584 |               0.597 |                     0.607 |
| PGS003400 |              0.546 |               0.569 |                     0.591 |

Everything is helpful: score, sex, and PCs.


Now, let's test the same model, except jointly fitting all scores using cross-validation to avoid overfitting.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs = pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
                  sep="\t", storage_options=so, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs["research_id"].astype(str)).set_index("person_id"))

sex_df = (pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                      sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
           .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"]
           .map({"XX":0,"XY":1}).to_frame("sex"))

covars = pc_df.join(sex_df, how="inner").dropna()
case_set = set(pd.Series(cases, dtype=str))

Xy = covars.join(d[pgs], how="inner").dropna()
y = Xy.index.to_series().isin(case_set).astype('i1').to_numpy()
X = Xy.to_numpy()

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]
auc = roc_auc_score(y, p)

pd.DataFrame([{
    "n": int(len(y)),
    "cases": int(y.sum()),
    "controls": int(len(y) - y.sum()),
    "predictors": int(X.shape[1]),
    "AUROC": float(auc)
}])
```

This yields an AUROC of 0.646124.

But do score ensembles really improve the result?

Let's compare:
```
import numpy as np, pandas as pd, math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs_cols = [c for c in d.columns if c.endswith('_AVG')]
idx_all = d[pgs_cols].dropna().index
X_all = d.loc[idx_all, pgs_cols].astype(float)
X_507 = d.loc[idx_all, ['PGS005198_AVG']].astype(float)
y = pd.Index(idx_all).to_series().isin(pd.Series(cases, dtype=str)).astype('i1').to_numpy()

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))

p_all = np.zeros(len(y)); p_507 = np.zeros(len(y))
auc_all_f, auc_507_f = [], []

for tr, te in cv.split(X_all, y):
    m_all = pipe(); m_507 = pipe()
    m_all.fit(X_all.iloc[tr], y[tr]); m_507.fit(X_507.iloc[tr], y[tr])
    p_all[te] = m_all.predict_proba(X_all.iloc[te])[:,1]
    p_507[te] = m_507.predict_proba(X_507.iloc[te])[:,1]
    auc_all_f.append(roc_auc_score(y[te], p_all[te]))
    auc_507_f.append(roc_auc_score(y[te], p_507[te]))

auc_all = roc_auc_score(y, p_all)
auc_507 = roc_auc_score(y, p_507)

perf = pd.DataFrame([
    {"model":"All AVG", "n":len(y), "cases":int(y.sum()), "controls":int(len(y)-y.sum()), "AUROC":float(auc_all)},
    {"model":"PGS005198 only", "n":len(y), "cases":int(y.sum()), "controls":int(len(y)-y.sum()), "AUROC":float(auc_507)},
]).sort_values("AUROC", ascending=False).reset_index(drop=True)

diff =  np.array(auc_all_f) - np.array(auc_507_f)
nz = diff[diff!=0]
s = int((nz>0).sum()); n = int(len(nz))
p_one_sided = sum(math.comb(n,k) for k in range(s, n+1)) / (2**n) if n>0 else np.nan

print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"One-sided sign-test p (PGS005198 < All AVG): {p_one_sided:.3g} over {s}/{n} folds; ΔAUROC (OOF) = {auc_all-auc_507:+.3f}")
```

We see that the AUC of the ensemble (0.622) is significantly better than that of the single score (0.603).


Let's plot risk in each decline of the multivariate model.
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs_cols = [c for c in d.columns if c.endswith('_AVG')]
single_col = "PGS005198_AVG"

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs = pd.read_csv(
    "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
    sep="\t", storage_options=so, usecols=["research_id","pca_features"]
)
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (
    pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
      .assign(person_id=pcs["research_id"].astype(str))
      .set_index("person_id")
)

sex_df = (
    pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
      .assign(person_id=lambda x: x["research_id"].astype(str))
      .set_index("person_id")["dragen_sex_ploidy"]
      .map({"XX":0,"XY":1})
      .to_frame("sex")
)

case_idx = pd.Index(pd.Series(cases, dtype=str).unique(), name="person_id")

# Build one common analysis set so all models use identical rows
base = (
    d[pgs_cols]
      .join(sex_df, how="inner")
      .join(pc_df, how="inner")
      .dropna()
      .astype(float)
)

y = base.index.to_series().isin(case_idx).astype('i1').to_numpy()
X_single = base[[single_col]]
X_all = base[pgs_cols]
X_all_spc = base[pgs_cols + ["sex"] + pc_cols]

def oof_pred(X, y):
    cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    m = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p

def deciles_obs(p, y, q=10):
    df = pd.DataFrame({"p": p, "y": y})
    bins = pd.qcut(df["p"], q=q, labels=False, duplicates="drop")
    g = df.assign(decile=bins).groupby("decile", observed=True)
    r = g["y"].mean().to_frame("rate")
    r.index = r.index.astype(int) + 1
    return r

p_single = oof_pred(X_single, y)
p_all = oof_pred(X_all, y)
p_all_spc = oof_pred(X_all_spc, y)

d_single = deciles_obs(p_single, y)
d_all = deciles_obs(p_all, y)
d_all_spc = deciles_obs(p_all_spc, y)

plt.figure()
plt.plot(d_single.index, d_single["rate"], marker="o", label=f"{single_col} only")
plt.plot(d_all.index, d_all["rate"], marker="o", label="All scores")
plt.plot(d_all_spc.index, d_all_spc["rate"], marker="o", label="Scores + Sex + PCs")
plt.xlabel("Predicted risk decile")
plt.ylabel("Observed case rate")
plt.title(f"Deciles (observed only) — n={len(y)}")
plt.legend()
plt.tight_layout()
```

We can see that the ensemble helps, and also adding PCs + sex help.

**Note:** we do not include age at all, which adds noise to the results.

Let's check some final metrics.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# data
d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs = pd.read_csv(
    "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
    sep="\t", storage_options=so, usecols=["research_id","pca_features"]
)
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (
    pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
      .assign(person_id=pcs["research_id"].astype(str))
      .set_index("person_id")
).dropna()

sex_df = (
    pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
      .assign(person_id=lambda x: x["research_id"].astype(str))
      .set_index("person_id")["dragen_sex_ploidy"]
      .map({"XX":0,"XY":1})
      .to_frame("sex")
)

# predictors = all scores + sex + PCs
X = (
    d[pgs]
      .join(sex_df, how="inner")
      .join(pc_df, how="inner")
      .dropna()
      .astype(float)
)

y = X.index.to_series().isin(pd.Series(cases, dtype=str)).astype('i1').to_numpy()

# 10-fold CV, out-of-fold predictions
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
p = np.zeros(len(y))
for tr, te in cv.split(X, y):
    mdl.fit(X.iloc[tr], y[tr])
    p[te] = mdl.predict_proba(X.iloc[te])[:,1]

# metrics
auc = roc_auc_score(y, p)
z = (p - p.mean()) / p.std(ddof=0)
beta = LogisticRegression(solver="lbfgs", max_iter=1000).fit(z.reshape(-1,1), y).coef_[0,0]
or_per_sd = float(np.exp(beta))
base = y.mean()
thr = min(2*base, 0.999)
frac_2x = float((p >= thr).mean())

pd.DataFrame([{
    "n": int(len(y)),
    "cases": int(y.sum()),
    "controls": int(len(y)-y.sum()),
    "predictors": int(X.shape[1]),
    "AUROC": float(auc),
    "OR_per_SD_pred": or_per_sd,
    "fraction_at_≥2x_risk": frac_2x,
    "baseline_risk": float(base),
    "2x_threshold": float(thr)
}])
```

The AUROC is 0.64612, with an OR/SD of 1.647. The fraction at 2x+ risk is only 0.017, since that corresponds to an absolute risk threshold of 0.416


Let's check accuracy within each inferred genetic ancestry group:

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/4d29d084-db43-44c5-bdf0-bb3ca589b53d" />

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/5786af4c-2140-4e95-b2be-cf3684722c17" />

Let's check portability trends per score:
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID'); d.index = d.index.astype(str)
pgs_cols = [c for c in d.columns if c.endswith('_AVG')]

PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
anc_series = (
    pd.read_csv(PCS_URI, sep="\t", storage_options=so, usecols=["research_id","ancestry_pred"])
      .assign(person_id=lambda x: x["research_id"].astype(str))
      .set_index("person_id")["ancestry_pred"]
)

case_set = set(pd.Series(cases, dtype=str))

def oof_pred(X, y, n_splits=10):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    m = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:,1]
    return p

def or_per_sd(y, p):
    if p.std(ddof=0) == 0 or y.sum() == 0 or y.sum() == len(y):
        return np.nan
    z = (p - p.mean()) / p.std(ddof=0)
    beta = sm.Logit(y, sm.add_constant(z)).fit(disp=False, maxiter=200).params[1]
    return float(np.exp(beta))

rows = []
for sc in pgs_cols:
    base = d[[sc]].join(anc_series.to_frame("ancestry"), how="inner").dropna()
    if base.empty:
        continue
    y = base.index.to_series().isin(case_set).astype('i1').to_numpy()
    X = base[[sc]].astype(float)
    p = oof_pred(X, y)

    df = pd.DataFrame({"y": y, "p": p, "ancestry": base["ancestry"].astype(str).values}, index=base.index)
    for anc, g in df.groupby("ancestry", observed=True):
        if g["y"].nunique() < 2 or g["p"].std(ddof=0) == 0:
            continue
        rows.append({
            "score": sc.replace("_AVG",""),
            "ancestry": anc,
            "AUROC": float(roc_auc_score(g["y"], g["p"])),
            "ACC": float(accuracy_score(g["y"], (g["p"] >= 0.5).astype(int))),
            "OR_per_SD": or_per_sd(g["y"].to_numpy(), g["p"].to_numpy())
        })

res = pd.DataFrame(rows)
if res.empty:
    raise RuntimeError("No results computed. Check inputs and ancestry labels.")

acc_median = res.groupby("ancestry", observed=True)["ACC"].median().dropna().sort_values(ascending=False)
anc_order = acc_median.index.tolist()

plt.figure(figsize=(10,6))
au_piv = res.pivot_table(index="ancestry", columns="score", values="AUROC")
au_piv = au_piv.reindex(anc_order)
for sc in au_piv.columns:
    plt.plot(range(len(anc_order)), au_piv[sc].to_numpy(), marker="o", label=sc)
plt.xticks(range(len(anc_order)), anc_order, rotation=25, ha="right")
plt.ylabel("AUROC")
plt.title("Per-score AUROC by ancestry (single-score models, no sex/PCs)\nAncestries sorted by median accuracy within category")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()

plt.figure(figsize=(10,6))
or_piv = res.pivot_table(index="ancestry", columns="score", values="OR_per_SD")
or_piv = or_piv.reindex(anc_order)
for sc in or_piv.columns:
    plt.plot(range(len(anc_order)), or_piv[sc].to_numpy(), marker="o", label=sc)
plt.xticks(range(len(anc_order)), anc_order, rotation=25, ha="right")
plt.ylabel("OR per SD of prediction")
plt.title("Per-score OR/SD by ancestry (single-score models, no sex/PCs)\nAncestries sorted by median accuracy within category")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()

res.sort_values(["score","ancestry"]).reset_index(drop=True)
```

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/242731a9-c025-4d07-a564-134da76d06c8" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/c71e65cc-f242-4904-ab67-5f43276ac42a" />


