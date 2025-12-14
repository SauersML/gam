## Running a polygenic score for breast cancer on the _All of Us_ cohort

First, let's see which colorectal cancer scores exist in the PGS Catalog:

[https://www.pgscatalog.org/trait/MONDO_0005575/
](https://www.pgscatalog.org/trait/MONDO_0007254/)

Strong performers include:
- PGS000332
- PGS000335
- PGS000015
- PGS000508
- PGS000344
- PGS000317
- PGS000007
- PGS000507
- PGS004869

We'll assess these scores on three metrics in the test data:
- Odds ratio (per standard deviation of score)
- AUROC
- Nagelkerke's R²

`gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed` is 10.51 TiB, so we can use `gs://fc-aou-datasets-controlled/v8/microarray/plink/` instead.

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
./gnomon/target/release/gnomon score "PGS000332,PGS000015,PGS000508,PGS000344,PGS000317,PGS000007,PGS000507,PGS004869" arrays
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
#IID	PGS000007_AVG	PGS000007_MISSING_PCT	PGS000015_AVG	PGS000015_MISSING_PCT	PGS000317_AVG	PGS000317_MISSING_PCT	PGS000332_AVG	PGS000332_MISSING_PCT	PGS000344_AVG	PGS000344_MISSING_PCT	PGS000507_AVG	PGS000507_MISSING_PCT	PGS000508_AVG	PGS000508_MISSING_PCT	PGS004869_AVG	PGS004869_MISSING_PCT
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

<img width="780" height="758" alt="image" src="https://github.com/user-attachments/assets/ecbf0e0e-1e08-4401-8534-0e3879d2f037" />

Let's make the case definition.





```
import os
cdr_id = os.environ["WORKSPACE_CDR"]
from google.cloud import bigquery as bq

# ICD-10: C50.0–C50.9 (malignant), D05.0–D05.9 (in situ), Z85.3 (personal history)
icd10_breast = [f"C50.{i}" for i in range(10)] + [f"D05.{i}" for i in range(10)] + ["Z85.3"]

# ICD-9: 174.0–174.9 (female), 175.0–175.9 (male), 233.0 (in situ), V10.3 (personal history)
icd9_breast = [f"174.{i}" for i in range(10)] + [f"175.{i}" for i in range(10)] + ["233.0", "V10.3"]

codes = icd10_breast + icd9_breast
codes = [c.upper() for c in (codes + [c.replace(".", "") for c in codes])]

sql = f"""
SELECT DISTINCT CAST(person_id AS STRING) AS person_id
FROM `{cdr_id}.condition_occurrence`
WHERE condition_source_value IS NOT NULL
  AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
"""

cases = (
    bq.Client()
    .query(sql, job_config=bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("codes", "STRING", codes)]))
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

| Score         |       N | Cases | Controls | OR per SD | P (OR>1) | AUROC | P (AUC>0.5) | Nagelkerke R² | P (R²>0) |
| ------------- | ------: | ----: | -------: | --------: | -------: | ----: | ----------: | ------------: | -------: |
| PGS000007_AVG | 447,278 | 6,614 |  440,664 |     1.275 |  <1e-10 | 0.569 |    3.49e-82 |         0.61% |  <1e-10 |
| PGS000317_AVG | 447,278 | 6,614 |  440,664 |     1.269 |  <1e-10 | 0.567 |    1.30e-79 |         0.57% |  <1e-10 |
| PGS004869_AVG | 447,278 | 6,614 |  440,664 |     1.201 |  <1e-10 | 0.552 |    1.57e-47 |         0.34% |  <1e-10 |
| PGS000507_AVG | 447,278 | 6,614 |  440,664 |     1.179 |  <1e-10 | 0.547 |    1.68e-40 |         0.28% |  <1e-10 |
| PGS000344_AVG | 447,278 | 6,614 |  440,664 |     1.152 |  <1e-10 | 0.540 |    1.29e-29 |         0.20% |  <1e-10 |
| PGS000508_AVG | 447,278 | 6,614 |  440,664 |     1.151 |  <1e-10 | 0.542 |    2.69e-32 |         0.20% |  <1e-10 |
| PGS000015_AVG | 447,278 | 6,614 |  440,664 |     1.150 |  <1e-10 | 0.539 |    5.96e-28 |         0.20% |  <1e-10 |
| PGS000332_AVG | 447,278 | 6,614 |  440,664 |     1.131 |  <1e-10 | 0.537 |    4.62e-25 |         0.15% |  <1e-10 |

PGS000007 is the best so far. Let's add 16 principal components and sex to the model as predictors (not controls) and look at the resulting AUC.

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


| Score     |      nₛ | casesₛ | AUROCₛ |  ACCₛ |     nₛₓ | casesₛₓ | AUROCₛₓ | ACCₛₓ |   nₛₓₚ꜀ | casesₛₓₚ꜀ | AUROCₛₓₚ꜀ | ACCₛₓₚ꜀ |
| --------- | ------: | -----: | -----: | ----: | ------: | ------: | ------: | ----: | ------: | --------: | --------: | ------: |
| PGS000507 | 447,278 |  6,614 |  0.547 | 0.985 | 413,337 |   6,210 |   0.721 | 0.985 | 413,337 |     6,210 |     0.785 |   0.985 |
| PGS000508 | 447,278 |  6,614 |  0.542 | 0.985 | 413,337 |   6,210 |   0.718 | 0.985 | 413,337 |     6,210 |     0.784 |   0.985 |
| PGS000332 | 447,278 |  6,614 |  0.537 | 0.985 | 413,337 |   6,210 |   0.715 | 0.985 | 413,337 |     6,210 |     0.783 |   0.985 |
| PGS004869 | 447,278 |  6,614 |  0.552 | 0.985 | 413,337 |   6,210 |   0.724 | 0.985 | 413,337 |     6,210 |     0.779 |   0.985 |
| PGS000317 | 447,278 |  6,614 |  0.567 | 0.985 | 413,337 |   6,210 |   0.733 | 0.985 | 413,337 |     6,210 |     0.778 |   0.985 |
| PGS000007 | 447,278 |  6,614 |  0.569 | 0.985 | 413,337 |   6,210 |   0.735 | 0.985 | 413,337 |     6,210 |     0.776 |   0.985 |
| PGS000015 | 447,278 |  6,614 |  0.539 | 0.985 | 413,337 |   6,210 |   0.715 | 0.985 | 413,337 |     6,210 |     0.776 |   0.985 |
| PGS000344 | 447,278 |  6,614 |  0.540 | 0.985 | 413,337 |   6,210 |   0.717 | 0.985 | 413,337 |     6,210 |     0.774 |   0.985 |

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

This yields an AUROC of 0.787221, much better!

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
X_507 = d.loc[idx_all, ['PGS000507_AVG']].astype(float)
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
    {"model":"PGS000507 only", "n":len(y), "cases":int(y.sum()), "controls":int(len(y)-y.sum()), "AUROC":float(auc_507)},
]).sort_values("AUROC", ascending=False).reset_index(drop=True)

diff =  np.array(auc_all_f) - np.array(auc_507_f)
nz = diff[diff!=0]
s = int((nz>0).sum()); n = int(len(nz))
p_one_sided = sum(math.comb(n,k) for k in range(s, n+1)) / (2**n) if n>0 else np.nan

print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"One-sided sign-test p (PGS000507 < All AVG): {p_one_sided:.3g} over {s}/{n} folds; ΔAUROC (OOF) = {auc_all-auc_507:+.3f}")
```

We see that the AUC of the ensemble (0.588) is significantly better than that of the single score (0.547).




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
X_all = d[pgs_cols].dropna().astype(float)
X_507 = d[['PGS000507_AVG']].dropna().astype(float)

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
pcs = pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
                  sep="\t", storage_options=so, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")).dropna()
X_spc = X_all.join(pc_df, how="inner").dropna()

sex = (pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                   sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
         .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"])

case_idx = pd.Index(pd.Series(cases, dtype=str).unique())

def oof(X, y):
    cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    m = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:,1]
    return p

def deciles_obs(p, y, q=10):
    df = pd.DataFrame({"p": p, "y": y})
    bins = pd.qcut(df["p"], q=q, labels=False, duplicates="drop")
    g = df.assign(decile=bins).groupby("decile", observed=True)
    r = (g["y"].mean()).to_frame("rate")
    r.index = r.index.astype(int) + 1
    return r

# Women: three lines (observed only)
w_ix = sex[sex=="XX"].index
ix_507 = X_507.index.intersection(w_ix)
ix_all = X_all.index.intersection(w_ix)
ix_spc = X_spc.index.intersection(w_ix)

y_507 = pd.Index(ix_507).to_series().isin(case_idx).astype('i1').to_numpy()
y_all = pd.Index(ix_all).to_series().isin(case_idx).astype('i1').to_numpy()
y_spc = pd.Index(ix_spc).to_series().isin(case_idx).astype('i1').to_numpy()

p_507 = oof(X_507.loc[ix_507], y_507)
p_all = oof(X_all.loc[ix_all], y_all)
p_spc = oof(X_spc.loc[ix_spc], y_spc)

d_507 = deciles_obs(p_507, y_507)
d_all = deciles_obs(p_all, y_all)
d_spc = deciles_obs(p_spc, y_spc)

plt.figure()
plt.plot(d_507.index, d_507["rate"], marker="o", label="PGS000507 only")
plt.plot(d_all.index, d_all["rate"], marker="o", label="All scores")
plt.plot(d_spc.index, d_spc["rate"], marker="o", label="Scores + PCs")
plt.xlabel("Predicted risk decile (women)")
plt.ylabel("Observed case rate")
plt.title(f"Deciles (observed only) — Women  |  n={len(ix_all)}")
plt.legend()
plt.tight_layout()

# Men: bar chart bottom 90% vs top 10% using Scores+PCs ensemble
m_ix = sex[sex=="XY"].index
ix_spc_m = X_spc.index.intersection(m_ix)
y_m = pd.Index(ix_spc_m).to_series().isin(case_idx).astype('i1').to_numpy()
p_spc_m = oof(X_spc.loc[ix_spc_m], y_m)
thr = np.quantile(p_spc_m, 0.9)
top = y_m[p_spc_m >= thr]
bot = y_m[p_spc_m < thr]

rates = [bot.mean() if len(bot) else np.nan, top.mean() if len(top) else np.nan]
counts = [len(bot), len(top)]

plt.figure()
plt.bar(["Bottom 90%", "Top 10%"], rates)
plt.ylabel("Observed case rate")
plt.title(f"Scores + PCs — Men  |  n={len(y_m)}  |  n90={counts[0]}, n10={counts[1]}")
plt.tight_layout()
```

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/d8a3a4b0-63f9-472c-9cdf-c2028780e910" />

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/5e6f0368-a946-4291-ac18-c54f044f25c2" />

We can see that the ensemble helps, and also adding PCs help.

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

pcs = pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
                  sep="\t", storage_options=so, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")).dropna()

sex = (pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                   sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
         .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"])

# women only; predictors = all scores + PCs
X = d[pgs].join(pc_df, how="inner").dropna()
idx_w = X.index.intersection(sex[sex=="XX"].index)
Xw = X.loc[idx_w].astype(float)
y = pd.Index(idx_w).to_series().isin(pd.Series(cases, dtype=str)).astype('i1').to_numpy()

# 10-fold CV, OOF predictions
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
p = np.zeros(len(y))
for tr, te in cv.split(Xw, y):
    mdl.fit(Xw.iloc[tr], y[tr])
    p[te] = mdl.predict_proba(Xw.iloc[te])[:,1]

# metrics
auc = roc_auc_score(y, p)
z = (p - p.mean()) / p.std(ddof=0)
beta = LogisticRegression(solver="lbfgs", max_iter=1000).fit(z.reshape(-1,1), y).coef_[0,0]
or_per_sd = float(np.exp(beta))
base = y.mean()
thr = min(2*base, 0.999)
frac_2x = float((p >= thr).mean())

pd.DataFrame([{
    "n_women": int(len(y)),
    "cases": int(y.sum()),
    "controls": int(len(y)-y.sum()),
    "AUROC": float(auc),
    "OR_per_SD_pred": or_per_sd,
    "fraction_women_at_≥2x_risk": frac_2x,
    "baseline_risk": float(base),
    "2x_threshold": float(thr)
}])
```

In women, PGS ensemble + PCs:
- 0.657404 AUROC
- 1.539613 OR/SD
- 6.2% at 2x risk

Let's check ICD code heterogeneity:

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/93b62d4d-49a0-4301-bc0b-fafd69844fba" />

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/d5397d26-b530-4378-bf8a-7b71bb2cd577" />

Let's check accuracy within each inferred genetic ancestry group:
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# data
d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID'); d.index = d.index.astype(str)
pgs_cols = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs_raw = pd.read_csv(PCS_URI, sep="\t", storage_options=so, usecols=["research_id","pca_features","ancestry_pred"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs_raw["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs_raw["research_id"].astype(str), ancestry_pred=pcs_raw["ancestry_pred"])
           .set_index("person_id"))

sex = (pd.read_csv(SEX_URI, sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
         .assign(person_id=lambda x: x["research_id"].astype(str))
         .set_index("person_id")["dragen_sex_ploidy"])

women_ids = d.index.intersection(sex[sex=="XX"].index)
case_set = set(pd.Series(cases, dtype=str))

X = d.loc[women_ids, pgs_cols].join(pc_df[pc_cols], how="inner").dropna()
y = pd.Index(X.index).to_series().isin(case_set).astype('i1').to_numpy()
anc = pc_df.loc[X.index, "ancestry_pred"]

# 10-fold OOF probabilities
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
clf = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000))
p = np.zeros(len(y))
for tr, te in cv.split(X.values, y):
    clf.fit(X.iloc[tr], y[tr]); p[te] = clf.predict_proba(X.iloc[te])[:,1]

def auc_ci_boot(y, p, B=500, rng=123):
    rs = np.random.RandomState(rng)
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    aucs = []
    for _ in range(B):
        idx = np.concatenate([rs.choice(pos, len(pos), True), rs.choice(neg, len(neg), True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, p)), float(lo), float(hi)

def orsd_ci(y, p):
    z = (p - p.mean()) / p.std(ddof=0)
    m = sm.Logit(y, sm.add_constant(z)).fit(disp=False, maxiter=200)
    beta, se = float(m.params[1]), float(m.bse[1])
    or_sd = np.exp(beta)
    return float(or_sd), float(np.exp(beta-1.96*se)), float(np.exp(beta+1.96*se))

rows = []
for anc_val, idx in anc.groupby(anc).groups.items():
    idx = pd.Index(idx)
    y_g = pd.Series(y, index=X.index).loc[idx].to_numpy()
    p_g = pd.Series(p, index=X.index).loc[idx].to_numpy()
    if y_g.sum()==0 or y_g.sum()==len(y_g) or p_g.std(ddof=0)==0: 
        continue
    auc, auc_lo, auc_hi = auc_ci_boot(y_g, p_g)
    orsd, or_lo, or_hi = orsd_ci(y_g, p_g)
    rows.append({"ancestry": str(anc_val), "n_women": int(len(y_g)), "cases": int(y_g.sum()),
                 "AUROC": auc, "AUROC_lo": auc_lo, "AUROC_hi": auc_hi,
                 "OR_per_SD": orsd, "OR_lo": or_lo, "OR_hi": or_hi})

res = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# two plots: AUROC and OR/SD with 95% CI
anc_order = res["ancestry"].tolist()
x = np.arange(len(anc_order)); w = 0.6

plt.figure(figsize=(9,5))
yv = res.set_index("ancestry").loc[anc_order, "AUROC"].to_numpy()
lo = yv - res.set_index("ancestry").loc[anc_order, "AUROC_lo"].to_numpy()
hi = res.set_index("ancestry").loc[anc_order, "AUROC_hi"].to_numpy() - yv
plt.bar(x, yv, width=w, yerr=[lo, hi], capsize=4)
plt.xticks(x, anc_order, rotation=25, ha="right"); plt.ylabel("AUROC")
plt.title("Women — AUROC by ancestry (Scores + PCs, 10-fold OOF)"); plt.tight_layout()

plt.figure(figsize=(9,5))
yv = res.set_index("ancestry").loc[anc_order, "OR_per_SD"].to_numpy()
lo = yv - res.set_index("ancestry").loc[anc_order, "OR_lo"].to_numpy()
hi = res.set_index("ancestry").loc[anc_order, "OR_hi"].to_numpy() - yv
plt.bar(x, yv, width=w, yerr=[lo, hi], capsize=4)
plt.xticks(x, anc_order, rotation=25, ha="right"); plt.ylabel("OR per SD of prediction")
plt.title("Women — OR/SD by ancestry (Scores + PCs, 10-fold OOF)"); plt.tight_layout()

res
```

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/7ddcf049-1906-47ba-bf8a-7318b0738e36" />

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/cb2253bf-59b2-4415-a3a2-70ace6713e64" />

