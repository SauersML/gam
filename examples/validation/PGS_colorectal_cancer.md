## Running a polygenic score for colorectal cancer on the _All of Us_ cohort

First, let's see which colorectal cancer scores exist in the PGS Catalog:
```
https://www.pgscatalog.org/trait/MONDO_0005575/
```

There are a few types of statistics to consider:
- AUROC / AUC / c-statistic: cases vs. controls
- C-index: like AUROC, but for time-to-event data
- AUPRC: precision vs recall across thresholds
- OR, HR: relative risk (hopefully normalized per-SD of score)
- Nagelkerke’s pseudo-R²: explained variance but for binary

Strong performers seem to include:
- PGS000765
- PGS004904
- PGS003433
- PGS003979
- PGS003386
- PGS003852
- PGS004303

We'll assess these scores on three metrics in the test data:
- Odds ratio (per standard deviation of score)
- AUROC
- Nagelkerke's R²

We want to avoid imputation, but we still want lots of samples with many variants.

Let's look at our options:

| Group             | Subset         | PLINK version | File types                | Total size | GCS path                                                                              |
| ----------------- | -------------- | ---------- | ------------------------- | ---------: | ------------------------------------------------------------------------------------- |
| srWGS SNP & Indel | Exome          | 1.9        | `.bed`, `.bim`, `.fam`    |   3.94 TiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/exome/plink_bed`          |
| srWGS SNP & Indel | Exome          | 2.0        | `.pgen`, `.pvar`, `.psam` |  96.65 GiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/exome/pgen`               |
| srWGS SNP & Indel | ACAF Threshold | 1.9        | `.bed`, `.bim`, `.fam`    |  10.51 TiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed` |
| srWGS SNP & Indel | ACAF Threshold | 2.0        | `.pgen`, `.pvar`, `.psam` |   1.11 TiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/pgen`      |
| srWGS SNP & Indel | ClinVar        | 1.9        | `.bed`, `.bim`, `.fam`    | 204.68 GiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/plink_bed`        |
| srWGS SNP & Indel | ClinVar        | 2.0        | `.pgen`, `.pvar`, `.psam` |    9.3 GiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/pgen`             |
| Genotyping Array  | Array          | 1.9        | `.bed`, `.bim`, `.fam`    |  181.2 GiB | `gs://fc-aou-datasets-controlled/v8/microarray/plink`                                 |

ClinVar and exome subsets will not have enough variants. Array data might work for most scores, and ACAF Threshold will likely work even better. The best option is gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed, though it will take a very long time to stream 10.51 TiB.

Create a standard cloud analysis environment with 8 CPUs and 30 GB RAM (though using lower or higher values should also work similarly).

Install gnomon if you haven't already:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && rm -rf gnomon && git clone https://github.com/SauersML/gnomon.git && cd gnomon && cargo build --release && cd ~
```

For now, let's download the microarray data. This may impact the variant overlap of our scores.
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Now we can run the scores. It should be faster to run them all at once instead of one at a time.
```
./gnomon/target/release/gnomon score "PGS000765,PGS004904,PGS003433,PGS003979,PGS003386,PGS003852,PGS004303" arrays
```

This should take 11 minutes to run, and output a file called:
```
arrays.sscore
```
Let's open a Jupyter analysis notebook.
```
!head ../../arrays.sscore
```

This shows us the columns of the score output:
```
#IID	PGS000765_AVG	PGS000765_MISSING_PCT	PGS003386_AVG	PGS003386_MISSING_PCT	PGS003433_AVG	PGS003433_MISSING_PCT	PGS003852_AVG	PGS003852_MISSING_PCT	PGS003979_AVG	PGS003979_MISSING_PCT	PGS004303_AVG	PGS004303_MISSING_PCT	PGS004904_AVG	PGS004904_MISSING_PCT
```

Let's plot missingness for each score:
```
import pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv('../../arrays.sscore', sep='\t')
for c in d.columns[d.columns.str.endswith('_MISSING_PCT')]: d[c].hist(bins=50); plt.title(c); plt.xlabel(c); plt.ylabel('count'); plt.show()
```

Here's one example:

<img width="597" height="454" alt="image" src="https://github.com/user-attachments/assets/bbe62bb4-6f93-4953-b441-c31ef1859804" />

This is not too bad given that we are using microarray data.

Now let's check the correlation matrix of the scores. We expect moderate correlations.

```
import pandas as pd, matplotlib.pyplot as plt
c=pd.read_csv('../../arrays.sscore',sep='\t').filter(regex='_AVG$').corr()
plt.figure(figsize=(8,8)); im=plt.imshow(c,vmin=-1,vmax=1); plt.colorbar(im); plt.xticks(range(len(c)),c.columns,rotation=90); plt.yticks(range(len(c)),c.columns); plt.tight_layout(); plt.show()
```

<img width="390" height="379" alt="image" src="https://github.com/user-attachments/assets/96f101be-c18f-494f-bdd6-4550364c6644" />

Let's make the case definition.

**ICD-10:**
- C18.0–C18.9 — Malignant neoplasm of colon
- C19 — Malignant neoplasm of rectosigmoid junction. 
- C20 — Malignant neoplasm of rectum. 

**ICD-9:**
- 153.0–153.9 — Malignant neoplasm of colon.
- 154.0 — Malignant neoplasm of rectosigmoid junction.
- 154.1 — Malignant neoplasm of rectum.

```
cdr_id = os.environ["WORKSPACE_CDR"]
from google.cloud import bigquery as bq
codes=[f"C18.{i}" for i in range(10)]+["C19","C20"]+[f"153.{i}" for i in range(10)]+["154.0","154.1"]; codes=[c.upper() for c in (codes+[c.replace(".","") for c in codes])]
sql=f"""SELECT DISTINCT CAST(person_id AS STRING) person_id
        FROM `{cdr_id}.condition_occurrence`
        WHERE condition_source_value IS NOT NULL
          AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)"""
cases=bq.Client().query(sql, job_config=bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("codes","STRING",codes)])).to_dataframe()["person_id"].astype(str)
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
| PGS003852_AVG | 447,278 | 3,192 |  444,086 |     1.293 |  <1e-100 | 0.573 |    1.04e-45 |         0.57% |  <1e-100 |
| PGS003979_AVG | 447,278 | 3,192 |  444,086 |     1.294 |  <1e-100 | 0.571 |    5.83e-44 |         0.57% |  <1e-100 |
| PGS003433_AVG | 447,278 | 3,192 |  444,086 |     1.276 |  <1e-100 | 0.569 |    1.27e-41 |         0.52% |  <1e-100 |
| PGS000765_AVG | 447,278 | 3,192 |  444,086 |     1.211 |  <1e-100 | 0.554 |    1.28e-26 |         0.32% |  <1e-100 |
| PGS004904_AVG | 447,278 | 3,192 |  444,086 |     1.176 |  <1e-100 | 0.545 |    5.37e-19 |         0.23% |  <1e-100 |
| PGS003386_AVG | 447,278 | 3,192 |  444,086 |     1.095 | 1.63e-07 | 0.526 |    3.09e-07 |         0.07% | 3.29e-07 |
| PGS004303_AVG | 447,278 | 3,192 |  444,086 |     1.027 |    0.070 | 0.507 |       0.096 |         0.01% |    0.141 |

The GenoBoost method (PGS004303), which uses non-additive models, doesn't perform well relative to the others. The ICD-10 codes included in the construction of PGS004303 were similar to ours: C18,C19,C20.

The best performing scores are PGS003852, PGS003979, and PGS003433. One thing common to each of these scores is that the associated study involves colorectal cancer specifically (as opposed to being general and studying many diseases at once).

Let's add 16 principal components and sex to the model as predictors (not controls) and look at the resulting AUC.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t')
d['#IID'] = d['#IID'].astype(str)
d = d.set_index('#IID')
score_cols = [c for c in d.columns if c.endswith('_AVG')]

try:
    covars
except NameError:
    NUM_PCS = 16
    PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
    SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
    storage_opts = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
    pcs_raw = pd.read_csv(PCS_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","pca_features"])
    parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
    pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
    pc_df = pd.DataFrame(pcs_raw["pca_features"].apply(parse).to_list(), columns=pc_cols)\
              .assign(person_id=pcs_raw["research_id"].astype(str)).set_index("person_id")
    sex_df = pd.read_csv(SEX_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","dragen_sex_ploidy"])\
              .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"]\
              .map({"XX":0,"XY":1}).to_frame("sex").dropna()
    covars = pc_df.join(sex_df, how="inner").filter(pc_cols+["sex"]).dropna()

if 'cases' not in globals():
    raise NameError("Variable 'cases' not found.")
case_set = set(pd.Series(cases, dtype=str))

rows = []
for sc in score_cols:
    tmp = d[[sc]].join(covars, how='inner').dropna()
    if tmp.empty: 
        continue
    y = tmp.index.to_series().isin(case_set).astype('i1').to_numpy()
    X = tmp[["sex"] + pc_cols + [sc]].astype(float).to_numpy()
    mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    mdl.fit(X, y)
    auc = roc_auc_score(y, mdl.predict_proba(X)[:,1])
    rows.append({"score": sc.replace("_AVG",""), "n": int(len(tmp)), "cases": int(y.sum()), "controls": int(len(y)-y.sum()), "AUROC": float(auc)})

out = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
display(out)
```


| score     | cases | controls | AUROC    |
| :-------- | ----: | -------: | :------- |
| PGS003852 |  2985 |   410352 | 0.619005 |
| PGS003979 |  2985 |   410352 | 0.617833 |
| PGS003433 |  2985 |   410352 | 0.610325 |
| PGS000765 |  2985 |   410352 | 0.600296 |
| PGS004904 |  2985 |   410352 | 0.599918 |
| PGS003386 |  2985 |   410352 | 0.596961 |
| PGS004303 |  2985 |   410352 | 0.592681 |

This improves the scores somewhat, but there's a few problems. First, we typically want one prediction per individual, but now we have seven (one per score). Further, we're training on the same dataset we're evaluating on.

Now, let's test the same model, except jointly fitting all scores (except for PGS004303) using cross-validation to avoid overfitting.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs = [c for c in d.columns if c.endswith('_AVG') and c != 'PGS004303_AVG']

try:
    covars
except NameError:
    NUM_PCS=16; pc_cols=[f"PC{i}" for i in range(1,NUM_PCS+1)]
    so={"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
    pcs=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv", sep="\t", storage_options=so, usecols=["research_id","pca_features"])
    parse=lambda s,k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
    pc_df=pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols).assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")
    sex=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv", sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
    sex_df=sex.assign(person_id=sex["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"].map({"XX":0,"XY":1}).to_frame("sex").dropna()
    covars=pc_df.join(sex_df, how="inner").filter(pc_cols+["sex"]).dropna()

covars = covars.copy()
covars.index = covars.index.astype(str)

if 'cases' not in globals(): raise NameError("Variable 'cases' not found.")
case_set=set(pd.Series(cases, dtype=str))

Xy = covars.join(d[pgs], how="inner").dropna()
y = Xy.index.to_series().isin(case_set).astype('i1').to_numpy()
X = Xy.to_numpy()

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:,1]
auc = roc_auc_score(y, p)

pd.DataFrame([{"n": int(len(y)), "cases": int(y.sum()), "controls": int(len(y)-y.sum()), "predictors": X.shape[1], "AUROC": float(auc)}])
```

This yields an AUROC of 0.617375. (Adding a ridge penalty gives a similar value of 0.617.)

Let's check the colorectal cancer-related ICD codes individually.

```
import os
from google.cloud import bigquery as bq
import matplotlib.pyplot as plt

cdr_id=os.environ["WORKSPACE_CDR"]
codes=[f"C18.{i}" for i in range(10)]+["C19","C20"]+[f"153.{i}" for i in range(10)]+["154.0","154.1"]
codes=[c.upper() for c in (codes+[c.replace(".","") for c in codes])]
sql=f"""
WITH code_list AS (
  SELECT DISTINCT UPPER(code) AS code FROM UNNEST(@codes) AS code
),
norm_codes AS (
  SELECT ANY_VALUE(code) AS code, REPLACE(code,'.','') AS code_n
  FROM code_list
  GROUP BY code_n
),
cond AS (
  SELECT CAST(person_id AS STRING) person_id,
         REPLACE(UPPER(TRIM(condition_source_value)),'.','') AS csv_n
  FROM `{cdr_id}.condition_occurrence`
  WHERE condition_source_value IS NOT NULL
),
joined AS (
  SELECT nc.code AS code, c.person_id
  FROM norm_codes nc
  JOIN cond c ON c.csv_n = nc.code_n
)
SELECT code, COUNT(DISTINCT person_id) AS n_persons
FROM joined
GROUP BY code
ORDER BY n_persons DESC, code
"""
code_counts=bq.Client().query(sql, job_config=bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("codes","STRING",codes)])).to_dataframe()
print(code_counts.to_string(index=False))
print(code_counts[code_counts["n_persons"]<20].sort_values(["n_persons","code"]).to_string(index=False))
df=code_counts[code_counts["n_persons"]>=20].sort_values("n_persons",ascending=True)
plt.figure(figsize=(8, max(2, 0.3*len(df))))
plt.barh(df["code"], df["n_persons"])
plt.xlabel("Distinct persons")
plt.ylabel("ICD code")
plt.title("CRC ICD code counts (≥20 cases)")
plt.tight_layout(); plt.show()
```
<img width="790" height="679" alt="image" src="https://github.com/user-attachments/assets/2e1d2983-d9e9-4268-a378-699b2f0cfd04" />

We haven't assessed a few important codes.

ICD-10
- Z85.038 — Personal history of malignant neoplasm of large intestine.
- Z85.04x — Personal history of malignant neoplasm of rectum/rectosigmoid junction/anal canal.

```
cdr_id=os.environ["WORKSPACE_CDR"]
codes_exact=["Z85.038","V10.05"]
codes_exact=[c.upper() for c in (codes_exact+[c.replace(".","") for c in codes_exact])]

sql=f"""
SELECT DISTINCT CAST(person_id AS STRING) person_id
FROM `{cdr_id}.observation`
WHERE observation_source_value IS NOT NULL
  AND (
    UPPER(TRIM(observation_source_value)) IN UNNEST(@codes_exact)
    OR REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]','') IN UNNEST(@codes_exact)
    OR STARTS_WITH(REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]',''), @z8503_prefix)
  )
"""
cases=bq.Client().query(
    sql,
    job_config=bq.QueryJobConfig(query_parameters=[
        bq.ArrayQueryParameter("codes_exact","STRING",codes_exact),
        bq.ScalarQueryParameter("z8503_prefix","STRING","Z8503"),
    ])
).to_dataframe()["person_id"].astype(str)
```

PGS003852 has an accuracy by itself of 0.619311 for this new case definition, while the multi-PGS + covariates model has a cross-validated AUC of 0.648.

Let's plot risk in each decline of the multivariate model.
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d=pd.read_csv('../../arrays.sscore',sep='\t').set_index('#IID'); d.index=d.index.astype(str)
pgs=[c for c in d.columns if c.endswith('_AVG') and c!='PGS004303_AVG']

if 'covars' not in globals():
    NUM_PCS=16; pc_cols=[f"PC{i}" for i in range(1,NUM_PCS+1)]
    so={"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
    pcs=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",sep="\t",storage_options=so,usecols=["research_id","pca_features"])
    parse=lambda s,k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")]+[np.nan]*k)[:k]
    pc_df=pd.DataFrame(pcs["pca_features"].apply(parse).to_list(),columns=pc_cols).assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")
    sex=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",sep="\t",storage_options=so,usecols=["research_id","dragen_sex_ploidy"])
    sex_df=sex.assign(person_id=sex["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"].map({"XX":0,"XY":1}).to_frame("sex").dropna()
    covars=pc_df.join(sex_df,how="inner").filter(pc_cols+["sex"]).dropna()
covars=covars.copy(); covars.index=covars.index.astype(str)

if 'cases' not in globals(): raise NameError("Variable 'cases' not found.")
case_set=set(pd.Series(cases,dtype=str))

Xy=covars.join(d[pgs],how="inner").dropna()
y=Xy.index.to_series().isin(case_set).astype('i1').to_numpy()
X=Xy.to_numpy()

cv=StratifiedKFold(n_splits=15,shuffle=True,random_state=42)
mdl=make_pipeline(StandardScaler(),LogisticRegression(solver="lbfgs",max_iter=1000))
p=cross_val_predict(mdl,X,y,cv=cv,method="predict_proba")[:,1]
auc=roc_auc_score(y,p)
print(pd.DataFrame([{"n":int(len(y)),"cases":int(y.sum()),"controls":int(len(y)-y.sum()),"predictors":X.shape[1],"AUROC":float(auc)}]).to_string(index=False))

dec=pd.qcut(p,10,labels=False,duplicates='drop')+1
g=pd.DataFrame({"decile":dec,"y":y,"p":p}).groupby("decile").agg(n=("y","size"),cases=("y","sum"),obs_risk=("y","mean"),pred_risk=("p","mean")).reset_index()
print(g[["decile","n","cases","obs_risk","pred_risk"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

plt.figure(figsize=(7,4.5))
plt.bar(g["decile"],g["obs_risk"],width=0.9,label="Observed",color="#4C78A8")
plt.plot(g["decile"],g["pred_risk"],marker="o",linewidth=2,label="Predicted",color="#F58518")
plt.xlabel("Risk decile (low → high)"); plt.ylabel("Event rate"); plt.title("Decile risk: observed vs predicted")
plt.xticks(g["decile"]); plt.legend(); plt.tight_layout(); plt.show()
```

<img width="690" height="440" alt="image" src="https://github.com/user-attachments/assets/d5e0e3ac-2028-4553-be52-5cc328ef6b94" />

Let's get the per-code OR/SD:

```
import os, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt
from scipy.stats import norm
from google.cloud import bigquery as bq

cdr_id=os.environ["WORKSPACE_CDR"]
pgs_col="PGS003852_AVG"

cond_codes=[f"C18.{i}" for i in range(10)]+["C19","C20"]+[f"153.{i}" for i in range(10)]+["154.0","154.1"]
cond_codes=[c.upper() for c in (cond_codes+[c.replace(".","") for c in cond_codes])]
cond_codes_n=[c.replace(".","") for c in cond_codes]
codes_exact=["Z85.038","V10.05"]; codes_exact=[c.upper() for c in (codes_exact+[c.replace(".","") for c in codes_exact])]

sql=f"""
WITH obs_raw AS (
  SELECT DISTINCT CAST(person_id AS STRING) person_id,
         REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]','') AS code_n
  FROM `{cdr_id}.observation`
  WHERE observation_source_value IS NOT NULL
    AND (
      UPPER(TRIM(observation_source_value)) IN UNNEST(@codes_exact)
      OR REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]','') IN UNNEST(@codes_exact)
      OR STARTS_WITH(REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]',''),'Z8503')
    )
),
obs AS (
  SELECT person_id,
         CASE
           WHEN code_n='V1005' THEN 'V10.05'
           WHEN code_n='Z85038' THEN 'Z85.038'
           WHEN REGEXP_CONTAINS(code_n, r'^Z8503[0-9]$') THEN CONCAT('Z85.03', SUBSTR(code_n,6,1))
           ELSE code_n
         END AS code
  FROM obs_raw
),
cond_raw AS (
  SELECT DISTINCT CAST(person_id AS STRING) person_id,
         REGEXP_REPLACE(UPPER(TRIM(condition_source_value)),'[^A-Z0-9]','') AS code_n
  FROM `{cdr_id}.condition_occurrence`
  WHERE condition_source_value IS NOT NULL
    AND (
      UPPER(TRIM(condition_source_value)) IN UNNEST(@cond_codes)
      OR REGEXP_REPLACE(UPPER(TRIM(condition_source_value)),'[^A-Z0-9]','') IN UNNEST(@cond_codes_n)
    )
),
cond AS (
  SELECT person_id,
         CASE
           WHEN REGEXP_CONTAINS(code_n, r'^C18[0-9]$') THEN CONCAT('C18.', SUBSTR(code_n,4,1))
           WHEN code_n='C19' THEN 'C19'
           WHEN code_n='C20' THEN 'C20'
           WHEN REGEXP_CONTAINS(code_n, r'^153[0-9]$') THEN CONCAT('153.', SUBSTR(code_n,4,1))
           WHEN code_n IN ('1540','1541') THEN CONCAT('154.', SUBSTR(code_n,4,1))
           ELSE code_n
         END AS code
  FROM cond_raw
),
all_codes AS (
  SELECT 'OBS' AS src, code, person_id FROM obs WHERE code IS NOT NULL
  UNION ALL
  SELECT 'COND' AS src, code, person_id FROM cond WHERE code IS NOT NULL
)
SELECT src, code, person_id FROM all_codes
"""
cp=bq.Client().query(sql, job_config=bq.QueryJobConfig(query_parameters=[
    bq.ArrayQueryParameter("codes_exact","STRING",codes_exact),
    bq.ArrayQueryParameter("cond_codes","STRING",cond_codes),
    bq.ArrayQueryParameter("cond_codes_n","STRING",cond_codes_n),
])).to_dataframe()

df=pd.read_csv('../../arrays.sscore', sep='\t')
idcol=next((c for c in ['#IID','IID','person_id','research_id','sample_id','ID'] if c in df.columns), None)
if idcol is None: raise ValueError("ID column not found in ../../arrays.sscore")
df[idcol]=df[idcol].astype(str); df=df.set_index(idcol)
x=pd.to_numeric(df[pgs_col], errors='coerce'); x=x.dropna(); ids=pd.Index(x.index.astype(str))

rows=[]
for code in sorted(cp['code'].unique()):
    pid=set(cp.loc[cp['code']==code, 'person_id'].astype(str))
    idx=ids
    y=idx.to_series().isin(pid).astype(int).to_numpy()
    xv=x.reindex(idx).to_numpy()
    m=float(np.nanmean(xv)); s=float(np.nanstd(xv))
    if not np.isfinite(s) or s==0: continue
    z=(xv-m)/s
    n=len(z); n1=int(y.sum())
    if n1==0 or n1==n: continue
    try:
        X=sm.add_constant(pd.DataFrame({'z':z}))
        fit=sm.Logit(y, X).fit(disp=0, maxiter=200)
        beta=float(fit.params['z']); se=float(fit.bse['z'])
    except Exception:
        try:
            fit=sm.GLM(y, X, family=sm.families.Binomial()).fit()
            beta=float(fit.params['z']); se=float(fit.bse['z'])
        except Exception:
            continue
    OR=float(np.exp(beta))
    zstat=beta/se if se>0 else np.nan
    p=float(norm.sf(zstat)) if np.isfinite(zstat) else np.nan
    rows.append({'code':code,'n':n,'cases':n1,'OR_perSD':OR,'p_one_sided':p})

res=pd.DataFrame(rows)
res=res[res['n']>=30].dropna(subset=['OR_perSD']).sort_values('OR_perSD', ascending=True)

plt.figure(figsize=(9, max(3, 0.5*len(res))))
bars=plt.barh(res['code'], res['OR_perSD'])
xmax=float(res['OR_perSD'].max()*1.15) if len(res) else 1.0
plt.xlim(0, xmax)
for rect,pv in zip(bars, res['p_one_sided']):
    txt = f"p={pv:.1e}" if np.isfinite(pv) else "p=n/a"
    plt.text(rect.get_width()*1.01, rect.get_y()+rect.get_height()/2, txt, va="center")
plt.xlabel("OR per SD (PGS003852)"); plt.ylabel("ICD code"); plt.title("PGS003852 OR/SD by ICD code (n≥30)")
plt.tight_layout(); plt.show()
```

<img width="889" height="1339" alt="image" src="https://github.com/user-attachments/assets/3f9a820e-1632-471a-b2da-ac802d009401" />


