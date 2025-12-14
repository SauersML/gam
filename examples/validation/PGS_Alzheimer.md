## Running a polygenic score for Alzheimer's disease on the _All of Us_ cohort

Cognitive decline has a heterogeneous etiology, and may be related to Alzheimer's disease, dementia, or other genetic and environmental causes. This is complicated by the age of individuals in the biobank.

Let's view the polygenic score catalog for dementia:
https://www.pgscatalog.org/trait/MONDO_0001627/

And Alzheimer's disease:
https://www.pgscatalog.org/trait/MONDO_0004975/


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
- PGS004146
- PGS004863
- PGS003957
- PGS003334
- PGS004589
- PGS004898
- PGS004227

These also contain some scores for breast cancer, which I forgot to remove from a list from a previous analysis.

We will use `gs://fc-aou-datasets-controlled/v8/microarray/plink/`, though it may have higher missingness compared to WGS.

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
./gnomon/target/release/gnomon score "PGS000332,PGS000335,PGS000015,PGS000508,PGS000344,PGS000317,PGS000007,PGS000507,PGS004869,PGS004146,PGS004863,PGS003957,PGS003334,PGS004589,PGS004898,PGS004227" arrays
```

This should take 95 minutes to run, and output a file called:
```
arrays.sscore
```
Let's open a Jupyter analysis notebook.
```
!head ../../arrays.sscore
```

This shows us the columns of the score output:
```
#IID	PGS000007_AVG	PGS000007_MISSING_PCT	PGS000015_AVG	PGS000015_MISSING_PCT	PGS000317_AVG	PGS000317_MISSING_PCT	PGS000332_AVG	PGS000332_MISSING_PCT	PGS000335_AVG	PGS000335_MISSING_PCT	PGS000344_AVG	PGS000344_MISSING_PCT	PGS000507_AVG	PGS000507_MISSING_PCT	PGS000508_AVG	PGS000508_MISSING_PCT	PGS003334_AVG	PGS003334_MISSING_PCT	PGS003957_AVG	PGS003957_MISSING_PCT	PGS004146_AVG	PGS004146_MISSING_PCT	PGS004227_AVG	PGS004227_MISSING_PCT	PGS004589_AVG	PGS004589_MISSING_PCT	PGS004863_AVG	PGS004863_MISSING_PCT	PGS004869_AVG	PGS004869_MISSING_PCT	PGS004898_AVG	PGS004898_MISSING_PCT
```

Let's plot missingness for each score:
```
import pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv('../../arrays.sscore', sep='\t')
for c in d.columns[d.columns.str.endswith('_MISSING_PCT')]: d[c].hist(bins=50); plt.title(c); plt.xlabel(c); plt.ylabel('count'); plt.show()
```

Most samples' missingness should be under 2%. But there are a few scores that have high missingness for some samples:

<img width="597" height="454" alt="image" src="https://github.com/user-attachments/assets/493766a6-870c-4ed4-8481-9eaa1ead61af" />
<img width="597" height="454" alt="image" src="https://github.com/user-attachments/assets/f345c05b-7025-4b96-924b-f7dbd422b401" />
<img width="597" height="454" alt="image" src="https://github.com/user-attachments/assets/66b15e0c-9ef3-4adb-98ea-da892886c71d" />
<img width="597" height="454" alt="image" src="https://github.com/user-attachments/assets/5d9139cd-d6e9-4aaf-a466-46c7c25eec15" />

Now let's check the correlation matrix of the scores. We expect moderate correlations.

```
import pandas as pd, matplotlib.pyplot as plt
c=pd.read_csv('../../arrays.sscore',sep='\t').filter(regex='_AVG$').corr()
plt.figure(figsize=(8,8)); im=plt.imshow(c,vmin=-1,vmax=1); plt.colorbar(im); plt.xticks(range(len(c)),c.columns,rotation=90); plt.yticks(range(len(c)),c.columns); plt.tight_layout(); plt.show()
```

<img width="780" height="758" alt="image" src="https://github.com/user-attachments/assets/54cb4a28-f73c-4b5b-8795-d99d45c0ed57" />

Let's make the case definition.


ICD-10:
R41.81
G31.84
F06.70
F06.71
F03.90
F03.911
F03.918
F03.92
F03.93
F03.94
F03.A0
F03.A11
F03.A18
F03.A2
F03.A3
F03.A4
F03.B0
F03.B11
F03.B18
F03.B2
F03.B3
F03.B4
F03.C0
F03.C11
F03.C18
F03.C2
F03.C3
F03.C4
F02.80
F02.811
F02.818
F02.82
F02.83
F02.84
F02.A0
F02.A11
F02.A18
F02.A2
F02.A3
F02.A4
F02.B0
F02.B11
F02.B18
F02.B2
F02.B3
F02.B4
F02.C0
F02.C11
F02.C18
F02.C2
F02.C3
F02.C4
F01.50
F01.511
F01.518
F01.52
F01.53
F01.54
F01.A0
F01.A11
F01.A18
F01.A2
F01.A3
F01.A4
F01.B0
F01.B11
F01.B18
F01.B2
F01.B3
F01.B4
F01.C0
F01.C11
F01.C18
F01.C2
F01.C3
F01.C4
G30.0
G30.1
G30.8
G30.9

ICD-9:
331.83
294.20
294.21
294.10
294.11
331.0
290.0
290.10
290.11
290.12
290.13
290.20
290.21
290.3
290.40
290.41
290.42
290.43

This is too many ICD codes, which makes both modeling and interpretation more difficult. Let's use Alzheimer's codes only instead:

```
import os
from google.cloud import bigquery as bq

cdr_id = os.environ["WORKSPACE_CDR"]

icd10_alz = [
    "G30.0",
    "G30.1",
    "G30.8",
    "G30.9",
]

icd9_alz = [
    "331.0",
]

codes = icd10_alz + icd9_alz
codes = [c.upper() for c in (codes + [c.replace(".", "") for c in codes])]

sql = f"""
SELECT DISTINCT CAST(person_id AS STRING) AS person_id
FROM `{cdr_id}.condition_occurrence`
WHERE condition_source_value IS NOT NULL
  AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
"""

cases = (
    bq.Client()
    .query(
        sql,
        job_config=bq.QueryJobConfig(
            query_parameters=[bq.ArrayQueryParameter("codes", "STRING", codes)]
        ),
    )
    .to_dataframe()["person_id"]
    .astype(str)
)
```

For simplicity, we aren't using a time-to-event model. Some of our "controls" will go on to develop colorectal cancer later in life. This is important to keep in mind since the phenotype is age-related.

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

Done.

| Score ID      | N (Total) | Cases | Controls | OR per SD | One-sided p (OR>1) | AUROC | One-sided p (AUC>0.5) |
| ------------- | --------- | ----- | -------- | --------- | ------------------ | ----- | --------------------- |
| PGS004146_AVG | 447278    | 1083  | 446195   | 1.318     | 0                  | 0.574 | 2.28e-17              |
| PGS004898_AVG | 447278    | 1083  | 446195   | 1.158     | 7.82e-07           | 0.540 | 2.60e-06              |
| PGS000015_AVG | 447278    | 1083  | 446195   | 0.875     | 1.000              | 0.460 | 1.000                 |
| PGS000332_AVG | 447278    | 1083  | 446195   | 0.886     | 1.000              | 0.469 | 1.000                 |
| PGS000508_AVG | 447278    | 1083  | 446195   | 0.887     | 1.000              | 0.471 | 0.999                 |
| PGS004869_AVG | 447278    | 1083  | 446195   | 0.892     | 1.000              | 0.465 | 1.000                 |
| PGS000507_AVG | 447278    | 1083  | 446195   | 0.894     | 1.000              | 0.472 | 0.999                 |
| PGS000344_AVG | 447278    | 1083  | 446195   | 0.904     | 1.000              | 0.471 | 0.999                 |
| PGS003334_AVG | 447278    | 1083  | 446195   | 1.084     | 0.004              | 0.519 | 0.018                 |
| PGS000335_AVG | 447278    | 1083  | 446195   | 0.924     | 0.995              | 0.483 | 0.970                 |
| PGS000007_AVG | 447278    | 1083  | 446195   | 0.930     | 0.991              | 0.480 | 0.988                 |
| PGS000317_AVG | 447278    | 1083  | 446195   | 0.944     | 0.972              | 0.482 | 0.980                 |
| PGS004589_AVG | 447278    | 1083  | 446195   | 1.048     | 0.064              | 0.512 | 0.083                 |
| PGS003957_AVG | 447278    | 1083  | 446195   | 0.960     | 0.908              | 0.505 | 0.286                 |
| PGS004863_AVG | 447278    | 1083  | 446195   | 1.038     | 0.110              | 0.509 | 0.151                 |
| PGS004227_AVG | 447278    | 1083  | 446195   | 0.992     | 0.600              | 0.499 | 0.553                 |

PGS004146 is the best. The p-values are one-sided, but it's likely that the effect direction is reversed in some of these scores. Take PGS000015_AVG, for example. OR per SD shows substantially decreased risk per SD of the score, with an AUROC under 0.5. If we set a simple +/- 0.1 OR change per SD, we are left with: PGS004146_AVG, PGS004898_AVG, PGS000015_AVG, PGS000332_AVG, PGS000508_AVG, PGS004869_AVG, and PGS000507_AVG.

Let's do decile plots for each score:

```
import pandas as pd, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm

# Only these scores
keep = [
    "PGS004146_AVG", "PGS004898_AVG", "PGS000015_AVG",
    "PGS000332_AVG", "PGS000508_AVG", "PGS004869_AVG", "PGS000507_AVG",
]

d = pd.read_csv('../../arrays.sscore', sep='\t')
d['#IID'] = d['#IID'].astype(str)
d = d.set_index('#IID')
cols = [c for c in keep if c in d.columns]

y = d.index.to_series().isin(pd.Series(cases, dtype=str)).astype('i1')

def or_per_sd_within_deciles(x, y, q=10):
    t = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': y}).dropna()
    if t.empty:
        return pd.Series(dtype=float)
    bins = pd.qcut(t['x'], q=q, labels=False, duplicates='drop')
    out = []
    for dec in range(int(bins.min()), int(bins.max()) + 1):
        idx = bins.index[bins == dec]
        xi = t.loc[idx, 'x']; yi = t.loc[idx, 'y']
        if yi.nunique() < 2:
            out.append(np.nan); continue
        sd = xi.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out.append(np.nan); continue
        z = (xi - xi.mean()) / sd
        m = sm.Logit(yi.to_numpy(), sm.add_constant(z.to_numpy())).fit(disp=False, maxiter=200)
        out.append(float(np.exp(m.params[1])))
    # 1-based decile indexing
    return pd.Series(out, index=np.arange(int(bins.min()) + 1, int(bins.max()) + 2))

plt.figure(figsize=(9, 6))
for c in cols:
    seq = or_per_sd_within_deciles(d[c], y)
    if not seq.empty:
        plt.plot(seq.index, seq.values, marker='o', label=c.replace('_AVG',''))
plt.axhline(1.0, linewidth=1)
plt.xlabel('Decile of raw score (per-score quantiles)')
plt.ylabel('OR per SD within decile')
plt.title('OR/SD by decile — raw PGS only')
plt.legend()
plt.tight_layout()
plt.show()
```

<img width="889" height="590" alt="image" src="https://github.com/user-attachments/assets/4e4243b8-ab66-41b2-97a1-4d876d0985f5" />

This is not great. We should check if these scores are useful using a two-sided test. Let's compare the top 10% to the bottom 90% for each score. To avoid making assumptions about score construction, we might as well also test the top 90% and bottom 10%.

```
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import Table2x2
import numpy as np, pandas as pd

S = cols if 'cols' in globals() else [c for c in d.columns if c.endswith('_AVG')]
fmt = lambda v: np.format_float_positional(float(v), trim='-')

def do_test(mask, other_mask, label):
    n1, n0 = int(mask.sum()), int(other_mask.sum())
    if n1 == 0 or n0 == 0:
        return None
    a = int(y[mask].sum()); b = n1 - a; c = int(y[other_mask].sum()); d0 = n0 - c
    z, p = proportions_ztest([a, c], [n1, n0], alternative='two-sided')
    A, B, C, D = (a, b, c, d0)
    if min(A, B, C, D) == 0:
        A += 0.5; B += 0.5; C += 0.5; D += 0.5
    t = Table2x2([[A, B], [C, D]])
    OR = float(t.oddsratio); lo, hi = map(float, t.oddsratio_confint())
    r1, r0 = a / n1, c / n0
    return [label, n1, a, r1, n0, c, r0, r1 - r0, float(z), fmt(p), OR, lo, hi,
            'higher' if r1 > r0 else ('lower' if r1 < r0 else 'no diff')]

rows = []
for s in S:
    x = pd.to_numeric(d[s], errors='coerce'); m = x.notna()
    q90, q10 = np.nanquantile(x[m], 0.90), np.nanquantile(x[m], 0.10)
    top = (x >= q90) & m; rest_for_top = (~top) & m
    bot = (x <= q10) & m; rest_for_bot = (~bot) & m
    r1 = do_test(top, rest_for_top, f"{s.replace('_AVG','')} | top10_vs_rest")
    r0 = do_test(bot, rest_for_bot, f"{s.replace('_AVG','')} | bottom10_vs_rest")
    if r1: rows.append(r1)
    if r0: rows.append(r0)

out = pd.DataFrame(rows, columns=[
    'score|comparison','n_group','cases_group','rate_group',
    'n_other','cases_other','rate_other','rate_diff',
    'z','p_two_sided','odds_ratio','or_95ci_low','or_95ci_high','direction'
]).sort_values('p_two_sided', key=lambda s: s.astype(float))

out
```

### Top 10% vs rest

| Score ID  | Two-sided p | Odds ratio | Direction |
| --------- | ----------- | ---------- | --------- |
| PGS004146 | 1.34e-11    | 1.737      | higher    |
| PGS004898 | 7.40e-05    | 1.418      | higher    |
| PGS000332 | 0.002       | 0.688      | lower     |
| PGS000507 | 0.002       | 0.688      | lower     |
| PGS000508 | 0.002       | 0.698      | lower     |
| PGS004869 | 0.040       | 0.796      | lower     |
| PGS000015 | 0.177       | 0.865      | lower     |

### Bottom 10% vs rest

| Score ID  | Two-sided p | Odds ratio | Direction |
| --------- | ----------- | ---------- | --------- |
| PGS004146 | 1.13e-07    | 0.490      | lower     |
| PGS000015 | 1.98e-04    | 1.393      | higher    |
| PGS000507 | 4.33e-04    | 1.370      | higher    |
| PGS000508 | 0.001       | 1.337      | higher    |
| PGS000332 | 0.007       | 1.282      | higher    |
| PGS004869 | 0.007       | 1.282      | higher    |
| PGS004898 | 0.010       | 0.746      | lower     |

It was a good idea to do both the top 10% and bottom 10%: PGS000015, for example, is only significant if we consider the bottom 10% vs. the rest.

Let's add 16 principal components and sex to the model as predictors (not controls) and look at the resulting AUC.

Let's drop the remaining scores not used in the above.

```
bad = ["PGS000007","PGS000317","PGS000335","PGS000344","PGS003334","PGS003957","PGS004227","PGS004589","PGS004863"]
t = pd.read_csv('../../arrays_cog.sscore', sep='\t')
t = t[[c for c in t.columns if not any(c.startswith(b + '_') for b in bad)]]
t.to_csv('../../arrays_cog.filtered.sscore', sep='\t', index=False)  # writes cleaned file
```

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

From this, we can see that everything is helpful: score, sex, and PCs. The best AUROC is from PGS004146 + sex + PCs, at 0.630.

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

This yields an AUROC of 0.623147, similar to before.

Do score ensembles really improve the result?

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
X_507 = d.loc[idx_all, ['PGS004146_AVG']].astype(float)
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
    {"model":"PGS004146 only", "n":len(y), "cases":int(y.sum()), "controls":int(len(y)-y.sum()), "AUROC":float(auc_507)},
]).sort_values("AUROC", ascending=False).reset_index(drop=True)

diff =  np.array(auc_all_f) - np.array(auc_507_f)
nz = diff[diff!=0]
s = int((nz>0).sum()); n = int(len(nz))
p_one_sided = sum(math.comb(n,k) for k in range(s, n+1)) / (2**n) if n>0 else np.nan

print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"One-sided sign-test p (PGS004146 < All AVG): {p_one_sided:.3g} over {s}/{n} folds; ΔAUROC (OOF) = {auc_all-auc_507:+.3f}")
```

We see that the AUC of the ensemble (0.586) is significantly better than that of the single score (0.573) at p = 0.00369.

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
single_col = "PGS004146_AVG"

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

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/6ee3c9d8-2009-4d05-9f81-2d1e925027bc" />

This is much more stable than the per-score decile plots.

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
d = pd.read_csv('../../arrays_cog.sscore', sep='\t').set_index('#IID')
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

PGS ensemble + PCs + sex:
- 0.625009 AUROC
- 1.399734 OR/SD
- 5.43% at 2x risk

Before, we saw that high missingness affected some samples. Let's check if missingness affects accuracy per score:
```
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

if '#IID' in d.columns:
    ids = d['#IID'].astype(str)
    y_series = ids.isin(set(pd.Series(cases, dtype=str))).astype('i1')
    y_series.index = d.index
else:
    idx = pd.Index(d.index.astype(str))
    y_series = pd.Series(idx.isin(set(pd.Series(cases, dtype=str))).astype('i1'), index=d.index)

rows = []
for s in [c for c in d.columns if c.endswith('_AVG')]:
    mcol = s.replace('_AVG', '_MISSING_PCT')
    if mcol not in d.columns:
        continue
    x = pd.to_numeric(d[s], errors='coerce')
    miss = pd.to_numeric(d[mcol], errors='coerce')
    t = pd.DataFrame({'x': x, 'miss': miss, 'y': y_series}).dropna()
    if len(t) < 3 or t['y'].nunique() < 2 or t['x'].std(ddof=0) == 0 or t['miss'].std(ddof=0) == 0:
        rows.append([s.replace('_AVG',''), int(len(t)), np.nan, np.nan])
        continue

    auc = roc_auc_score(t['y'], t['x'])
    xo = (-t['x']) if auc < 0.5 else t['x']
    pred = (xo > xo.median()).astype('i1')
    acc = (pred == t['y']).astype('i1')
    if acc.nunique() < 2:
        r, p = np.nan, np.nan
    else:
        r, p = pearsonr(t['miss'].to_numpy(), acc.to_numpy())

    rows.append([s.replace('_AVG',''), int(len(t)), float(r), float(p)])

out = pd.DataFrame(rows, columns=['score','n','pearson_r_missing_vs_accuracy','p_value_raw']) \
        .sort_values('p_value_raw', na_position='last') \
        .reset_index(drop=True)
out['p_value'] = out['p_value_raw'].map(lambda v: '' if pd.isna(v) else np.format_float_positional(v, trim='-'))
out[['score','n','pearson_r_missing_vs_accuracy','p_value']]
```
| Score     | N      | Pearson r (missing vs accuracy) | p-value                           |
| --------- | ------ | ------------------------------- | --------------------------------- |
| PGS003334 | 447278 | -0.068224                       | ≈ 0                                 |
| PGS004146 | 447278 | -0.043374                       | ≈ 0  |
| PGS000015 | 447278 | 0.017543                        | 8.60183791610128e-35              |
| PGS004589 | 447278 | 0.013929                        | 1.2085366582732694e-20            |
| PGS003957 | 447278 | 0.009972                        | 2.5712636769050822e-11            |
| PGS000317 | 447278 | -0.006459                       | 1.5614608261813486e-05            |
| PGS000508 | 447278 | 0.005580                        | 0.0001899198792989028             |
| PGS000335 | 447278 | 0.004918                        | 0.0010049042456912638             |
| PGS000507 | 447278 | 0.004231                        | 0.004663657541919483              |
| PGS000344 | 447278 | 0.004183                        | 0.005152200021396543              |
| PGS004869 | 447278 | 0.003794                        | 0.011165516697301309              |
| PGS004898 | 447278 | 0.003043                        | 0.041806129699096355              |
| PGS000332 | 447278 | 0.002060                        | 0.16834350526542918               |
| PGS004863 | 447278 | 0.001538                        | 0.3037079706476949                |
| PGS004227 | 447278 | -0.001128                       | 0.4505731903914835                |
| PGS000007 | 447278 | 0.000455                        | 0.7607642498413164                |

This makes sense for the negative correlations. But why do many have positive correlations, in which more missingness results in higher accuracy? That is odd.

Let's test in a different way.
```
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve

case_set = set(pd.Series(cases, dtype=str))
ids = d['#IID'].astype(str) if '#IID' in d.columns else pd.Series(d.index.astype(str), index=d.index)
y_series = ids.isin(case_set).astype('i1')

rows = []
for s in [c for c in d.columns if c.endswith('_AVG')]:
    mcol = s.replace('_AVG','_MISSING_PCT')
    if mcol not in d.columns:
        continue
    x = pd.to_numeric(d[s], errors='coerce')
    miss = pd.to_numeric(d[mcol], errors='coerce')
    ok = x.notna() & miss.notna()
    if ok.sum() < 3 or y_series[ok].nunique() < 2:
        rows.append([s.replace('_AVG',''), int(ok.sum()), np.nan, ''])
        continue

    yv = y_series[ok]
    auc = roc_auc_score(yv, x[ok])
    xo = (-x) if auc < 0.5 else x

    fpr, tpr, thr = roc_curve(yv, xo[ok])
    tstar = thr[np.argmax(tpr - fpr)]
    pred = (xo[ok] >= tstar).astype('i1')
    correct = (pred == yv).astype('i1')

    if correct.nunique() < 2 or miss[ok].std(ddof=0) == 0:
        r, p = np.nan, ''
    else:
        r, p_raw = pearsonr(miss[ok].to_numpy(), correct.to_numpy())
        r = float(r)
        p = np.format_float_positional(float(p_raw), trim='-')

    rows.append([s.replace('_AVG',''), int(ok.sum()), r, p])

out = pd.DataFrame(rows, columns=['score','n','pearson_r_missing_vs_correct','p_value']).sort_values(
    'p_value', key=lambda s: pd.to_numeric(s, errors='coerce'), na_position='last'
)
out
```

Hmmm, we still see positive correlations between missingness and accuracy. PGS000015 and PGS004589 have r=0.02 for accuracy vs. missingness.

Perhaps genetic ancestry is causing a bias?

```
# Correlate per-individual missingness with ancestry PCs for PGS000015 and PGS004589
scores = ["PGS000015", "PGS004589"]
ids = d["#IID"].astype(str) if "#IID" in d.columns else pd.Series(d.index.astype(str), index=d.index)
pcs = [c for c in pc_cols if c in pc_df.columns]

rows = []
for s in scores:
    mcol = f"{s}_MISSING_PCT"
    if mcol not in d.columns or not pcs:
        continue
    miss = pd.to_numeric(d[mcol], errors="coerce")
    miss_df = pd.DataFrame({"miss": miss.values}, index=ids)
    X = pc_df[pcs].join(miss_df, how="inner").dropna()
    if X.empty or X["miss"].std(ddof=0) == 0:
        continue
    for pc in pcs:
        r, p = pearsonr(X["miss"].to_numpy(), X[pc].to_numpy())
        rows.append([s, pc, int(len(X)), float(r), np.format_float_positional(float(p), trim='-')])

out = pd.DataFrame(rows, columns=["score","PC","n","pearson_r","p_value"]) \
       .sort_values(["score", "PC"]).reset_index(drop=True)
out
```

Indeed, there are many correlations between PCs and missingness.

Does the correlation between accuracy and missingness hold up after controlling for PCs?
```
case_set = set(pd.Series(cases, dtype=str))
ids = d['#IID'].astype(str) if '#IID' in d.columns else pd.Series(d.index.astype(str), index=d.index)
y_series = ids.isin(case_set).astype('i1')
pcs_use = [c for c in pc_cols if c in pc_df.columns]

rows = []
for s in [c for c in d.columns if c.endswith('_AVG')]:
    mcol = s.replace('_AVG','_MISSING_PCT')
    if mcol not in d.columns or not pcs_use:
        continue

    x = pd.to_numeric(d[s], errors='coerce')
    miss = pd.to_numeric(d[mcol], errors='coerce')
    base = pd.DataFrame({'x': x, 'miss': miss, 'y': y_series, 'person_id': ids}).dropna(subset=['x','miss','y'])
    if base.empty:
        continue

    T = (base.set_index('person_id')
               .join(pc_df[pcs_use], how='inner')
               .dropna())
    if len(T) < 50 or T['y'].nunique() < 2 or T['miss'].std(ddof=0) == 0:
        rows.append([s.replace('_AVG',''), int(len(T)), np.nan, np.nan, np.nan, ''])
        continue

    auc = roc_auc_score(T['y'], T['x'])
    xo = (-T['x']) if auc < 0.5 else T['x']
    fpr, tpr, thr = roc_curve(T['y'], xo)
    tstar = thr[np.argmax(tpr - fpr)]
    correct = (xo >= tstar).astype('i1') == T['y']

    X = sm.add_constant(pd.concat([T[pcs_use].astype(float), T['miss'].astype(float)], axis=1))
    m = sm.Logit(correct.astype('i1'), X).fit(disp=False, maxiter=200)

    beta = float(m.params['miss'])
    se = float(m.bse['miss'])
    OR = float(np.exp(beta))
    p = float(m.pvalues['miss'])

    rows.append([s.replace('_AVG',''), int(len(T)), beta, se, OR, np.format_float_positional(p, trim='-')])

out = pd.DataFrame(rows, columns=['score','n','beta_missing','se','OR_missing','p_two_sided']) \
       .sort_values('p_two_sided', key=lambda s: pd.to_numeric(s, errors='coerce'), na_position='last') \
       .reset_index(drop=True)
out
```

This doesn't account for all of it. PGS003957 still has a positive correlation between missingness and accuracy after accounting for PCs.

Let's inspect a few scores:
```
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

scores = sorted({c[:-4] for c in d.columns if isinstance(c, str) and c.endswith("_AVG")})
y = pd.Index(d.index.astype(str)).isin(set(pd.Series(cases, dtype=str))).astype("i1")

plt.figure(figsize=(8,6))
plotted = False
for s in scores:
    xcol, mcol = f"{s}_AVG", f"{s}_MISSING_PCT"
    if xcol not in d.columns or mcol not in d.columns: continue
    t = pd.DataFrame({"x": pd.to_numeric(d[xcol], errors="coerce"),
                      "m": pd.to_numeric(d[mcol], errors="coerce"),
                      "y": y}).dropna()
    if len(t) < 50 or t["y"].nunique() < 2 or t["m"].std(ddof=0) == 0: continue

    auc = roc_auc_score(t["y"], t["x"])
    xo = -t["x"] if np.isfinite(auc) and auc < 0.5 else t["x"]
    fpr, tpr, thr = roc_curve(t["y"], xo); tstar = thr[np.argmax(tpr - fpr)]
    acc = ((xo >= tstar).astype("i1") == t["y"]).astype(int)

    p90 = t["m"].quantile(0.9)
    keep = pd.concat([t[t["m"] >= p90],
                      t[t["m"] < p90].sample(frac=0.1, random_state=0)], ignore_index=False)
    ak = acc.loc[keep.index]
    if ak.std(ddof=0) == 0: continue

    zm = (keep["m"] - keep["m"].mean()) / keep["m"].std(ddof=0)
    za = (ak - ak.mean()) / ak.std(ddof=0)

    sc = plt.scatter(zm, za, s=3, alpha=0.18, label=s)
    c = sc.get_facecolor()[0]; plotted = True

    bins = pd.qcut(keep["m"], q=20, labels=False, duplicates="drop")
    bx = pd.Series(zm.values, index=bins.index).groupby(bins, observed=True).mean()
    by = pd.Series(za.values, index=bins.index).groupby(bins, observed=True).mean()
    if len(bx) > 1:
        order = np.argsort(bx.values)
        plt.plot(bx.values[order], by.values[order], marker="o", linewidth=1.5, markersize=4, color=c)

    lo = lowess(za.values, zm.values, frac=0.2, it=0, return_sorted=True)
    plt.plot(lo[:,0], lo[:,1], linewidth=2, color=c)

if plotted:
    plt.axhline(0, linewidth=1); plt.axvline(0, linewidth=1); plt.legend()
plt.xlabel("z-missingness"); plt.ylabel("z-accuracy (per individual)")
plt.title("z-missingness vs accuracy — scatter, percentile-bin means, and LOESS")
plt.tight_layout(); plt.show()
```

<img width="690" height="489" alt="image" src="https://github.com/user-attachments/assets/668f0d8c-f7a9-4098-a26b-1e3debdb282a" />
<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/c4730611-8f2d-448e-9f64-2a400d9115ca" />

Let's check accuracy within each inferred genetic ancestry group:
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# data
d = pd.read_csv('../../arrays_cog.sscore', sep='\t').set_index('#IID'); d.index = d.index.astype(str)
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

sex_df = (pd.read_csv(SEX_URI, sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
            .assign(person_id=lambda x: x["research_id"].astype(str))
            .set_index("person_id")["dragen_sex_ploidy"]
            .map({"XX":0,"XY":1})
            .to_frame("sex"))

case_set = set(pd.Series(cases, dtype=str))

# predictors = all scores + SEX + PCs
X = (d[pgs_cols]
       .join(sex_df, how="inner")
       .join(pc_df[pc_cols], how="inner")
       .dropna()
       .astype(float))
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
    rows.append({"ancestry": str(anc_val), "n": int(len(y_g)), "cases": int(y_g.sum()),
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
plt.title("AUROC by ancestry (All scores + Sex + PCs, 10-fold OOF)"); plt.tight_layout()

plt.figure(figsize=(9,5))
yv = res.set_index("ancestry").loc[anc_order, "OR_per_SD"].to_numpy()
lo = yv - res.set_index("ancestry").loc[anc_order, "OR_lo"].to_numpy()
hi = res.set_index("ancestry").loc[anc_order, "OR_hi"].to_numpy() - yv
plt.bar(x, yv, width=w, yerr=[lo, hi], capsize=4)
plt.xticks(x, anc_order, rotation=25, ha="right"); plt.ylabel("OR per SD of prediction")
plt.title("OR/SD by ancestry (All scores + Sex + PCs, 10-fold OOF)"); plt.tight_layout()

res
```

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/b5872265-60d1-4e7d-9b25-91f115ea91f1" />
<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/6b5dc483-00e1-49a9-9b7f-30d88532218e" />

Overall, the best score for Alzheimer's disease is PGS004146, with PGS004898 in second place.

This is because the rest were actually scores for breast cancer.

Let's try with other Alzheimer's scores:
- PGS004146
- PGS003957
- PGS003334
- PGS004589
- PGS004898
- PGS004229

This should complete in 35 minutes:
```
./gnomon/target/release/gnomon score "PGS004146, PGS003957, PGS003334, PGS004589, PGS004898, PGS004229" arrays
```

Decent missingness:
<img width="1011" height="1031" alt="image" src="https://github.com/user-attachments/assets/34e7ade4-e7c4-4b5a-a8ee-aecc5e4153af" />

A mix of normal and non-normal distributions:
<img width="1011" height="971" alt="image" src="https://github.com/user-attachments/assets/ae696dc4-ddbe-415f-8838-5b54124913eb" />

The top three scores are PGS004146_AVG, with an OR per SD of 1.32 and an AUROC of 0.57, PGS004898_AVG with an AUROC of 0.54, and PGS003334_AVG, with an AUROC of 0.52.

