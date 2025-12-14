If you haven't already, create a v8 workspace and download the microarray data:
```
!gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* ../..!gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* ../..
```

View sex chromosome ploidy counts:
```
import os
import io
import pandas as pd
import gcsfs

SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
SEX_COL = "dragen_sex_ploidy"
FAM_PATH = os.path.join("..", "..", "arrays.fam")

def load_genetic_sex():
    project = os.environ.get("GOOGLE_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_PROJECT is not set; cannot read requester-pays bucket.")

    fs = gcsfs.GCSFileSystem(project=project, token="cloud", requester_pays=True)
    with fs.open(SEX_URI, "rb") as f:
        df = pd.read_csv(f, sep="\t")
    return df

sex_df = load_genetic_sex()

if SEX_COL not in sex_df.columns:
    raise ValueError(
        f"Expected '{SEX_COL}' in genetic sex table; "
        f"found columns: {list(sex_df.columns)}"
    )

id_candidates = [
    "person_id",
    "participant_id",
    "aou_participant_id",
    "research_id",
    "sample_id",
]
id_col_candidates = [c for c in id_candidates if c in sex_df.columns]

if not id_col_candidates:
    raise ValueError(
        "No suitable participant ID column found in genetic sex table; "
        f"checked {id_candidates}"
    )

id_col = id_col_candidates[0]

sex_df[id_col] = sex_df[id_col].astype(str)
sex_df[SEX_COL] = sex_df[SEX_COL].astype(str)

fam_cols = [
    "family_id",
    "individual_id",
    "father_id",
    "mother_id",
    "sex_code",
    "phenotype",
]
fam_df = pd.read_csv(
    FAM_PATH,
    delim_whitespace=True,
    header=None,
    names=fam_cols,
    dtype=str,
)

array_ids = set(fam_df["individual_id"])

sex_df_with_array = sex_df[sex_df[id_col].isin(array_ids)]

def count_ploidy(values):
    v = values.fillna("NA")
    xx = (v == "XX").sum()
    xy = (v == "XY").sum()
    other = len(v) - xx - xy
    return xx, xy, other

overall_xx, overall_xy, overall_other = count_ploidy(sex_df[SEX_COL])
array_xx, array_xy, array_other = count_ploidy(sex_df_with_array[SEX_COL])

print("Overall genetic sex ploidy counts:")
print(f"  XX: {overall_xx}")
print(f"  XY: {overall_xy}")
print(f"  Other: {overall_other}")

print("\nGenetic sex ploidy counts among participants with microarray data:")
print(f"  XX: {array_xx}")
print(f"  XY: {array_xy}")
print(f"  Other: {array_other}")
```

```
import os
import io
import pandas as pd

SEX_COL = "dragen_sex_ploidy"

# Reconstruct microarray subset if it doesn't exist yet
if "sex_df_with_array" not in globals():
    FAM_PATH = os.path.join("..", "..", "arrays.fam")
    fam_cols = [
        "family_id",
        "individual_id",
        "father_id",
        "mother_id",
        "sex_code",
        "phenotype",
    ]
    fam_df = pd.read_csv(
        FAM_PATH,
        delim_whitespace=True,
        header=None,
        names=fam_cols,
        dtype=str,
    )

    array_ids = set(fam_df["individual_id"].astype(str))

    id_candidates = [
        "person_id",
        "participant_id",
        "aou_participant_id",
        "research_id",
        "sample_id",
    ]
    id_col = None
    for c in id_candidates:
        if c in sex_df.columns:
            id_col = c
            break

    if id_col is None:
        raise ValueError(
            "No suitable participant ID column found in genetic sex table; "
            f"checked {id_candidates}"
        )

    sex_df[id_col] = sex_df[id_col].astype(str)
    sex_df[SEX_COL] = sex_df[SEX_COL].astype(str)

    sex_df_with_array = sex_df[sex_df[id_col].isin(array_ids)]

# Helper: restrict to "Other" = not XX/XY and non-missing
def other_mask(df):
    return (
        df[SEX_COL].notna()
        & (~df[SEX_COL].isin(["XX", "XY"]))
    )

overall_other = sex_df.loc[other_mask(sex_df), SEX_COL]
array_other = sex_df_with_array.loc[other_mask(sex_df_with_array), SEX_COL]

print("Breakdown of 'Other' overall:")
print(overall_other.value_counts().sort_index())

print("\nBreakdown of 'Other' among participants with microarray data:")
print(array_other.value_counts().sort_index())
```

Let's run a polygenic score for colorectal cancer:
```
!../../gnomon/target/release/gnomon score "PGS003852" ../../arrays
```

Check missingness by sex chromosome ploidy:
```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEX_COL = "dragen_sex_ploidy"
SCORE_COL = "PGS003852_AVG"
MISS_COL = "PGS003852_MISSING_PCT"
SSCORE_PATH = os.path.join("..", "..", "arrays.sscore")

if "sex_df" not in globals():
    raise RuntimeError("sex_df is not defined; run the genetic sex load cell first.")

sscore_df = pd.read_csv(SSCORE_PATH, delim_whitespace=True)
sscore_df = sscore_df.rename(columns={"#IID": "IID"})
sscore_df["IID"] = sscore_df["IID"].astype(str)

id_candidates = [
    "person_id",
    "participant_id",
    "aou_participant_id",
    "research_id",
    "sample_id",
]
id_col = None
for c in id_candidates:
    if c in sex_df.columns:
        id_col = c
        break

if id_col is None:
    raise ValueError(
        "No suitable participant ID column found in sex_df; "
        f"checked {id_candidates}"
    )

sex_df[id_col] = sex_df[id_col].astype(str)

merged = sscore_df.merge(
    sex_df[[id_col, SEX_COL]],
    left_on="IID",
    right_on=id_col,
    how="left",
)

merged["sex_group"] = merged[SEX_COL].where(
    merged[SEX_COL].isin(["XX", "XY"]),
    "Other",
)

positive_missing = merged[MISS_COL] > 0
merged.loc[positive_missing, "log_missing"] = np.log(
    merged.loc[positive_missing, MISS_COL]
)

plt.figure()
for sex, sub in merged.groupby("sex_group"):
    plt.hist(
        sub["log_missing"].dropna(),
        bins=50,
        alpha=0.5,
        label=sex,
    )
plt.xlabel("log(PGS003852_MISSING_PCT)")
plt.ylabel("Count")
plt.title("Distribution of log(missingness) by genetic sex")
plt.legend()
plt.show()

plt.figure()
for sex, sub in merged.groupby("sex_group"):
    plt.hist(
        sub[SCORE_COL].dropna(),
        bins=50,
        alpha=0.5,
        label=sex,
    )
plt.xlabel("PGS003852_AVG")
plt.ylabel("Count")
plt.title("Distribution of PGS003852_AVG by genetic sex")
plt.legend()
plt.show()
```

There is no large visible bias, which is good.

<img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/18b7f98a-e80d-4858-8b1b-bc61edeefec2" />

Let's define colorectal cancer cases:
```
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery as bq
from sklearn.metrics import roc_auc_score

# Case definition: CRC diagnosis codes plus personal-history codes (incl. Z85.038, Z85.04x)
cdr_id = os.environ["WORKSPACE_CDR"]

cond_codes = (
    [f"C18.{i}" for i in range(10)]
    + ["C19", "C20"]
    + [f"153.{i}" for i in range(10)]
    + ["154.0", "154.1"]
)
cond_codes = [c.upper() for c in cond_codes]
cond_codes_n = sorted({c.replace(".", "") for c in cond_codes})

obs_codes_exact = ["Z85.038", "V10.05"]
obs_codes_exact = [c.upper() for c in obs_codes_exact]
obs_codes_n = sorted({c.replace(".", "") for c in obs_codes_exact})

sql = f"""
WITH cond_raw AS (
  SELECT DISTINCT CAST(person_id AS STRING) AS person_id,
         REGEXP_REPLACE(UPPER(TRIM(condition_source_value)), '[^A-Z0-9]', '') AS code_n
  FROM `{cdr_id}.condition_occurrence`
  WHERE condition_source_value IS NOT NULL
),
cond AS (
  SELECT DISTINCT person_id
  FROM cond_raw
  WHERE code_n IN UNNEST(@cond_codes_n)
),
obs_raw AS (
  SELECT DISTINCT CAST(person_id AS STRING) AS person_id,
         REGEXP_REPLACE(UPPER(TRIM(observation_source_value)), '[^A-Z0-9]', '') AS code_n
  FROM `{cdr_id}.observation`
  WHERE observation_source_value IS NOT NULL
),
obs AS (
  SELECT DISTINCT person_id
  FROM obs_raw
  WHERE code_n IN UNNEST(@obs_codes_n)
     OR STARTS_WITH(code_n, 'Z8504')  -- Z85.04x personal history codes
),
cases AS (
  SELECT person_id FROM cond
  UNION DISTINCT
  SELECT person_id FROM obs
)
SELECT person_id FROM cases
"""

cases = (
    bq.Client()
    .query(
        sql,
        job_config=bq.QueryJobConfig(
            query_parameters=[
                bq.ArrayQueryParameter("cond_codes_n", "STRING", cond_codes_n),
                bq.ArrayQueryParameter("obs_codes_n", "STRING", obs_codes_n),
            ]
        ),
    )
    .to_dataframe()["person_id"]
    .astype(str)
)
case_set = set(cases)
```

Let's check AUC:
```
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

if "sex_df" not in globals():
    raise RuntimeError("sex_df is not defined; load the genetic sex table in a previous cell.")
if "cases" not in globals():
    raise RuntimeError("cases is not defined; construct the colorectal cancer case set (incl. Z85.038, Z85.04x) first.")

sscore = pd.read_csv("../../arrays.sscore", sep="\t")
iid_col = sscore.columns[0]
sscore[iid_col] = sscore[iid_col].astype(str)

if "PGS003852_AVG" not in sscore.columns:
    raise RuntimeError("PGS003852_AVG not found in ../../arrays.sscore.")

id_candidates = [
    "person_id",
    "participant_id",
    "aou_participant_id",
    "research_id",
    "sample_id",
    "person_id_src",
]
iid_values = set(sscore[iid_col])
best_id = None
best_overlap = -1
for c in id_candidates:
    if c in sex_df.columns:
        ids = set(sex_df[c].astype(str))
        ov = len(iid_values & ids)
        if ov > best_overlap:
            best_overlap = ov
            best_id = c

if best_id is None or best_overlap == 0:
    raise RuntimeError("Could not match IIDs in arrays.sscore to any ID column in sex_df.")

sex_use = sex_df[[best_id, "dragen_sex_ploidy"]].copy()
sex_use[best_id] = sex_use[best_id].astype(str)
sex_use = sex_use.rename(columns={best_id: iid_col})

df = sscore.merge(sex_use, on=iid_col, how="left")

case_set = set(pd.Series(cases, dtype=str))
df["case"] = df[iid_col].isin(case_set).astype("i1")

df["sex_group"] = df["dragen_sex_ploidy"].where(
    df["dragen_sex_ploidy"].isin(["XX", "XY"]),
    "Other",
)

rows = []
for g in ["XX", "XY", "Other"]:
    sub = df[(df["sex_group"] == g) & df["PGS003852_AVG"].notna()]
    n = len(sub)
    n_cases = int(sub["case"].sum())
    n_ctrl = n - n_cases
    if n_cases > 0 and n_ctrl > 0:
        auc = float(roc_auc_score(sub["case"], sub["PGS003852_AVG"]))
    else:
        auc = np.nan
    rows.append(
        {
            "sex_group": g,
            "n": n,
            "cases": n_cases,
            "controls": n_ctrl,
            "AUC_PGS003852": auc,
        }
    )

print(pd.DataFrame(rows))
```

AUC looks about the same in each group (0.566504 in XX, 0.563028 in XY).

Let's check prevalence per-group:
```
import pandas as pd
import numpy as np

g = (
    df.groupby("sex_group")["case"]
      .agg(cases="sum", n="count")
      .assign(prevalence_pct=lambda x: 100 * x["cases"] / x["n"])
      .reset_index()
)

for _, row in g.iterrows():
    print(
        f"{row['sex_group']}: "
        f"{row['cases']} / {int(row['n'])} "
        f"({row['prevalence_pct']:.3f}%)"
    )
```
Other: 259 / 33941 (0.763%)
XX: 2089 / 252079 (0.829%)
XY: 1713 / 161258 (1.062%)

Let's define age:
```
import os
import pandas as pd
from google.cloud import bigquery as bq

cdr_id = os.environ["WORKSPACE_CDR"]
iid_col = next(c for c in ["#IID", "IID", "person_id", "research_id", "sample_id", "ID"] if c in df.columns)

client = bq.Client()

yob = client.query(
    f"SELECT person_id, year_of_birth FROM `{cdr_id}.person`"
).to_dataframe()
yob["person_id"] = yob["person_id"].astype(str)

obs = client.query(
    f"""
    SELECT person_id,
           EXTRACT(YEAR FROM MAX(observation_period_end_date)) AS obs_end_year
    FROM `{cdr_id}.observation_period`
    GROUP BY person_id
    """
).to_dataframe()
obs["person_id"] = obs["person_id"].astype(str)

demo = yob.merge(obs, on="person_id", how="inner")
demo["year_of_birth"] = pd.to_numeric(demo["year_of_birth"], errors="coerce")
demo["AGE"] = (demo["obs_end_year"] - demo["year_of_birth"]).clip(lower=0, upper=120)

df_age = df.merge(
    demo[["person_id", "AGE"]],
    left_on=iid_col,
    right_on="person_id",
    how="left"
).rename(columns={"AGE": "age"})
```

Let's check the age distribution in each group:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

data = df_age.copy()

def classify_sex(x):
    if pd.isna(x):
        return "Missing"
    if x == "XX":
        return "XX"
    if x == "XY":
        return "XY"
    return "Other"

data["sex_group4"] = data["dragen_sex_ploidy"].map(classify_sex)

valid = data[data["age"].notna()]
age_min = float(valid["age"].min())
age_max = float(valid["age"].max())
xs = np.linspace(age_min, age_max, 400)

plt.figure(figsize=(8, 5))

for grp in ["XX", "XY", "Other", "Missing"]:
    sub = valid.loc[valid["sex_group4"] == grp, "age"]
    if len(sub) < 2:
        continue
    kde = gaussian_kde(sub.to_numpy())
    ys = kde(xs)
    plt.plot(xs, ys, linewidth=2, label=f"{grp} (n={len(sub)})")
    mean_age = sub.mean()
    plt.axvline(mean_age, linestyle="--", linewidth=1)

plt.xlabel("Age (years)")
plt.ylabel("Density")
plt.title("Age distribution by genetic sex group")
plt.legend()
plt.tight_layout()
plt.show()
```


<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/344ee484-9d95-4b6d-93b0-347c8bae6f8e" />


This is interesting. Why do people with non-XY and non-XX ploidy have a much higher mean age?

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Specific non-XX/XY ploidy categories to display (including literal "nan" value, excluding true missing)
ploidy_labels = ["X0", "XO", "XXX", "XXY", "XYY", "nan"]
allowed = {p.lower() for p in ploidy_labels}

ploidy = df_age["dragen_sex_ploidy"]
age = df_age["age"]

is_missing_ploidy = ploidy.isna()
ploidy_str = ploidy.astype(str)
mask = age.notna() & (~is_missing_ploidy) & (ploidy_str.str.lower().isin(allowed))

sub = df_age.loc[mask].copy()
sub["ploidy_str"] = sub["dragen_sex_ploidy"].astype(str)

if not sub.empty:
    age_min = float(sub["age"].min())
    age_max = float(sub["age"].max())
    xs = np.linspace(age_min, age_max, 400)

    plt.figure(figsize=(8, 5))

    for p in ploidy_labels:
        grp = sub[sub["ploidy_str"].str.lower() == p.lower()]["age"].dropna()
        if len(grp) < 2:
            continue
        kde = gaussian_kde(grp.to_numpy())
        ys = kde(xs)
        plt.plot(xs, ys, linewidth=2, label=f"{p} (n={len(grp)})")
        mean_age = grp.mean()
        plt.axvline(mean_age, linestyle="--", linewidth=1)

    plt.xlabel("Age (years)")
    plt.ylabel("Density")
    plt.title("Age distribution by specific non-XX/XY ploidy (excluding missing)")
    plt.legend()
    plt.tight_layout()
    plt.show()
```


XO and X0 individuals are driving this trend, with a mean age of around 80. Mosaic loss of X is at a fairly high prevalence in women over 70 years old, and similarly in men for mosaic loss of the Y chromosome. I believe that this trend is due to somatic mosaicism. This ploidy data is from short-read WGS, which includes samples using blood-derived DNA.

Let's check colorectal cancer prevalences: 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n**2)) / denom
    return center - margin, center + margin

rows = []

x0_xo_mask = df["dragen_sex_ploidy"].isin(["X0", "XO"])
for label, mask in [
    ("X0/XO", x0_xo_mask),
    ("XXX", df["dragen_sex_ploidy"] == "XXX"),
    ("XXY", df["dragen_sex_ploidy"] == "XXY"),
    ("XYY", df["dragen_sex_ploidy"] == "XYY"),
    ("XY", df["dragen_sex_ploidy"] == "XY"),
    ("XX", df["dragen_sex_ploidy"] == "XX"),
]:
    sub = df[mask]
    n = int(len(sub))
    k = int(sub["case"].sum()) if n > 0 else 0
    p = (k / n) if n > 0 else np.nan
    lo, hi = wilson_ci(k, n)
    rows.append(
        {
            "group": label,
            "n": n,
            "cases": k,
            "prevalence": p,
            "ci_low": lo,
            "ci_high": hi,
        }
    )

res = pd.DataFrame(rows)

x = np.arange(len(res))
y = res["prevalence"].to_numpy()
yerr = np.vstack(
    [
        y - res["ci_low"].to_numpy(),
        res["ci_high"].to_numpy() - y,
    ]
)

plt.figure(figsize=(8, 5))
plt.bar(x, y * 100)
plt.errorbar(
    x,
    y * 100,
    yerr=yerr * 100,
    fmt="none",
    capsize=4,
    linewidth=1,
)
plt.xticks(x, res["group"])
plt.ylabel("Prevalence (%)")
plt.title("Colorectal cancer prevalence with 95% CI by genetic sex/ploidy group")
plt.tight_layout()
plt.show()

print(res.assign(prevalence_pct=lambda d: d["prevalence"] * 100)[
    ["group", "n", "cases", "prevalence_pct", "ci_low", "ci_high"]
])
```

Sex chromosome ploidy is relevant to understanding colorectal cancer risk:

<img width="690" height="440" alt="image" src="https://github.com/user-attachments/assets/17f9eac5-b3c1-433b-9aac-80ffa3f7550f" />

Let's prepare training and testing data for sex calibration:
```
import numpy as np
import pandas as pd

score_col = "PGS003852_AVG"
phenotype_src = "case"
sex_ploidy_col = "dragen_sex_ploidy"

if "df" not in globals():
    raise RuntimeError("df is not defined; run the merge/AUC construction cell first.")
if score_col not in df.columns:
    raise RuntimeError(f"{score_col} not found in df.")
if phenotype_src not in df.columns:
    raise RuntimeError(f"{phenotype_src} not found in df.")
if sex_ploidy_col not in df.columns:
    raise RuntimeError(f"{sex_ploidy_col} not found in df.")

# Resolve sample_id column
if "iid_col" in globals():
    sample_id_col = iid_col
elif "#IID" in df.columns:
    sample_id_col = "#IID"
elif "IID" in df.columns:
    sample_id_col = "IID"
else:
    sample_id_col = df.columns[0]

base = df[[sample_id_col, score_col, phenotype_src, sex_ploidy_col]].copy()
base = base.rename(columns={sample_id_col: "sample_id"})

is_binary_sex = base[sex_ploidy_col].isin(["XX", "XY"])
nonmissing = base[score_col].notna() & base[phenotype_src].notna()
model = base[is_binary_sex & nonmissing].copy()

sex_map = {"XX": 0, "XY": 1}
model["sex"] = model[sex_ploidy_col].map(sex_map).astype("int8")
model["phenotype"] = model[phenotype_src].astype("int8")

model = model.rename(columns={score_col: "score"})
model = model[["sample_id", "phenotype", "score", "sex"]]

model = model.replace([np.inf, -np.inf], np.nan).dropna()

rng = np.random.RandomState(20251108)

case_idx = model.index[model["phenotype"] == 1]
ctrl_idx = model.index[model["phenotype"] == 0]

test_case_size = int(0.2 * len(case_idx))
test_ctrl_size = int(0.2 * len(ctrl_idx))

if test_case_size == 0 or test_ctrl_size == 0:
    raise RuntimeError("Insufficient cases or controls for an 80/20 stratified split.")

test_cases = rng.choice(case_idx, size=test_case_size, replace=False)
test_ctrls = rng.choice(ctrl_idx, size=test_ctrl_size, replace=False)
test_idx = np.concatenate([test_cases, test_ctrls])

train_idx = model.index.difference(test_idx)

train = model.loc[train_idx].copy()
test = model.loc[test_idx].copy()

expected_cols = ["sample_id", "phenotype", "score", "sex"]
if list(train.columns) != expected_cols or list(test.columns) != expected_cols:
    raise RuntimeError("Column mismatch in prepared data.")
\]
train_path = "../../gnomon_train_PGS003852.tsv"
test_path = "../../gnomon_test_PGS003852.tsv"

train.to_csv(train_path, sep="\t", index=False)
test.to_csv(test_path, sep="\t", index=False)

print("Wrote:")
print(f"  {train_path}: {len(train)} rows")
print(f"  {test_path}: {len(test)} rows")
print("Train prevalence:", train["phenotype"].mean())
print("Test prevalence:", test["phenotype"].mean())
print("Train score summary:")
print(train["score"].describe())
print("Test score summary:")
print(test["score"].describe())
```

Let's train with gnomon:
```
!../../gnomon/target/release/gnomon train ../../gnomon_train_PGS003852.tsv --num-pcs 0
```

Let's train some other models:
```
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb

# Number of bins for the nonparametric bin model (per sex)
N_BINS = 20

train_path = "../../gnomon_train_PGS003852.tsv"
test_path = "../../gnomon_test_PGS003852.tsv"

if "train" in globals() and "test" in globals():
    train_df = train.copy()
    test_df = test.copy()
else:
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

required_cols = ["sample_id", "phenotype", "score", "sex"]
for name, df_ in [("train", train_df), ("test", test_df)]:
    missing = set(required_cols) - set(df_.columns)
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")

X_train_basic = train_df[["score", "sex"]].to_numpy()
y_train = train_df["phenotype"].astype(int).to_numpy()

X_test_basic = test_df[["score", "sex"]].to_numpy()
y_test = test_df["phenotype"].astype(int).to_numpy()

# Model 1: Logistic regression with score + sex
logit_main = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000,
)
logit_main.fit(X_train_basic, y_train)

# Model 2: Logistic regression with score + sex + score*sex
train_df_int = train_df.copy()
test_df_int = test_df.copy()
train_df_int["sex_score"] = train_df_int["sex"] * train_df_int["score"]
test_df_int["sex_score"] = test_df_int["sex"] * test_df_int["score"]

X_train_int = train_df_int[["score", "sex", "sex_score"]].to_numpy()
X_test_int = test_df_int[["score", "sex", "sex_score"]].to_numpy()

logit_int = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=1000,
)
logit_int.fit(X_train_int, y_train)

# Model 3: Sex-specific N_BINS-bin model
def make_bin_edges(scores, n_bins):
    lo = float(scores.min())
    hi = float(scores.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError("Non-finite score bounds in training.")
    if lo == hi:
        return np.array([lo, hi])
    return np.linspace(lo, hi, n_bins + 1)

edges = {}
bin_prevalence = {}

for sex_val in [0, 1]:
    mask = train_df["sex"] == sex_val
    s = train_df.loc[mask, "score"].to_numpy()
    y = train_df.loc[mask, "phenotype"].to_numpy()
    if len(s) == 0:
        raise RuntimeError(f"No training samples for sex={sex_val}.")
    e = make_bin_edges(s, N_BINS)
    edges[sex_val] = e

    idx = np.clip(
        np.searchsorted(e, s, side="right") - 1,
        0,
        len(e) - 2,
    )

    sex_mean = y.mean()
    preds = {}
    for b in range(len(e) - 1):
        in_bin = idx == b
        if in_bin.any():
            preds[b] = float(y[in_bin].mean())
        else:
            preds[b] = float(sex_mean)
    bin_prevalence[sex_val] = preds

def bin_model_predict(df, edges_dict, bin_prev_dict):
    out = np.zeros(len(df), dtype=float)
    for sex_val in [0, 1]:
        mask = df["sex"] == sex_val
        if not mask.any():
            continue
        s = df.loc[mask, "score"].to_numpy()
        e = edges_dict[sex_val]
        idx = np.clip(
            np.searchsorted(e, s, side="right") - 1,
            0,
            len(e) - 2,
        )
        preds_map = bin_prev_dict[sex_val]
        out[mask.to_numpy()] = [preds_map[b] for b in idx]
    return out

# Model 4: XGBoost with score + sex
xgb_model = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    nthread=4,
)
xgb_model.fit(X_train_basic, y_train)

# Test-set predictions
proba_logit_main = logit_main.predict_proba(X_test_basic)[:, 1]
proba_logit_int = logit_int.predict_proba(X_test_int)[:, 1]
proba_bin = bin_model_predict(test_df, edges, bin_prevalence)
proba_xgb = xgb_model.predict_proba(X_test_basic)[:, 1]

def summarize(name, y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return name, auc, brier

rows = []
for name, proba in [
    ("Logistic: score + sex", proba_logit_main),
    ("Logistic: + interaction", proba_logit_int),
    (f"{N_BINS}-bin model", proba_bin),
    ("XGBoost", proba_xgb),
]:
    rows.append(summarize(name, y_test, proba))

results_df = pd.DataFrame(rows, columns=["Model", "AUC", "Brier"])
results_df["AUC"] = results_df["AUC"].map("{:.4f}".format)
results_df["Brier"] = results_df["Brier"].map("{:.4f}".format)

print("\nTest-set performance")
print(results_df.to_string(index=False))
```

Test-set performance
                  Model    AUC  Brier
  Logistic: score + sex 0.5711 0.0091
Logistic: + interaction 0.5707 0.0091
           20-bin model 0.5705 0.0091
                XGBoost 0.5686 0.0091

The interaction doesn't help. All are decently calibrated. Adding sex improved the AUC a bit.

Let's train a simple model and evaluate it in each ancestry group predefined by All of Us. First, we'll run it without standardizing score:
```
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

print("="*80)
print("TRAINING EUR MODEL: score + sex + age + age²")
print("="*80)

# Get EUR data with sex
eur_ids = set(ancestry_labels_df[ancestry_labels_df['ANCESTRY'] == 'eur']['person_id'].astype(str))
sample_id_col = 'person_id' if 'person_id' in df_age.columns else df_age.columns[0]

df_eur = df_age[df_age[sample_id_col].astype(str).isin(eur_ids)].copy()
df_eur = df_eur[df_eur['dragen_sex_ploidy'].isin(['XX', 'XY'])].copy()
df_eur['sex'] = (df_eur['dragen_sex_ploidy'] == 'XY').astype(int)
df_eur['age_sq'] = df_eur['age'] ** 2
df_eur = df_eur.dropna(subset=['PGS003852_AVG', 'age', 'case'])

print(f"EUR training: n={len(df_eur)}, cases={df_eur['case'].sum()}")

# Train model
X = sm.add_constant(df_eur[['PGS003852_AVG', 'sex', 'age', 'age_sq']])
y = df_eur['case']
model = sm.Logit(y, X).fit(disp=0)

# Get coefficients
b0 = model.params['const']
b_score = model.params['PGS003852_AVG']
b_sex = model.params['sex']
b_age = model.params['age']
b_age_sq = model.params['age_sq']

# Calculate AUC
y_pred = model.predict(X)
auc_train = roc_auc_score(y, y_pred)

print("\n" + "="*80)
print("MODEL COEFFICIENTS")
print("="*80)
print(f"intercept      = {b0:.8f}  (p={model.pvalues['const']:.2e})")
print(f"PGS003852_AVG  = {b_score:.8f}  (p={model.pvalues['PGS003852_AVG']:.2e})")
print(f"sex            = {b_sex:.8f}  (p={model.pvalues['sex']:.2e})")
print(f"age            = {b_age:.8f}  (p={model.pvalues['age']:.2e})")
print(f"age_sq         = {b_age_sq:.8f}  (p={model.pvalues['age_sq']:.2e})")

print("\n" + "="*80)
print("FORMULA")
print("="*80)
print("logit = intercept + (β_score × score) + (β_sex × sex) + (β_age × age) + (β_age_sq × age²)")
print(f"\nlogit = {b0:.8f} + ({b_score:.8f} × score) + ({b_sex:.8f} × sex) + ({b_age:.8f} × age) + ({b_age_sq:.8f} × age²)")
print("\nP(CRC) = 1 / (1 + exp(-logit))")
print("\nEncoding:")
print("  score = PGS003852_AVG (polygenic score)")
print("  sex = 0 (XX/female), 1 (XY/male)")
print("  age = years")

print("\n" + "="*80)
print("ODDS RATIOS")
print("="*80)
print(f"Score (per SD): OR={np.exp(b_score * df_eur['PGS003852_AVG'].std()):.4f}")
print(f"Sex (XY vs XX): OR={np.exp(b_sex):.4f}")
print(f"Age @ 50 years: OR={np.exp(b_age * 50 + b_age_sq * 50**2):.4f}")
print(f"Age @ 60 years: OR={np.exp(b_age * 60 + b_age_sq * 60**2):.4f}")

print(f"\nTraining AUC (EUR): {auc_train:.4f}")

# Now apply to all ancestries
print("\n" + "="*80)
print("APPLYING EUR MODEL TO ALL ANCESTRIES")
print("="*80)

df_all = df_age.merge(ancestry_labels_df, left_on=sample_id_col, right_on='person_id', how='inner')
df_all = df_all[df_all['dragen_sex_ploidy'].isin(['XX', 'XY'])].copy()
df_all['sex'] = (df_all['dragen_sex_ploidy'] == 'XY').astype(int)
df_all['age_sq'] = df_all['age'] ** 2
df_all = df_all.dropna(subset=['PGS003852_AVG', 'age', 'case'])

# Predict using EUR model coefficients
df_all['logit'] = (b0 + 
                   b_score * df_all['PGS003852_AVG'] + 
                   b_sex * df_all['sex'] + 
                   b_age * df_all['age'] + 
                   b_age_sq * df_all['age_sq'])
df_all['pred'] = 1 / (1 + np.exp(-df_all['logit']))

# Calculate AUC per ancestry
results = []
for anc in sorted(df_all['ANCESTRY'].unique()):
    subset = df_all[df_all['ANCESTRY'] == anc]
    n = len(subset)
    cases = subset['case'].sum()
    if cases > 0 and cases < n:
        auc = roc_auc_score(subset['case'], subset['pred'])
    else:
        auc = np.nan
    results.append({
        'ancestry': anc,
        'n': n,
        'cases': cases,
        'AUC': auc
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))
```

TRAINING EUR MODEL: score + sex + age + age²

EUR training: n=233145, cases=2626


MODEL COEFFICIENTS

intercept      = -9.09711332  (p=2.73e-154)
PGS003852_AVG  = -1098.75158677  (p=9.56e-01)
sex            = 0.10227056  (p=1.00e-02)
age            = 0.09743362  (p=9.71e-20)
age_sq         = -0.00038742  (p=2.77e-06)


FORMULA

logit = intercept + (β_score × score) + (β_sex × sex) + (β_age × age) + (β_age_sq × age²)

logit = -9.09711332 + (-1098.75158677 × score) + (0.10227056 × sex) + (0.09743362 × age) + (-0.00038742 × age²)

P(CRC) = 1 / (1 + exp(-logit))

Encoding:
  score = PGS003852_AVG (polygenic score)
  sex = 0 (XX/female), 1 (XY/male)
  age = years


ODDS RATIOS

Score (per SD): OR=0.9989
Sex (XY vs XX): OR=1.1077
Age @ 50 years: OR=49.5569
Age @ 60 years: OR=85.7375

Training AUC (EUR): 0.6962


APPLYING EUR MODEL TO ALL ANCESTRIES


ancestry      n  cases      AUC
     afr  84008    628 0.733068
     amr  78984    470 0.744153
     eas  10086     56 0.709979
     eur 233145   2626 0.696179
     mid   [low number of cases]
     sas   [low number of cases]

Now, we'll run it with score standardization:
```
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

print("="*80)
print("TRAINING EUR MODEL: score + sex + age + age²")
print("="*80)

# Get EUR data with sex
eur_ids = set(ancestry_labels_df[ancestry_labels_df['ANCESTRY'] == 'eur']['person_id'].astype(str))
sample_id_col = 'person_id' if 'person_id' in df_age.columns else df_age.columns[0]

df_eur = df_age[df_age[sample_id_col].astype(str).isin(eur_ids)].copy()
df_eur = df_eur[df_eur['dragen_sex_ploidy'].isin(['XX', 'XY'])].copy()
df_eur['sex'] = (df_eur['dragen_sex_ploidy'] == 'XY').astype(int)
df_eur['age_sq'] = df_eur['age'] ** 2
df_eur = df_eur.dropna(subset=['PGS003852_AVG', 'age', 'case'])

print(f"EUR training: n={len(df_eur)}, cases={df_eur['case'].sum()}")

# Standardize score using EUR training data
score_mean = df_eur['PGS003852_AVG'].mean()
score_std = df_eur['PGS003852_AVG'].std()
df_eur['score_std'] = (df_eur['PGS003852_AVG'] - score_mean) / score_std

print(f"\nScore standardization parameters (EUR):")
print(f"  mean = {score_mean:.10f}")
print(f"  std  = {score_std:.10f}")

# Train model on standardized score
X = sm.add_constant(df_eur[['score_std', 'sex', 'age', 'age_sq']])
y = df_eur['case']
model = sm.Logit(y, X).fit(disp=0)

# Get coefficients
b0 = model.params['const']
b_score = model.params['score_std']
b_sex = model.params['sex']
b_age = model.params['age']
b_age_sq = model.params['age_sq']

# Calculate AUC
y_pred = model.predict(X)
auc_train = roc_auc_score(y, y_pred)

print("\n" + "="*80)
print("MODEL COEFFICIENTS")
print("="*80)
print(f"intercept   = {b0:.10f}  (p={model.pvalues['const']:.2e})")
print(f"score_std   = {b_score:.10f}  (p={model.pvalues['score_std']:.2e})")
print(f"sex         = {b_sex:.10f}  (p={model.pvalues['sex']:.2e})")
print(f"age         = {b_age:.10f}  (p={model.pvalues['age']:.2e})")
print(f"age_sq      = {b_age_sq:.10f}  (p={model.pvalues['age_sq']:.2e})")

print("\n" + "="*80)
print("FORMULA (WITH STANDARDIZED SCORE)")
print("="*80)
print("Step 1: Standardize the score")
print(f"  score_std = (PGS003852_AVG - {score_mean:.10f}) / {score_std:.10f}")
print("\nStep 2: Calculate logit")
print(f"  logit = {b0:.10f}")
print(f"        + {b_score:.10f} × score_std")
print(f"        + {b_sex:.10f} × sex")
print(f"        + {b_age:.10f} × age")
print(f"        + {b_age_sq:.10f} × age²")
print("\nStep 3: Get probability")
print("  P(CRC) = 1 / (1 + exp(-logit))")

# Expanded formula (one-step)
b0_expanded = b0 - (b_score * score_mean / score_std)
b_score_expanded = b_score / score_std

print("\n" + "="*80)
print("FORMULA (ONE-STEP, USING RAW SCORE)")
print("="*80)
print(f"logit = {b0_expanded:.10f}")
print(f"      + {b_score_expanded:.10f} × PGS003852_AVG")
print(f"      + {b_sex:.10f} × sex")
print(f"      + {b_age:.10f} × age")
print(f"      + {b_age_sq:.10f} × age²")
print("\nP(CRC) = 1 / (1 + exp(-logit))")

print("\n" + "="*80)
print("ENCODING")
print("="*80)
print("PGS003852_AVG: Raw polygenic score value")
print("sex: 0 = XX (female), 1 = XY (male)")
print("age: Age in years")

print("\n" + "="*80)
print("ODDS RATIOS")
print("="*80)
print(f"Score (per 1 SD):   OR = {np.exp(b_score):.4f}")
print(f"Sex (XY vs XX):     OR = {np.exp(b_sex):.4f}")
print(f"Age (per year @ 50): OR = {np.exp(b_age + 2*b_age_sq*50):.4f}")
print(f"Age (per year @ 60): OR = {np.exp(b_age + 2*b_age_sq*60):.4f}")

print(f"\nTraining AUC (EUR): {auc_train:.4f}")

# Apply to all ancestries
print("\n" + "="*80)
print("APPLYING EUR MODEL TO ALL ANCESTRIES")
print("="*80)

df_all = df_age.merge(ancestry_labels_df, left_on=sample_id_col, right_on='person_id', how='inner')
df_all = df_all[df_all['dragen_sex_ploidy'].isin(['XX', 'XY'])].copy()
df_all['sex'] = (df_all['dragen_sex_ploidy'] == 'XY').astype(int)
df_all['age_sq'] = df_all['age'] ** 2
df_all = df_all.dropna(subset=['PGS003852_AVG', 'age', 'case'])

# Standardize using EUR parameters
df_all['score_std'] = (df_all['PGS003852_AVG'] - score_mean) / score_std

# Predict
df_all['logit'] = (b0 + 
                   b_score * df_all['score_std'] + 
                   b_sex * df_all['sex'] + 
                   b_age * df_all['age'] + 
                   b_age_sq * df_all['age_sq'])
df_all['pred'] = 1 / (1 + np.exp(-df_all['logit']))

# Calculate AUC per ancestry
results = []
for anc in sorted(df_all['ANCESTRY'].unique()):
    subset = df_all[df_all['ANCESTRY'] == anc]
    n = len(subset)
    cases = subset['case'].sum()
    if cases > 0 and cases < n:
        auc = roc_auc_score(subset['case'], subset['pred'])
    else:
        auc = np.nan
    results.append({
        'ancestry': anc,
        'n': n,
        'cases': cases,
        'AUC': auc
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))
```

Results:
```
TRAINING EUR MODEL: score + sex + age + age²

EUR training: n=233145, cases=2626

Score standardization parameters (EUR):
  mean = -0.0000009977
  std  = 0.0000009789


MODEL COEFFICIENTS

intercept   = -9.1206459889  (p=1.76e-155)
score_std   = 0.2886184933  (p=1.46e-48)
sex         = 0.1019907222  (p=1.03e-02)
age         = 0.0963858986  (p=2.48e-19)
age_sq      = -0.0003751652  (p=5.72e-06)


FORMULA (WITH STANDARDIZED SCORE)

Step 1: Standardize the score
  score_std = (PGS003852_AVG - -0.0000009977) / 0.0000009789

Step 2: Calculate logit
  logit = -9.1206459889
        + 0.2886184933 × score_std
        + 0.1019907222 × sex
        + 0.0963858986 × age
        + -0.0003751652 × age²

Step 3: Get probability
  P(CRC) = 1 / (1 + exp(-logit))


FORMULA (ONE-STEP, USING RAW SCORE)

logit = -8.8264812706
      + 294848.8164193559 × PGS003852_AVG
      + 0.1019907222 × sex
      + 0.0963858986 × age
      + -0.0003751652 × age²

P(CRC) = 1 / (1 + exp(-logit))


ENCODING

PGS003852_AVG: Raw polygenic score value
sex: 0 = XX (female), 1 = XY (male)
age: Age in years


ODDS RATIOS

Score (per 1 SD):   OR = 1.3346
Sex (XY vs XX):     OR = 1.1074
Age (per year @ 50): OR = 1.0606
Age (per year @ 60): OR = 1.0527

Training AUC (EUR): 0.7118


APPLYING EUR MODEL TO ALL ANCESTRIES


ancestry      n  cases      AUC
     afr  84008    628 0.730533
     amr  78984    470 0.758445
     eas  10086     56 0.722547
     eur 233145   2626 0.711833
```

Standardizing the score is important. I believe this is actually due to the optimizer, not the math per se. We do not detect any obvious portability issues across ancestries.

Now, can we manually apply this formula to predict new cases?

Let's try it:
```
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Formula coefficients
b0 = -8.8264812706
b_pgs = 294848.8164193559
b_sex = 0.1019907222
b_age = 0.0963858986
b_age2 = -0.0003751652

# Prepare data
sample_id_col = 'person_id' if 'person_id' in df_age.columns else df_age.columns[0]
df_all = df_age.merge(ancestry_labels_df, left_on=sample_id_col, right_on='person_id', how='inner')
df_all = df_all[df_all['dragen_sex_ploidy'].isin(['XX', 'XY'])].copy()
df_all['sex'] = (df_all['dragen_sex_ploidy'] == 'XY').astype(int)
df_all = df_all.dropna(subset=['PGS003852_AVG', 'age', 'case'])

# Apply formula
df_all['logit'] = (b0 + 
                   b_pgs * df_all['PGS003852_AVG'] + 
                   b_sex * df_all['sex'] + 
                   b_age * df_all['age'] + 
                   b_age2 * df_all['age']**2)
df_all['P_CRC'] = 1 / (1 + np.exp(-df_all['logit']))

# Results for AMR and AFR
for ancestry in ['afr', 'amr']:
    subset = df_all[df_all['ANCESTRY'] == ancestry]
    n = len(subset)
    cases = subset['case'].sum()
    auc = roc_auc_score(subset['case'], subset['P_CRC'])
    
    print(f"{ancestry.upper()}: n={n:,}, cases={cases:,}, AUC={auc:.4f}")
```

AFR: n=84,008, cases=628, AUC=0.7305
AMR: n=78,984, cases=470, AUC=0.7584

Yes. These are the same results as earlier. It works.

Let's get baseline risk as a function of age and sex, per ancestry group:
```
import pandas as pd
import numpy as np
import statsmodels.api as sm

print("="*80)
print("BASELINE RISK MODELS (age + age² + sex only, no PGS)")
print("="*80)

# Prepare data
sample_id_col = 'person_id' if 'person_id' in df_age.columns else df_age.columns[0]
df_all = df_age.merge(ancestry_labels_df, left_on=sample_id_col, right_on='person_id', how='inner')
df_all = df_all[df_all['dragen_sex_ploidy'].isin(['XX', 'XY'])].copy()
df_all['sex'] = (df_all['dragen_sex_ploidy'] == 'XY').astype(int)
df_all = df_all.dropna(subset=['age', 'case'])

# Fit baseline model for each ancestry
for ancestry in ['eur', 'afr', 'amr', 'eas']:
    subset = df_all[df_all['ANCESTRY'] == ancestry].copy()
    subset['age_sq'] = subset['age'] ** 2
    
    X = sm.add_constant(subset[['sex', 'age', 'age_sq']])
    y = subset['case']
    
    model = sm.Logit(y, X).fit(disp=0)
    
    b0 = model.params['const']
    b_sex = model.params['sex']
    b_age = model.params['age']
    b_age2 = model.params['age_sq']
    
    print(f"\n{ancestry.upper()}: n={len(subset):,}, cases={y.sum():,}")
    print(f"  intercept = {b0:.10f}  (p={model.pvalues['const']:.2e})")
    print(f"  sex       = {b_sex:.10f}  (p={model.pvalues['sex']:.2e})")
    print(f"  age       = {b_age:.10f}  (p={model.pvalues['age']:.2e})")
    print(f"  age_sq    = {b_age2:.10f}  (p={model.pvalues['age_sq']:.2e})")
    print(f"\n  Formula:")
    print(f"  logit = {b0:.10f} + {b_sex:.10f} × sex + {b_age:.10f} × age + {b_age2:.10f} × age²")
```

Results:
```
BASELINE RISK MODELS (age + age² + sex only, no PGS)


EUR: n=233,145, cases=2,626
  intercept = -9.0959649138  (p=8.04e-155)
  sex       = 0.1022705366  (p=1.00e-02)
  age       = 0.0974301035  (p=9.72e-20)
  age_sq    = -0.0003873825  (p=2.77e-06)

  Formula:
  logit = -9.0959649138 + 0.1022705366 × sex + 0.0974301035 × age + -0.0003873825 × age²

AFR: n=84,008, cases=628
  intercept = -9.9101766486  (p=5.15e-45)
  sex       = -0.0306824327  (p=7.07e-01)
  age       = 0.1089007907  (p=1.73e-06)
  age_sq    = -0.0003786147  (p=3.78e-02)

  Formula:
  logit = -9.9101766486 + -0.0306824327 × sex + 0.1089007907 × age + -0.0003786147 × age²

AMR: n=78,984, cases=470
  intercept = -11.1002708853  (p=1.37e-58)
  sex       = 0.2636941754  (p=5.02e-03)
  age       = 0.1629602797  (p=7.97e-12)
  age_sq    = -0.0009405383  (p=3.36e-06)

  Formula:
  logit = -11.1002708853 + 0.2636941754 × sex + 0.1629602797 × age + -0.0009405383 × age²

EAS: n=10,086, cases=56
  intercept = -11.5423621124  (p=2.44e-10)
  sex       = -0.2764595500  (p=3.37e-01)
  age       = 0.2077715206  (p=1.58e-03)
  age_sq    = -0.0014940009  (p=9.62e-03)

  Formula:
  logit = -11.5423621124 + -0.2764595500 × sex + 0.2077715206 × age + -0.0014940009 × age²
```

Now let's try inferring sex with gnomon:
```
!../../gnomon/target/release/gnomon terms --sex ../../arrays.bed
```

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

gnomon = pd.read_csv("../../arrays.sex.tsv", sep="\t")
gnomon['IID'] = gnomon['IID'].astype(str)

id_col = next(c for c in ["person_id", "research_id", "sample_id"] if c in sex_df.columns)
sex_df[id_col] = sex_df[id_col].astype(str)

merged = gnomon.merge(sex_df[[id_col, 'dragen_sex_ploidy']], left_on='IID', right_on=id_col, how='inner')

ct = pd.crosstab(merged['dragen_sex_ploidy'], merged['Sex'], margins=True)
print(ct)

plt.figure(figsize=(8, 6))
sns.heatmap(ct.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Gnomon Sex')
plt.ylabel('DRAGEN Sex Ploidy')
plt.title('DRAGEN vs Gnomon Sex Assignment')
plt.tight_layout()
plt.show()
```

We can see that gnomon thinks that the X0, XO, XXY, XYY, and XY individuals are mostly male, whereas the XXX and XX individuals are mostly female.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gcsfs
import os

project = os.environ.get("GOOGLE_PROJECT")
fs = gcsfs.GCSFileSystem(project=project, token="cloud", requester_pays=True)

# Load sex_at_birth
with fs.open("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv", "rb") as f:
    metrics = pd.read_csv(f, sep="\t")

# Load gnomon predictions
gnomon = pd.read_csv("../../arrays.sex.tsv", sep="\t")
gnomon['IID'] = gnomon['IID'].astype(str)
gnomon = gnomon.rename(columns={'Sex': 'gnomon_sex'})

# Try to load germline allosomal aneuploidy
aneu_path = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/structural_variants/aux/aneuploidies"
aneu_files = fs.ls(aneu_path)
print("Available aneuploidy files:")
for f in aneu_files:
    print(f"  {f}")

germline_file = None
for f in aneu_files:
    fname = f.lower()
    if 'germline' in fname and ('allosomal' in fname or 'sex' in fname or 'x' in fname or 'y' in fname):
        germline_file = f
        break

if germline_file:
    with fs.open(germline_file, "rb") as f:
        germline_aneu = pd.read_csv(f, sep="\t")
    print(f"\nLoaded: {germline_file}")
    print(f"Columns: {list(germline_aneu.columns)}")
else:
    print("\nNo germline aneuploidy file found, creating empty placeholder")
    germline_aneu = pd.DataFrame()

# Find ID column
id_col = next((c for c in ["research_id", "person_id", "sample_id"] if c in metrics.columns), None)
metrics[id_col] = metrics[id_col].astype(str)

# Merge all sources
merged = metrics[[id_col, 'sex_at_birth', 'dragen_sex_ploidy']].copy()
merged = merged.merge(gnomon, left_on=id_col, right_on='IID', how='left')

if not germline_aneu.empty:
    germline_aneu_id = next((c for c in ["research_id", "person_id", "sample_id"] if c in germline_aneu.columns), germline_aneu.columns[0])
    germline_aneu[germline_aneu_id] = germline_aneu[germline_aneu_id].astype(str)
    germline_aneu['has_germline_aneu'] = True
    merged = merged.merge(germline_aneu[[germline_aneu_id, 'has_germline_aneu']], 
                          left_on=id_col, right_on=germline_aneu_id, how='left')
merged['has_germline_aneu'] = merged.get('has_germline_aneu', pd.Series(False, index=merged.index)).fillna(False)

# Check for Hail ploidy columns
hail_cols = [c for c in metrics.columns if 'ploidy' in c.lower() and c != 'dragen_sex_ploidy']
if hail_cols:
    print(f"\nFound Hail-related columns: {hail_cols}")
    for col in hail_cols[:5]:  # limit to first 5
        merged = merged.merge(metrics[[id_col, col]], on=id_col, how='left')

print("\n" + "="*60)
print("data coverage")
print("="*60)
print(f"total samples: {len(merged):,}")
print(f"has sex_at_birth: {merged['sex_at_birth'].notna().sum():,} ({100*merged['sex_at_birth'].notna().mean():.1f}%)")
print(f"has dragen_sex_ploidy: {merged['dragen_sex_ploidy'].notna().sum():,} ({100*merged['dragen_sex_ploidy'].notna().mean():.1f}%)")
print(f"has gnomon_sex: {merged['gnomon_sex'].notna().sum():,} ({100*merged['gnomon_sex'].notna().mean():.1f}%)")
print(f"has germline aneuploidy flag: {merged['has_germline_aneu'].sum():,} ({100*merged['has_germline_aneu'].mean():.1f}%)")

print("\n" + "="*60)
print("value distributions")
print("="*60)
print("\nsex_at_birth:")
print(merged['sex_at_birth'].value_counts(dropna=False))
print("\ndragen_sex_ploidy:")
print(merged['dragen_sex_ploidy'].value_counts(dropna=False))
print("\ngnomon_sex:")
print(merged['gnomon_sex'].value_counts(dropna=False))

# Concordance matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Self-report vs dragen
ct1 = pd.crosstab(merged['sex_at_birth'], merged['dragen_sex_ploidy'], dropna=False)
sns.heatmap(ct1, annot=True, fmt='g', cmap='Blues', ax=axes[0,0], cbar_kws={'label': 'count'})
axes[0,0].set_title('self-report vs dragen')

# Self-report vs gnomon
ct2 = pd.crosstab(merged['sex_at_birth'], merged['gnomon_sex'], dropna=False)
sns.heatmap(ct2, annot=True, fmt='g', cmap='Greens', ax=axes[0,1], cbar_kws={'label': 'count'})
axes[0,1].set_title('self-report vs gnomon')

# dragen vs gnomon
ct3 = pd.crosstab(merged['dragen_sex_ploidy'], merged['gnomon_sex'], dropna=False)
sns.heatmap(ct3, annot=True, fmt='g', cmap='Oranges', ax=axes[0,2], cbar_kws={'label': 'count'})
axes[0,2].set_title('dragen vs gnomon')

# Aneuploidy breakdown by dragen
aneu_dragen = merged.groupby('dragen_sex_ploidy')['has_germline_aneu'].agg(['sum', 'count', 'mean']).reset_index()
aneu_dragen = aneu_dragen.sort_values('mean', ascending=False)
axes[1,0].barh(range(len(aneu_dragen)), aneu_dragen['mean']*100)
axes[1,0].set_yticks(range(len(aneu_dragen)))
axes[1,0].set_yticklabels(aneu_dragen['dragen_sex_ploidy'])
axes[1,0].set_xlabel('% with germline aneuploidy')
axes[1,0].set_title('germline aneuploidy by dragen ploidy')

# Discordance rates
merged['dragen_binary'] = merged['dragen_sex_ploidy'].map({'XX': 'Female', 'XY': 'Male'})
merged['self_vs_dragen_discord'] = (merged['sex_at_birth'] != merged['dragen_binary']) & merged['sex_at_birth'].notna() & merged['dragen_binary'].notna()
merged['self_vs_gnomon_discord'] = (merged['sex_at_birth'].str.lower() != merged['gnomon_sex'].str.lower()) & merged['sex_at_birth'].notna() & merged['gnomon_sex'].notna()
merged['dragen_vs_gnomon_discord'] = (merged['dragen_binary'].str.lower() != merged['gnomon_sex'].str.lower()) & merged['dragen_binary'].notna() & merged['gnomon_sex'].notna()

discord_data = pd.DataFrame({
    'comparison': ['self vs dragen', 'self vs gnomon', 'dragen vs gnomon'],
    'discordance_pct': [
        100*merged['self_vs_dragen_discord'].mean(),
        100*merged['self_vs_gnomon_discord'].mean(),
        100*merged['dragen_vs_gnomon_discord'].mean()
    ]
})
axes[1,1].bar(discord_data['comparison'], discord_data['discordance_pct'])
axes[1,1].set_ylabel('discordance %')
axes[1,1].set_title('binary sex discordance rates')
axes[1,1].tick_params(axis='x', rotation=45)

# Non-xx/xy breakdown
non_binary = merged[~merged['dragen_sex_ploidy'].isin(['XX', 'XY'])].copy()
ploidy_counts = non_binary['dragen_sex_ploidy'].value_counts()
axes[1,2].bar(range(len(ploidy_counts)), ploidy_counts.values)
axes[1,2].set_xticks(range(len(ploidy_counts)))
axes[1,2].set_xticklabels(ploidy_counts.index, rotation=45)
axes[1,2].set_ylabel('count')
axes[1,2].set_title(f'non-xx/xy ploidy (n={len(non_binary):,})')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("concordance summary")
print("="*60)
print(f"self-report vs dragen (binary): {100*(1-merged['self_vs_dragen_discord'].mean()):.2f}% concordance")
print(f"self-report vs gnomon: {100*(1-merged['self_vs_gnomon_discord'].mean()):.2f}% concordance")
print(f"dragen vs gnomon (binary): {100*(1-merged['dragen_vs_gnomon_discord'].mean()):.2f}% concordance")

print("\n" + "="*60)
print("aneuploidy enrichment")
print("="*60)
print(aneu_dragen.to_string(index=False))
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Filter to confident XX/XY
confident = merged[
    (merged['dragen_sex_ploidy'].isin(['XX', 'XY'])) &
    (merged['sex_at_birth'].isin(['Female', 'Male'])) &
    (merged['gnomon_sex'].notna())
].copy()

confident['dragen_binary'] = confident['dragen_sex_ploidy'].map({'XX': 'female', 'XY': 'male'})
confident['self_binary'] = confident['sex_at_birth'].str.lower()

print("="*60)
print(f"confident xx/xy subset: {len(confident):,} / {len(merged):,} ({100*len(confident)/len(merged):.1f}%)")
print("="*60)

# All pairwise concordances
self_dragen_match = (confident['self_binary'] == confident['dragen_binary']).sum()
self_gnomon_match = (confident['self_binary'] == confident['gnomon_sex']).sum()
dragen_gnomon_match = (confident['dragen_binary'] == confident['gnomon_sex']).sum()

print(f"\nself ↔ dragen:  {self_dragen_match:,} / {len(confident):,} ({100*self_dragen_match/len(confident):.3f}%)")
print(f"self ↔ gnomon:  {self_gnomon_match:,} / {len(confident):,} ({100*self_gnomon_match/len(confident):.3f}%)")
print(f"dragen ↔ gnomon: {dragen_gnomon_match:,} / {len(confident):,} ({100*dragen_gnomon_match/len(confident):.3f}%)")

# Three-way concordance
all_three_match = (
    (confident['self_binary'] == confident['dragen_binary']) &
    (confident['self_binary'] == confident['gnomon_sex'])
).sum()
print(f"\nall three agree: {all_three_match:,} / {len(confident):,} ({100*all_three_match/len(confident):.3f}%)")

# Discordance patterns
confident['pattern'] = (
    confident['self_binary'] + '/' +
    confident['dragen_binary'] + '/' +
    confident['gnomon_sex']
)

discord_patterns = confident[confident['self_binary'] != confident['dragen_binary']]['pattern'].value_counts()
print(f"\ndiscordance patterns (n={discord_patterns.sum():,}):")
print(discord_patterns.head(10))

# Crosstabs
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ct1 = pd.crosstab(confident['self_binary'], confident['dragen_binary'], normalize='index') * 100
sns.heatmap(ct1, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=100, ax=axes[0], cbar_kws={'label': '% within row'})
axes[0].set_title('self → dragen (%)')

ct2 = pd.crosstab(confident['self_binary'], confident['gnomon_sex'], normalize='index') * 100
sns.heatmap(ct2, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=100, ax=axes[1], cbar_kws={'label': '% within row'})
axes[1].set_title('self → gnomon (%)')

ct3 = pd.crosstab(confident['dragen_binary'], confident['gnomon_sex'], normalize='index') * 100
sns.heatmap(ct3, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=100, ax=axes[2], cbar_kws={'label': '% within row'})
axes[2].set_title('dragen → gnomon (%)')

plt.tight_layout()
plt.show()

# Discordance breakdown by method pair
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Self vs dragen discordances
self_dragen_discord = confident[confident['self_binary'] != confident['dragen_binary']]
if len(self_dragen_discord) > 0:
    sd_counts = self_dragen_discord.groupby(['self_binary', 'dragen_binary']).size().reset_index(name='count')
    sd_counts['label'] = sd_counts['self_binary'] + '→' + sd_counts['dragen_binary']
    axes[0].bar(sd_counts['label'], sd_counts['count'])
    axes[0].set_title(f'self ↔ dragen discord (n={len(self_dragen_discord):,})')
    axes[0].set_ylabel('count')
    axes[0].tick_params(axis='x', rotation=45)

# Self vs gnomon discordances
self_gnomon_discord = confident[confident['self_binary'] != confident['gnomon_sex']]
if len(self_gnomon_discord) > 0:
    sg_counts = self_gnomon_discord.groupby(['self_binary', 'gnomon_sex']).size().reset_index(name='count')
    sg_counts['label'] = sg_counts['self_binary'] + '→' + sg_counts['gnomon_sex']
    axes[1].bar(sg_counts['label'], sg_counts['count'])
    axes[1].set_title(f'self ↔ gnomon discord (n={len(self_gnomon_discord):,})')
    axes[1].set_ylabel('count')
    axes[1].tick_params(axis='x', rotation=45)

# Dragen vs gnomon discordances
dragen_gnomon_discord = confident[confident['dragen_binary'] != confident['gnomon_sex']]
if len(dragen_gnomon_discord) > 0:
    dg_counts = dragen_gnomon_discord.groupby(['dragen_binary', 'gnomon_sex']).size().reset_index(name='count')
    dg_counts['label'] = dg_counts['dragen_binary'] + '→' + dg_counts['gnomon_sex']
    axes[2].bar(dg_counts['label'], dg_counts['count'])
    axes[2].set_title(f'dragen ↔ gnomon discord (n={len(dragen_gnomon_discord):,})')
    axes[2].set_ylabel('count')
    axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Check if discordances are symmetric
print("\n" + "="*60)
print("discordance symmetry")
print("="*60)
female_to_male_self_dragen = ((confident['self_binary'] == 'female') & (confident['dragen_binary'] == 'male')).sum()
male_to_female_self_dragen = ((confident['self_binary'] == 'male') & (confident['dragen_binary'] == 'female')).sum()
print(f"self=female, dragen=male: {female_to_male_self_dragen:,}")
print(f"self=male, dragen=female: {male_to_female_self_dragen:,}")

female_to_male_self_gnomon = ((confident['self_binary'] == 'female') & (confident['gnomon_sex'] == 'male')).sum()
male_to_female_self_gnomon = ((confident['self_binary'] == 'male') & (confident['gnomon_sex'] == 'female')).sum()
print(f"self=female, gnomon=male: {female_to_male_self_gnomon:,}")
print(f"self=male, gnomon=female: {male_to_female_self_gnomon:,}")

female_to_male_dragen_gnomon = ((confident['dragen_binary'] == 'female') & (confident['gnomon_sex'] == 'male')).sum()
male_to_female_dragen_gnomon = ((confident['dragen_binary'] == 'male') & (confident['gnomon_sex'] == 'female')).sum()
print(f"dragen=female, gnomon=male: {female_to_male_dragen_gnomon:,}")
print(f"dragen=male, gnomon=female: {male_to_female_dragen_gnomon:,}")
```

In 0.14% of cases, gnomon incorrectly predicted male when sex assigned at birth was female, but all male sex assignments were predicted correctly by gnomon. This implies gnomon miscalls XX as male in certain samples, which should be fixed.

