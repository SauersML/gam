Microarray data should be downloaded if you haven't already:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Install gnomon:
```
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && git clone https://github.com/SauersML/gnomon.git && cd gnomon && rustup override set nightly && cargo build --release && cd ~
```

Run scores:
```
!./gnomon/target/release/gnomon score "PGS004146, PGS004898, PGS003334, PGS003852, PGS005199, PGS004150, PGS000007, PGS000508, PGS000332" ../../arrays
```

Run this:
```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery as bq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DISEASES = {
    'alzheimers': {
        'icd10': ['G30.0', 'G30.1', 'G30.8', 'G30.9'],
        'icd9': ['331.0'],
        'pgs': ['PGS004146', 'PGS004898', 'PGS003334']
    },
    'colorectal': {
        'icd10': [f'C18.{i}' for i in range(10)] + ['C19', 'C20'],
        'icd9': [f'153.{i}' for i in range(10)] + ['154.0', '154.1'],
        'obs_exact': ['Z85.038', 'V10.05'],
        'obs_prefix': 'Z8504',  # Z85.04x codes
        'pgs': ['PGS003852']
    },
    'obesity': {
        'icd10': ['E66.811', 'E66.812', 'E66.813', 'E66.01', 'E66.09', 
                  'E66.1', 'E66.2', 'E66.89', 'E66.9', 'E88.82'],
        'icd9': ['278.00', '278.01', '278.03'],
        'pgs': ['PGS005199', 'PGS004150']
    },
    'breast': {
        'icd10': [f'C50.{i}' for i in range(10)] + [f'D05.{i}' for i in range(10)] + ['Z85.3'],
        'icd9': [f'174.{i}' for i in range(10)] + [f'175.{i}' for i in range(10)] + ['233.0', 'V10.3'],
        'pgs': ['PGS000007', 'PGS000508', 'PGS000332']
    }
}

cdr_id = os.environ["WORKSPACE_CDR"]
google_project = os.getenv('GOOGLE_PROJECT')

print("="*80)
print("DISEASE RISK MODELING: CV EVALUATION + FINAL CALIBRATION")
print("="*80)

# ============================================================================
# LOAD BASE DATA
# ============================================================================

print("\n[1/7] Loading base genomic data...")

# Load genetic sex
import gcsfs
fs = gcsfs.GCSFileSystem(project=google_project, token="cloud", requester_pays=True)
with fs.open("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv", "rb") as f:
    sex_df = pd.read_csv(f, sep="\t")

# Load gnomon sex predictions
gnomon = pd.read_csv("../../arrays.sex.tsv", sep="\t")
gnomon['IID'] = gnomon['IID'].astype(str)
gnomon = gnomon.rename(columns={'Sex': 'gnomon_sex'})

# Load ancestry
with fs.open("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv", "rb") as f:
    ancestry_df = pd.read_csv(f, sep="\t")
ancestry_df = ancestry_df[["research_id", "ancestry_pred"]].copy()
ancestry_df.columns = ["person_id", "ancestry"]
ancestry_df["person_id"] = ancestry_df["person_id"].astype(str)

# Load age data
client = bq.Client()
yob = client.query(f"SELECT person_id, year_of_birth FROM `{cdr_id}.person`").to_dataframe()
yob["person_id"] = yob["person_id"].astype(str)

obs = client.query(f"""
    SELECT person_id, EXTRACT(YEAR FROM MAX(observation_period_end_date)) AS obs_end_year
    FROM `{cdr_id}.observation_period` GROUP BY person_id
""").to_dataframe()
obs["person_id"] = obs["person_id"].astype(str)

demo = yob.merge(obs, on="person_id", how="inner")
demo["year_of_birth"] = pd.to_numeric(demo["year_of_birth"], errors="coerce")
demo["age"] = (demo["obs_end_year"] - demo["year_of_birth"]).clip(lower=0, upper=120)

# Load polygenic scores
sscore = pd.read_csv("../../arrays.sscore", sep="\t")
iid_col = sscore.columns[0]
sscore = sscore.rename(columns={iid_col: 'IID'})
sscore['IID'] = sscore['IID'].astype(str)

# Find ID column for merging
id_col = next((c for c in ["research_id", "person_id", "sample_id"] if c in sex_df.columns), None)
sex_df[id_col] = sex_df[id_col].astype(str)

# Merge base data
base_df = sex_df[[id_col]].copy()
base_df = base_df.merge(gnomon, left_on=id_col, right_on='IID', how='inner')
base_df = base_df.merge(ancestry_df, left_on=id_col, right_on='person_id', how='left')
base_df = base_df.merge(demo[['person_id', 'age']], left_on=id_col, right_on='person_id', how='left')
base_df = base_df.merge(sscore, on='IID', how='left')
base_df['sex_binary'] = (base_df['gnomon_sex'].str.lower() == 'male').astype(int)
base_df['age_sq'] = base_df['age'] ** 2

print(f"  Base cohort: {len(base_df):,} individuals")

# ============================================================================
# DEFINE CASES FOR EACH DISEASE
# ============================================================================

print("\n[2/7] Defining disease cases from EHR...")

def get_cases(disease_config):
    """Extract cases from BigQuery based on ICD codes"""
    
    # Handle standard condition codes
    if 'icd10' in disease_config or 'icd9' in disease_config:
        cond_codes = disease_config.get('icd10', []) + disease_config.get('icd9', [])
        cond_codes = [c.upper() for c in cond_codes]
        cond_codes_n = sorted(set([c.replace(".", "") for c in cond_codes]))
        
        cond_sql = f"""
        SELECT DISTINCT CAST(person_id AS STRING) AS person_id
        FROM `{cdr_id}.condition_occurrence`
        WHERE REGEXP_REPLACE(UPPER(TRIM(condition_source_value)), '[^A-Z0-9]', '') IN UNNEST(@codes_n)
        """
        
        cases_cond = client.query(
            cond_sql,
            job_config=bq.QueryJobConfig(
                query_parameters=[bq.ArrayQueryParameter("codes_n", "STRING", cond_codes_n)]
            )
        ).to_dataframe()['person_id'].astype(str)
    else:
        cases_cond = pd.Series([], dtype=str)
    
    # Handle observation codes (for colorectal)
    if 'obs_exact' in disease_config:
        obs_codes = disease_config['obs_exact']
        obs_codes = [c.upper() for c in obs_codes]
        obs_codes_n = sorted(set([c.replace(".", "") for c in obs_codes]))
        obs_prefix = disease_config.get('obs_prefix', '')
        
        obs_sql = f"""
        WITH obs_raw AS (
          SELECT DISTINCT CAST(person_id AS STRING) AS person_id,
                 REGEXP_REPLACE(UPPER(TRIM(observation_source_value)), '[^A-Z0-9]', '') AS code_n
          FROM `{cdr_id}.observation`
          WHERE observation_source_value IS NOT NULL
        )
        SELECT DISTINCT person_id
        FROM obs_raw
        WHERE code_n IN UNNEST(@obs_codes_n)
           OR STARTS_WITH(code_n, @obs_prefix)
        """
        
        cases_obs = client.query(
            obs_sql,
            job_config=bq.QueryJobConfig(
                query_parameters=[
                    bq.ArrayQueryParameter("obs_codes_n", "STRING", obs_codes_n),
                    bq.ScalarQueryParameter("obs_prefix", "STRING", obs_prefix)
                ]
            )
        ).to_dataframe()['person_id'].astype(str)
    else:
        cases_obs = pd.Series([], dtype=str)
    
    # Combine
    all_cases = pd.concat([cases_cond, cases_obs]).unique()
    return set(all_cases)

disease_cases = {}
for disease_name, config in DISEASES.items():
    cases = get_cases(config)
    disease_cases[disease_name] = cases
    print(f"  {disease_name}: {len(cases):,} cases")

# ============================================================================
# CREATE DATASETS
# ============================================================================

print("\n[3/7] Creating disease-specific datasets...")

disease_dfs = {}
for disease_name, config in DISEASES.items():
    df = base_df.copy()
    df['case'] = df[id_col].isin(disease_cases[disease_name]).astype(int)
    
    pgs_cols = [f"{pgs}_AVG" for pgs in config['pgs']]
    available_pgs = [col for col in pgs_cols if col in df.columns]
    
    if not available_pgs:
        continue
    
    keep_cols = [id_col, 'IID', 'gnomon_sex', 'sex_binary', 'ancestry', 'age', 'age_sq', 'case'] + available_pgs
    df = df[keep_cols].dropna().copy()
    
    disease_dfs[disease_name] = {'data': df, 'pgs_cols': available_pgs}
    print(f"  {disease_name}: {len(df):,} samples, {df['case'].sum():,} cases")

# ============================================================================
# CV EVALUATION & FINAL MODEL TRAINING
# ============================================================================

print("\n[4/7] Running 5-Fold Cross-Validation and Final Training...")

final_models = []
eval_metrics = []

for disease_name, info in disease_dfs.items():
    print(f"\n  Processing {disease_name.upper()}...")
    df = info['data']
    pgs_cols = info['pgs_cols']
    
    # Use only European ancestry for consistent risk modeling
    df_eur = df[df['ancestry'] == 'eur'].reset_index(drop=True)
    if len(df_eur) < 100:
        print("    Skipping: Not enough EUR samples")
        continue

    # --- PART A: 5-FOLD CROSS-VALIDATION (Metric Estimation) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs_base = []
    fold_aucs_full = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_eur, df_eur['case'])):
        train_set = df_eur.iloc[train_idx].copy()
        val_set = df_eur.iloc[val_idx].copy()
        
        # Standardize based on TRAIN fold only
        fold_means = train_set[pgs_cols].mean()
        fold_stds = train_set[pgs_cols].std()
        
        train_pgs_std = (train_set[pgs_cols] - fold_means) / fold_stds
        val_pgs_std = (val_set[pgs_cols] - fold_means) / fold_stds
        
        # Baseline (Age + Sex)
        X_train_base = sm.add_constant(train_set[['age', 'age_sq', 'sex_binary']])
        X_val_base = sm.add_constant(val_set[['age', 'age_sq', 'sex_binary']], has_constant='add')
        m_base = sm.Logit(train_set['case'], X_train_base).fit(disp=0)
        p_base = m_base.predict(X_val_base)
        
        # Full (Age + Sex + PGS)
        X_train_full = pd.concat([X_train_base, train_pgs_std.add_suffix('_std')], axis=1)
        X_val_full = pd.concat([X_val_base, val_pgs_std.add_suffix('_std')], axis=1)
        m_full = sm.Logit(train_set['case'], X_train_full).fit(disp=0)
        p_full = m_full.predict(X_val_full)
        
        if val_set['case'].sum() > 0:
            fold_aucs_base.append(roc_auc_score(val_set['case'], p_base))
            fold_aucs_full.append(roc_auc_score(val_set['case'], p_full))

    # --- PART B: FINAL MODEL TRAINING (On ALL EUR Data) ---
    # Standardize on FULL dataset
    final_means = df_eur[pgs_cols].mean()
    final_stds = df_eur[pgs_cols].std()
    
    df_eur_std = df_eur.copy()
    for col in pgs_cols:
        df_eur_std[f'{col}_std'] = (df_eur[col] - final_means[col]) / final_stds[col]
    
    std_cols = [f'{c}_std' for c in pgs_cols]
    
    # Train Baseline
    X_base = sm.add_constant(df_eur_std[['age', 'age_sq', 'sex_binary']])
    model_base = sm.Logit(df_eur_std['case'], X_base).fit(disp=0)
    
    # Train Full
    X_full = sm.add_constant(df_eur_std[['age', 'age_sq', 'sex_binary'] + std_cols])
    model_full = sm.Logit(df_eur_std['case'], X_full).fit(disp=0)
    
    # Store everything needed for plotting and manual calc
    final_models.append({
        'disease': disease_name,
        'model_full': model_full,
        'model_base': model_base,
        'means': final_means,
        'stds': final_stds,
        'pgs_cols': pgs_cols,
        'data': df_eur,
        'std_cols': std_cols
    })
    
    eval_metrics.append({
        'disease': disease_name,
        'n': len(df_eur),
        'cases': df_eur['case'].sum(),
        'cv_auc_base': np.mean(fold_aucs_base),
        'cv_auc_full': np.mean(fold_aucs_full)
    })

# ============================================================================
# METRICS REPORT
# ============================================================================

print("\n[5/7] 5-Fold Cross-Validation Results (Robust AUC)")
print("-" * 80)
print(f"{'DISEASE':<15} {'N':>8} {'CASES':>6} {'BASE AUC':>10} {'FULL AUC':>10} {'GAIN':>8}")
print("-" * 80)
for m in eval_metrics:
    gain = m['cv_auc_full'] - m['cv_auc_base']
    print(f"{m['disease']:<15} {m['n']:8,} {m['cases']:6,} {m['cv_auc_base']:10.4f} {m['cv_auc_full']:10.4f} {gain:+8.4f}")

# ============================================================================
# PLOTTING: DECILES AND AGE CURVES
# ============================================================================

print("\n[6/7] Generating Plots...")
# 2 plots per disease: 1. PGS Decile Bar, 2. Age-Risk Curves
n_dis = len(final_models)
fig, axes = plt.subplots(n_dis, 2, figsize=(15, 6 * n_dis))
if n_dis == 1: axes = axes.reshape(1, -1)

for idx, res in enumerate(final_models):
    disease = res['disease']
    df = res['data'].copy()
    pgs_cols = res['pgs_cols']
    
    # --- PLOT 1: PGS DECILE vs PREVALENCE ---
    # Create a composite PGS Score using the coefficients from the model
    # LogOdds = Intercept + Age... + (Beta1 * PGS1_std) + (Beta2 * PGS2_std)
    # We isolate just the genetic part: GeneticScore = Sum(Beta_i * PGS_i_std)
    
    weights = res['model_full'].params
    df['genetic_component'] = 0
    for col, std_col in zip(pgs_cols, res['std_cols']):
        if std_col in weights:
            # Re-standardize here just to be safe
            val_std = (df[col] - res['means'][col]) / res['stds'][col]
            df['genetic_component'] += val_std * weights[std_col]
            
    df['pgs_decile'] = pd.qcut(df['genetic_component'], 10, labels=False) + 1
    
    decile_stats = df.groupby('pgs_decile')['case'].mean() * 100 # Prevalence %
    
    ax1 = axes[idx, 0]
    bars = ax1.bar(decile_stats.index, decile_stats.values, color='teal', alpha=0.7)
    ax1.set_xlabel('Polygenic Score Decile (Low -> High)')
    ax1.set_ylabel('Observed Prevalence (%)')
    ax1.set_title(f'{disease.upper()}: Disease Rate by Genetic Risk')
    ax1.set_xticks(range(1, 11))
    ax1.grid(axis='y', alpha=0.3)
    
    # --- PLOT 2: ABSOLUTE RISK OVER AGE ---
    ax2 = axes[idx, 1]
    
    # Create synthetic data for ages 40 to 90
    ages = np.arange(40, 91, 1)
    
    # Determine reference sex (Majority sex in cases)
    pct_male_cases = df[df['case']==1]['sex_binary'].mean()
    ref_sex = 1 if pct_male_cases > 0.5 else 0
    sex_label = "Males" if ref_sex == 1 else "Females"
    
    # Determine genetic levels (10th, 50th, 90th percentile of the genetic component)
    gen_low = df['genetic_component'].quantile(0.10)
    gen_mid = df['genetic_component'].quantile(0.50)
    gen_high = df['genetic_component'].quantile(0.90)
    
    # Function to calculate probability
    def get_prob(age_arr, gen_score):
        # LogOdds = Intercept + B_age*Age + B_age2*Age^2 + B_sex*Sex + GenScore
        lp = (weights['const'] + 
              weights['age'] * age_arr + 
              weights['age_sq'] * (age_arr**2) + 
              weights['sex_binary'] * ref_sex + 
              gen_score)
        return 1 / (1 + np.exp(-lp))
    
    prob_low = get_prob(ages, gen_low) * 100
    prob_mid = get_prob(ages, gen_mid) * 100
    prob_high = get_prob(ages, gen_high) * 100
    
    ax2.plot(ages, prob_low, label='Low Risk (Bottom 10%)', color='green', linestyle='--')
    ax2.plot(ages, prob_mid, label='Average Risk', color='gray')
    ax2.plot(ages, prob_high, label='High Risk (Top 10%)', color='red', linewidth=2)
    
    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Predicted Absolute Risk (%)')
    ax2.set_title(f'{disease.upper()}: Risk over Age ({sex_label})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../disease_risk_summary.png', dpi=150)
plt.show()

# ============================================================================
# FINAL OUTPUT: COEFFICIENTS AND MANUAL CALCULATOR
# ============================================================================

print("\n[7/7] FINAL MODEL PARAMETERS (FOR MANUAL CALCULATION)")
print("="*80)
print("INSTRUCTIONS FOR MANUAL CALCULATION:")
print("1. Get your RAW Polygenic Score(s).")
print("2. STANDARDIZE each score: (Raw - Mean) / StdDev")
print("3. CALCULATE Log-Odds (L): Intercept + (Age*Coeff) + (Age^2*Coeff) + (Sex*Coeff) + (StdPGS*Coeff)")
print("   * Note: Age^2 is Age squared. Sex is 1 for Male, 0 for Female.")
print("4. CALCULATE Risk Probability: 1 / (1 + exp(-L))")
print("="*80)

for res in final_models:
    d = res['disease'].upper()
    print(f"\nDISEASE: {d}")
    print("-" * 60)
    
    # 1. Print Standardization Params
    print("STANDARDIZATION PARAMETERS (Use these first):")
    for col in res['pgs_cols']:
        m = res['means'][col]
        s = res['stds'][col]
        print(f"  {col}:")
        print(f"    Mean (μ)   = {m:.6f}")
        print(f"    StdDev (σ) = {s:.6f}")
    
    # 2. Print Regression Coefficients
    print("\nMODEL COEFFICIENTS (Use these second):")
    params = res['model_full'].params
    for name, val in params.items():
        print(f"  {name:20s}: {val:12.6f}")
        
    print("-" * 60)
```

Make some plots:
```
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

DECILE_COLORS = sns.color_palette("RdYlGn_r", 10)
RISK_COLORS = {'low': '#2ecc71', 'average': '#95a5a6', 'high': '#e74c3c'}

DISEASE_NAMES = {
    'alzheimers': "Alzheimer's Disease",
    'colorectal': 'Colorectal Cancer',
    'obesity': 'Obesity',
    'breast': 'Breast Cancer'
}

def wilson_ci(n_success, n_total, confidence=0.95):
    if n_total == 0:
        return 0, 0
    p = n_success / n_total
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
    return centre - margin, centre + margin

summary_data = []

for res in final_models:
    disease_key = res['disease']
    disease = DISEASE_NAMES.get(disease_key, disease_key.title())
    df = res['data'].copy()
    weights = res['model_full'].params
    
    df['genetic_component'] = 0
    for col, std_col in zip(res['pgs_cols'], res['std_cols']):
        if std_col in weights:
            val_std = (df[col] - res['means'][col]) / res['stds'][col]
            df['genetic_component'] += val_std * weights[std_col]
    
    df['pgs_decile'] = pd.qcut(df['genetic_component'], 10, labels=False, duplicates='drop') + 1
    decile_stats = df.groupby('pgs_decile').agg({
        'case': ['sum', 'count', 'mean']
    }).reset_index()
    decile_stats.columns = ['decile', 'cases', 'total', 'prevalence']
    decile_stats['prevalence_pct'] = decile_stats['prevalence'] * 100
    decile_stats['ci_lower'] = decile_stats.apply(
        lambda x: wilson_ci(x['cases'], x['total'])[0] * 100, axis=1)
    decile_stats['ci_upper'] = decile_stats.apply(
        lambda x: wilson_ci(x['cases'], x['total'])[1] * 100, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(decile_stats['decile'], decile_stats['prevalence_pct'],
                  color=DECILE_COLORS, edgecolor='white', linewidth=1.5, alpha=0.85)
    ax.errorbar(decile_stats['decile'], decile_stats['prevalence_pct'],
                yerr=[decile_stats['prevalence_pct'] - decile_stats['ci_lower'],
                      decile_stats['ci_upper'] - decile_stats['prevalence_pct']],
                fmt='none', ecolor='black', capsize=3, alpha=0.4, linewidth=1.5)
    
    ax.set_xlabel('Combined Genetic Risk Score (1=Lowest, 10=Highest)', fontsize=11)
    ax.set_ylabel('Observed Prevalence (%)', fontsize=11)
    ax.set_title(f'{disease} — Disease Rate by Genetic Risk Decile', fontsize=13, pad=15)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f'{i}\n(Low)' if i==1 else f'{i}\n(High)' if i==10 else str(i) 
                        for i in range(1, 11)], fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, (_, row) in zip(bars, decile_stats.iterrows()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'n={int(row["total"]):,}',
                ha='center', va='bottom', fontsize=7, color='black', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'../../{disease_key}_decile.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    df['genetic_risk_group'] = pd.cut(
        df['genetic_component'],
        bins=[df['genetic_component'].min(), df['genetic_component'].quantile(0.10),
              df['genetic_component'].quantile(0.90), df['genetic_component'].max()],
        labels=['Low (Bottom 10%)', 'Average (Middle 80%)', 'High (Top 10%)'],
        include_lowest=True)
    
    age_thresholds = np.arange(40, 91, 1)
    
    for sex_val, sex_label in [(0, 'Females'), (1, 'Males')]:
        df_sex = df[df['sex_binary'] == sex_val].copy()
        
        if len(df_sex) < 100:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_created = False
        for risk_group, color, label_base in [
            ('Low (Bottom 10%)', RISK_COLORS['low'], 'Low genetic risk (Bottom 10%)'),
            ('Average (Middle 80%)', RISK_COLORS['average'], 'Average genetic risk'),
            ('High (Top 10%)', RISK_COLORS['high'], 'High genetic risk (Top 10%)')]:
            
            df_group = df_sex[df_sex['genetic_risk_group'] == risk_group].copy()
            if len(df_group) < 10:
                continue
            
            cumulative_ages = []
            cumulative_risks = []
            cumulative_ns = []
            ci_lowers = []
            ci_uppers = []
            
            for age_threshold in age_thresholds:
                df_up_to_age = df_group[df_group['age'] <= age_threshold]
                
                if len(df_up_to_age) >= 20:
                    n_total = len(df_up_to_age)
                    n_cases = df_up_to_age['case'].sum()
                    cumulative_risk = (n_cases / n_total) * 100
                    
                    ci_low, ci_high = wilson_ci(n_cases, n_total)
                    
                    cumulative_ages.append(age_threshold)
                    cumulative_risks.append(cumulative_risk)
                    cumulative_ns.append(n_total)
                    ci_lowers.append(ci_low * 100)
                    ci_uppers.append(ci_high * 100)
            
            if len(cumulative_ages) < 3:
                continue
            
            total_n = df_group.shape[0]
            label_with_n = f'{label_base} (n={total_n:,})'
            
            linewidth = 3 if 'High' in risk_group else 2
            linestyle = '-' if 'Average' in risk_group else '--'
            
            ax.plot(cumulative_ages, cumulative_risks,
                   marker='o', label=label_with_n, color=color, linewidth=linewidth, 
                   linestyle=linestyle, markersize=4, alpha=0.9, markevery=5)
            ax.fill_between(cumulative_ages, ci_lowers, ci_uppers, 
                           alpha=0.15, color=color)
            
            plot_created = True
        
        if not plot_created:
            plt.close(fig)
            continue
        
        ax.set_xlabel('Age (years)', fontsize=11)
        ax.set_ylabel('Cumulative Risk (%)', fontsize=11)
        ax.set_title(f'{disease} — Cumulative Risk by Age and Genetics ({sex_label})', 
                    fontsize=13, pad=15)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(38, 92)
        
        plt.tight_layout()
        sex_suffix = 'female' if sex_val == 0 else 'male'
        plt.savefig(f'../../{disease_key}_age_{sex_suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    low_risk = df[df['genetic_component'] <= df['genetic_component'].quantile(0.10)]['case'].mean() * 100
    high_risk = df[df['genetic_component'] >= df['genetic_component'].quantile(0.90)]['case'].mean() * 100
    fold_change = high_risk / low_risk if low_risk > 0 else float('inf')
    summary_data.append({
        'Disease': disease,
        'Low 10%': low_risk,
        'High 10%': high_risk,
        'Fold Change': fold_change
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*70)
print("GENETIC RISK IMPACT SUMMARY")
print("="*70)
print(f"{'Disease':<22} {'Low 10%':>10} {'High 10%':>10} {'Fold Change':>12}")
print("-"*70)
for _, row in summary_df.iterrows():
    print(f"{row['Disease']:<22} {row['Low 10%']:>9.2f}% {row['High 10%']:>9.2f}% {row['Fold Change']:>11.2f}x")
print("="*70)
```

Let's say we want to run these scores on a new sample with a different genome build (hg37). First, we download the scores into a new directory `scorefiles`:
```
mkdir -p scorefiles && printf "%s\n" PGS004146 PGS004898 PGS003334 PGS003852 PGS005199 PGS004150 PGS000007 PGS000508 PGS000332 | xargs -I {} -P 4 wget -P scorefiles "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{}/ScoringFiles/Harmonized/{}_hmPOS_GRCh37.txt.gz"
```

Unzip them:
```
ls scorefiles/*.gz | xargs -P 4 -n 1 gunzip
```

Let's assume we have files called multiallelic.bed, multiallelic.bim, and multiallelic.fam in the current directory. We can now run:

```
./gnomon/target/release/gnomon score scorefiles multiallelic
```

The "scorefiles" argument means to look for the scorefiles in the scorefiles directory, while the "multiallelic" argument specifies the genotype file prefix.



