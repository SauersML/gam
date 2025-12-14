import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
from pygam import LogisticGAM, s, te
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# --- PyGam Patches for Compatibility ---
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(scipy.sparse.csc_matrix, 'A'):
    scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())
if not hasattr(scipy.sparse.csr_matrix, 'A'):
    scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())

# --- Helper Functions for Metrics and P-values ---
def nagelkerkes_r2(y_true, y_prob):
    """Calculates Nagelkerke's R-squared for model performance."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(y_true) < 2: return np.nan
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    p_mean = np.mean(y_true)
    if p_mean == 0 or p_mean == 1: return 0.0
    ll_null = np.sum(y_true * np.log(p_mean) + (1 - y_true) * np.log(1 - p_mean))
    ll_model = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    n = len(y_true)
    r2_cs = 1 - np.exp((2/n) * (ll_null - ll_model))
    max_r2_cs = 1 - np.exp((2/n) * ll_null)
    return r2_cs / max_r2_cs if max_r2_cs > 0 else 0.0

def calculate_and_store_metrics(model_name, y_true, y_prob, pc_values, results_list):
    """Calculates metrics for global, bottom 20%, and top 20% PC groups."""
    bottom_thresh = np.percentile(pc_values, 20)
    top_thresh = np.percentile(pc_values, 80)
    
    groups = {
        'Global': np.ones_like(y_true, dtype=bool),
        'Bottom 20% of Individuals (Low PC)': pc_values <= bottom_thresh,
        'Top 20% of Individuals (High PC)': pc_values >= top_thresh
    }
    
    for group_name, mask in groups.items():
        auc = roc_auc_score(y_true[mask], y_prob[mask]) if len(np.unique(y_true[mask])) > 1 else np.nan
        results_list.append({
            'Model': model_name, 'Group': group_name, 'AUC': auc,
            'Brier Score': brier_score_loss(y_true[mask], y_prob[mask]),
            "Nagelkerke's R²": nagelkerkes_r2(y_true[mask], y_prob[mask])
        })

def format_p_value(p, n_bootstraps):
    """Formats p-values to avoid rounding to zero."""
    limit = 1.0 / n_bootstraps
    if p < limit:
        return f"< {limit:.1e}"
    return f"{p:.2e}"

def bootstrap_significance_test(y_true, probs_basic, probs_gam, n_bootstraps=2000):
    """Performs bootstrap test for difference in metrics between two models."""
    n_samples = len(y_true)
    auc_diffs, brier_diffs, r2_diffs = [], [], []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_boot = y_true[indices]
        if len(np.unique(y_boot)) < 2: continue
            
        probs_basic_boot, probs_gam_boot = probs_basic[indices], probs_gam[indices]

        auc_diffs.append(roc_auc_score(y_boot, probs_gam_boot) - roc_auc_score(y_boot, probs_basic_boot))
        brier_diffs.append(brier_score_loss(y_boot, probs_basic_boot) - brier_score_loss(y_boot, probs_gam_boot))
        r2_diffs.append(nagelkerkes_r2(y_boot, probs_gam_boot) - nagelkerkes_r2(y_boot, probs_basic_boot))

    p_auc = 2 * min(np.mean(np.array(auc_diffs) <= 0), np.mean(np.array(auc_diffs) > 0))
    p_brier = 2 * min(np.mean(np.array(brier_diffs) <= 0), np.mean(np.array(brier_diffs) > 0))
    p_r2 = 2 * min(np.mean(np.array(r2_diffs) <= 0), np.mean(np.array(r2_diffs) > 0))

    return (format_p_value(p_auc, n_bootstraps), 
            format_p_value(p_brier, n_bootstraps), 
            format_p_value(p_r2, n_bootstraps))

# --- 1. Style, Colors, and Data Generation ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'sans-serif', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 26, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18, 'figure.titlesize': 32})
COLOR_GAM_FIT, COLOR_EMPIRICAL, COLOR_LINEAR_FAIL = '#333333', '#0072B2', '#D55E00'
np.random.seed(42)

# --- REVISED DATA GENERATION with SKEWED PC DISTRIBUTION ---
print("--- Generating Simulated Data with Skewed PC Distribution ---")
N = 100000
beta, gamma, sigma_g_sq = 0.8, 0.4, 1.0
c0_std, c1_std = 0.05, 5.0

target_quantile, target_prob = 0.2, 0.8
k = np.log(target_prob) / np.log(target_quantile)
u = np.random.uniform(0, 1, N)
PC = u**(1/k)

G = np.random.normal(0, np.sqrt(sigma_g_sq), N)
P = G + np.random.normal(0, c0_std + c1_std * PC, N)
prob_Y = 1 / (1 + np.exp(-(beta * G + gamma * PC)))
Y = np.random.binomial(1, prob_Y)

print(f"Generated {N} samples with a STEEP noise gradient and SKEWED PC distribution.")
print(f"Verification: Percentage of samples with PC <= {target_quantile}: {np.mean(PC <= target_quantile):.2%}")
print(f"Overall prevalence of the outcome: {np.mean(Y):.2%}\n")

# --- 2. Data Splitting (50/50 split) ---
(P_train, P_test, PC_train, PC_test, Y_train, Y_test) = train_test_split(
    P, PC, Y, test_size=0.5, random_state=42
)
print(f"Data split into {len(Y_train)} training and {len(Y_test)} testing samples.\n")

# --- 3. Model Implementation ---
results = []
print("--- Training Model 1: Basic PGS Model ---")
model1 = LogisticRegression(solver='liblinear').fit(P_train.reshape(-1, 1), Y_train)
probs1 = model1.predict_proba(P_test.reshape(-1, 1))[:, 1]
calculate_and_store_metrics('1: Basic (PGS Only)', Y_test, probs1, PC_test, results)
print("Model 1 complete.\n")

print("--- Training Model 2: Normalized PGS (Regress Out PCs) ---")
pc_regressor = LinearRegression().fit(PC_train.reshape(-1, 1), P_train)
P_adj_train = P_train - pc_regressor.predict(PC_train.reshape(-1, 1))
P_adj_test = P_test - pc_regressor.predict(PC_test.reshape(-1, 1))
model2 = LogisticRegression(solver='liblinear').fit(P_adj_train.reshape(-1, 1), Y_train)
probs2 = model2.predict_proba(P_adj_test.reshape(-1, 1))[:, 1]
calculate_and_store_metrics('2: Normalized PGS', Y_test, probs2, PC_test, results)
print("Model 2 complete.\n")

print("--- Training Model 3: Our GAM ---")
X_train_gam = np.c_[P_train, PC_train]
model3 = LogisticGAM(s(0, n_splines=15) + s(1, n_splines=15) + te(0, 1, n_splines=[15, 15]))
model3.gridsearch(X_train_gam, Y_train)
probs3 = model3.predict_proba(np.c_[P_test, PC_test])
calculate_and_store_metrics('3: Our GAM', Y_test, probs3, PC_test, results)
print("Model 3 complete.\n")

# --- 4. Significance Testing via Bootstrapping ---
print("--- Performing Significance Testing (GAM vs Basic PGS) ---")
significance_results = []
bottom_thresh = np.percentile(PC_test, 20)
top_thresh = np.percentile(PC_test, 80)
groups_masks = {
    'Global': np.ones_like(Y_test, dtype=bool),
    'Bottom 20% of Individuals (Low PC)': PC_test <= bottom_thresh,
    'Top 20% of Individuals (High PC)': PC_test >= top_thresh
}
n_boots = 2000

for group_name, mask in groups_masks.items():
    print(f"Bootstrapping for group: {group_name} ({np.sum(mask)} samples)...")
    p_auc, p_brier, p_r2 = bootstrap_significance_test(
        Y_test[mask], probs1[mask], probs3[mask], n_bootstraps=n_boots
    )
    significance_results.append({
        'Group': group_name, 'p_AUC': p_auc, 'p_Brier': p_brier, 'p_R2': p_r2
    })

# --- 5. Display Final Metric Results and Save to Disk ---
results_df = pd.DataFrame(results).set_index(['Model', 'Group'])
significance_df = pd.DataFrame(significance_results).set_index('Group')

print("\n" + "="*80)
print("--- Final Model Comparison (Point Estimates) ---")
print(results_df)
print("\n" + "-"*80)
print("--- Significance of Improvement: GAM vs Basic PGS (p-values) ---")
print(significance_df)
print("="*80 + "\n")

results_filename = "simulation_results.csv"
significance_filename = "significance_results.csv"
results_df.to_csv(results_filename)
significance_df.to_csv(significance_filename)
print(f"Point estimate metrics saved to '{results_filename}'")
print(f"Significance test results saved to '{significance_filename}'")

# --- 6. Plot Generation ---
print("\n--- Generating Plots ---")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), sharex=True)
fig.suptitle('GAMs Correct for Ancestry-Dependent PGS Error', fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.35)

ax1.set_title('PGS Error Increases Systematically with PC Value', fontweight='bold')
prediction_error = P_test - Y_test
plot_idx = np.random.choice(len(P_test), 1000, replace=False)
ax1.scatter(PC_test[plot_idx], prediction_error[plot_idx], c=PC_test[plot_idx], cmap='plasma', alpha=0.7, s=50)
ax1.set_ylabel('Prediction Error (P - Y)')
ax1.grid(True, which='both', linestyle=':', linewidth=1.0)

ax2.set_title('GAM Recovers the Optimal PGS Calibration Curve', fontweight='bold')
pc_smooth = np.linspace(PC.min(), PC.max(), 200)
grid_p0 = np.c_[np.zeros_like(pc_smooth), pc_smooth]
grid_p1 = np.c_[np.ones_like(pc_smooth), pc_smooth]
epsilon = 1e-10
probs_p0 = np.clip(model3.predict_mu(grid_p0), epsilon, 1 - epsilon)
probs_p1 = np.clip(model3.predict_mu(grid_p1), epsilon, 1 - epsilon)
pred_p0_link = np.log(probs_p0 / (1 - probs_p0))
pred_p1_link = np.log(probs_p1 / (1 - probs_p1))
gam_derived_coefficient = pred_p1_link - pred_p0_link
ax2.plot(pc_smooth, gam_derived_coefficient, color=COLOR_GAM_FIT, linewidth=8, label='GAM-derived Coefficient', zorder=10)

# --- FIX IS HERE ---
# Create the 3-feature data for the linear interaction model
X_linear_train = np.c_[P_train, PC_train, P_train * PC_train]
# Train the model on the correct 3-feature data
lin_model = LogisticRegression(solver='liblinear').fit(X_linear_train, Y_train)
# Unpack the 3 coefficients correctly
coef_p, _, coef_int = lin_model.coef_[0]
linear_fit_coefficient = coef_p + coef_int * pc_smooth
ax2.plot(pc_smooth, linear_fit_coefficient, color=COLOR_LINEAR_FAIL, linestyle='--', linewidth=6, label='Misspecified Linear Fit', zorder=8)

n_bins = 10
bin_edges = np.linspace(PC_train.min(), PC_train.max(), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
empirical_coeffs = []
for i in range(n_bins):
    mask = (PC_train >= bin_edges[i]) & (PC_train < bin_edges[i+1])
    if np.sum(mask) > 50:
        binned_model = LogisticRegression(solver='liblinear').fit(P_train[mask].reshape(-1, 1), Y_train[mask])
        empirical_coeffs.append(binned_model.coef_[0][0])
    else:
        empirical_coeffs.append(np.nan)
ax2.plot(bin_centers, empirical_coeffs, 'o', color=COLOR_EMPIRICAL, markersize=18, markeredgecolor='white', markeredgewidth=2.0, label='Empirical Optimum (Binned)', zorder=12)

ax2.set_xlabel('Principal Component Value')
ax2.set_ylabel('Effective PGS Coefficient\n(on Log-Odds Scale)')
ax2.legend(loc='upper right')
ax2.grid(True, which='both', linestyle=':', linewidth=1.0)
ax2.set_ylim(0, max(np.nanmax(empirical_coeffs), np.nanmax(gam_derived_coefficient)) * 1.2 if not np.all(np.isnan(empirical_coeffs)) else 1.0)

# --- 7. Final Touches ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("gam_metrics_and_plot_figure.png", dpi=300, bbox_inches='tight')
print("\nPlot saved to gam_metrics_and_plot_figure.png")
plt.show()
