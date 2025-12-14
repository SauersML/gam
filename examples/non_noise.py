import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
from pygam import LinearGAM, s, te
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- PyGam Patches for Compatibility with newer NumPy/SciPy versions ---
# This ensures the code runs on modern library versions without errors.
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(scipy.sparse.csc_matrix, 'A'):
    scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())
if not hasattr(scipy.sparse.csr_matrix, 'A'):
    scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())

# --- 1. Global Style and Helper Functions ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.weight': 'bold', 'axes.labelweight': 'bold',
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 26,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18,
    'figure.titlesize': 32
})
# A professional, high-contrast, and colorblind-safe palette
COLOR_TRUE_EFFECT = '#333333'      # Dark Gray: The ground truth
COLOR_GAM_FIT = '#0072B2'   # Vibrant Blue: The flexible GAM fit
COLOR_LINEAR_FAIL = '#D55E00'      # Burnt Orange: The flawed linear model
np.random.seed(1)

def format_p_value(p, n_bootstraps):
    """Formats p-values to avoid rounding to zero and show bootstrap limit."""
    limit = 1.0 / n_bootstraps
    if p < limit:
        return f"< {limit:.1e}"
    return f"{p:.3f}"

def bootstrap_significance_test(y_true, preds_base, preds_gam, n_bootstraps=1000, metric='r2'):
    """
    Performs a bootstrap test for the difference in a performance metric.
    Calculates the p-value for the hypothesis that the GAM is better than the base model.
    """
    n_samples = len(y_true)
    delta_metrics = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_boot = y_true[indices]
        preds_base_boot, preds_gam_boot = preds_base[indices], preds_gam[indices]

        if metric == 'r2':
            # We want R2(GAM) - R2(Base) to be > 0
            metric_base = r2_score(y_boot, preds_base_boot)
            metric_gam = r2_score(y_boot, preds_gam_boot)
            delta_metrics.append(metric_gam - metric_base)
        elif metric == 'mse':
            # We want MSE(Base) - MSE(GAM) to be > 0 (lower MSE is better)
            metric_base = mean_squared_error(y_boot, preds_base_boot)
            metric_gam = mean_squared_error(y_boot, preds_gam_boot)
            delta_metrics.append(metric_base - metric_gam)

    delta_metrics = np.array(delta_metrics)
    # The p-value is the proportion of times the GAM was NOT better than the base model
    p_value = np.mean(delta_metrics <= 0)
    
    mean_improvement = np.mean(delta_metrics)
    return p_value, mean_improvement


def run_interaction_simulation():
    """
    PART 1: Test if a GAM can recover a true, non-linear INTERACTION
    between PGS and PC.

    In this simulation:
    - The effect of the PGS on the phenotype is a sinusoidal function of the PC.
    - There is NO heteroscedasticity (i.e., noise is constant).
    - We compare a simple model, a linear interaction model, and a GAM.
    """
    print("\n" + "="*80)
    print("--- Running Simulation 1: Recovering a True Non-Linear Interaction ---")
    print("="*80 + "\n")

    # --- Data Generation ---
    N = 20000
    PC = np.random.uniform(0, 1, N)
    P = np.random.normal(0, 1, N) # PGS

    true_beta_P_fn = 0.4 + 0.8 * np.sin(PC * np.pi * 2)
    Y = true_beta_P_fn * P + np.random.normal(0, 0.75, N)

    # --- Data Splitting ---
    X = np.c_[P, PC]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
    P_train, PC_train = X_train[:, 0], X_train[:, 1]
    P_test, PC_test = X_test[:, 0], X_test[:, 1]

    # --- Model Training ---
    print("Training models...")
    # Model 1: Basic (ignores PC and interaction)
    model_basic = LinearRegression().fit(P_train.reshape(-1, 1), Y_train)
    # Model 2: Linear interaction (misspecifies the interaction form)
    model_linear_int = LinearRegression().fit(np.c_[P_train, PC_train, P_train * PC_train], Y_train)
    # Model 3: GAM (can flexibly model the 2D interaction surface)
    gam = LinearGAM(te(0, 1, n_splines=[15, 15])).gridsearch(X_train, Y_train)
    print("Training complete.\n")

    # --- Model Evaluation & Significance Testing ---
    y_pred_basic = model_basic.predict(P_test.reshape(-1, 1))
    y_pred_linear_int = model_linear_int.predict(np.c_[P_test, PC_test, P_test * PC_test])
    y_pred_gam = gam.predict(X_test)
    
    metrics = [
        {'Model': 'Basic (PGS only)', 'R²': r2_score(Y_test, y_pred_basic), 'MSE': mean_squared_error(Y_test, y_pred_basic)},
        {'Model': 'Linear Interaction', 'R²': r2_score(Y_test, y_pred_linear_int), 'MSE': mean_squared_error(Y_test, y_pred_linear_int)},
        {'Model': 'GAM (Tensor Spline)', 'R²': r2_score(Y_test, y_pred_gam), 'MSE': mean_squared_error(Y_test, y_pred_gam)}
    ]
    metrics_df = pd.DataFrame(metrics).set_index('Model')
    print("--- Predictive Performance on Test Set ---")
    print(metrics_df.to_string(formatters={'R²': '{:.4f}'.format, 'MSE': '{:.4f}'.format}))

    print("\n--- Significance of GAM Improvement (Bootstrap Test) ---")
    n_boots = 1000
    p_vs_linear, r2_imp_linear = bootstrap_significance_test(Y_test, y_pred_linear_int, y_pred_gam, n_boots, 'r2')
    print(f"GAM vs. Linear Interaction: p-value(R²) = {format_p_value(p_vs_linear, n_boots)} (Mean ΔR² = {r2_imp_linear:+.4f})")
    p_vs_basic, r2_imp_basic = bootstrap_significance_test(Y_test, y_pred_basic, y_pred_gam, n_boots, 'r2')
    print(f"GAM vs. Basic PGS Model:    p-value(R²) = {format_p_value(p_vs_basic, n_boots)} (Mean ΔR² = {r2_imp_basic:+.4f})\n")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.suptitle('GAM Recovers True Non-Linear PGS x PC Interaction', fontweight='bold', y=0.98)
    ax.set_title('The effect of PGS varies non-linearly with PC value', fontweight='bold')
    
    pc_smooth = np.linspace(0, 1, 200)
    true_coef_smooth = 0.4 + 0.8 * np.sin(pc_smooth * np.pi * 2)
    ax.plot(pc_smooth, true_coef_smooth, color=COLOR_TRUE_EFFECT, linewidth=8, label='True Underlying Coefficient', zorder=10)
    
    coef_p, _, coef_int = model_linear_int.coef_
    linear_fit_coefficient = coef_p + coef_int * pc_smooth
    ax.plot(pc_smooth, linear_fit_coefficient, color=COLOR_LINEAR_FAIL, linestyle='--', linewidth=6, label='Misspecified Linear Fit', zorder=8)
    
    grid_p0 = np.c_[np.zeros_like(pc_smooth), pc_smooth]
    grid_p1 = np.c_[np.ones_like(pc_smooth), pc_smooth]
    gam_derived_coefficient = gam.predict(grid_p1) - gam.predict(grid_p0)
    ax.plot(pc_smooth, gam_derived_coefficient, color=COLOR_GAM_FIT, linewidth=6, linestyle=':', label='GAM-derived Coefficient', zorder=9)
    
    ax.set_xlabel('Principal Component Value')
    ax.set_ylabel('Effective PGS Coefficient')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle=':', linewidth=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("gam_interaction_recovery.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_nonlinear_pc_effect_simulation():
    """
    PART 2: Test if a GAM can recover a true, non-linear DIRECT EFFECT of a PC
    on the phenotype, thereby improving the PGS effect estimate.

    In this simulation:
    - The PC has a direct, non-linear (cosine) effect on the phenotype (confounding).
    - The PGS has a CONSTANT, linear effect on the phenotype (no interaction).
    - Correlation is induced between P and PC to demonstrate omitted-variable bias.
    """
    print("\n" + "="*80)
    print("--- Running Simulation 2: Recovering a Non-Linear PC Effect ---")
    print("="*80 + "\n")

    # --- Data Generation ---
    N = 20000
    true_beta_P = 0.8

    PC = np.random.uniform(0, 1, N)
    # Induce P~PC correlation to create confounding
    P = 0.5 * PC + np.random.normal(0, np.sqrt(1 - 0.5**2), N)
    
    true_pc_effect_fn = 1.5 * np.cos((PC - 0.5) * np.pi * 2)
    Y = true_beta_P * P + true_pc_effect_fn + np.random.normal(0, 0.75, N)

    # --- Data Splitting ---
    X = np.c_[P, PC]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
    P_train, PC_train = X_train[:, 0], X_train[:, 1]
    
    # --- Model Training ---
    print("Training models...")
    # Model 1: Basic (omitted-variable bias)
    model_basic = LinearRegression().fit(P_train.reshape(-1, 1), Y_train)
    # Model 2: Linear adjustment (misspecifies PC effect)
    model_linear_adj = LinearRegression().fit(np.c_[P_train, PC_train], Y_train)
    # Model 3: GAM (flexibly models PC effect)
    gam = LinearGAM(s(0) + s(1)).gridsearch(X_train, Y_train)
    print("Training complete.\n")
    
    # --- Model Evaluation (Coefficient Recovery) ---
    p_vals = np.array([-0.5, 0.5])
    pdep_p = gam.partial_dependence(term=0, X=np.c_[p_vals, np.zeros_like(p_vals)])
    gam_p_coeff = (pdep_p[1] - pdep_p[0]) / (p_vals[1] - p_vals[0])

    results = [
        {'Model': 'Basic (PGS only)', 'Est. PGS Coeff.': model_basic.coef_[0]},
        {'Model': 'Linear PC Adjustment', 'Est. PGS Coeff.': model_linear_adj.coef_[0]},
        {'Model': 'GAM PC Adjustment', 'Est. PGS Coeff.': gam_p_coeff}
    ]
    results_df = pd.DataFrame(results).set_index('Model')
    results_df['True PGS Coeff.'] = true_beta_P
    results_df['Error'] = abs(results_df['Est. PGS Coeff.'] - results_df['True PGS Coeff.'])
    print("--- Recovery of True PGS Coefficient ---")
    print(results_df.to_string(formatters={'Est. PGS Coeff.': '{:.4f}'.format, 'Error': '{:.4f}'.format}))

    # --- Predictive Performance & Significance ---
    y_pred_basic = model_basic.predict(X_test[:, 0].reshape(-1, 1))
    y_pred_linear_adj = model_linear_adj.predict(X_test)
    y_pred_gam = gam.predict(X_test)
    
    perf_metrics = [
        {'Model': 'Basic (PGS only)', 'R²': r2_score(Y_test, y_pred_basic), 'MSE': mean_squared_error(Y_test, y_pred_basic)},
        {'Model': 'Linear PC Adjustment', 'R²': r2_score(Y_test, y_pred_linear_adj), 'MSE': mean_squared_error(Y_test, y_pred_linear_adj)},
        {'Model': 'GAM PC Adjustment', 'R²': r2_score(Y_test, y_pred_gam), 'MSE': mean_squared_error(Y_test, y_pred_gam)}
    ]
    perf_df = pd.DataFrame(perf_metrics).set_index('Model')
    print("\n--- Predictive Performance on Test Set ---")
    print(perf_df.to_string(formatters={'R²': '{:.4f}'.format, 'MSE': '{:.4f}'.format}))
    
    print("\n--- Significance of GAM Improvement (Bootstrap Test) ---")
    n_boots = 1000
    p_vs_linear, r2_imp_linear = bootstrap_significance_test(Y_test, y_pred_linear_adj, y_pred_gam, n_boots, 'r2')
    print(f"GAM vs. Linear Adjustment: p-value(R²) = {format_p_value(p_vs_linear, n_boots)} (Mean ΔR² = {r2_imp_linear:+.4f})")
    p_vs_basic, r2_imp_basic = bootstrap_significance_test(Y_test, y_pred_basic, y_pred_gam, n_boots, 'r2')
    print(f"GAM vs. Basic PGS Model:   p-value(R²) = {format_p_value(p_vs_basic, n_boots)} (Mean ΔR² = {r2_imp_basic:+.4f})\n")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.suptitle('GAM Corrects for Non-Linear Confounding by PC', fontweight='bold', y=0.98)
    ax.set_title('The PC has a direct, non-linear effect on the phenotype', fontweight='bold')

    pc_smooth = np.linspace(0, 1, 200)
    true_pc_effect_smooth = 1.5 * np.cos((pc_smooth - 0.5) * np.pi * 2)
    ax.plot(pc_smooth, true_pc_effect_smooth, color=COLOR_TRUE_EFFECT, linewidth=8, label='True PC Effect', zorder=10)

    linear_pc_effect = model_linear_adj.coef_[1] * (pc_smooth - np.mean(PC_train))
    ax.plot(pc_smooth, linear_pc_effect, color=COLOR_LINEAR_FAIL, linestyle='--', linewidth=6, label='Misspecified Linear Fit for PC', zorder=8)
    
    pdep_pc, _ = gam.partial_dependence(term=1, X=np.c_[np.zeros_like(pc_smooth), pc_smooth], width=0.95)
    ax.plot(pc_smooth, pdep_pc, color=COLOR_GAM_FIT, linewidth=6, linestyle=':', label='GAM-derived PC Effect', zorder=9)

    ax.set_xlabel('Principal Component Value')
    ax.set_ylabel('Effect on Phenotype Y')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle=':', linewidth=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("gam_nonlinear_confounding.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Run the first simulation testing for a true interaction effect
    run_interaction_simulation()

    # Run the second simulation testing for a direct, non-linear PC effect
    run_nonlinear_pc_effect_simulation()
