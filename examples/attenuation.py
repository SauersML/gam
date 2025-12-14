import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- 1. Style, Colors, and Data Generation ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.weight': 'bold', 'axes.labelweight': 'bold',
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 26,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 20,
    'figure.titlesize': 32
})
# A professional, high-contrast, and colorblind-safe palette
COLOR_THEORY = '#333333'      # Dark Gray: The ground truth
COLOR_EMPIRICAL = '#0072B2'   # Vibrant Blue: The data-driven evidence
COLOR_LINEAR = '#C42169'      # Assertive Magenta: The flawed model
np.random.seed(42)

# Generate 20k data points where the ERROR STD DEV increases linearly with PC
N = 20000
beta, sigma_g_sq = 0.5, 1.0
# Std dev = c0_std + c1_std * PC. Creates good predictions at low PC
c0_std, c1_std = 0.1, 3.5

PC = np.random.uniform(0, 1, N)
G = np.random.normal(0, np.sqrt(sigma_g_sq), N)
Y = beta * G
error_std = c0_std + c1_std * PC
P = G + np.random.normal(0, error_std, N)

# --- 2. Figure Creation (Top-Bottom Layout) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), sharex=True)
fig.suptitle('Linear Models Fail When PGS Error Varies', fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.35)

# --- PLOT 1 (TOP): The Problem - Prediction Error vs. PC ---
ax1.set_title('Prediction Error Increases with PC Value', fontweight='bold')
prediction_error = P - Y

# Subsample 1000 points for a clean scatter plot
plot_idx = np.random.choice(N, 1000, replace=False)
scatter = ax1.scatter(PC[plot_idx], prediction_error[plot_idx],
                      c=PC[plot_idx], cmap='plasma', alpha=0.7, s=50)

ax1.set_ylabel('Prediction Error')
ax1.grid(True, which='both', linestyle=':', linewidth=1.0)

# --- PLOT 2 (BOTTOM): The Solution - Non-Linear Coefficient ---
ax2.set_title('A Non-Linear Model is Required to Recover the Coefficient', fontweight='bold')

# A) Theoretical Optimal Coefficient
pc_smooth = np.linspace(PC.min(), PC.max(), 200)
var_error = (c0_std + c1_std * pc_smooth)**2
optimal_coefficient = (beta * sigma_g_sq) / (sigma_g_sq + var_error)
ax2.plot(pc_smooth, optimal_coefficient, color=COLOR_THEORY, linewidth=8,
         label='Theoretical Optimum', zorder=10)

# B) Misspecified Linear Interaction Model
X_linear = np.c_[P, PC, P * PC]
lin_model = LinearRegression().fit(X_linear, Y)
coef_p, _, coef_int = lin_model.coef_
linear_fit_coefficient = coef_p + coef_int * pc_smooth
ax2.plot(pc_smooth, linear_fit_coefficient, color=COLOR_LINEAR, linestyle='--', linewidth=6,
         label='Misspecified Linear Fit', zorder=8)

# C) Empirical Binned Coefficient (Plotted on top)
n_bins = 10
bin_edges = np.linspace(PC.min(), PC.max(), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
empirical_coeffs = [
    np.linalg.lstsq(P[(PC >= bin_edges[i]) & (PC < bin_edges[i+1])][:, np.newaxis],
                    Y[(PC >= bin_edges[i]) & (PC < bin_edges[i+1])], rcond=None)[0][0]
    for i in range(n_bins)
]
ax2.plot(bin_centers, empirical_coeffs, 'o', color=COLOR_EMPIRICAL, markersize=18,
         markeredgecolor='white', markeredgewidth=2.0, label='Empirical Optimum', zorder=12)

ax2.set_xlabel('Principal Component Value')
ax2.set_ylabel('PGS Coefficient')
ax2.legend(loc='upper right')
ax2.grid(True, which='both', linestyle=':', linewidth=1.0)
ax2.set_ylim(0, max(empirical_coeffs) * 1.2)

# --- 3. Final Touches ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("final_corrected_figure.png", dpi=300, bbox_inches='tight')
plt.show()
