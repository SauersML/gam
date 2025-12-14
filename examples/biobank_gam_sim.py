import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse


if not hasattr(np, 'int'):
    np.int = int

# pygam uses both csc_matrix and csr_matrix internally, so we patch both.
if not hasattr(scipy.sparse.csc_matrix, 'A'):
    scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())
if not hasattr(scipy.sparse.csr_matrix, 'A'):
    scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
# --- End of patches ---


# A GAM is used for the non-linear fit, so we need the pygam library
# You may need to install it: pip install pygam
from pygam import LinearGAM, s, te

# --- 1. Style, Colors, and Data Generation ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.weight': 'bold', 'axes.labelweight': 'bold',
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 26,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 20,
    'figure.titlesize': 32
})
# A professional, high-contrast, and colorblind-safe palette
COLOR_GAM_FIT = '#333333'     # Dark Gray: The sophisticated model fit
COLOR_EMPIRICAL = '#0072B2'   # Vibrant Blue: The data-driven evidence
np.random.seed(42)

# --- REVISED DATA GENERATION ---
# Generate 20k data points where ERROR is lowest at the center of the PC range
# and increases towards the extremes (a "U-shaped" or parabolic error).
N = 20000
beta, sigma_g_sq = 0.5, 1.0

# Std dev = c0_std + c1_std * (PC - 0.5)^2. This creates the lowest error at PC=0.5.
c0_std = 0.5  # Minimum standard deviation (at the center)
c1_std = 8.0  # Curvature (how fast error increases towards extremes)

PC = np.random.uniform(0, 1, N)
G = np.random.normal(0, np.sqrt(sigma_g_sq), N)
Y = beta * G

# The error term now follows the new U-shaped pattern
error_std = c0_std + c1_std * (PC - 0.5)**2
P = G + np.random.normal(0, error_std, N) # P is the PGS (proxy for G)

# --- 2. Figure Creation (Top-Bottom Layout) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), sharex=True)

fig.suptitle('Generalized Additive Models Calibrate Polygenic Score Attenuation', fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.35)

# --- PLOT 1 (TOP): The Problem - Prediction Error vs. PC ---
ax1.set_title('Prediction Error Increases With Distance from Biobank Sample (middle)', fontweight='bold')
prediction_error = P - Y

# Subsample 1000 points for a clean scatter plot
plot_idx = np.random.choice(N, 1000, replace=False)
scatter = ax1.scatter(PC[plot_idx], prediction_error[plot_idx],
                      c=PC[plot_idx], cmap='plasma', alpha=0.7, s=50)

ax1.set_ylabel('Prediction Error')
ax1.grid(True, which='both', linestyle=':', linewidth=1.0)

# --- PLOT 2 (BOTTOM): The Solution - GAM-derived Coefficient ---
ax2.set_title('GAM Recovers the Optimal PGS Calibration', fontweight='bold')

# A) Empirical Binned Coefficient (The "ground truth" from the data)
n_bins = 10
bin_edges = np.linspace(PC.min(), PC.max(), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
empirical_coeffs = [
    np.linalg.lstsq(P[(PC >= bin_edges[i]) & (PC < bin_edges[i+1])][:, np.newaxis],
                    Y[(PC >= bin_edges[i]) & (PC < bin_edges[i+1])], rcond=None)[0][0]
    for i in range(n_bins)
]


# B) REVISED FIT: Generalized Additive Model (GAM) using gridsearch
# In pygam, using REML or GCV is achieved by tuning the smoothing parameter `lam`.
# The `gridsearch` method automatically finds the best `lam` values from the data.
X_gam = np.c_[P, PC]
gam = LinearGAM(s(0, n_splines=15) + s(1, n_splines=15) + te(0, 1, n_splines=[15, 15]))
gam.gridsearch(X_gam, Y) # This tunes the model, analogous to REML

# To find the GAM-derived coefficient for P, we see how the prediction changes
# for a one-unit increase in P across the range of PC values.
pc_smooth = np.linspace(PC.min(), PC.max(), 200)
# Create two grids: one with P=0 and one with P=1
grid_p0 = np.c_[np.zeros_like(pc_smooth), pc_smooth]
grid_p1 = np.c_[np.ones_like(pc_smooth), pc_smooth]

# Predict Y for both grids
pred_p0 = gam.predict(grid_p0)
pred_p1 = gam.predict(grid_p1)

# The difference in predictions is the effective coefficient of P at each PC value
gam_derived_coefficient = pred_p1 - pred_p0


ax2.plot(bin_centers, empirical_coeffs, 'o', color=COLOR_EMPIRICAL, markersize=18,
         markeredgecolor='white', markeredgewidth=2.0, label='Empirical Optimum (Binned)', zorder=12)

ax2.plot(pc_smooth, gam_derived_coefficient, color=COLOR_GAM_FIT, linewidth=8,
         label='GAM-derived Coefficient', zorder=10)

ax2.set_xlabel('Principal Component Value')
ax2.set_ylabel('PGS Coefficient')

# LEGEND LOCATION AND LAYOUT
# ncol is set to 1 to stack legend items vertically.
ax2.legend(loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.05))

ax2.grid(True, which='both', linestyle=':', linewidth=1.0)
ax2.set_ylim(0, max(np.max(empirical_coeffs), np.max(gam_derived_coefficient)) * 1.2)

# --- 3. Final Touches ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("gam_fit_figure.png", dpi=300, bbox_inches='tight')
plt.show()
