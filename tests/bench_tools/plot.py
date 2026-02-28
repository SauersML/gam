#!/usr/bin/env python3
"""
A test script to simulate data, train a GAM model using the 'gnomon' Rust
executable, evaluate its performance against a baseline, and generate
a comprehensive set of analysis plots.
"""

import subprocess
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import platform

# Configure matplotlib backend BEFORE importing pyplot
import matplotlib
if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
    print("Running in headless mode. Setting non-interactive backend.")
    matplotlib.use('Agg')  # Must happen before importing pyplot

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# --- Path Configuration ---
# Get the absolute path of the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up to the project root directory (assumes tests/ is under calibrate/)
PROJECT_ROOT = SCRIPT_DIR.parent
# Get the workspace root (one level up from the calibrate directory)
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Define all paths relative to the project root
_is_win = platform.system().lower().startswith("win")
_preferred_bins = ["gam.exe", "gnomon.exe"] if _is_win else ["gam", "gnomon"]
EXECUTABLE_NAME = _preferred_bins[0]
EXECUTABLE_PATH = WORKSPACE_ROOT / "target" / "release" / EXECUTABLE_NAME
MODEL_PATH = PROJECT_ROOT / "model.toml"
PREDICTIONS_PATH = PROJECT_ROOT / "predictions.tsv"  # The fixed output file for the tool
TRAIN_DATA_PATH = PROJECT_ROOT / "training_data.tsv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.tsv"
GRID_DATA_PATH = PROJECT_ROOT / "grid_data_for_surface.tsv"


# --- Simulation Configuration ---
N_SAMPLES_TRAIN = 5000
N_SAMPLES_TEST = 1000
NUM_PCS = 1  # Simulate 1 PC for easy 2D visualization
SIMULATION_SIGNAL_STRENGTH = 0.6 # The coefficient for the true logit function


# --- Helper Functions ---

def build_rust_project():
    """Checks for an executable and compiles the Rust project if not found."""
    global EXECUTABLE_NAME, EXECUTABLE_PATH
    for candidate in _preferred_bins:
        candidate_path = WORKSPACE_ROOT / "target" / "release" / candidate
        if candidate_path.is_file():
            EXECUTABLE_NAME = candidate
            EXECUTABLE_PATH = candidate_path
            print(f"--- Found existing executable '{EXECUTABLE_NAME}'. ---")
            return

    print("--- Executable not found. Compiling Rust Project... ---")
    for candidate in _preferred_bins:
        try:
            subprocess.run(
                ["cargo", "build", "--release", "--bin", candidate],
                check=True,
                text=True,
                cwd=WORKSPACE_ROOT,
            )
        except subprocess.CalledProcessError:
            continue
        except FileNotFoundError as e:
            print(f"\n--- ERROR: Rust compilation failed: {e} ---")
            print("Please ensure Rust/Cargo is installed and in your PATH,")
            print(f"and that the workspace root is correctly set to: {WORKSPACE_ROOT}")
            sys.exit(1)
        candidate_path = WORKSPACE_ROOT / "target" / "release" / candidate
        if candidate_path.is_file():
            EXECUTABLE_NAME = candidate
            EXECUTABLE_PATH = candidate_path
            print(f"--- Compilation successful for '{EXECUTABLE_NAME}'. ---")
            return
    print("\n--- ERROR: No supported Rust executable target could be built. ---")
    sys.exit(1)


def simulate_data(n_samples: int, seed: int):
    """
    Simulates a dataset with heteroscedastic noise.

    Args:
        n_samples: The number of samples to generate.
        seed: The random seed for reproducibility.

    Returns:
        the pure 'signal_prob', and the final 'oracle_prob'.
    """
    np.random.seed(seed)
    pc1 = np.random.normal(0, 1.2, n_samples)
    pc2 = np.random.normal(0, 1.0, n_samples)

    # The true, noise-free relationship between features and the outcome logit
    true_logit = SIMULATION_SIGNAL_STRENGTH * (
        0.9 * pc1
        - 0.5 * pc2
        + 0.35 * pc1 * pc2
        - 0.25 * pc2**2
    )

    # Calculate the "signal" probability (the ideal, noise-free probability)
    signal_probability = 1 / (1 + np.exp(-true_logit))

    # --- HETEROSCEDASTIC NOISE MODEL ---
    # The noise level depends on the value of PC1
    base_noise_std = 2.0
    pc1_noise_factor = 0.75
    dynamic_noise_std = base_noise_std + pc1_noise_factor * np.maximum(0, pc1)
    noise = np.random.normal(0, dynamic_noise_std, n_samples)

    # Final "oracle" probability (signal + instance-specific noise) for generating the outcome
    oracle_probability = 1 / (1 + np.exp(-(true_logit + noise)))
    phenotype = np.random.binomial(1, oracle_probability, n_samples)

    df = pd.DataFrame(
        {
            "variable_one": pc1,
            "variable_two": pc2,
            "phenotype": phenotype,
        }
    )
    # Add both the pure signal and the final oracle probabilities for evaluation
    df['signal_prob'] = signal_probability
    df['oracle_prob'] = oracle_probability

    return df


def run_subprocess(command, cwd):
    """Runs a subprocess and handles common errors cleanly."""
    try:
        print(f"Executing: {' '.join(map(str, command))}\n")
        subprocess.run(command, check=True, text=True, cwd=cwd)
    except KeyboardInterrupt:
        print("\n--- Process interrupted by user (Ctrl+C). Cleaning up. ---")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n--- A command FAILED (Exit Code: {e.returncode}) ---")
        print(f"Failed command: {' '.join(map(str, command))}")
        sys.exit(1)


def tjurs_r2(y_true, y_prob):
    """Calculates Tjur's Coefficient of Discrimination (R-squared)."""
    y_true = pd.Series(y_true)
    mean_prob_cases = y_prob[y_true == 1].mean()
    mean_prob_controls = y_prob[y_true == 0].mean()
    return mean_prob_cases - mean_prob_controls


def nagelkerkes_r2(y_true, y_prob):
    """Calculates Nagelkerke's Pseudo R-squared."""
    y_true = np.asarray(y_true, float)
    y_prob = np.clip(np.asarray(y_prob, float), 1e-15, 1 - 1e-15)
    
    # Log-likelihood of the model with only an intercept (null model)
    p_mean = np.mean(y_true)
    if p_mean == 0.0 or p_mean == 1.0:
        return np.nan  # NaN is more honest than 0.0 for this edge case
    
    log_likelihood_null = np.sum(y_true * np.log(p_mean) + (1 - y_true) * np.log(1 - p_mean))
    
    # Log-likelihood of the full model
    log_likelihood_model = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    n = len(y_true)
    r2_cs = 1 - np.exp((2/n) * (log_likelihood_null - log_likelihood_model))
    max_r2_cs = 1 - np.exp((2/n) * log_likelihood_null)
    
    return r2_cs / max_r2_cs if max_r2_cs > 0 else np.nan


# --- Robust Calibration Metrics ---

# Small constant to prevent numerical issues
_EPS = 1e-15

def _prep_vec(y, p):
    """Prepare data vectors by converting to proper types and handling NaNs."""
    y = pd.Series(y).astype(float)
    p = pd.Series(p).astype(float).clip(_EPS, 1-_EPS)
    m = (~y.isna()) & (~p.isna())
    return y[m].values, p[m].values

def safe_auc(y_true, y_prob):
    """Safely calculate AUC with proper edge case handling."""
    y, p = _prep_vec(y_true, y_prob)
    if len(np.unique(y)) < 2:  # Need both classes for ROC
        return np.nan
    return roc_auc_score(y, p)

def safe_pr_auc(y_true, y_prob):
    """Safely calculate PR-AUC with proper edge case handling."""
    y, p = _prep_vec(y_true, y_prob)
    if (y==1).sum() == 0 or (y==0).sum() == 0:  # Need both classes
        return np.nan
    return average_precision_score(y, p)

def safe_logloss(y_true, y_prob):
    """Safely calculate log loss with proper edge case handling."""
    y, p = _prep_vec(y_true, y_prob)
    if len(y) == 0:
        return np.nan
    from sklearn.metrics import log_loss
    return log_loss(y, p)

def best_threshold_youden(y_true, y_prob):
    """Find the optimal threshold that maximizes Youden's J statistic (sensitivity + specificity - 1).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        float: The threshold that maximizes Youden's J
    """
    y, p = _prep_vec(y_true, y_prob)
    if len(np.unique(y)) < 2:  # Need both classes
        return 0.5  # Default threshold
    
    fpr, tpr, thresholds = roc_curve(y, p)
    best_idx = np.argmax(tpr - fpr)
    return float(thresholds[best_idx])

def _ece_from_edges(y, p, edges):
    """Calculate ECE given bin edges, using bin-mass weighting."""
    # Assign points to bins formed by 'edges' in probability space
    ids = np.digitize(p, edges, right=True) - 1
    ids = np.clip(ids, 0, len(edges)-2)
    N = len(p)
    ece = 0.0
    for b in range(len(edges)-1):
        idx = (ids == b)
        n_b = int(idx.sum())
        if n_b == 0:
            continue
        conf = p[idx].mean()
        acc  = y[idx].mean()
        ece += (n_b / N) * abs(acc - conf)
    return ece

def ece_quantile(y_true, y_prob, n_bins=20):
    """Mass-weighted ECE using equal-frequency (quantile) bins.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins to use
        
    Returns:
        Bin-mass-weighted ECE value
    """
    y, p = _prep_vec(y_true, y_prob)
    if len(y) == 0:
        return np.nan
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(p, qs)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    if len(edges) < 2:
        return abs(y.mean() - p.mean())  # degenerate case
    return _ece_from_edges(y, p, edges)

def ece_randomized_quantile(y_true, y_prob, bin_counts=(10,20,40), repeats=50, min_per_bin=20, rng=None):
    """Averaged, bin-insensitive ECE: quantile bins with random offsets, multi-resolution.
    
    This method creates multiple random binnings in rank space and
    averages the ECE across them, making the metric more stable and
    less sensitive to bin boundaries.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        bin_counts: Tuple of bin counts to try for multi-resolution averaging
        repeats: Number of random offsets per bin count
        min_per_bin: Minimum examples per bin (auto-reduces bin count if needed)
        rng: Random number generator or seed
        
    Returns:
        Dict with 'ece_mean', 'ece_std', and other details
    """
    rng = np.random.default_rng(rng)
    y, p = _prep_vec(y_true, y_prob)
    if len(y) == 0:
        return {'ece_mean': np.nan, 'ece_std': np.nan, 'details': {}}
    eces = []
    for M in bin_counts:
        # Ensure enough examples per bin
        M_eff = min(M, max(1, len(y)//max(1, min_per_bin)))
        if M_eff < 2:
            eces.append(abs(y.mean() - p.mean()))
            continue
        for _ in range(repeats):
            delta = rng.uniform(0, 1.0/M_eff)
            qs = (delta + np.arange(M_eff+1)/M_eff).clip(0, 1)
            edges = np.quantile(p, qs)
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.unique(edges)
            if len(edges) < 2:
                eces.append(abs(y.mean() - p.mean()))
                continue
            eces.append(_ece_from_edges(y, p, edges))
    eces = np.array(eces, dtype=float)
    return {
        'ece_mean': float(np.mean(eces)),
        'ece_std':  float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0,
        'details':  {'bin_counts': list(bin_counts), 'repeats': repeats}
    }

def wilson_ci(k, n, z=1.959963984540054):  # 95% CI
    if n == 0: 
        return (np.nan, np.nan)
    p = k/n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z*np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return center - half, center + half

def ici_isotonic(y_true, y_prob):
    """Compute Integrated Calibration Index with isotonic regression.

    This is a bin-free calibration metric measuring the mean absolute
    difference between predictions and an isotonic fit of outcomes.
    """
    y, p = _prep_vec(y_true, y_prob)
    if len(y) < 2:
        return np.nan
    iso = IsotonicRegression(out_of_bounds='clip')
    # Fit expected outcome as function of predicted probability
    m = iso.fit_transform(p, y)
    return float(np.mean(np.abs(m - p)))


def ece_rq_shared_edges(preds_dict, y_true, bin_counts=(10,20,40), repeats=50, min_per_bin=20, rng=None):
    """Calculate randomized quantile ECE using shared bin edges across all models.
    
    This ensures that all models are evaluated on the exact same bin edges,
    making the ECE values strictly comparable across models.
    
    Args:
        preds_dict: Dictionary of model_name -> predicted_probabilities
        y_true: True binary labels
        bin_counts: Tuple of bin counts to try for multi-resolution averaging
        repeats: Number of random offsets per bin count
        min_per_bin: Minimum examples per bin (auto-reduces bin count if needed)
        rng: Random number generator or seed
        
    Returns:
        Dictionary of model_name -> {'ece_mean': float, 'ece_std': float}
    """
    rng = np.random.default_rng(rng)
    
    # Prep per-model and find the min N across models
    prepared = {name: _prep_vec(y_true, p) for name, p in preds_dict.items()}
    Ns = [len(y) for (y, p) in prepared.values()]
    N_min = min(Ns)
    
    # Pool predictions (use clipped/cleaned predictions)
    pooled = np.concatenate([p for (_, p) in prepared.values()])
    
    results, edge_sets = {}, []
    
    # Pre-generate all edge sets from pooled predictions
    for M in bin_counts:
        M_eff = min(M, max(1, N_min//max(1, min_per_bin)))
        if M_eff < 2:
            edge_sets.append([(None, M_eff)])  # marker for degenerate case
            continue
        sets_M = []
        for _ in range(repeats):
            delta = rng.uniform(0, 1.0/M_eff)
            qs = (delta + np.arange(M_eff+1)/M_eff).clip(0,1)
            edges = np.quantile(pooled, qs)
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.unique(edges)
            sets_M.append((edges, M_eff))
        edge_sets.append(sets_M)

    # Evaluate each model on the same edge sets
    for name, (y_m, p_m) in prepared.items():
        eces = []
        for sets_M in edge_sets:
            for edges, M_eff in sets_M:
                if edges is None:  # degenerate case
                    eces.append(abs(y_m.mean() - p_m.mean()))
                else:
                    eces.append(_ece_from_edges(y_m, p_m, edges))
        eces = np.array(eces, dtype=float)
        results[name] = {
            'ece_mean': float(np.mean(eces)),
            'ece_std': float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0
        }
    return results


def ece_rq_shared_edges_bootstrap_ci(models, y_true, n_boot=500, alpha=0.05, rng=None):
    """Calculate bootstrap confidence intervals for shared-edges randomized quantile ECE.
    
    This measures sampling uncertainty in the shared-edges ECE estimate by bootstrapping
    the data and recalculating shared-edges ECE for each bootstrap sample. It ensures
    that the same bootstrap indices are used for all models for consistency.
    
    Args:
        models: Dictionary of model_name -> predicted_probabilities
        y_true: True binary labels
        n_boot: Number of bootstrap samples
        alpha: Alpha level for confidence interval (e.g., 0.05 for 95% CI)
        rng: Random number generator or seed
        
    Returns:
        Dictionary of model_name -> {'mean': float, 'lo': float, 'hi': float}
    """
    rng = np.random.default_rng(rng)
    # Prepare once
    y = pd.Series(y_true).values
    preds = {k: pd.Series(v).values for k, v in models.items()}
    n = len(y)
    stats = {k: [] for k in models}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # Bootstrap sampling with replacement
        # Slice all arrays with the same bootstrap indices
        y_b = y[idx]
        preds_b = {k: v[idx] for k, v in preds.items()}
        # Recompute shared-edges randomized ECE on the bootstrap sample
        shared = ece_rq_shared_edges(preds_b, y_b, bin_counts=(10,20,40), 
                                    repeats=30, min_per_bin=20, 
                                    rng=rng.integers(1<<32))
        for k in models:
            stats[k].append(shared[k]['ece_mean'])

    out = {}
    for k, vals in stats.items():
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]  # Remove NaNs
        if len(arr) == 0:
            out[k] = {'mean': np.nan, 'lo': np.nan, 'hi': np.nan}
            continue
        lo, hi = np.quantile(arr, [alpha/2, 1-alpha/2])
        out[k] = {'mean': float(np.mean(arr)), 'lo': float(lo), 'hi': float(hi)}
    return out


def brier_decomposition(y_true, y_prob, n_bins=20):
    """Decompose Brier score into reliability, resolution, uncertainty.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for grouping predictions
        
    Returns:
        Tuple of (reliability, resolution, uncertainty)
    """
    y, p = _prep_vec(y_true, y_prob)
    
    # Use quantile binning for more balanced bins
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(p, qs)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    
    ids = np.digitize(p, edges, right=True) - 1
    ids = np.clip(ids, 0, len(edges)-2)
    
    # Overall mean outcome (prevalence)
    p_bar = y.mean()
    reliability = 0.0
    resolution = 0.0
    
    for b in range(len(edges)-1):
        idx = (ids == b)
        n_b = idx.sum()
        if n_b == 0:
            continue
        
        # Mean prediction and outcome in this bin
        conf = p[idx].mean()
        acc = y[idx].mean()
        
        # Bin weight
        w = n_b/len(y)
        
        # Accumulate components
        reliability += w * (conf - acc)**2     # Squared calibration error in bin
        resolution += w * (acc - p_bar)**2     # How far bin's outcomes deviate from overall mean
    
    # Inherent uncertainty in the data
    uncertainty = p_bar * (1 - p_bar)
    
    return reliability, resolution, uncertainty


def generate_performance_report(df_results):
    """Calculates and prints a side-by-side comparison of model metrics."""
    y_true = df_results['phenotype']

    models = {
        "GAM (gnomon)": df_results['prediction'],
        "Baseline (Logistic)": df_results['baseline_prediction'],
        "Signal Model (Noise-Free)": df_results['signal_prob'],
        "Oracle (Instance Best)": df_results['oracle_prob']
    }

    print("\n" + "="*60)
    print("      Model Performance Comparison on Test Set")
    print("="*60)

    print("\n[Discrimination: AUC (Area Under ROC Curve) & PR-AUC]")
    print("Higher is better. ROC-AUC for overall discrimination, PR-AUC for imbalanced data.")
    for name, y_prob in models.items():
        auc = safe_auc(y_true, y_prob)
        pr_auc_val = safe_pr_auc(y_true, y_prob)
        print(f"  - {name:<28}: AUC={auc:.4f} | PR-AUC={pr_auc_val:.4f}")

    print("Lower is better. Brier for squared error, Log Loss for likelihood-based assessment.")
    for name, y_prob in models.items():
        y, p = _prep_vec(y_true, y_prob)  # Handle NaNs and invalid values
        brier = brier_score_loss(y, p) if len(y) else np.nan
        logloss = safe_logloss(y_true, y_prob)
        print(f"  - {name:<28}: Brier={brier:.4f} | LogLoss={logloss:.4f}")

    print("\n[Fit: Nagelkerke's R-squared]")
    for name, y_prob in models.items():
        r2_n = nagelkerkes_r2(y_true, y_prob)
        print(f"  - {name:<28}: {r2_n:.4f}")

    print("\n[Separation: Tjur's R-squared]")
    print("Higher is better (0-1). The mean difference in prediction between classes.")
    for name, y_prob in models.items():
        r2_t = tjurs_r2(y_true, y_prob)
        print(f"  - {name:<28}: {r2_t:.4f}")

    print("\n[Calibration: Expected Calibration Error (ECE)]")
    print("Lower is better. Mass-weighted; shared-edges for fair comparison.")
    
    # Calculate shared-edges ECE across all models (strictly comparable)
    shared_ece = ece_rq_shared_edges(models, y_true, 
                                   bin_counts=(10, 20, 40),
                                   repeats=50, min_per_bin=20, rng=123)
    
    # Calculate bootstrap CIs for shared-edges ECE
    shared_boot = ece_rq_shared_edges_bootstrap_ci(models, y_true, 
                                              n_boot=200, alpha=0.05, rng=123)
    
    for name, y_prob in models.items():
        # Calculate individual metrics
        ece_q = ece_quantile(y_true, y_prob, n_bins=20)
        ici = ici_isotonic(y_true, y_prob)
        shared_result = shared_ece[name]
        
        # Print metrics with bootstrap CI for consistency
        ci = shared_boot[name] if name in shared_boot else {'lo': np.nan, 'hi': np.nan}
        print(f"  - {name:<28}: ECE_q20={ece_q:.4f} | ECE_shared={shared_result['ece_mean']:.4f} ±{shared_result['ece_std']:.4f} | ICI={ici:.4f}")
        print(f"      Bootstrap 95% CI for shared-edges ECE: [{ci['lo']:.4f}, {ci['hi']:.4f}]")
            
    # Add Brier decomposition
    print("reliability ↓, resolution ↑, uncertainty (data)")
    for name, y_prob in models.items():
        y_clean, p_clean = _prep_vec(y_true, y_prob)  # Handle NaNs and invalid values
        brier = brier_score_loss(y_clean, p_clean) if len(y_clean) else np.nan
        rel, res, unc = brier_decomposition(y_clean, p_clean, n_bins=20)
        err = abs(brier - (rel - res + unc))
        print(f"  - {name:<28}: rel={rel:.4f}, res={res:.4f}, unc={unc:.4f} (brier={brier:.4f})")
        if err > 1e-4:  # Check if identity doesn't hold within tolerance
            print(f"    (Warning: Brier decomposition mismatch Δ={err:.2e})")

    print("\n[Confusion Matrices (with tuned thresholds)]")
    for name, y_prob in models.items():
        # Clean data and handle NaNs
        y_clean, p_clean = _prep_vec(y_true, y_prob)
        
        # Find optimal threshold using Youden's J statistic
        threshold = best_threshold_youden(y_clean, p_clean)
        
        # Apply the tuned threshold for classification
        y_pred_class = (p_clean > threshold).astype(int)
        cm = confusion_matrix(y_clean, y_pred_class)
        cm_proportions = cm / cm.sum()
        
        # Calculate sensitivity and specificity
        if len(np.unique(y_clean)) < 2:  # Edge case: only one class
            sensitivity = np.nan
            specificity = np.nan
        else:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        
        print(f"\n  --- {name} ---")
        print(f"  (Optimal threshold = {threshold:.4f}, Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f})")
        print("             Predicted 0   Predicted 1")
        print(f"    True 0     {cm_proportions[0,0]:<12.3f}  {cm_proportions[0,1]:<12.3f}")
        print(f"    True 1     {cm_proportions[1,0]:<12.3f}  {cm_proportions[1,1]:<12.3f}")

    print("\n" + "="*60)
    return

def main():
    """Main function to run the end-to-end simulation and plotting."""
    # Set default figure format and output path for saved plots
    output_dir = Path(os.environ.get('PLOT_OUTPUT_DIR', './'))
    output_dir.mkdir(exist_ok=True)
        
    try:
        print(f"Project Root Detected: {PROJECT_ROOT}")
        build_rust_project()
        print(f"Using executable: {EXECUTABLE_PATH}\n")

        # 1. Simulate and prepare training data
        print(f"--- Simulating {N_SAMPLES_TRAIN} samples for training ---")
        train_df = simulate_data(N_SAMPLES_TRAIN, seed=42)
        train_df.to_csv(TRAIN_DATA_PATH, sep="\t", index=False)
        print(f"Saved training data to '{TRAIN_DATA_PATH}'")

        # 2. Train baseline Logistic Regression model
        print("\n--- Training Baseline Logistic Regression Model ---")
        baseline_model = LogisticRegression(solver='liblinear')
        baseline_model.fit(train_df[['variable_one', 'variable_two']], train_df['phenotype'])
        print("Baseline model trained.")

        # 3. Run the gnomon/gam 'train' command
        print("\n--- Running 'train' command for GAM model ---")
        train_command = [
            str(EXECUTABLE_PATH), "train", "--num-pcs", str(NUM_PCS),
            str(TRAIN_DATA_PATH)
        ]
        run_subprocess(train_command, cwd=PROJECT_ROOT)
        print(f"\nGAM model training complete. Model saved to '{MODEL_PATH}'")

        # 4. Simulate and prepare test data
        print(f"\n--- Simulating {N_SAMPLES_TEST} samples for inference ---")
        test_df_full = simulate_data(N_SAMPLES_TEST, seed=101)
        test_df_full.to_csv(TEST_DATA_PATH, sep="\t", index=False)
        print(f"Saved test data to '{TEST_DATA_PATH}'")

        # 5. Get predictions from baseline model
        print("\n--- Generating predictions with Baseline Model ---")
        baseline_probs = baseline_model.predict_proba(
            test_df_full[['variable_one', 'variable_two']]
        )[:, 1]
        
        # 6. Run 'infer' for GAM on test data and load results
        print("\n--- Running 'infer' command for GAM model on test data ---")
        infer_command = [str(EXECUTABLE_PATH), "infer", "--model", str(MODEL_PATH), str(TEST_DATA_PATH)]
        run_subprocess(infer_command, cwd=PROJECT_ROOT)
        print(f"Inference complete. Predictions saved to '{PREDICTIONS_PATH}'")
        gam_test_predictions = pd.read_csv(PREDICTIONS_PATH, sep="\t")

        # 7. Combine all results for reporting
        print("\n--- Analyzing Results ---")
        results_df = test_df_full.copy()
        if "prediction" in gam_test_predictions.columns:
            results_df["prediction"] = gam_test_predictions["prediction"].values
        else:
            first_numeric = gam_test_predictions.select_dtypes(include=[np.number]).columns
            if len(first_numeric) == 0:
                raise RuntimeError("No numeric prediction column found in predictions.tsv")
            results_df["prediction"] = gam_test_predictions[first_numeric[0]].values
        results_df['baseline_prediction'] = baseline_probs
        generate_performance_report(results_df)
        print("\nSkipping surface plot generation because this script is in reduced safe mode.")
        return

    finally:
        # Clean up temporary files created during the run
        print("\n--- Cleaning up temporary files ---")
        files_to_clean = [TRAIN_DATA_PATH, TEST_DATA_PATH, GRID_DATA_PATH, PREDICTIONS_PATH]
        for f in files_to_clean:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Removed {f.name}")
                except OSError as e:
                    print(f"Error removing file {f.name}: {e}")

if __name__ == "__main__":
    main()
