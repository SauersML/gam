import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tomli

# --- 1. Define Paths and Parameters ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
R_MODEL_PATH = SCRIPT_DIR / 'gam_model_fit.rds'
RUST_MODEL_CONFIG_PATH = PROJECT_ROOT / 'model.toml'
N_POINTS_PLOT = 400

def print_array_summary(name, arr):
    """Helper to print detailed diagnostics for a numpy array."""
    if arr.ndim == 1:
        print(f"  [DIAGNOSTIC] {name} | Shape: {arr.shape} | Min: {np.min(arr):.4f} | Max: {np.max(arr):.4f} | Mean: {np.mean(arr):.4f} | Std: {np.std(arr):.4f}")
        print(f"    -> First 5 elements: {arr[:5]}")
    else:
        print(f"  [DIAGNOSTIC] {name} | Shape: {arr.shape} | Min: {np.min(arr):.4f} | Max: {np.max(arr):.4f} | Mean: {np.mean(arr):.4f}")
        col_stds = np.std(arr, axis=0)
        print(f"    -> Stds of first 5 columns: {col_stds[:5]}")
        print(f"    -> First 2x5 slice:\n{arr[:2, :5]}")

def evaluate_bspline_basis(x, knots, degree):
    """
    A Python implementation of the Cox-de Boor algorithm that mirrors the Rust code's
    logic, including state persistence for intermediate calculations and boundary handling.
    """
    num_knots = len(knots)
    num_bases = num_knots - degree - 1
    
    basis_matrix = np.zeros((len(x), num_bases))
    x_clamped = np.clip(x, knots[degree], knots[num_bases])

    for i, val in enumerate(x_clamped):
        if val >= knots[num_bases]:
            mu = num_bases - 1
        else:
            mu = np.searchsorted(knots, val, side='right') - 1
            mu = max(degree, mu)

        b = np.zeros(degree + 1)
        b[0] = 1.0
        
        left = np.zeros(degree + 1)
        right = np.zeros(degree + 1)
        
        for d in range(1, degree + 1):
            left[d] = val - knots[mu + 1 - d]
            right[d] = knots[mu + d] - val

            saved = 0.0
            for r in range(d):
                den = right[r + 1] + left[d - r]
                temp = 0.0
                if abs(den) > 1e-12:
                    temp = b[r] / den
                
                b[r] = saved + right[r + 1] * temp
                saved = left[d - r] * temp
            b[d] = saved
        
             basis_matrix[i, 0 : len(b_valid)] = b_valid
        else:

    return basis_matrix

def get_mgcv_basis_data():
    """
    Runs an R script to extract the mgcv model's basis matrix and coefficients.
    """
    print("\n" + "="*80)
    print("--- STAGE 1: Extracting CONSTRAINED basis from R/mgcv model ---")
    print("="*80)
    
    x_axis_file, basis_file, coeffs_file = [SCRIPT_DIR / f for f in ["t_x.csv", "t_b.csv", "t_c.csv"]]
    
    r_script = f"""
    suppressPackageStartupMessages(library(mgcv))
    model <- readRDS('{R_MODEL_PATH}')
    var_range <- range(model$model$variable_one)
    x_seq <- seq(var_range[1], var_range[2], length.out = {N_POINTS_PLOT})
    newdata <- data.frame(variable_one = x_seq, variable_two = 0)
    lp_matrix <- predict(model, newdata = newdata, type = "lpmatrix")
    smooth_info <- model$smooth[[1]]
    write.csv(data.frame(x=x_seq), '{x_axis_file.name}', row.names=FALSE)
    write.csv(constrained_basis_functions, '{basis_file.name}', row.names=FALSE)
    write.csv(data.frame(coeffs=basis_coeffs), '{coeffs_file.name}', row.names=FALSE)
    cat("R: Extracted main effect basis for '", smooth_info$label, "'.\\n", sep="")
    """
    
    try:
        result = subprocess.run(["Rscript", "-e", r_script], check=True, text=True, cwd=SCRIPT_DIR, capture_output=True)
        print(f"  [INFO] R stdout: {result.stdout.strip()}")
        if result.stderr: print(f"  [INFO] R stderr: {result.stderr.strip()}")

        x_axis = pd.read_csv(x_axis_file)['x'].values
        basis_matrix = pd.read_csv(basis_file).values
        coeffs = pd.read_csv(coeffs_file)['coeffs'].values
        
        print(f"  [PRINT] mgcv: Loaded x-axis vector.")
        print_array_summary("mgcv x_axis", x_axis)
        print(f"  [PRINT] mgcv: Loaded constrained basis matrix.")
        print_array_summary("mgcv basis_matrix", basis_matrix)
        print(f"  [PRINT] mgcv: Loaded coefficients.")
        print_array_summary("mgcv coeffs", coeffs)
        
        return {"x_axis": x_axis, "basis_matrix": basis_matrix, "coeffs": coeffs}
    except subprocess.CalledProcessError as e:
        print(f"\n--- FATAL ERROR: R script execution failed. ---\n{e.stderr}")
        sys.exit(1)
    finally:
        for f in [x_axis_file, basis_file, coeffs_file]:
            if f.exists(): f.unlink()

def get_gnomon_basis_data():
    """
    Reconstructs the gnomon constrained basis from the model.toml file.
    """
    print("\n" + "="*80)
    print("--- STAGE 2: Reconstructing CONSTRAINED basis from Rust/gnomon model ---")
    print("="*80)
    
    with open(RUST_MODEL_CONFIG_PATH, "rb") as f:
        toml_data = tomli.load(f)

    print_array_summary("Knot Vector", knots)
    print(f"  [PRINT] gnomon: Loaded spline degree {degree}.")
    print_array_summary("Coefficients", coeffs)
    
    num_raw_bases = len(knots) - degree - 1
    print(f"  [INFO] gnomon: Derived k={num_raw_bases} total raw B-spline bases from knot vector.")

    z_dims, z_data = constraint_info['dim'], constraint_info['data']
    z_transform = np.array(z_data).reshape(z_dims)
    print_array_summary("z_transform", z_transform)

    x_axis = np.linspace(x_range[0], x_range[1], N_POINTS_PLOT)
    raw_basis_matrix = evaluate_bspline_basis(x_axis, knots, degree)
    print(f"  [PRINT] gnomon: Reconstructed FULL raw basis matrix.")
    print_array_summary("raw_basis_matrix", raw_basis_matrix)

    raw_main_basis_functions = raw_basis_matrix[:, 1:]
    print(f"  [INFO] gnomon: Sliced raw basis to get the non-constant bases for constraining.")
    print_array_summary("raw_main_basis_functions", raw_main_basis_functions)
    
    if raw_main_basis_functions.shape[1] != z_transform.shape[0]:
        print(f"\n--- FATAL ERROR: Dimension mismatch for constraint! ---")
        print(f"Raw main basis columns: {raw_main_basis_functions.shape[1]}, Z-transform rows: {z_transform.shape[0]}")
        sys.exit(1)

    constrained_basis_matrix = raw_main_basis_functions @ z_transform
    print(f"  [PRINT] gnomon: Created FINAL constrained basis matrix.")
    print_array_summary("constrained_basis_matrix", constrained_basis_matrix)
    
    if constrained_basis_matrix.shape[1] != len(coeffs):
        print(f"\n--- FATAL ERROR: Final dimension mismatch! ---")
        print(f"Final basis columns: {constrained_basis_matrix.shape[1]}, Coefficients length: {len(coeffs)}")
        sys.exit(1)
        
    print("  [INFO] All dimension checks passed successfully.")
    return {"x_axis": x_axis, "basis_matrix": constrained_basis_matrix, "coeffs": coeffs}

def create_comparison_plot(mgcv_data, gnomon_data):
    """
    Creates a 3x2 plot comparing all components of the main smooth term.
    The mgcv components are centered for visual comparability.
    """
    print("\n" + "="*80)
    print("--- STAGE 3: Generating the SINGLE 3x2 Comparison Plot ---")
    print("="*80)

    # --- gnomon calculations (straightforward) ---
    gnomon_basis = gnomon_data['basis_matrix']
    gnomon_coeffs = gnomon_data['coeffs']
    gnomon_weighted = gnomon_basis * gnomon_coeffs
    gnomon_final_curve = gnomon_weighted.sum(axis=1)

    # --- mgcv calculations (requires centering for visualization) ---
    mgcv_basis = mgcv_data['basis_matrix']
    mgcv_coeffs = mgcv_data['coeffs']
    
    # 1. Calculate the original, uncentered weighted basis and final curve
    mgcv_weighted_uncentered = mgcv_basis * mgcv_coeffs
    mgcv_final_curve_uncentered = mgcv_weighted_uncentered.sum(axis=1)
    
    # 2. Calculate the mean of the final curve. This is the offset we need to remove.
    mean_offset = np.mean(mgcv_final_curve_uncentered)
    
    # 3. Create the final, centered curve for plotting. This represents the true shape of the smooth.
    mgcv_final_curve_centered = mgcv_final_curve_uncentered - mean_offset

    # 4. Create a centered version of the weighted basis functions FOR PLOTTING ONLY.
    # We subtract the average contribution of each weighted basis function.
    mgcv_weighted_centered = mgcv_weighted_uncentered - np.mean(mgcv_weighted_uncentered, axis=0)

    # --- Print Diagnostics ---
    print(f"  [PRINT] mgcv: Calculated components.")
    print_array_summary("mgcv_weighted_centered", mgcv_weighted_centered)
    print_array_summary("mgcv_final_curve_centered", mgcv_final_curve_centered)
    print(f"  [PRINT] gnomon: Calculated components.")
    print_array_summary("gnomon_weighted", gnomon_weighted)
    print_array_summary("gnomon_final_curve", gnomon_final_curve)

    # --- Plotting ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), sharex=True, constrained_layout=True)
    fig.suptitle("Internal Component Comparison: mgcv vs gnomon", fontsize=20)

    axes[0, 0].set_title("mgcv Model (Computational Basis)", fontsize=16)
    axes[0, 1].set_title("gnomon Model (Interpretable Basis)", fontsize=16)

    # Row 1: Constrained Basis Functions
    axes[0, 0].plot(mgcv_data['x_axis'], mgcv_basis, alpha=0.7)
    axes[0, 0].set_ylabel("Basis Value", fontsize=12)
    axes[0, 1].plot(gnomon_data['x_axis'], gnomon_basis, alpha=0.7)

    # Row 2: Weighted Basis Functions (using the centered mgcv version)
    axes[1, 0].plot(mgcv_data['x_axis'], mgcv_weighted_centered, alpha=0.7)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_ylabel("Centered Weighted Basis Value", fontsize=12)
    axes[1, 1].plot(gnomon_data['x_axis'], gnomon_weighted, alpha=0.7)
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)

    # Row 3: Final Smooth Curve and Overlay (using the centered mgcv version)
    axes[2, 0].plot(mgcv_data['x_axis'], mgcv_final_curve_centered, color='crimson', linewidth=3)
    axes[2, 0].set_xlabel(mgcv_data.get('var_name', 'variable_one'), fontsize=12)
    axes[2, 0].set_ylabel("Centered Smooth Contribution", fontsize=12)
    axes[2, 0].grid(True, linestyle=':', alpha=0.7)
    
    axes[2, 1].plot(mgcv_data['x_axis'], mgcv_final_curve_centered, label='mgcv (Centered)', color='blue', linewidth=6, alpha=0.6)
    axes[2, 1].plot(gnomon_data['x_axis'], gnomon_final_curve, label='gnomon', color='red', linewidth=2.5, linestyle='--')
    axes[2, 1].legend(title="Model")
    axes[2, 1].set_title("Verification: Overlay of Final Curves", fontsize=14)
    axes[2, 1].grid(True, linestyle=':', alpha=0.7)

    plt.show()

def main():
    """Main script to generate and display the basis function plots."""
    for f in [R_MODEL_PATH, RUST_MODEL_CONFIG_PATH]:
        if not f.is_file():
            print(f"--- FATAL ERROR: Required file not found: '{f}' ---"); sys.exit(1)

    mgcv_data = get_mgcv_basis_data()
    gnomon_data = get_gnomon_basis_data()
    create_comparison_plot(mgcv_data, gnomon_data)
    
    print("\n--- Script finished successfully. ---")

if __name__ == "__main__":
    main()
