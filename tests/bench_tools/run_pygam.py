from pathlib import Path
import itertools

import joblib
import numpy as np
import pandas as pd
from pygam import LogisticGAM, s, te
from tqdm import tqdm


def main():
    script_dir = Path(__file__).resolve().parent
    input_csv_file = script_dir / "synthetic_classification_data.csv"
    output_joblib_file = script_dir / "gam_model_fit.joblib"

    if not input_csv_file.exists():
        raise FileNotFoundError(
            f"Required input CSV not found: {input_csv_file}. "
            "Generate it before running this script."
        )

    print(f"Loading data from '{input_csv_file}'...\n")
    data = pd.read_csv(input_csv_file)

    feature_cols = ["variable_one", "variable_two"]
    target_col = "outcome"
    X = data[feature_cols].values
    y = data[target_col].values
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}\n")

    gam_formula = s(0, n_splines=21) + s(1, n_splines=21) + te(0, 1, n_splines=[22, 22])
    print("Starting manual grid search to tune penalized terms...")
    lam_grid = np.logspace(-3, 3, 11)
    best_gam = None
    best_lams = None

    for lams_for_s1, lams_for_te in tqdm(list(itertools.product(lam_grid, lam_grid)), desc="Fitting models"):
        current_lams = [[0], [lams_for_s1], [lams_for_te, lams_for_te]]
        try:
            gam = LogisticGAM(gam_formula, lam=current_lams).fit(X, y)
            current_aic = gam.statistics_.get("AIC", np.inf)
            best_aic = best_gam.statistics_.get("AIC", np.inf) if best_gam is not None else np.inf
            if current_aic < best_aic:
                best_gam = gam
                best_lams = current_lams
        except Exception:
            continue

    if best_gam is None:
        raise RuntimeError("Model fitting failed for all lambda combinations.")

    print("\nManual grid search finished.")
    print(f"Found best lambdas: {best_lams}")
    print(f"\nSaving fitted model object to '{output_joblib_file}'...")
    joblib.dump(best_gam, output_joblib_file)
    print("Script finished successfully.\n")


if __name__ == "__main__":
    main()
