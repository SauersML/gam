import pandas as pd
from pygam import LogisticGAM, s, te
import joblib
import numpy as np
import itertools
from tqdm import tqdm

def main():
    input_csv_file = "synthetic_classification_data.csv"
    output_joblib_file = "gam_model_fit.joblib"

    print(f"Loading data from '{input_csv_file}'...\n")
    data = pd.read_csv(input_csv_file)

    feature_cols = ['variable_one', 'variable_two']
    target_col = 'outcome'
    
    X = data[feature_cols].values
    y = data[target_col].values
    
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}\n")

    gam_formula = s(0, n_splines=21) + \
                  s(1, n_splines=21) + \
                  te(0, 1, n_splines=[22, 22])

    print("Starting manual grid search to tune penalized terms...")

    lam_grid = np.logspace(-3, 3, 11)

    best_gam = None
    best_lams = None

    search_space = list(itertools.product(lam_grid, lam_grid))
    
    for lams_for_s1, lams_for_te in tqdm(search_space, desc="Fitting models"):
        current_lams = [[0], [lams_for_s1], [lams_for_te, lams_for_te]]

        try:
            gam = LogisticGAM(gam_formula, lam=current_lams).fit(X, y)
            

                best_gam = gam
                best_lams = current_lams

        except Exception as e:
            continue
            
    if best_gam is None:
        print("\nERROR: Model fitting failed for all lambda combinations. The model is too unstable.")
        return

    print("\nManual grid search finished.")
    print(f"Found best lambdas: {best_lams}")

    print("\n--- Best GAM Model Summary ---\n")
    best_gam.summary()

    print(f"\nSaving best fitted GAM object to '{output_joblib_file}'...")
    joblib.dump(best_gam, output_joblib_file)
    print("Script finished successfully.\n")

if __name__ == '__main__':
    main()