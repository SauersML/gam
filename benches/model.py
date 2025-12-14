import re
import itertools
import io
import csv
from typing import List, Dict, Any, Set, Tuple
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

def parse_and_validate_benchmark_logs(log_content: str) -> str:
    # === 1. Source of Truth Definition ===
    EXPECTED_N_VALUES = [1, 100, 1000, 5000, 10000, 40000]
    EXPECTED_K_VALUES = [1, 5, 50, 100]
    EXPECTED_SUBSET_VALUES = [1, 5, 50, 100]
    EXPECTED_FREQ_VALUES = [0.00001, 0.001, 0.02, 0.4]
    EXPECTED_PATH_VALUES = ['No-Pivot', 'Pivot']
    
    param_columns = ['N (Cohort)', 'K (Scores)', 'Subset', 'Freq', 'Path']

    expected_combinations: Set[Tuple] = set(itertools.product(
        EXPECTED_N_VALUES,
        EXPECTED_K_VALUES,
        EXPECTED_SUBSET_VALUES,
        EXPECTED_FREQ_VALUES,
        EXPECTED_PATH_VALUES
    ))
    expected_count = len(expected_combinations)

    # === 2. Parsing Logic ===
    param_regex = re.compile(
        r"Benchmarking Path Crossover \(Multi-Dimensional\)/"
        r"((?:No-)?Pivot)__N=(\d+)_K=(\d+)_Subset=(\d+)%_Freq=[\d.]+/([\d.]+)"
    )
    time_regex = re.compile(r"time:\s+\[.*? ([\d.]+) (ns|µs|ms) .*?\]")

    parsed_records: List[Dict[str, Any]] = []
    current_params: Dict[str, Any] = {}

    def normalize_to_ns(value: float, unit: str) -> float:
        if unit == 'µs': return value * 1_000
        if unit == 'ms': return value * 1_000_000
        return value

    for line in log_content.splitlines():
        param_match = param_regex.search(line)
        if param_match:
            groups = param_match.groups()
            current_params = {
                'Path': groups[0],
                'N (Cohort)': int(groups[1]),
                'K (Scores)': int(groups[2]),
                'Subset': int(groups[3]),
                'Freq': float(groups[4]),
            }
            continue

        time_match = time_regex.search(line)
        if time_match and current_params:
            time_value, time_unit = time_match.groups()
            record = current_params.copy()
            record['Time (Median)'] = normalize_to_ns(float(time_value), time_unit)
            parsed_records.append(record)
            current_params = {}

    # === 3. Enhanced Validation and Imputation Phase ===
    validation_summary = []
    error_details = []
    is_valid = True

    # --- Initial Checks ---
    initial_parsed_count = len(parsed_records)
    parsed_combinations_set = set(tuple(rec[col] for col in param_columns) for rec in parsed_records)
    
    if len(parsed_combinations_set) != initial_parsed_count:
        is_valid = False
        duplicates_count = initial_parsed_count - len(parsed_combinations_set)
        validation_summary.append(f"❌ FAIL: Found {duplicates_count} duplicate benchmark record(s).")
        error_details.append("Duplicate records were detected, which may indicate a parsing error.")
    else:
        validation_summary.append("✅ PASS: No duplicate benchmark records found.")

    # --- Identify and Handle Missing Data ---
    missing_benchmarks = expected_combinations - parsed_combinations_set
    
    if not missing_benchmarks:
        validation_summary.append("✅ PASS: All expected benchmarks are present.")
    else:
        unexpectedly_missing = []
        imputed_count = 0
        for b in missing_benchmarks:
            # Unpack the tuple for clarity
            n_val, _, subset_val, _, path_val = b
            
            # This is the specific condition identified as a zero-workload scenario
            if n_val == 1 and subset_val in [1, 5] and path_val == 'Pivot':
                # Impute the missing record with a time of 0.0
                imputed_record = dict(zip(param_columns, b))
                imputed_record['Time (Median)'] = 0.0
                parsed_records.append(imputed_record)
                imputed_count += 1
            else:
                unexpectedly_missing.append(b)
        
        validation_summary.append(f"INFO: Found {len(missing_benchmarks)} total missing records. Imputed {imputed_count} as known zero-workload scenarios.")

        if unexpectedly_missing:
            is_valid = False
            validation_summary.append(f"❌ FAIL: Found {len(unexpectedly_missing)} UNEXPECTED missing benchmark(s).")
            error_details.append(f"--- Analysis of {len(unexpectedly_missing)} Unexpectedly Missing Benchmarks ---")
            missing_df = pd.DataFrame(unexpectedly_missing, columns=param_columns)
            for col in param_columns:
                error_details.append(f"Value counts for '{col}' in unexpected missing data:\n{missing_df[col].value_counts().to_string()}\n")
        else:
            validation_summary.append("✅ PASS: All missing records were expected and have been imputed.")

    # --- Final Count Check Post-Imputation ---
    final_parsed_count = len(parsed_records)
    if final_parsed_count == expected_count:
        validation_summary.append(f"✅ PASS: Final record count matches expected after imputation. (Total: {final_parsed_count})")
    else:
        is_valid = False
        validation_summary.append(f"❌ FAIL: Final record count is incorrect even after imputation. (Found: {final_parsed_count}, Expected: {expected_count})")

    # --- Raise Error if Any Validation Failed ---
    if not is_valid:
        report = "VALIDATION FAILED\n\n" + \
                 "--- SUMMARY ---\n" + "\n".join(validation_summary) + \
                 "\n\n--- ERROR DETAILS ---\n" + "\n\n".join(error_details)
        raise ValueError(report)
    
    print("--- VALIDATION SUCCEEDED ---\n" + "\n".join(validation_summary))
    
    # === 4. Output Generation ===
    output = io.StringIO()
    header = ['N (Cohort)', 'K (Scores)', 'Subset', 'Freq', 'Path', 'Time (Median)']
    writer = csv.DictWriter(output, fieldnames=header)
    writer.writeheader()
    writer.writerows(parsed_records)
    
    return output.getvalue()

def build_and_evaluate_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    log_feature_map: Dict[str, str],
    time_pivot: pd.Series,
    time_nopivot: pd.Series,
    tree_name: str
) -> DecisionTreeClassifier:
    print(f"\n{'='*25} BUILDING {tree_name.upper()} {'='*25}")

    # --- Helper Function to Calculate Metrics ---
    def _calculate_metrics(model, X_val, y_val, weight_val, t_pivot_val, t_nopivot_val):
        y_pred = model.predict(X_val)

        # Metric 1: Unweighted Accuracy
        unweighted_acc = accuracy_score(y_val, y_pred)

        # Metric 2: Weighted Accuracy
        weighted_acc = accuracy_score(y_val, y_pred, sample_weight=weight_val)

        # Metric 3: Oracle Performance Ratio (OPR)
        # Baseline is always choosing the 'No-Pivot' path
        T_baseline = t_nopivot_val
        T_oracle = np.minimum(t_pivot_val, t_nopivot_val)
        
        # Determine the time taken if we follow the model's advice
        pivot_class_label = model.classes_[1] # Assumes positive class is 'Pivot'
        T_model = np.where(y_pred == pivot_class_label, t_pivot_val, t_nopivot_val)

        savings_possible = (T_baseline - T_oracle).sum()
        savings_by_model = (T_baseline - T_model).sum()

        opr = savings_by_model / savings_possible if savings_possible > 0 else 1.0

        return {
            "OPR": opr,
            "Weighted Accuracy": weighted_acc,
            "Unweighted Accuracy": unweighted_acc,
        }

    # --- Common Setup ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    feature_names = X.columns.tolist()
    IMPROVEMENT_THRESHOLD = 0.003
    
    # --- Tuning Procedure for Max Depth ---
    print("\n--- Tuning Max Depth (5-Fold CV) ---")
    best_depth = 1
    # Initialize with values that will be beaten by the first iteration
    best_metrics_depth = {k: -1.0 for k in ["OPR", "Weighted Accuracy", "Unweighted Accuracy"]}

    for depth in range(1, 20):  # A reasonable upper limit for depth
        fold_metrics = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sw_train, sw_val = sample_weight.iloc[train_idx], sample_weight.iloc[val_idx]
            tp_val, tnp_val = time_pivot.iloc[val_idx], time_nopivot.iloc[val_idx]

            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            clf.fit(X_train, y_train, sample_weight=sw_train)
            
            metrics = _calculate_metrics(clf, X_val, y_val, sw_val, tp_val, tnp_val)
            fold_metrics.append(metrics)

        mean_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        print(f"Depth {depth:2d}: OPR={mean_metrics['OPR']:.4f}, "
              f"WeightedAcc={mean_metrics['Weighted Accuracy']:.4f}, "
              f"UnweightedAcc={mean_metrics['Unweighted Accuracy']:.4f}")

        # Check for substantial improvement across any metric
        gains = [
            (mean_metrics[k] - best_metrics_depth[k]) / abs(best_metrics_depth[k]) if best_metrics_depth[k] != 0 else float('inf')
            for k in best_metrics_depth
        ]
        
        if any(g > IMPROVEMENT_THRESHOLD for g in gains):
            best_depth = depth
            best_metrics_depth = mean_metrics
        else:
            print(f"-> No substantial improvement over {IMPROVEMENT_THRESHOLD*100:.1f}%. Stopping at max_depth={best_depth}.")
            break
    
    print(f"\n>>> Optimal Max Depth found: {best_depth}")

    # --- Tuning Procedure for Max Leaf Nodes ---
    print("\n--- Tuning Max Leaf Nodes (5-Fold CV) ---")
    best_leaf_nodes = None
    best_metrics_leaves = best_metrics_depth

    for leaves in range(2, 50): # A reasonable upper limit for leaves
        fold_metrics = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sw_train, sw_val = sample_weight.iloc[train_idx], sample_weight.iloc[val_idx]
            tp_val, tnp_val = time_pivot.iloc[val_idx], time_nopivot.iloc[val_idx]

            clf = DecisionTreeClassifier(max_depth=best_depth, max_leaf_nodes=leaves, random_state=42)
            clf.fit(X_train, y_train, sample_weight=sw_train)
            
            metrics = _calculate_metrics(clf, X_val, y_val, sw_val, tp_val, tnp_val)
            fold_metrics.append(metrics)
            
        mean_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        print(f"Leaves {leaves:2d}: OPR={mean_metrics['OPR']:.4f}, "
              f"WeightedAcc={mean_metrics['Weighted Accuracy']:.4f}, "
              f"UnweightedAcc={mean_metrics['Unweighted Accuracy']:.4f}")

        gains = [
            (mean_metrics[k] - best_metrics_leaves[k]) / abs(best_metrics_leaves[k]) if best_metrics_leaves[k] != 0 else float('inf')
            for k in best_metrics_leaves
        ]
        
        if any(g > IMPROVEMENT_THRESHOLD for g in gains):
            best_leaf_nodes = leaves
            best_metrics_leaves = mean_metrics
        else:
            print(f"-> No substantial improvement. Stopping at max_leaf_nodes={best_leaf_nodes}.")
            break

    print(f"\n>>> Optimal Max Leaf Nodes found: {best_leaf_nodes}")
    
    # --- Final Model Training and Reporting ---
    final_clf = DecisionTreeClassifier(
        max_depth=best_depth,
        max_leaf_nodes=best_leaf_nodes,
        random_state=42
    )
    final_clf.fit(X, y, sample_weight=sample_weight)
    
    print("\n--- Final Model ---")
    print(f"Hyperparameters: max_depth={best_depth}, max_leaf_nodes={best_leaf_nodes}")
    
    # --- Print Tree Rules (Un-logged) ---
    print("\n--- Decision Tree Rules (Un-logged) ---")
    tree = final_clf.tree_
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree.feature[node] != -2:  # Is a split node
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            
            # Determine the display name and threshold value, un-logging if necessary.
            display_name = name
            display_threshold = threshold
            if name in log_feature_map:
                display_name = log_feature_map[name]
                display_threshold = np.exp(threshold)

            # Print the 'if' condition for the current split.
            print(f"{indent}if {display_name} <= {display_threshold:,.4f}:")
            recurse(tree.children_left[node], depth + 1)
            
            # Print the 'else' condition and recurse into the right child.
            print(f"{indent}else:  # > {display_threshold:,.4f}")
            recurse(tree.children_right[node], depth + 1)
        else:  # Is a leaf
            class_values = tree.value[node][0]
            predicted_class_idx = np.argmax(class_values)
            predicted_class = final_clf.classes_[predicted_class_idx]
            print(f"{indent}return '{predicted_class}' (counts: {class_values})")

    recurse(0, 0)

    # --- Print Final Performance Metrics (from last CV step) ---
    print("\n--- Final Model Performance (Mean of 5-Fold CV) ---")
    final_metrics = best_metrics_leaves
    print(f"Oracle Performance Ratio: {final_metrics['OPR']:.4%}")
    print(f"Weighted Accuracy:        {final_metrics['Weighted Accuracy']:.4%}")
    print(f"Unweighted Accuracy:      {final_metrics['Unweighted Accuracy']:.4%}")

    # Calculate and print confusion matrix from CV predictions
    y_preds_all_folds = np.array([])
    y_true_all_folds = np.array([])
    
    for _, val_idx in kf.split(X):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        y_preds_all_folds = np.concatenate([y_preds_all_folds, final_clf.predict(X_val)])
        y_true_all_folds = np.concatenate([y_true_all_folds, y_val])

    cm = confusion_matrix(y_true_all_folds, y_preds_all_folds, labels=final_clf.classes_)
    print("\nConfusion Matrix (summed over folds):")
    print(pd.DataFrame(cm, index=[f'Actual: {c}' for c in final_clf.classes_],
                             columns=[f'Predicted: {c}' for c in final_clf.classes_]))

    print(f"\n{'='*25} END {tree_name.upper()} {'='*25}\n")
    
    return final_clf

# =======================================================================================
# Main Execution Logic
# =======================================================================================

if __name__ == "__main__":
    LOG_FILE_PATH = 'bench.txt'

    try:
        print(f"--- Loading and Parsing Log File: {LOG_FILE_PATH} ---")
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            log_data = f.read()
        
        csv_string = parse_and_validate_benchmark_logs(log_data)
        df_long = pd.read_csv(io.StringIO(csv_string))
        print(f"Successfully loaded {len(df_long)} log entries.")

    except FileNotFoundError:
        print(f"FATAL: Log file '{LOG_FILE_PATH}' not found. Please place it in the current directory.")
        exit(1)
    except ValueError as e:
        print(f"FATAL: Log data validation failed. Reason: {e}")
        exit(1)

    # --- Data Preparation and Feature Engineering ---
    print("\n--- Preparing Data for Modeling ---")
    
    # Pivot the table to get one row per unique benchmark scenario
    df_wide = df_long.pivot_table(
        index=['N (Cohort)', 'K (Scores)', 'Subset', 'Freq'],
        columns='Path',
        values='Time (Median)'
    ).reset_index()
    
    df_wide.rename(columns={'Pivot': 'Time_Pivot', 'No-Pivot': 'Time_NoPivot'}, inplace=True)
    
    # Feature Construction
    df_wide['log_N'] = np.log(df_wide['N (Cohort)'])
    df_wide['log_K'] = np.log(df_wide['K (Scores)'])
    df_wide['Subset_frac'] = df_wide['Subset'] / 100.0
    # Add a small epsilon for log_Freq to avoid log(0) issues, though not strictly needed here
    df_wide['log_Freq'] = np.log(df_wide['Freq'] + 1e-12)
    
    log_feature_map = {
        'log_N': 'N (Cohort)',
        'log_K': 'K (Scores)',
        'log_Freq': 'Freq'
    }

    # Target and Sample Weight for Tree 1 & 2
    # The "win" is assigned to the path with the lower time.
    y12 = pd.Series(np.where(df_wide['Time_Pivot'] < df_wide['Time_NoPivot'], 'Pivot', 'No-Pivot'), name="Winning_Path")
    
    time_diff = (df_wide['Time_NoPivot'] - df_wide['Time_Pivot']).abs()
    min_nonzero_weight = time_diff[time_diff > 0].min()
    min_weight = min_nonzero_weight / 2.0
    
    sample_weight12 = time_diff.copy()
    sample_weight12[sample_weight12 == 0] = min_weight

    print("Data preparation and feature engineering complete.")
    
    # --- Tree One: Full Feature Set ---
    X1 = df_wide[['log_N', 'log_K', 'Subset_frac', 'log_Freq']]
    tree_one_model = build_and_evaluate_decision_tree(
        X=X1,
        y=y12,
        sample_weight=sample_weight12,
        log_feature_map=log_feature_map,
        time_pivot=df_wide['Time_Pivot'],
        time_nopivot=df_wide['Time_NoPivot'],
        tree_name="Tree One (All Features)"
    )

    # --- Tree Two: Reduced Feature Set (No Frequency) ---
    X2 = df_wide[['log_N', 'log_K', 'Subset_frac']]
    tree_two_model = build_and_evaluate_decision_tree(
        X=X2,
        y=y12,
        sample_weight=sample_weight12,
        log_feature_map=log_feature_map,
        time_pivot=df_wide['Time_Pivot'],
        time_nopivot=df_wide['Time_NoPivot'],
        tree_name="Tree Two (No Freq Feature)"
    )
    
    # --- Tree Three: Meta-Model (Tree One vs. Tree Two) ---
    print("\n--- Preparing Data for Meta-Model (Tree Three) ---")
    
    # Simulate the total time that would have been taken by following each tree's advice
    preds_t1 = tree_one_model.predict(X1)
    preds_t2 = tree_two_model.predict(X2)
    
    T_tree1 = pd.Series(np.where(preds_t1 == 'Pivot', df_wide['Time_Pivot'], df_wide['Time_NoPivot']), name="Time_Tree1")
    T_tree2 = pd.Series(np.where(preds_t2 == 'Pivot', df_wide['Time_Pivot'], df_wide['Time_NoPivot']), name="Time_Tree2")
    
    # The target for Tree Three is to predict which of the first two models will be faster
    y3 = pd.Series(np.where(T_tree1 < T_tree2, 'TREE_ONE_WIN', 'TREE_TWO_WIN'), name="Winning_Tree")

    # The sample weight is the magnitude of the time difference between the two models' performance
    time_diff_trees = (T_tree2 - T_tree1).abs()
    min_nonzero_weight_trees = time_diff_trees[time_diff_trees > 0].min()
    min_weight_trees = min_nonzero_weight_trees / 2.0
    
    sample_weight3 = time_diff_trees.copy()
    sample_weight3[sample_weight3 == 0] = min_weight_trees
    
    print("Meta-model data prepared. Target is to predict which tree model performs better.")
    
    X3 = df_wide[['log_N', 'log_K', 'Subset_frac']]
    tree_three_model = build_and_evaluate_decision_tree(
        X=X3,
        y=y3,
        sample_weight=sample_weight3,
        log_feature_map=log_feature_map,
        time_pivot=T_tree1,      # "Positive" class time is Tree 1's time
        time_nopivot=T_tree2,  # "Negative" class time is Tree 2's time
        tree_name="Tree Three (Meta-Model)"
    )

    print("\n\nAll models built and evaluated.")
