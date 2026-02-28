import subprocess
import sys
from pathlib import Path
import pandas as pd

# Get the absolute path of the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up to the project root directory (assumes tests/ is under calibrate/)
PROJECT_ROOT = SCRIPT_DIR.parent
# Get the workspace root (one level up from the calibrate directory)
WORKSPACE_ROOT = PROJECT_ROOT.parent

EXECUTABLE_NAME = "gnomon"
EXECUTABLE_PATH = WORKSPACE_ROOT / "target" / "release" / EXECUTABLE_NAME

# Input data file provided by the user
INPUT_CSV_FILE = SCRIPT_DIR / 'synthetic_classification_data.csv'
# Temporary file to hold data formatted for the Rust tool
RUST_TRAIN_DATA_PATH = SCRIPT_DIR / "rust_formatted_training_data.tsv"

# --- Model Parameters ---
# The model requires one penalized variable, so we set num_pcs to 1.
NUM_PCS = 1

def prepare_data_for_rust():
    """
    Reads the user-provided CSV, renames columns to match the Rust tool's
    required schema, and saves it as a tab-separated file (TSV).
    """
    print(f"--- Preparing data from '{INPUT_CSV_FILE}' for Rust tool ---")
    if not INPUT_CSV_FILE.is_file():
        print(f"\n--- ERROR: Input data file not found at '{INPUT_CSV_FILE}' ---")
        sys.exit(1)

    # Read the source CSV
    data = pd.read_csv(INPUT_CSV_FILE)

    # Define the mapping from the CSV columns to the Rust tool's required names
    # This mapping is CRITICAL to ensure the correct penalization is applied.
    column_mapping = {
        'variable_two': 'PC1',        # Mapped to 'PC1' to be PENALIZED
        'outcome':      'phenotype'   # The response variable
    }
    required_cols = ["PC1", "phenotype"]

    # Rename the columns
    data_renamed = data.rename(columns=column_mapping)

    # Ensure all required columns are present after renaming
    if not all(col in data_renamed.columns for col in required_cols):
        print(f"Columns found: {list(data_renamed.columns)}")
        sys.exit(1)

    # Save to the TSV format that the Rust tool expects
    print(f"Successfully created formatted data at '{RUST_TRAIN_DATA_PATH}'\n")

def build_rust_project():
    """Checks for the executable and compiles the Rust project if it's not found."""
    if not EXECUTABLE_PATH.is_file():
        print("--- Executable not found. Compiling Rust Project... ---")
        try:
            # Using "--release" for an optimized build
            subprocess.run(
                ["cargo", "build", "--release"],
                check=True, text=True, cwd=PROJECT_ROOT
            )
            print("--- Compilation successful. ---")
            if not EXECUTABLE_PATH.is_file():
                raise FileNotFoundError(f"Compilation succeeded but executable not found at {EXECUTABLE_PATH}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n--- ERROR: Rust compilation failed: {e} ---")
            print("Please ensure Rust/Cargo is installed and you are in the project directory.")
            sys.exit(1)
    else:
        print("--- Found existing release executable. Skipping build. ---")


def run_subprocess(command):
    """Runs a subprocess and streams its output in real-time."""
    print(f"Executing: {' '.join(map(str, command))}\n")
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=PROJECT_ROOT) as proc:
            for line in proc.stdout:
                print(line, end='')
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)
    except KeyboardInterrupt:
        print("\n--- Process interrupted by user. ---")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n--- A command FAILED (Exit Code: {e.returncode}) ---")
        print(f"Failed command: {' '.join(map(str, e.cmd))}")
        sys.exit(1)

def main():
    """Main script execution: Prepare data, build if needed, train model."""
    build_rust_project()
    prepare_data_for_rust()

    print("\n--- Running 'train' command to fit and save the GAM ---")
    
    # This command fits the exact model structure requested.
    # The tool's internal logic applies the correct penalization based on column names.
    train_command = [
        str(EXECUTABLE_PATH), "train",
        "--num-pcs", str(NUM_PCS),
        # These knot/degree values are analogous to the basis complexity in the R code
        "--pc-knots", "11",   # for 'PC1' (variable_two)
        "--pc-degree", "3",
        str(RUST_TRAIN_DATA_PATH),
    ]
    
    run_subprocess(train_command)
    
    print("\n--- Model training complete. ---")

if __name__ == "__main__":
    main()
