# Reference Implementation: Polya-Gamma Sampler Comparison

This directory compares the Rust Polya-Gamma sampler to the BayesLogit reference implementation in R.

## How to Run the Comparison

1. **Edit the shell script**
   
   Open `run_all.sh` and modify the arguments at the top if you want to change the z values or the random seed. By default, it runs:
   
   ```bash
   cargo run --release --example reference_implementation -- 0.5 1 2 3.2 5 --seed 123 --b 1.0
   Rscript check.R 0.5 1 2 3.2 5 --seed 123 --b 1.0
   ```

2. **Run the script**

   From this directory, run:
   
   ```bash
   bash run_all.sh
   ```

   This will:
   - Generate Polya-Gamma samples using the Rust implementation
   - Run the R script to compare Rust samples with the BayesLogit reference
   - Produce summary statistics, formal test results, and plots

3. **Inspect the results**

   - Plots and detailed results are saved in the `results/` directory.
   - The console and summary table will show PASS/FAIL for formal tests and summary statistics for all z values.


