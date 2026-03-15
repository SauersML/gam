#!/bin/bash

set -e  # Exit on error

# Set up directories
PROJECT_ROOT=$(git rev-parse --show-toplevel)
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$BENCH_DIR/results"
PLOTS_DIR="$BENCH_DIR/plots"
R_SCRIPT="$BENCH_DIR/benchmark.R"
PY_SCRIPT="$BENCH_DIR/compare.py"
VENV_DIR="$BENCH_DIR/venv"

# Create output directories
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR"

echo "📊 Setting up benchmark environment..."

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✅ Created virtual environment at $VENV_DIR"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install pandas matplotlib seaborn

echo "✅ Python environment ready"

# Install R dependencies if needed
echo "📦 Installing R dependencies..."
Rscript -e 'if (!requireNamespace("BayesLogit", quietly = TRUE)) install.packages("BayesLogit", repos="https://cloud.r-project.org")'
Rscript -e 'if (!requireNamespace("microbenchmark", quietly = TRUE)) install.packages("microbenchmark", repos="https://cloud.r-project.org")'
Rscript -e 'if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", repos="https://cloud.r-project.org")'

# Check if R is installed
if ! command -v Rscript &> /dev/null; then
    echo "❌ Rscript is not installed. Please install R to run the benchmarks."
    exit 1
fi

# # Run R benchmarks
echo "🏃 Running R benchmarks (this may take a while)..."
R_CSV="$RESULTS_DIR/bayeslogit_results.csv"
Rscript $R_SCRIPT

echo "🏃 Running Rust benchmarks (this may take a while)..."
RUST_CSV="$RESULTS_DIR/rust_bench_output.csv"
cargo bench --bench comparison -- --save-baseline comparison --output-format bencher | tee "$RESULTS_DIR/rust_bench_output.txt"

# Convert Rust benchmark output to CSV format
echo "📊 Converting benchmark results..."
python "$BENCH_DIR/convert.py" "$RESULTS_DIR"

# Generate comparison plots
echo "📈 Generating comparison visualizations..."
python "$PY_SCRIPT" "$R_CSV" "$RUST_CSV" "$PLOTS_DIR"

# Deactivate virtual environment
deactivate

echo "✅ Benchmarking complete!"
echo "📊 Results saved to: $RESULTS_DIR"
echo "📈 Visualizations saved to: $PLOTS_DIR"
