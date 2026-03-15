#!/bin/bash
set -e

# Arguments to pass to both Rust and R
ARGS="0.5 1 2 3.2 5 --seed 2025 --b 1.0"

cargo build --release --example reference_implementation

# Run the Rust sampler (timed, excluding compile time)
echo "Running Rust sampler..."

./target/release/examples/reference_implementation $ARGS
echo "[Rust] Wrote samples to data/pg_samples.csv"

echo "Running R sampler and diagnostics..."

Rscript examples/reference_implementation/check.R $ARGS

echo "[R] Diagnostics completed"
