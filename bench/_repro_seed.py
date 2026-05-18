#!/usr/bin/env python3
"""Reproduce a single fuzz seed: emit train/test CSVs + rust + mgcv formulas + a small mgcv diag script.

Usage: python bench/_repro_seed.py <seed> [outdir]
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fuzz_vs_mgcv import (
    generate_scenario, generate_data,
    rust_mean_formula,
    mgcv_formula,
)

seed = int(sys.argv[1])
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"/tmp/fuzz_seed_{seed}")
outdir.mkdir(parents=True, exist_ok=True)

sc = generate_scenario(seed)
print("scenario:", sc)
print("rust_mean_formula:", rust_mean_formula([f"x{i}" for i in range(sc.n_smooths)], sc))
print("mgcv_formula     :", mgcv_formula([f"x{i}" for i in range(sc.n_smooths)], sc))

train_df, test_df, cols = generate_data(sc)
train_csv = outdir / "train.csv"
test_csv = outdir / "test.csv"
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("cols:", cols)
print("train_csv:", train_csv, "n_train=", len(train_df))
print("test_csv :", test_csv,  "n_test =", len(test_df))

scenario_json = outdir / "scenario.json"
with open(scenario_json, "w") as f:
    json.dump({
        "seed": seed,
        "scenario": sc.__dict__,
        "cols": cols,
        "rust_formula": rust_mean_formula(cols, sc),
        "mgcv_formula": mgcv_formula(cols, sc),
    }, f, indent=2, default=str)
print("scenario_json:", scenario_json)
