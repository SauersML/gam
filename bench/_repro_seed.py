#!/usr/bin/env python3
"""Reproduce a single fuzz seed: emit train/test CSVs and formula metadata.

Usage: python bench/_repro_seed.py <seed> [outdir]
"""
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
sys.path.insert(0, str(REPO_DIR))
from bench.fuzz_vs_mgcv import (
    generate_data,
    generate_scenario,
    mgcv_formula,
    rust_mean_formula,
)


def main() -> None:
    seed = int(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"/tmp/fuzz_seed_{seed}")
    outdir.mkdir(parents=True, exist_ok=True)

    sc = generate_scenario(seed)
    train_df, test_df, cols = generate_data(sc)
    rust_formula = rust_mean_formula(cols, sc)
    mgcv_formula_text = mgcv_formula(cols, sc)

    train_csv = outdir / "train.csv"
    test_csv = outdir / "test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("scenario:", sc)
    print("rust_mean_formula:", rust_formula)
    print("mgcv_formula     :", mgcv_formula_text)
    print("cols:", cols)
    print("train_csv:", train_csv, "n_train=", len(train_df))
    print("test_csv :", test_csv, "n_test =", len(test_df))

    scenario_json = outdir / "scenario.json"
    payload = {
        "seed": seed,
        "scenario": sc.__dict__,
        "cols": cols,
        "rust_formula": rust_formula,
        "mgcv_formula": mgcv_formula_text,
    }
    with open(scenario_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("scenario_json:", scenario_json)


if __name__ == "__main__":
    main()
