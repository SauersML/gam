#!/usr/bin/env python3
"""Reproduce a single fuzz seed: emit train/test CSVs and formula metadata.

Usage: python bench/_repro_seed.py <seed> [outdir]
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
sys.path.insert(0, str(REPO_DIR))
from bench.fuzz_vs_mgcv import select_scenarios_backfilled
from bench.run_suite import dataset_for_scenario, folds_for_dataset, zscore_train_test


def main() -> None:
    seed = int(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"/tmp/fuzz_seed_{seed}")
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios, _ = select_scenarios_backfilled(
        seed_start=seed,
        target_count=1,
        excluded_ids=set(),
        max_scenario_cost=None,
    )
    sc = scenarios[0]
    ds = dataset_for_scenario({"name": sc.name})
    folds = folds_for_dataset(ds)
    if not folds:
        raise RuntimeError(f"{sc.name}: no folds generated")
    fold = folds[0]
    train_df = pd.DataFrame(ds["rows"]).iloc[fold.train_idx].copy()
    test_df = pd.DataFrame(ds["rows"]).iloc[fold.test_idx].copy()
    train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
    cols = ds["features"]
    rust_formula = sc.formula
    mgcv_formula_text = sc.mgcv_formula

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
        "scenario": asdict(sc),
        "cols": cols,
        "rust_formula": rust_formula,
        "mgcv_formula": mgcv_formula_text,
        "noise_formula": sc.noise_formula,
    }
    with open(scenario_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("scenario_json:", scenario_json)


if __name__ == "__main__":
    main()
