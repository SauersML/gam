# GAM Bench

This folder is the engine-level benchmark harness for `gam`.

It measures:
- Fit time
- Predict time
- AUC / Brier (binomial tasks)
- RMSE / R² (gaussian tasks)
- EDF (from fitted engine result)

## What it runs

- Rust engine contender: `gam` via `src/bin/cli.rs`
- External contenders: R `mgcv`, Python `pygam` (run by default)

Canonical benchmark Python code lives in:
- `/Users/user/gam/bench`
- `/Users/user/gam/tests/bench_tools`

## Quick start

```bash
cd /Users/user/gam
python3 bench/run_suite.py
```

Output:
- `benchmarks/results.json`

## Scenario config

Edit:
- `/Users/user/gam/benchmarks/scenarios.json`

Default scenarios:
- `small_dense` (`n=1000, p=10`)
- `medium` (`n=50000, p=50`)
- `pathological_ill_conditioned` (`n=50000, p=80`)
- `lidar_semipar` (real dataset, `n=221`, Gaussian spline GAM on `range -> logratio`)
- `bone_gamair` (real dataset, `n=23`, binomial spline GAM on relapse/death `d` using treatment + smooth time `t`)
- `wine_gamair` (real dataset, `n=38` after dropping missing `price`, Gaussian GAM on `price` with linear year/rain/harvest-temp + smooth summer-temp)
- `horse_colic` (real dataset, rows with complete `rectal_temp/pulse/packed_cell_volume`; binomial GAM on severe outcome `outcome != lived` with linear temp/pcv + smooth pulse)
- `us48_demand_5day` (real dataset, 5-day hourly demand; Gaussian GAM on `Demand` with smooth `hour` + linear forecast/generation/interchange)
- `us48_demand_31day` (real dataset, 31-day hourly demand; same model family as 5-day benchmark)
- `haberman_survival` (real dataset, binomial GAM on 5-year survival outcome using linear age/op-year + smooth axillary nodes)
- `icu_survival_death` (real ICU dataset; time-to-event benchmark with Cox model on time=`pre_icu_los_days`, event=`hospital_death`, and age/BMI/hemodynamics covariates)
- `icu_survival_los` (real ICU dataset; time-to-event benchmark with Cox model on time=`pre_icu_los_days`, event=`hospital_death`, and age/BMI/hemodynamics/temp covariates)
- `heart_failure_survival` (real heart-failure dataset; time-to-event benchmark on follow-up `time` and death event with clinical covariates)
- `cirrhosis_survival` (real cirrhosis dataset; time-to-event benchmark with imputation/encoding and Cox-style survival modeling)

Dataset files live in:
- `/Users/user/gam/benchmarks/datasets`

`run_suite.py` now executes all three contenders (`rust_gam`, `r_mgcv`, `python_pygam`) for every scenario.
