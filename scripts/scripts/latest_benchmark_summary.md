# Latest Benchmark Run With Data (Partial)

- Workflow: `benchmark.yml`
- Run ID: `22609300722`
- Status: `queued`
- Conclusion: `None`
- URL: https://github.com/SauersML/gam/actions/runs/22609300722
- Completed shard artifacts found: `45`

**Scenario:** `bone_gamair` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.645868 | 1 | 0.226882 | 3 | 0.040119 | 1 | 0.717391 | 2 | 0.306 | 0.031 |
| `python_pygam` | 1.312763 | 2 | 0.112720 | 1 | -6462.229750 | 2 | 0.927536 | 1 | 0.520 | 0.003 |
| `r_mgcv` | 4.969274 | 3 | 0.226304 | 2 | -111895014928077424.000000 | 3 | 0.710152 | 3 | 0.459 | 0.020 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LogisticGAM(l(0) + s(1, n_splines=8)) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv`: `y ~ trt_auto + s(t, bs='ps', k=min(12, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `geo_disease_eas3_psperpc_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.241750 | 1 | 0.070301 | 1 | 0.242982 | 1 | 0.816407 | 3 | 52.370 | 0.057 |
| `r_mgcv` | 0.241942 | 2 | 0.070400 | 2 | 0.242215 | 2 | 0.816677 | 1 | 7.803 | 0.039 |
| `python_pygam` | 0.242011 | 3 | 0.070406 | 3 | 0.241989 | 3 | 0.816644 | 2 | 4.039 | 0.015 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(16, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=12), j=0..2) [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `geo_disease_eas3_psperpc_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.241685 | 1 | 0.070316 | 1 | 0.243246 | 1 | 0.816555 | 2 | 119.505 | 0.071 |
| `python_pygam` | 0.241779 | 2 | 0.070378 | 2 | 0.242915 | 2 | 0.816933 | 1 | 7.853 | 0.019 |
| `r_mgcv` | 0.242022 | 3 | 0.070420 | 3 | 0.241915 | 3 | 0.816177 | 3 | 11.722 | 0.046 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=24), j=0..2) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(28, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(28, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(28, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `geo_disease_eas3_psperpc_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.241668 | 1 | 0.070340 | 1 | 0.243313 | 1 | 0.816580 | 3 | 28.650 | 0.035 |
| `r_mgcv` | 0.241822 | 2 | 0.070420 | 2 | 0.242715 | 2 | 0.816957 | 2 | 1.559 | 0.038 |
| `python_pygam` | 0.241893 | 3 | 0.070424 | 3 | 0.242407 | 3 | 0.818522 | 1 | 2.677 | 0.012 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(10, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=6), j=0..2) [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `geo_disease_eas3_tp_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.242202 | 1 | 0.070520 | 1 | 0.241195 | 1 | 0.815797 | 1 | 3.376 | 0.187 |
| `rust_gam` | 0.242523 | 2 | 0.070606 | 2 | 0.239970 | 2 | 0.814707 | 2 | 26.787 | 0.053 |

**Model specs**

* `r_mgcv`: `y ~ s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `geo_disease_eas3_tp_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.242202 | 1 | 0.070480 | 1 | 0.241275 | 1 | 0.815897 | 1 | 4.676 | 0.238 |
| `rust_gam` | 0.243424 | 2 | 0.070809 | 2 | 0.236349 | 2 | 0.812580 | 2 | 448.276 | 0.071 |

**Model specs**

* `r_mgcv`: `y ~ s(pc1, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(24, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `geo_disease_eas3_tp_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.242382 | 1 | 0.070500 | 1 | 0.240575 | 1 | 0.814037 | 2 | 3.654 | 0.170 |
| `rust_gam` | 0.242688 | 2 | 0.070644 | 2 | 0.239312 | 2 | 0.814771 | 1 | 18.150 | 0.059 |

**Model specs**

* `r_mgcv`: `y ~ s(pc1, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(6, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `geo_disease_eas_psperpc_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.230396 | 1 | 0.066348 | 1 | 0.287220 | 1 | 0.833930 | 2 | 1250.825 | 0.098 |
| `r_mgcv` | 0.231100 | 2 | 0.066500 | 2 | 0.284480 | 2 | 0.832318 | 3 | 118.880 | 0.130 |
| `python_pygam` | 0.233222 | 3 | 0.066923 | 3 | 0.276208 | 3 | 0.836159 | 1 | 20.580 | 0.052 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc5, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc6, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc7, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc8, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc9, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc10, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc11, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc12, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc13, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc14, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc15, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc16, bs='ps', k=min(10, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=6), j=0..15) [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `geo_disease_eas_tp_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.232224 | 1 | 0.066495 | 1 | 0.280217 | 1 | 0.836893 | 1 | 46.560 | 0.060 |
| `r_mgcv` | 0.232320 | 2 | 0.066520 | 2 | 0.279781 | 2 | 0.836400 | 2 | 9.455 | 0.195 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `geo_disease_eas_tp_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.232360 | 1 | 0.066500 | 1 | 0.279761 | 1 | 0.836440 | 1 | 14.697 | 0.248 |
| `rust_gam` | 0.233035 | 2 | 0.066695 | 2 | 0.277079 | 2 | 0.835304 | 2 | 152.039 | 0.081 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(24, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `geo_disease_eas_tp_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.232800 | 1 | 0.066540 | 1 | 0.277961 | 1 | 0.835220 | 1 | 4.508 | 0.181 |
| `rust_gam` | 0.233442 | 2 | 0.066761 | 2 | 0.275487 | 2 | 0.834315 | 2 | 28.699 | 0.056 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(6, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `geo_disease_ps_per_pc` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.647863 | 1 | 0.227861 | 1 | 0.105592 | 1 | 0.665316 | 1 | 95.779 | 0.270 |
| `rust_gam` | 0.648542 | 2 | 0.228200 | 2 | 0.103891 | 2 | 0.663814 | 2 | 1670.041 | 0.148 |

**Model specs**

* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc5, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc6, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc7, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc8, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc9, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc10, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc11, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc12, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc13, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc14, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc15, bs='ps', k=min(14, nrow(train_df)-1)) + s(pc16, bs='ps', k=min(14, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `geo_disease_shrinkage` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.653415 | 1 | 0.230571 | 1 | 0.091807 | 1 | 0.654388 | 1 | 61.792 | 0.054 |
| `r_mgcv` | 0.653483 | 2 | 0.230601 | 2 | 0.091613 | 2 | 0.654234 | 2 | 4.237 | 0.168 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc4, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `geo_disease_tp` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.653415 | 1 | 0.230571 | 1 | 0.091807 | 1 | 0.654388 | 1 | 62.172 | 0.055 |
| `r_mgcv` | 0.653483 | 2 | 0.230601 | 2 | 0.091613 | 2 | 0.654234 | 2 | 4.235 | 0.170 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc4, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `geo_latlon_equatornoise_tp_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.281297 | 1 | 0.075512 | 2 | 0.024754 | 1 | 0.611412 | 1 | 10.450 | 0.036 |
| `r_mgcv` | 0.281359 | 2 | 0.075500 | 1 | 0.024505 | 2 | 0.610211 | 2 | 2.839 | 0.123 |
| `r_gamlss` | 0.287457 | 3 | 0.076380 | 3 | -0.003489 | 3 | 0.593862 | 3 | 35.989 | 0.304 |
| `rust_gamlss` | 0.301068 | 4 | 0.080442 | 4 | -0.068589 | 4 | 0.584234 | 4 | 158.356 | 0.073 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pb(pc1, df=min(12, nrow(train_df)-1)) + pb(pc2, df=min(12, nrow(train_df)-1)) + pb(pc3, df=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_latlon_equatornoise_tp_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.281329 | 1 | 0.075512 | 2 | 0.024612 | 1 | 0.610932 | 1 | 70.991 | 0.063 |
| `r_mgcv` | 0.281359 | 2 | 0.075500 | 1 | 0.024525 | 2 | 0.610311 | 2 | 6.660 | 0.162 |
| `r_gamlss` | 0.282479 | 3 | 0.075640 | 3 | 0.019445 | 3 | 0.603591 | 3 | 13.474 | 0.305 |
| `rust_gamlss` | 0.298995 | 4 | 0.080332 | 4 | -0.059974 | 4 | 0.595919 | 4 | 362.035 | 0.111 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + s(pc1, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(24, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pb(pc1, df=min(24, nrow(train_df)-1)) + pb(pc2, df=min(24, nrow(train_df)-1)) + pb(pc3, df=min(24, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_latlon_equatornoise_tp_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.281281 | 1 | 0.075511 | 2 | 0.024826 | 1 | 0.611700 | 1 | 5.524 | 0.059 |
| `r_mgcv` | 0.281359 | 2 | 0.075500 | 1 | 0.024505 | 2 | 0.610131 | 2 | 1.515 | 0.118 |
| `r_gamlss` | 0.283179 | 3 | 0.075780 | 3 | 0.016106 | 3 | 0.600227 | 3 | 32.330 | 0.321 |
| `rust_gamlss` | 0.369363 | 4 | 0.110930 | 4 | -0.542414 | 4 | 0.584271 | 4 | 151.881 | 0.100 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + s(pc1, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(6, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pb(pc1, df=min(6, nrow(train_df)-1)) + pb(pc2, df=min(6, nrow(train_df)-1)) + pb(pc3, df=min(6, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_latlon_superpopnoise_psperpc_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.252769 | 1 | 0.064544 | 1 | -0.006675 | 1 | 0.466192 | 1 | 191.351 | 0.058 |
| `r_mgcv` | 0.252902 | 2 | 0.064560 | 2 | -0.007365 | 2 | 0.453750 | 3 | 14.095 | 0.041 |
| `r_gamlss` | 0.267105 | 3 | 0.066361 | 3 | -0.080663 | 3 | 0.446011 | 4 | 136.222 | 0.678 |
| `rust_gamlss` | 0.292741 | 4 | 0.075517 | 4 | -0.218936 | 4 | 0.464595 | 2 | 284.272 | 0.105 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc5, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc6, bs='ps', k=min(16, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pb(pc1, df=min(12, nrow(train_df)-1)) + pb(pc2, df=min(12, nrow(train_df)-1)) + pb(pc3, df=min(12, nrow(train_df)-1)) + pb(pc4, df=min(12, nrow(train_df)-1)) + pb(pc5, df=min(12, nrow(train_df)-1)) + pb(pc6, df=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_latlon_superpopnoise_psperpc_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.252760 | 1 | 0.064549 | 1 | -0.006629 | 1 | 0.469698 | 1 | 79.912 | 0.056 |
| `r_mgcv` | 0.252882 | 2 | 0.064560 | 2 | -0.007245 | 2 | 0.459251 | 3 | 6.705 | 0.041 |
| `python_pygam` | 0.253951 | 3 | 0.064699 | 3 | -0.012701 | 3 | 0.458009 | 4 | 3.339 | 0.018 |
| `r_gamlss` | 0.257744 | 4 | 0.065141 | 4 | -0.032077 | 4 | 0.455024 | 5 | 82.909 | 0.672 |
| `rust_gamlss` | 0.279601 | 5 | 0.071369 | 5 | -0.147842 | 5 | 0.468098 | 2 | 163.530 | 0.083 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc5, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc6, bs='ps', k=min(10, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=6), j=0..5) [geo_latlon] [lam by UBRE/GCV; 5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pb(pc1, df=min(6, nrow(train_df)-1)) + pb(pc2, df=min(6, nrow(train_df)-1)) + pb(pc3, df=min(6, nrow(train_df)-1)) + pb(pc4, df=min(6, nrow(train_df)-1)) + pb(pc5, df=min(6, nrow(train_df)-1)) + pb(pc6, df=min(6, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_latlon_superpopnoise_tp_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.252976 | 1 | 0.064578 | 1 | -0.007734 | 1 | 0.470944 | 2 | 2.861 | 0.055 |
| `r_mgcv` | 0.253002 | 2 | 0.064581 | 2 | -0.007847 | 2 | 0.469639 | 3 | 2.525 | 0.124 |
| `r_gamlss` | 0.260184 | 3 | 0.065481 | 3 | -0.044595 | 3 | 0.465073 | 4 | 37.001 | 0.305 |
| `rust_gamlss` | 0.267098 | 4 | 0.068356 | 4 | -0.083229 | 4 | 0.507329 | 1 | 88.769 | 0.092 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pb(pc1, df=min(12, nrow(train_df)-1)) + pb(pc2, df=min(12, nrow(train_df)-1)) + pb(pc3, df=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_latlon_superpopnoise_tp_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.252992 | 1 | 0.064579 | 1 | -0.007812 | 1 | 0.470006 | 3 | 53.946 | 0.053 |
| `r_mgcv` | 0.253002 | 2 | 0.064581 | 2 | -0.007847 | 2 | 0.469639 | 4 | 6.622 | 0.165 |
| `r_gamlss` | 0.254143 | 3 | 0.064741 | 3 | -0.013789 | 3 | 0.486224 | 2 | 33.169 | 0.309 |
| `rust_gamlss` | 0.255690 | 4 | 0.064996 | 4 | -0.021623 | 4 | 0.497633 | 1 | 720.758 | 0.081 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + s(pc1, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(24, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pb(pc1, df=min(24, nrow(train_df)-1)) + pb(pc2, df=min(24, nrow(train_df)-1)) + pb(pc3, df=min(24, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_latlon_superpopnoise_tp_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.252998 | 1 | 0.064580 | 1 | -0.007843 | 1 | 0.471797 | 3 | 4.130 | 0.054 |
| `r_mgcv` | 0.253002 | 2 | 0.064581 | 2 | -0.007847 | 2 | 0.469639 | 4 | 1.428 | 0.115 |
| `r_gamlss` | 0.255223 | 3 | 0.064861 | 3 | -0.019231 | 3 | 0.479968 | 1 | 29.778 | 0.396 |
| `rust_gamlss` | 0.277312 | 4 | 0.070538 | 4 | -0.138125 | 4 | 0.477116 | 2 | 47.039 | 0.074 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + s(pc1, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(6, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pb(pc1, df=min(6, nrow(train_df)-1)) + pb(pc2, df=min(6, nrow(train_df)-1)) + pb(pc3, df=min(6, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_subpop16_tp_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.478756 | 1 | 0.155539 | 1 | 0.111312 | 1 | 0.682310 | 1 | 3.634 | 0.129 |
| `rust_gam` | 0.479152 | 2 | 0.155717 | 2 | 0.110069 | 2 | 0.681206 | 2 | 169.427 | 0.055 |
| `r_gamlss` | 0.483317 | 3 | 0.156658 | 3 | 0.098151 | 3 | 0.679272 | 3 | 38.607 | 0.321 |
| `rust_gamlss` | 0.484505 | 4 | 0.157428 | 4 | 0.094573 | 4 | 0.671933 | 4 | 176.924 | 0.088 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + pb(pc1, df=min(12, nrow(train_df)-1)) + pb(pc2, df=min(12, nrow(train_df)-1)) + pb(pc3, df=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_subpop16_tp_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.478716 | 1 | 0.155519 | 1 | 0.111412 | 1 | 0.682590 | 1 | 5.028 | 0.165 |
| `r_gamlss` | 0.478776 | 2 | 0.155559 | 2 | 0.111132 | 2 | 0.682550 | 2 | 17.375 | 0.311 |
| `rust_gam` | 0.480437 | 3 | 0.156228 | 3 | 0.106400 | 3 | 0.677252 | 3 | 124.526 | 0.069 |
| `rust_gamlss` | 0.483894 | 4 | 0.157141 | 4 | 0.096379 | 4 | 0.672940 | 4 | 248.955 | 0.111 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(24, nrow(train_df)-1)) [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + pb(pc1, df=min(24, nrow(train_df)-1)) + pb(pc2, df=min(24, nrow(train_df)-1)) + pb(pc3, df=min(24, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `geo_subpop16_tp_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.478816 | 1 | 0.155599 | 1 | 0.111012 | 1 | 0.681771 | 1 | 3.091 | 0.121 |
| `rust_gam` | 0.479157 | 2 | 0.155739 | 2 | 0.110061 | 2 | 0.681126 | 2 | 18.806 | 0.055 |
| `r_gamlss` | 0.480316 | 3 | 0.155939 | 3 | 0.106733 | 3 | 0.680510 | 3 | 24.037 | 0.402 |
| `rust_gamlss` | 0.484372 | 4 | 0.157340 | 4 | 0.094926 | 4 | 0.673806 | 4 | 77.699 | 0.076 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(6, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_gamlss`: `gamlss(BI; sigma.formula=~1): y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + pb(pc1, df=min(6, nrow(train_df)-1)) + pb(pc2, df=min(6, nrow(train_df)-1)) + pb(pc3, df=min(6, nrow(train_df)-1)) [5-fold CV]`
* `rust_gamlss`: `gamlss binomial-probit location-scale via release binary [5-fold CV]`

**Scenario:** `haberman_survival` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.546400 | 1 | 0.180297 | 1 | 0.084888 | 1 | 0.674624 | 2 | 0.075 | 0.056 |
| `r_mgcv` | 0.549404 | 2 | 0.180628 | 2 | 0.077073 | 2 | 0.677850 | 1 | 0.363 | 0.025 |
| `python_pygam` | 0.593828 | 3 | 0.187367 | 3 | -0.058754 | 3 | 0.663294 | 3 | 0.735 | 0.004 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ age + op_year + s(axil_nodes, bs='ps', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LogisticGAM(l(0) + l(1) + s(2, n_splines=8)) [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `horse_colic` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.489385 | 1 | 0.160527 | 1 | 0.359602 | 1 | 0.822402 | 1 | 0.057 | 0.052 |
| `r_mgcv` | 0.490626 | 2 | 0.161119 | 2 | 0.356112 | 2 | 0.820074 | 2 | 0.211 | 0.023 |
| `python_pygam` | 0.497308 | 3 | 0.162545 | 3 | 0.337119 | 3 | 0.818164 | 3 | 0.709 | 0.004 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ rectal_temp + packed_cell_volume + s(pulse, bs='ps', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LogisticGAM(l(0) + s(1, n_splines=8) + l(2)) [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `lidar_semipar` (gaussian)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | MSE (↓ better) | MSE rank | LogLoss (↓ better) | LogLoss rank | RMSE (↓ better) | RMSE rank | MAE (↓ better) | MAE rank | R2 (↑ better) | R2 rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv_gaulss` | 0.006707 | 1 | 3.855891 | 5 | 0.081769 | 1 | 0.056971 | 1 | 0.914753 | 1 | 0.937 | 0.022 |
| `r_brms` | 0.006727 | 2 | -1.455716 | 1 | 0.081890 | 2 | 0.057052 | 2 | 0.914431 | 2 | 485.153 | 1.883 |
| `r_mgcv` | 0.006798 | 3 | -1.070717 | 2 | 0.082326 | 3 | 0.057210 | 3 | 0.913579 | 3 | 0.518 | 0.020 |
| `rust_gam` | 0.006853 | 4 | -1.065930 | 4 | 0.082658 | 4 | 0.057448 | 4 | 0.912856 | 4 | 0.161 | 0.053 |
| `python_pygam` | 0.006855 | 5 | -1.066313 | 3 | 0.082675 | 5 | 0.057513 | 5 | 0.912782 | 5 | 0.499 | 0.003 |

**Model specs**

* `r_mgcv_gaulss`: `gam(list(y ~ s(range, bs='ps', k=min(28, nrow(train_df)-1)), ~ range), family=gaulss()) [5-fold CV]`
* `r_brms`: `brms::brm(bf(y ~ s(range, bs='ps', k=min(28, nrow(train_df)-1)), sigma ~ range); gaussian) [5-fold CV]`
* `r_mgcv`: `y ~ s(range, bs='ps', k=min(28, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LinearGAM(s(0, n_splines=25)) [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `medium` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `python_pygam` | 0.543938 | 1 | 0.183825 | 1 | 0.336135 | 1 | 0.793847 | 1 | 7.548 | 0.026 |
| `rust_gam` | 0.543943 | 2 | 0.183827 | 2 | 0.336125 | 2 | 0.793841 | 2 | 5.528 | 0.074 |
| `r_mgcv` | 0.543980 | 3 | 0.183860 | 3 | 0.336060 | 3 | 0.793840 | 3 | 15.972 | 0.083 |

**Model specs**

* `python_pygam`: `LogisticGAM(l(0) + s(1, n_splines=8)) [lam by UBRE/GCV; 5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ x1 + s(x2, bs='ps', k=min(11, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `papuan_oce4_psperpc_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.326016 | 1 | 0.106519 | 1 | 0.369192 | 1 | 0.826845 | 1 | 13.383 | 0.047 |
| `rust_gam` | 0.326288 | 2 | 0.106655 | 2 | 0.368530 | 2 | 0.826279 | 2 | 149.242 | 0.055 |
| `python_pygam` | 0.328338 | 3 | 0.107204 | 3 | 0.363112 | 3 | 0.822986 | 3 | 4.426 | 0.019 |

**Model specs**

* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(16, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=12), j=0..3) [papuan_oce] [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `papuan_oce4_psperpc_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.325976 | 1 | 0.106499 | 1 | 0.369312 | 1 | 0.826865 | 1 | 31.825 | 0.056 |
| `rust_gam` | 0.326275 | 2 | 0.106642 | 2 | 0.368570 | 2 | 0.826511 | 2 | 337.886 | 0.087 |
| `python_pygam` | 0.327970 | 3 | 0.107120 | 3 | 0.364088 | 3 | 0.822972 | 3 | 11.380 | 0.025 |

**Model specs**

* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(28, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(28, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(28, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(28, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=24), j=0..3) [papuan_oce] [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `papuan_oce4_psperpc_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.326652 | 1 | 0.106691 | 1 | 0.367558 | 1 | 0.824721 | 2 | 74.573 | 0.055 |
| `r_mgcv` | 0.326916 | 2 | 0.106799 | 2 | 0.366852 | 2 | 0.824284 | 3 | 6.097 | 0.042 |
| `python_pygam` | 0.328337 | 3 | 0.106955 | 3 | 0.363067 | 3 | 0.825386 | 1 | 2.624 | 0.016 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(10, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(10, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LogisticGAM(sum_j s(j, n_splines=6), j=0..3) [papuan_oce] [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `papuan_oce4_tp_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.326076 | 1 | 0.106499 | 1 | 0.369032 | 1 | 0.825965 | 2 | 5.365 | 0.186 |
| `rust_gam` | 0.326549 | 2 | 0.106668 | 2 | 0.367802 | 2 | 0.826623 | 1 | 42.506 | 0.055 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `papuan_oce4_tp_k24` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.325996 | 1 | 0.106479 | 1 | 0.369232 | 1 | 0.826885 | 1 | 8.929 | 0.246 |
| `rust_gam` | 0.326281 | 2 | 0.106613 | 2 | 0.368508 | 2 | 0.825746 | 2 | 121.018 | 0.072 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + s(pc1, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(24, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(24, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `papuan_oce4_tp_k6` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.327054 | 1 | 0.106847 | 2 | 0.366439 | 1 | 0.824556 | 1 | 27.614 | 0.055 |
| `r_mgcv` | 0.327136 | 2 | 0.106819 | 1 | 0.366252 | 2 | 0.823944 | 2 | 2.998 | 0.180 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv`: `y ~ pc4 + s(pc1, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(6, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(6, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `papuan_oce_psperpc_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.327456 | 1 | 0.107239 | 1 | 0.365472 | 1 | 0.821407 | 1 | 223.054 | 0.122 |
| `rust_gam` | 0.327963 | 2 | 0.107513 | 2 | 0.364118 | 2 | 0.820189 | 2 | 3793.615 | 0.198 |

**Model specs**

* `r_mgcv`: `y ~ s(pc1, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc2, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc3, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc4, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc5, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc6, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc7, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc8, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc9, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc10, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc11, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc12, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc13, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc14, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc15, bs='ps', k=min(16, nrow(train_df)-1)) + s(pc16, bs='ps', k=min(16, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `papuan_oce_tp_k12` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.327996 | 1 | 0.107439 | 1 | 0.364030 | 1 | 0.819583 | 1 | 5.900 | 0.200 |
| `rust_gam` | 0.328445 | 2 | 0.107607 | 2 | 0.362839 | 2 | 0.818526 | 2 | 66.825 | 0.057 |

**Model specs**

* `r_mgcv`: `y ~ pc4 + pc5 + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 + pc14 + pc15 + pc16 + s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) + s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `pathological_ill_conditioned` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `rust_gam` | 0.546839 | 1 | 0.184599 | 2 | 0.329418 | 1 | 0.791643 | 2 | 5.217 | 0.075 |
| `python_pygam` | 0.546843 | 2 | 0.184589 | 1 | 0.329409 | 2 | 0.791664 | 1 | 7.609 | 0.026 |
| `r_mgcv` | 0.546880 | 3 | 0.184620 | 3 | 0.329380 | 3 | 0.791640 | 3 | 16.214 | 0.279 |

**Model specs**

* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LogisticGAM(l(0) + s(1, n_splines=8)) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv`: `y ~ x1 + s(x2, bs='ps', k=min(11, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `prostate_gamair` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 0.613043 | 1 | 0.215088 | 1 | 0.195739 | 1 | 0.703762 | 2 | 0.321 | 0.025 |
| `rust_gam` | 0.613567 | 2 | 0.215174 | 2 | 0.194536 | 2 | 0.703727 | 3 | 0.138 | 0.056 |
| `python_pygam` | 0.632911 | 3 | 0.215498 | 3 | 0.144018 | 3 | 0.704720 | 1 | 0.708 | 0.004 |

**Model specs**

* `r_mgcv`: `y ~ pc1 + s(pc2, bs='ps', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LogisticGAM(l(0) + s(1, n_splines=8)) [lam by UBRE/GCV; 5-fold CV]`

**Scenario:** `small_dense` (binomial)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | LogLoss (↓ better) | LogLoss rank | Brier (↓ better) | Brier rank | NagelkerkeR2 (↑ better) | NagelkerkeR2 rank | AUC (↑ better) | AUC rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `python_pygam` | 0.535306 | 1 | 0.181100 | 1 | 0.343402 | 1 | 0.796148 | 3 | 0.776 | 0.004 |
| `r_mgcv` | 0.535659 | 2 | 0.181240 | 2 | 0.342723 | 2 | 0.796399 | 1 | 0.315 | 0.024 |
| `rust_gam` | 0.536154 | 3 | 0.181387 | 3 | 0.341680 | 3 | 0.796231 | 2 | 0.110 | 0.052 |

**Model specs**

* `python_pygam`: `LogisticGAM(l(0) + s(1, n_splines=8)) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv`: `y ~ x1 + s(x2, bs='ps', k=min(11, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`

**Scenario:** `us48_demand_31day` (gaussian)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | MSE (↓ better) | MSE rank | LogLoss (↓ better) | LogLoss rank | RMSE (↓ better) | RMSE rank | MAE (↓ better) | MAE rank | R2 (↑ better) | R2 rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `python_pygam` | 10528141.583381 | 1 | 9.515914 | 1 | 3219.685160 | 1 | 1771.666234 | 4 | 0.996504 | 1 | 0.676 | 0.005 |
| `r_mgcv_gaulss` | 10542129.338531 | 2 | 105279854912168.625000 | 5 | 3220.371980 | 2 | 1708.333020 | 1 | 0.996480 | 4 | 0.787 | 0.020 |
| `r_mgcv` | 10546532.864293 | 3 | 9.516860 | 2 | 3222.484880 | 3 | 1759.275780 | 2 | 0.996500 | 2 | 0.401 | 0.024 |
| `rust_gam` | 10557005.891706 | 4 | 9.517609 | 3 | 3224.325807 | 4 | 1761.910960 | 3 | 0.996495 | 3 | 0.100 | 0.050 |
| `r_brms` | 8140310681.950356 | 5 | 9.965560 | 4 | 58631.328700 | 5 | 21008.647360 | 5 | -1.715900 | 5 | 5726.897 | 1.957 |

**Model specs**

* `python_pygam`: `LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3)) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv_gaulss`: `gam(list(y ~ demand_forecast + net_generation + total_interchange + s(hour, bs='ps', k=min(16, nrow(train_df)-1)), ~ hour + demand_forecast + net_generation + total_interchange), family=gaulss()) [5-fold CV]`
* `r_mgcv`: `y ~ demand_forecast + net_generation + total_interchange + s(hour, bs='ps', k=min(16, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_brms`: `brms::brm(bf(y ~ demand_forecast + net_generation + total_interchange + s(hour, bs='ps', k=min(16, nrow(train_df)-1)), sigma ~ hour + demand_forecast + net_generation + total_interchange); gaussian) [5-fold CV]`

**Scenario:** `us48_demand_5day` (gaussian)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | MSE (↓ better) | MSE rank | LogLoss (↓ better) | LogLoss rank | RMSE (↓ better) | RMSE rank | MAE (↓ better) | MAE rank | R2 (↑ better) | R2 rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv` | 7764914.796517 | 1 | 9.378167 | 1 | 2763.450019 | 1 | 2368.046965 | 2 | 0.995078 | 1 | 0.235 | 0.024 |
| `python_pygam` | 7896366.226414 | 2 | 9.392661 | 2 | 2781.444950 | 2 | 2374.608015 | 3 | 0.994992 | 2 | 0.622 | 0.004 |
| `rust_gam` | 7946968.766657 | 3 | 9.393784 | 3 | 2798.664490 | 3 | 2380.923926 | 4 | 0.994947 | 3 | 0.056 | 0.050 |
| `r_mgcv_gaulss` | 8839428.630580 | 4 | 71141734575247.234375 | 5 | 2963.760626 | 4 | 2306.335242 | 1 | 0.994179 | 4 | 0.484 | 0.027 |
| `r_brms` | 192973533849.457886 | 5 | 10.830951 | 4 | 292098.694877 | 5 | 129509.402468 | 5 | -103.803853 | 5 | 1418.310 | 1.304 |

**Model specs**

* `r_mgcv`: `y ~ demand_forecast + net_generation + total_interchange + s(hour, bs='ps', k=min(12, nrow(train_df)-1)) [5-fold CV]`
* `python_pygam`: `LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3)) [lam by UBRE/GCV; 5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_mgcv_gaulss`: `gam(list(y ~ demand_forecast + net_generation + total_interchange + s(hour, bs='ps', k=min(12, nrow(train_df)-1)), ~ hour + demand_forecast + net_generation + total_interchange), family=gaulss()) [5-fold CV]`
* `r_brms`: `brms::brm(bf(y ~ demand_forecast + net_generation + total_interchange + s(hour, bs='ps', k=min(12, nrow(train_df)-1)), sigma ~ hour + demand_forecast + net_generation + total_interchange); gaussian) [5-fold CV]`

**Scenario:** `wine_gamair` (gaussian)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | MSE (↓ better) | MSE rank | LogLoss (↓ better) | LogLoss rank | RMSE (↓ better) | RMSE rank | MAE (↓ better) | MAE rank | R2 (↑ better) | R2 rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `python_pygam` | 146.306165 | 1 | 4.082031 | 2 | 11.665187 | 1 | 9.221229 | 1 | 0.381105 | 1 | 0.636 | 0.005 |
| `r_mgcv` | 147.170600 | 2 | 4.081308 | 1 | 11.748674 | 2 | 9.670384 | 2 | 0.311150 | 3 | 0.286 | 0.023 |
| `rust_gam` | 164.771424 | 3 | 4.177695 | 3 | 12.366721 | 3 | 10.047063 | 4 | 0.255456 | 4 | 0.214 | 0.051 |
| `r_brms` | 171.675091 | 4 | 6.087926 | 4 | 12.366729 | 4 | 9.859755 | 3 | 0.320045 | 2 | 363.742 | 2.265 |
| `r_mgcv_gaulss` | 200.529171 | 5 | 478451.422684 | 5 | 13.674876 | 5 | 11.409718 | 5 | 0.114263 | 5 | 5.241 | 0.023 |

**Model specs**

* `python_pygam`: `LinearGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10)) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv`: `price ~ year + h_rain + w_rain + h_temp + s(s_temp, bs='ps', k=min(11, nrow(train_df)-1)) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_brms`: `brms::brm(bf(price ~ year + h_rain + w_rain + h_temp + s(s_temp, bs='ps', k=min(11, nrow(train_df)-1)), sigma ~ year + h_rain + w_rain + h_temp + s_temp); gaussian) [5-fold CV]`
* `r_mgcv_gaulss`: `gam(list(price ~ year + h_rain + w_rain + h_temp + s(s_temp, bs='ps', k=min(11, nrow(train_df)-1)), ~ year + h_rain + w_rain + h_temp + s_temp), family=gaulss()) [5-fold CV]`

**Scenario:** `wine_price_vs_temp` (gaussian)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | MSE (↓ better) | MSE rank | LogLoss (↓ better) | LogLoss rank | RMSE (↓ better) | RMSE rank | MAE (↓ better) | MAE rank | R2 (↑ better) | R2 rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `r_mgcv_gaulss` | 288.297035 | 1 | 230109.844139 | 5 | 15.855137 | 1 | 11.014221 | 1 | 0.032353 | 1 | 0.923 | 0.023 |
| `r_brms` | 303.482780 | 2 | 4.061332 | 1 | 16.366368 | 3 | 11.537832 | 2 | -0.046624 | 3 | 367.569 | 1.377 |
| `rust_gam` | 305.013104 | 3 | 4.410877 | 3 | 16.268199 | 2 | 12.022229 | 5 | -0.043587 | 2 | 0.162 | 0.052 |
| `python_pygam` | 307.903551 | 4 | 4.410590 | 2 | 16.443719 | 4 | 11.818274 | 3 | -0.053511 | 4 | 0.437 | 0.003 |
| `r_mgcv` | 310.836971 | 5 | 4.422071 | 4 | 16.544739 | 5 | 12.004471 | 4 | -0.081776 | 5 | 0.265 | 0.024 |

**Model specs**

* `r_mgcv_gaulss`: `gam(list(price ~ s(temp, bs='ps', k=min(11, nrow(train_df)-1)), ~ temp), family=gaulss()) [5-fold CV]`
* `r_brms`: `brms::brm(bf(price ~ s(temp, bs='ps', k=min(11, nrow(train_df)-1)), sigma ~ temp); gaussian) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `python_pygam`: `LinearGAM(s(0, n_splines=10)) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv`: `price ~ s(temp, bs='ps', k=min(11, nrow(train_df)-1)) [5-fold CV]`

**Scenario:** `wine_temp_vs_year` (gaussian)
**CV:** 5-fold (seed=42, leakage_safe=true)
**Run timestamp:** 2026-03-03 05:11:05 UTC (2026-03-02 23:11:05 America/Chicago)

| Contender | MSE (↓ better) | MSE rank | LogLoss (↓ better) | LogLoss rank | RMSE (↓ better) | RMSE rank | MAE (↓ better) | MAE rank | R2 (↑ better) | R2 rank | Fit (s) | Predict (s) |
| :-------------- | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | -------------: | -------: | ------: | ----------: |
| `python_pygam` | 0.461118 | 1 | 1.125791 | 1 | 0.655831 | 1 | 0.554076 | 1 | 0.413882 | 3 | 0.439 | 0.003 |
| `r_mgcv` | 0.466284 | 2 | 1.137698 | 2 | 0.657347 | 2 | 0.557487 | 2 | 0.419311 | 1 | 0.218 | 0.024 |
| `r_mgcv_gaulss` | 0.468663 | 3 | 1.491385 | 5 | 0.659421 | 3 | 0.559472 | 3 | 0.416419 | 2 | 0.509 | 0.027 |
| `rust_gam` | 0.480872 | 4 | 1.158919 | 4 | 0.667853 | 4 | 0.565580 | 4 | 0.404157 | 4 | 0.054 | 0.052 |
| `r_brms` | 0.487358 | 5 | 1.153809 | 3 | 0.674206 | 5 | 0.573355 | 5 | 0.381374 | 5 | 375.424 | 1.703 |

**Model specs**

* `python_pygam`: `LinearGAM(s(0, n_splines=10)) [lam by UBRE/GCV; 5-fold CV]`
* `r_mgcv`: `s_temp ~ s(year, bs='ps', k=min(11, nrow(train_df)-1)) [5-fold CV]`
* `r_mgcv_gaulss`: `gam(list(s_temp ~ s(year, bs='ps', k=min(11, nrow(train_df)-1)), ~ year), family=gaulss()) [5-fold CV]`
* `rust_gam`: `gam fit/predict via release binary [5-fold CV]`
* `r_brms`: `brms::brm(bf(s_temp ~ s(year, bs='ps', k=min(11, nrow(train_df)-1)), sigma ~ year); gaussian) [5-fold CV]`
