# p-value calibration report

- plan: `quick`; records: 2600
- git sha: `4f17dc876bda56b4e7b0102e262f339eed90a09e`; host: `vm` (4 CPUs); jobs: 4
- versions: gamfit=0.1.268, pygam=0.12.0, pygam_gs=0.12.0
- timeout_s (per chunk) and memcap_mb (per worker) are a harness safety net, not a solver budget

## Calibration

79 rows, 413 two-sided checks (size at each level, and KS), family-wise false-alarm rate 0.001 (Bonferroni: each check at 2.4e-06).

| cell | surface | usable | size@0.10 | size@0.05 | size@0.01 | KS D (p) | power@0.05 | verdict |
|---|---|---|---|---|---|---|---|---|
| binomial/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.090 ± 0.029 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.289 (6.5e-08) | 0.300 | **NOT UNIFORM** |
| binomial/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.080 ± 0.027 | 0.070 ± 0.026 | 0.020 ± 0.014 | 0.577 (3.8e-32) | 0.160 | **NOT UNIFORM** |
| binomial/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.080 ± 0.027 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.072 (0.65) | 0.130 | calibrated |
| binomial/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.110 ± 0.031 | 0.060 ± 0.024 | 0.000 ± 0.000 | 0.106 (0.2) | 0.180 | calibrated |
| binomial/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/factor | pyGAM, fixed lam | 100/100 | 0.110 ± 0.031 | 0.040 ± 0.020 | 0.000 ± 0.000 | 0.056 (0.9) | 0.720 | calibrated |
| binomial/n=200/factor | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.074 (0.61) | 0.730 | calibrated |
| binomial/n=200/linear | gamfit coefficient | 100/100 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.748 (2.5e-58) | 0.670 | **NOT UNIFORM** |
| binomial/n=200/linear | pyGAM, fixed lam | 100/100 | 0.050 ± 0.022 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.106 (0.2) | 0.810 | calibrated |
| binomial/n=200/linear | pyGAM, gridsearch | 100/100 | 0.050 ± 0.022 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.124 (0.084) | 0.820 | calibrated |
| binomial/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| binomial/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.100 ± 0.030 | 0.050 ± 0.022 | 0.030 ± 0.017 | 0.265 (1.1e-06) | 0.700 | **NOT UNIFORM** |
| binomial/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.060 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.555 (1.5e-29) | 0.510 | **NOT UNIFORM** |
| binomial/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.103 (0.22) | 0.290 | calibrated |
| binomial/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.070 ± 0.026 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.065 (0.76) | 0.230 | calibrated |
| gamma/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.200 ± 0.040 | 0.140 ± 0.035 | 0.100 ± 0.030 | 0.331 (2.9e-10) | 0.420 | **ANTI-CONSERVATIVE** at 0.01; **NOT UNIFORM** |
| gamma/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.040 ± 0.020 | 0.020 ± 0.014 | 0.010 ± 0.010 | 0.536 (2e-27) | 0.190 | **NOT UNIFORM** |
| gamma/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.150 ± 0.036 | 0.100 ± 0.030 | 0.020 ± 0.014 | 0.123 (0.089) | 0.160 | calibrated |
| gamma/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.174 (0.0041) | 0.210 | calibrated |
| gamma/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/factor | pyGAM, fixed lam | 100/100 | 0.140 ± 0.035 | 0.080 ± 0.027 | 0.010 ± 0.010 | 0.095 (0.31) | 0.730 | calibrated |
| gamma/n=200/factor | pyGAM, gridsearch | 100/100 | 0.140 ± 0.035 | 0.070 ± 0.026 | 0.020 ± 0.014 | 0.104 (0.21) | 0.720 | calibrated |
| gamma/n=200/linear | gamfit coefficient | 100/100 | 0.040 ± 0.020 | 0.030 ± 0.017 | 0.030 ± 0.017 | 0.658 (5.5e-43) | 0.760 | **NOT UNIFORM** |
| gamma/n=200/linear | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.078 (0.55) | 0.880 | calibrated |
| gamma/n=200/linear | pyGAM, gridsearch | 100/100 | 0.080 ± 0.027 | 0.040 ± 0.020 | 0.020 ± 0.014 | 0.067 (0.74) | 0.890 | calibrated |
| gamma/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gamma/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.150 ± 0.036 | 0.120 ± 0.032 | 0.060 ± 0.024 | 0.423 (1e-16) | 0.680 | **NOT UNIFORM** |
| gamma/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.100 ± 0.030 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.508 (2e-24) | 0.520 | **NOT UNIFORM** |
| gamma/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.160 ± 0.037 | 0.070 ± 0.026 | 0.030 ± 0.017 | 0.168 (0.0061) | 0.290 | calibrated |
| gamma/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.110 ± 0.031 | 0.060 ± 0.024 | 0.000 ± 0.000 | 0.064 (0.79) | 0.270 | calibrated |
| gaussian/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.080 ± 0.027 | 0.020 ± 0.014 | 0.000 ± 0.000 | 0.435 (9.3e-18) | 0.360 | **NOT UNIFORM** |
| gaussian/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.040 ± 0.020 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.633 (2.6e-39) | 0.200 | **NOT UNIFORM** |
| gaussian/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.070 ± 0.026 | 0.020 ± 0.014 | 0.000 ± 0.000 | 0.081 (0.5) | 0.120 | calibrated |
| gaussian/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.040 ± 0.020 | 0.000 ± 0.000 | 0.083 (0.47) | 0.240 | calibrated |
| gaussian/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/factor | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.060 (0.84) | 0.730 | calibrated |
| gaussian/n=200/factor | pyGAM, gridsearch | 100/100 | 0.130 ± 0.034 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.062 (0.82) | 0.750 | calibrated |
| gaussian/n=200/linear | gamfit coefficient | 100/100 | 0.030 ± 0.017 | 0.020 ± 0.014 | 0.010 ± 0.010 | 0.690 (6.5e-48) | 0.740 | **NOT UNIFORM** |
| gaussian/n=200/linear | pyGAM, fixed lam | 100/100 | 0.080 ± 0.027 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.105 (0.2) | 0.830 | calibrated |
| gaussian/n=200/linear | pyGAM, gridsearch | 100/100 | 0.070 ± 0.026 | 0.030 ± 0.017 | 0.020 ± 0.014 | 0.052 (0.93) | 0.850 | calibrated |
| gaussian/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| gaussian/n=200/smooth | gamfit LR (`smooth_significance`) | 99/100 | 0.121 ± 0.033 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.434 (1.7e-17) | 0.670 | **NOT UNIFORM**; 1 unusable |
| gaussian/n=200/smooth | gamfit Wald (`summary`) | 99/100 | 0.061 ± 0.024 | 0.051 ± 0.022 | 0.000 ± 0.000 | 0.485 (6e-22) | 0.420 | **NOT UNIFORM**; 1 unusable |
| gaussian/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.081 (0.51) | 0.250 | calibrated |
| gaussian/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.142 (0.032) | 0.220 | calibrated |
| gaussian/n=200/ti | gamfit LR (`smooth_significance`) | 91/100 | 0.110 ± 0.033 | 0.044 ± 0.021 | 0.011 ± 0.011 | 0.257 (9e-06) | 0.418 | **ANTI-CONSERVATIVE** at 0.01; **NOT UNIFORM**; 9 unusable |
| gaussian/n=200/ti | gamfit Wald (`summary`) | 91/100 | 0.088 ± 0.030 | 0.033 ± 0.019 | 0.022 ± 0.015 | 0.370 (1e-11) | 0.187 | **ANTI-CONSERVATIVE** at 0.01; **NOT UNIFORM**; 9 unusable |
| negbin/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.100 ± 0.030 | 0.060 ± 0.024 | 0.010 ± 0.010 | 0.272 (4.8e-07) | 0.350 | **NOT UNIFORM** |
| negbin/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.070 ± 0.026 | 0.070 ± 0.026 | 0.040 ± 0.020 | 0.698 (2.8e-49) | 0.180 | **NOT UNIFORM** |
| negbin/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/linear | gamfit coefficient | 100/100 | 0.030 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.737 (3.9e-56) | 0.690 | **NOT UNIFORM** |
| negbin/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| negbin/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.140 ± 0.035 | 0.080 ± 0.027 | 0.030 ± 0.017 | 0.242 (1.3e-05) | 0.700 | calibrated |
| negbin/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.070 ± 0.026 | 0.040 ± 0.020 | 0.010 ± 0.010 | 0.606 (8.9e-36) | 0.610 | **NOT UNIFORM** |
| poisson/n=200/concurvity | gamfit LR (`smooth_significance`) | 100/100 | 0.070 ± 0.026 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.226 (5.9e-05) | 0.310 | calibrated |
| poisson/n=200/concurvity | gamfit Wald (`summary`) | 100/100 | 0.080 ± 0.027 | 0.040 ± 0.020 | 0.010 ± 0.010 | 0.578 (3.3e-32) | 0.170 | **NOT UNIFORM** |
| poisson/n=200/concurvity | pyGAM, fixed lam | 100/100 | 0.060 ± 0.024 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.067 (0.74) | 0.110 | calibrated |
| poisson/n=200/concurvity | pyGAM, gridsearch | 100/100 | 0.100 ± 0.030 | 0.050 ± 0.022 | 0.000 ± 0.000 | 0.155 (0.015) | 0.220 | calibrated |
| poisson/n=200/factor | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/factor | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/factor | pyGAM, fixed lam | 100/100 | 0.120 ± 0.032 | 0.070 ± 0.026 | 0.010 ± 0.010 | 0.112 (0.15) | 0.780 | calibrated |
| poisson/n=200/factor | pyGAM, gridsearch | 100/100 | 0.120 ± 0.032 | 0.060 ± 0.024 | 0.010 ± 0.010 | 0.065 (0.76) | 0.800 | calibrated |
| poisson/n=200/linear | gamfit coefficient | 100/100 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.000 ± 0.000 | 0.648 (1.4e-41) | 0.680 | **NOT UNIFORM** |
| poisson/n=200/linear | pyGAM, fixed lam | 100/100 | 0.100 ± 0.030 | 0.020 ± 0.014 | 0.010 ± 0.010 | 0.088 (0.39) | 0.820 | calibrated |
| poisson/n=200/linear | pyGAM, gridsearch | 100/100 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.094 (0.32) | 0.830 | calibrated |
| poisson/n=200/re | gamfit LR (`smooth_significance`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/re | gamfit Wald (`summary`) | 0/100 | - | - | - | - | - | **NO P-VALUE**; 100 unusable |
| poisson/n=200/smooth | gamfit LR (`smooth_significance`) | 100/100 | 0.120 ± 0.032 | 0.040 ± 0.020 | 0.010 ± 0.010 | 0.214 (0.00016) | 0.810 | calibrated |
| poisson/n=200/smooth | gamfit Wald (`summary`) | 100/100 | 0.070 ± 0.026 | 0.060 ± 0.024 | 0.020 ± 0.014 | 0.537 (1.6e-27) | 0.610 | **NOT UNIFORM** |
| poisson/n=200/smooth | pyGAM, fixed lam | 100/100 | 0.060 ± 0.024 | 0.030 ± 0.017 | 0.010 ± 0.010 | 0.051 (0.95) | 0.340 | calibrated |
| poisson/n=200/smooth | pyGAM, gridsearch | 100/100 | 0.050 ± 0.022 | 0.010 ± 0.010 | 0.010 ± 0.010 | 0.066 (0.75) | 0.340 | calibrated |

## Miscalibrated

- `binomial/n=200/concurvity` gamfit.lr: NOT UNIFORM: KS D 0.289 over 100 usable reps, worst-case KS p 6.5e-08 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `binomial/n=200/concurvity` gamfit.wald: NOT UNIFORM: KS D 0.577 over 100 usable reps, worst-case KS p 3.8e-32 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `binomial/n=200/linear` gamfit.coef: NOT UNIFORM: KS D 0.748 over 100 usable reps, worst-case KS p 2.5e-58 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `binomial/n=200/smooth` gamfit.lr: NOT UNIFORM: KS D 0.265 over 100 usable reps, worst-case KS p 1.1e-06 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `binomial/n=200/smooth` gamfit.wald: NOT UNIFORM: KS D 0.555 over 100 usable reps, worst-case KS p 1.5e-29 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gamma/n=200/concurvity` gamfit.lr: ANTI-CONSERVATIVE at 0.01: 10 rejections plus 0 unusable of 100 (size 0.100 over 100 usable); a calibrated p-value exceeds 8 only with the stated false-alarm probability
- `gamma/n=200/concurvity` gamfit.lr: NOT UNIFORM: KS D 0.331 over 100 usable reps, worst-case KS p 2.9e-10 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gamma/n=200/concurvity` gamfit.wald: NOT UNIFORM: KS D 0.536 over 100 usable reps, worst-case KS p 2e-27 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gamma/n=200/linear` gamfit.coef: NOT UNIFORM: KS D 0.658 over 100 usable reps, worst-case KS p 5.5e-43 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gamma/n=200/smooth` gamfit.lr: NOT UNIFORM: KS D 0.423 over 100 usable reps, worst-case KS p 1e-16 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gamma/n=200/smooth` gamfit.wald: NOT UNIFORM: KS D 0.508 over 100 usable reps, worst-case KS p 2e-24 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gaussian/n=200/concurvity` gamfit.lr: NOT UNIFORM: KS D 0.435 over 100 usable reps, worst-case KS p 9.3e-18 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gaussian/n=200/concurvity` gamfit.wald: NOT UNIFORM: KS D 0.633 over 100 usable reps, worst-case KS p 2.6e-39 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gaussian/n=200/linear` gamfit.coef: NOT UNIFORM: KS D 0.690 over 100 usable reps, worst-case KS p 6.5e-48 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gaussian/n=200/smooth` gamfit.lr: NOT UNIFORM: KS D 0.434 over 99 usable reps, worst-case KS p 1.1e-17 with 1 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gaussian/n=200/smooth` gamfit.wald: NOT UNIFORM: KS D 0.485 over 99 usable reps, worst-case KS p 1.2e-22 with 1 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gaussian/n=200/ti` gamfit.lr: ANTI-CONSERVATIVE at 0.01: 1 rejections plus 9 unusable of 100 (size 0.011 over 91 usable); a calibrated p-value exceeds 8 only with the stated false-alarm probability
- `gaussian/n=200/ti` gamfit.lr: NOT UNIFORM: KS D 0.257 over 91 usable reps, worst-case KS p 2.3e-06 with 9 unusable placed at 0 or 1; the check fires at 2.4e-06
- `gaussian/n=200/ti` gamfit.wald: ANTI-CONSERVATIVE at 0.01: 2 rejections plus 9 unusable of 100 (size 0.022 over 91 usable); a calibrated p-value exceeds 8 only with the stated false-alarm probability
- `gaussian/n=200/ti` gamfit.wald: NOT UNIFORM: KS D 0.370 over 91 usable reps, worst-case KS p 5.7e-17 with 9 unusable placed at 0 or 1; the check fires at 2.4e-06
- `negbin/n=200/concurvity` gamfit.lr: NOT UNIFORM: KS D 0.272 over 100 usable reps, worst-case KS p 4.8e-07 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `negbin/n=200/concurvity` gamfit.wald: NOT UNIFORM: KS D 0.698 over 100 usable reps, worst-case KS p 2.8e-49 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `negbin/n=200/linear` gamfit.coef: NOT UNIFORM: KS D 0.737 over 100 usable reps, worst-case KS p 3.9e-56 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `negbin/n=200/smooth` gamfit.wald: NOT UNIFORM: KS D 0.606 over 100 usable reps, worst-case KS p 8.9e-36 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `poisson/n=200/concurvity` gamfit.wald: NOT UNIFORM: KS D 0.578 over 100 usable reps, worst-case KS p 3.3e-32 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `poisson/n=200/linear` gamfit.coef: NOT UNIFORM: KS D 0.648 over 100 usable reps, worst-case KS p 1.4e-41 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06
- `poisson/n=200/smooth` gamfit.wald: NOT UNIFORM: KS D 0.537 over 100 usable reps, worst-case KS p 1.6e-27 with 0 unusable placed at 0 or 1; the check fires at 2.4e-06

## Unusable reps

- `binomial/n=200/factor` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `binomial/n=200/factor` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `binomial/n=200/factor` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `binomial/n=200/factor` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `binomial/n=200/re` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `binomial/n=200/re` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `binomial/n=200/re` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `binomial/n=200/re` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gamma/n=200/factor` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gamma/n=200/factor` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gamma/n=200/factor` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gamma/n=200/factor` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gamma/n=200/re` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gamma/n=200/re` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gamma/n=200/re` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gamma/n=200/re` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gaussian/n=200/factor` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gaussian/n=200/factor` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gaussian/n=200/factor` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gaussian/n=200/factor` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gaussian/n=200/re` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gaussian/n=200/re` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gaussian/n=200/re` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `gaussian/n=200/re` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `gaussian/n=200/smooth` null.gamfit: 1x, seeds [51]: category: convergence
- `gaussian/n=200/ti` *: 9x, seeds [0, 35, 25, 26, 58]: rep timeout
- `negbin/n=200/factor` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `negbin/n=200/factor` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `negbin/n=200/factor` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `negbin/n=200/factor` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `negbin/n=200/re` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `negbin/n=200/re` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `negbin/n=200/re` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `negbin/n=200/re` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `poisson/n=200/factor` alt.gamfit.lr: 100x, seeds [20, 21, 22, 23, 24]: LookupError: no row named 'g'; rows are ['s(x1)']
- `poisson/n=200/factor` alt.gamfit.wald: 100x, seeds [20, 21, 22, 23, 24]: p_value=None
- `poisson/n=200/factor` null.gamfit.lr: 100x, seeds [20, 21, 22, 23, 24]: LookupError: no row named 'g'; rows are ['s(x1)']
- `poisson/n=200/factor` null.gamfit.wald: 100x, seeds [20, 21, 22, 23, 24]: p_value=None
- `poisson/n=200/re` alt.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `poisson/n=200/re` alt.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
- `poisson/n=200/re` null.gamfit.lr: 100x, seeds [0, 1, 2, 3, 4]: LookupError: no row named 'g'; rows are ['s(x1)']
- `poisson/n=200/re` null.gamfit.wald: 100x, seeds [0, 1, 2, 3, 4]: p_value=None
