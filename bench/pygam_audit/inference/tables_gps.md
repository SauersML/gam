
## cell gauss  (replicates=200)
| method | mean cov | mean width | obs cov | obs width | PD cov x1 | PD cov x2 | PD cov x3 | PD whole-curve x1/x2/x3 | fit s |
|---|---|---|---|---|---|---|---|---|---|
| gamfit | 0.969 | 0.886 | 0.948 | 3.979 | 0.979 | 0.985 | 0.943 | 0.80/0.96/0.81 | 0.222 |
| pygam_default | 0.948 | 1.419 | 0.947 | 4.136 | 0.956 | 0.962 | 0.958 | 0.53/0.49/0.53 | 0.060 |
| pygam_gridsearch | 0.887 | 0.814 | 0.948 | 4.021 | 0.835 | 0.954 | 0.949 | 0.49/0.70/0.68 | 0.724 |
| gamfit_cond | 0.935 | 0.762 | - | - | - | - | - | - | - |

| method | test | P(p<.05) null s(x2) | P(p<.01) null | KS-uniform p (null) | frac p>0.99 null | power s(x3) @.05 | power s(x1) @.05 |
|---|---|---|---|---|---|---|---|
| gamfit | p_wald | 0.050 | 0.005 | 3.9e-35 | 0.44 | 0.540 | 1.000 |
  (gamfit p_lr: 2 None p-values; null-term None=0)
| gamfit | p_lr | 0.060 | 0.020 | 1.5e-24 | 0.07 | 0.648 | 1.000 |
  (gamfit p_lr_unc: 2 None p-values; null-term None=0)
| gamfit | p_lr_unc | 0.060 | 0.020 | 1.5e-24 | 0.07 | 0.648 | 1.000 |
| pygam_default | p_wald | 0.035 | 0.025 | 0.52 | 0.01 | 0.290 | 1.000 |
| pygam_gridsearch | p_wald | 0.035 | 0.010 | 0.015 | 0.00 | 0.300 | 1.000 |

 gamfit edf s(x2): median 0.167, frac<0.05 0.45, mean 0.488; edf s(x3) median 1.74
 MCSE for coverage at .95 over nrep*60 points (indep approx) ~ 0.015 per-replicate-level
 pygam_default: edof total median 31.1; lam median 0.6; |uncentred shift x2| median 0.047
 pygam_gridsearch: edof total median 12.4; lam median 63.1; |uncentred shift x2| median 0.020

## cell gauss_small  (replicates=200)
| method | mean cov | mean width | obs cov | obs width | PD cov x1 | PD cov x2 | PD cov x3 | PD whole-curve x1/x2/x3 | fit s |
|---|---|---|---|---|---|---|---|---|---|
| gamfit | 0.934 | 0.776 | 0.937 | 2.046 | 0.955 | 0.984 | 0.887 | 0.70/0.94/0.70 | 0.171 |
| pygam_default | 0.936 | 1.223 | 0.938 | 2.263 | 0.942 | 0.941 | 0.948 | 0.49/0.49/0.55 | 0.037 |
| pygam_gridsearch | 0.839 | 0.810 | 0.932 | 2.173 | 0.739 | 0.942 | 0.933 | 0.27/0.69/0.62 | 0.653 |
| gamfit_cond | 0.921 | 0.735 | - | - | - | - | - | - | - |

| method | test | P(p<.05) null s(x2) | P(p<.01) null | KS-uniform p (null) | frac p>0.99 null | power s(x3) @.05 | power s(x1) @.05 |
|---|---|---|---|---|---|---|---|
| gamfit | p_wald | 0.040 | 0.000 | 1.7e-19 | 0.23 | 0.525 | 1.000 |
  (gamfit p_lr: 1 None p-values; null-term None=0)
| gamfit | p_lr | 0.070 | 0.035 | 4e-40 | 0.00 | 0.724 | 1.000 |
  (gamfit p_lr_unc: 1 None p-values; null-term None=0)
| gamfit | p_lr_unc | 0.070 | 0.035 | 4e-40 | 0.00 | 0.724 | 1.000 |
| pygam_default | p_wald | 0.055 | 0.005 | 0.00027 | 0.00 | 0.270 | 1.000 |
| pygam_gridsearch | p_wald | 0.050 | 0.015 | 0.0033 | 0.00 | 0.310 | 1.000 |

 gamfit edf s(x2): median 0.475, frac<0.05 0.00, mean 0.719; edf s(x3) median 1.85
 MCSE for coverage at .95 over nrep*60 points (indep approx) ~ 0.015 per-replicate-level
 pygam_default: edof total median 23.5; lam median 0.6; |uncentred shift x2| median 0.032
 pygam_gridsearch: edof total median 12.5; lam median 15.8; |uncentred shift x2| median 0.016

## cell pois  (replicates=200)
  gamfit: 115 errors, e.g. IntegrationError('smoothing cubature has no positive-width proposal in the resolved domain')
| method | mean cov | mean width | obs cov | obs width | PD cov x1 | PD cov x2 | PD cov x3 | PD whole-curve x1/x2/x3 | fit s |
|---|---|---|---|---|---|---|---|---|---|
| gamfit | 0.978 | 1.399 | 0.990 | 5.359 | 0.985 | 0.974 | 0.980 | 0.84/0.93/0.89 | 2.517 |
| pygam_default | 0.950 | 2.445 | - | - | 0.966 | 0.964 | 0.960 | 0.51/0.51/0.49 | 0.115 |
| pygam_gridsearch | 0.862 | 1.207 | - | - | 0.745 | 0.967 | 0.949 | 0.23/0.76/0.62 | 1.156 |
| gamfit_cond | 0.952 | 1.187 | - | - | - | - | - | - | - |

| method | test | P(p<.05) null s(x2) | P(p<.01) null | KS-uniform p (null) | frac p>0.99 null | power s(x3) @.05 | power s(x1) @.05 |
|---|---|---|---|---|---|---|---|
| gamfit | p_wald | 0.035 | 0.024 | 0.071 | 0.14 | 0.694 | 1.000 |
  (gamfit p_lr: 28 None p-values; null-term None=0)
| gamfit | p_lr | 0.103 | 0.000 | 2.4e-11 | 0.09 | 0.894 | 1.000 |
  (gamfit p_lr_unc: 28 None p-values; null-term None=0)
| gamfit | p_lr_unc | 0.103 | 0.000 | 1.9e-11 | 0.09 | 0.894 | 1.000 |
| pygam_default | p_wald | 0.035 | 0.005 | 0.024 | 0.02 | 0.410 | 1.000 |
| pygam_gridsearch | p_wald | 0.055 | 0.000 | 0.012 | 0.03 | 0.475 | 1.000 |

 gamfit edf s(x2): median 0.601, frac<0.05 0.15, mean 0.713; edf s(x3) median 1.95
 MCSE for coverage at .95 over nrep*60 points (indep approx) ~ 0.015 per-replicate-level
 pygam_default: edof total median 34.2; lam median 0.6; |uncentred shift x2| median 0.042
 pygam_gridsearch: edof total median 10.7; lam median 251; |uncentred shift x2| median 0.025
