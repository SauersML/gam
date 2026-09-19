
## cell binom  (replicates=200)
  gamfit: 40 errors, e.g. IntegrationError('smoothing cubature has no positive-width proposal in the resolved domain; the automatic Firth/Jeffreys rescue WAS attempted and also failed to certify, so enabling Firth explicitly w
| method | mean cov | mean width | obs cov | obs width | PD cov x1 | PD cov x2 | PD cov x3 | PD whole-curve x1/x2/x3 | fit s |
|---|---|---|---|---|---|---|---|---|---|
| gamfit | 0.952 | 0.262 | 1.000 | 1.000 | 0.978 | 0.991 | 0.899 | 0.80/0.97/0.72 | 2.418 |
| pygam_default | 0.945 | 0.351 | - | - | 0.961 | 0.963 | 0.961 | 0.57/0.59/0.57 | 0.060 |
| pygam_gridsearch | 0.869 | 0.238 | - | - | 0.782 | 0.960 | 0.911 | 0.34/0.73/0.55 | 0.702 |
| gamfit_cond | 0.921 | 0.229 | - | - | - | - | - | - | - |

| method | test | P(p<.05) null s(x2) | P(p<.01) null | KS-uniform p (null) | frac p>0.99 null | power s(x3) @.05 | power s(x1) @.05 |
|---|---|---|---|---|---|---|---|
| gamfit | p_wald | 0.019 | 0.006 | 3.7e-22 | 0.39 | 0.700 | 1.000 |
  (gamfit p_lr: 78 None p-values; null-term None=0)
| gamfit | p_lr | 0.069 | 0.023 | 2.2e-07 | 0.18 | 0.798 | 1.000 |
  (gamfit p_lr_unc: 78 None p-values; null-term None=0)
| gamfit | p_lr_unc | 0.069 | 0.023 | 2.8e-08 | 0.17 | 0.798 | 1.000 |
| pygam_default | p_wald | 0.045 | 0.000 | 0.42 | 0.02 | 0.485 | 1.000 |
| pygam_gridsearch | p_wald | 0.030 | 0.000 | 0.00088 | 0.05 | 0.460 | 1.000 |

 gamfit edf s(x2): median 0.27, frac<0.05 0.41, mean 0.507; edf s(x3) median 1.98
 MCSE for coverage at .95 over nrep*60 points (indep approx) ~ 0.015 per-replicate-level
 pygam_default: edof total median 26.1; lam median 0.6; |uncentred shift x2| median 0.068
 pygam_gridsearch: edof total median 10.3; lam median 63.1; |uncentred shift x2| median 0.029
