## Tally (gamfit perspective)

| comparison | metric | WIN | TIE | LOSS | both-fail |
|---|---|---|---|---|---|
| vs pygam_default | heldout | 9 | 31 | 15 | 0 |
| vs pygam_default | truth_mse | 20 | 3 | 15 | 0 |
| vs pygam_grid | heldout | 8 | 26 | 21 | 0 |
| vs pygam_grid | truth_mse | 12 | 10 | 16 | 0 |

## Losses

- vs pygam_default: **bike** [dev] gamfit 0.167 vs 0.1521 (+9.8%, d=0.01486+-0.0055)
- vs pygam_default: **binom_add4_n300** [truth_mse] fail gamfit=1 pygam=0
- vs pygam_default: **binom_add4_n300** [logloss] fail gamfit=1 pygam=0
- vs pygam_default: **bump2d_n4000** [truth_mse] gamfit 0.007943 vs 0.006392 (+24.3%, d=0.001551+-0.00022)
- vs pygam_default: **cake** [dev] fail gamfit=2 pygam=0
- vs pygam_default: **g1d_doppler_n2000** [truth_mse] gamfit 0.2493 vs 0.1714 (+45.5%, d=0.07791+-0.0034)
- vs pygam_default: **g1d_doppler_n2000** [dev] gamfit 0.3784 vs 0.2997 (+26.3%, d=0.07872+-0.0025)
- vs pygam_default: **g1d_doppler_n500** [truth_mse] gamfit 0.2826 vs 0.2527 (+11.8%, d=0.02982+-0.0023)
- vs pygam_default: **g1d_doppler_n500** [dev] gamfit 0.4074 vs 0.3773 (+8.0%, d=0.03005+-0.0022)
- vs pygam_default: **g1d_sin3_n10000** [truth_mse] gamfit 0.001695 vs 0.0001414 (+1098.8%, d=0.001553+-1.2e-05)
- vs pygam_default: **g1d_sin3_n10000** [dev] gamfit 0.09986 vs 0.0988 (+1.1%, d=0.001058+-0.00029)
- vs pygam_default: **g1d_sin3_n2000** [truth_mse] gamfit 0.002002 vs 0.0006013 (+233.0%, d=0.001401+-8.4e-05)
- vs pygam_default: **g1d_sin3_n2000** [dev] gamfit 0.0996 vs 0.09779 (+1.8%, d=0.001802+-0.00055)
- vs pygam_default: **g1d_sin3_n500** [truth_mse] gamfit 0.004122 vs 0.003512 (+17.4%, d=0.0006097+-0.00027)
- vs pygam_default: **g1d_sin6_n10000** [truth_mse] gamfit 0.4067 vs 0.005788 (+6926.1%, d=0.4009+-0.0036)
- vs pygam_default: **g1d_sin6_n10000** [dev] gamfit 0.5117 vs 0.1052 (+386.2%, d=0.4065+-0.004)
- vs pygam_default: **g1d_sin6_n2000** [truth_mse] gamfit 0.4061 vs 0.03918 (+936.7%, d=0.367+-0.011)
- vs pygam_default: **g1d_sin6_n2000** [dev] gamfit 0.5109 vs 0.1408 (+262.9%, d=0.3701+-0.014)
- vs pygam_default: **g1d_sin6_n500** [truth_mse] gamfit 0.4537 vs 0.1766 (+156.8%, d=0.277+-0.0063)
- vs pygam_default: **g1d_sin6_n500** [dev] gamfit 0.5257 vs 0.2613 (+101.2%, d=0.2644+-0.012)
- vs pygam_default: **haberman** [logloss] fail gamfit=2 pygam=0
- vs pygam_default: **nearsep_n200** [truth_mse] fail gamfit=2 pygam=0
- vs pygam_default: **nearsep_n200** [logloss] fail gamfit=2 pygam=0
- vs pygam_default: **outlier_n300** [truth_mse] gamfit 0.3235 vs 0.214 (+51.2%, d=0.1095+-0.054)
- vs pygam_default: **pois_add2_n2000** [truth_mse] fail gamfit=1 pygam=0
- vs pygam_default: **pois_add2_n2000** [dev] fail gamfit=1 pygam=0
- vs pygam_default: **pois_add2_n300** [truth_mse] fail gamfit=1 pygam=0
- vs pygam_default: **pois_add2_n300** [dev] fail gamfit=1 pygam=0
- vs pygam_default: **pois_lowcount_n500** [truth_mse] gamfit 0.006897 vs 0.005193 (+32.8%, d=0.001704+-0.00034)
- vs pygam_default: **wage** [dev] fail gamfit=2 pygam=0
- vs pygam_grid: **bike** [dev] gamfit 0.167 vs 0.06734 (+148.0%, d=0.09967+-0.0084)
- vs pygam_grid: **binom_add4_n300** [truth_mse] fail gamfit=1 pygam=0
- vs pygam_grid: **binom_add4_n300** [logloss] fail gamfit=1 pygam=0
- vs pygam_grid: **bump2d_n1000** [truth_mse] gamfit 0.01046 vs 0.006226 (+68.0%, d=0.004233+-0.00059)
- vs pygam_grid: **bump2d_n1000** [dev] gamfit 0.1019 vs 0.09653 (+5.6%, d=0.005411+-0.00094)
- vs pygam_grid: **bump2d_n4000** [truth_mse] gamfit 0.007943 vs 0.002295 (+246.1%, d=0.005648+-0.00026)
- vs pygam_grid: **bump2d_n4000** [dev] gamfit 0.1025 vs 0.09604 (+6.7%, d=0.006467+-0.00079)
- vs pygam_grid: **cake** [dev] fail gamfit=2 pygam=0
- vs pygam_grid: **g1d_doppler_n100** [truth_mse] gamfit 0.4025 vs 0.2877 (+39.9%, d=0.1148+-0.05)
- vs pygam_grid: **g1d_doppler_n100** [dev] gamfit 0.5243 vs 0.4049 (+29.5%, d=0.1194+-0.056)
- vs pygam_grid: **g1d_doppler_n2000** [truth_mse] gamfit 0.2493 vs 0.1523 (+63.7%, d=0.09699+-0.0038)
- vs pygam_grid: **g1d_doppler_n2000** [dev] gamfit 0.3784 vs 0.2783 (+36.0%, d=0.1001+-0.0025)
- vs pygam_grid: **g1d_doppler_n500** [truth_mse] gamfit 0.2826 vs 0.1897 (+48.9%, d=0.09284+-0.013)
- vs pygam_grid: **g1d_doppler_n500** [dev] gamfit 0.4074 vs 0.3048 (+33.7%, d=0.1026+-0.012)
- vs pygam_grid: **g1d_sin3_n10000** [truth_mse] gamfit 0.001695 vs 0.0001436 (+1080.3%, d=0.001551+-1.2e-05)
- vs pygam_grid: **g1d_sin3_n10000** [dev] gamfit 0.09986 vs 0.0988 (+1.1%, d=0.001062+-0.00029)
- vs pygam_grid: **g1d_sin3_n2000** [truth_mse] gamfit 0.002002 vs 0.0007016 (+185.4%, d=0.0013+-7.2e-05)
- vs pygam_grid: **g1d_sin3_n2000** [dev] gamfit 0.0996 vs 0.09798 (+1.6%, d=0.001616+-0.0006)
- vs pygam_grid: **g1d_sin3_n500** [truth_mse] gamfit 0.004122 vs 0.003112 (+32.5%, d=0.00101+-0.00019)
- vs pygam_grid: **g1d_sin6_n100** [truth_mse] gamfit 0.5095 vs 0.04173 (+1120.9%, d=0.4677+-0.03)
- vs pygam_grid: **g1d_sin6_n100** [dev] gamfit 0.627 vs 0.1063 (+490.0%, d=0.5207+-0.062)
- vs pygam_grid: **g1d_sin6_n10000** [truth_mse] gamfit 0.4067 vs 0.003758 (+10722.7%, d=0.4029+-0.0035)
- vs pygam_grid: **g1d_sin6_n10000** [dev] gamfit 0.5117 vs 0.1029 (+397.3%, d=0.4088+-0.0042)
- vs pygam_grid: **g1d_sin6_n2000** [truth_mse] gamfit 0.4061 vs 0.004352 (+9232.8%, d=0.4018+-0.011)
- vs pygam_grid: **g1d_sin6_n2000** [dev] gamfit 0.5109 vs 0.1057 (+383.3%, d=0.4052+-0.015)
- vs pygam_grid: **g1d_sin6_n500** [truth_mse] gamfit 0.4537 vs 0.007974 (+5589.4%, d=0.4457+-0.013)
- vs pygam_grid: **g1d_sin6_n500** [dev] gamfit 0.5257 vs 0.111 (+373.6%, d=0.4147+-0.03)
- vs pygam_grid: **gamma_add2_n2000** [dev] gamfit 0.5472 vs 0.5459 (+0.2%, d=0.001257+-0.0003)
- vs pygam_grid: **haberman** [logloss] fail gamfit=2 pygam=0
- vs pygam_grid: **head_circumference** [dev] gamfit 2.834 vs 2.771 (+2.3%, d=0.06313+-0.0095)
- vs pygam_grid: **nearsep_n200** [truth_mse] fail gamfit=2 pygam=0
- vs pygam_grid: **nearsep_n200** [logloss] fail gamfit=2 pygam=0
- vs pygam_grid: **pois_add2_n2000** [truth_mse] fail gamfit=1 pygam=0
- vs pygam_grid: **pois_add2_n2000** [dev] fail gamfit=1 pygam=0
- vs pygam_grid: **pois_add2_n300** [truth_mse] fail gamfit=1 pygam=0
- vs pygam_grid: **pois_add2_n300** [dev] fail gamfit=1 pygam=0
- vs pygam_grid: **wage** [dev] fail gamfit=2 pygam=0

### gamfit vs pygam_default

| case | n | family | metric | gamfit | pyGAM | diff (gamfit-pyGAM) +- SE | rel % | verdict | extra |
|---|---|---|---|---|---|---|---|---|---|
| add4_n1000 | 1000 | gaussian | truth_mse | 0.1049 | 0.2026 | -0.09762 +- 0.019 | -48.2 | **WIN** |  |
| add4_n1000 | 1000 | gaussian | dev | 3.883 | 4.002 | -0.1189 +- 0.038 | -3.0 | **WIN** | rmse: 1.967 vs 1.996 [WIN] |
| add4_n200 | 200 | gaussian | truth_mse | 0.49 | 1.008 | -0.5184 +- 0.09 | -51.4 | **WIN** |  |
| add4_n200 | 200 | gaussian | dev | 4.038 | 4.356 | -0.3178 +- 0.29 | -7.3 | **TIE** | rmse: 2.003 vs 2.084 [TIE] |
| add4_n5000 | 5000 | gaussian | truth_mse | 0.01441 | 0.05201 | -0.0376 +- 0.0016 | -72.3 | **WIN** |  |
| add4_n5000 | 5000 | gaussian | dev | 4.035 | 4.08 | -0.04418 +- 0.0062 | -1.1 | **WIN** | rmse: 2.009 vs 2.02 [WIN] |
| bike | 1752 | gaussian | dev | 0.167 | 0.1521 | 0.01486 +- 0.0055 | +9.8 | **LOSS** | rmse: 0.4082 vs 0.389 [LOSS] |
| binom_add4_n1000 | 1000 | binomial | truth_mse | 0.00434 | 0.009202 | -0.004862 +- 0.0009 | -52.8 | **WIN** |  |
| binom_add4_n1000 | 1000 | binomial | logloss | 0.4514 | 0.4646 | -0.01321 +- 0.0033 | -2.8 | **WIN** | brier: 0.1454 vs 0.1507 [WIN]; auc: 0.8698 vs 0.8619 [WIN] |
| binom_add4_n300 | 300 | binomial | truth_mse | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| binom_add4_n300 | 300 | binomial | logloss | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| binom_add4_n5000 | 5000 | binomial | truth_mse | 0.0006195 | 0.001957 | -0.001338 +- 9.5e-05 | -68.3 | **WIN** |  |
| binom_add4_n5000 | 5000 | binomial | logloss | 0.4224 | 0.4261 | -0.003698 +- 0.00077 | -0.9 | **WIN** | brier: 0.1377 vs 0.1392 [WIN]; auc: 0.8847 vs 0.8828 [WIN] |
| binom_sin2_n3000 | 3000 | binomial | truth_mse | 0.0008579 | 0.0009458 | -8.787e-05 +- 2.5e-05 | -9.3 | **WIN** |  |
| binom_sin2_n3000 | 3000 | binomial | logloss | 0.5123 | 0.513 | -0.0006537 +- 0.00047 | -0.1 | **TIE** | brier: 0.1671 vs 0.1672 [TIE]; auc: 0.826 vs 0.8252 [WIN] |
| binom_sin2_n500 | 500 | binomial | truth_mse | 0.003972 | 0.004176 | -0.0002038 +- 0.00016 | -4.9 | **TIE** |  |
| binom_sin2_n500 | 500 | binomial | logloss | 0.5213 | 0.5222 | -0.0009765 +- 0.00074 | -0.2 | **TIE** | brier: 0.1712 vs 0.1715 [TIE]; auc: 0.8198 vs 0.818 [WIN] |
| bump2d_n1000 | 1000 | gaussian | truth_mse | 0.01046 | 0.01869 | -0.008226 +- 0.0011 | -44.0 | **WIN** |  |
| bump2d_n1000 | 1000 | gaussian | dev | 0.1019 | 0.1083 | -0.006357 +- 0.0026 | -5.9 | **WIN** | rmse: 0.319 vs 0.3289 [WIN] |
| bump2d_n4000 | 4000 | gaussian | truth_mse | 0.007943 | 0.006392 | 0.001551 +- 0.00022 | +24.3 | **LOSS** |  |
| bump2d_n4000 | 4000 | gaussian | dev | 0.1025 | 0.1007 | 0.001815 +- 0.00095 | +1.8 | **TIE** | rmse: 0.32 vs 0.3172 [TIE] |
| cake | 270 | gaussian | dev | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| city_temp | 100 | gaussian | dev | 11.01 | 11.7 | -0.6929 +- 1 | -5.9 | **TIE** | rmse: 3.289 vs 3.394 [TIE] |
| coal | 150 | poisson | dev | 1.175 | 1.159 | 0.01639 +- 0.062 | +1.4 | **TIE** | rmse: 1.125 vs 1.114 [TIE] |
| default | 10000 | binomial | logloss | 0.07897 | 0.07985 | -0.0008776 +- 0.00056 | -1.1 | **TIE** | brier: 0.0214 vs 0.02156 [TIE]; auc: 0.9491 vs 0.9479 [WIN] |
| faithful | 200 | poisson | dev | 1.279 | 1.297 | -0.01802 +- 0.024 | -1.4 | **TIE** | rmse: 1.378 vs 1.38 [TIE] |
| g1d_doppler_n100 | 100 | gaussian | truth_mse | 0.4025 | 0.3969 | 0.0056 +- 0.013 | +1.4 | **TIE** |  |
| g1d_doppler_n100 | 100 | gaussian | dev | 0.5243 | 0.5104 | 0.01384 +- 0.015 | +2.7 | **TIE** | rmse: 0.7121 vs 0.7027 [TIE] |
| g1d_doppler_n2000 | 2000 | gaussian | truth_mse | 0.2493 | 0.1714 | 0.07791 +- 0.0034 | +45.5 | **LOSS** |  |
| g1d_doppler_n2000 | 2000 | gaussian | dev | 0.3784 | 0.2997 | 0.07872 +- 0.0025 | +26.3 | **LOSS** | rmse: 0.615 vs 0.5472 [LOSS] |
| g1d_doppler_n500 | 500 | gaussian | truth_mse | 0.2826 | 0.2527 | 0.02982 +- 0.0023 | +11.8 | **LOSS** |  |
| g1d_doppler_n500 | 500 | gaussian | dev | 0.4074 | 0.3773 | 0.03005 +- 0.0022 | +8.0 | **LOSS** | rmse: 0.637 vs 0.6127 [LOSS] |
| g1d_sin1_n100 | 100 | gaussian | truth_mse | 0.008675 | 0.01245 | -0.003776 +- 0.0017 | -30.3 | **WIN** |  |
| g1d_sin1_n100 | 100 | gaussian | dev | 0.08987 | 0.08968 | 0.0001848 +- 0.0024 | +0.2 | **TIE** | rmse: 0.296 vs 0.2962 [TIE] |
| g1d_sin1_n2000 | 2000 | gaussian | truth_mse | 0.0005457 | 0.0007238 | -0.0001781 +- 1.6e-05 | -24.6 | **WIN** |  |
| g1d_sin1_n2000 | 2000 | gaussian | dev | 0.101 | 0.1011 | -0.000144 +- 9.1e-05 | -0.1 | **TIE** | rmse: 0.3176 vs 0.3179 [TIE] |
| g1d_sin1_n500 | 500 | gaussian | truth_mse | 0.002081 | 0.002533 | -0.0004526 +- 0.00011 | -17.9 | **WIN** |  |
| g1d_sin1_n500 | 500 | gaussian | dev | 0.08785 | 0.08842 | -0.0005693 +- 0.00016 | -0.6 | **WIN** | rmse: 0.2963 vs 0.2973 [WIN] |
| g1d_sin3_n100 | 100 | gaussian | truth_mse | 0.02206 | 0.04328 | -0.02122 +- 0.0059 | -49.0 | **WIN** |  |
| g1d_sin3_n100 | 100 | gaussian | dev | 0.1376 | 0.1466 | -0.009013 +- 0.0075 | -6.1 | **TIE** | rmse: 0.368 vs 0.3807 [TIE] |
| g1d_sin3_n10000 | 10000 | gaussian | truth_mse | 0.001695 | 0.0001414 | 0.001553 +- 1.2e-05 | +1098.8 | **LOSS** |  |
| g1d_sin3_n10000 | 10000 | gaussian | dev | 0.09986 | 0.0988 | 0.001058 +- 0.00029 | +1.1 | **LOSS** | rmse: 0.3159 vs 0.3143 [LOSS] |
| g1d_sin3_n2000 | 2000 | gaussian | truth_mse | 0.002002 | 0.0006013 | 0.001401 +- 8.4e-05 | +233.0 | **LOSS** |  |
| g1d_sin3_n2000 | 2000 | gaussian | dev | 0.0996 | 0.09779 | 0.001802 +- 0.00055 | +1.8 | **LOSS** | rmse: 0.3156 vs 0.3127 [LOSS] |
| g1d_sin3_n500 | 500 | gaussian | truth_mse | 0.004122 | 0.003512 | 0.0006097 +- 0.00027 | +17.4 | **LOSS** |  |
| g1d_sin3_n500 | 500 | gaussian | dev | 0.09179 | 0.09334 | -0.001551 +- 0.00048 | -1.7 | **WIN** | rmse: 0.3029 vs 0.3055 [WIN] |
| g1d_sin6_n100 | 100 | gaussian | truth_mse | 0.5095 | 0.4546 | 0.05484 +- 0.034 | +12.1 | **TIE** |  |
| g1d_sin6_n100 | 100 | gaussian | dev | 0.627 | 0.5804 | 0.04656 +- 0.029 | +8.0 | **TIE** | rmse: 0.7894 vs 0.7557 [TIE] |
| g1d_sin6_n10000 | 10000 | gaussian | truth_mse | 0.4067 | 0.005788 | 0.4009 +- 0.0036 | +6926.1 | **LOSS** |  |
| g1d_sin6_n10000 | 10000 | gaussian | dev | 0.5117 | 0.1052 | 0.4065 +- 0.004 | +386.2 | **LOSS** | rmse: 0.7153 vs 0.3244 [LOSS] |
| g1d_sin6_n2000 | 2000 | gaussian | truth_mse | 0.4061 | 0.03918 | 0.367 +- 0.011 | +936.7 | **LOSS** |  |
| g1d_sin6_n2000 | 2000 | gaussian | dev | 0.5109 | 0.1408 | 0.3701 +- 0.014 | +262.9 | **LOSS** | rmse: 0.7144 vs 0.375 [LOSS] |
| g1d_sin6_n500 | 500 | gaussian | truth_mse | 0.4537 | 0.1766 | 0.277 +- 0.0063 | +156.8 | **LOSS** |  |
| g1d_sin6_n500 | 500 | gaussian | dev | 0.5257 | 0.2613 | 0.2644 +- 0.012 | +101.2 | **LOSS** | rmse: 0.7238 vs 0.5097 [LOSS] |
| gagurine | 314 | gaussian | dev | 0.09792 | 0.1026 | -0.004681 +- 0.0012 | -4.6 | **WIN** | rmse: 0.3118 vs 0.3193 [WIN] |
| gamma_add2_n2000 | 2000 | gamma | truth_mse | 0.0588 | 0.1799 | -0.1211 +- 0.011 | -67.3 | **WIN** |  |
| gamma_add2_n2000 | 2000 | gamma | dev | 0.5472 | 0.5497 | -0.002513 +- 0.002 | -0.5 | **TIE** | rmse: 2.647 vs 2.64 [TIE] |
| gamma_add2_n300 | 300 | gamma | truth_mse | 0.2611 | 0.6371 | -0.376 +- 0.08 | -59.0 | **WIN** |  |
| gamma_add2_n300 | 300 | gamma | dev | 0.6003 | 0.6137 | -0.01347 +- 0.013 | -2.2 | **TIE** | rmse: 3.034 vs 2.987 [TIE] |
| haberman | 306 | binomial | logloss | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| head_circumference | 7040 | gaussian | dev | 2.834 | 2.828 | 0.005628 +- 0.0037 | +0.2 | **TIE** | rmse: 1.683 vs 1.681 [TIE] |
| hepatitis | 83 | gaussian | dev | 0.01311 | 0.0124 | 0.0007059 +- 0.00051 | +5.7 | **TIE** | rmse: 0.108 vs 0.1059 [TIE] |
| hetero_n3000 | 3000 | gaussian | truth_mse | 0.001298 | 0.002293 | -0.0009948 +- 0.00029 | -43.4 | **WIN** |  |
| hetero_n3000 | 3000 | gaussian | dev | 0.3352 | 0.3345 | 0.0006885 +- 0.0012 | +0.2 | **TIE** | rmse: 0.5778 vs 0.5771 [TIE] |
| hetero_n500 | 500 | gaussian | truth_mse | 0.0116 | 0.01294 | -0.001343 +- 0.00033 | -10.4 | **WIN** |  |
| hetero_n500 | 500 | gaussian | dev | 0.3393 | 0.3409 | -0.001583 +- 0.0016 | -0.5 | **TIE** | rmse: 0.5792 vs 0.5804 [TIE] |
| lidar | 221 | gaussian | dev | 0.006779 | 0.006832 | -5.257e-05 +- 4.2e-05 | -0.8 | **TIE** | rmse: 0.08151 vs 0.08183 [TIE] |
| mcycle | 133 | gaussian | dev | 544.9 | 556.3 | -11.42 +- 15 | -2.1 | **TIE** | rmse: 23.12 vs 23.41 [TIE] |
| nearsep_n1000 | 1000 | binomial | truth_mse | 0.0007301 | 0.001437 | -0.0007073 +- 0.00013 | -49.2 | **WIN** |  |
| nearsep_n1000 | 1000 | binomial | logloss | 0.1294 | 0.1308 | -0.001363 +- 0.0022 | -1.0 | **TIE** | brier: 0.03844 vs 0.03843 [TIE]; auc: 0.9899 vs 0.9897 [TIE] |
| nearsep_n200 | 200 | binomial | truth_mse | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| nearsep_n200 | 200 | binomial | logloss | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| nottem | 240 | gaussian | dev | 5.448 | 5.547 | -0.09898 +- 0.17 | -1.8 | **TIE** | rmse: 2.327 vs 2.347 [TIE] |
| null3_n200 | 200 | gaussian | truth_mse | 0.04495 | 0.2876 | -0.2426 +- 0.025 | -84.4 | **WIN** |  |
| null3_n200 | 200 | gaussian | dev | 1.045 | 1.122 | -0.07781 +- 0.041 | -6.9 | **TIE** | rmse: 1.019 vs 1.057 [TIE] |
| null3_n2000 | 2000 | gaussian | truth_mse | 0.0006451 | 0.02285 | -0.0222 +- 0.0015 | -97.2 | **WIN** |  |
| null3_n2000 | 2000 | gaussian | dev | 1.001 | 1.019 | -0.01768 +- 0.0092 | -1.7 | **TIE** | rmse: 1 vs 1.009 [TIE] |
| outlier_n2000 | 2000 | gaussian | truth_mse | 0.06005 | 0.166 | -0.1059 +- 0.027 | -63.8 | **WIN** |  |
| outlier_n2000 | 2000 | gaussian | dev | 21.18 | 21.23 | -0.05206 +- 0.071 | -0.2 | **TIE** | rmse: 4.091 vs 4.112 [TIE] |
| outlier_n300 | 300 | gaussian | truth_mse | 0.3235 | 0.214 | 0.1095 +- 0.054 | +51.2 | **LOSS** |  |
| outlier_n300 | 300 | gaussian | dev | 5.752 | 5.719 | 0.03264 +- 0.11 | +0.6 | **TIE** | rmse: 2.171 vs 2.133 [TIE] |
| pois_add2_n10000 | 10000 | poisson | truth_mse | 0.019 | 0.03554 | -0.01654 +- 0.0021 | -46.5 | **WIN** |  |
| pois_add2_n10000 | 10000 | poisson | dev | 1.086 | 1.087 | -0.00126 +- 0.0013 | -0.1 | **TIE** | rmse: 2.176 vs 2.178 [TIE] |
| pois_add2_n2000 | 2000 | poisson | truth_mse | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_add2_n2000 | 2000 | poisson | dev | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_add2_n300 | 300 | poisson | truth_mse | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_add2_n300 | 300 | poisson | dev | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_lowcount_n500 | 500 | poisson | truth_mse | 0.006897 | 0.005193 | 0.001704 +- 0.00034 | +32.8 | **LOSS** |  |
| pois_lowcount_n500 | 500 | poisson | dev | 0.7184 | 0.7159 | 0.002483 +- 0.0074 | +0.3 | **TIE** | rmse: 0.5608 vs 0.5587 [TIE] |
| prostate_pc | 654 | binomial | logloss | 0.6146 | 0.619 | -0.004416 +- 0.0069 | -0.7 | **TIE** | brier: 0.2155 vs 0.2164 [TIE]; auc: 0.7118 vs 0.7059 [TIE] |
| sleepstudy | 180 | gaussian | dev | 1079 | 2028 | -948.8 +- 2e+02 | -46.8 | **WIN** | rmse: 32.16 vs 44.49 [WIN] |
| trees | 31 | gamma | dev | 0.007332 | 0.4456 | -0.4383 +- 0.29 | -98.4 | **TIE** | rmse: 2.824 vs 15.02 [WIN] |
| wage | 3000 | gaussian | dev | fail=2 | fail=0 | - | - | **LOSS(fail)** | |

### gamfit vs pygam_grid

| case | n | family | metric | gamfit | pyGAM | diff (gamfit-pyGAM) +- SE | rel % | verdict | extra |
|---|---|---|---|---|---|---|---|---|---|
| add4_n1000 | 1000 | gaussian | truth_mse | 0.1049 | 0.2056 | -0.1006 +- 0.015 | -49.0 | **WIN** |  |
| add4_n1000 | 1000 | gaussian | dev | 3.883 | 3.953 | -0.07003 +- 0.034 | -1.8 | **WIN** | rmse: 1.967 vs 1.985 [TIE] |
| add4_n200 | 200 | gaussian | truth_mse | 0.49 | 0.7201 | -0.2301 +- 0.052 | -31.9 | **WIN** |  |
| add4_n200 | 200 | gaussian | dev | 4.038 | 4.423 | -0.3845 +- 0.18 | -8.7 | **WIN** | rmse: 2.003 vs 2.099 [WIN] |
| add4_n5000 | 5000 | gaussian | truth_mse | 0.01441 | 0.04004 | -0.02563 +- 0.0022 | -64.0 | **WIN** |  |
| add4_n5000 | 5000 | gaussian | dev | 4.035 | 4.063 | -0.0279 +- 0.01 | -0.7 | **WIN** | rmse: 2.009 vs 2.016 [WIN] |
| bike | 1752 | gaussian | dev | 0.167 | 0.06734 | 0.09967 +- 0.0084 | +148.0 | **LOSS** | rmse: 0.4082 vs 0.2591 [LOSS] |
| binom_add4_n1000 | 1000 | binomial | truth_mse | 0.00434 | 0.007874 | -0.003534 +- 0.00061 | -44.9 | **WIN** |  |
| binom_add4_n1000 | 1000 | binomial | logloss | 0.4514 | 0.4602 | -0.008795 +- 0.0034 | -1.9 | **WIN** | brier: 0.1454 vs 0.1487 [WIN]; auc: 0.8698 vs 0.8644 [TIE] |
| binom_add4_n300 | 300 | binomial | truth_mse | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| binom_add4_n300 | 300 | binomial | logloss | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| binom_add4_n5000 | 5000 | binomial | truth_mse | 0.0006195 | 0.001541 | -0.0009217 +- 5.9e-05 | -59.8 | **WIN** |  |
| binom_add4_n5000 | 5000 | binomial | logloss | 0.4224 | 0.425 | -0.002645 +- 0.00067 | -0.6 | **WIN** | brier: 0.1377 vs 0.1386 [WIN]; auc: 0.8847 vs 0.8835 [WIN] |
| binom_sin2_n3000 | 3000 | binomial | truth_mse | 0.0008579 | 0.001025 | -0.0001668 +- 4.4e-05 | -16.3 | **WIN** |  |
| binom_sin2_n3000 | 3000 | binomial | logloss | 0.5123 | 0.5129 | -0.0005526 +- 0.00031 | -0.1 | **TIE** | brier: 0.1671 vs 0.1673 [TIE]; auc: 0.826 vs 0.8254 [TIE] |
| binom_sin2_n500 | 500 | binomial | truth_mse | 0.003972 | 0.004254 | -0.0002817 +- 0.00013 | -6.6 | **WIN** |  |
| binom_sin2_n500 | 500 | binomial | logloss | 0.5213 | 0.5222 | -0.0009641 +- 0.0013 | -0.2 | **TIE** | brier: 0.1712 vs 0.1714 [TIE]; auc: 0.8198 vs 0.8171 [TIE] |
| bump2d_n1000 | 1000 | gaussian | truth_mse | 0.01046 | 0.006226 | 0.004233 +- 0.00059 | +68.0 | **LOSS** |  |
| bump2d_n1000 | 1000 | gaussian | dev | 0.1019 | 0.09653 | 0.005411 +- 0.00094 | +5.6 | **LOSS** | rmse: 0.319 vs 0.3105 [LOSS] |
| bump2d_n4000 | 4000 | gaussian | truth_mse | 0.007943 | 0.002295 | 0.005648 +- 0.00026 | +246.1 | **LOSS** |  |
| bump2d_n4000 | 4000 | gaussian | dev | 0.1025 | 0.09604 | 0.006467 +- 0.00079 | +6.7 | **LOSS** | rmse: 0.32 vs 0.3097 [LOSS] |
| cake | 270 | gaussian | dev | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| city_temp | 100 | gaussian | dev | 11.01 | 10.52 | 0.4894 +- 0.43 | +4.7 | **TIE** | rmse: 3.289 vs 3.218 [TIE] |
| coal | 150 | poisson | dev | 1.175 | 1.131 | 0.04446 +- 0.041 | +3.9 | **TIE** | rmse: 1.125 vs 1.098 [TIE] |
| default | 10000 | binomial | logloss | 0.07897 | 0.07916 | -0.0001853 +- 0.00023 | -0.2 | **TIE** | brier: 0.0214 vs 0.02147 [TIE]; auc: 0.9491 vs 0.9491 [TIE] |
| faithful | 200 | poisson | dev | 1.279 | 1.304 | -0.02474 +- 0.019 | -1.9 | **TIE** | rmse: 1.378 vs 1.388 [TIE] |
| g1d_doppler_n100 | 100 | gaussian | truth_mse | 0.4025 | 0.2877 | 0.1148 +- 0.05 | +39.9 | **LOSS** |  |
| g1d_doppler_n100 | 100 | gaussian | dev | 0.5243 | 0.4049 | 0.1194 +- 0.056 | +29.5 | **LOSS** | rmse: 0.7121 vs 0.631 [LOSS] |
| g1d_doppler_n2000 | 2000 | gaussian | truth_mse | 0.2493 | 0.1523 | 0.09699 +- 0.0038 | +63.7 | **LOSS** |  |
| g1d_doppler_n2000 | 2000 | gaussian | dev | 0.3784 | 0.2783 | 0.1001 +- 0.0025 | +36.0 | **LOSS** | rmse: 0.615 vs 0.5274 [LOSS] |
| g1d_doppler_n500 | 500 | gaussian | truth_mse | 0.2826 | 0.1897 | 0.09284 +- 0.013 | +48.9 | **LOSS** |  |
| g1d_doppler_n500 | 500 | gaussian | dev | 0.4074 | 0.3048 | 0.1026 +- 0.012 | +33.7 | **LOSS** | rmse: 0.637 vs 0.5512 [LOSS] |
| g1d_sin1_n100 | 100 | gaussian | truth_mse | 0.008675 | 0.006247 | 0.002427 +- 0.0013 | +38.9 | **TIE** |  |
| g1d_sin1_n100 | 100 | gaussian | dev | 0.08987 | 0.08842 | 0.001444 +- 0.0024 | +1.6 | **TIE** | rmse: 0.296 vs 0.2933 [TIE] |
| g1d_sin1_n2000 | 2000 | gaussian | truth_mse | 0.0005457 | 0.0006401 | -9.445e-05 +- 6.7e-05 | -14.8 | **TIE** |  |
| g1d_sin1_n2000 | 2000 | gaussian | dev | 0.101 | 0.101 | -2.11e-05 +- 0.00025 | -0.0 | **TIE** | rmse: 0.3176 vs 0.3176 [TIE] |
| g1d_sin1_n500 | 500 | gaussian | truth_mse | 0.002081 | 0.001952 | 0.0001288 +- 9.9e-05 | +6.6 | **TIE** |  |
| g1d_sin1_n500 | 500 | gaussian | dev | 0.08785 | 0.08734 | 0.0005025 +- 0.00035 | +0.6 | **TIE** | rmse: 0.2963 vs 0.2955 [TIE] |
| g1d_sin3_n100 | 100 | gaussian | truth_mse | 0.02206 | 0.02522 | -0.00316 +- 0.0016 | -12.5 | **TIE** |  |
| g1d_sin3_n100 | 100 | gaussian | dev | 0.1376 | 0.1364 | 0.001187 +- 0.0042 | +0.9 | **TIE** | rmse: 0.368 vs 0.3668 [TIE] |
| g1d_sin3_n10000 | 10000 | gaussian | truth_mse | 0.001695 | 0.0001436 | 0.001551 +- 1.2e-05 | +1080.3 | **LOSS** |  |
| g1d_sin3_n10000 | 10000 | gaussian | dev | 0.09986 | 0.0988 | 0.001062 +- 0.00029 | +1.1 | **LOSS** | rmse: 0.3159 vs 0.3143 [LOSS] |
| g1d_sin3_n2000 | 2000 | gaussian | truth_mse | 0.002002 | 0.0007016 | 0.0013 +- 7.2e-05 | +185.4 | **LOSS** |  |
| g1d_sin3_n2000 | 2000 | gaussian | dev | 0.0996 | 0.09798 | 0.001616 +- 0.0006 | +1.6 | **LOSS** | rmse: 0.3156 vs 0.313 [LOSS] |
| g1d_sin3_n500 | 500 | gaussian | truth_mse | 0.004122 | 0.003112 | 0.00101 +- 0.00019 | +32.5 | **LOSS** |  |
| g1d_sin3_n500 | 500 | gaussian | dev | 0.09179 | 0.09235 | -0.0005591 +- 0.00064 | -0.6 | **TIE** | rmse: 0.3029 vs 0.3038 [TIE] |
| g1d_sin6_n100 | 100 | gaussian | truth_mse | 0.5095 | 0.04173 | 0.4677 +- 0.03 | +1120.9 | **LOSS** |  |
| g1d_sin6_n100 | 100 | gaussian | dev | 0.627 | 0.1063 | 0.5207 +- 0.062 | +490.0 | **LOSS** | rmse: 0.7894 vs 0.3222 [LOSS] |
| g1d_sin6_n10000 | 10000 | gaussian | truth_mse | 0.4067 | 0.003758 | 0.4029 +- 0.0035 | +10722.7 | **LOSS** |  |
| g1d_sin6_n10000 | 10000 | gaussian | dev | 0.5117 | 0.1029 | 0.4088 +- 0.0042 | +397.3 | **LOSS** | rmse: 0.7153 vs 0.3207 [LOSS] |
| g1d_sin6_n2000 | 2000 | gaussian | truth_mse | 0.4061 | 0.004352 | 0.4018 +- 0.011 | +9232.8 | **LOSS** |  |
| g1d_sin6_n2000 | 2000 | gaussian | dev | 0.5109 | 0.1057 | 0.4052 +- 0.015 | +383.3 | **LOSS** | rmse: 0.7144 vs 0.325 [LOSS] |
| g1d_sin6_n500 | 500 | gaussian | truth_mse | 0.4537 | 0.007974 | 0.4457 +- 0.013 | +5589.4 | **LOSS** |  |
| g1d_sin6_n500 | 500 | gaussian | dev | 0.5257 | 0.111 | 0.4147 +- 0.03 | +373.6 | **LOSS** | rmse: 0.7238 vs 0.3331 [LOSS] |
| gagurine | 314 | gaussian | dev | 0.09792 | 0.103 | -0.005058 +- 0.0049 | -4.9 | **TIE** | rmse: 0.3118 vs 0.3204 [TIE] |
| gamma_add2_n2000 | 2000 | gamma | truth_mse | 0.0588 | 0.09055 | -0.03175 +- 0.0056 | -35.1 | **WIN** |  |
| gamma_add2_n2000 | 2000 | gamma | dev | 0.5472 | 0.5459 | 0.001257 +- 0.0003 | +0.2 | **LOSS** | rmse: 2.647 vs 2.635 [LOSS] |
| gamma_add2_n300 | 300 | gamma | truth_mse | 0.2611 | 0.4224 | -0.1613 +- 0.063 | -38.2 | **WIN** |  |
| gamma_add2_n300 | 300 | gamma | dev | 0.6003 | 0.607 | -0.0067 +- 0.0076 | -1.1 | **TIE** | rmse: 3.034 vs 2.98 [TIE] |
| haberman | 306 | binomial | logloss | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| head_circumference | 7040 | gaussian | dev | 2.834 | 2.771 | 0.06313 +- 0.0095 | +2.3 | **LOSS** | rmse: 1.683 vs 1.664 [LOSS] |
| hepatitis | 83 | gaussian | dev | 0.01311 | 0.01321 | -9.856e-05 +- 8.8e-05 | -0.7 | **TIE** | rmse: 0.108 vs 0.1088 [TIE] |
| hetero_n3000 | 3000 | gaussian | truth_mse | 0.001298 | 0.001964 | -0.0006659 +- 0.00077 | -33.9 | **TIE** |  |
| hetero_n3000 | 3000 | gaussian | dev | 0.3352 | 0.336 | -0.0007797 +- 0.0013 | -0.2 | **TIE** | rmse: 0.5778 vs 0.5784 [TIE] |
| hetero_n500 | 500 | gaussian | truth_mse | 0.0116 | 0.01396 | -0.002364 +- 0.00085 | -16.9 | **WIN** |  |
| hetero_n500 | 500 | gaussian | dev | 0.3393 | 0.3399 | -0.0006026 +- 0.0038 | -0.2 | **TIE** | rmse: 0.5792 vs 0.5795 [TIE] |
| lidar | 221 | gaussian | dev | 0.006779 | 0.006815 | -3.59e-05 +- 1.4e-05 | -0.5 | **WIN** | rmse: 0.08151 vs 0.08175 [WIN] |
| mcycle | 133 | gaussian | dev | 544.9 | 548.1 | -3.138 +- 10 | -0.6 | **TIE** | rmse: 23.12 vs 23.21 [TIE] |
| nearsep_n1000 | 1000 | binomial | truth_mse | 0.0007301 | 0.0009052 | -0.0001751 +- 0.00019 | -19.3 | **TIE** |  |
| nearsep_n1000 | 1000 | binomial | logloss | 0.1294 | 0.1301 | -0.0006296 +- 0.0019 | -0.5 | **TIE** | brier: 0.03844 vs 0.0385 [TIE]; auc: 0.9899 vs 0.9897 [TIE] |
| nearsep_n200 | 200 | binomial | truth_mse | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| nearsep_n200 | 200 | binomial | logloss | fail=2 | fail=0 | - | - | **LOSS(fail)** | |
| nottem | 240 | gaussian | dev | 5.448 | 5.57 | -0.1221 +- 0.12 | -2.2 | **TIE** | rmse: 2.327 vs 2.353 [TIE] |
| null3_n200 | 200 | gaussian | truth_mse | 0.04495 | 0.05472 | -0.009768 +- 0.0071 | -17.9 | **TIE** |  |
| null3_n200 | 200 | gaussian | dev | 1.045 | 1.01 | 0.03433 +- 0.017 | +3.4 | **TIE** | rmse: 1.019 vs 1.002 [LOSS] |
| null3_n2000 | 2000 | gaussian | truth_mse | 0.0006451 | 0.003316 | -0.002671 +- 0.00031 | -80.5 | **WIN** |  |
| null3_n2000 | 2000 | gaussian | dev | 1.001 | 1.006 | -0.00476 +- 0.0013 | -0.5 | **WIN** | rmse: 1 vs 1.003 [WIN] |
| outlier_n2000 | 2000 | gaussian | truth_mse | 0.06005 | 0.06772 | -0.007675 +- 0.0013 | -11.3 | **WIN** |  |
| outlier_n2000 | 2000 | gaussian | dev | 21.18 | 21.19 | -0.009673 +- 0.0062 | -0.0 | **TIE** | rmse: 4.091 vs 4.092 [WIN] |
| outlier_n300 | 300 | gaussian | truth_mse | 0.3235 | 0.2611 | 0.06239 +- 0.032 | +23.9 | **TIE** |  |
| outlier_n300 | 300 | gaussian | dev | 5.752 | 5.705 | 0.0472 +- 0.049 | +0.8 | **TIE** | rmse: 2.171 vs 2.148 [TIE] |
| pois_add2_n10000 | 10000 | poisson | truth_mse | 0.019 | 0.02097 | -0.001975 +- 0.0012 | -9.4 | **TIE** |  |
| pois_add2_n10000 | 10000 | poisson | dev | 1.086 | 1.086 | -0.0004539 +- 0.0013 | -0.0 | **TIE** | rmse: 2.176 vs 2.177 [TIE] |
| pois_add2_n2000 | 2000 | poisson | truth_mse | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_add2_n2000 | 2000 | poisson | dev | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_add2_n300 | 300 | poisson | truth_mse | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_add2_n300 | 300 | poisson | dev | fail=1 | fail=0 | - | - | **LOSS(fail)** | |
| pois_lowcount_n500 | 500 | poisson | truth_mse | 0.006897 | 0.005684 | 0.001213 +- 0.00095 | +21.4 | **TIE** |  |
| pois_lowcount_n500 | 500 | poisson | dev | 0.7184 | 0.7207 | -0.002349 +- 0.0042 | -0.3 | **TIE** | rmse: 0.5608 vs 0.56 [TIE] |
| prostate_pc | 654 | binomial | logloss | 0.6146 | 0.6138 | 0.0007841 +- 0.00041 | +0.1 | **TIE** | brier: 0.2155 vs 0.2155 [TIE]; auc: 0.7118 vs 0.7106 [WIN] |
| sleepstudy | 180 | gaussian | dev | 1079 | 2106 | -1027 +- 1.4e+02 | -48.8 | **WIN** | rmse: 32.16 vs 45.59 [WIN] |
| trees | 31 | gamma | dev | 0.007332 | 0.7066 | -0.6993 +- 0.66 | -99.0 | **TIE** | rmse: 2.824 vs 10.12 [TIE] |
| wage | 3000 | gaussian | dev | fail=2 | fail=0 | - | - | **LOSS(fail)** | |

## Errors

- binom_add4_n300 fold 0 gamfit: IntegrationError: Outer smoothing-parameter optimization did not certify a stationary optimum (standard REML): gradient |g|=1.456e-3 |Pg|=1.456e-3 bound=1.343e-3 (rung=curvature-resolvability derived_standard=true) hessian_psd=yes curvature_source=terminal-analytic railed=[] → NOT STATIONARY (|Pg|=1
- cake fold 0 gamfit: IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain
- cake fold 3 gamfit: IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain
- haberman fold 2 gamfit: IntegrationError: Outer smoothing-parameter optimization did not certify a stationary optimum (standard REML): gradient |g|=7.516e-6 |Pg|=7.516e-6 bound=3.651e-6 (rung=solver-band derived_standard=false) hessian_psd=NO (cleared=true λ_min(H)=-1.675372e-6 λ_min(H+diag|g|)=1.285158e-7 max_k|g_k|=6.978
- haberman fold 4 gamfit: IntegrationError: Outer smoothing-parameter optimization did not certify a stationary optimum (standard REML): gradient |g|=6.989e-3 |Pg|=6.989e-3 bound=3.651e-6 (rung=solver-band derived_standard=false) hessian_psd=NO (cleared=false λ_min(H)=-1.351374e-6 λ_min(H+diag|g|)=-1.218216e-6 max_k|g_k|=6.5
- nearsep_n200 fold 0 gamfit: IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain; the automatic Firth/Jeffreys rescue WAS attempted and also failed to certify, so enabling Firth explicitly will not change this outcome: Outer smoothing-parameter optimization did not certify a stationary opt
- nearsep_n200 fold 2 gamfit: IntegrationError: Outer smoothing-parameter optimization did not certify a stationary optimum (standard REML): gradient |g|=8.523e-3 |Pg|=8.523e-3 bound=2.169e-4 (rung=curvature-resolvability derived_standard=true) hessian_psd=yes curvature_source=terminal-analytic railed=[] → NOT STATIONARY (|Pg|=8
- pois_add2_n2000 fold 4 gamfit: IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain
- pois_add2_n300 fold 3 gamfit: IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain
- wage fold 2 gamfit: IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain
- wage fold 3 gamfit: IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain