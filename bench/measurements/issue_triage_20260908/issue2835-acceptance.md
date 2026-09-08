# Public acceptance for #2835

All 30 public Firth seeds and all three paired plain/Firth controls pass through
`gamfit.fit` and `fit.predict`. The canonical tests are in
`tests/test_reml_rank_deficient_penalty_geometry.py`: each seed fits two
independently penalized smooths to 120 Bernoulli observations, requires finite
predictions, and requires correlation with the generating probability above
0.80. The paired controls use seeds 0, 2, and 9 and fit both Firth settings.

The successful source uses the actual likelihood/penalty union rank for the
Firth Hessian and one bounded positive cubature proposal for smoothing
uncertainty. The production changes are commits `3f9e1025d` and `30a462b62`.
Rank revelation operates on normalized roots in the same coefficient frame;
large smoothing strengths no longer erase other identifiable directions.
Resolved negative or excluded curvature remains an error.

Validation ran on MSI acn112 using the existing isolated Python 3.12.4
environment and four single-CPU lanes (112–115). Each process verified its
loaded extension SHA256 before invoking pytest. All 33 cases passed against
`d449d353e3c7cd2f0cf3fc3061a6b7085aa90b1ea61a78ede34937ce75f8eef8`.
After the combined optimizer/latent/spatial build, all seven previously failing
Firth seeds (5, 8, 9, 11, 13, 17, 20) passed again against
`7112c1fffe857fb5eaf709c93bbc6de6b899925329f287b53ef66ba506c7509c`.
The 146 public block/constrained Gaussian regressions also passed on this final
artifact in 5.50 seconds.

The adjacent `issue2835-public33-acceptance.json` records each case's process
status, duration, CPU, pytest summary, loaded artifact, and log digest. These
are CPU acceptance results; no GPU or SAE execution was performed. The
factor-smooth issue #2834 has a separate log-determinant agreement failure and
is not covered by this acceptance.
