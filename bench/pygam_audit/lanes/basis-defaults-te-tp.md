# basis-defaults-te-tp

TITLE: Adequate default te/1-D basis dimensions and refuse silently-flat tp smooths on outlying x
WORK ITEM: Read audit/accuracy.md section A (te/1-D basis defaults) and audit/robustness.md F2b (tp remnant). basis-size owns s() k scaling and quantile knots; the te margin default and the 1-D high-frequency case it does not test are uncovered.
Evidence: heuristic_tensor_margin_knots (crates/gam-terms/src/term_builder.rs ~4550) gives te margins of about 7, capped at unique/4, so hour with 24 levels gets 6. Results: bump2d_n4000 te truth-MSE 0.00794 vs grid 0.00230 (te k=10: 0.00222); bike te(season,hour) held-out dev 0.167 vs 0.0673 (k=[4,20]: 0.0511); g1d_sin6_n500 truth-MSE 0.4537 vs 0.00797, flat in n (0.4067 at n=1e4), 0.0078 at k=20.
basis_adequacy.rs (#2774) fires (p=7.7e-22 bump2d, 9.3e-41 sin6) but only warns and never enlarges the basis.
Robustness F2b: s(x, bs=tp) with one x=1e6 and the rest U(0,1) returns edf 1.0 and prediction -0.104 everywhere with no error.
Fix (SPEC: no magic constants, penalties on the function): derive te margin dims from data/unique counts and n as basis-size does for s(). When the adequacy test rejects, refit with a larger basis (a converged REML refit, no grid). For tp, the duchon_thinplate path should use scale-adapted knots/centres, or refuse with an explicit error naming the outlier span.
Coordinate: basis-size (same term_builder.rs; rebase onto it, keep te-only hunks separate), degenerate-smooths, shape-te-by.
Acceptance: new tests/accuracy regressions g1d_sin6 (n=500, truth-MSE < 0.02), bump2d_n4000 te (< 0.004) and bike te(season,hour) held-out dev (< 0.08). A tp-outlier test asserts edf > 1.5 on the U(0,1) bulk, or a raised error. All fail at HEAD.
