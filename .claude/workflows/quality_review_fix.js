export const meta = {
  name: 'quality-review-fix',
  description: 'Adversarially review + fix the wave-1 reference-quality tests (the review pass the rate limit skipped): make each compile, call the right mature comparator, feed identical data, and assert a principled un-weakened bound',
  phases: [{ title: 'ReviewFix', detail: 'one agent per test file: review + fix in place' }],
}

// The committed wave-1 set to review+fix (hardcoded for robustness; args
// plumbing proved unreliable).
const files = [
  "tests/quality_vs_betareg_beta_logit.rs", "tests/quality_vs_flexsurv_rp_baseline.rs",
  "tests/quality_vs_flexsurv_rp_spline.rs", "tests/quality_vs_flexsurv_weibull_aft.rs",
  "tests/quality_vs_gam_competing_risks_integral_identity.rs",
  "tests/quality_vs_gamlss_binomial_location_scale.rs", "tests/quality_vs_gamlss_gaussian_location_scale.rs",
  "tests/quality_vs_gamlss_gaussian_location_scale_by_group.rs", "tests/quality_vs_gamlss_gaussian_location_scale_cyclic.rs",
  "tests/quality_vs_gamlss_gaussian_multi_smooth.rs", "tests/quality_vs_gamlss_gaussian_survival_ls.rs",
  "tests/quality_vs_lifelines_competing_risks_cif.rs", "tests/quality_vs_lifelines_cox_like_marginal.rs",
  "tests/quality_vs_lifelines_loglogistic_aft.rs", "tests/quality_vs_lifelines_lognormal_aft.rs",
  "tests/quality_vs_lifelines_smooth_tensor_baseline.rs", "tests/quality_vs_lifelines_weibull_aft_by.rs",
  "tests/quality_vs_lme4_mgcv_random_intercept_by_smooth.rs", "tests/quality_vs_lme4_random_intercept.rs",
  "tests/quality_vs_lme4_random_slope.rs", "tests/quality_vs_mass_ordinal_polr.rs",
  "tests/quality_vs_mgcv_cyclic_cubic.rs", "tests/quality_vs_mgcv_duchon_smooth.rs",
  "tests/quality_vs_mgcv_factor_smooth_fs.rs", "tests/quality_vs_mgcv_factor_smooth_sz.rs",
  "tests/quality_vs_mgcv_gaulss_gaussian.rs", "tests/quality_vs_mgcv_gaulss_tensor.rs",
  "tests/quality_vs_mgcv_gaussian_smooth.rs", "tests/quality_vs_mgcv_matern_smooth.rs",
  "tests/quality_vs_mgcv_poisson_tensor.rs", "tests/quality_vs_mgcv_pspline_smooth.rs",
  "tests/quality_vs_mgcv_tensor_additive_tp_te.rs", "tests/quality_vs_mgcv_tensor_te_2d_binomial.rs",
  "tests/quality_vs_mgcv_tensor_te_2d_gaussian.rs", "tests/quality_vs_mgcv_tensor_te_2d_poisson.rs",
  "tests/quality_vs_mgcv_tensor_te_3d_gaussian.rs", "tests/quality_vs_mgcv_tensor_ti_2d_gaussian.rs",
  "tests/quality_vs_mgcv_tensor_tp_2d_gaussian.rs", "tests/quality_vs_mgcv_thin_plate_1d.rs",
  "tests/quality_vs_mgcv_thin_plate_by_factor.rs", "tests/quality_vs_pygam_pspline.rs",
  "tests/quality_vs_scam_monotone_baseline.rs", "tests/quality_vs_sklearn_binomial_logit.rs",
  "tests/quality_vs_sklearn_poisson_log.rs", "tests/quality_vs_statsmodels_binomial_probit.rs",
  "tests/quality_vs_statsmodels_gam_additive.rs", "tests/quality_vs_statsmodels_gamma_log.rs",
  "tests/quality_vs_statsmodels_multinomial.rs", "tests/quality_vs_statsmodels_negbin.rs",
  "tests/quality_vs_statsmodels_ordinal_mnlogit.rs", "tests/quality_vs_statsmodels_transformation_survival.rs",
  "tests/quality_vs_statsmodels_tweedie.rs", "tests/quality_vs_survival_location_scale_lognormal.rs",
  "tests/quality_vs_synthetic_multinomial_deviance_identity.rs", "tests/quality_vs_vgam_multinomial_smooth_by_factor.rs",
  "tests/quality_vs_vgam_multinomial_softmax.rs",
]
if (files.length === 0) return { error: 'no files' }

const GUIDE = `
You are reviewing and FIXING one existing gam reference-quality integration test. These were authored in a
batch whose adversarial-review pass was interrupted, so they may have compile errors, wrong/weak comparisons,
or un-justified bounds. Your job is to make this ONE file correct and high-quality, editing in place.

Read FIRST: src/test_support/reference.rs (the harness) and tests/quality_vs_mgcv_gaussian_smooth.rs (the
canonical example). Harness API: gam::test_support::reference::{Column, run_r, run_python, relative_l2, rmse,
max_abs_diff, pearson, ReferenceResult}; run_r/run_python take &[Column::new("name",&vec)] and an R/Python body
that calls emit("key", vec); read back via result.scalar/vector. A missing tool/package => the test FAILS (no skip).
gam fit API: fit_from_formula(formula, &dataset, &FitConfig) -> FitResult (match the variant; see
src/solver/workflow.rs); fit.fit.edf_total(), fit.fit.beta; build_term_collection_design for prediction.

CHECK AND FIX (with the Edit tool, in place):
  1. COMPILES: real APIs only (verify by reading the relevant src), every import USED (each integration test is
     its own crate under warnings=deny — an unused import fails the build), no banned patterns (no \`let _ =\`,
     no allow(dead_code/unused), no #[cfg(feature)], no std::env::var, no black_box silencer). panic!/println! are
     allowed in tests/. If you reference a gam API, confirm its signature in src.
  2. IDENTICAL DATA fed to gam and the comparator (same vectors / same CSV columns / same seed).
  3. CORRECT MATURE COMPARATOR: the right package and call (family/bs=/dist=/link/method), matching the gam model.
  4. RIGHT QUANTITY, ALIGNED: compares fitted function / smooth shape / EDF / coefficients / survival curve / CIF /
     CI width / posterior — element-wise or on a shared grid, in the same order.
  5. PRINCIPLED BOUND: a tolerance justified by the math in a one-line comment, neither weakened to trivially pass
     NOR so loose it asserts nothing. gam genuinely diverging and FAILING is acceptable and must NOT be hidden by
     loosening — do NOT modify gam source, only the test.
  6. Single focused #[test] fn with a module doc-comment naming the comparator and why it is the mature standard.

Do NOT run cargo / python / R (no compiling). Only Edit this file. If the file is already correct, say so.
COMMIT FREQUENTLY: after EVERY Edit, commit the file you just touched — \`git add <this exact file path>\`
then \`git commit -m 'wip: review fix'\`. Commit even if it does NOT compile yet, even mid-edit / breaking;
we want very frequent small commits (after every single edit; never more than ~5 minutes uncommitted). Stage
ONLY this file by explicit path — NEVER \`git add -A\`/\`-u\`. Retry on \`.git/index.lock\`. Do NOT push.
Reply with: the file, what you changed (or "already correct"), and any remaining concern (e.g. "expected to
diverge because ...").`

const results = await pipeline(
  files,
  (file) => agent(
    `Review and fix the reference-quality test at ${file}.\n${GUIDE}`,
    { phase: 'ReviewFix', label: `fix:${file.replace(/^tests\/quality_vs_/, '').replace(/\.rs$/, '')}` },
  ).then(text => ({ file, text })),
)

const done = results.filter(Boolean)
log(`review+fixed ${done.length}/${files.length} wave-1 tests`)
return { reviewed: done.length, files: done.map(d => d.file) }
