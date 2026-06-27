#!/bin/zsh
cd /Users/user/gam
SPEC=/Users/user/gam/scratchpad/CARVE_REWRITE_SPEC.md
LOG=/Users/user/gam/scratchpad/codex_logs
run_group () {
  local name="$1"; shift
  local files="$*"
  local prompt="NO-COMPILE mechanical edit for the #1521 crate extraction. Read ${SPEC} and apply its path-rewrite rules to ONLY these files (all under crates/gam-inference/src/): ${files}. Change ONLY module-path prefixes per the spec; do NOT change any logic, do NOT add/remove/stub anything, do NOT edit imports beyond the prefix rewrites. CRITICAL: do NOT run cargo, rustc, cargo build, cargo check, or cargo test — NO COMPILE of any kind. Do not touch any file outside the listed ones. When finished, print the distinct path-prefix replacements you applied."
  codex exec "$prompt" > "${LOG}/${name}.log" 2>&1 &
}
run_group g1_hmc_io        crates/gam-inference/src/hmc_io.rs
run_group g2_sampling      crates/gam-inference/src/sample.rs crates/gam-inference/src/posterior.rs
run_group g3_polyagamma    crates/gam-inference/src/polya_gamma.rs crates/gam-inference/src/polya_gamma_core.rs crates/gam-inference/src/gpu_polya_gamma.rs
run_group g4_probability   crates/gam-inference/src/probability.rs crates/gam-inference/src/quadrature.rs crates/gam-inference/src/truncated_gaussian.rs crates/gam-inference/src/functionals.rs
run_group g5_rho_modelcmp  crates/gam-inference/src/rho_posterior.rs crates/gam-inference/src/model_comparison.rs crates/gam-inference/src/skovgaard.rs
run_group g6_geom_certs    crates/gam-inference/src/fisher_rao.rs crates/gam-inference/src/row_metric.rs crates/gam-inference/src/certificate_impls.rs crates/gam-inference/src/certificates.rs crates/gam-inference/src/marginal_slope_predict_tests.rs
run_group g7_libroot       crates/gam-inference/src/lib.rs
wait
echo "ALL_CODEX_DONE"
