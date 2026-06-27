# IDEAL crate/module org plan (post-#1521) — data-grounded

## gam-models (255k) → clean islands + 1 irreducible SCC
- CLEAN leaves (extract): transformation_normal 16k, gamlss 34k, multinomial/vector 8k
- Shared base (extract): cubic_cell_kernel/row_kernel/cell_moment_family/scale_design/sigma_link/
  inverse_link/monotone_root/wiggle/spatial_psi_bridge/gpu_kernels/block_layout/parameter_block/
  family_runtime/location_scale_engine/marginal_slope_shared ~24.6k -> gam-model-kernels
- IRREDUCIBLE SCC (SKIP): survival+bms ~128k (marginal_slope_shared back-refs ProbitFrailtyScaleJet+DeviationRuntime; survival<->bms)
- PREREQUISITE (unlocks ALL family splits): hoist ~5 fit_orchestration::drivers design helpers DOWN to gam-terms
  (build_term_collection_design, all_spatial_terms_kappa_fixed, freeze_measure_jet_length_scale_learning,
   resolve_family, response_column_kind — they operate on TermCollectionSpec = gam-terms type, mis-placed)

## gam-solve (235k) → mostly irreducible core + clean shavings
- IRREDUCIBLE SCC (SKIP): reml<->estimate<->pirls<->rho_optimizer ~150k (inner-PIRLS/outer-REML loop = one algorithm)
- EXTRACT: visualizer.rs 1.8k -> gam-cli (removes crossterm+ratatui from numeric crate); quadrature.rs 7k -> gam-math;
  custom_family 36k -> own crate (hoist ~5 shared types to gam-model-api); arrow_schur+gpu_kernels+gpu ~38k -> linalg substrate (gpu->pirls back-edge)

## within-crate clarity: gam-pyffi (40k, 17 flat files) -> submodule tree; gam-geometry manifolds/ subdir; gam-math test file -> tests/

## RANKED (benefit/risk):
1 GO-FIRST: driver helpers -> gam-terms (enabler)
2 GO: visualizer -> gam-cli   3 GO: quadrature -> gam-math   4 GO: transformation_normal crate
5 GO: gam-model-kernels base (keystone)   6 GO: gamlss crate   7 GO: vector/multinomial crate
8 GO(after models): solve custom_family crate   9 GO-if-cheap: solve arrow/gpu substrate
10 GO(independent): gam-pyffi reorg   11 nice: gam-geometry clarity
SKIP: survival/bms SCC, reml/pirls core.  DEFER: gam-fit (fit_orchestration+inference, inference<->survival cycle)

## team waves: A=#1 then handoff; B(after1)=#4,#7; C(after1)=#5 then #6; D(now,independent)=#2,#3,#10,#11; E(later)=#8,#9,#14
