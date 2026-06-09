# Reference-as-truth paradigm audit

**Paradigm (the bar):** a quality test must never assert that gam is *close to a
reference implementation's OUTPUT* as if the reference were ground truth.
Reference tools (mgcv, lme4, VGAM, gamlss, lifelines, geomstats, statsmodels, …)
have bugs and convention quirks — proven: geomstats' Grassmann/affine SPD
projector `metric.dist` disagrees with the analytic principal-angle / log-spectrum
closed form near the π/2 cut locus, while gam matches the analytic formula (#904).
Ground truth must be **constructed by us**: an analytic closed form, a known
data-generating process (DGP), or held-out predictive evaluation. The reference is
at most a **match-or-beat baseline scored against our self-constructed truth**.

## Classification key

- **GOOD** — asserts truth-recovery vs DGP / analytic closed form / held-out
  predictive metric; the reference (if used) is only a match-or-beat baseline on
  that self-constructed truth.
- **OFFENDER** — asserts gam's internal/output quantity ≈ a reference's
  internal/output quantity *as the truth* (canonical bad form
  `assert!((gam_x - reference_x).abs() < tol)` treating the reference value as
  correct).
- **BORDERLINE (GOOD-by-exception)** — uses a reference value, but that value is
  math ground truth the reference merely *computes* (analytic CDF/link/derivative,
  exact geometric mean / CLR, analytic posterior, exact-LOO). The reference is a
  calculator, not a fitted peer. OK.

## Headline result

The codebase already overwhelmingly follows the GOOD paradigm. The dominant
asserted pattern across the ~110 mgcv / gamlss / lme4 / statsmodels / lifelines /
INLA / pyGAM / VGAM / GP files is:

```
gam_rmse_vs_truth = rmse(gam, DGP_truth);     assert!(gam_rmse_vs_truth < bound);   // truth recovery vs DGP
ref_rmse_vs_truth = rmse(ref, DGP_truth);     // baseline computed on the SAME truth
rel_to_ref        = relative_l2(gam, ref);    // PRINTED "(context only)" — NOT asserted
                                              assert!(gam_rmse <= ref_rmse * 1.10); // match-or-beat
```

i.e. truth-recovery-vs-DGP + match-or-beat, with the gam-vs-ref cross-term
printed for context, not asserted. These are all GOOD.

**Proven OFFENDER class = "manifold library treated as geometric ground truth"**
(the #904 family). Reference-output equality is asserted at a tight tolerance
because the geodesic is "an exact closed form, so the library is mathematical
truth" — but that closed form is exactly what the test can (and should) compute
itself, and the library is provably wrong at the cut locus.

| count | classification |
|------:|----------------|
| 3 | OFFENDER (manifold-library-as-truth) — **2 converted this pass**, 1 remaining |
| ~8 | BORDERLINE / GOOD-by-exception (analytic calculator: closed geometric mean, CLR, normal CDF / PIT, sinh-arcsinh CDF + FD derivatives, exact conjugate posterior) |
| ~137 | GOOD (truth-recovery vs DGP / analytic / held-out + match-or-beat) |

## OFFENDERS

| test_file | classification | specific assertion (file:line) | how to convert (self-constructed truth that replaces the reference) |
|---|---|---|---|
| `quality_vs_geomstats_olive_oils_grassmann.rs` | OFFENDER → **CONVERTED** (commit `debfa0325`) | was `assert!(dist_diff < 1e-9)` with `dist_diff = max_abs_diff(&gam_dist, gs_dist)` (geomstats `metric.dist`); 2nd fn `assert!(dist_diff < 1e-7)` with `dist_diff = max_abs_diff(&dist_matrix, gs_dist)` | Now asserts gam geodesic distance == ANALYTIC arc length `√(Σ arccos²(σ_i))` (σ = SVD of YᵢᵀYⱼ), computed in Rust via the existing `principal_angles` helper. geomstats kept only as informational match-or-beat AWAY from the π/2 cut locus + a cross-tool witness that geomstats' OWN angle-rss matches the analytic arc length (isolating the bug to its projector `metric.dist`). |
| `quality_vs_geomstats_grassmann_exp_log_roundtrip.rs` | OFFENDER (trailing cross-check only; primary was already analytic) → **CONVERTED** (commit `c40927b95`) | was `assert!(exp_angle_diff < 1e-9)` / `assert!(log_angle_diff < 1e-9)` with `*_diff = max_abs_diff(&gam_*_angles, gs_*)` calling geomstats "mathematical truth" | Cross-check is now a match-or-beat ACCURACY baseline: geomstats scores its OWN axiom errors (endpoint angles vs σ, log spectrum vs σ, isometry) against the SAME analytic σ the test builds, and gam's error must be `<=` geomstats' (+1e-12 float slack). Both helpers (Rust + Python) hardened to the well-conditioned `atan2(sin θ, cos θ)` path (no `arccos(~1)`). Primary analytic axioms (1e-10/1e-9) untouched. |
| `quality_vs_geomstats_crabs_spd_manifold.rs` | **OFFENDER (remaining — top conversion target for next pass)** | `assert!(dist_err < 1e-8)` at L405 and L638 with `dist_err = max_abs_diff(&gam_dist, ref_dist)`; `assert!(mean_rel < 1e-6)` at L416 with `mean_rel = relative_l2(&gam_mean_flat, ref_mean)` — all vs geomstats output, called "geomstats ground truth" | Replace the distance assertion with gam's affine-invariant SPD distance == the ANALYTIC closed form `d(A,B) = √(Σ log²(λ_i))`, where λ_i are the generalized eigenvalues of (A,B) (equivalently eigenvalues of `A^{-1/2} B A^{-1/2}`), computed in Rust directly from the SPD matrices. Replace the Fréchet-mean-vs-geomstats equality with a match-or-beat on the Fréchet-variance OBJECTIVE (gam's intrinsic mean must be at least as central as geomstats', evaluated on a common metric) — exactly as the olive-oils real-data arm now does. The held-out classification accuracy assertions (L621 `gam_acc≈1`, L629 match-or-beat geomstats) are already GOOD; only the distance/mean equality gates are the offense. Same #904 cut-locus risk applies to the affine-invariant metric. |

## BORDERLINE (GOOD-by-exception — reference is a calculator of exact math)

These use a reference value at a tight tolerance, but the value is an exact
analytic quantity the reference merely evaluates (the documented "calculator"
exception). Leave as-is; optionally harden later by computing the closed form in
Rust so no library is in the truth path at all.

| test_file | the assertion (file:line) | why it is GOOD-by-exception |
|---|---|---|
| `quality_vs_scipy_johnsonsu_sas_link_transform_math.rs` | `mu_max < 1e-10` (L212), `d1/d2/d3_rel < 1e-7/1e-5/1e-3` (L224/229/234) vs scipy | The reference is the EXACT sinh-arcsinh (SAS) CDF and INDEPENDENT finite-difference derivatives — analytic math truth, not a fit. Comment says so explicitly. |
| `quality_vs_scoringrules_pit_gaussian_location_scale.rs` | `pit_max_diff < 1e-6` vs `ref_pit` (L309) | `ref_pit` is the analytic normal CDF (`scipy norm.cdf`); the bound is the A&S erf approximation floor. Exact-CDF correctness check, documented exception. |
| `quality_vs_scipy_conjugate_gaussian_posterior.rs` | `mean_abs`/`sd_abs` vs `mu_exact`/`sd_exact` (L226/229) | Compared against the EXACT closed-form conjugate Gaussian posterior mean/sd (analytic), with scipy MC draws only as a sampling-floor reference (L232/233). Self-constructed analytic truth. |
| `quality_vs_robcompositions_generalized_mean_coordinates.rs` | `center_err < 1e-10` vs `ref_center`, `clr_err_comp/rob < 1e-9`, `center_agree < 1e-9` (L236/240/474) | PRIMARY truth is gam_clr vs the DGP center `mu_clr` (self-constructed, L159-167) + match-or-beat RMSE (L249). The tight gates are the documented closed-geometric-mean / CLR "GROUND-TRUTH exception". Mild improvement: compute the closed geometric mean in Rust instead of trusting R's `mean.acomp` output, to remove the library from the truth path. |
| `quality_vs_compositions_skye_afm_aitchison_center.rs` | `clr_err < 1e-10` vs R `compositions::clr` (L340) | Primary truth is `rec_rel` vs Rust `closed_geometric_mean` (L215-216, self-constructed). `clr_err` is a cross-tool check on the ANALYTIC CLR transform. Could compute CLR in Rust to fully de-reference it. |
| `quality_vs_python_scipy_alr_geometric_mean.rs` | `coord_dev` vs `scipy_mean` is PRINTED not asserted (L288); asserts F(gam) ≤ F(scipy)·(1+tiny) and analytic-gradient ≈ 0 | GOOD: closed form `clo(exp(mean log x))` computed in Rust; scipy `gmean` is the documented exact-math baseline used only in a match-or-beat objective. |
| `quality_vs_compositions_alr_closure.rs` | truth is `closed_geometric_mean` in Rust (L160); ref used in match-or-beat Fréchet functional only | GOOD: self-constructed analytic closed form is the asserted truth. |
| `quality_vs_geomstats_poincare_*` , `quality_vs_geomstats_stiefel_exp_log_roundtrip.rs`, `quality_vs_scipy_spd_frechet_mean.rs`, `quality_vs_scipy_sphere_geodesic_consistency.rs` | primary = analytic geodesic / closed-form roundtrip; reference cross-terms printed or match-or-beat | GOOD. Poincaré was already converted (f1dc00fa7) to the analytic-truth + geomstats-match-or-beat template that the two Grassmann files now follow. Stiefel exp/log roundtrip (verified): primary = intrinsic orthonormality + analytic Gram-metric invariants; geomstats `exp_max_abs` is a PRINTED diagnostic and the only geomstats assertion is a match-or-beat on the orthonormality objective (gam's frame at least as orthonormal as geomstats') — already paradigm-compliant, no conversion needed. |

## GOOD (representative — the ~137 truth-recovery + match-or-beat tests)

All assert `gam_*_vs_truth < bound` against a DGP/analytic truth and treat the
reference only as a `gam_err <= ref_err * 1.10` match-or-beat baseline, with any
`relative_l2(gam, ref)` cross-term printed "(context only)". Spot-verified:

- mgcv smooth/tensor/manifold family: `quality_vs_mgcv_tensor_te_2d_{gaussian,poisson,beta,gamma,negbin,tweedie,binomial}.rs`, `..._te_3d_gaussian.rs`, `..._ti_2d_gaussian.rs`, `..._tp_2d_gaussian.rs`, `quality_vs_mgcv_{duchon_*,matern_*,pspline,gaussian,cyclic_cubic,nottem_cyclic,thin_plate_*,factor_smooth_*,torus_*,cylinder_*,solar_zenith_*,quakes_*,sphere_*,global_city_temp_sphere_s2,bike_sharing_torus,poisson_tensor,poisson_badhealth_*,gaulss_*,high_leverage_*,tensor_additive_tp_te}.rs`.
- gamlss location-scale: `quality_vs_gamlss_{gaussian_location_scale,gaussian_location_scale_cyclic,gaussian_location_scale_by_group,gaussian_multi_smooth,gagurine_location_scale,binomial_location_scale,crps_gaussian_location_scale,custom_family_location_scale_gaussian,gaussian_survival_ls}.rs`.
- mixed models: `quality_vs_lme4_{random_intercept,random_slope,sleepstudy_random_slope_forecast}.rs`, `quality_vs_lme4_mgcv_random_intercept_by_smooth.rs`.
- GLM families: `quality_vs_statsmodels_{negbin,tweedie,gamma_log,binomial_probit,gam_additive,multinomial,ordinal_mnlogit,custom_family_poisson_loglink,transformation_survival}.rs`, `quality_vs_sklearn_{binomial_logit,poisson_log,gp_matern_regression}.rs`, `quality_vs_betareg_beta_logit.rs`, `quality_vs_simplex_dirichlet_regression.rs`, `quality_vs_nnet_multinom_penguins_species.rs`, `quality_vs_vgam_*.rs`, `quality_vs_mass_ordinal_polr.rs`.
- survival: `quality_vs_lifelines_{lognormal_aft,loglogistic_aft,weibull_aft_by,crps_lognormal_aft,competing_risks_cif,cox_like_marginal,smooth_tensor_baseline}.rs`, `quality_vs_flexsurv_{rp_baseline,rp_spline,weibull_aft,piecewise_constant_vs_rp_baseline}.rs`, `quality_vs_rstpm2_*.rs`, `quality_vs_scam_monotone_baseline.rs`, `quality_vs_survival_*.rs`, `quality_vs_synthetic_frailty_*.rs`.
- Bayesian / GP / INLA / LOO: `quality_vs_inla_*.rs`, `quality_vs_pymc_*.rs`, `quality_vs_numpyro_nuts_poisson_loglink.rs`, `quality_vs_loo_psis_gaussian_smooth.rs`, `quality_vs_r_{gpgp_likelihood,fields_mkriging}.rs`, `quality_vs_gpyto_gp_regression.rs`.
- LOO/leverage analytic: `quality_vs_brute_force_loo_{binomial_logit,poisson_log}.rs` assert gam ALO vs EXACT brute-force LOO (self-computed) — GOOD-by-exception (exact-LOO calculator).
- SINDy: `quality_vs_pysindy_{lorenz_sparse_identification,linear_oscillator_coefficients,auto_lambda_selection_bic,threshold_convergence_stlsq_rounds,penalty_families_ridge_scad}.rs` assert recovery of the analytic governing-equation coefficient matrix with PySINDy as match-or-beat; the gam-vs-pysindy coef diff is printed context only.
- transforms / identities: `quality_vs_scipy_{boxcox_univariate_lambda,yeojohnson_skewed_positive_transformation,pit_transformation_normal,sandwich_glm_gaussian}.rs`, `quality_vs_gam_competing_risks_integral_identity.rs`, `quality_vs_synthetic_multinomial_deviance_identity.rs`, `quality_vs_hand_coded_beta_logit_negative_log_likelihood.rs` (asserts analytic NLL/Jacobian vs finite-difference — analytic exception).

## Recommended next-pass conversions (seeded for future work, not done here)

1. **`quality_vs_geomstats_crabs_spd_manifold.rs`** — the one remaining proven
   OFFENDER. Same #904 fix as Grassmann: assert gam's affine-invariant SPD
   distance vs the analytic `√(Σ log²(λ_i))` closed form computed in Rust;
   convert the Fréchet-mean-vs-geomstats equality to a match-or-beat on the
   Fréchet-variance objective.
2. De-reference the BORDERLINE composition tests (robcompositions / skye_afm) by
   computing the closed geometric mean + CLR in Rust rather than asserting tight
   equality to the R library's output — removes the library from the truth path
   even though the quantity is analytic.
3. Audit the Stiefel roundtrip's trailing geomstats cross-check; if it asserts
   gam==geomstats equality (rather than match-or-beat on the analytic axioms),
   apply the Grassmann-roundtrip template.
