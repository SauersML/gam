//! Grouped integration-test crate root (issue #1146).
//!
//! Each formerly-top-level `tests/quality_vs_*.rs` crate is included here as a
//! module so the 150 quality-vs-reference tests link as ONE binary instead of
//! 150 separate linker invocations. Add new quality_vs_* tests as a module here.

#[path = "quality/quality_vs_betareg_beta_logit.rs"]
mod quality_vs_betareg_beta_logit;
#[path = "quality/quality_vs_brute_force_loo_binomial_logit.rs"]
mod quality_vs_brute_force_loo_binomial_logit;
#[path = "quality/quality_vs_brute_force_loo_poisson_log.rs"]
mod quality_vs_brute_force_loo_poisson_log;
#[path = "quality/quality_vs_compositions_alr_closure.rs"]
mod quality_vs_compositions_alr_closure;
#[path = "quality/quality_vs_compositions_frechet_mean.rs"]
mod quality_vs_compositions_frechet_mean;
#[path = "quality/quality_vs_compositions_skye_afm_aitchison_center.rs"]
mod quality_vs_compositions_skye_afm_aitchison_center;
#[path = "quality/quality_vs_flexsurv_piecewise_constant_vs_rp_baseline.rs"]
mod quality_vs_flexsurv_piecewise_constant_vs_rp_baseline;
#[path = "quality/quality_vs_flexsurv_rp_baseline.rs"]
mod quality_vs_flexsurv_rp_baseline;
#[path = "quality/quality_vs_flexsurv_rp_spline.rs"]
mod quality_vs_flexsurv_rp_spline;
#[path = "quality/quality_vs_flexsurv_weibull_aft.rs"]
mod quality_vs_flexsurv_weibull_aft;
#[path = "quality/quality_vs_gam_competing_risks_integral_identity.rs"]
mod quality_vs_gam_competing_risks_integral_identity;
#[path = "quality/quality_vs_gamlss_beta_dispersion_location_scale_1060.rs"]
mod quality_vs_gamlss_beta_dispersion_location_scale_1060;
#[path = "quality/quality_vs_gamlss_binomial_location_scale.rs"]
mod quality_vs_gamlss_binomial_location_scale;
#[path = "quality/quality_vs_gamlss_crps_gaussian_location_scale.rs"]
mod quality_vs_gamlss_crps_gaussian_location_scale;
#[path = "quality/quality_vs_gamlss_custom_family_location_scale_gaussian.rs"]
mod quality_vs_gamlss_custom_family_location_scale_gaussian;
#[path = "quality/quality_vs_gamlss_gagurine_location_scale.rs"]
mod quality_vs_gamlss_gagurine_location_scale;
#[path = "quality/quality_vs_gamlss_gamma_dispersion_location_scale_1060.rs"]
mod quality_vs_gamlss_gamma_dispersion_location_scale_1060;
#[path = "quality/quality_vs_gamlss_gaussian_location_scale.rs"]
mod quality_vs_gamlss_gaussian_location_scale;
#[path = "quality/quality_vs_gamlss_gaussian_location_scale_by_group.rs"]
mod quality_vs_gamlss_gaussian_location_scale_by_group;
#[path = "quality/quality_vs_gamlss_gaussian_location_scale_cyclic.rs"]
mod quality_vs_gamlss_gaussian_location_scale_cyclic;
#[path = "quality/quality_vs_gamlss_gaussian_multi_smooth.rs"]
mod quality_vs_gamlss_gaussian_multi_smooth;
#[path = "quality/quality_vs_gamlss_gaussian_survival_ls.rs"]
mod quality_vs_gamlss_gaussian_survival_ls;
#[path = "quality/quality_vs_gamlss_negbin_dispersion_location_scale_1060.rs"]
mod quality_vs_gamlss_negbin_dispersion_location_scale_1060;
#[path = "quality/quality_vs_geomstats_crabs_spd_manifold.rs"]
mod quality_vs_geomstats_crabs_spd_manifold;
#[path = "quality/quality_vs_geomstats_grassmann_exp_log_roundtrip.rs"]
mod quality_vs_geomstats_grassmann_exp_log_roundtrip;
#[path = "quality/quality_vs_geomstats_olive_oils_grassmann.rs"]
mod quality_vs_geomstats_olive_oils_grassmann;
#[path = "quality/quality_vs_geomstats_poincare_exp_log_roundtrip.rs"]
mod quality_vs_geomstats_poincare_exp_log_roundtrip;
#[path = "quality/quality_vs_geomstats_poincare_geodesic.rs"]
mod quality_vs_geomstats_poincare_geodesic;
#[path = "quality/quality_vs_geomstats_stiefel_exp_log_roundtrip.rs"]
mod quality_vs_geomstats_stiefel_exp_log_roundtrip;
#[path = "quality/quality_vs_gpyto_gp_regression.rs"]
mod quality_vs_gpyto_gp_regression;
#[path = "quality/quality_vs_hand_coded_beta_logit_negative_log_likelihood.rs"]
mod quality_vs_hand_coded_beta_logit_negative_log_likelihood;
#[path = "quality/quality_vs_inla_binomial_smooth_probability.rs"]
mod quality_vs_inla_binomial_smooth_probability;
#[path = "quality/quality_vs_inla_gaussian_smooth_posterior_mean.rs"]
mod quality_vs_inla_gaussian_smooth_posterior_mean;
#[path = "quality/quality_vs_inla_rw2_spde_penalized_baseline.rs"]
mod quality_vs_inla_rw2_spde_penalized_baseline;
#[path = "quality/quality_vs_inla_smooth_posterior_sd_credible_interval.rs"]
mod quality_vs_inla_smooth_posterior_sd_credible_interval;
#[path = "quality/quality_vs_inla_survival_random_intercept_baseline.rs"]
mod quality_vs_inla_survival_random_intercept_baseline;
#[path = "quality/quality_vs_inla_tensor_product_spde.rs"]
mod quality_vs_inla_tensor_product_spde;
#[path = "quality/quality_vs_interpretML_pygam_binomial_logit_comparative.rs"]
mod quality_vs_interpretML_pygam_binomial_logit_comparative;
#[path = "quality/quality_vs_interpretml_ebm_binomial_logit.rs"]
mod quality_vs_interpretml_ebm_binomial_logit;
#[path = "quality/quality_vs_interpretml_ebm_poisson_log.rs"]
mod quality_vs_interpretml_ebm_poisson_log;
#[path = "quality/quality_vs_lifelines_competing_risks_cif.rs"]
mod quality_vs_lifelines_competing_risks_cif;
#[path = "quality/quality_vs_lifelines_cox_like_marginal.rs"]
mod quality_vs_lifelines_cox_like_marginal;
#[path = "quality/quality_vs_lifelines_crps_lognormal_aft.rs"]
mod quality_vs_lifelines_crps_lognormal_aft;
#[path = "quality/quality_vs_lifelines_interval_censored_truth_recovery.rs"]
mod quality_vs_lifelines_interval_censored_truth_recovery;
#[path = "quality/quality_vs_lifelines_loglogistic_aft.rs"]
mod quality_vs_lifelines_loglogistic_aft;
#[path = "quality/quality_vs_lifelines_lognormal_aft.rs"]
mod quality_vs_lifelines_lognormal_aft;
#[path = "quality/quality_vs_lifelines_rmst_truth_recovery.rs"]
mod quality_vs_lifelines_rmst_truth_recovery;
#[path = "quality/quality_vs_lifelines_smooth_tensor_baseline.rs"]
mod quality_vs_lifelines_smooth_tensor_baseline;
#[path = "quality/quality_vs_lifelines_weibull_aft_by.rs"]
mod quality_vs_lifelines_weibull_aft_by;
#[path = "quality/quality_vs_lme4_mgcv_random_intercept_by_smooth.rs"]
mod quality_vs_lme4_mgcv_random_intercept_by_smooth;
#[path = "quality/quality_vs_lme4_random_intercept.rs"]
mod quality_vs_lme4_random_intercept;
#[path = "quality/quality_vs_lme4_random_slope.rs"]
mod quality_vs_lme4_random_slope;
#[path = "quality/quality_vs_lme4_sleepstudy_random_slope_forecast.rs"]
mod quality_vs_lme4_sleepstudy_random_slope_forecast;
#[path = "quality/quality_vs_loo_psis_gaussian_smooth.rs"]
mod quality_vs_loo_psis_gaussian_smooth;
#[path = "quality/quality_vs_manual_mixture_link_logistic_probit_cloglog_blend.rs"]
mod quality_vs_manual_mixture_link_logistic_probit_cloglog_blend;
#[path = "quality/quality_vs_mass_ordinal_polr.rs"]
mod quality_vs_mass_ordinal_polr;
#[path = "quality/quality_vs_mgcv_bike_sharing_torus.rs"]
mod quality_vs_mgcv_bike_sharing_torus;
#[path = "quality/quality_vs_mgcv_confidence_interval_gaussian_coverage_90.rs"]
mod quality_vs_mgcv_confidence_interval_gaussian_coverage_90;
#[path = "quality/quality_vs_mgcv_confidence_interval_gaussian_logistic_link.rs"]
mod quality_vs_mgcv_confidence_interval_gaussian_logistic_link;
#[path = "quality/quality_vs_mgcv_confidence_interval_gaussian_response_scale.rs"]
mod quality_vs_mgcv_confidence_interval_gaussian_response_scale;
#[path = "quality/quality_vs_mgcv_confidence_interval_gaussian_se.rs"]
mod quality_vs_mgcv_confidence_interval_gaussian_se;
#[path = "quality/quality_vs_mgcv_confidence_interval_gaussian_sweep_coverage.rs"]
mod quality_vs_mgcv_confidence_interval_gaussian_sweep_coverage;
#[path = "quality/quality_vs_mgcv_cyclic_cubic.rs"]
mod quality_vs_mgcv_cyclic_cubic;
#[path = "quality/quality_vs_mgcv_cylinder_tensor_cc_ps.rs"]
mod quality_vs_mgcv_cylinder_tensor_cc_ps;
#[path = "quality/quality_vs_mgcv_duchon_2d.rs"]
mod quality_vs_mgcv_duchon_2d;
#[path = "quality/quality_vs_mgcv_duchon_regimes.rs"]
mod quality_vs_mgcv_duchon_regimes;
#[path = "quality/quality_vs_mgcv_duchon_smooth.rs"]
mod quality_vs_mgcv_duchon_smooth;
#[path = "quality/quality_vs_mgcv_factor_smooth_fs.rs"]
mod quality_vs_mgcv_factor_smooth_fs;
#[path = "quality/quality_vs_mgcv_factor_smooth_sz.rs"]
mod quality_vs_mgcv_factor_smooth_sz;
#[path = "quality/quality_vs_mgcv_gaulss_gaussian.rs"]
mod quality_vs_mgcv_gaulss_gaussian;
#[path = "quality/quality_vs_mgcv_gaulss_tensor.rs"]
mod quality_vs_mgcv_gaulss_tensor;
#[path = "quality/quality_vs_mgcv_gaussian_smooth.rs"]
mod quality_vs_mgcv_gaussian_smooth;
#[path = "quality/quality_vs_mgcv_global_city_temp_sphere_s2.rs"]
mod quality_vs_mgcv_global_city_temp_sphere_s2;
#[path = "quality/quality_vs_mgcv_grid_spline_2d_truth_recovery.rs"]
mod quality_vs_mgcv_grid_spline_2d_truth_recovery;
#[path = "quality/quality_vs_mgcv_high_leverage_gaussian_alo_stabilized.rs"]
mod quality_vs_mgcv_high_leverage_gaussian_alo_stabilized;
#[path = "quality/quality_vs_mgcv_matern_smooth.rs"]
mod quality_vs_mgcv_matern_smooth;
#[path = "quality/quality_vs_mgcv_matern_varying_nu.rs"]
mod quality_vs_mgcv_matern_varying_nu;
#[path = "quality/quality_vs_mgcv_nottem_cyclic.rs"]
mod quality_vs_mgcv_nottem_cyclic;
#[path = "quality/quality_vs_mgcv_pair_surface_live_backend.rs"]
mod quality_vs_mgcv_pair_surface_live_backend;
#[path = "quality/quality_vs_mgcv_poisson_badhealth_doctor_visits.rs"]
mod quality_vs_mgcv_poisson_badhealth_doctor_visits;
#[path = "quality/quality_vs_mgcv_poisson_tensor.rs"]
mod quality_vs_mgcv_poisson_tensor;
#[path = "quality/quality_vs_mgcv_pspline_smooth.rs"]
mod quality_vs_mgcv_pspline_smooth;
#[path = "quality/quality_vs_mgcv_quakes_spatial_smooth.rs"]
mod quality_vs_mgcv_quakes_spatial_smooth;
#[path = "quality/quality_vs_mgcv_solar_zenith_cylinder.rs"]
mod quality_vs_mgcv_solar_zenith_cylinder;
#[path = "quality/quality_vs_mgcv_sphere_rotation_equivariance_s2.rs"]
mod quality_vs_mgcv_sphere_rotation_equivariance_s2;
#[path = "quality/quality_vs_mgcv_sphere_s2_wahba_vs_sos.rs"]
mod quality_vs_mgcv_sphere_s2_wahba_vs_sos;
#[path = "quality/quality_vs_mgcv_tensor_additive_tp_te.rs"]
mod quality_vs_mgcv_tensor_additive_tp_te;
#[path = "quality/quality_vs_mgcv_tensor_te_2d_beta.rs"]
mod quality_vs_mgcv_tensor_te_2d_beta;
#[path = "quality/quality_vs_mgcv_tensor_te_2d_binomial.rs"]
mod quality_vs_mgcv_tensor_te_2d_binomial;
#[path = "quality/quality_vs_mgcv_tensor_te_2d_gamma.rs"]
mod quality_vs_mgcv_tensor_te_2d_gamma;
#[path = "quality/quality_vs_mgcv_tensor_te_2d_gaussian.rs"]
mod quality_vs_mgcv_tensor_te_2d_gaussian;
#[path = "quality/quality_vs_mgcv_tensor_te_2d_negbin.rs"]
mod quality_vs_mgcv_tensor_te_2d_negbin;
#[path = "quality/quality_vs_mgcv_tensor_te_2d_poisson.rs"]
mod quality_vs_mgcv_tensor_te_2d_poisson;
#[path = "quality/quality_vs_mgcv_tensor_te_2d_tweedie.rs"]
mod quality_vs_mgcv_tensor_te_2d_tweedie;
#[path = "quality/quality_vs_mgcv_tensor_te_3d_gaussian.rs"]
mod quality_vs_mgcv_tensor_te_3d_gaussian;
#[path = "quality/quality_vs_mgcv_tensor_ti_2d_gaussian.rs"]
mod quality_vs_mgcv_tensor_ti_2d_gaussian;
#[path = "quality/quality_vs_mgcv_tensor_tp_2d_gaussian.rs"]
mod quality_vs_mgcv_tensor_tp_2d_gaussian;
#[path = "quality/quality_vs_mgcv_thin_plate_1d.rs"]
mod quality_vs_mgcv_thin_plate_1d;
#[path = "quality/quality_vs_mgcv_thin_plate_by_factor.rs"]
mod quality_vs_mgcv_thin_plate_by_factor;
#[path = "quality/quality_vs_mgcv_torus_tensor_cc_cc.rs"]
mod quality_vs_mgcv_torus_tensor_cc_cc;
#[path = "quality/quality_vs_nnet_multinom_penguins_species.rs"]
mod quality_vs_nnet_multinom_penguins_species;
#[path = "quality/quality_vs_numpyro_nuts_poisson_loglink.rs"]
mod quality_vs_numpyro_nuts_poisson_loglink;
#[path = "quality/quality_vs_properscoring_crps_gaussian_location_scale.rs"]
mod quality_vs_properscoring_crps_gaussian_location_scale;
#[path = "quality/quality_vs_pygam_logistic_1d_shape.rs"]
mod quality_vs_pygam_logistic_1d_shape;
#[path = "quality/quality_vs_pygam_poisson_2d_shape.rs"]
mod quality_vs_pygam_poisson_2d_shape;
#[path = "quality/quality_vs_pygam_pspline.rs"]
mod quality_vs_pygam_pspline;
#[path = "quality/quality_vs_pymc_hmc_binomial_penalized_vs_unpenalized.rs"]
mod quality_vs_pymc_hmc_binomial_penalized_vs_unpenalized;
#[path = "quality/quality_vs_pymc_nuts_binomial_logit.rs"]
mod quality_vs_pymc_nuts_binomial_logit;
#[path = "quality/quality_vs_python_scipy_alr_geometric_mean.rs"]
mod quality_vs_python_scipy_alr_geometric_mean;
#[path = "quality/quality_vs_r_fields_mkriging.rs"]
mod quality_vs_r_fields_mkriging;
#[path = "quality/quality_vs_r_gpgp_likelihood.rs"]
mod quality_vs_r_gpgp_likelihood;
#[path = "quality/quality_vs_r_mlt_conditional_transformation_binary_covariate.rs"]
mod quality_vs_r_mlt_conditional_transformation_binary_covariate;
#[path = "quality/quality_vs_r_tram_smooth_continuous_covariate_transformation.rs"]
mod quality_vs_r_tram_smooth_continuous_covariate_transformation;
#[path = "quality/quality_vs_robcompositions_generalized_mean_coordinates.rs"]
mod quality_vs_robcompositions_generalized_mean_coordinates;
#[path = "quality/quality_vs_rstpm2_pstpm2_monotone_baseline_constraint.rs"]
mod quality_vs_rstpm2_pstpm2_monotone_baseline_constraint;
#[path = "quality/quality_vs_rstpm2_pstpm2_penalized_baseline.rs"]
mod quality_vs_rstpm2_pstpm2_penalized_baseline;
#[path = "quality/quality_vs_rstpm2_pstpm2_smooth_covariate.rs"]
mod quality_vs_rstpm2_pstpm2_smooth_covariate;
#[path = "quality/quality_vs_scam_monotone_baseline.rs"]
mod quality_vs_scam_monotone_baseline;
#[path = "quality/quality_vs_scipy_boxcox_univariate_lambda.rs"]
mod quality_vs_scipy_boxcox_univariate_lambda;
#[path = "quality/quality_vs_scipy_conjugate_gaussian_posterior.rs"]
mod quality_vs_scipy_conjugate_gaussian_posterior;
#[path = "quality/quality_vs_scipy_johnsonsu_sas_link_transform_math.rs"]
mod quality_vs_scipy_johnsonsu_sas_link_transform_math;
#[path = "quality/quality_vs_scipy_pit_transformation_normal.rs"]
mod quality_vs_scipy_pit_transformation_normal;
#[path = "quality/quality_vs_scipy_sandwich_glm_gaussian.rs"]
mod quality_vs_scipy_sandwich_glm_gaussian;
#[path = "quality/quality_vs_scipy_spd_frechet_mean.rs"]
mod quality_vs_scipy_spd_frechet_mean;
#[path = "quality/quality_vs_scipy_sphere_geodesic_consistency.rs"]
mod quality_vs_scipy_sphere_geodesic_consistency;
#[path = "quality/quality_vs_scipy_yeojohnson_skewed_positive_transformation.rs"]
mod quality_vs_scipy_yeojohnson_skewed_positive_transformation;
#[path = "quality/quality_vs_scoringrules_pit_gaussian_location_scale.rs"]
mod quality_vs_scoringrules_pit_gaussian_location_scale;
#[path = "quality/quality_vs_simplex_dirichlet_regression.rs"]
mod quality_vs_simplex_dirichlet_regression;
#[path = "quality/quality_vs_sklearn_binomial_logit.rs"]
mod quality_vs_sklearn_binomial_logit;
#[path = "quality/quality_vs_sklearn_gp_matern_regression.rs"]
mod quality_vs_sklearn_gp_matern_regression;
#[path = "quality/quality_vs_sklearn_poisson_log.rs"]
mod quality_vs_sklearn_poisson_log;
#[path = "quality/quality_vs_statsmodels_binomial_probit.rs"]
mod quality_vs_statsmodels_binomial_probit;
#[path = "quality/quality_vs_statsmodels_custom_family_poisson_loglink.rs"]
mod quality_vs_statsmodels_custom_family_poisson_loglink;
#[path = "quality/quality_vs_statsmodels_gam_additive.rs"]
mod quality_vs_statsmodels_gam_additive;
#[path = "quality/quality_vs_statsmodels_gamma_log.rs"]
mod quality_vs_statsmodels_gamma_log;
#[path = "quality/quality_vs_statsmodels_gamma_log_coefficient_se.rs"]
mod quality_vs_statsmodels_gamma_log_coefficient_se;
#[path = "quality/quality_vs_statsmodels_multinomial.rs"]
mod quality_vs_statsmodels_multinomial;
#[path = "quality/quality_vs_statsmodels_negbin.rs"]
mod quality_vs_statsmodels_negbin;
#[path = "quality/quality_vs_statsmodels_negbin_coefficient_se.rs"]
mod quality_vs_statsmodels_negbin_coefficient_se;
#[path = "quality/quality_vs_statsmodels_ordinal_mnlogit.rs"]
mod quality_vs_statsmodels_ordinal_mnlogit;
#[path = "quality/quality_vs_statsmodels_transformation_survival.rs"]
mod quality_vs_statsmodels_transformation_survival;
#[path = "quality/quality_vs_statsmodels_tweedie.rs"]
mod quality_vs_statsmodels_tweedie;
#[path = "quality/quality_vs_survival_coxph_frailty_hazard_multiplier.rs"]
mod quality_vs_survival_coxph_frailty_hazard_multiplier;
#[path = "quality/quality_vs_survival_location_scale_lognormal.rs"]
mod quality_vs_survival_location_scale_lognormal;
#[path = "quality/quality_vs_survival_weibull_veteran_lung_aft.rs"]
mod quality_vs_survival_weibull_veteran_lung_aft;
#[path = "quality/quality_vs_synthetic_frailty_hazard_multiplier_likelihood.rs"]
mod quality_vs_synthetic_frailty_hazard_multiplier_likelihood;
#[path = "quality/quality_vs_synthetic_multinomial_deviance_identity.rs"]
mod quality_vs_synthetic_multinomial_deviance_identity;
#[path = "quality/quality_vs_vgam_beta_logistic_link_parameterization.rs"]
mod quality_vs_vgam_beta_logistic_link_parameterization;
#[path = "quality/quality_vs_vgam_multinomial_smooth_by_factor.rs"]
mod quality_vs_vgam_multinomial_smooth_by_factor;
#[path = "quality/quality_vs_vgam_multinomial_softmax.rs"]
mod quality_vs_vgam_multinomial_softmax;
