//! Integration-test harness for gam-models: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-models and
//! its dependency tree. One binary, same tests, same names.

mod adaptive_basis_resolution_3078;
mod bms_generated_regressor_covariance_2943;
mod bms_latent_conditional_residual_3016;
mod bms_learned_sigma_finite_law_3059;
mod bms_per_smooth_summary_2997;
mod bms_route_arming_3164;
mod bspline_nonzero_anchor_affine_2297;
mod duchon_grid_fit_and_rotation_2319;
mod exact_gaussian_boundary_2663;
mod multinomial_contracted_jeffreys_2612;
mod multinomial_covariance_mode_2612;
mod multinomial_dominated_face_2627;
mod multinomial_lambda_selection_561;
mod multinomial_parametric_penalty_2612;
mod multinomial_payload_matches_the_fit_2612;
mod multinomial_predictive_ratio_2612;
mod multinomial_separation_arming_2612;
mod null_rail_outer_certify;
mod production_row_program_policy;
mod standard_reml_outer_search_2817;
mod two_level_label_response_and_row_floor;
mod warm_start_from;
mod weibull_survival_summary_3297;
mod binomial_outer_certificate_3305;
