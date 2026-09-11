//! Integration-test harness for gam-models: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-models and
//! its dependency tree. One binary, same tests, same names.

mod bspline_nonzero_anchor_affine_2297;
mod duchon_grid_fit_and_rotation_2319;
mod exact_gaussian_boundary_2663;
mod multinomial_contracted_jeffreys_2612;
mod multinomial_covariance_mode_2612;
mod multinomial_jeffreys_outer_gradient_fd_2612;
mod multinomial_lambda_selection_561;
mod multinomial_parametric_penalty_2612;
mod multinomial_payload_matches_the_fit_2612;
mod multinomial_predictive_ratio_2612;
mod multinomial_separation_arming_2612;
mod probe_2695_warp_smoothness;
mod production_row_program_policy;
