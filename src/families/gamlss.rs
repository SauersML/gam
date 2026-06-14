// Split from the original oversized module; keep included in order.
include!("gamlss/imports.rs");

mod dispersion_family;

include!("gamlss/dispersion_family_imports.rs");

mod binomial_q_derivs;

include!("gamlss/binomial_q_derivs_imports.rs");

mod binomial_q_coeffs;

include!("gamlss/binomial_q_coeffs_imports.rs");

mod validation;

include!("gamlss/validation_imports.rs");

mod weighted_design_products;

include!("gamlss/weighted_design_products_imports.rs");

mod row_linalg;

include!("gamlss/row_linalg_imports.rs");

mod joint_packing;

include!("gamlss/core_errors_and_builders.rs");
include!("gamlss/gaussian_and_log_families.rs");
include!("gamlss/binomial_location_scale.rs");
include!("gamlss/binomial_location_scale_wiggle_workspace.rs");
include!("gamlss/gamlss_tests.rs");
