// Split from the original oversized module; keep included in order.
include!("gamlss/part_000.rs");

mod dispersion_family;

include!("gamlss/part_001.rs");

mod binomial_q_derivs;

include!("gamlss/part_002.rs");

mod binomial_q_coeffs;

include!("gamlss/part_003.rs");

mod validation;

include!("gamlss/part_004.rs");

mod weighted_design_products;

include!("gamlss/part_005.rs");

mod row_linalg;

include!("gamlss/part_006.rs");

mod joint_packing;

include!("gamlss/part_007.rs");
include!("gamlss/part_008.rs");
include!("gamlss/part_009.rs");
include!("gamlss/part_010.rs");
include!("gamlss/part_011.rs");
