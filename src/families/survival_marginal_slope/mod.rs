// Split from the original oversized module; keep included in order.
include!("imports.rs");

mod poly_arith;

include!("types.rs");
include!("family_impl.rs");
include!("joint_newton.rs");
include!("fit_entry.rs");
include!("tests.rs");
