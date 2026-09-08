// Compile with rustc --test and the warm gam_math/ndarray dependencies on MSI.
// The test module includes the production source directly; there is no copied
// implementation whose behavior could drift from the live model.
#[path = "../../../crates/gam-models/src/survival/marginal_slope/timewiggle_geometry/scalar_q.rs"]
mod scalar_q;
