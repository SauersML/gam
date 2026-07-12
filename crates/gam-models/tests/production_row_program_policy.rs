//! #932 production-artifact policy.
//!
//! Integration tests compile `gam_models` as a dependency without `cfg(test)`.
//! These public imports and trait bounds therefore fail at compile time if a
//! canonical family row program is moved back under a unit-test gate or loses
//! its `RowProgram` implementation. The numerical parity/finite-difference
//! pins remain beside each specialized production path and instantiate these
//! same exported types through absolute paths.

use gam_math::jet_tower::RowProgram;
use gam_models::gamlss::{GaussianJointRowProgram, GaussianJointRowScalars};
use gam_models::survival::CauseSpecificRowProgram;
use gam_models::MultinomialLogitRowProgram;

fn require_row_program<const K: usize, P: RowProgram<K>>() {}

#[test]
fn canonical_row_programs_are_production_artifacts() {
    require_row_program::<2, MultinomialLogitRowProgram<2>>();
    require_row_program::<3, CauseSpecificRowProgram>();

    fn require_gaussian<'a>() {
        require_row_program::<2, GaussianJointRowProgram<'a>>();
    }
    require_gaussian();

    let _: for<'a> fn(&'a GaussianJointRowScalars) -> GaussianJointRowProgram<'a> =
        GaussianJointRowProgram::new;
}
