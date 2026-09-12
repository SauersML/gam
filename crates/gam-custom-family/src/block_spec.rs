//! The custom-family block-role heuristic and the fit-level blockspec
//! validator for the blockwise carrier.
//!
//! The block data-model types (`ParameterBlockSpec`, `ParameterBlockState`,
//! `BlockWorkingSet`, `BlockGeometryDirectionalDerivative`) and the
//! effective-Jacobian / channel-Hessian abstractions live in `gam-problem`
//! (#1521) and reach this module through the parent `use super::*` prelude
//! (`pub use gam_problem::*`), so existing `crate::*` paths stay
//! stable. The internal-consistency validator `validate_blockspec_consistency`
//! also lives in `gam-problem` (`custom_family_blockwise`) and is likewise
//! pulled in via the prelude. `custom_family_block_role` and the fit-level
//! `validate_blockspecs` precondition stay here because they depend on
//! `BlockRole` and `CustomFamilyError`.

use super::*;

pub(crate) fn custom_family_block_role(
    name: &str,
    index: usize,
    n_blocks: usize,
) -> gam_problem::BlockRole {
    use gam_problem::BlockRole;

    if n_blocks == 1 {
        return BlockRole::Mean;
    }

    match name.trim().to_ascii_lowercase().as_str() {
        "eta" | "mean" | "beta" => BlockRole::Mean,
        "mu" | "location" | "marginal_surface" => BlockRole::Location,
        "threshold" => BlockRole::Threshold,
        "log_sigma" | "scale" | "slope_surface" => BlockRole::Scale,
        "time" | "time_transform" | "time_surface" => BlockRole::Time,
        name if name.starts_with("time_cause_") => BlockRole::Time,
        "wiggle" | "linkwiggle" => BlockRole::LinkWiggle,
        _ if index == 0 => BlockRole::Location,
        _ => BlockRole::Scale,
    }
}

pub fn validate_blockspecs(
    specs: &[ParameterBlockSpec],
) -> Result<Vec<usize>, CustomFamilyError> {
    // `fit_custom_family` is a fit entry point and genuinely requires at least
    // one parameter block — an empty model has nothing to estimate. This is a
    // *fit-level precondition*, distinct from the *consistency* of the block
    // specs themselves, which is checked by `validate_blockspec_consistency`.
    if specs.is_empty() {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "fit_custom_family requires at least one parameter block".to_string(),
        });
    }
    validate_blockspec_consistency(specs)
}
