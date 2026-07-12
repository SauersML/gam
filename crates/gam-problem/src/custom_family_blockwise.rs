//! Neutral blockwise custom-family contract primitives shared by the
//! `CustomFamily` trait layer (`gam-model-api`) and the solver
//! (`gam-solve`): the IRLS weight / ridge floors, the block-spec consistency
//! validator, and the exact-Newton outer-curvature payload.
//!
//! These carry no dependency on the `CustomFamily` trait itself, so they live
//! in the neutral `gam-problem` crate and are re-exported upward, keeping a
//! single definition shared across crates.

use crate::{CustomFamilyError, ParameterBlockSpec};
use ndarray::Array2;
use std::collections::BTreeMap;

/// Floor applied to IRLS working weights so downstream divisions cannot hit
/// exact zero. Used as the default `minweight` in `CustomFamilyOptions` and
/// mirrored in tests that override it.
///
/// Sourced from the canonical positive-weight floor
/// ([`crate::types::MIN_WEIGHT`] = `1e-12`) so every floored family shares one
/// definition; this alias keeps the descriptive local name at the `minweight`
/// defaults.
pub const CUSTOM_FAMILY_WEIGHT_FLOOR: f64 = crate::types::MIN_WEIGHT;

/// Default initial ridge δ for the explicit-stabilization Cholesky escalation
/// schedule. Enters the quadratic term, the Laplace Hessian, and the penalty
/// log-determinant via the active `RidgePolicy`.
pub const CUSTOM_FAMILY_RIDGE_FLOOR: f64 = 1e-12;

pub fn validate_blockspec_consistency(specs: &[ParameterBlockSpec]) -> Result<Vec<usize>, String> {
    let mut seen_names = BTreeMap::<String, usize>::new();
    for (b, spec) in specs.iter().enumerate() {
        if let Some(prev) = seen_names.insert(spec.name.clone(), b) {
            return Err(CustomFamilyError::ConstraintViolation {
                reason: format!(
                    "duplicate parameter block name '{}' at indices {prev} and {b}: block names must be unique so coefficient labels resolved by name are unambiguous",
                    spec.name
                ),
            }
            .into());
        }
    }
    let mut penalty_counts = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let n = spec.design.nrows();
        if spec.offset.len() != n {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} offset length mismatch: got {}, expected {}",
                    spec.offset.len(),
                    n
                ),
            }
            .into());
        }
        // `stacked_design` and `stacked_offset` must be `Some` together
        // and their row/length must agree.  This enforces the contract
        // that `solver_design()` and `solver_offset()` always return a
        // matched pair.
        match (&spec.stacked_design, &spec.stacked_offset) {
            (Some(sd), Some(so)) => {
                if sd.nrows() != so.len() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} stacked_design/stacked_offset row mismatch: \
                             stacked_design.nrows()={}, stacked_offset.len()={}",
                            sd.nrows(),
                            so.len(),
                        ),
                    }
                    .into());
                }
                if sd.ncols() != spec.design.ncols() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} stacked_design column count {} disagrees with \
                             design column count {}",
                            sd.ncols(),
                            spec.design.ncols(),
                        ),
                    }
                    .into());
                }
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "block {b} stacked_design and stacked_offset must be Some together \
                         or both None"
                    ),
                }
                .into());
            }
        }
        let p = spec.design.ncols();
        if let Some(beta0) = &spec.initial_beta
            && beta0.len() != p
        {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} initial_beta length mismatch: got {}, expected {p}",
                    beta0.len()
                ),
            }
            .into());
        }
        if spec.initial_log_lambdas.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} initial_log_lambdas length {} does not match penalties {}",
                    spec.initial_log_lambdas.len(),
                    spec.penalties.len()
                ),
            }
            .into());
        }
        for (k, &log_lambda) in spec.initial_log_lambdas.iter().enumerate() {
            if let Err(error) = crate::validate_log_strength(log_lambda) {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "block {b} initial log-precision {k}: {error}"
                    ),
                }
                .into());
            }
        }
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.shape();
            if r != p || c != p {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!("block {b} penalty {k} must be {p}x{p}, got {r}x{c}"),
                }
                .into());
            }
            // Establish the full quadratic-penalty contract (finite, symmetric,
            // PSD, consistent embedding) once at the boundary; every downstream
            // gradient, root, and pseudo-logdet assumes it without re-checking.
            if let Err(reason) = s.validate(p) {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!("block {b} penalty {k}: {reason}"),
                }
                .into());
            }
        }
        penalty_counts.push(spec.penalties.len());
    }
    Ok(penalty_counts)
}

/// Scale-aware exact joint curvature payload for the outer REML evaluator.
pub struct ExactNewtonOuterCurvature {
    pub hessian: Array2<f64>,
    pub rho_curvature_scale: f64,
    pub hessian_logdet_correction: f64,
}
