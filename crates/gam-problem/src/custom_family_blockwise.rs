//! Neutral blockwise custom-family contract primitives shared by the
//! `CustomFamily` trait layer (`gam-model-api`) and the solver
//! (`gam-solve`): the block-spec consistency validator and the
//! exact-Newton outer-curvature payload.
//!
//! These carry no dependency on the `CustomFamily` trait itself, so they live
//! in the neutral `gam-problem` crate and are re-exported upward, keeping a
//! single definition shared across crates.

use crate::{CustomFamilyError, ParameterBlockSpec};
use ndarray::Array2;
use std::collections::BTreeMap;

/// # Why this returns the typed error and not a rendered `String`
///
/// Every refusal below is STRUCTURAL: a duplicate block name, a design/offset
/// dimension mismatch, a penalty that is not `p x p`. None of them becomes true
/// or false by moving `theta`. This function used to build each one as its
/// proper `ConstraintViolation` / `DimensionMismatch` and then `Into::into` it
/// to text at the `return` -- ten typed errors flattened at the point of
/// construction, which is the most expensive place to lose a variant and the
/// least visible.
///
/// Any caller that carried the result back into a
/// `Result<_, CustomFamilyError>` then got the blanket `From<String>`, which
/// answers `TrialPointRefused` for everything. So ten FATAL validation
/// failures arrived graded recoverable, and the outer smoothing search was
/// invited to "step away" from a block spec that is wrong at every `theta`.
/// That is gam#2590's mistake with the sign reversed (gam#2667).
pub fn validate_blockspec_consistency(
    specs: &[ParameterBlockSpec],
) -> Result<Vec<usize>, CustomFamilyError> {
    let mut seen_names = BTreeMap::<String, usize>::new();
    for (b, spec) in specs.iter().enumerate() {
        if let Some(prev) = seen_names.insert(spec.name.clone(), b) {
            return Err(CustomFamilyError::ConstraintViolation {
                reason: format!(
                    "duplicate parameter block name '{}' at indices {prev} and {b}: block names must be unique so coefficient labels resolved by name are unambiguous",
                    spec.name
                ),
            });
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
            });
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
                    });
                }
                if sd.ncols() != spec.design.ncols() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} stacked_design column count {} disagrees with \
                             design column count {}",
                            sd.ncols(),
                            spec.design.ncols(),
                        ),
                    });
                }
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "block {b} stacked_design and stacked_offset must be Some together \
                         or both None"
                    ),
                });
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
            });
        }
        if spec.initial_log_lambdas.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} initial_log_lambdas length {} does not match penalties {}",
                    spec.initial_log_lambdas.len(),
                    spec.penalties.len()
                ),
            });
        }
        // `nullspace_dims` is either empty (eigenvalue rank detection) or one
        // structural nullity per penalty. Any other length is not a third
        // mode: the penalty-root assembly would silently drop the declared
        // nullities and fall back to numerical rank detection.
        if !spec.nullspace_dims.is_empty() && spec.nullspace_dims.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} nullspace_dims length {} does not match penalties {} \
                     (must be empty or one structural nullity per penalty)",
                    spec.nullspace_dims.len(),
                    spec.penalties.len()
                ),
            });
        }
        for (k, &log_lambda) in spec.initial_log_lambdas.iter().enumerate() {
            if let Err(error) = crate::validate_log_strength(log_lambda) {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!("block {b} initial log-precision {k}: {error}"),
                });
            }
        }
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.shape();
            if r != p || c != p {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!("block {b} penalty {k} must be {p}x{p}, got {r}x{c}"),
                });
            }
            // Establish the full quadratic-penalty contract (finite, symmetric,
            // PSD, consistent embedding) once at the boundary; every downstream
            // gradient, root, and pseudo-logdet assumes it without re-checking.
            if let Err(reason) = s.validate(p) {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!("block {b} penalty {k}: {reason}"),
                });
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

#[cfg(test)]
mod validate_blockspec_tests {
    use crate::{CustomFamilyError, ParameterBlockSpec};
    use gam_linalg::matrix::DesignMatrix;
    use ndarray::{Array1, array};

    fn block(name: &str) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [1.0]
            ])),
            offset: array![0.0, 0.0],
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    #[test]
    fn a_duplicate_block_name_refuses_with_a_structural_variant_2667() {
        // Two blocks with the same name is wrong at EVERY theta, so it must
        // arrive as a structural variant and NOT as something
        // `is_trial_point_infeasible()` invites the outer search to step away
        // from. This used to be `.into()`d to text at the `return`, after
        // which the blanket `From<String>` could only guess it back as
        // `TrialPointRefused`: fatal, graded recoverable (gam#2667).
        let error = super::validate_blockspec_consistency(&[block("dup"), block("dup")])
            .expect_err("duplicate block names must be refused");
        assert!(
            matches!(error, CustomFamilyError::ConstraintViolation { .. }),
            "a duplicate block name is structural, got: {error:?}"
        );
        assert!(
            !error.is_trial_point_infeasible(),
            "a refusal true at every theta must not be graded rho-local: {error}"
        );
        assert!(
            error.to_string().contains("duplicate parameter block name"),
            "the reason must survive the typed return: {error}"
        );
    }

    #[test]
    fn a_dimension_mismatch_refuses_with_a_structural_variant_2667() {
        let mut spec = block("b0");
        spec.offset = array![0.0, 0.0, 0.0];
        let error = super::validate_blockspec_consistency(&[spec])
            .expect_err("an offset/design row mismatch must be refused");
        assert!(
            matches!(error, CustomFamilyError::DimensionMismatch { .. }),
            "an offset length mismatch is structural, got: {error:?}"
        );
        assert!(!error.is_trial_point_infeasible(), "{error}");
    }

    #[test]
    fn a_nullspace_dims_length_mismatch_refuses_instead_of_falling_back() {
        // A declared nullity list whose length disagrees with the penalty list
        // used to pass validation and be silently ignored by the penalty-root
        // assembly (`declared = len == penalties.len()`), so the structural
        // nullities the caller supplied were replaced by eigenvalue rank
        // detection without a word.
        let mut spec = block("b0");
        spec.nullspace_dims = vec![1];
        let error = super::validate_blockspec_consistency(&[spec])
            .expect_err("a nullspace_dims/penalties length mismatch must be refused");
        assert!(
            matches!(error, CustomFamilyError::DimensionMismatch { .. }),
            "a nullspace_dims length mismatch is structural, got: {error:?}"
        );
        assert!(
            error.to_string().contains("nullspace_dims length 1"),
            "{error}"
        );
    }

    #[test]
    fn consistent_specs_still_report_their_penalty_counts() {
        let counts = super::validate_blockspec_consistency(&[block("a"), block("b")])
            .expect("distinct, dimensionally consistent blocks must validate");
        assert_eq!(counts, vec![0, 0]);
    }
}
