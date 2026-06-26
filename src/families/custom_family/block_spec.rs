//! Data model for the blockwise carrier: parameter-block specs, coefficient
//! groups/labels/priors, per-block working sets and states, the effective-Jacobian
//! and channel-Hessian abstractions, and blockspec validation.
//!
//! The block data-model types (`ParameterBlockSpec`, `ParameterBlockState`,
//! `BlockWorkingSet`, `BlockGeometryDirectionalDerivative`) and the
//! effective-Jacobian / channel-Hessian abstractions now live in `gam-problem`
//! (#1521) and are re-exported here so existing
//! `crate::families::custom_family::*` paths remain stable. The coefficient
//! group/label/prior types, `custom_family_block_role`, and the blockspec
//! validators stay here because they depend on `CoefficientGroupPrior`,
//! `RhoPrior`, `BlockRole`, and `CustomFamilyError`.

use super::*;

use crate::types::CoefficientGroupPrior;

pub use gam_problem::{
    AdditiveBlockJacobian, BlockEffectiveJacobian, BlockGeometryDirectionalDerivative,
    BlockWorkingSet, FamilyChannelHessian, FamilyLinearizationState, GaugeComposedJacobian,
    ParameterBlockSpec, ParameterBlockState, RowScaledJacobian, TensorChannelHessian,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoefficientBlockSelector {
    Name(String),
    Index(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientLabel {
    pub block: CoefficientBlockSelector,
    pub column: usize,
}

impl CoefficientLabel {
    pub fn by_block_name(block: impl Into<String>, column: usize) -> Self {
        Self {
            block: CoefficientBlockSelector::Name(block.into()),
            column,
        }
    }
}

pub fn coefficient_label(block: impl Into<String>, column: usize) -> CoefficientLabel {
    CoefficientLabel::by_block_name(block, column)
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoefficientGroupSpec {
    pub label: String,
    pub coefficients: Vec<CoefficientLabel>,
    pub parent: Option<String>,
    pub prior: Option<CoefficientGroupPrior>,
    pub initial_log_precision: Option<f64>,
}

impl CoefficientGroupSpec {
    pub fn new(label: impl Into<String>, coefficients: Vec<CoefficientLabel>) -> Self {
        Self {
            label: label.into(),
            coefficients,
            parent: None,
            prior: None,
            initial_log_precision: None,
        }
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_prior(mut self, prior: CoefficientGroupPrior) -> Self {
        self.prior = Some(prior);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealizedCoefficientGroup {
    pub label: String,
    pub parent: Option<String>,
    pub coefficients: Vec<(usize, usize)>,
    pub prior: Option<CoefficientGroupPrior>,
    pub initial_log_precision: f64,
}

#[derive(Debug, Clone)]
pub struct RealizedCoefficientGroupSpecs {
    pub specs: Vec<ParameterBlockSpec>,
    pub groups: Vec<RealizedCoefficientGroup>,
    /// One entry per realized penalty in flattened block order. Built-in
    /// penalties receive unique internal labels; user groups carry their
    /// declared labels. Consumers that optimize one coordinate per label can
    /// use this to tie cross-block penalty pieces to a shared precision.
    pub penalty_labels: Vec<String>,
    /// Per-coordinate priors in `outer_labels` order.
    pub rho_prior: crate::types::RhoPrior,
    pub outer_labels: Vec<String>,
}

pub(crate) fn custom_family_block_role(
    name: &str,
    index: usize,
    n_blocks: usize,
) -> crate::model_types::BlockRole {
    use crate::model_types::BlockRole;

    if n_blocks == 1 {
        return BlockRole::Mean;
    }

    match name.trim().to_ascii_lowercase().as_str() {
        "eta" | "mean" | "beta" => BlockRole::Mean,
        "mu" | "location" | "marginal_surface" => BlockRole::Location,
        "threshold" => BlockRole::Threshold,
        "log_sigma" | "scale" | "logslope_surface" => BlockRole::Scale,
        "time" | "time_transform" | "time_surface" => BlockRole::Time,
        name if name.starts_with("time_cause_") => BlockRole::Time,
        "wiggle" | "linkwiggle" => BlockRole::LinkWiggle,
        _ if index == 0 => BlockRole::Location,
        _ => BlockRole::Scale,
    }
}

pub(crate) fn validate_blockspecs(specs: &[ParameterBlockSpec]) -> Result<Vec<usize>, String> {
    // `fit_custom_family` is a fit entry point and genuinely requires at least
    // one parameter block — an empty model has nothing to estimate. This is a
    // *fit-level precondition*, distinct from the *consistency* of the block
    // specs themselves, which is checked by `validate_blockspec_consistency`.
    if specs.is_empty() {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "fit_custom_family requires at least one parameter block".to_string(),
        }
        .into());
    }
    validate_blockspec_consistency(specs)
}

/// Validate the *internal consistency* of a slice of parameter block specs
/// (unique names; design/offset/initial_beta/penalty dimensions agree) without
/// imposing the fit-level "at least one block" precondition.
///
/// An empty slice is vacuously consistent and returns an empty penalty-count
/// vector. The non-empty fit precondition lives in [`validate_blockspecs`];
/// pure operator-materialization hooks (e.g. `batched_outer_hessian_terms`)
/// must use this consistency check instead, so they can be probed with an
/// empty, self-consistent argument set without tripping a fit precondition
/// that does not apply to them.
pub(crate) fn validate_blockspec_consistency(
    specs: &[ParameterBlockSpec],
) -> Result<Vec<usize>, String> {
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
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.shape();
            if r != p || c != p {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!("block {b} penalty {k} must be {p}x{p}, got {r}x{c}"),
                }
                .into());
            }
        }
        penalty_counts.push(spec.penalties.len());
    }
    Ok(penalty_counts)
}
