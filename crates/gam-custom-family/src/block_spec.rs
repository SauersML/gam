//! Coefficient-group data model for the blockwise carrier: coefficient
//! groups/labels/priors, the custom-family block-role heuristic, and the
//! fit-level blockspec validator.
//!
//! The block data-model types (`ParameterBlockSpec`, `ParameterBlockState`,
//! `BlockWorkingSet`, `BlockGeometryDirectionalDerivative`) and the
//! effective-Jacobian / channel-Hessian abstractions live in `gam-problem`
//! (#1521) and reach this module through the parent `use super::*` prelude
//! (`pub use gam_problem::*`), so existing `crate::*` paths stay
//! stable. The internal-consistency validator `validate_blockspec_consistency`
//! also lives in `gam-problem` (`custom_family_blockwise`) and is likewise
//! pulled in via the prelude. The coefficient group/label/prior types,
//! `custom_family_block_role`, and the fit-level `validate_blockspecs`
//! precondition stay here because they depend on `CoefficientGroupPrior`,
//! `RhoPrior`, `BlockRole`, and `CustomFamilyError`.

use super::*;

use gam_spec::CoefficientGroupPrior;

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
    pub rho_prior: gam_problem::RhoPrior,
    pub outer_labels: Vec<String>,
    /// The group labels, i.e. the precision labels whose penalty pieces are
    /// INDEPENDENT Gaussian prior factors rather than additive pieces of one
    /// smooth prior. A fit consuming these specs must copy this into
    /// `BlockwiseFitOptions::independent_prior_factor_labels`, so the outer
    /// evidence uses the per-factor normalizer `Σ_k ½(rank Sₖ·log λₖ +
    /// log|Sₖ|₊)` for them instead of the coalesced `½ log|Σ λₖSₖ|₊` — the
    /// two differ exactly when hierarchical groups overlap, where coalescing
    /// loses `½ log λ` per shared dimension.
    pub independent_prior_factor_labels: Vec<String>,
}

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
        "log_sigma" | "scale" | "logslope_surface" => BlockRole::Scale,
        "time" | "time_transform" | "time_surface" => BlockRole::Time,
        name if name.starts_with("time_cause_") => BlockRole::Time,
        "wiggle" | "linkwiggle" => BlockRole::LinkWiggle,
        _ if index == 0 => BlockRole::Location,
        _ => BlockRole::Scale,
    }
}

pub fn validate_blockspecs(specs: &[ParameterBlockSpec]) -> Result<Vec<usize>, String> {
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
