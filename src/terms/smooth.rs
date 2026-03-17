use crate::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisBuildResult, BasisError,
    BasisMetadata, BasisPsiDerivativeResult, BasisPsiSecondDerivativeResult, CenterStrategy,
    DuchonBasisSpec, MaternBasisSpec, MaternIdentifiability, PenaltyCandidate, PenaltyInfo,
    PenaltySource, SpatialIdentifiability, ThinPlateBasisSpec, apply_sum_to_zero_constraint,
    applyweighted_orthogonality_constraint, build_bspline_basis_1d, build_duchon_basis,
    build_duchon_basis_log_kappa_aniso_derivatives, build_duchon_basis_log_kappa_derivative,
    build_duchon_basis_log_kappasecond_derivative, build_duchon_basiswithworkspace,
    build_matern_basis,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basis_log_kappa_derivative,
    build_matern_basiswithworkspace,
    build_matern_basis_log_kappasecond_derivative, build_matern_collocation_operator_matrices,
    build_thin_plate_basis, build_thin_plate_basis_log_kappa_derivative,
    build_thin_plate_basis_log_kappasecond_derivative, estimate_penalty_nullity,
    filter_active_penalty_candidates,
};
use crate::construction::kronecker_product;
use crate::custom_family::{
    BlockGeometryDirectionalDerivative, BlockWorkingSet, BlockwiseFitOptions, CustomFamily,
    CustomFamilyBlockPsiDerivative, CustomFamilyWarmStart, ExactNewtonJointPsiTerms,
    ExactNewtonOuterObjective, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    evaluate_custom_family_joint_hyper, fit_custom_family,
};
use crate::estimate::{
    EstimationError, ExternalOptimOptions, FitInference, FitOptions, FittedLinkState,
    UnifiedFitResult, UnifiedFitResultParts, fit_gamwith_heuristic_lambdas,
    reml::DirectionalHyperParam,
};
use crate::faer_ndarray::fast_atv;
use crate::families::strategy::{FamilyStrategy, strategy_for_family};
use crate::matrix::{
    BlockDesignOperator, DesignBlock, DesignMatrix, RandomEffectOperator, SymmetricMatrix,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::pirls::LinearInequalityConstraints;
use crate::types::{InverseLink, LikelihoodFamily, MixtureLinkState, SasLinkState};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::f64;
use std::ops::Range;
use std::sync::Arc;

fn describe_thin_plate_center_request(strategy: &CenterStrategy) -> String {
    match strategy {
        CenterStrategy::UserProvided(centers) => format!("{} centers", centers.nrows()),
        CenterStrategy::EqualMass { num_centers }
        | CenterStrategy::EqualMassCovarRepresentative { num_centers }
        | CenterStrategy::FarthestPoint { num_centers }
        | CenterStrategy::KMeans { num_centers, .. } => format!("{num_centers} centers"),
        CenterStrategy::UniformGrid { points_per_dim } => {
            format!("uniform grid with {points_per_dim} points per dimension")
        }
    }
}

fn rewrite_thin_plate_knots_error(
    err: BasisError,
    termname: &str,
    feature_count: usize,
    spec: &ThinPlateBasisSpec,
) -> BasisError {
    match err {
        BasisError::InvalidInput(msg)
            if msg.contains("thin-plate spline requires at least d+1 knots") =>
        {
            let min_centers = feature_count + 1;
            let requested = describe_thin_plate_center_request(&spec.center_strategy);
            BasisError::InvalidInput(format!(
                "joint TPS term '{termname}' over {feature_count} covariates with {requested} is invalid; minimum centers is {min_centers}"
            ))
        }
        other => other,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeConstraint {
    None,
    MonotoneIncreasing,
    MonotoneDecreasing,
    Convex,
    Concave,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmoothBasisSpec {
    BSpline1D {
        feature_col: usize,
        spec: BSplineBasisSpec,
    },
    ThinPlate {
        feature_cols: Vec<usize>,
        spec: ThinPlateBasisSpec,
        /// Per-column standard deviations used to standardize input dimensions
        /// before kernel evaluation when d > 1. `None` means no standardization
        /// (either d == 1 or explicitly disabled).
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
    },
    Matern {
        feature_cols: Vec<usize>,
        spec: MaternBasisSpec,
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
    },
    Duchon {
        feature_cols: Vec<usize>,
        spec: DuchonBasisSpec,
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
    },
    /// Tensor-product smooth built from 1D B-spline marginals.
    ///
    /// This is the `te()`-style construction used when axes have different units/scales
    /// (for example, space x time) and isotropic radial kernels are not appropriate.
    TensorBSpline {
        feature_cols: Vec<usize>,
        spec: TensorBSplineSpec,
    },
}

/// Tensor-product B-spline smooth specification.
///
/// `marginalspecs[i]` is the 1D B-spline setup for `feature_cols[i]`.
/// The final penalty set is one Kronecker penalty per margin:
/// `S_i = I ⊗ ... ⊗ S_marginal_i ⊗ ... ⊗ I`, plus optional global ridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBSplineSpec {
    pub marginalspecs: Vec<BSplineBasisSpec>,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: TensorBSplineIdentifiability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorBSplineIdentifiability {
    None,
    SumToZero,
    FrozenTransform { transform: Array2<f64> },
}

impl Default for TensorBSplineIdentifiability {
    fn default() -> Self {
        Self::SumToZero
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothTermSpec {
    pub name: String,
    pub basis: SmoothBasisSpec,
    pub shape: ShapeConstraint,
}

#[derive(Debug, Clone)]
pub struct SmoothTerm {
    pub name: String,
    pub coeff_range: Range<usize>,
    pub shape: ShapeConstraint,
    pub penalties_local: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo_local: Vec<PenaltyInfo>,
    pub metadata: BasisMetadata,
    /// Optional term-local lower bounds for constrained coefficients.
    /// `-inf` means unconstrained.
    pub lower_bounds_local: Option<Array1<f64>>,
    /// Optional term-local inequality constraints in local coefficient coordinates.
    /// `A_local * beta_local >= b_local`.
    pub linear_constraints_local: Option<LinearInequalityConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyBlockInfo {
    pub global_index: usize,
    pub termname: Option<String>,
    pub penalty: PenaltyInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroppedPenaltyBlockInfo {
    pub termname: Option<String>,
    pub penalty: PenaltyInfo,
}

#[derive(Debug, Clone)]
pub struct SmoothDesign {
    pub term_designs: Vec<Array2<f64>>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    pub terms: Vec<SmoothTerm>,
    /// Optional smooth-block lower bounds in smooth coefficient coordinates.
    /// Length equals `total_smooth_cols()` when present.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional smooth-block inequality constraints:
    /// `A_smooth * beta_smooth >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
}

impl SmoothDesign {
    pub fn total_smooth_cols(&self) -> usize {
        self.term_designs.iter().map(|d| d.ncols()).sum()
    }
    pub fn nrows(&self) -> usize {
        self.term_designs.first().map_or(0, |d| d.nrows())
    }
}

#[derive(Debug, Clone)]
pub struct RawSmoothDesign {
    pub term_designs: Vec<Array2<f64>>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    pub terms: Vec<SmoothTerm>,
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    pub linear_constraints: Option<LinearInequalityConstraints>,
}

impl RawSmoothDesign {
    pub fn total_smooth_cols(&self) -> usize {
        self.term_designs.iter().map(|d| d.ncols()).sum()
    }
    pub fn nrows(&self) -> usize {
        self.term_designs.first().map_or(0, |d| d.nrows())
    }
}

impl From<RawSmoothDesign> for SmoothDesign {
    fn from(value: RawSmoothDesign) -> Self {
        Self {
            term_designs: value.term_designs,
            penalties: value.penalties,
            nullspace_dims: value.nullspace_dims,
            penaltyinfo: value.penaltyinfo,
            dropped_penaltyinfo: value.dropped_penaltyinfo,
            terms: value.terms,
            coefficient_lower_bounds: value.coefficient_lower_bounds,
            linear_constraints: value.linear_constraints,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundedCoefficientPriorSpec {
    None,
    Uniform,
    Beta { a: f64, b: f64 },
}

impl Default for BoundedCoefficientPriorSpec {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LinearCoefficientGeometry {
    #[default]
    Unconstrained,
    Bounded {
        min: f64,
        max: f64,
        #[serde(default)]
        prior: BoundedCoefficientPriorSpec,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearTermSpec {
    pub name: String,
    pub feature_col: usize,
    /// Default ridge penalty on this linear coefficient.
    /// Non-intercept linear terms are penalized by default; set false only to
    /// opt into an explicitly unpenalized parametric effect.
    #[serde(default = "default_linear_term_double_penalty")]
    pub double_penalty: bool,
    #[serde(default)]
    pub coefficient_geometry: LinearCoefficientGeometry,
    #[serde(default)]
    pub coefficient_min: Option<f64>,
    #[serde(default)]
    pub coefficient_max: Option<f64>,
}

const fn default_linear_term_double_penalty() -> bool {
    true
}

/// Random-effects term specification.
///
/// The selected feature column is interpreted as a categorical grouping variable.
/// The term contributes a one-hot dummy block with an identity penalty on group
/// coefficients, equivalent to i.i.d. Gaussian random effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffectTermSpec {
    pub name: String,
    pub feature_col: usize,
    /// If true, drop the lexicographically first group level to use treatment coding.
    /// If false, keep all levels (full one-hot block, still identifiable under ridge).
    pub drop_first_level: bool,
    /// Optional fixed kept-level set (sorted by f64 bit pattern) captured at fit time.
    /// When present, prediction uses exactly these columns to avoid design drift.
    #[serde(default)]
    pub frozen_levels: Option<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermCollectionSpec {
    pub linear_terms: Vec<LinearTermSpec>,
    pub random_effect_terms: Vec<RandomEffectTermSpec>,
    pub smooth_terms: Vec<SmoothTermSpec>,
}

impl TermCollectionSpec {
    /// Validate that a term collection spec represents a fully frozen model
    /// (i.e. all knots/centers are pre-computed, identifiability transforms are
    /// baked in, and random-effect levels are fixed).
    pub fn validate_frozen(&self, label: &str) -> Result<(), String> {
        for linear in &self.linear_terms {
            if let (Some(min), Some(max)) = (linear.coefficient_min, linear.coefficient_max)
                && (!min.is_finite() || !max.is_finite() || min > max)
            {
                return Err(format!(
                    "{label} linear term '{}' has invalid coefficient constraint [{min}, {max}]",
                    linear.name
                ));
            }
            if let Some(min) = linear.coefficient_min
                && !min.is_finite()
            {
                return Err(format!(
                    "{label} linear term '{}' has non-finite coefficient minimum {min}",
                    linear.name
                ));
            }
            if let Some(max) = linear.coefficient_max
                && !max.is_finite()
            {
                return Err(format!(
                    "{label} linear term '{}' has non-finite coefficient maximum {max}",
                    linear.name
                ));
            }
            if let LinearCoefficientGeometry::Bounded { min, max, prior } =
                &linear.coefficient_geometry
            {
                if !min.is_finite() || !max.is_finite() || min >= max {
                    return Err(format!(
                        "{label} bounded term '{}' has invalid bounds [{min}, {max}]",
                        linear.name
                    ));
                }
                match prior {
                    BoundedCoefficientPriorSpec::None | BoundedCoefficientPriorSpec::Uniform => {}
                    BoundedCoefficientPriorSpec::Beta { a, b } => {
                        if !a.is_finite() || !b.is_finite() || *a < 1.0 || *b < 1.0 {
                            return Err(format!(
                                "{label} bounded term '{}' has invalid Beta prior ({a}, {b})",
                                linear.name
                            ));
                        }
                    }
                }
            }
        }
        for st in &self.smooth_terms {
            match &st.basis {
                SmoothBasisSpec::BSpline1D { spec, .. } => {
                    if !matches!(spec.knotspec, BSplineKnotSpec::Provided(_)) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: BSpline knotspec must be Provided",
                            st.name
                        ));
                    }
                }
                SmoothBasisSpec::ThinPlate { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: ThinPlate centers must be UserProvided",
                            st.name
                        ));
                    }
                    if matches!(
                        spec.identifiability,
                        SpatialIdentifiability::OrthogonalToParametric
                    ) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: ThinPlate identifiability must be FrozenTransform or None",
                            st.name
                        ));
                    }
                }
                SmoothBasisSpec::Matern { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: Matern centers must be UserProvided",
                            st.name
                        ));
                    }
                }
                SmoothBasisSpec::Duchon { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: Duchon centers must be UserProvided",
                            st.name
                        ));
                    }
                    if matches!(
                        spec.identifiability,
                        SpatialIdentifiability::OrthogonalToParametric
                    ) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: Duchon identifiability must be FrozenTransform or None",
                            st.name
                        ));
                    }
                }
                SmoothBasisSpec::TensorBSpline { spec, .. } => {
                    for (dim, marginal) in spec.marginalspecs.iter().enumerate() {
                        if !matches!(marginal.knotspec, BSplineKnotSpec::Provided(_)) {
                            return Err(format!(
                                "{label} term '{}' dim {} is not frozen: tensor marginal knotspec must be Provided",
                                st.name, dim
                            ));
                        }
                    }
                    if matches!(
                        spec.identifiability,
                        TensorBSplineIdentifiability::SumToZero
                    ) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: tensor identifiability must be FrozenTransform or None",
                            st.name
                        ));
                    }
                }
            }
        }

        for rt in &self.random_effect_terms {
            if rt.frozen_levels.is_none() {
                return Err(format!(
                    "{label} random-effect term '{}' is not frozen: missing frozen_levels",
                    rt.name
                ));
            }
        }

        Ok(())
    }
}

/// A penalty matrix stored at its natural block size together with the
/// column range it occupies in the global coefficient vector.
///
/// Instead of embedding every penalty into a full `p_total × p_total` dense
/// matrix filled with zeros, we keep the compact local matrix and reconstruct
/// the global view only when a downstream consumer explicitly requires it.
#[derive(Debug, Clone)]
pub struct BlockwisePenalty {
    /// Column range in the global coefficient vector that this penalty covers.
    pub col_range: Range<usize>,
    /// The local penalty matrix — dimensions `block_p × block_p` where
    /// `block_p = col_range.len()`.
    pub local: Array2<f64>,
}

impl BlockwisePenalty {
    /// Create a new blockwise penalty.
    pub fn new(col_range: Range<usize>, local: Array2<f64>) -> Self {
        debug_assert_eq!(col_range.len(), local.nrows());
        debug_assert_eq!(col_range.len(), local.ncols());
        Self { col_range, local }
    }

    /// Expand this blockwise penalty into a full `p_total × p_total` dense
    /// matrix (mostly zeros). Use sparingly — the whole point of blockwise
    /// storage is to avoid this allocation.
    pub fn to_global(&self, p_total: usize) -> Array2<f64> {
        let mut g = Array2::<f64>::zeros((p_total, p_total));
        let r = &self.col_range;
        assert!(
            r.end <= p_total && self.local.nrows() == r.len() && self.local.ncols() == r.len(),
            "BlockwisePenalty::to_global shape invariant violated: \
             col_range={}..{}, local={}x{}, p_total={}",
            r.start,
            r.end,
            self.local.nrows(),
            self.local.ncols(),
            p_total,
        );
        g.slice_mut(s![r.start..r.end, r.start..r.end])
            .assign(&self.local);
        g
    }

    /// The block size of this penalty.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.col_range.len()
    }
}

/// Expand a full set of blockwise penalties into global `p_total × p_total`
/// dense matrices. This is a compatibility shim for code paths that still
/// consume `&[Array2<f64>]`.
pub fn penalties_to_global(penalties: &[BlockwisePenalty], p_total: usize) -> Vec<Array2<f64>> {
    penalties.iter().map(|bp| bp.to_global(p_total)).collect()
}

/// Compute `Σ_k λ_k S_k` directly from blockwise penalties, accumulating
/// into a pre-allocated `p_total × p_total` output without ever materializing
/// individual global matrices.
pub fn weighted_blockwise_penalty_sum(
    penalties: &[BlockwisePenalty],
    lambdas: &[f64],
    p_total: usize,
) -> Array2<f64> {
    debug_assert_eq!(penalties.len(), lambdas.len());
    let mut out = Array2::<f64>::zeros((p_total, p_total));
    for (bp, &lam) in penalties.iter().zip(lambdas.iter()) {
        let r = &bp.col_range;
        let mut slice = out.slice_mut(s![r.start..r.end, r.start..r.end]);
        slice.scaled_add(lam, &bp.local);
    }
    out
}

#[derive(Clone)]
pub struct TermCollectionDesign {
    /// The full design matrix.  When random effects are present this is an
    /// operator-based `BlockDesignOperator` wrapping a dense core and O(n)
    /// random-effect operators; otherwise a plain `Dense` matrix.
    pub design: DesignMatrix,
    pub penalties: Vec<BlockwisePenalty>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    /// Optional global coefficient lower bounds for constrained fitting.
    /// Length equals `design.ncols()` when present. Unconstrained entries are `-inf`.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional global inequality constraints:
    /// `A * beta >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
    pub intercept_range: Range<usize>,
    pub linear_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_levels: Vec<(String, Vec<u64>)>,
    pub smooth: SmoothDesign,
}

impl TermCollectionDesign {
    /// Expand blockwise penalties to global `p_total × p_total` dense matrices.
    /// This is a compatibility shim; prefer operating on blockwise penalties
    /// directly when possible.
    pub fn global_penalties(&self) -> Vec<Array2<f64>> {
        let p = self.design.ncols();
        penalties_to_global(&self.penalties, p)
    }

    /// Number of penalty blocks.
    #[inline]
    pub fn num_penalties(&self) -> usize {
        self.penalties.len()
    }
}

#[derive(Clone)]
pub struct FittedTermCollection {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone)]
pub struct FittedTermCollectionWithSpec {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveSpatialMap {
    pub termname: String,
    pub feature_cols: Vec<usize>,
    pub collocation_points: Array2<f64>,
    pub inv_magweight: Array1<f64>,
    pub invgradweight: Array1<f64>,
    pub inv_lapweight: Array1<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveRegularizationDiagnostics {
    pub epsilon_0: f64,
    pub epsilon_g: f64,
    pub epsilon_c: f64,
    pub epsilon_outer_iterations: usize,
    pub mm_iterations: usize,
    pub converged: bool,
    pub maps: Vec<AdaptiveSpatialMap>,
}

#[derive(Debug, Clone)]
struct LinearColumnConditioning {
    col_idx: usize,
    mean: f64,
    scale: f64,
}

#[derive(Debug, Clone, Default)]
struct LinearFitConditioning {
    intercept_idx: usize,
    columns: Vec<LinearColumnConditioning>,
}

#[derive(Debug, Clone)]
pub(crate) struct SpatialPsiDerivative {
    // These are derivatives with respect to psi = log(kappa), not log(length_scale).
    pub penalty_index: usize,
    pub penalty_indices: Vec<usize>,
    pub global_range: Range<usize>,
    pub total_p: usize,
    pub x_psi_local: Array2<f64>,
    pub s_psi_components_local: Vec<Array2<f64>>,
    pub x_psi_psi_local: Array2<f64>,
    pub s_psi_psi_components_local: Vec<Array2<f64>>,
    pub aniso_group_id: Option<usize>,
    /// Pre-computed cross-derivative design matrices for other axes
    /// in the same aniso group: Vec of (axis_offset_in_group, matrix).
    pub aniso_cross_designs: Option<Vec<(usize, Array2<f64>)>>,
    /// Pre-computed cross-penalty second derivatives ∂²S_m/∂ψ_a∂ψ_b for axes
    /// in the same aniso group. Each entry is (b_axis, Vec<Array2<f64>>),
    /// where b_axis is the axis offset within the group and the Vec has one
    /// matrix per active penalty. Only stored for pairs where this entry's
    /// axis a < b_axis (upper triangle); the symmetric pair is the transpose.
    pub aniso_cross_penalties: Option<Vec<(usize, Vec<Array2<f64>>)>>,
    /// Optional implicit design-derivative operator (shared across all axes
    /// in the same aniso group). When present, `x_psi_local` and
    /// `x_psi_psi_local` may be zero-sized, and design-derivative matvecs
    /// should go through this operator using `implicit_axis` as the axis index.
    pub implicit_operator: Option<std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>>,
    /// Which axis in the implicit operator this entry corresponds to.
    pub implicit_axis: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SpatialLogKappaCoords {
    /// Flattened ψ values. For isotropic terms, one entry per term.
    /// For anisotropic terms, d entries per term (one ψ_a per axis).
    values: Array1<f64>,
    /// Dimensionality of each term: 1 for isotropic, d for anisotropic.
    dims_per_term: Vec<usize>,
}

impl SpatialLogKappaCoords {
    /// Construct from an explicit dims layout plus values.
    pub(crate) fn new_with_dims(values: Array1<f64>, dims_per_term: Vec<usize>) -> Self {
        debug_assert_eq!(
            values.len(),
            dims_per_term.iter().sum::<usize>(),
            "SpatialLogKappaCoords: values length {} != sum of dims_per_term {}",
            values.len(),
            dims_per_term.iter().sum::<usize>(),
        );
        Self {
            values,
            dims_per_term,
        }
    }

    /// Isotropic initialization (backward-compatible path).
    pub(crate) fn from_length_scales(
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut out = Array1::<f64>::zeros(term_indices.len());
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let length_scale = get_spatial_length_scale(spec, term_idx)
                .unwrap_or(options.min_length_scale)
                .clamp(options.min_length_scale, options.max_length_scale);
            out[slot] = -length_scale.ln();
        }
        Self {
            values: out,
            dims_per_term: vec![1; term_indices.len()],
        }
    }

    /// Anisotropic-aware initialization.
    ///
    /// Initialization strategy (per math team recommendation): standardize the
    /// knot cloud axiswise, then run the existing isotropic κ initializer in
    /// the standardized space. This reuses the trusted isotropic initializer
    /// and gives initial η_a = −ln(σ_a) + mean(ln(σ_a)), which satisfies
    /// Ση_a = 0 by construction.
    ///
    /// For each term, checks whether it has `aniso_log_scales` set on its basis spec.
    /// - If isotropic (no aniso_log_scales, or 1-D): 1 entry = −ln(length_scale).
    /// - If anisotropic (aniso_log_scales present): d entries, one ψ_a per axis.
    ///   Initialized as ψ_a = −ln(length_scale) + η_a  where η_a are the existing
    ///   aniso_log_scales (which sum to zero). If aniso_log_scales is empty or missing
    ///   for a multi-dimensional term, the scalar −ln(length_scale) is broadcast.
    pub(crate) fn from_length_scales_aniso(
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut vals = Vec::new();
        let mut dims = Vec::new();
        for &term_idx in term_indices {
            let length_scale = get_spatial_length_scale(spec, term_idx)
                .unwrap_or(options.min_length_scale)
                .clamp(options.min_length_scale, options.max_length_scale);
            let psi_bar = -length_scale.ln(); // global scale = −ln(length_scale)

            let aniso = get_spatial_aniso_log_scales(spec, term_idx);
            let d = get_spatial_feature_dim(spec, term_idx).unwrap_or(1);

            match aniso {
                Some(ref eta) if eta.len() == d && d > 1 => {
                    // Existing per-axis anisotropy: ψ_a = ψ̄ + η_a
                    for &eta_a in eta {
                        vals.push(psi_bar + eta_a);
                    }
                    dims.push(d);
                }
                _ if d > 1 => {
                    // Anisotropic term but no existing scales: broadcast scalar
                    for _ in 0..d {
                        vals.push(psi_bar);
                    }
                    dims.push(d);
                }
                _ => {
                    // Isotropic (1-D or no multi-dim info)
                    vals.push(psi_bar);
                    dims.push(1);
                }
            }
        }
        Self {
            values: Array1::from_vec(vals),
            dims_per_term: dims,
        }
    }

    /// Isotropic lower bounds (backward-compatible): one bound per term.
    pub(crate) fn lower_bounds(
        dim: usize,
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        Self {
            values: Array1::<f64>::from_elem(dim, -options.max_length_scale.ln()),
            dims_per_term: vec![1; dim],
        }
    }

    /// Isotropic upper bounds (backward-compatible): one bound per term.
    pub(crate) fn upper_bounds(
        dim: usize,
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        Self {
            values: Array1::<f64>::from_elem(dim, -options.min_length_scale.ln()),
            dims_per_term: vec![1; dim],
        }
    }

    /// Anisotropic-aware lower bounds: d bounds per aniso term.
    pub(crate) fn lower_bounds_aniso(
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let total: usize = dims_per_term.iter().sum();
        Self {
            values: Array1::<f64>::from_elem(total, -options.max_length_scale.ln()),
            dims_per_term: dims_per_term.to_vec(),
        }
    }

    /// Anisotropic-aware upper bounds: d bounds per aniso term.
    pub(crate) fn upper_bounds_aniso(
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let total: usize = dims_per_term.iter().sum();
        Self {
            values: Array1::<f64>::from_elem(total, -options.min_length_scale.ln()),
            dims_per_term: dims_per_term.to_vec(),
        }
    }

    /// Reconstruct from theta tail with known dimensionality layout.
    pub(crate) fn from_theta_tail_with_dims(
        theta: &Array1<f64>,
        start: usize,
        dims_per_term: Vec<usize>,
    ) -> Self {
        let total: usize = dims_per_term.iter().sum();
        Self {
            values: theta.slice(s![start..start + total]).to_owned(),
            dims_per_term,
        }
    }

    /// Total number of ψ values in the flat array (= sum of dims_per_term).
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// Dimensionality layout: how many ψ values each term contributes.
    pub(crate) fn dims_per_term(&self) -> &[usize] {
        &self.dims_per_term
    }

    /// Get the offset into the flat array for logical term i.
    fn term_offset(&self, term_idx: usize) -> usize {
        self.dims_per_term[..term_idx].iter().sum()
    }

    /// Get the slice of ψ values for logical term i.
    fn term_slice(&self, term_idx: usize) -> &[f64] {
        let offset = self.term_offset(term_idx);
        let d = self.dims_per_term[term_idx];
        &self.values.as_slice().unwrap()[offset..offset + d]
    }

    pub(crate) fn as_array(&self) -> &Array1<f64> {
        &self.values
    }

    /// Split at a logical-term boundary. `mid` is the number of terms in the
    /// first half (not a flat-array index).
    pub(crate) fn split_at(&self, mid: usize) -> (Self, Self) {
        let flat_mid: usize = self.dims_per_term[..mid].iter().sum();
        (
            Self {
                values: self.values.slice(s![0..flat_mid]).to_owned(),
                dims_per_term: self.dims_per_term[..mid].to_vec(),
            },
            Self {
                values: self.values.slice(s![flat_mid..]).to_owned(),
                dims_per_term: self.dims_per_term[mid..].to_vec(),
            },
        )
    }

    /// Apply optimized ψ values back to the spec.
    ///
    /// For isotropic terms (dims=1): sets scalar length_scale = exp(−ψ).
    /// For anisotropic terms (dims=d): sets length_scale = exp(−ψ̄) where
    /// ψ̄ = mean(ψ_a), and sets aniso_log_scales = Some([η_a]) where
    /// η_a = ψ_a − ψ̄ (these sum to zero by construction).
    pub(crate) fn apply_tospec(
        &self,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
    ) -> Result<TermCollectionSpec, EstimationError> {
        if term_indices.len() != self.dims_per_term.len() {
            return Err(EstimationError::InvalidInput(format!(
                "SpatialLogKappaCoords::apply_tospec: term count mismatch: \
                 term_indices={} dims_per_term={}",
                term_indices.len(),
                self.dims_per_term.len()
            )));
        }
        let mut updated = spec.clone();
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let psi = self.term_slice(slot);
            let d = self.dims_per_term[slot];
            if d == 1 {
                // Isotropic: length_scale = exp(−ψ)
                let length_scale = (-psi[0]).exp();
                set_spatial_length_scale(&mut updated, term_idx, length_scale)?;
            } else {
                // Anisotropic: decompose ψ_a = ψ̄ + η_a where ψ̄ = mean(ψ_a).
                // This is the Λ = κA decomposition (det A = 1):
                //   κ = exp(ψ̄)     → stored as length_scale = exp(−ψ̄)
                //   A = diag(exp(η_a)) → stored as aniso_log_scales = η
                // The sum-to-zero constraint Ση_a = 0 is satisfied by construction.
                let psi_bar = psi.iter().sum::<f64>() / d as f64;
                let length_scale = (-psi_bar).exp();
                let eta: Vec<f64> = psi.iter().map(|&p| p - psi_bar).collect();
                set_spatial_length_scale(&mut updated, term_idx, length_scale)?;
                set_spatial_aniso_log_scales(&mut updated, term_idx, eta)?;
            }
        }
        Ok(updated)
    }
}

/// Get the `aniso_log_scales` from a spatial term, if present.
pub fn get_spatial_aniso_log_scales(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<Vec<f64>> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.aniso_log_scales.clone(),
            SmoothBasisSpec::Duchon { spec, .. } => spec.aniso_log_scales.clone(),
            _ => None,
        })
}

/// Get the number of feature columns (spatial dimensionality) for a spatial term.
fn get_spatial_feature_dim(spec: &TermCollectionSpec, term_idx: usize) -> Option<usize> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { feature_cols, .. } => Some(feature_cols.len()),
            SmoothBasisSpec::Matern { feature_cols, .. } => Some(feature_cols.len()),
            SmoothBasisSpec::Duchon { feature_cols, .. } => Some(feature_cols.len()),
            _ => None,
        })
}

/// Log the learned per-axis anisotropic length scales for all spatial terms
/// that have `aniso_log_scales` set after optimization.
///
/// For each anisotropic term, reports the per-axis eta values (deviation from
/// the mean log-kappa), the effective per-axis length scales, and the per-axis
/// kappa values.
pub fn log_spatial_aniso_scales(spec: &TermCollectionSpec) {
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let (aniso, length_scale) = match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                (spec.aniso_log_scales.as_ref(), Some(spec.length_scale))
            }
            SmoothBasisSpec::Duchon { spec, .. } => {
                (spec.aniso_log_scales.as_ref(), spec.length_scale)
            }
            _ => (None, None),
        };
        let Some(eta) = aniso else { continue };
        if eta.is_empty() {
            continue;
        }
        let Some(ls) = length_scale else { continue };
        // psi_bar = -ln(length_scale), kappa_bar = 1/length_scale
        // per-axis: kappa_a = kappa_bar * exp(eta_a), length_a = ls * exp(-eta_a)
        let mut lines = format!(
            "[spatial-kappa] term {} (\"{}\"): anisotropic length scales optimized (global length_scale={:.4})",
            term_idx, term.name, ls
        );
        for (a, &eta_a) in eta.iter().enumerate() {
            let length_a = ls * (-eta_a).exp();
            let kappa_a = (1.0 / ls) * eta_a.exp();
            lines.push_str(&format!(
                "\n  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                a, eta_a, length_a, kappa_a
            ));
        }
        log::info!("{}", lines);
    }
}

/// Set `aniso_log_scales` on a spatial term's basis spec.
fn set_spatial_aniso_log_scales(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    eta: Vec<f64>,
) -> Result<(), EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        return Err(EstimationError::InvalidInput(format!(
            "spatial aniso_log_scales term index {term_idx} out of range"
        )));
    };
    match &mut term.basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not support aniso_log_scales",
            term.name
        ))),
    }
}

/// Sync knot-cloud-derived anisotropy contrasts from basis metadata back into
/// the mutable spec so the optimizer starts from the correct eta values.
///
/// Call this after building the smooth design but before initializing the
/// optimizer's psi coordinates. For each spatial term whose metadata contains
/// computed `aniso_log_scales`, this writes them into the spec.
pub(crate) fn sync_aniso_contrasts_from_metadata(
    spec: &mut TermCollectionSpec,
    design: &SmoothDesign,
) {
    for (term_idx, term) in design.terms.iter().enumerate() {
        let meta_aniso = match &term.metadata {
            BasisMetadata::Matern {
                aniso_log_scales, ..
            } => aniso_log_scales.clone(),
            BasisMetadata::Duchon {
                aniso_log_scales, ..
            } => aniso_log_scales.clone(),
            _ => None,
        };
        if let Some(eta) = meta_aniso {
            if eta.len() > 1 {
                set_spatial_aniso_log_scales(spec, term_idx, eta).ok();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialLengthScaleOptimizationOptions {
    /// Enable outer-loop optimization over spatial κ (= 1 / length_scale)
    /// for supported radial-kernel smooths.
    /// This applies to ThinPlate, Matérn, and hybrid Duchon terms.
    pub enabled: bool,
    /// Maximum number of outer iterations in the exact joint [rho, psi] solve.
    pub max_outer_iter: usize,
    /// Relative improvement threshold for terminating the outer solve.
    pub rel_tol: f64,
    /// Initial log(length_scale) perturbation used for seed construction.
    pub log_step: f64,
    /// Minimum allowed length_scale during κ search.
    pub min_length_scale: f64,
    /// Maximum allowed length_scale during κ search.
    pub max_length_scale: f64,
}

impl Default for SpatialLengthScaleOptimizationOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_outer_iter: 3,
            rel_tol: 1e-4,
            // Seed auxiliary candidates around current scale by approximately x0.5 and x2.0.
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-3,
            max_length_scale: 1e3,
        }
    }
}

#[derive(Debug, Clone)]
struct RandomEffectBlock {
    name: String,
    /// O(n) group-label vector: group_ids[i] = column index in [0, num_groups).
    /// `None` if the observation's level is not in the kept set.
    group_ids: Vec<Option<usize>>,
    num_groups: usize,
    kept_levels: Vec<u64>,
}

/// Compute per-column standard deviations for multivariate spatial inputs (d > 1).
/// Returns `None` when d == 1 (standardization unnecessary) or when the caller
/// already supplies frozen scales (prediction path).
fn compute_spatial_input_scales(x: ArrayView2<'_, f64>) -> Option<Vec<f64>> {
    let d = x.ncols();
    if d <= 1 {
        return None;
    }
    let n = x.nrows() as f64;
    if n < 2.0 {
        return None;
    }
    let mut scales = Vec::with_capacity(d);
    for j in 0..d {
        let col = x.column(j);
        let mean = col.sum() / n;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        scales.push(var.sqrt().max(1e-12));
    }
    Some(scales)
}

/// Apply per-column standardization to a data matrix using precomputed scales.
fn apply_input_standardization(x: &mut Array2<f64>, scales: &[f64]) {
    for j in 0..x.ncols() {
        let inv = 1.0 / scales[j];
        x.column_mut(j).mapv_inplace(|v| v * inv);
    }
}

fn select_columns(data: ArrayView2<'_, f64>, cols: &[usize]) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    for &c in cols {
        if c >= p {
            return Err(BasisError::DimensionMismatch(format!(
                "feature column {c} is out of bounds for data with {p} columns"
            )));
        }
    }
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&data.column(c));
    }
    Ok(out)
}

impl LinearFitConditioning {
    fn from_columns(design: &TermCollectionDesign, selected_cols: &[usize]) -> Self {
        const SCALE_EPS: f64 = 1e-12;
        let dense = design.design.as_dense_cow();
        let mut columns = Vec::with_capacity(selected_cols.len());
        for &col_idx in selected_cols {
            let col = dense.column(col_idx);
            let n = col.len();
            if n == 0 {
                continue;
            }
            let mean = col.iter().copied().sum::<f64>() / n as f64;
            let var = col
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            let (mean, scale) = if var.is_finite() && var > SCALE_EPS * SCALE_EPS {
                (mean, var.sqrt())
            } else {
                // Leave nearly-constant columns untouched; centering them would collapse
                // the design column to ~0 and change the model rather than just condition it.
                (0.0, 1.0)
            };
            columns.push(LinearColumnConditioning {
                col_idx,
                mean,
                scale,
            });
        }
        Self {
            intercept_idx: design.intercept_range.start,
            columns,
        }
    }

    fn apply_to_design(&self, design: &Array2<f64>) -> Array2<f64> {
        let mut out = design.clone();
        for col in &self.columns {
            {
                let mut dst = out.column_mut(col.col_idx);
                dst -= col.mean;
            }
            if col.scale != 1.0 {
                out.column_mut(col.col_idx).mapv_inplace(|v| v / col.scale);
            }
        }
        out
    }

    fn transform_matrix_columnswith_a(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            let intercept_col = out.column(intercept).to_owned();
            let mut target = out.column_mut(col.col_idx);
            target -= &(intercept_col * col.mean);
            if col.scale != 1.0 {
                target.mapv_inplace(|v| v / col.scale);
            }
        }
        out
    }

    fn transform_matrixrowswith_a_transpose(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            let interceptrow = out.row(intercept).to_owned();
            let mut target = out.row_mut(col.col_idx);
            target -= &(interceptrow * col.mean);
            if col.scale != 1.0 {
                target.mapv_inplace(|v| v / col.scale);
            }
        }
        out
    }

    fn transform_matrix_columnswith_b(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            let intercept_col = out.column(intercept).to_owned();
            let mut target = out.column_mut(col.col_idx);
            if col.mean != 0.0 {
                target += &(intercept_col * col.mean);
            }
            if col.scale != 1.0 {
                target.mapv_inplace(|v| v * col.scale);
            }
        }
        out
    }

    fn transform_matrixrowswith_b_transpose(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            let interceptrow = out.row(intercept).to_owned();
            let mut target = out.row_mut(col.col_idx);
            if col.mean != 0.0 {
                target += &(interceptrow * col.mean);
            }
            if col.scale != 1.0 {
                target.mapv_inplace(|v| v * col.scale);
            }
        }
        out
    }

    fn transform_penalties_to_internal(&self, penalties: &[Array2<f64>]) -> Vec<Array2<f64>> {
        penalties
            .iter()
            .map(|penalty| {
                let right = self.transform_matrix_columnswith_a(penalty);
                self.transform_matrixrowswith_a_transpose(&right)
            })
            .collect()
    }

    fn backtransform_beta(&self, beta_internal: &Array1<f64>) -> Array1<f64> {
        let mut beta = beta_internal.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            beta[intercept] -= beta_internal[col.col_idx] * col.mean / col.scale;
            beta[col.col_idx] = beta_internal[col.col_idx] / col.scale;
        }
        beta
    }

    fn transform_penalized_hessian_to_original(&self, h_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.transform_matrix_columnswith_b(h_internal);
        self.transform_matrixrowswith_b_transpose(&right)
    }

    fn backtransform_covariance(&self, cov_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.transform_matrix_columnswith_a(cov_internal);
        self.transform_matrixrowswith_a_transpose(&right)
    }

    fn internal_bounds_for(&self, col_idx: usize, min: f64, max: f64) -> (f64, f64) {
        if let Some(col) = self.columns.iter().find(|c| c.col_idx == col_idx) {
            (min * col.scale, max * col.scale)
        } else {
            (min, max)
        }
    }
}

fn cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += values[i].exp();
        out[i] = sign * run;
    }
    out
}

fn second_cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let first = cumulative_exp(values, sign);
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += first[i];
        out[i] = run;
    }
    out
}

fn cumulative_sum_transform_matrix(dim: usize, order: usize, sign: f64) -> Array2<f64> {
    let mut t = Array2::<f64>::eye(dim);
    for _ in 0..order {
        let mut next = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..=i {
                next[[i, j]] = 1.0;
            }
        }
        t = t.dot(&next);
    }
    if sign < 0.0 {
        t.mapv_inplace(|v| -v);
    }
    t
}

fn shape_order_and_sign(shape: ShapeConstraint) -> Option<(usize, f64)> {
    match shape {
        ShapeConstraint::None => None,
        ShapeConstraint::MonotoneIncreasing => Some((1, 1.0)),
        ShapeConstraint::MonotoneDecreasing => Some((1, -1.0)),
        ShapeConstraint::Convex => Some((2, 1.0)),
        ShapeConstraint::Concave => Some((2, -1.0)),
    }
}

fn shape_lower_bounds_local(shape: ShapeConstraint, dim: usize) -> Option<Array1<f64>> {
    let (order, _) = shape_order_and_sign(shape)?;
    let mut lb = Array1::<f64>::from_elem(dim, f64::NEG_INFINITY);
    for j in order..dim {
        lb[j] = 0.0;
    }
    Some(lb)
}

fn shape_supports_basis(term: &SmoothTermSpec) -> bool {
    !matches!(term.basis, SmoothBasisSpec::TensorBSpline { .. })
}

fn freeze_raw_spatial_metadata(metadata: BasisMetadata, raw_cols: usize) -> BasisMetadata {
    match metadata {
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            identifiability_transform: None,
            input_scales,
        } => BasisMetadata::ThinPlate {
            centers,
            length_scale,
            identifiability_transform: Some(Array2::eye(raw_cols)),
            input_scales,
        },
        BasisMetadata::Duchon {
            centers,
            length_scale,
            power,
            nullspace_order,
            identifiability_transform: None,
            input_scales,
            aniso_log_scales,
        } => BasisMetadata::Duchon {
            centers,
            length_scale,
            power,
            nullspace_order,
            identifiability_transform: Some(Array2::eye(raw_cols)),
            input_scales,
            aniso_log_scales,
        },
        other => other,
    }
}

fn matern_operator_penalty_triplet_from_metadata(
    metadata: &BasisMetadata,
) -> Result<(Vec<Array2<f64>>, Vec<usize>, Vec<PenaltyInfo>), BasisError> {
    let BasisMetadata::Matern {
        centers,
        length_scale,
        nu,
        include_intercept,
        identifiability_transform,
        aniso_log_scales,
        ..
    } = metadata
    else {
        return Err(BasisError::InvalidInput(
            "Matérn operator penalties require Matérn metadata".to_string(),
        ));
    };
    let ops = build_matern_collocation_operator_matrices(
        centers.view(),
        None,
        *length_scale,
        *nu,
        *include_intercept,
        identifiability_transform.as_ref().map(|z| z.view()),
        aniso_log_scales.as_deref(),
    )?;
    let mut candidates = Vec::with_capacity(3);
    for (raw, source) in [
        (ops.d0.t().dot(&ops.d0), PenaltySource::OperatorMass),
        (ops.d1.t().dot(&ops.d1), PenaltySource::OperatorTension),
        (ops.d2.t().dot(&ops.d2), PenaltySource::OperatorStiffness),
    ] {
        let sym = (&raw + &raw.t()) * 0.5;
        let (matrix, normalization_scale) = normalize_penalty_in_constrained_space(&sym);
        candidates.push(PenaltyCandidate {
            matrix,
            nullspace_dim_hint: 0,
            source,
            normalization_scale,
        });
    }
    filter_active_penalty_candidates(candidates)
}

fn shape_uses_box_reparameterization(basis: &SmoothBasisSpec) -> bool {
    matches!(basis, SmoothBasisSpec::BSpline1D { .. })
}

fn build_shape_constraint_grid_1d(x: ArrayView1<'_, f64>) -> Result<Array1<f64>, BasisError> {
    if x.is_empty() {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires non-empty covariate values".to_string(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires finite covariate values".to_string(),
        ));
    }

    let mut x_sorted: Vec<f64> = x.iter().copied().collect();
    x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut x_unique: Vec<f64> = Vec::with_capacity(x_sorted.len());
    let mut last: Option<f64> = None;
    for v in x_sorted {
        let take = match last {
            None => true,
            Some(prev) => (v - prev).abs() > 1e-12 * prev.abs().max(v.abs()).max(1.0),
        };
        if take {
            x_unique.push(v);
            last = Some(v);
        }
    }
    if x_unique.len() < 2 {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires at least two unique covariate values".to_string(),
        ));
    }

    let min_x = x_unique[0];
    let max_x = *x_unique
        .last()
        .expect("x_unique has at least two elements by construction");
    if (max_x - min_x).abs() <= 1e-12 {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires non-degenerate covariate range".to_string(),
        ));
    }

    let target_points = x_unique.len().clamp(96, 320);
    let mut grid = Array1::<f64>::zeros(target_points);
    let denom = (target_points - 1) as f64;
    for i in 0..target_points {
        let t = i as f64 / denom;
        grid[i] = min_x + t * (max_x - min_x);
    }
    Ok(grid)
}

fn build_shape_constraint_design_1d(
    data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
    metadata: &BasisMetadata,
    axis_col: usize,
) -> Result<(Array1<f64>, Array2<f64>), BasisError> {
    let x_grid = build_shape_constraint_grid_1d(data.column(axis_col))?;
    let grid_2d = x_grid
        .clone()
        .into_shape_with_order((x_grid.len(), 1))
        .map_err(|e| {
            BasisError::InvalidInput(format!(
                "failed to construct 1D shape grid matrix for term '{}': {e}",
                term.name
            ))
        })?;

    let design = match (&term.basis, metadata) {
        (
            SmoothBasisSpec::BSpline1D { spec, .. },
            BasisMetadata::BSpline1D {
                knots,
                identifiability_transform,
            },
        ) => {
            let evalspec = BSplineBasisSpec {
                degree: spec.degree,
                penalty_order: spec.penalty_order,
                knotspec: BSplineKnotSpec::Provided(knots.clone()),
                double_penalty: false,
                identifiability: identifiability_transform
                    .as_ref()
                    .map(|z| BSplineIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    })
                    .unwrap_or(BSplineIdentifiability::None),
            };
            build_bspline_basis_1d(x_grid.view(), &evalspec)?.design
        }
        (
            SmoothBasisSpec::ThinPlate { .. },
            BasisMetadata::ThinPlate {
                centers,
                length_scale,
                identifiability_transform,
                ..
            },
        ) => {
            let evalspec = ThinPlateBasisSpec {
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                double_penalty: false,
                identifiability: identifiability_transform
                    .as_ref()
                    .map(|z| SpatialIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    })
                    .unwrap_or(SpatialIdentifiability::None),
            };
            build_thin_plate_basis(grid_2d.view(), &evalspec)?.design
        }
        (
            SmoothBasisSpec::Matern { .. },
            BasisMetadata::Matern {
                centers,
                length_scale,
                nu,
                include_intercept,
                identifiability_transform,
                aniso_log_scales,
                ..
            },
        ) => {
            let ident = identifiability_transform
                .as_ref()
                .map(|z| MaternIdentifiability::FrozenTransform {
                    transform: z.clone(),
                })
                .unwrap_or(MaternIdentifiability::None);
            let evalspec = MaternBasisSpec {
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                nu: *nu,
                include_intercept: *include_intercept,
                double_penalty: false,
                identifiability: ident,
                aniso_log_scales: aniso_log_scales.clone(),
            };
            build_matern_basis(grid_2d.view(), &evalspec)?.design
        }
        (
            SmoothBasisSpec::Duchon { spec, .. },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                power,
                nullspace_order,
                identifiability_transform,
                aniso_log_scales,
                ..
            },
        ) => {
            let evalspec = DuchonBasisSpec {
                center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                length_scale: *length_scale,
                power: *power,
                nullspace_order: *nullspace_order,
                identifiability: identifiability_transform
                    .as_ref()
                    .map(|z| SpatialIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    })
                    .unwrap_or_else(|| spec.identifiability.clone()),
                aniso_log_scales: aniso_log_scales.clone(),
            };
            build_duchon_basis(grid_2d.view(), &evalspec)?.design
        }
        _ => {
            return Err(BasisError::InvalidInput(format!(
                "shape-constraint grid reconstruction metadata mismatch for term '{}'",
                term.name
            )));
        }
    };

    Ok((x_grid, design))
}

fn build_shape_linear_constraints_1d(
    x: ArrayView1<'_, f64>,
    design_local: ArrayView2<'_, f64>,
    shape: ShapeConstraint,
) -> Result<Option<LinearInequalityConstraints>, BasisError> {
    let (order, sign) = match shape_order_and_sign(shape) {
        Some(v) => v,
        None => return Ok(None),
    };
    let n = x.len();
    let p = design_local.ncols();
    if n == 0 || p == 0 {
        return Ok(None);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "shape-constrained smooth requires finite covariate values".to_string(),
        ));
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));

    let x_scale = x.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    let x_tol = 1e-12 * x_scale;
    let mut collapsedrows: Vec<Array1<f64>> = Vec::new();
    let mut group_sum = Array1::<f64>::zeros(p);
    let mut group_count = 0usize;
    let mut last_x: Option<f64> = None;
    for &r in &idx {
        let xr = x[r];
        let start_new = match last_x {
            None => false,
            Some(prev) => (xr - prev).abs() > x_tol,
        };
        if start_new {
            if group_count > 0 {
                collapsedrows.push(group_sum.mapv(|v| v / group_count as f64));
            }
            group_sum.fill(0.0);
            group_count = 0;
        }
        group_sum += &design_local.row(r).to_owned();
        group_count += 1;
        last_x = Some(xr);
    }
    if group_count > 0 {
        collapsedrows.push(group_sum.mapv(|v| v / group_count as f64));
    }

    let m = collapsedrows.len();
    if m <= order {
        return Err(BasisError::InvalidInput(format!(
            "shape-constrained smooth requires at least {} unique covariate locations; found {}",
            order + 1,
            m
        )));
    }

    let q_raw = m - order;
    let mut arows: Vec<Array1<f64>> = Vec::with_capacity(q_raw);
    for i in 0..q_raw {
        let row = if order == 1 {
            &collapsedrows[i + 1] - &collapsedrows[i]
        } else {
            &collapsedrows[i + 2] - &collapsedrows[i + 1].mapv(|v| 2.0 * v) + &collapsedrows[i]
        };
        let mut row_signed = row;
        if sign < 0.0 {
            row_signed.mapv_inplace(|v| -v);
        }
        let norm = row_signed.dot(&row_signed).sqrt();
        if norm > 1e-12 {
            arows.push(row_signed);
        }
    }
    if arows.is_empty() {
        return Ok(None);
    }

    let mut a = Array2::<f64>::zeros((arows.len(), p));
    for (i, row) in arows.iter().enumerate() {
        a.row_mut(i).assign(row);
    }
    let b = Array1::<f64>::zeros(a.nrows());
    Ok(Some(LinearInequalityConstraints { a, b }))
}

fn linear_constraints_from_lower_bounds_global(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    let rows: Vec<usize> = (0..lower_bounds.len())
        .filter(|&i| lower_bounds[i].is_finite())
        .collect();
    if rows.is_empty() {
        return None;
    }
    let p = lower_bounds.len();
    let mut a = Array2::<f64>::zeros((rows.len(), p));
    let mut b = Array1::<f64>::zeros(rows.len());
    for (r, &idx) in rows.iter().enumerate() {
        a[[r, idx]] = 1.0;
        b[r] = lower_bounds[idx];
    }
    Some(LinearInequalityConstraints { a, b })
}

fn merge_linear_constraints_global(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    match (first, second) {
        (None, None) => None,
        (Some(c), None) | (None, Some(c)) => Some(c),
        (Some(a), Some(b)) => {
            if a.a.ncols() != b.a.ncols() {
                return None;
            }
            let m1 = a.a.nrows();
            let m2 = b.a.nrows();
            let p = a.a.ncols();
            let mut mat = Array2::<f64>::zeros((m1 + m2, p));
            mat.slice_mut(s![0..m1, ..]).assign(&a.a);
            mat.slice_mut(s![m1..(m1 + m2), ..]).assign(&b.a);
            let mut rhs = Array1::<f64>::zeros(m1 + m2);
            rhs.slice_mut(s![0..m1]).assign(&a.b);
            rhs.slice_mut(s![m1..(m1 + m2)]).assign(&b.b);
            Some(LinearInequalityConstraints { a: mat, b: rhs })
        }
    }
}

fn normalize_penalty_in_constrained_space(matrix: &Array2<f64>) -> (Array2<f64>, f64) {
    // Constrained-space normalization:
    //   c = ||S_con||_F,  S_tilde = S_con / c.
    // This is the only normalization coherent with a REML objective that is
    // evaluated entirely in constrained coordinates.
    let c = matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    if c.is_finite() && c > 0.0 {
        (matrix.mapv(|v| v / c), c)
    } else {
        (matrix.clone(), 1.0)
    }
}

fn build_tensor_bspline_basis(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    spec: &TensorBSplineSpec,
) -> Result<BasisBuildResult, BasisError> {
    if feature_cols.is_empty() {
        return Err(BasisError::InvalidInput(
            "TensorBSpline requires at least one feature column".to_string(),
        ));
    }
    if feature_cols.len() != spec.marginalspecs.len() {
        return Err(BasisError::DimensionMismatch(format!(
            "TensorBSpline feature/spec mismatch: feature_cols={}, marginalspecs={}",
            feature_cols.len(),
            spec.marginalspecs.len()
        )));
    }
    let p = data.ncols();
    for &c in feature_cols {
        if c >= p {
            return Err(BasisError::DimensionMismatch(format!(
                "tensor feature column {c} is out of bounds for data with {p} columns"
            )));
        }
    }

    let mut marginal_knots = Vec::<Array1<f64>>::with_capacity(feature_cols.len());
    let mut marginal_degrees = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginalnum_basis = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginal_penalties = Vec::<Array2<f64>>::with_capacity(feature_cols.len());
    let mut marginal_designs = Vec::<Array2<f64>>::with_capacity(feature_cols.len());

    // Reuse the robust 1D builder to ensure the same knot validation and
    // marginal difference-penalty construction as standalone smooth terms.
    for (dim, (&col, marginalspec)) in feature_cols
        .iter()
        .zip(spec.marginalspecs.iter())
        .enumerate()
    {
        // Tensor basis uses raw marginal knot-product columns. Applying 1D
        // identifiability constraints here would change marginal penalty sizes
        // without changing the tensor design construction, causing dimension
        // mismatch. Keep marginal builders unconstrained at this stage.
        let mut marginal_unconstrained = marginalspec.clone();
        marginal_unconstrained.identifiability = BSplineIdentifiability::None;
        let built = build_bspline_basis_1d(data.column(col), &marginal_unconstrained)?;
        let knots = match built.metadata {
            BasisMetadata::BSpline1D { knots, .. } => knots,
            _ => {
                return Err(BasisError::InvalidInput(format!(
                    "internal TensorBSpline error at dim {dim}: expected BSpline1D metadata"
                )));
            }
        };
        marginal_knots.push(knots);
        marginal_degrees.push(marginalspec.degree);
        marginalnum_basis.push(built.design.ncols());
        marginal_designs.push(built.design);
        marginal_penalties.push(
            built
                .penalties
                .first()
                .ok_or_else(|| {
                    BasisError::InvalidInput(format!(
                        "internal TensorBSpline error at dim {dim}: missing marginal penalty"
                    ))
                })?
                .clone(),
        );
        built.nullspace_dims.first().ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "internal TensorBSpline error at dim {dim}: missing marginal nullspace dim"
            ))
        })?;
    }

    let mut design = tensor_product_design_from_marginals(&marginal_designs)?;

    let total_cols = design.ncols();
    let mut candidates = Vec::<PenaltyCandidate>::with_capacity(
        marginal_penalties.len() + if spec.double_penalty { 1 } else { 0 },
    );

    for dim in 0..marginal_penalties.len() {
        let mut s_dim = Array2::<f64>::eye(1);
        for (j, &qj) in marginalnum_basis.iter().enumerate() {
            let factor = if j == dim {
                marginal_penalties[j].clone()
            } else {
                Array2::<f64>::eye(qj)
            };
            s_dim = kronecker_product(&s_dim, &factor);
        }

        candidates.push(PenaltyCandidate {
            matrix: s_dim,
            nullspace_dim_hint: 0,
            source: PenaltySource::TensorMarginal { dim },
            normalization_scale: 1.0,
        });
    }

    if spec.double_penalty {
        candidates.push(PenaltyCandidate {
            matrix: Array2::<f64>::eye(total_cols),
            nullspace_dim_hint: 0,
            source: PenaltySource::TensorGlobalRidge,
            normalization_scale: 1.0,
        });
    }

    let z_opt = match &spec.identifiability {
        TensorBSplineIdentifiability::None => None,
        TensorBSplineIdentifiability::SumToZero => {
            if total_cols < 2 {
                return Err(BasisError::InvalidInput(
                    "TensorBSpline requires at least 2 basis coefficients to enforce sum-to-zero identifiability".to_string(),
                ));
            }
            let (_, z) = apply_sum_to_zero_constraint(design.view(), None)?;
            Some(z)
        }
        TensorBSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != total_cols {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen tensor identifiability transform mismatch: design has {} columns but transform has {} rows",
                    total_cols,
                    transform.nrows()
                )));
            }
            Some(transform.clone())
        }
    };

    if let Some(z) = z_opt.as_ref() {
        design = design.dot(z);
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                let zt_s = z.t().dot(&candidate.matrix);
                let matrix = zt_s.dot(z);
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: estimate_penalty_nullity(&matrix)?,
                    matrix,
                    source: candidate.source,
                    normalization_scale: candidate.normalization_scale * c_new,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    } else {
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: estimate_penalty_nullity(&candidate.matrix)?,
                    ..candidate
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    }

    let (penalties, nullspace_dims, penaltyinfo) = filter_active_penalty_candidates(candidates)?;

    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.to_vec(),
            knots: marginal_knots,
            degrees: marginal_degrees,
            identifiability_transform: z_opt,
        },
    })
}

fn tensor_product_design_from_marginals(
    marginal_designs: &[Array2<f64>],
) -> Result<Array2<f64>, BasisError> {
    if marginal_designs.is_empty() {
        return Err(BasisError::InvalidInput(
            "TensorBSpline requires at least one marginal basis".to_string(),
        ));
    }
    let n = marginal_designs[0].nrows();
    for (i, b) in marginal_designs.iter().enumerate().skip(1) {
        if b.nrows() != n {
            return Err(BasisError::DimensionMismatch(format!(
                "tensor marginal row mismatch at dim {i}: expected {n}, got {}",
                b.nrows()
            )));
        }
    }
    let total_cols = marginal_designs.iter().try_fold(1usize, |acc, b| {
        acc.checked_mul(b.ncols())
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))
    })?;
    let mut design = Array2::<f64>::zeros((n, total_cols));
    for i in 0..n {
        let mut rowvals = vec![1.0_f64];
        for b in marginal_designs {
            let q = b.ncols();
            let mut next = vec![0.0_f64; rowvals.len() * q];
            for (a_idx, &aval) in rowvals.iter().enumerate() {
                for col in 0..q {
                    next[a_idx * q + col] = aval * b[[i, col]];
                }
            }
            rowvals = next;
        }
        for (j, &v) in rowvals.iter().enumerate() {
            design[[i, j]] = v;
        }
    }
    Ok(design)
}

fn build_random_effect_block(
    data: ArrayView2<'_, f64>,
    spec: &RandomEffectTermSpec,
) -> Result<RandomEffectBlock, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    if spec.feature_col >= p {
        return Err(BasisError::DimensionMismatch(format!(
            "random-effect term '{}' feature column {} out of bounds for {} columns",
            spec.name, spec.feature_col, p
        )));
    }

    let col = data.column(spec.feature_col);
    if col.iter().any(|v| !v.is_finite()) {
        return Err(BasisError::InvalidInput(format!(
            "random-effect term '{}' contains non-finite group values",
            spec.name
        )));
    }

    let mut kept_levels: Vec<u64> = if let Some(levels) = spec.frozen_levels.as_ref() {
        if levels.is_empty() {
            return Err(BasisError::InvalidInput(format!(
                "random-effect term '{}' has empty frozen_levels",
                spec.name
            )));
        }
        levels.clone()
    } else {
        let mut levels_set = BTreeSet::<u64>::new();
        for &v in col {
            levels_set.insert(v.to_bits());
        }
        if levels_set.is_empty() {
            return Err(BasisError::InvalidInput(format!(
                "random-effect term '{}' has no observed levels",
                spec.name
            )));
        }
        let levels: Vec<u64> = levels_set.into_iter().collect();
        let start_idx = if spec.drop_first_level && levels.len() > 1 {
            1usize
        } else {
            0usize
        };
        levels[start_idx..].to_vec()
    };
    kept_levels.sort_unstable();
    kept_levels.dedup();

    if kept_levels.is_empty() {
        return Err(BasisError::InvalidInput(format!(
            "random-effect term '{}' drops all levels; keep at least one level",
            spec.name
        )));
    }

    let q = kept_levels.len();
    let mut group_ids = Vec::with_capacity(n);
    for &v in col {
        let bits = v.to_bits();
        group_ids.push(kept_levels.binary_search(&bits).ok());
    }

    Ok(RandomEffectBlock {
        name: spec.name.clone(),
        group_ids,
        num_groups: q,
        kept_levels,
    })
}

impl SmoothDesign {
    /// Map an unconstrained term coefficient vector to its constrained shape space.
    /// This is useful for nonlinear fits that optimize unconstrained parameters.
    pub fn map_term_coefficients(
        unconstrained: &Array1<f64>,
        shape: ShapeConstraint,
    ) -> Result<Array1<f64>, BasisError> {
        if unconstrained.is_empty() {
            return Err(BasisError::InvalidInput(
                "unconstrained coefficient vector cannot be empty".to_string(),
            ));
        }
        let mapped = match shape {
            ShapeConstraint::None => unconstrained.clone(),
            ShapeConstraint::MonotoneIncreasing => cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::MonotoneDecreasing => cumulative_exp(unconstrained, -1.0),
            ShapeConstraint::Convex => second_cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::Concave => second_cumulative_exp(unconstrained, -1.0),
        };
        Ok(mapped)
    }
}

pub fn build_smooth_design(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
) -> Result<RawSmoothDesign, BasisError> {
    let mut ws = crate::basis::BasisWorkspace::new();
    build_smooth_design_withworkspace(data, terms, &mut ws)
}

/// Like `build_smooth_design` but reuses a persistent workspace for
/// distance-matrix caching across repeated κ-proposal basis rebuilds.
pub fn build_smooth_design_withworkspace(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<RawSmoothDesign, BasisError> {
    let n = data.nrows();
    let mut local_designs = Vec::<Array2<f64>>::with_capacity(terms.len());
    let mut local_penalties = Vec::<Vec<Array2<f64>>>::with_capacity(terms.len());
    let mut local_nullspaces = Vec::<Vec<usize>>::with_capacity(terms.len());
    let mut local_penaltyinfo = Vec::<Vec<PenaltyInfo>>::with_capacity(terms.len());
    let mut local_pre_dropped_penaltyinfo = Vec::<Vec<PenaltyInfo>>::with_capacity(terms.len());
    let mut local_metadata = Vec::<BasisMetadata>::with_capacity(terms.len());
    let mut local_dims = Vec::<usize>::with_capacity(terms.len());
    let mut local_linear_constraints =
        Vec::<Option<LinearInequalityConstraints>>::with_capacity(terms.len());
    let mut local_box_reparam = Vec::<bool>::with_capacity(terms.len());
    let mut local_kronecker_factored =
        Vec::<Option<KroneckerFactoredBasis>>::with_capacity(terms.len());

    for term in terms {
        if term.shape != ShapeConstraint::None && !shape_supports_basis(term) {
            return Err(BasisError::InvalidInput(format!(
                "ShapeConstraint::{:?} is unsupported for term '{}'",
                term.shape, term.name
            )));
        }
        let mut shape_axis_col: Option<usize> = None;
        let mut built: BasisBuildResult = match &term.basis {
            SmoothBasisSpec::BSpline1D { feature_col, spec } => {
                if *feature_col >= data.ncols() {
                    return Err(BasisError::DimensionMismatch(format!(
                        "term '{}' feature column {} out of bounds for {} columns",
                        term.name,
                        feature_col,
                        data.ncols()
                    )));
                }
                let mut spec_local = spec.clone();
                if term.shape != ShapeConstraint::None {
                    // Shape-constrained B-splines are anchored by construction.
                    // Sum-to-zero side constraints conflict with monotonic/convex cones.
                    spec_local.identifiability = BSplineIdentifiability::None;
                }
                build_bspline_basis_1d(data.column(*feature_col), &spec_local)?
            }
            SmoothBasisSpec::ThinPlate {
                feature_cols,
                spec,
                input_scales,
            } => {
                if term.shape != ShapeConstraint::None {
                    if feature_cols.len() != 1 {
                        return Err(BasisError::InvalidInput(format!(
                            "ShapeConstraint::{:?} for term '{}' on ThinPlate basis requires exactly 1 feature axis; found {}",
                            term.shape,
                            term.name,
                            feature_cols.len()
                        )));
                    }
                    shape_axis_col = Some(feature_cols[0]);
                }
                let mut x = select_columns(data, feature_cols)?;
                // Auto-standardize multivariate inputs: use stored scales (prediction)
                // or compute fresh ones (training).
                let scales = if let Some(s) = input_scales {
                    apply_input_standardization(&mut x, s);
                    Some(s.clone())
                } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                    apply_input_standardization(&mut x, &s);
                    Some(s)
                } else {
                    None
                };
                let mut spec_local = spec.clone();
                if matches!(
                    spec_local.identifiability,
                    SpatialIdentifiability::OrthogonalToParametric
                        | SpatialIdentifiability::FrozenTransform { .. }
                ) {
                    spec_local.identifiability = SpatialIdentifiability::None;
                }
                let mut result = build_thin_plate_basis(x.view(), &spec_local).map_err(|err| {
                    rewrite_thin_plate_knots_error(err, &term.name, feature_cols.len(), spec)
                })?;
                // Inject input scales into metadata for downstream storage.
                if let BasisMetadata::ThinPlate { input_scales, .. } = &mut result.metadata {
                    *input_scales = scales;
                }
                result
            }
            SmoothBasisSpec::Matern {
                feature_cols,
                spec,
                input_scales,
            } => {
                if term.shape != ShapeConstraint::None {
                    if feature_cols.len() != 1 {
                        return Err(BasisError::InvalidInput(format!(
                            "ShapeConstraint::{:?} for term '{}' on Matern basis requires exactly 1 feature axis; found {}",
                            term.shape,
                            term.name,
                            feature_cols.len()
                        )));
                    }
                    shape_axis_col = Some(feature_cols[0]);
                }
                let mut x = select_columns(data, feature_cols)?;
                let scales = if let Some(s) = input_scales {
                    apply_input_standardization(&mut x, s);
                    Some(s.clone())
                } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                    apply_input_standardization(&mut x, &s);
                    Some(s)
                } else {
                    None
                };
                let mut result = build_matern_basiswithworkspace(x.view(), spec, workspace)?;
                if let BasisMetadata::Matern { input_scales, .. } = &mut result.metadata {
                    *input_scales = scales;
                }
                result
            }
            SmoothBasisSpec::Duchon {
                feature_cols,
                spec,
                input_scales,
            } => {
                if term.shape != ShapeConstraint::None {
                    if feature_cols.len() != 1 {
                        return Err(BasisError::InvalidInput(format!(
                            "ShapeConstraint::{:?} for term '{}' on Duchon basis requires exactly 1 feature axis; found {}",
                            term.shape,
                            term.name,
                            feature_cols.len()
                        )));
                    }
                    shape_axis_col = Some(feature_cols[0]);
                }
                let mut x = select_columns(data, feature_cols)?;
                let scales = if let Some(s) = input_scales {
                    apply_input_standardization(&mut x, s);
                    Some(s.clone())
                } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                    apply_input_standardization(&mut x, &s);
                    Some(s)
                } else {
                    None
                };
                let mut spec_local = spec.clone();
                if matches!(
                    spec_local.identifiability,
                    SpatialIdentifiability::OrthogonalToParametric
                        | SpatialIdentifiability::FrozenTransform { .. }
                ) {
                    spec_local.identifiability = SpatialIdentifiability::None;
                }
                let mut result = build_duchon_basiswithworkspace(x.view(), &spec_local, workspace)?;
                if let BasisMetadata::Duchon { input_scales, .. } = &mut result.metadata {
                    *input_scales = scales;
                }
                result
            }
            SmoothBasisSpec::TensorBSpline { feature_cols, spec } => {
                build_tensor_bspline_basis(data, feature_cols, spec)?
            }
        };

        match &term.basis {
            SmoothBasisSpec::Matern { .. } => {
                let (penalties, nullspace_dims, penaltyinfo) =
                    matern_operator_penalty_triplet_from_metadata(&built.metadata)?;
                built.penalties = penalties;
                built.nullspace_dims = nullspace_dims;
                built.penaltyinfo = penaltyinfo;
            }
            _ => {}
        }

        let p_local = built.design.ncols();
        let mut metadata = built.metadata.clone();
        // Extract factored Kronecker representation before consuming fields.
        // Invalidate it if shape transforms will be applied (they break structure).
        let kron_factored = if term.shape == ShapeConstraint::None {
            built.kronecker_factored
        } else {
            None
        };
        let mut design_t = built.design;
        let mut penalties_t: Vec<Array2<f64>> = built.penalties;
        if matches!(
            spatial_identifiability_policy(term),
            Some(SpatialIdentifiability::OrthogonalToParametric)
        ) {
            metadata = freeze_raw_spatial_metadata(metadata, design_t.ncols());
        }

        let active_penaltyinfo_t = built
            .penaltyinfo
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect::<Vec<_>>();
        let pre_dropped_penaltyinfo_t = built
            .penaltyinfo
            .iter()
            .filter(|info| !info.active)
            .cloned()
            .collect::<Vec<_>>();
        let use_box_reparam =
            term.shape != ShapeConstraint::None && shape_uses_box_reparameterization(&term.basis);
        if let Some((order, sign)) = shape_order_and_sign(term.shape)
            && use_box_reparam
        {
            let t = cumulative_sum_transform_matrix(p_local, order, sign);
            design_t = design_t.dot(&t);
            penalties_t = penalties_t
                .into_iter()
                .map(|s_local| {
                    // Congruence transform preserves PSD:
                    //   S_new = T^T S T.
                    let tt_s = t.t().dot(&s_local);
                    tt_s.dot(&t)
                })
                .collect();
        }
        if penalties_t.len() != active_penaltyinfo_t.len() {
            return Err(BasisError::InvalidInput(format!(
                "internal penalty metadata mismatch for term '{}': active penalties={}, active infos={}",
                term.name,
                penalties_t.len(),
                active_penaltyinfo_t.len()
            )));
        }
        let penalty_candidates = penalties_t
            .into_iter()
            .zip(active_penaltyinfo_t.into_iter())
            .map(|(matrix, info)| -> Result<PenaltyCandidate, BasisError> {
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: estimate_penalty_nullity(&matrix)?,
                    matrix,
                    source: info.source,
                    normalization_scale: info.normalization_scale * c_new,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (penalties_t, nullspaces_t, penaltyinfo_t) =
            filter_active_penalty_candidates(penalty_candidates)?;
        let linear_constraints_local = if term.shape != ShapeConstraint::None && !use_box_reparam {
            let axis = shape_axis_col.ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "internal shape-constraint axis missing for term '{}'",
                    term.name
                ))
            })?;
            let (x_shape_eval, design_shape_eval) =
                build_shape_constraint_design_1d(data, term, &metadata, axis)?;
            build_shape_linear_constraints_1d(
                x_shape_eval.view(),
                design_shape_eval.view(),
                term.shape,
            )?
        } else {
            None
        };

        local_dims.push(p_local);
        local_designs.push(design_t);
        local_penalties.push(penalties_t);
        local_nullspaces.push(nullspaces_t);
        local_penaltyinfo.push(penaltyinfo_t);
        local_pre_dropped_penaltyinfo.push(pre_dropped_penaltyinfo_t);
        local_metadata.push(metadata);
        local_linear_constraints.push(linear_constraints_local);
        local_box_reparam.push(use_box_reparam);
        local_kronecker_factored.push(kron_factored);
    }

    let total_p: usize = local_dims.iter().sum();
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(terms.len());
    let mut penalties_global = Vec::<Array2<f64>>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut penaltyinfo_global = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo_global = Vec::<DroppedPenaltyBlockInfo>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintsrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for (idx, term) in terms.iter().enumerate() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;
        let lb_local = if local_box_reparam[idx] {
            shape_lower_bounds_local(term.shape, p_local)
        } else {
            None
        };

        let activeinfos = local_penaltyinfo[idx]
            .iter()
            .filter(|info| info.active)
            .collect::<Vec<_>>();
        if activeinfos.len() != local_penalties[idx].len() {
            return Err(BasisError::InvalidInput(format!(
                "internal penalty info mismatch for term '{}': activeinfos={}, penalties={}",
                term.name,
                activeinfos.len(),
                local_penalties[idx].len()
            )));
        }
        for ((s_local, &ns), info) in local_penalties[idx]
            .iter()
            .zip(local_nullspaces[idx].iter())
            .zip(activeinfos.into_iter())
        {
            let global_index = penalties_global.len();
            let mut s_global = Array2::<f64>::zeros((total_p, total_p));
            s_global
                .slice_mut(s![col_start..col_end, col_start..col_end])
                .assign(s_local);
            penalties_global.push(s_global);
            nullspace_dims_global.push(ns);
            let mut penalty = info.clone();
            penalty.nullspace_dim_hint = ns;
            penaltyinfo_global.push(PenaltyBlockInfo {
                global_index,
                termname: Some(term.name.clone()),
                penalty,
            });
        }
        for info in local_penaltyinfo[idx].iter().filter(|info| !info.active) {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(term.name.clone()),
                penalty: info.clone(),
            });
        }
        for info in &local_pre_dropped_penaltyinfo[idx] {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(term.name.clone()),
                penalty: info.clone(),
            });
        }

        terms_out.push(SmoothTerm {
            name: term.name.clone(),
            coeff_range: col_start..col_end,
            shape: term.shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            penaltyinfo_local: local_penaltyinfo[idx].clone(),
            metadata: local_metadata[idx].clone(),
            lower_bounds_local: lb_local.clone(),
            linear_constraints_local: local_linear_constraints[idx].clone(),
            kronecker_factored: local_kronecker_factored[idx].take(),
        });
        if let Some(lin_local) = &local_linear_constraints[idx] {
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![col_start..col_end])
                    .assign(&lin_local.a.row(r));
                linear_constraintsrows.push(row);
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = lb_local {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(&lb_local);
            any_bounds = true;
        }

        col_start = col_end;
    }

    debug_assert_eq!(
        penalties_global.len(),
        nullspace_dims_global.len(),
        "global smooth penalty/nullspace bookkeeping diverged"
    );
    debug_assert_eq!(
        penalties_global.len(),
        penaltyinfo_global.len(),
        "global smooth penalty metadata bookkeeping diverged"
    );

    Ok(RawSmoothDesign {
        term_designs: local_designs,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        penaltyinfo: penaltyinfo_global,
        dropped_penaltyinfo: dropped_penaltyinfo_global,
        terms: terms_out,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints: if linear_constraintsrows.is_empty() {
            None
        } else {
            let mut a = Array2::<f64>::zeros((linear_constraintsrows.len(), total_p));
            for (i, row) in linear_constraintsrows.iter().enumerate() {
                a.row_mut(i).assign(row);
            }
            Some(LinearInequalityConstraints {
                a,
                b: Array1::from_vec(linear_constraints_b),
            })
        },
    })
}

pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();
    let smooth_raw = build_smooth_design(data, &spec.smooth_terms)?;
    let random_blocks: Vec<RandomEffectBlock> = spec
        .random_effect_terms
        .iter()
        .map(|term| build_random_effect_block(data, term))
        .collect::<Result<_, _>>()?;

    for linear in &spec.linear_terms {
        if linear.feature_col >= p_data {
            return Err(BasisError::DimensionMismatch(format!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name, linear.feature_col, p_data
            )));
        }
    }

    let smooth = apply_spatial_orthogonality_to_parametric(
        smooth_raw,
        data,
        &spec.linear_terms,
        &spec.smooth_terms,
    )?;

    let p_intercept = 1usize;
    let p_lin = spec.linear_terms.len();
    let p_rand: usize = random_blocks.iter().map(|b| b.num_groups).sum();
    let p_smooth = smooth.total_smooth_cols();
    let p_total = p_intercept + p_lin + p_rand + p_smooth;

    // Build the dense core: intercept + linear + smooth columns.
    let p_dense = p_intercept + p_lin + p_smooth;
    let mut dense_core = Array2::<f64>::zeros((n, p_dense));
    dense_core.column_mut(0).fill(1.0);

    let mut linear_ranges = Vec::<(String, Range<usize>)>::with_capacity(p_lin);
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        dense_core
            .column_mut(col)
            .assign(&data.column(linear.feature_col));
        // Column ranges are in the global (full) coordinate system:
        // [intercept | linear | random_effects | smooth]
        linear_ranges.push((linear.name.clone(), col..(col + 1)));
    }
    if p_smooth > 0 {
        let mut smooth_col = p_intercept + p_lin;
        for td in &smooth.term_designs {
            let k = td.ncols();
            dense_core
                .slice_mut(s![.., smooth_col..(smooth_col + k)])
                .assign(td);
            smooth_col += k;
        }
    }

    // Track random-effect column ranges in the global coordinate system.
    // Global layout: [intercept(1) | linear(p_lin) | RE_0(q0) | RE_1(q1) | … | smooth(p_smooth)]
    let mut random_effect_ranges =
        Vec::<(String, Range<usize>)>::with_capacity(random_blocks.len());
    let mut random_effect_levels =
        Vec::<(String, Vec<u64>)>::with_capacity(random_blocks.len());
    let mut col_cursor = p_intercept + p_lin;
    for block in &random_blocks {
        let q = block.num_groups;
        let end = col_cursor + q;
        random_effect_ranges.push((block.name.clone(), col_cursor..end));
        random_effect_levels.push((block.name.clone(), block.kept_levels.clone()));
        col_cursor = end;
    }

    // Assemble the full DesignMatrix.
    //
    // When random effects are present, use a BlockDesignOperator that composes
    // the dense core (intercept + linear) with O(n) random-effect operators and
    // the dense smooth block.  This avoids materializing the n × q one-hot
    // matrices — the dominant memory cost for biobank-scale random intercepts.
    //
    // When there are no random effects, the dense core already contains all
    // columns and is wrapped directly.
    let design: DesignMatrix = if random_blocks.is_empty() {
        // No RE — dense_core already has [intercept | linear | smooth].
        DesignMatrix::Dense(Arc::new(dense_core))
    } else {
        // Build the block operator.
        //
        // Global column layout:
        //   [intercept + linear] | [RE_0] | [RE_1] | … | [smooth]
        //
        // dense_core has [intercept + linear + smooth] columns.  We split it
        // into two dense blocks (pre-RE and post-RE) with RE operators in between.
        let p_pre_re = p_intercept + p_lin; // intercept + linear columns
        let mut blocks = Vec::<DesignBlock>::new();

        // Block 0: intercept + linear (dense).
        if p_pre_re > 0 {
            let pre_re = dense_core.slice(s![.., ..p_pre_re]).to_owned();
            blocks.push(DesignBlock::Dense(Arc::new(pre_re)));
        }

        // Blocks 1..k: random-effect operators.
        for block in &random_blocks {
            let re_op = RandomEffectOperator::new(
                block.group_ids.clone(),
                block.num_groups,
            );
            blocks.push(DesignBlock::RandomEffect(Arc::new(re_op)));
        }

        // Final block: smooth (dense), if present.
        if p_smooth > 0 {
            let smooth_part = dense_core.slice(s![.., p_pre_re..]).to_owned();
            blocks.push(DesignBlock::Dense(Arc::new(smooth_part)));
        }

        let block_op = BlockDesignOperator::new(blocks).map_err(|e| {
            BasisError::InvalidInput(format!("failed to build block design operator: {e}"))
        })?;
        DesignMatrix::Operator(Arc::new(block_op))
    };

    let mut penalties = Vec::<BlockwisePenalty>::new();
    let mut nullspace_dims = Vec::<usize>::new();
    let mut penaltyinfo = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo = Vec::<DroppedPenaltyBlockInfo>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(p_total, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows = Vec::<Array1<f64>>::new();
    let mut linear_constraint_b = Vec::<f64>::new();

    let mut penalized_linear_cols = Vec::<usize>::new();
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        if let Some(lb) = linear.coefficient_min {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = 1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(lb);
        }
        if let Some(ub) = linear.coefficient_max {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = -1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(-ub);
        }
        if linear.double_penalty {
            penalized_linear_cols.push(col);
        }
    }

    if !penalized_linear_cols.is_empty() {
        // Build a compact penalty covering the range of penalized linear columns.
        let min_col = *penalized_linear_cols.iter().min().unwrap();
        let max_col = *penalized_linear_cols.iter().max().unwrap();
        let block_range = min_col..(max_col + 1);
        let block_size = block_range.len();
        let mut s_local = Array2::<f64>::zeros((block_size, block_size));
        for &col in &penalized_linear_cols {
            let local_idx = col - min_col;
            s_local[[local_idx, local_idx]] = 1.0;
        }
        let global_index = penalties.len();
        penalties.push(BlockwisePenalty::new(block_range, s_local));
        nullspace_dims.push(0);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: Some("linear".to_string()),
            penalty: PenaltyInfo {
                source: PenaltySource::Other("LinearDoublePenaltyGroup".to_string()),
                original_index: 0,
                active: true,
                effective_rank: penalized_linear_cols.len(),
                dropped_reason: None,
                nullspace_dim_hint: 0,
                normalization_scale: 1.0,
                kronecker_factors: None,
            },
        });
    }

    for (re_idx, (name, range)) in random_effect_ranges.iter().enumerate() {
        if range.is_empty() {
            continue;
        }
        let block_size = range.len();
        let global_index = penalties.len();
        penalties.push(BlockwisePenalty::ridge(range.clone(), 1.0));
        nullspace_dims.push(0);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: Some(name.clone()),
            penalty: PenaltyInfo {
                source: PenaltySource::Other(format!("RandomEffectRidge({name})")),
                original_index: re_idx,
                active: true,
                effective_rank: block_size,
                dropped_reason: None,
                nullspace_dim_hint: 0,
                normalization_scale: 1.0,
                kronecker_factors: None,
            },
        });
    }

    if smooth.penaltyinfo.len() != smooth.penalties.len() {
        return Err(BasisError::InvalidInput(format!(
            "smooth penalty metadata mismatch: penalties={}, metadata={}",
            smooth.penalties.len(),
            smooth.penaltyinfo.len()
        )));
    }
    for ((s_local, &ns), localinfo) in smooth
        .penalties
        .iter()
        .zip(smooth.nullspace_dims.iter())
        .zip(smooth.penaltyinfo.iter())
    {
        let start = p_intercept + p_lin + p_rand;
        let global_index = penalties.len();
        // Propagate structural hints from PenaltyInfo to BlockwisePenalty
        // for efficient block-scale spectral decomposition.
        let bp = {
            let col_range = start..(start + p_smooth);
            let local = s_local.clone();
            if let Some(factors) = localinfo.penalty.kronecker_factors.as_ref() {
                BlockwisePenalty::kronecker(col_range, local, factors.clone())
            } else if matches!(localinfo.penalty.source, PenaltySource::TensorGlobalRidge)
                || matches!(
                    localinfo.penalty.source,
                    PenaltySource::Other(ref s) if s.starts_with("RandomEffectRidge")
                )
            {
                // Detect identity/ridge penalties from their source tag.
                BlockwisePenalty::ridge(col_range, 1.0)
            } else {
                BlockwisePenalty::new(col_range, local)
            }
        };
        penalties.push(bp);
        nullspace_dims.push(ns);
        let mut penalty = localinfo.penalty.clone();
        penalty.nullspace_dim_hint = ns;
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: localinfo.termname.clone(),
            penalty,
        });
    }
    dropped_penaltyinfo.extend(smooth.dropped_penaltyinfo.iter().cloned());

    debug_assert_eq!(
        penalties.len(),
        nullspace_dims.len(),
        "term-collection penalty/nullspace bookkeeping diverged"
    );
    debug_assert_eq!(
        penalties.len(),
        penaltyinfo.len(),
        "term-collection penalty metadata bookkeeping diverged"
    );

    if let Some(lb_smooth) = smooth.coefficient_lower_bounds.as_ref() {
        let start = p_intercept + p_lin + p_rand;
        coefficient_lower_bounds
            .slice_mut(s![start..(start + p_smooth)])
            .assign(lb_smooth);
        any_bounds = true;
    }
    if let Some(lin_smooth) = smooth.linear_constraints.as_ref() {
        let mut a_global = Array2::<f64>::zeros((lin_smooth.a.nrows(), p_total));
        let start = p_intercept + p_lin + p_rand;
        a_global
            .slice_mut(s![.., start..(start + p_smooth)])
            .assign(&lin_smooth.a);
        for r in 0..a_global.nrows() {
            linear_constraintrows.push(a_global.row(r).to_owned());
            linear_constraint_b.push(lin_smooth.b[r]);
        }
    }

    // Canonical constraint path: convert any explicit lower bounds into linear
    // inequalities and merge into the global constraint matrix. This keeps fitting
    // behavior independent of user-facing lower-bound options.
    let lower_bound_constraints = if any_bounds {
        linear_constraints_from_lower_bounds_global(&coefficient_lower_bounds)
    } else {
        None
    };
    let explicit_linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), p_total));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };
    let linear_constraints =
        merge_linear_constraints_global(explicit_linear_constraints, lower_bound_constraints);

    Ok(TermCollectionDesign {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        dropped_penaltyinfo,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints,
        intercept_range: 0..1,
        linear_ranges,
        random_effect_ranges,
        random_effect_levels,
        smooth,
    })
}

fn apply_spatial_orthogonality_to_parametric(
    smooth: RawSmoothDesign,
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    smoothspecs: &[SmoothTermSpec],
) -> Result<SmoothDesign, BasisError> {
    // Option 5 identifiability policy:
    //
    // Build a term-local parametric confounding block C_j = [1 | X_lin,overlap(j)],
    // then for each spatial smooth basis B_j enforce orthogonality to C_j in the
    // unweighted inner product:
    //   B_con^T C = 0.
    //
    // Reparameterization derivation:
    //   M = B^T C.
    //   If columns of Z span null(M^T), then
    //      (B Z)^T C = Z^T (B^T C) = Z^T M = 0.
    //
    // So B_con = B Z has no component in the parametric column space, eliminating
    // intercept/linear confounding without hand-picking polynomial columns.
    //
    // Penalties transform by congruence:
    //   S_con = Z^T S Z.
    // This preserves PSD and keeps curvature geometry consistent in constrained coords.
    if smoothspecs.len() != smooth.terms.len() {
        return Err(BasisError::DimensionMismatch(format!(
            "smooth spec count ({}) does not match built term count ({})",
            smoothspecs.len(),
            smooth.terms.len()
        )));
    }

    // Fast-path: if no spatial term participates in Option 5 (orthogonal or frozen),
    // return the smooth bundle unchanged and skip all matrix work.
    let any_spatial_transform = smoothspecs.iter().any(|t| {
        !matches!(
            spatial_identifiability_policy(t),
            Some(SpatialIdentifiability::None) | None
        )
    });
    if !any_spatial_transform {
        return Ok(smooth.into());
    }

    let n = smooth.nrows();
    let mut local_designs = Vec::<Array2<f64>>::with_capacity(smooth.terms.len());
    let mut local_penalties = Vec::<Vec<Array2<f64>>>::with_capacity(smooth.terms.len());
    let mut local_nullspaces = Vec::<Vec<usize>>::with_capacity(smooth.terms.len());
    let mut local_penaltyinfo = Vec::<Vec<PenaltyInfo>>::with_capacity(smooth.terms.len());
    let mut local_metadata = Vec::<BasisMetadata>::with_capacity(smooth.terms.len());
    let mut local_dims = Vec::<usize>::with_capacity(smooth.terms.len());
    let mut local_linear_constraints =
        Vec::<Option<LinearInequalityConstraints>>::with_capacity(smooth.terms.len());

    for (idx, term) in smooth.terms.iter().enumerate() {
        let termspec = &smoothspecs[idx];
        let design_local = smooth.term_designs[idx].clone();
        let use_frozen_transform = matches!(
            spatial_identifiability_policy(termspec),
            Some(SpatialIdentifiability::FrozenTransform { .. })
        );
        let c_local = if !use_frozen_transform
            && matches!(
                spatial_identifiability_policy(termspec),
                Some(SpatialIdentifiability::OrthogonalToParametric)
            ) {
            Some(build_parametric_constraint_block_for_term(
                data,
                linear_terms,
                termspec,
            )?)
        } else {
            None
        };
        let (design_constrained, z_opt) = maybe_spatial_identifiability_transform(
            termspec,
            design_local.view(),
            c_local.as_ref().map(|mat| mat.view()),
        )?;

        // Mathematical acceptance criterion:
        //   ||B_con^T C||_F / (||B_con||_F ||C||_F) <= tol.
        if matches!(
            spatial_identifiability_policy(termspec),
            Some(SpatialIdentifiability::OrthogonalToParametric)
        ) {
            let c_ref = c_local
                .as_ref()
                .expect("parametric constraint block must exist for orthogonal policy");
            let rel = orthogonality_relative_residual(design_constrained.view(), c_ref.view());
            let tol = 1e-8;
            if rel > tol {
                return Err(BasisError::InvalidInput(format!(
                    "spatial orthogonality residual too large for term '{}': {:.3e} > {:.1e}",
                    term.name, rel, tol
                )));
            }
        }

        let mut penalties_constrained =
            Vec::<Array2<f64>>::with_capacity(term.penalties_local.len());
        let active_penaltyinfo = term
            .penaltyinfo_local
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect::<Vec<_>>();
        if active_penaltyinfo.len() != term.penalties_local.len() {
            return Err(BasisError::InvalidInput(format!(
                "internal penalty metadata mismatch for term '{}': activeinfos={}, penalties={}",
                term.name,
                active_penaltyinfo.len(),
                term.penalties_local.len()
            )));
        }
        for s_local in &term.penalties_local {
            let s_con = if let Some(z) = z_opt.as_ref() {
                let zt_s = z.t().dot(s_local);
                zt_s.dot(z)
            } else {
                s_local.clone()
            };
            penalties_constrained.push(s_con);
        }
        let penalty_candidates = penalties_constrained
            .into_iter()
            .zip(active_penaltyinfo.into_iter())
            .map(|(matrix, info)| -> Result<PenaltyCandidate, BasisError> {
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: estimate_penalty_nullity(&matrix)?,
                    matrix,
                    source: info.source,
                    normalization_scale: info.normalization_scale * c_new,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (penalties_constrained, nullspace_constrained, penaltyinfo_constrained) =
            filter_active_penalty_candidates(penalty_candidates)?;
        let linear_constraints_constrained =
            if let Some(lin_local) = term.linear_constraints_local.as_ref() {
                if let Some(z) = z_opt.as_ref() {
                    Some(LinearInequalityConstraints {
                        a: lin_local.a.dot(z),
                        b: lin_local.b.clone(),
                    })
                } else {
                    Some(lin_local.clone())
                }
            } else {
                None
            };

        local_dims.push(design_constrained.ncols());
        local_designs.push(design_constrained);
        local_penalties.push(penalties_constrained);
        local_nullspaces.push(nullspace_constrained);
        local_penaltyinfo.push(penaltyinfo_constrained);
        local_linear_constraints.push(linear_constraints_constrained);
        local_metadata.push(with_spatial_identifiability_transform(
            &term.metadata,
            z_opt.as_ref(),
        ));
    }

    let total_p: usize = local_dims.iter().sum();
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(smooth.terms.len());
    let mut penalties_global = Vec::<Array2<f64>>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut penaltyinfo_global = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo_global = smooth.dropped_penaltyinfo.clone();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintsrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for idx in 0..smooth.terms.len() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;

        let activeinfos = local_penaltyinfo[idx]
            .iter()
            .filter(|info| info.active)
            .collect::<Vec<_>>();
        if activeinfos.len() != local_penalties[idx].len() {
            return Err(BasisError::InvalidInput(format!(
                "internal penalty info mismatch for term '{}': activeinfos={}, penalties={}",
                smooth.terms[idx].name,
                activeinfos.len(),
                local_penalties[idx].len()
            )));
        }
        for ((s_local, &ns), info) in local_penalties[idx]
            .iter()
            .zip(local_nullspaces[idx].iter())
            .zip(activeinfos.into_iter())
        {
            let global_index = penalties_global.len();
            let mut s_global = Array2::<f64>::zeros((total_p, total_p));
            s_global
                .slice_mut(s![col_start..col_end, col_start..col_end])
                .assign(s_local);
            penalties_global.push(s_global);
            nullspace_dims_global.push(ns);
            let mut penalty = info.clone();
            penalty.nullspace_dim_hint = ns;
            penaltyinfo_global.push(PenaltyBlockInfo {
                global_index,
                termname: Some(smooth.terms[idx].name.clone()),
                penalty,
            });
        }
        for info in local_penaltyinfo[idx].iter().filter(|info| !info.active) {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(smooth.terms[idx].name.clone()),
                penalty: info.clone(),
            });
        }

        terms_out.push(SmoothTerm {
            name: smooth.terms[idx].name.clone(),
            coeff_range: col_start..col_end,
            shape: smooth.terms[idx].shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            penaltyinfo_local: local_penaltyinfo[idx].clone(),
            metadata: local_metadata[idx].clone(),
            lower_bounds_local: smooth.terms[idx].lower_bounds_local.clone(),
            linear_constraints_local: local_linear_constraints[idx].clone(),
        });
        if let Some(lin_local) = &local_linear_constraints[idx] {
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![col_start..col_end])
                    .assign(&lin_local.a.row(r));
                linear_constraintsrows.push(row);
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = smooth.terms[idx].lower_bounds_local.as_ref()
            && lb_local.len() == p_local
        {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(lb_local);
            any_bounds = true;
        }

        col_start = col_end;
    }

    debug_assert_eq!(
        penalties_global.len(),
        nullspace_dims_global.len(),
        "spatially reparameterized smooth penalty/nullspace bookkeeping diverged"
    );
    debug_assert_eq!(
        penalties_global.len(),
        penaltyinfo_global.len(),
        "spatially reparameterized smooth penalty metadata bookkeeping diverged"
    );

    Ok(SmoothDesign {
        term_designs: local_designs,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        penaltyinfo: penaltyinfo_global,
        dropped_penaltyinfo: dropped_penaltyinfo_global,
        terms: terms_out,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints: if linear_constraintsrows.is_empty() {
            None
        } else {
            let mut a = Array2::<f64>::zeros((linear_constraintsrows.len(), total_p));
            for (i, row) in linear_constraintsrows.iter().enumerate() {
                a.row_mut(i).assign(row);
            }
            Some(LinearInequalityConstraints {
                a,
                b: Array1::from_vec(linear_constraints_b),
            })
        },
    })
}

fn build_parametric_constraint_block_for_term(
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    termspec: &SmoothTermSpec,
) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();

    let overlapping_linear_term_indices: Vec<usize> = match &termspec.basis {
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => linear_terms
            .iter()
            .enumerate()
            .filter_map(|(idx, linear)| {
                if feature_cols.contains(&linear.feature_col) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect(),
        SmoothBasisSpec::Duchon { feature_cols, .. } => linear_terms
            .iter()
            .enumerate()
            .filter_map(|(idx, linear)| {
                if feature_cols.contains(&linear.feature_col) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect(),
        _ => Vec::new(),
    };

    let mut c = Array2::<f64>::zeros((n, 1 + overlapping_linear_term_indices.len()));
    c.column_mut(0).fill(1.0);
    for (j, &lin_idx) in overlapping_linear_term_indices.iter().enumerate() {
        let linear = &linear_terms[lin_idx];
        if linear.feature_col >= p_data {
            return Err(BasisError::DimensionMismatch(format!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name, linear.feature_col, p_data
            )));
        }
        c.column_mut(j + 1).assign(&data.column(linear.feature_col));
    }
    Ok(c)
}

fn maybe_spatial_identifiability_transform(
    termspec: &SmoothTermSpec,
    design_local: ArrayView2<'_, f64>,
    parametric_block: Option<ArrayView2<'_, f64>>,
) -> Result<(Array2<f64>, Option<Array2<f64>>), BasisError> {
    let maybe_policy = spatial_identifiability_policy(termspec);
    let Some(policy) = maybe_policy else {
        return Ok((design_local.to_owned(), None));
    };

    match policy {
        SpatialIdentifiability::None => Ok((design_local.to_owned(), None)),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = parametric_block.ok_or_else(|| {
                BasisError::InvalidInput(
                    "missing parametric constraint block for OrthogonalToParametric policy"
                        .to_string(),
                )
            })?;
            let (b_c, z) = applyweighted_orthogonality_constraint(
                design_local,
                c,
                None, // fixed subspace: do not use iteration-varying PIRLS weights
            )?;
            Ok((b_c, Some(z)))
        }
        SpatialIdentifiability::FrozenTransform { transform } => {
            if design_local.ncols() != transform.nrows() {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen spatial identifiability transform mismatch: design has {} columns but transform has {} rows",
                    design_local.ncols(),
                    transform.nrows()
                )));
            }
            let z = transform.clone();
            Ok((design_local.dot(&z), Some(z)))
        }
    }
}

fn spatial_identifiability_policy(termspec: &SmoothTermSpec) -> Option<&SpatialIdentifiability> {
    match &termspec.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => Some(&spec.identifiability),
        SmoothBasisSpec::Duchon { spec, .. } => Some(&spec.identifiability),
        _ => None,
    }
}

pub(crate) fn orthogonality_relative_residual(
    basis_matrix: ArrayView2<'_, f64>,
    constraint_matrix: ArrayView2<'_, f64>,
) -> f64 {
    let cross = basis_matrix.t().dot(&constraint_matrix);
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = basis_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let c_norm = constraint_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = (b_norm * c_norm).max(1e-300);
    num / denom
}

fn with_spatial_identifiability_transform(
    metadata: &BasisMetadata,
    transform: Option<&Array2<f64>>,
) -> BasisMetadata {
    match metadata {
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            identifiability_transform,
            input_scales,
        } => BasisMetadata::ThinPlate {
            centers: centers.clone(),
            length_scale: *length_scale,
            identifiability_transform: transform
                .cloned()
                .or_else(|| identifiability_transform.clone()),
            input_scales: input_scales.clone(),
        },
        BasisMetadata::Duchon {
            centers,
            length_scale,
            power,
            nullspace_order,
            identifiability_transform,
            input_scales,
            aniso_log_scales,
        } => BasisMetadata::Duchon {
            centers: centers.clone(),
            length_scale: *length_scale,
            power: *power,
            nullspace_order: *nullspace_order,
            input_scales: input_scales.clone(),
            aniso_log_scales: aniso_log_scales.clone(),
            identifiability_transform: transform
                .cloned()
                .or_else(|| identifiability_transform.clone()),
        },
        _ => metadata.clone(),
    }
}

pub fn fit_term_collection_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    fit_term_collection_forspecwith_heuristic_lambdas(
        data, y, weights, offset, spec, None, family, options,
    )
}

fn fit_term_collection_forspecwith_heuristic_lambdas(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    heuristic_lambdas: Option<&[f64]>,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let base_design = build_term_collection_design(data, spec)?;
    fit_term_collection_on_realized_design(
        y,
        weights,
        offset,
        spec,
        &base_design,
        heuristic_lambdas,
        family,
        options,
    )
}

fn has_bounded_linear_terms(spec: &TermCollectionSpec) -> bool {
    spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    })
}

fn fit_term_collection_on_realized_design(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    heuristic_lambdas: Option<&[f64]>,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    if has_bounded_linear_terms(spec) {
        return fit_bounded_term_collection_with_design(
            y,
            weights,
            offset,
            spec,
            design,
            heuristic_lambdas,
            family,
            options,
        );
    }
    let base_fit_opts = adaptive_fit_options_base(options, design);
    let fitted = FittedTermCollection {
        fit: fit_gamwith_heuristic_lambdas(
            design.design.view(),
            y,
            weights,
            offset,
            &design.penalties,
            heuristic_lambdas,
            family,
            &base_fit_opts,
        )?,
        design: design.clone(),
        adaptive_diagnostics: None,
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;

    let adaptive_opts = options.adaptive_regularization.clone().unwrap_or_default();
    if !adaptive_opts.enabled {
        return Ok(fitted);
    }
    let runtime_caches = extract_spatial_operator_runtime_caches(spec, &fitted.design)?;
    if runtime_caches.is_empty() {
        return Ok(fitted);
    }
    fit_term_collectionwith_exact_spatial_adaptive_regularization(
        fitted,
        y,
        weights,
        offset,
        family,
        options,
        &runtime_caches,
    )
}

#[derive(Clone)]
struct SpatialOperatorRuntimeCache {
    termname: String,
    feature_cols: Vec<usize>,
    coeff_global_range: Range<usize>,
    mass_penalty_global_idx: usize,
    tension_penalty_global_idx: usize,
    stiffness_penalty_global_idx: usize,
    d0: Array2<f64>,
    d1: Array2<f64>,
    d2: Array2<f64>,
    collocation_points: Array2<f64>,
    dimension: usize,
}

#[derive(Clone)]
struct SpatialAdaptiveWeights {
    #[cfg(test)]
    magweight: Array1<f64>,
    #[cfg(test)]
    gradweight: Array1<f64>,
    #[cfg(test)]
    lapweight: Array1<f64>,
    inv_magweight: Array1<f64>,
    invgradweight: Array1<f64>,
    inv_lapweight: Array1<f64>,
}

#[derive(Clone)]
struct CharbonnierScalarBlockState {
    signal: Array1<f64>,
    radius: Array1<f64>,
    epsilon: f64,
}

impl CharbonnierScalarBlockState {
    fn from_signal(signal: Array1<f64>, epsilon: f64) -> Self {
        let eps = epsilon.max(1e-12);
        let radius = signal.mapv(|t| (t * t + eps * eps).sqrt());
        Self {
            signal,
            radius,
            epsilon: eps,
        }
    }

    fn absolute_signal(&self) -> Array1<f64> {
        self.signal.mapv(f64::abs)
    }

    fn penalty_value(&self) -> f64 {
        self.radius
            .iter()
            .map(|r| self.epsilon * (r - self.epsilon))
            .sum::<f64>()
    }

    fn betagradient_coeff(&self) -> Array1<f64> {
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(t, r)| self.epsilon * t / r),
        )
    }

    fn betahessian_diag(&self) -> Array1<f64> {
        let eps3 = self.epsilon.powi(3);
        self.radius.mapv(|r| eps3 / (r * r * r))
    }

    fn log_epsilon_gradient_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        let eps3 = eps2 * epsilon;
        // Closed form in eta = log eps:
        //
        //   d/deta psi(t; eps) = eps * r + eps^3 / r - 2 eps^2.
        self.radius.mapv(|r| epsilon * r + eps3 / r - 2.0 * eps2)
    }

    fn log_epsilon_betagradient_coeff(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(t, r)| epsilon * t.powi(3) / r.powi(3)),
        )
    }

    fn log_epsilon_hessian_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        let eps3 = eps2 * epsilon;
        let eps5 = eps3 * eps2;
        self.radius
            .mapv(|r| epsilon * r + 4.0 * eps3 / r - eps5 / r.powi(3) - 4.0 * eps2)
    }

    fn surrogateweights(
        &self,
        weight_floor: f64,
        weight_ceiling: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        // Exact scalar scaled-Charbonnier / pseudo-Huber block and its MM majorizer.
        //
        // The exact scalar penalty used by the nonquadratic spatial regularizer is
        //
        //   psi(t; eps) = eps * (sqrt(t^2 + eps^2) - eps),
        //
        // where:
        //   - t is the scalar operator response at one collocation point,
        //   - eps > 0 is the Charbonnier transition scale.
        //
        // The key exact scalar derivatives are:
        //
        //   d/dt psi(t; eps)        = eps * t / sqrt(t^2 + eps^2),
        //   d^2/dt^2 psi(t; eps)    = eps^3 / (t^2 + eps^2)^(3/2),
        //   d/deps psi(t; eps)      = sqrt(t^2 + eps^2) + eps^2 / sqrt(t^2 + eps^2) - 2 eps,
        //   d^2/(dt deps) psi       = t^3 / (t^2 + eps^2)^(3/2).
        //
        // These exact formulas define the real adaptive model used by the
        // pseudo-Laplace hyperobjective. We still keep the legacy MM majorizer
        // weights here because they remain useful for diagnostics/tests and for
        // comparing the old surrogate path against the exact Charbonnier model.
        //
        // For a reference point t0, the same tangent majorizer in t^2 gives
        // tangent majorizer in the variable t^2:
        //
        //   psi(t; eps) <= u(t0) * t^2 + const(t0),
        //   u(t0) = eps / (2 * sqrt(t0^2 + eps^2)).
        //
        // So the MM algorithm reuses only the scalar weight
        //
        //   u_k = eps / (2 * sqrt(t_k^2 + eps^2)),
        //
        // while the exact derivatives above remain the source of truth for the
        // production direct inner optimizer and exact outer hypergradient.
        //
        // Keeping the surrogate weight generation immediately beside the exact
        // scalar Charbonnier formulas is deliberate:
        //   1. it avoids a second copy of the scalar algebra,
        //   2. it makes the current approximation explicit rather than implicit,
        //   3. it gives one audited location for the transition from the true
        //      penalty to the MM surrogate.
        let weight = self
            .radius
            .mapv(|r| (1.0 / r).clamp(weight_floor, weight_ceiling));
        let invweight = weight.mapv(|u| 1.0 / u);
        (weight, invweight)
    }

    fn directionalhessian_diag(&self, direction_signal: &Array1<f64>) -> Array1<f64> {
        // Scalar-image directional third derivative:
        //
        // If t(beta) = A beta and
        //   H(beta) = A^T diag( eps^2 / (t_k(beta)^2 + eps^2)^(3/2) ) A,
        // then for q = A u,
        //
        //   D(H)[u]
        //   = A^T diag( -3 eps^2 t_k q_k / (t_k^2 + eps^2)^(5/2) ) A.
        //
        // This is one of the exact P_{beta,beta,beta}[u] terms needed by the
        // Laplace hypergradient
        //
        //   d/dtheta log det H = tr(H^{-1} Hdot_theta),
        //   Hdot_theta = J_{beta,beta,theta} + D_beta(H)[beta_theta].
        let eps3 = self.epsilon.powi(3);
        Array1::from_iter(
            self.signal
                .iter()
                .zip(direction_signal.iter())
                .zip(self.radius.iter())
                .map(|((t, q), r)| -3.0 * eps3 * t * q / r.powi(5)),
        )
    }

    fn log_epsilon_betahessian_diag(&self) -> Array1<f64> {
        let eps3 = self.epsilon.powi(3);
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(t, r)| 3.0 * eps3 * t * t / r.powi(5)),
        )
    }

    fn log_epsilon_beta_mixed_second_coeff(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        Array1::from_iter(
            self.signal
                .iter()
                .zip(self.radius.iter())
                .map(|(t, r)| epsilon * t.powi(3) * (t * t - 2.0 * epsilon * epsilon) / r.powi(5)),
        )
    }

    fn log_epsilon_betahessian_second_diag(&self) -> Array1<f64> {
        let eps3 = self.epsilon.powi(3);
        Array1::from_iter(self.signal.iter().zip(self.radius.iter()).map(|(t, r)| {
            3.0 * eps3 * t * t * (3.0 * t * t - 2.0 * self.epsilon * self.epsilon) / r.powi(7)
        }))
    }

    fn log_epsilon_betahessian_directional_diag(
        &self,
        direction_signal: &Array1<f64>,
    ) -> Array1<f64> {
        let eps3 = self.epsilon.powi(3);
        Array1::from_iter(
            self.signal
                .iter()
                .zip(direction_signal.iter())
                .zip(self.radius.iter())
                .map(|((t, q), r)| {
                    3.0 * eps3 * t * (2.0 * self.epsilon * self.epsilon - 3.0 * t * t) * q
                        / r.powi(7)
                }),
        )
    }
}

#[derive(Clone)]
struct CharbonnierGroupedBlockState {
    norm: Array1<f64>,
    radius: Array1<f64>,
    signal_blocks: Array2<f64>,
    epsilon: f64,
}

impl CharbonnierGroupedBlockState {
    fn from_signal_blocks(signal_blocks: Array2<f64>, epsilon: f64) -> Self {
        let eps = epsilon.max(1e-12);
        let norm = Array1::from_iter(
            signal_blocks
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt()),
        );
        let radius = norm.mapv(|g| (g * g + eps * eps).sqrt());
        Self {
            norm,
            radius,
            signal_blocks,
            epsilon: eps,
        }
    }

    fn penalty_value(&self) -> f64 {
        self.radius
            .iter()
            .map(|r| self.epsilon * (r - self.epsilon))
            .sum::<f64>()
    }

    fn norm_signal(&self) -> Array1<f64> {
        self.norm.clone()
    }

    fn betagradient_blocks(&self) -> Array2<f64> {
        let mut out = self.signal_blocks.clone();
        for (k, mut row) in out.rows_mut().into_iter().enumerate() {
            let scale = self.epsilon / self.radius[k];
            row.mapv_inplace(|v| v * scale);
        }
        out
    }

    fn betahessian_blocks(&self) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        for (k, row) in self.signal_blocks.rows().into_iter().enumerate() {
            let dim = row.len();
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|v| self.epsilon * v / self.radius[k]);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] -= self.epsilon * row[i] * row[j] / self.radius[k].powi(3);
                }
            }
            out.push(block);
        }
        out
    }

    fn log_epsilon_gradient_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        let eps3 = eps2 * epsilon;
        self.radius.mapv(|r| epsilon * r + eps3 / r - 2.0 * eps2)
    }

    fn log_epsilon_betagradient_blocks(&self) -> Array2<f64> {
        let mut out = self.signal_blocks.clone();
        let epsilon = self.epsilon;
        for (k, mut row) in out.rows_mut().into_iter().enumerate() {
            let scale = epsilon * self.norm[k] * self.norm[k] / self.radius[k].powi(3);
            row.mapv_inplace(|v| v * scale);
        }
        out
    }

    fn log_epsilon_hessian_terms(&self) -> Array1<f64> {
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        let eps3 = eps2 * epsilon;
        let eps5 = eps3 * eps2;
        self.radius
            .mapv(|r| epsilon * r + 4.0 * eps3 / r - eps5 / r.powi(3) - 4.0 * eps2)
    }

    fn surrogateweights(
        &self,
        weight_floor: f64,
        weight_ceiling: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        // Grouped scaled-Charbonnier / pseudo-Huber MM weights for the slope block.
        //
        // For the grouped penalty, each collocation point contributes
        //
        //   psi(g_k; eps_g) = eps_g * (sqrt(g_k^2 + eps_g^2) - eps_g),
        //   g_k = ||v_k||_2,
        //   v_k = G_k beta.
        //
        // The exact block gradient is
        //
        //   d/d beta psi(g_k; eps_g)
        //   = G_k^T (eps_g * v_k / sqrt(||v_k||^2 + eps_g^2)),
        //
        // and the exact block Hessian is
        //
        //   G_k^T (eps_g B_k) G_k,
        //
        //   B_k
        //   = (1 / r_k) I - (1 / r_k^3) v_k v_k^T,
        //   r_k = sqrt(||v_k||^2 + eps_g^2).
        //
        // The current adaptive path does not use that exact Hessian directly.
        // Instead it uses the grouped MM majorizer
        //
        //   psi(g; eps_g) <= u(g0) * g^2 + const(g0),
        //   u(g0) = eps_g / (2 * sqrt(g0^2 + eps_g^2)),
        //
        // which yields the quadratic surrogate
        //
        //   K_g = D1^T (diag(u_g) \kron I_d) D1.
        //
        // These weights are therefore the grouped analogue of the scalar MM
        // majorizer above: they are not the exact slope Hessian, they are the
        // tangent quadratic envelope used by the existing iterative reweighting
        // scheme. The exact grouped Hessian and third-derivative maps live in the
        // neighboring methods and are what the direct pseudo-Laplace solver will
        // ultimately use.
        let weight = self
            .radius
            .mapv(|r| (1.0 / r).clamp(weight_floor, weight_ceiling));
        let invweight = weight.mapv(|u| 1.0 / u);
        (weight, invweight)
    }

    fn directionalhessian_blocks(&self, direction_blocks: &Array2<f64>) -> Vec<Array2<f64>> {
        // Exact grouped directional third derivative for the slope penalty.
        //
        // For each collocation block k:
        //   v_k = G_k beta,
        //   q_k = G_k u,
        //   r_k = sqrt(||v_k||^2 + eps^2),
        //
        // the exact Hessian block for psi(g; eps) = eps * (sqrt(g^2 + eps^2) - eps) is
        //   eps * B_k,
        //   B_k = (1 / r_k) I - v_k v_k^T / r_k^3.
        //
        // Differentiating B_k along u gives
        //   M_k(u)
        //   = -(v_k^T q_k / r_k^3) I
        //     - (q_k v_k^T + v_k q_k^T) / r_k^3
        //     + 3 (v_k^T q_k) v_k v_k^T / r_k^5.
        //
        // This expression must be symmetric because it is the directional
        // derivative of the symmetric matrix
        //
        //   B_k = (1 / r_k) I - v_k v_k^T / r_k^3.
        //
        // The full directional penalty Hessian map is then
        //   D(H_g)[u] = lambda_g * eps * sum_k G_k^T M_k(u) G_k.
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        for (k, (v, q)) in self
            .signal_blocks
            .rows()
            .into_iter()
            .zip(direction_blocks.rows().into_iter())
            .enumerate()
        {
            let dim = v.len();
            let dot = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f64>();
            let r3 = self.radius[k].powi(3);
            let r5 = self.radius[k].powi(5);
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|x| -dot * x / r3);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] -= (q[i] * v[j] + v[i] * q[j]) / r3;
                    block[[i, j]] += 3.0 * dot * v[i] * v[j] / r5;
                }
            }
            block.mapv_inplace(|x| self.epsilon * x);
            out.push(block);
        }
        out
    }

    fn log_epsilon_betahessian_blocks(&self) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        for (k, row) in self.signal_blocks.rows().into_iter().enumerate() {
            let dim = row.len();
            let r3 = self.radius[k].powi(3);
            let r5 = self.radius[k].powi(5);
            let mut block = Array2::<f64>::eye(dim);
            let norm2 = self.norm[k] * self.norm[k];
            block.mapv_inplace(|v| self.epsilon * norm2 * v / r3);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] += self.epsilon
                        * (2.0 * self.epsilon * self.epsilon - norm2)
                        * row[i]
                        * row[j]
                        / r5;
                }
            }
            out.push(block);
        }
        out
    }

    fn log_epsilon_beta_mixed_second_blocks(&self) -> Array2<f64> {
        let mut out = self.signal_blocks.clone();
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        for (k, mut row) in out.rows_mut().into_iter().enumerate() {
            let norm2 = self.norm[k] * self.norm[k];
            let scale = epsilon * norm2 * (norm2 - 2.0 * eps2) / self.radius[k].powi(5);
            row.mapv_inplace(|v| v * scale);
        }
        out
    }

    fn log_epsilon_betahessian_second_blocks(&self) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        let eps4 = eps2 * eps2;
        for (k, row) in self.signal_blocks.rows().into_iter().enumerate() {
            let dim = row.len();
            let norm2 = self.norm[k] * self.norm[k];
            let r5 = self.radius[k].powi(5);
            let r7 = self.radius[k].powi(7);
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|v| epsilon * norm2 * (norm2 - 2.0 * eps2) * v / r5);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] += epsilon
                        * (-norm2 * norm2 + 10.0 * eps2 * norm2 - 4.0 * eps4)
                        * row[i]
                        * row[j]
                        / r7;
                }
            }
            out.push(block);
        }
        out
    }

    fn log_epsilon_betahessian_directional_blocks(
        &self,
        direction_blocks: &Array2<f64>,
    ) -> Vec<Array2<f64>> {
        let mut out = Vec::with_capacity(self.signal_blocks.nrows());
        let epsilon = self.epsilon;
        let eps2 = epsilon * epsilon;
        for (k, (v, q)) in self
            .signal_blocks
            .rows()
            .into_iter()
            .zip(direction_blocks.rows().into_iter())
            .enumerate()
        {
            let dim = v.len();
            let norm2 = self.norm[k] * self.norm[k];
            let dot = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f64>();
            let coeff = (2.0 * eps2 - norm2) / self.radius[k].powi(5);
            let r7 = self.radius[k].powi(7);
            let mut block = Array2::<f64>::eye(dim);
            block.mapv_inplace(|x| epsilon * coeff * dot * x);
            for i in 0..dim {
                for j in 0..dim {
                    block[[i, j]] += epsilon * coeff * (q[i] * v[j] + v[i] * q[j]);
                    block[[i, j]] += 3.0 * epsilon * (norm2 - 4.0 * eps2) * dot * v[i] * v[j] / r7;
                }
            }
            out.push(block);
        }
        out
    }
}

fn scalar_operatorgradient(operator: &Array2<f64>, coeff: &Array1<f64>) -> Array1<f64> {
    operator.t().dot(coeff)
}

fn scalar_operatorhessian(operator: &Array2<f64>, diag: &Array1<f64>) -> Array2<f64> {
    let mut weighted = operator.clone();
    for (k, &w) in diag.iter().enumerate() {
        weighted.row_mut(k).mapv_inplace(|v| v * w);
    }
    let gram = operator.t().dot(&weighted);
    (&gram + &gram.t().to_owned()) * 0.5
}

fn grouped_operatorgradient(
    d1: &Array2<f64>,
    dimension: usize,
    blocks: &Array2<f64>,
) -> Result<Array1<f64>, EstimationError> {
    if blocks.ncols() != dimension {
        return Err(EstimationError::InvalidInput(format!(
            "grouped gradient block dimension mismatch: got {}, expected {dimension}",
            blocks.ncols()
        )));
    }
    if d1.nrows() != blocks.nrows() * dimension {
        return Err(EstimationError::InvalidInput(format!(
            "grouped gradient row mismatch: D1 has {} rows, blocks imply {}",
            d1.nrows(),
            blocks.nrows() * dimension
        )));
    }
    let mut out = Array1::<f64>::zeros(d1.ncols());
    for k in 0..blocks.nrows() {
        let gk = d1
            .slice(s![k * dimension..(k + 1) * dimension, ..])
            .to_owned();
        out += &gk.t().dot(&blocks.row(k).to_owned());
    }
    Ok(out)
}

fn grouped_operatorhessian(
    d1: &Array2<f64>,
    dimension: usize,
    blocks: &[Array2<f64>],
) -> Result<Array2<f64>, EstimationError> {
    if d1.nrows() != blocks.len() * dimension {
        return Err(EstimationError::InvalidInput(format!(
            "grouped Hessian row mismatch: D1 has {} rows, blocks imply {}",
            d1.nrows(),
            blocks.len() * dimension
        )));
    }
    let p = d1.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    for (k, block) in blocks.iter().enumerate() {
        if block.nrows() != dimension || block.ncols() != dimension {
            return Err(EstimationError::InvalidInput(format!(
                "grouped Hessian block {k} has shape {}x{}, expected {}x{}",
                block.nrows(),
                block.ncols(),
                dimension,
                dimension
            )));
        }
        let gk = d1
            .slice(s![k * dimension..(k + 1) * dimension, ..])
            .to_owned();
        out += &gk.t().dot(&block.dot(&gk));
    }
    Ok((&out + &out.t().to_owned()) * 0.5)
}

#[derive(Clone)]
struct SpatialPenaltyExactState {
    magnitude: CharbonnierScalarBlockState,
    gradient: CharbonnierGroupedBlockState,
    curvature: CharbonnierScalarBlockState,
}

fn collocationgradient_blocks(
    gradrows: &Array1<f64>,
    dimension: usize,
) -> Result<Array2<f64>, EstimationError> {
    if dimension == 0 || gradrows.len() % dimension != 0 {
        return Err(EstimationError::InvalidInput(format!(
            "invalid collocation gradient layout: rows={}, dimension={dimension}",
            gradrows.len()
        )));
    }
    let p = gradrows.len() / dimension;
    let mut out = Array2::<f64>::zeros((p, dimension));
    for k in 0..p {
        for axis in 0..dimension {
            out[[k, axis]] = gradrows[k * dimension + axis];
        }
    }
    Ok(out)
}

impl SpatialPenaltyExactState {
    fn from_beta_local(
        beta_local: ArrayView1<'_, f64>,
        cache: &SpatialOperatorRuntimeCache,
        epsilons: [f64; 3],
    ) -> Result<Self, EstimationError> {
        // Exact collocation-state extraction for the three Charbonnier penalty blocks.
        //
        // For one spatial smooth term with coefficient vector beta_local, the exact
        // operator-decomposition penalty is built from three collocation images:
        //
        //   magnitude:  f = D0 beta_local
        //   slope:      v_k = G_k beta_local
        //   curvature:  c = D2 beta_local
        //
        // where the gradient operator is stored in row-stacked form:
        //
        //   D1 beta_local in R^(P * d),
        //   row layout = (point 0, axis 0..d-1), (point 1, axis 0..d-1), ...
        //
        // so we first reshape that stacked vector into the grouped block array
        //
        //   [v_0^T
        //    ...
        //    v_(P-1)^T]  in R^(P x d).
        //
        // The three exact Charbonnier block states then carry:
        //   - the raw operator signals,
        //   - their radii sqrt(signal^2 + eps^2) or sqrt(||v_k||^2 + eps^2),
        //   - and all exact derivatives derived from those radii.
        //
        // This is the canonical translation from coefficient-space beta to the
        // penalty-side mathematical objects used throughout the implementation.
        let gradientrows = cache.d1.dot(&beta_local);
        Ok(Self {
            magnitude: CharbonnierScalarBlockState::from_signal(
                cache.d0.dot(&beta_local),
                epsilons[0],
            ),
            gradient: CharbonnierGroupedBlockState::from_signal_blocks(
                collocationgradient_blocks(&gradientrows, cache.dimension)?,
                epsilons[1],
            ),
            curvature: CharbonnierScalarBlockState::from_signal(
                cache.d2.dot(&beta_local),
                epsilons[2],
            ),
        })
    }

    fn absolute_collocation_magnitudes(&self) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (
            self.magnitude.absolute_signal(),
            self.gradient.norm_signal(),
            self.curvature.absolute_signal(),
        )
    }
}

fn quantile_from_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let qq = q.clamp(0.0, 1.0);
    let pos = qq * (sorted.len().saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let t = pos - lo as f64;
        sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

fn robust_epsilon_from_samples(values: &[f64], min_epsilon_cfg: f64) -> f64 {
    if values.is_empty() {
        return min_epsilon_cfg.max(1e-12);
    }
    let mut clean = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect::<Vec<_>>();
    if clean.is_empty() {
        return min_epsilon_cfg.max(1e-12);
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = clean.len();
    let median = quantile_from_sorted(&clean, 0.5);
    let q75 = quantile_from_sorted(&clean, 0.75);
    let q95 = quantile_from_sorted(&clean, 0.95);

    let mut abs_dev = clean
        .iter()
        .map(|v| (v - median).abs())
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = 1.4826 * quantile_from_sorted(&abs_dev, 0.5);

    // Charbonnier/MM requires eps bounded away from zero:
    //   u(t0) = 1 / (2*sqrt(t0^2 + eps^2)) ~ 1/(2*eps) near t0=0.
    // Use robust pilot scale:
    //   s = max(median(z), 1.4826*MAD(z), Q75(z)).
    // If s is tiny (<= delta), fallback to:
    //   s <- max(Q95(z), RMS(z)).
    // If still tiny, fallback to absolute floor s_min.
    // Then eps = kappa * s.
    // Primary robust scale: s = max(median, 1.4826*MAD, Q75).
    let mut scale = median.max(mad).max(q75);

    // Safety threshold delta and absolute floor s_min.
    let delta = (f64::EPSILON.sqrt() * q95.max(1.0))
        .max(min_epsilon_cfg)
        .max(1e-12);
    let s_min = min_epsilon_cfg.max(1e-12);

    // If robust scale is tiny, use high-quantile / RMS fallback.
    if scale <= delta {
        let rms = (clean.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        scale = q95.max(rms);
    }
    if scale <= delta {
        scale = s_min;
    }

    // Start near the observed operator scale so the optimizer begins in a
    // neutral regime where both quadratic and linear behavior are reachable.
    let kappa = 1.0_f64;
    (kappa * scale).max(s_min)
}

fn extract_spatial_operator_runtime_caches(
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<Vec<SpatialOperatorRuntimeCache>, EstimationError> {
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());
    let mut out = Vec::<SpatialOperatorRuntimeCache>::new();
    for (term_idx, (termspec, term_fit)) in spec
        .smooth_terms
        .iter()
        .zip(design.smooth.terms.iter())
        .enumerate()
    {
        let Some(global_base_idx) = smooth_term_penalty_index(spec, design, term_idx) else {
            continue;
        };
        let mut active_local_idx = 0usize;
        let mut mass_local_idx = None;
        let mut tension_local_idx = None;
        let mut stiffness_local_idx = None;
        let mut mass_norm = None;
        let mut tension_norm = None;
        let mut stiffness_norm = None;
        for info in &term_fit.penaltyinfo_local {
            if !info.active {
                continue;
            }
            match info.source {
                PenaltySource::OperatorMass => {
                    mass_local_idx = Some(active_local_idx);
                    mass_norm = Some(info.normalization_scale);
                }
                PenaltySource::OperatorTension => {
                    tension_local_idx = Some(active_local_idx);
                    tension_norm = Some(info.normalization_scale);
                }
                PenaltySource::OperatorStiffness => {
                    stiffness_local_idx = Some(active_local_idx);
                    stiffness_norm = Some(info.normalization_scale);
                }
                _ => {}
            }
            active_local_idx += 1;
        }
        let (
            Some(mass_local),
            Some(tension_local),
            Some(stiffness_local),
            Some(mass_scale),
            Some(tension_scale),
            Some(stiffness_scale),
        ) = (
            mass_local_idx,
            tension_local_idx,
            stiffness_local_idx,
            mass_norm,
            tension_norm,
            stiffness_norm,
        )
        else {
            continue;
        };
        let mass_global_idx = global_base_idx + mass_local;
        let tension_global_idx = global_base_idx + tension_local;
        let stiffness_global_idx = global_base_idx + stiffness_local;

        let (feature_cols, mut d0, mut d1, mut d2, collocation_points, dim) =
            match (&termspec.basis, &term_fit.metadata) {
                (
                    SmoothBasisSpec::Matern { feature_cols, .. },
                    BasisMetadata::Matern {
                        centers,
                        length_scale,
                        nu,
                        include_intercept,
                        identifiability_transform,
                        aniso_log_scales,
                        ..
                    },
                ) => {
                    let ops = build_matern_collocation_operator_matrices(
                        centers.view(),
                        None,
                        *length_scale,
                        *nu,
                        *include_intercept,
                        identifiability_transform.as_ref().map(|z| z.view()),
                        aniso_log_scales.as_deref(),
                    )?;
                    (
                        feature_cols.clone(),
                        ops.d0,
                        ops.d1,
                        ops.d2,
                        ops.collocation_points,
                        centers.ncols(),
                    )
                }
                (
                    SmoothBasisSpec::Duchon { feature_cols, .. },
                    BasisMetadata::Duchon {
                        centers,
                        length_scale,
                        power,
                        nullspace_order,
                        identifiability_transform,
                        aniso_log_scales,
                        ..
                    },
                ) => {
                    let mut ws = crate::basis::BasisWorkspace::default();
                    let ops =
                        crate::basis::build_duchon_collocation_operator_matriceswithworkspace(
                            centers.view(),
                            None,
                            *length_scale,
                            *power,
                            *nullspace_order,
                            aniso_log_scales.as_deref(),
                            identifiability_transform.as_ref().map(|z| z.view()),
                            &mut ws,
                        )?;
                    (
                        feature_cols.clone(),
                        ops.d0,
                        ops.d1,
                        ops.d2,
                        ops.collocation_points,
                        centers.ncols(),
                    )
                }
                _ => continue,
            };

        // Runtime operator caches must live on the same normalized penalty scale as the
        // shipped design penalties. The basis builders normalize S0=D0'D0, S1=D1'D1, and
        // S2=D2'D2 before exposing them as smoothing blocks, recording the corresponding
        // Frobenius norms in penaltyinfo_local.normalization_scale. If the exact adaptive
        // path uses raw collocation operators here, then its Charbonnier penalties live on a
        // different geometry from the ordinary Matérn/Duchon penalties:
        //
        //   raw quadratic limit:        beta' (D'D) beta
        //   shipped design penalty:     beta' (D'D / c) beta
        //
        // The correct operator-level normalization is therefore
        //
        //   D_norm = D / sqrt(c),
        //
        // so that D_norm' D_norm = (D'D)/c matches the design penalty exactly. Without this,
        // adaptive lambdas compensate for hidden operator-scale mismatches and are no longer
        // comparable to the baseline smoothing parameters.
        let mass_scale = mass_scale.max(1e-12).sqrt();
        let tension_scale = tension_scale.max(1e-12).sqrt();
        let stiffness_scale = stiffness_scale.max(1e-12).sqrt();
        d0.mapv_inplace(|v| v / mass_scale);
        d1.mapv_inplace(|v| v / tension_scale);
        d2.mapv_inplace(|v| v / stiffness_scale);

        let coeff_global_range =
            (smooth_start + term_fit.coeff_range.start)..(smooth_start + term_fit.coeff_range.end);
        if d0.ncols() != coeff_global_range.len()
            || d1.ncols() != coeff_global_range.len()
            || d2.ncols() != coeff_global_range.len()
        {
            return Err(EstimationError::InvalidInput(format!(
                "spatial operator dimension mismatch for term '{}': D0 cols={}, D1 cols={}, D2 cols={}, coeffs={}",
                term_fit.name,
                d0.ncols(),
                d1.ncols(),
                d2.ncols(),
                coeff_global_range.len()
            )));
        }
        out.push(SpatialOperatorRuntimeCache {
            termname: term_fit.name.clone(),
            feature_cols,
            coeff_global_range,
            mass_penalty_global_idx: mass_global_idx,
            tension_penalty_global_idx: tension_global_idx,
            stiffness_penalty_global_idx: stiffness_global_idx,
            d0,
            d1,
            d2,
            collocation_points,
            dimension: dim,
        });
    }
    Ok(out)
}

fn compute_spatial_adaptiveweights_for_beta(
    beta: &Array1<f64>,
    caches: &[SpatialOperatorRuntimeCache],
    epsilon_0: f64,
    epsilon_g: f64,
    epsilon_c: f64,
    weight_floor: f64,
    weight_ceiling: f64,
) -> Result<Vec<SpatialAdaptiveWeights>, EstimationError> {
    // Scaled Charbonnier / pseudo-Huber MM derivation (per collocation scalar t):
    //   psi(t; eps) = eps * (sqrt(t^2 + eps^2) - eps)
    // and for reference t0 the tangent majorizer in t^2 gives:
    //   psi(t) <= u(t0) * t^2 + const(t0),
    //   u(t0) = eps / (2*sqrt(t0^2 + eps^2)).
    //
    // We apply this to:
    //   t = f_k = |f(z_k)|            (magnitude),
    //   t = g_k = ||nabla f(z_k)||_2  (gradient magnitude),
    //   t = c_k = |Delta f(z_k)|      (laplacian magnitude),
    // both computed from beta^(t-1).
    //
    // These u values define the quadratic surrogate penalties:
    //   K0 = D0_con^T W_0 D0_con,  W_0 = diag(u_0)
    //   K1 = D1_con^T W_g D1_con,  W_g = diag(u_g) \otimes I_d  (k,axis order)
    //   K2 = D2_con^T W_c D2_con,  W_c = diag(u_c).
    //
    // We clamp u directly, then derive inv_u=1/u for diagnostics and row scaling.
    caches
        .iter()
        .map(|cache| {
            let beta_local = beta.slice(s![cache.coeff_global_range.clone()]);
            let exact = SpatialPenaltyExactState::from_beta_local(
                beta_local,
                cache,
                [epsilon_0, epsilon_g, epsilon_c],
            )?;
            #[cfg(test)]
            let (u_0, inv_0) = exact
                .magnitude
                .surrogateweights(weight_floor, weight_ceiling);
            #[cfg(not(test))]
            let (_, inv_0) = exact
                .magnitude
                .surrogateweights(weight_floor, weight_ceiling);
            #[cfg(test)]
            let (u_g, inv_g) = exact
                .gradient
                .surrogateweights(weight_floor, weight_ceiling);
            #[cfg(not(test))]
            let (_, inv_g) = exact
                .gradient
                .surrogateweights(weight_floor, weight_ceiling);
            #[cfg(test)]
            let (u_c, inv_c) = exact
                .curvature
                .surrogateweights(weight_floor, weight_ceiling);
            #[cfg(not(test))]
            let (_, inv_c) = exact
                .curvature
                .surrogateweights(weight_floor, weight_ceiling);
            Ok(SpatialAdaptiveWeights {
                #[cfg(test)]
                magweight: u_0,
                #[cfg(test)]
                gradweight: u_g,
                #[cfg(test)]
                lapweight: u_c,
                inv_magweight: inv_0,
                invgradweight: inv_g,
                inv_lapweight: inv_c,
            })
        })
        .collect()
}

fn compute_initial_epsilons(
    beta: &Array1<f64>,
    caches: &[SpatialOperatorRuntimeCache],
    min_epsilon: f64,
) -> Result<(f64, f64, f64), EstimationError> {
    let mut fvals = Vec::<f64>::new();
    let mut gvals = Vec::<f64>::new();
    let mut cvals = Vec::<f64>::new();
    for cache in caches {
        let beta_local = beta.slice(s![cache.coeff_global_range.clone()]);
        let exact = SpatialPenaltyExactState::from_beta_local(
            beta_local,
            cache,
            [min_epsilon, min_epsilon, min_epsilon],
        )?;
        let (f, g, c) = exact.absolute_collocation_magnitudes();
        fvals.extend(f.iter().copied());
        gvals.extend(g.iter().copied());
        cvals.extend(c.iter().copied());
    }
    // Robust epsilon initialization from pilot magnitudes:
    //   s = max(median(z), 1.4826*MAD(z), Q75(z)),
    //   if s is tiny then fallback to max(Q95(z), RMS(z)),
    //   if still tiny then use absolute floor min_epsilon.
    // Epsilon is then kappa * s.
    let eps_0 = robust_epsilon_from_samples(&fvals, min_epsilon);
    let eps_g = robust_epsilon_from_samples(&gvals, min_epsilon);
    let eps_c = robust_epsilon_from_samples(&cvals, min_epsilon);
    Ok((eps_0, eps_g, eps_c))
}

fn exact_spatial_adaptive_penalty_index_set(
    caches: &[SpatialOperatorRuntimeCache],
) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for cache in caches {
        out.insert(cache.mass_penalty_global_idx);
        out.insert(cache.tension_penalty_global_idx);
        out.insert(cache.stiffness_penalty_global_idx);
    }
    out
}

fn build_spatial_adaptive_hyperspecs(cache_count: usize) -> Vec<SpatialAdaptiveHyperSpec> {
    let mut out = Vec::with_capacity(cache_count * 3 + 3);
    for cache_index in 0..cache_count {
        out.push(SpatialAdaptiveHyperSpec {
            cache_index,
            kind: SpatialAdaptiveHyperKind::LogLambdaMagnitude,
        });
        out.push(SpatialAdaptiveHyperSpec {
            cache_index,
            kind: SpatialAdaptiveHyperKind::LogLambdaGradient,
        });
        out.push(SpatialAdaptiveHyperSpec {
            cache_index,
            kind: SpatialAdaptiveHyperKind::LogLambdaCurvature,
        });
    }
    out.push(SpatialAdaptiveHyperSpec {
        cache_index: 0,
        kind: SpatialAdaptiveHyperKind::LogEpsilonMagnitude,
    });
    out.push(SpatialAdaptiveHyperSpec {
        cache_index: 0,
        kind: SpatialAdaptiveHyperKind::LogEpsilonGradient,
    });
    out.push(SpatialAdaptiveHyperSpec {
        cache_index: 0,
        kind: SpatialAdaptiveHyperKind::LogEpsilonCurvature,
    });
    out
}

fn penalty_matrixwith_local_block(
    total_dim: usize,
    coeff_range: Range<usize>,
    local: &Array2<f64>,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((total_dim, total_dim));
    out.slice_mut(s![coeff_range.clone(), coeff_range])
        .assign(local);
    out
}

fn fit_term_collectionwith_exact_spatial_adaptive_regularization(
    baseline: FittedTermCollection,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodFamily,
    options: &FitOptions,
    runtime_caches: &[SpatialOperatorRuntimeCache],
) -> Result<FittedTermCollection, EstimationError> {
    // Exact adaptive-regularization hyperfit.
    //
    // This replaces the old MM-plus-approximate hyperfit with the
    // exact pseudo-Laplace objective agreed in the math notes:
    //
    //   L_tilde(theta)
    //   = J(beta_hat(theta); theta) + 0.5 log det H(beta_hat(theta), theta),
    //
    // where:
    //   - beta_hat(theta) is the exact inner mode of the true nonquadratic
    //     Charbonnier-penalized objective,
    //   - theta contains:
    //       * retained quadratic log-lambdas for non-adaptive penalties,
    //       * one log-lambda per adaptive operator block,
    //       * three global log-epsilons shared by every adaptive spatial term,
    //   - H is the exact beta-Hessian of the true objective at the mode.
    //
    // Implementation structure:
    //   1. keep ordinary quadratic penalties that are unrelated to adaptive
    //      spatial terms in the standard outer-rho path;
    //   2. move the adaptive Charbonnier penalties into a one-block exact-Newton
    //      custom family so the inner solve uses the real model rather than an
    //      MM surrogate;
    //   3. expose exact psi-gradients for adaptive log-lambda / log-epsilon
    //      coordinates through the custom-family pseudo-Laplace hook;
    //   4. refit once at the optimized hyperparameters with all penalties frozen
    //      inside the exact family, so covariance and final diagnostics are
    //      computed on the same exact surface.
    let adaptive_opts = options.adaptive_regularization.clone().unwrap_or_default();
    let adaptive_penalty_indices = exact_spatial_adaptive_penalty_index_set(runtime_caches);
    let p_total = baseline.design.design.ncols();
    let mut retained_penalties = Vec::<Array2<f64>>::new();
    let mut retained_nullspace_dims = Vec::<usize>::new();
    let mut retained_log_lambdas = Vec::<f64>::new();
    let mut retained_global_indices = Vec::<usize>::new();
    let mut fixed_quadratichessian = Array2::<f64>::zeros((p_total, p_total));
    for (idx, bp) in baseline.design.penalties.iter().enumerate() {
        if adaptive_penalty_indices.contains(&idx) {
            continue;
        }
        retained_penalties.push(bp.to_global(p_total));
        retained_nullspace_dims.push(
            baseline
                .design
                .nullspace_dims
                .get(idx)
                .copied()
                .unwrap_or(0),
        );
        retained_log_lambdas.push(baseline.fit.lambdas[idx].max(1e-12).ln());
        retained_global_indices.push(idx);
        let r = &bp.col_range;
        fixed_quadratichessian
            .slice_mut(s![r.start..r.end, r.start..r.end])
            .scaled_add(baseline.fit.lambdas[idx], &bp.local);
    }

    let (eps_0_init, eps_g_init, eps_c_init) = compute_initial_epsilons(
        &baseline.fit.beta,
        runtime_caches,
        adaptive_opts.min_epsilon,
    )?;
    let mut initial_theta =
        Array1::<f64>::zeros(retained_penalties.len() + runtime_caches.len() * 3 + 3);
    for (idx, value) in retained_log_lambdas.iter().enumerate() {
        initial_theta[idx] = *value;
    }
    let mut at = retained_penalties.len();
    for cache in runtime_caches {
        initial_theta[at] = baseline.fit.lambdas[cache.mass_penalty_global_idx]
            .max(1e-12)
            .ln();
        initial_theta[at + 1] = baseline.fit.lambdas[cache.tension_penalty_global_idx]
            .max(1e-12)
            .ln();
        initial_theta[at + 2] = baseline.fit.lambdas[cache.stiffness_penalty_global_idx]
            .max(1e-12)
            .ln();
        at += 3;
    }
    initial_theta[at] = eps_0_init.max(adaptive_opts.min_epsilon).ln();
    initial_theta[at + 1] = eps_g_init.max(adaptive_opts.min_epsilon).ln();
    initial_theta[at + 2] = eps_c_init.max(adaptive_opts.min_epsilon).ln();

    let hyperspecs = build_spatial_adaptive_hyperspecs(runtime_caches.len());
    let derivative_blocks = vec![
        hyperspecs
            .iter()
            .enumerate()
            .map(|(_, _)| CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: Array2::<f64>::zeros((
                    baseline.design.design.nrows(),
                    baseline.design.design.ncols(),
                )),
                s_psi: Array2::<f64>::zeros((
                    baseline.design.design.ncols(),
                    baseline.design.design.ncols(),
                )),
                s_psi_components: None,
                x_psi_psi: None,
                s_psi_psi: None,
                s_psi_psi_components: None,
                implicit_operator: None,
                implicit_axis: 0,
                implicit_group_id: None,
            })
            .collect::<Vec<_>>(),
    ];

    let mixture_link_state = options
        .mixture_link
        .clone()
        .as_ref()
        .map(state_fromspec)
        .transpose()
        .map_err(EstimationError::InvalidInput)?;
    let sas_link_state = options
        .sas_link
        .map(|spec| {
            if matches!(family, LikelihoodFamily::BinomialBetaLogistic) {
                state_from_beta_logisticspec(spec)
            } else {
                state_from_sasspec(spec)
            }
        })
        .transpose()
        .map_err(EstimationError::InvalidInput)?;
    let shared_y = Arc::new(y.to_owned());
    let sharedweights = Arc::new(weights.to_owned());
    let shared_design = Arc::new(baseline.design.design.clone());
    let shared_offset = Arc::new(offset.to_owned());
    let shared_runtime_caches = Arc::new(runtime_caches.to_vec());
    let shared_hyperspecs = Arc::new(hyperspecs.clone());
    let zero_quadratic = Arc::new(Array2::<f64>::zeros((
        baseline.design.design.ncols(),
        baseline.design.design.ncols(),
    )));
    let base_family = SpatialAdaptiveExactFamily {
        family,
        mixture_link_state: mixture_link_state.clone(),
        sas_link_state,
        y: shared_y.clone(),
        weights: sharedweights.clone(),
        design: shared_design.clone(),
        offset: shared_offset.clone(),
        linear_constraints: baseline.design.linear_constraints.clone(),
        runtime_caches: shared_runtime_caches.clone(),
        adaptive_params: Vec::new(),
        fixed_quadratichessian: zero_quadratic.clone(),
        hyperspecs: shared_hyperspecs.clone(),
    };

    let rho_dim = retained_penalties.len();
    const EPSILON_LOG_WINDOW: f64 = 6.0;
    let eps_lower = Array1::from_iter((0..initial_theta.len()).map(|idx| {
        if idx < rho_dim + runtime_caches.len() * 3 {
            -30.0_f64
        } else {
            (initial_theta[idx] - EPSILON_LOG_WINDOW).max(adaptive_opts.min_epsilon.max(1e-12).ln())
        }
    }));
    let eps_upper = Array1::from_iter((0..initial_theta.len()).map(|idx| {
        if idx < rho_dim + runtime_caches.len() * 3 {
            30.0_f64
        } else {
            initial_theta[idx] + EPSILON_LOG_WINDOW
        }
    }));
    let blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: DesignMatrix::Dense(Arc::new(baseline.design.design.clone())),
        offset: offset.to_owned(),
        penalties: retained_penalties.iter().cloned().map(PenaltyMatrix::Dense).collect(),
        nullspace_dims: retained_nullspace_dims.clone(),
        initial_log_lambdas: Array1::from_vec(retained_log_lambdas.clone()),
        initial_beta: Some(baseline.fit.beta.clone()),
    };
    let outer_opts = BlockwiseFitOptions {
        inner_max_cycles: options.max_iter,
        inner_tol: options.tol,
        outer_max_iter: options.max_iter,
        outer_tol: options.tol,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };

    use crate::solver::outer_strategy::{
        ClosureObjective, Derivative, FallbackPolicy, HessianResult, OuterCapability, OuterConfig,
        OuterEval,
    };

    struct SpatialAdaptiveOuterState {
        warm_cache: Option<CustomFamilyWarmStart>,
        last_eval: Option<(
            Array1<f64>,
            f64,
            Array1<f64>,
            Array2<f64>,
            CustomFamilyWarmStart,
        )>,
    }

    let n_theta = initial_theta.len();

    // Clamp theta to the asymmetric epsilon bounds that run_outer's symmetric
    // rho_bound cannot express directly.
    let theta_bounds = Some((eps_lower.clone(), eps_upper.clone()));
    let clamp_theta = {
        let lo = eps_lower;
        let hi = eps_upper;
        move |theta: &Array1<f64>| -> Array1<f64> {
            let mut clamped = theta.clone();
            for i in 0..clamped.len() {
                clamped[i] = clamped[i].clamp(lo[i], hi[i]);
            }
            clamped
        }
    };

    let decode_theta = |theta: &Array1<f64>| -> (Array1<f64>, Vec<SpatialAdaptiveTermHyperParams>) {
        let rho = theta.slice(s![..rho_dim]).to_owned();
        let adaptive_lambda_start = rho_dim;
        let adaptive_lambda_end = adaptive_lambda_start + runtime_caches.len() * 3;
        let eps = [
            theta[adaptive_lambda_end].exp(),
            theta[adaptive_lambda_end + 1].exp(),
            theta[adaptive_lambda_end + 2].exp(),
        ];
        let adaptive_params = runtime_caches
            .iter()
            .enumerate()
            .map(|(cache_idx, _)| SpatialAdaptiveTermHyperParams {
                lambda: [
                    theta[adaptive_lambda_start + cache_idx * 3].exp(),
                    theta[adaptive_lambda_start + cache_idx * 3 + 1].exp(),
                    theta[adaptive_lambda_start + cache_idx * 3 + 2].exp(),
                ],
                epsilon: eps,
            })
            .collect::<Vec<_>>();
        (rho, adaptive_params)
    };

    let outer_config = OuterConfig {
        tolerance: options.tol,
        max_iter: options.max_iter,
        fd_step: 1e-4,
        bounds: theta_bounds,
        seed_config: crate::seeding::SeedConfig::default(),
        rho_bound: 30.0,
        heuristic_lambdas: None,
        initial_rho: Some(initial_theta.clone()),
        fallback_policy: FallbackPolicy::Automatic,
        screening_cap: None,
    };

    let mut obj = ClosureObjective {
        state: SpatialAdaptiveOuterState {
            warm_cache: None,
            last_eval: None,
        },
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: if crate::custom_family::joint_exact_analytic_outer_hessian_available(
                blockspec.design.ncols(),
            ) {
                Derivative::Analytic
            } else {
                Derivative::Unavailable
            },
            n_params: n_theta,
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: false,
            barrier_config: None,
        },
        cost_fn: |st: &mut SpatialAdaptiveOuterState, theta: &Array1<f64>| {
            let theta = clamp_theta(theta);
            let (rho, adaptive_params) = decode_theta(&theta);
            let family_eval =
                base_family.with_adaptive_params(adaptive_params, zero_quadratic.clone());
            let result = evaluate_custom_family_joint_hyper(
                &family_eval,
                std::slice::from_ref(&blockspec),
                &outer_opts,
                &rho,
                &derivative_blocks,
                st.warm_cache.as_ref(),
                false,
            )
            .map_err(|e| {
                EstimationError::RemlOptimizationFailed(format!(
                    "spatial adaptive cost eval failed: {e}"
                ))
            })?;
            st.warm_cache = Some(result.warm_start);
            Ok(result.objective)
        },
        eval_fn: |st: &mut SpatialAdaptiveOuterState, theta: &Array1<f64>| {
            let theta = clamp_theta(theta);

            // Return cached result if theta has not moved.
            if let Some((cached_theta, cached_cost, cached_grad, cached_hess, cached_warm)) =
                &st.last_eval
            {
                if cached_theta.len() == theta.len()
                    && cached_theta
                        .iter()
                        .zip(theta.iter())
                        .all(|(&a, &b)| (a - b).abs() <= 1e-12)
                {
                    st.warm_cache = Some(cached_warm.clone());
                    return Ok(OuterEval {
                        cost: *cached_cost,
                        gradient: cached_grad.clone(),
                        hessian: HessianResult::Analytic(cached_hess.clone()),
                    });
                }
            }

            let (rho, adaptive_params) = decode_theta(&theta);
            let family_eval =
                base_family.with_adaptive_params(adaptive_params, zero_quadratic.clone());
            let result = evaluate_custom_family_joint_hyper(
                &family_eval,
                std::slice::from_ref(&blockspec),
                &outer_opts,
                &rho,
                &derivative_blocks,
                st.warm_cache.as_ref(),
                true,
            )
            .map_err(|e| {
                EstimationError::RemlOptimizationFailed(format!(
                    "spatial adaptive eval failed: {e}"
                ))
            })?;
            if !result.objective.is_finite() || result.gradient.iter().any(|v| !v.is_finite()) {
                return Err(EstimationError::RemlOptimizationFailed(
                    "exact spatial adaptive objective returned non-finite values".to_string(),
                ));
            }
            let hessian = result.outer_hessian.ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "exact spatial adaptive objective did not return an outer Hessian".to_string(),
                )
            })?;
            if hessian.nrows() != theta.len() || hessian.ncols() != theta.len() {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "exact spatial adaptive outer Hessian shape mismatch: got {}x{}, expected {}x{}",
                    hessian.nrows(),
                    hessian.ncols(),
                    theta.len(),
                    theta.len(),
                )));
            }
            if hessian.iter().any(|v| !v.is_finite()) {
                return Err(EstimationError::RemlOptimizationFailed(
                    "exact spatial adaptive outer Hessian contained non-finite values".to_string(),
                ));
            }
            st.warm_cache = Some(result.warm_start.clone());
            st.last_eval = Some((
                theta.clone(),
                result.objective,
                result.gradient.clone(),
                hessian.clone(),
                result.warm_start,
            ));
            Ok(OuterEval {
                cost: result.objective,
                gradient: result.gradient,
                hessian: HessianResult::Analytic(hessian),
            })
        },
        reset_fn: Some(|st: &mut SpatialAdaptiveOuterState| {
            st.warm_cache = None;
            st.last_eval = None;
        }),
        efs_fn: None::<
            fn(
                &mut SpatialAdaptiveOuterState,
                &Array1<f64>,
            ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError>,
        >,
    };

    let outer_result = crate::solver::outer_strategy::run_outer(
        &mut obj,
        &outer_config,
        "exact spatial adaptive regularization",
    )
    .map_err(|e| {
        EstimationError::InvalidInput(format!(
            "exact spatial adaptive outer optimization failed: {e}"
        ))
    })?;
    let outer_iterations = outer_result.iterations;
    let outer_grad_norm = outer_result.final_grad_norm;
    let theta_star = outer_result.rho;
    let rho_star = theta_star.slice(s![..rho_dim]).to_owned();
    let adaptive_lambda_start = rho_dim;
    let adaptive_lambda_end = adaptive_lambda_start + runtime_caches.len() * 3;
    let eps_star = [
        theta_star[adaptive_lambda_end].exp(),
        theta_star[adaptive_lambda_end + 1].exp(),
        theta_star[adaptive_lambda_end + 2].exp(),
    ];
    let adaptive_params = runtime_caches
        .iter()
        .enumerate()
        .map(|(cache_idx, _)| SpatialAdaptiveTermHyperParams {
            lambda: [
                theta_star[adaptive_lambda_start + cache_idx * 3].exp(),
                theta_star[adaptive_lambda_start + cache_idx * 3 + 1].exp(),
                theta_star[adaptive_lambda_start + cache_idx * 3 + 2].exp(),
            ],
            epsilon: eps_star,
        })
        .collect::<Vec<_>>();
    let mut fixed_total = Array2::<f64>::zeros((
        baseline.design.design.ncols(),
        baseline.design.design.ncols(),
    ));
    for (idx, penalty) in retained_penalties.iter().enumerate() {
        fixed_total.scaled_add(rho_star[idx].exp(), penalty);
    }
    let final_family =
        base_family.with_adaptive_params(adaptive_params.clone(), Arc::new(fixed_total.clone()));
    let final_blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: DesignMatrix::Dense(Arc::new(baseline.design.design.clone())),
        offset: offset.to_owned(),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(baseline.fit.beta.clone()),
    };
    let final_fit = fit_custom_family(
        &final_family,
        &[final_blockspec],
        &BlockwiseFitOptions {
            inner_max_cycles: options.max_iter,
            inner_tol: options.tol,
            outer_max_iter: 1,
            outer_tol: options.tol,
            compute_covariance: true,
            ..BlockwiseFitOptions::default()
        },
    )
    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
    let beta = final_fit.block_states[0].beta.clone();
    let final_eval = final_family
        .exact_evaluation(&beta)
        .map_err(EstimationError::InvalidInput)?;
    let penalized_hessian = final_eval
        .totalobjectivehessian(&final_family.design)
        .map_err(EstimationError::InvalidInput)?;
    let beta_covariance = final_fit.covariance_conditional.clone();
    let beta_standard_errors = beta_covariance
        .as_ref()
        .map(|cov| Array1::from_iter((0..cov.nrows()).map(|i| cov[[i, i]].max(0.0).sqrt())));

    let mut full_lambdas = baseline.fit.lambdas.clone();
    for (idx, &global_idx) in retained_global_indices.iter().enumerate() {
        full_lambdas[global_idx] = rho_star[idx].exp();
    }
    for (cache_idx, cache) in runtime_caches.iter().enumerate() {
        full_lambdas[cache.mass_penalty_global_idx] = adaptive_params[cache_idx].lambda[0];
        full_lambdas[cache.tension_penalty_global_idx] = adaptive_params[cache_idx].lambda[1];
        full_lambdas[cache.stiffness_penalty_global_idx] = adaptive_params[cache_idx].lambda[2];
    }

    let deviance = match family {
        LikelihoodFamily::GaussianIdentity => y
            .iter()
            .zip(final_eval.obs.mu.iter())
            .zip(weights.iter())
            .map(|((&yy, &mu), &w)| w.max(0.0) * (yy - mu) * (yy - mu))
            .sum(),
        _ => -2.0 * final_eval.obs.log_likelihood,
    };
    let mut local_penalty_blocks =
        Vec::<Array2<f64>>::with_capacity(baseline.design.penalties.len());
    for (global_idx, bp) in baseline.design.penalties.iter().enumerate() {
        if adaptive_penalty_indices.contains(&global_idx) {
            let cache = runtime_caches
                .iter()
                .find(|cache| {
                    cache.mass_penalty_global_idx == global_idx
                        || cache.tension_penalty_global_idx == global_idx
                        || cache.stiffness_penalty_global_idx == global_idx
                })
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "missing runtime cache for adaptive penalty index {global_idx}"
                    ))
                })?;
            let cache_idx = runtime_caches
                .iter()
                .position(|c| {
                    c.mass_penalty_global_idx == global_idx
                        || c.tension_penalty_global_idx == global_idx
                        || c.stiffness_penalty_global_idx == global_idx
                })
                .ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "missing adaptive cache position for penalty index {global_idx}"
                    ))
                })?;
            let state = &final_eval.adaptive_states[cache_idx];
            let local = if cache.mass_penalty_global_idx == global_idx {
                scalar_operatorhessian(&cache.d0, &state.magnitude.betahessian_diag())
                    .mapv(|v| adaptive_params[cache_idx].lambda[0] * v)
            } else if cache.tension_penalty_global_idx == global_idx {
                grouped_operatorhessian(
                    &cache.d1,
                    cache.dimension,
                    &state.gradient.betahessian_blocks(),
                )?
                .mapv(|v| adaptive_params[cache_idx].lambda[1] * v)
            } else {
                scalar_operatorhessian(&cache.d2, &state.curvature.betahessian_diag())
                    .mapv(|v| adaptive_params[cache_idx].lambda[2] * v)
            };
            local_penalty_blocks.push(penalty_matrixwith_local_block(
                baseline.design.design.ncols(),
                cache.coeff_global_range.clone(),
                &local,
            ));
        } else {
            local_penalty_blocks.push(bp.to_global(p_total).mapv(|v| v * full_lambdas[global_idx]));
        }
    }
    let (edf_by_block, edf_total) = if let Some(cov) = beta_covariance.as_ref() {
        exact_bounded_edf(
            &local_penalty_blocks,
            &Array1::from_elem(local_penalty_blocks.len(), 1.0),
            cov,
        )?
    } else {
        (vec![0.0; local_penalty_blocks.len()], 0.0)
    };
    let stable_penalty_term =
        2.0 * final_eval.adaptive_penalty_value + beta.dot(&fixed_total.dot(&beta));
    let standard_deviation = match family {
        LikelihoodFamily::GaussianIdentity => {
            let denom = (y.len() as f64 - edf_total).max(1.0);
            (deviance / denom).sqrt()
        }
        _ => 1.0,
    };
    let maps = compute_spatial_adaptiveweights_for_beta(
        &beta,
        runtime_caches,
        eps_star[0],
        eps_star[1],
        eps_star[2],
        adaptive_opts.weight_floor,
        adaptive_opts.weight_ceiling,
    )?
    .into_iter()
    .zip(runtime_caches.iter())
    .map(|(w, cache)| AdaptiveSpatialMap {
        termname: cache.termname.clone(),
        feature_cols: cache.feature_cols.clone(),
        collocation_points: cache.collocation_points.clone(),
        inv_magweight: w.inv_magweight,
        invgradweight: w.invgradweight,
        inv_lapweight: w.inv_lapweight,
    })
    .collect::<Vec<_>>();
    let fitted_link = match family {
        LikelihoodFamily::BinomialMixture => mixture_link_state
            .clone()
            .map(|state| FittedLinkState::Mixture {
                state,
                covariance: None,
            })
            .unwrap_or(FittedLinkState::Standard(None)),
        LikelihoodFamily::BinomialSas => sas_link_state
            .map(|state| FittedLinkState::Sas {
                state,
                covariance: None,
            })
            .unwrap_or(FittedLinkState::Standard(None)),
        LikelihoodFamily::BinomialBetaLogistic => sas_link_state
            .map(|state| FittedLinkState::BetaLogistic {
                state,
                covariance: None,
            })
            .unwrap_or(FittedLinkState::Standard(None)),
        _ => FittedLinkState::Standard(None),
    };
    let max_abs_eta = final_eval
        .obs
        .eta
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let fitted = FittedTermCollection {
        fit: {
            let log_lambdas = full_lambdas.mapv(|v| v.max(1e-300).ln());
            let inf = FitInference {
                edf_by_block,
                edf_total,
                smoothing_correction: None,
                penalized_hessian: penalized_hessian.clone(),
                working_weights: final_eval.obs.fisherweight.clone(),
                working_response: {
                    let mut out = final_eval.obs.eta.clone();
                    for i in 0..out.len() {
                        let wi = final_eval.obs.fisherweight[i].max(1e-12);
                        out[i] += final_eval.obs.score[i] / wi;
                    }
                    out
                },
                reparam_qs: None,
                beta_covariance,
                beta_standard_errors,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
            };
            let geometry = Some(crate::estimate::FitGeometry {
                penalized_hessian,
                working_weights: inf.working_weights.clone(),
                working_response: inf.working_response.clone(),
            });
            let covariance_conditional = inf.beta_covariance.clone();
            let pirls_status_val = if final_fit.outer_converged {
                crate::pirls::PirlsStatus::Converged
            } else {
                crate::pirls::PirlsStatus::StalledAtValidMinimum
            };
            UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
                blocks: vec![crate::estimate::FittedBlock {
                    beta: beta.clone(),
                    role: crate::estimate::BlockRole::Mean,
                    edf: edf_total,
                    lambdas: full_lambdas.clone(),
                }],
                log_lambdas,
                lambdas: full_lambdas,
                log_likelihood: -0.5 * deviance,
                reml_score: final_fit.penalized_objective,
                stable_penalty_term,
                penalized_objective: final_fit.penalized_objective,
                outer_iterations,
                outer_converged: final_fit.outer_converged,
                outer_gradient_norm: outer_grad_norm,
                standard_deviation,
                covariance_conditional,
                covariance_corrected: None,
                inference: Some(inf),
                fitted_link,
                geometry,
                block_states: Vec::new(),
                pirls_status: pirls_status_val,
                max_abs_eta,
                constraint_kkt: None,
                artifacts: crate::estimate::FitArtifacts { pirls: None },
                inner_cycles: 0,
            })?
        },
        design: baseline.design,
        adaptive_diagnostics: Some(AdaptiveRegularizationDiagnostics {
            epsilon_0: eps_star[0],
            epsilon_g: eps_star[1],
            epsilon_c: eps_star[2],
            epsilon_outer_iterations: outer_iterations,
            mm_iterations: 0,
            converged: final_fit.outer_converged,
            maps,
        }),
    };
    enforce_term_constraint_feasibility(&fitted.design, &fitted.fit)?;
    Ok(fitted)
}

#[cfg(test)]
fn weighted_operator_gram_from_d1(
    d1: &Array2<f64>,
    weight: &Array1<f64>,
    dimension: usize,
) -> Array2<f64> {
    let mut weighted = d1.clone();
    for k in 0..weight.len() {
        // Kronecker/stacking derivation:
        // D1 rows are stacked as (k, axis), i.e. axis-major inside each point block.
        // For this storage, the correct gradient weight matrix is:
        //   W_g = diag(u_g) \otimes I_d,
        // which repeats u_g[k] across all d axes at collocation point k.
        //
        // Instead of forming W_g explicitly, compute W_g^(1/2) D1:
        // each row in block k is multiplied by sqrt(u_g[k]).
        //
        // Then K1 = (W_g^(1/2) D1)^T (W_g^(1/2) D1) = D1^T W_g D1.
        let w = weight[k].sqrt();
        for axis in 0..dimension {
            let row = k * dimension + axis;
            weighted.row_mut(row).mapv_inplace(|v| v * w);
        }
    }
    let gram = weighted.t().dot(&weighted);
    (&gram + &gram.t().to_owned()) * 0.5
}

#[cfg(test)]
fn weighted_operator_gram_from_d2(d2: &Array2<f64>, weight: &Array1<f64>) -> Array2<f64> {
    let mut weighted = d2.clone();
    for k in 0..weight.len() {
        // Laplacian block is already one row per collocation point:
        //   c_k = |Delta f(z_k)|, u_k = 1/sqrt(c_k^2 + eps_c^2), K2 = D2^T diag(u) D2.
        // Scale each row by sqrt(u_k).
        let w = weight[k].sqrt();
        weighted.row_mut(k).mapv_inplace(|v| v * w);
    }
    let gram = weighted.t().dot(&weighted);
    (&gram + &gram.t().to_owned()) * 0.5
}

fn adaptive_fit_options_base(options: &FitOptions, design: &TermCollectionDesign) -> FitOptions {
    FitOptions {
        mixture_link: options.mixture_link.clone(),
        optimize_mixture: options.optimize_mixture,
        sas_link: options.sas_link,
        optimize_sas: options.optimize_sas,
        compute_inference: options.compute_inference,
        max_iter: options.max_iter,
        tol: options.tol,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        adaptive_regularization: None,
        penalty_shrinkage_floor: options.penalty_shrinkage_floor,
    }
}

#[derive(Clone)]
struct BoundedLinearTermMeta {
    col_idx: usize,
    min: f64,
    max: f64,
    prior: BoundedCoefficientPriorSpec,
}

#[derive(Clone)]
struct BoundedLinearFamily {
    family: LikelihoodFamily,
    mixture_link_state: Option<MixtureLinkState>,
    sas_link_state: Option<SasLinkState>,
    y: Array1<f64>,
    weights: Array1<f64>,
    design: Array2<f64>,
    designzeroed: Array2<f64>,
    offset: Array1<f64>,
    bounded_terms: Vec<BoundedLinearTermMeta>,
}

#[derive(Clone)]
struct StandardFamilyObservationState {
    eta: Array1<f64>,
    mu: Array1<f64>,
    score: Array1<f64>,
    fisherweight: Array1<f64>,
    neghessian_eta: Array1<f64>,
    neghessian_eta_derivative: Array1<f64>,
    log_likelihood: f64,
}

fn bounded_logit(z: f64) -> f64 {
    let zc = z.clamp(1e-12, 1.0 - 1e-12);
    (zc / (1.0 - zc)).ln()
}

fn stable_sigmoid(theta: f64) -> f64 {
    if theta >= 0.0 {
        let exp_neg = (-theta).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = theta.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

fn stable_softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn bounded_latent_to_user(theta: f64, min: f64, max: f64) -> (f64, f64, f64) {
    let z = stable_sigmoid(theta);
    let width = max - min;
    let beta = min + width * z;
    let db_dtheta = width * z * (1.0 - z);
    (beta, z, db_dtheta)
}

fn bounded_latent_derivatives(theta: f64, min: f64, max: f64) -> (f64, f64, f64, f64, f64) {
    let z = stable_sigmoid(theta);
    let width = max - min;
    let s = z * (1.0 - z);
    let beta = min + width * z;
    let db_dtheta = width * s;
    let d2b_dtheta2 = width * s * (1.0 - 2.0 * z);
    let d3b_dtheta3 = width * s * (1.0 - 6.0 * z + 6.0 * z * z);
    (beta, z, db_dtheta, d2b_dtheta2, d3b_dtheta3)
}

fn bounded_prior_terms(theta: f64, prior: &BoundedCoefficientPriorSpec) -> (f64, f64, f64, f64) {
    let (a, b) = match prior {
        // `None` means constrained MLE with no extra prior term on the bounded coefficient.
        BoundedCoefficientPriorSpec::None => return (0.0, 0.0, 0.0, 0.0),
        // Uniform on the normalized user-scale coefficient z in (0, 1). In latent space this is
        // exactly the Jacobian term for the logistic transform, up to an additive width constant.
        BoundedCoefficientPriorSpec::Uniform => (1.0, 1.0),
        BoundedCoefficientPriorSpec::Beta { a, b } => (*a, *b),
    };
    let z = stable_sigmoid(theta).clamp(1e-12, 1.0 - 1e-12);
    let logp = a * z.ln() + b * (1.0 - z).ln();
    let grad = a - (a + b) * z;
    let neghess = (a + b) * z * (1.0 - z);
    let neghess_derivative = (a + b) * z * (1.0 - z) * (1.0 - 2.0 * z);
    (logp, grad, neghess, neghess_derivative)
}

fn evaluate_standard_familyobservations(
    family: LikelihoodFamily,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    eta: &Array1<f64>,
) -> Result<StandardFamilyObservationState, EstimationError> {
    const PROB_EPS: f64 = 1e-10;
    const MU_DERIV_EPS: f64 = 1e-12;
    let n = y.len();
    if weights.len() != n || eta.len() != n {
        return Err(EstimationError::InvalidInput(
            "bounded family observation size mismatch".to_string(),
        ));
    }

    let mut mu = Array1::<f64>::zeros(n);
    let mut score = Array1::<f64>::zeros(n);
    let mut fisherweight = Array1::<f64>::zeros(n);
    let mut neghessian_eta = Array1::<f64>::zeros(n);
    let mut neghessian_eta_derivative = Array1::<f64>::zeros(n);
    let mut log_likelihood = 0.0;

    for i in 0..n {
        let w = weights[i].max(0.0);
        let yi = y[i];
        let eta_i = eta[i];
        match family {
            LikelihoodFamily::GaussianIdentity => {
                let resid = yi - eta_i;
                mu[i] = eta_i;
                score[i] = w * resid;
                fisherweight[i] = w.max(MU_DERIV_EPS);
                neghessian_eta[i] = w;
                neghessian_eta_derivative[i] = 0.0;
                log_likelihood += -0.5 * w * resid * resid;
            }
            LikelihoodFamily::BinomialLogit => {
                let mu_i = stable_sigmoid(eta_i);
                let curvature = (mu_i * (1.0 - mu_i)).max(0.0);
                mu[i] = mu_i;
                score[i] = w * (yi - mu_i);
                fisherweight[i] = curvature.max(MU_DERIV_EPS);
                neghessian_eta[i] = curvature;
                neghessian_eta_derivative[i] = curvature * (1.0 - 2.0 * mu_i);
                let logmu = -stable_softplus(-eta_i);
                let log_one_minusmu = -stable_softplus(eta_i);
                log_likelihood += w * (yi * logmu + (1.0 - yi) * log_one_minusmu);
            }
            LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture => {
                let inverse_link = if let Some(state) = mixture_link_state {
                    Some(InverseLink::Mixture(state.clone()))
                } else if let Some(state) = sas_link_state {
                    Some(
                        if matches!(family, LikelihoodFamily::BinomialBetaLogistic) {
                            InverseLink::BetaLogistic(*state)
                        } else {
                            InverseLink::Sas(*state)
                        },
                    )
                } else {
                    None
                };
                let jet =
                    strategy_for_family(family, inverse_link.as_ref()).inverse_link_jet(eta_i)?;
                let mu_i_raw = jet.mu;
                let dmu_deta_raw = jet.d1;
                let mu_i: f64 = mu_i_raw.clamp(PROB_EPS, 1.0 - PROB_EPS);
                let dmu_deta = dmu_deta_raw.max(MU_DERIV_EPS);
                let d2mu_deta2 = jet.d2;
                let d3mu_deta3 = jet.d3;
                let var = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                let lmu = (yi - mu_i) / var;
                let lmumu = -(yi / (mu_i * mu_i)) - ((1.0 - yi) / ((1.0 - mu_i) * (1.0 - mu_i)));
                let lmumumu = 2.0 * yi / (mu_i * mu_i * mu_i)
                    - 2.0 * (1.0 - yi) / ((1.0 - mu_i) * (1.0 - mu_i) * (1.0 - mu_i));
                mu[i] = mu_i;
                score[i] = w * lmu * dmu_deta;
                fisherweight[i] = (w * dmu_deta * dmu_deta / var).max(MU_DERIV_EPS);
                neghessian_eta[i] = -w * (lmumu * dmu_deta * dmu_deta + lmu * d2mu_deta2);
                neghessian_eta_derivative[i] = -w
                    * (lmumumu * dmu_deta * dmu_deta * dmu_deta
                        + 3.0 * lmumu * dmu_deta * d2mu_deta2
                        + lmu * d3mu_deta3);
                log_likelihood += w * (yi * mu_i.ln() + (1.0 - yi) * (1.0 - mu_i).ln());
            }
            LikelihoodFamily::PoissonLog => {
                return Err(EstimationError::InvalidInput(
                    "bounded linear terms are not supported for PoissonLog fits".to_string(),
                ));
            }
            LikelihoodFamily::GammaLog => {
                return Err(EstimationError::InvalidInput(
                    "bounded linear terms are not supported for GammaLog fits".to_string(),
                ));
            }
            LikelihoodFamily::RoystonParmar => {
                return Err(EstimationError::InvalidInput(
                    "bounded linear terms are not supported for survival model fits".to_string(),
                ));
            }
        }
    }

    Ok(StandardFamilyObservationState {
        eta: eta.clone(),
        mu,
        score,
        fisherweight,
        neghessian_eta,
        neghessian_eta_derivative,
        log_likelihood,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpatialAdaptiveHyperKind {
    LogLambdaMagnitude,
    LogLambdaGradient,
    LogLambdaCurvature,
    LogEpsilonMagnitude,
    LogEpsilonGradient,
    LogEpsilonCurvature,
}

impl SpatialAdaptiveHyperKind {
    fn component_index(self) -> usize {
        match self {
            SpatialAdaptiveHyperKind::LogLambdaMagnitude
            | SpatialAdaptiveHyperKind::LogEpsilonMagnitude => 0,
            SpatialAdaptiveHyperKind::LogLambdaGradient
            | SpatialAdaptiveHyperKind::LogEpsilonGradient => 1,
            SpatialAdaptiveHyperKind::LogLambdaCurvature
            | SpatialAdaptiveHyperKind::LogEpsilonCurvature => 2,
        }
    }

    fn is_log_lambda(self) -> bool {
        matches!(
            self,
            SpatialAdaptiveHyperKind::LogLambdaMagnitude
                | SpatialAdaptiveHyperKind::LogLambdaGradient
                | SpatialAdaptiveHyperKind::LogLambdaCurvature
        )
    }

    fn is_log_epsilon(self) -> bool {
        matches!(
            self,
            SpatialAdaptiveHyperKind::LogEpsilonMagnitude
                | SpatialAdaptiveHyperKind::LogEpsilonGradient
                | SpatialAdaptiveHyperKind::LogEpsilonCurvature
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct SpatialAdaptiveHyperSpec {
    cache_index: usize,
    kind: SpatialAdaptiveHyperKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpatialAdaptiveExplicitSecondOrderKind {
    StructuralZero,
    LocalAlphaAlpha,
    LocalAlphaEta,
    SharedEtaEta,
}

impl SpatialAdaptiveHyperSpec {
    fn component_index(self) -> usize {
        self.kind.component_index()
    }

    fn explicit_second_order_kind(self, other: Self) -> SpatialAdaptiveExplicitSecondOrderKind {
        if self.component_index() != other.component_index() {
            return SpatialAdaptiveExplicitSecondOrderKind::StructuralZero;
        }
        match (
            self.kind.is_log_lambda(),
            other.kind.is_log_lambda(),
            self.kind.is_log_epsilon(),
            other.kind.is_log_epsilon(),
        ) {
            (true, true, false, false) if self.cache_index == other.cache_index => {
                SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaAlpha
            }
            (true, false, false, true) | (false, true, true, false) => {
                SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta
            }
            (false, false, true, true) => SpatialAdaptiveExplicitSecondOrderKind::SharedEtaEta,
            _ => SpatialAdaptiveExplicitSecondOrderKind::StructuralZero,
        }
    }
}

#[derive(Clone, Debug)]
struct SpatialAdaptiveTermHyperParams {
    lambda: [f64; 3],
    epsilon: [f64; 3],
}

#[derive(Clone)]
struct SpatialAdaptiveExactEvaluation {
    obs: StandardFamilyObservationState,
    adaptive_states: Vec<SpatialPenaltyExactState>,
    adaptive_penalty_value: f64,
    adaptive_penaltygradient: Array1<f64>,
    adaptive_penaltyhessian: Array2<f64>,
    fixed_quadraticvalue: f64,
    fixed_quadraticgradient: Array1<f64>,
    fixed_quadratichessian: Array2<f64>,
}

impl SpatialAdaptiveExactEvaluation {
    fn total_penalty_value(&self) -> f64 {
        self.adaptive_penalty_value + self.fixed_quadraticvalue
    }

    fn total_penaltygradient(&self) -> Array1<f64> {
        &self.adaptive_penaltygradient + &self.fixed_quadraticgradient
    }

    fn total_penaltyhessian(&self) -> Array2<f64> {
        &self.adaptive_penaltyhessian + &self.fixed_quadratichessian
    }

    fn totalobjectivehessian(&self, design: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut out = xt_diag_x_dense(design.view(), self.obs.neghessian_eta.view())?;
        out += &self.total_penaltyhessian();
        Ok(out)
    }
}

#[derive(Clone)]
struct SpatialAdaptiveExactFamily {
    family: LikelihoodFamily,
    mixture_link_state: Option<MixtureLinkState>,
    sas_link_state: Option<SasLinkState>,
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    design: Arc<Array2<f64>>,
    offset: Arc<Array1<f64>>,
    linear_constraints: Option<LinearInequalityConstraints>,
    runtime_caches: Arc<Vec<SpatialOperatorRuntimeCache>>,
    adaptive_params: Vec<SpatialAdaptiveTermHyperParams>,
    fixed_quadratichessian: Arc<Array2<f64>>,
    hyperspecs: Arc<Vec<SpatialAdaptiveHyperSpec>>,
}

impl SpatialAdaptiveExactFamily {
    fn with_adaptive_params(
        &self,
        adaptive_params: Vec<SpatialAdaptiveTermHyperParams>,
        fixed_quadratichessian: Arc<Array2<f64>>,
    ) -> Self {
        Self {
            family: self.family,
            mixture_link_state: self.mixture_link_state.clone(),
            sas_link_state: self.sas_link_state,
            y: self.y.clone(),
            weights: self.weights.clone(),
            design: self.design.clone(),
            offset: self.offset.clone(),
            linear_constraints: self.linear_constraints.clone(),
            runtime_caches: self.runtime_caches.clone(),
            adaptive_params,
            fixed_quadratichessian,
            hyperspecs: self.hyperspecs.clone(),
        }
    }

    fn total_eta(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.design.dot(beta) + self.offset.as_ref()
    }

    fn fixed_quadratic_terms(&self, beta: &Array1<f64>) -> (f64, Array1<f64>) {
        let grad = self.fixed_quadratichessian.dot(beta);
        let value = 0.5 * beta.dot(&grad);
        (value, grad)
    }

    fn zero_hyper_parts(&self) -> (Array1<f64>, Array2<f64>) {
        let total_dim = self.design.ncols();
        (
            Array1::<f64>::zeros(total_dim),
            Array2::<f64>::zeros((total_dim, total_dim)),
        )
    }

    fn embed_local_hyper_parts(
        &self,
        coeff_range: &Range<usize>,
        local_grad: &Array1<f64>,
        local_hess: &Array2<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let (mut beta_mixed, mut betahessian) = self.zero_hyper_parts();
        beta_mixed
            .slice_mut(s![coeff_range.clone()])
            .assign(local_grad);
        betahessian
            .slice_mut(s![coeff_range.clone(), coeff_range.clone()])
            .assign(local_hess);
        (beta_mixed, betahessian)
    }

    fn embed_local_hyper_hessian(
        &self,
        coeff_range: &Range<usize>,
        local_hess: &Array2<f64>,
    ) -> Array2<f64> {
        let total_dim = self.design.ncols();
        let mut out = Array2::<f64>::zeros((total_dim, total_dim));
        out.slice_mut(s![coeff_range.clone(), coeff_range.clone()])
            .assign(local_hess);
        out
    }

    fn adaptive_block_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        cache_idx: usize,
        component: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let cache = self
            .runtime_caches
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive cache index {} out of bounds", cache_idx))?;
        let params = self
            .adaptive_params
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive hyperparameter block {} out of bounds", cache_idx))?;
        let state = eval
            .adaptive_states
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive exact state index {} out of bounds", cache_idx))?;

        match component {
            0 => {
                let lambda = params.lambda[0];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(&cache.d0, &state.magnitude.betagradient_coeff());
                let betahessian_local =
                    lambda * scalar_operatorhessian(&cache.d0, &state.magnitude.betahessian_diag());
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.magnitude.penalty_value(),
                    beta_mixed,
                    betahessian,
                ))
            }
            1 => {
                let lambda = params.lambda[1];
                let beta_mixed_local = lambda
                    * grouped_operatorgradient(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.betagradient_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                let betahessian_local = lambda
                    * grouped_operatorhessian(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.betahessian_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.gradient.penalty_value(),
                    beta_mixed,
                    betahessian,
                ))
            }
            2 => {
                let lambda = params.lambda[2];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(&cache.d2, &state.curvature.betagradient_coeff());
                let betahessian_local =
                    lambda * scalar_operatorhessian(&cache.d2, &state.curvature.betahessian_diag());
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.curvature.penalty_value(),
                    beta_mixed,
                    betahessian,
                ))
            }
            _ => Err(format!("invalid adaptive component index {}", component)),
        }
    }

    fn adaptive_block_log_epsilon_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        cache_idx: usize,
        component: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let cache = self
            .runtime_caches
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive cache index {} out of bounds", cache_idx))?;
        let params = self
            .adaptive_params
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive hyperparameter block {} out of bounds", cache_idx))?;
        let state = eval
            .adaptive_states
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive exact state index {} out of bounds", cache_idx))?;

        match component {
            0 => {
                let lambda = params.lambda[0];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(
                        &cache.d0,
                        &state.magnitude.log_epsilon_betagradient_coeff(),
                    );
                let betahessian_local = lambda
                    * scalar_operatorhessian(
                        &cache.d0,
                        &state.magnitude.log_epsilon_betahessian_diag(),
                    );
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.magnitude.log_epsilon_gradient_terms().sum(),
                    beta_mixed,
                    betahessian,
                ))
            }
            1 => {
                let lambda = params.lambda[1];
                let beta_mixed_local = lambda
                    * grouped_operatorgradient(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.log_epsilon_betagradient_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                let betahessian_local = lambda
                    * grouped_operatorhessian(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.log_epsilon_betahessian_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.gradient.log_epsilon_gradient_terms().sum(),
                    beta_mixed,
                    betahessian,
                ))
            }
            2 => {
                let lambda = params.lambda[2];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(
                        &cache.d2,
                        &state.curvature.log_epsilon_betagradient_coeff(),
                    );
                let betahessian_local = lambda
                    * scalar_operatorhessian(
                        &cache.d2,
                        &state.curvature.log_epsilon_betahessian_diag(),
                    );
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.curvature.log_epsilon_gradient_terms().sum(),
                    beta_mixed,
                    betahessian,
                ))
            }
            _ => Err(format!("invalid adaptive component index {}", component)),
        }
    }

    fn adaptive_block_log_epsilon_second_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        cache_idx: usize,
        component: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let cache = self
            .runtime_caches
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive cache index {} out of bounds", cache_idx))?;
        let params = self
            .adaptive_params
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive hyperparameter block {} out of bounds", cache_idx))?;
        let state = eval
            .adaptive_states
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive exact state index {} out of bounds", cache_idx))?;

        match component {
            0 => {
                let lambda = params.lambda[0];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(
                        &cache.d0,
                        &state.magnitude.log_epsilon_beta_mixed_second_coeff(),
                    );
                let betahessian_local = lambda
                    * scalar_operatorhessian(
                        &cache.d0,
                        &state.magnitude.log_epsilon_betahessian_second_diag(),
                    );
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.magnitude.log_epsilon_hessian_terms().sum(),
                    beta_mixed,
                    betahessian,
                ))
            }
            1 => {
                let lambda = params.lambda[1];
                let beta_mixed_local = lambda
                    * grouped_operatorgradient(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.log_epsilon_beta_mixed_second_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                let betahessian_local = lambda
                    * grouped_operatorhessian(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.log_epsilon_betahessian_second_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.gradient.log_epsilon_hessian_terms().sum(),
                    beta_mixed,
                    betahessian,
                ))
            }
            2 => {
                let lambda = params.lambda[2];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(
                        &cache.d2,
                        &state.curvature.log_epsilon_beta_mixed_second_coeff(),
                    );
                let betahessian_local = lambda
                    * scalar_operatorhessian(
                        &cache.d2,
                        &state.curvature.log_epsilon_betahessian_second_diag(),
                    );
                let (beta_mixed, betahessian) = self.embed_local_hyper_parts(
                    &cache.coeff_global_range,
                    &beta_mixed_local,
                    &betahessian_local,
                );
                Ok((
                    lambda * state.curvature.log_epsilon_hessian_terms().sum(),
                    beta_mixed,
                    betahessian,
                ))
            }
            _ => Err(format!("invalid adaptive component index {}", component)),
        }
    }

    fn adaptive_shared_log_epsilon_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        component: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Exact shared-log-epsilon first-order pieces:
        //
        //   J_{eta_p}         = sum_m lambda_{m,p} U_{m,p,eta},
        //   J_{beta,eta_p}    = sum_m lambda_{m,p} U_{m,p,beta eta},
        //   J_{beta,beta,eta} = sum_m lambda_{m,p} U_{m,p,beta beta eta}.
        let (mut score, mut hessian) = self.zero_hyper_parts();
        let mut objective = 0.0;
        for cache_idx in 0..self.runtime_caches.len() {
            let (local_objective, local_score, local_hessian) =
                self.adaptive_block_log_epsilon_parts(eval, cache_idx, component)?;
            objective += local_objective;
            score += &local_score;
            hessian += &local_hessian;
        }
        Ok((objective, score, hessian))
    }

    fn adaptive_shared_log_epsilon_second_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        component: usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Exact shared-log-epsilon second-order pieces:
        //
        //   J_{eta_p,eta_p}            = sum_m lambda_{m,p} U_{m,p,eta eta},
        //   J_{beta,eta_p,eta_p}       = sum_m lambda_{m,p} U_{m,p,beta eta eta},
        //   J_{beta,beta,eta_p,eta_p}  = sum_m lambda_{m,p} U_{m,p,beta beta eta eta}.
        let (mut score, mut hessian) = self.zero_hyper_parts();
        let mut objective = 0.0;
        for cache_idx in 0..self.runtime_caches.len() {
            let (local_objective, local_score, local_hessian) =
                self.adaptive_block_log_epsilon_second_parts(eval, cache_idx, component)?;
            objective += local_objective;
            score += &local_score;
            hessian += &local_hessian;
        }
        Ok((objective, score, hessian))
    }

    fn adaptive_shared_log_epsilon_drift(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        component: usize,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        // Exact shared-log-epsilon Hessian drift:
        //
        //   T_{eta_p}[u] = sum_m lambda_{m,p} D_beta(U_{m,p,beta beta eta})[u].
        let total_dim = self.design.ncols();
        let mut total = Array2::<f64>::zeros((total_dim, total_dim));
        for cache_idx in 0..self.runtime_caches.len() {
            total +=
                &self.adaptive_block_log_epsilon_drift(eval, cache_idx, component, direction)?;
        }
        Ok(total)
    }

    fn adaptive_explicit_second_order_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        left: SpatialAdaptiveHyperSpec,
        right: SpatialAdaptiveHyperSpec,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Structural sparsity from the adaptive penalty algebra:
        //
        //   - alpha_{m,p} / alpha_{n,r} is nonzero only when (m,p) = (n,r),
        //   - alpha_{m,p} / eta_r is nonzero only when p = r,
        //   - eta_p / eta_r is nonzero only when p = r,
        //
        // with eta_p contributions summed over all adaptive terms m because the
        // three log-epsilon coordinates are shared globally by penalty type.
        match left.explicit_second_order_kind(right) {
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero => {
                let (score, hessian) = self.zero_hyper_parts();
                Ok((0.0, score, hessian))
            }
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaAlpha => {
                self.adaptive_block_parts(eval, left.cache_index, left.component_index())
            }
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta => {
                let local_alpha = if left.kind.is_log_lambda() {
                    left
                } else {
                    right
                };
                self.adaptive_block_log_epsilon_parts(
                    eval,
                    local_alpha.cache_index,
                    local_alpha.component_index(),
                )
            }
            SpatialAdaptiveExplicitSecondOrderKind::SharedEtaEta => {
                self.adaptive_shared_log_epsilon_second_parts(eval, left.component_index())
            }
        }
    }

    fn adaptive_block_drift(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        cache_idx: usize,
        component: usize,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let cache = self
            .runtime_caches
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive cache index {} out of bounds", cache_idx))?;
        let params = self
            .adaptive_params
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive hyperparameter block {} out of bounds", cache_idx))?;
        let state = eval
            .adaptive_states
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive exact state index {} out of bounds", cache_idx))?;
        let direction_local = direction.slice(s![cache.coeff_global_range.clone()]);

        let local_hessian = match component {
            0 => {
                let d0_u = cache.d0.dot(&direction_local);
                params.lambda[0]
                    * scalar_operatorhessian(
                        &cache.d0,
                        &state.magnitude.directionalhessian_diag(&d0_u),
                    )
            }
            1 => {
                let d1_u = cache.d1.dot(&direction_local);
                params.lambda[1]
                    * grouped_operatorhessian(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.directionalhessian_blocks(
                            &collocationgradient_blocks(&d1_u, cache.dimension)
                                .map_err(|e| e.to_string())?,
                        ),
                    )
                    .map_err(|e| e.to_string())?
            }
            2 => {
                let d2_u = cache.d2.dot(&direction_local);
                params.lambda[2]
                    * scalar_operatorhessian(
                        &cache.d2,
                        &state.curvature.directionalhessian_diag(&d2_u),
                    )
            }
            _ => return Err(format!("invalid adaptive component index {}", component)),
        };

        Ok(self.embed_local_hyper_hessian(&cache.coeff_global_range, &local_hessian))
    }

    fn adaptive_block_log_epsilon_drift(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        cache_idx: usize,
        component: usize,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let cache = self
            .runtime_caches
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive cache index {} out of bounds", cache_idx))?;
        let params = self
            .adaptive_params
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive hyperparameter block {} out of bounds", cache_idx))?;
        let state = eval
            .adaptive_states
            .get(cache_idx)
            .ok_or_else(|| format!("adaptive exact state index {} out of bounds", cache_idx))?;
        let direction_local = direction.slice(s![cache.coeff_global_range.clone()]);

        let local_hessian = match component {
            0 => {
                let d0_u = cache.d0.dot(&direction_local);
                params.lambda[0]
                    * scalar_operatorhessian(
                        &cache.d0,
                        &state
                            .magnitude
                            .log_epsilon_betahessian_directional_diag(&d0_u),
                    )
            }
            1 => {
                let d1_u = cache.d1.dot(&direction_local);
                params.lambda[1]
                    * grouped_operatorhessian(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.log_epsilon_betahessian_directional_blocks(
                            &collocationgradient_blocks(&d1_u, cache.dimension)
                                .map_err(|e| e.to_string())?,
                        ),
                    )
                    .map_err(|e| e.to_string())?
            }
            2 => {
                let d2_u = cache.d2.dot(&direction_local);
                params.lambda[2]
                    * scalar_operatorhessian(
                        &cache.d2,
                        &state
                            .curvature
                            .log_epsilon_betahessian_directional_diag(&d2_u),
                    )
            }
            _ => return Err(format!("invalid adaptive component index {}", component)),
        };

        Ok(self.embed_local_hyper_hessian(&cache.coeff_global_range, &local_hessian))
    }

    fn adaptive_hyper_parts(
        &self,
        eval: &SpatialAdaptiveExactEvaluation,
        hyper: SpatialAdaptiveHyperSpec,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        match hyper.kind {
            SpatialAdaptiveHyperKind::LogLambdaMagnitude => {
                let (mut beta_mixed, mut betahessian) = self.zero_hyper_parts();
                let cache = self.runtime_caches.get(hyper.cache_index).ok_or_else(|| {
                    format!("adaptive cache index {} out of bounds", hyper.cache_index)
                })?;
                let params = self.adaptive_params.get(hyper.cache_index).ok_or_else(|| {
                    format!(
                        "adaptive hyperparameter block {} out of bounds",
                        hyper.cache_index
                    )
                })?;
                let state = eval.adaptive_states.get(hyper.cache_index).ok_or_else(|| {
                    format!(
                        "adaptive exact state index {} out of bounds",
                        hyper.cache_index
                    )
                })?;
                let lambda = params.lambda[0];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(&cache.d0, &state.magnitude.betagradient_coeff());
                let betahessian_local =
                    lambda * scalar_operatorhessian(&cache.d0, &state.magnitude.betahessian_diag());
                beta_mixed
                    .slice_mut(s![cache.coeff_global_range.clone()])
                    .assign(&beta_mixed_local);
                betahessian
                    .slice_mut(s![
                        cache.coeff_global_range.clone(),
                        cache.coeff_global_range.clone()
                    ])
                    .assign(&betahessian_local);
                Ok((
                    lambda * state.magnitude.penalty_value(),
                    beta_mixed,
                    betahessian,
                ))
            }
            SpatialAdaptiveHyperKind::LogLambdaGradient => {
                let (mut beta_mixed, mut betahessian) = self.zero_hyper_parts();
                let cache = self.runtime_caches.get(hyper.cache_index).ok_or_else(|| {
                    format!("adaptive cache index {} out of bounds", hyper.cache_index)
                })?;
                let params = self.adaptive_params.get(hyper.cache_index).ok_or_else(|| {
                    format!(
                        "adaptive hyperparameter block {} out of bounds",
                        hyper.cache_index
                    )
                })?;
                let state = eval.adaptive_states.get(hyper.cache_index).ok_or_else(|| {
                    format!(
                        "adaptive exact state index {} out of bounds",
                        hyper.cache_index
                    )
                })?;
                let lambda = params.lambda[1];
                let beta_mixed_local = lambda
                    * grouped_operatorgradient(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.betagradient_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                let betahessian_local = lambda
                    * grouped_operatorhessian(
                        &cache.d1,
                        cache.dimension,
                        &state.gradient.betahessian_blocks(),
                    )
                    .map_err(|e| e.to_string())?;
                beta_mixed
                    .slice_mut(s![cache.coeff_global_range.clone()])
                    .assign(&beta_mixed_local);
                betahessian
                    .slice_mut(s![
                        cache.coeff_global_range.clone(),
                        cache.coeff_global_range.clone()
                    ])
                    .assign(&betahessian_local);
                Ok((
                    lambda * state.gradient.penalty_value(),
                    beta_mixed,
                    betahessian,
                ))
            }
            SpatialAdaptiveHyperKind::LogLambdaCurvature => {
                let (mut beta_mixed, mut betahessian) = self.zero_hyper_parts();
                let cache = self.runtime_caches.get(hyper.cache_index).ok_or_else(|| {
                    format!("adaptive cache index {} out of bounds", hyper.cache_index)
                })?;
                let params = self.adaptive_params.get(hyper.cache_index).ok_or_else(|| {
                    format!(
                        "adaptive hyperparameter block {} out of bounds",
                        hyper.cache_index
                    )
                })?;
                let state = eval.adaptive_states.get(hyper.cache_index).ok_or_else(|| {
                    format!(
                        "adaptive exact state index {} out of bounds",
                        hyper.cache_index
                    )
                })?;
                let lambda = params.lambda[2];
                let beta_mixed_local = lambda
                    * scalar_operatorgradient(&cache.d2, &state.curvature.betagradient_coeff());
                let betahessian_local =
                    lambda * scalar_operatorhessian(&cache.d2, &state.curvature.betahessian_diag());
                beta_mixed
                    .slice_mut(s![cache.coeff_global_range.clone()])
                    .assign(&beta_mixed_local);
                betahessian
                    .slice_mut(s![
                        cache.coeff_global_range.clone(),
                        cache.coeff_global_range.clone()
                    ])
                    .assign(&betahessian_local);
                Ok((
                    lambda * state.curvature.penalty_value(),
                    beta_mixed,
                    betahessian,
                ))
            }
            SpatialAdaptiveHyperKind::LogEpsilonMagnitude
            | SpatialAdaptiveHyperKind::LogEpsilonGradient
            | SpatialAdaptiveHyperKind::LogEpsilonCurvature => {
                self.adaptive_shared_log_epsilon_parts(eval, hyper.component_index())
            }
        }
    }

    fn exact_evaluation(
        &self,
        beta: &Array1<f64>,
    ) -> Result<SpatialAdaptiveExactEvaluation, String> {
        let eta = self.total_eta(beta);
        let obs = evaluate_standard_familyobservations(
            self.family,
            self.mixture_link_state.as_ref(),
            self.sas_link_state.as_ref(),
            &self.y,
            &self.weights,
            &eta,
        )
        .map_err(|e| e.to_string())?;
        let p = beta.len();
        let mut penalty_value = 0.0;
        let mut penaltygradient = Array1::<f64>::zeros(p);
        let mut penaltyhessian = Array2::<f64>::zeros((p, p));
        let mut adaptive_states = Vec::with_capacity(self.runtime_caches.len());

        for (cache_idx, cache) in self.runtime_caches.iter().enumerate() {
            let params = self.adaptive_params.get(cache_idx).ok_or_else(|| {
                format!(
                    "missing adaptive parameter block for cache {}",
                    cache.termname
                )
            })?;
            let beta_local = beta.slice(s![cache.coeff_global_range.clone()]);
            let state =
                SpatialPenaltyExactState::from_beta_local(beta_local, cache, params.epsilon)
                    .map_err(|e| e.to_string())?;

            let g0 = scalar_operatorgradient(&cache.d0, &state.magnitude.betagradient_coeff());
            let gg = grouped_operatorgradient(
                &cache.d1,
                cache.dimension,
                &state.gradient.betagradient_blocks(),
            )
            .map_err(|e| e.to_string())?;
            let gc = scalar_operatorgradient(&cache.d2, &state.curvature.betagradient_coeff());
            let h0 = scalar_operatorhessian(&cache.d0, &state.magnitude.betahessian_diag());
            let hg = grouped_operatorhessian(
                &cache.d1,
                cache.dimension,
                &state.gradient.betahessian_blocks(),
            )
            .map_err(|e| e.to_string())?;
            let hc = scalar_operatorhessian(&cache.d2, &state.curvature.betahessian_diag());

            let lambda0 = params.lambda[0];
            let lambdag = params.lambda[1];
            let lambdac = params.lambda[2];

            penalty_value += lambda0 * state.magnitude.penalty_value();
            penalty_value += lambdag * state.gradient.penalty_value();
            penalty_value += lambdac * state.curvature.penalty_value();

            let range = cache.coeff_global_range.clone();
            {
                let mut grad_local = penaltygradient.slice_mut(s![range.clone()]);
                grad_local += &(g0.mapv(|v| lambda0 * v));
                grad_local += &(gg.mapv(|v| lambdag * v));
                grad_local += &(gc.mapv(|v| lambdac * v));
            }
            {
                let mut h_local = penaltyhessian.slice_mut(s![range.clone(), range]);
                h_local += &h0.mapv(|v| lambda0 * v);
                h_local += &hg.mapv(|v| lambdag * v);
                h_local += &hc.mapv(|v| lambdac * v);
            }

            adaptive_states.push(state);
        }

        let (fixed_quadraticvalue, fixed_quadraticgradient) = self.fixed_quadratic_terms(beta);
        Ok(SpatialAdaptiveExactEvaluation {
            obs,
            adaptive_states,
            adaptive_penalty_value: penalty_value,
            adaptive_penaltygradient: penaltygradient,
            adaptive_penaltyhessian: penaltyhessian,
            fixed_quadraticvalue,
            fixed_quadraticgradient,
            fixed_quadratichessian: self.fixed_quadratichessian.as_ref().clone(),
        })
    }

    fn exacthessian_directional_derivative_from_evaluation(
        &self,
        _: &Array1<f64>,
        eval: &SpatialAdaptiveExactEvaluation,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let d_eta = self.design.dot(direction);
        let mut total = xt_diag_x_dense(
            self.design.view(),
            (&eval.obs.neghessian_eta_derivative * &d_eta).view(),
        )?;
        for (cache_idx, cache) in self.runtime_caches.iter().enumerate() {
            let params = self.adaptive_params.get(cache_idx).ok_or_else(|| {
                format!(
                    "missing adaptive parameter block for cache {}",
                    cache.termname
                )
            })?;
            let state = eval
                .adaptive_states
                .get(cache_idx)
                .ok_or_else(|| format!("missing adaptive state for cache {}", cache.termname))?;
            let direction_local = direction.slice(s![cache.coeff_global_range.clone()]);
            let d0_u = cache.d0.dot(&direction_local);
            let d1_u = cache.d1.dot(&direction_local);
            let d2_u = cache.d2.dot(&direction_local);
            let h0 =
                scalar_operatorhessian(&cache.d0, &state.magnitude.directionalhessian_diag(&d0_u))
                    .mapv(|v| params.lambda[0] * v);
            let hg = grouped_operatorhessian(
                &cache.d1,
                cache.dimension,
                &state.gradient.directionalhessian_blocks(
                    &collocationgradient_blocks(&d1_u, cache.dimension)
                        .map_err(|e| e.to_string())?,
                ),
            )
            .map_err(|e| e.to_string())?
            .mapv(|v| params.lambda[1] * v);
            let hc =
                scalar_operatorhessian(&cache.d2, &state.curvature.directionalhessian_diag(&d2_u))
                    .mapv(|v| params.lambda[2] * v);
            let range = cache.coeff_global_range.clone();
            let mut local = total.slice_mut(s![range.clone(), range]);
            local += &h0;
            local += &hg;
            local += &hc;
        }
        Ok(total)
    }
}

impl CustomFamily for SpatialAdaptiveExactFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = &expect_single_block_state(block_states, "spatial adaptive exact family")?.beta;
        let eval = self.exact_evaluation(beta)?;
        let mut gradient = fast_atv(&self.design, &eval.obs.score);
        gradient -= &eval.total_penaltygradient();
        let mut hessian = xt_diag_x_dense(self.design.view(), eval.obs.neghessian_eta.view())?;
        hessian += &eval.total_penaltyhessian();
        Ok(FamilyEvaluation {
            log_likelihood: eval.obs.log_likelihood - eval.total_penalty_value(),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::PseudoLaplace
    }

    fn exact_newton_allows_semidefinitehessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &expect_single_block_state(block_states, "spatial adaptive exact family")?.beta;
        let eval = self.exact_evaluation(beta)?;
        Ok(Some(eval.totalobjectivehessian(&self.design)?))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        expect_block_idx_zero(block_idx, "spatial adaptive exact family", "")?;
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &expect_single_block_state(block_states, "spatial adaptive exact family")?.beta;
        if d_beta_flat.len() != beta.len() {
            return Err(format!(
                "spatial adaptive exact family direction length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                beta.len()
            ));
        }
        let eval = self.exact_evaluation(beta)?;
        Ok(Some(
            self.exacthessian_directional_derivative_from_evaluation(beta, &eval, d_beta_flat)?,
        ))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        expect_block_idx_zero(block_idx, "spatial adaptive exact family", "")?;
        Ok(self.linear_constraints.clone())
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if block_states.len() != 1 || specs.len() != 1 || derivative_blocks.len() != 1 {
            return Err(format!(
                "spatial adaptive exact family expects one block/state/spec/psi payload, got states={} specs={} deriv_blocks={}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        derivative_blocks[0]
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let hyper = self
            .hyperspecs
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let beta = &block_states[0].beta;
        let eval = self.exact_evaluation(beta)?;
        let (direct, beta_mixed, betahessian_explicit) =
            self.adaptive_hyper_parts(&eval, *hyper)?;

        // Exact pseudo-Laplace psi-gradient.
        //
        // For one hyperparameter coordinate a we use the exact formula
        //
        //   d/da L_tilde
        //   = J_a + 0.5 tr(H^{-1} Hdot_a),
        //
        // with
        //
        //   H u_a   = J_{beta,a},
        //   beta_a  = -u_a,
        //   Hdot_a  = J_{beta,beta,a} + D_beta(H)[beta_a]
        //           = J_{beta,beta,a} - D_beta(H)[u_a].
        //
        // Here:
        //   - `direct` is J_a,
        //   - `beta_mixed` is J_{beta,a},
        //   - `betahessian_explicit` is J_{beta,beta,a},
        //   - `exacthessian_directional_derivative_from_evaluation(..., u)` returns
        //     D_beta(H)[u] for the exact likelihood-plus-Charbonnier model.
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: direct,
            score_psi: beta_mixed,
            hessian_psi: betahessian_explicit,
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<crate::custom_family::ExactNewtonJointPsiSecondOrderTerms>, String> {
        if block_states.len() != 1 || specs.len() != 1 || derivative_blocks.len() != 1 {
            return Err(format!(
                "spatial adaptive exact family expects one block/state/spec/psi payload, got states={} specs={} deriv_blocks={}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        derivative_blocks[0]
            .get(psi_i)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_i))?;
        derivative_blocks[0]
            .get(psi_j)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_j))?;
        let hyper_i = self
            .hyperspecs
            .get(psi_i)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_i))?;
        let hyper_j = self
            .hyperspecs
            .get(psi_j)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_j))?;
        let beta = &block_states[0].beta;
        let eval = self.exact_evaluation(beta)?;
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi) =
            self.adaptive_explicit_second_order_parts(&eval, *hyper_i, *hyper_j)?;

        Ok(Some(
            crate::custom_family::ExactNewtonJointPsiSecondOrderTerms {
                objective_psi_psi,
                score_psi_psi,
                hessian_psi_psi,
            },
        ))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != 1 || specs.len() != 1 || derivative_blocks.len() != 1 {
            return Err(format!(
                "spatial adaptive exact family expects one block/state/spec/psi payload, got states={} specs={} deriv_blocks={}",
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        let beta = &block_states[0].beta;
        if direction.len() != beta.len() {
            return Err(format!(
                "spatial adaptive exact family direction length mismatch: got {}, expected {}",
                direction.len(),
                beta.len()
            ));
        }
        derivative_blocks[0]
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let hyper = self
            .hyperspecs
            .get(psi_index)
            .ok_or_else(|| format!("adaptive psi index {} out of bounds", psi_index))?;
        let eval = self.exact_evaluation(beta)?;
        let drift = match hyper.kind {
            SpatialAdaptiveHyperKind::LogLambdaMagnitude
            | SpatialAdaptiveHyperKind::LogLambdaGradient
            | SpatialAdaptiveHyperKind::LogLambdaCurvature => self.adaptive_block_drift(
                &eval,
                hyper.cache_index,
                hyper.kind.component_index(),
                direction,
            )?,
            SpatialAdaptiveHyperKind::LogEpsilonMagnitude
            | SpatialAdaptiveHyperKind::LogEpsilonGradient
            | SpatialAdaptiveHyperKind::LogEpsilonCurvature => self
                .adaptive_shared_log_epsilon_drift(
                    &eval,
                    hyper.kind.component_index(),
                    direction,
                )?,
        };
        Ok(Some(drift))
    }
}

fn expect_single_block_state<'a>(
    block_states: &'a [ParameterBlockState],
    family_name: &str,
) -> Result<&'a ParameterBlockState, String> {
    if block_states.len() != 1 {
        return Err(format!(
            "{family_name} expects 1 block, got {}",
            block_states.len()
        ));
    }
    Ok(&block_states[0])
}

fn expect_block_idx_zero(block_idx: usize, family_name: &str, context: &str) -> Result<(), String> {
    if block_idx != 0 {
        return Err(format!(
            "{family_name} expects block_idx 0{context}, got {block_idx}"
        ));
    }
    Ok(())
}

impl BoundedLinearFamily {
    fn bounded_term_derivative_data(
        &self,
        latent_beta: &Array1<f64>,
    ) -> (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        let p = latent_beta.len();
        let mut beta_user = latent_beta.clone();
        let mut jac_diag = Array1::<f64>::ones(p);
        let mut second_diag = Array1::<f64>::zeros(p);
        let mut third_diag = Array1::<f64>::zeros(p);
        let mut priorthird = Array1::<f64>::zeros(p);
        for term in &self.bounded_terms {
            let (beta, _, db_dtheta, d2b_dtheta2, d3b_dtheta3) =
                bounded_latent_derivatives(latent_beta[term.col_idx], term.min, term.max);
            beta_user[term.col_idx] = beta;
            jac_diag[term.col_idx] = db_dtheta;
            second_diag[term.col_idx] = d2b_dtheta2;
            third_diag[term.col_idx] = d3b_dtheta3;
            let (_, _, _, prior_neghess_derivative) =
                bounded_prior_terms(latent_beta[term.col_idx], &term.prior);
            priorthird[term.col_idx] = prior_neghess_derivative;
        }
        (beta_user, jac_diag, second_diag, third_diag, priorthird)
    }

    fn user_beta_and_jacobian(&self, latent_beta: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let (beta_user, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta);
        (beta_user, jac_diag)
    }

    fn nonlinear_offset_from_latent(&self, latent_beta: &Array1<f64>) -> Array1<f64> {
        let mut offset = self.offset.clone();
        for term in &self.bounded_terms {
            let (beta, _, _) =
                bounded_latent_to_user(latent_beta[term.col_idx], term.min, term.max);
            offset += &(self.design.column(term.col_idx).to_owned() * beta);
        }
        offset
    }

    fn effective_design_for_latent(&self, jac_diag: &Array1<f64>) -> Array2<f64> {
        let mut x_eff = self.design.clone();
        for term in &self.bounded_terms {
            let scaled = self.design.column(term.col_idx).to_owned() * jac_diag[term.col_idx];
            x_eff.column_mut(term.col_idx).assign(&scaled);
        }
        x_eff
    }

    fn exacthessian_andgradient(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<
        (
            StandardFamilyObservationState,
            Array2<f64>,
            Array1<f64>,
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
        ),
        String,
    > {
        let (_, jac_diag, second_diag, third_diag, priorthird) =
            self.bounded_term_derivative_data(latent_beta);
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let eta =
            self.designzeroed.dot(latent_beta) + self.nonlinear_offset_from_latent(latent_beta);
        let obs = evaluate_standard_familyobservations(
            self.family,
            self.mixture_link_state.as_ref(),
            self.sas_link_state.as_ref(),
            &self.y,
            &self.weights,
            &eta,
        )
        .map_err(|e| e.to_string())?;

        let mut priorgrad = Array1::<f64>::zeros(latent_beta.len());
        let mut prior_neghess = Array2::<f64>::zeros((latent_beta.len(), latent_beta.len()));
        let mut prior_loglik = 0.0;
        for term in &self.bounded_terms {
            let (logp, grad, neghess, _) =
                bounded_prior_terms(latent_beta[term.col_idx], &term.prior);
            prior_loglik += logp;
            priorgrad[term.col_idx] += grad;
            prior_neghess[[term.col_idx, term.col_idx]] += neghess;
        }

        let mut hessian = xt_diag_x_dense(x_eff.view(), obs.neghessian_eta.view())?;
        let mut gradient = fast_atv(&x_eff, &obs.score);
        for term in &self.bounded_terms {
            let score_beta = self.design.column(term.col_idx).dot(&obs.score);
            hessian[[term.col_idx, term.col_idx]] -= score_beta * second_diag[term.col_idx];
        }
        hessian += &prior_neghess;
        gradient += &priorgrad;

        Ok((
            obs,
            hessian,
            gradient,
            prior_loglik,
            second_diag,
            third_diag,
            priorthird,
        ))
    }

    fn evaluation_from_latent(
        &self,
        latent_beta: &Array1<f64>,
    ) -> Result<
        (
            StandardFamilyObservationState,
            Array2<f64>,
            Array1<f64>,
            f64,
        ),
        String,
    > {
        let (obs, hessian, gradient, prior_loglik, _, _, _) =
            self.exacthessian_andgradient(latent_beta)?;
        Ok((obs, hessian, gradient, prior_loglik))
    }
}

impl CustomFamily for BoundedLinearFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        let (obs, hessian, gradient, prior_loglik) = self.evaluation_from_latent(latent_beta)?;
        Ok(FamilyEvaluation {
            log_likelihood: obs.log_likelihood + prior_loglik,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        let (_, hessian, _, _) = self.evaluation_from_latent(latent_beta)?;
        Ok(Some(hessian))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        expect_block_idx_zero(block_idx, "bounded linear family", "")?;
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let latent_beta = &expect_single_block_state(block_states, "bounded linear family")?.beta;
        if d_beta_flat.len() != latent_beta.len() {
            return Err(format!(
                "bounded linear family directional derivative length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                latent_beta.len()
            ));
        }

        let (obs, _, _, _, second_diag, third_diag, priorthird) =
            self.exacthessian_andgradient(latent_beta)?;

        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(latent_beta);
        let x_eff = self.effective_design_for_latent(&jac_diag);
        let deta = x_eff.dot(d_beta_flat);
        let d_neghess_eta = &obs.neghessian_eta_derivative * &deta;

        let mut dx_eff = Array2::<f64>::zeros(x_eff.raw_dim());
        for term in &self.bounded_terms {
            let scale = second_diag[term.col_idx] * d_beta_flat[term.col_idx];
            if scale != 0.0 {
                let col = self.design.column(term.col_idx).to_owned() * scale;
                dx_eff.column_mut(term.col_idx).assign(&col);
            }
        }

        let mut dhessian = xt_diag_x_dense(x_eff.view(), d_neghess_eta.view())?;
        let mut wxdx = Array2::<f64>::zeros((x_eff.ncols(), x_eff.ncols()));
        for i in 0..x_eff.nrows() {
            let wi = obs.neghessian_eta[i];
            if wi == 0.0 {
                continue;
            }
            for a in 0..x_eff.ncols() {
                let xa = x_eff[[i, a]];
                for b in 0..x_eff.ncols() {
                    wxdx[[a, b]] += wi * (dx_eff[[i, a]] * x_eff[[i, b]] + xa * dx_eff[[i, b]]);
                }
            }
        }
        dhessian += &wxdx;

        let d_score = -&obs.neghessian_eta * &deta;
        for term in &self.bounded_terms {
            let score_beta = self.design.column(term.col_idx).dot(&obs.score);
            let d_score_beta = self.design.column(term.col_idx).dot(&d_score);
            dhessian[[term.col_idx, term.col_idx]] -= d_score_beta * second_diag[term.col_idx]
                + score_beta * third_diag[term.col_idx] * d_beta_flat[term.col_idx];
            dhessian[[term.col_idx, term.col_idx]] +=
                priorthird[term.col_idx] * d_beta_flat[term.col_idx];
        }

        Ok(Some(dhessian))
    }

    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if block_states.is_empty() {
            return Ok((
                DesignMatrix::Dense(Arc::new(self.designzeroed.clone())),
                self.offset.clone(),
            ));
        }
        let offset = self.nonlinear_offset_from_latent(
            &expect_single_block_state(block_states, "bounded linear family")?.beta,
        );
        let x = if spec.design.ncols() == self.designzeroed.ncols() {
            self.designzeroed.clone()
        } else {
            return Err("bounded linear family design column mismatch".to_string());
        };
        Ok((DesignMatrix::Dense(Arc::new(x)), offset))
    }

    fn block_geometry_is_dynamic(&self) -> bool {
        true
    }

    fn block_geometry_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
        d_beta: &Array1<f64>,
    ) -> Result<Option<BlockGeometryDirectionalDerivative>, String> {
        expect_block_idx_zero(
            block_idx,
            "bounded linear family",
            " for geometry derivative",
        )?;
        expect_single_block_state(block_states, "bounded linear family")?;
        if d_beta.len() != spec.design.ncols() {
            return Err(format!(
                "bounded linear family geometry derivative direction mismatch: got {}, expected {}",
                d_beta.len(),
                spec.design.ncols()
            ));
        }
        let (_, jac_diag, _, _, _) = self.bounded_term_derivative_data(&block_states[0].beta);
        let mut d_offset = Array1::<f64>::zeros(self.offset.len());
        let has_drift = self
            .bounded_terms
            .iter()
            .any(|term| jac_diag[term.col_idx] != 0.0 && d_beta[term.col_idx] != 0.0);
        if !has_drift {
            return Ok(Some(BlockGeometryDirectionalDerivative {
                d_design: None,
                d_offset,
            }));
        }
        for term in &self.bounded_terms {
            let col = term.col_idx;
            let drift = jac_diag[col] * d_beta[col];
            if drift != 0.0 {
                d_offset += &(self.design.column(col).to_owned() * drift);
            }
        }
        Ok(Some(BlockGeometryDirectionalDerivative {
            d_design: None,
            d_offset,
        }))
    }
}

fn xt_diag_x_dense(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
    if x.nrows() != w.len() {
        return Err("xt_diag_x_dense row mismatch".to_string());
    }
    let n = x.nrows();
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let wi = w[i];
        if wi == 0.0 {
            continue;
        }
        for a in 0..p {
            let xa = x[[i, a]];
            for b in a..p {
                let v = wi * xa * x[[i, b]];
                out[[a, b]] += v;
                if a != b {
                    out[[b, a]] += v;
                }
            }
        }
    }
    Ok(out)
}

fn trace_of_dense_product(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64, String> {
    if a.nrows() != a.ncols() || b.nrows() != b.ncols() || a.nrows() != b.nrows() {
        return Err("trace_of_dense_product dimension mismatch".to_string());
    }
    let mut trace = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            trace += a[[i, j]] * b[[j, i]];
        }
    }
    Ok(trace)
}

fn exact_bounded_edf(
    penalties: &[Array2<f64>],
    lambdas: &Array1<f64>,
    latent_cov: &Array2<f64>,
) -> Result<(Vec<f64>, f64), EstimationError> {
    if penalties.len() != lambdas.len() {
        return Err(EstimationError::InvalidInput(format!(
            "bounded EDF penalty/lambda mismatch: {} penalties vs {} lambdas",
            penalties.len(),
            lambdas.len()
        )));
    }
    if latent_cov.nrows() != latent_cov.ncols() {
        return Err(EstimationError::InvalidInput(
            "bounded EDF covariance must be square".to_string(),
        ));
    }

    let p = latent_cov.nrows();
    let mut s_lambda = Array2::<f64>::zeros((p, p));
    let mut edf_by_block = Vec::with_capacity(penalties.len());
    let mut trace_sum = 0.0;

    for (k, penalty) in penalties.iter().enumerate() {
        if penalty.nrows() != p || penalty.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "bounded EDF penalty {k} has shape {}x{}, expected {}x{}",
                penalty.nrows(),
                penalty.ncols(),
                p,
                p
            )));
        }
        let lambda_k = lambdas[k];
        s_lambda.scaled_add(lambda_k, penalty);
        let penalty_rank =
            penalty
                .nrows()
                .saturating_sub(estimate_penalty_nullity(penalty).map_err(|e| {
                    EstimationError::InvalidInput(format!("bounded EDF rank failed: {e}"))
                })?);
        let trace_k = lambda_k
            * trace_of_dense_product(latent_cov, penalty).map_err(EstimationError::InvalidInput)?;
        trace_sum += trace_k;
        let p_k = penalty_rank as f64;
        edf_by_block.push((p_k - trace_k).clamp(0.0, p_k));
    }

    let nullity_total = estimate_penalty_nullity(&s_lambda)
        .map_err(|e| EstimationError::InvalidInput(format!("bounded EDF nullity failed: {e}")))?
        as f64;
    let edf_total = (p as f64 - trace_sum).clamp(nullity_total, p as f64);
    Ok((edf_by_block, edf_total))
}

fn fit_bounded_term_collection_with_design(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    heuristic_lambdas: Option<&[f64]>,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let conditioning_cols: Vec<usize> = spec
        .linear_terms
        .iter()
        .enumerate()
        .filter_map(|(j, linear)| {
            (!linear.double_penalty).then_some(design.intercept_range.end + j)
        })
        .collect();
    let conditioning = LinearFitConditioning::from_columns(&design, &conditioning_cols);
    let fit_design = conditioning.apply_to_design(&design.design);
    let global_penalties = design.global_penalties();
    let fit_penalties = conditioning.transform_penalties_to_internal(&global_penalties);
    if design.linear_constraints.is_some() {
        return Err(EstimationError::InvalidInput(
            "bounded() terms are not yet compatible with explicit linear constraints".to_string(),
        ));
    }
    let mut bounded_terms = Vec::<BoundedLinearTermMeta>::new();
    for (j, term) in spec.linear_terms.iter().enumerate() {
        if term.double_penalty
            && matches!(
                term.coefficient_geometry,
                LinearCoefficientGeometry::Bounded { .. }
            )
        {
            return Err(EstimationError::InvalidInput(format!(
                "bounded linear term '{}' cannot also use double_penalty",
                term.name
            )));
        }
        if let LinearCoefficientGeometry::Bounded { min, max, prior } =
            term.coefficient_geometry.clone()
        {
            let col_idx = design.intercept_range.end + j;
            let (min_internal, max_internal) = conditioning.internal_bounds_for(col_idx, min, max);
            bounded_terms.push(BoundedLinearTermMeta {
                col_idx,
                min: min_internal,
                max: max_internal,
                prior,
            });
        }
    }
    if bounded_terms.is_empty() {
        return Err(EstimationError::InvalidInput(
            "internal bounded fit path called with no bounded terms".to_string(),
        ));
    }

    let mut designzeroed = fit_design.clone();
    let mut initial_beta = Array1::<f64>::zeros(fit_design.ncols());
    for term in &bounded_terms {
        designzeroed.column_mut(term.col_idx).fill(0.0);
        initial_beta[term.col_idx] = bounded_logit(0.5);
    }

    let initial_log_lambdas = heuristic_lambdas
        .map(|vals| Array1::from_vec(vals.to_vec()))
        .unwrap_or_else(|| Array1::zeros(fit_penalties.len()));
    if initial_log_lambdas.len() != fit_penalties.len() {
        return Err(EstimationError::InvalidInput(format!(
            "heuristic lambda length mismatch for bounded model: got {}, expected {}",
            initial_log_lambdas.len(),
            fit_penalties.len()
        )));
    }

    let family_adapter = BoundedLinearFamily {
        family,
        mixture_link_state: options
            .mixture_link
            .clone()
            .as_ref()
            .map(state_fromspec)
            .transpose()
            .map_err(EstimationError::InvalidInput)?,
        sas_link_state: options
            .sas_link
            .map(|spec| {
                if matches!(family, LikelihoodFamily::BinomialBetaLogistic) {
                    state_from_beta_logisticspec(spec)
                } else {
                    state_from_sasspec(spec)
                }
            })
            .transpose()
            .map_err(EstimationError::InvalidInput)?,
        y: y.to_owned(),
        weights: weights.to_owned(),
        design: fit_design.clone(),
        designzeroed: designzeroed.clone(),
        offset: offset.to_owned(),
        bounded_terms: bounded_terms.clone(),
    };
    let blockspec = ParameterBlockSpec {
        name: "eta".to_string(),
        design: DesignMatrix::Dense(Arc::new(designzeroed)),
        offset: offset.to_owned(),
        penalties: fit_penalties.iter().cloned().map(PenaltyMatrix::Dense).collect(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas,
        initial_beta: Some(initial_beta),
    };
    let fit = fit_custom_family(
        &family_adapter,
        &[blockspec],
        &BlockwiseFitOptions {
            inner_max_cycles: options.max_iter,
            inner_tol: options.tol,
            outer_max_iter: options.max_iter,
            outer_tol: options.tol,
            ..BlockwiseFitOptions::default()
        },
    )
    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

    let latent_beta = fit.block_states[0].beta.clone();
    let (beta_user_internal, jac_diag) = family_adapter.user_beta_and_jacobian(&latent_beta);
    let beta_user = conditioning.backtransform_beta(&beta_user_internal);
    let latent_cov = fit.covariance_conditional.clone();
    let beta_covariance = latent_cov.as_ref().map(|cov| {
        let mut out = cov.clone();
        let jac_col = jac_diag.view().insert_axis(ndarray::Axis(1));
        let jacrow = jac_diag.view().insert_axis(ndarray::Axis(0));
        out *= &(jac_col.to_owned() * jacrow);
        conditioning.backtransform_covariance(&out)
    });
    let beta_standard_errors = beta_covariance
        .as_ref()
        .map(|cov| Array1::from_iter((0..cov.nrows()).map(|i| cov[[i, i]].max(0.0).sqrt())));

    let (eta_state, h_data, _, _) = family_adapter
        .evaluation_from_latent(&latent_beta)
        .map_err(EstimationError::InvalidInput)?;
    let mut s_lambda_internal = Array2::<f64>::zeros((fit_design.ncols(), fit_design.ncols()));
    for (k, penalty) in fit_penalties.iter().enumerate() {
        s_lambda_internal.scaled_add(fit.lambdas[k], penalty);
    }
    let mut penalized_hessian = h_data.clone();
    penalized_hessian += &s_lambda_internal;
    let penalized_hessian =
        conditioning.transform_penalized_hessian_to_original(&penalized_hessian);
    let s_lambda_original = weighted_blockwise_penalty_sum(
        &design.penalties,
        fit.lambdas.as_slice().unwrap(),
        design.design.ncols(),
    );
    let penalty_term = beta_user.dot(&s_lambda_original.dot(&beta_user));
    let deviance = match family {
        LikelihoodFamily::GaussianIdentity => y
            .iter()
            .zip(eta_state.mu.iter())
            .zip(weights.iter())
            .map(|((&yy, &mu), &w)| w.max(0.0) * (yy - mu) * (yy - mu))
            .sum(),
        _ => -2.0 * eta_state.log_likelihood,
    };
    let (edf_by_block, edf_total) = if let Some(cov) = latent_cov.as_ref() {
        exact_bounded_edf(&fit_penalties, &fit.lambdas, cov)?
    } else {
        (vec![0.0; fit_penalties.len()], 0.0)
    };
    let geometry = Some(crate::estimate::FitGeometry {
        penalized_hessian: penalized_hessian.clone(),
        working_weights: eta_state.fisherweight.clone(),
        working_response: {
            let mut working_response = eta_state.eta.clone();
            for i in 0..working_response.len() {
                let wi = eta_state.fisherweight[i].max(1e-12);
                working_response[i] += eta_state.score[i] / wi;
            }
            working_response
        },
    });
    let max_abs_eta = eta_state
        .eta
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    Ok(FittedTermCollection {
        fit: {
            let log_lambdas = fit.lambdas.mapv(|v| v.max(1e-300).ln());
            let inf = FitInference {
                edf_by_block,
                edf_total,
                smoothing_correction: None,
                penalized_hessian: penalized_hessian.clone(),
                working_weights: eta_state.fisherweight.clone(),
                working_response: {
                    let mut working_response = eta_state.eta.clone();
                    for i in 0..working_response.len() {
                        let wi = eta_state.fisherweight[i].max(1e-12);
                        working_response[i] += eta_state.score[i] / wi;
                    }
                    working_response
                },
                reparam_qs: None,
                beta_covariance,
                beta_standard_errors,
                beta_covariance_corrected: None,
                beta_standard_errors_corrected: None,
            };
            let covariance_conditional = inf.beta_covariance.clone();
            let pirls_status_val = if fit.outer_converged {
                crate::pirls::PirlsStatus::Converged
            } else {
                crate::pirls::PirlsStatus::StalledAtValidMinimum
            };
            UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
                blocks: vec![crate::estimate::FittedBlock {
                    beta: beta_user.clone(),
                    role: crate::estimate::BlockRole::Mean,
                    edf: edf_total,
                    lambdas: fit.lambdas.clone(),
                }],
                log_lambdas,
                lambdas: fit.lambdas,
                log_likelihood: -0.5 * deviance,
                reml_score: fit.penalized_objective,
                stable_penalty_term: penalty_term,
                penalized_objective: fit.penalized_objective,
                outer_iterations: fit.outer_iterations,
                outer_converged: fit.outer_converged,
                outer_gradient_norm: fit.outer_gradient_norm,
                standard_deviation: 1.0,
                covariance_conditional,
                covariance_corrected: None,
                inference: Some(inf),
                fitted_link: crate::estimate::FittedLinkState::Standard(None),
                geometry,
                block_states: Vec::new(),
                pirls_status: pirls_status_val,
                max_abs_eta,
                constraint_kkt: None,
                artifacts: crate::estimate::FitArtifacts { pirls: None },
                inner_cycles: 0,
            })?
        },
        design: design.clone(),
        adaptive_diagnostics: None,
    })
}

fn enforce_term_constraint_feasibility(
    design: &TermCollectionDesign,
    fit: &UnifiedFitResult,
) -> Result<(), EstimationError> {
    let tol = 1e-7;
    let smooth_start = design
        .design
        .ncols()
        .saturating_sub(design.smooth.total_smooth_cols());
    let mut violations: Vec<String> = Vec::new();
    for term in &design.smooth.terms {
        let gr = (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);
        let beta_local = fit.beta.slice(s![gr.clone()]).to_owned();
        if let Some(lb) = term.lower_bounds_local.as_ref() {
            let mut worst = 0.0_f64;
            let mut worst_idx = 0usize;
            for i in 0..lb.len().min(beta_local.len()) {
                if lb[i].is_finite() {
                    let viol = (lb[i] - beta_local[i]).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worst_idx = i;
                    }
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=lower-bound maxviolation={:.3e} coeff_index={}",
                    term.name, worst, worst_idx
                ));
            }
        }
        if let Some(lin) = term.linear_constraints_local.as_ref() {
            let slack = lin.a.dot(&beta_local) - &lin.b;
            let mut worst = 0.0_f64;
            let mut worstrow = 0usize;
            for (i, &v) in slack.iter().enumerate() {
                let viol = (-v).max(0.0);
                if viol > worst {
                    worst = viol;
                    worstrow = i;
                }
            }
            if worst > tol {
                violations.push(format!(
                    "term='{}' kind=linear-inequality maxviolation={:.3e} row={}",
                    term.name, worst, worstrow
                ));
            }
        }
    }

    if !violations.is_empty() {
        let mut msg = format!(
            "constraint violation after fit ({} violating term constraints): {}",
            violations.len(),
            violations.join(" | ")
        );
        if let Some(kkt) = fit.constraint_kkt.as_ref() {
            msg.push_str(&format!(
                "; KKT[primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}]",
                kkt.primal_feasibility, kkt.dual_feasibility, kkt.complementarity, kkt.stationarity
            ));
        }
        return Err(EstimationError::ParameterConstraintViolation(msg));
    }
    Ok(())
}

pub(crate) fn spatial_length_scale_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    spec.smooth_terms
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| get_spatial_length_scale(spec, idx).map(|_| idx))
        .collect()
}

fn fit_score(fit: &UnifiedFitResult) -> f64 {
    if fit.reml_score.is_finite() {
        return fit.reml_score;
    }
    let score = 0.5 * fit.deviance + 0.5 * fit.stable_penalty_term;
    if score.is_finite() {
        score
    } else {
        f64::INFINITY
    }
}

fn external_opts_for_design(
    family: LikelihoodFamily,
    design: &TermCollectionDesign,
    options: &FitOptions,
) -> ExternalOptimOptions {
    ExternalOptimOptions {
        family,
        mixture_link: options.mixture_link.clone(),
        optimize_mixture: options.optimize_mixture,
        sas_link: options.sas_link,
        optimize_sas: options.optimize_sas,
        compute_inference: options.compute_inference,
        max_iter: options.max_iter,
        tol: options.tol,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: None,
        penalty_shrinkage_floor: options.penalty_shrinkage_floor,
    }
}

/// Evaluate the joint REML cost, gradient, and Hessian at a given θ = [ρ, ψ]
/// for a single-block term collection with spatial hyperparameters.
///
/// This provides a direct evaluation of the profiled REML objective using the
/// external-caller interface, which exposes exact cost/gradient/Hessian without
/// running the full outer smoothing loop. The returned tuple is
/// `(cost, gradient, hessian)` in the joint [ρ, ψ] space.
fn evaluate_joint_reml_at_theta(
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    design: &TermCollectionDesign,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: Vec<crate::estimate::reml::DirectionalHyperParam>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
    let ext_opts = external_opts_for_design(family, design, options);
    crate::estimate::compute_external_joint_hypercostgradienthessian(
        y,
        weights,
        design.design.clone(),
        offset,
        design.global_penalties(),
        theta,
        rho_dim,
        hyper_dirs,
        warm_start_beta,
        &ext_opts,
    )
}

fn smooth_term_penalty_index(
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
) -> Option<usize> {
    if term_idx >= design.smooth.terms.len() || term_idx >= spec.smooth_terms.len() {
        return None;
    }
    if design.smooth.terms[term_idx].penalties_local.is_empty() {
        return None;
    }
    let linear_penalties = usize::from(spec.linear_terms.iter().any(|t| t.double_penalty));
    let random_penalties = design
        .random_effect_ranges
        .iter()
        .filter(|(_, range)| !range.is_empty())
        .count();
    let smooth_offset = linear_penalties + random_penalties;
    let local_offset = design
        .smooth
        .terms
        .iter()
        .take(term_idx)
        .map(|term| term.penalties_local.len())
        .sum::<usize>();
    Some(smooth_offset + local_offset)
}

fn try_build_spatial_term_log_kappa_derivativeinfo(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
) -> Result<Option<SpatialPsiDerivative>, EstimationError> {
    let Some((
        global_range,
        total_p,
        x_psi_local,
        s_psi_local_check,
        x_psi_psi_local,
        s_psi_psi_local,
        s_psi_components_local,
        s_psi_psi_components_local,
    )) = try_build_spatial_term_log_kappa_derivative(data, resolvedspec, design, term_idx)?
    else {
        return Ok(None);
    };
    let Some(penalty_start) = smooth_term_penalty_index(resolvedspec, design, term_idx) else {
        return Ok(None);
    };
    if s_psi_components_local.is_empty() || s_psi_psi_components_local.is_empty() {
        return Ok(None);
    }
    if s_psi_components_local.len() != s_psi_psi_components_local.len() {
        return Ok(None);
    }
    let penalty_indices = (0..s_psi_components_local.len())
        .map(|j| penalty_start + j)
        .collect::<Vec<_>>();
    let penalty_index = penalty_indices[0];
    if s_psi_local_check.nrows() == 0 || s_psi_psi_local.nrows() == 0 {
        return Ok(None);
    }
    Ok(Some(SpatialPsiDerivative {
        penalty_index,
        penalty_indices,
        global_range,
        total_p,
        x_psi_local,
        s_psi_components_local,
        x_psi_psi_local,
        s_psi_psi_components_local,
        aniso_group_id: None,
        aniso_cross_designs: None,
        aniso_cross_penalties: None,
        implicit_operator: None,
        implicit_axis: 0,
    }))
}

pub(crate) fn try_build_spatial_log_kappa_derivativeinfo_list(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    spatial_terms: &[usize],
) -> Result<Option<Vec<SpatialPsiDerivative>>, EstimationError> {
    let mut out = Vec::new();
    let mut aniso_gid = 0usize;
    for &term_idx in spatial_terms {
        let aniso = get_spatial_aniso_log_scales(resolvedspec, term_idx);
        let dim = get_spatial_feature_dim(resolvedspec, term_idx);
        if let (Some(eta), Some(d)) = (&aniso, dim) {
            if eta.len() == d && d > 1 {
                if let Some(entries) = try_build_spatial_term_log_kappa_aniso_derivativeinfos(
                    data,
                    resolvedspec,
                    design,
                    term_idx,
                    aniso_gid,
                )? {
                    aniso_gid += 1;
                    out.extend(entries);
                    continue;
                } else {
                    return Ok(None);
                }
            }
        }
        let Some(info) =
            try_build_spatial_term_log_kappa_derivativeinfo(data, resolvedspec, design, term_idx)?
        else {
            return Ok(None);
        };
        out.push(info);
    }
    Ok(Some(out))
}

/// For an aniso term with d axes, produce d `SpatialPsiDerivative` entries.
fn try_build_spatial_term_log_kappa_aniso_derivativeinfos(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
    aniso_group_id: usize,
) -> Result<Option<Vec<SpatialPsiDerivative>>, EstimationError> {
    let smooth_term = match design.smooth.terms.get(term_idx) {
        Some(t) => t,
        None => return Ok(None),
    };
    let termspec = match resolvedspec.smooth_terms.get(term_idx) {
        Some(t) => t,
        None => return Ok(None),
    };
    let aniso_result = match &termspec.basis {
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_matern_basis_log_kappa_aniso_derivatives(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_duchon_basis_log_kappa_aniso_derivatives(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        _ => return Ok(None),
    };
    // Get number of axes from the shared operator when available; otherwise
    // fall back to the dense design list.
    let d = if let Some(ref op) = aniso_result.implicit_operator {
        op.n_axes()
    } else if !aniso_result.design_first.is_empty() {
        aniso_result.design_first.len()
    } else {
        0
    };
    if d == 0 {
        return Ok(None);
    }
    let Some(penalty_start) = smooth_term_penalty_index(resolvedspec, design, term_idx) else {
        return Ok(None);
    };
    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let global_range = (smooth_start + smooth_term.coeff_range.start)
        ..(smooth_start + smooth_term.coeff_range.end);
    let num_penalties = aniso_result.penalties_first[0].len();
    let penalty_indices: Vec<usize> = (0..num_penalties).map(|j| penalty_start + j).collect();

    // Dense first/diagonal-second matrices may be present even when the shared
    // operator is available. The operator remains the canonical source for
    // exact cross-axis second derivatives.
    let use_implicit_design = aniso_result.design_first.is_empty();
    let implicit_op_arc = aniso_result
        .implicit_operator
        .as_ref()
        .map(|op| std::sync::Arc::new(op.clone()));

    let mut entries = Vec::with_capacity(d);
    for a in 0..d {
        let (x_psi_local, x_psi_psi_local) = if use_implicit_design {
            // Implicit path: design-derivative matvecs will be dispatched through
            // the ImplicitDerivativeOp inside HyperDesignDerivative, so we do NOT
            // need to materialize the dense (n x p) matrices here.  Store empty
            // placeholders — they are never read when the implicit operator is
            // present (spatial_log_kappa_hyper_dirs_frominfo_list uses from_implicit).
            (Array2::<f64>::zeros((0, 0)), Array2::<f64>::zeros((0, 0)))
        } else {
            let x_first = aniso_result.design_first[a].clone();
            let x_second = aniso_result.design_second_diag[a].clone();
            if x_first.ncols() != smooth_term.coeff_range.len() {
                return Ok(None);
            }
            (x_first, x_second)
        };
        let s_psi_components = aniso_result.penalties_first[a].clone();
        let s_psi_psi_components = aniso_result.penalties_second_diag[a].clone();
        // Build cross-design entries for other axes b != a in this group.
        // These will be indexed by (b, cross_matrix) where b is the axis
        // offset within the d-entry block.
        // Cross-axis second derivatives are sourced from the shared operator,
        // so we only need placeholder entries to preserve the axis layout.
        let cross_designs = if implicit_op_arc.is_some() {
            let mut cd = Vec::with_capacity(d - 1);
            for b in 0..d {
                if b == a {
                    continue;
                }
                cd.push((b, Array2::<f64>::zeros((0, 0))));
            }
            cd
        } else {
            Vec::new()
        };
        // Build cross-penalty entries for this axis a.
        // For each pair (a, b) with a < b in penalties_cross_pairs, store it
        // under axis a with key b. For pairs (c, a) with c < a, store it
        // under axis a with key c (the penalty is symmetric).
        let mut cross_penalties = Vec::new();
        for (cp_idx, &(pa, pb)) in aniso_result.penalties_cross_pairs.iter().enumerate() {
            if pa == a {
                cross_penalties.push((pb, aniso_result.penalties_cross[cp_idx].clone()));
            } else if pb == a {
                // Symmetric: ∂²S/∂ψ_b∂ψ_a = ∂²S/∂ψ_a∂ψ_b (penalty matrices are symmetric)
                cross_penalties.push((pa, aniso_result.penalties_cross[cp_idx].clone()));
            }
        }
        let cross_penalties_opt = if cross_penalties.is_empty() {
            None
        } else {
            Some(cross_penalties)
        };

        entries.push(SpatialPsiDerivative {
            penalty_index: penalty_indices[0],
            penalty_indices: penalty_indices.clone(),
            global_range: global_range.clone(),
            total_p: p_total,
            x_psi_local,
            s_psi_components_local: s_psi_components,
            x_psi_psi_local,
            s_psi_psi_components_local: s_psi_psi_components,
            aniso_group_id: Some(aniso_group_id),
            aniso_cross_designs: if cross_designs.is_empty() {
                None
            } else {
                Some(cross_designs)
            },
            aniso_cross_penalties: cross_penalties_opt,
            implicit_operator: implicit_op_arc.clone(),
            implicit_axis: a,
        });
    }
    Ok(Some(entries))
}

fn try_build_spatial_term_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
) -> Result<
    Option<(
        Range<usize>,
        usize,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
    )>,
    EstimationError,
> {
    let smooth_term = match design.smooth.terms.get(term_idx) {
        Some(term) => term,
        None => return Ok(None),
    };
    let termspec = match resolvedspec.smooth_terms.get(term_idx) {
        Some(term) => term,
        None => return Ok(None),
    };

    let BasisPsiDerivativeResult {
        design_derivative: local_x_psi,
        penalties_derivative: local_s_psi,
    } = match &termspec.basis {
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_thin_plate_basis_log_kappa_derivative(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_matern_basis_log_kappa_derivative(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_duchon_basis_log_kappa_derivative(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::BSpline1D { .. } | SmoothBasisSpec::TensorBSpline { .. } => {
            return Ok(None);
        }
    };
    let BasisPsiSecondDerivativeResult {
        designsecond_derivative: local_x_psi_psi,
        penaltiessecond_derivative: local_s_psi_psi,
    } = match &termspec.basis {
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_thin_plate_basis_log_kappasecond_derivative(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_matern_basis_log_kappasecond_derivative(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
            }
            build_duchon_basis_log_kappasecond_derivative(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::BSpline1D { .. } | SmoothBasisSpec::TensorBSpline { .. } => {
            return Ok(None);
        }
    };

    if local_x_psi.ncols() != smooth_term.coeff_range.len() {
        return Ok(None);
    }
    if local_x_psi_psi.ncols() != smooth_term.coeff_range.len() {
        return Ok(None);
    }
    if local_s_psi.is_empty() || local_s_psi.len() != local_s_psi_psi.len() {
        return Ok(None);
    }
    if local_s_psi.iter().any(|s| {
        s.nrows() != smooth_term.coeff_range.len() || s.ncols() != smooth_term.coeff_range.len()
    }) {
        return Ok(None);
    }
    if local_s_psi_psi.iter().any(|s| {
        s.nrows() != smooth_term.coeff_range.len() || s.ncols() != smooth_term.coeff_range.len()
    }) {
        return Ok(None);
    }

    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let global_range = (smooth_start + smooth_term.coeff_range.start)
        ..(smooth_start + smooth_term.coeff_range.end);

    Ok(Some((
        global_range,
        p_total,
        local_x_psi,
        local_s_psi.iter().fold(
            Array2::<f64>::zeros((smooth_term.coeff_range.len(), smooth_term.coeff_range.len())),
            |acc, m| acc + m,
        ),
        local_x_psi_psi,
        local_s_psi_psi.iter().fold(
            Array2::<f64>::zeros((smooth_term.coeff_range.len(), smooth_term.coeff_range.len())),
            |acc, m| acc + m,
        ),
        local_s_psi,
        local_s_psi_psi,
    )))
}

fn try_build_spatial_log_kappa_hyper_dirs(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    spatial_terms: &[usize],
) -> Result<Option<Vec<DirectionalHyperParam>>, EstimationError> {
    // Each spatial term contributes one continuous scale hyperparameter
    //   psi = log(kappa) = -log(length_scale),
    // while rho = log(lambda) still indexes the smoothing parameters of the
    // three operator penalties. The joint outer vector is therefore
    //   theta = (rho_0, ..., rho_{K-1}, psi_1, ..., psi_q)
    // for q spatial terms participating in exact joint optimization.
    let Some(info_list) =
        try_build_spatial_log_kappa_derivativeinfo_list(data, resolvedspec, design, spatial_terms)?
    else {
        return Ok(None);
    };
    Ok(Some(spatial_log_kappa_hyper_dirs_frominfo_list(info_list)?))
}

fn spatial_log_kappa_hyper_dirs_frominfo_list(
    info_list: Vec<SpatialPsiDerivative>,
) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
    use crate::estimate::reml::ImplicitDerivLevel;

    let mut hyper_dirs = Vec::with_capacity(info_list.len());
    let log_kappa_dim = info_list.len();
    for (i, info) in info_list.iter().enumerate() {
        let mut xsecond = vec![None; log_kappa_dim];
        // Diagonal second derivative (same axis).
        xsecond[i] = Some(if let Some(ref op) = info.implicit_operator {
            crate::estimate::reml::HyperDesignDerivative::from_implicit(
                op.clone(),
                ImplicitDerivLevel::SecondDiag(info.implicit_axis),
            )
        } else {
            crate::estimate::reml::HyperDesignDerivative::from_embedded(
                info.x_psi_psi_local.clone(),
                info.global_range.clone(),
                info.total_p,
            )
        });
        // Cross second derivatives for axes in the same aniso group.
        if let Some(ref cross_designs) = info.aniso_cross_designs {
            // Find the base index of this aniso group in the info_list.
            // Entries for the same group are contiguous: the first entry
            // with matching group_id gives the base, and axis b is at base+b.
            if let Some(gid) = info.aniso_group_id {
                let base = info_list
                    .iter()
                    .position(|e| e.aniso_group_id == Some(gid))
                    .unwrap_or(i);
                for &(b_axis, ref cross_mat) in cross_designs {
                    let j = base + b_axis;
                    if j < log_kappa_dim {
                        xsecond[j] = Some(if let Some(ref op) = info.implicit_operator {
                            crate::estimate::reml::HyperDesignDerivative::from_implicit(
                                op.clone(),
                                ImplicitDerivLevel::SecondCross(info.implicit_axis, b_axis),
                            )
                        } else {
                            crate::estimate::reml::HyperDesignDerivative::from_embedded(
                                cross_mat.clone(),
                                info.global_range.clone(),
                                info.total_p,
                            )
                        });
                    }
                }
            }
        }
        let s_components = info
            .penalty_indices
            .iter()
            .copied()
            .zip(info.s_psi_components_local.iter().map(|local| {
                crate::estimate::reml::HyperPenaltyDerivative::from_embedded(
                    local.clone(),
                    info.global_range.clone(),
                    info.total_p,
                )
            }))
            .collect::<Vec<_>>();
        let s2_components = info
            .penalty_indices
            .iter()
            .copied()
            .zip(info.s_psi_psi_components_local.iter().map(|local| {
                crate::estimate::reml::HyperPenaltyDerivative::from_embedded(
                    local.clone(),
                    info.global_range.clone(),
                    info.total_p,
                )
            }))
            .collect::<Vec<_>>();
        let mut ssecond_components = vec![None; log_kappa_dim];
        ssecond_components[i] = Some(s2_components);
        // Cross penalty second derivatives (∂²S/∂ψ_a∂ψ_b, a≠b).
        if let Some(ref cross_penalties) = info.aniso_cross_penalties {
            if let Some(gid) = info.aniso_group_id {
                let base = info_list
                    .iter()
                    .position(|e| e.aniso_group_id == Some(gid))
                    .unwrap_or(i);
                for &(b_axis, ref cross_pens) in cross_penalties {
                    let j = base + b_axis;
                    if j < log_kappa_dim {
                        let cross_components = info
                            .penalty_indices
                            .iter()
                            .copied()
                            .zip(cross_pens.iter().map(|local| {
                                crate::estimate::reml::HyperPenaltyDerivative::from_embedded(
                                    local.clone(),
                                    info.global_range.clone(),
                                    info.total_p,
                                )
                            }))
                            .collect::<Vec<_>>();
                        ssecond_components[j] = Some(cross_components);
                    }
                }
            }
        }
        // First derivative: use implicit operator when available to avoid
        // storing dense (n x p) matrices for all D axes simultaneously.
        let x_first_hyper = if let Some(ref op) = info.implicit_operator {
            crate::estimate::reml::HyperDesignDerivative::from_implicit(
                op.clone(),
                ImplicitDerivLevel::First(info.implicit_axis),
            )
        } else {
            crate::estimate::reml::HyperDesignDerivative::from_embedded(
                info.x_psi_local.clone(),
                info.global_range.clone(),
                info.total_p,
            )
        };
        hyper_dirs.push(
            DirectionalHyperParam::new_compact(
                x_first_hyper,
                s_components,
                Some(xsecond),
                Some(ssecond_components),
            )?
            .not_penalty_like(),
        );
    }
    Ok(hyper_dirs)
}

/// Compute `dims_per_term` for a list of spatial term indices.
///
/// Returns a vector where entry i is the number of ψ values for spatial
/// term i: 1 for isotropic terms, d for d-dimensional anisotropic terms.
fn compute_spatial_dims_per_term(
    resolvedspec: &TermCollectionSpec,
    spatial_terms: &[usize],
) -> Vec<usize> {
    spatial_terms
        .iter()
        .map(|&term_idx| {
            let d = get_spatial_feature_dim(resolvedspec, term_idx).unwrap_or(1);
            let has_aniso = get_spatial_aniso_log_scales(resolvedspec, term_idx).is_some();
            if has_aniso && d > 1 { d } else { 1 }
        })
        .collect()
}

/// Check whether any spatial terms are anisotropic (dims_per_term > 1).
fn has_aniso_terms(dims_per_term: &[usize]) -> bool {
    dims_per_term.iter().any(|&d| d > 1)
}

#[derive(Debug, Clone)]
struct SingleBlockExactJointDesignCache<'d> {
    realizer: FrozenTermCollectionIncrementalRealizer<'d>,
    current_theta: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_eval: Option<(f64, Array1<f64>, Array2<f64>)>,
    spatial_terms: Vec<usize>,
    rho_dim: usize,
    dims_per_term: Vec<usize>,
}

impl<'d> SingleBlockExactJointDesignCache<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
        spatial_terms: Vec<usize>,
        rho_dim: usize,
        dims_per_term: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(Self {
            realizer: FrozenTermCollectionIncrementalRealizer::new(data, spec, design)?,
            current_theta: None,
            last_cost: None,
            last_eval: None,
            spatial_terms,
            rho_dim,
            dims_per_term,
        })
    }

    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), String> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }
        let log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
            theta,
            self.rho_dim,
            self.dims_per_term.clone(),
        );
        self.realizer
            .apply_log_kappa(&log_kappa, &self.spatial_terms)?;
        self.current_theta = Some(theta.clone());
        self.last_cost = None;
        self.last_eval = None;
        Ok(())
    }

    fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval
                .as_ref()
                .map(|cached| cached.0)
                .or(self.last_cost)
        } else {
            None
        }
    }

    fn memoized_eval(&self, theta: &Array1<f64>) -> Option<(f64, Array1<f64>, Array2<f64>)> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval.clone()
        } else {
            None
        }
    }

    fn store_eval(&mut self, eval: (f64, Array1<f64>, Array2<f64>)) {
        self.last_cost = Some(eval.0);
        self.last_eval = Some(eval);
    }

    fn spec(&self) -> &TermCollectionSpec {
        self.realizer.spec()
    }

    fn design(&self) -> &TermCollectionDesign {
        self.realizer.design()
    }
}

fn try_exact_joint_spatial_length_scale_optimization(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    best: &FittedTermCollection,
    family: LikelihoodFamily,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    spatial_terms: &[usize],
) -> Result<Option<FittedTermCollectionWithSpec>, EstimationError> {
    if spatial_terms.is_empty() {
        return Ok(None);
    }
    if try_build_spatial_log_kappa_hyper_dirs(data, resolvedspec, &best.design, spatial_terms)?
        .is_none()
    {
        return Ok(None);
    }

    const JOINT_RHO_BOUND: f64 = 12.0;
    let rho_dim = best.fit.lambdas.len();

    // Compute per-term dimensionality for anisotropic terms.
    let dims_per_term = compute_spatial_dims_per_term(resolvedspec, spatial_terms);
    let use_aniso = has_aniso_terms(&dims_per_term);

    // Build initial ψ values and bounds, using aniso-aware constructors
    // when any term has d > 1 axes.
    let log_kappa0 = if use_aniso {
        SpatialLogKappaCoords::from_length_scales_aniso(resolvedspec, spatial_terms, kappa_options)
    } else {
        SpatialLogKappaCoords::from_length_scales(resolvedspec, spatial_terms, kappa_options)
    };
    let log_kappa_lower = if use_aniso {
        SpatialLogKappaCoords::lower_bounds_aniso(&dims_per_term, kappa_options)
    } else {
        SpatialLogKappaCoords::lower_bounds(spatial_terms.len(), kappa_options)
    };
    let log_kappa_upper = if use_aniso {
        SpatialLogKappaCoords::upper_bounds_aniso(&dims_per_term, kappa_options)
    } else {
        SpatialLogKappaCoords::upper_bounds(spatial_terms.len(), kappa_options)
    };
    let setup = ExactJointHyperSetup::new(
        best.fit.lambdas.mapv(f64::ln),
        Array1::<f64>::from_elem(rho_dim, -JOINT_RHO_BOUND),
        Array1::<f64>::from_elem(rho_dim, JOINT_RHO_BOUND),
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );

    let theta0 = setup.theta0();
    let lower = setup.lower();
    let upper = setup.upper();

    // ───────────────────────────────────────────────────────────────────────
    //  Anisotropic analytic path: when any term has d > 1 axes, use the
    //  unified REML evaluator with ext_coords for joint [ρ, ψ] optimization.
    //
    //  The key advantage: analytic gradient + Hessian w.r.t. ψ_a via the
    //  AnisoBasisPsiDerivatives → DirectionalHyperParam → HyperCoord pipeline,
    //  giving Newton/BFGS quadratic convergence on the anisotropy parameters.
    // ───────────────────────────────────────────────────────────────────────
    let theta_star = if use_aniso {
        try_exact_joint_spatial_aniso_optimization(
            data,
            y,
            weights,
            offset,
            resolvedspec,
            &best.design,
            family,
            options,
            spatial_terms,
            &dims_per_term,
            &theta0,
            &lower,
            &upper,
            rho_dim,
        )?
    } else {
        // Isotropic analytic path: use the unified REML evaluator with
        // ext_coords for joint [ρ, κ] optimization via BFGS, analogous to
        // the anisotropic path but with a single κ per spatial term.
        // outer_strategy handles the centralized degradation path when the
        // analytic Hessian is unavailable.
        try_exact_joint_spatial_isotropic_optimization(
            data,
            y,
            weights,
            offset,
            resolvedspec,
            &best.design,
            family,
            options,
            spatial_terms,
            &dims_per_term,
            &theta0,
            &lower,
            &upper,
            rho_dim,
            kappa_options,
        )?
    };

    let rho_star = theta_star.slice(s![..rho_dim]).mapv(f64::exp);
    let log_kappa_star =
        SpatialLogKappaCoords::from_theta_tail_with_dims(&theta_star, rho_dim, dims_per_term);
    let resolvedspec = log_kappa_star.apply_tospec(resolvedspec, spatial_terms)?;
    let best = fit_term_collection_forspecwith_heuristic_lambdas(
        data,
        y,
        weights,
        offset,
        &resolvedspec,
        rho_star.as_slice(),
        family,
        options,
    )?;

    Ok(Some(FittedTermCollectionWithSpec {
        fit: best.fit,
        design: best.design,
        resolvedspec,
        adaptive_diagnostics: best.adaptive_diagnostics,
    }))
}

/// Joint [ρ, ψ] optimization for anisotropic spatial terms using analytic
/// derivatives through the unified REML evaluator.
///
/// At each outer iteration, the frozen term topology is reused and only the
/// spatial realized blocks affected by the current ψ are refreshed before the
/// unified evaluator returns cost + gradient + Hessian for the full
/// θ = [ρ, ψ] vector. The ψ derivatives flow through:
///
///   `AnisoBasisPsiDerivatives` → `SpatialPsiDerivative` → `DirectionalHyperParam`
///     → `build_tau_unified_objects` → `HyperCoord` ext_coords → unified evaluator
///
/// This is the analytic anisotropic path, giving Newton/BFGS quadratic convergence on the
/// anisotropy parameters while jointly optimizing the smoothing parameters.
///
/// The ψ_a are parameterized as unconstrained log-scale coordinates. The
/// decomposition into isotropic scale (ψ̄ = mean(ψ_a)) and anisotropy
/// (η_a = ψ_a − ψ̄, with Ση_a = 0) happens only on writeback via
/// `SpatialLogKappaCoords::apply_tospec`. The all-ones direction in ψ-space
/// is NOT a gauge direction — it controls the identifiable isotropic scale
/// κ = exp(ψ̄). No sum-to-zero constraint is enforced during optimization.
fn try_exact_joint_spatial_aniso_optimization(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    baseline_design: &TermCollectionDesign,
    family: LikelihoodFamily,
    options: &FitOptions,
    spatial_terms: &[usize],
    dims_per_term: &[usize],
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
) -> Result<Array1<f64>, EstimationError> {
    // Use bounds and design metadata for validation.
    assert!(lower.len() == theta0.len() && upper.len() == theta0.len());
    assert!(baseline_design.smooth.terms.len() >= spatial_terms.len());
    use crate::solver::outer_strategy::{
        ClosureObjective, Derivative, EfsEval, FallbackPolicy, HessianResult, OuterCapability,
        OuterConfig, OuterEval,
    };

    let theta_dim = theta0.len();
    let psi_dim = theta_dim - rho_dim;

    log::trace!(
        "[spatial-aniso-joint] starting analytic optimization: rho_dim={}, psi_dim={}, dims_per_term={:?}",
        rho_dim,
        psi_dim,
        dims_per_term,
    );

    // Shared context for the optimization closures. Holds immutable references
    // to data/spec/options and the mutable best-tracking state.
    struct AnisoJointContext<'d> {
        data: ArrayView2<'d, f64>,
        y: ArrayView1<'d, f64>,
        weights: ArrayView1<'d, f64>,
        offset: ArrayView1<'d, f64>,
        family: LikelihoodFamily,
        options: &'d FitOptions,
        rho_dim: usize,
        best_cost: f64,
        cache: SingleBlockExactJointDesignCache<'d>,
    }

    impl<'d> AnisoJointContext<'d> {
        /// Full evaluation on the current realized design + hyper_dirs.
        fn eval_full(
            &mut self,
            theta: &Array1<f64>,
        ) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
            if let Some(eval) = self.cache.memoized_eval(theta) {
                return Ok(eval);
            }
            self.cache
                .ensure_theta(theta)
                .map_err(EstimationError::InvalidInput)?;
            let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
                self.data,
                self.cache.spec(),
                self.cache.design(),
                &self.cache.spatial_terms,
            )?
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "failed to build aniso hyper_dirs at current psi".to_string(),
                )
            })?;

            let eval = evaluate_joint_reml_at_theta(
                self.y,
                self.weights,
                self.offset,
                self.cache.design(),
                theta,
                self.rho_dim,
                hyper_dirs,
                None,
                self.family,
                self.options,
            );
            if let Ok(ref value) = eval {
                self.cache.store_eval(value.clone());
            }
            eval
        }

        /// Cost-only evaluation — uses the same evaluator as eval_full to ensure
        /// the line search and gradient see the same objective function.
        fn eval_cost(&mut self, theta: &Array1<f64>) -> f64 {
            if let Some(cost) = self.cache.memoized_cost(theta) {
                return cost;
            }
            match self.eval_full(theta) {
                Ok((cost, _, _)) => cost,
                Err(_) => f64::INFINITY,
            }
        }

        fn track_best(&mut self, _: &Array1<f64>, cost: f64) {
            if cost < self.best_cost {
                self.best_cost = cost;
            }
        }
    }

    let mut ctx = AnisoJointContext {
        data,
        y,
        weights,
        offset,
        family,
        options,
        rho_dim,
        best_cost: f64::INFINITY,
        cache: SingleBlockExactJointDesignCache::new(
            data,
            resolvedspec.clone(),
            baseline_design.clone(),
            spatial_terms.to_vec(),
            rho_dim,
            dims_per_term.to_vec(),
        )
        .map_err(EstimationError::InvalidInput)?,
    };

    let outer_config = OuterConfig {
        tolerance: 1e-6,
        max_iter: 50,
        fd_step: 1e-5,
        bounds: Some((lower.clone(), upper.clone())),
        seed_config: crate::seeding::SeedConfig {
            max_seeds: 1,
            screening_budget: 1,
            num_auxiliary_trailing: psi_dim,
            ..Default::default()
        },
        rho_bound: 12.0,
        heuristic_lambdas: Some(theta0.as_slice().unwrap().to_vec()),
        initial_rho: Some(theta0.clone()),
        fallback_policy: FallbackPolicy::Automatic,
        screening_cap: None,
    };

    let mut obj = ClosureObjective {
        state: &mut ctx,
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: theta_dim,
            // ψ coordinates are NOT penalty-like (they move the design),
            // so EFS is not appropriate.
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: false,
            barrier_config: None,
        },
        cost_fn: |ctx: &mut &mut AnisoJointContext<'_>, theta: &Array1<f64>| {
            let cost = ctx.eval_cost(theta);
            ctx.track_best(theta, cost);
            Ok(cost)
        },
        eval_fn: |ctx: &mut &mut AnisoJointContext<'_>, theta: &Array1<f64>| match ctx
            .eval_full(theta)
        {
            Ok((cost, grad, hess)) => {
                ctx.track_best(theta, cost);
                // No ψ projection: the all-ones direction in raw ψ-space is the
                // real isotropic scale coordinate. Removing it would change the
                // model, not just fix a gauge.
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: HessianResult::Analytic(hess),
                })
            }
            Err(_) => Ok(OuterEval::infeasible(theta.len())),
        },
        reset_fn: None::<fn(&mut &mut AnisoJointContext<'_>)>,
        efs_fn: None::<
            fn(&mut &mut AnisoJointContext<'_>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    };

    let result =
        crate::solver::outer_strategy::run_outer(&mut obj, &outer_config, "aniso-psi joint REML")
            .map_err(|e| {
            EstimationError::InvalidInput(format!(
                "anisotropic analytic optimization failed after exhausting strategy fallbacks: {e}"
            ))
        })?;
    log::trace!(
        "[spatial-aniso-joint] converged in {} iterations, final_value={:.6e}, grad_norm={:.6e}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
    );
    // No sum-to-zero enforcement needed: ψ coordinates are unconstrained during
    // optimization. The decomposition into (ψ̄, η) happens in apply_tospec.
    let theta_star = result.rho;
    Ok(theta_star)
}

/// Joint [ρ, κ] optimization for isotropic spatial terms using analytic
/// derivatives through the unified REML evaluator.
///
/// This is the isotropic counterpart of the anisotropic exact spatial path.
/// Each spatial term contributes one log-κ coordinate, and the joint outer
/// optimization runs directly on `[ρ, κ]` while reusing the frozen term
/// topology and refreshing only the spatial realized blocks touched by κ.
/// The gradient and Hessian for log-κ flow through the same
/// directional-hyperparameter pipeline used elsewhere:
///
///   `SpatialPsiDerivative` → `DirectionalHyperParam` → `HyperCoord` ext_coords
///     → unified REML evaluator
/// This gives BFGS/Newton quadratic convergence on the length-scale parameters.
fn try_exact_joint_spatial_isotropic_optimization(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    baseline_design: &TermCollectionDesign,
    family: LikelihoodFamily,
    options: &FitOptions,
    spatial_terms: &[usize],
    dims_per_term: &[usize],
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
    _: &SpatialLengthScaleOptimizationOptions,
) -> Result<Array1<f64>, EstimationError> {
    assert!(lower.len() == theta0.len() && upper.len() == theta0.len());
    assert!(baseline_design.smooth.terms.len() >= spatial_terms.len());
    use crate::solver::outer_strategy::{
        ClosureObjective, Derivative, EfsEval, FallbackPolicy, HessianResult, OuterCapability,
        OuterConfig, OuterEval,
    };

    let theta_dim = theta0.len();
    let kappa_dim = theta_dim - rho_dim;

    log::trace!(
        "[spatial-iso-joint] starting analytic optimization: rho_dim={}, kappa_dim={}, dims_per_term={:?}",
        rho_dim,
        kappa_dim,
        dims_per_term,
    );

    struct IsoJointContext<'d> {
        data: ArrayView2<'d, f64>,
        y: ArrayView1<'d, f64>,
        weights: ArrayView1<'d, f64>,
        offset: ArrayView1<'d, f64>,
        family: LikelihoodFamily,
        options: &'d FitOptions,
        rho_dim: usize,
        best_cost: f64,
        cache: SingleBlockExactJointDesignCache<'d>,
    }

    impl<'d> IsoJointContext<'d> {
        /// Full evaluation on the current realized design + hyper_dirs.
        fn eval_full(
            &mut self,
            theta: &Array1<f64>,
        ) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
            if let Some(eval) = self.cache.memoized_eval(theta) {
                return Ok(eval);
            }
            self.cache
                .ensure_theta(theta)
                .map_err(EstimationError::InvalidInput)?;
            let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
                self.data,
                self.cache.spec(),
                self.cache.design(),
                &self.cache.spatial_terms,
            )?
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "failed to build isotropic hyper_dirs at current kappa".to_string(),
                )
            })?;

            let eval = evaluate_joint_reml_at_theta(
                self.y,
                self.weights,
                self.offset,
                self.cache.design(),
                theta,
                self.rho_dim,
                hyper_dirs,
                None,
                self.family,
                self.options,
            );
            if let Ok(ref value) = eval {
                self.cache.store_eval(value.clone());
            }
            eval
        }

        /// Cost-only evaluation — uses the same evaluator as eval_full to ensure
        /// the line search and gradient see the same objective function.
        fn eval_cost(&mut self, theta: &Array1<f64>) -> f64 {
            if let Some(cost) = self.cache.memoized_cost(theta) {
                return cost;
            }
            match self.eval_full(theta) {
                Ok((cost, _, _)) => cost,
                Err(_) => f64::INFINITY,
            }
        }

        fn track_best(&mut self, _: &Array1<f64>, cost: f64) {
            if cost < self.best_cost {
                self.best_cost = cost;
            }
        }
    }

    let mut ctx = IsoJointContext {
        data,
        y,
        weights,
        offset,
        family,
        options,
        rho_dim,
        best_cost: f64::INFINITY,
        cache: SingleBlockExactJointDesignCache::new(
            data,
            resolvedspec.clone(),
            baseline_design.clone(),
            spatial_terms.to_vec(),
            rho_dim,
            dims_per_term.to_vec(),
        )
        .map_err(EstimationError::InvalidInput)?,
    };

    let outer_config = OuterConfig {
        tolerance: 1e-6,
        max_iter: 50,
        fd_step: 1e-5,
        bounds: Some((lower.clone(), upper.clone())),
        seed_config: crate::seeding::SeedConfig {
            max_seeds: 1,
            screening_budget: 1,
            num_auxiliary_trailing: kappa_dim,
            ..Default::default()
        },
        rho_bound: 12.0,
        heuristic_lambdas: Some(theta0.as_slice().unwrap().to_vec()),
        initial_rho: Some(theta0.clone()),
        fallback_policy: FallbackPolicy::Automatic,
        screening_cap: None,
    };

    let mut obj = ClosureObjective {
        state: &mut ctx,
        cap: OuterCapability {
            gradient: Derivative::Analytic,
            hessian: Derivative::Analytic,
            n_params: theta_dim,
            // κ coordinates move the design (like ψ in aniso), so EFS is
            // not appropriate.
            all_penalty_like: false,
            has_psi_coords: true,
            fixed_point_available: false,
            barrier_config: None,
        },
        cost_fn: |ctx: &mut &mut IsoJointContext<'_>, theta: &Array1<f64>| {
            let cost = ctx.eval_cost(theta);
            ctx.track_best(theta, cost);
            Ok(cost)
        },
        eval_fn: |ctx: &mut &mut IsoJointContext<'_>, theta: &Array1<f64>| match ctx
            .eval_full(theta)
        {
            Ok((cost, grad, hess)) => {
                ctx.track_best(theta, cost);
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: HessianResult::Analytic(hess),
                })
            }
            Err(_) => Ok(OuterEval::infeasible(theta.len())),
        },
        reset_fn: None::<fn(&mut &mut IsoJointContext<'_>)>,
        efs_fn: None::<
            fn(&mut &mut IsoJointContext<'_>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    };

    let result =
        crate::solver::outer_strategy::run_outer(&mut obj, &outer_config, "iso-kappa joint REML")
            .map_err(|e| {
            EstimationError::InvalidInput(format!(
                "isotropic analytic optimization failed after exhausting strategy fallbacks: {e}"
            ))
        })?;
    log::trace!(
        "[spatial-iso-joint] converged in {} iterations, final_value={:.6e}, grad_norm={:.6e}",
        result.iterations,
        result.final_value,
        result.final_grad_norm,
    );
    Ok(result.rho)
}

fn set_spatial_length_scale(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    length_scale: f64,
) -> Result<(), EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        return Err(EstimationError::InvalidInput(format!(
            "spatial length-scale term index {term_idx} out of range"
        )));
    };
    match &mut term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.length_scale = Some(length_scale);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not expose a spatial length scale",
            term.name
        ))),
    }
}

pub fn get_spatial_length_scale(spec: &TermCollectionSpec, term_idx: usize) -> Option<f64> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { spec, .. } => Some(spec.length_scale),
            SmoothBasisSpec::Matern { spec, .. } => Some(spec.length_scale),
            SmoothBasisSpec::Duchon { spec, .. } => spec.length_scale,
            _ => None,
        })
}

pub(crate) fn freeze_spatial_length_scale_terms_from_design(
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<TermCollectionSpec, EstimationError> {
    if spec.smooth_terms.len() != design.smooth.terms.len() {
        return Err(EstimationError::InvalidInput(format!(
            "cannot freeze spatial length-scale terms: smooth spec count {} != fitted smooth term count {}",
            spec.smooth_terms.len(),
            design.smooth.terms.len()
        )));
    }

    let mut frozen = spec.clone();
    for (term, fitted) in frozen
        .smooth_terms
        .iter_mut()
        .zip(design.smooth.terms.iter())
    {
        if let (
            SmoothBasisSpec::Matern {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::Matern {
                centers,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                ..
            },
        ) = (&mut term.basis, &fitted.metadata)
        {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            if let Some(z) = identifiability_transform {
                s.identifiability = MaternIdentifiability::FrozenTransform {
                    transform: z.clone(),
                };
            }
            s.aniso_log_scales = meta_aniso.clone();
            *input_scales = meta_scales.clone();
        }
        if let (
            SmoothBasisSpec::Duchon {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::Duchon {
                centers,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                ..
            },
        ) = (&mut term.basis, &fitted.metadata)
        {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            if let Some(z) = identifiability_transform {
                s.identifiability = SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                };
            }
            s.aniso_log_scales = meta_aniso.clone();
            *input_scales = meta_scales.clone();
        }
        if let (
            SmoothBasisSpec::ThinPlate {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::ThinPlate {
                centers,
                length_scale,
                identifiability_transform,
                input_scales: meta_scales,
                ..
            },
        ) = (&mut term.basis, &fitted.metadata)
        {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.length_scale = *length_scale;
            if let Some(z) = identifiability_transform {
                s.identifiability = SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                };
            }
            *input_scales = meta_scales.clone();
        }
    }
    Ok(frozen)
}

#[derive(Debug, Clone)]
struct SingleSmoothTermRealization {
    design_local: Array2<f64>,
    term: SmoothTerm,
    dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
}

impl SingleSmoothTermRealization {
    fn active_penaltyinfo(&self) -> Vec<PenaltyInfo> {
        self.term
            .penaltyinfo_local
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect()
    }
}

fn build_single_smooth_term_realization(
    data: ArrayView2<'_, f64>,
    termspec: &SmoothTermSpec,
) -> Result<SingleSmoothTermRealization, BasisError> {
    let raw = build_smooth_design(data, std::slice::from_ref(termspec))?;
    finish_single_smooth_term_realization(raw, termspec)
}

/// Like `build_single_smooth_term_realization` but reuses a persistent
/// `BasisWorkspace` for distance-matrix caching across κ proposals.
fn build_single_smooth_term_realization_withworkspace(
    data: ArrayView2<'_, f64>,
    termspec: &SmoothTermSpec,
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<SingleSmoothTermRealization, BasisError> {
    let raw = build_smooth_design_withworkspace(data, std::slice::from_ref(termspec), workspace)?;
    finish_single_smooth_term_realization(raw, termspec)
}

fn finish_single_smooth_term_realization(
    raw: RawSmoothDesign,
    termspec: &SmoothTermSpec,
) -> Result<SingleSmoothTermRealization, BasisError> {
    let RawSmoothDesign {
        term_designs,
        dropped_penaltyinfo,
        terms,
        ..
    } = raw;
    let term = terms.into_iter().next().ok_or_else(|| {
        BasisError::InvalidInput("single-term smooth build returned no term".to_string())
    })?;
    let design = term_designs.into_iter().next().ok_or_else(|| {
        BasisError::InvalidInput("single-term smooth build returned no term design".to_string())
    })?;

    // build_smooth_design strips FrozenTransform identifiability to None for
    // ThinPlate/Duchon terms (so it can build the raw basis), but the frozen
    // transform must still be applied to reduce the column count to match the
    // original design.  Apply it here if the spec carries one.
    let design_local = match spatial_identifiability_policy(termspec) {
        Some(SpatialIdentifiability::FrozenTransform { transform }) => {
            if design.ncols() != transform.nrows() {
                return Err(BasisError::DimensionMismatch(format!(
                    "frozen spatial identifiability transform mismatch in incremental realizer: \
                     raw design has {} columns but transform has {} rows",
                    design.ncols(),
                    transform.nrows()
                )));
            }
            design.dot(transform)
        }
        _ => design,
    };

    Ok(SingleSmoothTermRealization {
        design_local,
        term,
        dropped_penaltyinfo,
    })
}

fn rebuild_smooth_auxiliary_state(
    smooth: &mut SmoothDesign,
    dropped_penaltyinfo_by_term: &[Vec<DroppedPenaltyBlockInfo>],
) -> Result<(), String> {
    if dropped_penaltyinfo_by_term.len() != smooth.terms.len() {
        return Err(format!(
            "smooth dropped-penalty cache mismatch: terms={}, dropped_sets={}",
            smooth.terms.len(),
            dropped_penaltyinfo_by_term.len()
        ));
    }

    let total_p = smooth.total_smooth_cols();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraint_b: Vec<f64> = Vec::new();

    for term in &smooth.terms {
        let range = term.coeff_range.clone();
        if let Some(lb_local) = term.lower_bounds_local.as_ref() {
            if lb_local.len() != range.len() {
                return Err(format!(
                    "smooth lower-bound cache mismatch for term '{}': bounds={}, coeffs={}",
                    term.name,
                    lb_local.len(),
                    range.len()
                ));
            }
            coefficient_lower_bounds
                .slice_mut(s![range.clone()])
                .assign(lb_local);
            any_bounds = true;
        }
        if let Some(lin_local) = term.linear_constraints_local.as_ref() {
            if lin_local.a.ncols() != range.len() {
                return Err(format!(
                    "smooth linear-constraint cache mismatch for term '{}': cols={}, coeffs={}",
                    term.name,
                    lin_local.a.ncols(),
                    range.len()
                ));
            }
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![range.clone()]).assign(&lin_local.a.row(r));
                linear_constraintrows.push(row);
                linear_constraint_b.push(lin_local.b[r]);
            }
        }
    }

    smooth.coefficient_lower_bounds = if any_bounds {
        Some(coefficient_lower_bounds)
    } else {
        None
    };
    smooth.linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), total_p));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };
    smooth.dropped_penaltyinfo = dropped_penaltyinfo_by_term
        .iter()
        .flat_map(|infos| infos.iter().cloned())
        .collect();
    Ok(())
}

fn rebuild_term_collection_auxiliary_state(
    spec: &TermCollectionSpec,
    design: &mut TermCollectionDesign,
) -> Result<(), String> {
    if spec.linear_terms.len() != design.linear_ranges.len() {
        return Err(format!(
            "term-collection linear bookkeeping mismatch: spec_terms={}, design_ranges={}",
            spec.linear_terms.len(),
            design.linear_ranges.len()
        ));
    }

    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(p_total, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraint_b: Vec<f64> = Vec::new();

    for (linear, (_, range)) in spec.linear_terms.iter().zip(design.linear_ranges.iter()) {
        if range.len() != 1 {
            return Err(format!(
                "linear term '{}' expected one coefficient column, found {}",
                linear.name,
                range.len()
            ));
        }
        let col = range.start;
        if let Some(lb) = linear.coefficient_min {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = 1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(lb);
        }
        if let Some(ub) = linear.coefficient_max {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = -1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(-ub);
        }
    }

    if let Some(lb_smooth) = design.smooth.coefficient_lower_bounds.as_ref() {
        if lb_smooth.len() != design.smooth.total_smooth_cols() {
            return Err(format!(
                "smooth lower-bound width mismatch: bounds={}, smooth_cols={}",
                lb_smooth.len(),
                design.smooth.total_smooth_cols()
            ));
        }
        coefficient_lower_bounds
            .slice_mut(s![
                smooth_start..(smooth_start + design.smooth.total_smooth_cols())
            ])
            .assign(lb_smooth);
        any_bounds = true;
    }
    if let Some(lin_smooth) = design.smooth.linear_constraints.as_ref() {
        if lin_smooth.a.ncols() != design.smooth.total_smooth_cols() {
            return Err(format!(
                "smooth linear-constraint width mismatch: cols={}, smooth_cols={}",
                lin_smooth.a.ncols(),
                design.smooth.total_smooth_cols()
            ));
        }
        let mut a_global = Array2::<f64>::zeros((lin_smooth.a.nrows(), p_total));
        a_global
            .slice_mut(s![
                ..,
                smooth_start..(smooth_start + design.smooth.total_smooth_cols())
            ])
            .assign(&lin_smooth.a);
        for r in 0..a_global.nrows() {
            linear_constraintrows.push(a_global.row(r).to_owned());
            linear_constraint_b.push(lin_smooth.b[r]);
        }
    }

    let lower_bound_constraints = if any_bounds {
        linear_constraints_from_lower_bounds_global(&coefficient_lower_bounds)
    } else {
        None
    };
    let explicit_linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), p_total));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };

    design.coefficient_lower_bounds = if any_bounds {
        Some(coefficient_lower_bounds)
    } else {
        None
    };
    design.linear_constraints =
        merge_linear_constraints_global(explicit_linear_constraints, lower_bound_constraints);
    design.dropped_penaltyinfo = design.smooth.dropped_penaltyinfo.clone();
    Ok(())
}

fn theta_values_match(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(&l, &r)| l.to_bits() == r.to_bits())
}

fn spatial_psi_to_length_scale_and_aniso(psi: &[f64]) -> (f64, Option<Vec<f64>>) {
    if psi.len() <= 1 {
        ((-psi.first().copied().unwrap_or(0.0)).exp(), None)
    } else {
        let psi_bar = psi.iter().sum::<f64>() / psi.len() as f64;
        (
            (-psi_bar).exp(),
            Some(psi.iter().map(|&value| value - psi_bar).collect()),
        )
    }
}

fn spatial_aniso_matches(left: Option<&[f64]>, right: Option<&[f64]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(&x, &y)| x.to_bits() == y.to_bits())
        }
        _ => false,
    }
}

#[derive(Debug, Clone)]
struct FrozenTermCollectionIncrementalRealizer<'d> {
    data: ArrayView2<'d, f64>,
    spec: TermCollectionSpec,
    design: TermCollectionDesign,
    dropped_penaltyinfo_by_term: Vec<Vec<DroppedPenaltyBlockInfo>>,
    smooth_penalty_ranges: Vec<Range<usize>>,
    full_penalty_ranges: Vec<Range<usize>>,
    full_smooth_start: usize,
    /// Persistent workspace for basis cache reuse across κ proposals.
    /// Distance matrices are cached here so they're computed once and
    /// reused across repeated `apply_log_kappa_to_term` calls.
    basisworkspace: crate::basis::BasisWorkspace,
}

impl<'d> FrozenTermCollectionIncrementalRealizer<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
    ) -> Result<Self, String> {
        if spec.smooth_terms.len() != design.smooth.terms.len() {
            return Err(format!(
                "incremental realizer smooth term mismatch: spec_terms={}, design_terms={}",
                spec.smooth_terms.len(),
                design.smooth.terms.len()
            ));
        }

        let mut smooth_cursor = 0usize;
        let mut smooth_penalty_ranges = Vec::with_capacity(design.smooth.terms.len());
        for term in &design.smooth.terms {
            let next = smooth_cursor + term.penalties_local.len();
            smooth_penalty_ranges.push(smooth_cursor..next);
            smooth_cursor = next;
        }
        if smooth_cursor != design.smooth.penalties.len() {
            return Err(format!(
                "incremental realizer smooth penalty mismatch: ranged={}, actual={}",
                smooth_cursor,
                design.smooth.penalties.len()
            ));
        }

        let fixed_penalty_offset = design
            .penalties
            .len()
            .checked_sub(design.smooth.penalties.len())
            .ok_or_else(|| {
                "incremental realizer encountered invalid penalty bookkeeping".to_string()
            })?;
        let full_penalty_ranges = smooth_penalty_ranges
            .iter()
            .map(|range| (fixed_penalty_offset + range.start)..(fixed_penalty_offset + range.end))
            .collect::<Vec<_>>();

        let mut dropped_penaltyinfo_by_term = Vec::with_capacity(spec.smooth_terms.len());
        for (term_idx, termspec) in spec.smooth_terms.iter().enumerate() {
            let realization =
                build_single_smooth_term_realization(data, termspec).map_err(|e| {
                    format!(
                        "failed to build cached realization for smooth term '{}' (index {}): {e}",
                        termspec.name, term_idx
                    )
                })?;
            let expected_cols = design.smooth.terms[term_idx].coeff_range.len();
            if realization.design_local.ncols() != expected_cols {
                return Err(format!(
                    "cached realization width mismatch for term '{}': cached_cols={}, design_cols={}",
                    termspec.name,
                    realization.design_local.ncols(),
                    expected_cols
                ));
            }
            if realization.active_penaltyinfo().len()
                != design.smooth.terms[term_idx].penalties_local.len()
            {
                return Err(format!(
                    "cached realization penalty mismatch for term '{}': cached_penalties={}, design_penalties={}",
                    termspec.name,
                    realization.active_penaltyinfo().len(),
                    design.smooth.terms[term_idx].penalties_local.len()
                ));
            }
            dropped_penaltyinfo_by_term.push(realization.dropped_penaltyinfo);
        }

        let full_smooth_start = design
            .design
            .ncols()
            .saturating_sub(design.smooth.total_smooth_cols());
        Ok(Self {
            data,
            spec,
            design,
            dropped_penaltyinfo_by_term,
            smooth_penalty_ranges,
            full_penalty_ranges,
            full_smooth_start,
            basisworkspace: crate::basis::BasisWorkspace::new(),
        })
    }

    fn spec(&self) -> &TermCollectionSpec {
        &self.spec
    }

    fn design(&self) -> &TermCollectionDesign {
        &self.design
    }

    fn apply_log_kappa(
        &mut self,
        log_kappa: &SpatialLogKappaCoords,
        term_indices: &[usize],
    ) -> Result<(), String> {
        if term_indices.len() != log_kappa.dims_per_term().len() {
            return Err(format!(
                "incremental realizer log-kappa term mismatch: term_indices={}, dims_per_term={}",
                term_indices.len(),
                log_kappa.dims_per_term().len()
            ));
        }

        let mut any_changed = false;
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            any_changed |= self.apply_log_kappa_to_term(term_idx, log_kappa.term_slice(slot))?;
        }

        if any_changed {
            rebuild_smooth_auxiliary_state(
                &mut self.design.smooth,
                &self.dropped_penaltyinfo_by_term,
            )?;
            rebuild_term_collection_auxiliary_state(&self.spec, &mut self.design)?;
        }
        Ok(())
    }

    fn apply_log_kappa_to_term(&mut self, term_idx: usize, psi: &[f64]) -> Result<bool, String> {
        let current_length_scale =
            get_spatial_length_scale(&self.spec, term_idx).ok_or_else(|| {
                format!(
                    "incremental realizer term {term_idx} does not expose a spatial length scale"
                )
            })?;
        let current_aniso = get_spatial_aniso_log_scales(&self.spec, term_idx);
        let (next_length_scale, next_aniso) = spatial_psi_to_length_scale_and_aniso(psi);
        let same_length = current_length_scale.to_bits() == next_length_scale.to_bits();
        let same_aniso = spatial_aniso_matches(current_aniso.as_deref(), next_aniso.as_deref());
        if same_length && same_aniso {
            return Ok(false);
        }

        set_spatial_length_scale(&mut self.spec, term_idx, next_length_scale)
            .map_err(|e| e.to_string())?;
        if let Some(eta) = next_aniso {
            set_spatial_aniso_log_scales(&mut self.spec, term_idx, eta)
                .map_err(|e| e.to_string())?;
        }

        let termspec = self
            .spec
            .smooth_terms
            .get(term_idx)
            .ok_or_else(|| format!("incremental realizer smooth term {term_idx} out of range"))?
            .clone();
        let realization =
            build_single_smooth_term_realization_withworkspace(
                self.data,
                &termspec,
                &mut self.basisworkspace,
            )
            .map_err(|e| {
                format!(
                    "failed to rebuild smooth term '{}' during incremental κ realization: {e}",
                    termspec.name
                )
            })?;
        self.replace_term_realization(term_idx, realization)?;
        Ok(true)
    }

    fn replace_term_realization(
        &mut self,
        term_idx: usize,
        realization: SingleSmoothTermRealization,
    ) -> Result<(), String> {
        let SingleSmoothTermRealization {
            design_local,
            term,
            dropped_penaltyinfo,
        } = realization;
        let SmoothTerm {
            name,
            penalties_local,
            nullspace_dims,
            penaltyinfo_local,
            metadata,
            lower_bounds_local,
            linear_constraints_local,
            ..
        } = term;
        let coeff_range = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .ok_or_else(|| format!("incremental realizer smooth term {term_idx} out of range"))?
            .coeff_range
            .clone();
        let full_range = (self.full_smooth_start + coeff_range.start)
            ..(self.full_smooth_start + coeff_range.end);
        if design_local.ncols() != coeff_range.len() {
            return Err(format!(
                "incremental realizer width mismatch for term {}: rebuilt_cols={}, cached_cols={}",
                term_idx,
                design_local.ncols(),
                coeff_range.len()
            ));
        }
        if design_local.nrows() != self.design.design.nrows() {
            return Err(format!(
                "incremental realizer row mismatch for term {}: rebuilt_rows={}, design_rows={}",
                term_idx,
                design_local.nrows(),
                self.design.design.nrows()
            ));
        }

        let active_penaltyinfo = penaltyinfo_local
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect::<Vec<_>>();
        let smooth_penalty_range = self
            .smooth_penalty_ranges
            .get(term_idx)
            .ok_or_else(|| {
                format!("incremental realizer missing smooth penalty range for term {term_idx}")
            })?
            .clone();
        let full_penalty_range = self
            .full_penalty_ranges
            .get(term_idx)
            .ok_or_else(|| {
                format!("incremental realizer missing full penalty range for term {term_idx}")
            })?
            .clone();
        if active_penaltyinfo.len() != smooth_penalty_range.len()
            || penalties_local.len() != smooth_penalty_range.len()
            || nullspace_dims.len() != smooth_penalty_range.len()
        {
            return Err(format!(
                "incremental realizer topology changed for term '{}': penalties={}, infos={}, nullspaces={}, cached_penalties={}",
                name,
                penalties_local.len(),
                active_penaltyinfo.len(),
                nullspace_dims.len(),
                smooth_penalty_range.len()
            ));
        }

        self.design.smooth.term_designs[term_idx] = design_local.clone();
        self.design
            .design
            .slice_mut(s![.., full_range.clone()])
            .assign(&design_local);

        for (offset, penalty_local) in penalties_local.iter().enumerate() {
            let smooth_penalty_idx = smooth_penalty_range.start + offset;
            let full_penalty_idx = full_penalty_range.start + offset;
            let nullspace_dim = nullspace_dims[offset];
            let penalty_info = active_penaltyinfo[offset].clone();

            if penalty_local.nrows() != coeff_range.len()
                || penalty_local.ncols() != coeff_range.len()
            {
                return Err(format!(
                    "incremental realizer penalty shape mismatch for term '{}' penalty {}: \
                     penalty is {}x{} but coeff_range has {} columns",
                    name,
                    offset,
                    penalty_local.nrows(),
                    penalty_local.ncols(),
                    coeff_range.len()
                ));
            }

            let smooth_penalty = self
                .design
                .smooth
                .penalties
                .get_mut(smooth_penalty_idx)
                .ok_or_else(|| {
                    format!(
                        "incremental realizer smooth penalty {} out of range for term {}",
                        smooth_penalty_idx, term_idx
                    )
                })?;
            smooth_penalty.fill(0.0);
            smooth_penalty
                .slice_mut(s![coeff_range.clone(), coeff_range.clone()])
                .assign(penalty_local);

            let full_bp = self
                .design
                .penalties
                .get_mut(full_penalty_idx)
                .ok_or_else(|| {
                    format!(
                        "incremental realizer full penalty {} out of range for term {}",
                        full_penalty_idx, term_idx
                    )
                })?;
            // Update the blockwise penalty in-place: the col_range stays the
            // same (same smooth block in the global layout), only the local
            // matrix changes.
            full_bp.local.fill(0.0);
            let local_start = coeff_range.start;
            let local_end = coeff_range.end;
            full_bp
                .local
                .slice_mut(s![local_start..local_end, local_start..local_end])
                .assign(penalty_local);

            self.design.smooth.nullspace_dims[smooth_penalty_idx] = nullspace_dim;
            self.design.nullspace_dims[full_penalty_idx] = nullspace_dim;

            self.design.smooth.penaltyinfo[smooth_penalty_idx].global_index = smooth_penalty_idx;
            self.design.smooth.penaltyinfo[smooth_penalty_idx].termname = Some(name.clone());
            self.design.smooth.penaltyinfo[smooth_penalty_idx].penalty = penalty_info.clone();

            self.design.penaltyinfo[full_penalty_idx].global_index = full_penalty_idx;
            self.design.penaltyinfo[full_penalty_idx].termname = Some(name.clone());
            self.design.penaltyinfo[full_penalty_idx].penalty = penalty_info;
        }

        let target_term = self.design.smooth.terms.get_mut(term_idx).ok_or_else(|| {
            format!("incremental realizer smooth term {term_idx} disappeared during replacement")
        })?;
        target_term.penalties_local = penalties_local;
        target_term.nullspace_dims = nullspace_dims;
        target_term.penaltyinfo_local = penaltyinfo_local;
        target_term.metadata = metadata;
        target_term.lower_bounds_local = lower_bounds_local;
        target_term.linear_constraints_local = linear_constraints_local;
        self.dropped_penaltyinfo_by_term[term_idx] = dropped_penaltyinfo;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// N-block spatial length-scale optimizer.
// ---------------------------------------------------------------------------

pub struct SpatialLengthScaleOptimizationResult<FitOut> {
    pub resolved_specs: Vec<TermCollectionSpec>,
    pub designs: Vec<TermCollectionDesign>,
    pub fit: FitOut,
}

/// Exact-joint hyper-parameter setup for N-block spatial length-scale optimization.
#[derive(Debug, Clone)]
pub struct ExactJointHyperSetup {
    rho0: Array1<f64>,
    rho_lower: Array1<f64>,
    rho_upper: Array1<f64>,
    log_kappa0: SpatialLogKappaCoords,
    log_kappa_lower: SpatialLogKappaCoords,
    log_kappa_upper: SpatialLogKappaCoords,
}

impl ExactJointHyperSetup {
    fn sanitize_rho_seed(
        rho0: Array1<f64>,
        rho_lower: &Array1<f64>,
        rho_upper: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::from_iter(rho0.iter().enumerate().map(|(idx, &value)| {
            let lo = rho_lower[idx];
            let hi = rho_upper[idx];
            let fallback = 0.0_f64.clamp(lo, hi);
            if value.is_finite() {
                value.clamp(lo, hi)
            } else {
                fallback
            }
        }))
    }

    pub(crate) fn new(
        rho0: Array1<f64>,
        rho_lower: Array1<f64>,
        rho_upper: Array1<f64>,
        log_kappa0: SpatialLogKappaCoords,
        log_kappa_lower: SpatialLogKappaCoords,
        log_kappa_upper: SpatialLogKappaCoords,
    ) -> Self {
        let rho0 = Self::sanitize_rho_seed(rho0, &rho_lower, &rho_upper);
        Self {
            rho0,
            rho_lower,
            rho_upper,
            log_kappa0,
            log_kappa_lower,
            log_kappa_upper,
        }
    }

    pub(crate) fn rho_dim(&self) -> usize {
        self.rho0.len()
    }

    pub(crate) fn log_kappa_dim(&self) -> usize {
        self.log_kappa0.len()
    }

    pub(crate) fn theta0(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho0);
        out.slice_mut(s![self.rho_dim()..])
            .assign(self.log_kappa0.as_array());
        out
    }

    pub(crate) fn lower(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho_lower);
        out.slice_mut(s![self.rho_dim()..])
            .assign(self.log_kappa_lower.as_array());
        out
    }

    pub(crate) fn upper(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho_upper);
        out.slice_mut(s![self.rho_dim()..])
            .assign(self.log_kappa_upper.as_array());
        out
    }

    /// Per-term dimensionality layout for the psi block.
    pub(crate) fn log_kappa_dims_per_term(&self) -> Vec<usize> {
        self.log_kappa0.dims_per_term().to_vec()
    }
}

/// N-block design cache for exact-joint spatial length-scale optimization.
///
/// Each block owns a `FrozenTermCollectionIncrementalRealizer` and a list of
/// spatial term indices within that block's spec. The cache splits the
/// combined psi vector into per-block slices using precomputed offsets.
struct ExactJointDesignCache<'d> {
    realizers: Vec<FrozenTermCollectionIncrementalRealizer<'d>>,
    block_term_indices: Vec<Vec<usize>>,
    current_theta: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_eval: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
    rho_dim: usize,
    all_dims: Vec<usize>,
    block_term_counts: Vec<usize>,
}

impl<'d> ExactJointDesignCache<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        blocks: Vec<(TermCollectionSpec, TermCollectionDesign, Vec<usize>)>,
        rho_dim: usize,
        all_dims: Vec<usize>,
    ) -> Result<Self, String> {
        let n_blocks = blocks.len();
        let mut realizers = Vec::with_capacity(n_blocks);
        let mut block_term_indices = Vec::with_capacity(n_blocks);
        let mut block_term_counts = Vec::with_capacity(n_blocks);

        for (spec, design, terms) in blocks {
            block_term_counts.push(terms.len());
            block_term_indices.push(terms);
            realizers.push(FrozenTermCollectionIncrementalRealizer::new(
                data, spec, design,
            )?);
        }

        Ok(Self {
            realizers,
            block_term_indices,
            current_theta: None,
            last_cost: None,
            last_eval: None,
            rho_dim,
            all_dims,
            block_term_counts,
        })
    }

    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), String> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }

        let full_log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
            theta,
            self.rho_dim,
            self.all_dims.clone(),
        );

        // Split the full log_kappa into per-block sub-coords using split_at.
        // We split from the front iteratively: after extracting block 0..N-2,
        // the remainder is the last block.
        let n = self.realizers.len();
        let mut remaining = full_log_kappa;
        for block_idx in 0..n {
            let count = self.block_term_counts[block_idx];
            if block_idx < n - 1 {
                let (block_lk, rest) = remaining.split_at(count);
                self.realizers[block_idx]
                    .apply_log_kappa(&block_lk, &self.block_term_indices[block_idx])?;
                remaining = rest;
            } else {
                // Last block gets the remainder.
                self.realizers[block_idx]
                    .apply_log_kappa(&remaining, &self.block_term_indices[block_idx])?;
            }
        }

        self.current_theta = Some(theta.clone());
        self.last_cost = None;
        self.last_eval = None;
        Ok(())
    }

    fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval
                .as_ref()
                .map(|cached| cached.0)
                .or(self.last_cost)
        } else {
            None
        }
    }

    fn memoized_eval(
        &self,
        theta: &Array1<f64>,
    ) -> Option<(f64, Array1<f64>, Option<Array2<f64>>)> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval.clone()
        } else {
            None
        }
    }

    fn store_eval(&mut self, eval: (f64, Array1<f64>, Option<Array2<f64>>)) {
        self.last_cost = Some(eval.0);
        self.last_eval = Some(eval);
    }

    fn specs(&self) -> Vec<&TermCollectionSpec> {
        self.realizers.iter().map(|r| r.spec()).collect()
    }

    fn designs(&self) -> Vec<&TermCollectionDesign> {
        self.realizers.iter().map(|r| r.design()).collect()
    }
}

pub fn optimize_spatial_length_scale_exact_joint<FitOut, FitFn, ExactFn>(
    data: ArrayView2<'_, f64>,
    block_specs: &[TermCollectionSpec],
    block_term_indices: &[Vec<usize>],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    joint_setup: &ExactJointHyperSetup,
    analytic_joint_gradient_available: bool,
    analytic_joint_hessian_available: bool,
    mut fit_fn: FitFn,
    mut exact_fn: ExactFn,
) -> Result<SpatialLengthScaleOptimizationResult<FitOut>, String>
where
    FitOut: Clone,
    FitFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
    ) -> Result<FitOut, String>,
    ExactFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
        bool,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), String>,
{
    let n_blocks = block_specs.len();
    if block_term_indices.len() != n_blocks {
        return Err(format!(
            "block_specs ({}) and block_term_indices ({}) length mismatch",
            n_blocks,
            block_term_indices.len()
        ));
    }

    let log_kappa_dim = joint_setup.log_kappa_dim();

    // -----------------------------------------------------------------------
    // Fast path: kappa disabled or no spatial terms — build designs once.
    // -----------------------------------------------------------------------
    if !kappa_options.enabled || log_kappa_dim == 0 {
        let mut resolved_specs = Vec::with_capacity(n_blocks);
        let mut designs = Vec::with_capacity(n_blocks);
        for (blk_idx, spec) in block_specs.iter().enumerate() {
            let design = build_term_collection_design(data, spec).map_err(|e| {
                format!(
                    "failed to build block-{blk_idx} design during exact joint kappa optimization: {e}"
                )
            })?;
            let resolved = freeze_spatial_length_scale_terms_from_design(spec, &design)
                .map_err(|e| {
                    format!(
                        "failed to freeze block-{blk_idx} spatial basis centers during exact joint kappa bootstrap: {e}"
                    )
                })?;
            resolved_specs.push(resolved);
            designs.push(design);
        }
        let rho_only = joint_setup
            .theta0()
            .slice(s![..joint_setup.rho_dim()])
            .to_owned();

        // Build temporary owned slices for the closure call.
        let spec_refs: Vec<TermCollectionSpec> = resolved_specs.clone();
        let design_refs: Vec<TermCollectionDesign> = designs.clone();
        let fit = fit_fn(&rho_only, &spec_refs, &design_refs)?;
        return Ok(SpatialLengthScaleOptimizationResult {
            resolved_specs,
            designs,
            fit,
        });
    }

    // -----------------------------------------------------------------------
    // Full optimization path.
    // -----------------------------------------------------------------------
    let theta0 = joint_setup.theta0();
    let lower = joint_setup.lower();
    let upper = joint_setup.upper();
    if theta0.len() < log_kappa_dim || lower.len() != theta0.len() || upper.len() != theta0.len() {
        return Err(format!(
            "invalid exact joint theta setup: theta0={}, lower={}, upper={}, required_log_kappa_dim={}",
            theta0.len(),
            lower.len(),
            upper.len(),
            log_kappa_dim
        ));
    }
    let rho_dim = joint_setup.rho_dim();
    let all_dims = joint_setup.log_kappa_dims_per_term();

    // Build bootstrap designs and frozen specs for each block.
    let mut boot_designs = Vec::with_capacity(n_blocks);
    let mut best_specs = Vec::with_capacity(n_blocks);
    let mut total_design_cols = 0usize;
    for (blk_idx, spec) in block_specs.iter().enumerate() {
        let design = build_term_collection_design(data, spec).map_err(|e| {
            format!(
                "failed to build block-{blk_idx} design during exact joint kappa optimization: {e}"
            )
        })?;
        let frozen = freeze_spatial_length_scale_terms_from_design(spec, &design).map_err(|e| {
            format!(
                "failed to freeze block-{blk_idx} spatial basis centers during exact joint kappa bootstrap: {e}"
            )
        })?;
        total_design_cols += design.design.ncols();
        best_specs.push(frozen);
        boot_designs.push(design);
    }

    let analytic_outer_hessian_available = analytic_joint_hessian_available
        && crate::custom_family::joint_exact_analytic_outer_hessian_available(total_design_cols);

    let theta_dim = theta0.len();
    let psi_dim = theta_dim - rho_dim;

    // Build the cache with one realizer per block.
    let cache_blocks: Vec<(TermCollectionSpec, TermCollectionDesign, Vec<usize>)> = best_specs
        .iter()
        .zip(boot_designs.iter())
        .zip(block_term_indices.iter())
        .map(|((spec, design), terms)| (spec.clone(), design.clone(), terms.clone()))
        .collect();

    struct NBlockExactJointState<'d> {
        best_cost: f64,
        cache: ExactJointDesignCache<'d>,
    }

    impl<'d> NBlockExactJointState<'d> {
        fn track_best(&mut self, _: &Array1<f64>, cost: f64) {
            if cost < self.best_cost {
                self.best_cost = cost;
            }
        }
    }

    let mut state = NBlockExactJointState {
        best_cost: f64::INFINITY,
        cache: ExactJointDesignCache::new(data, cache_blocks, rho_dim, all_dims.clone())?,
    };

    let exact_fn_cell = std::cell::RefCell::new(&mut exact_fn);

    use crate::solver::outer_strategy::{
        ClosureObjective, Derivative, EfsEval, FallbackPolicy, HessianResult, OuterCapability,
        OuterConfig, OuterEval,
    };

    let outer_config = OuterConfig {
        tolerance: kappa_options.rel_tol.max(1e-6),
        max_iter: kappa_options.max_outer_iter.max(1),
        fd_step: 1e-4,
        bounds: Some((lower.clone(), upper.clone())),
        seed_config: crate::seeding::SeedConfig {
            max_seeds: 1,
            screening_budget: 1,
            num_auxiliary_trailing: psi_dim,
            ..Default::default()
        },
        rho_bound: 12.0,
        heuristic_lambdas: Some(theta0.as_slice().unwrap().to_vec()),
        initial_rho: Some(theta0.clone()),
        fallback_policy: FallbackPolicy::Automatic,
        screening_cap: None,
    };

    // Helper: collect specs and designs from cache into owned Vecs for closure calls.
    fn collect_specs(cache: &ExactJointDesignCache<'_>) -> Vec<TermCollectionSpec> {
        cache.specs().into_iter().cloned().collect()
    }
    fn collect_designs(cache: &ExactJointDesignCache<'_>) -> Vec<TermCollectionDesign> {
        cache.designs().into_iter().cloned().collect()
    }

    let result = {
        let mut obj = ClosureObjective {
            state: &mut state,
            cap: OuterCapability {
                gradient: if analytic_joint_gradient_available {
                    Derivative::Analytic
                } else {
                    Derivative::FiniteDifference
                },
                hessian: if analytic_outer_hessian_available {
                    Derivative::Analytic
                } else {
                    Derivative::Unavailable
                },
                n_params: theta_dim,
                all_penalty_like: false,
                has_psi_coords: true,
                fixed_point_available: false,
                barrier_config: None,
            },
            cost_fn: |ctx: &mut &mut NBlockExactJointState<'_>, theta: &Array1<f64>| {
                if let Some(cost) = ctx.cache.memoized_cost(theta) {
                    ctx.track_best(theta, cost);
                    return Ok(cost);
                }
                if ctx.cache.ensure_theta(theta).is_err() {
                    return Ok(f64::INFINITY);
                }
                let specs = collect_specs(&ctx.cache);
                let designs = collect_designs(&ctx.cache);
                match (&mut *exact_fn_cell.borrow_mut())(
                    &theta.slice(s![..rho_dim]).to_owned(),
                    &specs,
                    &designs,
                    false, // cost-only: skip outer Hessian
                ) {
                    Ok((cost, grad, hess)) => {
                        ctx.cache.store_eval((cost, grad, hess));
                        ctx.track_best(theta, cost);
                        Ok(cost)
                    }
                    Err(_) => Ok(f64::INFINITY),
                }
            },
            eval_fn: |ctx: &mut &mut NBlockExactJointState<'_>, theta: &Array1<f64>| {
                if let Some((cost, grad, hess)) = ctx.cache.memoized_eval(theta) {
                    ctx.track_best(theta, cost);
                    if !cost.is_finite() {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    if grad.iter().any(|v| !v.is_finite()) {
                        return Err(EstimationError::RemlOptimizationFailed(
                            "n-block exact-joint gradient contained non-finite values".to_string(),
                        ));
                    }
                    let hessian_result = match hess {
                        Some(h)
                            if h.nrows() == theta.len()
                                && h.ncols() == theta.len()
                                && h.iter().all(|v| v.is_finite()) =>
                        {
                            HessianResult::Analytic(h)
                        }
                        _ => HessianResult::Unavailable,
                    };
                    return Ok(OuterEval {
                        cost,
                        gradient: grad,
                        hessian: hessian_result,
                    });
                }
                if ctx.cache.ensure_theta(theta).is_err() {
                    return Ok(OuterEval::infeasible(theta.len()));
                }
                let specs = collect_specs(&ctx.cache);
                let designs = collect_designs(&ctx.cache);
                match (&mut *exact_fn_cell.borrow_mut())(
                    &theta.slice(s![..rho_dim]).to_owned(),
                    &specs,
                    &designs,
                    analytic_outer_hessian_available,
                ) {
                    Ok((cost, grad, hess)) => {
                        ctx.cache.store_eval((cost, grad.clone(), hess.clone()));
                        ctx.track_best(theta, cost);
                        if !cost.is_finite() {
                            return Ok(OuterEval::infeasible(theta.len()));
                        }
                        if grad.iter().any(|v| !v.is_finite()) {
                            return Err(EstimationError::RemlOptimizationFailed(
                                "n-block exact-joint gradient contained non-finite values"
                                    .to_string(),
                            ));
                        }
                        let hessian_result = match hess {
                            Some(h)
                                if h.nrows() == theta.len()
                                    && h.ncols() == theta.len()
                                    && h.iter().all(|v| v.is_finite()) =>
                            {
                                HessianResult::Analytic(h)
                            }
                            _ => HessianResult::Unavailable,
                        };
                        Ok(OuterEval {
                            cost,
                            gradient: grad,
                            hessian: hessian_result,
                        })
                    }
                    Err(e) => Err(EstimationError::RemlOptimizationFailed(format!(
                        "n-block exact-joint derivative evaluation failed: {e}"
                    ))),
                }
            },
            reset_fn: None::<fn(&mut &mut NBlockExactJointState<'_>)>,
            efs_fn: None::<
                fn(
                    &mut &mut NBlockExactJointState<'_>,
                    &Array1<f64>,
                ) -> Result<EfsEval, EstimationError>,
            >,
        };

        crate::solver::outer_strategy::run_outer(
            &mut obj,
            &outer_config,
            "n-block exact-joint spatial",
        )
        .map_err(|e| e.to_string())?
    }; // obj dropped here, releasing mutable borrow on state

    let theta_star = result.rho;

    state.cache.ensure_theta(&theta_star)?;

    let resolved_specs: Vec<TermCollectionSpec> = collect_specs(&state.cache);
    let designs: Vec<TermCollectionDesign> = collect_designs(&state.cache);

    let rho_star = theta_star.slice(s![..rho_dim]).to_owned();
    let fit = fit_fn(&rho_star, &resolved_specs, &designs)?;

    for spec in &resolved_specs {
        log_spatial_aniso_scales(spec);
    }

    Ok(SpatialLengthScaleOptimizationResult {
        resolved_specs,
        designs,
        fit,
    })
}

pub fn fit_term_collectionwith_spatial_length_scale_optimization(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    // For spatial terms with an explicit length scale, κ (= 1/length_scale)
    // changes kernel geometry nonlinearly. That means both basis values B and
    // penalty blocks S change, so each κ proposal rebuilds the spatial basis.
    //
    // When exact derivative information is available for the rebuilt basis and
    // penalty, kappa is promoted to a first-class outer hyperparameter beside
    // rho = log(lambda). In that mode this routine runs a joint outer solve in
    // theta = [rho, psi], where psi = log(kappa) = -log(length_scale), and the
    // optimizer is expected to consume a real joint Hessian. NewtonTrustRegion
    // and ARC are not meant to run on a gradient-only surrogate here.
    //
    // Any smooth with an explicit spatial length scale participates in this
    // outer solve. If an eligible spatial basis does not expose exact
    // log-kappa derivatives, that is now a hard error.
    let mut resolvedspec = spec.clone();
    let spatial_terms = spatial_length_scale_term_indices(&resolvedspec);
    let n = data.nrows();
    if !(y.len() == n && weights.len() == n && offset.len() == n) {
        return Err(EstimationError::InvalidInput(format!(
            "fit_term_collectionwith_spatial_length_scale_optimization row mismatch: n={}, y={}, weights={}, offset={}",
            n,
            y.len(),
            weights.len(),
            offset.len()
        )));
    }
    if !kappa_options.enabled || spatial_terms.is_empty() {
        let out = fit_term_collection_forspec(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            family,
            options,
        )?;
        return Ok(FittedTermCollectionWithSpec {
            fit: out.fit,
            design: out.design,
            resolvedspec,
            adaptive_diagnostics: out.adaptive_diagnostics,
        });
    }
    if kappa_options.max_outer_iter == 0 {
        return Err(EstimationError::InvalidInput(
            "spatial kappa optimization requires max_outer_iter >= 1".to_string(),
        ));
    }
    if !(kappa_options.log_step.is_finite() && kappa_options.log_step > 0.0) {
        return Err(EstimationError::InvalidInput(
            "spatial kappa optimization requires log_step > 0".to_string(),
        ));
    }
    if !(kappa_options.min_length_scale.is_finite()
        && kappa_options.max_length_scale.is_finite()
        && kappa_options.min_length_scale > 0.0
        && kappa_options.max_length_scale >= kappa_options.min_length_scale)
    {
        return Err(EstimationError::InvalidInput(
            "spatial kappa optimization requires valid positive length_scale bounds".to_string(),
        ));
    }

    let best = fit_term_collection_forspec(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        family,
        options,
    )?;
    resolvedspec = freeze_spatial_length_scale_terms_from_design(&resolvedspec, &best.design)?;
    // Sync knot-cloud-derived aniso contrasts from the basis metadata back
    // into the spec so the optimizer starts from the geometry-informed η values
    // rather than the zero sentinel from --scale-dimensions.
    sync_aniso_contrasts_from_metadata(&mut resolvedspec, &best.design.smooth);
    let initial_score = fit_score(&best.fit);
    if !initial_score.is_finite() {
        log::debug!("[spatial-kappa] initial profiled score is non-finite");
    }
    if !spatial_terms.is_empty() {
        if let Some(exact_joint) = try_exact_joint_spatial_length_scale_optimization(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            &best,
            family,
            options,
            kappa_options,
            &spatial_terms,
        )? {
            let exact_score = fit_score(&exact_joint.fit);
            if exact_score <= initial_score + 1e-10 {
                log_spatial_aniso_scales(&exact_joint.resolvedspec);
                return Ok(exact_joint);
            }
            log::warn!(
                "[spatial-kappa] exact joint score regressed ({:.6e} -> {:.6e}); keeping baseline",
                initial_score,
                exact_score
            );
            return Ok(FittedTermCollectionWithSpec {
                fit: best.fit,
                design: best.design,
                resolvedspec,
                adaptive_diagnostics: best.adaptive_diagnostics,
            });
        }
        return Err(EstimationError::InvalidInput(
            "exact joint spatial length-scale path is unavailable for one or more eligible spatial terms"
                .to_string(),
        ));
    }

    Ok(FittedTermCollectionWithSpec {
        fit: best.fit,
        design: best.design,
        resolvedspec,
        adaptive_diagnostics: best.adaptive_diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{
        BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec,
        DuchonNullspaceOrder, MaternBasisSpec, MaternIdentifiability, MaternNu,
        SpatialIdentifiability, ThinPlateBasisSpec,
    };
    use crate::estimate::AdaptiveRegularizationOptions;
    use crate::faer_ndarray::FaerSvd;
    use ndarray::array;

    fn numerical_rank(x: &Array2<f64>) -> usize {
        let (_, s, _) = x
            .svd(false, false)
            .expect("SVD should succeed in rank test");
        let sigma_max = s.iter().copied().fold(0.0_f64, f64::max);
        let tol = (x.nrows().max(x.ncols()).max(1) as f64) * f64::EPSILON * sigma_max.max(1.0);
        s.iter().filter(|&&sv| sv > tol).count()
    }

    fn residual_norm_to_column_space(x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let (u_opt, _, _) = x
            .svd(true, false)
            .expect("SVD should succeed in projection residual test");
        let u = u_opt.expect("left singular vectors should be present");
        let rank = numerical_rank(x);
        let mut proj = Array1::<f64>::zeros(y.len());
        for j in 0..rank.min(u.ncols()) {
            let uj = u.column(j);
            let coeff = uj.dot(y);
            proj += &(&uj.to_owned() * coeff);
        }
        let resid = y - &proj;
        resid.dot(&resid).sqrt()
    }

    fn isotropic_two_block_exact_joint_setup(
        meanspec: &TermCollectionSpec,
        noisespec: &TermCollectionSpec,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> ExactJointHyperSetup {
        let mean_terms = spatial_length_scale_term_indices(meanspec);
        let noise_terms = spatial_length_scale_term_indices(noisespec);
        let mean_log_kappa =
            SpatialLogKappaCoords::from_length_scales(meanspec, &mean_terms, kappa_options);
        let noise_log_kappa =
            SpatialLogKappaCoords::from_length_scales(noisespec, &noise_terms, kappa_options);
        let dims_per_term = mean_log_kappa
            .dims_per_term()
            .iter()
            .chain(noise_log_kappa.dims_per_term().iter())
            .copied()
            .collect::<Vec<_>>();
        let log_kappa0 = SpatialLogKappaCoords::new_with_dims(
            Array1::from_iter(
                mean_log_kappa
                    .as_array()
                    .iter()
                    .chain(noise_log_kappa.as_array().iter())
                    .copied(),
            ),
            dims_per_term.clone(),
        );
        ExactJointHyperSetup::new(
            Array1::zeros(0),
            Array1::zeros(0),
            Array1::zeros(0),
            log_kappa0,
            SpatialLogKappaCoords::lower_bounds_aniso(&dims_per_term, kappa_options),
            SpatialLogKappaCoords::upper_bounds_aniso(&dims_per_term, kappa_options),
        )
    }

    fn max_abs_diff_matrix(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        assert_eq!(a.dim(), b.dim());
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn max_abs_diff_vector(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn assert_term_collection_designs_match(
        left: &TermCollectionDesign,
        right: &TermCollectionDesign,
        label: &str,
    ) {
        let design_diff = max_abs_diff_matrix(&left.design, &right.design);
        assert!(
            design_diff <= 1e-10,
            "{label} design mismatch max_abs={design_diff}"
        );
        assert_eq!(
            left.penalties.len(),
            right.penalties.len(),
            "{label} penalty count mismatch"
        );
        for (idx, (lp, rp)) in left
            .penalties
            .iter()
            .zip(right.penalties.iter())
            .enumerate()
        {
            assert_eq!(
                lp.col_range, rp.col_range,
                "{label} penalty {idx} col_range mismatch"
            );
            let penalty_diff = max_abs_diff_matrix(&lp.local, &rp.local);
            assert!(
                penalty_diff <= 1e-10,
                "{label} penalty {idx} mismatch max_abs={penalty_diff}"
            );
        }
        assert_eq!(
            left.nullspace_dims, right.nullspace_dims,
            "{label} nullspace dims mismatch"
        );
        assert_eq!(
            left.penaltyinfo.len(),
            right.penaltyinfo.len(),
            "{label} penaltyinfo length mismatch"
        );
        for (idx, (linfo, rinfo)) in left
            .penaltyinfo
            .iter()
            .zip(right.penaltyinfo.iter())
            .enumerate()
        {
            assert_eq!(
                linfo.termname, rinfo.termname,
                "{label} penaltyinfo termname mismatch at {idx}"
            );
            assert_eq!(
                linfo.penalty.source, rinfo.penalty.source,
                "{label} penalty source mismatch at {idx}"
            );
            assert_eq!(
                linfo.penalty.active, rinfo.penalty.active,
                "{label} penalty active mismatch at {idx}"
            );
            assert_eq!(
                linfo.penalty.effective_rank, rinfo.penalty.effective_rank,
                "{label} penalty rank mismatch at {idx}"
            );
            assert_eq!(
                linfo.penalty.nullspace_dim_hint, rinfo.penalty.nullspace_dim_hint,
                "{label} penalty nullspace hint mismatch at {idx}"
            );
            assert!(
                (linfo.penalty.normalization_scale - rinfo.penalty.normalization_scale).abs()
                    <= 1e-10,
                "{label} penalty normalization mismatch at {idx}"
            );
        }
        match (
            left.coefficient_lower_bounds.as_ref(),
            right.coefficient_lower_bounds.as_ref(),
        ) {
            (Some(lb_left), Some(lb_right)) => {
                let diff = max_abs_diff_vector(lb_left, lb_right);
                assert!(diff <= 1e-10, "{label} lower-bound mismatch max_abs={diff}");
            }
            (None, None) => {}
            _ => panic!("{label} lower-bound presence mismatch"),
        }
        match (
            left.linear_constraints.as_ref(),
            right.linear_constraints.as_ref(),
        ) {
            (Some(c_left), Some(c_right)) => {
                let a_diff = max_abs_diff_matrix(&c_left.a, &c_right.a);
                let b_diff = max_abs_diff_vector(&c_left.b, &c_right.b);
                assert!(
                    a_diff <= 1e-10,
                    "{label} linear-constraint A mismatch max_abs={a_diff}"
                );
                assert!(
                    b_diff <= 1e-10,
                    "{label} linear-constraint b mismatch max_abs={b_diff}"
                );
            }
            (None, None) => {}
            _ => panic!("{label} linear-constraint presence mismatch"),
        }
    }

    #[test]
    fn smooth_design_assembles_terms_and_penalties() {
        let data = array![
            [0.0, 0.0, 0.2],
            [0.2, 0.1, 0.4],
            [0.4, 0.2, 0.6],
            [0.6, 0.4, 0.7],
            [0.8, 0.7, 0.9],
            [1.0, 1.0, 1.1]
        ];

        let terms = vec![
            SmoothTermSpec {
                name: "s_x0".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 4,
                        },
                        double_penalty: true,
                        identifiability: BSplineIdentifiability::default(),
                    },
                },
                shape: ShapeConstraint::None,
            },
            SmoothTermSpec {
                name: "tps_x1x2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1, 2],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            },
        ];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.nrows(), data.nrows());
        assert_eq!(sd.terms.len(), 2);
        // bspline double-penalty contributes two blocks; tps double-penalty also
        // contributes two blocks (bending + nullspace ridge).
        assert_eq!(sd.penalties.len(), 4);
        assert_eq!(sd.nullspace_dims.len(), 4);
        for s in &sd.penalties {
            assert_eq!(s.nrows(), sd.total_smooth_cols());
            assert_eq!(s.ncols(), sd.total_smooth_cols());
        }
    }

    #[test]
    fn shape_mapping_monotone_increasing_is_non_decreasing() {
        let theta = array![-1.0, 0.5, -0.2, 0.3];
        let beta = SmoothDesign::map_term_coefficients(&theta, ShapeConstraint::MonotoneIncreasing)
            .unwrap();
        for i in 1..beta.len() {
            assert!(beta[i] >= beta[i - 1]);
        }
    }

    #[test]
    fn build_smooth_design_rejectsmultiaxis_spatial_shape_constraints() {
        let data = array![[0.0, 0.0], [0.5, 0.2], [1.0, 0.4], [1.5, 0.6],];
        let terms = vec![SmoothTermSpec {
            name: "tps_shape".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 3 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                },
                input_scales: None,
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];

        let err = build_smooth_design(data.view(), &terms).expect_err("shape should be rejected");
        match err {
            BasisError::InvalidInput(msg) => {
                assert!(msg.contains("requires exactly 1 feature axis"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn build_smooth_design_accepts_monotone_thin_plate_1dwith_linear_constraints() {
        let data = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_tps".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                },
                input_scales: None,
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained thin-plate");
        assert!(sd.coefficient_lower_bounds.is_none());
        let lin = sd
            .linear_constraints
            .as_ref()
            .expect("linear constraints should be generated");
        assert!(lin.a.nrows() > 0);
        assert_eq!(lin.a.ncols(), sd.total_smooth_cols());
        assert_eq!(lin.b.len(), lin.a.nrows());
    }

    #[test]
    fn build_smooth_design_rewrites_thin_plate_knot_count_errorwith_term_context() {
        let data = array![
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.3],
            [0.4, 0.3, 0.5],
            [0.7, 0.6, 0.8],
        ];
        let terms = vec![SmoothTermSpec {
            name: "thinplate(pc1, pc2, pc3)".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1, 2],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 3 },
                    length_scale: 1.0,
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::default(),
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }];

        let err =
            build_smooth_design(data.view(), &terms).expect_err("expected knot-count failure");
        match err {
            BasisError::InvalidInput(msg) => {
                assert!(msg.contains("joint TPS term 'thinplate(pc1, pc2, pc3)'"));
                assert!(msg.contains("over 3 covariates"));
                assert!(msg.contains("with 3 centers"));
                assert!(msg.contains("minimum centers is 4"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn build_smooth_design_accepts_monotone_matern_1dwith_linear_constraints() {
        let data = array![[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: 0.7,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained Matérn");
        assert!(sd.coefficient_lower_bounds.is_none());
        let lin = sd
            .linear_constraints
            .as_ref()
            .expect("linear constraints should be generated");
        assert!(lin.a.nrows() > 0);
        assert_eq!(lin.a.ncols(), sd.total_smooth_cols());
        assert_eq!(lin.b.len(), lin.a.nrows());
    }

    #[test]
    fn build_smooth_design_accepts_monotone_duchon_1dwith_linear_constraints() {
        let data = array![[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_duchon".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0],
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: Some(0.9),
                    power: 3,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained Duchon");
        assert!(sd.coefficient_lower_bounds.is_none());
        let lin = sd
            .linear_constraints
            .as_ref()
            .expect("linear constraints should be generated");
        assert!(lin.a.nrows() > 0);
        assert_eq!(lin.a.ncols(), sd.total_smooth_cols());
        assert_eq!(lin.b.len(), lin.a.nrows());
    }

    #[test]
    fn build_smooth_design_accepts_monotone_bsplinewith_bounds() {
        let data = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
        let terms = vec![SmoothTermSpec {
            name: "mono_bs".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (0.0, 1.0),
                        num_internal_knots: 3,
                    },
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];
        let sd = build_smooth_design(data.view(), &terms).expect("shape-constrained bspline");
        let lb = sd
            .coefficient_lower_bounds
            .as_ref()
            .expect("lower bounds should be generated");
        assert_eq!(lb.len(), sd.total_smooth_cols());
        assert!(lb[0].is_infinite() && lb[0].is_sign_negative());
        for j in 1..lb.len() {
            assert_eq!(lb[j], 0.0);
        }
    }

    #[test]
    fn term_collection_design_combines_linear_and_smooth() {
        let data = array![
            [0.0, 0.0, 0.2],
            [0.2, 0.1, 0.4],
            [0.4, 0.2, 0.6],
            [0.6, 0.4, 0.7],
            [0.8, 0.7, 0.9],
            [1.0, 1.0, 1.1]
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: true,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_x1x2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1, 2],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let design = build_term_collection_design(data.view(), &spec).unwrap();
        assert_eq!(design.design.nrows(), data.nrows());
        assert_eq!(design.intercept_range, 0..1);
        assert!(
            design
                .design
                .column(design.intercept_range.start)
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-12)
        );
        assert!(design.design.ncols() >= 2);
        assert_eq!(design.linear_ranges.len(), 1);
        assert_eq!(design.random_effect_ranges.len(), 0);
        assert_eq!(design.penalties.len(), 3); // linear ridge + 2 smooth penalties (bending + nullspace)
        assert_eq!(design.nullspace_dims.len(), 3);
    }

    #[test]
    fn spatial_smooth_columns_do_not_duplicate_global_intercept() {
        let data = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.3],
            [0.6, 0.6],
            [0.8, 0.7],
            [1.0, 1.0],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::default(),
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).unwrap();
        let smooth_start = 1usize;
        let smooth_end = smooth_start + design.smooth.total_smooth_cols();
        for col in smooth_start..smooth_end {
            let is_all_ones = design
                .design
                .column(col)
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-12);
            assert!(
                !is_all_ones,
                "smooth column {col} unexpectedly duplicated intercept"
            );
        }
    }

    #[test]
    fn spatial_smooth_drops_matching_linear_trend_columns() {
        let data = array![
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.7, 0.9],
            [1.0, 0.8],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: false,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::default(),
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).unwrap();

        // Raw TPS width for k=4,d=2 is 4; we drop intercept + matching x0 linear component.
        assert_eq!(design.smooth.total_smooth_cols(), 2);

        let lin_col = design.linear_ranges[0].1.start;
        let linvalues = design.design.column(lin_col).to_owned();
        let smooth_start = 1 + spec.linear_terms.len();
        let smooth_end = smooth_start + design.smooth.total_smooth_cols();
        for col in smooth_start..smooth_end {
            let same_as_linear = design
                .design
                .column(col)
                .iter()
                .zip(linvalues.iter())
                .all(|(&a, &b)| (a - b).abs() < 1e-12);
            assert!(
                !same_as_linear,
                "smooth column {col} unexpectedly duplicated linear term column"
            );
        }
    }

    #[test]
    fn spatial_option5_is_orthogonal_to_parametric_block() {
        let data = array![
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.7, 0.9],
            [1.0, 0.8],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: false,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).unwrap();
        let n = data.nrows();
        let mut c = Array2::<f64>::zeros((n, 2));
        c.column_mut(0).fill(1.0);
        c.column_mut(1).assign(&data.column(0));
        let smooth_start = 1 + spec.linear_terms.len();
        let b = design
            .design
            .slice(s![
                ..,
                smooth_start..(smooth_start + design.smooth.total_smooth_cols())
            ])
            .to_owned();
        let rel = orthogonality_relative_residual(b.view(), c.view());
        assert!(
            rel <= 1e-10,
            "Option 5 orthogonality residual too large: {rel}"
        );
    }

    #[test]
    fn spatial_option5_does_not_overconstrain_on_nonoverlapping_linear_terms() {
        let n = 40usize;
        let p = 16usize;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                // Deterministic, non-collinear synthetic PCs.
                data[[i, j]] =
                    (i as f64) * 0.03 + (j as f64) * 0.11 + ((i * (j + 1)) as f64) * 1e-3;
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: (5..16)
                .map(|j| LinearTermSpec {
                    name: format!("pc{j}"),
                    feature_col: j,
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                })
                .collect(),
            random_effect_terms: vec![],
            smooth_terms: vec![
                SmoothTermSpec {
                    name: "tps_pc1".to_string(),
                    basis: SmoothBasisSpec::ThinPlate {
                        feature_cols: vec![1],
                        spec: ThinPlateBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                            length_scale: 1.0,
                            double_penalty: true,
                            identifiability: SpatialIdentifiability::OrthogonalToParametric,
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
                SmoothTermSpec {
                    name: "tps_pc2".to_string(),
                    basis: SmoothBasisSpec::ThinPlate {
                        feature_cols: vec![2],
                        spec: ThinPlateBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                            length_scale: 1.0,
                            double_penalty: true,
                            identifiability: SpatialIdentifiability::OrthogonalToParametric,
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
            ],
        };

        let out = build_term_collection_design(data.view(), &spec);
        assert!(
            out.is_ok(),
            "term-local Option 5 should not over-constrain non-overlapping smooth/linear terms: {:?}",
            out.err()
        );
    }

    #[test]
    fn spatial_frozen_transform_rebuild_is_exact_on_trainingrows() {
        let data = array![
            [0.0, 0.1],
            [0.2, 0.0],
            [0.3, 0.4],
            [0.5, 0.2],
            [0.7, 0.9],
            [1.0, 0.8],
        ];
        let fitspec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: false,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: None,
                coefficient_max: None,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: 1.0,
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::OrthogonalToParametric,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit_design = build_term_collection_design(data.view(), &fitspec).unwrap();
        let term_meta = &fit_design.smooth.terms[0].metadata;
        let (centers, length_scale, z) = match term_meta {
            BasisMetadata::ThinPlate {
                centers,
                length_scale,
                identifiability_transform,
                ..
            } => (
                centers.clone(),
                *length_scale,
                identifiability_transform
                    .clone()
                    .expect("fit-time Option 5 should store transform"),
            ),
            other => panic!("unexpected metadata variant: {other:?}"),
        };

        let frozenspec = TermCollectionSpec {
            linear_terms: fitspec.linear_terms.clone(),
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_xy".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::UserProvided(centers),
                        length_scale,
                        double_penalty: false,
                        identifiability: SpatialIdentifiability::FrozenTransform { transform: z },
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let frozen_design = build_term_collection_design(data.view(), &frozenspec).unwrap();

        assert_eq!(
            fit_design.smooth.term_designs.len(),
            frozen_design.smooth.term_designs.len(),
            "frozen transform rebuild term count mismatch"
        );
        let max_abs = fit_design
            .smooth
            .term_designs
            .iter()
            .zip(frozen_design.smooth.term_designs.iter())
            .flat_map(|(a, b)| {
                assert_eq!(a.dim(), b.dim());
                a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs())
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs <= 1e-12,
            "frozen transform rebuild mismatch max_abs={max_abs}"
        );
    }

    #[test]
    fn term_collection_design_adds_random_effect_dummy_blockwithridge() {
        let data = array![
            [0.1, 0.0],
            [0.2, 1.0],
            [0.3, 0.0],
            [0.4, 2.0],
            [0.5, 1.0],
            [0.6, 2.0],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "id".to_string(),
                feature_col: 1,
                drop_first_level: false,
                frozen_levels: None,
            }],
            smooth_terms: vec![],
        };
        let design = build_term_collection_design(data.view(), &spec).unwrap();
        assert_eq!(design.intercept_range, 0..1);
        // 3 observed levels -> 3 dummy columns
        assert_eq!(design.design.ncols(), 4);
        assert_eq!(design.random_effect_ranges.len(), 1);
        assert_eq!(design.penalties.len(), 1);
        assert_eq!(design.nullspace_dims, vec![0]);
        let (_, range) = &design.random_effect_ranges[0];
        for i in 0..design.design.nrows() {
            let row_sum: f64 = design.design.slice(s![i, range.clone()]).sum();
            assert!((row_sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn matern_smooth_buildswith_double_penalty_in_high_dim() {
        let n = 12usize;
        let d = 10usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.1 + (j as f64) * 0.03;
            }
        }

        let terms = vec![SmoothTermSpec {
            name: "matern_x".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                    length_scale: 0.75,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        // Spatial smooths use the canonical operator penalty triplet:
        // mass + tension + stiffness.
        assert_eq!(sd.penalties.len(), 3);
        assert_eq!(sd.nullspace_dims.len(), 3);
    }

    #[test]
    fn duchon_linear_nullspace_builds_and_reports_nullspace_dim() {
        let n = 14usize;
        let d = 10usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.07 + (j as f64) * 0.05;
            }
        }

        let terms = vec![SmoothTermSpec {
            name: "duchon_x".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: Some(0.9),
                    power: 3,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        assert_eq!(sd.penalties.len(), 3);
        assert_eq!(sd.nullspace_dims.len(), 3);
    }

    #[test]
    fn joint_duchon_orderzero_raw_smooth_build_preserves_unconstrained_basis() {
        let n = 12usize;
        let d = 4usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.13 + (j as f64) * 0.17;
            }
        }

        let terms = vec![SmoothTermSpec {
            name: "duchon_joint".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    length_scale: Some(1.0),
                    power: 1,
                    nullspace_order: DuchonNullspaceOrder::Zero,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).expect("joint duchon build");
        assert_eq!(sd.total_smooth_cols(), 4);
        match &sd.terms[0].metadata {
            BasisMetadata::Duchon {
                identifiability_transform,
                ..
            } => {
                assert!(
                    identifiability_transform.is_some(),
                    "raw smooth build should freeze Duchon orthogonality once the basis is built"
                );
            }
            other => panic!("expected Duchon metadata, got {other:?}"),
        }
    }

    #[test]
    fn term_collection_joint_duchon_carries_frozen_transform_into_metadata() {
        let n = 12usize;
        let d = 4usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.13 + (j as f64) * 0.17;
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon_joint".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: (0..d).collect(),
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: Some(1.0),
                        power: 1,
                        nullspace_order: DuchonNullspaceOrder::Zero,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design =
            build_term_collection_design(data.view(), &spec).expect("term collection design");
        let term = &design.smooth.terms[0];
        assert_eq!(term.coeff_range.len(), 3);
        match &term.metadata {
            BasisMetadata::Duchon {
                identifiability_transform,
                ..
            } => {
                let z = identifiability_transform
                    .as_ref()
                    .expect("term collection should store frozen Duchon transform");
                assert_eq!(z.nrows(), 4);
                assert_eq!(z.ncols(), 3);
            }
            other => panic!("expected Duchon metadata, got {other:?}"),
        }
    }

    #[test]
    fn adaptive_cache_respects_frozen_joint_duchon_transform() {
        let n = 12usize;
        let d = 4usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.13 + (j as f64) * 0.17;
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon_joint".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: (0..d).collect(),
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: Some(1.0),
                        power: 1,
                        nullspace_order: DuchonNullspaceOrder::Zero,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design =
            build_term_collection_design(data.view(), &spec).expect("term collection design");
        let caches =
            extract_spatial_operator_runtime_caches(&spec, &design).expect("adaptive caches");
        assert_eq!(caches.len(), 1);
        assert_eq!(
            caches[0].coeff_global_range.len(),
            design.smooth.terms[0].coeff_range.len()
        );
    }

    #[test]
    fn frozen_joint_duchonspec_rebuild_keeps_adaptive_cache_in_sync() {
        let n = 12usize;
        let d = 4usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.13 + (j as f64) * 0.17;
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon_joint".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: (0..d).collect(),
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: Some(1.0),
                        power: 1,
                        nullspace_order: DuchonNullspaceOrder::Zero,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).expect("base design");
        let frozen =
            freeze_spatial_length_scale_terms_from_design(&spec, &design).expect("freeze spec");
        let rebuilt = build_term_collection_design(data.view(), &frozen).expect("rebuilt design");
        let caches =
            extract_spatial_operator_runtime_caches(&frozen, &rebuilt).expect("adaptive caches");
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].termname, "duchon_joint");
        assert_eq!(rebuilt.smooth.terms[0].coeff_range.len(), 3);
    }

    #[test]
    fn frozen_joint_maternspec_rebuild_keeps_adaptive_cache_in_sync() {
        let n = 12usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            data[[i, 0]] = i as f64 * 0.13;
            data[[i, 1]] = (i as f64 * 0.17).sin();
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern_joint".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: (0..d).collect(),
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                        length_scale: 1.0,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).expect("base design");
        let frozen =
            freeze_spatial_length_scale_terms_from_design(&spec, &design).expect("freeze spec");
        let rebuilt = build_term_collection_design(data.view(), &frozen).expect("rebuilt design");
        let caches =
            extract_spatial_operator_runtime_caches(&frozen, &rebuilt).expect("adaptive caches");
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].termname, "matern_joint");
        assert_eq!(rebuilt.smooth.terms.len(), 1);
        assert!(!rebuilt.smooth.terms[0].coeff_range.is_empty());
    }

    #[test]
    fn tensor_bspline_term_builds_te_style_design_and_penalties() {
        let n = 10usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
            data[[i, 1]] = (i as f64 / (n as f64 - 1.0)).powi(2);
        }

        let spec_x = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 3,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };
        let spec_y = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 2,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::default(),
        };

        let terms = vec![SmoothTermSpec {
            name: "te_xy".to_string(),
            basis: SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginalspecs: vec![spec_x, spec_y],
                    double_penalty: true,
                    identifiability: TensorBSplineIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        // one Kronecker penalty per marginal + optional ridge
        assert_eq!(sd.penalties.len(), 3);
        assert_eq!(sd.nullspace_dims.len(), 3);
        assert!(sd.penalties.iter().all(|s| s.nrows() == sd.total_smooth_cols()));
        assert!(sd.penalties.iter().all(|s| s.ncols() == sd.total_smooth_cols()));
    }

    #[test]
    fn tensor_bspline_design_matches_extended_marginal_kronecker_product() {
        let data = array![[-0.2, 0.1], [0.2, 0.4], [0.7, 0.8], [1.2, 1.1],];
        let spec_x = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 3,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };
        let spec_y = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 2,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
        };
        let mx = build_bspline_basis_1d(data.column(0), &spec_x)
            .unwrap()
            .design;
        let my = build_bspline_basis_1d(data.column(1), &spec_y)
            .unwrap()
            .design;
        let expected = tensor_product_design_from_marginals(&[mx.clone(), my.clone()]).unwrap();

        let term = SmoothTermSpec {
            name: "te_xy".to_string(),
            basis: SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginalspecs: vec![spec_x, spec_y],
                    double_penalty: false,
                    identifiability: TensorBSplineIdentifiability::None,
                },
            },
            shape: ShapeConstraint::None,
        };
        let got = build_smooth_design(data.view(), &[term]).unwrap().term_designs.into_iter().next().unwrap();
        assert_eq!(got.dim(), expected.dim());
        for i in 0..got.nrows() {
            for j in 0..got.ncols() {
                assert!((got[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn tensor_bspline_design_is_identifiable_against_global_intercept() {
        let n = 120usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (3.0 * t).sin();
        }

        let tensor_term = SmoothTermSpec {
            name: "te_xy".to_string(),
            basis: SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginalspecs: vec![
                        BSplineBasisSpec {
                            degree: 3,
                            penalty_order: 2,
                            knotspec: BSplineKnotSpec::Generate {
                                data_range: (0.0, 1.0),
                                num_internal_knots: 6,
                            },
                            double_penalty: false,
                            identifiability: BSplineIdentifiability::default(),
                        },
                        BSplineBasisSpec {
                            degree: 3,
                            penalty_order: 2,
                            knotspec: BSplineKnotSpec::Generate {
                                data_range: (-1.0, 1.0),
                                num_internal_knots: 6,
                            },
                            double_penalty: false,
                            identifiability: BSplineIdentifiability::default(),
                        },
                    ],
                    double_penalty: false,
                    identifiability: TensorBSplineIdentifiability::default(),
                },
            },
            shape: ShapeConstraint::None,
        };

        let sd = build_smooth_design(data.view(), &[tensor_term.clone()]).unwrap();
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![tensor_term],
        };
        let full = build_term_collection_design(data.view(), &spec).unwrap();
        let ones = Array1::<f64>::ones(n);
        let sd_assembled = ndarray::concatenate(
            ndarray::Axis(1),
            &sd.term_designs.iter().map(|d| d.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let residualvs_tensor = residual_norm_to_column_space(&sd_assembled, &ones);
        let residualvs_full = residual_norm_to_column_space(&full.design, &ones);

        // Tensor block alone must not be able to represent the constant surface.
        assert!(residualvs_tensor > 1e-6);
        // With explicit intercept, constants should be represented (near) exactly.
        assert!(residualvs_full < 1e-8);
    }

    #[test]
    fn spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern_two_feature() {
        let n = 60usize;
        let d = 3usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.13).sin();
            let x2 = (i as f64 * 0.07).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            data[[i, 2]] = x2;
            y[i] = (2.5 * x0).sin() + 0.4 * x1 - 0.2 * x2;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: 20.0,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit_opts = FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_opts,
        )
        .expect("baseline fit should succeed");
        let baseline_score = fit_score(&baseline.fit);

        let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
            data.view(),
            y.clone(),
            weights.clone(),
            offset.clone(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_opts,
            &SpatialLengthScaleOptimizationOptions {
                enabled: true,
                max_outer_iter: 2,
                rel_tol: 1e-5,
                log_step: std::f64::consts::LN_2,
                min_length_scale: 1e-3,
                max_length_scale: 1e3,
            },
        )
        .expect("optimized fit should succeed");
        let optimized_score = fit_score(&optimized.fit);
        assert!(optimized_score <= baseline_score + 1e-10);

        let ls = match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.length_scale,
            _ => panic!("expected Matérn term"),
        };
        assert!(ls.is_finite() && (1e-3..=1e3).contains(&ls));

        match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                assert!(matches!(
                    spec.center_strategy,
                    CenterStrategy::UserProvided(_)
                ));
                assert!(matches!(
                    spec.identifiability,
                    MaternIdentifiability::FrozenTransform { .. }
                ));
            }
            _ => panic!("expected Matérn term"),
        }
    }

    #[test]
    fn exact_joint_two_block_spatial_length_scale_freezes_matern_centers() {
        let n = 40usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.21).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
        }

        let matern_term = |name: &str, length_scale: f64| SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        };

        let meanspec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![matern_term("mean_matern", 0.8)],
        };
        let noisespec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![matern_term("noise_matern", 1.1)],
        };

        let kappa_options = SpatialLengthScaleOptimizationOptions {
            enabled: true,
            max_outer_iter: 1,
            rel_tol: 1e-6,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-3,
            max_length_scale: 1e3,
        };
        let joint_setup =
            isotropic_two_block_exact_joint_setup(&meanspec, &noisespec, &kappa_options);
        let theta_dim = joint_setup.theta0().len();

        let mean_terms = spatial_length_scale_term_indices(&meanspec);
        let noise_terms = spatial_length_scale_term_indices(&noisespec);
        let solved = optimize_spatial_length_scale_exact_joint(
            data.view(),
            &[meanspec.clone(), noisespec.clone()],
            &[mean_terms, noise_terms],
            &kappa_options,
            &joint_setup,
            true,
            true,
            |rho, specs, designs| {
                assert!(rho.is_empty());
                assert_eq!(specs.len(), 2);
                Ok(designs[0].design.ncols() as f64
                    + designs[1].design.ncols() as f64
                    + designs[0].penalties.len() as f64
                    + designs[1].penalties.len() as f64)
            },
            |rho, specs, designs, need_hessian| {
                assert!(rho.is_empty());
                assert_eq!(specs.len(), 2);
                assert!(!designs.is_empty());
                Ok((
                    0.0,
                    Array1::zeros(theta_dim),
                    need_hessian.then(|| Array2::zeros((theta_dim, theta_dim))),
                ))
            },
        )
        .expect("exact joint two-block κ optimization should succeed");

        for resolved in [&solved.resolved_specs[0], &solved.resolved_specs[1]] {
            match &resolved.smooth_terms[0].basis {
                SmoothBasisSpec::Matern { spec, .. } => {
                    assert!(matches!(
                        spec.center_strategy,
                        CenterStrategy::UserProvided(_)
                    ));
                    assert!(matches!(
                        spec.identifiability,
                        MaternIdentifiability::FrozenTransform { .. }
                    ));
                }
                _ => panic!("expected Matérn term"),
            }
        }
    }

    #[test]
    fn incremental_frozen_realizer_matches_unified_full_rebuild() {
        let n = 24usize;
        let mut data = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = t;
            data[[i, 1]] = (0.35 * i as f64).sin();
            data[[i, 2]] = (i % 3) as f64;
            data[[i, 3]] = t * t;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin".to_string(),
                feature_col: 1,
                double_penalty: false,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: Some(-0.5),
                coefficient_max: None,
            }],
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "grp".to_string(),
                feature_col: 2,
                drop_first_level: false,
                frozen_levels: None,
            }],
            smooth_terms: vec![
                SmoothTermSpec {
                    name: "spatial".to_string(),
                    basis: SmoothBasisSpec::Matern {
                        feature_cols: vec![0, 1],
                        spec: MaternBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                            length_scale: 0.8,
                            nu: MaternNu::FiveHalves,
                            include_intercept: false,
                            double_penalty: true,
                            identifiability: MaternIdentifiability::CenterSumToZero,
                            aniso_log_scales: Some(vec![0.15, -0.15]),
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
                SmoothTermSpec {
                    name: "mono".to_string(),
                    basis: SmoothBasisSpec::BSpline1D {
                        feature_col: 3,
                        spec: BSplineBasisSpec {
                            degree: 3,
                            penalty_order: 2,
                            knotspec: BSplineKnotSpec::Generate {
                                data_range: (0.0, 1.0),
                                num_internal_knots: 3,
                            },
                            double_penalty: false,
                            identifiability: BSplineIdentifiability::None,
                        },
                    },
                    shape: ShapeConstraint::MonotoneIncreasing,
                },
            ],
        };

        let base_design = build_term_collection_design(data.view(), &spec).expect("base design");
        let frozen =
            freeze_spatial_length_scale_terms_from_design(&spec, &base_design).expect("freeze");
        let frozen_design =
            build_term_collection_design(data.view(), &frozen).expect("frozen design");
        let spatial_terms = spatial_length_scale_term_indices(&frozen);
        assert_eq!(spatial_terms, vec![0]);

        let smooth_start = frozen_design.design.ncols() - frozen_design.smooth.total_smooth_cols();
        let fixed_before = frozen_design.design.clone();
        let nonspatial_range = frozen_design.smooth.terms[1].coeff_range.clone();
        let full_nonspatial_range =
            (smooth_start + nonspatial_range.start)..(smooth_start + nonspatial_range.end);
        let mut realizer = FrozenTermCollectionIncrementalRealizer::new(
            data.view(),
            frozen.clone(),
            frozen_design.clone(),
        )
        .expect("incremental realizer");

        let updated_log_kappa = SpatialLogKappaCoords::new_with_dims(array![0.30, -0.20], vec![2]);
        let updated_spec = updated_log_kappa
            .apply_tospec(&frozen, &spatial_terms)
            .expect("updated spec");
        realizer
            .apply_log_kappa(&updated_log_kappa, &spatial_terms)
            .expect("incremental update");
        let rebuilt =
            build_term_collection_design(data.view(), &updated_spec).expect("rebuilt design");

        assert_term_collection_designs_match(realizer.design(), &rebuilt, "incremental realizer");

        let linear_range = frozen_design.linear_ranges[0].1.clone();
        let random_range = frozen_design.random_effect_ranges[0].1.clone();
        let updated_full = &realizer.design().design;
        let linear_diff = max_abs_diff_matrix(
            &fixed_before.slice(s![.., linear_range.clone()]).to_owned(),
            &updated_full.slice(s![.., linear_range]).to_owned(),
        );
        let random_diff = max_abs_diff_matrix(
            &fixed_before.slice(s![.., random_range.clone()]).to_owned(),
            &updated_full.slice(s![.., random_range]).to_owned(),
        );
        let nonspatial_diff = max_abs_diff_matrix(
            &fixed_before
                .slice(s![.., full_nonspatial_range.clone()])
                .to_owned(),
            &updated_full
                .slice(s![.., full_nonspatial_range.clone()])
                .to_owned(),
        );
        let spatial_range = frozen_design.smooth.terms[0].coeff_range.clone();
        let full_spatial_range =
            (smooth_start + spatial_range.start)..(smooth_start + spatial_range.end);
        let spatial_change = max_abs_diff_matrix(
            &fixed_before
                .slice(s![.., full_spatial_range.clone()])
                .to_owned(),
            &updated_full.slice(s![.., full_spatial_range]).to_owned(),
        );
        assert!(
            linear_diff <= 1e-12,
            "linear block changed max_abs={linear_diff}"
        );
        assert!(
            random_diff <= 1e-12,
            "random-effect block changed max_abs={random_diff}"
        );
        assert!(
            nonspatial_diff <= 1e-12,
            "unchanged smooth block changed max_abs={nonspatial_diff}"
        );
        assert!(
            spatial_change > 1e-8,
            "spatial block did not update max_abs={spatial_change}"
        );
    }

    #[test]
    fn two_block_exact_joint_design_cache_clears_memo_on_theta_change() {
        let n = 20usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.19 * i as f64).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
        }

        let matern_term = |name: &str, length_scale: f64| SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                    length_scale,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        };

        let meanspec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![matern_term("mean", 0.7)],
        };
        let noisespec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![matern_term("noise", 1.1)],
        };
        let kappa_options = SpatialLengthScaleOptimizationOptions {
            enabled: true,
            max_outer_iter: 1,
            rel_tol: 1e-6,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-3,
            max_length_scale: 1e3,
        };
        let joint_setup =
            isotropic_two_block_exact_joint_setup(&meanspec, &noisespec, &kappa_options);
        let theta0 = joint_setup.theta0();

        let mean_design = build_term_collection_design(data.view(), &meanspec).expect("mean");
        let noise_design = build_term_collection_design(data.view(), &noisespec).expect("noise");
        let mean_frozen = freeze_spatial_length_scale_terms_from_design(&meanspec, &mean_design)
            .expect("freeze mean");
        let noise_frozen = freeze_spatial_length_scale_terms_from_design(&noisespec, &noise_design)
            .expect("freeze noise");

        let mean_term_indices = spatial_length_scale_term_indices(&mean_frozen);
        let noise_term_indices = spatial_length_scale_term_indices(&noise_frozen);
        let mut cache = ExactJointDesignCache::new(
            data.view(),
            vec![
                (
                    mean_frozen.clone(),
                    mean_design.clone(),
                    mean_term_indices.clone(),
                ),
                (
                    noise_frozen.clone(),
                    noise_design.clone(),
                    noise_term_indices.clone(),
                ),
            ],
            joint_setup.rho_dim(),
            joint_setup.log_kappa_dims_per_term(),
        )
        .expect("n-block cache");

        cache.ensure_theta(&theta0).expect("initial theta");
        assert!(cache.memoized_cost(&theta0).is_none());
        assert!(cache.memoized_eval(&theta0).is_none());

        let eval = (
            2.25,
            Array1::<f64>::ones(theta0.len()),
            Some(Array2::<f64>::eye(theta0.len())),
        );
        cache.store_eval(eval.clone());
        let cached_eval = cache.memoized_eval(&theta0).expect("cached eval");
        assert!((cached_eval.0 - eval.0).abs() <= 1e-12);
        assert_eq!(cached_eval.1, eval.1);
        assert_eq!(cached_eval.2, eval.2);

        let mut theta1 = theta0.clone();
        theta1[joint_setup.rho_dim()] += 0.25;
        cache.ensure_theta(&theta1).expect("updated theta");
        assert!(cache.memoized_cost(&theta1).is_none());
        assert!(cache.memoized_eval(&theta1).is_none());

        let log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
            &theta1,
            joint_setup.rho_dim(),
            joint_setup.log_kappa_dims_per_term(),
        );
        let mean_terms = spatial_length_scale_term_indices(&mean_frozen);
        let noise_terms = spatial_length_scale_term_indices(&noise_frozen);
        let (mean_lk, noise_lk) = log_kappa.split_at(mean_terms.len());
        let mean_updated = mean_lk
            .apply_tospec(&mean_frozen, &mean_terms)
            .expect("mean updated spec");
        let noise_updated = noise_lk
            .apply_tospec(&noise_frozen, &noise_terms)
            .expect("noise updated spec");
        let mean_rebuilt =
            build_term_collection_design(data.view(), &mean_updated).expect("mean rebuilt");
        let noise_rebuilt =
            build_term_collection_design(data.view(), &noise_updated).expect("noise rebuilt");
        let cache_designs = cache.designs();
        assert_term_collection_designs_match(cache_designs[0], &mean_rebuilt, "mean cache");
        assert_term_collection_designs_match(cache_designs[1], &noise_rebuilt, "noise cache");
    }

    #[test]
    fn single_block_exact_joint_design_cache_clears_memo_on_theta_change() {
        let n = 22usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.23 * i as f64).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                        length_scale: 0.9,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).expect("design");
        let frozen =
            freeze_spatial_length_scale_terms_from_design(&spec, &design).expect("freeze spec");
        let spatial_terms = spatial_length_scale_term_indices(&frozen);
        let rho_dim = design.penalties.len();
        let dims_per_term = vec![1];
        let mut theta0 = Array1::<f64>::zeros(rho_dim + 1);
        theta0[rho_dim] = -get_spatial_length_scale(&frozen, spatial_terms[0])
            .expect("length scale")
            .ln();

        let mut cache = SingleBlockExactJointDesignCache::new(
            data.view(),
            frozen.clone(),
            design.clone(),
            spatial_terms.clone(),
            rho_dim,
            dims_per_term.clone(),
        )
        .expect("single-block cache");

        cache.ensure_theta(&theta0).expect("initial theta");
        assert!(cache.memoized_cost(&theta0).is_none());
        assert!(cache.memoized_eval(&theta0).is_none());

        let eval = (
            0.5,
            Array1::<f64>::ones(theta0.len()),
            Array2::<f64>::eye(theta0.len()),
        );
        cache.store_eval(eval.clone());
        let cached_eval = cache.memoized_eval(&theta0).expect("cached eval");
        assert!((cached_eval.0 - eval.0).abs() <= 1e-12);
        assert_eq!(cached_eval.1, eval.1);
        assert_eq!(cached_eval.2, eval.2);

        let mut theta1 = theta0.clone();
        theta1[rho_dim] += 0.35;
        cache.ensure_theta(&theta1).expect("updated theta");
        assert!(cache.memoized_cost(&theta1).is_none());
        assert!(cache.memoized_eval(&theta1).is_none());

        let updated_log_kappa =
            SpatialLogKappaCoords::from_theta_tail_with_dims(&theta1, rho_dim, dims_per_term);
        let updated_spec = updated_log_kappa
            .apply_tospec(&frozen, &spatial_terms)
            .expect("updated spec");
        let rebuilt =
            build_term_collection_design(data.view(), &updated_spec).expect("rebuilt design");
        assert_term_collection_designs_match(cache.design(), &rebuilt, "single-block cache");
    }

    #[test]
    fn exact_matern_log_kappa_derivative_uses_feature_columns_only() {
        let n = 24usize;
        let p = 17usize;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            for j in 1..p {
                data[[i, j]] = ((i + j) as f64 * 0.13).sin();
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                        length_scale: 0.4,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec)
            .expect("baseline Matérn design should build");
        let frozenspec = freeze_spatial_length_scale_terms_from_design(&spec, &design)
            .expect("freezing Matérn centers from design should succeed");

        match &frozenspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => match &spec.center_strategy {
                CenterStrategy::UserProvided(centers) => {
                    assert_eq!(centers.ncols(), 1, "frozen centers should stay term-local");
                }
                _ => panic!("expected frozen user-provided centers"),
            },
            _ => panic!("expected Matérn term"),
        }

        let derivative =
            try_build_spatial_term_log_kappa_derivative(data.view(), &frozenspec, &design, 0);
        assert!(
            derivative.is_ok(),
            "exact Matérn log-kappa derivative should use only feature_cols; got {derivative:?}"
        );
        assert!(
            derivative
                .expect("derivative call should succeed")
                .is_some(),
            "Matérn term should expose an exact derivative"
        );
    }

    #[test]
    fn exact_thin_plate_log_kappa_derivative_uses_feature_columns_only() {
        let n = 28usize;
        let p = 15usize;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.17 * i as f64).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            for j in 2..p {
                data[[i, j]] = ((i + 3 * j) as f64 * 0.07).cos();
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "thinplate".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0, 1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 7 },
                        length_scale: 0.7,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec)
            .expect("baseline ThinPlate design should build");
        let frozenspec = freeze_spatial_length_scale_terms_from_design(&spec, &design)
            .expect("freezing ThinPlate centers from design should succeed");

        match &frozenspec.smooth_terms[0].basis {
            SmoothBasisSpec::ThinPlate { spec, .. } => match &spec.center_strategy {
                CenterStrategy::UserProvided(centers) => {
                    assert_eq!(centers.ncols(), 2, "frozen centers should stay term-local");
                }
                _ => panic!("expected frozen user-provided centers"),
            },
            _ => panic!("expected ThinPlate term"),
        }

        let smooth_term = &design.smooth.terms[0];
        let termspec = &frozenspec.smooth_terms[0];
        let BasisPsiDerivativeResult {
            design_derivative: local_x_psi,
            penalties_derivative: local_s_psi,
        } = match &termspec.basis {
            SmoothBasisSpec::ThinPlate {
                feature_cols, spec, ..
            } => {
                let x = select_columns(data.view(), feature_cols)
                    .expect("select ThinPlate feature cols");
                crate::basis::build_thin_plate_basis_log_kappa_derivative(x.view(), spec)
                    .expect("direct ThinPlate derivative should build")
            }
            _ => panic!("expected ThinPlate term"),
        };
        let BasisPsiSecondDerivativeResult {
            designsecond_derivative: local_x_psi_psi,
            penaltiessecond_derivative: local_s_psi_psi,
        } = match &termspec.basis {
            SmoothBasisSpec::ThinPlate {
                feature_cols, spec, ..
            } => {
                let x = select_columns(data.view(), feature_cols)
                    .expect("select ThinPlate feature cols");
                crate::basis::build_thin_plate_basis_log_kappasecond_derivative(x.view(), spec)
                    .expect("direct ThinPlate second derivative should build")
            }
            _ => panic!("expected ThinPlate term"),
        };
        assert_eq!(local_x_psi.ncols(), smooth_term.coeff_range.len());
        assert_eq!(local_x_psi_psi.ncols(), smooth_term.coeff_range.len());
        assert!(!local_s_psi.is_empty());
        assert_eq!(local_s_psi.len(), local_s_psi_psi.len());
        assert!(local_s_psi.iter().all(|s| {
            s.nrows() == smooth_term.coeff_range.len() && s.ncols() == smooth_term.coeff_range.len()
        }));
        assert!(local_s_psi_psi.iter().all(|s| {
            s.nrows() == smooth_term.coeff_range.len() && s.ncols() == smooth_term.coeff_range.len()
        }));

        let derivative =
            try_build_spatial_term_log_kappa_derivative(data.view(), &frozenspec, &design, 0);
        assert!(
            derivative.is_ok(),
            "exact ThinPlate log-kappa derivative should use only feature_cols; got {derivative:?}"
        );
        let derivative = derivative.expect("derivative call should succeed");
        assert!(
            derivative.is_some(),
            "ThinPlate term should expose an exact derivative"
        );
    }

    #[test]
    fn exact_duchon_log_kappa_derivative_uses_feature_columns_only() {
        let n = 28usize;
        let p = 15usize;
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.21 * i as f64).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            for j in 2..p {
                data[[i, j]] = ((i + 2 * j) as f64 * 0.09).sin();
            }
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1],
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 7 },
                        length_scale: Some(0.7),
                        power: 1,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec)
            .expect("baseline Duchon design should build");
        let frozenspec = freeze_spatial_length_scale_terms_from_design(&spec, &design)
            .expect("freezing Duchon centers from design should succeed");

        match &frozenspec.smooth_terms[0].basis {
            SmoothBasisSpec::Duchon { spec, .. } => match &spec.center_strategy {
                CenterStrategy::UserProvided(centers) => {
                    assert_eq!(centers.ncols(), 2, "frozen centers should stay term-local");
                }
                _ => panic!("expected frozen user-provided centers"),
            },
            _ => panic!("expected Duchon term"),
        }

        let smooth_term = &design.smooth.terms[0];
        let termspec = &frozenspec.smooth_terms[0];
        let BasisPsiDerivativeResult {
            design_derivative: local_x_psi,
            penalties_derivative: local_s_psi,
        } = match &termspec.basis {
            SmoothBasisSpec::Duchon {
                feature_cols, spec, ..
            } => {
                let x =
                    select_columns(data.view(), feature_cols).expect("select Duchon feature cols");
                build_duchon_basis_log_kappa_derivative(x.view(), spec)
                    .expect("direct Duchon derivative should build")
            }
            _ => panic!("expected Duchon term"),
        };
        let BasisPsiSecondDerivativeResult {
            designsecond_derivative: local_x_psi_psi,
            penaltiessecond_derivative: local_s_psi_psi,
        } = match &termspec.basis {
            SmoothBasisSpec::Duchon {
                feature_cols, spec, ..
            } => {
                let x =
                    select_columns(data.view(), feature_cols).expect("select Duchon feature cols");
                build_duchon_basis_log_kappasecond_derivative(x.view(), spec)
                    .expect("direct Duchon second derivative should build")
            }
            _ => panic!("expected Duchon term"),
        };
        assert_eq!(local_x_psi.ncols(), smooth_term.coeff_range.len());
        assert_eq!(local_x_psi_psi.ncols(), smooth_term.coeff_range.len());
        assert!(!local_s_psi.is_empty());
        assert_eq!(local_s_psi.len(), local_s_psi_psi.len());
        assert!(local_s_psi.iter().all(|s| {
            s.nrows() == smooth_term.coeff_range.len() && s.ncols() == smooth_term.coeff_range.len()
        }));
        assert!(local_s_psi_psi.iter().all(|s| {
            s.nrows() == smooth_term.coeff_range.len() && s.ncols() == smooth_term.coeff_range.len()
        }));

        let derivative =
            try_build_spatial_term_log_kappa_derivative(data.view(), &frozenspec, &design, 0);
        assert!(
            derivative.is_ok(),
            "exact Duchon log-kappa derivative should use only feature_cols; got {derivative:?}"
        );
        let derivative = derivative.expect("derivative call should succeed");
        assert!(
            derivative.is_some(),
            "Duchon term should expose an exact derivative"
        );
    }

    #[test]
    fn spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern() {
        let n = 60usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).cos() + 0.35 * x1;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: 12.0,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit_opts = FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_opts,
        )
        .expect("baseline fit should succeed");
        let baseline_score = fit_score(&baseline.fit);

        let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
            data.view(),
            y.clone(),
            weights.clone(),
            offset.clone(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_opts,
            &SpatialLengthScaleOptimizationOptions {
                enabled: true,
                max_outer_iter: 2,
                rel_tol: 1e-5,
                log_step: std::f64::consts::LN_2,
                min_length_scale: 1e-3,
                max_length_scale: 1e3,
            },
        )
        .expect("optimized fit should succeed");
        let optimized_score = fit_score(&optimized.fit);
        assert!(optimized_score <= baseline_score + 1e-10);

        let ls = match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.length_scale,
            _ => panic!("expected Matérn term"),
        };
        assert!(ls.is_finite() && (1e-3..=1e3).contains(&ls));

        match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                assert!(matches!(
                    spec.center_strategy,
                    CenterStrategy::UserProvided(_)
                ));
                assert!(matches!(
                    spec.identifiability,
                    MaternIdentifiability::FrozenTransform { .. }
                ));
            }
            _ => panic!("expected Matérn term"),
        }
    }

    #[test]
    fn spatial_length_scale_optimization_supports_binomial_logit_matern() {
        let n = 80usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.19).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            let eta = -0.8 + 2.0 * x0 - 1.1 * x1;
            let mu = 1.0 / (1.0 + (-eta).exp());
            y[i] = if mu > 0.5 { 1.0 } else { 0.0 };
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                        length_scale: 1.8,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit_opts = FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::BinomialLogit,
            &fit_opts,
            &SpatialLengthScaleOptimizationOptions {
                enabled: true,
                max_outer_iter: 2,
                rel_tol: 1e-5,
                log_step: std::f64::consts::LN_2,
                min_length_scale: 1e-3,
                max_length_scale: 1e3,
            },
        )
        .expect("binomial-logit Matérn spatial κ optimization should succeed");

        let ls = match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.length_scale,
            _ => panic!("expected Matérn term"),
        };
        assert!(ls.is_finite() && (1e-3..=1e3).contains(&ls));
        assert!(optimized.fit.beta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn duchon_terms_participate_in_kappa_optimization() {
        let data = array![
            [0.0, 0.1],
            [0.2, 0.0],
            [0.4, 0.3],
            [0.6, 0.5],
            [0.8, 0.7],
            [1.0, 0.9],
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1],
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        length_scale: Some(0.9),
                        power: 1,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        assert_eq!(spatial_length_scale_term_indices(&spec), vec![0]);

        let fit_opts = FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 40,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
        };
        let y = Array1::linspace(0.0, 1.0, data.nrows());
        let weights = Array1::ones(data.nrows());
        let offset = Array1::zeros(data.nrows());

        let design = build_term_collection_design(data.view(), &spec)
            .expect("baseline Duchon design should build");
        let frozenspec = freeze_spatial_length_scale_terms_from_design(&spec, &design)
            .expect("freezing Duchon centers from design should succeed");
        let derivative =
            try_build_spatial_term_log_kappa_derivative(data.view(), &frozenspec, &design, 0);
        assert!(
            derivative
                .expect("Duchon exact derivative call should succeed")
                .is_some(),
            "Duchon term should expose an exact derivative"
        );

        let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
            data.view(),
            y,
            weights,
            offset,
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &fit_opts,
            &SpatialLengthScaleOptimizationOptions::default(),
        )
        .expect("Duchon fit should use exact κ optimization");

        let optimized_ls = match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Duchon { spec, .. } => spec.length_scale,
            _ => panic!("expected Duchon term"),
        };
        assert!(optimized_ls.is_some());
        match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Duchon { spec, .. } => {
                assert!(matches!(
                    spec.center_strategy,
                    CenterStrategy::UserProvided(_)
                ));
                assert!(matches!(
                    spec.identifiability,
                    SpatialIdentifiability::FrozenTransform { .. }
                ));
            }
            _ => panic!("expected Duchon term"),
        }
    }

    #[test]
    fn exact_joint_two_block_spatial_length_scale_freezes_duchon_centers() {
        let n = 40usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.19).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
        }

        let duchon_term = |name: &str, length_scale: f64| SmoothTermSpec {
            name: name.to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: Some(length_scale),
                    power: 3,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        };

        let meanspec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![duchon_term("mean_duchon", 0.8)],
        };
        let noisespec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![duchon_term("noise_duchon", 1.1)],
        };

        let kappa_options = SpatialLengthScaleOptimizationOptions {
            enabled: true,
            max_outer_iter: 1,
            rel_tol: 1e-6,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-3,
            max_length_scale: 1e3,
        };
        let joint_setup =
            isotropic_two_block_exact_joint_setup(&meanspec, &noisespec, &kappa_options);
        let theta_dim = joint_setup.theta0().len();

        let mean_terms = spatial_length_scale_term_indices(&meanspec);
        let noise_terms = spatial_length_scale_term_indices(&noisespec);
        let solved = optimize_spatial_length_scale_exact_joint(
            data.view(),
            &[meanspec.clone(), noisespec.clone()],
            &[mean_terms, noise_terms],
            &kappa_options,
            &joint_setup,
            true,
            true,
            |rho, specs, designs| {
                assert!(rho.is_empty());
                assert_eq!(specs.len(), 2);
                Ok(designs[0].design.ncols() as f64
                    + designs[1].design.ncols() as f64
                    + designs[0].penalties.len() as f64
                    + designs[1].penalties.len() as f64)
            },
            |rho, specs, designs, need_hessian| {
                assert!(rho.is_empty());
                assert_eq!(specs.len(), 2);
                assert!(!designs.is_empty());
                Ok((
                    0.0,
                    Array1::zeros(theta_dim),
                    need_hessian.then(|| Array2::zeros((theta_dim, theta_dim))),
                ))
            },
        )
        .expect("exact joint two-block spatial length-scale optimization should succeed");

        for resolved in [&solved.resolved_specs[0], &solved.resolved_specs[1]] {
            match &resolved.smooth_terms[0].basis {
                SmoothBasisSpec::Duchon { spec, .. } => {
                    assert!(matches!(
                        spec.center_strategy,
                        CenterStrategy::UserProvided(_)
                    ));
                    assert!(matches!(
                        spec.identifiability,
                        SpatialIdentifiability::FrozenTransform { .. }
                    ));
                }
                _ => panic!("expected Duchon term"),
            }
        }
    }

    #[test]
    fn bounded_linear_gaussian_fit_respects_interval() {
        let n = 64usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            let z = (i as f64) / ((n - 1) as f64);
            data[[i, 0]] = x;
            data[[i, 1]] = z;
            y[i] = 0.25 + 0.8 * x + 0.05 * z;
        }
        let spec = TermCollectionSpec {
            linear_terms: vec![
                LinearTermSpec {
                    name: "x".to_string(),
                    feature_col: 0,
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Bounded {
                        min: 0.0,
                        max: 0.5,
                        prior: BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 },
                    },
                    coefficient_min: None,
                    coefficient_max: None,
                },
                LinearTermSpec {
                    name: "z".to_string(),
                    feature_col: 1,
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                },
            ],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };

        let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
            data.view(),
            y,
            Array1::ones(n),
            Array1::zeros(n),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                mixture_link: None,
                optimize_mixture: false,
                sas_link: None,
                optimize_sas: false,
                compute_inference: true,
                max_iter: 40,
                tol: 1e-6,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: None,
                penalty_shrinkage_floor: None,
            },
            &SpatialLengthScaleOptimizationOptions {
                enabled: false,
                ..SpatialLengthScaleOptimizationOptions::default()
            },
        )
        .expect("bounded gaussian fit");

        let bounded_idx = fitted.design.linear_ranges[0].1.start;
        let estimate = fitted.fit.beta[bounded_idx];
        assert!(
            (0.0..=0.5).contains(&estimate),
            "bounded coefficient escaped interval: {estimate}"
        );
        assert!(
            estimate > 0.1,
            "bounded coefficient should move into the positive interior, got {estimate}"
        );
    }

    #[test]
    fn term_collection_design_emits_linear_coefficient_constraints() {
        let data = array![[0.0], [1.0], [2.0], [3.0]];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "x".to_string(),
                feature_col: 0,
                double_penalty: false,
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min: Some(0.0),
                coefficient_max: Some(1.0),
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };
        let design = build_term_collection_design(data.view(), &spec).expect("design");
        let constraints = design.linear_constraints.expect("constraints");
        assert_eq!(constraints.a.ncols(), design.design.ncols());
        assert_eq!(constraints.a.nrows(), 2);
        let linear_idx = design.linear_ranges[0].1.start;
        assert_eq!(constraints.a[[0, linear_idx]], 1.0);
        assert_eq!(constraints.b[0], 0.0);
        assert_eq!(constraints.a[[1, linear_idx]], -1.0);
        assert_eq!(constraints.b[1], -1.0);
    }

    #[test]
    fn linear_termspec_defaults_to_penalizedwhen_field_is_omitted() {
        let json = r#"{"name":"x","feature_col":0}"#;
        let term: LinearTermSpec = serde_json::from_str(json).expect("deserialize linear term");
        assert!(term.double_penalty);
        assert!(matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Unconstrained
        ));
    }

    #[test]
    fn linear_double_penalties_share_one_globalridge_block() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let spec = TermCollectionSpec {
            linear_terms: vec![
                LinearTermSpec {
                    name: "x1".to_string(),
                    feature_col: 0,
                    double_penalty: true,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                },
                LinearTermSpec {
                    name: "x2".to_string(),
                    feature_col: 1,
                    double_penalty: true,
                    coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                    coefficient_min: None,
                    coefficient_max: None,
                },
            ],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        };
        let design = build_term_collection_design(data.view(), &spec).expect("design");
        assert_eq!(design.penalties.len(), 1);
        assert_eq!(design.penaltyinfo.len(), 1);
        assert_eq!(design.penaltyinfo[0].termname.as_deref(), Some("linear"));
        assert_eq!(design.penaltyinfo[0].penalty.effective_rank, 2);
        let x1 = design.linear_ranges[0].1.start;
        let x2 = design.linear_ranges[1].1.start;
        let bp = &design.penalties[0];
        let x1_local = x1 - bp.col_range.start;
        let x2_local = x2 - bp.col_range.start;
        assert_eq!(bp.local[[x1_local, x1_local]], 1.0);
        assert_eq!(bp.local[[x2_local, x2_local]], 1.0);
    }

    #[test]
    fn bounded_uniform_prior_matches_beta_one_one_terms() {
        let theta = 0.7;
        let uniform = bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::Uniform);
        let beta11 =
            bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::Beta { a: 1.0, b: 1.0 });
        assert!((uniform.0 - beta11.0).abs() < 1e-12);
        assert!((uniform.1 - beta11.1).abs() < 1e-12);
        assert!((uniform.2 - beta11.2).abs() < 1e-12);
        assert!((uniform.3 - beta11.3).abs() < 1e-12);
    }

    #[test]
    fn boundednone_prior_has_no_extra_latentobjective_terms() {
        let theta = 0.7;
        let none = bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::None);
        assert_eq!(none, (0.0, 0.0, 0.0, 0.0));

        let uniform = bounded_prior_terms(theta, &BoundedCoefficientPriorSpec::Uniform);
        assert!(uniform.0.is_finite());
        assert!(uniform.0 < 0.0);
        assert!(uniform.1.abs() > 1e-6);
        assert!(uniform.2 > 0.0);
        assert!(uniform.3.is_finite());
    }

    #[test]
    fn exact_bounded_edf_matches_trace_formula_for_simple_penalty() {
        let penalties = vec![Array2::eye(1)];
        let lambdas = array![0.25];
        let cov = array![[2.0]];
        let (edf_by_block, edf_total) =
            exact_bounded_edf(&penalties, &lambdas, &cov).expect("exact bounded edf");
        assert_eq!(edf_by_block.len(), 1);
        assert!((edf_by_block[0] - 0.5).abs() < 1e-12);
        assert!((edf_total - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bounded_joint_hessian_directional_derivative_matches_finite_difference() {
        let x = array![[0.2, -1.0], [0.8, 0.5], [1.1, 1.2], [1.7, -0.3]];
        let y = array![0.4, 1.0, 1.7, 2.2];
        let weights = Array1::ones(y.len());
        let family = BoundedLinearFamily {
            family: LikelihoodFamily::GaussianIdentity,
            mixture_link_state: None,
            sas_link_state: None,
            y: y.clone(),
            weights: weights.clone(),
            design: x.clone(),
            designzeroed: {
                let mut dz = x.clone();
                dz.column_mut(0).fill(0.0);
                dz
            },
            offset: Array1::zeros(y.len()),
            bounded_terms: vec![BoundedLinearTermMeta {
                col_idx: 0,
                min: 0.0,
                max: 1.0,
                prior: BoundedCoefficientPriorSpec::Uniform,
            }],
        };
        let state = vec![ParameterBlockState {
            beta: array![0.4, -0.2],
            eta: Array1::zeros(y.len()),
        }];
        let direction = array![0.3, -0.4];

        let analytic = family
            .exact_newton_joint_hessian_directional_derivative(&state, &direction)
            .expect("analytic derivative")
            .expect("joint derivative");

        let h = 1e-6;
        let plus_state = vec![ParameterBlockState {
            beta: &state[0].beta + &(direction.clone() * h),
            eta: Array1::zeros(y.len()),
        }];
        let minus_state = vec![ParameterBlockState {
            beta: &state[0].beta - &(direction.clone() * h),
            eta: Array1::zeros(y.len()),
        }];
        let plus = family
            .exact_newton_joint_hessian(&plus_state)
            .expect("plus hessian")
            .expect("plus exact hessian");
        let minus = family
            .exact_newton_joint_hessian(&minus_state)
            .expect("minus hessian")
            .expect("minus exact hessian");
        let fd = (plus - minus) / (2.0 * h);

        for i in 0..analytic.nrows() {
            for j in 0..analytic.ncols() {
                assert_eq!(
                    analytic[[i, j]].signum(),
                    fd[[i, j]].signum(),
                    "directional derivative sign mismatch at ({i},{j}): analytic={}, fd={}",
                    analytic[[i, j]],
                    fd[[i, j]]
                );
                assert!(
                    (analytic[[i, j]] - fd[[i, j]]).abs() < 1e-5,
                    "directional derivative mismatch at ({i},{j}): analytic={}, fd={}",
                    analytic[[i, j]],
                    fd[[i, j]]
                );
            }
        }
    }

    #[test]
    fn adaptive_initial_epsilons_use_mean_fallbackwhen_median_is_tiny() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[5e-10, 0.0], [6e-10, 0.0]],
            d1: array![[1e-10, 0.0], [0.0, 1e-10], [2e-10, 0.0], [0.0, 2e-10]],
            d2: array![[3e-10, 0.0], [4e-10, 0.0]],
            collocation_points: array![[0.0, 0.0], [1.0, 1.0]],
            dimension: 2,
        };
        let beta = array![1.0, 1.0];
        let (eps_0, eps_g, eps_c) =
            compute_initial_epsilons(&beta, &[cache], 1e-8).expect("initial epsilons");
        assert!(eps_0 >= 1e-8);
        assert!(eps_g >= 1e-8);
        assert!(eps_c >= 1e-8);
    }

    #[test]
    fn adaptive_exact_psigradient_symmetrizes_nearly_symmetrichessian() {
        let family = SpatialAdaptiveExactFamily {
            family: LikelihoodFamily::GaussianIdentity,
            mixture_link_state: None,
            sas_link_state: None,
            y: Arc::new(array![0.0, 0.0]),
            weights: Arc::new(array![1.0, 1.0]),
            design: Arc::new(array![[1.0, 0.0], [0.0, 1.0]]),
            offset: Arc::new(array![0.0, 0.0]),
            linear_constraints: None,
            runtime_caches: Arc::new(vec![SpatialOperatorRuntimeCache {
                termname: "toy".to_string(),
                feature_cols: vec![0],
                coeff_global_range: 0..2,
                mass_penalty_global_idx: 0,
                tension_penalty_global_idx: 1,
                stiffness_penalty_global_idx: 2,
                d0: array![[1.0, 0.0], [0.0, 1.0]],
                d1: array![[1.0, 0.0], [0.0, 1.0]],
                d2: array![[1.0, 0.0], [0.0, 1.0]],
                collocation_points: array![[0.0], [1.0]],
                dimension: 1,
            }]),
            adaptive_params: vec![SpatialAdaptiveTermHyperParams {
                lambda: [1.0, 1.0, 1.0],
                epsilon: [1.0, 1.0, 1.0],
            }],
            fixed_quadratichessian: Arc::new(array![[0.0, 0.1], [3.0, 0.0]]),
            hyperspecs: Arc::new(build_spatial_adaptive_hyperspecs(1)),
        };
        let spec = ParameterBlockSpec {
            name: "toy".to_string(),
            design: DesignMatrix::Dense(Arc::new(array![[1.0, 0.0], [0.0, 1.0]])),
            offset: array![0.0, 0.0],
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0, 0.0]),
        };
        let deriv = CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi: Array2::zeros((2, 2)),
            s_psi: Array2::zeros((2, 2)),
            s_psi_components: None,
            x_psi_psi: None,
            s_psi_psi: None,
            s_psi_psi_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        };
        let state = vec![ParameterBlockState {
            beta: array![0.0, 0.0],
            eta: array![0.0, 0.0],
        }];

        let gradient = family
            .exact_newton_joint_psi_terms(&state, std::slice::from_ref(&spec), &[vec![deriv]], 0)
            .expect("adaptive joint psi terms should tolerate nearly symmetric Hessian")
            .expect("adaptive joint psi terms should be present")
            .objective_psi;

        assert!(
            gradient.is_finite(),
            "expected finite adaptive joint psi objective term after symmetrization, got {gradient}"
        );
    }

    #[test]
    fn adaptiveweighted_operator_grams_are_symmetric_and_psd() {
        let d1 = array![
            [1.0, 0.0, 2.0],
            [0.5, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.5, 0.0, 0.5],
        ];
        let d2 = array![[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]];
        let weight = array![2.0, 3.0];
        let s1 = weighted_operator_gram_from_d1(&d1, &weight, 2);
        let s2 = weighted_operator_gram_from_d2(&d2, &weight);
        for i in 0..s1.nrows() {
            for j in 0..s1.ncols() {
                assert!((s1[[i, j]] - s1[[j, i]]).abs() < 1e-10);
                assert!((s2[[i, j]] - s2[[j, i]]).abs() < 1e-10);
            }
        }
        let v = array![0.2, -0.3, 0.5];
        let q1 = v.dot(&s1.dot(&v));
        let q2 = v.dot(&s2.dot(&v));
        assert!(q1 >= -1e-10);
        assert!(q2 >= -1e-10);
    }

    #[test]
    fn adaptiveweight_clamp_is_applied_in_u_space() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[0.0, 0.0]],
            d1: array![[0.0, 0.0], [0.0, 0.0]],
            d2: array![[0.0, 0.0]],
            collocation_points: array![[0.0, 0.0]],
            dimension: 2,
        };
        let beta = array![0.0, 0.0];
        let out =
            compute_spatial_adaptiveweights_for_beta(&beta, &[cache], 1e-8, 1e-8, 1e-8, 1e-8, 1e2)
                .expect("adaptive weights");
        assert_eq!(out.len(), 1);
        // Raw u would be 1/eps = 1e8, so clamp to 1e2.
        assert!((out[0].magweight[0] - 1e2).abs() < 1e-12);
        assert!((out[0].gradweight[0] - 1e2).abs() < 1e-12);
        assert!((out[0].lapweight[0] - 1e2).abs() < 1e-12);
        // Diagnostics are 1/u.
        assert!((out[0].inv_magweight[0] - 1e-2).abs() < 1e-12);
        assert!((out[0].invgradweight[0] - 1e-2).abs() < 1e-12);
        assert!((out[0].inv_lapweight[0] - 1e-2).abs() < 1e-12);
    }

    #[test]
    fn adaptiveweight_inverse_consistencywithout_clamp() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[1.0, 0.0], [2.0, 0.0]],
            d1: array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
            d2: array![[1.0, 0.0], [2.0, 0.0]],
            collocation_points: array![[0.0, 0.0], [1.0, 1.0]],
            dimension: 2,
        };
        let beta = array![1.0, 1.0];
        let out = compute_spatial_adaptiveweights_for_beta(
            &beta,
            &[cache],
            1e-6,
            1e-6,
            1e-6,
            1e-12,
            1e12,
        )
        .expect("adaptive weights");
        assert_eq!(out.len(), 1);
        for k in 0..out[0].gradweight.len() {
            let p0 = out[0].magweight[k] * out[0].inv_magweight[k];
            let pg = out[0].gradweight[k] * out[0].invgradweight[k];
            let pc = out[0].lapweight[k] * out[0].inv_lapweight[k];
            assert!((p0 - 1.0).abs() < 1e-10, "mag pair mismatch at {k}: {p0}");
            assert!((pg - 1.0).abs() < 1e-10, "grad pair mismatch at {k}: {pg}");
            assert!((pc - 1.0).abs() < 1e-10, "lap pair mismatch at {k}: {pc}");
        }
    }

    #[test]
    fn adaptiveweight_is_monotone_in_signal_magnitude() {
        let cache = SpatialOperatorRuntimeCache {
            termname: "matern".to_string(),
            feature_cols: vec![0, 1],
            coeff_global_range: 0..2,
            mass_penalty_global_idx: 0,
            tension_penalty_global_idx: 1,
            stiffness_penalty_global_idx: 2,
            d0: array![[1.0, 0.0]],
            d1: array![[1.0, 0.0], [0.0, 1.0]],
            d2: array![[1.0, 0.0]],
            collocation_points: array![[0.0, 0.0]],
            dimension: 2,
        };
        let beta_small = array![0.25, 0.25];
        let beta_large = array![2.0, 2.0];
        let small = compute_spatial_adaptiveweights_for_beta(
            &beta_small,
            std::slice::from_ref(&cache),
            1e-8,
            1e-8,
            1e-8,
            1e-12,
            1e12,
        )
        .expect("small adaptive weights");
        let large = compute_spatial_adaptiveweights_for_beta(
            &beta_large,
            &[cache],
            1e-8,
            1e-8,
            1e-8,
            1e-12,
            1e12,
        )
        .expect("large adaptive weights");
        assert!(small[0].magweight[0] > large[0].magweight[0]);
        assert!(small[0].gradweight[0] > large[0].gradweight[0]);
        assert!(small[0].lapweight[0] > large[0].lapweight[0]);
        assert!(small[0].inv_magweight[0] < large[0].inv_magweight[0]);
        assert!(small[0].invgradweight[0] < large[0].invgradweight[0]);
        assert!(small[0].inv_lapweight[0] < large[0].inv_lapweight[0]);
    }

    #[test]
    fn exact_spatial_adaptive_regularization_fit_runswithout_mm() {
        let n = 48usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.19 * i as f64).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).sin() + 0.25 * x1;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                        length_scale: 0.7,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                mixture_link: None,
                optimize_mixture: false,
                sas_link: None,
                optimize_sas: false,
                compute_inference: true,
                max_iter: 25,
                tol: 1e-5,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: Some(AdaptiveRegularizationOptions {
                    enabled: true,
                    max_mm_iter: 4,
                    beta_rel_tol: 1e-4,
                    max_epsilon_outer_iter: 2,
                    epsilon_log_step: std::f64::consts::LN_2,
                    min_epsilon: 1e-6,
                    weight_floor: 1e-8,
                    weight_ceiling: 1e8,
                }),
                penalty_shrinkage_floor: None,
            },
        )
        .expect("exact adaptive spatial fit should succeed");

        let diag = fit
            .adaptive_diagnostics
            .as_ref()
            .expect("adaptive diagnostics should be present");
        assert_eq!(diag.mm_iterations, 0);
        assert!(diag.epsilon_0.is_finite() && diag.epsilon_0 > 0.0);
        assert!(diag.epsilon_g.is_finite() && diag.epsilon_g > 0.0);
        assert!(diag.epsilon_c.is_finite() && diag.epsilon_c > 0.0);
        assert_eq!(diag.maps.len(), 1);
        assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
        assert!(fit.fit.reml_score.is_finite());
    }

    #[test]
    fn exact_spatial_adaptive_binomial_sas_fit_preserves_link_state() {
        let n = 36usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = -1.0 + 2.0 * (i as f64 / (n as f64 - 1.0));
            let x1 = (0.23 * i as f64).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            let eta = 0.55 * x0 - 0.2 * x1 + 0.1 * x0 * x1;
            let p = 1.0 / (1.0 + (-eta).exp());
            let u = ((i * 37 + 13) % 100) as f64 / 100.0;
            y[i] = if u < p { 1.0 } else { 0.0 };
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                        length_scale: 0.7,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let fit = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodFamily::BinomialSas,
            &FitOptions {
                mixture_link: None,
                optimize_mixture: false,
                sas_link: Some(crate::types::SasLinkSpec {
                    initial_epsilon: 0.1,
                    initial_log_delta: -0.2,
                }),
                optimize_sas: false,
                compute_inference: true,
                max_iter: 15,
                tol: 1e-5,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: Some(AdaptiveRegularizationOptions {
                    enabled: true,
                    max_mm_iter: 4,
                    beta_rel_tol: 1e-4,
                    max_epsilon_outer_iter: 2,
                    epsilon_log_step: std::f64::consts::LN_2,
                    min_epsilon: 1e-6,
                    weight_floor: 1e-8,
                    weight_ceiling: 1e8,
                }),
                penalty_shrinkage_floor: None,
            },
        )
        .expect("exact adaptive SAS fit should succeed");

        match fit.fit.fitted_link {
            FittedLinkState::Sas { state, covariance } => {
                assert!(state.epsilon.is_finite());
                assert!(state.log_delta.is_finite());
                assert!(state.delta.is_finite() && state.delta > 0.0);
                assert!(covariance.is_none());
            }
            other => panic!("expected SAS link parameters, got {other:?}"),
        }
    }

    #[test]
    fn exact_spatial_adaptive_joint_hypergradient_matches_finite_difference() {
        let n = 36usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (0.31 * i as f64).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (4.0 * x0).sin() + 0.35 * x1 + 0.2 * ((x0 - 0.55) * 18.0).tanh();
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                        length_scale: 0.6,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                mixture_link: None,
                optimize_mixture: false,
                sas_link: None,
                optimize_sas: false,
                compute_inference: true,
                max_iter: 30,
                tol: 1e-6,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: None,
                penalty_shrinkage_floor: None,
            },
        )
        .expect("baseline fit");
        let runtime_caches = extract_spatial_operator_runtime_caches(&spec, &baseline.design)
            .expect("runtime caches");
        assert_eq!(runtime_caches.len(), 1);

        let adaptive_opts = AdaptiveRegularizationOptions::default();
        let (eps_0_init, eps_g_init, eps_c_init) = compute_initial_epsilons(
            &baseline.fit.beta,
            &runtime_caches,
            adaptive_opts.min_epsilon,
        )
        .expect("initial epsilons");
        let hyperspecs = build_spatial_adaptive_hyperspecs(runtime_caches.len());
        let derivative_blocks = vec![
            hyperspecs
                .iter()
                .enumerate()
                .map(|(_, _)| CustomFamilyBlockPsiDerivative {
                    penalty_index: None,
                    x_psi: Array2::<f64>::zeros((
                        baseline.design.design.nrows(),
                        baseline.design.design.ncols(),
                    )),
                    s_psi: Array2::<f64>::zeros((
                        baseline.design.design.ncols(),
                        baseline.design.design.ncols(),
                    )),
                    s_psi_components: None,
                    x_psi_psi: None,
                    s_psi_psi: None,
                    s_psi_psi_components: None,
                    implicit_operator: None,
                    implicit_axis: 0,
                    implicit_group_id: None,
                })
                .collect::<Vec<_>>(),
        ];
        let base_family = SpatialAdaptiveExactFamily {
            family: LikelihoodFamily::GaussianIdentity,
            mixture_link_state: None,
            sas_link_state: None,
            y: Arc::new(y.clone()),
            weights: Arc::new(Array1::ones(n)),
            design: Arc::new(baseline.design.design.clone()),
            offset: Arc::new(Array1::zeros(n)),
            linear_constraints: baseline.design.linear_constraints.clone(),
            runtime_caches: Arc::new(runtime_caches.clone()),
            adaptive_params: Vec::new(),
            fixed_quadratichessian: Arc::new(Array2::<f64>::zeros((
                baseline.design.design.ncols(),
                baseline.design.design.ncols(),
            ))),
            hyperspecs: Arc::new(hyperspecs),
        };
        let blockspec = ParameterBlockSpec {
            name: "eta".to_string(),
            design: DesignMatrix::Dense(Arc::new(baseline.design.design.clone())),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(baseline.fit.beta.clone()),
        };
        let outer_opts = BlockwiseFitOptions {
            inner_max_cycles: 30,
            inner_tol: 1e-6,
            outer_max_iter: 30,
            outer_tol: 1e-6,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };

        let evaluate_theta = |theta: &Array1<f64>, need_hessian: bool| {
            let family = base_family.with_adaptive_params(
                vec![SpatialAdaptiveTermHyperParams {
                    lambda: [theta[0].exp(), theta[1].exp(), theta[2].exp()],
                    epsilon: [theta[3].exp(), theta[4].exp(), theta[5].exp()],
                }],
                Arc::new(Array2::<f64>::zeros((
                    baseline.design.design.ncols(),
                    baseline.design.design.ncols(),
                ))),
            );
            evaluate_custom_family_joint_hyper(
                &family,
                std::slice::from_ref(&blockspec),
                &outer_opts,
                &Array1::zeros(0),
                &derivative_blocks,
                None,
                need_hessian,
            )
            .expect("joint hyper eval")
        };

        let theta = array![
            baseline.fit.lambdas[runtime_caches[0].mass_penalty_global_idx]
                .max(1e-6)
                .ln(),
            baseline.fit.lambdas[runtime_caches[0].tension_penalty_global_idx]
                .max(1e-6)
                .ln(),
            baseline.fit.lambdas[runtime_caches[0].stiffness_penalty_global_idx]
                .max(1e-6)
                .ln(),
            eps_0_init.max(1e-6).ln(),
            eps_g_init.max(1e-6).ln(),
            eps_c_init.max(1e-6).ln(),
        ];
        let analytic = evaluate_theta(&theta, true);
        assert_eq!(analytic.gradient.len(), theta.len());
        let analytic_hessian = analytic
            .outer_hessian
            .clone()
            .expect("adaptive joint hyper Hessian should be present");
        let h = 1e-5;
        for j in 0..theta.len() {
            let mut plus = theta.clone();
            plus[j] += h;
            let mut minus = theta.clone();
            minus[j] -= h;
            let fd = (evaluate_theta(&plus, false).objective
                - evaluate_theta(&minus, false).objective)
                / (2.0 * h);
            assert!(
                (analytic.gradient[j] - fd).abs() < 5e-3 * (1.0 + fd.abs()),
                "adaptive joint hypergradient mismatch at {j}: analytic={}, fd={fd}",
                analytic.gradient[j]
            );
            let grad_fd = (evaluate_theta(&plus, false).gradient
                - evaluate_theta(&minus, false).gradient)
                / (2.0 * h);
            for i in 0..theta.len() {
                assert!(
                    (analytic_hessian[[i, j]] - grad_fd[i]).abs() < 5e-2 * (1.0 + grad_fd[i].abs()),
                    "adaptive joint hyper-Hessian mismatch at ({i},{j}): analytic={}, fd={}",
                    analytic_hessian[[i, j]],
                    grad_fd[i]
                );
            }
        }
    }

    #[test]
    fn exact_spatial_adaptive_1dobjective_profile_prefers_tinygradient_lambda() {
        let n = 96usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            y[i] = 0.12 * (2.0 * std::f64::consts::PI * x).sin()
                + 0.05 * (5.0 * std::f64::consts::PI * x).cos()
                + 1.4 / (1.0 + (-(x - 0.5) / 0.012).exp());
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0],
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 31 },
                        length_scale: None,
                        power: 2,
                        nullspace_order: DuchonNullspaceOrder::Zero,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                mixture_link: None,
                optimize_mixture: false,
                sas_link: None,
                optimize_sas: false,
                compute_inference: true,
                max_iter: 20,
                tol: 1e-6,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: None,
                penalty_shrinkage_floor: None,
            },
        )
        .expect("baseline fit");
        let runtime_caches = extract_spatial_operator_runtime_caches(&spec, &baseline.design)
            .expect("runtime caches");
        assert_eq!(runtime_caches.len(), 1);
        let (eps_0, eps_g, eps_c) =
            compute_initial_epsilons(&baseline.fit.beta, &runtime_caches, 1e-8)
                .expect("initial epsilons");
        let hyperspecs = build_spatial_adaptive_hyperspecs(runtime_caches.len());
        let derivative_blocks = vec![
            hyperspecs
                .iter()
                .enumerate()
                .map(|(_, _)| CustomFamilyBlockPsiDerivative {
                    penalty_index: None,
                    x_psi: Array2::<f64>::zeros((
                        baseline.design.design.nrows(),
                        baseline.design.design.ncols(),
                    )),
                    s_psi: Array2::<f64>::zeros((
                        baseline.design.design.ncols(),
                        baseline.design.design.ncols(),
                    )),
                    s_psi_components: None,
                    x_psi_psi: None,
                    s_psi_psi: None,
                    s_psi_psi_components: None,
                    implicit_operator: None,
                    implicit_axis: 0,
                    implicit_group_id: None,
                })
                .collect::<Vec<_>>(),
        ];
        let base_family = SpatialAdaptiveExactFamily {
            family: LikelihoodFamily::GaussianIdentity,
            mixture_link_state: None,
            sas_link_state: None,
            y: Arc::new(y.clone()),
            weights: Arc::new(Array1::ones(n)),
            design: Arc::new(baseline.design.design.clone()),
            offset: Arc::new(Array1::zeros(n)),
            linear_constraints: baseline.design.linear_constraints.clone(),
            runtime_caches: Arc::new(runtime_caches.clone()),
            adaptive_params: Vec::new(),
            fixed_quadratichessian: Arc::new(Array2::<f64>::zeros((
                baseline.design.design.ncols(),
                baseline.design.design.ncols(),
            ))),
            hyperspecs: Arc::new(hyperspecs),
        };
        let blockspec = ParameterBlockSpec {
            name: "eta".to_string(),
            design: DesignMatrix::Dense(Arc::new(baseline.design.design.clone())),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(baseline.fit.beta.clone()),
        };
        let outer_opts = BlockwiseFitOptions {
            inner_max_cycles: 20,
            inner_tol: 1e-6,
            outer_max_iter: 20,
            outer_tol: 1e-6,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        };

        let evaluate_theta = |log_lambda_g: f64| {
            let family = base_family.with_adaptive_params(
                vec![SpatialAdaptiveTermHyperParams {
                    lambda: [1e-12, log_lambda_g.exp(), 1e-12],
                    epsilon: [eps_0, eps_g, eps_c],
                }],
                Arc::new(Array2::<f64>::zeros((
                    baseline.design.design.ncols(),
                    baseline.design.design.ncols(),
                ))),
            );
            evaluate_custom_family_joint_hyper(
                &family,
                std::slice::from_ref(&blockspec),
                &outer_opts,
                &Array1::zeros(0),
                &derivative_blocks,
                None,
                false,
            )
            .expect("joint hyper eval")
        };

        let low = evaluate_theta((1e-8_f64).ln());
        let mid = evaluate_theta((1e-4_f64).ln());
        let high = evaluate_theta((1e-2_f64).ln());

        assert!(
            low.objective < mid.objective && mid.objective < high.objective,
            "expected pseudo-Laplace objective to worsen as gradient lambda increases: low={}, mid={}, high={}",
            low.objective,
            mid.objective,
            high.objective
        );
        assert!(
            low.gradient[1] > 0.0,
            "expected positive gradient wrt log lambda_g near tiny lambda, got {}",
            low.gradient[1]
        );
    }

    #[test]
    fn exact_spatial_adaptive_high_center_duchon_fit_no_longer_fails_in_outer_solver() {
        let n = 320usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = x;
            y[i] = 0.12 * (2.0 * std::f64::consts::PI * x).sin()
                + 0.05 * (5.0 * std::f64::consts::PI * x).cos()
                + 1.4 / (1.0 + (-(x - 0.5) / 0.012).exp());
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0],
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 120 },
                        length_scale: None,
                        power: 2,
                        nullspace_order: DuchonNullspaceOrder::Zero,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let fit = fit_term_collection_forspec(
            data.view(),
            y.view(),
            Array1::ones(n).view(),
            Array1::zeros(n).view(),
            &spec,
            LikelihoodFamily::GaussianIdentity,
            &FitOptions {
                mixture_link: None,
                optimize_mixture: false,
                sas_link: None,
                optimize_sas: false,
                compute_inference: true,
                max_iter: 40,
                tol: 1e-6,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: Some(AdaptiveRegularizationOptions {
                    enabled: true,
                    max_mm_iter: 10,
                    beta_rel_tol: 1e-3,
                    max_epsilon_outer_iter: 4,
                    epsilon_log_step: std::f64::consts::LN_2,
                    min_epsilon: 1e-8,
                    weight_floor: 1e-8,
                    weight_ceiling: 1e8,
                }),
                penalty_shrinkage_floor: None,
            },
        )
        .expect("high-center adaptive Duchon fit should not fail");

        assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
        assert!(fit.fit.deviance.is_finite());
        assert!(fit.fit.edf_total().is_some_and(f64::is_finite));
        let diag = fit
            .adaptive_diagnostics
            .as_ref()
            .expect("adaptive diagnostics should be present");
        assert!(diag.epsilon_0.is_finite() && diag.epsilon_0 > 0.0);
        assert!(diag.epsilon_g.is_finite() && diag.epsilon_g > 0.0);
        assert!(diag.epsilon_c.is_finite() && diag.epsilon_c > 0.0);
    }

    #[test]
    fn binomial_logit_tail_curvature_uses_stable_exact_formula() {
        let eta = array![30.0, 30.0, -30.0, -30.0, 40.0, -40.0];
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let weights = Array1::ones(eta.len());
        let obs = evaluate_standard_familyobservations(
            LikelihoodFamily::BinomialLogit,
            None,
            None,
            &y,
            &weights,
            &eta,
        )
        .expect("stable logit observations");

        for i in 0..eta.len() {
            let mu = stable_sigmoid(eta[i]);
            let target = mu * (1.0 - mu);
            let d_target = target * (1.0 - 2.0 * mu);
            assert!(
                (obs.neghessian_eta[i] - target).abs() <= 1e-12 * (1.0 + target.abs()),
                "eta={} y={} curvature={} target={target}",
                eta[i],
                y[i],
                obs.neghessian_eta[i]
            );
            assert!(
                (obs.neghessian_eta_derivative[i] - d_target).abs()
                    <= 1e-12 * (1.0 + d_target.abs()),
                "eta={} y={} dcurvature={} target={d_target}",
                eta[i],
                y[i],
                obs.neghessian_eta_derivative[i]
            );
            assert!(
                obs.neghessian_eta[i].is_finite()
                    && obs.neghessian_eta_derivative[i].is_finite()
                    && obs.log_likelihood.is_finite(),
                "expected finite logit tail observation state at eta={} y={}",
                eta[i],
                y[i]
            );
        }
    }

    #[test]
    fn non_logit_binomial_tailobservations_stay_finite() {
        let eta = array![12.0, -12.0, 18.0, -18.0];
        let y = array![0.0, 1.0, 1.0, 0.0];
        let weights = Array1::ones(eta.len());
        for family in [
            LikelihoodFamily::BinomialProbit,
            LikelihoodFamily::BinomialCLogLog,
        ] {
            let obs = evaluate_standard_familyobservations(family, None, None, &y, &weights, &eta)
                .expect("tail observations");
            assert!(obs.log_likelihood.is_finite(), "family={family:?}");
            assert!(
                obs.score.iter().all(|v| v.is_finite())
                    && obs.neghessian_eta.iter().all(|v| v.is_finite())
                    && obs.neghessian_eta_derivative.iter().all(|v| v.is_finite()),
                "family={family:?}"
            );
        }
    }

    #[test]
    fn two_block_exact_joint_setup_sanitizes_non_finite_rho_seed() {
        let setup = ExactJointHyperSetup::new(
            array![f64::NEG_INFINITY, 0.25, f64::INFINITY],
            array![-12.0, -12.0, -12.0],
            array![12.0, 12.0, 12.0],
            SpatialLogKappaCoords::new_with_dims(array![0.5], vec![1]),
            SpatialLogKappaCoords::new_with_dims(array![-2.0], vec![1]),
            SpatialLogKappaCoords::new_with_dims(array![2.0], vec![1]),
        );

        let theta0 = setup.theta0();
        assert!(theta0.iter().all(|v| v.is_finite()));
        assert_eq!(theta0[0], 0.0);
        assert_eq!(theta0[1], 0.25);
        assert_eq!(theta0[2], 0.0);
        assert_eq!(theta0[3], 0.5);
    }

    #[test]
    fn extracted_spatial_runtime_cache_matches_normalized_design_penalties() {
        let n = 24usize;
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
            data[[i, 1]] = (0.23 * i as f64).cos();
        }
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 7 },
                        length_scale: 0.8,
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).expect("design");
        let caches =
            extract_spatial_operator_runtime_caches(&spec, &design).expect("runtime caches");
        assert_eq!(caches.len(), 1);
        let cache = &caches[0];
        let s0 = {
            let raw = cache.d0.t().dot(&cache.d0);
            (&raw + &raw.t()) * 0.5
        };
        let s1 = {
            let raw = cache.d1.t().dot(&cache.d1);
            (&raw + &raw.t()) * 0.5
        };
        let s2 = {
            let raw = cache.d2.t().dot(&cache.d2);
            (&raw + &raw.t()) * 0.5
        };

        let s0_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s0,
        );
        let s1_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s1,
        );
        let s2_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s2,
        );

        let p_total = design.design.ncols();
        let err0 = (&s0_global
            - &design.penalties[cache.mass_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err1 = (&s1_global
            - &design.penalties[cache.tension_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err2 = (&s2_global
            - &design.penalties[cache.stiffness_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        assert!(err0 < 1e-8, "mass penalty mismatch too large: {err0}");
        assert!(err1 < 1e-8, "tension penalty mismatch too large: {err1}");
        assert!(err2 < 1e-8, "stiffness penalty mismatch too large: {err2}");
    }

    #[test]
    fn extracted_duchon_spatial_runtime_cache_matches_normalized_design_penalties() {
        let n = 32usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            data[[i, 0]] = i as f64 / (n as f64 - 1.0);
        }
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0],
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 11 },
                        length_scale: Some(0.8),
                        power: 2,
                        nullspace_order: DuchonNullspaceOrder::Zero,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };

        let design = build_term_collection_design(data.view(), &spec).expect("design");
        assert_eq!(design.penalties.len(), 3);
        let caches =
            extract_spatial_operator_runtime_caches(&spec, &design).expect("runtime caches");
        assert_eq!(caches.len(), 1);
        let cache = &caches[0];
        let s0 = {
            let raw = cache.d0.t().dot(&cache.d0);
            (&raw + &raw.t()) * 0.5
        };
        let s1 = {
            let raw = cache.d1.t().dot(&cache.d1);
            (&raw + &raw.t()) * 0.5
        };
        let s2 = {
            let raw = cache.d2.t().dot(&cache.d2);
            (&raw + &raw.t()) * 0.5
        };

        let s0_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s0,
        );
        let s1_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s1,
        );
        let s2_global = penalty_matrixwith_local_block(
            design.design.ncols(),
            cache.coeff_global_range.clone(),
            &s2,
        );

        let p_total = design.design.ncols();
        let err0 = (&s0_global
            - &design.penalties[cache.mass_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err1 = (&s1_global
            - &design.penalties[cache.tension_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let err2 = (&s2_global
            - &design.penalties[cache.stiffness_penalty_global_idx].to_global(p_total))
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        assert!(
            err0 < 1e-8,
            "Duchon mass penalty mismatch too large: {err0}"
        );
        assert!(
            err1 < 1e-8,
            "Duchon tension penalty mismatch too large: {err1}"
        );
        assert!(
            err2 < 1e-8,
            "Duchon stiffness penalty mismatch too large: {err2}"
        );
    }

    #[test]
    fn spatial_adaptive_explicit_second_order_kind_matches_block_sparsity() {
        let alpha_mass_0 = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogLambdaMagnitude,
        };
        let alpha_mass_1 = SpatialAdaptiveHyperSpec {
            cache_index: 1,
            kind: SpatialAdaptiveHyperKind::LogLambdaMagnitude,
        };
        let alpha_grad_0 = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogLambdaGradient,
        };
        let eta_mass = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogEpsilonMagnitude,
        };
        let eta_grad = SpatialAdaptiveHyperSpec {
            cache_index: 0,
            kind: SpatialAdaptiveHyperKind::LogEpsilonGradient,
        };

        assert_eq!(
            alpha_mass_0.explicit_second_order_kind(alpha_mass_0),
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaAlpha
        );
        assert_eq!(
            alpha_mass_0.explicit_second_order_kind(alpha_mass_1),
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero
        );
        assert_eq!(
            alpha_mass_0.explicit_second_order_kind(alpha_grad_0),
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero
        );
        assert_eq!(
            alpha_mass_1.explicit_second_order_kind(eta_mass),
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta
        );
        assert_eq!(
            eta_mass.explicit_second_order_kind(alpha_mass_1),
            SpatialAdaptiveExplicitSecondOrderKind::LocalAlphaEta
        );
        assert_eq!(
            eta_mass.explicit_second_order_kind(eta_mass),
            SpatialAdaptiveExplicitSecondOrderKind::SharedEtaEta
        );
        assert_eq!(
            eta_mass.explicit_second_order_kind(eta_grad),
            SpatialAdaptiveExplicitSecondOrderKind::StructuralZero
        );
    }

    #[test]
    fn scalar_charbonnier_exact_derivatives_match_finite_difference() {
        let signal = array![0.7, -1.1];
        let epsilon = 0.3;
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-5;
        let value = |x: &Array1<f64>| {
            CharbonnierScalarBlockState::from_signal(x.clone(), epsilon).penalty_value()
        };
        for i in 0..signal.len() {
            let mut plus = signal.clone();
            plus[i] += h;
            let mut minus = signal.clone();
            minus[i] -= h;
            let gradfd = (value(&plus) - value(&minus)) / (2.0 * h);
            let hessfd = (value(&plus) - 2.0 * value(&signal) + value(&minus)) / (h * h);
            assert!((state.betagradient_coeff()[i] - gradfd).abs() < 1e-6);
            assert!((state.betahessian_diag()[i] - hessfd).abs() < 1e-4);
        }
    }

    #[test]
    fn grouped_charbonnier_exactgradient_matches_finite_difference() {
        let blocks = array![[0.8, -0.4], [0.3, 0.9]];
        let epsilon = 0.25;
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let h = 1e-6;
        let value = |x: &Array2<f64>| {
            CharbonnierGroupedBlockState::from_signal_blocks(x.clone(), epsilon).penalty_value()
        };
        let analytic = state.betagradient_blocks();
        for k in 0..blocks.nrows() {
            for axis in 0..blocks.ncols() {
                let mut plus = blocks.clone();
                plus[[k, axis]] += h;
                let mut minus = blocks.clone();
                minus[[k, axis]] -= h;
                let gradfd = (value(&plus) - value(&minus)) / (2.0 * h);
                assert!((analytic[[k, axis]] - gradfd).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn scalar_charbonnier_log_epsilon_derivatives_match_finite_difference() {
        let signal = array![0.4, -0.9];
        let epsilon = 0.35_f64;
        let eta = epsilon.ln();
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-5;
        let value = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .penalty_value()
        };
        let gradfd = (value(eta + h) - value(eta - h)) / (2.0 * h);
        let hessfd = (value(eta + h) - 2.0 * value(eta) + value(eta - h)) / (h * h);
        assert!((state.log_epsilon_gradient_terms().sum() - gradfd).abs() < 1e-6);
        assert!((state.log_epsilon_hessian_terms().sum() - hessfd).abs() < 1e-4);

        let eval_grad = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .betagradient_coeff()
        };
        let mixedfd = (&eval_grad(eta + h) - &eval_grad(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((state.log_epsilon_betagradient_coeff()[i] - mixedfd[i]).abs() < 1e-6);
        }

        let eval_hess = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .betahessian_diag()
        };
        let betahessfd = (&eval_hess(eta + h) - &eval_hess(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((state.log_epsilon_betahessian_diag()[i] - betahessfd[i]).abs() < 1e-5);
        }

        let eval_log_grad = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .log_epsilon_betagradient_coeff()
        };
        let second_mixedfd = (&eval_log_grad(eta + h) - &eval_log_grad(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!(
                (state.log_epsilon_beta_mixed_second_coeff()[i] - second_mixedfd[i]).abs() < 1e-5
            );
        }

        let eval_log_hess = |eta_value: f64| {
            CharbonnierScalarBlockState::from_signal(signal.clone(), eta_value.exp())
                .log_epsilon_betahessian_diag()
        };
        let second_hessfd = (&eval_log_hess(eta + h) - &eval_log_hess(eta - h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!(
                (state.log_epsilon_betahessian_second_diag()[i] - second_hessfd[i]).abs() < 1e-4
            );
        }
    }

    #[test]
    fn scalar_charbonnier_directionalhessian_matches_finite_difference() {
        let signal = array![0.5, -0.6];
        let epsilon = 0.2;
        let direction = array![0.3, -0.1];
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-6;
        let analytic = state.directionalhessian_diag(&direction);
        let evalhess = |step: f64| {
            let shifted = &signal + &(direction.mapv(|v| step * v));
            CharbonnierScalarBlockState::from_signal(shifted, epsilon).betahessian_diag()
        };
        let fd = (&evalhess(h) - &evalhess(-h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((analytic[i] - fd[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn scalar_charbonnier_log_epsilon_directionalhessian_matches_finite_difference() {
        let signal = array![0.5, -0.6];
        let epsilon = 0.2;
        let direction = array![0.3, -0.1];
        let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
        let h = 1e-6;
        let analytic = state.log_epsilon_betahessian_directional_diag(&direction);
        let evalhess = |step: f64| {
            let shifted = &signal + &(direction.mapv(|v| step * v));
            CharbonnierScalarBlockState::from_signal(shifted, epsilon)
                .log_epsilon_betahessian_diag()
        };
        let fd = (&evalhess(h) - &evalhess(-h)) / (2.0 * h);
        for i in 0..signal.len() {
            assert!((analytic[i] - fd[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn grouped_charbonnier_log_epsilon_derivatives_match_finite_difference() {
        let blocks = array![[0.7, -0.2], [0.1, 0.8]];
        let epsilon = 0.3_f64;
        let eta = epsilon.ln();
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let h = 1e-5;
        let value = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .penalty_value()
        };
        let gradfd = (value(eta + h) - value(eta - h)) / (2.0 * h);
        let hessfd = (value(eta + h) - 2.0 * value(eta) + value(eta - h)) / (h * h);
        assert!((state.log_epsilon_gradient_terms().sum() - gradfd).abs() < 1e-6);
        assert!((state.log_epsilon_hessian_terms().sum() - hessfd).abs() < 1e-4);

        let eval_grad = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .betagradient_blocks()
        };
        let mixedfd = (&eval_grad(eta + h) - &eval_grad(eta - h)) / (2.0 * h);
        let analytic_mixed = state.log_epsilon_betagradient_blocks();
        for k in 0..blocks.nrows() {
            for axis in 0..blocks.ncols() {
                assert!((analytic_mixed[[k, axis]] - mixedfd[[k, axis]]).abs() < 1e-6);
            }
        }

        let eval_hess = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .betahessian_blocks()
        };
        let plus_hess = eval_hess(eta + h);
        let minus_hess = eval_hess(eta - h);
        let analytic_hess = state.log_epsilon_betahessian_blocks();
        for k in 0..analytic_hess.len() {
            let fd = (&plus_hess[k] - &minus_hess[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic_hess[k][[i, j]] - fd[[i, j]]).abs() < 1e-5);
                }
            }
        }

        let eval_log_grad = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .log_epsilon_betagradient_blocks()
        };
        let second_mixedfd = (&eval_log_grad(eta + h) - &eval_log_grad(eta - h)) / (2.0 * h);
        let analytic_second_mixed = state.log_epsilon_beta_mixed_second_blocks();
        for k in 0..blocks.nrows() {
            for axis in 0..blocks.ncols() {
                assert!(
                    (analytic_second_mixed[[k, axis]] - second_mixedfd[[k, axis]]).abs() < 1e-5
                );
            }
        }

        let eval_log_hess = |eta_value: f64| {
            CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), eta_value.exp())
                .log_epsilon_betahessian_blocks()
        };
        let plus_log_hess = eval_log_hess(eta + h);
        let minus_log_hess = eval_log_hess(eta - h);
        let analytic_second_hess = state.log_epsilon_betahessian_second_blocks();
        for k in 0..analytic_second_hess.len() {
            let fd = (&plus_log_hess[k] - &minus_log_hess[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic_second_hess[k][[i, j]] - fd[[i, j]]).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn grouped_charbonnier_directionalhessian_matches_finite_difference() {
        let blocks = array![[0.6, -0.2], [0.4, 0.5]];
        let direction = array![[0.1, -0.3], [0.2, 0.15]];
        let epsilon = 0.4;
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let analytic = state.directionalhessian_blocks(&direction);
        let h = 1e-6;
        let evalhess = |step: f64| {
            let shifted = &blocks + &(direction.mapv(|v| step * v));
            CharbonnierGroupedBlockState::from_signal_blocks(shifted, epsilon).betahessian_blocks()
        };
        let plus = evalhess(h);
        let minus = evalhess(-h);
        for k in 0..analytic.len() {
            let fd = (&plus[k] - &minus[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic[k][[i, j]] - fd[[i, j]]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn grouped_charbonnier_log_epsilon_directionalhessian_matches_finite_difference() {
        let blocks = array![[0.6, -0.2], [0.4, 0.5]];
        let direction = array![[0.1, -0.3], [0.2, 0.15]];
        let epsilon = 0.4;
        let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
        let analytic = state.log_epsilon_betahessian_directional_blocks(&direction);
        let h = 1e-6;
        let evalhess = |step: f64| {
            let shifted = &blocks + &(direction.mapv(|v| step * v));
            CharbonnierGroupedBlockState::from_signal_blocks(shifted, epsilon)
                .log_epsilon_betahessian_blocks()
        };
        let plus = evalhess(h);
        let minus = evalhess(-h);
        for k in 0..analytic.len() {
            let fd = (&plus[k] - &minus[k]) / (2.0 * h);
            for i in 0..fd.nrows() {
                for j in 0..fd.ncols() {
                    assert!((analytic[k][[i, j]] - fd[[i, j]]).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn grouped_charbonnier_directionalhessian_blocks_are_symmetric() {
        let blocks = array![[0.6, -0.2], [0.4, 0.5]];
        let direction = array![[0.1, -0.3], [0.2, 0.15]];
        let epsilon = 0.4;
        let analytic = CharbonnierGroupedBlockState::from_signal_blocks(blocks, epsilon)
            .directionalhessian_blocks(&direction);
        for (k, block) in analytic.iter().enumerate() {
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    assert!(
                        (block[[i, j]] - block[[j, i]]).abs() < 1e-12,
                        "directional Hessian block {k} is not symmetric at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_charbonnier_local_quadratic_curvature_is_epsilon_invariant() {
        let signal = array![0.0, 0.0, 0.0];
        let small = CharbonnierScalarBlockState::from_signal(signal.clone(), 1e-3);
        let large = CharbonnierScalarBlockState::from_signal(signal, 1e3);
        for (&a, &b) in small
            .betahessian_diag()
            .iter()
            .zip(large.betahessian_diag().iter())
        {
            assert!(
                (a - 1.0).abs() < 1e-10,
                "small-epsilon curvature should be 1, got {a}"
            );
            assert!(
                (b - 1.0).abs() < 1e-10,
                "large-epsilon curvature should be 1, got {b}"
            );
        }
    }

    #[test]
    fn grouped_charbonnier_local_quadratic_curvature_is_epsilon_invariant() {
        let blocks = array![[0.0, 0.0], [0.0, 0.0]];
        let small = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), 1e-3);
        let large = CharbonnierGroupedBlockState::from_signal_blocks(blocks, 1e3);
        for (small_block, large_block) in small
            .betahessian_blocks()
            .into_iter()
            .zip(large.betahessian_blocks().into_iter())
        {
            let eye = Array2::<f64>::eye(small_block.nrows());
            assert!(
                (&small_block - &eye).mapv(f64::abs).sum() < 1e-10,
                "small-epsilon grouped curvature should equal identity"
            );
            assert!(
                (&large_block - &eye).mapv(f64::abs).sum() < 1e-10,
                "large-epsilon grouped curvature should equal identity"
            );
        }
    }

    #[test]
    fn scalar_charbonnier_small_signal_matches_half_quadratic_across_epsilons() {
        let signal = array![1e-5, -2e-5, 3e-5];
        let target = 0.5 * signal.iter().map(|v| v * v).sum::<f64>();
        for &epsilon in &[1e-3, 1e-1, 1.0, 1e2] {
            let state = CharbonnierScalarBlockState::from_signal(signal.clone(), epsilon);
            let value = state.penalty_value();
            let rel = (value - target).abs() / target.max(1e-20);
            assert!(
                rel < 5e-3,
                "scaled scalar Charbonnier should match 0.5*t^2 locally across epsilons: eps={epsilon}, value={value}, target={target}, rel={rel}"
            );
        }
    }

    #[test]
    fn grouped_charbonnier_small_signal_matches_half_quadratic_across_epsilons() {
        let blocks = array![[1e-5, -2e-5], [3e-5, 4e-5]];
        let target = 0.5 * blocks.iter().map(|v| v * v).sum::<f64>();
        for &epsilon in &[1e-3, 1e-1, 1.0, 1e2] {
            let state = CharbonnierGroupedBlockState::from_signal_blocks(blocks.clone(), epsilon);
            let value = state.penalty_value();
            let rel = (value - target).abs() / target.max(1e-20);
            assert!(
                rel < 5e-3,
                "scaled grouped Charbonnier should match 0.5*||v||^2 locally across epsilons: eps={epsilon}, value={value}, target={target}, rel={rel}"
            );
        }
    }

    #[test]
    fn adaptive_diagnostics_json_roundtrip_preserves_shapes() {
        let diag = AdaptiveRegularizationDiagnostics {
            epsilon_0: 0.01,
            epsilon_g: 0.02,
            epsilon_c: 0.03,
            epsilon_outer_iterations: 2,
            mm_iterations: 3,
            converged: true,
            maps: vec![AdaptiveSpatialMap {
                termname: "matern".to_string(),
                feature_cols: vec![0, 1],
                collocation_points: array![[1.0, 2.0], [3.0, 4.0]],
                inv_magweight: array![0.05, 0.15],
                invgradweight: array![0.1, 0.2],
                inv_lapweight: array![0.3, 0.4],
            }],
        };
        let payload = serde_json::to_value(&diag).expect("serialize diagnostics");
        assert_eq!(payload["mm_iterations"].as_u64(), Some(3));
        assert_eq!(
            payload["maps"][0]["collocation_points"]["dim"]
                .as_array()
                .map(|v| v.len()),
            Some(2)
        );
        let decoded: AdaptiveRegularizationDiagnostics =
            serde_json::from_value(payload).expect("deserialize diagnostics");
        assert_eq!(decoded.epsilon_outer_iterations, 2);
        assert_eq!(decoded.mm_iterations, 3);
        assert!(decoded.converged);
        assert_eq!(decoded.maps.len(), 1);
        assert_eq!(decoded.maps[0].collocation_points.nrows(), 2);
        assert_eq!(decoded.maps[0].collocation_points.ncols(), 2);
        assert_eq!(decoded.maps[0].invgradweight.len(), 2);
        assert_eq!(decoded.maps[0].inv_lapweight.len(), 2);
    }
}
