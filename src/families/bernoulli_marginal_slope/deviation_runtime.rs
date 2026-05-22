use crate::basis::create_ispline_derivative_dense;
use crate::faer_ndarray::{FaerEigh, fast_ab};
use crate::families::cubic_cell_kernel as exact_kernel;
use crate::pirls::LinearInequalityConstraints;
use crate::span::{breakpoints_from_knots, span_index_for_breakpoints};
use ndarray::{Array1, Array2, ArrayView2};

/// Typed errors emitted by the deviation runtime construction and evaluation
/// helpers in this module.
///
/// Each variant carries a pre-formatted `reason` string so `Display` is
/// byte-equivalent to the original `format!(...)` outputs the module used
/// before the typed-error migration. The category split lets callers
/// pattern-match on the failure kind without parsing the message.
#[derive(Debug, Clone)]
pub enum DeviationRuntimeError {
    /// A scalar configuration value, index, derivative order, runtime value,
    /// or required metadata bundle did not satisfy the contract (out-of-range
    /// index, non-finite value, missing support points, span width <= 0).
    InvalidInput { reason: String },
    /// A matrix / vector shape did not match an expected dimension while
    /// composing transforms, validating anchors, or accepting beta vectors.
    DimensionMismatch { reason: String },
    /// A numerical kernel (eigendecomposition, I-spline construction,
    /// monotonicity slack search) failed or produced no usable output.
    NumericalFailure { reason: String },
}

impl std::fmt::Display for DeviationRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviationRuntimeError::InvalidInput { reason }
            | DeviationRuntimeError::DimensionMismatch { reason }
            | DeviationRuntimeError::NumericalFailure { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for DeviationRuntimeError {}

impl From<DeviationRuntimeError> for String {
    fn from(err: DeviationRuntimeError) -> String {
        err.to_string()
    }
}

/// Cross-block anchor residual stored on the runtime.
///
/// After cross-block orthogonalisation against parametric anchors, the
/// candidate flex block's design at any row is
///
///   design_row(x) = pure_span_row(x) − n_row · M
///
/// where `n_row` is the per-row stacked vector of parametric anchor values
/// (length `d` = sum of component ncols), and `M` is `residual_coefficients`
/// (shape `d × basis_dim`). The orthonormalising rotation `R` constructed
/// at training time has already been baked into `M = R · K_w · V`, so the
/// runtime's evaluation path only needs `n_row · M` per row.
#[derive(Clone, Debug)]
pub struct AnchorResidual {
    /// d × k matrix; design row subtracts `n_row · residual_coefficients`.
    pub residual_coefficients: Array2<f64>,
    pub null_basis_evaluator: AnchorNullSpaceEvaluator,
}

#[derive(Clone, Debug)]
pub enum AnchorNullSpaceEvaluator {
    Stacked {
        components: Vec<AnchorNullSpaceComponent>,
        /// d × d rotation R such that Q-row(x) = N-row(x) · R. In the
        /// current construction R is baked into `residual_coefficients`,
        /// so this field is the identity matrix and the design evaluator
        /// computes `subtract = n_row · residual_coefficients`.
        orthonormalising_rotation: Array2<f64>,
    },
}

#[derive(Clone, Debug)]
pub enum AnchorNullSpaceComponent {
    /// Parametric anchor — at predict time the parent predictor reconstructs
    /// the per-row vector from the saved marginal/logslope blocks; the
    /// runtime only needs to know which block and how many columns. The
    /// `block` tag is consumed by the serde plumbing in
    /// `inference::model::SavedAnchorComponent`.
    Parametric {
        block: ParametricAnchorBlock,
        ncols: usize,
    },
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum ParametricAnchorBlock {
    Marginal,
    Logslope,
}

fn integrate_polynomial_product(left: &[f64], right: &[f64], width: f64) -> f64 {
    let mut total = 0.0;
    for (left_power, &left_coeff) in left.iter().enumerate() {
        for (right_power, &right_coeff) in right.iter().enumerate() {
            let power = left_power + right_power + 1;
            total += left_coeff * right_coeff * width.powi(power as i32) / power as f64;
        }
    }
    total
}

/// Precomputed per-span polynomial coefficient matrices for a structurally
/// monotone anchored deviation basis.
///
/// Raw coefficients are monotone I-spline coefficients. The deviation
/// derivative `w'(x)` is a nonnegative quadratic B-spline combination, so
/// `w(x)` is a cubic I-spline combination with `C2` continuity at knots and
/// constant tails. Zero coefficients still mean the identity map. The fitted
/// coefficients live in the configured moment-anchor nullspace and are mapped
/// back to these raw coefficients for monotonicity.
///
/// Monotonicity of the full transform `x + w(x)` is enforced by lower bounds
/// on each span's quadratic Bernstein controls for `w'(x)`.
#[derive(Clone, Debug)]
pub struct DeviationRuntime {
    degree: usize,
    value_span_degree: usize,
    basis_dim: usize,
    monotonicity_eps: f64,
    endpoint_points: Array1<f64>,
    span_c0: Array2<f64>,
    span_c1: Array2<f64>,
    span_c2: Array2<f64>,
    span_c3: Array2<f64>,
    monotonicity_constraint_rows: Array2<f64>,
    /// Deviation basis values at the rightmost breakpoint (1 × basis_dim).
    /// Used for constant-tail continuation outside support: the deviation
    /// saturates at this value for all z > right endpoint.
    right_boundary_value_row: Array1<f64>,
    /// Cross-block anchor residualisation state. `None` until
    /// `compose_anchor_orthogonalisation` is called.
    anchor_residual: Option<AnchorResidual>,
    /// Stacked parametric-anchor rows at training rows (n × d). Used by
    /// `design_at_training_with_residual` to rebuild `block.design` after
    /// orthogonalisation. Dropped before serialisation; predict-time
    /// reconstruction rebuilds anchor rows fresh at the predict-time
    /// feature rows.
    anchor_rows_at_training: Option<Array2<f64>>,
}

/// Build the integrated derivative penalty matrix `P` on the *raw* I-spline
/// coefficients (before any null-space transform), where
/// `P_{ij} = ∫ b_i^(k)(x) b_j^(k)(x) dx` integrated piecewise over the knot
/// support. The null space of `P` is the function-space null space of the
/// k-th-derivative penalty: polynomials of degree < k. For k = 1 this is
/// {constants}; for k = 2 it is {constants, linears}; for k = 3 it is
/// {constants, linears, quadratics}. Dropping these directions from the
/// basis at construction time is what gives the link-deviation block
/// β-independent identifiability (the location block's intercept and any
/// unpenalized location-linear absorb constants/linears in η; β_dev contains
/// only the wiggle).
///
/// Mirrors `integrated_derivative_penalty_with_nullity` but operates on the
/// raw cubic span coefficients, so it can be evaluated *before* the basis
/// transform `Z` is constructed (which is what we need to compute `Z`
/// itself).
fn raw_integrated_derivative_penalty(
    endpoint_points: &Array1<f64>,
    raw_span_c0: &Array2<f64>,
    raw_span_c1: &Array2<f64>,
    raw_span_c2: &Array2<f64>,
    raw_span_c3: &Array2<f64>,
    derivative_order: usize,
) -> Result<Array2<f64>, String> {
    let raw_dim = raw_span_c0.ncols();
    let n_spans = endpoint_points.len().saturating_sub(1);
    if raw_span_c1.ncols() != raw_dim
        || raw_span_c2.ncols() != raw_dim
        || raw_span_c3.ncols() != raw_dim
    {
        return Err(DeviationRuntimeError::DimensionMismatch {
            reason: "raw smoothness penalty: span coefficient column dimensions disagree".to_string(),
        }
        .into());
    }
    let mut penalty = Array2::<f64>::zeros((raw_dim, raw_dim));
    for span_idx in 0..n_spans {
        let left = endpoint_points[span_idx];
        let right = endpoint_points[span_idx + 1];
        let width = right - left;
        if !width.is_finite() || width <= 0.0 {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "raw smoothness penalty span {span_idx} has invalid width {width}"
                ),
            }
            .into());
        }
        for i in 0..raw_dim {
            let ci = raw_span_derivative_polynomial_coefficients(
                span_idx,
                i,
                derivative_order,
                raw_span_c0,
                raw_span_c1,
                raw_span_c2,
                raw_span_c3,
            );
            for j in i..raw_dim {
                let cj = raw_span_derivative_polynomial_coefficients(
                    span_idx,
                    j,
                    derivative_order,
                    raw_span_c0,
                    raw_span_c1,
                    raw_span_c2,
                    raw_span_c3,
                );
                let contribution = integrate_polynomial_product(&ci, &cj, width);
                penalty[[i, j]] += contribution;
                if i != j {
                    penalty[[j, i]] += contribution;
                }
            }
        }
    }
    Ok(penalty)
}

/// Per-span polynomial coefficients of the `derivative_order`-th derivative
/// of raw basis function `basis_idx` on its parametric coordinate `t`. Mirrors
/// `DeviationRuntime::span_derivative_polynomial_coefficients` but on raw
/// coefficients so it's callable before `Z` exists.
fn raw_span_derivative_polynomial_coefficients(
    span_idx: usize,
    basis_idx: usize,
    derivative_order: usize,
    raw_span_c0: &Array2<f64>,
    raw_span_c1: &Array2<f64>,
    raw_span_c2: &Array2<f64>,
    raw_span_c3: &Array2<f64>,
) -> Vec<f64> {
    let c0 = raw_span_c0[[span_idx, basis_idx]];
    let c1 = raw_span_c1[[span_idx, basis_idx]];
    let c2 = raw_span_c2[[span_idx, basis_idx]];
    let c3 = raw_span_c3[[span_idx, basis_idx]];
    match derivative_order {
        0 => vec![c0, c1, c2, c3],
        1 => vec![c1, 2.0 * c2, 3.0 * c3],
        2 => vec![2.0 * c2, 6.0 * c3],
        3 => vec![6.0 * c3],
        _ => Vec::new(),
    }
}

/// Compute `Z` = orthonormal columns spanning the orthogonal complement of
/// the null space of `P_raw` (the integrated derivative penalty in raw
/// coordinates). Eigenvectors with strictly-positive eigenvalues are taken;
/// near-zero eigenvalues correspond to functions with zero `derivative_order`-
/// th derivative, i.e., polynomials of degree `< derivative_order` evaluated
/// in the raw basis.
///
/// Returned `Z` has shape `raw_dim × (raw_dim − nullity)`. After applying it
/// (`raw_basis · Z`), the transformed basis cannot represent any polynomial
/// of degree < `derivative_order` — that direction is structurally absent
/// from the parameterization. This is the β-independent identifiability
/// constraint that replaces the data-distribution-dependent moment anchor.
fn smoothness_nullspace_orthogonal_complement(
    raw_penalty: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let n = raw_penalty.nrows();
    if raw_penalty.ncols() != n {
        return Err(DeviationRuntimeError::DimensionMismatch {
            reason: "smoothness penalty matrix must be square for null-space drop".to_string(),
        }
        .into());
    }
    let (eigenvalues, eigenvectors) = raw_penalty
        .eigh(faer::Side::Lower)
        .map_err(|e| {
            String::from(DeviationRuntimeError::NumericalFailure {
                reason: format!("raw smoothness penalty eigendecomposition failed: {e}"),
            })
        })?;
    let evals = eigenvalues
        .as_slice()
        .ok_or_else(|| {
            String::from(DeviationRuntimeError::NumericalFailure {
                reason: "raw smoothness penalty eigenvalues are not contiguous".to_string(),
            })
        })?;
    let threshold = crate::estimate::reml::unified::positive_eigenvalue_threshold(evals);
    let kept: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v > threshold).then_some(i))
        .collect();
    if kept.is_empty() {
        return Err(
            "smoothness penalty has no positive eigenvalues; basis is entirely in the penalty's \
             null space and cannot be identified after the smoothness null-space drop"
                .to_string(),
        );
    }
    if kept.len() == n {
        return Err(
            "smoothness penalty has no null directions; nothing to drop. The link-deviation \
             basis was expected to carry a non-trivial null space (constants/linears) for \
             absorption by the location block — check the configured penalty derivative order"
                .to_string(),
        );
    }
    let mut z = Array2::<f64>::zeros((n, kept.len()));
    for (col_out, &col_in) in kept.iter().enumerate() {
        z.column_mut(col_out).assign(&eigenvectors.column(col_in));
    }
    Ok(z)
}

fn build_quadratic_derivative_bernstein_constraints(
    endpoint_points: &Array1<f64>,
    span_c1: &Array2<f64>,
    span_c2: &Array2<f64>,
    span_c3: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let n_spans = endpoint_points.len().saturating_sub(1);
    let basis_dim = span_c1.ncols();
    let mut rows = Array2::<f64>::zeros((3 * n_spans, basis_dim));
    for span_idx in 0..n_spans {
        let width = endpoint_points[span_idx + 1] - endpoint_points[span_idx];
        if !width.is_finite() || width <= 0.0 {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "DeviationRuntime monotonicity span {span_idx} has invalid width {width}"
                ),
            }
            .into());
        }
        let left_row = 3 * span_idx;
        let mid_row = left_row + 1;
        let right_row = left_row + 2;
        for basis_idx in 0..basis_dim {
            let c1 = span_c1[[span_idx, basis_idx]];
            let c2 = span_c2[[span_idx, basis_idx]];
            let c3 = span_c3[[span_idx, basis_idx]];
            // For w(t)=c0+c1*t+c2*t^2+c3*t^3 on t in [0,h],
            // w'(t)=c1+2*c2*t+3*c3*t^2. In quadratic Bernstein form over
            // s=t/h, the controls are:
            //   b0 = c1
            //   b1 = c1 + c2*h
            //   b2 = c1 + 2*c2*h + 3*c3*h^2
            // Since Bernstein basis functions are non-negative and sum to 1,
            // b_k >= eps-1 is a linear certificate for x + w(x) monotonicity.
            // `exact_monotonicity_min_slack` below still checks the true
            // quadratic minimum, including the interior vertex.
            rows[[left_row, basis_idx]] = c1;
            rows[[mid_row, basis_idx]] = c1 + c2 * width;
            rows[[right_row, basis_idx]] = c1 + 2.0 * c2 * width + 3.0 * c3 * width * width;
        }
    }
    Ok(rows)
}

impl DeviationRuntime {
    /// Construct the link-deviation runtime with a smoothness-null-space-drop
    /// basis transform. `max_penalty_derivative_order` is the highest
    /// derivative order of any penalty that will subsequently be applied to
    /// this block (computed by the caller from its `DeviationBlockConfig`).
    /// The returned basis structurally excludes polynomials of degree
    /// `< max_penalty_derivative_order`, so the configured smoothness
    /// penalties have no null space on the transformed basis and the
    /// joint Hessian + penalty system is well-conditioned at every PIRLS
    /// iteration regardless of how β shifts the linear predictor distribution.
    ///
    /// This replaces the previous data-distribution moment anchor (at the
    /// rigid-pilot η₀), which gave a β-dependent identifiability constraint
    /// that drifted out of alignment with η_current during PIRLS and produced
    /// near-singular joint Hessians (σ_min ≈ ridge_floor).
    pub(crate) fn try_new(
        knots: Array1<f64>,
        monotonicity_eps: f64,
        max_penalty_derivative_order: usize,
    ) -> Result<Self, String> {
        Self::try_new_with_smoothness_drop(knots, monotonicity_eps, max_penalty_derivative_order)
    }

    fn try_new_with_smoothness_drop(
        knots: Array1<f64>,
        monotonicity_eps: f64,
        max_penalty_derivative_order: usize,
    ) -> Result<Self, String> {
        if !monotonicity_eps.is_finite() || monotonicity_eps < 0.0 {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "DeviationRuntime monotonicity_eps must be finite and non-negative, got {monotonicity_eps}"
                ),
            }
            .into());
        }

        let bkpts = breakpoints_from_knots(
            knots
                .as_slice()
                .ok_or_else(|| {
                    String::from(DeviationRuntimeError::InvalidInput {
                        reason: "DeviationRuntime knots are not contiguous".to_string(),
                    })
                })?,
            "DeviationRuntime breakpoints",
        )?;
        let endpoint_points = Array1::from_vec(bkpts);
        if endpoint_points.len() < 3 {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: "DeviationRuntime requires at least two active knot spans and one interior node"
                    .to_string(),
            }
            .into());
        }
        let n_spans = endpoint_points.len() - 1;
        for span_idx in 0..n_spans {
            let left = endpoint_points[span_idx];
            let right = endpoint_points[span_idx + 1];
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                return Err(DeviationRuntimeError::InvalidInput {
                    reason: format!(
                        "DeviationRuntime requires strictly increasing span endpoints at span {span_idx}: left={left}, right={right}"
                    ),
                }
                .into());
            }
        }
        let span_lefts = Array1::from_iter((0..n_spans).map(|idx| endpoint_points[idx]));
        let span_midpoints = Array1::from_iter(
            (0..n_spans).map(|idx| 0.5 * (endpoint_points[idx] + endpoint_points[idx + 1])),
        );
        let right_endpoint = Array1::from_vec(vec![endpoint_points[n_spans]]);
        let internal_degree = 2usize;
        let raw_span_c0 =
            create_ispline_derivative_dense(span_lefts.view(), &knots, internal_degree, 0)
                .map_err(|e| {
                    String::from(DeviationRuntimeError::NumericalFailure {
                        reason: format!("DeviationRuntime cubic I-spline values failed: {e}"),
                    })
                })?;
        let raw_span_c1 =
            create_ispline_derivative_dense(span_lefts.view(), &knots, internal_degree, 1)
                .map_err(|e| {
                    String::from(DeviationRuntimeError::NumericalFailure {
                        reason: format!(
                            "DeviationRuntime cubic I-spline first derivatives failed: {e}"
                        ),
                    })
                })?;
        let raw_span_c2 =
            create_ispline_derivative_dense(span_lefts.view(), &knots, internal_degree, 2)
                .map_err(|e| {
                    String::from(DeviationRuntimeError::NumericalFailure {
                        reason: format!(
                            "DeviationRuntime cubic I-spline second derivatives failed: {e}"
                        ),
                    })
                })?
                .mapv(|value| 0.5 * value);
        let raw_span_c3 =
            create_ispline_derivative_dense(span_midpoints.view(), &knots, internal_degree, 3)
                .map_err(|e| {
                    String::from(DeviationRuntimeError::NumericalFailure {
                        reason: format!(
                            "DeviationRuntime cubic I-spline third derivatives failed: {e}"
                        ),
                    })
                })?
                .mapv(|value| value / 6.0);
        let raw_right_boundary_values =
            create_ispline_derivative_dense(right_endpoint.view(), &knots, internal_degree, 0)
                .map_err(|e| {
                    String::from(DeviationRuntimeError::NumericalFailure {
                        reason: format!(
                            "DeviationRuntime cubic I-spline right boundary failed: {e}"
                        ),
                    })
                })?;
        let raw_right_boundary_value_row = raw_right_boundary_values.row(0).to_owned();

        if max_penalty_derivative_order == 0 {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: "DeviationRuntime requires max_penalty_derivative_order >= 1 so the basis can \
                 drop the corresponding smoothness null space; an order-0 (mass) penalty alone \
                 has no null space and would not require any drop"
                    .to_string(),
            }
            .into());
        }
        if max_penalty_derivative_order > 3 {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "DeviationRuntime cubic basis supports derivative orders up to 3; got max \
                     penalty derivative order {max_penalty_derivative_order}"
                ),
            }
            .into());
        }
        let raw_smoothness_penalty = raw_integrated_derivative_penalty(
            &endpoint_points,
            &raw_span_c0,
            &raw_span_c1,
            &raw_span_c2,
            &raw_span_c3,
            max_penalty_derivative_order,
        )?;
        let coefficient_transform =
            smoothness_nullspace_orthogonal_complement(&raw_smoothness_penalty)?;
        let basis_dim = coefficient_transform.ncols();
        let span_c0 = fast_ab(&raw_span_c0, &coefficient_transform);
        let span_c1 = fast_ab(&raw_span_c1, &coefficient_transform);
        let span_c2 = fast_ab(&raw_span_c2, &coefficient_transform);
        let span_c3 = fast_ab(&raw_span_c3, &coefficient_transform);
        let right_boundary_value_row = raw_right_boundary_value_row.dot(&coefficient_transform);
        let monotonicity_constraint_rows = build_quadratic_derivative_bernstein_constraints(
            &endpoint_points,
            &span_c1,
            &span_c2,
            &span_c3,
        )?;

        Ok(Self {
            degree: 3,
            value_span_degree: 3,
            basis_dim,
            monotonicity_eps,
            endpoint_points,
            span_c0,
            span_c1,
            span_c2,
            span_c3,
            monotonicity_constraint_rows,
            right_boundary_value_row,
            anchor_residual: None,
            anchor_rows_at_training: None,
        })
    }

    // The per-block `smoothness_nullspace_orthogonal_complement` transform
    // above eliminates within-block polynomial aliasing (constants/linears in
    // η_pilot) so the location block can carry the intercept. That handles
    // single-flex-block configurations. When two flex blocks of η_pilot are
    // simultaneously active (score-warp + linkwiggle), each is individually
    // orthogonal to constants, but their column spans still overlap inside
    // the orthogonal complement of constants — both are cubic I-spline bases
    // of the same scalar argument. The overlap manifests as a near-null
    // direction in the joint penalized Hessian: a linear combination of
    // β_score_warp and β_link_dev that produces zero net η-contribution at
    // the rigid-pilot training points yet costs only the (penalised) basis
    // norm, so Newton steps along that direction blow up.
    //
    // Compose an external column transform `T` (shape `basis_dim × new_dim`)
    // into the cubic span tables and monotonicity constraints. After this
    // call every `design(...)`-style method returns matrices in the new
    // `new_dim`-column parameterisation: `runtime.design(values) ==
    // old_runtime.design(values) · T`. Penalties built later via
    // `integrated_derivative_penalty_with_nullity` are also expressed in
    // the new parameterisation.
    //
    // Used by `enforce_cross_block_identifiability_for_flex_block` to
    // enforce the joint-design identifiability invariant in the W-metric
    // (W = p(1−p) at training rows). With `A_train` the stacked parametric
    // anchors and `C_train = span_eval(values)` the candidate basis at the
    // training rows, the residualised candidate is
    //
    //     C̃_train = (I − P_A^{(W)}) C_train,    P_A^{(W)} = A(AᵀWA)⁻¹AᵀW
    //
    // and the kept directions are the eigenvectors of `C̃ᵀ W C̃` above the
    // numerical noise floor. The block-triangular reparameterisation
    // `Aβ_A + Cβ_C = A(β_A + Bβ_C) + (C − AB)β_C` with `B = (AᵀWA)⁻¹AᵀWC`
    // means dropping a direction in C̃ drops *exactly* a direction
    // span(C) shares with span(A) under W, leaving no aliasing in the
    // joint design `[X_loc | X_logslope | A | C·V − N·M]` (full column
    // rank up to numerical tolerance, so `σ_min(joint H+S) ≥ λ_min(S₊)`
    // regardless of how β shifts the linear-predictor distribution).
    //
    // The old `T = null(A_trainᵀ C_train)` algorithm was wrong: that
    // null-space is the candidate directions *already* exactly W-orthogonal
    // to A (Gram = 0), not the directions left after projecting A out.
    // `null(AᵀC) = ∅` does NOT imply `span(C) ⊆ span(A)` — counterexample
    // `A = e₁`, `C = e₁ + e₂` has `AᵀC = 1 ≠ 0` (empty null space) yet
    // `(I − P_A) C = e₂ ≠ 0`. Whenever the anchor is wider than the
    // candidate (d ≥ p_c) the old test generically returned ∅ even when
    // the residualised candidate had full rank, prompting a spurious
    // "fully aliased" hard-error. The current code residualises and keeps
    // exactly the surviving rank.
    /// Compose a rank-reveal right-selector and an optional anchor-residual.
    /// After this call, `design(x)` returns
    ///   design_row(x) = span_eval(x) · V  −  n_row(x) · residual.residual_coefficients
    /// where V is `right_selector` (applied via right-multiplication into
    /// `span_c{0..3}`). Only the `design()` path (derivative_order=0) subtracts
    /// the residual: the anchor argument is a different scalar variable than
    /// the candidate argument, so d/dx of `n_row(x)` w.r.t. the candidate
    /// argument is identically zero.
    pub(crate) fn compose_anchor_orthogonalisation(
        &mut self,
        right_selector: &Array2<f64>,
        residual: Option<AnchorResidual>,
    ) -> Result<(), String> {
        let old_dim = self.basis_dim;
        if right_selector.nrows() != old_dim {
            return Err(DeviationRuntimeError::DimensionMismatch {
                reason: format!(
                    "DeviationRuntime cross-block transform shape mismatch: \
                     transform rows={}, expected basis_dim={}",
                    right_selector.nrows(),
                    old_dim,
                ),
            }
            .into());
        }
        let new_dim = right_selector.ncols();
        if new_dim == 0 {
            return Err(DeviationRuntimeError::DimensionMismatch {
                reason: "DeviationRuntime cross-block transform reduces basis dim to 0; \
                 the candidate's column span is fully aliased by the anchor block"
                    .to_string(),
            }
            .into());
        }
        if new_dim > old_dim {
            return Err(DeviationRuntimeError::DimensionMismatch {
                reason: format!(
                    "DeviationRuntime cross-block transform must not increase basis dim; \
                     got new_dim={} from old_dim={}",
                    new_dim, old_dim,
                ),
            }
            .into());
        }
        if let Some(ref res) = residual {
            let d_expected: usize = match &res.null_basis_evaluator {
                AnchorNullSpaceEvaluator::Stacked {
                    components,
                    orthonormalising_rotation,
                } => {
                    let sum: usize = components
                        .iter()
                        .map(|c| match c {
                            AnchorNullSpaceComponent::Parametric { ncols, .. } => *ncols,
                        })
                        .sum();
                    if orthonormalising_rotation.nrows() != sum
                        || orthonormalising_rotation.ncols() != sum
                    {
                        return Err(DeviationRuntimeError::DimensionMismatch {
                            reason: format!(
                                "DeviationRuntime anchor residual: rotation must be {}×{}, got {}×{}",
                                sum,
                                sum,
                                orthonormalising_rotation.nrows(),
                                orthonormalising_rotation.ncols(),
                            ),
                        }
                        .into());
                    }
                    sum
                }
            };
            if res.residual_coefficients.nrows() != d_expected {
                return Err(DeviationRuntimeError::DimensionMismatch {
                    reason: format!(
                        "DeviationRuntime anchor residual: residual_coefficients rows={}, expected sum-of-component-ncols={}",
                        res.residual_coefficients.nrows(),
                        d_expected,
                    ),
                }
                .into());
            }
            if res.residual_coefficients.ncols() != new_dim {
                return Err(DeviationRuntimeError::DimensionMismatch {
                    reason: format!(
                        "DeviationRuntime anchor residual: residual_coefficients cols={}, expected new basis dim {}",
                        res.residual_coefficients.ncols(),
                        new_dim,
                    ),
                }
                .into());
            }
        }
        self.span_c0 = fast_ab(&self.span_c0, right_selector);
        self.span_c1 = fast_ab(&self.span_c1, right_selector);
        self.span_c2 = fast_ab(&self.span_c2, right_selector);
        self.span_c3 = fast_ab(&self.span_c3, right_selector);
        // `right_boundary_value_row` is a 1-D row vector of length basis_dim;
        // right-multiplying by V (basis_dim × new_dim) gives the new row.
        self.right_boundary_value_row = self.right_boundary_value_row.dot(right_selector);
        // Monotonicity rows (n_constraints × basis_dim) follow the same
        // right-multiplication. The constraint inequality `A β ≥ ε - 1`
        // becomes `(A · V) β_new ≥ ε - 1` under the reparameterisation
        // β = V β_new, so the row matrix is right-multiplied directly.
        self.monotonicity_constraint_rows =
            fast_ab(&self.monotonicity_constraint_rows, right_selector);
        self.basis_dim = new_dim;
        self.anchor_residual = residual;
        Ok(())
    }

    /// Accessor for the anchor-residual state set via
    /// `compose_anchor_orthogonalisation`. Save-time code uses this to
    /// snapshot the residual into the saved model; predict-time code
    /// reconstructs the per-row η correction `n_row · residual_coefficients · β`.
    pub fn anchor_residual(&self) -> Option<&AnchorResidual> {
        self.anchor_residual.as_ref()
    }

    /// Set / replace the cached parametric-anchor rows at training rows.
    /// Stored only at training time; predict-time reconstruction does not
    /// use this cache (it evaluates anchor rows fresh).
    pub(crate) fn set_anchor_rows_at_training(&mut self, rows: Array2<f64>) {
        self.anchor_rows_at_training = Some(rows);
    }

    /// Cached parametric-anchor matrix at training rows, installed by
    /// `enforce_cross_block_identifiability_for_flex_block` when the
    /// runtime is reparameterised against the parametric anchor union.
    /// Used by per-row link-deviation evaluators that need the row's
    /// anchor slice to apply `design_with_anchor_rows` correctly. Returns
    /// `None` for runtimes that have not been reparameterised.
    pub fn anchor_rows_at_training(&self) -> Option<&Array2<f64>> {
        self.anchor_rows_at_training.as_ref()
    }

    /// Evaluate `design(values) - anchor_rows · M` where `anchor_rows` is
    /// the n × d parametric-anchor matrix at the same rows as `values`.
    /// Mandatory when `anchor_residual` is set; for runtimes without a
    /// residual this is equivalent to `design(values)` and the
    /// `anchor_rows` shape must be `n × 0`.
    pub fn design_with_anchor_rows(
        &self,
        values: &Array1<f64>,
        anchor_rows: ArrayView2<f64>,
    ) -> Result<Array2<f64>, String> {
        let mut out = self.evaluate_span_polynomial_design_raw(values, 0)?;
        if let Some(residual) = &self.anchor_residual {
            if anchor_rows.nrows() != values.len() {
                return Err(DeviationRuntimeError::DimensionMismatch {
                    reason: format!(
                        "design_with_anchor_rows: anchor_rows has {} rows, expected {} (matching values)",
                        anchor_rows.nrows(),
                        values.len(),
                    ),
                }
                .into());
            }
            if anchor_rows.ncols() != residual.residual_coefficients.nrows() {
                return Err(DeviationRuntimeError::DimensionMismatch {
                    reason: format!(
                        "design_with_anchor_rows: anchor_rows has {} cols, expected {} (sum of component ncols)",
                        anchor_rows.ncols(),
                        residual.residual_coefficients.nrows(),
                    ),
                }
                .into());
            }
            let subtract = anchor_rows.dot(&residual.residual_coefficients);
            out = out - subtract;
        } else if anchor_rows.ncols() != 0 {
            // Permit empty 0-col anchor rows without complaint; otherwise
            // hard-error so callers don't silently pass mismatched rows.
            return Err(DeviationRuntimeError::DimensionMismatch {
                reason: format!(
                    "design_with_anchor_rows: runtime has no anchor residual but anchor_rows has {} cols",
                    anchor_rows.ncols(),
                ),
            }
            .into());
        }
        Ok(out)
    }

    /// Rebuild the training-row design after orthogonalisation, using
    /// `anchor_rows_at_training` set via `set_anchor_rows_at_training`.
    pub(crate) fn design_at_training_with_residual(
        &self,
        values: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if let Some(rows) = self.anchor_rows_at_training.as_ref() {
            self.design_with_anchor_rows(values, rows.view())
        } else if self.anchor_residual.is_some() {
            Err(
                "design_at_training_with_residual: runtime has anchor_residual but no cached training anchor rows"
                    .to_string(),
            )
        } else {
            self.design(values)
        }
    }

    // ── public field accessors ──

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn value_span_degree(&self) -> usize {
        self.value_span_degree
    }

    pub fn basis_dim(&self) -> usize {
        self.basis_dim
    }

    pub fn monotonicity_eps(&self) -> f64 {
        self.monotonicity_eps
    }

    pub fn span_c0(&self) -> &Array2<f64> {
        &self.span_c0
    }

    pub fn span_c1(&self) -> &Array2<f64> {
        &self.span_c1
    }

    pub fn span_c2(&self) -> &Array2<f64> {
        &self.span_c2
    }

    pub fn span_c3(&self) -> &Array2<f64> {
        &self.span_c3
    }

    // ── design evaluation ──

    fn validate_beta_shape(&self, beta: &Array1<f64>, label: &str) -> Result<(), String> {
        if beta.len() != self.basis_dim {
            return Err(DeviationRuntimeError::DimensionMismatch {
                reason: format!(
                    "{label} length mismatch: got {}, expected {}",
                    beta.len(),
                    self.basis_dim
                ),
            }
            .into());
        }
        Ok(())
    }

    /// Raw cubic-span polynomial design evaluation, without any
    /// anchor-residual subtraction. Internal — callers that need the
    /// residualised design must go through `design()` (which asserts no
    /// residual) or `design_with_anchor_rows()`.
    fn evaluate_span_polynomial_design_raw(
        &self,
        values: &Array1<f64>,
        derivative_order: usize,
    ) -> Result<Array2<f64>, String> {
        let (left_ep, right_ep) = self.support_interval()?;
        let mut out = Array2::<f64>::zeros((values.len(), self.basis_dim));
        for (row_idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DeviationRuntimeError::InvalidInput {
                    reason: format!(
                        "deviation runtime design value at row {row_idx} is non-finite ({value})"
                    ),
                }
                .into());
            }
            if value < left_ep {
                if derivative_order == 0 {
                    out.row_mut(row_idx).assign(&self.span_c0.row(0));
                }
                continue;
            }
            if value > right_ep {
                if derivative_order == 0 {
                    out.row_mut(row_idx)
                        .assign(&self.right_boundary_value_row.view());
                }
                continue;
            }
            let span_idx = self.left_biased_span_index_for(value)?;
            let left = self.endpoint_points[span_idx];
            let t = value - left;
            for basis_idx in 0..self.basis_dim {
                let c0 = self.span_c0[[span_idx, basis_idx]];
                let c1 = self.span_c1[[span_idx, basis_idx]];
                let c2 = self.span_c2[[span_idx, basis_idx]];
                let c3 = self.span_c3[[span_idx, basis_idx]];
                out[[row_idx, basis_idx]] = match derivative_order {
                    0 => c0 + c1 * t + c2 * t * t + c3 * t * t * t,
                    1 => c1 + 2.0 * c2 * t + 3.0 * c3 * t * t,
                    2 => 2.0 * c2 + 6.0 * c3 * t,
                    3 => 6.0 * c3,
                    4 => 0.0,
                    other => {
                        return Err(DeviationRuntimeError::InvalidInput {
                            reason: format!(
                                "deviation runtime only supports derivative orders up to 4, got {other}"
                            ),
                        }
                        .into());
                    }
                };
            }
        }
        Ok(out)
    }

    /// Pure-span design (no anchor-residual subtraction). Callers must
    /// ensure the runtime has no anchor residual; otherwise use
    /// `design_with_anchor_rows`. Derivative paths are unaffected: the
    /// residual subtraction `n_row · M` is constant in the candidate
    /// argument, so its derivatives are identically zero.
    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        debug_assert!(
            self.anchor_residual.is_none(),
            "DeviationRuntime::design called on a runtime with an anchor residual; \
             use design_with_anchor_rows or design_at_training_with_residual instead"
        );
        self.evaluate_span_polynomial_design_raw(values, 0)
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design_raw(values, 1)
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design_raw(values, 2)
    }

    pub fn third_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design_raw(values, 3)
    }

    pub fn fourth_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design_raw(values, 4)
    }

    pub(crate) fn integrated_derivative_penalty_with_nullity(
        &self,
        derivative_order: usize,
    ) -> Result<(Array2<f64>, usize), String> {
        if derivative_order > self.value_span_degree {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "deviation penalty derivative order {derivative_order} exceeds value-basis degree {}",
                    self.value_span_degree
                ),
            }
            .into());
        }
        let mut penalty = Array2::<f64>::zeros((self.basis_dim, self.basis_dim));
        for span_idx in 0..self.span_count() {
            let (left, right) = self.span_interval(span_idx)?;
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                return Err(DeviationRuntimeError::InvalidInput {
                    reason: format!(
                        "deviation penalty span {span_idx} has invalid width {width}"
                    ),
                }
                .into());
            }
            for i in 0..self.basis_dim {
                let ci =
                    self.span_derivative_polynomial_coefficients(span_idx, i, derivative_order)?;
                for j in i..self.basis_dim {
                    let cj = self.span_derivative_polynomial_coefficients(
                        span_idx,
                        j,
                        derivative_order,
                    )?;
                    let contribution = integrate_polynomial_product(&ci, &cj, width);
                    penalty[[i, j]] += contribution;
                    if i != j {
                        penalty[[j, i]] += contribution;
                    }
                }
            }
        }
        let (evals, _) = penalty
            .eigh(faer::Side::Lower)
            .map_err(|e| {
                String::from(DeviationRuntimeError::NumericalFailure {
                    reason: format!("deviation integrated penalty eigendecomposition failed: {e}"),
                })
            })?;
        let threshold = crate::estimate::reml::unified::positive_eigenvalue_threshold(
            evals
                .as_slice()
                .ok_or_else(|| {
                    String::from(DeviationRuntimeError::NumericalFailure {
                        reason: "deviation penalty eigenvalues are not contiguous".to_string(),
                    })
                })?,
        );
        let rank = evals.iter().filter(|&&value| value > threshold).count();
        let nullity = self.basis_dim.saturating_sub(rank);
        Ok((penalty, nullity))
    }

    pub(crate) fn structural_monotonicity_constraints(&self) -> LinearInequalityConstraints {
        LinearInequalityConstraints {
            a: self.monotonicity_constraint_rows.clone(),
            b: Array1::from_elem(
                self.monotonicity_constraint_rows.nrows(),
                self.monotonicity_eps - 1.0,
            ),
        }
    }

    // ── span geometry ──

    fn span_count(&self) -> usize {
        self.endpoint_points.len().saturating_sub(1)
    }

    pub fn breakpoints(&self) -> &Array1<f64> {
        &self.endpoint_points
    }

    fn span_interval(&self, span_idx: usize) -> Result<(f64, f64), String> {
        if span_idx >= self.span_count() {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "deviation span index {} out of range for {} spans",
                    span_idx,
                    self.span_count()
                ),
            }
            .into());
        }
        Ok((
            self.endpoint_points[span_idx],
            self.endpoint_points[span_idx + 1],
        ))
    }

    fn span_index_for(&self, value: f64) -> Result<usize, String> {
        span_index_for_breakpoints(
            self.endpoint_points
                .as_slice()
                .ok_or_else(|| {
                    String::from(DeviationRuntimeError::InvalidInput {
                        reason: "deviation runtime breakpoints are not contiguous".to_string(),
                    })
                })?,
            value,
            "deviation span lookup",
        )
    }

    fn left_biased_span_index_for(&self, value: f64) -> Result<usize, String> {
        let mut span_idx = self.span_index_for(value)?;
        // Bias to the LEFT-hand span at internal breakpoints. The cubic basis
        // is C², so value, first derivative, and second derivative are
        // unchanged; only the span-local third derivative needs a convention.
        if span_idx > 0 && value == self.endpoint_points[span_idx] {
            span_idx -= 1;
        }
        Ok(span_idx)
    }

    fn span_derivative_polynomial_coefficients(
        &self,
        span_idx: usize,
        basis_idx: usize,
        derivative_order: usize,
    ) -> Result<Vec<f64>, String> {
        if span_idx >= self.span_count() {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "deviation span index {} out of range for {} spans",
                    span_idx,
                    self.span_count()
                ),
            }
            .into());
        }
        if basis_idx >= self.basis_dim {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "deviation basis index {} out of range for {} coefficients",
                    basis_idx, self.basis_dim
                ),
            }
            .into());
        }
        let c0 = self.span_c0[[span_idx, basis_idx]];
        let c1 = self.span_c1[[span_idx, basis_idx]];
        let c2 = self.span_c2[[span_idx, basis_idx]];
        let c3 = self.span_c3[[span_idx, basis_idx]];
        match derivative_order {
            0 => Ok(vec![c0, c1, c2, c3]),
            1 => Ok(vec![c1, 2.0 * c2, 3.0 * c3]),
            2 => Ok(vec![2.0 * c2, 6.0 * c3]),
            3 => Ok(vec![6.0 * c3]),
            other => Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "deviation polynomial coefficients only support derivative orders up to 3, got {other}"
                ),
            }
            .into()),
        }
    }

    // ── cubic Taylor extraction ──

    pub(crate) fn local_cubic_on_span(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        self.validate_beta_shape(beta, "deviation local cubic coefficients")?;
        let (left, right) = self.span_interval(span_idx)?;
        Ok(exact_kernel::LocalSpanCubic {
            left,
            right,
            c0: self.span_c0.row(span_idx).dot(beta),
            c1: self.span_c1.row(span_idx).dot(beta),
            c2: self.span_c2.row(span_idx).dot(beta),
            c3: self.span_c3.row(span_idx).dot(beta),
        })
    }

    pub fn basis_span_cubic(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        if basis_idx >= self.basis_dim {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "deviation basis index {} out of range for {} coefficients",
                    basis_idx, self.basis_dim
                ),
            }
            .into());
        }
        let (left, right) = self.span_interval(span_idx)?;
        Ok(exact_kernel::LocalSpanCubic {
            left,
            right,
            c0: self.span_c0[[span_idx, basis_idx]],
            c1: self.span_c1[[span_idx, basis_idx]],
            c2: self.span_c2[[span_idx, basis_idx]],
            c3: self.span_c3[[span_idx, basis_idx]],
        })
    }

    /// Return the correct per-basis `LocalSpanCubic` for any evaluation
    /// point. Strictly outside the knot support, returns a constant cubic
    /// (c1=c2=c3=0) at the saturated tail value. Interior breakpoints use the
    /// left span so span-local third derivatives match derivative designs.
    pub fn basis_cubic_at(
        &self,
        basis_idx: usize,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        if basis_idx >= self.basis_dim {
            return Err(DeviationRuntimeError::InvalidInput {
                reason: format!(
                    "deviation basis index {} out of range for {} coefficients",
                    basis_idx, self.basis_dim
                ),
            }
            .into());
        }
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            return Ok(exact_kernel::LocalSpanCubic {
                left: left_ep,
                right: left_ep + 1.0,
                c0: self.span_c0[[0, basis_idx]],
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        if value > right_ep {
            return Ok(exact_kernel::LocalSpanCubic {
                left: right_ep,
                right: right_ep + 1.0,
                c0: self.right_boundary_value_row[basis_idx],
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        self.basis_span_cubic(span_idx, basis_idx)
    }

    pub fn for_each_basis_cubic_at<F>(&self, value: f64, mut visit: F) -> Result<(), String>
    where
        F: FnMut(usize, exact_kernel::LocalSpanCubic) -> Result<(), String>,
    {
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            for basis_idx in 0..self.basis_dim {
                visit(
                    basis_idx,
                    exact_kernel::LocalSpanCubic {
                        left: left_ep,
                        right: left_ep + 1.0,
                        c0: self.span_c0[[0, basis_idx]],
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    },
                )?;
            }
            return Ok(());
        }
        if value > right_ep {
            for basis_idx in 0..self.basis_dim {
                visit(
                    basis_idx,
                    exact_kernel::LocalSpanCubic {
                        left: right_ep,
                        right: right_ep + 1.0,
                        c0: self.right_boundary_value_row[basis_idx],
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    },
                )?;
            }
            return Ok(());
        }

        let span_idx = self.left_biased_span_index_for(value)?;
        let (left, right) = self.span_interval(span_idx)?;
        for basis_idx in 0..self.basis_dim {
            visit(
                basis_idx,
                exact_kernel::LocalSpanCubic {
                    left,
                    right,
                    c0: self.span_c0[[span_idx, basis_idx]],
                    c1: self.span_c1[[span_idx, basis_idx]],
                    c2: self.span_c2[[span_idx, basis_idx]],
                    c3: self.span_c3[[span_idx, basis_idx]],
                },
            )?;
        }
        Ok(())
    }

    /// Return the correct composite `LocalSpanCubic` for any evaluation
    /// point. Strictly outside the knot support, returns a constant cubic
    /// (c1=c2=c3=0) at the saturated tail value. Interior breakpoints use the
    /// left span so span-local third derivatives match derivative designs.
    pub(crate) fn local_cubic_at(
        &self,
        beta: &Array1<f64>,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        self.validate_beta_shape(beta, "deviation local cubic")?;
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            return Ok(exact_kernel::LocalSpanCubic {
                left: left_ep,
                right: left_ep + 1.0,
                c0: self.left_tail_value(beta),
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        if value > right_ep {
            return Ok(exact_kernel::LocalSpanCubic {
                left: right_ep,
                right: right_ep + 1.0,
                c0: self.right_tail_value(beta),
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            });
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        self.local_cubic_on_span(beta, span_idx)
    }

    // ── tail value helpers ──

    /// Left-tail constant: deviation value at the leftmost breakpoint.
    /// For anchored I-spline bases this is the anchor value (typically 0).
    fn left_tail_value(&self, beta: &Array1<f64>) -> f64 {
        self.span_c0.row(0).dot(beta)
    }

    /// Right-tail constant: deviation value at the rightmost breakpoint.
    /// For I-spline bases this is the saturated integral value.
    fn right_tail_value(&self, beta: &Array1<f64>) -> f64 {
        self.right_boundary_value_row.dot(beta)
    }

    /// Conservative L1 sup-norm bound for the deviation value basis.
    ///
    /// For every evaluation point `x`, this returns a finite `K` such that
    /// `|B(x)·β| <= K * ||β||_∞`.  Each basis column is a cubic on each
    /// finite span and constant in the two tails, so the supremum is attained
    /// at a span endpoint, an interior root of the derivative, or a tail
    /// value.  Summing per-column suprema gives a conservative row-wise L1
    /// bound that is independent of `x`.
    pub(crate) fn value_basis_l1_sup_norm(&self) -> f64 {
        let mut total = 0.0;
        for basis_idx in 0..self.basis_dim {
            let mut col_sup = self.span_c0[[0, basis_idx]]
                .abs()
                .max(self.right_boundary_value_row[basis_idx].abs());
            for span_idx in 0..self.span_count() {
                let left = self.endpoint_points[span_idx];
                let right = self.endpoint_points[span_idx + 1];
                let width = right - left;
                if !width.is_finite() || width <= 0.0 {
                    continue;
                }
                let c0 = self.span_c0[[span_idx, basis_idx]];
                let c1 = self.span_c1[[span_idx, basis_idx]];
                let c2 = self.span_c2[[span_idx, basis_idx]];
                let c3 = self.span_c3[[span_idx, basis_idx]];
                let eval_abs = |t: f64| (c0 + c1 * t + c2 * t * t + c3 * t * t * t).abs();
                col_sup = col_sup.max(eval_abs(0.0)).max(eval_abs(width));
                let a = 3.0 * c3;
                let b = 2.0 * c2;
                let c = c1;
                if a.abs() <= f64::EPSILON {
                    if b.abs() > f64::EPSILON {
                        let t = -c / b;
                        if t > 0.0 && t < width {
                            col_sup = col_sup.max(eval_abs(t));
                        }
                    }
                } else {
                    let disc = b * b - 4.0 * a * c;
                    if disc >= 0.0 {
                        let sqrt_disc = disc.sqrt();
                        for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
                            if t > 0.0 && t < width {
                                col_sup = col_sup.max(eval_abs(t));
                            }
                        }
                    }
                }
            }
            total += col_sup;
        }
        total
    }

    // ── monotonicity enforcement ──

    fn support_interval(&self) -> Result<(f64, f64), String> {
        match (self.endpoint_points.first(), self.endpoint_points.last()) {
            (Some(&left), Some(&right)) => Ok((left, right)),
            _ => Err(DeviationRuntimeError::InvalidInput {
                reason: "deviation runtime is missing monotonicity support points".to_string(),
            }
            .into()),
        }
    }

    pub(crate) fn exact_monotonicity_min_slack(&self, beta: &Array1<f64>) -> Result<f64, String> {
        if beta.len() != self.basis_dim {
            return Err(DeviationRuntimeError::DimensionMismatch {
                reason: format!(
                    "deviation monotonicity length mismatch: got {}, expected {}",
                    beta.len(),
                    self.basis_dim
                ),
            }
            .into());
        }
        if beta.iter().any(|value| !value.is_finite()) {
            let bad = beta
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
                .map(|(idx, value)| format!("deviation coefficient {idx} is non-finite ({value})"))
                .unwrap_or_else(|| "deviation coefficient is non-finite".to_string());
            return Err(DeviationRuntimeError::InvalidInput { reason: bad }.into());
        }

        let mut min_slack = f64::INFINITY;
        for span_idx in 0..self.span_count() {
            let left = self.endpoint_points[span_idx];
            let right = self.endpoint_points[span_idx + 1];
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                continue;
            }
            let c1 = self.span_c1.row(span_idx).dot(beta);
            let c2 = self.span_c2.row(span_idx).dot(beta);
            let c3 = self.span_c3.row(span_idx).dot(beta);
            let d1_left = c1;
            let d1_right = c1 + 2.0 * c2 * width + 3.0 * c3 * width * width;
            let d2_left = 2.0 * c2;
            let d3 = 6.0 * c3;
            let left_slack = 1.0 + d1_left - self.monotonicity_eps;
            let right_slack = 1.0 + d1_right - self.monotonicity_eps;
            min_slack = min_slack.min(left_slack.min(right_slack));

            if d3 > 0.0 {
                let t_star = -d2_left / d3;
                if t_star > 0.0 && t_star < width {
                    let interior = 1.0 + d1_left + d2_left * t_star + 0.5 * d3 * t_star * t_star
                        - self.monotonicity_eps;
                    min_slack = min_slack.min(interior);
                }
            }
        }
        if min_slack.is_finite() {
            Ok(min_slack)
        } else {
            Err(DeviationRuntimeError::NumericalFailure {
                reason: "deviation monotonicity slack computation produced no active spans"
                    .to_string(),
            }
            .into())
        }
    }

    pub(crate) fn monotonicity_feasible(
        &self,
        beta: &Array1<f64>,
        context: &str,
    ) -> Result<(), String> {
        let slack = self.exact_monotonicity_min_slack(beta)?;
        if slack >= -1e-10 {
            Ok(())
        } else {
            let (left, right) = self.support_interval()?;
            Err(DeviationRuntimeError::NumericalFailure {
                reason: format!(
                    "{context} violates exact monotonicity on [{left:.6}, {right:.6}] (minimum derivative slack {slack:.3e}, eps={:.3e})",
                    self.monotonicity_eps
                ),
            }
            .into())
        }
    }
}
