use crate::basis::create_ispline_derivative_dense;
use crate::faer_ndarray::{FaerEigh, fast_ab};
use crate::families::cubic_cell_kernel as exact_kernel;
use crate::pirls::LinearInequalityConstraints;
use crate::span::{breakpoints_from_knots, span_index_for_breakpoints};
use ndarray::{Array1, Array2};

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
        return Err("raw smoothness penalty: span coefficient column dimensions disagree".into());
    }
    let mut penalty = Array2::<f64>::zeros((raw_dim, raw_dim));
    for span_idx in 0..n_spans {
        let left = endpoint_points[span_idx];
        let right = endpoint_points[span_idx + 1];
        let width = right - left;
        if !width.is_finite() || width <= 0.0 {
            return Err(format!(
                "raw smoothness penalty span {span_idx} has invalid width {width}"
            ));
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
        return Err("smoothness penalty matrix must be square for null-space drop".to_string());
    }
    let (eigenvalues, eigenvectors) = raw_penalty
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("raw smoothness penalty eigendecomposition failed: {e}"))?;
    let evals = eigenvalues
        .as_slice()
        .ok_or_else(|| "raw smoothness penalty eigenvalues are not contiguous".to_string())?;
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
            return Err(format!(
                "DeviationRuntime monotonicity span {span_idx} has invalid width {width}"
            ));
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
            return Err(format!(
                "DeviationRuntime monotonicity_eps must be finite and non-negative, got {monotonicity_eps}"
            ));
        }

        let bkpts = breakpoints_from_knots(
            knots
                .as_slice()
                .ok_or_else(|| "DeviationRuntime knots are not contiguous".to_string())?,
            "DeviationRuntime breakpoints",
        )?;
        let endpoint_points = Array1::from_vec(bkpts);
        if endpoint_points.len() < 3 {
            return Err(
                "DeviationRuntime requires at least two active knot spans and one interior node"
                    .to_string(),
            );
        }
        let n_spans = endpoint_points.len() - 1;
        for span_idx in 0..n_spans {
            let left = endpoint_points[span_idx];
            let right = endpoint_points[span_idx + 1];
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                return Err(format!(
                    "DeviationRuntime requires strictly increasing span endpoints at span {span_idx}: left={left}, right={right}"
                ));
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
                .map_err(|e| format!("DeviationRuntime cubic I-spline values failed: {e}"))?;
        let raw_span_c1 =
            create_ispline_derivative_dense(span_lefts.view(), &knots, internal_degree, 1)
                .map_err(|e| {
                    format!("DeviationRuntime cubic I-spline first derivatives failed: {e}")
                })?;
        let raw_span_c2 =
            create_ispline_derivative_dense(span_lefts.view(), &knots, internal_degree, 2)
                .map_err(|e| {
                    format!("DeviationRuntime cubic I-spline second derivatives failed: {e}")
                })?
                .mapv(|value| 0.5 * value);
        let raw_span_c3 =
            create_ispline_derivative_dense(span_midpoints.view(), &knots, internal_degree, 3)
                .map_err(|e| {
                    format!("DeviationRuntime cubic I-spline third derivatives failed: {e}")
                })?
                .mapv(|value| value / 6.0);
        let raw_right_boundary_values =
            create_ispline_derivative_dense(right_endpoint.view(), &knots, internal_degree, 0)
                .map_err(|e| {
                    format!("DeviationRuntime cubic I-spline right boundary failed: {e}")
                })?;
        let raw_right_boundary_value_row = raw_right_boundary_values.row(0).to_owned();

        if max_penalty_derivative_order == 0 {
            return Err(
                "DeviationRuntime requires max_penalty_derivative_order >= 1 so the basis can \
                 drop the corresponding smoothness null space; an order-0 (mass) penalty alone \
                 has no null space and would not require any drop"
                    .to_string(),
            );
        }
        if max_penalty_derivative_order > 3 {
            return Err(format!(
                "DeviationRuntime cubic basis supports derivative orders up to 3; got max \
                 penalty derivative order {max_penalty_derivative_order}"
            ));
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
    // Used by `enforce_cross_block_identifiability_for_flex_block` to enforce
    // the joint-design identifiability invariant
    // `A_train^T (C_train · T) = 0`, where `A_train` is the score-warp
    // design at training points and `C_train` is the link-deviation design
    // at the same points. The choice `T = null(A_train^T C_train)` makes
    // the joint design `[X_loc | X_logslope | A | C·T]` full column rank
    // (up to numerical tolerance), so `σ_min(joint H+S)` is bounded below
    // by the smallest positive eigenvalue of S regardless of how β shifts
    // the linear-predictor distribution. This generalises the per-block
    // null-space drop above to the cross-block aliasing that arises
    // whenever multiple flex blocks parameterise overlapping function
    // classes.
    pub(crate) fn compose_external_column_transform(
        &mut self,
        transform: &Array2<f64>,
    ) -> Result<(), String> {
        let old_dim = self.basis_dim;
        if transform.nrows() != old_dim {
            return Err(format!(
                "DeviationRuntime cross-block transform shape mismatch: \
                 transform rows={}, expected basis_dim={}",
                transform.nrows(),
                old_dim,
            ));
        }
        let new_dim = transform.ncols();
        if new_dim == 0 {
            return Err(
                "DeviationRuntime cross-block transform reduces basis dim to 0; \
                 the candidate's column span is fully aliased by the anchor block"
                    .to_string(),
            );
        }
        if new_dim > old_dim {
            return Err(format!(
                "DeviationRuntime cross-block transform must not increase basis dim; \
                 got new_dim={} from old_dim={}",
                new_dim, old_dim,
            ));
        }
        self.span_c0 = fast_ab(&self.span_c0, transform);
        self.span_c1 = fast_ab(&self.span_c1, transform);
        self.span_c2 = fast_ab(&self.span_c2, transform);
        self.span_c3 = fast_ab(&self.span_c3, transform);
        // `right_boundary_value_row` is a 1-D row vector of length basis_dim;
        // right-multiplying by T (basis_dim × new_dim) gives the new row.
        self.right_boundary_value_row = self.right_boundary_value_row.dot(transform);
        // Monotonicity rows (n_constraints × basis_dim) follow the same
        // right-multiplication. The constraint inequality `A β ≥ ε - 1`
        // becomes `(A · T) β_new ≥ ε - 1` under the reparameterisation
        // β = T β_new, so the row matrix is right-multiplied directly.
        self.monotonicity_constraint_rows =
            fast_ab(&self.monotonicity_constraint_rows, transform);
        self.basis_dim = new_dim;
        Ok(())
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
            return Err(format!(
                "{label} length mismatch: got {}, expected {}",
                beta.len(),
                self.basis_dim
            ));
        }
        Ok(())
    }

    fn evaluate_span_polynomial_design(
        &self,
        values: &Array1<f64>,
        derivative_order: usize,
    ) -> Result<Array2<f64>, String> {
        let (left_ep, right_ep) = self.support_interval()?;
        let mut out = Array2::<f64>::zeros((values.len(), self.basis_dim));
        for (row_idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "deviation runtime design value at row {row_idx} is non-finite ({value})"
                ));
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
                        return Err(format!(
                            "deviation runtime only supports derivative orders up to 4, got {other}"
                        ));
                    }
                };
            }
        }
        Ok(out)
    }

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, 0)
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, 1)
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, 2)
    }

    pub fn third_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, 3)
    }

    pub fn fourth_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, 4)
    }

    pub(crate) fn integrated_derivative_penalty_with_nullity(
        &self,
        derivative_order: usize,
    ) -> Result<(Array2<f64>, usize), String> {
        if derivative_order > self.value_span_degree {
            return Err(format!(
                "deviation penalty derivative order {derivative_order} exceeds value-basis degree {}",
                self.value_span_degree
            ));
        }
        let mut penalty = Array2::<f64>::zeros((self.basis_dim, self.basis_dim));
        for span_idx in 0..self.span_count() {
            let (left, right) = self.span_interval(span_idx)?;
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                return Err(format!(
                    "deviation penalty span {span_idx} has invalid width {width}"
                ));
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
            .map_err(|e| format!("deviation integrated penalty eigendecomposition failed: {e}"))?;
        let threshold = crate::estimate::reml::unified::positive_eigenvalue_threshold(
            evals
                .as_slice()
                .ok_or_else(|| "deviation penalty eigenvalues are not contiguous".to_string())?,
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
            return Err(format!(
                "deviation span index {} out of range for {} spans",
                span_idx,
                self.span_count()
            ));
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
                .ok_or_else(|| "deviation runtime breakpoints are not contiguous".to_string())?,
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
            return Err(format!(
                "deviation span index {} out of range for {} spans",
                span_idx,
                self.span_count()
            ));
        }
        if basis_idx >= self.basis_dim {
            return Err(format!(
                "deviation basis index {} out of range for {} coefficients",
                basis_idx, self.basis_dim
            ));
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
            other => Err(format!(
                "deviation polynomial coefficients only support derivative orders up to 3, got {other}"
            )),
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
            return Err(format!(
                "deviation basis index {} out of range for {} coefficients",
                basis_idx, self.basis_dim
            ));
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
            return Err(format!(
                "deviation basis index {} out of range for {} coefficients",
                basis_idx, self.basis_dim
            ));
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
            _ => Err("deviation runtime is missing monotonicity support points".to_string()),
        }
    }

    pub(crate) fn exact_monotonicity_min_slack(&self, beta: &Array1<f64>) -> Result<f64, String> {
        if beta.len() != self.basis_dim {
            return Err(format!(
                "deviation monotonicity length mismatch: got {}, expected {}",
                beta.len(),
                self.basis_dim
            ));
        }
        if beta.iter().any(|value| !value.is_finite()) {
            let bad = beta
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
                .map(|(idx, value)| format!("deviation coefficient {idx} is non-finite ({value})"))
                .unwrap_or_else(|| "deviation coefficient is non-finite".to_string());
            return Err(bad);
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
            Err("deviation monotonicity slack computation produced no active spans".to_string())
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
            Err(format!(
                "{context} violates exact monotonicity on [{left:.6}, {right:.6}] (minimum derivative slack {slack:.3e}, eps={:.3e})",
                self.monotonicity_eps
            ))
        }
    }

}

