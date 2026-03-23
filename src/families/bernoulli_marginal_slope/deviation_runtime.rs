use crate::basis::BasisOptions;
use crate::families::cubic_cell_kernel as exact_kernel;
use crate::families::gamlss::{
    MonotoneWiggleStructure, monotone_wiggle_basis_with_derivative_order,
};
use crate::linalg::utils::matrix_inversewith_regularization;
use crate::matrix::DesignMatrix;
use crate::span::{breakpoints_from_knots, span_index_for_breakpoints};
use ndarray::{Array1, Array2};

/// Evaluate the monotone wiggle basis (or a derivative) at `values`, then
/// right-multiply by `basis_transform` to map from the raw I-spline columns
/// into the (possibly constrained) column space used at training time.
pub(crate) fn anchored_deviation_basis_with_transform(
    values: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    basis_transform: &Array2<f64>,
    derivative_order: usize,
) -> Result<Array2<f64>, String> {
    let raw = monotone_wiggle_basis_with_derivative_order(values, knots, degree, derivative_order)?;
    if raw.ncols() != basis_transform.nrows() {
        return Err(format!(
            "anchored deviation raw basis mismatch: transform expects {} rows but raw basis has {} columns",
            basis_transform.nrows(),
            raw.ncols()
        ));
    }
    Ok(raw.dot(basis_transform))
}

fn max_abs_matrix_diff(lhs: &Array2<f64>, rhs: &Array2<f64>) -> f64 {
    if lhs.dim() != rhs.dim() {
        return f64::INFINITY;
    }
    let mut max_diff = 0.0_f64;
    for i in 0..lhs.nrows() {
        for j in 0..lhs.ncols() {
            max_diff = max_diff.max((lhs[[i, j]] - rhs[[i, j]]).abs());
        }
    }
    max_diff
}

/// Recover the exact raw-basis-to-trained-basis replay map used for saved
/// anchored deviations. This must reproduce the training design exactly.
pub(crate) fn derive_deviation_basis_transform(
    seed: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    constrained_design: &DesignMatrix,
) -> Result<Array2<f64>, String> {
    let raw_design = monotone_wiggle_basis_with_derivative_order(seed.view(), knots, degree, 0)?;
    let constrained_dense = constrained_design.to_dense();
    if raw_design.nrows() != constrained_dense.nrows() {
        return Err(format!(
            "anchored deviation design row mismatch while deriving replay transform: raw rows={}, constrained rows={}",
            raw_design.nrows(),
            constrained_dense.nrows()
        ));
    }
    if raw_design.ncols() == constrained_dense.ncols()
        && max_abs_matrix_diff(&raw_design, &constrained_dense) <= 1e-12
    {
        return Ok(Array2::<f64>::eye(raw_design.ncols()));
    }

    let gram = raw_design.t().dot(&raw_design);
    let rhs = raw_design.t().dot(&constrained_dense);
    let gram_inv = matrix_inversewith_regularization(&gram, "anchored deviation basis replay")
        .ok_or_else(|| {
            "failed to derive anchored deviation replay transform from training design".to_string()
        })?;
    let basis_transform = gram_inv.dot(&rhs);
    let replayed = raw_design.dot(&basis_transform);
    let replay_error = max_abs_matrix_diff(&replayed, &constrained_dense);
    if !replay_error.is_finite() || replay_error > 1e-8 {
        return Err(format!(
            "anchored deviation replay transform does not exactly reproduce the training design (max abs error {replay_error:.3e})"
        ));
    }
    Ok(basis_transform)
}

/// Precomputed per-span Taylor design matrices for a monotone deviation basis.
///
/// **Structural invariant:** the value basis must have a structurally zero
/// fourth derivative on every knot span (`value_span_degree <= 3`). This
/// allows [`local_cubic_on_span`] and
/// [`basis_span_cubic`] to reconstruct each basis function as an *exact*
/// Taylor polynomial `c₀ + c₁t + c₂t² + c₃t³` — no truncation error.
///
/// The invariant is enforced at construction time in [`DeviationRuntime::try_new`]:
/// both a type-level check ([`MonotoneWiggleStructure::fourth_derivative_is_structurally_zero`])
/// and a numerical verification (max |d⁴/dx⁴| at span midpoints) must pass.
///
/// Because all fields are private to this submodule, `try_new` is the **sole
/// construction path** — no code outside this file can build a
/// `DeviationRuntime` via struct-literal syntax, so the invariant cannot be
/// bypassed.
#[derive(Clone, Debug)]
pub struct DeviationRuntime {
    knots: Array1<f64>,
    degree: usize,
    value_span_degree: usize,
    basis_dim: usize,
    monotonicity_eps: f64,
    basis_transform: Array2<f64>,
    endpoint_points: Array1<f64>,
    span_left_value_design: Array2<f64>,
    span_left_points: Array1<f64>,
    span_right_points: Array1<f64>,
    span_left_d1_design: Array2<f64>,
    span_right_d1_design: Array2<f64>,
    span_left_d2_design: Array2<f64>,
    span_mid_d3_design: Array2<f64>,
}

impl DeviationRuntime {
    /// Construct a `DeviationRuntime`, enforcing the zero-fourth-derivative invariant.
    ///
    /// Derives breakpoints and span intervals from `knots` internally — callers
    /// cannot supply inconsistent span geometry.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `structure.fourth_derivative_is_structurally_zero()` is false
    ///   (type-level rejection based on basis family and degree), OR
    /// - the numerically evaluated fourth-derivative design at span midpoints
    ///   has any entry with `|value| > 1e-10` (belt-and-suspenders), OR
    /// - the knot vector has fewer than two distinct breakpoints.
    pub(crate) fn try_new(
        knots: Array1<f64>,
        degree: usize,
        structure: &MonotoneWiggleStructure,
        basis_dim: usize,
        monotonicity_eps: f64,
        basis_transform: Array2<f64>,
    ) -> Result<Self, String> {
        if degree < 2 {
            return Err(format!(
                "DeviationRuntime requires a monotone wiggle degree >= 2, got {}",
                degree
            ));
        }
        if basis_dim == 0 {
            return Err("DeviationRuntime requires at least one deviation coefficient".to_string());
        }
        if !monotonicity_eps.is_finite() || monotonicity_eps < 0.0 {
            return Err(format!(
                "DeviationRuntime monotonicity_eps must be finite and non-negative, got {monotonicity_eps}"
            ));
        }
        if basis_transform.ncols() != basis_dim {
            return Err(format!(
                "DeviationRuntime basis transform width mismatch: transform has {} columns but basis_dim is {}",
                basis_transform.ncols(),
                basis_dim
            ));
        }
        if basis_transform.nrows() == 0 {
            return Err(
                "DeviationRuntime basis transform must have at least one raw column".to_string(),
            );
        }
        if basis_transform.iter().any(|value| !value.is_finite()) {
            return Err("DeviationRuntime basis transform contains non-finite entries".to_string());
        }

        // ── gate 1: structural / type-level ──
        if !structure.fourth_derivative_is_structurally_zero() {
            return Err(format!(
                "DeviationRuntime requires a value basis whose fourth derivative \
                 is structurally zero on every span, but the monotone wiggle \
                 basis has per-span polynomial degree {} (from public degree {})",
                structure.value_span_degree, degree
            ));
        }

        // ── derive breakpoints and span geometry from knots ──
        let bkpts = breakpoints_from_knots(
            knots
                .as_slice()
                .ok_or_else(|| "DeviationRuntime knots are not contiguous".to_string())?,
            "DeviationRuntime breakpoints",
        )?;
        let mut span_left = Vec::new();
        let mut span_right = Vec::new();
        let mut span_mid = Vec::new();
        for window in bkpts.windows(2) {
            let left = window[0];
            let right = window[1];
            if right - left > 1e-12 {
                span_left.push(left);
                span_right.push(right);
                span_mid.push(0.5 * (left + right));
            }
        }
        if span_left.is_empty() {
            return Err("DeviationRuntime requires at least one active knot span".to_string());
        }
        let endpoint_points = Array1::from_vec(bkpts);
        let span_left_points = Array1::from_vec(span_left);
        let span_right_points = Array1::from_vec(span_right);
        let span_mid_points = Array1::from_vec(span_mid);

        // ── gate 2: numerical fourth-derivative verification ──
        let d4_design = anchored_deviation_basis_with_transform(
            span_mid_points.view(),
            &knots,
            degree,
            &basis_transform,
            4,
        )?;
        let d4_max = d4_design
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        if d4_max > 1e-10 {
            return Err(format!(
                "DeviationRuntime numerical fourth-derivative check failed: \
                 max |d⁴/dx⁴| at span midpoints = {d4_max:.6e} (expected 0)"
            ));
        }

        // ── build design matrices ──
        let span_left_value_design = anchored_deviation_basis_with_transform(
            span_left_points.view(),
            &knots,
            degree,
            &basis_transform,
            0,
        )?;
        let span_left_d1_design = anchored_deviation_basis_with_transform(
            span_left_points.view(),
            &knots,
            degree,
            &basis_transform,
            1,
        )?;
        let span_right_d1_design = anchored_deviation_basis_with_transform(
            span_right_points.view(),
            &knots,
            degree,
            &basis_transform,
            1,
        )?;
        let span_left_d2_design = anchored_deviation_basis_with_transform(
            span_left_points.view(),
            &knots,
            degree,
            &basis_transform,
            2,
        )?;
        let span_mid_d3_design = anchored_deviation_basis_with_transform(
            span_mid_points.view(),
            &knots,
            degree,
            &basis_transform,
            3,
        )?;

        Ok(Self {
            knots,
            degree,
            value_span_degree: structure.value_span_degree,
            basis_dim,
            monotonicity_eps,
            basis_transform,
            endpoint_points,
            span_left_value_design,
            span_left_points,
            span_right_points,
            span_left_d1_design,
            span_right_d1_design,
            span_left_d2_design,
            span_mid_d3_design,
        })
    }

    // ── public field accessors ──

    pub fn knots(&self) -> &Array1<f64> {
        &self.knots
    }

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

    pub fn basis_transform(&self) -> &Array2<f64> {
        &self.basis_transform
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

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        anchored_deviation_basis_with_transform(
            values.view(),
            &self.knots,
            self.degree,
            &self.basis_transform,
            BasisOptions::value().derivative_order,
        )
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        anchored_deviation_basis_with_transform(
            values.view(),
            &self.knots,
            self.degree,
            &self.basis_transform,
            BasisOptions::first_derivative().derivative_order,
        )
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        anchored_deviation_basis_with_transform(
            values.view(),
            &self.knots,
            self.degree,
            &self.basis_transform,
            BasisOptions::second_derivative().derivative_order,
        )
    }

    pub fn third_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        anchored_deviation_basis_with_transform(
            values.view(),
            &self.knots,
            self.degree,
            &self.basis_transform,
            3,
        )
    }

    pub fn fourth_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        anchored_deviation_basis_with_transform(
            values.view(),
            &self.knots,
            self.degree,
            &self.basis_transform,
            4,
        )
    }

    // ── span geometry ──

    fn span_count(&self) -> usize {
        self.span_left_points.len()
    }

    pub(crate) fn breakpoints(&self) -> &Array1<f64> {
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
            self.span_left_points[span_idx],
            self.span_right_points[span_idx],
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

    // ── cubic Taylor extraction ──

    pub(crate) fn local_cubic_on_span(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        self.validate_beta_shape(beta, "deviation local cubic coefficients")?;
        let (left, right) = self.span_interval(span_idx)?;
        let value_design = self.span_left_value_design.row(span_idx);
        let d1_design = self.span_left_d1_design.row(span_idx);
        let d2_design = self.span_left_d2_design.row(span_idx);
        let d3 = self.span_mid_d3_design.row(span_idx).dot(beta);
        Ok(exact_kernel::LocalSpanCubic {
            left,
            right,
            c0: value_design.dot(beta),
            c1: d1_design.dot(beta),
            c2: 0.5 * d2_design.dot(beta),
            c3: d3 / 6.0,
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
            c0: self.span_left_value_design[[span_idx, basis_idx]],
            c1: self.span_left_d1_design[[span_idx, basis_idx]],
            c2: 0.5 * self.span_left_d2_design[[span_idx, basis_idx]],
            c3: self.span_mid_d3_design[[span_idx, basis_idx]] / 6.0,
        })
    }

    pub fn basis_cubic_at(
        &self,
        basis_idx: usize,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        let span_idx = self.span_index_for(value)?;
        self.basis_span_cubic(span_idx, basis_idx)
    }

    pub(crate) fn local_cubic_at(
        &self,
        beta: &Array1<f64>,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        let span_idx = self.span_index_for(value)?;
        self.local_cubic_on_span(beta, span_idx)
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

        let d1_left = self.span_left_d1_design.dot(beta);
        let d1_right = self.span_right_d1_design.dot(beta);
        let d2_left = self.span_left_d2_design.dot(beta);
        let d3_mid = self.span_mid_d3_design.dot(beta);

        let mut min_slack = f64::INFINITY;
        for span_idx in 0..self.span_left_points.len() {
            let left = self.span_left_points[span_idx];
            let right = self.span_right_points[span_idx];
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                continue;
            }
            let left_slack = 1.0 + d1_left[span_idx] - self.monotonicity_eps;
            let right_slack = 1.0 + d1_right[span_idx] - self.monotonicity_eps;
            min_slack = min_slack.min(left_slack.min(right_slack));

            let curvature = d3_mid[span_idx];
            if curvature > 0.0 {
                let t_star = -d2_left[span_idx] / curvature;
                if t_star > 0.0 && t_star < width {
                    let interior = 1.0
                        + d1_left[span_idx]
                        + d2_left[span_idx] * t_star
                        + 0.5 * curvature * t_star * t_star
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

    pub(crate) fn max_feasible_monotone_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if beta.len() != self.basis_dim || delta.len() != self.basis_dim {
            return Err(format!(
                "deviation monotone step length mismatch: beta={}, delta={}, expected {}",
                beta.len(),
                delta.len(),
                self.basis_dim
            ));
        }
        for (idx, step) in delta.iter().enumerate() {
            if !step.is_finite() {
                return Err(format!("deviation step direction {idx} is non-finite"));
            }
        }
        self.monotonicity_feasible(beta, "deviation monotone coefficients")?;

        let mut trial = beta.clone();
        for idx in 0..trial.len() {
            trial[idx] += delta[idx];
        }
        if self.exact_monotonicity_min_slack(&trial)? >= 0.0 {
            return Ok(Some(1.0));
        }

        let mut alpha_lo = 0.0f64;
        let mut alpha_hi = 1.0f64;
        for _ in 0..48 {
            let alpha_mid = 0.5 * (alpha_lo + alpha_hi);
            for idx in 0..trial.len() {
                trial[idx] = beta[idx] + alpha_mid * delta[idx];
            }
            if self.exact_monotonicity_min_slack(&trial)? >= 0.0 {
                alpha_lo = alpha_mid;
            } else {
                alpha_hi = alpha_mid;
            }
        }
        if alpha_lo >= 1.0 {
            return Ok(Some(1.0));
        }
        let conservative = if alpha_lo <= 0.0 {
            0.0
        } else {
            (alpha_lo * 0.999_999).clamp(0.0, 1.0)
        };
        Ok(Some(conservative))
    }
}
