use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::cubic_cell_kernel as exact_kernel;
use crate::pirls::LinearInequalityConstraints;
use crate::quadrature::compute_gauss_hermite_n;
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
/// Raw coefficients are quadratic Bernstein control values for the deviation
/// derivative `w'(x)`: interior endpoint controls shared across adjacent spans,
/// plus one midpoint control per span. The left and right endpoint controls
/// are fixed at `0`, so `w(x)` has constant tails and zero still means the
/// identity map. The fitted coefficients live in the configured moment-anchor
/// nullspace and are mapped back to these raw controls for monotonicity.
///
/// On each knot span `w'(x)` is a quadratic Bernstein polynomial, so `w(x)` is
/// truly piecewise cubic.  Exact monotonicity of the full transform `x + w(x)`
/// is guaranteed by lower bounds on every raw derivative control value.
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
    coefficient_transform: Array2<f64>,
    monotonicity_constraint_rows: Array2<f64>,
    /// Deviation basis values at the rightmost breakpoint (1 × basis_dim).
    /// Used for constant-tail continuation outside support: the deviation
    /// saturates at this value for all z > right endpoint.
    right_boundary_value_row: Array1<f64>,
}

enum DeviationMomentAnchor<'a> {
    StandardNormal,
    Empirical(&'a Array1<f64>),
}

fn anchor_coefficient_nullspace(
    endpoint_points: &Array1<f64>,
    raw_span_c0: &Array2<f64>,
    raw_span_c1: &Array2<f64>,
    raw_span_c2: &Array2<f64>,
    raw_span_c3: &Array2<f64>,
    raw_right_boundary_value_row: &Array1<f64>,
    anchor: DeviationMomentAnchor<'_>,
) -> Result<Array2<f64>, String> {
    let raw_dim = raw_span_c0.ncols();
    let mut c = Array2::<f64>::zeros((2, raw_dim));
    match anchor {
        DeviationMomentAnchor::StandardNormal => {
            let rule = compute_gauss_hermite_n(51);
            let inv_sqrt_pi = std::f64::consts::PI.sqrt().recip();
            for (&node, &weight) in rule.nodes.iter().zip(rule.weights.iter()) {
                let z = std::f64::consts::SQRT_2 * node;
                let row = raw_design_row(
                    z,
                    endpoint_points,
                    raw_span_c0,
                    raw_span_c1,
                    raw_span_c2,
                    raw_span_c3,
                    raw_right_boundary_value_row,
                )?;
                let w = weight * inv_sqrt_pi;
                for j in 0..raw_dim {
                    c[[0, j]] += w * row[j];
                    c[[1, j]] += w * z * row[j];
                }
            }
        }
        DeviationMomentAnchor::Empirical(values) => {
            if values.is_empty() {
                return Err(
                    "deviation empirical moment anchor requires at least one value".to_string(),
                );
            }
            let inv_n = 1.0 / values.len() as f64;
            for (idx, &q) in values.iter().enumerate() {
                if !q.is_finite() {
                    return Err(format!(
                        "deviation empirical moment anchor value at row {idx} is non-finite ({q})"
                    ));
                }
                let row = raw_design_row(
                    q,
                    endpoint_points,
                    raw_span_c0,
                    raw_span_c1,
                    raw_span_c2,
                    raw_span_c3,
                    raw_right_boundary_value_row,
                )?;
                for j in 0..raw_dim {
                    c[[0, j]] += inv_n * row[j];
                    c[[1, j]] += inv_n * q * row[j];
                }
            }
        }
    }
    let (z, rank) = rrqr_nullspace_basis(&c.t(), default_rrqr_rank_alpha())
        .map_err(|e| format!("deviation moment anchor RRQR failed: {e}"))?;
    if rank >= raw_dim || z.ncols() == 0 {
        return Err(
            "deviation moment anchor constraints removed all columns; increase basis richness"
                .to_string(),
        );
    }
    Ok(z)
}

fn raw_design_row(
    value: f64,
    endpoint_points: &Array1<f64>,
    raw_span_c0: &Array2<f64>,
    raw_span_c1: &Array2<f64>,
    raw_span_c2: &Array2<f64>,
    raw_span_c3: &Array2<f64>,
    raw_right_boundary_value_row: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    if !value.is_finite() {
        return Err(format!(
            "deviation moment anchor design value is non-finite ({value})"
        ));
    }
    let raw_dim = raw_span_c0.ncols();
    let left_ep = endpoint_points[0];
    let right_ep = endpoint_points[endpoint_points.len() - 1];
    if value < left_ep {
        return Ok(raw_span_c0.row(0).to_owned());
    }
    if value > right_ep {
        return Ok(raw_right_boundary_value_row.to_owned());
    }
    let mut span_idx = span_index_for_breakpoints(
        endpoint_points
            .as_slice()
            .ok_or_else(|| "deviation moment anchor breakpoints are not contiguous".to_string())?,
        value,
        "deviation moment anchor span lookup",
    )?;
    if span_idx > 0 && value == endpoint_points[span_idx] {
        span_idx -= 1;
    }
    let t = value - endpoint_points[span_idx];
    let mut out = Array1::<f64>::zeros(raw_dim);
    for j in 0..raw_dim {
        out[j] = raw_span_c0[[span_idx, j]]
            + raw_span_c1[[span_idx, j]] * t
            + raw_span_c2[[span_idx, j]] * t * t
            + raw_span_c3[[span_idx, j]] * t * t * t;
    }
    Ok(out)
}

pub(crate) fn transform_deviation_penalty(
    penalty: &Array2<f64>,
    transform: &Array2<f64>,
) -> Array2<f64> {
    fast_ab(&fast_atb(transform, penalty), transform)
}

impl DeviationRuntime {
    pub(crate) fn try_new_standard_normal_anchor(
        knots: Array1<f64>,
        monotonicity_eps: f64,
    ) -> Result<Self, String> {
        Self::try_new_with_anchor(
            knots,
            monotonicity_eps,
            DeviationMomentAnchor::StandardNormal,
        )
    }

    pub(crate) fn try_new_empirical_anchor(
        knots: Array1<f64>,
        monotonicity_eps: f64,
        reference: &Array1<f64>,
    ) -> Result<Self, String> {
        Self::try_new_with_anchor(
            knots,
            monotonicity_eps,
            DeviationMomentAnchor::Empirical(reference),
        )
    }

    fn try_new_with_anchor(
        knots: Array1<f64>,
        monotonicity_eps: f64,
        anchor: DeviationMomentAnchor<'_>,
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
        let interior_endpoint_controls = endpoint_points.len() - 2;
        let midpoint_control_offset = interior_endpoint_controls;
        let raw_basis_dim = interior_endpoint_controls + n_spans;
        let mut raw_span_c0 = Array2::<f64>::zeros((n_spans, raw_basis_dim));
        let mut raw_span_c1 = Array2::<f64>::zeros((n_spans, raw_basis_dim));
        let mut raw_span_c2 = Array2::<f64>::zeros((n_spans, raw_basis_dim));
        let mut raw_span_c3 = Array2::<f64>::zeros((n_spans, raw_basis_dim));
        let mut raw_right_boundary_value_row = Array1::<f64>::zeros(raw_basis_dim);

        for span_idx in 0..n_spans {
            let left = endpoint_points[span_idx];
            let right = endpoint_points[span_idx + 1];
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                return Err(format!(
                    "DeviationRuntime requires strictly increasing span endpoints at span {span_idx}: left={left}, right={right}"
                ));
            }
            let inv_width = 1.0 / width;
            let inv_width_sq = inv_width * inv_width;
            let mut active_controls = Vec::with_capacity(3);
            if span_idx > 0 {
                active_controls.push((span_idx - 1, 1.0, -inv_width, inv_width_sq / 3.0));
            }
            active_controls.push((
                midpoint_control_offset + span_idx,
                0.0,
                inv_width,
                -2.0 * inv_width_sq / 3.0,
            ));
            if span_idx + 1 < n_spans {
                active_controls.push((span_idx, 0.0, 0.0, inv_width_sq / 3.0));
            }

            for &(basis_idx, c1, c2, c3) in &active_controls {
                raw_span_c1[[span_idx, basis_idx]] = c1;
                raw_span_c2[[span_idx, basis_idx]] = c2;
                raw_span_c3[[span_idx, basis_idx]] = c3;
            }

            for basis_idx in 0..raw_basis_dim {
                let full_span_integral = (raw_span_c1[[span_idx, basis_idx]] * width
                    + raw_span_c2[[span_idx, basis_idx]] * width * width
                    + raw_span_c3[[span_idx, basis_idx]] * width * width * width)
                    / 1.0;
                let next_value = raw_right_boundary_value_row[basis_idx] + full_span_integral;
                if span_idx + 1 < n_spans {
                    raw_span_c0[[span_idx + 1, basis_idx]] = next_value;
                }
                raw_right_boundary_value_row[basis_idx] = next_value;
            }
        }

        let coefficient_transform = anchor_coefficient_nullspace(
            &endpoint_points,
            &raw_span_c0,
            &raw_span_c1,
            &raw_span_c2,
            &raw_span_c3,
            &raw_right_boundary_value_row,
            anchor,
        )?;
        let basis_dim = coefficient_transform.ncols();
        let span_c0 = fast_ab(&raw_span_c0, &coefficient_transform);
        let span_c1 = fast_ab(&raw_span_c1, &coefficient_transform);
        let span_c2 = fast_ab(&raw_span_c2, &coefficient_transform);
        let span_c3 = fast_ab(&raw_span_c3, &coefficient_transform);
        let right_boundary_value_row = raw_right_boundary_value_row.dot(&coefficient_transform);
        let monotonicity_constraint_rows = coefficient_transform.clone();

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
            coefficient_transform,
            monotonicity_constraint_rows,
            right_boundary_value_row,
        })
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

    pub(crate) fn coefficient_transform(&self) -> &Array2<f64> {
        &self.coefficient_transform
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
            let mut span_idx = self.span_index_for(value)?;
            // Bias to the LEFT-hand span at internal breakpoints so the
            // design / first / second-derivative outputs match
            // local_cubic_on_span(k) evaluated at its right endpoint; the
            // piecewise basis is C¹ but not C² across interior breaks.
            if span_idx > 0 && value == self.endpoint_points[span_idx] {
                span_idx -= 1;
            }
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

    pub(crate) fn integrated_derivative_penalty(
        &self,
        derivative_order: usize,
    ) -> Result<Array2<f64>, String> {
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
        Ok(penalty)
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
    /// (c1=c2=c3=0) at the saturated tail value. Exact support endpoints
    /// still use the adjacent interior span cubic.
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
        let span_idx = self.span_index_for(value)?;
        self.basis_span_cubic(span_idx, basis_idx)
    }

    /// Return the correct composite `LocalSpanCubic` for any evaluation
    /// point. Strictly outside the knot support, returns a constant cubic
    /// (c1=c2=c3=0) at the saturated tail value. Exact support endpoints
    /// still use the adjacent interior span cubic.
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
        let span_idx = self.span_index_for(value)?;
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
