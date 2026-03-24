use crate::families::cubic_cell_kernel as exact_kernel;
use crate::pirls::LinearInequalityConstraints;
use crate::span::{breakpoints_from_knots, span_index_for_breakpoints};
use ndarray::{Array1, Array2};

/// Precomputed per-span polynomial coefficient matrices for a structurally
/// monotone anchored deviation basis.
///
/// Free coefficients are the interior nodal values of the deviation derivative
/// `w'(x)`; the left and right endpoint node values are fixed at `0`, so
/// `w(x)` has constant tails and zero still means the identity map.
///
/// On each knot span `w'(x)` is linearly interpolated between adjacent node
/// values, so `w(x)` is piecewise quadratic. Exact monotonicity of the full
/// transform `x + w(x)` is therefore equivalent to simple lower bounds on the
/// interior nodal coefficients: `beta_j >= monotonicity_eps - 1`.
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
    /// Deviation basis values at the rightmost breakpoint (1 × basis_dim).
    /// Used for constant-tail continuation outside support: the deviation
    /// saturates at this value for all z > right endpoint.
    right_boundary_value_row: Array1<f64>,
}

impl DeviationRuntime {
    pub(crate) fn try_new(knots: Array1<f64>, monotonicity_eps: f64) -> Result<Self, String> {
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
        let basis_dim = endpoint_points.len() - 2;
        let mut span_c0 = Array2::<f64>::zeros((n_spans, basis_dim));
        let mut span_c1 = Array2::<f64>::zeros((n_spans, basis_dim));
        let mut span_c2 = Array2::<f64>::zeros((n_spans, basis_dim));
        let span_c3 = Array2::<f64>::zeros((n_spans, basis_dim));
        let mut right_boundary_value_row = Array1::<f64>::zeros(basis_dim);
        for basis_idx in 0..basis_dim {
            let node_idx = basis_idx + 1;
            let left = endpoint_points[node_idx - 1];
            let center = endpoint_points[node_idx];
            let right = endpoint_points[node_idx + 1];
            let left_width = center - left;
            let right_width = right - center;
            if !left_width.is_finite()
                || !right_width.is_finite()
                || left_width <= 0.0
                || right_width <= 0.0
            {
                return Err(format!(
                    "DeviationRuntime requires strictly increasing interior support around node {node_idx}: left_width={left_width}, right_width={right_width}"
                ));
            }
            let right_tail_value = 0.5 * (left_width + right_width);
            right_boundary_value_row[basis_idx] = right_tail_value;
            for span_idx in 0..n_spans {
                if span_idx + 1 < node_idx {
                    continue;
                }
                if span_idx == node_idx - 1 {
                    span_c2[[span_idx, basis_idx]] = 0.5 / left_width;
                    continue;
                }
                if span_idx == node_idx {
                    span_c0[[span_idx, basis_idx]] = 0.5 * left_width;
                    span_c1[[span_idx, basis_idx]] = 1.0;
                    span_c2[[span_idx, basis_idx]] = -0.5 / right_width;
                    continue;
                }
                span_c0[[span_idx, basis_idx]] = right_tail_value;
            }
        }

        Ok(Self {
            degree: 2,
            value_span_degree: 2,
            basis_dim,
            monotonicity_eps,
            endpoint_points,
            span_c0,
            span_c1,
            span_c2,
            span_c3,
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
            let span_idx = self.span_index_for(value)?;
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

    pub(crate) fn structural_monotonicity_constraints(&self) -> LinearInequalityConstraints {
        let mut a = Array2::<f64>::zeros((self.basis_dim, self.basis_dim));
        for idx in 0..self.basis_dim {
            a[[idx, idx]] = 1.0;
        }
        LinearInequalityConstraints {
            a,
            b: Array1::from_elem(self.basis_dim, self.monotonicity_eps - 1.0),
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
    /// point.  Outside the knot support, returns a constant cubic (c1=c2=c3=0)
    /// at the saturated tail value — the I-spline basis saturates, not
    /// extrapolates, outside its support.
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
    /// point.  Outside the knot support, returns a constant cubic (c1=c2=c3=0)
    /// at the saturated tail value.
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
