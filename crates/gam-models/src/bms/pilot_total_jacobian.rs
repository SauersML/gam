//! Calibrated pilot of the Bernoulli marginal-slope row predictor and its total
//! η-Jacobian, the geometry the flex-block identifiability audit judges
//! (gam#3320).
//!
//! # The row predictor
//!
//! With probit scale `s`, pre-scale slope `b_i` and observed slope `B_i = s·b_i`,
//! the row predictor is `η_i = A_i + B_i·z_i` plus the flex contributions, and
//! the observed intercept `A_i` solves the calibration
//!
//!   `E_π[Φ(A_i + B_i·u)] = Φ(q_i)`,     `q_i = q(η_m,i)` the marginal index,
//!
//! over the latent measure `π` (standard normal, or the row's empirical grid).
//! The score warp enters as `B_i·h(·)` at `z_i` and at every latent node `u`;
//! the link deviation enters as `s·w(t)` at the pre-scale argument
//! `t = A_i/s + b_i·(·)`.
//!
//! # The total Jacobian at the rigid-flex pilot (h = w = 0)
//!
//! Differentiating the calibration, every coefficient that moves the predictor
//! at a latent node also moves `A_i`:
//!
//!   `∂A_i/∂θ = −E_ν[∂_θ(direct contribution at u)] + (marginal term)`,
//!
//! where `ν_i(u) ∝ π(u)·φ(A_i + B_i·u)` is the calibration tilt. Hence
//!
//! * marginal: `∂η_i/∂β_m = g_i·q'_i·x_m,i`, `g_i = φ(q_i)/E_π φ(A_i + B_i·u)`;
//! * slope: `∂η_i/∂β_s = s·(z_i − E_ν u)·x_s,i`;
//! * score warp: `∂η_i/∂θ_h = B_i·(H(z_i) − E_ν H(u))`;
//! * link deviation: `∂η_i/∂θ_w = s·(C(t_i) − E_ν C(A_i/s + b_i·u))`,
//!
//! with `H`, `C` the flex bases. Under the standard normal law the tilt is
//! Gaussian, `ν_i = N(−A_i·B_i/P_i, 1/P_i)` with `P_i = 1 + B_i²`, and
//! `g_i = √P_i`; each basis expectation is then exact in closed form, because
//! the runtime basis is a piecewise cubic with constant extensions past its
//! support. Under an empirical law the tilt is the reweighted grid.
//!
//! The Fisher information of the likelihood is `JᵀWJ` with `J` this total
//! Jacobian and `W` the Bernoulli-probit Fisher weight at the pilot η, so the
//! flex audit judges its candidates on `J` in the metric `W`.

use super::gradient_paths::pilot_irls_hessian_row_metric_at_eta;
use super::*;
use crate::latent_anchor::{AnchorGridOwned, solve_anchor};
use gam_linalg::faer_ndarray::FaerEigh;
use std::borrow::Cow;

/// The BMS row predictor calibrated at a flex-free pilot: the rigid baseline
/// moved by one Fisher step on the marginal block.
pub(crate) struct BmsCalibratedPilot<'a> {
    latent_measure: &'a LatentMeasureKind,
    probit_scale: f64,
    z: &'a Array1<f64>,
    observed_intercept: Array1<f64>,
    observed_slope: Array1<f64>,
    eta: Array1<f64>,
    link_argument: Array1<f64>,
    /// `∂A_i/∂η_m,i = g_i·q'_i`.
    marginal_scale: Array1<f64>,
}

/// The training rows and rigid baseline a [`BmsCalibratedPilot`] is built from.
pub(crate) struct BmsPilotInputs<'a> {
    pub(crate) latent_measure: &'a LatentMeasureKind,
    pub(crate) base_link: &'a InverseLink,
    pub(crate) y: &'a Array1<f64>,
    pub(crate) z: &'a Array1<f64>,
    pub(crate) weights: &'a Array1<f64>,
    pub(crate) marginal_design: &'a DesignMatrix,
    pub(crate) marginal_offset: &'a Array1<f64>,
    pub(crate) slope_offset: &'a Array1<f64>,
    /// Rigid marginal predictor, added to each row's marginal offset.
    pub(crate) baseline_marginal: f64,
    /// Rigid pre-scale slope, added to each row's slope offset.
    pub(crate) baseline_slope: f64,
    /// Probit scale `s`.
    pub(crate) probit_scale: f64,
}

/// The calibration tilt `ν_i ∝ π(u)·φ(A_i + B_i·u)` of one row.
enum RowTilt<'g> {
    Gaussian {
        mean: f64,
        sd: f64,
    },
    Discrete {
        grid: Cow<'g, EmpiricalZGrid>,
        tilt: Vec<f64>,
    },
}

impl RowTilt<'_> {
    fn mean_node(&self) -> f64 {
        match self {
            Self::Gaussian { mean, .. } => *mean,
            Self::Discrete { grid, tilt } => {
                grid.nodes.iter().zip(tilt).map(|(u, w)| u * w).sum()
            }
        }
    }
}

impl<'a> BmsCalibratedPilot<'a> {
    /// Calibrate the rigid baseline, take one Fisher step on the marginal
    /// block through its total Jacobian, and recalibrate there.
    ///
    /// The step is the minimum-norm solution of `JₘᵀWJₘ·δ = JₘᵀW·r` on the
    /// Gram's resolved positive eigenspace: a direction the pilot data do not
    /// identify takes no step. It moves the pilot off the rigid point, where
    /// every row shares one `(A, B)` up to its offsets and the link argument
    /// is affine in `z`, onto the covariate structure of the marginal block.
    pub(crate) fn build(inputs: BmsPilotInputs<'a>) -> Result<Self, String> {
        let BmsPilotInputs {
            latent_measure,
            base_link,
            y,
            z,
            weights,
            marginal_design,
            marginal_offset,
            slope_offset,
            baseline_marginal,
            baseline_slope,
            probit_scale,
        } = inputs;
        let n = z.len();
        for (name, len) in [
            ("y", y.len()),
            ("weights", weights.len()),
            ("marginal design", marginal_design.nrows()),
            ("marginal offset", marginal_offset.len()),
            ("slope offset", slope_offset.len()),
        ] {
            if len != n {
                return Err(format!(
                    "BMS calibrated pilot: {name} has {len} rows, z has {n}"
                ));
            }
        }
        let observed_slope = slope_offset.mapv(|offset| probit_scale * (baseline_slope + offset));
        let rigid_marginal = marginal_offset.mapv(|offset| baseline_marginal + offset);
        let rigid = Self::calibrated(
            latent_measure,
            base_link,
            probit_scale,
            z,
            &rigid_marginal,
            observed_slope.clone(),
        )?;
        let marginal_dense = marginal_design.try_to_dense_arc("BMS calibrated pilot marginal design")?;
        let marginal_dense = marginal_dense.as_ref();
        if marginal_dense.ncols() == 0 {
            return Ok(rigid);
        }
        let delta = rigid.marginal_fisher_step(y, weights, marginal_dense)?;
        let stepped_marginal = &rigid_marginal + &marginal_dense.dot(&delta);
        Self::calibrated(
            latent_measure,
            base_link,
            probit_scale,
            z,
            &stepped_marginal,
            observed_slope,
        )
    }

    /// Calibrate every row at marginal predictor `marginal_eta` and observed
    /// slope `observed_slope`.
    fn calibrated(
        latent_measure: &'a LatentMeasureKind,
        base_link: &InverseLink,
        probit_scale: f64,
        z: &'a Array1<f64>,
        marginal_eta: &Array1<f64>,
        observed_slope: Array1<f64>,
    ) -> Result<Self, String> {
        if !(probit_scale.is_finite() && probit_scale > 0.0) {
            return Err(format!(
                "BMS calibrated pilot: probit scale {probit_scale} is not finite and positive"
            ));
        }
        let n = z.len();
        if marginal_eta.len() != n || observed_slope.len() != n {
            return Err(format!(
                "BMS calibrated pilot: marginal predictor has {} rows and slope {}, z has {n}",
                marginal_eta.len(),
                observed_slope.len(),
            ));
        }
        let rows = (0..n)
            .into_par_iter()
            .map(|row| {
                let link = bernoulli_marginal_link_map(base_link, marginal_eta[row])?;
                let (intercept, density_ratio) =
                    calibrate_row(latent_measure, row, link.q, observed_slope[row])?;
                Ok((intercept, density_ratio * link.q1))
            })
            .collect::<Result<Vec<(f64, f64)>, String>>()?;
        let observed_intercept = Array1::from_iter(rows.iter().map(|row| row.0));
        let marginal_scale = Array1::from_iter(rows.iter().map(|row| row.1));
        let eta = Array1::from_shape_fn(n, |i| observed_intercept[i] + observed_slope[i] * z[i]);
        if let Some(row) = eta
            .iter()
            .zip(marginal_scale.iter())
            .position(|(e, g)| !e.is_finite() || !g.is_finite())
        {
            return Err(format!(
                "BMS calibrated pilot: row {row} has non-finite η={} or ∂A/∂η_m={}",
                eta[row], marginal_scale[row],
            ));
        }
        let link_argument = eta.mapv(|value| value / probit_scale);
        Ok(Self {
            latent_measure,
            probit_scale,
            z,
            observed_intercept,
            observed_slope,
            eta,
            link_argument,
            marginal_scale,
        })
    }

    /// One Fisher step `δ` on the marginal coefficients at this pilot.
    fn marginal_fisher_step(
        &self,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        marginal_dense: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = marginal_dense.nrows();
        let p = marginal_dense.ncols();
        let mut jacobian = marginal_dense.clone();
        let mut weighted_jacobian = marginal_dense.clone();
        let mut score = Array1::<f64>::zeros(n);
        for i in 0..n {
            let eta = self.eta[i];
            // `V = Φ(η)·Φ(−η)` and `y − μ = y·Φ(−η) − (1 − y)·Φ(η)` are formed
            // from both tails without cancellation. A row whose density or
            // variance has underflowed carries no curvature and no score.
            let phi = normal_pdf(eta);
            let (mu, mu_complement) = (normal_cdf(eta), normal_cdf(-eta));
            let var = mu * mu_complement;
            let (fisher_weight, row_score) = if phi > 0.0 && var > 0.0 {
                (
                    weights[i] * phi * phi / var,
                    weights[i] * phi * (y[i] * mu_complement - (1.0 - y[i]) * mu) / var,
                )
            } else {
                (0.0, 0.0)
            };
            let scale = self.marginal_scale[i];
            jacobian.row_mut(i).mapv_inplace(|v| v * scale);
            weighted_jacobian
                .row_mut(i)
                .mapv_inplace(|v| v * scale * fisher_weight);
            score[i] = row_score * scale;
        }
        let gram = jacobian.t().dot(&weighted_jacobian);
        let rhs = marginal_dense.t().dot(&score);
        let (evals, evecs) = FaerEigh::eigh(&gram, faer::Side::Lower)
            .map_err(|e| format!("BMS calibrated pilot Fisher-step eigendecomposition failed: {e}"))?;
        let threshold = gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(
            evals
                .as_slice()
                .ok_or_else(|| "BMS calibrated pilot: eigenvalues are not contiguous".to_string())?,
        );
        let projected_rhs = evecs.t().dot(&rhs);
        let mut delta = Array1::<f64>::zeros(p);
        for k in 0..p {
            if evals[k] > threshold {
                delta.scaled_add(projected_rhs[k] / evals[k], &evecs.column(k));
            }
        }
        Ok(delta)
    }

    /// Bernoulli-probit Fisher row weight at the pilot η.
    pub(crate) fn fisher_row_metric(&self, weights: &Array1<f64>) -> Array1<f64> {
        pilot_irls_hessian_row_metric_at_eta(&self.eta, weights)
    }

    /// The link deviation's pre-scale argument `t_i = η_i/s` at the training rows.
    pub(crate) fn link_argument(&self) -> &Array1<f64> {
        &self.link_argument
    }

    /// `∂η/∂β_m`: the marginal design scaled by `g_i·q'_i`.
    pub(crate) fn marginal_jacobian(&self, design: &DesignMatrix) -> Result<Array2<f64>, String> {
        scaled_design_rows(design, &self.marginal_scale)
    }

    /// `∂η/∂β_s`: the slope design scaled by `s·(z_i − E_ν u)`.
    pub(crate) fn slope_jacobian(&self, design: &DesignMatrix) -> Result<Array2<f64>, String> {
        let scale = (0..self.z.len())
            .into_par_iter()
            .map(|row| Ok(self.probit_scale * (self.z[row] - self.row_tilt(row)?.mean_node())))
            .collect::<Result<Vec<f64>, String>>()?;
        scaled_design_rows(design, &Array1::from(scale))
    }

    /// `∂η/∂θ_h = B_i·(H(z_i) − E_ν H(u))` for a score-warp runtime.
    pub(crate) fn score_warp_jacobian(
        &self,
        runtime: &DeviationRuntime,
    ) -> Result<Array2<f64>, String> {
        require_plain_runtime(runtime, "score-warp")?;
        let at_rows = runtime.design(self.z)?;
        let global_node_design = match self.latent_measure {
            LatentMeasureKind::GlobalEmpirical { grid } => {
                Some(runtime.design(&Array1::from(grid.nodes.clone()))?)
            }
            _ => None,
        };
        let expected = self.row_expectations(runtime.basis_dim(), |_, tilt| match tilt {
            RowTilt::Gaussian { mean, sd } => gaussian_runtime_expectation(runtime, mean, sd),
            RowTilt::Discrete { grid, tilt } => {
                let tilt = ArrayView1::from(tilt.as_slice());
                match global_node_design.as_ref() {
                    Some(node_design) => Ok(node_design.t().dot(&tilt)),
                    None => Ok(runtime
                        .design(&Array1::from(grid.nodes.clone()))?
                        .t()
                        .dot(&tilt)),
                }
            }
        })?;
        let mut out = at_rows - expected;
        for (mut row, &slope) in out.rows_mut().into_iter().zip(self.observed_slope.iter()) {
            row.mapv_inplace(|v| v * slope);
        }
        Ok(out)
    }

    /// `∂η/∂θ_w = s·(C(t_i) − E_ν C(A_i/s + b_i·u))` for a link-deviation runtime.
    pub(crate) fn link_deviation_jacobian(
        &self,
        runtime: &DeviationRuntime,
    ) -> Result<Array2<f64>, String> {
        require_plain_runtime(runtime, "link-deviation")?;
        let s = self.probit_scale;
        let at_rows = runtime.design(&self.link_argument)?;
        let expected = self.row_expectations(runtime.basis_dim(), |row, tilt| {
            let intercept = self.observed_intercept[row] / s;
            let slope = self.observed_slope[row] / s;
            match tilt {
                RowTilt::Gaussian { mean, sd } => gaussian_runtime_expectation(
                    runtime,
                    intercept + slope * mean,
                    slope.abs() * sd,
                ),
                RowTilt::Discrete { grid, tilt } => {
                    let arguments = Array1::from_iter(grid.nodes.iter().map(|u| intercept + slope * u));
                    Ok(runtime
                        .design(&arguments)?
                        .t()
                        .dot(&ArrayView1::from(tilt.as_slice())))
                }
            }
        })?;
        Ok((at_rows - expected) * s)
    }

    fn row_tilt(&self, row: usize) -> Result<RowTilt<'a>, String> {
        let intercept = self.observed_intercept[row];
        let slope = self.observed_slope[row];
        let latent_measure: &'a LatentMeasureKind = self.latent_measure;
        match latent_measure.empirical_grid_for_training_row(row)? {
            None => {
                let precision = 1.0 + slope * slope;
                Ok(RowTilt::Gaussian {
                    mean: -intercept * slope / precision,
                    sd: precision.sqrt().recip(),
                })
            }
            Some(grid) => {
                let (tilt, _) = discrete_tilt(&grid, intercept, slope);
                Ok(RowTilt::Discrete { grid, tilt })
            }
        }
    }

    /// Stack `expectation(row, ν_row)` (each of length `p`) over the rows.
    fn row_expectations<F>(&self, p: usize, expectation: F) -> Result<Array2<f64>, String>
    where
        F: Fn(usize, RowTilt<'a>) -> Result<Array1<f64>, String> + Sync,
    {
        let n = self.z.len();
        let rows = (0..n)
            .into_par_iter()
            .map(|row| {
                let value = expectation(row, self.row_tilt(row)?)?;
                if value.len() != p {
                    return Err(format!(
                        "BMS calibrated pilot: row {row} expectation has length {}, basis has {p}",
                        value.len(),
                    ));
                }
                Ok(value)
            })
            .collect::<Result<Vec<Array1<f64>>, String>>()?;
        let mut out = Array2::<f64>::zeros((n, p));
        for (mut target, value) in out.rows_mut().into_iter().zip(rows.iter()) {
            target.assign(value);
        }
        Ok(out)
    }
}

/// Observed intercept `A` of one row and `g = φ(q)/E_π φ(A + B·u)`.
fn calibrate_row(
    latent_measure: &LatentMeasureKind,
    row: usize,
    q: f64,
    observed_slope: f64,
) -> Result<(f64, f64), String> {
    match latent_measure.empirical_grid_for_training_row(row)? {
        None => {
            let scale = (1.0 + observed_slope * observed_slope).sqrt();
            Ok((q * scale, scale))
        }
        Some(grid) => {
            let intercept =
                solve_anchor(q, observed_slope, AnchorGridOwned::from_grid(&grid).view())?;
            // `E_π φ(A + B·u) = exp(LSE)/√(2π)` and `φ(q) = exp(−q²/2)/√(2π)`.
            let (_, log_normaliser) = discrete_tilt(&grid, intercept, observed_slope);
            Ok((intercept, (-0.5 * q * q - log_normaliser).exp()))
        }
    }
}

/// Normalised tilt `ω_k ∝ π_k·φ(A + B·u_k)` on a grid and the log of
/// `Σ_k π_k·exp(−(A + B·u_k)²/2)`.
fn discrete_tilt(grid: &EmpiricalZGrid, intercept: f64, observed_slope: f64) -> (Vec<f64>, f64) {
    let log_terms: Vec<f64> = grid
        .pairs()
        .map(|(u, weight)| {
            let x = intercept + observed_slope * u;
            weight.ln() - 0.5 * x * x
        })
        .collect();
    let peak = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_normaliser = peak
        + log_terms
            .iter()
            .map(|term| (term - peak).exp())
            .sum::<f64>()
            .ln();
    let tilt = log_terms
        .iter()
        .map(|term| (term - log_normaliser).exp())
        .collect();
    (tilt, log_normaliser)
}

fn require_plain_runtime(runtime: &DeviationRuntime, label: &str) -> Result<(), String> {
    if runtime.installed_flex_block().is_some() {
        return Err(format!(
            "BMS calibrated pilot: the {label} runtime already carries an installed flex block; \
             its total Jacobian is taken on the plain basis"
        ));
    }
    Ok(())
}

fn scaled_design_rows(design: &DesignMatrix, scale: &Array1<f64>) -> Result<Array2<f64>, String> {
    if design.nrows() != scale.len() {
        return Err(format!(
            "BMS calibrated pilot: design has {} rows, pilot has {}",
            design.nrows(),
            scale.len(),
        ));
    }
    let mut out = design
        .try_to_dense_arc("BMS calibrated pilot Jacobian design")?
        .as_ref()
        .clone();
    for (mut row, &factor) in out.rows_mut().into_iter().zip(scale.iter()) {
        row.mapv_inplace(|v| v * factor);
    }
    Ok(out)
}

/// `E[C(X)]` for `X ~ N(mean, sd²)` and the runtime basis `C`: a piecewise
/// cubic on its spans, `C(l) + ...` in the span-local coordinate `x − l`, and
/// constant past each support end. Each span contributes
/// `Σ_k c_k·E[(X − l)^k; l ≤ X ≤ r]`, from the truncated standard-normal
/// moments `I_k = ∫_α^β v^k φ(v) dv` through `X − l = d + sd·v`, `d = mean − l`.
fn gaussian_runtime_expectation(
    runtime: &DeviationRuntime,
    mean: f64,
    sd: f64,
) -> Result<Array1<f64>, String> {
    if !(mean.is_finite() && sd.is_finite() && sd >= 0.0) {
        return Err(format!(
            "BMS calibrated pilot: Gaussian tilt N({mean}, {sd}²) is not finite"
        ));
    }
    if sd == 0.0 {
        return Ok(runtime.design(&Array1::from_elem(1, mean))?.row(0).to_owned());
    }
    let p = runtime.basis_dim();
    let (support_left, support_right) = runtime.support_interval()?;
    let mut out = Array1::<f64>::zeros(p);
    out.scaled_add(
        normal_cdf((support_left - mean) / sd),
        &runtime.span_c0().row(0),
    );
    out.scaled_add(
        normal_cdf((mean - support_right) / sd),
        &runtime.right_boundary_value_row,
    );
    // `v^k·φ(v)` is 0 wherever the density has underflowed, including a
    // standardised endpoint that overflowed.
    let tail_moment = |v: f64, density: f64| if density == 0.0 { 0.0 } else { v * density };
    let (c0, c1, c2, c3) = (
        runtime.span_c0(),
        runtime.span_c1(),
        runtime.span_c2(),
        runtime.span_c3(),
    );
    for span in 0..runtime.span_count() {
        let (left, right) = runtime.span_interval(span)?;
        let alpha = (left - mean) / sd;
        let beta = (right - mean) / sd;
        let i0 = if alpha >= 0.0 {
            normal_cdf(-alpha) - normal_cdf(-beta)
        } else {
            normal_cdf(beta) - normal_cdf(alpha)
        };
        let (phi_alpha, phi_beta) = (normal_pdf(alpha), normal_pdf(beta));
        let i1 = phi_alpha - phi_beta;
        let a1 = tail_moment(alpha, phi_alpha);
        let b1 = tail_moment(beta, phi_beta);
        let i2 = i0 + a1 - b1;
        let i3 = 2.0 * i1 + tail_moment(alpha, a1) - tail_moment(beta, b1);
        let d = mean - left;
        let j0 = i0;
        let j1 = d * i0 + sd * i1;
        let j2 = d * d * i0 + 2.0 * d * sd * i1 + sd * sd * i2;
        let j3 = d * d * d * i0 + 3.0 * d * d * sd * i1 + 3.0 * d * sd * sd * i2 + sd * sd * sd * i3;
        for k in 0..p {
            out[k] += c0[[span, k]] * j0 + c1[[span, k]] * j1 + c2[[span, k]] * j2 + c3[[span, k]] * j3;
        }
    }
    if let Some(k) = out.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "BMS calibrated pilot: Gaussian expectation of basis column {k} is non-finite at N({mean}, {sd}²)"
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bms::gradient_paths::padded_deviation_seed;

    const PROBIT: InverseLink = InverseLink::Standard(StandardLink::Probit);

    fn score_warp_runtime() -> DeviationRuntime {
        let seed = Array1::linspace(-2.0, 2.0, 101);
        build_score_warp_deviation_block_from_seed(&seed, &DeviationBlockConfig::default())
            .expect("score-warp runtime")
            .runtime
    }

    fn link_runtime() -> DeviationRuntime {
        let seed = padded_deviation_seed(&Array1::linspace(-2.5, 2.5, 101), 1.0, 0.5);
        build_link_deviation_block_from_knots_design_seed_and_weights(
            &seed,
            &seed,
            &DeviationBlockConfig::default(),
        )
        .expect("link-deviation runtime")
        .runtime
    }

    /// A grid with weights `∝ exp(−u²/2 + tilt·u)` on `count` equispaced nodes.
    fn grid(count: usize, half_width: f64, tilt: f64) -> EmpiricalZGrid {
        let nodes: Vec<f64> = Array1::linspace(-half_width, half_width, count).to_vec();
        let raw: Vec<f64> = nodes.iter().map(|u| (-0.5 * u * u + tilt * u).exp()).collect();
        let total: f64 = raw.iter().sum();
        let weights = raw.iter().map(|w| w / total).collect();
        EmpiricalZGrid::new(nodes, weights, "pilot total Jacobian test grid").expect("grid")
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        assert_eq!(a.dim(), b.dim());
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    fn ones(n: usize) -> DesignMatrix {
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::<f64>::ones((n, 1)),
        ))
    }

    #[test]
    fn gaussian_runtime_expectation_matches_quadrature() {
        let runtime = link_runtime();
        for &(mean, sd) in &[(0.7, 1.9), (-3.4, 0.3), (5.0, 0.8), (0.0, 0.05)] {
            let exact = gaussian_runtime_expectation(&runtime, mean, sd).expect("expectation");
            // Riemann sum over ±14 sd; the integrand is a C² piecewise cubic
            // with C⁰ corners at the support ends, so the node error is
            // O(h²·|jump in C′|) with h = 28 sd / 400000.
            let nodes = Array1::linspace(mean - 14.0 * sd, mean + 14.0 * sd, 400_001);
            let h = 28.0 * sd / 400_000.0;
            let density = nodes.mapv(|x| normal_pdf((x - mean) / sd) * h / sd);
            let quadrature = runtime.design(&nodes).expect("design").t().dot(&density);
            for k in 0..runtime.basis_dim() {
                assert!(
                    (exact[k] - quadrature[k]).abs() < 1e-7,
                    "column {k} at N({mean}, {sd}²): closed form {} vs quadrature {}",
                    exact[k],
                    quadrature[k],
                );
            }
        }
    }

    #[test]
    fn standard_normal_pilot_matches_fine_normal_grid() {
        let z = Array1::from(vec![-1.3, -0.2, 0.4, 1.1, 2.0]);
        let q = Array1::from(vec![-1.0, 0.3, -0.4, 0.9, 1.5]);
        let slope = Array1::from(vec![0.4, 1.2, -0.7, 0.9, 2.1]);
        let probit_scale = 1.3;
        let standard = LatentMeasureKind::StandardNormal;
        let fine = LatentMeasureKind::GlobalEmpirical {
            grid: grid(20_001, 10.0, 0.0),
        };
        let a = BmsCalibratedPilot::calibrated(&standard, &PROBIT, probit_scale, &z, &q, slope.clone())
            .expect("standard-normal pilot");
        let b = BmsCalibratedPilot::calibrated(&fine, &PROBIT, probit_scale, &z, &q, slope)
            .expect("fine-grid pilot");
        for i in 0..z.len() {
            assert!((a.observed_intercept[i] - b.observed_intercept[i]).abs() < 1e-8);
            assert!((a.marginal_scale[i] - b.marginal_scale[i]).abs() < 1e-8);
        }
        let design = ones(z.len());
        let slope_gap = max_abs_diff(
            &a.slope_jacobian(&design).expect("slope"),
            &b.slope_jacobian(&design).expect("slope"),
        );
        assert!(slope_gap < 1e-8, "slope Jacobian gap {slope_gap}");
        let sw = score_warp_runtime();
        let sw_gap = max_abs_diff(
            &a.score_warp_jacobian(&sw).expect("score warp"),
            &b.score_warp_jacobian(&sw).expect("score warp"),
        );
        assert!(sw_gap < 1e-6, "score-warp Jacobian gap {sw_gap}");
        let link = link_runtime();
        let link_gap = max_abs_diff(
            &a.link_deviation_jacobian(&link).expect("link deviation"),
            &b.link_deviation_jacobian(&link).expect("link deviation"),
        );
        assert!(link_gap < 1e-6, "link-deviation Jacobian gap {link_gap}");
    }

    /// The row predictor with the calibration solved by bisection over the
    /// grid: `index(a, u)` is the pre-scale index at node `u` for pre-scale
    /// intercept `a`; returns `s·index(a*, z)`.
    fn calibrated_row_eta(
        grid: &EmpiricalZGrid,
        q: f64,
        probit_scale: f64,
        z: f64,
        index: &dyn Fn(f64, &Array1<f64>) -> Array1<f64>,
    ) -> f64 {
        let nodes = Array1::from(grid.nodes.clone());
        let weights = Array1::from(grid.weights.clone());
        let target = normal_cdf(q);
        let (mut lo, mut hi) = (-60.0_f64, 60.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let mass: f64 = index(mid, &nodes)
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| w * normal_cdf(probit_scale * x))
                .sum();
            if mass < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let a = 0.5 * (lo + hi);
        probit_scale * index(a, &Array1::from_elem(1, z))[0]
    }

    #[test]
    fn empirical_total_jacobian_matches_finite_differences_of_the_calibration() {
        let latent = grid(41, 4.0, 0.4);
        let measure = LatentMeasureKind::GlobalEmpirical {
            grid: latent.clone(),
        };
        let z = Array1::from(vec![-1.1, 0.3, 1.7]);
        let q = Array1::from(vec![-0.6, 0.2, 1.1]);
        let pre_slope = Array1::from(vec![0.5, -0.8, 1.4]);
        let s = 1.3;
        let pilot = BmsCalibratedPilot::calibrated(&measure, &PROBIT, s, &z, &q, pre_slope.mapv(|b| s * b))
            .expect("pilot");
        let sw = score_warp_runtime();
        let link = link_runtime();
        let design = ones(z.len());
        let slope_j = pilot.slope_jacobian(&design).expect("slope");
        let marginal_j = pilot.marginal_jacobian(&design).expect("marginal");
        let sw_j = pilot.score_warp_jacobian(&sw).expect("score warp");
        let link_j = pilot.link_deviation_jacobian(&link).expect("link deviation");
        let h = 1e-5;
        let tol = 1e-7;
        let central = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
        for i in 0..z.len() {
            let b = pre_slope[i];
            let rigid = |b: f64| move |a: f64, u: &Array1<f64>| u.mapv(|u| a + b * u);
            let fd_marginal = central(&|e: f64| calibrated_row_eta(&latent, q[i] + e, s, z[i], &rigid(b)));
            assert!((fd_marginal - marginal_j[[i, 0]]).abs() < tol, "marginal row {i}");
            let fd_slope = central(&|e: f64| calibrated_row_eta(&latent, q[i], s, z[i], &rigid(b + e)));
            assert!((fd_slope - slope_j[[i, 0]]).abs() < tol, "slope row {i}");
            for k in 0..sw.basis_dim() {
                let fd = central(&|e: f64| {
                    calibrated_row_eta(&latent, q[i], s, z[i], &|a: f64, u: &Array1<f64>| {
                        let warp = sw.design(u).expect("design").column(k).to_owned();
                        u.mapv(|u| a + b * u) + &(warp * (b * e))
                    })
                });
                assert!((fd - sw_j[[i, k]]).abs() < tol, "score warp row {i} column {k}");
            }
            for k in 0..link.basis_dim() {
                let fd = central(&|e: f64| {
                    calibrated_row_eta(&latent, q[i], s, z[i], &|a: f64, u: &Array1<f64>| {
                        let t = u.mapv(|u| a + b * u);
                        let dev = link.design(&t).expect("design").column(k).to_owned();
                        &t + &(dev * e)
                    })
                });
                assert!((fd - link_j[[i, k]]).abs() < tol, "link deviation row {i} column {k}");
            }
        }
    }

    /// gam#3320: rows whose link argument sits inside one span see the basis
    /// as a cubic polynomial, so the raw training-row design has rank at most
    /// four however many columns the calibration identifies. The total
    /// Jacobian reads each column over the latent measure too.
    #[test]
    fn total_jacobian_resolves_columns_the_training_rows_cannot() {
        let link = link_runtime();
        let span = link.span_count() / 2;
        let (left, right) = link.span_interval(span).expect("span");
        let n = 12;
        let q = Array1::linspace(-1.5, 1.5, n);
        let observed_slope = Array1::from_elem(n, 0.8);
        let target_t = Array1::linspace(left + 0.25 * (right - left), left + 0.75 * (right - left), n);
        let z = Array1::from_shape_fn(n, |i| {
            let intercept = q[i] * (1.0_f64 + 0.64).sqrt();
            (target_t[i] - intercept) / 0.8
        });
        let measure = LatentMeasureKind::StandardNormal;
        let pilot = BmsCalibratedPilot::calibrated(&measure, &PROBIT, 1.0, &z, &q, observed_slope)
            .expect("pilot");
        for i in 0..n {
            assert!((pilot.link_argument()[i] - target_t[i]).abs() < 1e-12);
        }
        let rank = |m: &Array2<f64>| {
            let (evals, _) = FaerEigh::eigh(&m.t().dot(m), faer::Side::Lower).expect("eigh");
            let threshold = gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(
                evals.as_slice().expect("contiguous"),
            );
            evals.iter().filter(|&&v| v > threshold).count()
        };
        let raw = link.design(pilot.link_argument()).expect("raw design");
        let total = pilot.link_deviation_jacobian(&link).expect("total Jacobian");
        let raw_rank = rank(&raw);
        let total_rank = rank(&total);
        assert!(raw_rank <= 4, "raw rank {raw_rank} inside one cubic span");
        assert!(
            total_rank > raw_rank,
            "total Jacobian rank {total_rank} does not exceed the raw rank {raw_rank} (p = {})",
            link.basis_dim(),
        );
    }
}
