//! Exact PIRLS row geometry shared by every log-link exponential family.
//!
//! The statistical map is evaluated on one surface only:
//! `mu = exp(eta)`, `z = eta + (y - mu) / mu`, and each family's analytic
//! Fisher-weight jet.  There is no projection of `eta`, floor on `mu` or the
//! statistical weight, or derivative masking.  The central log-link solver
//! domain certifies `exp(eta)` first; this module then certifies every
//! row-dependent product and quotient that PIRLS actually needs.  A trial whose
//! exact row state is not representable is rejected through a typed
//! [`EstimationError`] instead of being changed into a different model.

use super::{WorkingDerivativeBuffersMut, working_deriv_slices, working_slices};
use crate::estimate::EstimationError;
use ndarray::{Array1, ArrayView1};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// Family-specific Fisher working weight, before multiplication by the row's
/// prior weight.
pub(super) enum WorkingWeight {
    /// Poisson: `mu`.
    PoissonIdentity,
    /// Gamma(shape): `shape`, independent of `eta`.
    Constant { factor: f64 },
    /// Tweedie(p, phi): `mu^(2-p) / phi`.
    TweediePower { p: f64, phi: f64 },
    /// Negative-binomial(theta): `mu * theta / (theta + mu)`.
    NegativeBinomial { theta: f64 },
}

/// First two eta derivatives of the Fisher working weight.
pub(super) enum WorkingCurvature {
    /// `(c_ratio * weight, d_ratio * weight)`.
    Proportional { c_ratio: f64, d_ratio: f64 },
    /// Negative-binomial(theta), evaluated with bounded ratios.
    NegativeBinomial { theta: f64 },
}

/// A log-link family's exact statistical row rule.
pub(super) struct LogLinkRule {
    pub weight: WorkingWeight,
    pub curvature: WorkingCurvature,
}

#[derive(Clone, Copy)]
struct ExactLogLinkRow {
    mu: f64,
    weight: f64,
    c: f64,
    d: f64,
}

#[inline]
fn unrepresentable(
    row: usize,
    quantity: &'static str,
    eta: f64,
    value: f64,
) -> EstimationError {
    EstimationError::PirlsRowGeometryUnrepresentable {
        row,
        quantity,
        eta,
        value,
    }
}

#[inline]
fn exact_prior_weight(row: usize, eta: f64, prior_weight: f64) -> Result<f64, EstimationError> {
    if prior_weight.is_finite() && prior_weight >= 0.0 {
        Ok(prior_weight)
    } else {
        Err(unrepresentable(row, "prior weight", eta, prior_weight))
    }
}

/// Unit-prior Fisher weight.  Branch forms keep the negative-binomial ratio
/// bounded without ever forming `theta + mu` when that sum would overflow.
#[inline]
fn unit_weight(weight: &WorkingWeight, mu: f64) -> f64 {
    match *weight {
        WorkingWeight::PoissonIdentity => mu,
        WorkingWeight::Constant { factor } => factor,
        WorkingWeight::TweediePower { p, phi } => mu.powf(2.0 - p) / phi,
        WorkingWeight::NegativeBinomial { theta } => {
            if theta >= mu {
                mu / (1.0 + mu / theta)
            } else {
                theta / (1.0 + theta / mu)
            }
        }
    }
}

/// Exact first/second eta derivatives `(c, d)` of a represented row weight.
#[inline]
fn curvature_terms(curvature: &WorkingCurvature, mu: f64, weight: f64) -> (f64, f64) {
    match *curvature {
        WorkingCurvature::Proportional { c_ratio, d_ratio } => {
            (c_ratio * weight, d_ratio * weight)
        }
        WorkingCurvature::NegativeBinomial { theta } => {
            // r = theta / (theta + mu), evaluated without an overflowing sum.
            let r = if theta >= mu {
                1.0 / (1.0 + mu / theta)
            } else {
                let theta_over_mu = theta / mu;
                theta_over_mu / (1.0 + theta_over_mu)
            };
            let c = weight * r;
            // theta(theta-mu)/(theta+mu)^2 = r(2r-1).
            let d = c * (2.0 * r - 1.0);
            (c, d)
        }
    }
}

/// Evaluate and certify the shared `(mu, W, dW/deta, d2W/deta2)` row state.
#[inline]
fn exact_log_link_row(
    rule: &LogLinkRule,
    row: usize,
    eta: f64,
    prior_weight: f64,
) -> Result<ExactLogLinkRow, EstimationError> {
    let mu = crate::mixture_link::log_link_solver_exp(eta)?;
    let prior_weight = exact_prior_weight(row, eta, prior_weight)?;
    if prior_weight == 0.0 {
        return Ok(ExactLogLinkRow {
            mu,
            weight: 0.0,
            c: 0.0,
            d: 0.0,
        });
    }

    let unit_weight = unit_weight(&rule.weight, mu);
    if !(unit_weight.is_finite() && unit_weight > 0.0) {
        return Err(unrepresentable(
            row,
            "unit Fisher weight",
            eta,
            unit_weight,
        ));
    }
    let weight = prior_weight * unit_weight;
    if !(weight.is_finite() && weight > 0.0) {
        return Err(unrepresentable(row, "Fisher weight", eta, weight));
    }
    let (c, d) = curvature_terms(&rule.curvature, mu, weight);
    if !c.is_finite() {
        return Err(unrepresentable(row, "dW/deta", eta, c));
    }
    if !d.is_finite() {
        return Err(unrepresentable(row, "d2W/deta2", eta, d));
    }
    Ok(ExactLogLinkRow { mu, weight, c, d })
}

#[inline]
fn exact_working_response(
    row: usize,
    eta: f64,
    y: f64,
    mu: f64,
) -> Result<f64, EstimationError> {
    // `y / mu - 1` avoids losing the residual to `y - mu` when both are large,
    // while retaining the exact algebraic PIRLS score carrier.
    let z = eta + y / mu - 1.0;
    if z.is_finite() {
        Ok(z)
    } else {
        Err(unrepresentable(row, "working response", eta, z))
    }
}

/// Write exact log-link `mu`, Fisher weights, working responses, and optional
/// derivative carriers for every row.
pub(super) fn write_log_link_working_state(
    rule: &LogLinkRule,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    if let Some(mut derivs) = derivatives {
        let slices = working_slices(mu, weights, z);
        let deriv_slices = working_deriv_slices(&mut derivs);
        slices
            .mu
            .par_iter_mut()
            .zip(slices.weights.par_iter_mut())
            .zip(slices.z.par_iter_mut())
            .zip(deriv_slices.dmu.par_iter_mut())
            .zip(deriv_slices.d2.par_iter_mut())
            .zip(deriv_slices.d3.par_iter_mut())
            .zip(deriv_slices.c.par_iter_mut())
            .zip(deriv_slices.d.par_iter_mut())
            .enumerate()
            .try_for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let row = exact_log_link_row(rule, i, eta[i], priorweights[i])?;
                    let z_i = if row.weight == 0.0 {
                        eta[i]
                    } else {
                        exact_working_response(i, eta[i], y[i], row.mu)?
                    };
                    *mu_o = row.mu;
                    *w_o = row.weight;
                    *z_o = z_i;
                    *dmu_o = row.mu;
                    *d2_o = row.mu;
                    *d3_o = row.mu;
                    *c_o = row.c;
                    *d_o = row.d;
                    Ok(())
                },
            )
    } else {
        let slices = working_slices(mu, weights, z);
        slices
            .mu
            .par_iter_mut()
            .zip(slices.weights.par_iter_mut())
            .zip(slices.z.par_iter_mut())
            .enumerate()
            .try_for_each(|(i, ((mu_o, w_o), z_o))| {
                let row = exact_log_link_row(rule, i, eta[i], priorweights[i])?;
                let z_i = if row.weight == 0.0 {
                    eta[i]
                } else {
                    exact_working_response(i, eta[i], y[i], row.mu)?
                };
                *mu_o = row.mu;
                *w_o = row.weight;
                *z_o = z_i;
                Ok(())
            })
    }
}

/// Write the same exact curvature/link jets used by the working-state seam for
/// outer derivative reconstruction.
pub(super) fn write_log_link_eta_curvature(
    rule: &LogLinkRule,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mut buffers: WorkingDerivativeBuffersMut<'_>,
) -> Result<(), EstimationError> {
    let slices = working_deriv_slices(&mut buffers);
    slices
        .c
        .par_iter_mut()
        .zip(slices.d.par_iter_mut())
        .zip(slices.dmu.par_iter_mut())
        .zip(slices.d2.par_iter_mut())
        .zip(slices.d3.par_iter_mut())
        .enumerate()
        .try_for_each(|(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| {
            let row = exact_log_link_row(rule, i, eta[i], priorweights[i])?;
            *c_o = row.c;
            *d_o = row.d;
            *dmu_o = row.mu;
            *d2_o = row.mu;
            *d3_o = row.mu;
            Ok(())
        })
}
