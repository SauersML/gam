//! Per-row GLM residuals of a fitted linear predictor.
//!
//! Each residual type is evaluated from the same per-row objects the inner
//! solver and the reported deviance use, so their defining identities hold
//! exactly rather than approximately:
//!
//! * response `r_i = y_i − μ_i`;
//! * working `r_i = (y_i − μ_i) / (dμ/dη)_i`, the IRLS working-response
//!   residual `z_i − η_i`;
//! * deviance `r_i = sign(y_i − μ_i)·√(d_i)` with `d_i` the prior-weighted unit
//!   deviance from the deviance row oracle, so `Σ r_i² = D` is the reported
//!   deviance;
//! * Pearson `r_i = (y_i − μ_i)·√(w_i / V(μ_i))` with `V` the family variance
//!   function *without* the dispersion, so `Σ r_i²` is the Pearson statistic
//!   and `φ·V(μ_i)/w_i = Var(Y_i)`.
//!
//! Binary-response rows take `y − μ` from the cancellation-free `(μ, 1−μ)`
//! pair, the same residual the inner solver's score uses.

use super::*;
use crate::mixture_link::{
    inverse_link_complement_for_inverse_link, inverse_link_mu_d1_for_inverse_link,
};
use serde::{Deserialize, Serialize};

/// The residual type returned by [`glm_residuals`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualKind {
    Response,
    Working,
    Deviance,
    Pearson,
}

impl ResidualKind {
    pub const ALL: [ResidualKind; 4] = [
        ResidualKind::Response,
        ResidualKind::Working,
        ResidualKind::Deviance,
        ResidualKind::Pearson,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ResidualKind::Response => "response",
            ResidualKind::Working => "working",
            ResidualKind::Deviance => "deviance",
            ResidualKind::Pearson => "pearson",
        }
    }
}

impl std::fmt::Display for ResidualKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for ResidualKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        ResidualKind::ALL
            .into_iter()
            .find(|kind| kind.name() == value)
            .ok_or_else(|| {
                let names: Vec<&str> = ResidualKind::ALL.iter().map(|kind| kind.name()).collect();
                format!(
                    "unknown residual type '{value}'; expected one of: {}",
                    names.join(", ")
                )
            })
    }
}

/// Family variance function `V(μ)` without the dispersion factor.
///
/// Beta's variance jet carries its `1/(1+φ)` precision scale, which is exactly
/// the dispersion the Pearson residual must leave out, so Beta takes the bare
/// Bernoulli variance `μ(1−μ)`.
fn unit_variance(family: WeightFamily, mu: f64, one_minus_mu: f64) -> f64 {
    match family {
        WeightFamily::Beta { .. } => mu * one_minus_mu,
        other => variance_jet_for_weight_family(other, mu, one_minus_mu).v,
    }
}

/// Per-row residuals of type `kind` at linear predictor `eta`.
pub fn glm_residuals(
    y: ArrayView1<f64>,
    eta: ArrayView1<f64>,
    prior_weights: ArrayView1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    kind: ResidualKind,
) -> Result<Array1<f64>, EstimationError> {
    if y.len() != eta.len() || prior_weights.len() != eta.len() {
        crate::bail_invalid_estim!(
            "residual length mismatch: y={}, eta={}, prior_weights={}",
            y.len(),
            eta.len(),
            prior_weights.len()
        );
    }
    let family = weight_family_for_glm_likelihood(likelihood)?;
    let mut out = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let (mu, dmu_deta) = inverse_link_mu_d1_for_inverse_link(inverse_link, eta[i])?;
        let one_minus_mu = inverse_link_complement_for_inverse_link(inverse_link, eta[i], mu);
        let raw = bernoulli_pair_residual(family, y[i], mu, one_minus_mu);
        let value = match kind {
            ResidualKind::Response => raw,
            ResidualKind::Working => raw / dmu_deta,
            ResidualKind::Deviance => {
                let row = deviance_eta_row_with_log_measure_scale(
                    i,
                    y[i],
                    eta[i],
                    likelihood,
                    inverse_link,
                    prior_weights[i],
                    0.0,
                )?;
                // The oracle's half-deviance is a non-negative KL divergence;
                // a negative value would surface as NaN and be refused below.
                raw.signum() * (2.0 * row.half_deviance).sqrt()
            }
            ResidualKind::Pearson => {
                raw * (prior_weights[i] / unit_variance(family, mu, one_minus_mu)).sqrt()
            }
        };
        if !value.is_finite() {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                kind.name(),
                eta[i],
                value,
            ));
        }
        out[i] = value;
    }
    Ok(out)
}
