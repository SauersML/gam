//! Shared PIRLS working-state engine for log-link exponential-family rows.
//!
//! Poisson, Gamma, Tweedie, and (fixed-theta) negative-binomial all share the
//! identical log-link IRLS row geometry: `mu = exp(eta)`, the canonical working
//! response `z = eta + (y - mu) / mu`, the same clamping/flooring policy, and
//! the same `dmu/deta` jet (`mu`, `mu`, `mu`). The families differ only in the
//! Fisher working weight and in the second/third curvature buffers `c`/`d`.
//!
//! [`write_log_link_working_state`] is the single row engine; each family
//! supplies a [`LogLinkRule`] that describes — as data — the weight rule, the
//! curvature rule, and two numerical policies (whether the weight is floored at
//! `MIN_WEIGHT`, and whether the `mu`-jet is zeroed when `eta` is clamped). The
//! rule is matched per row, so each family's formula consumes exactly the inputs
//! it depends on (the Gamma weight ignores `mu`, the Poisson curvature ignores
//! it) without any family being forced to accept an argument it never reads.

use super::{
    WorkingDerivSlices, WorkingDerivativeBuffersMut, WorkingSlices, standard_inverse_link_jet,
    working_deriv_slices, working_slices,
};
use crate::estimate::EstimationError;
use gam_problem::{InverseLink, MIN_WEIGHT};
use ndarray::{Array1, ArrayView1};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// Lower floor on `mu = exp(eta)`, shared by every log-link family so the
/// working response `(y - mu) / mu` and any fractional `mu` power stay finite.
pub(crate) const MIN_MU: f64 = 1e-10;

/// Clamp bound on `eta` so `exp(eta)` cannot overflow (`exp` overflows near
/// `709`). Shared by every PIRLS row that exponentiates a linear predictor.
pub(crate) const ETA_CLAMP: f64 = 700.0;

/// Family-specific raw (un-floored) Fisher working-weight rule, already scaled
/// by the row's prior weight. The Fisher weight never depends on the response
/// `y` (that is what distinguishes Fisher scoring from observed-information
/// Newton), so the response is not part of any rule.
pub(super) enum WorkingWeight {
    /// Poisson: `prior * mu` (`V(mu) = mu`).
    PoissonIdentity,
    /// Gamma(shape): `prior * shape`, independent of `eta`/`mu` (`V(mu) = mu^2`).
    Constant { factor: f64 },
    /// Tweedie(p, phi): `prior * floor(mu)^(2 - p) / phi`.
    TweediePower { p: f64, phi: f64 },
    /// Negative-binomial(theta): `prior * mu * theta / (theta + mu)`, written in
    /// the cancellation-stable branch form.
    NegativeBinomial { theta: f64 },
}

/// Family-specific second/third curvature-buffer rule `(c, d)`, evaluated from
/// the row's floored working weight (and, for negative-binomial, `mu`).
pub(super) enum WorkingCurvature {
    /// `(c_ratio * weight, d_ratio * weight)`. Covers the canonical/quasi
    /// families whose curvature is a fixed multiple of the weight: Poisson
    /// `(1, 1)`, Gamma `(0, 0)`, Tweedie `(2 - p, (2 - p)^2)`.
    Proportional { c_ratio: f64, d_ratio: f64 },
    /// Negative-binomial(theta): `mu`-dependent curvature.
    NegativeBinomial { theta: f64 },
}

/// Full description of a log-link family's contribution to the shared row
/// engine: the weight rule, the curvature rule, and the two numerical policies
/// (whether positive weights are floored at [`MIN_WEIGHT`], and whether the
/// `dmu/deta` jet is zeroed when `eta` is clamped).
pub(super) struct LogLinkRule {
    pub weight: WorkingWeight,
    pub curvature: WorkingCurvature,
    /// Families with a `mu`-dependent weight floor (Poisson/Tweedie/NegBin) set
    /// this; the Gamma weight is `eta`-independent and is written unfloored.
    pub floor_weight: bool,
    /// Only Tweedie (fractional `mu` power) needs the jet zeroed on clamp; the
    /// others keep the `mu`-valued jet regardless of clamping.
    pub zero_mu_jet_on_clamp: bool,
}

/// Raw (un-floored) Fisher working weight for one row. Each match arm reads only
/// the inputs its family depends on; `mu` is consumed by the identity, power,
/// and negative-binomial arms.
#[inline]
pub(crate) fn raw_weight(weight: &WorkingWeight, mu: f64, prior_weight: f64) -> f64 {
    match *weight {
        WorkingWeight::PoissonIdentity => prior_weight * mu,
        WorkingWeight::Constant { factor } => prior_weight * factor,
        WorkingWeight::TweediePower { p, phi } => {
            prior_weight * super::tweedie_log_weight_mu_power(mu, p) / phi
        }
        WorkingWeight::NegativeBinomial { theta } => {
            let negbin_weight = if theta > mu {
                mu / (1.0 + mu / theta)
            } else {
                theta / (1.0 + theta / mu)
            };
            prior_weight * negbin_weight
        }
    }
}

/// Second/third curvature buffers `(c, d)` for one row, from the row's floored
/// `weight`. The proportional arm ignores `mu`; the negative-binomial arm reads
/// it.
#[inline]
pub(crate) fn curvature_terms(curvature: &WorkingCurvature, mu: f64, weight: f64) -> (f64, f64) {
    match *curvature {
        WorkingCurvature::Proportional { c_ratio, d_ratio } => (c_ratio * weight, d_ratio * weight),
        WorkingCurvature::NegativeBinomial { theta } => {
            let denom = theta + mu;
            (
                weight * theta / denom,
                weight * theta * (theta - mu) / (denom * denom),
            )
        }
    }
}

/// Shared log-link IRLS working-state writer.
///
/// Computes `mu`, `weights`, and `z` for every row, and (when `derivatives` is
/// present) the `dmu/deta` jet plus the curvature buffers `c`/`d`, delegating
/// the weight rule, curvature, and numerical policies to `kernel`.
pub(super) fn write_log_link_working_state(
    rule: &LogLinkRule,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    let floor_weight = rule.floor_weight;
    let zero_mu_jet_on_clamp = rule.zero_mu_jet_on_clamp;
    if let Some(mut derivs) = derivatives {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        let WorkingDerivSlices {
            c: c_s,
            d: d_s,
            dmu: dmu_s,
            d2: d2_s,
            d3: d3_s,
        } = working_deriv_slices(&mut derivs);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
                    let clamp_active = eta_raw != eta_i;
                    let mu_i = eta_i.exp().max(MIN_MU);
                    *mu_o = mu_i;
                    let raw_weight = raw_weight(&rule.weight, mu_i, priorweights[i].max(0.0));
                    let floor_active = floor_weight && raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    *w_o = if raw_weight > 0.0 {
                        if floor_weight {
                            raw_weight.max(MIN_WEIGHT)
                        } else {
                            raw_weight
                        }
                    } else {
                        0.0
                    };
                    *z_o = eta_i + (y[i] - mu_i) / mu_i;
                    if zero_mu_jet_on_clamp && clamp_active {
                        *dmu_o = 0.0;
                        *d2_o = 0.0;
                        *d3_o = 0.0;
                    } else {
                        *dmu_o = mu_i;
                        *d2_o = mu_i;
                        *d3_o = mu_i;
                    }
                    if floor_active || clamp_active {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        let (c_i, d_i) = curvature_terms(&rule.curvature, mu_i, *w_o);
                        *c_o = c_i;
                        *d_o = d_i;
                    }
                },
            );
    } else {
        let WorkingSlices {
            mu: mu_s,
            weights: weights_s,
            z: z_s,
        } = working_slices(mu, weights, z);
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP);
                let mu_i = eta_i.exp().max(MIN_MU);
                *mu_o = mu_i;
                let raw_weight = raw_weight(&rule.weight, mu_i, priorweights[i].max(0.0));
                *w_o = if raw_weight > 0.0 {
                    if floor_weight {
                        raw_weight.max(MIN_WEIGHT)
                    } else {
                        raw_weight
                    }
                } else {
                    0.0
                };
                *z_o = eta_i + (y[i] - mu_i) / mu_i;
            });
    }
}

/// Shared log-link outer-derivative writer for the η-curvature carriers.
///
/// The exact-outer-derivative path needs, per row, the working-curvature
/// carriers `c = dW/dη`, `d = d²W/dη²` and the inverse-link jet
/// (`dmu/deta`, `d²mu/deta²`, `d³mu/deta³`) — but no `mu`/`weights`/`z`. The
/// weight rule and the `(c, d)` curvature are exactly the same data-driven
/// [`LogLinkRule`] the working-state engine uses, so this routes them through
/// the identical [`raw_weight`]/[`curvature_terms`] functions instead of
/// re-deriving them inline.
///
/// Parameter-range validation (Tweedie power/phi, negative-binomial theta)
/// belongs to the caller that owns the raw parameters; this path has no `y`,
/// so the response validation the working-state builders perform is skipped.
pub(super) fn write_log_link_eta_curvature(
    rule: &LogLinkRule,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mut buffers: WorkingDerivativeBuffersMut<'_>,
) -> Result<(), EstimationError> {
    let floor_weight = rule.floor_weight;
    let zero_mu_jet_on_clamp = rule.zero_mu_jet_on_clamp;
    let WorkingDerivSlices {
        c: c_s,
        d: d_s,
        dmu: dmu_s,
        d2: d2_s,
        d3: d3_s,
    } = working_deriv_slices(&mut buffers);
    c_s.par_iter_mut()
        .zip(d_s.par_iter_mut())
        .zip(dmu_s.par_iter_mut())
        .zip(d2_s.par_iter_mut())
        .zip(d3_s.par_iter_mut())
        .enumerate()
        .try_for_each(
            |(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| -> Result<(), EstimationError> {
                let eta_raw = eta[i];
                let eta_used = eta_raw.clamp(-ETA_CLAMP, ETA_CLAMP);
                let clamp_active = eta_raw != eta_used;
                let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                let raw_w = raw_weight(&rule.weight, jet.mu, priorweights[i].max(0.0));
                let floor_active = floor_weight && raw_w > 0.0 && raw_w <= MIN_WEIGHT;
                if clamp_active || floor_active {
                    *c_o = 0.0;
                    *d_o = 0.0;
                } else {
                    let (c_i, d_i) = curvature_terms(&rule.curvature, jet.mu, raw_w);
                    *c_o = c_i;
                    *d_o = d_i;
                }
                if zero_mu_jet_on_clamp && clamp_active {
                    *dmu_o = 0.0;
                    *d2_o = 0.0;
                    *d3_o = 0.0;
                } else {
                    *dmu_o = jet.d1;
                    *d2_o = jet.d2;
                    *d3_o = jet.d3;
                }
                Ok(())
            },
        )
}
