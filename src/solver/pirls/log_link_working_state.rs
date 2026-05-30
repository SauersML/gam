//! Shared PIRLS working-state engine for log-link exponential-family rows.
//!
//! Poisson, Gamma, Tweedie, and (fixed-theta) negative-binomial all share the
//! identical log-link IRLS row geometry: `mu = exp(eta)`, the canonical working
//! response `z = eta + (y - mu) / mu`, the same clamping/flooring policy, and
//! the same `dmu/deta` jet (`mu`, `mu`, `mu`). The families differ only in the
//! Fisher working weight and in the second/third curvature buffers `c`/`d`.
//!
//! [`write_log_link_working_state`] is the single row engine; each family
//! supplies a [`LogLinkWorkingKernel`] that contributes the weight rule, the
//! curvature terms, and two numerical policies (whether the weight is floored at
//! `MIN_WEIGHT`, and whether the `mu`-jet is zeroed when `eta` is clamped).

use super::{WorkingDerivativeBuffersMut, standard_inverse_link_jet};
use crate::estimate::EstimationError;
use crate::types::InverseLink;
use ndarray::{Array1, ArrayView1};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// Lower floor on `mu = exp(eta)`, shared by every log-link family so the
/// working response `(y - mu) / mu` and any fractional `mu` power stay finite.
pub(super) const MIN_MU: f64 = 1e-10;

/// Lower floor on positive working weights, used by the floored families so the
/// weighted normal equations stay numerically well posed.
pub(super) const MIN_WEIGHT: f64 = 1e-12;

/// Clamp bound on `eta` so `exp(eta)` cannot overflow.
const ETA_CLAMP: f64 = 700.0;

/// Family-specific contributions to the shared log-link row engine.
///
/// The engine owns the outer loop, the `eta` clamp, the `mu` floor, the working
/// response, the `dmu/deta` jet, and the buffer plumbing; the kernel supplies
/// only the pieces that genuinely vary between families.
pub(super) trait LogLinkWorkingKernel: Sync {
    /// Validate `y`/`priorweights` once before the row loop (integer counts,
    /// non-negative responses, parameter ranges, ...). Defaults to no-op.
    fn validate(
        &self,
        _y: ArrayView1<f64>,
        _priorweights: ArrayView1<f64>,
    ) -> Result<(), EstimationError> {
        Ok(())
    }

    /// Raw (un-floored) Fisher working weight for one row, already including the
    /// prior weight: `prior_weight * V_Fisher(mu)`-style contribution.
    fn raw_weight(&self, y: f64, mu: f64, prior_weight: f64) -> f64;

    /// Second/third curvature buffers `(c, d)` for one row, given the row's
    /// floored `raw_weight`. Returned only when the row keeps curvature; the
    /// engine zeros them when a floor/clamp invalidates the local jet.
    fn curvature_terms(&self, y: f64, mu: f64, raw_weight: f64) -> (f64, f64);

    /// Whether positive weights are floored at [`MIN_WEIGHT`]. Families with a
    /// `mu`-dependent weight floor (Poisson/Tweedie/NegBin) return `true`; the
    /// Gamma weight is independent of `eta` and is written unfloored.
    fn floor_weight(&self) -> bool {
        true
    }

    /// Whether the `dmu/deta` jet buffers are zeroed when `eta` is clamped. Only
    /// Tweedie (with its fractional `mu` power) needs this; the others keep the
    /// `mu`-valued jet regardless of clamping.
    fn zero_mu_jet_on_clamp(&self) -> bool {
        false
    }
}

/// Shared log-link IRLS working-state writer.
///
/// Computes `mu`, `weights`, and `z` for every row, and (when `derivatives` is
/// present) the `dmu/deta` jet plus the curvature buffers `c`/`d`, delegating
/// the weight rule, curvature, and numerical policies to `kernel`.
pub(super) fn write_log_link_working_state<K: LogLinkWorkingKernel>(
    kernel: &K,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    kernel.validate(y, priorweights)?;
    let floor_weight = kernel.floor_weight();
    let zero_mu_jet_on_clamp = kernel.zero_mu_jet_on_clamp();
    if let Some(derivs) = derivatives {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        let dmu_s = derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous");
        let d2_s = derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous");
        let d3_s = derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous");
        let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
        let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
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
                    let raw_weight = kernel.raw_weight(y[i], mu_i, priorweights[i].max(0.0));
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
                        let (c_i, d_i) = kernel.curvature_terms(y[i], mu_i, *w_o);
                        *c_o = c_i;
                        *d_o = d_i;
                    }
                },
            );
    } else {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP);
                let mu_i = eta_i.exp().max(MIN_MU);
                *mu_o = mu_i;
                let raw_weight = kernel.raw_weight(y[i], mu_i, priorweights[i].max(0.0));
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
    Ok(())
}

/// Shared log-link outer-derivative writer for the η-curvature carriers.
///
/// The exact-outer-derivative path needs, per row, the working-curvature
/// carriers `c = dW/dη`, `d = d²W/dη²` and the inverse-link jet
/// (`dmu/deta`, `d²mu/deta²`, `d³mu/deta³`) — but no `mu`/`weights`/`z`. The
/// weight rule and the `(c, d)` curvature are exactly the same per-family
/// functions the working-state engine uses, so this routes them through the
/// identical [`LogLinkWorkingKernel`] instead of re-deriving them inline.
///
/// `kernel.validate` is invoked with empty views so its parameter-range checks
/// (Tweedie power/phi, negative-binomial theta) still fire while the response
/// validation — which has no `y` to inspect on this path — is skipped.
pub(super) fn write_log_link_eta_curvature<K: LogLinkWorkingKernel>(
    kernel: &K,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    buffers: WorkingDerivativeBuffersMut<'_>,
) -> Result<(), EstimationError> {
    // Run the kernel's parameter-range validation only; the response loop in
    // `validate` is a no-op over these empty views, which is correct here since
    // the outer-derivative path has no `y` to inspect.
    const EMPTY: &[f64] = &[];
    kernel.validate(ArrayView1::from(EMPTY), ArrayView1::from(EMPTY))?;
    let floor_weight = kernel.floor_weight();
    let zero_mu_jet_on_clamp = kernel.zero_mu_jet_on_clamp();
    let c_s = buffers.c.as_slice_mut().expect("c must be contiguous");
    let d_s = buffers.d.as_slice_mut().expect("d must be contiguous");
    let dmu_s = buffers
        .dmu_deta
        .as_slice_mut()
        .expect("dmu_deta must be contiguous");
    let d2_s = buffers
        .d2mu_deta2
        .as_slice_mut()
        .expect("d2mu_deta2 must be contiguous");
    let d3_s = buffers
        .d3mu_deta3
        .as_slice_mut()
        .expect("d3mu_deta3 must be contiguous");
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
                let raw_weight = kernel.raw_weight(0.0, jet.mu, priorweights[i].max(0.0));
                let floor_active = floor_weight && raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                if clamp_active || floor_active {
                    *c_o = 0.0;
                    *d_o = 0.0;
                } else {
                    let (c_i, d_i) = kernel.curvature_terms(0.0, jet.mu, raw_weight);
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
