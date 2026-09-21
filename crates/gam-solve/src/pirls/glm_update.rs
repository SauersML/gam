//! GLM working-vector updates: the family-dispatched `update_glmvectors*`
//! entry points and the working-weight / Newton-curvature derivatives w.r.t.
//! `\eta`.

use super::*;

#[derive(Clone, Copy, Default)]
struct CertifiedBernoulliRow {
    geometry: WorkingBernoulliGeometry,
    jet: MixtureInverseLinkJet,
}

#[inline]
fn certify_bernoulli_row(
    inverse_link: &InverseLink,
    row: usize,
    eta: f64,
    y: f64,
    prior_weight: f64,
) -> Result<CertifiedBernoulliRow, EstimationError> {
    if matches!(inverse_link, InverseLink::Standard(StandardLink::Logit)) {
        let jet5 = logit_inverse_link_jet5(eta);
        let geometry = bernoulli_logit_geometry_from_jet(row, eta, y, prior_weight, jet5)?;
        Ok(CertifiedBernoulliRow {
            geometry,
            jet: MixtureInverseLinkJet {
                mu: jet5.mu,
                d1: jet5.d1,
                d2: jet5.d2,
                d3: jet5.d3,
            },
        })
    } else {
        let jet = standard_inverse_link_jet(inverse_link, eta)?;
        let omm = crate::mixture_link::inverse_link_complement_for_inverse_link(
            inverse_link,
            eta,
            jet.mu,
        );
        let geometry = bernoulli_geometry_from_jet(row, eta, y, prior_weight, jet, omm)?;
        Ok(CertifiedBernoulliRow { geometry, jet })
    }
}

fn certify_bernoulli_rows(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
) -> Result<Vec<CertifiedBernoulliRow>, EstimationError> {
    super::par_certified_rows(eta.len(), |i| {
        certify_bernoulli_row(inverse_link, i, eta[i], y[i], priorweights[i])
    })
}

/// Scatter certified Bernoulli rows into the PIRLS working vectors, and into
/// the working-derivative buffers when the caller supplied them.
fn scatter_certified_bernoulli_rows(
    certified: &[CertifiedBernoulliRow],
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
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
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(certified.par_iter())
            .for_each(
                |((((((((mu_o, w_o), z_o), c_o), d_o), dmu_o), d2_o), d3_o), row)| {
                    *mu_o = row.geometry.mu;
                    *w_o = row.geometry.weight;
                    *z_o = row.geometry.z;
                    *c_o = row.geometry.c;
                    *d_o = row.geometry.d;
                    *dmu_o = row.jet.d1;
                    *d2_o = row.jet.d2;
                    *d3_o = row.jet.d3;
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
            .zip(certified.par_iter())
            .for_each(|(((mu_o, w_o), z_o), row)| {
                *mu_o = row.geometry.mu;
                *w_o = row.geometry.weight;
                *z_o = row.geometry.z;
            });
    }
}

pub(crate) fn update_glmvectors(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    let link = inverse_link.link_function();
    match link {
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::LogLog
        | LinkFunction::Cauchit
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => {
            let certified = certify_bernoulli_rows(y, eta, inverse_link, priorweights)?;
            scatter_certified_bernoulli_rows(&certified, mu, weights, z, derivatives);
            Ok(())
        }
        LinkFunction::Identity => {
            write_identityworking_state(y, eta, priorweights, mu, weights, z, derivatives)
        }
        LinkFunction::Log => {
            write_poisson_log_working_state(y, eta, priorweights, mu, weights, z, derivatives)
        }
        // A reciprocal-power or square-root link's Fisher weight depends on
        // the variance function, which only the likelihood knows.
        LinkFunction::Inverse | LinkFunction::InverseSquared | LinkFunction::Sqrt => {
            crate::bail_invalid_estim!(
                "the {link:?} link's working state is family-specific; route it through the likelihood's IRLS update"
            )
        }
    }
}

/// Family-dispatched GLM vector update helper.
#[inline]
pub fn update_glmvectors_by_family(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    likelihood.irls_update(y, eta, priorweights, mu, weights, z, None)
}

/// Compute first/second eta derivatives of the exact PIRLS working curvature
/// `W(eta)` on the same represented inverse-link surface as `update_glmvectors`.
///
/// Math note:
/// - `c[i]` and `d[i]` are classical derivatives of the diagonal PIRLS
///   curvature W_i(eta):
///     c_i = dW_i/dη_i,  d_i = d²W_i/dη_i².
/// - For canonical GLM families, these are the per-observation carriers of
///   higher likelihood derivatives (`-ℓ'''(η_i)` and `-ℓ''''(η_i)`) expressed
///   through the working-curvature map W(η).
/// - They are load-bearing in exact outer derivatives:
///   `c` enters dH/dρ (outer gradient), and `d` enters d²H/dρ² (outer Hessian).
/// Any row for which the exact carriers are not representable is refused.
pub(crate) fn computeworkingweight_derivatives_from_eta(
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
) -> Result<
    (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ),
    EstimationError,
> {
    let n = eta.len();
    let mut c = Array1::<f64>::zeros(n);
    let mut d = Array1::<f64>::zeros(n);
    let mut dmu_deta = Array1::<f64>::zeros(n);
    let mut d2mu_deta2 = Array1::<f64>::zeros(n);
    let mut d3mu_deta3 = Array1::<f64>::zeros(n);
    if let Some(cell) = GenericEdmCell::classify(&likelihood.spec.response, inverse_link) {
        write_generic_edm_eta_curvature(
            cell,
            fixed_glm_dispersion(likelihood)?,
            eta,
            priorweights,
            WorkingDerivativeBuffersMut {
                c: &mut c,
                d: &mut d,
                dmu_deta: &mut dmu_deta,
                d2mu_deta2: &mut d2mu_deta2,
                d3mu_deta3: &mut d3mu_deta3,
            },
        )?;
        return Ok((c, d, dmu_deta, d2mu_deta2, d3mu_deta3));
    }
    if let Some((standard, exponent)) = reciprocal_power_link(inverse_link) {
        let family = match likelihood.spec.response {
            ResponseFamily::Gaussian => PowerVarianceEdm::Gaussian,
            ResponseFamily::Gamma => PowerVarianceEdm::Gamma,
            ResponseFamily::InverseGaussian => PowerVarianceEdm::InverseGaussian,
            ref other => crate::bail_invalid_estim!(
                "the {standard:?} link is not legal for the {other:?} family"
            ),
        };
        write_reciprocal_link_eta_curvature(
            family,
            standard,
            exponent,
            fixed_glm_dispersion(likelihood)?,
            eta,
            priorweights,
            WorkingDerivativeBuffersMut {
                c: &mut c,
                d: &mut d,
                dmu_deta: &mut dmu_deta,
                d2mu_deta2: &mut d2mu_deta2,
                d3mu_deta3: &mut d3mu_deta3,
            },
        )?;
        return Ok((c, d, dmu_deta, d2mu_deta2, d3mu_deta3));
    }
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            dmu_deta.fill(1.0);
        }
        ResponseFamily::InverseGaussian => {
            log_link_working_state::write_log_link_eta_curvature(
                &inverse_gaussian_log_link_rule(fixed_glm_dispersion(likelihood)?),
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::StudentT { .. } => {
            // The EM weight `(ν+1)/(A + r²)` depends on the response, so this
            // is the one family whose working-weight jet needs `y`.
            let scale = StudentTScale::from_likelihood(likelihood)?;
            let (mut mu, mut weights, mut z) =
                (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
            write_student_t_working_state(
                y,
                eta,
                priorweights,
                &scale,
                &mut mu,
                &mut weights,
                &mut z,
                Some(WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                }),
            )?;
        }
        ResponseFamily::Poisson => {
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::PoissonIdentity,
                    curvature: log_link_working_state::WorkingCurvature::Proportional {
                        c_ratio: 1.0,
                        d_ratio: 1.0,
                    },
                },
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood)?;
            if !is_valid_tweedie_power(p) {
                crate::bail_invalid_estim!(
                    "Tweedie variance power must be finite and strictly between 1 and 2; got {p}",
                    p = p
                );
            }
            if !(phi.is_finite() && phi > 0.0) {
                crate::bail_invalid_estim!(
                    "Tweedie dispersion phi must be finite and > 0; got {phi}",
                    phi = phi
                );
            }
            let exponent = 2.0 - p;
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::PowerVariance { p, phi },
                    curvature: log_link_working_state::WorkingCurvature::Proportional {
                        c_ratio: exponent,
                        d_ratio: exponent * exponent,
                    },
                },
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            if !valid_negbin_theta(theta) {
                crate::bail_invalid_estim!(
                    "negative-binomial theta must be finite and > 0; got {theta}",
                    theta = theta
                );
            }
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::NegativeBinomial { theta },
                    curvature: log_link_working_state::WorkingCurvature::NegativeBinomial { theta },
                },
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            if !valid_beta_phi(phi) {
                crate::bail_invalid_estim!("beta-regression phi must be finite and > 0; got {phi}");
            }
            let certified: Vec<ExactBetaLogitRow> = super::par_certified_rows(eta.len(), |i| {
                exact_beta_logit_row(i, eta[i], None, priorweights[i], phi)
            })?;
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .zip(certified.par_iter())
                .for_each(|(((((c_o, d_o), dmu_o), d2_o), d3_o), row)| {
                    *c_o = row.c;
                    *d_o = row.d;
                    *dmu_o = row.dmu;
                    *d2_o = row.d2mu;
                    *d3_o = row.d3mu;
                });
        }
        ResponseFamily::Gamma => {
            // The Gamma log-link Fisher weight is independent of η, so the
            // working-curvature carriers `c`/`d` vanish identically (the kernel
            // returns `(0, 0)`); only the link jet is written here.
            log_link_working_state::write_log_link_eta_curvature(
                &log_link_working_state::LogLinkRule {
                    weight: log_link_working_state::WorkingWeight::Constant { factor: 1.0 },
                    curvature: log_link_working_state::WorkingCurvature::Proportional {
                        c_ratio: 0.0,
                        d_ratio: 0.0,
                    },
                },
                eta,
                priorweights,
                WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut dmu_deta,
                    d2mu_deta2: &mut d2mu_deta2,
                    d3mu_deta3: &mut d3mu_deta3,
                },
            )?;
        }
        ResponseFamily::Binomial => {
            let certified: Vec<CertifiedBernoulliRow> = super::par_certified_rows(eta.len(), |i| {
                let jet = if matches!(inverse_link, InverseLink::Standard(StandardLink::Logit)) {
                    let jet = logit_inverse_link_jet5(eta[i]);
                    MixtureInverseLinkJet {
                        mu: jet.mu,
                        d1: jet.d1,
                        d2: jet.d2,
                        d3: jet.d3,
                    }
                } else {
                    standard_inverse_link_jet(inverse_link, eta[i])?
                };
                certify_bernoulli_row(inverse_link, i, eta[i], jet.mu, priorweights[i])
            })?;
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .zip(certified.par_iter())
                .for_each(|(((((c_o, d_o), dmu_o), d2_o), d3_o), row)| {
                    *c_o = row.geometry.c;
                    *d_o = row.geometry.d;
                    *dmu_o = row.jet.d1;
                    *d2_o = row.jet.d2;
                    *d3_o = row.jet.d3;
                });
        }
        ResponseFamily::RoystonParmar => {
            crate::bail_invalid_estim!(
                "RoystonParmar is survival-specific and not a GLM IRLS family"
            );
        }
    }
    Ok((c, d, dmu_deta, d2mu_deta2, d3mu_deta3))
}

// General noncanonical observed-information weight corrections
//
// For an exponential-dispersion family with noncanonical link g, where
// h(η) = g⁻¹(η) is the inverse link and μ = h(η):
//
// Notation (all evaluated at a single observation):
//   h₁ = h'(η),  h₂ = h''(η),  h₃ = h'''(η),  h₄ = h''''(η)
//   V  = V(μ),   V₁ = V'(μ),   V₂ = V''(μ),    V₃ = V'''(μ)
//   φ  = dispersion parameter
//   pw = prior weight for this observation
//
// Fisher (expected) weight and its first two η-derivatives:
//   w_F = h₁² / (φV)
//   c_F = (2 h₁ h₂ V − h₁³ V₁) / (φ V²)
//   d_F = ∂c_F/∂η   (derived below)
//
// The observed weight subtracts a (y−μ)-dependent correction:
//   B   = (h₂ V − h₁² V₁) / (φ V²)
//   w_obs = w_F − (y−μ) · B
//
// First η-derivative of B:
//   B_η = (h₃ V² − 3 h₁ h₂ V V₁ − h₁³ V V₂ + 2 h₁³ V₁²) / (φ V³)
//
// Observed c (∂w_obs/∂η):
//   c_obs = c_F + h₁·B − (y−μ)·B_η
//
// Second η-derivative of B:
//   B_ηη = ∂B_η/∂η  (full expression in code below)
//
// Observed d (∂²w_obs/∂η²):
//   d_obs = d_F + h₂·B + 2 h₁·B_η − (y−μ)·B_ηη
//
// This function unifies all per-link hardcoded c/d computations: given the
// inverse-link jet (h₁…h₄) and the variance-function jet (V…V₃), it returns
// (w_obs, c_obs, d_obs) without any family- or link-specific dispatch.
