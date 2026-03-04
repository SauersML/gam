use crate::probability::{normal_cdf_approx, normal_pdf};
use crate::types::{LinkComponent, MixtureLinkSpec, MixtureLinkState, SasLinkSpec, SasLinkState};
use ndarray::Array1;

const PROB_EPS: f64 = 1e-8;
const ETA_CLAMP_GENERAL: f64 = 30.0;
const ETA_CLAMP_LOGIT: f64 = 700.0;
const SAS_U_CLAMP: f64 = 50.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseLinkJet {
    pub mu: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MixtureJetWithRhoPartials {
    pub jet: InverseLinkJet,
    /// Partial derivatives wrt free logits rho_j, j in [0, K-2].
    /// Each entry stores derivatives of (mu, d1, d2, d3) wrt one rho_j.
    pub djet_drho: Vec<InverseLinkJet>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SasJetWithParamPartials {
    pub jet: InverseLinkJet,
    pub djet_depsilon: InverseLinkJet,
    pub djet_dlog_delta: InverseLinkJet,
}

#[inline]
fn clamp_prob(p: f64) -> f64 {
    p.clamp(PROB_EPS, 1.0 - PROB_EPS)
}

pub fn state_from_sas_spec(spec: SasLinkSpec) -> Result<SasLinkState, String> {
    if !spec.initial_epsilon.is_finite() || !spec.initial_log_delta.is_finite() {
        return Err("SAS link parameters must be finite".to_string());
    }
    let log_delta = spec.initial_log_delta.clamp(-12.0, 12.0);
    let delta = log_delta.exp();
    Ok(SasLinkState {
        epsilon: spec.initial_epsilon,
        log_delta,
        delta,
    })
}

pub fn validate_mixture_spec(spec: &MixtureLinkSpec) -> Result<(), String> {
    if spec.components.len() < 2 {
        return Err("mixture link requires at least 2 components".to_string());
    }
    if spec.initial_rho.len() + 1 != spec.components.len() {
        return Err(format!(
            "mixture link rho length mismatch: expected {}, got {}",
            spec.components.len() - 1,
            spec.initial_rho.len()
        ));
    }
    for i in 0..spec.components.len() {
        for j in (i + 1)..spec.components.len() {
            if spec.components[i] == spec.components[j] {
                return Err("mixture link components must be unique".to_string());
            }
        }
    }
    Ok(())
}

pub fn softmax_last_fixed_zero(rho: &Array1<f64>) -> Array1<f64> {
    let k = rho.len() + 1;
    let mut logits = Vec::with_capacity(k);
    let mut max_v = 0.0_f64;
    for &v in rho {
        max_v = max_v.max(v);
        logits.push(v);
    }
    max_v = max_v.max(0.0);
    logits.push(0.0);

    let mut sum = 0.0_f64;
    let mut exps = vec![0.0_f64; k];
    for i in 0..k {
        let e = (logits[i] - max_v).exp();
        exps[i] = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Array1::from_elem(k, 1.0 / k as f64);
    }
    let inv = 1.0 / sum;
    Array1::from_iter(exps.into_iter().map(|v| v * inv))
}

/// Returns softmax weights and Jacobian wrt free logits (last logit fixed at zero).
/// Jacobian shape is (K, K-1): d pi_k / d rho_j.
pub fn softmax_with_jacobian_last_fixed_zero(
    rho: &Array1<f64>,
) -> (Array1<f64>, ndarray::Array2<f64>) {
    let pi = softmax_last_fixed_zero(rho);
    let k = pi.len();
    let m = k.saturating_sub(1);
    let mut jac = ndarray::Array2::<f64>::zeros((k, m));
    for j in 0..m {
        let pi_j = pi[j];
        for kk in 0..k {
            let delta = if kk == j { 1.0 } else { 0.0 };
            jac[[kk, j]] = pi[kk] * (delta - pi_j);
        }
    }
    (pi, jac)
}

pub fn state_from_spec(spec: &MixtureLinkSpec) -> Result<MixtureLinkState, String> {
    validate_mixture_spec(spec)?;
    let pi = softmax_last_fixed_zero(&spec.initial_rho);
    Ok(MixtureLinkState {
        components: spec.components.clone(),
        rho: spec.initial_rho.clone(),
        pi,
    })
}

#[inline]
pub fn component_inverse_link_jet(component: LinkComponent, eta: f64) -> InverseLinkJet {
    match component {
        LinkComponent::Logit => {
            let e = eta.clamp(-ETA_CLAMP_LOGIT, ETA_CLAMP_LOGIT);
            let mu = 1.0 / (1.0 + (-e).exp());
            let d1 = mu * (1.0 - mu);
            let d2 = d1 * (1.0 - 2.0 * mu);
            let d3 = d1 * (1.0 - 6.0 * d1);
            InverseLinkJet {
                mu: clamp_prob(mu),
                d1,
                d2,
                d3,
            }
        }
        LinkComponent::Probit => {
            let e = eta.clamp(-ETA_CLAMP_GENERAL, ETA_CLAMP_GENERAL);
            let d1 = normal_pdf(e);
            InverseLinkJet {
                mu: clamp_prob(normal_cdf_approx(e)),
                d1,
                d2: -e * d1,
                d3: (e * e - 1.0) * d1,
            }
        }
        LinkComponent::CLogLog => {
            let e = eta.clamp(-ETA_CLAMP_GENERAL, ETA_CLAMP_GENERAL);
            let t = e.exp();
            let s = (-t).exp();
            let d1 = t * s;
            InverseLinkJet {
                mu: clamp_prob(1.0 - s),
                d1,
                d2: d1 * (1.0 - t),
                d3: d1 * (1.0 - 3.0 * t + t * t),
            }
        }
        LinkComponent::LogLog => {
            let e = eta.clamp(-ETA_CLAMP_GENERAL, ETA_CLAMP_GENERAL);
            let r = (-e).exp();
            let mu = (-r).exp();
            let d1 = mu * r;
            let d2 = d1 * (r - 1.0);
            let d3 = d1 * (r * r - 3.0 * r + 1.0);
            InverseLinkJet {
                mu: clamp_prob(mu),
                d1,
                d2,
                d3,
            }
        }
        LinkComponent::Cauchit => {
            let e = eta;
            let den = 1.0 + e * e;
            let d1 = 1.0 / (std::f64::consts::PI * den);
            let d2 = -2.0 * e / (std::f64::consts::PI * den * den);
            let d3 = (6.0 * e * e - 2.0) / (std::f64::consts::PI * den * den * den);
            InverseLinkJet {
                mu: clamp_prob(0.5 + e.atan() / std::f64::consts::PI),
                d1,
                d2,
                d3,
            }
        }
    }
}

#[inline]
pub fn mixture_inverse_link_jet(state: &MixtureLinkState, eta: f64) -> InverseLinkJet {
    let mut mu = 0.0_f64;
    let mut d1 = 0.0_f64;
    let mut d2 = 0.0_f64;
    let mut d3 = 0.0_f64;
    let k = state.components.len().min(state.pi.len());
    for i in 0..k {
        let jet = component_inverse_link_jet(state.components[i], eta);
        let w = state.pi[i];
        mu += w * jet.mu;
        d1 += w * jet.d1;
        d2 += w * jet.d2;
        d3 += w * jet.d3;
    }
    InverseLinkJet {
        mu: clamp_prob(mu),
        d1: d1.max(1e-12),
        d2,
        d3,
    }
}

/// Computes mixture jet and exact partial derivatives wrt free softmax logits.
///
/// Uses identities:
///   d mu     / d rho_j = pi_j (mu_j     - mu)
///   d mu'    / d rho_j = pi_j (mu_j'    - mu')
///   d mu''   / d rho_j = pi_j (mu_j''   - mu'')
///   d mu'''  / d rho_j = pi_j (mu_j'''  - mu''')
pub fn mixture_inverse_link_jet_with_rho_partials(
    state: &MixtureLinkState,
    eta: f64,
) -> MixtureJetWithRhoPartials {
    let k = state.components.len().min(state.pi.len());
    let m = k.saturating_sub(1);
    let mut djet_drho = vec![
        InverseLinkJet {
            mu: 0.0,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
        };
        m
    ];
    let jet = mixture_inverse_link_jet_with_rho_partials_into(state, eta, &mut djet_drho);
    MixtureJetWithRhoPartials { jet, djet_drho }
}

/// Computes mixture jet and writes exact rho partial jets into `out` (length >= K-1).
/// This avoids heap allocation in hot loops.
pub fn mixture_inverse_link_jet_with_rho_partials_into(
    state: &MixtureLinkState,
    eta: f64,
    out: &mut [InverseLinkJet],
) -> InverseLinkJet {
    let k = state.components.len().min(state.pi.len());
    let m = k.saturating_sub(1);
    debug_assert!(
        out.len() >= m,
        "rho-partial output buffer too small: got {}, need {}",
        out.len(),
        m
    );
    let mut mixed = InverseLinkJet {
        mu: 0.0,
        d1: 0.0,
        d2: 0.0,
        d3: 0.0,
    };
    for i in 0..k {
        let jet_i = component_inverse_link_jet(state.components[i], eta);
        let w = state.pi[i];
        mixed.mu += w * jet_i.mu;
        mixed.d1 += w * jet_i.d1;
        mixed.d2 += w * jet_i.d2;
        mixed.d3 += w * jet_i.d3;
    }
    mixed.mu = clamp_prob(mixed.mu);
    mixed.d1 = mixed.d1.max(1e-12);
    for j in 0..m {
        let pi_j = state.pi[j];
        let cj = component_inverse_link_jet(state.components[j], eta);
        out[j] = InverseLinkJet {
            mu: pi_j * (cj.mu - mixed.mu),
            d1: pi_j * (cj.d1 - mixed.d1),
            d2: pi_j * (cj.d2 - mixed.d2),
            d3: pi_j * (cj.d3 - mixed.d3),
        };
    }
    mixed
}

/// SAS inverse-link jet for:
///   mu(eta) = Phi(sinh(delta * asinh(eta) - epsilon)), delta = exp(log_delta).
pub fn sas_inverse_link_jet(eta: f64, epsilon: f64, log_delta: f64) -> InverseLinkJet {
    let e = if eta.is_finite() { eta } else { 0.0 };
    let a = e.asinh();
    let ld = log_delta.clamp(-12.0, 12.0);
    let delta = ld.exp();
    let u_raw = delta * a - epsilon;
    let u = u_raw.clamp(-SAS_U_CLAMP, SAS_U_CLAMP);
    let u_active = if (u - u_raw).abs() < 1e-15 { 1.0 } else { 0.0 };
    let z = u.sinh();
    let phi = normal_pdf(z);
    let q = (1.0 + e * e).sqrt();
    let u1 = u_active * delta / q;
    let u2 = u_active * (-delta * e / (q * q * q));
    let u3 = u_active * (delta * (2.0 * e * e - 1.0) / (q * q * q * q * q));
    let s = u.sinh();
    let c = u.cosh();
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + 3.0 * s * u1 * u2 + c * u3;
    let mu = normal_cdf_approx(z);
    let d1 = phi * z1;
    let d2 = phi * (z2 - z * z1 * z1);
    let d3 = phi * (z3 - 3.0 * z * z1 * z2 + (z * z - 1.0) * z1 * z1 * z1);
    InverseLinkJet {
        mu: clamp_prob(mu),
        d1: d1.max(1e-12),
        d2,
        d3,
    }
}

pub fn sas_inverse_link_jet_with_param_partials(
    eta: f64,
    epsilon: f64,
    log_delta: f64,
) -> SasJetWithParamPartials {
    let e = if eta.is_finite() { eta } else { 0.0 };
    let a = e.asinh();
    let ld_raw = log_delta;
    let ld = ld_raw.clamp(-12.0, 12.0);
    let ld_active = if (ld - ld_raw).abs() < 1e-15 {
        1.0
    } else {
        0.0
    };
    let delta = ld.exp();
    let u_raw = delta * a - epsilon;
    let u = u_raw.clamp(-SAS_U_CLAMP, SAS_U_CLAMP);
    let u_active = if (u - u_raw).abs() < 1e-15 { 1.0 } else { 0.0 };
    let z = u.sinh();
    let phi = normal_pdf(z);

    let q = (1.0 + e * e).sqrt();
    let u1 = u_active * delta / q;
    let u2 = u_active * (-delta * e / (q * q * q));
    let u3 = u_active * (delta * (2.0 * e * e - 1.0) / (q * q * q * q * q));
    let s = u.sinh();
    let c = u.cosh();
    let z1 = c * u1;
    let z2 = s * u1 * u1 + c * u2;
    let z3 = c * u1 * u1 * u1 + 3.0 * s * u1 * u2 + c * u3;

    let base = InverseLinkJet {
        mu: clamp_prob(normal_cdf_approx(z)),
        d1: (phi * z1).max(1e-12),
        d2: phi * (z2 - z * z1 * z1),
        d3: phi * (z3 - 3.0 * z * z1 * z2 + (z * z - 1.0) * z1 * z1 * z1),
    };

    // Generic chain for parameter t:
    // u_t, u1_t, u2_t, u3_t -> z_t,z1_t,z2_t,z3_t -> mu_t,d1_t,d2_t,d3_t
    let param_partials = |u_t: f64, u1_t: f64, u2_t: f64, u3_t: f64| -> InverseLinkJet {
        let z_t = c * u_t;
        let z1_t = s * u_t * u1 + c * u1_t;
        let z2_t = c * u_t * u1 * u1 + 2.0 * s * u1 * u1_t + s * u_t * u2 + c * u2_t;
        let z3_t = s * u_t * u1 * u1 * u1
            + 3.0 * c * u1 * u1 * u1_t
            + 3.0 * c * u_t * u1 * u2
            + 3.0 * s * (u1_t * u2 + u1 * u2_t)
            + s * u_t * u3
            + c * u3_t;

        let phi_t = -z * phi * z_t;
        let mu_t = phi * z_t;
        let d1_t = phi_t * z1 + phi * z1_t;

        let k2 = z2 - z * z1 * z1;
        let k2_t = z2_t - z_t * z1 * z1 - z * 2.0 * z1 * z1_t;
        let d2_t = phi_t * k2 + phi * k2_t;

        let k3 = z3 - 3.0 * z * z1 * z2 + (z * z - 1.0) * z1 * z1 * z1;
        let k3_t = z3_t - 3.0 * (z_t * z1 * z2 + z * z1_t * z2 + z * z1 * z2_t)
            + 2.0 * z * z_t * z1 * z1 * z1
            + (z * z - 1.0) * 3.0 * z1 * z1 * z1_t;
        let d3_t = phi_t * k3 + phi * k3_t;

        InverseLinkJet {
            mu: mu_t,
            d1: d1_t,
            d2: d2_t,
            d3: d3_t,
        }
    };

    // epsilon partials: delta fixed, u_eps = -1.
    let djet_depsilon = param_partials(-u_active, 0.0, 0.0, 0.0);
    // log-delta partials: only active when unclamped.
    let u_ld = ld_active * u_active * delta * a;
    let djet_dlog_delta = param_partials(u_ld, ld_active * u1, ld_active * u2, ld_active * u3);

    SasJetWithParamPartials {
        jet: base,
        djet_depsilon,
        djet_dlog_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LinkComponent;

    #[test]
    fn softmax_jacobian_matches_fd() {
        let rho = Array1::from_vec(vec![0.7, -1.2, 0.4]);
        let (pi, jac) = softmax_with_jacobian_last_fixed_zero(&rho);
        let h = 1e-6;
        for j in 0..rho.len() {
            let mut rp = rho.clone();
            rp[j] += h;
            let mut rm = rho.clone();
            rm[j] -= h;
            let pp = softmax_last_fixed_zero(&rp);
            let pm = softmax_last_fixed_zero(&rm);
            let fd = (&pp - &pm).mapv(|v| v / (2.0 * h));
            for k in 0..pi.len() {
                let err = (jac[[k, j]] - fd[k]).abs();
                assert!(err < 5e-6, "jac mismatch at ({k},{j}): err={err:e}");
            }
        }
    }

    #[test]
    fn mixture_jet_rho_partials_match_fd() {
        let spec = MixtureLinkSpec {
            components: vec![
                LinkComponent::Probit,
                LinkComponent::Logit,
                LinkComponent::CLogLog,
                LinkComponent::Cauchit,
            ],
            initial_rho: Array1::from_vec(vec![0.3, -0.6, 0.2]),
        };
        let state = state_from_spec(&spec).expect("state");
        let eta = 0.35;
        let out = mixture_inverse_link_jet_with_rho_partials(&state, eta);
        let h = 1e-6;
        for j in 0..state.rho.len() {
            let mut rp = state.rho.clone();
            rp[j] += h;
            let sp = MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: rp,
            };
            let jp = mixture_inverse_link_jet(&state_from_spec(&sp).expect("sp"), eta);
            let mut rm = state.rho.clone();
            rm[j] -= h;
            let sm = MixtureLinkSpec {
                components: state.components.clone(),
                initial_rho: rm,
            };
            let jm = mixture_inverse_link_jet(&state_from_spec(&sm).expect("sm"), eta);
            let fd = InverseLinkJet {
                mu: (jp.mu - jm.mu) / (2.0 * h),
                d1: (jp.d1 - jm.d1) / (2.0 * h),
                d2: (jp.d2 - jm.d2) / (2.0 * h),
                d3: (jp.d3 - jm.d3) / (2.0 * h),
            };
            let an = out.djet_drho[j];
            assert!((an.mu - fd.mu).abs() < 1e-6);
            assert!((an.d1 - fd.d1).abs() < 1e-6);
            assert!((an.d2 - fd.d2).abs() < 1e-6);
            assert!((an.d3 - fd.d3).abs() < 1e-6);
        }
    }

    #[test]
    fn sas_param_partials_match_fd() {
        let eta = 0.37;
        let epsilon = -0.12;
        let log_delta = 0.21;
        let out = sas_inverse_link_jet_with_param_partials(eta, epsilon, log_delta);
        let h = 1e-6;

        let ep_p = sas_inverse_link_jet(eta, epsilon + h, log_delta);
        let ep_m = sas_inverse_link_jet(eta, epsilon - h, log_delta);
        let fd_ep = InverseLinkJet {
            mu: (ep_p.mu - ep_m.mu) / (2.0 * h),
            d1: (ep_p.d1 - ep_m.d1) / (2.0 * h),
            d2: (ep_p.d2 - ep_m.d2) / (2.0 * h),
            d3: (ep_p.d3 - ep_m.d3) / (2.0 * h),
        };
        assert!((out.djet_depsilon.mu - fd_ep.mu).abs() < 5e-5);
        assert!((out.djet_depsilon.d1 - fd_ep.d1).abs() < 5e-5);
        assert!((out.djet_depsilon.d2 - fd_ep.d2).abs() < 5e-5);
        assert!((out.djet_depsilon.d3 - fd_ep.d3).abs() < 5e-4);

        let ld_p = sas_inverse_link_jet(eta, epsilon, log_delta + h);
        let ld_m = sas_inverse_link_jet(eta, epsilon, log_delta - h);
        let fd_ld = InverseLinkJet {
            mu: (ld_p.mu - ld_m.mu) / (2.0 * h),
            d1: (ld_p.d1 - ld_m.d1) / (2.0 * h),
            d2: (ld_p.d2 - ld_m.d2) / (2.0 * h),
            d3: (ld_p.d3 - ld_m.d3) / (2.0 * h),
        };
        assert!((out.djet_dlog_delta.mu - fd_ld.mu).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d1 - fd_ld.d1).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d2 - fd_ld.d2).abs() < 5e-5);
        assert!((out.djet_dlog_delta.d3 - fd_ld.d3).abs() < 5e-4);
    }

    #[test]
    fn sas_jet_extreme_inputs_stay_finite() {
        let cases = [
            (-1e6, 0.0, 0.0),
            (1e6, 0.0, 0.0),
            (3.0, 12.0, 12.0),
            (-3.0, -12.0, -12.0),
            (0.5, 40.0, 10.0),
            (0.5, -40.0, -10.0),
        ];
        for (eta, eps, log_delta) in cases {
            let j = sas_inverse_link_jet(eta, eps, log_delta);
            assert!(j.mu.is_finite());
            assert!(j.d1.is_finite());
            assert!(j.d2.is_finite());
            assert!(j.d3.is_finite());
            let p = sas_inverse_link_jet_with_param_partials(eta, eps, log_delta);
            assert!(p.djet_depsilon.mu.is_finite());
            assert!(p.djet_depsilon.d1.is_finite());
            assert!(p.djet_depsilon.d2.is_finite());
            assert!(p.djet_depsilon.d3.is_finite());
            assert!(p.djet_dlog_delta.mu.is_finite());
            assert!(p.djet_dlog_delta.d1.is_finite());
            assert!(p.djet_dlog_delta.d2.is_finite());
            assert!(p.djet_dlog_delta.d3.is_finite());
        }
    }

    #[test]
    fn sas_param_partials_zero_when_u_clamped() {
        // Pick a setup where u = delta*asinh(eta) - epsilon saturates at +SAS_U_CLAMP.
        let eta = 10.0;
        let epsilon = -60.0;
        let log_delta = 8.0;
        let j = sas_inverse_link_jet_with_param_partials(eta, epsilon, log_delta);
        // With u clamp active, all eta-derivatives and param partials are masked to stable zeros.
        assert!(j.djet_depsilon.mu.abs() < 1e-10);
        assert!(j.djet_depsilon.d1.abs() < 1e-10);
        assert!(j.djet_depsilon.d2.abs() < 1e-10);
        assert!(j.djet_depsilon.d3.abs() < 1e-10);
        assert!(j.djet_dlog_delta.mu.abs() < 1e-10);
        assert!(j.djet_dlog_delta.d1.abs() < 1e-10);
        assert!(j.djet_dlog_delta.d2.abs() < 1e-10);
        assert!(j.djet_dlog_delta.d3.abs() < 1e-10);
    }

    #[test]
    fn sas_log_delta_partials_zero_when_log_delta_clamped() {
        // log_delta clamps to +12 internally; derivative wrt log_delta is masked.
        let eta = 0.7;
        let epsilon = 0.2;
        let log_delta = 40.0;
        let j = sas_inverse_link_jet_with_param_partials(eta, epsilon, log_delta);
        assert!(j.djet_dlog_delta.mu.abs() < 1e-10);
        assert!(j.djet_dlog_delta.d1.abs() < 1e-10);
        assert!(j.djet_dlog_delta.d2.abs() < 1e-10);
        assert!(j.djet_dlog_delta.d3.abs() < 1e-10);
    }

    #[test]
    fn sas_eta_jets_match_fd() {
        let eta = -0.43;
        let epsilon = 0.27;
        let log_delta = -0.31;
        let h = 1e-5;
        let j0 = sas_inverse_link_jet(eta, epsilon, log_delta);
        let jp = sas_inverse_link_jet(eta + h, epsilon, log_delta);
        let jm = sas_inverse_link_jet(eta - h, epsilon, log_delta);
        let d1_fd = (jp.mu - jm.mu) / (2.0 * h);
        let d2_fd = (jp.d1 - jm.d1) / (2.0 * h);
        let d3_fd = (jp.d2 - jm.d2) / (2.0 * h);
        assert!((j0.d1 - d1_fd).abs() < 5e-5);
        assert!((j0.d2 - d2_fd).abs() < 2e-4);
        assert!((j0.d3 - d3_fd).abs() < 1e-3);
    }
}
