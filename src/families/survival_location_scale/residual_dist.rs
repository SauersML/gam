use super::*;

/// Layer 2 defense: compute q0 = -eta_t * exp(-eta_ls) with log-space
/// overflow detection.  When log|q0| = ln|eta_t| + (-eta_ls) exceeds the
/// clamp ceiling, the product would overflow; we saturate to ±MAX instead.
#[inline]
pub(crate) fn survival_q0_from_eta(eta_t: f64, eta_ls: f64) -> f64 {
    if eta_t == 0.0 {
        return 0.0;
    }
    let log_abs = eta_t.abs().ln() + (-eta_ls).min(EXP_NEG_STABLE_MAX_ARG);
    if log_abs > EXP_NEG_STABLE_MAX_ARG {
        if eta_t > 0.0 { -f64::MAX } else { f64::MAX }
    } else {
        -eta_t * exp_sigma_inverse_from_eta_scalar(eta_ls)
    }
}

#[inline]
pub(crate) fn probit_survival_value(eta: f64) -> f64 {
    if eta.is_nan() {
        f64::NAN
    } else if eta == f64::INFINITY {
        0.0
    } else if eta == f64::NEG_INFINITY {
        1.0
    } else {
        0.5 * erfc(eta / std::f64::consts::SQRT_2)
    }
}

#[inline]
pub(crate) fn probit_log_survival_and_ratio_derivatives(eta: f64) -> (f64, f64, f64, f64, f64) {
    if eta.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if eta == f64::NEG_INFINITY {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let x = eta / std::f64::consts::SQRT_2;
    let (log_survival, ratio) = if eta >= 0.0 {
        // erfcx(x) = exp(x²)·erfc(x); compute once and reuse for both
        // log-survival and the hazard ratio.
        let erfcx_val = erfcx_nonnegative(x);
        let log_surv = -0.5 * eta * eta + (0.5 * erfcx_val).ln();
        let r = std::f64::consts::FRAC_2_SQRT_PI / (std::f64::consts::SQRT_2 * erfcx_val);
        (log_surv, r)
    } else {
        let survival = probit_survival_value(eta);
        (survival.ln(), normal_pdf(eta) / survival)
    };
    let dr = ratio * (ratio - eta);
    let ddr = 2.0 * ratio.powi(3) - 3.0 * eta * ratio.powi(2) + (eta * eta - 1.0) * ratio;
    let dddr = 6.0 * ratio.powi(4) - 12.0 * eta * ratio.powi(3)
        + (7.0 * eta * eta - 4.0) * ratio.powi(2)
        + (-eta * eta * eta + 3.0 * eta) * ratio;
    (log_survival, ratio, dr, ddr, dddr)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResidualDistribution {
    Gaussian,
    Gumbel,
    Logistic,
}

pub trait ResidualDistributionOps {
    fn cdf(&self, z: f64) -> f64;
    fn pdf(&self, z: f64) -> f64;
    fn pdf_derivative(&self, z: f64) -> f64;
    fn pdfsecond_derivative(&self, z: f64) -> f64;
    fn pdfthird_derivative(&self, z: f64) -> f64;

    /// Fourth derivative of the residual-distribution PDF, f''''(z).
    ///
    /// This is the m4 ingredient for the outer REML Hessian's Q[v_k, v_l] term.
    /// The second directional derivative of the inner Hessian (used by the outer
    /// Hessian drift) requires the 4th derivative of the composed likelihood
    /// F_αβγδ via the Arbogast chain rule. That chain rule's leading term
    /// m4·u_α·u_β·u_γ·u_δ needs this quantity.
    ///
    /// See response.md Section 6 for the mathematical derivation.
    fn pdffourth_derivative(&self, z: f64) -> f64;
}

impl ResidualDistributionOps for ResidualDistribution {
    fn cdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_cdf(z),
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).mu
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).mu
            }
        }
    }

    fn pdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_pdf(z),
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).d1
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).d1
            }
        }
    }

    fn pdf_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => -z * normal_pdf(z),
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).d2
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).d2
            }
        }
    }

    fn pdfsecond_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                (z * z - 1.0) * f
            }
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).d3
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).d3
            }
        }
    }

    fn pdfthird_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                -(z * z * z - 3.0 * z) * f
            }
            ResidualDistribution::Gumbel => inverse_link_pdfthird_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::CLogLog),
                z,
            )
            .expect("standard cloglog inverse-link third derivative should evaluate"),
            ResidualDistribution::Logistic => inverse_link_pdfthird_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::Logit),
                z,
            )
            .expect("standard logit inverse-link third derivative should evaluate"),
        }
    }

    /// Fourth derivative of the residual-distribution PDF.
    ///
    /// # Derivations
    ///
    /// **Gaussian**: f(z) = φ(z). The n-th derivative of the Gaussian PDF is
    /// (-1)^n He_n(z) φ(z) where He_n is the probabilist's Hermite polynomial.
    /// He_4(z) = z⁴ - 6z² + 3, so f''''(z) = (z⁴ - 6z² + 3) φ(z).
    ///
    /// **Logistic**: f(z) = s(1-s) with s = σ(z). The k-th derivative of f is
    /// f · P_k(s) where P_k satisfies the Euler-polynomial recurrence
    /// P_{k+1}(s) = (1-2s) P_k(s) + s(1-s) P_k'(s).
    /// P_4(s) = 1 - 30s + 150s² - 240s³ + 120s⁴.
    ///
    /// **Gumbel**: f(z) = exp(z - e^z). Let e = e^z. The k-th derivative of f
    /// is f · Q_k(e) where Q_k satisfies Q_{k+1}(e) = (1-e) Q_k(e) + e Q_k'(e).
    /// Q_4(e) = 1 - 15e + 25e² - 10e³ + e⁴.
    fn pdffourth_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                let z2 = z * z;
                // He_4(z) = z^4 - 6z^2 + 3
                (z2 * z2 - 6.0 * z2 + 3.0) * f
            }
            ResidualDistribution::Gumbel => inverse_link_pdffourth_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::CLogLog),
                z,
            )
            .expect("standard cloglog inverse-link fourth derivative should evaluate"),
            ResidualDistribution::Logistic => inverse_link_pdffourth_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::Logit),
                z,
            )
            .expect("standard logit inverse-link fourth derivative should evaluate"),
        }
    }
}

#[inline]
pub(crate) fn residual_distribution_link(distribution: ResidualDistribution) -> StandardLink {
    match distribution {
        ResidualDistribution::Gaussian => StandardLink::Probit,
        ResidualDistribution::Gumbel => StandardLink::CLogLog,
        ResidualDistribution::Logistic => StandardLink::Logit,
    }
}

#[inline]
pub fn residual_distribution_inverse_link(distribution: ResidualDistribution) -> InverseLink {
    InverseLink::Standard(residual_distribution_link(distribution))
}

/// Maps an `InverseLink` to its `ResidualDistribution` counterpart when the
/// link is one of the three standard survival residual-distribution links
/// (Probit/Logit/CLogLog). Returns `None` for stateful / mixture links (Sas,
/// BetaLogistic, Mixture, LatentCLogLog) and for non-residual-distribution
/// standard links — those carry their full state via `payload.link` and have
/// no `ResidualDistribution` representation.
#[inline]
pub fn residual_distribution_from_inverse_link(link: &InverseLink) -> Option<ResidualDistribution> {
    match link {
        InverseLink::Standard(StandardLink::Probit) => Some(ResidualDistribution::Gaussian),
        InverseLink::Standard(StandardLink::CLogLog) => Some(ResidualDistribution::Gumbel),
        InverseLink::Standard(StandardLink::Logit) => Some(ResidualDistribution::Logistic),
        _ => None,
    }
}

/// Fourth derivative of the inverse-link PDF (= 5th derivative of the CDF).
///
/// This is the f'''' quantity used in the 4th derivative of log f(u), which
/// in turn enters the m4 ingredient of the Arbogast chain rule for
/// the outer REML Hessian Q[v_k, v_l] term.
///
/// For the three standard survival residual distributions (Probit, Logit,
/// CLogLog), uses the closed-form ResidualDistribution implementations.
/// For all other inverse links (SAS, BetaLogistic, Mixture), delegates
/// to the generic `inverse_link_pdffourth_derivative_for_inverse_link`
/// dispatcher in mixture_link.rs.
pub(crate) fn inverse_link_pdffourth_derivative(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    match inverse_link {
        InverseLink::Standard(StandardLink::Probit) => {
            Ok(ResidualDistribution::Gaussian.pdffourth_derivative(eta))
        }
        InverseLink::Standard(StandardLink::Logit) => {
            Ok(ResidualDistribution::Logistic.pdffourth_derivative(eta))
        }
        InverseLink::Standard(StandardLink::CLogLog) => {
            Ok(ResidualDistribution::Gumbel.pdffourth_derivative(eta))
        }
        _ => crate::solver::mixture_link::inverse_link_pdffourth_derivative_for_inverse_link(
            inverse_link,
            eta,
        )
        .map_err(|e| SurvivalLocationScaleError::NumericalFailure {
            reason: format!("inverse link fourth-derivative evaluation failed at eta={eta}: {e}"),
        }),
    }
}
