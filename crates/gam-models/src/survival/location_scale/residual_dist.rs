use super::*;
use gam_math::probability::{normal_cdf, normal_logcdf_derivatives};

// Layer 2 defense (canonical implementation lives beside the σ-link in
// `gam_model_kernels::sigma_link` so the fit engine and the prediction
// crates share one guarded product): q0 = -eta_t · exp(-eta_ls) with exact
// log-space overflow detection, saturating only past f64 representability.
pub(crate) use crate::sigma_link::survival_q0_from_eta;

#[inline]
pub(crate) fn probit_survival_value(eta: f64) -> f64 {
    if eta.is_nan() {
        f64::NAN
    } else if eta == f64::INFINITY {
        0.0
    } else if eta == f64::NEG_INFINITY {
        1.0
    } else {
        normal_cdf(-eta)
    }
}

#[inline]
pub(crate) fn probit_log_survival_and_ratio_derivatives(eta: f64) -> (f64, f64, f64, f64, f64) {
    // log S(eta) = log Phi(-eta). If d_k is the kth derivative of log Phi at
    // -eta, the hazard ratio and its first three eta derivatives are
    // d_1, -d_2, d_3, -d_4. The shared stack owns all tail cancellation and
    // representability policy, including the exact infinite limits.
    let d = normal_logcdf_derivatives(-eta);
    (d[0], d[1], -d[2], d[3], -d[4])
}

#[cfg(test)]
mod probit_tail_tests {
    use super::probit_log_survival_and_ratio_derivatives;

    #[test]
    fn probit_survival_derivatives_have_exact_infinite_limits() {
        assert_eq!(
            probit_log_survival_and_ratio_derivatives(f64::NEG_INFINITY),
            (0.0, 0.0, -0.0, 0.0, -0.0)
        );
        assert_eq!(
            probit_log_survival_and_ratio_derivatives(f64::INFINITY),
            (f64::NEG_INFINITY, f64::INFINITY, 1.0, 0.0, -0.0)
        );
        let nan_derivatives = probit_log_survival_and_ratio_derivatives(f64::NAN);
        assert!(
            [
                nan_derivatives.0,
                nan_derivatives.1,
                nan_derivatives.2,
                nan_derivatives.3,
                nan_derivatives.4,
            ]
            .into_iter()
            .all(f64::is_nan)
        );
    }

    #[test]
    fn probit_survival_derivatives_preserve_both_extreme_tails() {
        let (_, ratio, dr, ddr, dddr) = probit_log_survival_and_ratio_derivatives(1.0e100);
        assert_eq!(ratio, 1.0e100);
        assert_eq!(dr, 1.0);
        assert!(ddr > 0.0 && ddr.is_finite());
        assert_eq!(dddr, -0.0);

        let (_, ratio, dr, ddr, dddr) = probit_log_survival_and_ratio_derivatives(-38.6);
        assert_eq!(ratio, 0.0);
        assert!(dr > 0.0 && dr.is_subnormal());
        assert!(ddr > 0.0 && ddr.is_subnormal());
        assert!(dddr > 0.0 && dddr.is_subnormal());
    }
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

impl ResidualDistribution {
    /// The inverse-link component whose CDF is this residual law.
    #[inline]
    fn link_component(self) -> gam_problem::LinkComponent {
        match self {
            ResidualDistribution::Gaussian => gam_problem::LinkComponent::Probit,
            ResidualDistribution::Gumbel => gam_problem::LinkComponent::CLogLog,
            ResidualDistribution::Logistic => gam_problem::LinkComponent::Logit,
        }
    }

    #[inline]
    fn jet(self, z: f64) -> gam_solve::mixture_link::InverseLinkJet {
        component_inverse_link_jet(self.link_component(), z)
    }
}

/// Every residual law is the CDF of a standard inverse link, so each
/// derivative is read from the shared inverse-link jet and dispatchers in
/// `gam_solve::mixture_link` (the tail-stable closed forms the row kernel
/// already uses), never from a second hand-written copy here.
impl ResidualDistributionOps for ResidualDistribution {
    fn cdf(&self, z: f64) -> f64 {
        self.jet(z).mu
    }

    fn pdf(&self, z: f64) -> f64 {
        self.jet(z).d1
    }

    fn pdf_derivative(&self, z: f64) -> f64 {
        self.jet(z).d2
    }

    fn pdfsecond_derivative(&self, z: f64) -> f64 {
        self.jet(z).d3
    }

    fn pdfthird_derivative(&self, z: f64) -> f64 {
        inverse_link_pdfthird_derivative_for_inverse_link(
            &residual_distribution_inverse_link(*self),
            z,
        )
        .expect("standard residual inverse-link third derivative is total")
    }

    /// Fourth derivative of the residual-distribution PDF.
    ///
    /// Gaussian: `(z⁴ − 6z² + 3) φ(z)` (probabilist's Hermite `He_4`).
    /// Logistic: `f · (1 − 30s + 150s² − 240s³ + 120s⁴)` with `s = σ(z)`.
    /// Gumbel: `f · (1 − 15e + 25e² − 10e³ + e⁴)` with `e = exp(z)`.
    fn pdffourth_derivative(&self, z: f64) -> f64 {
        inverse_link_pdffourth_derivative_for_inverse_link(
            &residual_distribution_inverse_link(*self),
            z,
        )
        .expect("standard residual inverse-link fourth derivative is total")
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
