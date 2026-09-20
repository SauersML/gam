//! #932: the dispersion members' row programs against the hand derivatives they
//! replaced.
//!
//! The `retired_*` functions are the closed forms production ran from `bf64d52d8`
//! and `de14c2367` until the row programs replaced them, copied verbatim: the
//! η-space score and observed Hessian, the third-derivative tensor, and the row
//! kernel's working sets. The first test holds every production surface to them on
//! randomized rows of every member, shows the band resolves a one-part-per-million
//! change in the supplied stacks, and pins each declaration's value channel to the
//! row log-likelihood. The second races production's entry points against them.
#![cfg(test)]

use super::*;
use gam_math::paired_timing::{SpeedGate, paired_interleaved};
use statrs::function::gamma::ln_gamma;

fn retired_eta_loglik_second(
    kind: DispersionFamilyKind,
    yi: f64,
    em: f64,
    ed: f64,
) -> ([f64; 2], [f64; 3]) {
    use gam_math::special::{digamma, trigamma};
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp();
            let theta = ed.exp();
            let q = positive_share(mu, theta);
            let r = positive_share(theta, mu);
            let s = q * r;
            let log_r = log_positive_share(theta, mu);
            let psi_gap = digamma(theta + yi) - digamma(theta);
            let tri_gap = trigamma(theta + yi) - trigamma(theta);
            let total = theta + yi;
            let l_m = yi * r - theta * q;
            let l_d = theta * (psi_gap + log_r + q) - yi * r;
            let l_dd = theta * (psi_gap + log_r) + theta * theta * tri_gap + 2.0 * theta * q
                - total * s;
            ([l_m, l_d], [-total * s, total * s - theta * q, l_dd])
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp();
            let nu = ed.exp();
            let ratio = (1.0 / mu) * yi;
            let shape_score = ed + 1.0 - em - digamma(nu) + yi.ln() - ratio;
            let l_m = nu * (ratio - 1.0);
            let l_d = nu * shape_score;
            (
                [l_m, l_d],
                [-nu * ratio, l_m, l_d + nu - nu * nu * trigamma(nu)],
            )
        }
        DispersionFamilyKind::Beta => {
            let logit = gam_solve::mixture_link::logit_inverse_link_jet5(em);
            let (mu, d1, d2) = (logit.mu, logit.d1, logit.d2);
            let phi = ed.exp();
            let one_minus_mu = 1.0 - mu;
            let a = mu * phi;
            let b = one_minus_mu * phi;
            let psi_a = digamma(a);
            let psi_b = digamma(b);
            let tri_a = trigamma(a);
            let tri_b = trigamma(b);
            let ln_y = yi.ln();
            let ln_one_minus_y = (-yi).ln_1p();
            let k = psi_b - psi_a + ln_y - ln_one_minus_y;
            let cross = one_minus_mu * tri_b - mu * tri_a;
            let l_mu = phi * k;
            let l_phi = digamma(phi) - mu * psi_a - one_minus_mu * psi_b
                + mu * ln_y
                + one_minus_mu * ln_one_minus_y;
            let l_mumu = -phi * phi * (tri_a + tri_b);
            let l_muphi = k + phi * cross;
            let l_phiphi =
                trigamma(phi) - mu * mu * tri_a - one_minus_mu * one_minus_mu * tri_b;
            (
                [l_mu * d1, l_phi * phi],
                [
                    l_mumu * d1 * d1 + l_mu * d2,
                    l_muphi * d1 * phi,
                    l_phiphi * phi * phi + l_phi * phi,
                ],
            )
        }
        DispersionFamilyKind::Tweedie { p } => {
            let one_minus_p = 1.0 - p;
            let two_minus_p = 2.0 - p;
            let mu = em.exp();
            let kappa = ed.exp();
            let mu_two = mu.powf(two_minus_p);
            if yi > 0.0 {
                let mu_one = mu.powf(one_minus_p);
                let dev = 2.0
                    * (mu_two / two_minus_p - yi * mu_one / one_minus_p
                        + yi.powf(two_minus_p) / (one_minus_p * two_minus_p));
                let dev_m = 2.0 * (mu_two - yi * mu_one);
                let dev_mm = 2.0 * (two_minus_p * mu_two - one_minus_p * yi * mu_one);
                let half_kappa = 0.5 * kappa;
                (
                    [-half_kappa * dev_m, 0.5 - half_kappa * dev],
                    [-half_kappa * dev_mm, -half_kappa * dev_m, -half_kappa * dev],
                )
            } else {
                let c = mu_two / two_minus_p;
                (
                    [-kappa * mu_two, -kappa * c],
                    [-kappa * two_minus_p * mu_two, -kappa * mu_two, -kappa * c],
                )
            }
        }
    }
}

fn retired_eta_loglik_third(kind: DispersionFamilyKind, yi: f64, em: f64, ed: f64) -> [f64; 4] {
    use gam_math::special::{digamma, tetragamma, trigamma};
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp();
            let theta = ed.exp();
            let q = positive_share(mu, theta);
            let r = positive_share(theta, mu);
            let s = q * r;
            let log_r = log_positive_share(theta, mu);
            let psi_gap = digamma(theta + yi) - digamma(theta);
            let tri_gap = trigamma(theta + yi) - trigamma(theta);
            let tetra_gap = tetragamma(theta + yi) - tetragamma(theta);
            let total = theta + yi;
            let spread = total * s * (q - r);
            [
                spread,
                -theta * s - spread,
                -theta * q + 2.0 * theta * s + spread,
                theta * (psi_gap + log_r)
                    + 3.0 * theta * theta * tri_gap
                    + theta.powi(3) * tetra_gap
                    + 3.0 * theta * q
                    - 3.0 * theta * s
                    - spread,
            ]
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp();
            let nu = ed.exp();
            let ratio = (1.0 / mu) * yi;
            let shape_score = ed + 1.0 - em - digamma(nu) + yi.ln() - ratio;
            [
                nu * ratio,
                -nu * ratio,
                nu * (ratio - 1.0),
                nu * shape_score + 2.0 * nu
                    - 3.0 * nu * nu * trigamma(nu)
                    - nu.powi(3) * tetragamma(nu),
            ]
        }
        DispersionFamilyKind::Beta => {
            let logit = gam_solve::mixture_link::logit_inverse_link_jet5(em);
            let (mu, d1, d2, d3) = (logit.mu, logit.d1, logit.d2, logit.d3);
            let phi = ed.exp();
            let one_minus_mu = 1.0 - mu;
            let a = mu * phi;
            let b = one_minus_mu * phi;
            let psi_a = digamma(a);
            let psi_b = digamma(b);
            let tri_a = trigamma(a);
            let tri_b = trigamma(b);
            let tetra_a = tetragamma(a);
            let tetra_b = tetragamma(b);
            let ln_y = yi.ln();
            let ln_one_minus_y = (-yi).ln_1p();
            let k = psi_b - psi_a + ln_y - ln_one_minus_y;
            let cross = one_minus_mu * tri_b - mu * tri_a;
            let l_mu = phi * k;
            let l_phi = digamma(phi) - mu * psi_a - one_minus_mu * psi_b
                + mu * ln_y
                + one_minus_mu * ln_one_minus_y;
            let l_mumu = -phi * phi * (tri_a + tri_b);
            let l_muphi = k + phi * cross;
            let l_phiphi =
                trigamma(phi) - mu * mu * tri_a - one_minus_mu * one_minus_mu * tri_b;
            let l_mumumu = -phi.powi(3) * (tetra_a - tetra_b);
            let l_mumuphi =
                -2.0 * phi * (tri_a + tri_b) - phi * phi * (mu * tetra_a + one_minus_mu * tetra_b);
            let l_muphiphi = 2.0 * cross
                + phi * (one_minus_mu * one_minus_mu * tetra_b - mu * mu * tetra_a);
            let l_phiphiphi =
                tetragamma(phi) - mu.powi(3) * tetra_a - one_minus_mu.powi(3) * tetra_b;
            [
                l_mumumu * d1.powi(3) + 3.0 * l_mumu * d1 * d2 + l_mu * d3,
                (l_mumuphi * d1 * d1 + l_muphi * d2) * phi,
                (l_muphiphi * phi * phi + l_muphi * phi) * d1,
                l_phiphiphi * phi.powi(3) + 3.0 * l_phiphi * phi * phi + l_phi * phi,
            ]
        }
        DispersionFamilyKind::Tweedie { p } => {
            let one_minus_p = 1.0 - p;
            let two_minus_p = 2.0 - p;
            let mu = em.exp();
            let kappa = ed.exp();
            let mu_two = mu.powf(two_minus_p);
            if yi > 0.0 {
                let mu_one = mu.powf(one_minus_p);
                let dev = 2.0
                    * (mu_two / two_minus_p - yi * mu_one / one_minus_p
                        + yi.powf(two_minus_p) / (one_minus_p * two_minus_p));
                let dev_m = 2.0 * (mu_two - yi * mu_one);
                let dev_mm = 2.0 * (two_minus_p * mu_two - one_minus_p * yi * mu_one);
                let dev_mmm = 2.0
                    * (two_minus_p * two_minus_p * mu_two
                        - one_minus_p * one_minus_p * yi * mu_one);
                let half_kappa = 0.5 * kappa;
                [
                    -half_kappa * dev_mmm,
                    -half_kappa * dev_mm,
                    -half_kappa * dev_m,
                    -half_kappa * dev,
                ]
            } else {
                let c = mu_two / two_minus_p;
                [
                    -kappa * two_minus_p * two_minus_p * mu_two,
                    -kappa * two_minus_p * mu_two,
                    -kappa * mu_two,
                    -kappa * c,
                ]
            }
        }
    }
}

fn retired_observed_hessian_weights(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> (f64, f64, f64) {
    if prior_weight <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let (_, [l_mm, l_md, l_dd]) = retired_eta_loglik_second(kind, yi, eta_mu, eta_d);
    (
        -prior_weight * l_mm,
        -prior_weight * l_md,
        -prior_weight * l_dd,
    )
}

fn retired_observed_hessian_directional(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
    du_mu: f64,
    du_d: f64,
) -> (f64, f64, f64) {
    if prior_weight <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let [l_mmm, l_mmd, l_mdd, l_ddd] = retired_eta_loglik_third(kind, yi, eta_mu, eta_d);
    let scale = -prior_weight;
    (
        scale * (l_mmm * du_mu + l_mmd * du_d),
        scale * (l_mmd * du_mu + l_mdd * du_d),
        scale * (l_mdd * du_mu + l_ddd * du_d),
    )
}

fn retired_row_kernel(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> DispersionRowKernel {
    let em = eta_mu;
    let ed = eta_d;
    if prior_weight <= 0.0 {
        return DispersionRowKernel {
            loglik: 0.0,
            mean_weight: 0.0,
            mean_response: em,
            disp_weight: 0.0,
            disp_response: ed,
        };
    }
    let wi = prior_weight;
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp();
            let theta = ed.exp();
            let loglik = dispersion_nb_loglik(yi, mu, theta, wi);
            let mean_eta_information = if mu >= theta {
                theta / (1.0 + theta / mu)
            } else {
                mu / (1.0 + mu / theta)
            };
            let mean_weight = wi * mean_eta_information;
            let mean_response = em + (yi - mu) / mu;
            let theta_fraction = if theta >= mu {
                (mu / theta - yi / theta) / (1.0 + mu / theta)
            } else {
                (1.0 - yi / mu) / (1.0 + theta / mu)
            };
            let score_theta = gam_math::special::digamma(theta + yi)
                - gam_math::special::digamma(theta)
                + log_positive_share(theta, mu)
                + theta_fraction;
            let score_eta = theta * score_theta;
            // The helper evaluated ψ′(θ) itself when this was production.
            let eta_information =
                nb_log_precision_fisher_jensen(mu, theta, gam_math::special::trigamma(theta));
            let disp_weight = wi * eta_information;
            let disp_response = ed + score_eta / eta_information;
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp();
            let nu = ed.exp();
            let loglik = dispersion_gamma_loglik(yi, yi, mu, nu, wi);
            let s_nu = nu.ln() + 1.0 - mu.ln() - gam_math::special::digamma(nu) + yi.ln()
                - (1.0 / mu) * yi;
            let info_nu = gam_math::special::trigamma(nu) - nu.recip();
            let mean_weight = wi * nu;
            let mean_response = em + (yi - mu) / mu;
            let disp_weight = wi * nu * nu * info_nu;
            let disp_response = ed + s_nu / (nu * info_nu);
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
        DispersionFamilyKind::Beta => {
            let logit = gam_solve::mixture_link::logit_inverse_link_jet5(em);
            let mu = logit.mu;
            let phi = ed.exp();
            let q = logit.d1;
            let loglik = dispersion_beta_loglik(yi, mu, phi, wi);
            let one_minus_mu = 1.0 - mu;
            let a = mu * phi;
            let b = one_minus_mu * phi;
            let psi_a = gam_math::special::digamma(a);
            let psi_b = gam_math::special::digamma(b);
            let ln_y = yi.ln();
            let ln_one_minus_y = (-yi).ln_1p();
            let score_mu = phi * (psi_b - psi_a + ln_y - ln_one_minus_y);
            let s_phi = gam_math::special::digamma(phi) - mu * psi_a - one_minus_mu * psi_b
                + mu * ln_y
                + one_minus_mu * ln_one_minus_y;
            let tri_a = gam_math::special::trigamma(a);
            let tri_b = gam_math::special::trigamma(b);
            let tri_phi = gam_math::special::trigamma(phi);
            let info_mu = phi * phi * (tri_a + tri_b);
            let info_phi = mu * mu * tri_a + one_minus_mu * one_minus_mu * tri_b - tri_phi;
            let mean_weight = wi * q * q * info_mu;
            let mean_response = em + score_mu / (q * info_mu);
            let disp_weight = wi * phi * phi * info_phi;
            let disp_response = ed + s_phi / (phi * info_phi);
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
        DispersionFamilyKind::Tweedie { p } => {
            let mu = em.exp();
            let phi = (-ed).exp();
            let two_minus_p = 2.0 - p;
            let mean_weight = wi * mu.powf(two_minus_p) / phi;
            let mean_response = em + (yi - mu) / mu;
            let loglik = dispersion_tweedie_loglik(yi, em, ed, p, wi);
            let one_minus_p = 1.0 - p;
            let (s_eta, curvature_eta) = if yi > 0.0 {
                let dev = (mu.powf(two_minus_p) * (1.0 / two_minus_p)
                    - mu.powf(one_minus_p) * (yi / one_minus_p)
                    + yi.powf(two_minus_p) / (one_minus_p * two_minus_p))
                    * 2.0;
                (0.5 - 0.5 * dev / phi, 0.5)
            } else {
                let info = mu.powf(two_minus_p) * (1.0 / two_minus_p) / phi;
                (-info, info)
            };
            let disp_weight = wi * curvature_eta;
            let disp_response = ed + s_eta / curvature_eta;
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight,
                disp_response,
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Member {
    NegativeBinomial,
    Gamma,
    Beta,
    Tweedie,
}

const MEMBERS: [Member; 4] = [
    Member::NegativeBinomial,
    Member::Gamma,
    Member::Beta,
    Member::Tweedie,
];

impl Member {
    fn label(self) -> &'static str {
        match self {
            Member::NegativeBinomial => "negative_binomial",
            Member::Gamma => "gamma",
            Member::Beta => "beta",
            Member::Tweedie => "tweedie",
        }
    }
}

#[derive(Clone, Copy)]
struct Row {
    kind: DispersionFamilyKind,
    y: f64,
    eta_mu: f64,
    eta_d: f64,
    weight: f64,
    direction: [f64; 2],
}

/// Deterministic rows over the ranges the tower oracle tests draw
/// (`eta_space_row_program_derivatives_match_the_towers`); about three Tweedie
/// rows in ten are the point mass.
fn fixture(member: Member, count: usize, seed: u64) -> Vec<Row> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..count)
        .map(|_| {
            let eta_mu = -2.5 + 5.0 * next();
            let eta_d = -2.0 + 4.0 * next();
            let weight = 0.25 + 3.0 * next();
            let direction = [-1.0 + 2.0 * next(), -1.0 + 2.0 * next()];
            let (kind, y) = match member {
                Member::NegativeBinomial => (
                    DispersionFamilyKind::NegativeBinomial,
                    (12.0 * next()).floor(),
                ),
                Member::Gamma => (DispersionFamilyKind::Gamma, 0.01 + 8.0 * next()),
                Member::Beta => (DispersionFamilyKind::Beta, 0.005 + 0.99 * next()),
                Member::Tweedie => {
                    let p = 1.1 + 0.8 * next();
                    let point_mass = next() < 0.3;
                    let positive = 0.01 + 9.0 * next();
                    (
                        DispersionFamilyKind::Tweedie { p },
                        if point_mass { 0.0 } else { positive },
                    )
                }
            };
            Row {
                kind,
                y,
                eta_mu,
                eta_d,
                weight,
                direction,
            }
        })
        .collect()
}

/// `|a − b|` in units of the agreement band `1e-12 · max(1, |a|, |b|)`; a
/// non-finite side is infinitely far outside it.
fn in_bands(a: f64, b: f64) -> f64 {
    if !(a.is_finite() && b.is_finite()) {
        return f64::INFINITY;
    }
    (a - b).abs() / (1e-12 * a.abs().max(b.abs()).max(1.0))
}

const STACK_CHANNELS: [&str; 9] = [
    "score_mu",
    "score_d",
    "hessian_mm",
    "hessian_md",
    "hessian_dd",
    "drift_mm",
    "drift_md",
    "drift_dm",
    "drift_dd",
];

/// The unweighted log-likelihood surfaces of one row program: score, observed
/// Hessian, and the third derivative contracted along the row's direction.
fn program_surfaces(stacks: DispersionRowStacks, direction: &[f64; 2]) -> [f64; 9] {
    let (_, score, hessian) = stacks.order2();
    let drift = stacks.third_contracted(direction);
    [
        score[0],
        score[1],
        hessian[0][0],
        hessian[0][1],
        hessian[1][1],
        drift[0][0],
        drift[0][1],
        drift[1][0],
        drift[1][1],
    ]
}

fn retired_surfaces(row: Row) -> [f64; 9] {
    let ([l_m, l_d], [l_mm, l_md, l_dd]) =
        retired_eta_loglik_second(row.kind, row.y, row.eta_mu, row.eta_d);
    let [t_mmm, t_mmd, t_mdd, t_ddd] =
        retired_eta_loglik_third(row.kind, row.y, row.eta_mu, row.eta_d);
    let [u, v] = row.direction;
    [
        l_m,
        l_d,
        l_mm,
        l_md,
        l_dd,
        t_mmm * u + t_mmd * v,
        t_mmd * u + t_mdd * v,
        t_mmd * u + t_mdd * v,
        t_mdd * u + t_ddd * v,
    ]
}

/// The row's stacks with every supplied derivative entry and every jet-multiplying
/// value moved by one part per million: the smallest stack error the agreement
/// band has to resolve.
fn corrupted(mut stacks: DispersionRowStacks) -> DispersionRowStacks {
    let bump = 1.0 + 1e-6;
    match &mut stacks {
        DispersionRowStacks::NegativeBinomial {
            digamma_gap,
            trigamma_gap,
            tetragamma_gap,
            neg_log_theta_share,
            mu_share,
            theta_share,
            ..
        } => {
            *digamma_gap *= bump;
            *trigamma_gap *= bump;
            *tetragamma_gap *= bump;
            *neg_log_theta_share *= bump;
            *mu_share *= bump;
            *theta_share *= bump;
        }
        DispersionRowStacks::Gamma {
            response_ratio,
            digamma_shape,
            trigamma_shape,
            tetragamma_shape,
            ..
        } => {
            *response_ratio *= bump;
            *digamma_shape *= bump;
            *trigamma_shape *= bump;
            *tetragamma_shape *= bump;
        }
        DispersionRowStacks::Beta {
            mean_first,
            mean_second,
            mean_third,
            digamma_precision,
            trigamma_precision,
            tetragamma_precision,
            digamma_first_shape,
            trigamma_first_shape,
            tetragamma_first_shape,
            ..
        } => {
            *mean_first *= bump;
            *mean_second *= bump;
            *mean_third *= bump;
            *digamma_precision *= bump;
            *trigamma_precision *= bump;
            *tetragamma_precision *= bump;
            *digamma_first_shape *= bump;
            *trigamma_first_shape *= bump;
            *tetragamma_first_shape *= bump;
        }
        DispersionRowStacks::TweediePositive {
            mean_term,
            mean_first,
            mean_second,
            mean_third,
            response_term,
            response_first,
            response_second,
            response_third,
            deviance_offset,
            ..
        } => {
            *mean_term *= bump;
            *mean_first *= bump;
            *mean_second *= bump;
            *mean_third *= bump;
            *response_term *= bump;
            *response_first *= bump;
            *response_second *= bump;
            *response_third *= bump;
            *deviance_offset *= bump;
        }
        DispersionRowStacks::TweedieZero {
            mean_term,
            mean_first,
            mean_second,
            mean_third,
            ..
        } => {
            *mean_term *= bump;
            *mean_first *= bump;
            *mean_second *= bump;
            *mean_third *= bump;
        }
    }
    stacks
}

/// The row's stacks with the value entries production leaves at zero filled in,
/// so the order-2 surface's value is the row log-likelihood.
fn with_values(mut stacks: DispersionRowStacks, row: Row) -> DispersionRowStacks {
    match &mut stacks {
        DispersionRowStacks::NegativeBinomial {
            theta,
            count,
            ln_gamma_count,
            ln_gamma_gap,
            neg_log_mu_share,
            ..
        } => {
            *ln_gamma_count = ln_gamma(*count + 1.0);
            *ln_gamma_gap = gam_math::special::ln_gamma_shift_gap(*theta, *count);
            *neg_log_mu_share = -log_positive_share(row.eta_mu.exp(), *theta);
        }
        DispersionRowStacks::Gamma {
            shape,
            ln_gamma_shape,
            ..
        } => {
            *ln_gamma_shape = ln_gamma(*shape);
        }
        DispersionRowStacks::Beta {
            precision,
            mean,
            ln_gamma_precision,
            ln_gamma_first_shape,
            ln_gamma_second_shape,
            ..
        } => {
            *ln_gamma_precision = ln_gamma(*precision);
            *ln_gamma_first_shape = ln_gamma(*mean * *precision);
            *ln_gamma_second_shape = ln_gamma((1.0 - *mean) * *precision);
        }
        DispersionRowStacks::TweediePositive { log_normalizer, .. } => {
            // A Tweedie stack comes only from a Tweedie row; any other kind poisons
            // the value channel instead of passing silently.
            let power = match row.kind {
                DispersionFamilyKind::Tweedie { p } => p,
                _ => f64::NAN,
            };
            *log_normalizer = 0.5 * row.eta_d
                - 0.5 * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * power * row.y.ln();
        }
        DispersionRowStacks::TweedieZero { .. } => {}
    }
    stacks
}

const ENTRY_CHANNELS: [&str; 16] = [
    "weights_mm",
    "weights_md",
    "weights_dd",
    "directional_mm",
    "directional_md",
    "directional_dd",
    "alo_score_mu",
    "alo_score_d",
    "alo_hessian_mm",
    "alo_hessian_md",
    "alo_hessian_dd",
    "kernel_loglik",
    "kernel_mean_weight",
    "kernel_mean_response",
    "kernel_disp_weight",
    "kernel_disp_response",
];

/// Worst disagreement, in bands, between production's entry points and the
/// retired hand ones over `rows`, per entry channel.
fn entry_point_worst(rows: &[Row]) -> [f64; 16] {
    let mut worst = [0.0_f64; 16];
    for row in rows {
        let Row {
            kind,
            y,
            eta_mu,
            eta_d,
            weight,
            direction: [du_mu, du_d],
        } = *row;
        let weights = dispersion_row_observed_hessian_weights(kind, y, eta_mu, eta_d, weight);
        let retired_weights = retired_observed_hessian_weights(kind, y, eta_mu, eta_d, weight);
        let directional =
            dispersion_row_observed_hessian_directional(kind, y, eta_mu, eta_d, weight, du_mu, du_d);
        let retired_directional =
            retired_observed_hessian_directional(kind, y, eta_mu, eta_d, weight, du_mu, du_d);
        let geometry = dispersion_alo_row_geometry(kind, 0, y, eta_mu, eta_d, weight)
            .expect("fixture rows are representable");
        let ([l_m, l_d], [l_mm, l_md, l_dd]) = retired_eta_loglik_second(kind, y, eta_mu, eta_d);
        let kernel = dispersion_row_kernel(kind, y, eta_mu, eta_d, weight);
        let retired_kernel = retired_row_kernel(kind, y, eta_mu, eta_d, weight);
        let pairs = [
            (weights.0, retired_weights.0),
            (weights.1, retired_weights.1),
            (weights.2, retired_weights.2),
            (directional.0, retired_directional.0),
            (directional.1, retired_directional.1),
            (directional.2, retired_directional.2),
            (geometry.nll_score[0], -weight * l_m),
            (geometry.nll_score[1], -weight * l_d),
            (geometry.observed_hessian[0][0], -weight * l_mm),
            (geometry.observed_hessian[0][1], -weight * l_md),
            (geometry.observed_hessian[1][1], -weight * l_dd),
            (kernel.loglik, retired_kernel.loglik),
            (kernel.mean_weight, retired_kernel.mean_weight),
            (kernel.mean_response, retired_kernel.mean_response),
            (kernel.disp_weight, retired_kernel.disp_weight),
            (kernel.disp_response, retired_kernel.disp_response),
        ];
        for (channel, (production, retired)) in pairs.into_iter().enumerate() {
            worst[channel] = worst[channel].max(in_bands(production, retired));
        }
    }
    worst
}

/// Every production surface of every member reproduces the retired hand
/// derivatives to roundoff, the band that says so resolves a one-part-per-million
/// stack error, and each declaration's value channel is the row log-likelihood.
#[test]
fn dispersion_row_programs_match_the_retired_hand_derivatives_932() {
    let mut failures = Vec::new();
    for (index, member) in MEMBERS.into_iter().enumerate() {
        let rows = fixture(member, 400, 0x932_D15_0000 + index as u64);
        let name = member.label();
        let mut agreement = [0.0_f64; 9];
        let mut corruption = [0.0_f64; 9];
        let mut value = 0.0_f64;
        for row in &rows {
            let stacks = DispersionRowStacks::at(row.kind, row.y, row.eta_mu, row.eta_d, 3);
            let retired = retired_surfaces(*row);
            let program = program_surfaces(stacks, &row.direction);
            let corrupt = program_surfaces(corrupted(stacks), &row.direction);
            for channel in 0..9 {
                agreement[channel] = agreement[channel].max(in_bands(program[channel], retired[channel]));
                corruption[channel] =
                    corruption[channel].max(in_bands(corrupt[channel], retired[channel]));
            }
            let (program_value, _, _) = with_values(stacks, *row).order2();
            let loglik = dispersion_row_loglik(row.kind, row.y, row.eta_mu, row.eta_d, 1.0);
            value = value.max(in_bands(program_value, loglik));
        }
        let entry = entry_point_worst(&rows);
        for (channel, label) in STACK_CHANNELS.into_iter().enumerate() {
            eprintln!(
                "DISPERSION-ROW-PROGRAM-932 member={name} channel={label} \
                 worst_in_bands={:.3e} one_ppm_corruption_in_bands={:.3e}",
                agreement[channel], corruption[channel]
            );
            if !(agreement[channel] <= 1.0) {
                failures.push(format!("{name} {label}: {:.3e} bands", agreement[channel]));
            }
            if !(corruption[channel] > 1.0) {
                failures.push(format!(
                    "{name} {label}: a one-ppm stack corruption stayed inside the band \
                     ({:.3e} bands)",
                    corruption[channel]
                ));
            }
        }
        for (channel, label) in ENTRY_CHANNELS.into_iter().enumerate() {
            eprintln!(
                "DISPERSION-ROW-PROGRAM-932 member={name} entry={label} worst_in_bands={:.3e}",
                entry[channel]
            );
            if !(entry[channel] <= 1.0) {
                failures.push(format!("{name} {label}: {:.3e} bands", entry[channel]));
            }
        }
        eprintln!(
            "DISPERSION-ROW-PROGRAM-932 member={name} value_vs_row_loglik_in_bands={value:.3e}"
        );
        if !(value <= 1.0) {
            failures.push(format!("{name} value channel: {value:.3e} bands"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// The passes are outlined and generic over the entry point, so each arm's entry
// inlines into its own row loop as it does in production's, and neither arm pays an
// indirect call the other does not.
#[inline(never)]
fn weights_pass<F>(rows: &[Row], nudge: f64, entry: F) -> f64
where
    F: Fn(DispersionFamilyKind, f64, f64, f64, f64) -> (f64, f64, f64),
{
    let mut fold = 0.0;
    for row in rows {
        let (mm, md, dd) = entry(row.kind, row.y, row.eta_mu + nudge, row.eta_d, row.weight);
        fold += mm + md + dd;
    }
    fold
}

#[inline(never)]
fn directional_pass<F>(rows: &[Row], nudge: f64, entry: F) -> f64
where
    F: Fn(DispersionFamilyKind, f64, f64, f64, f64, f64, f64) -> (f64, f64, f64),
{
    let mut fold = 0.0;
    for row in rows {
        let [du_mu, du_d] = row.direction;
        let (mm, md, dd) = entry(
            row.kind,
            row.y,
            row.eta_mu + nudge,
            row.eta_d,
            row.weight,
            du_mu,
            du_d,
        );
        fold += mm + md + dd;
    }
    fold
}

#[inline(never)]
fn kernel_pass<F>(rows: &[Row], nudge: f64, entry: F) -> f64
where
    F: Fn(DispersionFamilyKind, f64, f64, f64, f64) -> DispersionRowKernel,
{
    let mut fold = 0.0;
    for row in rows {
        let kernel = entry(row.kind, row.y, row.eta_mu + nudge, row.eta_d, row.weight);
        fold += kernel.loglik
            + kernel.mean_weight
            + kernel.mean_response
            + kernel.disp_weight
            + kernel.disp_response;
    }
    fold
}

/// SPEC rule 1: production's generated dispersion lowerings must match or beat the
/// hand derivatives they replaced. Each cell races one production entry point
/// against its retired twin over 64 rows per pass, consuming every channel. The
/// contract is `not_slower`: a loss is a median ratio below one by more than the
/// measurement's own resolution.
///
/// The arms are the two shipped paths, not the same stacks. Production reads each
/// argument's polygamma entries from one recurrence (`polygamma_stack`), and the
/// row kernel reads its Fisher information's `ψ′` from the score's stack. The
/// retired forms call the per-order scalars, which walk the recurrence once per
/// order. The Tweedie cells evaluate no polygamma at all, so they race the
/// generated arithmetic against the hand arithmetic on equal transcendental
/// calls.
#[test]
fn dispersion_row_programs_are_not_slower_than_the_retired_hand_932() {
    let fixtures: Vec<(Member, Vec<Row>)> = MEMBERS
        .into_iter()
        .enumerate()
        .map(|(index, member)| (member, fixture(member, 64, 0x932_D15_1000 + index as u64)))
        .collect();
    // Parity on the timed rows runs in every build, before the gate opens.
    for (member, rows) in &fixtures {
        let name = member.label();
        let worst = entry_point_worst(rows);
        assert!(
            worst.iter().all(|bands| *bands <= 1.0),
            "{name}: timed rows disagree with the retired hand ({worst:?} bands)"
        );
    }
    if cfg!(debug_assertions) {
        return;
    }
    let mut gate = SpeedGate::open("DISPERSION-ROW-PROGRAM-932");
    let reps = 15usize;
    let passes = 256usize;
    for (index, (member, rows)) in fixtures.iter().enumerate() {
        let name = member.label();
        let seed = 0x932_D15_2000 + 16 * index as u64;
        for (surface, timing) in [
            (
                "observed_hessian",
                paired_interleaved(
                    reps,
                    passes,
                    seed,
                    |nudge| weights_pass(rows, nudge, dispersion_row_observed_hessian_weights),
                    |nudge| weights_pass(rows, nudge, retired_observed_hessian_weights),
                ),
            ),
            (
                "hessian_directional",
                paired_interleaved(
                    reps,
                    passes,
                    seed + 1,
                    |nudge| {
                        directional_pass(rows, nudge, dispersion_row_observed_hessian_directional)
                    },
                    |nudge| directional_pass(rows, nudge, retired_observed_hessian_directional),
                ),
            ),
            (
                "row_kernel",
                paired_interleaved(
                    reps,
                    passes,
                    seed + 2,
                    |nudge| kernel_pass(rows, nudge, dispersion_row_kernel),
                    |nudge| kernel_pass(rows, nudge, retired_row_kernel),
                ),
            ),
        ] {
            // `median_ratio` is retired / production: above 1 means the row
            // program is faster.
            gate.not_slower(
                &format!("member={name} surface={surface} rows={}", rows.len()),
                &timing,
                "row_program",
                "retired_hand",
            );
        }
    }
    gate.finish();
}
