use gam::families::family_runtime::{FamilyStrategy, strategy_for_spec};
use gam::solver::pirls::update_glmvectors_by_family;
use gam::types::{GlmLikelihoodSpec, LikelihoodSpec, ResponseFamily};
use ndarray::{Array1, Array2, array};
use statrs::function::gamma::ln_gamma;

fn loglik(y: &Array1<f64>, eta: &Array1<f64>, spec: &GlmLikelihoodSpec, w: &Array1<f64>) -> f64 {
    let strategy = strategy_for_spec(&spec.spec);
    let mu = strategy
        .inverse_link_array(eta.view())
        .expect("inverse-link evaluation must succeed on closed-form GLM links");
    match spec.spec.response {
        ResponseFamily::Gaussian => y
            .iter()
            .zip(mu.iter())
            .zip(w.iter())
            .map(|((&yi, &mi), &wi)| -0.5 * wi * (yi - mi) * (yi - mi))
            .sum(),
        ResponseFamily::Binomial => y
            .iter()
            .zip(mu.iter())
            .zip(w.iter())
            .map(|((&yi, &mi), &wi)| {
                wi * (yi * mi.max(1e-15).ln() + (1.0 - yi) * (1.0 - mi).max(1e-15).ln())
            })
            .sum(),
        ResponseFamily::Poisson => y
            .iter()
            .zip(mu.iter())
            .zip(w.iter())
            .map(|((&yi, &mi), &wi)| wi * (yi * mi.max(1e-15).ln() - mi))
            .sum(),
        ResponseFamily::Tweedie { p } => y
            .iter()
            .zip(mu.iter())
            .zip(w.iter())
            .map(|((&yi, &mi), &wi)| {
                wi * (yi * mi.powf(1.0 - p) / (1.0 - p) - mi.powf(2.0 - p) / (2.0 - p))
            })
            .sum(),
        ResponseFamily::NegativeBinomial { theta, .. } => y
            .iter()
            .zip(mu.iter())
            .zip(w.iter())
            .map(|((&yi, &mi), &wi)| {
                wi * (yi * (mi.max(1e-15) / (mi + theta)).ln()
                    + theta * (theta / (mi + theta)).ln())
            })
            .sum(),
        // Full Beta(a, b) log-density with a = mu*phi, b = (1-mu)*phi (up to
        // the mu-independent -ln y - ln(1-y) constant, whose derivative w.r.t.
        // beta is zero and so does not affect the finite-difference comparison
        // against the PIRLS score). Includes the log-Beta normalizer
        // ln Gamma(phi) - ln Gamma(a) - ln Gamma(b); without it, the
        // derivative w.r.t. mu is missing the digamma(a) - digamma(b) terms
        // that PIRLS's Fisher-scoring Beta update correctly carries.
        ResponseFamily::Beta { phi } => y
            .iter()
            .zip(mu.iter())
            .zip(w.iter())
            .map(|((&yi, &mi), &wi)| {
                let a = mi * phi;
                let b = (1.0 - mi) * phi;
                let log_norm = ln_gamma(phi) - ln_gamma(a) - ln_gamma(b);
                wi * (log_norm
                    + (a - 1.0) * yi.max(1e-15).ln()
                    + (b - 1.0) * (1.0 - yi).max(1e-15).ln())
            })
            .sum(),
        ResponseFamily::Gamma => y
            .iter()
            .zip(mu.iter())
            .zip(w.iter())
            .map(|((&yi, &mi), &wi)| wi * (-yi / mi.max(1e-15) - mi.max(1e-15).ln()))
            .sum(),
        _ => unreachable!(),
    }
}

