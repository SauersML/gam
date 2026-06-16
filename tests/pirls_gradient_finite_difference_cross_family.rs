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

#[test]
fn hunter() {
    let x: Array2<f64> = array![[1.0, -0.5], [1.0, 0.4], [1.0, 1.2], [1.0, -1.0], [1.0, 0.8]];
    let prior = Array1::ones(5);
    let beta = array![0.2, -0.3];
    let families = vec![
        (
            "Gaussian",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::gaussian_identity()),
            array![0.2, -0.1, 0.5, -0.3, 0.7],
        ),
        (
            "Binomial-Logit",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::binomial_logit()),
            array![0.0, 1.0, 1.0, 0.0, 1.0],
        ),
        (
            "Binomial-Probit",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::binomial_probit()),
            array![0.0, 1.0, 1.0, 0.0, 1.0],
        ),
        (
            "Binomial-CLogLog",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::binomial_cloglog()),
            array![0.0, 1.0, 1.0, 0.0, 1.0],
        ),
        (
            "Poisson",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::poisson_log()),
            array![1.0, 2.0, 3.0, 1.0, 2.0],
        ),
        (
            "Tweedie-1.5",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::tweedie_log(1.5)),
            array![0.8, 1.5, 2.2, 0.6, 1.7],
        ),
        (
            "Tweedie-1.7",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::tweedie_log(1.7)),
            array![0.8, 1.5, 2.2, 0.6, 1.7],
        ),
        (
            "NegBin-2.0",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::negative_binomial_log(2.0)),
            array![1.0, 2.0, 3.0, 1.0, 2.0],
        ),
        (
            "Beta-10",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::beta_logit(10.0)),
            array![0.2, 0.7, 0.6, 0.3, 0.8],
        ),
        (
            "Gamma",
            GlmLikelihoodSpec::canonical(LikelihoodSpec::gamma_log()),
            array![0.8, 1.5, 2.2, 0.6, 1.7],
        ),
    ];
    let h = 1e-6;
    for (name, spec, y) in families {
        let eta = x.dot(&beta);
        let mut mu = Array1::zeros(5);
        let mut ww = Array1::zeros(5);
        let mut z = Array1::zeros(5);
        update_glmvectors_by_family(
            y.view(),
            &eta,
            &spec,
            prior.view(),
            &mut mu,
            &mut ww,
            &mut z,
        )
        .unwrap();
        let score = &ww * (&z - &eta);
        let g = x.t().dot(&score);
        for j in 0..2 {
            let mut bp = beta.clone();
            bp[j] += h;
            let mut bm = beta.clone();
            bm[j] -= h;
            let fp = loglik(&y, &x.dot(&bp), &spec, &prior);
            let fm = loglik(&y, &x.dot(&bm), &spec, &prior);
            let fd = (fp - fm) / (2.0 * h);
            if (g[j] - fd).abs() > 1e-5 {
                panic!("{} fails at {} analytic={} fd={}", name, j, g[j], fd);
            }
        }
    }
}
