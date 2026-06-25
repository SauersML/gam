//! Regression for #1125 (and its NB scalar sibling #1124): a dispersion
//! location-scale fit — a `--predict-noise` / `noise_formula` smooth on an
//! overdispersed family (Gamma / Negative-Binomial / Beta / Tweedie) — learns a
//! genuine per-row precision surface `exp(eta_d(x))`. `gam generate` /
//! `sample_replicates` must thread that per-row dispersion into the generative
//! observation model so synthetic data reproduces the fitted *non-constant*
//! dispersion. The bug was that every dispersion-LS family fell into the scalar
//! `else` branch of `run_generate_unified`, so the whole `eta_d(x)` surface was
//! dropped and synthetic data came out homoscedastic at the seed dispersion.
//!
//! ## The invariant under test (the root cause, from a deterministic angle)
//!
//! The model's *predict* path already honours the per-row precision: its
//! `predict_noise_scale` builds `sqrt(Var(y))` from `precision = exp(eta_d)`
//! per the family mean–variance law. The *generate* path must agree: the
//! per-row dispersion `predict_dispersion_scale` hands to the generative
//! `NoiseModel` must, when run back through that NoiseModel's own variance law,
//! reproduce exactly the same per-row `Var(y)` the predict path reports.
//!
//! This is the literal statement of the bug — predict used the channel, generate
//! dropped it — so asserting `Var_generate(x) == Var_predict(x)` per row is the
//! root-cause check. It is exercised here for ALL FOUR dispersion-LS families
//! through the production predictor methods (`predict_dispersion_scale`,
//! `predict_noise_scale`, `NoiseModel::from_likelihood_with_per_row_dispersion`)
//! — the exact composition `run_generate_unified`'s DispersionLocationScale
//! branch performs — with no Monte-Carlo noise.
//!
//! It specifically pins the units conversion that is easy to get wrong:
//! NB θ, Gamma shape and Beta φ ARE the precision `exp(eta_d)` directly, while
//! Tweedie φ is its RECIPROCAL (`Var = φ·μ^p`, precision `= 1/φ`). A units slip
//! in any arm (most plausibly forgetting the Tweedie reciprocal) makes the
//! generate-side variance diverge from the predict-side variance and trips this
//! test.

use gam::estimate::BlockRole;
use gam::gamlss::DispersionFamilyKind;
use gam::generative::NoiseModel;
use gam_predict::{DispersionLocationScalePredictor, PredictInput, PredictableModel};
use gam::smooth::build_term_collection_design;
use gam::types::LikelihoodSpec;
use gam::{
    DispersionLocationScaleFitResult, FitConfig, FitResult, encode_recordswith_inferred_schema,
    fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// Deterministic seeded uniform in [0,1) (Numerical Recipes LCG, high bits) with
/// Gamma / Poisson / Beta / Tweedie draws, so the dataset is byte-reproducible
/// without depending on the platform RNG.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Gamma(shape k>0, scale) via Marsaglia–Tsang (Ahrens–Dieter boost k<1).
    fn gamma(&mut self, k: f64, scale: f64) -> f64 {
        if k < 1.0 {
            let g = self.gamma(k + 1.0, scale);
            let u = self.unit().max(1e-300);
            return g * u.powf(1.0 / k);
        }
        let d = k - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1 = self.unit().max(1e-300);
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit().max(1e-300);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * scale;
            }
        }
    }
    fn poisson(&mut self, lambda: f64) -> f64 {
        let l = (-lambda).exp();
        let mut k = 0.0;
        let mut p = 1.0;
        loop {
            p *= self.unit();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
    /// NB2(mu, theta): lambda ~ Gamma(theta, mu/theta), y ~ Poisson(lambda).
    fn negbin(&mut self, mu: f64, theta: f64) -> f64 {
        let lambda = self.gamma(theta, mu / theta);
        self.poisson(lambda)
    }
    /// Beta(mu, phi): X ~ Gamma(mu*phi,1), Y ~ Gamma((1-mu)*phi,1), X/(X+Y).
    fn beta(&mut self, mu: f64, phi: f64) -> f64 {
        let a = self.gamma((mu * phi).max(1e-6), 1.0);
        let b = self.gamma(((1.0 - mu) * phi).max(1e-6), 1.0);
        (a / (a + b)).clamp(1e-6, 1.0 - 1e-6)
    }
    /// Tweedie(mu, phi, p), 1<p<2, via the compound Poisson–Gamma representation.
    fn tweedie(&mut self, mu: f64, phi: f64, p: f64) -> f64 {
        let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
        let alpha = (2.0 - p) / (p - 1.0);
        let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
        let n = self.poisson(lambda) as usize;
        (0..n).map(|_| self.gamma(alpha, scale)).sum()
    }
}

/// Which dispersion-LS family a scenario exercises.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fam {
    Gamma,
    NegBin,
    Beta,
    Tweedie,
}

/// One dispersion-LS scenario: family config + truth surfaces + a draw.
struct Scenario {
    name: &'static str,
    family: &'static str,
    fam: Fam,
    /// The seed-spec the picker / NoiseModel see (carries the construction seed,
    /// not the fitted dispersion — exactly what the generate path presents).
    likelihood: LikelihoodSpec,
}

/// The production `kind` must be the dispersion family this scenario fits. The
/// Tweedie power `p` is checked loosely (any positive power) because the data
/// power and the fitted power need not coincide for the variance-agreement
/// invariant, which is self-consistent within the predictor.
fn kind_matches(kind: &DispersionFamilyKind, fam: Fam) -> bool {
    matches!(
        (kind, fam),
        (DispersionFamilyKind::Gamma, Fam::Gamma)
            | (DispersionFamilyKind::NegativeBinomial, Fam::NegBin)
            | (DispersionFamilyKind::Beta, Fam::Beta)
            | (DispersionFamilyKind::Tweedie { .. }, Fam::Tweedie)
    )
}

/// The generative `NoiseModel`'s own per-row variance law, evaluated at the mean
/// `mu` and the per-row dispersion the NoiseModel actually carries. This is the
/// analytic variance of `sampleobservations` for each family (the sampler is
/// covered separately in the `generative` unit tests) — what synthetic draws
/// would empirically converge to.
fn noise_model_variance(noise: &NoiseModel, mu: &Array1<f64>) -> Array1<f64> {
    match noise {
        NoiseModel::NegativeBinomial { theta } => {
            Array1::from_shape_fn(mu.len(), |i| mu[i] + mu[i] * mu[i] / theta[i])
        }
        NoiseModel::Gamma { shape } => {
            Array1::from_shape_fn(mu.len(), |i| mu[i] * mu[i] / shape[i])
        }
        NoiseModel::Beta { phi } => {
            Array1::from_shape_fn(mu.len(), |i| mu[i] * (1.0 - mu[i]) / (1.0 + phi[i]))
        }
        NoiseModel::Tweedie { p, phi } => {
            Array1::from_shape_fn(mu.len(), |i| phi[i] * mu[i].powf(*p))
        }
        other => panic!("unexpected NoiseModel for a dispersion-LS family: {other:?}"),
    }
}

fn run_scenario(s: &Scenario) {
    init_parallelism();

    // ---- deterministic heteroscedastic-dispersion data (seed varies per fam) -
    let n = 600usize;
    let mut rng = Lcg(0x5125 ^ (s.name.len() as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let x: Vec<f64> = (0..n)
        .map(|i| -1.5 + 3.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| match s.fam {
            Fam::Gamma => {
                let mu = (0.6 + 0.4 * xi).exp();
                let shape = (1.0 - 0.8 * xi).exp(); // precision falls with x
                rng.gamma(shape, mu / shape)
            }
            Fam::NegBin => {
                let mu = (0.7 + 0.5 * xi).exp();
                let theta = (1.2 + 0.9 * xi).exp(); // overdispersion shrinks with x
                rng.negbin(mu, theta)
            }
            Fam::Beta => {
                let mu = 1.0 / (1.0 + (-(0.2 + 0.5 * xi)).exp());
                let phi = (1.6 + 1.0 * xi).exp(); // precision rises with x
                rng.beta(mu, phi)
            }
            Fam::Tweedie => {
                let mu = (0.5 + 0.4 * xi).exp();
                let phi = (-0.4 + 0.8 * xi).exp(); // dispersion rises with x
                rng.tweedie(mu, phi, 1.5)
            }
        })
        .collect();

    // ---- gam dispersion location-scale fit (noise_formula on the dispersion) -
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records)
        .unwrap_or_else(|e| panic!("[{}] encode data: {e}", s.name));
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some(s.family.to_string()),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=6)", &ds, &cfg)
        .unwrap_or_else(|e| panic!("[{}] dispersion-LS fit failed: {e}", s.name));
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("[{}] expected a DispersionLocationScale fit result", s.name);
    };
    assert!(
        kind_matches(&kind, s.fam),
        "[{}] routed to the wrong dispersion family: {kind:?}",
        s.name
    );

    let beta_mu = fit
        .fit
        .block_by_role(BlockRole::Location)
        .unwrap_or_else(|| panic!("[{}] missing location block", s.name))
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .unwrap_or_else(|| panic!("[{}] missing scale block", s.name))
        .beta
        .clone();

    // Reconstruct the production predictor exactly as `SavedModel::predictor`
    // builds it for a DispersionLocationScale model (beta_mu/beta_noise + the
    // family likelihood; covariance/inverse_link are not needed for the point
    // dispersion/noise channels under test).
    let predictor = DispersionLocationScalePredictor {
        beta_mu,
        beta_noise,
        likelihood: s.likelihood.clone(),
        inverse_link: None,
        covariance: None,
    };

    // ---- evaluate predict vs generate on a fresh grid -----------------------
    let grid_n = 40usize;
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| -1.4 + 2.8 * (i as f64) / (grid_n as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for i in 0..grid_n {
        grid[[i, x_idx]] = grid_x[i];
    }
    let mean_design = build_term_collection_design(grid.view(), &fit.meanspec_resolved)
        .unwrap_or_else(|e| panic!("[{}] rebuild mean design: {e}", s.name))
        .design;
    let noise_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .unwrap_or_else(|e| panic!("[{}] rebuild noise design: {e}", s.name))
        .design;
    let input = PredictInput {
        design: mean_design,
        offset: Array1::zeros(grid_n),
        design_noise: Some(noise_design),
        offset_noise: Some(Array1::zeros(grid_n)),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };

    // PREDICT side: per-row sqrt(Var(y)).
    let predict_sd = predictor
        .predict_noise_scale(&input)
        .unwrap_or_else(|e| panic!("[{}] predict_noise_scale: {e}", s.name))
        .unwrap_or_else(|| panic!("[{}] predict_noise_scale returned None", s.name));

    // GENERATE side: the per-row dispersion threaded into the NoiseModel, exactly
    // as run_generate_unified's DispersionLocationScale branch does.
    let dispersion = predictor
        .predict_dispersion_scale(&input)
        .unwrap_or_else(|e| panic!("[{}] predict_dispersion_scale: {e}", s.name))
        .unwrap_or_else(|| panic!("[{}] predict_dispersion_scale returned None", s.name));
    let noise =
        NoiseModel::from_likelihood_with_per_row_dispersion(&s.likelihood, dispersion.clone())
            .unwrap_or_else(|e| panic!("[{}] build per-row NoiseModel: {e}", s.name));

    let mean = predictor
        .predict_plugin_response(&input)
        .unwrap_or_else(|e| panic!("[{}] predict_plugin_response: {e}", s.name))
        .mean;
    let generate_var = noise_model_variance(&noise, &mean);

    // (1) ROOT-CAUSE INVARIANT: the variance the generate path will realise must
    //     equal, per row, the variance the predict path reports — the per-row
    //     precision channel is honoured identically by both, with the correct
    //     family units (Tweedie reciprocal included).
    let mut max_rel = 0.0_f64;
    for i in 0..grid_n {
        let predict_var = predict_sd[i] * predict_sd[i];
        let rel = (generate_var[i] - predict_var).abs() / predict_var.abs().max(1e-12);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1e-9,
        "[{}] generate-side per-row variance disagrees with predict-side variance \
         (max rel dev {max_rel:.3e}); the per-row precision channel / units are wrong (#1125)",
        s.name
    );

    // (2) The channel is genuinely non-constant — the data has a real dispersion
    //     gradient, so the recovered per-row dispersion must span a wide range.
    //     The pre-fix bug produced a CONSTANT dispersion (the whole eta_d(x)
    //     surface dropped); a flat dispersion here would make (1) vacuous.
    let dmin = dispersion.iter().cloned().fold(f64::INFINITY, f64::min);
    let dmax = dispersion.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spread = dmax / dmin;
    assert!(
        spread > 2.0,
        "[{}] recovered per-row dispersion is nearly constant (max/min={spread:.3}); \
         the per-row precision surface is not being exercised",
        s.name
    );

    eprintln!(
        "[{}] dispersion-LS predict↔generate variance agreement: max rel dev {max_rel:.2e}, \
         dispersion spread max/min={spread:.2}",
        s.name
    );
}

#[test]
fn dispersion_location_scale_generate_matches_predict_variance_gamma() {
    run_scenario(&Scenario {
        name: "gamma-LS",
        family: "gamma",
        fam: Fam::Gamma,
        likelihood: LikelihoodSpec::gamma_log(),
    });
}

#[test]
fn dispersion_location_scale_generate_matches_predict_variance_negbin() {
    run_scenario(&Scenario {
        name: "negbin-LS",
        family: "nb",
        fam: Fam::NegBin,
        // seed theta = 1.0: the worst case the generate path presents (#1124).
        likelihood: LikelihoodSpec::negative_binomial_log(1.0),
    });
}

#[test]
fn dispersion_location_scale_generate_matches_predict_variance_beta() {
    run_scenario(&Scenario {
        name: "beta-LS",
        family: "beta",
        fam: Fam::Beta,
        likelihood: LikelihoodSpec::beta_logit(1.0),
    });
}

#[test]
fn dispersion_location_scale_generate_matches_predict_variance_tweedie() {
    run_scenario(&Scenario {
        name: "tweedie-LS",
        family: "tweedie",
        fam: Fam::Tweedie,
        // Tweedie carries the variance power p on the spec; phi is the reciprocal
        // of the precision exp(eta_d) — the arm most prone to a units slip.
        likelihood: LikelihoodSpec::tweedie_log(1.5),
    });
}
