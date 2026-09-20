//! gam#2926 acceptance: the Bernoulli marginal-slope default anchors on the
//! estimated law of the score, and the Gaussian closed form is a declaration.
//!
//! The model is `P(Y = 1 | a, z) = Φ(α(a) + b·z)` with `α` defined by the
//! anchoring equation on the law of `z | a`:
//!
//! ```text
//!     E[Φ(α(a) + b·z) | a] = Φ(q(a)) = π(a).
//! ```
//!
//! Two claims, each at the level of a whole fit:
//!
//! 1. **On a skewed score the default is calibrated and the Gaussian form is
//!    not.** The score is a standardised two-component normal mixture, the same
//!    law in every context, and the outcome is simulated from the anchored model
//!    on that law. The default fit anchors on the law it estimates from the
//!    score; the Gaussian form is reached through a declared Gauss–Hermite law,
//!    which is its anchor to quadrature tolerance (claim 2). A Gaussian
//!    declaration on this score is refused (gam#2968): its excess anchoring loss
//!    is many standard errors beyond zero. The default and the Gaussian form are judged against the
//!    TRUE law, in closed form: `E[Φ(α + b·z)] = Σ_j p_j Φ((α + b·μ_j)/√(1 + b²σ_j²))`
//!    for a normal mixture.
//! 2. **On a Gaussian score the declared law and the closed form agree to
//!    quadrature tolerance**, with the grid sized to the drive scale: at drive
//!    SD 2 a 64-node Gauss–Hermite law is within about 1.5e-4 of the closed form
//!    and a 128-node law within about 4e-8; there the default IS the closed form,
//!    chosen by the adequacy evidence.
//! 3. **The adequacy-gated choice hides at most sampling-scale error.** On a
//!    score whose skewness sits just inside the adequacy bound the default
//!    consumes the closed form, and its prediction averaged under the TRUE law
//!    stays within the tolerance claim 1 holds the default to.

use csv::StringRecord;
use gam::solver::fit_orchestration::DeclaredLatentLaw;
use gam::utils::splitmix64;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn normal_cdf(x: f64) -> f64 {
    gam::probability::normal_cdf(x)
}

fn standardized(mut v: Vec<f64>) -> Vec<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n)
        .sqrt()
        .max(1e-12);
    for value in v.iter_mut() {
        *value = (*value - mean) / sd;
    }
    v
}

/// Root of a strictly increasing function by bisection on `[-40, 40]`.
fn increasing_root(f: impl Fn(f64) -> f64, target: f64) -> f64 {
    let (mut low, mut high) = (-40.0_f64, 40.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if f(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// A finite law: nodes and weights summing to one.
struct FiniteLaw {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl FiniteLaw {
    /// Probabilists' Gauss–Hermite on `m` nodes, from the ORTHONORMAL recurrence
    /// `p_n = (x·p_{n−1} − √(n−1)·p_{n−2})/√n`, whose values stay finite at every
    /// root for the node counts used here; weights are `1/(m·p_{m−1}(x_k)²)`.
    /// Built here so the fixture does not share code with the crate.
    fn gauss_hermite(m: usize) -> Self {
        let orthonormal = |x: f64| -> (f64, f64) {
            let mut prev = 0.0;
            let mut current = 1.0;
            for n in 1..=m {
                let next = (x * current - (n as f64 - 1.0).sqrt() * prev) / (n as f64).sqrt();
                prev = current;
                current = next;
            }
            (current, prev)
        };
        let lo = -(2.0 * (m as f64).sqrt() + 2.0);
        let steps = 400_000;
        let width = -2.0 * lo / steps as f64;
        let mut roots = Vec::with_capacity(m);
        let mut previous = orthonormal(lo).0;
        for step in 1..=steps {
            let x = lo + width * step as f64;
            let value = orthonormal(x).0;
            if previous.signum() != value.signum() {
                let (mut a, mut b) = (x - width, x);
                for _ in 0..200 {
                    let mid = 0.5 * (a + b);
                    if orthonormal(a).0.signum() == orthonormal(mid).0.signum() {
                        a = mid;
                    } else {
                        b = mid;
                    }
                }
                roots.push(0.5 * (a + b));
            }
            previous = value;
        }
        assert_eq!(roots.len(), m, "found every Hermite root");
        let mut weights: Vec<f64> = roots
            .iter()
            .map(|&x| {
                let (_, prev) = orthonormal(x);
                1.0 / (m as f64 * prev * prev)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= total;
        }
        Self {
            nodes: roots,
            weights,
        }
    }

    fn declared(&self) -> DeclaredLatentLaw {
        DeclaredLatentLaw {
            nodes: self.nodes.clone(),
            weights: self.weights.clone(),
        }
    }

    /// The anchor `α(q, b)` on this law.
    fn anchor(&self, q: f64, slope: f64) -> f64 {
        increasing_root(
            |alpha| {
                self.nodes
                    .iter()
                    .zip(&self.weights)
                    .map(|(&u, &w)| w * normal_cdf(alpha + slope * u))
                    .sum()
            },
            normal_cdf(q),
        )
    }
}

/// The skewed score law: a two-component normal mixture standardised to zero
/// mean and unit variance (skewness about 1.5), the same law in every context.
struct Mixture {
    weights: [f64; 2],
    means: [f64; 2],
    sd: f64,
}

impl Mixture {
    fn skewed() -> Self {
        let weights = [0.83, 0.17];
        let raw_means = [-0.9, 2.8];
        let raw_sd = 0.5;
        let mean = weights[0] * raw_means[0] + weights[1] * raw_means[1];
        let var = weights
            .iter()
            .zip(raw_means.iter())
            .map(|(&p, &mu)| p * (raw_sd * raw_sd + mu * mu))
            .sum::<f64>()
            - mean * mean;
        let scale = var.sqrt();
        Self {
            weights,
            means: [(raw_means[0] - mean) / scale, (raw_means[1] - mean) / scale],
            sd: raw_sd / scale,
        }
    }

    fn draw(&self, state: &mut u64) -> f64 {
        let component = usize::from(next_unit(state) >= self.weights[0]);
        self.means[component] + self.sd * next_gauss(state)
    }

    /// `E[Φ(α + b·z)]` under this law, in closed form.
    fn expected_probability(&self, alpha: f64, slope: f64) -> f64 {
        let spread = (1.0 + slope * slope * self.sd * self.sd).sqrt();
        self.weights
            .iter()
            .zip(self.means.iter())
            .map(|(&p, &mu)| p * normal_cdf((alpha + slope * mu) / spread))
            .sum()
    }

    fn anchor(&self, q: f64, slope: f64) -> f64 {
        increasing_root(|alpha| self.expected_probability(alpha, slope), normal_cdf(q))
    }

    /// The standard normal law, as a one-component mixture.
    fn standard_normal() -> Self {
        Self {
            weights: [1.0, 0.0],
            means: [0.0, 0.0],
            sd: 1.0,
        }
    }

    /// `(E[Φ(α + b·z)], Var[Φ(α + b·z)])` under this law: the mean in closed form,
    /// the second moment by Gauss–Hermite quadrature within each component.
    fn probability_moments(&self, alpha: f64, slope: f64, quadrature: &FiniteLaw) -> (f64, f64) {
        let mean = self.expected_probability(alpha, slope);
        let second: f64 = self
            .weights
            .iter()
            .zip(self.means.iter())
            .map(|(&p, &mu)| {
                p * quadrature
                    .nodes
                    .iter()
                    .zip(&quadrature.weights)
                    .map(|(&g, &w)| {
                        let value = normal_cdf(alpha + slope * (mu + self.sd * g));
                        w * value * value
                    })
                    .sum::<f64>()
            })
            .sum();
        (mean, second - mean * mean)
    }
}

const SKEW_N: usize = 20_000;
const SKEW_SLOPE: f64 = 1.2;
const SKEW_INTERCEPT: f64 = -0.3;
const SKEW_BETA_X: f64 = 0.5;

fn skewed_fixture() -> (gam::inference::data::EncodedDataset, Vec<f64>) {
    let law = Mixture::skewed();
    let mut state: u64 = 0x2926_5EED_0000_0001;
    let x = standardized((0..SKEW_N).map(|_| next_gauss(&mut state)).collect());
    let mut rows = Vec::with_capacity(SKEW_N);
    for &xi in &x {
        let z = law.draw(&mut state);
        let q = SKEW_INTERCEPT + SKEW_BETA_X * xi;
        let eta = law.anchor(q, SKEW_SLOPE) + SKEW_SLOPE * z;
        let y = u8::from(next_unit(&mut state) < normal_cdf(eta));
        rows.push(StringRecord::from(vec![
            y.to_string(),
            z.to_string(),
            xi.to_string(),
        ]));
    }
    let headers = ["y", "z", "x"].iter().map(|s| s.to_string()).collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode the #2926 skewed fixture"),
        x,
    )
}

fn config(latent_measure: Option<&str>, declared: Option<DeclaredLatentLaw>) -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        latent_measure: latent_measure.map(str::to_string),
        declared_latent_law: declared,
        ..FitConfig::default()
    }
}

struct Fitted {
    law: &'static str,
    /// `q̂(x) = intercept + slope·x`, read by projecting the fitted marginal
    /// index on `x` (the marginal formula is linear, so the projection is exact).
    index_intercept: f64,
    index_slope: f64,
    slope: f64,
    coefficients: Vec<f64>,
    log_likelihood: f64,
    /// The finite law the fit anchored on, `None` for the closed form.
    grid: Option<FiniteLaw>,
    /// `|skew| / bound` from the recorded adequacy evidence, when the adequacy
    /// screen passed.
    skew_ratio: Option<f64>,
    /// The recorded closed-form certificate, when the screen passed.
    certificate: Option<gam::families::bms::ClosedFormAnchorResidual>,
}

fn fit(data: &gam::inference::data::EncodedDataset, x: &[f64], config: &FitConfig) -> Fitted {
    let result = fit_from_formula("y ~ x", data, config)
        .unwrap_or_else(|e| panic!("bernoulli marginal-slope fit: {e}"));
    let FitResult::BernoulliMarginalSlope(fit) = result else {
        panic!("expected a BernoulliMarginalSlope fit");
    };
    let index = fit.marginal_design.design.dot(&fit.fit.blocks[0].beta) + fit.baseline_marginal;
    let slope_eta = fit.slope_design.design.dot(&fit.fit.blocks[1].beta);
    let slope = fit.baseline_slope + slope_eta.iter().sum::<f64>() / slope_eta.len() as f64;
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let index_mean = index.iter().sum::<f64>() / n;
    let (mut cov, mut var) = (0.0, 0.0);
    for (i, &xi) in x.iter().enumerate() {
        cov += (xi - x_mean) * (index[i] - index_mean);
        var += (xi - x_mean) * (xi - x_mean);
    }
    let index_slope = cov / var;
    let grid = match &fit.latent_measure {
        gam::families::bms::LatentMeasureKind::GlobalEmpirical { grid } => Some(FiniteLaw {
            nodes: grid.nodes.clone(),
            weights: grid.weights.clone(),
        }),
        gam::families::bms::LatentMeasureKind::StandardNormal => None,
        other => panic!("this fixture's laws are global; got {other:?}"),
    };
    let (skew_ratio, certificate) = match &fit.latent_law_consumed {
        gam::families::bms::LatentLawConsumed::EstimatedGaussianAdequate {
            adequacy,
            residual,
            ..
        } => (Some(adequacy.skew.abs() / adequacy.skew_tol), residual.clone()),
        gam::families::bms::LatentLawConsumed::EstimatedGlobalByResidual {
            adequacy,
            residual,
            ..
        } => (
            Some(adequacy.skew.abs() / adequacy.skew_tol),
            Some(residual.clone()),
        ),
        gam::families::bms::LatentLawConsumed::DeclaredGaussian { residual, .. } => {
            (None, residual.clone())
        }
        _ => (None, None),
    };
    Fitted {
        skew_ratio,
        certificate,
        law: fit.latent_law_consumed.label(),
        index_intercept: index_mean - index_slope * x_mean,
        index_slope,
        slope,
        coefficients: fit.fit.beta.to_vec(),
        log_likelihood: fit.fit.log_likelihood,
        grid,
    }
}

/// Over a grid of contexts in the bulk of `x`: the largest `|Φ(q̂(x)) − π(x)|`
/// (is `q̂` the marginal index?) and the largest `|E_true[p̂ | x] − π(x)|` (does
/// the fit's own prediction average to the population risk under the TRUE law?).
fn calibration(fitted: &Fitted, law: &Mixture, intercept: f64, beta_x: f64) -> (f64, f64) {
    let mut index_error = 0.0_f64;
    let mut prediction_error = 0.0_f64;
    for step in 0..=6 {
        let x = -1.5 + 0.5 * step as f64;
        let pi = normal_cdf(intercept + beta_x * x);
        let q_hat = fitted.index_intercept + fitted.index_slope * x;
        let alpha_hat = match fitted.grid.as_ref() {
            Some(grid) => grid.anchor(q_hat, fitted.slope),
            None => q_hat * (1.0 + fitted.slope * fitted.slope).sqrt(),
        };
        index_error = index_error.max((normal_cdf(q_hat) - pi).abs());
        prediction_error =
            prediction_error.max((law.expected_probability(alpha_hat, fitted.slope) - pi).abs());
    }
    (index_error, prediction_error)
}

#[test]
fn default_law_is_calibrated_on_a_skewed_score_and_the_gaussian_form_is_not_2926() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let law = Mixture::skewed();
    let (data, x) = skewed_fixture();

    // A Gaussian declaration on this score is refused (gam#2968): the score fails
    // the standard-normal screen, and at the converged declared fit the closed
    // form's excess anchoring loss is many standard errors beyond zero, so the
    // declared anchor misstates the probabilities it anchors.
    let refusal = match fit_from_formula("y ~ x", &data, &config(Some("gaussian"), None)) {
        Ok(_) => panic!("a Gaussian declaration on this skewed score must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        refusal.contains("excess anchoring loss is beyond its sampling noise")
            && refusal.contains("Refused"),
        "the declaration must be refused by its anchoring-loss test: {refusal}"
    );
    eprintln!("[2926/2968 skewed] declared gaussian refused: {refusal}");

    let default = fit(&data, &x, &config(None, None));
    let gaussian = fit(
        &data,
        &x,
        &config(None, Some(FiniteLaw::gauss_hermite(64).declared())),
    );
    assert_eq!(default.law, "estimated-global", "the skewed law does not move with x");
    assert_eq!(gaussian.law, "declared-finite-law");

    let (default_index, default_prediction) =
        calibration(&default, &law, SKEW_INTERCEPT, SKEW_BETA_X);
    let (gaussian_index, gaussian_prediction) =
        calibration(&gaussian, &law, SKEW_INTERCEPT, SKEW_BETA_X);
    // What the Gaussian anchor claims at the TRUE coefficients.
    let theoretical_bias = (0..=6)
        .map(|step| {
            let x = -1.5 + 0.5 * step as f64;
            let q = SKEW_INTERCEPT + SKEW_BETA_X * x;
            (law.expected_probability(q * (1.0 + SKEW_SLOPE * SKEW_SLOPE).sqrt(), SKEW_SLOPE)
                - normal_cdf(q))
            .abs()
        })
        .fold(0.0_f64, f64::max);
    eprintln!(
        "[2926 skewed] n={SKEW_N} b={SKEW_SLOPE} | slope default={:.4} gaussian={:.4} | max \
         |Φ(q̂)−π|: default={default_index:.4} gaussian={gaussian_index:.4} | max \
         |E_true[p̂|x]−π|: default={default_prediction:.4} gaussian={gaussian_prediction:.4} | \
         Gaussian anchor bias at the truth={theoretical_bias:.4} | log-lik default={:.3} \
         gaussian={:.3}",
        default.slope, gaussian.slope, default.log_likelihood, gaussian.log_likelihood,
    );

    assert!(
        default_index < 0.02 && default_prediction < 0.02,
        "the default must be calibrated in context: max |Φ(q̂)−π| = {default_index:.4}, max \
         |E_true[p̂|x]−π| = {default_prediction:.4}"
    );
    assert!(
        gaussian_index > 0.5 * theoretical_bias && gaussian_index > 3.0 * default_index,
        "the Gaussian form must be measurably miscalibrated on a skewed score: max |Φ(q̂)−π| = \
         {gaussian_index:.4} against a closed-form anchor bias of {theoretical_bias:.4} and the \
         default's {default_index:.4}"
    );
    assert!(
        (default.slope - SKEW_SLOPE).abs() < 0.12,
        "the default must recover the planted slope; got {}",
        default.slope
    );
}

const GAUSS_N: usize = 12_000;
/// Slope 2 on a unit score: the drive `b·z` has SD 2, where the node count of
/// the declared Gauss–Hermite law starts to matter.
const GAUSS_SLOPE: f64 = 2.0;

#[test]
fn declared_gauss_hermite_law_agrees_with_the_closed_form_on_a_gaussian_score_2926() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let mut state: u64 = 0x2926_5EED_0000_0002;
    let x = standardized((0..GAUSS_N).map(|_| next_gauss(&mut state)).collect());
    let z = standardized((0..GAUSS_N).map(|_| next_gauss(&mut state)).collect());
    let mut rows = Vec::with_capacity(GAUSS_N);
    let c = (1.0 + GAUSS_SLOPE * GAUSS_SLOPE).sqrt();
    for i in 0..GAUSS_N {
        let q = -0.4 + 0.6 * x[i];
        let y = u8::from(next_unit(&mut state) < normal_cdf(q * c + GAUSS_SLOPE * z[i]));
        rows.push(StringRecord::from(vec![
            y.to_string(),
            z[i].to_string(),
            x[i].to_string(),
        ]));
    }
    let headers = ["y", "z", "x"].iter().map(|s| s.to_string()).collect();
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode the #2926 Gaussian fixture");

    let closed_form = fit(&data, &x, &config(Some("gaussian"), None));
    assert_eq!(closed_form.law, "declared-gaussian");
    let gh64 = fit(
        &data,
        &x,
        &config(None, Some(FiniteLaw::gauss_hermite(64).declared())),
    );
    let gh128 = fit(
        &data,
        &x,
        &config(None, Some(FiniteLaw::gauss_hermite(128).declared())),
    );
    let default = fit(&data, &x, &config(None, None));
    let certificate = certified(&default);
    let agreement = closed_form_agreement_in_se(&default, &Mixture::standard_normal(), GAUSS_N);
    let (default_index, default_prediction) =
        calibration(&default, &Mixture::standard_normal(), -0.4, 0.6);
    let estimated = fit(&data, &x, &config(Some("global-empirical"), None));
    assert_eq!(estimated.law, "requested-global-empirical");

    let coefficient_gap = |left: &Fitted, right: &Fitted| -> f64 {
        let beta = left
            .coefficients
            .iter()
            .zip(&right.coefficients)
            .map(|(a, b)| (a - b).abs() / (1.0 + a.abs()))
            .fold(0.0_f64, f64::max);
        let index = (left.index_intercept - right.index_intercept)
            .abs()
            .max((left.index_slope - right.index_slope).abs());
        beta.max(index).max((left.slope - right.slope).abs())
    };
    let gap64 = coefficient_gap(&closed_form, &gh64);
    let gap128 = coefficient_gap(&closed_form, &gh128);
    let gap_estimated = coefficient_gap(&closed_form, &estimated);
    let gap_default = coefficient_gap(&closed_form, &default);
    let ll64 = (closed_form.log_likelihood - gh64.log_likelihood).abs();
    let ll128 = (closed_form.log_likelihood - gh128.log_likelihood).abs();
    eprintln!(
        "[2926 gaussian] n={GAUSS_N} drive SD={GAUSS_SLOPE} | slope closed-form={:.6} gh64={:.6} \
         gh128={:.6} estimated={:.6} | max coefficient gap: gh64={gap64:.3e} gh128={gap128:.3e} \
         estimated-law={gap_estimated:.3e} default={gap_default:.3e} | |Δ log-lik|: gh64={ll64:.3e} gh128={ll128:.3e}",
        closed_form.slope, gh64.slope, gh128.slope, estimated.slope,
    );
    assert!(
        gap128 < 1e-5 && ll128 < 1e-4,
        "at drive SD {GAUSS_SLOPE} a 128-node Gauss–Hermite law must reproduce the closed form to \
         quadrature tolerance; coefficient gap {gap128:.3e}, log-lik gap {ll128:.3e}"
    );
    assert!(
        gap64 < 1e-3,
        "a 64-node Gauss–Hermite law must agree with the closed form at the 1e-4 level at drive \
         SD {GAUSS_SLOPE}; coefficient gap {gap64:.3e}"
    );
    assert!(
        gap_estimated < 0.05,
        "on a Gaussian score the estimated law must agree with the closed form to sampling \
         tolerance; coefficient gap {gap_estimated:.3e}"
    );
    eprintln!(
        "[2926 gaussian default] law={} certificate: {certificate:?} | closed-form vs empirical \
         agreement {agreement:.2} se | max |Φ(q̂)−π|={default_index:.4} max \
         |E_true[p̂|x]−π|={default_prediction:.4}",
        default.law,
    );
    if certificate.closed_form_chosen {
        assert!(
            gap_default < 1e-9,
            "a default that kept the closed form is the closed form; coefficient gap \
             {gap_default:.3e}"
        );
    }
    assert!(
        agreement <= 3.0,
        "on a Gaussian score the closed-form and empirical predictions must agree within 3·se; \
         got {agreement:.2} se"
    );
    assert!(
        default_index < 0.02 && default_prediction < 0.02,
        "the default must be calibrated under the TRUE law in either branch: max |Φ(q̂)−π| = \
         {default_index:.4}, max |E_true[p̂|x]−π| = {default_prediction:.4}"
    );
}

/// The standard-normal adequacy screen's skewness bound at the fixture's size, which
/// the fit also records beside the statistic: the two-sided normal quantile at the
/// screen's level 0.05 split over its 8 clauses, over the exact standard error of a
/// normal sample's skewness (gam#2926). The fixture aims at a fraction of it.
fn adequacy_skew_bound() -> f64 {
    let n = SKEW_N as f64;
    let z = gam::probability::standard_normal_quantile(1.0 - 0.05 / 16.0)
        .expect("a probability inside (0, 1)");
    z * (6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))).sqrt()
}
/// How far inside the bound the fixture's sample skewness is placed.
const JUST_INSIDE_SKEW_RATIO: f64 = 0.9;

/// Drive at which the closed form's anchoring error on the just-inside law is far
/// below the certificate's tolerance: it grows like `b³`.
const WEAK_SLOPE: f64 = 0.1;

/// A score just inside the adequacy screen, with the outcome simulated at `slope`
/// from the anchored model on the score's TRUE law: `(data, x, law, shift,
/// sample skewness)`.
///
/// The law is a normal mixture `0.85·N(0, 1) + 0.15·N(shift, 1)`, with `shift` placing
/// its population skewness at `JUST_INSIDE_SKEW_RATIO` of the screen's bound. The
/// scores are the mixture's quantiles at a shuffled grid of probabilities, standardised
/// on the sample, so every statistic the screen reads (the tail masses and the KS
/// distance as well as the moments) is the law's own to within the grid's rounding
/// rather than a draw's. A random sample of this size puts the `|z| > 4σ` count above
/// its bound by chance often enough to fake a screen failure. The TRUE law is the
/// mixture through the same standardising map, and the shuffle keeps the score
/// independent of `x`.
fn just_inside_fixture(
    slope: f64,
) -> (
    gam::inference::data::EncodedDataset,
    Vec<f64>,
    Mixture,
    f64,
    f64,
) {
    const FAR_WEIGHT: f64 = 0.15;
    let mut state: u64 = 0x2926_5EED_0000_0003;
    let x = standardized((0..SKEW_N).map(|_| next_gauss(&mut state)).collect());
    // The mixture's skewness `p(1−p)(1−2p)θ³ / (1 + p(1−p)θ²)^{3/2}` increases in θ on
    // [0, 3], so bisection places it at the target.
    let population_skew = |theta: f64| {
        let pq = FAR_WEIGHT * (1.0 - FAR_WEIGHT);
        pq * (1.0 - 2.0 * FAR_WEIGHT) * theta.powi(3) / (1.0 + pq * theta * theta).powf(1.5)
    };
    let target = JUST_INSIDE_SKEW_RATIO * adequacy_skew_bound();
    let (mut low, mut high) = (0.0_f64, 3.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if population_skew(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    let shift = 0.5 * (low + high);
    let mixture_cdf =
        |r: f64| (1.0 - FAR_WEIGHT) * normal_cdf(r) + FAR_WEIGHT * normal_cdf(r - shift);
    // Midpoint probabilities in a Fisher–Yates order.
    let mut probabilities: Vec<f64> =
        (0..SKEW_N).map(|k| (k as f64 + 0.5) / SKEW_N as f64).collect();
    for k in (1..SKEW_N).rev() {
        let j = (splitmix64(&mut state) % (k as u64 + 1)) as usize;
        probabilities.swap(k, j);
    }
    let raw_z: Vec<f64> = probabilities
        .iter()
        .map(|&p| {
            let (mut low, mut high) = (-12.0_f64, 12.0 + shift);
            for _ in 0..100 {
                let mid = 0.5 * (low + high);
                if mixture_cdf(mid) < p {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            0.5 * (low + high)
        })
        .collect();
    let n = raw_z.len() as f64;
    let mean = raw_z.iter().sum::<f64>() / n;
    let sd = (raw_z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let sample_skew = raw_z.iter().map(|v| ((v - mean) / sd).powi(3)).sum::<f64>() / n;
    // The TRUE law of the standardised score: the mixture through the same map.
    let law = Mixture {
        weights: [0.85, 0.15],
        means: [-mean / sd, (shift - mean) / sd],
        sd: 1.0 / sd,
    };
    let mut rows = Vec::with_capacity(SKEW_N);
    for (i, &xi) in x.iter().enumerate() {
        let z = (raw_z[i] - mean) / sd;
        let q = SKEW_INTERCEPT + SKEW_BETA_X * xi;
        let eta = law.anchor(q, slope) + slope * z;
        let y = u8::from(next_unit(&mut state) < normal_cdf(eta));
        rows.push(StringRecord::from(vec![
            y.to_string(),
            z.to_string(),
            xi.to_string(),
        ]));
    }
    let headers = ["y", "z", "x"].iter().map(|s| s.to_string()).collect();
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode the #2926 just-inside fixture");
    (data, x, law, shift, sample_skew)
}

/// Over the grid of contexts, the largest gap between the closed-form and the
/// estimated-law predictions at the FITTED coefficients under the TRUE law, in units
/// of the estimated law's sampling standard error at `n` scores:
/// `|E_true[Φ(a_cf + b̂·z)] − Φ(q̂)| / (sd_true[Φ(a_cf + b̂·z)] / √n)`. An anchor on the
/// law itself averages to `Φ(q̂)`, so the gap is the closed form's.
fn closed_form_agreement_in_se(fitted: &Fitted, law: &Mixture, n: usize) -> f64 {
    let quadrature = FiniteLaw::gauss_hermite(64);
    (0..=6)
        .map(|step| {
            let x = -1.5 + 0.5 * step as f64;
            let q_hat = fitted.index_intercept + fitted.index_slope * x;
            let alpha_cf = q_hat * (1.0 + fitted.slope * fitted.slope).sqrt();
            let (mean, variance) = law.probability_moments(alpha_cf, fitted.slope, &quadrature);
            (mean - normal_cdf(q_hat)).abs() / (variance.sqrt() / (n as f64).sqrt())
        })
        .fold(0.0_f64, f64::max)
}

/// The recorded certificate of a fit whose adequacy screen passed, checked against
/// the fit's recorded decision: the closed form is kept exactly when the residual
/// energy's null tail is at or above the design rate, and the label follows the
/// decision.
fn certified(fitted: &Fitted) -> gam::families::bms::ClosedFormAnchorResidual {
    let Some(certificate) = fitted.certificate.clone() else {
        panic!(
            "a score that passes the adequacy screen must carry a recorded certificate; got law={}",
            fitted.law
        )
    };
    assert_eq!(
        certificate.closed_form_chosen,
        certificate
            .null_p_value
            .is_some_and(|p| p >= gam::families::bms::CLOSED_FORM_CERTIFICATE_ALPHA),
        "the recorded decision must be the recorded null tail against the design rate: \
         {certificate:?}"
    );
    let expected = if certificate.closed_form_chosen {
        "estimated-gaussian-adequate"
    } else {
        "estimated-global-by-residual"
    };
    assert_eq!(
        fitted.law, expected,
        "the label must follow the recorded decision: {certificate:?}"
    );
    certificate
}

/// `|skew| / bound` of a just-inside fixture's fit, which must sit just inside the
/// screen or the fixture does not probe its edge.
fn just_inside(fitted: &Fitted) -> f64 {
    let skew_ratio = fitted.skew_ratio.unwrap_or_else(|| {
        panic!(
            "a score at {JUST_INSIDE_SKEW_RATIO} of the skewness bound must pass the adequacy \
             screen; got law={}",
            fitted.law
        )
    });
    assert!(
        skew_ratio > 0.8,
        "fixture invariant: the recorded skewness must sit just inside its bound, or this test does \
         not probe the edge of the screen; |skew|/bound = {skew_ratio:.3}"
    );
    skew_ratio
}

/// gam#2926 acceptance: on a law just inside the adequacy screen, at a weak drive,
/// the fit records whichever anchor its excess-KL estimate expects to be the more
/// accurate, the closed form agrees with the estimated law within that law's
/// sampling error, and the fit is calibrated under the TRUE law in either branch.
#[test]
fn a_law_just_inside_the_screen_at_a_weak_drive_records_its_anchor_choice_2926() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let (data, x, law, shift, sample_skew) = just_inside_fixture(WEAK_SLOPE);
    let fitted = fit(&data, &x, &config(None, None));
    let skew_ratio = just_inside(&fitted);
    let certificate = certified(&fitted);
    let agreement = closed_form_agreement_in_se(&fitted, &law, SKEW_N);
    let (index_error, prediction_error) =
        calibration(&fitted, &law, SKEW_INTERCEPT, SKEW_BETA_X);
    eprintln!(
        "[2926 just-inside weak] n={SKEW_N} b={WEAK_SLOPE} shift={shift:.4} sample skew={sample_skew:.4} \
         | law={} recorded |skew|/bound={skew_ratio:.3} | slope={:.4} | certificate: {certificate:?} | \
         closed-form vs estimated-law agreement {agreement:.2} se | max |Φ(q̂)−π|={index_error:.4} \
         max |E_true[p̂|x]−π|={prediction_error:.4}",
        fitted.law, fitted.slope,
    );
    assert!(
        agreement <= 3.0,
        "at a weak drive the closed-form and estimated-law predictions must agree within 3·se; got \
         {agreement:.2} se"
    );
    assert!(
        index_error < 0.02 && prediction_error < 0.02,
        "the fit must be calibrated under the TRUE law in either branch: max |Φ(q̂)−π| = \
         {index_error:.4}, max |E_true[p̂|x]−π| = {prediction_error:.4}"
    );
}

/// gam#2926 acceptance: the same law at `b = 1.2` passes the adequacy screen, and the
/// closed form's anchoring error grows like `b³`, so this is where the certificate
/// rather than the screen decides. The fit records the anchor its `D̂` prefers under
/// that anchor's label, and is calibrated under the TRUE law in either branch.
#[test]
fn a_law_just_inside_the_screen_at_a_strong_drive_records_its_anchor_choice_2926() {
    init_parallelism();
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);

    let (data, x, law, shift, sample_skew) = just_inside_fixture(SKEW_SLOPE);
    let fitted = fit(&data, &x, &config(None, None));
    let skew_ratio = just_inside(&fitted);
    let certificate = certified(&fitted);
    let (index_error, prediction_error) =
        calibration(&fitted, &law, SKEW_INTERCEPT, SKEW_BETA_X);
    eprintln!(
        "[2926 just-inside strong] n={SKEW_N} b={SKEW_SLOPE} shift={shift:.4} sample skew={sample_skew:.4} \
         | law={} recorded |skew|/bound={skew_ratio:.3} | slope={:.4} | certificate: {certificate:?} | \
         max |Φ(q̂)−π|={index_error:.4} max |E_true[p̂|x]−π|={prediction_error:.4}",
        fitted.law, fitted.slope,
    );
    assert!(
        index_error < 0.02 && prediction_error < 0.02,
        "the fit must be calibrated under the TRUE law in either branch: max |Φ(q̂)−π| = \
         {index_error:.4}, max |E_true[p̂|x]−π| = {prediction_error:.4}"
    );
}
