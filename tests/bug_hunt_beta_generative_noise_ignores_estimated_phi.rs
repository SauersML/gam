//! Bug hunt: `gam generate` (and every consumer of the generative observation
//! model) draws Beta responses with the **seed** precision `phi = 1.0` instead
//! of the precision the fit estimated, so synthetic draws / prediction variances
//! for a Beta model have grossly inflated dispersion.
//!
//! The fit records its estimated precision in `likelihood_scale`
//! (`LikelihoodScaleMetadata::EstimatedBetaPhi { phi }`), and the generate CLI's
//! `family_noise_parameter` correctly forwards it as the `gaussian_scale`
//! dispersion argument (it even documents: "the authoritative value is the fit's
//! scale metadata, not the seed phi on the original family spec"). But
//! `NoiseModel::from_likelihood` (src/inference/generative.rs) reads the Beta
//! precision off the *embedded* `likelihood.response = Beta { phi }` — which is
//! the un-updated seed `phi = 1.0` — and ignores the `gaussian_scale` it was
//! handed. Gamma and Tweedie in the same function correctly use `gaussian_scale`.
//!
//! Concretely (n = 6000 Beta data with true phi = 20):
//!   * fit.likelihood_scale = EstimatedBetaPhi { phi ~= 6.6 }
//!   * fit.likelihood_family.response = Beta { phi: 1.0 }   <- seed, never updated
//!   * NoiseModel::from_likelihood(Beta{1.0}, n, Some(6.6)) -> Beta { phi: 1.0 }
//!   * `gam generate` then yields draws whose empirical variance implies phi ~= 1.
//!
//! This is a distinct code path from the mean-coefficient attenuation bug: even
//! given an estimated phi, the generative/observation model drops it. The fix is
//! to thread the estimated dispersion into the Beta noise model (use
//! `gaussian_scale`, as Gamma/Tweedie do, or update the embedded family phi).
//!
//! RNG-free and deterministic: a fixed-seed inline Beta sampler builds the data.
//! The assertion mirrors exactly what the generate CLI composes.

use gam::generative::NoiseModel;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

// ----- deterministic sampling primitives (no external RNG crate) -----

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unif();
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn gamma_ge1(&mut self, shape: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        for _ in 0..10_000 {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unif();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
        d
    }
    fn beta(&mut self, a: f64, b: f64) -> f64 {
        let ga = self.gamma_ge1(a);
        let gb = self.gamma_ge1(b);
        ga / (ga + gb)
    }
}

#[inline]
fn logistic(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// High-precision Beta data: `logit(mu) = 0.4 + 0.8*x1 - 0.5*x2`, precision
/// `phi = 20`. `eta` is clamped so every Beta shape stays >= 1.
fn make_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const N: usize = 6000;
    const PHI: f64 = 20.0;
    let mut rng = SplitMix64::new(0x0BADF00D_DEADBEEF);
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let a = 0.7 * rng.normal();
        let b = 0.7 * rng.normal();
        let eta = (0.4 + 0.8 * a - 0.5 * b).clamp(-2.2, 2.2);
        let mu = logistic(eta);
        let yi = rng
            .beta(mu * PHI, (1.0 - mu) * PHI)
            .clamp(1.0e-6, 1.0 - 1.0e-6);
        x1.push(a);
        x2.push(b);
        y.push(yi);
    }
    (y, x1, x2)
}

fn encode(y: &[f64], x1: &[f64], x2: &[f64]) -> gam::inference::data::EncodedDataset {
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let rows: Vec<csv::StringRecord> = (0..y.len())
        .map(|i| {
            csv::StringRecord::from(vec![y[i].to_string(), x1[i].to_string(), x2[i].to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode beta dataset")
}

#[test]
fn beta_generative_noise_uses_estimated_phi_not_seed() {
    init_parallelism();
    let (y, x1, x2) = make_dataset();
    let ds = encode(&y, &x1, &x2);

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x1 + x2", &ds, &cfg).expect("beta-regression fit succeeds");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };

    // The fit records an estimated precision in its scale metadata. This is the
    // exact value the `generate` CLI forwards as the dispersion argument (see
    // `family_noise_parameter`: for Beta it returns `likelihood_scale.fixed_phi()`).
    let estimated_phi = fit
        .fit
        .likelihood_scale
        .fixed_phi()
        .expect("a fitted Beta model must record an estimated precision phi");
    // Sanity: the data is high-precision, so the estimate must be well above the
    // seed phi = 1. (If this ever fails the test is mis-set-up, not the bug.)
    assert!(
        estimated_phi > 1.5,
        "fit did not estimate a high precision for low-variance Beta data: phi={estimated_phi}"
    );

    // Reproduce exactly what `gam generate` composes: the saved family likelihood
    // (whose embedded phi is the un-updated seed) plus the estimated dispersion.
    let likelihood = fit
        .fit
        .likelihood_family
        .clone()
        .expect("beta fit records a likelihood family");
    let noise = NoiseModel::from_likelihood(&likelihood, ds.values.nrows(), Some(estimated_phi))
        .expect("beta generative noise model builds");

    let NoiseModel::Beta { phi: noise_phi } = noise else {
        panic!("expected a Beta observation noise model, got {noise:?}");
    };

    // The observation model that generation/prediction-variance uses must carry
    // the *estimated* precision, not the seed phi = 1.0. The bug leaves it at 1.0,
    // so `gam generate` draws Beta responses with ~20x too much variance. The
    // dispersion field is now a per-row vector (#1125) broadcast from the scalar
    // estimate, so every row must carry the fitted precision.
    assert!(
        noise_phi.iter().all(|&p| (p - estimated_phi).abs() < 1e-6),
        "Beta generative noise uses the seed precision, not the fitted one: \
         noise phi={noise_phi:?}, estimated phi={estimated_phi} \
         (gam generate would draw responses with phi={noise_phi:?})"
    );
}
