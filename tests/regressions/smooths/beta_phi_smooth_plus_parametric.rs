//! Bug hunt (regression, REML path): a Beta model with *both* a parametric slope
//! and a penalized smooth — `y ~ x1 + s(x2)` — must recover the parametric slope
//! and the true precision.
//!
//! This complements `bug_hunt_beta_phi_frozen_at_null_predictor` (purely
//! parametric, no λ search). The smooth term introduces a smoothing parameter, so
//! the fit runs the outer REML λ optimization, where each inner cost evaluation
//! holds the Beta precision φ frozen (refreshing φ inside the λ search would
//! couple the scale to λ and reward over-smoothing). The φ ↔ mean fixed point is
//! therefore driven only at the single final reported fit at the selected λ. This
//! test pins that the final-fit fixed point recovers the mean and precision even
//! when REML smoothing selection sits in between — the case the original report
//! noted still attenuated to b1 ≈ 0.63.
//!
//! RNG-free and deterministic (fixed-seed splitmix64 → Box-Muller → Marsaglia-
//! Tsang gamma → beta sampler).

use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

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

const N: usize = 6000;
const B0: f64 = 0.4;
const B1: f64 = 0.8;
const PHI: f64 = 20.0;

fn make_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let a = 0.7 * rng.normal();
        let b = 1.2 * rng.normal();
        // True mean: parametric slope in x1 plus a genuine smooth (sine) in x2.
        let eta = (B0 + B1 * a + 0.6 * b.sin()).clamp(-2.2, 2.2);
        let mu = logistic(eta);
        let yi = rng.beta(mu * PHI, (1.0 - mu) * PHI).clamp(1e-6, 1.0 - 1e-6);
        x1.push(a);
        x2.push(b);
        y.push(yi);
    }
    (y, x1, x2)
}

#[test]
fn beta_smooth_plus_parametric_recovers_slope_and_precision() {
    init_parallelism();
    let (y, x1, x2) = make_dataset();
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![y[i].to_string(), x1[i].to_string(), x2[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode beta dataset");

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x1 + s(x2)", &ds, &cfg).expect("beta smooth fit succeeds");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };
    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "beta coefficients must be finite"
    );

    // Slope of the parametric x1 term via a difference of predicted η at fixed
    // x2 (the intercept and the x2 smooth contribution cancel).
    //
    // Both grid rows go through ONE design build, and neither covariate column
    // may be identically zero — the design builder rejects a parametric column
    // of all zeros as a term that cannot carry a recoverable effect. So x1 is
    // evaluated at ±1 (column [1, −1]) rather than at 1 and 0, and x2 is held at
    // a fixed NON-zero value (column [0.5, 0.5]) whose smooth contribution
    // cancels in the difference exactly as a held-at-zero one would.
    use gam::matrix::LinearOperator;
    use gam::smooth::build_term_collection_design;
    use ndarray::Array2;
    let col = ds.column_map();
    let i1 = col["x1"];
    let i2 = col["x2"];
    let mut grid = Array2::<f64>::zeros((2, ds.headers.len()));
    grid[[0, i1]] = 1.0;
    grid[[1, i1]] = -1.0;
    grid[[0, i2]] = 0.5;
    grid[[1, i2]] = 0.5;
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at grid points");
    let eta = design.design.apply(&fit.fit.beta).to_vec();
    // η(x1=1) − η(x1=−1) = 2·B1.
    let b1_hat = 0.5 * (eta[0] - eta[1]);

    // The parametric slope must recover the truth. A φ frozen at the null
    // predictor attenuated this to ~0.63 with a smooth term present; the
    // final-fit fixed point restores ~0.79.
    assert!(
        (b1_hat - B1).abs() < 0.07,
        "Beta parametric slope not recovered under a smooth term: got {b1_hat:.4}, true {B1}"
    );

    // The reported precision must reflect the high-precision data, not the seed
    // φ = 1 nor the null-predictor moment estimate (~3). It is the value
    // `gam generate` / prediction-variance consumers read.
    let phi = fit
        .fit
        .likelihood_scale
        .fixed_phi()
        .expect("a fitted Beta model records an estimated precision phi");
    assert!(
        phi.is_finite() && phi > 8.0,
        "reported precision did not reflect the conditional spread: phi={phi:.4} \
         (seed=1, null-predictor≈3, truth=20)"
    );
}
