//! Bug hunt: Beta regression mean coefficients are biased toward zero because
//! the precision `phi` is estimated from the *null* linear predictor (the cold
//! warm-start `eta ~= 0`, i.e. `mu ~= 0.5` everywhere) and then frozen for the
//! whole fit, instead of being refreshed from the fitted mean.
//!
//! Why this biases the *mean*: Beta regression does NOT factor the precision out
//! of the mean score the way a canonical-link exponential-family GLM does. The
//! score for `beta` is
//!     d l / d beta = phi * sum_i x_i * (y*_i - mu*_i),
//!     with y*_i = logit(y_i)  and  mu*_i = psi(mu_i*phi) - psi((1-mu_i)*phi),
//! so `mu*` — and hence the root `beta_hat` — depends on `phi` through the
//! digamma terms. Using a `phi` that is far too small (because it was measured at
//! `mu = 0.5`, where the Pearson residuals reflect the full marginal spread of
//! `y` rather than its conditional spread) shrinks every fitted coefficient
//! toward zero.
//!
//! Reproduction: genuine `Beta(mu*phi, (1-mu)*phi)` data with a known mean
//! `logit(mu) = 0.4 + 0.8*x1 - 0.5*x2` and a high precision `phi = 20`. With
//! n = 8000 a consistent estimator must recover the slopes to within a few
//! hundredths. The maximum-likelihood / joint (`phi` estimated) fit recovers
//! `b1 ~= 0.79`, `b2 ~= -0.49`. gam instead returns `b1 ~= 0.70`, `b2 ~= -0.44`
//! — the MLE for `phi ~= 3.5`, the value `estimate_beta_phi_from_eta` returns
//! when evaluated at the null predictor `mu = 0.5`.
//!
//! Root cause pointer: `src/solver/pirls/mod.rs`
//!   * `estimate_beta_phi_from_eta` (the textbook moment estimator, itself
//!     correct: it returns `~phi` when given the *fitted* `eta`), and
//!   * the freeze at lines ~1861-1866 (`beta_phi_is_estimated() && !beta_phi_locked`)
//!     which seeds `phi` from `self.workspace.eta_buf` at the very start of the
//!     inner solve — before the mean has been fit — and locks it.
//!
//! This test is RNG-free and self-contained: it draws genuine Beta variates with
//! a fixed-seed splitmix64 -> Box-Muller -> Marsaglia-Tsang gamma sampler, so the
//! data (and therefore the assertion) is fully deterministic. When the precision
//! is estimated from the fitted predictor (or jointly with the mean) the test
//! passes without edits.

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
    /// Uniform in the open interval (0, 1).
    fn unif(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
    /// Standard normal via Box-Muller (cosine branch).
    fn normal(&mut self) -> f64 {
        let u1 = self.unif();
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Gamma(shape, 1) for shape >= 1 via Marsaglia-Tsang.
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
    /// Beta(a, b) for a, b >= 1 as the ratio of two gammas.
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

/// Build the deterministic Beta dataset. Truth: `logit(mu) = B0 + B1*x1 + B2*x2`
/// with precision `PHI`. `eta` is clamped so every Beta shape stays >= 1 (so the
/// shape >= 1 gamma sampler is valid and no row sits at the 0/1 boundary).
const N: usize = 8000;
const B0: f64 = 0.4;
const B1: f64 = 0.8;
const B2: f64 = -0.5;
const PHI: f64 = 20.0;

fn make_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let a = 0.7 * rng.normal();
        let b = 0.7 * rng.normal();
        let eta = (B0 + B1 * a + B2 * b).clamp(-2.2, 2.2);
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

/// Fitted linear predictor at an explicit `(x1, x2)` grid, by rebuilding the
/// design and applying `beta`. Returns one `eta` per grid row.
fn predict_eta(
    ds: &gam::inference::data::EncodedDataset,
    fit: &gam::StandardFitResult,
    pts: &[(f64, f64)],
) -> Vec<f64> {
    use gam::matrix::LinearOperator;
    use gam::smooth::build_term_collection_design;
    use ndarray::Array2;
    let col = ds.column_map();
    let i1 = col["x1"];
    let i2 = col["x2"];
    let mut grid = Array2::<f64>::zeros((pts.len(), ds.headers.len()));
    for (r, &(a, b)) in pts.iter().enumerate() {
        grid[[r, i1]] = a;
        grid[[r, i2]] = b;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at grid points");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn beta_regression_recovers_known_slopes_under_high_precision() {
    init_parallelism();
    let (y, x1, x2) = make_dataset();
    let ds = encode(&y, &x1, &x2);

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x1 + x2", &ds, &cfg)
        .expect("beta-regression parametric fit succeeds");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };
    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "beta coefficients must be finite, got {:?}",
        fit.fit.beta
    );

    // Read the fitted slopes off the linear predictor: with a parametric linear
    // term, eta(x1, x2) = intercept + b1*x1 + b2*x2, so the per-covariate slope
    // is a difference of predictions (the intercept cancels). This is robust to
    // the model's internal coefficient scaling/ordering.
    let eta = predict_eta(&ds, &fit, &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
    let b1_hat = eta[1] - eta[0];
    let b2_hat = eta[2] - eta[0];

    // Consistency: at n = 8000 with the correct family, a maximum-likelihood
    // (joint phi) fit recovers the truth to within a few hundredths
    // (b1 ~= 0.79, b2 ~= -0.49). The freeze-phi-at-the-null-predictor bug instead
    // attenuates every slope toward zero (b1 ~= 0.70, b2 ~= -0.44), the MLE for a
    // spuriously small phi ~= 3.5. Tolerances are wide enough to admit the honest
    // sampling spread of the joint MLE, but exclude the biased estimate.
    assert!(
        (b1_hat - B1).abs() < 0.06,
        "Beta slope b1 not recovered: got {b1_hat:.4}, true {B1} (joint-MLE recovers ~0.79; \
         a phi frozen at the null predictor attenuates this toward ~0.70)"
    );
    assert!(
        (b2_hat - B2).abs() < 0.06,
        "Beta slope b2 not recovered: got {b2_hat:.4}, true {B2} (joint-MLE recovers ~-0.49; \
         a phi frozen at the null predictor attenuates this toward ~-0.44)"
    );
}
