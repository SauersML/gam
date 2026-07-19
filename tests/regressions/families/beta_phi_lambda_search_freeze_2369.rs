//! Regression (#2369): an estimated-φ Beta fit must be able to certify a
//! stationary optimum.
//!
//! Beta was the one estimated-dispersion family without a λ-search freeze latch
//! (the siblings are Gamma `k` #1074/#2361, Tweedie φ #1477, NB θ #1082). A
//! fresh `GamWorkingModel` is built per inner solve and re-derives φ by the
//! Pearson moment estimator from that solve's warm-start η, so φ drifted with ρ
//! across the outer search.
//!
//! That drift is fatal for Beta specifically, because φ enters the working
//! *response* rather than only the weight: with the logit link the mean score is
//!     ∂ℓ/∂β = φ · Σᵢ xᵢ (y*ᵢ − μ*ᵢ),
//!     y*ᵢ = logit(yᵢ),  μ*ᵢ = ψ(μᵢφ) − ψ((1−μᵢ)φ),
//! so β̂ itself moves with φ through the digamma terms. The realized outer
//! criterion was therefore `V(ρ, φ(ρ))` while the analytic outer gradient is
//! `∂V/∂ρ` at fixed φ. The two differ by the chain term `(∂V/∂φ)·(dφ/dρ)`, which
//! cannot vanish — the moment estimator is not the REML maximizer in φ, so
//! `∂V/∂φ ≠ 0` at the installed φ. The projected gradient consequently floored
//! above tolerance at EVERY ρ and no estimated-φ Beta fit could satisfy the
//! stationarity certificate: the reported failure was universal (every seed,
//! every φ, every n, every formula shape), refusing with the interior-PSD-
//! non-railed `|Pg|≈1.6e-1 bound≈1.0e-2 hessian_psd=yes railed=[]` signature.
//!
//! Pure noise is the sharpest form of the gate: with no signal to recover, the
//! only thing a fit can fail at is certifying its own optimum. This test would
//! not compile-fail or assert-fail on a numerical technicality — it fails iff
//! the outer criterion is non-stationary.
//!
//! RNG-free and deterministic (fixed-seed splitmix64 → Box-Muller → Marsaglia-
//! Tsang gamma → beta sampler), mirroring
//! `beta_phi_smooth_plus_parametric`.

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

const N: usize = 800;
/// True precision of the pure-noise response. Well away from the φ = 1 seed and
/// from the ≈3 null-predictor moment estimate, so a fit that reports either is
/// distinguishable from one that found the truth.
const PHI: f64 = 8.0;
/// Constant true mean: `x` carries NO signal, so the correct smooth is flat.
const MU: f64 = 0.5;

/// `y ~ Beta(MU·PHI, (1−MU)·PHI)` with `x` an independent covariate.
fn make_pure_noise_dataset() -> (Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(0x2369_2369_2369_2369);
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        x.push(rng.normal());
        let yi = rng
            .beta(MU * PHI, (1.0 - MU) * PHI)
            .clamp(1e-6, 1.0 - 1e-6);
        y.push(yi);
    }
    (y, x)
}

/// The #2369 gate: a penalized estimated-φ Beta fit on pure noise must CERTIFY.
///
/// Before the λ-search freeze this refused unconditionally — the fit could not
/// be produced at all, so `fit_from_formula` returned the non-stationary
/// refusal and this `expect` is what fails on a regression.
#[test]
fn beta_pure_noise_smooth_fit_certifies_stationary_optimum() {
    init_parallelism();
    let (y, x) = make_pure_noise_dataset();
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| csv::StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode beta noise dataset");

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &ds, &cfg)
        .expect("estimated-phi Beta fit on pure noise must certify a stationary optimum (#2369)");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };
    assert!(
        fit.fit.beta.iter().all(|b| b.is_finite()),
        "beta coefficients must be finite"
    );

    // The reported precision is the converged-η fixed point, which the accept-fit
    // still refreshes after the search. It must reflect the actual spread of the
    // response — not the φ = 1 seed, and not the ≈3 null-predictor estimate that
    // the frozen-at-warm-start-η path produced. A generous band around the truth
    // (PHI = 8) keeps this a contract gate on the freeze rather than a
    // tolerance-tuned estimator-accuracy assertion.
    let phi = fit
        .fit
        .likelihood_scale
        .fixed_phi()
        .expect("a fitted Beta model records an estimated precision phi");
    assert!(
        phi.is_finite() && (4.0..20.0).contains(&phi),
        "reported precision did not reflect the response spread: phi={phi:.4} \
         (seed=1, null-predictor≈3, truth={PHI})"
    );
}

/// The reporter's sharpest #2369 datum: a PURELY PARAMETRIC estimated-φ Beta
/// fit, with real signal, must certify and recover the slope.
///
/// `y ~ x` carries no `s()` term, so this exercises the same φ-drift seam under
/// a minimal outer coordinate set — it was the case that made "the desync is in
/// smoothing selection" untenable as an explanation and pointed at the
/// dispersion coordinate instead.
#[test]
fn beta_parametric_fit_certifies_and_recovers_slope() {
    init_parallelism();

    const B0: f64 = 0.3;
    const B1: f64 = 0.9;

    let mut rng = SplitMix64::new(0x0FED_CBA9_8765_4321);
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.normal();
        let eta = (B0 + B1 * xi).clamp(-2.2, 2.2);
        let mu = 1.0 / (1.0 + (-eta).exp());
        let yi = rng.beta(mu * PHI, (1.0 - mu) * PHI).clamp(1e-6, 1.0 - 1e-6);
        x.push(xi);
        y.push(yi);
    }

    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| csv::StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode beta signal dataset");

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x", &ds, &cfg)
        .expect("pure-parametric estimated-phi Beta fit must certify (#2369)");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the beta family");
    };

    // Recover the parametric slope by differencing predicted η — the intercept
    // cancels, so this reads B1 directly off the fitted surface. Both grid rows
    // go through ONE design build: a single-row grid at x = 0 would present an
    // identically-zero `x` column, which the design builder rejects as a term
    // that cannot carry a recoverable effect.
    use gam::matrix::LinearOperator;
    use gam::smooth::build_term_collection_design;
    use ndarray::Array2;
    let col = ds.column_map();
    let ix = col["x"];
    let mut grid = Array2::<f64>::zeros((2, ds.headers.len()));
    grid[[0, ix]] = 1.0;
    grid[[1, ix]] = -1.0;
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at grid points");
    let eta = design.design.apply(&fit.fit.beta).to_vec();
    // η(1) − η(−1) = 2·B1.
    let b1_hat = 0.5 * (eta[0] - eta[1]);
    assert!(
        (b1_hat - B1).abs() < 0.15,
        "Beta parametric slope not recovered: got {b1_hat:.4}, true {B1}"
    );
}
