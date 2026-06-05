//! Bug hunt (regression, different angle from
//! `bug_hunt_beta_phi_frozen_at_null_predictor`): the precision `phi` a Beta fit
//! *reports* must be the moment estimate evaluated at the **fitted** linear
//! predictor, not at the cold null predictor (`eta = 0`, `mu = 0.5`).
//!
//! The original bug froze `phi` at the very first inner-solve curvature build,
//! when the warm-start `eta` is still ~0, so the moment estimator
//!     `1 + phi = sum_i w_i / sum_i w_i * s_i`,   s_i = (y_i - mu_i)^2 / (mu_i (1 - mu_i))
//! measured the full *marginal* spread of `y` (at `mu = 0.5`) instead of its
//! *conditional* spread around the fitted mean. That collapses `phi` toward the
//! marginal value (~3 here) regardless of the true precision, and the wrong,
//! frozen value is exactly what is persisted into `likelihood_scale`
//! (`EstimatedBetaPhi`) — the value `gam generate` and every prediction-variance
//! consumer read (issue #770), and the value whose smallness attenuates the mean
//! slopes (issue #769).
//!
//! Where the slope test asserts on the *mean* coefficients, this test asserts on
//! the *reported scale metadata* directly, and pins the precise root-cause
//! invariant: the reported `phi` equals the moment estimate at the fitted `eta`
//! (the betareg fixed point) and is clearly NOT the moment estimate at the null
//! predictor. It is RNG-free and deterministic (same fixed-seed sampler).

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

const N: usize = 8000;
const B0: f64 = 0.3;
const B1: f64 = 0.9;
const B2: f64 = -0.6;
// A different true precision and seed from the slope test, so this exercises the
// fixed point on a distinct dataset.
const PHI: f64 = 30.0;

fn make_dataset() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(0x0BAD_F00D_DEAD_BEEF);
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
/// design and applying `beta`.
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

/// Textbook Beta moment estimator `phi = (sum w) / (sum w s) - 1`,
/// `s_i = (y_i - mu_i)^2 / (mu_i (1 - mu_i))`, evaluated at the supplied `mu`.
/// This mirrors `estimate_beta_phi_from_eta` (logit inverse link, unit prior
/// weights) so the test can check the self-consistency invariant without
/// reaching into private solver internals.
fn moment_phi(y: &[f64], mu: &[f64]) -> f64 {
    const MU_EPS: f64 = 1e-9;
    let mut weighted_pearson = 0.0;
    let mut total_weight = 0.0;
    for (&yi, &mui) in y.iter().zip(mu.iter()) {
        let mui = mui.clamp(MU_EPS, 1.0 - MU_EPS);
        let var_unit = mui * (1.0 - mui);
        let resid = yi - mui;
        weighted_pearson += resid * resid / var_unit;
        total_weight += 1.0;
    }
    (total_weight / weighted_pearson - 1.0).max(1e-3)
}

#[test]
fn beta_reported_phi_is_estimated_at_the_fitted_mean_not_the_null_predictor() {
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

    // The precision the fit persists — the exact value `gam generate` and every
    // prediction-variance consumer read (issues #769 / #770).
    let reported_phi = fit
        .fit
        .likelihood_scale
        .fixed_phi()
        .expect("a fitted Beta model must record an estimated precision phi");
    assert!(
        reported_phi.is_finite() && reported_phi > 0.0,
        "reported phi must be finite and positive, got {reported_phi}"
    );

    // Reconstruct the fitted mean at every data row.
    let pts: Vec<(f64, f64)> = x1.iter().zip(x2.iter()).map(|(&a, &b)| (a, b)).collect();
    let eta_fitted = predict_eta(&ds, &fit, &pts);
    let mu_fitted: Vec<f64> = eta_fitted.iter().map(|&e| logistic(e)).collect();

    // Two reference moment estimates:
    //  * at the fitted mean — what a correct fit converges to (the fixed point);
    //  * at the null predictor (mu = 0.5 everywhere) — the marginal-spread value
    //    the bug froze.
    let phi_at_fitted = moment_phi(&y, &mu_fitted);
    let mu_null = vec![0.5_f64; y.len()];
    let phi_at_null = moment_phi(&y, &mu_null);

    // Sanity on the dataset: the conditional precision is materially larger than
    // the marginal one (otherwise the test could not distinguish the two).
    assert!(
        phi_at_fitted > 1.5 * phi_at_null,
        "test mis-set-up: conditional phi {phi_at_fitted:.3} not clearly above \
         marginal phi {phi_at_null:.3}"
    );

    // Root-cause invariant #1: the reported phi is the moment estimate at the
    // *fitted* mean (the betareg fixed point), to a tight relative tolerance.
    let rel_err = (reported_phi - phi_at_fitted).abs() / phi_at_fitted;
    assert!(
        rel_err < 0.05,
        "reported phi {reported_phi:.4} is not the moment estimate at the fitted mean \
         {phi_at_fitted:.4} (rel err {rel_err:.4}); the precision is not being refreshed \
         from the converged predictor"
    );

    // Root-cause invariant #2: the reported phi is decisively NOT the frozen
    // null-predictor value. A regression that re-freezes phi at the cold start
    // would land near `phi_at_null` and trip this.
    assert!(
        reported_phi > 1.5 * phi_at_null,
        "reported phi {reported_phi:.4} collapsed toward the null-predictor moment estimate \
         {phi_at_null:.4} — phi is frozen at mu = 0.5 again (regression of #769)"
    );
}
