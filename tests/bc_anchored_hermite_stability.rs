//! Regression guard for cycle-18 fix: `bc=anchored` pins BOTH value AND
//! first derivative at the endpoint (Hermite-style C¹ pin) rather than
//! value-only.
//!
//! Background: the value-only `Anchored` semantic was numerically unstable
//! when training data was sparse near the pinned endpoint — the basis
//! could swing arbitrarily steeply between the pinned point and the next
//! data point because only curvature was penalized. The classic failure
//! mode was `s(x, bc=anchored, k=20)` on data with 20 sparse points in
//! [0, 0.5] and 2000 dense in [0.5, 1]: the fit dove to f(x=0.005)=-1.78
//! against a truth value of +0.03 — a ~7σ blow-up.
//!
//! After fix: `Anchored` co-pins the slope at the endpoint, eliminating
//! the swing-direction degree of freedom. The same probe now gives
//! pred(x=0.005) within ±1.0 of truth, span ratio ≈ 1.0 (no more
//! 54% overshoot).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const PI: f64 = std::f64::consts::PI;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        )
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform_01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.uniform_01() - 1.0;
            let v = 2.0 * self.uniform_01() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }
}

fn make_sparse_dense_dataset() -> gam::data::EncodedDataset {
    let sigma = 0.05_f64;
    let n_sparse = 20;
    let n_dense = 2000;
    let mut rng = Lcg::new(505);
    let mut x = Vec::with_capacity(n_sparse + n_dense);
    for _ in 0..n_sparse {
        x.push(0.5 * rng.uniform_01());
    }
    for _ in 0..n_dense {
        x.push(0.5 + 0.5 * rng.uniform_01());
    }
    let f = |t: f64| (2.0 * PI * t).sin() + 0.3 * t;
    let y_noisy: Vec<f64> = x.iter().map(|&t| f(t) + sigma * rng.normal()).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn truth(t: f64) -> f64 {
    (2.0 * PI * t).sin() + 0.3 * t
}

#[test]
fn bc_anchored_sparse_dense_does_not_blow_up_near_pin() {
    init_parallelism();
    let data = make_sparse_dense_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bc=anchored, k=20)", &data, &cfg)
        .expect("bc=anchored fit must succeed");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    // Dense probe across the sparse region [0.005, 0.495].
    let probes: Vec<f64> = (0..100).map(|i| 0.005 + 0.49 * (i as f64) / 99.0).collect();
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &v) in probes.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    // Truth across the sparse region has values roughly in [-0.5, 0.5]; with
    // the Hermite pin in place the worst pointwise deviation must stay below
    // ~1.0 (before fix it was 1.78). Use 1.3 as a generous guard with some
    // headroom for stochastic variation.
    let mut worst_dev = 0.0_f64;
    for (i, &xt) in probes.iter().enumerate() {
        let dev = (pred[i] - truth(xt)).abs();
        if dev > worst_dev {
            worst_dev = dev;
        }
    }
    eprintln!("[hermite] worst sparse dev = {worst_dev:.4}");
    assert!(
        worst_dev < 1.3,
        "bc=anchored Hermite fix regressed: worst sparse dev {worst_dev:.4} >= 1.3 (pre-fix was 1.78)",
    );
    // Predictions across the sparse region must stay within ±2 (truth max
    // amplitude ≈ 0.5).
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        mn > -2.0 && mx < 2.0,
        "bc=anchored predictions exploded in sparse region: [{mn:.3}, {mx:.3}]",
    );
}

#[test]
fn bc_anchored_pins_slope_zero_at_basis_boundary() {
    init_parallelism();
    let data = make_sparse_dense_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bc=anchored, k=20)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    // Find training data extremes (the basis knot boundaries).
    let values = data.values.column(0);
    let x_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Estimate f'(x_min) and f'(x_max) via tiny one-sided differences just
    // inside the basis support.
    let h = 1e-5_f64;
    let mut m = Array2::<f64>::zeros((4, 2));
    m[[0, 0]] = x_min;
    m[[1, 0]] = x_min + h;
    m[[2, 0]] = x_max - h;
    m[[3, 0]] = x_max;
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    let slope_left = (pred[1] - pred[0]) / h;
    let slope_right = (pred[3] - pred[2]) / h;
    eprintln!("[hermite] slope@x_min={slope_left:.5e} slope@x_max={slope_right:.5e}");
    // Both slopes should be ~0 (Hermite pin). Allow some slack for
    // smooth-floor + finite-difference noise.
    assert!(
        slope_left.abs() < 5e-2,
        "Hermite anchor failed to pin left slope: f'(x_min)={slope_left:.5e}",
    );
    assert!(
        slope_right.abs() < 5e-2,
        "Hermite anchor failed to pin right slope: f'(x_max)={slope_right:.5e}",
    );
}
