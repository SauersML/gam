//! Regression: `duchon(x, periodic=true)` exposed through the formula DSL
//! must surface a clear actionable error, not a deep "spatial kappa
//! optimization failed: ... periodic Duchon log-kappa derivatives are not
//! defined" panic from the REML inner loop.
//!
//! Background: the periodic Duchon basis IS implemented at the basis layer
//! (`build_periodic_duchon_basis_1d` builds a working wrapped-distance
//! kernel and is exercised by unit tests in `basis.rs`), but the REML
//! hyperparameter pipeline has no working kappa-derivative path for the
//! wrapped-distance kernel — `prepare_duchon_derivative_context` rejects
//! `spec.periodic` outright. As long as that derivative is missing, the
//! formula DSL must steer users to the existing alternatives instead of
//! letting a fit attempt blow up mid-REML.
//!
//! Two equivalent recommended substitutes exist:
//!   1D periodic smoothing       → `s(x, periodic=true, period=2*pi)`
//!   higher-D periodic geometry  → `te(...)` with `bc=['periodic', ...]`

use csv::StringRecord;
use gam::{
    FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const TAU: f64 = std::f64::consts::TAU;

fn make_periodic_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| 1.0 + 0.6 * theta.cos() + 0.3 * (2.0 * theta).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn periodic_duchon_formula_returns_actionable_redirect_error() {
    init_parallelism();
    let data = make_periodic_dataset(200, 0.05, 11);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for formula in &[
        "y ~ duchon(t, periodic=true)",
        "y ~ duchon(t, cyclic=true)",
    ] {
        let err = fit_from_formula(formula, &data, &cfg).err().expect(
            "duchon(..., periodic=true) must fail at fit-time until the periodic kappa derivative is implemented",
        );
        let msg = err.to_string();
        let lower = msg.to_lowercase();
        // Must NOT leak the internal REML "spatial kappa optimization failed"
        // wrapper or the deep "periodic Duchon log-kappa derivatives are not
        // defined" string — the user should see a focused actionable
        // redirect at the term-builder level.
        assert!(
            !lower.contains("spatial kappa optimization failed"),
            "expected term-builder redirect for `{formula}`, got REML wrapper leak: {msg}",
        );
        assert!(
            !lower.contains("global kappa path"),
            "expected term-builder redirect for `{formula}`, got internal-derivative leak: {msg}",
        );
        // Must mention the recommended alternative so the user can act.
        assert!(
            lower.contains("s(") && (lower.contains("periodic=true") || lower.contains("periodic = true")),
            "expected redirect to s(..., periodic=true) for `{formula}`, got: {msg}",
        );
    }
}
