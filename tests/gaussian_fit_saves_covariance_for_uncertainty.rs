//! Regression: a saved standard Gaussian fit MUST carry the conditional
//! covariance through to predict so that `gam predict --uncertainty` works.
//!
//! Previously the CLI's standard fit path passed
//! `compute_inference: !matches!(family, LikelihoodFamily::GaussianIdentity)`,
//! i.e. Gaussian fits skipped covariance computation as a perf optimization.
//! The saved model's `fit_result.covariance_conditional` field then ended
//! up `None`, and `predict --uncertainty` errored with
//!     "fit result does not contain conditional covariance or a usable
//!      penalized Hessian"
//! on any standard Gaussian fit — the most common usage.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gaussian_smooth_fit_exposes_conditional_covariance() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.10).expect("normal");
    let n = 200usize;
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ smooth(x)", &data, &cfg).expect("gaussian fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };

    let p = fit.fit.beta.len();
    let cov = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("Gaussian fit must expose conditional covariance for predict --uncertainty");
    assert_eq!(
        cov.dim(),
        (p, p),
        "covariance shape {:?} does not match coefficient count {p}",
        cov.dim()
    );
    // Must be PSD-ish: at least all diagonal entries positive.
    for i in 0..p {
        let v = cov[[i, i]];
        assert!(
            v.is_finite() && v >= 0.0,
            "covariance diagonal[{i}] = {v} is not finite-and-nonneg",
        );
    }
    let positive_diag_count = (0..p).filter(|&i| cov[[i, i]] > 0.0).count();
    assert!(
        positive_diag_count >= p - 1, // allow one rank-deficient direction (the intercept's identifiability column)
        "covariance has too many zero diagonals ({}/{} positive); fit may have skipped cov computation",
        positive_diag_count,
        p,
    );
}
