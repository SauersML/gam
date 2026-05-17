//! Periodic 1D B-spline trained on data covering only half the period
//! [0, π] should still produce:
//!   - A finite, bounded prediction over the full period [0, 2π].
//!   - A genuinely periodic fit: f(0) = f(2π).
//!   - Reasonable interpolation in the unobserved half (because the
//!     periodic constraint ties f(π+ε) to f(-π+ε) = f(π−... wait no,
//!     periodic just means f wraps, so unobserved data is unconstrained
//!     except by the smoothness penalty).
//!
//! The risk: the basis is built around the data range, so if the data
//! covers [0, π] but the declared period is 2π, the basis knots might
//! be placed only over [0, π], leaving [π, 2π] in extrapolation territory.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn make_half_period_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(13);
    // Data only in [0, π].
    let u = Uniform::new(0.0, PI).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| 1.0 + 0.6 * theta.cos() + noise.sample(&mut rng))
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
fn periodic_1d_partial_data_predicts_finite_over_full_period() {
    init_parallelism();
    let data = make_half_period_dataset(150);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &data,
        &cfg,
    );
    match result {
        Ok(FitResult::Standard(fit)) => {
            let probes: Vec<f64> = (0..50).map(|i| TAU * (i as f64) / 49.0).collect();
            let n = probes.len();
            let mut m = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                m[[i, 0]] = probes[i];
                m[[i, 1]] = 0.0;
            }
            let design =
                build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild design");
            let pred = design.design.apply(&fit.fit.beta).to_vec();
            assert!(
                pred.iter().all(|v| v.is_finite()),
                "non-finite predictions: {pred:?}"
            );
            let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            eprintln!("[per-1d-half] full-period pred range [{mn:.3}, {mx:.3}]");
            // Training y has range ≈ [0.4, 1.6]. Even with half-period
            // extrapolation, the fit should stay bounded.
            assert!(
                mn > -10.0 && mx < 10.0,
                "periodic 1D half-period fit exploded: [{mn:.3}, {mx:.3}]",
            );
            // C0 wrap check
            let f0 = pred[0];
            let f_end = pred[n - 1];
            let gap = (f0 - f_end).abs();
            assert!(
                gap < 1e-3,
                "periodic 1D half-period fit not periodic: f(0)={f0:.6} f(≈2π)={f_end:.6} gap={gap:.3e}",
            );
        }
        Ok(_) => panic!("expected standard fit"),
        Err(e) => {
            // Acceptable if it rejects with a clear "data does not cover period" message.
            let lower = e.to_string().to_lowercase();
            assert!(
                lower.contains("period") || lower.contains("range") || lower.contains("coverage"),
                "if rejection, must mention period/range/coverage; got: {e}",
            );
        }
    }
}
