//! TEMPORARY probe (not a kept regression): does a design that PASSES the
//! row-count guard but is rank-deficient (collinear covariates) drive the
//! outer REML loop into the #1089 non-termination? Run with a short external
//! timeout; if it terminates fast, the loop is robust to rank deficiency.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn probe_rank_deficient_passes_guard() {
    let mut rng = StdRng::seed_from_u64(7);
    let ux = Uniform::new(0.0, 1.0).expect("u");
    let noise = Normal::new(0.0, 0.1).expect("n");
    let n = 80; // > 57 floor, passes check_smooth_capacity
    let headers = ["x0", "x1", "x2", "x3", "x4", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x0 = ux.sample(&mut rng);
            // x1..x4 collinear with x0 (tiny jitter) -> rank-deficient design.
            let jit = || 1e-6 * ux.sample(&mut rng);
            let xs = [x0, x0 + jit(), x0 + jit(), x0 + jit(), x0 + jit()];
            let y = (1.3 * x0).sin() + noise.sample(&mut rng);
            let mut f: Vec<String> = xs.iter().map(|v| v.to_string()).collect();
            f.push(y.to_string());
            StringRecord::from(f)
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("enc");
    let formula = "y ~ s(x0, type=ps, knots=7, double_penalty=true) \
                   + s(x1, type=ps, knots=7, double_penalty=true) \
                   + s(x2, type=ps, knots=7, double_penalty=true) \
                   + s(x3, type=ps, knots=7, double_penalty=true) \
                   + s(x4, type=ps, knots=7, double_penalty=true)";
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let start = std::time::Instant::now();
    let result = fit_from_formula(formula, &data, &cfg);
    let elapsed = start.elapsed();
    eprintln!(
        "PROBE_RESULT ok={} elapsed_s={:.2}",
        result.is_ok(),
        elapsed.as_secs_f64()
    );
    if let Ok(FitResult::Standard(fit)) = &result {
        eprintln!(
            "PROBE_FIT outer_iterations={} converged={}",
            fit.fit.outer_iterations, fit.fit.outer_converged
        );
    }
    if let Err(e) = &result {
        eprintln!("PROBE_ERR {}", e);
    }
    assert!(
        elapsed.as_secs() < 300,
        "rank-deficient fit did not terminate promptly: {:.1}s",
        elapsed.as_secs_f64()
    );
}
