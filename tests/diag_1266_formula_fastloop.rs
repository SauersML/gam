//! SCRATCH fast-loop repro for #1266 (NOT a committed regression test).
//!
//! Calls `fit_from_formula` directly with the exact failing formula
//! `y ~ s(x, k=20, bs=ps, double_penalty=True)` so the lowering (knot count
//! via `parse_ps_internal_knots`, scan-vs-dense routing, identifiability) is
//! byte-identical to the failing Python path — but builds via
//! `cargo test --no-run` (no maturin, ~1min on a warm target). Used to confirm
//! the inflation reproduces in-process (EDF ~4.96 for double, ~2.0 for single)
//! and as the iteration harness for instrumenting the *actual* dense EDF path.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn linear_records(n: usize, seed: u64) -> (Vec<String>, Vec<StringRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0_f64, 0.15).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let xi = i as f64 / (n as f64 - 1.0);
            let yi = 2.0 + 3.0 * xi + noise.sample(&mut rng);
            StringRecord::from(vec![xi.to_string(), yi.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    (headers, rows)
}

fn fit_edf(formula: &str, seed: u64) -> f64 {
    let (headers, rows) = linear_records(800, seed);
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &ds, &cfg).expect("fit");
    match result {
        FitResult::Standard(fit) => fit.fit.edf_total().unwrap_or(f64::NAN),
        _ => {
            eprintln!("[1266-fastloop] non-standard result variant");
            f64::NAN
        }
    }
}

#[test]
fn formula_fastloop_1266_reproduces_inflation() {
    let on = fit_edf("y ~ s(x, k=20, bs=ps, double_penalty=True)", 0);
    let off = fit_edf("y ~ s(x, k=20, bs=ps, double_penalty=False)", 0);
    eprintln!("[1266-fastloop] double_penalty=True  edf_total={on:.4}");
    eprintln!("[1266-fastloop] double_penalty=False edf_total={off:.4}");
    eprintln!("[1266-fastloop] mgcv target ~2.10; inflation reproduces iff edf(double) >> 2");
    // Report-only — no assertion; this is the localization harness.
}
