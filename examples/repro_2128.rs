//! Fast-iteration harness for issue #2128 — Gamma penalized-smooth REML aborts
//! with "objective returned a non-finite cost" for every seed at moderate/high
//! dispersion (shape ≈ 2, CV ≈ 0.71). Mirrors the data of the regression test
//! `tests/regressions/families/gamma_smooth_reml_highvar_rejects_all_seeds_2128.rs`.
//!
//! Run: `RUST_LOG=debug cargo run --profile release-dev --example repro_2128`

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn build_data(n: usize) -> gam::data::EncodedDataset {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let exp_table: Vec<f64> = (0..n)
        .map(|i| {
            let u = (i as f64 + 0.5) / n as f64;
            -(-u).ln_1p()
        })
        .collect();
    let strides = [7919usize, 6311];
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let g2: f64 = strides
                .iter()
                .enumerate()
                .map(|(k, stride)| exp_table[(i * stride + 101 * k) % n])
                .sum();
            let mu = (0.2 + 1.5 * x[i]).exp();
            mu * g2 / 2.0
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

struct StderrLogger;
impl log::Log for StderrLogger {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}
static LOGGER: StderrLogger = StderrLogger;

fn arg_usize(idx: usize, default: usize) -> usize {
    std::env::args()
        .nth(idx)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    gam::init_parallelism();
    // Positional argv: `repro_2128 [n] [log_level]`, log_level ∈ {0=info,1=debug}.
    log::set_logger(&LOGGER)
        .map(|()| {
            let lvl = if arg_usize(2, 1) >= 1 {
                log::LevelFilter::Debug
            } else {
                log::LevelFilter::Info
            };
            log::set_max_level(lvl)
        })
        .ok();

    let n = arg_usize(1, 200);
    let data = build_data(n);

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    println!("=== parametric control: y ~ x ===");
    match fit_from_formula("y ~ x", &data, &cfg) {
        Ok(FitResult::Standard(fit)) => {
            println!("  OK beta = {:?}", fit.fit.beta.to_vec());
        }
        Ok(_) => println!("  non-standard fit"),
        Err(e) => println!("  ERR {e}"),
    }

    println!("=== gamma smooth: y ~ s(x, k=10) ===");
    match fit_from_formula("y ~ s(x, k=10)", &data, &cfg) {
        Ok(FitResult::Standard(fit)) => {
            println!("  OK beta = {:?}", fit.fit.beta.to_vec());
        }
        Ok(_) => println!("  non-standard fit"),
        Err(e) => println!("  ERR {e}"),
    }
}
