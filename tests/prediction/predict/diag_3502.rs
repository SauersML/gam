//! TEMPORARY diagnostic for #3502 (removed before merge).

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::sync::atomic::{AtomicUsize, Ordering};

struct DiagLogger;
static COUNT: AtomicUsize = AtomicUsize::new(0);

impl log::Log for DiagLogger {
    fn enabled(&self, _m: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        let msg = format!("{}", record.args());
        let keep = msg.contains("refused")
            || msg.contains("[ARC]")
            || msg.contains("frozen")
            || msg.contains("reject")
            || msg.contains("certif")
            || msg.contains("iso-kappa")
            || msg.contains("spatial-iso")
            || msg.contains("OUTER")
            || msg.contains("outer iter");
        if keep {
            let c = COUNT.fetch_add(1, Ordering::Relaxed);
            if c < 3000 {
                let cut: String = msg.chars().take(900).collect();
                eprintln!("[diag3502 {c}] {cut}");
            }
        }
    }
    fn flush(&self) {}
}
static LOGGER: DiagLogger = DiagLogger;

#[test]
fn diag_3502_matern_iso_kappa() {
    init_parallelism();
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }
    let n = 240;
    let mut rng = StdRng::seed_from_u64(251);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
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
    match fit_from_formula("y ~ matern(x)", &data, &cfg) {
        Ok(_) => eprintln!("[diag3502] FIT OK"),
        Err(e) => eprintln!("[diag3502] FIT ERR: {e}"),
    }
    eprintln!("[diag3502] total kept log lines {}", COUNT.load(Ordering::Relaxed));
}
