//! #2748: a wheel-free reproduction of the `haberman_5yr` benchmark cell.
//!
//! `haberman_5yr` is the one member of #2748's twenty sub-budget benchmark
//! failures that is NOT the `matern` curvature-refusal signature. It dies at
//!
//! ```text
//! NOT STATIONARY (|Pg|=1.101e0 > bound=3.636e-6) ... railed=[5]
//! line_search=StepSizeTooSmall after 50 attempt(s)
//! ```
//!
//! on 305 rows, three P-spline smooths, `k = 8`, binomial-logit — the smallest
//! failing fit in the whole suite. `StepSizeTooSmall` means the direction WAS a
//! descent direction and no step improved the objective, which has exactly two
//! causes: a gradient that disagrees with its objective, or an objective that is
//! not a function of the point it is evaluated at.
//!
//! Run:
//!   cargo run --release --example probe_2748_haberman_outer -- [double_penalty] [n_smooths]
//! e.g. `-- true 3` (the bench cell), `-- false 3`, `-- true 1`.
//!
//! NOT a test — examples skip dev-deps, so the CSV read and the z-scoring are
//! inlined rather than taken from `bench/_run_suite_datasets.py`.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

const FEATURES: [&str; 3] = ["age", "op_year", "axil_nodes"];

/// A pre-z-scored fold frame written by the bench harness: a header row of
/// `age,op_year,axil_nodes,y` followed by the fold's training rows. Used so the
/// benchmark CELL — not just the full dataset — reproduces without a wheel; the
/// folds are stratified with the harness's own `CV_SEED`, which is Rust code
/// living in `gam-pyffi` and not reachable from an example.
fn load_prepared_csv(path: &std::path::Path) -> (Vec<[f64; 3]>, Vec<f64>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("prepared fold frame {}: {error}", path.display()));
    let mut features = Vec::new();
    let mut response = Vec::new();
    let mut header: Option<Vec<String>> = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let Some(columns) = header.as_ref() else {
            header = Some(fields.iter().map(|f| f.to_string()).collect());
            continue;
        };
        let column = |name: &str| -> f64 {
            let index = columns
                .iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("prepared frame has no column '{name}'"));
            fields[index].parse().expect("numeric field")
        };
        features.push([column("age"), column("op_year"), column("axil_nodes")]);
        response.push(column("y"));
    }
    (features, response)
}

/// `bench/datasets/haberman.csv`, the first four columns, `status == 2` as the
/// positive class — verbatim from `_load_haberman_dataset`.
fn load_rows() -> (Vec<[f64; 3]>, Vec<f64>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/datasets/haberman.csv");
    let text = std::fs::read_to_string(&path).expect("bench/datasets/haberman.csv");
    let mut features = Vec::new();
    let mut response = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let parsed: Option<Vec<f64>> = fields[..4].iter().map(|f| f.trim().parse().ok()).collect();
        let Some(values) = parsed else { continue };
        features.push([values[0], values[1], values[2]]);
        response.push(if values[3].round() as i64 == 2 { 1.0 } else { 0.0 });
    }
    (features, response)
}

/// The bench's `zscore_train_test` on a single frame: centre and scale each
/// feature by its own mean and (population) standard deviation.
fn zscore(features: &mut [[f64; 3]]) {
    let n = features.len() as f64;
    for j in 0..3 {
        let mean = features.iter().map(|row| row[j]).sum::<f64>() / n;
        let variance = features.iter().map(|row| (row[j] - mean).powi(2)).sum::<f64>() / n;
        let sd = variance.sqrt();
        let scale = if sd > 0.0 { sd } else { 1.0 };
        for row in features.iter_mut() {
            row[j] = (row[j] - mean) / scale;
        }
    }
}

fn main() {
    init_parallelism();
    let args: Vec<String> = std::env::args().collect();
    // Arg 4 is the log level; the repo's own progress logger owns verbosity
    // (examples skip dev-deps, so `env_logger` is not available here).
    gam_solve::progress_log::init_logging_at(
        gam_solve::progress_log::parse_level_directive(args.get(4).map_or("warn", String::as_str))
            .unwrap_or(log::LevelFilter::Warn),
    );
    let double_penalty = args.get(1).map(|a| a != "false").unwrap_or(true);
    // Arg 2 is either a smooth COUNT (the first `n` features) or an explicit
    // comma-separated feature list, so a two-term isolation does not need a
    // recompile.
    let selected: Vec<&str> = match args.get(2) {
        None => FEATURES.to_vec(),
        Some(raw) => match raw.parse::<usize>() {
            Ok(n) => FEATURES[..n.min(FEATURES.len())].to_vec(),
            Err(_) => raw
                .split(',')
                .map(|name| {
                    *FEATURES
                        .iter()
                        .find(|feature| **feature == name.trim())
                        .unwrap_or_else(|| panic!("unknown haberman feature '{name}'"))
                })
                .collect(),
        },
    };
    let knots: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(8);

    // Arg 5 is an optional prepared (already z-scored) fold frame; without it
    // the probe fits the whole dataset with its own z-scoring.
    let (features, response) = match args.get(5) {
        Some(path) => load_prepared_csv(std::path::Path::new(path)),
        None => {
            let (mut raw, response) = load_rows();
            zscore(&mut raw);
            (raw, response)
        }
    };
    println!(
        "[2748-haberman] n={} positives={} double_penalty={double_penalty} smooths={selected:?} knots={knots}",
        features.len(),
        response.iter().sum::<f64>() as usize,
    );

    let headers: Vec<String> = FEATURES
        .iter()
        .map(|name| name.to_string())
        .chain(std::iter::once("y".to_string()))
        .collect();
    let records: Vec<StringRecord> = features
        .iter()
        .zip(response.iter())
        .map(|(row, y)| {
            StringRecord::from(vec![
                row[0].to_string(),
                row[1].to_string(),
                row[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, records).expect("encode");

    let dp = if double_penalty { "true" } else { "false" };
    let body = selected
        .iter()
        .map(|name| format!("s({name}, type=ps, knots={knots}, double_penalty={dp})"))
        .collect::<Vec<_>>()
        .join(" + ");
    let formula = format!("y ~ {body}");
    println!("[2748-haberman] formula: {formula}");

    let config = FitConfig {
        family: Some("binomial-logit".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    match gam::fit_from_formula(&formula, &data, &config) {
        Ok(gam::FitResult::Standard(standard)) => println!(
            "[2748-haberman] OK in {:.1}s: deviance={:.6e} rho={:?} reml={:.6e}",
            started.elapsed().as_secs_f64(),
            standard.fit.deviance,
            standard.fit.log_lambdas.to_vec(),
            standard.fit.reml_score().unwrap_or(f64::NAN),
        ),
        Ok(other) => println!(
            "[2748-haberman] OK in {:.1}s but returned a non-standard result shape: {:?}",
            started.elapsed().as_secs_f64(),
            std::mem::discriminant(&other),
        ),
        Err(error) => println!(
            "[2748-haberman] FAILED in {:.1}s: {error}",
            started.elapsed().as_secs_f64()
        ),
    }
}
