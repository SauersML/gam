//! #2569: the grouped-binomial REML shape that costs 507–1714 s per fit in the
//! supplied run record, rebuilt from synthetic data with the same cardinalities
//! so the timing distribution can be measured on the current tree.
//!
//! The design (from the issue): 13,897 rows; grouped binomial `y ∈ {0, ½, 1}`
//! with prior weight 2; three `group(...)` ridges of cardinality 174, 14 and 5;
//! one `s(x, k=6)`; one 5-level factor smooth (`bs='fs'`, `k=4`); eight linear
//! terms. Each fit differs from the others only by its seed, exactly as the 32
//! recorded fits differed only by the response draw.
//!
//! The assertion is the one the record failed 17 times in 32: every fit must be
//! minted, i.e. the outer optimization must converge. Wall time per fit is
//! printed (`[2569-SWEEP]`) and summarised, never asserted: a bar needs the
//! measurement first.
use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

const N_ROWS: usize = 13_897;
const GROUP_CARDINALITIES: [usize; 3] = [174, 14, 5];
const FACTOR_LEVELS: usize = 5;
const N_LINEAR: usize = 8;
const N_TRIALS: usize = 2;
const SEEDS: [u64; 3] = [0x2569_0001, 0x2569_0002, 0x2569_0003];

struct SweepRow {
    y: f64,
    w: f64,
    x: f64,
    z: f64,
    factor: usize,
    groups: [usize; 3],
    linear: [f64; N_LINEAR],
}

fn synthetic_rows(seed: u64) -> Vec<SweepRow> {
    let mut rng = StdRng::seed_from_u64(seed);
    let effect = Normal::new(0.0, 0.5).expect("finite scale");
    let slope = Normal::new(0.0, 0.3).expect("finite scale");
    let standard = Normal::new(0.0, 1.0).expect("finite scale");
    let group_effects: Vec<Vec<f64>> = GROUP_CARDINALITIES
        .iter()
        .map(|&cardinality| (0..cardinality).map(|_| effect.sample(&mut rng)).collect())
        .collect();
    let factor_amplitude: Vec<f64> = (0..FACTOR_LEVELS).map(|_| effect.sample(&mut rng)).collect();
    let factor_slope: Vec<f64> = (0..FACTOR_LEVELS).map(|_| effect.sample(&mut rng)).collect();
    let linear_beta: Vec<f64> = (0..N_LINEAR).map(|_| slope.sample(&mut rng)).collect();
    let unit = Uniform::new(0.0_f64, 1.0).expect("unit interval");
    let factor_draw = Uniform::new(0usize, FACTOR_LEVELS).expect("factor levels");
    let group_draws = [
        Uniform::new(0usize, GROUP_CARDINALITIES[0]).expect("group 1 levels"),
        Uniform::new(0usize, GROUP_CARDINALITIES[1]).expect("group 2 levels"),
        Uniform::new(0usize, GROUP_CARDINALITIES[2]).expect("group 3 levels"),
    ];
    (0..N_ROWS)
        .map(|_| {
            let x: f64 = unit.sample(&mut rng);
            let z: f64 = unit.sample(&mut rng);
            let factor = factor_draw.sample(&mut rng);
            let groups = [
                group_draws[0].sample(&mut rng),
                group_draws[1].sample(&mut rng),
                group_draws[2].sample(&mut rng),
            ];
            let mut linear = [0.0; N_LINEAR];
            for value in linear.iter_mut() {
                *value = standard.sample(&mut rng);
            }
            let mut eta = -0.5 + 1.5 * (2.0 * std::f64::consts::PI * x).sin();
            eta += factor_amplitude[factor] * (2.0 * std::f64::consts::PI * z).sin()
                + factor_slope[factor] * z;
            for (block, &level) in groups.iter().enumerate() {
                eta += group_effects[block][level];
            }
            for (value, beta) in linear.iter().zip(linear_beta.iter()) {
                eta += value * beta;
            }
            let p = 1.0 / (1.0 + (-eta).exp());
            let successes = (0..N_TRIALS)
                .filter(|_| unit.sample(&mut rng) < p)
                .count();
            SweepRow {
                y: successes as f64 / N_TRIALS as f64,
                w: N_TRIALS as f64,
                x,
                z,
                factor,
                groups,
                linear,
            }
        })
        .collect()
}

fn dataset(rows: &[SweepRow]) -> gam::data::EncodedDataset {
    let mut headers: Vec<String> = vec![
        "y".into(),
        "w".into(),
        "x".into(),
        "z".into(),
        "f".into(),
        "g1".into(),
        "g2".into(),
        "g3".into(),
    ];
    headers.extend((1..=N_LINEAR).map(|j| format!("x{j}")));
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|row| {
            let mut fields: Vec<String> = vec![
                row.y.to_string(),
                row.w.to_string(),
                row.x.to_string(),
                row.z.to_string(),
                format!("f{}", row.factor),
                format!("a{}", row.groups[0]),
                format!("b{}", row.groups[1]),
                format!("c{}", row.groups[2]),
            ];
            fields.extend(row.linear.iter().map(|value| value.to_string()));
            StringRecord::from(fields)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the synthetic sweep table")
}

fn formula() -> String {
    let linear = (1..=N_LINEAR)
        .map(|j| format!("x{j}"))
        .collect::<Vec<_>>()
        .join(" + ");
    format!("y ~ {linear} + s(x, k=6) + s(z, f, bs='fs', k=4) + group(g1) + group(g2) + group(g3)")
}

#[test]
fn grouped_binomial_sweep_every_fit_is_minted_2569() {
    init_parallelism();
    // The phase clocks and the geometry decision are `log::info!` lines; a
    // test binary without a logger prints none of them.
    gam_runtime::test_support::install_diagnostic_logger();
    let config = FitConfig {
        family: Some("binomial-logit".to_string()),
        weight_column: Some("w".to_string()),
        ..FitConfig::default()
    };
    let formula = formula();
    let mut seconds: Vec<f64> = Vec::with_capacity(SEEDS.len());
    for &seed in &SEEDS {
        let rows = synthetic_rows(seed);
        let data = dataset(&rows);
        let started = Instant::now();
        let outcome = fit_from_formula(&formula, &data, &config);
        let elapsed = started.elapsed().as_secs_f64();
        match &outcome {
            Ok(gam::FitResult::Standard(standard)) => {
                let reml = standard.fit.reml_score().unwrap_or(f64::NAN);
                println!(
                    "[2569-SWEEP] seed={seed:#x} seconds={elapsed:.1} reml={reml:.6e} minted=true"
                );
            }
            Ok(_) => println!("[2569-SWEEP] seed={seed:#x} seconds={elapsed:.1} minted=true (non-standard shape)"),
            Err(error) => println!(
                "[2569-SWEEP] seed={seed:#x} seconds={elapsed:.1} minted=false error={}",
                error.to_string().chars().take(400).collect::<String>()
            ),
        }
        seconds.push(elapsed);
        outcome.unwrap_or_else(|error| {
            panic!(
                "#2569: the grouped-binomial sweep fit (seed {seed:#x}) must be minted; \
                 a fit is only minted from a converged optimization: {error}"
            )
        });
    }
    let mut sorted = seconds.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    println!(
        "[2569-SWEEP] fits={} min={:.1}s median={:.1}s max={:.1}s",
        sorted.len(),
        sorted[0],
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1]
    );
}
