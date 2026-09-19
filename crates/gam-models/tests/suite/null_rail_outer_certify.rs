//! pyGAM audit B2 (`bench/pygam_audit`, lane `null-rail-certify`): a Poisson or
//! binomial additive fit whose smoothing parameter rails to the top of the box,
//! so that its penalty's limit projects the term onto the penalty's null space,
//! must still end on a certified outer optimum.
//!
//! The fixtures are the audit's Monte Carlo draws (`inference/mc.py`, cells
//! `pois` and `binom`): `y ~ s(x1) + s(x2) + s(x3)` with truth
//! `b0 + a1 sin(2 pi x1) + a3 cos(2 pi x3)`. The outer coordinates are each
//! smooth's wiggle penalty followed by its null-space shrinkage penalty, so
//! `railed=[4]` below is the wiggle penalty of the weak `s(x3)`. The rows are
//! NumPy `default_rng(1000 + rep)` draws, stored as CSV because the generator
//! is not reproducible from Rust.
//!
//! On the released 0.1.267 wheel every fixture raised
//! `Outer smoothing-parameter optimization did not certify a stationary
//! optimum`, with the null coordinate railed at the upper edge of the box:
//!
//! * binomial rep 0 (`inference/repro_binom0.py`): `railed=[4] theta=29.86
//!   box=[-30,30]`, `|Pg|=1.331e-4 > bound=1.010e-4`, `asymptote-rail declined:
//!   interior not stationary`, `line_search=StepSizeTooSmall after 50
//!   attempt(s)`, after 150-198 s;
//! * Poisson rep 163: `railed=[4] theta=30`, `|Pg|=1.577e-3 > bound=1.494e-3`;
//! * Poisson rep 92: `railed=[4] theta=30`, `|Pg|=6.170e-4 > bound=2.477e-4`,
//!   after 24-40 s.
//!
//! A fit is only minted from a converged optimization, so a returned standard
//! fit is itself the certificate; the time bound is the audit's acceptance bar
//! against the multi-minute stalled searches above.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use std::time::{Duration, Instant};

const FORMULA: &str = "y ~ s(x1) + s(x2) + s(x3)";

/// The audit's acceptance bar for one fit.
const ACCEPTANCE_WALL: Duration = Duration::from_secs(10);

const BINOMIAL_REP0: &str = include_str!("fixtures/null_rail_binomial_rep0.csv");
const POISSON_REP92: &str = include_str!("fixtures/null_rail_poisson_rep92.csv");
const POISSON_REP163: &str = include_str!("fixtures/null_rail_poisson_rep163.csv");

fn dataset(csv_text: &str) -> EncodedDataset {
    let mut reader = csv::Reader::from_reader(csv_text.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture header")
        .iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<StringRecord> = reader
        .records()
        .map(|record| record.expect("fixture row"))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture")
}

fn assert_certified_within_acceptance(label: &str, family: &str, csv_text: &str) {
    let data = dataset(csv_text);
    let config = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let started = Instant::now();
    let fit = match fit_from_formula(FORMULA, &data, &config) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("[{label}] an additive {family} formula must produce a standard fit"),
        Err(error) => panic!("[{label}] the outer search must reach a certified optimum: {error}"),
    };
    let elapsed = started.elapsed();
    eprintln!(
        "[null-rail {label}] elapsed={elapsed:?} outer_iterations={} log_lambdas={:?}",
        fit.fit.outer_iterations, fit.fit.log_lambdas,
    );
    assert!(
        elapsed < ACCEPTANCE_WALL,
        "[{label}] the certified fit took {elapsed:?}, over the {ACCEPTANCE_WALL:?} acceptance \
         bar; the released wheel stalled here for minutes in a line search against the railed \
         null coordinate",
    );
}

#[test]
fn binomial_rep0_with_railed_null_coordinate_certifies() {
    assert_certified_within_acceptance("binomial rep 0", "binomial", BINOMIAL_REP0);
}

#[test]
fn poisson_rep163_with_railed_null_coordinate_certifies() {
    assert_certified_within_acceptance("poisson rep 163", "poisson", POISSON_REP163);
}

#[test]
fn poisson_rep92_with_railed_null_coordinate_certifies() {
    assert_certified_within_acceptance("poisson rep 92", "poisson", POISSON_REP92);
}
