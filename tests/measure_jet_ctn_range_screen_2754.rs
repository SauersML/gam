//! #2754: the transformation-normal entry point REACHES the measure-jet range
//! resolver, and the fit that follows it mints.
//!
//! ## What is being pinned
//!
//! `length_scale == 0.0` is an unresolved representer range with two resolvers
//! in the tree — the basis builder's pure-geometry median-nearest-node rule and
//! the gam#2750 response screen — and which one a model gets must not depend on
//! which family entry point it took. `fit_bernoulli_marginal_slope_terms` is
//! gated end to end by
//! `measure_jet_range_resolver_entry_point_invariance_2754`, which asserts exact
//! `f64` equality of the realized range against a standard fit on the same
//! response.
//!
//! Here the claim is pinned at the moment it is made: the screen runs before
//! the design is built and logs what it resolved, so the record is counted
//! whether or not the fit that follows converges, and a missing record names
//! the resolver bypass rather than some later failure.
//!
//! The fit itself is asserted too. The fixture is a right-skewed positive
//! (log-normal) response, the shape CTN exists for. This test used to leave the
//! fit's outcome unasserted because CTN refused such a response inside the
//! inner solve (`physical reduced-face first-order KKT failed`), which was the
//! gam#2600 refusal class. That class is closed: the old likelihood
//! renormalized every row by the normal mass between two FITTED endpoints, so
//! the objective had no finite mode, and the untruncated `φ(h)·h′` density that
//! replaced it is convex and coercive. A refusal on this fixture is therefore
//! no longer an open defect in another subsystem to route around; it is a
//! solver failure, and SPEC.md says not to paper over those.
//!
//! ## Why this is its own test binary
//!
//! It installs a process-global `log` sink and raises the max level to `Info`.
//! That is process state, and `log::debug!` evaluates its format arguments
//! eagerly, so a logger installed by one test silently taxes every other test in
//! the same process — `measure_jet`'s target carries a wall-clock speed gate.

use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam::utils::splitmix64;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counts the resolver's own record. The screen emits exactly one `info` line
/// per entry point it runs at, naming the count of terms it moved.
static SCREENED_RECORDS: AtomicUsize = AtomicUsize::new(0);

struct ScreenSink;

impl log::Log for ScreenSink {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }

    fn log(&self, record: &log::Record<'_>) {
        let line = format!("{}", record.args());
        if line.contains("[#2750] screened the representer range")
            && line.contains("transformation-normal")
        {
            println!("[2754-ctn]   {line}");
            SCREENED_RECORDS.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush(&self) {}
}

static SCREEN_SINK: ScreenSink = ScreenSink;

const N: usize = 600;

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_unit(&mut self) -> f64 {
        ((splitmix64(&mut self.state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn dataset() -> gam::data::EncodedDataset {
    let mut rng = SplitMix64::new(0x2754_2026_0811_00c7);
    let headers = vec!["x1".to_string(), "x2".to_string(), "w".to_string()];
    let records: Vec<csv::StringRecord> = (0..N)
        .map(|_| {
            let x1 = rng.next_unit();
            let x2 = rng.next_unit();
            let surface = -0.2
                + 0.7 * (std::f64::consts::PI * x1).sin()
                + 0.3 * (std::f64::consts::PI * x2).cos();
            let w = (0.8 * surface + 0.5 * rng.next_normal()).exp();
            csv::StringRecord::from(vec![
                format!("{x1:.17e}"),
                format!("{x2:.17e}"),
                format!("{w:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode CTN screen fixture")
}

#[test]
fn transformation_normal_entry_reaches_the_measure_jet_range_screen_2754() {
    init_parallelism();
    if log::set_logger(&SCREEN_SINK).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }
    let ds = dataset();
    let config = FitConfig {
        family: Some("transformation-normal".to_string()),
        ..FitConfig::default()
    };

    // The screen's record is counted by the sink as the design is built, so it
    // is read after the fit whatever the fit's outcome; the fit is judged first
    // only so its failure is reported with its own reason.
    let fit = fit_from_formula(
        "w ~ mjs(x1, x2, centers=10, learn_length_scale=false)",
        &ds,
        &config,
    );
    let screened = SCREENED_RECORDS.load(Ordering::Relaxed);
    match fit {
        Ok(FitResult::TransformationNormal(_)) => {
            println!("[2754-ctn] the CTN fit converged on this fixture")
        }
        Ok(_) => panic!("a transformation-normal config must return a TransformationNormal fit"),
        Err(e) => panic!(
            "the transformation-normal fit on a log-normal response must mint (screen records \
             seen: {screened}); its gam#2600 refusal class is closed, so this refusal is a \
             solver failure, not an expected decline: {e}"
        ),
    }

    assert!(
        screened >= 1,
        "the transformation-normal entry point built its covariate design without reaching the \
         #2750 measure-jet range screen: no `[#2750] screened the representer range ... \
         transformation-normal` record was emitted. `length_scale == 0.0` has ONE resolver, and \
         a term that reaches none of them silently takes the basis builder's geometry heuristic \
         — which is a different model, not a different tuning, because lambda cannot move a span."
    );
}
