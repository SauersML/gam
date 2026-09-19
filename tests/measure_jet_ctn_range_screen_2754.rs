//! #2754: the transformation-normal entry points REACH the measure-jet range
//! resolver — asserted on the reaching, not on the fit that follows it.
//!
//! ## What is being pinned, and why not end to end
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
//! CTN cannot be gated that way today, and the reason is not this lane's. On a
//! near-noiseless Gaussian response the outer search rails at the box floor and
//! declines to mint a fit (`NOT STATIONARY (|Pg| = 4.224e-1 > 3.493e-2)`,
//! `railed = [0, 1, 2, 3]`), legitimately: a `p_resp × p_cov` tensor can
//! interpolate a smooth surface at that noise level. On a right-skewed positive
//! response — the shape CTN exists for — it refuses in the inner solve instead
//! (`physical reduced-face first-order KKT failed`,
//! `projected_residual_inf = 5.99` against `6.2e-3`), which is the gam#2600
//! refusal class, open and not about ranges at all.
//!
//! Tying a range-resolver gate to an open refusal class in another subsystem
//! would make it red for a reason it does not measure. So this pins the claim
//! that is actually being made — *the resolver is REACHED from this entry
//! point* — at the moment it is reached: the screen runs before the design is
//! built and logs what it resolved, so the record exists whether or not the fit
//! that follows converges. The day gam#2600 lifts, the sibling gate's exact
//! equality assertion is the stronger statement to add here.
//!
//! ## Why this is its own test binary
//!
//! It installs a process-global `log` sink and raises the max level to `Info`.
//! That is process state, and `log::debug!` evaluates its format arguments
//! eagerly, so a logger installed by one test silently taxes every other test in
//! the same process — `measure_jet`'s target carries a wall-clock speed gate.

use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
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

    // The fit's outcome is deliberately not asserted (see the module docs): what
    // is asserted is that the resolver ran before the design was built. Its
    // result is reported either way so a future reader can see which arm of
    // gam#2600 this fixture is in today.
    match fit_from_formula(
        "w ~ mjs(x1, x2, centers=10, learn_length_scale=false)",
        &ds,
        &config,
    ) {
        Ok(_) => println!("[2754-ctn] the CTN fit converged on this fixture"),
        Err(e) => println!(
            "[2754-ctn] the CTN fit declined (not asserted here): {}",
            e.to_string().chars().take(220).collect::<String>()
        ),
    }

    let screened = SCREENED_RECORDS.load(Ordering::Relaxed);
    assert!(
        screened >= 1,
        "the transformation-normal entry point built its covariate design without reaching the \
         #2750 measure-jet range screen: no `[#2750] screened the representer range ... \
         transformation-normal` record was emitted. `length_scale == 0.0` has ONE resolver, and \
         a term that reaches none of them silently takes the basis builder's geometry heuristic \
         — which is a different model, not a different tuning, because lambda cannot move a span."
    );
}
