//! THE ONE PUBLISHED SMOOTH-TERM p-VALUE IS THE CALIBRATED ONE.
//!
//! `SmoothTermLrInference` used to publish four tail probabilities of the same
//! statistic — uncorrected, Bartlett-corrected, "conditional" and a bound — and
//! let the reader choose. They do not agree under the null, and the one a
//! reader is most likely to reach for is the one that is wrong: the
//! CONDITIONAL tail reads the statistic against the fixed-`λ` law, pricing the
//! REML `λ̂` as if it had been given rather than chosen from the same data.
//! A smoothing parameter chosen to fit the data inflates the likelihood-ratio
//! statistic of the very term it was chosen for, so that tail is
//! anti-conservative. The audit measured it at size 0.140 on 100 null
//! replicates at `α = 0.05`, against 0.060 for the corrected lane.
//!
//! The report now carries a single `p_value`: the Bartlett-corrected statistic
//! read against the reference with the λ̂-selection replay applied. This test
//! is the contract for that choice, on the design a user actually fits — a
//! REAL nonlinear smooth next to the one under test, so the tested term's
//! reference is built on a fit that carries a second, non-trivial `λ̂`:
//!
//!   1. SIZE. On `y ~ s(x) + s(z)` with `z` null-true, the published `p_value`
//!      for `s(z)` rejects at the nominal rate at BOTH `α = 0.05` and
//!      `α = 0.01`, within `3·SE` of Monte-Carlo error, on a Gaussian and on a
//!      Bernoulli response.
//!   2. THE DELETED LANE IS WRONG. On the same Bernoulli replicates the
//!      conditional tail (the fixed-`λ` law read at the same corrected
//!      statistic, still reachable as a building block of the reference) is
//!      anti-conservative beyond Monte-Carlo error. This is the reason it is
//!      not offered. (On the Gaussian cell its excess is real but inside a
//!      200-replicate band, so it is printed there, not asserted.)
//!   3. NO SILENT FALLBACK. When `λ̂` was chosen but its selection replay
//!      refused, the published p-value is NaN and the refusal is named; it is
//!      never the conditional tail in disguise.
//!   4. POWER. The real term `s(x)`, and a modest nonlinear effect in `z`, are
//!      rejected at `α = 0.05` above the top of the null Monte-Carlo band — so
//!      the size in (1) is not bought by a test that never rejects.
//!
//! The Monte-Carlo band is `3·SE = 3·√(α(1−α)/R)` with nothing added: the
//! claim is that the published p-value is the right size, not that it is close.

use gam::smooth::{SmoothTermLrInference, smooth_term_lr_inference_forspec};
use gam::{
    FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism, materialize,
};

use csv::StringRecord;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Bernoulli, Distribution, Normal};
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum Family {
    /// `y = sin(2πx) + f(z) + N(0, 0.5²)`, identity link.
    Gaussian,
    /// `y ~ Bernoulli(logit⁻¹(−0.2 + 1.2·sin(2πx) + f(z)))`, logit link.
    Bernoulli,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::Gaussian => "gaussian",
            Family::Bernoulli => "binomial",
        }
    }

    /// The design point every claim for this family is made at: two smooths on
    /// two hundred Gaussian rows, or on a hundred Bernoulli ones.
    ///
    /// The Bernoulli cell is the smaller one for its RUNTIME, not its verdict.
    /// Its λ̂-selection replay costs a few seconds on most null draws and
    /// minutes on a few (on 48 null seeds at `n = 100`: a median of 3 s, a
    /// slowest of more than 15 min, on four cores); that cost grows with `n`,
    /// and a cell of this many replicates has to finish inside its test budget.
    fn n(self) -> usize {
        match self {
            Family::Gaussian => 200,
            Family::Bernoulli => 100,
        }
    }
}

/// One replicate of `y ~ s(x) + s(z)` data. `x` always carries a nonlinear
/// effect; `z` carries `z_amplitude · sin(3πz)`, so `z_amplitude = 0` is the
/// null for `s(z)`.
fn replicate(family: Family, n: usize, z_amplitude: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let headers = vec!["y".to_string(), "x".to_string(), "z".to_string()];
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for _ in 0..n {
        let x: f64 = rng.random_range(0.0..1.0);
        let z: f64 = rng.random_range(0.0..1.0);
        let f_x = (2.0 * std::f64::consts::PI * x).sin();
        let f_z = z_amplitude * (3.0 * std::f64::consts::PI * z).sin();
        let y = match family {
            Family::Gaussian => Normal::new(f_x + f_z, 0.5)
                .expect("normal")
                .sample(&mut rng),
            Family::Bernoulli => {
                let eta = -0.2 + 1.2 * f_x + f_z;
                let mu = 1.0 / (1.0 + (-eta).exp());
                if Bernoulli::new(mu).expect("bernoulli p").sample(&mut rng) {
                    1.0
                } else {
                    0.0
                }
            }
        };
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            z.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// The two smooth-term reports of one replicate, `(s(x), s(z))`.
fn fit_reports(
    family: Family,
    data: &gam::data::EncodedDataset,
) -> Result<(SmoothTermLrInference, SmoothTermLrInference), String> {
    let cfg = FitConfig {
        family: Some(family.name().to_string()),
        ..FitConfig::default()
    };
    let mat = materialize("y ~ s(x) + s(z)", data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request");
    };
    let reports = smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .map_err(|error| error.to_string())?;
    let find = |covariate: &str| {
        reports
            .iter()
            .find(|r| r.name.contains(covariate))
            .cloned()
            .ok_or_else(|| format!("no report for s({covariate}) among {} terms", reports.len()))
    };
    Ok((find("x")?, find("z")?))
}

/// One replicate's readout: the published p-values of both terms, the
/// conditional tail of the tested one, and whether the tested term's
/// selection replay refused.
struct Readout {
    p_x: f64,
    p_z: f64,
    p_z_conditional: f64,
    z_replay_refused: bool,
}

/// Run `reps` replicates in parallel and return the readouts of the ones that
/// produced a statistic.
///
/// A replicate whose full fit refuses, or whose null refit refuses (the driver
/// then publishes a NaN statistic for that term), is a SOLVER verdict on that
/// draw, not a calibration datum — the outer-optimizer cluster the size grids
/// in `smooth_term_lr_size_calibration` count the same way. It is counted, not
/// absorbed: more than [`MAX_REFUSED_FRACTION`] of a cell refusing fails the
/// test, and the count is printed.
///
/// A replicate whose statistic is finite but whose p-value is NaN is a
/// selection-replay refusal. That is asserted to be LOUD — the NaN always
/// comes with a refusal label in `ref_df_provenance.selection`, and a refusal
/// label always comes with a NaN — and it is counted under the same cap.
fn run(family: Family, n: usize, z_amplitude: f64, reps: usize) -> Vec<Readout> {
    let outcomes: Vec<Result<Readout, String>> = (0..reps)
        .into_par_iter()
        .map(|rep| {
            let seed = mix_seed(family.name(), n, z_amplitude.to_bits(), rep);
            let data = replicate(family, n, z_amplitude, seed);
            let (x, z) = fit_reports(family, &data)?;
            if !(x.statistic_lr.is_finite() && z.statistic_lr.is_finite()) {
                return Err(format!("rep {rep}: a null refit refused (statistic is NaN)"));
            }
            for term in [&x, &z] {
                let refused = term
                    .ref_df_provenance
                    .selection
                    .decline()
                    .is_some_and(|reason| reason.is_refusal());
                assert_eq!(
                    term.p_value.is_nan(),
                    refused,
                    "rep {rep} {}: p_value={} with selection {:?} — a NaN p-value must \
                     name its refusal, and a refusal must not publish a number",
                    term.name,
                    term.p_value,
                    term.ref_df_provenance.selection.decline().map(|d| d.label())
                );
            }
            let p_z_conditional = z
                .ref_df_provenance
                .conditional_tail_with_bound(z.statistic_corrected)
                .0;
            Ok(Readout {
                p_x: x.p_value,
                p_z: z.p_value,
                p_z_conditional,
                z_replay_refused: z.p_value.is_nan(),
            })
        })
        .collect();
    let mut solver_refusals = Vec::<String>::new();
    let mut readouts = Vec::<Readout>::with_capacity(reps);
    for outcome in outcomes {
        match outcome {
            Ok(readout) => readouts.push(readout),
            Err(refusal) => solver_refusals.push(refusal),
        }
    }
    let replay_refusals = readouts.iter().filter(|r| r.z_replay_refused).count();
    eprintln!(
        "[single p-value] {} n={n} amplitude={z_amplitude}: {}/{reps} solver refusals{}, \
         {replay_refusals}/{reps} selection-replay refusals of s(z)",
        family.name(),
        solver_refusals.len(),
        solver_refusals.first().map_or(String::new(), |first| format!(" (first: {first})")),
    );
    let lost = solver_refusals.len() + replay_refusals;
    assert!(
        lost as f64 <= MAX_REFUSED_FRACTION * reps as f64,
        "{} n={n} amplitude={z_amplitude}: {lost}/{reps} replicates refused, more than the \
         {MAX_REFUSED_FRACTION} a calibration cell can lose and still say anything",
        family.name(),
    );
    readouts.into_iter().filter(|r| !r.z_replay_refused).collect()
}

/// The largest share of a cell's replicates that may refuse (see [`run`]).
const MAX_REFUSED_FRACTION: f64 = 0.05;

fn rate(values: impl Iterator<Item = f64>, alpha: f64) -> (f64, usize) {
    let values: Vec<f64> = values.collect();
    assert!(
        values.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
        "every published p-value must be a probability"
    );
    let rejected = values.iter().filter(|&&p| p <= alpha).count();
    (rejected as f64 / values.len() as f64, values.len())
}

fn mc_band(alpha: f64, reps: usize) -> f64 {
    3.0 * (alpha * (1.0 - alpha) / reps as f64).sqrt()
}

/// Null replicates per family. At 200 the `α = 0.01` band is `±0.021`.
const NULL_REPS: usize = 200;
/// Replicates of the nonlinear-`z` alternative per family.
const POWER_REPS: usize = 40;

/// SIZE of the published p-value for the null `s(z)`, at `α = 0.05` and
/// `α = 0.01`, and power for the real `s(x)` on the same replicates. Returns
/// the conditional tail's size at `α = 0.05` for the caller to judge.
fn assert_null_calibration(family: Family) -> (f64, f64) {
    let n = family.n();
    let readouts = run(family, n, 0.0, NULL_REPS);
    let mut conditional_at_05 = (f64::NAN, f64::NAN);
    for alpha in [0.05, 0.01] {
        let (size, reps) = rate(readouts.iter().map(|r| r.p_z), alpha);
        let (size_conditional, _) = rate(readouts.iter().map(|r| r.p_z_conditional), alpha);
        let band = mc_band(alpha, reps);
        eprintln!(
            "[single p-value] {} n={n} R={reps} α={alpha}: size(p_value)={size:.3} \
             size(conditional)={size_conditional:.3} band=±{band:.3}",
            family.name(),
        );
        assert!(
            (size - alpha).abs() <= band,
            "{} n={n}: published p_value for the null s(z) has size {size:.3} at α={alpha}, \
             outside the Monte-Carlo band {alpha} ± {band:.3} on {reps} replicates",
            family.name(),
        );
        if alpha == 0.05 {
            conditional_at_05 = (size_conditional, alpha + band);
        }
    }
    // The real term is found beside the null one.
    let (power_x, reps) = rate(readouts.iter().map(|r| r.p_x).filter(|p| !p.is_nan()), 0.05);
    eprintln!(
        "[single p-value] {} n={n} R={reps}: power(s(x))@.05={power_x:.3}",
        family.name()
    );
    let floor = detection_floor(0.05, reps);
    assert!(
        power_x > floor,
        "{} n={n}: the real nonlinear s(x) must be rejected at α=0.05 above the null band \
         {floor:.3}; rate {power_x:.3} on {reps}",
        family.name()
    );
    conditional_at_05
}

/// The rejection rate a real effect must exceed: the top of the Monte-Carlo
/// band a correctly sized test's null rejection rate falls in. Beating it
/// says the test rejects beyond what nominal size alone explains — a floor
/// against a test that never rejects, not a power claim for the design.
fn detection_floor(alpha: f64, reps: usize) -> f64 {
    alpha + mc_band(alpha, reps)
}

#[test]
fn published_smooth_p_value_is_calibrated_for_a_gaussian_null_term_beside_a_real_one() {
    init_parallelism();
    assert_null_calibration(Family::Gaussian);
}

/// On the Bernoulli family the deleted conditional lane is not just
/// uncorrected but wrong beyond Monte-Carlo error, which is the reason it is
/// not offered.
#[test]
fn published_smooth_p_value_is_calibrated_for_a_binomial_null_term_and_the_conditional_tail_is_not()
{
    init_parallelism();
    let (size_conditional, upper) = assert_null_calibration(Family::Bernoulli);
    assert!(
        size_conditional > upper,
        "the conditional (fixed-λ) tail was expected to be anti-conservative — that is why it \
         is not published — but its size at α=0.05 is {size_conditional:.3}, inside {upper:.3}"
    );
}

/// POWER against a nonlinear alternative in the tested term: the calibrated
/// p-value is not a test that never rejects.
fn assert_power(family: Family, amplitude: f64) {
    let n = family.n();
    let readouts = run(family, n, amplitude, POWER_REPS);
    let (power, reps) = rate(readouts.iter().map(|r| r.p_z), 0.05);
    eprintln!(
        "[single p-value] {} n={n} amplitude={amplitude} R={reps}: power@.05={power:.3}",
        family.name()
    );
    let floor = detection_floor(0.05, reps);
    assert!(
        power > floor,
        "{} n={n}: a {amplitude}·sin(3πz) effect must be detected at α=0.05 above the null \
         band {floor:.3}; power {power:.3} on {reps}",
        family.name()
    );
}

#[test]
fn published_smooth_p_value_detects_a_nonlinear_gaussian_effect_in_the_tested_term() {
    init_parallelism();
    assert_power(Family::Gaussian, 0.3);
}

#[test]
fn published_smooth_p_value_detects_a_nonlinear_binomial_effect_in_the_tested_term() {
    init_parallelism();
    assert_power(Family::Bernoulli, 0.8);
}

/// Deterministic per-cell, per-replicate seed (FNV-1a), so the whole study is
/// reproducible and no two cells share a stream.
fn mix_seed(label: &str, n: usize, amplitude_bits: u64, rep: usize) -> u64 {
    let mut h = 1469598103934665603u64;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(1099511628211);
    };
    for b in label.bytes() {
        mix(b as u64);
    }
    mix(n as u64);
    mix(amplitude_bits);
    mix(rep as u64);
    mix(0x5167_1e9a); // domain tag for this harness
    h
}
