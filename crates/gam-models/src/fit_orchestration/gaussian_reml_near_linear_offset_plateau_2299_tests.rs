//! Regression repro for #2299: the dense outer-REML plateau stall on a
//! near-linear `y ~ smooth(x)` Gaussian fit with a nonzero model offset.
//!
//! ROOT CAUSE (see the #2299 entrance-diff investigation): the Python and CLI
//! entrances build a byte-identical `StandardFitRequest` and both reach the same
//! dense `fit_model`; there is NO entrance-configuration divergence. The observed
//! "Python stalls at 200 iters, CLI converges in 5" was CSV-round-trip input
//! perturbation landing in different basins of a near-flat REML surface.
//!
//! The surface is near-flat because the fixture's signal (`0.6 + 1.2·x`, plus a
//! small offset the model subtracts exactly) lies in the smooth's polynomial
//! NULL SPACE `{1, x}`. The REML criterion is then asymptotically flat in `ρ` as
//! `λ → ∞` (the range-space EDF → 0), so the outer optimizer grinds up the
//! infinite-smoothing plateau — checkpoint `ρ ≈ [28.1, −7.6]` ⇒ `λ₀ = e^28.1 ≈
//! 1.6e12` — with the projected gradient plateauing at `|Pg| ≈ 3.2e-2`, far above
//! the n-scaled stationarity bound, until it exhausts the 200-iteration cap.
//!
//! This is the SAME pathology class as #1788
//! ([`super::gaussian_reml_stall_edf_collapse_1788_tests`]), whose fix (grid-free
//! stationary-point enumeration in `gam-solve gaussian_reml.rs`) lands the
//! interior `ρ` optimum on a genuinely wiggly fixture but does not certify the
//! `λ → ∞` asymptote as a stationary point for a near-null-space fixture. The
//! estimand-correct fix is to detect that asymptote analytically (the REML
//! objective has a known limit as the range-space EDF → 0) and certify
//! stationarity-at-infinity as a `+∞`-RAILED convergence, rather than iterating to
//! the cap. The fixture's TRUTH is the null-space (linear) fit, so a `+∞` rail is
//! the correct answer, not a stall.
//!
//! `#[ignore]` is banned workspace-wide, so this cannot be a RED-until-fixed gate.
//! It is instead a GREEN CHARACTERIZATION of the current stall: it asserts the
//! fixture does NOT cleanly converge today (a typed non-convergence refusal, or a
//! minted fit that exhausted the outer iteration cap). When the `gaussian_reml.rs`
//! plateau-rail fix lands, the SAME commit flips this to the estimand-correct
//! convergence assertion (mint + `outer_iterations` well under the cap + a finite,
//! low null-space EDF). The top-level `gam` crate cannot build here (a `build.rs`
//! author tripwire), so the fixture is exercised through `fit_from_formula` in
//! `gam-models`, which builds standalone — mirroring the #1788 repro.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// The #2299 fixture, structurally mirroring the Python repro
/// (`bug_hunt_2299_affine_design_link_wiggle_test.py`): a near-linear signal
/// `0.6 + 1.2·x` plus a small non-linear model offset `0.35·sin(1.7·x) − 0.1`,
/// low Gaussian noise, fit `y ~ smooth(x)` with the offset threaded as a model
/// offset. The exact numpy PCG64 stream is not reproducible under `StdRng`, but
/// the plateau pathology is a STRUCTURAL property of the near-null-space signal,
/// not of one seed — so a deterministic near-linear fixture reproduces it and is
/// a stabler regression pin than a single hard-coded draw.
fn fit_near_linear_with_offset(n: usize, seed: u64) -> Result<StandardFitResult, String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(-1.5_f64, 1.5).expect("valid uniform range");
    let noise = Normal::new(0.0, 0.08).expect("valid normal");

    let headers: Vec<String> = ["x", "offset", "y"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = ux.sample(&mut rng);
        let offset = 0.35 * (1.7 * x).sin() - 0.1;
        let y = 0.6 + 1.2 * x + offset + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            offset.to_string(),
            y.to_string(),
        ]));
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode fixture");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        offset_column: Some("offset".to_string()),
        ..FitConfig::default()
    };
    // The nonzero offset routes this to the dense outer-REML path (the exact O(n)
    // spline scan bails on any nonzero offset), so this exercises exactly the
    // stalling surface — no explicit forcing needed.
    match fit_from_formula("y ~ smooth(x)", &ds, &cfg).map_err(|error| error.to_string())? {
        FitResult::Standard(standard) => Ok(standard),
        _ => Err("expected a Standard fit from a single-smooth Gaussian formula".to_string()),
    }
}

/// #2299 characterization: the near-linear `s(x)` + offset fit does NOT cleanly
/// converge today — it either refuses to mint a fit (typed non-convergence) or
/// mints one only after exhausting the outer iteration loop on the
/// infinite-smoothing plateau. This is GREEN as a record of the open stall.
///
/// FLIP-ON-FIX: when the `gaussian_reml.rs` `λ → ∞` plateau-rail fix lands,
/// replace the body below with the estimand-correct assertion — the fit must MINT
/// (`fit_near_linear_with_offset(...).expect(...)`), converge well under the outer
/// cap (`outer_iterations` in the single/low double digits), and report a finite,
/// low EDF near its polynomial null-space dimension (`0.5 ≤ edf_total < 8`), since
/// the truth lives in `{1, x}`.
#[test]
fn near_linear_offset_fit_currently_stalls_on_the_infinite_smoothing_plateau_2299() {
    // A very low outer-iteration count would be a clean converge — proof the
    // plateau stall is gone. Anything at/above this many outer steps (well below
    // the 200 cap, far above a healthy ~5-step converge) is the grinding stall.
    const STALL_WITNESS_ITERS: usize = 100;
    match fit_near_linear_with_offset(160, 2299) {
        // The stall surfaces as a typed non-convergence refusal (the sealed
        // FitConvergenceEvidence contract refuses to mint a stalled fit)...
        Err(_) => {}
        // ...or as a fit minted only after grinding the outer loop up the plateau.
        Ok(result) => assert!(
            result.fit.outer_iterations >= STALL_WITNESS_ITERS,
            "#2299: expected the near-linear+offset fit to STALL on the \
             infinite-smoothing plateau (typed refusal, or an outer loop grinding to \
             the cap), but it converged in {} outer iterations. If the \
             gaussian_reml.rs plateau-rail fix has landed, flip this characterization \
             to the convergence assertion in the doc comment.",
            result.fit.outer_iterations,
        ),
    }
}
