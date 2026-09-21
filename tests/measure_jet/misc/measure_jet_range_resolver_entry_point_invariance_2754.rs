//! #2754/#2761 regression gate: the realized measure-jet representer range is a
//! property of `(rows, declaration, screening response)` — **not** of the family
//! entry point the fit happened to take.
//!
//! ## The defect this pins
//!
//! `length_scale == 0.0` is an UNRESOLVED request, and the tree carries two
//! resolvers for it: a pure-geometry rule inside the basis builder (the median
//! nearest-node spacing) and the gam#2750 response screen. `fit_standard_model`
//! runs the screen so every standard-fit branch gets the same one. The Bernoulli
//! marginal-slope family has its own entry point and did not pass through it, so
//! the identical declaration on byte-identical rows realized two different spans:
//!
//! ```text
//!   gaussian entry, screened   ell = 2.5197
//!   BMS marginal, geometry     ell = 1.0807     (2.33x apart)
//!   BMS slope, geometry     ell = 1.0807
//!   ...on the SAME 10 centers, the same extent, the same band floor 1.0807.
//! ```
//!
//! (Those digits are from the #2754 length-scale sweep on the
//! #1041 parity fixture, which is where the defect was found. This file's own
//! fixture is smaller and prints its own numbers; what it asserts is the
//! equality, not the value.)
//!
//! `ell` decides WHICH span the representers occupy and `lambda` cannot move a
//! span, so that is not a tuning difference between entry points; it is a
//! different model reached by typing a different family name. gam#2750 measured
//! the geometry heuristic landing 21.7 nats away from the criterion's global
//! optimum and gam#2761 measured its span floor four orders above the chosen
//! range, so "different" is not "equivalent".
//!
//! ## Why the assertion is exact equality rather than a tolerance
//!
//! The screen is a deterministic function of `(feature columns, response,
//! weights, spec)`. Handed the same four, the two entry points must return the
//! same `f64`. A tolerance here would hide exactly the failure mode being
//! pinned: a second resolver that happens to land nearby on this fixture.
//!
//! The comparison arm pins `learn_length_scale=false` because the standard entry
//! LEARNS the range after seeding it, and this test is about the seed. BMS
//! freezes the dial itself (its coupled marginal/slope pair cannot carry a
//! design-moving coordinate), so its realized range IS the seed.

use gam::families::bms::BernoulliMarginalSlopeFitResult;
use gam::terms::smooth::{SmoothBasisSpec, TermCollectionSpec};
use gam::utils::splitmix64;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_math::probability::normal_cdf;

const N: usize = 800;
const CENTERS: usize = 10;

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

fn beta_true(x1: f64) -> f64 {
    0.2 + 0.9 * x1
}

fn alpha_true(x1: f64, x2: f64) -> f64 {
    -0.2 + 0.7 * (std::f64::consts::PI * x1).sin() + 0.3 * (std::f64::consts::PI * x2).cos()
}

/// Two responses on one set of rows, because the families being compared do not
/// accept the same one, and each entry point is therefore compared against a
/// standard fit **on its own response**. That is the contract anyway: the screen
/// is a function of (columns, response, weights, spec), so equality is only
/// required where all four match.
///
/// `y` is the binary probit draw the marginal-slope family requires. Handing
/// that to CTN is a degenerate transformation problem and it refuses — measured,
/// not assumed: 324 inner cycles, `rejected_by_nonconvergence`.
///
/// `w` is what CTN is FOR: a right-skewed positive response,
/// `w = exp(0.8·α(x1,x2) + 0.5·N(0,1))`, whose transformation to normality is
/// nontrivial and whose noise is large enough that smoothing has work to do.
/// The obvious alternative — `α(x1,x2) + 0.1·N(0,1)`, a near-noiseless Gaussian
/// — was tried first and is the wrong fixture for this family: the outer search
/// rails at the box floor on three of five coordinates (`ρ = −10`, i.e. λ ≈ 0)
/// and declines to mint a fit, because a `p_resp × p_cov` tensor can interpolate
/// a smooth surface at that noise level and the REML optimum is genuinely at the
/// edge. A gate has to run its family on a problem that family can solve.
fn dataset() -> gam::data::EncodedDataset {
    let mut rng = SplitMix64::new(0x2754_2026_0811_0007);
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "y".to_string(),
        "z".to_string(),
        "w".to_string(),
    ];
    let mut rng_y = SplitMix64::new(0x2754_2026_0811_0008);
    let records: Vec<csv::StringRecord> = (0..N)
        .map(|_| {
            let x1 = rng.next_unit();
            let x2 = rng.next_unit();
            let z = rng.next_normal();
            let p = normal_cdf(alpha_true(x1, x2) + beta_true(x1) * z).clamp(1e-9, 1.0 - 1e-9);
            let y = f64::from(rng_y.next_unit() < p);
            let w = (0.8 * alpha_true(x1, x2) + 0.5 * rng_y.next_normal()).exp();
            csv::StringRecord::from(vec![
                format!("{x1:.17e}"),
                format!("{x2:.17e}"),
                format!("{y:.17e}"),
                format!("{z:.17e}"),
                format!("{w:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode entry-point fixture")
}

fn realized_ell(spec: &TermCollectionSpec, what: &str) -> f64 {
    spec.smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec: mj, .. } => Some(mj.length_scale),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{what} resolved spec must carry the mjs term"))
}

/// The frozen center layout, so a range difference cannot be explained away as
/// the two entry points having realized different geometry.
fn realized_geometry(spec: &TermCollectionSpec, what: &str) -> (usize, Vec<f64>) {
    spec.smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec: mj, .. } => {
                let band = mj
                    .frozen_quadrature
                    .as_ref()
                    .map(|q| q.eps_band.clone())
                    .unwrap_or_default();
                let m = match &mj.center_strategy {
                    gam::terms::basis::CenterStrategy::UserProvided(c) => c.nrows(),
                    _ => 0,
                };
                Some((m, band))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("{what} resolved spec must carry the mjs term"))
}

fn fit_bms(body: &str, ds: &gam::data::EncodedDataset) -> BernoulliMarginalSlopeFitResult {
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        link: Some("probit".to_string()),
        slope_formula: Some(body.to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(&format!("y ~ {body}"), ds, &config) {
        Ok(FitResult::BernoulliMarginalSlope(fit)) => fit,
        Ok(_) => panic!("expected a BernoulliMarginalSlope fit for '{body}'"),
        Err(e) => panic!("bms fit '{body}': {e}"),
    }
}

#[test]
fn measure_jet_auto_range_is_the_same_through_every_family_entry_point_2754() {
    gam::init_parallelism();
    let ds = dataset();

    // Arm 1 — the standard entry, screened against `y` and with the learning
    // dial pinned off so what is read back is the SEED.
    let standard = match fit_from_formula(
        &format!("y ~ mjs(x1, x2, centers={CENTERS}, learn_length_scale=false)"),
        &ds,
        &FitConfig::default(),
    ) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => panic!("expected a standard fit"),
        Err(e) => panic!("standard fit: {e}"),
    };
    let standard_ell = realized_ell(&standard.resolvedspec, "standard");
    let standard_geometry = realized_geometry(&standard.resolvedspec, "standard");

    // Arm 2 — the BMS entry, same rows, same declaration. Its marginal block is
    // screened against the same `y`, so its seed must be the same number.
    let bms = fit_bms(&format!("mjs(x1, x2, centers={CENTERS})"), &ds);
    let marginal_ell = realized_ell(&bms.marginalspec_resolved, "bms marginal");
    let marginal_geometry = realized_geometry(&bms.marginalspec_resolved, "bms marginal");
    let slope_ell = realized_ell(&bms.slopespec_resolved, "bms slope");

    println!(
        "[#2754 entry-point] standard ell={standard_ell:.6} bms-marginal ell={marginal_ell:.6} \
         bms-slope ell={slope_ell:.6} | geometry standard={standard_geometry:?} \
         bms={marginal_geometry:?}"
    );

    // The controls first: a range difference is only a resolver difference if
    // the two entry points realized the same geometry to begin with.
    assert_eq!(
        standard_geometry, marginal_geometry,
        "the two entry points must realize the same measure-jet geometry (centers, band) on the \
         same rows; they did not, so the range comparison below would not be about the resolver"
    );

    assert_eq!(
        standard_ell, marginal_ell,
        "the same mjs declaration on the same rows realized two different representer ranges \
         through two family entry points (standard {standard_ell:.6} vs BMS marginal \
         {marginal_ell:.6}). `length_scale == 0.0` has ONE resolver — the #2750 response screen — \
         and every entry point must reach it; lambda cannot move a span, so this is a different \
         model, not a different tuning."
    );

    // Arm 3 (the transformation-normal entry) does NOT live here. It takes the
    // same bypass and the same fix, but this fixture's near-noiseless Gaussian
    // response is not a CTN fixture: the outer search legitimately rails at the
    // box floor (a `p_resp x p_cov` tensor interpolates a smooth surface at low
    // noise). The CTN arm is gated on a log-normal response, the shape CTN
    // exists for, in `tests/measure_jet_ctn_range_screen_2754.rs`, which asserts
    // both that the screen's record is emitted before the design is built and
    // that the fit converges to a transformation-normal model (gam#2600 closed
    // the inner-solve refusal that once blocked it).

    // The slope block is screened against its OWN target (the first-order
    // score surrogate), so it is not required to equal the marginal's range, and
    // this gate deliberately does not assert a value for it.
    //
    // It is tempting to assert `slope_ell != eps_band[0]` — the unscreened
    // geometry heuristic is exactly the band floor by construction, so that
    // inequality reads like "the screen ran". It is not: a screen that RAN and
    // whose criterion preferred the floor produces the identical number, and on
    // this fixture it does. An assertion that cannot distinguish its two
    // outcomes is not a gate, it is a coin, so what is pinned here is only that
    // the range is a usable one, and the slope target itself is graded where
    // it can be — against its derivation, in
    // `slope_screen_surrogate_tracks_the_slope_surface_not_the_marginal_2754`.
    let band_floor = marginal_geometry.1.first().copied().unwrap_or(f64::NAN);
    assert!(
        band_floor.is_finite() && band_floor > 0.0,
        "the frozen quadrature must carry a positive band floor"
    );
    assert!(
        slope_ell.is_finite() && slope_ell >= band_floor,
        "the BMS slope range {slope_ell} must be finite and at or above the node-spacing \
         floor {band_floor} — below it neighbouring representers stop overlapping at all"
    );
}
