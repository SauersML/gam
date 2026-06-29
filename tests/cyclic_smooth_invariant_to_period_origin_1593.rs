//! #1593 gauge-invariance guard — a cyclic (periodic) smooth is invariant to the
//! arbitrary PHASE ORIGIN of its period window (`period_start`, i.e. where the
//! wrap seam / knot grid is anchored on the loop).
//!
//! For a periodic covariate (angle, time-of-day, day-of-year) the likelihood is
//! blind to WHERE on the ring we declare the period to "start": the data are the
//! same physical points on the same circle, and `cyclic(theta, period_start=s,
//! period_end=s+2π)` for any `s` describes the identical wrap-around topology.
//! The cyclic penalty `S = D'D` is built on the coefficient RING with
//! wrap-around difference stencils, so it is circulant — no coefficient (hence no
//! seam location) is privileged — and the uniform cyclic knot grid is a rigid
//! rotation of itself under a shift of `period_start`. REML therefore selects one
//! smoothing parameter for a penalty family that is symmetric under the seam
//! shift, and the fitted curve as a function of the PHYSICAL angle must be a
//! property of the data, not of which phase the user anchored the seam to.
//!
//! This is the periodic-smooth sibling of the gauge-invariance family the issue
//! audits (multinomial reference class #1587, simplex ALR reference #1549, tensor
//! `te` margin order — RED). The seam/knot-phase anchor is exactly a
//! frame-anchored choice the penalty could silently depend on (were the penalty
//! NOT circulant, or were the knot grid anchored to the data range rather than
//! rigidly rotated). It SHOULD be invariant by construction, so this test LOCKS
//! THAT IN as a green guard.
//!
//! It fits the SAME physical periodic data under several `period_start` anchors
//! (each a different fraction of a knot spacing around the loop, including a
//! sub-knot rotation so the seam genuinely lands between knots), predicts every
//! fit on a shared PHYSICAL angle grid through the public design + `predict_gam`
//! path (exactly what a user sees), and asserts the curves agree to a tight
//! fraction of the signal range — while a refit under the SAME anchor is
//! confirmed deterministic, so any cross-anchor drift would be a real seam-phase
//! dependence and not refit noise.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use gam_predict::predict_gam;
use ndarray::{Array1, Array2};

const TWO_PI: f64 = std::f64::consts::TAU;

/// Deterministic SplitMix64 → byte-identical data run-to-run (no external RNG),
/// so any cross-anchor disagreement is a fit property, not sampling noise.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform on (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller (one of the pair).
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A genuinely periodic true curve on the circle `[0, 2π)` with several
/// harmonics so the smooth has real wiggle (a flat fit would make the invariant
/// vacuous) and a definite, asymmetric shape so a seam-phase-dependent fit would
/// have something to disagree about. The seam at any anchor cuts through a
/// non-trivial part of the curve.
fn build(seed: u64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let n = 300usize;
    let mut rng = SplitMix64::new(seed);
    let mut thetas = Vec::with_capacity(n);
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let theta = rng.unit() * TWO_PI;
        let f = (theta).sin() + 0.6 * (2.0 * theta).cos() + 0.4 * (3.0 * theta).sin();
        let y = f + 0.2 * rng.normal();
        thetas.push(theta);
        rows.push(StringRecord::from(vec![theta.to_string(), y.to_string()]));
    }
    let headers = ["theta", "y"].into_iter().map(String::from).collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode dataset"),
        thetas,
    )
}

/// Fit a cyclic smooth with the seam anchored at `period_start = s` (period 2π)
/// and return the fitted mean at every physical angle in `grid`, produced through
/// the public design + `predict_gam` path so it is exactly the curve a user sees.
///
/// The prediction grid carries the PHYSICAL angles; each fit's frozen
/// `resolvedspec` carries its own seam anchor, so rebuilding the design at the
/// shared physical grid lands every fit on the same physical points — only the
/// internal seam/knot phase differs.
fn fit_and_predict(
    seam_start: f64,
    data: &gam::data::EncodedDataset,
    grid_angles: &[f64],
) -> Vec<f64> {
    let formula = format!(
        "y ~ cyclic(theta, period_start={seam_start}, period_end={})",
        seam_start + TWO_PI
    );
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) =
        fit_from_formula(&formula, data, &cfg).expect("standard cyclic GAM fit")
    else {
        panic!("expected a standard GAM fit for {formula}");
    };

    let ti = data
        .headers
        .iter()
        .position(|h| h == "theta")
        .expect("theta column");
    let hlen = data.headers.len();
    let m = grid_angles.len();
    let mut grid = Array2::<f64>::zeros((m, hlen));
    for (i, &theta) in grid_angles.iter().enumerate() {
        grid[[i, ti]] = theta;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild cyclic design at the prediction grid");
    let dense = design.design.to_dense();
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gam(dense, fit.fit.beta.view(), offset.view(), family)
        .expect("predict on the angle grid");
    pred.mean.to_vec()
}

#[test]
fn cyclic_smooth_fit_is_invariant_to_period_origin_1593() {
    // A correct seam-phase-invariant cyclic fit reproduces the curve to numerical
    // precision: the penalty is circulant and the knot grid rigidly rotates with
    // the seam, so two anchors fit the identical penalized objective up to
    // linear-algebra round-off. We hold a tight 1e-3 of the signal range — far
    // below the 2–6 % at which the order-DEPENDENT `te` fit drifts — so this is a
    // real (non-vacuous) guard and a regression that anchored the cyclic penalty
    // or knot grid to a privileged seam would trip it.
    const REL_TOL: f64 = 1.0e-3;

    // Eight basis functions ⇒ knot spacing 2π/8. Anchors include a whole-knot
    // shift (2π/8, where circulancy makes the rotation exact even for a
    // grid-anchored penalty) AND sub-knot shifts (the seam lands BETWEEN knots),
    // which only stay invariant if the knot grid truly rotates rigidly with the
    // seam rather than re-snapping to the data range.
    let knot_spacing = TWO_PI / 8.0;
    let anchors = [
        0.0,
        0.37 * knot_spacing,
        knot_spacing,
        1.5 * knot_spacing,
        2.8 * knot_spacing,
    ];

    // A dense shared physical-angle grid spanning the whole circle.
    let grid_angles: Vec<f64> = (0..120).map(|i| (i as f64 + 0.5) / 120.0 * TWO_PI).collect();

    let mut worst_rel = 0.0_f64;
    let mut worst_seed = 0_u64;
    let mut worst_anchor = 0.0_f64;
    for seed in [1_u64, 3, 5] {
        let (data, _thetas) = build(seed);

        let reference = fit_and_predict(0.0, &data, &grid_angles);

        // A refit under the SAME anchor must be deterministic, else cross-anchor
        // drift could not be attributed to the seam frame.
        let reference_again = fit_and_predict(0.0, &data, &grid_angles);
        let refit_noise: f64 = reference
            .iter()
            .zip(&reference_again)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            refit_noise < 1e-9,
            "seed {seed}: same-anchor refit is non-deterministic (max|Δμ̂|={refit_noise:.3e}); \
             cannot attribute cross-anchor drift to the seam frame"
        );

        let max = reference.iter().cloned().fold(f64::MIN, f64::max);
        let min = reference.iter().cloned().fold(f64::MAX, f64::min);
        let range = max - min;
        assert!(
            range > 0.5,
            "seed {seed}: degenerate cyclic fit (signal range {range:.4}); the invariant \
             would be vacuous"
        );

        for &anchor in &anchors[1..] {
            let shifted = fit_and_predict(anchor, &data, &grid_angles);
            let max_abs: f64 = reference
                .iter()
                .zip(&shifted)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let rel = max_abs / range;
            if rel > worst_rel {
                worst_rel = rel;
                worst_seed = seed;
                worst_anchor = anchor;
            }
            eprintln!(
                "[cyclic-origin] seed={seed} anchor={anchor:.4} | max|Δμ̂|/range={rel:.3e} \
                 (refit noise {refit_noise:.3e}, signal range {range:.3})"
            );
        }
    }
    eprintln!(
        "[cyclic-origin] worst max|Δμ̂|/range across seeds/anchors = {worst_rel:.3e} \
         (seed {worst_seed}, anchor {worst_anchor:.4})"
    );

    assert!(
        worst_rel < REL_TOL,
        "cyclic(theta) fits DIFFERENT curves under different period-origin (seam) anchors: \
         worst max|Δμ̂| across seeds/anchors is {worst_rel:.3e} of the signal range \
         (seed {worst_seed}, anchor {worst_anchor:.4}, tol {REL_TOL:.0e}). The cyclic penalty is \
         circulant and the uniform knot grid rotates rigidly with the seam, so the period origin \
         is a pure gauge choice and the fitted periodic curve must be invariant to it \
         (#1593 gauge-invariance class). A drift here is a real seam-anchor dependence of the \
         #1549/#1587 family."
    );
}
