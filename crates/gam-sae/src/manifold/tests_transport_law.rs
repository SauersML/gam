//! Transport-law measurement tests: does layer-to-layer transport of a fitted
//! circle atom obey the phase-shift law?
//!
//! Two arms, both built as SYNTHETIC 2-layer crosscoders through the landed M1
//! driver [`SaeManifoldTerm::run_multiblock_reml_fit`] (mirroring the fixtures in
//! `tests_crosscoder_multiblock`):
//!
//! 1. **Planted phase shift** — layer 2 IS layer 1 reparameterized by a constant
//!    phase `φ0` (`Y_2(θ) = circle(θ + 2π·φ0)` on the SAME frame). The fitted
//!    transport must recover `φ0` with `phase_r2 > 0.95` and
//!    `smooth_r2 − phase_r2 < 0.02` — the law holds, the extra harmonics buy
//!    nothing.
//! 2. **Planted nonlinear transport** — layer 2 is a *squashed* image of the same
//!    circle (a different-shape ellipse, not a phase rotation), so projecting the
//!    round layer-1 image onto it is a genuinely nonlinear (2nd-harmonic) map. The
//!    verdict must FLIP: `smooth_r2 − phase_r2` clears a margin derived from the
//!    phase-shift arm's own gap.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, CrosscoderLayer, LatentManifold, OutputBlock, PeriodicHarmonicEvaluator,
    SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm, TwoBlockRemlControls, measure_atom_transport,
};

const ON: f64 = 6.0;

/// Deterministic xorshift noise in `[-1, 1)`, so a layer carries a known tiny
/// noise scale without an RNG dependency (same generator as the crosscoder
/// fixtures).
fn noise_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 52) as f64) - 1.0
    }
}

/// A cold K=1 circle atom at augmented width `p_tot`, seeded at the true chart
/// coordinates `coords` (mirrors `tests_crosscoder_multiblock::circle_atom`).
fn circle_atom(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> SaeManifoldAtom {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    SaeManifoldAtom::new(
        "cc",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone())
}

/// A K=1 softmax term (single always-on atom) at augmented width `p_tot`.
fn build_k1(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = coords.nrows();
    let atom = circle_atom(evaluator, coords, p_tot);
    let logits = Array2::<f64>::from_elem((n, 1), ON);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

/// One joint fit at equal block weight then one closed-form λ read — the honest
/// single-sweep regime for these small-`n`, `K = 1`-circle synthetic problems
/// (the `(fit, λ)` alternation is not contractive here; see the crosscoder
/// fixtures). A generous inner budget so the near-noise-free curves fit cleanly.
fn controls() -> TwoBlockRemlControls {
    TwoBlockRemlControls {
        max_sweeps: 1,
        inner_max_iter: 120,
        step_size: 1.0,
        ridge_ext_coord: 1e-6,
        ridge_beta: 1e-6,
        log_lambda_tol: 1e-3,
    }
}

/// Circular mean (in turns, `[-½, ½)`) of the empirical drift `t'_g − t_g` read
/// off a report's transport samples — an `s`/gauge-independent recovery of the
/// constant phase offset.
fn mean_drift_turns(grid: &[(f64, f64)]) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let (s, c) = grid.iter().fold((0.0, 0.0), |(a, b), &(t, tp)| {
        let d = tp - t;
        (a + (two_pi * d).sin(), b + (two_pi * d).cos())
    });
    s.atan2(c) / two_pi
}

/// Arm 1: layer 2 is an exact constant-phase reparameterization of layer 1. The
/// fitted transport recovers the phase shift, and the smooth alternative buys
/// essentially nothing over the phase-shift law.
#[test]
fn planted_phase_shift_transport_obeys_the_law() {
    let n = 128usize;
    let p_x = 4usize;
    let p_2 = 4usize;
    let phi0 = 0.15_f64; // planted phase offset in turns
    let sigma = 0.004_f64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut y2 = Array2::<f64>::zeros((n, p_2));
    let mut nx = noise_stream(0x7a11_0001);
    let mut n2 = noise_stream(0x7a11_0002);
    let two_pi = std::f64::consts::TAU;
    for i in 0..n {
        let theta = two_pi * (i as f64 / n as f64);
        let shifted = theta + two_pi * phi0;
        // Anchor image: a round circle carrying a 2nd harmonic too.
        z[[i, 0]] = theta.cos() + sigma * nx();
        z[[i, 1]] = theta.sin() + sigma * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos() + sigma * nx();
        z[[i, 3]] = 0.5 * (2.0 * theta).sin() + sigma * nx();
        // Layer 2 = the SAME frame at the SHIFTED angle: an exact phase shift.
        y2[[i, 0]] = shifted.cos() + sigma * n2();
        y2[[i, 1]] = shifted.sin() + sigma * n2();
        y2[[i, 2]] = 0.5 * (2.0 * shifted).cos() + sigma * n2();
        y2[[i, 3]] = 0.5 * (2.0 * shifted).sin() + sigma * n2();
    }

    let p_tot = p_x + p_2;
    let (mut term, mut rho) = build_k1(&evaluator, &coords, p_tot);
    term.set_guards_enabled(false);
    let mut blocks = vec![OutputBlock::new("layer2", y2.clone(), 0.0).unwrap()];
    term.run_multiblock_reml_fit(z.view(), &mut blocks, &mut rho, None, controls())
        .unwrap();

    let layout = term
        .crosscoder_layout()
        .expect("multiblock fit installs a crosscoder layout")
        .clone();
    let report = measure_atom_transport(&term, &layout, 0, 256).unwrap();

    // The LAW: the phase-shift model fits the transport, and the smooth
    // alternative's extra harmonics buy essentially nothing.
    assert!(
        report.phase_r2 > 0.95,
        "phase-shift fit should explain the transport: phase_r2 = {}",
        report.phase_r2
    );
    assert!(
        report.law_gap() < 0.02,
        "the smooth alternative should buy < 0.02 R² over the phase law; gap = {} \
         (phase_r2 = {}, smooth_r2 = {})",
        report.law_gap(),
        report.phase_r2,
        report.smooth_r2
    );
    assert!(
        report.law_holds(0.02),
        "law verdict should HOLD for a planted phase shift"
    );

    // Recovery: the empirical constant drift magnitude equals the planted φ0.
    let recovered = mean_drift_turns(&report.transport_grid).abs();
    assert!(
        (recovered - phi0).abs() < 0.05,
        "recovered phase magnitude {recovered} should match planted φ0 = {phi0}"
    );
}

/// Arm 2: layer 2 traces the SAME circle but under a nonlinear reparameterization
/// `θ' = θ + a·sin θ` (the brief's planted nonlinear transport). Both layer images
/// are the unit circle, so projecting the round layer-1 image onto layer 2 recovers
/// the reparameterization inverse `g⁻¹` — a large sinusoidal drift a constant phase
/// shift cannot represent. The verdict FLIPS: the smooth alternative clears a
/// margin derived from the phase-law tolerance (5× the `0.02` phase-arm bound).
#[test]
fn planted_nonlinear_transport_flips_the_verdict() {
    let n = 160usize;
    let p_x = 2usize;
    let p_2 = 2usize;
    let a = 0.8_f64; // reparam amplitude; a < 1 keeps θ ↦ θ + a·sin θ monotone
    let sigma = 0.002_f64;
    // H = 5 capacity so the fitted layer-2 decoder represents the nonlinearly
    // reparameterized circle (a Jacobi–Anger series whose harmonics J_n(a) decay
    // past n ≈ 5) cleanly; the smooth alternative then has ample harmonic order to
    // fit the drift, which the constant phase model cannot.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(11).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut y2 = Array2::<f64>::zeros((n, p_2));
    let mut nx = noise_stream(0x7a12_0001);
    let mut n2 = noise_stream(0x7a12_0002);
    let two_pi = std::f64::consts::TAU;
    for i in 0..n {
        let theta = two_pi * (i as f64 / n as f64);
        let reparam = theta + a * theta.sin(); // θ' = θ + a·sin θ
        // Anchor: a round circle traced uniformly.
        z[[i, 0]] = theta.cos() + sigma * nx();
        z[[i, 1]] = theta.sin() + sigma * nx();
        // Layer 2: the SAME circle traced under the nonlinear reparameterization.
        y2[[i, 0]] = reparam.cos() + sigma * n2();
        y2[[i, 1]] = reparam.sin() + sigma * n2();
    }

    let p_tot = p_x + p_2;
    let (mut term, mut rho) = build_k1(&evaluator, &coords, p_tot);
    term.set_guards_enabled(false);
    let mut blocks = vec![OutputBlock::new("layer2", y2.clone(), 0.0).unwrap()];
    term.run_multiblock_reml_fit(z.view(), &mut blocks, &mut rho, None, controls())
        .unwrap();

    let layout = term.crosscoder_layout().unwrap().clone();
    let nonlinear = measure_atom_transport(&term, &layout, 0, 256).unwrap();

    // The phase-law tolerance (0.02, the arm-1 bound) is the reference; a nonlinear
    // transport must beat it by a full order of magnitude — the two verdicts are
    // separated by the measurement, not a tuned threshold.
    let phase_tol = 0.02_f64;
    assert!(
        nonlinear.smooth_r2.is_finite() && nonlinear.phase_r2.is_finite(),
        "R² must be finite: phase_r2 = {}, smooth_r2 = {}",
        nonlinear.phase_r2,
        nonlinear.smooth_r2
    );
    assert!(
        nonlinear.law_gap() > 5.0 * phase_tol,
        "a reparameterized transport is nonlinear: the smooth alternative must beat the phase \
         law by > {} R² (5× the phase-law tolerance); gap = {} (phase_r2 = {}, smooth_r2 = {})",
        5.0 * phase_tol,
        nonlinear.law_gap(),
        nonlinear.phase_r2,
        nonlinear.smooth_r2
    );
    assert!(
        !nonlinear.law_holds(phase_tol),
        "law verdict should FLIP (not hold) for a nonlinear transport"
    );
    // The smooth alternative should itself fit well — the nonlinear map IS a
    // few-harmonic drift, just not a constant one.
    assert!(
        nonlinear.smooth_r2 > 0.9,
        "the smooth (few-harmonic) alternative should capture the reparam drift: smooth_r2 = {}",
        nonlinear.smooth_r2
    );

    // The deviation locus is a real chart location where linear transport breaks.
    let locus = nonlinear.deviation_locus().expect("non-empty transport grid");
    assert!(
        (0.0..1.0).contains(&locus),
        "deviation locus {locus} should be a chart coordinate in [0, 1)"
    );
}

/// The drift statistics (gam#2231 §3) are populated and sane: an exact phase
/// shift keeps the two layer images congruent, so the principal angles between
/// their row spaces are ~0, and the honest-units drift is finite.
#[test]
fn phase_shift_keeps_layer_images_congruent() {
    let n = 128usize;
    let p_x = 4usize;
    let p_2 = 4usize;
    let phi0 = 0.2_f64;
    let sigma = 0.003_f64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut y2 = Array2::<f64>::zeros((n, p_2));
    let mut nx = noise_stream(0x7a13_0001);
    let mut n2 = noise_stream(0x7a13_0002);
    let two_pi = std::f64::consts::TAU;
    for i in 0..n {
        let theta = two_pi * (i as f64 / n as f64);
        let shifted = theta + two_pi * phi0;
        z[[i, 0]] = theta.cos() + sigma * nx();
        z[[i, 1]] = theta.sin() + sigma * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos() + sigma * nx();
        z[[i, 3]] = 0.5 * (2.0 * theta).sin() + sigma * nx();
        y2[[i, 0]] = shifted.cos() + sigma * n2();
        y2[[i, 1]] = shifted.sin() + sigma * n2();
        y2[[i, 2]] = 0.5 * (2.0 * shifted).cos() + sigma * n2();
        y2[[i, 3]] = 0.5 * (2.0 * shifted).sin() + sigma * n2();
    }

    let p_tot = p_x + p_2;
    let (mut term, mut rho) = build_k1(&evaluator, &coords, p_tot);
    term.set_guards_enabled(false);
    let mut blocks = vec![OutputBlock::new("layer2", y2.clone(), 0.0).unwrap()];
    term.run_multiblock_reml_fit(z.view(), &mut blocks, &mut rho, None, controls())
        .unwrap();
    let layout = term.crosscoder_layout().unwrap().clone();
    let report = measure_atom_transport(&term, &layout, 0, 256).unwrap();

    assert!(report.drift.is_finite(), "drift must be finite: {}", report.drift);
    assert_eq!(report.source, CrosscoderLayer::Anchor);
    assert_eq!(report.target, CrosscoderLayer::Block(0));
    // Congruent images (a pure phase shift is a reparameterization of the SAME
    // curve): the principal angles between the two row spaces are ~0.
    assert!(
        !report.principal_angles.is_empty(),
        "a rank-≥1 circle image should yield principal angles"
    );
    let max_angle = report
        .principal_angles
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    assert!(
        max_angle < 0.05,
        "phase-shifted layer images should be congruent (principal angles ~0); \
         max angle = {max_angle} rad"
    );
}
