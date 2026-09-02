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

use crate::manifold::{AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, TwoBlockRemlControls};

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
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the fixture's coordinate block is a valid input for this evaluator");
    let m = phi.ncols();
    SaeManifoldAtom::new_with_provided_function_gram(
        "cc",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .expect("the fixture's basis, decoder and Gram blocks agree in dimension")
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
    .expect("the fixture's logits, coordinate blocks and manifolds agree in length");
    let term = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("the fixture's atoms and assignment describe the same latent blocks");
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

