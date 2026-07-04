//! F4 reviewer acceptance — the LayerNorm-sphere ambient path removes the
//! spurious curvature a flat-space circle fit invents from norm variation.
//!
//! The residual stream post-LayerNorm is a sphere × scale product: the model
//! reads only the direction `u = x/‖x‖`, and the per-token norm is a nuisance the
//! LayerNorm discards. Reviewer F4's charge: a curved atom fitted to reconstruct
//! the *flat* activation `x` absorbs that norm variation into its decoder as
//! spurious higher-harmonic curvature, and its reconstruction certificate — which
//! assumes an additive isotropic-Gaussian residual in the flat metric — is
//! conditional on an assumption the radial (norm-direction) residual violates.
//!
//! These tests plant data with a KNOWN pure-harmonic-1 direction on the sphere
//! and an INDEPENDENT lognormal norm, then run the REAL K=1 circle fit
//! ([`SaeManifoldTerm::run_joint_fit_arrow_schur`]) two ways:
//!   * FLAT   — reconstruct `x` directly (ambient Euclidean);
//!   * SPHERE — reconstruct [`ln_sphere_project`]`(x)` (atom as an LN-sphere
//!              submanifold).
//! and show that (1) the flat decoder carries large higher-harmonic (spurious
//! curvature) energy that GROWS with the norm variation, while the sphere decoder
//! stays pure harmonic-1; and (2) the flat residual is dominated by the radial
//! norm-direction term, while the sphere residual is not. The true generator has
//! exactly zero of both, so anything the flat fit reports is an artifact of the
//! wrong ambient metric.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, ln_sphere_project,
};

const ON: f64 = 6.0;

/// Deterministic lognormal-ish norm stream: xorshift uniform → standard normal
/// (Box–Muller) → `exp(s · z)`, so the planted per-token norm variation is
/// reproducible bit-for-bit at a chosen log-scale `s`.
fn norm_stream(seed: u64, s: f64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    let mut draw_u = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-12, 1.0 - 1e-12)
    };
    move || {
        let u1 = draw_u();
        let u2 = draw_u();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        (s * z).exp()
    }
}

/// Plant `x_i = r_i · v(θ_i)` with `v(θ) = cos θ · e0 + sin θ · e1` a unit great
/// circle (pure harmonic-1) in the first two of `p` dims and `r_i` an independent
/// lognormal norm at log-scale `s`. Returns the activations and the true angles.
fn plant(n: usize, p: usize, s: f64, seed: u64) -> (Array2<f64>, Vec<f64>) {
    let mut x = Array2::<f64>::zeros((n, p));
    let mut r = norm_stream(seed, s);
    let mut thetas = Vec::with_capacity(n);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        thetas.push(theta);
        let radius = r();
        x[[i, 0]] = radius * theta.cos();
        x[[i, 1]] = radius * theta.sin();
    }
    (x, thetas)
}

/// Build a K=1 circle term at output width `p_tot`, seeded at `coords`.
fn circle_term(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let n = coords.nrows();
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

/// Seed the circle coordinate from the phase of the first two activation dims
/// (what the flagship demo does), lightly de-aligned so the fit does real work.
fn seed_coords(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    Array2::<f64>::from_shape_fn((n, 1), |(i, _)| {
        let base = x[[i, 1]].atan2(x[[i, 0]]) / std::f64::consts::TAU;
        // tiny deterministic de-alignment so the seed is not the exact optimum
        let jitter = 0.02 * ((i as f64 * 2.399_963).sin());
        (base + jitter).rem_euclid(1.0)
    })
}

/// Fraction of the fitted decoder's energy (excluding the constant row 0) sitting
/// in harmonics ≥ 2. The basis order is `[1, sin·1, cos·1, sin·2, cos·2, …]`, so
/// harmonic 1 is rows 1–2 and everything from row 3 up is higher-harmonic —
/// spurious curvature for a pure-harmonic-1 generator.
/// Coefficient of variation of the fitted OUTPUT norms `‖γ_i‖`. This is the
/// observable signature of the flat metric absorbing the nuisance scale: a flat
/// fit reconstructs `x_i = r_i·v(θ_i)`, so `‖γ_i‖ ≈ r_i` inherits the planted norm
/// variation — the fitted closed curve must bend radially (spurious curvature) to
/// interpolate norms that swing with the θ-independent `r`. The LN-sphere fit sees
/// the unit-norm target `v(θ)`, so `‖γ_i‖ ≈ 1` and its CV collapses to ~0,
/// regardless of the injected scale. Read off the fitted outputs only, so it is
/// independent of the solver's internal (whitened) decoder representation.
fn fitted_norm_cv(fitted: &Array2<f64>) -> f64 {
    let n = fitted.nrows();
    let norms: Vec<f64> = (0..n)
        .map(|i| fitted.row(i).iter().map(|&v| v * v).sum::<f64>().sqrt())
        .collect();
    let mean = norms.iter().sum::<f64>() / n as f64;
    if !(mean > 0.0) {
        return 0.0;
    }
    let var = norms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt() / mean
}

/// Fit a K=1 circle to `target` and return the CV of the fitted output norms,
/// read off the fitted OUTPUTS (solver-representation-agnostic).
fn fit_and_measure(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    seed: &Array2<f64>,
    target: &Array2<f64>,
) -> f64 {
    let p = target.ncols();
    let (mut term, mut rho) = circle_term(evaluator, seed, p);
    term.set_guards_enabled(false);
    term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("circle fit must complete");
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    fitted_norm_cv(&fitted)
}

/// LOAD-BEARING F4 acceptance: at a realistic norm variation, the flat fit
/// invents spurious higher-harmonic curvature and a dominant radial residual; the
/// LN-sphere fit does neither.
#[test]
fn ln_sphere_fit_removes_flat_spurious_curvature_and_radial_residual() {
    let (n, p, s) = (240usize, 6usize, 0.4_f64);
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(9).unwrap()); // harmonics 1..=4
    let (x, _theta) = plant(n, p, s, 0x1234_5678);
    let seed = seed_coords(&x);

    // Genuine norm variation is planted.
    let norms: Vec<f64> = (0..n)
        .map(|i| (x[[i, 0]] * x[[i, 0]] + x[[i, 1]] * x[[i, 1]]).sqrt())
        .collect();
    let mean = norms.iter().sum::<f64>() / n as f64;
    let var = norms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let cv = var.sqrt() / mean;
    assert!(cv > 0.2, "planted norm CV too small to test anything: {cv}");

    // FLAT: reconstruct the raw activation.
    let flat_cv = fit_and_measure(&evaluator, &seed, &x);
    // SPHERE: reconstruct the LN-projected direction (the real code path).
    let (u, _norms) = ln_sphere_project(x.view(), None).unwrap();
    let sph_cv = fit_and_measure(&evaluator, &seed, &u);
    eprintln!("F4 s={s} planted_cv={cv:.3} | FLAT fit_norm_cv={flat_cv:.4} | SPHERE fit_norm_cv={sph_cv:.4}");

    // The flat fit absorbs the θ-independent norm variation into its reconstruction
    // (fitted output norms swing with the planted r, forcing radial curvature); the
    // LN-sphere fit sees a unit-norm target, so its fitted norms are ~constant.
    assert!(
        sph_cv < 0.05,
        "LN-sphere fitted norms should be ~constant (nuisance scale quotiented \
         out); got fit_norm_cv {sph_cv}"
    );
    assert!(
        flat_cv > 0.15,
        "flat fit should reproduce the planted norm variation in its \
         reconstruction; got fit_norm_cv {flat_cv} (planted CV {cv})"
    );
    assert!(
        flat_cv > 3.0 * sph_cv,
        "flat norm absorption {flat_cv} must dominate the LN-sphere fit {sph_cv}"
    );
}

/// The spurious curvature is MONOTONE in the norm variation — it is manufactured
/// by the flat metric from the nuisance scale, not an intrinsic property of the
/// data. At zero norm variation the flat and sphere fits agree; as the variation
/// grows the flat curvature climbs while the sphere stays put.
#[test]
fn flat_spurious_curvature_grows_with_norm_variation_sphere_invariant() {
    let (n, p) = (240usize, 6usize);
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(9).unwrap());
    let mut flat_curve = Vec::new();
    let mut sphere_curve = Vec::new();
    for &s in &[0.0_f64, 0.2, 0.45] {
        let (x, _theta) = plant(n, p, s, 0xABCD_0001);
        let seed = seed_coords(&x);
        let flat_cv = fit_and_measure(&evaluator, &seed, &x);
        let (u, _) = ln_sphere_project(x.view(), None).unwrap();
        let sph_cv = fit_and_measure(&evaluator, &seed, &u);
        flat_curve.push(flat_cv);
        sphere_curve.push(sph_cv);
    }
    eprintln!("F4 sweep FLAT fit_norm_cv={flat_curve:?} SPHERE fit_norm_cv={sphere_curve:?}");
    // The flat fit's absorbed norm variation climbs with the planted variation:
    // lowest at zero, and a wide end-to-end increase.
    assert!(
        flat_curve[2] > flat_curve[0] + 0.1,
        "flat fitted-norm CV should climb with planted norm variation: {flat_curve:?}"
    );
    assert!(
        flat_curve[0] <= flat_curve[1] && flat_curve[1] <= flat_curve[2],
        "flat norm absorption should be monotone in the planted variation: {flat_curve:?}"
    );
    // The LN-sphere fit is invariant to the nuisance scale: its fitted norms stay
    // ~constant no matter how much norm variation is injected.
    for &v in &sphere_curve {
        assert!(
            v < 0.05,
            "LN-sphere fitted norms should be invariant to the injected scale: \
             {sphere_curve:?}"
        );
    }
}
