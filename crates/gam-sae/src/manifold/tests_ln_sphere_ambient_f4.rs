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
fn higher_harmonic_fraction(decoder: &Array2<f64>) -> f64 {
    let m = decoder.nrows();
    let row_energy = |row: usize| -> f64 { decoder.row(row).iter().map(|&v| v * v).sum::<f64>() };
    let h1 = row_energy(1) + row_energy(2);
    let mut hi = 0.0_f64;
    for row in 3..m {
        hi += row_energy(row);
    }
    hi / (h1 + hi + 1e-30)
}

/// Fit a K=1 circle to `target` and return `(higher_harmonic_fraction, radial
/// residual RMS, total residual RMS)`. The radial residual is the component of
/// `target − fitted` along the fitted curve's radial direction `γ/‖γ‖` — the
/// norm-direction term the flat metric leaves structured.
fn fit_and_measure(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    seed: &Array2<f64>,
    target: &Array2<f64>,
) -> (f64, f64, f64) {
    let p = target.ncols();
    let (mut term, mut rho) = circle_term(evaluator, seed, p);
    term.set_guards_enabled(false);
    term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("circle fit must complete");
    let hh = higher_harmonic_fraction(&term.atoms[0].decoder_coefficients);
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    let n = target.nrows();
    let mut rad_sq = 0.0_f64;
    let mut tot_sq = 0.0_f64;
    for i in 0..n {
        let g = fitted.row(i);
        let gnorm = g.iter().map(|&v| v * v).sum::<f64>().sqrt() + 1e-30;
        let mut along = 0.0_f64;
        for j in 0..p {
            let resid = target[[i, j]] - fitted[[i, j]];
            along += resid * g[j] / gnorm;
            tot_sq += resid * resid;
        }
        rad_sq += along * along;
    }
    (hh, (rad_sq / n as f64).sqrt(), (tot_sq / n as f64).sqrt())
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
    let (flat_hh, flat_rad, flat_tot) = fit_and_measure(&evaluator, &seed, &x);
    // SPHERE: reconstruct the LN-projected direction (the real code path).
    let (u, _norms) = ln_sphere_project(x.view(), None).unwrap();
    let (sph_hh, sph_rad, sph_tot) = fit_and_measure(&evaluator, &seed, &u);

    // (1) Spurious curvature: the flat decoder bends into higher harmonics; the
    // sphere decoder stays pure harmonic-1.
    assert!(
        flat_hh > 0.05,
        "flat fit should absorb norm variation as spurious higher-harmonic \
         curvature; got hh_frac {flat_hh} (norm CV {cv})"
    );
    assert!(
        sph_hh < 0.02,
        "LN-sphere fit should stay pure harmonic-1; got hh_frac {sph_hh}"
    );
    assert!(
        flat_hh > 3.0 * sph_hh,
        "flat spurious curvature {flat_hh} must dominate sphere {sph_hh}"
    );

    // (2) Certificate assumption: the flat residual is dominated by the radial
    // (norm-direction) term; the sphere residual is not (its total residual is
    // near zero and carries no free radial variance).
    let flat_radial_share = flat_rad / (flat_tot + 1e-30);
    assert!(
        flat_radial_share > 0.5,
        "flat residual should be radial-dominated (certificate-violating); share \
         {flat_radial_share} (rad {flat_rad}, tot {flat_tot})"
    );
    assert!(
        sph_tot < 0.2 * flat_tot,
        "LN-sphere fit should reconstruct the direction far more tightly than the \
         flat fit fits the raw activation: sphere tot {sph_tot} vs flat tot {flat_tot}"
    );
    assert!(
        sph_rad < flat_rad,
        "sphere radial residual {sph_rad} should be below flat {flat_rad}"
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
        let (x, _t) = plant(n, p, s, 0xABCD_0001);
        let seed = seed_coords(&x);
        let (flat_hh, _, _) = fit_and_measure(&evaluator, &seed, &x);
        let (u, _) = ln_sphere_project(x.view(), None).unwrap();
        let (sph_hh, _, _) = fit_and_measure(&evaluator, &seed, &u);
        flat_curve.push(flat_hh);
        sphere_curve.push(sph_hh);
    }
    // Flat spurious curvature is monotone increasing in norm variation.
    assert!(
        flat_curve[0] < flat_curve[1] && flat_curve[1] < flat_curve[2],
        "flat higher-harmonic energy should grow with norm variation: {flat_curve:?}"
    );
    // And it grows by a wide margin end-to-end.
    assert!(
        flat_curve[2] > flat_curve[0] + 0.1,
        "flat curvature barely moved with norm variation: {flat_curve:?}"
    );
    // The sphere fit is invariant to the nuisance scale (stays small throughout).
    for &v in &sphere_curve {
        assert!(
            v < 0.02,
            "LN-sphere higher-harmonic energy should be invariant to norm \
             variation: {sphere_curve:?}"
        );
    }
}
