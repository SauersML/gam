//! #2015 — the per-atom representation–behavior isometry defect, read off a REAL
//! two-block joint fit.
//!
//! Both cases share ONE activation image: a clean unit circle, constant speed in
//! the shared latent `t`. What differs is HOW the behavior traverses its own
//! circle as a function of the SAME `t`:
//!
//!  * **isometric** — behavior winds at *constant* angular speed `ψ(t) = 2π t`,
//!    so its induced speed `s_y` is (nearly) constant and the ratio `r = s_x/s_y`
//!    is constant: a scaled isometry, low reported defect.
//!  * **broken**    — behavior winds at *uneven* speed `ψ(t) = 2π t + 0.8·sin 2π t`
//!    (`ψ'` sweeps from `0.2·2π` to `1.8·2π`), so `s_y` varies strongly while
//!    `s_x` stays constant: the correspondence bends along the atom, high defect.
//!
//! The activation-only geometry is IDENTICAL in the two cases (same circle), so
//! no activation-side statistic can tell them apart — the defect is a genuinely
//! cross-block quantity that only the two-block fit exposes. The test asserts the
//! reported defect separates the two, and that the isometric defect is small.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    atom_behavior_isometry, reconstruction_explained_variance,
};

/// Numerically stable softmax — the planted behavior law.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// K=1 periodic (circle) atom at the augmented output width, cold decoders.
fn augmented_circle_atom(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> (SaeManifoldAtom, Array2<f64>) {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new(
        "b0",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p_tot)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    (atom, coords.clone())
}

/// K=1 softmax term (single always-on atom) at augmented width.
fn build_k1(atom: SaeManifoldAtom, coord_block: Array2<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = coord_block.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coord_block],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

fn block_ev(target: &Array2<f64>, fitted: &Array2<f64>, c0: usize, c1: usize) -> f64 {
    let t = target.slice(ndarray::s![.., c0..c1]).to_owned();
    let f = fitted.slice(ndarray::s![.., c0..c1]).to_owned();
    reconstruction_explained_variance(t.view(), f.view()).unwrap_or(0.0)
}

/// Fit a two-block circle whose behavior traverses at angular speed `ψ(t)`, and
/// return the fitted atom's reported representation–behavior isometry defect.
fn fitted_defect(uneven: bool) -> (f64, f64, f64) {
    let n = 96usize;
    let p_x = 4usize;
    let vocab = 4usize; // p_y = 3
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(6).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    for i in 0..n {
        let t = i as f64 / n as f64;
        let theta = std::f64::consts::TAU * t;
        // Activation: a clean unit circle — constant speed in t, IDENTICAL across
        // the two cases.
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = 0.4 * (2.0 * theta).cos();
        // Behavior winds through its own circle at angular position ψ(t).
        let psi = if uneven {
            theta + 0.8 * theta.sin()
        } else {
            theta
        };
        let law = softmax(&[0.7 * psi.cos(), 0.7 * psi.sin(), 0.2, 0.0]);
        for j in 0..vocab {
            probs[[i, j]] = law[j];
        }
    }

    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0).unwrap();
    let p_tot = p_x + block.behavior_dim();
    let augmented = block.augmented_target(z.view()).unwrap();

    let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term, mut rho) = build_k1(atom, cb);
    term.set_behavior_block(block).unwrap();
    term.set_guards_enabled(false);
    term.run_joint_fit_arrow_schur(augmented.view(), &mut rho, None, 64, 1.0, 1e-6, 1e-6)
        .expect("two-block fit must complete");

    // Precondition: both blocks are actually reconstructed, so the induced speeds
    // read off the fitted decoders reflect the planted geometry (not fit noise).
    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    let ev_act = block_ev(&augmented, &fitted, 0, p_x);
    let ev_beh = block_ev(&augmented, &fitted, p_x, p_tot);
    assert!(ev_act > 0.85, "activation EV too low ({ev_act}) to trust the speeds");
    assert!(ev_beh > 0.85, "behavior EV too low ({ev_beh}) to trust the speeds");

    let cert = atom_behavior_isometry(&term, 0)
        .expect("isometry certificate must compute")
        .expect("a d=1 two-block atom must yield a certificate");
    assert!(cert.behavior_engaged, "behavior must be engaged (it moves)");
    assert!(cert.scale.is_finite() && cert.scale > 0.0, "scale {}", cert.scale);
    assert!(
        cert.nats_per_unit_t.is_finite() && cert.nats_per_unit_t > 0.0,
        "nats/unit t {}",
        cert.nats_per_unit_t
    );
    (cert.defect_cv, cert.scale, cert.nats_per_unit_t)
}

/// The reported isometry defect separates a scaled-isometric two-block atom (low
/// defect) from one whose behavior winds unevenly relative to its activation
/// (high defect) — even though the two share an IDENTICAL activation image, so no
/// activation-only statistic could tell them apart.
#[test]
fn isometry_defect_separates_isometric_from_broken() {
    let (defect_iso, _scale_iso, _nats_iso) = fitted_defect(false);
    let (defect_broken, _scale_broken, _nats_broken) = fitted_defect(true);

    // The scaled-isometric atom reports a small defect: the two induced metrics
    // are proportional along the shared coordinate.
    assert!(
        defect_iso < 0.15,
        "scaled-isometric atom should report a low defect, got {defect_iso}"
    );
    // The broken atom's behavior winds ~9× faster/slower across the circle, so its
    // ratio-to-activation is far from constant.
    assert!(
        defect_broken > 0.30,
        "broken-isometry atom should report a high defect, got {defect_broken}"
    );
    // And the separation is wide — the statistic is decisive, not marginal.
    assert!(
        defect_broken > 2.0 * defect_iso,
        "defect must separate the two cases: isometric {defect_iso} vs broken {defect_broken}"
    );
}
