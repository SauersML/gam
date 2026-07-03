//! Rung-2 two-block joint fit: behavior as a jointly-fitted output block that
//! shares the latent coordinate `t` and the gate `a` with the activation block.
//!
//! These tests exercise the REAL arrow-Schur joint fit on the AUGMENTED output
//! `[Z | √λ_y · Y]` built by [`BehaviorBlock`]. No new solver path is needed:
//! the augmented output shares `t` and `a` by construction, so the ordinary
//! joint fit reconstructs both blocks. The tests assert (1) the plumbing round
//! trips — the augmented target fits and `split_decoder` recovers the true
//! `[B_k | C_k]`; (2) both blocks are well reconstructed under one shared
//! coordinate; and (3) selection-for-mattering: an activation pattern whose
//! behavior does not vary earns a ≈ 0 behavior decoder while keeping its
//! activation decoder.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    reconstruction_explained_variance,
};

const ON: f64 = 6.0;

/// Softmax of a logit vector (numerically stable) — the planted behavior law.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Explained variance of a column block `[c0, c1)` of a fitted augmented output
/// against the corresponding block of the (unscaled) target.
fn block_ev(target: &Array2<f64>, fitted: &Array2<f64>, c0: usize, c1: usize) -> f64 {
    let t = target.slice(ndarray::s![.., c0..c1]).to_owned();
    let f = fitted.slice(ndarray::s![.., c0..c1]).to_owned();
    reconstruction_explained_variance(t.view(), f.view())
        .unwrap_or_else(|| panic!("EV undefined for block [{c0},{c1})"))
}

/// Build a K=1 periodic (circle) atom at the AUGMENTED output width `p_tot`,
/// with cold (zero) decoders. Returns the atom and the shared coordinate block.
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

/// Assemble a K=1 softmax term (single always-on atom) at augmented width.
fn build_k1(atom: SaeManifoldAtom, coord_block: Array2<f64>) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = coord_block.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), ON);
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

/// Both blocks — activation AND behavior — are driven by ONE shared circle
/// coordinate, so a single-atom two-block fit must reconstruct BOTH, and
/// `split_decoder` must recover a behavior decoder that decodes back to the
/// planted distributions.
#[test]
fn two_block_joint_fit_reconstructs_activation_and_behavior() {
    let n = 60usize;
    let p_x = 4usize;
    let vocab = 4usize; // behavior tangent dim p_y = 3
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    // Planted activation: two channels are a clean cos/sin of the circle angle,
    // the rest zero — a genuine curved circle image in activation space.
    let mut z = Array2::<f64>::zeros((n, p_x));
    // Planted behavior distributions: a softmax whose logits rotate with θ, so
    // the next-token law moves smoothly along the SAME coordinate.
    let mut probs = Array2::<f64>::zeros((n, vocab));
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        z[[i, 2]] = (2.0 * theta).cos();
        let law = softmax(&[
            1.5 * theta.cos(),
            1.5 * theta.sin(),
            0.3 * (2.0 * theta).cos(),
            0.0,
        ]);
        for j in 0..vocab {
            probs[[i, j]] = law[j];
        }
    }

    let block = BehaviorBlock::fit(probs.view(), p_x, -4.0).unwrap();
    let p_y = block.behavior_dim();
    assert_eq!(p_y, vocab - 1);
    let p_tot = p_x + p_y;

    let augmented = block.augmented_target(z.view()).unwrap();
    assert_eq!(augmented.dim(), (n, p_tot));

    let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term, mut rho) = build_k1(atom, cb);
    term.set_behavior_block(block.clone()).unwrap();
    assert_eq!(term.activation_output_dim(), p_x);
    assert_eq!(term.behavior_output_range(), Some(p_x..p_tot));

    term.set_guards_enabled(false);
    term.run_joint_fit_arrow_schur(augmented.view(), &mut rho, None, 48, 1.0, 1e-6, 1e-6)
        .expect("two-block joint fit must complete");

    let fitted = term.try_fitted_for_rho(&rho).unwrap();
    // Compare each block on the SAME (scaled) augmented target the fit saw.
    let ev_act = block_ev(&augmented, &fitted, 0, p_x);
    let ev_beh = block_ev(&augmented, &fitted, p_x, p_tot);
    assert!(ev_act > 0.9, "activation block EV too low: {ev_act}");
    assert!(ev_beh > 0.9, "behavior block EV too low: {ev_beh}");

    // split_decoder recovers a NON-trivial behavior decoder, and decoding the
    // fitted behavior reconstruction returns to the planted distributions.
    let (b_k, c_k) = block
        .split_decoder(term.atoms[0].decoder_coefficients.view())
        .unwrap();
    assert_eq!(b_k.dim().1, p_x);
    assert_eq!(c_k.dim().1, p_y);
    let c_norm = c_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        c_norm > 0.1,
        "behavior decoder collapsed to ~0 despite real behavior: {c_norm}"
    );

    // Decode the fitted behavior tangent at a few rows and compare KL to planted.
    let mut worst_kl = 0.0_f64;
    for &i in &[0usize, n / 4, n / 2, 3 * n / 4] {
        // Fitted behavior tangent in nats units = (augmented fitted behavior cols)/√λ_y.
        let inv = 1.0 / block.sqrt_lambda_y();
        let y_hat = Array1::from_shape_fn(p_y, |j| fitted[[i, p_x + j]] * inv);
        let p_hat = block.embedding.decode(y_hat.view()).unwrap();
        let kl =
            crate::manifold::SphereTangentEmbedding::exact_kl(probs.row(i), p_hat.view()).unwrap();
        worst_kl = worst_kl.max(kl);
    }
    assert!(
        worst_kl < 0.02,
        "decoded behavior diverges from planted (worst KL {worst_kl} nats)"
    );
}

/// Selection-for-mattering: when the behavior does NOT vary across rows (the
/// activation pattern has no behavioral correlate), the behavior target is
/// identically zero, so the fitted behavior decoder is ≈ 0 — the behavior block
/// contributes nothing — while the activation block is still reconstructed.
#[test]
fn constant_behavior_yields_zero_behavior_decoder() {
    let n = 48usize;
    let p_x = 3usize;
    let vocab = 4usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p_x));
    let mut probs = Array2::<f64>::zeros((n, vocab));
    let flat = softmax(&[0.2, 0.1, -0.1, 0.0]); // SAME law on every row
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos();
        z[[i, 1]] = theta.sin();
        for j in 0..vocab {
            probs[[i, j]] = flat[j];
        }
    }

    let block = BehaviorBlock::fit(probs.view(), p_x, 0.0).unwrap();
    // Constant behavior ⇒ zero tangent target by construction.
    assert!(block.target.iter().all(|v| v.abs() < 1e-10));
    let p_y = block.behavior_dim();
    let p_tot = p_x + p_y;

    let augmented = block.augmented_target(z.view()).unwrap();
    let (atom, cb) = augmented_circle_atom(&evaluator, &coords, p_tot);
    let (mut term, mut rho) = build_k1(atom, cb);
    term.set_behavior_block(block.clone()).unwrap();
    term.set_guards_enabled(false);
    term.run_joint_fit_arrow_schur(augmented.view(), &mut rho, None, 48, 1.0, 1e-6, 1e-6)
        .expect("fit must complete");

    let (b_k, c_k) = block
        .split_decoder(term.atoms[0].decoder_coefficients.view())
        .unwrap();
    let b_norm = b_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    let c_norm = c_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        b_norm > 0.1,
        "activation decoder should still fit the circle: {b_norm}"
    );
    assert!(
        c_norm < 1e-6,
        "constant behavior must earn a ~0 behavior decoder; got {c_norm}"
    );
}
