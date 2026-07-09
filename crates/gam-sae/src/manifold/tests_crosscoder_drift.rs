//! Cross-layer drift statistic tests (gam#2231 Inc E).
//!
//! The drift geometry is measured on the atoms' honest-units per-layer decoders, so
//! these tests construct the decoders DIRECTLY with a planted rotation between
//! layers (no dependence on a joint-fit converging to a particular basin). Each
//! layer's decoder is `M × p` in a shared ambient `ℝ^p`; a planted rotation by
//! angle `α` in the `(e₁, e₂)` plane makes the layer-image row space rotate by
//! exactly `α`, so the principal angle and the Frobenius drift are known in closed
//! form and the assertions are exact.

use ndarray::{Array2, s};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, CrosscoderLayer, CrosscoderLayout, LatentManifold, PeriodicHarmonicEvaluator,
    SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldTerm,
    measure_crosscoder_drift,
};

const P: usize = 4; // shared residual-stream ambient width per layer
const H: usize = 2; // harmonic order ⇒ M = 2H + 1 = 5 basis rows

/// A single layer's `M × P` decoder carrying the planted geometry in its first two
/// basis rows: row 0 = `e₀`, row 1 = the in-plane direction `cos α·e₁ + sin α·e₂`.
/// A rotation angle `α` rotates the layer-image row space by exactly `α`.
fn layer_block(m: usize, alpha: f64) -> Array2<f64> {
    let mut b = Array2::<f64>::zeros((m, P));
    b[[0, 0]] = 1.0;
    b[[1, 1]] = alpha.cos();
    b[[1, 2]] = alpha.sin();
    b
}

/// Assemble a `p_tot = P·(1 + n_blocks)` augmented decoder from per-layer rotation
/// angles `[anchor, block0, …]`, optionally scaling block `ℓ`'s columns by
/// `√λ_ℓ = exp(½·log λ_ℓ)` (what `stack_augmented_target` bakes into the stored
/// decoder — the honest drift must be invariant to it).
fn augmented_decoder(m: usize, angles: &[f64], block_log_lambda: &[f64]) -> Array2<f64> {
    let l = angles.len();
    let mut d = Array2::<f64>::zeros((m, P * l));
    for (layer, &alpha) in angles.iter().enumerate() {
        let mut block = layer_block(m, alpha);
        if layer >= 1 {
            let sqrt_lambda = (0.5 * block_log_lambda[layer - 1]).exp();
            block.mapv_inplace(|v| v * sqrt_lambda);
        }
        d.slice_mut(s![.., layer * P..(layer + 1) * P]).assign(&block);
    }
    d
}

/// Build a K-atom crosscoder term whose atom `k` has the augmented decoder built
/// from `angles_per_atom[k]`, and install the matching layout. `block_log_lambda`
/// scales the stored decoders (parallel to the layout's stored `log λ_ℓ`), so a
/// non-zero choice exercises the honest-units unscaling.
fn build_term(
    angles_per_atom: &[Vec<f64>],
    block_dims: Vec<usize>,
    block_log_lambda: Vec<f64>,
) -> (SaeManifoldTerm, CrosscoderLayout) {
    let n = 8usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(H).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();

    let p_tot = P * angles_per_atom[0].len();
    let atoms: Vec<SaeManifoldAtom> = angles_per_atom
        .iter()
        .map(|angles| {
            let decoder = augmented_decoder(m, angles, &block_log_lambda);
            assert_eq!(decoder.ncols(), p_tot);
            SaeManifoldAtom::new(
                "cc",
                SaeAtomBasisKind::Periodic,
                1,
                phi.clone(),
                jet.clone(),
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
        })
        .collect();
    let k = atoms.len();

    let logits = Array2::<f64>::from_elem((n, k), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone(); k],
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();

    let labels: Vec<String> = (0..block_dims.len()).map(|l| format!("layer{}", l + 1)).collect();
    let layout = CrosscoderLayout::new(P, block_dims, labels, block_log_lambda).unwrap();
    (term, layout)
}

/// Closed-form principal angle for a planted rotation `α` in the `(e₁, e₂)` plane:
/// the two row spaces share `e₀` (angle 0) and differ by `α` in the plane, so the
/// max principal angle is `|α|`.
fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() < tol,
        "{what}: got {got}, want {want} (|Δ| = {})",
        (got - want).abs()
    );
}

#[test]
fn drift_recovers_planted_layer_rotations() {
    // 3-layer chain: anchor at 0, block0 at 0.3, block1 at 0.7 (all from e₁).
    let angles = vec![0.0, 0.3, 0.7];
    let (term, layout) = build_term(&[angles], vec![P, P], vec![0.0, 0.0]);
    let report = measure_crosscoder_drift(&term, &layout).unwrap();

    assert_eq!(report.num_atoms, 1);
    assert_eq!(report.num_steps(), 2);
    assert_eq!(report.layer_chain.len(), 3);
    assert_eq!(report.layer_chain[0], CrosscoderLayer::Anchor);
    assert_eq!(report.layer_chain[1], CrosscoderLayer::Block(0));
    assert_eq!(report.layer_chain[2], CrosscoderLayer::Block(1));

    // Step 0: anchor(0) → block0(0.3), a 0.3 rad rotation of the second direction.
    let step0 = &report.steps[0];
    assert_close(step0.max_principal_angle(), 0.3, 1e-9, "step0 max principal angle");
    assert_close(step0.drift, (1.0 - 0.3_f64.cos()).sqrt(), 1e-9, "step0 drift");

    // Step 1: block0(0.3) → block1(0.7), a 0.4 rad rotation.
    let step1 = &report.steps[1];
    assert_close(step1.max_principal_angle(), 0.4, 1e-9, "step1 max principal angle");
    assert_close(step1.drift, (1.0 - 0.4_f64.cos()).sqrt(), 1e-9, "step1 drift");

    let profile = report.atom_drift_profile(0);
    assert_eq!(profile.len(), 2);
    assert_close(profile[0], step0.drift, 1e-12, "profile[0] == step0.drift");
    assert_close(profile[1], step1.drift, 1e-12, "profile[1] == step1.drift");
    assert_close(
        report.atom_total_drift(0),
        step0.drift + step1.drift,
        1e-12,
        "atom total drift == sum of steps",
    );
    // Only one atom, so it is both the most-drifting and the most-stable.
    assert_eq!(report.most_drifting_atom(), Some(0));
    assert_eq!(report.most_stable_atom(), Some(0));
}

#[test]
fn honest_drift_is_invariant_to_block_lambda() {
    // Identical planted geometry, but the stored decoders are scaled by √λ_ℓ for
    // non-zero block weights. The honest-units drift must be byte-for-byte the same
    // as the λ = 1 (log λ = 0) case — λ_ℓ cancels in the honest decoder.
    let angles = vec![0.0, 0.3, 0.7];
    let (term_unit, layout_unit) = build_term(&[angles.clone()], vec![P, P], vec![0.0, 0.0]);
    let (term_scaled, layout_scaled) =
        build_term(&[angles], vec![P, P], vec![1.3, -0.8]);

    let a = measure_crosscoder_drift(&term_unit, &layout_unit).unwrap();
    let b = measure_crosscoder_drift(&term_scaled, &layout_scaled).unwrap();

    for (sa, sb) in a.steps.iter().zip(b.steps.iter()) {
        assert_close(sb.drift, sa.drift, 1e-12, "λ-scaled drift matches unit-λ drift");
        assert_close(
            sb.max_principal_angle(),
            sa.max_principal_angle(),
            1e-12,
            "λ-scaled principal angle matches unit-λ",
        );
    }
}

#[test]
fn ranking_picks_most_and_least_drifting_atoms() {
    // Atom 0 rotates layer-to-layer; atom 1 is layer-STABLE (identical blocks).
    let moving = vec![0.0, 0.4, 0.9];
    let stable = vec![0.2, 0.2, 0.2];
    let (term, layout) = build_term(&[moving, stable], vec![P, P], vec![0.0, 0.0]);
    let report = measure_crosscoder_drift(&term, &layout).unwrap();

    assert_eq!(report.num_atoms, 2);
    // Atom 1's blocks are identical across layers ⇒ zero drift, zero rotation.
    assert_close(report.atom_total_drift(1), 0.0, 1e-12, "stable atom total drift");
    for s in report.steps.iter().filter(|s| s.atom == 1) {
        assert_close(s.max_principal_angle(), 0.0, 1e-9, "stable atom principal angle");
    }
    assert!(report.atom_total_drift(0) > 0.0, "moving atom must drift");

    assert_eq!(report.most_drifting_atom(), Some(0));
    assert_eq!(report.most_stable_atom(), Some(1));
    assert!(report.mean_drift().is_finite());
}

#[test]
fn rejects_layout_that_does_not_describe_the_term() {
    let angles = vec![0.0, 0.3];
    let (term, _good) = build_term(&[angles], vec![P], vec![0.0]);
    // A layout whose total width disagrees with the term's augmented width.
    let bad = CrosscoderLayout::new(P, vec![P + 1], vec!["x".into()], vec![0.0]).unwrap();
    let err = measure_crosscoder_drift(&term, &bad).unwrap_err();
    assert!(err.contains("!= term output_dim"), "unexpected error: {err}");
}

#[test]
fn rejects_layers_of_differing_ambient_width() {
    // A term whose blocks are NOT all P wide: anchor P, block0 P, block1 = P-1.
    // Build the augmented decoder by hand at the ragged width and install a ragged
    // layout — cross-layer drift needs one shared ambient, so this must be refused.
    let n = 8usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(H).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let p_tot = P + P + (P - 1);
    let decoder = Array2::<f64>::from_shape_fn((m, p_tot), |(r, c)| {
        if r == 0 && c == 0 { 1.0 } else { 0.0 }
    });
    let atom = SaeManifoldAtom::new(
        "cc",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .unwrap();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    let ragged = CrosscoderLayout::new(
        P,
        vec![P, P - 1],
        vec!["layer1".into(), "layer2".into()],
        vec![0.0, 0.0],
    )
    .unwrap();
    let err = measure_crosscoder_drift(&term, &ragged).unwrap_err();
    assert!(err.contains("layer widths differ"), "unexpected error: {err}");
}
