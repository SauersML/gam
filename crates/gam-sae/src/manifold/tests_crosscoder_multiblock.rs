//! The crosscoder stacked-column layout: [`CrosscoderLayout`] owns the offset
//! arithmetic and the per-block `√λ_ℓ` unscaling, and installing one does not
//! perturb a fit.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::manifold::{
    AssignmentMode, CrosscoderLayout, LatentManifold, OutputBlock, PeriodicHarmonicEvaluator,
    SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho,
    SaeManifoldTerm,
};

const ON: f64 = 6.0;

/// Deterministic xorshift noise in `[-1, 1)` (variance 1/3), so a block carries a
/// known noise scale without an RNG dependency.
fn noise_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 52) as f64) - 1.0
    }
}

/// A cold K=1 circle atom at augmented width `p_tot`, seeded at `coords`.
fn circle_atom(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_tot: usize,
) -> SaeManifoldAtom {
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the fixture coords are finite and lie inside the evaluator chart");
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
    .expect("the fixture atom's phi/jet/decoder/gram shapes agree by construction")
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
    .expect("the fixture logits, coord blocks and manifolds all have length k");
    let term = SaeManifoldTerm::new(vec![atom], assignment)
        .expect("the fixture atoms and assignment agree on the atom count");
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

/// Bit-identical comparison of two f64 matrices (NaN-free by construction here).
fn bit_identical(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The [`CrosscoderLayout`] owns the stacked-column offset arithmetic and the
/// per-block `√λ_ℓ` unscaling: its ranges and total width round-trip, its
/// `√λ_ℓ` matches an [`OutputBlock`] to the bit, and `from_blocks` reconstructs
/// the same layout as the explicit constructor.
#[test]
fn crosscoder_layout_round_trips_offsets_and_unscaling() {
    let p_x = 4usize;
    let dims = vec![3usize, 5, 2];
    let logs = vec![0.1_f64, -0.4, 0.7];
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let layout = CrosscoderLayout::new(p_x, dims.clone(), labels.clone(), logs.clone()).unwrap();

    assert_eq!(layout.anchor_dim(), p_x);
    assert_eq!(layout.num_blocks(), 3);
    assert_eq!(layout.block_dims(), dims.as_slice());
    assert_eq!(layout.labels(), labels.as_slice());
    assert_eq!(layout.total_dim(), p_x + 3 + 5 + 2);
    // Offsets accumulate in stacked order: 4 | [4,7) | [7,12) | [12,14).
    assert_eq!(layout.block_range(0), 4..7);
    assert_eq!(layout.block_range(1), 7..12);
    assert_eq!(layout.block_range(2), 12..14);

    // √λ_ℓ and log λ_ℓ match OutputBlock bit-for-bit (so an unscaled decoder is
    // identical whether carved via the layout or the block).
    for (l, (&dim, &ll)) in dims.iter().zip(logs.iter()).enumerate() {
        let block = OutputBlock::new("x", Array2::<f64>::zeros((2, dim)), ll).unwrap();
        assert_eq!(
            layout.sqrt_lambda(l).to_bits(),
            block.sqrt_lambda().to_bits()
        );
        assert_eq!(layout.log_lambda(l).to_bits(), ll.to_bits());
    }

    // from_blocks reconstructs the identical layout.
    let blocks = vec![
        OutputBlock::new("a", Array2::<f64>::zeros((2, 3)), 0.1).unwrap(),
        OutputBlock::new("b", Array2::<f64>::zeros((2, 5)), -0.4).unwrap(),
        OutputBlock::new("c", Array2::<f64>::zeros((2, 2)), 0.7).unwrap(),
    ];
    assert_eq!(CrosscoderLayout::from_blocks(p_x, &blocks), layout);

    // Validation: mismatched parallel-vector lengths and zero anchor are rejected.
    assert!(CrosscoderLayout::new(p_x, vec![3], vec![], vec![0.1]).is_err());
    assert!(CrosscoderLayout::new(0, vec![], vec![], vec![]).is_err());
}

/// An anchor-only (`L = 1`, no output blocks) crosscoder layout is a pure
/// descriptor: installing it perturbs NOTHING, so the fitted decoders and
/// reconstruction are byte-identical to the plain joint-fit path that has no
/// layout at all. This pins the layout as fit-inert (it is read only by
/// `layer_decoder`, never by the fit).
#[test]
fn empty_layout_is_a_pure_descriptor_byte_identical_to_plain_fit() {
    let n = 72usize;
    let p = 4usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);

    let mut z = Array2::<f64>::zeros((n, p));
    let mut nx = noise_stream(0x5eed_4001);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64 / n as f64);
        z[[i, 0]] = theta.cos() + 0.05 * nx();
        z[[i, 1]] = theta.sin() + 0.05 * nx();
        z[[i, 2]] = 0.5 * (2.0 * theta).cos();
        z[[i, 3]] = 0.05 * nx();
    }

    // Path A — plain joint fit, no layout.
    let (mut term_a, mut rho_a) = build_k1(&evaluator, &coords, p);
    term_a.set_guards_enabled(false);
    term_a
        .run_joint_fit_arrow_schur(z.view(), &mut rho_a, None, 48, 1.0, 1e-6, 1e-6)
        .unwrap();

    // Path B — identical fit, but with an anchor-only crosscoder layout installed.
    let (mut term_b, mut rho_b) = build_k1(&evaluator, &coords, p);
    term_b.set_guards_enabled(false);
    let empty = CrosscoderLayout::new(p, vec![], vec![], vec![]).unwrap();
    assert_eq!(empty.total_dim(), p);
    assert_eq!(empty.num_blocks(), 0);
    term_b.set_crosscoder_layout(empty).unwrap();
    term_b
        .run_joint_fit_arrow_schur(z.view(), &mut rho_b, None, 48, 1.0, 1e-6, 1e-6)
        .unwrap();

    assert!(
        bit_identical(
            term_a.atoms[0].decoder_coefficients(),
            term_b.atoms[0].decoder_coefficients()
        ),
        "an empty crosscoder layout must not perturb the fitted decoders"
    );
    let fitted_a = term_a.try_fitted_for_rho(&rho_a).unwrap();
    let fitted_b = term_b.try_fitted_for_rho(&rho_b).unwrap();
    assert!(
        bit_identical(&fitted_a, &fitted_b),
        "an empty crosscoder layout must not perturb the reconstruction"
    );

    // With no output blocks there is no layer to carve.
    assert!(term_b.layer_decoder(0, 0).is_err());
}
