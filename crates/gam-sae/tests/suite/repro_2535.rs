//! #2535 K>=2 fails-first witness — NOT FOR LANDING until it is shown to FAIL.
//!
//! This is an integration test deliberately: it exercises only public gam-sae
//! APIs and must not require linking the crate's monolithic unit-test binary.

use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, OutputBlock, SaeAssignment, SaeAtomBasisKind,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, stack_augmented_target,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const ON: f64 = 6.0;

fn softmax(logits: &[f64]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

fn noise_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 52) as f64) - 1.0
    }
}

fn build_k2(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    output_dim: usize,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let (basis_values, basis_jacobian) = evaluator.evaluate(coords.view()).unwrap();
    let basis_width = basis_values.ncols();
    let atoms: Vec<SaeManifoldAtom> = (0..2)
        .map(|index| {
            SaeManifoldAtom::new_with_provided_function_gram(
                if index == 0 { "cc0" } else { "cc1" },
                SaeAtomBasisKind::Periodic,
                1,
                basis_values.clone(),
                basis_jacobian.clone(),
                Array2::<f64>::zeros((basis_width, output_dim)),
                Array2::<f64>::eye(basis_width),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone())
        })
        .collect();
    // Softmax over TWO atoms: every row carries mass on both, so the sparse G
    // has a live off-diagonal (i,j) block — the co-occurrence the racy
    // `arrow_sae_sparse_g_matvec` atomicAdd needs to collide on.
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_shape_fn((coords.nrows(), 2), |(_, atom)| {
            if atom == 0 { ON } else { ON * 0.5 }
        }),
        vec![coords.clone(), coords.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)]);
    (term, rho)
}

#[test]
fn fresh_arrow_schur_joint_fits_are_bit_reproducible_at_k2_2535() {
    // Shape chosen from the #2512 offload gate so the device PCG can engage:
    // k = (M1+M2)*p = 80 clears DEVICE_LOOP_MIN_P, d = row_block_dim and
    // cg_iters = 200, so the solve's `cg_iters·n·(4·d·k + d²)` arithmetic clears
    // the most permissive calibrated dense launch floor and the device's
    // calibrated policy decides engagement.
    let n = 400usize;
    let p_x = 4usize;
    let vocabulary = 5usize;
    let basis_width = 5usize;
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);

    let mut activations = Array2::<f64>::zeros((n, p_x));
    let mut probabilities = Array2::<f64>::zeros((n, vocabulary));
    let mut noise = noise_stream(0x5eed_2535);
    for row in 0..n {
        let theta = std::f64::consts::TAU * (row as f64 / n as f64);
        activations[[row, 0]] = theta.cos() + 0.05 * noise();
        activations[[row, 1]] = theta.sin() + 0.05 * noise();
        activations[[row, 2]] = 0.5 * (2.0 * theta).cos();
        activations[[row, 3]] = 0.05 * noise();
        let law = softmax(&[
            1.2 * theta.cos(),
            1.2 * theta.sin(),
            0.4 * (2.0 * theta).sin(),
            0.2,
            0.0,
        ]);
        for column in 0..vocabulary {
            probabilities[[row, column]] = law[column];
        }
    }

    let behavior = BehaviorBlock::fit(probabilities.view(), p_x, 0.0).unwrap();
    let blocks = vec![OutputBlock::new("behavior", behavior.target, 0.0).unwrap()];
    let target = stack_augmented_target(activations.view(), &blocks).unwrap();
    let output_dim = target.ncols();

    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(basis_width).unwrap());
    let mut fits: Vec<(SaeManifoldTerm, SaeManifoldRho)> = (0..4)
        .map(|_| build_k2(&evaluator, &coords, output_dim))
        .collect();
    for (term, rho) in &mut fits {
        term.set_guards_enabled(false);
        term.run_joint_fit_arrow_schur(target.view(), rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .unwrap();
    }

    for atom_index in 0..2 {
        let reference = fits[0].0.atoms[atom_index].decoder_coefficients();
        let reference_norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            reference_norm > 0.0,
            "#2535 atom {atom_index} must carry a fitted decoder, got norm {reference_norm:.6e}"
        );
        for (repetition, (term, _)) in fits.iter().enumerate().skip(1) {
            let decoder = term.atoms[atom_index].decoder_coefficients();
            let differing = reference
                .iter()
                .zip(decoder.iter())
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
            assert_eq!(
                differing,
                0,
                "#2535 atom {atom_index}: fresh one-step fit {repetition} differs from fit 0 in \
                 {differing}/{} decoder coefficients",
                decoder.len()
            );
        }
    }
}
