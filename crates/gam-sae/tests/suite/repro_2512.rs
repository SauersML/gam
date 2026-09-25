//! Exact fresh-allocation reproducibility witness for #2512.
//!
//! This is an integration test deliberately: it exercises only public gam-sae
//! APIs and must not require linking the crate's monolithic unit-test binary.

use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisEvaluator};
use gam_sae::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, SaeAssignment, SaeAtomBasisKind,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Axis, concatenate};
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

fn build_k1(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    output_dim: usize,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let (basis_values, basis_jacobian) = evaluator.evaluate(coords.view()).unwrap();
    let basis_width = basis_values.ncols();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "cc",
        SaeAtomBasisKind::Periodic,
        1,
        basis_values,
        basis_jacobian,
        Array2::<f64>::zeros((basis_width, output_dim)),
        Array2::<f64>::eye(basis_width),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((coords.nrows(), 1), ON),
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

#[test]
fn fresh_arrow_schur_joint_fits_are_bit_reproducible_above_61_rows_2512() {
    let n = 62usize;
    let p_x = 4usize;
    let vocabulary = 5usize;
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);

    let mut activations = Array2::<f64>::zeros((n, p_x));
    let mut probabilities = Array2::<f64>::zeros((n, vocabulary));
    let mut noise = noise_stream(0x5eed_1001);
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
    // The augmented target `[Z | √λ_y·Y]` at log λ_y = 0, where √λ_y = exp(0) = 1.
    let target = concatenate(Axis(1), &[activations.view(), behavior.target.view()]).unwrap();
    let output_dim = target.ncols();
    assert_eq!(output_dim, 8, "#2512 fixture requires p_x + p_y = 8");

    // The report names M=11 but also says 40 coefficients at p=8, which
    // implies M=5. Preserve both literal interpretations until provenance
    // resolves that contradiction; either shape drifting is a real regression.
    for basis_width in [5usize, 11usize] {
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(basis_width).unwrap());
        let mut fits: Vec<(SaeManifoldTerm, SaeManifoldRho)> = (0..4)
            .map(|_| build_k1(&evaluator, &coords, output_dim))
            .collect();
        for (term, _) in &fits {
            assert_eq!(
                term.atoms[0].basis_values.ncols(),
                basis_width,
                "#2512 fixture requires M={basis_width} periodic basis columns"
            );
            assert_eq!(
                term.atoms[0].decoder_coefficients().dim(),
                (basis_width, output_dim),
                "#2512 fixture requires an M={basis_width} by p={output_dim} Arrow border"
            );
        }
        for (term, rho) in &mut fits {
            // The joint fit refuses an identically zero decoder (it has no column
            // space to install a frame from), so each fresh term is seeded from the
            // data first, deterministically, as that refusal directs (gam#4582).
            term.refit_decoder_least_squares_at_current_state(target.view(), Some(rho))
                .unwrap();
            term.run_joint_fit_arrow_schur(target.view(), rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
                .unwrap();
        }

        let reference = fits[0].0.atoms[0].decoder_coefficients();
        assert_eq!(
            reference.len(),
            basis_width * output_dim,
            "#2512 fitted decoder must retain its M={basis_width} by p={output_dim} Arrow border"
        );
        let reference_norm = reference
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(
            reference_norm > 1.0,
            "#2512 M={basis_width} fixture must exercise a real fitted decoder, got norm {reference_norm:.6e}"
        );
        for (repetition, (term, _)) in fits.iter().enumerate().skip(1) {
            let decoder = term.atoms[0].decoder_coefficients();
            let differing = reference
                .iter()
                .zip(decoder.iter())
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
            assert_eq!(
                differing,
                0,
                "#2512 M={basis_width}: fresh one-step fit {repetition} differs from fit 0 in {differing}/{} decoder coefficients",
                decoder.len()
            );
        }
    }
}
