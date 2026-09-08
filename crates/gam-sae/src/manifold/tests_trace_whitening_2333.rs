//! The actual Trace consumer must preserve row whitening and the deflation
//! differential, independently of the dense adjoint's contraction code.
#![cfg(test)]

use super::*;
use ndarray::array;

fn inverse_fixture() -> Array2<f64> {
    array![[2.0, 0.3, -0.2], [0.3, 3.0, 0.4], [-0.2, 0.4, 4.0]]
}

fn spectrum(
    raw: [f64; 3],
    conditioned: [f64; 3],
    decisions: [RowSpectralConditioning; 3],
) -> RowDeflationSpectrum {
    RowDeflationSpectrum {
        evecs: array![[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]],
        raw_evals: Array1::from_vec(raw.to_vec()),
        cond_evals: Array1::from_vec(conditioned.to_vec()),
        conditioning: std::sync::Arc::from(decisions),
    }
}

fn verify_fold(directions: &[Array1<f64>], spectrum: Option<&RowDeflationSpectrum>) -> Array2<f64> {
    let inverse = inverse_fixture();
    let folded = SaeManifoldTerm::deflation_folded_trace_weight(&inverse, directions, spectrum);
    let mut largest_correction = 0.0_f64;
    // The diagonal and symmetric off-diagonal basis spans every symmetric D.
    // This proves the linear contraction identity without sampling directions.
    for row in 0..3 {
        for column in row..3 {
            let mut derivative = Array2::<f64>::zeros((3, 3));
            derivative[[row, column]] = 1.0;
            derivative[[column, row]] = 1.0;
            let raw = inverse.dot(&derivative).diag().sum();
            let correction = SaeManifoldTerm::deflation_block_correction(
                &inverse,
                &derivative,
                directions,
                spectrum,
            );
            let actual = (&folded * &derivative).sum();
            let expected = raw - correction;
            assert!(
                (actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                "D[{row},{column}]: folded={actual} independent={expected}"
            );
            largest_correction = largest_correction.max(correction.abs());
        }
    }
    assert!(
        largest_correction > 1e-6,
        "an identity fold must fail this fixture"
    );
    folded
}

#[test]
fn deflation_folded_trace_weight_reproduces_contract_then_subtract_2333() {
    let spectrum = spectrum(
        [0.25, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [
            RowSpectralConditioning::UnitDeflated,
            RowSpectralConditioning::Raw,
            RowSpectralConditioning::Raw,
        ],
    );
    verify_fold(&[spectrum.evecs.column(0).to_owned()], Some(&spectrum));
}

#[test]
fn deflation_folded_trace_weight_covers_gauge_only_and_undeflated_branches_2333() {
    verify_fold(&[array![0.6, -0.8]], None);
    let inverse = inverse_fixture();
    let folded = SaeManifoldTerm::deflation_folded_trace_weight(&inverse, &[], None);
    assert_eq!(folded, inverse);
}

#[test]
fn deflation_folded_trace_weight_handles_a_degenerate_pair_split_by_conditioning_2333() {
    let spectrum = spectrum(
        [0.25, 0.25, 3.0],
        [0.25, 1.0, 3.0],
        [
            RowSpectralConditioning::Raw,
            RowSpectralConditioning::UnitDeflated,
            RowSpectralConditioning::Raw,
        ],
    );
    let folded = verify_fold(&[], Some(&spectrum));
    assert!(
        (folded[[0, 1]] - folded[[1, 0]]).abs() > 1e-6,
        "both triangle halves are required for the degenerate split convention"
    );
}

#[test]
fn deflation_folded_trace_weight_prices_a_spectrum_with_no_deflated_direction_2333() {
    let spectrum = spectrum(
        [0.25, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [
            RowSpectralConditioning::FloorClamped,
            RowSpectralConditioning::Raw,
            RowSpectralConditioning::Raw,
        ],
    );
    verify_fold(&[], Some(&spectrum));
}

fn assert_adjoint_parity(reference: &SaeArrowVector, actual: &SaeArrowVector) -> (f64, f64) {
    assert_eq!(reference.t.len(), actual.t.len());
    assert_eq!(reference.beta.len(), actual.beta.len());
    let mut gap = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for (expected, observed) in reference
        .t
        .iter()
        .chain(reference.beta.iter())
        .zip(actual.t.iter().chain(actual.beta.iter()))
    {
        assert!(expected.is_finite() && observed.is_finite());
        gap = gap.max((expected - observed).abs());
        magnitude = magnitude.max(expected.abs());
    }
    assert!(magnitude > 0.0);
    assert!(
        gap <= 1e-12 * (1.0 + magnitude),
        "gap={gap:e}, magnitude={magnitude:e}"
    );
    (gap, magnitude)
}

#[test]
fn softmax_trace_whitening_prefold_matches_dense_adjoint_2333() {
    let (mut term, target, rho) =
        crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
    term.gpu_policy = gam_gpu::GpuPolicy::Off;
    // One exactly saturated softmax row has zero free-logit sensitivity and
    // zero entropy curvature. Its coordinate block therefore has an exact
    // null direction, forcing the production spectral quotient to run. Other
    // rows retain both live atoms and nonzero whitening/adjoint channels.
    term.assignment.logits[[0, 0]] = 1.0e3;
    let (n, p) = (term.n_obs(), term.output_dim());
    assert_eq!((n, p), (10, 3));
    let factors = Array2::from_shape_fn((n, p * p), |(row, column)| {
        let (out, rank) = (column / p, column % p);
        if out == rank {
            1.0 + (row + rank + 1) as f64 / (n + p) as f64
        } else {
            (out + rank + 1) as f64 / (4 * (n + p)) as f64
        }
    });
    term.set_row_metric(
        gam_problem::RowMetric::behavioral_fisher(std::sync::Arc::new(factors), p, p).unwrap(),
    )
    .unwrap();
    assert!(term.whiten_logdet_row_jets());
    let metric = term.row_metric().unwrap();
    assert_ne!(
        metric.factor_entry(0, 0, 0),
        metric.factor_entry(n - 1, 0, 0)
    );

    // Factor one declared state. This is an algebraic adjoint test; no fitted
    // result or optimizer convergence is asserted and no fixture search runs.
    let mut system = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options).unwrap();
    let live_rows = cache
        .deflation_row_spectra
        .iter()
        .filter(|spectrum| {
            spectrum.as_ref().is_some_and(|spectrum| {
                spectrum
                    .raw_evals
                    .iter()
                    .zip(spectrum.cond_evals.iter())
                    .any(|(raw, conditioned)| raw != conditioned)
            })
        })
        .count();
    assert!(
        live_rows > 0,
        "the Trace consumer must exercise spectral deflation"
    );
    let solver = DeflatedArrowSolver::plain(&cache);
    let joint_inverse = term.materialize_joint_inverse(&cache, &solver).unwrap();
    let coordinate_inverse = term.materialize_block_diag_t_inverse(&cache);
    let dense = |inverse: &Array2<f64>| {
        term.logdet_theta_adjoint_dense(
            &rho,
            &cache,
            inverse,
            ThetaAdjointDhChannel::All,
            false,
            false,
            None,
        )
        .unwrap()
    };
    let joint = term.logdet_theta_adjoint(&rho, &cache, &solver).unwrap();
    let coordinate = term
        .coordinate_block_logdet_theta_adjoint(&rho, &cache, EvidenceOperator::Majorizer, None)
        .unwrap();
    let (joint_gap, joint_scale) = assert_adjoint_parity(&dense(&joint_inverse), &joint);
    let (coordinate_gap, coordinate_scale) =
        assert_adjoint_parity(&dense(&coordinate_inverse), &coordinate);
    eprintln!(
        "#2333 live_rows={live_rows} joint_gap={joint_gap:e} joint_scale={joint_scale:e} coordinate_gap={coordinate_gap:e} coordinate_scale={coordinate_scale:e}"
    );
    let repeated = term.logdet_theta_adjoint(&rho, &cache, &solver).unwrap();
    assert_eq!(joint.t, repeated.t);
    assert_eq!(joint.beta, repeated.beta);
}
