//! Unit tests of the support-sparse term, moved out of `support_term.rs` unchanged (#780
//! line-count gate). The module is still `support_term::tests`.
#![cfg(test)]

use super::*;
use crate::assignment_state::SaeAssignmentAtomSpec;
use ndarray::array;

/// Three atoms of two bases over two channels, with every atom shared by two
/// rows so the scatter has a real accumulation order to preserve.
fn beta_operator_fixture() -> SupportBetaOperator {
    let width = 2usize;
    let basis_sizes = vec![2usize, 2, 2];
    let beta_offsets = vec![0usize, 4, 8];
    // Deliberately not round: partial sums of these are not exactly
    // representable, so a reassociated sum would differ in the low bits.
    let phi = |a: f64, b: f64| ndarray::Array1::from(vec![a, b]);
    let mk = |offset: usize, a: f64, b: f64| SupportBasisBlock {
        beta_offset: offset,
        phi: phi(a, b),
    };
    let third = 1.0_f64 / 3.0;
    let root = 2.0_f64.sqrt() / 7.0;
    let rows = vec![
        SupportLinearizedRow {
            blocks: vec![mk(0, third, root), mk(4, -root, third * 0.5)],
            jacobian: ndarray::Array2::zeros((1, 1)),
        },
        SupportLinearizedRow {
            blocks: vec![mk(4, third * 1.7, -root), mk(8, root * 3.1, third)],
            jacobian: ndarray::Array2::zeros((1, 1)),
        },
        SupportLinearizedRow {
            blocks: vec![mk(0, -third * 0.9, root * 2.3), mk(8, third, -root)],
            jacobian: ndarray::Array2::zeros((1, 1)),
        },
    ];
    let mut atom_blocks: Vec<Vec<(u32, u32)>> = vec![Vec::new(); 3];
    for (row_index, row) in rows.iter().enumerate() {
        for (block_index, block) in row.blocks.iter().enumerate() {
            let atom = beta_offsets
                .iter()
                .position(|&o| o == block.beta_offset)
                .expect("offset belongs to an atom");
            atom_blocks[atom].push((row_index as u32, block_index as u32));
        }
    }
    let penalties = vec![
        array![[2.0, -0.5], [-0.5, 1.25]],
        array![[1.0, third], [third, 3.0]],
        array![[0.75, 0.0], [0.0, 0.5]],
    ];
    SupportBetaOperator {
        rows,
        atom_blocks,
        beta_offsets,
        basis_sizes,
        penalties,
        lambda_smooth: vec![0.7, 1.9, third],
        output_dim: width,
        beta_dim: 12,
    }
}

/// The serial sweep `apply` replaced, kept here as the reference the fan-out
/// has to reproduce exactly.
fn beta_operator_apply_serially(
    op: &SupportBetaOperator,
    vector: ndarray::ArrayView1<'_, f64>,
    out: &mut Array1<f64>,
) {
    out.fill(0.0);
    let mut output = vec![0.0; op.output_dim];
    for row in &op.rows {
        output.fill(0.0);
        for block in &row.blocks {
            for basis in 0..block.phi.len() {
                let base = block.beta_offset + basis * op.output_dim;
                for channel in 0..op.output_dim {
                    output[channel] += block.phi[basis] * vector[base + channel];
                }
            }
        }
        for block in &row.blocks {
            for basis in 0..block.phi.len() {
                let base = block.beta_offset + basis * op.output_dim;
                for channel in 0..op.output_dim {
                    out[base + channel] += block.phi[basis] * output[channel];
                }
            }
        }
    }
    for atom in 0..op.penalties.len() {
        let lambda = op.lambda_smooth[atom];
        let m = op.basis_sizes[atom];
        let offset = op.beta_offsets[atom];
        for left in 0..m {
            for right in 0..m {
                let weight = lambda * op.penalties[atom][[left, right]];
                for channel in 0..op.output_dim {
                    out[offset + left * op.output_dim + channel] +=
                        weight * vector[offset + right * op.output_dim + channel];
                }
            }
        }
    }
}

#[test]
fn beta_operator_fan_out_is_bit_identical_to_the_serial_sweep() {
    let op = beta_operator_fixture();
    let vector = Array1::from(
        (0..12)
            .map(|i| ((i as f64) * 0.37).sin() + (i as f64) / 7.0)
            .collect::<Vec<f64>>(),
    );

    let mut expected = Array1::<f64>::zeros(12);
    beta_operator_apply_serially(&op, vector.view(), &mut expected);
    let mut actual = Array1::<f64>::zeros(12);
    op.apply(vector.view(), &mut actual);

    for index in 0..12 {
        // EXACT: reassociating the sum would move the low bits, and moving
        // them is precisely what this test exists to forbid.
        assert_eq!(
            actual[index].to_bits(),
            expected[index].to_bits(),
            "entry {index}: fan-out {} is not bit-identical to serial {}",
            actual[index],
            expected[index]
        );
    }
    // Guard against a fixture that made the assertion trivial.
    assert!(
        expected.iter().any(|v| v.abs() > 1e-6),
        "fixture produced an all-zero reference, so the comparison proves nothing"
    );
}

#[test]
fn beta_operator_fan_out_is_stable_across_repeated_application() {
    // rayon may split the work differently between calls; the result must
    // not depend on how it happened to schedule.
    let op = beta_operator_fixture();
    let vector = Array1::from((0..12).map(|i| 1.0 / (i as f64 + 1.3)).collect::<Vec<f64>>());
    let mut first = Array1::<f64>::zeros(12);
    op.apply(vector.view(), &mut first);
    for _ in 0..8 {
        let mut again = Array1::<f64>::zeros(12);
        op.apply(vector.view(), &mut again);
        for index in 0..12 {
            assert_eq!(again[index].to_bits(), first[index].to_bits());
        }
    }
}

#[test]
fn beta_operator_blocks_diagonal_and_dense_match_column_probes() {
    // #2576: the block-Jacobi build reads these instead of probing `apply`, so
    // they must equal the probed columns. They add the same terms in the same
    // order, so the comparison is exact.
    use gam_solve::arrow_schur::{BetaBlockId, BetaPenaltyOp};
    let op = beta_operator_fixture();
    let k = op.beta_dim;
    let mut probed = Array2::<f64>::zeros((k, k));
    for column in 0..k {
        let mut unit = Array1::<f64>::zeros(k);
        unit[column] = 1.0;
        let mut applied = Array1::<f64>::zeros(k);
        op.apply(unit.view(), &mut applied);
        probed.column_mut(column).assign(&applied);
    }
    let mut diagonal = vec![0.0_f64; k];
    op.diagonal(&mut diagonal);
    for index in 0..k {
        assert_eq!(diagonal[index], probed[[index, index]], "diagonal entry {index}");
    }
    // Each atom's own range, then one range that spans atoms 0, 1 and 2.
    let ranges = vec![0..4, 4..8, 8..12, 2..10];
    for id in 0..ranges.len() {
        let range = ranges[id].clone();
        let width = range.end - range.start;
        let mut block = Array2::<f64>::zeros((width, width));
        op.block(BetaBlockId(id), &ranges, &mut block);
        for bi in 0..width {
            for bj in 0..width {
                assert_eq!(
                    block[[bi, bj]],
                    probed[[range.start + bi, range.start + bj]],
                    "range {range:?} entry ({bi}, {bj})"
                );
            }
        }
    }
    let dense = op.to_dense();
    for i in 0..k {
        for j in 0..k {
            assert_eq!(dense[[i, j]], probed[[i, j]], "dense entry ({i}, {j})");
        }
    }
    // The spanning range has to hold a cross-atom coupling, or it checks
    // nothing the per-atom ranges do not.
    assert!(
        (2..4).any(|i| (4..10).any(|j| probed[[i, j]] != 0.0)),
        "fixture has no cross-atom coupling inside the spanning range"
    );
}

/// `S` is rank 2 with null direction `e3`; `G` is full rank.
fn penalized_solve_fixture() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let penalty = array![[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
    let gram = array![[3.0, 1.0, 0.5], [1.0, 4.0, 0.25], [0.5, 0.25, 2.0]];
    let rhs = array![[1.0], [2.0], [3.0]];
    (gram, penalty, rhs)
}

#[test]
fn penalized_solve_survives_the_smoothing_fellner_schall_actually_produces() {
    let (gram, penalty, rhs) = penalized_solve_fixture();
    // The magnitude Fellner-Schall reaches when an atom's roughness goes to
    // zero -- the ladder picking the linear rung, not a divergence.
    let lambda = 2.2e16;

    // The old route: assemble `G + lambda*S` and solve it. This must FAIL,
    // or the fix below is answering a question nobody asked.
    let mut assembled = &penalty * lambda;
    assembled += &gram;
    let assembled_result = SaeSupportSparseTerm::solve_psd_minimum_norm(
        &assembled,
        &rhs,
        "assembled",
    );
    assert!(
        assembled_result.is_err(),
        "assembling G + lambda*S was expected to lose null(S) to its own rank floor, \
         but it returned {assembled_result:?}"
    );

    let solved = SaeSupportSparseTerm::solve_penalized_normal_equations(
        &gram, &penalty, lambda, &rhs, "penalized",
    )
    .expect("the lambda-free scaling must solve what the assembled matrix could not");

    // Exact limit, by hand: the penalty annihilates everything outside
    // null(S) = span(e3), so beta = e3 * (rhs_3 / G_33) = e3 * 3/2.
    assert!(solved[[0, 0]].abs() < 1e-9, "range(S) must be driven to zero, got {}", solved[[0, 0]]);
    assert!(solved[[1, 0]].abs() < 1e-9, "range(S) must be driven to zero, got {}", solved[[1, 0]]);
    assert!(
        (solved[[2, 0]] - 1.5).abs() < 1e-9,
        "null(S) must keep its unpenalised least squares value 1.5, got {}",
        solved[[2, 0]]
    );
}

#[test]
fn penalized_solve_agrees_with_the_assembled_matrix_where_that_is_conditioned() {
    let (gram, penalty, rhs) = penalized_solve_fixture();
    // Stability at 1e16 is worthless if it moved the answer at lambdas that
    // were never in trouble.
    for lambda in [0.0, 1e-3, 1.0, 25.0, 1e4] {
        let mut assembled = &penalty * lambda;
        assembled += &gram;
        let reference =
            SaeSupportSparseTerm::solve_psd_minimum_norm(&assembled, &rhs, "reference")
                .expect("well-conditioned assembled solve");
        let solved = SaeSupportSparseTerm::solve_penalized_normal_equations(
            &gram, &penalty, lambda, &rhs, "penalized",
        )
        .expect("well-conditioned scaled solve");
        for index in 0..3 {
            let gap = (solved[[index, 0]] - reference[[index, 0]]).abs();
            assert!(
                gap < 1e-9 * reference[[index, 0]].abs().max(1.0),
                "lambda={lambda} entry {index}: scaled {} vs assembled {}",
                solved[[index, 0]],
                reference[[index, 0]]
            );
        }
    }
}
use std::sync::Arc;

fn atom(
    name: &str,
    kind: SaeAtomBasisKind,
    d: usize,
    evaluator: Arc<dyn SaeBasisSecondJet>,
    coords: &[f64],
    decoder: Array2<f64>,
) -> SaeManifoldAtom {
    let coord = Array2::from_shape_vec((1, d), coords.to_vec()).expect("coords");
    let (phi, jet) = evaluator.evaluate(coord.view()).expect("evaluate");
    let m = phi.ncols();
    SaeManifoldAtom::new_with_provided_function_gram(
        name,
        kind,
        d,
        phi,
        jet,
        decoder,
        Array2::eye(m),
    )
    .expect("atom")
    .with_basis_second_jet(evaluator)
}

/// #2469: a frozen-decoder sweep that moves no coordinate while the certificate fails
/// repeats forever, so the solve refuses instead of sweeping on. One row on a linear
/// chart with slope `1.6e-8` sits at `t = 0` against target `1.3`, with no prior. Its
/// gradient, `1.3 · 1.6e-8 ≈ 2.1e-8`, is above the certificate's bar
/// `tol · max(1, |f|) = tol` (|f| = 0.845, tol ≈ 1.49e-8) and at or below the row skip's
/// `tol · (1 + |f|)`, so every sweep skips the row and the second sweep proves the stall.
#[test]
fn frozen_decoder_solve_refuses_a_motionless_sweep_2469() {
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
    let atoms = vec![atom(
        "motionless-line",
        SaeAtomBasisKind::Linear,
        1,
        evaluator,
        &[0.0],
        array![[0.0], [1.6e-8]],
    )];
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        1,
        1,
        1,
        vec![SaeAssignmentAtomSpec::euclidean(1)],
        vec![vec![0]],
        vec![vec![1.0]],
        vec![vec![0.0]],
    )
    .expect("state");
    let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let target = array![[1.3_f64]];
    let ard = vec![vec![0.0_f64]];
    let tolerance = term.fixed_point_tolerance();
    let error = term
        .solve_coordinates_fixed_decoder(target.view(), &ard, tolerance, 1.0)
        .expect_err("a row the sweep never moves cannot certify");
    assert!(
        error.contains("stalled at sweep 2"),
        "the second motionless sweep proves the stall; got: {error}"
    );
}

/// #2469 (SPEC rule 23): the frozen-decoder solve has no sweep budget. One row on a
/// linear chart, decode `f(t) = t`, starts at `t = 0` against target `4`, and a trust
/// radius of `1e-2` moves it at most `1e-2` per sweep. It cannot come near the
/// minimizer before sweep 400, so it must certify well past the 256 sweeps the engine
/// budget used to allow.
#[test]
fn frozen_decoder_solve_certifies_a_slow_row_past_256_sweeps_2469() {
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
    let atoms = vec![atom(
        "slow-line",
        SaeAtomBasisKind::Linear,
        1,
        evaluator,
        &[0.0],
        array![[0.0], [1.0]],
    )];
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        1,
        1,
        1,
        vec![SaeAssignmentAtomSpec::euclidean(1)],
        vec![vec![0]],
        vec![vec![1.0]],
        vec![vec![0.0]],
    )
    .expect("state");
    let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let target = array![[4.0_f64]];
    let ard = vec![vec![1.0e-6_f64]];
    let tolerance = term.fixed_point_tolerance();
    let report = term
        .solve_coordinates_fixed_decoder(target.view(), &ard, tolerance, 1.0e-2)
        .expect("a row that crawls without stalling must certify");
    assert!(report.recurred, "the solve returns only a recurred state");
    assert!(
        report.iterations > 256,
        "a row limited to 1e-2 per sweep needs over 400 sweeps to reach t near 4; \
         certified at sweep {}",
        report.iterations
    );
}

/// #2576: where the evidence lane takes the dense reduced Schur's complete spectrum, the
/// profile adjoint's majorizer inverse folds over the bundle's vectors instead of running
/// a √ε CG. The fold is that inverse only if the bundle priced the SAME reduced Schur the
/// adjoint eliminates against: one assembled system and the same undamped row factors.
/// On a resolved positive definite fixture the fold must solve the majorizer arrow
/// `B x = r` to the backward error of a backward-stable dense solve, `dim²·ε` (the γ bound
/// of the Householder tridiagonalisation the eigensystem rests on). A bundle priced at
/// another λ, a stale operator, must miss that bar by orders of magnitude, or the bar
/// discriminates nothing.
#[test]
fn majorizer_inverse_folds_the_dense_spectrum_bundle_into_the_arrow_inverse_2576() {
    let periodic: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
    let patch: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
    let atoms = vec![
        atom(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            periodic,
            &[0.05],
            array![[0.2, -0.3], [1.1, 0.4], [-0.4, 0.9]],
        ),
        atom(
            "plane",
            SaeAtomBasisKind::Linear,
            2,
            patch,
            &[0.1, -0.2],
            array![[0.3, 0.1], [2.0, -0.7], [-1.0, 1.3]],
        ),
    ];
    let specs = vec![
        SaeAssignmentAtomSpec {
            latent_dim: 1,
            manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
            retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
        },
        SaeAssignmentAtomSpec::euclidean(2),
    ];
    // Rows fire both atoms, the circle alone, or the plane alone, so both constant
    // columns are identified separately and the decoder gram spans every coefficient
    // without leaning on the penalties' rank.
    let n_obs = 9usize;
    let mut indices = Vec::with_capacity(n_obs);
    let mut gates = Vec::with_capacity(n_obs);
    let mut coords = Vec::with_capacity(n_obs);
    let mut target = Array2::<f64>::zeros((n_obs, 2));
    for row in 0..n_obs {
        let s = row as f64;
        let phase = 0.18 * (0.7 * s + 0.3).sin();
        let plane = [0.8 * (1.3 * s).cos(), 0.2 * s - 0.8];
        match row % 3 {
            0 => {
                indices.push(vec![0u32, 1]);
                gates.push(vec![1.0, 1.0]);
                coords.push(vec![phase, plane[0], plane[1]]);
            }
            1 => {
                indices.push(vec![0u32]);
                gates.push(vec![1.0]);
                coords.push(vec![phase]);
            }
            _ => {
                indices.push(vec![1u32]);
                gates.push(vec![1.0]);
                coords.push(plane.to_vec());
            }
        }
        target[[row, 0]] = (0.9 * s).sin() + 0.3;
        target[[row, 1]] = (0.5 * s).cos() - 0.1 * s;
    }
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        n_obs, 2, 2, specs, indices, gates, coords,
    )
    .expect("state");
    let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let ard = vec![vec![1.0], vec![1.0, 1.0]];
    let lambda = vec![0.4, 2.2];
    let system = term
        .assemble_arrow_schur(target.view(), &lambda, &ard)
        .expect("assemble");
    let dense_bundle = |system: &ArrowSchurSystem| {
        let mut lane =
            gam_solve::arrow_schur::SurrogateLaneState::new(sae_surrogate_lane_config());
        lane.request_logdet_derivative_bundle();
        let options = gam_solve::arrow_schur::ArrowSolveOptions::inexact_pcg()
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        gam_solve::arrow_schur::matrix_free_arrow_evidence_evaluation(
            system,
            0.0,
            0.0,
            &options,
            gam_solve::arrow_schur::SCHUR_SLQ_LOGDET_PROBES,
            gam_solve::arrow_schur::SCHUR_SLQ_LOGDET_LANCZOS_STEPS,
            gam_solve::arrow_schur::SCHUR_SLQ_LOGDET_SEED,
            &mut lane,
            true,
        )
        .expect("dense evidence evaluation");
        lane.take_logdet_derivative_bundle()
            .expect("the evaluation emits its derivative bundle")
    };
    let bundle = dense_bundle(&system);
    let vectors = bundle
        .exact_inverse_vectors()
        .expect("a dense spectrum bundle spans the inverse it priced");
    assert_eq!(vectors.len(), system.k, "one vector per reduced-Schur mode");
    let factors = CpuBatchedBlockSolver
        .factor_blocks(&system.rows, 0.0, system.d, true)
        .expect("row factors");

    let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
    let dim = coordinate_dim + system.k;
    let mut majorizer = Array2::<f64>::zeros((dim, dim));
    for column in 0..dim {
        let mut unit = SaeArrowVector {
            t: Array1::zeros(coordinate_dim),
            beta: Array1::zeros(system.k),
        };
        if column < coordinate_dim {
            unit.t[column] = 1.0;
        } else {
            unit.beta[column - coordinate_dim] = 1.0;
        }
        let applied = support_arrow_majorizer_apply(&system, &unit).expect("majorizer apply");
        majorizer
            .slice_mut(ndarray::s![..coordinate_dim, column])
            .assign(&applied.t);
        majorizer
            .slice_mut(ndarray::s![coordinate_dim.., column])
            .assign(&applied.beta);
    }
    let symmetric = (&majorizer + &majorizer.t()) * 0.5;
    let (spectrum, _) = symmetric.eigh(Side::Lower).expect("majorizer spectrum");
    let lowest = spectrum.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = spectrum.iter().copied().fold(0.0_f64, f64::max);
    // The reduced Schur's spectrum lies inside B's, so a B resolved above √ε·‖B‖ leaves
    // no mode for the evidence to pin, and the bundle must be the plain inverse.
    assert!(
        lowest > f64::EPSILON.sqrt() * highest,
        "the fixture must be resolved positive definite: spectrum [{lowest:.3e}, {highest:.3e}]"
    );

    let rhs = SaeArrowVector {
        t: Array1::from_shape_fn(coordinate_dim, |i| (0.37 * i as f64).sin() + 0.1),
        beta: Array1::from_shape_fn(system.k, |i| (0.61 * i as f64).cos() - 0.2),
    };
    let mut rhs_flat = Array1::<f64>::zeros(dim);
    rhs_flat
        .slice_mut(ndarray::s![..coordinate_dim])
        .assign(&rhs.t);
    rhs_flat
        .slice_mut(ndarray::s![coordinate_dim..])
        .assign(&rhs.beta);
    let backward_error = |solved: &SaeArrowVector| {
        let mut flat = Array1::<f64>::zeros(dim);
        flat.slice_mut(ndarray::s![..coordinate_dim])
            .assign(&solved.t);
        flat.slice_mut(ndarray::s![coordinate_dim..])
            .assign(&solved.beta);
        let residual = &rhs_flat - &majorizer.dot(&flat);
        residual.dot(&residual).sqrt()
            / (highest * flat.dot(&flat).sqrt() + rhs_flat.dot(&rhs_flat).sqrt())
    };

    let folded = support_arrow_majorizer_inverse(&system, &factors, &rhs, Some(vectors))
        .expect("folded majorizer inverse");
    let iterated = support_arrow_majorizer_inverse(&system, &factors, &rhs, None)
        .expect("CG majorizer inverse");
    let folded_error = backward_error(&folded);
    let iterated_error = backward_error(&iterated);
    let bar = (dim * dim) as f64 * f64::EPSILON;
    assert!(
        folded_error <= bar,
        "the dense-spectrum fold must invert the majorizer arrow: backward error \
         {folded_error:.3e} > dim²·ε = {bar:.3e} (the √ε CG route reaches {iterated_error:.3e})"
    );

    let stale_lambda = vec![lambda[0] * 16.0, lambda[1] / 16.0];
    let stale_system = term
        .assemble_arrow_schur(target.view(), &stale_lambda, &ard)
        .expect("stale assemble");
    let stale_bundle = dense_bundle(&stale_system);
    let stale = support_arrow_majorizer_inverse(
        &system,
        &factors,
        &rhs,
        stale_bundle.exact_inverse_vectors(),
    )
    .expect("stale fold");
    let stale_error = backward_error(&stale);
    assert!(
        stale_error > f64::EPSILON.sqrt(),
        "a bundle priced at another λ must not invert this arrow, or the bar above \
         discriminates nothing: backward error {stale_error:.3e}"
    );
}

/// #2634: the positive-definite certificate that spares the saddle classifier
/// its eigensystems decides in both directions at its own band. `S₀` is
/// planted with a unit diagonal and `λ_min(S₀) = t`: the block
/// `[[1, 1 − t], [1 − t, 1]]` (eigenvalues `t` and `2 − t`) beside an identity,
/// so the certificate's scaling is exact to rounding and its band is
/// `δ = (3n² + n + 2)·ε·‖S₀‖_F`. With `A = S₀ + τ·I` and `B = I`, `A − τ·B = S₀`
/// for either sign of `τ`.
#[test]
fn shifted_pd_certificate_decides_at_its_band_in_both_directions_2634() {
    let dim = 6usize;
    let planted = |t: f64, tau: f64| {
        let mut a = Array2::<f64>::eye(dim);
        a[[0, 1]] = 1.0 - t;
        a[[1, 0]] = 1.0 - t;
        for index in 0..dim {
            a[[index, index]] += tau;
        }
        a
    };
    let identity = Array2::<f64>::eye(dim);
    let n = dim as f64;
    let delta = (3.0 * n * n + n + 2.0) * f64::EPSILON * (n + 2.0).sqrt();
    for tau in [-0.5 * support_outer_curvature_floor(), 0.3] {
        let certified =
            certify_shifted_pd(planted(2.0 * delta, tau).view(), identity.view(), tau)
                .expect("λ_min(S₀) = 2δ is certified");
        assert_eq!(certified.dim, dim);
        assert_eq!(certified.tau, tau);
        assert!(
            matches!(certified.band, BandProvenance::Dense { scaled_frobenius } if scaled_frobenius > 0.0),
            "{:?}",
            certified.band
        );
        assert!(
            (certified.delta - delta).abs() <= 1.0e-6 * delta,
            "band {} against the planted {delta}",
            certified.delta
        );
        for t in [0.5 * delta, -delta] {
            let refusal = certify_shifted_pd(planted(t, tau).view(), identity.view(), tau)
                .expect_err("λ_min(S₀) inside or below the band is refused");
            assert!(
                matches!(refusal, ShiftedPdRefusal::CholeskyFailed { pivot: Some(_), .. }),
                "λ_min(S₀) = {t:e} at τ = {tau:e}: {refusal:?}"
            );
        }
    }
    let mut indefinite = Array2::<f64>::eye(dim);
    indefinite[[3, 3]] = -1.0;
    assert_eq!(
        certify_shifted_pd(indefinite.view(), identity.view(), 0.0),
        Err(ShiftedPdRefusal::NonPositiveDiagonal {
            index: 3,
            value: -1.0
        })
    );
}

/// #2634 second-order gate: a periodic chart can be first-order attracted
/// to a blockwise fixed point whose exact joint curvature is negative.  The
/// generalized mode has arbitrary sign, so the escape must test both signs,
/// choose an actual objective decrease, and leave the ordinary fixed-point
/// solver in a certifiable minimum basin.
#[test]
fn exact_generalized_mode_escapes_a_periodic_saddle_2634() {
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
    // At t=1/2 the von-Mises prior has curvature -1.  A small sine decoder
    // contributes only (0.1*2pi)^2 through Gauss--Newton, so the exact
    // coordinate curvature remains resolved negative while the majorizer
    // pencil is strictly positive and therefore classifiable.
    let atoms = vec![atom(
        "periodic-saddle",
        SaeAtomBasisKind::Periodic,
        1,
        evaluator,
        &[0.5],
        array![[0.0], [0.1], [0.0]],
    )];
    let specs = vec![SaeAssignmentAtomSpec {
        latent_dim: 1,
        manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
        retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
    }];
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        1,
        1,
        1,
        specs,
        vec![vec![0]],
        vec![vec![1.0]],
        vec![vec![0.5]],
    )
    .expect("state");
    let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let target = array![[0.0_f64]];
    let lambda = vec![1.0_f64];
    let ard = vec![vec![1.0_f64]];
    let system = term
        .assemble_arrow_schur(target.view(), &lambda, &ard)
        .expect("arrow system");
    let (beta_offsets, _) = term.beta_layout().expect("beta layout");
    let rows = term
        .support_outer_differential_rows(target.view(), &ard, &beta_offsets)
        .expect("exact differential rows");
    let mode = term
        .support_outer_negative_curvature_mode(&system, &rows)
        .expect("curvature classification")
        .expect("the planted saddle has a resolved negative mode");
    assert!(
        mode.curvature < -f64::EPSILON.sqrt(),
        "planted generalized curvature must be resolved negative, got {}",
        mode.curvature,
    );

    let before = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("saddle objective");
    let escaped = term
        .escape_support_negative_curvature(
            target.view(),
            &lambda,
            &ard,
            &mode,
            before,
            1.0,
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .expect("exact saddle escape")
        .expect("one orientation must descend exact negative curvature");
    assert!(escaped < before, "escape must strictly lower the objective");

    let report = term
        .solve_fixed_point(target.view(), &lambda, &ard, 1.0e-8, 1.0)
        .expect("escaped state converges to a minimum basin");
    assert!(report.recurred && report.objective < before);
}

/// #2634 — the support first-order screen accepts the same two currencies as
/// the dense manifold lane. Replicating rows makes the raw decoder gradient
/// extensive while its diagonal curvature grows by the identical factor;
/// only the componentwise diagonal-scaled limb remains invariant. The screen
/// schedules the exact Newton displacement certificate (#2933 F08).
#[test]
fn support_first_order_screen_survives_raw_global_refusal_2634() {
    let rows = 1_024usize;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
    let atoms = vec![atom(
        "replicated-line",
        SaeAtomBasisKind::Linear,
        1,
        evaluator,
        &[0.0],
        array![[0.0], [10.0]],
    )];
    let coordinates: Vec<Vec<f64>> = (0..rows)
        .map(|row| vec![if row % 2 == 0 { -10.0 } else { 10.0 }])
        .collect();
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        rows,
        1,
        1,
        vec![SaeAssignmentAtomSpec::euclidean(1)],
        vec![vec![0]; rows],
        vec![vec![1.0]; rows],
        coordinates,
    )
    .expect("state");
    let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let fitted = term.reconstruct().expect("fitted");
    let target = fitted.mapv(|value| value + 1.0e-4);
    let lambda = vec![0.0];
    let ard = vec![vec![0.0]];
    let stationarity = term
        .raw_stationarity(target.view(), &lambda, &ard)
        .expect("stationarity");
    let objective = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("objective");
    let objective_scale = objective.abs().max(1.0);
    let parameter_scale = term.parameter_iterate_scale().expect("parameter scale");
    let tolerance = 1.0e-5;

    assert!(
        stationarity.max_abs() > tolerance * objective_scale,
        "the replicated raw/global certificate must refuse"
    );
    assert!(
        stationarity.scaled_max_abs() <= tolerance * parameter_scale,
        "the componentwise diagonal-scaled screen must be intensive"
    );
    assert!(stationarity.first_order_screen(
        objective_scale,
        parameter_scale,
        tolerance
    ));

    let error = accumulate_parameter_scaled_gradient(
        &mut 0.0,
        1.0,
        0.0,
        SaeInnerKktScaleBlock::SharedDecoder,
        0,
    )
    .expect_err("zero curvature cannot scale a nonzero gradient");
    assert!(matches!(
        error,
        SaeInnerKktScaleError::InvalidCurvature { .. }
    ));
}

/// #2576 — a linear atom's affine gauge is profiled to the prior-selected
/// representative: the fit is unchanged, the penalized objective strictly
/// decreases, the routed coordinates are centered, and a second profile finds
/// nothing left to install.
#[test]
fn linear_affine_gauge_profile_lowers_the_prior_at_a_fixed_fit_2576() {
    let rows = 8usize;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
    let probe = Array2::from_shape_vec((1, 1), vec![0.0]).expect("probe");
    let (phi, jet) = evaluator.evaluate(probe.view()).expect("evaluate");
    let atoms = vec![
        SaeManifoldAtom::new_with_provided_function_gram(
            "offset-line",
            SaeAtomBasisKind::Linear,
            1,
            phi,
            jet,
            array![[0.3, -0.2], [2.0, 0.5]],
            array![[0.0, 0.0], [0.0, 1.0]],
        )
        .expect("atom")
        .with_basis_second_jet(evaluator),
    ];
    let coordinates: Vec<Vec<f64>> = (0..rows)
        .map(|row| vec![5.0 + 3.0 * (row as f64 - 3.5) / 3.5])
        .collect();
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        rows,
        1,
        1,
        vec![SaeAssignmentAtomSpec::euclidean(1)],
        vec![vec![0]; rows],
        vec![vec![1.0]; rows],
        coordinates,
    )
    .expect("state");
    let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let before = term.reconstruct().expect("fitted");
    let target = before.mapv(|value| value + 1.0e-3);
    let lambda = vec![0.5];
    let ard = vec![vec![1.0]];
    let objective_before = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("objective");

    assert_eq!(
        term.profile_linear_affine_gauges(&lambda, &ard)
            .expect("profile"),
        1,
        "an offset, unscaled linear atom must be profiled"
    );
    let after = term.reconstruct().expect("fitted");
    let scale = before.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        before
            .iter()
            .zip(after.iter())
            .all(|(left, right)| (left - right).abs() <= 1.0e-10 * scale),
        "the affine gauge must leave the reconstruction unchanged"
    );
    let objective_after = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("objective");
    assert!(
        objective_after < objective_before,
        "profiling must lower the penalized objective: {objective_before} -> {objective_after}"
    );
    let mean = (0..rows)
        .map(|row| term.assignment.coords_for_slot(row, 0)[0])
        .sum::<f64>()
        / rows as f64;
    assert!(mean.abs() <= 1.0e-12, "profiled coordinates must be centered, mean {mean}");
    assert_eq!(
        term.profile_linear_affine_gauges(&lambda, &ard)
            .expect("profile"),
        0,
        "a profiled atom is already at its prior-selected representative"
    );
}

/// #2576 — the degree-2 patch counterpart: profiling an offset, unscaled patch
/// atom leaves the fit unchanged and strictly lowers the penalized objective,
/// and a further profile never raises it.
#[test]
fn euclidean_patch_affine_gauge_profile_lowers_the_prior_at_a_fixed_fit_2576() {
    let rows = 8usize;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("patch"));
    let probe = Array2::from_shape_vec((1, 1), vec![0.0]).expect("probe");
    let (phi, jet) = evaluator.evaluate(probe.view()).expect("evaluate");
    // The flat Dirichlet Gram over reference rows [-1, 0, 1]: the sum of
    // `∇φ(r)∇φ(r)ᵀ` with `∇φ = [0, 1, 2r]`, zero on the constant.
    let atoms = vec![
        SaeManifoldAtom::new_with_provided_function_gram(
            "offset-patch",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            array![[0.3, -0.2], [2.0, 0.5], [0.4, 0.1]],
            array![[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 8.0]],
        )
        .expect("atom")
        .with_basis_second_jet(evaluator),
    ];
    let coordinates: Vec<Vec<f64>> = (0..rows)
        .map(|row| vec![2.5 + 1.5 * (row as f64 - 3.5) / 3.5])
        .collect();
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        rows,
        1,
        1,
        vec![SaeAssignmentAtomSpec::euclidean(1)],
        vec![vec![0]; rows],
        vec![vec![1.0]; rows],
        coordinates,
    )
    .expect("state");
    let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let before = term.reconstruct().expect("fitted");
    let target = before.mapv(|value| value + 1.0e-3);
    let lambda = vec![0.5];
    let ard = vec![vec![1.0]];
    let objective_before = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("objective");

    assert_eq!(
        term.profile_euclidean_patch_affine_gauges(&lambda, &ard)
            .expect("profile"),
        1,
        "an offset, unscaled patch atom must be profiled"
    );
    let after = term.reconstruct().expect("fitted");
    let scale = before.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        before
            .iter()
            .zip(after.iter())
            .all(|(left, right)| (left - right).abs() <= 1.0e-9 * scale),
        "the affine gauge must leave the reconstruction unchanged"
    );
    let objective_after = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("objective");
    assert!(
        objective_after < objective_before,
        "profiling must lower the penalized objective: {objective_before} -> {objective_after}"
    );
    term.profile_euclidean_patch_affine_gauges(&lambda, &ard)
        .expect("profile");
    let objective_again = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("objective");
    assert!(
        objective_again <= objective_after,
        "a further profile must not raise the objective: {objective_after} -> {objective_again}"
    );
}

/// #2933 F08 fixture: one degree-1 patch atom `f(t) = β₀ + β₁t` over eight rows
/// at `t₀ = ±1`, penalty `S = I`, ARD precision `α` and smoothing `λ = α`. The
/// fit is invariant along the scale orbit `(s·t₀, β₁*/s)`, so only the priors
/// curve it, `V(s) = ½αTs² + ½λβ₁*²s⁻²` with `T = Σt₀² = 8`, stationary at `s = 1`
/// when `β₁*² = αT/λ = T`. The target `y = (β₁* + α/β₁*)·t₀` leaves the residual
/// `r = α·t₀/β₁*`. That residual balances every coordinate's prior pull, and
/// because `Σt₀ = 0` and `Σt₀r = αT/β₁* = λβ₁*` it balances both decoder
/// gradients too. Neither affine profiler touches a degree-1 patch, so the orbit
/// is left to the solve.
fn scale_orbit_fixture_2933(
    alpha: f64,
    orbit_scale: f64,
) -> (SaeSupportSparseTerm, Array2<f64>, Vec<f64>, Vec<Vec<f64>>, f64) {
    let rows = 8usize;
    let base: Vec<f64> = (0..rows)
        .map(|row| if row % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    let slope = (rows as f64).sqrt();
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
    let atoms = vec![atom(
        "scale-orbit",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        evaluator,
        &[0.0],
        array![[0.0], [slope / orbit_scale]],
    )];
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        rows,
        1,
        1,
        vec![SaeAssignmentAtomSpec::euclidean(1)],
        vec![vec![0]; rows],
        vec![vec![1.0]; rows],
        base.iter().map(|&t| vec![orbit_scale * t]).collect(),
    )
    .expect("state");
    let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let target =
        Array2::from_shape_fn((rows, 1), |(row, _)| (slope + alpha / slope) * base[row]);
    (term, target, vec![alpha], vec![vec![alpha]], slope)
}

/// #2933 F08 — a diagonal-scaled gradient is not the remaining Newton
/// displacement. Five percent along the scale orbit (`α = 1e-6`) the state is
/// `β₁* − β₁*/1.05 ≈ 0.135` from the optimum in the slope and `0.05` in every
/// coordinate, yet the first-order screen passes on its diagonal limb: the
/// gradient is small only because the orbit's curvature is. The exact Newton
/// displacement recovers both distances to first order and refuses. At the
/// stationary point itself it certifies.
#[test]
fn exact_newton_displacement_sees_the_orbit_the_diagonal_misses_2933_f08() {
    let alpha = 1.0e-6;
    let orbit_scale = 1.05;
    let (term, target, lambda, ard, slope) = scale_orbit_fixture_2933(alpha, orbit_scale);
    let tolerance = term.fixed_point_tolerance();
    let parameter_scale = term.parameter_iterate_scale().expect("parameter scale");
    let objective = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("objective");
    let stationarity = term
        .raw_stationarity(target.view(), &lambda, &ard)
        .expect("stationarity");
    assert!(
        stationarity.scaled_max_abs() <= tolerance * parameter_scale
            && stationarity.first_order_screen(
                objective.abs().max(1.0),
                parameter_scale,
                tolerance
            ),
        "the fixture must hide the orbit from the diagonal: scaled {:.3e} vs bound {:.3e}",
        stationarity.scaled_max_abs(),
        tolerance * parameter_scale,
    );
    let (displacement, _) = term
        .exact_newton_solve(target.view(), &lambda, &ard)
        .expect("exact Newton displacement");
    let slope_distance = slope - slope / orbit_scale;
    let coordinate_distance = orbit_scale - 1.0;
    assert!(
        displacement.decoder_max_abs >= 0.5 * slope_distance
            && displacement.decoder_max_abs <= 2.0 * slope_distance,
        "decoder displacement {:.6e} must recover the slope distance {slope_distance:.6e}",
        displacement.decoder_max_abs,
    );
    assert!(
        displacement.coordinate_max_abs >= 0.5 * coordinate_distance
            && displacement.coordinate_max_abs <= 2.0 * coordinate_distance,
        "coordinate displacement {:.6e} must recover the coordinate distance \
         {coordinate_distance:.6e}",
        displacement.coordinate_max_abs,
    );
    assert!(!displacement.certifies(parameter_scale, tolerance));

    let (optimum, target, lambda, ard, _) = scale_orbit_fixture_2933(alpha, 1.0);
    let optimum_scale = optimum.parameter_iterate_scale().expect("parameter scale");
    let (at_optimum, _) = optimum
        .exact_newton_solve(target.view(), &lambda, &ard)
        .expect("exact Newton displacement at the optimum");
    assert!(
        at_optimum.certifies(optimum_scale, tolerance),
        "the stationary point must certify: displacement {:.3e} vs bound {:.3e}",
        at_optimum.max_abs(),
        tolerance * optimum_scale,
    );
}

/// #2933 F08 — `solve_fixed_point` returns only a state whose exact Newton
/// displacement is within tolerance, and that state is the optimum. Started five
/// percent along the scale orbit with `α = 1e-7`, the alternation moves along the
/// orbit by about `α/2` of its remaining distance per cycle. So the objective
/// recurrence, the state recurrence and the diagonal-scaled screen all pass
/// while the slope is still `0.135` from `β₁* = √8`. The analytic optimum, not
/// the certificate's own number, is the oracle.
#[test]
fn fixed_point_certifies_the_orbit_optimum_not_its_crawl_2933_f08() {
    let alpha = 1.0e-7;
    let (mut term, target, lambda, ard, slope) = scale_orbit_fixture_2933(alpha, 1.05);
    let tolerance = term.fixed_point_tolerance();
    let report = term
        .solve_fixed_point(target.view(), &lambda, &ard, tolerance, 1.0)
        .expect("the orbit optimum is certifiable");
    assert!(report.recurred);
    let parameter_scale = term.parameter_iterate_scale().expect("parameter scale");
    assert!(
        report.newton_displacement.certifies(parameter_scale, tolerance),
        "returned displacement {:.3e} vs bound {:.3e}",
        report.newton_displacement.max_abs(),
        tolerance * parameter_scale,
    );
    let fitted_slope = term.atoms[0].decoder_coefficients()[[1, 0]];
    assert!(
        (fitted_slope - slope).abs() <= 1.0e-4 * slope,
        "certified slope {fitted_slope:.9e} must be the orbit optimum {slope:.9e}",
    );
    for row in 0..term.n_obs() {
        let expected = if row % 2 == 0 { -1.0 } else { 1.0 };
        let coordinate = term.assignment.coords_for_slot(row, 0)[0];
        assert!(
            (coordinate - expected).abs() <= 1.0e-4,
            "row {row}: certified coordinate {coordinate:.9e} must be the orbit optimum \
             {expected}",
        );
    }
}

/// `support_outer`'s two-row fixture: a periodic harmonic atom and a degree-1 plane
/// patch, one row each, `P = 1`, with a residual that does not vanish.
fn support_outer_fixture_2933() -> (SaeSupportSparseTerm, Array2<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let periodic_eval: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
    let patch_eval: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
    let atoms = vec![
        atom(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            periodic_eval,
            &[0.3],
            array![[0.2], [1.1], [-0.4]],
        ),
        atom(
            "plane",
            SaeAtomBasisKind::Linear,
            2,
            patch_eval,
            &[0.1, -0.2],
            array![[0.3], [2.0], [-1.0]],
        ),
    ];
    let specs = vec![
        SaeAssignmentAtomSpec {
            latent_dim: 1,
            manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
            retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
        },
        SaeAssignmentAtomSpec::euclidean(2),
    ];
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        2,
        2,
        1,
        specs,
        vec![vec![0], vec![1]],
        vec![vec![9.0], vec![-4.0]],
        vec![vec![0.1], vec![3.0, 1.0]],
    )
    .expect("state");
    let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    (
        term,
        array![[1.4], [4.3]],
        vec![0.35, 2.8],
        vec![vec![1.0], vec![1.0, 1.0]],
    )
}

/// #2933 F08 — a support fit driven to round-off certifies. `support_outer`'s
/// objective asks this fixture's fixed point for `1e-9`, below the fixture's
/// resolution `fixed_point_tolerance()` (≈ 2.3e-8 at six cells), so the fixed point
/// runs on to a state whose gradient is round-off. There `‖g − AΔ‖ ≤ √ε‖g‖` is
/// unreachable: with the floor removed, `support_penalized_deviance_derivative_
/// equals_penalty_energy` (same fixture and tolerance) refused at ‖g‖ 3.2e-15 against
/// a rounding band of 1.7e-15, relative residual 3.3e-3 (job 1117261). Only the
/// rounding floor `β_g + γ_dim·‖A‖·‖Δ‖` certifies that state. At
/// `fixed_point_tolerance()` the certificate already passed above round-off, so this
/// test passed with the floor removed and did not pin it (same job).
#[test]
fn fixed_point_certifies_a_state_converged_to_roundoff_2933_f08() {
    let (mut term, target, lambda, ard) = support_outer_fixture_2933();
    // `support_outer`'s inner tolerance, which drives this fixture to round-off.
    let tolerance = 1.0e-9;
    assert!(
        tolerance < term.fixed_point_tolerance(),
        "the control must ask for less than the fixture resolves, or it never reaches round-off",
    );
    let report = term
        .solve_fixed_point(target.view(), &lambda, &ard, tolerance, 1.0)
        .expect("a support fit converged to round-off must certify");
    assert!(report.recurred);
    let parameter_scale = term.parameter_iterate_scale().expect("parameter scale");
    assert!(
        report.newton_displacement.certifies(parameter_scale, tolerance),
        "returned displacement {:.3e} vs bound {:.3e}",
        report.newton_displacement.max_abs(),
        tolerance * parameter_scale,
    );
}

/// #2576 — no cycle count ends the support fixed point: a state no cycle can move is
/// refused where it is reached. At the scale-orbit fixture's analytic optimum (`α = 1`),
/// asked for a tolerance below anything f64 resolves, the sweeps change nothing the
/// objective's rounding band or the gradients' rounding bands can see, the first-order
/// screen cannot pass, and the coupled step finds no measured decrease. Cycle 1 has no
/// previous state to compare with and cycle 2 no previous gradient band, so cycle 3 is
/// the first the progress rule can judge, and the refusal must come there. Under the
/// retired cycle budget the same state cycled to `2·max_iter` and refused as a
/// non-recurrence.
#[test]
fn a_state_no_cycle_moves_refuses_as_a_proven_stall_2576() {
    let (mut term, target, lambda, ard, _) = scale_orbit_fixture_2933(1.0, 1.0);
    let error = term
        .solve_fixed_point(target.view(), &lambda, &ard, f64::MIN_POSITIVE, 1.0)
        .expect_err("a tolerance below the arithmetic's resolution cannot certify");
    assert!(
        error.contains("stalled at cycle 3:"),
        "the refusal must be the proven stall at the first cycle the rule can judge; \
         got: {error}"
    );
}

/// #2576 — no cycle count phases or stops the support fixed point, so a solve that
/// needs more cycles than any budget would have allowed still certifies on its own
/// trajectory. The scale-orbit fixture (`α = 1`) starts at orbit scale 2: every row's
/// coordinate sits at `|t| = 2` and the optimum is `|t| = 1`. A trust radius of `r`
/// moves a row by at most `r` per cycle, so no alternating trajectory can certify in
/// fewer than `1/r` cycles, and every cycle it takes lowers the objective measurably.
/// A count that stops the alternation or hands it to the coupled step first cannot
/// return this trajectory: with the retired 256-cycle budget the solve ended its
/// alternation at cycle 256.
#[test]
fn fixed_point_certifies_a_crawl_past_any_cycle_budget_2576() {
    let (mut term, target, lambda, ard, slope) = scale_orbit_fixture_2933(1.0, 2.0);
    let trust_radius = 1.0e-3;
    let tolerance = term.fixed_point_tolerance();
    let report = term
        .solve_fixed_point(target.view(), &lambda, &ard, tolerance, trust_radius)
        .expect("a crawl that lowers the objective every cycle must run to its certificate");
    assert!(report.recurred);
    let minimum_cycles = ((2.0 - 1.0) / trust_radius) as usize;
    assert!(
        report.iterations >= minimum_cycles,
        "a row moving at most {trust_radius:.1e} per cycle cannot travel from |t| = 2 to 1 \
         in {} cycles",
        report.iterations
    );
    let fitted_slope = term.atoms[0].decoder_coefficients()[[1, 0]];
    assert!(
        (fitted_slope - slope).abs() <= 1.0e-4 * slope,
        "certified slope {fitted_slope:.9e} must be the orbit optimum {slope:.9e}",
    );
}

/// #2576 — certification never skips the curvature audit. One degree-1 patch atom
/// `f(t) = β₀ + β₁t` (penalty `S = I`, `λ = 1`, ARD precision `α = 1`) with a zero
/// decoder sits at `t = 0` on each of `n = 512` rows, against the balanced target
/// `y = ±1`. Every gradient is exactly zero: the residual is `y`, and `Σy = 0`, `t = 0`
/// and `β₁ = 0` zero the decoder, coordinate and prior terms. Every block's own curvature
/// is positive (`α` per coordinate, `n + λ` and `λ` on the decoder), so no sweep moves the
/// state, and the first-order screen and the exact Newton displacement both certify it.
/// Only the coupled curvature sees the saddle: the data term couples each coordinate to
/// the slope by `−y`, so the slope's Schur complement is `λ − Σy²/α = 1 − n < 0`. The
/// only other stationary points are the two minima `β₁ = ±√(√n − 1)`,
/// `t = β₁y/(β₁² + α)`, `β₀ = 0`.
///
/// The pencil has `n + 2 = 514` directions, more than the `2·256 = 512` cycles the
/// retired budget bought, so an audit admitted by that count was skipped and the saddle
/// was certified. Admitted by the in-core ledger alone, the audit runs, the solve
/// escapes, and the state it certifies is the analytic minimum. The planted state's own
/// resolved negative mode is the control that the audit is what separates them.
#[test]
fn certification_audits_curvature_at_any_cycle_count_2576() {
    let rows = 512usize;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
    let atoms = vec![atom(
        "bilinear-saddle",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        evaluator,
        &[0.0],
        array![[0.0], [0.0]],
    )];
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        rows,
        1,
        1,
        vec![SaeAssignmentAtomSpec::euclidean(1)],
        vec![vec![0]; rows],
        vec![vec![1.0]; rows],
        vec![vec![0.0]; rows],
    )
    .expect("state");
    let mut term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let target =
        Array2::from_shape_fn((rows, 1), |(row, _)| if row % 2 == 0 { -1.0 } else { 1.0 });
    let lambda = vec![1.0_f64];
    let ard = vec![vec![1.0_f64]];
    let (beta_offsets, beta_dim) = term.beta_layout().expect("beta layout");
    // The retired budget bought `2 · SAE_SUPPORT_INNER_FIXED_POINT_MAX_ITER` = 512 directions.
    assert!(term.coordinate_state_len() + beta_dim > 2 * 256);
    let negative_mode = |term: &SaeSupportSparseTerm| {
        let system = term
            .assemble_arrow_schur(target.view(), &lambda, &ard)
            .expect("arrow system");
        let differential = term
            .support_outer_differential_rows(target.view(), &ard, &beta_offsets)
            .expect("exact differential rows");
        term.support_outer_negative_curvature_mode(&system, &differential)
            .expect("curvature classification")
    };
    let planted = negative_mode(&term).expect("the planted state carries a resolved negative mode");
    assert!(
        planted.curvature < -f64::EPSILON.sqrt(),
        "planted generalized curvature must be resolved negative, got {}",
        planted.curvature
    );
    let saddle_objective = term
        .penalized_objective(target.view(), &lambda, &ard)
        .expect("saddle objective");
    let tolerance = term.fixed_point_tolerance();
    let report = term
        .solve_fixed_point(target.view(), &lambda, &ard, tolerance, 1.0)
        .expect("the audited solve escapes the saddle and certifies a minimum");
    assert!(report.recurred && report.objective < saddle_objective);
    assert!(
        negative_mode(&term).is_none(),
        "the certified state must carry no resolved negative curvature"
    );
    let minimum_slope = ((rows as f64).sqrt() - 1.0).sqrt();
    let decoder = term.atoms[0].decoder_coefficients();
    assert!(
        (decoder[[1, 0]].abs() - minimum_slope).abs() <= 1.0e-4 * minimum_slope
            && decoder[[0, 0]].abs() <= 1.0e-4 * minimum_slope,
        "certified decoder ({:.9e}, {:.9e}) must be a minimum (0, ±{minimum_slope:.9e})",
        decoder[[0, 0]],
        decoder[[1, 0]],
    );
}

/// #2933 F08 — the round-off floor does not certify a displaced state. At the
/// analytic optimum of the scale-orbit fixture with `α = 1` the displacement
/// certifies. Moving the slope by `offset` puts the state, to first order, `offset`
/// from its root in the slope alone, so the decoder displacement must read back
/// `offset` there and the certificate must refuse. The planted offset is the oracle.
#[test]
fn planted_decoder_offset_refuses_under_the_roundoff_floor_2933_f08() {
    let (mut term, target, lambda, ard, _) = scale_orbit_fixture_2933(1.0, 1.0);
    let tolerance = term.fixed_point_tolerance();
    let optimum_scale = term.parameter_iterate_scale().expect("parameter scale");
    let (at_optimum, _) = term
        .exact_newton_solve(target.view(), &lambda, &ard)
        .expect("exact Newton displacement at the optimum");
    assert!(
        at_optimum.certifies(optimum_scale, tolerance),
        "the optimum must certify: displacement {:.3e} vs bound {:.3e}",
        at_optimum.max_abs(),
        tolerance * optimum_scale,
    );

    let offset = 1.0e-3;
    let mut decoder = term.atoms[0].decoder_coefficients().clone();
    decoder[[1, 0]] += offset;
    term.atoms[0]
        .set_decoder_coefficients(decoder)
        .expect("plant the slope offset");
    let displaced_scale = term.parameter_iterate_scale().expect("parameter scale");
    let (displaced, _) = term
        .exact_newton_solve(target.view(), &lambda, &ard)
        .expect("exact Newton displacement at the displaced state");
    let (beta_offsets, _) = term.beta_layout().expect("beta layout");
    // `offset(atom) + basis · P + channel` for basis 1, channel 0.
    let slope_index = beta_offsets[0] + term.output_dim;
    assert!(
        (displaced.decoder[slope_index] - offset).abs() <= 0.5 * offset,
        "the slope displacement {:.6e} must read back the planted offset {offset:.3e}",
        displaced.decoder[slope_index],
    );
    assert!(
        displaced.max_abs() <= 2.0 * offset,
        "no parameter may be displaced beyond the planted offset: {:.6e}",
        displaced.max_abs(),
    );
    assert!(
        !displaced.certifies(displaced_scale, tolerance),
        "a planted offset {offset:.3e} must not certify against bound {:.3e}",
        tolerance * displaced_scale,
    );
}

/// #2576 — the dense stationarity pencil read off the arrow's blocks is the column-probe
/// pencil bit for bit. Four rows each carry a periodic atom and a plane patch, so every
/// row has two slots and the cross-atom `H_ββ` and cross-slot `H_tt` entries exist.
/// `P = 2`, the periodic axis has the von-Mises prior, and the residual does not vanish,
/// so both blocks of exact corrections are populated. The oracle is the probe build that
/// `support_outer_dense_hessian_matrices` replaced.
#[test]
fn dense_pencil_assembly_is_its_column_probes_2576() {
    let periodic_eval: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
    let patch_eval: Arc<dyn SaeBasisSecondJet> =
        Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
    let atoms = vec![
        atom(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            periodic_eval,
            &[0.3],
            array![[0.2, -0.7], [1.1, 0.4], [-0.4, 0.9]],
        ),
        atom(
            "plane",
            SaeAtomBasisKind::Linear,
            2,
            patch_eval,
            &[0.1, -0.2],
            array![[0.3, 1.2], [2.0, -0.6], [-1.0, 0.5]],
        ),
    ];
    let specs = vec![
        SaeAssignmentAtomSpec {
            latent_dim: 1,
            manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
            retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
        },
        SaeAssignmentAtomSpec::euclidean(2),
    ];
    let rows = 4usize;
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        rows,
        2,
        2,
        specs,
        vec![vec![0, 1]; rows],
        vec![vec![1.3, -0.8], vec![0.6, 2.1], vec![-1.7, 0.9], vec![2.4, -0.3]],
        vec![
            vec![0.1, 3.0, 1.0],
            vec![0.37, -0.5, 0.8],
            vec![0.62, 1.4, -2.2],
            vec![0.85, 0.2, 0.6],
        ],
    )
    .expect("state");
    let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let target = array![[1.4, -0.3], [4.3, 0.8], [-2.1, 1.7], [0.6, -1.1]];
    let lambda = vec![0.35, 2.8];
    let ard = vec![vec![1.0], vec![1.0, 0.5]];
    let system = term
        .assemble_arrow_schur(target.view(), &lambda, &ard)
        .expect("arrow system");
    let (beta_offsets, beta_len) = term.beta_layout().expect("beta layout");
    let differential = term
        .support_outer_differential_rows(target.view(), &ard, &beta_offsets)
        .expect("exact differential rows");
    let t_len = *system.row_offsets.last().unwrap_or(&0);
    assert_eq!((t_len, beta_len), (12, 12));
    let (exact, majorizer) = term
        .support_outer_dense_hessian_matrices(&system, &differential, t_len, beta_len)
        .expect("assembled pencil");
    let (probed_exact, probed_majorizer) =
        dense_pencil_by_column_probes(&term, &system, &differential, t_len, beta_len);
    assert_eq!(majorizer, probed_majorizer, "the assembled majorizer must be the probed one");
    assert_eq!(exact, probed_exact, "the assembled exact Hessian must be the probed one");
    // Both correction blocks are populated, or the equality above could not see one of
    // them missing.
    let largest_gap = |block: ndarray::ArrayView2<'_, f64>, reference: ndarray::ArrayView2<'_, f64>| {
        block
            .iter()
            .zip(reference.iter())
            .fold(0.0_f64, |largest, (left, right)| largest.max((left - right).abs()))
    };
    let coordinate_gap = largest_gap(
        exact.slice(ndarray::s![..t_len, ..t_len]),
        majorizer.slice(ndarray::s![..t_len, ..t_len]),
    );
    let cross_gap = largest_gap(
        exact.slice(ndarray::s![..t_len, t_len..]),
        majorizer.slice(ndarray::s![..t_len, t_len..]),
    );
    assert!(
        coordinate_gap > 0.0 && cross_gap > 0.0,
        "the exact corrections must be populated: coordinate block {coordinate_gap:.3e}, \
         cross block {cross_gap:.3e}"
    );
}

/// The column-probe build `support_outer_dense_hessian_matrices` replaced (#2576): both
/// operators applied to every unit vector, then symmetrized. The oracle of
/// `dense_pencil_assembly_is_its_column_probes_2576`.
fn dense_pencil_by_column_probes(
    term: &SaeSupportSparseTerm,
    system: &ArrowSchurSystem,
    rows: &[SupportOuterDifferentialRow],
    t_len: usize,
    beta_len: usize,
) -> (Array2<f64>, Array2<f64>) {
    let dim = t_len + beta_len;
    let mut exact = Array2::<f64>::zeros((dim, dim));
    let mut majorizer = Array2::<f64>::zeros((dim, dim));
    for column in 0..dim {
        let mut unit = SaeArrowVector {
            t: Array1::zeros(t_len),
            beta: Array1::zeros(beta_len),
        };
        if column < t_len {
            unit.t[column] = 1.0;
        } else {
            unit.beta[column - t_len] = 1.0;
        }
        let applied = term
            .support_outer_exact_hessian_apply(system, rows, &unit)
            .expect("exact Hessian apply");
        exact
            .slice_mut(ndarray::s![..t_len, column])
            .assign(&applied.t);
        exact
            .slice_mut(ndarray::s![t_len.., column])
            .assign(&applied.beta);
        let applied_b = support_arrow_majorizer_apply(system, &unit).expect("majorizer apply");
        majorizer
            .slice_mut(ndarray::s![..t_len, column])
            .assign(&applied_b.t);
        majorizer
            .slice_mut(ndarray::s![t_len.., column])
            .assign(&applied_b.beta);
    }
    for row in 0..dim {
        for column in 0..row {
            let symmetric = 0.5 * (exact[[row, column]] + exact[[column, row]]);
            exact[[row, column]] = symmetric;
            exact[[column, row]] = symmetric;
            let symmetric_b = 0.5 * (majorizer[[row, column]] + majorizer[[column, row]]);
            majorizer[[row, column]] = symmetric_b;
            majorizer[[column, row]] = symmetric_b;
        }
    }
    (exact, majorizer)
}

/// #2933 F08 — the exact Newton displacement solves at a row the majorizer does not
/// curve. At `t = 1/2 + 1e-8` the von-Mises prior's PSD clamp is 0 and a cosine
/// decoder's tangent is about `4π²·1e-8`, so the row's majorizer block is about
/// 1.6e-13, while the exact block carries the residual curvature `−r·f'' = 4π²` less
/// the prior's `1`. Preconditioned by `B⁻¹`, such a row was amplified by the
/// reciprocal of its majorizer block, and flexible GMRES could not certify: the
/// two-circle Tier-2 witness had `B = 5.8e-14` against `A = 12.4` (lane probe
/// 1249076). The exact-A preconditioner's row block `B_i + (A_i − B_i)_+` is `A_i`
/// there.
#[test]
fn exact_newton_displacement_solves_at_a_row_the_majorizer_does_not_curve_2933_f08() {
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
    let decoder = array![[0.2], [0.0], [1.0]];
    let coordinates = [0.5 + 1.0e-8, 0.1, 0.3];
    let residuals = [-1.0, 0.3, -0.2];
    let n = coordinates.len();
    let atoms = vec![atom(
        "flat-tangent",
        SaeAtomBasisKind::Periodic,
        1,
        evaluator,
        &[coordinates[0]],
        decoder.clone(),
    )];
    let specs = vec![SaeAssignmentAtomSpec {
        latent_dim: 1,
        manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
        retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
    }];
    let state = SaeAssignmentState::from_topk_support_heterogeneous(
        n,
        1,
        1,
        specs,
        vec![vec![0]; n],
        vec![vec![1.0]; n],
        coordinates.iter().map(|&t| vec![t]).collect(),
    )
    .expect("state");
    let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
    let target = Array2::from_shape_fn((n, 1), |(row, _)| {
        let phase = std::f64::consts::TAU * coordinates[row];
        decoder[[0, 0]] + decoder[[1, 0]] * phase.sin() + decoder[[2, 0]] * phase.cos()
            + residuals[row]
    });
    let lambda = vec![1.0_f64];
    let ard = vec![vec![1.0_f64]];
    let system = term
        .assemble_arrow_schur(target.view(), &lambda, &ard)
        .expect("arrow system");
    let (beta_offsets, beta_dim) = term.beta_layout().expect("beta layout");
    let rows = term
        .support_outer_differential_rows(target.view(), &ard, &beta_offsets)
        .expect("exact differential rows");
    let coordinate_dim = *system.row_offsets.last().unwrap_or(&0);
    let (exact, majorizer) = term
        .support_outer_dense_hessian_matrices(&system, &rows, coordinate_dim, beta_dim)
        .expect("dense pencil");
    let preconditioner = term
        .support_exact_a_preconditioner_rows(&system, &rows)
        .expect("preconditioner rows");
    let solved = term.exact_newton_solve(target.view(), &lambda, &ard);
    println!(
        "[#2933 F08 flat tangent] majorizer {:.3e}, exact {:.3e}, preconditioner {:.3e}; \
         solve {:?}",
        majorizer[[0, 0]],
        exact[[0, 0]],
        preconditioner[0].htt[[0, 0]],
        solved.as_ref().map(|(displacement, _)| displacement.max_abs()),
    );
    assert!(
        majorizer[[0, 0]] <= f64::EPSILON.sqrt() * exact[[0, 0]],
        "the fixture's row must be one the majorizer does not curve: B {:.3e} vs A {:.3e}",
        majorizer[[0, 0]],
        exact[[0, 0]],
    );
    assert!(
        (preconditioner[0].htt[[0, 0]] - exact[[0, 0]]).abs() <= 1.0e-12 * exact[[0, 0]],
        "the preconditioner's row block must be the exact block where it exceeds the \
         majorizer: {:.12e} vs {:.12e}",
        preconditioner[0].htt[[0, 0]],
        exact[[0, 0]],
    );
    let (displacement, _) = solved
        .expect("the exact Newton displacement certifies at a row the majorizer does not curve");
    assert!(displacement.max_abs().is_finite());
}
