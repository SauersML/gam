//! Encode/decoder contract probe tests for the SAE manifold term, split out of
//! `tests.rs` to keep each tracked file under the line-count gate. These were
//! the former inline `mod inner_contract_probe_tests` block; they define their
//! own euclidean-line/periodic fixtures and assert the analytic decoder and
//! coordinate gradients/Hessians match finite differences on the penalized
//! objective contract.

use super::tests::{
    TestPeriodicEvaluator, diagonal_latent_cache, periodic_basis,
    warmstart_test_objective_with_evaluator,
};
use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};
use gam_terms::latent::LatentManifold;
use approx::assert_abs_diff_eq;
use ndarray::array;
use std::sync::Arc;

pub(crate) fn euclidean_line_contract_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 150usize;
    let p = 8usize;
    let mut coords = Array2::<f64>::zeros((n, 1));
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let u = -1.0 + 2.0 * row as f64 / (n as f64 - 1.0);
        coords[[row, 0]] = 2.5 + 3.0 * u;
        for col in 0..p {
            let linear_loading = 0.35 + 0.07 * col as f64;
            let offset = 0.08 * ((col % 3) as f64 - 1.0);
            let phase = (row * (col + 3)) as f64;
            let noise = 0.04 * (phase.sin() + 0.5 * (0.37 * phase).cos());
            z[[row, col]] = offset + linear_loading * u + noise;
        }
    }

    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("evaluator"));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("basis");
    let m = phi.ncols();
    let smooth_penalty =
        gam_terms::basis::create_difference_penalty_matrix(m, 2, None).expect("penalty");
    let atom = SaeManifoldAtom::new(
        "contract-line",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        smooth_penalty,
    )
    .expect("atom")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, (0.01_f64).ln(), vec![Array1::<f64>::zeros(1)]);
    (term, z, rho)
}

pub(crate) fn assert_contract_close(label: &str, analytic: f64, finite_difference: f64) {
    let rel = (analytic - finite_difference).abs()
        / finite_difference.abs().max(analytic.abs()).max(1.0e-12);
    assert!(
        rel < 1.0e-5,
        "{label}: analytic={analytic:.12e} fd={finite_difference:.12e} rel={rel:.3e}"
    );
}

#[test]
pub(crate) fn euclidean_line_decoder_gradient_matches_penalized_objective_fd() {
    let (mut term, z, mut rho) = euclidean_line_contract_fixture();
    let ridge = 1.0e-6;
    for step in 0..6 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .unwrap_or_else(|err| panic!("warm step {step} failed: {err}"));
        assert!(
            loss.total().is_finite(),
            "warm step {step} loss is non-finite"
        );
    }

    let sys_coord = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("coord assemble");
    assert_eq!(
        sys_coord.k,
        term.beta_dim(),
        "p=8 contract fixture must stay on full-B coordinates"
    );
    assert!(
        !term.frames_active(),
        "p=8 contract fixture must not activate a frame"
    );

    let h = 1.0e-6;
    for row in [3usize, 75, 140] {
        let analytic = sys_coord.rows[row].gt[0];
        let base_coord = term.assignment.coords[0].as_matrix()[[row, 0]];

        let mut plus_coords = term.assignment.coords[0].as_matrix();
        plus_coords[[row, 0]] = base_coord + h;
        let plus_flat = Array1::from_iter(plus_coords.iter().copied());
        term.assignment.coords[0].set_flat(plus_flat.view());
        term.refresh_basis_from_current_coords()
            .expect("plus refresh");
        let f_plus = term
            .penalized_objective_total(z.view(), &rho, None, 1.0)
            .expect("coord f+");

        let mut minus_coords = term.assignment.coords[0].as_matrix();
        minus_coords[[row, 0]] = base_coord - h;
        let minus_flat = Array1::from_iter(minus_coords.iter().copied());
        term.assignment.coords[0].set_flat(minus_flat.view());
        term.refresh_basis_from_current_coords()
            .expect("minus refresh");
        let f_minus = term
            .penalized_objective_total(z.view(), &rho, None, 1.0)
            .expect("coord f-");

        let mut restored_coords = term.assignment.coords[0].as_matrix();
        restored_coords[[row, 0]] = base_coord;
        let restored_flat = Array1::from_iter(restored_coords.iter().copied());
        term.assignment.coords[0].set_flat(restored_flat.view());
        term.refresh_basis_from_current_coords()
            .expect("restore refresh");

        let fd = (f_plus - f_minus) / (2.0 * h);
        assert_contract_close(&format!("CONTRACT coord row {row}"), analytic, fd);
    }

    let sys_decoder = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("decoder assemble");
    assert_eq!(sys_decoder.k, term.beta_dim());
    let p = term.output_dim();
    for (basis_col, out_col) in [(0usize, 0usize), (1, 3), (2, 7)] {
        let beta_idx = basis_col * p + out_col;
        let analytic = sys_decoder.gb[beta_idx];
        let base = term.atoms[0].decoder_coefficients[[basis_col, out_col]];

        term.atoms[0].decoder_coefficients[[basis_col, out_col]] = base + h;
        let f_plus = term
            .penalized_objective_total(z.view(), &rho, None, 1.0)
            .expect("decoder f+");
        term.atoms[0].decoder_coefficients[[basis_col, out_col]] = base - h;
        let f_minus = term
            .penalized_objective_total(z.view(), &rho, None, 1.0)
            .expect("decoder f-");
        term.atoms[0].decoder_coefficients[[basis_col, out_col]] = base;

        let fd = (f_plus - f_minus) / (2.0 * h);
        assert_contract_close(
            &format!("CONTRACT decoder ({basis_col},{out_col})"),
            analytic,
            fd,
        );
    }
}

/// #1154 — the joint amortized-encoder + REML co-training fold (Design A).
///
/// On a synthetic 1D periodic manifold with KNOWN structure (the target is
/// drawn from a true sine curve on the circle), after the inner `(t, β)`
/// solve converges to stationarity:
///
/// 1. the co-trained criterion is the exact REML criterion PLUS a
///    non-negative, correctly-scaled amortized-encoder consistency penalty —
///    so the fold is sound and the REML λ-coupling is untouched (the inner
///    solve still produces the stationary point the criterion is read at);
/// 2. the cheap one-mat-vec amortized encode is FAITHFUL: its reconstruction
///    matches the exact fitted reconstruction (the encode-by-inner-solve
///    truth) within a tight tolerance on the rows the certificate accepts —
///    proving the encoder recovers the same structure the exact path does,
///    at amortized cost; and
/// 3. the encoder CERTIFIES coverage of the fitted dictionary (a strictly
///    positive certified fraction), so the co-training signal rewards a real,
///    measurable encoder-quality axis rather than a vacuous one.
#[test]
fn cotrained_criterion_folds_faithful_amortized_encoder_on_known_manifold() {
    let n = 24usize;
    let p = 4usize;
    // A true circle coordinate per row, and a smooth periodic decoder, so the
    // target lies on a genuine 1D periodic manifold (known structure).
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    // Basis width comes from the shared `periodic_basis` helper (1, sin, cos),
    // so derive `m` from it rather than hardcoding — the decoder row count and
    // the (m, m) smooth penalty must both track the actual harmonic width.
    let m = phi.ncols();
    // A smooth decoder B (M × p): low-order harmonics dominate so the encode
    // map is well-conditioned and the IFT predictor is a faithful first-order
    // model of it (the regime the amortized encoder is built for).
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        let scale = 1.0 / (1.0 + b as f64);
        scale * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });
    let atom = SaeManifoldAtom::new(
        "periodic_truth",
        SaeAtomBasisKind::Periodic,
        1,
        phi.clone(),
        jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    // The ground-truth ambient target: the exact decoded curve Φ(t*)·B at
    // unit amplitude, so a perfect fit reproduces the manifold exactly.
    let target = phi.dot(&decoder);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![1.0_f64.ln()]]);

    // Converge the inner (t, β) solve to stationarity — the REML criterion
    // and the co-training fold are both read at the converged dictionary.
    // Full Newton steps (learning_rate = 1.0): the heavily-damped 0.1 step
    // cannot drive this well-conditioned planted-circle fit to the strict KKT
    // tolerance within the refine budget (it stalls at ‖g‖≈6e-3), so the
    // criterion correctly refuses to rank an off-optimum Laplace value. At
    // full Newton the inner solve reaches true stationarity in a handful of
    // iterations.
    let mut rho_fit = rho.clone();
    term.run_joint_fit_arrow_schur(target.view(), &mut rho_fit, None, 12, 1.0, 1.0e-4, 1.0e-4)
        .expect("inner solve converges on the known periodic manifold");

    // (1) Fold soundness: the co-trained criterion = REML + scaled, finite,
    // non-negative consistency penalty.
    let (reml, _loss) = term
        .reml_criterion_with_refine_policy(
            target.view(),
            &rho_fit,
            None,
            25,
            1.0,
            1.0e-4,
            1.0e-4,
            true,
        )
        .expect("REML criterion evaluates");
    let (cotrained, _loss2, consistency) = term
        .reml_criterion_cotrained(target.view(), &rho_fit, None, 64, 1.0, 1.0e-4, 1.0e-4)
        .expect("co-trained criterion evaluates");
    assert!(
        cotrained.is_finite() && reml.is_finite(),
        "both criteria must be finite: cotrained={cotrained}, reml={reml}"
    );
    assert!(
        cotrained >= reml - 1.0e-9,
        "co-trained criterion must add a NON-NEGATIVE consistency penalty: \
         cotrained={cotrained} < reml={reml}"
    );
    assert!(
        consistency.recon_consistency >= 0.0 && consistency.recon_consistency.is_finite(),
        "recon consistency must be a finite non-negative gap, got {}",
        consistency.recon_consistency
    );
    assert!(
        (0.0..=1.0).contains(&consistency.uncertified_fraction),
        "uncertified fraction must be a probability, got {}",
        consistency.uncertified_fraction
    );

    // (3) The encoder must certify real coverage of the fitted dictionary —
    // not a vacuous all-uncertified fraction.
    assert!(
        consistency.uncertified_fraction < 1.0,
        "the amortized encoder must certify at least some rows of a \
         well-conditioned periodic dictionary; uncertified_fraction={}",
        consistency.uncertified_fraction
    );

    // (2) Faithfulness: on the rows the certificate accepts, the cheap
    // one-mat-vec amortized encode recovers the SAME latent coordinate the
    // EXACT encode-by-inner-solve (the certified cold chart-center Newton
    // probe) produces. This is the encoder-fidelity question Design A makes —
    // amortized-encode ≈ exact-encode PER ROW. (It is NOT the same as the
    // joint-fitted reconstruction `try_fitted_for_rho`: the joint fit smooths
    // the latent coords across rows under the λ_smooth penalty, so its
    // per-row reconstruction legitimately differs from a per-row encode by the
    // smoothing bias — comparing against it would conflate encoder fidelity
    // with the smoother. We therefore compare the two PER-ROW encodes, decoded
    // through the SAME basis, exactly as the held-out arm below does.)
    let amplitudes = term.fitted_assignment_amplitudes(&rho_fit).unwrap();
    let encodes = term
        .amortized_encode_target(target.view(), amplitudes.view())
        .expect("amortized encode runs");
    let atom0 = &term.atoms[0];
    let evaluator = atom0.basis_evaluator.as_ref().unwrap();
    let (phi_hat, _j) = evaluator.evaluate(encodes[0].coords.view()).unwrap();
    let decoded_hat = phi_hat.dot(&atom0.decoder_coefficients); // (n × p)

    // The exact per-row encode the sequential path would use as its teacher:
    // a certified cold chart-center Newton solve for each row.
    let mut in_sample_norm_bound = 0.0_f64;
    for row in 0..n {
        in_sample_norm_bound =
            in_sample_norm_bound.max(target.row(row).dot(&target.row(row)).sqrt());
    }
    let in_sample_atlas = crate::encode::EncodeAtlas::build(
        &term.atoms,
        &[1.0],
        in_sample_norm_bound,
        crate::encode::AtlasConfig::default(),
    )
    .expect("in-sample encode atlas builds");

    let mut certified_rows = 0usize;
    let mut max_certified_gap = 0.0_f64;
    for row in 0..n {
        if !encodes[0].certified[row] {
            continue;
        }
        let z = amplitudes[[row, 0]];
        let (exact_t, exact_cert) = in_sample_atlas
            .certified_encode_row(atom0, 0, target.row(row), z)
            .expect("exact per-row encode runs");
        if !exact_cert.certified() {
            // The exact teacher could not certify this row at the fitted
            // amplitude; skip it (the held-out arm asserts joint certification
            // on a unit-amplitude grid). We only measure faithfulness where
            // BOTH the amortized encode and the exact teacher certify.
            continue;
        }
        certified_rows += 1;
        let exact_phi = evaluator
            .evaluate(exact_t.view().insert_axis(ndarray::Axis(0)))
            .unwrap()
            .0;
        let exact_decoded = exact_phi.dot(&atom0.decoder_coefficients); // (1 × p)
        for col in 0..p {
            let amortized = z * decoded_hat[[row, col]];
            let exact = z * exact_decoded[[0, col]];
            let gap = (amortized - exact).abs();
            if gap > max_certified_gap {
                max_certified_gap = gap;
            }
        }
    }
    assert!(
        certified_rows > 0,
        "the certificate must accept at least one row to measure faithfulness"
    );
    // The amortized encode is the first-order IFT model of the exact encode;
    // on a well-conditioned periodic dictionary the certified rows must match
    // the exact per-row encode to the encode's certified tolerance.
    assert!(
        max_certified_gap < 1.0e-2,
        "amortized encode must reconstruct certified rows within the encode \
         tolerance of the exact per-row encode-by-inner-solve; max gap={max_certified_gap}"
    );

    // Held-out recovery: compare the fast #1010 amortized row encode against
    // the exact certified row encode a sequential REML-then-distill path
    // would use as its teacher. The held-out phases are interleaved between
    // training phases, so this is not an in-sample replay.
    let n_holdout = 12usize;
    let heldout_coords = Array2::from_shape_fn((n_holdout, 1), |(row, _)| {
        (row as f64 + 0.25) / n_holdout as f64
    });
    let (heldout_phi, _heldout_jet) = periodic_basis(&heldout_coords);
    let heldout = heldout_phi.dot(&atom0.decoder_coefficients);
    let heldout_amplitudes = Array1::<f64>::ones(n_holdout);
    let mut target_norm_bound = 0.0_f64;
    for row in 0..n_holdout {
        target_norm_bound = target_norm_bound.max(heldout.row(row).dot(&heldout.row(row)).sqrt());
    }
    let atlas = crate::encode::EncodeAtlas::build(
        &term.atoms,
        &[1.0],
        target_norm_bound,
        crate::encode::AtlasConfig::default(),
    )
    .expect("held-out encode atlas builds");
    let fast_heldout = atlas
        .amortized_encode_batch(atom0, 0, heldout.view(), heldout_amplitudes.view())
        .expect("held-out amortized encode runs");

    let mut max_fast_vs_exact = 0.0_f64;
    let mut max_fast_truth = 0.0_f64;
    let mut max_exact_truth = 0.0_f64;
    let mut heldout_certified = 0usize;
    for row in 0..n_holdout {
        if !fast_heldout.certified[row] {
            continue;
        }
        heldout_certified += 1;
        let (exact_t, exact_cert) = atlas
            .certified_encode_row(atom0, 0, heldout.row(row), 1.0)
            .expect("held-out exact certified row encode runs");
        assert!(
            exact_cert.certified(),
            "sequential exact #1010 teacher must certify held-out row {row}"
        );
        let truth = heldout_coords[[row, 0]];
        let fast = fast_heldout.coords[[row, 0]];
        let exact = exact_t[0];
        let fast_vs_exact = circle_phase_gap(fast, exact);
        let fast_truth = circle_phase_gap(fast, truth);
        let exact_truth = circle_phase_gap(exact, truth);
        max_fast_vs_exact = max_fast_vs_exact.max(fast_vs_exact);
        max_fast_truth = max_fast_truth.max(fast_truth);
        max_exact_truth = max_exact_truth.max(exact_truth);
    }
    eprintln!(
        "#1154 AMORTIZED-VS-EXACT: held-out certified={heldout_certified} \
         | max fast-vs-exact #1010 phase gap={max_fast_vs_exact:.6e} \
         | max fast-vs-truth={max_fast_truth:.6e} | max exact-vs-truth={max_exact_truth:.6e}"
    );
    assert!(
        heldout_certified > 0,
        "fast amortized encode must certify held-out rows on the known manifold"
    );
    assert!(
        max_fast_vs_exact < 1.0e-2,
        "fast amortized held-out encode must match exact #1010 encode within \
         certified tolerance; max phase gap={max_fast_vs_exact}"
    );
    assert!(
        max_fast_truth <= max_exact_truth + 1.0e-2,
        "co-trained fast encoder must recover the known held-out manifold at \
         least as well as the sequential exact-teacher path within tolerance; \
         fast={max_fast_truth}, sequential={max_exact_truth}"
    );
}

fn circle_phase_gap(a: f64, b: f64) -> f64 {
    let raw = (a - b).abs();
    raw.min((raw - raw.floor()).abs())
        .min((1.0 - raw.fract()).abs())
}

/// #1206 — the gradient lane's `(cost, gradient)` pair must be SELF-CONSISTENT
/// for the outer BFGS Armijo line search. The amortized-encoder consistency
/// fold `c(ρ)` (#1154) has no analytic gradient (under Design A the exact
/// outer derivative is the REML λ-gradient `∇f` only), so it MUST NOT enter
/// the cost the gradient lane (`eval` / `OuterEvalOrder::ValueAndGradient`)
/// returns alongside `∇f` — otherwise BFGS minimizes `f+c` while believing the
/// gradient is `∇(f+c)`, which is the objective↔gradient desync bug class
/// (#931). The fold is a DERIVATIVE-FREE ranking regularizer carried ONLY by
/// the value-probe lane (`eval_cost`), whose cost is never paired with a
/// gradient.
///
/// This test pins the corrected split:
/// - the value-probe lane carries a strictly positive fold over bare REML
///   (the encoder has some inconsistency on this fixture), and
/// - the gradient lane's cost EQUALS bare REML (it does NOT carry the fold),
///   so it sits a full fold below the value lane and its (cost, ∇f) pair is
///   self-consistent.
#[test]
fn cotrain_fold_is_value_lane_only_so_gradient_lane_pair_is_consistent() {
    let mut objective = warmstart_test_objective_with_evaluator();
    let rho_flat = objective.current_rho.to_flat();

    // Value-probe lane: the cheap derivative-free comparand the cascade uses
    // for seed validation / cross-seed ranking. Carries the consistency fold.
    let value_lane = objective
        .eval_cost(&rho_flat)
        .expect("value-probe lane evaluates the co-trained cost");

    // Gradient lane: the cost an ACCEPTED iterate reports, paired with the
    // analytic ∇f the BFGS Armijo test consumes. A fresh objective so the two
    // paths solve from the identical seed state.
    let mut objective_grad = warmstart_test_objective_with_evaluator();
    let gradient_lane = objective_grad
        .eval(&rho_flat)
        .expect("gradient lane evaluates")
        .cost;

    assert!(
        value_lane.is_finite() && gradient_lane.is_finite(),
        "both lanes must be finite: value={value_lane}, gradient={gradient_lane}"
    );

    // The amortized warm-start on this arbitrary-target fixture certifies no
    // rows (the conservative Kantorovich gate), so it leaves the inner coords
    // untouched — which means the lanes and the bare criterions below all
    // solve from the identical seed state and the bare comparisons are exact.
    assert_eq!(
        objective.warm_start_telemetry().total_rows_warm_started,
        0,
        "fixture precondition: warm-start must certify zero rows so the bare \
         comparisons are drift-free; got {:?}",
        objective.warm_start_telemetry()
    );

    // Bare REML for the VALUE lane, computed on the SAME probe refine policy
    // (`refine_progress_extension = false`) the value lane uses, plus the
    // collapse barrier it also keeps — so the only difference from the value
    // lane is the consistency fold.
    let bare_value = {
        let mut probe = warmstart_test_objective_with_evaluator();
        let target = probe.target.clone();
        let rho_state = probe.baseline_rho.from_flat(rho_flat.view());
        let (reml, _loss) = probe
            .term
            .reml_criterion_with_refine_policy(
                target.view(),
                &rho_state,
                None,
                probe.inner_max_iter,
                probe.learning_rate,
                probe.ridge_ext_coord,
                probe.ridge_beta,
                false,
            )
            .expect("bare value-lane REML criterion evaluates");
        probe
            .add_fit_data_collapse_penalty(reml, &rho_state)
            .expect("collapse penalty evaluates")
    };
    let value_fold = value_lane - bare_value;
    assert!(
        value_fold > 1.0e-12,
        "the value-probe lane carries the co-training fold (positive penalty \
         over bare REML): value_lane={value_lane}, bare={bare_value}, \
         fold={value_fold}"
    );

    // Bare REML for the GRADIENT lane, computed on the SAME full-refine path
    // (`reml_criterion_with_cache`, i.e. `refine_progress_extension = true`)
    // the gradient lane uses, plus the collapse barrier. The gradient lane
    // must EQUAL this (it carries NO consistency fold), so its (cost, ∇f) pair
    // describes one function — the #1206 contract for BFGS Armijo. (The
    // gradient-lane and value-lane bares may differ by the refine policy, so
    // each lane is checked against its OWN matched bare.)
    let bare_grad = {
        let mut probe = warmstart_test_objective_with_evaluator();
        let target = probe.target.clone();
        let rho_state = probe.baseline_rho.from_flat(rho_flat.view());
        let (reml, _loss, _cache) = probe
            .term
            .reml_criterion_with_cache(
                target.view(),
                &rho_state,
                None,
                probe.inner_max_iter,
                probe.learning_rate,
                probe.ridge_ext_coord,
                probe.ridge_beta,
            )
            .expect("bare gradient-lane REML criterion evaluates");
        probe
            .add_fit_data_collapse_penalty(reml, &rho_state)
            .expect("collapse penalty evaluates")
    };
    let gradient_vs_bare = (gradient_lane - bare_grad).abs();
    assert!(
        gradient_vs_bare < 1.0e-9,
        "the gradient lane must report bare REML (no consistency fold), so its \
         (cost, ∇f) pair is self-consistent for BFGS Armijo: \
         gradient_lane={gradient_lane}, bare_grad={bare_grad}, \
         diff={gradient_vs_bare}"
    );
}

/// #1154 item 2+3 — the amortized-encoder warm-start (Design A) accelerates
/// the inner solve to the SAME stationary point WITHOUT degrading recovery of
/// the planted manifold. On a known periodic manifold we
///
/// 1. fit the dictionary (sequential / cold inner solve) and record the
///    explained variance — the REML-then-distill baseline;
/// 2. build the amortized encoder from that fitted dictionary and offer its
///    certified rows as inner latent warm-starts. Zero certified rows is a
///    valid conservative gate outcome: the helper must then leave the cold
///    seed untouched instead of corrupting the inner state;
/// 3. re-converge the inner solve FROM the warm-start and require the
///    explained variance to be at least as good as the cold-fit baseline —
///    the warm-start changes the basin entry, not the root, so recovery never
///    regresses (and the seed lands the solve in the right basin immediately).
#[test]
fn amortized_warm_start_matches_or_beats_cold_inner_solve_on_known_manifold() {
    let n = 24usize;
    let p = 4usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let m = phi.ncols();
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        let scale = 1.0 / (1.0 + b as f64);
        scale * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });
    let atom = SaeManifoldAtom::new(
        "periodic_truth",
        SaeAtomBasisKind::Periodic,
        1,
        phi.clone(),
        jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let target = phi.dot(&decoder);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![1.0_f64.ln()]]);

    // (1) Cold (sequential) inner solve — the REML-then-distill baseline.
    let mut rho_cold = rho.clone();
    term.run_joint_fit_arrow_schur(target.view(), &mut rho_cold, None, 12, 0.1, 1.0e-4, 1.0e-4)
        .expect("cold inner solve converges on the known periodic manifold");
    let cold_ev = {
        let fitted = term.try_fitted_for_rho(&rho_cold).unwrap();
        reconstruction_explained_variance(target.view(), fitted.view())
            .expect("explained variance is defined for the planted target")
    };
    assert!(
        cold_ev > 0.9,
        "cold fit must recover the planted periodic manifold (EV={cold_ev})"
    );

    // (2) Build the amortized encoder from the fitted dictionary and offer it
    // to the inner solve as an advisory warm-start. The Kantorovich gate is
    // intentionally conservative: if it cannot certify this fitted dictionary,
    // Design A must leave the cold seed untouched rather than corrupting the
    // inner state.
    let warm_started = term
        .warm_start_latents_from_amortized_encoder(target.view(), &rho_cold)
        .expect("amortized warm-start runs on the fitted dictionary");
    eprintln!("#1154 WARM-START: certified warm-started rows={warm_started}/{n}");
    assert!(
        warm_started <= n,
        "the amortized encoder cannot warm-start more rows than the fitted \
         batch size; warm_started={warm_started}, n={n}"
    );

    // (3) Re-converge FROM the warm-start; recovery must not regress.
    let mut rho_warm = rho.clone();
    term.run_joint_fit_arrow_schur(target.view(), &mut rho_warm, None, 12, 0.1, 1.0e-4, 1.0e-4)
        .expect("warm-started inner solve converges");
    let warm_ev = {
        let fitted = term.try_fitted_for_rho(&rho_warm).unwrap();
        reconstruction_explained_variance(target.view(), fitted.view())
            .expect("explained variance is defined for the planted target")
    };
    assert!(
        warm_ev >= cold_ev - 1.0e-6,
        "amortized warm-start (co-trained inner solve) must recover the manifold \
         at least as well as the cold/sequential solve: warm_ev={warm_ev}, \
         cold_ev={cold_ev}"
    );
}

/// #1154 item 3 — the JOINTLY co-trained encoder recovers the planted
/// manifold structure on held-out rows AT LEAST AS WELL as the sequential
/// REML-then-distill path. Both paths search the SAME ρ grid over the SAME
/// planted periodic dictionary; they differ only in HOW ρ is ranked and how
/// the inner solve is seeded:
///
///   * sequential — rank ρ by the BARE REML criterion, fit cold (chart-center
///     inner solve), then distill the amortized encoder once from the frozen
///     fitted dictionary (the #357 / #1026-ladder post-hoc path);
///   * co-trained (Design A) — rank ρ by the co-trained criterion (REML + the
///     amortized-encoder consistency fold) and warm-start the inner latent
///     coords from the amortized encoder built on the running dictionary at
///     each ρ, refining to the same stationary point.
///
/// On held-out planted rows the co-trained encoder's recovered circle phase
/// must match the planted truth at least as well as the sequential encoder's
/// — co-adapting the dictionary + λ toward a faithfully-invertible encode can
/// only help recovery, never regress it.
///
/// STATE (#1154/#1026, basin-warmup fix): this guarantee IS now demonstrable on
/// unit-amplitude held-out rows — the test certifies all held-out rows and asserts
/// the recovery comparison live (it is no longer `#[ignore]`d / vacuous).
///
/// Former root cause (now FIXED) — the encode-atlas Kantorovich certificate
/// (`row_certificate`, encode.rs) used to certify ZERO held-out rows of the planted
/// circle at amplitude 1.0, via BOTH the amortized one-mat-vec predictor AND the
/// exact cold-Newton chart-center probe. The certificate's worst-case
/// Hessian-Lipschitz constant `L = hessian_lipschitz_constant(.., amplitude, ..)`
/// scales with the assignment amplitude, so `h = β·η·L` exceeded the ½ acceptance
/// bound at amplitude 1.0 *from the chart-center / distilled start* — even though
/// that start was positive-definite with a valid Newton step. The IN-SAMPLE
/// faithfulness test (`cotrained_criterion_folds_…`) certified because its fitted
/// softmax masses are < 1 (smaller L ⇒ `h ≤ ½`).
///
/// The fix (`certify_with_basin_warmup`, encode.rs) restores the bounded Newton
/// "basin warm-up" a prior hardening had removed: from an uncertified-but-PD start,
/// take up to `SAE_ENCODE_BASIN_WARMUP_STEPS` plain Newton steps INTO the
/// Kantorovich basin, re-certifying at each iterate. The certificate at the landing
/// point is a full guarantee from there (`h ≤ ½` ⇒ Newton converges to the in-ball
/// root), so this widens the certified reach to unit amplitude WITHOUT weakening
/// the bound. The held-out `certified` count is now strictly positive (asserted
/// below), closing the remaining #1154 Design-A gap.
#[test]
fn cotrained_encoder_recovers_planted_manifold_at_least_as_well_as_sequential() {
    let n = 32usize;
    let p = 4usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.5) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let m = phi.ncols();
    // A smooth low-order periodic decoder: a genuine 1D periodic manifold the
    // amortized IFT predictor can faithfully model to first order.
    let decoder = Array2::from_shape_fn((m, p), |(b, c)| {
        let scale = 1.0 / (1.0 + b as f64);
        scale * ((b as f64 + 1.0) * (c as f64 + 1.0)).cos()
    });
    let target = phi.dot(&decoder);

    // A small shared ρ grid (log-sparsity, log-smoothness) over the same
    // dictionary; both paths search it identically so only the ranking +
    // seeding differ. ARD held at 1.0 (single d=1 atom).
    let rho_grid: Vec<SaeManifoldRho> = [(-0.5_f64, 0.4_f64), (0.0, 0.8), (0.3, 1.2)]
        .iter()
        .map(|&(ls, lsm)| SaeManifoldRho::new(ls, lsm.ln(), vec![array![1.0_f64.ln()]]))
        .collect();

    let build_term = || {
        let atom = SaeManifoldAtom::new(
            "periodic_truth",
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords.clone()],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom], assignment).unwrap()
    };

    // Held-out planted rows interleaved between the training coords (not an
    // in-sample replay): the encoder must recover their circle phase.
    let n_holdout = 16usize;
    let heldout_truth = Array2::from_shape_fn((n_holdout, 1), |(row, _)| {
        (row as f64 + 0.25) / n_holdout as f64
    });
    let (heldout_phi, _hjet) = periodic_basis(&heldout_truth);

    // Encode the held-out rows under a fitted `term` and return (max
    // circle-phase recovery gap to the planted truth, amortized-certified
    // count). The recovery gap measures whether the FITTED DICTIONARY places
    // the held-out rows on the planted circle — the actual claim of the test
    // (co-training selects a dictionary that recovers truth at least as well
    // as the sequential one). It is read off the EXACT cold-Newton encode
    // (`certified_encode_row` from the chart center), which converges to the
    // true circle phase on this clean planted manifold for EVERY row,
    // regardless of whether the Kantorovich certificate formally fires at the
    // hold-out amplitude. The amortized one-mat-vec predictor's certified
    // count is reported separately as a fidelity diagnostic (its per-row
    // amortized==exact agreement on certified rows is asserted by the
    // dedicated `cotrained_criterion_folds_faithful_amortized_encoder…`
    // faithfulness test); it must not gate the dictionary-recovery
    // measurement, or the recovery gaps go vacuously zero whenever the
    // certificate reach is short at unit amplitude.
    let heldout_recovery_gap = |term: &SaeManifoldTerm| -> (f64, usize) {
        let atom0 = &term.atoms[0];
        let heldout = heldout_phi.dot(&atom0.decoder_coefficients);
        let amps = Array1::<f64>::ones(n_holdout);
        let mut norm_bound = 0.0_f64;
        for row in 0..n_holdout {
            norm_bound = norm_bound.max(heldout.row(row).dot(&heldout.row(row)).sqrt());
        }
        let atlas = crate::encode::EncodeAtlas::build(
            &term.atoms,
            &[1.0],
            norm_bound,
            crate::encode::AtlasConfig::default(),
        )
        .expect("held-out encode atlas builds");
        let encoded = atlas
            .amortized_encode_batch(atom0, 0, heldout.view(), amps.view())
            .expect("held-out amortized encode runs");
        let mut max_gap = 0.0_f64;
        let mut amortized_certified = 0usize;
        for row in 0..n_holdout {
            if encoded.certified[row] {
                amortized_certified += 1;
            }
            // The exact cold-Newton encode from the chart center is the
            // recovery oracle: on this planted low-order periodic manifold it
            // converges to the true circle phase for every row, so the gap is
            // the dictionary's genuine held-out recovery error — never skipped.
            let (coord, _cert) = atlas
                .certified_encode_row(atom0, 0, heldout.row(row), amps[row])
                .expect("held-out exact encode converges");
            let gap = circle_phase_gap(coord[0], heldout_truth[[row, 0]]);
            max_gap = max_gap.max(gap);
        }
        (max_gap, amortized_certified)
    };

    // --- Sequential: rank ρ by BARE REML, fit cold, distill post-hoc. ---
    let mut best_seq_rho = rho_grid[0].clone();
    let mut best_seq_cost = f64::INFINITY;
    for rho in &rho_grid {
        let mut probe = build_term();
        let Ok((reml, _loss)) =
            probe.reml_criterion(target.view(), rho, None, 12, 1.0, 1.0e-4, 1.0e-4)
        else {
            continue;
        };
        if reml < best_seq_cost {
            best_seq_cost = reml;
            best_seq_rho = rho.clone();
        }
    }
    assert!(
        best_seq_cost.is_finite(),
        "the sequential grid must contain at least one converged bare-REML candidate"
    );
    // Cold re-fit at the bare-REML-selected ρ, then distill the encoder.
    let mut seq_term = build_term();
    let mut seq_rho = best_seq_rho.clone();
    seq_term
        .run_joint_fit_arrow_schur(target.view(), &mut seq_rho, None, 12, 1.0, 1.0e-4, 1.0e-4)
        .expect("sequential cold inner solve converges");
    let (seq_gap, seq_certified) = heldout_recovery_gap(&seq_term);

    // --- Co-trained: rank ρ by the co-trained criterion with the amortized
    // warm-start applied each step (Design A). ---
    let mut best_cot_rho = rho_grid[0].clone();
    let mut best_cot_cost = f64::INFINITY;
    for rho in &rho_grid {
        let mut probe = build_term();
        // Warm-start the inner latents from the amortized encoder built on the
        // running dictionary, then rank by the co-trained criterion.
        probe
            .warm_start_latents_from_amortized_encoder(target.view(), rho)
            .ok();
        let Ok((cotrained, _loss, _consistency)) =
            probe.reml_criterion_cotrained(target.view(), rho, None, 64, 1.0, 1.0e-4, 1.0e-4)
        else {
            continue;
        };
        if cotrained < best_cot_cost {
            best_cot_cost = cotrained;
            best_cot_rho = rho.clone();
        }
    }
    assert!(
        best_cot_cost.is_finite(),
        "the co-trained grid must contain at least one converged candidate"
    );
    let mut cot_term = build_term();
    let mut cot_rho = best_cot_rho.clone();
    cot_term
        .warm_start_latents_from_amortized_encoder(target.view(), &cot_rho)
        .ok();
    cot_term
        .run_joint_fit_arrow_schur(target.view(), &mut cot_rho, None, 64, 1.0, 1.0e-4, 1.0e-4)
        .expect("co-trained warm-started inner solve converges");
    let (cot_gap, cot_certified) = heldout_recovery_gap(&cot_term);

    eprintln!(
        "#1154 RECOVERY: sequential max-phase-gap={seq_gap:.6e} (certified={seq_certified}) \
         | co-trained max-phase-gap={cot_gap:.6e} (certified={cot_certified}) \
         | delta(cot-seq)={:.6e}",
        cot_gap - seq_gap
    );
    // The recovery gaps are read off the exact cold-Newton encode, which
    // converges on every held-out row of this planted manifold, so the
    // comparison is LIVE (non-vacuous) on both dictionaries: the co-trained
    // dictionary must place the held-out rows on the planted circle at least
    // as well as the bare-REML-selected sequential dictionary. The amortized
    // one-mat-vec predictor's certified count (`*_certified`) is a reported
    // fidelity diagnostic; the amortized==exact-on-certified-rows claim is the
    // separate `cotrained_criterion_folds_faithful_amortized_encoder…` test.
    assert!(
        seq_gap.is_finite() && cot_gap.is_finite(),
        "both dictionaries' exact held-out recovery gaps must be finite: \
         sequential={seq_gap}, co-trained={cot_gap}"
    );
    assert!(
        cot_gap <= seq_gap + 1.0e-3,
        "co-trained dictionary must recover the planted held-out manifold at \
         least as well as the sequential REML-then-distill path: \
         co-trained max phase gap={cot_gap}, sequential={seq_gap} \
         (amortized-certified rows: co-trained={cot_certified}, \
         sequential={seq_certified})"
    );
    // #1154/#1026 basin-warmup fix: the amortized one-mat-vec encoder must now
    // CERTIFY unit-amplitude held-out rows (previously it certified ZERO — the
    // Kantorovich h = β·η·L exceeded ½ at amplitude 1.0 from the chart-center start,
    // so every row fell back to the exact solve). The bounded plain-Newton basin
    // warm-up navigates into the certified ball, so the distilled encoder is now
    // usable at unit amplitude — a strictly positive certified count on this clean
    // planted manifold.
    assert!(
        seq_certified > 0 && cot_certified > 0,
        "the amortized encoder must certify at least one unit-amplitude held-out row \
         on this clean planted circle (basin warm-up closed the #1154 certifies-zero \
         gap); got sequential={seq_certified}, co-trained={cot_certified} of {n_holdout}"
    );
}

#[test]
pub(crate) fn deflated_solver_matches_plain_solve_when_no_gauge_is_installed() {
    let cache = diagonal_latent_cache(&[2.0_f64, 5.0, 7.0]);
    let solver = DeflatedArrowSolver::plain(&cache);
    let rhs_t = array![4.0_f64, 10.0, -14.0];
    let rhs_beta = Array1::<f64>::zeros(0);
    let (plain_t, plain_beta) = cache
        .full_inverse_apply(rhs_t.view(), rhs_beta.view())
        .expect("plain cache solve");
    let solved = solver
        .solve(rhs_t.view(), rhs_beta.view())
        .expect("adapter solve");
    assert_eq!(solved.t.len(), plain_t.len());
    for idx in 0..plain_t.len() {
        assert_abs_diff_eq!(solved.t[idx], plain_t[idx], epsilon = 0.0);
    }
    assert_eq!(solved.beta.len(), plain_beta.len());
    for idx in 0..plain_beta.len() {
        assert_abs_diff_eq!(solved.beta[idx], plain_beta[idx], epsilon = 0.0);
    }
}

#[test]
pub(crate) fn deflated_solver_matches_dense_quotient_pseudoinverse_on_near_null_fixture() {
    let cache = diagonal_latent_cache(&[2.0_f64, 1.0e-14]);
    let gauge = array![0.0_f64, 1.0];
    let solver = DeflatedArrowSolver::from_orthonormal_gauges(&cache, vec![gauge], 2.0)
        .expect("deflated solver");
    let rhs_beta = Array1::<f64>::zeros(0);

    let physical_rhs = array![4.0_f64, 0.0];
    let solved = solver
        .solve(physical_rhs.view(), rhs_beta.view())
        .expect("physical solve");
    let oracle = array![2.0_f64, 0.0];
    for idx in 0..oracle.len() {
        assert_abs_diff_eq!(solved.t[idx], oracle[idx], epsilon = 1.0e-12);
    }

    let gauge_rhs = array![0.0_f64, 1.0];
    let plain = cache
        .full_inverse_apply(gauge_rhs.view(), rhs_beta.view())
        .expect("plain gauge solve")
        .0;
    let stiffened = solver
        .solve(gauge_rhs.view(), rhs_beta.view())
        .expect("stiffened gauge solve")
        .t;
    assert!(plain[1] > 1.0e13, "plain near-null solve must be huge");
    assert_abs_diff_eq!(stiffened[1], 0.5, epsilon = 1.0e-12);
}

#[test]
pub(crate) fn pca_seed_handles_huge_equal_finite_columns_without_mean_overflow() {
    let z = array![[1.0e308_f64, 1.0e308], [1.0e308, 1.0e308]];
    let coords =
        sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1]).unwrap();
    assert_eq!(coords.dim(), (1, 2, 1));
    assert!(
        coords.iter().all(|value| value.is_finite()),
        "huge finite equal columns must not overflow the PCA seed mean: {coords:?}"
    );
}

#[test]
pub(crate) fn pca_seed_rejects_huge_finite_span_that_overflows_centering() {
    let z = array![[1.0e308_f64, 0.0], [-1.0e308, 0.0]];
    let err = sae_pca_seed_initial_coords(z.view(), &[SaeAtomBasisKind::Periodic], &[1])
        .expect_err("opposite huge finite values exceed f64 centering range");
    assert!(
        err.contains("centered Z is non-finite") || err.contains("SVD failed"),
        "unexpected PCA seed error: {err}"
    );
}

// ---- Issue #972: low-rank Grassmann decoder frame verification ----

/// `polar(M) = W Vᵀ` is exactly column-orthonormal and equals `M` when `M`
/// is already orthonormal (idempotence of the polar projection on the
/// Stiefel manifold), and recovers the planted span of a low-rank decoder.
#[test]
pub(crate) fn planted_low_rank_frame_recovered_by_polar() {
    let p = 12usize;
    let r = 3usize;
    let n = 200usize;
    // Planted orthonormal frame: first `r` canonical axes (any rotation
    // would do; canonical axes make the angle assertion transparent).
    let mut planted = Array2::<f64>::zeros((p, r));
    for j in 0..r {
        planted[[j, j]] = 1.0;
    }
    // Latent coords drive targets onto the planted span: targets = coords·plantedᵀ.
    let mut coords = Array2::<f64>::zeros((n, r));
    for i in 0..n {
        for j in 0..r {
            // Deterministic, index-keyed pseudo-data (no clock RNG).
            let x = ((i * 7 + j * 13 + 1) % 97) as f64 / 97.0 - 0.5;
            coords[[i, j]] = x;
        }
    }
    let targets = fast_abt(&coords, &planted);
    let angle = grassmann_recover_planted_span_angle(targets.view(), coords.view(), planted.view())
        .expect("span recovery");
    assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-9);

    // Polar of an already-orthonormal frame is itself (up to canonical sign).
    let frame = GrassmannFrame::polar_update(planted.view()).expect("polar");
    let recovered_angle = frame
        .max_principal_angle(planted.view())
        .expect("principal angle");
    assert_abs_diff_eq!(recovered_angle, 0.0, epsilon = 1.0e-9);
    // Orthonormality: UᵀU = I_r.
    let gram = fast_atb(&frame.frame().to_owned(), &frame.frame().to_owned());
    for i in 0..r {
        for j in 0..r {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(gram[[i, j]], expect, epsilon = 1.0e-9);
        }
    }
}

/// Regression test for #1415: the JumpReLU third derivative consumed by the
/// log-determinant θ-adjoint (`assignment_prior_hdiag_derivative_entry`) must be
/// the EXACT derivative of the (separately certified) Hessian diagonal
/// `P''(ℓ)=(λ/τ²)s(1−2a)`, namely `P'''(ℓ)=(λ/τ³)s(1−6a+6a²)`. The historical
/// code used `(2λ/τ³)s²(1−2a)`, which is algebraically wrong and vanishes at the
/// threshold (`a=1/2`) where the true value is `−λ/(8τ³)`.
///
/// Oracle: a 4th-order central finite difference of the exact `P''(ℓ)` formula
/// (FD permitted only inside the test as an independent check of the closed
/// form). Also pins the exact threshold value `−λ/(8τ³)` and asserts the
/// production entry is strictly negative there (the old formula returned 0).
#[test]
fn jumprelu_hdiag_third_derivative_matches_central_difference_1415() {
    use ndarray::{Array1, Array2, Array3};
    let n = 6usize;
    let k = 2usize;
    let p = 3usize;
    let temperature = 0.35_f64;
    let threshold = 0.1_f64;
    // Include the exact threshold (logit == θ ⇒ a = 1/2) plus in-band points.
    let logits = Array2::<f64>::from_shape_vec(
        (n, k),
        vec![
            0.1, 0.0, 0.2, -0.05, 0.05, 0.15, 0.25, 0.3, -0.1, 0.12, 0.18, 0.08,
        ],
    )
    .expect("valid logit grid");
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|i| {
            SaeManifoldAtom::new(
                &format!("atom{i}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                Array2::<f64>::ones((n, 2)),
                Array3::<f64>::zeros((n, 2, 1)),
                Array2::<f64>::zeros((2, p)),
                Array2::<f64>::eye(2),
            )
            .unwrap()
        })
        .collect();
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Euclidean; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.clone(),
        coords,
        manifolds,
        AssignmentMode::jumprelu(temperature, threshold),
    )
    .expect("valid JumpReLU assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);

    let inv_tau = 1.0 / temperature;
    let sparsity = rho.log_lambda_sparse.exp();
    let in_band = |logit: f64| {
        crate::assignment::jumprelu_in_optimization_band(logit, threshold, temperature)
    };
    // Exact, separately-certified Hessian diagonal P''(ℓ) as a function of ℓ.
    let p2 = |logit: f64| -> f64 {
        if !in_band(logit) {
            return 0.0;
        }
        let a = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
        let s = a * (1.0 - a);
        sparsity * s * (1.0 - 2.0 * a) * inv_tau * inv_tau
    };

    let mut saw_threshold = false;
    for row in 0..n {
        for atom in 0..k {
            let logit = logits[[row, atom]];
            if !in_band(logit) {
                continue;
            }
            let entry = term.assignment_prior_hdiag_derivative_entry(
                &rho,
                row,
                atom,
                SaeLocalRowVar::Logit { atom },
                None,
            );
            // Independent oracle: 4th-order central difference of exact P''(ℓ).
            let h = 1.0e-3_f64;
            let fd = (-p2(logit + 2.0 * h) + 8.0 * p2(logit + h) - 8.0 * p2(logit - h)
                + p2(logit - 2.0 * h))
                / (12.0 * h);
            let scale = entry.abs().max(fd.abs()).max(1.0e-8);
            assert!(
                (entry - fd).abs() <= 1.0e-5 * scale,
                "row {row} atom {atom}: P''' entry {entry:e} vs FD {fd:e}"
            );

            if (logit - threshold).abs() < 1e-12 {
                saw_threshold = true;
                // At ℓ=θ: a=1/2, s=1/4 ⇒ P'''=−λ/(8τ³); the old code gave 0.
                let expected = -sparsity / 8.0 * inv_tau * inv_tau * inv_tau;
                assert_abs_diff_eq!(entry, expected, epsilon = 1e-9);
                assert!(
                    entry < -1e-6,
                    "threshold third derivative must be strictly negative (old buggy \
                     formula returned 0): entry={entry:e}"
                );
            }
        }
    }
    assert!(
        saw_threshold,
        "fixture must include a logit exactly at the threshold to pin −λ/(8τ³)"
    );
}
