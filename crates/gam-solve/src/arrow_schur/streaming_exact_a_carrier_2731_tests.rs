//! #2731 — the streaming/chunked evidence lane must carry the #2515 exact-A
//! classification, because it already carries one half of the policy that reads
//! it.
//!
//! `StreamingArrowSchur` was taught `refuse_resolved_indefinite` — the verdict
//! half of [`ArrowEvidencePolicy::UnitDeflationRefusingIndefinite`] — but not
//! the raw `B`/`delta`/clamp carrier that verdict is taken against. The two
//! halves of one policy then described different operators, and the chunked
//! in-core SAE evidence route (`SaeManifoldTerm::streaming_exact_arrow_log_det`,
//! which sets exactly that policy) failed at EVERY evaluation with
//! `evidence reduced Schur unit-deflation declined (exact-A evidence
//! classification requires its raw B/delta/clamp carrier)`. The route was not
//! numerically fragile; it was structurally unreachable, and nothing was red
//! because nothing scheduled it.
//!
//! The pins below fix both halves: the per-row factorization must see the
//! carrier (or the streaming log-determinant silently disagrees with the dense
//! one, the #1377 invariant), and the accumulated reduced Schur must be
//! classified against the accumulated carrier.
#![cfg(test)]

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::array;

/// The #2515 clamp-basin anchor, verbatim: `A_tt = 1`, `A_tbeta = 1`,
/// `A_betabeta = 1/2` gives `S_A = -1/2`; `delta_tt = -2` makes `B_tt = 3`, so
/// the lifted Schur direction has majorizer curvature `3/2` and clamp basin
/// `-1/2 + clamp`. This is the same fixture
/// `matrix_free_exact_a_prices_a_clamp_basin_before_refusing_a_saddle_2515`
/// pins the SLQ and rational arms against, so the streaming arm's numbers are
/// comparable to an already-established value rather than to itself.
fn clamp_basin_system(clamp_value: f64, carrier: bool) -> ArrowSchurSystem {
    let mut system = ArrowSchurSystem::new(1, 1, 1);
    system.rows[0].htt[[0, 0]] = 1.0;
    system.rows[0].htbeta[[0, 0]] = 1.0;
    system.hbb[[0, 0]] = 0.5;
    if carrier {
        system.exact_a_classification = Some(ExactAClassificationGeometry {
            rows: vec![ExactAClassificationRow {
                delta_tt: array![[-2.0_f64]],
                delta_tbeta: Array2::<f64>::zeros((1, 0)),
                clamp_diag: array![clamp_value],
            }]
            .into(),
            delta_beta: Arc::from([]),
            border_indices: Arc::from([] as [usize; 0]),
        });
    }
    system
}

/// A two-row exact-A system whose FIRST row carries a spectrally flat latent
/// direction (`1e-12` against a `4.0` row norm). The exact-A carrier is the only
/// thing that routes a streaming row through the deflating recovery, so this
/// row's evidence log-determinant is `log 1 = 0` on the dense lane and the raw
/// `log 1e-12` on a lane that dropped the carrier.
///
/// `delta_tt = 0` and `clamp_diag = 0` make `B_raw = A`, which is what lets the
/// majorizer metric be checked against the reduced Schur itself.
fn flat_direction_system(rows: &[usize], hbb: f64) -> ArrowSchurSystem {
    let htt = [array![[4.0_f64, 0.0], [0.0, 1.0e-12]], array![
        [2.0_f64, 0.0],
        [0.0, 3.0]
    ]];
    let htbeta = [array![[1.0_f64], [0.0]], array![[0.5_f64], [0.25]]];
    let mut system = ArrowSchurSystem::new(rows.len(), 2, 1);
    for (slot, &source) in rows.iter().enumerate() {
        system.rows[slot].htt = htt[source].clone();
        system.rows[slot].htbeta = htbeta[source].clone();
    }
    system.hbb[[0, 0]] = hbb;
    system.exact_a_classification = Some(ExactAClassificationGeometry {
        rows: rows
            .iter()
            .map(|_| ExactAClassificationRow {
                delta_tt: Array2::<f64>::zeros((2, 2)),
                delta_tbeta: Array2::<f64>::zeros((2, 0)),
                clamp_diag: Array1::<f64>::zeros(2),
            })
            .collect::<Vec<_>>()
            .into(),
        delta_beta: Arc::from([]),
        border_indices: Arc::from([] as [usize; 0]),
    });
    system
}

fn refusing_options() -> ArrowSolveOptions {
    ArrowSolveOptions::direct()
        .with_gpu_policy(gam_gpu::GpuPolicy::Off)
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR)
}

fn slab_log_det(factors: &ArrowFactorSlab) -> f64 {
    (0..factors.len())
        .map(|row| {
            let factor = factors.factor(row);
            (0..factor.nrows())
                .map(|axis| 2.0 * factor[[axis, axis]].ln())
                .sum::<f64>()
        })
        .sum()
}

/// The chunked lane prices the clamp basin at the number the matrix-free arms
/// were already pinned to. Before the carrier travelled, this call could not
/// return a value at all.
#[test]
fn streaming_evidence_chunk_prices_the_clamp_basin_like_the_matrix_free_arm_2731() {
    let sys = clamp_basin_system(2.0, true);
    let options = refusing_options();
    let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len());
    let chunk = streaming
        .evidence_schur_chunk(&sys, 0.0, 0.0, &options)
        .expect("#2731: a carried exact-A classification must let the chunk be accumulated");
    assert_abs_diff_eq!(chunk.log_det_tt, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(chunk.schur[[0, 0]], -0.5, epsilon = 1.0e-12);

    let classification = chunk
        .exact_a_classification(&sys)
        .expect("#2731: the classification must be readable off the chunk's own factors")
        .expect("#2731: a system carrying the geometry must produce reduced metrics");
    // `B` curvature `3/2` and clamp basin `2` on the lifted direction — the two
    // scalars the shared classifier turns into the priced basin.
    assert_abs_diff_eq!(classification.majorizer_metric[[0, 0]], 1.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(classification.clamp_metric[[0, 0]], 2.0, epsilon = 1.0e-12);

    let log_det_schur =
        StreamingArrowSchur::reduced_schur_log_det(&chunk.schur, &options, Some(&classification))
            .expect("#2731: the clamp basin is a priced direction, not a refusal");
    assert_abs_diff_eq!(log_det_schur, 1.5_f64.ln(), epsilon = 1.0e-12);
}

/// Negative curvature BEYOND the clamp basin is still the typed saddle refusal,
/// so the carrier restores the route without weakening the verdict.
#[test]
fn streaming_evidence_chunk_refuses_a_saddle_beyond_the_clamp_basin_2731() {
    let sys = clamp_basin_system(0.0, true);
    let options = refusing_options();
    let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len());
    let chunk = streaming
        .evidence_schur_chunk(&sys, 0.0, 0.0, &options)
        .expect("#2731: accumulation succeeds; the verdict is taken at the factorization");
    let classification = chunk
        .exact_a_classification(&sys)
        .expect("#2731: the classification must be readable")
        .expect("#2731: a system carrying the geometry must produce reduced metrics");
    assert_abs_diff_eq!(classification.clamp_metric[[0, 0]], 0.0, epsilon = 1.0e-12);
    let refusal =
        StreamingArrowSchur::reduced_schur_log_det(&chunk.schur, &options, Some(&classification))
            .expect_err("#2731: an unclamped resolved negative direction is a saddle");
    let rendered = format!("{refusal}");
    assert!(
        ArrowSchurError::rendered_is_indefinite_evidence(&rendered),
        "#2731: the streaming lane must raise the SHARED typed saddle marker: {rendered}"
    );
}

/// The per-row half of the carrier. The streaming and dense lanes must factor
/// the SAME operator, and the value they agree on must be the deflated one — a
/// hard-coded `log 4 + log 1 + log 2 + log 3`, so the two lanes cannot pass by
/// agreeing on the undeflated number.
#[test]
fn the_streaming_row_factor_sees_the_exact_a_carrier_like_the_dense_one_2731() {
    let sys = flat_direction_system(&[0, 1], 8.0);
    let options = refusing_options();
    let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len());
    let chunk = streaming
        .evidence_schur_chunk(&sys, 0.0, 0.0, &options)
        .expect("#2731: the carried geometry routes the flat row through the deflating recovery");

    let dense = factor_blocks_for_system(
        &sys,
        0.0,
        options.evidence_policy,
        &CpuBatchedBlockSolver,
        gam_gpu::GpuPolicy::Off,
    )
    .expect("#2731: the dense lane factors the same exact-A system");
    let dense_log_det_tt = slab_log_det(&dense.factors);

    // The invariant, exactly: the two lanes factored the same operator.
    assert_abs_diff_eq!(chunk.log_det_tt, dense_log_det_tt, epsilon = 1.0e-12);
    // And the value they agree on is the DEFLATED one. Two lanes agreeing prove
    // nothing on their own, so this pins the number: an unpinned flat direction
    // reads `log 1e-12` and lands 27.6 nats below, seven decades outside this
    // window, while the window itself is wide enough for the eigendecomposition
    // round-trip the deflating recovery makes (measured 4.0e-8 here).
    let deflated = 4.0_f64.ln() + 2.0_f64.ln() + 3.0_f64.ln();
    assert_abs_diff_eq!(dense_log_det_tt, deflated, epsilon = 1.0e-6);
    assert_abs_diff_eq!(chunk.log_det_tt, deflated, epsilon = 1.0e-6);

    // `delta_tt = 0` makes `B_raw = A`, so the majorizer metric on the lifted
    // direction is the reduced Schur itself. Two independent assemblies of the
    // same quantity from the same factors.
    let classification = chunk
        .exact_a_classification(&sys)
        .expect("#2731: the classification must be readable")
        .expect("#2731: a system carrying the geometry must produce reduced metrics");
    assert_abs_diff_eq!(
        classification.majorizer_metric[[0, 0]],
        chunk.schur[[0, 0]],
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        chunk.schur[[0, 0]],
        8.0 - 0.25 - 0.5 * 0.25 - 0.0625 / 3.0,
        epsilon = 1.0e-9
    );
}

/// The whole point of the repair: all three accumulated quantities are additive
/// over a row split, so a chunked caller sums them and factors the total ONCE.
#[test]
fn the_evidence_chunk_is_additive_over_a_row_split_2731() {
    let options = refusing_options();
    let whole = flat_direction_system(&[0, 1], 8.0);
    let mut whole_streaming = StreamingArrowSchur::from_system(&whole, whole.rows.len());
    let whole_chunk = whole_streaming
        .evidence_schur_chunk(&whole, 0.0, 0.0, &options)
        .expect("#2731: the whole system accumulates");
    let whole_classification = whole_chunk
        .exact_a_classification(&whole)
        .expect("#2731: readable")
        .expect("#2731: present");

    // The beta block is split in the same proportion the SAE chunked route
    // splits it (`penalty_scale`), so the halves sum to the whole.
    let mut split_log_det_tt = 0.0_f64;
    let mut split_schur = Array2::<f64>::zeros((1, 1));
    let mut split_classification = ExactAReducedClassification::zeros(1);
    for row in [0usize, 1usize] {
        let part = flat_direction_system(&[row], 4.0);
        let mut part_streaming = StreamingArrowSchur::from_system(&part, part.rows.len());
        let part_chunk = part_streaming
            .evidence_schur_chunk(&part, 0.0, 0.0, &options)
            .expect("#2731: each half accumulates");
        let part_classification = part_chunk
            .exact_a_classification(&part)
            .expect("#2731: readable")
            .expect("#2731: present");
        split_log_det_tt += part_chunk.log_det_tt;
        split_schur += &part_chunk.schur;
        split_classification
            .accumulate(&part_classification)
            .expect("#2731: same border width");
    }

    assert_abs_diff_eq!(split_log_det_tt, whole_chunk.log_det_tt, epsilon = 1.0e-12);
    assert_abs_diff_eq!(split_schur[[0, 0]], whole_chunk.schur[[0, 0]], epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        split_classification.majorizer_metric[[0, 0]],
        whole_classification.majorizer_metric[[0, 0]],
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        split_classification.clamp_metric[[0, 0]],
        whole_classification.clamp_metric[[0, 0]],
        epsilon = 1.0e-12
    );

    let split_log_det = StreamingArrowSchur::reduced_schur_log_det(
        &split_schur,
        &options,
        Some(&split_classification),
    )
    .expect("#2731: the summed carrier classifies the summed Schur");
    let whole_log_det = StreamingArrowSchur::reduced_schur_log_det(
        &whole_chunk.schur,
        &options,
        Some(&whole_classification),
    )
    .expect("#2731: the whole system's carrier classifies its own Schur");
    assert_abs_diff_eq!(split_log_det, whole_log_det, epsilon = 1.0e-12);
}

/// Contract control, accumulation side: the fix makes the carrier TRAVEL, it
/// does not switch the requirement off. A refusing policy handed a system with
/// no carrier still fails — and now fails where the dense lane fails, naming the
/// policy rather than the factorization.
#[test]
fn a_refusing_policy_without_a_carrier_still_fails_at_accumulation_2731() {
    let sys = clamp_basin_system(2.0, false);
    let options = refusing_options();
    let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len());
    let refusal = streaming
        .evidence_schur_chunk(&sys, 0.0, 0.0, &options)
        .expect_err("#2731: the verdict half of the policy cannot run without its operand");
    let rendered = format!("{refusal}");
    assert!(
        rendered.contains("exact-A evidence classification requires the raw B/delta/clamp carrier"),
        "#2731: the streaming lane must state the DENSE lane's precondition: {rendered}"
    );
}

/// Contract control, factorization side: handing the reduced factorization a
/// refusing policy and no carrier is still a decline, so a chunked caller that
/// drops the accumulated carrier gets an error rather than a laxer verdict.
#[test]
fn the_reduced_schur_declines_a_refusing_policy_handed_no_carrier_2731() {
    let sys = clamp_basin_system(2.0, true);
    let options = refusing_options();
    let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len());
    let chunk = streaming
        .evidence_schur_chunk(&sys, 0.0, 0.0, &options)
        .expect("#2731: accumulation succeeds");
    let refusal = StreamingArrowSchur::reduced_schur_log_det(&chunk.schur, &options, None)
        .expect_err("#2731: the refusing policy needs the carrier at the factorization too");
    let rendered = format!("{refusal}");
    assert!(
        rendered.contains("exact-A evidence classification requires its raw B/delta/clamp carrier"),
        "#2731: the decline must name the missing carrier: {rendered}"
    );
}

/// A policy that never asks for the carrier is untouched: the ordinary
/// unit-deflating evidence lane still runs on a carrier-free system, with the
/// generic entry point and no classification argument.
#[test]
fn a_non_refusing_evidence_policy_is_unchanged_without_a_carrier_2731() {
    let sys = clamp_basin_system(2.0, false);
    let options = ArrowSolveOptions::direct()
        .with_gpu_policy(gam_gpu::GpuPolicy::Off)
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);
    let mut streaming = StreamingArrowSchur::from_system(&sys, sys.rows.len());
    let (log_det_tt, schur) = streaming
        .reduced_schur_and_log_det_tt(0.0, 0.0, &options)
        .expect("#2731: a policy that does not ask for the carrier must not require one");
    assert_abs_diff_eq!(log_det_tt, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(schur[[0, 0]], -0.5, epsilon = 1.0e-12);
    let log_det_schur = StreamingArrowSchur::reduced_schur_log_det(&schur, &options, None)
        .expect("#2731: one-sided unit deflation still pins the negative direction");
    assert_abs_diff_eq!(log_det_schur, 0.0, epsilon = 1.0e-12);
}
