//! #2333 — rows the exact-A evidence factor prices as a clamp basin.
//!
//! `factor_spectral_deflated_criterion_row_with_geometry` prices an
//! `ExactADirectionClassification::ClampBasin` direction at its basin curvature
//! without unit-deflating it, so the row reaches the cache with a recorded,
//! non-identity spectrum and an EMPTY direction list. Every ρ/θ trace that
//! contracts the row's raw derivative must subtract `deflation_block_correction`
//! there. Before `SaeManifoldTerm::row_deflation_is_live`, the θ-adjoints, the
//! dense assignment log-strength trace and the ARD traces gated on the direction
//! list and skipped those rows.
//!
//! The arbiter reads the SAME geometry-factored cache as the producer under test
//! and already branched on the spectrum before the fix: the probes log-strength
//! trace for its dense sibling. A frozen-gate finite difference of
//! `exact_observed_information_log_dets` is NOT an arbiter for these rows: it
//! prices basins on the dense joint spectrum and never sees a per-row clamp basin.
#![cfg(test)]
use super::*;
use crate::assignment::AssignmentMode;
use crate::manifold::arrow_solver::DeflatedArrowSolver;
use crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use gam_solve::arrow_schur::{
    ArrowSolveOptions, SPECTRAL_DEFLATION_REL_FLOOR, solve_arrow_newton_step_with_options,
};
use ndarray::{Array1, Array2};

/// One declared state whose exact-A evidence factor carries clamp-basin rows.
struct ClampBasinState {
    term: SaeManifoldTerm,
    rho: SaeManifoldRho,
    cache: ArrowFactorCache,
    rows: Vec<usize>,
}

/// The exact-A evidence factor the matrix-free outer gradient consumes
/// (`BundleEvidenceGeometry::cache`): the majorizer system corrected to
/// `A = B + ΔC`, carrying the classification geometry, and factored under the
/// refusing unit-deflation policy of that lane.
fn exact_a_evidence_cache(
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> Result<(SaeManifoldTerm, ArrowFactorCache), String> {
    let mut anchor = term.clone();
    let mut majorizer = anchor.assemble_arrow_schur(target.view(), rho, None)?;
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut majorizer);
    let exact = anchor.exact_a_evidence_system(target.view(), rho, &majorizer)?;
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&exact, 0.0, 0.0, &options)
            .map_err(|err| format!("{err:?}"))?;
    anchor.streaming_gates_frozen = true;
    Ok((anchor, cache))
}

/// Rows recorded with a spectrum that reprices at least one eigenvalue and no
/// deflated direction: the clamp-basin shape. An identity spectrum is not
/// counted, so a recorded-but-inert map cannot satisfy the premise.
fn clamp_basin_rows(cache: &ArrowFactorCache) -> Vec<usize> {
    cache
        .deflation_row_spectra
        .iter()
        .zip(cache.deflated_row_directions.iter())
        .enumerate()
        .filter_map(|(row, (spectrum, directions))| {
            let spectrum = spectrum.as_ref()?;
            let repriced = spectrum
                .raw_evals
                .iter()
                .zip(spectrum.cond_evals.iter())
                .any(|(raw, priced)| {
                    (raw - priced).abs() > f64::EPSILON * raw.abs().max(priced.abs()).max(1.0)
                });
            (directions.is_empty() && repriced).then_some(row)
        })
        .collect()
}

/// Walk a declared ARD-precision ladder and return the first state whose factor
/// records a clamp-basin row. The rung is selected on the cache's own
/// classification, never on agreement with any producer. Both clamp producers
/// can fire: the periodic ARD prior's concave half on coordinate slots, and the
/// ThresholdGate's concave remainder on logits above the threshold. The census
/// of every rung prints unconditionally.
fn first_clamp_basin_state(mode: AssignmentMode, straddle: bool) -> ClampBasinState {
    let (mut term, target, fixture_rho) = threshold_gate_tiny_fixture(straddle);
    term.assignment.mode = mode;
    let base_rho = fixture_rho.for_assignment(mode);
    let mut census = Vec::new();
    for log_ard in [-1.0_f64, 0.0, 1.0, 2.0, 3.0, 4.0] {
        let mut rho = base_rho.clone();
        for axes in rho.log_ard.iter_mut() {
            axes.fill(log_ard);
        }
        match exact_a_evidence_cache(&term, &target, &rho) {
            Ok((anchor, cache)) => {
                let rows = clamp_basin_rows(&cache);
                let spectral = cache
                    .deflation_row_spectra
                    .iter()
                    .filter(|spectrum| spectrum.is_some())
                    .count();
                let directions: usize = cache.deflated_row_directions.iter().map(Vec::len).sum();
                census.push(format!(
                    "log_ard={log_ard}: spectral_rows={spectral} \
                     deflated_directions={directions} clamp_basin_rows={rows:?}"
                ));
                if !rows.is_empty() {
                    eprintln!(
                        "#2333 CLAMP_BASIN_CENSUS mode={mode:?} straddle={straddle}\n{}",
                        census.join("\n")
                    );
                    return ClampBasinState {
                        term: anchor,
                        rho,
                        cache,
                        rows,
                    };
                }
            }
            Err(err) => census.push(format!("log_ard={log_ard}: no factor: {err}")),
        }
    }
    panic!(
        "#2333 premise: no rung of the declared ladder produced a clamp-basin row \
         (spectrum recorded and repriced, no deflated direction), so this gate \
         would compare producers where the two conventions coincide:\n{}",
        census.join("\n")
    );
}

/// The exact `(z_j, S⁻¹ z_j)` bundle at full-basis probes `√k·e_j`, where the
/// Hutchinson outer products are algebraically exact.
fn full_basis_bundle(cache: &ArrowFactorCache) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut probe = Array1::<f64>::zeros(k);
            probe[j] = sqrt_k;
            probe
        })
        .collect();
    let sinv = probes
        .iter()
        .map(|probe| {
            cache
                .schur_inverse_apply(probe.view())
                .expect("exact reduced-Schur solve at a full-basis probe")
        })
        .collect();
    (probes, sinv)
}

#[test]
fn clamp_basin_rows_enter_the_dense_assignment_log_strength_trace_2333() {
    let state = first_clamp_basin_state(AssignmentMode::threshold_gate(1.0, 0.0), true);
    let solver = DeflatedArrowSolver::plain(&state.cache);
    let dense = state
        .term
        .assignment_log_strength_hessian_trace(&state.rho, &state.cache, &solver)
        .expect("dense assignment log-strength trace on the exact-A factor");
    let (probes, sinv) = full_basis_bundle(&state.cache);
    // The dense trace contracts the majorizer's prior curvature; the probes
    // sibling at the majorizer operator contracts the same curvature.
    let from_probes = state
        .term
        .assignment_log_strength_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("probes assignment log-strength trace on the exact-A factor");
    let gap = (dense - from_probes).abs();
    let bar = 1.0e-9 * (1.0 + dense.abs().max(from_probes.abs()));
    eprintln!(
        "#2333 CLAMP_BASIN_LOG_STRENGTH rows={:?} dense={dense:.12e} \
         probes={from_probes:.12e} gap={gap:.6e}",
        state.rows
    );
    assert!(
        gap <= bar,
        "#2333: on clamp-basin rows the dense assignment log-strength trace must \
         subtract the correction its probes sibling subtracts: dense={dense:e} \
         probes={from_probes:e} gap={gap:e} bar={bar:e}"
    );
}

/// Largest per-entry relative departure of `actual` from `reference`
/// (`|r − a| / (1 + |r|)`), and the largest reference magnitude.
fn adjoint_gap(reference: &SaeArrowVector, actual: &SaeArrowVector) -> (f64, f64) {
    assert_eq!(reference.t.len(), actual.t.len());
    assert_eq!(reference.beta.len(), actual.beta.len());
    let mut gap = 0.0_f64;
    let mut scale = 0.0_f64;
    for (expected, observed) in reference
        .t
        .iter()
        .chain(reference.beta.iter())
        .zip(actual.t.iter().chain(actual.beta.iter()))
    {
        assert!(
            expected.is_finite() && observed.is_finite(),
            "non-finite adjoint entry: reference={expected} actual={observed}"
        );
        gap = gap.max((expected - observed).abs() / (1.0 + expected.abs()));
        scale = scale.max(expected.abs());
    }
    (gap, scale)
}

/// #2333 — the resident Trace θ-adjoint of an independent-logistic gate on
/// clamp-basin rows.
///
/// `logdet_theta_adjoint` reduces threshold-gate rows through the Trace seam:
/// the independent row program, the `E_tt` fold and the host post-folds. The
/// arbiter is `logdet_theta_adjoint_from_probes` at full-basis probes, where its
/// Hutchinson outer products are exact; it keeps its own hand tower loop
/// (host-whitened channels, contract-then-subtract correction), so the two share
/// no reduction code. A clamp-basin row is the shape on which the fold must
/// branch on the spectrum with an empty direction list. Non-vacuity is the
/// measured separation of the arbiter from itself on a deflation-blind copy of
/// the cache.
#[test]
fn threshold_gate_trace_theta_adjoint_matches_from_probes_on_clamp_basin_rows_2333() {
    let state = first_clamp_basin_state(AssignmentMode::threshold_gate(1.0, 0.0), true);
    let solver = DeflatedArrowSolver::plain(&state.cache);
    let resident = state
        .term
        .logdet_theta_adjoint(&state.rho, &state.cache, &solver)
        .expect("resident Trace theta-adjoint on the exact-A factor");
    let (probes, sinv) = full_basis_bundle(&state.cache);
    let reference = state
        .term
        .logdet_theta_adjoint_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("from-probes theta-adjoint on the exact-A factor");
    let (parity, scale) = adjoint_gap(&reference, &resident);
    let mut blind = state.cache.clone();
    let rows = blind.row_dims.len();
    blind.deflated_row_directions = std::sync::Arc::from(vec![Vec::new(); rows]);
    blind.deflation_row_spectra = std::sync::Arc::from(vec![None; rows]);
    let blind_reference = state
        .term
        .logdet_theta_adjoint_from_probes(
            &state.rho,
            &blind,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
            None,
        )
        .expect("deflation-blind from-probes theta-adjoint");
    let (separation, _) = adjoint_gap(&reference, &blind_reference);
    eprintln!(
        "#2333 CLAMP_BASIN_TRACE_ADJOINT rows={:?} scale={scale:.6e} parity={parity:.6e} \
         deflation_separation={separation:.6e}",
        state.rows
    );
    assert!(
        scale > 1.0e-6,
        "#2333: a vanishing theta-adjoint cannot establish parity; scale={scale:e}"
    );
    assert!(
        parity <= 1.0e-10,
        "#2333: the resident Trace theta-adjoint must equal its from-probes arbiter on \
         clamp-basin rows: worst relative entry gap {parity:e}"
    );
    assert!(
        separation > 1.0e-10 && separation > 1.0e3 * parity,
        "#2333: the parity bar must reject a deflation-blind adjoint on this fixture: \
         separation={separation:e} parity={parity:e}"
    );
}

/// #2333 — the whitening pre-fold for independent-logistic gates.
///
/// Under a row-varying full-rank likelihood metric the resident route projects
/// the four semantic output bases into each row's metric chart before the seam
/// builds the tower, while the from-probes arbiter whitens every materialized
/// channel on the host. Both the threshold gate and the ordered Beta–Bernoulli
/// gate run here; the latter also carries the empirical-mass column pass, which
/// the resident route weights by the folded diagonal `E_tt[a,a]`. Non-vacuity is
/// the measured separation from the resident adjoint with the metric removed.
#[test]
fn independent_gate_trace_theta_adjoint_whitens_like_from_probes_2333() {
    for mode in [
        AssignmentMode::threshold_gate(1.0, 0.0),
        AssignmentMode::ordered_beta_bernoulli(1.0, 0.9, false),
    ] {
        let (mut term, target, fixture_rho) = threshold_gate_tiny_fixture(true);
        term.assignment.mode = mode;
        let rho = fixture_rho.for_assignment(mode);
        let (n, p) = (term.n_obs(), term.output_dim());
        assert_eq!(p, 3, "#2333 the metric cell below is 3x3");
        let cell = [
            [1.05_f64, 0.07, -0.03],
            [-0.04, 0.90, 0.06],
            [0.02, -0.05, 1.15],
        ];
        let drift_cell = [
            [0.08_f64, 0.0, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.0, -0.07],
        ];
        let factors = Array2::<f64>::from_shape_fn((n, p * p), |(row, col)| {
            let (out_col, rank_col) = (col / p, col % p);
            let drift = row as f64 / (n - 1) as f64;
            cell[out_col][rank_col] + drift * drift_cell[out_col][rank_col]
        });
        term.set_row_metric(
            gam_problem::RowMetric::behavioral_fisher(std::sync::Arc::new(factors), p, p)
                .expect("#2333 row metric"),
        )
        .expect("#2333 row metric installs");
        assert!(
            term.whiten_logdet_row_jets(),
            "#2333 the metric must engage row whitening, else the pre-fold is untested"
        );
        if let AssignmentMode::OrderedBetaBernoulli { .. } = mode {
            let channels = crate::assignment::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                &term.assignment,
                &rho,
                term.row_loss_weights.as_deref(),
            )
            .expect("#2333 ordered Beta-Bernoulli majorizer channels")
            .expect("#2333 the ordered Beta-Bernoulli mode builds its shared-mass channels");
            let mass_coupling = channels
                .m_channel
                .iter()
                .zip(channels.z_jac.iter())
                .map(|(m, z)| (m * z).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                mass_coupling > 1.0e-8,
                "#2333 the empirical-mass column pass must be live on this fixture; \
                 max |m_channel * z_jac| = {mass_coupling:e}"
            );
        }
        let mut system = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("#2333 arrow assembly under the row metric");
        SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut system);
        let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
        let (_delta_t, _delta_beta, cache) =
            solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
                .expect("#2333 spectrally conditioned evidence factor");
        let solver = DeflatedArrowSolver::plain(&cache);
        let resident = term
            .logdet_theta_adjoint(&rho, &cache, &solver)
            .expect("#2333 resident Trace theta-adjoint");
        let (probes, sinv) = full_basis_bundle(&cache);
        let reference = term
            .logdet_theta_adjoint_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::Majorizer,
                None,
            )
            .expect("#2333 from-probes theta-adjoint");
        let (parity, scale) = adjoint_gap(&reference, &resident);
        let mut unwhitened = term.clone();
        unwhitened.row_metric = None;
        let counterfactual = unwhitened
            .logdet_theta_adjoint(&rho, &cache, &solver)
            .expect("#2333 resident theta-adjoint with the row metric removed");
        let (separation, _) = adjoint_gap(&reference, &counterfactual);
        let live_rows = (0..cache.row_dims.len())
            .filter(|&row| {
                SaeManifoldTerm::row_deflation_is_live(
                    cache
                        .deflated_row_directions
                        .get(row)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                    cache.deflation_row_spectra.get(row).and_then(Option::as_ref),
                )
            })
            .count();
        eprintln!(
            "#2333 INDEPENDENT_TRACE_WHITENING mode={mode:?} live_rows={live_rows} \
             scale={scale:.6e} parity={parity:.6e} whitening_separation={separation:.6e}"
        );
        assert!(
            scale > 1.0e-6,
            "#2333 ({mode:?}): a vanishing theta-adjoint cannot establish parity; scale={scale:e}"
        );
        assert!(
            parity <= 1.0e-10,
            "#2333 ({mode:?}): the resident Trace theta-adjoint must equal its from-probes \
             arbiter under row whitening: worst relative entry gap {parity:e}"
        );
        assert!(
            separation > 1.0e-10 && separation > 1.0e3 * parity,
            "#2333 ({mode:?}): the parity bar must reject an unwhitened adjoint on this \
             fixture: separation={separation:e} parity={parity:e}"
        );
    }
}
