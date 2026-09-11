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
