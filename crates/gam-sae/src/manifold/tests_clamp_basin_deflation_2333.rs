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
    target: Array2<f64>,
}

/// A clone of `anchor` that keeps its three per-assembly gates and its freeze
/// flag. `SaeManifoldTerm::clone` drops the gates, so a finite-difference
/// endpoint built from a plain clone re-derives them from its own state, a
/// motion production holds fixed across a step. A fresh fixture carries no
/// gates and is not frozen, so its first assembly derives them.
fn frozen_gate_endpoint(anchor: &SaeManifoldTerm) -> SaeManifoldTerm {
    let mut endpoint = anchor.clone();
    endpoint.decoder_repulsion_gate = anchor.decoder_repulsion_gate.clone();
    endpoint.barrier_coactivation_gate = anchor.barrier_coactivation_gate.clone();
    endpoint.amplitude_barrier_gate = anchor.amplitude_barrier_gate;
    endpoint.streaming_gates_frozen = anchor.streaming_gates_frozen;
    endpoint
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
    let mut anchor = frozen_gate_endpoint(term);
    let mut majorizer = anchor.assemble_arrow_schur(target.view(), rho, None)?;
    SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut majorizer);
    let exact = anchor.exact_a_evidence_system(target.view(), rho, &majorizer, 1.0)?;
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

/// A factor's discrete stratum: the rows pricing a clamp basin, the deflated
/// direction count, and the reduced-Schur conditioning pattern. A central
/// difference arbitrates only between endpoints on the anchor's stratum.
fn factor_stratum(cache: &ArrowFactorCache) -> (Vec<usize>, usize, String) {
    let directions: usize = cache.deflated_row_directions.iter().map(Vec::len).sum();
    let schur = cache
        .beta_schur_conditioning
        .as_ref()
        .map_or_else(String::new, |spectrum| format!("{:?}", spectrum.conditioning));
    (clamp_basin_rows(cache), directions, schur)
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
    let base_rho = fixture_rho.for_assignment(&term.assignment);
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
                        target: target.clone(),
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

/// The first rung of a declared `(log_ard, log_lambda_sparse shift)` ladder whose
/// exact-A factor records a clamp-basin row, for a mode that does not factor on the
/// ARD-only ladder of [`first_clamp_basin_state`]. The census of every rung prints
/// unconditionally.
fn first_clamp_basin_state_on_ladder(
    mode: AssignmentMode,
    straddle: bool,
    redraw_target: bool,
    ladder: &[(f64, f64)],
) -> ClampBasinState {
    let (mut term, logistic_target, fixture_rho) = threshold_gate_tiny_fixture(straddle);
    term.assignment.mode = mode;
    let base_rho = fixture_rho.for_assignment(&term.assignment);
    // The fixture draws its target under independent logistic gates. `redraw_target`
    // re-draws it under `mode`'s own assignments at the fixture state, so the
    // residual curvature of the exact-A rows vanishes there.
    let target = if redraw_target {
        let (n, p, k_atoms) = (term.n_obs(), term.output_dim(), term.k_atoms());
        let mut redrawn = Array2::<f64>::zeros((n, p));
        let mut assignments = vec![0.0_f64; k_atoms];
        let mut decoded = vec![0.0_f64; p];
        for row in 0..n {
            term.assignment
                .try_assignments_row_into(row, &mut assignments)
                .expect("#2333 assignments at the fixture state");
            for atom in 0..k_atoms {
                term.atoms[atom].fill_decoded_row(row, &mut decoded);
                for col in 0..p {
                    redrawn[[row, col]] += assignments[atom] * decoded[col];
                }
            }
        }
        redrawn
    } else {
        logistic_target
    };
    let mut census = Vec::new();
    for &(log_ard, sparse_shift) in ladder {
        let mut rho = base_rho.clone();
        rho.log_lambda_sparse += sparse_shift;
        for axes in rho.log_ard.iter_mut() {
            axes.fill(log_ard);
        }
        match exact_a_evidence_cache(&term, &target, &rho) {
            Ok((anchor, cache)) => {
                let rows = clamp_basin_rows(&cache);
                census.push(format!(
                    "log_ard={log_ard} sparse_shift={sparse_shift}: stratum={:?}",
                    factor_stratum(&cache)
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
                        target: target.clone(),
                    };
                }
            }
            Err(err) => census.push(format!(
                "log_ard={log_ard} sparse_shift={sparse_shift}: no factor: {err}"
            )),
        }
    }
    panic!(
        "#2333 premise: no rung of the declared ladder produced a clamp-basin row:\n{}",
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

/// #2915 — on the matrix-free lane a threshold gate's sparse log-det trace must
/// differentiate the exact observed information the lane's value prices, on every
/// rung of a declared ladder that factors.
///
/// The value is `½ arrow_log_det` of the exact-A evidence factor, and its central
/// difference in `log_lambda_sparse` over frozen θ̂, on the rung's own discrete
/// stratum, is the arbiter. Since b5bf5a390 the arrow rows carry the gate's concave
/// remainder, so a row with a switched-on logit prices a clamp basin, and no rung of
/// this straddling gate is basin-free (job 603114: all ten rows on every rung). The
/// rungs differ instead in how many reduced-Schur directions they price as basins
/// (job 603114: 3, 1, 1, 1, 3, 2 and 4), which is what this ladder adds over the
/// single rung of `clamp_basin_price_derivative_matches_the_lane_value_2915`. The
/// majorizer-operator trace on the same factor differentiates `B` and is the
/// positive control, because a straddling gate carries a nonzero remainder `ΔC` on
/// every switched-on logit.
#[test]
fn threshold_gate_exact_sparse_logdet_trace_matches_the_lane_value_2915() {
    let (term, target, fixture_rho) = threshold_gate_tiny_fixture(true);
    let mut census = Vec::new();
    let mut measured = 0usize;
    let base_sparse = fixture_rho.log_lambda_sparse;
    for (log_ard, log_lambda_sparse) in [
        (-1.0_f64, base_sparse),
        (-2.0, base_sparse),
        (-3.0, base_sparse),
        (-4.0, base_sparse),
        (-6.0, base_sparse),
        (-6.0, base_sparse - 1.0),
        (-6.0, base_sparse - 2.0),
    ] {
        let mut rho = fixture_rho.clone();
        rho.log_lambda_sparse = log_lambda_sparse;
        for axes in rho.log_ard.iter_mut() {
            axes.fill(log_ard);
        }
        let (anchor, cache) = match exact_a_evidence_cache(&term, &target, &rho) {
            Ok(factored) => factored,
            Err(err) => {
                census.push(format!(
                    "log_ard={log_ard} log_lambda_sparse={log_lambda_sparse}: no factor: {err}"
                ));
                continue;
            }
        };
        let (probes, sinv) = full_basis_bundle(&cache);
        let exact = anchor
            .assignment_log_strength_hessian_trace_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::ExactObservedInformation,
            )
            .expect("#2915 exact-operator sparse trace");
        let majorizer = anchor
            .assignment_log_strength_hessian_trace_from_probes(
                &rho,
                &cache,
                &probes,
                &sinv,
                EvidenceOperator::Majorizer,
            )
            .expect("#2915 majorizer-operator sparse trace");
        let anchor_stratum = factor_stratum(&cache);
        let half_log_det = |log_lambda_sparse: f64| -> f64 {
            let mut moved = rho.clone();
            moved.log_lambda_sparse = log_lambda_sparse;
            let (_, endpoint) = exact_a_evidence_cache(&anchor, &target, &moved)
                .expect("#2915 exact-A factor at a finite-difference endpoint");
            assert_eq!(
                factor_stratum(&endpoint),
                anchor_stratum,
                "#2915: both endpoints must sit on the rung's discrete stratum"
            );
            0.5 * endpoint
                .arrow_log_det()
                .expect("#2915 authoritative joint log-det of the exact-A factor")
        };
        let h = 1.0e-5;
        let fd = (half_log_det(rho.log_lambda_sparse + h) - half_log_det(rho.log_lambda_sparse - h))
            / (2.0 * h);
        let bar = 1.0e-6 * (1.0 + fd.abs());
        eprintln!(
            "#2915 LANE_SPARSE_TRACE log_ard={log_ard} log_lambda_sparse={log_lambda_sparse} \
             stratum={anchor_stratum:?} fd={fd:.12e} exact={exact:.12e} \
             majorizer={majorizer:.12e} bar={bar:.3e}"
        );
        census.push(format!(
            "log_ard={log_ard} log_lambda_sparse={log_lambda_sparse}: fd={fd:e} exact={exact:e} \
             majorizer={majorizer:e}"
        ));
        assert!(
            (exact - fd).abs() <= bar,
            "#2915: the exact-operator sparse log-det trace must be the central difference of \
             the lane value at log_ard={log_ard} log_lambda_sparse={log_lambda_sparse}: \
             exact={exact:e} fd={fd:e} bar={bar:e}"
        );
        assert!(
            (majorizer - fd).abs() > 1.0e3 * bar,
            "#2915 positive control: the majorizer-operator trace must miss the remainder on a \
             straddling gate: majorizer={majorizer:e} fd={fd:e} bar={bar:e}"
        );
        measured += 1;
    }
    eprintln!("#2915 LANE_SPARSE_TRACE census\n{}", census.join("\n"));
    assert!(
        measured > 0,
        "#2915 premise: no rung of the declared ladder factored:\n{}",
        census.join("\n")
    );
}

/// #2915 — the clamp-basin price's own derivative on the matrix-free lane.
///
/// On a threshold-gate state whose exact-A evidence factor prices clamp-basin
/// rows, every exact-operator from-probes channel must be the central difference
/// of the lane value on one discrete stratum: the sparse log-strength trace and
/// one ARD precision trace against `½ arrow_log_det` over frozen θ̂, and the
/// θ-adjoint entry of a logit on a clamp-basin row against `arrow_log_det` at
/// frozen ρ. The premise requires a material basin price, so the clamp leg is
/// live on the rows the gate measures. Each gap prints before any assertion.
#[test]
fn clamp_basin_price_derivative_matches_the_lane_value_2915() {
    let mode = AssignmentMode::threshold_gate(1.0, 0.0);
    let state = first_clamp_basin_state(mode, true);
    let (_, target, _) = threshold_gate_tiny_fixture(true);
    let basin_price = state
        .rows
        .iter()
        .filter_map(|&row| state.cache.deflation_row_spectra[row].as_ref())
        .flat_map(|spectrum| {
            spectrum
                .raw_evals
                .iter()
                .zip(spectrum.cond_evals.iter())
                .zip(spectrum.conditioning.iter())
                .filter(|((raw, _), conditioning)| {
                    **raw < 0.0 && **conditioning == RowSpectralConditioning::Raw
                })
                .map(|((raw, priced), _)| priced - raw)
                .collect::<Vec<f64>>()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        basin_price > 1.0e-3,
        "#2915 premise: the clamp-basin rows {:?} must carry a material price; largest \
         v'Ev = {basin_price:e}",
        state.rows
    );
    let (probes, sinv) = full_basis_bundle(&state.cache);
    let operator = EvidenceOperator::ExactObservedInformation;
    let anchor_stratum = factor_stratum(&state.cache);
    let log_det_at = |moved_term: &SaeManifoldTerm, moved_rho: &SaeManifoldRho| -> f64 {
        let (_, endpoint) = exact_a_evidence_cache(moved_term, &target, moved_rho)
            .expect("#2915 exact-A factor at a finite-difference endpoint");
        assert_eq!(
            factor_stratum(&endpoint),
            anchor_stratum,
            "#2915: both endpoints must sit on the anchor's clamp-basin stratum"
        );
        endpoint
            .arrow_log_det()
            .expect("#2915 authoritative joint log-det of the exact-A factor")
    };
    let h = 1.0e-5;

    let sparse = state
        .term
        .assignment_log_strength_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
        )
        .expect("#2915 exact-operator sparse trace");
    let sparse_fd = {
        let mut plus = state.rho.clone();
        let mut minus = state.rho.clone();
        plus.log_lambda_sparse += h;
        minus.log_lambda_sparse -= h;
        0.5 * (log_det_at(&state.term, &plus) - log_det_at(&state.term, &minus)) / (2.0 * h)
    };

    let ard_atom = (0..state.rho.log_ard.len())
        .find(|&atom| !state.rho.log_ard[atom].is_empty())
        .expect("#2915 premise: the fixture must carry an ARD precision");
    let ard = state
        .term
        .ard_log_precision_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
        )
        .expect("#2915 exact-operator ARD trace")[ard_atom][0];
    let ard_fd = {
        let mut plus = state.rho.clone();
        let mut minus = state.rho.clone();
        plus.log_ard[ard_atom][0] += h;
        minus.log_ard[ard_atom][0] -= h;
        0.5 * (log_det_at(&state.term, &plus) - log_det_at(&state.term, &minus)) / (2.0 * h)
    };

    let row = state.rows[0];
    let variables = state
        .term
        .row_vars_for_cache_row(row, &state.cache)
        .expect("#2915 row variables");
    let (position, atom) = variables
        .iter()
        .enumerate()
        .find_map(|(position, variable)| match *variable {
            SaeLocalRowVar::Logit { atom } => Some((position, atom)),
            SaeLocalRowVar::Coord { .. } => None,
        })
        .expect("#2915 premise: a threshold-gate row carries a free logit");
    let theta = state
        .term
        .logdet_theta_adjoint_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
            Some(target.view()),
        )
        .expect("#2915 exact-operator theta-adjoint")
        .t[state.cache.row_offsets[row] + position];
    let theta_fd = {
        let mut plus = frozen_gate_endpoint(&state.term);
        let mut minus = frozen_gate_endpoint(&state.term);
        plus.assignment.logits[[row, atom]] += h;
        minus.assignment.logits[[row, atom]] -= h;
        (log_det_at(&plus, &state.rho) - log_det_at(&minus, &state.rho)) / (2.0 * h)
    };

    let rho_bar = |fd: f64| 1.0e-6 * (1.0 + fd.abs());
    let theta_bar = 1.0e-5 * (1.0 + theta_fd.abs());
    eprintln!(
        "#2915 CLAMP_BASIN_PRICE rows={:?} basin_price={basin_price:.6e}\n  sparse={sparse:.12e} \
         fd={sparse_fd:.12e} gap={:.3e}\n  ard[{ard_atom},0]={ard:.12e} fd={ard_fd:.12e} \
         gap={:.3e}\n  theta[row {row}, logit {atom}]={theta:.12e} fd={theta_fd:.12e} gap={:.3e}",
        state.rows,
        (sparse - sparse_fd).abs(),
        (ard - ard_fd).abs(),
        (theta - theta_fd).abs()
    );
    assert!(
        (sparse - sparse_fd).abs() <= rho_bar(sparse_fd),
        "#2915: the exact-operator sparse trace must differentiate the lane value on \
         clamp-basin rows: trace={sparse:e} fd={sparse_fd:e}"
    );
    assert!(
        (ard - ard_fd).abs() <= rho_bar(ard_fd),
        "#2915: the exact-operator ARD trace must differentiate the lane value on \
         clamp-basin rows: trace={ard:e} fd={ard_fd:e}"
    );
    assert!(
        (theta - theta_fd).abs() <= theta_bar,
        "#2915: the exact-operator theta-adjoint must differentiate the lane value on a \
         clamp-basin row: adjoint={theta:e} fd={theta_fd:e}"
    );
}

/// #2914 — the dense and from-probes ARD log-precision traces on clamp-basin rows.
///
/// Both siblings gate their Daleckii–Krein correction on `row_deflation_is_live`,
/// and neither had an arbiter on a row whose spectrum is recorded while its
/// direction list is empty. On the first clamp-basin rung they must agree per
/// entry. The from-probes trace on a deflation-blind copy of the same cache must
/// separate from them, or the bar would also accept the direction-list convention.
#[test]
fn ard_trace_dense_and_probes_agree_on_clamp_basin_rows_2914() {
    let state = first_clamp_basin_state(AssignmentMode::threshold_gate(1.0, 0.0), true);
    let solver = DeflatedArrowSolver::plain(&state.cache);
    let dense = state
        .term
        .ard_log_precision_hessian_trace(
            &state.rho,
            &state.cache,
            &solver,
            EvidenceOperator::Majorizer,
        )
        .expect("#2914 dense ARD trace on the exact-A factor");
    let (probes, sinv) = full_basis_bundle(&state.cache);
    let from_probes = state
        .term
        .ard_log_precision_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("#2914 from-probes ARD trace on the exact-A factor");
    let mut blind = state.cache.clone();
    let rows = blind.row_dims.len();
    blind.deflated_row_directions = std::sync::Arc::from(vec![Vec::new(); rows]);
    blind.deflation_row_spectra = std::sync::Arc::from(vec![None; rows]);
    let blind_probes = state
        .term
        .ard_log_precision_hessian_trace_from_probes(
            &state.rho,
            &blind,
            &probes,
            &sinv,
            EvidenceOperator::Majorizer,
        )
        .expect("#2914 deflation-blind from-probes ARD trace");
    assert_eq!(dense.len(), from_probes.len());
    let mut gap = 0.0_f64;
    let mut separation = 0.0_f64;
    let mut scale = 0.0_f64;
    for atom in 0..dense.len() {
        assert_eq!(dense[atom].len(), from_probes[atom].len());
        for axis in 0..dense[atom].len() {
            let reference = dense[atom][axis];
            let probed = from_probes[atom][axis];
            gap = gap.max((reference - probed).abs() / (1.0 + reference.abs()));
            separation = separation
                .max((probed - blind_probes[atom][axis]).abs() / (1.0 + probed.abs()));
            scale = scale.max(reference.abs());
        }
    }
    eprintln!(
        "#2914 ARD_TRACE_CLAMP_BASIN rows={:?} scale={scale:.6e} gap={gap:.6e} \
         deflation_separation={separation:.6e}",
        state.rows
    );
    assert!(
        scale > 1.0e-6,
        "#2914: a vanishing ARD trace cannot establish agreement; scale={scale:e}"
    );
    assert!(
        gap <= 1.0e-9,
        "#2914: the dense and from-probes ARD traces must agree on clamp-basin rows: worst \
         relative entry gap {gap:e}"
    );
    assert!(
        separation > 1.0e-9 && separation > 1.0e3 * gap,
        "#2914: the agreement bar must reject a deflation-blind trace on this fixture: \
         separation={separation:e} gap={gap:e}"
    );
}

/// #2913 — the Softmax lane on clamp-basin rows, against the lane value.
///
/// A Softmax row writes no clamp on its logits, so its clamp-basin rows come from
/// the periodic ARD prior's concave half on coordinate slots, and its reduced Schur
/// prices basins through that clamp. On the first rung of the declared ladder whose
/// exact-A factor prices such a row, every exact-operator from-probes channel must
/// be the central difference of the lane value on the anchor's stratum: the sparse
/// log-strength trace and one ARD precision trace against `½ arrow_log_det` over
/// frozen θ̂, and the θ-adjoint of every latent variable of a clamp-basin row against
/// `arrow_log_det` at frozen ρ. Parity between `logdet_theta_adjoint_dense` and the
/// from-probes adjoint on one cache cannot see a convention both share, as the #2915
/// numbers showed for the threshold gate, so the arbiter is the value. Every gap
/// prints before the assertion.
#[test]
fn softmax_lane_channels_match_the_lane_value_on_clamp_basin_rows_2913() {
    // Job 609221: in Softmax mode no rung of the fixture's logistic-target state
    // factors (PerRowFactorFailed on row 3). Census job 629295: once the target is
    // re-drawn under the Softmax assignments, the residual curvature vanishes, and at a
    // lower assignment strength the rungs price clamp-basin rows (log_ard = −3 at
    // sparse shift −4: rows [1, 4, 6] with 3 reduced-Schur basins).
    let state = first_clamp_basin_state_on_ladder(
        AssignmentMode::softmax(1.0),
        true,
        true,
        &[
            (-3.0, -4.0),
            (-2.0, -4.0),
            (-1.0, -4.0),
            (0.0, -4.0),
            (-1.0, -2.0),
            (0.0, -2.0),
        ],
    );
    let target = state.target.clone();
    let (probes, sinv) = full_basis_bundle(&state.cache);
    let operator = EvidenceOperator::ExactObservedInformation;
    let anchor_stratum = factor_stratum(&state.cache);
    let log_det_at = |moved_term: &SaeManifoldTerm, moved_rho: &SaeManifoldRho| -> f64 {
        let (_, endpoint) = exact_a_evidence_cache(moved_term, &target, moved_rho)
            .expect("#2913 exact-A factor at a finite-difference endpoint");
        assert_eq!(
            factor_stratum(&endpoint),
            anchor_stratum,
            "#2913: both endpoints must sit on the anchor's stratum"
        );
        endpoint
            .arrow_log_det()
            .expect("#2913 authoritative joint log-det of the exact-A factor")
    };
    let h = 1.0e-5;
    let mut report = Vec::new();
    let mut failures = Vec::new();

    let sparse = state
        .term
        .assignment_log_strength_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
        )
        .expect("#2913 exact-operator sparse trace");
    let sparse_fd = {
        let mut plus = state.rho.clone();
        let mut minus = state.rho.clone();
        plus.log_lambda_sparse += h;
        minus.log_lambda_sparse -= h;
        0.5 * (log_det_at(&state.term, &plus) - log_det_at(&state.term, &minus)) / (2.0 * h)
    };
    report.push(format!(
        "sparse={sparse:.12e} fd={sparse_fd:.12e} gap={:.3e}",
        (sparse - sparse_fd).abs()
    ));
    if (sparse - sparse_fd).abs() > 1.0e-6 * (1.0 + sparse_fd.abs()) {
        failures.push("sparse".to_string());
    }

    let ard_atom = (0..state.rho.log_ard.len())
        .find(|&atom| !state.rho.log_ard[atom].is_empty())
        .expect("#2913 premise: the fixture must carry an ARD precision");
    let ard = state
        .term
        .ard_log_precision_hessian_trace_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
        )
        .expect("#2913 exact-operator ARD trace")[ard_atom][0];
    let ard_fd = {
        let mut plus = state.rho.clone();
        let mut minus = state.rho.clone();
        plus.log_ard[ard_atom][0] += h;
        minus.log_ard[ard_atom][0] -= h;
        0.5 * (log_det_at(&state.term, &plus) - log_det_at(&state.term, &minus)) / (2.0 * h)
    };
    report.push(format!(
        "ard[{ard_atom},0]={ard:.12e} fd={ard_fd:.12e} gap={:.3e}",
        (ard - ard_fd).abs()
    ));
    if (ard - ard_fd).abs() > 1.0e-6 * (1.0 + ard_fd.abs()) {
        failures.push("ard".to_string());
    }

    let theta = state
        .term
        .logdet_theta_adjoint_from_probes(
            &state.rho,
            &state.cache,
            &probes,
            &sinv,
            operator,
            Some(target.view()),
        )
        .expect("#2913 exact-operator theta-adjoint");
    let row = state.rows[0];
    let variables = state
        .term
        .row_vars_for_cache_row(row, &state.cache)
        .expect("#2913 row variables");
    for (position, variable) in variables.iter().enumerate() {
        let (label, endpoint) = match *variable {
            SaeLocalRowVar::Logit { atom } => (format!("logit {atom}"), (atom, None)),
            SaeLocalRowVar::Coord { atom, axis } => {
                (format!("coord {atom}.{axis}"), (atom, Some(axis)))
            }
        };
        let moved = |shift: f64| {
            let mut term = frozen_gate_endpoint(&state.term);
            match endpoint {
                (atom, None) => term.assignment.logits[[row, atom]] += shift,
                (atom, Some(axis)) => {
                    let index = row * term.assignment.coords[atom].latent_dim() + axis;
                    let mut flat = term.assignment.coords[atom].as_flat().clone();
                    flat[index] += shift;
                    term.assignment.coords[atom].set_flat(flat.view());
                    // The atoms cache their basis at the coordinates they were last
                    // refreshed at, and the arrow assembly reads that cache, so a moved
                    // coordinate must refresh it or the endpoint moves only the ARD
                    // prior (production refreshes after every coordinate write).
                    term.refresh_basis_from_current_coords()
                        .expect("#2913 basis refresh at a coordinate endpoint");
                }
            }
            term
        };
        let adjoint = theta.t[state.cache.row_offsets[row] + position];
        // Job 656665: this state's θ entries reach 1e3, and a single central difference
        // at h = 1e-5 carries 1.4e-5 to 5.6e-5 relative truncation. Richardson over
        // (2h, h) removes the O(h²) term, and on every correct entry it agreed with
        // production to 1e-10..3e-8.
        let central = |step: f64| {
            (log_det_at(&moved(step), &state.rho) - log_det_at(&moved(-step), &state.rho))
                / (2.0 * step)
        };
        let fd = (4.0 * central(h) - central(2.0 * h)) / 3.0;
        report.push(format!(
            "theta[row {row}, {label}]={adjoint:.12e} richardson={fd:.12e} gap={:.3e}",
            (adjoint - fd).abs()
        ));
        if (adjoint - fd).abs() > 1.0e-6 * (1.0 + fd.abs()) {
            failures.push(format!("theta {label}"));
        }
    }
    eprintln!(
        "#2913 SOFTMAX_CLAMP_BASIN rows={:?}\n  {}",
        state.rows,
        report.join("\n  ")
    );
    assert!(
        failures.is_empty(),
        "#2913: the exact-operator lane channels must differentiate the lane value on Softmax \
         clamp-basin rows; failing: {failures:?}\n  {}",
        report.join("\n  ")
    );
}
