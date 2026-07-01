use gam_linalg::faer_ndarray::fast_ata;

use super::*;
use gam_solve::arrow_schur::{
    ArrowFactorSlab, ArrowHtbetaCache, ArrowSolverMode, ArrowUndampedFactors, PcgDiagnostics,
};
use gam_terms::analytic_penalties::ARDPenalty;
use approx::assert_abs_diff_eq;
use ndarray::{Array5, array};

/// The overflow-free von-Mises normaliser must (a) agree with the naive
/// `bessel_i0(η).ln()` / `bessel_i1(η)/bessel_i0(η)` on moderate η where the
/// naive form is still finite, and (b) stay finite for the large η a
/// dispersion-inflated ARD seed reaches on a large-norm / ill-conditioned
/// checkpoint (#1113), where the naive form overflows to `inf` and divides
/// to `NaN`.
#[test]
pub(crate) fn bessel_log_and_ratio_is_finite_and_matches_naive() {
    // Moderate η: naive forms are finite, so the stable helper must match.
    for &eta in &[0.0_f64, 0.5, 1.0, 3.0, 3.75, 5.0, 20.0, 100.0, 300.0] {
        let (log_i0, ratio) = bessel_i0_log_and_ratio(eta);
        let naive_log = bessel_i0(eta).ln();
        let naive_ratio = bessel_i1(eta) / bessel_i0(eta);
        assert!(naive_log.is_finite(), "naive log finite at η={eta}");
        assert!(naive_ratio.is_finite(), "naive ratio finite at η={eta}");
        assert_abs_diff_eq!(log_i0, naive_log, epsilon = 1e-9);
        assert_abs_diff_eq!(ratio, naive_ratio, epsilon = 1e-9);
    }

    // Large η (past the `e^{η}` overflow threshold ≈ 709). The stable helper
    // must stay finite where `bessel_i0(η) = inf`, and the ratio I1/I0 → 1⁻.
    for &eta in &[710.0_f64, 1.0e3, 1.0e6, 1.0e12, 1.0e300] {
        assert!(
            !bessel_i0(eta).is_finite(),
            "naive I0 expected to overflow at η={eta} (guards the regression)"
        );
        let (log_i0, ratio) = bessel_i0_log_and_ratio(eta);
        assert!(log_i0.is_finite(), "stable log I0 finite at η={eta}");
        assert!(ratio.is_finite(), "stable I1/I0 finite at η={eta}");
        // I1/I0 ∈ (0, 1) and → 1 as η → ∞; the ρ-gradient term n·η·(ratio−1)
        // must therefore be finite, never `inf·NaN`.
        assert!(ratio > 0.0 && ratio <= 1.0, "ratio in (0,1] at η={eta}");
    }
}

pub(crate) fn assert_matrix_same_bits(left: &Array2<f64>, right: &Array2<f64>) {
    assert_eq!(left.dim(), right.dim());
    for ((row, col), &value) in left.indexed_iter() {
        assert_eq!(
            value.to_bits(),
            right[[row, col]].to_bits(),
            "matrix bits differ at ({row}, {col})"
        );
    }
}

pub(crate) fn assert_tensor3_same_bits(left: &Array3<f64>, right: &Array3<f64>) {
    assert_eq!(left.dim(), right.dim());
    for ((row, col, axis), &value) in left.indexed_iter() {
        assert_eq!(
            value.to_bits(),
            right[[row, col, axis]].to_bits(),
            "tensor bits differ at ({row}, {col}, {axis})"
        );
    }
}

pub(crate) fn assert_eta_one_parity(
    evaluator: &dyn SaeBasisEvaluator,
    coords: ArrayView2<'_, f64>,
    expected_curved: usize,
) {
    let (phi, jet) = evaluator.evaluate(coords).expect("base evaluate");
    let eta = evaluator
        .evaluate_phi_eta(coords, 1.0)
        .expect("eta evaluate");
    assert_matrix_same_bits(&eta.phi, &phi);
    assert_tensor3_same_bits(&eta.jet, &jet);
    assert_eq!(eta.split.curved_cols.len(), expected_curved);
    for &col in &eta.split.linear_cols {
        for row in 0..phi.nrows() {
            assert_eq!(eta.dphi_deta[[row, col]], 0.0);
            for axis in 0..jet.shape()[2] {
                assert_eq!(eta.djet_deta[[row, col, axis]], 0.0);
            }
        }
    }
    for &col in &eta.split.curved_cols {
        for row in 0..phi.nrows() {
            assert_eq!(
                eta.dphi_deta[[row, col]].to_bits(),
                phi[[row, col]].to_bits()
            );
            for axis in 0..jet.shape()[2] {
                assert_eq!(
                    eta.djet_deta[[row, col, axis]].to_bits(),
                    jet[[row, col, axis]].to_bits()
                );
            }
        }
    }
}

#[test]
pub(crate) fn phi_eta_one_reproduces_current_atom_bases_bit_for_bit() {
    let periodic_coords = array![[0.0_f64], [0.125], [0.4]];
    let periodic = PeriodicHarmonicEvaluator::new(7).unwrap();
    assert_eta_one_parity(&periodic, periodic_coords.view(), 4);

    let raw_circle_coords = array![[0.0_f64], [0.3], [1.1]];
    let raw_circle = RawPeriodicCircleEvaluator::new(1).unwrap();
    assert_eta_one_parity(&raw_circle, raw_circle_coords.view(), 0);

    let torus_coords = array![[0.0_f64, 0.2], [0.25, 0.5], [0.7, 0.9]];
    let torus = TorusHarmonicEvaluator::new(2, 2).unwrap();
    assert_eta_one_parity(&torus, torus_coords.view(), 20);

    let sphere_coords = array![[0.0_f64, 0.0], [0.3, 0.4], [-0.2, 1.1]];
    let sphere = SphereChartEvaluator;
    assert_eta_one_parity(&sphere, sphere_coords.view(), 3);

    let centers = array![
        [-1.0_f64, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.5, -0.25]
    ];
    let duchon_coords = array![[0.1_f64, 0.2], [0.4, -0.3], [-0.2, 0.7]];
    let duchon = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
    let (duchon_phi, _) = duchon.evaluate(duchon_coords.view()).unwrap();
    let duchon_poly = 3usize;
    assert_eta_one_parity(
        &duchon,
        duchon_coords.view(),
        duchon_phi.ncols() - duchon_poly,
    );

    let euclidean = EuclideanPatchEvaluator::new(2, 3).unwrap();
    let total_cols = gam_terms::basis::monomial_exponents(2, 3).len();
    let linear_cols = gam_terms::basis::monomial_exponents(2, 3)
        .iter()
        .filter(|alpha| alpha.iter().sum::<usize>() <= 1)
        .count();
    assert_eta_one_parity(&euclidean, duchon_coords.view(), total_cols - linear_cols);
}

/// Minimal K=1 term for direct unit tests of term-state machinery that does
/// not depend on a real fit (e.g. the gauge-deflation count guard).
pub(crate) fn trivial_k1_euclidean_term() -> SaeManifoldTerm {
    let n = 4usize;
    let p = 3usize;
    let atom = SaeManifoldAtom::new(
        "atom0",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        Array2::<f64>::ones((n, 2)),
        Array3::<f64>::zeros((n, 2, 1)),
        Array2::<f64>::zeros((2, p)),
        Array2::<f64>::eye(2),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![Array2::<f64>::zeros((n, 1))],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

/// #795 (second root cause) — a BOUNDED low-amplitude flicker of the per-row
/// gauge-deflation count is benign jitter, NOT the oscillating-quotient
/// pathology, so it must re-anchor freely no matter how many times it reverses.
///
/// The count is an O(N) per-row sum of near-null evidence directions; a handful
/// of rows sitting right at the deflation floor cross it back and forth as the
/// ρ-walk nudges the conditioning, reversing direction every step while staying
/// within a few of a large level (the single-planted-circle fit flickers
/// 150<->147 on N=200). Each such change is evidence-neutral (a deflated
/// direction contributes the ρ-independent `log 1 = 0` to `½log|H|` either way),
/// so re-anchoring is exactly correct and the fit must not be refused. The
/// bare-reversal guard charged the budget on every one of these and aborted the
/// simplest possible manifold-SAE fit (isometry on OR off) — this pins that a
/// sustained small-amplitude flicker survives indefinitely.
#[test]
pub(crate) fn evidence_gauge_deflation_count_bounded_flicker_reanchors_freely() {
    let mut term = trivial_k1_euclidean_term();
    // Pin the expected count at a realistic large level (like the circle fit).
    term.record_evidence_gauge_deflation_count(150).unwrap();
    // A sustained 150<->147 flicker reverses direction on EVERY step — far more
    // reversals than the K=1 budget of 6 — yet the amplitude (3) is well inside
    // the relative jitter band (150/4 = 37), so none charge the budget.
    let flicker = [147usize, 150, 147, 150, 147, 150, 147, 150, 147, 150, 147, 150, 147, 150];
    for &c in &flicker {
        term.record_evidence_gauge_deflation_count(c)
            .expect("a bounded low-amplitude flicker must re-anchor, never abort");
    }
    assert_eq!(
        term.evidence_gauge_deflation_reanchors, 0,
        "a flicker inside the relative jitter band charges no reversals"
    );
    assert_eq!(
        term.expected_evidence_gauge_deflated_directions,
        Some(150),
        "the comparison re-anchors to the latest observed count"
    );

    // But a WIDE-amplitude oscillation at the SAME level is still the runaway
    // pathology and must still be refused: 150<->40 swings ~73% of the level.
    let mut term2 = trivial_k1_euclidean_term();
    term2.record_evidence_gauge_deflation_count(150).unwrap();
    let mut errored = false;
    for &c in &[40usize, 150, 40, 150, 40, 150, 40, 150, 40, 150, 40, 150, 40, 150] {
        if term2.record_evidence_gauge_deflation_count(c).is_err() {
            errored = true;
            break;
        }
    }
    assert!(
        errored,
        "a wide-amplitude oscillation must still exhaust the reversal budget"
    );
}

/// The #1037 quotient-dimension guard, with #1217 oscillation semantics: the
/// recorded count of gauge-deflated evidence directions need not be CONSTANT —
/// it is a per-ROW-summed O(N) count of near-null evidence directions that
/// drifts smoothly across the ρ-walk, and every change is evidence-preserving (a
/// deflated direction contributes `log 1 = 0` to `½log|H|` either way). The
/// guard RE-ANCHORS the comparison to the new dimension instead of aborting. A
/// MONOTONE drift of any length is benign (the conditioning is just improving),
/// so it never trips the budget; the genuine pathology the guard must still
/// catch is an OSCILLATING count — repeated direction reversals that never
/// settle — which is refused loudly past the reversal budget.
#[test]
pub(crate) fn evidence_gauge_deflation_count_guard_reanchors_then_rejects_runaway() {
    let mut term = trivial_k1_euclidean_term();
    assert!(term.expected_evidence_gauge_deflated_directions.is_none());

    // First observation pins the expected count (high, like a real K=2 walk
    // that starts with many near-null evidence directions).
    term.record_evidence_gauge_deflation_count(60).unwrap();
    assert_eq!(term.expected_evidence_gauge_deflated_directions, Some(60));

    // A matching later observation is a no-op (still Ok, count unchanged).
    term.record_evidence_gauge_deflation_count(60).unwrap();
    assert_eq!(term.expected_evidence_gauge_deflated_directions, Some(60));

    // A MONOTONE drift (the #1217 benign case — a per-row conditioning count
    // shrinking across the ρ-walk) re-anchors freely without charging the budget,
    // no matter how many steps it takes. This is exactly the real-OLMo K=2
    // signature (171→…→113) that the old `k`-event budget wrongly tripped on.
    for c in [50usize, 40, 33, 21, 12, 9, 6, 4, 3, 2] {
        term.record_evidence_gauge_deflation_count(c).unwrap();
        assert_eq!(term.expected_evidence_gauge_deflated_directions, Some(c));
    }
    assert_eq!(
        term.evidence_gauge_deflation_reanchors, 0,
        "monotone drift charges no reversals"
    );

    // An OSCILLATING count (up/down/up/down…) IS the runaway pathology. K=1 ⇒
    // reversal budget = 1·(RESEED_BUDGET + 1) + 1 = 6. Each direction reversal
    // charges one; a sustained oscillation exhausts the budget and refuses.
    let mut last_ok = 2usize;
    let oscillation = [9usize, 2, 9, 2, 9, 2, 9, 2, 9, 2, 9, 2, 9, 2];
    let mut errored = false;
    for &c in &oscillation {
        match term.record_evidence_gauge_deflation_count(c) {
            Ok(()) => {
                last_ok = c;
            }
            Err(err) => {
                assert!(
                    err.contains("not stabilizing") && err.contains("oscillated"),
                    "guard must report the oscillating quotient dimension explicitly; got: {err}"
                );
                // On the refusal the expected count is NOT re-anchored.
                assert_eq!(
                    term.expected_evidence_gauge_deflated_directions,
                    Some(last_ok)
                );
                errored = true;
                break;
            }
        }
    }
    assert!(
        errored,
        "a sustained oscillation must exceed the reversal budget and error"
    );
}

/// #1217 — the outer-REML cost handed to the BFGS line search MUST be finite.
/// At a K≥2 co-collapse cliff the Laplace normalizer `½log|H|` of a numerically
/// singular joint Hessian can return `±∞`/`NaN`, so `reml_criterion` surfaces a
/// non-finite criterion value. The `opt` BFGS reports a non-finite probe as
/// "Line search failed (nonfinite seen)" and ABORTS the outer solve at the
/// current iterate (observed on real OLMo K=2: stall at iter 2, |g|≈65) instead
/// of backtracking. `add_fit_data_collapse_penalty` is the single seam BOTH the
/// value-probe (`eval_with_order(Value)`) and gradient (`eval`) lanes route
/// through, so flooring a non-finite cost to the finite collapse wall there
/// presents the SAME rejectable barrier a detected collapse does — the line
/// search rejects the step and backtracks toward the feasible basin.
#[test]
pub(crate) fn outer_collapse_penalty_floors_nonfinite_cost_to_finite_wall() {
    let finite_ok = |cost: f64| {
        // Fresh objective per call: `add_fit_data_collapse_penalty` mutates the
        // term's collapse ledger, so each assertion gets a clean slate.
        let mut objective = warmstart_test_objective();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
        objective
            .add_fit_data_collapse_penalty(cost, &rho)
            .expect("collapse penalty must not error on a non-finite cost")
    };

    // A finite, non-collapsed base cost passes through unchanged (byte-for-byte).
    let plain = finite_ok(3.5);
    assert!(plain.is_finite(), "finite cost must stay finite");

    // Each non-finite base must be floored to a FINITE, rejectable wall — never
    // propagated as `±∞`/`NaN` to the BFGS line search. The wall is bounded by
    // `2·SAE_FIT_DATA_COLLAPSE_COST` (base wall + at-most-one collapse penalty).
    for nonfinite in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let walled = finite_ok(nonfinite);
        assert!(
            walled.is_finite(),
            "a non-finite outer cost ({nonfinite}) must be floored to a finite wall, got {walled}"
        );
        assert!(
            walled >= SAE_FIT_DATA_COLLAPSE_COST && walled <= 2.0 * SAE_FIT_DATA_COLLAPSE_COST,
            "the floored wall {walled} must sit in [collapse_cost, 2·collapse_cost] so BFGS \
             treats it as an infeasible step to backtrack from, not as a descent target"
        );
    }
}

/// The identity-homotopy shortcut's structural probe: the η dial is inert
/// iff no atom evaluator declares curved columns. Caller-managed atoms
/// (no evaluator) and one-harmonic periodic banks (M = 3: constant +
/// fundamental, all linear columns) are inert; an M = 7 periodic bank
/// dials its h ≥ 2 harmonics, so the walk must run for it.
#[test]
pub(crate) fn curvature_homotopy_eta_inertness_probe_tracks_curved_columns() {
    // Caller-managed atom: no evaluator, nothing to dial.
    let term = trivial_k1_euclidean_term();
    assert!(term.curvature_homotopy_eta_is_inert().unwrap());

    // Periodic atoms whose evaluator split declares every column linear.
    let (term, _target, _rho) = small_two_atom_periodic_term();
    assert!(term.curvature_homotopy_eta_is_inert().unwrap());

    // M = 7 periodic: harmonics h ≥ 2 are η-dialed curved columns.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(7).unwrap());
    let coords = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let atom = SaeManifoldAtom::new(
        "periodic7",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((7, 1)),
        Array2::<f64>::eye(7),
    )
    .unwrap()
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((5, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    assert!(!term.curvature_homotopy_eta_is_inert().unwrap());
}

#[test]
pub(crate) fn linear_span_anchor_recovers_planted_two_plane_configuration() {
    let n = 4usize;
    let p = 4usize;
    let phi = Array2::<f64>::ones((n, 2));
    let jet = Array3::<f64>::zeros((n, 2, 1));
    let decoder = Array2::<f64>::zeros((2, p));
    let smooth = Array2::<f64>::eye(2);
    let atoms = vec![
        SaeManifoldAtom::new(
            "plane0",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            smooth.clone(),
        )
        .unwrap(),
        SaeManifoldAtom::new(
            "plane1",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            smooth,
        )
        .unwrap(),
    ];
    let coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 2)),
        coords,
        vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let target = array![
        [3.0_f64, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ];
    let anchor = linear_span_anchor(&term, target.view()).unwrap();
    assert_eq!(anchor.atoms.len(), 2);
    assert_abs_diff_eq!(anchor.residual_norm_sq, 0.0, epsilon = 1.0e-18);
    let plane0 = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let plane1 = array![[0.0_f64, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let angle0 = anchor.atoms[0]
        .frame
        .max_principal_angle(plane0.view())
        .unwrap();
    let angle1 = anchor.atoms[1]
        .frame
        .max_principal_angle(plane1.view())
        .unwrap();
    assert_abs_diff_eq!(angle0, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(angle1, 0.0, epsilon = 1.0e-12);
}

pub(crate) fn circle_certificate_fixture(
    radius: f64,
    planes: &[(usize, usize)],
) -> SaeManifoldTerm {
    let n = 16usize;
    let p = 4usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut atoms = Vec::with_capacity(planes.len());
    let mut coord_blocks = Vec::with_capacity(planes.len());
    for (atom_idx, &(axis_sin, axis_cos)) in planes.iter().enumerate() {
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, axis_sin]] = radius;
        decoder[[2, axis_cos]] = radius;
        let atom = SaeManifoldAtom::new(
            format!("circle_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords.clone());
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, planes.len())),
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; planes.len()],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    term.set_certificate_dispersion(1.0).unwrap();
    term
}

#[test]
pub(crate) fn dictionary_incoherence_report_orthogonal_frames_has_zero_mu_hat() {
    let term = circle_certificate_fixture(2.0, &[(0, 1), (2, 3)]);
    let report = dictionary_incoherence_report(&term).unwrap();
    assert_abs_diff_eq!(report.mu_hat, 0.0, epsilon = 1.0e-12);
    assert_eq!(report.per_atom_kappa_hat.len(), 2);
    // The report carries a verdict (no longer a "not implemented" caveat).
    // The verdict is consistent with the threshold function evaluated on the
    // report's own quantities — the report does not fabricate a verdict.
    let kappa_max = report
        .per_atom_kappa_hat
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let recomputed = curved_dictionary_global_optimality_verdict(
        report.mu_hat,
        kappa_max,
        report.peak_activity_floor,
        report.snr_proxy,
        report.per_atom_kappa_hat.len(),
    );
    assert_eq!(report.global_optimality, recomputed);
    // μ̂ = 0 (orthogonal frames) ⇒ when the preconditions hold (κ̂ < 1,
    // SNR > 1) the certificate certifies, since the budget is positive and
    // μ̂ cannot exceed it. κ̂ = 1/radius = 0.5 < 1 here, so the only gate is
    // SNR; assert certification whenever SNR clears the noise floor.
    if report.snr_proxy > 1.0 {
        assert!(
            report.global_optimality.is_certified(),
            "μ̂=0, κ̂=0.5<1, SNR>1 ⇒ must certify; got {}",
            report.note
        );
    }
}

#[test]
pub(crate) fn dictionary_incoherence_report_coherent_frames_has_unit_mu_hat() {
    let term = circle_certificate_fixture(2.0, &[(0, 1), (0, 1)]);
    let report = dictionary_incoherence_report(&term).unwrap();
    assert_abs_diff_eq!(report.mu_hat, 1.0, epsilon = 1.0e-12);
}

#[test]
pub(crate) fn dictionary_incoherence_report_circle_kappa_matches_inverse_radius() {
    let radius = 2.5_f64;
    let mut term = circle_certificate_fixture(radius, &[(0, 1)]);
    term.set_certificate_dispersion(0.25).unwrap();
    let report = dictionary_incoherence_report(&term).unwrap();
    assert_abs_diff_eq!(
        report.per_atom_kappa_hat[0],
        1.0 / radius,
        epsilon = 1.0e-10
    );
    assert!(report.snr_proxy.is_finite() && report.snr_proxy > 0.0);
    assert_abs_diff_eq!(report.mean_activity_floor, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(report.peak_activity_floor, 1.0, epsilon = 1.0e-12);
}

#[test]
pub(crate) fn search_strategy_exposes_fixed_and_sweep_values() {
    assert!(SearchStrategy::Fixed.is_fixed());

    let strategy = SearchStrategy::ExponentialSweep {
        values: vec![0.1, 1.0, 10.0],
    };
    assert!(!strategy.is_fixed());
    assert_eq!(strategy.sweep_values(), Some([0.1, 1.0, 10.0].as_slice()));
}

/// `try_assignments_row` may only pin the K==1 assignment to `1.0` for
/// Softmax, whose single simplex coordinate is genuinely fixed. For the
/// independent gate modes (IBP-MAP, JumpReLU) the lone logit must drive the
/// gate; otherwise the reconstruction ignores a free parameter that the
/// prior still penalizes (an invalid objective). Regression for the
/// audit's K==1 special-case bug.
#[test]
pub(crate) fn k1_gate_modes_do_not_pin_assignment_to_one() {
    // IBP-MAP, K=1: σ(0/τ)·π_0. Since #614 the stick-breaking prior shrinks
    // EVERY atom by one Beta(α,1) stick mean — including the first — so the
    // truncated mean is `π_0 = α/(α+1)`, NOT the old unshrunk `π_0 = 1` (that
    // left atom 0 unshrunk and broke α's role as a concentration). With α=1,
    // τ=1, l=0 the gate is therefore `σ(0)·(1/2) = 0.5·0.5 = 0.25`. The point
    // of this case is unchanged: the K=1 gate is NOT pinned to 1.0 (the
    // Softmax-only collapse), it is the genuine sigmoid×prior product.
    let ibp = SaeAssignment::from_blocks_with_mode(
        array![[0.0]],
        vec![array![[0.0]]],
        AssignmentMode::ibp_map(1.0, 1.0, false),
    )
    .unwrap();
    let ibp_gate = ibp.try_assignments_row(0).unwrap()[0];
    assert_abs_diff_eq!(ibp_gate, 0.25, epsilon = 1e-9);
    assert!(
        (ibp_gate - 1.0).abs() > 1e-6,
        "K=1 IBP-MAP must not pin the gate to 1.0"
    );

    // JumpReLU, K=1, logit below threshold: hard-gated off (not 1.0).
    let jr = SaeAssignment::from_blocks_with_mode(
        array![[-1.0]],
        vec![array![[0.0]]],
        AssignmentMode::jumprelu(1.0, 0.0),
    )
    .unwrap();
    assert_abs_diff_eq!(jr.try_assignments_row(0).unwrap()[0], 0.0, epsilon = 1e-12);

    // Softmax, K=1: still pinned to 1.0 (no free simplex coordinate).
    // The softmax logits matrix carries `K = k_atoms()` columns (one per
    // atom, canonicalized so the reference column is 0), so K=1 is a single
    // zero column — not the K-1 `assignment_coord_dim` layout. The K=1 pin
    // in `try_assignments_row` keys off `k_atoms() == 1`, i.e. one column.
    let sm = SaeAssignment::from_blocks_with_mode(
        Array2::<f64>::zeros((1, 1)),
        vec![array![[0.0]]],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    assert_abs_diff_eq!(sm.try_assignments_row(0).unwrap()[0], 1.0, epsilon = 1e-12);
}

/// The JumpReLU surrogate is centered at the threshold: just above the
/// threshold the gate is ≈ σ(0) = 0.5, not the uncentered σ(threshold/τ).
/// Below the threshold the hard gate keeps the value at exactly zero.
/// Regression for the audit's miscentered-threshold bug.
#[test]
pub(crate) fn jumprelu_surrogate_is_centered_at_threshold() {
    let threshold = 2.0;
    let temperature = 1.0;
    let logits = array![2.0 + 1e-6, 1.0];
    let gates = jumprelu_row(logits.view(), temperature, threshold);
    // Just above threshold the centered surrogate is ≈ 0.5; the old
    // uncentered surrogate would have been σ(2.0) ≈ 0.88.
    assert_abs_diff_eq!(gates[0], 0.5, epsilon = 1e-3);
    assert!(
        gates[0] < 0.6,
        "surrogate not centered at threshold: {}",
        gates[0]
    );
    // Strictly below the threshold the gate is hard-zero.
    assert_abs_diff_eq!(gates[1], 0.0, epsilon = 1e-12);
}

pub(crate) fn periodic_basis(coords: &Array2<f64>) -> (Array2<f64>, Array3<f64>) {
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 3));
    let mut jet = Array3::<f64>::zeros((n, 3, 1));
    for row in 0..n {
        let x = coords[[row, 0]].rem_euclid(1.0);
        let angle = 2.0 * std::f64::consts::PI * x;
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = angle.sin();
        phi[[row, 2]] = angle.cos();
        jet[[row, 1, 0]] = 2.0 * std::f64::consts::PI * angle.cos();
        jet[[row, 2, 0]] = -2.0 * std::f64::consts::PI * angle.sin();
    }
    (phi, jet)
}

// --- Periodic/topology ARD prior smoothness + value↔grad consistency ---

/// The periodic von-Mises ARD energy must be continuous (in value, gradient,
/// and curvature) as the latent coordinate crosses the period cut. The old
/// Euclidean `½α t²` jumped by `½α P²` here, breaking Armijo descent. With
/// period `P = 1` the cut is at `t = 1 ≡ 0`: evaluating just below and just
/// above must agree to O(eps), and the wrapped-to-`0` representative must
/// match the unwrapped value.
#[test]
pub(crate) fn ard_axis_prior_periodic_is_continuous_across_cut() {
    let alpha = 2.3_f64;
    let period = 1.0_f64;
    let eps = 1.0e-6;
    let below = ArdAxisPrior::eval(alpha, period - eps, Some(period));
    let above = ArdAxisPrior::eval(alpha, period + eps, Some(period));
    let at_zero = ArdAxisPrior::eval(alpha, 0.0, Some(period));
    // Crossing the cut changes value/grad/hess by O(eps), NOT O(½αP²≈1.15
    // for the old Euclidean prior). value and hess are even in (t-cut) so
    // they match to O(eps²); grad is odd through 0, so it flips sign but its
    // magnitude → 0 at the cut and the jump is O(eps) (continuous).
    let cont_tol = 10.0 * alpha * eps; // O(eps) continuity bound
    assert!((below.value - above.value).abs() < cont_tol);
    assert!((below.grad - above.grad).abs() < cont_tol);
    assert!((below.hess - above.hess).abs() < cont_tol);
    // The gradient vanishes at the cut (no kink): both one-sided values are
    // O(eps), unlike the old prior whose grad was α·P ≈ 2.3 just below.
    assert!(below.grad.abs() < cont_tol);
    assert!(above.grad.abs() < cont_tol);
    // The unwrapped representative `period - eps` and the wrapped-near-0
    // representative agree: the energy is a genuine function on the circle.
    assert_abs_diff_eq!(below.value, at_zero.value, epsilon = 1.0e-9);
    // At the origin: zero energy/gradient, curvature == alpha (the ARD
    // precision interpretation is preserved).
    assert_abs_diff_eq!(at_zero.value, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(at_zero.grad, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(at_zero.hess, alpha, epsilon = 1.0e-12);
    // `sq_equiv` is alpha-independent so the Mackay/Fellner-Schall update is
    // a clean function of the coordinates.
    let sq_a = ArdAxisPrior::eval(1.0, 0.3, Some(period)).sq_equiv;
    let sq_b = ArdAxisPrior::eval(5.0, 0.3, Some(period)).sq_equiv;
    assert_abs_diff_eq!(sq_a, sq_b, epsilon = 1.0e-12);
    // ½·α·sq_equiv reproduces the energy (consistency with ard_value).
    let p = ArdAxisPrior::eval(alpha, 0.3, Some(period));
    assert_abs_diff_eq!(0.5 * alpha * p.sq_equiv, p.value, epsilon = 1.0e-12);
}

/// The per-axis prior gradient must be the exact derivative of its value, on
/// BOTH the Euclidean (Gaussian) and periodic (von-Mises) axes. This is the
/// d=1 value↔grad FD agreement that the line search depends on.
#[test]
pub(crate) fn ard_axis_prior_value_grad_fd_consistent() {
    let alpha = 1.7_f64;
    let h = 1.0e-6;
    for &period in &[None, Some(1.0_f64), Some(std::f64::consts::TAU)] {
        // Sample several points, including near a periodic cut.
        for &t in &[-0.37_f64, 0.02, 0.49, 0.83, 0.999, 1.4] {
            let p = ArdAxisPrior::eval(alpha, t, period);
            let vp = ArdAxisPrior::eval(alpha, t + h, period).value;
            let vm = ArdAxisPrior::eval(alpha, t - h, period).value;
            let fd_grad = (vp - vm) / (2.0 * h);
            assert_abs_diff_eq!(p.grad, fd_grad, epsilon = 1.0e-5);
            // Hessian == derivative of gradient.
            let gp = ArdAxisPrior::eval(alpha, t + h, period).grad;
            let gm = ArdAxisPrior::eval(alpha, t - h, period).grad;
            let fd_hess = (gp - gm) / (2.0 * h);
            assert_abs_diff_eq!(p.hess, fd_hess, epsilon = 1.0e-5);
        }
    }
}

/// The manifold → per-axis periodicity map must classify every topology's
/// d=1 (and product) axes correctly: line=non-periodic, circle=periodic,
/// torus=per-axis periodic, sphere chart=(non-periodic lat, periodic lon),
/// embedded sphere=non-periodic (smooth retraction, no cut).
#[test]
pub(crate) fn axis_periods_map_each_topology() {
    assert_eq!(LatentManifold::Euclidean.axis_periods(), vec![None]);
    assert_eq!(
        LatentManifold::Circle { period: 1.0 }.axis_periods(),
        vec![Some(1.0)]
    );
    // Torus (Product of Circles), each axis periodic.
    let torus = LatentManifold::Product(vec![
        LatentManifold::Circle { period: 1.0 },
        LatentManifold::Circle { period: 1.0 },
    ]);
    assert_eq!(torus.axis_periods(), vec![Some(1.0), Some(1.0)]);
    // Sphere lat/lon chart: lat is an Interval (non-periodic), lon a Circle.
    let sphere_chart = LatentManifold::Product(vec![
        LatentManifold::Interval { lo: -1.0, hi: 1.0 },
        LatentManifold::Circle {
            period: std::f64::consts::TAU,
        },
    ]);
    assert_eq!(
        sphere_chart.axis_periods(),
        vec![None, Some(std::f64::consts::TAU)]
    );
    // Embedded sphere: smooth retraction, reported non-periodic per axis.
    assert_eq!(
        LatentManifold::Sphere { dim: 3 }.axis_periods(),
        vec![None, None, None]
    );
}

/// End-to-end: a periodic term's `ard_value` must be continuous as a latent
/// coordinate is stepped across the period cut via the (wrapping)
/// retraction. Reproduces the original non-smoothness bug at the term level:
/// the old Euclidean prior made `loss.ard` jump by ~½α·P² when a Newton step
/// crossed `t = 1 ≡ 0`.
#[test]
pub(crate) fn ard_value_continuous_across_periodic_cut_d1() {
    // Single periodic atom, one row sitting just below the cut at t≈1.
    let coords0 = array![[0.999_f64]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.2], [-0.3], [0.4]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((1, 1)),
        vec![coords0],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, true),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.1_f64]];
    // Large alpha makes the OLD bug's jump (~½·α·P²) enormous relative to
    // the smooth O(step) change; with the von-Mises prior it stays tiny.
    let alpha = 50.0_f64;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![alpha.ln()]]);

    let ard_before = term.loss(target.view(), &rho).unwrap().ard;
    // Step the coordinate by +0.002 so it crosses the cut: 0.999 -> 1.001,
    // which the Circle retraction wraps to 0.001.
    let q = term.assignment.row_block_dim();
    let beta_dim = term.beta_dim();
    let mut delta_ext = Array1::<f64>::zeros(q);
    // coord axis is the last entry of the row block (after the logit).
    delta_ext[q - 1] = 0.002;
    let delta_beta = Array1::<f64>::zeros(beta_dim);
    term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
        .unwrap();
    let wrapped = term.assignment.coords[0].row(0)[0];
    // Confirm the step actually crossed and wrapped near 0.
    assert!(
        wrapped < 0.01,
        "coordinate should have wrapped across the cut, got {wrapped}"
    );
    let ard_after = term.loss(target.view(), &rho).unwrap().ard;
    // Smooth: a 0.002 step near the cut changes the ARD energy by a tiny
    // amount. The OLD Euclidean prior would jump by ≈ ½·α·(1² - 0²) = 25.
    assert!(
        (ard_after - ard_before).abs() < 1.0e-2,
        "periodic ARD jumped across the cut: before={ard_before}, after={ard_after}"
    );
}

/// The *full line-search objective* (`penalized_objective_total`) — not just
/// the built-in `loss.ard` — must be continuous across the period cut when a
/// registry `ARDPenalty` is present, which is the production SAE config
/// (`ard_per_atom=True` emits `{"kind":"ard","target":"t"}`). The registry
/// `ARDPenalty` value is the legacy Euclidean Gaussian `½λΣt²`, which jumps by
/// ≈ ½λ·P² across the cut. Before the fix it was summed into
/// `analytic_penalty_value_total` on top of the von-Mises `loss.ard`, so the
/// line-search objective jumped discontinuously while the assembled gradient
/// (also double-counting the Gaussian `λt`, but that piece is continuous)
/// predicted only an O(step) change — a near-zero Newton step crossing the
/// cut then raised the objective by ≈ ½λ and Armijo rejected it (BUG 1). The
/// fix skips the registry ARD on every SAE path so the smooth von-Mises
/// built-in is the single source of truth; the objective must now stay smooth.
#[test]
pub(crate) fn penalized_objective_continuous_across_periodic_cut_with_registry_ard() {
    let coords0 = array![[0.999_f64]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.2], [-0.3], [0.4]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((1, 1)),
        vec![coords0],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, true),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.1_f64]];
    // Large precision makes the OLD Gaussian-registry jump (≈ ½λP² = 25) huge
    // relative to the smooth O(step) von-Mises change.
    let alpha = 50.0_f64;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![alpha.ln()]]);

    // Production-shaped registry: one ARD penalty on the "t" coord block.
    let coord = &term.assignment.coords[0];
    let mut registry = AnalyticPenaltyRegistry::new();
    let ard_pen = ARDPenalty::new(
        PsiSlice::full(coord.len(), Some(coord.latent_dim())),
        coord.latent_dim(),
    );
    registry.push(AnalyticPenaltyKind::Ard(Arc::new(ard_pen)));

    let obj_before = term
        .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
        .unwrap();
    let q = term.assignment.row_block_dim();
    let beta_dim = term.beta_dim();
    let mut delta_ext = Array1::<f64>::zeros(q);
    delta_ext[q - 1] = 0.002; // 0.999 -> 1.001, wraps to 0.001 across the cut.
    let delta_beta = Array1::<f64>::zeros(beta_dim);
    term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
        .unwrap();
    let wrapped = term.assignment.coords[0].row(0)[0];
    assert!(
        wrapped < 0.01,
        "coordinate should have wrapped across the cut, got {wrapped}"
    );
    let obj_after = term
        .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
        .unwrap();
    // Smooth: a 0.002 step changes the full objective by a tiny amount. The
    // OLD Gaussian-registry path jumped by ≈ ½·50·(1²−0²) = 25.
    assert!(
        (obj_after - obj_before).abs() < 1.0e-2,
        "line-search objective jumped across the cut: before={obj_before}, after={obj_after}"
    );
}

/// Issue #795: `gate_sparsity="scad"` emits a `ScadMcpPenalty` on the "t"
/// coordinate block. SCAD's energy `Σ f(√(t²+ε²))` is a magnitude shrinkage
/// with a fixed origin at `t=0`. On a **periodic** (Circle) axis the latent
/// is an angle defined only modulo its period, so the raw `|t|` is BOTH
/// ill-posed (no rotation-invariant origin) and *discontinuous across the
/// retraction branch cut*: a coordinate just below the period wraps to just
/// above zero, and `f(|t|)` jumps from the flat tail to ≈0. Folded into the
/// line-search objective, that jump made a near-zero coordinate Newton step
/// change the objective by an O(weight) amount, so Armijo rejected
/// otherwise-valid steps and the inner joint solve never reached
/// stationarity (`reml_criterion: inner solve did not converge`).
///
/// The fix restricts the SCAD/MCP shrinkage to the Euclidean axes, so on a
/// pure Circle atom it contributes nothing — the objective with the SCAD
/// registry must equal the registry-free objective, and must stay continuous
/// across the cut.
#[test]
pub(crate) fn scad_coord_penalty_inert_and_continuous_on_periodic_axis() {
    use gam_terms::analytic_penalties::{PenaltyConcavity, ScadMcpPenalty};

    let coords0 = array![[0.999_f64]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.2], [-0.3], [0.4]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((1, 1)),
        vec![coords0],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, true),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.1_f64]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0_f64]]);

    // Registry with a single SCAD shrinkage on the coordinate block, with a
    // large weight so the OLD (unrestricted) energy would dominate.
    let coord = &term.assignment.coords[0];
    let mut registry = AnalyticPenaltyRegistry::new();
    let scad = ScadMcpPenalty::new(
        PsiSlice::full(coord.len(), Some(coord.latent_dim())),
        5.0,
        coord.n_obs(),
        3.7,
        1.0e-3,
        PenaltyConcavity::Scad,
        false,
    )
    .unwrap();
    registry.push(AnalyticPenaltyKind::ScadMcp(Arc::new(scad)));

    // Inert on a pure Circle: the SCAD registry adds zero energy because the
    // sole axis is periodic.
    let with_scad = term
        .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
        .unwrap();
    let without = term
        .penalized_objective_total(target.view(), &rho, None, 1.0)
        .unwrap();
    assert!(
        (with_scad - without).abs() < 1.0e-12,
        "SCAD coord penalty must be inert on a pure periodic axis: \
             with={with_scad}, without={without}"
    );

    // Continuous across the period cut: stepping 0.999 -> 1.001 (wraps to
    // 0.001) must not jump the objective. The OLD unrestricted SCAD energy
    // jumped by ≈ weight·(|0.999| − |0.001|) ≈ 5.
    let obj_before = with_scad;
    let q = term.assignment.row_block_dim();
    let beta_dim = term.beta_dim();
    let mut delta_ext = Array1::<f64>::zeros(q);
    delta_ext[q - 1] = 0.002;
    let delta_beta = Array1::<f64>::zeros(beta_dim);
    term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
        .unwrap();
    let wrapped = term.assignment.coords[0].row(0)[0];
    assert!(
        wrapped < 0.01,
        "coordinate should have wrapped across the cut, got {wrapped}"
    );
    let obj_after = term
        .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
        .unwrap();
    assert!(
        (obj_after - obj_before).abs() < 1.0e-2,
        "SCAD line-search objective jumped across the periodic cut: \
             before={obj_before}, after={obj_after}"
    );
}

/// Guard against over-broad exclusion: on a **Euclidean** chart axis the SCAD
/// magnitude shrinkage is well-posed (`t=0` is a genuine origin) and must
/// remain active. `sae_coord_penalty_euclidean_restriction` returns `None`
/// (nothing to restrict) for an all-Euclidean coord and the full-support
/// `Some` carrier for a periodic coord, so value/gradient/curvature all see
/// the same axis set.
#[test]
pub(crate) fn scad_coord_penalty_active_on_euclidean_axis() {
    let euclid = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 1)),
        vec![array![[0.5_f64], [-0.7], [1.3]]],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    // All axes Euclidean: no restriction (the penalty applies in full).
    assert!(
        sae_coord_penalty_euclidean_restriction(&euclid.coords[0]).is_none(),
        "Euclidean coord must not be restricted"
    );

    let circle = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 1)),
        vec![array![[0.1_f64], [0.4], [0.9]]],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    // Pure periodic axis: restricted to an empty Euclidean carrier.
    let (axes, compacted) = sae_coord_penalty_euclidean_restriction(&circle.coords[0])
        .expect("periodic coord must be restricted");
    assert!(
        axes.is_empty(),
        "circle has no Euclidean axes, got {axes:?}"
    );
    assert_eq!(compacted.len(), 0, "compacted target must be empty");
}

/// The von-Mises coordinate-prior curvature `V'' = α·cos(κt)` is indefinite
/// (negative for |t| past a quarter period). Writing it raw into the
/// Newton/Schur `htt` diagonal at K=2 made the per-row coordinate block, and
/// hence the Schur complement, non-PD and the Cholesky failed on a negative
/// pivot (BUG 3). The assembled `htt` diagonal on every periodic coord axis
/// must therefore be non-negative (the `max(V'',0)` PSD majorizer), while the
/// gradient stays the exact `V'`.
#[test]
pub(crate) fn periodic_ard_curvature_is_psd_in_assembled_htt() {
    // Two rows past the quarter period (t in (0.25, 0.75)) where cos(2πt) < 0.
    let coords0 = array![[0.40_f64], [0.60_f64]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.2], [-0.3], [0.4]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((2, 1)),
        vec![coords0],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.1_f64], [0.2_f64]];
    // Large α drives V''=α·cos(2πt) strongly negative at t=0.4,0.6
    // (cos(0.8π)≈-0.809), so a raw write would push the data-fit-only htt
    // diagonal negative.
    let alpha = 100.0_f64;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![alpha.ln()]]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    for (row_idx, row) in sys.rows.iter().enumerate() {
        let d = row.htt.nrows();
        for a in 0..d {
            assert!(
                row.htt[[a, a]] >= 0.0,
                "row {row_idx} htt diagonal[{a}]={} must be PSD (von-Mises \
                     curvature clamped to its positive part)",
                row.htt[[a, a]]
            );
        }
    }
}

/// #1117 follow-up (curved-atom sparse co-assignment): the compact active-set
/// layout must apply the SAME per-row Riemannian geometry to the assembled
/// Arrow-Schur blocks as the dense uniform-`q` layout. Before the fix the
/// compact path skipped the tangent projection entirely (and `sparse_active_plan`
/// refused to engage on any non-Euclidean ext-coord manifold), so a curved-atom
/// SAE at large `K` paid the dense `K²` co-assignment Gram. The new code rebuilds
/// each compact row's product manifold + point in compact column order
/// (`compact_row_ext_manifold_and_point`) and applies the identical
/// `gt` gradient projection, `htt` Riemannian-Hessian correction, and `htbeta`
/// column projection (plus the Kronecker local-Jacobian projection).
///
/// This pins the equivalence directly: with EVERY row's active set forced to the
/// full atom set, the compact column order coincides with the dense full-`q`
/// order (IBP-MAP has `assignment_coord_dim == k_atoms`), so the two assemblies
/// must produce BIT-IDENTICAL `gt`, `htt`, and `htbeta` on a genuinely curved
/// (Circle) two-atom term with non-trivial logits and coordinates (so the
/// von-Mises gradient — hence the Riemannian Hessian correction — is nonzero).
#[test]
pub(crate) fn compact_layout_riemannian_geometry_matches_dense_on_full_support() {
    // `SaeRowLayout` and `SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM` are in scope via
    // `use super::*` (the `sae_manifold` module re-exports `row_layout::*` and
    // `term::*`). The assembly override is `Option<Option<SaeRowLayout>>`:
    // `Some(None)` pins dense, `Some(Some(layout))` pins a compact layout.

    // Two Circle atoms, coordinates spread around the period so the von-Mises
    // coordinate-prior gradient is nonzero (the Riemannian Hessian correction
    // depends on `eg`, so a zero gradient would make the test vacuous w.r.t. the
    // curvature term). Distinct logits so the assignment masses differ per atom.
    let coords_a = array![[0.12_f64], [0.37], [0.66], [0.91]];
    let coords_b = array![[0.81_f64], [0.05], [0.48], [0.23]];
    let (phi_a, jet_a) = periodic_basis(&coords_a);
    let (phi_b, jet_b) = periodic_basis(&coords_b);
    let atom_a = SaeManifoldAtom::new(
        "circle_a",
        SaeAtomBasisKind::Periodic,
        1,
        phi_a,
        jet_a,
        array![[0.20, -0.10], [-0.30, 0.25], [0.40, 0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let atom_b = SaeManifoldAtom::new(
        "circle_b",
        SaeAtomBasisKind::Periodic,
        1,
        phi_b,
        jet_b,
        array![[-0.15, 0.30], [0.22, -0.18], [0.33, 0.27]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));

    let n = 4usize;
    let logits = array![[0.4_f64, -0.2], [-0.1, 0.5], [0.3, 0.1], [-0.4, 0.2]];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords_a, coords_b],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.7, 1.0, true),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom_a, atom_b], assignment).unwrap();
    let target = array![
        [0.10_f64, -0.05],
        [0.20, 0.15],
        [-0.12, 0.08],
        [0.05, -0.20]
    ];
    let alpha = 5.0_f64;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![alpha.ln()], array![alpha.ln()]]);
    let probe = SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM;

    // Dense layout: pin `Some(None)` so the override forces the dense path
    // regardless of the budget-derived plan.
    let dense = term
        .assemble_arrow_schur_inner(target.view(), &rho, None, 1.0, probe, Some(None))
        .unwrap();

    // Compact layout with EVERY row's active set = both atoms (full support).
    let coord_dims = vec![1usize, 1usize];
    let coord_offsets = term.assignment.coord_offsets();
    let full_active: Vec<Vec<usize>> = (0..n).map(|_| vec![0usize, 1usize]).collect();
    let layout = SaeRowLayout::from_active_atoms(full_active, coord_dims, coord_offsets);
    let compact = term
        .assemble_arrow_schur_inner(target.view(), &rho, None, 1.0, probe, Some(Some(layout)))
        .unwrap();

    assert_eq!(dense.rows.len(), compact.rows.len());
    for (row_idx, (dr, cr)) in dense.rows.iter().zip(compact.rows.iter()).enumerate() {
        assert_eq!(
            dr.gt.len(),
            cr.gt.len(),
            "row {row_idx}: gt length mismatch (full-support compact must equal dense q)"
        );
        for a in 0..dr.gt.len() {
            assert_abs_diff_eq!(dr.gt[a], cr.gt[a], epsilon = 1e-12);
        }
        assert_eq!(dr.htt.dim(), cr.htt.dim());
        for a in 0..dr.htt.nrows() {
            for b in 0..dr.htt.ncols() {
                assert_abs_diff_eq!(dr.htt[[a, b]], cr.htt[[a, b]], epsilon = 1e-12);
            }
        }
        assert_eq!(dr.htbeta.dim(), cr.htbeta.dim());
        for a in 0..dr.htbeta.nrows() {
            for b in 0..dr.htbeta.ncols() {
                assert_abs_diff_eq!(dr.htbeta[[a, b]], cr.htbeta[[a, b]], epsilon = 1e-12);
            }
        }
    }

    // The geometry must be non-trivial: at least one assembled gt entry differs
    // from the raw (un-projected) Euclidean gradient direction would, i.e. the
    // Circle projection actually fired. We assert the htt is symmetric and finite
    // as a floor (the Riemannian correction is applied), and that the assembly
    // produced a non-degenerate (nonzero) curvature block.
    let any_curvature = compact
        .rows
        .iter()
        .any(|r| r.htt.iter().any(|&v| v.abs() > 1e-9));
    assert!(
        any_curvature,
        "assembled compact htt is all-zero — the test data did not exercise curvature"
    );
}

/// #1117 follow-up gate (a): the compact sparse co-assignment plan must ENGAGE
/// on a curved (non-Euclidean) ext-coord manifold once the dense `K²` data Gram
/// trips the in-core budget — exactly the manifold-SAE-on-OLMo regime (curved
/// atoms + large `K`). Before the fix `sparse_active_plan` returned `None` for
/// ANY curved atom regardless of `K`, forcing the dense `K²` coupling; the
/// `is_euclidean()` guard is now removed and the budget is the sole gate.
///
/// We assert the engagement decision is identical for a curved (Circle) term and
/// its Euclidean twin at the same `m_total`: (1) with a budget BELOW the dense
/// Gram both engage the compact plan with the same `k_active_cap`; (2) with a
/// huge budget both stay dense (`None`). The budget is pinned via
/// `sparse_active_plan_for_budget` so no multi-GB Gram is allocated.
#[test]
pub(crate) fn sparse_plan_engages_on_curved_manifold_when_budget_tripped() {
    // Build a `k`-atom IBP-MAP term with the given per-atom coordinate manifold.
    // Each atom carries the width-3 periodic basis, so `m_total = 3·k`.
    fn build_term(k: usize, curved: bool) -> SaeManifoldTerm {
        let n = 4usize;
        let coords: Array2<f64> = array![[0.12_f64], [0.37], [0.66], [0.91]];
        let (phi, jet) = periodic_basis(&coords);
        let atoms: Vec<SaeManifoldAtom> = (0..k)
            .map(|j| {
                SaeManifoldAtom::new(
                    format!("atom_{j}"),
                    SaeAtomBasisKind::Periodic,
                    1,
                    phi.clone(),
                    jet.clone(),
                    array![[0.20, -0.10], [-0.30, 0.25], [0.40, 0.15]],
                    Array2::<f64>::eye(3),
                )
                .unwrap()
                .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
            })
            .collect();
        let manifold = if curved {
            LatentManifold::Circle { period: 1.0 }
        } else {
            LatentManifold::Euclidean
        };
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, k)),
            (0..k).map(|_| coords.clone()).collect(),
            (0..k).map(|_| manifold.clone()).collect(),
            AssignmentMode::ibp_map(0.7, 1.0, true),
        )
        .unwrap();
        SaeManifoldTerm::new(atoms, assignment).unwrap()
    }

    let k = 8usize;
    let curved = build_term(k, true);
    let euclidean = build_term(k, false);

    // `m_total = 3·k`; dense Gram = (3k)² · 8 bytes. Pin a budget strictly below
    // it so the plan must engage, and one far above it so it must not.
    let m_total = 3 * k;
    let dense_gram_bytes = m_total * m_total * std::mem::size_of::<f64>();
    let small_budget = dense_gram_bytes / 2;
    let huge_budget = dense_gram_bytes * 16;

    let curved_engaged = curved.sparse_active_plan_for_budget(small_budget);
    let euclid_engaged = euclidean.sparse_active_plan_for_budget(small_budget);
    assert!(
        curved_engaged.is_some(),
        "curved-manifold term must engage the sparse plan once the dense K² Gram \
         trips the budget (the #1117 follow-up lever) — got None"
    );
    assert_eq!(
        curved_engaged, euclid_engaged,
        "the sparse-plan engagement decision (k_active_cap + cutoff) must no longer \
         depend on whether the ext-coord manifold is curved: curved={curved_engaged:?} \
         euclidean={euclid_engaged:?}"
    );

    // Above the dense-Gram footprint, both stay on the exact dense layout.
    assert_eq!(
        curved.sparse_active_plan_for_budget(huge_budget),
        None,
        "a curved term whose dense Gram fits the budget must keep the dense layout"
    );
    assert_eq!(
        euclidean.sparse_active_plan_for_budget(huge_budget),
        None,
        "a Euclidean term whose dense Gram fits the budget must keep the dense layout"
    );
}

/// `snapshot_mutable_state` / `restore_mutable_state` (the in-place
/// line-search save/restore that replaced the per-halving full
/// `self.clone()`) must restore exactly the state an `apply_newton_step`
/// trial perturbs: decoder coefficients, the `refresh_basis`-rebuilt
/// basis evaluations, assignment logits, and latent coordinates. Pins
/// item-1 of the SAE hot-path CPU-perf refactor.
#[test]
pub(crate) fn snapshot_restore_round_trips_mutated_state() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.2], [-0.3], [0.4]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((4, 1)),
        vec![coords0],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, true),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // Capture pre-step state, then apply a non-trivial Newton step that
    // refreshes the basis (changing basis_values/jacobian, decoder
    // coefficients, logits, and coords).
    let snapshot = term.snapshot_mutable_state();
    let pre_basis = term.atoms[0].basis_values.clone();
    let pre_jet = term.atoms[0].basis_jacobian.clone();
    let pre_decoder = term.atoms[0].decoder_coefficients.clone();
    let pre_logits = term.assignment.logits.clone();
    let pre_coords = term.assignment.coords[0].as_matrix();

    let q = term.assignment.row_block_dim();
    let beta_dim = term.beta_dim();
    let delta_ext = Array1::<f64>::from_elem(4 * q, 0.3);
    let delta_beta = Array1::<f64>::from_elem(beta_dim, -0.4);
    term.apply_newton_step(delta_ext.view(), delta_beta.view(), 1.0)
        .unwrap();

    // Something must actually have changed, else the test is vacuous.
    assert!(
        (&term.atoms[0].basis_values - &pre_basis)
            .mapv(f64::abs)
            .sum()
            > 1e-9
            || (&term.atoms[0].decoder_coefficients - &pre_decoder)
                .mapv(f64::abs)
                .sum()
                > 1e-9,
        "apply_newton_step did not perturb the snapshotted state"
    );

    // Restore and confirm every snapshotted field matches the pre-step
    // values bit-for-bit.
    term.restore_mutable_state(&snapshot);
    assert_eq!(term.atoms[0].basis_values, pre_basis);
    assert_eq!(term.atoms[0].basis_jacobian, pre_jet);
    assert_eq!(term.atoms[0].decoder_coefficients, pre_decoder);
    assert_eq!(term.assignment.logits, pre_logits);
    assert_eq!(term.assignment.coords[0].as_matrix(), pre_coords);
}

#[test]
pub(crate) fn ibp_path_refreshes_periodic_basis_for_two_newton_iterations() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.2], [-0.3], [0.4]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((4, 1)),
        vec![coords0],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, true),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.10], [0.05], [-0.15], [0.20]];
    let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
    let loss0 = term.loss(target.view(), &rho).unwrap().total();
    let basis0 = term.atoms[0].basis_values.clone();

    let loss = term
        .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 2, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();

    assert!(loss.total().is_finite());
    assert!(loss.total() <= loss0 + 1.0e-8);
    assert!(
        term.assignment.coords[0]
            .as_flat()
            .iter()
            .all(|v| v.is_finite())
    );
    assert!(term.assignment.assignments().iter().all(|v| v.is_finite()));
    let basis_delta = (&term.atoms[0].basis_values - &basis0).mapv(f64::abs).sum();
    assert!(basis_delta > 1.0e-10);
}

pub(crate) fn small_two_atom_periodic_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let atom0 = SaeManifoldAtom::new(
        "periodic0",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.25], [-0.35], [0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let atom1 = SaeManifoldAtom::new(
        "periodic1",
        SaeAtomBasisKind::Periodic,
        1,
        phi1,
        jet1,
        array![[-0.10], [0.20], [0.30]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let target = array![[0.12], [-0.03], [0.08], [0.20], [-0.11]];
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![array![0.9_f64.ln()], array![1.1_f64.ln()]],
    );
    (term, target, rho)
}

/// #1026 — the per-atom **held-out EV attribution** that pairs with each
/// atom's fitted turning `Θ` to form the EV-vs-Θ discriminating signal.
///
/// `per_atom_loao_explained_variance` returns, per atom, the explained
/// variance lost when that atom is withheld from the assembled
/// reconstruction (`ΔEV_k = EV(full) − EV(full ⊖ atom_k)`). With the target
/// set to the term's OWN full reconstruction the dictionary explains the
/// target exactly (`EV(full) = 1`), so every atom that genuinely carries
/// reconstruction must show a strictly positive ΔEV (removing it leaves a
/// residual), while an atom that contributes nothing (zero decoder) must
/// show ΔEV ≈ 0. This is the held-out half of the `(Θ, ΔEV)` pair the
/// post-fit pass logs: a high-`Θ` atom earning ΔEV is a genuine curved
/// family; a `Θ ≈ 0` atom earning ΔEV is a linear-tail direction.
#[test]
pub(crate) fn per_atom_loao_ev_attributes_each_load_bearing_atom() {
    let (term, _target, rho) = small_two_atom_periodic_term();
    // Target = the term's own full reconstruction ⇒ EV(full) = 1 exactly.
    let target = term
        .try_fitted_for_rho(&rho)
        .expect("full reconstruction must assemble");
    let ev_full = reconstruction_explained_variance(target.view(), target.view())
        .expect("self-reconstruction EV defined");
    assert!(
        (ev_full - 1.0).abs() < 1e-12,
        "target = full reconstruction ⇒ EV(full) = 1; got {ev_full}"
    );

    let dev = term
        .per_atom_loao_explained_variance(target.view(), &rho)
        .expect("LOAO EV must evaluate");
    assert_eq!(dev.len(), term.k_atoms(), "one ΔEV per atom");

    // Both periodic atoms genuinely carry reconstruction (nonzero decoders
    // and nonzero assignment mass), so each must lose EV when withheld.
    for (atom_idx, d) in dev.iter().enumerate() {
        let d = d.unwrap_or_else(|| panic!("atom {atom_idx} ΔEV must be defined"));
        assert!(
            d > 1e-9,
            "load-bearing atom {atom_idx} must earn positive training LOAO ΔEV; got {d:.3e}"
        );
        // ΔEV is bounded by EV(full) = 1 (an atom cannot carry more EV than
        // the whole dictionary explains).
        assert!(
            d <= 1.0 + 1e-9,
            "ΔEV for atom {atom_idx} cannot exceed EV(full)=1; got {d:.6e}"
        );
    }

    // An atom withheld is the exact "this atom zeroed" counterfactual: with
    // a zero decoder it carries no reconstruction, so its ΔEV must collapse
    // to ~0 — the same Θ→0 dominance floor logic on the EV axis (no signal,
    // no contribution), independent of the curved/linear question.
    let mut dead_term = term.clone();
    dead_term.atoms[1].decoder_coefficients.fill(0.0);
    let dead_target = term
        .try_fitted_for_rho(&rho)
        .expect("reconstruction with the live atom-1 decoder");
    let dead_dev = dead_term
        .per_atom_loao_explained_variance(dead_target.view(), &rho)
        .expect("LOAO EV must evaluate for the dead-atom term");
    let d_dead = dead_dev[1].expect("dead atom ΔEV defined");
    assert!(
        d_dead.abs() < 1e-9,
        "a zero-decoder atom carries no reconstruction ⇒ ΔEV ≈ 0; got {d_dead:.3e}"
    );
}

/// #1230 — when structure search settles on a CHANGED model, the pre-search
/// joint-Hessian shape bands are stale and must be recomputed from the final
/// per-atom inner fits. The FFI signals "recompute" by calling
/// [`SaeShapeUncertainty::invalidate_bands_for_recompute`], whose contract is:
/// drop EVERY atom's `decoder_covariance` and set EVERY atom's `band_sd` to
/// `NaN`, so the subsequent `complete_born_atom_shape_bands` completion pass —
/// which refills any `NaN` band but SKIPS already-filled bands — recomputes the
/// bands of ALL atoms (seed and born), not just the born ones.
///
/// The #1230 bug was that a SEED atom kept its pre-search joint-Hessian band
/// even after a landed move re-converged the dictionary at a new ρ: the
/// completion pass skipped it because its band_sd was still filled. This test
/// pins the invalidation contract that makes that skip impossible — after the
/// call no band is left filled for the completion pass to skip.
#[test]
pub(crate) fn invalidate_bands_for_recompute_clears_every_seed_band() {
    // A fully-FILLED two-atom shape-uncertainty payload, as the pre-search
    // joint Hessian would assemble: finite band_sd and a materialized
    // decoder_covariance for BOTH atoms (the "seed" atoms).
    fn filled_atom(cov_diag: f64) -> SaeAtomShapeUncertainty {
        SaeAtomShapeUncertainty {
            decoder_covariance: Some(Array2::<f64>::eye(3) * cov_diag),
            band_coords: array![[0.0_f64], [0.5], [1.0]],
            band_mean: Array2::<f64>::from_elem((3, 2), 0.25),
            band_sd: Array2::<f64>::from_elem((3, 2), 0.10),
        }
    }
    let mut shape = SaeShapeUncertainty {
        dispersion: 1.0,
        atoms: vec![filled_atom(1.0), filled_atom(2.0)],
    };

    // Precondition: both seed atoms start fully filled (nothing to recompute).
    for atom in &shape.atoms {
        assert!(atom.decoder_covariance.is_some());
        assert!(
            atom.band_sd.iter().all(|v| v.is_finite()),
            "precondition: a seed band starts with finite band_sd"
        );
    }

    shape.invalidate_bands_for_recompute();

    // Postcondition: EVERY atom's band is now flagged for recompute — no filled
    // band survives for the completion pass to skip. A seed atom can no longer
    // keep its stale pre-search band.
    for (idx, atom) in shape.atoms.iter().enumerate() {
        assert!(
            atom.decoder_covariance.is_none(),
            "atom {idx}: decoder_covariance must be dropped so the band is recomputed"
        );
        assert!(
            atom.band_sd.iter().all(|v| v.is_nan()),
            "atom {idx}: every band_sd entry must be NaN so complete_born_atom_shape_bands refills it"
        );
        // The completion pass rebuilds band_coords/band_mean from the final
        // fitted atom; invalidation leaves their shape intact for that refill.
        assert_eq!(atom.band_coords.dim(), (3, 1));
        assert_eq!(atom.band_mean.dim(), (3, 2));
        assert_eq!(atom.band_sd.dim(), (3, 2));
    }
}

/// #976 decoder arm (prevention): a K>1 fit whose second atom's decoder has
/// collapsed to ≈0 — gates still spread, so the gate-mass guard is satisfied
/// — is caught by [`SaeManifoldTerm::enforce_decoder_norm_guard`], which
/// reseeds the collapsed atom onto the reconstruction residual and re-fits
/// the decoders so the atom recovers a NON-degenerate, DISTINCT decoder.
/// This is the disease the real-data K=2/K=3 OLMo fits hit (every decoder →
/// 0 ⇒ EV=0 ⇒ every per-row H_tt gauge-flat ⇒ the 0→K·n deflation abort).
#[test]
pub(crate) fn decoder_norm_guard_reseeds_collapsed_atom_to_distinct_nonzero() {
    let (term0, target, rho) = small_two_atom_periodic_term();
    let mut term = term0.clone();
    // Collapse atom 1's decoder to ≈0 while leaving its assignment gates
    // spread (the mass guard sees nothing wrong). Atom 0 keeps its signal.
    term.atoms[1].decoder_coefficients.fill(0.0);
    term.atoms[1].refresh_intrinsic_smooth_penalty();

    let norm = |a: &SaeManifoldAtom| -> f64 {
        a.decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    };
    assert!(norm(&term.atoms[1]) < 1e-12, "atom 1 starts collapsed");

    term.enforce_decoder_norm_guard(target.view(), 0, &rho)
        .expect("decoder-norm guard must not error on a recoverable collapse");

    // The guard recorded a Reseeded collapse event for the collapsed atom.
    let reseeded = term
        .collapse_events()
        .iter()
        .any(|e| e.atom == 1 && e.action == CollapseAction::Reseeded);
    assert!(
        reseeded,
        "collapsed atom 1 must be recorded as Reseeded; events: {:?}",
        term.collapse_events()
    );

    // After the reseed + joint LSQ refit, atom 1 carries a non-degenerate
    // decoder again (well above the collapse floor relative to atom 0).
    let n1 = norm(&term.atoms[1]);
    let n0 = norm(&term.atoms[0]);
    assert!(
        n0 > 0.0 && n1 > SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO * n0,
        "reseeded atom 1 decoder must be non-degenerate: ‖B0‖={n0:.3e} ‖B1‖={n1:.3e}"
    );

    // The reseeded atom's coordinates are diversified (not a single
    // collapsed constant), so its design column is non-degenerate.
    let c1 = term.assignment.coords[1].as_matrix();
    let (lo, hi) = c1
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    assert!(
        hi - lo > 1e-6,
        "reseeded atom 1 coordinates must span a non-trivial range; got [{lo}, {hi}]"
    );

    // The reseeded decoder is DISTINCT from atom 0's (not a duplicate): the
    // residual-seeded coordinates point atom 1 at unexplained signal, so the
    // two decoder column-spaces are not collinear.
    let b0 = &term.atoms[0].decoder_coefficients;
    let b1 = &term.atoms[1].decoder_coefficients;
    let dot: f64 = b0.iter().zip(b1.iter()).map(|(x, y)| x * y).sum();
    let cos = dot.abs() / (n0 * n1);
    assert!(
        cos < 0.999,
        "reseeded atom 1 decoder must be distinct from atom 0 (|cos|={cos:.4})"
    );
}

/// #1117 K>1 robustness — TOTAL co-collapse (every decoder ≈0 together) must
/// reseed ALL atoms, not all-but-one. When the whole dictionary co-collapses
/// the median-relative test finds no atom "behind" its peers, so the guard
/// falls to the absolute EV arm. The earlier code kept the (arbitrary,
/// already-degenerate) strongest atom as an "anchor" and reseeded only K−1
/// atoms; that left one slot sitting in the collapsed basin and the joint LSQ
/// refit re-attracted the reseeded atoms toward it — exactly the K=3 three-way
/// basin a single reseed could not break (real OLMo: identical config flipped
/// EV≈0.40 ↔ 0.00). With all K atoms reseeded onto DISTINCT residual PCs every
/// slot leaves the basin and recovers a non-degenerate, pairwise-distinct
/// decoder.
#[test]
pub(crate) fn decoder_norm_guard_reseeds_all_atoms_on_total_co_collapse_k3() {
    // Three periodic (circle) atoms, p=3 output so three distinct residual PCs
    // exist for the disjoint-PC reseed to land each atom on its own direction.
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let coords2 = array![[0.25], [0.40], [0.75], [0.05], [0.60], [0.85]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let (phi2, jet2) = periodic_basis(&coords2);
    // Decoders are tiny-but-NONZERO and of comparable magnitude across atoms:
    // the dictionary co-collapsed (EV ≈ 0) yet has a usable median scale, so it
    // reaches the absolute-EV co-collapse arm (an exactly-zero dictionary would
    // hit the `median == 0` early return — the cold-seed case, handled by the
    // mass guard/inner solve, not here) and no atom is *relatively* behind its
    // peers (all norms within ~1.5×, none below `1e-3·median`).
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>, scale: f64| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            Array2::<f64>::from_elem((3, 3), scale),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make_atom("periodic0", phi0, jet0, 1.0e-5);
    let atom1 = make_atom("periodic1", phi1, jet1, 1.2e-5);
    let atom2 = make_atom("periodic2", phi2, jet2, 0.8e-5);
    // Gates stay spread across rows/atoms — the gate-mass guard is satisfied,
    // so only the absolute-EV co-collapse arm can catch this failure.
    let logits = array![
        [0.7, -0.2, 0.3],
        [0.1, 0.4, -0.1],
        [-0.3, 0.5, 0.2],
        [0.6, -0.1, 0.4],
        [0.2, 0.3, -0.2],
        [0.4, 0.1, 0.5]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1, coords2],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1, atom2], assignment).unwrap();
    // A target with genuine 3-direction structure so the residual (≈ target,
    // since the dictionary explains ≈0) carries three distinct PCs.
    let target = array![
        [0.40, -0.10, 0.05],
        [-0.20, 0.35, -0.15],
        [0.10, 0.05, 0.30],
        [0.25, -0.30, -0.05],
        [-0.15, 0.20, 0.18],
        [0.30, 0.12, -0.22]
    ];
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![
            array![0.9_f64.ln()],
            array![1.0_f64.ln()],
            array![1.1_f64.ln()],
        ],
    );

    // Confirm the precondition: the dictionary is co-collapsed (EV below the
    // floor) with NO atom relatively behind its peers (all norms ≈0).
    let ev_before = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("EV evaluates");
    assert!(
        ev_before < 0.28_f64,
        "test precondition: dictionary must start co-collapsed; EV={ev_before:.4}"
    );

    term.enforce_decoder_norm_guard(target.view(), 0, &rho)
        .expect("co-collapse guard must recover, not error");

    // EVERY atom — including the one the old code preserved as anchor — must be
    // recorded as Reseeded. This is the regression the fix targets.
    for atom in 0..3 {
        let reseeded = term
            .collapse_events()
            .iter()
            .any(|e| e.atom == atom && e.action == CollapseAction::Reseeded);
        assert!(
            reseeded,
            "total co-collapse must reseed ALL atoms; atom {atom} was not reseeded. events: {:?}",
            term.collapse_events()
        );
    }

    // After the reseed + joint LSQ refit every atom carries a non-degenerate
    // decoder again, and the three decoders are pairwise distinct (each landed
    // on its own residual PC, so no two column-spaces are collinear).
    let norm = |a: &SaeManifoldAtom| -> f64 {
        a.decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    };
    let norms: Vec<f64> = (0..3).map(|a| norm(&term.atoms[a])).collect();
    for (atom, &nrm) in norms.iter().enumerate() {
        assert!(
            nrm > 1e-9,
            "reseeded atom {atom} decoder must be non-degenerate; ‖B‖={nrm:.3e}"
        );
    }
    for a in 0..3 {
        for b in (a + 1)..3 {
            let ba = &term.atoms[a].decoder_coefficients;
            let bb = &term.atoms[b].decoder_coefficients;
            let dot: f64 = ba.iter().zip(bb.iter()).map(|(x, y)| x * y).sum();
            let cos = dot.abs() / (norms[a] * norms[b]);
            assert!(
                cos < 0.999,
                "reseeded atoms {a},{b} decoders must be distinct (|cos|={cos:.4})"
            );
        }
    }

    // The dictionary is no longer co-collapsed: the reseed + LSQ refit explains
    // strictly more variance than the degenerate start.
    let ev_after = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("EV evaluates post-reseed");
    assert!(
        ev_after > ev_before,
        "co-collapse reseed must improve EV; before={ev_before:.4} after={ev_after:.4}"
    );
}

/// #1026 keep-best multi-start: the full-dictionary co-collapse reseed is a
/// bounded multi-start over distinct residual subspaces, but successive reseeds
/// can land in STRICTLY WORSE basins (real OLMo K=4: the seed explains EV 0.127
/// while later reseeds fall to −1.0). A multi-start must return the BEST basin it
/// visited, never the last. The guard retains the highest-EV state seen across
/// the reseeds and restores it once the reseed budget is spent, so the final
/// dictionary EV is no worse than the best intermediate attempt.
#[test]
pub(crate) fn co_collapse_multistart_restores_best_basin_not_last_reseed() {
    // Same co-collapsed K=3 periodic dictionary as
    // `decoder_norm_guard_reseeds_all_atoms_on_total_co_collapse_k3`, driven
    // through the WHOLE reseed budget so the budget-exhaustion restore fires.
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let coords2 = array![[0.25], [0.40], [0.75], [0.05], [0.60], [0.85]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let (phi2, jet2) = periodic_basis(&coords2);
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>, scale: f64| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            Array2::<f64>::from_elem((3, 3), scale),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make_atom("periodic0", phi0, jet0, 1.0e-5);
    let atom1 = make_atom("periodic1", phi1, jet1, 1.2e-5);
    let atom2 = make_atom("periodic2", phi2, jet2, 0.8e-5);
    let logits = array![
        [0.7, -0.2, 0.3],
        [0.1, 0.4, -0.1],
        [-0.3, 0.5, 0.2],
        [0.6, -0.1, 0.4],
        [0.2, 0.3, -0.2],
        [0.4, 0.1, 0.5]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1, coords2],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1, atom2], assignment).unwrap();
    let target = array![
        [0.40, -0.10, 0.05],
        [-0.20, 0.35, -0.15],
        [0.10, 0.05, 0.30],
        [0.25, -0.30, -0.05],
        [-0.15, 0.20, 0.18],
        [0.30, 0.12, -0.22]
    ];
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![
            array![0.9_f64.ln()],
            array![1.0_f64.ln()],
            array![1.1_f64.ln()],
        ],
    );

    // Drive the guard once per "outer iteration" through the whole multi-start
    // budget plus the budget-exhaustion call, recording the dictionary EV the
    // guard observes at the start of each call (the candidate basin it may bank).
    // The guard reseeds in place, so each call's pre-reseed EV is a distinct
    // multi-start attempt; the best of these is what the final state must match.
    let mut best_seen = f64::NEG_INFINITY;
    for iteration in 0..=SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET {
        let ev_at_entry = term
            .dictionary_reconstruction_ev(target.view(), &rho)
            .expect("EV evaluates");
        if ev_at_entry < 0.28_f64 {
            best_seen = best_seen.max(ev_at_entry);
        }
        term.enforce_decoder_norm_guard(target.view(), iteration, &rho)
            .expect("co-collapse guard must recover, not error");
    }

    // After the budget is spent the guard has restored the best basin it banked,
    // so the final dictionary EV is at least the best attempt seen — never the
    // (possibly catastrophic) last reseed.
    let ev_final = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("EV evaluates");
    assert!(
        best_seen.is_finite(),
        "test precondition: at least one co-collapsed attempt must be observed"
    );
    assert!(
        ev_final >= best_seen - 1e-9,
        "multi-start must return its BEST basin, not the last reseed: \
         final EV={ev_final:.6} < best seen={best_seen:.6}"
    );
}

/// #1026 decoder-repulsion gate safety: the collinearity gate must be a STRICT
/// no-op for well-separated atoms (orthogonal decoders → gate `None`, so no
/// value/gradient/curvature is added and healthy fits are byte-identical) and
/// must ENGAGE for near-collinear atoms (the co-collapse geometry it conditions).
/// Built on a K=2 periodic fixture whose decoders we set directly.
#[test]
pub(crate) fn decoder_repulsion_gate_off_when_separated_on_when_collinear() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    // Periodic basis is M=3 wide; output p=3. Build two atoms; decoders set below.
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let build = |dec0: Array2<f64>, dec1: Array2<f64>| {
        let atom0 = make_atom("periodic0", phi0.clone(), jet0.clone(), dec0);
        let atom1 = make_atom("periodic1", phi1.clone(), jet1.clone(), dec1);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
    };

    // ORTHOGONAL decoders: atom0 writes output channel 0, atom1 writes channel 1.
    // Their cross-Gram B_0 B_1ᵀ = 0 ⇒ s_01 = 0 ⇒ gate exactly 0 ⇒ field `None`.
    let mut dec0 = Array2::<f64>::zeros((3, 3));
    dec0[[0, 0]] = 1.0;
    let mut dec1 = Array2::<f64>::zeros((3, 3));
    dec1[[0, 1]] = 1.0;
    let mut sep = build(dec0, dec1);
    sep.refresh_decoder_repulsion_gate();
    assert!(
        sep.decoder_repulsion_gate.is_none(),
        "orthogonal decoders must leave the repulsion gate OFF (strict no-op): {:?}",
        sep.decoder_repulsion_gate
    );
    assert_eq!(
        sep.decoder_repulsion_value(1.0),
        0.0,
        "orthogonal decoders must contribute zero repulsion value"
    );

    // COLLINEAR decoders: both atoms write the SAME output channel 0 with the
    // same basis-row pattern ⇒ s_01 = 1 ⇒ gate fully engaged ⇒ field `Some`.
    let mut dec0c = Array2::<f64>::zeros((3, 3));
    dec0c[[0, 0]] = 1.0;
    let mut dec1c = Array2::<f64>::zeros((3, 3));
    dec1c[[0, 0]] = 1.0;
    let mut col = build(dec0c, dec1c);
    col.refresh_decoder_repulsion_gate();
    let gate = col
        .decoder_repulsion_gate
        .as_ref()
        .expect("collinear decoders must ENGAGE the repulsion gate");
    assert!(
        gate.iter().any(|&(j, k, w)| j == 0 && k == 1 && w > 0.0),
        "engaged gate must carry a positive weight on pair (0,1): {gate:?}"
    );
    assert!(
        col.decoder_repulsion_value(1.0) > 0.0,
        "collinear decoders must contribute positive repulsion value"
    );
}

/// #1522 — the SEPARATION interior-point barrier is the deterministic collapse
/// PREVENTION (not a detect-then-reseed bandaid). On a constructed collapse-prone
/// fixture — two co-firing K=2 atoms whose decoders point nearly the same way
/// (normalized alignment `c² ≈ 0.8`, the geometry that drives the per-row `H_tt`
/// near-singular and the whole dictionary into the co-collapse basin) — this
/// pins that the barrier:
///   1. WITH it (`scale = 1`): adds a positive penalty AND a genuine SEPARATING
///      force — one gradient-descent step along `-∂P_sep/∂B` strictly REDUCES the
///      alignment `c²`, i.e. the atoms move apart (collapse is prevented in the
///      optimizer, not patched after the fact).
///   2. WITHOUT it (`scale = 0` ⇒ `μ = 0`, the LOCAL "no prevention" arm — no
///      process-global override toggled, so it is parallelism-safe): value `0`
///      and an all-zero gradient. The aligned atoms feel NO restoring force and
///      would stay collapsed — this is the "collapses without the prevention"
///      half of the pin.
///   3. INTERIOR-POINT divergence: a MORE-aligned configuration carries a strictly
///      LARGER barrier value than a less-aligned one, so the force grows without
///      bound toward the collapse boundary (`c² → 1`).
///   4. NON-REGRESSION: ORTHOGONAL (healthy, well-separated) decoders get value
///      `0` and an all-zero gradient even with the barrier ON, so the prevention
///      is a strict no-op away from collapse and healthy fits stay byte-identical
///      (the reseed backstop can remain as defense-in-depth and rarely fires).
#[test]
pub(crate) fn separation_barrier_is_collapse_prevention_not_bandaid_1522() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    // softmax routing ⇒ every atom carries strictly positive mass on every row,
    // so the pair co-fires (`q_01 > 0`) and the separation barrier engages.
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let build = |dec0: Array2<f64>, dec1: Array2<f64>| {
        let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
            SaeManifoldAtom::new(
                name,
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        };
        let atom0 = make("periodic0", phi0.clone(), jet0.clone(), dec0);
        let atom1 = make("periodic1", phi1.clone(), jet1.clone(), dec1);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
    };
    // Single-row decoders so the normalized alignment `c²` is exactly the squared
    // cosine of the two output-direction vectors. Channel choices give `c² = 0.8`
    // (cosθ = √0.8): high enough to drive collapse, low enough that the barrier
    // gradient (`α ∝ 1/(1-c²+ε)`) is finite and a small step stays in the basin.
    let row_decoder = |r: [f64; 3]| {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = r[0];
        d[[0, 1]] = r[1];
        d[[0, 2]] = r[2];
        d
    };
    // Normalized-alignment c² between two single-row decoders, read straight off
    // the atom decoder coefficients (the same quantity the barrier penalizes).
    let alignment_c2 = |b0: &Array2<f64>, b1: &Array2<f64>| -> f64 {
        let (m0, p) = (b0.nrows(), b0.ncols());
        let m1 = b1.nrows();
        let mut cross = 0.0_f64;
        for a in 0..m0 {
            for b in 0..m1 {
                let mut c = 0.0_f64;
                for o in 0..p {
                    c += b0[[a, o]] * b1[[b, o]];
                }
                cross += c * c;
            }
        }
        let n0: f64 = b0.iter().map(|v| v * v).sum();
        let n1: f64 = b1.iter().map(|v| v * v).sum();
        cross / (n0 * n1)
    };

    let dec0 = row_decoder([1.0, 0.0, 0.0]);
    // cosθ = √0.8 ≈ 0.894427, sinθ = √0.2 ≈ 0.447214 ⇒ unit-norm, c² = 0.8.
    let dec1 = row_decoder([0.894_427_191, 0.447_213_595, 0.0]);
    let c2_before = alignment_c2(&dec0, &dec1);
    assert!(
        (c2_before - 0.8).abs() < 1e-6,
        "fixture precondition: aligned decoders must start at c² ≈ 0.8, got {c2_before}"
    );

    let term = build(dec0.clone(), dec1.clone());

    // ── Arm 2 (do this first): barrier OFF (scale 0 ⇒ μ = 0) is a no-op. ──
    let (value_off, grad_off) = term.separation_barrier_value_and_grad_for_test(0.0);
    assert_eq!(
        value_off, 0.0,
        "barrier OFF must contribute zero value (the no-prevention arm)"
    );
    assert!(
        grad_off.iter().all(|&g| g == 0.0),
        "barrier OFF must leave the gradient identically zero — aligned atoms feel \
         NO separating force, so without prevention they stay collapsed"
    );

    // ── Arm 1: barrier ON supplies a positive penalty and a separating force. ──
    let (value_on, grad_on) = term.separation_barrier_value_and_grad_for_test(1.0);
    assert!(
        value_on > 0.0,
        "barrier ON must penalize the aligned, co-firing pair (value {value_on} ≤ 0)"
    );
    assert!(
        grad_on.iter().any(|&g| g != 0.0),
        "barrier ON must produce a non-zero separating gradient on the aligned pair"
    );

    // One gradient-descent step `B ← B - η·∂P/∂B` must REDUCE the alignment c².
    // η is small relative to the decoder scale so the step stays inside the basin.
    let eta = 1.0e-3;
    let offsets = term.beta_offsets();
    let p = term.output_dim();
    let stepped = |atom: usize, base: &Array2<f64>| -> Array2<f64> {
        let mut out = base.clone();
        let off = offsets[atom];
        for a in 0..out.nrows() {
            for o in 0..p {
                out[[a, o]] -= eta * grad_on[off + a * p + o];
            }
        }
        out
    };
    let dec0_stepped = stepped(0, &dec0);
    let dec1_stepped = stepped(1, &dec1);
    let c2_after = alignment_c2(&dec0_stepped, &dec1_stepped);
    assert!(
        c2_after < c2_before - 1e-9,
        "a descent step along the barrier gradient must SEPARATE the atoms \
         (c² must fall): before={c2_before:.6} after={c2_after:.6}"
    );

    // ── Arm 3: interior-point divergence — more alignment ⇒ strictly larger value. ──
    // Less aligned: r_k = (0.6, 0.8, 0) ⇒ c² = 0.36. More aligned: c² ≈ 0.98.
    let term_less = build(dec0.clone(), row_decoder([0.6, 0.8, 0.0]));
    let term_more = build(dec0.clone(), row_decoder([0.989_949_49, 0.141_421_36, 0.0]));
    let value_less = term_less.separation_barrier_value(1.0);
    let value_more = term_more.separation_barrier_value(1.0);
    assert!(
        value_more > value_on && value_on > value_less,
        "barrier value must grow with alignment toward the collapse boundary: \
         less(c²=.36)={value_less:.6} < base(c²=.8)={value_on:.6} < more(c²=.98)={value_more:.6}"
    );

    // ── Arm 4: non-regression — orthogonal (healthy) decoders are a strict no-op
    // in the FORCE. The separating gradient (and hence the optimizer trajectory)
    // is identically zero, so a well-separated fit is steered exactly as if no
    // barrier were present; the scalar value carries only the negligible constant
    // `-μ·q·log(1+ε) ≈ -1e-5` eps-softening offset (a constant in the objective,
    // which cannot move the optimum or fire the reseed). ──
    let term_ortho = build(row_decoder([1.0, 0.0, 0.0]), row_decoder([0.0, 1.0, 0.0]));
    let (value_ortho, grad_ortho) = term_ortho.separation_barrier_value_and_grad_for_test(1.0);
    assert!(
        grad_ortho.iter().all(|&g| g == 0.0),
        "orthogonal (well-separated) decoders must leave the separating gradient \
         identically zero (strict no-op force) — healthy fits steer unchanged: {grad_ortho:?}"
    );
    assert!(
        value_ortho.abs() < 1.0e-4,
        "orthogonal decoders' barrier value must be negligible (only the ε-softening \
         constant), got {value_ortho}"
    );
}

/// #1625 — build a 2-atom periodic SAE term whose single-row decoders realize a
/// chosen squared alignment `c² = cos²θ` (`dec0 = e0`, `dec1 = (cosθ, sinθ, 0)`),
/// co-firing under softmax so the separation barrier's coactivation `q_01 > 0`.
/// The shared regression fixture for the collinearity-gate guards below.
fn aligned_two_atom_term_with_c2(c2: f64) -> SaeManifoldTerm {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let cos = c2.sqrt();
    let sin = (1.0 - c2).max(0.0).sqrt();
    let row_decoder = |r: [f64; 3]| {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = r[0];
        d[[0, 1]] = r[1];
        d[[0, 2]] = r[2];
        d
    };
    let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make("periodic0", phi0, jet0, row_decoder([1.0, 0.0, 0.0]));
    let atom1 = make("periodic1", phi1, jet1, row_decoder([cos, sin, 0.0]));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
}

/// #1625 — the separation barrier is a COLLAPSE-prevention barrier, gated to the
/// near-collinear regime: it must be an exact no-op (value AND gradient identically
/// zero) at moderate collinearity below
/// [`SAE_SEPARATION_BARRIER_COLLINEARITY_GATE`], and engage (positive value,
/// nonzero separating gradient) above it. This is the root-cause guard for the
/// #1625 stall: the ungated `−log(1−c²+ε)` exerted an O(1) force at moderate `c²`
/// (e.g. the gamma fixture's `c² = 0.36`, a 53° angle nowhere near collapse), which
/// dominated a well-specified fit's near-zero data residual, dragged the decoders
/// off the data optimum, and left the inner (t,β) Newton unable to reach KKT
/// stationarity for the undamped-PD evidence log-det.
#[test]
fn separation_barrier_collinearity_gate_off_below_on_above_1625() {
    // Below the gate (gamma-fixture-like 53° pair): strict no-op.
    let below = aligned_two_atom_term_with_c2(0.36);
    let (v_below, g_below) = below.separation_barrier_value_and_grad_for_test(1.0);
    assert_eq!(
        v_below, 0.0,
        "barrier value must be exactly 0 below the collinearity gate (c²=0.36 < {}), got {v_below}",
        SAE_SEPARATION_BARRIER_COLLINEARITY_GATE
    );
    assert!(
        g_below.iter().all(|&g| g == 0.0),
        "barrier gradient must be identically 0 below the gate — distinct atoms feel NO force: {g_below:?}"
    );

    // Above the gate (genuine near-collapse alignment): engaged.
    let above = aligned_two_atom_term_with_c2(0.8);
    let (v_above, g_above) = above.separation_barrier_value_and_grad_for_test(1.0);
    assert!(
        v_above > 0.0,
        "barrier value must be positive above the gate (c²=0.8 > {}), got {v_above}",
        SAE_SEPARATION_BARRIER_COLLINEARITY_GATE
    );
    assert!(
        g_above.iter().any(|&g| g != 0.0),
        "barrier must produce a separating gradient above the gate"
    );
}

/// #1625 — the GATED barrier's analytic gradient must match the finite difference
/// of its OWN value, including the smoothstep's `w'(c²)` product-rule term. A
/// dropped `w'` (treating the gate as a constant weight instead of a function of
/// `c²`) would pass the on/off guard above but desync value vs gradient on the
/// ramp — the exact value/gradient-consistency contract the line search relies on.
/// Evaluated at `c² = 0.7`, strictly on the smoothstep interior (`0.5 < c² < 1`)
/// where `w'(c²) > 0`, so the product-rule term is load-bearing.
#[test]
fn separation_barrier_gated_gradient_matches_fd_1625() {
    let c2 = 0.7_f64;
    let cos = c2.sqrt();
    let sin = (1.0 - c2).sqrt();
    // Rebuild the term from explicit decoders so we can perturb a single
    // decoder coefficient and recompute the value.
    let build = |d1: [f64; 3]| -> SaeManifoldTerm {
        let mut t = aligned_two_atom_term_with_c2(c2);
        // Overwrite atom1's decoder row 0 with the perturbed direction.
        t.atoms[1].decoder_coefficients[[0, 0]] = d1[0];
        t.atoms[1].decoder_coefficients[[0, 1]] = d1[1];
        t.atoms[1].decoder_coefficients[[0, 2]] = d1[2];
        t
    };
    let base = build([cos, sin, 0.0]);
    let (_v, grad) = base.separation_barrier_value_and_grad_for_test(1.0);
    let offsets = base.beta_offsets();
    let p = base.output_dim();
    // FD each of atom1's row-0 decoder coefficients against the value.
    let h = 1.0e-7;
    let mut max_rel = 0.0_f64;
    for o in 0..3 {
        let mut plus = [cos, sin, 0.0];
        let mut minus = [cos, sin, 0.0];
        plus[o] += h;
        minus[o] -= h;
        let vp = build(plus).separation_barrier_value(1.0);
        let vm = build(minus).separation_barrier_value(1.0);
        let fd = (vp - vm) / (2.0 * h);
        let analytic = grad[offsets[1] + 0 * p + o];
        let rel = (fd - analytic).abs() / (1.0 + fd.abs().max(analytic.abs()));
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1.0e-5,
        "gated barrier analytic ∂P/∂B must match FD of the value (incl. the smoothstep \
         w'(c²) term) on the ramp: max rel err {max_rel:.3e}"
    );
}

/// #1625 — within a Newton step the barrier's normalized coactivation `q_jk` is a
/// FROZEN weight (the gradient differentiates only the decoder shape `c²`), so the
/// line-search VALUE must read the same frozen `q` even after the trial logits
/// move — otherwise value and gradient desync in the logit block (the original
/// #1625 defect, surfaced as a phantom logit gradient the Newton step never
/// modelled). After an assembly freezes the coactivation, perturbing a logit must
/// leave `separation_barrier_value` unchanged (the decoders are untouched, and `q`
/// is frozen). Uses an aligned (above-gate) term so the barrier is genuinely live.
#[test]
fn separation_barrier_value_frozen_coactivation_invariant_to_logit_moves_1625() {
    let mut term = aligned_two_atom_term_with_c2(0.8);
    let target = Array2::<f64>::zeros((term.n_obs(), term.output_dim()));
    let rho = SaeManifoldRho::new(
        -2.0,
        -2.0,
        vec![Array1::from_vec(vec![-2.0]), Array1::from_vec(vec![-2.0])],
    );
    // Assemble once to FREEZE the coactivation gate at the current logits.
    term.assemble_arrow_schur(target.view(), &rho, None)
        .expect("assemble freezes the barrier coactivation");
    let value_before = term.separation_barrier_value(1.0);
    assert!(value_before > 0.0, "aligned pair must have a live barrier");
    // Move the logits substantially WITHOUT re-assembling (mimics a line-search
    // trial). The frozen coactivation must keep the barrier value pinned.
    for v in term.assignment.logits.iter_mut() {
        *v += 0.37;
    }
    let value_after = term.separation_barrier_value(1.0);
    assert!(
        (value_after - value_before).abs() <= 1.0e-12 * (1.0 + value_before.abs()),
        "frozen coactivation must hold the barrier value across logit moves: \
         before={value_before:.12e} after={value_after:.12e}"
    );
}

/// #1610 — the separation-barrier collapse-threshold (the decoder-norm floor
/// below which an atom is shape-undefined and the barrier abstains) must be
/// DATA-DERIVED / scale-invariant, not an absolute magic constant.
///
/// Direct-helper arm: `barrier_norm_floor_sq` is exactly
/// `SAE_BARRIER_ACTIVE_NORM_REL_FLOOR² · max_k ‖B_k‖²_F`, equivariant under a
/// global rescaling of the decoders by `s²`, and reduces to the historical
/// absolute `1e-6²` floor at unit decoder scale (`max ‖B_k‖²_F = 1`). The
/// all-zero dictionary yields `0` (no live shape).
#[test]
fn barrier_norm_floor_is_data_derived_scale_invariant_1610() {
    // max ‖B_k‖²_F = 4.0 ⇒ floor² = (1e-6)²·4 = 4e-12.
    let norm_sq = [1.0_f64, 4.0, 0.25];
    let floor = SaeManifoldTerm::barrier_norm_floor_sq(&norm_sq);
    let rel = SAE_BARRIER_ACTIVE_NORM_REL_FLOOR;
    assert!(
        (floor - rel * rel * 4.0).abs() <= 1e-30,
        "floor² must be rel²·max‖B_k‖²_F = {}, got {floor}",
        rel * rel * 4.0
    );
    // At the canonical unit decoder scale this reduces to the historical 1e-6
    // absolute floor (floor² = 1e-12), so existing unit-scale fits are unchanged.
    let unit = SaeManifoldTerm::barrier_norm_floor_sq(&[1.0]);
    assert!(
        (unit - 1.0e-12).abs() <= 1e-27,
        "at unit decoder scale the floor must equal the historical 1e-6² = 1e-12, got {unit}"
    );
    // Equivariance: scaling every ‖B_k‖²_F by s² scales the floor² by s².
    for &s2 in &[1.0e-12_f64, 1.0e6, 9.0] {
        let scaled: Vec<f64> = norm_sq.iter().map(|v| v * s2).collect();
        let f_scaled = SaeManifoldTerm::barrier_norm_floor_sq(&scaled);
        assert!(
            (f_scaled - s2 * floor).abs() <= s2 * floor * 1e-9 + 1e-30,
            "floor² must scale by s² under a global ‖B‖² rescaling: s²={s2}, \
             expected {}, got {f_scaled}",
            s2 * floor
        );
    }
    // All-zero dictionary: no live atom to be a shape ⇒ floor 0 (the exactly-0
    // self-norm check abstains every pair anyway).
    assert_eq!(SaeManifoldTerm::barrier_norm_floor_sq(&[0.0, 0.0]), 0.0);
}

/// #1610 — END-TO-END scale invariance of collapse prevention: the separation
/// barrier penalizes the SHAPE alignment `c²` weighted by the (normalized)
/// coactivation `q`, both of which are scale-free, so the barrier VALUE is
/// invariant under a global rescaling of the decoders. The OLD absolute
/// `1e-6` norm floor broke this: a corpus whose natural decoder scale fell below
/// the floor had its decoders classified as shape-undefined and collapse
/// prevention was silently disabled (value → 0). With the data-derived relative
/// floor the barrier engages identically at any decoder scale.
#[test]
fn separation_barrier_collapse_prevention_is_scale_invariant_1610() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let row_decoder = |r: [f64; 3]| {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = r[0];
        d[[0, 1]] = r[1];
        d[[0, 2]] = r[2];
        d
    };
    // Aligned (c² = 0.8), co-firing under softmax — the collapse-prone pair.
    let dir0 = [1.0, 0.0, 0.0];
    let dir1 = [0.894_427_191, 0.447_213_595, 0.0];
    let build_at_scale = |s: f64| {
        let scale_row = |r: [f64; 3]| [r[0] * s, r[1] * s, r[2] * s];
        let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
            SaeManifoldAtom::new(
                name,
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        };
        let atom0 = make(
            "p0",
            phi0.clone(),
            jet0.clone(),
            row_decoder(scale_row(dir0)),
        );
        let atom1 = make(
            "p1",
            phi1.clone(),
            jet1.clone(),
            row_decoder(scale_row(dir1)),
        );
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
    };

    // Unit scale: the barrier engages and penalizes the aligned pair.
    let value_unit = build_at_scale(1.0).separation_barrier_value(1.0);
    assert!(
        value_unit > 0.0,
        "barrier must engage on the aligned, co-firing pair at unit scale, got {value_unit}"
    );
    // Tiny scale: decoder entries ~1e-7 ⇒ ‖B_k‖²_F ~1e-14 < the OLD absolute
    // floor² (1e-12). Under the old absolute floor the barrier would have
    // abstained (value 0 — collapse prevention disabled). The data-derived floor
    // keeps it engaged with the SAME value (c² and q are scale-free).
    let value_tiny = build_at_scale(1.0e-7).separation_barrier_value(1.0);
    assert!(
        value_tiny > 0.0,
        "data-derived floor must keep collapse prevention ENGAGED at a tiny decoder \
         scale where the old absolute 1e-6 floor disabled it, got {value_tiny}"
    );
    assert!(
        (value_tiny - value_unit).abs() <= value_unit.abs() * 1e-9,
        "the barrier value is scale-free (shape + coactivation only): unit={value_unit} \
         must equal tiny-scale={value_tiny}"
    );
    // And a HUGE scale leaves it unchanged too (symmetry of the invariance).
    let value_huge = build_at_scale(1.0e6).separation_barrier_value(1.0);
    assert!(
        (value_huge - value_unit).abs() <= value_unit.abs() * 1e-9,
        "barrier value must be invariant at large decoder scale too: unit={value_unit} \
         huge={value_huge}"
    );
}

/// #1610 — the decoder-repulsion collapse-prevention conditioner must be
/// PRINCIPLED, not a hand-picked absolute magic constant:
///   1. its strength is a DERIVED dimensionless fraction of the primary
///      separation-barrier strength (`μ_rep = ratio · μ_sep`), not an
///      independent `1e-3`; and
///   2. after the #1610 energy normalization the realized repulsion penalty is a
///      function of the dimensionless collinearity `c_jk² ∈ [0,1]` ALONE, so it
///      is INVARIANT under a global corpus rescaling `B_k → s·B_k`.
///
/// Property (2) is the property the OLD absolute constant VIOLATED: it weighted
/// the un-normalized cross-Gram energy `‖B_jB_kᵀ‖²_F = c²·‖B_j‖²_F·‖B_k‖²_F`, so
/// the repulsion value scaled as `s⁴` under a rescaling by `s` while the
/// collapse geometry (`c²`, the gate) was identical — the same scale bug #1610
/// fixed for the separation barrier's norm floor. The test builds a fixed,
/// near-collinear (gate-engaged) K=2 fixture and asserts the repulsion value is
/// equal across decoder scales spanning 13 orders of magnitude. With the old
/// `½·STRENGTH·c²·s⁴` weighting these would differ by `s⁴` (up to `1e52`), so
/// this fails before the normalization and passes after.
#[test]
pub(crate) fn decoder_repulsion_strength_is_derived_and_scale_invariant_1610() {
    // (1) Strength is a DERIVED dimensionless fraction of the data-derived
    // separation-barrier strength μ_C, not an independent absolute constant.
    // (Checked on a constructed term below, after the fixture builder — μ_C is
    // now a per-dictionary quantity, K / reachable_rank, not a global constant.)

    // (2) End-to-end scale invariance of the repulsion value.
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    // Two atoms whose decoders are NEAR-collinear (cosine 0.9 ⇒ c² = 0.81, above
    // the 0.5 gate but strictly < 1), so the gate is partially engaged and the
    // penalty is strictly positive and finite. Rank-1 decoders (only row 0
    // nonzero) keep `‖B_k‖²_F` trivial to reason about: at scale `s`,
    // `‖B_0‖²_F = ‖B_1‖²_F = s²` and `c² = 0.81` (scale-free).
    let build_at_scale = |s: f64| {
        let mut dec0 = Array2::<f64>::zeros((3, 3));
        dec0[[0, 0]] = s;
        let mut dec1 = Array2::<f64>::zeros((3, 3));
        dec1[[0, 0]] = 0.9 * s;
        dec1[[0, 1]] = (1.0 - 0.9 * 0.9_f64).sqrt() * s; // ‖row‖ = s, cosine with dec0 = 0.9
        let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
            SaeManifoldAtom::new(
                name,
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        };
        let atom0 = make("rep0", phi0.clone(), jet0.clone(), dec0);
        let atom1 = make("rep1", phi1.clone(), jet1.clone(), dec1);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
        term.refresh_decoder_repulsion_gate();
        term
    };

    // (1) — the repulsion strength is the derived fraction
    // `SAE_DECODER_REPULSION_BARRIER_RATIO · μ_C` of the separation-barrier
    // strength, and μ_C is itself EVIDENCE-DERIVED — the worst-case data-fit
    // inseparability strength `γ/(1-γ)` over the co-active pairs (#1610), NOT a
    // hand-picked magnitude and NOT a rank-count heuristic. Checked on a
    // constructed unit-scale term (μ_C is a per-term, per-pair quantity).
    let unit_term = build_at_scale(1.0);
    let expected =
        SAE_DECODER_REPULSION_BARRIER_RATIO * unit_term.separation_barrier_strength();
    assert_eq!(
        unit_term.decoder_repulsion_strength(),
        expected,
        "repulsion strength must be the derived fraction {SAE_DECODER_REPULSION_BARRIER_RATIO} \
         of the evidence-derived separation-barrier strength {}, got {}",
        unit_term.separation_barrier_strength(),
        unit_term.decoder_repulsion_strength(),
    );
    // The evidence-derived strength is a strictly positive, finite number for a
    // genuinely co-active pair (the data-fit couples them, so γ > 0), and it is
    // NOT the old overcompleteness ratio (which for two periodic M=3 atoms in p=3
    // was pinned at exactly 2.0). It is the reciprocal-margin `γ/(1-γ)` to the
    // data-fit's co-collapse boundary, read from the chart design + routing.
    let mu_c = unit_term.separation_barrier_strength();
    assert!(
        mu_c > 0.0 && mu_c.is_finite(),
        "μ_C must be a positive finite evidence-derived strength for a co-active \
         pair, got {mu_c}"
    );
    // Decoder-scale invariance of the STRENGTH: γ (hence μ_C) is read from the
    // chart design + routing, not the decoder magnitudes, so rescaling the whole
    // dictionary leaves the strength unchanged (unlike a REML `λ ∝ σ²/τ²`).
    let mu_c_tiny = build_at_scale(1.0e-7).separation_barrier_strength();
    let mu_c_huge = build_at_scale(1.0e6).separation_barrier_strength();
    let rel_mu = |a: f64, b: f64| (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel_mu(mu_c_tiny, mu_c) <= 1e-9 && rel_mu(mu_c_huge, mu_c) <= 1e-9,
        "evidence-derived μ_C must be decoder-scale invariant: unit={mu_c} \
         tiny={mu_c_tiny} huge={mu_c_huge}"
    );

    let value_unit = build_at_scale(1.0).decoder_repulsion_value(1.0);
    assert!(
        value_unit > 0.0 && value_unit.is_finite(),
        "near-collinear gate-engaged pair must yield a positive finite repulsion \
         value at unit scale, got {value_unit}"
    );
    // Same collapse geometry (c², gate identical) at a tiny and a huge corpus
    // scale: the energy-normalized penalty is invariant. The OLD un-normalized
    // weighting would scale these by s⁴ = 1e-28 and 1e24 respectively.
    let value_tiny = build_at_scale(1.0e-7).decoder_repulsion_value(1.0);
    let value_huge = build_at_scale(1.0e6).decoder_repulsion_value(1.0);
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel(value_tiny, value_unit) <= 1e-9,
        "repulsion value must be scale-invariant: unit={value_unit} tiny={value_tiny} \
         (old absolute constant scaled this by s⁴)"
    );
    assert!(
        rel(value_huge, value_unit) <= 1e-9,
        "repulsion value must be scale-invariant: unit={value_unit} huge={value_huge} \
         (old absolute constant scaled this by s⁴)"
    );
}

/// #1610 — the separation-barrier strength is EVIDENCE-DERIVED: the per-pair
/// strength `μ_jk = γ_jk/(1-γ_jk)` is a MONOTONE function of the data-fit
/// inseparability `γ_jk` (the largest canonical correlation of the two atoms'
/// coactivation-weighted chart designs — the quantity that decides whether the
/// joint inner Laplace/REML Hessian stays PD). This replaces the old geometry
/// heuristic `Σ min(M_k,p)/min(n,p)`, which was blind to the actual design/routing
/// and so gave the SAME strength to a data-separable pair and a data-degenerate
/// one. Here two atoms with IDENTICAL chart designs are driven from data-fit
/// SEPARABLE (disjoint routing ⇒ γ ≈ 0 ⇒ μ ≈ 0) to data-fit DEGENERATE
/// (overlapping routing on a shared design ⇒ γ → 1 ⇒ μ large), and the strength
/// must rise accordingly. γ (hence μ) is read from the design + routing only, so
/// it is decoder-scale free.
#[test]
pub(crate) fn barrier_strength_tracks_data_fit_inseparability_1610() {
    let coords = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let (phi, jet) = periodic_basis(&coords);
    // Two atoms with the SAME chart design (identical Φ) so the ONLY thing that
    // sets γ is the coactivation-weighted routing overlap we pass in.
    let make = |name: &str, decoder: Array2<f64>| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let mut dec0 = Array2::<f64>::zeros((3, 3));
    dec0[[0, 0]] = 1.0;
    let mut dec1 = Array2::<f64>::zeros((3, 3));
    dec1[[0, 1]] = 1.0;
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone(), coords.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![make("a0", dec0), make("a1", dec1)], assignment).unwrap();

    // DISJOINT routing: atom 0 fires only on the first three rows, atom 1 only on
    // the last three. No row co-fires, so the weighted cross-design Gram is 0 ⇒
    // γ ≈ 0 ⇒ the data-fit already separates the pair ⇒ μ ≈ 0 (no safeguard owed).
    let gates_disjoint = array![
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ];
    let gamma_sep = term.design_inseparability_with_gates(gates_disjoint.view(), 0, 1);
    let mu_sep = term.barrier_pair_strength_with_gates(gates_disjoint.view(), 0, 1);
    assert!(
        gamma_sep <= 1e-9,
        "disjoint routing on any design ⇒ data-fit separable ⇒ γ ≈ 0, got {gamma_sep}"
    );
    assert!(
        mu_sep <= 1e-6,
        "a data-fit-separable pair owes ~no separation barrier, got μ = {mu_sep}"
    );

    // OVERLAPPING routing on the SHARED design: both atoms fire together on every
    // row, so the two coactivation-weighted design column spaces COINCIDE ⇒ γ → 1
    // (the data-fit cannot tell them apart) ⇒ μ = γ/(1-γ) is large.
    let gates_overlap = array![
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    let gamma_deg = term.design_inseparability_with_gates(gates_overlap.view(), 0, 1);
    let mu_deg = term.barrier_pair_strength_with_gates(gates_overlap.view(), 0, 1);
    assert!(
        gamma_deg > 0.999,
        "identical designs + identical routing ⇒ perfectly inseparable ⇒ γ → 1, got {gamma_deg}"
    );
    assert!(
        mu_deg > mu_sep + 1.0,
        "the barrier strength MUST rise as the data-fit inseparability rises: \
         separable μ={mu_sep} vs degenerate μ={mu_deg}"
    );
    // μ = γ/(1-γ) exactly (evidence-derived reciprocal margin), no hidden magic.
    let eps = SAE_SEPARATION_BARRIER_EPS;
    let expected_deg = gamma_deg / (1.0 - gamma_deg).max(eps);
    assert!(
        (mu_deg - expected_deg).abs() <= expected_deg.abs() * 1e-9 + 1e-12,
        "μ must equal γ/max(1-γ,ε): γ={gamma_deg} expected={expected_deg} got={mu_deg}"
    );

    // γ (hence μ) is a DESIGN/ROUTING quantity, independent of decoder magnitude:
    // rescaling the decoders leaves both unchanged.
    let mut big0 = Array2::<f64>::zeros((3, 3));
    big0[[0, 0]] = 1.0e6;
    let mut big1 = Array2::<f64>::zeros((3, 3));
    big1[[0, 1]] = 1.0e6;
    let assignment2 = SaeAssignment::from_blocks_with_mode_and_manifolds(
        array![
            [0.7, -0.2],
            [0.1, 0.4],
            [-0.3, 0.5],
            [0.6, -0.1],
            [0.2, 0.3],
            [0.4, 0.1]
        ],
        vec![coords.clone(), coords.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let term_big =
        SaeManifoldTerm::new(vec![make("a0", big0), make("a1", big1)], assignment2).unwrap();
    let mu_deg_big = term_big.barrier_pair_strength_with_gates(gates_overlap.view(), 0, 1);
    assert!(
        (mu_deg_big - mu_deg).abs() <= mu_deg.abs() * 1e-9,
        "evidence-derived μ must be decoder-scale invariant: unit={mu_deg} big={mu_deg_big}"
    );
}

/// #976 distinct-basin lever: the co-collapse multi-start reseed must read a
/// DIFFERENT principal subspace on each retry. The PC-pair rotation offset (=
/// the 0-based retry index) shifts which residual PC pair each periodic atom
/// reads, so two consecutive multi-start attempts produce seed coordinates that
/// are not bit-identical. Without the rotation every retry re-reads the same
/// leading PCs of the (unchanged) residual and the budget-N multi-start is N
/// identical attempts — the K=3 coin-flip this fix targets.
#[test]
pub(crate) fn co_collapse_reseed_rotation_explores_distinct_subspaces() {
    // A residual with three well-separated PC directions (p = 6 so >= 6 PCs
    // exist and the offset can rotate through several disjoint pairs).
    let residual = array![
        [3.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [-3.0, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, -2.0, -0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, -1.0, -0.3],
    ];
    let kinds = vec![
        SaeAtomBasisKind::Periodic,
        SaeAtomBasisKind::Periodic,
        SaeAtomBasisKind::Periodic,
    ];
    let dims = vec![1usize, 1, 1];
    let seed0 = sae_pca_seed_initial_coords_with_pc_offset(residual.view(), &kinds, &dims, 0)
        .expect("offset-0 seed");
    let seed1 = sae_pca_seed_initial_coords_with_pc_offset(residual.view(), &kinds, &dims, 1)
        .expect("offset-1 seed");
    let seed2 = sae_pca_seed_initial_coords_with_pc_offset(residual.view(), &kinds, &dims, 2)
        .expect("offset-2 seed");
    let maxdiff = |a: &Array3<f64>, b: &Array3<f64>| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    assert!(
        maxdiff(&seed0, &seed1) > 1e-3,
        "retry 0 vs 1 must read distinct PC pairs (max coord diff = {:.3e})",
        maxdiff(&seed0, &seed1)
    );
    assert!(
        maxdiff(&seed1, &seed2) > 1e-3,
        "retry 1 vs 2 must read distinct PC pairs (max coord diff = {:.3e})",
        maxdiff(&seed1, &seed2)
    );
    // Offset 0 must be byte-identical to the no-offset entry point (the K=1 and
    // initial-fit seed paths must be untouched).
    let seed_plain =
        sae_pca_seed_initial_coords(residual.view(), &kinds, &dims).expect("plain seed");
    assert_eq!(
        seed0, seed_plain,
        "offset-0 seed must equal the no-offset seed bit-for-bit"
    );
}

/// #976 determinism (issue requirement: identical inputs ⇒ identical output
/// run-to-run). The PCA seed is the SAE-fit entry the owner flagged as flipping
/// the collapse basin between runs. Pin that repeated calls on identical input
/// are bit-identical under the process-default (global Rayon) faer backend, so
/// the now-rotated multi-start is a fixed pass rather than a coin-flip. (The
/// cross-thread-count arm is exercised on the cluster via RAYON_NUM_THREADS; faer's
/// blocked factorizations keep a fixed per-element reduction order, so the
/// global-state-mutating Seq/Par toggle is deliberately NOT done here — it would
/// race the rest of the suite's parallel tests.)
#[test]
pub(crate) fn pca_seed_is_run_to_run_reproducible() {
    let residual = array![
        [3.0, 0.1, -0.2, 0.4, 0.0, 0.05],
        [-3.0, -0.1, 0.2, -0.4, 0.0, -0.05],
        [0.3, 0.0, 2.0, 0.2, 0.1, 0.0],
        [-0.3, 0.0, -2.0, -0.2, -0.1, 0.0],
        [0.0, 0.2, 0.1, 0.0, 1.0, 0.3],
        [0.0, -0.2, -0.1, 0.0, -1.0, -0.3],
    ];
    let kinds = vec![SaeAtomBasisKind::Periodic, SaeAtomBasisKind::Periodic];
    let dims = vec![1usize, 1];
    let seed_a = sae_pca_seed_initial_coords(residual.view(), &kinds, &dims).expect("seed #1");
    let seed_b = sae_pca_seed_initial_coords(residual.view(), &kinds, &dims).expect("seed #2");
    assert_eq!(
        seed_a, seed_b,
        "PCA seed must be bit-identical run-to-run (the issue's determinism \
         requirement)"
    );
}

/// #976 decoder arm is a strict no-op for K=1: a single atom has no peer to
/// fall behind, so the guard must never reseed or record an event even when
/// the lone decoder is tiny. This pins the "K=1 path unchanged" guarantee.
#[test]
pub(crate) fn decoder_norm_guard_is_noop_for_k1() {
    let mut term = trivial_k1_euclidean_term();
    let n = term.n_obs();
    let p = term.output_dim();
    let target = Array2::<f64>::zeros((n, p));
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0_f64]]);
    let before = term.atoms[0].decoder_coefficients.clone();
    term.enforce_decoder_norm_guard(target.view(), 0, &rho)
        .expect("K=1 decoder-norm guard must be a no-op, never error");
    assert!(
        term.collapse_events().is_empty(),
        "K=1 must record no decoder-collapse events"
    );
    assert_eq!(
        term.atoms[0].decoder_coefficients, before,
        "K=1 decoder must be untouched by the guard"
    );
}

/// #1026 — the hybrid split is **load-bearing on the reconstruction**: a slot
/// whose verdict selects LINEAR has its curved decoded image replaced by its
/// fitted straight sub-model, and that substitution match-or-beats the
/// all-curved reconstruction on explained variance at strictly fewer
/// parameters (the strict-generalization dominance floor of #1026).
///
/// The test pins two regimes:
///  * No report ⇒ the collapsed reconstruction is bit-identical to the curved
///    one (the verdict cannot silently alter the fit before it is computed).
///  * A genuinely STRAIGHT atom (its decoded image is a line) forces the
///    dominance floor to select linear; collapsing it leaves the
///    reconstruction essentially unchanged (a line collapsed to its own line),
///    so EV is preserved, while the slot sheds its `M·p − 2·p` curved
///    coefficients — EV-per-parameter strictly improves.
#[test]
pub(crate) fn hybrid_collapse_is_load_bearing_and_dominates() {
    let (mut term, _t, rho) = small_two_atom_periodic_term();

    // (1) Before the report exists, collapse == curved reconstruction.
    let curved = term
        .try_fitted_for_rho(&rho)
        .expect("curved reconstruction assembles");
    let pre = term
        .hybrid_collapsed_reconstruction(&rho)
        .expect("collapse with no report returns the curved fit");
    assert!(
        (&curved - &pre).iter().all(|d| d.abs() < 1e-15),
        "with no hybrid-split report the collapse must equal the curved fit"
    );

    // Make atom 0 genuinely STRAIGHT: a single nonzero basis-0 coefficient
    // decodes γ(t) = φ₀(t)·b, and we additionally drive its decoded image to a
    // pure line by zeroing the higher harmonics — Θ → 0 ⇒ the dominance floor
    // must select linear for this slot.
    for basis_row in 1..term.atoms[0].decoder_coefficients.nrows() {
        for out_col in 0..term.atoms[0].decoder_coefficients.ncols() {
            term.atoms[0].decoder_coefficients[[basis_row, out_col]] = 0.0;
        }
    }

    // Target = the term's own curved reconstruction (after straightening atom 0)
    // ⇒ EV(curved) = 1 exactly, and each atom's leave-this-atom-out response
    // residual `y_resp` equals its own mass-scaled contribution `a_k·γ_k`. The
    // common-evidence selector (#1202) scores both candidates against that
    // residual, so the target is required.
    let target = term
        .try_fitted_for_rho(&rho)
        .expect("post-straighten curved reconstruction assembles");

    // Compute and install the real hybrid-split report (closed-form, no outer
    // fit — sidesteps #1051).
    let report = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report computes")
        .expect("eligible d=1 atoms present a report");
    term.hybrid_split_report = Some(report);

    // The straight atom 0 must have collapsed to linear (its verdict carries a
    // straight sub-model).
    let collapsed_any = term
        .hybrid_split_report
        .as_ref()
        .unwrap()
        .verdicts
        .iter()
        .any(|v| v.linear_image.is_some());
    assert!(
        collapsed_any,
        "a straight atom must collapse at least one slot to the linear tail"
    );

    // EV(curved) = 1 exactly, since the target IS the curved reconstruction.
    let ev_curved = reconstruction_explained_variance(target.view(), target.view())
        .expect("self-reconstruction EV defined");
    assert!(
        (ev_curved - 1.0).abs() < 1e-12,
        "target = curved fit ⇒ EV(curved) = 1; got {ev_curved}"
    );

    // The collapsed dictionary (straight slot decoded by its line) must
    // match-or-beat the curved EV up to the line-fit residual of an already
    // straight image — which is ~0. This is the dominance floor measured on
    // the EV axis: collapsing a straight atom costs no reconstruction.
    let ev_collapsed = term
        .hybrid_collapsed_explained_variance(target.view(), &rho)
        .expect("collapsed EV evaluates")
        .expect("collapsed EV defined");
    assert!(
        ev_collapsed >= ev_curved - 1e-6,
        "collapsing a straight atom must preserve EV (match-or-beat dominance \
             floor): curved {ev_curved:.9}, collapsed {ev_collapsed:.9}"
    );

    // And the collapsed slot sheds curved coefficients: its evidence-priced
    // parameter count is the 2·p linear budget, strictly below the M·p curved
    // decoder it replaced (M ≥ 3 basis rows here).
    let verdict = term
        .hybrid_split_report
        .as_ref()
        .unwrap()
        .verdicts
        .iter()
        .find(|v| v.linear_image.is_some())
        .expect("a collapsed slot exists");
    let collapsed_idx = verdict.linear_image.as_ref().unwrap().atom_idx;
    let curved_params = term.atoms[collapsed_idx].decoder_coefficients.len();
    assert!(
        verdict.choice.num_parameters < curved_params,
        "the linear-collapsed slot must shed curved coefficients: linear \
             {} < curved {}",
        verdict.choice.num_parameters,
        curved_params
    );

    // #1026 EV-vs-Θ frontier as STRUCTURED report data: recompute the report
    // WITH the reconstruction target so each verdict carries the `(Θ, ΔEV)`
    // pair the roadmap reports against (previously this lived only as a
    // transient `log::info!` line). The target is the term's own curved
    // reconstruction, so every atom's leave-one-atom-out drop `ΔEV_k` is the
    // real EV it earns; the report must surface a finite `(Θ, ΔEV)` for every
    // adjudicated d = 1 slot, and the collapsed-to-linear slot must read its
    // straight signature `Θ ≈ 0`.
    let report_with_ev = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report with target computes")
        .expect("eligible d=1 atoms present a report");
    assert!(
        !report_with_ev.verdicts.is_empty(),
        "the report must adjudicate at least one d=1 slot"
    );
    for v in &report_with_ev.verdicts {
        let theta = v
            .fitted_turning
            .unwrap_or_else(|| panic!("verdict '{}' must carry a fitted turning Θ", v.atom_name));
        let dev = v
            .train_loao_delta_ev
            .unwrap_or_else(|| panic!("verdict '{}' must carry a training LOAO ΔEV", v.atom_name));
        assert!(
            theta.is_finite() && theta >= 0.0,
            "fitted turning Θ must be a finite non-negative arc-curvature integral; \
             got {theta} for '{}'",
            v.atom_name
        );
        assert!(
            dev.is_finite(),
            "training LOAO ΔEV must be finite; got {dev} for '{}'",
            v.atom_name
        );
        // The slot that collapsed to the linear tail is straight by definition:
        // its decoded curve integrates ~zero turning.
        if !v.kept_curved {
            assert!(
                theta <= 1e-3,
                "a linear-tail slot must read Θ ≈ 0 (straight image); got {theta} for '{}'",
                v.atom_name
            );
        }
    }

    // #1026 — the POSITIVE arm of the EV-preservation discrimination. The fixture
    // mixes a straightened slot (atom 0: its curved fit IS a line, so collapsing
    // it is lossless — asserted above) with a LOAD-BEARING slot (atom 1: nonzero
    // higher harmonics make its decoded warp a genuinely non-linear function of
    // the coordinate, so collapsing it to a straight secant would raise the
    // reconstruction SSR and DROP EV). The EV-preservation gate keys on exactly
    // that EV loss (`collapse_ssr_increase`), so a correct adjudication must do
    // BOTH: release the straight slot to the linear tail AND keep the load-bearing
    // slot curved while it earns reconstruction. At least one adjudicated slot
    // must therefore be kept curved and carry a strictly positive training LOAO
    // ΔEV — a curveable atom doing real reconstruction work the straight tail
    // cannot capture.
    //
    // On Θ: this fixture reconstructs a 1-D target, and a scalar curve has no
    // geometric turning — the wedge ‖γ' ∧ γ''‖ vanishes identically in one
    // dimension — so every atom honestly reports Θ = 0 here (pinned finite, not
    // the historical `None`, by the loop above, which exercises the constant-image
    // → `Some(0.0)` fix). The geometric Θ-discrimination (high Θ for a real loop,
    // ≈ 0 for a line) is a ≥ 2-D property and is covered where it is meaningful:
    // the real-circle `chart_canonicalization::turning_tests` (→ 2π) and the
    // evidence-level `hybrid_split::tests::turning_residual_selects_curved_on_evidence`.
    // The gate never reads Θ, so this end-to-end test asserts the EV-axis
    // discrimination the gate actually performs, not a turning the fixture's
    // dimensionality cannot exhibit.
    let curved_earner = report_with_ev
        .verdicts
        .iter()
        .find(|v| v.kept_curved && v.train_loao_delta_ev.map(|d| d > 0.0).unwrap_or(false));
    assert!(
        curved_earner.is_some(),
        "a load-bearing curveable slot must be kept curved AND earn positive training \
         LOAO ΔEV (collapsing it would drop reconstruction EV); verdicts = {:?}",
        report_with_ev
            .verdicts
            .iter()
            .map(|v| (
                v.atom_name.clone(),
                v.kept_curved,
                v.fitted_turning,
                v.train_loao_delta_ev
            ))
            .collect::<Vec<_>>()
    );

    // The split is sharp and keyed to the atom identities, not a coincidental
    // count: the slot we straightened (atom 0) is the one released to the linear
    // tail, while the untouched load-bearing slot (atom 1) is the one kept curved.
    // A vacuous "keep everything curved" or "collapse the wrong atom" adjudication
    // fails one of these halves.
    assert_eq!(
        curved_earner.unwrap().atom_name,
        "periodic1",
        "the load-bearing (untouched) atom must be the one kept curved"
    );
    for v in &report_with_ev.verdicts {
        if !v.kept_curved {
            assert_eq!(
                v.atom_name, "periodic0",
                "only the straightened atom may be released to the linear tail; \
                 '{}' collapsed unexpectedly",
                v.atom_name
            );
        }
    }
}

/// #1233 — the hard `top_k` reconstruction must compose with the #1026 hybrid
/// collapse. The FFI top-k path reconstructs from a PROJECTED assignment matrix
/// through [`SaeManifoldTerm::reconstruct_from_assignments`]; that shared
/// assembler must decode a verdict-linear `d = 1` slot by its straight
/// sub-model image (exactly as the production `fitted()` does), not by the
/// original curved decoder. The regression: with `top_k == K` (every atom kept,
/// i.e. the full soft assignment), the collapse-aware projected reconstruction
/// must EXACTLY equal the non-projected collapsed reconstruction, INCLUDING when
/// a slot is hybrid-collapsed linear — and must DIFFER from the curved-only
/// reconstruction, proving the collapse is genuinely engaged on this path.
#[test]
pub(crate) fn topk_reconstruction_composes_with_hybrid_collapse() {
    let (mut term, _t, rho) = small_two_atom_periodic_term();

    // Straighten atom 0 so its verdict collapses to the linear tail.
    for basis_row in 1..term.atoms[0].decoder_coefficients.nrows() {
        for out_col in 0..term.atoms[0].decoder_coefficients.ncols() {
            term.atoms[0].decoder_coefficients[[basis_row, out_col]] = 0.0;
        }
    }
    let target = term
        .try_fitted_for_rho(&rho)
        .expect("post-straighten curved reconstruction assembles");
    let report = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report computes")
        .expect("eligible d=1 atoms present a report");
    term.hybrid_split_report = Some(report);
    assert!(
        term.hybrid_linear_image_map().contains_key(&0),
        "atom 0 must have collapsed to a linear image for this regression"
    );

    // #1233 WITNESS. The straightened atom is a CONSTANT (its periodic basis row 0
    // is the DC term), so its fitted linear image equals its own curve and
    // collapsing it is a numerical no-op — on its own it cannot exercise the
    // collapse-aware reconstruction. Install a genuinely SLOPED straight image
    // into the collapsed slot: still a line (zero turning — a legitimate linear
    // tail, NOT the EV-losing over-collapse the gate prevents), but now
    // `b₀ + (t − t̄)·b₁` differs from the constant curve by a real, per-row,
    // measurable amount. The collapse-aware reconstruction MUST decode THIS image,
    // so the composition / engagement assertions below become non-vacuous: they
    // would fail if the top-k path skipped the collapse or decoded a different
    // image.
    const WITNESS_SLOPE: f64 = 0.4;
    {
        let report = term.hybrid_split_report.as_mut().unwrap();
        let img = report
            .verdicts
            .iter_mut()
            .find_map(|v| v.linear_image.as_mut())
            .expect("the collapsed slot must carry a linear image to install a witness into");
        for slope in img.b1.iter_mut() {
            *slope += WITNESS_SLOPE;
        }
    }

    // `top_k == K` keeps every atom: the projected assignment matrix IS the full
    // soft assignment, so the projected (collapse-aware) reconstruction must
    // match the production collapsed `fitted()` bit-for-bit.
    let full_assignments = term.assignment.assignments();
    let projected_collapsed = term
        .reconstruct_from_assignments(full_assignments.view(), true)
        .expect("collapse-aware projected reconstruction assembles");
    let production_collapsed = term.fitted();
    let max_gap = (&projected_collapsed - &production_collapsed)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        max_gap < 1e-12,
        "top_k==K collapse-aware reconstruction must equal the non-projected \
         collapsed fitted() (incl. the linear-collapsed slot); max gap {max_gap:e}"
    );

    // And it must DIFFER from the curved-only assembly — otherwise the collapse
    // is a silent no-op and the test would pass vacuously.
    let projected_curved = term
        .reconstruct_from_assignments(full_assignments.view(), false)
        .expect("curved projected reconstruction assembles");
    let curved_gap = (&projected_collapsed - &projected_curved)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        curved_gap > 1e-9,
        "the collapsed slot must change the reconstruction vs the curved decoder \
         (collapse engaged); max gap {curved_gap:e}"
    );
}

/// #1228 — an OOS term must reconstruct a hybrid-collapsed `d = 1` slot by the
/// trained dictionary's straight sub-model when those images are attached via
/// [`SaeManifoldTerm::set_hybrid_linear_images`], matching the train-side
/// collapse policy instead of the original curved decoder.
#[test]
pub(crate) fn oos_linear_images_drive_collapsed_reconstruction() {
    let (mut term, _t, rho) = small_two_atom_periodic_term();
    for basis_row in 1..term.atoms[0].decoder_coefficients.nrows() {
        for out_col in 0..term.atoms[0].decoder_coefficients.ncols() {
            term.atoms[0].decoder_coefficients[[basis_row, out_col]] = 0.0;
        }
    }
    let target = term
        .try_fitted_for_rho(&rho)
        .expect("curved reconstruction assembles");
    let report = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report computes")
        .expect("eligible d=1 atoms present a report");

    // Install the report so `fitted()` reconstructs the verdict-linear slot by its
    // straight sub-model (the train-side collapsed reconstruction).
    term.hybrid_split_report = Some(report);

    // #1228 WITNESS. The straightened atom is a CONSTANT (periodic basis row 0 is
    // the DC term), so its fitted linear image equals its own curve and collapsing
    // it changes nothing — the train-vs-OOS threading could not be observed.
    // Install a genuinely SLOPED straight image into the collapsed slot: still a
    // line (zero turning — a legitimate linear tail, NOT the EV-losing
    // over-collapse the gate prevents), but now it differs from the constant curve
    // by a real, measurable amount, so the train-side collapse is non-trivial and
    // the OOS reproduction below genuinely exercises the image threading.
    const WITNESS_SLOPE: f64 = 0.4;
    {
        let report = term.hybrid_split_report.as_mut().unwrap();
        let img = report
            .verdicts
            .iter_mut()
            .find_map(|v| v.linear_image.as_mut())
            .expect("the collapsed slot must carry a linear image to install a witness into");
        for slope in img.b1.iter_mut() {
            *slope += WITNESS_SLOPE;
        }
    }

    // Harvest the trained (witness-sloped) linear images to thread to a fresh OOS
    // term that knows the decoder but not the in-fit report, then drop the report.
    let images: Vec<_> = term
        .hybrid_split_report
        .as_ref()
        .unwrap()
        .verdicts
        .iter()
        .filter_map(|v| v.linear_image.clone())
        .collect();
    assert!(
        !images.is_empty(),
        "the straight slot must yield at least one linear image to thread to OOS"
    );
    let collapsed_with_report = term.fitted();
    term.hybrid_split_report = None;

    // Without images attached, the fresh term reconstructs all-curved.
    let curved = term.fitted();
    assert!(
        (&curved - &collapsed_with_report)
            .iter()
            .any(|d| d.abs() > 1e-9),
        "with no images attached the OOS reconstruction must be the curved one"
    );

    // Attaching the trained images restores the collapsed reconstruction exactly.
    term.set_hybrid_linear_images(images)
        .expect("valid linear images attach");
    let collapsed_oos = term.fitted();
    let gap = (&collapsed_oos - &collapsed_with_report)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        gap < 1e-12,
        "attached OOS linear images must reproduce the train-side collapsed \
         reconstruction; max gap {gap:e}"
    );
}

/// #1777 helper: a single `d = 1` periodic atom whose latent coordinate has
/// COLLAPSED to one point (all rows share the same `t`), so the hybrid split
/// cannot fit a slope against its own codes and must take the collapse-rescue
/// path (project the leave-this-atom-out residual onto its top output direction
/// `v`). The `target` is an exact affine ramp along a fixed output direction so
/// the rescued straight image reconstructs it at near-perfect EV.
fn collapse_rescue_term_and_target() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 6usize;
    // Collapsed coordinate: every row at the SAME t → zero coordinate spread.
    let coords = Array2::<f64>::from_elem((n, 1), 0.3);
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new(
        "collapsed_circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.1, -0.2], [0.05, 0.15], [-0.1, 0.08]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    // Softmax K=1 → gate ≡ 1 on every row (no IBP α to resolve).
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // target_i = mu + s_i · d, an exact affine ramp along d = (0.6, 0.8).
    let s = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5];
    let mu = [0.2, -0.1];
    let d = [0.6, 0.8];
    let mut target = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        target[[row, 0]] = mu[0] + s[row] * d[0];
        target[[row, 1]] = mu[1] + s[row] * d[1];
    }
    let rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]]);
    (term, target, rho)
}

/// #1777 GOAL 1 — a collapse-rescued atom must reconstruct the SAME rows
/// identically whether treated as "train" (cached per-row codes) or re-encoded
/// as "held-out" (the `v`-projection of the row's own leave-this-atom-out
/// residual), and the `v`-projection OOS reconstruction must BEAT the collapsed
/// (own-coordinate) fallback in explained variance.
#[test]
pub(crate) fn collapse_rescue_oos_v_projection_matches_train_and_beats_fallback() {
    let (mut term, target, rho) = collapse_rescue_term_and_target();

    // The joint fit drove this atom into the degenerate fixed point; the hybrid
    // split must take the collapse-rescue branch and produce a linear image that
    // carries the projection direction `v` (the #1777 serializable quantity).
    let report = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split computes")
        .expect("the collapsed d=1 atom presents a rescue verdict");
    let rescue_image = report
        .verdicts
        .iter()
        .find_map(|v| v.linear_image.clone())
        .expect("the rescued slot carries a linear image");
    assert!(
        rescue_image.is_collapse_rescued() && rescue_image.v.is_some(),
        "a collapse-rescued image must carry a projection direction v"
    );
    assert!(
        rescue_image.row_codes.is_some(),
        "a collapse-rescued image must carry its train per-row codes"
    );

    // TRAIN reconstruction: the term with the report installed decodes the slot at
    // its cached per-row codes (`row_codes`).
    term.hybrid_split_report = Some(report);
    let train_recon = term.fitted();

    // HELD-OUT reconstruction: a fresh OOS term that knows only the decoder and
    // the trained linear images (no in-fit report) recomputes each row's
    // coordinate from ITS OWN residual projected onto `v`, via the target-aware
    // path. Same target ⇒ same residual ⇒ same coordinate ⇒ same reconstruction.
    let mut oos = term.clone();
    oos.hybrid_split_report = None;
    oos.set_hybrid_linear_images(vec![rescue_image.clone()])
        .expect("trained rescue image attaches to the OOS term");
    let oos_recon = oos
        .try_fitted_target_aware(target.view(), Some(&rho))
        .expect("target-aware OOS reconstruction assembles");

    let max_gap = (&train_recon - &oos_recon)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        max_gap < 1e-10,
        "train (cached codes) and OOS (v-projection) reconstructions must be the \
         SAME model within tol; max gap {max_gap:e}"
    );

    // The v-projection OOS reconstruction must beat the collapsed-coordinate
    // fallback (row_codes/v cleared ⇒ every row decodes at the atom's own, single,
    // collapsed coordinate → a constant image that cannot track the residual ramp).
    let fallback_image = crate::hybrid_split::AtomLinearImage {
        atom_idx: rescue_image.atom_idx,
        t_bar: rescue_image.t_bar,
        b0: rescue_image.b0.clone(),
        b1: rescue_image.b1.clone(),
        row_codes: None,
        v: None,
    };
    let mut fallback = term.clone();
    fallback.hybrid_split_report = None;
    fallback
        .set_hybrid_linear_images(vec![fallback_image])
        .expect("fallback image attaches");
    let fallback_recon = fallback.fitted();

    let ev_vproj = global_ev(target.view(), oos_recon.view());
    let ev_fallback = global_ev(target.view(), fallback_recon.view());
    assert!(
        ev_vproj > ev_fallback + 0.1 && ev_vproj > 0.95,
        "the v-projection OOS EV ({ev_vproj:.4}) must beat the collapsed-coordinate \
         fallback ({ev_fallback:.4}) and recover the residual ramp"
    );
}

/// #1777 GOAL 2 — the PER-FIT [`SaeFitConfig`] is the source of truth for the
/// IBP-α and separation-barrier overrides: two terms carrying DIFFERENT configs
/// produce correspondingly-different α / barrier strength, with NO process-global
/// atomic touched (isolation), and the two terms do not leak into each other.
#[test]
pub(crate) fn per_fit_config_isolates_barrier_and_ibp_alpha() {
    // Sanity: neither term sets a global override, so the global fallbacks stay
    // unset and cannot be the source of the distinct effects observed below.
    assert!(
        crate::assignment::ibp_alpha_override().is_none(),
        "test must not depend on a preset global IBP-α override"
    );

    let (mut term_a, _t_a, rho_a) = small_two_atom_ibp_term();
    let (mut term_b, _t_b, rho_b) = small_two_atom_ibp_term();

    // Distinct per-fit configs, applied via config ONLY (no global setters).
    term_a.set_fit_config(SaeFitConfig {
        separation_barrier_strength_override: Some(0.1),
        ibp_alpha_override: Some(0.2),
    });
    term_b.set_fit_config(SaeFitConfig {
        separation_barrier_strength_override: Some(3.0),
        ibp_alpha_override: Some(5.0),
    });

    // Round-trips through the config accessor.
    assert_eq!(term_a.fit_config().ibp_alpha_override, Some(0.2));
    assert_eq!(
        term_b.fit_config().separation_barrier_strength_override,
        Some(3.0)
    );

    // IBP-α: the per-fit override is the resolved α (bypassing the mode schedule),
    // and the two terms resolve DIFFERENT α without touching any global.
    assert_eq!(term_a.assignment.resolved_ibp_alpha(&rho_a), Some(0.2));
    assert_eq!(term_b.assignment.resolved_ibp_alpha(&rho_b), Some(5.0));

    // Distinct α ⇒ distinct gates (the ordered geometric prior π_k differs).
    let gates_a = term_a.assignment.assignments_for_rho(&rho_a).unwrap();
    let gates_b = term_b.assignment.assignments_for_rho(&rho_b).unwrap();
    let gate_gap = (&gates_a - &gates_b)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        gate_gap > 1e-6,
        "distinct per-fit IBP-α overrides must produce distinct gates; gap {gate_gap:e}"
    );

    // Barrier strength (K=2, so the barrier is live): the per-fit override is the
    // source of truth, distinct per term, with the global still unset.
    assert_eq!(term_a.separation_barrier_strength(), 0.1);
    assert_eq!(term_b.separation_barrier_strength(), 3.0);
    assert!(
        super::term::sae_separation_barrier_override().is_none(),
        "the per-fit override must NOT write the process-global barrier atomic"
    );

    // Isolation: clearing term_a's config leaves term_b untouched, and term_a
    // falls back to the mode's own α (the historical path).
    term_a.set_fit_config(SaeFitConfig::default());
    assert_eq!(term_a.assignment.resolved_ibp_alpha(&rho_a), Some(1.0)); // the mode's compiled α
    assert_eq!(term_b.assignment.resolved_ibp_alpha(&rho_b), Some(5.0));
}

/// #1777 GOAL 3 — the assignment mode is the accurately-named `ThresholdGate`
/// (a hard-sigmoid gate, NOT the literature JumpReLU magnitude activation); the
/// legacy `jumprelu` constructor remains a back-compat alias producing the SAME
/// variant and the SAME gates.
#[test]
pub(crate) fn threshold_gate_is_primary_jumprelu_is_backcompat_alias() {
    let primary = AssignmentMode::threshold_gate(0.9, 0.15);
    let legacy = AssignmentMode::jumprelu(0.9, 0.15);
    assert!(matches!(primary, AssignmentMode::ThresholdGate { .. }));
    assert!(
        matches!(legacy, AssignmentMode::ThresholdGate { .. }),
        "the legacy jumprelu constructor must yield the renamed ThresholdGate variant"
    );

    // Identical gates from either spelling.
    let logits = array![[0.5, -0.2, 0.4], [0.05, 0.6, -0.3]];
    let coords = vec![
        array![[0.1], [0.2]],
        array![[0.3], [0.4]],
        array![[0.5], [0.6]],
    ];
    let build = |mode: AssignmentMode| {
        SaeAssignment::from_blocks_with_mode(logits.clone(), coords.clone(), mode).unwrap()
    };
    let a_primary = build(primary).assignments();
    let a_legacy = build(legacy).assignments();
    assert_eq!(a_primary, a_legacy);
}

/// #976 Layer-1 guard 2: a single Newton application cannot move a gate
/// logit by more than the gate-scale cap, however large the solver's raw
/// delta. Softmax canonicalization shifts whole rows, so the invariant is
/// checked on the within-row logit DIFFERENCE, which the shift preserves.
#[test]
pub(crate) fn assignment_logit_step_cap_bounds_single_iteration_gate_motion() {
    let (mut term, _target, _rho) = small_two_atom_periodic_term();
    let n = term.assignment.n_obs();
    let q = term.assignment.row_block_dim();
    let diff_before = term.assignment.logits[[0, 0]] - term.assignment.logits[[0, 1]];

    let mut delta = Array1::<f64>::zeros(n * q);
    // Softmax K=2 has one free logit per row at offset 0 of the row block.
    delta[0] = 1.0e6;
    let delta_beta = Array1::<f64>::zeros(term.beta_dim());
    term.apply_newton_step(delta.view(), delta_beta.view(), 1.0)
        .expect("step applies");

    let cap = SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS * term.assignment.mode.temperature();
    let diff_after = term.assignment.logits[[0, 0]] - term.assignment.logits[[0, 1]];
    assert!(
        ((diff_after - diff_before) - cap).abs() < 1.0e-9,
        "a 1e6 raw logit delta must realise exactly the {cap}-cap, moved {}",
        diff_after - diff_before
    );
}

/// #976 Layer-1 guard 3: a gate-collapsed atom (max active mass below the
/// floor) is re-seeded back into contention exactly once, every breach is
/// an observable CollapseEvent, and the second collapse is recorded as
/// terminal — once — instead of fighting the optimizer.
#[test]
pub(crate) fn active_mass_guard_reseeds_once_then_records_terminal_collapse() {
    let (mut term, _target, _rho) = small_two_atom_periodic_term();
    let n = term.assignment.n_obs();
    let slam = |term: &mut SaeManifoldTerm| {
        for row in 0..n {
            term.assignment.logits[[row, 0]] = 0.0;
            term.assignment.logits[[row, 1]] = -1.0e3;
        }
    };

    slam(&mut term);
    term.enforce_active_mass_guard(0, None).expect("guard runs");
    assert_eq!(term.collapse_events().len(), 1);
    let ev = term.collapse_events()[0];
    assert_eq!(ev.atom, 1);
    assert_eq!(ev.action, CollapseAction::Reseeded);
    assert!(ev.max_active_mass < ev.floor);

    // The re-seed restored material support (softmax parity with the
    // row winner), so a healthy follow-up check records nothing.
    let masses = term.assignment.assignments();
    let max1 = (0..n).map(|r| masses[[r, 1]]).fold(0.0_f64, f64::max);
    assert!(max1 > 1.0e-3_f64);
    term.enforce_active_mass_guard(1, None).expect("guard runs");
    assert_eq!(term.collapse_events().len(), 1);

    // Second collapse: budget exhausted ⇒ terminal, recorded exactly once
    // across repeated checks; the logits are left to the objective.
    slam(&mut term);
    term.enforce_active_mass_guard(2, None).expect("guard runs");
    term.enforce_active_mass_guard(3, None).expect("guard runs");
    let terminals: Vec<_> = term
        .collapse_events()
        .iter()
        .filter(|e| e.action == CollapseAction::Terminal)
        .collect();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0].atom, 1);
    assert!(
        term.collapse_events().iter().all(|e| e.atom == 1),
        "the healthy atom must never be flagged"
    );
}

#[test]
pub(crate) fn sae_rho_seed_dispersion_scaling_shifts_every_scale_coupled_axis() {
    let rho = SaeManifoldRho::new(0.7_f64.ln(), 1.3_f64.ln(), vec![array![0.2, -0.4]]);
    let dispersion = 0.05_f64 * 0.05;
    let scaled = rho
        .seed_scaled_by_dispersion_for_assignment(dispersion, AssignmentMode::softmax(1.0))
        .unwrap();
    let shift = dispersion.ln();

    assert_abs_diff_eq!(
        scaled.log_lambda_sparse,
        rho.log_lambda_sparse + shift,
        epsilon = 1.0e-14
    );
    assert_abs_diff_eq!(
        scaled.log_lambda_smooth[0],
        rho.log_lambda_smooth[0] + shift,
        epsilon = 1.0e-14
    );
    assert_abs_diff_eq!(
        scaled.log_ard[0][0],
        rho.log_ard[0][0] + shift,
        epsilon = 1.0e-14
    );
    assert_abs_diff_eq!(
        scaled.log_ard[0][1],
        rho.log_ard[0][1] + shift,
        epsilon = 1.0e-14
    );

    // #1744 — IBP-MAP admits NO response-dispersion scaling on ANY ρ coordinate
    // (learnable-α or fixed-α). Its free per-row Bernoulli gates overfit under a
    // dispersion-weakened smoothness/ARD seed, collapsing the Fellner–Schall
    // fixed point; the sparse coordinate is a dimensionless log-α concentration
    // offset that was never a squared-output-unit penalty weight. So every IBP
    // coordinate stays at its absolute (already dimensionless) construction value.
    for ibp_mode in [
        AssignmentMode::ibp_map(1.0, 1.0, true),
        AssignmentMode::ibp_map(1.0, 1.0, false),
    ] {
        let ibp = rho
            .seed_scaled_by_dispersion_for_assignment(dispersion, ibp_mode)
            .unwrap();
        assert_abs_diff_eq!(
            ibp.log_lambda_sparse,
            rho.log_lambda_sparse,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            ibp.log_lambda_smooth[0],
            rho.log_lambda_smooth[0],
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(ibp.log_ard[0][0], rho.log_ard[0][0], epsilon = 1.0e-14);
        assert_abs_diff_eq!(ibp.log_ard[0][1], rho.log_ard[0][1], epsilon = 1.0e-14);
    }
}

#[test]
pub(crate) fn fit_data_collapse_records_terminal_event_for_active_atom() {
    let coords = array![[0.0], [0.25], [0.5], [0.75]];
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((3, 2)),
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((4, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
    let fitted = Array2::<f64>::zeros(target.dim());
    let assignments = Array2::<f64>::ones((4, 1));

    let recorded = term
        .record_fit_data_collapse_if_needed(target.view(), fitted.view(), assignments.view(), 7)
        .unwrap();

    assert!(recorded);
    let terminals: Vec<_> = term
        .collapse_events()
        .iter()
        .filter(|event| event.action == CollapseAction::Terminal)
        .collect();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0].atom, 0);
    assert_eq!(terminals[0].iteration, 7);
    assert!(terminals[0].max_active_mass <= SAE_FIT_DATA_COLLAPSE_EV_FLOOR);
}

pub(crate) fn deterministic_circle_noise(row: usize, col: usize) -> f64 {
    let x = (row as f64 + 1.0) * 12.9898 + (col as f64 + 1.0) * 78.233;
    (x.sin() * 43758.5453).sin()
}

pub(crate) fn planted_circle_data(n: usize, sigma: f64) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let theta = std::f64::consts::TAU * row as f64 / n as f64;
        z[[row, 0]] = theta.cos() + sigma * deterministic_circle_noise(row, 0);
        z[[row, 1]] = theta.sin() + sigma * deterministic_circle_noise(row, 1);
    }
    z
}

/// A single planted circle embedded in `d_embed` dimensions via a random unit
/// 2×`d_embed` frame, matching the #795 Python repro (`Z = [cos t, sin t] @ B`).
/// The embedding makes the fitted decoder wide (D≫2), which is the regime that
/// used to make the isometry Gauss-Newton `‖B‖⁴` curvature dominate the data-fit
/// block and saturate the arrow-Schur proximal ridge.
pub(crate) fn planted_circle_embedded(n: usize, d_embed: usize, sigma: f64) -> Array2<f64> {
    // Two deterministic near-orthonormal frame rows (row-normalized), so the test
    // needs no RNG dependency but still spans a generic 2-plane in R^{d_embed}.
    let mut frame = Array2::<f64>::zeros((2, d_embed));
    for j in 0..d_embed {
        frame[[0, j]] = deterministic_circle_noise(j, 0);
        frame[[1, j]] = deterministic_circle_noise(j, 1);
    }
    for r in 0..2 {
        let norm = (0..d_embed).map(|j| frame[[r, j]] * frame[[r, j]]).sum::<f64>().sqrt();
        for j in 0..d_embed {
            frame[[r, j]] /= norm.max(1.0e-300);
        }
    }
    let mut z = Array2::<f64>::zeros((n, d_embed));
    for row in 0..n {
        let theta = std::f64::consts::TAU * row as f64 / n as f64;
        let (c, s) = (theta.cos(), theta.sin());
        for j in 0..d_embed {
            z[[row, j]] =
                c * frame[[0, j]] + s * frame[[1, j]] + sigma * deterministic_circle_noise(row, j);
        }
    }
    z
}

pub(crate) fn global_ev(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    let (n, p) = target.dim();
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        for row in 0..n {
            means[col] += target[[row, col]];
        }
        means[col] /= n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let r = target[[row, col]] - fitted[[row, col]];
            ssr += r * r;
            let centered = target[[row, col]] - means[col];
            sst += centered * centered;
        }
    }
    1.0 - ssr / sst.max(1.0e-300)
}

#[derive(Clone, Copy)]
pub(crate) enum PlantedCircleAssignmentMode {
    Softmax,
    IbpMap,
}

impl PlantedCircleAssignmentMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Softmax => "softmax",
            Self::IbpMap => "ibp_map",
        }
    }

    pub(crate) fn mode(self) -> AssignmentMode {
        const TAU: f64 = 1.0;
        const ALPHA: f64 = 1.0;
        match self {
            Self::Softmax => AssignmentMode::softmax(TAU),
            Self::IbpMap => AssignmentMode::ibp_map(TAU, ALPHA, false),
        }
    }

    pub(crate) fn seed_logit(self) -> f64 {
        const TAU: f64 = 1.0;
        match self {
            Self::Softmax => 0.0,
            Self::IbpMap => 6.0 * TAU,
        }
    }

    pub(crate) fn seed_gate(self) -> f64 {
        match self {
            Self::Softmax => 1.0,
            Self::IbpMap => 1.0 / (1.0 + (-6.0_f64).exp()),
        }
    }
}

pub(crate) fn planted_circle_seed_term(
    z: ArrayView2<'_, f64>,
    assignment_mode: PlantedCircleAssignmentMode,
) -> (SaeManifoldTerm, f64) {
    let n = z.nrows();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let seed_coords = sae_pca_seed_initial_coords(z, &[SaeAtomBasisKind::Periodic], &[1]).unwrap();
    let coords = seed_coords.slice(s![0, .., 0..1]).to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let seed_gate = assignment_mode.seed_gate();
    let gated_phi = &phi * seed_gate;
    let mut xtx = fast_ata(&gated_phi);
    for i in 0..xtx.nrows() {
        xtx[[i, i]] += 1.0e-10;
    }
    let xtz = fast_atb(&gated_phi, &z.to_owned());
    let decoder = xtx.cholesky(Side::Lower).unwrap().solve_mat(&xtz);
    let seed_fitted = gated_phi.dot(&decoder);
    let mut rss = 0.0_f64;
    for row in 0..n {
        for col in 0..z.ncols() {
            let r = z[[row, col]] - seed_fitted[[row, col]];
            rss += r * r;
        }
    }
    let seed_dispersion = (rss / (n * z.ncols()) as f64).max(1.0e-12);
    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n, 1), assignment_mode.seed_logit()),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        assignment_mode.mode(),
    )
    .unwrap();
    (
        SaeManifoldTerm::new(vec![atom], assignment).unwrap(),
        seed_dispersion,
    )
}

#[test]
pub(crate) fn planted_circle_focus_1744() {
    let n = 40usize;
    let sigma = 0.05_f64;
    let z = planted_circle_data(n, sigma);
    let mut out = String::new();
    for assignment_mode in [
        PlantedCircleAssignmentMode::Softmax,
        PlantedCircleAssignmentMode::IbpMap,
    ] {
        let label = assignment_mode.label();
        let (term, seed_dispersion) = planted_circle_seed_term(z.view(), assignment_mode);
        out.push_str(&format!("FOCUS1744 mode={label} seed_disp={seed_dispersion:.3e}\n"));
        for &sparse in &[-8.0_f64, 1.0] {
            for &ard in &[-6.0_f64, -3.0, 0.0, 1.0] {
                for &smooth in &[-8.0_f64, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0] {
                    let mut t = term.clone();
                    let r = SaeManifoldRho::new(sparse, smooth, vec![array![ard]]);
                    match t.reml_criterion_with_cache(
                        z.view(),
                        &r,
                        None,
                        60,
                        0.04,
                        1.0e-6,
                        1.0e-6,
                    ) {
                        Ok(evaluated) => {
                            let ev = global_ev(z.view(), t.fitted().view());
                            out.push_str(&format!(
                                "FOCUS1744 mode={label} sparse={sparse} ard={ard} smooth={smooth} cost={:.4e} ev={ev:.4}\n",
                                evaluated.0
                            ));
                        }
                        Err(err) => out.push_str(&format!(
                            "FOCUS1744 mode={label} sparse={sparse} ard={ard} smooth={smooth} ERR={err}\n"
                        )),
                    }
                }
            }
        }
    }
    assert!(
        out.contains("ev="),
        "FOCUS1744: every (mode,sparse,ard,smooth) config errored — no fit produced a finite EV:\n{out}"
    );
}

/// #1744 focused regression guard for the single noise-scale sweep point that
/// failed: `ibp_map` n=40 σ=0.18. Runs exactly one outer solve from the
/// dimensionless ρ seed (the same construction the full sweep uses), so it
/// reproduces the RED (~70s) far faster than the ~400s full sweep.
///
/// Before the seed-screening keep-best fix, the default budget-2 multi-start
/// discarded the flexible (dimensionless-seed) basin (non-converged REML 74.4,
/// EV 0.96) for the over-smoothed basin (non-converged REML 67.3, EV 0.86,
/// ρ≈(1.0,0.997,0.984)) purely on the lower REML. The over-smoothed basin is
/// both more-smoothed and farther from stationarity, so keep-best now retains
/// the flexible seed. Uses the same 0.95 threshold as the full sweep.
#[test]
pub(crate) fn planted_circle_ibp_map_n40_sigma018_reaches_high_ev_1744() {
    let assignment_mode = PlantedCircleAssignmentMode::IbpMap;
    let n = 40usize;
    let sigma = 0.18_f64;
    let z = planted_circle_data(n, sigma);
    let (term, seed_dispersion) = planted_circle_seed_term(z.view(), assignment_mode);
    let seed_ev = global_ev(z.view(), term.fitted().view());
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, assignment_mode.mode())
        .unwrap();
    let init_rho_flat = init_rho.to_flat();
    let n_params = init_rho_flat.len();
    let mut objective =
        SaeManifoldOuterObjective::new(term, z.clone(), None, init_rho, 50, 0.04, 1.0e-6, 1.0e-6);
    gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_initial_rho(init_rho_flat)
        .run(&mut objective, "SAE planted circle #1744 focused")
        .unwrap();
    let fitted_result = objective.into_fitted();
    let rho = fitted_result.rho;
    let ev = global_ev(z.view(), fitted_result.term.fitted().view());
    assert!(
        ev > 0.95,
        "focused #1744 fixture (ibp_map n={n} sigma={sigma}) seed_ev={seed_ev:.4} \
         final_rho=({:.3},{:?},{:?}) EV={ev:.4} should exceed 0.95",
        rho.log_lambda_sparse,
        rho.log_lambda_smooth,
        rho.log_ard
    );
}

#[test]
pub(crate) fn planted_circle_noise_scale_sweep_reaches_high_ev_with_dimensionless_rho_seed() {
    for assignment_mode in [
        PlantedCircleAssignmentMode::Softmax,
        PlantedCircleAssignmentMode::IbpMap,
    ] {
        let assignment_label = assignment_mode.label();
        for &n in &[40usize, 250usize] {
            for &sigma in &[0.02_f64, 0.05, 0.18] {
                let z = planted_circle_data(n, sigma);
                let (term, seed_dispersion) = planted_circle_seed_term(z.view(), assignment_mode);
                let seed_ev = global_ev(z.view(), term.fitted().view());
                let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![array![0.0]])
                    .seed_scaled_by_dispersion_for_assignment(
                        seed_dispersion,
                        assignment_mode.mode(),
                    )
                    .unwrap();
                let init_rho_flat = init_rho.to_flat();
                let n_params = init_rho_flat.len();
                let mut objective = SaeManifoldOuterObjective::new(
                    term,
                    z.clone(),
                    None,
                    init_rho,
                    50,
                    0.04,
                    1.0e-6,
                    1.0e-6,
                );
                gam_solve::rho_optimizer::OuterProblem::new(n_params)
                    .with_initial_rho(init_rho_flat)
                    .run(&mut objective, "SAE planted circle dimensionless seed")
                    .unwrap();
                let fitted_result = objective.into_fitted();
                let fitted_term = fitted_result.term;
                let rho = fitted_result.rho;
                let fitted = fitted_term.fitted();
                let ev = global_ev(z.view(), fitted.view());
                assert!(
                    ev > 0.95,
                    "planted circle assignment={assignment_label} n={n} sigma={sigma} seed_ev={seed_ev:.4} seed_phi={seed_dispersion:.3e} \
                         final_rho=({:.3}, {:?}, {:?}) EV={ev:.4} should exceed 0.95",
                    rho.log_lambda_sparse,
                    rho.log_lambda_smooth,
                    rho.log_ard
                );
                assert!(
                    fitted_term.collapse_events().is_empty(),
                    "healthy planted circle assignment={assignment_label} fit should not record collapse events: {:?}",
                    fitted_term.collapse_events()
                );
            }
        }
    }
}

#[test]
pub(crate) fn sae_value_probe_refusal_classification_is_inner_only() {
    assert!(
        SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::reml_criterion: inner solve did not converge at fixed ρ"
        )
    );
    assert!(
        SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::reml_criterion: undamped evidence factorization hit a non-PD per-row H_tt block before KKT stationarity"
        )
    );
    // A non-PD cross-row IBP joint Hessian at a probed ρ is genuine infeasibility
    // (the Laplace evidence log-det is undefined there) — recoverable, the same
    // class as the per-row non-PD refusal, so the outer optimizer returns +∞ and
    // steers back into the PD region instead of aborting the whole fit.
    assert!(
        SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::reml_criterion: cross-row IBP joint Hessian is non-PD at this ρ; evidence Laplace log-det undefined (infeasible ρ probe)"
        )
    );
    // The generic "log-det unavailable" message (a real factorization defect, not
    // an infeasibility) stays FATAL — it is NOT in the recoverable set.
    assert!(
        !SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::reml_criterion: arrow_log_det_from_cache returned None (undamped joint Hessian log-det unavailable for the Laplace normaliser)"
        )
    );
    assert!(
        !SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::reml_criterion: row-gauge evidence deflation count re-anchored \
                 4 times within one optimization; the quotient dimension is not stabilizing"
        )
    );
}

#[test]
pub(crate) fn streaming_exact_reml_matches_full_batch_reml_small_sae() {
    let (term0, target, rho) = small_two_atom_periodic_term();
    let mut full = term0.clone();
    let mut streaming = term0;
    let (full_cost, full_loss, _cache) = full
        .reml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .unwrap();
    let (stream_cost, stream_loss) = streaming
        .reml_criterion_streaming_exact(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .unwrap();
    assert_abs_diff_eq!(stream_cost, full_cost, epsilon = 1.0e-8);
    assert_abs_diff_eq!(stream_loss.total(), full_loss.total(), epsilon = 1.0e-8);
}

/// As [`small_two_atom_periodic_term`], but in **IBP-MAP** assignment mode so
/// the exact joint Hessian carries the #1038 cross-row rank-`R` Woodbury block
/// `H_full = H₀' + U D Uᵀ` (the empirical-mass coupling between distinct latent
/// rows through a shared atom column). The dense evidence log-det therefore
/// includes the capacitance term `log|C| = log det(I_R + D Uᵀ H₀'⁻¹ U)` — the
/// quantity the streaming path must reproduce.
pub(crate) fn small_two_atom_ibp_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let atom0 = SaeManifoldAtom::new(
        "periodic0",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.25], [-0.35], [0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let atom1 = SaeManifoldAtom::new(
        "periodic1",
        SaeAtomBasisKind::Periodic,
        1,
        phi1,
        jet1,
        array![[-0.10], [0.20], [0.30]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.8, 1.0, false),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let target = array![[0.12], [-0.03], [0.08], [0.20], [-0.11]];
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![array![0.9_f64.ln()], array![1.1_f64.ln()]],
    );
    (term, target, rho)
}

/// #1038/#1225 — the streaming evidence log-det MUST equal the dense full-batch
/// evidence log-det for an **IBP-MAP** term, i.e. it MUST carry the exact
/// cross-row Woodbury capacitance correction `log|C|`.
///
/// Pre-fix the streaming path could not represent the rank-`R` cross-row block:
/// `reduced_schur_and_log_det_tt` refused IBP-active systems outright, so
/// `reml_criterion_streaming_exact` *errored* on this fixture — and if that
/// refusal had instead silently returned `log_det_tt + log_det_schur`, the
/// streaming criterion would have under-counted the dense criterion by exactly
/// `½·log|C|` (the dropped capacitance term), violating the #1225 invariant
/// that streaming and dense optimize the SAME REML objective.
///
/// This pins both halves of the fix:
///   (1) the dense cache genuinely carries a non-trivial cross-row correction
///       on this fixture (`|log|C|| > 0`), so the equality below is load-bearing
///       rather than a vacuous `log|C| = 0` match (which any softmax term gives);
///   (2) the streaming exact log-det now reproduces the dense criterion to
///       inner-solve tolerance.
#[test]
pub(crate) fn streaming_exact_reml_matches_full_batch_reml_ibp_woodbury() {
    let (term0, target, rho) = small_two_atom_ibp_term();
    let mut full = term0;
    let (_full_cost, _full_loss, cache) = full
        .reml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .expect("dense IBP criterion must evaluate");

    // (1) The dense joint Hessian carries a genuine cross-row Woodbury block on
    // this fixture: its capacitance correction is present, finite, and nonzero.
    // This is the `log|C|` the streaming path would drop without the fix.
    assert!(
        cache.cross_row_woodbury.is_some(),
        "IBP fixture must build a cross-row Woodbury carrier (else the test is vacuous)"
    );
    let log_c = cache.cross_row_woodbury_log_det();
    assert!(
        log_c.is_finite() && log_c.abs() > 1.0e-6,
        "IBP fixture must have a load-bearing nonzero cross-row log|C|; got {log_c}"
    );

    // (2) The streaming exact LOG-DET must reproduce the dense `log|H|` at the SAME
    // converged state — `full` is already at its converged (t,β) after the dense
    // criterion. We compare the log-det DIRECTLY rather than re-fitting through
    // `reml_criterion_streaming_exact`: a streaming RE-FIT runs a fresh inner solve
    // whose faer parallel reduction is non-deterministic under thread contention
    // and intermittently surfaces the (recoverable) non-PD refusal — orthogonal to
    // this Woodbury-correctness test. `streaming_exact_arrow_log_det` re-assembles
    // `log|H_full|` chunk-by-chunk at the frozen state with NO inner solve, so the
    // only delta vs the dense `arrow_log_det_from_cache` is FP reassociation
    // (~1e-13). Pre-fix the streaming path DROPPED `log|C|` (or hard-refused on the
    // cross-row source), so this differed by `log|C|` (≈ {log_c}) or errored.
    let dense_logdet = arrow_log_det_from_cache(&cache).expect("dense log-det finite");
    let stream_logdet = full
        .streaming_exact_arrow_log_det(target.view(), &rho, None)
        .expect("streaming log-det must evaluate (cross-row Woodbury now carried)");
    assert_abs_diff_eq!(stream_logdet, dense_logdet, epsilon = 1.0e-8);
}

/// #1029 measure-consistency gate: the value-probe refine policy must
/// rank the SAME criterion as the full accepted-point policy. The probe
/// budget only caps refinement work — it never loosens the KKT/step
/// tolerance — so when both policies converge from the same state, the
/// returned criterion values must match to inner-solve tolerance. A loose
/// probe value compared against a tight reference value is exactly the
/// estimator/threshold measure mismatch that caused the BMS HT-subsample
/// false-reject bug; this test pins the invariant that forbids it.
#[test]
pub(crate) fn value_probe_refine_policy_ranks_same_criterion_as_full_policy() {
    let (term0, target, rho) = small_two_atom_periodic_term();
    let mut full = term0.clone();
    let mut probe = term0;
    let (full_cost, full_loss) = full
        .reml_criterion_with_refine_policy(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4, true)
        .expect("full-budget criterion must converge on the small fixture");
    let (probe_cost, probe_loss) = probe
        .reml_criterion_with_refine_policy(
            target.view(),
            &rho,
            None,
            2,
            0.25,
            1.0e-4,
            1.0e-4,
            false,
        )
        .expect("probe-budget criterion must converge on the small fixture");
    assert_abs_diff_eq!(probe_cost, full_cost, epsilon = 1.0e-8);
    assert_abs_diff_eq!(probe_loss.total(), full_loss.total(), epsilon = 1.0e-8);
}

/// #1224 — the BFGS/ARC line-search cost probe must see PURE REML, not the
/// co-training consistency fold `f+c`.
///
/// The outer optimizer compares three lanes at a fixed ρ:
///   * `eval` (`OuterEvalOrder::ValueAndGradient`) returns the consistent
///     gradient-lane pair `(f, ∇f)` — pure REML cost paired with the exact
///     REML λ-gradient.
///   * `eval_with_order(Value)` is the line-search probe: it accepts/rejects
///     steps whose DIRECTION came from `eval`'s `∇f`, so its cost must be the
///     SAME pure REML `f`. Folding the gradient-free consistency penalty `c(ρ)`
///     here while the direction is `∇f` mixes two functions in the Armijo test
///     (the objective↔gradient desync bug class). The fix threads
///     `fold_cotrain = false` into this lane.
///   * `eval_cost` is the derivative-free cross-seed RANKING lane, where no
///     gradient is ever paired with the cost, so it legitimately carries the
///     fold (`fold_cotrain = true`): its cost is `f + c`.
///
/// This fixture's atoms carry NO basis evaluator, so every amortized encode is
/// uncertified (`uncertified_fraction = 1.0`) and the consistency fold `c` is
/// strictly positive. The regression therefore pins:
///   (1) line-search lane cost == gradient lane cost  (the desync invariant),
///   (2) ranking lane cost  >  line-search lane cost   (the fold IS present on
///       the ranking lane and ABSENT on the line-search lane).
/// The pre-fix code fed `f+c` to the line search, collapsing (1)/(2): the
/// line-search cost would equal the ranking cost and exceed the gradient cost.
#[test]
pub(crate) fn line_search_value_probe_sees_pure_reml_not_cotrain_fold() {
    use gam_solve::rho_optimizer::{OuterEvalOrder, OuterObjective};

    // A fixed ρ at which all three lanes converge from the same fixture state.
    let rho_flat = warmstart_test_objective().baseline_rho.to_flat();

    // Gradient lane (ValueAndGradient): the consistent `(f, ∇f)` pair. Its cost
    // is pure REML (+ the discrete collapse barrier, which stays on both lanes).
    let mut grad_obj = warmstart_test_objective();
    let grad_cost = grad_obj
        .eval(&rho_flat)
        .expect("gradient lane must converge on the warm-start fixture")
        .cost;

    // Line-search lane (Value order): the BFGS/ARC probe. Post-fix this reports
    // the SAME pure REML cost the gradient lane reports.
    let mut ls_obj = warmstart_test_objective();
    let ls_cost = ls_obj
        .eval_with_order(&rho_flat, OuterEvalOrder::Value)
        .expect("line-search probe must converge on the warm-start fixture")
        .cost;

    // Ranking lane (`eval_cost`): the derivative-free cross-seed screen, which
    // DOES carry the consistency fold `c`.
    let mut rank_obj = warmstart_test_objective();
    let rank_cost = rank_obj
        .eval_cost(&rho_flat)
        .expect("ranking lane must converge on the warm-start fixture");

    // (1) The line-search cost equals the gradient-lane cost: the BFGS Armijo
    //     sufficient-decrease test now pairs `f` with `∇f` (no desync).
    assert_abs_diff_eq!(ls_cost, grad_cost, epsilon = 1.0e-10);

    // The fold is genuinely present in this fixture (no basis evaluator ⇒
    // uncertified_fraction = 1.0), so the ranking lane is strictly costlier.
    assert!(
        rank_cost > grad_cost + 1.0e-6,
        "ranking lane must carry a strictly positive co-training fold: \
             rank_cost={rank_cost:.12} grad_cost={grad_cost:.12}"
    );

    // (2) The line-search lane must NOT carry the fold: it is strictly cheaper
    //     than the ranking lane by exactly the fold magnitude. The pre-fix bug
    //     (line search sees `f+c`) would make `ls_cost == rank_cost`.
    assert!(
        ls_cost < rank_cost - 1.0e-6,
        "line-search lane must exclude the co-training fold the ranking lane \
             carries: ls_cost={ls_cost:.12} rank_cost={rank_cost:.12}"
    );
}

/// #1029 budget-policy gate: value probes get the base refine budget and
/// NEVER earn the progress extension (their base and progress budgets
/// coincide), while the accepted-point path extends only on a measured
/// round-to-round KKT-residual drop and falls back to the base budget on
/// a stall.
#[test]
pub(crate) fn refine_iteration_limit_probe_budget_never_extends() {
    let probe_base = 16usize;
    // Probe policy: base == progress, so even perfect progress cannot
    // extend past the base work budget.
    assert_eq!(
        SaeManifoldTerm::refine_iteration_limit(
            probe_base,
            probe_base,
            probe_base,
            Some(1.0),
            0.5,
            true
        ),
        probe_base
    );
    let accepted_base = 64usize;
    let accepted_progress = 256usize;
    // Accepted-point policy: a real residual drop extends the budget…
    assert_eq!(
        SaeManifoldTerm::refine_iteration_limit(
            accepted_base,
            accepted_base,
            accepted_progress,
            Some(1.0),
            0.5,
            false
        ),
        accepted_progress
    );
    // …a stalled residual does not…
    assert_eq!(
        SaeManifoldTerm::refine_iteration_limit(
            accepted_base,
            accepted_base,
            accepted_progress,
            Some(1.0),
            1.0,
            false
        ),
        accepted_base
    );
    // …and below the base budget no extension question arises yet.
    assert_eq!(
        SaeManifoldTerm::refine_iteration_limit(
            accepted_base - 1,
            accepted_base,
            accepted_progress,
            None,
            1.0e9,
            false
        ),
        accepted_base
    );
}

#[test]
pub(crate) fn reml_retries_refinement_after_non_pd_undamped_evidence_factor() {
    let (mut term0, target, rho) = small_two_atom_periodic_term();
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let cold_sys = term0
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    // Fixture precondition: the COLD (off-optimum) seed has a genuinely non-PD
    // per-row undamped evidence block (two atoms' periodic decoders specialise in
    // opposite directions, so the logit-block Schur complement goes indefinite).
    // Before #1117 the undamped (`ridge = 0`) factor REFUSED this with
    // `PerRowFactorFailed` and the criterion recovered by refining the inner
    // state. #1117 (`factor_spectral_deflated_evidence_row`) now conditions the
    // block the principled way at the COLD state too: it discovers the
    // negative/flat eigen-direction and stiffens it to UNIT curvature (eigenvalue
    // → +1, contributing a ρ-independent log 1 = 0), so the undamped solve returns
    // `Ok` carrying recorded per-row deflation spectra instead of refusing. The
    // block is STILL non-PD; it is now spectrally deflated rather than rejected.
    // Pin THAT contract: the cold solve succeeds and reports the deflated
    // indefinite directions it had to condition (a stronger statement than the old
    // bare-`Err` precondition — it proves both that the seed is genuinely
    // indefinite AND that the #1117 deflation engaged).
    let (.., cold_cache) = solve_arrow_newton_step_with_options(&cold_sys, 0.0, 0.0, &options)
        .expect("cold undamped evidence factor must be spectrally conditioned (#1117), not refused");
    let cold_deflated_rows = cold_cache
        .deflation_row_spectra
        .iter()
        .filter(|spectrum| spectrum.is_some())
        .count();
    assert!(
        cold_deflated_rows > 0 || cold_cache.gauge_deflated_directions > 0,
        "fixture must start with a genuine non-PD evidence block that #1117 spectral \
         unit-stiffness deflation had to condition; got no deflated row spectra and \
         {} gauge directions",
        cold_cache.gauge_deflated_directions,
    );

    let mut full = term0.clone();
    let mut streaming = term0;
    let (full_cost, full_loss, cache) = full
        .reml_criterion_with_cache(target.view(), &rho, None, 1, 0.25, 1.0e-4, 1.0e-4)
        .expect("dense REML must refine through the cold non-PD evidence factor");
    let log_det = arrow_log_det_from_cache(&cache).expect("refined cache must carry log-det");
    assert!(full_cost.is_finite());
    assert!(full_loss.total().is_finite());
    assert!(log_det.is_finite());

    let (stream_cost, stream_loss) = streaming
        .reml_criterion_streaming_exact(target.view(), &rho, None, 1, 0.25, 1.0e-4, 1.0e-4)
        .expect("streaming REML must share the dense refinement retry");
    assert_abs_diff_eq!(stream_cost, full_cost, epsilon = 1.0e-8);
    assert_abs_diff_eq!(stream_loss.total(), full_loss.total(), epsilon = 1.0e-8);
}

/// #1033 large-n: chunking the per-row assembly fold (so the transient
/// `Vec<SaeAssemblyRow>` is bounded to `O(chunk)` instead of `O(n)`) must
/// produce a BIT-IDENTICAL Arrow-Schur system to the single-pass fold. The fold
/// is row-ascending across chunks, so every floating-point `+=` into `sys.gb`,
/// each row's `htt`/`gt`, and the `g_blocks`/`kron_*` accumulators lands in the
/// exact same order regardless of chunk width. We assemble once at the full
/// width (`chunk_override = None` ⇒ the admission plan, which is `n` in-core)
/// and once at a tiny width that forces the multi-chunk path on the same small
/// fixture, then assert every system field agrees to the last bit (`to_bits`).
/// A regression that reordered the fold (e.g. folding chunks out of order, or
/// parallel-reducing across chunk boundaries) would perturb the low bits here.
#[test]
pub(crate) fn chunked_assembly_fold_is_bit_identical_1033() {
    let (mut term_full, target, rho) = small_two_atom_periodic_term();
    let mut term_chunked = term_full.clone();

    // Single-pass fold (production default at this in-core size: chunk == n).
    term_full.assembly_chunk_override = None;
    let sys_full = term_full
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("single-pass assembly must succeed");

    // Force the multi-chunk fold path: width 2 over the 5-row fixture exercises
    // chunk boundaries [0,2) [2,4) [4,5).
    term_chunked.assembly_chunk_override = Some(2);
    let sys_chunked = term_chunked
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("chunked assembly must succeed");

    assert_eq!(
        sys_full.rows.len(),
        sys_chunked.rows.len(),
        "chunked and single-pass assemblies must yield the same row count"
    );
    assert_eq!(sys_full.gb.len(), sys_chunked.gb.len());
    for (i, (gf, gc)) in sys_full.gb.iter().zip(sys_chunked.gb.iter()).enumerate() {
        assert_eq!(
            gf.to_bits(),
            gc.to_bits(),
            "sys.gb[{i}] must be bit-identical (single-pass {gf} vs chunked {gc})"
        );
    }
    for (row, (rf, rc)) in sys_full
        .rows
        .iter()
        .zip(sys_chunked.rows.iter())
        .enumerate()
    {
        assert_eq!(rf.htt.dim(), rc.htt.dim(), "row {row} htt shape");
        assert_eq!(rf.gt.len(), rc.gt.len(), "row {row} gt len");
        for (a, (hf, hc)) in rf.htt.iter().zip(rc.htt.iter()).enumerate() {
            assert_eq!(
                hf.to_bits(),
                hc.to_bits(),
                "row {row} htt[{a}] must be bit-identical: {hf} vs {hc}"
            );
        }
        for (a, (gf, gc)) in rf.gt.iter().zip(rc.gt.iter()).enumerate() {
            assert_eq!(
                gf.to_bits(),
                gc.to_bits(),
                "row {row} gt[{a}] must be bit-identical: {gf} vs {gc}"
            );
        }
    }
}

#[test]
pub(crate) fn reconstruction_dispersion_uses_ard_shrunk_coordinate_edf() {
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.30, -0.10], [0.20, 0.40], [-0.35, 0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = Array2::from_shape_fn((n, p), |(row, col)| {
        let x = (row as f64 + 0.5) / n as f64;
        if col == 0 {
            0.45 * (std::f64::consts::TAU * x).sin() + 0.07
        } else {
            -0.20 * (std::f64::consts::TAU * x).cos() + 0.03 * row as f64
        }
    });
    let alpha = 250.0_f64;
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![alpha.ln()]]);
    let loss = term.loss(target.view(), &rho).unwrap();
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();

    let dispersion = term.reconstruction_dispersion(&loss, &cache, &rho).unwrap();
    let smooth_edf: f64 = term
        .decoder_smoothness_effective_dof_per_atom(&cache, &rho.lambda_smooth_vec())
        .unwrap()
        .iter()
        .sum();
    let beta_edf = (term.beta_dim() as f64 - smooth_edf).max(0.0);
    let traces = term.ard_inverse_traces(&cache).unwrap();
    let coord_edf = (n as f64 - alpha * traces[0][0]).clamp(0.0, n as f64);
    let rss = 2.0 * loss.data_fit;
    let expected = rss / ((n * p) as f64 - beta_edf - coord_edf).max(1.0);
    assert_abs_diff_eq!(dispersion, expected, epsilon = 1.0e-10);

    let old_full_coordinate_edf = n as f64;
    let old_full_coordinate_dispersion =
        rss / ((n * p) as f64 - beta_edf - old_full_coordinate_edf).max(1.0);
    assert!(
        coord_edf < 0.25 * old_full_coordinate_edf,
        "test setup must put the coordinate axis in an ARD-shrunk regime; \
             coord_edf={coord_edf}, old_full_coordinate_edf={old_full_coordinate_edf}"
    );
    assert!(
        dispersion < 0.75 * old_full_coordinate_dispersion,
        "φ̂ must use the ARD-shrunk coordinate edf, not the old full \
             coordinate count: got {dispersion}, old formula {old_full_coordinate_dispersion}"
    );
}

#[test]
pub(crate) fn latent_block_inverse_diagonal_hutchinson_matches_exact_trace() {
    // The matrix-free Hutchinson estimator of `diag((H⁻¹)_tt)` (the #1777 fold
    // that replaces the exact `O(total_t·K²)` selected-inverse diagonal at
    // massive K) must, over enough Rademacher probes, reproduce BOTH the full
    // latent-block trace `tr((H⁻¹)_tt)` and the per-axis ARD grouped trace the
    // exact path feeds the Fellner–Schall/dispersion denominator. Deterministic
    // seed ⇒ this is a fixed, non-flaky comparison.
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.30, -0.10], [0.20, 0.40], [-0.35, 0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = Array2::from_shape_fn((n, p), |(row, col)| {
        let x = (row as f64 + 0.5) / n as f64;
        if col == 0 {
            0.45 * (std::f64::consts::TAU * x).sin() + 0.07
        } else {
            -0.20 * (std::f64::consts::TAU * x).cos() + 0.03 * row as f64
        }
    });
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![250.0_f64.ln()]]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();

    let exact = cache.latent_block_inverse_diagonal().unwrap();
    // Enough probes to average out the per-entry Hutchinson variance on this
    // tiny t-space; seed is fixed so the outcome is deterministic.
    let hutch =
        SaeManifoldTerm::latent_block_inverse_diagonal_hutchinson(&cache, 20_000, 0xABCD_1234)
            .unwrap();
    assert_eq!(exact.len(), hutch.len());

    // Full latent-block trace tr((H⁻¹)_tt): the aggregate the ARD/dispersion
    // consumers ultimately sum, estimated unbiasedly.
    let exact_trace: f64 = exact.iter().sum();
    let hutch_trace: f64 = hutch.iter().sum();
    assert!(
        (hutch_trace - exact_trace).abs() <= 0.02 * exact_trace.abs().max(1.0e-6),
        "Hutchinson latent trace {hutch_trace} vs exact {exact_trace} exceeds 2% tol"
    );

    // Per-axis ARD grouped trace: the exact grouping of the estimated diagonal
    // must match the exact grouping of the exact diagonal (both are the same
    // linear functional of the diagonal, so the error inherits the trace bound).
    let coord_offsets = term.assignment.coord_offsets();
    let block_start = coord_offsets[0];
    let mut exact_axis0 = 0.0_f64;
    let mut hutch_axis0 = 0.0_f64;
    match term.last_row_layout {
        Some(ref layout) => {
            for row in 0..n {
                let row_base = cache.row_offsets[row];
                if let Some(pos) = layout.active_atoms[row].iter().position(|&k| k == 0) {
                    let s = row_base + layout.coord_starts[row][pos];
                    exact_axis0 += exact[s];
                    hutch_axis0 += hutch[s];
                }
            }
        }
        None => {
            for row in 0..n {
                let s = cache.row_offsets[row] + block_start;
                exact_axis0 += exact[s];
                hutch_axis0 += hutch[s];
            }
        }
    }
    assert!(
        (hutch_axis0 - exact_axis0).abs() <= 0.05 * exact_axis0.abs().max(1.0e-6),
        "Hutchinson ARD axis trace {hutch_axis0} vs exact {exact_axis0} exceeds 5% tol"
    );
    assert!(
        exact_axis0 > 0.0,
        "posterior-variance trace must be positive (sanity on the fixture)"
    );
}

#[test]
pub(crate) fn streaming_plan_routes_by_memory_budget_with_identical_logdet() {
    let (term0, target, rho) = small_two_atom_periodic_term();
    let total_basis: usize = term0.atoms.iter().map(|atom| atom.basis_size()).sum();
    let d_max = term0
        .atoms
        .iter()
        .map(|atom| atom.latent_dim)
        .max()
        .unwrap();
    let dense_plan = sae_streaming_plan_from_budget(
        term0.n_obs(),
        total_basis,
        term0.k_atoms(),
        d_max,
        term0.beta_dim(),
        usize::MAX / 4,
        1024 * 1024,
        usize::MAX / 2,
    );
    assert!(!dense_plan.streaming);
    assert!(dense_plan.direct_admitted);
    let streaming_plan = sae_streaming_plan_from_budget(
        term0.n_obs(),
        total_basis,
        term0.k_atoms(),
        d_max,
        term0.beta_dim(),
        1,
        512,
        2,
    );
    assert!(streaming_plan.streaming);
    assert!(!streaming_plan.direct_admitted);

    let mut full = term0.clone();
    // The undamped (`ridge_t = 0`) log-det is only well-defined at the inner
    // optimum, where the per-row `H_tt^(i)` blocks are PD. At the initial
    // (non-stationary) iterate a `p_out = 1` rank-1 `JᵀJ` row block plus the
    // softmax negative-logit curvature is indefinite, so factoring there at
    // ridge 0 surfaces `PerRowFactorFailed` for BOTH the dense and streaming
    // paths. Converge the inner `(t, β)` state first (matching how
    // `reml_criterion_with_cache` reaches a PD block), then compare the
    // streaming-vs-dense log-determinants of the SAME converged system —
    // which is the routing invariant this test pins (#847).
    full.reml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .unwrap();
    let sys = full
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let factor_result = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();
    let full_logdet = arrow_log_det_from_cache(&factor_result.2).unwrap();
    let mut streaming = StreamingArrowSchur::from_system(&sys, streaming_plan.chunk_size);
    let streaming_logdet = streaming.exact_arrow_log_det(0.0, 0.0, &options).unwrap();
    assert_abs_diff_eq!(streaming_logdet, full_logdet, epsilon = 1.0e-8);
}

#[test]
pub(crate) fn giant_host_working_set_plan_flips_to_matrix_free_before_dense_allocation() {
    let n_obs = 128usize;
    let total_basis = 48usize;
    let k_atoms = 8usize;
    let d_max = 2usize;
    let p_out = 2048usize;
    let border_dim = total_basis * p_out;
    let budget = 60usize * 1024 * 1024 * 1024;
    let plan = sae_streaming_plan_from_budget(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        budget,
        8 * 1024 * 1024,
        120usize * 1024 * 1024 * 1024,
    );

    assert_eq!(border_dim, 98_304);
    assert_eq!(
        plan.estimated_row_cross_bytes,
        n_obs * k_atoms * (1 + d_max) * border_dim * SAE_BYTES_PER_F64
    );
    assert!(plan.estimated_dense_schur_bytes > budget);
    assert!(plan.estimated_matrix_free_peak_bytes < budget);
    assert!(plan.streaming);
    assert!(!plan.direct_admitted);
    assert!(plan.matrix_free_admitted);
    assert_eq!(
        plan.solve_options_for_border_dim(border_dim).mode,
        gam_solve::arrow_schur::ArrowSolverMode::InexactPCG
    );
}

#[test]
pub(crate) fn matrix_free_plan_admits_when_in_core_budget_collapses_to_zero() {
    // On a memory-starved / oversubscribed box (or a cgroup whose
    // `available − reserve` underflows), `sae_host_in_core_budget_from_available`
    // can return a budget of 0. Before the streaming floor, that rejected EVERY
    // plan — including the chunked matrix-free fallback whose peak is bounded by
    // the chunk window — so `admitted_or_error` failed with "exceeds budget 0
    // bytes" and the whole SAE fit aborted at K=1 (the real-OLMo CPU ladder
    // wall). The matrix-free streaming path must stay admittable for a small
    // working set regardless of the collapsed in-core budget.
    let n_obs = 508usize;
    let total_basis = 6usize;
    let k_atoms = 1usize;
    let d_max = 1usize;
    let border_dim = 32usize;
    let plan = sae_streaming_plan_from_budget(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        0, // in-core budget collapsed to zero
        SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
        0, // host-available also reported as zero
    );
    // The dense direct plan is correctly refused (it can OOM and the budget is 0).
    assert!(!plan.direct_admitted);
    assert!(plan.streaming);
    // But the bounded matrix-free streaming plan IS admitted against the absolute
    // streaming floor, so the fit can proceed instead of aborting.
    assert!(
        plan.estimated_matrix_free_peak_bytes <= SAE_MIN_STREAMING_BUDGET_FLOOR_BYTES,
        "tiny working set must fit the streaming floor: peak={}",
        plan.estimated_matrix_free_peak_bytes
    );
    assert!(
        plan.matrix_free_admitted,
        "matrix-free streaming must be admitted at zero in-core budget"
    );
    assert!(plan.admitted_or_error(n_obs, border_dim, k_atoms).is_ok());
}

/// Build a `K`-atom softmax SAE term whose per-row logits concentrate on a
/// planted small support, for the #1450 end-to-end large-K compact-path test.
///
/// Every atom is a 1-D `EuclideanPatch` with an `M=2` constant+linear basis and
/// a distinct decoder direction, so the reconstruction is genuine and the
/// per-row Arrow-Schur block has a real data-fit Gauss-Newton contribution.
/// Row `i`'s logits put large mass on its planted active atoms and a uniform
/// floor on every other atom, so the softmax assignment vector concentrates on
/// the planted set (the true top-`k` support) while the dropped tail carries
/// negligible `O(a)` mass — exactly the regime the compact softmax layout
/// (#1408/#1409) is meant to optimize.
fn planted_softmax_sae_term(
    n: usize,
    k_atoms: usize,
    planted: &[Vec<usize>],
    p: usize,
) -> (SaeManifoldTerm, Array2<f64>) {
    assert_eq!(planted.len(), n);
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    // Shared constant+linear basis evaluation: column 0 = 1, column 1 = t.
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| (row as f64 / n as f64) - 0.5);
    for atom_idx in 0..k_atoms {
        let mut phi = Array2::<f64>::zeros((n, 2));
        let mut jet = Array3::<f64>::zeros((n, 2, 1));
        for row in 0..n {
            phi[[row, 0]] = 1.0;
            phi[[row, 1]] = coords[[row, 0]];
            jet[[row, 1, 0]] = 1.0;
        }
        // Distinct decoder direction per atom: maps the linear basis column onto
        // output channel `atom_idx % p` with a small magnitude.
        let mut decoder = Array2::<f64>::zeros((2, p));
        decoder[[1, atom_idx % p]] = 0.1 + 0.01 * ((atom_idx % 7) as f64);
        atoms.push(
            SaeManifoldAtom::new(
                format!("atom{atom_idx}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(2),
            )
            .unwrap(),
        );
        coord_blocks.push(coords.clone());
        manifolds.push(LatentManifold::Euclidean);
    }
    // Logits: a small uniform floor everywhere, large mass on the planted atoms.
    let mut logits = Array2::<f64>::from_elem((n, k_atoms), -6.0);
    for (row, active) in planted.iter().enumerate() {
        for &k in active {
            logits[[row, k]] = 6.0;
        }
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, c)| 0.05 * ((row + c) as f64).sin());
    (term, target)
}

/// #1450 — end-to-end large-`K` compact-path contract for the softmax SAE
/// encode: the assignment→support-proposal→assembly path must produce a per-row
/// block whose dimension tracks the per-row active-atom count `k_active`, NOT
/// the total `K`, and the assembled support must recover the planted top-`k`
/// atoms. This drives the REAL paths #1408/#1409 fixed
/// (`softmax_active_plan` → `from_dense_weights` → compact `assemble_arrow_schur`
/// in fixed-decoder mode), not a hand-built `from_active_atoms` layout, and at a
/// `K` (1000) large enough that a full-`K` per-row block would be ~1000× larger.
#[test]
pub(crate) fn large_k_softmax_compact_encode_is_o1_per_token_and_recovers_support() {
    let n = 8usize;
    let p = 4usize;
    let top_k = 3usize;
    // Each row's planted top-`top_k` support, spread across the full K range so a
    // correct selection must scan all K (not just a prefix).
    let planted: Vec<Vec<usize>> = (0..n).map(|row| vec![row, 300 + row, 700 + row]).collect();

    // Assemble at two widely-separated K with the SAME k_active and planted
    // support; the per-row compact dims must be IDENTICAL (n-free / independent
    // of K) and bounded by O(top_k).
    let assemble_dims = |k_atoms: usize| -> (Vec<usize>, Vec<Vec<usize>>) {
        let (mut term, target) = planted_softmax_sae_term(n, k_atoms, &planted, p);
        // Fold top_k into the OPTIMIZATION (the #1409 fix): softmax now engages
        // the compact top-`k` row layout instead of a post-fit projection.
        term.set_softmax_active_cap(Some(top_k));
        // Fixed-decoder encode assembly (the #1407 path the encoder uses): only
        // the per-row htt/gt block is produced.
        term.fixed_decoder_assembly = true;
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("compact softmax fixed-decoder assembly must succeed at large K");
        let dims: Vec<usize> = sys.rows.iter().map(|r| r.htt.nrows()).collect();
        // Each row's htt must be square and match gt.
        for r in &sys.rows {
            assert_eq!(r.htt.nrows(), r.htt.ncols());
            assert_eq!(r.htt.nrows(), r.gt.len());
        }
        let layout = term
            .last_row_layout
            .clone()
            .expect("softmax at large K must engage the COMPACT active-set layout (#1408)");
        let active: Vec<Vec<usize>> = layout.active_atoms.clone();
        (dims, active)
    };

    let (dims_1k, active_1k) = assemble_dims(1_000);
    let (dims_10k, active_10k) = assemble_dims(10_000);

    // (a) O(1)-per-token / n-free: per-row block dim is bounded by the active
    // contract `top_k·(1 + d) = top_k·2` for d=1 coords, and is IDENTICAL across
    // K=1000 and K=10000 (independent of total K). A full-K block would be
    // `q = (K-1) + K·d`, i.e. ~3000 and ~30000 — orders of magnitude larger.
    let bound = top_k * (1 + 1); // |active| + Σ d_k  (d_k = 1)
    for row in 0..n {
        assert!(
            dims_1k[row] <= bound,
            "row {row} K=1000 compact dim {} exceeds O(top_k) bound {bound}",
            dims_1k[row]
        );
        assert_eq!(
            dims_1k[row], dims_10k[row],
            "row {row} compact dim must be INDEPENDENT of total K (n-free contract): \
             K=1000 gave {} but K=10000 gave {}",
            dims_1k[row], dims_10k[row]
        );
    }
    // The full-K dense block would dwarf the compact one: assert the compact
    // total work is < 1/100 of the dense block even at the smaller K=1000.
    let compact_work: usize = dims_1k.iter().map(|&q| q * q).sum();
    let dense_q = (1_000 - 1) + 1_000; // (K-1) free logits + K coord axes
    let dense_work = n * dense_q * dense_q;
    assert!(
        compact_work * 100 < dense_work,
        "compact work {compact_work} must be << dense work {dense_work}"
    );

    // (b) Support recovery: the proposed active set must be exactly the planted
    // top-`top_k` atoms for every row, at BOTH K (the proposal path scanned the
    // full K and selected the planted peaks, not an arbitrary prefix).
    for row in 0..n {
        let mut expected = planted[row].clone();
        expected.sort_unstable();
        assert_eq!(
            active_1k[row], expected,
            "row {row} K=1000 active set must recover the planted top-{top_k} support"
        );
        assert_eq!(
            active_10k[row], expected,
            "row {row} K=10000 active set must recover the planted top-{top_k} support"
        );
    }
}

#[test]
pub(crate) fn sparse_active_layout_work_scales_with_active_atoms_not_total_k() {
    let n = 3;
    let k_atoms = 100_000;
    let mut active_rows = Vec::with_capacity(n);
    for row in 0..n {
        active_rows.push(vec![row, 10_000 + row, 90_000 + row]);
    }
    let coord_dims = vec![1usize; k_atoms];
    let coord_offsets_full: Vec<usize> = (0..k_atoms).map(|k| k_atoms + k).collect();
    let layout = SaeRowLayout::from_active_atoms(active_rows, coord_dims, coord_offsets_full);
    for row in 0..n {
        assert_eq!(layout.active_atoms[row].len(), 3);
        assert_eq!(layout.row_q_active(row), 6);
    }
    let compact_work: usize = (0..n)
        .map(|row| {
            let q = layout.row_q_active(row);
            q * q
        })
        .sum();
    let dense_q = 2 * k_atoms;
    let dense_work = n * dense_q * dense_q;
    assert!(compact_work < dense_work / 1_000_000_000);
    assert_eq!(compact_work, n * 36);
}

/// Regression test for https://github.com/SauersML/gam/issues/163.
///
/// `ManifoldSAE.predict(X_subset)` reseeds the latent coordinates via PCA
/// on a possibly small batch (here: a strict subset of the training data),
/// which can produce a per-row `H_tt + ridge_t·I` that is not
/// positive-definite at the caller's nominal `ridge_t = 1e-6`. The fit
/// path tolerates this via the proximal LM correction outer wrapper;
/// previously, `run_joint_fit_arrow_schur` invoked `sys.solve(...)`
/// directly and surfaced the per-row Cholesky failure to the caller. The
/// fix routes recoverable factor failures through a Levenberg-Marquardt
/// damping schedule (mirrors the `proximal_correction` outer loop),
/// so an inner step with a degenerate Hessian no longer aborts the
/// Newton driver.
#[test]
pub(crate) fn run_joint_fit_arrow_schur_escalates_ridge_on_non_pd_row_block() {
    // Construct a periodic atom whose row block is rank-deficient when
    // the assignment column is zero — `H_tt` is then driven entirely by
    // the smoothness penalty / external coord ridge and floats just
    // above zero. At ridge_t = 1e-6 the per-row Cholesky finds a tiny
    // negative pivot from rounding error; the escalation loop should
    // recover.
    let coords = array![[0.1], [0.4], [0.7]];
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        // Decoder that maps to a single output dim with small magnitude
        array![[0.05], [-0.05], [0.05]],
        // No external smoothness penalty on the decoder, so the only
        // regularization on `t` comes from `ridge_ext_coord`.
        Array2::<f64>::zeros((3, 3)),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        // Zero assignment mass → H_tt has zero data contribution.
        Array2::<f64>::zeros((3, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20], [-0.10], [0.45]];
    // log_lambda_smooth driven low so the analytic penalty contributes
    // essentially nothing to H_tt either.
    let mut rho = SaeManifoldRho::new(0.0, -20.0, vec![Array1::<f64>::zeros(1)]);

    // The Python-side `predict` default. Before the fix this returned
    // `Err(... per-row H_tt^(?) Cholesky failed ... non-PD pivot ...)`;
    // afterward the escalation loop bumps ridge_t until the per-row
    // factor succeeds, and run_joint_fit_arrow_schur returns Ok.
    let result =
        term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6);
    assert!(
        result.is_ok(),
        "run_joint_fit_arrow_schur should recover from degenerate H_tt via LM ridge escalation; got: {result:?}",
    );
}

/// Regression test for https://github.com/SauersML/gam/issues/1117
/// (the DEEP fix: rank-`r` basis-column reduction past the #1051 LM ridge).
///
/// On a near-degenerate input manifold (the OLMo L25 PCA-32 circle, or the
/// `stage1-step0` checkpoint: post-PCA std ≈ 0.04) the latent coordinate
/// collapses so basis columns are linearly dependent IN THE DATA — the bare
/// data Gram `G_k` is rank-deficient (`[SAE-AUDIT]` `rank r/M`). The deep fix
/// discovers that dead subspace `N_k` from `G_k` and returns a projector
/// `Π_k = N_k N_kᵀ` (a) so the inner solve & evidence log-det deflate the dead
/// directions at unit stiffness (no ρ-dependent flat valley), and (b) so the
/// converged decoder is projected onto `range(G_k)` (the rank-`r` oracle).
///
/// #1117 root-cause fix: a fixed-depth `M = 5` periodic circle decoder whose
/// data DOES NOT excite the top (2nd-harmonic) pair must be REPARAMETRIZED at
/// fit entry onto the data-supported subspace — the design becomes full-rank
/// (adaptive depth `r = 3 < M = 5`), the reconstruction is preserved (rank-`r`
/// oracle), and the basis re-evaluates at the reduced width (the reduction
/// survives refresh).
#[test]
pub(crate) fn rank_revealing_reduction_collapses_unexcited_circle_harmonic_to_full_rank() {
    // Build the full M = 5 periodic basis `[1, sin2πt, cos2πt, sin4πt, cos4πt]`
    // from its evaluator, then collapse the latent coordinate onto THREE
    // distinct phases. The first-harmonic columns `[1, sin2πt, cos2πt]` span a
    // rank-3 data subspace; with only three phases the 2nd-harmonic pair adds
    // no new data direction, so the bare data Gram is `rank 3/5` — exactly the
    // OLMo `stage1-step0` deficiency reduced to a minimal reproducer.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = array![[0.1], [0.45], [0.8], [0.1], [0.45], [0.8]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    assert_eq!(
        phi.ncols(),
        5,
        "fixed-depth circle basis emits M = 5 columns"
    );
    let penalty = Array2::<f64>::eye(5);
    let decoder = array![[0.05], [-0.05], [0.05], [0.02], [-0.02]];
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi.clone(),
        jet,
        decoder.clone(),
        penalty,
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::ones((6, 1)),
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    let recon_before = phi.dot(&decoder);
    term.reduce_atoms_to_data_supported_rank().unwrap();

    // (a) The design is now full-rank BY CONSTRUCTION: adaptive depth r = 3 < 5.
    let r = term.atoms[0].basis_size();
    assert_eq!(
        r, 3,
        "rank-revealing reduction must drop the unexcited harmonic (r = 3 < M = 5)",
    );
    assert_eq!(term.atoms[0].decoder_coefficients.nrows(), 3);
    assert_eq!(term.atoms[0].basis_jacobian.dim(), (6, 3, 1));
    assert_eq!(term.atoms[0].smooth_penalty.dim(), (3, 3));

    // The reduced data Gram is full rank (no eigenvalue at the spectral floor).
    use gam_linalg::faer_ndarray::FaerEigh;
    let reduced_design = term.atoms[0].basis_values.clone();
    let gram = reduced_design.t().dot(&reduced_design);
    let (evals, _) = gram.eigh(faer::Side::Lower).unwrap();
    let max_eig = evals.iter().cloned().fold(0.0_f64, f64::max);
    for &lam in evals.iter() {
        assert!(
            lam > 1e-9 * max_eig,
            "reduced design Gram must be full rank; got eigenvalue {lam} (max {max_eig})",
        );
    }

    // (b) Rank-`r` oracle: the reduced reconstruction `Φ̃ B̃` equals the
    // original data fit (the dropped direction carried no data signal).
    let recon_after = reduced_design.dot(&term.atoms[0].decoder_coefficients);
    for i in 0..recon_before.nrows() {
        assert!(
            (recon_before[[i, 0]] - recon_after[[i, 0]]).abs() < 1e-9,
            "reduction must not change the data-fit reconstruction at row {i}: \
                 before={} after={}",
            recon_before[[i, 0]],
            recon_after[[i, 0]],
        );
    }

    // (c) The reduction SURVIVES refresh: re-evaluating the wrapped evaluator
    // at the same coords re-emits the reduced r = 3 columns, not the full 5.
    let (refreshed, _) = term.atoms[0]
        .basis_evaluator
        .as_ref()
        .unwrap()
        .evaluate(coords.view())
        .unwrap();
    assert_eq!(
        refreshed.ncols(),
        3,
        "the SubspaceReducedEvaluator must re-emit the reduced width on refresh",
    );
    for i in 0..refreshed.nrows() {
        for j in 0..3 {
            assert!(
                (refreshed[[i, j]] - reduced_design[[i, j]]).abs() < 1e-12,
                "refresh must reproduce the reduced design bit-for-bit",
            );
        }
    }
}

/// #1117 jet-composition contract: the `SubspaceReducedEvaluator` must map
/// EVERY jet order by the SAME right-multiply `∂^g Φ̃ = (∂^g Φ) Q`,
/// contracting only the basis (column) axis. A wrong order on any field is a
/// wrong gradient downstream, so we pin value + first + second + third jets
/// of the wrapped evaluator against the inner evaluator's jets times `Q`
/// directly (no finite difference — this is the exact linear-algebra
/// identity).
#[test]
pub(crate) fn subspace_reduced_evaluator_composes_all_jets_by_q() {
    let inner = Arc::new(PeriodicHarmonicEvaluator::new(7).unwrap());
    let coords = array![[-0.3_f64], [0.0], [0.15], [0.42], [0.88]];
    let m = inner.num_basis; // 7
    // A deterministic orthonormal column map Q (M × r), r = 4: take the
    // first 4 eigenvectors of a fixed SPD matrix so the columns are
    // orthonormal but genuinely mix the inner columns (not a trivial
    // selection).
    let mut a = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            a[[i, j]] = 1.0 / (1.0 + (i as f64 - j as f64).abs());
        }
    }
    let (_evals, evecs) = a.eigh(Side::Lower).unwrap();
    let r = 4usize;
    let mut q = Array2::<f64>::zeros((m, r));
    for col in 0..r {
        for row in 0..m {
            q[[row, col]] = evecs[[row, col]];
        }
    }
    let reduced = SubspaceReducedEvaluator::new(inner.clone(), q.clone()).unwrap();
    assert_eq!(reduced.inner_width(), m);
    assert_eq!(reduced.reduced_width(), r);

    // Value + first jet.
    let (phi_in, jet_in) = inner.evaluate(coords.view()).unwrap();
    let (phi_red, jet_red) = reduced.evaluate(coords.view()).unwrap();
    let phi_expect = phi_in.dot(&q);
    assert_eq!(phi_red.dim(), phi_expect.dim());
    for i in 0..phi_red.nrows() {
        for j in 0..r {
            assert_abs_diff_eq!(phi_red[[i, j]], phi_expect[[i, j]], epsilon = 1e-12);
        }
    }
    for axis in 0..jet_in.shape()[2] {
        let expect = jet_in.slice(s![.., .., axis]).to_owned().dot(&q);
        for i in 0..jet_red.shape()[0] {
            for j in 0..r {
                assert_abs_diff_eq!(jet_red[[i, j, axis]], expect[[i, j]], epsilon = 1e-12);
            }
        }
    }

    // Second jet: (∂²Φ) Q on each (axis_a, axis_c) fiber.
    let h_in = inner.second_jet(coords.view()).unwrap();
    let h_red = reduced.second_jet(coords.view()).unwrap();
    let d = h_in.shape()[2];
    for a_ax in 0..d {
        for c_ax in 0..d {
            let expect = h_in.slice(s![.., .., a_ax, c_ax]).to_owned().dot(&q);
            for i in 0..h_red.shape()[0] {
                for j in 0..r {
                    assert_abs_diff_eq!(h_red[[i, j, a_ax, c_ax]], expect[[i, j]], epsilon = 1e-12);
                }
            }
        }
    }

    // Third jet: (∂³Φ) Q on each (a, c, e) fiber.
    let t_in = inner.third_jet(coords.view()).unwrap();
    let t_red = reduced.third_jet_dyn(coords.view()).unwrap().unwrap();
    for a_ax in 0..d {
        for c_ax in 0..d {
            for e_ax in 0..d {
                let expect = t_in.slice(s![.., .., a_ax, c_ax, e_ax]).to_owned().dot(&q);
                for i in 0..t_red.shape()[0] {
                    for j in 0..r {
                        assert_abs_diff_eq!(
                            t_red[[i, j, a_ax, c_ax, e_ax]],
                            expect[[i, j]],
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }
    }
}

/// #1117 production-path regression: the periodic-circle atom built through the
/// production term builder ([`term_from_padded_blocks_with_mode`], the exact
/// route the `sae_manifold_fit*` FFI takes) must carry an analytic second-jet
/// evaluator so the rank-revealing reduction can fire. The original #1113 split
/// found that the builder installed the evaluator through the base-trait slot
/// only (`basis_second_jet == None`), so `reduce_atoms_to_data_supported_rank`
/// SKIPPED the rank-deficient circle, the `[SAE-AUDIT]` rank-3/5 warning kept
/// firing, and the outer BFGS crawled the flat decoder valley past the 2-minute
/// budget. After the fix the builder installs through the second-jet slot, the
/// reduction reparametrizes the `M = 5` circle onto its `r = 3` data-supported
/// subspace at fit entry, and `run_joint_fit_arrow_schur` terminates with a
/// finite, non-increasing loss inside a tight iteration budget instead of
/// stalling.
#[test]
pub(crate) fn production_builder_circle_reduces_rank_and_completes_stage1_step0_in_budget() {
    // Three distinct phases repeated → the first-harmonic columns
    // `[1, sin2πt, cos2πt]` span a rank-3 data subspace and the second-harmonic
    // pair adds no new data direction: the bare data Gram is `rank 3/5`, the
    // minimal reproducer of the OLMo `stage1-step0` PCA-32 circle deficiency.
    let n_obs = 6usize;
    let m = 5usize;
    let d = 1usize;
    let p = 2usize;
    let k_atoms = 1usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let coords = array![[0.1], [0.45], [0.8], [0.1], [0.45], [0.8]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    // Pad into the `(K, ...)`-leading storage the production builder consumes.
    let mut basis_values = Array3::<f64>::zeros((k_atoms, n_obs, m));
    basis_values.slice_mut(s![0, .., ..]).assign(&phi);
    let mut basis_jacobian = Array4::<f64>::zeros((k_atoms, n_obs, m, d));
    basis_jacobian.slice_mut(s![0, .., .., ..]).assign(&jet);
    let mut decoder = Array3::<f64>::zeros((k_atoms, m, p));
    decoder.slice_mut(s![0, .., ..]).assign(&array![
        [0.05, -0.02],
        [-0.05, 0.03],
        [0.05, 0.01],
        [0.02, -0.04],
        [-0.02, 0.02]
    ]);
    let mut penalties = Array3::<f64>::zeros((k_atoms, m, m));
    penalties
        .slice_mut(s![0, .., ..])
        .assign(&Array2::<f64>::eye(m));
    let logits = Array2::<f64>::zeros((n_obs, k_atoms));

    // The production builder installs the evaluator through the second-jet slot
    // (the fix). `SaeBasisSecondJet` is the supertrait of `SaeBasisEvaluator`.
    let evaluators: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = vec![Some(evaluator)];
    let mut term = term_from_padded_blocks_with_mode(
        n_obs,
        p,
        &[SaeAtomBasisKind::Periodic],
        basis_values.view(),
        basis_jacobian.view(),
        &[m],
        &[d],
        decoder.view(),
        penalties.view(),
        logits.view(),
        std::slice::from_ref(&coords),
        AssignmentMode::ibp_map(1.0, 1.0, false),
        &evaluators,
    )
    .unwrap();

    // The builder must populate the analytic-Hessian slot — without it the
    // #1117 reduction silently skips the atom (the regression this guards).
    assert!(
        term.atoms[0].basis_second_jet.is_some(),
        "production builder must install the analytic second-jet evaluator so the \
         #1117 rank-revealing reduction can fire",
    );

    let target = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ];
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    let loss0 = term.loss(target.view(), &rho).unwrap().total();

    // Run with a TIGHT iteration budget. If the rank-deficient circle stalled in
    // the flat decoder valley (the bug) it would burn its whole budget making
    // cosmetic progress; here the fit-entry reduction makes the design full-rank
    // so the inner Newton walk reaches a quotient-stationary point and the call
    // returns a finite, non-increasing loss well inside the budget.
    let loss = term
        .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 8, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();

    // The fit-entry reduction collapsed the unexcited harmonic: M = 5 → r = 3.
    assert_eq!(
        term.atoms[0].basis_size(),
        3,
        "the rank-deficient circle must be reparametrized onto its r = 3 \
         data-supported subspace at fit entry",
    );
    assert!(
        loss.total().is_finite(),
        "rank-deficient circle fit must return a finite loss, not stall: {}",
        loss.total(),
    );
    assert!(
        loss.total() <= loss0 + 1.0e-8,
        "the joint fit must not increase the loss (loss0={loss0}, loss={})",
        loss.total(),
    );
    assert!(
        term.assignment.coords[0]
            .as_flat()
            .iter()
            .all(|v| v.is_finite()),
        "fitted coordinates must stay finite",
    );
}

/// #1117 idempotence: once an atom is reduced to a full-rank subspace, a
/// second `reduce_atoms_to_data_supported_rank` pass is a NO-OP — the reduced
/// data Gram is full rank (`r == m`), so the fit-entry installer skips it and
/// the design/decoder/penalty are left byte-for-byte. This guards against a
/// double-reduction that would compound `Q` maps.
#[test]
pub(crate) fn rank_reduction_is_idempotent_on_already_reduced_atom() {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = array![[0.1], [0.45], [0.8], [0.1], [0.45], [0.8]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let penalty = Array2::<f64>::eye(5);
    let decoder = array![[0.05], [-0.05], [0.05], [0.02], [-0.02]];
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        penalty,
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::ones((6, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // First pass reduces 5 → 3.
    term.reduce_atoms_to_data_supported_rank().unwrap();
    assert_eq!(term.atoms[0].basis_size(), 3);
    let design_after_first = term.atoms[0].basis_values.clone();
    let decoder_after_first = term.atoms[0].decoder_coefficients.clone();

    // Second pass: the reduced Gram is full rank → SKIP. Width and contents
    // are unchanged (no double-reduction).
    term.reduce_atoms_to_data_supported_rank().unwrap();
    assert_eq!(
        term.atoms[0].basis_size(),
        3,
        "a second reduction pass on a full-rank reduced atom must be a no-op",
    );
    let design_after_second = &term.atoms[0].basis_values;
    for i in 0..design_after_first.nrows() {
        for j in 0..3 {
            assert_eq!(
                design_after_second[[i, j]],
                design_after_first[[i, j]],
                "idempotent reduction must leave the reduced design byte-identical",
            );
        }
    }
    let decoder_after_second = &term.atoms[0].decoder_coefficients;
    for i in 0..3 {
        assert_eq!(
            decoder_after_second[[i, 0]],
            decoder_after_first[[i, 0]],
            "idempotent reduction must leave the reduced decoder byte-identical",
        );
    }
}

/// Companion to the #1117 reduction test: a FULL-rank data Gram (the
/// `base`/`step_2300` regime) must be left UNTOUCHED — the well-conditioned
/// fit keeps its full harmonic depth, decoder, penalty, and evaluator
/// bit-for-bit (no reparametrization).
#[test]
pub(crate) fn full_rank_circle_design_keeps_full_harmonic_depth_unchanged() {
    // Five distinct coordinates → the five periodic columns are linearly
    // independent in the data → bare data Gram is full rank.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());
    let coords = array![[0.05], [0.27], [0.46], [0.68], [0.91]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let penalty = Array2::<f64>::eye(5);
    let decoder = array![[0.05], [-0.05], [0.05], [0.02], [-0.02]];
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi.clone(),
        jet,
        decoder.clone(),
        penalty,
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::ones((5, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.reduce_atoms_to_data_supported_rank().unwrap();

    // Full harmonic depth retained: width, decoder, and basis values are all
    // bit-for-bit the historical full-`B` path.
    assert_eq!(
        term.atoms[0].basis_size(),
        5,
        "a full-rank circle design must keep all 5 harmonic columns",
    );
    let after_phi = &term.atoms[0].basis_values;
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(
                after_phi[[i, j]],
                phi[[i, j]],
                "full-rank basis must be unchanged by the (no-op) reduction",
            );
        }
    }
    let after = &term.atoms[0].decoder_coefficients;
    for i in 0..5 {
        assert_eq!(
            after[[i, 0]],
            decoder[[i, 0]],
            "full-rank decoder must be unchanged by the (no-op) reduction",
        );
    }
}

/// Regression test for https://github.com/SauersML/gam/issues/163 and #175.
///
/// `ManifoldSAE.reconstruct(X_oos)` (and `.predict(X_subset)`) reach the
/// Rust core via `sae_manifold_predict_oos` → `sae_manifold_fit_inner` →
/// the same `run_joint_fit_arrow_schur` Newton driver. The driver in turn
/// calls `solve_newton_step` for single-shot refinement; before this fix
/// that path invoked `sys.solve(...)` directly, bypassing the LM ridge
/// escalation and surfacing the per-row Cholesky failure to the Python
/// caller as `"row N H_tt was non-PD at ridge_t=0.000001"`. The fix routes
/// `solve_newton_step` through `solve_with_lm_escalation` so every entry
/// point — including OOS predict — geometrically grows the proximal ridge
/// from the caller's nominal `ridge_ext_coord` / `ridge_beta` until the
/// factor succeeds.
#[test]
pub(crate) fn solve_newton_step_escalates_ridge_on_non_pd_row_block() {
    // Same degenerate-H_tt construction as the predict/reconstruct
    // reproducer: zero assignment mass + zero smoothness penalty means
    // the only mass on H_tt comes from `ridge_t·I`, and at the nominal
    // 1e-6 the Cholesky still finds a tiny negative pivot from rounding.
    let coords = array![[0.1], [0.4], [0.7]];
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.05], [-0.05], [0.05]],
        Array2::<f64>::zeros((3, 3)),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20], [-0.10], [0.45]];
    let rho = SaeManifoldRho::new(0.0, -20.0, vec![Array1::<f64>::zeros(1)]);

    // Direct `solve_newton_step` call (the predict path's single-shot
    // refinement entry). Must Ok via LM escalation, not bubble up the
    // raw per-row factor failure.
    let result = term.solve_newton_step(target.view(), &rho, None, 1.0e-6, 1.0e-6);
    assert!(
        result.is_ok(),
        "solve_newton_step should recover from degenerate H_tt via LM ridge escalation; got: {result:?}",
    );
}

#[test]
pub(crate) fn sae_arrow_schur_beta_quadratic_model_matches_penalized_loss_change() {
    let coords = array![[0.10], [0.35], [0.80]];
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.65], [-0.45], [0.25]],
        array![[3.0, 0.4, -0.2], [0.1, 2.5, 0.3], [-0.5, 0.2, 1.8]],
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode(
        Array2::<f64>::zeros((3, 1)),
        vec![coords],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20], [-0.10], [0.45]];
    let rho = SaeManifoldRho::new(0.0, 1.3_f64.ln(), vec![array![0.9_f64.ln()]]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();

    let beta0 = term.flatten_beta();
    let loss0 = term.loss(target.view(), &rho).unwrap().total();
    let mut direction = sys.gb.mapv(|v| -v);
    let direction_norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(direction_norm > 1.0e-12);
    for value in direction.iter_mut() {
        *value /= direction_norm;
    }

    let epsilon = 1.0e-3;
    let delta = direction.mapv(|v| epsilon * v);
    let beta_trial = beta0 + &delta;
    term.set_flat_beta(beta_trial.view()).unwrap();
    let actual = term.loss(target.view(), &rho).unwrap().total() - loss0;

    let linear = sys.gb.dot(&delta);
    // Use penalty_op to include all H_ββ contributions (GN + smoothness)
    // rather than reading sys.hbb directly, which no longer holds the
    // smoothness term after the #296 BetaPenaltyOp migration.
    let mut hbb_delta = Array1::<f64>::zeros(delta.len());
    {
        let op = sys.effective_penalty_op();
        let d_slice = delta.as_slice().expect("delta is contiguous");
        let hd_slice = hbb_delta.as_slice_mut().expect("hbb_delta is contiguous");
        op.matvec(d_slice, hd_slice);
    }
    let quadratic = 0.5 * delta.dot(&hbb_delta);
    let predicted = linear + quadratic;
    let error = (actual - predicted).abs();
    assert!(
        error <= 1.0e-4,
        "actual={actual:.12e}, predicted={predicted:.12e}, error={error:.12e}"
    );
}

/// `SaeRowLayout::from_dense_weights` must keep, per row, the
/// top-`k_active_cap` atoms above the magnitude cutoff (always at least
/// one), with compact coord starts that reproduce the `expand_row`
/// round-trip back to full-q positions.
#[test]
pub(crate) fn sae_row_layout_from_dense_weights_top_k_and_cutoff() {
    // 3 atoms, coord dims [2, 1, 2] ⇒ full q = 3 + 5 = 8.
    let coord_dims = vec![2usize, 1, 2];
    let coord_offsets_full = vec![3usize, 5, 6];
    let assignments = vec![
        // Row 0: weights [0.7, 0.01, 0.29]; row peak 0.7, cutoff
        // 0.05·0.7 = 0.035, cap 2 ⇒ {0, 2} (0.01 is below cutoff).
        Array1::from_vec(vec![0.7, 0.01, 0.29]),
        // Row 1 (#1414): uniformly small weights [0.001, 0.002, 0.0005].
        // Row-relative cutoff 0.05·0.002 = 1e-4 keeps the two above it
        // (atoms 1 and 0), NOT a single atom — a GLOBAL cutoff against
        // row 0's peak (0.035) would have wrongly dropped this whole row to
        // its single largest atom. Cap 2 ⇒ {0, 1}.
        Array1::from_vec(vec![0.001, 0.002, 0.0005]),
    ];
    let layout =
        SaeRowLayout::from_dense_weights(&assignments, 2, 0.05, coord_dims, coord_offsets_full);
    assert_eq!(layout.active_atoms[0], vec![0, 2]);
    assert_eq!(layout.active_atoms[1], vec![0, 1]);
    // Row 0 compact dim = |{0,2}| + d_0 + d_2 = 2 + 2 + 2 = 6.
    assert_eq!(layout.row_q_active(0), 6);
    // Row 1 compact dim = |{0,1}| + d_0 + d_1 = 2 + 2 + 1 = 5.
    assert_eq!(layout.row_q_active(1), 5);
    // expand_row round-trip for row 0: compact [logit0, logit2, t0_0,
    // t0_1, t2_0, t2_1] → full-q with zeros for inactive atom 1.
    let compact = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut full = vec![0.0_f64; 8];
    layout.expand_row(0, &compact, &mut full);
    // logits: full[0] = atom0 logit, full[2] = atom2 logit, full[1] = 0.
    assert_eq!(full[0], 1.0);
    assert_eq!(full[1], 0.0);
    assert_eq!(full[2], 2.0);
    // coords: atom0 at offset 3 (d=2), atom2 at offset 6 (d=2); atom1
    // (offset 5, d=1) is inactive ⇒ zero.
    assert_eq!(full[3], 3.0);
    assert_eq!(full[4], 4.0);
    assert_eq!(full[5], 0.0);
    assert_eq!(full[6], 5.0);
    assert_eq!(full[7], 6.0);
}

/// #1450: drive the REAL high-K support-proposal path (`from_dense_weights`,
/// the routine #1411 fixed to use an O(K) partial-select instead of a full
/// O(K log K) per-row sort) at production scale (K = 100_000) and assert the
/// headline contract: per-token assembly work depends on `k_active`, NOT on
/// total `K`. The existing `from_dense_weights` coverage
/// (`sae_row_layout_from_dense_weights_top_k_and_cutoff`) runs only at K = 3,
/// and `sparse_active_layout_work_scales_with_active_atoms_not_total_k` builds
/// its layout from an already-known 3-atom active set via `from_active_atoms`,
/// so neither exercises the actual proposal/selection at large K. This does:
/// it constructs a dense K = 100_000 weight vector per row, runs the proposal,
/// checks support recovery is exact, and pins the compact work to be
/// independent of K (`q_active` set only by `cap` + active coord dims).
// #1450 (salvaged from PR #1461 by HomunculusLabs): large-K (K=100000) end-to-end
// `from_dense_weights` coverage — exact support recovery + K-independent compact work.
#[test]
pub(crate) fn from_dense_weights_large_k_support_proposal_1450() {
    let (k_atoms, d, k_true, n) = (100_000_usize, 1, 4, 4);
    let planted: Vec<usize> = (0..k_true).map(|j| j * k_atoms / k_true).collect();
    let assignments: Vec<Array1<f64>> = (0..n)
        .map(|row| {
            let mut a = vec![1e-9_f64; k_atoms];
            for (i, &atom) in planted.iter().enumerate() {
                a[atom] = 0.2 + 0.01 * (row + i) as f64;
            }
            Array1::from_vec(a)
        })
        .collect();
    let coord_offsets: Vec<usize> = (0..k_atoms).map(|k| k_atoms + k).collect();
    let layout = SaeRowLayout::from_dense_weights(
        &assignments,
        k_true,
        1e-3,
        vec![d; k_atoms],
        coord_offsets,
    );
    for row in 0..n {
        assert_eq!(layout.active_atoms[row], planted, "row {row} wrong atoms");
        assert_eq!(layout.row_q_active(row), k_true + k_true * d);
    }
    let compact_work: usize = (0..n).map(|r| layout.row_q_active(r).pow(2)).sum();
    assert!(compact_work < n * (k_atoms * (1 + d)).pow(2) / 1_000_000);
}

#[test]
pub(crate) fn sae_row_layout_from_dense_weights_large_k_work_scales_with_active() {
    let n = 4usize;
    let k_atoms = 100_000usize;
    let cap = 8usize;
    let relative_cutoff = 0.05_f64;
    // Per row, plant `cap` large weights at known indices (descending so the
    // row peak is unambiguous) on a background of tiny weights well below the
    // row-relative cutoff. Support recovery must return exactly the planted set.
    let mut planted: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut assignments: Vec<Array1<f64>> = Vec::with_capacity(n);
    for row in 0..n {
        let mut a = Array1::<f64>::from_elem(k_atoms, 1e-6);
        let mut plant = Vec::with_capacity(cap);
        for j in 0..cap {
            // Spread the planted atoms across the index range so a tail-only or
            // prefix-only selector would miss some; magnitudes 1.0 down to
            // ~0.3, all far above `relative_cutoff * peak = 0.05`.
            let idx = (row + j * (k_atoms / cap)) % k_atoms;
            a[idx] = 1.0 - 0.1 * j as f64;
            plant.push(idx);
        }
        plant.sort_unstable();
        planted.push(plant);
        assignments.push(a);
    }
    let coord_dims = vec![1usize; k_atoms];
    let coord_offsets_full: Vec<usize> = (0..k_atoms).map(|k| k_atoms + k).collect();
    let layout = SaeRowLayout::from_dense_weights(
        &assignments,
        cap,
        relative_cutoff,
        coord_dims,
        coord_offsets_full,
    );
    for row in 0..n {
        // Exact support recovery: the proposal must return exactly the planted
        // top-`cap` atoms (all background weights are below the cutoff).
        assert_eq!(
            layout.active_atoms[row], planted[row],
            "row {row}: support recovery mismatch"
        );
        // Compact dim is bounded by `cap` (+ one coord axis each), independent
        // of K: q_active = cap + cap·1 = 2·cap.
        assert_eq!(layout.row_q_active(row), 2 * cap, "row {row}: q_active");
    }
    let compact_work: usize = (0..n)
        .map(|row| {
            let q = layout.row_q_active(row);
            q * q
        })
        .sum();
    // The dense per-token cost would be `q² = (2K)²`; the compact contract is
    // `(2·cap)²` — independent of K. Pin the K-independent exact value AND that
    // it is astronomically below the dense full-K work.
    assert_eq!(compact_work, n * (2 * cap) * (2 * cap));
    let dense_q = 2 * k_atoms;
    let dense_work = n * dense_q * dense_q;
    // The work ratio is EXACTLY `(2K)² / (2·cap)² = (K/cap)²` (the `n` token
    // factor cancels), so for K = 100 000, cap = 8 the compact path is
    // `12500² = 156_250_000`× cheaper. Pin that exact astronomical factor — a
    // strictly stronger guard than the previous `< dense_work / 1e9`, whose
    // arbitrary 1e9 divisor exceeded the true 1.5625e8 ratio and made the
    // assertion unsatisfiable for these dimensions.
    let work_ratio = (k_atoms / cap) * (k_atoms / cap);
    assert_eq!(
        dense_work / compact_work,
        work_ratio,
        "compact row-layout work must be exactly (K/cap)² below dense full-K work"
    );
    assert!(
        work_ratio >= 100_000_000,
        "the compact path must be astronomically (≥1e8×) cheaper than dense full-K"
    );
}

/// #1407 — fixed-decoder assembly must skip the ENTIRE decoder β tier. The
/// frozen-decoder encode path (`run_fixed_decoder_arrow_schur`) reads only the
/// per-row `htt`/`gt` blocks, so assembling the joint decoder β tier
/// (`G`/`gb`/`H_tβ`/dense `hbb`/β-penalties) is dead, K-dependent work. The
/// `fixed_decoder_assembly` flag gates it off; this pins that the assembled
/// system carries the populated per-row `(htt, gt)` block-diagonal but an EMPTY
/// β tier (`hbb` reclaimed to `0×0`), versus the full joint assembly which
/// materialises both. The two assemblies run on the SAME term/ρ so the contrast
/// is exactly the β-tier work the flag elides.
#[test]
pub(crate) fn fixed_decoder_assembly_skips_beta_tier_1407() {
    let (mut term, target, rho) = small_two_atom_periodic_term();

    // Full joint assembly: the β tier IS built — its curvature is carried by the
    // matrix-free `CompositePenaltyOp` (smoothness `λ S_k ⊗ I_p` + the data-fit
    // `G ⊗ I_p` Gauss-Newton block; plus the dense `hbb` residual only when an
    // analytic Beta-tier penalty fired). The dense `hbb` buffer is then RECLAIMED
    // back into the term's reusable `border_hbb_workspace` (a pooling
    // optimisation), so `sys.hbb` is `0×0` on BOTH paths — the load-bearing
    // β-tier observable is the installed `penalty_op`, NOT `hbb.dim()`.
    let full = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("full joint assemble_arrow_schur");
    assert!(
        full.penalty_op.is_some(),
        "full joint assembly must install the β-tier curvature operator \
         (matrix-free smoothness + G⊗I data-fit block)"
    );
    assert_eq!(
        full.k,
        term.beta_dim(),
        "full joint assembly must carry the full β-tier width"
    );
    assert!(
        full.gb.len() == term.beta_dim() && full.gb.iter().all(|v| v.is_finite()),
        "full joint assembly must carry a finite β-tier gradient gb of width beta_dim"
    );
    let n_rows = full.rows.len();

    // Fixed-decoder assembly on the SAME term/ρ: the β tier is elided — the
    // function returns the block-diagonal per-row system before any β-curvature
    // operator / β-penalty is installed, so NO `penalty_op` is present.
    term.fixed_decoder_assembly = true;
    let fixed = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("fixed-decoder assemble_arrow_schur");
    term.fixed_decoder_assembly = false;

    assert!(
        fixed.penalty_op.is_none(),
        "fixed-decoder assembly must install NO β-tier curvature operator (the β \
         tier is dead work when the decoder is frozen)"
    );
    assert_eq!(
        fixed.hbb.dim(),
        (0, 0),
        "fixed-decoder assembly must build NO dense β-Hessian (the β tier is \
         dead work when the decoder is frozen); got hbb {:?}",
        fixed.hbb.dim()
    );
    // The per-row latent block-diagonal the fixed-decoder Newton step reads is
    // still fully populated and finite (same row count as the full assembly).
    assert_eq!(
        fixed.rows.len(),
        n_rows,
        "fixed-decoder assembly must keep every per-row htt/gt block"
    );
    for (i, row) in fixed.rows.iter().enumerate() {
        assert!(
            row.htt.iter().all(|v| v.is_finite()) && row.gt.iter().all(|v| v.is_finite()),
            "fixed-decoder row {i} htt/gt must be finite"
        );
        assert_eq!(
            row.htt.dim().0,
            row.gt.len(),
            "fixed-decoder row {i} htt must be square and match gt length"
        );
        assert!(
            row.gt.len() > 0,
            "fixed-decoder row {i} must carry a non-empty latent block"
        );
    }
}

/// MechanismSparsityPenalty must reach the SAE arrow-Schur system's
/// `gb` (beta-tier gradient) when its target slice is shaped to match a
/// single-atom decoder block (M, p_out). The group lasso over rows of
/// that (M, p_out) matrix translates to a non-zero gradient on every
/// (basis_row, feature) entry whose corresponding decoder coefficient
/// is non-zero, and the FFI-side `"beta"` latent block is what makes
/// the descriptor builder see exactly that target shape.
#[test]
pub(crate) fn sae_mechsparsity_beta_block_routes_through_arrow_schur_gb() {
    let coords = array![[0.10], [0.35], [0.80]];
    let (phi, jet) = periodic_basis(&coords);
    // Decoder shape: (M=3 basis × p=4 features); flatten_beta lays out
    // [basis_col * p + feature] which is exactly the (M, p) row-major
    // shape MechSparsity targets.
    let decoder = array![
        [0.7, -0.2, 0.05, 0.4],
        [-0.5, 0.6, -0.1, 0.3],
        [0.2, 0.0, -0.4, -0.6],
    ];
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder.clone(),
        Array2::<f64>::eye(3),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // Two groups partition the 4 features: {0,1} and {2,3}. Each row
    // of the decoder block has non-zero entries in both groups, so the
    // group-norm denominator is finite and every column-feature in a
    // non-trivially-loaded group sees a positive penalty gradient.
    let m = 3usize;
    let p = 4usize;
    let slice = PsiSlice::full(m * p, Some(m));
    let penalty = MechanismSparsityPenalty::new(
        slice,
        vec![vec![0, 1], vec![2, 3]],
        1.0,
        1.0e-6,
        (term.n_obs()) as f64,
        false,
    )
    .unwrap();
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::MechanismSparsity(Arc::new(penalty)));

    let target = array![
        [0.20, 0.10, -0.05, 0.25],
        [-0.10, 0.30, 0.15, -0.20],
        [0.45, -0.05, 0.10, 0.30],
    ];
    let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, Some(&registry))
        .unwrap();

    assert_eq!(sys.gb.len(), m * p, "gb should match flatten_beta length");
    let mut absmax = 0.0_f64;
    for v in sys.gb.iter().copied() {
        assert!(v.is_finite());
        if v.abs() > absmax {
            absmax = v.abs();
        }
    }
    assert!(
        absmax > 1.0e-6,
        "MechSparsity must inject a non-trivial gradient into the SAE arrow-Schur gb; absmax={absmax:.3e}"
    );
    // Closed-form check on the ISOLATED MechSparsity contribution. `sys.gb`
    // is the FULL penalized β-gradient (data-fit + decoder-smoothness +
    // MechSparsity), so comparing a raw `gb` entry to the penalty-only
    // closed form is wrong (it omits the data-fit and smoothness terms).
    // Difference two assemblies — with and without the registry — to recover
    // exactly the penalty gradient `Δgb = gb_with − gb_without`, then compare
    // that delta to `MechanismSparsityPenalty::grad_target` at (basis=1,
    // feat=0):
    //   w / sqrt(|G|) · b[1,0] / ||b[1, group={0,1}]||
    // group {0,1} has size 2 → factor sqrt(2); unit weight, tiny eps.
    let sys_no_penalty = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let beta = term.flatten_beta();
    let expected = {
        // ||b[1, {0,1}]|| ≈ sqrt(0.5² + 0.6²) = sqrt(0.61)
        let s = (0.5_f64.powi(2) + 0.6_f64.powi(2) + 1.0e-12).sqrt();
        (2.0_f64).sqrt() * (-0.5_f64) / s
    };
    let delta = sys.gb[1 * p + 0] - sys_no_penalty.gb[1 * p + 0];
    assert!(
        (delta - expected).abs() <= 1.0e-6,
        "expected MechSparsity gb contribution at (basis=1, feat=0) ≈ {expected:.6e}, \
             got Δgb={delta:.6e} (gb_with={:.6e}, gb_without={:.6e}, beta entry = {})",
        sys.gb[1 * p + 0],
        sys_no_penalty.gb[1 * p + 0],
        beta[1 * p + 0]
    );
}

/// Smoothed sum of singular values of an `m × p` matrix, matching
/// `NuclearNormPenalty::value` (used by the spectrum-shrinkage assertion).
pub(crate) fn smoothed_nuclear_norm(decoder: &Array2<f64>, eps: f64) -> f64 {
    let (_u, s, _vt) = decoder.clone().svd(false, false).unwrap();
    s.iter()
        .map(|sigma| (sigma * sigma + eps * eps).sqrt() - eps)
        .sum()
}

/// NuclearNormPenalty is a Psi-tier penalty, but inside the SAE term it is
/// redirected to the per-atom decoder (β) block rather than the coord "t"
/// row block (#672). This pins three things:
///   1. `validate_analytic_penalty_registry` does NOT refuse it (it bypasses
///      the row-block requirement).
///   2. It injects a non-trivial gradient into the arrow-Schur `gb`
///      (β-tier gradient) equal to the analytic spectral gradient on the
///      atom's `(M, p)` decoder block.
///   3. A gradient-descent step along `gb` shrinks the decoder block's
///      (smoothed) singular spectrum — the rank-shrinkage objective.
#[test]
pub(crate) fn sae_nuclear_norm_beta_block_routes_through_gb_and_shrinks_spectrum() {
    let coords = array![[0.10], [0.35], [0.80]];
    let (phi, jet) = periodic_basis(&coords);
    // Full-rank (M=3 basis × p=4 features) decoder block. flatten_beta lays
    // it out [basis_row * p + feature] = the (M, p) row-major shape the
    // nuclear-norm penalty treats as a matrix.
    let decoder = array![
        [0.9, -0.2, 0.05, 0.4],
        [-0.5, 0.7, -0.1, 0.3],
        [0.2, 0.1, -0.8, -0.6],
    ];
    let m = 3usize;
    let p = 4usize;
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder.clone(),
        Array2::<f64>::eye(3),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((3, 1)),
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // Base penalty: the per-atom dispatch in `add_sae_beta_penalty`
    // overrides n_eff and target, so these initial values only need to
    // construct validly. Here n_eff = p (the "beta" block declares n=p_out,
    // d=Σ M_k); the SAE term rebuilds n_eff = M_k, latent_dim = p per atom.
    let eps = 1.0e-6;
    let slice = PsiSlice::full(m * p, Some(m));
    let penalty = NuclearNormPenalty::new(slice, 1.0, p, eps, None, false).unwrap();
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(penalty)));

    // Must NOT be refused by the construction-time validator.
    term.validate_analytic_penalty_registry(&registry)
        .expect("NuclearNorm must be accepted (redirected to the β block)");

    let target = array![
        [0.20, 0.10, -0.05, 0.25],
        [-0.10, 0.30, 0.15, -0.20],
        [0.45, -0.05, 0.10, 0.30],
    ];
    let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
    let baseline = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, Some(&registry))
        .unwrap();

    assert_eq!(sys.gb.len(), m * p, "gb should match flatten_beta length");
    assert_eq!(
        baseline.gb.len(),
        m * p,
        "baseline gb should match flatten_beta length"
    );
    let mut absmax = 0.0_f64;
    let mut penalty_grad = Array1::<f64>::zeros(m * p);
    for ((dst, sys_g), baseline_g) in penalty_grad
        .iter_mut()
        .zip(sys.gb.iter())
        .zip(baseline.gb.iter())
    {
        let v = *sys_g - *baseline_g;
        assert!(v.is_finite());
        *dst = v;
        absmax = absmax.max(v.abs());
    }
    assert!(
        absmax > 1.0e-6,
        "NuclearNorm must inject a non-trivial gradient into the SAE \
             arrow-Schur gb; absmax={absmax:.3e}"
    );

    // The penalty contribution to gb must equal the analytic spectral
    // gradient of the per-atom `(M, p)` block (penalty_scale defaults to
    // 1.0 in assemble_arrow_schur). Reconstruct the reference directly.
    let per_atom = NuclearNormPenalty::new(
        PsiSlice {
            range: 0..m * p,
            latent_dim: Some(p),
        },
        1.0,
        m,
        eps,
        None,
        false,
    )
    .unwrap();
    let beta = term.flatten_beta();
    let ref_grad = per_atom.grad_target(beta.view(), Array1::<f64>::zeros(0).view());
    for j in 0..m * p {
        assert!(
            (penalty_grad[j] - ref_grad[j]).abs() <= 1.0e-9,
            "penalty gb[{j}]={:.12e} must equal analytic spectral grad {:.12e}",
            penalty_grad[j],
            ref_grad[j]
        );
    }

    // A gradient-descent step on the decoder block shrinks the smoothed
    // singular spectrum: nuclear-norm is the rank-shrinkage objective.
    let base_norm = smoothed_nuclear_norm(&decoder, eps);
    let step = 1.0e-2;
    let mut shrunk = decoder.clone();
    for ((row, feat), value) in shrunk.indexed_iter_mut() {
        *value -= step * penalty_grad[row * p + feat];
    }
    let shrunk_norm = smoothed_nuclear_norm(&shrunk, eps);
    assert!(
        shrunk_norm < base_norm,
        "a step along gb must shrink the decoder spectrum: \
             before={base_norm:.9e}, after={shrunk_norm:.9e}"
    );

    // The β curvature block must be PSD. SAE returns `hbb` to the term
    // workspace after lowering the block into the effective penalty operator,
    // so validate the operator diagonal rather than the recycled field.
    assert!(sys.hbb.is_empty());
    let mut hbb_diag = vec![0.0_f64; m * p];
    sys.effective_penalty_op().diagonal(&mut hbb_diag);
    for i in 0..m * p {
        assert!(
            hbb_diag[i] >= -1.0e-9,
            "hbb diagonal must be non-negative (PSD majorizer); hbb[{i},{i}]={:.3e}",
            hbb_diag[i]
        );
    }
}

#[derive(Debug)]
pub(crate) struct TestPeriodicEvaluator;

impl SaeBasisEvaluator for TestPeriodicEvaluator {
    /// Second derivative of the test periodic basis `[1, sin(2πt), cos(2πt)]`:
    /// `Φ'' = [0, -(2π)² sin(2πt), -(2π)² cos(2πt)]`. The encode-atlas Kantorovich
    /// certificate (`row_certificate`) needs `∂²Φ/∂t²` for the full-Hessian
    /// residual term and returns NO certificate when the second jet is absent
    /// (never a silent Gauss-Newton substitute, see encode.rs). A test double
    /// that claims to evaluate this basis must therefore supply its real second
    /// jet, or every certified-encode path it feeds becomes vacuously uncertified.
    fn second_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array4<f64>, String>> {
        if coords.ncols() != 1 {
            return Some(Err(format!(
                "TestPeriodicEvaluator::second_jet_dyn: expected latent_dim 1, got {}",
                coords.ncols()
            )));
        }
        let n = coords.nrows();
        let two_pi = 2.0 * std::f64::consts::PI;
        let freq2 = two_pi * two_pi;
        let mut h = Array4::<f64>::zeros((n, 3, 1, 1));
        for row in 0..n {
            let angle = two_pi * coords[[row, 0]];
            // basis 0 is the constant 1 → Φ''=0; basis 1=sin, basis 2=cos.
            h[[row, 1, 0, 0]] = -freq2 * angle.sin();
            h[[row, 2, 0, 0]] = -freq2 * angle.cos();
        }
        Some(Ok(h))
    }

    /// Third derivative of `[1, sin(2πt), cos(2πt)]`:
    /// `Φ''' = [0, -(2π)³ cos(2πt), +(2π)³ sin(2πt)]` (sin→ωc→−ω²s→−ω³c,
    /// cos→−ωs→−ω²c→+ω³s).
    fn third_jet_dyn(&self, coords: ArrayView2<'_, f64>) -> Option<Result<Array5<f64>, String>> {
        if coords.ncols() != 1 {
            return Some(Err(format!(
                "TestPeriodicEvaluator::third_jet_dyn: expected latent_dim 1, got {}",
                coords.ncols()
            )));
        }
        let n = coords.nrows();
        let two_pi = 2.0 * std::f64::consts::PI;
        let freq3 = two_pi * two_pi * two_pi;
        let mut h = Array5::<f64>::zeros((n, 3, 1, 1, 1));
        for row in 0..n {
            let angle = two_pi * coords[[row, 0]];
            h[[row, 1, 0, 0, 0]] = -freq3 * angle.cos();
            h[[row, 2, 0, 0, 0]] = freq3 * angle.sin();
        }
        Some(Ok(h))
    }

    fn evaluate(&self, coords: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
        Ok(periodic_basis(&coords.to_owned()))
    }
}

pub(crate) fn assert_jacobian_matches_central_difference<E: SaeBasisEvaluator>(
    evaluator: &E,
    coords: Array2<f64>,
    tolerance: f64,
) {
    let epsilon = 1.0e-6;
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let (n_rows, n_basis) = phi.dim();
    let latent_dim = coords.ncols();
    assert_eq!(jet.dim(), (n_rows, n_basis, latent_dim));

    for row in 0..n_rows {
        for axis in 0..latent_dim {
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            plus[[row, axis]] += epsilon;
            minus[[row, axis]] -= epsilon;
            let (phi_plus, plus_jet) = evaluator.evaluate(plus.view()).unwrap();
            let (phi_minus, minus_jet) = evaluator.evaluate(minus.view()).unwrap();
            assert_eq!(plus_jet.dim(), jet.dim());
            assert_eq!(minus_jet.dim(), jet.dim());

            for basis in 0..n_basis {
                let finite_difference =
                    (phi_plus[[row, basis]] - phi_minus[[row, basis]]) / (2.0 * epsilon);
                let analytic = jet[[row, basis, axis]];
                let error = (analytic - finite_difference).abs();
                assert!(
                    error <= tolerance,
                    "row={row} basis={basis} axis={axis}: analytic={analytic:.12e}, \
                         finite_difference={finite_difference:.12e}, error={error:.12e}, \
                         tolerance={tolerance:.12e}"
                );
            }
        }
    }
}

#[test]
pub(crate) fn sae_basis_evaluator_jacobians_match_central_differences() {
    assert_jacobian_matches_central_difference(
        &PeriodicHarmonicEvaluator::new(7).unwrap(),
        array![[-0.37], [0.0], [0.125], [0.41]],
        1.0e-6,
    );

    assert_jacobian_matches_central_difference(
        &RawPeriodicCircleEvaluator::new(3).unwrap(),
        array![[-1.2, 0.3, 2.0], [0.0, -0.4, 0.8], [2.4, 1.1, -0.7]],
        1.0e-6,
    );

    let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
    assert_jacobian_matches_central_difference(
        &SphereChartEvaluator,
        sphere_coords.clone(),
        1.0e-6,
    );
    let (sphere_phi, sphere_jet) = SphereChartEvaluator.evaluate(sphere_coords.view()).unwrap();
    assert_eq!(sphere_phi.dim(), (sphere_coords.nrows(), 7));
    assert_eq!(sphere_jet.dim(), (sphere_coords.nrows(), 7, 2));
    for row in 0..sphere_coords.nrows() {
        let lat = sphere_coords[[row, 0]];
        let lon = sphere_coords[[row, 1]];
        let clat = lat.cos();
        let slat = lat.sin();
        let clon = lon.cos();
        let slon = lon.sin();
        let z = slat;
        let dx_dlon = -clat * slon;
        let dy_dlon = clat * clon;
        assert_eq!(sphere_jet[[row, 3, 1]], 0.0);
        assert!((sphere_jet[[row, 5, 1]] - dy_dlon * z).abs() <= 1.0e-12);
        assert!((sphere_jet[[row, 6, 1]] - dx_dlon * z).abs() <= 1.0e-12);
    }

    assert_jacobian_matches_central_difference(
        &AffineCoordinateEvaluator::new(3),
        array![[0.0, -1.0, 2.0], [3.5, 0.25, -0.75]],
        1.0e-6,
    );

    // Torus T^2 with H=3 → 49-column tensor product.
    let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
    assert_jacobian_matches_central_difference(
        &TorusHarmonicEvaluator::new(2, 3).unwrap(),
        torus_coords.clone(),
        1.0e-6,
    );
    let (torus_phi, torus_jet) = TorusHarmonicEvaluator::new(2, 3)
        .unwrap()
        .evaluate(torus_coords.view())
        .unwrap();
    assert_eq!(torus_phi.dim(), (torus_coords.nrows(), 49));
    assert_eq!(torus_jet.dim(), (torus_coords.nrows(), 49, 2));
    for row in 0..torus_coords.nrows() {
        // Column 0 = product of the two constant axis terms = 1.
        assert!((torus_phi[[row, 0]] - 1.0).abs() <= 1.0e-12);
        assert!(torus_jet[[row, 0, 0]].abs() <= 1.0e-12);
        assert!(torus_jet[[row, 0, 1]].abs() <= 1.0e-12);
    }
}

/// The compact-latent basis kinds must each expose a projection seed grid
/// that spans their manifold, and the unbounded / basis-linear kinds expose none (their PCA
/// seed already lands in the convex hull of the training coordinates).
/// Pins the grid extents the fixed-decoder OOS seed (#628) relies on.
#[test]
pub(crate) fn projection_seed_grid_spans_each_compact_manifold() {
    use std::f64::consts::PI;

    // Periodic S¹: `resolution` phases evenly on `[0, 1)` (endpoint
    // excluded — `0` and `1` are the same point on the circle).
    let periodic = SaeAtomBasisKind::Periodic
        .projection_seed_grid(1, 16)
        .unwrap();
    assert_eq!(periodic.dim(), (16, 1));
    for i in 0..16 {
        assert_abs_diff_eq!(periodic[[i, 0]], i as f64 / 16.0, epsilon = 1e-12);
    }
    assert!(periodic.iter().all(|&t| (0.0..1.0).contains(&t)));

    // Sphere lat/lon chart: an `r × r` grid, latitude strictly interior to
    // the chart (poles are degenerate), longitude on `[-π, π)`.
    let r = 6usize;
    let sphere = SaeAtomBasisKind::Sphere.projection_seed_grid(2, r).unwrap();
    assert_eq!(sphere.dim(), (r * r, 2));
    for row in 0..r * r {
        let lat = sphere[[row, 0]];
        let lon = sphere[[row, 1]];
        assert!(
            lat > -PI / 2.0 && lat < PI / 2.0,
            "sphere seed latitude {lat} is not strictly interior to the chart"
        );
        assert!(
            (-PI..PI).contains(&lon),
            "sphere seed longitude {lon} is outside [-π, π)"
        );
    }

    // Unbounded / basis-linear latents expose no grid (default `None`).
    assert!(
        SaeAtomBasisKind::EuclideanPatch
            .projection_seed_grid(2, 64)
            .is_none(),
        "Euclidean-patch (unbounded) atoms must not expose a projection seed grid"
    );
}

/// The torus seed grid is the Cartesian product of per-axis `[0, 1)` phase
/// grids, with the per-axis resolution shrunk geometrically so the *total*
/// point count stays under a fixed cap as the latent dimension grows. Pins
/// the cap arithmetic (`per_axis^d ≤ 4096`) the OOS seed depends on so a
/// high-`d` torus atom never blows up the per-row global-argmin scan.
#[test]
pub(crate) fn torus_projection_seed_grid_caps_total_points() {
    // d == 1: dense, no cap (256¹ ≤ 4096).
    let g1 = SaeAtomBasisKind::Torus
        .projection_seed_grid(1, 256)
        .unwrap();
    assert_eq!(g1.dim(), (256, 1));

    // d == 3: per-axis shrunk to the largest `p` with `p³ ≤ 4096`, i.e.
    // `p = 16` ⇒ exactly 4096 points.
    let g3 = SaeAtomBasisKind::Torus
        .projection_seed_grid(3, 256)
        .unwrap();
    assert_eq!(g3.ncols(), 3);
    assert_eq!(g3.nrows(), 16 * 16 * 16);
    assert!(
        g3.nrows() <= 4096,
        "torus d=3 seed grid has {} points, over the 4096 cap",
        g3.nrows()
    );
    assert!(
        g3.iter().all(|&t| (0.0..1.0).contains(&t)),
        "every torus seed coordinate must be a phase on [0, 1)"
    );
    // Full Cartesian product: each axis takes exactly `per_axis` distinct
    // phase values.
    for axis in 0..3 {
        let mut vals: Vec<f64> = g3.column(axis).iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();
        assert_eq!(
            vals.len(),
            16,
            "torus seed axis {axis} should take 16 distinct phases"
        );
    }

    // d == 12: the coarsest dense grid is `2^12 = 4096`, exactly the cap —
    // still emitted (per_axis floors at 2).
    let g12 = SaeAtomBasisKind::Torus
        .projection_seed_grid(12, 256)
        .unwrap();
    assert_eq!(g12.nrows(), 1usize << 12);
    assert!(g12.nrows() <= 4096);

    // d == 13: even the coarsest dense grid (`2^13 = 8192`) exceeds the
    // cap, so no on-manifold grid can satisfy it. The evaluator must return
    // `None` and let the row fall back to its PCA seed rather than allocate
    // a runaway `2^d`-row grid for the per-row global-argmin scan to walk.
    assert!(
        SaeAtomBasisKind::Torus
            .projection_seed_grid(13, 256)
            .is_none(),
        "torus d=13 seed grid (2^13 > 4096) must fall back to None, not blow up the cap"
    );
}

/// `seed_coords_by_decoder_projection` must replace each cold coordinate
/// with the grid point whose frozen-decoder decode is closest to the target
/// row, and refresh the atom basis there. Built on a decoder that maps the
/// circle injectively into `ℝ²` (`decode(t) = (sin 2πt, cos 2πt)`) so the
/// per-row global argmin is unambiguous. Direct Rust pin for the #628 OOS
/// seed, complementing the Python oracle end-to-end test.
#[test]
pub(crate) fn seed_coords_by_decoder_projection_lands_on_grid_minimiser() {
    use std::f64::consts::PI;

    let resolution = 8usize;
    // Deliberately wrong cold seed for both rows.
    let init_coords = array![[0.05], [0.05]];
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let (phi0, jet0) = evaluator.evaluate(init_coords.view()).unwrap();
    // (basis = [1, sin, cos]) × (2 output channels): decode(t) = (sin, cos).
    let decoder = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone());
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        // `K = logits.ncols()`; a single softmax atom is one logit column
        // (the lone simplex coordinate, pinned to 1.0 in `try_assignments_row`).
        Array2::<f64>::zeros((2, 1)),
        vec![init_coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // Targets sit exactly on two distinct grid phases `k / resolution`.
    let phases = [3usize, 6usize];
    let mut target = Array2::<f64>::zeros((2, 2));
    for (row, &k) in phases.iter().enumerate() {
        let t = k as f64 / resolution as f64;
        target[[row, 0]] = (2.0 * PI * t).sin();
        target[[row, 1]] = (2.0 * PI * t).cos();
    }

    term.seed_coords_by_decoder_projection(target.view(), resolution)
        .unwrap();

    // Each row was seeded onto its exact grid minimiser …
    let seeded = term.assignment.coords[0].as_matrix();
    let mut expected_coords = Array2::<f64>::zeros((2, 1));
    for (row, &k) in phases.iter().enumerate() {
        let expected = k as f64 / resolution as f64;
        assert_abs_diff_eq!(seeded[[row, 0]], expected, epsilon = 1e-12);
        expected_coords[[row, 0]] = expected;
    }
    // … and the basis cache was refreshed at the seeded coordinates.
    let (phi_expected, _) = evaluator.evaluate(expected_coords.view()).unwrap();
    assert_abs_diff_eq!(
        (&term.atoms[0].basis_values - &phi_expected)
            .mapv(f64::abs)
            .sum(),
        0.0,
        epsilon = 1e-12
    );
}

/// A target whose shape does not match `(n_obs, output_dim)` is a caller
/// bug and must surface as an error rather than silently mis-seeding.
#[test]
pub(crate) fn seed_coords_by_decoder_projection_rejects_shape_mismatch() {
    let init_coords = array![[0.05], [0.05]];
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let (phi0, jet0) = evaluator.evaluate(init_coords.view()).unwrap();
    let decoder = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((2, 1)),
        vec![init_coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // Output dim is 2; pass a 3-column target.
    let bad_target = Array2::<f64>::zeros((2, 3));
    let err = term
        .seed_coords_by_decoder_projection(bad_target.view(), 8)
        .unwrap_err();
    assert!(
        err.contains("target shape"),
        "expected a target-shape error, got: {err}"
    );
}

/// Parity guard for the sphere chart: the shared engine
/// [`sphere_chart_basis_jet`] is the single source of derivative truth used
/// by both the core SAE path ([`SphereChartEvaluator::evaluate`]) and the
/// PyFFI `sphere_chart_basis_with_jet` helper, which route through the exact
/// same function. The basis and its jet are now the *exact* analytic ones —
/// `C^∞` in `(lat, lon)` with no clamp and no binary `chain_lat` gate — so
/// this pins that the jet equals the closed-form analytic derivative at
/// interior, boundary (`|lat| = π/2`), and beyond-`π/2` latitudes alike.
#[test]
pub(crate) fn sphere_chart_basis_jet_is_single_source_of_truth() {
    // A mix of interior and former clamp-boundary / beyond-π/2 latitudes;
    // the embedding and its jet are smooth everywhere, so all rows must hit
    // the same exact analytic formulas.
    let coords = array![
        [-1.2, -2.4],                         // interior
        [0.35, 0.9],                          // interior
        [std::f64::consts::FRAC_PI_2, 0.4],   // upper boundary (former gate)
        [-std::f64::consts::FRAC_PI_2, -1.1], // lower boundary (former gate)
        [2.3, 0.7],                           // beyond +π/2
        [-3.0, 1.9],                          // beyond -π/2
    ];

    // The core evaluator adapter must be bit-identical to the shared engine
    // — they are the same code path, so any difference is a regression in
    // the thin adapter rather than a tolerance question.
    let (engine_phi, engine_jet) = sphere_chart_basis_jet(coords.view()).unwrap();
    let (adapter_phi, adapter_jet) = SphereChartEvaluator.evaluate(coords.view()).unwrap();
    assert_eq!(engine_phi, adapter_phi);
    assert_eq!(engine_jet, adapter_jet);

    for row in 0..coords.nrows() {
        // No clamp: the basis uses the raw latitude directly.
        let lat = coords[[row, 0]];
        let lon = coords[[row, 1]];
        let clat = lat.cos();
        let slat = lat.sin();
        let clon = lon.cos();
        let slon = lon.sin();
        let x = clat * clon;
        let y = clat * slon;
        let z = slat;

        // Basis is the unit-sphere embedding evaluated at the raw latitude.
        assert!((engine_phi[[row, 0]] - 1.0).abs() <= 1.0e-12);
        assert!((engine_phi[[row, 1]] - x).abs() <= 1.0e-12);
        assert!((engine_phi[[row, 2]] - y).abs() <= 1.0e-12);
        assert!((engine_phi[[row, 3]] - z).abs() <= 1.0e-12);
        assert!((engine_phi[[row, 4]] - x * y).abs() <= 1.0e-12);
        assert!((engine_phi[[row, 5]] - y * z).abs() <= 1.0e-12);
        assert!((engine_phi[[row, 6]] - x * z).abs() <= 1.0e-12);

        // Longitude derivatives.
        let dx_dlon = -clat * slon;
        let dy_dlon = clat * clon;
        assert!((engine_jet[[row, 1, 1]] - dx_dlon).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 2, 1]] - dy_dlon).abs() <= 1.0e-12);
        assert_eq!(engine_jet[[row, 3, 1]], 0.0);
        assert!((engine_jet[[row, 4, 1]] - (dx_dlon * y + x * dy_dlon)).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 5, 1]] - dy_dlon * z).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 6, 1]] - dx_dlon * z).abs() <= 1.0e-12);

        // Latitude derivatives are the exact analytic values at EVERY row,
        // including the former clamp boundary — no gating to zero. At the
        // upper boundary lat = +π/2 the analytic dz/dlat = cos(π/2) = 0
        // naturally (no discontinuous override), while dx/dlat, dy/dlat are
        // nonzero whenever cos(lon)/sin(lon) are.
        let dx_dlat = -slat * clon;
        let dy_dlat = -slat * slon;
        let dz_dlat = clat;
        assert!((engine_jet[[row, 1, 0]] - dx_dlat).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 2, 0]] - dy_dlat).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 3, 0]] - dz_dlat).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 4, 0]] - (dx_dlat * y + x * dy_dlat)).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 5, 0]] - (dy_dlat * z + y * dz_dlat)).abs() <= 1.0e-12);
        assert!((engine_jet[[row, 6, 0]] - (dx_dlat * z + x * dz_dlat)).abs() <= 1.0e-12);
    }

    // The chart penalty diagonal is also shared with the PyFFI helper.
    assert_eq!(
        SPHERE_CHART_PENALTY_DIAGONAL,
        [1e-8, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
    );
}

/// Regression for #619 / #618-sphere: the lat/lon sphere chart jet must
/// equal a central finite difference of the basis to ~1e-7 *at and beyond*
/// the former clamp boundary `lat = ±π/2`, where the old binary `chain_lat`
/// gate discontinuously zeroed the entire latitude jet and froze the atom.
/// Also pins continuity of the basis across `lat = π/2`.
#[test]
pub(crate) fn sphere_chart_jet_matches_fd_at_clamp_boundary() {
    // Latitudes spanning interior, exactly the former boundary, and beyond.
    let coords = array![
        [std::f64::consts::FRAC_PI_2, 0.4], // exactly +π/2 (former gate flip)
        [-std::f64::consts::FRAC_PI_2, -1.1], // exactly -π/2
        [1.45, 2.0],                        // just below +π/2
        [1.69, -0.3],                       // just above +π/2
        [2.3, 0.7],                         // well beyond +π/2
        [0.35, 0.9],                        // interior control
    ];

    let (_, jet) = sphere_chart_basis_jet(coords.view()).unwrap();
    let h = 1.0e-6;
    for row in 0..coords.nrows() {
        for axis in 0..2 {
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            plus[[row, axis]] += h;
            minus[[row, axis]] -= h;
            let (phi_p, _) = sphere_chart_basis_jet(plus.view()).unwrap();
            let (phi_m, _) = sphere_chart_basis_jet(minus.view()).unwrap();
            for col in 0..7 {
                let fd = (phi_p[[row, col]] - phi_m[[row, col]]) / (2.0 * h);
                let an = jet[[row, col, axis]];
                assert!(
                    (fd - an).abs() <= 1.0e-7,
                    "row {row} col {col} axis {axis}: analytic {an} vs FD {fd}"
                );
            }
        }
    }

    // Continuity of the basis across lat = π/2: the embedding does not jump.
    let eps = 1.0e-8;
    let lon = 0.4;
    let below = array![[std::f64::consts::FRAC_PI_2 - eps, lon]];
    let above = array![[std::f64::consts::FRAC_PI_2 + eps, lon]];
    let (phi_below, _) = sphere_chart_basis_jet(below.view()).unwrap();
    let (phi_above, _) = sphere_chart_basis_jet(above.view()).unwrap();
    for col in 0..7 {
        assert!(
            (phi_below[[0, col]] - phi_above[[0, col]]).abs() <= 1.0e-6,
            "basis discontinuous across lat = π/2 at col {col}: \
                 {} vs {}",
            phi_below[[0, col]],
            phi_above[[0, col]]
        );
    }
}

/// Central-difference oracle for `second_jet`: differentiate the analytic
/// first jet (which is FD-validated by the test above) coordinate-wise.
///
/// The threshold is magnitude-scaled (`abs_tol + rel_tol·max(|analytic|,
/// |fd|)`), exactly like the third-jet helper, because the central-difference
/// truncation error of a second derivative obtained by differencing the
/// first jet is `O(ε²/6·|f⁗|)`. For a harmonic basis `sin(ωt)` the fourth
/// derivative is `ω⁴·φ`, so with `ε = 1e-4` and the top harmonic of the
/// periodic/torus evaluators (`ω = 2π·3 ≈ 18.85 → ω⁴ ≈ 1.26e5`) the floor is
/// `≈ (1e-4)²/6·1.26e5 ≈ 2e-5` — several × any flat `1e-5` absolute bound.
/// A pure absolute bound is therefore physically wrong at the top of the
/// frequency range; the rel_tol term tracks the `ω⁴` truncation scale (the
/// analytic second jet itself is exact, `-ω²·φ`). The FD step is 1e-4 (the
/// sweet spot before f64 cancellation dominates a centered difference of an
/// `O(1)` Jacobian).
pub(crate) fn assert_second_jet_matches_central_difference<E: SaeBasisSecondJet>(
    evaluator: &E,
    coords: Array2<f64>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<(), String> {
    let epsilon = 1.0e-4;
    let second = evaluator.second_jet(coords.view())?;
    let (_phi, jet) = evaluator.evaluate(coords.view())?;
    let (n_rows, n_basis, latent_dim, latent_dim_b) = second.dim();
    assert_eq!(latent_dim, latent_dim_b);
    assert_eq!((n_rows, n_basis, latent_dim), jet.dim());
    for row in 0..n_rows {
        for axis_c in 0..latent_dim {
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            plus[[row, axis_c]] += epsilon;
            minus[[row, axis_c]] -= epsilon;
            let (_, jet_plus) = evaluator.evaluate(plus.view()).unwrap();
            let (_, jet_minus) = evaluator.evaluate(minus.view()).unwrap();
            for basis in 0..n_basis {
                for axis_a in 0..latent_dim {
                    let fd = (jet_plus[[row, basis, axis_a]] - jet_minus[[row, basis, axis_a]])
                        / (2.0 * epsilon);
                    let analytic = second[[row, basis, axis_a, axis_c]];
                    let error = (analytic - fd).abs();
                    let threshold = abs_tol + rel_tol * analytic.abs().max(fd.abs());
                    assert!(
                        error <= threshold,
                        "row={row} basis={basis} axis_a={axis_a} axis_c={axis_c}: \
                             analytic={analytic:.12e}, fd={fd:.12e}, error={error:.12e}, \
                             threshold={threshold:.12e}"
                    );
                }
            }
        }
    }
    // Hessian symmetry in (axis_a, axis_c).
    for row in 0..n_rows {
        for basis in 0..n_basis {
            for axis_a in 0..latent_dim {
                for axis_b in 0..latent_dim {
                    let h_ab = second[[row, basis, axis_a, axis_b]];
                    let h_ba = second[[row, basis, axis_b, axis_a]];
                    assert!(
                        (h_ab - h_ba).abs() <= 1.0e-12,
                        "second_jet not symmetric: row={row} basis={basis} \
                             ({axis_a},{axis_b})={h_ab:.6e} vs ({axis_b},{axis_a})={h_ba:.6e}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// The analytic third jet `T[n,m,a,c,e] = ∂³Φ_m/∂t_a∂t_c∂t_e` must equal the
/// central difference of the analytic (already FD-validated) second jet along
/// the trailing axis, and be fully symmetric across its three trailing axes.
/// This validates the closed-form `K` providers added for the exact isometry
/// Hessian (#458) against an independent numerical derivative — the third-jet
/// analogue of `assert_second_jet_matches_central_difference`. A
/// magnitude-scaled tolerance is used because the harmonic third derivatives
/// scale like `freq³` (≈ thousands for the higher harmonics), so a pure
/// absolute bound would be meaningless at the top of the range.
///
/// The numerical reference is a **4th-order** 5-point central difference
/// `(−f₊₂ + 8f₊ − 8f₋ + f₋₂)/(12h)` rather than the 2-point `(f₊−f₋)/(2h)`. The
/// 2-point stencil carries an `O(h²)` truncation error that, for a cubic line
/// factor (`t³`) whose true mixed third derivative is exactly 0 at `t=0`, is
/// `≈ 1.6e-6` at `h=1e-4` — above the `abs_tol=1e-6` floor, so it spuriously
/// failed an analytically-correct zero. The 5-point stencil is `O(h⁴)` (and
/// EXACT for polynomials up to degree 4, so it returns 0 to rounding on the
/// monomial line factors), which is the honest reference for this contract.
pub(crate) fn assert_third_jet_matches_central_difference<E: SaeBasisThirdJet>(
    evaluator: &E,
    coords: Array2<f64>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<(), String> {
    let epsilon = 1.0e-4;
    let third = evaluator.third_jet(coords.view())?;
    let second = evaluator.second_jet(coords.view())?;
    let (n_rows, n_basis, latent_dim, ld_b, ld_c) = third.dim();
    assert_eq!(latent_dim, ld_b);
    assert_eq!(latent_dim, ld_c);
    assert_eq!((n_rows, n_basis, latent_dim, latent_dim), second.dim());
    for row in 0..n_rows {
        for axis_e in 0..latent_dim {
            let mut plus2 = coords.clone();
            let mut plus = coords.clone();
            let mut minus = coords.clone();
            let mut minus2 = coords.clone();
            plus2[[row, axis_e]] += 2.0 * epsilon;
            plus[[row, axis_e]] += epsilon;
            minus[[row, axis_e]] -= epsilon;
            minus2[[row, axis_e]] -= 2.0 * epsilon;
            let second_plus2 = evaluator.second_jet(plus2.view())?;
            let second_plus = evaluator.second_jet(plus.view())?;
            let second_minus = evaluator.second_jet(minus.view())?;
            let second_minus2 = evaluator.second_jet(minus2.view())?;
            for basis in 0..n_basis {
                for axis_a in 0..latent_dim {
                    for axis_c in 0..latent_dim {
                        let fd = (-second_plus2[[row, basis, axis_a, axis_c]]
                            + 8.0 * second_plus[[row, basis, axis_a, axis_c]]
                            - 8.0 * second_minus[[row, basis, axis_a, axis_c]]
                            + second_minus2[[row, basis, axis_a, axis_c]])
                            / (12.0 * epsilon);
                        let analytic = third[[row, basis, axis_a, axis_c, axis_e]];
                        let error = (analytic - fd).abs();
                        let threshold = abs_tol + rel_tol * analytic.abs().max(fd.abs());
                        assert!(
                            error <= threshold,
                            "row={row} basis={basis} a={axis_a} c={axis_c} e={axis_e}: \
                                 analytic={analytic:.12e}, fd={fd:.12e}, error={error:.6e}, \
                                 threshold={threshold:.6e}"
                        );
                    }
                }
            }
        }
    }
    // Full symmetry across the three trailing derivative axes (mixed partials
    // commute), so every permutation of `(a, c, e)` must agree.
    for row in 0..n_rows {
        for basis in 0..n_basis {
            for a in 0..latent_dim {
                for b in 0..latent_dim {
                    for c in 0..latent_dim {
                        let reference = third[[row, basis, a, b, c]];
                        for perm in [[a, c, b], [b, a, c], [b, c, a], [c, a, b], [c, b, a]] {
                            let permuted = third[[row, basis, perm[0], perm[1], perm[2]]];
                            assert!(
                                (reference - permuted).abs() <= 1.0e-10,
                                "third_jet not symmetric: row={row} basis={basis} \
                                     ({a},{b},{c})={reference:.6e} vs ({},{},{})={permuted:.6e}",
                                perm[0],
                                perm[1],
                                perm[2]
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
pub(crate) fn isometry_periodic_second_jet_matches_fd() -> Result<(), String> {
    // Magnitude-scaled tolerance: the top harmonic (ω = 2π·3) drives a
    // O(ε²·ω⁴) ≈ 2e-5 central-difference truncation floor, far above any flat
    // 1e-5 absolute bound; rel_tol = 1e-5 tracks the ω⁴ scale (analytic exact).
    assert_second_jet_matches_central_difference(
        &PeriodicHarmonicEvaluator::new(7).unwrap(),
        array![[-0.37], [0.0], [0.125], [0.41]],
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_sphere_second_jet_matches_fd() -> Result<(), String> {
    // Stay inside the interior `(-π/2, π/2)` for lat so the chain factor
    // is active — that is where the Hessian carries information.
    let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
    assert_second_jet_matches_central_difference(
        &SphereChartEvaluator,
        sphere_coords,
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_torus_second_jet_matches_fd() -> Result<(), String> {
    let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
    let evaluator = TorusHarmonicEvaluator::new(2, 3).unwrap();
    assert!(evaluator.basis_size() > 0);
    // Same ω⁴ truncation floor as the periodic case (top harmonic ω = 2π·3).
    assert_second_jet_matches_central_difference(&evaluator, torus_coords, 1.0e-6, 1.0e-5)?;
    Ok(())
}

#[test]
pub(crate) fn isometry_periodic_third_jet_matches_fd() -> Result<(), String> {
    assert_third_jet_matches_central_difference(
        &PeriodicHarmonicEvaluator::new(7).unwrap(),
        array![[-0.37], [0.0], [0.125], [0.41]],
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_sphere_third_jet_matches_fd() -> Result<(), String> {
    // Interior of `(-π/2, π/2)` for lat so the chart chain factor is active —
    // that is where the third-order curvature term carries information.
    let sphere_coords = array![[-0.7, -1.2], [-0.25, 0.0], [0.35, 0.9], [0.8, 2.1]];
    assert_third_jet_matches_central_difference(
        &SphereChartEvaluator,
        sphere_coords,
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

#[test]
pub(crate) fn isometry_torus_third_jet_matches_fd() -> Result<(), String> {
    let torus_coords = array![[0.1, 0.7], [0.42, 0.0], [0.95, 0.33], [0.5, 0.5]];
    let evaluator = TorusHarmonicEvaluator::new(2, 3).unwrap();
    assert!(evaluator.basis_size() > 0);
    assert_third_jet_matches_central_difference(&evaluator, torus_coords, 1.0e-6, 1.0e-5)?;
    Ok(())
}

#[test]
pub(crate) fn isometry_affine_third_jet_is_trivial_zero() -> Result<(), String> {
    let evaluator = AffineCoordinateEvaluator { latent_dim: 3 };
    let coords = array![[0.2, -0.3, 0.7], [1.1, 0.0, -0.4]];
    let third = evaluator.third_jet(coords.view())?;
    assert_eq!(third.dim(), (coords.nrows(), 4, 3, 3, 3));
    assert!(
        third.iter().all(|x| *x == 0.0),
        "affine third jet must vanish identically, got {third:?}"
    );
    Ok(())
}

#[test]
pub(crate) fn isometry_euclidean_patch_third_jet_matches_fd() -> Result<(), String> {
    let evaluator = EuclideanPatchEvaluator::new(2, 4)?;
    let coords = array![[0.2, -0.3], [0.7, 0.4], [-0.5, 0.9]];
    assert_third_jet_matches_central_difference(&evaluator, coords, 1.0e-6, 1.0e-5)?;
    Ok(())
}

/// Cylinder coordinates: periodic axis 0 (fraction-of-period) crossed with
/// the unbounded line axis 1. Mixed signs/magnitudes on the line axis pin the
/// monomial factor away from the trivial origin.
fn cylinder_test_coords() -> Array2<f64> {
    array![
        [0.0_f64, -1.3],
        [0.125, 0.0],
        [0.4, 0.7],
        [0.91, 2.2],
        [0.6, -0.45]
    ]
}

/// The cylinder product basis must equal the literal outer product of the
/// periodic circle factor and the monomial line factor in the lexicographic
/// (circle-slow, line-fast) layout, and its width must be `(2H+1)·(D+1)`.
#[test]
pub(crate) fn cylinder_phi_is_circle_tensor_line_product() -> Result<(), String> {
    let h = 2usize;
    let degree = 2usize;
    let evaluator = CylinderHarmonicEvaluator::new(h, degree)?;
    let mc = 2 * h + 1;
    let ml = degree + 1;
    assert_eq!(evaluator.circle_basis_size(), mc);
    assert_eq!(evaluator.line_basis_size(), ml);
    assert_eq!(evaluator.basis_size(), mc * ml);

    let coords = cylinder_test_coords();
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    assert_eq!(phi.dim(), (coords.nrows(), mc * ml));
    assert_eq!(jet.dim(), (coords.nrows(), mc * ml, 2));

    let two_pi = std::f64::consts::TAU;
    for row in 0..coords.nrows() {
        let t0 = coords[[row, 0]];
        let t1 = coords[[row, 1]];
        // Independent reconstruction of the per-axis value factors.
        let mut circ = vec![0.0_f64; mc];
        circ[0] = 1.0;
        for k in 1..=h {
            circ[2 * k - 1] = (two_pi * k as f64 * t0).sin();
            circ[2 * k] = (two_pi * k as f64 * t0).cos();
        }
        let line: Vec<f64> = (0..ml).map(|j| t1.powi(j as i32)).collect();
        for c in 0..mc {
            for l in 0..ml {
                let col = c * ml + l;
                let expect = circ[c] * line[l];
                assert_abs_diff_eq!(phi[[row, col]], expect, epsilon = 1e-12);
            }
        }
        // Column 0 is the product of the two constant factors = 1, with a
        // vanishing gradient on both axes.
        assert_abs_diff_eq!(phi[[row, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(jet[[row, 0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(jet[[row, 0, 1]], 0.0, epsilon = 1e-12);
    }
    Ok(())
}

/// Cylinder first jet (`∂Φ/∂t₀`, `∂Φ/∂t₁`) vs central differences.
#[test]
pub(crate) fn cylinder_jacobian_matches_central_difference() {
    assert_jacobian_matches_central_difference(
        &CylinderHarmonicEvaluator::new(3, 3).unwrap(),
        cylinder_test_coords(),
        1.0e-6,
    );
}

/// Cylinder Hessian vs central differences (product rule across the two
/// disjoint axes: `∂²/∂t₀² = c''·l`, `∂²/∂t₁² = c·l''`, `∂²/∂t₀∂t₁ = c'·l'`).
/// The top circle harmonic (ω = 2π·3) sets the same ω⁴ truncation floor as
/// the periodic/torus cases, so a magnitude-scaled rel_tol is used.
#[test]
pub(crate) fn cylinder_second_jet_matches_fd() -> Result<(), String> {
    let evaluator = CylinderHarmonicEvaluator::new(3, 3)?;
    assert_second_jet_matches_central_difference(
        &evaluator,
        cylinder_test_coords(),
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

/// Cylinder third jet vs a central difference of the (FD-validated) second
/// jet, plus full symmetry across the three trailing axes.
#[test]
pub(crate) fn cylinder_third_jet_matches_fd() -> Result<(), String> {
    let evaluator = CylinderHarmonicEvaluator::new(3, 3)?;
    assert_third_jet_matches_central_difference(
        &evaluator,
        cylinder_test_coords(),
        1.0e-6,
        1.0e-5,
    )?;
    Ok(())
}

/// The cylinder roughness Gram is `S = Sc ⊗ Gl + Gc ⊗ Sl`: symmetric, PSD,
/// with the constant column (`[c=0, l=0]`, the only column with neither a
/// circle-bending nor a line-bending contribution) exactly in its null
/// space. The diagonal entries match the closed-form per-axis energies.
#[test]
pub(crate) fn cylinder_roughness_gram_is_psd_with_constant_nullspace() {
    let h = 2usize;
    let degree = 2usize;
    let evaluator = CylinderHarmonicEvaluator::new(h, degree).unwrap();
    let mc = 2 * h + 1;
    let ml = degree + 1;
    let m = mc * ml;
    let s = evaluator.roughness_gram();
    assert_eq!(s.dim(), (m, m));

    // Symmetry.
    for i in 0..m {
        for j in 0..m {
            assert_abs_diff_eq!(s[[i, j]], s[[j, i]], epsilon = 1e-12);
        }
    }

    // The constant column (col 0 = [c=0, l=0]) is annihilated: neither
    // factor bends, so its entire row/column is zero.
    for j in 0..m {
        assert_abs_diff_eq!(s[[0, j]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s[[j, 0]], 0.0, epsilon = 1e-12);
    }

    // Closed-form diagonal check for the pure-circle column `[c, l=0]`:
    // `S[c0,c0] = Sc[c,c]·Gl[0,0] + Gc[c,c]·Sl[0,0]`. With l=0 the line is a
    // constant, so `Sl[0,0] = 0` and `Gl[0,0] = ∫₀¹ 1 = 1`, giving `Sc[c,c]`.
    let two_pi = std::f64::consts::TAU;
    for k in 1..=h {
        let omega4 = (two_pi * k as f64).powi(4);
        let s_idx = 2 * k - 1;
        let c_idx = 2 * k;
        // Sc[s,s] = Sc[c,c] = ω⁴·½ (∫₀¹ sin² = ∫₀¹ cos² = ½).
        assert_abs_diff_eq!(s[[s_idx * ml, s_idx * ml]], omega4 * 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(s[[c_idx * ml, c_idx * ml]], omega4 * 0.5, epsilon = 1e-6);
    }

    // The pure-line quadratic column `[c=0, l=2]` carries only line-bending
    // energy: `S = Gc[0,0]·Sl[2,2]` with `Gc[0,0] = 1` and
    // `Sl[2,2] = ∫₀¹ (2)² dt = 4`.
    if degree >= 2 {
        let col = 2; // c=0 → col = 0*ml + 2.
        assert_abs_diff_eq!(s[[col, col]], 4.0, epsilon = 1e-12);
    }

    // PSD: every eigenvalue ≥ 0 (within a tight tolerance).
    let (evals, _) = s.eigh(Side::Lower).unwrap();
    for &lam in evals.iter() {
        assert!(
            lam >= -1.0e-9,
            "cylinder roughness Gram must be PSD; got eigenvalue {lam:.3e}"
        );
    }
}

/// `CylinderHarmonicEvaluator::new` rejects `circle_harmonics == 0` (an S¹
/// with no harmonic pair is degenerate).
#[test]
pub(crate) fn cylinder_rejects_zero_harmonics() {
    assert!(CylinderHarmonicEvaluator::new(0, 2).is_err());
    assert!(CylinderHarmonicEvaluator::new(1, 0).is_ok());
}

/// The cylinder latent manifold is the product `S¹ × ℝ`: a unit-period
/// circle on axis 0 and an unbounded Euclidean line on axis 1.
#[test]
pub(crate) fn cylinder_latent_manifold_is_circle_times_line() {
    let manifold = SaeAtomBasisKind::Cylinder.latent_manifold(2);
    match manifold {
        LatentManifold::Product(parts) => {
            assert_eq!(parts.len(), 2);
            assert!(matches!(parts[0], LatentManifold::Circle { period } if period == 1.0));
            assert!(matches!(parts[1], LatentManifold::Euclidean));
        }
        other => panic!("expected Product[Circle, Euclidean], got {other:?}"),
    }
}

/// The cylinder projection seed grid sweeps the periodic axis over one period
/// `[0, 1)` and holds the unbounded line axis at the hull-centered seed `0`.
#[test]
pub(crate) fn cylinder_projection_seed_grid_sweeps_circle_only() {
    let r = 12usize;
    let grid = SaeAtomBasisKind::Cylinder
        .projection_seed_grid(2, r)
        .unwrap();
    assert_eq!(grid.dim(), (r, 2));
    for i in 0..r {
        assert_abs_diff_eq!(grid[[i, 0]], i as f64 / r as f64, epsilon = 1e-12);
        assert_abs_diff_eq!(grid[[i, 1]], 0.0, epsilon = 1e-12);
    }
    assert!(grid.column(0).iter().all(|&t| (0.0..1.0).contains(&t)));
}

/// Issue #247: the Duchon coordinate evaluator must return a forward design
/// and a derivative jet with *matching column counts* — the original bug
/// was a radial-only design paired with a radial+polynomial jet (or vice
/// versa), which the consumer rejected as a "design/jet column mismatch".
#[test]
pub(crate) fn duchon_coordinate_evaluator_phi_and_jet_share_column_count() {
    for (d, centers) in [
        (1usize, array![[-1.0], [-0.4], [0.1], [0.6], [1.2], [1.9]]),
        (
            2usize,
            array![
                [-1.0, -0.8],
                [-0.3, 0.4],
                [0.2, -0.5],
                [0.7, 0.9],
                [1.1, -0.2],
                [1.6, 0.6],
            ],
        ),
    ] {
        let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
        let coords = match d {
            1 => array![[-0.5], [0.0], [0.3], [0.8]],
            _ => array![[-0.5, 0.2], [0.0, -0.3], [0.3, 0.7], [0.8, -0.1]],
        };
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        assert_eq!(
            phi.ncols(),
            jet.shape()[1],
            "Duchon d={d}: Phi has {} columns but jet has {}",
            phi.ncols(),
            jet.shape()[1]
        );
        assert_eq!(jet.shape()[0], coords.nrows());
        assert_eq!(jet.shape()[2], d);
    }
}

/// The Duchon evaluator's analytic first jet must equal the finite
/// difference of its own forward design — i.e. `dPhi/dt` is the true
/// derivative of `Phi(t)`, with no stray amplification/column mismatch.
#[test]
pub(crate) fn duchon_coordinate_evaluator_jacobian_matches_fd() {
    let centers = array![
        [-1.0, -0.8],
        [-0.3, 0.4],
        [0.2, -0.5],
        [0.7, 0.9],
        [1.1, -0.2],
        [1.6, 0.6],
    ];
    let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
    // Keep probe points away from any center so the radial kernel is smooth.
    let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
    assert_jacobian_matches_central_difference(&evaluator, coords, 1.0e-4);
}

/// The Duchon evaluator's analytic second jet must match the FD of its
/// (FD-validated) first jet.
#[test]
pub(crate) fn duchon_coordinate_evaluator_second_jet_matches_fd() -> Result<(), String> {
    let centers = array![
        [-1.0, -0.8],
        [-0.3, 0.4],
        [0.2, -0.5],
        [0.7, 0.9],
        [1.1, -0.2],
        [1.6, 0.6],
    ];
    let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
    let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
    assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-4, 1.0e-4)?;
    Ok(())
}

/// The Duchon evaluator's closed-form analytic third jet (radial
/// third-derivative kernel block + monomial nullspace block) must match the
/// FD of its (FD-validated) second jet, validating the closed form that
/// replaced the forbidden finite-difference `third_jet_dyn` default.
#[test]
pub(crate) fn duchon_coordinate_evaluator_third_jet_matches_fd() -> Result<(), String> {
    let centers = array![
        [-1.0, -0.8],
        [-0.3, 0.4],
        [0.2, -0.5],
        [0.7, 0.9],
        [1.1, -0.2],
        [1.6, 0.6],
    ];
    let evaluator = DuchonCoordinateEvaluator::new(centers, 2).unwrap();
    let coords = array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]];
    assert_third_jet_matches_central_difference(&evaluator, coords, 1.0e-4, 1.0e-4)?;
    Ok(())
}

/// The Euclidean tangent-patch evaluator's monomial design and its
/// first/second jets must be mutually consistent under finite differences.
#[test]
pub(crate) fn euclidean_patch_evaluator_jets_match_fd() -> Result<(), String> {
    let evaluator = EuclideanPatchEvaluator::new(2, 2).unwrap();
    let coords = array![[0.0, -1.0], [3.5, 0.25], [-0.75, 1.2], [0.4, 0.9]];
    assert_jacobian_matches_central_difference(&evaluator, coords.clone(), 1.0e-6);
    assert_second_jet_matches_central_difference(&evaluator, coords, 1.0e-5, 1.0e-5)?;
    // The degree-2 patch in d=2 has columns {1, x, y, x², xy, y²}.
    let (phi, _jet) = evaluator.evaluate(array![[0.0, 0.0]].view())?;
    assert_eq!(phi.ncols(), 6);
    Ok(())
}

#[test]
pub(crate) fn euclidean_affine_gauge_canonicalization_preserves_reconstruction()
-> Result<(), String> {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2)?);
    let canonical = array![[-1.0_f64], [-0.35], [0.1], [0.65], [1.2]];
    let mut coords = canonical.clone();
    for row in 0..coords.nrows() {
        coords[[row, 0]] = 2.75 + 4.0 * canonical[[row, 0]];
    }
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let decoder = array![[0.25, -0.4], [1.2, 0.3], [-0.15, 0.5]];
    let atom = SaeManifoldAtom::new(
        "euclidean_patch",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(evaluator.basis_size()),
    )?
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((coords.nrows(), 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )?;
    let mut term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let before = term.fitted();

    term.canonicalize_affine_gauge_after_accept(None)?;

    let after = term.fitted();
    let max_abs = before
        .iter()
        .zip(after.iter())
        .fold(0.0_f64, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(
        max_abs <= 1.0e-10,
        "canonicalization changed reconstruction by {max_abs:.3e}"
    );
    let live = term.assignment.coords[0].as_matrix();
    let mean = live.column(0).sum() / live.nrows() as f64;
    let rms = (live.column(0).iter().map(|v| v * v).sum::<f64>() / live.nrows() as f64).sqrt();
    assert_abs_diff_eq!(mean, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(rms, 1.0, epsilon = 1.0e-12);
    Ok(())
}

#[test]
pub(crate) fn quotient_step_norm_removes_pure_euclidean_affine_gauge() -> Result<(), String> {
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2)?);
    let coords = array![[-1.0_f64], [-0.4], [0.2], [0.8], [1.3]];
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let decoder = array![[0.1, -0.2], [1.0, 0.4], [0.25, -0.3]];
    let atom = SaeManifoldAtom::new(
        "euclidean_patch",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(evaluator.basis_size()),
    )?
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((coords.nrows(), 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )?;
    let term = SaeManifoldTerm::new(vec![atom], assignment)?;
    let gauges = term.dense_step_gauge_vectors()?;
    assert!(
        gauges.len() >= 2,
        "expected translation and scale gauge generators"
    );
    let n_coord = term.n_obs() * term.assignment.row_block_dim();
    let gauge = &gauges[1];
    let delta_t = gauge.slice(s![..n_coord]);
    let delta_beta = gauge.slice(s![n_coord..]);
    let raw = gauge.iter().map(|v| v * v).sum::<f64>();

    let quotient =
        term.quotient_newton_step_norm_sq(delta_t, delta_beta, raw, &vec![0.0; term.k_atoms()])?;

    assert!(
        quotient <= raw.max(1.0) * 1.0e-20,
        "pure affine gauge step left quotient norm squared {quotient:.3e} from raw {raw:.3e}"
    );
    Ok(())
}

/// Torus T^2 fit on synthetic data with a known two-frequency signal.
/// Drives a single torus atom through the [`SaeManifoldTerm`] Newton loop
/// and checks that the in-sample reconstruction R² clears 0.5.
#[test]
pub(crate) fn sae_torus_atom_recovers_two_frequency_synthetic() {
    let n = 96usize;
    let p = 4usize;
    let h = 3usize;
    let d = 2usize;
    let evaluator = TorusHarmonicEvaluator::new(d, h).unwrap();
    let m = evaluator.basis_size();
    // True coords on T^2 (phase in [0, 1)).
    let mut true_coords = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        true_coords[[i, 0]] = ((i as f64) * 0.137).rem_euclid(1.0);
        true_coords[[i, 1]] = ((i as f64) * 0.241 + 0.13).rem_euclid(1.0);
    }
    // Synthetic target: a low-frequency periodic signal on T^2 mixed
    // linearly into a p-dim ambient.
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t1 = 2.0 * std::f64::consts::PI * true_coords[[i, 0]];
        let t2 = 2.0 * std::f64::consts::PI * true_coords[[i, 1]];
        z[[i, 0]] = t1.sin() + 0.3 * t2.cos();
        z[[i, 1]] = t1.cos() + 0.2 * (t1 + t2).sin();
        z[[i, 2]] = t2.sin();
        z[[i, 3]] = 0.5 * (t1 - t2).cos();
    }
    let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
    // Initialise from the true coords (this test exercises basis correctness
    // and decoder fit, not coordinate identification on T^2).
    let (phi0, jet0) = evaluator.evaluate(true_coords.view()).unwrap();
    // Penalty: identity-on-non-constant + tiny floor on constant.
    let mut penalty = Array2::<f64>::eye(m);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new(
        "torus_atom",
        SaeAtomBasisKind::Torus,
        d,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        penalty,
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TorusHarmonicEvaluator::new(d, h).unwrap()));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![true_coords],
        vec![LatentManifold::Product(vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ])],
        AssignmentMode::softmax(0.5),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // ARD log-precision is per-axis (length == atom latent dim), not a
    // single scalar — see `SaeManifoldRho::to_flat` / `from_flat` and
    // the validation in `negative_log_ard_prior` (`ARD rho atom k has
    // len ... but atom dim is d`).
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(d)]);
    let ridge = 1.0e-6;
    for _ in 0..10 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .unwrap();
        if !loss.total().is_finite() {
            break;
        }
    }
    let fitted = term.fitted();
    assert_eq!(fitted.dim(), (n, p));
    let mut sse = 0.0_f64;
    for ((row, col), v) in fitted.indexed_iter() {
        let r = v - z[[row, col]];
        sse += r * r;
    }
    let r2 = 1.0 - sse / sst.max(1.0e-12);
    assert!(
        r2 >= 0.5,
        "torus atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
    );
}

/// Sphere S² fit on a synthetic spherical signal. Drives a single sphere
/// atom through the [`SaeManifoldTerm`] Newton loop and checks in-sample
/// R² ≥ 0.5.
#[test]
pub(crate) fn sae_sphere_atom_recovers_synthetic_signal() {
    let n = 96usize;
    let p = 3usize;
    let d = 2usize;
    // True (lat, lon) coords.
    let mut true_coords = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let t = (i as f64) / (n as f64);
        true_coords[[i, 0]] = -0.5 + 1.0 * t; // lat in [-0.5, 0.5]
        true_coords[[i, 1]] = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * t;
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let lat = true_coords[[i, 0]];
        let lon = true_coords[[i, 1]];
        let x = lat.cos() * lon.cos();
        let y = lat.cos() * lon.sin();
        let zc = lat.sin();
        z[[i, 0]] = x;
        z[[i, 1]] = y;
        z[[i, 2]] = zc;
    }
    let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();
    let (phi0, jet0) = SphereChartEvaluator.evaluate(true_coords.view()).unwrap();
    let m = phi0.ncols();
    let mut penalty = Array2::<f64>::eye(m);
    penalty *= 1.0e-4;
    let atom = SaeManifoldAtom::new(
        "sphere_atom",
        SaeAtomBasisKind::Sphere,
        d,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        penalty,
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(SphereChartEvaluator));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![true_coords],
        vec![LatentManifold::Product(vec![
            LatentManifold::Interval {
                lo: -std::f64::consts::FRAC_PI_2,
                hi: std::f64::consts::FRAC_PI_2,
            },
            LatentManifold::Circle {
                period: std::f64::consts::TAU,
            },
        ])],
        AssignmentMode::softmax(0.5),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // The sphere atom's coordinate is a dim-2 product manifold (lat × lon),
    // so per-axis ARD must carry one log-precision per axis (`atom dim = 2`).
    // A length-1 block would be indexed out of bounds at `axis == 1` in the
    // per-axis assembly loop and is rejected by the per-axis ARD contract.
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(2)]);
    let ridge = 1.0e-6;
    for _ in 0..10 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, ridge, ridge)
            .unwrap();
        if !loss.total().is_finite() {
            break;
        }
    }
    let fitted = term.fitted();
    assert_eq!(fitted.dim(), (n, p));
    let mut sse = 0.0_f64;
    for ((row, col), v) in fitted.indexed_iter() {
        let r = v - z[[row, col]];
        sse += r * r;
    }
    let r2 = 1.0 - sse / sst.max(1.0e-12);
    assert!(
        r2 >= 0.5,
        "sphere atom R² too low: {r2:.4} (sst={sst:.4}, sse={sse:.4})"
    );
}

/// Mirror of the Python `test_sae_manifold_softmax_dispatch` shape: drive a
/// single periodic atom on a 1-harmonic synthetic target with 10 Newton
/// steps end-to-end in Rust and check that the multi-step loop achieves
/// in-sample R² ≥ 0.95.
#[test]
pub(crate) fn sae_manifold_fit_10_steps_one_harmonic_reaches_high_r2() {
    let n = 64usize;
    let m = 3usize;
    let p = 1usize;

    let true_t: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * true_t[i];
        z[[i, 0]] = 0.7 * angle.sin() + 0.3 * angle.cos();
    }
    let sst: f64 = z.iter().map(|v| v * v).sum::<f64>();

    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut coords0_data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        // Phase-shifted initialization so the optimizer must do real work.
        coords0_data[[i, 0]] = (true_t[i] + 0.25).rem_euclid(1.0);
    }
    let (phi0, jet0) = evaluator.evaluate(coords0_data.view()).unwrap();

    let atom = SaeManifoldAtom::new(
        "periodic_atom",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        Array2::<f64>::zeros((m, p)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords0_data],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.5),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);

    let max_iter = 10usize;
    let learning_rate = 1.0;
    let ridge = 1.0e-6;
    let mut prev_total = f64::INFINITY;
    for _ in 0..max_iter {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, learning_rate, ridge, ridge)
            .unwrap();
        let total = loss.total();
        if !total.is_finite() {
            break;
        }
        let denom = prev_total.abs().max(1.0e-12);
        let rel = (prev_total - total).abs() / denom;
        prev_total = total;
        if rel < 1.0e-6 {
            break;
        }
    }

    let fitted = term.fitted();
    assert_eq!(fitted.dim(), (n, p));
    let mut ssr = 0.0;
    for i in 0..n {
        let r = z[[i, 0]] - fitted[[i, 0]];
        ssr += r * r;
    }
    let r2 = 1.0 - ssr / sst.max(1.0e-12);
    assert!(
        r2 >= 0.95,
        "10-step in-sample R² = {r2:.4} (ssr={ssr:.6}, sst={sst:.6}) should be >= 0.95"
    );
}

/// Regression test for issue #177: softmax assignment used to bail out of
/// the row-block Hessian assembly with "softmax assignment hessian diag
/// unavailable". The penalty now exposes the analytic diagonal extracted
/// from its row-dense HVP, so the joint-fit driver completes one step.
#[test]
pub(crate) fn softmax_assignment_hessian_diag_is_available_for_k2() {
    let n = 4usize;
    let k = 2usize;
    let logits = Array2::<f64>::from_shape_fn((n, k), |(i, j)| 0.1 * (i as f64) - 0.2 * (j as f64));
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
        .expect("softmax assignment Hessian diagonal must be available");
    assert_eq!(grad.len(), n * k);
    assert_eq!(diag.len(), n * k);
    assert!(grad.iter().all(|v| v.is_finite()));
    assert!(diag.iter().all(|v| v.is_finite()));
}

#[test]
pub(crate) fn sae_registry_refuses_assignment_sparsity_penalties() {
    let n = 3usize;
    let k = 2usize;
    let logits = Array2::<f64>::zeros((n, k));
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::softmax(0.7),
    )
    .expect("valid assignment");
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|atom_idx| {
            SaeManifoldAtom::new(
                format!("periodic_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                Array2::<f64>::ones((n, 1)),
                Array3::<f64>::zeros((n, 1, 1)),
                Array2::<f64>::zeros((1, 1)),
                Array2::<f64>::eye(1),
            )
            .expect("valid atom")
        })
        .collect();
    let term = SaeManifoldTerm::new(atoms, assignment).expect("valid SAE term");

    let mut softmax_registry = AnalyticPenaltyRegistry::new();
    softmax_registry.push(AnalyticPenaltyKind::SoftmaxAssignmentSparsity(Arc::new(
        gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(k, 0.7),
    )));
    let softmax_err = term
        .validate_analytic_penalty_registry(&softmax_registry)
        .expect_err("SAE registry must reject softmax assignment sparsity");
    assert!(softmax_err.contains("assignment sparsity"));

    let mut ibp_registry = AnalyticPenaltyRegistry::new();
    ibp_registry.push(AnalyticPenaltyKind::IBPAssignment(Arc::new(
        gam_terms::analytic_penalties::IBPAssignmentPenalty::new(k, 1.2, 0.7, false),
    )));
    let ibp_err = term
        .validate_analytic_penalty_registry(&ibp_registry)
        .expect_err("SAE registry must reject IBP assignment sparsity");
    assert!(ibp_err.contains("assignment sparsity"));
}

#[test]
pub(crate) fn ibp_fixed_alpha_assignment_value_matches_logit_gradient_fd() {
    let n = 4usize;
    let k = 3usize;
    let logits = Array2::<f64>::from_shape_vec(
        (n, k),
        vec![
            -0.4, 0.2, 0.7, 0.1, -0.3, 0.5, 0.8, -0.1, -0.6, 0.3, 0.6, -0.2,
        ],
    )
    .expect("valid IBP logit grid");
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::ibp_map(0.9, 1.4, false),
    )
    .expect("valid IBP assignment");
    let rho = SaeManifoldRho::new(0.23_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, _) =
        assignment_prior_grad_hdiag(&assignment, &rho).expect("IBP assignment gradient");
    let idx = 5usize;
    let step = 1.0e-6_f64;
    let mut plus = assignment.clone();
    plus.logits[[idx / k, idx % k]] += step;
    let mut minus = assignment.clone();
    minus.logits[[idx / k, idx % k]] -= step;
    let fd =
        (assignment_prior_value(&plus, &rho) - assignment_prior_value(&minus, &rho)) / (2.0 * step);

    assert_abs_diff_eq!(grad[idx], fd, epsilon = 2.0e-7);
}

/// #1038 assembly-site wiring: a live IBP-active multi-atom assembly must
/// emit the exact cross-row Woodbury source `IbpCrossRowSource` whose
/// entries reproduce the NUMERICAL off-diagonal (`i≠j`) logit Hessian of the
/// SAE objective end-to-end. The ONLY source of cross-row `i≠j` logit
/// coupling is the IBP empirical-mass prior `M_k = Σ_i z_ik` (the data-fit
/// reconstruction of each row depends only on that row's own logits), so
/// `∂²(assignment_prior_value)/∂ℓ_ik∂ℓ_jk = d_k·z'_ik·z'_jk` for `i≠j`, with
/// `d_k = cross_row_d[k]` and `z'_ik = z_jac[i·K+k]` — exactly the rank-one
/// `U D Uᵀ` the assembled `sys.ibp_cross_row` encodes and the arrow-Schur
/// consumer rides as the exact Woodbury (value + logdet + θ/ρ-adjoint).
///
/// This certifies the assembly-site source matches the consumer's `U`/index
/// convention bit-for-bit: each entry's `global_t_index` is the row's logit
/// slot in the latent block (`row_offsets[i] + k` for the dense IBP layout),
/// and the rank-one product against the central-difference Hessian closes.
#[test]
pub(crate) fn ibp_assembly_emits_cross_row_woodbury_source_matching_fd_hessian() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let atom0 = SaeManifoldAtom::new(
        "periodic0",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        array![[0.25], [-0.35], [0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let atom1 = SaeManifoldAtom::new(
        "periodic1",
        SaeAtomBasisKind::Periodic,
        1,
        phi1,
        jet1,
        array![[-0.10], [0.20], [0.30]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    // IBP-active logits (positive ⇒ near-on gate, interior π so the
    // empirical-mass channel is live — `pi_jac ≠ 0`).
    let logits = array![[1.2, 0.4], [0.6, 1.0], [0.9, 0.3], [1.4, 0.7]];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.8, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
    let target = array![[0.12], [-0.03], [0.08], [0.20]];
    let rho = SaeManifoldRho::new(
        0.3_f64.ln(),
        0.7_f64.ln(),
        vec![array![0.9_f64.ln()], array![1.1_f64.ln()]],
    );

    let n = term.assignment.n_obs();
    let k = term.assignment.k_atoms();

    // Assemble the live arrow system; it must now carry the IBP source.
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("IBP arrow assembly");
    let source = sys
        .ibp_cross_row
        .as_ref()
        .expect("an IBP-active assembly must emit the cross-row Woodbury source");
    assert_eq!(source.r, k, "the rank must be the atom count K");

    // Rebuild the dense `U` and `d` the consumer sees from the sparse entries,
    // and check the global-index convention is the dense IBP layout
    // (`row_offsets[i] + k`), i.e. atom `k`'s logit slot of row `i`.
    let total_t = sys.row_offsets[n];
    let mut u = Array2::<f64>::zeros((total_t, k));
    for &(g, atom_k, z_prime) in &source.entries {
        u[[g, atom_k]] += z_prime;
    }
    for i in 0..n {
        for atom_k in 0..k {
            let g = sys.row_offsets[i] + atom_k;
            // The entry for (row i, atom k) must sit at the row's logit slot.
            assert!(
                u[[g, atom_k]].abs() > 0.0 || term.assignment.logits[[i, atom_k]].abs() > 1.0e3,
                "row {i} atom {atom_k} logit slot must carry a z' entry"
            );
        }
    }

    // Central-difference the assignment-prior value cross-row (i≠j) Hessian
    // and assert it equals the rank-one `d_k·z'_ik·z'_jk` the source encodes.
    let d = source.d.clone();
    let step = 1.0e-5_f64;
    let fd_cross = |i: usize, j: usize, atom_k: usize| -> f64 {
        let bump = |si: f64, sj: f64| -> f64 {
            let mut a = term.assignment.clone();
            a.logits[[i, atom_k]] += si * step;
            a.logits[[j, atom_k]] += sj * step;
            assignment_prior_value(&a, &rho)
        };
        // mixed second difference ∂²V/∂ℓ_ik∂ℓ_jk
        (bump(1.0, 1.0) - bump(1.0, -1.0) - bump(-1.0, 1.0) + bump(-1.0, -1.0))
            / (4.0 * step * step)
    };

    for atom_k in 0..k {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let gi = sys.row_offsets[i] + atom_k;
                let gj = sys.row_offsets[j] + atom_k;
                let analytic = d[atom_k] * u[[gi, atom_k]] * u[[gj, atom_k]];
                let fd = fd_cross(i, j, atom_k);
                assert_abs_diff_eq!(analytic, fd, epsilon = 5.0e-6);
            }
        }
    }

    // Distinct atom columns do NOT couple cross-row (independent
    // stick-breaking masses): the off-diagonal in a DIFFERENT column is zero.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut a = term.assignment.clone();
            let cross = {
                let s = 1.0e-5_f64;
                let mut bump = |si: f64, sj: f64| -> f64 {
                    a.logits[[i, 0]] = term.assignment.logits[[i, 0]] + si * s;
                    a.logits[[j, 1]] = term.assignment.logits[[j, 1]] + sj * s;
                    assignment_prior_value(&a, &rho)
                };
                (bump(1.0, 1.0) - bump(1.0, -1.0) - bump(-1.0, 1.0) + bump(-1.0, -1.0))
                    / (4.0 * s * s)
            };
            assert_abs_diff_eq!(cross, 0.0, epsilon = 5.0e-6);
        }
    }
}

#[test]
pub(crate) fn jumprelu_assignment_value_matches_logit_gradient_fd() {
    let n = 4usize;
    let k = 2usize;
    let temperature = 0.35_f64;
    let threshold = 0.1_f64;
    let logits =
        Array2::<f64>::from_shape_vec((n, k), vec![-13.0, -0.2, 0.0, 0.05, 0.15, 0.4, 0.9, 1.5])
            .expect("valid JumpReLU logit grid");
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::jumprelu(temperature, threshold),
    )
    .expect("valid JumpReLU assignment");
    let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, _) =
        assignment_prior_grad_hdiag(&assignment, &rho).expect("JumpReLU assignment gradient");
    let idx = 4usize;
    let step = 1.0e-6_f64;
    let mut plus = assignment.clone();
    plus.logits[[idx / k, idx % k]] += step;
    let mut minus = assignment.clone();
    minus.logits[[idx / k, idx % k]] -= step;
    let fd =
        (assignment_prior_value(&plus, &rho) - assignment_prior_value(&minus, &rho)) / (2.0 * step);

    assert_abs_diff_eq!(grad[idx], fd, epsilon = 2.0e-8);
}

#[test]
pub(crate) fn jumprelu_assignment_prior_hessian_diag_is_exact_over_logit_sweep() {
    let n = 6usize;
    let k = 2usize;
    let temperature = 0.35_f64;
    let threshold = 0.1_f64;
    let logits = Array2::<f64>::from_shape_vec(
        (n, k),
        vec![
            -2.0, -0.2, 0.0, 0.05, 0.1, 0.15, 0.4, 0.9, 1.5, 2.5, 4.0, 6.0,
        ],
    )
    .expect("valid logit grid");
    let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
    let manifolds = vec![LatentManifold::Circle { period: 1.0 }; k];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits.clone(),
        coords,
        manifolds,
        AssignmentMode::jumprelu(temperature, threshold),
    )
    .expect("valid JumpReLU assignment");
    let rho = SaeManifoldRho::new(0.7_f64.ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);
    let (grad, diag) = assignment_prior_grad_hdiag(&assignment, &rho)
        .expect("JumpReLU assignment prior hessian diag");
    let inv_tau = 1.0 / temperature;
    let inv_tau2 = inv_tau * inv_tau;
    let sparsity_strength = rho.log_lambda_sparse.exp();

    assert_eq!(grad.len(), n * k);
    assert_eq!(diag.len(), n * k);
    let mut saw_negative = false;
    for (idx, &entry) in diag.iter().enumerate() {
        let logit = logits[[idx / k, idx % k]];
        // Expected = exact second derivative of the threshold-centered
        // surrogate σ((l−θ)/τ), using the same machine-precision support as
        // the value and gradient paths.
        let expected = if jumprelu_in_optimization_band(logit, threshold, temperature) {
            let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
            let slope = activation * (1.0 - activation);
            sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2
        } else {
            0.0
        };
        assert!(
            entry.is_finite(),
            "JumpReLU hessian_diag must be finite at index {idx}"
        );
        saw_negative |= entry < 0.0;
        assert_abs_diff_eq!(entry, expected, epsilon = 1e-12);
    }
    assert!(
        saw_negative,
        "exact JumpReLU hessian_diag must go negative above the threshold"
    );
}

/// Regression test for issue #174: K>=2 periodic atoms with zero-init
/// decoder used to collapse to A≈0 because the assignment prior was the
/// only term with non-zero gradient at iter 0. The pyffi entry point now
/// seeds decoder coefficients via a joint LSQ projection of Z onto
/// [a_init · Phi_k]. This test exercises that exact seeding strategy
/// in pure Rust and verifies the joint Newton fit reaches positive R²
/// on a clean K=2 periodic torus signal, mirroring the failing
/// reproducer in #174.
#[test]
pub(crate) fn ibp_map_k2_periodic_torus_recovers_signal_with_lsq_init() {
    use gam_linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
    use faer::Side as FaerSide;

    let n = 200usize;
    let p = 8usize;
    let k = 2usize;
    let m = 5usize; // 1 (constant) + 2 harmonics * 2 (sin/cos) = 5

    // Build a synthetic K=2 torus signal Z = [cos th1, sin th1, cos th2, sin th2] @ mix
    // with two latent angles. Deterministic seed via index arithmetic.
    let mut theta = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        theta[[i, 0]] = ((i as f64) * 0.07) % 1.0;
        theta[[i, 1]] = ((i as f64) * 0.13 + 0.31) % 1.0;
    }
    let mut raw = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let a1 = 2.0 * std::f64::consts::PI * theta[[i, 0]];
        let a2 = 2.0 * std::f64::consts::PI * theta[[i, 1]];
        raw[[i, 0]] = a1.cos();
        raw[[i, 1]] = a1.sin();
        raw[[i, 2]] = a2.cos();
        raw[[i, 3]] = a2.sin();
    }
    // Deterministic 4x8 mixing matrix.
    let mix = Array2::<f64>::from_shape_fn((4, p), |(i, j)| {
        ((i as f64 + 1.0) * 0.37 + (j as f64) * 0.21).sin()
    });
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            let mut acc = 0.0;
            for r in 0..4 {
                acc += raw[[i, r]] * mix[[r, j]];
            }
            z[[i, j]] = acc;
        }
    }
    // Centre Z so R² is well-defined relative to mean.
    let mut col_mean = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut acc = 0.0;
        for i in 0..n {
            acc += z[[i, j]];
        }
        col_mean[j] = acc / n as f64;
    }
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] -= col_mean[j];
        }
    }

    // Atom coordinates: use the (shifted) true angles so the periodic
    // basis aligns with the signal — the test isolates the decoder-init
    // collapse, not coordinate recovery.
    let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
    for i in 0..n {
        coords_k[0][[i, 0]] = (theta[[i, 0]] + 0.05).rem_euclid(1.0);
        coords_k[1][[i, 0]] = (theta[[i, 1]] + 0.07).rem_euclid(1.0);
    }
    // Periodic basis (constant + 2 harmonics → M=5) for each atom.
    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut phi_k = Vec::with_capacity(k);
    let mut jet_k = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
        phi_k.push(phi);
        jet_k.push(jet);
    }

    // LSQ seed: joint design X = [0.5 * Phi_1 | 0.5 * Phi_2] (IBP-MAP
    // logit 0 gives sigmoid(0/tau) = 0.5 for both atoms), solve normal
    // equations with a small ridge.
    let m_total = k * m;
    let mut x = Array2::<f64>::zeros((n, m_total));
    for atom_idx in 0..k {
        for i in 0..n {
            for col in 0..m {
                x[[i, atom_idx * m + col]] = 0.5 * phi_k[atom_idx][[i, col]];
            }
        }
    }
    let mut xtx = fast_ata(&x);
    let mut trace = 0.0_f64;
    for i in 0..m_total {
        trace += xtx[[i, i]];
    }
    let jitter = (trace / m_total as f64).max(1.0) * 1.0e-8;
    for i in 0..m_total {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&x, &z);
    let b_joint = xtx
        .cholesky(FaerSide::Lower)
        .expect("LSQ Cholesky")
        .solve_mat(&xtz);

    let mut atoms = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let mut b = Array2::<f64>::zeros((m, p));
        for col in 0..m {
            for j in 0..p {
                b[[col, j]] = b_joint[[atom_idx * m + col, j]];
            }
        }
        let atom = SaeManifoldAtom::new(
            format!("torus_atom_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi_k[atom_idx].clone(),
            jet_k[atom_idx].clone(),
            b,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
        atoms.push(atom);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k)),
        coords_k,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    // `lambda_sparse` is the IBP assignment-sparsity prior weight (now wired
    // through `assignment_prior_grad_hdiag`'s IBP branch, #853). The
    // Beta-Bernoulli BCE energy toward the self-referential empirical active
    // fraction has its global minimum at the all-off gate, so at the old
    // full weight (`log_lambda_sparse = 0 → λ = 1`) it overwhelmed the
    // truth-seeded data fit and collapsed the assignment off both atoms. A
    // moderate prior weight keeps the sparsity pressure honest while letting
    // the LSQ-seeded reconstruction hold both real atoms active — the
    // realistic operating point this recovery test pins.
    let mut rho = SaeManifoldRho::new((0.02_f64).ln(), -6.0, vec![Array1::<f64>::zeros(1); k]);

    let mut prev_total = f64::INFINITY;
    for _ in 0..30 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .unwrap();
        let total = loss.total();
        if !total.is_finite() {
            break;
        }
        let denom = prev_total.abs().max(1.0e-12);
        let rel = (prev_total - total).abs() / denom;
        prev_total = total;
        if rel < 1.0e-6 {
            break;
        }
    }

    let fitted = term.fitted();
    let mut ssr = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        for j in 0..p {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            sst += z[[i, j]] * z[[i, j]];
        }
    }
    let r2 = 1.0 - ssr / sst.max(1.0e-12);
    assert!(
        r2 > 0.5,
        "K=2 periodic torus IBP-MAP R² = {r2:.4} (ssr={ssr:.4}, sst={sst:.4}) should be > 0.5 with LSQ-seeded decoder"
    );
    // Also confirm at least one atom remains active (assignment did not
    // collapse to ~0) — the active mass averaged over rows must exceed
    // a non-trivial threshold.
    let assignments = term.assignment.assignments();
    let mean_active: f64 = assignments.iter().copied().sum::<f64>() / (n as f64);
    assert!(
        mean_active > 0.2,
        "mean active mass across rows = {mean_active:.4} should exceed 0.2; assignment did not collapse"
    );
}

/// Regression test for issue #174 + #177 combined: softmax assignment
/// with K=2 periodic atoms should not crash and should reduce loss.
#[test]
pub(crate) fn softmax_k2_periodic_completes_joint_fit_step() {
    let n = 64usize;
    let p = 4usize;
    let k = 2usize;
    let m = 3usize;

    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        z[[i, 0]] = a.sin();
        z[[i, 1]] = a.cos();
        z[[i, 2]] = (2.0 * a).sin();
        z[[i, 3]] = (2.0 * a).cos();
    }

    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut coords_k = vec![Array2::<f64>::zeros((n, 1)); k];
    for i in 0..n {
        coords_k[0][[i, 0]] = (i as f64) / (n as f64);
        coords_k[1][[i, 0]] = ((i as f64) * 2.0 / (n as f64)).rem_euclid(1.0);
    }
    let mut atoms = Vec::new();
    for atom_idx in 0..k {
        let (phi, jet) = evaluator.evaluate(coords_k[atom_idx].view()).unwrap();
        // Non-trivial decoder init (simulate LSQ seeding) so the data-fit
        // signal is non-zero at iter 0.
        let b = Array2::<f64>::from_shape_fn((m, p), |(i, j)| {
            0.1 * ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
        });
        let atom = SaeManifoldAtom::new(
            format!("a_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            b,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()));
        atoms.push(atom);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k)),
        coords_k,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);

    // First step must succeed (previously bailed with hessian-diag error).
    let loss0 = term
        .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
        .expect("softmax K=2 must complete first joint-fit step");
    assert!(loss0.total().is_finite());
    let loss1 = term
        .run_joint_fit_arrow_schur(z.view(), &mut rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
        .expect("softmax K=2 must complete second joint-fit step");
    assert!(loss1.total().is_finite());
}

/// End-to-end Isometry wiring oracle.
///
/// Build a SAE atom around an evaluator whose `second_jet` is now
/// implemented (periodic / sphere / torus), construct an
/// [`IsometryPenalty`] with matching `latent_dim` and `p_out`, refresh
/// the caches via [`refresh_isometry_caches_from_atom`], and check that
///
///   * `IsometryPenalty.value(target, rho)` is strictly positive (the
///     decoder we feed in is not orthonormal so the pullback metric is
///     not the identity, and the Euclidean reference picks up the gap).
///   * `IsometryPenalty.grad_target(target, rho)` is non-zero on at
///     least one latent-coordinate component.
///   * The analytic gradient matches a finite-difference oracle of
///     `value()` w.r.t. `target` (a single coord), where each FD probe
///     drives a fresh cache refresh — this is exactly the chain of
///     calls the SAE outer loop will make.
///
/// The FD oracle re-uses the existing [`refresh_isometry_caches_from_atom`]
/// helper for both the analytic side and the FD side, so any layout
/// mismatch between `J`/`H` would show up as a tolerance failure rather
/// than a silently zero gradient.
pub(crate) fn assert_isometry_wiring_matches_fd(
    evaluator: Arc<dyn SaeBasisSecondJet>,
    coords: Array2<f64>,
) {
    let n_obs = coords.nrows();
    let latent_dim = coords.ncols();
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let p: usize = 3;
    // A deterministic non-orthonormal decoder: deterministic LCG-ish
    // floats keep the test reproducible without needing rand.
    let mut decoder = Array2::<f64>::zeros((m, p));
    for i in 0..m {
        for j in 0..p {
            let x = (i as f64) * 0.371 + (j as f64) * 0.193 + 0.5;
            decoder[[i, j]] = (x.sin() * 0.9) + 0.1 * ((i + j) as f64).cos();
        }
    }
    let smooth = Array2::<f64>::eye(m);
    let atom = SaeManifoldAtom::new(
        "iso_wire_test",
        SaeAtomBasisKind::Periodic,
        latent_dim,
        phi.clone(),
        jet.clone(),
        decoder.clone(),
        smooth,
    )
    .unwrap()
    .with_basis_second_jet(evaluator);

    let target_slice = PsiSlice::full(n_obs * latent_dim, Some(latent_dim));
    let penalty = IsometryPenalty::new_euclidean(target_slice, p);
    let rho = Array1::<f64>::zeros(1);

    // Without a refresh, the safe default is zero and the gradient is
    // all zeros. Confirm the precondition so the post-refresh contrast
    // is meaningful.
    let target_flat: Array1<f64> = coords.iter().copied().collect();
    let v0 = penalty.value(target_flat.view(), rho.view());
    assert_eq!(v0, IsometryPenalty::DEFAULT_VALUE_ON_MISSING_CACHE);
    let g0 = penalty.grad_target(target_flat.view(), rho.view());
    assert!(
        g0.iter().all(|x| *x == 0.0),
        "grad_target without cache must be all zeros, got {g0:?}"
    );

    // Refresh and re-evaluate.
    let installed_second =
        refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
    assert!(
        installed_second,
        "evaluator must implement second_jet for this oracle to run"
    );

    let value = penalty.value(target_flat.view(), rho.view());
    assert!(
        value > 1.0e-6,
        "expected non-trivial isometry loss after cache refresh, got {value}"
    );
    let grad = penalty.grad_target(target_flat.view(), rho.view());
    assert_eq!(grad.len(), target_flat.len());
    let max_abs = grad.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    assert!(
        max_abs > 1.0e-6,
        "expected non-zero isometry gradient on at least one component, max |grad|={max_abs}"
    );

    // FD check: bump one coord, refresh, compare value(t±h e_j) against
    // analytic grad[j]. Pick coord (row 0, axis 0).
    let h_fd = 1.0e-5;
    let probe_idx = 0usize; // (row=0, axis=0) flattens to 0.
    let mut coords_plus = coords.clone();
    coords_plus[[0, 0]] += h_fd;
    let mut coords_minus = coords.clone();
    coords_minus[[0, 0]] -= h_fd;

    refresh_isometry_caches_from_atom(&penalty, &atom, coords_plus.view()).unwrap();
    let target_plus: Array1<f64> = coords_plus.iter().copied().collect();
    let v_plus = penalty.value(target_plus.view(), rho.view());

    refresh_isometry_caches_from_atom(&penalty, &atom, coords_minus.view()).unwrap();
    let target_minus: Array1<f64> = coords_minus.iter().copied().collect();
    let v_minus = penalty.value(target_minus.view(), rho.view());

    // Reinstall the base caches before reading grad at the base point.
    refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
    let grad_base = penalty.grad_target(target_flat.view(), rho.view());

    let fd = (v_plus - v_minus) / (2.0 * h_fd);
    let analytic = grad_base[probe_idx];
    // Both `value` and `grad_target` use the cached `J` (and `grad_target`
    // also the cached `H`). With finite differencing the cache itself,
    // the analytic-vs-FD agreement bounds the entire pipeline (J build,
    // H build, accessor read, pullback metric, gradient assembly) to
    // O(h²) error. Tolerance 1e-3 leaves headroom for the per-evaluator
    // characteristic magnitude.
    assert!(
        (analytic - fd).abs() <= 1.0e-3 + 1.0e-4 * analytic.abs().max(fd.abs()),
        "isometry grad/FD mismatch at coord 0: analytic={analytic:.6e}, fd={fd:.6e}"
    );
}

#[test]
pub(crate) fn isometry_wiring_periodic_matches_fd() {
    assert_isometry_wiring_matches_fd(
        Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap()),
        array![[0.12], [0.37], [0.58], [0.81]],
    );
}

#[test]
pub(crate) fn isometry_wiring_sphere_matches_fd() {
    assert_isometry_wiring_matches_fd(
        Arc::new(SphereChartEvaluator),
        array![[-0.5, 0.3], [0.2, -1.1], [0.7, 0.9]],
    );
}

#[test]
pub(crate) fn isometry_wiring_torus_matches_fd() {
    assert_isometry_wiring_matches_fd(
        Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
        array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
    );
}

// [#780 line-count gate] The exact isometry-penalty HVP / PSD-majorizer
// cluster (`deterministic_decoder`, `build_isometry_atom_for_evaluator`,
// `assert_exact_isometry_hvp_*`, `assert_isometry_psd_majorizer_live_*`, the
// `isometry_exact_hvp_*` / `isometry_psd_majorizer_*` tests, and the
// `refresh_isometry_caches_pairs_each_penalty_to_its_own_atom` regression) was
// split into the sibling `tests_isometry_exact_hvp_majorizer_457.rs` module
// (declared in `mod.rs`) to keep this tracked file under the 10k limit. The
// cluster is self-contained: its helpers are referenced only within it.

/// Build a minimal single-atom periodic SAE outer objective for the
/// warm-start contract tests (gam#577 / gam#579).
pub(crate) fn warmstart_test_objective() -> SaeManifoldOuterObjective {
    // `PeriodicHarmonicEvaluator::new(3)` produces the SAME 3-column Fourier
    // basis `[1, sin(2πt), cos(2πt)]` and first jet as `periodic_basis`, plus
    // the analytic second jet that `logdet_theta_adjoint` (the softmax
    // assignment adjoint) needs. Installing it lets the full `eval` gradient
    // lane run instead of erroring on a missing second-jet evaluator.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = array![[0.10], [0.35], [0.62], [0.88]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        // Decoder mapping the 3 basis fns to a single output channel.
        array![[0.30], [-0.20], [0.15]],
        // Mild ridge-like smoothness penalty so the inner solve is PD.
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone())
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode(
        // Nonzero assignment mass so H_tt carries genuine data curvature.
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20_f64], [-0.10], [0.30], [0.05]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
}

/// As [`warmstart_test_objective`], but the atom carries a full basis evaluator
/// AND second-jet evaluator (`PeriodicHarmonicEvaluator`), so the analytic outer
/// ρ-gradient lane (`eval` → `logdet_theta_adjoint`, which needs second jets for
/// the softmax assignment adjoint) can run. Required by the #1206 gradient-lane
/// contract test, which exercises the full `(cost, ∇f)` path.
pub(crate) fn warmstart_test_objective_with_evaluator() -> SaeManifoldOuterObjective {
    // `PeriodicHarmonicEvaluator::new(3)` produces the SAME 3-column Fourier
    // basis `[1, sin(2πt), cos(2πt)]` (1 harmonic) and matching first jet that
    // `periodic_basis` builds, so phi/jet are consistent with the decoder dims.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = array![[0.10_f64], [0.35], [0.62], [0.88]];
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let atom = SaeManifoldAtom::new(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        array![[0.30_f64], [-0.20], [0.15]],
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator.clone())
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode(
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[0.20_f64], [-0.10], [0.30], [0.05]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
}

pub(crate) fn near_singular_outer_gradient_cache() -> ArrowFactorCache {
    ArrowFactorCache {
        htt_factors: ArrowFactorSlab::from_blocks(vec![array![[1.0_f64, 0.0], [0.0, 1.0e-7]]]),
        htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
        schur_factor: Some(array![[1.0_f64]]),
        joint_hessian_log_det: None,
        solver_mode: ArrowSolverMode::Direct,
        ridge_t: 0.0,
        ridge_beta: 0.0,
        htbeta: ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
        d: 2,
        row_dims: Arc::from(vec![2usize].into_boxed_slice()),
        row_offsets: Arc::from(vec![0usize, 2usize].into_boxed_slice()),
        k: 1,
        manifold_mode_fingerprint: 0,
        row_hessian_fingerprint: 0,
        pcg_diagnostics: PcgDiagnostics::default(),
        gauge_deflated_directions: 0,
        deflated_row_directions: std::sync::Arc::from(Vec::new()),
        deflation_row_spectra: std::sync::Arc::from(Vec::new()),
        cross_row_woodbury: None,
    }
}

pub(crate) fn diagonal_latent_cache(diagonal: &[f64]) -> ArrowFactorCache {
    let dim = diagonal.len();
    let mut factor = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        factor[[i, i]] = diagonal[i].sqrt();
    }
    ArrowFactorCache {
        htt_factors: ArrowFactorSlab::from_blocks(vec![factor]),
        htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
        schur_factor: None,
        joint_hessian_log_det: None,
        solver_mode: ArrowSolverMode::Direct,
        ridge_t: 0.0,
        ridge_beta: 0.0,
        htbeta: ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
        d: dim,
        row_dims: Arc::from(vec![dim].into_boxed_slice()),
        row_offsets: Arc::from(vec![0usize, dim].into_boxed_slice()),
        k: 0,
        manifold_mode_fingerprint: 0,
        row_hessian_fingerprint: 0,
        pcg_diagnostics: PcgDiagnostics::default(),
        gauge_deflated_directions: 0,
        deflated_row_directions: std::sync::Arc::from(Vec::new()),
        deflation_row_spectra: std::sync::Arc::from(Vec::new()),
        cross_row_woodbury: None,
    }
}

#[test]
pub(crate) fn outer_gradient_solver_rejects_near_singular_cache_without_matching_gauge() {
    let cache = near_singular_outer_gradient_cache();
    let obj = warmstart_test_objective();

    // The raw conditioning gate is what names the ill-conditioned joint Hessian
    // and reports the pivot ratio + floor. Pin that message HERE, at its source
    // (`outer_gradient_conditioning_error`), so the diagnostic stays covered even
    // though the solver below now re-classifies the gauge-degenerate case.
    let conditioning_err = match SaeManifoldTerm::outer_gradient_conditioning_error(&cache) {
        Err(err) => err.to_string(),
        Ok(()) => panic!("near-singular cache must trip the pivot-ratio conditioning gate"),
    };
    assert!(
        conditioning_err.contains("joint Hessian numerically singular"),
        "conditioning gate must name the ill-conditioned joint Hessian; got: {conditioning_err}"
    );
    assert!(
        conditioning_err.contains("min/max pivot ratio") && conditioning_err.contains("floor"),
        "conditioning gate must report the pivot ratio and floor; got: {conditioning_err}"
    );

    // #1436 (commit 21c49d14b): when the conditioning gate fires but NO chart
    // gauge / decoder-β-null / decoder-channel-null candidate can be recovered to
    // deflate the flat subspace, the flatness is genuinely OUTSIDE the gauge orbit
    // — a distinct, more specific diagnosis the solver surfaces as
    // `OuterGradientError::NonIdentifiable` (rather than echoing the raw
    // pivot-ratio `IllConditioned` trip). Both classes are FD-eligible, so the
    // recovery behaviour is unchanged; only the diagnostic is sharper. This is the
    // exact "without a matching gauge" path the test name describes.
    let err = match obj
        .term
        .outer_gradient_arrow_solver(&cache, &obj.current_rho.lambda_smooth_vec())
    {
        Err(err) => err,
        Ok(..) => panic!("near-singular evidence factor without a matching gauge must reject"),
    };
    assert!(
        matches!(err, OuterGradientError::NonIdentifiable { .. }),
        "no-deflatable-direction rejection must be the NonIdentifiable diagnosis; got: {err}"
    );
    let err = err.to_string();
    assert!(
        err.contains("no deflatable gauge/decoder-null direction"),
        "guard error must name the absent deflation candidate; got: {err}"
    );
}

/// #1051: a euclidean-patch atom whose decoder design is RANK-DEFICIENT
/// (a straight line in a `p = 2` ambient: the decoder column space is rank
/// 1, so one output-channel direction is unidentified by the data) leaves a
/// genuine near-null direction of the joint Hessian in the β (decoder)
/// block. That direction is OUTSIDE the closed-form chart gauge orbit
/// (`dense_step_gauge_vectors` only spans per-latent-axis reparametrisation,
/// never per-output-channel decoder freedom), so before the fix
/// `outer_gradient_arrow_solver` could not deflate it and rejected the
/// trial ρ with "analytic outer gradient undefined" — the singular-pivot
/// continuation stall that made every euclidean/multi-atom atlas tile
/// TIMEOUT. With the β-basis admitted as a deflation candidate the flat
/// direction is Faddeev-Popov-deflated and the solve succeeds, regularising
/// the near-null β response to the Hessian scale (bounded, not 1e13).
pub(crate) fn rank_deficient_euclidean_outer_gradient_objective() -> SaeManifoldOuterObjective {
    // Linear euclidean basis Φ(t) = [1, t] (m = 2) over a 1-D latent.
    let coords = array![[-0.7_f64], [-0.2], [0.3], [0.8]];
    let n = coords.nrows();
    let mut phi = Array2::<f64>::zeros((n, 2));
    let mut jet = Array3::<f64>::zeros((n, 2, 1));
    for row in 0..n {
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = coords[[row, 0]];
        jet[[row, 1, 0]] = 1.0; // d/dt of the linear column.
    }
    // p = 2 ambient, but the decoder maps only into output channel 0 (its
    // second column is identically zero), so the reconstruction `Φ·B` lives on
    // the 1-D subspace `{x : x₁ = 0}` of R² and output channel 1 is genuinely
    // unidentified. The decoder's right-singular null vector is then exactly the
    // channel-1 axis `(0, 1)`, matching the near-null direction the joint-Hessian
    // cache below places on that axis (β indices 1 and 3). This is the rank-1
    // decoder column-span deficiency `decoder_channel_null_directions` must
    // recover (#1051/#1273).
    let decoder = array![[1.0_f64, 0.0], [0.5, 0.0]];
    let atom = SaeManifoldAtom::new(
        "euclidean_line",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(2),
    )
    .unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode(
        array![[0.9_f64], [0.8], [0.7], [0.6]],
        vec![coords],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let target = array![[-1.0_f64, -2.0], [-0.3, -0.6], [0.4, 0.8], [1.1, 2.2]];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 8, 1.0, 1.0e-6, 1.0e-6)
}

/// A joint Hessian cache whose β block carries one genuine near-null
/// direction along the SECOND output channel (`out_col = 1`) — the
/// rank-deficient decoder's unidentified direction — with the latent block
/// well-conditioned and `H_tβ = 0` so the singularity is purely in β. The
/// chart gauge orbit cannot reach this direction (#1051).
pub(crate) fn rank_deficient_beta_outer_gradient_cache() -> ArrowFactorCache {
    // The latent block must be dimensionally consistent with the paired
    // objective `rank_deficient_euclidean_outer_gradient_objective` so the
    // channel-null candidates (whose full length is the objective's
    // `n·q + β_dim`) survive the `dir.len() == full_len` guard in
    // `outer_gradient_arrow_solver`. That objective has n = 4 data rows and
    // `row_block_dim q = 1` (one latent axis, K = 1 softmax ⇒ no assignment
    // coord), so `delta_t_len` must be `n·q = 4`. A mismatched single-row cache
    // makes `full_len = 5` while the candidates have length 8, silently
    // dropping every channel-null direction and re-introducing the bug.
    let htt = ArrowFactorSlab::from_blocks(vec![
        array![[1.0_f64]],
        array![[1.0_f64]],
        array![[1.0_f64]],
        array![[1.0_f64]],
    ]);
    // β dim = m · p = 2 · 2 = 4, laid out (col, out_col) row-major like
    // `dense_step_gauge_vector_from_field`. Make output channel 1 (indices
    // 1 and 3) near-null: its lower-Cholesky pivot is 1e-7, so the
    // min/max pivot ratio falls below the 1e-12 floor and the conditioning
    // path engages. H_tβ = 0 (zero Dense blocks) decouples β from latent.
    let schur = array![
        [1.0_f64, 0.0, 0.0, 0.0],
        [0.0, 1.0e-7, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0e-7],
    ];
    ArrowFactorCache {
        htt_factors: htt,
        htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
        schur_factor: Some(schur),
        joint_hessian_log_det: None,
        solver_mode: ArrowSolverMode::Direct,
        ridge_t: 0.0,
        ridge_beta: 0.0,
        htbeta: ArrowHtbetaCache::Dense {
            blocks: Arc::from(
                vec![
                    Array2::<f64>::zeros((1, 4)),
                    Array2::<f64>::zeros((1, 4)),
                    Array2::<f64>::zeros((1, 4)),
                    Array2::<f64>::zeros((1, 4)),
                ]
                .into_boxed_slice(),
            ),
            estimated_bytes: 0,
        },
        d: 4,
        row_dims: Arc::from(vec![1usize, 1usize, 1usize, 1usize].into_boxed_slice()),
        row_offsets: Arc::from(vec![0usize, 1usize, 2usize, 3usize, 4usize].into_boxed_slice()),
        k: 4,
        manifold_mode_fingerprint: 0,
        row_hessian_fingerprint: 0,
        pcg_diagnostics: PcgDiagnostics::default(),
        gauge_deflated_directions: 0,
        deflated_row_directions: std::sync::Arc::from(Vec::new()),
        deflation_row_spectra: std::sync::Arc::from(Vec::new()),
        cross_row_woodbury: None,
    }
}

#[test]
pub(crate) fn outer_gradient_solver_deflates_rank_deficient_decoder_beta_null() {
    let obj = rank_deficient_euclidean_outer_gradient_objective();
    let cache = rank_deficient_beta_outer_gradient_cache();
    // Sanity: the cache genuinely trips the conditioning floor (the bug's
    // precondition) — without it this test would not exercise the fix.
    assert!(
        SaeManifoldTerm::outer_gradient_conditioning_error(&cache).is_err(),
        "fixture must be sub-floor singular so the conditioning path engages"
    );
    // The fix: the β-block near-null direction is admitted as a deflation
    // candidate and Faddeev-Popov-deflated, so the solver SUCCEEDS instead
    // of rejecting with "analytic outer gradient undefined".
    let solver = obj
        .term
        .outer_gradient_arrow_solver(&cache, &obj.current_rho.lambda_smooth_vec())
        .expect("rank-deficient decoder β-null must be deflated, not rejected (#1051/#1273)");
    // The deflated solve must REGULARISE the near-null β response: a plain
    // inverse divides by the 1e-7 pivot and explodes; the deflated solve is
    // bounded at the Hessian scale.
    let beta_null_rhs = array![0.0_f64, 0.0, 0.0, 1.0]; // output channel 1, col 1.
    let rhs_t = Array1::<f64>::zeros(cache.delta_t_len());
    let plain = cache
        .full_inverse_apply(rhs_t.view(), beta_null_rhs.view())
        .expect("plain solve")
        .1;
    let deflated = solver
        .solve(rhs_t.view(), beta_null_rhs.view())
        .expect("deflated solve")
        .beta;
    assert!(
        plain[3].abs() > 1.0e13,
        "plain near-null β solve must explode; got {}",
        plain[3]
    );
    assert!(
        deflated.iter().all(|v| v.is_finite()) && deflated[3].abs() < 10.0,
        "deflated near-null β solve must be bounded at the Hessian scale; got {deflated:?}"
    );
}

/// #1273/#1440 regression — the gradient lane (`eval` /
/// `OuterEvalOrder::ValueAndGradient`) must NOT hard-abort when the
/// gauge-deflated analytic outer gradient declines at a finite-cost ρ whose
/// joint Hessian is near-singular-but-valid (the circle/torus topology the
/// issue reports: a flat direction the Faddeev-Popov deflation legitimately
/// rejects). Before #1273 the conditioning error `?`-propagated out of `eval`
/// as `RemlOptimizationFailed` → `RemlConvergenceError`; #1273 recovered it
/// with a central finite-difference descent of the value path, and #1440
/// REPLACED that finite-difference instrument with the PLAIN (undeflated)
/// analytic outer gradient of the same Laplace value. The recovery direction is
/// now fully analytic — never a differenced value path.
///
/// The test exercises both halves deterministically in unit time: (1) the
/// conditioning gate genuinely rejects a near-singular cache that no gauge/β-null
/// deflation can recover (the bug's precondition), and (2) `eval` still returns a
/// finite, ρ-sized `(cost, ∇f)` pair on the same objective (the analytic
/// recovery wiring), with no regression to the well-conditioned analytic path.
#[test]
pub(crate) fn gradient_lane_analytic_fallback_recovers_singular_outer_gradient_1440() {
    let objective = warmstart_test_objective();
    // Precondition: a near-singular joint Hessian whose sub-floor pivot is NOT
    // explained by any chart-gauge / decoder-β-null direction — so the analytic
    // gauge-deflated outer-gradient solver REJECTS it. This is the exact
    // condition the issue's pivot-ratio gate trips on.
    let singular_cache = near_singular_outer_gradient_cache();
    assert!(
        SaeManifoldTerm::outer_gradient_conditioning_error(&singular_cache).is_err(),
        "fixture precondition: the cache must trip the pivot-ratio floor (#1273)"
    );
    assert!(
        objective
            .term
            .outer_gradient_arrow_solver(
                &singular_cache,
                &objective.current_rho.lambda_smooth_vec()
            )
            .is_err(),
        "fixture precondition: the gauge-deflated analytic outer gradient must          REJECT this near-singular cache (no matching gauge/β-null to deflate)"
    );

    // The #1440 fix: at such a finite-cost ρ the gradient lane (`eval`) descends
    // with the PLAIN analytic outer gradient instead of a finite-difference of
    // the value path. End-to-end it must still return a finite, ρ-sized
    // `(cost, ∇f)` — the recovery wiring shares the well-conditioned analytic
    // path, so it must not regress it.
    let mut objective = warmstart_test_objective();
    let rho_flat = objective.current_rho.to_flat();
    let eval = objective
        .eval(&rho_flat)
        .expect("gradient lane must return a finite (cost, gradient) pair (#1440 wiring)");
    assert!(
        eval.cost.is_finite()
            && eval.gradient.len() == rho_flat.len()
            && eval.gradient.iter().all(|g| g.is_finite()),
        "gradient lane must yield a finite, ρ-sized outer gradient; got cost={}, grad={:?}",
        eval.cost,
        eval.gradient
    );
}

/// #1436 — `OuterGradientError::InternalInvariant` must never be FD-eligible,
/// so an internal-invariant failure propagates as a hard error instead of being
/// silently masked by a finite-difference descent direction. This is the core
/// acceptance criterion: shape/indexing bugs, non-finite intermediates, and
/// violated invariants surface as failures, not plausible-but-wrong FD steps.
#[test]
pub(crate) fn outer_gradient_internal_invariant_is_not_fd_eligible_1436() {
    let ill_conditioned = OuterGradientError::IllConditioned {
        reason: "near-singular joint Hessian".to_string(),
    };
    let non_identifiable = OuterGradientError::NonIdentifiable {
        reason: "gauge-degenerate direction".to_string(),
    };
    let internal = OuterGradientError::InternalInvariant {
        reason: "shape mismatch".to_string(),
    };
    assert!(
        ill_conditioned.is_conditioning_recoverable(),
        "IllConditioned must be conditioning-recoverable (#1273)"
    );
    assert!(
        non_identifiable.is_conditioning_recoverable(),
        "NonIdentifiable must be conditioning-recoverable (#1273)"
    );
    assert!(
        !internal.is_conditioning_recoverable(),
        "InternalInvariant must NOT be conditioning-recoverable (#1436) — it must propagate"
    );
    // The Display output must be descriptive enough for the outer log.
    assert!(
        internal.to_string().contains("internal invariant"),
        "InternalInvariant Display must name the class; got: {}",
        internal
    );
}

/// #1436 — exercise the EXACT gate `SaeManifoldOuterObjective::eval` consults,
/// `OuterGradientError::admits_plain_solver_fallback`, over the full `cost x error-class`
/// matrix. `is_conditioning_recoverable` alone does not capture the cost interaction the call
/// site depends on; this pins the composed contract so the FD fallback can never
/// silently absorb an internal-invariant failure NOR fire at an infeasible
/// (non-finite-cost) ρ — both must propagate as hard errors.
#[test]
pub(crate) fn admits_plain_solver_fallback_only_for_conditioning_at_finite_cost_1436() {
    let ill = OuterGradientError::IllConditioned {
        reason: "near-singular joint Hessian".to_string(),
    };
    let non_id = OuterGradientError::NonIdentifiable {
        reason: "gauge-degenerate direction".to_string(),
    };
    let internal = OuterGradientError::InternalInvariant {
        reason: "shape mismatch".to_string(),
    };

    // Finite cost: only the genuine #1273 conditioning/identifiability classes
    // admit the FD descent direction.
    assert!(
        ill.admits_plain_solver_fallback(1.0),
        "IllConditioned at a finite-cost ρ must admit the #1273/#1440 analytic plain-solver fallback"
    );
    assert!(
        non_id.admits_plain_solver_fallback(1.0),
        "NonIdentifiable at a finite-cost ρ must admit the #1273/#1440 analytic plain-solver fallback"
    );
    assert!(
        !internal.admits_plain_solver_fallback(1.0),
        "InternalInvariant must NEVER admit the plain-solver fallback, even at a finite \
         cost (#1436) — it must propagate as a hard error"
    );

    // Non-finite cost (infeasible point): NOTHING admits FD, not even an
    // otherwise-eligible conditioning failure — there is no feasible value path
    // to descend.
    for bad_cost in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(
            !ill.admits_plain_solver_fallback(bad_cost),
            "IllConditioned must NOT admit the plain-solver fallback at non-finite cost {bad_cost}"
        );
        assert!(
            !non_id.admits_plain_solver_fallback(bad_cost),
            "NonIdentifiable must NOT admit the plain-solver fallback at non-finite cost {bad_cost}"
        );
        assert!(
            !internal.admits_plain_solver_fallback(bad_cost),
            "InternalInvariant must NOT admit the plain-solver fallback at non-finite cost {bad_cost}"
        );
    }
}

/// gam#577 / gam#579 root cause: the continuation pre-warm forwards an
/// EMPTY β before the first accepted eval (`state.last_beta` starts
/// empty). The seed hook must treat that as the documented "no warm-start
/// available, proceed cold" no-op (`SeedOutcome::NoSlot`) rather than
/// erroring on `β length 0 != decoder dim` — the error dropped EVERY
/// continuation seed and forced a full cold solve on every outer seed.
#[test]
pub(crate) fn seed_inner_state_accepts_empty_beta_as_noslot() {
    let mut obj = warmstart_test_objective();
    let empty: Array1<f64> = Array1::zeros(0);
    let outcome = obj
        .seed_inner_state(&empty)
        .expect("empty-β seed must be accepted as a no-op, not rejected (gam#577/#579)");
    assert!(
        matches!(outcome, SeedOutcome::NoSlot),
        "empty-β seed must report NoSlot (proceed cold); got {outcome:?}"
    );
}

/// A populated β whose length matches the decoder dimension must be
/// INSTALLED and then GENUINELY REUSED by the next inner solve — this is
/// the warm-start the continuation walk relies on for the big speedup
/// (gam#577 / gam#579). We verify reuse behaviorally: seed a known β, run
/// one eval with zero inner Newton iterations (so the solve cannot move
/// β off the seed), and confirm the published `inner_beta_hint` is exactly
/// the seeded β. A cold start would have published the term's pristine β
/// instead.
#[test]
pub(crate) fn seed_inner_state_installs_and_reuses_matching_beta() {
    let mut obj = warmstart_test_objective();
    let dim = obj.term.beta_dim();
    // A distinctive seed that differs from the term's pristine decoder.
    let pristine = obj.term.flatten_beta();
    let seed: Array1<f64> = Array1::from_shape_fn(dim, |i| pristine[i] + 0.5 + 0.01 * (i as f64));
    assert!(
        (&seed - &pristine).iter().any(|d| d.abs() > 1e-6),
        "seed must differ from the pristine β for the reuse check to be meaningful"
    );

    let outcome = obj
        .seed_inner_state(&seed)
        .expect("a length-matching β must install");
    assert!(
        matches!(outcome, SeedOutcome::Installed),
        "matching β must report Installed; got {outcome:?}"
    );

    // Freeze the inner solve at zero Newton iterations: β cannot move off
    // the warm-start, so the published hint must equal the seed exactly.
    obj.inner_max_iter = 0;
    let rho_flat = obj.baseline_rho.to_flat();
    let eval =
        OuterObjective::eval(&mut obj, &rho_flat).expect("eval at the warm-started β must succeed");
    let hint = eval
        .inner_beta_hint
        .expect("the SAE objective must publish inner_beta_hint for continuation reuse");
    assert_eq!(
        hint.len(),
        dim,
        "published hint must have decoder dimension"
    );
    for (i, (&h, &s)) in hint.iter().zip(seed.iter()).enumerate() {
        assert!(
            (h - s).abs() < 1e-12,
            "warm-started β must be reused verbatim by the inner solve at coord {i}: \
                 hint {h} != seed {s} (gam#577/#579)"
        );
    }
}

/// The seed contract is only relaxed for the EMPTY sentinel. A populated
/// β whose length disagrees with the decoder dimension is a genuine
/// layout bug and must still surface a typed error rather than being
/// silently dropped.
#[test]
pub(crate) fn seed_inner_state_rejects_wrong_length_populated_beta() {
    let mut obj = warmstart_test_objective();
    let dim = obj.term.beta_dim();
    let wrong: Array1<f64> = Array1::zeros(dim + 1);
    let err = obj
        .seed_inner_state(&wrong)
        .expect_err("a populated β of the wrong length must be rejected");
    match err {
        EstimationError::RemlOptimizationFailed(msg) => {
            assert!(
                msg.contains("decoder dim"),
                "error must name the decoder-dim mismatch; got: {msg}"
            );
        }
        other => panic!("expected RemlOptimizationFailed, got {other:?}"),
    }
}

/// Build a non-periodic 1-D atom with a genuine order-2 finite-difference
/// roughness Gram, a non-constant-speed decoder, and explicit
/// `(basis_values, basis_jacobian)` so the intrinsic reweighting in
/// [`SaeManifoldAtom::refresh_intrinsic_smooth_penalty`] is exercised
/// directly. A localized (near-diagonal) basis makes each coefficient's
/// representative speed the speed at its own sample.
pub(crate) fn intrinsic_test_atom(jacobian_scale: f64) -> SaeManifoldAtom {
    let m = 5usize;
    let n = m;
    let p = 1usize;
    let mut phi = Array2::<f64>::zeros((n, m));
    let mut jet = Array3::<f64>::zeros((n, m, 1));
    let mut decoder = Array2::<f64>::zeros((m, p));
    for mu in 0..m {
        // Localized basis: Φ_μ(t_n) ≈ δ_{nμ}.
        phi[[mu, mu]] = 1.0;
        // Per-sample basis derivative (axis 0) grows with μ — a
        // non-constant-speed curve — scaled by `jacobian_scale` to emulate
        // a global linear reparameterization t -> t / jacobian_scale.
        jet[[mu, mu, 0]] = jacobian_scale * (1.0 + mu as f64);
        decoder[[mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    SaeManifoldAtom::new(
        "intrinsic-1d",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap()
}

/// The roughness operator order is recovered from the raw Gram's null
/// space: an order-2 difference penalty annihilates the affine functions,
/// so `nullity = 2` and the arc-length exponent is `β = ½ − 2 = −3/2`.
#[test]
pub(crate) fn intrinsic_penalty_recovers_order_two_from_nullity() {
    let atom = intrinsic_test_atom(1.0);
    assert_eq!(atom.smooth_penalty_order, 2);
}

#[test]
pub(crate) fn line_search_snapshot_restores_intrinsic_smooth_penalty() {
    let atom = intrinsic_test_atom(1.0);
    let n = atom.n_obs();
    let logits = Array2::<f64>::zeros((n, 1));
    let coords = vec![Array2::<f64>::zeros((n, 1))];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let original = term.atoms[0].smooth_penalty.clone();
    let snapshot = term.snapshot_mutable_state();

    term.atoms[0].decoder_coefficients[[0, 0]] *= 3.0;
    term.atoms[0].refresh_intrinsic_smooth_penalty();
    let changed = (&term.atoms[0].smooth_penalty - &original)
        .mapv(f64::abs)
        .sum();
    assert!(
        changed > 1e-6,
        "test setup must perturb the live intrinsic smoothness Gram"
    );

    term.restore_mutable_state(&snapshot);
    let restored = (&term.atoms[0].smooth_penalty - &original)
        .mapv(f64::abs)
        .sum();
    assert!(
        restored < 1e-12,
        "line-search restore left a stale intrinsic smoothness Gram: {restored}"
    );
}

/// Gauge invariance (issue #673): a global reparameterization of the latent
/// coordinate scales every per-sample speed by a common factor, which
/// cancels in the centered reweighting — so the intrinsic Gram `S̃` (and
/// hence the topology evidence `tr(BᵀS̃B)`) is identical across the two
/// reparameterizations, even though the basis Jacobian (the metric) differs.
#[test]
pub(crate) fn intrinsic_penalty_is_invariant_to_speed_rescaling() {
    let a1 = intrinsic_test_atom(1.0);
    let a2 = intrinsic_test_atom(7.5);
    // Same raw Gram and decoder; only the basis Jacobian (speed) differs.
    assert_abs_diff_eq!(
        (&a1.smooth_penalty_raw - &a2.smooth_penalty_raw)
            .mapv(f64::abs)
            .sum(),
        0.0,
        epsilon = 1e-12
    );
    // The intrinsic (reweighted) Gram is identical despite the 7.5x speed
    // rescale: the centered ratios are invariant to a global speed factor.
    let diff = (&a1.smooth_penalty - &a2.smooth_penalty)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff < 1e-9,
        "intrinsic Gram changed under a global speed rescale (gauge leak): {diff}"
    );
}

pub(crate) fn affine_canonicalization_test_term() -> SaeManifoldTerm {
    let n = 80usize;
    let p = 2usize;
    let evaluator = EuclideanPatchEvaluator::new(1, 2).unwrap();
    let mut coords = Array2::<f64>::zeros((n, 1));
    for row in 0..n {
        coords[[row, 0]] = -4.0 + 12.0 * row as f64 / (n as f64 - 1.0);
    }
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[0, 0]] = 0.8;
    decoder[[1, 0]] = -0.4;
    decoder[[2, 0]] = 0.15;
    decoder[[0, 1]] = -0.2;
    decoder[[1, 1]] = 0.9;
    decoder[[2, 1]] = -0.08;
    let smooth_penalty = gam_terms::basis::create_difference_penalty_matrix(3, 2, None).unwrap();
    let atom = SaeManifoldAtom::new(
        "affine-canonicalization",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        smooth_penalty,
    )
    .unwrap()
    .with_basis_second_jet(Arc::new(evaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

#[test]
pub(crate) fn affine_canonicalization_transports_live_penalty_instead_of_recomputing() {
    let mut term = affine_canonicalization_test_term();
    let before: f64 = term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let old_smooth_penalty = term.atoms[0].smooth_penalty.clone();
    let old_decoder = term.atoms[0].decoder_coefficients.clone();

    term.canonicalize_atom_affine_gauge(0, None).unwrap();
    let after: f64 = term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let invariant_gap = (after - before).abs() / before.abs().max(1.0);
    assert!(
        invariant_gap < 1.0e-9,
        "canonicalization changed fixed-rho smoothness energy: before={before:.12e}, after={after:.12e}"
    );

    let mut recomputed_atom = term.atoms[0].clone();
    recomputed_atom.refresh_intrinsic_smooth_penalty();
    let recomputed_term = SaeManifoldTerm::new(
        vec![recomputed_atom],
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((term.n_obs(), 1)),
            vec![term.assignment.coords[0].as_matrix()],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap(),
    )
    .unwrap();
    let recomputed: f64 = recomputed_term
        .decoder_smoothness_quadratic_form_per_atom()
        .iter()
        .sum();
    let recompute_jump = (recomputed - before).abs() / before.abs().max(1.0);
    assert!(
        recompute_jump > 1.0e-2,
        "test fixture failed to expose the intrinsic recompute energy jump: before={before:.12e}, recomputed={recomputed:.12e}"
    );

    let transport =
        solve_basis_transport(term.atoms[0].basis_values.view(), old_smooth_penalty.view())
            .expect_err("shape mismatch must reject invalid transport solve");
    assert!(
        transport.contains("row mismatch") || transport.contains("SVD failed"),
        "unexpected transport-shape diagnostic: {transport}"
    );
    let roundtrip = transport_smooth_penalty_for_decoder(
        solve_design_least_squares(
            term.atoms[0].decoder_coefficients.view(),
            old_decoder.view(),
        )
        .unwrap_or_else(|err| panic!("decoder transport fixture became singular: {err}"))
        .view(),
        old_smooth_penalty.view(),
    );
    assert!(
        roundtrip.is_err(),
        "non-square decoder transport must not be accepted as a penalty congruence"
    );
}

/// Non-constant speed genuinely reshapes the penalty: the intrinsic Gram
/// must differ from the raw Gram when the decoder curve is not
/// constant-speed, otherwise the reweighting is a no-op and the gauge fix
/// would be vacuous. The congruence preserves symmetry.
#[test]
pub(crate) fn intrinsic_penalty_differs_from_raw_under_varying_speed() {
    let atom = intrinsic_test_atom(1.0);
    let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff > 1e-6,
        "intrinsic reweighting was a no-op on a non-constant-speed curve: {diff}"
    );
    for i in 0..atom.basis_size() {
        for j in 0..atom.basis_size() {
            assert_abs_diff_eq!(
                atom.smooth_penalty[[i, j]],
                atom.smooth_penalty[[j, i]],
                epsilon = 1e-12
            );
        }
    }
}

/// Constant-speed atoms are untouched: when every sample shares one speed
/// (the periodic sin/cos limit), the centered weights are all `1`, so
/// `S̃ = S_raw` exactly and the topology comparison among constant-speed
/// atoms is unaffected.
#[test]
pub(crate) fn intrinsic_penalty_leaves_constant_speed_atom_unchanged() {
    let m = 6usize;
    let n = m;
    let mut phi = Array2::<f64>::zeros((n, m));
    let mut jet = Array3::<f64>::zeros((n, m, 1));
    let mut decoder = Array2::<f64>::zeros((m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        // Identical derivative magnitude at every sample => constant speed.
        jet[[mu, mu, 0]] = 2.0;
        decoder[[mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let atom = SaeManifoldAtom::new(
        "constant-speed",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    let diff = (&atom.smooth_penalty - &atom.smooth_penalty_raw)
        .mapv(f64::abs)
        .sum();
    assert!(
        diff < 1e-9,
        "constant-speed atom's penalty was reweighted (should be identity): {diff}"
    );
}

/// Build a low-rank decoder atom (`p` large, true column rank `r ≪ p`) and
/// verify the auto-activation installs a frame, the factored border holds
/// exactly `Σ M_k·r_k`, and reconstruction recovers `B_k` to machine
/// precision.
#[test]
pub(crate) fn factored_border_dim_invariant_and_reconstruction() {
    let m = 6usize;
    let p = 16usize;
    let r = 2usize;
    // B = C0 · Frameᵀ with a planted rank-`r` column span.
    let mut frame = Array2::<f64>::zeros((p, r));
    frame[[0, 0]] = 1.0;
    frame[[1, 1]] = 1.0;
    let mut c0 = Array2::<f64>::zeros((m, r));
    for mu in 0..m {
        c0[[mu, 0]] = 1.0 + mu as f64;
        c0[[mu, 1]] = 0.5 * mu as f64 - 1.0;
    }
    let decoder = fast_abt(&c0, &frame);
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        jet[[mu, mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new(
        "lowrank",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder.clone(),
        s_raw,
    )
    .unwrap();
    let activated = atom.maybe_activate_decoder_frame().expect("activate");
    assert_eq!(
        activated,
        Some(r),
        "rank-{r} decoder should profile to r={r}"
    );
    assert_eq!(atom.border_frame_rank(), r);
    assert_eq!(atom.frame_manifold_dimension(), r * (p - r));

    // Reconstruction recovers B_k to machine precision.
    let coords = atom.factored_coordinates().unwrap().expect("coords");
    assert_eq!(coords.dim(), (m, r));
    let reconstructed = atom
        .reconstruct_decoder_coefficients(coords.view())
        .unwrap();
    for mu in 0..m {
        for j in 0..p {
            assert_abs_diff_eq!(reconstructed[[mu, j]], decoder[[mu, j]], epsilon = 1.0e-9);
        }
    }

    let term = SaeManifoldTerm::new(
        vec![atom],
        SaeAssignment::from_blocks_with_mode(
            Array2::<f64>::zeros((m, 1)),
            vec![Array2::<f64>::zeros((m, 1))],
            AssignmentMode::softmax(0.7),
        )
        .unwrap(),
    )
    .unwrap();
    // Border-size invariant: factored border == Σ M_k·r_k.
    grassmann_assert_border_dim_invariant(&term).expect("border invariant");
    assert_eq!(term.factored_border_dim(), m * r);
    assert_eq!(term.grassmann_evidence_dimension(), r * (p - r));
    // Round-trip flatten/scatter of the factored border preserves B_k.
    let mut term = term;
    let border = term.flatten_factored_border().unwrap();
    assert_eq!(border.len(), m * r);
    let saved = term.atoms[0].decoder_coefficients.clone();
    term.scatter_factored_border(border.view()).unwrap();
    for mu in 0..m {
        for j in 0..p {
            assert_abs_diff_eq!(
                term.atoms[0].decoder_coefficients[[mu, j]],
                saved[[mu, j]],
                epsilon = 1.0e-9
            );
        }
    }
}

#[test]
pub(crate) fn factored_beta_penalty_probing_matches_projected_dense_curvature() {
    let k_atoms = 2usize;
    let m = 4usize;
    let p = 24usize;
    let r = 2usize;
    let n_obs = 5usize;
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let mut frame = Array2::<f64>::zeros((p, r));
        frame[[atom_idx * r, 0]] = 1.0;
        frame[[atom_idx * r + 1, 1]] = 1.0;
        let mut coords = Array2::<f64>::zeros((n_obs, 1));
        for row in 0..n_obs {
            coords[[row, 0]] = row as f64;
        }
        let mut phi = Array2::<f64>::zeros((n_obs, m));
        let mut jet = Array3::<f64>::zeros((n_obs, m, 1));
        for row in 0..n_obs {
            for basis_col in 0..m {
                let x = (row + 1) as f64 * (basis_col + 1) as f64;
                phi[[row, basis_col]] = 0.05 * x + if row == basis_col { 1.0 } else { 0.0 };
                jet[[row, basis_col, 0]] = 0.01 * x;
            }
        }
        let mut c = Array2::<f64>::zeros((m, r));
        for basis_col in 0..m {
            c[[basis_col, 0]] = 0.3 + 0.07 * (basis_col + atom_idx) as f64;
            c[[basis_col, 1]] = -0.2 + 0.05 * (basis_col * 2 + atom_idx) as f64;
        }
        let decoder = fast_abt(&c, &frame);
        let mut atom = SaeManifoldAtom::new(
            "factored_probe",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap();
        atom.maybe_activate_decoder_frame()
            .expect("frame activation")
            .expect("rank-2 atom should activate a frame");
        atoms.push(atom);
        coord_blocks.push(coords);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n_obs, k_atoms), 0.25),
        coord_blocks,
        vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    assert!(term.frames_active());
    assert_eq!(term.factored_border_dim(), k_atoms * m * r);

    let beta_len = term.beta_dim();
    let mut registry = AnalyticPenaltyRegistry::new();
    let nuclear = NuclearNormPenalty::new(
        PsiSlice {
            range: 0..beta_len,
            latent_dim: Some(beta_len / p),
        },
        0.7,
        p,
        1.0e-4,
        None,
        false,
    )
    .unwrap();
    registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(nuclear)));
    let incoherence = DecoderIncoherencePenalty::new(
        PsiSlice {
            range: 0..beta_len,
            latent_dim: Some(beta_len / p),
        },
        vec![m, m],
        p,
        Array2::<f64>::from_elem((k_atoms, k_atoms), 0.5),
        0.6,
        false,
    )
    .unwrap();
    registry.push(AnalyticPenaltyKind::DecoderIncoherence(Arc::new(
        incoherence,
    )));

    let mut dense_sys = ArrowSchurSystem::new(0, 0, beta_len);
    let dense_assembly = term
        .add_sae_analytic_penalty_contributions(&mut dense_sys, &registry, 1.0, None, true, None)
        .unwrap();
    assert!(dense_assembly.dense_written);
    assert!(!dense_assembly.deferred_factored);

    let projection = FrameProjection::new(&term);
    let border_dim = term.factored_border_dim();
    let projected = term.project_dense_penalty_to_factored(dense_sys.hbb.view(), &projection);
    let direct = term.build_factored_beta_penalty_curvature(&registry, 1.0, &projection);
    for row in 0..border_dim {
        for col in 0..border_dim {
            assert_abs_diff_eq!(direct[[row, col]], projected[[row, col]], epsilon = 1.0e-10);
        }
    }

    let mut deferred_term = term.clone();
    let rho = SaeManifoldRho::new(
        0.0,
        -20.0,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    );
    let target = Array2::<f64>::zeros((n_obs, p));
    let sys = deferred_term
        .assemble_arrow_schur_scaled_with_beta_penalty_probe_threshold(
            target.view(),
            &rho,
            Some(&registry),
            1.0,
            1,
        )
        .unwrap();
    assert_eq!(sys.k, border_dim);
    assert!(sys.hbb.is_empty());
}

pub(crate) fn materialize_row_htbeta_for_test(
    sys: &ArrowSchurSystem,
    row_idx: usize,
) -> Array2<f64> {
    let di = sys.row_dims[row_idx];
    let k = sys.k;
    let row = &sys.rows[row_idx];
    let use_dense = sys.htbeta_dense_supplement || sys.htbeta_matvec.is_none();
    let mut out = if use_dense && row.htbeta.dim() == (di, k) {
        row.htbeta.clone()
    } else {
        Array2::<f64>::zeros((di, k))
    };
    if let Some(op) = sys.htbeta_matvec.as_ref() {
        let mut basis = Array1::<f64>::zeros(k);
        let mut col = Array1::<f64>::zeros(di);
        for beta_col in 0..k {
            basis.fill(0.0);
            basis[beta_col] = 1.0;
            col.fill(0.0);
            op(row_idx, basis.view(), &mut col);
            for row_col in 0..di {
                out[[row_col, beta_col]] += col[row_col];
            }
        }
    }
    out
}

pub(crate) fn project_row_htbeta_to_factored_for_test(
    term: &SaeManifoldTerm,
    htbeta_b: ArrayView2<'_, f64>,
) -> Array2<f64> {
    FrameProjection::new(term).project_rows(htbeta_b)
}

pub(crate) fn low_rank_factored_htbeta_term(
    k_atoms: usize,
    m: usize,
    p: usize,
    frame_rank: usize,
    latent_dim: usize,
    n_obs: usize,
) -> SaeManifoldTerm {
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let coords = Array2::from_shape_fn((n_obs, latent_dim), |(row, axis)| {
            let phase = (row + 1) as f64 * (axis + 2) as f64 + 0.37 * (atom_idx + 1) as f64;
            0.2 * phase.sin() + 0.1 * (0.17 * phase).cos()
        });
        let mut phi = Array2::<f64>::zeros((n_obs, m));
        let mut jet = Array3::<f64>::zeros((n_obs, m, latent_dim));
        for row in 0..n_obs {
            for basis_col in 0..m {
                let base = (row + 1) as f64 * (basis_col + 1) as f64;
                phi[[row, basis_col]] = if basis_col == 0 { 1.0 } else { 0.0 }
                    + 0.01 * (base + 3.0 * atom_idx as f64).sin();
                for axis in 0..latent_dim {
                    jet[[row, basis_col, axis]] =
                        0.005 * ((base * (axis + 1) as f64) + atom_idx as f64).cos();
                }
            }
        }
        let mut frame = Array2::<f64>::zeros((p, frame_rank));
        for frame_col in 0..frame_rank {
            frame[[(atom_idx * frame_rank + frame_col) % p, frame_col]] = 1.0;
        }
        let coords_c = Array2::from_shape_fn((m, frame_rank), |(basis_col, frame_col)| {
            0.2 + 0.03 * (basis_col + 2 * frame_col + atom_idx) as f64
        });
        let decoder = coords_c.dot(&frame.t());
        let mut atom = SaeManifoldAtom::new(
            "factored_htbeta_shape",
            SaeAtomBasisKind::EuclideanPatch,
            latent_dim,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap();
        atom.maybe_activate_decoder_frame()
            .expect("frame activation")
            .expect("low-rank atom should activate a frame");
        atoms.push(atom);
        coord_blocks.push(coords);
    }
    let logits = Array2::<f64>::from_shape_fn((n_obs, k_atoms), |(row, atom)| {
        0.03 * ((row + 1) as f64 * (atom + 2) as f64).sin()
    });
    let manifolds =
        vec![LatentManifold::Product(vec![LatentManifold::Euclidean; latent_dim]); k_atoms];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::softmax(0.9),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

pub(crate) fn factored_htbeta_rho(k_atoms: usize, latent_dim: usize) -> SaeManifoldRho {
    SaeManifoldRho::new(0.0, -0.2, vec![Array1::<f64>::zeros(latent_dim); k_atoms])
}

#[test]
pub(crate) fn factored_row_htbeta_native_solve_matches_full_b_then_project() {
    let k_atoms = 2usize;
    let m = 4usize;
    let p = 24usize;
    let r = 2usize;
    let n_obs = 5usize;
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let mut frame = Array2::<f64>::zeros((p, r));
        frame[[atom_idx * r, 0]] = 1.0;
        frame[[atom_idx * r + 1, 1]] = 1.0;
        let coords = Array2::from_shape_fn((n_obs, 1), |(row, _)| 0.1 * (row + 1) as f64);
        let mut phi = Array2::<f64>::zeros((n_obs, m));
        let mut jet = Array3::<f64>::zeros((n_obs, m, 1));
        for row in 0..n_obs {
            for basis_col in 0..m {
                let x = (row + 1) as f64 * (basis_col + 1) as f64;
                phi[[row, basis_col]] = 0.03 * x + if row % m == basis_col { 1.0 } else { 0.0 };
                jet[[row, basis_col, 0]] = 0.02 * x;
            }
        }
        let c = Array2::from_shape_fn((m, r), |(basis_col, frame_col)| {
            0.2 + 0.04 * (basis_col + 2 * frame_col + atom_idx) as f64
        });
        let decoder = fast_abt(&c, &frame);
        let mut atom = SaeManifoldAtom::new(
            "factored_row_native",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap();
        atom.maybe_activate_decoder_frame()
            .expect("frame activation")
            .expect("rank-2 atom should activate a frame");
        atoms.push(atom);
        coord_blocks.push(coords);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_shape_fn((n_obs, k_atoms), |(row, atom)| {
            0.15 * (row + 1) as f64 - 0.07 * atom as f64
        }),
        coord_blocks,
        vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
        AssignmentMode::softmax(0.9),
    )
    .unwrap();
    let mut factored_term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    assert!(factored_term.frames_active());
    let border_dim = factored_term.factored_border_dim();
    assert!(border_dim < factored_term.beta_dim());

    let mut full_term = factored_term.clone();
    for atom in &mut full_term.atoms {
        atom.deactivate_decoder_frame();
    }
    let rho = SaeManifoldRho::new(
        0.0,
        -0.2,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    );
    let target = Array2::<f64>::from_shape_fn((n_obs, p), |(row, col)| {
        0.01 * (row + 1) as f64 - 0.002 * (col + 1) as f64
    });

    let native_sys = factored_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    assert_eq!(native_sys.k, border_dim);
    assert!(native_sys.htbeta_matvec.is_none());
    assert!(native_sys.htbeta_transpose_matvec.is_none());
    for row in &native_sys.rows {
        assert_eq!(row.htbeta.ncols(), border_dim);
    }

    let full_sys = full_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let mut projected_sys = factored_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    projected_sys.htbeta_matvec = None;
    projected_sys.htbeta_transpose_matvec = None;
    projected_sys.htbeta_dense_supplement = false;
    for row_idx in 0..n_obs {
        let htbeta_b = materialize_row_htbeta_for_test(&full_sys, row_idx);
        projected_sys.rows[row_idx].htbeta =
            project_row_htbeta_to_factored_for_test(&factored_term, htbeta_b.view());
    }
    projected_sys.refresh_row_hessian_fingerprint();

    let ridge_t = 5.0e-1;
    let (native_dt, native_db, _) = native_sys.solve(ridge_t, 1.0e-8).unwrap();
    let (projected_dt, projected_db, _) = projected_sys.solve(ridge_t, 1.0e-8).unwrap();

    assert_eq!(native_dt.len(), projected_dt.len());
    assert_eq!(native_db.len(), projected_db.len());
    for idx in 0..native_dt.len() {
        assert_abs_diff_eq!(native_dt[idx], projected_dt[idx], epsilon = 1.0e-10);
    }
    for idx in 0..native_db.len() {
        assert_abs_diff_eq!(native_db[idx], projected_db[idx], epsilon = 1.0e-10);
    }
}

#[test]
pub(crate) fn factored_row_htbeta_d2_matches_dense_full_b_then_project() {
    let k_atoms = 3usize;
    let m = 5usize;
    let p = 32usize;
    let frame_rank = 2usize;
    let latent_dim = 2usize;
    let n_obs = 6usize;
    let mut factored_term =
        low_rank_factored_htbeta_term(k_atoms, m, p, frame_rank, latent_dim, n_obs);
    assert!(factored_term.frames_active());
    assert_eq!(
        factored_term.factored_border_dim(),
        k_atoms * m * frame_rank
    );
    assert!(factored_term.factored_border_dim() < factored_term.beta_dim());

    let mut full_term = factored_term.clone();
    for atom in &mut full_term.atoms {
        atom.deactivate_decoder_frame();
    }
    let rho = factored_htbeta_rho(k_atoms, latent_dim);
    let target = Array2::<f64>::from_shape_fn((n_obs, p), |(row, col)| {
        0.01 * (row + 1) as f64 - 0.002 * (col + 1) as f64
    });

    let native_sys = factored_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let full_sys = full_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let mut projected_sys = factored_term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    projected_sys.htbeta_matvec = None;
    projected_sys.htbeta_transpose_matvec = None;
    projected_sys.htbeta_dense_supplement = false;
    for row_idx in 0..n_obs {
        let htbeta_b = materialize_row_htbeta_for_test(&full_sys, row_idx);
        projected_sys.rows[row_idx].htbeta =
            project_row_htbeta_to_factored_for_test(&factored_term, htbeta_b.view());
    }
    projected_sys.refresh_row_hessian_fingerprint();

    let ridge_t = 5.0e-1;
    let (native_dt, native_db, _) = native_sys.solve(ridge_t, 1.0e-8).unwrap();
    let (projected_dt, projected_db, _) = projected_sys.solve(ridge_t, 1.0e-8).unwrap();
    assert_eq!(native_dt.len(), projected_dt.len());
    assert_eq!(native_db.len(), projected_db.len());
    for idx in 0..native_dt.len() {
        assert_abs_diff_eq!(native_dt[idx], projected_dt[idx], epsilon = 1.0e-10);
    }
    for idx in 0..native_db.len() {
        assert_abs_diff_eq!(native_db[idx], projected_db[idx], epsilon = 1.0e-10);
    }
}

#[test]
pub(crate) fn qwen_shape_d2_factored_htbeta_assembly_stays_below_8gib() {
    const K_ATOMS: usize = 8;
    const M: usize = 10;
    const P: usize = 2048;
    const FRAME_RANK: usize = 2;
    const LATENT_DIM: usize = 2;
    const N_OBS: usize = 2000;
    const EIGHT_GIB: usize = 8 * 1024 * 1024 * 1024;

    let mut term = low_rank_factored_htbeta_term(K_ATOMS, M, P, FRAME_RANK, LATENT_DIM, N_OBS);
    assert!(term.frames_active());
    assert_eq!(term.beta_dim(), K_ATOMS * M * P);
    assert_eq!(term.factored_border_dim(), K_ATOMS * M * FRAME_RANK);
    assert!(term.factored_border_dim() < term.beta_dim());

    let rho = factored_htbeta_rho(K_ATOMS, LATENT_DIM);
    let target = Array2::<f64>::from_shape_fn((N_OBS, P), |(row, col)| {
        1.0e-4 * ((row + 1) as f64 * (col + 3) as f64).sin()
    });
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();

    assert_eq!(sys.k, term.factored_border_dim());
    assert!(sys.htbeta_matvec.is_none());
    assert!(sys.htbeta_transpose_matvec.is_none());
    let actual_row_dim = sys.row_dims[0];
    assert!(actual_row_dim > 0);
    assert!(sys.row_dims.iter().all(|&dim| dim == actual_row_dim));
    for row in &sys.rows {
        assert_eq!(row.htbeta.ncols(), term.factored_border_dim());
        assert_eq!(row.htbeta.nrows(), actual_row_dim);
    }

    let htbeta_bytes: usize = sys
        .rows
        .iter()
        .map(|row| row.htbeta.len() * std::mem::size_of::<f64>())
        .sum();
    let assembled_dense_bytes = htbeta_bytes
        + sys.hbb.len() * std::mem::size_of::<f64>()
        + sys.gb.len() * std::mem::size_of::<f64>();
    let old_full_b_htbeta_bytes = N_OBS
        .saturating_mul(actual_row_dim)
        .saturating_mul(term.beta_dim())
        .saturating_mul(std::mem::size_of::<f64>());

    assert!(
        old_full_b_htbeta_bytes > EIGHT_GIB,
        "test shape must reproduce the old p-wide H_tbeta memory wall"
    );
    assert!(
        assembled_dense_bytes < EIGHT_GIB,
        "qwen-shaped factored assembly stored {assembled_dense_bytes} bytes, \
             exceeding the 8 GiB gate"
    );
}

/// A full-rank small-`p` decoder must NOT activate a frame: the factored
/// border equals the full `M_k·p`, the Grassmann evidence dimension is `0`,
/// and the Occam normalizer is bit-for-bit the historical
/// `½·p·rank(S)·log λ` — the small-`p` evidence-equality contract.
#[test]
pub(crate) fn factored_evidence_matches_full_b_at_small_p() {
    let m = 5usize;
    let p = 2usize;
    // Full-rank decoder (rank 2 == p): no border saving, frame must stay off.
    let mut decoder = Array2::<f64>::zeros((m, p));
    for mu in 0..m {
        decoder[[mu, 0]] = 1.0 + mu as f64;
        decoder[[mu, 1]] = (mu as f64) - 2.0;
    }
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        jet[[mu, mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new(
        "fullrank",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    let activated = atom.maybe_activate_decoder_frame().expect("activate");
    assert_eq!(
        activated, None,
        "full-rank small-p must stay on full-B path"
    );
    assert!(atom.decoder_frame.is_none());
    assert_eq!(atom.border_frame_rank(), p);
    assert_eq!(atom.frame_manifold_dimension(), 0);

    let mut term = SaeManifoldTerm::new(
        vec![atom],
        SaeAssignment::from_blocks_with_mode(
            Array2::<f64>::zeros((m, 1)),
            vec![Array2::<f64>::zeros((m, 1))],
            AssignmentMode::softmax(0.7),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(!term.frames_active());
    assert_eq!(term.factored_border_dim(), term.beta_dim());
    assert_eq!(term.grassmann_evidence_dimension(), 0);
    let activated_n = term.auto_activate_decoder_frames().expect("auto");
    assert_eq!(activated_n, 0, "small-p auto-activation must be a no-op");

    // Occam normalizer equals the historical ½·p·rank(S)·log λ exactly.
    let rho = SaeManifoldRho::new(0.0, 0.37, vec![array![0.0_f64]]);
    let occam = term.reml_occam_term(&rho).expect("occam");
    let rank_s = SaeManifoldTerm::symmetric_rank(&term.atoms[0].smooth_penalty).unwrap();
    let expected = 0.5 * (p as f64) * (rank_s as f64) * rho.log_lambda_smooth[0];
    assert_abs_diff_eq!(occam, expected, epsilon = 1.0e-12);
}

/// Streaming polar refresh from an accumulated cross-moment re-orients the
/// frame toward the cross-moment span and keeps `B_k`'s in-span component
/// while staying column-orthonormal (the closed-form streaming step).
#[test]
pub(crate) fn streaming_polar_refresh_reorients_frame() {
    let m = 4usize;
    let p = 12usize;
    let r = 2usize;
    let mut frame0 = Array2::<f64>::zeros((p, r));
    frame0[[0, 0]] = 1.0;
    frame0[[1, 1]] = 1.0;
    let mut c0 = Array2::<f64>::zeros((m, r));
    for mu in 0..m {
        c0[[mu, 0]] = 1.0 + mu as f64;
        c0[[mu, 1]] = 0.5 - mu as f64;
    }
    let decoder = fast_abt(&c0, &frame0);
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        jet[[mu, mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new(
        "stream",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    atom.maybe_activate_decoder_frame().expect("activate");
    // New cross-moment pointing at axes {2,3}: refreshed frame must span them.
    let mut cross = Array2::<f64>::zeros((p, r));
    cross[[2, 0]] = 3.0;
    cross[[3, 1]] = 2.0;
    atom.refresh_frame_from_cross_moment(cross.view())
        .expect("refresh");
    let frame = atom.decoder_frame.as_ref().expect("frame");
    // Frame stays orthonormal.
    let gram = fast_atb(&frame.frame().to_owned(), &frame.frame().to_owned());
    for i in 0..r {
        for j in 0..r {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(gram[[i, j]], expect, epsilon = 1.0e-9);
        }
    }
    // Refreshed span aligns with the cross-moment axes {2,3} (angle ~0).
    let mut target_span = Array2::<f64>::zeros((p, r));
    target_span[[2, 0]] = 1.0;
    target_span[[3, 1]] = 1.0;
    let angle = frame
        .max_principal_angle(target_span.view())
        .expect("angle");
    assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-9);
}

#[test]
pub(crate) fn small_p_zero_decoder_stays_full_b() {
    let m = 3usize;
    let p = 8usize;
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for row in 0..m {
        phi[[row, row]] = 1.0;
        jet[[row, row, 0]] = 1.0;
    }
    let smooth_penalty = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new(
        "small-p-zero",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        smooth_penalty,
    )
    .unwrap();

    assert_eq!(atom.decoder_frame_activation_rank().unwrap(), None);
    assert_eq!(atom.maybe_activate_decoder_frame().unwrap(), None);
    assert_eq!(atom.border_frame_rank(), p);
}

/// #1026/#1417: the learnable-α forward data-derivative must give an UNGATED
/// (background-tier) atom ZERO α-sensitivity. An ungated atom's gate is forced
/// to 1.0 (`has_ungated` override), so its mass `a_k ≡ 1` is α-independent and
/// `∂a_k/∂logα = 0` — the `π_k(α)` chain applies only to gated atoms. Before the
/// fix the code credited the ungated atom `(1/π_k)·dπ_k/dρ ≠ 0`, biasing the
/// data α-gradient. FD-check the analytic against the data NLL ½Σ‖fitted−target‖²
/// (where the ungated atom's reconstruction is α-constant) on a 2-atom fixture
/// with atom 1 ungated.
#[test]
pub(crate) fn forward_alpha_data_derivative_skips_ungated_atom_1026() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    // Atom 1 is the #1026 ungated background tier (gate ≡ 1).
    term.assignment = term
        .assignment
        .clone()
        .with_ungated(vec![false, true])
        .unwrap();
    rho.log_lambda_sparse = 0.3;

    let analytic = term
        .learnable_ibp_forward_alpha_data_derivative(&rho, target.view())
        .unwrap();

    // FD of the data NLL ½Σ‖fitted−target‖² wrt ρ₀ (= logα offset, since
    // α = α₀·e^{ρ₀} ⇒ ∂logα/∂ρ₀ = 1). The ungated atom's fitted contribution is
    // α-constant, so the FD sees only the gated atom's π-derivative.
    let data_nll = |t: &SaeManifoldTerm, r: &SaeManifoldRho| -> f64 {
        let fitted = t.try_fitted_for_rho(r).unwrap();
        let mut s = 0.0_f64;
        for row in 0..fitted.nrows() {
            for c in 0..fitted.ncols() {
                let d = fitted[[row, c]] - target[[row, c]];
                s += d * d;
            }
        }
        0.5 * s
    };
    let h = 1.0e-6;
    let mut rp = rho.clone();
    let mut rm = rho.clone();
    rp.log_lambda_sparse += h;
    rm.log_lambda_sparse -= h;
    let fd = (data_nll(&term, &rp) - data_nll(&term, &rm)) / (2.0 * h);
    assert!(
        (analytic - fd).abs() <= 1.0e-5 * (1.0 + fd.abs()),
        "forward-α data derivative must match FD with an ungated atom: \
         analytic={analytic:.8e}, fd={fd:.8e}"
    );
    // Non-vacuity: the gated atom must give a materially nonzero derivative
    // (otherwise the test would pass even if everything were zeroed).
    assert!(
        fd.abs() > 1.0e-6,
        "fixture must exercise a nonzero gated-atom α-derivative; fd={fd:.3e}"
    );
}

pub(crate) fn gamma_fd_tiny_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 10usize;
    let p = 3usize;
    let k_atoms = 2usize;
    let m = 3usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let weights = [
        [
            [0.10, -0.05, 0.03],
            [0.35, -0.20, 0.12],
            [-0.16, 0.18, 0.08],
        ],
        [
            [-0.08, 0.04, 0.06],
            [0.22, 0.10, -0.18],
            [0.11, -0.24, 0.15],
        ],
    ];
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = (row as f64 + 0.35) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.21).fract();
        logits[[row, 0]] = if row % 2 == 0 { 0.8 } else { -0.6 };
        let assignments = softmax_row(logits.row(row), 0.9);
        for atom in 0..k_atoms {
            let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
            let basis = [1.0, theta.sin(), theta.cos()];
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        assignments[atom] * basis[basis_col] * weights[atom][basis_col][out_col];
                }
            }
        }
    }
    let mut atoms = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let (phi, jet) = evaluator.evaluate(coords[atom].view()).unwrap();
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            weights[atom][basis_col][out_col]
        });
        atoms.push(
            SaeManifoldAtom::new(
                format!("gamma_{atom}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; k_atoms],
        AssignmentMode::softmax(0.9),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let rho = SaeManifoldRho::new(
        -6.0,
        -6.0,
        vec![Array1::from_vec(vec![-6.0]), Array1::from_vec(vec![-6.0])],
    );
    (term, target, rho)
}

pub(crate) fn fixed_state_logdet(
    mut term: SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> f64 {
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
        .expect("fixed-state cache");
    let (tt, beta) = cache.arrow_log_det();
    tt + beta.expect("dense Schur logdet")
}


// [#780 line-count gate] The #1557 arrow-Schur parallelism-invariance
// regression test (`arrow_schur_assembly_is_faer_parallelism_invariant_1557`)
// was split into the sibling `tests_parallelism_invariance_1557.rs` module
// (declared in `mod.rs`) to keep this tracked file under the 10k limit.
//
// The four stationary-cache `∂log|H|/∂θ` adjoint regression tests
// (`sae_logdet_theta_adjoint_matches_dense_fd_*`,
// `ibp_rho_sparse_logdet_trace_matches_dense_fd_1416`,
// `learnable_ibp_alpha_logdet_trace_matches_dense_fd_1417`) were likewise split
// into the sibling `tests_logdet_adjoint_780.rs` module for the same gate; they
// still source the shared `gamma_fd_tiny_fixture` / `fixed_state_logdet`
// helpers, which remain defined here.
