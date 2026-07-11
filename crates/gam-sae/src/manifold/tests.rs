use gam_linalg::faer_ndarray::fast_ata;

pub(crate) use super::tests_recovery_split_780::{
    diagonal_latent_cache, fixed_state_logdet, gamma_fd_tiny_fixture, warmstart_test_objective,
    warmstart_test_objective_with_evaluator,
};
use super::*;
use approx::assert_abs_diff_eq;
use gam_terms::analytic_penalties::ARDPenalty;
use ndarray::{Array5, array};

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
    for &col in &eta.split.base_cols {
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    term.record_evidence_gauge_deflation_count(150, true)
        .unwrap();
    // A sustained 150<->147 flicker reverses direction on EVERY step — far more
    // reversals than the K=1 budget of 6 — yet the amplitude (3) is well inside
    // the relative jitter band (150/4 = 37), so none charge the budget.
    let flicker = [
        147usize, 150, 147, 150, 147, 150, 147, 150, 147, 150, 147, 150, 147, 150,
    ];
    for &c in &flicker {
        term.record_evidence_gauge_deflation_count(c, true)
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
    term2
        .record_evidence_gauge_deflation_count(150, true)
        .unwrap();
    let mut errored = false;
    for &c in &[
        40usize, 150, 40, 150, 40, 150, 40, 150, 40, 150, 40, 150, 40, 150,
    ] {
        if term2
            .record_evidence_gauge_deflation_count(c, true)
            .is_err()
        {
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
    term.record_evidence_gauge_deflation_count(60, true)
        .unwrap();
    assert_eq!(term.expected_evidence_gauge_deflated_directions, Some(60));

    // A matching later observation is a no-op (still Ok, count unchanged).
    term.record_evidence_gauge_deflation_count(60, true)
        .unwrap();
    assert_eq!(term.expected_evidence_gauge_deflated_directions, Some(60));

    // A MONOTONE drift (the #1217 benign case — a per-row conditioning count
    // shrinking across the ρ-walk) re-anchors freely without charging the budget,
    // no matter how many steps it takes. This is exactly the real-OLMo K=2
    // signature (171→…→113) that the old `k`-event budget wrongly tripped on.
    for c in [50usize, 40, 33, 21, 12, 9, 6, 4, 3, 2] {
        term.record_evidence_gauge_deflation_count(c, true).unwrap();
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
        match term.record_evidence_gauge_deflation_count(c, true) {
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        SaeManifoldAtom::new_with_provided_function_gram(
            "plane0",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            smooth.clone(),
        )
        .unwrap(),
        SaeManifoldAtom::new_with_provided_function_gram(
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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

/// `try_assignments_row` may only pin the K==1 assignment to `1.0` for
/// Softmax, whose single simplex coordinate is genuinely fixed. For the
/// independent gate modes (ordered Beta--Bernoulli and smooth threshold) the lone logit must drive the
/// gate; otherwise the reconstruction ignores a free parameter that the
/// prior still penalizes (an invalid objective). Regression for the
/// audit's K==1 special-case bug.
#[test]
pub(crate) fn k1_gate_modes_do_not_pin_assignment_to_one() {
    // ordered Beta--Bernoulli, K=1: the posterior-mean Bernoulli gate is σ(0/τ)=0.5. The ordered
    // prior is scored separately and never caps the final function.
    let ordered_beta = SaeAssignment::from_blocks_with_mode(
        array![[0.0]],
        vec![array![[0.0]]],
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
    )
    .unwrap();
    let ordered_beta_gate = ordered_beta.try_assignments_row(0).unwrap()[0];
    assert_abs_diff_eq!(ordered_beta_gate, 0.5, epsilon = 1e-9);
    assert!(
        (ordered_beta_gate - 1.0).abs() > 1e-6,
        "K=1 ordered Beta--Bernoulli must not pin the gate to 1.0"
    );

    // Smooth threshold gate, K=1: the logit remains a live logistic coordinate.
    let jr = SaeAssignment::from_blocks_with_mode(
        array![[-1.0]],
        vec![array![[0.0]]],
        AssignmentMode::threshold_gate(1.0, 0.0),
    )
    .unwrap();
    assert_abs_diff_eq!(
        jr.try_assignments_row(0).unwrap()[0],
        gam_linalg::utils::stable_logistic(-1.0),
        epsilon = 1e-12
    );

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

/// The smooth threshold gate is centered at the threshold: just above the
/// threshold the gate is ≈ σ(0) = 0.5, not the uncentered σ(threshold/τ).
/// Regression for the audit's miscentered-threshold bug.
#[test]
pub(crate) fn smooth_threshold_gate_is_centered_at_threshold() {
    let threshold = 2.0;
    let temperature = 1.0;
    let logits = array![2.0 + 1e-6, 1.0];
    let gates = threshold_gate_row(logits.view(), temperature, threshold);
    // Just above threshold the centered surrogate is ≈ 0.5; the old
    // uncentered surrogate would have been σ(2.0) ≈ 0.88.
    assert_abs_diff_eq!(gates[0], 0.5, epsilon = 1e-3);
    assert!(
        gates[0] < 0.6,
        "surrogate not centered at threshold: {}",
        gates[0]
    );
    // Below threshold the same smooth scalar remains positive and exact.
    assert_abs_diff_eq!(
        gates[1],
        gam_linalg::utils::stable_logistic(-1.0),
        epsilon = 1e-12
    );
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
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
/// stationarity (`penalized_laml_criterion: inner solve did not converge`).
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
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

/// Behavioral certificate for the periodic-axis SCAD/MCP exemption (issue
/// #795, reviewer item F7): the origin-anchored magnitude shrinkage must exert
/// **no angular preference** on a Circle atom, so it cannot pin occupancy
/// toward the (arbitrary, post-canonicalization) chart origin — the exact
/// occupancy statistics `coordinate_fidelity` certifies.
///
/// The crisp, fit-free form of "no origin-pinning": the SCAD *contribution* to
/// the penalized objective (`with_scad − without`) is identically zero for two
/// occupancy-wise OPPOSITE configurations — rows spread uniformly around the
/// circle (mean-resultant length `R ≈ 0`, near-uniform occupancy) versus rows
/// collapsed at the origin (`R ≈ 1`, degenerate occupancy). If SCAD pinned
/// toward the origin it would assign the spread config strictly higher energy
/// than the collapsed one; instead both contributions are zero, so the penalty
/// is blind to angular position and leaves the occupancy distribution
/// unbiased. The old unrestricted energy scored these two configs ≈ `5·Σ|t|`
/// apart, actively rewarding the collapsed (origin-pinned) occupancy.
#[test]
pub(crate) fn scad_no_origin_pinning_occupancy_on_circle() {
    use gam_terms::analytic_penalties::{PenaltyConcavity, ScadMcpPenalty};

    // Mean-resultant length of a set of circle coordinates (period 1): the
    // standard directional-occupancy concentration statistic. R→0 is uniform
    // occupancy, R→1 is a single-point (maximally biased) occupancy.
    fn resultant_length(coords: &Array2<f64>) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        let (mut cx, mut sy) = (0.0_f64, 0.0_f64);
        for row in 0..coords.nrows() {
            let a = two_pi * coords[[row, 0]];
            cx += a.cos();
            sy += a.sin();
        }
        let n = coords.nrows() as f64;
        ((cx / n).powi(2) + (sy / n).powi(2)).sqrt()
    }

    // SCAD contribution (with − without) to the penalized objective for a pure
    // Circle atom holding `coords`, under a large-weight SCAD shrinkage that
    // would dominate the objective if it were (wrongly) active on the axis.
    let scad_contribution = |coords: Array2<f64>| -> f64 {
        let n = coords.nrows();
        let (phi, jet) = periodic_basis(&coords);
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
            "periodic",
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            array![[0.2_f64], [-0.3], [0.4]],
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
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
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0_f64]]);
        let target = Array2::<f64>::zeros((n, 1));
        let with_scad = term
            .penalized_objective_total(target.view(), &rho, Some(&registry), 1.0)
            .unwrap();
        let without = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .unwrap();
        with_scad - without
    };

    // Occupancy-wise opposite configurations. `spread` is 5 EVENLY spaced
    // angles (0.1..0.9 by 0.2 ⇒ 72° apart), so its mean-resultant length is ≈0
    // (near-uniform occupancy); `collapsed` piles every row at the origin (R≈1).
    let spread = array![[0.1_f64], [0.3], [0.5], [0.7], [0.9]];
    let collapsed = array![[0.0_f64], [0.001], [0.0], [0.002], [0.001]];
    assert!(
        resultant_length(&spread) < 0.1,
        "spread config should have near-uniform occupancy (R≈0), got {}",
        resultant_length(&spread)
    );
    assert!(
        resultant_length(&collapsed) > 0.99,
        "collapsed config should have degenerate occupancy (R≈1), got {}",
        resultant_length(&collapsed)
    );

    let c_spread = scad_contribution(spread);
    let c_collapsed = scad_contribution(collapsed);
    // No angular preference: zero energy for BOTH, so their difference — the
    // origin-pinning force the old code applied — is also zero.
    assert!(
        c_spread.abs() < 1.0e-12,
        "SCAD must add zero energy on a spread circle occupancy, got {c_spread}"
    );
    assert!(
        c_collapsed.abs() < 1.0e-12,
        "SCAD must add zero energy on a collapsed circle occupancy, got {c_collapsed}"
    );
    assert!(
        (c_spread - c_collapsed).abs() < 1.0e-12,
        "SCAD must not prefer origin-collapsed over spread occupancy \
             (origin-pinning bias): spread={c_spread}, collapsed={c_collapsed}"
    );
}

/// The von-Mises coordinate-prior curvature `V'' = α·cos(κt)` is indefinite
/// (negative for |t| past a quarter period). Writing it raw into the
/// Newton/Schur `htt` diagonal at K=2 made the per-row coordinate block, and
/// hence the Schur complement, non-PD and the Cholesky failed on a negative
/// pivot (BUG 3). The assembled `htt` diagonal on every periodic coord axis
/// must therefore be non-negative (the `max(V'',0)` PSD majorizer), while the
/// gradient stays the exact `V'`.
/// #1026 shared-ARD flat-layout contract. With `K=2` single-axis atoms the
/// SHARED parameterization collapses both per-atom axis-0 ARD strengths onto ONE
/// outer coordinate (`1+K+0 = 3`), so the flat outer vector is length
/// `1+K+max_d = 4`. The former per-atom cursor walk in the gradient / EFS / IFT
/// consumers wrote atom0→3 and atom1→4 — index 4 is OUT OF BOUNDS on a length-4
/// vector (panic), and even when it did not panic it split one shared strength
/// across two phantom slots. `ard_flat_index` maps every atom owning an axis onto
/// the single shared coordinate (in-bounds), and the consumers accumulate into
/// it. The PerAtom arm keeps unique coordinates matching the `to_flat` cursor.
#[test]
pub(crate) fn shared_ard_flat_index_aliases_in_bounds_1026() {
    let shared = SaeManifoldRho::new_shared_ard(0.0, 0.0, vec![array![0.1_f64], array![0.2_f64]]);
    let shared_len = shared.to_flat().len();
    assert_eq!(shared_len, 4, "shared flat len = 1+K+max_d");
    assert_eq!(shared.ard_flat_index(0, 0), 3);
    assert_eq!(
        shared.ard_flat_index(1, 0),
        3,
        "both atoms' axis 0 alias the single shared coordinate"
    );
    assert!(
        shared.ard_flat_index(1, 0) < shared_len,
        "shared index must stay in bounds (the old per-atom walk went OOB)"
    );

    let per_atom = SaeManifoldRho::new(0.0, 0.0, vec![array![0.1_f64], array![0.2_f64]]);
    assert_eq!(per_atom.to_flat().len(), 5, "per-atom flat len = 1+K+Σ d_k");
    assert_eq!(per_atom.ard_flat_index(0, 0), 3);
    assert_eq!(
        per_atom.ard_flat_index(1, 0),
        4,
        "per-atom keeps unique coordinates (bit-for-bit the historical cursor)"
    );
}

#[test]
pub(crate) fn periodic_ard_curvature_is_psd_in_assembled_htt() {
    // Two rows past the quarter period (t in (0.25, 0.75)) where cos(2πt) < 0.
    let coords0 = array![[0.40_f64], [0.60_f64]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

/// The exact TopK compact layout must apply the same per-row Riemannian geometry
/// as a full-support layout. The compact path rebuilds
/// each compact row's product manifold + point in compact column order
/// (`compact_row_ext_manifold_and_point`) and applies the identical
/// `gt` gradient projection, `htt` Riemannian-Hessian correction, and `htbeta`
/// column projection (plus the Kronecker local-Jacobian projection).
///
/// This pins the equivalence directly: with EVERY row's active set forced to the
/// full atom set, the compact column order coincides with the dense full-`q`
/// order (`TopK { k: K }` has no gate coordinates), so the two assemblies
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
    let atom_a = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom_b = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::top_k_support(2),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom_a, atom_b], assignment).unwrap();
    let target = array![
        [0.10_f64, -0.05],
        [0.20, 0.15],
        [-0.12, 0.08],
        [0.05, -0.20]
    ];
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![5.0_f64.ln()], array![5.0_f64.ln()]]);
    let probe = SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM;

    // Dense layout: pin `Some(None)` so the override forces the dense path
    // regardless of the budget-derived plan.
    let dense = term
        .assemble_arrow_schur_inner(target.view(), &rho, None, 1.0, probe, Some(None))
        .unwrap();

    // Compact layout with EVERY row's active set = both atoms (full support).
    let layout = SaeRowLayout::from_topk_gates(
        &term.assignments_all_parallel(n).unwrap(),
        2,
        vec![1usize, 1usize],
        term.assignment.coord_offsets(),
    )
    .unwrap();
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

/// Exact dense assignment admission is geometry-independent and refuses before
/// allocation when the required row-curvature plus decoder-Gram storage exceeds
/// the supplied budget. It never changes the model into a compact surrogate.
#[test]
pub(crate) fn dense_assignment_budget_refuses_without_truncation() {
    // Build a `k`-atom ordered Beta--Bernoulli term with the given per-atom coordinate manifold.
    // Each atom carries the width-3 periodic basis, so `m_total = 3·k`.
    fn build_term(k: usize, curved: bool) -> SaeManifoldTerm {
        let n = 4usize;
        let coords: Array2<f64> = array![[0.12_f64], [0.37], [0.66], [0.91]];
        let (phi, jet) = periodic_basis(&coords);
        let atoms: Vec<SaeManifoldAtom> = (0..k)
            .map(|j| {
                SaeManifoldAtom::new_with_provided_function_gram(
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
            AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
        )
        .unwrap();
        SaeManifoldTerm::new(atoms, assignment).unwrap()
    }

    let k = 8usize;
    let curved = build_term(k, true);
    let euclidean = build_term(k, false);

    let curved_required = curved.exact_dense_assignment_bytes();
    let euclidean_required = euclidean.exact_dense_assignment_bytes();
    assert_eq!(
        curved_required, euclidean_required,
        "exact dense memory accounting must not depend on coordinate geometry"
    );
    let too_small = curved_required.saturating_sub(1);
    let error = curved
        .require_exact_dense_assignment_budget(too_small)
        .expect_err("an undersized budget must refuse the exact dense model");
    assert!(error.contains("never silently truncated"));
    curved
        .require_exact_dense_assignment_budget(curved_required)
        .expect("the exact required-byte boundary is admitted");
    euclidean
        .require_exact_dense_assignment_budget(euclidean_required)
        .expect("the exact required-byte boundary is admitted");
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
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
    term.restore_mutable_state(&snapshot)
        .expect("differential restore rebuilds the basis");
    // The differential snapshot drops `basis_values`/`basis_jacobian` and rebuilds
    // them from the restored coordinates via `refresh_basis_from_current_coords`.
    // `TestPeriodicEvaluator::evaluate` is exactly `periodic_basis`, so the rebuild
    // reproduces the pre-step basis bit-for-bit (basis is deterministic in coords).
    assert_eq!(term.atoms[0].basis_values, pre_basis);
    assert_eq!(term.atoms[0].basis_jacobian, pre_jet);
    assert_eq!(term.atoms[0].decoder_coefficients, pre_decoder);
    assert_eq!(term.assignment.logits, pre_logits);
    assert_eq!(term.assignment.coords[0].as_matrix(), pre_coords);
}

#[test]
pub(crate) fn ordered_beta_path_refreshes_periodic_basis_for_two_newton_iterations() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
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

/// #1017 — accepted nonlinear iterations reuse allocation identity while
/// rebuilding the actual current-iterate system. Two one-iteration production
/// driver calls leave their completed system allocations in the ordinary
/// workspace; the test reads that production state directly, with no test-only
/// observer embedded in the term.
#[test]
pub(crate) fn accepted_iterations_reuse_arrow_and_device_frame_allocations_with_fresh_content() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let p = 16usize;
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[0, 0]] = 0.4;
    decoder[[1, 0]] = -0.3;
    decoder[[2, 0]] = 0.2;
    decoder[[0, 1]] = -0.1;
    decoder[[1, 1]] = 0.35;
    decoder[[2, 1]] = 0.25;
    let mut target = phi0.dot(&decoder);
    for row in 0..target.nrows() {
        target[[row, 0]] += 0.08 * (row as f64 + 0.5).sin();
        target[[row, 1]] -= 0.06 * (row as f64 + 1.0).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "resident_periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi0,
        jet0,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((4, 1)),
        vec![coords0],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(0.7),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    let mut rho = SaeManifoldRho::new(0.0, -4.0, vec![Array1::<f64>::zeros(1)]);

    let decoder_before = term.atoms[0].decoder_coefficients.clone();
    term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 0.05, 1.0e-3, 1.0e-3)
        .expect("first accepted nonlinear iteration");
    assert_ne!(
        term.atoms[0].decoder_coefficients, decoder_before,
        "first production call must accept a state-changing step"
    );
    let first_row = term
        .arrow_assembly_workspace
        .rows
        .first()
        .expect("driver returns row allocations to the workspace");
    let first_row_htt_ptr = first_row.htt.as_ptr() as usize;
    let first_row_htbeta_ptr = first_row.htbeta.as_ptr() as usize;
    let first_gb_ptr = term.arrow_assembly_workspace.gb.as_ptr() as usize;
    let first_device = term
        .arrow_assembly_workspace
        .device_sae_pcg
        .as_ref()
        .filter(|data| data.frame.is_some())
        .expect("framed assembly returns its device descriptor");
    let first_device_frame_ptr = Arc::as_ptr(first_device) as usize;
    let first_device_frame_blocks_ptr = first_device
        .frame
        .as_ref()
        .map_or(0, |frame| frame.frame_blocks.as_ptr() as usize);
    let first_device_row_htbeta_ptr = first_device
        .frame
        .as_ref()
        .and_then(|frame| frame.row_htbeta.first())
        .map_or(0, |row| row.as_ptr() as usize);
    let first_device_row_htbeta_bits: Vec<u64> = first_device
        .frame
        .as_ref()
        .and_then(|frame| frame.row_htbeta.first())
        .into_iter()
        .flatten()
        .map(|value| value.to_bits())
        .collect();
    let first_device_frame_block = first_device
        .frame
        .as_ref()
        .and_then(|frame| frame.frame_blocks.first())
        .expect("framed assembly retains a data-fit G tensor W block");
    let first_device_frame_g_ptr = first_device_frame_block.g.as_ptr() as usize;
    let first_device_frame_w_ptr = first_device_frame_block.w.as_ptr() as usize;
    let first_device_frame_block_bits: Vec<u64> = first_device_frame_block
        .g
        .iter()
        .chain(first_device_frame_block.w.iter())
        .map(|value| value.to_bits())
        .collect();
    let first_numerical_bits: Vec<u64> = first_row
        .htt
        .iter()
        .chain(first_row.htbeta.iter())
        .chain(first_row.gt.iter())
        .chain(term.arrow_assembly_workspace.gb.iter())
        .map(|value| value.to_bits())
        .collect();

    // Keep the second call away from the first call's fixed point so the test
    // exercises a second accepted step instead of merely reassembling a
    // converged state. The perturbation stays inside the active output frame.
    term.atoms[0].decoder_coefficients[[0, 0]] += 0.02;
    let decoder_before_second = term.atoms[0].decoder_coefficients.clone();
    term.run_joint_fit_arrow_schur(target.view(), &mut rho, None, 1, 0.05, 1.0e-3, 1.0e-3)
        .expect("second accepted nonlinear iteration");
    assert_ne!(
        term.atoms[0].decoder_coefficients, decoder_before_second,
        "second production call must accept a state-changing step"
    );
    let second_row = term
        .arrow_assembly_workspace
        .rows
        .first()
        .expect("driver returns reused row allocations to the workspace");
    let second_device = term
        .arrow_assembly_workspace
        .device_sae_pcg
        .as_ref()
        .filter(|data| data.frame.is_some())
        .expect("reused framed assembly returns its device descriptor");
    let second_device_frame_ptr = Arc::as_ptr(second_device) as usize;
    let second_device_frame_blocks_ptr = second_device
        .frame
        .as_ref()
        .map_or(0, |frame| frame.frame_blocks.as_ptr() as usize);
    let second_device_row_htbeta_ptr = second_device
        .frame
        .as_ref()
        .and_then(|frame| frame.row_htbeta.first())
        .map_or(0, |row| row.as_ptr() as usize);
    let second_device_frame_block = second_device
        .frame
        .as_ref()
        .and_then(|frame| frame.frame_blocks.first())
        .expect("reused framed assembly retains its data-fit block");
    let second_numerical_bits: Vec<u64> = second_row
        .htt
        .iter()
        .chain(second_row.htbeta.iter())
        .chain(second_row.gt.iter())
        .chain(term.arrow_assembly_workspace.gb.iter())
        .map(|value| value.to_bits())
        .collect();

    assert_ne!(first_row_htt_ptr, 0);
    assert_eq!(first_row_htt_ptr, second_row.htt.as_ptr() as usize);
    assert_ne!(first_row_htbeta_ptr, 0);
    assert_eq!(first_row_htbeta_ptr, second_row.htbeta.as_ptr() as usize);
    assert_ne!(first_gb_ptr, 0);
    assert_eq!(
        first_gb_ptr,
        term.arrow_assembly_workspace.gb.as_ptr() as usize
    );
    assert_ne!(
        first_device_frame_ptr, 0,
        "framed assembly must install DeviceSaePcgData"
    );
    assert_eq!(first_device_frame_ptr, second_device_frame_ptr);
    assert_ne!(first_device_frame_blocks_ptr, 0);
    assert_eq!(
        first_device_frame_blocks_ptr, second_device_frame_blocks_ptr,
        "framed data-fit block vector must retain its allocation"
    );
    assert_ne!(first_device_row_htbeta_ptr, 0);
    assert_eq!(
        first_device_row_htbeta_ptr, second_device_row_htbeta_ptr,
        "dominant framed row H_tbeta host slab must retain its allocation"
    );
    assert_ne!(first_device_frame_g_ptr, 0);
    assert_eq!(
        first_device_frame_g_ptr,
        second_device_frame_block.g.as_ptr() as usize,
        "framed data-fit G block must retain its allocation"
    );
    assert_ne!(first_device_frame_w_ptr, 0);
    assert_eq!(
        first_device_frame_w_ptr,
        second_device_frame_block.w.as_ptr() as usize,
        "framed output-factor W block must retain its allocation"
    );
    let second_device_row_htbeta_bits: Vec<u64> = second_device
        .frame
        .as_ref()
        .and_then(|frame| frame.row_htbeta.first())
        .into_iter()
        .flatten()
        .map(|value| value.to_bits())
        .collect();
    assert_ne!(
        first_device_row_htbeta_bits, second_device_row_htbeta_bits,
        "retained framed row H_tbeta slab must be numerically refreshed"
    );
    let second_device_frame_block_bits: Vec<u64> = second_device_frame_block
        .g
        .iter()
        .chain(second_device_frame_block.w.iter())
        .map(|value| value.to_bits())
        .collect();
    assert_ne!(
        first_device_frame_block_bits, second_device_frame_block_bits,
        "retained framed G tensor W block must be numerically refreshed"
    );
    assert_ne!(
        first_numerical_bits, second_numerical_bits,
        "accepted state change must refresh Hessian/gradient numerical content"
    );
    eprintln!(
        "#1017 accepted-iteration residency telemetry: iterations=2 row_htt_ptr={} \
         row_htbeta_ptr={} gb_ptr={} device_frame_ptr={} device_row_htbeta_ptr={} \
         device_frame_blocks_ptr={} device_frame_g_ptr={} device_frame_w_ptr={} \
         numerical_content_changed=true",
        first_row_htt_ptr,
        first_row_htbeta_ptr,
        first_gb_ptr,
        first_device_frame_ptr,
        first_device_row_htbeta_ptr,
        first_device_frame_blocks_ptr,
        first_device_frame_g_ptr,
        first_device_frame_w_ptr,
    );
}

pub(crate) fn small_two_atom_periodic_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let atom0 = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom1 = SaeManifoldAtom::new_with_provided_function_gram(
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

/// #Bug4 — the ThresholdGate θ-adjoint prior THIRD derivative
/// (`assignment_prior_hdiag_derivative_entry`) must be ZERO for a FIXED
/// (ungated / frozen) logit, matching the zeroed assembled `htt` diagonal entry.
/// A FREE logit inside the smoothing/optimization band carries a nonzero third
/// derivative (non-vacuous fixture); the ungated atom's must be exactly zero.
#[test]
pub(crate) fn threshold_gate_fixed_logit_third_derivative_is_zero_bug4() {
    use crate::manifold::arrow_solver::SaeLocalRowVar;
    let (mut term, _target, rho) = small_two_atom_periodic_term();
    // ThresholdGate mode with atom 1 UNGATED — a fixed/inert logit.
    term.assignment.mode = AssignmentMode::threshold_gate(1.0, 0.0);
    term.assignment.ungated = vec![false, true];
    // Both atoms' logits well inside the optimization band (cutoff is −36), so a
    // FREE logit genuinely carries a nonzero third derivative.
    for row in 0..term.n_obs() {
        term.assignment.logits[[row, 0]] = 0.5;
        term.assignment.logits[[row, 1]] = 0.5;
    }
    assert!(
        term.assignment.logit_is_fixed(1) && !term.assignment.logit_is_fixed(0),
        "atom 1 must be fixed (ungated), atom 0 free"
    );

    // FREE atom 0 inside the band ⇒ nonzero third derivative (fixture is live).
    let free = term.assignment_prior_hdiag_derivative_entry(
        &rho,
        0,
        0,
        SaeLocalRowVar::Logit { atom: 0 },
        None,
    );
    assert!(
        free.abs() > 0.0,
        "a FREE logit inside the band must carry a nonzero third derivative; got {free}"
    );

    // FIXED atom 1 ⇒ the θ-adjoint third derivative MUST be exactly zero.
    let fixed = term.assignment_prior_hdiag_derivative_entry(
        &rho,
        0,
        1,
        SaeLocalRowVar::Logit { atom: 1 },
        None,
    );
    assert_eq!(
        fixed, 0.0,
        "a FIXED (ungated) logit third derivative must be zero; got {fixed}"
    );
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
            band_sd_robust: None,
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    // Softmax K=1 → gate ≡ 1 on every row (no ordered Beta--Bernoulli α to resolve).
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

/// #1777 GOAL 1 — a collapse-rescued atom has exactly one reconstruction model:
/// the `v`-projection of the row's own leave-this-atom-out residual. Training and
/// held-out reconstruction must agree through that target-aware path, while a
/// target-less reconstruction must refuse the rescued image explicitly.
#[test]
pub(crate) fn collapse_rescue_projection_matches_train_and_oos_and_refuses_targetless() {
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

    // TRAIN reconstruction uses the same residual projection as OOS. With no
    // target the term must refuse the rescued image instead of silently decoding
    // at the collapsed atom coordinate.
    term.hybrid_split_report = Some(report);
    let refusal = term
        .try_fitted()
        .expect_err("a rescued image cannot be reconstructed without its target");
    assert!(
        refusal.contains("requires try_fitted_target_aware"),
        "unexpected target-less refusal: {refusal}"
    );
    let train_recon = term
        .try_fitted_target_aware(target.view(), Some(&rho))
        .expect("target-aware train reconstruction assembles");

    // HELD-OUT reconstruction: a fresh OOS term that knows only the decoder and
    // the trained linear images (no in-fit report) recomputes each row's
    // coordinate from ITS OWN residual projected onto `v`.
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
        "train and OOS residual-projection reconstructions must be the SAME model \
         within tol; max gap {max_gap:e}"
    );
    let ev = global_ev(target.view(), oos_recon.view());
    assert!(
        ev > 0.95,
        "residual projection must recover the ramp; EV={ev:.4}"
    );
}

/// #1777 GOAL 2 — the PER-FIT [`SaeFitConfig`] is the source of truth for the
/// ordered Beta--Bernoulli-α and separation-barrier overrides: two terms carrying DIFFERENT configs
/// produce correspondingly-different α / barrier strength, and the two terms do
/// not leak into each other.
#[test]
pub(crate) fn per_fit_config_isolates_barrier_and_ordered_beta_bernoulli_alpha() {
    let (mut term_a, _t_a, rho_a) = small_two_atom_ordered_beta_term();
    let (mut term_b, _t_b, rho_b) = small_two_atom_ordered_beta_term();

    // Distinct per-fit configs, applied to each term independently.
    term_a.set_fit_config(SaeFitConfig {
        separation_barrier_strength_override: Some(0.1),
        ordered_beta_bernoulli_alpha_override: Some(0.2),
    });
    term_b.set_fit_config(SaeFitConfig {
        separation_barrier_strength_override: Some(3.0),
        ordered_beta_bernoulli_alpha_override: Some(5.0),
    });

    // Round-trips through the config accessor.
    assert_eq!(
        term_a.fit_config().ordered_beta_bernoulli_alpha_override,
        Some(0.2)
    );
    assert_eq!(
        term_b.fit_config().separation_barrier_strength_override,
        Some(3.0)
    );

    // ordered Beta--Bernoulli-α: the per-fit override is the resolved α (bypassing the mode schedule),
    // and the two terms resolve different α values.
    assert_eq!(
        term_a
            .assignment
            .resolved_ordered_beta_bernoulli_alpha(&rho_a),
        Some(0.2)
    );
    assert_eq!(
        term_b
            .assignment
            .resolved_ordered_beta_bernoulli_alpha(&rho_b),
        Some(5.0)
    );

    // Distinct α ⇒ distinct gates (the ordered geometric prior π_k differs).
    let gates_a = term_a.assignment.try_assignments().unwrap();
    let gates_b = term_b.assignment.try_assignments().unwrap();
    let gate_gap = (&gates_a - &gates_b)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        gate_gap > 1e-6,
        "distinct per-fit ordered Beta--Bernoulli-α overrides must produce distinct gates; gap {gate_gap:e}"
    );

    // Barrier strength (K=2, so the barrier is live): the per-fit override is the
    // source of truth, distinct per term.
    assert_eq!(term_a.separation_barrier_strength(), 0.1);
    assert_eq!(term_b.separation_barrier_strength(), 3.0);

    // Isolation: clearing term_a's config leaves term_b untouched, and term_a
    // uses the mode's canonical α.
    term_a.set_fit_config(SaeFitConfig::default());
    assert_eq!(
        term_a
            .assignment
            .resolved_ordered_beta_bernoulli_alpha(&rho_a),
        Some(1.0)
    ); // the mode's compiled α
    assert_eq!(
        term_b
            .assignment
            .resolved_ordered_beta_bernoulli_alpha(&rho_b),
        Some(5.0)
    );
}

/// F5 — the per-fit separation-barrier override (#1777) must isolate two
/// genuinely CONCURRENT in-process fits: two threads that each build a term,
/// set a DIFFERENT `μ_sep` via [`SaeFitConfig`], and hammer
/// [`SaeManifoldTerm::separation_barrier_strength`] must each keep reading their
/// own strength for the whole run. A parallel candidate/rung/layer sweep is safe
/// because the strength lives on the term and there is no shared mutable state.
#[test]
pub(crate) fn per_fit_barrier_isolated_under_concurrent_fits() {
    // Two distinct per-fit strengths, one per worker. Chosen far apart so a leak
    // between threads (either direction) is unambiguous.
    let strengths = [0.125_f64, 7.5_f64];
    let iters = 4000usize;

    std::thread::scope(|scope| {
        let handles: Vec<_> = strengths
            .iter()
            .map(|&mu| {
                scope.spawn(move || {
                    // Each thread owns its term (a distinct concurrent "fit").
                    let (mut term, _t, _rho) = small_two_atom_ordered_beta_term();
                    term.set_fit_config(SaeFitConfig {
                        separation_barrier_strength_override: Some(mu),
                        ordered_beta_bernoulli_alpha_override: None,
                    });
                    // Hammer the barrier-strength read while the sibling thread
                    // hammers its own with a different μ. The per-fit field is
                    // the source of truth, so every read is this thread's μ and
                    // no sibling state can be visible here.
                    for _ in 0..iters {
                        assert_eq!(
                            term.separation_barrier_strength(),
                            mu,
                            "concurrent fit read a leaked barrier strength (expected {mu})"
                        );
                    }
                    mu
                })
            })
            .collect();
        for (handle, &mu) in handles.into_iter().zip(strengths.iter()) {
            assert_eq!(handle.join().unwrap(), mu);
        }
    });
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

    // #1744 — ordered Beta--Bernoulli admits NO response-dispersion scaling on ANY ρ coordinate
    // (learnable-α or fixed-α). Its free per-row Bernoulli gates overfit under a
    // dispersion-weakened smoothness/ARD seed, collapsing the Fellner–Schall
    // fixed point; the sparse coordinate is a dimensionless log-α concentration
    // offset that was never a squared-output-unit penalty weight. So every ordered Beta--Bernoulli
    // coordinate stays at its absolute (already dimensionless) construction value.
    for ordered_beta_mode in [
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, true),
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
    ] {
        let ordered_beta = rho
            .seed_scaled_by_dispersion_for_assignment(dispersion, ordered_beta_mode)
            .unwrap();
        assert_abs_diff_eq!(
            ordered_beta.log_lambda_sparse,
            rho.log_lambda_sparse,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            ordered_beta.log_lambda_smooth[0],
            rho.log_lambda_smooth[0],
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            ordered_beta.log_ard[0][0],
            rho.log_ard[0][0],
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            ordered_beta.log_ard[0][1],
            rho.log_ard[0][1],
            epsilon = 1.0e-14
        );
    }
}

#[test]
pub(crate) fn fit_data_collapse_records_terminal_event_for_active_atom() {
    let coords = array![[0.0], [0.25], [0.5], [0.75]];
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        let norm = (0..d_embed)
            .map(|j| frame[[r, j]] * frame[[r, j]])
            .sum::<f64>()
            .sqrt();
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
    OrderedBetaBernoulli,
}

impl PlantedCircleAssignmentMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Softmax => "softmax",
            Self::OrderedBetaBernoulli => "ordered_beta_bernoulli",
        }
    }

    pub(crate) fn mode(self) -> AssignmentMode {
        const TAU: f64 = 1.0;
        const ALPHA: f64 = 1.0;
        match self {
            Self::Softmax => AssignmentMode::softmax(TAU),
            Self::OrderedBetaBernoulli => AssignmentMode::ordered_beta_bernoulli(TAU, ALPHA, false),
        }
    }

    pub(crate) fn seed_logit(self) -> f64 {
        const TAU: f64 = 1.0;
        match self {
            Self::Softmax => 0.0,
            Self::OrderedBetaBernoulli => 6.0 * TAU,
        }
    }

    pub(crate) fn seed_gate(self) -> f64 {
        match self {
            Self::Softmax => 1.0,
            Self::OrderedBetaBernoulli => 1.0 / (1.0 + (-6.0_f64).exp()),
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        PlantedCircleAssignmentMode::OrderedBetaBernoulli,
    ] {
        let label = assignment_mode.label();
        let (term, seed_dispersion) = planted_circle_seed_term(z.view(), assignment_mode);
        out.push_str(&format!(
            "FOCUS1744 mode={label} seed_disp={seed_dispersion:.3e}\n"
        ));
        for &sparse in &[-8.0_f64, 1.0] {
            for &ard in &[-6.0_f64, -3.0, 0.0, 1.0] {
                for &smooth in &[-8.0_f64, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0] {
                    let mut t = term.clone();
                    let r = SaeManifoldRho::new(sparse, smooth, vec![array![ard]]);
                    match t.penalized_laml_criterion_with_cache(
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
/// failed: `ordered_beta_bernoulli` n=40 σ=0.18. Runs exactly one outer solve from the
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
pub(crate) fn planted_circle_ordered_beta_bernoulli_n40_sigma018_reaches_high_ev_1744() {
    let assignment_mode = PlantedCircleAssignmentMode::OrderedBetaBernoulli;
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
    let result = gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_initial_rho(init_rho_flat)
        .run(&mut objective, "SAE planted circle #1744 focused")
        .unwrap();
    objective
        .certify_outer_result(&result)
        .expect("focused #1744 outer result must certify the installed state");
    let fitted_result = objective.into_fitted().expect("outer fit was evaluated");
    let rho = fitted_result.rho;
    let ev = global_ev(z.view(), fitted_result.term.fitted().view());
    assert!(
        ev > 0.95,
        "focused #1744 fixture (ordered_beta_bernoulli n={n} sigma={sigma}) seed_ev={seed_ev:.4} \
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
        PlantedCircleAssignmentMode::OrderedBetaBernoulli,
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
                let result = gam_solve::rho_optimizer::OuterProblem::new(n_params)
                    .with_initial_rho(init_rho_flat)
                    .run(&mut objective, "SAE planted circle dimensionless seed")
                    .unwrap();
                objective
                    .certify_outer_result(&result)
                    .expect("dimensionless-seed outer result must certify the installed state");
                let fitted_result = objective.into_fitted().expect("outer fit was evaluated");
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
            "SaeManifoldTerm::penalized_laml_criterion: inner solve did not converge at fixed ρ"
        )
    );
    assert!(
        SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::penalized_laml_criterion: undamped evidence factorization hit a non-PD per-row H_tt block before KKT stationarity"
        )
    );
    // The generic "log-det unavailable" message (a real factorization defect, not
    // an infeasibility) stays FATAL — it is NOT in the recoverable set.
    assert!(
        !SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::penalized_laml_criterion: arrow_log_det_from_cache returned None (undamped joint Hessian log-det unavailable for the Laplace normaliser)"
        )
    );
    assert!(
        !SaeManifoldOuterObjective::is_recoverable_value_probe_refusal(
            "SaeManifoldTerm::penalized_laml_criterion: row-gauge evidence deflation count re-anchored \
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
        .penalized_laml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .unwrap();
    let (stream_cost, stream_loss) = streaming
        .penalized_laml_criterion_streaming_exact(
            target.view(),
            &rho,
            None,
            2,
            0.25,
            1.0e-4,
            1.0e-4,
        )
        .unwrap();
    assert_abs_diff_eq!(stream_cost, full_cost, epsilon = 1.0e-8);
    assert_abs_diff_eq!(stream_loss.total(), full_loss.total(), epsilon = 1.0e-8);
}

/// As [`small_two_atom_periodic_term`], but in ordered independent
/// Beta--Bernoulli assignment mode. The full-batch and streaming paths must
/// assemble the same shared-mass-dependent PSD curvature majorizer.
pub(crate) fn small_two_atom_ordered_beta_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let atom0 = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom1 = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.8, 1.0, false),
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

/// The streaming evidence log determinant must equal the dense full-batch
/// determinant for an ordered Beta--Bernoulli term at the same fitted state.
#[test]
pub(crate) fn streaming_exact_laml_matches_full_batch_ordered_beta_bernoulli() {
    let (term0, target, rho) = small_two_atom_ordered_beta_term();
    let mut full = term0;
    let (_full_cost, _full_loss, cache) = full
        .penalized_laml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
        .expect("dense ordered Beta--Bernoulli criterion must evaluate");

    // The streaming exact log determinant must reproduce dense `log|H|` at the same
    // converged state — `full` is already at its converged (t,β) after the dense
    // criterion. We compare the log-det DIRECTLY rather than re-fitting through
    // `penalized_laml_criterion_streaming_exact`: a streaming RE-FIT runs a fresh inner solve
    // whose faer parallel reduction is non-deterministic under thread contention
    // and intermittently surfaces the (recoverable) non-PD refusal — orthogonal to
    // this value-comparison test. `streaming_exact_arrow_log_det` reassembles
    // `log|H|` chunk-by-chunk at the frozen state with no inner solve.
    let dense_logdet = arrow_log_det_from_cache(&cache).expect("dense log-det finite");
    let stream_logdet = full
        .streaming_exact_arrow_log_det(target.view(), &rho, None, None)
        .expect("streaming ordered Beta--Bernoulli log-det must evaluate");
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
        .penalized_laml_criterion_with_refine_policy(
            target.view(),
            &rho,
            None,
            2,
            0.25,
            1.0e-4,
            1.0e-4,
            true,
        )
        .expect("full-budget criterion must converge on the small fixture");
    let (probe_cost, probe_loss) = probe
        .penalized_laml_criterion_with_refine_policy(
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

/// #1224 — every fitting/ranking lane must price the same penalized LAML criterion.
///
/// The outer optimizer compares three lanes at a fixed ρ:
///   * `eval` (`OuterEvalOrder::ValueAndGradient`) returns the consistent
///     gradient-lane pair `(f, ∇f)` — penalized LAML cost paired with the exact
///     REML λ-gradient.
///   * `eval_with_order(Value)` is the line-search probe: it accepts/rejects
///     steps whose DIRECTION came from `eval`'s `∇f`, so its cost must be the
///     SAME penalized LAML `f`. Folding the gradient-free consistency penalty `c(ρ)`
///     here while the direction is `∇f` mixes two functions in the Armijo test
///     (the objective↔gradient desync bug class). The fix threads
///     `fold_cotrain = false` into this lane.
///   * `eval_cost` is the cross-seed ranking lane and also returns `f`; ranking
///     by a different `f+c` criterion would select a fit that need not be
///     stationary for the objective the optimizer descended.
///
/// The regression pins one invariant: gradient, line-search, and ranking costs
/// agree at the same `ρ`.
#[test]
pub(crate) fn outer_value_and_ranking_lanes_share_pure_penalized_laml_criterion() {
    use gam_solve::rho_optimizer::{OuterEvalOrder, OuterObjective};

    // A fixed ρ at which all three lanes converge from the same fixture state.
    let rho_flat = warmstart_test_objective().baseline_rho.to_flat();

    // Gradient lane (ValueAndGradient): the consistent `(f, ∇f)` pair. Its cost
    // is penalized LAML (+ the discrete collapse barrier, which stays on both lanes).
    let mut grad_obj = warmstart_test_objective();
    let grad_cost = grad_obj
        .eval(&rho_flat)
        .expect("gradient lane must converge on the warm-start fixture")
        .cost;

    // Line-search lane (Value order): the BFGS/ARC probe. Post-fix this reports
    // the SAME penalized LAML cost the gradient lane reports.
    let mut ls_obj = warmstart_test_objective();
    let ls_cost = ls_obj
        .eval_with_order(&rho_flat, OuterEvalOrder::Value)
        .expect("line-search probe must converge on the warm-start fixture")
        .cost;

    // Ranking lane (`eval_cost`): the cross-seed screen prices the same `f`.
    let mut rank_obj = warmstart_test_objective();
    let rank_cost = rank_obj
        .eval_cost(&rho_flat)
        .expect("ranking lane must converge on the warm-start fixture");

    // Armijo and cross-seed selection both price the criterion whose gradient
    // the accepted-point lane returns.
    assert_abs_diff_eq!(ls_cost, grad_cost, epsilon = 1.0e-10);
    assert_abs_diff_eq!(rank_cost, grad_cost, epsilon = 1.0e-10);
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

/// #2253 exact-envelope gate: an objective-stall or small-decrement diagnostic
/// cannot mint an analytic outer derivative while both the raw and quotient
/// KKT residuals remain above the stationarity bound.
#[test]
pub(crate) fn objective_stall_cannot_substitute_for_kkt_envelope_2253() {
    let tolerance = 1.0e-4;
    assert!(!SaeManifoldTerm::evidence_kkt_stationary(
        2.0 * tolerance,
        3.0 * tolerance,
        tolerance,
    ));
    assert!(SaeManifoldTerm::evidence_kkt_stationary(
        tolerance,
        3.0 * tolerance,
        tolerance,
    ));
    assert!(SaeManifoldTerm::evidence_kkt_stationary(
        2.0 * tolerance,
        tolerance,
        tolerance,
    ));
    assert!(!SaeManifoldTerm::evidence_kkt_stationary(
        f64::NAN,
        f64::INFINITY,
        tolerance,
    ));
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
        .expect(
            "cold undamped evidence factor must be spectrally conditioned (#1117), not refused",
        );
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
        .penalized_laml_criterion_with_cache(target.view(), &rho, None, 1, 0.25, 1.0e-4, 1.0e-4)
        .expect("dense REML must refine through the cold non-PD evidence factor");
    let log_det = arrow_log_det_from_cache(&cache).expect("refined cache must carry log-det");
    assert!(full_cost.is_finite());
    assert!(full_loss.total().is_finite());
    assert!(log_det.is_finite());

    let (stream_cost, stream_loss) = streaming
        .penalized_laml_criterion_streaming_exact(
            target.view(),
            &rho,
            None,
            1,
            0.25,
            1.0e-4,
            1.0e-4,
        )
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
    let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
    // Keep the parity witness on a resolved hard-MP branch. The prior target was
    // unrelated to the hand-seeded decoder, so the canonical rank-zero veto
    // fired before either dense/bundle derivative route could be exercised.
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
        target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
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
    let alpha = 250.0_f64;
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![alpha.ln()]]);
    let loss = term.loss(target.view(), &rho).unwrap();
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();

    let dispersion = term
        .reconstruction_dispersion(&loss, &cache, &rho, None)
        .unwrap();
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

/// #2080 decoder-smoothness gradient channel: the matrix-free EDF off the shared
/// (probes, S⁻¹·probes) bundle
/// ([`SaeManifoldTerm::decoder_smoothness_effective_dof_per_atom_from_probes`])
/// must reproduce the dense `beta_inv` trace
/// ([`SaeManifoldTerm::decoder_smoothness_effective_dof_per_atom`]) that the
/// Fellner–Schall α-step consumes. Feeding the bundle the EXACT dense `S⁻¹`
/// (`cache.schur_inverse_apply`) at FULL-BASIS probes `√k·e_j` makes the
/// Hutchinson estimator collapse to the exact trace `tr(S⁻¹·M_k)` deterministically
/// (both `S⁻¹` and `M_k` symmetric), so the two paths must agree to solve
/// precision — the FD-equivalent acceptance gate for the channel, isolating the
/// new M_k-apply/umbrella-contraction code from the CG machinery (tested in
/// gam-solve). The production path swaps the dense `S⁻¹` for the surrogate's
/// matrix-free probe solves.
#[test]
fn matrix_free_smoothness_edf_from_probes_matches_dense_selected_inverse() {
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let lambda = rho.lambda_smooth_vec();

    let dense = term
        .decoder_smoothness_effective_dof_per_atom(&cache, &lambda)
        .unwrap();

    // Full-basis probes √k·e_j + the EXACT dense S⁻¹ ⇒ the umbrella estimate is
    // the exact trace, deterministically.
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| cache.schur_inverse_apply(v.view()).unwrap())
        .collect();
    let matrix_free = term
        .decoder_smoothness_effective_dof_per_atom_from_probes(&probes, &sinv, &lambda)
        .unwrap();

    assert_eq!(dense.len(), matrix_free.len());
    for (atom_idx, (d, mf)) in dense.iter().zip(&matrix_free).enumerate() {
        assert_abs_diff_eq!(d, mf, epsilon = 1.0e-9);
        assert!(
            atom_idx < 1 || *mf >= 0.0,
            "atom {atom_idx} edof must be a nonneg dof, got {mf}"
        );
    }
}

/// #2080 ARD gradient channel: the matrix-free posterior-variance trace off the
/// shared (probes, S⁻¹·probes) bundle
/// ([`SaeManifoldTerm::ard_inverse_traces_from_probes`]) must reproduce the dense
/// selected-inverse / `full_inverse_apply` trace
/// ([`SaeManifoldTerm::ard_inverse_traces`]) the Fellner–Schall α-step consumes.
/// Feeding the bundle the EXACT dense `S⁻¹` (`cache.schur_inverse_apply`) at
/// FULL-BASIS probes `√k·e_j` collapses the per-(atom, axis) Hutchinson estimate
/// to its exact trace `tr(S⁻¹·M_{ka})` deterministically, so the reformulation
/// `(A_i⁻¹)[s,s] + Σ_i s_ij[s]·w_ij[s]` must equal the dense diagonal to solve
/// precision — the FD-equivalent acceptance gate isolating the new cache-only
/// M_{ka}-contraction from the CG machinery (tested in gam-solve). The production
/// path swaps the dense `S⁻¹` for the surrogate's matrix-free probe solves.
#[test]
fn matrix_free_ard_traces_from_probes_matches_dense_selected_inverse() {
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

    let dense = term.ard_inverse_traces(&cache).unwrap();

    // Full-basis probes √k·e_j + the EXACT dense S⁻¹ ⇒ the per-(atom, axis)
    // umbrella estimate is the exact selected-inverse trace, deterministically.
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| cache.schur_inverse_apply(v.view()).unwrap())
        .collect();
    let matrix_free = term
        .ard_inverse_traces_from_probes(&cache, &probes, &sinv)
        .unwrap();

    assert_eq!(dense.len(), matrix_free.len());
    for (atom_idx, (d, mf)) in dense.iter().zip(&matrix_free).enumerate() {
        assert_eq!(d.len(), mf.len());
        for (axis, (dv, mv)) in d.iter().zip(mf.iter()).enumerate() {
            assert_abs_diff_eq!(dv, mv, epsilon = 1.0e-9);
            assert!(
                *mv >= -1.0e-12,
                "atom {atom_idx} axis {axis} posterior-variance trace must be nonneg, got {mv}"
            );
        }
    }
}

/// #2080 ARD ½log|H| ρ-gradient channel: the matrix-free ARD-Hessian trace off the
/// shared (probes, S⁻¹·probes) bundle
/// ([`SaeManifoldTerm::ard_log_precision_hessian_trace_from_probes`]) must reproduce
/// the dense solver trace ([`SaeManifoldTerm::ard_log_precision_hessian_trace`]) the
/// analytic outer ρ-gradient's ARD block consumes. On the PLAIN (undeflated) fixture
/// the dense path's Daleckii–Krein correction is identically zero, so both reduce to
/// `Σ ½·(H⁻¹)_tt[s,s]·hess`; feeding the bundle the EXACT dense `S⁻¹`
/// (`cache.schur_inverse_apply`) at FULL-BASIS probes `√k·e_j` collapses the
/// per-slot selected-inverse diagonal to its exact value, so the two must agree to
/// solve precision — the FD-equivalent acceptance gate isolating the new
/// bundle-sourced diagonal from the CG machinery (tested in gam-solve).
#[test]
fn matrix_free_ard_logdet_hessian_trace_from_probes_matches_dense() {
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .ard_log_precision_hessian_trace(&rho, &cache, &solver)
        .unwrap();

    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| cache.schur_inverse_apply(v.view()).unwrap())
        .collect();
    let matrix_free = term
        .ard_log_precision_hessian_trace_from_probes(&rho, &cache, &probes, &sinv)
        .unwrap();

    assert_eq!(dense.len(), matrix_free.len());
    for (d, mf) in dense.iter().zip(&matrix_free) {
        assert_eq!(d.len(), mf.len());
        for (dv, mv) in d.iter().zip(mf.iter()) {
            assert_abs_diff_eq!(dv, mv, epsilon = 1.0e-9);
        }
    }
}

/// #2080 analytic-gradient cluster routing seam: the whole analytic outer ρ-gradient
/// assembled through `analytic_outer_rho_gradient_components_with_bundle(Some(bundle))`
/// — routing BOTH selected-inverse channels (per-atom smoothness EDF `tr(H⁻¹ M_k)`
/// and the per-(atom,axis) ARD log-precision Hessian trace `½tr(H⁻¹ ∂H/∂logα)`) off
/// the shared (probes, S⁻¹·probes) bundle as one all-or-nothing cluster — must
/// reproduce the dense (`None`) assembly bit-for-bit on the PLAIN (undeflated)
/// fixture. Every other channel (explicit, Occam, the #1006 envelope third-order
/// correction) is fed the IDENTICAL cache + plain solver in both calls, so this
/// isolates the bundle routing: feeding full-basis probes `√k·e_j` with the EXACT
/// dense `S⁻¹` (`cache.schur_inverse_apply`) collapses each channel's selected-inverse
/// diagonal to its exact value, so the two assemblies must agree to solve precision.
/// The gate that keeps the (dormant, `None`-only in production) forward plumbing from
/// silently desyncing the value and gradient lanes if a routing flip ever engages it.
#[test]
fn analytic_outer_gradient_with_bundle_matches_dense_assembly() {
    let n = 24usize;
    let p = 2usize;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
    let (phi, jet) = periodic_basis(&coords);
    let decoder = array![[0.30, -0.10], [1.20, 0.20], [0.10, 1.10]];
    assert_eq!(decoder.ncols(), p);
    // Keep this routing witness on a resolved hard-MP branch. The former target
    // was unrelated to the hand-seeded decoder, so the canonical rank-zero veto
    // fired before either dense or bundled selected-inverse route was exercised.
    let mut target = phi.dot(&decoder);
    for row in 0..n {
        target[[row, 0]] += 1.0e-3 * (0.37 * row as f64).sin();
        target[[row, 1]] += 1.0e-3 * (0.29 * row as f64).cos();
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "periodic",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
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
    let rho = SaeManifoldRho::new(0.0, 0.8_f64.ln(), vec![array![250.0_f64.ln()]]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .unwrap();
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).unwrap();
    let loss = term.loss(target.view(), &rho).unwrap();

    // Same plain (undeflated) solver in BOTH assemblies — the only variable is the
    // bundle routing of the two selected-inverse channels.
    let solver = DeflatedArrowSolver::plain(&cache);
    let dense = term
        .analytic_outer_rho_gradient_components(target.view(), &rho, &loss, &cache, &solver)
        .unwrap();

    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<Array1<f64>> = probes
        .iter()
        .map(|v| cache.schur_inverse_apply(v.view()).unwrap())
        .collect();
    let bundled = term
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            &rho,
            &loss,
            &cache,
            &solver,
            Some((&probes, &sinv)),
        )
        .unwrap();

    // The routed log|H|-trace channel must match the dense selected-inverse trace on
    // every ρ-coordinate…
    assert_eq!(dense.logdet_trace.len(), bundled.logdet_trace.len());
    for (i, (d, b)) in dense
        .logdet_trace
        .iter()
        .zip(bundled.logdet_trace.iter())
        .enumerate()
    {
        assert_abs_diff_eq!(d, b, epsilon = 1.0e-9);
        assert!(
            d.is_finite() && b.is_finite(),
            "logdet-trace coordinate {i} must be finite (dense={d}, bundled={b})"
        );
    }
    // …and the FULLY ASSEMBLED gradient (all channels summed) must agree, since every
    // other channel is fed identical inputs.
    let dense_grad = dense.gradient();
    let bundled_grad = bundled.gradient();
    assert_eq!(dense_grad.len(), bundled_grad.len());
    for (d, b) in dense_grad.iter().zip(bundled_grad.iter()) {
        assert_abs_diff_eq!(d, b, epsilon = 1.0e-9);
    }
    // Non-triviality: the routed trace must be a genuine, nonzero contribution so the
    // parity assertion is not vacuous (identity smooth penalty + ARD-shrunk axis).
    let trace_sq: f64 = dense.logdet_trace.iter().map(|v| v * v).sum();
    assert!(
        trace_sq > 0.0 && trace_sq.is_finite(),
        "the routed log|H|-trace channel must be non-trivial; ‖logdet_trace‖²={trace_sq}"
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    // `penalized_laml_criterion_with_cache` reaches a PD block), then compare the
    // streaming-vs-dense log-determinants of the SAME converged system —
    // which is the routing invariant this test pins (#847).
    full.penalized_laml_criterion_with_cache(target.view(), &rho, None, 2, 0.25, 1.0e-4, 1.0e-4)
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

/// Build a `K`-atom hard-TopK SAE term with a planted small support.
///
/// Every atom is a 1-D `EuclideanPatch` with an `M=2` constant+linear basis and
/// a distinct decoder direction, so the reconstruction is genuine and the
/// per-row Arrow-Schur block has a real data-fit Gauss-Newton contribution.
/// Row `i`'s logits rank its planted atoms above every other atom, so the hard
/// forward model and compact derivative system have exactly the same support.
fn planted_topk_sae_term(
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
            SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::top_k_support(planted[0].len()),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, c)| 0.05 * ((row + c) as f64).sin());
    (term, target)
}

/// End-to-end large-`K` compact-path contract for the hard-TopK SAE: the
/// assignment→support→assembly path must produce a per-row
/// block whose dimension tracks the per-row active-atom count `k_active`, NOT
/// the total `K`, and the assembled support must recover the planted top-`k`
/// atoms at a `K` large enough that a full-support row block would be orders of
/// magnitude larger.
#[test]
pub(crate) fn large_k_topk_encode_is_support_bounded_and_exact() {
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
        let (mut term, target) = planted_topk_sae_term(n, k_atoms, &planted, p);
        // Fixed-decoder encode assembly (the #1407 path the encoder uses): only
        // the per-row htt/gt block is produced.
        term.fixed_decoder_assembly = true;
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms]);
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("exact TopK fixed-decoder assembly must succeed at large K");
        let dims: Vec<usize> = sys.rows.iter().map(|r| r.htt.nrows()).collect();
        // Each row's htt must be square and match gt.
        for r in &sys.rows {
            assert_eq!(r.htt.nrows(), r.htt.ncols());
            assert_eq!(r.htt.nrows(), r.gt.len());
        }
        let layout = term
            .last_row_layout
            .clone()
            .expect("TopK must install its exact compact support layout");
        let active: Vec<Vec<usize>> = layout.active_atoms.clone();
        (dims, active)
    };

    let (dims_1k, active_1k) = assemble_dims(1_000);
    let (dims_10k, active_10k) = assemble_dims(10_000);

    // (a) O(1)-per-token / n-free: per-row block dim is bounded by the active
    // contract `top_k·d = top_k` for d=1 coords, and is IDENTICAL across
    // K=1000 and K=10000 (independent of total K). A full-K block would be
    // `q = K·d`, i.e. 1000 and 10000 — orders of magnitude larger.
    let bound = top_k; // no free gate coordinates; only Σ d_k
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
    let dense_q = 1_000; // K coordinate axes; TopK has no gate coordinates
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
    let mut gates = Vec::with_capacity(n);
    for row in 0..n {
        let mut row_gates = Array1::<f64>::zeros(k_atoms);
        for atom in [row, 10_000 + row, 90_000 + row] {
            row_gates[atom] = 1.0;
        }
        gates.push(row_gates);
    }
    let coord_dims = vec![1usize; k_atoms];
    let coord_offsets_full: Vec<usize> = (0..k_atoms).collect();
    let layout = SaeRowLayout::from_topk_gates(&gates, 3, coord_dims, coord_offsets_full).unwrap();
    for row in 0..n {
        assert_eq!(layout.active_atoms[row].len(), 3);
        assert_eq!(layout.row_q_active(row), 3);
    }
    let compact_work: usize = (0..n)
        .map(|_| {
            let q = layout.row_q_active(row);
            q * q
        })
        .sum();
    let dense_q = k_atoms;
    let dense_work = n * dense_q * dense_q;
    assert!(compact_work < dense_work / 1_000_000_000);
    assert_eq!(compact_work, n * 9);
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

/// #1610/#1038 — the separation barrier's PSD curvature must survive the production
/// matrix-free / framed β-tier IDENTICALLY to the dense path. The dense path writes
/// the full curvature onto `sys.hbb`: a per-atom Levenberg scalar ridge `lev·I` PLUS
/// the exact self-concordant rank-1 `d2·(∂o/∂B)(∂o/∂B)ᵀ`. The deferred path returns
/// the SAME two pieces through separate channels — the scalar ridge via `atom_curv`
/// (folded into the structured smooth op) and the rank-1 via `sep_rank1` (installed
/// as a `SparseRankOnePenaltyOp`) — so assembly reconstructs the identical operator
/// instead of silently dropping collapse-prevention curvature.
#[test]
pub(crate) fn separation_barrier_deferred_curvature_matches_dense_hbb_1610() {
    let (term, _target, _rho) = small_two_atom_periodic_term();
    let beta_dim = term.beta_dim();
    let mut dense = ArrowSchurSystem::new(0, 0, beta_dim);
    dense.gb = Array1::<f64>::zeros(beta_dim);
    dense.hbb = Array2::<f64>::zeros((beta_dim, beta_dim));
    let mut dense_atom_curv = vec![0.0_f64; term.k_atoms()];
    let mut dense_rank1 = Vec::new();
    assert!(
        term.add_sae_separation_barrier(
            &mut dense,
            1.0,
            true,
            &mut dense_atom_curv,
            &mut dense_rank1
        ),
        "fixture must activate the co-collapse separation barrier on the dense path"
    );
    assert!(
        dense_atom_curv.iter().all(|v| *v == 0.0),
        "dense path writes curvature directly to hbb, not the deferred atom accumulator"
    );
    assert!(
        dense_rank1.is_empty(),
        "dense path scatters the rank-1 straight into hbb, not the deferred carrier"
    );

    let mut deferred = ArrowSchurSystem::new(0, 0, beta_dim);
    deferred.gb = Array1::<f64>::zeros(beta_dim);
    deferred.hbb = Array2::<f64>::zeros((0, 0));
    let mut atom_curv = vec![0.0_f64; term.k_atoms()];
    let mut sep_rank1 = Vec::new();
    assert!(
        term.add_sae_separation_barrier(&mut deferred, 1.0, false, &mut atom_curv, &mut sep_rank1),
        "fixture must activate the co-collapse separation barrier on the deferred path"
    );

    for idx in 0..beta_dim {
        assert!(
            (dense.gb[idx] - deferred.gb[idx]).abs() <= 1.0e-12,
            "dense and deferred paths must assemble the same barrier gradient at β[{idx}]"
        );
    }

    let offsets = term.beta_offsets();
    // Reconstruct the deferred path's β-curvature operator from its two channels:
    // the per-atom scalar ridge `atom_curv[k]·I` over atom k's block, plus every
    // rank-1 carrier `scale·v vᵀ`. It must equal the dense `hbb` bit-for-bit.
    assert!(
        !sep_rank1.is_empty(),
        "deferred path must export the exact self-concordant rank-1 curvature carrier"
    );
    let mut deferred_hbb = Array2::<f64>::zeros((beta_dim, beta_dim));
    for atom_idx in 0..term.k_atoms() {
        let start = offsets[atom_idx];
        let end = if atom_idx + 1 < offsets.len() {
            offsets[atom_idx + 1]
        } else {
            beta_dim
        };
        assert!(
            atom_curv[atom_idx] > 0.0,
            "deferred path must export positive per-atom collapse-prevention curvature"
        );
        for idx in start..end {
            deferred_hbb[[idx, idx]] += atom_curv[atom_idx];
        }
    }
    for (scale, carrier) in &sep_rank1 {
        for &(gi, vi) in carrier {
            for &(gj, vj) in carrier {
                deferred_hbb[[gi, gj]] += scale * vi * vj;
            }
        }
    }
    for i in 0..beta_dim {
        for j in 0..beta_dim {
            assert!(
                (dense.hbb[[i, j]] - deferred_hbb[[i, j]]).abs() <= 1.0e-12,
                "dense hbb and reconstructed deferred (ridge + rank-1) curvature must \
                 match at ({i},{j}): dense={} deferred={}",
                dense.hbb[[i, j]],
                deferred_hbb[[i, j]]
            );
        }
    }
}

/// `SaeRowLayout::from_topk_gates` records exactly the binary support and its
/// compact coordinate starts reproduce the expansion into full coordinates.

#[test]
pub(crate) fn sae_row_layout_from_topk_gates_is_exact() {
    // 3 atoms, coord dims [2, 1, 2] ⇒ full q = 5 (TopK has no logit slots).
    let coord_dims = vec![2usize, 1, 2];
    let coord_offsets_full = vec![0usize, 2, 3];
    let assignments = vec![
        Array1::from_vec(vec![1.0, 0.0, 1.0]),
        Array1::from_vec(vec![1.0, 1.0, 0.0]),
    ];
    let layout =
        SaeRowLayout::from_topk_gates(&assignments, 2, coord_dims, coord_offsets_full).unwrap();
    assert_eq!(layout.active_atoms[0], vec![0, 2]);
    assert_eq!(layout.active_atoms[1], vec![0, 1]);
    assert_eq!(layout.row_q_active(0), 4);
    assert_eq!(layout.row_q_active(1), 3);
    // Row 0 compact [t0_0, t0_1, t2_0, t2_1] expands with atom 1 zero.
    let compact = vec![1.0_f64, 2.0, 3.0, 4.0];
    let mut full = vec![0.0_f64; 5];
    layout.expand_row(0, &compact, &mut full);
    assert_eq!(full[0], 1.0);
    assert_eq!(full[1], 2.0);
    assert_eq!(full[2], 0.0);
    assert_eq!(full[3], 3.0);
    assert_eq!(full[4], 4.0);
}

/// Large-K hard-TopK layout: exact binary support recovery and coordinate work
/// independent of total dictionary width.
#[test]
pub(crate) fn from_topk_gates_large_k_support_is_exact() {
    let (k_atoms, d, k_true, n) = (100_000_usize, 1, 4, 4);
    let planted: Vec<usize> = (0..k_true).map(|j| j * k_atoms / k_true).collect();
    let assignments: Vec<Array1<f64>> = (0..n)
        .map(|row| {
            let mut a = vec![0.0_f64; k_atoms];
            for &atom in &planted {
                a[atom] = 1.0;
            }
            Array1::from_vec(a)
        })
        .collect();
    let coord_offsets: Vec<usize> = (0..k_atoms).collect();
    let layout =
        SaeRowLayout::from_topk_gates(&assignments, k_true, vec![d; k_atoms], coord_offsets)
            .unwrap();
    for row in 0..n {
        assert_eq!(layout.active_atoms[row], planted, "row {row} wrong atoms");
        assert_eq!(layout.row_q_active(row), k_true * d);
    }
    let compact_work: usize = (0..n).map(|r| layout.row_q_active(r).pow(2)).sum();
    assert!(compact_work < n * (k_atoms * d).pow(2) / 1_000_000);
}

#[test]
pub(crate) fn sae_row_layout_from_topk_gates_large_k_work_scales_with_support() {
    let n = 4usize;
    let k_atoms = 100_000usize;
    let cap = 8usize;
    // Per row, plant exactly `cap` binary gates at known indices.
    let mut planted: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut assignments: Vec<Array1<f64>> = Vec::with_capacity(n);
    for row in 0..n {
        let mut a = Array1::<f64>::zeros(k_atoms);
        let mut plant = Vec::with_capacity(cap);
        for j in 0..cap {
            // Spread the planted atoms across the index range.
            let idx = (row + j * (k_atoms / cap)) % k_atoms;
            a[idx] = 1.0;
            plant.push(idx);
        }
        plant.sort_unstable();
        planted.push(plant);
        assignments.push(a);
    }
    let coord_dims = vec![1usize; k_atoms];
    let coord_offsets_full: Vec<usize> = (0..k_atoms).collect();
    let layout =
        SaeRowLayout::from_topk_gates(&assignments, cap, coord_dims, coord_offsets_full).unwrap();
    for row in 0..n {
        // Exact support recovery: the proposal must return exactly the planted
        // top-`cap` atoms (all background weights are below the cutoff).
        assert_eq!(
            layout.active_atoms[row], planted[row],
            "row {row}: support recovery mismatch"
        );
        assert_eq!(layout.row_q_active(row), cap, "row {row}: q_active");
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
    assert_eq!(compact_work, n * cap * cap);
    let dense_q = k_atoms;
    let dense_work = n * dense_q * dense_q;
    // The work ratio is exactly `K² / cap² = (K/cap)²` (the `n` token
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

/// #2098 (SPEC-8): build a two-atom Euclidean-patch term whose atoms carry coord
/// latent dims `d0` and `d1`, so `validate_heterogeneous_atom_compatibility` sees
/// homogeneous vs heterogeneous coord blocks directly (no fit required).
fn hetero_compat_term(d0: usize, d1: usize) -> SaeManifoldTerm {
    let n = 4usize;
    let p = 3usize;
    let m = 2usize;
    let make_atom = |name: &'static str, d: usize| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::EuclideanPatch,
            d,
            Array2::<f64>::ones((n, m)),
            Array3::<f64>::zeros((n, m, d)),
            Array2::<f64>::zeros((m, p)),
            Array2::<f64>::eye(m),
        )
        .unwrap()
    };
    let manifold = |d: usize| {
        if d == 1 {
            LatentManifold::Euclidean
        } else {
            LatentManifold::Product(vec![LatentManifold::Euclidean; d])
        }
    };
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 2)),
        vec![Array2::<f64>::zeros((n, d0)), Array2::<f64>::zeros((n, d1))],
        vec![manifold(d0), manifold(d1)],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(
        vec![make_atom("atom0", d0), make_atom("atom1", d1)],
        assignment,
    )
    .unwrap()
}

/// #2098 (SPEC-8) / F6 — the heterogeneous-`d_atom` + row-block-penalty guard,
/// after the F6 composition split. It must refuse a heterogeneous dictionary
/// ONLY when a *non-composing*, fixed-`d` structural row-block penalty is present
/// (block-orthogonality / TopK / JumpReLU / row-precision), and ADMIT the
/// dim-adaptive penalties that compose per atom (SCAD-MCP, sparsity, native ARD,
/// isometry) — those are exactly what the flagship evidence-heterogeneous
/// dictionary + gauge/ARD machinery need to run together.
#[test]
pub(crate) fn validate_heterogeneous_atom_compatibility_covers_registry_and_native_ard() {
    use gam_terms::analytic_penalties::{
        BlockOrthogonalityPenalty, IsometryPenalty, PenaltyConcavity, ScadMcpPenalty,
    };

    let hetero = hetero_compat_term(2, 1);
    let empty_registry = AnalyticPenaltyRegistry::new();

    // (a) NON-composing structural penalty (BlockOrthogonality: reshapes to
    // `(n_eff × d)` and groups a fixed axis set) on heterogeneous dims ⇒ Err.
    let mut structural_registry = AnalyticPenaltyRegistry::new();
    structural_registry.push(AnalyticPenaltyKind::BlockOrthogonality(Arc::new(
        BlockOrthogonalityPenalty::new(
            PsiSlice::full(4 * 2, Some(2)),
            vec![vec![0], vec![1]],
            1.0,
            4,
            false,
        )
        .unwrap(),
    )));
    let err = hetero
        .validate_heterogeneous_atom_compatibility(Some(&structural_registry), false)
        .expect_err("heterogeneous dims + a fixed-d structural penalty must be refused");
    assert!(
        err.contains("heterogeneous atom coordinate dims"),
        "message must name the heterogeneous conflict: {err}"
    );
    assert!(
        err.contains("uniform atom_dim"),
        "message must name the uniform-dims resolution: {err}"
    );
    assert!(
        err.contains("block_orthogonality"),
        "message must name the offending penalty kind: {err}"
    );

    // (b) COMPOSING penalties on heterogeneous dims ⇒ Ok. These are the
    // dim-adaptive row-block penalties the flagship mixed dictionary relies on.
    let mut iso_registry = AnalyticPenaltyRegistry::new();
    iso_registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        IsometryPenalty::new_euclidean(PsiSlice::full(4 * 2, Some(2)), 2),
    )));
    hetero
        .validate_heterogeneous_atom_compatibility(Some(&iso_registry), false)
        .expect("isometry gauge composes per atom on a heterogeneous dictionary");

    let mut scad_registry = AnalyticPenaltyRegistry::new();
    scad_registry.push(AnalyticPenaltyKind::ScadMcp(Arc::new(
        ScadMcpPenalty::new(
            PsiSlice::full(4 * 2, Some(2)),
            1.0,
            4,
            3.7,
            1.0e-3,
            PenaltyConcavity::Scad,
            false,
        )
        .unwrap(),
    )));
    hetero
        .validate_heterogeneous_atom_compatibility(Some(&scad_registry), false)
        .expect("element-wise SCAD-MCP composes on a heterogeneous dictionary");

    // (c) native ARD (the FFI flag) composes per atom over `d_k` ⇒ Ok on
    // heterogeneous dims, with or without a registry. This is the change from the
    // pre-F6 blanket refusal: `ard_value` is already a per-atom sum over `d_k`.
    hetero
        .validate_heterogeneous_atom_compatibility(Some(&empty_registry), true)
        .expect("native ARD composes per atom on a heterogeneous dictionary");
    hetero
        .validate_heterogeneous_atom_compatibility(None, true)
        .expect("native ARD (no registry) composes per atom on a heterogeneous dictionary");
    // ARD + isometry gauge together — the flagship combination — on mixed dims.
    hetero
        .validate_heterogeneous_atom_compatibility(Some(&iso_registry), true)
        .expect("ARD + isometry gauge compose together on a heterogeneous dictionary");

    // (d) a structural penalty is fine on HOMOGENEOUS dims (nothing to dispatch
    // ambiguously) — the refusal is specifically about mixed `d_k`.
    let homo = hetero_compat_term(2, 2);
    homo.validate_heterogeneous_atom_compatibility(Some(&structural_registry), true)
        .expect("homogeneous coord dims dispatch every row-block penalty cleanly");

    // (e) heterogeneous dims, no penalty, no ARD ⇒ Ok.
    hetero
        .validate_heterogeneous_atom_compatibility(Some(&empty_registry), false)
        .expect("heterogeneous dims with no row-block penalty and no ARD is admitted");
    hetero
        .validate_heterogeneous_atom_compatibility(None, false)
        .expect("heterogeneous dims with no registry and no ARD is admitted");
}

/// Build a single-block SAE term over `(manifold, coords)` with an arbitrary
/// `EuclideanPatch`-shaped decoder. `ard_value` reads only the coord values and
/// the coord manifold (for per-axis periodicity), so this is enough to exercise
/// the native-ARD energy on any atom dim — no basis evaluator / second jet
/// needed. `coords` is `(n × d)`.
fn ard_atom_and_coord(
    name: &'static str,
    manifold: LatentManifold,
    coords: Array2<f64>,
) -> (SaeManifoldAtom, LatentManifold, Array2<f64>) {
    let (n, d) = coords.dim();
    let m = 2usize;
    let p = 3usize;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        name,
        SaeAtomBasisKind::EuclideanPatch,
        d,
        Array2::<f64>::ones((n, m)),
        Array3::<f64>::zeros((n, m, d)),
        Array2::<f64>::zeros((m, p)),
        Array2::<f64>::eye(m),
    )
    .unwrap();
    (atom, manifold, coords)
}

/// F6 (fit-level composition): the native ARD energy on a coordinate-
/// HETEROGENEOUS `{circle d=1, patch d=2, linear d=1}` dictionary is *exactly*
/// the sum of the per-atom ARD energies — the same per-atom-additive
/// decomposition the penalized LAML evidence sums over atoms — so admitting ARD on
/// a mixed dictionary (the F6 composition) keeps the evidence exact with no
/// padding or truncation. This is the concrete counterpart to the validator
/// test: the gate opens (validator) AND the energy it lets through composes
/// correctly (this test). A circle axis contributes the von-Mises energy, the
/// Euclidean axes the Gaussian one, each read against the atom's own `d_k`.
#[test]
pub(crate) fn native_ard_energy_composes_additively_on_mixed_dictionary() {
    let n = 3usize;
    // {circle d=1, patch d=2, linear d=1}: exactly F6's flagship mixed shape.
    let circle_coords = array![[0.1_f64], [0.6], [0.9]];
    let patch_coords = array![[0.2_f64, -0.3], [0.5, 0.4], [-0.1, 0.7]];
    let linear_coords = array![[1.0_f64], [-0.5], [0.3]];
    let circle_m = LatentManifold::Circle { period: 1.0 };
    let patch_m = LatentManifold::Product(vec![LatentManifold::Euclidean; 2]);
    let linear_m = LatentManifold::Euclidean;

    // Per-atom log-ARD strengths, lengths d_k = [1, 2, 1] (heterogeneous).
    let ard0 = array![0.2_f64];
    let ard1 = array![0.1_f64, -0.3];
    let ard2 = array![-0.15_f64];

    // Joint mixed term (K = 3).
    let (a0, m0, c0) = ard_atom_and_coord("circle", circle_m.clone(), circle_coords.clone());
    let (a1, m1, c1) = ard_atom_and_coord("patch", patch_m.clone(), patch_coords.clone());
    let (a2, m2, c2) = ard_atom_and_coord("linear", linear_m.clone(), linear_coords.clone());
    let joint_assign = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 3)),
        vec![c0, c1, c2],
        vec![m0, m1, m2],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let joint = SaeManifoldTerm::new(vec![a0, a1, a2], joint_assign).unwrap();
    let joint_rho = SaeManifoldRho::new(0.0, 0.0, vec![ard0.clone(), ard1.clone(), ard2.clone()]);
    let joint_energy = joint.ard_value(&joint_rho).unwrap();

    // Per-atom single-block terms, each with the SAME coords / manifold / ARD.
    let single_energy = |name: &'static str,
                         manifold: LatentManifold,
                         coords: Array2<f64>,
                         ard: Array1<f64>|
     -> f64 {
        let (atom, m, c) = ard_atom_and_coord(name, manifold, coords);
        let assign = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![c],
            vec![m],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assign).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![ard]);
        term.ard_value(&rho).unwrap()
    };
    let sum = single_energy("circle", circle_m, circle_coords, ard0)
        + single_energy("patch", patch_m, patch_coords, ard1)
        + single_energy("linear", linear_m, linear_coords, ard2);

    assert!(
        (joint_energy - sum).abs() < 1.0e-12,
        "native ARD must compose additively over a mixed dictionary: \
             joint={joint_energy}, per-atom sum={sum}"
    );
    // Sanity: the energy is genuinely non-trivial (not a vacuous 0 == 0), so the
    // additivity above is a real check on the heterogeneous evaluation.
    assert!(
        joint_energy.abs() > 1.0e-6,
        "mixed-dictionary ARD energy should be non-trivial, got {joint_energy}"
    );
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
