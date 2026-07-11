use gam::solver::rho_optimizer::OuterObjective;
use gam::terms::decoders::gated_decoder::GatedSAEDecoder;
use gam::terms::latent::{LatentCoordValues, LatentIdMode};
use gam::terms::sae::manifold::{
    AssignmentMode, GumbelTemperatureSchedule, SaeAssignment, SaeManifoldAtom,
    SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm, ScheduleKind,
};
use ndarray::{Array2, Array3, array};

#[test]
fn latent_coord_assignment_decode_roundtrip_matches_dictionary_atom() {
    let coords =
        LatentCoordValues::from_matrix(array![[0.3, -0.7], [1.2, 0.5]].view(), LatentIdMode::None);
    let logits = array![[20.0, -20.0], [-20.0, 20.0]];
    let assignment = SaeAssignment::from_blocks_with_mode(
        logits,
        vec![coords.as_matrix(), coords.as_matrix()],
        AssignmentMode::softmax(1e-3),
    )
    .expect("test setup must construct assignment");

    let atom0 = SaeManifoldAtom::new(
        "a0",
        gam::terms::sae::manifold::SaeAtomBasisKind::Precomputed("dict0".into()),
        2,
        array![[1.0], [1.0]],
        Array3::zeros((2, 1, 2)),
        array![[2.0, -1.0]],
        array![[1.0]],
    )
    .expect("test setup must construct atom0");
    let atom1 = SaeManifoldAtom::new(
        "a1",
        gam::terms::sae::manifold::SaeAtomBasisKind::Precomputed("dict1".into()),
        2,
        array![[1.0], [1.0]],
        Array3::zeros((2, 1, 2)),
        array![[-3.0, 4.0]],
        array![[1.0]],
    )
    .expect("test setup must construct atom1");

    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).expect("term should build");
    let fitted = term.try_fitted().expect("fitted should evaluate");

    assert!(
        (fitted[[0, 0]] - 2.0).abs() < 1e-6
            && (fitted[[0, 1]] + 1.0).abs() < 1e-6
            && (fitted[[1, 0]] + 3.0).abs() < 1e-6
            && (fitted[[1, 1]] - 4.0).abs() < 1e-6,
        "Assignment-then-decode should round-trip each row back to its selected dictionary atom even at random latent_dim."
    );
}

#[test]
fn sae_assignment_modes_follow_documented_behavior() {
    let coord_blocks = vec![
        array![[0.0], [0.0]],
        array![[0.0], [0.0]],
        array![[0.0], [0.0]],
    ];

    let soft = SaeAssignment::from_blocks_with_mode(
        array![[1.0, 0.0, -1.0], [3.0, 2.0, 1.0]],
        coord_blocks.clone(),
        AssignmentMode::softmax(0.7),
    )
    .expect("softmax assignment should build");
    for row in 0..soft.n_obs() {
        let w = soft
            .try_assignments_row(row)
            .expect("softmax row should evaluate");
        let s: f64 = w.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-10,
            "Softmax assignments must be normalized and sum to one for every row."
        );
    }

    let ordered_beta_bernoulli = SaeAssignment::from_blocks_with_mode(
        array![[0.0, 0.0, 0.0]],
        vec![array![[0.0]], array![[0.0]], array![[0.0]]],
        AssignmentMode::ordered_beta_bernoulli(1.0, 0.1, false),
    )
    .expect("ordered Beta--Bernoulli assignment should build");
    let ordered_beta_bernoulli_row = ordered_beta_bernoulli
        .try_assignments_row(0)
        .expect("ordered Beta--Bernoulli row should evaluate");
    assert!(
        ordered_beta_bernoulli_row
            .iter()
            .all(|value| (*value - 0.5).abs() < 1.0e-12),
        "The ordered prior is scored once by the penalty and must not be multiplied into the posterior-mean reconstruction gate."
    );

    let threshold_gate = SaeAssignment::from_blocks_with_mode(
        array![[0.2, 0.6, -2.0]],
        vec![array![[0.0]], array![[0.0]], array![[0.0]]],
        AssignmentMode::threshold_gate(0.5, 0.5),
    )
    .expect("threshold-gate assignment should build");
    let threshold_gate_row = threshold_gate
        .try_assignments_row(0)
        .expect("threshold-gate row should evaluate");
    let expected = [
        1.0 / (1.0 + 0.6_f64.exp()),
        1.0 / (1.0 + (-0.2_f64).exp()),
        1.0 / (1.0 + 5.0_f64.exp()),
    ];
    assert!(
        threshold_gate_row
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-12),
        "ThresholdGate must return the exact smooth logistic gate on both sides of its center."
    );
}

#[test]
fn gated_sae_decoder_reconstructs_dictionary_atom_at_zero_residual() {
    let decoder =
        GatedSAEDecoder::new(Array2::eye(3), Array2::eye(3)).expect("decoder should build");
    let x = array![1.2, -0.8, 0.4];
    let y = decoder.decode_row(x.view()).expect("decode must succeed");
    assert!(
        (y[0] - x[0]).abs() < 1e-12 && (y[1] - x[1]).abs() < 1e-12 && (y[2] - x[2]).abs() < 1e-12,
        "With zero training residual and identity dictionary, gate activation plus linear decode should reproduce the original dictionary atom exactly."
    );
}

/// Build a minimal single-atom SAE term with `n` rows, `d`-dim latent, and a
/// precomputed identity-like decoder basis. The coordinates of axis 0 are
/// `coord_axis0`; the remaining axes are set to a constant so one axis can be
/// driven toward collapse independently. The decoder atom is identifiable
/// (the precomputed basis is the supplied `phi`).
fn build_collapse_probe_term(coords: Array2<f64>) -> SaeManifoldTerm {
    let n = coords.nrows();
    let d = coords.ncols();
    let assignment = SaeAssignment::from_blocks_with_mode(
        Array2::<f64>::zeros((n, 1)),
        vec![coords.clone()],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment should build");
    // Identity basis φ(t) = t: m = d basis columns, basis_values = the coords,
    // Jacobian = per-row I_d. This makes the decoded output genuinely depend on
    // the latent coords, so each axis is IDENTIFIED by data and carries real
    // per-axis data curvature in the inner Hessian. Decoder B = I_d routes axis
    // j to output channel j (p = d). With real curvature, the ARD α on a
    // small-but-nonzero-spread axis has a genuine FINITE interior REML optimum
    // (≈ √(n·c / ‖t‖²)) — far below the deleted α = n/‖t‖² rule's explosion
    // past the log-α clamp. A *zero*-Jacobian basis would leave the coords
    // unidentified (the criterion would be flat in α), which is why this probe
    // supplies real first-jet curvature.
    let basis_values = coords.clone();
    let mut basis_jacobian = Array3::<f64>::zeros((n, d, d));
    for i in 0..n {
        for j in 0..d {
            basis_jacobian[[i, j, j]] = 1.0;
        }
    }
    let atom = SaeManifoldAtom::new(
        "a",
        gam::terms::sae::manifold::SaeAtomBasisKind::Precomputed("dict".into()),
        d,
        basis_values,
        basis_jacobian,
        Array2::<f64>::eye(d),
        Array2::<f64>::zeros((d, d)),
    )
    .expect("atom should build");
    SaeManifoldTerm::new(vec![atom], assignment).expect("term should build")
}

#[test]
fn penalized_quasi_laplace_criterion_keeps_alpha_finite_on_collapsing_axis() {
    // A coordinate axis that collapses (‖t_·j‖² → ~0) would make the deleted
    // α = n/‖t‖² rule explode toward the clamp ceiling. The true REML
    // criterion instead has a finite interior argmin in log α because the
    // ½log|H| Laplace term (rising in α) balances the −½n·logα data/prior term
    // (falling in α). Sweep log α on the collapsing axis and assert the argmin
    // is finite and interior — no clamp needed.
    let coords = array![[1.0, 0.01], [3.0, -0.01], [-2.0, 0.01], [0.5, -0.01]];
    // p = d = 2 (identity basis routes axis j → output channel j). target ≈ the
    // coords so the decoder fits β ≈ I with small residual; the ARD α on the
    // small-spread axis 1 is then set by the curvature/‖t‖² balance and has a
    // finite interior optimum ≈ √(n·c/‖t‖²) well inside the swept grid.
    let target = array![[1.0, 0.01], [3.0, -0.01], [-2.0, 0.01], [0.5, -0.01]];
    let base_term = build_collapse_probe_term(coords);

    let log_alpha_grid = [-6.0_f64, -2.0, 0.0, 2.0, 6.0, 10.0, 16.0];
    let mut best = (f64::INFINITY, f64::NAN);
    for &la in &log_alpha_grid {
        // Fresh term per grid point: penalized_quasi_laplace_criterion runs the inner (t,β) fit
        // IN PLACE, so reusing one term would make each evaluation start from
        // the previous α's fitted state (path-dependent, monotone drift). The
        // criterion is a function of ρ from a FIXED initial state, so each
        // sweep point must start from the same baseline.
        let mut term = base_term.clone();
        // Axis 0 keeps a mild prior; axis 1 (the collapsing one) is swept.
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0, la]]);
        let (v, _loss) = term
            .penalized_quasi_laplace_criterion(target.view(), &rho, None, 50, 1.0, 1.0e-6, 1.0e-6)
            .expect("criterion should evaluate");
        assert!(
            v.is_finite(),
            "criterion must be finite at log α={la}; got {v}"
        );
        if v < best.0 {
            best = (v, la);
        }
    }
    let argmin = best.1;
    assert!(
        argmin > log_alpha_grid[0] && argmin < *log_alpha_grid.last().unwrap(),
        "REML criterion argmin over log α must be a finite INTERIOR minimum \
         (balancing ½log|H| against −½n·logα), not pinned to a grid endpoint; \
         got argmin={argmin}"
    );
}

#[test]
fn penalized_quasi_laplace_criterion_has_interior_minimum_in_log_lambda_smooth() {
    // The −½·p·rank(S)·log λ_smooth Occam term gives the criterion a finite
    // interior argmin in log λ_smooth: too small λ underfits (large penalised
    // loss + logdet), too large λ is penalised by the rank·logλ normaliser.
    // Use a 2-basis atom with a non-trivial rank-1 difference penalty so
    // rank(S) > 0 and the Occam term is active.
    let n = 6;
    let coords = array![[0.2], [0.8], [-0.5], [1.3], [-1.1], [0.4]];
    let assignment = SaeAssignment::from_blocks_with_mode(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment should build");
    // 2 basis columns, p = 1; smooth penalty is the rank-1 first-difference
    // penalty S = [[1,-1],[-1,1]] (null space = constants), so rank(S) = 1.
    let phi = array![
        [1.0, 0.2],
        [1.0, 0.8],
        [1.0, -0.5],
        [1.0, 1.3],
        [1.0, -1.1],
        [1.0, 0.4]
    ];
    let atom = SaeManifoldAtom::new(
        "a",
        gam::terms::sae::manifold::SaeAtomBasisKind::Precomputed("dict".into()),
        1,
        phi,
        Array3::zeros((n, 2, 1)),
        array![[0.5], [0.5]],
        array![[1.0, -1.0], [-1.0, 1.0]],
    )
    .expect("atom should build");
    let base_term = SaeManifoldTerm::new(vec![atom], assignment).expect("term should build");
    // Signal target = 10·(1 − coord) (β0=+10, β1=−10). The rank-1 penalty
    // S=[[1,-1],[-1,1]] penalises (β0−β1)², so over-smoothing (large λ → β0≈β1)
    // collapses the fit onto span(1+coord) — orthogonal to this 10·(1−coord)
    // signal — so the data-fit cost of over-smoothing is LARGE (residual grows
    // to O(10²) per row), dominating the O(logλ) logdet/Occam scale. That steep
    // bias/variance cost is what turns the criterion back up at large λ, giving
    // the −½·rank·logλ Occam balance a genuine finite INTERIOR argmin (a target
    // near the penalty null space would leave it monotone/endpoint-pinned).
    let target = array![[8.0], [2.0], [15.0], [-3.0], [21.0], [6.0]];

    let log_lambda_grid = [-8.0_f64, -4.0, -1.0, 1.0, 4.0, 8.0, 12.0];
    let mut best = (f64::INFINITY, f64::NAN);
    for &ll in &log_lambda_grid {
        // Fresh term per grid point — penalized_quasi_laplace_criterion fits (t,β) in place, so a
        // shared term would carry the previous λ's fit forward and make the
        // sweep path-dependent (monotone), masking the true interior optimum.
        // Run the inner fit to convergence (50 iters) so loss.total() and
        // ½log|H| reflect the genuine λ-conditioned penalised optimum the REML
        // criterion is defined at.
        let mut term = base_term.clone();
        let rho = SaeManifoldRho::new(0.0, ll, vec![array![0.0]]);
        let (v, _loss) = term
            .penalized_quasi_laplace_criterion(target.view(), &rho, None, 50, 1.0, 1.0e-6, 1.0e-6)
            .expect("criterion should evaluate");
        assert!(
            v.is_finite(),
            "criterion must be finite at log λ={ll}; got {v}"
        );
        if v < best.0 {
            best = (v, ll);
        }
    }
    let argmin = best.1;
    assert!(
        argmin > log_lambda_grid[0] && argmin < *log_lambda_grid.last().unwrap(),
        "REML criterion over log λ_smooth must have a finite INTERIOR argmin \
         (validating the −½·rank·logλ Occam term's presence and sign); got \
         argmin={argmin} (monotone to an endpoint means the term is missing or \
         wrong-signed)"
    );
}

#[test]
fn efs_ard_fixed_point_recovers_cost_criterion_argmin_and_stays_finite() {
    // Build a small single-atom SAE with a 2-axis latent. Axis 1 is driven
    // near collapse (‖t‖²→~0) to exercise the posterior-variance term. The EFS
    // Fellner-Schall/Mackay fixed point α_new = n/(‖t‖²+tr(H⁻¹)) must:
    //   (a) converge to a FINITE α on the collapsing axis (no clamp), and
    //   (b) agree with the α that minimizes the v1 cost criterion.
    let coords = array![
        [1.0, 1.0e-5],
        [3.0, -1.0e-5],
        [-2.0, 1.0e-5],
        [0.5, -1.0e-5]
    ];
    // `build_collapse_probe_term`'s identity decoder B = I_2 routes latent axis
    // j → output channel j, so `output_dim = 2` and the target MUST be (n, 2)
    // (a 1-column target indexes past the reconstruction's second channel during
    // per-row assembly). Channel 0 carries a real signal so the reconstruction
    // residual — and hence the dispersion φ̂ — is nonzero, keeping the ARD α
    // finite. Channel 1 carries the same small-but-identified ±0.01 signal the
    // sibling `penalized_quasi_laplace_criterion_keeps_alpha_finite_on_collapsing_axis` uses to hold
    // axis 1 near collapse (‖t_·1‖² → ~0 while its decoder column stays nonzero,
    // so the axis keeps real data curvature): exactly the regime where the FS
    // denominator on axis 1 is dominated by the posterior-variance trace
    // tr_1(H⁻¹) that this test exercises.
    let target = array![[0.4, 0.01], [1.1, -0.01], [-0.7, 0.01], [0.2, -0.01]];
    let term = build_collapse_probe_term(coords);

    let init_rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0, 0.0]]);
    let mut obj = SaeManifoldOuterObjective::new(
        term.clone(),
        target.clone(),
        None,
        init_rho.clone(),
        50,
        1.0,
        1.0e-6,
        1.0e-6,
    );

    // Iterate the EFS fixed point: rho_new = rho + steps (additive in log
    // space = multiplicative FS). Converge when the step norm is tiny.
    let mut rho_flat = init_rho.to_flat();
    let mut converged_log_alpha1 = f64::NAN;
    for _ in 0..60 {
        let efs = obj.eval_efs(&rho_flat).expect("EFS eval should succeed");
        assert_eq!(efs.steps.len(), rho_flat.len());
        let mut step_norm = 0.0_f64;
        for (i, s) in efs.steps.iter().enumerate() {
            assert!(
                s.is_finite(),
                "EFS step[{i}]={s} must be finite (no α blow-up)"
            );
            rho_flat[i] += s;
            step_norm += s * s;
        }
        // index 3 = atom 0, axis 1 (layout [sparse, smooth, axis0, axis1]).
        converged_log_alpha1 = rho_flat[3];
        assert!(
            converged_log_alpha1.is_finite() && converged_log_alpha1 < 30.0,
            "EFS α on the collapsing axis must stay finite (no clamp); got log α={converged_log_alpha1}"
        );
        if step_norm.sqrt() < 1.0e-8 {
            break;
        }
    }

    // Cross-check: the EFS-converged α minimizes the v1 cost criterion. Sweep
    // log α on axis 1 with all other ρ at the EFS optimum and confirm the
    // criterion at the EFS α is no worse than its neighbours (interior min).
    // Each cost_at call clones a FRESH term from the baseline — penalized_quasi_laplace_criterion
    // fits (t,β) in place, so a shared term would make the three probes
    // path-dependent. Inner fit runs to convergence (50 iters).
    let base_term = term;
    let cost_at = |la1: f64| -> f64 {
        let mut t = base_term.clone();
        let rho = SaeManifoldRho::new(rho_flat[0], rho_flat[1], vec![array![rho_flat[2], la1]]);
        t.penalized_quasi_laplace_criterion(target.view(), &rho, None, 50, 1.0, 1.0e-6, 1.0e-6)
            .expect("criterion should evaluate")
            .0
    };
    let v_star = cost_at(converged_log_alpha1);
    let v_lo = cost_at(converged_log_alpha1 - 1.0);
    let v_hi = cost_at(converged_log_alpha1 + 1.0);
    assert!(
        v_star <= v_lo + 1.0e-6 && v_star <= v_hi + 1.0e-6,
        "EFS fixed-point α must sit at (or below) the v1 cost-criterion minimum: \
         V(α*)={v_star}, V(α*/e)={v_lo}, V(α*·e)={v_hi} — the FS step and the exact \
         criterion must agree at the optimum"
    );
}

#[test]
fn temperature_schedule_is_applied_each_iteration_and_near_zero_behaves_like_argmax() {
    let logits = array![[2.0, 1.0, -4.0]];
    let mut term = {
        let assignment = SaeAssignment::from_blocks_with_mode(
            logits.clone(),
            vec![array![[0.0]], array![[0.0]], array![[0.0]]],
            AssignmentMode::softmax(2.0),
        )
        .expect("assignment should build");
        let atoms = (0..3)
            .map(|k| {
                SaeManifoldAtom::new(
                    format!("a{k}"),
                    gam::terms::sae::manifold::SaeAtomBasisKind::Precomputed("dict".into()),
                    1,
                    array![[1.0]],
                    Array3::zeros((1, 1, 1)),
                    array![[1.0]],
                    array![[1.0]],
                )
                .expect("atom should build")
            })
            .collect::<Vec<_>>();
        SaeManifoldTerm::new(atoms, assignment).expect("term should build")
    };

    let schedule = GumbelTemperatureSchedule::new(2.0, 1e-6, ScheduleKind::Geometric { rate: 0.1 })
        .expect("schedule should build");
    term.set_temperature_schedule(schedule)
        .expect("schedule should apply");
    assert!(
        (term.assignment.mode.temperature() - 2.0).abs() < 1e-12,
        "When a temperature schedule is provided, the assignment mode should immediately use the schedule temperature for the current iteration."
    );

    let near_zero = SaeAssignment::from_blocks_with_mode(
        logits,
        vec![array![[0.0]], array![[0.0]], array![[0.0]]],
        AssignmentMode::softmax(1e-6),
    )
    .expect("near-zero temperature assignment should build");
    let w = near_zero
        .try_assignments_row(0)
        .expect("assignments should evaluate");
    assert!(
        w[0] > 1.0 - 1e-8 && w[1] < 1e-8 && w[2] < 1e-8,
        "As temperature approaches zero, assignment should collapse to argmax on the largest logit."
    );
}
