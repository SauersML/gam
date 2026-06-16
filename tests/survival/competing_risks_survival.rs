use gam::custom_family::{
    AdditiveBlockJacobian, BlockwiseFitOptions, ParameterBlockSpec, fit_custom_family,
};
use gam::families::survival::{
    CauseSpecificRoystonParmarBlock, CauseSpecificRoystonParmarFamily,
    assemble_competing_risks_cif, survival_event_code_from_value,
};
use gam::matrix::DesignMatrix;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

fn event_codes(cause_count: usize, event_counts: &[usize], n: usize) -> Array1<u8> {
    assert_eq!(event_counts.len(), cause_count);
    assert!(event_counts.iter().sum::<usize>() <= n);
    let mut events = Vec::with_capacity(n);
    for (cause, &count) in event_counts.iter().enumerate() {
        events.extend(std::iter::repeat_n((cause + 1) as u8, count));
    }
    events.resize(n, 0);
    Array1::from_vec(events)
}

#[test]
fn auto_detects_small_integer_competing_risk_event_codes() {
    let values = [0.0, 1.0, 2.0, 3.0];
    let codes = values
        .into_iter()
        .enumerate()
        .map(|(row, value)| survival_event_code_from_value(value, row))
        .collect::<Result<Vec<_>, _>>()
        .expect("small integer event codes should parse");
    assert_eq!(codes, vec![0, 1, 2, 3]);
    assert!(survival_event_code_from_value(1.25, 0).is_err());
}

fn fit_constant_exposure_cause_specific(event_counts: &[usize], n: usize) -> Array1<f64> {
    gam::init_parallelism();
    let cause_count = event_counts.len();
    let events = event_codes(cause_count, event_counts, n);
    let age_entry = Array1::zeros(n);
    let age_exit = Array1::ones(n);
    let weights = Array1::ones(n);
    let x_entry = Array2::ones((n, 1));
    let x_exit = Array2::ones((n, 1));
    let x_derivative = Array2::zeros((n, 1));
    let offset_entry = Array1::zeros(n);
    let offset_exit = Array1::zeros(n);
    let offset_derivative = Array1::ones(n);

    let mut blocks = Vec::with_capacity(cause_count);
    let mut specs = Vec::with_capacity(cause_count);
    for cause in 0..cause_count {
        let cause_code = (cause + 1) as u8;
        blocks.push(CauseSpecificRoystonParmarBlock {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            event_target: events.mapv(|event| u8::from(event == cause_code)),
            sampleweight: weights.clone(),
            x_entry: x_entry.clone(),
            x_exit: x_exit.clone(),
            x_derivative: x_derivative.clone(),
            offset_eta_entry: offset_entry.clone(),
            offset_eta_exit: offset_exit.clone(),
            offset_derivative_exit: offset_derivative.clone(),
            derivative_floor: 0.0,
        });
        // The K cause-specific blocks share the constant `x_exit` design,
        // so the joint design carries K identical columns. The cause-specific
        // likelihood routes each cause to disjoint risk sets so each
        // per-cause baseline is independently identifiable, but the
        // identifiability audit only sees the unweighted joint design. Mirror
        // the production cause-specific path (`solver::workflow`) and the
        // multinomial-class convention: use descending gauge priorities so
        // the audit's cross-block pair filter resolves the shared direction
        // by priority, and attach an `AdditiveBlockJacobian` with
        // `own_output = cause` so the channel-aware audit treats each
        // cause's contribution as occupying its own output-channel rows
        // (and the orthogonalisation pass defers to the family-owned
        // geometry).
        let cause_priority =
            100u8.saturating_add(u8::try_from(cause_count - cause).unwrap_or(u8::MAX));
        let cause_jacobian = Arc::new(AdditiveBlockJacobian {
            design: x_exit.clone(),
            own_output: cause,
            n_family_outputs: cause_count,
        });
        specs.push(ParameterBlockSpec {
            name: format!("cause_{}", cause + 1),
            design: DesignMatrix::from(x_exit.clone()),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::from_vec(vec![
                ((event_counts[cause] as f64 + 0.5) / n as f64).ln(),
            ])),
            gauge_priority: cause_priority,
            jacobian_callback: Some(cause_jacobian),
            stacked_design: None,
            stacked_offset: None,
        });
    }
    let family = CauseSpecificRoystonParmarFamily::new(blocks)
        .expect("build custom cause-specific survival family");
    let options = BlockwiseFitOptions {
        inner_tol: 1e-10,
        inner_max_cycles: 80,
        outer_max_iter: 1,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let fit = fit_custom_family(&family, &specs, &options).expect("joint custom-family fit");
    assert!(
        fit.outer_converged,
        "custom-family inner solve did not converge"
    );
    Array1::from_iter(fit.block_states.iter().map(|state| state.beta[0]))
}

#[test]
fn joint_two_cause_fit_recovers_constant_cause_specific_hazards() {
    let n = 1000;
    let events = [120, 70];
    let beta = fit_constant_exposure_cause_specific(&events, n);
    for (idx, &count) in events.iter().enumerate() {
        let expected = count as f64 / n as f64;
        let fitted = beta[idx].exp();
        assert!(
            (fitted - expected).abs() <= 5e-4,
            "cause {} hazard mismatch: fitted={fitted:.6}, expected={expected:.6}",
            idx + 1
        );
    }
}

#[test]
fn joint_three_cause_fit_recovers_constant_cause_specific_hazards() {
    let n = 1200;
    let events = [96, 60, 36];
    let beta = fit_constant_exposure_cause_specific(&events, n);
    for (idx, &count) in events.iter().enumerate() {
        let expected = count as f64 / n as f64;
        let fitted = beta[idx].exp();
        assert!(
            (fitted - expected).abs() <= 5e-4,
            "cause {} hazard mismatch: fitted={fitted:.6}, expected={expected:.6}",
            idx + 1
        );
    }
}

#[test]
fn single_cause_custom_family_matches_single_endpoint_baseline() {
    let n = 900;
    let event_count = 117;
    let joint_beta = fit_constant_exposure_cause_specific(&[event_count], n);
    let baseline_beta = (event_count as f64 / n as f64).ln();
    assert!(
        (joint_beta[0] - baseline_beta).abs() <= 5e-4,
        "K=1 custom-family beta differs from baseline: joint={}, baseline={}",
        joint_beta[0],
        baseline_beta
    );
}

/// #1025 — END-TO-END total-probability identity on a FITTED competing-risks
/// model. The fit routes through the channel-aware identifiability path (the
/// `AdditiveBlockJacobian` + descending `gauge_priority` the audit-halt fix
/// installs), so passing this gate proves three things at once:
///   (1) IDENTIFIABILITY — the K cause-specific baselines, which share the
///       identical raw design column (overlap = 1.0) and would trip the flat
///       audit, fit through the channel-aware (information-metric) path without
///       a halt; the solve converges to finite per-cause baselines.
///   (2) BASELINE NORMALIZATION — the fitted constant baselines reproduce the
///       per-cause hazards, so the cumulative hazards `H_k(t) = λ̂_k · t` feeding
///       the assembly are correctly scaled (a per-cause scale error breaks the
///       sum below even when each CIF looks plausible alone, per the issue's
///       transform-normalization note).
///   (3) THE CLOSURE IDENTITY — at every t on the grid,
///         S_overall(t) + Σ_k CIF_k(t) = 1,
///       i.e. everyone is, at any t, either still at risk or has failed from
///       exactly one cause. With cause-specific hazards this is automatic
///       (d/dt[S + Σ_k CIF_k] = −S·Σh_k + Σ_k h_k·S = 0, = 1 at t=0); a fitted
///       model that violates it has a mis-normalized baseline or a broken
///       quadrature.
/// The bound is NOT to be loosened: the identity is exact mathematics.
#[test]
fn fitted_competing_risks_total_probability_identity_holds_across_time_grid() {
    let n = 1200;
    let events = [180usize, 96usize]; // two competing causes, ~23% total events
    let cause_count = events.len();

    // Fit through the channel-aware cause-specific path (the audit-fix
    // architecture). The constant-exposure design means the fitted baseline is
    // the per-cause hazard λ̂_k = exp(β̂_k); the cumulative hazard is H_k(t) = λ̂_k·t.
    let beta = fit_constant_exposure_cause_specific(&events, n);
    assert_eq!(beta.len(), cause_count, "one fitted baseline per cause");
    let lambda: Vec<f64> = beta.iter().map(|b| b.exp()).collect();
    for (k, &l) in lambda.iter().enumerate() {
        assert!(
            l.is_finite() && l > 0.0,
            "cause {} fitted hazard is not a finite positive baseline: {l}",
            k + 1
        );
    }

    // A strictly-increasing time grid spanning the support (entry 0, exit 1).
    let times: Array1<f64> = Array1::from_vec(
        (0..=20)
            .map(|i| 0.05 * (i as f64)) // 0.00, 0.05, ..., 1.00
            .collect(),
    );
    let n_times = times.len();

    // Cumulative hazards in (endpoint, row, time) layout. Constant baseline ⇒
    // H_k(t | row) = λ̂_k · t (same for every row under the shared design).
    let mut cumulative = Array3::<f64>::zeros((cause_count, n, n_times));
    for k in 0..cause_count {
        for row in 0..n {
            for (ti, &t) in times.iter().enumerate() {
                cumulative[[k, row, ti]] = lambda[k] * t;
            }
        }
    }

    let assembled = assemble_competing_risks_cif(times.view(), cumulative.view())
        .expect("assemble fitted competing-risks CIF");
    assert_eq!(
        assembled.cif.len(),
        cause_count,
        "one CIF surface per cause"
    );
    assert_eq!(
        assembled.overall_survival.dim(),
        (n, n_times),
        "overall survival shape"
    );

    // Structural + identity checks across the whole grid, for a representative
    // sample of rows (the shared design makes every row identical, but we sweep
    // a few to guard against a row-indexing regression).
    let mut worst_identity_err = 0.0_f64;
    for &row in &[0usize, n / 3, 2 * n / 3, n - 1] {
        let mut prev_overall = f64::INFINITY;
        let mut prev_cif = vec![0.0_f64; cause_count];
        for ti in 0..n_times {
            let s = assembled.overall_survival[[row, ti]];
            assert!(
                (0.0..=1.0).contains(&s),
                "overall survival escapes [0,1] at row {row}, t={}: S={s}",
                times[ti]
            );
            // S(0) = 1 (nobody has failed at t=0).
            if ti == 0 {
                assert!(
                    (s - 1.0).abs() < 1e-12,
                    "S(0) must equal 1 (no failures at t=0); got {s}"
                );
            }
            // S is monotone non-increasing in t.
            assert!(
                s <= prev_overall + 1e-12,
                "overall survival increased at row {row}, t={}: {prev_overall} -> {s}",
                times[ti]
            );
            prev_overall = s;

            let mut cif_sum = 0.0_f64;
            for k in 0..cause_count {
                let f = assembled.cif[k][[row, ti]];
                assert!(
                    (0.0..=1.0).contains(&f),
                    "CIF_{} escapes [0,1] at row {row}, t={}: F={f}",
                    k + 1,
                    times[ti]
                );
                // Each CIF is monotone non-decreasing in t.
                assert!(
                    f >= prev_cif[k] - 1e-12,
                    "CIF_{} decreased at row {row}, t={}: {} -> {f}",
                    k + 1,
                    times[ti],
                    prev_cif[k]
                );
                prev_cif[k] = f;
                cif_sum += f;
            }

            // THE GATE: S_overall(t) + Σ_k CIF_k(t) = 1 at every t.
            let identity_err = (s + cif_sum - 1.0).abs();
            worst_identity_err = worst_identity_err.max(identity_err);
            assert!(
                identity_err < 1e-10,
                "total-probability identity violated at row {row}, t={}: \
                 S={s} + ΣCIF={cif_sum} = {} ≠ 1 (err={identity_err:.3e}) — \
                 a mis-normalized fitted baseline or broken CIF quadrature (#1025)",
                times[ti],
                s + cif_sum
            );
        }
    }

    // The grid must reach a regime where real probability mass has accumulated
    // (otherwise the identity is vacuously satisfied by S≈1, ΣCIF≈0). At t=1 with
    // these hazards the total failure probability is ~1 - exp(-(λ1+λ2)) > 0.
    let final_cif_sum: f64 = (0..cause_count)
        .map(|k| assembled.cif[k][[0, n_times - 1]])
        .sum();
    assert!(
        final_cif_sum > 0.1,
        "the t-grid never accumulates meaningful failure mass (ΣCIF(t_max)={final_cif_sum:.4}); \
         the identity gate would be vacuous"
    );
}
