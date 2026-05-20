use gam::custom_family::{BlockwiseFitOptions, ParameterBlockSpec, fit_custom_family};
use gam::matrix::DesignMatrix;
use gam::survival::{
    CauseSpecificRoystonParmarBlock, CauseSpecificRoystonParmarFamily,
    survival_event_code_from_value,
};
use ndarray::{Array1, Array2};

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
