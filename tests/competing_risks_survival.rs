use gam::pirls::{PirlsStatus, WorkingModelPirlsOptions, runworking_model_pirls};
use gam::survival::{
    MonotonicityPenalty, PenaltyBlocks, SurvivalBaselineOffsets, SurvivalEngineInputs,
    SurvivalSpec, WorkingModelSurvival, expand_cause_specific_survival_flat_inputs,
};
use gam::types::Coefficients;
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

fn fit_constant_exposure_cause_specific(
    cause_count: usize,
    event_counts: &[usize],
    n: usize,
) -> Array1<f64> {
    gam::init_parallelism();
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
    let expanded = expand_cause_specific_survival_flat_inputs(
        age_entry.view(),
        age_exit.view(),
        events.view(),
        weights.view(),
        x_entry.view(),
        x_exit.view(),
        x_derivative.view(),
        Some(SurvivalBaselineOffsets {
            eta_entry: offset_entry.view(),
            eta_exit: offset_exit.view(),
            derivative_exit: offset_derivative.view(),
        }),
    )
    .expect("expand cause-specific survival inputs");
    let mut model = WorkingModelSurvival::from_engine_inputswith_offsets(
        SurvivalEngineInputs {
            age_entry: expanded.age_entry.view(),
            age_exit: expanded.age_exit.view(),
            event_target: expanded.event_target.view(),
            event_competing: expanded.event_competing.view(),
            sampleweight: expanded.sampleweight.view(),
            x_entry: expanded.x_entry.view(),
            x_exit: expanded.x_exit.view(),
            x_derivative: expanded.x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
        },
        Some(SurvivalBaselineOffsets {
            eta_entry: expanded.offset_eta_entry.view(),
            eta_exit: expanded.offset_eta_exit.view(),
            derivative_exit: expanded.offset_derivative_exit.view(),
        }),
        PenaltyBlocks::new(Vec::new()),
        MonotonicityPenalty { tolerance: 0.0 },
        SurvivalSpec::Net,
    )
    .expect("build joint cause-specific survival model");
    let opts = WorkingModelPirlsOptions {
        max_iterations: 80,
        convergence_tolerance: 1e-10,
        max_step_halving: 30,
        min_step_size: 1e-14,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: None,
        initial_lm_lambda: None,
    };
    let beta0 = Array1::from_iter(
        event_counts
            .iter()
            .map(|&count| ((count as f64 + 0.5) / n as f64).ln()),
    );
    let summary = runworking_model_pirls(&mut model, Coefficients::new(beta0), &opts, |_| {})
        .expect("joint cause-specific survival fit");
    assert_eq!(summary.status, PirlsStatus::Converged);
    summary.beta.as_ref().to_owned()
}

fn fit_constant_exposure_single_endpoint(event_count: usize, n: usize) -> Array1<f64> {
    gam::init_parallelism();
    let mut events = Array1::<u8>::zeros(n);
    for i in 0..event_count {
        events[i] = 1;
    }
    let age_entry = Array1::zeros(n);
    let age_exit = Array1::ones(n);
    let weights = Array1::ones(n);
    let x_entry = Array2::ones((n, 1));
    let x_exit = Array2::ones((n, 1));
    let x_derivative = Array2::zeros((n, 1));
    let offset_entry = Array1::zeros(n);
    let offset_exit = Array1::zeros(n);
    let offset_derivative = Array1::ones(n);
    let event_competing = Array1::<u8>::zeros(n);
    let mut model = WorkingModelSurvival::from_engine_inputswith_offsets(
        SurvivalEngineInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: events.view(),
            event_competing: event_competing.view(),
            sampleweight: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
        },
        Some(SurvivalBaselineOffsets {
            eta_entry: offset_entry.view(),
            eta_exit: offset_exit.view(),
            derivative_exit: offset_derivative.view(),
        }),
        PenaltyBlocks::new(Vec::new()),
        MonotonicityPenalty { tolerance: 0.0 },
        SurvivalSpec::Net,
    )
    .expect("build single-endpoint survival model");
    let opts = WorkingModelPirlsOptions {
        max_iterations: 80,
        convergence_tolerance: 1e-10,
        max_step_halving: 30,
        min_step_size: 1e-14,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: None,
        initial_lm_lambda: None,
    };
    let beta0 = Array1::from_vec(vec![((event_count as f64 + 0.5) / n as f64).ln()]);
    let summary = runworking_model_pirls(&mut model, Coefficients::new(beta0), &opts, |_| {})
        .expect("single endpoint survival fit");
    assert_eq!(summary.status, PirlsStatus::Converged);
    summary.beta.as_ref().to_owned()
}

#[test]
fn joint_two_cause_fit_recovers_constant_cause_specific_hazards() {
    let n = 1000;
    let events = [120, 70];
    let beta = fit_constant_exposure_cause_specific(2, &events, n);
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
    let beta = fit_constant_exposure_cause_specific(3, &events, n);
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
fn single_cause_expansion_matches_single_endpoint_baseline() {
    let n = 900;
    let event_count = 117;
    let joint_beta = fit_constant_exposure_cause_specific(1, &[event_count], n);
    let baseline_beta = fit_constant_exposure_single_endpoint(event_count, n);
    assert!(
        (joint_beta[0] - baseline_beta[0]).abs() <= 1e-12,
        "K=1 expanded beta differs from baseline: joint={}, baseline={}",
        joint_beta[0],
        baseline_beta[0]
    );
}
