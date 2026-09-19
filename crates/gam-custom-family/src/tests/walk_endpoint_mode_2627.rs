//! #2627: an evaluation at a θ the outer walk accepted is solved from that
//! iterate's certified inner mode, including after the reset that precedes each
//! terminal installation.
//!
//! After the search converges, the runner installs the winner's θ six times:
//! the Screening certificate's value and analytic lanes, two
//! `finalize_outer_result` calls, and the Mint certificate's two lanes. Four
//! resets separate them. Each reset used to leave only the caller's seed, and it
//! also dropped the final accepted step, which is still pending when the
//! optimizer stops. So all six re-solved the inner problem cold at a θ whose mode
//! the walk had already certified. On the event-history prior-centred fit that
//! was six identical 26-cycle solves, about 61 s (job 1212656). This replays that
//! sequence against `CustomOuterState` and counts the family evaluations.
use super::*;
use crate::penalty_labels::penalty_label_layout_with_joint;

/// The Gaussian fixture, counting every family evaluation. The inner solve's
/// share of the count is the work a cold re-solve repeats.
#[derive(Clone)]
struct CountingGaussianFamily {
    gaussian: OneBlockGaussianFamily,
    evaluations: Arc<AtomicUsize>,
}

impl CustomFamily for CountingGaussianFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        self.gaussian.evaluate(block_states)
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        self.gaussian
            .diagonalworking_weights_directional_derivative(block_states, block_idx, d_eta)
    }

    fn diagonalworking_weights_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_eta_u: &Array1<f64>,
        d_eta_v: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        self.gaussian.diagonalworking_weights_second_directional_derivative(
            block_states,
            block_idx,
            d_eta_u,
            d_eta_v,
        )
    }
}

#[test]
fn terminal_installations_reuse_the_walks_certified_mode_after_reset_2627() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let family = CountingGaussianFamily {
        gaussian: OneBlockGaussianFamily {
            y: array![0.3, -1.1, 0.8, 2.0, -0.4, 1.5],
        },
        evaluations: Arc::clone(&evaluations),
    };
    let specs = vec![ParameterBlockSpec {
        name: "walk_endpoint".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, -1.0],
            [1.0, -0.6],
            [1.0, -0.2],
            [1.0, 0.2],
            [1.0, 0.6],
            [1.0, 1.0],
        ])),
        offset: Array1::zeros(6),
        penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(2))],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(Array1::zeros(2)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let penalty_counts = validate_blockspecs(&specs).expect("valid walk-endpoint spec");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid walk-endpoint layout");
    // One outer evaluation from `seed`, with the number of family evaluations
    // it made.
    let evaluate = |seed: Option<&ConstrainedWarmStart>, theta: &Array1<f64>, mode: EvalMode| {
        evaluations.store(0, Ordering::Relaxed);
        let eval = outerobjectivegradienthessian_labeled(
            &family,
            &specs,
            &options,
            &layout,
            theta,
            seed,
            &gam_problem::RhoPrior::Flat,
            mode,
        )
        .expect("walk-endpoint outer evaluation");
        assert!(eval.inner_converged, "the Gaussian inner solve converges at {theta:?}");
        (eval, evaluations.load(Ordering::Relaxed))
    };

    let accepted_steps = Arc::new(AtomicUsize::new(0));
    let mut state = CustomOuterState::new_with_cold_signal(
        None,
        Arc::new(AtomicBool::new(false)),
        Arc::clone(&accepted_steps),
    );

    // The walk. Its starting iterate is accepted as soon as it is evaluated. A
    // trial at 3.0 is rejected; the trial at 1.5 is accepted, and the optimizer
    // stops there without evaluating again, so that step is still pending.
    let start_theta = array![0.5];
    let rejected_theta = array![3.0];
    let final_theta = array![1.5];
    for theta in [&start_theta, &rejected_theta, &final_theta] {
        state.adopt_accepted_steps();
        let eval = evaluate(state.warm_start_for(theta), theta, EvalMode::ValueAndGradient).0;
        state.record_first_order_mode(eval.warm_start.clone());
    }
    accepted_steps.fetch_add(1, Ordering::Relaxed);
    let certified_mode = state
        .pending_first_order_mode
        .clone()
        .expect("the accepted final step is still pending when the walk stops");
    assert_eq!(certified_mode.rho, final_theta);

    // Controls at the final θ. The positive control solves from the walk's own
    // certified mode, which is a same-ρ reuse. The negative control solves cold
    // from the caller's seed, which is what every terminal installation did.
    let (reuse_gradient, reuse_count) = evaluate(
        Some(&certified_mode),
        &final_theta,
        EvalMode::ValueAndGradient,
    );
    let (reuse_value, reuse_value_count) =
        evaluate(Some(&certified_mode), &final_theta, EvalMode::ValueOnly);
    let cold_count = evaluate(None, &final_theta, EvalMode::ValueAndGradient).1;
    assert!(
        cold_count > reuse_count,
        "the evaluation count must tell a cold solve ({cold_count}) from a reuse \
         ({reuse_count}), or this test cannot see the defect",
    );

    // The runner's terminal sequence at the final θ. Each installation must cost
    // exactly a reuse and reproduce the certified mode's evaluation bit for bit.
    let terminal_sequence = [
        (true, EvalMode::ValueOnly),
        (false, EvalMode::ValueAndGradient),
        (true, EvalMode::ValueAndGradient),
        (true, EvalMode::ValueAndGradient),
        (true, EvalMode::ValueOnly),
        (false, EvalMode::ValueAndGradient),
    ];
    for (installation, (reset_first, mode)) in terminal_sequence.into_iter().enumerate() {
        if reset_first {
            state.reset();
        }
        state.adopt_accepted_steps();
        let (eval, count) = evaluate(state.warm_start_for(&final_theta), &final_theta, mode);
        let (control, control_count) = if mode == EvalMode::ValueOnly {
            (&reuse_value, reuse_value_count)
        } else {
            (&reuse_gradient, reuse_count)
        };
        assert_eq!(
            count, control_count,
            "terminal installation {installation} ({mode:?}) made {count} family evaluations; \
             a reuse of the certified mode makes {control_count} and a cold solve {cold_count}",
        );
        assert_eq!(
            eval.objective.to_bits(),
            control.objective.to_bits(),
            "terminal installation {installation} priced {:.17e}, the certified mode {:.17e}",
            eval.objective,
            control.objective,
        );
        let gradient_bits = |g: &Array1<f64>| g.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(gradient_bits(&eval.gradient), gradient_bits(&control.gradient));
        let beta_bits = |e: &OuterObjectiveEvalResult| {
            e.inner.block_states[0].beta.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(beta_bits(&eval), beta_bits(control));
        if mode != EvalMode::ValueOnly {
            state.record_first_order_mode(eval.warm_start.clone());
        }
    }

    // Only the accepted final iterate was filed. The rejected trial and the
    // superseded starting iterate still solve from the caller's seed.
    state.reset();
    assert!(state.warm_start_for(&rejected_theta).is_none());
    assert!(state.warm_start_for(&start_theta).is_none());
    assert_eq!(
        state.warm_start_for(&final_theta).map(|seed| seed.rho.clone()),
        Some(final_theta),
    );
}

/// #979: a line search prices a trial θ by value and then asks for its gradient at the
/// same θ, and both lanes start from one seed. So the gradient lane's inner solve
/// re-derived the mode the value probe had just converged to. On the n=2000 BMS flex fit
/// that was 12 value/gradient pairs whose cycle counts matched lane for lane: 7.48 s of
/// solve wall (job 1244570, arm B). The probe's mode is now served to the gradient lane at
/// bitwise that θ, while the seed it was solved from is still the seed. This replays one
/// trial step against `CustomOuterState` and counts the family evaluations.
#[test]
fn a_value_probe_hands_its_mode_to_the_gradient_at_the_same_theta_979() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let family = CountingGaussianFamily {
        gaussian: OneBlockGaussianFamily {
            y: array![0.3, -1.1, 0.8, 2.0, -0.4, 1.5],
        },
        evaluations: Arc::clone(&evaluations),
    };
    let specs = vec![ParameterBlockSpec {
        name: "value_probe".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, -1.0],
            [1.0, -0.6],
            [1.0, -0.2],
            [1.0, 0.2],
            [1.0, 0.6],
            [1.0, 1.0],
        ])),
        offset: Array1::zeros(6),
        penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(2))],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(Array1::zeros(2)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let penalty_counts = validate_blockspecs(&specs).expect("valid value-probe spec");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid value-probe layout");
    let evaluate = |seed: Option<&ConstrainedWarmStart>, theta: &Array1<f64>, mode: EvalMode| {
        evaluations.store(0, Ordering::Relaxed);
        let eval = outerobjectivegradienthessian_labeled(
            &family,
            &specs,
            &options,
            &layout,
            theta,
            seed,
            &gam_problem::RhoPrior::Flat,
            mode,
        )
        .expect("value-probe outer evaluation");
        assert!(eval.inner_converged, "the Gaussian inner solve converges at {theta:?}");
        (eval, evaluations.load(Ordering::Relaxed))
    };
    let bits = |values: &Array1<f64>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();

    let accepted_steps = Arc::new(AtomicUsize::new(0));
    let mut state = CustomOuterState::new_with_cold_signal(
        None,
        Arc::new(AtomicBool::new(false)),
        Arc::clone(&accepted_steps),
    );
    // The walk's starting iterate, accepted as soon as it is evaluated.
    let start_theta = array![0.5];
    state.adopt_accepted_steps();
    let start = evaluate(state.warm_start_for(&start_theta), &start_theta, EvalMode::ValueAndGradient).0;
    state.record_first_order_mode(start.warm_start.clone());

    // A trial θ, priced first by the line search's value probe.
    let trial_theta = array![1.5];
    state.adopt_accepted_steps();
    let seed_identity = crate::warm_start::SeedIdentity::of(state.seed_for(&trial_theta));
    let probe = evaluate(state.warm_start_for(&trial_theta), &trial_theta, EvalMode::ValueOnly).0;
    state.record_value_probe(&trial_theta, seed_identity, probe.warm_start.clone());

    // Controls at the trial θ. The negative control is what the gradient lane did: solve
    // from the seed. The positive control solves from the probe's mode, a same-ρ reuse.
    let seed = state.seed_for(&trial_theta).cloned();
    let (fresh, fresh_count) = evaluate(seed.as_ref(), &trial_theta, EvalMode::ValueAndGradient);
    let (_, reuse_count) =
        evaluate(Some(&probe.warm_start), &trial_theta, EvalMode::ValueAndGradient);
    assert!(
        fresh_count > reuse_count,
        "the evaluation count must tell a solve from the seed ({fresh_count}) from a reuse \
         ({reuse_count}), or this test cannot see the repeated solve",
    );

    // The gradient lane: it must cost a reuse and price what a solve from the seed prices,
    // bit for bit.
    state.adopt_accepted_steps();
    let (lane, lane_count) =
        evaluate(state.warm_start_for(&trial_theta), &trial_theta, EvalMode::ValueAndGradient);
    assert_eq!(
        lane_count, reuse_count,
        "the gradient lane made {lane_count} family evaluations; a reuse of the probe's mode \
         makes {reuse_count} and a solve from the seed {fresh_count}",
    );
    assert_eq!(lane.objective.to_bits(), fresh.objective.to_bits());
    assert_eq!(bits(&lane.gradient), bits(&fresh.gradient));
    assert_eq!(
        bits(&lane.inner.block_states[0].beta),
        bits(&fresh.inner.block_states[0].beta),
    );

    // The probe seeds no other θ: every other θ still starts from the incumbent.
    let other_theta = array![2.0];
    assert_eq!(
        state.warm_start_for(&other_theta).map(|seed| seed.rho.clone()),
        Some(start_theta.clone()),
    );

    // A seed that moved never inherits the probe. Once the trial step is accepted the seed
    // is the gradient lane's own mode, and that is what the trial θ is served.
    state.record_first_order_mode(lane.warm_start.clone());
    accepted_steps.fetch_add(1, Ordering::Relaxed);
    state.adopt_accepted_steps();
    let served = state.warm_start_for(&trial_theta).expect("the accepted mode seeds the trial θ");
    let moved_seed = state.seed_for(&trial_theta).expect("the accepted mode is the seed");
    assert!(
        std::ptr::eq(served, moved_seed),
        "after the seed moved, the trial θ must be served the new seed, not the older probe",
    );
}
