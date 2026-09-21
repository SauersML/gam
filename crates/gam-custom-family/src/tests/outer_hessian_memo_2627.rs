//! #2627: a stored outer Hessian answers only for the exact state it was
//! assembled at.
//!
//! After a latched saddle escape, the Mint certificate assembled the dense outer
//! Hessian again at the ARC walk's last iterate, at bit-identical ρ and from the
//! mode that walk had certified there: 14.8 s per latched fit (job 1230177).
//! `evaluate_with_outer_hessian_memo` serves the stored Hessian when the
//! evaluation installs the same state. One ρ can hold two certified inner modes
//! with different Hessians, so a key on ρ alone would serve the wrong one. The
//! tilted double well has two such modes at ρ = −6, and it is what these tests
//! use.
use super::*;
use crate::penalty_labels::penalty_label_layout_with_joint;

/// `w(β) = (β² − 1)² + c·β` as a negative log-likelihood, through the block
/// exact-Newton hooks that the outer Hessian reads. It counts its second
/// directional derivative, which only an outer-Hessian assembly calls.
#[derive(Clone)]
struct CountingDoubleWellFamily {
    tilt: f64,
    second_directional_calls: Arc<AtomicUsize>,
}

impl CustomFamily for CountingDoubleWellFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states[0].beta[0];
        let well = beta * beta - 1.0;
        Ok(FamilyEvaluation {
            log_likelihood: -(well * well + self.tilt * beta),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-(4.0 * beta * beta * beta - 4.0 * beta + self.tilt)],
                hessian: SymmetricMatrix::Dense(array![[12.0 * beta * beta - 4.0]]),
            }],
        })
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0);
        let beta = block_states[0].beta[0];
        Ok(Some(array![[24.0 * beta * direction[0]]]))
    }

    fn exact_newton_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0);
        assert_states_finite(block_states, "double-well second directional derivative");
        self.second_directional_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(array![[24.0 * u[0] * v[0]]]))
    }
}

struct DoubleWellFixture {
    family: CountingDoubleWellFamily,
    calls: Arc<AtomicUsize>,
    specs: Vec<ParameterBlockSpec>,
    options: BlockwiseFitOptions,
    layout: PenaltyLabelLayout,
    theta: Array1<f64>,
}

impl DoubleWellFixture {
    fn new() -> Self {
        let calls = Arc::new(AtomicUsize::new(0));
        let specs = vec![ParameterBlockSpec {
            name: "well".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
                [1.0]
            ])),
            offset: array![0.0],
            penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.0],
            initial_beta: Some(array![0.0]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }];
        let penalty_counts = validate_blockspecs(&specs).expect("valid double-well spec");
        let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
            .expect("valid double-well layout");
        Self {
            family: CountingDoubleWellFamily {
                tilt: 0.3,
                second_directional_calls: Arc::clone(&calls),
            },
            calls,
            specs,
            options: BlockwiseFitOptions {
                inner_tol: 1e-11,
                inner_max_cycles: 200,
                use_remlobjective: true,
                use_outer_hessian: true,
                compute_covariance: false,
                ..BlockwiseFitOptions::default()
            },
            layout,
            // λ = e⁻⁶ < 4 leaves the barrier between the wells standing, so each
            // side holds a certified mode.
            theta: array![-6.0],
        }
    }

    fn seed(&self, beta: f64) -> ConstrainedWarmStart {
        constrained_warm_start_from_cached_beta(1, &self.specs, &array![beta])
            .expect("a finite one-coefficient seed")
    }

    fn evaluate(
        &self,
        seed: Option<&ConstrainedWarmStart>,
        mode: EvalMode,
    ) -> Result<OuterObjectiveEvalResult, CustomFamilyError> {
        outerobjectivegradienthessian_labeled(
            &self.family,
            &self.specs,
            &self.options,
            &self.layout,
            &self.theta,
            seed,
            &gam_problem::RhoPrior::Flat,
            mode,
        )
    }

    /// One evaluation through the memo, with the outer-Hessian assemblies it
    /// ran.
    fn through_memo(
        &self,
        memo: &mut Option<OuterHessianMemo>,
        seed: Option<&ConstrainedWarmStart>,
        epoch: Option<usize>,
    ) -> (OuterObjectiveEvalResult, usize) {
        self.calls.store(0, Ordering::Relaxed);
        let evaluation = evaluate_with_outer_hessian_memo(memo, &self.theta, || epoch, |mode| {
            self.evaluate(seed, mode)
        })
        .expect("double-well evaluation through the memo");
        assert!(evaluation.inner_converged);
        (evaluation, self.calls.load(Ordering::Relaxed))
    }

    /// One fresh `ValueGradientHessian` evaluation, with its assemblies.
    fn fresh(&self, seed: Option<&ConstrainedWarmStart>) -> (OuterObjectiveEvalResult, usize) {
        self.calls.store(0, Ordering::Relaxed);
        let evaluation = self
            .evaluate(seed, EvalMode::ValueGradientHessian)
            .expect("fresh double-well evaluation");
        assert!(evaluation.inner_converged);
        (evaluation, self.calls.load(Ordering::Relaxed))
    }
}

fn dense_hessian_bits(evaluation: &OuterObjectiveEvalResult) -> Vec<u64> {
    match &evaluation.outer_hessian {
        gam_problem::HessianValue::Dense(hessian) => {
            hessian.iter().map(|value| value.to_bits()).collect()
        }
        other => panic!("the double well assembles a dense outer Hessian, got {other:?}"),
    }
}

fn bits(values: &Array1<f64>) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_same_evaluation(served: &OuterObjectiveEvalResult, fresh: &OuterObjectiveEvalResult) {
    assert_eq!(served.objective.to_bits(), fresh.objective.to_bits());
    assert_eq!(bits(&served.gradient), bits(&fresh.gradient));
    assert_eq!(
        bits(&served.inner.block_states[0].beta),
        bits(&fresh.inner.block_states[0].beta)
    );
    assert_eq!(
        dense_hessian_bits(served),
        dense_hessian_bits(fresh),
        "the served outer Hessian differs from the one this state assembles",
    );
}

#[test]
fn a_stored_outer_hessian_is_served_only_for_the_state_it_was_assembled_at_2627() {
    let fixture = DoubleWellFixture::new();
    let mut memo = None;

    // The walk's evaluation at θ stores its Hessian. Production then seeds every
    // later evaluation at θ from the mode that evaluation certified (the walk
    // endpoint), so the second evaluation installs the same state.
    let (walk, walk_calls) = fixture.through_memo(&mut memo, Some(&fixture.seed(-2.0)), None);
    assert!(walk_calls > 0, "an evaluation with nothing stored assembles the Hessian");
    let certified = walk.warm_start.clone();

    let (fresh, fresh_calls) = fixture.fresh(Some(&certified));
    assert!(
        fresh_calls > 0,
        "the assembly count must see a recompute ({fresh_calls}), or it cannot see a reuse",
    );
    let (served, served_calls) = fixture.through_memo(&mut memo, Some(&certified), None);
    assert_eq!(
        served_calls, 0,
        "the same installed state must be served its stored Hessian, not re-assembled",
    );
    assert_same_evaluation(&served, &fresh);

    // Another measure at the same θ and state is another function.
    let (other_measure, other_measure_calls) =
        fixture.through_memo(&mut memo, Some(&certified), Some(1));
    assert!(
        other_measure_calls > 0,
        "a Hessian stored under one measure must not answer for another",
    );
    assert_same_evaluation(&other_measure, &fresh);
}

#[test]
fn an_outer_hessian_stored_for_one_mode_never_answers_for_the_other_at_the_same_rho_2627() {
    let fixture = DoubleWellFixture::new();
    let deep_seed = fixture.seed(-2.0);
    let shallow_seed = fixture.seed(2.0);

    // CONTROL: at this ρ the two seeds certify two modes with different
    // Hessians, so a key that ignores the mode cannot tell them apart.
    let (deep, deep_calls) = fixture.fresh(Some(&deep_seed));
    let (shallow, shallow_calls) = fixture.fresh(Some(&shallow_seed));
    assert!(deep_calls > 0 && shallow_calls > 0);
    let deep_beta = deep.inner.block_states[0].beta[0];
    let shallow_beta = shallow.inner.block_states[0].beta[0];
    assert!(
        deep_beta < -0.5 && shallow_beta > 0.5,
        "the fixture must certify both wells at one ρ; got {deep_beta} and {shallow_beta}",
    );
    assert_ne!(
        dense_hessian_bits(&deep),
        dense_hessian_bits(&shallow),
        "the two modes must carry different outer Hessians",
    );

    let mut memo = None;
    fixture.through_memo(&mut memo, Some(&deep_seed), None);
    let (from_shallow, from_shallow_calls) =
        fixture.through_memo(&mut memo, Some(&shallow_seed), None);
    assert!(
        from_shallow_calls > 0,
        "the shallow mode has no stored Hessian and must assemble its own",
    );
    assert_ne!(
        dense_hessian_bits(&from_shallow),
        dense_hessian_bits(&deep),
        "served another mode's outer Hessian at the same ρ",
    );
    assert_same_evaluation(&from_shallow, &shallow);

    // The shallow mode's Hessian is now the stored one, and is served to it.
    let (again, again_calls) = fixture.through_memo(&mut memo, Some(&shallow_seed), None);
    assert_eq!(again_calls, 0);
    assert_same_evaluation(&again, &shallow);
}
