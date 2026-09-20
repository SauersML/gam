//! #2366: the inner mode must be a FUNCTION of ρ, not a functional of the seed.
//!
//! The profiled outer criterion is `V(ρ) = ℓ_p(θ̂(ρ), ρ)`. When `ℓ_p(·, ρ)` is
//! nonconvex its `argmin` is a set, so `V` is a function of ρ only once a
//! selection rule fixes which element is meant. `anchored_continuation_seed`
//! supplies that rule: `θ̂(ρ)` is the endpoint of the continuation from the
//! effective-df-floor anchor, where the term sits on its penalty nullspace and
//! the mode is unique.
//!
//! These tests are built on a fixture whose two modes are known in closed form,
//! and — critically — they include a CONTROL that proves the fixture actually
//! discriminates. A seed-invariance assertion on a unimodal problem would pass
//! no matter what the code did.
use super::*;
use crate::fit::{
    AnchoredContinuationRefusal, ContinuationRefinement, anchored_continuation_seed,
    continuation_refinement_decision,
};
use crate::penalty_labels::penalty_label_layout_with_joint;

/// A one-coefficient family with two known, unequal modes.
///
/// The log-likelihood is `−w(β)` for the tilted double well
///
/// ```text
///     w(β) = (β² − 1)² + c·β,      c = TILT > 0
/// ```
///
/// which has a deep well near `β = −1` (value `≈ −c`) and a shallow well near
/// `β = +1` (value `≈ +c`), separated by a barrier at `β = 0`. The observed
/// information `−d²ℓ/dβ² = 12β² − 4` is genuinely indefinite between the wells,
/// so this is a real nonconvex inner problem rather than a convex one dressed up
/// as one, and `exact_newton_joint_hessian_beta_dependent` is honestly `true`.
///
/// Under the ridge penalty `½λβ²` the barrier is convexified for `λ > 4`, and
/// the unique mode there is `β ≈ −c/(λ−4) < 0`: the maximally-smoothed anchor
/// sits on the DEEP well's side. That is the whole point of anchoring — the
/// continuation from maximal smoothing tracks the deep well, while a caller's
/// coefficients on the other side of the barrier fall into the shallow one.
#[derive(Clone)]
struct TiltedDoubleWellFamily {
    tilt: f64,
    outer_curvature_calls: Option<Arc<AtomicUsize>>,
}

impl TiltedDoubleWellFamily {
    fn new(tilt: f64) -> Self {
        Self {
            tilt,
            outer_curvature_calls: None,
        }
    }

    fn beta(block_states: &[ParameterBlockState]) -> Result<f64, String> {
        block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .first()
            .copied()
            .ok_or_else(|| "missing coefficient".to_string())
    }
}

impl CustomFamily for TiltedDoubleWellFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = Self::beta(block_states)?;
        let well = beta * beta - 1.0;
        Ok(FamilyEvaluation {
            log_likelihood: -(well * well + self.tilt * beta),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                // dℓ/dβ, and the observed information −d²ℓ/dβ².
                gradient: array![-(4.0 * beta * beta * beta - 4.0 * beta + self.tilt)],
                hessian: SymmetricMatrix::Dense(array![[12.0 * beta * beta - 4.0]]),
            }],
        })
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states)?;
        Ok(Some(array![[12.0 * beta * beta - 4.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states)?;
        let step = direction.first().copied().unwrap_or(0.0);
        Ok(Some(array![[24.0 * beta * step]]))
    }

    fn exact_newton_outer_curvature(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        let Some(calls) = self.outer_curvature_calls.as_ref() else {
            return Ok(None);
        };
        calls.fetch_add(1, Ordering::Relaxed);
        let beta = Self::beta(block_states)?;
        Ok(Some(ExactNewtonOuterCurvature {
            hessian: array![[12.0 * beta * beta - 4.0]],
            rho_curvature_scale: 1.0,
            hessian_logdet_correction: 0.0,
        }))
    }
}

const TILT: f64 = 0.3;

/// A continuation path with a SCRIPTED endpoint sequence.
///
/// The ladder's stopping rule is a property of the sequence of endpoints, not of
/// any one comparison, so the fixtures below drive the whole ladder rather than
/// a single decision. Scripting the criterion values is what lets the #2612
/// shape — an agreement that a further refinement leaves — be asserted without
/// a nonconvex fixture that takes minutes to produce it.
struct ScriptedContinuationPath {
    /// Criterion value returned for the sweep at `2^k` steps, index `k`. The
    /// last entry repeats for any deeper refinement.
    criterion_by_refinement: Vec<f64>,
    sweeps: std::cell::RefCell<Vec<usize>>,
}

impl ScriptedContinuationPath {
    fn new(criterion_by_refinement: Vec<f64>) -> Self {
        Self {
            criterion_by_refinement,
            sweeps: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn criterion_for(&self, steps: usize) -> f64 {
        let index = steps.trailing_zeros() as usize;
        let last = self.criterion_by_refinement.len() - 1;
        self.criterion_by_refinement[index.min(last)]
    }
}

impl crate::fit::RefinedContinuationPath for ScriptedContinuationPath {
    fn sweep(
        &self,
        steps: usize,
    ) -> Result<crate::fit::SweptEndpoint, AnchoredContinuationRefusal> {
        self.sweeps.borrow_mut().push(steps);
        let criterion = self.criterion_for(steps);
        Ok(crate::fit::SweptEndpoint {
            warm_start: crate::assembly::ConstrainedWarmStart {
                rho: array![0.0],
                // The endpoint's state stands in for the mode; it tracks the
                // criterion so the state discrepancy and the criterion
                // agreement move together, as they do on a real path.
                block_beta: vec![array![criterion]],
                active_sets: vec![None],
                cached_inner: None,
            },
            criterion_value: criterion,
        })
    }

    fn resolves_steps(&self, refined_steps: usize) -> bool {
        1.0 / refined_steps as f64 > f64::EPSILON
    }

    fn endpoint_discrepancy(
        &self,
        steps: usize,
        coarser: &crate::assembly::ConstrainedWarmStart,
        finer: &crate::assembly::ConstrainedWarmStart,
    ) -> Result<f64, AnchoredContinuationRefusal> {
        // The scripted endpoints carry the criterion in their own state, so the
        // discrepancy is a function of the two endpoints alone. `steps` is
        // still the rung the ladder is asking about, and the ladder refines
        // only by doubling — asserting that here is what keeps this stand-in
        // honest about the contract it stands in for, rather than discarding
        // the one input it does not otherwise read.
        assert!(
            steps.is_power_of_two(),
            "the refinement ladder asked for an endpoint discrepancy at {steps} steps, which is \
             not a doubling of the anchored sweep"
        );
        Ok((coarser.block_beta[0][0] - finer.block_beta[0][0]).abs())
    }

    fn label(&self) -> &'static str {
        "scripted"
    }
}

fn scripted_options(outer_max_iter: usize) -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        outer_max_iter,
        // The ladder is judged in the criterion's own units, so this is the
        // resolution the fixtures below are written against.
        outer_rel_cost_tol: Some(1e-6),
        ..double_well_options()
    }
}

/// #2661's guarantee, asserted as the property it actually is: **the refinement
/// loop terminates in bounded work**, whatever the sequence does.
///
/// The original form of this test asserted a per-refinement contraction ratio,
/// which #2612 measured cannot read this ladder at all — the endpoint sequence
/// is mode-valued, so its discrepancies alternate between `O(1)` and the
/// corrector's floor and never exhibit a rate. The requirement #2661 stated
/// ("each round doubles the number of full corrector solves, so accepting
/// arbitrarily slow progress makes the loop operationally unbounded") is
/// preserved exactly, and is now bounded as the resource it is.
#[test]
fn arbitrarily_slow_progress_still_terminates_in_bounded_work_2661() {
    // A criterion that creeps toward its limit by a factor 0.999 per refinement:
    // strictly improving, never agreeing to `1e-6`.
    let script: Vec<f64> = (0..40).map(|k| 1.0 + 0.999_f64.powi(k)).collect();
    let path = ScriptedContinuationPath::new(script);
    let options = scripted_options(100);
    let budget = crate::fit::continuation_refinement_budget(options.outer_max_iter);
    let refusal = match crate::fit::certify_refined_continuation(&path, &options, false) {
        Ok(certified) => panic!(
            "a creeping criterion must terminate with a typed refusal, not a certificate at \
             {} steps",
            certified.certificate.steps
        ),
        Err(refusal) => refusal,
    };
    match refusal {
        AnchoredContinuationRefusal::RefinementBudgetExhausted {
            refinements,
            max_refinements,
            ..
        } => {
            assert_eq!(max_refinements, budget);
            assert_eq!(refinements, budget);
        }
        other => panic!("a creeping criterion produced the wrong typed refusal: {other:?}"),
    }
    // The bound is on WORK, so the work is what is asserted: the ladder ran the
    // budgeted number of refinements and not one sweep more.
    assert_eq!(
        *path.sweeps.borrow(),
        (0..=budget).map(|k| 1usize << k).collect::<Vec<_>>(),
        "the ladder must run exactly the sweeps its budget allows"
    );
}

/// The bound is derived from the outer search's own budget, so it moves with it.
#[test]
fn the_refinement_budget_is_the_outer_searchs_corrector_budget_2661() {
    for (outer_max_iter, expected) in [(64usize, 5usize), (100, 5), (128, 6), (1000, 8)] {
        assert_eq!(
            crate::fit::continuation_refinement_budget(outer_max_iter),
            expected,
            "a ladder through D refinements runs 2^(D+1)-1 correctors, which must fit in \
             outer_max_iter={outer_max_iter}"
        );
        assert!(
            (1usize << (expected + 1)) <= outer_max_iter,
            "the derivation must hold at outer_max_iter={outer_max_iter}"
        );
    }
    // Below the point where a verdict is reachable at all, the budget is floored
    // at the fewest refinements that can produce one rather than disabling the
    // ladder outright.
    assert_eq!(crate::fit::continuation_refinement_budget(1), 3);
    assert_eq!(crate::fit::continuation_refinement_budget(8), 3);
}

/// The #2612 shape, from the direction the penguins fixture cannot be run in
/// under a second: **an agreement that a further refinement LEAVES must not be
/// certified, and must not be refused either.**
///
/// Scripted from the measured stride-4 trail — two coarse sweeps landing on one
/// mode, the next refinement landing on another, then a plateau — so the assert
/// is on the exact sequence that used to produce
/// `endpoint discrepancy violates the dyadic contraction premise:
///  1.713372e0 / 3.328619e-5`.
#[test]
fn a_plateau_a_later_refinement_leaves_is_neither_certified_nor_refused_2612() {
    // steps 1, 2, 4 : one mode        (an agreement at 2->4)
    // steps 8, 16, 32: another mode   (disagreement at 8, then agreements)
    let script = vec![10.607, 11.387, 11.387, 10.594, 10.594, 10.594];
    let path = ScriptedContinuationPath::new(script);
    let options = scripted_options(100);
    let certified = crate::fit::certify_refined_continuation(&path, &options, false).expect(
        "a ladder that changes branch and then settles must certify, not refuse: the coarse \
         pair's agreement was never a discretization error, so it cannot be a contraction \
         baseline",
    );
    // Certification must NOT have happened at the coarse plateau (4 steps): the
    // whole point is that a further refinement left it.
    assert_eq!(
        certified.certificate.steps, 32,
        "the ladder certified at the plateau a later refinement left"
    );
    assert!(
        certified.certificate.consecutive_agreements >= 2,
        "certification must rest on more than one refinement agreeing"
    );
    assert!(
        certified.certificate.criterion_agreement <= certified.certificate.criterion_resolution,
        "the certificate's own claim must hold: {:.3e} <= {:.3e}",
        certified.certificate.criterion_agreement,
        certified.certificate.criterion_resolution,
    );
    // And the state discrepancy at that point is NOT what certified it — on this
    // script it is exactly zero, but on the real fixture it sits at `5.4e-5`
    // against an `inner_tol` of `1e-5`, which is why it cannot be the verdict.
    assert_eq!(
        *path.sweeps.borrow(),
        vec![1, 2, 4, 8, 16, 32],
        "the ladder must have refined past the coarse plateau"
    );
}

/// One agreement is not a certificate. Pinned separately so a future change that
/// drops [`REQUIRED_CONSECUTIVE_AGREEMENTS`] to one has to argue with the
/// measured counterexample rather than with a comment.
#[test]
fn a_single_agreement_does_not_certify_2612() {
    let path = ScriptedContinuationPath::new(vec![10.607, 11.387, 11.387, 10.594, 10.594, 10.594]);
    let options = scripted_options(100);
    let certified = crate::fit::certify_refined_continuation(&path, &options, false)
        .expect("this script settles");
    assert!(
        certified.certificate.steps > 4,
        "a single agreement (2 -> 4) certified, which the measured stride-4 trail shows is \
         wrong: the 8-step sweep leaves that mode"
    );
}

/// The decision function is still the place the certificate is minted, so its
/// contract is pinned directly too.
#[test]
fn the_decision_certifies_only_on_enough_consecutive_agreements_2612() {
    let refine = continuation_refinement_decision(
        crate::fit::ContinuationRefinementReading {
            steps: 4,
            discrepancy: 1e-9,
            previous_discrepancy: Some(1.0),
            criterion_agreement: 1e-12,
            consecutive_agreements: 1,
        },
        1e-12,
        1e-6,
    )
    .expect("a valid reading");
    assert_eq!(
        refine,
        ContinuationRefinement::Refine,
        "one agreement, however tight, is not a limit"
    );
    let certified = continuation_refinement_decision(
        crate::fit::ContinuationRefinementReading {
            steps: 8,
            discrepancy: 1e-9,
            previous_discrepancy: Some(1e-9),
            criterion_agreement: 1e-12,
            consecutive_agreements: 2,
        },
        1e-12,
        1e-6,
    )
    .expect("a valid reading");
    match certified {
        ContinuationRefinement::Certified(certificate) => {
            assert_eq!(certificate.steps, 8);
            assert_eq!(certificate.consecutive_agreements, 2);
            // The ratio is reported, and it is 1.0 here — a sequence that has
            // stalled at the corrector's floor, which the old rule would have
            // refused outright.
            assert_eq!(certificate.observed_contraction_factor, Some(1.0));
        }
        other => panic!("two consecutive agreements did not certify: {other:?}"),
    }
}

fn double_well_spec(initial_beta: f64) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "well".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![initial_beta]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn double_well_options() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: 200,
        inner_tol: 1e-10,
        outer_max_iter: 50,
        outer_tol: 1e-8,
        outer_rel_cost_tol: None,
        rho_lower_bound: Some(-10.0),
        ridge_floor: 1e-8,
        use_remlobjective: true,
        compute_covariance: false,
        use_outer_hessian: false,
        early_exit_threshold: None,
        outer_score_subsample: None,
        auto_outer_subsample: false,
        cache_session: None,
        warm_start: None,
        persistent_warm_start_store: None,
        cache_mirror_sessions: Vec::new(),
        joint_penalties: None,
    }
}

/// The continuation corrector and the ordinary inner solve must return the
/// identical coefficient mode, while the corrector constructs zero of the two
/// determinant artifacts. Feeding that owned mode to the endpoint evaluator
/// must then recover the bit-identical complete criterion. This pins both sides
/// of the typed product boundary: the omitted work is genuinely absent, and no
/// scalar or certificate fact is lost at the endpoint that consumes it.
#[test]
fn continuation_corrector_builds_only_the_coefficient_product_2714() {
    let outer_curvature_calls = Arc::new(AtomicUsize::new(0));
    let family = TiltedDoubleWellFamily {
        tilt: TILT,
        outer_curvature_calls: Some(Arc::clone(&outer_curvature_calls)),
    };
    let specs = vec![double_well_spec(-2.0)];
    let options = double_well_options();
    let rho = array![-2.0];
    let penalty_counts = vec![1];
    let per_block = split_log_lambdas(&rho, &penalty_counts).expect("one penalty block");

    let laplace_ready = inner_blockwise_fit(&family, &specs, &per_block, &options, None)
        .expect("ordinary coefficient solve");
    assert_eq!(
        outer_curvature_calls.swap(0, Ordering::Relaxed),
        1,
        "the ordinary inner product must construct its determinant curvature once"
    );
    let coefficient_mode =
        inner_blockwise_coefficient_mode(&family, &specs, &per_block, &options, None)
            .expect("continuation coefficient corrector");
    assert_eq!(
        outer_curvature_calls.load(Ordering::Relaxed),
        0,
        "a continuation corrector must not enter determinant curvature assembly"
    );
    assert!(laplace_ready.converged && coefficient_mode.converged);
    assert!(laplace_ready.block_logdet_h.is_some());
    assert!(laplace_ready.block_logdet_s.is_some());
    assert_eq!(coefficient_mode.block_logdet_h, None);
    assert_eq!(coefficient_mode.block_logdet_s, None);
    assert_eq!(
        laplace_ready.block_states[0].beta[0].to_bits(),
        coefficient_mode.block_states[0].beta[0].to_bits(),
        "changing the requested product must not change the corrected mode"
    );
    assert_eq!(
        laplace_ready.log_likelihood.to_bits(),
        coefficient_mode.log_likelihood.to_bits()
    );
    assert_eq!(
        laplace_ready.penalty_value.to_bits(),
        coefficient_mode.penalty_value.to_bits()
    );
    assert_eq!(
        laplace_ready
            .kkt_residual
            .as_ref()
            .map(ProjectedKktResidual::inf_norm),
        coefficient_mode
            .kkt_residual
            .as_ref()
            .map(ProjectedKktResidual::inf_norm),
        "the coefficient product must retain the same KKT certificate"
    );

    let expected = evaluate_custom_family_hyper_from_coefficient_mode(
        &family,
        &specs,
        &options,
        &penalty_counts,
        &rho,
        gam_problem::RhoPrior::Flat,
        laplace_ready,
        EvalMode::ValueOnly,
    )
    .expect("endpoint criterion from ordinary mode");
    assert_eq!(
        outer_curvature_calls.swap(0, Ordering::Relaxed),
        1,
        "one endpoint mode must produce one complete criterion"
    );
    let actual = evaluate_custom_family_hyper_from_coefficient_mode(
        &family,
        &specs,
        &options,
        &penalty_counts,
        &rho,
        gam_problem::RhoPrior::Flat,
        coefficient_mode,
        EvalMode::ValueOnly,
    )
    .expect("endpoint criterion from continuation-owned mode");
    assert_eq!(
        outer_curvature_calls.load(Ordering::Relaxed),
        1,
        "the owned continuation mode must be scored once without a second coefficient solve"
    );
    assert_eq!(
        expected.objective.to_bits(),
        actual.objective.to_bits(),
        "endpoint evaluation must restore the complete criterion without re-solving beta"
    );
    assert_eq!(
        expected.warm_start.block_beta[0][0].to_bits(),
        actual.warm_start.block_beta[0][0].to_bits()
    );
}

/// Solve the inner problem at `rho` exactly the way the outer search does, from
/// the seed carried in `specs` — i.e. the pre-#2366 "whatever the caller handed
/// us" mode.
fn cold_direct_mode(
    family: &TiltedDoubleWellFamily,
    specs: &[ParameterBlockSpec],
    rho: f64,
) -> f64 {
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let eval = outerobjectivegradienthessian_labeled(
        family,
        specs,
        &options,
        &layout,
        &array![rho],
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueOnly,
    )
    .expect("cold-direct inner solve");
    assert!(
        eval.inner_converged,
        "cold-direct inner solve must converge"
    );
    eval.warm_start.block_beta[0][0]
}

fn continuation_mode(
    family: &TiltedDoubleWellFamily,
    specs: &[ParameterBlockSpec],
    rho: f64,
) -> f64 {
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let (_, anchor) = resolvability_rho_domain(specs, &layout, 1, options.rho_lower_bound)
        .expect("the resolvability domain of a single penalized term");
    let certified = anchored_continuation_seed(
        family,
        specs,
        &options,
        &layout,
        &gam_problem::RhoPrior::Flat,
        &anchor,
        &array![rho],
    )
    .expect("the continuation from the maximally-smoothed anchor must reach the target rho");
    assert!(
        certified.certificate.endpoint_discrepancy <= certified.certificate.inner_tolerance,
        "a returned continuation seed must carry its endpoint-invariance certificate"
    );
    certified.warm_start.block_beta[0][0]
}

/// CONTROL. Without a selection rule the mode is a functional of the seed: two
/// callers who ask the same question get different answers.
///
/// If this test ever starts failing because the two seeds agree, the fixture has
/// stopped discriminating and the two tests below prove nothing — so this one is
/// load-bearing, not decorative.
#[test]
fn cold_direct_mode_depends_on_the_seed_2366() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let target_rho = -6.0;
    let from_positive = cold_direct_mode(&family, &[double_well_spec(2.0)], target_rho);
    let from_negative = cold_direct_mode(&family, &[double_well_spec(-2.0)], target_rho);
    assert!(
        from_positive > 0.5,
        "a seed inside the shallow well should stay in it; got {from_positive}"
    );
    assert!(
        from_negative < -0.5,
        "a seed inside the deep well should stay in it; got {from_negative}"
    );
}

/// The continuation endpoint is the SAME mode no matter what the caller seeds,
/// which is exactly the statement that `θ̂` is a function of ρ.
#[test]
fn anchored_continuation_mode_is_independent_of_the_seed_2366() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let target_rho = -6.0;
    let from_positive = continuation_mode(&family, &[double_well_spec(2.0)], target_rho);
    let from_negative = continuation_mode(&family, &[double_well_spec(-2.0)], target_rho);
    let from_origin = continuation_mode(&family, &[double_well_spec(0.0)], target_rho);
    // Bitwise, not "close": a nonconvex profiled objective has no tolerance in
    // which two different branches are interchangeable provenance.
    assert_eq!(
        from_positive.to_bits(),
        from_negative.to_bits(),
        "continuation endpoints differ across seeds: {from_positive} vs {from_negative}"
    );
    assert_eq!(
        from_positive.to_bits(),
        from_origin.to_bits(),
        "continuation endpoints differ across seeds: {from_positive} vs {from_origin}"
    );
}

/// #2901, the ruling's pin (3): at the certified mode the double well's data
/// curvature `12β² − 4` is negative and only the penalty makes `H = 12β² − 4 + λ`
/// positive definite. `H ⪰ λS` fails, so the block has no certified rank bound, and the
/// exact trace `λ/H` publishes unclamped above its rank of 1 instead of refusing the
/// fit or being clamped to the rank.
#[test]
fn a_double_well_fit_publishes_its_uncertified_trace_2901() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let result = fit_custom_family(&family, &[double_well_spec(2.0)], &double_well_options())
        .expect("a certified double-well mode publishes EDF without refusing");
    let bound = result.edf_rank_bound();
    assert_eq!(bound.len(), 1, "one penalty block: {bound:?}");
    assert!(
        matches!(
            bound[0],
            gam_solve::estimate::EdfRankBound::Uncertified { smallest_pivot, band }
                if smallest_pivot < -band
        ),
        "the double well's data curvature is resolved negative at its mode: {bound:?}"
    );
    let inference = result.inference.as_ref().expect("the fit computed inference");
    let hessian = inference.penalized_hessian.as_array()[[0, 0]];
    let lambda = result.lambdas[0];
    let exact = lambda / hessian;
    let trace = result.penalty_block_trace()[0];
    assert!(trace > 1.0, "the trace {trace} lies above the block's rank of 1");
    // The trace is `λ·(r·x̂)` with `x̂` a dense Cholesky solve of the 1×1 `H` and
    // `r = 1` the exact root of `S = [1]`. To first order in the unit roundoff
    // `u = ε/2`: `l = fl(√H)` enters squared (2u), the forward and back divisions
    // (2u), the product with λ (u), and the reference quotient `λ/H` itself (u),
    // so the two agree to `6u = 3ε` relative.
    assert!(
        (trace - exact).abs() <= 3.0 * f64::EPSILON * exact,
        "the published trace {trace} is λ/H = {exact} to one Cholesky solve and one product"
    );
    assert_eq!(
        result.edf_by_block()[0],
        1.0 - trace,
        "an uncertified block's EDF is published unclamped"
    );
}

/// A custom-family fit publishes its likelihood curvature `H − S(λ)` beside the
/// penalized Hessian, so the smooth-term score test has the `G` it needs. It is
/// the observed information, not a Gram: at the double well's certified mode
/// it is negative, and that sign is published rather than projected away.
#[test]
fn a_custom_family_fit_publishes_its_likelihood_curvature() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let result = fit_custom_family(&family, &[double_well_spec(2.0)], &double_well_options())
        .expect("a certified double-well mode fits");
    let inference = result.inference.as_ref().expect("the fit computed inference");
    let hessian = inference.penalized_hessian.as_array()[[0, 0]];
    let curvature = inference
        .weighted_gram
        .as_ref()
        .expect("an identity-gauge custom-family fit publishes its likelihood curvature");
    assert_eq!(curvature.dim(), (1, 1));
    let expected = hessian - result.lambdas[0];
    assert!(
        (curvature[[0, 0]] - expected).abs() <= 4.0 * f64::EPSILON * hessian.abs().max(result.lambdas[0]),
        "published curvature {} is H − λS = {expected}",
        curvature[[0, 0]]
    );
    assert!(curvature[[0, 0]] < 0.0, "the double well's data curvature is negative at its mode");
}

/// The end-to-end property: a whole production fit is a function of the model
/// and the data, not of the coefficients the caller happened to pass in.
///
/// This is the statement #2363 wants for cache state, obtained here from the
/// definition rather than from a per-family patch: the persistent cache seeds β,
/// and once β is selected by the continuation instead of by the seed, a warm
/// cache can change how fast the fit is reached but not where it lands.
/// The measured residual across seeds, `3.4e-14` relative, is far below what the
/// inner corrector resolves and far above zero. The cross-seed bound is stated
/// as the inner tolerance for that reason; a repeat-run control pins down which
/// of the two possible explanations applies, so the bound cannot quietly hide a
/// returning branch dependence. See the discussion on #2366.
#[test]
fn production_fit_is_independent_of_the_caller_seed_2366() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let options = double_well_options();
    let fit = |seed: f64| {
        let result = fit_custom_family(&family, &[double_well_spec(seed)], &options)
            .expect("double-well fit");
        (result.block_states[0].beta[0], result.log_lambdas[0])
    };

    // CONTROL: two runs from the SAME seed must be bitwise identical. Without
    // this, a cross-seed bound stated at a tolerance could be satisfied by a
    // run-to-run wobble that has nothing to do with the seed, and the test would
    // stop measuring what it claims to measure.
    let (repeat_a, _) = fit(2.0);
    let (repeat_b, _) = fit(2.0);
    assert_eq!(
        repeat_a.to_bits(),
        repeat_b.to_bits(),
        "two fits of the same problem from the same seed disagree ({repeat_a} vs \
         {repeat_b}), so this fixture cannot attribute any difference to the seed"
    );

    let (from_positive, rho_positive) = fit(2.0);
    let (from_negative, rho_negative) = fit(-2.0);
    let cross_seed_gap = (from_positive - from_negative).abs();
    // The certified smoothing parameter is asserted alongside the coefficient
    // because they fail separately: an earlier revision of this fix left β
    // agreeing to 3.4e-14 while ρ still moved by 1.3e-13, which is how the
    // remaining leak (the stall guard's cold pulse falling back to the caller's
    // coefficients) was found. Both are bitwise now, and both are checked so
    // that channel cannot silently reopen.
    assert_eq!(
        rho_positive.to_bits(),
        rho_negative.to_bits(),
        "the certified smoothing parameter depends on the caller's seed: \
         {rho_positive} vs {rho_negative}"
    );

    // The qualitative property: both seeds select the SAME branch. Before the
    // anchored continuation these two seeds converged into opposite wells, so
    // this gap was ≈ 2. Sign agreement is the branch statement; the magnitude
    // bound is what makes it a fit-level statement rather than a sign check.
    assert!(
        from_positive < 0.0 && from_negative < 0.0,
        "both fits should land on the anchor's branch (beta < 0); got \
         {from_positive} and {from_negative}"
    );
    assert_eq!(
        from_positive.to_bits(),
        from_negative.to_bits(),
        "the fitted coefficient depends on the caller's seed: {from_positive} \
         vs {from_negative} (gap {cross_seed_gap:.3e})"
    );
}

/// The mode the rule selects is the DEEP well — the anchor's own branch — not
/// merely a consistent one.
///
/// A rule that consistently picked the worse mode would satisfy the invariance
/// test above while making every fit worse, so the selection has to be checked
/// against the closed-form geometry as well.
#[test]
fn anchored_continuation_selects_the_anchor_branch_2366() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let target_rho = -6.0;
    let selected = continuation_mode(&family, &[double_well_spec(2.0)], target_rho);
    assert!(
        selected < -0.5,
        "the anchor is convexified around beta<0, so its branch is the deep well; got {selected}"
    );
    // The deep well is genuinely the better mode: w(-1) = -c < +c = w(+1).
    let shallow = cold_direct_mode(&family, &[double_well_spec(2.0)], target_rho);
    let objective = |beta: f64| {
        let well = beta * beta - 1.0;
        well * well + TILT * beta + 0.5 * (target_rho.exp()) * beta * beta
    };
    assert!(
        objective(selected) < objective(shallow),
        "selected mode {selected} (obj {}) should beat the seed-dependent mode {shallow} (obj {})",
        objective(selected),
        objective(shallow)
    );
}

/// gam#2928: the ladder keeps the anchor's corrected mode for its later sweeps,
/// and the kept mode answers only the anchor it was solved at. A different
/// anchor, one rounding step away, a different width, or a zero of the other
/// sign, is solved rather than read.
#[test]
fn the_kept_anchor_mode_answers_only_its_own_anchor_2928() {
    let anchor = array![8.0, 3.5];
    let kept = crate::fit::AnchorWaypointMode::new(
        &anchor,
        crate::assembly::ConstrainedWarmStart {
            rho: anchor.clone(),
            block_beta: vec![array![-0.25]],
            active_sets: vec![None],
            cached_inner: None,
        },
    );
    let read = kept.at(&anchor).expect("the anchor it was solved at");
    assert_eq!(read.block_beta[0][0].to_bits(), (-0.25_f64).to_bits());
    for other in [
        array![8.0, f64::from_bits(3.5_f64.to_bits() + 1)],
        array![8.0],
        array![8.0, 3.5, 0.0],
        array![8.0, -3.5],
    ] {
        assert!(
            kept.at(&other).is_none(),
            "anchor {other} read the mode kept for {anchor}"
        );
    }
    assert!(
        crate::fit::AnchorWaypointMode::new(&array![0.0], kept.at(&anchor).expect("kept").clone())
            .at(&array![-0.0])
            .is_none(),
        "a zero of the other sign is a different anchor"
    );
}

/// gam#2661: the rule that selected a fit's coefficient mode is recorded on the
/// fit, so a caller reads it rather than a log line. A certified anchored
/// continuation records itself. A nonconvex fit with no smoothing parameter has
/// no anchor, so it records that the caller's seed selected its mode. A declined
/// continuation records its refusal, which `require_rule_selected` names.
#[test]
fn the_fit_records_which_rule_selected_its_mode_2661() {
    use gam_solve::model_types::CoefficientModeSelection;
    let family = TiltedDoubleWellFamily::new(TILT);
    let options = double_well_options();

    let certified =
        fit_custom_family(&family, &[double_well_spec(2.0)], &options).expect("double-well fit");
    let selection = &certified.artifacts.coefficient_mode_selection;
    assert!(
        matches!(
            selection,
            CoefficientModeSelection::AnchoredContinuation { steps, endpoint_discrepancy }
                if *steps >= 1 && endpoint_discrepancy.is_finite()
        ),
        "a certified continuation recorded {selection:?}"
    );
    selection
        .require_rule_selected("double well")
        .expect("the anchored continuation selected the mode");

    let mut unpenalized = double_well_spec(2.0);
    unpenalized.penalties.clear();
    unpenalized.nullspace_dims.clear();
    unpenalized.initial_log_lambdas = Array1::zeros(0);
    let seed_fit =
        fit_custom_family(&family, &[unpenalized], &options).expect("unpenalized double-well fit");
    let seed_selection = &seed_fit.artifacts.coefficient_mode_selection;
    let CoefficientModeSelection::SeedSelected { reason } = seed_selection else {
        panic!("an unpenalized nonconvex fit recorded {seed_selection:?}");
    };
    assert!(reason.contains("no smoothing parameter"), "{reason}");
    assert!(
        seed_selection
            .require_rule_selected("unpenalized double well")
            .is_err()
    );

    let refusal = AnchoredContinuationRefusal::EmptySweep { steps: 1 };
    let declined = crate::fit::declined_continuation_selection(&refusal);
    assert_eq!(
        declined,
        CoefficientModeSelection::SeedSelected {
            reason: refusal.to_string()
        }
    );
    let refused = declined
        .require_rule_selected("declined")
        .expect_err("a seed-selected mode is refused");
    assert!(
        refused.contains(&refusal.to_string()),
        "the refusal is named: {refused}"
    );
    assert!(
        CoefficientModeSelection::NotRecorded
            .require_rule_selected("old payload")
            .is_err(),
        "an unrecorded rule is refused"
    );
}

/// The tilted double well's stationary points at `rho`, in closed form: the real roots of
/// `4β³ + (λ − 4)β + c = 0` with `λ = e^ρ`, ascending. Below the shallow well's fold there are
/// three (the deep minimum, the barrier, the shallow minimum); above it only the deep minimum.
fn double_well_stationary_points_2973(rho: f64) -> Vec<f64> {
    let lambda = rho.exp();
    let p = (lambda - 4.0) / 4.0;
    let q = TILT / 4.0;
    if 4.0 * p * p * p + 27.0 * q * q < 0.0 {
        let radius = 2.0 * (-p / 3.0).sqrt();
        let angle = ((3.0 * q / (2.0 * p)) * (-3.0 / p).sqrt()).acos() / 3.0;
        let mut roots: Vec<f64> = (0..3)
            .map(|k| radius * (angle - 2.0 * std::f64::consts::PI * f64::from(k) / 3.0).cos())
            .collect();
        roots.sort_by(f64::total_cmp);
        roots
    } else {
        let shift = (q * q / 4.0 + p * p * p / 27.0).sqrt();
        vec![(-q / 2.0 + shift).cbrt() + (-q / 2.0 - shift).cbrt()]
    }
}

/// Whether `beta` is the shallow minimum at `rho`: the closed form has three stationary points
/// there and `beta` is nearest the largest.
fn is_shallow_mode_2973(rho: f64, beta: f64) -> bool {
    let points = double_well_stationary_points_2973(rho);
    points.len() == 3
        && points
            .iter()
            .enumerate()
            .min_by(|left, right| (beta - left.1).abs().total_cmp(&(beta - right.1).abs()))
            .is_some_and(|(index, _)| index == 2)
}

/// Whether `beta` is the deep minimum at `rho`: nearest the smallest stationary point.
fn is_deep_mode_2973(rho: f64, beta: f64) -> bool {
    double_well_stationary_points_2973(rho)
        .iter()
        .enumerate()
        .min_by(|left, right| (beta - left.1).abs().total_cmp(&(beta - right.1).abs()))
        .is_some_and(|(index, _)| index == 0)
}

/// #2973: the Newton-region contraction test is Kantorovich's `h ≤ ½` read through Deuflhard's
/// computational estimate `[h₀] = 2Θ`. `Θ = ¼` is the last contraction it admits, with root
/// radius `2‖Δ⁰‖`; a stationary predictor is `Θ = 0` with radius `‖Δ⁰‖ = 0`; a non-finite
/// correction measures nothing.
#[test]
fn the_newton_region_test_is_kantorovich_h_at_most_one_half_2973() {
    let at_bound = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: 0.25,
    };
    assert!(at_bound.in_newton_region(), "Θ = ¼ is h = ½, inside the Newton region");
    assert_eq!(at_bound.root_radius(), Some(2.0), "at h = ½ the root radius is 2‖Δ⁰‖");
    let past = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: 0.25 + f64::EPSILON,
    };
    assert!(!past.in_newton_region(), "Θ above ¼ is outside the Newton region");
    assert_eq!(past.root_radius(), None);
    let stationary = NewtonRegionContraction {
        first_correction: 0.0,
        second_correction: 0.0,
    };
    assert_eq!(stationary.contraction_factor(), Some(0.0));
    assert_eq!(stationary.root_radius(), Some(0.0));
    let unmeasured = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: f64::NAN,
    };
    assert_eq!(unmeasured.contraction_factor(), None, "a non-finite correction measures nothing");
    assert!(!unmeasured.in_newton_region());
}

/// #2973 pin 1: the shallow branch is followed to its fold, and the continuation declines past
/// it.
///
/// From the shallow mode at ρ = 0.5, continuation to ρ = 0.9 and 0.97 publishes the closed
/// form's shallow minimum. ρ = 1.2 lies past the fold at `ρ_fold = ln(4 − 3c^{2/3}) ≈ 0.97666`,
/// where the shallow well no longer exists, so the continuation refuses as `FoldReached`, with
/// its last certified point between the start and the fold. The warm-start-only rule published
/// the deep mode at ρ = 1.2 instead (census 1334963: β = −0.5528 from β = 0.4775).
#[test]
fn the_shallow_branch_is_followed_to_its_fold_and_declines_past_it_2973() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let specs = [double_well_spec(2.0)];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let rho_fold = (4.0 - 3.0 * TILT.powf(2.0 / 3.0)).ln();
    let start = outerobjectivegradienthessian_labeled(
        &family,
        &specs,
        &options,
        &layout,
        &array![0.5],
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueAndGradient,
    )
    .expect("the derivative-bearing evaluation at the shallow start");
    assert!(start.inner_converged, "the shallow start converges");
    let start_beta = start.warm_start.block_beta[0][0];
    assert!(
        is_shallow_mode_2973(0.5, start_beta),
        "a seed at +2 starts in the shallow well; got {start_beta}"
    );
    for target in [0.9, 0.97] {
        match continue_branch(
            &family,
            &specs,
            &options,
            &layout,
            &gam_problem::RhoPrior::Flat,
            &start.warm_start,
            &array![target],
            EvalMode::ValueAndGradient,
        ) {
            Ok(continuation) => {
                let beta = continuation.eval.warm_start.block_beta[0][0];
                assert!(
                    continuation.eval.inner_converged,
                    "the continued evaluation at rho={target} converges"
                );
                assert!(
                    is_shallow_mode_2973(target, beta),
                    "the shallow branch exists at rho={target}, so the continuation publishes it; \
                     got beta={beta} after {} sub-step attempts",
                    continuation.attempts
                );
            }
            Err(refusal) => panic!(
                "the shallow branch exists at rho={target}, so the continuation must reach it: \
                 {refusal}"
            ),
        }
    }
    // The Newton-region test at the IFT predictor for `to`, from the certified shallow mode at
    // `from`: the mode's coefficient, the predictor, and the two corrections.
    let predicted_step = |from: f64, to: f64| {
        let mode = match continue_branch(
            &family,
            &specs,
            &options,
            &layout,
            &gam_problem::RhoPrior::Flat,
            &start.warm_start,
            &array![from],
            EvalMode::ValueAndGradient,
        ) {
            Ok(continuation) => continuation.eval.warm_start,
            Err(refusal) => panic!("the shallow branch exists at rho={from}: {refusal}"),
        };
        // The closed-form IFT tangent of the tilted double well: stationarity is
        // 4β³ − 4β + c + λβ = 0, so dβ/dρ = −λβ / (12β² − 4 + λ).
        let beta = mode.block_beta[0][0];
        let lambda = from.exp();
        let tangent = -lambda * beta / (12.0 * beta * beta - 4.0 + lambda);
        let predictor_beta = beta + (to - from) * tangent;
        let predictor = crate::assembly::ConstrainedWarmStart {
            rho: array![to],
            block_beta: vec![array![predictor_beta]],
            active_sets: mode.active_sets.clone(),
            cached_inner: None,
        };
        let test = newton_region_contraction(
            &family,
            &specs,
            &options,
            &layout,
            &array![to],
            &predictor,
        )
        .expect("the Newton-region test runs at the predictor");
        let theta = test
            .contraction
            .contraction_factor()
            .map_or_else(|| "unmeasured".to_string(), |theta| format!("{theta:.6e}"));
        eprintln!(
            "[2973 predicted step] rho {from} -> {to} from beta={:.6e}: predictor beta={predictor_beta:.6e}, \
             Newton correction {:.6e}, simplified correction {:.6e}, contraction {theta}",
            mode.block_beta[0][0],
            test.contraction.first_correction,
            test.contraction.second_correction,
        );
        (test.contraction, theta)
    };
    // Gate 1336821's failing step: from the certified shallow mode at 0.85 the tangent predicts
    // past the fold at 1.2. The Newton correction jumps the barrier (to β ≈ −1.96), and the
    // simplified correction, with the curvature of the predictor, does not contract.
    let (fold_step, fold_theta) = predicted_step(0.85, 1.2);
    assert!(
        !fold_step.in_newton_region(),
        "the predictor past the fold is outside the Newton region of any root the corrector \
         reaches; contraction {fold_theta}"
    );
    // A step the frozen curvature must refuse and a re-read curvature would admit: from 0.8 the
    // tangent predicts at 1.37, the Newton correction lands in the deep basin, and in the closed
    // form the simplified correction contracts by 2.6 while a second Newton step at the deep
    // point would read 0.185.
    let (jump_step, jump_theta) = predicted_step(0.8, 1.37);
    assert!(
        !jump_step.in_newton_region(),
        "a first correction that jumps the barrier is outside the Newton region with the \
         curvature frozen at the predictor; contraction {jump_theta}"
    );
    match continue_branch(
        &family,
        &specs,
        &options,
        &layout,
        &gam_problem::RhoPrior::Flat,
        &start.warm_start,
        &array![1.2],
        EvalMode::ValueAndGradient,
    ) {
        Ok(continuation) => panic!(
            "past the fold at rho={rho_fold} the shallow branch does not exist, so the \
             continuation must decline; it published beta={}",
            continuation.eval.warm_start.block_beta[0][0]
        ),
        Err(refusal) => {
            eprintln!("[2973 fold] {refusal}");
            let BranchContinuationRefusal::FoldReached {
                last_certified_rho, ..
            } = &refusal
            else {
                panic!("the continuation must decline at the fold, not with: {refusal}");
            };
            assert!(
                last_certified_rho[0] > 0.5 && last_certified_rho[0] < rho_fold,
                "the last certified point lies on the shallow branch between the start and the \
                 fold at {rho_fold}; got {}",
                last_certified_rho[0]
            );
            // The branch is followed to its fold, not abandoned short of it: the last certified
            // point is within 1e-3 of the closed-form fold (measured 9.766626e-1 against
            // 9.76663e-1).
            assert!(
                rho_fold - last_certified_rho[0] < 1e-3,
                "the continuation stops {} short of the fold at {rho_fold}",
                rho_fold - last_certified_rho[0]
            );
        }
    }
}

/// #2973: a continuation predicts along the IFT tangent of the certified mode, formed from the
/// curvature that mode's own solve ended on. On the tilted double well the stationarity condition
/// is `4β³ − 4β + c + λβ = 0`, so `dβ̂/dρ = −λβ̂ / (12β̂² − 4 + λ)`, and the predictor from the deep
/// mode at ρ = 0 to ρ = 0.3 must be `β̂ + 0.3·dβ̂/dρ` to rounding. A predictor that dropped the
/// tangent, flipped its sign, or read it per λ instead of per ρ fails here.
#[test]
fn the_ift_predictor_is_the_closed_form_tangent_2973() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let specs = [double_well_spec(-2.0)];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let certified = outerobjectivegradienthessian_labeled(
        &family,
        &specs,
        &options,
        &layout,
        &array![0.0],
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueOnly,
    )
    .expect("the deep mode at rho=0")
    .warm_start;
    let beta = certified.block_beta[0][0];
    assert!(
        is_deep_mode_2973(0.0, beta),
        "a seed at -2 certifies the deep mode; got {beta}"
    );
    let curvature = certified
        .cached_inner
        .as_ref()
        .and_then(|cached| cached.terminal_working_sets.as_deref())
        .and_then(|sets| sets.first())
        .expect("the certified mode files its terminal exact-Newton working set");
    let predictor = single_block_ift_predictor(
        &family,
        &specs,
        &[array![0.0]],
        &[array![0.3]],
        &options,
        &certified.block_beta,
        curvature,
        None,
    )
    .expect("the IFT predictor forms at a certified mode")[0][0];
    let lambda = 0.0_f64.exp();
    let expected = beta + 0.3 * (-lambda * beta / (12.0 * beta * beta - 4.0 + lambda));
    assert!(
        (predictor - expected).abs() <= 1e-12 * (1.0 + beta.abs()),
        "the IFT predictor {predictor} is not the closed-form tangent predictor {expected}"
    );
}

/// #2973 pin 2: a walk that tries to cross the fold publishes one branch per θ.
///
/// Census 1334963's scripted walk: `CustomOuterState` driven as the fit's closures drive it, from
/// a shallow starting incumbent through ρ = 0.5, 0.9, 1.2, 0.9, 0.5, every evaluated trial
/// accepted, now through the rule the closures apply, [`evaluate_on_branch`]. The trial at 1.2
/// is refused, a rejected trial that records nothing, so every θ the walk visits publishes the
/// shallow branch. Under the warm-start-only rule the 1.2 trial published the deep mode, was
/// accepted, and the return visits to 0.9 and 0.5 published the deep mode too.
#[test]
fn a_walk_that_tries_to_cross_the_fold_publishes_one_branch_per_theta_2973() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let specs = [double_well_spec(2.0)];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let accepted = Arc::new(AtomicUsize::new(0));
    let mut state = CustomOuterState::new_with_cold_signal(
        None,
        Arc::new(AtomicBool::new(false)),
        Arc::clone(&accepted),
    );
    let mut published: Vec<(f64, f64)> = Vec::new();
    let mut refused: Vec<f64> = Vec::new();
    for theta in [0.5, 0.9, 1.2, 0.9, 0.5] {
        state.adopt_accepted_steps();
        let point = array![theta];
        let result = evaluate_on_branch(
            &family,
            &specs,
            &options,
            &layout,
            &point,
            state.warm_start_for(&point),
            &gam_problem::RhoPrior::Flat,
            EvalMode::ValueAndGradient,
        );
        match result {
            Ok(eval) => {
                assert!(eval.inner_converged, "the evaluation at rho={theta} converges");
                published.push((theta, eval.warm_start.block_beta[0][0]));
                state.record_first_order_mode(eval.warm_start.clone());
                accepted.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                assert!(
                    error.is_trial_point_infeasible(),
                    "a branch that ends is a rejected trial, not a fit failure: {error}"
                );
                refused.push(theta);
            }
        }
    }
    assert_eq!(
        refused,
        vec![1.2],
        "only the trial past the fold is refused; published {published:?}"
    );
    for (theta, beta) in &published {
        assert!(
            is_shallow_mode_2973(*theta, *beta),
            "every evaluation of the walk publishes its shallow branch; rho={theta} gave \
             beta={beta} (published {published:?})"
        );
    }
}

/// #2973 pin 3, the negative control: the deep branch has no fold, so its continuation certifies
/// from ρ = −6 to ρ = 18. Census 1288798's leading-order fold radius τ* predicted folds on this
/// branch 0.13–0.3 away near ρ ≈ 1.6; the contraction test measures the corrector instead.
#[test]
fn the_deep_branch_continues_across_every_rho_2973() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let specs = [double_well_spec(-2.0)];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let start = outerobjectivegradienthessian_labeled(
        &family,
        &specs,
        &options,
        &layout,
        &array![-6.0],
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueAndGradient,
    )
    .expect("the derivative-bearing evaluation at the deep start");
    assert!(start.inner_converged, "the deep start converges");
    assert!(
        is_deep_mode_2973(-6.0, start.warm_start.block_beta[0][0]),
        "a seed at -2 starts in the deep well"
    );
    match continue_branch(
        &family,
        &specs,
        &options,
        &layout,
        &gam_problem::RhoPrior::Flat,
        &start.warm_start,
        &array![18.0],
        EvalMode::ValueAndGradient,
    ) {
        Ok(continuation) => {
            let beta = continuation.eval.warm_start.block_beta[0][0];
            assert!(
                is_deep_mode_2973(18.0, beta),
                "the deep branch continues to rho=18; got beta={beta}"
            );
        }
        Err(refusal) => panic!("the deep branch has no fold, so it must continue: {refusal}"),
    }
}
