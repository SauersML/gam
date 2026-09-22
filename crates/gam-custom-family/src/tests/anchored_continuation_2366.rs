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
    coefficient_objective_homotopy_seed, continuation_refinement_decision,
};
use crate::penalty_labels::penalty_label_layout_with_joint;
use crate::test_support::outerobjectivegradienthessian_labeled;

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
    /// `Some(t)` on a homotopy member at progress `t`, `None` on the production
    /// family — which evaluates its expressions term for term as it always has, so
    /// every other fixture in this file reads bit-identical arithmetic (gam#2661).
    homotopy_progress: Option<f64>,
    /// Counts every homotopy member this family is asked to construct, and every
    /// ZERO member separately, so a test can count the zero-member solves one ladder
    /// pays. `None` offers no homotopy at all, which is what every other fixture here
    /// wants: the seed tries the objective homotopy BEFORE the anchored continuation.
    homotopy_builds: Option<Arc<AtomicUsize>>,
    homotopy_zero_builds: Option<Arc<AtomicUsize>>,
}

/// The convexifier a homotopy member blends the double well into, so the zero
/// member's coefficient mode is unique as the homotopy contract requires: at
/// `progress = 0` the objective is `c·β + κ·β²` with `κ > 0`, whose observed
/// information `2κ` is positive definite on its own.
const HOMOTOPY_CONVEXIFIER: f64 = 2.0;

impl TiltedDoubleWellFamily {
    fn new(tilt: f64) -> Self {
        Self {
            tilt,
            outer_curvature_calls: None,
            homotopy_progress: None,
            homotopy_builds: None,
            homotopy_zero_builds: None,
        }
    }

    /// The same family, offering the Jeffreys-style objective homotopy and counting
    /// what it is asked to build.
    fn with_homotopy(tilt: f64, builds: &Arc<AtomicUsize>, zero_builds: &Arc<AtomicUsize>) -> Self {
        Self {
            homotopy_builds: Some(Arc::clone(builds)),
            homotopy_zero_builds: Some(Arc::clone(zero_builds)),
            ..Self::new(tilt)
        }
    }

    /// `(quartic weight, convexifier weight)` at a homotopy member, `None` on the
    /// production family.
    fn homotopy_weights(&self) -> Option<(f64, f64)> {
        self.homotopy_progress
            .map(|progress| (progress, (1.0 - progress) * HOMOTOPY_CONVEXIFIER))
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
        if let Some((quartic, convex)) = self.homotopy_weights() {
            return Ok(FamilyEvaluation {
                log_likelihood: -(quartic * well * well + self.tilt * beta + convex * beta * beta),
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![
                        -(quartic * (4.0 * beta * beta * beta - 4.0 * beta)
                            + self.tilt
                            + 2.0 * convex * beta)
                    ],
                    hessian: SymmetricMatrix::Dense(array![[
                        quartic * (12.0 * beta * beta - 4.0) + 2.0 * convex
                    ]]),
                }],
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: -(well * well + self.tilt * beta),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                // dℓ/dβ, and the observed information −d²ℓ/dβ².
                gradient: array![-(4.0 * beta * beta * beta - 4.0 * beta + self.tilt)],
                hessian: SymmetricMatrix::Dense(array![[12.0 * beta * beta - 4.0]]),
            }],
        })
    }

    fn coefficient_mode_homotopy_member(&self, progress: f64) -> Result<Option<Self>, String> {
        if !progress.is_finite() || !(0.0..=1.0).contains(&progress) {
            return Err(format!(
                "tilted double-well homotopy progress must lie in [0, 1], got {progress}"
            ));
        }
        let Some(builds) = self.homotopy_builds.as_ref() else {
            return Ok(None);
        };
        builds.fetch_add(1, Ordering::Relaxed);
        if progress == 0.0
            && let Some(zero_builds) = self.homotopy_zero_builds.as_ref()
        {
            zero_builds.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Some(Self {
            homotopy_progress: Some(progress),
            ..self.clone()
        }))
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states)?;
        if let Some((quartic, convex)) = self.homotopy_weights() {
            return Ok(Some(array![[
                quartic * (12.0 * beta * beta - 4.0) + 2.0 * convex
            ]]));
        }
        Ok(Some(array![[12.0 * beta * beta - 4.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states)?;
        let step = direction.first().copied().unwrap_or(0.0);
        if let Some((quartic, _)) = self.homotopy_weights() {
            return Ok(Some(array![[quartic * 24.0 * beta * step]]));
        }
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
        let hessian = match self.homotopy_weights() {
            Some((quartic, convex)) => {
                array![[quartic * (12.0 * beta * beta - 4.0) + 2.0 * convex]]
            }
            None => array![[12.0 * beta * beta - 4.0]],
        };
        Ok(Some(ExactNewtonOuterCurvature {
            hessian,
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
    /// The depth this witness declares, or `None` to be priced by the production
    /// corrector budget (gam#2612).
    refinement_budget: Option<usize>,
}

impl ScriptedContinuationPath {
    /// A scripted path priced by the production corrector budget, for the fixtures that
    /// assert what that budget buys.
    fn new(criterion_by_refinement: Vec<f64>) -> Self {
        Self {
            criterion_by_refinement,
            sweeps: std::cell::RefCell::new(Vec::new()),
            refinement_budget: None,
        }
    }

    /// A scripted path refined through its WHOLE script: one rung per entry, so
    /// `len - 1` refinements (gam#2612).
    ///
    /// A script is a witness, not a fit. It runs no correctors — [`Self::sweep`] returns a
    /// canned criterion value — so the corrector budget does not price it, and a script
    /// written to show a shape needs exactly the rungs that shape takes. `ecfd2c33ee`
    /// obtained this by passing `outer_max_iter = 100` to the then-parameterized budget,
    /// which is `floor(log2 100) - 1 = 5`, the depth its six-rung trail needs; the depth is
    /// now taken from the script itself so it cannot drift from it.
    fn walking_its_whole_script(criterion_by_refinement: Vec<f64>) -> Self {
        let refinement_budget = Some(criterion_by_refinement.len().saturating_sub(1));
        Self {
            criterion_by_refinement,
            sweeps: std::cell::RefCell::new(Vec::new()),
            refinement_budget,
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

    fn refinement_budget(&self) -> usize {
        self.refinement_budget
            .unwrap_or_else(crate::fit::continuation_refinement_budget)
    }

    fn observation_count(&self) -> usize {
        SCRIPTED_OBSERVATIONS
    }
}

/// The observation count the scripted criterion is summed over. The ladder is
/// judged in the criterion's own units against `τ_stat = 1/(2n)`, so this sets
/// the resolution the fixtures below are written against: `1e-6`.
const SCRIPTED_OBSERVATIONS: usize = 500_000;

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
    // strictly improving, never agreeing to `τ_stat = 1e-6`.
    let script: Vec<f64> = (0..40).map(|k| 1.0 + 0.999_f64.powi(k)).collect();
    let path = ScriptedContinuationPath::new(script);
    let options = double_well_options();
    let budget = crate::fit::continuation_refinement_budget();
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

/// The bound is the seed ladder's own declared corrector budget, and the outer
/// search's stop cannot move it (#4566).
#[test]
fn the_refinement_budget_is_the_seed_ladders_corrector_budget_2661() {
    let budget = crate::fit::continuation_refinement_budget();
    let correctors = (1usize << (budget + 1)) - 1;
    assert!(
        correctors <= crate::fit::CONTINUATION_SEED_CORRECTOR_BUDGET,
        "a ladder through D={budget} refinements runs {correctors} correctors, which must fit \
         in the declared seed budget {}",
        crate::fit::CONTINUATION_SEED_CORRECTOR_BUDGET
    );
    assert!(
        ((1usize << (budget + 2)) - 1) > crate::fit::CONTINUATION_SEED_CORRECTOR_BUDGET,
        "D={budget} must be the LARGEST depth the declared seed budget affords, otherwise the \
         ladder is refusing paths the budget can pay for"
    );
    assert!(
        budget >= crate::fit::REQUIRED_CONSECUTIVE_AGREEMENTS + 1,
        "a budget below the fewest refinements that can produce a verdict is a disablement, \
         not a budget: D={budget}"
    );

    // The decoupling itself is the signature: the budget reads no outer count,
    // so no outer stop -- bounded or unbounded -- can move the ladder's depth.
    // Read through the retired expression `floor(log2(outer_max_iter)) - 1`, the
    // shipped unbounded outer search would give D = 62, a ladder of 2^63
    // correctors, which is the operationally unbounded loop #2661 closed. The
    // two brackets above are the whole verdict: D is the largest depth the
    // declared corrector budget pays for, whatever that budget is set to.
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
    let path = ScriptedContinuationPath::walking_its_whole_script(script);
    let options = double_well_options();
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
    let path = ScriptedContinuationPath::walking_its_whole_script(vec![
        10.607, 11.387, 11.387, 10.594, 10.594, 10.594,
    ]);
    let options = double_well_options();
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
        outer_curvature_calls: Some(Arc::clone(&outer_curvature_calls)),
        ..TiltedDoubleWellFamily::new(TILT)
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

/// gam#2661/#2928: the objective-homotopy ladder solves its ZERO member ONCE, not once
/// per refinement.
///
/// `2c68304d2c` established this for the sibling anchored ladder: every sweep's step 0
/// corrects the same problem from the same cold start, so each refinement repeated one
/// deterministic computation. The homotopy ladder's step 0 has the identical property —
/// the member at progress `0.0`, at the same `ρ`, with no predecessor to carry, since
/// step 0 is where the carry begins — and did not have the fix.
///
/// Counting member CONSTRUCTIONS counts zero-member solves exactly: a sweep builds the
/// member immediately before correcting it, and a sweep that reads the kept mode builds
/// nothing. One construction belongs to the seed's own availability gate, so the ladder's
/// zero-member solves are `zero − 1`.
#[test]
fn the_homotopy_ladder_solves_its_zero_member_once_2661() {
    let builds = Arc::new(AtomicUsize::new(0));
    let zero_builds = Arc::new(AtomicUsize::new(0));
    let family = TiltedDoubleWellFamily::with_homotopy(TILT, &builds, &zero_builds);
    let specs = vec![double_well_spec(-2.0)];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");

    let seed = coefficient_objective_homotopy_seed(
        &family,
        &specs,
        &options,
        &layout,
        &gam_problem::RhoPrior::Flat,
        &array![-2.0],
    );
    let verdict = match &seed {
        Ok(Some(certified)) => format!("certified at {} steps", certified.certificate.steps),
        Ok(None) => "no homotopy offered".to_string(),
        Err(refusal) => format!("declined: {refusal}"),
    };
    let total = builds.load(Ordering::Relaxed);
    let zero = zero_builds.load(Ordering::Relaxed);

    // CONTROL, and it does not read the ladder's verdict — the count below must
    // discriminate whether the ladder certifies or spends its budget. A sweep at `s`
    // steps builds members at progress `1/s … (s−1)/s`, since the target waypoint uses
    // the family itself, plus the zero member on whichever sweep solves it. So sweeps at
    // 1, 2 and 4 steps build `1 (gate) + 1 (zero) + 0 + 1 + 3 = 6`, while two sweeps
    // build 3. More than four constructions therefore proves at least three sweeps ran,
    // and an unfixed ladder would have solved the zero member once in each of them.
    assert!(
        total > 4,
        "the ladder must run more than two sweeps or this test cannot discriminate \
         (member constructions: {total}, {verdict})"
    );
    assert_eq!(
        zero, 2,
        "the zero member must be built twice — once by the seed's availability gate and \
         once by the ladder's single solve — but was built {zero} times across {total} \
         member constructions ({verdict})"
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
        first_resolution: 0.0,
        second_resolution: 0.0,
    };
    assert!(at_bound.in_newton_region(), "Θ = ¼ is h = ½, inside the Newton region");
    assert_eq!(at_bound.root_radius(), Some(2.0), "at h = ½ the root radius is 2‖Δ⁰‖");
    let past = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: 0.25 + f64::EPSILON,
        first_resolution: 0.0,
        second_resolution: 0.0,
    };
    assert!(!past.in_newton_region(), "Θ above ¼ is outside the Newton region");
    assert_eq!(past.root_radius(), None);
    let stationary = NewtonRegionContraction {
        first_correction: 0.0,
        second_correction: 0.0,
        first_resolution: 0.0,
        second_resolution: 0.0,
    };
    assert_eq!(stationary.contraction_factor(), Some(0.0));
    assert_eq!(stationary.root_radius(), Some(0.0));
    let unmeasured = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: f64::NAN,
        first_resolution: 0.0,
        second_resolution: 0.0,
    };
    assert_eq!(unmeasured.contraction_factor(), None, "a non-finite correction measures nothing");
    assert!(!unmeasured.in_newton_region());
}

/// #2973: `Θ` is read only where both corrections are resolved.
///
/// A correction at or below its resolution is zero on the arithmetic, so the iterate it starts
/// from is at its root: `Θ = 0`, whatever the ratio of the two rounding errors. The root radius is
/// widened by both resolutions. A resolved pair is judged by its ratio as before, and a
/// non-finite resolution measures nothing.
#[test]
fn a_correction_within_its_resolution_is_converged_2973() {
    // The slope-surface refusal: both corrections under a resolution of 7.1e-9.
    let rounding = NewtonRegionContraction {
        first_correction: 9.2e-11,
        second_correction: 8.2e-11,
        first_resolution: 7.1e-9,
        second_resolution: 7.1e-9,
    };
    assert_eq!(rounding.contraction_factor(), Some(0.0));
    assert!(rounding.in_newton_region());
    assert_eq!(rounding.root_radius(), Some(9.2e-11 + 7.1e-9 + 7.1e-9));
    // A resolved Newton correction whose simplified correction is rounding: Newton converged in
    // one step.
    let converged_in_one = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: 0.5,
        first_resolution: 1.0e-9,
        second_resolution: 0.6,
    };
    assert_eq!(converged_in_one.contraction_factor(), Some(0.0));
    // The same pair with both corrections resolved does not contract.
    let resolved = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: 0.5,
        first_resolution: 1.0e-9,
        second_resolution: 0.1,
    };
    assert_eq!(resolved.contraction_factor(), Some(0.5));
    assert!(!resolved.in_newton_region());
    let unresolvable = NewtonRegionContraction {
        first_correction: 1.0,
        second_correction: 0.5,
        first_resolution: f64::INFINITY,
        second_resolution: 0.1,
    };
    assert_eq!(unresolvable.contraction_factor(), None);
}

/// A two-coefficient quadratic `ℓ(β) = −½w‖β − c‖²` under the rank-one penalty `λ·vvᵀ`,
/// `v = (1, −3)`, whose mode is closed-form: `β̂ = c − λ(vᵀc) / (w + λ‖v‖²)·v`.
///
/// At `λ = 1e9` the penalty product `S_λβ̂` sums terms of order `λ‖β̂‖` to an order-one result, so
/// the Newton correction at `β̂` is the rounding of that sum, carried along the penalty's null
/// direction `(3, 1)`, where the curvature is `w`.
#[derive(Clone)]
struct RoundedQuadraticFamily {
    center: Array1<f64>,
    curvature: f64,
}

impl CustomFamily for RoundedQuadraticFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = &block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta;
        let residual = &self.center - beta;
        Ok(FamilyEvaluation {
            log_likelihood: -0.5 * self.curvature * residual.dot(&residual),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: &residual * self.curvature,
                hessian: SymmetricMatrix::Dense(Array2::eye(2) * self.curvature),
            }],
        })
    }

    // The joint Hessian is declared, so the inner certificate measures the stationarity residual.
    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let width = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .len();
        Ok(Some(Array2::eye(width) * self.curvature))
    }
}

/// The rounded quadratic's one block: an identity design and the penalty `vvᵀ`, whose null
/// direction is declared.
fn rounded_quadratic_spec() -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "rounded".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::eye(2))),
        offset: Array1::zeros(2),
        penalties: vec![PenaltyMatrix::Dense(array![[1.0, -3.0], [-3.0, 9.0]])],
        nullspace_dims: vec![1],
        initial_log_lambdas: array![0.0],
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// #2973: a predictor at its root to rounding is in the Newton region.
///
/// The predictor is the closed-form mode at `λ = 1e9`. Its Newton correction and the simplified
/// correction after it are both rounding of the penalty product, so their ratio is a ratio of
/// rounding errors, which the test used to read as a contraction and refuse as a fold. Each lies
/// within its resolution, so the predictor is converged: `Θ = 0`, and the root radius covers both
/// corrections.
#[test]
fn a_predictor_at_its_root_to_rounding_is_in_the_newton_region_2973() {
    let specs = [rounded_quadratic_spec()];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let rho = 1.0e9_f64.ln();
    let lambda = rho.exp();
    let v = array![1.0, -3.0];
    for k in 0..8 {
        let center = array![1.0 + 0.37 * k as f64, -0.5 + 0.21 * k as f64];
        let mode = &center - &(&v * (lambda * v.dot(&center) / (1.0 + lambda * v.dot(&v))));
        let predictor = crate::assembly::ConstrainedWarmStart {
            rho: array![rho],
            block_beta: vec![mode],
            active_sets: vec![None],
            cached_inner: None,
        };
        let family = RoundedQuadraticFamily {
            center,
            curvature: 1.0,
        };
        let test =
            newton_region_contraction(&family, &specs, &options, &layout, &array![rho], &predictor)
                .expect("the Newton-region test runs at the closed-form mode");
        let contraction = test.contraction;
        eprintln!("[2973 rounded mode] center {k}: {contraction:?}");
        assert!(
            contraction.first_correction <= contraction.first_resolution,
            "the closed-form mode's Newton correction is rounding: {contraction:?}"
        );
        assert!(
            contraction.in_newton_region(),
            "a predictor at its root to rounding is in the Newton region: {contraction:?}"
        );
        assert!(
            contraction.root_radius().is_some_and(
                |radius| radius >= contraction.first_correction + contraction.first_resolution
            ),
            "the root radius covers the rounding it was measured through: {contraction:?}"
        );
    }
}

/// #2973: a mode at its root to rounding certifies, and its branch continues.
///
/// At `λ = 1e9` the rounded quadratic's stationarity residual at its mode is the rounding of the
/// penalty product, far above the `inner_tol`-scaled target, and the line search takes steps of
/// the same rounding. The certificate's target is never below the residual's own rounding band,
/// as on the joint path (#2812), and a residual inside that band makes the cycle's step rounding
/// too, so the inner solve certifies the mode, and the continuation from it to `λ = 2e9`
/// publishes a mode that is again at its root to rounding. Without the band the solve reported
/// the mode unconverged and every corrector of the continuation refused, which is how the
/// event-history slope surface's continuations ended as folds.
#[test]
fn a_mode_at_its_root_to_rounding_certifies_and_continues_2973() {
    let family = RoundedQuadraticFamily {
        center: array![1.3, -0.4],
        curvature: 1.0,
    };
    let specs = [rounded_quadratic_spec()];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let from = 1.0e9_f64.ln();
    let start = outerobjectivegradienthessian_labeled(
        &family,
        &specs,
        &options,
        &layout,
        &array![from],
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueAndGradient,
    )
    .expect("the evaluation at λ = 1e9");
    assert!(
        start.inner_converged,
        "the mode at its root to rounding certifies"
    );
    let to = 2.0e9_f64.ln();
    let continuation = match continue_branch(
        &family,
        &specs,
        &options,
        &layout,
        &start.warm_start,
        &array![to],
    ) {
        Ok(continuation) => continuation,
        Err(refusal) => panic!("the quadratic's single branch continues to λ = 2e9: {refusal}"),
    };
    assert!(
        continuation.inner.converged,
        "the continued mode certifies"
    );
    let at_target = newton_region_contraction(
        &family,
        &specs,
        &options,
        &layout,
        &array![to],
        &constrained_warm_start_from_inner(&array![to], &continuation.inner),
    )
    .expect("the Newton-region test runs at the continued mode");
    assert!(
        at_target.contraction.first_correction <= at_target.contraction.first_resolution,
        "the continued mode is at its root to rounding: {:?}",
        at_target.contraction
    );
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
            &start.warm_start,
            &array![target],
        ) {
            Ok(continuation) => {
                let beta = continuation.inner.block_states[0].beta[0];
                assert!(
                    continuation.inner.converged,
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
            &start.warm_start,
            &array![from],
        ) {
            Ok(continuation) => {
                constrained_warm_start_from_inner(&array![from], &continuation.inner)
            }
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
        &start.warm_start,
        &array![1.2],
    ) {
        Ok(continuation) => panic!(
            "past the fold at rho={rho_fold} the shallow branch does not exist, so the \
             continuation must decline; it published beta={}",
            continuation.inner.block_states[0].beta[0]
        ),
        Err(refusal) => {
            eprintln!("[2973 fold] {refusal}");
            let BranchContinuationRefusal::FoldReached {
                last_certified_rho, ..
            } = &refusal
            else {
                panic!("the continuation must decline at the fold, not with: {refusal}");
            };
            // The branch is followed to its fold, to the arithmetic's resolution: the last certified
            // point is the closed-form fold in f64 (measured 0.9766625503626531 for both; the next
            // sub-step past it reads λ_min = −5.8e-9 and is refused as indefinite), and never past
            // it.
            assert!(
                last_certified_rho[0] > 0.5 && last_certified_rho[0] <= rho_fold,
                "the last certified point lies on the shallow branch between the start and the \
                 fold at {rho_fold}; got {}",
                last_certified_rho[0]
            );
            // Not abandoned short of the fold either.
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

/// #2973: the IFT predictor moves along the tangent of the penalty the solve minimizes.
///
/// The rounded quadratic's stored penalty here is `vvᵀ + ε·nnᵀ` with `n = (3, 1)` its declared
/// null direction and `ε = 1e-12` the formation error on it. The solve reads the structural root,
/// which the declared nullity caps at rank one (#2954), so its penalty is `½λ(vᵀβ)²` and its
/// tangent `dβ̂/dρ = −λ(vᵀβ̂)/(w + λ‖v‖²)·v` has no component along `n`. The predictor from
/// `λ = 1e9` to `2e9` must be `β̂ + ln 2·dβ̂/dρ` to rounding. A predictor that formed `S_λ` and
/// `S_k β̂` from the stored matrix moved `β̂` by `−ln 2·λε(nᵀβ̂)/(w + λε‖n‖²)·n`, a coefficient of
/// 2.4e-3 at this fixture's centre, off the branch the solve publishes.
#[test]
fn the_ift_predictor_moves_along_the_solves_own_penalty_2973() {
    let v = array![1.0, -3.0];
    let n = array![3.0, 1.0];
    let formation_error = 1.0e-12;
    let mut spec = rounded_quadratic_spec();
    let stored = {
        let range = v.view().insert_axis(ndarray::Axis(1));
        let null = n.view().insert_axis(ndarray::Axis(1));
        range.dot(&range.t()) + null.dot(&null.t()) * formation_error
    };
    spec.penalties = vec![PenaltyMatrix::Dense(stored)];
    let specs = [spec];
    let options = double_well_options();
    let curvature = 1.0;
    let center = array![1.3, -0.4];
    let family = RoundedQuadraticFamily {
        center: center.clone(),
        curvature,
    };
    let from = 1.0e9_f64.ln();
    let to = 2.0e9_f64.ln();
    let lambda = from.exp();
    let mode = &center - &(&v * (lambda * v.dot(&center) / (curvature + lambda * v.dot(&v))));
    let working_set = BlockWorkingSet::ExactNewton {
        gradient: (&center - &mode) * curvature,
        hessian: SymmetricMatrix::Dense(Array2::eye(2) * curvature),
    };
    let predictor = single_block_ift_predictor(
        &family,
        &specs,
        &[array![from]],
        &[array![to]],
        &options,
        std::slice::from_ref(&mode),
        &working_set,
        None,
    )
    .expect("the IFT predictor forms at the closed-form mode")
    .remove(0);
    let tangent = &v * (-lambda * v.dot(&mode) / (curvature + lambda * v.dot(&v)));
    let expected = &mode + &(&tangent * (to - from));
    let miss = (&predictor - &expected)
        .mapv(f64::abs)
        .fold(0.0_f64, |a, &b| a.max(b));
    assert!(
        miss <= 1e-12 * (1.0 + mode.dot(&mode).sqrt()),
        "the IFT predictor {predictor} is not the tangent predictor {expected} of the solve's \
         penalty (miss {miss:.3e})"
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
            ModeStarts {
                incumbent: state.warm_start_for(&point),
                fixed: &[],
            },
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
        &start.warm_start,
        &array![18.0],
    ) {
        Ok(continuation) => {
            let beta = continuation.inner.block_states[0].beta[0];
            assert!(
                is_deep_mode_2973(18.0, beta),
                "the deep branch continues to rho=18; got beta={beta}"
            );
        }
        Err(refusal) => panic!("the deep branch has no fold, so it must continue: {refusal}"),
    }
}

/// The tilted double well's penalized objective `f(β; ρ) = (β² − 1)² + cβ + ½e^ρβ²`.
fn double_well_penalized_objective_3173(rho: f64, beta: f64) -> f64 {
    let well = beta * beta - 1.0;
    well * well + TILT * beta + 0.5 * rho.exp() * beta * beta
}

/// A start in the deep well, as a fit's fixed seed: coefficients only, at another θ.
fn deep_well_seed_3173() -> crate::assembly::ConstrainedWarmStart {
    crate::assembly::ConstrainedWarmStart {
        rho: array![0.0],
        block_beta: vec![array![-2.0]],
        active_sets: vec![None],
        cached_inner: None,
    }
}

/// A penalized objective of `value` summed from one likelihood row and one penalty entry, with
/// nothing else accumulated: its comparison rounds by `γ₂·(|a| + |b|)`.
fn penalized_3173(value: f64) -> PenalizedObjective {
    PenalizedObjective {
        value,
        likelihood_rows: 1,
        penalty_entries: 1,
        penalty_accumulation: 0.0,
        jeffreys_roundoff: 0.0,
    }
}

/// #3173: the selection's index is the lowest penalized objective among the certified starts,
/// taken in order. A later start replaces the current choice only when it is below by more than
/// the comparison rounds by, so an exact tie and a tie within rounding both keep the earlier start
/// (the incumbent, listed first).
#[test]
fn the_lowest_penalized_start_wins_and_a_tie_keeps_the_earlier_3173() {
    let at = |values: &[Option<f64>]| {
        let penalized: Vec<Option<PenalizedObjective>> = values
            .iter()
            .map(|value| value.map(penalized_3173))
            .collect();
        lowest_penalized_index(&penalized)
    };
    assert_eq!(at(&[Some(2.0), None, Some(1.0)]), Some(2));
    assert_eq!(at(&[Some(1.0), Some(0.5)]), Some(1));
    assert_eq!(
        at(&[Some(1.0), Some(1.0)]),
        Some(0),
        "an exact tie keeps the earlier start"
    );
    // One ulp below 1 lies inside `γ₂·(|a| + |b|) ≈ 4.4e-16`, the comparison's rounding.
    assert_eq!(
        at(&[Some(1.0), Some(1.0 - f64::EPSILON / 2.0)]),
        Some(0),
        "a difference within the comparison's rounding keeps the earlier start"
    );
    assert_eq!(at(&[None, None]), None);
}

/// #3173: an evaluation publishes the certified mode with the lowest penalized objective among
/// its starts.
///
/// At ρ = 0.5 both wells are minima, and the closed form orders them: the deep minimum's `f` is
/// below the shallow one's. From the shallow incumbent alone the evaluation publishes the shallow
/// mode, as the warm start chose it. With the deep well among the fit's fixed starts it publishes
/// the deep mode, and it publishes the deep mode from the deep incumbent too, at one criterion
/// value to its own roundoff, so `V(0.5)` does not depend on which mode the walk carried there.
#[test]
fn the_published_mode_is_the_lowest_penalized_mode_among_the_starts_3173() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let specs = [double_well_spec(2.0)];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
    let rho = array![0.5];
    let evaluate = |starts: ModeStarts<'_>| {
        evaluate_on_branch(
            &family,
            &specs,
            &options,
            &layout,
            &rho,
            starts,
            &gam_problem::RhoPrior::Flat,
            EvalMode::ValueAndGradient,
        )
        .expect("an evaluation at rho=0.5 certifies a mode")
    };
    let shallow = evaluate(ModeStarts {
        incumbent: None,
        fixed: &[],
    });
    let shallow_beta = shallow.warm_start.block_beta[0][0];
    assert!(
        is_shallow_mode_2973(0.5, shallow_beta),
        "the caller's seed at +2 reaches the shallow well; got {shallow_beta}"
    );
    let fixed = [Some(deep_well_seed_3173())];
    let from_shallow = evaluate(ModeStarts {
        incumbent: Some(&shallow.warm_start),
        fixed: &fixed,
    });
    let published = from_shallow.warm_start.block_beta[0][0];
    assert!(
        is_deep_mode_2973(0.5, published),
        "the deep well's mode is the lower of the two, so it is published; got {published}"
    );
    assert!(
        double_well_penalized_objective_3173(0.5, published)
            < double_well_penalized_objective_3173(0.5, shallow_beta),
        "the closed form orders the deep minimum below the shallow one"
    );
    let from_deep = evaluate(ModeStarts {
        incumbent: Some(&from_shallow.warm_start),
        fixed: &fixed,
    });
    assert!(
        is_deep_mode_2973(0.5, from_deep.warm_start.block_beta[0][0]),
        "from the deep incumbent the deep mode is published too"
    );
    let bound = gam_solve::rho_optimizer::outer_value_agreement_bound(
        from_shallow.objective,
        from_deep.objective,
    );
    assert!(
        (from_shallow.objective - from_deep.objective).abs() <= bound,
        "V(0.5) is one value whichever mode the walk carried: {:.17e} against {:.17e} (roundoff \
         bound {bound:.3e})",
        from_shallow.objective,
        from_deep.objective,
    );
}

/// #3173: a branch that ends at its fold hands over to the lowest-f certified rival.
///
/// Past the shallow well's fold at `ρ_fold = ln(4 − 3c^{2/3}) ≈ 0.97666` only the deep minimum
/// exists. From the shallow incumbent at ρ = 0.5 alone, the evaluation at ρ = 1.2 is refused, the
/// #2973 fold refusal. With the deep well among the fit's fixed starts, the shallow branch ends at
/// its fold and the deep minimum is published instead.
#[test]
fn a_branch_that_ends_at_its_fold_hands_over_to_the_lowest_rival_3173() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let specs = [double_well_spec(2.0)];
    let options = double_well_options();
    let penalty_counts: Vec<usize> = specs.iter().map(|spec| spec.penalties.len()).collect();
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("single-penalty label layout");
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
    assert!(
        is_shallow_mode_2973(0.5, start.warm_start.block_beta[0][0]),
        "a seed at +2 starts in the shallow well"
    );
    let evaluate = |fixed: &[Option<crate::assembly::ConstrainedWarmStart>]| {
        evaluate_on_branch(
            &family,
            &specs,
            &options,
            &layout,
            &array![1.2],
            ModeStarts {
                incumbent: Some(&start.warm_start),
                fixed,
            },
            &gam_problem::RhoPrior::Flat,
            EvalMode::ValueAndGradient,
        )
    };
    match evaluate(&[]) {
        Ok(eval) => panic!(
            "with no rival the shallow branch's fold refuses the trial; it published beta={}",
            eval.warm_start.block_beta[0][0]
        ),
        Err(refusal) => assert!(
            refusal.is_trial_point_infeasible(),
            "a branch that ends with no rival is a rejected trial: {refusal}"
        ),
    }
    let handed = evaluate(&[Some(deep_well_seed_3173())])
        .expect("the deep well's mode certifies past the shallow fold");
    let beta = handed.warm_start.block_beta[0][0];
    assert!(
        is_deep_mode_2973(1.2, beta),
        "past the fold the lowest certified rival, the deep minimum, is published; got {beta}"
    );
}

/// #3173: an outer evaluation's starts are the incumbent's and the fit's fixed starts, except at
/// a θ where the rule already published a mode (here a value probe's): the fixed starts would
/// reproduce that selection there, so they are left out, and only there.
#[test]
fn the_fixed_starts_are_left_out_only_where_the_rule_already_published_3173() {
    let mut state =
        CustomOuterState::new(None).with_fixed_starts(vec![Some(deep_well_seed_3173())]);
    let theta = array![0.5];
    assert_eq!(state.mode_starts_for(&theta).fixed.len(), 1);
    let seed = crate::warm_start::SeedIdentity::of(state.seed_for(&theta));
    let published = crate::assembly::ConstrainedWarmStart {
        rho: theta.clone(),
        ..deep_well_seed_3173()
    };
    state.record_value_probe(&theta, seed, published);
    assert!(
        state.mode_starts_for(&theta).fixed.is_empty(),
        "the value probe's published mode at this θ was selected against the same fixed starts"
    );
    assert_eq!(
        state.mode_starts_for(&array![0.6]).fixed.len(),
        1,
        "at any other θ the fixed starts are solved again"
    );
}

/// The same tilted double well, over a family that declares its inner objective globally convex.
///
/// The declaration is false of this objective, and that is what the fixture is for: it is the one
/// input the published-mode rule reads to decide whether an evaluation solves more than one start
/// ([`inner_objective_may_have_several_modes`]), so a control that shows the completion moved the
/// published mode has to drive that input and leave the objective, the specs and the starts alone.
#[derive(Clone)]
struct ConvexDeclaringDoubleWellFamily(TiltedDoubleWellFamily);

impl CustomFamily for ConvexDeclaringDoubleWellFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.0.evaluate(block_states)
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        self.0.exact_newton_joint_hessian_beta_dependent()
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.0.exact_newton_joint_hessian(block_states)
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0
            .exact_newton_joint_hessian_directional_derivative(block_states, direction)
    }

    fn exact_newton_outer_curvature(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        self.0.exact_newton_outer_curvature(block_states)
    }

    fn inner_coefficient_objective_is_globally_convex(&self) -> bool {
        true
    }
}

/// The walk's accepted incumbent, in the shallow well: coefficients certified at another θ, which
/// is what an exact-joint driver's coefficient-mode branch hands its evaluation.
fn shallow_well_incumbent_3173() -> CustomFamilyWarmStart {
    CustomFamilyWarmStart {
        inner: crate::assembly::ConstrainedWarmStart {
            rho: array![0.4],
            block_beta: vec![array![2.0]],
            active_sets: vec![None],
            cached_inner: None,
        },
    }
}

/// #3173: an exact-joint evaluation's starts are the driver's, completed with the fit's fixed
/// start, so the published mode is the lowest-`f` certified mode over a set the walk did not
/// choose.
///
/// The exact-joint drivers hand the evaluator ONE start, their coefficient-mode branch's
/// incumbent. Here that incumbent sits in the shallow well while the fit's blocks are seeded in
/// the deep one. At ρ = 0.5 both wells are minima and the closed form orders them, so the
/// completed evaluation publishes the deep mode.
///
/// The control is the same evaluation, the same specs and the same start, on a family that
/// certifies one mode: it solves the one start it was handed and publishes the shallow
/// incumbent's mode although the blocks are seeded in the deep well. So it is the completion, not
/// the seeding, that moves the published mode. A caller that already carries the fixed start keeps
/// the list it passed: the fixed start is taken once.
#[test]
fn an_exact_joint_evaluation_completes_its_starts_with_the_fits_fixed_start_3173() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let options = double_well_options();
    let rho = array![0.5];
    let deep_seeded = [double_well_spec(-2.0)];
    let layout = || Arc::new(test_design_hyper_layout(vec![Vec::new()]));
    let completed = evaluate_custom_family_joint_hyper_best_mode_shared(
        &family,
        &deep_seeded,
        &options,
        &rho,
        layout(),
        &[Some(shallow_well_incumbent_3173())],
        EvalMode::ValueOnly,
    )
    .expect("an evaluation at rho=0.5 certifies a mode");
    assert_eq!(
        completed.screened_objectives.len(),
        2,
        "the driver's one start is completed with the fit's fixed start"
    );
    let published = completed
        .result
        .warm_start
        .block_beta_view(0)
        .expect("one coefficient")[0];
    assert!(
        is_deep_mode_2973(0.5, published),
        "the deep well's mode is the lower of the two, so it is published; got {published}"
    );

    let convex = ConvexDeclaringDoubleWellFamily(TiltedDoubleWellFamily::new(TILT));
    let one_start = evaluate_custom_family_joint_hyper_best_mode_shared(
        &convex,
        &deep_seeded,
        &options,
        &rho,
        layout(),
        &[Some(shallow_well_incumbent_3173())],
        EvalMode::ValueOnly,
    )
    .expect("a family that certifies one mode still certifies the start it was handed");
    assert_eq!(
        one_start.screened_objectives.len(),
        1,
        "a family whose inner objective has one mode is solved from one start"
    );
    let carried = one_start
        .result
        .warm_start
        .block_beta_view(0)
        .expect("one coefficient")[0];
    assert!(
        is_shallow_mode_2973(0.5, carried),
        "the incumbent alone reaches the shallow well; got {carried}"
    );
    assert!(
        double_well_penalized_objective_3173(0.5, published)
            < double_well_penalized_objective_3173(0.5, carried),
        "the closed form orders the deep minimum below the shallow one"
    );

    let already_carried = evaluate_custom_family_joint_hyper_best_mode_shared(
        &family,
        &deep_seeded,
        &options,
        &rho,
        layout(),
        &[Some(shallow_well_incumbent_3173()), None],
        EvalMode::ValueOnly,
    )
    .expect("an evaluation that already carries the fixed start certifies a mode");
    assert_eq!(
        already_carried.screened_objectives.len(),
        2,
        "the fit's fixed start is taken once"
    );
}

/// The tilted double well's cubic share `5t₃²/(24σ³)` at the mode `beta`, in closed form:
/// `σ = f''(β) = 12β² − 4 + e^ρ` and `t₃ = f'''(β) = 24β`. At or above one the barrier to the
/// neighbouring basin is below the correction the Laplace series makes for it.
fn double_well_cubic_share_3173(rho: f64, beta: f64) -> f64 {
    let sigma = 12.0 * beta * beta - 4.0 + rho.exp();
    let third = 24.0 * beta;
    5.0 * third * third / (24.0 * sigma * sigma * sigma)
}

/// #3173 positive control for the start past the saddle: where the mode the rule publishes among
/// the starts is the one whose barrier is below its own correction, the probe is spent from it and
/// crosses into the lower basin.
///
/// The incumbent and the fit's blocks both sit in the shallow well, so both starts certify the
/// shallow mode and it is the one the rule would publish. At ρ = 0.5 its closed-form share is
/// above one, and its saddle crossing `β̂ + 2s*·v`, `s* = −2σ/t₃`, lands past the barrier, so the
/// probe certifies the deep mode as a third candidate and the rule publishes it. The census test
/// above is the negative arm on the same θ and the same incumbent: there the fixed start publishes
/// the deep mode, whose share is below one, so no probe is spent off the losing shallow incumbent.
#[test]
fn the_start_past_the_saddle_is_spent_from_the_published_mode_3173() {
    let family = TiltedDoubleWellFamily::new(TILT);
    let options = double_well_options();
    let rho = 0.5;
    let points = double_well_stationary_points_2973(rho);
    assert_eq!(points.len(), 3, "both wells are minima at rho=0.5");
    let (deep, shallow) = (points[0], points[2]);
    assert!(
        double_well_cubic_share_3173(rho, shallow) >= 1.0,
        "the shallow mode's barrier is below its own correction at rho=0.5"
    );
    assert!(
        double_well_cubic_share_3173(rho, deep) < 1.0,
        "the deep mode's barrier is not, so the census arm spends no probe off it"
    );
    let shallow_seeded = [double_well_spec(2.0)];
    let selection = evaluate_custom_family_joint_hyper_best_mode_shared(
        &family,
        &shallow_seeded,
        &options,
        &array![rho],
        Arc::new(test_design_hyper_layout(vec![Vec::new()])),
        &[Some(shallow_well_incumbent_3173())],
        EvalMode::ValueOnly,
    )
    .expect("an evaluation at rho=0.5 certifies a mode");
    assert_eq!(
        selection.screened_objectives.len(),
        3,
        "the two starts, and the probe past the published shallow mode's saddle"
    );
    assert_eq!(
        selection.selected_candidate, 2,
        "the probe's mode is the lowest certified one, so it is published"
    );
    let published = selection
        .result
        .warm_start
        .block_beta_view(0)
        .expect("one coefficient")[0];
    assert!(
        is_deep_mode_2973(rho, published),
        "the probe crossed into the deep well; got {published}"
    );
    assert!(
        double_well_penalized_objective_3173(rho, deep)
            < double_well_penalized_objective_3173(rho, shallow),
        "the closed form orders the deep minimum below the shallow one"
    );
}
