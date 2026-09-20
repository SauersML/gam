#![cfg(test)]
//! #979 Jeffreys ruling (b), pin 1: a clean fit through the arming lifecycle IS the
//! unarmed objective's fit. The lifecycle hands back the unarmed member's certified fit
//! unchanged and publishes no arming evidence. The negative control shows that the armed
//! member prices a measurably different objective at the same smoothing strength, so the
//! equality can fail.
//!
//! The route-level lifecycle `arm_on_evidence` (gam#2994 / gam#2995): a refusal carrying
//! typed evidence refits the armed member exactly once, from no certified mode, and that
//! refit publishes the evidence; a refusal without evidence, and a failed armed refit, reach
//! the caller unchanged.

use super::*;
use crate::test_support::outerobjectivegradienthessian_labeled;
use gam_solve::model_types::UnifiedFitResult;
use ndarray::array;

/// The one-coefficient quartic likelihood the #2898 mint pins certify, with an arming switch.
#[derive(Clone)]
struct ArmableQuarticFamily {
    linear: f64,
    curvature: f64,
    armed: bool,
}

impl JeffreysArming for ArmableQuarticFamily {
    fn with_jeffreys_armed(
        &self,
        evidence: Option<&gam_problem::jeffreys_arming::JeffreysArmingEvidence>,
    ) -> Self {
        Self {
            armed: evidence.is_some(),
            ..self.clone()
        }
    }
}

impl CustomFamily for ArmableQuarticFamily {
    fn joint_jeffreys_term_required(&self) -> bool {
        self.armed
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states[0].beta[0];
        Ok(FamilyEvaluation {
            log_likelihood: self.linear * beta
                - 0.5 * beta * beta
                - self.curvature * beta.powi(4) / 12.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![self.linear - beta - self.curvature * beta.powi(3) / 3.0],
                hessian: SymmetricMatrix::Dense(array![[1.0 + self.curvature * beta * beta]]),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = block_states[0].beta[0];
        Ok(Some(array![[1.0 + self.curvature * beta * beta]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = block_states[0].beta[0];
        Ok(Some(array![[2.0 * self.curvature * beta * d_beta_flat[0]]]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0, "the quartic family has one block");
        let beta = block_states[0].beta[0];
        Ok(Some(array![[2.0 * self.curvature * beta * direction[0]]]))
    }

    fn exact_newton_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0, "the quartic family has one block");
        assert!(
            block_states[0].beta.iter().all(|value| value.is_finite()),
            "the quartic second directional derivative owes a finite mode"
        );
        Ok(Some(array![[2.0 * self.curvature * u[0] * v[0]]]))
    }
}

fn quartic_family() -> ArmableQuarticFamily {
    ArmableQuarticFamily {
        linear: 3.0,
        curvature: 0.5,
        armed: true,
    }
}

fn quartic_specs() -> Vec<ParameterBlockSpec> {
    vec![ParameterBlockSpec {
        name: "quartic".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.75]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }]
}

fn quartic_options() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_tol: 1e-11,
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    }
}

fn saddle_evidence() -> gam_problem::jeffreys_arming::JeffreysArmingEvidence {
    gam_problem::jeffreys_arming::JeffreysArmingEvidence::StrictSaddle {
        stationarity_residual: 0.0,
    }
}

#[test]
fn a_clean_fit_through_the_arming_lifecycle_is_the_unarmed_fit_979() {
    let family = quartic_family();
    let specs = quartic_specs();
    let options = quartic_options();

    let lifecycle = fit_custom_family_arming_on_evidence(&family, &specs, &options)
        .expect("the clean quartic fit must certify through the arming lifecycle");
    let plain = fit_custom_family(&family.with_jeffreys_armed(None), &specs, &options)
        .expect("the unarmed quartic fit must certify");

    let penalty_counts = validate_blockspecs(&specs).expect("valid quartic spec");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid label layout");
    let value_at = |member: &ArmableQuarticFamily| {
        outerobjectivegradienthessian_labeled(
            member,
            &specs,
            &options,
            &layout,
            &array![0.0],
            None,
            &gam_problem::RhoPrior::Flat,
            EvalMode::ValueOnly,
        )
        .expect("the quartic outer value must evaluate")
        .objective
    };
    let unarmed_value = value_at(&family.with_jeffreys_armed(None));
    let armed_value = value_at(&family.with_jeffreys_armed(Some(&saddle_evidence())));
    println!(
        "[#979 pin 1] lifecycle beta={:?} plain beta={:?} evidence={:?} \
         lifecycle objective={:?} plain objective={:?} unarmed_value={unarmed_value:.12e} \
         armed_value={armed_value:.12e}",
        lifecycle.blocks[0].beta,
        plain.blocks[0].beta,
        lifecycle.artifacts.jeffreys_arming_evidence,
        lifecycle.penalized_objective(),
        plain.penalized_objective(),
    );

    assert!(
        lifecycle.artifacts.jeffreys_arming_evidence.is_none(),
        "a clean fit must publish no arming evidence"
    );
    assert_eq!(
        lifecycle.blocks[0].beta[0].to_bits(),
        plain.blocks[0].beta[0].to_bits(),
        "a clean fit through the lifecycle must be the unarmed member's fit, bit for bit"
    );
    assert_eq!(
        lifecycle.penalized_objective().map(f64::to_bits),
        plain.penalized_objective().map(f64::to_bits),
        "a clean fit through the lifecycle must price the unarmed objective, bit for bit"
    );
    // Negative control: the armed member prices a different objective at the same rho, so an
    // armed fit could not pass the equalities above by accident.
    assert!(
        (armed_value - unarmed_value).abs() > 1e-6 * (1.0 + unarmed_value.abs()),
        "the armed member must price a measurably different objective: armed {armed_value:e}, \
         unarmed {unarmed_value:e}"
    );
}

#[test]
fn a_refusal_with_evidence_refits_armed_once_and_publishes_it_2995() {
    let family = quartic_family();
    let specs = quartic_specs();
    let options = quartic_options();
    let unarmed = fit_custom_family(&family.with_jeffreys_armed(None), &specs, &options)
        .expect("the unarmed quartic fit must certify");
    let mut armed_calls = 0usize;
    let refit = arm_on_evidence(
        Err::<UnifiedFitResult, String>("the unarmed route refused".to_string()),
        |fit| fit,
        |_| Some(saddle_evidence()),
        |evidence, certified| {
            armed_calls += 1;
            assert_eq!(evidence, &saddle_evidence(), "the refit must receive the evidence");
            assert!(certified.is_none(), "a refusal leaves no certified mode to start from");
            fit_custom_family(&family.with_jeffreys_armed(Some(evidence)), &specs, &options)
                .map_err(|error| error.to_string())
        },
    )
    .expect("the armed quartic refit must certify");
    println!(
        "[#2995 arming] armed beta={:?} unarmed beta={:?} evidence={:?}",
        refit.blocks[0].beta, unarmed.blocks[0].beta, refit.artifacts.jeffreys_arming_evidence,
    );
    assert_eq!(armed_calls, 1, "evidence arms exactly one refit");
    assert_eq!(
        refit.artifacts.jeffreys_arming_evidence,
        Some(saddle_evidence()),
        "the armed refit must publish the evidence it armed on"
    );
    assert!(
        (refit.blocks[0].beta[0] - unarmed.blocks[0].beta[0]).abs() > 1e-8,
        "the refit must solve the armed objective, not return the unarmed mode: armed {}, \
         unarmed {}",
        refit.blocks[0].beta[0],
        unarmed.blocks[0].beta[0],
    );
}

#[test]
fn a_refusal_without_evidence_is_returned_unchanged_2995() {
    let outcome = arm_on_evidence(
        Err::<UnifiedFitResult, String>("the unarmed route refused".to_string()),
        |fit| fit,
        |_| None,
        |_, _| panic!("a refusal without evidence must not arm"),
    );
    assert_eq!(
        outcome.err().as_deref(),
        Some("the unarmed route refused"),
        "a refusal without evidence reaches the caller unchanged"
    );
}

#[test]
fn a_failed_armed_refit_reaches_the_caller_2995() {
    let outcome = arm_on_evidence(
        Err::<UnifiedFitResult, String>("the unarmed route refused".to_string()),
        |fit| fit,
        |_| Some(saddle_evidence()),
        |_, _| Err("the armed route refused".to_string()),
    );
    assert_eq!(
        outcome.err().as_deref(),
        Some("the armed route refused"),
        "the armed refit's refusal is the outcome, not the unarmed one"
    );
}

/// Two coupled quartic coefficients, one block each, so the solve takes the joint path. The
/// family has no contracted-trace completion hook: its armed Jeffreys completion is assembled
/// from the second directional information derivative.
#[derive(Clone)]
struct CoupledQuarticPairFamily {
    linear: [f64; 2],
    curvature: f64,
    coupling: f64,
}

impl CoupledQuarticPairFamily {
    fn beta(block_states: &[ParameterBlockState]) -> [f64; 2] {
        [block_states[0].beta[0], block_states[1].beta[0]]
    }

    fn joint_hessian(&self, beta: [f64; 2]) -> Array2<f64> {
        array![
            [1.0 + self.curvature * beta[0] * beta[0], -self.coupling],
            [-self.coupling, 1.0 + self.curvature * beta[1] * beta[1]],
        ]
    }
}

impl CustomFamily for CoupledQuarticPairFamily {
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = Self::beta(block_states);
        let quartic = |j: usize| {
            self.linear[j] * beta[j] - 0.5 * beta[j] * beta[j] - self.curvature * beta[j].powi(4) / 12.0
        };
        let working_set = |j: usize| BlockWorkingSet::ExactNewton {
            gradient: array![
                self.linear[j] - beta[j] - self.curvature * beta[j].powi(3) / 3.0
                    + self.coupling * beta[1 - j]
            ],
            hessian: SymmetricMatrix::Dense(array![[1.0 + self.curvature * beta[j] * beta[j]]]),
        };
        Ok(FamilyEvaluation {
            log_likelihood: quartic(0) + quartic(1) + self.coupling * beta[0] * beta[1],
            blockworking_sets: vec![working_set(0), working_set(1)],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.joint_hessian(Self::beta(block_states))))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states);
        let c = 2.0 * self.curvature;
        Ok(Some(array![
            [c * beta[0] * d_beta_flat[0], 0.0],
            [0.0, c * beta[1] * d_beta_flat[1]],
        ]))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let c = 2.0 * self.curvature;
        Ok(Some(array![
            [c * d_beta_u_flat[0] * d_betav_flat[0], 0.0],
            [0.0, c * d_beta_u_flat[1] * d_betav_flat[1]],
        ]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = Self::beta(block_states);
        Ok(Some(array![[2.0 * self.curvature * beta[block_idx] * direction[0]]]))
    }

    fn exact_newton_hessian_second_directional_derivative(
        &self,
        _block_states: &[ParameterBlockState],
        _block_idx: usize,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[2.0 * self.curvature * u[0] * v[0]]]))
    }
}

/// #3444: the armed outer gradient of a family without a contracted-trace completion is the
/// derivative of its value. The mode is stationary for the Φ-augmented objective, whose Hessian
/// is `M_true = H + S_λ + H_Φ + completion`, so the mode response must be solved on `M_true`;
/// solving it on `M_DD = H + S_λ + H_Φ` left a gap of about 1e-3 to 6e-3 against a central
/// difference. The central difference's own error at `h = 1e-4` is `h²/6·|V'''|` (~1e-9 here)
/// plus the value's resolution over `h` (inner tolerance 1e-11, so ~1e-7), which the 1e-6
/// bound clears with margin while sitting three decades below the defect.
#[test]
fn armed_outer_gradient_without_a_contracted_completion_is_the_value_derivative_3444() {
    let family = CoupledQuarticPairFamily {
        linear: [3.0, 2.0],
        curvature: 0.5,
        coupling: 0.3,
    };
    let block = |name: &str| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.75]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![block("first"), block("second")];
    let options = quartic_options();
    let penalty_counts = validate_blockspecs(&specs).expect("valid coupled quartic specs");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid label layout");
    let evaluate = |rho: &Array1<f64>, mode: EvalMode| {
        outerobjectivegradienthessian_labeled(
            &family,
            &specs,
            &options,
            &layout,
            rho,
            None,
            &gam_problem::RhoPrior::Flat,
            mode,
        )
        .expect("the armed coupled quartic outer criterion must evaluate")
    };
    let h = 1e-4;
    for point in [[0.0, 0.0], [1.0, 1.0], [-1.2, 0.0], [0.0, -1.2]] {
        let rho = array![point[0], point[1]];
        let analytic = evaluate(&rho, EvalMode::ValueAndGradient).gradient;
        for axis in 0..2 {
            let mut plus = rho.clone();
            plus[axis] += h;
            let mut minus = rho.clone();
            minus[axis] -= h;
            let central = (evaluate(&plus, EvalMode::ValueOnly).objective
                - evaluate(&minus, EvalMode::ValueOnly).objective)
                / (2.0 * h);
            let gap = (analytic[axis] - central).abs();
            println!(
                "[#3444] rho={point:?} axis={axis} analytic={:.9} central={central:.9} gap={gap:.3e}",
                analytic[axis]
            );
            assert!(
                gap <= 1e-6 * (1.0 + central.abs()),
                "armed outer gradient at rho={point:?}, axis {axis}: analytic {} vs central \
                 difference {central} (gap {gap:e})",
                analytic[axis]
            );
        }
    }
}
