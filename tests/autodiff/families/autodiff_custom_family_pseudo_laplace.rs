use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix,
};
use gam::families::custom_family::{
    CustomFamilyBlockPsiDerivative, ExactNewtonOuterObjective, evaluate_custom_family_joint_hyper,
};
use gam::matrix::SymmetricMatrix;
use gam_problem::ExactNewtonJointPsiTerms;
use ndarray::{Array1, Array2, array};
use num_dual::{DualNum, first_derivative};

#[derive(Clone)]
struct ScalarPseudoLaplaceRhoFamily {
    target: f64,
}

impl CustomFamily for ScalarPseudoLaplaceRhoFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta[0];
        let resid = beta - self.target;
        Ok(FamilyEvaluation {
            log_likelihood: -resid * resid,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-2.0 * resid],
                hessian: SymmetricMatrix::Dense(array![[2.0]]),
            }],
        })
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(!block_states.is_empty(), "rho joint hessian needs blocks");
        Ok(Some(array![[2.0]]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_index < block_states.len(), "rho block index in range");
        assert_eq!(
            direction.len(),
            block_states[block_index].beta.len(),
            "rho dir len matches beta"
        );
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let total: usize = block_states.iter().map(|s| s.beta.len()).sum();
        assert_eq!(direction.len(), total, "rho joint dir matches total beta");
        Ok(Some(array![[0.0]]))
    }
}

#[derive(Clone)]
struct ScalarPseudoLaplacePsiFamily {
    psi: f64,
}

impl CustomFamily for ScalarPseudoLaplacePsiFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta[0];
        let resid = beta - self.psi;
        Ok(FamilyEvaluation {
            log_likelihood: -(resid * resid + 0.25 * self.psi * self.psi),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-2.0 * resid],
                hessian: SymmetricMatrix::Dense(array![[2.0]]),
            }],
        })
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_index < block_states.len(), "psi block index in range");
        assert_eq!(
            direction.len(),
            block_states[block_index].beta.len(),
            "psi dir len matches beta"
        );
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(!block_states.is_empty(), "psi joint hessian needs blocks");
        Ok(Some(array![[2.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let total: usize = block_states.iter().map(|s| s.beta.len()).sum();
        assert_eq!(direction.len(), total, "psi joint dir matches total beta");
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        assert_eq!(
            block_states.len(),
            block_specs.len(),
            "psi terms: states/specs aligned"
        );
        assert_eq!(
            derivative_blocks.len(),
            block_states.len(),
            "psi terms: derivs/states aligned"
        );
        assert_eq!(psi_index, 0, "psi terms: scalar psi index expected 0");
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta[0];
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: -2.0 * (beta - self.psi) + 0.5 * self.psi,
            score_psi: array![0.0],
            hessian_psi: array![[0.0]],
            hessian_psi_operator: None,
        }))
    }
}

fn scalar_pseudo_laplace_rhoobjective_numdual<D: DualNum<f64> + Copy>(rho: D, target: f64) -> D {
    let lambda = rho.exp();
    let beta_hat = D::from(2.0 * target) / (D::from(2.0) + lambda);
    let resid = beta_hat - D::from(target);
    resid * resid
        + D::from(0.5) * lambda * beta_hat * beta_hat
        + D::from(0.5) * (D::from(2.0) + lambda).ln()
}

fn scalar_pseudo_laplace_psiobjective_numdual<D: DualNum<f64> + Copy>(psi: D) -> D {
    D::from(0.25) * psi * psi + D::from(0.5) * D::from(2.0).ln()
}

fn scalar_pseudo_laplace_rhoobjective_f64(rho: f64, target: f64) -> f64 {
    let lambda = rho.exp();
    let beta_hat = 2.0 * target / (2.0 + lambda);
    let resid = beta_hat - target;
    resid * resid + 0.5 * lambda * beta_hat * beta_hat + 0.5 * (2.0 + lambda).ln()
}

fn scalar_pseudo_laplace_psiobjective_f64(psi: f64) -> f64 {
    0.25 * psi * psi + 0.5 * 2.0_f64.ln()
}

#[test]
fn exact_newton_pseudo_laplace_rhogradient_matches_num_dual_band() {
    let spec = ParameterBlockSpec {
        name: "rho_block".to_string(),
        design: gam::matrix::DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new()];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };
    let family = ScalarPseudoLaplaceRhoFamily { target: 1.4 };
    let rho_points = [-1.7, -0.8, 0.0, 0.9, 1.8];

    for rho in rho_points {
        let result = evaluate_custom_family_joint_hyper(
            &family,
            std::slice::from_ref(&spec),
            &options,
            &array![rho],
            &derivative_blocks,
            None,
            gam::families::custom_family::EvalMode::ValueAndGradient,
        )
        .expect("pseudo-laplace rho hyper eval");
        let (value_nd, grad_nd) = first_derivative(
            |x| scalar_pseudo_laplace_rhoobjective_numdual(x, family.target),
            rho,
        );
        let value_f64 = scalar_pseudo_laplace_rhoobjective_f64(rho, family.target);
        let h = 1e-6;
        let gradfd = (scalar_pseudo_laplace_rhoobjective_f64(rho + h, family.target)
            - scalar_pseudo_laplace_rhoobjective_f64(rho - h, family.target))
            / (2.0 * h);
        assert!(
            (result.objective - value_nd).abs() < 5e-12,
            "pseudo_laplace_rho x={rho:.6} objective mismatch: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            value_f64
        );
        assert_manual_ad_band!(
            "pseudo_laplace_rho",
            rho,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "fd" => gradfd
        );
    }
}

#[test]
fn exact_newton_pseudo_laplace_psigradient_matches_num_dual_band() {
    let spec = ParameterBlockSpec {
        name: "psi_block".to_string(),
        design: gam::matrix::DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let deriv = CustomFamilyBlockPsiDerivative::new(
        Some(0),
        Array2::zeros((1, 1)),
        Array2::zeros((1, 1)),
        Some(Vec::new()),
        None,
        None,
        None,
    );
    let derivative_blocks = vec![vec![deriv]];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };
    let psi_points = [-2.0, -0.7, 0.0, 0.5, 1.6];

    for psi in psi_points {
        let family = ScalarPseudoLaplacePsiFamily { psi };
        let result = evaluate_custom_family_joint_hyper(
            &family,
            std::slice::from_ref(&spec),
            &options,
            &Array1::zeros(0),
            &derivative_blocks,
            None,
            gam::families::custom_family::EvalMode::ValueAndGradient,
        )
        .expect("pseudo-laplace psi hyper eval");
        let (value_nd, grad_nd) = first_derivative(scalar_pseudo_laplace_psiobjective_numdual, psi);
        let value_f64 = scalar_pseudo_laplace_psiobjective_f64(psi);
        let h = 1e-6;
        let gradfd = (scalar_pseudo_laplace_psiobjective_f64(psi + h)
            - scalar_pseudo_laplace_psiobjective_f64(psi - h))
            / (2.0 * h);
        assert!(
            (result.objective - value_nd).abs() < 5e-12,
            "pseudo_laplace_psi x={psi:.6} objective mismatch: analytic={} num_dual={} closed_form={}",
            result.objective,
            value_nd,
            value_f64
        );
        assert_manual_ad_band!(
            "pseudo_laplace_psi",
            psi,
            "gradient",
            result.gradient[0],
            "num_dual" => grad_nd,
            "fd" => gradfd
        );
    }
}

/// Regression guard for gam#1395: the analytic pseudo-Laplace objective must
/// keep the FULL `0.5·log|H|` Laplace term and not let it collapse.
///
/// The reported #1395 failure was a value disagreement
/// (`analytic=1.004… vs closed_form=1.346…` at psi=-2.0): the `0.5·log|H|`
/// term (`0.5·ln 2 = 0.346574…`) silently collapsed to ~0.0044 — an effective
/// Hessian eigenvalue of ~1.0088 instead of the true H=[[2]]. The
/// `*_matches_num_dual_band` test above already pins `result.objective` to the
/// num_dual closed form, but it routes through a Rust helper; this test pins
/// the objective to HARD-CODED constants derived purely from the math
/// (`objective(psi) = 0.25·psi² + 0.5·ln 2`), so the guard survives even if the
/// helper were ever changed, and it isolates the log|H| term directly:
///
/// * At `psi = 0` the quadratic term vanishes, so the objective is EXACTLY the
///   log-determinant term `0.5·ln 2`. A collapse to ~0.0044 (the #1395 symptom)
///   would fail this assertion outright.
/// * At `psi = -2.0` (the exact point the issue reported) the objective is
///   `0.25·4 + 0.5·ln 2 = 1.0 + 0.5·ln 2 = 1.346573…`.
#[test]
fn pseudo_laplace_objective_keeps_full_logdet_term_1395() {
    let half_ln2 = 0.5 * std::f64::consts::LN_2;

    let spec = ParameterBlockSpec {
        name: "psi_block".to_string(),
        design: gam::matrix::DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(array![[
            1.0
        ]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let deriv = CustomFamilyBlockPsiDerivative::new(
        Some(0),
        Array2::zeros((1, 1)),
        Array2::zeros((1, 1)),
        Some(Vec::new()),
        None,
        None,
        None,
    );
    let derivative_blocks = vec![vec![deriv]];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        ..BlockwiseFitOptions::default()
    };

    // (psi, expected objective = 0.25·psi² + 0.5·ln 2)
    let cases = [
        (0.0_f64, half_ln2),
        (-2.0_f64, 1.0 + half_ln2),
        (1.6_f64, 0.25 * 1.6 * 1.6 + half_ln2),
    ];

    for (psi, expected) in cases {
        let family = ScalarPseudoLaplacePsiFamily { psi };
        let result = evaluate_custom_family_joint_hyper(
            &family,
            std::slice::from_ref(&spec),
            &options,
            &Array1::zeros(0),
            &derivative_blocks,
            None,
            gam::families::custom_family::EvalMode::ValueAndGradient,
        )
        .expect("pseudo-laplace psi hyper eval");

        assert!(
            (result.objective - expected).abs() < 1e-10,
            "gam#1395 pseudo_laplace psi={psi}: objective={} expected={} \
             (0.5·log|H| term must NOT collapse below 0.5·ln2={:.6})",
            result.objective,
            expected,
            half_ln2
        );
    }
}
