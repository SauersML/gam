//! Analytic active-face adjoint for constrained Gaussian REML.
//!
//! At a locally constant active set, `A_a beta = b_a`, every coefficient
//! perturbation lies in `null(A_a)`.  If `Z` is an orthonormal basis of that
//! tangent space and
//!
//! ```text
//! H = X' W X + lambda S,
//! P = Z (Z' H Z)^-1 Z',
//! Q = Z (Z' S Z)^+ Z',
//! ```
//!
//! then `P` is the response kernel for the constrained KKT system and `Q` is
//! the penalty pseudo-inverse on the same face.  Crucially, the formulas below
//! retain the *full affine coefficient* `beta`; replacing it by `Z' beta`
//! loses the penalty cross and constant terms whenever `b_a != 0`.

use crate::active_set::{
    feasible_point_for_linear_constraints, solve_quadratic_with_linear_constraints,
};
use crate::estimate::EstimationError;
use crate::gaussian_reml::{
    GaussianRemlBackwardResult, gaussian_reml_multi_closed_form_with_cache,
};
use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerCholesky, FaerEigh, default_rrqr_rank_alpha, rrqr_nullspace_basis, rrqr_with_permutation,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use opt::{Bfgs, Bounds, FirstOrderSample, FusedObjective, GradientTolerance, ObjectiveEvalError};

/// Inputs to the single-penalty constrained Gaussian REML fit.
///
/// The public operation defines one criterion in both regimes: ordinary
/// closed-form Gaussian REML when the unconstrained optimum is interior, and
/// the exact affine-face restriction of that same criterion when constraints
/// bind.  There is deliberately no generic-GAM rho prior, shrinkage floor, or
/// ALO correction hidden in this low-level primitive.
pub struct ConstrainedGaussianRemlForwardProblem<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
    pub penalty: ArrayView2<'a, f64>,
    pub weights: Option<ArrayView1<'a, f64>>,
    pub constraints: Option<&'a LinearInequalityConstraints>,
    pub init_lambda: Option<f64>,
}

/// Accepted state of the constrained Gaussian REML fit.
pub struct ConstrainedGaussianRemlForwardResult {
    pub lambda: f64,
    pub coefficients: Array2<f64>,
    pub fitted: Array2<f64>,
    pub reml_score: f64,
    pub edf: f64,
    pub active_indices: Array1<u64>,
}

/// Inputs to the constrained Gaussian REML active-face VJP.
///
/// Constraint geometry is deliberately non-differentiable.  `coefficients`
/// and `lambda` are the accepted forward state; unlike the old reduced helper,
/// this routine never launches a second smoothing optimization in backward.
pub struct ConstrainedGaussianRemlBackwardProblem<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
    pub penalty: ArrayView2<'a, f64>,
    pub weights: Option<ArrayView1<'a, f64>>,
    pub a_inequality: ArrayView2<'a, f64>,
    pub b_inequality: ArrayView1<'a, f64>,
    pub active_indices: ArrayView1<'a, u64>,
    pub lambda: f64,
    pub coefficients: ArrayView2<'a, f64>,
    pub grad_coefficients: Option<ArrayView2<'a, f64>>,
    pub grad_fitted: Option<ArrayView2<'a, f64>>,
    pub grad_lambda: f64,
    pub grad_log_lambda: f64,
    pub grad_reml_score: f64,
    pub grad_edf: f64,
}

#[derive(Clone)]
struct ActiveFace {
    a: Array2<f64>,
    b: Array1<f64>,
    z: Array2<f64>,
}

#[derive(Clone)]
struct AffineFaceProfile {
    x: Array2<f64>,
    y: Array2<f64>,
    penalty: Array2<f64>,
    weights: Array1<f64>,
    face: ActiveFace,
    beta_particular: Array2<f64>,
    tangent_gram: Array2<f64>,
    tangent_penalty: Array2<f64>,
    tangent_rhs_data: Array2<f64>,
    tangent_penalty_particular: Array2<f64>,
    penalty_rank: usize,
    penalty_logdet: f64,
    residual_df: f64,
}

struct AffineFaceEvaluation {
    rho: f64,
    lambda: f64,
    score: f64,
    rho_gradient: f64,
    rho_curvature: f64,
    edf: f64,
    beta: Array2<f64>,
    fitted: Array2<f64>,
}

struct TangentPenaltyGeometry {
    rank: usize,
    logdet: f64,
    pseudoinverse: Array2<f64>,
}

struct FaceState {
    penalty: Array2<f64>,
    weights: Array1<f64>,
    beta: Array2<f64>,
    residual: Array2<f64>,
    gram: Array2<f64>,
    p_response: Array2<f64>,
    q_penalty: Array2<f64>,
    penalty_rank: usize,
    residual_df: f64,
}

struct VjpAccumulator {
    x: Array2<f64>,
    y: Array2<f64>,
    penalty: Array2<f64>,
    weights: Array1<f64>,
    lambda: f64,
}

impl VjpAccumulator {
    fn zeros(n: usize, p: usize, d: usize) -> Self {
        Self {
            x: Array2::zeros((n, p)),
            y: Array2::zeros((n, d)),
            penalty: Array2::zeros((p, p)),
            weights: Array1::zeros(n),
            lambda: 0.0,
        }
    }
}

/// Fit the pure single-penalty Gaussian REML criterion on the accepted affine
/// KKT face.  The active face and smoothing parameter are alternated until the
/// constrained quadratic solve returns the same face optimized by the scalar
/// REML step.  Because there are finitely many faces, a repeated non-fixed face
/// is a typed optimization failure rather than a best-effort fit.
pub fn constrained_gaussian_reml_forward(
    problem: ConstrainedGaussianRemlForwardProblem<'_>,
) -> Result<ConstrainedGaussianRemlForwardResult, EstimationError> {
    validate_forward_problem(&problem)?;
    let canonical_constraints = problem
        .constraints
        .map(LinearInequalityConstraints::canonicalized)
        .transpose()
        .map_err(|message| EstimationError::InvalidInput(message))?;

    let unconstrained = gaussian_reml_multi_closed_form_with_cache(
        problem.x,
        problem.y,
        problem.penalty,
        problem.weights,
        problem.init_lambda,
        None,
    )?;
    let Some(constraints) = canonical_constraints.as_ref() else {
        return Ok(unconstrained_result(unconstrained));
    };
    if constraints.a.nrows() == 0 {
        return Ok(unconstrained_result(unconstrained));
    }
    let unconstrained_active = binding_rows(constraints, unconstrained.coefficients.column(0));
    if unconstrained_active.is_empty() {
        return Ok(unconstrained_result(unconstrained));
    }

    let n = problem.x.nrows();
    let p = problem.x.ncols();
    let penalty = symmetric_average(problem.penalty);
    let weights = problem
        .weights
        .map_or_else(|| Array1::ones(n), |values| values.to_owned());
    let weighted_x = &problem.x.to_owned() * &weights.view().insert_axis(Axis(1));
    let gram = problem.x.t().dot(&weighted_x);
    let weighted_y = &problem.y.to_owned() * &weights.view().insert_axis(Axis(1));
    let rhs = problem.x.t().dot(&weighted_y).column(0).to_owned();
    let mut beta_start =
        feasible_point_for_linear_constraints(constraints, p).ok_or_else(|| {
            EstimationError::ParameterConstraintViolation(
                "constrained Gaussian REML could not construct a feasible coefficient seed"
                    .to_string(),
            )
        })?;
    let mut rho = unconstrained.rho;
    let mut active_hint: Vec<usize> = unconstrained_active
        .iter()
        .map(|&index| index as usize)
        .collect();
    let mut visited_faces: Vec<Vec<u64>> = Vec::new();

    loop {
        let lambda = gam_problem::checked_exp_log_strength(rho)
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
        let hessian = &gram + &(penalty.clone() * lambda);
        let (qp_beta, qp_active) = solve_quadratic_with_linear_constraints(
            &hessian,
            &rhs,
            &beta_start,
            constraints,
            Some(&active_hint),
        )?;
        let active = binding_rows(constraints, qp_beta.view());
        if active.is_empty() {
            return Err(EstimationError::GradientUnavailable {
                context: "constrained Gaussian REML forward",
                mode: "active-face selection reached an unresolved constraint transition",
            });
        }
        let active_key = active.to_vec();
        if visited_faces.iter().any(|seen| seen == &active_key) {
            return Err(EstimationError::GradientUnavailable {
                context: "constrained Gaussian REML forward",
                mode: "active-face smoothing iteration cycles at a constraint transition",
            });
        }
        visited_faces.push(active_key);

        let profile = AffineFaceProfile::new(
            problem.x,
            problem.y,
            penalty.view(),
            weights.view(),
            constraints,
            active.view(),
        )?;
        let accepted = optimize_affine_face(&profile, rho)?;

        let accepted_hessian = &gram + &(penalty.clone() * accepted.lambda);
        let accepted_beta = accepted.beta.column(0).to_owned();
        let (qp_check, next_hint) = solve_quadratic_with_linear_constraints(
            &accepted_hessian,
            &rhs,
            &accepted_beta,
            constraints,
            Some(&qp_active),
        )?;
        let next_active = binding_rows(constraints, qp_check.view());
        if next_active == active {
            let beta_scale = accepted_beta
                .iter()
                .chain(qp_check.iter())
                .fold(1.0_f64, |scale, &value| scale.max(value.abs()));
            let agreement = accepted_beta
                .iter()
                .zip(qp_check.iter())
                .fold(0.0_f64, |maximum, (&left, &right)| {
                    maximum.max((left - right).abs())
                });
            let resolution = crate::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL * beta_scale;
            if agreement > resolution {
                return Err(EstimationError::GradientUnavailable {
                    context: "constrained Gaussian REML forward",
                    mode: "affine-face optimum does not agree with its constrained KKT solve",
                });
            }
            return Ok(ConstrainedGaussianRemlForwardResult {
                lambda: accepted.lambda,
                coefficients: accepted.beta,
                fitted: accepted.fitted,
                reml_score: accepted.score,
                edf: accepted.edf,
                active_indices: active,
            });
        }

        beta_start = qp_check;
        rho = accepted.rho;
        active_hint = next_hint;
    }
}

fn validate_forward_problem(
    problem: &ConstrainedGaussianRemlForwardProblem<'_>,
) -> Result<(), EstimationError> {
    let n = problem.x.nrows();
    let p = problem.x.ncols();
    if problem.y.dim() != (n, 1) || problem.penalty.dim() != (p, p) {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML forward input shapes are inconsistent"
        );
    }
    if let Some(weights) = problem.weights
        && weights.len() != n
    {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML weights length {} does not match row count {n}",
            weights.len()
        );
    }
    if let Some(constraints) = problem.constraints
        && (constraints.a.ncols() != p || constraints.b.len() != constraints.a.nrows())
    {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML constraint dimensions are inconsistent"
        );
    }
    if problem
        .x
        .iter()
        .chain(problem.y.iter())
        .chain(problem.penalty.iter())
        .any(|value| !value.is_finite())
    {
        crate::bail_invalid_estim!("constrained Gaussian REML inputs must be finite");
    }
    if let Some(weights) = problem.weights
        && weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML weights must be finite and non-negative"
        );
    }
    Ok(())
}

fn unconstrained_result(
    result: crate::gaussian_reml::GaussianRemlMultiResult,
) -> ConstrainedGaussianRemlForwardResult {
    ConstrainedGaussianRemlForwardResult {
        lambda: result.lambda,
        coefficients: result.coefficients,
        fitted: result.fitted,
        reml_score: result.reml_score,
        edf: result.edf,
        active_indices: Array1::zeros(0),
    }
}

fn binding_rows(
    constraints: &LinearInequalityConstraints,
    beta: ArrayView1<'_, f64>,
) -> Array1<u64> {
    let beta_scale = beta
        .iter()
        .fold(1.0_f64, |scale, &value| scale.max(value.abs()));
    let mut active = Vec::new();
    for row in 0..constraints.a.nrows() {
        let normal = constraints.a.row(row);
        if normal.iter().all(|&value| value == 0.0) {
            continue;
        }
        let slack = normal.dot(&beta) - constraints.b[row];
        let resolution = crate::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
            * beta_scale.max(constraints.b[row].abs().max(1.0));
        if slack <= resolution {
            active.push(row as u64);
        }
    }
    Array1::from_vec(active)
}

impl AffineFaceProfile {
    fn new(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        constraints: &LinearInequalityConstraints,
        active_indices: ArrayView1<'_, u64>,
    ) -> Result<Self, EstimationError> {
        let face = active_face_from_parts(
            constraints.a.view(),
            constraints.b.view(),
            active_indices,
            x.ncols(),
        )?;
        let normal_gram = face.a.dot(&face.a.t());
        let normal_factor = normal_gram
            .cholesky(Side::Lower)
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        let normal_coordinates = normal_factor.solvevec(&face.b);
        let beta_particular_vec = face.a.t().dot(&normal_coordinates);
        let beta_particular = beta_particular_vec.insert_axis(Axis(1));
        let tangent_design = x.dot(&face.z);
        let weighted_tangent_design = &tangent_design * &weights.view().insert_axis(Axis(1));
        let tangent_gram = tangent_design.t().dot(&weighted_tangent_design);
        let tangent_penalty = face.z.t().dot(&penalty).dot(&face.z);
        let base_response = y.to_owned() - &x.dot(&beta_particular);
        let weighted_base_response = &base_response * &weights.view().insert_axis(Axis(1));
        let tangent_rhs_data = tangent_design.t().dot(&weighted_base_response);
        let tangent_penalty_particular = face.z.t().dot(&penalty).dot(&beta_particular);
        let penalty_geometry = tangent_penalty_geometry(&tangent_penalty)?;
        let penalty_rank = penalty_geometry.rank;
        let penalty_logdet = penalty_geometry.logdet;
        let n_effective = weights.iter().filter(|&&value| value > 0.0).count();
        let penalty_nullity = face.z.ncols().saturating_sub(penalty_rank);
        if n_effective <= penalty_nullity {
            crate::bail_invalid_estim!(
                "constrained Gaussian REML requires more positive-weight rows than tangent penalty nullity"
            );
        }
        Ok(Self {
            x: x.to_owned(),
            y: y.to_owned(),
            penalty: penalty.to_owned(),
            weights: weights.to_owned(),
            face,
            beta_particular,
            tangent_gram,
            tangent_penalty,
            tangent_rhs_data,
            tangent_penalty_particular,
            penalty_rank,
            penalty_logdet,
            residual_df: (n_effective - penalty_nullity) as f64,
        })
    }

    fn evaluate(&self, rho: f64) -> Result<AffineFaceEvaluation, EstimationError> {
        let lambda = gam_problem::checked_exp_log_strength(rho)
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
        let tangent_dim = self.face.z.ncols();
        let (gamma, inverse, logdet_h) = if tangent_dim == 0 {
            (Array2::zeros((0, 1)), Array2::zeros((0, 0)), 0.0)
        } else {
            let hessian = &self.tangent_gram + &(self.tangent_penalty.clone() * lambda);
            let factor = hessian
                .cholesky(Side::Lower)
                .map_err(EstimationError::LinearSystemSolveFailed)?;
            let rhs = &self.tangent_rhs_data - &(self.tangent_penalty_particular.clone() * lambda);
            let gamma = factor.solve_mat(&rhs);
            let inverse = factor.solve_mat(&Array2::<f64>::eye(tangent_dim));
            let logdet_h = 2.0 * factor.diag().iter().map(|value| value.ln()).sum::<f64>();
            (gamma, inverse, logdet_h)
        };
        let beta = &self.beta_particular + &self.face.z.dot(&gamma);
        let fitted = self.x.dot(&beta);
        let residual = &self.y - &fitted;
        let weighted_rss =
            (&residual * &residual * &self.weights.view().insert_axis(Axis(1))).sum();
        let penalty_beta = self.penalty.dot(&beta);
        let energy = (&beta * &penalty_beta).sum();
        let penalized_deviance = weighted_rss + lambda * energy;
        if !penalized_deviance.is_finite() || penalized_deviance <= 0.0 {
            crate::bail_invalid_estim!(
                "constrained Gaussian REML profiled deviance must be positive"
            );
        }

        let inverse_penalty = inverse.dot(&self.tangent_penalty);
        let trace_inverse_penalty = trace(&inverse_penalty);
        let trace_inverse_penalty_squared = trace(&inverse_penalty.dot(&inverse_penalty));
        let tangent_penalty_beta = self.face.z.t().dot(&penalty_beta);
        let curvature_energy = if tangent_dim == 0 {
            0.0
        } else {
            (&tangent_penalty_beta * &inverse.dot(&tangent_penalty_beta)).sum()
        };
        let logdet_penalty = self.penalty_logdet + self.penalty_rank as f64 * rho;
        let score = 0.5 * (logdet_h - logdet_penalty)
            + 0.5
                * self.residual_df
                * (1.0 + (2.0 * std::f64::consts::PI * penalized_deviance / self.residual_df).ln());
        let lambda_energy = lambda * energy;
        let rho_gradient = 0.5 * (lambda * trace_inverse_penalty - self.penalty_rank as f64)
            + 0.5 * self.residual_df * lambda_energy / penalized_deviance;
        let rho_curvature = 0.5
            * (lambda * trace_inverse_penalty - lambda * lambda * trace_inverse_penalty_squared)
            + 0.5
                * self.residual_df
                * ((lambda_energy - 2.0 * lambda * lambda * curvature_energy) / penalized_deviance
                    - (lambda_energy / penalized_deviance).powi(2));
        let edf = tangent_dim as f64 - lambda * trace_inverse_penalty;
        Ok(AffineFaceEvaluation {
            rho,
            lambda,
            score,
            rho_gradient,
            rho_curvature,
            edf,
            beta,
            fitted,
        })
    }
}

fn tangent_penalty_geometry(
    penalty: &Array2<f64>,
) -> Result<TangentPenaltyGeometry, EstimationError> {
    if penalty.is_empty() {
        return Ok(TangentPenaltyGeometry {
            rank: 0,
            logdet: 0.0,
            pseudoinverse: Array2::zeros(penalty.dim()),
        });
    }
    let (eigenvalues, eigenvectors) = penalty
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    let scale = eigenvalues
        .iter()
        .fold(0.0_f64, |maximum, &value| maximum.max(value.abs()));
    let tolerance =
        default_rrqr_rank_alpha() * f64::EPSILON * penalty.nrows().max(1) as f64 * scale;
    let mut rank = 0usize;
    let mut logdet = 0.0;
    let mut scaled_eigenvectors = Array2::<f64>::zeros(eigenvectors.dim());
    for (index, &value) in eigenvalues.iter().enumerate() {
        if !value.is_finite() {
            return Err(EstimationError::PenaltySpectrumNonFinite {
                context: "constrained Gaussian REML tangent penalty".to_string(),
                index,
                value,
            });
        }
        if value < -tolerance {
            return Err(EstimationError::PenaltySpectrumIndefinite {
                context: "constrained Gaussian REML tangent penalty".to_string(),
                index,
                value,
                tolerance,
                scale,
            });
        }
        if value > tolerance {
            rank += 1;
            logdet += value.ln();
            for row in 0..eigenvectors.nrows() {
                scaled_eigenvectors[[row, index]] = eigenvectors[[row, index]] / value;
            }
        }
    }
    Ok(TangentPenaltyGeometry {
        rank,
        logdet,
        pseudoinverse: scaled_eigenvectors.dot(&eigenvectors.t()),
    })
}

fn optimize_affine_face(
    profile: &AffineFaceProfile,
    initial_rho: f64,
) -> Result<AffineFaceEvaluation, EstimationError> {
    let bound = crate::estimate::RHO_BOUND;
    let seed_rho = initial_rho.clamp(-bound, bound);
    let seed = profile.evaluate(seed_rho)?;
    let seed_point = Array1::from_vec(vec![seed_rho]);
    let initial_sample = FirstOrderSample {
        value: seed.score,
        gradient: Array1::from_vec(vec![seed.rho_gradient]),
    };
    let objective_profile = profile.clone();
    let objective = FusedObjective::new(move |point: &Array1<f64>| {
        objective_profile
            .evaluate(point[0])
            .map(|evaluation| FirstOrderSample {
                value: evaluation.score,
                gradient: Array1::from_vec(vec![evaluation.rho_gradient]),
            })
            .map_err(|error| ObjectiveEvalError::fatal(error.to_string()))
    });
    let bound_resolution = f64::EPSILON.sqrt();
    let stationarity_resolution = rho_stationarity_resolution();
    let bounds = Bounds::new(
        Array1::from_vec(vec![-bound]),
        Array1::from_vec(vec![bound]),
        bound_resolution,
    )
    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let mut optimizer = Bfgs::new(seed_point.clone(), objective)
        .with_initial_sample(seed_point, initial_sample)
        .with_bounds(bounds)
        .with_gradient_tolerance(GradientTolerance::relative_to_cost(stationarity_resolution));
    let solution = optimizer.run().map_err(|error| {
        EstimationError::RemlOptimizationFailed(format!(
            "affine-face Gaussian REML optimization did not converge: {error}"
        ))
    })?;
    let mut accepted = polish_affine_rho(profile, solution.final_point[0])?;
    for endpoint in [-bound, bound] {
        let candidate = profile.evaluate(endpoint)?;
        let projected_stationary = (endpoint < 0.0 && candidate.rho_gradient >= 0.0)
            || (endpoint > 0.0 && candidate.rho_gradient <= 0.0);
        if projected_stationary && candidate.score < accepted.score {
            accepted = candidate;
        }
    }
    Ok(accepted)
}

fn polish_affine_rho(
    profile: &AffineFaceProfile,
    initial_rho: f64,
) -> Result<AffineFaceEvaluation, EstimationError> {
    let bound = crate::estimate::RHO_BOUND;
    let resolution = rho_stationarity_resolution();
    let bound_resolution = f64::EPSILON.sqrt();
    let mut rho = initial_rho.clamp(-bound, bound);
    loop {
        let current = profile.evaluate(rho)?;
        let at_lower = rho <= -bound + bound_resolution * bound.max(1.0);
        let at_upper = rho >= bound - bound_resolution * bound.max(1.0);
        if (at_lower && current.rho_gradient >= 0.0) || (at_upper && current.rho_gradient <= 0.0) {
            return Ok(current);
        }
        let curvature_resolution = resolution * (1.0 + current.rho_gradient.abs());
        if !current.rho_curvature.is_finite() || current.rho_curvature <= curvature_resolution {
            return Err(EstimationError::GradientUnavailable {
                context: "constrained Gaussian REML forward",
                mode: "affine-face smoothing optimum has unresolved positive curvature",
            });
        }
        if current.rho_gradient.abs() <= resolution * (1.0 + current.score.abs()) {
            return Ok(current);
        }
        let mut candidate_rho =
            (rho - current.rho_gradient / current.rho_curvature).clamp(-bound, bound);
        if candidate_rho.to_bits() == rho.to_bits() {
            return Err(EstimationError::GradientUnavailable {
                context: "constrained Gaussian REML forward",
                mode: "affine-face smoothing root is below floating-point resolution",
            });
        }
        let mut candidate = profile.evaluate(candidate_rho)?;
        while candidate.score >= current.score {
            candidate_rho = 0.5 * (rho + candidate_rho);
            if candidate_rho.to_bits() == rho.to_bits() {
                return Err(EstimationError::GradientUnavailable {
                    context: "constrained Gaussian REML forward",
                    mode: "affine-face smoothing polish cannot resolve a descent step",
                });
            }
            candidate = profile.evaluate(candidate_rho)?;
        }
        rho = candidate_rho;
    }
}

fn rho_stationarity_resolution() -> f64 {
    let central_difference_scale = f64::EPSILON.cbrt();
    central_difference_scale * central_difference_scale
}

/// Exact VJP on a certified, locally constant affine active face.
pub fn constrained_gaussian_reml_backward(
    problem: ConstrainedGaussianRemlBackwardProblem<'_>,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    validate_problem(&problem)?;
    let constraints = LinearInequalityConstraints::new(
        problem.a_inequality.to_owned(),
        problem.b_inequality.to_owned(),
    )
    .and_then(|constraints| constraints.canonicalized())
    .map_err(EstimationError::InvalidInput)?;
    let face = active_face_from_parts(
        constraints.a.view(),
        constraints.b.view(),
        problem.active_indices,
        problem.x.ncols(),
    )?;
    let state = face_state(&problem, &face)?;
    certify_strict_complementarity(&problem, &state, &face, &constraints)?;

    let n = problem.x.nrows();
    let p = problem.x.ncols();
    let d = problem.y.ncols();
    let mut out = VjpAccumulator::zeros(n, p, d);

    // Coefficient and fitted-value outputs share the KKT mode-response channel.
    let mut mode_seed = Array2::<f64>::zeros((p, d));
    if let Some(seed) = problem.grad_coefficients {
        mode_seed += &seed;
    }
    if let Some(seed) = problem.grad_fitted {
        mode_seed += &problem.x.t().dot(&seed);
        out.x += &seed.dot(&state.beta.t());
    }
    add_mode_vjp(&problem, &state, mode_seed.view(), 1.0, &mut out);

    out.lambda += problem.grad_lambda;
    out.lambda += problem.grad_log_lambda / problem.lambda;

    if problem.grad_reml_score != 0.0 {
        add_score_vjp(&problem, &state, problem.grad_reml_score, &mut out);
    }
    if problem.grad_edf != 0.0 {
        add_face_edf_vjp(&problem, &state, problem.grad_edf, &mut out);
    }

    // Every output other than the optimized score itself can pull through the
    // stationary smoothing root.  At an optimizer box face rho is locally
    // constant, so that root channel is exactly absent.
    if out.lambda != 0.0 && !rho_is_box_active(problem.lambda) {
        let curvature = rho_score_curvature(&problem, &state);
        let scale = rho_curvature_scale(&problem, &state);
        let resolution = f64::EPSILON.sqrt() * ((n + p + d).max(1) as f64).sqrt() * scale.max(1.0);
        if !curvature.is_finite() || curvature <= resolution {
            return Err(EstimationError::GradientUnavailable {
                context: "constrained Gaussian REML backward",
                mode: "active-face smoothing root has unresolved curvature",
            });
        }
        let root_seed = -problem.lambda * out.lambda / curvature;
        // The explicit lambda cotangent has now been consumed by the implicit
        // root.  The public result has no lambda-gradient field.
        add_rho_score_vjp(&problem, &state, root_seed, &mut out);
    }

    symmetrize_in_place(&mut out.penalty);
    Ok(GaussianRemlBackwardResult {
        grad_x: out.x,
        grad_y: out.y,
        grad_penalty: out.penalty,
        grad_weights: out.weights,
    })
}

fn validate_problem(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
) -> Result<(), EstimationError> {
    let n = problem.x.nrows();
    let p = problem.x.ncols();
    let d = problem.y.ncols();
    if d != 1 {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML backward requires one response column; got {d}"
        );
    }
    if problem.y.nrows() != n
        || problem.penalty.dim() != (p, p)
        || problem.coefficients.dim() != (p, d)
        || problem.a_inequality.ncols() != p
        || problem.b_inequality.len() != problem.a_inequality.nrows()
    {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML backward input shapes are inconsistent"
        );
    }
    if problem.active_indices.is_empty() {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML active-face backward requires a non-empty active set"
        );
    }
    if let Some(weights) = problem.weights
        && weights.len() != n
    {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML weights length {} does not match row count {n}",
            weights.len()
        );
    }
    if let Some(seed) = problem.grad_coefficients
        && seed.dim() != (p, d)
    {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML coefficient cotangent shape mismatch"
        );
    }
    if let Some(seed) = problem.grad_fitted
        && seed.dim() != (n, d)
    {
        crate::bail_invalid_estim!("constrained Gaussian REML fitted cotangent shape mismatch");
    }
    if !problem.lambda.is_finite() || problem.lambda <= 0.0 {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML backward requires a positive finite lambda"
        );
    }
    let all_finite = problem
        .x
        .iter()
        .chain(problem.y.iter())
        .chain(problem.penalty.iter())
        .chain(problem.coefficients.iter())
        .chain(problem.a_inequality.iter())
        .chain(problem.b_inequality.iter())
        .all(|value| value.is_finite());
    if !all_finite {
        crate::bail_invalid_estim!("constrained Gaussian REML backward inputs must be finite");
    }
    if let Some(weights) = problem.weights
        && weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML weights must be finite and non-negative"
        );
    }
    Ok(())
}

fn active_face_from_parts(
    a_inequality: ArrayView2<'_, f64>,
    b_inequality: ArrayView1<'_, f64>,
    active_indices: ArrayView1<'_, u64>,
    p: usize,
) -> Result<ActiveFace, EstimationError> {
    let mut seen = vec![false; a_inequality.nrows()];
    let mut active = Array2::<f64>::zeros((active_indices.len(), p));
    let mut active_bounds = Array1::<f64>::zeros(active_indices.len());
    for (row, &raw_index) in active_indices.iter().enumerate() {
        let index = raw_index as usize;
        if index >= a_inequality.nrows() {
            crate::bail_invalid_estim!(
                "constrained Gaussian REML active index {index} is out of range"
            );
        }
        if seen[index] {
            crate::bail_invalid_estim!(
                "constrained Gaussian REML active index {index} occurs more than once"
            );
        }
        seen[index] = true;
        active.row_mut(row).assign(&a_inequality.row(index));
        active_bounds[row] = b_inequality[index];
    }

    // Canonicalize a redundant representation to independent face equations.
    // RRQR is run on A_a' so its pivoted columns name active constraint rows.
    let active_t = active.t().to_owned();
    let rrqr = rrqr_with_permutation(&active_t, default_rrqr_rank_alpha())
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    if rrqr.rank == 0 {
        return Err(EstimationError::GradientUnavailable {
            context: "constrained Gaussian REML backward",
            mode: "active constraint rows have zero numerical rank",
        });
    }
    let mut a = Array2::<f64>::zeros((rrqr.rank, p));
    let mut b = Array1::<f64>::zeros(rrqr.rank);
    for (row, &source) in rrqr.column_permutation.iter().take(rrqr.rank).enumerate() {
        a.row_mut(row).assign(&active.row(source));
        b[row] = active_bounds[source];
    }
    let (z, rank) = rrqr_nullspace_basis(&a.t().to_owned(), default_rrqr_rank_alpha())
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    if rank != rrqr.rank {
        return Err(EstimationError::GradientUnavailable {
            context: "constrained Gaussian REML backward",
            mode: "active-face rank certificate is inconsistent",
        });
    }
    Ok(ActiveFace { a, b, z })
}

fn face_state(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    face: &ActiveFace,
) -> Result<FaceState, EstimationError> {
    let n = problem.x.nrows();
    let p = problem.x.ncols();
    let penalty = symmetric_average(problem.penalty);
    let weights = problem
        .weights
        .map_or_else(|| Array1::ones(n), |values| values.to_owned());
    let beta = problem.coefficients.to_owned();
    let fitted = problem.x.dot(&beta);
    let residual = problem.y.to_owned() - &fitted;
    let weighted_x = &problem.x.to_owned() * &weights.view().insert_axis(Axis(1));
    let gram = problem.x.t().dot(&weighted_x);
    let hessian = &gram + &(penalty.clone() * problem.lambda);

    let k = face.z.ncols();
    let (p_response, q_penalty, penalty_rank) = if k == 0 {
        (Array2::zeros((p, p)), Array2::zeros((p, p)), 0)
    } else {
        let tangent_hessian = face.z.t().dot(&hessian).dot(&face.z);
        let inverse = tangent_hessian
            .cholesky(Side::Lower)
            .map_err(EstimationError::LinearSystemSolveFailed)?
            .solve_mat(&Array2::<f64>::eye(k));
        let p_response = face.z.dot(&inverse).dot(&face.z.t());
        let tangent_penalty = face.z.t().dot(&penalty).dot(&face.z);
        let penalty_geometry = tangent_penalty_geometry(&tangent_penalty)?;
        let q_penalty = face.z.dot(&penalty_geometry.pseudoinverse).dot(&face.z.t());
        (p_response, q_penalty, penalty_geometry.rank)
    };
    let n_effective = weights.iter().filter(|&&value| value > 0.0).count();
    let nullity = k.saturating_sub(penalty_rank);
    if n_effective <= nullity {
        crate::bail_invalid_estim!(
            "constrained Gaussian REML requires more positive-weight rows than tangent penalty nullity"
        );
    }
    Ok(FaceState {
        penalty,
        weights,
        beta,
        residual,
        gram,
        p_response,
        q_penalty,
        penalty_rank,
        residual_df: (n_effective - nullity) as f64,
    })
}

fn certify_strict_complementarity(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
    face: &ActiveFace,
    constraints: &LinearInequalityConstraints,
) -> Result<(), EstimationError> {
    let p = problem.x.ncols();
    let beta = state.beta.column(0);
    let beta_scale = beta
        .iter()
        .fold(1.0_f64, |scale, &value| scale.max(value.abs()));
    let mut active_mask = vec![false; constraints.a.nrows()];
    for &index in problem.active_indices {
        active_mask[index as usize] = true;
    }
    for row in 0..constraints.a.nrows() {
        let a = constraints.a.row(row);
        if a.iter().all(|&value| value == 0.0) {
            continue;
        }
        let slack = a.dot(&beta) - constraints.b[row];
        let tolerance = crate::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
            * beta_scale.max(constraints.b[row].abs().max(1.0));
        if active_mask[row] {
            if slack.abs() > tolerance {
                return Err(EstimationError::GradientUnavailable {
                    context: "constrained Gaussian REML backward",
                    mode: "reported active row is not on the accepted face",
                });
            }
        } else if slack <= tolerance {
            return Err(EstimationError::GradientUnavailable {
                context: "constrained Gaussian REML backward",
                mode: "inactive constraint is not strictly separated from the face",
            });
        }
    }

    // Stationarity for A beta >= b is grad f - A' mu = 0, mu >= 0.
    let weighted_residual = &state.residual.column(0) * &state.weights;
    let gradient =
        -problem.x.t().dot(&weighted_residual) + problem.lambda * state.penalty.dot(&beta);
    let normal_gram = face.a.dot(&face.a.t());
    let normal_inverse = normal_gram
        .cholesky(Side::Lower)
        .map_err(EstimationError::LinearSystemSolveFailed)?
        .solve_mat(&Array2::<f64>::eye(face.a.nrows()));
    let multipliers = normal_inverse.dot(&face.a.dot(&gradient));
    let reconstructed = face.a.t().dot(&multipliers);
    let residual = &gradient - &reconstructed;
    let residual_inf = residual
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    let gradient_scale = gradient
        .iter()
        .chain(reconstructed.iter())
        .fold(1.0_f64, |scale, &value| scale.max(value.abs()));
    let arithmetic_uncertainty =
        f64::EPSILON * ((p + face.a.nrows()).max(1) as f64) * gradient_scale;
    let normal_left_inverse = normal_inverse.dot(&face.a);
    let inverse_inf_norm = normal_left_inverse
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    // The accepted active-set solve is certified only to its public geometric
    // feasibility resolution.  Propagate that resolution through the normal
    // left inverse as a dual uncertainty as well: an O(1e-11) positive number
    // reconstructed from an O(1e-8)-accurate KKT state is not evidence of
    // strict complementarity.  Using arithmetic error alone incorrectly
    // blessed exactly the weak active-set transition this guard exists for.
    let solver_uncertainty = crate::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL * gradient_scale;
    let multiplier_uncertainty =
        inverse_inf_norm * (residual_inf + arithmetic_uncertainty + solver_uncertainty);
    if multipliers
        .iter()
        .any(|&value| !value.is_finite() || value <= multiplier_uncertainty)
    {
        return Err(EstimationError::GradientUnavailable {
            context: "constrained Gaussian REML backward",
            mode: "active constraint is weakly active; the derivative is set-valued",
        });
    }
    Ok(())
}

fn add_mode_vjp(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
    seed: ArrayView2<'_, f64>,
    scale: f64,
    out: &mut VjpAccumulator,
) {
    if scale == 0.0 || seed.iter().all(|&value| value == 0.0) {
        return;
    }
    let u = state.p_response.dot(&seed) * scale;
    let xu = problem.x.dot(&u);
    let weighted_residual = &state.residual * &state.weights.view().insert_axis(Axis(1));
    let weighted_xu = &xu * &state.weights.view().insert_axis(Axis(1));
    out.y += &weighted_xu;
    out.x += &weighted_residual.dot(&u.t());
    out.x -= &weighted_xu.dot(&state.beta.t());
    out.weights += &(&xu * &state.residual).sum_axis(Axis(1));
    let s_beta = state.penalty.dot(&state.beta);
    let raw_penalty = u.dot(&state.beta.t());
    out.penalty -= &(symmetric_average(raw_penalty.view()) * problem.lambda);
    out.lambda -= (&u * &s_beta).sum();
}

fn add_score_vjp(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
    seed: f64,
    out: &mut VjpAccumulator,
) {
    let d = problem.y.ncols() as f64;
    let wxp =
        (&problem.x.to_owned() * &state.weights.view().insert_axis(Axis(1))).dot(&state.p_response);
    out.x += &(wxp * (seed * d));
    out.weights += &(row_quadratic(problem.x, &state.p_response) * (0.5 * seed * d));
    out.penalty += &((&state.p_response * problem.lambda - &state.q_penalty) * (0.5 * seed * d));
    let trace_ps = trace_product(&state.p_response, &state.penalty);
    out.lambda += seed * 0.5 * d * (trace_ps - state.penalty_rank as f64 / problem.lambda);

    for output in 0..problem.y.ncols() {
        let residual = state.residual.column(output);
        let beta = state.beta.column(output);
        let weighted_rss = residual.dot(&(&residual * &state.weights));
        let s_beta = state.penalty.dot(&beta);
        let energy = beta.dot(&s_beta);
        let deviance = weighted_rss + problem.lambda * energy;
        let tau = state.residual_df / deviance;
        let weighted_residual = &residual * &state.weights;
        out.y
            .column_mut(output)
            .scaled_add(seed * tau, &weighted_residual);
        let x_term = weighted_residual
            .insert_axis(Axis(1))
            .dot(&beta.insert_axis(Axis(0)));
        out.x -= &(x_term * (seed * tau));
        out.weights += &(&residual * &residual * (0.5 * seed * tau));
        let beta_outer = beta.insert_axis(Axis(1)).dot(&beta.insert_axis(Axis(0)));
        out.penalty += &(beta_outer * (0.5 * seed * tau * problem.lambda));
        out.lambda += 0.5 * seed * tau * energy;
    }
}

fn add_face_edf_vjp(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
    seed: f64,
    out: &mut VjpAccumulator,
) {
    let pgp = state.p_response.dot(&state.gram).dot(&state.p_response);
    let gram_seed = &state.p_response - &pgp;
    add_gram_vjp(problem.x, &state.weights, &gram_seed, seed, out);
    out.penalty -= &(pgp.clone() * (seed * problem.lambda));
    out.lambda -= seed * trace_product(&pgp, &state.penalty);
}

fn add_rho_score_vjp(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
    seed: f64,
    out: &mut VjpAccumulator,
) {
    if seed == 0.0 {
        return;
    }
    let psp = state.p_response.dot(&state.penalty).dot(&state.p_response);
    let gram_seed = psp.clone() * (-0.5 * seed * problem.lambda * problem.y.ncols() as f64);
    add_gram_vjp(problem.x, &state.weights, &gram_seed, 1.0, out);
    out.penalty += &((&state.p_response - &(psp * problem.lambda))
        * (0.5 * seed * problem.lambda * problem.y.ncols() as f64));

    for output in 0..problem.y.ncols() {
        let beta = state.beta.column(output);
        let residual = state.residual.column(output);
        let s_beta = state.penalty.dot(&beta);
        let energy = beta.dot(&s_beta);
        let weighted_rss = residual.dot(&(&residual * &state.weights));
        let deviance = weighted_rss + problem.lambda * energy;
        let energy_seed = 0.5 * seed * state.residual_df * problem.lambda / deviance;
        let beta_seed = s_beta.insert_axis(Axis(1)) * (2.0 * energy_seed);
        add_mode_vjp(problem, state, beta_seed.view(), 1.0, out);
        let beta_outer = beta.insert_axis(Axis(1)).dot(&beta.insert_axis(Axis(0)));
        out.penalty += &(&beta_outer * energy_seed);

        let deviance_seed =
            -0.5 * seed * state.residual_df * problem.lambda * energy / (deviance * deviance);
        let weighted_residual = &residual * &state.weights;
        out.y
            .column_mut(output)
            .scaled_add(2.0 * deviance_seed, &weighted_residual);
        let x_term = weighted_residual
            .insert_axis(Axis(1))
            .dot(&beta.insert_axis(Axis(0)));
        out.x -= &(x_term * (2.0 * deviance_seed));
        out.weights += &(&residual * &residual * deviance_seed);
        out.penalty += &(beta_outer * (deviance_seed * problem.lambda));
    }
}

fn rho_score_curvature(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
) -> f64 {
    let ps = state.p_response.dot(&state.penalty);
    let trace_ps = trace(&ps);
    let trace_psps = trace(&ps.dot(&ps));
    let mut value = 0.5
        * problem.y.ncols() as f64
        * (problem.lambda * trace_ps - problem.lambda * problem.lambda * trace_psps);
    for output in 0..problem.y.ncols() {
        let beta = state.beta.column(output);
        let residual = state.residual.column(output);
        let s_beta = state.penalty.dot(&beta);
        let energy = beta.dot(&s_beta);
        let curvature_energy = s_beta.dot(&state.p_response.dot(&s_beta));
        let deviance = residual.dot(&(&residual * &state.weights)) + problem.lambda * energy;
        let first = problem.lambda * energy;
        value += 0.5
            * state.residual_df
            * ((first - 2.0 * problem.lambda * problem.lambda * curvature_energy) / deviance
                - (first / deviance).powi(2));
    }
    value
}

fn rho_curvature_scale(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
) -> f64 {
    let ps = state.p_response.dot(&state.penalty);
    let mut scale = problem.y.ncols() as f64 * (problem.lambda * trace(&ps)).abs()
        + problem.y.ncols() as f64 * (problem.lambda * problem.lambda * trace(&ps.dot(&ps))).abs();
    for output in 0..problem.y.ncols() {
        let beta = state.beta.column(output);
        let residual = state.residual.column(output);
        let s_beta = state.penalty.dot(&beta);
        let energy = beta.dot(&s_beta);
        let curvature_energy = s_beta.dot(&state.p_response.dot(&s_beta));
        let deviance = residual.dot(&(&residual * &state.weights)) + problem.lambda * energy;
        scale += state.residual_df
            * ((problem.lambda * energy / deviance).abs()
                + (2.0 * problem.lambda * problem.lambda * curvature_energy / deviance).abs()
                + (problem.lambda * energy / deviance).powi(2));
    }
    scale
}

fn add_gram_vjp(
    x: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    gram_seed: &Array2<f64>,
    scale: f64,
    out: &mut VjpAccumulator,
) {
    let symmetric = symmetric_average(gram_seed.view());
    let x_seed = x.dot(&symmetric) * (2.0 * scale);
    out.x += &(&x_seed * &weights.view().insert_axis(Axis(1)));
    out.weights += &(row_quadratic(x, &symmetric) * scale);
}

fn row_quadratic(x: ArrayView2<'_, f64>, matrix: &Array2<f64>) -> Array1<f64> {
    let xm = x.dot(matrix);
    (&xm * &x).sum_axis(Axis(1))
}

fn symmetric_average(matrix: ArrayView2<'_, f64>) -> Array2<f64> {
    (&matrix + &matrix.t()) * 0.5
}

fn symmetrize_in_place(matrix: &mut Array2<f64>) {
    for row in 0..matrix.nrows() {
        for col in (row + 1)..matrix.ncols() {
            let value = 0.5 * (matrix[[row, col]] + matrix[[col, row]]);
            matrix[[row, col]] = value;
            matrix[[col, row]] = value;
        }
    }
}

fn trace(matrix: &Array2<f64>) -> f64 {
    matrix.diag().sum()
}

fn trace_product(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    (&left.t() * right).sum()
}

fn rho_is_box_active(lambda: f64) -> bool {
    let rho = lambda.ln();
    let bound = crate::estimate::RHO_BOUND;
    rho.abs() >= bound - f64::EPSILON * bound.max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    struct Fixture {
        x: Array2<f64>,
        y: Array2<f64>,
        penalty: Array2<f64>,
        weights: Array1<f64>,
        constraints: LinearInequalityConstraints,
    }

    fn affine_fixture() -> Fixture {
        let n = 16usize;
        let mut x = Array2::<f64>::zeros((n, 3));
        let mut y = Array2::<f64>::zeros((n, 1));
        let mut weights = Array1::<f64>::zeros(n);
        for row in 0..n {
            let t = -1.0 + 2.0 * row as f64 / (n - 1) as f64;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = t;
            x[[row, 2]] = t * t;
            y[[row, 0]] = 0.2 - 0.4 * t - 1.8 * t * t + 0.02 * (3.0 * t).sin();
            weights[row] = 0.7 + 0.6 * row as f64 / (n - 1) as f64;
        }
        let penalty = array![[0.30, 0.05, 0.08], [0.05, 1.00, 0.12], [0.08, 0.12, 1.20]];
        let constraints =
            LinearInequalityConstraints::new(array![[0.0, 0.0, 1.0]], array![0.1]).unwrap();
        Fixture {
            x,
            y,
            penalty,
            weights,
            constraints,
        }
    }

    fn fit_fixture(fixture: &Fixture) -> ConstrainedGaussianRemlForwardResult {
        constrained_gaussian_reml_forward(ConstrainedGaussianRemlForwardProblem {
            x: fixture.x.view(),
            y: fixture.y.view(),
            penalty: fixture.penalty.view(),
            weights: Some(fixture.weights.view()),
            constraints: Some(&fixture.constraints),
            init_lambda: None,
        })
        .unwrap_or_else(|error| panic!("affine constrained REML fit failed: {error}"))
    }

    fn scalar_loss(result: &ConstrainedGaussianRemlForwardResult) -> f64 {
        let coefficient_seed = array![[0.7], [-0.3], [1.1]];
        let fitted_seed = Array2::from_shape_fn(result.fitted.dim(), |(row, _)| {
            0.2 + row as f64 / result.fitted.nrows() as f64
        });
        (&result.coefficients * &coefficient_seed).sum()
            + (&result.fitted * &fitted_seed).sum()
            + 0.23 * result.lambda
            + 0.41 * result.lambda.ln()
            + 0.67 * result.reml_score
            - 0.29 * result.edf
    }

    fn finite_difference_step(value: f64) -> f64 {
        f64::EPSILON.cbrt() * value.abs().max(1.0)
    }

    fn assert_fd(analytic: f64, numerical: f64, label: &str) {
        let tolerance = 64.0 * f64::EPSILON.cbrt() * (1.0 + analytic.abs().max(numerical.abs()));
        assert!(
            (analytic - numerical).abs() <= tolerance,
            "{label}: analytic={analytic:.12e}, numerical={numerical:.12e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn active_nonzero_affine_face_vjp_matches_central_differences() {
        let fixture = affine_fixture();
        let fit = fit_fixture(&fixture);
        assert_eq!(fit.active_indices.as_slice().unwrap(), &[0]);
        assert!((fit.coefficients[[2, 0]] - 0.1).abs() <= 1.0e-8);

        let coefficient_seed = array![[0.7], [-0.3], [1.1]];
        let fitted_seed = Array2::from_shape_fn(fit.fitted.dim(), |(row, _)| {
            0.2 + row as f64 / fit.fitted.nrows() as f64
        });
        let backward = constrained_gaussian_reml_backward(ConstrainedGaussianRemlBackwardProblem {
            x: fixture.x.view(),
            y: fixture.y.view(),
            penalty: fixture.penalty.view(),
            weights: Some(fixture.weights.view()),
            a_inequality: fixture.constraints.a.view(),
            b_inequality: fixture.constraints.b.view(),
            active_indices: fit.active_indices.view(),
            lambda: fit.lambda,
            coefficients: fit.coefficients.view(),
            grad_coefficients: Some(coefficient_seed.view()),
            grad_fitted: Some(fitted_seed.view()),
            grad_lambda: 0.23,
            grad_log_lambda: 0.41,
            grad_reml_score: 0.67,
            grad_edf: -0.29,
        })
        .unwrap_or_else(|error| panic!("affine constrained REML backward failed: {error}"));

        for row in 0..fixture.x.nrows() {
            for col in 0..fixture.x.ncols() {
                let step = finite_difference_step(fixture.x[[row, col]]);
                let mut plus = affine_fixture();
                let mut minus = affine_fixture();
                plus.x[[row, col]] += step;
                minus.x[[row, col]] -= step;
                let numerical = (scalar_loss(&fit_fixture(&plus))
                    - scalar_loss(&fit_fixture(&minus)))
                    / (2.0 * step);
                assert_fd(backward.grad_x[[row, col]], numerical, "grad_x");
            }
        }
        for row in 0..fixture.y.nrows() {
            let step = finite_difference_step(fixture.y[[row, 0]]);
            let mut plus = affine_fixture();
            let mut minus = affine_fixture();
            plus.y[[row, 0]] += step;
            minus.y[[row, 0]] -= step;
            let numerical = (scalar_loss(&fit_fixture(&plus)) - scalar_loss(&fit_fixture(&minus)))
                / (2.0 * step);
            assert_fd(backward.grad_y[[row, 0]], numerical, "grad_y");
        }
        for row in 0..fixture.penalty.nrows() {
            for col in 0..fixture.penalty.ncols() {
                let step = finite_difference_step(fixture.penalty[[row, col]]);
                let mut plus = affine_fixture();
                let mut minus = affine_fixture();
                plus.penalty[[row, col]] += step;
                minus.penalty[[row, col]] -= step;
                let numerical = (scalar_loss(&fit_fixture(&plus))
                    - scalar_loss(&fit_fixture(&minus)))
                    / (2.0 * step);
                assert_fd(backward.grad_penalty[[row, col]], numerical, "grad_penalty");
            }
        }
        for row in 0..fixture.weights.len() {
            let step = finite_difference_step(fixture.weights[row]);
            let mut plus = affine_fixture();
            let mut minus = affine_fixture();
            plus.weights[row] += step;
            minus.weights[row] -= step;
            let numerical = (scalar_loss(&fit_fixture(&plus)) - scalar_loss(&fit_fixture(&minus)))
                / (2.0 * step);
            assert_fd(backward.grad_weights[row], numerical, "grad_weights");
        }
    }

    #[test]
    fn weakly_active_exact_kkt_state_has_no_derivative() {
        let n = 24usize;
        let mut x = Array2::<f64>::zeros((n, 3));
        let mut cubic = Array1::<f64>::zeros(n);
        for row in 0..n {
            let t = -1.0 + 2.0 * row as f64 / (n - 1) as f64;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = t;
            x[[row, 2]] = t * t;
            cubic[row] = t * t * t;
        }
        let slope_projection = x.column(1).dot(&cubic) / x.column(1).dot(&x.column(1));
        let orthogonal = &cubic - &(x.column(1).to_owned() * slope_projection);
        let mut y = Array2::<f64>::from_elem((n, 1), 0.3);
        y.slice_mut(s![.., 0]).scaled_add(0.1, &orthogonal);
        let beta = array![[0.3], [0.0], [0.0]];
        let penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let weights = Array1::<f64>::ones(n);
        let a = array![[0.0, 0.0, 1.0]];
        let b = array![0.0];
        let active = array![0_u64];
        let seed = Array2::<f64>::ones((3, 1));
        let result = constrained_gaussian_reml_backward(ConstrainedGaussianRemlBackwardProblem {
            x: x.view(),
            y: y.view(),
            penalty: penalty.view(),
            weights: Some(weights.view()),
            a_inequality: a.view(),
            b_inequality: b.view(),
            active_indices: active.view(),
            lambda: 1.0,
            coefficients: beta.view(),
            grad_coefficients: Some(seed.view()),
            grad_fitted: None,
            grad_lambda: 0.0,
            grad_log_lambda: 0.0,
            grad_reml_score: 0.0,
            grad_edf: 0.0,
        });
        match result {
            Err(EstimationError::GradientUnavailable { mode, .. }) => {
                assert!(mode.contains("weakly active"), "unexpected mode: {mode}");
            }
            Err(error) => panic!("expected GradientUnavailable, got {error}"),
            Ok(_) => panic!("weakly active KKT state unexpectedly returned a gradient"),
        }
    }
}
