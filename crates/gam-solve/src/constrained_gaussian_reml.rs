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

use crate::estimate::EstimationError;
use crate::gaussian_reml::GaussianRemlBackwardResult;
use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerCholesky, default_rrqr_rank_alpha, rrqr_nullspace_basis, rrqr_with_permutation,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

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

struct ActiveFace {
    a: Array2<f64>,
    z: Array2<f64>,
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

/// Exact VJP on a certified, locally constant affine active face.
pub fn constrained_gaussian_reml_backward(
    problem: ConstrainedGaussianRemlBackwardProblem<'_>,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    validate_problem(&problem)?;
    let face = active_face(&problem)?;
    let state = face_state(&problem, &face)?;
    certify_strict_complementarity(&problem, &state, &face)?;

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
        add_full_space_edf_vjp(&problem, &state, problem.grad_edf, &mut out)?;
    }

    // Every output other than the optimized score itself can pull through the
    // stationary smoothing root.  At an optimizer box face rho is locally
    // constant, so that root channel is exactly absent.
    if out.lambda != 0.0 && !rho_is_box_active(problem.lambda) {
        let curvature = rho_score_curvature(&problem, &state);
        let scale = rho_curvature_scale(&problem, &state);
        let resolution = f64::EPSILON * ((n + p + d).max(1) as f64) * scale.max(1.0);
        if !curvature.is_finite() || curvature.abs() <= resolution {
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

fn active_face(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
) -> Result<ActiveFace, EstimationError> {
    let p = problem.x.ncols();
    let mut seen = vec![false; problem.a_inequality.nrows()];
    let mut active = Array2::<f64>::zeros((problem.active_indices.len(), p));
    for (row, &raw_index) in problem.active_indices.iter().enumerate() {
        let index = raw_index as usize;
        if index >= problem.a_inequality.nrows() {
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
        active.row_mut(row).assign(&problem.a_inequality.row(index));
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
    for (row, &source) in rrqr.column_permutation.iter().take(rrqr.rank).enumerate() {
        a.row_mut(row).assign(&active.row(source));
    }
    let (z, rank) = rrqr_nullspace_basis(&a.t().to_owned(), default_rrqr_rank_alpha())
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    if rank != rrqr.rank {
        return Err(EstimationError::GradientUnavailable {
            context: "constrained Gaussian REML backward",
            mode: "active-face rank certificate is inconsistent",
        });
    }
    Ok(ActiveFace { a, z })
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
        let (penalty_rank, tangent_pinv) =
            gam_linalg::utils::block_penalty_rank_and_pinv(&tangent_penalty)?;
        let q_penalty = face.z.dot(&tangent_pinv).dot(&face.z.t());
        (p_response, q_penalty, penalty_rank)
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
) -> Result<(), EstimationError> {
    let p = problem.x.ncols();
    let beta = state.beta.column(0);
    let beta_scale = beta
        .iter()
        .fold(1.0_f64, |scale, &value| scale.max(value.abs()));
    let mut active_mask = vec![false; problem.a_inequality.nrows()];
    for &index in problem.active_indices {
        active_mask[index as usize] = true;
    }
    for row in 0..problem.a_inequality.nrows() {
        let a = problem.a_inequality.row(row);
        let slack = a.dot(&beta) - problem.b_inequality[row];
        let row_scale = a
            .iter()
            .fold(1.0_f64, |scale, &value| scale.max(value.abs()));
        let tolerance = crate::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
            * row_scale
            * beta_scale.max(problem.b_inequality[row].abs().max(1.0));
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
    let multiplier_uncertainty = inverse_inf_norm * (residual_inf + arithmetic_uncertainty);
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
    for (gradient, &weight) in out.weights.iter_mut().zip(state.weights.iter()) {
        if weight > 0.0 {
            *gradient -= 0.5 * seed * d / weight;
        }
    }
}

fn add_full_space_edf_vjp(
    problem: &ConstrainedGaussianRemlBackwardProblem<'_>,
    state: &FaceState,
    seed: f64,
    out: &mut VjpAccumulator,
) -> Result<(), EstimationError> {
    let p = problem.x.ncols();
    let hessian = &state.gram + &(state.penalty.clone() * problem.lambda);
    let inverse = hessian
        .cholesky(Side::Lower)
        .map_err(EstimationError::LinearSystemSolveFailed)?
        .solve_mat(&Array2::<f64>::eye(p));
    let rgr = inverse.dot(&state.gram).dot(&inverse);
    let gram_seed = &inverse - &rgr;
    add_gram_vjp(problem.x, &state.weights, &gram_seed, seed, out);
    out.penalty -= &(rgr.clone() * (seed * problem.lambda));
    out.lambda -= seed * trace_product(&rgr, &state.penalty);
    Ok(())
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
