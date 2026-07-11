//! Constrained ring-of-clusters density for shape adjudication (#2262).
//!
//! A free Gaussian mixture explains discrete concepts but discards the cyclic
//! relationship between its clusters; a uniform ring preserves the topology but
//! cannot explain clumpy density. This module supplies the missing composite:
//! an isotropic Gaussian mixture whose `k` component means share one fitted
//! center and radius while retaining independent angles and mixing weights.
//!
//! The fit is a deterministic EM algorithm. Its M-step is exact for weights and
//! shared variance and solves the weighted geometric-circle problem for the
//! component centroids with analytic Gauss-Newton steps and monotone line search.
//! A fit object is minted only after both the likelihood-map and parameter-map
//! residuals certify a fixed point. Exhaustion returns a typed, resumable
//! checkpoint; it never yields comparable evidence.

use crate::evidence::{
    EvidenceHvpLogDet, EvidenceLogDetSource, GaussianMixtureConfig, fit_gaussian_mixture,
    laplace_evidence,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

const LOG_TAU: f64 = 1.8378770664093453_f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct DataFingerprint {
    nrows: usize,
    ncols: usize,
    lane_a: u64,
    lane_b: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RingClusterCertificate {
    pub mean_log_likelihood: f64,
    pub mean_log_likelihood_gain: f64,
    pub monotonicity_roundoff: f64,
    pub objective_residual: f64,
    pub objective_tolerance: f64,
    pub parameter_residual: f64,
    pub parameter_tolerance: f64,
}

/// Exact constrained-mixture state carried across an iteration-exhaustion wall.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RingClusterCheckpoint {
    pub weights: Array1<f64>,
    pub center: [f64; 2],
    pub radius: f64,
    pub angles: Array1<f64>,
    pub variance: f64,
    pub mean_log_likelihood: f64,
    pub completed_iterations: usize,
    data_fingerprint: DataFingerprint,
    covariance_floor: f64,
}

#[derive(Clone, Debug)]
pub enum RingClusterError {
    InvalidInput { message: String },
    NumericalFailure {
        message: String,
        checkpoint: Option<RingClusterCheckpoint>,
    },
    MonotonicityViolation {
        previous_mean_log_likelihood: f64,
        next_mean_log_likelihood: f64,
        allowed_decrease: f64,
        checkpoint: RingClusterCheckpoint,
    },
    DidNotConverge {
        max_iterations: usize,
        certificate: RingClusterCertificate,
        checkpoint: RingClusterCheckpoint,
    },
}

impl std::fmt::Display for RingClusterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(formatter, "invalid ring-cluster model: {message}"),
            Self::NumericalFailure { message, checkpoint } => write!(
                formatter,
                "ring-cluster numerical failure: {message} (checkpoint iterations {})",
                checkpoint.as_ref().map_or(0, |state| state.completed_iterations)
            ),
            Self::MonotonicityViolation {
                previous_mean_log_likelihood,
                next_mean_log_likelihood,
                allowed_decrease,
                checkpoint,
            } => write!(
                formatter,
                "ring-cluster EM violated monotone ascent at iteration {}: mean log likelihood {:.12e} -> {:.12e} (allowed numerical decrease {:.3e})",
                checkpoint.completed_iterations,
                previous_mean_log_likelihood,
                next_mean_log_likelihood,
                allowed_decrease,
            ),
            Self::DidNotConverge {
                max_iterations,
                certificate,
                checkpoint,
            } => write!(
                formatter,
                "ring-cluster EM did not certify after {max_iterations} additional iterations (total {}): signed mean-log-likelihood gain {:.6e} (roundoff {:.3e}), objective residual {:.6e}/{:.3e}, parameter-map residual {:.6e}/{:.3e}; resume from the carried checkpoint, which is not comparable evidence",
                checkpoint.completed_iterations,
                certificate.mean_log_likelihood_gain,
                certificate.monotonicity_roundoff,
                certificate.objective_residual,
                certificate.objective_tolerance,
                certificate.parameter_residual,
                certificate.parameter_tolerance,
            ),
        }
    }
}

impl std::error::Error for RingClusterError {}

#[derive(Clone, Debug)]
pub struct RingClusterFit {
    weights: Array1<f64>,
    center: [f64; 2],
    radius: f64,
    angles: Array1<f64>,
    variance: f64,
    n_obs: usize,
    log_likelihood: f64,
    iterations: usize,
    certificate: RingClusterCertificate,
    data_fingerprint: DataFingerprint,
}

impl RingClusterFit {
    pub fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }

    pub fn center(&self) -> [f64; 2] {
        self.center
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn angles(&self) -> ArrayView1<'_, f64> {
        self.angles.view()
    }

    pub fn variance(&self) -> f64 {
        self.variance
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn certificate(&self) -> RingClusterCertificate {
        self.certificate
    }

    /// `(k-1)` simplex coordinates + center(2) + log-radius(1) + `k`
    /// component angles + log-variance(1).
    pub fn num_free_parameters(&self) -> usize {
        2 * self.weights.len() + 3
    }

    pub fn component_means(&self) -> Array2<f64> {
        component_means(self.center, self.radius, self.angles.view())
    }

    pub fn per_point_log_density(
        &self,
        data: ArrayView2<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        validate_scoring_data(data)?;
        let (mean_log_likelihood, _, per_point) = expectation(
            data,
            self.weights.view(),
            self.center,
            self.radius,
            self.angles.view(),
            self.variance,
            false,
        )?;
        if !mean_log_likelihood.is_finite() {
            return Err("ring-cluster held-out density is not finite".to_string());
        }
        Ok(per_point)
    }

    /// Rank-aware Laplace negative log evidence using the observed empirical
    /// Fisher information at the certified constrained-mixture fixed point.
    pub fn laplace_negative_log_evidence(
        &self,
        data: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        if data.nrows() != self.n_obs || fingerprint(data) != self.data_fingerprint {
            return Err(
                "ring-cluster Laplace evidence must be evaluated on the exact data whose certified EM fixed point is stored in this fit"
                    .to_string(),
            );
        }
        let information = self.empirical_fisher_information(data)?;
        let parameter_count = self.num_free_parameters();
        let apply_information = |input: &[f64]| -> Vec<f64> {
            let mut output = vec![0.0_f64; parameter_count];
            for row in 0..parameter_count {
                for col in 0..parameter_count {
                    output[row] += information[[row, col]] * input[col];
                }
            }
            output
        };
        let evidence = laplace_evidence(
            EvidenceLogDetSource::Hvp(EvidenceHvpLogDet {
                dim: parameter_count,
                apply: &apply_information,
            }),
            0.0,
            -self.log_likelihood,
            parameter_count as f64,
            0.0,
        );
        if !evidence.is_finite() {
            return Err("ring-cluster Laplace evidence is not finite".to_string());
        }
        Ok(evidence)
    }

    fn empirical_fisher_information(
        &self,
        data: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let (_, responsibilities, _) = expectation(
            data,
            self.weights.view(),
            self.center,
            self.radius,
            self.angles.view(),
            self.variance,
            true,
        )?;
        let responsibilities = responsibilities.ok_or_else(|| {
            "ring-cluster information requested without responsibilities".to_string()
        })?;
        let k = self.weights.len();
        let p = self.num_free_parameters();
        let center_base = k - 1;
        let radius_index = center_base + 2;
        let angle_base = radius_index + 1;
        let variance_index = angle_base + k;
        let means = self.component_means();
        let mut information = Array2::<f64>::zeros((p, p));
        let mut score = vec![0.0_f64; p];
        for row in 0..data.nrows() {
            score.fill(0.0);
            for component in 0..k.saturating_sub(1) {
                score[component] =
                    responsibilities[[row, component]] - self.weights[component];
            }
            let mut variance_score = 0.0_f64;
            for component in 0..k {
                let responsibility = responsibilities[[row, component]];
                let dx = data[[row, 0]] - means[[component, 0]];
                let dy = data[[row, 1]] - means[[component, 1]];
                score[center_base] += responsibility * dx / self.variance;
                score[center_base + 1] += responsibility * dy / self.variance;
                let cosine = self.angles[component].cos();
                let sine = self.angles[component].sin();
                score[radius_index] += responsibility
                    * self.radius
                    * (dx * cosine + dy * sine)
                    / self.variance;
                score[angle_base + component] += responsibility
                    * self.radius
                    * (-dx * sine + dy * cosine)
                    / self.variance;
                variance_score += responsibility
                    * (-1.0 + (dx * dx + dy * dy) / (2.0 * self.variance));
            }
            score[variance_index] = variance_score;
            for a in 0..p {
                for b in 0..=a {
                    information[[a, b]] += score[a] * score[b];
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                information[[b, a]] = information[[a, b]];
            }
        }
        Ok(information)
    }
}

pub fn fit_ring_cluster(
    data: ArrayView2<'_, f64>,
    k: usize,
    config: GaussianMixtureConfig,
) -> Result<RingClusterFit, RingClusterError> {
    validate_problem(data, k, config)?;
    let unconstrained = fit_gaussian_mixture(data, k, config).map_err(|error| {
        RingClusterError::NumericalFailure {
            message: format!("free-mixture initialization failed: {error}"),
            checkpoint: None,
        }
    })?;
    let weights = unconstrained.weights().to_owned();
    let (center, radius) = fit_weighted_circle(
        unconstrained.means(),
        weights.view(),
        None,
        config.parameter_tol,
        config.max_iter,
    )
    .map_err(|message| RingClusterError::NumericalFailure {
        message: format!("initial constrained centroid circle failed: {message}"),
        checkpoint: None,
    })?;
    let angles = angles_from_points(unconstrained.means(), center)?;
    let projected = component_means(center, radius, angles.view());
    let mut variance = 0.0_f64;
    for component in 0..k {
        let covariance = &unconstrained.covariances()[component];
        let dx = unconstrained.means()[[component, 0]] - projected[[component, 0]];
        let dy = unconstrained.means()[[component, 1]] - projected[[component, 1]];
        variance += weights[component]
            * (covariance[[0, 0]] + covariance[[1, 1]] + dx * dx + dy * dy)
            / 2.0;
    }
    variance = variance.max(config.covariance_floor);
    let (mean_log_likelihood, _, _) = expectation(
        data,
        weights.view(),
        center,
        radius,
        angles.view(),
        variance,
        false,
    )
    .map_err(|message| RingClusterError::NumericalFailure {
        message,
        checkpoint: None,
    })?;
    run_em(
        data,
        config,
        RingClusterCheckpoint {
            weights,
            center,
            radius,
            angles,
            variance,
            mean_log_likelihood,
            completed_iterations: 0,
            data_fingerprint: fingerprint(data),
            covariance_floor: config.covariance_floor,
        },
    )
}

pub fn resume_ring_cluster(
    data: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
    checkpoint: RingClusterCheckpoint,
) -> Result<RingClusterFit, RingClusterError> {
    validate_problem(data, checkpoint.weights.len(), config)?;
    validate_checkpoint(data, config, &checkpoint)?;
    run_em(data, config, checkpoint)
}

fn run_em(
    data: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
    mut current: RingClusterCheckpoint,
) -> Result<RingClusterFit, RingClusterError> {
    let mut last_certificate = RingClusterCertificate {
        mean_log_likelihood: current.mean_log_likelihood,
        mean_log_likelihood_gain: f64::INFINITY,
        monotonicity_roundoff: 0.0,
        objective_residual: f64::INFINITY,
        objective_tolerance: config.loglik_tol,
        parameter_residual: f64::INFINITY,
        parameter_tolerance: config.parameter_tol,
    };
    for _ in 0..config.max_iter {
        let (_, responsibilities, _) = expectation(
            data,
            current.weights.view(),
            current.center,
            current.radius,
            current.angles.view(),
            current.variance,
            true,
        )
        .map_err(|message| RingClusterError::NumericalFailure {
            message,
            checkpoint: Some(current.clone()),
        })?;
        let responsibilities = responsibilities.ok_or_else(|| {
            RingClusterError::NumericalFailure {
                message: "ring-cluster E-step did not return responsibilities".to_string(),
                checkpoint: Some(current.clone()),
            }
        })?;
        let mut proposal = maximization(data, responsibilities.view(), &current, config)
            .map_err(|message| RingClusterError::NumericalFailure {
                message,
                checkpoint: Some(current.clone()),
            })?;
        let (next_mean_log_likelihood, _, _) = expectation(
            data,
            proposal.weights.view(),
            proposal.center,
            proposal.radius,
            proposal.angles.view(),
            proposal.variance,
            false,
        )
        .map_err(|message| RingClusterError::NumericalFailure {
            message,
            checkpoint: Some(current.clone()),
        })?;
        proposal.mean_log_likelihood = next_mean_log_likelihood;
        proposal.completed_iterations = current.completed_iterations + 1;

        let gain = next_mean_log_likelihood - current.mean_log_likelihood;
        let roundoff = likelihood_roundoff(
            data.nrows(),
            current.weights.len(),
            current.mean_log_likelihood,
            next_mean_log_likelihood,
        );
        if gain < -roundoff {
            return Err(RingClusterError::MonotonicityViolation {
                previous_mean_log_likelihood: current.mean_log_likelihood,
                next_mean_log_likelihood,
                allowed_decrease: roundoff,
                checkpoint: current,
            });
        }
        let objective_residual = gain.abs() / (1.0 + current.mean_log_likelihood.abs());
        let parameter_residual = parameter_map_residual(&current, &proposal);
        last_certificate = RingClusterCertificate {
            mean_log_likelihood: current.mean_log_likelihood,
            mean_log_likelihood_gain: gain,
            monotonicity_roundoff: roundoff,
            objective_residual,
            objective_tolerance: config.loglik_tol,
            parameter_residual,
            parameter_tolerance: config.parameter_tol,
        };
        if objective_residual <= config.loglik_tol
            && parameter_residual <= config.parameter_tol
        {
            return Ok(RingClusterFit {
                weights: current.weights,
                center: current.center,
                radius: current.radius,
                angles: current.angles,
                variance: current.variance,
                n_obs: data.nrows(),
                log_likelihood: current.mean_log_likelihood * data.nrows() as f64,
                iterations: current.completed_iterations,
                certificate: last_certificate,
                data_fingerprint: current.data_fingerprint,
            });
        }
        current = proposal;
    }
    Err(RingClusterError::DidNotConverge {
        max_iterations: config.max_iter,
        certificate: last_certificate,
        checkpoint: current,
    })
}

fn maximization(
    data: ArrayView2<'_, f64>,
    responsibilities: ArrayView2<'_, f64>,
    current: &RingClusterCheckpoint,
    config: GaussianMixtureConfig,
) -> Result<RingClusterCheckpoint, String> {
    let n = data.nrows();
    let k = responsibilities.ncols();
    let mut masses = Array1::<f64>::zeros(k);
    let mut centroids = Array2::<f64>::zeros((k, 2));
    for row in 0..n {
        for component in 0..k {
            let responsibility = responsibilities[[row, component]];
            masses[component] += responsibility;
            centroids[[component, 0]] += responsibility * data[[row, 0]];
            centroids[[component, 1]] += responsibility * data[[row, 1]];
        }
    }
    for component in 0..k {
        if !(masses[component].is_finite() && masses[component] > 0.0) {
            return Err(format!(
                "ring-cluster component {component} has no finite positive posterior mass"
            ));
        }
        centroids[[component, 0]] /= masses[component];
        centroids[[component, 1]] /= masses[component];
    }
    let (center, radius) = fit_weighted_circle(
        centroids.view(),
        masses.view(),
        Some((current.center, current.radius)),
        config.parameter_tol,
        config.max_iter,
    )?;
    let angles = angles_from_points(centroids.view(), center)
        .map_err(|error| error.to_string())?;
    let means = component_means(center, radius, angles.view());
    let mut squared_error = 0.0_f64;
    for row in 0..n {
        for component in 0..k {
            let dx = data[[row, 0]] - means[[component, 0]];
            let dy = data[[row, 1]] - means[[component, 1]];
            squared_error += responsibilities[[row, component]] * (dx * dx + dy * dy);
        }
    }
    let variance = (squared_error / (2.0 * n as f64)).max(config.covariance_floor);
    let weights = masses / n as f64;
    Ok(RingClusterCheckpoint {
        weights,
        center,
        radius,
        angles,
        variance,
        mean_log_likelihood: current.mean_log_likelihood,
        completed_iterations: current.completed_iterations,
        data_fingerprint: current.data_fingerprint,
        covariance_floor: current.covariance_floor,
    })
}

fn expectation(
    data: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    center: [f64; 2],
    radius: f64,
    angles: ArrayView1<'_, f64>,
    variance: f64,
    want_responsibilities: bool,
) -> Result<(f64, Option<Array2<f64>>, Array1<f64>), String> {
    validate_scoring_data(data)?;
    if weights.len() != angles.len() || weights.is_empty() {
        return Err("ring-cluster weights/angles must have equal positive length".to_string());
    }
    if !(radius.is_finite() && radius > 0.0 && variance.is_finite() && variance > 0.0) {
        return Err("ring-cluster radius and variance must be finite and positive".to_string());
    }
    let weight_sum = weights.sum();
    if !weight_sum.is_finite()
        || (weight_sum - 1.0).abs() > f64::EPSILON.sqrt() * weights.len() as f64
        || weights.iter().any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err("ring-cluster weights must be a finite positive simplex".to_string());
    }
    let means = component_means(center, radius, angles);
    let k = weights.len();
    let mut responsibilities = want_responsibilities.then(|| Array2::<f64>::zeros((data.nrows(), k)));
    let mut per_point = Array1::<f64>::zeros(data.nrows());
    let log_normalizer = -(LOG_TAU + variance.ln());
    let mut terms = vec![0.0_f64; k];
    for row in 0..data.nrows() {
        let mut maximum = f64::NEG_INFINITY;
        for component in 0..k {
            let dx = data[[row, 0]] - means[[component, 0]];
            let dy = data[[row, 1]] - means[[component, 1]];
            terms[component] = weights[component].ln() + log_normalizer
                - (dx * dx + dy * dy) / (2.0 * variance);
            maximum = maximum.max(terms[component]);
        }
        if !maximum.is_finite() {
            return Err(format!("ring-cluster row {row} has no finite component density"));
        }
        let sum_exp = terms.iter().map(|term| (*term - maximum).exp()).sum::<f64>();
        let log_density = maximum + sum_exp.ln();
        if !log_density.is_finite() {
            return Err(format!("ring-cluster row {row} log density is not finite"));
        }
        per_point[row] = log_density;
        if let Some(matrix) = responsibilities.as_mut() {
            for component in 0..k {
                matrix[[row, component]] = (terms[component] - log_density).exp();
            }
        }
    }
    let mean_log_likelihood = per_point.sum() / data.nrows() as f64;
    Ok((mean_log_likelihood, responsibilities, per_point))
}

fn fit_weighted_circle(
    points: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    initial: Option<([f64; 2], f64)>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<([f64; 2], f64), String> {
    if points.ncols() != 2 || points.nrows() < 3 || points.nrows() != weights.len() {
        return Err("weighted circle fit requires matching k×2 points and k>=3 weights".to_string());
    }
    let (mut center, mut radius) = match initial {
        Some(value) => value,
        None => algebraic_circle_seed(points, weights)?,
    };
    if !(radius.is_finite() && radius > 0.0 && center.iter().all(|value| value.is_finite())) {
        return Err("weighted circle initial state is not finite and positive".to_string());
    }
    let mut objective = circle_objective(points, weights, center, radius)?;
    for _ in 0..max_iterations {
        let mut gradient = [0.0_f64; 3];
        let mut normal = [[0.0_f64; 3]; 3];
        let mut gradient_scale = 1.0_f64;
        for row in 0..points.nrows() {
            let dx = center[0] - points[[row, 0]];
            let dy = center[1] - points[[row, 1]];
            let distance = dx.hypot(dy);
            if !(distance.is_finite() && distance > 0.0) {
                return Err(format!(
                    "weighted circle center coincides with component centroid {row}"
                ));
            }
            let residual = distance - radius;
            let jacobian = [dx / distance, dy / distance, -1.0];
            let weight = weights[row];
            gradient_scale += weight * (distance + radius + residual.abs());
            for a in 0..3 {
                gradient[a] += weight * jacobian[a] * residual;
                for b in 0..3 {
                    normal[a][b] += weight * jacobian[a] * jacobian[b];
                }
            }
        }
        let gradient_norm = gradient.iter().map(|value| value * value).sum::<f64>().sqrt();
        if gradient_norm <= tolerance * gradient_scale {
            return Ok((center, radius));
        }
        let step = solve_three_by_three(normal, gradient.map(|value| -value))?;
        let mut fraction = 1.0_f64;
        let accepted = loop {
            let candidate_center = [
                center[0] + fraction * step[0],
                center[1] + fraction * step[1],
            ];
            let candidate_radius = radius + fraction * step[2];
            if candidate_radius.is_finite() && candidate_radius > 0.0 {
                let candidate_objective =
                    circle_objective(points, weights, candidate_center, candidate_radius)?;
                if candidate_objective < objective {
                    break Some((candidate_center, candidate_radius, candidate_objective));
                }
            }
            let next = fraction * 0.5;
            if next == 0.0 || next == fraction {
                break None;
            }
            fraction = next;
        };
        let Some((next_center, next_radius, next_objective)) = accepted else {
            return Err(format!(
                "weighted circle solve stagnated away from stationarity (gradient norm {gradient_norm:.6e}, tolerance {:.6e})",
                tolerance * gradient_scale,
            ));
        };
        center = next_center;
        radius = next_radius;
        objective = next_objective;
    }
    Err(format!(
        "weighted circle solve did not certify within {max_iterations} iterations"
    ))
}

fn algebraic_circle_seed(
    points: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<([f64; 2], f64), String> {
    let mut normal = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for row in 0..points.nrows() {
        let x = points[[row, 0]];
        let y = points[[row, 1]];
        let design = [2.0 * x, 2.0 * y, 1.0];
        let target = x * x + y * y;
        for a in 0..3 {
            rhs[a] += weights[row] * design[a] * target;
            for b in 0..3 {
                normal[a][b] += weights[row] * design[a] * design[b];
            }
        }
    }
    let solution = solve_three_by_three(normal, rhs)?;
    let radius_squared = solution[2] + solution[0] * solution[0] + solution[1] * solution[1];
    if !(radius_squared.is_finite() && radius_squared > 0.0) {
        return Err("algebraic circle seed has no positive finite radius".to_string());
    }
    Ok(([solution[0], solution[1]], radius_squared.sqrt()))
}

fn circle_objective(
    points: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    center: [f64; 2],
    radius: f64,
) -> Result<f64, String> {
    let mut objective = 0.0_f64;
    for row in 0..points.nrows() {
        let distance = (center[0] - points[[row, 0]])
            .hypot(center[1] - points[[row, 1]]);
        let residual = distance - radius;
        objective += 0.5 * weights[row] * residual * residual;
    }
    if !objective.is_finite() {
        return Err("weighted circle objective is not finite".to_string());
    }
    Ok(objective)
}

fn solve_three_by_three(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Result<[f64; 3], String> {
    let determinant = determinant_three(matrix);
    let scale = matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    if !determinant.is_finite()
        || determinant.abs() <= f64::EPSILON.sqrt() * scale * scale * scale
    {
        return Err("weighted circle normal matrix is rank deficient".to_string());
    }
    let mut solution = [0.0_f64; 3];
    for column in 0..3 {
        let mut replaced = matrix;
        for row in 0..3 {
            replaced[row][column] = rhs[row];
        }
        solution[column] = determinant_three(replaced) / determinant;
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("weighted circle linear solve is not finite".to_string());
    }
    Ok(solution)
}

fn determinant_three(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn angles_from_points(
    points: ArrayView2<'_, f64>,
    center: [f64; 2],
) -> Result<Array1<f64>, RingClusterError> {
    let mut angles = Array1::<f64>::zeros(points.nrows());
    for row in 0..points.nrows() {
        let dx = points[[row, 0]] - center[0];
        let dy = points[[row, 1]] - center[1];
        if !(dx.is_finite() && dy.is_finite() && dx.hypot(dy) > 0.0) {
            return Err(RingClusterError::NumericalFailure {
                message: format!("ring-cluster centroid {row} does not define an angle"),
                checkpoint: None,
            });
        }
        angles[row] = dy.atan2(dx);
    }
    Ok(angles)
}

fn component_means(
    center: [f64; 2],
    radius: f64,
    angles: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let mut means = Array2::<f64>::zeros((angles.len(), 2));
    for component in 0..angles.len() {
        means[[component, 0]] = center[0] + radius * angles[component].cos();
        means[[component, 1]] = center[1] + radius * angles[component].sin();
    }
    means
}

fn parameter_map_residual(
    current: &RingClusterCheckpoint,
    proposal: &RingClusterCheckpoint,
) -> f64 {
    let current_means = component_means(current.center, current.radius, current.angles.view());
    let proposal_means = component_means(proposal.center, proposal.radius, proposal.angles.view());
    let mut residual = 0.0_f64;
    for component in 0..current.weights.len() {
        residual = residual.max(
            (proposal.weights[component] - current.weights[component]).abs()
                / (1.0 + current.weights[component].abs()),
        );
        for dimension in 0..2 {
            residual = residual.max(
                (proposal_means[[component, dimension]]
                    - current_means[[component, dimension]])
                    .abs()
                    / (1.0 + current_means[[component, dimension]].abs()),
            );
        }
    }
    residual.max(
        (proposal.variance - current.variance).abs() / (1.0 + current.variance.abs()),
    )
}

fn validate_problem(
    data: ArrayView2<'_, f64>,
    k: usize,
    config: GaussianMixtureConfig,
) -> Result<(), RingClusterError> {
    validate_scoring_data(data)
        .map_err(|message| RingClusterError::InvalidInput { message })?;
    if k < 3 || k > data.nrows() {
        return Err(RingClusterError::InvalidInput {
            message: format!("ring-cluster order must satisfy 3 <= k <= n; got k={k}"),
        });
    }
    if config.max_iter == 0
        || !(config.loglik_tol.is_finite() && config.loglik_tol > 0.0)
        || !(config.parameter_tol.is_finite() && config.parameter_tol > 0.0)
        || !(config.covariance_floor.is_finite() && config.covariance_floor > 0.0)
    {
        return Err(RingClusterError::InvalidInput {
            message: "ring-cluster convergence controls must be finite and positive".to_string(),
        });
    }
    Ok(())
}

fn validate_scoring_data(data: ArrayView2<'_, f64>) -> Result<(), String> {
    if data.nrows() == 0 || data.ncols() != 2 {
        return Err(format!(
            "ring-cluster data must be a non-empty n×2 matrix; got {:?}",
            data.dim()
        ));
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err("ring-cluster data must be finite".to_string());
    }
    Ok(())
}

fn validate_checkpoint(
    data: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
    checkpoint: &RingClusterCheckpoint,
) -> Result<(), RingClusterError> {
    if checkpoint.data_fingerprint != fingerprint(data) {
        return Err(RingClusterError::InvalidInput {
            message: "ring-cluster checkpoint belongs to different data".to_string(),
        });
    }
    if checkpoint.covariance_floor.to_bits() != config.covariance_floor.to_bits() {
        return Err(RingClusterError::InvalidInput {
            message: "ring-cluster checkpoint covariance constraint differs from config"
                .to_string(),
        });
    }
    expectation(
        data,
        checkpoint.weights.view(),
        checkpoint.center,
        checkpoint.radius,
        checkpoint.angles.view(),
        checkpoint.variance,
        false,
    )
    .map_err(|message| RingClusterError::InvalidInput { message })?;
    Ok(())
}

fn fingerprint(data: ArrayView2<'_, f64>) -> DataFingerprint {
    let mut lane_a = 0x243F_6A88_85A3_08D3_u64;
    let mut lane_b = 0x1319_8A2E_0370_7344_u64;
    for (index, value) in data.iter().enumerate() {
        let bits = value.to_bits();
        lane_a ^= bits.wrapping_add(index as u64).rotate_left((index % 63) as u32);
        lane_a = lane_a.wrapping_mul(0x9E37_79B1_85EB_CA87);
        lane_b ^= bits.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        lane_b = lane_b.rotate_left(29).wrapping_add(lane_a);
    }
    DataFingerprint {
        nrows: data.nrows(),
        ncols: data.ncols(),
        lane_a,
        lane_b,
    }
}

fn likelihood_roundoff(
    n: usize,
    k: usize,
    previous: f64,
    next: f64,
) -> f64 {
    let operations = n.saturating_mul(k).saturating_mul(8).max(1) as f64;
    let product = operations * f64::EPSILON;
    let gamma = if product < 1.0 {
        product / (1.0 - product)
    } else {
        f64::EPSILON.sqrt()
    };
    gamma * (1.0 + previous.abs() + next.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clustered_ring(k: usize, per_cluster: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((k * per_cluster, 2));
        for component in 0..k {
            let angle = std::f64::consts::TAU * component as f64 / k as f64;
            for within in 0..per_cluster {
                let row = component * per_cluster + within;
                let offset = (within as f64 - (per_cluster as f64 - 1.0) / 2.0)
                    / per_cluster as f64;
                data[[row, 0]] = 1.25 + 3.0 * angle.cos() + 0.08 * offset;
                data[[row, 1]] = -0.75 + 3.0 * angle.sin() - 0.05 * offset;
            }
        }
        data
    }

    #[test]
    fn ring_cluster_certifies_and_prices_fewer_parameters_2262() {
        let data = clustered_ring(7, 32);
        let fit = fit_ring_cluster(data.view(), 7, GaussianMixtureConfig::default())
            .expect("ring-cluster fit");
        assert_eq!(fit.num_free_parameters(), 17);
        assert!(fit.radius() > 2.9 && fit.radius() < 3.1, "radius={}", fit.radius());
        assert!(fit.certificate().objective_residual <= fit.certificate().objective_tolerance);
        assert!(fit.certificate().parameter_residual <= fit.certificate().parameter_tolerance);
        assert!(fit.laplace_negative_log_evidence(data.view()).unwrap().is_finite());
    }

    #[test]
    fn ring_cluster_exhaustion_is_typed_and_resumable_2262() {
        let data = clustered_ring(5, 24);
        let mut short = GaussianMixtureConfig::default();
        short.max_iter = 1;
        let error = fit_ring_cluster(data.view(), 5, short).expect_err("one iteration exhausts");
        let checkpoint = match error {
            RingClusterError::DidNotConverge { checkpoint, .. } => checkpoint,
            other => panic!("expected typed exhaustion, got {other}"),
        };
        let resumed = resume_ring_cluster(data.view(), GaussianMixtureConfig::default(), checkpoint)
            .expect("resume reaches fixed point");
        assert!(resumed.certificate().parameter_residual <= resumed.certificate().parameter_tolerance);
    }
}
