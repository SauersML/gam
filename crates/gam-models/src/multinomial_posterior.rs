//! Deterministic posterior moments for logistic-normal softmax probabilities.
//!
//! A reference-coded multinomial model has `M = K - 1` active logits.  At one
//! prediction row the Laplace posterior induces
//!
//! ```text
//! eta ~ Normal(mu, V),
//! p(eta) = softmax(eta_0, ..., eta_{M-1}, 0).
//! ```
//!
//! This module computes `E[p]` and `Cov(p)` rather than the plug-in quantity
//! `softmax(E[eta])`.  The binary case is reduced to the controlled scalar
//! logistic-normal evaluator in `gam-solve`.  For `K > 2`, the covariance is
//! eigendecomposed and quadrature is performed only over its positive range.
//! Successive Smolyak levels built from odd-order Gauss-Hermite rules provide a
//! deterministic error check.  Failure to establish the requested tolerance is
//! an error; there is deliberately no Monte Carlo or plug-in fallback.

use crate::model_types::EstimationError;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Backward-error multiplier used when deciding whether a symmetric covariance
/// eigenvalue is negative beyond floating-point eigensolver roundoff.
///
/// This is not a variance jitter: the input matrix is never modified by adding
/// a diagonal ridge.  Eigenvalues below `-tol` are rejected, while values whose
/// magnitude is within the backward-error envelope are treated as numerical
/// zero.
const PSD_BACKWARD_ERROR_MULTIPLIER: f64 = 16.0;

/// Floating-point summation envelope for the signed Smolyak combination.
const SUMMATION_ROUNDOFF_MULTIPLIER: f64 = 16.0;

/// Explicit accuracy and work controls for multinomial posterior integration.
///
/// The production default is explicit through [`Default`] and is carried by
/// the prediction request into this kernel. `minimum_sparse_level >= 1`
/// guarantees at least one comparison against a preceding Smolyak level.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialPosteriorIntegrationControl {
    /// Per raw moment absolute tolerance.  Raw moments comprise every `E[p_c]`
    /// and `E[p_c p_d]` for `c <= d`.
    pub absolute_tolerance: f64,
    /// Per raw moment relative tolerance.
    pub relative_tolerance: f64,
    /// Earliest Smolyak refinement level that may certify convergence.
    pub minimum_sparse_level: usize,
    /// Last Smolyak refinement level attempted.
    pub maximum_sparse_level: usize,
    /// Maximum total integrand evaluations across all attempted levels.
    pub maximum_function_evaluations: usize,
}

impl Default for MultinomialPosteriorIntegrationControl {
    fn default() -> Self {
        // sqrt(machine epsilon) is the natural accuracy target for a nonlinear
        // transform of a covariance estimated in double precision: asking for
        // substantially more would certify quadrature noise below the input's
        // own numerical resolution. Three sparse levels are required before a
        // result may certify; level eight reaches the 17-point one-dimensional
        // Gauss-Hermite rule. The streaming evaluator never stores the node
        // set, and the evaluation ceiling bounds work on high-rank rows.
        let tolerance = f64::EPSILON.sqrt();
        Self {
            absolute_tolerance: tolerance,
            relative_tolerance: tolerance,
            minimum_sparse_level: 2,
            maximum_sparse_level: 8,
            maximum_function_evaluations: 2_000_000,
        }
    }
}

/// Integrated posterior means and marginal standard deviations for every row
/// of a multinomial prediction design.
#[derive(Clone, Debug)]
pub struct MultinomialPosteriorRowMoments {
    pub class_mean: Array2<f64>,
    pub class_standard_deviation: Array2<f64>,
}

/// Integrate the logistic-normal posterior induced by a coefficient mode and
/// its full joint covariance over every design row.
///
/// Coefficients have shape `(P, M)`, covariance has block-major shape
/// `(P*M, P*M)`, and `design` has shape `(N, P)`. For row `x`, this constructs
/// `mu_a = x' beta_a` and `V_ab = x' Sigma_ab x`, then delegates to the
/// controlled one-row integrator. Cross-class covariance blocks are retained.
pub fn integrate_multinomial_design_moments(
    coefficients: ArrayView2<'_, f64>,
    coefficient_covariance: ArrayView2<'_, f64>,
    design: ArrayView2<'_, f64>,
    control: &MultinomialPosteriorIntegrationControl,
) -> Result<MultinomialPosteriorRowMoments, EstimationError> {
    let (p, m) = coefficients.dim();
    if p == 0 || m == 0 {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior prediction needs nonempty coefficients, got {p}x{m}"
        )));
    }
    if design.ncols() != p {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior prediction design has {} columns, expected {p}",
            design.ncols()
        )));
    }
    let d = p.checked_mul(m).ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial posterior prediction coefficient dimension overflowed usize".to_string(),
        )
    })?;
    if coefficient_covariance.dim() != (d, d) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior prediction covariance shape {:?} does not match (P*M, P*M) = ({d}, {d})",
            coefficient_covariance.dim()
        )));
    }

    let n = design.nrows();
    let k = m + 1;
    let mut class_mean = Array2::<f64>::zeros((n, k));
    let mut class_standard_deviation = Array2::<f64>::zeros((n, k));
    let mut active_mean = Array1::<f64>::zeros(m);
    let mut active_covariance = Array2::<f64>::zeros((m, m));
    for row in 0..n {
        let x = design.row(row);
        for a in 0..m {
            active_mean[a] = x.dot(&coefficients.column(a));
        }
        for a in 0..m {
            for b in 0..m {
                let mut value = 0.0_f64;
                let a_base = a * p;
                let b_base = b * p;
                for i in 0..p {
                    let xi = x[i];
                    if xi == 0.0 {
                        continue;
                    }
                    let mut row_product = 0.0_f64;
                    for j in 0..p {
                        row_product += coefficient_covariance[[a_base + i, b_base + j]] * x[j];
                    }
                    value += xi * row_product;
                }
                active_covariance[[a, b]] = value;
            }
        }
        let moments = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            control,
        )?;
        class_mean.row_mut(row).assign(&moments.class_mean);
        class_standard_deviation
            .row_mut(row)
            .assign(&moments.class_standard_deviation);
    }
    Ok(MultinomialPosteriorRowMoments {
        class_mean,
        class_standard_deviation,
    })
}

impl MultinomialPosteriorIntegrationControl {
    fn validate(&self) -> Result<(), EstimationError> {
        if !(self.absolute_tolerance.is_finite() && self.absolute_tolerance >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integration absolute_tolerance must be finite and >= 0, got {}",
                self.absolute_tolerance
            )));
        }
        if !(self.relative_tolerance.is_finite() && self.relative_tolerance >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integration relative_tolerance must be finite and >= 0, got {}",
                self.relative_tolerance
            )));
        }
        if self.absolute_tolerance == 0.0 && self.relative_tolerance == 0.0 {
            return Err(EstimationError::InvalidInput(
                "multinomial posterior integration requires a positive absolute or relative tolerance"
                    .to_string(),
            ));
        }
        if self.minimum_sparse_level == 0 {
            return Err(EstimationError::InvalidInput(
                "multinomial posterior integration minimum_sparse_level must be >= 1 so a level difference exists"
                    .to_string(),
            ));
        }
        if self.maximum_sparse_level < self.minimum_sparse_level {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integration maximum_sparse_level ({}) is below minimum_sparse_level ({})",
                self.maximum_sparse_level, self.minimum_sparse_level
            )));
        }
        if self.maximum_function_evaluations == 0 {
            return Err(EstimationError::InvalidInput(
                "multinomial posterior integration maximum_function_evaluations must be positive"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Integrated class-probability moments for one prediction row.
///
/// `class_covariance` includes the reference class and is singular in the
/// all-ones direction, as required by `sum_c p_c = 1`.  A value of this type is
/// only constructed after the requested level-difference certificate succeeds.
#[derive(Clone, Debug)]
pub struct MultinomialPosteriorMoments {
    /// `E[p_c]`, length `K`, including the reference class last.
    pub class_mean: Array1<f64>,
    /// `Cov(p_c, p_d)`, shape `(K, K)`.
    pub class_covariance: Array2<f64>,
    /// Marginal posterior standard deviations `sqrt(Var(p_c))`.
    pub class_standard_deviation: Array1<f64>,
    /// Positive numerical rank of the active-logit covariance.
    pub latent_rank: usize,
    /// Smolyak level that certified convergence.  `None` denotes an exact
    /// binary reduction or an exact point-mass covariance.
    pub sparse_level: Option<usize>,
    /// Total softmax evaluations across all attempted sparse levels.
    pub function_evaluations: usize,
    /// Largest absolute difference among raw first/second moments at the final
    /// two sparse levels.  Zero on the exact binary and point-mass paths.
    pub max_raw_moment_level_difference: f64,
    /// Bound used for positive covariance eigenmodes discarded inside the
    /// eigensolver backward-error envelope.  Such modes are discarded only
    /// when this bound fits inside the requested absolute tolerance.
    pub covariance_range_projection_bound: f64,
}

/// Integrate reference-coded logistic-normal softmax moments for one row.
///
/// `active_mean` has length `M = K - 1`; `active_covariance` must be a finite,
/// symmetric positive-semidefinite `(M, M)` matrix in the same active-class
/// order.  The returned arrays include the implicit reference class as their
/// final entry.
pub fn integrate_logistic_normal_softmax_moments(
    active_mean: ArrayView1<'_, f64>,
    active_covariance: ArrayView2<'_, f64>,
    control: &MultinomialPosteriorIntegrationControl,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    control.validate()?;
    validate_inputs(active_mean, active_covariance)?;

    let mean = active_mean.to_vec();
    let m = mean.len();
    if m == 1 {
        return integrate_binary(mean[0], active_covariance[[0, 0]]);
    }

    let maximum_covariance_entry = active_covariance
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    if maximum_covariance_entry == 0.0 {
        return point_mass_moments(&mean);
    }

    let projected = project_active_covariance(active_covariance, control.absolute_tolerance)?;
    if projected.factor.ncols() == 0 {
        // This arm is reachable only when every positive eigenmode lies inside
        // the eigensolver backward-error envelope and its explicit probability
        // bound fits within the caller's tolerance.  It is therefore a
        // certified point-mass approximation, not a silent plug-in fallback.
        let mut out = point_mass_moments(&mean)?;
        out.covariance_range_projection_bound = projected.projection_bound;
        return Ok(out);
    }

    integrate_general(&mean, &projected, control)
}

fn validate_inputs(
    active_mean: ArrayView1<'_, f64>,
    active_covariance: ArrayView2<'_, f64>,
) -> Result<(), EstimationError> {
    let m = active_mean.len();
    if m == 0 {
        return Err(EstimationError::InvalidInput(
            "multinomial posterior integration needs at least one active logit (K >= 2)"
                .to_string(),
        ));
    }
    if active_covariance.dim() != (m, m) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration covariance shape {:?} does not match active mean length {m}",
            active_covariance.dim()
        )));
    }
    if let Some((index, value)) = active_mean
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration active_mean[{index}] is non-finite: {value}"
        )));
    }
    if let Some(((row, column), value)) = active_covariance
        .indexed_iter()
        .map(|(index, &value)| (index, value))
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration covariance[{row},{column}] is non-finite: {value}"
        )));
    }

    let scale = active_covariance
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let symmetry_tolerance = covariance_roundoff_tolerance(scale, m);
    let mut maximum_asymmetry = 0.0_f64;
    for row in 0..m {
        for column in (row + 1)..m {
            maximum_asymmetry = maximum_asymmetry
                .max((active_covariance[[row, column]] - active_covariance[[column, row]]).abs());
        }
    }
    if maximum_asymmetry > symmetry_tolerance {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integration covariance is not symmetric: max asymmetry {maximum_asymmetry:.6e} exceeds backward-error tolerance {symmetry_tolerance:.6e}"
        )));
    }
    Ok(())
}

fn covariance_roundoff_tolerance(scale: f64, dimension: usize) -> f64 {
    PSD_BACKWARD_ERROR_MULTIPLIER * f64::EPSILON * (dimension.max(1) as f64) * scale
}

fn integrate_binary(
    active_mean: f64,
    active_variance: f64,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    if active_variance < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "binary logistic-normal variance must be non-negative, got {active_variance:.6e}"
        )));
    }
    let sigma = active_variance.sqrt();
    let (probability_mean, mean_logistic_slope) =
        gam_solve::quadrature::logit_posterior_meanwith_deriv(active_mean, sigma)?;

    // sigmoid'(eta) = p(1-p) = p-p^2, hence
    // E[p^2] = E[p] - d/dmu E[p].  This supplies the binary probability
    // variance from the same controlled scalar integral without a second
    // numerical approximation.
    let probability_second_moment = probability_mean - mean_logistic_slope;
    let variance = (probability_second_moment - probability_mean * probability_mean).max(0.0);
    let reference_mean = 1.0 - probability_mean;

    let class_mean = Array1::from_vec(vec![probability_mean, reference_mean]);
    let class_covariance =
        Array2::from_shape_vec((2, 2), vec![variance, -variance, -variance, variance]).map_err(
            |error| {
                EstimationError::InvalidInput(format!(
                    "binary logistic-normal covariance construction failed: {error}"
                ))
            },
        )?;
    let standard_deviation = variance.sqrt();
    Ok(MultinomialPosteriorMoments {
        class_mean,
        class_covariance,
        class_standard_deviation: Array1::from_vec(vec![standard_deviation, standard_deviation]),
        latent_rank: if active_variance > 0.0 { 1 } else { 0 },
        sparse_level: None,
        function_evaluations: 0,
        max_raw_moment_level_difference: 0.0,
        covariance_range_projection_bound: 0.0,
    })
}

fn point_mass_moments(active_mean: &[f64]) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let class_mean = Array1::from_vec(softmax_with_reference(active_mean)?);
    let k = class_mean.len();
    Ok(MultinomialPosteriorMoments {
        class_mean,
        class_covariance: Array2::zeros((k, k)),
        class_standard_deviation: Array1::zeros(k),
        latent_rank: 0,
        sparse_level: None,
        function_evaluations: 1,
        max_raw_moment_level_difference: 0.0,
        covariance_range_projection_bound: 0.0,
    })
}

struct ProjectedGaussian {
    /// `factor factor^T` is the retained active-logit covariance.
    factor: Array2<f64>,
    projection_bound: f64,
}

fn project_active_covariance(
    covariance: ArrayView2<'_, f64>,
    absolute_tolerance: f64,
) -> Result<ProjectedGaussian, EstimationError> {
    let m = covariance.nrows();
    let symmetric = (&covariance.to_owned() + &covariance.t().to_owned()) * 0.5;
    let (eigenvalues, eigenvectors) = symmetric.eigh(faer::Side::Lower).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior covariance eigendecomposition failed: {error}"
        ))
    })?;
    let eigenvalue_scale = eigenvalues
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    let tolerance = covariance_roundoff_tolerance(eigenvalue_scale, m);
    let minimum_eigenvalue = eigenvalues
        .iter()
        .fold(f64::INFINITY, |minimum, &value| minimum.min(value));
    if minimum_eigenvalue < -tolerance {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior active-logit covariance is not positive semidefinite: minimum eigenvalue {minimum_eigenvalue:.6e} is below -{tolerance:.6e} (scale {eigenvalue_scale:.6e})"
        )));
    }

    let small_positive_trace: f64 = eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > 0.0 && *value <= tolerance)
        .sum();
    // For every softmax raw moment used here, the Euclidean gradient norm is
    // at most one.  Coupling the retained Gaussian with the full Gaussian gives
    // |E f(full)-E f(retained)| <= E||delta|| <= sqrt(tr(V_discarded)).
    let candidate_projection_bound = small_positive_trace.sqrt();
    let discard_small_positive = candidate_projection_bound <= absolute_tolerance;

    let retained: Vec<(usize, f64)> = eigenvalues
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value > 0.0 && (!discard_small_positive || *value > tolerance))
        .collect();
    let projection_bound = if discard_small_positive {
        candidate_projection_bound
    } else {
        0.0
    };
    let mut factor = Array2::<f64>::zeros((m, retained.len()));
    for (output_column, (eigenvector_column, eigenvalue)) in retained.into_iter().enumerate() {
        let scale = eigenvalue.sqrt();
        for row in 0..m {
            factor[[row, output_column]] = eigenvectors[[row, eigenvector_column]] * scale;
        }
    }
    Ok(ProjectedGaussian {
        factor,
        projection_bound,
    })
}

fn integrate_general(
    active_mean: &[f64],
    projected: &ProjectedGaussian,
    control: &MultinomialPosteriorIntegrationControl,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let rank = projected.factor.ncols();
    let k = active_mean.len() + 1;
    let mut rules = Vec::<GaussHermiteRule>::new();
    let mut previous: Option<Vec<f64>> = None;
    let mut total_evaluations = 0usize;
    let mut last_max_difference = f64::INFINITY;
    let mut last_max_normalized_error = f64::INFINITY;

    for level in 0..=control.maximum_sparse_level {
        let required_rule_count = level.checked_add(1).ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior sparse level overflowed usize".to_string(),
            )
        })?;
        while rules.len() < required_rule_count {
            let rule_index = rules.len() + 1;
            rules.push(gauss_hermite_rule(rule_index)?);
        }

        let evaluation = evaluate_smolyak_level(
            active_mean,
            projected,
            &rules,
            level,
            k,
            &mut total_evaluations,
            control.maximum_function_evaluations,
            control.absolute_tolerance,
        )?;
        let current = evaluation.raw_moments;

        if let Some(previous_moments) = previous.as_ref() {
            let mut certified = level >= control.minimum_sparse_level;
            let mut maximum_difference = 0.0_f64;
            let mut maximum_normalized_error = 0.0_f64;
            for (&new_value, &old_value) in current.iter().zip(previous_moments.iter()) {
                let difference = (new_value - old_value).abs();
                maximum_difference = maximum_difference.max(difference);
                let tolerance = control.absolute_tolerance
                    + control.relative_tolerance * new_value.abs().max(old_value.abs());
                let controlled_error = difference + projected.projection_bound;
                if controlled_error > tolerance {
                    certified = false;
                }
                if tolerance > 0.0 {
                    maximum_normalized_error =
                        maximum_normalized_error.max(controlled_error / tolerance);
                }
            }
            last_max_difference = maximum_difference;
            last_max_normalized_error = maximum_normalized_error;

            if certified {
                return moments_from_raw(
                    current,
                    k,
                    rank,
                    level,
                    total_evaluations,
                    maximum_difference,
                    projected.projection_bound,
                );
            }
        }
        previous = Some(current);
    }

    Err(EstimationError::InvalidInput(format!(
        "multinomial logistic-normal quadrature did not converge through Smolyak level {}: final max raw-moment level difference {last_max_difference:.6e}, max normalized error {last_max_normalized_error:.6e}, projection bound {:.6e}, evaluations {total_evaluations}/{}",
        control.maximum_sparse_level,
        projected.projection_bound,
        control.maximum_function_evaluations
    )))
}

struct SmolyakEvaluation {
    raw_moments: Vec<f64>,
}

fn evaluate_smolyak_level(
    active_mean: &[f64],
    projected: &ProjectedGaussian,
    rules: &[GaussHermiteRule],
    level: usize,
    k: usize,
    total_evaluations: &mut usize,
    maximum_function_evaluations: usize,
    absolute_tolerance: f64,
) -> Result<SmolyakEvaluation, EstimationError> {
    let rank = projected.factor.ncols();
    let q = rank.checked_add(level).ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial posterior Smolyak index overflowed usize".to_string(),
        )
    })?;
    let lower_total = q.saturating_sub(rank.saturating_sub(1)).max(rank);
    let moment_count = packed_moment_count(k)?;
    let upper_offsets = upper_triangle_offsets(k)?;
    let mut workspace = QuadratureWorkspace::new(
        active_mean,
        projected,
        rules,
        &upper_offsets,
        moment_count,
        total_evaluations,
        maximum_function_evaluations,
    )?;
    let mut indices = vec![1usize; rank];

    for total in lower_total..=q {
        let alternating_power = q - total;
        let mut coefficient = binomial_as_f64(rank - 1, alternating_power)?;
        if alternating_power % 2 == 1 {
            coefficient = -coefficient;
        }
        workspace.stream_compositions(0, total, &mut indices, coefficient)?;
    }

    let (mut raw_moments, mass, absolute_weight_sum) = workspace.accumulator.finish();
    if !(mass.is_finite() && mass > 0.0 && absolute_weight_sum.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior Smolyak level {level} produced invalid total weight {mass} (absolute sum {absolute_weight_sum})"
        )));
    }
    let mass_error = (mass - 1.0).abs();
    let summation_envelope =
        SUMMATION_ROUNDOFF_MULTIPLIER * f64::EPSILON * absolute_weight_sum.max(1.0);
    if mass_error > absolute_tolerance + summation_envelope {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior Smolyak level {level} failed constant-function exactness: total weight {mass:.17e}, error {mass_error:.6e}, allowed {:.6e}",
            absolute_tolerance + summation_envelope
        )));
    }
    for value in &mut raw_moments {
        *value /= mass;
    }
    Ok(SmolyakEvaluation { raw_moments })
}

fn packed_moment_count(k: usize) -> Result<usize, EstimationError> {
    let triangular = k
        .checked_add(1)
        .and_then(|next| k.checked_mul(next))
        .map(|product| product / 2)
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior moment dimension overflowed usize".to_string(),
            )
        })?;
    k.checked_add(triangular).ok_or_else(|| {
        EstimationError::InvalidInput(
            "multinomial posterior packed moment count overflowed usize".to_string(),
        )
    })
}

fn upper_triangle_offsets(k: usize) -> Result<Vec<usize>, EstimationError> {
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(k).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior could not allocate upper-triangle offsets: {error}"
        ))
    })?;
    let mut cursor = 0usize;
    for row in 0..k {
        offsets.push(cursor);
        cursor = cursor.checked_add(k - row).ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior upper-triangle offset overflowed usize".to_string(),
            )
        })?;
    }
    Ok(offsets)
}

fn zeroed_vec(length: usize, label: &str) -> Result<Vec<f64>, EstimationError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior could not allocate {label} (length {length}): {error}"
        ))
    })?;
    values.resize(length, 0.0);
    Ok(values)
}

struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn new() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let combined = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - combined) + value;
        } else {
            self.correction += (value - combined) + self.sum;
        }
        self.sum = combined;
    }

    fn value(&self) -> f64 {
        self.sum + self.correction
    }
}

struct QuadratureAccumulator {
    sums: Vec<f64>,
    corrections: Vec<f64>,
    mass: CompensatedSum,
    absolute_weight_sum: f64,
}

impl QuadratureAccumulator {
    fn new(moment_count: usize) -> Result<Self, EstimationError> {
        Ok(Self {
            sums: zeroed_vec(moment_count, "quadrature sums")?,
            corrections: zeroed_vec(moment_count, "quadrature corrections")?,
            mass: CompensatedSum::new(),
            absolute_weight_sum: 0.0,
        })
    }

    fn add_moment(&mut self, index: usize, value: f64) {
        let combined = self.sums[index] + value;
        if self.sums[index].abs() >= value.abs() {
            self.corrections[index] += (self.sums[index] - combined) + value;
        } else {
            self.corrections[index] += (value - combined) + self.sums[index];
        }
        self.sums[index] = combined;
    }

    fn add_weight(&mut self, weight: f64) {
        self.mass.add(weight);
        self.absolute_weight_sum += weight.abs();
    }

    fn finish(mut self) -> (Vec<f64>, f64, f64) {
        for (sum, correction) in self.sums.iter_mut().zip(self.corrections.iter()) {
            *sum += *correction;
        }
        (self.sums, self.mass.value(), self.absolute_weight_sum)
    }
}

struct QuadratureWorkspace<'a, 'b> {
    active_mean: &'a [f64],
    projected: &'a ProjectedGaussian,
    rules: &'a [GaussHermiteRule],
    upper_offsets: &'a [usize],
    z: Vec<f64>,
    active_eta: Vec<f64>,
    probabilities: Vec<f64>,
    accumulator: QuadratureAccumulator,
    total_evaluations: &'b mut usize,
    maximum_function_evaluations: usize,
}

impl<'a, 'b> QuadratureWorkspace<'a, 'b> {
        fn new(
        active_mean: &'a [f64],
        projected: &'a ProjectedGaussian,
        rules: &'a [GaussHermiteRule],
        upper_offsets: &'a [usize],
        moment_count: usize,
        total_evaluations: &'b mut usize,
        maximum_function_evaluations: usize,
    ) -> Result<Self, EstimationError> {
        let rank = projected.factor.ncols();
        let m = active_mean.len();
        Ok(Self {
            active_mean,
            projected,
            rules,
            upper_offsets,
            z: zeroed_vec(rank, "standard-normal quadrature coordinate")?,
            active_eta: zeroed_vec(m, "active-logit quadrature buffer")?,
            probabilities: zeroed_vec(m + 1, "softmax quadrature buffer")?,
            accumulator: QuadratureAccumulator::new(moment_count)?,
            total_evaluations,
            maximum_function_evaluations,
        })
    }

    fn stream_compositions(
        &mut self,
        position: usize,
        remaining: usize,
        indices: &mut [usize],
        coefficient: f64,
    ) -> Result<(), EstimationError> {
        let dimensions_left = indices.len() - position;
        if dimensions_left == 1 {
            if remaining == 0 {
                return Ok(());
            }
            indices[position] = remaining;
            return self.stream_tensor(0, indices, coefficient);
        }
        let maximum_here = remaining.saturating_sub(dimensions_left - 1);
        for index in 1..=maximum_here {
            indices[position] = index;
            self.stream_compositions(position + 1, remaining - index, indices, coefficient)?;
        }
        Ok(())
    }

    fn stream_tensor(
        &mut self,
        axis: usize,
        indices: &[usize],
        weight: f64,
    ) -> Result<(), EstimationError> {
        if axis == indices.len() {
            return self.accumulate_node(weight);
        }
        let rule_index = indices[axis] - 1;
        let node_count = self.rules[rule_index].nodes.len();
        for node_index in 0..node_count {
            let node = self.rules[rule_index].nodes[node_index];
            let node_weight = self.rules[rule_index].weights[node_index];
            self.z[axis] = node;
            self.stream_tensor(axis + 1, indices, weight * node_weight)?;
        }
        Ok(())
    }

    fn accumulate_node(&mut self, weight: f64) -> Result<(), EstimationError> {
        if *self.total_evaluations >= self.maximum_function_evaluations {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial logistic-normal quadrature exhausted its function-evaluation budget ({}) before convergence",
                self.maximum_function_evaluations
            )));
        }
        *self.total_evaluations += 1;

        for row in 0..self.active_mean.len() {
            let mut value = self.active_mean[row];
            for column in 0..self.z.len() {
                value += self.projected.factor[[row, column]] * self.z[column];
            }
            self.active_eta[row] = value;
        }
        softmax_with_reference_into(&self.active_eta, &mut self.probabilities)?;

        let k = self.probabilities.len();
        self.accumulator.add_weight(weight);
        for class in 0..k {
            self.accumulator
                .add_moment(class, weight * self.probabilities[class]);
        }
        let second_offset = k;
        for row in 0..k {
            for column in row..k {
                let packed = second_offset + self.upper_offsets[row] + column - row;
                self.accumulator.add_moment(
                    packed,
                    weight * self.probabilities[row] * self.probabilities[column],
                );
            }
        }
        Ok(())
    }
}

struct GaussHermiteRule {
    /// Nodes already transformed to standard-normal coordinates.
    nodes: Vec<f64>,
    /// Normalized standard-normal expectation weights (sum to one).
    weights: Vec<f64>,
}

fn gauss_hermite_rule(index: usize) -> Result<GaussHermiteRule, EstimationError> {
    let node_count = index
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "multinomial posterior Gauss-Hermite order overflowed usize".to_string(),
            )
        })?;
    let mut jacobi = Array2::<f64>::zeros((node_count, node_count));
    for diagonal in 0..node_count.saturating_sub(1) {
        // Physicists' Hermite weight exp(-x^2): Jacobi off-diagonal sqrt(i/2).
        let value = (((diagonal + 1) as f64) * 0.5).sqrt();
        jacobi[[diagonal, diagonal + 1]] = value;
        jacobi[[diagonal + 1, diagonal]] = value;
    }
    let (eigenvalues, eigenvectors) = jacobi.eigh(faer::Side::Lower).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior Gauss-Hermite rule {node_count} eigendecomposition failed: {error}"
        ))
    })?;
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    nodes.try_reserve_exact(node_count).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior could not allocate Gauss-Hermite nodes: {error}"
        ))
    })?;
    weights.try_reserve_exact(node_count).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial posterior could not allocate Gauss-Hermite weights: {error}"
        ))
    })?;
    for column in 0..node_count {
        nodes.push(std::f64::consts::SQRT_2 * eigenvalues[column]);
        weights.push(eigenvectors[[0, column]] * eigenvectors[[0, column]]);
    }
    let weight_sum: f64 = weights.iter().sum();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior Gauss-Hermite rule {node_count} has invalid weight sum {weight_sum}"
        )));
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }
    Ok(GaussHermiteRule { nodes, weights })
}

fn binomial_as_f64(n: usize, k: usize) -> Result<f64, EstimationError> {
    if k > n {
        return Ok(0.0);
    }
    let k = k.min(n - k);
    let mut value = 1.0_f64;
    for step in 1..=k {
        value *= (n - k + step) as f64 / step as f64;
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior Smolyak binomial coefficient C({n},{k}) overflowed f64"
            )));
        }
    }
    Ok(value)
}

fn softmax_with_reference(active_eta: &[f64]) -> Result<Vec<f64>, EstimationError> {
    let mut probabilities = zeroed_vec(active_eta.len() + 1, "softmax result")?;
    softmax_with_reference_into(active_eta, &mut probabilities)?;
    Ok(probabilities)
}

fn softmax_with_reference_into(
    active_eta: &[f64],
    probabilities: &mut [f64],
) -> Result<(), EstimationError> {
    if probabilities.len() != active_eta.len() + 1 {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior softmax buffer length {} does not equal active-logit length {} + 1",
            probabilities.len(),
            active_eta.len()
        )));
    }
    let maximum = active_eta.iter().copied().fold(0.0_f64, f64::max);
    let reference = probabilities.len() - 1;
    let mut denominator = (-maximum).exp();
    probabilities[reference] = denominator;
    for (class, &eta) in active_eta.iter().enumerate() {
        let numerator = (eta - maximum).exp();
        probabilities[class] = numerator;
        denominator += numerator;
    }
    if !(denominator.is_finite() && denominator > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior softmax produced invalid denominator {denominator}"
        )));
    }
    for probability in probabilities {
        *probability /= denominator;
    }
    Ok(())
}

fn moments_from_raw(
    raw_moments: Vec<f64>,
    k: usize,
    latent_rank: usize,
    sparse_level: usize,
    function_evaluations: usize,
    max_level_difference: f64,
    projection_bound: f64,
) -> Result<MultinomialPosteriorMoments, EstimationError> {
    let upper_offsets = upper_triangle_offsets(k)?;
    let raw_error = max_level_difference + projection_bound;
    let covariance_error = 3.0 * raw_error + raw_error * raw_error;

    let mut means = raw_moments[..k].to_vec();
    for (class, mean) in means.iter_mut().enumerate() {
        if *mean < -raw_error || *mean > 1.0 + raw_error || !mean.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior integrated mean for class {class} is outside its certified probability envelope: {mean} (raw error {raw_error:.6e})"
            )));
        }
        *mean = mean.clamp(0.0, 1.0);
    }
    let mean_sum: f64 = means.iter().sum();
    if !(mean_sum.is_finite() && mean_sum > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integrated class means have invalid sum {mean_sum}"
        )));
    }
    let simplex_error = (mean_sum - 1.0).abs();
    if simplex_error > (k as f64) * raw_error + covariance_roundoff_tolerance(1.0, k) {
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior integrated class means violate the simplex: sum {mean_sum:.17e}, error {simplex_error:.6e}, raw moment error {raw_error:.6e}"
        )));
    }
    for mean in &mut means {
        *mean /= mean_sum;
    }

    let second_offset = k;
    let mut covariance = Array2::<f64>::zeros((k, k));
    for row in 0..k {
        for column in row..k {
            let packed = second_offset + upper_offsets[row] + column - row;
            let value = raw_moments[packed] - means[row] * means[column];
            covariance[[row, column]] = value;
            covariance[[column, row]] = value;
        }
    }
    covariance = project_covariance_to_simplex_tangent(&covariance);
    covariance = remove_covariance_roundoff(covariance, covariance_error)?;
    covariance = project_covariance_to_simplex_tangent(&covariance);

    let mut standard_deviation = Array1::<f64>::zeros(k);
    for class in 0..k {
        let variance = covariance[[class, class]];
        if variance < -covariance_error || !variance.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "multinomial posterior variance for class {class} is invalid: {variance:.6e} (covariance error envelope {covariance_error:.6e})"
            )));
        }
        standard_deviation[class] = variance.max(0.0).sqrt();
    }

    Ok(MultinomialPosteriorMoments {
        class_mean: Array1::from_vec(means),
        class_covariance: covariance,
        class_standard_deviation: standard_deviation,
        latent_rank,
        sparse_level: Some(sparse_level),
        function_evaluations,
        max_raw_moment_level_difference: max_level_difference,
        covariance_range_projection_bound: projection_bound,
    })
}

fn project_covariance_to_simplex_tangent(covariance: &Array2<f64>) -> Array2<f64> {
    let k = covariance.nrows();
    let inverse_k = 1.0 / k as f64;
    let row_means: Vec<f64> = (0..k)
        .map(|row| covariance.row(row).sum() * inverse_k)
        .collect();
    let column_means: Vec<f64> = (0..k)
        .map(|column| covariance.column(column).sum() * inverse_k)
        .collect();
    let grand_mean = row_means.iter().sum::<f64>() * inverse_k;
    Array2::from_shape_fn((k, k), |(row, column)| {
        covariance[[row, column]] - row_means[row] - column_means[column] + grand_mean
    })
}

fn remove_covariance_roundoff(
    covariance: Array2<f64>,
    integration_error: f64,
) -> Result<Array2<f64>, EstimationError> {
    let symmetric = (&covariance + &covariance.t().to_owned()) * 0.5;
    let (eigenvalues, eigenvectors) = symmetric.eigh(faer::Side::Lower).map_err(|error| {
        EstimationError::InvalidInput(format!(
            "multinomial probability covariance eigendecomposition failed: {error}"
        ))
    })?;
    let scale = eigenvalues
        .iter()
        .fold(0.0_f64, |maximum, &value| maximum.max(value.abs()));
    let allowed_negative =
        integration_error + covariance_roundoff_tolerance(scale, covariance.nrows());
    let minimum = eigenvalues
        .iter()
        .fold(f64::INFINITY, |value, &candidate| value.min(candidate));
    if minimum < -allowed_negative {
        let negative_limit = -allowed_negative;
        return Err(EstimationError::InvalidInput(format!(
            "multinomial posterior probability covariance is indefinite beyond the integration error: min eigenvalue {minimum:.6e}, allowed {negative_limit:.6e}"
        )));
    }
    let mut scaled_eigenvectors = eigenvectors.clone();
    for (column, &eigenvalue) in eigenvalues.iter().enumerate() {
        let scale = eigenvalue.max(0.0);
        scaled_eigenvectors
            .column_mut(column)
            .mapv_inplace(|value| value * scale);
    }
    let reconstructed = scaled_eigenvectors.dot(&eigenvectors.t());
    Ok((&reconstructed + &reconstructed.t().to_owned()) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(absolute_tolerance: f64) -> MultinomialPosteriorIntegrationControl {
        MultinomialPosteriorIntegrationControl {
            absolute_tolerance,
            relative_tolerance: absolute_tolerance,
            minimum_sparse_level: 2,
            maximum_sparse_level: 8,
            maximum_function_evaluations: 2_000_000,
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn binary_reduction_matches_controlled_logistic_normal_identity() {
        let active_mean = Array1::from_vec(vec![1.1]);
        let active_covariance = Array2::from_shape_vec((1, 1), vec![0.64]).unwrap();
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(1.0e-10),
        )
        .expect("binary posterior moments");
        let (expected_mean, expected_slope) =
            gam_solve::quadrature::logit_posterior_meanwith_deriv(1.1, 0.8).unwrap();
        let expected_variance = expected_mean - expected_slope - expected_mean * expected_mean;

        assert_close(result.class_mean[0], expected_mean, 2.0e-14, "binary mean");
        assert_close(
            result.class_mean[1],
            1.0 - expected_mean,
            2.0e-14,
            "reference mean",
        );
        assert_close(
            result.class_covariance[[0, 0]],
            expected_variance,
            2.0e-14,
            "binary variance",
        );
        assert_close(
            result.class_covariance[[0, 1]],
            -expected_variance,
            2.0e-14,
            "binary covariance",
        );
        assert_eq!(result.latent_rank, 1);
        assert_eq!(result.sparse_level, None);
    }

    #[test]
    fn zero_covariance_is_exact_softmax_point_mass() {
        let active_mean = Array1::from_vec(vec![0.7, -0.4]);
        let active_covariance = Array2::<f64>::zeros((2, 2));
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(1.0e-10),
        )
        .expect("point-mass posterior moments");
        let expected = softmax_with_reference(active_mean.as_slice().unwrap()).unwrap();
        for class in 0..3 {
            assert_close(
                result.class_mean[class],
                expected[class],
                1.0e-15,
                "point mean",
            );
            assert_eq!(result.class_standard_deviation[class], 0.0);
            for other in 0..3 {
                assert_eq!(result.class_covariance[[class, other]], 0.0);
            }
        }
        assert_eq!(result.latent_rank, 0);
        assert_eq!(result.sparse_level, None);
    }

    #[test]
    fn exchangeable_full_logits_require_cross_covariance_and_integrate_to_uniform() {
        // If full logits gamma_c are iid N(0,s^2), reference coding gives
        // eta_a=gamma_a-gamma_ref, hence diag(V)=2s^2 and offdiag(V)=s^2.
        // Exchangeability makes E[p_c]=1/3 exactly.  Dropping the off-diagonal
        // covariance destroys that identity for the reference class.
        let variance = 0.7;
        let active_mean = Array1::zeros(2);
        let active_covariance = Array2::from_shape_vec(
            (2, 2),
            vec![2.0 * variance, variance, variance, 2.0 * variance],
        )
        .unwrap();
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(2.0e-7),
        )
        .expect("exchangeable posterior moments");

        for class in 0..3 {
            assert_close(result.class_mean[class], 1.0 / 3.0, 8.0e-7, "uniform mean");
        }
        for class in 1..3 {
            assert_close(
                result.class_covariance[[class, class]],
                result.class_covariance[[0, 0]],
                2.0e-6,
                "exchangeable variance",
            );
        }
        for row in 0..3 {
            assert_close(
                result.class_covariance.row(row).sum(),
                0.0,
                2.0e-12,
                "simplex covariance row sum",
            );
        }
        assert_eq!(result.latent_rank, 2);
        assert!(result.sparse_level.is_some());
    }

    #[test]
    fn rank_one_general_case_matches_independent_one_dimensional_gh_oracle() {
        let active_mean = Array1::from_vec(vec![0.45, -0.7]);
        let loading = [0.8_f64, -0.35_f64];
        let active_covariance =
            Array2::from_shape_fn((2, 2), |(row, column)| loading[row] * loading[column]);
        let result = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(5.0e-8),
        )
        .expect("rank-one posterior moments");
        assert_eq!(result.latent_rank, 1);

        // Independent high-order one-dimensional GH evaluation of the exact
        // rank-one representation eta=mu+loading*Z.
        let oracle_rule = gauss_hermite_rule(21).unwrap(); // 41 nodes
        let mut oracle_mean = [0.0_f64; 3];
        let mut oracle_second = [[0.0_f64; 3]; 3];
        for (&z, &weight) in oracle_rule.nodes.iter().zip(oracle_rule.weights.iter()) {
            let eta = [
                active_mean[0] + loading[0] * z,
                active_mean[1] + loading[1] * z,
            ];
            let probability = softmax_with_reference(&eta).unwrap();
            for row in 0..3 {
                oracle_mean[row] += weight * probability[row];
                for column in 0..3 {
                    oracle_second[row][column] += weight * probability[row] * probability[column];
                }
            }
        }
        for row in 0..3 {
            assert_close(
                result.class_mean[row],
                oracle_mean[row],
                3.0e-7,
                "rank-one mean",
            );
            for column in 0..3 {
                let oracle_covariance =
                    oracle_second[row][column] - oracle_mean[row] * oracle_mean[column];
                assert_close(
                    result.class_covariance[[row, column]],
                    oracle_covariance,
                    8.0e-7,
                    "rank-one covariance",
                );
            }
        }
        let allowed = 5.0e-8
            + 5.0e-8
                * result
                    .class_mean
                    .iter()
                    .fold(0.0_f64, |scale, &value| scale.max(value.abs()));
        assert!(
            result.max_raw_moment_level_difference + result.covariance_range_projection_bound
                <= allowed * 1.01,
            "returned result must carry the level-difference certificate"
        );
    }

    #[test]
    fn insufficient_sparse_level_is_a_typed_error_not_a_plugin_result() {
        let active_mean = Array1::from_vec(vec![1.2, -0.8]);
        let active_covariance = Array2::from_shape_vec((2, 2), vec![2.0, 0.9, 0.9, 1.5]).unwrap();
        let strict_control = MultinomialPosteriorIntegrationControl {
            absolute_tolerance: 1.0e-14,
            relative_tolerance: 1.0e-14,
            minimum_sparse_level: 1,
            maximum_sparse_level: 1,
            maximum_function_evaluations: 100_000,
        };
        let error = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &strict_control,
        )
        .expect_err("one sparse refinement cannot certify this nonlinear integral");
        assert!(
            error.to_string().contains("did not converge"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn materially_indefinite_active_covariance_is_rejected() {
        let active_mean = Array1::from_vec(vec![0.0, 0.0]);
        let active_covariance = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let error = integrate_logistic_normal_softmax_moments(
            active_mean.view(),
            active_covariance.view(),
            &control(1.0e-7),
        )
        .expect_err("indefinite covariance must fail");
        assert!(error.to_string().contains("not positive semidefinite"));
    }
}
