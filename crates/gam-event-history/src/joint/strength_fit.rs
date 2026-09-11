//! Empirical-Bayes strength optimization of the coefficient integral. Generic
//! optimization stays in opt. Acceptance additionally uses independent-bank
//! comparisons in the local evidence geometry; a solver status alone is not
//! a resolution assessment or a finished joint-model fit.
use super::*;
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct StrengthOptimizationOptions {
    pub stationarity_tolerance: f64,
    pub log_evidence_tolerance: f64,
    /// Error in whitened strength score/curvature and coefficient means in
    /// posterior-SD units. These are numerical approximation targets.
    pub relative_resolution_tolerance: f64,
    pub standard_error_multiplier: f64,
    pub minimum_effective_samples: f64,
    pub maximum_iterations: usize,
}

impl Default for StrengthOptimizationOptions {
    fn default() -> Self {
        Self {
            stationarity_tolerance: 1e-6,
            log_evidence_tolerance: 0.01,
            relative_resolution_tolerance: 0.05,
            standard_error_multiplier: 3.0,
            minimum_effective_samples: 32.0,
            maximum_iterations: 100,
        }
    }
}

impl StrengthOptimizationOptions {
    pub(super) fn validate(&self) -> Result<(), EventHistoryError> {
        if [
            self.stationarity_tolerance,
            self.log_evidence_tolerance,
            self.relative_resolution_tolerance,
            self.standard_error_multiplier,
            self.minimum_effective_samples,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
            || self.relative_resolution_tolerance >= 1.0
            || self.maximum_iterations == 0
        {
            return Err(invalid(
                "strength optimization requires positive finite tolerances, relative resolution below one, and a positive iteration limit",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StrengthResolutionReport {
    pub gradient_infinity_norm: f64,
    pub log_evidence_error_estimate: f64,
    pub whitened_score_error_estimate: f64,
    pub whitened_curvature_error_estimate: f64,
    pub maximum_standardized_mean_error: f64,
    pub validation_log_evidence: f64,
    pub validation_effective_samples: f64,
    pub iterations: usize,
}

impl StrengthResolutionReport {
    pub(super) fn resolved(
        &self,
        fitted: &JointCoefficientEvidence,
        options: &StrengthOptimizationOptions,
    ) -> bool {
        [
            self.gradient_infinity_norm,
            self.log_evidence_error_estimate,
            self.whitened_score_error_estimate,
            self.whitened_curvature_error_estimate,
            self.maximum_standardized_mean_error,
        ]
        .iter()
        .all(|v| v.is_finite())
            && self.gradient_infinity_norm <= options.stationarity_tolerance
            && self.log_evidence_error_estimate <= options.log_evidence_tolerance
            && self.whitened_score_error_estimate <= options.relative_resolution_tolerance
            && self.whitened_curvature_error_estimate <= options.relative_resolution_tolerance
            && self.maximum_standardized_mean_error <= options.relative_resolution_tolerance
            && fitted.effective_samples >= options.minimum_effective_samples
            && self.validation_effective_samples >= options.minimum_effective_samples
    }
}

/// A converged, numerically assessed INTERIOR strength optimum of a sampled
/// coefficient integral. It retains its coefficient draws and posterior means.
/// It is not a finished model: proposal coverage, null-boundary comparisons,
/// selected entry laws and integrated serving still need their own machinery.
pub struct JointStrengthOptimum<'i, 'p, 'm> {
    integral: &'i JointCoefficientIntegral<'p, 'm>,
    log_strengths: Vec<f64>,
    evidence: JointCoefficientEvidence,
    report: StrengthResolutionReport,
}

impl JointStrengthOptimum<'_, '_, '_> {
    pub fn log_strengths(&self) -> &[f64] {
        &self.log_strengths
    }
    pub fn evidence(&self) -> &JointCoefficientEvidence {
        &self.evidence
    }
    pub fn report(&self) -> &StrengthResolutionReport {
        &self.report
    }
    pub fn draws(&self) -> &[CoefficientImportanceDraw] {
        self.integral.draws()
    }
}

fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, |a, b| a.hypot(b))
}

fn multiply(matrix: &Array2<f64>, values: &[f64]) -> Vec<f64> {
    matrix
        .rows()
        .into_iter()
        .map(|row| sum(row.iter().zip(values).map(|(a, b)| a * b)))
        .collect()
}

fn inverse_lower(lower: &Array2<f64>) -> Array2<f64> {
    let h = lower.nrows();
    let mut inverse = Array2::zeros((h, h));
    for j in 0..h {
        for i in j..h {
            inverse[[i, j]] = (f64::from(i == j)
                - sum((j..i).map(|k| lower[[i, k]] * inverse[[k, j]])))
                / lower[[i, i]];
        }
    }
    inverse
}

struct WhitenedError {
    mean: Vec<f64>,
    standard_error: Vec<f64>,
    inner_score: f64,
    inner_curvature: f64,
}

fn whitened_error(value: &JointCoefficientEvidence, inverse: &Array2<f64>) -> WhitenedError {
    let h = value.gradient.len();
    let mut standard_error = vec![0.0_f64; h];
    let mut norm_mean = 0.0;
    let mut squared_norm_mean = 0.0;
    let mut conditional_curvature = vec![0.0; h];
    for (i, &w) in value.weights.iter().enumerate() {
        let centered: Vec<_> = (0..h)
            .map(|j| value.conditional_gradient[[i, j]] - value.gradient[j])
            .collect();
        let white = multiply(inverse, &centered);
        let length = norm(white.iter().copied());
        norm_mean += w * length;
        squared_norm_mean += w * length * length;
        for j in 0..h {
            standard_error[j] = standard_error[j].hypot(w * white[j]);
            conditional_curvature[j] += w * value.conditional_second[[i, j]].abs();
        }
    }
    let n = value.weights.len() as f64;
    for se in &mut standard_error {
        *se *= (n / (n - 1.0)).sqrt();
    }
    let conditional_norm_bound = sum(
        (0..h).map(|j| conditional_curvature[j] * sum(inverse.column(j).iter().map(|v| v * v)))
    );
    // If |delta log likelihood|<=d, normalized weight ratios lie in
    // [exp(-2d),exp(2d)]. The inner d is only an estimated error, so these
    // propagated quantities remain estimates, not deterministic certificates.
    let epsilon = (2.0 * value.inner_log_error_estimate).exp_m1();
    WhitenedError {
        mean: multiply(inverse, &value.gradient),
        standard_error,
        inner_score: epsilon * norm_mean,
        inner_curvature: epsilon * (conditional_norm_bound + squared_norm_mean)
            + (epsilon * norm_mean).powi(2),
    }
}

pub(super) fn assessment(
    fitted: &JointCoefficientEvidence,
    validation: &JointCoefficientEvidence,
    options: &StrengthOptimizationOptions,
    iterations: usize,
) -> Result<StrengthResolutionReport, EventHistoryError> {
    let h = fitted.gradient.len();
    let mut precision = Array2::zeros((h, h));
    for j in 0..h {
        let mut direction = vec![0.0; h];
        direction[j] = 1.0;
        let product = fitted.hessian_product(&direction)?;
        for i in 0..h {
            precision[[i, j]] = -product.value[i];
        }
    }
    // No curvature clipping or diagonal jitter: an indefinite/singular
    // evidence point is not an identified interior maximum.
    let factor = precision::Cholesky::new(&precision)
        .map_err(|_| numerical("strength evidence has no identified interior maximum"))?;
    let inverse = inverse_lower(&factor.lower);
    let a = whitened_error(fitted, &inverse);
    let b = whitened_error(validation, &inverse);
    let score_change = norm(a.mean.iter().zip(&b.mean).map(|(x, y)| x - y));
    let score_se = norm(a.standard_error.iter().chain(&b.standard_error).copied());
    let mut curvature_rows = vec![0.0; h];
    for j in 0..h {
        // L^-T e_j is row j of L^-1. Transform both HVP values and SEs;
        // sums of absolute coefficients are conservative for correlated SEs.
        let direction = inverse.row(j).to_vec();
        let left = fitted.hessian_product(&direction)?;
        let right = validation.hessian_product(&direction)?;
        let difference: Vec<_> = left
            .value
            .iter()
            .zip(&right.value)
            .map(|(x, y)| x - y)
            .collect();
        let white = multiply(&inverse, &difference);
        for i in 0..h {
            let se = sum((0..h).map(|k| {
                inverse[[i, k]].abs() * (left.standard_error[k] + right.standard_error[k])
            }));
            curvature_rows[i] += white[i].abs() + options.standard_error_multiplier * se;
        }
    }
    let mut maximum_mean_error = 0.0_f64;
    let left_epsilon = (2.0 * fitted.inner_log_error_estimate).exp_m1();
    let right_epsilon = (2.0 * validation.inner_log_error_estimate).exp_m1();
    for j in 0..fitted.coefficient_mean.len() {
        let scale = fitted.coefficient_variance[j].sqrt();
        if scale <= 0.0 || !scale.is_finite() {
            return Err(numerical(
                "coefficient proposal does not resolve a posterior spread",
            ));
        }
        let error = (fitted.coefficient_mean[j] - validation.coefficient_mean[j]).abs()
            + options.standard_error_multiplier
                * fitted.mean_standard_error[j].hypot(validation.mean_standard_error[j])
            + left_epsilon * scale
            + right_epsilon * validation.coefficient_variance[j].sqrt();
        maximum_mean_error = maximum_mean_error.max(error / scale);
    }
    let report = StrengthResolutionReport {
        gradient_infinity_norm: fitted.gradient.iter().map(|v| v.abs()).fold(0.0, f64::max),
        log_evidence_error_estimate: (fitted.log_evidence - validation.log_evidence).abs()
            + options.standard_error_multiplier
                * fitted
                    .log_standard_error
                    .hypot(validation.log_standard_error)
            + fitted.inner_log_error_estimate
            + validation.inner_log_error_estimate,
        whitened_score_error_estimate: score_change
            + options.standard_error_multiplier * score_se
            + a.inner_score
            + b.inner_score,
        whitened_curvature_error_estimate: curvature_rows.into_iter().fold(0.0, f64::max)
            + a.inner_curvature
            + b.inner_curvature,
        maximum_standardized_mean_error: maximum_mean_error,
        validation_log_evidence: validation.log_evidence,
        validation_effective_samples: validation.effective_samples,
        iterations,
    };
    Ok(report)
}

fn assess(
    fitted: &JointCoefficientEvidence,
    validation: &JointCoefficientEvidence,
    options: &StrengthOptimizationOptions,
    iterations: usize,
) -> Result<StrengthResolutionReport, EventHistoryError> {
    let report = assessment(fitted, validation, options, iterations)?;
    if !report.resolved(fitted, options) {
        return Err(numerical(format!(
            "strength integral unresolved: {report:?}"
        )));
    }
    Ok(report)
}

impl<'p, 'm> JointCoefficientIntegral<'p, 'm> {
    /// Maximize the integrated evidence through opt, then assess a separate
    /// independent coefficient bank from the same cohort and function measure.
    /// Validation draws must not have been used to choose the fitting bank or
    /// its optimum. Failure calls for proposal/refinement work; it never
    /// returns a best-effort model or disguises a null boundary as convergence.
    pub fn optimize_strengths<'i>(
        &'i self,
        initial: &[f64],
        validation: &JointCoefficientIntegral<'_, '_>,
        options: &StrengthOptimizationOptions,
    ) -> Result<JointStrengthOptimum<'i, 'p, 'm>, EventHistoryError> {
        let h = self.priors.penalties().len();
        if initial.len() != h
            || initial.iter().any(|v| !v.is_finite())
            || !std::ptr::eq(self.priors, validation.priors)
            || !std::sync::Arc::ptr_eq(&self.cohort_identity, &validation.cohort_identity)
            || std::ptr::eq(self, validation)
            || self.draws.len() == validation.draws.len()
                && self.draws.iter().zip(&validation.draws).all(|(a, b)| {
                    a.coefficients == b.coefficients
                        && a.log_proposal_density == b.log_proposal_density
                })
        {
            return Err(invalid(
                "strength optimization requires finite seeds and separate independent banks from one cohort/function measure",
            ));
        }
        options.validate()?;
        // Include BOTH retained input banks, evaluations, BFGS's dense metric
        // and the curvature assessment before starting any expensive work.
        let p = self.draws[0].coefficients.len();
        let words = self
            .draws
            .len()
            .checked_add(validation.draws.len())
            .and_then(|n| n.checked_mul(p.checked_add(h.checked_mul(6)?)?.checked_add(16)?))
            .and_then(|n| n.checked_add(h.checked_mul(h)?.checked_mul(16)?))
            .and_then(|n| n.checked_add(p.checked_mul(16)?));
        if words
            .and_then(|n| n.checked_mul(8))
            .is_none_or(|bytes| bytes > self.memory_limit_bytes.min(validation.memory_limit_bytes))
        {
            return Err(invalid(
                "strength optimization exceeds its combined bank/optimizer memory budget",
            ));
        }
        let (log_strengths, evidence, iterations) =
            self.strength_stationary_point(initial, options)?;
        let checked = validation.evaluate(&log_strengths)?;
        let report = assess(&evidence, &checked, options, iterations)?;
        Ok(JointStrengthOptimum {
            integral: self,
            log_strengths,
            evidence,
            report,
        })
    }

    pub(super) fn strength_stationary_point(
        &self,
        initial: &[f64],
        options: &StrengthOptimizationOptions,
    ) -> Result<(Vec<f64>, JointCoefficientEvidence, usize), EventHistoryError> {
        let objective = opt::FusedObjective::new(|rho: &Array1<f64>| {
            let values: Vec<_> = rho.iter().copied().collect();
            let value = self
                .evaluate(&values)
                .map_err(opt::ObjectiveEvalError::recoverable_from)?;
            Ok(opt::FirstOrderSample {
                value: -value.log_evidence,
                gradient: Array1::from_iter(value.gradient.iter().map(|v| -v)),
            })
        });
        let solution = opt::Bfgs::new(Array1::from_vec(initial.to_vec()), objective)
            .without_relative_stall()
            .with_tolerance(
                opt::Tolerance::new(options.stationarity_tolerance)
                    .map_err(|e| invalid(e.to_string()))?,
            )
            .with_gradient_tolerance(opt::GradientTolerance::absolute(
                options.stationarity_tolerance,
            ))
            .with_max_iterations(
                opt::MaxIterations::new(options.maximum_iterations)
                    .map_err(|e| invalid(e.to_string()))?,
            )
            .run()
            .map_err(|e| numerical(format!("strength evidence optimization failed: {e}")))?;
        let log_strengths = solution.final_point.to_vec();
        let evidence = self.evaluate(&log_strengths)?;
        if evidence
            .gradient
            .iter()
            .any(|g| g.abs() > options.stationarity_tolerance)
        {
            return Err(numerical(
                "strength solver stopped without meeting the final evidence score tolerance",
            ));
        }
        Ok((log_strengths, evidence, solution.iterations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};
    use rand_distr::{Distribution, Gamma};

    fn bank<'p, 'm>(
        priors: &'p JointFunctionPriors<'m>,
        identity: &std::sync::Arc<()>,
        seed: u64,
    ) -> JointCoefficientIntegral<'p, 'm> {
        // Seven Poisson-process events, exposure four, and an exponential
        // prior on 2*hazard. Its exact EB optimum is lambda=4/(2*7).
        let shape = 8.0_f64;
        let rate = 4.0 + 4.0 / 7.0;
        let proposal = Gamma::new(shape, 1.0 / rate).unwrap();
        let log_gamma = (1..=7).map(|v| (v as f64).ln()).sum::<f64>();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut draws = Vec::new();
        let mut log_likelihood = Vec::new();
        for _ in 0..32768 {
            let hazard: f64 = proposal.sample(&mut rng);
            let beta = hazard.ln();
            draws.push(CoefficientImportanceDraw {
                coefficients: vec![beta],
                log_proposal_density: shape * rate.ln() - log_gamma + shape * beta - rate * hazard,
            });
            log_likelihood.push(7.0 * beta - 4.0 * hazard);
        }
        JointCoefficientIntegral {
            cohort_identity: std::sync::Arc::clone(identity),
            priors,
            draws,
            log_likelihood,
            inner_log_error_estimate: 0.0,
            memory_limit_bytes: 32 << 20,
        }
    }

    #[test]
    fn opt_strength_solution_matches_conjugate_evidence_and_rejects_unresolved_results() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let history = JointHistory {
            times: vec![0.0, 1.0, 2.0],
            exposure: vec![0.0, 1.0, 1.0],
            events: vec![None; 3],
            initially_at_risk: vec![true],
            baseline_design: Array2::ones((3, 1)),
            drive_design: Array2::ones((2, 1)),
            entry_design: vec![],
            genetics: vec![],
            measurements: vec![],
        };
        let priors = model.function_priors(&[&history], 32 << 20).unwrap();
        let identity = std::sync::Arc::new(());
        let fitting = bank(&priors, &identity, 1949);
        let mut validation = bank(&priors, &identity, 1951);
        let options = StrengthOptimizationOptions::default();
        let result = fitting
            .optimize_strengths(&[0.0], &validation, &options)
            .unwrap();
        let exact_rho = (4.0_f64 / 14.0).ln();
        eprintln!(
            "strength optimum {:?}, analytic {exact_rho}; {:?}",
            result.log_strengths(),
            result.report()
        );
        assert!((result.log_strengths()[0] - exact_rho).abs() < 0.04);
        assert!(result.report().gradient_infinity_norm < options.stationarity_tolerance);
        assert!(result.report().iterations > 0);
        assert_eq!(result.draws().len(), 32768);
        assert!(result.evidence().coefficient_mean()[0].is_finite());
        assert!(result.report().whitened_curvature_error_estimate > 0.0);
        assert!(
            fitting
                .optimize_strengths(&[0.0], &fitting, &options)
                .is_err()
        );
        let duplicate = bank(&priors, &identity, 1949);
        assert!(
            fitting
                .optimize_strengths(&[0.0], &duplicate, &options)
                .is_err()
        );
        let unrelated = bank(&priors, &std::sync::Arc::new(()), 1951);
        assert!(
            fitting
                .optimize_strengths(&[0.0], &unrelated, &options)
                .is_err()
        );
        let strict = StrengthOptimizationOptions {
            relative_resolution_tolerance: 1e-8,
            ..options.clone()
        };
        let checked = validation.evaluate(result.log_strengths()).unwrap();
        assert!(assess(result.evidence(), &checked, &strict, 0).is_err());
        validation.inner_log_error_estimate = 0.1;
        let uncertain = validation.evaluate(result.log_strengths()).unwrap();
        assert!(assess(result.evidence(), &uncertain, &options, 0).is_err());
        let nonstationary = fitting.evaluate(&[0.0]).unwrap();
        assert!(assess(&nonstationary, &checked, &options, 0).is_err());
        let short = StrengthOptimizationOptions {
            maximum_iterations: 1,
            ..options.clone()
        };
        assert!(
            fitting
                .optimize_strengths(&[0.0], &validation, &short)
                .is_err()
        );
        validation.memory_limit_bytes = 1;
        assert!(
            fitting
                .optimize_strengths(&[0.0], &validation, &options)
                .is_err()
        );
        assert!(
            fitting
                .optimize_strengths(&[f64::NAN], &validation, &options)
                .is_err()
        );
    }

    #[test]
    fn inner_log_error_bounds_correlated_score_and_curvature_changes_after_whitening() {
        fn value(shift: &[f64; 4]) -> JointCoefficientEvidence {
            let log_total = log_sum_exp(shift);
            let weights: Vec<_> = shift.iter().map(|v| (v - log_total).exp()).collect();
            let conditional_gradient =
                ndarray::array![[1e-4, 2e4], [-1e-4, -2e4], [1e-4, -1e4], [-1e-4, 1e4]];
            let conditional_second =
                ndarray::array![[-3e-8, -4e8], [-3e-8, -4e8], [-3e-8, -4e8], [-3e-8, -4e8]];
            let gradient = (0..2)
                .map(|j| sum((0..4).map(|i| weights[i] * conditional_gradient[[i, j]])))
                .collect();
            JointCoefficientEvidence {
                log_evidence: log_total - 4.0_f64.ln(),
                log_standard_error: 0.0,
                effective_samples: 1.0 / sum(weights.iter().map(|w| w * w)),
                inner_log_error_estimate: 0.01,
                weights,
                coefficient_mean: vec![0.0; 2],
                coefficient_variance: vec![1.0; 2],
                mean_standard_error: vec![0.0; 2],
                gradient,
                gradient_standard_error: vec![0.0; 2],
                conditional_gradient,
                conditional_second,
            }
        }
        let original = value(&[0.0; 4]);
        // Independent exact precision of this four-point mixture. Its
        // off-diagonal covariance and 16-order diagonal scale ratio exercise
        // the full coordinate transformation, not just a scalar error formula.
        let precision = ndarray::array![[2e-8, -0.5], [-0.5, 1.5e8]];
        let inverse = inverse_lower(&precision::Cholesky::new(&precision).unwrap().lower);
        let bound = whitened_error(&original, &inverse);
        for shift in [
            [0.01, -0.01, 0.01, -0.01],
            [-0.01, 0.01, 0.01, -0.01],
            [0.01, -0.003, -0.01, 0.007],
        ] {
            let perturbed = value(&shift);
            let gradient_change: Vec<_> = perturbed
                .gradient
                .iter()
                .zip(&original.gradient)
                .map(|(a, b)| a - b)
                .collect();
            assert!(norm(multiply(&inverse, &gradient_change).into_iter()) <= bound.inner_score);
            let mut change = Array2::zeros((2, 2));
            for j in 0..2 {
                let direction = inverse.row(j).to_vec();
                let left = original.hessian_product(&direction).unwrap();
                let right = perturbed.hessian_product(&direction).unwrap();
                let delta: Vec<_> = right
                    .value
                    .iter()
                    .zip(&left.value)
                    .map(|(a, b)| a - b)
                    .collect();
                for (i, v) in multiply(&inverse, &delta).into_iter().enumerate() {
                    change[[i, j]] = v;
                }
            }
            assert!((change[[0, 1]] - change[[1, 0]]).abs() < 1e-12);
            let operator_norm = 0.5
                * ((change[[0, 0]] + change[[1, 1]]).abs()
                    + (change[[0, 0]] - change[[1, 1]]).hypot(2.0 * change[[0, 1]]));
            assert!(operator_norm <= bound.inner_curvature);
        }
    }
}
