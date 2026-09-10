//! Normalized priors on final-function properties. Gaussian blocks use a
//! function-measure root and its exact determinant; nonlinear blocks include
//! their chart Jacobians. Global coefficient integration remains necessary.
use super::*;
use std::sync::Arc;

#[path = "structural_prior.rs"]
mod structural;
use structural::{ScalarPriorEvaluation, StructuralFunction};
#[path = "category_prior.rs"]
mod category;
use category::{CategoryEvaluation, CategoryPriors};
#[path = "coefficient_proposal.rs"]
mod proposal;
pub use proposal::PriorCoefficientProposal;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionPenalty {
    Decoder { mark: usize },
    BaselineVariation { mark: usize },
    GeneticDrive,
    EntryMean,
    EntryPrevalence { mark: usize },
    DiseaseJump { mark: usize },
    MeasurementEffect { channel: usize },
    TemporalVariation,
    MeasurementPrecision { channel: usize },
    TailVarianceInflation { channel: usize },
    CountOverdispersion { channel: usize },
    CountMean { channel: usize },
    BaselineLevel,
}

struct FunctionRoot {
    root: Array2<f64>,
    log_determinant: f64,
}

struct GaussianFunction {
    root: Arc<FunctionRoot>,
    coordinates: Vec<Range<usize>>,
    /// Student-t effects are measured in residual-scale units. The prior
    /// precision is exp(rho - 2*log_scale), including its normalizer.
    log_scale: Option<usize>,
}

/// Frozen function measures for one model and cohort design. The coordinate
/// order of strengths is exposed by `penalties`; inactive zero-dimensional
/// functions introduce no strength coordinates.
pub struct JointFunctionPriors<'m> {
    model: &'m JointLikelihood,
    penalties: Vec<FunctionPenalty>,
    gaussian: Vec<GaussianFunction>,
    structural: Vec<Vec<StructuralFunction>>,
    category: CategoryPriors,
    baseline_mean_design: Vec<f64>,
    log_followup_scale: f64,
    decoder_strengths: usize,
    memory_limit_bytes: usize,
}

struct GaussianEvaluation {
    half_log_precision: f64,
    weighted_penalty: f64,
    /// Derivatives in each repeated function's contiguous coordinate range.
    gradients: Vec<Vec<f64>>,
}

pub struct FunctionPriorEvaluation<'a, 'm> {
    prior: &'a JointFunctionPriors<'m>,
    decoder: DecoderPriorEvaluation,
    gaussian: Vec<GaussianEvaluation>,
    structural: Vec<Vec<ScalarPriorEvaluation>>,
    category: CategoryEvaluation,
    baseline_weights: Vec<f64>,
    log_density: f64,
    gradient: Vec<f64>,
    strength_gradient: Vec<f64>,
    strength_second: Vec<f64>,
}

fn scaled(value: f64, log_scale: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        (value.abs().ln() + log_scale).exp() * value.signum()
    }
}

impl FunctionRoot {
    fn from_gram(gram: Array2<f64>) -> Result<Self, EventHistoryError> {
        let factor = precision::Cholesky::new(&gram).map_err(|_| invalid(
            "function measure does not identify an independent basis; remove aliased basis directions before fitting"))?;
        Ok(Self {
            root: factor.lower.t().to_owned(),
            log_determinant: factor.log_determinant,
        })
    }
    fn identity(width: usize) -> Self {
        Self {
            root: Array2::eye(width),
            log_determinant: 0.0,
        }
    }
    fn quadratic(&self, coefficients: &[f64], half_log_precision: f64) -> (f64, Vec<f64>) {
        // Split the precision between the coefficient vector and the final
        // transpose product. exp(rho) need not itself be representable.
        let x: Vec<_> = coefficients
            .iter()
            .map(|&v| scaled(v, half_log_precision))
            .collect();
        let width = x.len();
        let transformed: Vec<f64> = (0..width)
            .map(|i| (0..width).map(|j| self.root[[i, j]] * x[j]).sum())
            .collect();
        let norm = transformed.iter().fold(0.0_f64, |a, &b| a.hypot(b));
        let gradient = (0..width)
            .map(|j| {
                scaled(
                    (0..width).map(|i| self.root[[i, j]] * transformed[i]).sum(),
                    half_log_precision,
                )
            })
            .collect();
        (0.5 * norm * norm, gradient)
    }
}

fn add_outer(gram: &mut Array2<f64>, row: &[f64], weight: f64) {
    for i in 0..row.len() {
        for j in 0..=i {
            gram[[i, j]] += weight * row[i] * row[j];
            gram[[j, i]] = gram[[i, j]];
        }
    }
}

fn product_root(left: &FunctionRoot, right: &FunctionRoot) -> FunctionRoot {
    let l = left.root.ncols();
    let r = right.root.ncols();
    let mut root = Array2::zeros((l * r, l * r));
    for i in 0..l {
        for j in 0..l {
            for a in 0..r {
                for b in 0..r {
                    root[[i * r + a, j * r + b]] = left.root[[i, j]] * right.root[[a, b]];
                }
            }
        }
    }
    FunctionRoot {
        root,
        log_determinant: r as f64 * left.log_determinant + l as f64 * right.log_determinant,
    }
}

impl JointLikelihood {
    /// Equal subject measure; within a follow-up, time measure is normalized
    /// Lebesgue exposure. Genetic effects use the declared Gaussian score law.
    /// Baseline bases must contain their constant as column zero. The
    /// variation penalty removes it; a separate positive-level prior uses
    /// the geometric baseline rate under the same function measure.
    ///
    /// Entry prevalence is a function contrast against no prevalent diagnosis,
    /// not an invented distribution of pre-entry event-free follow-up.
    pub fn function_priors<'m>(
        &'m self,
        histories: &[&JointHistory],
        memory_limit_bytes: usize,
    ) -> Result<JointFunctionPriors<'m>, EventHistoryError> {
        if histories.is_empty() {
            return Err(invalid("function priors require a cohort design"));
        }
        for h in histories {
            self.validate_history(h)?;
        }
        let k = self.spec.signatures;
        let genes = self.spec.genetic_mean.len() + 1;
        let b = self.spec.baseline_columns;
        let c = self.spec.drive_columns;
        let e = self.spec.entry_columns + 1;
        let once = self
            .spec
            .marks
            .iter()
            .filter(|&&v| v == MarkKind::Once)
            .count();
        // Bound the dense local roots and their construction work before any
        // allocation. Roots are shared across signatures, marks, and channels.
        let drive_width = c
            .checked_mul(genes)
            .ok_or_else(|| invalid("function-prior drive dimension overflow"))?;
        let context_width = e
            .checked_mul(genes)
            .ok_or_else(|| invalid("function-prior entry dimension overflow"))?;
        let entries = [b, c, e, genes, drive_width, context_width, k]
            .iter()
            .try_fold(0usize, |sum, &n| {
                n.checked_mul(n).and_then(|v| sum.checked_add(v))
            })
            .and_then(|v| v.checked_mul(6))
            .and_then(|v| v.checked_add(self.layout.width.checked_mul(16)?))
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| invalid("function-prior workspace dimension overflow"))?;
        if entries > memory_limit_bytes {
            return Err(numerical(format!(
                "function priors require {entries} workspace bytes, above limit {memory_limit_bytes}"
            )));
        }
        let decoder_strengths = if k == 0 { 0 } else { self.spec.marks.len() };
        let mut result = JointFunctionPriors {
            model: self,
            penalties: (0..decoder_strengths)
                .map(|mark| FunctionPenalty::Decoder { mark })
                .collect(),
            gaussian: Vec::new(),
            structural: Vec::new(),
            category: CategoryPriors::new(self),
            baseline_mean_design: vec![0.0; b],
            log_followup_scale: 0.0,
            decoder_strengths,
            memory_limit_bytes,
        };
        let push = |out: &mut JointFunctionPriors<'m>,
                    label,
                    root: Arc<FunctionRoot>,
                    coordinates,
                    log_scale| {
            out.penalties.push(label);
            out.gaussian.push(GaussianFunction {
                root,
                coordinates,
                log_scale,
            });
        };
        // The variance of the baseline log-rate surface is a function norm;
        // a constant shift has exactly zero penalty and no prior coordinate.
        for h in histories {
            if h.baseline_design.column(0).iter().any(|&v| v != 1.0) {
                return Err(invalid(
                    "joint function priors require a unit baseline intercept in column zero",
                ));
            }
        }
        if b > 1 {
            let mut mean = vec![0.0; b - 1];
            for h in histories {
                let duration = h.exposure.iter().sum::<f64>();
                for n in 0..h.times.len() {
                    for j in 1..b {
                        mean[j - 1] += h.exposure[n] / duration / histories.len() as f64
                            * h.baseline_design[[n, j]];
                    }
                }
            }
            let mut gram = Array2::zeros((b - 1, b - 1));
            for h in histories {
                let duration = h.exposure.iter().sum::<f64>();
                for n in 0..h.times.len() {
                    let row: Vec<_> = (1..b)
                        .map(|j| h.baseline_design[[n, j]] - mean[j - 1])
                        .collect();
                    add_outer(
                        &mut gram,
                        &row,
                        h.exposure[n] / duration / histories.len() as f64,
                    );
                }
            }
            let root = Arc::new(FunctionRoot::from_gram(gram)?);
            for mark in 0..self.spec.marks.len() {
                let base = self.layout.baseline.start + mark * b;
                push(
                    &mut result,
                    FunctionPenalty::BaselineVariation { mark },
                    Arc::clone(&root),
                    vec![base + 1..base + b],
                    None,
                );
            }
        }
        if k == 0 {
            result.add_structural_priors(histories)?;
            return Ok(result);
        }
        // E[(a + b'g)^2] = (a+b'mu)^2 + ||L^{-1}b||^2,
        // where L L' is genetic precision. Construct this root directly;
        // forming mu*mu' first would lose small genetic variances.
        let mut genetic_root = Array2::<f64>::zeros((genes, genes));
        genetic_root[[0, 0]] = 1.0;
        for j in 1..genes {
            genetic_root[[0, j]] = self.spec.genetic_mean[j - 1];
        }
        for column in 1..genes {
            for i in 1..genes {
                let value = f64::from(i == column)
                    - (1..i)
                        .map(|j| self.genetic_factor[[i - 1, j - 1]] * genetic_root[[j, column]])
                        .sum::<f64>();
                genetic_root[[i, column]] = value / self.genetic_factor[[i - 1, i - 1]];
            }
        }
        let genetic = Arc::new(FunctionRoot {
            root: genetic_root,
            log_determinant: -self.genetic_log_determinant,
        });
        let mut drive_gram = Array2::zeros((c, c));
        let mut entry_gram = Array2::zeros((e, e));
        for h in histories {
            let duration = h.times[h.times.len() - 1] - h.times[0];
            for n in 0..h.times.len() - 1 {
                add_outer(
                    &mut drive_gram,
                    &h.drive_design.row(n).to_vec(),
                    (h.times[n + 1] - h.times[n]) / duration / histories.len() as f64,
                );
            }
            let entry: Vec<_> = std::iter::once(1.0)
                .chain(h.entry_design.iter().copied())
                .collect();
            add_outer(&mut entry_gram, &entry, 1.0 / histories.len() as f64);
        }
        let drive = Arc::new(product_root(
            &FunctionRoot::from_gram(drive_gram)?,
            &genetic,
        ));
        let entry = Arc::new(product_root(
            &FunctionRoot::from_gram(entry_gram)?,
            &genetic,
        ));
        push(
            &mut result,
            FunctionPenalty::GeneticDrive,
            drive,
            (0..k)
                .map(|axis| {
                    let start = self.layout.drive.start + axis * c * genes;
                    start..start + c * genes
                })
                .collect(),
            None,
        );
        let entry_width = (e + once) * genes;
        push(
            &mut result,
            FunctionPenalty::EntryMean,
            entry,
            (0..k)
                .map(|axis| {
                    let start = self.layout.entry.start + axis * entry_width;
                    start..start + e * genes
                })
                .collect(),
            None,
        );
        let mut prevalence = 0;
        for (mark, &kind) in self.spec.marks.iter().enumerate() {
            if kind == MarkKind::Once {
                push(
                    &mut result,
                    FunctionPenalty::EntryPrevalence { mark },
                    Arc::clone(&genetic),
                    (0..k)
                        .map(|axis| {
                            let start = self.layout.entry.start
                                + axis * entry_width
                                + (e + prevalence) * genes;
                            start..start + genes
                        })
                        .collect(),
                    None,
                );
                prevalence += 1;
            }
        }
        let unit = Arc::new(FunctionRoot::identity(k));
        for (mark, jump) in self.layout.jumps.iter().enumerate() {
            if let Some(range) = jump {
                push(
                    &mut result,
                    FunctionPenalty::DiseaseJump { mark },
                    Arc::clone(&unit),
                    vec![range.clone()],
                    None,
                );
            }
        }
        for (channel, family) in self.spec.measurements.iter().enumerate() {
            let range = &self.layout.measurement_location[channel];
            let scale = matches!(family, MeasurementFamily::StudentT)
                .then_some(self.layout.measurement_shape[channel].start);
            push(
                &mut result,
                FunctionPenalty::MeasurementEffect { channel },
                Arc::clone(&unit),
                vec![range.start + 1..range.end],
                scale,
            );
        }
        result.add_structural_priors(histories)?;
        Ok(result)
    }
}

impl<'m> JointFunctionPriors<'m> {
    pub(super) fn belongs_to(&self, model: &JointLikelihood) -> bool {
        std::ptr::eq(self.model, model)
    }
    pub fn penalties(&self) -> &[FunctionPenalty] {
        &self.penalties
    }
    pub fn evaluate<'a>(
        &'a self,
        theta: &[f64],
        log_strengths: &[f64],
    ) -> Result<FunctionPriorEvaluation<'a, 'm>, EventHistoryError> {
        self.model.validate_parameters(theta)?;
        if log_strengths.len() != self.penalties.len()
            || log_strengths.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "function prior requires one finite log strength per active function penalty",
            ));
        }
        let decoder = self
            .model
            .decoder_prior(theta, &log_strengths[..self.decoder_strengths])?;
        let mut gradient = decoder.gradient().to_vec();
        let category = self.category.evaluate(theta, &mut gradient);
        let mut result = FunctionPriorEvaluation {
            prior: self,
            log_density: decoder.log_density() + category.log_density(),
            gradient,
            strength_gradient: decoder.log_strength_gradient().to_vec(),
            strength_second: decoder.log_strength_second_derivative().to_vec(),
            decoder,
            gaussian: Vec::with_capacity(self.gaussian.len()),
            structural: Vec::with_capacity(self.structural.len()),
            category,
            baseline_weights: Vec::with_capacity(self.model.spec.marks.len()),
        };
        for (index, function) in self.gaussian.iter().enumerate() {
            let half = 0.5 * log_strengths[self.decoder_strengths + index]
                - function.log_scale.map_or(0.0, |j| theta[j]);
            let rank = function.root.root.ncols() * function.coordinates.len();
            let mut evaluation = GaussianEvaluation {
                half_log_precision: half,
                weighted_penalty: 0.0,
                gradients: Vec::new(),
            };
            for range in &function.coordinates {
                let (energy, positive_gradient) =
                    function.root.quadratic(&theta[range.clone()], half);
                evaluation.weighted_penalty += energy;
                let gradient: Vec<_> = positive_gradient.iter().map(|v| -v).collect();
                for (q, &g) in range.clone().zip(&gradient) {
                    result.gradient[q] += g;
                }
                evaluation.gradients.push(gradient);
            }
            result.log_density += rank as f64 * (half - 0.5 * (2.0 * std::f64::consts::PI).ln())
                + 0.5 * function.coordinates.len() as f64 * function.root.log_determinant
                - evaluation.weighted_penalty;
            let strength_score = 0.5 * rank as f64 - evaluation.weighted_penalty;
            result.strength_gradient.push(strength_score);
            result.strength_second.push(-evaluation.weighted_penalty);
            if let Some(q) = function.log_scale {
                result.gradient[q] -= 2.0 * strength_score;
            }
            result.gaussian.push(evaluation);
        }
        // Independent exponential priors on the dimensionless geometric
        // baseline levels, with one shared learned strength across marks.
        // The change from intercept to level is triangular with the shape
        // coordinates and has Jacobian equal to that positive level.
        let rho = log_strengths[self.decoder_strengths + self.gaussian.len()];
        let b = self.baseline_mean_design.len();
        let mut baseline_weight = 0.0;
        for mark in 0..self.model.spec.marks.len() {
            let start = self.model.layout.baseline.start + mark * b;
            let log_weight = (rho + theta[start])
                + self.log_followup_scale
                + self.baseline_mean_design[1..]
                    .iter()
                    .zip(&theta[start + 1..start + b])
                    .map(|(a, v)| a * v)
                    .sum::<f64>();
            let weight = log_weight.exp();
            result.log_density += log_weight - weight;
            for (j, &a) in self.baseline_mean_design.iter().enumerate() {
                result.gradient[start + j] += (1.0 - weight) * a;
            }
            baseline_weight += weight;
            result.baseline_weights.push(weight);
        }
        result
            .strength_gradient
            .push(self.model.spec.marks.len() as f64 - baseline_weight);
        result.strength_second.push(-baseline_weight);
        for (index, functions) in self.structural.iter().enumerate() {
            let rho = log_strengths[self.decoder_strengths + self.gaussian.len() + 1 + index];
            let mut first = 0.0;
            let mut second = 0.0;
            let mut values = Vec::with_capacity(functions.len());
            for function in functions {
                let value = function.evaluate(theta, rho);
                result.log_density += value.log_density;
                result.gradient[value.coordinate] += value.first;
                first += value.strength_first;
                second += value.strength_second;
                values.push(value);
            }
            result.strength_gradient.push(first);
            result.strength_second.push(second);
            result.structural.push(values);
        }
        if !result.log_density.is_finite()
            || result
                .gradient
                .iter()
                .chain(&result.strength_gradient)
                .chain(&result.strength_second)
                .any(|v| !v.is_finite())
        {
            return Err(numerical(
                "function prior or its derivatives are not representable",
            ));
        }
        Ok(result)
    }
}

impl FunctionPriorEvaluation<'_, '_> {
    pub fn log_density(&self) -> f64 {
        self.log_density
    }
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }
    pub fn log_strength_gradient(&self) -> &[f64] {
        &self.strength_gradient
    }
    /// Strength Hessian is diagonal conditional on coefficients. Integrating
    /// coefficients introduces covariance terms; this is not REML curvature.
    pub fn log_strength_second_derivative(&self) -> &[f64] {
        &self.strength_second
    }
    pub fn negative_hessian_product(
        &self,
        direction: &[f64],
    ) -> Result<Vec<f64>, EventHistoryError> {
        let mut out = self.decoder.negative_hessian_product(direction)?;
        if direction
            .len()
            .checked_mul(32)
            .is_none_or(|n| n > self.prior.memory_limit_bytes)
        {
            return Err(numerical(
                "function prior Hessian product exceeds its workspace budget",
            ));
        }
        for (function, evaluation) in self.prior.gaussian.iter().zip(&self.gaussian) {
            let scale_direction = function.log_scale.map_or(0.0, |j| direction[j]);
            let mut cross = 0.0;
            for (range, gradient) in function.coordinates.iter().zip(&evaluation.gradients) {
                let (_, positive) = function
                    .root
                    .quadratic(&direction[range.clone()], evaluation.half_log_precision);
                for ((q, &g), h) in range.clone().zip(gradient).zip(positive) {
                    out[q] += h + 2.0 * g * scale_direction;
                    cross += 2.0 * g * direction[q];
                }
            }
            if let Some(q) = function.log_scale {
                out[q] += cross + 4.0 * evaluation.weighted_penalty * scale_direction;
            }
        }
        for values in &self.structural {
            for value in values {
                out[value.coordinate] -= value.second * direction[value.coordinate];
            }
        }
        self.category.add_negative_hessian(direction, &mut out);
        let b = self.prior.baseline_mean_design.len();
        for (mark, &weight) in self.baseline_weights.iter().enumerate() {
            let start = self.prior.model.layout.baseline.start + mark * b;
            let delta: f64 = self
                .prior
                .baseline_mean_design
                .iter()
                .zip(&direction[start..start + b])
                .map(|(a, v)| a * v)
                .sum();
            for (j, &a) in self.prior.baseline_mean_design.iter().enumerate() {
                out[start + j] += weight * delta * a;
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "function prior Hessian product is not representable",
            ));
        }
        Ok(out)
    }
    /// Coefficient-by-strength mixed derivative product without storing a
    /// coefficients x strengths matrix.
    pub fn coefficient_strength_product(
        &self,
        direction: &[f64],
    ) -> Result<Vec<f64>, EventHistoryError> {
        if direction.len() != self.strength_gradient.len()
            || direction.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "function prior strength direction has invalid dimensions or values",
            ));
        }
        let mut out = vec![0.0; self.gradient.len()];
        let layout = self.prior.model.layout();
        let k = self.prior.model.spec.signatures;
        for (d, &value) in direction[..self.prior.decoder_strengths].iter().enumerate() {
            for axis in 0..k {
                out[layout.decoder.start + d * k + axis] =
                    self.decoder.coefficient_strength_cross()[[d, axis]] * value;
            }
        }
        for (index, (function, evaluation)) in
            self.prior.gaussian.iter().zip(&self.gaussian).enumerate()
        {
            let value = direction[self.prior.decoder_strengths + index];
            for (range, gradient) in function.coordinates.iter().zip(&evaluation.gradients) {
                for (q, &g) in range.clone().zip(gradient) {
                    out[q] += g * value;
                }
            }
            if let Some(q) = function.log_scale {
                out[q] += 2.0 * evaluation.weighted_penalty * value;
            }
        }
        for (index, values) in self.structural.iter().enumerate() {
            let direction =
                direction[self.prior.decoder_strengths + self.prior.gaussian.len() + 1 + index];
            for value in values {
                out[value.coordinate] += value.mixed * direction;
            }
        }
        let level_direction = direction[self.prior.decoder_strengths + self.prior.gaussian.len()];
        let b = self.prior.baseline_mean_design.len();
        for (mark, &weight) in self.baseline_weights.iter().enumerate() {
            let start = self.prior.model.layout.baseline.start + mark * b;
            for (j, &a) in self.prior.baseline_mean_design.iter().enumerate() {
                out[start + j] -= weight * a * level_direction;
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "function prior mixed derivative product is not representable",
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "function_prior_tests.rs"]
mod test_support;

#[cfg(test)]
mod tests {
    use super::test_support::{category_oracle, structural_oracle};
    use super::*;
    use crate::scalar::Mixed;

    pub(super) fn fixture() -> (JointLikelihood, Vec<JointHistory>, Vec<f64>) {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 2,
            marks: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
            baseline_columns: 2,
            drive_columns: 2,
            entry_columns: 1,
            measurements: vec![
                MeasurementFamily::StudentT,
                MeasurementFamily::BinaryProbit,
                MeasurementFamily::OrdinalProbit { categories: 4 },
                MeasurementFamily::NegativeBinomial,
            ],
            genetic_mean: vec![0.7, -0.3],
            genetic_precision: ndarray::array![[2.0, 0.4], [0.4, 1.0]],
        })
        .unwrap();
        let histories: Vec<_> = [-1.0, 0.5, 2.0]
            .iter()
            .map(|&context| JointHistory {
                times: vec![0.0, 0.5, 1.0, 1.5, 2.0],
                exposure: vec![0.0, 1.0, 0.0, 1.0, 0.0],
                events: vec![None; 5],
                initially_at_risk: vec![true; 3],
                baseline_design: ndarray::array![
                    [1.0, 0.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                    [1.0, 1.5],
                    [1.0, 2.0]
                ],
                drive_design: ndarray::array![[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5]],
                entry_design: vec![context],
                genetics: vec![Some(0.3), None],
                measurements: vec![
                    MeasurementRecord {
                        node: 4,
                        channel: 0,
                        value: Some(0.4),
                        after_event: false,
                    },
                    MeasurementRecord {
                        node: 4,
                        channel: 1,
                        value: Some(1.0),
                        after_event: false,
                    },
                    MeasurementRecord {
                        node: 4,
                        channel: 2,
                        value: Some(2.0),
                        after_event: false,
                    },
                    MeasurementRecord {
                        node: 4,
                        channel: 3,
                        value: Some(3.0),
                        after_event: false,
                    },
                ],
            })
            .collect();
        let theta = (0..model.layout.width)
            .map(|j| 0.2 * (j as f64 + 0.3).cos())
            .collect();
        (model, histories, theta)
    }

    fn oracle<S: JetField>(prior: &JointFunctionPriors<'_>, theta: &[S], rho: &[S]) -> S {
        let k = prior.model.spec.signatures;
        let mut value = theta[0].constant_like(0.0);
        for (d, strength) in rho[..prior.decoder_strengths].iter().enumerate() {
            let lambda = exp(strength);
            let start = prior.model.layout.decoder.start + d * k;
            let mut logits = vec![theta[0].constant_like(0.0)];
            for j in 0..k {
                logits.push(theta[start + j].clone());
                value = value
                    .add(&ln(&add_real(&lambda, (j + 1) as f64)))
                    .add(&theta[start + j]);
            }
            value = value.sub(&add_real(&lambda, (k + 1) as f64).mul(&log_sum_exp(&logits)));
        }
        for (index, function) in prior.gaussian.iter().enumerate() {
            let mut precision = rho[prior.decoder_strengths + index].clone();
            if let Some(q) = function.log_scale {
                precision = precision.sub(&theta[q].scale(2.0));
            }
            let rank = function.root.root.ncols() * function.coordinates.len();
            value = value.add(&precision.scale(0.5 * rank as f64));
            value = add_real(
                &value,
                0.5 * function.coordinates.len() as f64 * function.root.log_determinant
                    - 0.5 * rank as f64 * (2.0 * std::f64::consts::PI).ln(),
            );
            for range in &function.coordinates {
                for row in function.root.root.rows() {
                    let dot = range
                        .clone()
                        .zip(row)
                        .fold(theta[0].constant_like(0.0), |sum, (j, &r)| {
                            sum.add(&theta[j].scale(r))
                        });
                    value = value.sub(&exp(&precision).mul(&dot.mul(&dot)).scale(0.5));
                }
            }
        }
        let b = prior.baseline_mean_design.len();
        for mark in 0..prior.model.spec.marks.len() {
            let start = prior.model.layout.baseline.start + mark * b;
            let log_level = theta[start..start + b]
                .iter()
                .zip(&prior.baseline_mean_design)
                .fold(
                    theta[0].constant_like(prior.log_followup_scale),
                    |v, (x, &a)| v.add(&x.scale(a)),
                );
            let total = rho[prior.decoder_strengths + prior.gaussian.len()].add(&log_level);
            value = value.add(&total).sub(&exp(&total));
        }
        value = value.add(&category_oracle(&prior.category, theta));
        for (index, functions) in prior.structural.iter().enumerate() {
            for function in functions {
                value = value.add(&structural_oracle(
                    function,
                    theta,
                    &rho[prior.decoder_strengths + prior.gaussian.len() + 1 + index],
                ));
            }
        }
        value
    }

    #[test]
    fn positive_baseline_level_prior_has_the_poisson_gamma_limit_even_without_events() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![MarkKind::Recurrent, MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 0,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let history = JointHistory {
            times: vec![0.0, 1.0, 2.0],
            exposure: vec![0.0, 2.0, 0.0],
            events: vec![None; 3],
            initially_at_risk: vec![true; 2],
            baseline_design: Array2::ones((3, 1)),
            drive_design: Array2::zeros((2, 0)),
            entry_design: vec![],
            genetics: vec![],
            measurements: vec![],
        };
        let prior = model.function_priors(&[&history], 1 << 20).unwrap();
        assert_eq!(prior.penalties(), &[FunctionPenalty::BaselineLevel]);
        let (nodes, weights) = gam_math::special::gauss_legendre(129);
        let rho = 0.3_f64;
        for events in [0, 5] {
            let exposure = 3.0;
            let rate = exposure + 2.0 * rho.exp();
            let mut mass = 0.0;
            let mut mean = 0.0;
            for (&node, &weight) in nodes.iter().zip(&weights) {
                let hazard = 50.0 * (node + 1.0) / rate;
                let q = hazard.ln();
                let out = prior.evaluate(&[q, -0.4], &[rho]).unwrap();
                let w = (out.log_density() + events as f64 * q - exposure * hazard - q).exp()
                    * 50.0
                    * weight
                    / rate;
                mass += w;
                mean += w * hazard;
            }
            let expected = (events + 1) as f64 / rate;
            assert!((mean / mass / expected - 1.0).abs() < 1e-12);
        }
        let extreme = prior.evaluate(&[-1e200, -1e200], &[1e200]).unwrap();
        for g in extreme.gradient() {
            assert!((g + 1.0).abs() < 1e-12);
        }
        assert!((extreme.log_strength_gradient()[0] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn function_prior_all_coefficient_strength_and_scale_curvatures_match_the_density() {
        let (model, histories, theta) = fixture();
        let prior = model
            .function_priors(&histories.iter().collect::<Vec<_>>(), 64 << 20)
            .unwrap();
        let rho: Vec<_> = (0..prior.penalties().len())
            .map(|j| 0.5 * (j as f64).sin())
            .collect();
        let out = prior.evaluate(&theta, &rho).unwrap();
        let p = theta.len();
        let total = p + rho.len();
        let hessian: Vec<_> = (0..p)
            .map(|j| {
                let mut d = vec![0.0; p];
                d[j] = 1.0;
                out.negative_hessian_product(&d).unwrap()
            })
            .collect();
        let cross: Vec<_> = (0..rho.len())
            .map(|j| {
                let mut d = vec![0.0; rho.len()];
                d[j] = 1.0;
                out.coefficient_strength_product(&d).unwrap()
            })
            .collect();
        for i in 0..total {
            for j in 0..=i {
                let beta: Vec<_> = theta
                    .iter()
                    .enumerate()
                    .map(|(q, &v)| Mixed::seed(v, f64::from(q == i), f64::from(q == j)))
                    .collect();
                let strengths: Vec<_> = rho
                    .iter()
                    .enumerate()
                    .map(|(q, &v)| Mixed::seed(v, f64::from(p + q == i), f64::from(p + q == j)))
                    .collect();
                let jet = oracle(&prior, &beta, &strengths);
                let gradient = if i < p {
                    out.gradient()[i]
                } else {
                    out.log_strength_gradient()[i - p]
                };
                let second = if i < p {
                    -hessian[j][i]
                } else if j < p {
                    cross[i - p][j]
                } else if i == j {
                    out.log_strength_second_derivative()[i - p]
                } else {
                    0.0
                };
                for (a, b) in [
                    (out.log_density(), jet.base),
                    (gradient, jet.u),
                    (second, jet.uv),
                ] {
                    assert!(
                        (a - b).abs() < 1e-10 * (1.0 + b.abs()),
                        "{i},{j}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn function_energies_equal_direct_integration_of_the_model_functions() {
        let (model, histories, theta) = fixture();
        let prior = model
            .function_priors(&histories.iter().collect::<Vec<_>>(), 64 << 20)
            .unwrap();
        let rho = vec![0.2; prior.penalties().len()];
        let evaluation = prior.evaluate(&theta, &rho).unwrap();
        let nodes = [-3.0_f64.sqrt(), 0.0, 3.0_f64.sqrt()];
        let weights = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
        // Invert the fixture's 2x2 precision independently of its stored
        // genetic root, then integrate the actual complete-law functions.
        let det = 2.0 - 0.4 * 0.4;
        let l00 = (1.0_f64 / det).sqrt();
        let l10 = -0.4 / det / l00;
        let l11 = (2.0 / det - l10 * l10).sqrt();
        for (index, label) in prior.penalties()
            [prior.decoder_strengths..prior.decoder_strengths + prior.gaussian.len()]
            .iter()
            .enumerate()
        {
            let mut energy = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    let gene = [0.7 + l00 * nodes[a], -0.3 + l10 * nodes[a] + l11 * nodes[b]];
                    let weight = weights[a] * weights[b];
                    match *label {
                        FunctionPenalty::GeneticDrive => {
                            for h in &histories {
                                for n in 0..4 {
                                    for axis in 0..2 {
                                        energy += weight / 12.0
                                            * model
                                                .mean(
                                                    &theta,
                                                    model.layout.drive.start,
                                                    &h.drive_design.row(n).to_vec(),
                                                    &gene,
                                                    axis,
                                                )
                                                .powi(2);
                                    }
                                }
                            }
                        }
                        FunctionPenalty::EntryMean | FunctionPenalty::EntryPrevalence { .. } => {
                            for h in &histories {
                                for axis in 0..2 {
                                    let base = [1.0, h.entry_design[0], 0.0];
                                    let mean = model.mean(
                                        &theta,
                                        model.layout.entry.start,
                                        &base,
                                        &gene,
                                        axis,
                                    );
                                    let effect = if matches!(label, FunctionPenalty::EntryMean) {
                                        mean
                                    } else {
                                        model.mean(
                                            &theta,
                                            model.layout.entry.start,
                                            &[1.0, h.entry_design[0], 1.0],
                                            &gene,
                                            axis,
                                        ) - mean
                                    };
                                    energy += weight / 3.0 * effect * effect;
                                }
                            }
                        }
                        FunctionPenalty::DiseaseJump { mark } => {
                            for axis in 0..2 {
                                energy += weight * model.jump(&theta, Some(mark), axis).powi(2);
                            }
                        }
                        FunctionPenalty::MeasurementEffect { channel } => {
                            let start = model.layout.measurement_location[channel].start;
                            let mut value =
                                theta[start + 1] * nodes[a] + theta[start + 2] * nodes[b];
                            if channel == 0 {
                                value *= (-theta[model.layout.measurement_shape[0].start]).exp();
                            }
                            energy += weight * value * value;
                        }
                        FunctionPenalty::BaselineVariation { mark } => {
                            let start = model.layout.baseline.start + 2 * mark;
                            // Equal time mass at t=.5 and t=1.5 gives variance .25.
                            energy += weight * 0.25 * theta[start + 1].powi(2);
                        }
                        FunctionPenalty::Decoder { .. }
                        | FunctionPenalty::TemporalVariation
                        | FunctionPenalty::MeasurementPrecision { .. }
                        | FunctionPenalty::TailVarianceInflation { .. }
                        | FunctionPenalty::CountOverdispersion { .. }
                        | FunctionPenalty::CountMean { .. }
                        | FunctionPenalty::BaselineLevel => unreachable!(),
                    }
                }
            }
            let expected = 0.5 * 0.2_f64.exp() * energy;
            assert!(
                (evaluation.gaussian[index].weighted_penalty - expected).abs() < 1e-13,
                "{label:?}"
            );
        }
    }

    #[test]
    fn function_prior_respects_basis_changes_and_rejects_aliases_and_storage_overruns() {
        let (model, histories, theta) = fixture();
        let prior = model
            .function_priors(&histories.iter().collect::<Vec<_>>(), 64 << 20)
            .unwrap();
        let rho = vec![0.0; prior.penalties().len()];
        let out = prior.evaluate(&theta, &rho).unwrap();
        let mut transformed = histories.clone();
        for h in &mut transformed {
            h.baseline_design
                .column_mut(1)
                .mapv_inplace(|v| 7.0 * v + 3.0);
        }
        let changed = model
            .function_priors(&transformed.iter().collect::<Vec<_>>(), 64 << 20)
            .unwrap();
        let mut beta = theta.clone();
        for mark in 0..3 {
            beta[2 * mark + 1] /= 7.0;
            beta[2 * mark] -= 3.0 * beta[2 * mark + 1];
        }
        let new = changed.evaluate(&beta, &rho).unwrap();
        assert!((new.log_density() - out.log_density() - 3.0 * 7.0_f64.ln()).abs() < 1e-12);
        for (a, b) in new.gaussian.iter().zip(&out.gaussian) {
            assert!((a.weighted_penalty - b.weighted_penalty).abs() < 1e-13);
        }
        for (a, b) in new.baseline_weights.iter().zip(&out.baseline_weights) {
            assert!((a - b).abs() < 1e-13);
        }
        assert!(
            model
                .function_priors(&histories.iter().collect::<Vec<_>>(), 1)
                .is_err()
        );
        for h in &mut transformed {
            h.drive_design.column_mut(1).fill(1.0);
        }
        assert!(
            model
                .function_priors(&transformed.iter().collect::<Vec<_>>(), 64 << 20)
                .is_err()
        );
        assert!(prior.evaluate(&theta, &[]).is_err());
    }

    #[test]
    fn function_prior_preserves_extreme_precision_products_and_student_scale_units() {
        let (model, histories, _) = fixture();
        let prior = model
            .function_priors(&histories.iter().collect::<Vec<_>>(), 64 << 20)
            .unwrap();
        let drive = prior
            .penalties()
            .iter()
            .position(|p| *p == FunctionPenalty::GeneticDrive)
            .unwrap();
        let measurement = prior
            .penalties()
            .iter()
            .position(|p| *p == FunctionPenalty::MeasurementEffect { channel: 0 })
            .unwrap();
        for (strength, coefficient) in [(800.0, 1e-200), (-800.0, 1e200)] {
            let mut theta = vec![0.0; model.layout.width];
            theta[model.layout.drive.start] = coefficient;
            let mut rho = vec![0.0; prior.penalties().len()];
            rho[drive] = strength;
            let out = prior.evaluate(&theta, &rho).unwrap();
            let expected = -(strength + coefficient.ln()).exp();
            assert!((out.gradient()[model.layout.drive.start] / expected - 1.0).abs() < 1e-12);
            let mut direction = vec![0.0; theta.len()];
            direction[model.layout.drive.start] = coefficient;
            let product = out.negative_hessian_product(&direction).unwrap();
            assert!((product[model.layout.drive.start] / (-expected) - 1.0).abs() < 1e-12);
        }
        let mut theta = vec![0.0; model.layout.width];
        theta[model.layout.measurement_location[0].start + 1] = 0.7;
        let mut rho = vec![0.0; prior.penalties().len()];
        let standard = prior.evaluate(&theta, &rho).unwrap();
        rho[measurement] = 800.0;
        let noise = prior
            .penalties()
            .iter()
            .position(|p| *p == FunctionPenalty::MeasurementPrecision { channel: 0 })
            .unwrap();
        rho[noise] = 800.0;
        theta[model.layout.measurement_shape[0].start] = 400.0;
        let shifted = prior.evaluate(&theta, &rho).unwrap();
        assert_eq!(standard.log_density(), shifted.log_density());
        assert_eq!(standard.gradient(), shifted.gradient());
    }
}
