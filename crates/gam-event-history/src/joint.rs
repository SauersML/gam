//! Complete-path probability law for positive latent signatures and measurements.
//!
//! This is the shared density used to construct latent-state inference. It is
//! not a marginal fit: latent trajectories and missing genetic scores remain
//! integration variables. Reference log moments must be supplied by reference
//! evolution, carrying their parameter derivatives when the scalar is a jet.

use crate::chain::log_sum_exp;
use crate::scalar::{add_real, div, exp, ln};
use crate::{EventHistoryError, MarkKind};
use gam_math::nested_dual::JetField;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::ops::Range;

mod cohort;
mod emission;
mod integration;
mod posterior;
mod precision;
mod reference;
mod resolution;
mod score;
pub use cohort::{JointCohortEvaluation, JointCohortIntegration, JointCohortPosterior};
pub use integration::{
    IntegratedLikelihood, IntegratedPosterior, IntegratedScore, IntegrationAccuracy,
    IntegrationOptions, JointIntegration,
};
pub use posterior::{LaplacePosterior, PosteriorOptions};
pub use reference::{
    JointReferenceBank, JointReferenceEvolution, JointReferenceProfile, ReferenceAccuracy,
    ReferenceDiagnostics, ReferenceOptions,
};
pub use resolution::{
    ReferenceResolutionOptions, ReferenceResolutionReport, ResolvedReference,
    ResolvedReferenceEvolution,
};
pub use score::JointPathScore;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MeasurementFamily {
    StudentT,
    BinaryProbit,
    OrdinalProbit { categories: usize },
    NegativeBinomial,
}

impl MeasurementFamily {
    fn shape_width(&self) -> usize {
        match self {
            Self::StudentT => 2,
            Self::BinaryProbit => 0,
            Self::OrdinalProbit { categories } => categories - 2,
            Self::NegativeBinomial => 1,
        }
    }
}

/// Model dimensions and the joint Gaussian law of baseline genetic scores.
/// Basis rows are supplied separately from responses, so the same frozen GAM
/// bases can be used for fitting, entry conditioning, and serving.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointSpecification {
    pub signatures: usize,
    pub marks: Vec<MarkKind>,
    pub baseline_columns: usize,
    pub drive_columns: usize,
    pub entry_columns: usize,
    pub measurements: Vec<MeasurementFamily>,
    pub genetic_mean: Vec<f64>,
    pub genetic_precision: Array2<f64>,
}

/// Offsets in the common coefficient vector. Decoder logits are relative to
/// a background logit fixed at zero. Drives and entry means have one genetic
/// interaction per feature, including the intercept genetic channel.
#[derive(Clone, Debug)]
pub struct JointParameterLayout {
    pub baseline: Range<usize>,
    pub decoder: Range<usize>,
    pub rates: Range<usize>,
    pub drive: Range<usize>,
    pub entry: Range<usize>,
    pub jumps: Vec<Option<Range<usize>>>,
    pub measurement_location: Vec<Range<usize>>,
    pub measurement_shape: Vec<Range<usize>>,
    pub width: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasurementRecord {
    pub node: usize,
    pub channel: usize,
    /// None contributes its integrated likelihood, one.
    pub value: Option<f64>,
    /// A simultaneous measurement can explicitly observe the post-event state.
    pub after_event: bool,
}

/// One observation window on a quadrature/event grid, including its entry.
/// Visits are conditioned-on observation times; informative visits must also
/// be represented as event marks. Entry covariates are observed context, not
/// invented pre-entry event-free follow-up. Once-only prevalence is appended
/// to the entry design automatically.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointHistory {
    pub times: Vec<f64>,
    pub exposure: Vec<f64>,
    pub events: Vec<Option<usize>>,
    pub initially_at_risk: Vec<bool>,
    pub baseline_design: Array2<f64>,
    /// One row per gap, the predictable drive held over that gap.
    pub drive_design: Array2<f64>,
    pub entry_design: Vec<f64>,
    pub genetics: Vec<Option<f64>>,
    pub measurements: Vec<MeasurementRecord>,
}

fn invalid(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::InvalidInput {
        reason: reason.into(),
    }
}

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

fn block(cursor: &mut usize, dimensions: &[usize]) -> Result<Range<usize>, EventHistoryError> {
    let size = dimensions
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| invalid("joint model dimension overflow"))?;
    let end = cursor
        .checked_add(size)
        .ok_or_else(|| invalid("joint parameter count overflow"))?;
    let range = *cursor..end;
    *cursor = end;
    Ok(range)
}

/// Validated, immutable probability-law specification and parameter layout.
pub struct JointLikelihood {
    spec: JointSpecification,
    layout: JointParameterLayout,
    genetic_log_determinant: f64,
    genetic_factor: Array2<f64>,
}

impl JointLikelihood {
    pub fn new(spec: JointSpecification) -> Result<Self, EventHistoryError> {
        let k = spec.signatures;
        let d = spec.marks.len();
        let g = spec.genetic_mean.len();
        if d == 0 || spec.baseline_columns == 0 || (k > 0 && spec.drive_columns == 0) {
            return Err(invalid(
                "joint model needs marks, a baseline basis, and a state drive basis",
            ));
        }
        if spec.measurements.iter().any(|m| {
            matches!(m,
            MeasurementFamily::OrdinalProbit { categories } if *categories < 2)
        }) {
            return Err(invalid("ordinal channels need at least two categories"));
        }
        if spec.genetic_precision.dim() != (g, g)
            || spec
                .genetic_precision
                .iter()
                .chain(&spec.genetic_mean)
                .any(|v| !v.is_finite())
        {
            return Err(invalid(
                "genetic mean and precision must have matching finite dimensions",
            ));
        }
        let mut lower = Array2::<f64>::zeros((g, g));
        let mut genetic_log_determinant = 0.0;
        for i in 0..g {
            for j in 0..=i {
                if spec.genetic_precision[[i, j]] != spec.genetic_precision[[j, i]] {
                    return Err(invalid("genetic precision must be symmetric"));
                }
                let value = spec.genetic_precision[[i, j]]
                    - (0..j).map(|q| lower[[i, q]] * lower[[j, q]]).sum::<f64>();
                if i == j {
                    if !value.is_finite() || value <= 0.0 {
                        return Err(invalid("genetic precision must be positive definite"));
                    }
                    lower[[i, j]] = value.sqrt();
                    genetic_log_determinant += value.ln();
                } else {
                    lower[[i, j]] = value / lower[[j, j]];
                }
            }
        }
        let mut cursor = 0;
        let baseline = block(&mut cursor, &[d, spec.baseline_columns])?;
        let decoder = block(&mut cursor, &[d, k])?;
        let rates = block(&mut cursor, &[k])?;
        let interactions = g
            .checked_add(1)
            .ok_or_else(|| invalid("genetic dimension overflow"))?;
        let drive = block(&mut cursor, &[k, spec.drive_columns, interactions])?;
        let once = spec.marks.iter().filter(|m| **m == MarkKind::Once).count();
        let entry_columns = spec
            .entry_columns
            .checked_add(1)
            .and_then(|v| v.checked_add(once))
            .ok_or_else(|| invalid("entry dimension overflow"))?;
        let entry = block(&mut cursor, &[k, entry_columns, interactions])?;
        let mut jumps = Vec::with_capacity(d);
        for kind in &spec.marks {
            jumps.push(if *kind == MarkKind::Terminal {
                None
            } else {
                Some(block(&mut cursor, &[k])?)
            });
        }
        let mut measurement_location = Vec::new();
        let mut measurement_shape = Vec::new();
        for family in &spec.measurements {
            measurement_location.push(block(&mut cursor, &[k + 1])?);
            measurement_shape.push(block(&mut cursor, &[family.shape_width()])?);
        }
        let layout = JointParameterLayout {
            baseline,
            decoder,
            rates,
            drive,
            entry,
            jumps,
            measurement_location,
            measurement_shape,
            width: cursor,
        };
        Ok(Self {
            spec,
            layout,
            genetic_log_determinant,
            genetic_factor: lower,
        })
    }

    pub fn specification(&self) -> &JointSpecification {
        &self.spec
    }
    pub fn layout(&self) -> &JointParameterLayout {
        &self.layout
    }

    pub fn validate_history(&self, h: &JointHistory) -> Result<(), EventHistoryError> {
        let n = h.times.len();
        if n < 2
            || h.times.iter().any(|x| !x.is_finite())
            || h.times.windows(2).any(|t| t[0] >= t[1])
        {
            return Err(invalid(
                "joint history needs strictly increasing finite times including entry and exit",
            ));
        }
        if h.exposure.len() != n
            || h.events.len() != n
            || h.initially_at_risk.len() != self.spec.marks.len()
            || h.baseline_design.dim() != (n, self.spec.baseline_columns)
            || h.drive_design.dim() != (n - 1, self.spec.drive_columns)
            || h.entry_design.len() != self.spec.entry_columns
            || h.genetics.len() != self.spec.genetic_mean.len()
        {
            return Err(invalid("joint history dimensions do not match the model"));
        }
        if h.baseline_design
            .iter()
            .chain(h.drive_design.iter())
            .chain(&h.entry_design)
            .chain(h.genetics.iter().flatten())
            .any(|x| !x.is_finite())
            || h.exposure.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(invalid(
                "joint designs, observed genetics, and exposures must be finite; exposures must be nonnegative",
            ));
        }
        let duration = h.times[n - 1] - h.times[0];
        if !duration.is_finite()
            || !h.exposure.iter().sum::<f64>().is_finite()
            || h.exposure[0] != 0.0
            || h.events[0].is_some()
            || (h.exposure.iter().sum::<f64>() - duration).abs() > 1e-10 * duration
        {
            return Err(invalid(
                "entry is an unexposed anchor; quadrature exposures must span the observation window",
            ));
        }
        let mut risk = h.initially_at_risk.clone();
        for (d, kind) in self.spec.marks.iter().enumerate() {
            if *kind != MarkKind::Once && !risk[d] {
                return Err(invalid(
                    "entry must be alive and recurrent marks remain at risk",
                ));
            }
        }
        for (node, event) in h.events.iter().enumerate() {
            if let Some(d) = event {
                if *d >= risk.len() || !risk[*d] || h.exposure[node] != 0.0 {
                    return Err(invalid(
                        "an event must name an at-risk mark on a zero-exposure node",
                    ));
                }
                match self.spec.marks[*d] {
                    MarkKind::Recurrent => (),
                    MarkKind::Once => risk[*d] = false,
                    MarkKind::Terminal => {
                        if node != n - 1 {
                            return Err(invalid("a terminal event must end observation"));
                        }
                    }
                }
            }
        }
        for record in &h.measurements {
            if record.node >= n || record.channel >= self.spec.measurements.len() {
                return Err(invalid("measurement names an unknown node or channel"));
            }
            if record.after_event
                && h.events[record.node].is_some_and(|d| self.spec.marks[d] == MarkKind::Terminal)
            {
                return Err(invalid(
                    "a measurement cannot observe a state after termination",
                ));
            }
            if let Some(value) = record.value {
                emission::validate_value(&self.spec.measurements[record.channel], value)?;
            }
        }
        Ok(())
    }

    /// Integration coordinates are missing genetic scores, followed by the
    /// pre-event state at every node (node-major, signatures contiguous).
    pub fn latent_dimension(&self, h: &JointHistory) -> Result<usize, EventHistoryError> {
        self.validate_history(h)?;
        h.times
            .len()
            .checked_mul(self.spec.signatures)
            .and_then(|v| v.checked_add(h.genetics.iter().filter(|g| g.is_none()).count()))
            .ok_or_else(|| invalid("joint latent dimension overflow"))
    }

    fn validate_parameters<S: JetField>(&self, theta: &[S]) -> Result<(), EventHistoryError> {
        if theta.len() != self.layout.width || theta.iter().any(|x| !x.value().is_finite()) {
            return Err(invalid(
                "joint coefficient vector has invalid dimensions or non-finite values",
            ));
        }
        Ok(())
    }

    /// Log of a positive sum of signature activities, with a positive
    /// background. Computed entirely in the log domain, including weights.
    pub fn log_relative_activity<S: JetField>(
        &self,
        theta: &[S],
        mark: usize,
        state: &[S],
    ) -> Result<S, EventHistoryError> {
        self.validate_parameters(theta)?;
        if mark >= self.spec.marks.len()
            || state.len() != self.spec.signatures
            || state.iter().any(|x| !x.value().is_finite())
        {
            return Err(invalid("invalid mark or state for the signature decoder"));
        }
        Ok(self.activity(theta, mark, state))
    }

    fn activity<S: JetField>(&self, theta: &[S], mark: usize, state: &[S]) -> S {
        let mut numerator = vec![theta[0].constant_like(0.0)];
        let mut denominator = numerator.clone();
        for (k, x) in state.iter().enumerate() {
            let weight = &theta[self.layout.decoder.start + mark * self.spec.signatures + k];
            numerator.push(weight.add(&emission::log_softplus(x)));
            denominator.push(weight.clone());
        }
        log_sum_exp(&numerator).sub(&log_sum_exp(&denominator))
    }

    fn mean<S: JetField>(
        &self,
        theta: &[S],
        start: usize,
        columns: &[f64],
        genes: &[S],
        k: usize,
    ) -> S {
        let mut value = theta[0].constant_like(0.0);
        let width = genes.len() + 1;
        for (j, &feature) in columns.iter().enumerate() {
            let base = start + (k * columns.len() + j) * width;
            let mut interaction = theta[base].clone();
            for (g, gene) in genes.iter().enumerate() {
                interaction = interaction.add(&theta[base + g + 1].mul(gene));
            }
            value = value.add(&interaction.scale(feature));
        }
        value
    }

    fn entry_features(&self, h: &JointHistory) -> Vec<f64> {
        let mut entry = vec![1.0];
        entry.extend_from_slice(&h.entry_design);
        for (d, kind) in self.spec.marks.iter().enumerate() {
            if *kind == MarkKind::Once {
                entry.push(f64::from(!h.initially_at_risk[d]));
            }
        }
        entry
    }

    fn jump<S: JetField>(&self, theta: &[S], mark: Option<usize>, k: usize) -> S {
        match mark.and_then(|d| self.layout.jumps[d].as_ref()) {
            Some(range) => theta[range.start + k].clone(),
            None => theta[0].constant_like(0.0),
        }
    }

    /// Joint log density of observations, missing scores, and a latent path.
    /// The reference moments are part of the caller's differentiated reference
    /// evolution, not independent coefficients or frozen score offsets.
    pub fn log_density<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        path: &[S],
        log_reference_moments: &[S],
    ) -> Result<S, EventHistoryError> {
        self.validate_parameters(theta)?;
        let dimension = self.latent_dimension(h)?;
        let marks = self.spec.marks.len();
        let k = self.spec.signatures;
        if path.len() != dimension
            || log_reference_moments.len() != h.times.len() * marks
            || path
                .iter()
                .chain(log_reference_moments)
                .any(|x| !x.value().is_finite())
        {
            return Err(invalid(
                "joint latent path or reference moments have invalid dimensions or non-finite values",
            ));
        }
        let zero = theta[0].constant_like(0.0);
        let mut missing = 0;
        let genes: Vec<S> = h
            .genetics
            .iter()
            .map(|value| match value {
                Some(value) => zero.constant_like(*value),
                None => {
                    let gene = path[missing].clone();
                    missing += 1;
                    gene
                }
            })
            .collect();
        let state = |n: usize| &path[missing + n * k..missing + (n + 1) * k];
        let log_tau = (2.0 * std::f64::consts::PI).ln();
        let mut result =
            zero.constant_like(0.5 * (self.genetic_log_determinant - genes.len() as f64 * log_tau));
        for i in 0..genes.len() {
            let di = add_real(&genes[i], -self.spec.genetic_mean[i]);
            for j in 0..genes.len() {
                let dj = add_real(&genes[j], -self.spec.genetic_mean[j]);
                result = result.sub(&di.mul(&dj).scale(0.5 * self.spec.genetic_precision[[i, j]]));
            }
        }
        let entry = self.entry_features(h);
        for axis in 0..k {
            let mean = self.mean(theta, self.layout.entry.start, &entry, &genes, axis);
            let residual = state(0)[axis].sub(&mean);
            result = result.sub(&add_real(&residual.mul(&residual), log_tau).scale(0.5));
        }
        let rates: Vec<S> = theta[self.layout.rates.clone()]
            .iter()
            .map(emission::softplus)
            .collect();
        for n in 1..h.times.len() {
            let dt = h.times[n] - h.times[n - 1];
            let columns = h.drive_design.row(n - 1).to_vec();
            for axis in 0..k {
                let decay = rates[axis].scale(-dt);
                let phi = exp(&decay);
                let weight = emission::expm1(&decay).neg();
                let variance = emission::expm1(&decay.scale(2.0)).neg();
                if variance.value() <= 0.0 || !variance.value().is_finite() {
                    return Err(numerical("joint OU innovation variance is unresolved"));
                }
                let drive = self.mean(theta, self.layout.drive.start, &columns, &genes, axis);
                let after = state(n - 1)[axis].add(&self.jump(theta, h.events[n - 1], axis));
                let mean = phi.mul(&after).add(&weight.mul(&drive));
                let residual = state(n)[axis].sub(&mean);
                result = result.sub(
                    &add_real(
                        &div(&residual.mul(&residual), &variance).add(&ln(&variance)),
                        log_tau,
                    )
                    .scale(0.5),
                );
            }
        }
        let mut risk = h.initially_at_risk.clone();
        for n in 0..h.times.len() {
            for d in 0..marks {
                if !risk[d] {
                    continue;
                }
                let mut eta = zero.clone();
                for b in 0..self.spec.baseline_columns {
                    eta = eta.add(
                        &theta[self.layout.baseline.start + d * self.spec.baseline_columns + b]
                            .scale(h.baseline_design[[n, b]]),
                    );
                }
                let log_rate = eta
                    .add(&self.activity(theta, d, state(n)))
                    .sub(&log_reference_moments[n * marks + d]);
                if h.exposure[n] > 0.0 {
                    result = result.sub(&exp(&log_rate).scale(h.exposure[n]));
                }
                if h.events[n] == Some(d) {
                    result = result.add(&log_rate);
                }
            }
            if let Some(d) = h.events[n] {
                if self.spec.marks[d] == MarkKind::Once {
                    risk[d] = false;
                }
            }
        }
        for record in &h.measurements {
            let Some(value) = record.value else {
                continue;
            };
            let location = self.layout.measurement_location[record.channel].clone();
            let mut eta = theta[location.start].clone();
            for axis in 0..k {
                let x = if record.after_event {
                    state(record.node)[axis].add(&self.jump(theta, h.events[record.node], axis))
                } else {
                    state(record.node)[axis].clone()
                };
                eta = eta.add(&theta[location.start + axis + 1].mul(&x));
            }
            result = result.add(&emission::log_density(
                &self.spec.measurements[record.channel],
                value,
                &eta,
                &theta[self.layout.measurement_shape[record.channel].clone()],
            )?);
        }
        if !result.value().is_finite() {
            return Err(numerical("non-finite joint path log density"));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reference_tests;
