//! Independent-reference replication and fixed-parameter refinement.
//! Error estimates include Monte Carlo dispersion and observed discretization
//! changes. They are numerical diagnostics, not deterministic error bounds.
use super::*;
use rand::Rng;

#[path = "reference_functional.rs"]
pub(super) mod functional;
pub(super) use functional::{FunctionalAssessment, FunctionalValue};

#[derive(Clone, Debug)]
pub struct ReferenceResolutionOptions {
    pub replicates: usize,
    pub initial_particles: usize,
    pub maximum_particles: usize,
    pub maximum_rounds: usize,
    /// Total retained storage across all three ensembles; transient
    /// scalar/jet evaluations and returned curves are additional.
    pub memory_limit_bytes: usize,
    pub log_moment_tolerance: f64,
    pub risk_mass_tolerance: f64,
    /// Margin in estimated standard errors; no simultaneous coverage claim.
    pub standard_error_multiplier: f64,
    pub maximum_step_hazard: f64,
    pub minimum_risk_effective_samples: f64,
}

impl Default for ReferenceResolutionOptions {
    fn default() -> Self {
        Self {
            replicates: 8,
            initial_particles: 256,
            maximum_particles: 32768,
            maximum_rounds: 8,
            memory_limit_bytes: 512 * 1024 * 1024,
            log_moment_tolerance: 0.01,
            risk_mass_tolerance: 0.01,
            standard_error_multiplier: 3.0,
            maximum_step_hazard: 0.1,
            minimum_risk_effective_samples: 32.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceResolutionReport {
    pub replicates: usize,
    pub particles_per_risk_set: usize,
    pub time_intervals: usize,
    pub rounds: usize,
    pub maximum_log_moment_standard_error: f64,
    pub maximum_risk_mass_standard_error: f64,
    pub maximum_time_log_discrepancy: f64,
    pub maximum_time_risk_discrepancy: f64,
    pub maximum_particle_log_discrepancy: f64,
    pub maximum_particle_risk_discrepancy: f64,
    pub log_error_estimate: f64,
    pub risk_error_estimate: f64,
    pub minimum_risk_effective_samples: f64,
}

pub struct ResolvedReferenceEvolution<S> {
    reference: JointReferenceEvolution<S>,
    report: ReferenceResolutionReport,
}

/// Analytic sensitivities of a value-resolved reference. The report certifies
/// neither sampling nor discretization accuracy of the derivative channels.
pub struct ResolvedReferenceSensitivity {
    sensitivity: JointReferenceSensitivity,
    report: ReferenceResolutionReport,
}

impl ResolvedReferenceSensitivity {
    pub fn reference(&self) -> &JointReferenceEvolution<f64> {
        self.sensitivity.reference()
    }
    pub fn report(&self) -> &ReferenceResolutionReport {
        &self.report
    }
    pub fn log_moment_jacobian(&self) -> &Array2<f64> {
        self.sensitivity.log_moment_jacobian()
    }
    pub fn log_risk_mass_jacobian(&self) -> &Array2<f64> {
        self.sensitivity.log_risk_mass_jacobian()
    }
    pub fn at(&self, times: &[f64]) -> Result<(Vec<f64>, Array2<f64>), EventHistoryError> {
        self.sensitivity.at(times)
    }
    pub(in crate::joint) fn into_evolution(self) -> ResolvedReferenceEvolution<f64> {
        ResolvedReferenceEvolution {
            reference: self.sensitivity.into_reference(),
            report: self.report,
        }
    }
}

impl<S: JetField> ResolvedReferenceEvolution<S> {
    pub fn reference(&self) -> &JointReferenceEvolution<S> {
        &self.reference
    }
    pub fn report(&self) -> &ReferenceResolutionReport {
        &self.report
    }
}

/// Three independent ensembles: coarse time, fine time at the same particle
/// count, and fine time with twice the particles. All banks stay fixed during
/// coefficient/jet evaluation; invalidated accuracy returns an error instead
/// of silently changing the objective's random samples or mesh.
pub struct ResolvedReference<'a> {
    coarse: Vec<JointReferenceBank<'a>>,
    fine: Vec<JointReferenceBank<'a>>,
    large: Vec<JointReferenceBank<'a>>,
    options: ReferenceResolutionOptions,
    step_limits: ReferenceAccuracy,
    particles: usize,
    intervals: usize,
    rounds: usize,
}

struct Aggregate<S> {
    reference: JointReferenceEvolution<S>,
    log_standard_error: Vec<f64>,
    risk_standard_error: Vec<f64>,
}

/// Pool risk masses and risk-weighted activities, not unweighted averages of
/// conditional moments. Each entire interacting population is one replicate.
fn aggregate<S: JetField>(
    curves: &[JointReferenceEvolution<S>],
    times: &[f64],
) -> Result<Aggregate<S>, EventHistoryError> {
    let first = &curves[0];
    let replicates = curves.len();
    let moments: Vec<Vec<S>> = curves
        .iter()
        .map(|c| c.at(times))
        .collect::<Result<_, _>>()?;
    let masses: Vec<Vec<S>> = curves
        .iter()
        .map(|c| c.risk_mass_at(times))
        .collect::<Result<_, _>>()?;
    let mut log_moments = Vec::new();
    let mut log_risk_mass = Vec::new();
    let mut log_standard_error = Vec::new();
    let mut risk_standard_error = Vec::new();
    for i in 0..times.len() * first.marks {
        let mass: Vec<S> = masses.iter().map(|m| m[i].clone()).collect();
        let activity: Vec<S> = masses
            .iter()
            .zip(&moments)
            .map(|(m, a)| m[i].add(&a[i]))
            .collect();
        let mass_sum = log_sum_exp(&mass);
        let activity_sum = log_sum_exp(&activity);
        let moment = activity_sum.sub(&mass_sum);
        let mean_mass = add_real(&mass_sum, -(replicates as f64).ln());
        let log_variance: f64 = activity
            .iter()
            .zip(&mass)
            .map(|(a, m)| {
                ((a.value() - activity_sum.value()).exp() - (m.value() - mass_sum.value()).exp())
                    .powi(2)
            })
            .sum();
        let probability = mean_mass.value().exp();
        let risk_variance: f64 = mass
            .iter()
            .map(|m| (m.value().exp() - probability).powi(2))
            .sum();
        let log_se = (replicates as f64 / (replicates - 1) as f64 * log_variance).sqrt();
        let risk_se = (risk_variance / (replicates * (replicates - 1)) as f64).sqrt();
        if [moment.value(), mean_mass.value(), log_se, risk_se]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(numerical(
                "independent reference replicate estimate is non-finite",
            ));
        }
        log_moments.push(moment);
        log_risk_mass.push(mean_mass);
        log_standard_error.push(log_se);
        risk_standard_error.push(risk_se);
    }
    let diagnostics = ReferenceDiagnostics {
        maximum_log_moment_standard_error: curves
            .iter()
            .map(|c| c.diagnostics.maximum_log_moment_standard_error)
            .fold(0.0, f64::max),
        minimum_risk_effective_samples: curves
            .iter()
            .map(|c| c.diagnostics.minimum_risk_effective_samples)
            .fold(f64::INFINITY, f64::min),
        maximum_step_hazard: curves
            .iter()
            .map(|c| c.diagnostics.maximum_step_hazard)
            .fold(0.0, f64::max),
    };
    Ok(Aggregate {
        reference: JointReferenceEvolution {
            theta: first.theta.clone(),
            times: times.to_vec(),
            log_moments,
            log_risk_mass,
            marks: first.marks,
            diagnostics,
        },
        log_standard_error,
        risk_standard_error,
    })
}

impl ResolvedReference<'_> {
    /// Recheck value resolution at theta, then differentiate the pooled fine
    /// population analytically. Pooling includes risk-mass sensitivities;
    /// averaging conditional moment Jacobians alone would omit selection.
    /// The configured memory limit also bounds additional derivative workspace.
    pub fn sensitivity(
        &self,
        theta: &[f64],
    ) -> Result<ResolvedReferenceSensitivity, EventHistoryError> {
        let value = self.evolve(theta)?;
        Ok(ResolvedReferenceSensitivity {
            sensitivity: self.pooled_sensitivity(theta, &self.large, value.reference)?,
            report: value.report,
        })
    }

    fn pooled_sensitivity(
        &self,
        theta: &[f64],
        banks: &[JointReferenceBank<'_>],
        reference: JointReferenceEvolution<f64>,
    ) -> Result<JointReferenceSensitivity, EventHistoryError> {
        let rows = reference.log_moments.len();
        let bytes = rows
            .checked_mul(theta.len())
            .and_then(|n| n.checked_mul(16))
            .ok_or_else(|| invalid("pooled reference Jacobian dimension overflow"))?;
        let available = self
            .options
            .memory_limit_bytes
            .checked_sub(bytes)
            .ok_or_else(|| {
                numerical("pooled reference Jacobians exceed the derivative workspace budget")
            })?;
        let mut moment = Array2::<f64>::zeros((rows, theta.len()));
        let mut mass = moment.clone();
        let log_count = (banks.len() as f64).ln();
        for bank in banks {
            let next = bank.sensitivity_for_resolution(theta, &self.step_limits, available)?;
            let curve = next.reference();
            if curve.times() != reference.times() {
                return Err(invalid(
                    "reference Jacobian pooling requires identical fine grids",
                ));
            }
            for n in 0..rows {
                let log_mass_weight =
                    curve.log_risk_mass()[n] - reference.log_risk_mass[n] - log_count;
                let mass_weight = log_mass_weight.exp();
                let activity_weight =
                    (log_mass_weight + curve.log_moments()[n] - reference.log_moments[n]).exp();
                for q in 0..theta.len() {
                    let dm = next.log_risk_mass_jacobian()[[n, q]];
                    mass[[n, q]] += mass_weight * dm;
                    moment[[n, q]] += activity_weight * next.log_moment_jacobian()[[n, q]]
                        + (activity_weight - mass_weight) * dm;
                }
            }
        }
        if moment.iter().chain(mass.iter()).any(|v| !v.is_finite()) {
            return Err(numerical("non-finite pooled reference Jacobian"));
        }
        Ok(JointReferenceSensitivity::pooled(
            reference,
            moment,
            mass,
            self.options.memory_limit_bytes,
        ))
    }

    pub(super) fn belongs_to(&self, model: &JointLikelihood) -> bool {
        self.large[0].belongs_to(model)
    }

    fn assess<S: JetField>(
        &self,
        coarse: &[JointReferenceEvolution<S>],
        fine: &[JointReferenceEvolution<S>],
        large: &[JointReferenceEvolution<S>],
    ) -> Result<ResolvedReferenceEvolution<S>, EventHistoryError> {
        let times = large[0].times();
        let c = aggregate(coarse, times)?;
        let f = aggregate(fine, times)?;
        let h = aggregate(large, times)?;
        let mut report = ReferenceResolutionReport {
            replicates: self.options.replicates,
            particles_per_risk_set: self.particles,
            time_intervals: self.intervals,
            rounds: self.rounds,
            maximum_log_moment_standard_error: 0.0,
            maximum_risk_mass_standard_error: 0.0,
            maximum_time_log_discrepancy: 0.0,
            maximum_time_risk_discrepancy: 0.0,
            maximum_particle_log_discrepancy: 0.0,
            maximum_particle_risk_discrepancy: 0.0,
            log_error_estimate: 0.0,
            risk_error_estimate: 0.0,
            minimum_risk_effective_samples: c
                .reference
                .diagnostics
                .minimum_risk_effective_samples
                .min(f.reference.diagnostics.minimum_risk_effective_samples)
                .min(h.reference.diagnostics.minimum_risk_effective_samples),
        };
        for i in 0..h.reference.log_moments.len() {
            let time_log =
                (c.reference.log_moments[i].value() - f.reference.log_moments[i].value()).abs();
            let particle_log =
                (f.reference.log_moments[i].value() - h.reference.log_moments[i].value()).abs();
            let time_risk = (c.reference.log_risk_mass[i].value().exp()
                - f.reference.log_risk_mass[i].value().exp())
            .abs();
            let particle_risk = (f.reference.log_risk_mass[i].value().exp()
                - h.reference.log_risk_mass[i].value().exp())
            .abs();
            let log_margin = self.options.standard_error_multiplier
                * (c.log_standard_error[i].hypot(f.log_standard_error[i])
                    + f.log_standard_error[i].hypot(h.log_standard_error[i])
                    + h.log_standard_error[i]);
            let risk_margin = self.options.standard_error_multiplier
                * (c.risk_standard_error[i].hypot(f.risk_standard_error[i])
                    + f.risk_standard_error[i].hypot(h.risk_standard_error[i])
                    + h.risk_standard_error[i]);
            report.maximum_log_moment_standard_error = report
                .maximum_log_moment_standard_error
                .max(h.log_standard_error[i]);
            report.maximum_risk_mass_standard_error = report
                .maximum_risk_mass_standard_error
                .max(h.risk_standard_error[i]);
            report.maximum_time_log_discrepancy = report.maximum_time_log_discrepancy.max(time_log);
            report.maximum_particle_log_discrepancy =
                report.maximum_particle_log_discrepancy.max(particle_log);
            report.maximum_time_risk_discrepancy =
                report.maximum_time_risk_discrepancy.max(time_risk);
            report.maximum_particle_risk_discrepancy =
                report.maximum_particle_risk_discrepancy.max(particle_risk);
            report.log_error_estimate = report
                .log_error_estimate
                .max(time_log + particle_log + log_margin);
            report.risk_error_estimate = report
                .risk_error_estimate
                .max(time_risk + particle_risk + risk_margin);
        }
        Ok(ResolvedReferenceEvolution {
            reference: h.reference,
            report,
        })
    }

    fn accepted(&self, report: &ReferenceResolutionReport) -> bool {
        report.log_error_estimate <= self.options.log_moment_tolerance
            && report.risk_error_estimate <= self.options.risk_mass_tolerance
            && report.minimum_risk_effective_samples >= self.options.minimum_risk_effective_samples
    }

    /// Recheck all three ensembles at the requested parameters. The value
    /// diagnostics do not certify derivative errors; jets differentiate the
    /// identical pooled, finite-reference objective used for the value.
    pub fn evolve<S: JetField>(
        &self,
        theta: &[S],
    ) -> Result<ResolvedReferenceEvolution<S>, EventHistoryError> {
        let evaluate = |banks: &[JointReferenceBank<'_>]| {
            banks
                .iter()
                .map(|b| b.evolve_for_resolution(theta, &self.step_limits))
                .collect::<Result<Vec<_>, _>>()
        };
        let result = self.assess(
            &evaluate(&self.coarse)?,
            &evaluate(&self.fine)?,
            &evaluate(&self.large)?,
        )?;
        if !self.accepted(&result.report) {
            return Err(numerical(format!(
                "joint reference accuracy invalidated at these coefficients: log error {}, risk error {}, minimum risk ESS {}; resolve again outside the objective evaluation",
                result.report.log_error_estimate,
                result.report.risk_error_estimate,
                result.report.minimum_risk_effective_samples
            )));
        }
        Ok(result)
    }
}

impl JointLikelihood {
    pub fn resolve_reference<'a, R: Rng + ?Sized>(
        &'a self,
        theta: &[f64],
        profile: &JointReferenceProfile,
        options: &ReferenceResolutionOptions,
        rng: &mut R,
    ) -> Result<(ResolvedReference<'a>, ResolvedReferenceEvolution<f64>), EventHistoryError> {
        self.validate_parameters(theta)?;
        if options.replicates < 4
            || options.initial_particles < 2
            || options.maximum_rounds == 0
            || options
                .initial_particles
                .checked_mul(2)
                .is_none_or(|p| p > options.maximum_particles)
            || [
                options.log_moment_tolerance,
                options.risk_mass_tolerance,
                options.standard_error_multiplier,
                options.maximum_step_hazard,
                options.minimum_risk_effective_samples,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || options.standard_error_multiplier < 1.0
            || options.minimum_risk_effective_samples < 2.0
            || options.minimum_risk_effective_samples > options.initial_particles as f64
        {
            return Err(invalid(
                "invalid joint reference resolution tolerances, replicate count, or resource bounds",
            ));
        }
        let banks = options
            .replicates
            .checked_mul(3)
            .ok_or_else(|| invalid("reference replicate count overflow"))?;
        let bank_memory = options.memory_limit_bytes / banks;
        let step_limits = ReferenceAccuracy {
            log_moment_standard_error: 1.0,
            minimum_risk_effective_samples: options.minimum_risk_effective_samples,
            maximum_step_hazard: options.maximum_step_hazard,
        };
        let mut current = profile.clone();
        let mut particles = options.initial_particles;
        let mut last_report = None;
        for round in 1..=options.maximum_rounds {
            let high_particles = particles
                .checked_mul(2)
                .filter(|p| *p <= options.maximum_particles)
                .ok_or_else(|| {
                    numerical(
                        "joint reference reached its particle budget before resolving accuracy",
                    )
                })?;
            // Build the coarse bank first: validation precedes any indexing
            // of the caller's design while constructing the refined profile.
            let mut build = |profile: &JointReferenceProfile, particles| {
                let mut banks = Vec::new();
                let mut curves = Vec::new();
                for _ in 0..options.replicates {
                    let (bank, curve) = self.reference_bank_generated(
                        theta,
                        profile,
                        &ReferenceOptions {
                            particles,
                            memory_limit_bytes: bank_memory,
                        },
                        &step_limits,
                        rng,
                        false,
                    )?;
                    banks.push(bank);
                    curves.push(curve);
                }
                Ok::<_, EventHistoryError>((banks, curves))
            };
            let coarse = match build(&current, particles) {
                Ok(group) => group,
                Err(EventHistoryError::ReferenceStep { .. }) => {
                    current = current.refined()?;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let finer = current.refined()?;
            let fine = match build(&finer, particles) {
                Ok(group) => group,
                Err(EventHistoryError::ReferenceStep { .. }) => {
                    current = finer;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let large = match build(&finer, high_particles) {
                Ok(group) => group,
                Err(EventHistoryError::ReferenceStep { .. }) => {
                    current = finer;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let resolved = ResolvedReference {
                coarse: coarse.0,
                fine: fine.0,
                large: large.0,
                options: options.clone(),
                step_limits: step_limits.clone(),
                particles: high_particles,
                intervals: finer.times.len() - 1,
                rounds: round,
            };
            let result = resolved.assess(&coarse.1, &fine.1, &large.1)?;
            if resolved.accepted(&result.report) {
                return Ok((resolved, result));
            }
            let report = &result.report;
            let time = (report.maximum_time_log_discrepancy / options.log_moment_tolerance)
                .max(report.maximum_time_risk_discrepancy / options.risk_mass_tolerance);
            let particle = (report.maximum_particle_log_discrepancy / options.log_moment_tolerance)
                .max(report.maximum_particle_risk_discrepancy / options.risk_mass_tolerance);
            let sampling = options.standard_error_multiplier
                * (report.maximum_log_moment_standard_error / options.log_moment_tolerance)
                    .max(report.maximum_risk_mass_standard_error / options.risk_mass_tolerance);
            if report.minimum_risk_effective_samples < options.minimum_risk_effective_samples
                || sampling > 0.2
                || particle >= time
            {
                particles = high_particles;
            } else {
                current = finer;
            }
            last_report = Some(result.report);
        }
        Err(numerical(format!(
            "joint reference did not resolve within {} rounds; last comparison: {last_report:?}",
            options.maximum_rounds
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_replicates_pool_risk_weighted_moments_and_measure_between_bank_error() {
        let curves: Vec<_> = (1..=4)
            .map(|r| JointReferenceEvolution {
                theta: vec![0.0],
                times: vec![0.0, 1.0],
                marks: 1,
                log_moments: vec![0.0, (r as f64).ln()],
                log_risk_mass: vec![0.0, (r as f64 / 5.0).ln()],
                diagnostics: ReferenceDiagnostics {
                    maximum_log_moment_standard_error: 0.0,
                    minimum_risk_effective_samples: 1000.0,
                    maximum_step_hazard: 0.01,
                },
            })
            .collect();
        let combined = aggregate(&curves, &[0.0, 1.0]).unwrap();
        assert!((combined.reference.log_moments[1] - 3.0_f64.ln()).abs() < 1e-14);
        assert!((combined.reference.log_risk_mass[1] - 0.5_f64.ln()).abs() < 1e-14);
        assert!(combined.log_standard_error[1] > 0.0);
        assert!((combined.risk_standard_error[1] - (0.2_f64 / 12.0).sqrt()).abs() < 1e-14);
        assert_eq!(combined.log_standard_error[0], 0.0);
        assert_eq!(combined.risk_standard_error[0], 0.0);
    }
}
