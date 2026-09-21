//! Independent-reference replication and fixed-parameter refinement.
//! Error estimates combine dispersion between independent populations with
//! observed time and particle discrepancies. They are numerical diagnostics,
//! not deterministic error bounds: two-level differences can underestimate
//! discretization error, and a finite replicate sample can miss rare behavior.
use super::reference::{JointReferenceBank, ReferenceGrid, materialization_budget};
use super::*;
use rand::{SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use std::sync::Arc;

#[path = "reference_functional.rs"]
mod functional;
pub(in crate::joint) use functional::{
    FunctionalAssessment, FunctionalPoint, FunctionalValue, ReferencePullback,
};

/// Retained storage across all three ensembles is bounded by this machine's
/// materialization budget; refinement fails explicitly beyond it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceResolutionOptions {
    /// Independent populations in each ensemble; at least two, so that one can
    /// be deleted from a pooled estimate.
    pub replicates: usize,
    /// Starting particle count in each conditional risk population.
    pub initial_particles: usize,
    /// Error budgets derived by the caller from its own accuracy requirement.
    pub log_moment_tolerance: f64,
    pub risk_mass_tolerance: f64,
    /// Declared probability, under the replicate central-limit approximation,
    /// that sampling error exceeds its margin in some cell of the returned
    /// curve. The margin quantile is Student t with R - 1 degrees of freedom at
    /// two-sided level `sampling_error_rate / m` over its m cells (Bonferroni).
    pub sampling_error_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// The largest coarse/fine discrepancy beyond its own sampling band,
    /// `margin (se_coarse^2 + se_fine^2)^(1/2)`: only this part calls for time
    /// refinement, since a discrepancy at its noise floor is sampling error.
    pub maximum_time_log_excess: f64,
    pub maximum_time_risk_excess: f64,
    pub log_error_estimate: f64,
    pub risk_error_estimate: f64,
    pub minimum_risk_effective_samples: f64,
    /// Largest fine-grid ensemble's omitted multiple-event mass. It is the
    /// diagnostic for leaving the asymptotic regime the time term assumes, and
    /// is reported rather than added, because its effect on a functional
    /// depends on the jumps it omits.
    pub omitted_event_mass: f64,
    /// The Student quantile multiplying the replicate standard errors.
    pub margin_quantile: f64,
}

/// Per stored row (times x marks) of a resolved curve: the replicate standard
/// errors of its pooled estimate, and the observed coarse/fine time
/// discrepancy. With `e(h) = C h^p`, `e(h) - e(h/2) >= e(h/2)` exactly when
/// `p >= 1`, so the discrepancy bounds the fine grid's discretization bias for
/// a time scheme of order at least one in its asymptotic, monotone regime; the
/// reported omitted event mass is the diagnostic for leaving that regime.
#[derive(Clone, Debug)]
pub struct ReferenceCellErrors {
    pub log_moment_standard_error: Vec<f64>,
    pub risk_mass_standard_error: Vec<f64>,
    pub time_log_discrepancy: Vec<f64>,
    pub time_risk_discrepancy: Vec<f64>,
}

pub struct ResolvedReferenceEvolution<S> {
    reference: JointReferenceEvolution<S>,
    report: ReferenceResolutionReport,
    cells: ReferenceCellErrors,
}

impl<S: JetField> ResolvedReferenceEvolution<S> {
    pub fn reference(&self) -> &JointReferenceEvolution<S> {
        &self.reference
    }
    pub fn report(&self) -> &ReferenceResolutionReport {
        &self.report
    }
    pub fn cells(&self) -> &ReferenceCellErrors {
        &self.cells
    }
}

/// Everything that regenerates a resolved reference exactly: the anchor at
/// which its event histories were proposed, the declared profile, the
/// tolerances, the seed, and the accepted round's refinement. The banks
/// themselves are never stored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedReference {
    pub anchor: Vec<f64>,
    pub profile: JointReferenceProfile,
    pub options: ReferenceResolutionOptions,
    pub seed: u64,
    pub round: usize,
    /// Times the declared profile was halved for the coarse ensemble.
    pub time_refinements: usize,
    /// Particles per risk set in the coarse and fine ensembles.
    pub particles: usize,
}

/// Three independent ensembles over one declared profile: coarse time, fine
/// time at the same particle count, and fine time with twice the particles.
/// All banks stay fixed during coefficient/jet evaluation; invalidated
/// accuracy returns an error instead of silently changing the objective's
/// random samples or mesh.
pub struct ResolvedReference {
    model: Arc<JointLikelihood>,
    anchor: Vec<f64>,
    profile: JointReferenceProfile,
    options: ReferenceResolutionOptions,
    seed: u64,
    round: usize,
    time_refinements: usize,
    particles: usize,
    intervals: usize,
    coarse: Vec<JointReferenceBank>,
    fine: Vec<JointReferenceBank>,
    large: Vec<JointReferenceBank>,
}

struct Aggregate<S> {
    reference: JointReferenceEvolution<S>,
    log_standard_error: Vec<f64>,
    risk_standard_error: Vec<f64>,
}

fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The generator of one population, keyed only by the seed and the
/// population's place in its round, so a saved round regenerates without
/// replaying earlier rounds and whatever evaluates the steps.
fn population_rng(seed: u64, round: usize, ensemble: usize, replicate: usize) -> SmallRng {
    let key = [round as u64, ensemble as u64, replicate as u64]
        .into_iter()
        .fold(splitmix64(seed), |key, part| splitmix64(key ^ splitmix64(part)));
    SmallRng::seed_from_u64(key)
}

fn validate_options(options: &ReferenceResolutionOptions) -> Result<(), EventHistoryError> {
    if options.replicates < 2
        || options.initial_particles < 2
        || [options.log_moment_tolerance, options.risk_mass_tolerance]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        || !(options.sampling_error_rate > 0.0 && options.sampling_error_rate < 1.0)
    {
        return Err(invalid(
            "joint reference resolution needs two or more replicates and particles, positive finite tolerances, and a sampling error rate in (0, 1)",
        ));
    }
    Ok(())
}

/// The two-sided Student quantile `t` with `P(|T_nu| >= t) = tail`, from
/// `P(|T_nu| >= t) = I_{nu/(nu + t^2)}(nu/2, 1/2)`.
fn student_quantile(tail: f64, freedom: f64) -> Result<f64, EventHistoryError> {
    let x = gam_math::probability::beta_quantile(tail, 0.5 * freedom, 0.5);
    let t = (freedom * (1.0 / x - 1.0)).sqrt();
    if !(t.is_finite() && t > 0.0) {
        return Err(numerical(format!(
            "no representable Student quantile at two-sided tail {tail} with {freedom} degrees of freedom"
        )));
    }
    Ok(t)
}

/// Pool risk masses and risk-weighted activities, not unweighted averages of
/// conditional moments. Each entire interacting population is one replicate.
fn aggregate<S: JetField>(
    curves: &[JointReferenceEvolution<S>],
    times: &[f64],
    row_times: &[S],
) -> Result<Aggregate<S>, EventHistoryError> {
    let first = &curves[0];
    let replicates = curves.len();
    let moments: Vec<Vec<S>> = curves
        .iter()
        .map(|c| c.at_over(times, row_times))
        .collect::<Result<_, _>>()?;
    let masses: Vec<Vec<S>> = curves
        .iter()
        .map(|c| c.risk_mass_over(times, row_times))
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
        minimum_risk_effective_samples: curves
            .iter()
            .map(|c| c.diagnostics.minimum_risk_effective_samples)
            .fold(f64::INFINITY, f64::min),
        omitted_event_mass: curves
            .iter()
            .map(|c| c.diagnostics.omitted_event_mass)
            .fold(0.0, f64::max),
    };
    Ok(Aggregate {
        reference: JointReferenceEvolution {
            theta: first.theta.clone(),
            times: times.to_vec(),
            row_times: row_times.to_vec(),
            log_moments,
            log_risk_mass,
            marks: first.marks,
            diagnostics,
        },
        log_standard_error,
        risk_standard_error,
    })
}

impl ResolvedReference {
    /// One comparison round over `grid`, the declared profile halved
    /// `time_refinements` times: coarse and fine ensembles at `particles` per
    /// risk set, and a fine ensemble with twice as many.
    fn round(
        model: &Arc<JointLikelihood>,
        theta: &[f64],
        profile: &JointReferenceProfile,
        grid: &ReferenceGrid,
        time_refinements: usize,
        particles: usize,
        round: usize,
        seed: u64,
        options: &ReferenceResolutionOptions,
    ) -> Result<(Self, ResolvedReferenceEvolution<f64>), EventHistoryError> {
        let finer = grid.refined()?;
        let high_particles = particles
            .checked_mul(2)
            .ok_or_else(|| invalid("joint reference particle count overflow"))?;
        let mut bytes = 0usize;
        for (profile, count) in [(grid, particles), (&finer, particles), (&finer, high_particles)] {
            bytes = JointReferenceBank::storage_bytes(model, profile, count)?
                .checked_mul(options.replicates)
                .and_then(|b| b.checked_add(bytes))
                .ok_or_else(|| invalid("joint reference storage overflow"))?;
        }
        let budget = materialization_budget();
        if bytes > budget {
            return Err(numerical(format!(
                "joint reference round {round} needs {bytes} bank bytes, above this machine's {budget}-byte materialization budget"
            )));
        }
        // Each population owns its stream and takes all of its draws itself, so
        // replicates are generated in parallel and collected in replicate order.
        let build = |ensemble: usize, profile: &ReferenceGrid, count: usize| {
            let generated: Vec<_> = (0..options.replicates)
                .into_par_iter()
                .map(|replicate| {
                    let mut rng = population_rng(seed, round, ensemble, replicate);
                    JointReferenceBank::generate(model, theta, profile, count, &mut rng)
                })
                .collect();
            let mut banks = Vec::with_capacity(options.replicates);
            let mut curves = Vec::with_capacity(options.replicates);
            for population in generated {
                let (bank, curve) = population?;
                banks.push(bank);
                curves.push(curve);
            }
            Ok::<_, EventHistoryError>((banks, curves))
        };
        let coarse = build(0, grid, particles)?;
        let fine = build(1, &finer, particles)?;
        let large = build(2, &finer, high_particles)?;
        let resolved = Self {
            model: Arc::clone(model),
            anchor: theta.to_vec(),
            profile: profile.clone(),
            options: options.clone(),
            seed,
            round,
            time_refinements,
            particles,
            intervals: finer.nodes() - 1,
            coarse: coarse.0,
            fine: fine.0,
            large: large.0,
        };
        let result = resolved.assess(&coarse.1, &fine.1, &large.1)?;
        Ok((resolved, result))
    }

    /// Resolve the declared profile at fixed coefficients, outside any
    /// objective evaluation. Each round compares the three ensembles and
    /// refines whichever of time or particles dominates the error estimate,
    /// until both estimates meet their tolerances. Every round grows retained
    /// storage, so the materialization budget is the only other outcome.
    pub fn resolve(
        model: &Arc<JointLikelihood>,
        theta: &[f64],
        profile: &JointReferenceProfile,
        options: &ReferenceResolutionOptions,
        seed: u64,
    ) -> Result<(Self, ResolvedReferenceEvolution<f64>), EventHistoryError> {
        model.validate_parameters(theta)?;
        profile.validate(model)?;
        validate_options(options)?;
        let mut grid = ReferenceGrid::new(profile.clone(), 0)?;
        let mut time_refinements = 0;
        let mut particles = options.initial_particles;
        let mut last_report = None;
        let mut round = 0;
        loop {
            round += 1;
            let (resolved, result) = match Self::round(
                model,
                theta,
                profile,
                &grid,
                time_refinements,
                particles,
                round,
                seed,
                options,
            ) {
                Ok(pair) => pair,
                Err(EventHistoryError::NumericalFailure { reason }) => {
                    return Err(numerical(format!(
                        "{reason}; last comparison: {last_report:?}"
                    )));
                }
                Err(error) => return Err(error),
            };
            if resolved.accepted(&result.report) {
                return Ok((resolved, result));
            }
            let report = &result.report;
            // Refine time only for a discrepancy beyond its own sampling band;
            // at the noise floor, halving the step leaves the estimate in place.
            let time = (report.maximum_time_log_excess / options.log_moment_tolerance)
                .max(report.maximum_time_risk_excess / options.risk_mass_tolerance);
            let sampling = ((report.maximum_particle_log_discrepancy
                + report.margin_quantile * report.maximum_log_moment_standard_error)
                / options.log_moment_tolerance)
                .max(
                    (report.maximum_particle_risk_discrepancy
                        + report.margin_quantile * report.maximum_risk_mass_standard_error)
                        / options.risk_mass_tolerance,
                );
            if sampling >= time {
                particles = particles
                    .checked_mul(2)
                    .ok_or_else(|| invalid("joint reference particle count overflow"))?;
            } else {
                grid = grid.refined()?;
                time_refinements += 1;
            }
            last_report = Some(result.report);
        }
    }

    /// The regeneration record of this reference.
    pub fn saved(&self) -> SavedReference {
        SavedReference {
            anchor: self.anchor.clone(),
            profile: self.profile.clone(),
            options: self.options.clone(),
            seed: self.seed,
            round: self.round,
            time_refinements: self.time_refinements,
            particles: self.particles,
        }
    }

    /// Regenerate a saved reference's accepted round and recheck it.
    pub fn restore(
        model: &Arc<JointLikelihood>,
        saved: &SavedReference,
    ) -> Result<(Self, ResolvedReferenceEvolution<f64>), EventHistoryError> {
        model.validate_parameters(&saved.anchor)?;
        saved.profile.validate(model)?;
        validate_options(&saved.options)?;
        let refinements = u32::try_from(saved.time_refinements).map_err(|_| {
            invalid("a saved joint reference declares more refinements than a grid represents")
        })?;
        let grid = ReferenceGrid::new(saved.profile.clone(), refinements)?;
        let (resolved, result) = Self::round(
            model,
            &saved.anchor,
            &saved.profile,
            &grid,
            saved.time_refinements,
            saved.particles,
            saved.round,
            saved.seed,
            &saved.options,
        )?;
        if !resolved.accepted(&result.report) {
            return Err(EventHistoryError::IntegrationResolution {
                reason: format!(
                    "a restored joint reference no longer meets its tolerances: log error {}, risk error {}",
                    result.report.log_error_estimate, result.report.risk_error_estimate
                ),
            });
        }
        Ok((resolved, result))
    }

    /// The declared profile: origin, horizon, and the functions evolved over it.
    pub fn profile(&self) -> &JointReferenceProfile {
        &self.profile
    }

    /// Gradient of `moment_adjoint' m` for the resolved log moments on the
    /// pooled fine grid, `reference().times() x marks`, node-major. Pooling
    /// shares the adjoint between the retained populations by their
    /// risk-weighted activity and mass, including the selection through risk
    /// mass, and one reverse sweep per population returns its part. Accuracy is
    /// rechecked at `theta` first; the report certifies no derivative error.
    pub fn pullback<S: JetField + Send + Sync>(
        &self,
        theta: &[S],
        moment_adjoint: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        let value = self.evolve(theta)?;
        let pooled = value.reference();
        if moment_adjoint.len() != pooled.log_moments.len() {
            return Err(invalid(
                "reference pullback adjoint needs one entry per pooled row and mark",
            ));
        }
        let curves = self.evolve_all(&self.large, theta)?;
        let masses = vec![theta[0].constant_like(0.0); moment_adjoint.len()];
        self.pooled_pullback(theta, &self.large, &curves, None, moment_adjoint, &masses)
    }

    /// Every population's evolution at `theta`. Populations are independent,
    /// so they evolve in parallel; results and the first failure are taken in
    /// replicate order.
    fn evolve_all<S: JetField + Send + Sync>(
        &self,
        banks: &[JointReferenceBank],
        theta: &[S],
    ) -> Result<Vec<JointReferenceEvolution<S>>, EventHistoryError> {
        banks
            .par_iter()
            .map(|b| b.evolve(&self.model, theta))
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }

    /// Gradient of `moment_adjoint' m + mass_adjoint' M` for the curve pooled
    /// from `banks` at `theta`, leaving out `removed`. With a population's
    /// activity share `a_r` and mass share `m_r` over the populations still
    /// pooled, pooled `log M = log sum_r M_r - log R` and pooled
    /// `log m = log sum_r M_r m_r - log sum_r M_r`, so its rows receive the
    /// moment adjoint `g a_r` and the mass adjoint `g (a_r - m_r) + h m_r`, and
    /// one reverse sweep returns its part.
    fn pooled_pullback<S: JetField + Send + Sync>(
        &self,
        theta: &[S],
        banks: &[JointReferenceBank],
        curves: &[JointReferenceEvolution<S>],
        removed: Option<usize>,
        moment_adjoint: &[S],
        mass_adjoint: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        let rows = curves[0].log_moments.len();
        if banks.len() != curves.len()
            || moment_adjoint.len() != rows
            || mass_adjoint.len() != rows
            || curves.iter().any(|c| c.times != curves[0].times)
        {
            return Err(invalid(
                "a pooled reference pullback needs one curve per population on one grid and one adjoint per row",
            ));
        }
        let included: Vec<usize> = (0..banks.len()).filter(|&r| Some(r) != removed).collect();
        // The populations' reverse sweeps run in parallel, so their workspaces
        // are admitted together, from counts alone.
        let mut bytes = 0usize;
        for &r in &included {
            bytes = banks[r]
                .pullback_bytes::<S>(&self.model, theta.len())?
                .checked_add(bytes)
                .ok_or_else(|| invalid("pooled reference pullback workspace overflow"))?;
        }
        let budget = materialization_budget();
        if bytes > budget {
            return Err(numerical(format!(
                "a pooled reference pullback needs {bytes} workspace bytes, above this machine's {budget}-byte materialization budget"
            )));
        }
        let mut mass_total = Vec::with_capacity(rows);
        let mut activity_total = Vec::with_capacity(rows);
        let mut mass_terms = Vec::with_capacity(included.len());
        let mut activity_terms = Vec::with_capacity(included.len());
        for i in 0..rows {
            mass_terms.clear();
            activity_terms.clear();
            for &r in &included {
                mass_terms.push(curves[r].log_risk_mass[i].clone());
                activity_terms.push(curves[r].log_risk_mass[i].add(&curves[r].log_moments[i]));
            }
            mass_total.push(log_sum_exp(&mass_terms));
            activity_total.push(log_sum_exp(&activity_terms));
        }
        let parts: Vec<Result<Vec<S>, EventHistoryError>> = included
            .par_iter()
            .map(|&r| {
                let mut moment = Vec::with_capacity(rows);
                let mut mass = Vec::with_capacity(rows);
                for i in 0..rows {
                    let mass_share = exp(&curves[r].log_risk_mass[i].sub(&mass_total[i]));
                    let activity_share = exp(&curves[r].log_risk_mass[i]
                        .add(&curves[r].log_moments[i])
                        .sub(&activity_total[i]));
                    moment.push(moment_adjoint[i].mul(&activity_share));
                    mass.push(
                        moment_adjoint[i]
                            .mul(&activity_share.sub(&mass_share))
                            .add(&mass_adjoint[i].mul(&mass_share)),
                    );
                }
                banks[r].pullback(&self.model, theta, &moment, &mass)
            })
            .collect();
        let mut theta_bar = vec![theta[0].constant_like(0.0); theta.len()];
        for part in parts {
            for (slot, value) in theta_bar.iter_mut().zip(&part?) {
                *slot = slot.add(value);
            }
        }
        Ok(theta_bar)
    }

    fn assess<S: JetField>(
        &self,
        coarse: &[JointReferenceEvolution<S>],
        fine: &[JointReferenceEvolution<S>],
        large: &[JointReferenceEvolution<S>],
    ) -> Result<ResolvedReferenceEvolution<S>, EventHistoryError> {
        let times = large[0].times();
        let row_times = &large[0].row_times;
        let c = aggregate(coarse, times, row_times)?;
        let f = aggregate(fine, times, row_times)?;
        let h = aggregate(large, times, row_times)?;
        let rows = h.reference.log_moments.len();
        let margin = student_quantile(
            self.options.sampling_error_rate / (2 * rows) as f64,
            (self.options.replicates - 1) as f64,
        )?;
        let mut report = ReferenceResolutionReport {
            replicates: self.options.replicates,
            particles_per_risk_set: 2 * self.particles,
            time_intervals: self.intervals,
            rounds: self.round,
            maximum_log_moment_standard_error: 0.0,
            maximum_risk_mass_standard_error: 0.0,
            maximum_time_log_discrepancy: 0.0,
            maximum_time_risk_discrepancy: 0.0,
            maximum_particle_log_discrepancy: 0.0,
            maximum_particle_risk_discrepancy: 0.0,
            maximum_time_log_excess: 0.0,
            maximum_time_risk_excess: 0.0,
            log_error_estimate: 0.0,
            risk_error_estimate: 0.0,
            minimum_risk_effective_samples: c
                .reference
                .diagnostics
                .minimum_risk_effective_samples
                .min(f.reference.diagnostics.minimum_risk_effective_samples)
                .min(h.reference.diagnostics.minimum_risk_effective_samples),
            omitted_event_mass: f
                .reference
                .diagnostics
                .omitted_event_mass
                .max(h.reference.diagnostics.omitted_event_mass),
            margin_quantile: margin,
        };
        let mut cells = ReferenceCellErrors {
            log_moment_standard_error: h.log_standard_error.clone(),
            risk_mass_standard_error: h.risk_standard_error.clone(),
            time_log_discrepancy: Vec::with_capacity(rows),
            time_risk_discrepancy: Vec::with_capacity(rows),
        };
        for i in 0..rows {
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
            cells.time_log_discrepancy.push(time_log);
            cells.time_risk_discrepancy.push(time_risk);
            let log_margin = margin
                * (c.log_standard_error[i].hypot(f.log_standard_error[i])
                    + f.log_standard_error[i].hypot(h.log_standard_error[i])
                    + h.log_standard_error[i]);
            let risk_margin = margin
                * (c.risk_standard_error[i].hypot(f.risk_standard_error[i])
                    + f.risk_standard_error[i].hypot(h.risk_standard_error[i])
                    + h.risk_standard_error[i]);
            report.maximum_time_log_excess = report.maximum_time_log_excess.max(
                time_log - margin * c.log_standard_error[i].hypot(f.log_standard_error[i]),
            );
            report.maximum_time_risk_excess = report.maximum_time_risk_excess.max(
                time_risk - margin * c.risk_standard_error[i].hypot(f.risk_standard_error[i]),
            );
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
            cells,
        })
    }

    /// Both error estimates within their budgets. The time part of each
    /// estimate is the coarse/fine discrepancy, which bounds the returned fine
    /// grid's discretization bias only for a time scheme of order at least one
    /// in its asymptotic, monotone regime (`e(h) - e(h/2) >= e(h/2)` iff
    /// `p >= 1`). The scheme is first order: at most one non-killing event per
    /// step. The reported `omitted_event_mass`, the probability mass needing
    /// several events in one step, is the diagnostic for leaving that regime.
    /// It is not added to the risk estimate, because its effect on a
    /// functional depends on the jumps it omits, and bounding it as mass
    /// alone needs about `Lambda^2 / (2 tol)` intervals at population hazards.
    fn accepted(&self, report: &ReferenceResolutionReport) -> bool {
        report.log_error_estimate <= self.options.log_moment_tolerance
            && report.risk_error_estimate <= self.options.risk_mass_tolerance
    }

    /// Recheck all three ensembles at the requested parameters. The value
    /// diagnostics do not certify derivative errors; jets differentiate the
    /// identical pooled, finite-reference objective used for the value.
    pub fn evolve<S: JetField + Send + Sync>(
        &self,
        theta: &[S],
    ) -> Result<ResolvedReferenceEvolution<S>, EventHistoryError> {
        let result = self.assess(
            &self.evolve_all(&self.coarse, theta)?,
            &self.evolve_all(&self.fine, theta)?,
            &self.evolve_all(&self.large, theta)?,
        )?;
        if !self.accepted(&result.report) {
            return Err(EventHistoryError::IntegrationResolution {
                reason: format!(
                    "joint reference accuracy invalidated at these coefficients: log error {}, risk error {}; resolve again outside the objective evaluation",
                    result.report.log_error_estimate, result.report.risk_error_estimate
                ),
            });
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::{Bound, agrees};
    use super::*;

    /// The pooled derivative with its common-mode rows removed. Both routes
    /// pool the same replicate curves, so rounding in those rows moves both
    /// identically; their accuracy is certified separately by the refinement
    /// band and by replay matching the anchor. Feeding the rows as exact values
    /// into the production pooled pullback and into the oracle's pooling
    /// leaves each route's bound counting its own derivative propagation:
    /// shares and adjoints in production, derivative channels in the oracle.
    /// The all-bound arm on the same banks is printed beside it.
    #[test]
    fn pooled_pullback_agrees_with_jets_over_common_mode_rows() {
        use crate::joint::reference_tests::{Arm, judge};
        use crate::scalar::Rows;
        let model = Arc::new(
            JointLikelihood::new(JointSpecification {
                signatures: 2,
                marks: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
                baseline_columns: 1,
                drive_columns: 1,
                entry_columns: 0,
                measurements: vec![],
                genetic_mean: vec![0.0],
                genetic_precision: Array2::eye(1),
            })
            .unwrap(),
        );
        let jump = model.layout.jumps[0].as_ref().unwrap().start;
        let mut theta: Vec<f64> = (0..model.layout.width)
            .map(|q| 0.1 * (q as f64).sin())
            .collect();
        theta[0] = -1.2;
        theta[1] = -1.4;
        theta[2] = -2.0;
        theta[model.layout.drive.start + 1] = 0.3;
        theta[model.layout.entry.start + 1] = 0.4;
        theta[jump] = 0.6;
        let profile = JointReferenceProfile {
            times: (0..=16).map(|n| n as f64 / 16.0).collect(),
            baseline_design: Array2::ones((17, 1)),
            drive_design: Array2::ones((16, 1)),
            entry_design: vec![],
            genetics: vec![None],
        };
        let options = ReferenceResolutionOptions {
            replicates: 16,
            initial_particles: 64,
            log_moment_tolerance: 0.5,
            risk_mass_tolerance: 0.1,
            sampling_error_rate: 1e-3,
        };
        let (resolved, _) = ResolvedReference::resolve(&model, &theta, &profile, &options, 971).unwrap();
        let bounded: Vec<Bound> = theta.iter().map(|&v| Bound::exact(v)).collect();
        let curves = resolved.evolve_all(&resolved.large, &bounded).unwrap();
        let times = curves[0].times.clone();
        let rows = curves[0].log_moments.len();
        let adjoint: Vec<Bound> = (0..rows).map(|i| Bound::exact((0.29 * i as f64).cos())).collect();
        let masses = vec![Bound::exact(0.0); rows];
        let exact_rows = |curve: &JointReferenceEvolution<Bound>| JointReferenceEvolution {
            theta: curve.theta.clone(),
            times: curve.times.clone(),
            row_times: curve.row_times.clone(),
            log_moments: curve.log_moments.iter().map(|b| Bound::exact(b.value)).collect(),
            log_risk_mass: curve.log_risk_mass.iter().map(|b| Bound::exact(b.value)).collect(),
            marks: curve.marks,
            diagnostics: curve.diagnostics.clone(),
        };
        let common: Vec<_> = curves.iter().map(exact_rows).collect();
        let all_bound = resolved
            .pooled_pullback(&bounded, &resolved.large, &curves, None, &adjoint, &masses)
            .unwrap();
        let common_mode = resolved
            .pooled_pullback(&bounded, &resolved.large, &common, None, &adjoint, &masses)
            .unwrap();
        let mut cells = Vec::new();
        for q in [0, model.layout.entry.start + 1, model.layout.rates.start, jump] {
            let seeds: Vec<Rows<Bound, 1>> = theta
                .iter()
                .enumerate()
                .map(|(j, &v)| Rows::seed(Bound::exact(v), [f64::from(j == q)]))
                .collect();
            let jets = resolved.evolve_all(&resolved.large, &seeds).unwrap();
            let directional = |pooled: &JointReferenceEvolution<Rows<Bound, 1>>| {
                pooled
                    .log_moments
                    .iter()
                    .zip(&adjoint)
                    .fold(Bound::exact(0.0), |acc, (m, a)| acc.add(&m.rows[0].mul(a)))
            };
            let oracle_all = directional(&aggregate(&jets, &times, &jets[0].row_times).unwrap().reference);
            let stripped: Vec<_> = jets
                .iter()
                .map(|curve| JointReferenceEvolution {
                    theta: curve.theta.clone(),
                    times: curve.times.clone(),
                    row_times: curve.row_times.clone(),
                    log_moments: curve
                        .log_moments
                        .iter()
                        .map(|m| Rows { base: Bound::exact(m.base.value), rows: m.rows })
                        .collect(),
                    log_risk_mass: curve
                        .log_risk_mass
                        .iter()
                        .map(|m| Rows { base: Bound::exact(m.base.value), rows: m.rows })
                        .collect(),
                    marks: curve.marks,
                    diagnostics: curve.diagnostics.clone(),
                })
                .collect();
            let oracle_common = directional(&aggregate(&stripped, &times, &stripped[0].row_times).unwrap().reference);
            for (arm, production, oracle) in [
                ("all", all_bound[q], oracle_all),
                ("common", common_mode[q], oracle_common),
            ] {
                println!(
                    "POOLED arm {arm} coefficient {q}: oracle {} production {} mu_oracle {} mu_production {} bar/|oracle| {}",
                    oracle.value,
                    production.value,
                    oracle.scale,
                    production.scale,
                    production.bar(&oracle) / oracle.value.abs()
                );
            }
            cells.push((q, common_mode[q], oracle_common));
        }
        let judged: Vec<_> = cells
            .iter()
            .map(|(q, production, oracle)| {
                (format!("common-mode pooled coefficient {q}"), *production, *oracle, Arm::Agreement)
            })
            .collect();
        judge(&judged);
    }

    #[test]
    fn reference_replicates_pool_risk_weighted_moments_and_measure_between_bank_error() {
        let curves: Vec<_> = (1..=4)
            .map(|r| JointReferenceEvolution {
                theta: vec![Bound::exact(0.0)],
                times: vec![0.0, 1.0],
                row_times: vec![Bound::exact(0.0), Bound::exact(1.0)],
                marks: 1,
                log_moments: vec![Bound::exact(0.0), ln(&Bound::exact(r as f64))],
                log_risk_mass: vec![Bound::exact(0.0), ln(&Bound::exact(r as f64 / 8.0))],
                diagnostics: ReferenceDiagnostics {
                    minimum_risk_effective_samples: 1000.0,
                    omitted_event_mass: 0.01 * r as f64,
                },
            })
            .collect();
        let combined =
            aggregate(&curves, &[0.0, 1.0], &[Bound::exact(0.0), Bound::exact(1.0)]).unwrap();
        // Risk-weighted over masses r/8: (1*1 + 2*2 + 3*3 + 4*4)/(1+2+3+4) = 3,
        // where the unweighted mean of the moments would be 2.5; the mean mass
        // is 10/32. Both targets and every input mass are exactly representable.
        agrees(
            &combined.reference.log_moments[1],
            &ln(&Bound::exact(3.0)),
            "pooled log moment",
        );
        agrees(
            &combined.reference.log_risk_mass[1],
            &ln(&Bound::exact(10.0 / 32.0)),
            "pooled log mass",
        );
        assert!(combined.log_standard_error[1] > 0.0);
        assert!(combined.risk_standard_error[1] > 0.0);
        assert_eq!(combined.log_standard_error[0], 0.0);
        assert_eq!(combined.risk_standard_error[0], 0.0);
        assert_eq!(combined.reference.diagnostics.omitted_event_mass, 0.01 * 4.0);
    }

    #[test]
    fn population_streams_are_distinct_per_place_and_repeatable() {
        use rand::RngExt;
        let mut draws = Vec::new();
        for (seed, round, ensemble, replicate) in
            [(7, 1, 0, 0), (7, 1, 0, 1), (7, 1, 1, 0), (7, 2, 0, 0), (8, 1, 0, 0)]
        {
            draws.push(population_rng(seed, round, ensemble, replicate).random::<u64>());
        }
        for i in 0..draws.len() {
            for j in 0..i {
                assert_ne!(draws[i], draws[j], "streams {i} and {j} coincide");
            }
        }
        assert_eq!(population_rng(7, 1, 0, 1).random::<u64>(), draws[1]);
    }

    #[test]
    fn student_quantile_inverts_the_two_sided_tail() {
        // A quantile resolved to working precision lies strictly between the
        // tail probabilities at relative perturbations of sqrt(eps) in t.
        let delta = f64::EPSILON.sqrt();
        for (tail, freedom) in [(0.05, 1.0), (1e-3 / 40.0, 7.0), (1e-8, 15.0), (0.5, 30.0)] {
            let t = student_quantile(tail, freedom).unwrap();
            let lower =
                gam_math::probability::student_t_two_sided_probability(t * (1.0 - delta), freedom);
            let upper =
                gam_math::probability::student_t_two_sided_probability(t * (1.0 + delta), freedom);
            assert!(
                lower > tail && tail > upper,
                "tail {tail}, nu {freedom}: t {t} gives tails {upper}..{lower}"
            );
        }
        // Positive control: Student t with one degree of freedom is Cauchy,
        // whose two-sided 5% point is tan(0.475 pi).
        let cauchy = student_quantile(0.05, 1.0).unwrap();
        assert!((cauchy / (0.475 * std::f64::consts::PI).tan() - 1.0).abs() < delta);
    }

    /// Acceptance A2 at fixed coefficients: two independent rounds of one
    /// declared reference, from independent streams at different time and
    /// particle resolutions, agree in every compared cell within
    /// `t (se_a^2 + se_b^2)^(1/2) + time_a + time_b`. `t` is the Student
    /// quantile with R - 1 degrees of freedom at two-sided level alpha/m over
    /// the m compared cells (Bonferroni), for the declared family-wise false
    /// failure rate alpha = 1e-3 under the replicate central-limit
    /// approximation. `time` is each round's observed coarse/fine discrepancy,
    /// which bounds its fine-grid discretization bias for a time scheme of
    /// order at least one. A round with the recurrent jump removed must exceed
    /// the same band in some cell, so the band detects a change of that size.
    #[test]
    fn independent_reference_refinements_agree_within_the_derived_band() {
        let alpha = 1e-3;
        let model = Arc::new(
            JointLikelihood::new(JointSpecification {
                signatures: 1,
                marks: vec![MarkKind::Recurrent, MarkKind::Once],
                baseline_columns: 1,
                drive_columns: 1,
                entry_columns: 0,
                measurements: vec![],
                genetic_mean: vec![0.0],
                genetic_precision: Array2::eye(1),
            })
            .unwrap(),
        );
        let jump = model.layout.jumps[0].as_ref().unwrap().start;
        let mut theta = vec![0.0; model.layout.width];
        theta[model.layout.baseline.start] = -1.0;
        theta[model.layout.baseline.start + 1] = -1.5;
        theta[model.layout.decoder.start] = 2.0;
        theta[model.layout.drive.start + 1] = 0.3;
        theta[model.layout.entry.start + 1] = 0.4;
        theta[jump] = 3.0;
        let declared = |steps: usize| JointReferenceProfile {
            times: (0..=steps).map(|n| n as f64 / steps as f64).collect(),
            baseline_design: Array2::ones((steps + 1, 1)),
            drive_design: Array2::ones((steps, 1)),
            entry_design: vec![],
            genetics: vec![None],
        };
        // Rounds at declared resolutions, so no stopping tolerance is consulted.
        let options = ReferenceResolutionOptions {
            replicates: 8,
            initial_particles: 512,
            log_moment_tolerance: f64::MAX,
            risk_mass_tolerance: f64::MAX,
            sampling_error_rate: alpha,
        };
        let round = |theta: &[f64], steps: usize, particles: usize, seed: u64| {
            let grid = declared(steps);
            ResolvedReference::round(&model, theta, &grid, &grid, 0, particles, 1, seed, &options)
                .unwrap()
                .1
        };
        let first = round(&theta, 8, 512, 101);
        let second = round(&theta, 16, 1024, 202);
        let mut no_jump = theta.clone();
        no_jump[jump] = 0.0;
        let removed = round(&no_jump, 8, 512, 303);
        let times = [0.0, 0.25, 0.5, 0.75, 1.0];
        let marks = 2;
        let compared = 2 * times.len() * marks;
        let t = student_quantile(alpha / compared as f64, (options.replicates - 1) as f64).unwrap();
        // The largest excess of a cell's gap over its band, and the widest band.
        let excess = |a: &ResolvedReferenceEvolution<f64>, b: &ResolvedReferenceEvolution<f64>| {
            let (ca, cb) = (a.cells(), b.cells());
            let mut worst = f64::NEG_INFINITY;
            let mut widest: f64 = 0.0;
            for &time in &times {
                let ra = a.reference().times().iter().position(|&s| s == time).unwrap();
                let rb = b.reference().times().iter().position(|&s| s == time).unwrap();
                for d in 0..marks {
                    let (ia, ib) = (ra * marks + d, rb * marks + d);
                    let log_band = t
                        * ca.log_moment_standard_error[ia].hypot(cb.log_moment_standard_error[ib])
                        + ca.time_log_discrepancy[ia]
                        + cb.time_log_discrepancy[ib];
                    let risk_band = t
                        * ca.risk_mass_standard_error[ia].hypot(cb.risk_mass_standard_error[ib])
                        + ca.time_risk_discrepancy[ia]
                        + cb.time_risk_discrepancy[ib];
                    let log_gap =
                        (a.reference().log_moments()[ia] - b.reference().log_moments()[ib]).abs();
                    let risk_gap = (a.reference().log_risk_mass()[ia].exp()
                        - b.reference().log_risk_mass()[ib].exp())
                    .abs();
                    worst = worst.max(log_gap - log_band).max(risk_gap - risk_band);
                    widest = widest.max(log_band);
                }
            }
            (worst, widest)
        };
        let (agreement, band) = excess(&first, &second);
        let (control, control_band) = excess(&first, &removed);
        eprintln!(
            "A2 band: t {t}, agreement excess {agreement}, widest log band {band}; removed-jump excess {control}, widest log band {control_band}"
        );
        assert!(
            agreement <= 0.0,
            "a cell exceeds its band by {agreement}; widest log band {band}"
        );
        assert!(
            control > 0.0,
            "a removed jump must exceed the band: excess {control}, widest log band {control_band}"
        );
    }
}
