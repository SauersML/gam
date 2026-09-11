//! Weighted reference evolution with genetic states and disease jumps.
//! OU half steps surround a competing-event step. This is a finite-time
//! approximation: sampling diagnostics do not certify its time error.
use super::precision::{Factorization, Precision};
use super::*;
use crate::scalar::sqrt;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};

#[path = "reference_sensitivity.rs"]
mod sensitivity;
pub use sensitivity::JointReferenceSensitivity;
#[path = "reference_rank_zero.rs"]
mod rank_zero;

/// Origin population, initially alive and free of every once-only mark.
/// A late-origin profile is a declared entry law, not conditioning on an
/// unobserved disease-free interval before that origin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointReferenceProfile {
    pub times: Vec<f64>,
    pub baseline_design: Array2<f64>,
    pub drive_design: Array2<f64>,
    pub entry_design: Vec<f64>,
    pub genetics: Vec<Option<f64>>,
}

impl JointReferenceProfile {
    fn history(&self, marks: usize) -> JointHistory {
        let mut exposure = vec![0.0; self.times.len()];
        for n in 1..self.times.len() {
            exposure[n] = self.times[n] - self.times[n - 1];
        }
        JointHistory {
            times: self.times.clone(),
            exposure,
            events: vec![None; self.times.len()],
            initially_at_risk: vec![true; marks],
            baseline_design: self.baseline_design.clone(),
            drive_design: self.drive_design.clone(),
            entry_design: self.entry_design.clone(),
            genetics: self.genetics.clone(),
            measurements: vec![],
        }
    }

    pub(super) fn midpoint(left: f64, right: f64) -> Result<f64, EventHistoryError> {
        let mid = left + 0.5 * (right - left);
        if !mid.is_finite() || mid <= left || mid >= right {
            return Err(numerical(
                "joint reference interval has no representable interior midpoint",
            ));
        }
        Ok(mid)
    }

    /// Split the declared piecewise-linear baseline and piecewise-constant
    /// drive without changing either function during a time comparison.
    pub(super) fn refined(&self) -> Result<Self, EventHistoryError> {
        let n = self.times.len();
        let rows = n
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| invalid("joint reference refinement dimension overflow"))?;
        let mut times = Vec::with_capacity(rows);
        for pair in self.times.windows(2) {
            times.push(pair[0]);
            times.push(Self::midpoint(pair[0], pair[1])?);
        }
        times.push(self.times[n - 1]);
        let baseline_design =
            Array2::from_shape_fn((rows, self.baseline_design.ncols()), |(i, j)| {
                if i % 2 == 0 {
                    self.baseline_design[[i / 2, j]]
                } else {
                    0.5 * self.baseline_design[[i / 2, j]]
                        + 0.5 * self.baseline_design[[i / 2 + 1, j]]
                }
            });
        let drive_design =
            Array2::from_shape_fn((rows - 1, self.drive_design.ncols()), |(i, j)| {
                self.drive_design[[i / 2, j]]
            });
        Ok(Self {
            times,
            baseline_design,
            drive_design,
            entry_design: self.entry_design.clone(),
            genetics: self.genetics.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ReferenceOptions {
    /// Particle count in each required conditional risk population.
    pub particles: usize,
    /// Retained bank storage; transient scalar/jet work is additional.
    pub memory_limit_bytes: usize,
}

impl Default for ReferenceOptions {
    fn default() -> Self {
        Self {
            particles: 4096,
            memory_limit_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Within-bank diagnostic limits and a local event-step limit. Shared
/// normalization couples particles, so plug-in standard errors do not capture
/// every dependence term. Independent replicates and time/particle refinement
/// must establish reference accuracy before fitting or serving a resolved model.
#[derive(Clone, Debug)]
pub struct ReferenceAccuracy {
    pub log_moment_standard_error: f64,
    pub minimum_risk_effective_samples: f64,
    /// Per-particle pre-jump total hazard times step. This limits the frozen
    /// rate exposure; it does not bound omitted multiple events when a jump
    /// increases subsequent rates, or establish the total time error.
    pub maximum_step_hazard: f64,
}

impl Default for ReferenceAccuracy {
    fn default() -> Self {
        Self {
            log_moment_standard_error: 0.01,
            minimum_risk_effective_samples: 64.0,
            maximum_step_hazard: 0.1,
        }
    }
}

/// Dispersion within this bank; neither a time-error certificate nor an
/// independent-replicate uncertainty estimate for the interacting population.
#[derive(Clone, Debug)]
pub struct ReferenceDiagnostics {
    pub maximum_log_moment_standard_error: f64,
    pub minimum_risk_effective_samples: f64,
    pub maximum_step_hazard: f64,
}

/// One jointly evaluated coefficient/reference state. Stored times include
/// interval midpoints; interpolation never extrapolates beyond its origin
/// and horizon. Reference sensitivities stay attached to their coefficients.
pub struct JointReferenceEvolution<S> {
    pub(super) theta: Vec<S>,
    pub(super) times: Vec<f64>,
    pub(super) log_moments: Vec<S>,
    pub(super) log_risk_mass: Vec<S>,
    pub(super) marks: usize,
    pub(super) diagnostics: ReferenceDiagnostics,
}

impl<S: JetField> JointReferenceEvolution<S> {
    pub fn coefficients(&self) -> &[S] {
        &self.theta
    }
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    pub fn log_moments(&self) -> &[S] {
        &self.log_moments
    }
    pub fn log_risk_mass(&self) -> &[S] {
        &self.log_risk_mass
    }
    pub fn diagnostics(&self) -> &ReferenceDiagnostics {
        &self.diagnostics
    }

    pub fn at(&self, times: &[f64]) -> Result<Vec<S>, EventHistoryError> {
        self.interpolate(&self.log_moments, times)
    }

    pub(super) fn risk_mass_at(&self, times: &[f64]) -> Result<Vec<S>, EventHistoryError> {
        self.interpolate(&self.log_risk_mass, times)
    }

    fn interpolate(&self, values: &[S], times: &[f64]) -> Result<Vec<S>, EventHistoryError> {
        let mut out = Vec::with_capacity(times.len() * self.marks);
        for &time in times {
            if !time.is_finite() || time < self.times[0] || time > self.times[self.times.len() - 1]
            {
                return Err(invalid(
                    "joint reference request is outside its supported origin/horizon",
                ));
            }
            let right = self.times.partition_point(|&t| t < time);
            if right == 0 || self.times[right] == time {
                out.extend_from_slice(&values[right * self.marks..(right + 1) * self.marks]);
            } else {
                let fraction =
                    (time - self.times[right - 1]) / (self.times[right] - self.times[right - 1]);
                for d in 0..self.marks {
                    out.push(
                        values[(right - 1) * self.marks + d]
                            .scale(1.0 - fraction)
                            .add(&values[right * self.marks + d].scale(fraction)),
                    );
                }
            }
        }
        Ok(out)
    }
}

/// Fixed normal innovations and fixed proposed event histories. The event
/// proposal was generated at an anchor, but its probabilities are held as
/// sampling data only. Re-evaluation differentiates target/proposal weights
/// as well as the OU states and the evolving risk-set moments.
pub struct JointReferenceBank<'a> {
    model: &'a JointLikelihood,
    profile: JointReferenceProfile,
    genes: Vec<Vec<f64>>,
    normals: Vec<f64>,
    choices: Vec<Option<usize>>,
    log_proposal: Vec<f64>,
    groups: Vec<Option<usize>>,
    mark_groups: Vec<usize>,
    particles: usize,
}

struct Population<S> {
    states: Vec<Vec<S>>,
    log_weight: Vec<S>,
    risk: Vec<Vec<bool>>,
    log_mass: Vec<S>,
}

impl JointLikelihood {
    pub fn reference_bank<'a, R: Rng + ?Sized>(
        &'a self,
        theta: &[f64],
        profile: &JointReferenceProfile,
        options: &ReferenceOptions,
        accuracy: &ReferenceAccuracy,
        rng: &mut R,
    ) -> Result<JointReferenceBank<'a>, EventHistoryError> {
        Ok(self
            .reference_bank_generated(theta, profile, options, accuracy, rng, true)?
            .0)
    }

    pub(super) fn reference_bank_generated<'a, R: Rng + ?Sized>(
        &'a self,
        theta: &[f64],
        profile: &JointReferenceProfile,
        options: &ReferenceOptions,
        accuracy: &ReferenceAccuracy,
        rng: &mut R,
        check_within_bank: bool,
    ) -> Result<(JointReferenceBank<'a>, JointReferenceEvolution<f64>), EventHistoryError> {
        self.validate_parameters(theta)?;
        self.validate_history(&profile.history(self.spec.marks.len()))?;
        for pair in profile.times.windows(2) {
            JointReferenceProfile::midpoint(pair[0], pair[1])?;
        }
        if options.particles < 2 {
            return Err(invalid("joint reference needs at least two particles"));
        }
        let k = self.spec.signatures;
        let n = profile.times.len();
        let mut groups = Vec::new();
        let mut mark_groups = Vec::new();
        for (d, kind) in self.spec.marks.iter().enumerate() {
            let key = (*kind == MarkKind::Once).then_some(d);
            let index = match groups.iter().position(|g| *g == key) {
                Some(index) => index,
                None => {
                    groups.push(key);
                    groups.len() - 1
                }
            };
            mark_groups.push(index);
        }
        let count = options
            .particles
            .checked_mul(groups.len())
            .ok_or_else(|| invalid("joint reference population dimension overflow"))?;
        let normal_count = n
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| n.checked_mul(k))
            .and_then(|n| n.checked_mul(count))
            .ok_or_else(|| invalid("joint reference noise dimension overflow"))?;
        let steps = (n - 1)
            .checked_mul(count)
            .ok_or_else(|| invalid("joint reference history dimension overflow"))?;
        let bytes = normal_count
            .checked_mul(8)
            .and_then(|b| {
                steps
                    .checked_mul(8 + std::mem::size_of::<Option<usize>>())
                    .and_then(|s| b.checked_add(s))
            })
            .and_then(|b| {
                count
                    .checked_mul(profile.genetics.len())
                    .and_then(|g| g.checked_mul(8))
                    .and_then(|g| b.checked_add(g))
            })
            .and_then(|b| {
                count
                    .checked_mul(std::mem::size_of::<Vec<f64>>())
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| {
                profile
                    .times
                    .len()
                    .checked_add(profile.baseline_design.len())
                    .and_then(|v| v.checked_add(profile.drive_design.len()))
                    .and_then(|v| v.checked_add(profile.entry_design.len()))
                    .and_then(|v| v.checked_mul(8))
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| {
                profile
                    .genetics
                    .len()
                    .checked_mul(std::mem::size_of::<Option<f64>>())
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| b.checked_add(std::mem::size_of::<JointReferenceBank<'_>>()))
            .and_then(|b| {
                groups
                    .len()
                    .checked_mul(std::mem::size_of::<Option<usize>>())
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| {
                mark_groups
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|v| b.checked_add(v))
            })
            .ok_or_else(|| invalid("joint reference storage overflow"))?;
        if bytes > options.memory_limit_bytes {
            return Err(numerical(format!(
                "joint reference needs {bytes} bank bytes, above its {}-byte limit",
                options.memory_limit_bytes
            )));
        }
        let missing: Vec<usize> = profile
            .genetics
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.is_none().then_some(i))
            .collect();
        let mut precision = Precision::new(0, 0, missing.len());
        let mut information = vec![0.0; missing.len()];
        for (i, &gi) in missing.iter().enumerate() {
            for (j, &gj) in missing.iter().enumerate() {
                precision.corner[[i, j]] = self.spec.genetic_precision[[gi, gj]];
                information[i] += precision.corner[[i, j]] * self.spec.genetic_mean[gj];
            }
            for (gj, observed) in profile.genetics.iter().enumerate() {
                if let Some(value) = observed {
                    information[i] -= self.spec.genetic_precision[[gi, gj]]
                        * (value - self.spec.genetic_mean[gj]);
                }
            }
        }
        let factor = Factorization::new(&precision)?;
        let mean = factor.solve(&information);
        let mut genes = Vec::with_capacity(count);
        for _ in 0..count {
            let z: Vec<f64> = (0..missing.len())
                .map(|_| StandardNormal.sample(rng))
                .collect();
            let draw = factor.gaussian_draw(&z);
            let mut genome: Vec<f64> = profile.genetics.iter().map(|v| v.unwrap_or(0.0)).collect();
            for (i, &g) in missing.iter().enumerate() {
                genome[g] = mean[i] + draw[i];
            }
            if genome.iter().any(|g| !g.is_finite()) {
                return Err(numerical("joint reference genetic draw is unresolved"));
            }
            genes.push(genome);
        }
        let normals: Vec<f64> = (0..normal_count)
            .map(|_| StandardNormal.sample(rng))
            .collect();
        let mut choices = vec![None; steps];
        let mut log_proposal = vec![0.0; steps];
        let mut bank = JointReferenceBank {
            model: self,
            profile: profile.clone(),
            genes,
            normals,
            choices: vec![],
            log_proposal: vec![],
            groups,
            mark_groups,
            particles: options.particles,
        };
        let evolution = bank.run(
            theta,
            accuracy,
            |index, probabilities| {
                let total: f64 = probabilities.iter().map(|(_, p)| p.exp()).sum();
                let threshold = rng.random::<f64>() * total;
                let mut cumulative = 0.0;
                let mut selected = probabilities.len() - 1;
                for (j, (_, p)) in probabilities.iter().enumerate() {
                    cumulative += p.exp();
                    if threshold < cumulative {
                        selected = j;
                        break;
                    }
                }
                let (choice, probability) = probabilities[selected];
                choices[index] = choice;
                log_proposal[index] = probability - total.ln();
                Ok((choice, log_proposal[index]))
            },
            check_within_bank,
        )?;
        bank.choices = choices;
        bank.log_proposal = log_proposal;
        Ok((bank, evolution))
    }
}

// log(1-exp(-exp(log_hazard))), stable even for subnormal hazards.
fn log_event<S: JetField>(log_hazard: &S) -> S {
    let hazard = exp(log_hazard);
    if log_hazard.value() < -18.0 {
        let polynomial = add_real(&hazard.scale(1.0 / 120.0), -1.0 / 24.0);
        let polynomial = add_real(&hazard.mul(&polynomial), 1.0 / 6.0);
        let polynomial = add_real(&hazard.mul(&polynomial), -0.5);
        log_hazard.add(&ln(&add_real(&hazard.mul(&polynomial), 1.0)))
    } else {
        ln(&emission::expm1(&hazard.neg()).neg())
    }
}

impl JointReferenceBank<'_> {
    pub(super) fn belongs_to(&self, model: &JointLikelihood) -> bool {
        std::ptr::eq(self.model, model)
    }

    pub fn evolve<S: JetField>(
        &self,
        theta: &[S],
        accuracy: &ReferenceAccuracy,
    ) -> Result<JointReferenceEvolution<S>, EventHistoryError> {
        self.run(
            theta,
            accuracy,
            |i, _| Ok((self.choices[i], self.log_proposal[i])),
            true,
        )
    }

    pub(super) fn evolve_for_resolution<S: JetField>(
        &self,
        theta: &[S],
        accuracy: &ReferenceAccuracy,
    ) -> Result<JointReferenceEvolution<S>, EventHistoryError> {
        self.run(
            theta,
            accuracy,
            |i, _| Ok((self.choices[i], self.log_proposal[i])),
            false,
        )
    }

    fn moments<S: JetField>(
        &self,
        theta: &[S],
        population: &Population<S>,
        diagnostics: &mut ReferenceDiagnostics,
    ) -> Result<(Vec<S>, Vec<S>), EventHistoryError> {
        if population
            .states
            .iter()
            .flatten()
            .chain(&population.log_weight)
            .any(|x| !x.value().is_finite())
        {
            return Err(numerical(
                "joint reference state or path weight is non-finite",
            ));
        }
        let count = self.particles;
        let correction = count as f64 / (count - 1) as f64;
        let mut moments = Vec::new();
        let mut masses = Vec::new();
        for d in 0..self.model.spec.marks.len() {
            let group = self.mark_groups[d];
            let active: Vec<usize> = (group * count..(group + 1) * count).collect();
            let weights: Vec<S> = active
                .iter()
                .map(|&p| population.log_weight[p].clone())
                .collect();
            let mass = log_sum_exp(&weights);
            let activities: Vec<S> = active
                .iter()
                .map(|&p| self.model.activity(theta, d, &population.states[p]))
                .collect();
            let numerator: Vec<S> = weights
                .iter()
                .zip(&activities)
                .map(|(w, a)| w.add(a))
                .collect();
            let moment = log_sum_exp(&numerator).sub(&mass);
            let squared_weight: f64 = weights
                .iter()
                .map(|w| (2.0 * (w.value() - mass.value())).exp())
                .sum();
            diagnostics.minimum_risk_effective_samples = diagnostics
                .minimum_risk_effective_samples
                .min(1.0 / squared_weight);
            let moment_variance: f64 = weights
                .iter()
                .zip(&activities)
                .map(|(w, a)| {
                    // Both contributions are normalized weights in [0,1].
                    // Do not form a huge activity ratio then multiply its
                    // squared overflow by an underflowed importance weight.
                    let deviation = (w.value() - mass.value() + a.value() - moment.value()).exp()
                        - (w.value() - mass.value()).exp();
                    deviation * deviation
                })
                .sum();
            diagnostics.maximum_log_moment_standard_error = diagnostics
                .maximum_log_moment_standard_error
                .max((correction * moment_variance).sqrt());
            moments.push(moment);
            masses.push(population.log_mass[group].clone());
        }
        Ok((moments, masses))
    }

    fn run<S: JetField, F>(
        &self,
        theta: &[S],
        accuracy: &ReferenceAccuracy,
        mut select: F,
        check_within_bank: bool,
    ) -> Result<JointReferenceEvolution<S>, EventHistoryError>
    where
        F: FnMut(usize, &[(Option<usize>, f64)]) -> Result<(Option<usize>, f64), EventHistoryError>,
    {
        self.model.validate_parameters(theta)?;
        let count = self.genes.len();
        if [
            accuracy.log_moment_standard_error,
            accuracy.maximum_step_hazard,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
            || !accuracy.minimum_risk_effective_samples.is_finite()
            || accuracy.minimum_risk_effective_samples < 2.0
            || accuracy.minimum_risk_effective_samples > self.particles as f64
        {
            return Err(invalid(
                "joint reference needs positive finite error limits and an effective sample count in [2, particles]",
            ));
        }
        let k = self.model.spec.signatures;
        let marks = self.model.spec.marks.len();
        let nodes = self.profile.times.len();
        if k == 0 {
            return self.rank_zero_evolution(theta);
        }
        let zero = theta[0].constant_like(0.0);
        let entry = self.model.entry_features(&self.profile.history(marks));
        let genes: Vec<Vec<S>> = self
            .genes
            .iter()
            .map(|g| g.iter().map(|&v| zero.constant_like(v)).collect())
            .collect();
        let noise = |p: usize, block: usize, axis: usize| {
            self.normals[(p * (2 * nodes - 1) + block) * k + axis]
        };
        let mut population = Population {
            states: vec![vec![zero.clone(); k]; count],
            log_weight: vec![zero.clone(); count],
            risk: vec![vec![true; marks]; count],
            log_mass: vec![zero.clone(); self.groups.len()],
        };
        for p in 0..count {
            for axis in 0..k {
                population.states[p][axis] = add_real(
                    &self.model.mean(
                        theta,
                        self.model.layout.entry.start,
                        &entry,
                        &genes[p],
                        axis,
                    ),
                    noise(p, 0, axis),
                );
            }
        }
        let rates: Vec<S> = theta[self.model.layout.rates.clone()]
            .iter()
            .map(emission::softplus)
            .collect();
        let mut diagnostics = ReferenceDiagnostics {
            maximum_log_moment_standard_error: 0.0,
            minimum_risk_effective_samples: self.particles as f64,
            maximum_step_hazard: 0.0,
        };
        let (mut log_moments, mut log_risk_mass) =
            self.moments(theta, &population, &mut diagnostics)?;
        let mut times = vec![self.profile.times[0]];
        for n in 1..nodes {
            let dt = self.profile.times[n] - self.profile.times[n - 1];
            let columns = self.profile.drive_design.row(n - 1).to_vec();
            let decay: Vec<S> = rates.iter().map(|r| r.scale(-0.5 * dt)).collect();
            let phi: Vec<S> = decay.iter().map(exp).collect();
            let weight: Vec<S> = decay.iter().map(|r| emission::expm1(r).neg()).collect();
            let variance: Vec<S> = decay
                .iter()
                .map(|r| emission::expm1(&r.scale(2.0)).neg())
                .collect();
            if variance
                .iter()
                .any(|v| v.value() <= 0.0 || !v.value().is_finite())
            {
                return Err(numerical(
                    "joint reference OU half-step variance is unresolved",
                ));
            }
            let spread: Vec<S> = variance.iter().map(sqrt).collect();
            let propagate = |population: &mut Population<S>, block: usize| {
                for p in 0..count {
                    for axis in 0..k {
                        let drive = self.model.mean(
                            theta,
                            self.model.layout.drive.start,
                            &columns,
                            &genes[p],
                            axis,
                        );
                        population.states[p][axis] = phi[axis]
                            .mul(&population.states[p][axis])
                            .add(&weight[axis].mul(&drive))
                            .add(&spread[axis].scale(noise(p, block, axis)));
                    }
                }
            };
            propagate(&mut population, 2 * n - 1);
            let (midpoint, mid_mass) = self.moments(theta, &population, &mut diagnostics)?;
            log_moments.extend(midpoint.iter().cloned());
            let middle_mass_start = log_risk_mass.len();
            log_risk_mass.extend(mid_mass);
            times.push(self.profile.times[n - 1] + 0.5 * dt);
            let mut killing = vec![zero.clone(); count];
            for p in 0..count {
                let active: Vec<usize> = (0..marks).filter(|&d| population.risk[p][d]).collect();
                let focal = self.groups[p / self.particles];
                let mut log_rates = Vec::with_capacity(active.len());
                for &d in &active {
                    let mut baseline = zero.clone();
                    for b in 0..self.model.spec.baseline_columns {
                        let feature = 0.5 * self.profile.baseline_design[[n - 1, b]]
                            + 0.5 * self.profile.baseline_design[[n, b]];
                        baseline = baseline.add(
                            &theta[self.model.layout.baseline.start
                                + d * self.model.spec.baseline_columns
                                + b]
                                .scale(feature),
                        );
                    }
                    log_rates.push(
                        baseline
                            .add(&self.model.activity(theta, d, &population.states[p]))
                            .sub(&midpoint[d]),
                    );
                }
                let total_rate = log_sum_exp(&log_rates);
                let log_hazard = add_real(&total_rate, dt.ln());
                if !log_hazard.value().is_finite() {
                    return Err(numerical("joint reference log rate is non-finite"));
                }
                if log_hazard.value() > accuracy.maximum_step_hazard.ln() {
                    return Err(EventHistoryError::ReferenceStep {
                        log_hazard: log_hazard.value(),
                        maximum: accuracy.maximum_step_hazard,
                    });
                }
                diagnostics.maximum_step_hazard = diagnostics
                    .maximum_step_hazard
                    .max(log_hazard.value().exp());
                let mut retained = Vec::new();
                for (&d, rate) in active.iter().zip(&log_rates) {
                    if self.model.spec.marks[d] == MarkKind::Terminal || focal == Some(d) {
                        killing[p] = killing[p].add(&exp(&add_real(rate, dt.ln())));
                    } else {
                        retained.push((d, rate.clone()));
                    }
                }
                let mut probabilities = vec![(None, zero.clone())];
                if !retained.is_empty() {
                    let total =
                        log_sum_exp(&retained.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>());
                    let exposure = add_real(&total, dt.ln());
                    probabilities[0].1 = exp(&exposure).neg();
                    let event = log_event(&exposure);
                    for (d, rate) in retained {
                        probabilities.push((Some(d), event.add(&rate).sub(&total)));
                    }
                }
                let values: Vec<(Option<usize>, f64)> =
                    probabilities.iter().map(|(d, p)| (*d, p.value())).collect();
                let (choice, proposal) = select((n - 1) * count + p, &values)?;
                let target = &probabilities
                    .iter()
                    .find(|(d, _)| *d == choice)
                    .ok_or_else(|| {
                        invalid("joint reference event proposal violates its risk history")
                    })?
                    .1;
                population.log_weight[p] =
                    population.log_weight[p].add(&add_real(target, -proposal));
                if let Some(d) = choice {
                    if self.model.spec.marks[d] == MarkKind::Once {
                        population.risk[p][d] = false;
                    }
                    for axis in 0..k {
                        population.states[p][axis] =
                            population.states[p][axis].add(&self.model.jump(theta, Some(d), axis));
                    }
                }
            }
            // Integrate the killing rather than randomly losing survivors.
            // Normalize the non-killing transition first (its exact mass is
            // one), then retain its survival fraction and conditional law.
            for group in 0..self.groups.len() {
                let start = group * self.particles;
                let end = start + self.particles;
                // Remove a common log-weight offset before subtracting the
                // normalizers. Otherwise a finite survival loss can disappear
                // beside a very negative event importance log weight.
                let shift = population.log_weight[start..end]
                    .iter()
                    .map(JetField::value)
                    .fold(f64::NEG_INFINITY, f64::max);
                for p in start..end {
                    population.log_weight[p] = add_real(&population.log_weight[p], -shift);
                }
                let before = log_sum_exp(&population.log_weight[start..end]);
                for p in start..end {
                    population.log_weight[p] = population.log_weight[p].sub(&killing[p]);
                }
                let after = log_sum_exp(&population.log_weight[start..end]);
                population.log_mass[group] = population.log_mass[group].add(&after.sub(&before));
                for p in start..end {
                    population.log_weight[p] = population.log_weight[p].sub(&after);
                }
            }
            propagate(&mut population, 2 * n);
            let (moment, mass) = self.moments(theta, &population, &mut diagnostics)?;
            // The full killed step supplies the endpoint mass. Its log-linear
            // interpolation supplies the midpoint mass at the midpoint time;
            // labeling the interval-start mass as a midpoint creates a lag
            // even for a constant, non-latent hazard.
            for d in 0..marks {
                log_risk_mass[middle_mass_start + d] = log_risk_mass[middle_mass_start + d]
                    .scale(0.5)
                    .add(&mass[d].scale(0.5));
            }
            log_moments.extend(moment);
            log_risk_mass.extend(mass);
            times.push(self.profile.times[n]);
        }
        if check_within_bank
            && (diagnostics.maximum_log_moment_standard_error > accuracy.log_moment_standard_error
                || diagnostics.minimum_risk_effective_samples
                    + 16.0 * f64::EPSILON * (count as f64)
                    < accuracy.minimum_risk_effective_samples)
        {
            return Err(numerical(format!(
                "joint reference sampling unresolved: maximum log-moment SE {}, minimum risk ESS {}",
                diagnostics.maximum_log_moment_standard_error,
                diagnostics.minimum_risk_effective_samples
            )));
        }
        Ok(JointReferenceEvolution {
            theta: theta.to_vec(),
            times,
            log_moments,
            log_risk_mass,
            marks,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;

    #[test]
    fn rare_event_probabilities_keep_their_log_sensitivities() {
        let value = log_event(&Mixed::seed(-800.0, 1.0, 1.0));
        assert_eq!(value.base, -800.0);
        assert_eq!(value.u, 1.0);
        assert_eq!(value.uv, 0.0);
        let value = log_event(&Mixed::seed(0.03_f64.ln(), 1.0, 1.0));
        assert!((value.base - (-(-0.03_f64).exp_m1()).ln()).abs() < 1e-14);
    }

    #[test]
    fn reference_moment_errors_do_not_multiply_underflow_by_overflow() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let profile = JointReferenceProfile {
            times: vec![0.0, 1.0],
            baseline_design: Array2::ones((2, 1)),
            drive_design: Array2::ones((1, 1)),
            entry_design: vec![],
            genetics: vec![],
        };
        let bank = JointReferenceBank {
            model: &model,
            profile,
            genes: vec![vec![]; 2],
            normals: vec![],
            choices: vec![],
            log_proposal: vec![],
            groups: vec![None],
            mark_groups: vec![0],
            particles: 2,
        };
        let theta = vec![0.0; model.layout.width];
        let mut population = Population {
            states: vec![vec![1e300], vec![0.0]],
            log_weight: vec![-800.0, 0.0],
            risk: vec![vec![true]; 2],
            log_mass: vec![0.0],
        };
        let mut diagnostic = ReferenceDiagnostics {
            maximum_log_moment_standard_error: 0.0,
            minimum_risk_effective_samples: 2.0,
            maximum_step_hazard: 0.0,
        };
        let (moments, _) = bank.moments(&theta, &population, &mut diagnostic).unwrap();
        assert!(moments[0].is_finite());
        assert!(diagnostic.maximum_log_moment_standard_error > 1e-50);
        assert!(diagnostic.maximum_log_moment_standard_error < 1e-45);
        population.states[0][0] = f64::INFINITY;
        assert!(bank.moments(&theta, &population, &mut diagnostic).is_err());
    }
}
