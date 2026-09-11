//! Analytic sensitivities of a retained reference population. Derivatives
//! propagate through state transitions, event weights, conditional moments,
//! and survival normalization; event proposals remain integration data.
use super::*;

pub struct JointReferenceSensitivity {
    reference: JointReferenceEvolution<f64>,
    moment_jacobian: Array2<f64>,
    mass_jacobian: Array2<f64>,
    workspace_limit: usize,
}

impl JointReferenceSensitivity {
    pub(in crate::joint) fn pooled(
        reference: JointReferenceEvolution<f64>,
        moment_jacobian: Array2<f64>,
        mass_jacobian: Array2<f64>,
        workspace_limit: usize,
    ) -> Self {
        Self {
            reference,
            moment_jacobian,
            mass_jacobian,
            workspace_limit,
        }
    }
    pub(in crate::joint) fn into_reference(self) -> JointReferenceEvolution<f64> {
        self.reference
    }
    pub fn reference(&self) -> &JointReferenceEvolution<f64> {
        &self.reference
    }
    pub fn log_moment_jacobian(&self) -> &Array2<f64> {
        &self.moment_jacobian
    }
    pub fn log_risk_mass_jacobian(&self) -> &Array2<f64> {
        &self.mass_jacobian
    }

    pub fn at(&self, times: &[f64]) -> Result<(Vec<f64>, Array2<f64>), EventHistoryError> {
        let requested = times
            .len()
            .checked_mul(self.reference.marks)
            .and_then(|n| n.checked_mul(self.moment_jacobian.ncols() + 1))
            .and_then(|n| n.checked_add(self.moment_jacobian.len()))
            .and_then(|n| n.checked_add(self.mass_jacobian.len()))
            .and_then(|n| n.checked_add(self.reference.log_moments.len()))
            .and_then(|n| n.checked_add(self.reference.log_risk_mass.len()))
            .and_then(|n| n.checked_add(self.reference.times.len()))
            .and_then(|n| n.checked_add(self.reference.theta.len()))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("reference Jacobian interpolation size overflow"))?;
        if requested > self.workspace_limit {
            return Err(numerical(
                "reference Jacobian interpolation exceeds its workspace budget",
            ));
        }
        let values = self.reference.at(times)?;
        Ok((values, self.interpolate(&self.moment_jacobian, times)))
    }

    fn interpolate(&self, source: &Array2<f64>, times: &[f64]) -> Array2<f64> {
        let marks = self.reference.marks;
        let mut out = Array2::zeros((times.len() * marks, source.ncols()));
        for (n, &time) in times.iter().enumerate() {
            let right = self.reference.times.partition_point(|&t| t < time);
            if right == 0 || self.reference.times[right] == time {
                for d in 0..marks {
                    out.row_mut(n * marks + d)
                        .assign(&source.row(right * marks + d));
                }
            } else {
                let fraction = (time - self.reference.times[right - 1])
                    / (self.reference.times[right] - self.reference.times[right - 1]);
                for d in 0..marks {
                    for q in 0..source.ncols() {
                        out[[n * marks + d, q]] = (1.0 - fraction)
                            * source[[(right - 1) * marks + d, q]]
                            + fraction * source[[right * marks + d, q]];
                    }
                }
            }
        }
        out
    }
}

struct PopulationSensitivity {
    value: Population<f64>,
    state: Array2<f64>,
    weight: Array2<f64>,
    mass: Array2<f64>,
}

fn add_regression(
    out: &mut ndarray::ArrayViewMut1<'_, f64>,
    start: usize,
    columns: &[f64],
    genes: &[f64],
    axis: usize,
    factor: f64,
) {
    for (j, &feature) in columns.iter().enumerate() {
        let base = start + (axis * columns.len() + j) * (genes.len() + 1);
        out[base] += factor * feature;
        for (g, &gene) in genes.iter().enumerate() {
            out[base + g + 1] += factor * feature * gene;
        }
    }
}

impl JointReferenceBank<'_> {
    fn activity_sensitivity(
        &self,
        theta: &[f64],
        mark: usize,
        particle: usize,
        population: &PopulationSensitivity,
        out: &mut [f64],
    ) -> f64 {
        out.fill(0.0);
        let k = self.model.spec.signatures;
        let start = self.model.layout.decoder.start + mark * k;
        let mut numerator = vec![0.0];
        let mut denominator = vec![0.0];
        for axis in 0..k {
            numerator.push(
                theta[start + axis]
                    + emission::log_softplus(&population.value.states[particle][axis]),
            );
            denominator.push(theta[start + axis]);
        }
        let log_num = log_sum_exp(&numerator);
        let log_den = log_sum_exp(&denominator);
        for axis in 0..k {
            let x = population.value.states[particle][axis];
            let state_score = (theta[start + axis] - emission::softplus(&(-x)) - log_num).exp();
            for q in 0..theta.len() {
                out[q] += state_score * population.state[[particle * k + axis, q]];
            }
            out[start + axis] +=
                (numerator[axis + 1] - log_num).exp() - (denominator[axis + 1] - log_den).exp();
        }
        log_num - log_den
    }

    fn moment_sensitivity(
        &self,
        theta: &[f64],
        population: &PopulationSensitivity,
        expected: &[f64],
    ) -> Result<Array2<f64>, EventHistoryError> {
        let width = theta.len();
        let mut out = Array2::zeros((self.model.spec.marks.len(), width));
        let mut derivative = vec![0.0; width];
        for d in 0..self.model.spec.marks.len() {
            let start = self.mark_groups[d] * self.particles;
            let end = start + self.particles;
            let weights = &population.value.log_weight[start..end];
            let mass = log_sum_exp(weights);
            let activities: Vec<_> = (start..end)
                .map(|p| self.model.activity(theta, d, &population.value.states[p]))
                .collect();
            let weighted: Vec<_> = weights
                .iter()
                .zip(&activities)
                .map(|(w, a)| w + a)
                .collect();
            let numerator = log_sum_exp(&weighted);
            if numerator - mass != expected[d] {
                return Err(numerical(
                    "analytic reference replay disagrees with its value evolution",
                ));
            }
            for p in start..end {
                self.activity_sensitivity(theta, d, p, population, &mut derivative);
                let active = (weighted[p - start] - numerator).exp();
                let risk = (weights[p - start] - mass).exp();
                for q in 0..width {
                    out[[d, q]] +=
                        active * derivative[q] + (active - risk) * population.weight[[p, q]];
                }
            }
        }
        Ok(out)
    }

    fn propagate_sensitivity(
        &self,
        theta: &[f64],
        population: &mut PopulationSensitivity,
        n: usize,
        block: usize,
    ) {
        let k = self.model.spec.signatures;
        let dt = self.profile.times[n] - self.profile.times[n - 1];
        let columns = self.profile.drive_design.row(n - 1).to_vec();
        for axis in 0..k {
            let raw = theta[self.model.layout.rates.start + axis];
            let decay = emission::softplus(&raw) * (-0.5 * dt);
            let phi = decay.exp();
            let weight = -decay.exp_m1();
            let spread = (-(decay * 2.0).exp_m1()).sqrt();
            let dphi = (-0.5 * dt) * phi * (-emission::softplus(&(-raw))).exp();
            let dspread = -phi * dphi / spread;
            for p in 0..self.genes.len() {
                let drive = self.model.mean(
                    theta,
                    self.model.layout.drive.start,
                    &columns,
                    &self.genes[p],
                    axis,
                );
                let old = population.value.states[p][axis];
                let noise =
                    self.normals[(p * (2 * self.profile.times.len() - 1) + block) * k + axis];
                let mut row = population.state.row_mut(p * k + axis);
                row *= phi;
                add_regression(
                    &mut row,
                    self.model.layout.drive.start,
                    &columns,
                    &self.genes[p],
                    axis,
                    weight,
                );
                row[self.model.layout.rates.start + axis] += dphi * (old - drive) + dspread * noise;
                population.value.states[p][axis] = phi * old + weight * drive + spread * noise;
            }
        }
    }

    /// Analytic Jacobians of one fixed population's moments and risk masses.
    /// Workspace is additional to retained bank storage. The value calculation
    /// supplies all resolution/step checks; every replayed moment must agree
    /// with it before its derivative can be returned. Sampling or refinement
    /// accuracy of the Jacobian itself is not established by these checks.
    pub fn sensitivity(
        &self,
        theta: &[f64],
        accuracy: &ReferenceAccuracy,
        memory_limit_bytes: usize,
    ) -> Result<JointReferenceSensitivity, EventHistoryError> {
        self.sensitivity_impl(theta, accuracy, memory_limit_bytes, true)
    }

    pub(in crate::joint) fn sensitivity_for_resolution(
        &self,
        theta: &[f64],
        accuracy: &ReferenceAccuracy,
        memory_limit_bytes: usize,
    ) -> Result<JointReferenceSensitivity, EventHistoryError> {
        self.sensitivity_impl(theta, accuracy, memory_limit_bytes, false)
    }

    fn sensitivity_impl(
        &self,
        theta: &[f64],
        accuracy: &ReferenceAccuracy,
        memory_limit_bytes: usize,
        check_within_bank: bool,
    ) -> Result<JointReferenceSensitivity, EventHistoryError> {
        self.model.validate_parameters(theta)?;
        let count = self.genes.len();
        let k = self.model.spec.signatures;
        let marks = self.model.spec.marks.len();
        let nodes = self.profile.times.len();
        let width = theta.len();
        let rows = (2 * nodes - 1)
            .checked_mul(marks)
            .ok_or_else(|| invalid("reference sensitivity row overflow"))?;
        // State, log-weight, killing, mass and temporary mark derivatives;
        // two returned Jacobians. A second allowance covers scalar replay
        // arrays and row headers; it is bounded before the primal is run.
        let entries = count
            .checked_mul(k + 3)
            .and_then(|a| a.checked_add(self.groups.len() + 4 * marks + 2 * rows + 4))
            .and_then(|a| a.checked_mul(width + 1))
            .and_then(|a| a.checked_mul(16))
            .and_then(|bytes| {
                count
                    .checked_mul(self.model.spec.genetic_mean.len())
                    .and_then(|n| n.checked_mul(8))
                    .and_then(|n| bytes.checked_add(n))
            })
            .ok_or_else(|| invalid("reference sensitivity workspace overflow"))?;
        if entries > memory_limit_bytes {
            return Err(numerical(format!(
                "reference sensitivities need {entries} workspace bytes, above limit {memory_limit_bytes}"
            )));
        }
        let reference = if check_within_bank {
            self.evolve(theta, accuracy)?
        } else {
            self.evolve_for_resolution(theta, accuracy)?
        };
        if k == 0 {
            return Ok(JointReferenceSensitivity {
                reference,
                moment_jacobian: Array2::zeros((rows, width)),
                mass_jacobian: self.rank_zero_mass_jacobian(theta)?,
                workspace_limit: memory_limit_bytes,
            });
        }
        let mut population = PopulationSensitivity {
            value: Population {
                states: vec![vec![0.0; k]; count],
                log_weight: vec![0.0; count],
                risk: vec![vec![true; marks]; count],
                log_mass: vec![0.0; self.groups.len()],
            },
            state: Array2::zeros((count * k, width)),
            weight: Array2::zeros((count, width)),
            mass: Array2::zeros((self.groups.len(), width)),
        };
        let entry = self.model.entry_features(&self.profile.history(marks));
        for p in 0..count {
            for axis in 0..k {
                population.value.states[p][axis] = self.model.mean(
                    theta,
                    self.model.layout.entry.start,
                    &entry,
                    &self.genes[p],
                    axis,
                ) + self.normals[p * (2 * nodes - 1) * k + axis];
                add_regression(
                    &mut population.state.row_mut(p * k + axis),
                    self.model.layout.entry.start,
                    &entry,
                    &self.genes[p],
                    axis,
                    1.0,
                );
            }
        }
        let mut moment_jacobian = Array2::zeros((rows, width));
        let mut mass_jacobian = Array2::zeros((rows, width));
        let initial =
            self.moment_sensitivity(theta, &population, &reference.log_moments[..marks])?;
        for d in 0..marks {
            moment_jacobian.row_mut(d).assign(&initial.row(d));
        }
        for n in 1..nodes {
            let dt = self.profile.times[n] - self.profile.times[n - 1];
            let midrow = (2 * n - 1) * marks;
            let endrow = 2 * n * marks;
            self.propagate_sensitivity(theta, &mut population, n, 2 * n - 1);
            let midpoint = &reference.log_moments[midrow..midrow + marks];
            let mid = self.moment_sensitivity(theta, &population, midpoint)?;
            for d in 0..marks {
                moment_jacobian.row_mut(midrow + d).assign(&mid.row(d));
                mass_jacobian
                    .row_mut(midrow + d)
                    .assign(&population.mass.row(self.mark_groups[d]));
            }
            let mut killing = vec![0.0; count];
            let mut killing_score = Array2::<f64>::zeros((count, width));
            for p in 0..count {
                let focal = self.groups[p / self.particles];
                let mut retained = Vec::new();
                let mut rates = Vec::new();
                let mut derivatives: Vec<Vec<f64>> = Vec::new();
                for d in 0..marks {
                    if !population.value.risk[p][d] {
                        continue;
                    }
                    let mut derivative = vec![0.0; width];
                    let activity =
                        self.activity_sensitivity(theta, d, p, &population, &mut derivative);
                    let mut baseline = 0.0;
                    for b in 0..self.model.spec.baseline_columns {
                        let feature = 0.5 * self.profile.baseline_design[[n - 1, b]]
                            + 0.5 * self.profile.baseline_design[[n, b]];
                        let index = self.model.layout.baseline.start
                            + d * self.model.spec.baseline_columns
                            + b;
                        baseline += theta[index] * feature;
                        derivative[index] += feature;
                    }
                    for q in 0..width {
                        derivative[q] -= mid[[d, q]];
                    }
                    let rate = baseline + activity - midpoint[d];
                    if self.model.spec.marks[d] == MarkKind::Terminal || focal == Some(d) {
                        let hazard = (rate + dt.ln()).exp();
                        killing[p] += hazard;
                        for q in 0..width {
                            killing_score[[p, q]] += hazard * derivative[q];
                        }
                    } else {
                        retained.push(d);
                        rates.push(rate);
                        derivatives.push(derivative);
                    }
                }
                let chosen = self.choices[(n - 1) * count + p];
                let mut target = 0.0;
                let mut target_score = vec![0.0; width];
                if !rates.is_empty() {
                    let total = log_sum_exp(&rates);
                    let exposure = total + dt.ln();
                    let hazard = exposure.exp();
                    let event_index = chosen.and_then(|d| retained.iter().position(|&v| v == d));
                    let factor = if chosen.is_none() {
                        -hazard
                    } else if exposure < -18.0 {
                        // H/expm1(H)-1, preserving the small departure from
                        // zero when the event probability approaches H.
                        -0.5 * hazard + hazard * hazard / 12.0
                    } else {
                        hazard / hazard.exp_m1() - 1.0
                    };
                    for (rate, derivative) in rates.iter().zip(&derivatives) {
                        let fraction = (rate - total).exp();
                        for q in 0..width {
                            target_score[q] += factor * fraction * derivative[q];
                        }
                    }
                    if let Some(index) = event_index {
                        target = log_event(&exposure) + rates[index] - total;
                        for q in 0..width {
                            target_score[q] += derivatives[index][q];
                        }
                    } else if chosen.is_none() {
                        target = -hazard;
                    } else {
                        return Err(invalid(
                            "reference sensitivity event proposal violates risk history",
                        ));
                    }
                } else if chosen.is_some() {
                    return Err(invalid(
                        "reference sensitivity event has no active non-killing rate",
                    ));
                }
                population.value.log_weight[p] += target - self.log_proposal[(n - 1) * count + p];
                for q in 0..width {
                    population.weight[[p, q]] += target_score[q];
                }
                if let Some(d) = chosen {
                    if self.model.spec.marks[d] == MarkKind::Once {
                        population.value.risk[p][d] = false;
                    }
                    for axis in 0..k {
                        population.value.states[p][axis] += self.model.jump(theta, Some(d), axis);
                        if let Some(range) = &self.model.layout.jumps[d] {
                            population.state[[p * k + axis, range.start + axis]] += 1.0;
                        }
                    }
                }
            }
            for group in 0..self.groups.len() {
                let start = group * self.particles;
                let end = start + self.particles;
                let shift = population.value.log_weight[start..end]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                for p in start..end {
                    population.value.log_weight[p] -= shift;
                }
                let before = log_sum_exp(&population.value.log_weight[start..end]);
                let mut before_score = vec![0.0; width];
                for p in start..end {
                    let fraction = (population.value.log_weight[p] - before).exp();
                    for q in 0..width {
                        before_score[q] += fraction * population.weight[[p, q]];
                    }
                    population.value.log_weight[p] -= killing[p];
                    for q in 0..width {
                        population.weight[[p, q]] -= killing_score[[p, q]];
                    }
                }
                let after = log_sum_exp(&population.value.log_weight[start..end]);
                let mut after_score = vec![0.0; width];
                for p in start..end {
                    let fraction = (population.value.log_weight[p] - after).exp();
                    for q in 0..width {
                        after_score[q] += fraction * population.weight[[p, q]];
                    }
                }
                population.value.log_mass[group] += after - before;
                for q in 0..width {
                    population.mass[[group, q]] += after_score[q] - before_score[q];
                }
                for p in start..end {
                    population.value.log_weight[p] -= after;
                    for q in 0..width {
                        population.weight[[p, q]] -= after_score[q];
                    }
                }
            }
            self.propagate_sensitivity(theta, &mut population, n, 2 * n);
            let endpoint = self.moment_sensitivity(
                theta,
                &population,
                &reference.log_moments[endrow..endrow + marks],
            )?;
            for d in 0..marks {
                moment_jacobian.row_mut(endrow + d).assign(&endpoint.row(d));
                let group = self.mark_groups[d];
                if population.value.log_mass[group] != reference.log_risk_mass[endrow + d] {
                    return Err(numerical(
                        "analytic reference survival replay disagrees with its value evolution",
                    ));
                }
                for q in 0..width {
                    mass_jacobian[[endrow + d, q]] = population.mass[[group, q]];
                    mass_jacobian[[midrow + d, q]] =
                        0.5 * mass_jacobian[[midrow + d, q]] + 0.5 * population.mass[[group, q]];
                }
            }
        }
        if moment_jacobian
            .iter()
            .chain(mass_jacobian.iter())
            .any(|v| !v.is_finite())
        {
            return Err(numerical("non-finite reference sensitivity"));
        }
        Ok(JointReferenceSensitivity {
            reference,
            moment_jacobian,
            mass_jacobian,
            workspace_limit: memory_limit_bytes,
        })
    }
}
