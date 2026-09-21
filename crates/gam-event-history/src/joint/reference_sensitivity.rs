//! Reverse-mode sensitivities of a retained reference population: the gradient
//! of a linear functional of every stored log moment and log risk mass in one
//! sweep, with no coefficient-sized tangent per particle. The sweep
//! differentiates the OU transitions, event weights and jumps, the midpoint
//! normaliser inside every rate, and survival normalization; event proposals
//! remain integration data. Interval-start checkpoints bound the retained work,
//! and each interval is recomputed through the same step kernel. It is written
//! over any `JetField`, so seeding the coefficients gives the derivative of
//! that gradient along the seed.
//!
//! Risk sets are swept in parallel. Each set accumulates its own share of the
//! adjoints every set adds to, and the shares are added in set order, so the
//! gradient is the same at any thread count.
use super::*;

fn logistic<S: JetField>(x: &S) -> S {
    exp(&emission::softplus(&x.neg()).neg())
}

fn softmax<S: JetField>(values: &[S]) -> Vec<S> {
    let total = log_sum_exp(values);
    values.iter().map(|v| exp(&v.sub(&total))).collect()
}

fn accumulate<S: JetField>(slot: &mut S, value: &S) {
    *slot = slot.add(value);
}

/// Adds one risk set's share into the totals.
fn fold<S: JetField>(total: &mut [S], share: &[S]) {
    for (slot, value) in total.iter_mut().zip(share) {
        accumulate(slot, value);
    }
}

/// `values` split into `sets` equal consecutive slices, including empty ones
/// when a population has no signatures.
fn set_slices<S>(values: &mut [S], sets: usize) -> Vec<&mut [S]> {
    let width = values.len() / sets;
    let mut rest = values;
    let mut slices = Vec::with_capacity(sets);
    for _ in 0..sets {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(width);
        slices.push(head);
        rest = tail;
    }
    slices
}

/// One risk set's share of the adjoints every set adds to: coefficients, OU
/// decay and spread per signature, and the midpoint normaliser per mark.
struct SetShare<S> {
    theta: Vec<S>,
    phi: Vec<S>,
    spread: Vec<S>,
    normaliser: Vec<S>,
}

impl<S: JetField> SetShare<S> {
    fn new(zero: &S, coefficients: usize, signatures: usize, marks: usize) -> Self {
        Self {
            theta: vec![zero.clone(); coefficients],
            phi: vec![zero.clone(); signatures],
            spread: vec![zero.clone(); signatures],
            normaliser: vec![zero.clone(); marks],
        }
    }
}

/// Adjoint of a mean that is linear in the genes over `columns`, into its
/// coefficients.
fn add_mean_adjoint<S: JetField>(
    theta_bar: &mut [S],
    start: usize,
    columns: &[f64],
    genes: &[S],
    axis: usize,
    adjoint: &S,
) {
    let width = genes.len() + 1;
    for (j, &feature) in columns.iter().enumerate() {
        let base = start + (axis * columns.len() + j) * width;
        let scaled = adjoint.scale(feature);
        accumulate(&mut theta_bar[base], &scaled);
        for (g, gene) in genes.iter().enumerate() {
            accumulate(&mut theta_bar[base + g + 1], &scaled.mul(gene));
        }
    }
}

/// Adds `weight` times the gradient of `mark`'s log activity `activity` at
/// `state`: into the state adjoint, and through the normalized decoder weights
/// into the coefficient adjoint.
fn add_activity_adjoint<S: JetField>(
    model: &JointLikelihood,
    decoder: &super::super::decoder::PreparedDecoder<S>,
    mark: usize,
    state: &[S],
    activity: &S,
    weight: &S,
    state_bar: &mut [S],
    theta_bar: &mut [S],
) {
    let k = model.spec.signatures;
    let log_weights = decoder.weights(mark);
    let start = model.layout.decoder.start + mark * k;
    for axis in 0..k {
        let x = &state[axis];
        // d a / d x_k = pi_k sigma(x_k) / R.
        let slope = exp(&log_weights[axis + 1].sub(activity)).mul(&logistic(x));
        accumulate(&mut state_bar[axis], &weight.mul(&slope));
        // d a / d log pi_j sums to one over j, and d log pi_j / d logit_i is
        // [j == i] - pi_i, so a logit's derivative is its share minus pi_i.
        let share = exp(&log_weights[axis + 1]
            .add(&emission::log_softplus(x))
            .sub(activity));
        accumulate(
            &mut theta_bar[start + axis],
            &weight.mul(&share.sub(&exp(&log_weights[axis + 1]))),
        );
    }
}

/// Adds `rate_bar` times the gradient of one mark's log rate `b_d + a_d(x) -
/// m_d` at `state`: into the set's baseline, normaliser and decoder shares and
/// the particle's state adjoint.
fn add_rate_adjoint<S: JetField>(
    model: &JointLikelihood,
    decoder: &super::super::decoder::PreparedDecoder<S>,
    mark: usize,
    baseline: &DyadicRow<'_>,
    state: &[S],
    activity: &S,
    rate_bar: &S,
    state_bar: &mut [S],
    share: &mut SetShare<S>,
) {
    accumulate(&mut share.normaliser[mark], &rate_bar.neg());
    for b in 0..baseline.columns() {
        accumulate(
            &mut share.theta[model.layout.baseline.start + mark * model.spec.baseline_columns + b],
            &baseline.feature(rate_bar, b),
        );
    }
    add_activity_adjoint(
        model,
        decoder,
        mark,
        state,
        activity,
        rate_bar,
        state_bar,
        &mut share.theta,
    );
}

impl JointReferenceBank {
    /// Adds `adjoint[d]` times the gradient of the log moment of every mark
    /// taken over risk set `group` into that set's state and log-weight
    /// adjoints and the coefficients.
    fn add_moment_adjoint<S: JetField>(
        &self,
        model: &JointLikelihood,
        decoder: &super::super::decoder::PreparedDecoder<S>,
        population: &Population<S>,
        adjoint: &[S],
        group: usize,
        state_bar: &mut [S],
        weight_bar: &mut [S],
        theta_bar: &mut [S],
    ) {
        let k = model.spec.signatures;
        let marks = model.spec.marks.len();
        let start = group * self.particles;
        let weights = &population.log_weight[start..start + self.particles];
        let states = &population.states[start..start + self.particles];
        let risk = softmax(weights);
        for d in (0..marks).filter(|&d| self.mark_groups[d] == group) {
            let activity: Vec<S> = states.iter().map(|state| decoder.activity(d, state)).collect();
            let numerator: Vec<S> = weights.iter().zip(&activity).map(|(w, a)| w.add(a)).collect();
            let active = softmax(&numerator);
            for i in 0..self.particles {
                let share = adjoint[d].mul(&active[i]);
                accumulate(&mut weight_bar[i], &share.sub(&adjoint[d].mul(&risk[i])));
                add_activity_adjoint(
                    model,
                    decoder,
                    d,
                    &states[i],
                    &activity[i],
                    &share,
                    &mut state_bar[i * k..(i + 1) * k],
                    theta_bar,
                );
            }
        }
    }

    /// Adjoint of `half_step` over risk set `group` from the states before it:
    /// maps that set's state adjoint back and accumulates `phi`, spread and
    /// drive adjoints into its share.
    fn add_half_step_adjoint<S: JetField>(
        &self,
        model: &JointLikelihood,
        drive: &MeanTable<S>,
        columns: &[f64],
        genes: &[Vec<S>],
        step: &HalfStep<S>,
        before: &[Vec<S>],
        normals: &[f64],
        group: usize,
        state_bar: &mut [S],
        share: &mut SetShare<S>,
    ) {
        let k = model.spec.signatures;
        for i in 0..self.particles {
            let p = group * self.particles + i;
            for axis in 0..k {
                let bar = state_bar[i * k + axis].clone();
                let mean = drive.mean(axis, &genes[p]);
                accumulate(&mut share.phi[axis], &bar.mul(&before[p][axis].sub(&mean)));
                accumulate(&mut share.spread[axis], &bar.scale(normals[p * k + axis]));
                add_mean_adjoint(
                    &mut share.theta,
                    model.layout.drive.start,
                    columns,
                    &genes[p],
                    axis,
                    &bar.mul(&step.weight[axis]),
                );
                state_bar[i * k + axis] = bar.mul(&step.phi[axis]);
            }
        }
    }

    /// Workspace of one reverse sweep over `coefficients` scalars of type `S`,
    /// additional to retained bank storage: checkpoints per node, four interval
    /// work copies of the population with their activity tables, the
    /// coefficient adjoint, and one share of the coefficient, OU and normaliser
    /// adjoints per risk set. It is computed from counts alone.
    pub(in crate::joint) fn pullback_bytes<S>(
        &self,
        model: &JointLikelihood,
        coefficients: usize,
    ) -> Result<usize, EventHistoryError> {
        let k = model.spec.signatures;
        let marks = model.spec.marks.len();
        let nodes = self.grid.nodes();
        let groups = self.groups.len();
        let scalar = std::mem::size_of::<S>();
        let count = self.particles.checked_mul(groups);
        let particle_bytes = (k + 1 + 2 * marks)
            .checked_mul(scalar)
            .and_then(|b| b.checked_add(marks))
            .and_then(|b| b.checked_add(std::mem::size_of::<Vec<S>>()))
            .and_then(|b| b.checked_add(std::mem::size_of::<Vec<bool>>()));
        count
            .and_then(|count| {
                particle_bytes
                    .and_then(|b| b.checked_mul(count))
                    .and_then(|b| b.checked_mul(nodes + 4))
                    .and_then(|b| {
                        count
                            .checked_mul(4)
                            .and_then(|t| t.checked_add(coefficients + 2 * marks + 4 * k))
                            .and_then(|t| t.checked_mul(scalar))
                            .and_then(|t| b.checked_add(t))
                    })
            })
            .and_then(|b| {
                coefficients
                    .checked_add(2 * k + marks)
                    .and_then(|t| t.checked_mul(groups))
                    .and_then(|t| t.checked_mul(scalar))
                    .and_then(|t| b.checked_add(t))
            })
            .ok_or_else(|| invalid("reference pullback workspace overflow"))
    }

    /// Gradient of `moment_adjoint' m + mass_adjoint' M` over every stored row
    /// of this population's evolution at `theta`. The primal is replayed once
    /// with interval-start checkpoints; each interval is recomputed during the
    /// reverse sweep. The workspace, additional to retained bank storage, is
    /// admitted against the machine's materialization budget from counts
    /// alone, before the replay allocates anything.
    pub(in crate::joint) fn pullback<S: JetField + Send + Sync>(
        &self,
        model: &JointLikelihood,
        theta: &[S],
        moment_adjoint: &[S],
        mass_adjoint: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        model.validate_parameters(theta)?;
        let k = model.spec.signatures;
        let marks = model.spec.marks.len();
        let nodes = self.grid.nodes();
        let groups = self.groups.len();
        let count = self
            .particles
            .checked_mul(groups)
            .ok_or_else(|| invalid("reference pullback population overflow"))?;
        let rows = (2 * nodes - 1)
            .checked_mul(marks)
            .ok_or_else(|| invalid("reference pullback row overflow"))?;
        if moment_adjoint.len() != rows || mass_adjoint.len() != rows {
            return Err(invalid(
                "reference pullback adjoints need one entry per stored row and mark",
            ));
        }
        let bytes = self.pullback_bytes::<S>(model, theta.len())?;
        let budget = materialization_budget();
        if bytes > budget {
            return Err(numerical(format!(
                "reference pullback needs {bytes} workspace bytes, above this machine's {budget}-byte materialization budget"
            )));
        }
        let mut checkpoints = Vec::with_capacity(nodes - 1);
        self.run(model, theta, Proposal::Replay, Some(&mut checkpoints))?;
        let gene_count = model.spec.genetic_mean.len();
        let zero = theta[0].constant_like(0.0);
        let decoder = super::super::decoder::PreparedDecoder::new(model, theta);
        let decoder_weights = DecoderWeights::new(&decoder, marks, k);
        let genes: Vec<Vec<S>> = self
            .genes
            .iter()
            .map(|g| g.iter().map(|&v| zero.constant_like(v)).collect())
            .collect();
        let rates: Vec<S> = theta[model.layout.rates.clone()]
            .iter()
            .map(emission::softplus)
            .collect();
        let masks = self.killing_masks(model);
        let killing = KillingSets {
            masks: &masks,
            per_set: self.particles,
        };
        let mut scratch = ReferenceDiagnostics {
            minimum_risk_effective_samples: self.particles as f64,
            omitted_event_mass: 0.0,
        };
        let mut theta_bar = vec![zero.clone(); theta.len()];
        let mut state_bar = vec![zero.clone(); count * k];
        let mut weight_bar = vec![zero.clone(); count];
        let mut mass_bar = vec![zero.clone(); groups];
        for n in (1..nodes).rev() {
            let start = &checkpoints[n - 1];
            let dt = self.grid.dt(&zero, n);
            let columns = self.grid.drive_row(n).to_vec();
            let step = HalfStep::new(&rates, &dt)?;
            let drive = MeanTable::new(theta, model.layout.drive.start, &columns, k, gene_count);
            let baseline = self.grid.midpoint_row(n);
            let steps = (n - 1) * count..n * count;
            // Recompute the interval through the kernel.
            let mut middle = start.clone();
            half_step(&step, &drive, &genes, self.block(2 * n - 1, k), &mut middle.states);
            let midpoint = self.moments(marks, &middle, &decoder, &mut scratch)?.0;
            let step_rates =
                StepRates::new(model, theta, &decoder_weights, &baseline, &midpoint, &killing)?;
            let mut events = middle.clone();
            let outcome = event_step(
                model,
                theta,
                &decoder,
                &decoder_weights,
                &step_rates,
                &dt,
                &killing,
                &mut events,
                EventChoice::Replay {
                    choices: &self.choices[steps.clone()],
                    log_proposal: &self.log_proposal[steps],
                },
            )?;
            let mut shifted = events.log_weight.clone();
            let mut killed = events.log_weight.clone();
            let mut end = events.clone();
            shifted
                .par_chunks_mut(self.particles)
                .zip(killed.par_chunks_mut(self.particles))
                .zip(end.log_weight.par_chunks_mut(self.particles))
                .enumerate()
                .for_each(|(group, ((shifted, killed), weights))| {
                    let first = group * self.particles;
                    let before = &events.log_weight[first..first + self.particles];
                    let shift = before
                        .iter()
                        .map(JetField::value)
                        .fold(f64::NEG_INFINITY, f64::max);
                    for (i, weight) in before.iter().enumerate() {
                        let p = first + i;
                        let hazard = outcome.log_hazard[outcome.offsets[p]..outcome.offsets[p + 1]]
                            .iter()
                            .fold(zero.clone(), |acc, h| acc.add(&exp(h)));
                        shifted[i] = add_real(weight, -shift);
                        killed[i] = shifted[i].sub(&hazard);
                    }
                    let after = log_sum_exp(killed);
                    for (slot, value) in weights.iter_mut().zip(killed.iter()) {
                        *slot = value.sub(&after);
                    }
                });
            half_step(&step, &drive, &genes, self.block(2 * n, k), &mut end.states);
            let end_row = 2 * n * marks;
            let mid_row = (2 * n - 1) * marks;
            for d in 0..marks {
                accumulate(
                    &mut mass_bar[self.mark_groups[d]],
                    &mass_adjoint[end_row + d].add(&mass_adjoint[mid_row + d].scale(0.5)),
                );
            }
            // Per risk set: endpoint rows, the second half step, survival
            // normalization and the event step.
            let shares: Vec<Result<SetShare<S>, EventHistoryError>> = set_slices(&mut state_bar, groups)
                .into_par_iter()
                .zip(weight_bar.par_chunks_mut(self.particles))
                .enumerate()
                .map(|(group, (state_bar, weight_bar))| -> Result<SetShare<S>, EventHistoryError> {
                    let mut share = SetShare::new(&zero, theta.len(), k, marks);
                    self.add_moment_adjoint(
                        model,
                        &decoder,
                        &end,
                        &moment_adjoint[end_row..end_row + marks],
                        group,
                        state_bar,
                        weight_bar,
                        &mut share.theta,
                    );
                    self.add_half_step_adjoint(
                        model,
                        &drive,
                        &columns,
                        &genes,
                        &step,
                        &events.states,
                        self.block(2 * n, k),
                        group,
                        state_bar,
                        &mut share,
                    );
                    // Survival normalization: w_new = killed - after, and the
                    // mass increment is after - before, with before over the
                    // shifted weights.
                    let first = group * self.particles;
                    let last = first + self.particles;
                    let after_bar = weight_bar
                        .iter()
                        .fold(mass_bar[group].clone(), |acc, w| acc.sub(w));
                    let killed_share = softmax(&killed[first..last]);
                    let shifted_share = softmax(&shifted[first..last]);
                    let mut kill_bar = Vec::with_capacity(self.particles);
                    for i in 0..self.particles {
                        let killed_bar = weight_bar[i].add(&after_bar.mul(&killed_share[i]));
                        kill_bar.push(killed_bar.neg());
                        weight_bar[i] = killed_bar.sub(&mass_bar[group].mul(&shifted_share[i]));
                    }
                    // Event step: jumps, target weights and killing, all at the
                    // pre-jump state and the midpoint normaliser. Killing marks
                    // and a selected mark are differentiated individually. With
                    // `e_p = target_bar scale / total`, every retained mark's share
                    // `e_p w_d R_d(x_p)` is gathered through `S_j = sum_p e_p
                    // softplus_j(x_p)` (`softplus_0 = 1`) over the set, less the
                    // same sums over the particles that fired `d`, so each mark
                    // costs once per step rather than once per particle.
                    let mask = &masks[group];
                    let width = k + 1;
                    let log_dt = ln(&dt);
                    let mut spread = vec![zero.clone(); width];
                    let mut spread_abs = vec![0.0; width];
                    let mut fired: Vec<Option<(Vec<S>, Vec<f64>)>> = vec![None; marks];
                    let mut coefficients = Vec::with_capacity(self.particles);
                    let mut softplus_all = Vec::with_capacity(self.particles * k);
                    let mut axis_sums = Vec::with_capacity(width);
                    for i in 0..self.particles {
                        let p = first + i;
                        let state = &middle.states[p];
                        let risk = &middle.risk[p];
                        let choice = self.choices[(n - 1) * count + p];
                        if let Some(d) = choice
                            && let Some(range) = &model.layout.jumps[d]
                        {
                            for axis in 0..k {
                                accumulate(&mut share.theta[range.start + axis], &state_bar[i * k + axis]);
                            }
                        }
                        let target_bar = weight_bar[i].clone();
                        let softplus: Vec<S> = state.iter().map(emission::softplus).collect();
                        for d in (0..marks).filter(|&d| mask[d] && risk[d]) {
                            let activity = decoder.activity(d, state);
                            let rate = step_rates.log_scale(d).add(&activity).add(&log_dt);
                            let rate_bar = kill_bar[i].mul(&exp(&rate));
                            add_rate_adjoint(
                                model,
                                &decoder,
                                d,
                                &baseline,
                                state,
                                &activity,
                                &rate_bar,
                                &mut state_bar[i * k..(i + 1) * k],
                                &mut share,
                            );
                        }
                        let retained = step_rates.retained_sums(
                            group,
                            mask,
                            risk,
                            &decoder_weights,
                            &softplus,
                            &mut axis_sums,
                        );
                        let coefficient = if retained > 0 {
                            let total = combine(&axis_sums, &softplus);
                            let exposure = add_real(&ln(&total), step_rates.shift(group)).add(&log_dt);
                            let hazard = exp(&exposure);
                            // d target / d rate_j = scale * fraction_j (+ 1 for the
                            // chosen mark): -H without an event, H/expm1(H) - 1 with one.
                            // H/expm1(H) - 1 = -H/2 + H^2/12 - H^4/720 + ...: the series
                            // omits H^4/720, a relative error H^3/360, while the direct
                            // difference loses 2 eps/H to cancellation; they cross at
                            // H^4 = 720 eps.
                            let scale = match choice {
                                None => hazard.neg(),
                                Some(_) if exposure.value() < (720.0_f64.ln() + f64::EPSILON.ln()) / 4.0 => hazard
                                    .scale(-0.5)
                                    .add(&hazard.mul(&hazard).scale(1.0 / 12.0)),
                                Some(_) => add_real(&div(&hazard, &emission::expm1(&hazard)), -1.0),
                            };
                            // rate_d / total = w_d R_d(x) / (K_0 + sum_k K_k softplus(x_k)).
                            let coefficient = div(&target_bar.mul(&scale), &total);
                            for axis in 0..k {
                                accumulate(
                                    &mut state_bar[i * k + axis],
                                    &coefficient.mul(&axis_sums[axis + 1]).mul(&logistic(&state[axis])),
                                );
                            }
                            if let Some(d) = choice {
                                let activity = decoder.activity(d, state);
                                add_rate_adjoint(
                                    model,
                                    &decoder,
                                    d,
                                    &baseline,
                                    state,
                                    &activity,
                                    &target_bar,
                                    &mut state_bar[i * k..(i + 1) * k],
                                    &mut share,
                                );
                            }
                            coefficient
                        } else {
                            zero.clone()
                        };
                        let terms: Vec<S> = std::iter::once(coefficient.clone())
                            .chain(softplus.iter().map(|s| coefficient.mul(s)))
                            .collect();
                        for (j, term) in terms.iter().enumerate() {
                            accumulate(&mut spread[j], term);
                            spread_abs[j] += term.value().abs();
                        }
                        for d in (0..marks).filter(|&d| !mask[d] && !risk[d]) {
                            let (sums, absolute) =
                                fired[d].get_or_insert_with(|| (vec![zero.clone(); width], vec![0.0; width]));
                            for (j, term) in terms.iter().enumerate() {
                                accumulate(&mut sums[j], term);
                                absolute[j] += term.value().abs();
                            }
                        }
                        coefficients.push(coefficient);
                        softplus_all.extend(softplus);
                    }
                    // Each retained mark's `Q_d = w_d pi_d . (S - G_d)` is the adjoint
                    // of `b_d - m_d`. Removing the fired particles' sums `G_d` amplifies
                    // rounding by at most `(S + G) / (S - G) <= 3` in absolute terms
                    // while their absolute share is at most half; beyond it the
                    // remaining particles are summed directly.
                    for d in (0..marks).filter(|&d| !mask[d]) {
                        let pi = decoder_weights.row(d);
                        let absolute = |sums: &[f64]| pi.iter().zip(sums).map(|(w, s)| w.value() * s).sum::<f64>();
                        let remaining: Vec<S> = match &fired[d] {
                            None => spread.clone(),
                            Some((sums, fired_abs)) if absolute(fired_abs) <= 0.5 * absolute(&spread_abs) => {
                                spread.iter().zip(sums).map(|(s, g)| s.sub(g)).collect()
                            }
                            Some(_) => {
                                let mut sums = vec![zero.clone(); width];
                                for i in (0..self.particles).filter(|&i| middle.risk[first + i][d]) {
                                    accumulate(&mut sums[0], &coefficients[i]);
                                    for axis in 0..k {
                                        accumulate(&mut sums[axis + 1], &coefficients[i].mul(&softplus_all[i * k + axis]));
                                    }
                                }
                                sums
                            }
                        };
                        let w = step_rates.scaled(group, d);
                        let q = w.mul(&pi.iter().zip(&remaining).fold(zero.clone(), |acc, (a, s)| acc.add(&a.mul(s))));
                        accumulate(&mut share.normaliser[d], &q.neg());
                        for b in 0..baseline.columns() {
                            accumulate(
                                &mut share.theta[model.layout.baseline.start + d * model.spec.baseline_columns + b],
                                &baseline.feature(&q, b),
                            );
                        }
                        let start = model.layout.decoder.start + d * k;
                        for axis in 0..k {
                            accumulate(
                                &mut share.theta[start + axis],
                                &w.mul(&pi[axis + 1]).mul(&remaining[axis + 1]).sub(&pi[axis + 1].mul(&q)),
                            );
                        }
                    }
                    Ok(share)
                })
                .collect();
            let mut phi_bar = vec![zero.clone(); k];
            let mut spread_bar = vec![zero.clone(); k];
            let mut mid_bar = vec![zero.clone(); marks];
            for share in shares {
                let share = share?;
                fold(&mut theta_bar, &share.theta);
                fold(&mut phi_bar, &share.phi);
                fold(&mut spread_bar, &share.spread);
                fold(&mut mid_bar, &share.normaliser);
            }
            // Midpoint rows and their normaliser.
            for d in 0..marks {
                accumulate(&mut mid_bar[d], &moment_adjoint[mid_row + d]);
                accumulate(
                    &mut mass_bar[self.mark_groups[d]],
                    &mass_adjoint[mid_row + d].scale(0.5),
                );
            }
            // Per risk set: midpoint moments, then the first half step.
            let shares: Vec<SetShare<S>> = set_slices(&mut state_bar, groups)
                .into_par_iter()
                .zip(weight_bar.par_chunks_mut(self.particles))
                .enumerate()
                .map(|(group, (state_bar, weight_bar))| {
                    let mut share = SetShare::new(&zero, theta.len(), k, 0);
                    self.add_moment_adjoint(
                        model,
                        &decoder,
                        &middle,
                        &mid_bar,
                        group,
                        state_bar,
                        weight_bar,
                        &mut share.theta,
                    );
                    self.add_half_step_adjoint(
                        model,
                        &drive,
                        &columns,
                        &genes,
                        &step,
                        &start.states,
                        self.block(2 * n - 1, k),
                        group,
                        state_bar,
                        &mut share,
                    );
                    share
                })
                .collect();
            for share in shares {
                fold(&mut theta_bar, &share.theta);
                fold(&mut phi_bar, &share.phi);
                fold(&mut spread_bar, &share.spread);
            }
            // phi = exp(-dt softplus(rho) / 2) and spread = sqrt(1 - phi^2).
            for axis in 0..k {
                let phi = &step.phi[axis];
                // A static axis has spread 0, which carries no derivative.
                let total = if step.spread[axis].value() > 0.0 {
                    phi_bar[axis].sub(&div(&spread_bar[axis].mul(phi), &step.spread[axis]))
                } else {
                    phi_bar[axis].clone()
                };
                accumulate(
                    &mut theta_bar[model.layout.rates.start + axis],
                    &total
                        .mul(phi)
                        .mul(&logistic(&theta[model.layout.rates.start + axis]))
                        .mul(&dt.scale(0.5))
                        .neg(),
                );
            }
        }
        let entry = model.entry_features(&self.grid.profile().history(marks));
        let shares: Vec<Vec<S>> = set_slices(&mut state_bar, groups)
            .into_par_iter()
            .zip(weight_bar.par_chunks_mut(self.particles))
            .enumerate()
            .map(|(group, (state_bar, weight_bar))| {
                let mut theta_share = vec![zero.clone(); theta.len()];
                self.add_moment_adjoint(
                    model,
                    &decoder,
                    &checkpoints[0],
                    &moment_adjoint[..marks],
                    group,
                    state_bar,
                    weight_bar,
                    &mut theta_share,
                );
                for i in 0..self.particles {
                    for axis in 0..k {
                        add_mean_adjoint(
                            &mut theta_share,
                            model.layout.entry.start,
                            &entry,
                            &genes[group * self.particles + i],
                            axis,
                            &state_bar[i * k + axis],
                        );
                    }
                }
                theta_share
            })
            .collect();
        for share in shares {
            fold(&mut theta_bar, &share);
        }
        if theta_bar.iter().any(|v| !v.value().is_finite()) {
            return Err(numerical("non-finite reference pullback"));
        }
        Ok(theta_bar)
    }
}
