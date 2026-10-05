//! A learned mixture prior over the library's gate directions: soft weight-sharing (Nowlan and
//! Hinton 1992) that finds read–write ties by gradient (#2951).
//!
//! # The prior
//!
//! Function `i` of layer `l` reads its input through its gate direction `g_i` (row `i` of
//! `library.l{l}.mlp.gate`, `d` entries). The Gaussian prior `N(0, v_G I)` of its group
//! (`library_mdl`, `v_G` the group's empirical-Bayes variance) becomes the mixture
//!
//! `p(g_i) = π_0 N(g_i; 0, v_G I) + Σ_{j ∈ C_i} π_j N(g_i; c_j u_j, s_i² I)`
//!
//! over `K` candidate writes `u_j`: earlier functions' output vectors (columns of
//! `library.l{l'}.mlp.out`, `l' < l`, at the weight sample) and `M`'s token embedding rows (the
//! columns of `wte`). The scales `c_j`, the weights `π = softmax(z)` of the logits `z` and the
//! variance `s_i²` are learned. As the group's own Gaussian times
//! `r(g) = π_0 + Σ_j π_j N(g; c_j u_j, s_i² I) / N(g; 0, v_G I)`, the divergence is
//! `KL(q ‖ N(0, v_G I)) − E_q[ln r(g)]`. `library_mdl` keeps the first term's closed form; this
//! module estimates the second from the fit's own reparameterized weight sample, an unbiased
//! estimate, so `F` stays a code length. `v_G` enters `r` as the closed form sets it.
//!
//! The mixture's own parameters are sent too: each target's `K` candidates among its `n` choices
//! (`ln C(n, K)` nats), and its `K` free logits, `K` scales and its variance, each at the precision
//! of a value estimated from the group's `|G|` entries (`½ ln |G|` nats), as `library_mdl` prices a
//! group's variance. A removed target sends nothing.
//!
//! # Candidates
//!
//! The `K` candidates only bound the compute; the weights decide. Each epoch they are re-chosen
//! per target by the posterior-normalized misfit of the best scaled candidate,
//! `m(u) = min_c Σ_k (μ_k − c u_k)² / σ_k²` over the gate's posterior means `μ` and deviations `σ`:
//! the `K` candidates of smallest misfit. A candidate kept keeps its logit and scale; a new one
//! starts at its least-squares scale and the zero component's logit.
//!
//! # Hardening
//!
//! When one candidate holds more than half of its target's mixture weight, the tie is made exact
//! (`library_sharing::tie`, `library_sharing::tie_token`): the vector is stored once, its scale is
//! one prior group, and the choice among the target's `n` writes costs `ln n` nats
//! (`Explanation::fixed_nats`). The hardened explanation is accepted only if its `F` falls after the
//! fit re-converges.

use crate::{
    library_mdl::{Explanation, Posterior, PriorTerm},
    library_sharing::{self, Tie},
    operator_program::OperatorProgram,
};
use gam_linalg::faer_ndarray::fast_ab;
use gam_math::categorical::{log_softmax, log_sum_exp};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;
use std::collections::BTreeMap;
use std::f64::consts::PI;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A candidate write: an earlier function's output vector, or a token's embedding row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Write {
    Output { layer: usize, function: usize },
    Token(usize),
}

/// One mixture component: its write, its scale `c` and its logit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub write: Write,
    pub scale: f64,
    pub logit: f64,
}

/// The mixture prior of one gate direction (module note).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Target {
    pub layer: usize,
    pub function: usize,
    /// The gate's prior group in `Explanation::groups`, and the writes it may choose among.
    pub group: usize,
    pub choices: usize,
    pub zero_logit: f64,
    pub components: Vec<Component>,
    /// `ln s²`.
    pub log_variance: f64,
}

impl Target {
    /// The mixture weights `π`, the zero component's first.
    pub fn weights(&self) -> Result<Vec<f64>, String> {
        let logits: Vec<f64> = std::iter::once(self.zero_logit).chain(self.components.iter().map(|c| c.logit)).collect();
        Ok(log_softmax(&logits).map_err(error)?.into_iter().map(f64::exp).collect())
    }
}

/// Adam's settings for the mixture's own parameters.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Steps {
    pub rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

/// The value of one target's term `−ln r(g)` (module note) and its derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub value: f64,
    /// In the gate sample `g`, in each candidate `u_j`, in each scale, in each logit (the zero
    /// component's first) and in `ln s²`.
    pub gate: Array1<f64>,
    pub writes: Vec<Array1<f64>>,
    pub scales: Vec<f64>,
    pub logits: Vec<f64>,
    pub log_variance: f64,
}

/// `−ln r(g)` of a gate sample `g` against the candidates `(u_j, c_j)` with the logits `z` (the zero
/// component's first), `ln s²` and the group's variance `v`, and its derivatives (module note).
pub fn term(g: ArrayView1<'_, f64>, writes: &[(ArrayView1<'_, f64>, f64)], logits: &[f64], log_variance: f64, v: f64) -> Result<Term, String> {
    if logits.len() != writes.len() + 1 || !(v > 0.0) || !log_variance.is_finite() {
        return Err("a mixture term needs one logit per component and the zero's, and positive variances".into());
    }
    let d = g.len() as f64;
    let s2 = log_variance.exp();
    let pi = log_softmax(logits).map_err(error)?;
    let zero = -0.5 * d * (2.0 * PI * v).ln() - g.dot(&g) / (2.0 * v);
    let residuals: Vec<Array1<f64>> = writes.iter().map(|(u, c)| &g - &(u * *c)).collect();
    let mut ell = vec![pi[0]];
    for (j, r) in residuals.iter().enumerate() {
        ell.push(pi[j + 1] - 0.5 * d * (2.0 * PI * s2).ln() - r.dot(r) / (2.0 * s2) - zero);
    }
    let value = -log_sum_exp(&ell).map_err(error)?;
    let w: Vec<f64> = log_softmax(&ell).map_err(error)?.into_iter().map(f64::exp).collect();
    let mut gate = Array1::zeros(g.len());
    let (mut out_writes, mut scales) = (Vec::with_capacity(writes.len()), Vec::with_capacity(writes.len()));
    let mut log_variance_derivative = 0.0;
    for (j, ((u, c), r)) in writes.iter().zip(&residuals).enumerate() {
        let wj = w[j + 1];
        gate.scaled_add(wj / s2, r);
        gate.scaled_add(-wj / v, &g);
        out_writes.push(r * (-wj * c / s2));
        scales.push(-wj * u.dot(r) / s2);
        log_variance_derivative += wj * (0.5 * d - r.dot(r) / (2.0 * s2));
    }
    let logits_derivative = pi.iter().zip(&w).map(|(p, w)| p.exp() - w).collect();
    Ok(Term { value, gate, writes: out_writes, scales, logits: logits_derivative, log_variance: log_variance_derivative })
}

/// Adam's moments of one parameter.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Moment {
    first: f64,
    second: f64,
}

/// The mixture prior of every gate direction of a library explanation (module note).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mixture {
    pub targets: Vec<Target>,
    /// Candidates per target.
    pub width: usize,
    steps: Steps,
    taken: u64,
    /// Per target, Adam's moments of its zero logit, then per component its logit and scale, then
    /// its `ln s²`.
    moments: Vec<Vec<Moment>>,
    /// Per layer the trainable indices of its gate and output operators; the token embedding
    /// (`d × vocabulary`); each target group's cells (trainable index, rows, columns).
    gates: Vec<usize>,
    outputs: Vec<usize>,
    #[serde(skip)]
    embedding: Array2<f64>,
    cells: Vec<Vec<(usize, Vec<usize>, std::ops::Range<usize>)>>,
}

fn operator_index(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(index), None) => Ok(index),
        _ => Err(format!("no unique operator {name}")),
    }
}

impl Mixture {
    /// The mixture prior of every MLP function's gate of `explanation`, with `width` candidates each,
    /// stepped with `steps`; the candidates are chosen by the first epoch ([`PriorTerm::epoch`]).
    pub fn new(explanation: &Explanation, width: usize, steps: Steps) -> Result<Self, String> {
        let program = &explanation.artifact.program;
        let position = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("operator {op} is not trainable"));
        let embedding = program.operators[operator_index(program, "wte")?].matrix();
        let (mut gates, mut outputs) = (Vec::new(), Vec::new());
        for l in 0..explanation.layers.len() {
            gates.push(position(operator_index(program, &format!("library.l{l}.mlp.gate"))?)?);
            outputs.push(position(operator_index(program, &format!("library.l{l}.mlp.out"))?)?);
        }
        let mut targets = Vec::new();
        let mut cells = Vec::new();
        let mut written = 0;
        for (l, layer) in explanation.layers.iter().enumerate() {
            for function in 0..layer.functions.len() {
                let name = format!("library.l{l}.mlp.f{function}.gate");
                let group = explanation.groups.iter().position(|g| g.name == name).ok_or_else(|| format!("no group {name}"))?;
                targets.push(Target { layer: l, function, group, choices: written + embedding.ncols(), zero_logit: 0.0, components: Vec::new(), log_variance: 0.0 });
                cells.push(explanation.groups[group].cells.iter().map(|c| Ok((position(c.operator)?, c.rows.clone(), c.cols.clone()))).collect::<Result<Vec<_>, String>>()?);
            }
            written += layer.functions.len();
        }
        let moments = targets.iter().map(|_| vec![Moment::default(); 2]).collect();
        Ok(Self { targets, width, steps, taken: 0, moments, gates, outputs, embedding, cells })
    }

    /// The token embedding (`d × vocabulary`) after a restore, which does not carry it.
    pub fn with_embedding(mut self, explanation: &Explanation) -> Result<Self, String> {
        let program = &explanation.artifact.program;
        self.embedding = program.operators[operator_index(program, "wte")?].matrix();
        Ok(self)
    }

    /// The vector of `write` at the per-operator `values` (by trainable index).
    fn write<'a>(&'a self, write: Write, values: &'a BTreeMap<usize, Array2<f64>>) -> Result<ArrayView1<'a, f64>, String> {
        match write {
            Write::Output { layer, function } => Ok(values.get(&self.outputs[layer]).ok_or("an output operator's values")?.column(function)),
            Write::Token(t) => Ok(self.embedding.column(t)),
        }
    }

    /// The group's empirical-Bayes variance `v_G` at `posterior`.
    fn group_variance(&self, t: usize, posterior: &Posterior) -> f64 {
        let (mut count, mut second) = (0.0, 0.0);
        for (i, rows, cols) in &self.cells[t] {
            for &r in rows {
                for c in cols.clone() {
                    let (m, s) = (posterior.mean[*i][[r, c]], posterior.log_sd[*i][[r, c]]);
                    count += 1.0;
                    second += m * m + (2.0 * s).exp();
                }
            }
        }
        second / count
    }

    /// The earlier outputs whose groups are in the explanation, for a target of layer `layer`: a
    /// `d × n` matrix of their posterior means and their writes (the token rows follow them).
    fn earlier(&self, layer: usize, posterior: &Posterior, explanation: &Explanation) -> Result<(Array2<f64>, Vec<Write>), String> {
        let mut columns = Vec::new();
        let mut writes = Vec::new();
        for earlier in 0..layer {
            let out = &posterior.mean[self.outputs[earlier]];
            for j in 0..out.ncols() {
                let name = format!("library.l{earlier}.mlp.f{j}.out");
                let group = explanation.groups.iter().position(|g| g.name == name).ok_or_else(|| format!("no group {name}"))?;
                if posterior.active[group] {
                    columns.push(out.column(j));
                    writes.push(Write::Output { layer: earlier, function: j });
                }
            }
        }
        let views: Vec<_> = columns.iter().map(|c| c.view().insert_axis(Axis(1))).collect();
        let matrix = if views.is_empty() { Array2::zeros((self.embedding.nrows(), 0)) } else { ndarray::concatenate(Axis(1), &views).map_err(error)? };
        Ok((matrix, writes))
    }

    /// Re-choose every target's candidates at `posterior` (module note). The candidates are scored
    /// `d` at a time, so the scores held at once are no larger than the gate operator.
    pub fn choose(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        for layer in 0..self.gates.len() {
            let (outputs, mut writes) = self.earlier(layer, posterior, explanation)?;
            writes.extend((0..self.embedding.ncols()).map(Write::Token));
            let (mu, log_sd) = (&posterior.mean[self.gates[layer]], &posterior.log_sd[self.gates[layer]]);
            let precision = log_sd.mapv(|s| (-2.0 * s).exp());
            let weighted = mu * &precision;
            let rows: Vec<usize> = (0..self.targets.len()).filter(|&t| self.targets[t].layer == layer && posterior.active[self.targets[t].group]).collect();
            // Per target its best candidates so far: (−(Σ μ u/σ²)² / Σ u²/σ², least-squares scale, candidate).
            let mut best: Vec<Vec<(f64, f64, usize)>> = vec![Vec::new(); rows.len()];
            let d = self.embedding.nrows();
            let total = writes.len();
            let mut start = 0;
            while start < total {
                let end = (start + d).min(total);
                let block: Array2<f64> = Array2::from_shape_fn((d, end - start), |(r, k)| {
                    let k = start + k;
                    if k < outputs.ncols() { outputs[[r, k]] } else { self.embedding[[r, k - outputs.ncols()]] }
                });
                let (cross, norm) = (fast_ab(&weighted, &block), fast_ab(&precision, &block.mapv(|v| v * v)));
                for (slot, &t) in rows.iter().enumerate() {
                    let i = self.targets[t].function;
                    for k in 0..end - start {
                        if norm[[i, k]] > 0.0 {
                            best[slot].push((-cross[[i, k]] * cross[[i, k]] / norm[[i, k]], cross[[i, k]] / norm[[i, k]], start + k));
                        }
                    }
                    best[slot].sort_by(|a, b| a.0.total_cmp(&b.0));
                    best[slot].truncate(self.width);
                }
                start = end;
            }
            for (slot, &t) in rows.iter().enumerate() {
                let old = std::mem::take(&mut self.targets[t].components);
                let zero = self.targets[t].zero_logit;
                let mut moments = vec![self.moments[t][0]];
                let mut components = Vec::with_capacity(best[slot].len());
                for &(_, scale, k) in &best[slot] {
                    let (component, moment) = match old.iter().position(|c| c.write == writes[k]) {
                        Some(at) => (old[at].clone(), [self.moments[t][1 + 2 * at], self.moments[t][2 + 2 * at]]),
                        None => (Component { write: writes[k], scale, logit: zero }, [Moment::default(); 2]),
                    };
                    components.push(component);
                    moments.extend(moment);
                }
                // A new target's variance starts at its gate's mean posterior variance.
                if old.is_empty() {
                    let i = self.targets[t].function;
                    self.targets[t].log_variance = log_sd.row(i).mapv(|s| (2.0 * s).exp()).mean().ok_or("an empty gate")?.ln();
                }
                moments.push(*self.moments[t].last().ok_or("moments")?);
                self.targets[t].components = components;
                self.moments[t] = moments;
            }
        }
        Ok(())
    }

    /// The targets whose one candidate holds more than half of the mixture weight, with it.
    pub fn dominant(&self, posterior: &Posterior) -> Result<Vec<(usize, usize)>, String> {
        let mut out = Vec::new();
        for (t, target) in self.targets.iter().enumerate() {
            if !posterior.active[target.group] {
                continue;
            }
            let weights = target.weights()?;
            if let Some(j) = (1..weights.len()).find(|&j| weights[j] > 0.5) {
                out.push((t, j - 1));
            }
        }
        Ok(out)
    }

    /// `explanation` with every candidate that dominates at `posterior` made an exact tie (module
    /// note): an earlier output becomes a scale times the gate (`library_sharing::tie`), a gate a
    /// scale times a token's embedding row (`library_sharing::tie_token`), each scale starting at
    /// the least-squares fit of the vector it replaces at `explanation`'s values; each tie's choice
    /// costs `ln n` nats. An earlier output that two gates dominate is tied to the first.
    pub fn harden(&self, explanation: &Explanation, posterior: &Posterior) -> Result<Explanation, String> {
        let program = &explanation.artifact.program;
        let values = |name: String| -> Result<Array2<f64>, String> { Ok(program.operators[operator_index(program, &name)?].matrix()) };
        let mut ties: Vec<Tie> = Vec::new();
        let mut tokens = Vec::new();
        let mut nats = 0.0;
        for (t, j) in self.dominant(posterior)? {
            let target = &self.targets[t];
            let g = values(format!("library.l{}.mlp.gate", target.layer))?.row(target.function).to_owned();
            match target.components[j].write {
                Write::Output { layer, function } => {
                    let u = values(format!("library.l{layer}.mlp.out"))?.column(function).to_owned();
                    if ties.iter().any(|tie| tie.source == (layer, function)) || g.dot(&g) == 0.0 {
                        continue;
                    }
                    let scale = u.dot(&g) / g.dot(&g);
                    ties.push(Tie { source: (layer, function), target: (target.layer, target.function), scale });
                }
                Write::Token(token) => {
                    let e = self.embedding.column(token);
                    if e.dot(&e) == 0.0 {
                        continue;
                    }
                    tokens.push(((target.layer, target.function), token, g.dot(&e) / e.dot(&e)));
                }
            }
            nats += (target.choices as f64).ln();
        }
        let mut out = library_sharing::tie(explanation, &ties)?;
        for (target, token, scale) in tokens {
            out = library_sharing::tie_token(&out, target, token, scale)?;
        }
        out.fixed_nats += nats;
        Ok(out)
    }

    /// One Adam step of the mixture's own parameters along `gradients` (per target, in the
    /// moments' order).
    fn learn(&mut self, gradients: &[Vec<f64>]) {
        self.taken += 1;
        let Steps { rate, beta1, beta2, epsilon } = self.steps;
        let (c1, c2) = (1.0 - beta1.powf(self.taken as f64), 1.0 - beta2.powf(self.taken as f64));
        for (t, gradient) in gradients.iter().enumerate() {
            if gradient.is_empty() {
                continue;
            }
            let step = |moment: &mut Moment, value: &mut f64, g: f64| {
                moment.first = beta1 * moment.first + (1.0 - beta1) * g;
                moment.second = beta2 * moment.second + (1.0 - beta2) * g * g;
                *value -= rate * (moment.first / c1) / ((moment.second / c2).sqrt() + epsilon);
            };
            let target = &mut self.targets[t];
            let moments = &mut self.moments[t];
            step(&mut moments[0], &mut target.zero_logit, gradient[0]);
            for (j, component) in target.components.iter_mut().enumerate() {
                step(&mut moments[1 + 2 * j], &mut component.logit, gradient[1 + 2 * j]);
                step(&mut moments[2 + 2 * j], &mut component.scale, gradient[2 + 2 * j]);
            }
            let last = moments.len() - 1;
            step(&mut moments[last], &mut target.log_variance, gradient[gradient.len() - 1]);
        }
    }
}

impl PriorTerm for Mixture {
    fn operators(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.gates.iter().chain(&self.outputs).copied().chain(self.cells.iter().flatten().map(|c| c.0)).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        self.choose(explanation, posterior)
    }

    fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
        let mut value = 0.0;
        let mut gradient: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        let mut learned = vec![Vec::new(); self.targets.len()];
        for (t, target) in self.targets.iter().enumerate() {
            if !posterior.active[target.group] || target.components.is_empty() {
                continue;
            }
            let gates = theta.get(&self.gates[target.layer]).ok_or("a gate operator's sample")?;
            let g = gates.row(target.function);
            let writes = target.components.iter().map(|c| Ok((self.write(c.write, theta)?, c.scale))).collect::<Result<Vec<_>, String>>()?;
            let logits: Vec<f64> = std::iter::once(target.zero_logit).chain(target.components.iter().map(|c| c.logit)).collect();
            let found = term(g, &writes, &logits, target.log_variance, self.group_variance(t, posterior))?;
            value += found.value;
            let shape = |i: usize| theta[&i].dim();
            gradient.entry(self.gates[target.layer]).or_insert_with(|| Array2::zeros(shape(self.gates[target.layer]))).row_mut(target.function).scaled_add(1.0, &found.gate);
            for (component, derivative) in target.components.iter().zip(&found.writes) {
                if let Write::Output { layer, function } = component.write {
                    let i = self.outputs[layer];
                    gradient.entry(i).or_insert_with(|| Array2::zeros(shape(i))).column_mut(function).scaled_add(1.0, derivative);
                }
            }
            let mut own = vec![found.logits[0]];
            for j in 0..target.components.len() {
                own.extend([found.logits[j + 1], found.scales[j]]);
            }
            own.push(found.log_variance);
            learned[t] = own;
        }
        if learn {
            self.learn(&learned);
        }
        Ok((value, gradient))
    }

    fn cost(&self, posterior: &Posterior) -> f64 {
        let k = self.width as f64;
        self.targets
            .iter()
            .enumerate()
            .filter(|(_, t)| posterior.active[t.group] && !t.components.is_empty())
            .map(|(i, t)| {
                let size: f64 = self.cells[i].iter().map(|(_, rows, cols)| (rows.len() * cols.len()) as f64).sum();
                let n = t.choices as f64;
                // ln C(n, K) for the candidates, ½ ln |G| for each of the 2K + 1 values.
                ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0) + (2.0 * k + 1.0) * 0.5 * size.ln()
            })
            .sum()
    }

    fn save(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(error)
    }

    fn load(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let embedding = std::mem::take(&mut self.embedding);
        let mut restored: Mixture = serde_json::from_value(value.clone()).map_err(error)?;
        if restored.targets.len() != self.targets.len() || restored.gates != self.gates || restored.outputs != self.outputs {
            return Err("a checkpoint's mixture of another explanation".into());
        }
        restored.embedding = embedding;
        *self = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_gpu::tensor::posterior_normal;

    fn gaussian_log_density(x: ArrayView1<'_, f64>, mean: ArrayView1<'_, f64>, variance: f64) -> f64 {
        let r = &x - &mean;
        -0.5 * x.len() as f64 * (2.0 * PI * variance).ln() - r.dot(&r) / (2.0 * variance)
    }

    #[test]
    fn the_sampled_divergence_matches_a_high_sample_reference() {
        // q = N(μ, diag σ²) over d = 3; the mixture's zero component N(0, v I) and two writes.
        let (mu, sigma): (Array1<f64>, Array1<f64>) = (ndarray::array![0.9, -0.4, 1.3], ndarray::array![0.3, 0.5, 0.2]);
        let (u1, u2) = (ndarray::array![1.0, -0.5, 1.4], ndarray::array![-0.2, 0.8, 0.1]);
        let writes = [(u1.view(), 0.85), (u2.view(), 1.5)];
        let (logits, log_variance, v): ([f64; 3], f64, f64) = ([0.1, 0.7, -0.4], (0.08_f64).ln(), 1.1);
        let closed = 0.5 * (0..3).map(|k| (v / sigma[k].powi(2)).ln() + (sigma[k].powi(2) + mu[k].powi(2)) / v - 1.0).sum::<f64>();
        let draw = |n: usize, stream: u64| -> Vec<Array1<f64>> {
            (0..n).map(|i| Array1::from_shape_fn(3, |k| mu[k] + sigma[k] * f64::from(posterior_normal(5, stream, (3 * i + k) as u64)))).collect()
        };
        // The estimate: the closed form minus the sampled ln r, one sample per draw.
        let estimates: Vec<f64> = draw(4000, 0).iter().map(|g| closed + term(g.view(), &writes, &logits, log_variance, v).unwrap().value).collect();
        let n = estimates.len() as f64;
        let mean = estimates.iter().sum::<f64>() / n;
        let error = (estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
        // The reference: ln q − ln p of the mixture's own density, averaged over many more draws.
        let pi: Vec<f64> = log_softmax(&logits).unwrap();
        let reference_draws = draw(400_000, 1);
        let reference = reference_draws
            .iter()
            .map(|g| {
                let log_q: f64 = (0..3).map(|k| -0.5 * (2.0 * PI * sigma[k].powi(2)).ln() - (g[k] - mu[k]).powi(2) / (2.0 * sigma[k].powi(2))).sum();
                let parts = [
                    pi[0] + gaussian_log_density(g.view(), Array1::zeros(3).view(), v),
                    pi[1] + gaussian_log_density(g.view(), (&u1 * 0.85).view(), log_variance.exp()),
                    pi[2] + gaussian_log_density(g.view(), (&u2 * 1.5).view(), log_variance.exp()),
                ];
                log_q - log_sum_exp(&parts).unwrap()
            })
            .sum::<f64>()
            / reference_draws.len() as f64;
        assert!((mean - reference).abs() < 4.0 * error, "sampled {mean} ± {error} against {reference}");
    }

    #[test]
    fn an_exact_copy_is_found_by_its_weights_and_tied_keeping_the_outputs() {
        use crate::{import::import_language_model, library_mdl::{Settings, explanation, fit}, operator_program::{Provenance, SlotValues, exact_precision}, run_check::{layer_nodes, split_sites}};
        use gam_gpu::tensor::Device;
        use std::sync::Arc;
        let dir = crate::test_support::tiny_export("library_mixture_copy", 2);
        let imported = import_language_model(&dir, 6, 12).expect("import");
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).expect("split");
        let mut start = explanation(&native, &layer_nodes(&native, 2).expect("layers")).expect("explanation");
        // Function 5 of layer 1 reads 2.5 times what function 3 of layer 0 writes.
        let program = &mut start.artifact.program;
        let (out, gate) = (operator_index(program, "library.l0.mlp.out").unwrap(), operator_index(program, "library.l1.mlp.gate").unwrap());
        let mut values = program.operators[gate].matrix();
        values.row_mut(5).assign(&(&program.operators[out].matrix().column(3) * 2.5));
        let source = &program.operators[gate];
        let precision = exact_precision(values.iter().copied()).unwrap();
        program.operators[gate] = Arc::new(crate::operator_program::Operator::dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, precision, Provenance::default()).unwrap());
        let settings = Settings { batch_sequences: 2, mean_step: 0.01, log_sd_step: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, seed: 3, numeric_bytes: 1 << 26, head_tile_rows: 64 };
        let mut mixture = Mixture::new(&start, 2, Steps { rate: settings.log_sd_step, beta1: settings.beta1, beta2: settings.beta2, epsilon: settings.epsilon }).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let (train, held) = sequences.split_at(4);
        let fitted = fit(&Device::host(), &native, &start, train, held, &settings, "tiny", None, Some(&mut mixture)).unwrap();
        let target = mixture.targets.iter().position(|t| (t.layer, t.function) == (1, 5)).unwrap();
        // The learned weights pick the copy among the candidates.
        let weights = mixture.targets[target].weights().unwrap();
        let copy = mixture.targets[target].components.iter().position(|c| c.write == Write::Output { layer: 0, function: 3 }).expect("the copy is a candidate");
        assert!(weights[1 + copy] > 0.5, "the copy's weight dominates its mixture: {weights:?}");
        assert!(fitted.report.start.prior_bits.is_finite() && fitted.report.start.prior_bits != 0.0, "F holds the mixture's term");
        // Made exact at the start (every group in), the tie keeps the outputs and pays for its choice.
        let all_in = crate::library_mdl::Posterior::new(&start, 96).unwrap();
        assert!(mixture.dominant(&all_in).unwrap().contains(&(target, copy)));
        let hardened = mixture.harden(&start, &all_in).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), hardened.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[hardened.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the exact tie keeps the outputs");
        assert!(hardened.fixed_nats >= (mixture.targets[target].choices as f64).ln());
    }

    #[test]
    fn the_term_is_differentiated_exactly() {
        let g = ndarray::array![0.7, -0.2, 1.1, 0.3];
        let (u1, u2) = (ndarray::array![0.6, -0.1, 1.0, 0.4], ndarray::array![-0.3, 0.9, 0.2, -0.5]);
        let (c, logits, log_variance, v) = ([1.1, -0.6], [0.2, 0.5, -0.3], (0.3_f64).ln(), 0.9);
        let at = |g: &Array1<f64>, u1: &Array1<f64>, u2: &Array1<f64>, c: [f64; 2], logits: [f64; 3], lv: f64| {
            term(g.view(), &[(u1.view(), c[0]), (u2.view(), c[1])], &logits, lv, v).unwrap()
        };
        let found = at(&g, &u1, &u2, c, logits, log_variance);
        let h = 1e-6;
        let central = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
        let close = |a: f64, b: f64| assert!((a - b).abs() <= 1e-6 * (1.0 + b.abs()), "{a} against {b}");
        for k in 0..4 {
            close(found.gate[k], central(&|e| {
                let mut x = g.clone();
                x[k] += e;
                at(&x, &u1, &u2, c, logits, log_variance).value
            }));
            close(found.writes[0][k], central(&|e| {
                let mut x = u1.clone();
                x[k] += e;
                at(&g, &x, &u2, c, logits, log_variance).value
            }));
        }
        for j in 0..2 {
            close(found.scales[j], central(&|e| {
                let mut x = c;
                x[j] += e;
                at(&g, &u1, &u2, x, logits, log_variance).value
            }));
        }
        for j in 0..3 {
            close(found.logits[j], central(&|e| {
                let mut x = logits;
                x[j] += e;
                at(&g, &u1, &u2, c, x, log_variance).value
            }));
        }
        close(found.log_variance, central(&|e| at(&g, &u1, &u2, c, logits, log_variance + e).value));
    }
}
