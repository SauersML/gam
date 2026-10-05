//! A learned mixture prior over the library's read directions: soft weight-sharing (Nowlan and
//! Hinton 1992) that finds read–write ties and shared query–key functions by gradient (#2951).
//!
//! # The prior
//!
//! A target is a vector of the library's parameters `g`: an MLP function's gate direction `g_i`
//! (row `i` of `library.l{l}.mlp.gate`), or a head's query–key map (the rows of its `Q` and `K` in
//! the rotary planes still in the explanation). The Gaussian prior of its groups, `N(0, diag v)`
//! with `v` each entry's group's empirical-Bayes variance (`library_mdl`), becomes the mixture
//!
//! `p(g) = π_0 N(g; 0, diag v) + Σ_{j ∈ C} π_j N(g; c_j u_j, s² I)`
//!
//! over `K` candidates `u_j`. A gate's are writes: earlier functions' output vectors (columns of
//! `library.l{l'}.mlp.out`, `l' < l`, at the weight sample) and `M`'s token embedding rows (the
//! columns of `wte`). A head's are the query–key maps of heads in other layers with keys of their
//! own, each brought to the target's gauge plane by plane (`library_sharing::gauge`, a rotation and
//! a scale that leave its scores unchanged), the gauge fixed for the epoch. The scales `c_j`, the
//! weights `π = softmax(z)` of the logits `z` and the variance `s²` are learned. As the groups' own
//! Gaussian times `r(g) = π_0 + Σ_j π_j N(g; c_j u_j, s² I) / N(g; 0, diag v)`, the divergence is
//! `KL(q ‖ N(0, diag v)) − E_q[ln r(g)]`. `library_mdl` keeps the first term's closed form; this
//! module estimates the second from the fit's own reparameterized weight sample, an unbiased
//! estimate, so `F` stays a code length. `v` enters `r` as the closed form sets it.
//!
//! The mixture's own parameters are sent too: each target's `K` candidates among its `n` choices
//! (`ln C(n, K)` nats), and its `K` free logits, `K` scales and its variance, each at the precision
//! of a value estimated from the target's `|G|` entries (`½ ln |G|` nats), as `library_mdl` prices
//! a group's variance. A removed target sends nothing.
//!
//! # Candidates
//!
//! The `K` candidates only bound the compute; the weights decide. Each epoch they are re-chosen
//! per target by the posterior-normalized misfit of the best scaled candidate,
//! `m(u) = min_c Σ_k (μ_k − c u_k)² / σ_k²` over the target's posterior means `μ` and deviations
//! `σ`: the `K` candidates of smallest misfit. A candidate kept keeps its logit and scale; a new
//! one starts at its least-squares scale and the zero component's logit.
//!
//! # Hardening
//!
//! When one candidate holds more than half of its target's mixture weight, the sharing is made
//! exact: a gate's write becomes a tie (`library_sharing::tie`, `library_sharing::tie_token`), a
//! head's pair one shared query–key function (`library_sharing::share_query_key`). The vector is
//! stored once, its scale is one prior group, and the choice among the target's `n` candidates
//! costs `ln n` nats (`Explanation::fixed_nats`). The hardened explanation is accepted only if its
//! `F` falls after the fit re-converges.

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

/// What a target is: a function's gate direction, or a head's query–key map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Kind {
    Gate { layer: usize, function: usize },
    Head { layer: usize, head: usize },
}

/// A candidate: an earlier function's output vector, a token's embedding row, or another head's
/// query–key map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Write {
    Output { layer: usize, function: usize },
    Token(usize),
    Head { layer: usize, head: usize },
}

/// One mixture component: its candidate, its scale `c`, its logit, and for a head the gauge that
/// brings it to the target's (per plane a rotation and a scale), fixed for the epoch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub write: Write,
    pub scale: f64,
    pub logit: f64,
    pub gauge: Vec<(Array2<f64>, f64)>,
}

/// The mixture prior of one target (module note).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Target {
    pub kind: Kind,
    /// Its prior groups in `Explanation::groups`, and the candidates it may choose among.
    pub groups: Vec<usize>,
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
    /// In the target sample `g`, in each candidate `u_j`, in each scale, in each logit (the zero
    /// component's first) and in `ln s²`.
    pub target: Array1<f64>,
    pub writes: Vec<Array1<f64>>,
    pub scales: Vec<f64>,
    pub logits: Vec<f64>,
    pub log_variance: f64,
}

/// `−ln r(g)` of a target sample `g` against the candidates `(u_j, c_j)` with the logits `z` (the
/// zero component's first), `ln s²` and the groups' variances `v` per entry, and its derivatives
/// (module note).
pub fn term(g: ArrayView1<'_, f64>, writes: &[(ArrayView1<'_, f64>, f64)], logits: &[f64], log_variance: f64, v: ArrayView1<'_, f64>) -> Result<Term, String> {
    if logits.len() != writes.len() + 1 || v.len() != g.len() || v.iter().any(|v| !(*v > 0.0)) || !log_variance.is_finite() {
        return Err("a mixture term needs one logit per component and the zero's, and positive variances".into());
    }
    let d = g.len() as f64;
    let s2 = log_variance.exp();
    let pi = log_softmax(logits).map_err(error)?;
    let zero: f64 = g.iter().zip(&v).map(|(x, v)| -0.5 * (2.0 * PI * v).ln() - x * x / (2.0 * v)).sum();
    let residuals: Vec<Array1<f64>> = writes.iter().map(|(u, c)| &g - &(u * *c)).collect();
    let mut ell = vec![pi[0]];
    for (j, r) in residuals.iter().enumerate() {
        ell.push(pi[j + 1] - 0.5 * d * (2.0 * PI * s2).ln() - r.dot(r) / (2.0 * s2) - zero);
    }
    let value = -log_sum_exp(&ell).map_err(error)?;
    let w: Vec<f64> = log_softmax(&ell).map_err(error)?.into_iter().map(f64::exp).collect();
    let g_over_v = &g / &v;
    let mut target = Array1::zeros(g.len());
    let (mut out_writes, mut scales) = (Vec::with_capacity(writes.len()), Vec::with_capacity(writes.len()));
    let mut log_variance_derivative = 0.0;
    for (j, ((u, c), r)) in writes.iter().zip(&residuals).enumerate() {
        let wj = w[j + 1];
        target.scaled_add(wj / s2, r);
        target.scaled_add(-wj, &g_over_v);
        out_writes.push(r * (-wj * c / s2));
        scales.push(-wj * u.dot(r) / s2);
        log_variance_derivative += wj * (0.5 * d - r.dot(r) / (2.0 * s2));
    }
    let logits_derivative = pi.iter().zip(&w).map(|(p, w)| p.exp() - w).collect();
    Ok(Term { value, target, writes: out_writes, scales, logits: logits_derivative, log_variance: log_variance_derivative })
}

/// Adam's moments of one parameter.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Moment {
    first: f64,
    second: f64,
}

/// A head's query and key operators (trainable indices) and its rotary planes.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct HeadMaps {
    query: usize,
    key: usize,
    planes: Vec<Vec<usize>>,
    /// Per plane, its prior group.
    groups: Vec<usize>,
}

/// The mixture prior of every gate direction and every head with a key of its own of a library
/// explanation (module note).
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
    /// Per layer the trainable indices of its gate and output operators; the heads' maps; the
    /// token embedding (`d × vocabulary`); each gate target's cells (trainable index, rows,
    /// columns).
    gates: Vec<usize>,
    outputs: Vec<usize>,
    heads: Vec<((usize, usize), HeadMaps)>,
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

/// The group variance `Σ (μ² + σ²) / n` over `cells` of `posterior`.
fn group_variance(cells: &[(usize, Vec<usize>, std::ops::Range<usize>)], posterior: &Posterior) -> f64 {
    let (mut count, mut second) = (0.0, 0.0);
    for (i, rows, cols) in cells {
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

impl Mixture {
    /// The mixture prior of every MLP function's gate and every head with a key of its own of
    /// `explanation`, with `width` candidates each, stepped with `steps`; the candidates are chosen
    /// by the first epoch ([`PriorTerm::epoch`]).
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
                targets.push(Target { kind: Kind::Gate { layer: l, function }, groups: vec![group], choices: written + embedding.ncols(), zero_logit: 0.0, components: Vec::new(), log_variance: 0.0 });
                cells.push(explanation.groups[group].cells.iter().map(|c| Ok((position(c.operator)?, c.rows.clone(), c.cols.clone()))).collect::<Result<Vec<_>, String>>()?);
            }
            written += layer.functions.len();
        }
        let mut heads = Vec::new();
        for (&(l, h), found) in &library_sharing::heads(explanation)? {
            let planes = library_sharing::planes(program.operators[found.query].rows.width(), found.rotary);
            let groups = explanation.layers[l].heads[h].0.clone();
            if groups.len() != planes.len() {
                return Err(format!("head {l}.{h}: one prior group per plane required"));
            }
            heads.push(((l, h), HeadMaps { query: position(found.query)?, key: position(found.key)?, planes, groups }));
        }
        for &((l, h), ref maps) in &heads {
            let choices = heads.iter().filter(|((other, _), _)| *other != l).count();
            targets.push(Target { kind: Kind::Head { layer: l, head: h }, groups: maps.groups.clone(), choices, zero_logit: 0.0, components: Vec::new(), log_variance: 0.0 });
            cells.push(Vec::new());
        }
        let moments = targets.iter().map(|_| vec![Moment::default(); 2]).collect();
        Ok(Self { targets, width, steps, taken: 0, moments, gates, outputs, heads, embedding, cells })
    }

    /// The maps of head `head` of layer `layer`.
    fn head(&self, layer: usize, head: usize) -> Result<&HeadMaps, String> {
        self.heads.iter().find(|(at, _)| *at == (layer, head)).map(|(_, m)| m).ok_or_else(|| format!("no head {layer}.{head} in the mixture"))
    }

    /// Whether target `t` is in the explanation (any of its groups is).
    fn active(&self, t: usize, posterior: &Posterior) -> bool {
        self.targets[t].groups.iter().any(|g| posterior.active[*g])
    }

    /// Target `t`'s planes in the explanation (a head's), as `(plane, group)`.
    fn live_planes(&self, maps: &HeadMaps, posterior: &Posterior) -> Vec<usize> {
        (0..maps.planes.len()).filter(|p| posterior.active[maps.groups[*p]]).collect()
    }

    /// A head's vector over the rows of the planes `live` of `(q, k)`: its query rows, then its key
    /// rows, each row's `d` entries in turn.
    fn head_vector(maps: &HeadMaps, live: &[usize], q: &Array2<f64>, k: &Array2<f64>) -> Array1<f64> {
        let rows: Vec<usize> = live.iter().flat_map(|p| maps.planes[*p].clone()).collect();
        let mut out = Vec::with_capacity(2 * rows.len() * q.ncols());
        for m in [q, k] {
            for &r in &rows {
                out.extend(m.row(r).iter().copied());
            }
        }
        Array1::from(out)
    }

    /// The transpose of [`Self::head_vector`]: `vector`'s entries added into `(dq, dk)`.
    fn head_scatter(maps: &HeadMaps, live: &[usize], vector: &Array1<f64>, dq: &mut Array2<f64>, dk: &mut Array2<f64>) {
        let rows: Vec<usize> = live.iter().flat_map(|p| maps.planes[*p].clone()).collect();
        let d = dq.ncols();
        for (side, m) in [dq, dk].into_iter().enumerate() {
            for (i, &r) in rows.iter().enumerate() {
                let start = (side * rows.len() + i) * d;
                m.row_mut(r).scaled_add(1.0, &vector.slice(ndarray::s![start..start + d]));
            }
        }
    }

    /// The earlier outputs whose groups are in the explanation, for a gate of layer `layer`: a
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

    /// Target `t`'s new candidates `(write, least-squares scale, gauge)` replace its old ones: kept
    /// candidates keep their logit, scale and moments, new ones start at the zero logit.
    fn adopt(&mut self, t: usize, chosen: Vec<(Write, f64, Vec<(Array2<f64>, f64)>)>, initial_log_variance: f64) {
        let old = std::mem::take(&mut self.targets[t].components);
        let zero = self.targets[t].zero_logit;
        let mut moments = vec![self.moments[t][0]];
        let mut components = Vec::with_capacity(chosen.len());
        for (write, scale, gauge) in chosen {
            let (component, moment) = match old.iter().position(|c| c.write == write) {
                Some(at) => (Component { gauge, ..old[at].clone() }, [self.moments[t][1 + 2 * at], self.moments[t][2 + 2 * at]]),
                None => (Component { write, scale, logit: zero, gauge }, [Moment::default(); 2]),
            };
            components.push(component);
            moments.extend(moment);
        }
        if old.is_empty() {
            self.targets[t].log_variance = initial_log_variance;
        }
        moments.push(self.moments[t].last().copied().unwrap_or_default());
        self.targets[t].components = components;
        self.moments[t] = moments;
    }

    /// Re-choose every target's candidates at `posterior` (module note). A gate's candidates are
    /// scored `d` at a time, so the scores held at once are no larger than the gate operator.
    pub fn choose(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        for layer in 0..self.gates.len() {
            let (outputs, mut writes) = self.earlier(layer, posterior, explanation)?;
            writes.extend((0..self.embedding.ncols()).map(Write::Token));
            let (mu, log_sd) = (&posterior.mean[self.gates[layer]], &posterior.log_sd[self.gates[layer]]);
            let precision = log_sd.mapv(|s| (-2.0 * s).exp());
            let weighted = mu * &precision;
            let rows: Vec<usize> = (0..self.targets.len())
                .filter(|&t| matches!(self.targets[t].kind, Kind::Gate { layer: l, .. } if l == layer) && self.active(t, posterior))
                .collect();
            // Per target its best candidates so far: (−(Σ μ u/σ²)² / Σ u²/σ², scale, candidate).
            let mut best: Vec<Vec<(f64, f64, usize)>> = vec![Vec::new(); rows.len()];
            let d = self.embedding.nrows();
            let mut start = 0;
            while start < writes.len() {
                let end = (start + d).min(writes.len());
                let block: Array2<f64> = Array2::from_shape_fn((d, end - start), |(r, k)| {
                    let k = start + k;
                    if k < outputs.ncols() { outputs[[r, k]] } else { self.embedding[[r, k - outputs.ncols()]] }
                });
                let (cross, norm) = (fast_ab(&weighted, &block), fast_ab(&precision, &block.mapv(|v| v * v)));
                for (slot, &t) in rows.iter().enumerate() {
                    let Kind::Gate { function: i, .. } = self.targets[t].kind else { continue };
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
                let Kind::Gate { function: i, .. } = self.targets[t].kind else { continue };
                // A new target's variance starts at its gate's mean posterior variance.
                let initial = log_sd.row(i).mapv(|s| (2.0 * s).exp()).mean().ok_or("an empty gate")?.ln();
                let chosen = best[slot].iter().map(|&(_, scale, k)| (writes[k], scale, Vec::new())).collect();
                self.adopt(t, chosen, initial);
            }
        }
        // Heads: every head of another layer, brought to the target's gauge at the means.
        for t in 0..self.targets.len() {
            let Kind::Head { layer, head } = self.targets[t].kind else { continue };
            if !self.active(t, posterior) {
                continue;
            }
            let maps = self.head(layer, head)?;
            let live = self.live_planes(maps, posterior);
            let (q1, k1) = (&posterior.mean[maps.query], &posterior.mean[maps.key]);
            let mu = Self::head_vector(maps, &live, q1, k1);
            let sd = Self::head_vector(maps, &live, &posterior.log_sd[maps.query], &posterior.log_sd[maps.key]);
            let precision = sd.mapv(|s| (-2.0 * s).exp());
            let mut scored = Vec::new();
            for &((l, h), ref other) in &self.heads {
                if l == layer || other.planes != maps.planes {
                    continue;
                }
                let (q, k) = (&posterior.mean[other.query], &posterior.mean[other.key]);
                let gauge = library_sharing::gauge(q1, k1, q, k, &maps.planes);
                let (aq, ak) = library_sharing::apply_gauge(q, k, &maps.planes, &gauge, false);
                let u = Self::head_vector(maps, &live, &aq, &ak);
                let (cross, norm) = ((&mu * &precision).dot(&u), (&u * &u * &precision).sum());
                if norm > 0.0 {
                    scored.push((-cross * cross / norm, cross / norm, Write::Head { layer: l, head: h }, gauge));
                }
            }
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            scored.truncate(self.width);
            let initial = sd.mapv(|s| (2.0 * s).exp()).mean().ok_or("an empty head")?.ln();
            self.adopt(t, scored.into_iter().map(|(_, scale, write, gauge)| (write, scale, gauge)).collect(), initial);
        }
        Ok(())
    }

    /// The targets whose one candidate holds more than half of the mixture weight, with it.
    pub fn dominant(&self, posterior: &Posterior) -> Result<Vec<(usize, usize)>, String> {
        let mut out = Vec::new();
        for (t, target) in self.targets.iter().enumerate() {
            if !self.active(t, posterior) {
                continue;
            }
            let weights = target.weights()?;
            if let Some(j) = (1..weights.len()).find(|&j| weights[j] > 0.5) {
                out.push((t, j - 1));
            }
        }
        Ok(out)
    }

    /// `explanation` with every candidate that dominates at `posterior` made exact (module note):
    /// an earlier output becomes a scale times the gate (`library_sharing::tie`), a gate a scale
    /// times a token's embedding row (`library_sharing::tie_token`), each scale starting at the
    /// least-squares fit of the vector it replaces at `explanation`'s values, and two heads one
    /// shared query–key function (`library_sharing::share_query_key`). Each choice costs `ln n`
    /// nats. An earlier output or a head that two targets dominate is taken by the first.
    pub fn harden(&self, explanation: &Explanation, posterior: &Posterior) -> Result<Explanation, String> {
        let program = &explanation.artifact.program;
        let values = |name: String| -> Result<Array2<f64>, String> { Ok(program.operators[operator_index(program, &name)?].matrix()) };
        let mut ties: Vec<Tie> = Vec::new();
        let mut tokens = Vec::new();
        let mut pairs: Vec<[(usize, usize); 2]> = Vec::new();
        let mut nats = 0.0;
        for (t, j) in self.dominant(posterior)? {
            let target = &self.targets[t];
            match (target.kind, target.components[j].write) {
                (Kind::Gate { layer: gl, function: gi }, Write::Output { layer, function }) => {
                    let g = values(format!("library.l{gl}.mlp.gate"))?.row(gi).to_owned();
                    let u = values(format!("library.l{layer}.mlp.out"))?.column(function).to_owned();
                    if ties.iter().any(|tie| tie.source == (layer, function)) || g.dot(&g) == 0.0 {
                        continue;
                    }
                    ties.push(Tie { source: (layer, function), target: (gl, gi), scale: u.dot(&g) / g.dot(&g) });
                }
                (Kind::Gate { layer: gl, function: gi }, Write::Token(token)) => {
                    let g = values(format!("library.l{gl}.mlp.gate"))?.row(gi).to_owned();
                    let e = self.embedding.column(token);
                    if e.dot(&e) == 0.0 {
                        continue;
                    }
                    tokens.push(((gl, gi), token, g.dot(&e) / e.dot(&e)));
                }
                (Kind::Head { layer, head }, Write::Head { layer: other, head: h }) => {
                    let (a, b) = ((layer, head), (other, h));
                    if pairs.iter().flatten().any(|m| *m == a || *m == b) {
                        continue;
                    }
                    pairs.push(if a < b { [a, b] } else { [b, a] });
                }
                _ => return Err("a component's candidate of another kind than its target".into()),
            }
            nats += (target.choices as f64).ln();
        }
        let mut out = library_sharing::tie(explanation, &ties)?;
        for (target, token, scale) in tokens {
            out = library_sharing::tie_token(&out, target, token, scale)?;
        }
        for pair in pairs {
            out = library_sharing::share_query_key(&out, &pair)?;
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

    /// Target `t`'s sample, its candidates' samples and its groups' variances per entry at `theta`
    /// and `posterior`.
    fn vectors(&self, t: usize, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>) -> Result<(Array1<f64>, Vec<Array1<f64>>, Array1<f64>), String> {
        let get = |i: usize| theta.get(&i).ok_or("an operator's sample");
        let target = &self.targets[t];
        match target.kind {
            Kind::Gate { layer, function } => {
                let g = get(self.gates[layer])?.row(function).to_owned();
                let writes = target
                    .components
                    .iter()
                    .map(|c| match c.write {
                        Write::Output { layer, function } => Ok(get(self.outputs[layer])?.column(function).to_owned()),
                        Write::Token(token) => Ok(self.embedding.column(token).to_owned()),
                        Write::Head { .. } => Err("a head candidate for a gate".to_string()),
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let v = Array1::from_elem(g.len(), group_variance(&self.cells[t], posterior));
                Ok((g, writes, v))
            }
            Kind::Head { layer, head } => {
                let maps = self.head(layer, head)?;
                let live = self.live_planes(maps, posterior);
                let g = Self::head_vector(maps, &live, get(maps.query)?, get(maps.key)?);
                let writes = target
                    .components
                    .iter()
                    .map(|c| match c.write {
                        Write::Head { layer, head } => {
                            let other = self.head(layer, head)?;
                            let (aq, ak) = library_sharing::apply_gauge(get(other.query)?, get(other.key)?, &maps.planes, &c.gauge, false);
                            Ok(Self::head_vector(maps, &live, &aq, &ak))
                        }
                        _ => Err("a write candidate for a head".to_string()),
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let d = posterior.mean[maps.query].ncols();
                let mut v = Vec::with_capacity(g.len());
                for _side in 0..2 {
                    for &p in &live {
                        let cells: Vec<(usize, Vec<usize>, std::ops::Range<usize>)> = [maps.query, maps.key].iter().map(|&i| (i, maps.planes[p].clone(), 0..d)).collect();
                        let variance = group_variance(&cells, posterior);
                        v.extend(std::iter::repeat_n(variance, maps.planes[p].len() * d));
                    }
                }
                Ok((g, writes, Array1::from(v)))
            }
        }
    }
}

impl PriorTerm for Mixture {
    fn operators(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .gates
            .iter()
            .chain(&self.outputs)
            .copied()
            .chain(self.cells.iter().flatten().map(|c| c.0))
            .chain(self.heads.iter().flat_map(|(_, m)| [m.query, m.key]))
            .collect();
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
        let slot = |gradient: &mut BTreeMap<usize, Array2<f64>>, i: usize| -> Result<(), String> {
            if !gradient.contains_key(&i) {
                gradient.insert(i, Array2::zeros(theta.get(&i).ok_or("an operator's sample")?.dim()));
            }
            Ok(())
        };
        for t in 0..self.targets.len() {
            let target = &self.targets[t];
            if !self.active(t, posterior) || target.components.is_empty() {
                continue;
            }
            let (g, writes, v) = self.vectors(t, posterior, theta)?;
            let views: Vec<(ArrayView1<'_, f64>, f64)> = writes.iter().zip(&target.components).map(|(u, c)| (u.view(), c.scale)).collect();
            let logits: Vec<f64> = std::iter::once(target.zero_logit).chain(target.components.iter().map(|c| c.logit)).collect();
            let found = term(g.view(), &views, &logits, target.log_variance, v.view())?;
            value += found.value;
            match target.kind {
                Kind::Gate { layer, function } => {
                    slot(&mut gradient, self.gates[layer])?;
                    gradient.get_mut(&self.gates[layer]).ok_or("slot")?.row_mut(function).scaled_add(1.0, &found.target);
                    for (component, derivative) in target.components.iter().zip(&found.writes) {
                        if let Write::Output { layer, function } = component.write {
                            slot(&mut gradient, self.outputs[layer])?;
                            gradient.get_mut(&self.outputs[layer]).ok_or("slot")?.column_mut(function).scaled_add(1.0, derivative);
                        }
                    }
                }
                Kind::Head { layer, head } => {
                    let maps = self.head(layer, head)?;
                    let live = self.live_planes(maps, posterior);
                    let (q, k) = (theta.get(&maps.query).ok_or("a sample")?, theta.get(&maps.key).ok_or("a sample")?);
                    let (mut dq, mut dk) = (Array2::zeros(q.dim()), Array2::zeros(k.dim()));
                    Self::head_scatter(maps, &live, &found.target, &mut dq, &mut dk);
                    for (i, m) in [(maps.query, dq), (maps.key, dk)] {
                        slot(&mut gradient, i)?;
                        *gradient.get_mut(&i).ok_or("slot")? += &m;
                    }
                    for (component, derivative) in target.components.iter().zip(&found.writes) {
                        let Write::Head { layer, head } = component.write else { continue };
                        let other = self.head(layer, head)?;
                        let (mut aq, mut ak) = (Array2::zeros(q.dim()), Array2::zeros(k.dim()));
                        Self::head_scatter(maps, &live, derivative, &mut aq, &mut ak);
                        // The gauge is linear: its transpose takes the derivative back to the head's maps.
                        let (dq, dk) = library_sharing::apply_gauge(&aq, &ak, &maps.planes, &component.gauge, true);
                        for (i, m) in [(other.query, dq), (other.key, dk)] {
                            slot(&mut gradient, i)?;
                            *gradient.get_mut(&i).ok_or("slot")? += &m;
                        }
                    }
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

    fn cost(&self, posterior: &Posterior) -> Result<f64, String> {
        let k = self.width as f64;
        let mut total = 0.0;
        for t in (0..self.targets.len()).filter(|&t| self.active(t, posterior) && !self.targets[t].components.is_empty()) {
            let target = &self.targets[t];
            let size = match target.kind {
                Kind::Gate { .. } => self.cells[t].iter().map(|(_, rows, cols)| (rows.len() * cols.len()) as f64).sum::<f64>(),
                Kind::Head { layer, head } => {
                    let maps = self.head(layer, head)?;
                    let d = posterior.mean[maps.query].ncols();
                    self.live_planes(maps, posterior).iter().map(|p| (2 * maps.planes[*p].len() * d) as f64).sum()
                }
            };
            let n = target.choices as f64;
            let kept = k.min(n);
            // ln C(n, K) for the candidates, ½ ln |G| for each of the 2K + 1 values.
            total += ln_gamma(n + 1.0) - ln_gamma(kept + 1.0) - ln_gamma(n - kept + 1.0) + (2.0 * kept + 1.0) * 0.5 * size.ln();
        }
        Ok(total)
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
        let variances = Array1::from_elem(3, v);
        let estimates: Vec<f64> = draw(4000, 0).iter().map(|g| closed + term(g.view(), &writes, &logits, log_variance, variances.view()).unwrap().value).collect();
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
    fn exact_copies_are_found_by_their_weights_and_made_exact_keeping_the_outputs() {
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
        let replace = |program: &mut crate::operator_program::OperatorProgram, op: usize, values: Array2<f64>| {
            let source = &program.operators[op];
            let precision = exact_precision(values.iter().copied()).unwrap();
            program.operators[op] = Arc::new(crate::operator_program::Operator::dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, precision, Provenance::default()).unwrap());
        };
        replace(program, gate, values);
        // Head 0 of layer 1 attends as head 0 of layer 0 does.
        let found = library_sharing::heads(&start).unwrap();
        let (first, second) = (&found[&(0, 0)], &found[&(1, 0)]);
        let program = &mut start.artifact.program;
        for (from, to) in [(first.query, second.query), (first.key, second.key)] {
            let values = program.operators[from].matrix();
            replace(program, to, values);
        }
        let settings = Settings { batch_sequences: 2, rate: 0.1, beta1: 0.9, seed: 3, numeric_bytes: 1 << 26, head_tile_rows: 64 };
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let (train, held) = sequences.split_at(4);
        let fitted = fit(&Device::host(), &native, &start, train, held, &settings, "tiny", None, Some(&mut mixture)).unwrap();
        // The learned weights pick each copy among the candidates.
        let found = |kind: Kind, write: Write| -> (usize, usize) {
            let target = mixture.targets.iter().position(|t| t.kind == kind).unwrap();
            let weights = mixture.targets[target].weights().unwrap();
            let copy = mixture.targets[target].components.iter().position(|c| c.write == write).expect("the copy is a candidate");
            assert!(weights[1 + copy] > 0.5, "the copy's weight dominates its mixture: {weights:?}");
            (target, copy)
        };
        let (target, copy) = found(Kind::Gate { layer: 1, function: 5 }, Write::Output { layer: 0, function: 3 });
        let head_copy = found(Kind::Head { layer: 1, head: 0 }, Write::Head { layer: 0, head: 0 });
        assert!(fitted.report.start.prior_bits.is_finite() && fitted.report.start.prior_bits != 0.0, "F holds the mixture's term");
        // Made exact at the start (every group in), the tie keeps the outputs and pays for its choice.
        let all_in = crate::library_mdl::Posterior::new(&start, 96).unwrap();
        let dominant = mixture.dominant(&all_in).unwrap();
        assert!(dominant.contains(&(target, copy)) && dominant.contains(&head_copy));
        let hardened = mixture.harden(&start, &all_in).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), hardened.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[hardened.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the exact tie keeps the outputs");
        assert!(hardened.fixed_nats >= (mixture.targets[target].choices as f64).ln() + (mixture.targets[head_copy.0].choices as f64).ln());
        assert!(hardened.groups.iter().any(|g| g.name == "library.l1.h0.q_shared_scale") || hardened.groups.iter().any(|g| g.name == "library.l0.h0.q_shared_scale"), "the heads share one query-key function");
    }

    #[test]
    fn the_term_is_differentiated_exactly() {
        let g = ndarray::array![0.7, -0.2, 1.1, 0.3];
        let (u1, u2) = (ndarray::array![0.6, -0.1, 1.0, 0.4], ndarray::array![-0.3, 0.9, 0.2, -0.5]);
        let (c, logits, log_variance, v) = ([1.1, -0.6], [0.2, 0.5, -0.3], (0.3_f64).ln(), 0.9);
        let at = |g: &Array1<f64>, u1: &Array1<f64>, u2: &Array1<f64>, c: [f64; 2], logits: [f64; 3], lv: f64| {
            term(g.view(), &[(u1.view(), c[0]), (u2.view(), c[1])], &logits, lv, ndarray::array![v, 1.3 * v, 0.8 * v, 1.1 * v].view()).unwrap()
        };
        let found = at(&g, &u1, &u2, c, logits, log_variance);
        let h = 1e-6;
        let central = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
        let close = |a: f64, b: f64| assert!((a - b).abs() <= 1e-6 * (1.0 + b.abs()), "{a} against {b}");
        for k in 0..4 {
            close(found.target[k], central(&|e| {
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
