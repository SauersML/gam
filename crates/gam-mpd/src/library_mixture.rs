//! A learned mixture prior over the library's parameter blocks: soft weight-sharing (Nowlan and
//! Hinton 1992) that finds read–write ties, blocks shared across layers and shared query–key
//! functions by gradient (#2951).
//!
//! # The prior
//!
//! A target is one block of the library's parameters `g`, each block separately: an MLP function's
//! gate direction `g_i` (row `i` of `library.l{l}.mlp.gate`), its up direction when the MLP is
//! gated (row `i` of `library.l{l}.mlp.up`), its output vector (column `i` of `library.l{l}.mlp.out`),
//! a native key-value group's query–key maps (the rows of its query heads' `Q_i` and its key's
//! `K` in the rotary planes still in the explanation; one head where heads do not share keys), or
//! its value map `V` (its coordinates still in the explanation). The
//! Gaussian prior of its groups, `N(0, diag v)` with `v` each entry's group's empirical-Bayes
//! variance (`library_mdl`), becomes the mixture
//!
//! `p(g) = π_0 N(g; 0, diag v) + Σ_{j ∈ C} π_j N(g; c_j u_j, s² I)`
//!
//! over `K` candidates `u_j`, all at the weight sample and all of earlier layers `l' < l`. A read
//! (a gate or up direction)'s are earlier reads of either part and `M`'s token embedding rows (the
//! columns of `wte`), and a gate's also earlier functions' output vectors (a read of what they
//! write); an output's are earlier output vectors. A key-value group's are the query–key maps of groups
//! in earlier layers with keys of their own, as many query heads and the same planes, each
//! brought to the target's gauge plane by plane (`library_sharing::gauge`, a rotation and a scale
//! that leave every one of its heads' scores unchanged) with each target query head facing one of
//! the candidate's (the optimal assignment of their misfits), gauge and assignment fixed for the
//! epoch. A value map's are the value maps `V_s` of earlier groups with as many query heads, each
//! moved by the transport `T` that `M`'s output projections fix (`library_sharing::transports`):
//! `M` keeps the projections `O`, so a group's value–output maps `O_i V` are those of the
//! candidate when `V = T V_s`, the exact symmetry `V → R V`, `O → O R⁻¹` read through the
//! projections; `T` is sent with `M` and costs nothing. The scales `c_j`, the
//! weights `π = softmax(z)` of the logits `z` and the variance `s²` are learned. As the groups' own
//! Gaussian times `r(g) = π_0 + Σ_j π_j N(g; c_j u_j, s² I) / N(g; 0, diag v)`, the divergence is
//! `KL(q ‖ N(0, diag v)) − E_q[ln r(g)]`. `library_mdl` keeps the first term's closed form; this
//! module estimates the second from the fit's own reparameterized weight sample, an unbiased
//! estimate. These are conditional priors: the candidates use the same weight sample as the
//! targets. Every target conditions only on earlier layers, so their product is a normalized joint
//! prior for fixed mixture parameters and gauges. Allowing mutual candidates would instead form
//! a product of cyclic conditionals, which need not be normalizable. `v` enters `r` as the closed
//! form sets it.
//!
//! The mixture's own parameters are sent too: each target's `K` candidates among its `n` choices
//! (`ln C(n, K)` nats), and its `K` free logits, `K` scales and its variance, each at the precision
//! of a value estimated from the target's `|G|` entries (`½ ln |G|` nats), as `library_mdl` prices
//! a group's variance. Head alignment gauges additionally cost 64 bits per stored matrix entry
//! and scale, and each assignment of `m` query heads `ln m!` nats: they depend on the target means
//! and cannot be reconstructed from a parent sample alone. This is a conservative literal charge; the other parameter costs remain the existing
//! asymptotic estimates. A removed target sends nothing.
//!
//! The sample gradient treats `v` and the epoch's gauges as fixed hyperparameters. Recomputing
//! `v` from the posterior is an adaptive update, not the optimum of the mixture objective:
//! the Gaussian-only empirical-Bayes identity does not cancel the mixture's derivative through
//! `v`. This path does not yet implement that total posterior derivative.
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
//! exact: a gate reading an earlier write becomes a tie (`library_sharing::tie`), a gate or up
//! direction a scale times an earlier read or a token's embedding row (`library_sharing::tie_row`),
//! an output vector a scale times an earlier one
//! (`library_sharing::tie_column`), two key-value groups one shared query–key function with the
//! target's query heads assigned as the component's (`library_sharing::share_query_key`), a value
//! map a scale times the transported earlier one (`library_sharing::share_value`). A block takes
//! part in one such sharing per hardening. The vector is
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
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;
use std::collections::BTreeMap;
use std::f64::consts::PI;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// What a target is: one block of an MLP function (its gate direction, its up direction when
/// gated, its output vector), or a key-value group's query–key maps or value map
/// (`library_sharing::KeyValue`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Kind {
    Gate { layer: usize, function: usize },
    Up { layer: usize, function: usize },
    Output { layer: usize, function: usize },
    QueryKey { layer: usize, group: usize },
    Value { layer: usize, group: usize },
}

/// A candidate: an earlier function's output vector, gate direction or up direction, a token's
/// embedding row, or an earlier key-value group's query–key maps or value map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Write {
    Output { layer: usize, function: usize },
    Token(usize),
    QueryKey { layer: usize, group: usize },
    Gate { layer: usize, function: usize },
    Value { layer: usize, group: usize },
    Up { layer: usize, function: usize },
}

impl Write {
    /// The block a target of `kind` is, as a candidate of a later target.
    fn of(kind: Kind) -> Self {
        match kind {
            Kind::Gate { layer, function } => Self::Gate { layer, function },
            Kind::Up { layer, function } => Self::Up { layer, function },
            Kind::Output { layer, function } => Self::Output { layer, function },
            Kind::QueryKey { layer, group } => Self::QueryKey { layer, group },
            Kind::Value { layer, group } => Self::Value { layer, group },
        }
    }

    /// The prior group's name of an MLP block.
    fn group(self) -> Option<String> {
        match self {
            Self::Output { layer, function } => Some(format!("library.l{layer}.mlp.f{function}.out")),
            Self::Gate { layer, function } => Some(format!("library.l{layer}.mlp.f{function}.gate")),
            Self::Up { layer, function } => Some(format!("library.l{layer}.mlp.f{function}.up")),
            Self::Token(_) | Self::QueryKey { .. } | Self::Value { .. } => None,
        }
    }
}

/// One mixture component: its candidate, its scale `c`, its logit, and for a key-value group the
/// gauge that brings its query–key maps to the target's (per plane a rotation and a scale), the
/// candidate's query head each of the target's query heads faces, and the transport `T` that
/// brings its value map to the target's (`library_sharing::transports`, from `M`), fixed for the
/// epoch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub write: Write,
    pub scale: f64,
    pub logit: f64,
    pub gauge: Vec<(Array2<f64>, f64)>,
    #[serde(default)]
    pub assignment: Vec<usize>,
    #[serde(default)]
    pub transport: Option<Array2<f64>>,
}

/// A candidate chosen for a target at an epoch: its least-squares scale and, for a key-value
/// group, its gauge, assignment and transport ([`Component`]).
struct Choice {
    write: Write,
    scale: f64,
    gauge: Vec<(Array2<f64>, f64)>,
    assignment: Vec<usize>,
    transport: Option<Array2<f64>>,
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

/// A key-value group's query operators (in head order) and key operator (trainable indices) and
/// its rotary planes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct GroupMaps {
    queries: Vec<usize>,
    key: usize,
    planes: Vec<Vec<usize>>,
    /// Per plane, its prior group.
    groups: Vec<usize>,
}

/// A key-value group's value operator (trainable index), its number of query heads and its value
/// coordinates' prior groups.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ValueMaps {
    value: usize,
    heads: usize,
    groups: Vec<usize>,
}

/// The mixture prior of every MLP block and every key-value group's maps of its own of a library
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
    /// Per layer the trainable indices of its gate and output operators; the key-value groups'
    /// maps; the token embedding (`d × vocabulary`); each MLP block target's cells (trainable
    /// index, rows, columns).
    gates: Vec<usize>,
    outputs: Vec<usize>,
    /// Per layer, its up operator's trainable index when its MLP is gated.
    #[serde(default)]
    ups: Vec<Option<usize>>,
    key_values: Vec<((usize, usize), GroupMaps)>,
    #[serde(default)]
    values: Vec<((usize, usize), ValueMaps)>,
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
    /// The mixture prior of every MLP block and every key-value group's own query–key maps and
    /// value map of `explanation`, with `width` candidates each, stepped with `steps`; the
    /// candidates are chosen by the first epoch ([`PriorTerm::epoch`]).
    pub fn new(explanation: &Explanation, width: usize, steps: Steps) -> Result<Self, String> {
        let program = &explanation.artifact.program;
        let position = |op: usize| explanation.trainable.iter().position(|t| *t == op).ok_or_else(|| format!("operator {op} is not trainable"));
        let embedding = program.operators[operator_index(program, "wte")?].matrix();
        let (mut gates, mut outputs, mut ups) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..explanation.layers.len() {
            gates.push(position(operator_index(program, &format!("library.l{l}.mlp.gate"))?)?);
            outputs.push(position(operator_index(program, &format!("library.l{l}.mlp.out"))?)?);
            ups.push(match operator_index(program, &format!("library.l{l}.mlp.up")) {
                Ok(op) => Some(position(op)?),
                Err(_) => None,
            });
        }
        let mut targets = Vec::new();
        let mut cells = Vec::new();
        // Each block's candidates are blocks of earlier layers (and token rows for a gate), so the
        // product of the conditional mixtures is a joint density.
        let mut written = 0;
        for (l, layer) in explanation.layers.iter().enumerate() {
            for function in 0..layer.functions.len() {
                let earlier_ups: usize = (0..l).filter(|e| ups[*e].is_some()).map(|e| explanation.layers[e].functions.len()).sum();
                let mut blocks = vec![(Kind::Gate { layer: l, function }, 2 * written + earlier_ups + embedding.ncols()), (Kind::Output { layer: l, function }, written)];
                if ups[l].is_some() {
                    blocks.push((Kind::Up { layer: l, function }, written + earlier_ups + embedding.ncols()));
                }
                for (kind, choices) in blocks {
                    let name = Write::of(kind).group().ok_or("an MLP block's group")?;
                    let group = explanation.groups.iter().position(|g| g.name == name).ok_or_else(|| format!("no group {name}"))?;
                    targets.push(Target { kind, groups: vec![group], choices, zero_logit: 0.0, components: Vec::new(), log_variance: 0.0 });
                    cells.push(explanation.groups[group].cells.iter().map(|c| Ok((position(c.operator)?, c.rows.clone(), c.cols.clone()))).collect::<Result<Vec<_>, String>>()?);
                }
            }
            written += layer.functions.len();
        }
        // Every key-value group whose key is its own; a member of a shared function is not.
        let mut key_values = Vec::new();
        for (&(l, g), found) in &library_sharing::key_values(explanation)? {
            if !found.own_key {
                continue;
            }
            let planes = library_sharing::planes(program.operators[found.key].rows.width(), found.rotary);
            let groups = explanation.layers[l].heads[found.heads[0].0].0.clone();
            if groups.len() != planes.len() {
                return Err(format!("key-value group {l}.{g}: one prior group per plane required"));
            }
            let queries = found.queries().into_iter().map(position).collect::<Result<Vec<_>, _>>()?;
            key_values.push(((l, g), GroupMaps { queries, key: position(found.key)?, planes, groups }));
        }
        for &((l, g), ref maps) in &key_values {
            let choices = key_values.iter().filter(|((other, _), other_maps)| *other < l && Self::compatible(maps, other_maps)).count();
            targets.push(Target { kind: Kind::QueryKey { layer: l, group: g }, groups: maps.groups.clone(), choices, zero_logit: 0.0, components: Vec::new(), log_variance: 0.0 });
            cells.push(Vec::new());
        }
        // Every key-value group whose value map is its own.
        let mut values = Vec::new();
        for (&(l, g), found) in &library_sharing::key_values(explanation)? {
            if found.own_value {
                values.push(((l, g), ValueMaps { value: position(found.value)?, heads: found.heads.len(), groups: explanation.layers[l].heads[found.heads[0].0].1.clone() }));
            }
        }
        for &((l, g), ref maps) in &values {
            let choices = values.iter().filter(|((other, _), other_maps)| *other < l && other_maps.heads == maps.heads).count();
            targets.push(Target { kind: Kind::Value { layer: l, group: g }, groups: maps.groups.clone(), choices, zero_logit: 0.0, components: Vec::new(), log_variance: 0.0 });
            cells.push(Vec::new());
        }
        let moments = targets.iter().map(|_| vec![Moment::default(); 2]).collect();
        Ok(Self { targets, width, steps, taken: 0, moments, gates, outputs, ups, key_values, values, embedding, cells })
    }

    /// The maps of key-value group `group` of layer `layer`.
    fn key_value(&self, layer: usize, group: usize) -> Result<&GroupMaps, String> {
        self.key_values.iter().find(|(at, _)| *at == (layer, group)).map(|(_, m)| m).ok_or_else(|| format!("no key-value group {layer}.{group} in the mixture"))
    }

    /// The value maps of key-value group `group` of layer `layer`.
    fn value(&self, layer: usize, group: usize) -> Result<&ValueMaps, String> {
        self.values.iter().find(|(at, _)| *at == (layer, group)).map(|(_, m)| m).ok_or_else(|| format!("no value map of {layer}.{group} in the mixture"))
    }

    /// A value map's coordinates in the explanation.
    fn live_rows(maps: &ValueMaps, posterior: &Posterior) -> Vec<usize> {
        (0..maps.groups.len()).filter(|j| posterior.active[maps.groups[*j]]).collect()
    }

    /// The rows `live` of `m`, each row's entries in turn.
    fn rows_vector(live: &[usize], m: &Array2<f64>) -> Array1<f64> {
        Array1::from_iter(live.iter().flat_map(|&r| m.row(r).to_vec()))
    }

    /// The transpose of [`Self::rows_vector`] as a `live × d` matrix.
    fn rows_matrix(live: &[usize], vector: &Array1<f64>, d: usize) -> Result<Array2<f64>, String> {
        Array2::from_shape_vec((live.len(), d), vector.to_vec()).map_err(error)
    }

    /// Whether one group's query–key maps can stand for another's: the same planes and as many
    /// query heads.
    fn compatible(a: &GroupMaps, b: &GroupMaps) -> bool {
        a.planes == b.planes && a.queries.len() == b.queries.len()
    }

    /// Whether target `t` is in the explanation (any of its groups is).
    fn active(&self, t: usize, posterior: &Posterior) -> bool {
        self.targets[t].groups.iter().any(|g| posterior.active[*g])
    }

    /// A key-value group's planes in the explanation.
    fn live_planes(&self, maps: &GroupMaps, posterior: &Posterior) -> Vec<usize> {
        (0..maps.planes.len()).filter(|p| posterior.active[maps.groups[*p]]).collect()
    }

    /// A key-value group's vector over the rows of the planes `live` of its maps `(q, k)`: each
    /// query map's rows in turn, then its key's, each row's `d` entries in turn.
    fn group_vector(maps: &GroupMaps, live: &[usize], q: &[&Array2<f64>], k: &Array2<f64>) -> Array1<f64> {
        let rows: Vec<usize> = live.iter().flat_map(|p| maps.planes[*p].clone()).collect();
        let mut out = Vec::with_capacity((q.len() + 1) * rows.len() * k.ncols());
        for m in q.iter().copied().chain([k]) {
            for &r in &rows {
                out.extend(m.row(r).iter().copied());
            }
        }
        Array1::from(out)
    }

    /// The transpose of [`Self::group_vector`]: `vector`'s entries added into `(dq, dk)`.
    fn group_scatter(maps: &GroupMaps, live: &[usize], vector: &Array1<f64>, dq: &mut [Array2<f64>], dk: &mut Array2<f64>) {
        let rows: Vec<usize> = live.iter().flat_map(|p| maps.planes[*p].clone()).collect();
        let d = dk.ncols();
        for (side, m) in dq.iter_mut().chain([dk]).enumerate() {
            for (i, &r) in rows.iter().enumerate() {
                let start = (side * rows.len() + i) * d;
                m.row_mut(r).scaled_add(1.0, &vector.slice(ndarray::s![start..start + d]));
            }
        }
    }

    /// The candidate group `source`'s maps brought to the target group `target`'s: per plane the
    /// gauge, and per target query head the source query head it faces (the optimal assignment
    /// of their misfits under the gauge, `library_bodies::hungarian`), alternated twice from the
    /// keys' gauge.
    fn align(target: (&[&Array2<f64>], &Array2<f64>), source: (&[&Array2<f64>], &Array2<f64>), planes: &[Vec<usize>]) -> Result<(Vec<usize>, Vec<(Array2<f64>, f64)>), String> {
        let ((qt, kt), (qs, ks)) = (target, source);
        let mut assignment: Vec<usize> = (0..qt.len()).collect();
        let mut gauge = library_sharing::gauge(&[], kt, &[], ks, planes);
        if qt.len() == 1 {
            return Ok((assignment, library_sharing::gauge(qt, kt, qs, ks, planes)));
        }
        for _ in 0..2 {
            let turned: Vec<Array2<f64>> = qs.iter().map(|q| library_sharing::turn(q, planes, &gauge, true, false)).collect();
            let cost = Array2::from_shape_fn((qt.len(), qs.len()), |(i, j)| (qt[i] - &turned[j]).iter().map(|v| v * v).sum::<f64>());
            assignment = crate::library_bodies::hungarian(&cost)?;
            let facing: Vec<&Array2<f64>> = assignment.iter().map(|&j| qs[j]).collect();
            gauge = library_sharing::gauge(qt, kt, &facing, ks, planes);
        }
        Ok((assignment, gauge))
    }

    /// The vector of an MLP block or a token row at `get`'s values (by trainable index): a gate or
    /// up row, an output column, an embedding row.
    fn vector<'a>(&'a self, write: Write, get: &dyn Fn(usize) -> Result<&'a Array2<f64>, String>) -> Result<ArrayView1<'a, f64>, String> {
        match write {
            Write::Output { layer, function } => Ok(get(self.outputs[layer])?.column(function)),
            Write::Gate { layer, function } => Ok(get(self.gates[layer])?.row(function)),
            Write::Up { layer, function } => Ok(get(self.ups[layer].ok_or("an up row of an ungated MLP")?)?.row(function)),
            Write::Token(token) => Ok(self.embedding.column(token)),
            Write::QueryKey { .. } | Write::Value { .. } => Err("a key-value group is no MLP block".into()),
        }
    }

    /// The trainable index holding an MLP block, and whether the block is a column.
    fn place(&self, write: Write) -> Result<(usize, bool), String> {
        match write {
            Write::Output { layer, .. } => Ok((self.outputs[layer], true)),
            Write::Gate { layer, .. } => Ok((self.gates[layer], false)),
            Write::Up { layer, .. } => Ok((self.ups[layer].ok_or("an up row of an ungated MLP")?, false)),
            Write::Token(_) | Write::QueryKey { .. } | Write::Value { .. } => Err("not a trainable MLP block".into()),
        }
    }

    /// The candidates of an MLP block target of `kind` whose groups are in the explanation: earlier
    /// layers' blocks of its kind, and for a gate earlier outputs and every token row.
    fn candidates(&self, kind: Kind, posterior: &Posterior, explanation: &Explanation) -> Result<Vec<Write>, String> {
        let layer = match kind {
            Kind::Gate { layer, .. } | Kind::Up { layer, .. } | Kind::Output { layer, .. } => layer,
            Kind::QueryKey { .. } | Kind::Value { .. } => return Err("a key-value group is no MLP block".into()),
        };
        let mut out = Vec::new();
        for earlier in 0..layer {
            let units = explanation.layers[earlier].functions.len();
            for function in 0..units {
                // A read's candidates are earlier reads, of either part, and for a gate earlier writes.
                let up = self.ups[earlier].is_some().then_some(Write::Up { layer: earlier, function });
                let writes: Vec<Write> = match kind {
                    Kind::Gate { .. } => [Some(Write::Output { layer: earlier, function }), Some(Write::Gate { layer: earlier, function }), up].into_iter().flatten().collect(),
                    Kind::Up { .. } => [Some(Write::Gate { layer: earlier, function }), up].into_iter().flatten().collect(),
                    Kind::Output { .. } => vec![Write::Output { layer: earlier, function }],
                    _ => Vec::new(),
                };
                for write in writes {
                    let name = write.group().ok_or("an MLP block's group")?;
                    let group = explanation.groups.iter().position(|g| g.name == name).ok_or_else(|| format!("no group {name}"))?;
                    if posterior.active[group] {
                        out.push(write);
                    }
                }
            }
        }
        if matches!(kind, Kind::Gate { .. } | Kind::Up { .. }) {
            out.extend((0..self.embedding.ncols()).map(Write::Token));
        }
        Ok(out)
    }

    /// Target `t`'s new candidates replace its old ones: kept
    /// candidates keep their logit, scale and moments, new ones start at the zero logit.
    fn adopt(&mut self, t: usize, chosen: Vec<Choice>, initial_log_variance: f64) {
        let old = std::mem::take(&mut self.targets[t].components);
        let zero = self.targets[t].zero_logit;
        let mut moments = vec![self.moments[t][0]];
        let mut components = Vec::with_capacity(chosen.len());
        for Choice { write, scale, gauge, assignment, transport } in chosen {
            let (component, moment) = match old.iter().position(|c| c.write == write) {
                Some(at) => (Component { gauge, assignment, transport, ..old[at].clone() }, [self.moments[t][1 + 2 * at], self.moments[t][2 + 2 * at]]),
                None => (Component { write, scale, logit: zero, gauge, assignment, transport }, [Moment::default(); 2]),
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
        // MLP blocks: per layer and kind, the targets' posterior rows against every candidate, `d`
        // candidates at a time.
        let d = self.embedding.nrows();
        let kinds: Vec<Kind> = self.targets.iter().map(|t| t.kind).filter(|k| !matches!(k, Kind::QueryKey { .. } | Kind::Value { .. })).collect();
        let mut seen: Vec<(usize, u8)> = Vec::new();
        for kind in kinds {
            let (layer, part) = match kind {
                Kind::Gate { layer, .. } => (layer, 0u8),
                Kind::Up { layer, .. } => (layer, 1),
                Kind::Output { layer, .. } => (layer, 2),
                Kind::QueryKey { .. } | Kind::Value { .. } => continue,
            };
            if seen.contains(&(layer, part)) {
                continue;
            }
            seen.push((layer, part));
            let same = |k: Kind| match (k, part) {
                (Kind::Gate { layer: l, .. }, 0) | (Kind::Up { layer: l, .. }, 1) | (Kind::Output { layer: l, .. }, 2) => l == layer,
                _ => false,
            };
            let rows: Vec<usize> = (0..self.targets.len()).filter(|&t| same(self.targets[t].kind) && self.active(t, posterior)).collect();
            if rows.is_empty() {
                continue;
            }
            let (operator, column) = self.place(Write::of(kind))?;
            let orient = |m: &Array2<f64>| if column { m.t().to_owned() } else { m.clone() };
            let (mu, log_sd) = (orient(&posterior.mean[operator]), orient(&posterior.log_sd[operator]));
            let precision = log_sd.mapv(|s| (-2.0 * s).exp());
            let weighted = &mu * &precision;
            let writes = self.candidates(kind, posterior, explanation)?;
            let means = |i: usize| -> Result<&Array2<f64>, String> { posterior.mean.get(i).ok_or_else(|| "a posterior mean".to_string()) };
            // Per target its best candidates so far: (−(Σ μ u/σ²)² / Σ u²/σ², scale, candidate).
            let mut best: Vec<Vec<(f64, f64, usize)>> = vec![Vec::new(); rows.len()];
            let mut start = 0;
            while start < writes.len() {
                let end = (start + d).min(writes.len());
                let mut block = Array2::zeros((d, end - start));
                for k in start..end {
                    block.column_mut(k - start).assign(&self.vector(writes[k], &means)?);
                }
                let (cross, norm) = (fast_ab(&weighted, &block), fast_ab(&precision, &block.mapv(|v| v * v)));
                for (slot, &t) in rows.iter().enumerate() {
                    let i = match self.targets[t].kind {
                        Kind::Gate { function, .. } | Kind::Up { function, .. } | Kind::Output { function, .. } => function,
                        Kind::QueryKey { .. } | Kind::Value { .. } => continue,
                    };
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
                let i = match self.targets[t].kind {
                    Kind::Gate { function, .. } | Kind::Up { function, .. } | Kind::Output { function, .. } => function,
                    Kind::QueryKey { .. } | Kind::Value { .. } => continue,
                };
                // A new target's variance starts at its block's mean posterior variance.
                let initial = log_sd.row(i).mapv(|s| (2.0 * s).exp()).mean().ok_or("an empty block")?.ln();
                let chosen = best[slot].iter().map(|&(_, scale, k)| Choice { write: writes[k], scale, gauge: Vec::new(), assignment: Vec::new(), transport: None }).collect();
                self.adopt(t, chosen, initial);
            }
        }
        // Earlier key-value groups are the parents in a fixed layer order. This preserves every
        // possible cross-layer pair while making the product of conditional mixtures a joint
        // density.
        for t in 0..self.targets.len() {
            let Kind::QueryKey { layer, group } = self.targets[t].kind else { continue };
            if !self.active(t, posterior) {
                continue;
            }
            let maps = self.key_value(layer, group)?;
            let live = self.live_planes(maps, posterior);
            let means = |m: &GroupMaps| -> (Vec<&Array2<f64>>, &Array2<f64>) { (m.queries.iter().map(|&q| &posterior.mean[q]).collect(), &posterior.mean[m.key]) };
            let (q1, k1) = means(maps);
            let mu = Self::group_vector(maps, &live, &q1, k1);
            let sd = Self::group_vector(maps, &live, &maps.queries.iter().map(|&q| &posterior.log_sd[q]).collect::<Vec<_>>(), &posterior.log_sd[maps.key]);
            let precision = sd.mapv(|s| (-2.0 * s).exp());
            let mut scored = Vec::new();
            for &((l, g), ref other) in &self.key_values {
                if l >= layer || !Self::compatible(maps, other) {
                    continue;
                }
                let (q, k) = means(other);
                let (assignment, gauge) = Self::align((&q1, k1), (&q, k), &maps.planes)?;
                let turned: Vec<Array2<f64>> = assignment.iter().map(|&j| library_sharing::turn(q[j], &maps.planes, &gauge, true, false)).collect();
                let u = Self::group_vector(maps, &live, &turned.iter().collect::<Vec<_>>(), &library_sharing::turn(k, &maps.planes, &gauge, false, false));
                let (cross, norm) = ((&mu * &precision).dot(&u), (&u * &u * &precision).sum());
                if norm > 0.0 {
                    scored.push((-cross * cross / norm, Choice { write: Write::QueryKey { layer: l, group: g }, scale: cross / norm, gauge, assignment, transport: None }));
                }
            }
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            scored.truncate(self.width);
            let initial = sd.mapv(|s| (2.0 * s).exp()).mean().ok_or("an empty key-value group")?.ln();
            self.adopt(t, scored.into_iter().map(|(_, choice)| choice).collect(), initial);
        }
        // Value maps: earlier groups' value maps moved by the transports `M`'s output projections
        // fix, scored over the target's coordinates in the explanation.
        for t in 0..self.targets.len() {
            let Kind::Value { layer, group } = self.targets[t].kind else { continue };
            if !self.active(t, posterior) {
                continue;
            }
            let maps = self.value(layer, group)?;
            let live = Self::live_rows(maps, posterior);
            let mu = Self::rows_vector(&live, &posterior.mean[maps.value]);
            let sd = Self::rows_vector(&live, &posterior.log_sd[maps.value]);
            let precision = sd.mapv(|s| (-2.0 * s).exp());
            let sources: Vec<((usize, usize), usize)> = self.values.iter().filter(|((l, _), m)| *l < layer && m.heads == maps.heads).map(|(at, m)| (*at, m.value)).collect();
            let found = library_sharing::transports(explanation, (layer, group), &sources.iter().map(|s| s.0).collect::<Vec<_>>())?;
            let mut scored = Vec::new();
            for (((l, g), value), transport) in sources.into_iter().zip(found) {
                let u = Self::rows_vector(&live, &transport.matrix.dot(&posterior.mean[value]));
                let (cross, norm) = ((&mu * &precision).dot(&u), (&u * &u * &precision).sum());
                if norm > 0.0 {
                    scored.push((-cross * cross / norm, Choice { write: Write::Value { layer: l, group: g }, scale: cross / norm, gauge: Vec::new(), assignment: transport.assignment, transport: Some(transport.matrix) }));
                }
            }
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            scored.truncate(self.width);
            let initial = sd.mapv(|s| (2.0 * s).exp()).mean().ok_or("an empty value map")?.ln();
            self.adopt(t, scored.into_iter().map(|(_, choice)| choice).collect(), initial);
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
    /// an earlier output becomes a scale times the gate (`library_sharing::tie`), a gate or up
    /// direction a scale times an earlier read or a token's embedding row
    /// (`library_sharing::tie_row`), an output vector a scale times an earlier
    /// one (`library_sharing::tie_column`), each scale starting at the least-squares fit of the
    /// vector it replaces at `explanation`'s values, and two heads one shared query–key function
    /// (`library_sharing::share_query_key`). Each choice costs `ln n` nats. An MLP block that two
    /// choices involve, as target or source, is taken by the first.
    pub fn harden(&self, explanation: &Explanation, posterior: &Posterior) -> Result<Explanation, String> {
        let program = &explanation.artifact.program;
        let values = |name: String| -> Result<Array2<f64>, String> { Ok(program.operators[operator_index(program, &name)?].matrix()) };
        let mut ties: Vec<Tie> = Vec::new();
        let mut tokens = Vec::new();
        let mut rows: Vec<(&'static str, (usize, usize), (usize, usize, &'static str), f64)> = Vec::new();
        let mut columns: Vec<((usize, usize), (usize, usize), f64)> = Vec::new();
        let mut pairs: Vec<[library_sharing::Member; 2]> = Vec::new();
        let mut shared_values: Vec<((usize, usize), (usize, usize), f64)> = Vec::new();
        // An MLP block takes part in one exact sharing per hardening: as a target or as a source.
        let mut taken: Vec<Write> = Vec::new();
        let mut nats = 0.0;
        for (t, j) in self.dominant(posterior)? {
            let target = &self.targets[t];
            let (own, source) = (Write::of(target.kind), target.components[j].write);
            if taken.contains(&own) || taken.contains(&source) {
                continue;
            }
            let least_squares = |g: &Array1<f64>, s: &Array1<f64>| if s.dot(s) > 0.0 { Some(g.dot(s) / s.dot(s)) } else { None };
            let block = |write: Write| -> Result<Array1<f64>, String> {
                match write {
                    Write::Gate { layer, function } => Ok(values(format!("library.l{layer}.mlp.gate"))?.row(function).to_owned()),
                    Write::Up { layer, function } => Ok(values(format!("library.l{layer}.mlp.up"))?.row(function).to_owned()),
                    Write::Output { layer, function } => Ok(values(format!("library.l{layer}.mlp.out"))?.column(function).to_owned()),
                    _ => Err("not an MLP block".into()),
                }
            };
            match (target.kind, source) {
                (Kind::Gate { layer: gl, function: gi } | Kind::Up { layer: gl, function: gi }, Write::Gate { layer, function } | Write::Up { layer, function }) => {
                    let Some(scale) = least_squares(&block(own)?, &block(source)?) else { continue };
                    let part = |w: Write| if matches!(w, Write::Gate { .. }) { "gate" } else { "up" };
                    rows.push((part(own), (gl, gi), (layer, function, part(source)), scale));
                }
                (Kind::Output { layer: ol, function: oi }, Write::Output { layer, function }) => {
                    let Some(scale) = least_squares(&block(own)?, &block(source)?) else { continue };
                    columns.push(((ol, oi), (layer, function), scale));
                }
                (Kind::Gate { layer: gl, function: gi }, Write::Output { layer, function }) => {
                    let g = values(format!("library.l{gl}.mlp.gate"))?.row(gi).to_owned();
                    let u = values(format!("library.l{layer}.mlp.out"))?.column(function).to_owned();
                    if ties.iter().any(|tie| tie.source == (layer, function)) || g.dot(&g) == 0.0 {
                        continue;
                    }
                    ties.push(Tie { source: (layer, function), target: (gl, gi), scale: u.dot(&g) / g.dot(&g) });
                }
                (Kind::Gate { layer: gl, function: gi } | Kind::Up { layer: gl, function: gi }, Write::Token(token)) => {
                    let part = if matches!(own, Write::Gate { .. }) { "gate" } else { "up" };
                    let g = values(format!("library.l{gl}.mlp.{part}"))?.row(gi).to_owned();
                    let e = self.embedding.column(token);
                    if e.dot(&e) == 0.0 {
                        continue;
                    }
                    tokens.push((part, (gl, gi), token, g.dot(&e) / e.dot(&e)));
                }
                (Kind::QueryKey { layer, group }, Write::QueryKey { layer: other, group: g }) => {
                    // The earlier group owns the shared maps; the target's head `i` reads the
                    // owner's head its assignment names.
                    let width = target.components[j].assignment.len();
                    pairs.push([
                        library_sharing::Member { layer: other, group: g, queries: (0..width).collect() },
                        library_sharing::Member { layer, group, queries: target.components[j].assignment.clone() },
                    ]);
                }
                (Kind::Value { layer, group }, Write::Value { layer: other, group: g }) => {
                    let Some(transport) = &target.components[j].transport else { continue };
                    let flat = |m: Array2<f64>| Array1::from_iter(m.iter().copied());
                    let own_values = flat(values(format!("library.l{layer}.kv{group}.v"))?);
                    let moved = flat(transport.dot(&values(format!("library.l{other}.kv{g}.v"))?));
                    let Some(scale) = least_squares(&own_values, &moved) else { continue };
                    shared_values.push(((layer, group), (other, g), scale));
                }
                _ => return Err("a component's candidate of another kind than its target".into()),
            }
            taken.push(own);
            if !matches!(source, Write::Token(_)) {
                taken.push(source);
            }
            nats += (target.choices as f64).ln();
        }
        let mut out = library_sharing::tie(explanation, &ties)?;
        for (part, target, token, scale) in tokens {
            out = library_sharing::tie_row(&out, part, target, library_sharing::RowSource::Token(token), scale)?;
        }
        for (part, target, (layer, function, from), scale) in rows {
            out = library_sharing::tie_row(&out, part, target, library_sharing::RowSource::Row { layer, part: from, function }, scale)?;
        }
        for (target, source, scale) in columns {
            out = library_sharing::tie_column(&out, target, source, scale)?;
        }
        for pair in pairs {
            out = library_sharing::share_query_key(&out, &pair)?;
        }
        for (target, source, scale) in shared_values {
            out = library_sharing::share_value(&out, target, source, scale)?;
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
            Kind::Gate { .. } | Kind::Up { .. } | Kind::Output { .. } => {
                let sample = |i: usize| -> Result<&Array2<f64>, String> { theta.get(&i).ok_or_else(|| "an operator's sample".to_string()) };
                let g = self.vector(Write::of(target.kind), &sample)?.to_owned();
                let writes = target.components.iter().map(|c| Ok(self.vector(c.write, &sample)?.to_owned())).collect::<Result<Vec<_>, String>>()?;
                let v = Array1::from_elem(g.len(), group_variance(&self.cells[t], posterior));
                Ok((g, writes, v))
            }
            Kind::QueryKey { layer, group } => {
                let maps = self.key_value(layer, group)?;
                let live = self.live_planes(maps, posterior);
                let queries = |m: &GroupMaps| m.queries.iter().map(|&q| get(q)).collect::<Result<Vec<_>, _>>();
                let g = Self::group_vector(maps, &live, &queries(maps)?, get(maps.key)?);
                let writes = target
                    .components
                    .iter()
                    .map(|c| match c.write {
                        Write::QueryKey { layer, group } => {
                            let other = self.key_value(layer, group)?;
                            let q = queries(other)?;
                            let turned: Vec<Array2<f64>> = c.assignment.iter().map(|&j| library_sharing::turn(q[j], &maps.planes, &c.gauge, true, false)).collect();
                            Ok(Self::group_vector(maps, &live, &turned.iter().collect::<Vec<_>>(), &library_sharing::turn(get(other.key)?, &maps.planes, &c.gauge, false, false)))
                        }
                        _ => Err("a write candidate for a key-value group".to_string()),
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let d = posterior.mean[maps.key].ncols();
                let mut v = Vec::with_capacity(g.len());
                for _side in 0..maps.queries.len() + 1 {
                    for &p in &live {
                        let cells: Vec<(usize, Vec<usize>, std::ops::Range<usize>)> = maps.queries.iter().chain([&maps.key]).map(|&i| (i, maps.planes[p].clone(), 0..d)).collect();
                        let variance = group_variance(&cells, posterior);
                        v.extend(std::iter::repeat_n(variance, maps.planes[p].len() * d));
                    }
                }
                Ok((g, writes, Array1::from(v)))
            }
            Kind::Value { layer, group } => {
                let maps = self.value(layer, group)?;
                let live = Self::live_rows(maps, posterior);
                let g = Self::rows_vector(&live, get(maps.value)?);
                let writes = target
                    .components
                    .iter()
                    .map(|c| match (c.write, &c.transport) {
                        (Write::Value { layer, group }, Some(transport)) => Ok(Self::rows_vector(&live, &transport.dot(get(self.value(layer, group)?.value)?))),
                        _ => Err("a value map's candidate without its transport".to_string()),
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let d = posterior.mean[maps.value].ncols();
                let mut v = Vec::with_capacity(g.len());
                for &j in &live {
                    let variance = group_variance(&[(maps.value, vec![j], 0..d)], posterior);
                    v.extend(std::iter::repeat_n(variance, d));
                }
                Ok((g, writes, Array1::from(v)))
            }
        }
    }
}

/// One target's share of a derivative in an operator's sample: a row, a column or the whole.
enum Piece {
    Row { operator: usize, index: usize, values: Array1<f64> },
    Column { operator: usize, index: usize, values: Array1<f64> },
    Whole { operator: usize, values: Array2<f64> },
}

impl Mixture {
    /// Target `t`'s term `−ln r(g)` at `theta`, its derivatives in the operators' samples, and in
    /// its own parameters (the zero logit, then per component its logit and scale, then `ln s²`);
    /// nothing for a target out of the explanation or without candidates.
    fn target_term(&self, t: usize, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>) -> Result<Option<(f64, Vec<Piece>, Vec<f64>)>, String> {
        let target = &self.targets[t];
        if !self.active(t, posterior) || target.components.is_empty() {
            return Ok(None);
        }
        let (g, writes, v) = self.vectors(t, posterior, theta)?;
        let views: Vec<(ArrayView1<'_, f64>, f64)> = writes.iter().zip(&target.components).map(|(u, c)| (u.view(), c.scale)).collect();
        let logits: Vec<f64> = std::iter::once(target.zero_logit).chain(target.components.iter().map(|c| c.logit)).collect();
        let found = term(g.view(), &views, &logits, target.log_variance, v.view())?;
        let mut pieces = Vec::new();
        // A block's derivative goes to its row or its column; a token row is no parameter.
        let block = |write: Write, derivative: &Array1<f64>| -> Result<Option<Piece>, String> {
            let index = match write {
                Write::Token(_) => return Ok(None),
                Write::Output { function, .. } | Write::Gate { function, .. } | Write::Up { function, .. } => function,
                Write::QueryKey { .. } | Write::Value { .. } => return Err("not an MLP block".into()),
            };
            let (operator, column) = self.place(write)?;
            let values = derivative.clone();
            Ok(Some(if column { Piece::Column { operator, index, values } } else { Piece::Row { operator, index, values } }))
        };
        match target.kind {
            Kind::Gate { .. } | Kind::Up { .. } | Kind::Output { .. } => {
                pieces.extend(block(Write::of(target.kind), &found.target)?);
                for (component, derivative) in target.components.iter().zip(&found.writes) {
                    pieces.extend(block(component.write, derivative)?);
                }
            }
            Kind::QueryKey { layer, group } => {
                let maps = self.key_value(layer, group)?;
                let live = self.live_planes(maps, posterior);
                let (query, key) = (theta.get(&maps.queries[0]).ok_or("a sample")?.dim(), theta.get(&maps.key).ok_or("a sample")?.dim());
                let zeros = || -> (Vec<Array2<f64>>, Array2<f64>) { (vec![Array2::zeros(query); maps.queries.len()], Array2::zeros(key)) };
                let (mut dq, mut dk) = zeros();
                Self::group_scatter(maps, &live, &found.target, &mut dq, &mut dk);
                pieces.extend(maps.queries.iter().zip(dq).chain([(&maps.key, dk)]).map(|(&operator, values)| Piece::Whole { operator, values }));
                for (component, derivative) in target.components.iter().zip(&found.writes) {
                    let Write::QueryKey { layer, group } = component.write else { continue };
                    let other = self.key_value(layer, group)?;
                    let (mut aq, mut ak) = zeros();
                    Self::group_scatter(maps, &live, derivative, &mut aq, &mut ak);
                    // The gauge is linear: its transpose takes the derivative back to the
                    // candidate's maps, each target head's to the query head it faces.
                    for (i, &j) in component.assignment.iter().enumerate() {
                        pieces.push(Piece::Whole { operator: other.queries[j], values: library_sharing::turn(&aq[i], &maps.planes, &component.gauge, true, true) });
                    }
                    pieces.push(Piece::Whole { operator: other.key, values: library_sharing::turn(&ak, &maps.planes, &component.gauge, false, true) });
                }
            }
            Kind::Value { layer, group } => {
                let maps = self.value(layer, group)?;
                let live = Self::live_rows(maps, posterior);
                let d = theta.get(&maps.value).ok_or("a sample")?.ncols();
                let own = Self::rows_matrix(&live, &found.target, d)?;
                pieces.extend(live.iter().zip(own.rows()).map(|(&index, row)| Piece::Row { operator: maps.value, index, values: row.to_owned() }));
                // The candidate `T V_s` takes its derivative back through `Tᵀ`.
                for (component, derivative) in target.components.iter().zip(&found.writes) {
                    let (Write::Value { layer, group }, Some(transport)) = (component.write, &component.transport) else { continue };
                    let source = self.value(layer, group)?.value;
                    pieces.push(Piece::Whole { operator: source, values: transport.select(ndarray::Axis(0), &live).t().dot(&Self::rows_matrix(&live, derivative, d)?) });
                }
            }
        }
        let mut own = vec![found.logits[0]];
        for j in 0..target.components.len() {
            own.extend([found.logits[j + 1], found.scales[j]]);
        }
        own.push(found.log_variance);
        Ok(Some((found.value, pieces, own)))
    }
}

impl PriorTerm for Mixture {
    fn operators(&self) -> Vec<usize> {
        // Only the selected conditional factors need per-step host samples. Candidate
        // selection itself sees the complete posterior at the epoch boundary.
        let mut out = Vec::new();
        for (t, target) in self.targets.iter().enumerate().filter(|(_, t)| !t.components.is_empty()) {
            match target.kind {
                Kind::Gate { .. } | Kind::Up { .. } | Kind::Output { .. } => {
                    out.extend(self.place(Write::of(target.kind)).map(|p| p.0));
                    out.extend(self.cells[t].iter().map(|c| c.0)); // includes bias in v_G
                }
                Kind::QueryKey { layer, group } => {
                    if let Ok(maps) = self.key_value(layer, group) { out.extend(maps.queries.iter().chain([&maps.key])); }
                }
                Kind::Value { layer, group } => {
                    if let Ok(maps) = self.value(layer, group) { out.push(maps.value); }
                }
            }
            for candidate in &target.components {
                match candidate.write {
                    Write::Output { .. } | Write::Gate { .. } | Write::Up { .. } => out.extend(self.place(candidate.write).map(|p| p.0)),
                    Write::QueryKey { layer, group } => {
                        if let Ok(maps) = self.key_value(layer, group) { out.extend(maps.queries.iter().chain([&maps.key])); }
                    }
                    Write::Value { layer, group } => {
                        if let Ok(maps) = self.value(layer, group) { out.push(maps.value); }
                    }
                    Write::Token(_) => {} // fixed embedding, not a posterior parameter
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        self.choose(explanation, posterior)
    }

    fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
        use rayon::prelude::*;
        // Each target's term and derivatives on its own (in parallel), then summed in target order.
        let found: Vec<Option<(f64, Vec<Piece>, Vec<f64>)>> = (0..self.targets.len()).into_par_iter().map(|t| self.target_term(t, posterior, theta)).collect::<Result<_, String>>()?;
        let mut value = 0.0;
        let mut gradient: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        let mut learned = vec![Vec::new(); self.targets.len()];
        for (t, found) in found.into_iter().enumerate() {
            let Some((term, pieces, own)) = found else { continue };
            value += term;
            for piece in pieces {
                let i = match &piece {
                    Piece::Row { operator, .. } | Piece::Column { operator, .. } | Piece::Whole { operator, .. } => *operator,
                };
                let into = match gradient.entry(i) {
                    std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::btree_map::Entry::Vacant(entry) => entry.insert(Array2::zeros(theta.get(&i).ok_or("an operator's sample")?.dim())),
                };
                match piece {
                    Piece::Row { index, values, .. } => into.row_mut(index).scaled_add(1.0, &values),
                    Piece::Column { index, values, .. } => into.column_mut(index).scaled_add(1.0, &values),
                    Piece::Whole { values, .. } => *into += &values,
                }
            }
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
                Kind::Gate { .. } | Kind::Up { .. } | Kind::Output { .. } => self.cells[t].iter().map(|(_, rows, cols)| (rows.len() * cols.len()) as f64).sum::<f64>(),
                Kind::QueryKey { layer, group } => {
                    let maps = self.key_value(layer, group)?;
                    let d = posterior.mean[maps.key].ncols();
                    self.live_planes(maps, posterior).iter().map(|p| ((maps.queries.len() + 1) * maps.planes[*p].len() * d) as f64).sum()
                }
                Kind::Value { layer, group } => {
                    let maps = self.value(layer, group)?;
                    (Self::live_rows(maps, posterior).len() * posterior.mean[maps.value].ncols()) as f64
                }
            };
            let n = target.choices as f64;
            // The candidates actually selected (at most `K`, fewer when fewer exist).
            let kept = (target.components.len() as f64).min(k.min(n));
            // ln C(n, K) for the candidates, ½ ln |G| for each of the 2K + 1 values.
            total += ln_gamma(n + 1.0) - ln_gamma(kept + 1.0) - ln_gamma(n - kept + 1.0) + (2.0 * kept + 1.0) * 0.5 * size.ln();
            if matches!(target.kind, Kind::QueryKey { .. }) {
                // A target-dependent alignment is additional information even though its
                // transform preserves the parent head's attention scores.
                let gauge_values = target.components.iter().flat_map(|c| &c.gauge)
                    .map(|(rotation, _)| rotation.len() + 1).sum::<usize>();
                total += gauge_values as f64 * 64.0 * std::f64::consts::LN_2;
                // So is each assignment of the query heads, one of `m!`.
                total += target.components.iter().map(|c| ln_gamma(c.assignment.len() as f64 + 1.0)).sum::<f64>();
            }
        }
        Ok(total)
    }

    fn save(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(error)
    }

    fn load(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let mut restored: Mixture = serde_json::from_value(value.clone()).map_err(error)?;
        if restored.targets.len() != self.targets.len() || restored.gates != self.gates || restored.outputs != self.outputs || restored.ups != self.ups || restored.key_values != self.key_values || restored.values != self.values || restored.cells != self.cells {
            return Err("a checkpoint's mixture of another explanation".into());
        }
        for (target, expected) in restored.targets.iter().zip(&self.targets) {
            if target.kind != expected.kind || target.groups != expected.groups {
                return Err("a checkpoint's mixture targets of another explanation".into());
            }
            // Every candidate is of an earlier layer (a joint density), and a key-value group's is
            // compatible and assigned by a permutation of its query heads.
            for component in &target.components {
                let earlier = match (target.kind, component.write) {
                    (Kind::QueryKey { layer, group }, Write::QueryKey { layer: parent, group: g }) => {
                        let (maps, other) = (self.key_value(layer, group)?, self.key_value(parent, g)?);
                        let mut seen = component.assignment.clone();
                        seen.sort_unstable();
                        parent < layer && Self::compatible(maps, other) && seen == (0..maps.queries.len()).collect::<Vec<_>>()
                    }
                    (Kind::Value { layer, group }, Write::Value { layer: parent, group: g }) => parent < layer && self.value(parent, g)?.heads == self.value(layer, group)?.heads && component.transport.is_some(),
                    (Kind::QueryKey { .. } | Kind::Value { .. }, _) | (_, Write::QueryKey { .. } | Write::Value { .. }) => false,
                    (Kind::Gate { layer, .. } | Kind::Up { layer, .. } | Kind::Output { layer, .. }, write) => match write {
                        Write::Token(_) => matches!(target.kind, Kind::Gate { .. } | Kind::Up { .. }),
                        Write::Output { layer: parent, .. } | Write::Gate { layer: parent, .. } | Write::Up { layer: parent, .. } => parent < layer,
                        Write::QueryKey { .. } | Write::Value { .. } => false,
                    },
                };
                if !earlier {
                    return Err("a checkpoint's mixture must condition only on compatible earlier heads and blocks".into());
                }
            }
            if target.choices != expected.choices || target.components.len() > target.choices {
                return Err("a checkpoint's mixture has an invalid candidate count".into());
            }
        }
        // Validate before moving the embedding: a rejected checkpoint leaves all state intact.
        restored.embedding = std::mem::take(&mut self.embedding);
        *self = restored;
        Ok(())
    }
}

/// Prior terms over disjoint targets as one term of `F` (`library_mdl::fit` takes one): the
/// blocks' mixture and the bodies' (`library_bodies::BodyMixture`), each choosing, sampling and
/// learning on its own; their values, gradients and costs add.
pub struct Priors(pub Vec<Box<dyn PriorTerm>>);

impl PriorTerm for Priors {
    fn operators(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.0.iter().flat_map(|p| p.operators()).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        self.0.iter_mut().try_for_each(|p| p.epoch(explanation, posterior))
    }

    fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
        let mut total = 0.0;
        let mut gradient: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        for prior in &mut self.0 {
            // Each term reads its own operators' samples among `theta`'s.
            let (value, part) = prior.sample(posterior, theta, learn)?;
            total += value;
            for (i, g) in part {
                match gradient.get_mut(&i) {
                    Some(sum) => *sum += &g,
                    None => {
                        gradient.insert(i, g);
                    }
                }
            }
        }
        Ok((total, gradient))
    }

    fn cost(&self, posterior: &Posterior) -> Result<f64, String> {
        self.0.iter().map(|p| p.cost(posterior)).sum()
    }

    fn save(&self) -> Result<serde_json::Value, String> {
        Ok(serde_json::Value::Array(self.0.iter().map(|p| p.save()).collect::<Result<_, _>>()?))
    }

    fn load(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let states = value.as_array().filter(|a| a.len() == self.0.len()).ok_or("a checkpoint's prior terms of another count")?;
        // Every state is checked against a copy first, so a rejected checkpoint changes nothing.
        let saved: Vec<serde_json::Value> = self.0.iter().map(|p| p.save()).collect::<Result<_, _>>()?;
        for (at, (prior, state)) in self.0.iter_mut().zip(states).enumerate() {
            if let Err(e) = prior.load(state) {
                for (prior, before) in self.0.iter_mut().zip(&saved).take(at) {
                    prior.load(before)?;
                }
                return Err(e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_gpu::tensor::posterior_normal;

    fn head_fixture(name: &str) -> (Explanation, Posterior, Mixture) {
        use crate::{import::import_language_model, library_mdl::explanation, run_check::{layer_nodes, split_sites}};

        let dir = crate::test_support::tiny_export(name, 2);
        let imported = import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let explanation = explanation(&native, &layer_nodes(&native, 2).unwrap()).unwrap();
        let mut posterior = Posterior::new(&explanation, 96).unwrap();
        let mixture = Mixture::new(&explanation, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        let (earlier, later) = (mixture.key_value(0, 0).unwrap(), mixture.key_value(1, 0).unwrap());
        // Identical nonzero maps previously admitted reciprocal, unit-scale components.
        for (from, to) in [(earlier.queries[0], later.queries[0]), (earlier.key, later.key)] {
            posterior.mean[to] = posterior.mean[from].clone();
        }
        (explanation, posterior, mixture)
    }

    #[test]
    fn head_candidates_follow_layer_order_and_keep_exact_copies() {
        let (explanation, posterior, mut mixture) = head_fixture("library_mixture_head_order");
        mixture.choose(&explanation, &posterior).unwrap();
        for target in &mixture.targets {
            let Kind::QueryKey { layer, group } = target.kind else { continue };
            let maps = mixture.key_value(layer, group).unwrap();
            let choices = mixture.key_values.iter().filter(|((l, _), m)| *l < layer && Mixture::compatible(maps, m)).count();
            assert_eq!(target.choices, choices);
            if layer == 0 {
                assert!(target.components.is_empty(), "the first layer keeps its Gaussian prior");
            }
            for component in &target.components {
                assert!(matches!(component.write, Write::QueryKey { layer: parent, .. } if parent < layer));
            }
        }
        let copied = mixture.targets.iter().find(|t| t.kind == Kind::QueryKey { layer: 1, group: 0 }).unwrap();
        let source = copied.components.iter().find(|c| c.write == Write::QueryKey { layer: 0, group: 0 }).expect("the cross-layer copy stays available");
        assert!((source.scale - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_cyclic_checkpoint_is_rejected_without_changing_the_mixture() {
        let (explanation, posterior, mut mixture) = head_fixture("library_mixture_head_restore");
        mixture.choose(&explanation, &posterior).unwrap();
        let before = mixture.save().unwrap();
        let embedding = mixture.embedding.clone();
        let mut cyclic = mixture.clone();
        let later = cyclic.targets.iter().find(|t| t.kind == Kind::QueryKey { layer: 1, group: 0 }).unwrap();
        let mut reverse = later.components.iter().find(|c| c.write == Write::QueryKey { layer: 0, group: 0 }).unwrap().clone();
        reverse.write = Write::QueryKey { layer: 1, group: 0 };
        cyclic.targets.iter_mut().find(|t| t.kind == Kind::QueryKey { layer: 0, group: 0 }).unwrap().components.push(reverse);
        assert!(mixture.load(&cyclic.save().unwrap()).unwrap_err().contains("earlier heads"));
        assert_eq!(mixture.save().unwrap(), before);
        assert_eq!(mixture.embedding, embedding);
        assert!(mixture.load(&serde_json::json!({})).is_err());
        assert_eq!(mixture.save().unwrap(), before);
        assert_eq!(mixture.embedding, embedding);
        mixture.load(&before).unwrap();
        assert_eq!(mixture.save().unwrap(), before);
        assert_eq!(mixture.embedding, embedding);
    }

    #[test]
    fn head_cost_counts_the_components_actually_selected() {
        let (explanation, posterior, mut mixture) = head_fixture("library_mixture_head_cost");
        mixture.choose(&explanation, &posterior).unwrap();
        let keep = mixture.targets.iter().position(|t| t.kind == Kind::QueryKey { layer: 1, group: 0 }).unwrap();
        for (t, target) in mixture.targets.iter_mut().enumerate() {
            target.components.truncate(if t == keep { 1 } else { 0 });
        }
        let target = &mixture.targets[keep];
        assert_eq!(target.components.len(), 1);
        let maps = mixture.key_value(1, 0).unwrap();
        let size = 2 * maps.planes.iter().map(Vec::len).sum::<usize>() * posterior.mean[maps.key].ncols();
        let gauge_values = target.components[0].gauge.iter().map(|(rotation, _)| rotation.len() + 1).sum::<usize>();
        assert!(gauge_values > 0, "the target-dependent gauge must be charged");
        let expected = (target.choices as f64).ln() + 1.5 * (size as f64).ln()
            + gauge_values as f64 * 64.0 * std::f64::consts::LN_2;
        assert!((mixture.cost(&posterior).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn prior_samples_only_selected_operators_and_reselects_them() {
        let (explanation, posterior, mut mixture) = head_fixture("library_mixture_selected_operators");
        assert!(mixture.operators().is_empty());
        mixture.choose(&explanation, &posterior).unwrap();
        for target in &mixture.targets {
            let layer = match target.kind {
                Kind::Gate { layer, .. } | Kind::Up { layer, .. } | Kind::Output { layer, .. } | Kind::QueryKey { layer, .. } | Kind::Value { layer, .. } => layer,
            };
            for component in &target.components {
                let parent = match component.write {
                    Write::Output { layer, .. } | Write::Gate { layer, .. } | Write::Up { layer, .. } | Write::QueryKey { layer, .. } | Write::Value { layer, .. } => Some(layer),
                    Write::Token(_) => None,
                };
                assert!(parent.is_none_or(|p| p < layer), "a prior parent is in an earlier layer");
            }
        }
        for target in &mut mixture.targets {
            if target.kind != (Kind::QueryKey { layer: 1, group: 0 }) { target.components.clear(); }
        }
        let selected = mixture.operators();
        assert!(selected.len() <= 6, "one target and at most two parent Q/K pairs");
        let theta = selected.iter().map(|&i| (i, posterior.mean[i].clone())).collect();
        let (value, gradient) = mixture.sample(&posterior, &theta, false).unwrap();
        assert!(value.is_finite());
        assert!(gradient.keys().all(|i| selected.contains(i)));
        mixture.choose(&explanation, &posterior).unwrap();
        let reselected = mixture.operators();
        assert!(reselected.len() > selected.len());
        let theta = reselected.iter().map(|&i| (i, posterior.mean[i].clone())).collect();
        assert!(mixture.sample(&posterior, &theta, false).unwrap().0.is_finite());
    }

    fn gaussian_log_density(x: ArrayView1<'_, f64>, mean: ArrayView1<'_, f64>, variance: f64) -> f64 {
        let r = &x - &mean;
        -0.5 * x.len() as f64 * (2.0 * PI * variance).ln() - r.dot(&r) / (2.0 * variance)
    }

    #[test]
    fn cyclic_conditionals_do_not_define_the_claimed_joint_density() {
        // Two half-weight mixtures with mutual scales a=b=1/2 have
        // Z = 1 - 1/4 + (1/4)/|1-ab| = 13/12, not one. Integrate the actual
        // implemented correction against the two base Gaussian densities.
        // A single directed conditional, with the other's base Gaussian, has Z=1.
        let (mut directed, mut cyclic) = (0.0, 0.0);
        let variance = ndarray::array![1.0];
        for ix in -120..=120 {
            let x = ndarray::array![ix as f64 * 0.1];
            for iy in -120..=120 {
                let y = ndarray::array![iy as f64 * 0.1];
                let xy = term(x.view(), &[(y.view(), 0.5)], &[0.0, 0.0], 0.0, variance.view()).unwrap().value;
                let yx = term(y.view(), &[(x.view(), 0.5)], &[0.0, 0.0], 0.0, variance.view()).unwrap().value;
                let base = -(2.0 * PI).ln() - 0.5 * (x[0] * x[0] + y[0] * y[0]);
                directed += (base - yx).exp() * 0.01;
                cyclic += (base - xy - yx).exp() * 0.01;
            }
        }
        assert!((directed - 1.0).abs() < 1e-9, "{directed}");
        assert!((cyclic - 13.0 / 12.0).abs() < 1e-9, "{cyclic}");
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

    /// The weight sample of `key` of `posterior`'s operators the mixture reads.
    fn draw(mixture: &Mixture, posterior: &Posterior, key: u64) -> BTreeMap<usize, Array2<f64>> {
        mixture
            .operators()
            .into_iter()
            .map(|i| {
                let (mean, log_sd) = (&posterior.mean[i], &posterior.log_sd[i]);
                let cols = mean.ncols();
                (i, Array2::from_shape_fn(mean.dim(), |(r, c)| mean[[r, c]] + log_sd[[r, c]].exp() * f64::from(gam_gpu::tensor::posterior_normal(key, i as u64, (r * cols + c) as u64))))
            })
            .collect()
    }

    /// `steps` steps of the mixture's own parameters on samples of `posterior`.
    fn learn(mixture: &mut Mixture, posterior: &Posterior, steps: u64) {
        for key in 0..steps {
            let theta = draw(mixture, posterior, key);
            mixture.sample(posterior, &theta, true).unwrap();
        }
    }

    /// A tiny grouped-query explanation whose layer-1 key-value group is layer 0's with its two
    /// query heads swapped.
    fn swapped_group(name: &str) -> (crate::import::Imported, Explanation) {
        use crate::{library_mdl::explanation, operator_program::{Operator, Provenance, exact_precision}, run_check::{layer_nodes, split_sites}};
        let imported = library_sharing::grouped(name);
        let native = split_sites(&imported.program).unwrap();
        let mut start = explanation(&native, &layer_nodes(&native, 2).unwrap()).unwrap();
        let found = library_sharing::key_values(&start).unwrap();
        let (first, second) = (&found[&(0, 0)], &found[&(1, 0)]);
        let copies = [(first.heads[1].1.query, second.heads[0].1.query), (first.heads[0].1.query, second.heads[1].1.query), (first.key, second.key)];
        let program = &mut start.artifact.program;
        for (from, to) in copies {
            let values = program.operators[from].matrix();
            let precision = exact_precision(values.iter().copied()).unwrap();
            let source = &program.operators[to];
            program.operators[to] = std::sync::Arc::new(Operator::dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, precision, Provenance::default()).unwrap());
        }
        (imported, start)
    }

    #[test]
    fn a_key_value_group_copied_with_its_query_heads_swapped_is_found_assigned_and_shared() {
        let (imported, start) = swapped_group("library_mixture_grouped");
        let posterior = Posterior::new(&start, 96).unwrap();
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        mixture.epoch(&start, &posterior).unwrap();
        let t = mixture.targets.iter().position(|t| t.kind == Kind::QueryKey { layer: 1, group: 0 }).unwrap();
        let copy = mixture.targets[t].components.iter().position(|c| c.write == Write::QueryKey { layer: 0, group: 0 }).expect("the copy is a candidate");
        assert_eq!(mixture.targets[t].components[copy].assignment, vec![1, 0], "each query head faces the one it copies");
        learn(&mut mixture, &posterior, 300);
        let weights = mixture.targets[t].weights().unwrap();
        assert!(weights[1 + copy] > 0.5, "the copy's weight dominates its mixture: {weights:?}");
        assert!(mixture.dominant(&posterior).unwrap().contains(&(t, copy)));
        let hardened = mixture.harden(&start, &posterior).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), hardened.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[hardened.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the shared group keeps the outputs");
        library_sharing::same_native_blocks(&start, &hardened);
        let groups = library_sharing::key_values(&hardened).unwrap();
        assert!(!groups[&(1, 0)].own_key, "layer 1's query heads read layer 0's maps");
        assert!(hardened.fixed_nats >= (mixture.targets[t].choices as f64).ln() - 1e-12, "the choice is paid for");
    }

    #[test]
    fn a_key_value_group_term_is_differentiated_through_its_gauge_and_assignment() {
        let (_, start) = swapped_group("library_mixture_grouped_gradient");
        let posterior = Posterior::new(&start, 96).unwrap();
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        mixture.epoch(&start, &posterior).unwrap();
        // The layer-1 group's term alone, against both candidates with their gauges and assignments.
        let t = mixture.targets.iter().position(|t| t.kind == Kind::QueryKey { layer: 1, group: 0 }).unwrap();
        for (u, target) in mixture.targets.iter_mut().enumerate() {
            if u != t {
                target.components.clear();
            }
        }
        assert!(mixture.targets[t].components.iter().any(|c| c.assignment == vec![1, 0]));
        let theta = draw(&mixture, &posterior, 7);
        let (_, gradient) = mixture.sample(&posterior, &theta, false).unwrap();
        let (target, source) = (mixture.key_value(1, 0).unwrap().clone(), mixture.key_value(0, 0).unwrap().clone());
        let h = 1e-6;
        for (i, entry) in [(target.queries[0], (0, 1)), (target.queries[1], (3, 2)), (target.key, (1, 0)), (source.queries[0], (2, 5)), (source.queries[1], (0, 0)), (source.key, (3, 7))] {
            let mut at = |e: f64| {
                let mut moved = theta.clone();
                moved.get_mut(&i).unwrap()[entry] += e;
                mixture.sample(&posterior, &moved, false).unwrap().0
            };
            let central = (at(h) - at(-h)) / (2.0 * h);
            let found = gradient[&i][entry];
            assert!((found - central).abs() <= 1e-6 * (1.0 + central.abs()), "operator {i} {entry:?}: {found} against {central}");
        }
    }

    /// The tiny grouped-query explanation whose layer-1 value map is `0.7 T V` of layer 0's.
    fn moved_value(name: &str) -> (crate::import::Imported, Explanation) {
        use crate::{library_mdl::explanation, operator_program::{Operator, Provenance, exact_precision}, run_check::{layer_nodes, split_sites}};
        let imported = library_sharing::grouped(name);
        let native = split_sites(&imported.program).unwrap();
        let mut start = explanation(&native, &layer_nodes(&native, 2).unwrap()).unwrap();
        let transport = library_sharing::transports(&start, (1, 0), &[(0, 0)]).unwrap().pop().unwrap();
        let found = library_sharing::key_values(&start).unwrap();
        let (to, from) = (found[&(1, 0)].value, found[&(0, 0)].value);
        let program = &mut start.artifact.program;
        let values = transport.matrix.dot(&program.operators[from].matrix()) * 0.7;
        let precision = exact_precision(values.iter().copied()).unwrap();
        let source = &program.operators[to];
        program.operators[to] = std::sync::Arc::new(Operator::dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, precision, Provenance::default()).unwrap());
        (imported, start)
    }

    #[test]
    fn a_value_map_moved_through_the_output_projections_is_found_and_shared() {
        let (imported, start) = moved_value("library_mixture_value");
        let posterior = Posterior::new(&start, 96).unwrap();
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        mixture.epoch(&start, &posterior).unwrap();
        let t = mixture.targets.iter().position(|t| t.kind == Kind::Value { layer: 1, group: 0 }).unwrap();
        let copy = mixture.targets[t].components.iter().position(|c| c.write == Write::Value { layer: 0, group: 0 }).expect("the moved value map is a candidate");
        assert!((mixture.targets[t].components[copy].scale - 0.7).abs() < 1e-9, "its least-squares scale is the planted one");
        learn(&mut mixture, &posterior, 300);
        let weights = mixture.targets[t].weights().unwrap();
        assert!(weights[1 + copy] > 0.5, "the moved map's weight dominates its mixture: {weights:?}");
        assert!(mixture.dominant(&posterior).unwrap().contains(&(t, copy)));
        let hardened = mixture.harden(&start, &posterior).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), hardened.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[hardened.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the shared value map keeps the outputs");
        library_sharing::same_native_blocks(&start, &hardened);
        assert!(!library_sharing::key_values(&hardened).unwrap()[&(1, 0)].own_value, "layer 1 reads layer 0's value map");
    }

    #[test]
    fn a_value_term_is_differentiated_through_its_transport() {
        let (_, start) = moved_value("library_mixture_value_gradient");
        let posterior = Posterior::new(&start, 96).unwrap();
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        mixture.epoch(&start, &posterior).unwrap();
        let t = mixture.targets.iter().position(|t| t.kind == Kind::Value { layer: 1, group: 0 }).unwrap();
        for (u, target) in mixture.targets.iter_mut().enumerate() {
            if u != t {
                target.components.clear();
            }
        }
        let theta = draw(&mixture, &posterior, 11);
        let (_, gradient) = mixture.sample(&posterior, &theta, false).unwrap();
        let (target, source) = (mixture.value(1, 0).unwrap().value, mixture.value(0, 0).unwrap().value);
        let h = 1e-6;
        for (i, entry) in [(target, (0, 1)), (target, (3, 6)), (source, (1, 2)), (source, (2, 7))] {
            let mut at = |e: f64| {
                let mut moved = theta.clone();
                moved.get_mut(&i).unwrap()[entry] += e;
                mixture.sample(&posterior, &moved, false).unwrap().0
            };
            let central = (at(h) - at(-h)) / (2.0 * h);
            let found = gradient[&i][entry];
            assert!((found - central).abs() <= 1e-6 * (1.0 + central.abs()), "operator {i} {entry:?}: {found} against {central}");
        }
    }

    #[test]
    fn prior_terms_add_their_values_gradients_and_costs() {
        let (explanation, posterior, mut first) = head_fixture("library_mixture_priors");
        first.choose(&explanation, &posterior).unwrap();
        let mut second = Mixture::new(&explanation, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        second.choose(&explanation, &posterior).unwrap();
        // Two terms over one explanation: the second keeps only its value-map targets.
        for target in &mut second.targets {
            if !matches!(target.kind, Kind::Value { .. }) {
                target.components.clear();
            }
        }
        for target in &mut first.targets {
            if matches!(target.kind, Kind::Value { .. }) {
                target.components.clear();
            }
        }
        let mut joint = Priors(vec![Box::new(first.clone()), Box::new(second.clone())]);
        let theta = draw(&first, &posterior, 5).into_iter().chain(draw(&second, &posterior, 5)).collect::<BTreeMap<_, _>>();
        let own = |m: &Mixture| m.operators().into_iter().map(|i| (i, theta[&i].clone())).collect::<BTreeMap<_, _>>();
        let ((a, ga), (b, gb)) = (first.sample(&posterior, &own(&first), false).unwrap(), second.sample(&posterior, &own(&second), false).unwrap());
        let (value, gradient) = joint.sample(&posterior, &theta, false).unwrap();
        assert!((value - (a + b)).abs() <= 1e-12 * (1.0 + value.abs()));
        for (i, g) in &gradient {
            let expected = match (ga.get(i), gb.get(i)) {
                (Some(x), Some(y)) => x + y,
                (Some(x), None) | (None, Some(x)) => x.clone(),
                (None, None) => panic!("a gradient of no term"),
            };
            assert!(g.iter().zip(expected.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + y.abs())));
        }
        assert!((joint.cost(&posterior).unwrap() - first.cost(&posterior).unwrap() - second.cost(&posterior).unwrap()).abs() < 1e-9);
        let saved = joint.save().unwrap();
        assert!(joint.load(&serde_json::json!([saved[0].clone()])).is_err(), "a checkpoint of another count is refused");
        joint.load(&saved).unwrap();
        assert_eq!(joint.save().unwrap(), saved);
    }

    #[test]
    fn an_up_direction_copying_an_earlier_gate_is_found_and_tied() {
        use crate::{library_mdl::explanation, operator_program::{Operator, Provenance, exact_precision}, run_check::{layer_nodes, split_sites}};
        let imported = library_sharing::gated("library_mixture_gated");
        let native = split_sites(&imported.program).unwrap();
        let mut start = explanation(&native, &layer_nodes(&native, 2).unwrap()).unwrap();
        // Function 6 of layer 1 reads up 1.5 times what function 2 of layer 0 reads as its gate.
        let program = &mut start.artifact.program;
        let (gate, up) = (operator_index(program, "library.l0.mlp.gate").unwrap(), operator_index(program, "library.l1.mlp.up").unwrap());
        let mut values = program.operators[up].matrix();
        values.row_mut(6).assign(&(&program.operators[gate].matrix().row(2) * 1.5));
        let precision = exact_precision(values.iter().copied()).unwrap();
        let source = &program.operators[up];
        program.operators[up] = std::sync::Arc::new(Operator::dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values, precision, Provenance::default()).unwrap());
        let posterior = Posterior::new(&start, 96).unwrap();
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        mixture.epoch(&start, &posterior).unwrap();
        let t = mixture.targets.iter().position(|t| t.kind == Kind::Up { layer: 1, function: 6 }).unwrap();
        let copy = mixture.targets[t].components.iter().position(|c| c.write == Write::Gate { layer: 0, function: 2 }).expect("the earlier gate is a candidate");
        assert!((mixture.targets[t].components[copy].scale - 1.5).abs() < 1e-9);
        learn(&mut mixture, &posterior, 300);
        let weights = mixture.targets[t].weights().unwrap();
        assert!(weights[1 + copy] > 0.5, "the copy's weight dominates its mixture: {weights:?}");
        let dominant = mixture.dominant(&posterior).unwrap();
        assert_eq!(dominant, vec![(t, copy)], "only the copy dominates: {:?}", dominant.iter().map(|(t, j)| (mixture.targets[*t].kind, mixture.targets[*t].components[*j].write)).collect::<Vec<_>>());
        let hardened = mixture.harden(&start, &posterior).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), hardened.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[hardened.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the tied up direction keeps the outputs");
        library_sharing::same_native_blocks(&start, &hardened);
        let own = hardened.groups.iter().position(|g| g.name == "library.l1.mlp.f6.up").unwrap();
        assert!(hardened.removed.contains(&own), "the up direction is stored once");
    }

    #[test]
    fn exact_copies_of_ties_functions_and_heads_are_found_by_their_weights_and_made_exact() {
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
        // Function 7 of layer 1 is function 4 of layer 0: its gate 2 times, its output half.
        let earlier_gate = operator_index(program, "library.l0.mlp.gate").unwrap();
        let mut values = program.operators[gate].matrix();
        values.row_mut(7).assign(&(&program.operators[earlier_gate].matrix().row(4) * 2.0));
        replace(program, gate, values);
        let later_out = operator_index(program, "library.l1.mlp.out").unwrap();
        let mut values = program.operators[later_out].matrix();
        values.column_mut(7).assign(&(&program.operators[out].matrix().column(4) * 0.5));
        replace(program, later_out, values);
        // Head 0 of layer 1 attends as head 0 of layer 0 does.
        let found = library_sharing::key_values(&start).unwrap();
        let (first, second) = (&found[&(0, 0)], &found[&(1, 0)]);
        let program = &mut start.artifact.program;
        for (from, to) in [(first.heads[0].1.query, second.heads[0].1.query), (first.key, second.key)] {
            let values = program.operators[from].matrix();
            replace(program, to, values);
        }
        let settings = Settings { batch_sequences: 2, rate: 0.1, beta1: 0.9, seed: 3, numeric_bytes: 1 << 26, head_tile_rows: 64 };
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let (train, held) = sequences.split_at(4);
        let fitted = fit(&Device::host(), &native, &start, train, held, &settings, "tiny", None, Some(&mut mixture)).unwrap();
        assert!(fitted.report.start.prior_bits.is_finite() && fitted.report.start.prior_bits != 0.0, "F holds the mixture's term");
        // The fit removes the copies this random model does not need; the weights are learned at the
        // start, every group in, from samples of its posterior.
        let all_in = crate::library_mdl::Posterior::new(&start, 96).unwrap();
        let mut learned = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        learned.epoch(&start, &all_in).unwrap();
        learn(&mut learned, &all_in, 300);
        // The learned weights pick each copy among the candidates.
        let found = |kind: Kind, write: Write| -> (usize, usize) {
            let target = learned.targets.iter().position(|t| t.kind == kind).unwrap();
            let weights = learned.targets[target].weights().unwrap();
            let copy = learned.targets[target].components.iter().position(|c| c.write == write).expect("the copy is a candidate");
            assert!(weights[1 + copy] > 0.5, "the copy's weight dominates its mixture: {weights:?}");
            (target, copy)
        };
        let (target, copy) = found(Kind::Gate { layer: 1, function: 5 }, Write::Output { layer: 0, function: 3 });
        let head_copy = found(Kind::QueryKey { layer: 1, group: 0 }, Write::QueryKey { layer: 0, group: 0 });
        let gate_copy = found(Kind::Gate { layer: 1, function: 7 }, Write::Gate { layer: 0, function: 4 });
        let output_copy = found(Kind::Output { layer: 1, function: 7 }, Write::Output { layer: 0, function: 4 });
        // Made exact, the tie keeps the outputs and pays for its choice.
        let dominant = learned.dominant(&all_in).unwrap();
        assert!([(target, copy), head_copy, gate_copy, output_copy].iter().all(|c| dominant.contains(c)));
        let hardened = learned.harden(&start, &all_in).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), hardened.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[hardened.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the exact tie keeps the outputs");
        library_sharing::same_native_blocks(&start, &hardened);
        let choices: f64 = [target, head_copy.0, gate_copy.0, output_copy.0].iter().map(|t| (learned.targets[*t].choices as f64).ln()).sum();
        assert!(hardened.fixed_nats >= choices - 1e-9, "every exact choice is paid for");
        // The copied function's blocks are stored once: its own gate and output leave the explanation.
        for name in ["library.l1.mlp.f7.gate", "library.l1.mlp.f7.out"] {
            let g = hardened.groups.iter().position(|g| g.name == name).unwrap();
            assert!(hardened.removed.contains(&g), "{name} is not charged");
        }
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
