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
//! brought to the target's gauge plane by plane (`library_sharing::gauge`, a rotation and, for heads
//! without query and key norms, a scale, leaving every one of its heads' scores unchanged:
//! `library_sharing::Symmetry`) with each target query head facing one of
//! the candidate's (the optimal assignment of their misfits), gauge and assignment fixed for the
//! epoch. A value map's are the value maps `V_s` of earlier groups with as many query heads, each
//! moved by the transport `T` that `M`'s output projections fix (`library_sharing::transports`):
//! `M` keeps the projections `O`, so a group's value–output maps `O_i V` are those of the
//! candidate when `V = T V_s`: the symmetry `V → R V`, `O → O R⁻¹` read through the projections
//! where the transport's residual is zero, an approximation otherwise; `T` is sent with `M` and
//! costs nothing. The weights `π = softmax(z)` of the logits `z` are learned; each epoch sets the
//! scales `c_j` (the posterior-weighted least-squares fit of the target's means by the
//! candidate's) and the variance `s²` (the best candidate's expected residual per entry, below)
//! at the posterior, as `library_mdl` sets `v`: a step of a learning rate in `c` would move an
//! exact copy far beyond its posterior deviations. As the groups' own
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
//! In a fit the blocks' term runs on the device (`PriorTerm::sample_device`): the weight sample of
//! every block a term reads is drawn into one table of rows (an output vector turned to a row),
//! the targets' and candidates' rows gathered, the scaled residuals `r_j = g − c_j u_j` and their
//! sums of squares `‖r_j‖²`, `u_j · r_j` and `‖g‖²` formed there, and only these per-target scalars
//! come to the host, which forms the term ([`scalar_term`]) and each row's coefficients; the
//! gradient, `α g + Σ_j β_j r_j` in a target and `γ_j r_j` in a candidate, is summed per row on
//! the device. Key-value groups' terms, few and gauged, are formed on the host.
//!
//! The sample gradient treats `v` and the epoch's gauges as fixed hyperparameters. Recomputing
//! `v` from the posterior is an adaptive update, not the optimum of the mixture objective:
//! the Gaussian-only empirical-Bayes identity does not cancel the mixture's derivative through
//! `v`. This path does not yet implement that total posterior derivative.
//!
//! # Candidates
//!
//! Each epoch the candidates are re-chosen per target by the posterior-normalized misfit of the
//! best scaled candidate, `m(u) = min_c Σ_k (μ_k − c u_k)² / σ_k²` over the target's posterior
//! means `μ` and deviations `σ`: the `K` of smallest misfit, of which the target keeps those that
//! pay for themselves in `F`. A candidate alone, taking the whole weight at the best
//! `s² = E_q‖g − c u‖² / D`, saves the divergence `Σ_k [½ ln(v_k / s²) − ½ + E_q[g_k²] / (2 v_k)]`;
//! in decreasing saving each is kept while its saving exceeds what it adds to the mixture's own
//! code length below. A target keeping none has no mixture, costs nothing and is not sampled. A
//! candidate kept keeps its logit; a new one starts at the zero component's.
//!
//! # Hardening
//!
//! When one candidate holds more than half of its target's mixture weight, and making it exact is
//! predicted to lower `F` (the target's groups' costs exceed half the misfit of the equality in
//! the posterior's deviations, the data term's rise at its curvature, plus the choice's `ln n`),
//! the sharing is made exact: a gate reading an earlier write becomes a tie (`library_sharing::tie`), a gate or up
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
    device_posterior::DevicePosterior,
    library_mdl::{Explanation, Posterior, PriorTerm},
    library_sharing::{self, Tie},
    operator_program::OperatorProgram,
};
use gam_gpu::tensor::{Arithmetic, Device, Indices, Op, Storage, Tensor};
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

/// [`term`] of a target whose entries share one variance `v`, from its sample's sums of squares
/// alone: `‖g‖²`, and per component `‖r_j‖²` and `u_j · r_j` of its scaled residual
/// `r_j = g − c_j u_j`, in `d` coordinates. Its derivative in `g` is `target · g + Σ_j residuals_j r_j`,
/// in `u_j` `writes_j r_j`.
#[derive(Clone, Debug, PartialEq)]
pub struct Scalars {
    pub value: f64,
    pub target: f64,
    pub residuals: Vec<f64>,
    pub writes: Vec<f64>,
    pub scales: Vec<f64>,
    pub logits: Vec<f64>,
    pub log_variance: f64,
}

/// [`Scalars`] of a target sample with `‖g‖² = gg`, `‖r_j‖² = rr_j` and `u_j · r_j = ur_j` against
/// the components' scales `c_j`, the logits `z` (the zero component's first), `ln s²` and the
/// groups' variance `v`, in `d` coordinates ([`term`] with `v` the same for every entry).
pub fn scalar_term(gg: f64, (rr, ur): (&[f64], &[f64]), scales: &[f64], logits: &[f64], log_variance: f64, v: f64, d: usize) -> Result<Scalars, String> {
    if logits.len() != rr.len() + 1 || ur.len() != rr.len() || scales.len() != rr.len() || !(v > 0.0) || !log_variance.is_finite() {
        return Err("a mixture term needs one logit per component and the zero's, and a positive variance".into());
    }
    let d = d as f64;
    let s2 = log_variance.exp();
    let pi = log_softmax(logits).map_err(error)?;
    let zero = -0.5 * d * (2.0 * PI * v).ln() - gg / (2.0 * v);
    let mut ell = vec![pi[0]];
    for (j, r) in rr.iter().enumerate() {
        ell.push(pi[j + 1] - 0.5 * d * (2.0 * PI * s2).ln() - r / (2.0 * s2) - zero);
    }
    let value = -log_sum_exp(&ell).map_err(error)?;
    let w: Vec<f64> = log_softmax(&ell).map_err(error)?.into_iter().map(f64::exp).collect();
    let mut target = 0.0;
    let (mut residuals, mut writes, mut scale_derivatives) = (Vec::with_capacity(rr.len()), Vec::with_capacity(rr.len()), Vec::with_capacity(rr.len()));
    let mut log_variance_derivative = 0.0;
    for j in 0..rr.len() {
        let wj = w[j + 1];
        target -= wj / v;
        residuals.push(wj / s2);
        writes.push(-wj * scales[j] / s2);
        scale_derivatives.push(-wj * ur[j] / s2);
        log_variance_derivative += wj * (0.5 * d - rr[j] / (2.0 * s2));
    }
    let logits_derivative = pi.iter().zip(&w).map(|(p, w)| p.exp() - w).collect();
    Ok(Scalars { value, target, residuals, writes, scales: scale_derivatives, logits: logits_derivative, log_variance: log_variance_derivative })
}

/// The epoch's layout of the blocks' term on the device ([`Mixture::sample_device`]), built at the
/// first step after the components or the groups' activity change.
struct Layout {
    active: Vec<bool>,
    /// The rows of the table: per block operator read (trainable index), whether its blocks are
    /// its columns, its first row and its count of blocks; the token rows follow at `rows`.
    segments: Vec<(usize, bool, usize, usize)>,
    rows: usize,
    tokens: Option<Tensor>,
    d: usize,
    /// Per target with components, in order: its index and its first pair (its components'
    /// pairs follow in order).
    targets: Vec<(usize, usize)>,
    pairs: usize,
    /// Per pair, the table rows of its target and its candidate, and its scale `c` (`pairs × 1`).
    target_rows: Indices,
    candidate_rows: Indices,
    scales: Tensor,
    /// Per target, its table row.
    first_rows: Indices,
    /// The pairs whose candidate is a parameter's (not a token row), in pair order, and per pair
    /// its place among them.
    parameter_pairs: Option<Indices>,
    parameter_place: Vec<Option<usize>>,
    /// The sources' count (targets, then pairs, then parameter pairs) and the destination rows'.
    sources: usize,
    destinations: usize,
    /// Per slot `m`, the `m`-th source of each destination with more than `m` (destinations in
    /// decreasing count, so the slot's destinations are the first).
    slots: Vec<Indices>,
    /// Per segment with a destination, the place of each of its rows among the destinations
    /// (`destinations` for none).
    scatter: Vec<(usize, Indices)>,
    eye: Tensor,
    ones_row: Tensor,
    ones_column: Tensor,
}

/// [`Layout`] held by a [`Mixture`]: not saved, and rebuilt by a copy.
#[derive(Default)]
struct Plan(Option<Layout>);

impl Clone for Plan {
    fn clone(&self) -> Self {
        Self(None)
    }
}

impl std::fmt::Debug for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Plan(laid out)" } else { "Plan(none)" })
    }
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
    /// The gauges that leave its heads' scores unchanged.
    symmetry: library_sharing::Symmetry,
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
    /// Per trainable index, its operator in the program (`Explanation::trainable`).
    #[serde(skip)]
    trainable: Vec<usize>,
    #[serde(skip)]
    plan: Plan,
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
            let symmetry = library_sharing::symmetry(program, &[found], &planes)?;
            key_values.push(((l, g), GroupMaps { queries, key: position(found.key)?, planes, groups, symmetry }));
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
        Ok(Self { targets, width, steps, taken: 0, moments, gates, outputs, ups, key_values, values, embedding, cells, trainable: explanation.trainable.clone(), plan: Plan::default() })
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
    fn align(target: (&[&Array2<f64>], &Array2<f64>), source: (&[&Array2<f64>], &Array2<f64>), planes: &[Vec<usize>], symmetry: &library_sharing::Symmetry) -> Result<(Vec<usize>, Vec<(Array2<f64>, f64)>), String> {
        let ((qt, kt), (qs, ks)) = (target, source);
        let mut assignment: Vec<usize> = (0..qt.len()).collect();
        let mut gauge = library_sharing::gauge(&[], kt, &[], ks, planes, symmetry);
        if qt.len() == 1 {
            return Ok((assignment, library_sharing::gauge(qt, kt, qs, ks, planes, symmetry)));
        }
        for _ in 0..2 {
            let turned: Vec<Array2<f64>> = qs.iter().map(|q| library_sharing::turn(q, planes, &gauge, true, false)).collect();
            let cost = Array2::from_shape_fn((qt.len(), qs.len()), |(i, j)| (qt[i] - &turned[j]).iter().map(|v| v * v).sum::<f64>());
            assignment = crate::library_bodies::hungarian(&cost)?;
            let facing: Vec<&Array2<f64>> = assignment.iter().map(|&j| qs[j]).collect();
            gauge = library_sharing::gauge(qt, kt, &facing, ks, planes, symmetry);
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

    /// Target `t`'s new candidates, with their scales and the variance `ln s²` the epoch sets,
    /// replace its old ones: a kept candidate keeps its logit and its moments, a new one starts at
    /// the zero logit.
    fn adopt(&mut self, t: usize, chosen: Vec<Choice>, log_variance: f64) {
        let old = std::mem::take(&mut self.targets[t].components);
        let zero = self.targets[t].zero_logit;
        let mut moments = vec![self.moments[t][0]];
        let mut components = Vec::with_capacity(chosen.len());
        for Choice { write, scale, gauge, assignment, transport } in chosen {
            let (component, moment) = match old.iter().position(|c| c.write == write) {
                Some(at) => (Component { scale, gauge, assignment, transport, ..old[at].clone() }, [self.moments[t][1 + 2 * at], self.moments[t][2 + 2 * at]]),
                None => (Component { write, scale, logit: zero, gauge, assignment, transport }, [Moment::default(); 2]),
            };
            components.push(component);
            moments.extend(moment);
        }
        self.targets[t].log_variance = log_variance;
        moments.push(self.moments[t].last().copied().unwrap_or_default());
        self.targets[t].components = components;
        self.moments[t] = moments;
    }

    /// Target `t`'s size `|G|` at `posterior`: the entries its groups hold in the explanation.
    fn size(&self, t: usize, posterior: &Posterior) -> Result<f64, String> {
        Ok(match self.targets[t].kind {
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
        })
    }

    /// The nats of a target's `k` candidates among `n` and their parameters (module note):
    /// `ln C(n, k)`, and `½ ln |G|` for each of its `k` logits, `k` scales and its variance.
    fn choice_nats(n: usize, k: usize, size: f64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        let (n, k) = (n as f64, k.min(n) as f64);
        ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0) + (2.0 * k + 1.0) * 0.5 * size.ln()
    }

    /// The nats of a component's target-dependent alignment: 64 bits per stored gauge entry and
    /// scale, and its assignment of `m` query heads, one of `m!` (module note). A value map's
    /// transport and a block's scale are not sent this way: `M` fixes the first, the second is a
    /// learned parameter.
    fn alignment_nats(component: &Component) -> f64 {
        let gauge = component.gauge.iter().map(|(rotation, _)| rotation.len() + 1).sum::<usize>();
        gauge as f64 * 64.0 * std::f64::consts::LN_2 + if component.assignment.len() > 1 { ln_gamma(component.assignment.len() as f64 + 1.0) } else { 0.0 }
    }

    /// A candidate's divergence saving alone ([`Mixture::admit`]) at scale `c`, from the target's
    /// means, variances and prior variances per entry and the candidate's mean vector and total
    /// variance, with the best `s²`.
    fn saving((mean, variance, v): (ArrayView1<'_, f64>, ArrayView1<'_, f64>, ArrayView1<'_, f64>), (u, u_variance): (ArrayView1<'_, f64>, f64), c: f64) -> (f64, f64) {
        let second: Array1<f64> = &mean * &mean + &variance;
        let base: f64 = second.iter().zip(&v).map(|(e, v)| e / (2.0 * v) - 0.5).sum::<f64>();
        let residual = (&mean - &(&u * c)).mapv(|r| r * r).sum() + variance.sum() + c * c * u_variance;
        let s2 = residual / mean.len() as f64;
        (base + v.iter().map(|v| 0.5 * (v / s2).ln()).sum::<f64>(), s2)
    }

    /// The candidates target `t` keeps (module note, # Candidates): each candidate's gain alone at
    /// `posterior`, `Σ_k [½ ln(v_k / s²) − ½ + E[g_k²] / (2 v_k)]` with `s² = E‖g − c u‖² / D`
    /// (the divergence it saves when its component takes the whole weight at the best `s²`), from
    /// the target's means `mean`, variances `variance` and prior variances `v` per entry and each
    /// candidate's mean vector and total variance; in decreasing gain, a candidate is kept while
    /// its gain exceeds what keeping it adds to the mixture's own code length.
    fn admit(&self, t: usize, posterior: &Posterior, (mean, variance, v): (ArrayView1<'_, f64>, ArrayView1<'_, f64>, ArrayView1<'_, f64>), candidates: Vec<(Choice, Array1<f64>, f64)>) -> Result<(Vec<Choice>, f64), String> {
        let mut gains: Vec<(f64, f64, Choice)> = candidates
            .into_iter()
            .map(|(choice, u, u_variance)| {
                let (gain, s2) = Self::saving((mean, variance, v), (u.view(), u_variance), choice.scale);
                (gain, s2, choice)
            })
            .filter(|(gain, s2, _)| gain.is_finite() && *s2 > 0.0)
            .collect();
        gains.sort_by(|a, b| b.0.total_cmp(&a.0));
        let s2 = gains.first().map_or(1.0, |g| g.1);
        let (n, size) = (self.targets[t].choices, self.size(t, posterior)?);
        let mut kept = Vec::new();
        for (gain, _, choice) in gains {
            let k = kept.len() + 1;
            let added = Self::choice_nats(n, k, size) - Self::choice_nats(n, k - 1, size) + Self::alignment_nats(&Component { write: choice.write, scale: choice.scale, logit: 0.0, gauge: choice.gauge.clone(), assignment: choice.assignment.clone(), transport: None });
            if gain <= added {
                break;
            }
            kept.push(choice);
        }
        Ok((kept, s2.ln()))
    }

    /// Re-choose every target's candidates at `posterior` (module note). A gate's candidates are
    /// scored `d` at a time, so the scores held at once are no larger than the gate operator.
    pub fn choose(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        use rayon::prelude::*;
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
                // Each target's `K` best so far and this block's, merged in parallel over targets.
                let width = self.width;
                best.par_iter_mut().zip(&rows).for_each(|(kept, &t)| {
                    let i = match self.targets[t].kind {
                        Kind::Gate { function, .. } | Kind::Up { function, .. } | Kind::Output { function, .. } => function,
                        Kind::QueryKey { .. } | Kind::Value { .. } => return,
                    };
                    for k in 0..end - start {
                        if norm[[i, k]] > 0.0 {
                            kept.push((-cross[[i, k]] * cross[[i, k]] / norm[[i, k]], cross[[i, k]] / norm[[i, k]], start + k));
                        }
                    }
                    if kept.len() > width {
                        kept.select_nth_unstable_by(width, |a, b| a.0.total_cmp(&b.0));
                        kept.truncate(width);
                    }
                });
                start = end;
            }
            for kept in &mut best {
                kept.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            let deviations = |i: usize| -> Result<&Array2<f64>, String> { posterior.log_sd.get(i).ok_or_else(|| "a posterior deviation".to_string()) };
            let mut adopted = Vec::with_capacity(rows.len());
            for (slot, &t) in rows.iter().enumerate() {
                let i = match self.targets[t].kind {
                    Kind::Gate { function, .. } | Kind::Up { function, .. } | Kind::Output { function, .. } => function,
                    Kind::QueryKey { .. } | Kind::Value { .. } => continue,
                };
                let variance = log_sd.row(i).mapv(|s| (2.0 * s).exp());
                let v = Array1::from_elem(variance.len(), group_variance(&self.cells[t], posterior));
                let candidates = best[slot]
                    .iter()
                    .map(|&(_, scale, k)| {
                        let u = self.vector(writes[k], &means)?.to_owned();
                        let u_variance = match writes[k] {
                            Write::Token(_) => 0.0,
                            write => self.vector(write, &deviations)?.mapv(|s| (2.0 * s).exp()).sum(),
                        };
                        Ok((Choice { write: writes[k], scale, gauge: Vec::new(), assignment: Vec::new(), transport: None }, u, u_variance))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let (chosen, log_variance) = self.admit(t, posterior, (mu.row(i), variance.view(), v.view()), candidates)?;
                adopted.push((t, chosen, log_variance));
            }
            for (t, chosen, initial) in adopted {
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
            // A plane's rotation keeps the sum of its rows' variances; its scale multiplies a
            // query's by `s²` and a key's by `1 / s²`.
            let turned_variance = |m: &GroupMaps, assignment: &[usize], gauge: &[(Array2<f64>, f64)]| -> f64 {
                let total = |op: usize, p: usize| maps.planes[p].iter().map(|&r| posterior.log_sd[op].row(r).mapv(|s| (2.0 * s).exp()).sum()).sum::<f64>();
                live.iter().map(|&p| { let s2 = gauge[p].1 * gauge[p].1; assignment.iter().map(|&j| s2 * total(m.queries[j], p)).sum::<f64>() + total(m.key, p) / s2 }).sum()
            };
            let mut scored = Vec::new();
            for &((l, g), ref other) in &self.key_values {
                if l >= layer || !Self::compatible(maps, other) {
                    continue;
                }
                let (q, k) = means(other);
                let (assignment, gauge) = Self::align((&q1, k1), (&q, k), &maps.planes, &maps.symmetry.and(&other.symmetry))?;
                let turned: Vec<Array2<f64>> = assignment.iter().map(|&j| library_sharing::turn(q[j], &maps.planes, &gauge, true, false)).collect();
                let u = Self::group_vector(maps, &live, &turned.iter().collect::<Vec<_>>(), &library_sharing::turn(k, &maps.planes, &gauge, false, false));
                let (cross, norm) = ((&mu * &precision).dot(&u), (&u * &u * &precision).sum());
                if norm > 0.0 {
                    let u_variance = turned_variance(other, &assignment, &gauge);
                    scored.push((-cross * cross / norm, (Choice { write: Write::QueryKey { layer: l, group: g }, scale: cross / norm, gauge, assignment, transport: None }, u, u_variance)));
                }
            }
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            scored.truncate(self.width);
            let variance = sd.mapv(|s| (2.0 * s).exp());
            let d = posterior.mean[maps.key].ncols();
            let mut v = Vec::with_capacity(mu.len());
            for _side in 0..maps.queries.len() + 1 {
                for &p in &live {
                    let cells: Vec<(usize, Vec<usize>, std::ops::Range<usize>)> = maps.queries.iter().chain([&maps.key]).map(|&i| (i, maps.planes[p].clone(), 0..d)).collect();
                    v.extend(std::iter::repeat_n(group_variance(&cells, posterior), maps.planes[p].len() * d));
                }
            }
            let (chosen, log_variance) = self.admit(t, posterior, (mu.view(), variance.view(), Array1::from(v).view()), scored.into_iter().map(|(_, c)| c).collect())?;
            self.adopt(t, chosen, log_variance);
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
                    // `Var((T V_s)_kc) = Σ_r T_kr² σ²_rc`.
                    let rows = posterior.log_sd[value].mapv(|s| (2.0 * s).exp()).sum_axis(ndarray::Axis(1));
                    let u_variance = live.iter().map(|&k| transport.matrix.row(k).mapv(|x| x * x).dot(&rows)).sum::<f64>();
                    scored.push((-cross * cross / norm, (Choice { write: Write::Value { layer: l, group: g }, scale: cross / norm, gauge: Vec::new(), assignment: transport.assignment, transport: Some(transport.matrix) }, u, u_variance)));
                }
            }
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            scored.truncate(self.width);
            let variance = sd.mapv(|s| (2.0 * s).exp());
            let d = posterior.mean[maps.value].ncols();
            let v: Array1<f64> = live.iter().flat_map(|&j| std::iter::repeat_n(group_variance(&[(maps.value, vec![j], 0..d)], posterior), d)).collect();
            let (chosen, log_variance) = self.admit(t, posterior, (mu.view(), variance.view(), v.view()), scored.into_iter().map(|(_, c)| c).collect())?;
            self.adopt(t, chosen, log_variance);
        }
        Ok(())
    }

    /// Component `j` of target `t`'s mean and variance vectors at `posterior`, in the target's
    /// layout: the candidate's posterior means and variances moved as its vector is (a rotation's
    /// squared entries carry the variances, a scale's square).
    fn component_moments(&self, t: usize, j: usize, posterior: &Posterior) -> Result<(Array1<f64>, Array1<f64>), String> {
        let component = &self.targets[t].components[j];
        let variance = |i: usize| posterior.log_sd[i].mapv(|s| (2.0 * s).exp());
        match (self.targets[t].kind, component.write) {
            (Kind::QueryKey { layer, group }, Write::QueryKey { layer: l, group: g }) => {
                let (maps, other) = (self.key_value(layer, group)?, self.key_value(l, g)?);
                let live = self.live_planes(maps, posterior);
                let squared: Vec<(Array2<f64>, f64)> = component.gauge.iter().map(|(r, s)| (r.mapv(|x| x * x), s * s)).collect();
                let turn = |m: &Array2<f64>, gauge: &[(Array2<f64>, f64)], query: bool| library_sharing::turn(m, &maps.planes, gauge, query, false);
                let queries = |values: &dyn Fn(usize) -> Array2<f64>, gauge: &[(Array2<f64>, f64)]| -> Vec<Array2<f64>> { component.assignment.iter().map(|&q| turn(&values(other.queries[q]), gauge, true)).collect() };
                let (mean, var) = (queries(&|i| posterior.mean[i].clone(), &component.gauge), queries(&variance, &squared));
                Ok((
                    Self::group_vector(maps, &live, &mean.iter().collect::<Vec<_>>(), &turn(&posterior.mean[other.key], &component.gauge, false)),
                    Self::group_vector(maps, &live, &var.iter().collect::<Vec<_>>(), &turn(&variance(other.key), &squared, false)),
                ))
            }
            (Kind::Value { layer, group }, Write::Value { layer: l, group: g }) => {
                let (maps, source) = (self.value(layer, group)?, self.value(l, g)?.value);
                let transport = component.transport.as_ref().ok_or("a value map's candidate without its transport")?;
                let live = Self::live_rows(maps, posterior);
                Ok((Self::rows_vector(&live, &transport.dot(&posterior.mean[source])), Self::rows_vector(&live, &transport.mapv(|x| x * x).dot(&variance(source)))))
            }
            (_, write) => {
                let means = |i: usize| -> Result<&Array2<f64>, String> { posterior.mean.get(i).ok_or_else(|| "a posterior mean".to_string()) };
                let mean = self.vector(write, &means)?.to_owned();
                let var = match write {
                    Write::Token(_) => Array1::zeros(mean.len()),
                    write => {
                        let deviations = |i: usize| -> Result<&Array2<f64>, String> { posterior.log_sd.get(i).ok_or_else(|| "a posterior deviation".to_string()) };
                        self.vector(write, &deviations)?.mapv(|s| (2.0 * s).exp())
                    }
                };
                Ok((mean, var))
            }
        }
    }

    /// What making component `j` of target `t` exact is predicted to save in `F`, in nats: the
    /// target's groups' costs (`Posterior::costs`), less half the misfit of the equality
    /// `g = c u` in the posterior's deviations of both, `Σ_k (μ_k − c ν_k)² / (σ_k² + c² τ_k²)`
    /// (the data term's rise at the posterior's curvature), and the choice's `ln n`.
    fn hardening_saving(&self, t: usize, j: usize, posterior: &Posterior, costs: &[f64]) -> Result<f64, String> {
        let target = &self.targets[t];
        let variance = |i: usize| posterior.log_sd[i].mapv(|s| (2.0 * s).exp());
        let (mean, var) = match target.kind {
            Kind::Gate { .. } | Kind::Up { .. } | Kind::Output { .. } => {
                let means = |i: usize| -> Result<&Array2<f64>, String> { posterior.mean.get(i).ok_or_else(|| "a posterior mean".to_string()) };
                let deviations = |i: usize| -> Result<&Array2<f64>, String> { posterior.log_sd.get(i).ok_or_else(|| "a posterior deviation".to_string()) };
                (self.vector(Write::of(target.kind), &means)?.to_owned(), self.vector(Write::of(target.kind), &deviations)?.mapv(|s| (2.0 * s).exp()))
            }
            Kind::QueryKey { layer, group } => {
                let maps = self.key_value(layer, group)?;
                let live = self.live_planes(maps, posterior);
                let queries: Vec<Array2<f64>> = maps.queries.iter().map(|&q| variance(q)).collect();
                (
                    Self::group_vector(maps, &live, &maps.queries.iter().map(|&q| &posterior.mean[q]).collect::<Vec<_>>(), &posterior.mean[maps.key]),
                    Self::group_vector(maps, &live, &queries.iter().collect::<Vec<_>>(), &variance(maps.key)),
                )
            }
            Kind::Value { layer, group } => {
                let maps = self.value(layer, group)?;
                let live = Self::live_rows(maps, posterior);
                (Self::rows_vector(&live, &posterior.mean[maps.value]), Self::rows_vector(&live, &variance(maps.value)))
            }
        };
        let (nu, tau) = self.component_moments(t, j, posterior)?;
        let c = target.components[j].scale;
        let misfit: f64 = mean.iter().zip(&var).zip(nu.iter().zip(&tau)).map(|((m, s), (n, t))| (m - c * n).powi(2) / (s + c * c * t)).sum();
        let saved: f64 = target.groups.iter().map(|g| costs[*g]).sum();
        Ok(saved - 0.5 * misfit - (target.choices as f64).ln())
    }

    /// The dominant components ([`Mixture::dominant`]) that making exact is predicted to lower `F`
    /// by, with the predicted saving in nats (module note, # Hardening).
    pub fn proposals(&self, posterior: &Posterior) -> Result<Vec<(usize, usize, f64)>, String> {
        let costs = posterior.costs();
        let mut out = Vec::new();
        for (t, j) in self.dominant(posterior)? {
            let saving = self.hardening_saving(t, j, posterior, &costs)?;
            if saving > 0.0 {
                out.push((t, j, saving));
            }
        }
        Ok(out)
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
        for (t, j, _) in self.proposals(posterior)? {
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
            // Only the logits: the epoch sets the scales and the variance (module note).
            step(&mut moments[0], &mut target.zero_logit, gradient[0]);
            for (j, component) in target.components.iter_mut().enumerate() {
                step(&mut moments[1 + 2 * j], &mut component.logit, gradient[1 + 2 * j]);
            }
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

impl Mixture {
    /// The operators the terms of `targets` read (trainable indices).
    fn operators_of(&self, targets: &[usize]) -> Vec<usize> {
        let mut out = Vec::new();
        for &t in targets {
            let target = &self.targets[t];
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

    /// The terms of `targets` at `theta` on the host: their sum, its gradient in the operators'
    /// samples, and each target's derivatives in its own parameters into `learned`.
    fn host_terms(&self, targets: &[usize], posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learned: &mut [Vec<f64>]) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
        use rayon::prelude::*;
        // Each target's term and derivatives on its own (in parallel), then summed in target order.
        let found: Vec<Option<(f64, Vec<Piece>, Vec<f64>)>> = targets.par_iter().map(|&t| self.target_term(t, posterior, theta)).collect::<Result<_, String>>()?;
        let mut value = 0.0;
        let mut gradient: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        for (&t, found) in targets.iter().zip(found) {
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
        Ok((value, gradient))
    }

    /// Whether target `t` is a block's (not a key-value group's).
    fn is_block(&self, t: usize) -> bool {
        matches!(self.targets[t].kind, Kind::Gate { .. } | Kind::Up { .. } | Kind::Output { .. })
    }

    /// The table row of a block or token row in `rows` (a segment's first row per operator, the
    /// token rows' places).
    fn table_row(&self, write: Write, segments: &BTreeMap<usize, usize>, tokens: &BTreeMap<usize, usize>) -> Result<usize, String> {
        match write {
            Write::Token(token) => tokens.get(&token).copied().ok_or_else(|| "a token row outside the layout".to_string()),
            Write::Output { function, .. } | Write::Gate { function, .. } | Write::Up { function, .. } => Ok(segments.get(&self.place(write)?.0).ok_or("a block outside the layout")? + function),
            Write::QueryKey { .. } | Write::Value { .. } => Err("a key-value group is no MLP block".into()),
        }
    }

    /// The layout of the block targets `blocks` (each with components) on `device` ([`Layout`]).
    fn layout(&self, device: &Device, posterior: &Posterior, blocks: &[usize]) -> Result<Layout, String> {
        // The operators read, each its rows (gate, up) or its columns (output) as table rows.
        let mut read: Vec<Write> = Vec::new();
        for &t in blocks {
            read.push(Write::of(self.targets[t].kind));
            read.extend(self.targets[t].components.iter().map(|c| c.write));
        }
        let mut operators: Vec<(usize, bool)> = read.iter().filter(|w| !matches!(w, Write::Token(_))).map(|&w| self.place(w)).collect::<Result<_, _>>()?;
        operators.sort_unstable();
        operators.dedup();
        let (mut segments, mut first) = (Vec::new(), BTreeMap::new());
        let mut rows = 0;
        let mut d = None;
        for (i, column) in operators {
            let (r, c) = posterior.mean[i].dim();
            let (count, width) = if column { (c, r) } else { (r, c) };
            if d.is_some_and(|d| d != width) {
                return Err("the mixture's blocks differ in length".into());
            }
            d = Some(width);
            segments.push((i, column, rows, count));
            first.insert(i, rows);
            rows += count;
        }
        let d = d.ok_or("no block in the layout")?;
        let mut tokens: Vec<usize> = read.iter().filter_map(|w| if let Write::Token(t) = w { Some(*t) } else { None }).collect();
        tokens.sort_unstable();
        tokens.dedup();
        let token_rows: BTreeMap<usize, usize> = tokens.iter().enumerate().map(|(k, t)| (*t, rows + k)).collect();
        let upload = |a: &Array2<f64>| device.upload(a.view()).map_err(error);
        let indices = |v: &[usize]| -> Result<Indices, String> { device.upload_indices(&v.iter().map(|&x| u32::try_from(x).map_err(error)).collect::<Result<Vec<_>, _>>()?).map_err(error) };
        let token_table = if tokens.is_empty() { None } else { Some(upload(&Array2::from_shape_fn((tokens.len(), d), |(k, c)| self.embedding[[c, tokens[k]]]))?) };
        // The pairs, target by target.
        let (mut targets, mut target_rows, mut candidate_rows, mut scales, mut first_rows) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut parameter_pairs, mut parameter_place) = (Vec::new(), Vec::new());
        for &t in blocks {
            let target = &self.targets[t];
            let own = self.table_row(Write::of(target.kind), &first, &token_rows)?;
            targets.push((t, target_rows.len()));
            first_rows.push(own);
            for component in &target.components {
                let row = self.table_row(component.write, &first, &token_rows)?;
                parameter_place.push((row < rows).then(|| {
                    parameter_pairs.push(target_rows.len());
                    parameter_pairs.len() - 1
                }));
                target_rows.push(own);
                candidate_rows.push(row);
                scales.push(component.scale);
            }
        }
        let pairs = target_rows.len();
        // Each destination row's sources: a target's own row (coefficient α) and its residuals
        // (β); a parameter candidate's row its residual (γ).
        let mut incoming: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (k, &(_, start)) in targets.iter().enumerate() {
            let sources = incoming.entry(first_rows[k]).or_default();
            sources.push(k);
            let end = targets.get(k + 1).map_or(pairs, |next| next.1);
            sources.extend((start..end).map(|p| targets.len() + p));
        }
        for (q, &p) in parameter_pairs.iter().enumerate() {
            incoming.entry(candidate_rows[p]).or_default().push(targets.len() + pairs + q);
        }
        let mut order: Vec<(usize, Vec<usize>)> = incoming.into_iter().collect();
        order.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        let most = order.first().map_or(0, |o| o.1.len());
        let slots = (0..most).map(|m| indices(&order.iter().take_while(|o| o.1.len() > m).map(|o| o.1[m]).collect::<Vec<_>>())).collect::<Result<Vec<_>, _>>()?;
        let place: BTreeMap<usize, usize> = order.iter().enumerate().map(|(k, o)| (o.0, k)).collect();
        let destinations = order.len();
        let mut scatter = Vec::new();
        for (s, &(_, _, start, count)) in segments.iter().enumerate() {
            let at: Vec<usize> = (start..start + count).map(|r| place.get(&r).copied().unwrap_or(destinations)).collect();
            if at.iter().any(|&a| a < destinations) {
                scatter.push((s, indices(&at)?));
            }
        }
        Ok(Layout {
            active: posterior.active.clone(),
            segments,
            rows,
            tokens: token_table,
            d,
            targets,
            pairs,
            target_rows: indices(&target_rows)?,
            candidate_rows: indices(&candidate_rows)?,
            scales: device.upload_vec(pairs, 1, scales).map_err(error)?,
            first_rows: indices(&first_rows)?,
            parameter_pairs: if parameter_pairs.is_empty() { None } else { Some(indices(&parameter_pairs)?) },
            parameter_place,
            sources: first_rows.len() + pairs + parameter_pairs.len(),
            destinations,
            slots,
            scatter,
            eye: upload(&Array2::eye(d))?,
            ones_row: upload(&Array2::ones((1, d)))?,
            ones_column: upload(&Array2::ones((d, 1)))?,
        })
    }

    /// The block targets' term at the weight sample of `key` on `device` (module note): its value
    /// and gradient by trainable index, each target's derivatives in its own parameters into
    /// `learned`.
    fn blocks_on_device(&self, device: &Device, device_posterior: &DevicePosterior, key: u64, learned: &mut [Vec<f64>]) -> Result<(f64, BTreeMap<usize, Tensor>), String> {
        let layout = self.plan.0.as_ref().ok_or("the blocks' term has no layout")?;
        let arithmetic = if device.storage() == Storage::F64 { Arithmetic::F64 } else { Arithmetic::F32 };
        let (d, pairs) = (layout.d, layout.pairs);
        // The table: every block read at the sample (an output's column turned to a row), then the
        // token rows.
        let mut table = device.zeros(layout.rows + layout.tokens.as_ref().map_or(0, Tensor::rows), d).map_err(error)?;
        for &(i, column, start, count) in &layout.segments {
            let op = self.trainable[i];
            if column {
                let mut sample = device.zeros(d, count).map_err(error)?;
                device_posterior.sample_block(op, &mut sample, (0, 0), key)?;
                let mut turned = device.zeros(count, d).map_err(error)?;
                device.gemm(&mut turned, 1.0, &sample, Op::T, &layout.eye, Op::N, 0.0, arithmetic).map_err(error)?;
                device.set_rows(&mut table, start, &turned).map_err(error)?;
            } else {
                device_posterior.sample_block(op, &mut table, (start, 0), key)?;
            }
        }
        if let Some(tokens) = &layout.tokens {
            device.set_rows(&mut table, layout.rows, tokens).map_err(error)?;
        }
        // Per pair, the target's row `g`, the candidate's `u` and the scaled residual `r = g − c u`.
        let g = device.gather_rows(&table, &layout.target_rows).map_err(error)?;
        let u = device.gather_rows(&table, &layout.candidate_rows).map_err(error)?;
        let spread = |column: &Tensor, rows: usize| -> Result<Tensor, String> {
            let mut out = device.zeros(rows, d).map_err(error)?;
            device.gemm(&mut out, 1.0, column, Op::N, &layout.ones_row, Op::N, 0.0, arithmetic).map_err(error)?;
            Ok(out)
        };
        let mut r = device.copy(&g).map_err(error)?;
        {
            let mut scaled = device.zeros(pairs, d).map_err(error)?;
            device.hadamard(&mut scaled, &spread(&layout.scales, pairs)?, &u, false).map_err(error)?;
            device.axpy(&mut r, -1.0, &scaled).map_err(error)?;
        }
        let sums = |a: &Tensor, b: &Tensor| -> Result<Vec<f64>, String> {
            let mut product = device.zeros(pairs, d).map_err(error)?;
            device.hadamard(&mut product, a, b, false).map_err(error)?;
            let mut out = device.zeros(pairs, 1).map_err(error)?;
            device.gemm(&mut out, 1.0, &product, Op::N, &layout.ones_column, Op::N, 0.0, arithmetic).map_err(error)?;
            Ok(device.download(&out).map_err(error)?.into_iter().collect())
        };
        let (rr, ur, gg) = (sums(&r, &r)?, sums(&u, &r)?, sums(&g, &g)?);
        drop((g, u));
        // The terms and each source row's coefficient: per target α, per pair β, per parameter
        // pair γ.
        let variances = device_posterior.variances()?;
        let count = layout.targets.len();
        let mut coefficients = vec![0.0; layout.sources];
        let mut value = 0.0;
        for (k, &(t, start)) in layout.targets.iter().enumerate() {
            let target = &self.targets[t];
            let n = target.components.len();
            let logits: Vec<f64> = std::iter::once(target.zero_logit).chain(target.components.iter().map(|c| c.logit)).collect();
            let scales: Vec<f64> = target.components.iter().map(|c| c.scale).collect();
            let v = variances[*target.groups.first().ok_or("a block target without its group")?];
            let found = scalar_term(gg[start], (&rr[start..start + n], &ur[start..start + n]), &scales, &logits, target.log_variance, v, d)?;
            value += found.value;
            coefficients[k] = found.target;
            for j in 0..n {
                coefficients[count + start + j] = found.residuals[j];
                if let Some(q) = layout.parameter_place[start + j] {
                    coefficients[count + pairs + q] = found.writes[j];
                }
            }
            let mut own = vec![found.logits[0]];
            for j in 0..n {
                own.extend([found.logits[j + 1], found.scales[j]]);
            }
            own.push(found.log_variance);
            learned[t] = own;
        }
        // The weighted sources, then each destination row's sum, slot by slot.
        let mut sources = device.zeros(layout.sources, d).map_err(error)?;
        device.set_rows(&mut sources, 0, &device.gather_rows(&table, &layout.first_rows).map_err(error)?).map_err(error)?;
        drop(table);
        device.set_rows(&mut sources, count, &r).map_err(error)?;
        if let Some(parameter_pairs) = &layout.parameter_pairs {
            device.set_rows(&mut sources, count + pairs, &device.gather_rows(&r, parameter_pairs).map_err(error)?).map_err(error)?;
        }
        drop(r);
        let mut weighted = device.zeros(layout.sources, d).map_err(error)?;
        device.hadamard(&mut weighted, &spread(&device.upload_vec(layout.sources, 1, coefficients).map_err(error)?, layout.sources)?, &sources, false).map_err(error)?;
        drop(sources);
        let mut rows = device.zeros(layout.destinations + 1, d).map_err(error)?;
        for slot in &layout.slots {
            let mut prefix = device.rows_of(&rows, 0, slot.len()).map_err(error)?;
            device.axpy(&mut prefix, 1.0, &device.gather_rows(&weighted, slot).map_err(error)?).map_err(error)?;
            device.set_rows(&mut rows, 0, &prefix).map_err(error)?;
        }
        let mut gradient = BTreeMap::new();
        for (s, at) in &layout.scatter {
            let (i, column, _, count) = layout.segments[*s];
            let own = device.gather_rows(&rows, at).map_err(error)?;
            let g = if column {
                let mut g = device.zeros(d, count).map_err(error)?;
                device.gemm(&mut g, 1.0, &layout.eye, Op::N, &own, Op::T, 0.0, arithmetic).map_err(error)?;
                g
            } else {
                own
            };
            gradient.insert(i, g);
        }
        Ok((value, gradient))
    }
}

impl PriorTerm for Mixture {
    fn operators(&self) -> Vec<usize> {
        // Only the selected conditional factors need per-step samples. Candidate selection itself
        // sees the complete posterior at the epoch boundary.
        self.operators_of(&(0..self.targets.len()).filter(|&t| !self.targets[t].components.is_empty()).collect::<Vec<_>>())
    }

    fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
        self.plan = Plan::default();
        self.choose(explanation, posterior)
    }

    fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
        let mut learned = vec![Vec::new(); self.targets.len()];
        let found = self.host_terms(&(0..self.targets.len()).collect::<Vec<_>>(), posterior, theta, &mut learned)?;
        if learn {
            self.learn(&learned);
        }
        Ok(found)
    }

    fn sample_device(&mut self, device: &Device, device_posterior: &DevicePosterior, posterior: &mut Posterior, key: u64, learn: bool) -> Result<(f64, BTreeMap<usize, Tensor>), String> {
        let live: Vec<usize> = (0..self.targets.len()).filter(|&t| !self.targets[t].components.is_empty() && self.active(t, posterior)).collect();
        let (blocks, heads): (Vec<usize>, Vec<usize>) = live.into_iter().partition(|&t| self.is_block(t));
        let mut learned = vec![Vec::new(); self.targets.len()];
        let (mut value, mut gradient) = (0.0, BTreeMap::new());
        if !heads.is_empty() {
            // The key-value groups' terms on the host, at their operators' sample.
            let operators = self.operators_of(&heads);
            for &i in &operators {
                let (mean, log_sd) = device_posterior.values(i)?;
                posterior.mean[i] = mean;
                posterior.log_sd[i] = log_sd;
            }
            let theta = crate::library_mdl::host_sample(posterior, &operators, key);
            let (nats, host) = self.host_terms(&heads, posterior, &theta, &mut learned)?;
            value += nats;
            for (i, g) in host {
                gradient.insert(i, device.upload(g.view()).map_err(error)?);
            }
        }
        if !blocks.is_empty() {
            if self.plan.0.as_ref().is_none_or(|layout| layout.active != posterior.active) {
                self.plan = Plan(Some(self.layout(device, posterior, &blocks)?));
            }
            let (nats, on_device) = self.blocks_on_device(device, device_posterior, key, &mut learned)?;
            value += nats;
            for (i, g) in on_device {
                match gradient.get_mut(&i) {
                    Some(total) => device.axpy(total, 1.0, &g).map_err(error)?,
                    None => {
                        gradient.insert(i, g);
                    }
                }
            }
        }
        if learn {
            self.learn(&learned);
        }
        Ok((value, gradient))
    }

    fn cost(&self, posterior: &Posterior) -> Result<f64, String> {
        let mut total = 0.0;
        for t in (0..self.targets.len()).filter(|&t| self.active(t, posterior) && !self.targets[t].components.is_empty()) {
            let target = &self.targets[t];
            let size = self.size(t, posterior)?;
            total += Self::choice_nats(target.choices, target.components.len(), size) + target.components.iter().map(Self::alignment_nats).sum::<f64>();
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
        restored.trainable = std::mem::take(&mut self.trainable);
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

    fn sample_device(&mut self, device: &Device, device_posterior: &DevicePosterior, posterior: &mut Posterior, key: u64, learn: bool) -> Result<(f64, BTreeMap<usize, Tensor>), String> {
        let mut total = 0.0;
        let mut gradient: BTreeMap<usize, Tensor> = BTreeMap::new();
        for prior in &mut self.0 {
            let (value, part) = prior.sample_device(device, device_posterior, posterior, key, learn)?;
            total += value;
            for (i, g) in part {
                match gradient.get_mut(&i) {
                    Some(sum) => device.axpy(sum, 1.0, &g).map_err(error)?,
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
        let mut posterior = Posterior::new(&explanation, NARROW).unwrap();
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

    /// Training tokens that make a posterior narrow enough for an exact query–key copy of the
    /// tiny model (64 entries) to pay for its gauge's literal charge (module note): with
    /// `σ² = v / N`, its gain `½ D ln(N / 2)` exceeds the 64 bits per gauge value.
    const NARROW: usize = 1 << 40;

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
        let imported = crate::test_support::grouped(name);
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
        let posterior = Posterior::new(&start, NARROW).unwrap();
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
        crate::test_support::same_native_blocks(&start, &hardened);
        let groups = library_sharing::key_values(&hardened).unwrap();
        assert!(!groups[&(1, 0)].own_key, "layer 1's query heads read layer 0's maps");
        assert!(hardened.fixed_nats >= (mixture.targets[t].choices as f64).ln() - 1e-12, "the choice is paid for");
    }

    #[test]
    fn a_key_value_group_term_is_differentiated_through_its_gauge_and_assignment() {
        let (_, start) = swapped_group("library_mixture_grouped_gradient");
        let posterior = Posterior::new(&start, NARROW).unwrap();
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
        let imported = crate::test_support::grouped(name);
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
        crate::test_support::same_native_blocks(&start, &hardened);
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
    fn the_scalar_term_is_the_term_at_one_variance() {
        let d = 9;
        let at = |stream: u64| Array1::from_shape_fn(d, |k| f64::from(posterior_normal(3, stream, k as u64)));
        let g = at(0);
        let writes = [at(1), &g * 0.9 + &at(2) * 0.05, at(3)];
        let scales = [0.4, 1.1, -0.7];
        let logits = [0.2, -0.3, 0.6, 0.1];
        let (log_variance, v) = (-1.3, 0.8);
        let views: Vec<(ArrayView1<'_, f64>, f64)> = writes.iter().zip(scales).map(|(u, c)| (u.view(), c)).collect();
        let full = term(g.view(), &views, &logits, log_variance, Array1::from_elem(d, v).view()).unwrap();
        let residuals: Vec<Array1<f64>> = writes.iter().zip(scales).map(|(u, c)| &g - &(u * c)).collect();
        let rr: Vec<f64> = residuals.iter().map(|r| r.dot(r)).collect();
        let ur: Vec<f64> = writes.iter().zip(&residuals).map(|(u, r)| u.dot(r)).collect();
        let found = scalar_term(g.dot(&g), (&rr, &ur), &scales, &logits, log_variance, v, d).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
        assert!(close(found.value, full.value));
        let target = residuals.iter().zip(&found.residuals).fold(&g * found.target, |sum, (r, b)| sum + r * *b);
        assert!(target.iter().zip(&full.target).all(|(a, b)| close(*a, *b)), "the target's derivative");
        for (j, r) in residuals.iter().enumerate() {
            assert!((r * found.writes[j]).iter().zip(&full.writes[j]).all(|(a, b)| close(*a, *b)), "a candidate's derivative");
            assert!(close(found.scales[j], full.scales[j]));
        }
        assert!(found.logits.iter().zip(&full.logits).all(|(a, b)| close(*a, *b)));
        assert!(close(found.log_variance, full.log_variance));
    }

    #[test]
    fn the_blocks_term_on_the_device_is_the_host_term() {
        use crate::device_posterior::DevicePosterior;
        let (explanation, posterior, mut mixture) = head_fixture("library_mixture_device");
        mixture.choose(&explanation, &posterior).unwrap();
        // Planted components of every block kind and candidate: a gate reading an earlier output,
        // gate and token row, a second gate reading the same output, an output two earlier ones.
        let plant = |mixture: &mut Mixture, kind: Kind, writes: &[(Write, f64, f64)]| {
            let t = mixture.targets.iter().position(|t| t.kind == kind).unwrap();
            let target = &mut mixture.targets[t];
            target.components = writes.iter().map(|&(write, scale, logit)| Component { write, scale, logit, gauge: Vec::new(), assignment: Vec::new(), transport: None }).collect();
            target.zero_logit = 0.3;
            target.log_variance = -4.0;
            mixture.moments[t] = vec![Moment::default(); 2 + 2 * writes.len()];
        };
        plant(&mut mixture, Kind::Gate { layer: 1, function: 5 }, &[(Write::Output { layer: 0, function: 3 }, 0.8, 0.1), (Write::Gate { layer: 0, function: 2 }, -0.5, -0.2), (Write::Token(7), 1.2, 0.0)]);
        plant(&mut mixture, Kind::Gate { layer: 1, function: 6 }, &[(Write::Output { layer: 0, function: 3 }, 0.4, 0.5)]);
        plant(&mut mixture, Kind::Output { layer: 1, function: 2 }, &[(Write::Output { layer: 0, function: 3 }, 1.1, 0.2), (Write::Output { layer: 0, function: 1 }, 0.3, -0.1)]);
        assert!(mixture.targets.iter().any(|t| matches!(t.kind, Kind::QueryKey { .. }) && !t.components.is_empty()), "a key-value group's term on the host too");
        let device = Device::host();
        let resident = DevicePosterior::new(&device, &explanation, &posterior, 1e6, None, 0).unwrap();
        let mut host = mixture.clone();
        let theta = draw(&host, &posterior, 9);
        let (value, gradient) = host.sample(&posterior, &theta, true).unwrap();
        let (on_device, moved) = mixture.sample_device(&device, &resident, &mut posterior.clone(), 9, true).unwrap();
        assert!((on_device - value).abs() <= 1e-9 * (1.0 + value.abs()), "the value {on_device} against {value}");
        assert_eq!(moved.keys().collect::<Vec<_>>(), gradient.keys().collect::<Vec<_>>());
        for (i, g) in &gradient {
            let scale = g.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            let found = device.download(&moved[i]).unwrap();
            assert!(found.iter().zip(g.iter()).all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + scale)), "operator {i}'s gradient");
        }
        // One step of the mixture's own parameters from the same derivatives.
        for (a, b) in mixture.targets.iter().zip(&host.targets) {
            assert!((a.zero_logit - b.zero_logit).abs() < 1e-12 && a.components.iter().zip(&b.components).all(|(x, y)| (x.logit - y.logit).abs() < 1e-12));
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
        let imported = crate::test_support::gated("library_mixture_gated");
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
        assert!(mixture.dominant(&posterior).unwrap().contains(&(t, copy)));
        // The sampled term, once the copy holds the weight, is the closed-form saving it was kept
        // for, negated.
        let means = |i: usize| -> Result<&Array2<f64>, String> { Ok(&posterior.mean[i]) };
        let deviations = |i: usize| -> Result<&Array2<f64>, String> { Ok(&posterior.log_sd[i]) };
        let own = Write::Up { layer: 1, function: 6 };
        let (mean, variance) = (mixture.vector(own, &means).unwrap().to_owned(), mixture.vector(own, &deviations).unwrap().mapv(|s| (2.0 * s).exp()));
        let v = Array1::from_elem(mean.len(), group_variance(&mixture.cells[t], &posterior));
        let (u, tau) = mixture.component_moments(t, copy, &posterior).unwrap();
        let (saving, _) = Mixture::saving((mean.view(), variance.view(), v.view()), (u.view(), tau.sum()), mixture.targets[t].components[copy].scale);
        let sampled = (0..400_u64).map(|key| mixture.target_term(t, &posterior, &draw(&mixture, &posterior, 1000 + key)).unwrap().unwrap().0).sum::<f64>() / 400.0;
        let held = -weights[1 + copy].ln();
        assert!((sampled + saving).abs() <= 0.05 * saving.abs() + held + 0.5, "sampled term {sampled} against the saving {saving}");
        // Of the dominant components only those whose equality saves code length are made exact.
        let hardened = mixture.harden(&start, &posterior).unwrap();
        let (before, after) = (start.artifact.execute(&imported.family).unwrap(), hardened.artifact.execute(&imported.family).unwrap());
        let (a, b) = (&before.values[start.artifact.program.output], &after.values[hardened.artifact.program.output]);
        let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= 1e-12 * scale), "the tied up direction keeps the outputs");
        crate::test_support::same_native_blocks(&start, &hardened);
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
        let settings = Settings { batch_sequences: 2, rate: 0.1, beta1: 0.9, seed: 3, numeric_bytes: 1 << 26, head_tile_rows: 64, decoder: false, trust_rate: false, split_filter: false, deterministic: false };
        let mut mixture = Mixture::new(&start, 2, Steps { rate: 0.05, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let (train, held) = sequences.split_at(4);
        let fitted = fit(&Device::host(), &native, &start, train, held, &settings, "tiny", None, Some(&mut mixture)).unwrap();
        // F holds the mixture's term: the start's objective is its data term plus every part of its
        // description, the mixture's among them. At the Laplace start (ecf0f9c44b) this tiny
        // collection leaves most deviations near the prior's, where no copy pays for its choice
        // (1da5445a19), so the term may be zero there; the mixture's choices are tested below at a
        // narrow posterior.
        let at_start = &fitted.report.start;
        let description = at_start.divergence_bits + at_start.variance_bits + at_start.choice_bits + at_start.prior_bits;
        let assembled = at_start.data_bits_per_token + description / fitted.report.scored_tokens as f64;
        assert!(at_start.prior_bits.is_finite() && (at_start.objective_bits_per_token - assembled).abs() <= 1e-9 * at_start.objective_bits_per_token.abs(), "F holds the mixture's term");
        // The fit removes the copies this random model does not need; the weights are learned at the
        // start, every group in, from samples of its posterior.
        let all_in = crate::library_mdl::Posterior::new(&start, NARROW).unwrap();
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
        crate::test_support::same_native_blocks(&start, &hardened);
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
