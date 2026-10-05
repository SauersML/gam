//! The explanation as a library of learned functions, fitted end to end by variational minimum
//! description length (#2951).
//!
//! # The explanation
//!
//! Every attention head and every MLP of the native model `M` is replaced by a block of learned
//! functions that runs on the explanation's own vectors. `M`'s embedding, norms, attention output
//! projection, final norm and unembedding stay.
//!
//! * A head's block reads its layer's normed stream `x` and writes the head's read (the attention
//!   output projection's input), so removing, scaling or mixing a head acts on the explanation at
//!   the same place as on `M`. Its function attends with its own query, key and value maps
//!   `q = Q x`, `k = K x`, `v = V x` at the native rotary angles, scale and causal mask.
//! * An MLP's block reads its layer's second normed stream and writes the MLP's output. Its
//!   functions are `f_i(x) = relu(g_i·x + c_i) u_i`: a gate direction `g_i`, a gate bias `c_i` and
//!   an output `u_i`. A function is exactly zero wherever its gate is not positive. Removing or
//!   scaling the whole MLP stays expressible through a uniform-scale control on its output
//!   (`native_control`).
//!
//! The library starts at `M`: native neuron `i` is function `i` with its GELU replaced by the ReLU,
//! and a head is its own function.
//!
//! # The code length
//!
//! The library's parameters `θ` are partitioned into prior groups `G`: a head's rotary plane (the
//! query and key rows of the plane), a head's value coordinate (one value row), an MLP function's
//! gate `(g_i, c_i)` and its output `u_i`. With the posterior `q(θ) = Π_j N(μ_j, σ_j²)` and the
//! prior `p(θ_G) = N(0, v_G I)`, the code length in nats is
//!
//! `F = E_q[D(θ)] + Σ_G KL(q_G ‖ p_G) + |active groups| · 32 ln 2`,
//!
//! the bits-back code of the training data given the explanation plus the explanation. `D(θ)`
//! sums `KL(p_M ‖ p_θ)` of the next-token distributions over every training token, so the data
//! term's weight is the number of tokens and no tradeoff weight exists. Each prior variance is
//! chosen by empirical Bayes in closed form, `v_G = (1/n_G) Σ_{j∈G} (μ_j² + σ_j²)`, which makes
//! `KL(q_G ‖ p_G) = ½ (n_G ln v_G − Σ_{j∈G} ln σ_j²)` with derivatives `μ_j / v_G` in `μ_j` and
//! `σ_j² / v_G − 1` in `ln σ_j`. An active group sends its variance as one 32-bit literal. A group
//! the data does not inform sits at its prior with zero divergence and posterior mean zero, so the
//! null is recovered; removing it from the explanation is a discrete step of the same `F`.
//!
//! Weight noise costs data only on the tokens where a function is active, so a ReLU-gated function
//! that is exactly zero elsewhere can keep imprecise, cheap weights: the code length rewards
//! functions that are inactive on most inputs without any sparsity penalty.
//!
//! # The fit
//!
//! Each step draws one weight sample `θ = μ + σ ⊙ ε` (`ε` standard normal), runs the explanation on
//! a batch of complete training sequences, scores it against `M`'s distributions on the same batch
//! (compact fixed-head statistics, computed by `M` on the device), and takes one Adam step in
//! `(μ, ln σ)` on the batch estimate `(N/n) D_batch(θ) + Σ_G KL_G`, `N` the training tokens and `n`
//! the batch's. An epoch visits every training batch once, in a fixed order. The continuous fit
//! has converged when an epoch's mean improvement of the per-batch objective estimate over the
//! previous epoch, paired by batch, is smaller than its standard error.
//!
//! A converged fit then proposes removing groups in increasing order of their divergence `KL_G`
//! (their information content) and keeps the longest prefix, found by bisection, whose removal does
//! not increase `F`. `F` is evaluated over the whole training set with the same weight noise for
//! both sides of every comparison. The order is a proposal; the acceptance is the objective. The
//! fit alternates converging and removing until no removal is accepted.
//!
//! The explanation is reported at the posterior mean `μ` ([`posterior_mean`]); removed groups are
//! absent blocks of their operators, so they cost no literals.

use crate::{
    artifact::{Argument, Artifact, Callee},
    artifact_device::mapped_inlined,
    device_program::DeviceProgram,
    operator_program::{
        FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, Provenance, Rule, SequenceLayout, SlotValues,
        exact_precision,
    },
    resident_causal_fit::fixed_head_target::{Head, ResidentHead, Target},
    run_check::LayerNodes,
};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ops::Range, sync::Arc, time::Instant};

/// The bits of one independently sent literal (`acceptance::LITERAL_BITS`), as nats.
const LITERAL_NATS: f64 = crate::acceptance::LITERAL_BITS as f64 * std::f64::consts::LN_2;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Entries of one trainable operator: `rows × cols`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cells {
    pub operator: usize,
    pub rows: Vec<usize>,
    pub cols: Range<usize>,
}

/// A prior group: parameters sharing one prior variance (module note).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub name: String,
    pub cells: Vec<Cells>,
}

/// The library explanation of a language model: the artifact at its starting point, its trainable
/// operators and its prior groups.
#[derive(Clone, Debug)]
pub struct Explanation {
    pub artifact: Artifact,
    /// The library's operators of `artifact.program`, ascending.
    pub trainable: Vec<usize>,
    pub groups: Vec<Group>,
}

/// A library operator: dense, every block present, its reals exactly representable.
fn library_operator(name: &str, rows: Interface, cols: Interface, values: Array2<f64>, source: &str) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, Provenance::derived(&[&Provenance::native(source)], "library initialization".into()))
        .map_err(error)
}

/// The native operator of a bias-free affine node reading `input` alone.
fn single_map(native: &OperatorProgram, node: usize, input: usize) -> Result<Arc<Operator>, String> {
    match &native.nodes[node] {
        Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == input => Ok(Arc::clone(&native.operators[terms[0].1])),
        other => Err(format!("node {node} is not a bias-free map of node {input}: {other:?}")),
    }
}

fn index_of(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(index), None) => Ok(index),
        _ => Err(format!("no unique library operator {name}")),
    }
}

/// The library explanation of the split native language model `native` (`run_check::split_sites`)
/// with its `layers` (`run_check::layer_nodes`), at its starting point (module note).
pub fn explanation(native: &OperatorProgram, layers: &[LayerNodes]) -> Result<Explanation, String> {
    let mut artifact = Artifact::native(native)?;
    let mut planes = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        for (h, &read) in layer.reads.iter().enumerate() {
            let Node::Attend { query, key, value, scale, rotary, causal } = native.nodes[read].clone() else {
                return Err(format!("layer {l} head {h}: the read is not an attention node"));
            };
            let x = layer.normed_stream;
            let (q, k, v) = (single_map(native, query, x)?, single_map(native, key, x)?, single_map(native, value, x)?);
            if q.rows.width() != k.rows.width() {
                return Err(format!("layer {l} head {h}: query and key widths differ"));
            }
            // Query and key coordinates in their own groups, so a removed plane is an absent block.
            let coordinates = Interface::uniform(q.rows.width(), 1, LabelKind::Unit, 0).map_err(error)?;
            let name = format!("library.l{l}.h{h}");
            let base = artifact.program.operators.len();
            let operators = vec![
                library_operator(&format!("{name}.q"), coordinates.clone(), q.cols.clone(), q.matrix(), &q.name)?,
                library_operator(&format!("{name}.k"), coordinates, k.cols.clone(), k.matrix(), &k.name)?,
                library_operator(&format!("{name}.v"), v.rows.clone(), v.cols.clone(), v.matrix(), &v.name)?,
            ];
            let rule = Rule {
                name: name.clone(),
                inputs: vec![q.cols.clone()],
                nodes: vec![
                    Node::Param { index: 0 },
                    Node::Affine { terms: vec![(0, base)], bias: None },
                    Node::Affine { terms: vec![(0, base + 1)], bias: None },
                    Node::Affine { terms: vec![(0, base + 2)], bias: None },
                    Node::Attend { query: 1, key: 2, value: 3, scale, rotary, causal },
                ],
                output: 4,
            };
            artifact = artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(x)], read, operators)?;
            let pairs: Vec<Vec<usize>> = match rotary {
                Some(r) => {
                    let pairs = r.pairs();
                    let rotated: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    pairs.iter().map(|&(a, b)| vec![a, b]).chain((0..q.rows.width()).filter(|c| !rotated.contains(c)).map(|c| vec![c])).collect()
                }
                None => (0..q.rows.width()).map(|c| vec![c]).collect(),
            };
            planes.push((name, pairs, v.rows.width()));
        }
        let Node::Pointwise { input: pre, laws } = &native.nodes[layer.active] else {
            return Err(format!("layer {l}: the MLP activation is not one pointwise law (gated MLPs are not supported yet)"));
        };
        if laws.iter().any(|law| *law != laws[0]) {
            return Err(format!("layer {l}: the MLP's units have different laws"));
        }
        let (up, up_bias) = match &native.nodes[*pre] {
            Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == layer.normed => (Arc::clone(&native.operators[terms[0].1]), *bias),
            other => return Err(format!("layer {l}: the MLP's input map is {other:?}")),
        };
        let down = single_map(native, layer.mlp, layer.active)?;
        let units = up.rows.clone();
        let width = units.width();
        let bias = match up_bias {
            Some(op) => native.operators[op].matrix(),
            None => Array2::zeros((width, 1)),
        };
        let name = format!("library.l{l}.mlp");
        let base = artifact.program.operators.len();
        let operators = vec![
            library_operator(&format!("{name}.gate"), units.clone(), up.cols.clone(), up.matrix(), &up.name)?,
            library_operator(&format!("{name}.gate_bias"), units.clone(), Interface::constant(), bias, &up.name)?,
            library_operator(&format!("{name}.out"), down.rows.clone(), units.clone(), down.matrix(), &down.name)?,
        ];
        let rule = Rule {
            name: name.clone(),
            inputs: vec![up.cols.clone()],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Affine { terms: vec![(0, base)], bias: Some(base + 1) },
                Node::Pointwise { input: 1, laws: vec![Law::Relu; units.group_count()] },
                Node::Affine { terms: vec![(2, base + 2)], bias: None },
            ],
            output: 3,
        };
        artifact = artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(layer.normed)], layer.mlp, operators)?;
        artifact = artifact.with_uniform_scale_control(native, layer.active, layer.mlp)?;
    }
    // Groups by the final operator indices (each replacement renumbers the operators).
    let program = &artifact.program;
    let mut groups = Vec::new();
    let mut trainable = Vec::new();
    for (name, pairs, values) in &planes {
        let (q, k, v) = (index_of(program, &format!("{name}.q"))?, index_of(program, &format!("{name}.k"))?, index_of(program, &format!("{name}.v"))?);
        let d = program.operators[q].cols.width();
        for (p, rows) in pairs.iter().enumerate() {
            groups.push(Group {
                name: format!("{name}.plane{p}"),
                cells: vec![Cells { operator: q, rows: rows.clone(), cols: 0..d }, Cells { operator: k, rows: rows.clone(), cols: 0..d }],
            });
        }
        for j in 0..*values {
            groups.push(Group { name: format!("{name}.value{j}"), cells: vec![Cells { operator: v, rows: vec![j], cols: 0..d }] });
        }
        trainable.extend([q, k, v]);
    }
    for l in 0..layers.len() {
        let name = format!("library.l{l}.mlp");
        let (gate, bias, out) =
            (index_of(program, &format!("{name}.gate"))?, index_of(program, &format!("{name}.gate_bias"))?, index_of(program, &format!("{name}.out"))?);
        let (functions, d) = (program.operators[gate].rows.width(), program.operators[gate].cols.width());
        for i in 0..functions {
            groups.push(Group {
                name: format!("{name}.f{i}.gate"),
                cells: vec![Cells { operator: gate, rows: vec![i], cols: 0..d }, Cells { operator: bias, rows: vec![i], cols: 0..1 }],
            });
        }
        for i in 0..functions {
            groups.push(Group { name: format!("{name}.f{i}.out"), cells: vec![Cells { operator: out, rows: (0..d).collect(), cols: i..i + 1 }] });
        }
        trainable.extend([gate, bias, out]);
    }
    trainable.sort_unstable();
    Ok(Explanation { artifact, trainable, groups })
}

// ------------------------------------------------------------------------------------- posterior

/// The factorized Gaussian posterior over the library's parameters, and which groups are active.
#[derive(Clone, Debug)]
pub struct Posterior {
    /// Per trainable operator (in `Explanation::trainable` order), the posterior means `μ`.
    pub mean: Vec<Array2<f64>>,
    /// The posterior standard deviations' logarithms `ln σ`.
    pub log_sd: Vec<Array2<f64>>,
    /// Per group, whether it is in the explanation; a removed group's parameters are exactly zero.
    pub active: Vec<bool>,
    /// Per trainable operator, each entry's group.
    membership: Vec<Array2<u32>>,
    /// Per trainable operator, the range of its groups' indices.
    spans: Vec<Range<usize>>,
}

/// Per group: its size, `Σ (μ² + σ²)` and `Σ ln σ²`.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    count: f64,
    second: f64,
    log_variance: f64,
}

impl Moments {
    /// `KL(q_G ‖ p_G)` at the empirical-Bayes variance, in nats.
    fn divergence(&self) -> f64 {
        0.5 * (self.count * (self.second / self.count).ln() - self.log_variance)
    }
}

impl Posterior {
    /// The posterior at the explanation's starting point: means at the starting values, and each
    /// group's variance `v_G / N` with `v_G` the mean square of its starting values and `N` the
    /// training tokens (the precision of `N` unit-information observations).
    pub fn new(explanation: &Explanation, tokens: usize) -> Result<Self, String> {
        let program = &explanation.artifact.program;
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let mean: Vec<Array2<f64>> = explanation.trainable.iter().map(|op| program.operators[*op].matrix()).collect();
        let mut membership: Vec<Array2<u32>> = mean.iter().map(|m| Array2::from_elem(m.dim(), u32::MAX)).collect();
        let mut spans: Vec<Range<usize>> = vec![usize::MAX..0; mean.len()];
        if u32::try_from(explanation.groups.len()).is_err() {
            return Err("too many prior groups".into());
        }
        for (g, group) in explanation.groups.iter().enumerate() {
            for cell in &group.cells {
                let i = *position.get(&cell.operator).ok_or_else(|| format!("{}: operator {} is not trainable", group.name, cell.operator))?;
                for &row in &cell.rows {
                    for col in cell.cols.clone() {
                        let slot = membership[i].get_mut((row, col)).ok_or_else(|| format!("{}: entry ({row}, {col}) outside its operator", group.name))?;
                        if *slot != u32::MAX {
                            return Err(format!("{}: entry ({row}, {col}) of operator {} is in two groups", group.name, cell.operator));
                        }
                        *slot = g as u32;
                    }
                }
                spans[i] = spans[i].start.min(g)..spans[i].end.max(g + 1);
            }
        }
        if membership.iter().any(|m| m.iter().any(|g| *g == u32::MAX)) {
            return Err("a trainable entry is in no prior group".into());
        }
        if tokens == 0 {
            return Err("no training tokens".into());
        }
        // The starting variance of each group from its means alone.
        let mut squares = vec![(0.0, 0.0); explanation.groups.len()];
        for (mean, membership) in mean.iter().zip(&membership) {
            for (value, group) in mean.iter().zip(membership.iter()) {
                let entry = &mut squares[*group as usize];
                entry.0 += 1.0;
                entry.1 += value * value;
            }
        }
        if let Some(g) = squares.iter().position(|(_, sum)| !(*sum > 0.0 && sum.is_finite())) {
            return Err(format!("{}: a group starting at zero has no scale", explanation.groups[g].name));
        }
        let log_sd = membership
            .iter()
            .map(|membership| {
                membership.mapv(|group| {
                    let (count, sum) = squares[group as usize];
                    0.5 * (sum / count / tokens as f64).ln()
                })
            })
            .collect();
        Ok(Self { mean, log_sd, active: vec![true; explanation.groups.len()], membership, spans })
    }

    fn moments(&self) -> Vec<Moments> {
        let partial: Vec<(Range<usize>, Vec<Moments>)> = (0..self.mean.len())
            .into_par_iter()
            .map(|i| {
                let span = self.spans[i].clone();
                let mut local = vec![Moments::default(); span.len()];
                for ((mu, s), group) in self.mean[i].iter().zip(self.log_sd[i].iter()).zip(self.membership[i].iter()) {
                    let g = *group as usize;
                    if !self.active[g] {
                        continue;
                    }
                    let m = &mut local[g - span.start];
                    m.count += 1.0;
                    m.second += mu * mu + (2.0 * s).exp();
                    m.log_variance += 2.0 * s;
                }
                (span, local)
            })
            .collect();
        let mut out = vec![Moments::default(); self.active.len()];
        for (span, local) in partial {
            for (g, m) in span.zip(local) {
                out[g].count += m.count;
                out[g].second += m.second;
                out[g].log_variance += m.log_variance;
            }
        }
        out
    }

    /// Per group, `KL(q_G ‖ p_G)` in nats (zero for a removed group).
    pub fn divergences(&self) -> Vec<f64> {
        self.moments().iter().zip(&self.active).map(|(m, active)| if *active { m.divergence() } else { 0.0 }).collect()
    }

    /// `Σ_G KL(q_G ‖ p_G)` plus the active groups' variance literals, in nats.
    pub fn description(&self) -> f64 {
        self.divergences().iter().sum::<f64>() + self.active.iter().filter(|a| **a).count() as f64 * LITERAL_NATS
    }

    /// The derivatives in `μ` and in `ln σ` of `data(θ) + Σ_G KL_G`, per trainable operator, from
    /// `data`'s gradient at the sample `θ = μ + σ ⊙ noise` (zero in removed groups).
    fn derivatives(&self, data: &[Array2<f64>], noise: &[Array2<f64>]) -> Vec<(Array2<f64>, Array2<f64>)> {
        let variance: Vec<f64> = self.moments().iter().map(|m| if m.count > 0.0 { m.second / m.count } else { 0.0 }).collect();
        (0..data.len())
            .into_par_iter()
            .map(|i| {
                let (mut mean, mut log_sd) = (data[i].clone(), Array2::zeros(data[i].dim()));
                ndarray::Zip::from(&mut mean).and(&mut log_sd).and(&noise[i]).and(&self.mean[i]).and(&self.log_sd[i]).and(&self.membership[i]).for_each(
                    |gm, gs, e, mu, s, group| {
                        let v = variance[*group as usize];
                        if v > 0.0 {
                            let sd = s.exp();
                            *gs = *gm * e * sd + sd * sd / v - 1.0;
                            *gm += mu / v;
                        } else {
                            *gm = 0.0;
                        }
                    },
                );
                (mean, log_sd)
            })
            .collect()
    }

    /// A weight sample `μ + σ ⊙ ε` (zero in removed groups) and its noise `ε`, drawn from `seed`.
    fn sample(&self, seed: u64) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        (0..self.mean.len())
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let noise = standard_normal(&mut rng, self.mean[i].dim());
                let mut theta = self.mean[i].clone();
                ndarray::Zip::from(&mut theta).and(&noise).and(&self.log_sd[i]).and(&self.membership[i]).for_each(|t, e, s, g| {
                    *t = if self.active[*g as usize] { *t + s.exp() * e } else { 0.0 };
                });
                (theta, noise)
            })
            .unzip()
    }

    /// The posterior means with every removed group zeroed.
    pub fn means(&self) -> Vec<Array2<f64>> {
        self.mean
            .iter()
            .zip(&self.membership)
            .map(|(mean, membership)| {
                let mut out = mean.clone();
                ndarray::Zip::from(&mut out).and(membership).for_each(|v, g| {
                    if !self.active[*g as usize] {
                        *v = 0.0;
                    }
                });
                out
            })
            .collect()
    }

    /// Remove `groups` from the explanation: their parameters become exactly zero.
    fn remove(&mut self, groups: &[usize]) {
        for g in groups {
            self.active[*g] = false;
        }
        let active = &self.active;
        self.mean.par_iter_mut().zip(self.log_sd.par_iter_mut()).zip(self.membership.par_iter()).for_each(|((mean, log_sd), membership)| {
            ndarray::Zip::from(mean).and(log_sd).and(membership).for_each(|m, s, g| {
                if !active[*g as usize] {
                    *m = 0.0;
                    *s = f64::NEG_INFINITY;
                }
            });
        });
    }
}

/// Standard normal draws (Box–Muller) of shape `dim`.
fn standard_normal(rng: &mut StdRng, dim: (usize, usize)) -> Array2<f64> {
    let n = dim.0 * dim.1;
    let mut values = Vec::with_capacity(n + 1);
    while values.len() < n {
        let u = 1.0 - rng.random::<f64>();
        let angle = std::f64::consts::TAU * rng.random::<f64>();
        let radius = (-2.0 * u.ln()).sqrt();
        values.push(radius * angle.cos());
        values.push(radius * angle.sin());
    }
    values.truncate(n);
    Array2::from_shape_vec(dim, values).expect("shape of the drawn values")
}

/// Adam's moments for one parameter array.
#[derive(Clone)]
struct Moment {
    first: Array2<f64>,
    second: Array2<f64>,
}

impl Moment {
    fn zeros(dim: (usize, usize)) -> Self {
        Self { first: Array2::zeros(dim), second: Array2::zeros(dim) }
    }

    /// One Adam step of `value` along `gradient` (entries of removed groups are left alone).
    fn step(&mut self, value: &mut Array2<f64>, gradient: &Array2<f64>, membership: &Array2<u32>, active: &[bool], rate: f64, settings: &Settings, step: i32) {
        let (b1, b2) = (settings.beta1, settings.beta2);
        let (c1, c2) = (1.0 - b1.powi(step), 1.0 - b2.powi(step));
        ndarray::Zip::from(value).and(gradient).and(&mut self.first).and(&mut self.second).and(membership).for_each(|w, g, m, v, group| {
            if active[*group as usize] {
                *m = b1 * *m + (1.0 - b1) * g;
                *v = b2 * *v + (1.0 - b2) * g * g;
                *w -= rate * (*m / c1) / ((*v / c2).sqrt() + settings.epsilon);
            }
        });
    }
}

// ------------------------------------------------------------------------------------- the fit

/// The optimizer's step sizes and the run's resources (none of them is part of the objective).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Complete training sequences per step.
    pub batch_sequences: usize,
    /// Adam step sizes in `μ` and in `ln σ`.
    pub mean_step: f64,
    pub log_sd_step: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    /// The weight noise's seed.
    pub seed: u64,
    /// The device's numeric buffers for operators (each of the native and the explanation).
    pub numeric_bytes: usize,
    /// Rows of vocabulary logits formed at once.
    pub head_tile_rows: usize,
}

impl Settings {
    fn validate(&self) -> Result<(), String> {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if self.batch_sequences == 0
            || !positive(self.mean_step)
            || !positive(self.log_sd_step)
            || !positive(self.epsilon)
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
            || self.numeric_bytes == 0
            || self.head_tile_rows == 0
        {
            return Err("invalid library fit settings".into());
        }
        Ok(())
    }
}

/// One epoch of the continuous fit.
#[derive(Clone, Debug, Serialize)]
pub struct Epoch {
    pub epoch: usize,
    /// Mean over the epoch's steps of the objective estimate, and of its data and description
    /// parts, in bits.
    pub objective_bits: f64,
    pub data_bits: f64,
    pub description_bits: f64,
    /// Mean paired improvement over the previous epoch and its standard error, in bits.
    pub improvement_bits: Option<f64>,
    pub standard_error_bits: Option<f64>,
    pub active_groups: usize,
    /// Mean `KL(p_M ‖ p_θ)` per training token at the weight samples, in nats.
    pub mean_token_kl: f64,
    pub seconds: f64,
}

/// One removal step.
#[derive(Clone, Debug, Serialize)]
pub struct Removal {
    pub candidates: usize,
    pub removed: usize,
    /// `F` before and after, in bits.
    pub before_bits: f64,
    pub after_bits: f64,
    /// Every evaluated prefix `(k, F − F_before)` in bits.
    pub evaluations: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub settings: Settings,
    pub training_tokens: usize,
    pub groups: usize,
    pub parameters: usize,
    pub epochs: Vec<Epoch>,
    pub removals: Vec<Removal>,
    pub active_groups: usize,
    /// `F` of the converged fit, in bits.
    pub objective_bits: f64,
    pub seconds: f64,
}

pub struct Fit {
    pub posterior: Posterior,
    pub report: Report,
}

/// A family of complete sequences of equal length.
pub fn sequence_family(sequences: &[&[u32]]) -> Result<FamilyInputs, String> {
    let length = sequences.first().ok_or("no sequences")?.len();
    if length == 0 || sequences.iter().any(|s| s.len() != length) {
        return Err("sequences must be nonempty and of equal length".into());
    }
    let mut tokens = Vec::with_capacity(sequences.len() * length);
    let (mut sequence, mut position) = (Vec::new(), Vec::new());
    for (i, s) in sequences.iter().enumerate() {
        tokens.extend_from_slice(s);
        sequence.extend(std::iter::repeat_n(i as u32, length));
        position.extend(0..length as u32);
    }
    Ok(FamilyInputs { rows: tokens.len(), slots: vec![SlotValues::Tokens(tokens)], layout: Some(SequenceLayout { sequence, position }) })
}

/// The native model's distributions as compact fixed-head statistics, made per batch and dropped
/// after use.
struct Teacher {
    prefix: DeviceProgram,
    head: Arc<Head>,
    embedding: Tensor,
    tile_rows: usize,
}

impl Teacher {
    fn new(device: &Device, native: &OperatorProgram, numeric_bytes: usize, tile_rows: usize) -> Result<Self, String> {
        let (flat, _) = mapped_inlined(native)?;
        let head = Arc::new(Head::of(&flat)?);
        let prefix = DeviceProgram::compile_values_bounded(device, &head.prefix(&flat), numeric_bytes)?;
        let embedding = device.upload(head.embedding.view()).map_err(error)?;
        Ok(Self { prefix, head, embedding, tile_rows })
    }

    fn target(&self, family: &FamilyInputs) -> Result<Target, String> {
        let d = self.prefix.device();
        let trace = self.prefix.forward(family)?;
        let hidden = trace.value(self.prefix.hidden())?;
        let (rows, width, classes) = (hidden.rows(), self.head.embedding.ncols(), self.head.embedding.nrows());
        let mut mu = d.zeros(rows, width).map_err(error)?;
        let mut entropy = Vec::with_capacity(rows);
        for start in (0..rows).step_by(self.tile_rows) {
            let n = self.tile_rows.min(rows - start);
            let h = d.rows_of(hidden, start, n).map_err(error)?;
            let mut probabilities = d.zeros(n, classes).map_err(error)?;
            d.gemm(&mut probabilities, 1.0, &h, Op::N, &self.embedding, Op::T, 0.0, Arithmetic::F64).map_err(error)?;
            let stats = d.softmax_stats_rows(&mut probabilities, None).map_err(error)?;
            let mut projected = d.zeros(n, width).map_err(error)?;
            d.gemm(&mut projected, 1.0, &probabilities, Op::N, &self.embedding, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
            d.set_rows(&mut mu, start, &projected).map_err(error)?;
            entropy.extend(stats.into_iter().map(|s| s[1]));
        }
        Ok(Target { mu: Arc::new(mu), entropy, head: Arc::clone(&self.head), scored: None })
    }
}

/// The explanation on the device, scored against the teacher.
struct Student {
    program: DeviceProgram,
    head: ResidentHead,
    trainable: Vec<usize>,
}

impl Student {
    fn new(device: &Device, artifact: &Artifact, trainable: &[usize], numeric_bytes: usize, tile_rows: usize) -> Result<Self, String> {
        let (flat, _) = mapped_inlined(&artifact.program)?;
        let head = Head::of(&flat)?;
        let mut program = DeviceProgram::compile_values_bounded(device, &head.prefix(&flat), numeric_bytes)?;
        program.prepare_dense_parameters(trainable)?;
        Ok(Self { head: ResidentHead::new(device, &head, tile_rows)?, program, trainable: trainable.to_vec() })
    }

    fn load(&mut self, values: &[Array2<f64>]) -> Result<(), String> {
        for (&op, value) in self.trainable.iter().zip(values) {
            let tensor = self.program.device().upload(value.view()).map_err(error)?;
            self.program.replace_dense_parameter(op, tensor)?;
        }
        Ok(())
    }

    /// `Σ_rows KL(p_M ‖ p_θ)` in nats on `family` at the loaded parameters, and with `scale`, the
    /// gradient of `scale` times it in every trainable operator.
    fn score(&self, family: &FamilyInputs, target: &Target, scale: Option<f64>) -> Result<(f64, Option<Vec<Array2<f64>>>), String> {
        let d = self.program.device();
        let trace = self.program.forward(family)?;
        let (losses, seed) = self.head.score(d, trace.value(self.program.hidden())?, target, scale.is_some(), self.program.arithmetic())?;
        let total: f64 = losses.iter().sum();
        if !total.is_finite() {
            return Err("nonfinite explanation divergence".into());
        }
        let Some(scale) = scale else { return Ok((total, None)) };
        let seed = seed.ok_or("missing head cotangent")?;
        let mut scaled = d.zeros(seed.rows(), seed.cols()).map_err(error)?;
        d.axpy(&mut scaled, scale, &seed).map_err(error)?;
        let seeds = BTreeMap::from([(self.program.hidden(), scaled)]);
        let (_, gradients) = self.program.vjp_values_dense(&trace, seeds, &[], &self.trainable, self.program.arithmetic())?;
        let gradients = self
            .trainable
            .iter()
            .map(|op| d.download(gradients.get(op).ok_or("missing parameter gradient")?).map_err(error))
            .collect::<Result<Vec<_>, String>>()?;
        if gradients.iter().any(|g| g.iter().any(|v| !v.is_finite())) {
            return Err("nonfinite parameter gradient".into());
        }
        Ok((total, Some(gradients)))
    }
}

/// The noise seed of step `(epoch, batch)`, or of the removal comparisons' batch.
fn noise_seed(seed: u64, epoch: usize, batch: usize) -> u64 {
    seed.wrapping_add((epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)).wrapping_add((batch as u64).wrapping_mul(0x8CB9_2BA7_2F3D_8DD7))
}

/// Fit the library explanation of `native` to its distributions on the training `sequences` (module
/// note). `native` is the split native program the explanation was built from.
pub fn fit(device: &Device, native: &OperatorProgram, explanation: &Explanation, sequences: &[Vec<u32>], settings: &Settings) -> Result<Fit, String> {
    settings.validate()?;
    let started = Instant::now();
    let tokens: usize = sequences.iter().map(Vec::len).sum();
    let batches: Vec<FamilyInputs> =
        sequences.chunks(settings.batch_sequences).map(|chunk| sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())).collect::<Result<_, _>>()?;
    if batches.len() < 2 {
        return Err("the convergence test needs at least two training batches".into());
    }
    let teacher = Teacher::new(device, native, settings.numeric_bytes, settings.head_tile_rows)?;
    let mut student = Student::new(device, &explanation.artifact, &explanation.trainable, settings.numeric_bytes, settings.head_tile_rows)?;
    let mut posterior = Posterior::new(explanation, tokens)?;
    let parameters = posterior.mean.iter().map(Array2::len).sum();
    let mut mean_moments: Vec<Moment> = posterior.mean.iter().map(|m| Moment::zeros(m.dim())).collect();
    let mut log_sd_moments = mean_moments.clone();
    let mut epochs: Vec<Epoch> = Vec::new();
    let mut removals = Vec::new();
    let mut previous: Option<Vec<f64>> = None;
    let mut step = 0i32;
    for epoch in 0.. {
        let epoch_started = Instant::now();
        let mut estimates = Vec::with_capacity(batches.len());
        let (mut data_sum, mut description_sum, mut kl_sum) = (0.0, 0.0, 0.0);
        for (b, batch) in batches.iter().enumerate() {
            let target = teacher.target(batch)?;
            let (theta, noise) = posterior.sample(noise_seed(settings.seed, epoch + 1, b));
            student.load(&theta)?;
            let scale = tokens as f64 / batch.rows as f64;
            let (divergence, gradients) = student.score(batch, &target, Some(scale))?;
            let gradients = gradients.ok_or("missing gradients")?;
            let description = posterior.description();
            estimates.push(scale * divergence + description);
            data_sum += scale * divergence;
            description_sum += description;
            kl_sum += divergence / batch.rows as f64;
            step += 1;
            let results = posterior.derivatives(&gradients, &noise);
            let active = posterior.active.clone();
            let Posterior { mean, log_sd, membership, .. } = &mut posterior;
            mean.par_iter_mut()
                .zip(log_sd.par_iter_mut())
                .zip(mean_moments.par_iter_mut())
                .zip(log_sd_moments.par_iter_mut())
                .zip(results.par_iter())
                .zip(membership.par_iter())
                .for_each(|(((((mean, log_sd), mm), sm), (gm, gs)), membership)| {
                    mm.step(mean, gm, membership, &active, settings.mean_step, settings, step);
                    sm.step(log_sd, gs, membership, &active, settings.log_sd_step, settings, step);
                });
        }
        let count = batches.len() as f64;
        let (improvement, standard_error) = match &previous {
            Some(before) => {
                let differences: Vec<f64> = before.iter().zip(&estimates).map(|(a, b)| a - b).collect();
                let mean = differences.iter().sum::<f64>() / count;
                let variance = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (count - 1.0);
                (Some(mean), Some((variance / count).sqrt()))
            }
            None => (None, None),
        };
        let to_bits = |nats: f64| nats / std::f64::consts::LN_2;
        let record = Epoch {
            epoch,
            objective_bits: to_bits(estimates.iter().sum::<f64>() / count),
            data_bits: to_bits(data_sum / count),
            description_bits: to_bits(description_sum / count),
            improvement_bits: improvement.map(to_bits),
            standard_error_bits: standard_error.map(to_bits),
            active_groups: posterior.active.iter().filter(|a| **a).count(),
            mean_token_kl: kl_sum / count,
            seconds: epoch_started.elapsed().as_secs_f64(),
        };
        log::info!("library fit epoch {epoch}: {record:?}");
        epochs.push(record);
        previous = Some(estimates);
        let converged = matches!((improvement, standard_error), (Some(i), Some(se)) if i <= se);
        if !converged {
            continue;
        }
        let removal = remove(&teacher, &mut student, &mut posterior, &batches, settings)?;
        log::info!("library removal after epoch {epoch}: {} of {} candidates", removal.removed, removal.candidates);
        let removed = removal.removed;
        removals.push(removal);
        if removed == 0 {
            break;
        }
        // The objective changed discretely: convergence is judged afresh.
        previous = None;
    }
    let objective_bits = removals.last().map_or(f64::NAN, |r| r.after_bits);
    Ok(Fit {
        report: Report {
            settings: settings.clone(),
            training_tokens: tokens,
            groups: explanation.groups.len(),
            parameters,
            epochs,
            removals,
            active_groups: posterior.active.iter().filter(|a| **a).count(),
            objective_bits,
            seconds: started.elapsed().as_secs_f64(),
        },
        posterior,
    })
}

/// `E_q[D]` over every training batch, one weight sample per batch from the removal seeds, with the
/// groups `removed` (and the already removed ones) zeroed, in nats.
fn expected_divergence(teacher: &Teacher, student: &mut Student, posterior: &Posterior, batches: &[FamilyInputs], removed: &[usize], settings: &Settings) -> Result<f64, String> {
    let mut trial = posterior.clone();
    trial.remove(removed);
    let mut total = 0.0;
    for (b, batch) in batches.iter().enumerate() {
        let target = teacher.target(batch)?;
        // Removal zeroes entries, so the remaining entries see the same noise as the full posterior.
        let (theta, _) = trial.sample(noise_seed(settings.seed, 0, b));
        student.load(&theta)?;
        total += student.score(batch, &target, None)?.0;
    }
    Ok(total)
}

/// The removal step (module note): the longest prefix of the active groups in increasing divergence
/// whose removal does not increase `F`.
fn remove(teacher: &Teacher, student: &mut Student, posterior: &mut Posterior, batches: &[FamilyInputs], settings: &Settings) -> Result<Removal, String> {
    let divergences = posterior.divergences();
    let mut order: Vec<usize> = (0..divergences.len()).filter(|g| posterior.active[*g]).collect();
    order.sort_by(|a, b| divergences[*a].total_cmp(&divergences[*b]));
    let description = posterior.description();
    let base = expected_divergence(teacher, student, posterior, batches, &[], settings)? + description;
    // `F` with the first `k` candidates removed, minus `F` as it is.
    let mut evaluations: Vec<(usize, f64)> = Vec::new();
    let mut change = |k: usize| -> Result<f64, String> {
        if let Some((_, c)) = evaluations.iter().find(|(at, _)| *at == k) {
            return Ok(*c);
        }
        let saved: f64 = order[..k].iter().map(|g| divergences[*g] + LITERAL_NATS).sum();
        let c = expected_divergence(teacher, student, posterior, batches, &order[..k], settings)? + description - saved - base;
        evaluations.push((k, c));
        Ok(c)
    };
    let (mut low, mut high) = (0usize, order.len());
    if change(high)? <= 0.0 {
        low = high;
    } else {
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if change(middle)? <= 0.0 {
                low = middle;
            } else {
                high = middle;
            }
        }
    }
    let after = base + if low > 0 { change(low)? } else { 0.0 };
    posterior.remove(&order[..low]);
    let to_bits = |nats: f64| nats / std::f64::consts::LN_2;
    Ok(Removal {
        candidates: order.len(),
        removed: low,
        before_bits: to_bits(base),
        after_bits: to_bits(after),
        evaluations: evaluations.into_iter().map(|(k, c)| (k, to_bits(c))).collect(),
    })
}

// ----------------------------------------------------------------------------- the reported artifact

/// The explanation at the posterior mean: each library operator holds `μ`, and the blocks only
/// removed groups touch are absent (no literals).
pub fn posterior_mean(explanation: &Explanation, posterior: &Posterior) -> Result<Artifact, String> {
    let mut artifact = explanation.artifact.clone();
    for ((op, values), membership) in explanation.trainable.iter().zip(posterior.means()).zip(&posterior.membership) {
        let source = &artifact.program.operators[*op];
        let (rows, cols) = (source.rows.clone(), source.cols.clone());
        let mut present = Array2::from_elem((rows.group_count(), cols.group_count()), false);
        for r in 0..rows.group_count() {
            for c in 0..cols.group_count() {
                present[[r, c]] = rows.range(r).any(|i| cols.range(c).any(|j| posterior.active[membership[[i, j]] as usize]));
            }
        }
        let precision = exact_precision(values.iter().copied()).map_err(error)?;
        let operator = Operator::blocks(source.name.clone(), rows, cols, values, present, precision, source.provenance.clone()).map_err(error)?;
        if !matches!(operator.body, OperatorBody::Dense { .. }) {
            return Err("a library operator lost its dense body".into());
        }
        artifact.program.operators[*op] = Arc::new(operator);
    }
    artifact.program.interfaces().map_err(error)?;
    Ok(artifact)
}

/// Mean `KL(p_M ‖ p_P)` per token, in nats, of `artifact`'s explanation on `sequences` (complete,
/// equal length), `batch_sequences` at a time.
pub fn mean_divergence(
    device: &Device,
    native: &OperatorProgram,
    artifact: &Artifact,
    sequences: &[Vec<u32>],
    batch_sequences: usize,
    numeric_bytes: usize,
    tile_rows: usize,
) -> Result<f64, String> {
    if batch_sequences == 0 || sequences.is_empty() {
        return Err("no held-out sequences".into());
    }
    let teacher = Teacher::new(device, native, numeric_bytes, tile_rows)?;
    let student = Student::new(device, artifact, &[], numeric_bytes, tile_rows)?;
    let (mut total, mut rows) = (0.0, 0usize);
    for chunk in sequences.chunks(batch_sequences) {
        let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let target = teacher.target(&family)?;
        total += student.score(&family, &target, None)?.0;
        rows += family.rows;
    }
    Ok(total / rows as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        run_check::{layer_nodes, split_sites},
    };

    /// The tiny two-layer decoder export with MLP law `law`: its split program, layers, and its six
    /// sequences of twelve tokens.
    fn tiny(tag: &str, law: &str) -> (OperatorProgram, Vec<LayerNodes>, FamilyInputs, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let path = dir.join("export.json");
        let mut record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        record["config"]["mlp_act"] = law.into();
        std::fs::write(&path, record.to_string()).unwrap();
        let imported = import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let family = imported.contract.family;
        let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, layers, family, sequences)
    }

    #[test]
    fn the_starting_library_is_the_native_relu_model() {
        let (native, layers, family, _) = tiny("library_start", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        explanation.artifact.validate_coverage(&native).unwrap();
        assert_eq!(explanation.artifact.controls.len(), 2, "every MLP keeps its uniform-scale control");
        assert_eq!(explanation.artifact.blocks.len(), 2 * (2 + 1), "two heads and one MLP per layer");
        let expected = native.execute(&family, false).unwrap().values[native.output].clone();
        let actual = explanation.artifact.execute(&family).unwrap().values[explanation.artifact.program.output].clone();
        let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let difference = expected.iter().zip(&actual).fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(difference <= 1e-12 * scale, "the starting library differs from the native model by {difference}");
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let cells: usize = explanation.groups.iter().flat_map(|g| &g.cells).map(|c| c.rows.len() * c.cols.len()).sum();
        assert_eq!(cells, posterior.mean.iter().map(Array2::len).sum::<usize>(), "the groups partition the parameters");
    }

    #[test]
    fn description_derivatives_are_those_of_the_divergence() {
        let (native, layers, _, _) = tiny("library_derivatives", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 0.6 * (rng.random::<f64>() - 0.5));
        }
        let zeros: Vec<Array2<f64>> = posterior.mean.iter().map(|m| Array2::zeros(m.dim())).collect();
        let derivatives = posterior.derivatives(&zeros, &zeros);
        let last = posterior.mean.len() - 1;
        for (i, at) in [(0, (0, 0)), (1, (3, 5)), (2, (1, 2)), (last, (2, 7))] {
            let h = 1e-6;
            let central = |posterior: &Posterior, field: fn(&mut Posterior) -> &mut Vec<Array2<f64>>| {
                let (mut up, mut down) = (posterior.clone(), posterior.clone());
                field(&mut up)[i][at] += h;
                field(&mut down)[i][at] -= h;
                (up.description() - down.description()) / (2.0 * h)
            };
            let mean = central(&posterior, |p| &mut p.mean);
            let log_sd = central(&posterior, |p| &mut p.log_sd);
            let (analytic_mean, analytic_log_sd) = (derivatives[i].0[at], derivatives[i].1[at]);
            assert!((mean - analytic_mean).abs() <= 1e-5 * (1.0 + mean.abs()), "μ derivative {analytic_mean} against {mean}");
            assert!((log_sd - analytic_log_sd).abs() <= 1e-5 * (1.0 + log_sd.abs()), "ln σ derivative {analytic_log_sd} against {log_sd}");
        }
    }

    #[test]
    fn a_group_at_its_prior_costs_nothing() {
        let (native, layers, _, _) = tiny("library_prior", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let group = &explanation.groups[0];
        let position = |op: usize| explanation.trainable.iter().position(|t| *t == op).unwrap();
        for cell in &group.cells {
            let i = position(cell.operator);
            for &row in &cell.rows {
                for col in cell.cols.clone() {
                    posterior.mean[i][[row, col]] = 0.0;
                    posterior.log_sd[i][[row, col]] = -1.5;
                }
            }
        }
        let divergences = posterior.divergences();
        assert!(divergences[0].abs() < 1e-12, "an uninformed group at its prior costs {}", divergences[0]);
        assert!(divergences[1] > 0.0);
    }

    #[test]
    fn the_fit_converges_removes_and_reports_its_posterior_mean() {
        let (native, layers, _, sequences) = tiny("library_fit", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = Settings {
            batch_sequences: 2,
            mean_step: 0.01,
            log_sd_step: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            seed: 3,
            numeric_bytes: 1 << 26,
            head_tile_rows: 64,
        };
        let device = Device::host();
        let fit = fit(&device, &native, &explanation, &sequences, &settings).unwrap();
        let report = &fit.report;
        assert_eq!(report.training_tokens, 72);
        assert_eq!(report.removals.last().unwrap().removed, 0, "the fit ends when no removal is accepted");
        assert!(report.removals.iter().all(|r| r.after_bits <= r.before_bits), "a removal never increases the objective");
        assert!(report.epochs.iter().all(|e| e.objective_bits.is_finite()));
        let artifact = posterior_mean(&explanation, &fit.posterior).unwrap();
        artifact.validate_coverage(&native).unwrap();
        let (start, end) = (explanation.artifact.program.real_count(), artifact.program.real_count());
        let removed: usize = explanation
            .groups
            .iter()
            .zip(&fit.posterior.active)
            .filter(|(_, active)| !**active)
            .flat_map(|(g, _)| &g.cells)
            .filter(|c| explanation.artifact.program.operators[c.operator].rows.group_count() > 1)
            .map(|c| c.rows.len() * c.cols.len())
            .sum();
        assert!(end <= start && start - end >= removed.min(start), "removed groups are absent blocks: {start} → {end}");
        let kl = mean_divergence(&device, &native, &artifact, &sequences, 2, 1 << 26, 64).unwrap();
        assert!(kl.is_finite() && kl >= 0.0);
    }
}
