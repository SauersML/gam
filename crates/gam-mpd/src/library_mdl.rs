//! The explanation as a library of learned functions, fitted end to end by variational minimum
//! description length on interchange experiments of causal abstraction (#2951).
//!
//! # The explanation
//!
//! Every attention head and every MLP of the native model `M` is replaced by a block of learned
//! functions that runs on the explanation's own vectors. `M`'s embedding, norms, attention output
//! projection, final norm and unembedding stay.
//!
//! * A head's block reads its layer's normed stream `x` and writes the head's read (the attention
//!   output projection's input). Its function attends with its own query, key and value maps
//!   `q = Q x`, `k = K x`, `v = V x` at the native rotary angles, scale and causal mask; where `M`
//!   norms each head's query and key (Qwen3), `q = γ_q ⊙ rms(Q x)` and `k = γ_k ⊙ rms(K x)` with
//!   the head's own copy of `M`'s gains `γ`, which are part of the law and not trained. Heads that
//!   share a key and value in `M` (grouped-query attention) each have their own copies.
//! * An MLP's block reads its layer's second normed stream and writes the MLP's output. Its
//!   functions are `f_i(x) = φ(g_i·x + c_i) u_i`, with the native pointwise law `φ`,
//!   a gate direction `g_i`, a gate bias `c_i` and an output `u_i`; for a gated MLP (SwiGLU on
//!   Qwen3) `f_i(x) = φ(g_i·x) (b_i·x) u_i` with an up direction `b_i`, and biases only where `M`
//!   has them.
//!
//! The library starts at `M`: native neuron `i` is function `i` with its original
//! activation and coefficients, and a head is its own function.
//!
//! # The code length
//!
//! The library's parameters `θ` are partitioned into prior groups `G`: a head's rotary plane (the
//! query and key rows of the plane), a head's value coordinate (one value row), an MLP function's
//! gate `(g_i, c_i)`, its up direction `b_i` when gated, and its output `u_i`. With the posterior `q(θ) = Π_j N(μ_j, σ_j²)` and the
//! prior `p(θ_G) = N(0, v_G I)`, the code length in nats is
//!
//! `F = E_q[D(θ)] + Σ_G KL(q_G ‖ p_G) + Σ_{active G} ½ ln |G|`,
//!
//! the bits-back code of the training data given the explanation plus the explanation. `D(θ)`
//! sums `KL(M_e ‖ P_e)` of the next-token distributions over every token of every training
//! experiment `e` (below), so the data term's weight is the number of scored tokens and no
//! tradeoff weight exists. Each prior variance is chosen by empirical Bayes in closed form,
//! `v_G = (1/|G|) Σ_{j∈G} (μ_j² + σ_j²)`, which makes `KL(q_G ‖ p_G) = ½ (|G| ln v_G − Σ_{j∈G}
//! ln σ_j²)` with derivatives `μ_j / v_G` in `μ_j` and `σ_j² / v_G − 1` in `ln σ_j`. An active
//! group sends its variance at the precision of a parameter estimated from `|G|` values,
//! `½ log2 |G|` bits (the two-part code's parameter precision). A group the data does not inform
//! sits at its prior with zero divergence and posterior mean zero, so the null is recovered;
//! removing it from the explanation is a discrete step of the same `F`.
//!
//! Weight noise costs data only on the tokens where a function is active, so a ReLU-gated function
//! that is exactly zero elsewhere can keep imprecise, cheap weights: the code length rewards
//! functions that are inactive on most inputs without any sparsity penalty, when the native
//! activation is ReLU. This inactive-region argument does not apply to GELU or SiLU.
//!
//! # The experiments
//!
//! The data are interchange experiments (`interchange`): per training base sequence, one clean
//! experiment and one patched experiment (a read patch of one of the explanation's read variables
//! or the complement patch of one block, drawn uniformly over their union), each with its own cut
//! `ℓ` drawn uniformly on `1..=L`. A base's source is another training sequence, drawn uniformly.
//! The batches, sources and draws are made once from the seed, so the training data are one fixed
//! set of experiments while the explanation's read variables stay; a removal changes the
//! variables, and the draws are made again from the same seeds over the variables that remain.
//! The patch directions are data: each step takes them at the posterior mean `μ` (before the
//! weight sample is loaded), and no gradient passes through them.
//!
//! # The fit
//!
//! Each step draws one weight sample `θ = μ + σ ⊙ ε` (`ε` standard normal), runs a batch's
//! experiments, and takes one Adam step in `(μ, ln σ)` on the batch estimate
//! `(N/n) D_batch(θ) + Σ_G KL_G`, `N` the scored tokens of every training experiment and `n` the
//! batch's. The posterior stays on the device through an epoch (`device_posterior`): the sample is
//! written into the explanation's program, the gradient stays where the reverse pass left it, and
//! Adam's step and the groups' divergences run there; the host holds it between epochs, for the
//! held-out evaluation, the checkpoint and the removal step. An epoch visits every training batch
//! once, in a fixed order. The continuous fit
//! has converged when an epoch's mean improvement of the per-batch objective estimate over the
//! previous epoch, paired by batch, is smaller than its standard error.
//!
//! A converged fit then proposes removing groups in increasing order of their divergence `KL_G`
//! (their information content). Prefix lengths are searched by bisection, `O(log n)` full training
//! evaluations for `n` active groups, and the longest evaluated prefix that does not increase the
//! sampled objective is removed. The objective is estimated over the whole training set with one
//! common weight sample per batch for both sides of every comparison. Experiment identities and
//! patch directions are held at the posterior before removal throughout that round. The order and
//! the search are a proposal; acceptance never increases this round's sampled objective. After an
//! accepted removal, training redefines the experiments over the surviving variables, so this is
//! not a claim of monotonicity on one fixed dataset across rounds. Removal effects
//! can cancel, so the objective need not be monotone in the prefix length and the search may miss a
//! longer acceptable prefix; the next round, after the continuous fit converges again, proposes
//! again. The fit alternates converging and removing until no removal is accepted.
//!
//! # Evaluation
//!
//! After every epoch the held-out sequences are scored ([`HeldOut`]): per base, at every cut, one
//! clean and one patched experiment (sources among the held-out sequences). `F` per token is the
//! held-out data term at one weight sample per batch plus the description spread over the training
//! experiments' scored tokens; the divergences per experiment kind are taken at the posterior mean.
//! Per layer it counts the surviving heads and MLP functions and, per token, the functions whose
//! activation is not zero and those whose activation exceeds its posterior noise scale.
//!
//! The explanation is reported at the posterior mean `μ` ([`posterior_mean`]); removed groups are
//! absent only when they fill complete interface blocks.

use crate::{
    artifact::{Argument, Artifact, Callee},
    device_posterior::{Adam, DevicePosterior},
    interchange::{self, Batch, Experiment, Interchange, Patch, ReadVariable},
    operator_program::{
        FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, Provenance, Rule, SequenceLayout, SlotValues,
        exact_precision,
    },
    run_check::{LayerNodes, head_projection},
};
use gam_gpu::tensor::{Device, Op, Tensor};
use ndarray::{Array2, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, f64::consts::LN_2, io::Write, ops::Range, path::Path, sync::Arc, time::Instant};

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
/// operators, its prior groups and its layers.
#[derive(Clone, Debug)]
pub struct Explanation {
    pub artifact: Artifact,
    /// The library's operators of `artifact.program`, ascending.
    pub trainable: Vec<usize>,
    pub groups: Vec<Group>,
    pub layers: Vec<Layer>,
}

/// One layer of the explanation: its native sites (`run_check::layer_nodes`) and the prior groups
/// of its functions.
#[derive(Clone, Debug)]
pub struct Layer {
    pub sites: LayerNodes,
    /// Per head, its planes' groups and its value coordinates' groups.
    pub heads: Vec<(Vec<usize>, Vec<usize>)>,
    /// Per MLP function, its groups: its gate's, its up direction's when gated, and its output's.
    pub functions: Vec<Vec<usize>>,
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

/// The head norm on `node` (a head's query or key; `run_check::head_projection`): its RMS norm's
/// `ε` and its gain operator.
fn head_norm(native: &OperatorProgram, node: usize) -> Option<(f64, Arc<Operator>)> {
    if head_projection(native, node) == node {
        return None;
    }
    match &native.nodes[node] {
        Node::Affine { terms, .. } => match native.nodes[terms[0].0] {
            Node::RmsNorm { epsilon, .. } => Some((epsilon, Arc::clone(&native.operators[terms[0].1]))),
            _ => None,
        },
        _ => None,
    }
}

/// The native operator and bias of the affine node `node` reading `input` alone.
fn affine_map(native: &OperatorProgram, node: usize, input: usize) -> Result<(Arc<Operator>, Option<Array2<f64>>), String> {
    match &native.nodes[node] {
        Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == input => {
            Ok((Arc::clone(&native.operators[terms[0].1]), bias.map(|b| native.operators[b].matrix())))
        }
        other => Err(format!("node {node} is not an affine map of node {input}: {other:?}")),
    }
}

/// The library block of a gated MLP `down(φ(A x + c) ⊙ (B x + e))` (SwiGLU on Qwen3): functions
/// `f_i(x) = φ(a_i·x + c_i) (b_i·x + e_i) u_i` with `M`'s law `φ`, its gate `a_i`, up direction
/// `b_i` and output `u_i`, and the biases `c_i`, `e_i` only where `M` has them.
fn gated_mlp(native: &OperatorProgram, artifact: Artifact, layer: &LayerNodes, l: usize, left: usize, right: usize) -> Result<Artifact, String> {
    let Node::Pointwise { input: gate_pre, laws } = &native.nodes[left] else {
        return Err(format!("layer {l}: the gated product's left factor is not one pointwise law"));
    };
    if laws.iter().any(|law| *law != laws[0]) {
        return Err(format!("layer {l}: the MLP's units have different laws"));
    }
    let x = layer.normed;
    let ((gate, gate_bias), (up, up_bias)) = (affine_map(native, *gate_pre, x)?, affine_map(native, right, x)?);
    let down = single_map(native, layer.mlp, layer.active)?;
    let units = up.rows.clone();
    if gate.rows != units || down.cols != units {
        return Err(format!("layer {l}: the gate, up and down maps disagree on the MLP's units"));
    }
    let name = format!("library.l{l}.mlp");
    let base = artifact.program.operators.len();
    let mut operators = vec![
        library_operator(&format!("{name}.gate"), units.clone(), gate.cols.clone(), gate.matrix(), &gate.name)?,
        library_operator(&format!("{name}.up"), units.clone(), up.cols.clone(), up.matrix(), &up.name)?,
        library_operator(&format!("{name}.out"), down.rows.clone(), units.clone(), down.matrix(), &down.name)?,
    ];
    let mut bias = |values: Option<Array2<f64>>, part: &str, source: &str| -> Result<Option<usize>, String> {
        let Some(values) = values else { return Ok(None) };
        operators.push(library_operator(&format!("{name}.{part}_bias"), units.clone(), Interface::constant(), values, source)?);
        Ok(Some(base + operators.len() - 1))
    };
    let (gate_bias, up_bias) = (bias(gate_bias, "gate", &gate.name)?, bias(up_bias, "up", &up.name)?);
    let rule = Rule {
        name: name.clone(),
        inputs: vec![gate.cols.clone()],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, base)], bias: gate_bias },
            Node::Pointwise { input: 1, laws: laws.clone() },
            Node::Affine { terms: vec![(0, base + 1)], bias: up_bias },
            Node::Hadamard { left: 2, right: 3 },
            Node::Affine { terms: vec![(4, base + 2)], bias: None },
        ],
        output: 5,
    };
    artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(x)], layer.mlp, operators)
}

/// The operator named `name`, when the program has one (a part a law may lack: a bias, an up map).
fn operator_named(program: &OperatorProgram, name: &str) -> Option<usize> {
    program.operators.iter().position(|op| op.name == name)
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
            let (query_norm, key_norm) = (head_norm(native, query), head_norm(native, key));
            let (query, key) = (head_projection(native, query), head_projection(native, key));
            let (q, k, v) = (single_map(native, query, x)?, single_map(native, key, x)?, single_map(native, value, x)?);
            if q.rows.width() != k.rows.width() {
                return Err(format!("layer {l} head {h}: query and key widths differ"));
            }
            // Query and key coordinates in their own groups, so a removed plane is an absent block.
            let coordinates = Interface::uniform(q.rows.width(), 1, LabelKind::Unit, 0).map_err(error)?;
            let name = format!("library.l{l}.h{h}");
            let base = artifact.program.operators.len();
            let mut operators = vec![
                library_operator(&format!("{name}.q"), coordinates.clone(), q.cols.clone(), q.matrix(), &q.name)?,
                library_operator(&format!("{name}.k"), coordinates.clone(), k.cols.clone(), k.matrix(), &k.name)?,
                library_operator(&format!("{name}.v"), v.rows.clone(), v.cols.clone(), v.matrix(), &v.name)?,
            ];
            let mut nodes = vec![
                Node::Param { index: 0 },
                Node::Affine { terms: vec![(0, base)], bias: None },
                Node::Affine { terms: vec![(0, base + 1)], bias: None },
                Node::Affine { terms: vec![(0, base + 2)], bias: None },
            ];
            // A head norm: the RMS norm of the projection, then the head's own copy of `M`'s gain.
            let mut normed = |projection: usize, norm: Option<(f64, Arc<Operator>)>, part: &str| -> Result<usize, String> {
                let Some((epsilon, gain)) = norm else { return Ok(projection) };
                let values = gain.diagonal().ok_or_else(|| format!("{}: a head norm's gain is not diagonal", gain.name))?;
                let precision = exact_precision(values.iter().copied()).map_err(error)?;
                nodes.push(Node::RmsNorm { input: projection, epsilon });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, base + operators.len())], bias: None });
                operators.push(Operator::diag(format!("{name}.{part}_gain"), coordinates.clone(), values, precision, gain.provenance.clone()).map_err(error)?);
                Ok(nodes.len() - 1)
            };
            let (query_node, key_node) = (normed(1, query_norm, "q")?, normed(2, key_norm, "k")?);
            nodes.push(Node::Attend { query: query_node, key: key_node, value: 3, scale, rotary, causal });
            let rule = Rule { name: name.clone(), inputs: vec![q.cols.clone()], output: nodes.len() - 1, nodes };
            artifact = artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(x)], read, operators)?;
            let pairs: Vec<Vec<usize>> = match rotary {
                Some(r) => {
                    let pairs = r.pairs();
                    let rotated: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    pairs.iter().map(|&(a, b)| vec![a, b]).chain((0..q.rows.width()).filter(|c| !rotated.contains(c)).map(|c| vec![c])).collect()
                }
                None => (0..q.rows.width()).map(|c| vec![c]).collect(),
            };
            planes.push((l, name, pairs, v.rows.width()));
        }
        if let Node::Hadamard { left, right } = native.nodes[layer.active] {
            artifact = gated_mlp(native, artifact, layer, l, left, right)?;
            continue;
        }
        let Node::Pointwise { input: pre, laws } = &native.nodes[layer.active] else {
            return Err(format!("layer {l}: the MLP activation is neither one pointwise law nor a gated product"));
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
                Node::Pointwise { input: 1, laws: laws.clone() },
                Node::Affine { terms: vec![(2, base + 2)], bias: None },
            ],
            output: 3,
        };
        artifact = artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(layer.normed)], layer.mlp, operators)?;
    }
    // Groups by the final operator indices (each replacement renumbers the operators).
    let program = &artifact.program;
    let mut groups = Vec::new();
    let mut trainable = Vec::new();
    let mut out: Vec<Layer> = layers.iter().map(|sites| Layer { sites: sites.clone(), heads: Vec::new(), functions: Vec::new() }).collect();
    for (l, name, pairs, values) in &planes {
        let (q, k, v) = (index_of(program, &format!("{name}.q"))?, index_of(program, &format!("{name}.k"))?, index_of(program, &format!("{name}.v"))?);
        let d = program.operators[q].cols.width();
        let first = groups.len();
        for (p, rows) in pairs.iter().enumerate() {
            groups.push(Group {
                name: format!("{name}.plane{p}"),
                cells: vec![Cells { operator: q, rows: rows.clone(), cols: 0..d }, Cells { operator: k, rows: rows.clone(), cols: 0..d }],
            });
        }
        for j in 0..*values {
            groups.push(Group { name: format!("{name}.value{j}"), cells: vec![Cells { operator: v, rows: vec![j], cols: 0..d }] });
        }
        out[*l].heads.push(((first..first + pairs.len()).collect(), (first + pairs.len()..groups.len()).collect()));
        trainable.extend([q, k, v]);
    }
    for (l, layer) in out.iter_mut().enumerate() {
        let name = format!("library.l{l}.mlp");
        let output = index_of(program, &format!("{name}.out"))?;
        let (functions, d) = (program.operators[output].cols.width(), program.operators[output].rows.width());
        layer.functions = vec![Vec::new(); functions];
        // Per function its gate (with its bias), its up direction when gated (likewise), its output.
        for part in ["gate", "up"] {
            let Some(map) = operator_named(program, &format!("{name}.{part}")) else { continue };
            let bias = operator_named(program, &format!("{name}.{part}_bias"));
            for (i, function) in layer.functions.iter_mut().enumerate() {
                let mut cells = vec![Cells { operator: map, rows: vec![i], cols: 0..d }];
                cells.extend(bias.map(|b| Cells { operator: b, rows: vec![i], cols: 0..1 }));
                function.push(groups.len());
                groups.push(Group { name: format!("{name}.f{i}.{part}"), cells });
            }
            trainable.push(map);
            trainable.extend(bias);
        }
        for (i, function) in layer.functions.iter_mut().enumerate() {
            function.push(groups.len());
            groups.push(Group { name: format!("{name}.f{i}.out"), cells: vec![Cells { operator: output, rows: (0..d).collect(), cols: i..i + 1 }] });
        }
        trainable.push(output);
    }
    trainable.sort_unstable();
    Ok(Explanation { artifact, trainable, groups, layers: out })
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

    /// Per group, `KL(q_G ‖ p_G)` plus its variance's `½ ln |G|`, in nats (zero for a removed
    /// group).
    pub fn costs(&self) -> Vec<f64> {
        self.moments().iter().zip(&self.active).map(|(m, active)| if *active { m.divergence() + 0.5 * m.count.ln() } else { 0.0 }).collect()
    }

    /// `Σ_G KL(q_G ‖ p_G)` plus the active groups' variances, in nats.
    pub fn description(&self) -> f64 {
        self.costs().iter().sum()
    }

    /// Whether an entry of rows `rows` of trainable operator `i` is in an active group.
    fn holds(&self, i: usize, rows: Range<usize>) -> bool {
        self.membership[i].slice(s![rows, ..]).iter().any(|g| self.active[*g as usize])
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

// ------------------------------------------------------------------------------------- the fit

/// The optimizer's step sizes and the run's resources (none of them is part of the objective).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Base sequences per step.
    pub batch_sequences: usize,
    /// Adam step sizes in `μ` and in `ln σ`.
    pub mean_step: f64,
    pub log_sd_step: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    /// The seed of the weight noise and of the experiments' draws.
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

/// One layer's survivors and activity on the held-out sequences at the posterior mean.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerCount {
    /// Heads with an active value coordinate; active rotary planes and value coordinates.
    pub heads: usize,
    pub planes: usize,
    pub values: usize,
    /// MLP functions whose groups are all active.
    pub functions: usize,
    /// Mean per token of the surviving functions whose activation `a = φ(z) y` is not zero, and of
    /// those whose activation exceeds its posterior noise scale `√((φ′(z) y)² s_z² + φ(z)² s_y²)`,
    /// `s_z² = Σ_j σ²_{g_j} x_j² + σ²_c` the posterior variance of the gate value `z = g·x + c` at
    /// the token's input `x` and `s_y²` likewise of a gated law's up value `y = b·x + e` (an
    /// ungated law has `y = 1`, `s_y = 0`).
    pub nonzero_per_token: f64,
    pub resolved_per_token: f64,
}

/// The held-out evaluation (module note). Divergences are `KL(M_e ‖ P_e)` per scored token, in
/// bits; a kind no experiment drew is none.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeldOut {
    /// `F` per scored token: the held-out data term at one weight sample per batch plus the
    /// description over the training experiments' scored tokens.
    pub objective_bits_per_token: f64,
    pub data_bits_per_token: f64,
    /// `Σ_G KL(q_G ‖ p_G)` and the active groups' variances `Σ ½ log2 |G|`, in bits.
    pub divergence_bits: f64,
    pub variance_bits: f64,
    /// At the posterior mean: clean and patched experiments per cut `ℓ = 1..=L` (`ℓ = L` is the
    /// explanation alone), and the patched ones by kind.
    pub clean: Vec<Option<f64>>,
    pub patched: Vec<Option<f64>>,
    pub read_patch: Option<f64>,
    pub complement_patch: Option<f64>,
    pub layers: Vec<LayerCount>,
}

/// One epoch of the continuous fit.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// `KL(M_e ‖ P_e)` per scored token at the weight samples over the clean and the patched
    /// training experiments, in bits.
    pub clean_bits_per_token: f64,
    pub patched_bits_per_token: f64,
    pub seconds: f64,
    /// The held-out evaluation after the epoch's steps.
    pub held_out: HeldOut,
}

/// One removal step.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    pub training_sequences: usize,
    /// The scored tokens of every training experiment, `N`.
    pub scored_tokens: usize,
    pub groups: usize,
    pub parameters: usize,
    /// The held-out evaluation at the starting point.
    pub start: HeldOut,
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

/// A batch of experiments: its base sequences, each base's source (indices into the sequences),
/// and the seed of its experiments' draws.
#[derive(Clone, Debug)]
struct Draw {
    bases: Vec<usize>,
    sources: Vec<usize>,
    seed: u64,
}

impl Draw {
    fn batch(&self, sequences: &[Vec<u32>]) -> Result<Batch, String> {
        let pick = |indices: &[usize]| indices.iter().map(|i| sequences[*i].clone()).collect();
        Batch::new(pick(&self.bases), pick(&self.sources))
    }

    /// Per base one clean and one patched experiment over `variables` read variables
    /// (`interchange::sample`), from the batch's seed.
    fn experiments(&self, layers: usize, variables: usize) -> Vec<Experiment> {
        interchange::sample(&mut StdRng::seed_from_u64(self.seed), self.bases.len(), layers, variables)
    }
}

/// The `count` sequences in order, in batches of `size` bases, each base's source drawn uniformly
/// among the other sequences, all from `seed`.
fn draws(count: usize, size: usize, seed: u64) -> Result<Vec<Draw>, String> {
    if count < 2 || size == 0 {
        return Err("a source needs another sequence, and batches must be nonempty".into());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let all: Vec<usize> = (0..count).collect();
    Ok(all
        .chunks(size)
        .map(|bases| {
            let sources = bases
                .iter()
                .map(|&i| {
                    let j = rng.random_range(0..count - 1);
                    if j >= i { j + 1 } else { j }
                })
                .collect();
            Draw { bases: bases.to_vec(), sources, seed: rng.random() }
        })
        .collect())
}

/// An affine node of an MLP in the explanation's flat program: the node, and its operator and
/// bias operator.
struct Map {
    node: usize,
    operator: usize,
    bias: Option<usize>,
}

impl Map {
    /// The node of `flat` applying operator `name` (and its bias `{name}_bias`, when it exists) to
    /// one input, and that input.
    fn of(flat: &OperatorProgram, name: &str) -> Result<(Self, usize), String> {
        let (operator, bias) = (index_of(flat, name)?, operator_named(flat, &format!("{name}_bias")));
        let found = flat.nodes.iter().enumerate().find_map(|(n, node)| match node {
            Node::Affine { terms, bias: b } if *b == bias && terms.len() == 1 && terms[0].1 == operator => Some((n, terms[0].0)),
            _ => None,
        });
        let (node, input) = found.ok_or_else(|| format!("no node applies {name}"))?;
        Ok((Self { node, operator, bias }, input))
    }
}

/// An MLP's nodes in the explanation's flat program: the gate's input `x`, the gate `z = g·x + c`,
/// the activation `φ(z)` and its law, and for a gated law the up value `y = b·x + e`.
struct Mlp {
    input: usize,
    gate: Map,
    activation: usize,
    law: Law,
    up: Option<Map>,
}

impl Mlp {
    fn of(flat: &OperatorProgram, l: usize) -> Result<Self, String> {
        let name = format!("library.l{l}.mlp");
        let (gate, input) = Map::of(flat, &format!("{name}.gate"))?;
        let found = flat.nodes.iter().enumerate().find_map(|(n, node)| match node {
            Node::Pointwise { input, laws } if *input == gate.node => laws.first().map(|law| (n, *law)),
            _ => None,
        });
        let (activation, law) = found.ok_or_else(|| format!("layer {l}: no activation node"))?;
        let up = match operator_named(flat, &format!("{name}.up")) {
            Some(_) => Some(Map::of(flat, &format!("{name}.up"))?.0),
            None => None,
        };
        Ok(Self { input, gate, activation, law, up })
    }
}

/// `M` and the explanation compiled for the experiments, with the explanation's MLP nodes.
struct Scorer {
    experiments: Interchange,
    mlps: Vec<Mlp>,
    /// Each trainable operator's position in `Explanation::trainable`.
    position: BTreeMap<usize, usize>,
}

impl Scorer {
    fn new(device: &Device, native: &OperatorProgram, explanation: &Explanation, settings: &Settings) -> Result<Self, String> {
        let sites: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::library_reads(&explanation.artifact.program, sites.len())?;
        let experiments =
            Interchange::new(device, native, &sites, &explanation.artifact, &explanation.trainable, reads, settings.numeric_bytes, settings.head_tile_rows)?;
        let (flat, _, _) = interchange::sites(&explanation.artifact, &sites)?;
        let mlps = (0..sites.len()).map(|l| Mlp::of(&flat, l)).collect::<Result<_, _>>()?;
        let position = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        Ok(Self { experiments, mlps, position })
    }

    fn layers(&self) -> usize {
        self.mlps.len()
    }

    fn at(&self, operator: usize) -> Result<usize, String> {
        self.position.get(&operator).copied().ok_or_else(|| format!("operator {operator} is not trainable"))
    }

    /// The explanation's read variables that hold an entry of an active group of `posterior`.
    fn variables(&self, posterior: &Posterior) -> Result<Vec<ReadVariable>, String> {
        let mut out = Vec::new();
        for v in self.experiments.variables() {
            let mut held = false;
            for (op, rows) in &v.parts {
                held |= posterior.holds(self.at(*op)?, rows.clone());
            }
            if held {
                out.push(v.clone());
            }
        }
        Ok(out)
    }

    /// `KL(M_e ‖ P_e)` per base token in bits for `experiments` on `batch` over `variables`, with
    /// the patch directions at `mean` and the explanation at `theta` (at `mean` when none), and
    /// with `gradient` the gradient of its sum in every trainable operator.
    fn score(
        &mut self,
        batch: &Batch,
        experiments: &[Experiment],
        variables: &[ReadVariable],
        mean: &[Array2<f64>],
        theta: Option<&[Array2<f64>]>,
        gradient: bool,
    ) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), String> {
        let design = self.experiments.design_at(variables, experiments, mean)?;
        self.experiments.load(theta.unwrap_or(mean))?;
        let (m, p) = self.experiments.models();
        let head = self.experiments.head();
        let teacher = interchange::Teacher::new(&m, head, batch, variables, experiments)?;
        let evaluation = interchange::evaluate(&m, &p, head, batch, &teacher, experiments, &design, gradient)?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        if !gradient {
            return Ok((evaluation.bits, Vec::new()));
        }
        let d = p.program.device();
        let mut gradients = Vec::with_capacity(mean.len());
        for (op, values) in self.position.iter().map(|(op, i)| (*op, &mean[*i])) {
            gradients.push(match evaluation.gradient.get(&op) {
                Some(g) => d.download(g).map_err(error)?,
                None => Array2::zeros(values.dim()),
            });
        }
        if gradients.iter().any(|g| g.iter().any(|v| !v.is_finite())) {
            return Err("nonfinite parameter gradient".into());
        }
        Ok((evaluation.bits, gradients))
    }
}

impl Scorer {
    /// [`Scorer::score`] with the posterior on the device: the patch directions at its mean `μ`,
    /// the explanation at its weight sample of `key` (at `μ` when none), and with `gradient` the
    /// gradient of the sum per trainable operator, left on the device.
    fn score_device(
        &mut self,
        posterior: &DevicePosterior,
        batch: &Batch,
        experiments: &[Experiment],
        variables: &[ReadVariable],
        key: Option<u64>,
        gradient: bool,
    ) -> Result<(Vec<Vec<f64>>, BTreeMap<usize, Tensor>), String> {
        posterior.mean_into(self.experiments.explanation_mut())?;
        let design = interchange::design(&self.experiments.models().1, variables, experiments)?;
        if let Some(key) = key {
            posterior.sample_into(self.experiments.explanation_mut(), key)?;
        }
        let (m, p) = self.experiments.models();
        let head = self.experiments.head();
        let teacher = interchange::Teacher::new(&m, head, batch, variables, experiments)?;
        let evaluation = interchange::evaluate(&m, &p, head, batch, &teacher, experiments, &design, gradient)?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        Ok((evaluation.bits, evaluation.gradient))
    }
}

/// The noise seed of step `(epoch, batch)`, or of the removal comparisons' and the held-out
/// evaluation's batch (epoch 0).
fn noise_seed(seed: u64, epoch: usize, batch: usize) -> u64 {
    seed.wrapping_add((epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)).wrapping_add((batch as u64).wrapping_mul(0x8CB9_2BA7_2F3D_8DD7))
}

/// A running mean of bits over scored tokens.
#[derive(Clone, Copy, Debug, Default)]
struct Mean {
    bits: f64,
    tokens: usize,
}

impl Mean {
    fn add(&mut self, bits: &[f64]) {
        self.bits += bits.iter().sum::<f64>();
        self.tokens += bits.len();
    }

    fn mean(&self) -> Option<f64> {
        (self.tokens > 0).then(|| self.bits / self.tokens as f64)
    }
}

/// The held-out evaluation of `posterior` (held on the host, and as `device_posterior` on the
/// device) on `sequences` (module note); `tokens` is `N`.
fn held_out(
    scorer: &mut Scorer,
    explanation: &Explanation,
    posterior: &Posterior,
    device_posterior: &DevicePosterior,
    sequences: &[Vec<u32>],
    settings: &Settings,
    tokens: usize,
) -> Result<HeldOut, String> {
    let blocks = 2 * scorer.layers();
    let variables = scorer.variables(posterior)?;
    let (mut clean, mut patched) = (vec![Mean::default(); blocks], vec![Mean::default(); blocks]);
    let (mut read, mut complement, mut sampled) = (Mean::default(), Mean::default(), Mean::default());
    let size = |e: &Experiment| e.explained.iter().filter(|x| **x).count();
    for (b, draw) in draws(sequences.len(), settings.batch_sequences, settings.seed)?.iter().enumerate() {
        let batch = draw.batch(sequences)?;
        // Every hybrid size, each with one clean and one patched experiment per base.
        let mut rng = StdRng::seed_from_u64(draw.seed);
        let mut experiments = Vec::with_capacity(2 * blocks * draw.bases.len());
        for k in 1..=blocks {
            let drawn = interchange::sample(&mut rng, draw.bases.len(), blocks / 2, variables.len());
            experiments.extend(drawn.into_iter().map(|e| Experiment { explained: interchange::hybrid_of(&mut rng, blocks, k), ..e }));
        }
        let (bits, _) = scorer.score_device(device_posterior, &batch, &experiments, &variables, None, false)?;
        for (e, bits) in experiments.iter().zip(&bits) {
            match e.patch {
                None => clean[size(e) - 1].add(bits),
                Some(patch) => {
                    patched[size(e) - 1].add(bits);
                    match patch {
                        Patch::Read { .. } => read.add(bits),
                        Patch::Complement { .. } => complement.add(bits),
                    }
                }
            }
        }
        let (bits, _) = scorer.score_device(device_posterior, &batch, &experiments, &variables, Some(noise_seed(settings.seed, 0, b)), false)?;
        bits.iter().for_each(|b| sampled.add(b));
    }
    let data = sampled.mean().ok_or("no held-out tokens")?;
    let divergence: f64 = posterior.divergences().iter().sum();
    let description = posterior.description();
    Ok(HeldOut {
        objective_bits_per_token: data + description / LN_2 / tokens as f64,
        data_bits_per_token: data,
        divergence_bits: divergence / LN_2,
        variance_bits: (description - divergence) / LN_2,
        clean: clean.iter().map(Mean::mean).collect(),
        patched: patched.iter().map(Mean::mean).collect(),
        read_patch: read.mean(),
        complement_patch: complement.mean(),
        layers: activity(scorer, explanation, posterior, sequences, settings)?,
    })
}

/// Per layer, the survivors of `posterior` and its functions' activity on `sequences` at the
/// posterior mean (`LayerCount`).
fn activity(scorer: &mut Scorer, explanation: &Explanation, posterior: &Posterior, sequences: &[Vec<u32>], settings: &Settings) -> Result<Vec<LayerCount>, String> {
    scorer.experiments.load(&posterior.mean)?;
    let variance = |op: usize| -> Result<Array2<f64>, String> { Ok(posterior.log_sd[scorer.at(op)?].mapv(|s| (2.0 * s).exp())) };
    let (_, p) = scorer.experiments.models();
    let (program, d) = (p.program, p.program.device());
    let active = |g: &usize| posterior.active[*g];
    let mut out = Vec::with_capacity(scorer.layers());
    // Per map its weights' posterior variances on the device and its bias's (zero without one).
    let variances = |map: &Map| -> Result<(Tensor, Vec<f64>), String> {
        let weights = d.upload(variance(map.operator)?.view()).map_err(error)?;
        let bias = match map.bias {
            Some(b) => variance(b)?.column(0).to_vec(),
            None => vec![0.0; program.widths()[map.node]],
        };
        Ok((weights, bias))
    };
    let mut gates = Vec::with_capacity(scorer.layers());
    for (layer, mlp) in explanation.layers.iter().zip(&scorer.mlps) {
        let surviving: Vec<bool> = layer.functions.iter().map(|groups| groups.iter().all(active)).collect();
        out.push(LayerCount {
            heads: layer.heads.iter().filter(|(_, values)| values.iter().any(active)).count(),
            planes: layer.heads.iter().map(|(planes, _)| planes.iter().filter(|g| active(g)).count()).sum(),
            values: layer.heads.iter().map(|(_, values)| values.iter().filter(|g| active(g)).count()).sum(),
            functions: surviving.iter().filter(|s| **s).count(),
            nonzero_per_token: 0.0,
            resolved_per_token: 0.0,
        });
        gates.push((surviving, variances(&mlp.gate)?, mlp.up.as_ref().map(variances).transpose()?));
    }
    let mut rows = 0usize;
    for chunk in sequences.chunks(settings.batch_sequences) {
        let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let trace = program.forward(&family)?;
        for ((count, mlp), (surviving, gate, up)) in out.iter_mut().zip(&scorer.mlps).zip(&gates) {
            let x = d.download(trace.value(mlp.input)?).map_err(error)?;
            let squares = d.upload(x.mapv(|v| v * v).view()).map_err(error)?;
            // `s²` of each function's value at every token: `Σ_j σ²_j x_j²` plus the bias's `σ²`.
            let noise = |(weights, bias): &(Tensor, Vec<f64>)| -> Result<Array2<f64>, String> {
                let mut s2 = d.zeros(x.nrows(), surviving.len()).map_err(error)?;
                d.gemm(&mut s2, 1.0, &squares, Op::N, weights, Op::T, 0.0, program.arithmetic()).map_err(error)?;
                let mut s2 = d.download(&s2).map_err(error)?;
                s2.rows_mut().into_iter().for_each(|mut row| row.iter_mut().zip(bias).for_each(|(v, b)| *v += b));
                Ok(s2)
            };
            let sz2 = noise(gate)?;
            let z = d.download(trace.value(mlp.gate.node)?).map_err(error)?;
            let a = d.download(trace.value(mlp.activation)?).map_err(error)?;
            let gated = match (&mlp.up, up) {
                (Some(map), Some(variances)) => Some((d.download(trace.value(map.node)?).map_err(error)?, noise(variances)?)),
                _ => None,
            };
            let (nonzero, resolved) = (0..x.nrows())
                .into_par_iter()
                .map(|t| {
                    let (mut nonzero, mut resolved) = (0usize, 0usize);
                    for (i, alive) in surviving.iter().enumerate() {
                        if *alive {
                            let (zi, phi) = (z[[t, i]], a[[t, i]]);
                            let (yi, sy2) = gated.as_ref().map_or((1.0, 0.0), |(y, sy2)| (y[[t, i]], sy2[[t, i]]));
                            let slope = mlp.law.derivative(zi) * yi;
                            nonzero += usize::from(phi * yi != 0.0);
                            resolved += usize::from((phi * yi).abs() > (slope * slope * sz2[[t, i]].max(0.0) + phi * phi * sy2.max(0.0)).sqrt());
                        }
                    }
                    (nonzero, resolved)
                })
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            count.nonzero_per_token += nonzero as f64;
            count.resolved_per_token += resolved as f64;
        }
        rows += family.rows;
    }
    for count in &mut out {
        count.nonzero_per_token /= rows as f64;
        count.resolved_per_token /= rows as f64;
    }
    Ok(out)
}

/// Where a fit stands at the end of an epoch: with the posterior and the optimizer's moments, all a
/// fit needs to continue exactly as if it had not stopped.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Progress {
    settings: Settings,
    tokens: usize,
    shapes: Vec<(usize, usize)>,
    /// The held-out evaluation at the starting point.
    start: Option<HeldOut>,
    /// The next epoch, and the optimizer steps taken.
    epoch: usize,
    step: i32,
    epochs: Vec<Epoch>,
    removals: Vec<Removal>,
    /// The last epoch's per-batch objective estimates, when convergence is being judged.
    previous: Option<Vec<f64>>,
    active: Vec<bool>,
    done: bool,
    seconds: f64,
}

/// The fit's arrays in checkpoint order: per operator `μ`, `ln σ` and Adam's moments (`μ`'s first
/// and second, then `ln σ`'s).
fn arrays<'a>(posterior: &'a Posterior, moments: &'a [[Array2<f64>; 4]]) -> Vec<&'a Array2<f64>> {
    (0..posterior.mean.len()).flat_map(|i| [&posterior.mean[i], &posterior.log_sd[i], &moments[i][0], &moments[i][1], &moments[i][2], &moments[i][3]]).collect()
}

/// Write the checkpoint atomically: the progress as JSON after its length, then every array's
/// values as little-endian float64. The progress alone also goes to the path with extension
/// `json`, readable while the fit runs.
fn save_checkpoint(path: &Path, progress: &Progress, posterior: &Posterior, moments: &[[Array2<f64>; 4]]) -> Result<(), String> {
    let header = serde_json::to_vec(progress).map_err(error)?;
    let partial = path.with_extension("partial");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(error)?);
    file.write_all(&(header.len() as u64).to_le_bytes()).map_err(error)?;
    file.write_all(&header).map_err(error)?;
    for array in arrays(posterior, moments) {
        for value in array.iter() {
            file.write_all(&value.to_le_bytes()).map_err(error)?;
        }
    }
    file.into_inner().map_err(error)?.sync_all().map_err(error)?;
    std::fs::rename(&partial, path).map_err(error)?;
    let partial = path.with_extension("json.partial");
    std::fs::write(&partial, serde_json::to_vec_pretty(progress).map_err(error)?).map_err(error)?;
    std::fs::rename(&partial, path.with_extension("json")).map_err(error)
}

/// Restore a checkpoint of this fit into `posterior`, with Adam's moments per operator, or refuse
/// one of another fit.
fn load_checkpoint(path: &Path, expected: &Progress, posterior: &mut Posterior) -> Result<(Progress, Vec<[Array2<f64>; 4]>), String> {
    let bytes = std::fs::read(path).map_err(error)?;
    let length = u64::from_le_bytes(bytes.get(..8).ok_or("a truncated checkpoint")?.try_into().map_err(error)?) as usize;
    let header = bytes.get(8..8 + length).ok_or("a truncated checkpoint")?;
    let progress: Progress = serde_json::from_slice(header).map_err(error)?;
    let same_settings = serde_json::to_value(&progress.settings).map_err(error)? == serde_json::to_value(&expected.settings).map_err(error)?;
    if !same_settings || progress.tokens != expected.tokens || progress.shapes != expected.shapes || progress.active.len() != expected.active.len() {
        return Err(format!("{}: a checkpoint of another fit", path.display()));
    }
    let count: usize = progress.shapes.iter().map(|(r, c)| r * c * 6).sum();
    if bytes.len() - 8 - length != count * 8 {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    let mut values = bytes[8 + length..].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes")));
    let mut moments = Vec::with_capacity(posterior.mean.len());
    for i in 0..posterior.mean.len() {
        let dim = posterior.mean[i].dim();
        let mut next = || Array2::from_shape_simple_fn(dim, || values.next().expect("counted values"));
        posterior.mean[i] = next();
        posterior.log_sd[i] = next();
        moments.push([next(), next(), next(), next()]);
    }
    posterior.active = progress.active.clone();
    Ok((progress, moments))
}

/// Fit the library explanation of `native` to its interchange experiments on the training
/// `sequences`, evaluating on the `held_out` sequences after every epoch (module note). `native` is
/// the split native program the explanation was built from. With `checkpoint`, the fit is saved
/// there after every epoch and removal step, and resumed from it when it exists.
pub fn fit(
    device: &Device,
    native: &OperatorProgram,
    explanation: &Explanation,
    sequences: &[Vec<u32>],
    held: &[Vec<u32>],
    settings: &Settings,
    checkpoint: Option<&Path>,
) -> Result<Fit, String> {
    settings.validate()?;
    let length = sequences.first().map_or(0, Vec::len);
    if length == 0 || held.len() < 2 || sequences.iter().chain(held).any(|s| s.len() != length) {
        return Err("training and held-out sequences (at least two) must be nonempty and of one length".into());
    }
    let started = Instant::now();
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    if draws.len() < 2 {
        return Err("the convergence test needs at least two training batches".into());
    }
    // Two experiments per base, each scoring every token of its base.
    let tokens = 2 * sequences.len() * length;
    let mut posterior = Posterior::new(explanation, tokens)?;
    let parameters = posterior.mean.iter().map(Array2::len).sum();
    let mut progress = Progress {
        settings: settings.clone(),
        tokens,
        shapes: posterior.mean.iter().map(Array2::dim).collect(),
        start: None,
        epoch: 0,
        step: 0,
        epochs: Vec::new(),
        removals: Vec::new(),
        previous: None,
        active: posterior.active.clone(),
        done: false,
        seconds: 0.0,
    };
    let mut resumed = None;
    if let Some(path) = checkpoint.filter(|p| p.exists()) {
        let (loaded, moments) = load_checkpoint(path, &progress, &mut posterior)?;
        progress = loaded;
        resumed = Some(moments);
        log::info!("library fit resumed at epoch {} from {}", progress.epoch, path.display());
    }
    let resumed_seconds = progress.seconds;
    let mut scorer = Scorer::new(device, native, explanation, settings)?;
    let steps = |progress: &Progress| u64::try_from(progress.step).map_err(error);
    let mut device_posterior = DevicePosterior::new(device, explanation, &posterior, resumed.as_deref(), steps(&progress)?)?;
    drop(resumed);
    let adam = Adam { mean_rate: settings.mean_step, log_sd_rate: settings.log_sd_step, beta1: settings.beta1, beta2: settings.beta2, epsilon: settings.epsilon };
    // Each group's size, whose `½ ln |G|` an active group's variance costs.
    let sizes: Vec<f64> = explanation.groups.iter().map(|g| g.cells.iter().map(|c| (c.rows.len() * c.cols.len()) as f64).sum()).collect();
    // After every save, the posterior-mean artifact goes next to the checkpoint (extension
    // `artifact.bin`), so the current explanation can be read and scored while the fit runs.
    let save = |progress: &mut Progress, posterior: &Posterior, moments: &[[Array2<f64>; 4]]| -> Result<(), String> {
        progress.active = posterior.active.clone();
        progress.seconds = resumed_seconds + started.elapsed().as_secs_f64();
        let Some(path) = checkpoint else { return Ok(()) };
        save_checkpoint(path, progress, posterior, moments)?;
        let partial = path.with_extension("artifact.partial");
        std::fs::write(&partial, posterior_mean(explanation, posterior)?.f32_literals()?.to_bytes()?).map_err(error)?;
        std::fs::rename(&partial, path.with_extension("artifact.bin")).map_err(error)
    };
    if progress.start.is_none() {
        let start = held_out(&mut scorer, explanation, &posterior, &device_posterior, held, settings, tokens)?;
        log::info!("library start: {start:?}");
        progress.start = Some(start);
        let moments = device_posterior.download(&mut posterior)?;
        save(&mut progress, &posterior, &moments)?;
    }
    let mut variables = scorer.variables(&posterior)?;
    let mut moments: Vec<[Array2<f64>; 4]>;
    while !progress.done {
        let epoch = progress.epoch;
        let epoch_started = Instant::now();
        let mut estimates = Vec::with_capacity(draws.len());
        let (mut data_sum, mut description_sum) = (0.0, 0.0);
        let (mut clean, mut patched) = (Mean::default(), Mean::default());
        for (b, draw) in draws.iter().enumerate() {
            let step_started = Instant::now();
            let batch = draw.batch(sequences)?;
            let experiments = draw.experiments(scorer.layers(), variables.len());
            let key = noise_seed(settings.seed, epoch + 1, b);
            let (bits, gradients) = scorer.score_device(&device_posterior, &batch, &experiments, &variables, Some(key), true)?;
            for (e, bits) in experiments.iter().zip(&bits) {
                if e.patch.is_some() { patched.add(bits) } else { clean.add(bits) }
            }
            let scored = bits.iter().map(Vec::len).sum::<usize>();
            let scale = tokens as f64 / scored as f64;
            let data = scale * LN_2 * bits.iter().flatten().sum::<f64>();
            // `Σ_G KL_G` and the active groups' variances at the posterior the sample was drawn from.
            let description: f64 = device_posterior
                .divergences()?
                .iter()
                .zip(&sizes)
                .zip(&posterior.active)
                .map(|((d, n), active)| if *active { d + 0.5 * n.ln() } else { 0.0 })
                .sum();
            if !description.is_finite() {
                return Err("a nonfinite posterior divergence".into());
            }
            estimates.push(data + description);
            data_sum += data;
            description_sum += description;
            progress.step += 1;
            device_posterior.step(&gradients, scale * LN_2, &adam, key)?;
            log::info!(
                "library step {epoch}.{b}: {:.6} bits per scored token, F estimate {:.6e} bits, {:.2} s",
                bits.iter().flatten().sum::<f64>() / scored as f64,
                (data + description) / LN_2,
                step_started.elapsed().as_secs_f64()
            );
        }
        let count = draws.len() as f64;
        let (improvement, standard_error) = match &progress.previous {
            Some(before) => {
                let differences: Vec<f64> = before.iter().zip(&estimates).map(|(a, b)| a - b).collect();
                let mean = differences.iter().sum::<f64>() / count;
                let variance = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (count - 1.0);
                (Some(mean), Some((variance / count).sqrt()))
            }
            None => (None, None),
        };
        let to_bits = |nats: f64| nats / LN_2;
        let record = Epoch {
            epoch,
            objective_bits: to_bits(estimates.iter().sum::<f64>() / count),
            data_bits: to_bits(data_sum / count),
            description_bits: to_bits(description_sum / count),
            improvement_bits: improvement.map(to_bits),
            standard_error_bits: standard_error.map(to_bits),
            active_groups: posterior.active.iter().filter(|a| **a).count(),
            clean_bits_per_token: clean.mean().unwrap_or(f64::NAN),
            patched_bits_per_token: patched.mean().unwrap_or(f64::NAN),
            seconds: epoch_started.elapsed().as_secs_f64(),
            held_out: {
                moments = device_posterior.download(&mut posterior)?;
                held_out(&mut scorer, explanation, &posterior, &device_posterior, held, settings, tokens)?
            },
        };
        log::info!("library fit epoch {epoch}: {record:?}");
        progress.epochs.push(record);
        progress.previous = Some(estimates);
        progress.epoch += 1;
        let converged = matches!((improvement, standard_error), (Some(i), Some(se)) if i <= se);
        if converged {
            let removal = remove(&mut scorer, &mut posterior, &draws, sequences, settings)?;
            log::info!("library removal after epoch {epoch}: {} of {} candidates", removal.removed, removal.candidates);
            // The removed groups' entries are exactly zero with `ln σ = −∞`, which the device step
            // leaves alone.
            device_posterior = DevicePosterior::new(device, explanation, &posterior, Some(&moments), steps(&progress)?)?;
            progress.done = removal.removed == 0;
            progress.removals.push(removal);
            variables = scorer.variables(&posterior)?;
            // The objective changed discretely: convergence is judged afresh.
            progress.previous = None;
        }
        save(&mut progress, &posterior, &moments)?;
    }
    let objective_bits = progress.removals.last().map_or(f64::NAN, |r| r.after_bits);
    Ok(Fit {
        report: Report {
            settings: settings.clone(),
            training_sequences: sequences.len(),
            scored_tokens: tokens,
            groups: explanation.groups.len(),
            parameters,
            start: progress.start.ok_or("no starting evaluation")?,
            active_groups: posterior.active.iter().filter(|a| **a).count(),
            objective_bits,
            seconds: resumed_seconds + started.elapsed().as_secs_f64(),
            epochs: progress.epochs,
            removals: progress.removals,
        },
        posterior,
    })
}

/// `E_q[D]` over every training batch in nats, one weight sample per batch from the removal seeds,
/// with the groups `removed` (and the already removed ones) zeroed. Experiment identities and
/// patch directions come from `posterior`, before removal, so every trial scores the same evidence.
fn expected_divergence(scorer: &mut Scorer, posterior: &Posterior, draws: &[Draw], sequences: &[Vec<u32>], removed: &[usize], settings: &Settings) -> Result<f64, String> {
    let variables = scorer.variables(posterior)?;
    let mut trial = posterior.clone();
    trial.remove(removed);
    let mut bits = 0.0;
    for (b, draw) in draws.iter().enumerate() {
        // Removal zeroes entries, so the remaining entries see the same noise as the full posterior.
        let (theta, _) = trial.sample(noise_seed(settings.seed, 0, b));
        let experiments = draw.experiments(scorer.layers(), variables.len());
        let (scored, _) = scorer.score(&draw.batch(sequences)?, &experiments, &variables, &posterior.mean, Some(&theta), false)?;
        bits += scored.iter().flatten().sum::<f64>();
    }
    Ok(bits * LN_2)
}

/// The longest prefix among those bisection evaluates, out of `candidates`, whose `change` (the
/// objective's change on removing it) is not positive, or zero. Prefix effects can cancel, so a
/// longer acceptable prefix may go unevaluated; an increase is never accepted.
fn largest_accepted_prefix(
    candidates: usize,
    change: &mut impl FnMut(usize) -> Result<f64, String>,
) -> Result<usize, String> {
    let mut accepted = |k: usize| -> Result<bool, String> {
        let difference = change(k)?;
        if !difference.is_finite() {
            return Err("nonfinite group removal objective".into());
        }
        Ok(difference <= 0.)
    };
    if candidates == 0 {
        return Ok(0);
    }
    if accepted(candidates)? {
        return Ok(candidates);
    }
    let (mut low, mut high) = (0, candidates);
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if accepted(middle)? {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok(low)
}

/// The removal step (module note): a prefix of the active groups in increasing divergence whose
/// removal does not increase the sampled objective on this round's fixed evidence, found by bisection.
fn remove(scorer: &mut Scorer, posterior: &mut Posterior, draws: &[Draw], sequences: &[Vec<u32>], settings: &Settings) -> Result<Removal, String> {
    let divergences = posterior.divergences();
    let costs = posterior.costs();
    let mut order: Vec<usize> = (0..divergences.len()).filter(|g| posterior.active[*g]).collect();
    order.sort_by(|a, b| divergences[*a].total_cmp(&divergences[*b]));
    let description = posterior.description();
    let base = expected_divergence(scorer, posterior, draws, sequences, &[], settings)? + description;
    // `F` with the first `k` candidates removed, minus `F` as it is.
    let mut evaluations: Vec<(usize, f64)> = Vec::new();
    let mut change = |k: usize| -> Result<f64, String> {
        if let Some((_, c)) = evaluations.iter().find(|(at, _)| *at == k) {
            return Ok(*c);
        }
        let saved: f64 = order[..k].iter().map(|g| costs[*g]).sum();
        let c = expected_divergence(scorer, posterior, draws, sequences, &order[..k], settings)? + description - saved - base;
        log::info!("library removal of {k} of {} groups: F changes by {:.6e} bits", order.len(), c / LN_2);
        evaluations.push((k, c));
        Ok(c)
    };
    let low = largest_accepted_prefix(order.len(), &mut change)?;
    let after = base + if low > 0 { change(low)? } else { 0.0 };
    posterior.remove(&order[..low]);
    let to_bits = |nats: f64| nats / LN_2;
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
/// removed groups touch are absent (no literals). Partial interface blocks remain
/// present, so their zeroed entries still cost ordinary serialized literals.
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

#[cfg(test)]
mod tests {
    use super::*;
    use gam_gpu::tensor::posterior_normal;
    use crate::{
        import::import_language_model,
        run_check::{layer_nodes, split_sites},
    };

    /// Adam's moments for one parameter array: the host reference of the device step
    /// (`Device::posterior_adam`).
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

    /// The derivatives in `μ` and in `ln σ` of `data(θ) + Σ_G KL_G`, per trainable operator, from
    /// `data`'s gradient at the sample `θ = μ + σ ⊙ noise` (zero in removed groups); the host
    /// reference of the device step.
    fn derivatives(posterior: &Posterior, data: &[Array2<f64>], noise: &[Array2<f64>]) -> Vec<(Array2<f64>, Array2<f64>)> {
        let variance: Vec<f64> = posterior.moments().iter().map(|m| if m.count > 0.0 { m.second / m.count } else { 0.0 }).collect();
        (0..data.len())
            .into_par_iter()
            .map(|i| {
                let (mut mean, mut log_sd) = (data[i].clone(), Array2::zeros(data[i].dim()));
                ndarray::Zip::from(&mut mean).and(&mut log_sd).and(&noise[i]).and(&posterior.mean[i]).and(&posterior.log_sd[i]).and(&posterior.membership[i]).for_each(
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
        let family = imported.family;
        let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, layers, family, sequences)
    }

    /// The tiny decoder made like Qwen3: SiLU-gated MLPs, an RMS norm with a gain on every head's
    /// query and key, and one key-value head shared by both query heads.
    fn tiny_qwen3(tag: &str) -> (OperatorProgram, Vec<LayerNodes>, FamilyInputs, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let path = dir.join("export.json");
        let mut record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let (d, mlp, head) = (8, 16, 4);
        let mut rng = StdRng::seed_from_u64(11);
        for l in 0..2 {
            for (name, shape, centre) in [
                ("attn.k_proj", [head, d], 0.0),
                ("attn.v_proj", [head, d], 0.0),
                ("mlp.gate_proj", [mlp, d], 0.0),
                ("attn.q_norm.gain", [1, head], 1.0),
                ("attn.k_norm.gain", [1, head], 1.0),
            ] {
                let name = format!("blocks.{l}.{name}");
                let bytes: Vec<u8> = (0..shape[0] * shape[1]).flat_map(|_| (centre + rng.random::<f64>() - 0.5).to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{name}.f64")), bytes).unwrap();
                record["files"][name] = serde_json::json!({"shape": shape});
            }
        }
        let config = &mut record["config"];
        config["n_kv_heads"] = 1.into();
        config["mlp_act"] = "silu".into();
        config["mlp_gated"] = true.into();
        config["qk_norm"] = true.into();
        std::fs::write(&path, record.to_string()).unwrap();
        let imported = import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let family = imported.family;
        let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, layers, family, sequences)
    }

    #[test]
    fn the_starting_library_of_a_qwen3_decoder_is_the_model_in_every_experiment() {
        let (native, layers, family, sequences) = tiny_qwen3("library_start_qwen3");
        let explanation = explanation(&native, &layers).unwrap();
        explanation.artifact.validate_coverage(&native).unwrap();
        assert_eq!(explanation.artifact.blocks.len(), 2 * (2 + 1), "two heads and one MLP per layer");
        // Per head two rotary planes and four value rows; per MLP function its gate, up direction
        // and output.
        assert_eq!(explanation.groups.len(), 2 * (2 * (2 + 4) + 16 * 3));
        assert!(explanation.layers.iter().all(|l| l.functions.iter().all(|f| f.len() == 3)));
        let expected = native.execute(&family, false).unwrap().values[native.output].clone();
        let actual = explanation.artifact.execute(&family).unwrap().values[explanation.artifact.program.output].clone();
        let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let difference = expected.iter().zip(&actual).fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(difference <= 1e-12 * scale, "the starting library differs from the native model by {difference}");
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let cells: usize = explanation.groups.iter().flat_map(|g| &g.cells).map(|c| c.rows.len() * c.cols.len()).sum();
        assert_eq!(cells, posterior.mean.iter().map(Array2::len).sum::<usize>(), "the groups partition the parameters");
        let settings = settings();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, None, 0).unwrap();
        let evaluation = held_out(&mut scorer, &explanation, &posterior, &device_posterior, &sequences, &settings, 72).unwrap();
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]) {
            let bits = bits.expect("every cut and the read patches are drawn");
            assert!(bits.abs() < 1e-10, "the starting library diverges from the model by {bits} bits per token");
        }
        for (count, layer) in evaluation.layers.iter().zip(&explanation.layers) {
            assert_eq!(count.functions, layer.functions.len());
            assert!(count.resolved_per_token <= count.nonzero_per_token && count.nonzero_per_token <= layer.functions.len() as f64);
        }
    }

    #[test]
    fn a_qwen3_library_fit_never_increases_its_objective() {
        let (native, layers, _, sequences) = tiny_qwen3("library_fit_qwen3");
        let explanation = explanation(&native, &layers).unwrap();
        let (train, held) = sequences.split_at(4);
        let fit = fit(&Device::host(), &native, &explanation, train, held, &settings(), None).unwrap();
        let report = &fit.report;
        assert_eq!(report.removals.last().unwrap().removed, 0, "the fit ends when no removal is accepted");
        assert!(report.removals.iter().all(|r| r.after_bits <= r.before_bits), "a removal never increases the objective");
        assert!(report.epochs.iter().all(|e| e.objective_bits.is_finite() && e.held_out.objective_bits_per_token.is_finite()));
        posterior_mean(&explanation, &fit.posterior).unwrap().validate_coverage(&native).unwrap();
    }

    fn settings() -> Settings {
        Settings {
            batch_sequences: 2,
            mean_step: 0.01,
            log_sd_step: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            seed: 3,
            numeric_bytes: 1 << 26,
            head_tile_rows: 64,
        }
    }

    #[test]
    fn the_starting_library_preserves_native_activation_and_coefficients() {
        for law in ["relu", "gelu", "gelu_tanh", "silu"] {
            let (native, layers, family, _) = tiny(&format!("library_start_{law}"), law);
            let explanation = explanation(&native, &layers).unwrap();
            explanation.artifact.validate_coverage(&native).unwrap();
            assert_eq!(explanation.artifact.blocks.len(), 2 * (2 + 1), "two heads and one MLP per layer");
            let expected = native.execute(&family, false).unwrap().values[native.output].clone();
            let actual = explanation.artifact.execute(&family).unwrap().values[explanation.artifact.program.output].clone();
            let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let difference = expected.iter().zip(&actual).fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
            assert!(difference <= 1e-12 * scale, "the starting library differs from the native model by {difference}");
            let posterior = Posterior::new(&explanation, 72).unwrap();
            let cells: usize = explanation.groups.iter().flat_map(|g| &g.cells).map(|c| c.rows.len() * c.cols.len()).sum();
            assert_eq!(cells, posterior.mean.iter().map(Array2::len).sum::<usize>(), "the groups partition the parameters");
            let indexed: usize = explanation
                .layers
                .iter()
                .map(|l| l.heads.iter().map(|(p, v)| p.len() + v.len()).sum::<usize>() + l.functions.iter().map(Vec::len).sum::<usize>())
                .sum();
            assert_eq!(indexed, explanation.groups.len(), "every group belongs to one head or one MLP function");
        }
    }

    #[test]
    fn the_starting_library_matches_the_model_in_every_experiment() {
        let (native, layers, _, sequences) = tiny("library_experiments", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, None, 0).unwrap();
        let evaluation = held_out(&mut scorer, &explanation, &posterior, &device_posterior, &sequences, &settings, 72).unwrap();
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]) {
            let bits = bits.expect("every cut and the read patches are drawn");
            assert!(bits.abs() < 1e-10, "the starting library diverges from the model by {bits} bits per token");
        }
        assert!(evaluation.data_bits_per_token > 0.0, "weight noise costs data");
        for (count, layer) in evaluation.layers.iter().zip(&explanation.layers) {
            assert_eq!(count.functions, layer.functions.len());
            assert_eq!(count.heads, layer.heads.len());
            assert!(count.nonzero_per_token > 0.0 && count.nonzero_per_token < layer.functions.len() as f64, "ReLU functions are exactly zero on some tokens");
            assert!(count.resolved_per_token <= count.nonzero_per_token);
        }
    }

    #[test]
    fn removal_scores_the_reference_experiments_and_patch_directions() {
        let (native, layers, _, sequences) = tiny("library_removal_evidence", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 2 * sequences.len() * 12).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let variables = scorer.variables(&posterior).unwrap();
        let removed: Vec<usize> = (0..posterior.active.len()).collect();
        let mut trial = posterior.clone();
        trial.remove(&removed);
        let trial_variables = scorer.variables(&trial).unwrap();
        assert!(!variables.is_empty() && trial_variables.is_empty(), "removal must change the read-variable population");

        let (mut reference_bits, mut changed_data_bits, mut changed_directions_bits) = (0.0, 0.0, 0.0);
        let mut read_patches = 0;
        for (b, draw) in draws.iter().enumerate() {
            let batch = draw.batch(&sequences).unwrap();
            let experiments = draw.experiments(scorer.layers(), variables.len());
            read_patches += experiments.iter().filter(|e| matches!(e.patch, Some(Patch::Read { .. }))).count();
            let (theta, _) = trial.sample(noise_seed(settings.seed, 0, b));

            // Build the reference intervention independently of Scorer::score, before loading
            // the zeroed candidate. Both the native target and candidate use this same design.
            scorer.experiments.load(&posterior.mean).unwrap();
            let design = interchange::design(&scorer.experiments.models().1, &variables, &experiments).unwrap();
            scorer.experiments.load(&theta).unwrap();
            let (m, p) = scorer.experiments.models();
            let head = scorer.experiments.head();
            let teacher = interchange::Teacher::new(&m, head, &batch, &variables, &experiments).unwrap();
            let reference = interchange::evaluate(&m, &p, head, &batch, &teacher, &experiments, &design, false).unwrap();
            reference_bits += reference.bits.iter().flatten().sum::<f64>();

            // The former implementation redrew experiments and directions from the trial.
            let redrawn = draw.experiments(scorer.layers(), trial_variables.len());
            let (bits, _) = scorer.score(&batch, &redrawn, &trial_variables, &trial.mean, Some(&theta), false).unwrap();
            changed_data_bits += bits.iter().flatten().sum::<f64>();
            // Freezing identities alone is insufficient: zeroed reads change the patch basis.
            let (bits, _) = scorer.score(&batch, &experiments, &variables, &trial.mean, Some(&theta), false).unwrap();
            changed_directions_bits += bits.iter().flatten().sum::<f64>();
        }
        assert!(read_patches > 0, "the reference evidence must include a removed read variable");
        let actual = expected_divergence(&mut scorer, &posterior, &draws, &sequences, &removed, &settings).unwrap();
        let reference = reference_bits * LN_2;
        assert!((actual - reference).abs() < 1e-10 * reference.abs().max(1.0), "removal must score the reference evidence");
        assert!((reference_bits - changed_data_bits).abs() > 1e-8, "redrawing the evidence must expose the former comparison error");
        assert!((reference_bits - changed_directions_bits).abs() > 1e-8, "recomputing patch directions must change the evidence even with fixed identities");
    }

    #[test]
    fn removal_search_never_accepts_an_increase_and_stays_logarithmic() {
        // Three groups contribute +a,-a,b to one logit and the teacher predicts their sum b.
        // Removing the first pair preserves the prediction; removing the first one or all three
        // costs more data than the description saves. The objective is not monotone in the
        // prefix length, so bisection may stop short of the acceptable pair, but it never
        // accepts a prefix that increases the objective.
        let contributions = [10_f64, -10., 5.];
        let teacher_logit = contributions.iter().sum::<f64>();
        let softplus = |z: f64| z.max(0.) + (-z.abs()).exp().ln_1p();
        let probability = 1. / (1. + (-teacher_logit).exp());
        let loss = |logit: f64| 100. * (softplus(logit) - softplus(teacher_logit) - probability * (logit - teacher_logit));
        let change = |k: usize| loss(contributions[k..].iter().sum()) - k as f64;
        assert!(change(1) > 0. && change(2) < 0. && change(3) > 0.);
        let accepted = largest_accepted_prefix(3, &mut |k| Ok(change(k))).expect("finite prefix objectives");
        assert!(accepted == 0 || change(accepted) <= 0.);
        // A monotone boundary is found exactly, in logarithmically many evaluations.
        for boundary in [0, 1, 517, 999, 1000] {
            let mut tested = 0;
            let found = largest_accepted_prefix(1000, &mut |k| {
                tested += 1;
                Ok(if k <= boundary { -1. } else { 1. })
            })
            .unwrap();
            assert_eq!(found, boundary);
            assert!(tested <= 11, "{tested} evaluations for 1000 candidates");
        }
        assert!(largest_accepted_prefix(1, &mut |_| Ok(f64::NAN)).is_err());
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
        let derivatives = derivatives(&posterior, &zeros, &zeros);
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
    fn a_device_step_is_the_host_step() {
        let (native, layers, _, _) = tiny("library_device_step", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let (device, settings) = (Device::host(), settings());
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 0.6 * (rng.random::<f64>() - 0.5));
        }
        let mut device_posterior = DevicePosterior::new(&device, &explanation, &posterior, None, 0).unwrap();
        // A data gradient per operator, weighted by `scale`, at the sample whose noise the device
        // regenerates from `key` (stream `i` for operator `i`).
        let (key, scale) = (99, 3.0);
        let gradients: Vec<Array2<f64>> = posterior.mean.iter().map(|m| m.mapv(|_| rng.random::<f64>() - 0.5)).collect();
        let noise: Vec<Array2<f64>> = posterior
            .mean
            .iter()
            .enumerate()
            .map(|(i, m)| Array2::from_shape_fn(m.dim(), |(r, c)| f64::from(posterior_normal(key, i as u64, (r * m.ncols() + c) as u64))))
            .collect();
        let uploaded: BTreeMap<usize, Tensor> = explanation.trainable.iter().zip(&gradients).map(|(op, g)| (*op, device.upload(g.view()).unwrap())).collect();
        let adam = Adam { mean_rate: settings.mean_step, log_sd_rate: settings.log_sd_step, beta1: settings.beta1, beta2: settings.beta2, epsilon: settings.epsilon };
        device_posterior.step(&uploaded, scale, &adam, key).unwrap();
        let mut stepped = posterior.clone();
        device_posterior.download(&mut stepped).unwrap();
        let weighted: Vec<Array2<f64>> = gradients.iter().map(|g| g * scale).collect();
        let derivatives = derivatives(&posterior, &weighted, &noise);
        let mut reference = posterior.clone();
        let active = reference.active.clone();
        for (i, (mean, log_sd)) in derivatives.iter().enumerate() {
            let (dim, membership) = (reference.mean[i].dim(), reference.membership[i].clone());
            Moment::zeros(dim).step(&mut reference.mean[i], mean, &membership, &active, settings.mean_step, &settings, 1);
            Moment::zeros(dim).step(&mut reference.log_sd[i], log_sd, &membership, &active, settings.log_sd_step, &settings, 1);
        }
        for (field, (device_values, host_values)) in [("μ", (&stepped.mean, &reference.mean)), ("ln σ", (&stepped.log_sd, &reference.log_sd))] {
            for (a, b) in device_values.iter().zip(host_values) {
                let gap = a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs() / (1.0 + y.abs())));
                assert!(gap < 1e-12, "the device step's {field} differs from the host step's by {gap}");
            }
        }
    }

    #[test]
    fn a_group_at_its_prior_costs_only_its_variance() {
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
        let size: usize = group.cells.iter().map(|c| c.rows.len() * c.cols.len()).sum();
        assert!((posterior.costs()[0] - 0.5 * (size as f64).ln()).abs() < 1e-12, "its variance costs ½ ln |G|");
    }

    #[test]
    fn the_fit_converges_removes_and_reports_its_posterior_mean() {
        let (native, layers, _, sequences) = tiny("library_fit", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let (train, held) = sequences.split_at(4);
        let checkpoint = std::env::temp_dir().join(format!("library_fit_{}.bin", std::process::id()));
        let fit = fit(&device, &native, &explanation, train, held, &settings, Some(&checkpoint)).unwrap();
        // A finished fit's checkpoint resumes to the same posterior without another step.
        let resumed = super::fit(&device, &native, &explanation, train, held, &settings, Some(&checkpoint)).unwrap();
        std::fs::remove_file(&checkpoint).unwrap();
        std::fs::remove_file(checkpoint.with_extension("json")).unwrap();
        std::fs::remove_file(checkpoint.with_extension("artifact.bin")).unwrap();
        assert_eq!(resumed.posterior.active, fit.posterior.active);
        assert_eq!(resumed.posterior.means(), fit.posterior.means());
        assert_eq!(resumed.report.epochs.len(), fit.report.epochs.len());
        let report = &fit.report;
        assert_eq!(report.scored_tokens, 2 * 4 * 12, "two experiments per training base");
        assert_eq!(report.removals.last().unwrap().removed, 0, "the fit ends when no removal is accepted");
        assert!(report.removals.iter().all(|r| r.after_bits <= r.before_bits), "a removal never increases the objective");
        assert!(report.epochs.iter().all(|e| e.objective_bits.is_finite() && e.held_out.objective_bits_per_token.is_finite()));
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
    }
}
